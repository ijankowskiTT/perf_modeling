"""Human-readable reports of one analyze() result: markdown (tables) or a self-contained HTML page (inline SVG
charts, light and dark theme, no scripts). The numbers come from the result only; nothing is recomputed."""

from __future__ import annotations

import html
import json
import math

ROLE_PARTS = [("useful_busy_frac", "busy"), ("starved_frac", "starved"), ("backpressure_frac", "backpressure"),
              ("transfer_wait_frac", "waiting on NoC")]


def _ns(v):
    return "-" if v is None else f"{v:,.0f} ns"


def _ticks(vmax, n=4):
    """Round tick values 0..>=vmax (steps of 1, 2, 2.5 or 5 x 10^k)."""
    raw = max(vmax, 1e-9) / n
    mag = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    k = 1
    while step * k < vmax:
        k += 1
    return [step * i for i in range(k + 1)]


def _same(a, b):
    return len(a) == len(b) and all(abs(x["time_ns"] / y["time_ns"] - 1) < 1e-6 for x, y in zip(a, b))


def _rows(res):
    """(id, result row, is_baseline) for the baseline and every candidate / tune pick."""
    out = []
    if res.get("baseline"):
        out.append(("baseline", res["baseline"], True))
    for r in res.get("candidates", []):
        out.append((r.get("id") or r.get("label"), r, False))
    base_label = (res.get("baseline") or {}).get("label")
    for i, r in enumerate(res.get("top", [])):
        out.append((f"top{i + 1}" + (" (= baseline)" if r.get("label") == base_label else ""), r, False))
    return out


def _best(res):
    """The alternative with the highest nominal speedup, or None."""
    base_label = (res.get("baseline") or {}).get("label")
    cands = [(r["comparison"]["speedup_vs_baseline"]["nominal"], rid, r) for rid, r, b in _rows(res)
             if not b and r.get("valid") and r.get("comparison") and r.get("label") != base_label]
    return max(cands, key=lambda x: x[0]) if cands else None


def _workload(req):
    from .layout_api import layout_ops

    w = req.get("workload", {})
    p = w.get("placements", {})
    if w.get("op") in layout_ops():
        shape = "x".join(str(d) for d in w.get("shape", []))
        sh = w.get("shard")
        extra = f", {sh.get('strategy')} shards on {sh.get('cores')} cores" if sh else ""
        return f"{w['op']} {w.get('dtype', 'bf16')}, shape {shape}{extra}, interleaved side / input in {p.get('a', 'dram')}"
    return (f"{w.get('op')} {w.get('dtype')}, {w.get('num_tiles'):,} tiles"
            f"{' per cluster' if w.get('tiles_scope') == 'cluster' else ''}, {w.get('clusters', 32)} "
            f"cluster{'' if w.get('clusters', 32) == 1 else 's'}, "
            f"a/b/out in {p.get('a')}/{p.get('b')}/{p.get('out')}")


def key_points(req, res):
    pts = []
    b = res.get("baseline")
    if b and b.get("valid"):
        t = b["time_ns"]
        pts.append(f"Baseline {b['label']}: {_ns(t['nominal'])} ({_ns(t['low'])} .. {_ns(t['high'])}), "
                   f"limited by {b['bottleneck']['resource']}.")
    best = _best(res)
    if best:
        s, rid, r = best
        c = r["comparison"]
        if c["verdict"] == "faster":
            pts.append(f"Best alternative {rid} ({r['label']}): {s:.2f}x faster in every cost scenario.")
        else:
            pts.append(f"No alternative is clearly faster (best: {rid}, {s:.2f}x, {c['verdict']}).")
    sens = res.get("sensitivity")
    if sens:
        base_pts = [p for p in sens["series"].get("baseline", []) if p.get("time_ns")]
        for sid, series in sens["series"].items():
            ok = [p for p in series if p.get("time_ns")]
            if sid != "baseline" and _same(ok, base_pts):
                pts.append(f"{sid}: identical to the baseline at every {sens['param']} value: it changes nothing.")
                continue
            if len(ok) >= 2:
                a, z = ok[0], ok[-1]
                necks = sorted({p["bottleneck"] for p in ok})
                pts.append(f"{sid}: {sens['param']} {a['value']} -> {z['value']} moves time {_ns(a['time_ns'])} -> "
                           f"{_ns(z['time_ns'])} ({z['time_ns'] / a['time_ns'] - 1:+.0%}); bottleneck "
                           f"{' / '.join(necks)}.")
        for x in sens.get("ranking_changes", []):
            pts.append(f"Ranking changes between {x['between']} at about {x['at_about']}.")
    if res.get("search"):
        s = res["search"]
        pts.append(f"Search: {s['evaluated']} of {s['valid']} valid configs evaluated"
                   f"{' (exhaustive)' if s['exhaustive'] else ' (not exhaustive)'}.")
    warn = sorted({w for _, r, _ in _rows(res) for w in r.get("warnings", []) if "hypothetical" in w})
    pts += [w[0].upper() + w[1:] + "." for w in warn]
    return pts


# ---------------------------------------------------------------------------------------------- markdown

def to_markdown(req, res):
    prof = res.get("profiles", {}).get("costs") or {"name": req.get("profiles", {}).get("costs"), "revision": "?"}
    L = [f"# qperf report: {req.get('action')}", "", f"**Workload:** {_workload(req)}  ",
         f"**Profile:** {prof['name']}@{prof['revision']}  ", f"**Platform:** {res.get('platform_scope', '-')}", ""]
    if res["status"] not in ("ok", "partial"):
        return "\n".join(L + [f"**{res['status']}:** {res.get('message', '')}", ""])
    L += ["## Key points", ""] + [f"- {p}" for p in key_points(req, res)] + [""]
    L += ["## Configurations", "", "| | config | time | range | speedup | verdict | bottleneck |", "|---|---|---|---|---|---|---|"]
    for rid, r, base in _rows(res):
        if not r.get("valid"):
            L.append(f"| {rid} | {r.get('label', '')} | invalid | | | {'; '.join(r.get('errors', []))} | |")
            continue
        t = r["time_ns"]
        c = r.get("comparison") or {}
        sp = c.get("speedup_vs_baseline", {}).get("nominal")
        L.append(f"| {rid} | {r['label']} | {_ns(t['nominal'])} | {t['low']:,.0f} .. {t['high']:,.0f} | "
                 f"{'' if base else f'{sp:.2f}x'} | {c.get('verdict', '')} | {r['bottleneck']['resource']} |")
    b = res.get("baseline")
    if b and b.get("roles"):
        L += ["", "## Where the baseline's time goes", "", "| role | threads | " + " | ".join(n for _, n in ROLE_PARTS) + " |",
              "|---|---|" + "---|" * len(ROLE_PARTS)]
        for role, v in b["roles"].items():
            L.append(f"| {role} | {v['threads']} | " + " | ".join(f"{v.get(k, 0):.0%}" for k, _ in ROLE_PARTS) + " |")
    sens = res.get("sensitivity")
    if sens:
        vals = [p["value"] for p in next(iter(sens["series"].values()))]
        L += ["", f"## Sensitivity to {sens['param']}", "", "| | " + " | ".join(str(v) for v in vals) + " |",
              "|---|" + "---|" * len(vals)]
        for sid, series in sens["series"].items():
            L.append(f"| {sid} | " + " | ".join(_ns(p.get("time_ns")) for p in series) + " |")
    L += ["", "## Request", "", "```json", json.dumps(req, indent=1), "```", ""]
    return "\n".join(L)


# -------------------------------------------------------------------------------------------------- html

CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--deemph:#c3c2b7;--track:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;
--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--deemph:#52514e;--track:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--deemph:#52514e;--track:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px}.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile .l{color:var(--ink2);font-size:13px}.tile .v{font-size:26px;font-weight:600}.tile .d{color:var(--ink2);font-size:13px}
ul.kp{margin:0;padding-left:20px}ul.kp li{margin:4px 0}
svg{width:100%;height:auto;display:block}.chart{overflow-x:auto}.chart svg{min-width:620px}svg text{fill:var(--ink2);font:12px system-ui,sans-serif}
svg .v{fill:var(--ink)}.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:13px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
.wrap{overflow-x:auto}details{margin-top:12px}pre{white-space:pre-wrap;font-size:12px;color:var(--ink2)}
.note{color:var(--ink2);font-size:13px}
"""

E = html.escape


def _bars(res, best_id):
    """Horizontal time bars (ns) with low..high whiskers; the recommended config in the accent, others gray."""
    rows = [(rid, r) for rid, r, _ in _rows(res) if r.get("valid")]
    if not rows:
        return ""
    W, lab, right, bh, gap = 920, 300, 110, 22, 14
    ticks = _ticks(max(r["time_ns"]["high"] for _, r in rows))
    top = ticks[-1]
    sx = lambda v: lab + (W - lab - right) * v / top  # noqa: E731
    H = len(rows) * (bh + gap) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Predicted time per configuration">']
    for tv in ticks:
        x = sx(tv)
        out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H - 24}" stroke="var(--grid)"/>'
                   f'<text x="{x:.1f}" y="{H - 8}" text-anchor="middle">{tv:,.0f}</text>')
    for i, (rid, r) in enumerate(rows):
        y = i * (bh + gap) + 6
        t = r["time_ns"]
        x1 = sx(t["nominal"])
        fill = "var(--s1)" if rid == best_id or (best_id is None and rid == "baseline") else "var(--deemph)"
        w = max(2.0, x1 - lab)
        r4 = min(4, w / 2)
        tip = f"{rid}: {r['label']} | {t['nominal']:,.0f} ns ({t['low']:,.0f} .. {t['high']:,.0f})"
        out.append(f'<g><title>{E(tip)}</title>'
                   f'<text x="{lab - 10}" y="{y + bh / 2 + 4}" text-anchor="end" class="v">{E(str(rid))}</text>'
                   f'<path d="M{lab} {y} h{w - r4:.1f} q{r4:.1f} 0 {r4:.1f} {r4:.1f} v{bh - 2 * r4:.1f} '
                   f'q0 {r4:.1f} {-r4:.1f} {r4:.1f} h{-(w - r4):.1f} z" fill="{fill}"/>'
                   f'<line x1="{sx(t["low"]):.1f}" x2="{sx(t["high"]):.1f}" y1="{y + bh / 2}" y2="{y + bh / 2}" '
                   f'stroke="var(--ink2)" stroke-width="1.5"/>'
                   f'<text x="{sx(t["high"]) + 8:.1f}" y="{y + bh / 2 + 4}" class="v">{t["nominal"]:,.0f}</text></g>')
    out.append("</svg>")
    return "".join(out)


def _roles(r):
    """Small multiples, one panel per measure (busy / starved / backpressure / waiting on NoC), each bar against
    100% of the run. Not stacked: a compute thread's unpack/math and pack stages overlap, so shares can sum past 100%."""
    roles = r.get("roles") or {}
    if not roles:
        return ""
    W, lab, bh, gap, top = 920, 110, 16, 12, 22
    pw = (W - lab) / len(ROLE_PARTS)
    H = top + len(roles) * (bh + gap)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of the run per role and measure">']
    for i, role in enumerate(roles):
        y = top + i * (bh + gap)
        out.append(f'<text x="{lab - 10}" y="{y + bh / 2 + 4}" text-anchor="end" class="v">{E(role)} x{roles[role]["threads"]}</text>')
    for k, (key, name) in enumerate(ROLE_PARTS):
        x0 = lab + k * pw
        span = pw - 56
        out.append(f'<text x="{x0:.1f}" y="12" class="v">{name}</text>')
        for i, (role, v) in enumerate(roles.items()):
            y = top + i * (bh + gap)
            f = min(1.0, max(0.0, v.get(key, 0.0)))
            w = span * f
            out.append(f'<rect x="{x0:.1f}" y="{y}" width="{span:.1f}" height="{bh}" rx="3" fill="var(--track)"/>')
            if w >= 1:
                r4 = min(3.0, w / 2)
                out.append(f'<path d="M{x0:.1f} {y} h{w - r4:.1f} q{r4:.1f} 0 {r4:.1f} {r4:.1f} v{bh - 2 * r4:.1f} '
                           f'q0 {r4:.1f} {-r4:.1f} {r4:.1f} h{-(w - r4):.1f} z" fill="var(--s{k + 1})">'
                           f'<title>{E(role)}: {name} {f:.0%} of the run</title></path>')
            out.append(f'<text x="{x0 + w + 6:.1f}" y="{y + bh / 2 + 4}" class="v">{f:.0%}</text>')
    out.append("</svg>")
    return "".join(out)


def _xscale(xs, a, b):
    """Linear, or logarithmic when positive values span 20x or more. Returns (map, tick values)."""
    x0, x1 = min(xs), max(xs)
    if x0 > 0 and x1 / x0 >= 20:
        f = lambda v: a + (b - a) * (math.log(v / x0) / math.log(x1 / x0))  # noqa: E731
    else:
        f = lambda v: a + (b - a) * ((v - x0) / (x1 - x0) if x1 > x0 else 0.5)  # noqa: E731
    ticks, last = [], -1e9
    for v in sorted(set(xs)):
        if f(v) - last >= 44:
            ticks.append(v)
            last = f(v)
    return f, ticks


def _linechart(series, param, yval, yfmt, label, ref=None):
    """One line per series over the swept parameter; series = (id, color slot, points), so a series keeps its color
    across charts. End labels when they do not collide."""
    series = [(sid, k, [p for p in pts if yval(sid, p) is not None]) for sid, k, pts in series]
    pts_all = [yval(sid, p) for sid, _, s in series for p in s]
    if not pts_all:
        return ""
    W, H, l, t, b = 920, 280, 70, 12, 36
    r = 20 + 7 * max(len(" = ".join(sid for sid, _, _ in series)), 8)
    sx, xt = _xscale([p["value"] for _, _, s in series for p in s], l, W - r)
    lo = min(pts_all + ([ref] if ref else []))
    hi = max(pts_all + ([ref] if ref else []))
    if ref:  # speedup: a band around 1.0 with round steps, not from zero
        step = next(st for st in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0) if st * 4 >= max(hi - lo, 0.04))
        y0, y1 = math.floor(lo / step) * step, math.ceil(hi / step) * step
        yt = [round(y0 + step * k, 3) for k in range(int(round((y1 - y0) / step)) + 1)]
    else:
        yt = _ticks(hi)
        y0, y1 = 0.0, yt[-1]
    sy = lambda v: t + (H - t - b) * (1 - (v - y0) / (y1 - y0))  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(label)}">']
    for tv in yt:
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(tv):.1f}" y2="{sy(tv):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{l - 8}" y="{sy(tv) + 4:.1f}" text-anchor="end">{yfmt(tv)}</text>')
    if ref:
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" stroke="var(--axis)" stroke-width="1.5"/>'
                   f'<text x="{W - r + 6}" y="{sy(ref) + 4:.1f}">baseline</text>')
    for v in xt:
        out.append(f'<text x="{sx(v):.1f}" y="{H - 12}" text-anchor="middle">{v:,g}</text>')
    for sid, k, s in series:
        col = f"var(--s{k + 1})"
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(p['value']):.1f} {sy(yval(sid, p)):.1f}" for i, p in enumerate(s))
        out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for p in s:
            out.append(f'<circle cx="{sx(p["value"]):.1f}" cy="{sy(yval(sid, p)):.1f}" r="4" fill="{col}" stroke="var(--surface)" '
                       f'stroke-width="2"><title>{E(sid)}: {param}={p["value"]:,g} -> {yfmt(yval(sid, p))} '
                       f'({E(p["bottleneck"])})</title></circle>')
    groups = []
    for sid, k, s in series:
        if not s:
            continue
        for g in groups:
            if all(abs(yval(g[0][0], a) - yval(sid, c)) <= 1e-9 * max(1.0, abs(yval(sid, c))) for a, c in zip(g[1], s)) \
                    and len(g[1]) == len(s):
                g[0].append(sid)
                break
        else:
            groups.append(([sid], s))
    ys = sorted(sy(yval(ids[0], s[-1])) for ids, s in groups)
    if all(b2 - a2 >= 14 for a2, b2 in zip(ys, ys[1:])):
        for ids, s in groups:
            out.append(f'<text x="{sx(s[-1]["value"]) + 10:.1f}" y="{sy(yval(ids[0], s[-1])) + 4:.1f}" class="v">'
                       f'{E(" = ".join(ids))}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:var(--s{k + 1})"></i>{E(sid)}</span>' for sid, k, _ in series)
    return "".join(out) + (f'<div class="legend">{legend}</div>' if len(series) > 1 else "")


def _sensitivity_charts(sens):
    series = [(sid, k, pts) for k, (sid, pts) in enumerate(list(sens["series"].items())[:4])]
    out = _linechart(series, sens["param"], lambda sid, p: p.get("time_ns"), lambda v: f"{v:,.0f}",
                     f"Time vs {sens['param']}")
    base = {p["value"]: p.get("time_ns") for p in sens["series"].get("baseline", [])}
    cands = [x for x in series if x[0] != "baseline"]
    if cands and base:
        sp = lambda sid, p: (base[p["value"]] / p["time_ns"]) if p.get("time_ns") and base.get(p["value"]) else None  # noqa: E731
        out += (f'<h3 class="note" style="margin:16px 0 4px">Speedup vs baseline (above the line = faster)</h3>'
                + _linechart(cands, sens["param"], sp, lambda v: f"{v:.2f}x", "Speedup vs baseline", ref=1.0))
    return out


def to_html(req, res):
    prof = res.get("profiles", {}).get("costs", {})
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>qperf report</title><style>{CSS}</style></head><body><main>'
            f'<h1>qperf: {E(str(req.get("action")))}</h1><p class="sub">{E(_workload(req))} · '
            f'{E(prof.get("name", ""))}@{E(prof.get("revision", ""))}</p>')
    if res["status"] not in ("ok", "partial"):
        return head + f'<div class="card"><b>{E(res["status"])}</b>: {E(res.get("message", ""))}</div></main></body></html>'
    b = res["baseline"]
    best = _best(res)
    tiles = [("Baseline time", _ns(b["time_ns"]["nominal"]), b["label"]),
             ("Bottleneck", b["bottleneck"]["resource"], f"busiest role: {b['bottleneck'].get('busiest_role', '-')}")]
    if best and best[2]["comparison"]["verdict"] == "faster":
        tiles.insert(1, ("Best alternative", f"{best[0]:.2f}x faster", best[1]))
    elif best or res.get("search"):
        tiles.insert(1, ("Best alternative", "none faster",
                         f"best other: {best[1]} {best[0]:.2f}x" if best else "the baseline is the best config"))
    parts = [head, '<div class="tiles">']
    parts += [f'<div class="card tile"><div class="l">{E(l)}</div><div class="v">{E(v)}</div><div class="d">{E(d)}</div></div>'
              for l, v, d in tiles]
    parts.append('</div><div class="card"><h2 style="margin-top:0">Key points</h2><ul class="kp">')
    parts += [f"<li>{E(p)}</li>" for p in key_points(req, res)]
    parts.append("</ul></div>")
    best_id = best[1] if best and best[2]["comparison"]["verdict"] == "faster" else None
    if len(_rows(res)) > 1:
        parts.append('<h2>Predicted time per configuration (ns)</h2><div class="card chart">' + _bars(res, best_id) +
                     '<p class="note">Bar: nominal estimate. Whisker: all costs at their low / high scenario. '
                     'Highlighted: the alternative that is faster in every scenario, otherwise the baseline.</p></div>')
    if res.get("sensitivity"):
        parts.append(f'<h2>Sensitivity to {E(res["sensitivity"]["param"])}</h2><div class="card chart">'
                     + _sensitivity_charts(res["sensitivity"]) + "</div>")
    parts.append(f'<h2>Where the baseline\'s time goes (share of the run)</h2><div class="card chart">{_roles(b)}</div>')
    if best and best[2].get("roles") and best[2]["comparison"]["verdict"] == "faster":
        parts.append(f'<h2>Where {E(best[1])}\'s time goes</h2><div class="card chart">{_roles(best[2])}</div>')
    md = to_markdown(req, res)
    table = md.split("## Configurations", 1)[1].split("\n## ", 1)[0].strip().splitlines()
    head_cells = [c.strip() for c in table[0].strip("|").split("|")]
    body = "".join("<tr>" + "".join(f"<td>{E(c.strip())}</td>" for c in row.strip("|").split("|")) + "</tr>" for row in table[2:])
    parts.append('<h2>All numbers</h2><div class="card wrap"><table><tr>' + "".join(f"<th>{E(c)}</th>" for c in head_cells)
                 + f"</tr>{body}</table></div>")
    parts.append(f'<p class="note">{E(res.get("platform_scope", ""))}</p><details><summary>Request</summary><pre>'
                 f'{E(json.dumps(req, indent=1))}</pre></details></main></body></html>')
    return "".join(parts)
