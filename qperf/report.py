"""Human-readable reports of one analyze() result: markdown (tables) or a self-contained HTML page (inline SVG
charts, light and dark theme, no scripts). The numbers come from the result only; nothing is recomputed."""

from __future__ import annotations

import html
import json
import math

from .accuracy import range_txt as accuracy_range_txt
from .survey import describe

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
    if req.get("action") == "survey":
        sv = req.get("survey") or {}
        grid = (f"{len(sv['heights'])} heights x {len(sv['widths'])} widths" if sv.get("heights") and sv.get("widths")
                else "the default shape grid")
        return f"{w.get('op')} {w.get('dtype', 'bf16')}, survey over {grid}, input in {p.get('a', 'dram')}"
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


# ------------------------------------------------------------------------------------- configs and structure

ROLES = ("reader", "compute", "writer")
MEM = {"dram": "DRAM", "l1": "L1", "borrowed": "L1 shards", "hs": "L1 shards"}
ALIAS = {"R": "reader_threads", "C": "compute_threads", "W": "writer_threads", "cores": "shard_cores",
         "cores_used": "cores", "split": "split"}
MAX_SLOTS = 6  # categorical colors --s1..--s6; further configs are drawn in the de-emphasized gray


def _slots(res):
    """Color slot per config id, in row order: a config keeps its color in every chart."""
    return {rid: k for k, (rid, _, _) in enumerate(_rows(res))}


def _col(k):
    return f"var(--s{k + 1})" if k is not None and k < MAX_SLOTS else "var(--deemph)"


def _speedup(r):
    return ((r.get("comparison") or {}).get("speedup_vs_baseline") or {}).get("nominal")


def _hypothetical(r):
    return bool((r.get("config") or {}).get("hypothetical")) or any(
        str(w).lower().startswith("hypothetical") for w in r.get("warnings", []))


def _shown(res, n=4):
    """Configs drawn in 'How it runs': the baseline plus the fastest alternatives, at most n, in row order."""
    base_label = (res.get("baseline") or {}).get("label")
    rows = [(rid, r) for rid, r, b in _rows(res) if r.get("valid") and (b or r.get("label") != base_label)]
    alts = [x for x in rows if x[0] != "baseline"]
    keep = {rid for rid, _ in sorted(alts, key=lambda x: -(_speedup(x[1]) or 0))[:n - (len(rows) - len(alts))]}
    return [(rid, r) for rid, r in rows if rid == "baseline" or rid in keep]


def _pl(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def _fmtv(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmtv(x)}" for k, x in v.items())
    if isinstance(v, (int, float)):
        return f"{v:,g}"
    return str(v)


def _cores(r):
    ws = r.get("work_split") or []
    n = sum(e.get("cores", 0) for e in ws)
    return n or None


def _changes(req, base, rid, r):
    """Knobs of one config that differ from the baseline: the request's overrides first, then config and structure
    fields the overrides do not already name (e.g. the factory a knob selects, the cores a shard count uses)."""
    if rid == "baseline":
        return {}
    out = dict(next((c.get("overrides") or {} for c in req.get("candidates", []) if c.get("id") == rid), {}))

    def named(k):
        return k in out or ALIAS.get(k) in out or any(ALIAS.get(o) == k for o in out)

    bc, rc = base.get("config") or {}, r.get("config") or {}
    for k, v in rc.items():
        if k != "hypothetical" and v != bc.get(k) and not named(k):
            out[k] = v
    bs, rs = base.get("structure") or {}, r.get("structure") or {}
    for k, name in (("cores_used", "cores"), ("split", "split"), ("source", "source"), ("dest", "dest")):
        if rs.get(k) is not None and rs.get(k) != bs.get(k) and not named(k) and not named(name):
            out[name] = rs[k]
    if not rs and not bs and _cores(r) != _cores(base) and _cores(r) and not named("cores"):
        out["cores"] = _cores(r)
    return out


def _plan(req, r):
    """What 'How it runs' draws for one result row: from r['structure'] when the row has it, otherwise only fields
    the row carries (work_split, config threads / batch / ring depth, per-cluster traffic, roles). A field the row
    does not give is left out, never guessed."""
    s = r.get("structure") or {}
    cfg = r.get("config") or {}
    roles = r.get("roles") or {}
    w = req.get("workload") or {}
    pl = w.get("placements") or {}
    tr = r.get("traffic_per_cluster") or {}
    binary = "R" in cfg
    g = s.get("grid") or [8, 4]
    p = {"unit": "clusters" if binary else "cores", "cols": int(g[0]), "rows": int(g[1]),
         "factory": s.get("factory") or cfg.get("factory"), "split": s.get("split"),
         "scope": "per cluster" if binary and tr else ("busiest core" if s else None)}
    loads = {}
    for e in r.get("work_split") or []:
        if binary:
            n, wt = e.get("clusters"), e.get("tiles_per_cluster") or 0
            txt = _pl(wt, "tile")
        else:
            n, b, t = e.get("cores"), e.get("blocks_per_core") or 0, e.get("tiles_per_block") or 0
            wt, txt = b * t, f"{_pl(b, 'block')} x {_pl(t, 'tile')}"
        if n:
            loads[(wt, txt)] = loads.get((wt, txt), 0) + n
    p["loads"] = sorted(((n, wt, txt) for (wt, txt), n in loads.items()), key=lambda x: -x[1])
    p["used"] = s.get("cores_used") if s.get("cores_used") is not None else (sum(x[0] for x in p["loads"]) or None)
    if p["used"] is None and binary and w.get("clusters"):
        p["used"] = w["clusters"]  # compact responses drop work_split; the request names the clusters
    if p["split"] == "2d" and s.get("row_cores") and s.get("col_chunks"):
        p["split_detail"] = f"{s['row_cores']} row cores x {s['col_chunks']} column chunks"
    pc = s.get("per_core") or {}
    if pc.get("blocks") is not None and pc.get("tiles_per_block") is not None:
        p["per_core"] = (f"{_pl(pc['blocks'], 'block')} x {_pl(pc['tiles_per_block'], 'tile')}"
                         + (f" = {_pl(pc['tiles'], 'tile')}" if pc.get("tiles") else ""))
    elif pc.get("tiles") is not None:
        p["per_core"] = _pl(pc["tiles"], "tile")
    elif len(p["loads"]) == 1:
        p["per_core"] = p["loads"][0][2]
    elif p["loads"]:
        unit1 = p["unit"][:-1]
        p["per_core"] = ", ".join(f"{txt} ({n} {unit1 if n == 1 else p['unit']})" for n, _, txt in p["loads"])

    def mem(v):
        return MEM.get(v, str(v))

    if s.get("source"):
        src = [mem(s["source"])]
    elif binary:
        src = [f"{k}: {mem(pl[k])}" for k in ("a", "b") if pl.get(k)]
    else:
        src = [mem(cfg.get("input") or pl["a"])] if (cfg.get("input") or pl.get("a")) else []
    if s.get("dest"):
        dst = [mem(s["dest"])]
    else:
        dst = [f"out: {mem(pl['out'])}"] if binary and pl.get("out") else []

    b = r.get("bottleneck") or {}
    res_name = str(b.get("resource", ""))
    hot = next((x for x in ROLES if res_name.startswith(x)), None) or b.get("busiest_role")

    def threads(role):
        return (roles.get(role) or {}).get("threads") or cfg.get(role[0].upper())

    def busy(role):
        return (roles.get(role) or {}).get("useful_busy_frac")

    def dm(role, req_k, bytes_k, bar_k):
        sd = s.get(role) or {}
        lines = []
        if sd.get("requests") is not None:
            bpr = sd.get("bytes_per_request")
            lines.append(f"{sd['requests']:,} x {bpr:,} B" if bpr is not None else _pl(sd["requests"], "request"))
        elif tr.get(req_k):
            lines.append(f"{tr[req_k]:,} x {round(tr[bytes_k] / tr[req_k]):,} B" if tr.get(bytes_k) else f"{tr[req_k]:,} requests")
        bars = sd.get("barriers", tr.get(bar_k))
        if bars is not None:
            lines.append(_pl(bars, "barrier"))
        if cfg.get("dm_batch") is not None:
            lines.append(f"dm_batch {cfg['dm_batch']}")
        return lines

    def role_node(role, lines):
        t = threads(role)
        return {"kind": "role", "role": role, "title": role + (f" x{t}" if t else ""), "lines": lines,
                "busy": busy(role), "hot": role == hot}

    sc = s.get("compute") or {}
    has_compute = "compute" in roles or bool(sc) or not roles
    dfb = s.get("dfb") or {}
    nodes = [{"kind": "mem", "title": "source", "lines": src}]
    if "reader" in roles or s.get("reader") or not roles:
        nodes.append(role_node("reader", dm("reader", "noc_read_requests", "noc_read_bytes", "reader_barriers")))
    if has_compute:
        k = cfg.get("entries_per_thread")

        def ring(which):
            n = dfb.get(f"{which}_entries")
            if n is not None:
                eb = dfb.get("entry_bytes")
                return {"kind": "dfb", "title": f"{'input' if which == 'in' else 'output'} DFB", "slots": n,
                        "lines": [f"{n:,} x {eb:,} B" if eb is not None else f"{n:,} entr{'y' if n == 1 else 'ies'}"]}
            return {"kind": "dfb", "title": f"{'input' if which == 'in' else 'output'} DFB", "slots": k,
                    "lines": [f"{k} entr{'y' if k == 1 else 'ies'}", "per thread"] if k is not None else []}

        cl = []
        if sc.get("steps") is not None:
            cl.append(f"{sc['steps']:,} x {_pl(sc['tiles_per_step'], 'tile')}" if sc.get("tiles_per_step") is not None
                      else _pl(sc["steps"], "step"))
        elif tr.get("tiles"):
            cl.append(_pl(tr["tiles"], "tile"))
        if cfg.get("tiles_per_cycle") is not None:
            cl.append(f"tiles_per_cycle {cfg['tiles_per_cycle']}")
        if sc.get("kind"):
            cl.append(str(sc["kind"]))
        elif binary and w.get("op"):
            cl.append(str(w["op"]))
        nodes += [ring("in"), role_node("compute", cl), ring("out")]
    if "writer" in roles or s.get("writer") or not roles:
        nodes.append(role_node("writer", dm("writer", "noc_write_requests", "noc_write_bytes", "writer_barriers")))
    nodes.append({"kind": "mem", "title": "destination", "lines": dst})
    p["nodes"] = nodes
    return p


def _how_text(p):
    """One line per config for the markdown report."""
    total = p["cols"] * p["rows"]
    bits = []
    if p["used"] is not None:
        bits.append(f"{p['used']} of {total} {p['unit']}" + (f", {p['split']} split" if p.get("split") else "")
                    + (f" ({p['split_detail']})" if p.get("split_detail") else ""))
    if p.get("factory"):
        bits.append(f"factory {p['factory']}")
    if p.get("per_core"):
        bits.append(f"per {p['unit'][:-1]} {p['per_core']}")
    for n in p["nodes"]:
        if n["kind"] == "mem":
            if n["lines"]:
                bits.append(f"{n['title']} {', '.join(n['lines'])}")
            continue
        txt = n["title"] + (" " + ", ".join(n["lines"]) if n["lines"] else "")
        if n.get("busy") is not None:
            txt += f" (busy {n['busy']:.0%}{', bottleneck' if n.get('hot') else ''})"
        elif n.get("hot"):
            txt += " (bottleneck)"
        bits.append(txt)
    return "; ".join(bits) + (f" (counts {p['scope']})" if p.get("scope") else "")


# ---------------------------------------------------------------------------------------------- markdown

def _approaches(req, res):
    """(id, changes text, time, speedup, verdict, bottleneck, tags) per config, the baseline first."""
    base = res.get("baseline") or {}
    out = []
    for rid, r, is_base in _rows(res):
        ch = _changes(req, base, rid, r)
        tags = (["hypothetical"] if _hypothetical(r) else []) + ([] if r.get("valid") else ["invalid"])
        sp = _speedup(r)
        out.append((rid, "(baseline)" if is_base else (", ".join(f"{k} {_fmtv(v)}" for k, v in ch.items()) or "-"),
                    _ns((r.get("time_ns") or {}).get("nominal")) if r.get("valid") else "invalid",
                    "" if is_base or sp is None else f"{sp:.2f}x", (r.get("comparison") or {}).get("verdict", ""),
                    (r.get("bottleneck") or {}).get("resource", ""), ", ".join(tags)))
    return out


def to_markdown(req, res):
    prof = res.get("profiles", {}).get("costs") or {"name": req.get("profiles", {}).get("costs"), "revision": "?"}
    L = [f"# qperf report: {req.get('action')}", "", f"**Workload:** {_workload(req)}  ",
         f"**Profile:** {prof['name']}@{prof['revision']}  ", f"**Platform:** {res.get('platform_scope', '-')}"
         + ("  " if res.get("accuracy") else ""), *([f"**Accuracy:** {res['accuracy']}"] if res.get("accuracy") else []),
         ""]
    if res["status"] not in ("ok", "partial"):
        return "\n".join(L + [f"**{res['status']}:** {res.get('message', '')}", ""])
    if res.get("survey"):
        return "\n".join(L + _survey_md(req, res))
    L += ["## Key points", ""] + [f"- {p}" for p in key_points(req, res)] + [""]
    shown = _shown(res)
    if shown:
        L += ["## How it runs", ""] + [f"- **{rid}**: {_how_text(_plan(req, r))}" for rid, r in shown] + [""]
    if len(_rows(res)) > 1:
        L += ["## Approaches", "", "| config | changes vs baseline | time | speedup | verdict | bottleneck | tags |",
              "|---|---|---|---|---|---|---|"]
        L += ["| " + " | ".join(row) + " |" for row in _approaches(req, res)] + [""]
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
--s5:#e87ba4;--s6:#008300;--deemph:#c3c2b7;--track:#f0efec;--win:#2a78d6;--loss:#e34948;--mid:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;
--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--deemph:#52514e;--track:#262624;
--win:#3987e5;--loss:#e66767;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--s5:#d55181;--s6:#008300;--deemph:#52514e;--track:#262624;--win:#3987e5;--loss:#e66767;--mid:#383835}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 8px}.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile .l{color:var(--ink2);font-size:13px}.tile .v{font-size:26px;font-weight:600}.tile .d{color:var(--ink2);font-size:13px}
ul.kp{margin:0;padding-left:20px}ul.kp li{margin:4px 0}
svg{width:100%;height:auto;display:block}.chart{overflow-x:auto}.chart svg{min-width:620px}svg text{fill:var(--ink2);font:12px system-ui,sans-serif}
svg .v{fill:var(--ink)}svg .b{font-weight:600}.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:13px;margin-top:8px}
.legend i,.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
.wrap{overflow-x:auto}details{margin-top:12px}pre{white-space:pre-wrap;font-size:12px;color:var(--ink2)}
.note{color:var(--ink2);font-size:13px}
.hh{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}.hh .lab{color:var(--ink2);font-size:13px}
.hh .t{margin-left:auto;font-variant-numeric:tabular-nums}
.tag{font-size:12px;border:1px solid var(--ring);border-radius:999px;padding:0 8px;color:var(--ink2);white-space:nowrap}
.hrow{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin:10px 0 12px}
svg.cores{width:204px;flex:none}ul.facts{margin:0;padding-left:18px;font-size:13px;color:var(--ink2)}
ul.facts b{color:var(--ink)}.pipe-n{display:none}
.hm-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:0 12px}
.hm h3{font-size:14px;margin:0 0 2px}.hm .note{margin:0 0 8px}.hx{overflow-x:auto}.tile .v{overflow-wrap:anywhere}
svg .cell text{fill:var(--ink);font-size:11px}.legend .hatch{background:repeating-linear-gradient(45deg,var(--grid) 0 2px,transparent 2px 5px)}
.legend .xhatch{background:repeating-linear-gradient(-45deg,color-mix(in srgb,var(--ink) 30%,transparent) 0 1px,var(--mid) 1px 4px)}
@media (max-width:640px){.pipe-w{display:none}.pipe-n{display:block}.hh .t{margin-left:0}}
"""

E = html.escape
MID = " · "
CW = 6.6  # average glyph width of the 12px chart font, for fitting labels into boxes


def _fit(s, px, cw=CW):
    s = str(s)
    n = int(px / cw)
    return s if len(s) <= n else s[:max(1, n - 1)] + "…"


def _grid_svg(p, col):
    """The chip's core (cluster) grid; the cores a config uses are filled in its color, lighter when they carry a
    smaller share of the work (the uneven tail of the split)."""
    cols, rows = p["cols"], p["rows"]
    c, g = 22, 4
    W, H = cols * (c + g) - g, rows * (c + g) - g
    total = cols * rows
    used = p["used"] or 0
    cells = []
    if p["loads"] and sum(n for n, _, _ in p["loads"]) == used:
        top = max(wt for _, wt, _ in p["loads"]) or 1
        for n, wt, txt in p["loads"]:
            cells += [(wt / top, txt)] * n
    else:
        cells = [(1.0, "")] * used
    unit1 = p["unit"][:-1]
    out = [f'<svg class="cores" viewBox="0 0 {W} {H}" data-used="{min(used, total)}" role="img" '
           f'aria-label="{min(used, total)} of {total} {p["unit"]} used">']
    for i in range(total):
        x, y = (i % cols) * (c + g), (i // cols) * (c + g)
        if i < len(cells):
            f, txt = cells[i]
            out.append(f'<rect class="on" x="{x}" y="{y}" width="{c}" height="{c}" rx="4" fill="{col}" '
                       f'fill-opacity="{0.4 + 0.6 * f:.2f}"><title>{unit1} {i}{": " + E(txt) if txt else ""}</title></rect>')
        else:
            out.append(f'<rect class="off" x="{x + .5}" y="{y + .5}" width="{c - 1}" height="{c - 1}" rx="4" '
                       f'fill="var(--track)" stroke="var(--grid)"><title>{unit1} {i}: idle</title></rect>')
    out.append("</svg>")
    return "".join(out)


def _arrow(x1, y1, x2, y2):
    """A line with a filled head at (x2, y2); horizontal or vertical."""
    if y1 == y2:
        head = f"M{x2} {y2} l-7 -4 v8 z"
        x2 -= 6
    else:
        head = f"M{x2} {y2} l-4 -7 h8 z"
        y2 -= 6
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--axis)" stroke-width="1.5"/>'
            f'<path d="{head}" fill="var(--axis)"/>')


def _node_box(n, x, y, w, h, col):
    tip = E(n["title"] + (": " + ", ".join(n["lines"]) if n["lines"] else "")
            + (f" | busy {n['busy']:.0%} of the run" if n.get("busy") is not None else "")
            + (" | bottleneck" if n.get("hot") else ""))
    if n["kind"] == "mem":
        box = (f'<rect x="{x + .5}" y="{y + .5}" width="{w - 1}" height="{h - 1}" rx="8" fill="var(--page)" '
               f'stroke="var(--axis)" stroke-dasharray="4 3"/>')
    else:
        box = (f'<rect x="{x + .5}" y="{y + .5}" width="{w - 1}" height="{h - 1}" rx="{6 if n["kind"] == "dfb" else 8}" '
               f'fill="var(--surface)" stroke="{col if n.get("hot") else "var(--axis)"}" '
               f'stroke-width="{2 if n.get("hot") else 1}"/>')
        if n.get("hot"):
            box += f'<rect x="{x + 1}" y="{y + 1}" width="{w - 2}" height="{h - 2}" rx="8" fill="{col}" fill-opacity=".12"/>'
    return f'<g class="node {n["kind"]}"><title>{tip}</title>{box}', "</g>"


def _busy_bar(n, x, y, w, col):
    f = min(1.0, max(0.0, n["busy"]))
    bw = w - 64
    out = f'<rect x="{x}" y="{y}" width="{bw}" height="6" rx="3" fill="var(--track)"/>'
    if bw * f >= 1:
        out += f'<rect x="{x}" y="{y}" width="{bw * f:.1f}" height="6" rx="3" fill="{col if n.get("hot") else "var(--ink2)"}"/>'
    return out + f'<text x="{x + w}" y="{y + 7}" text-anchor="end" class="v">{f:.0%} busy</text>'


def _pipe_svg(nodes, col, narrow=False):
    """The per-core program as boxes and arrows, source memory to destination memory. Wide: one row; narrow (phone):
    one column. The bottleneck role is outlined and tinted in the config's color."""
    out = []
    if not narrow:
        WID = {"mem": 90, "dfb": 100, "role": 140}
        gap, y0, h = 20, 18, 100
        W = sum(WID[n["kind"]] for n in nodes) + gap * (len(nodes) - 1)
        x = 0
        for i, n in enumerate(nodes):
            w = WID[n["kind"]]
            a, z = _node_box(n, x, y0, w, h, col)
            out.append(a)
            if n.get("hot"):
                out.append(f'<text x="{x + w / 2}" y="12" text-anchor="middle" class="v b">bottleneck</text>')
            cx = x + w / 2
            if n["kind"] == "role":
                out.append(f'<text x="{x + 10}" y="{y0 + 20}" class="v b">{E(_fit(n["title"], w - 20, 7.2))}</text>')
                for j, ln in enumerate(n["lines"][:3]):
                    out.append(f'<text x="{x + 10}" y="{y0 + 40 + 15 * j}">{E(_fit(ln, w - 20))}</text>')
                if n.get("busy") is not None:
                    out.append(_busy_bar(n, x + 10, y0 + h - 18, w - 20, col))
            else:
                lines = n["lines"][:3]
                ty = y0 + (34 if n["kind"] == "dfb" else h / 2 - 7.5 * len(lines) + 4)
                out.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" class="v b">{E(n["title"])}</text>')
                for j, ln in enumerate(lines):
                    out.append(f'<text x="{cx}" y="{ty + 17 + 15 * j}" text-anchor="middle">{E(_fit(ln, w - 12))}</text>')
                if n["kind"] == "dfb" and n.get("slots"):
                    k = min(int(n["slots"]), 8)
                    sx0 = cx - (k * 12 - 3) / 2
                    for j in range(k):
                        out.append(f'<rect x="{sx0 + 12 * j:.1f}" y="{y0 + h - 22}" width="9" height="9" rx="2" '
                                   f'fill="none" stroke="var(--ink2)"/>')
                    if n["slots"] > 8:
                        out.append(f'<text x="{sx0 + 12 * k + 2:.1f}" y="{y0 + h - 14}">+</text>')
            out.append(z)
            if i < len(nodes) - 1:
                out.append(_arrow(x + w + 3, y0 + h / 2, x + w + gap - 3, y0 + h / 2))
            x += w + gap
        H = y0 + h + 2
    else:
        W, x, y, gap = 340, 10, 2, 18
        w = W - 20
        for i, n in enumerate(nodes):
            role = n["kind"] == "role"
            h = 28 + 15 * len(n["lines"][:3]) + (16 if n.get("busy") is not None else 0) if role else 30
            a, z = _node_box(n, x, y, w, h, col)
            out.append(a)
            if role:
                out.append(f'<text x="{x + 10}" y="{y + 19}" class="v b">{E(n["title"])}</text>')
                if n.get("hot"):
                    out.append(f'<text x="{x + w - 10}" y="{y + 19}" text-anchor="end" class="v b">bottleneck</text>')
                for j, ln in enumerate(n["lines"][:3]):
                    out.append(f'<text x="{x + 10}" y="{y + 37 + 15 * j}">{E(_fit(ln, w - 20))}</text>')
                if n.get("busy") is not None:
                    out.append(_busy_bar(n, x + 10, y + h - 16, w - 20, col))
            else:
                out.append(f'<text x="{x + 10}" y="{y + 19}" class="v b">{E(n["title"])}</text>')
                if n["lines"]:
                    out.append(f'<text x="{x + w - 10}" y="{y + 19}" text-anchor="end">'
                               f'{E(_fit(MID.join(n["lines"]), w - 110))}</text>')
            out.append(z)
            if i < len(nodes) - 1:
                out.append(_arrow(W / 2, y + h + 2, W / 2, y + h + gap - 2))
            y += h + gap
        H = y - gap + 2
    return (f'<svg class="pipe" viewBox="0 0 {W} {H}" style="max-width:{W}px" role="img" '
            f'aria-label="Per-core pipeline">' + "".join(out) + "</svg>")


def _how_html(req, res, slots):
    shown = _shown(res)
    if not shown:
        return ""
    plans = [(rid, r, _plan(req, r)) for rid, r in shown]
    p0 = plans[0][2]
    unit, total = p0["unit"], p0["cols"] * p0["rows"]
    roles = [n["role"] for n in p0["nodes"] if n["kind"] == "role"]
    if "compute" in roles:
        flow = ("a reader moves data from the source memory into the input DFB (a ring buffer in the core's L1), compute "
                "transforms it into the output DFB, and a writer moves it to the destination.")
    else:
        flow = "the data-movement threads (" + " and ".join(roles) + ") move data between the source and destination memory."
    out = [f'<h2>How it runs</h2><p class="note">Quasar has {p0["cols"]} x {p0["rows"]} = {total} {unit}. Each used '
           f'{unit[:-1]} runs the same program on its share of the work: {flow} Filled cells: {unit} the config uses '
           f'(lighter: a smaller share). Outlined stage: the bottleneck; bar: share of the run the role is busy.'
           + (f' Counts are {p0["scope"]}.' if p0.get("scope") else "") + '</p>']
    for rid, r, p in plans:
        col = _col(slots.get(rid))
        t = r["time_ns"]["nominal"]
        sp = _speedup(r)
        tags = ""
        if rid != "baseline" and r.get("comparison"):
            tags += f'<span class="tag">{sp:.2f}x · {E(r["comparison"].get("verdict", ""))}</span>'
        if _hypothetical(r):
            tags += '<span class="tag">hypothetical</span>'
        facts = []
        if p["used"] is not None:
            facts.append(f'<b>{p["used"]} of {total} {unit}</b>' + (f', {E(p["split"])} split' if p.get("split") else "")
                         + (f' ({E(p["split_detail"])})' if p.get("split_detail") else ""))
        if p.get("factory"):
            facts.append(f'factory {E(str(p["factory"]))}')
        if p.get("per_core"):
            facts.append(f'per {unit[:-1]}: {E(p["per_core"])}')
        grid = _grid_svg(p, col) if p["used"] is not None else ""
        out.append(f'<div class="card how" data-config="{E(str(rid))}"><div class="hh"><span><i class="sw" '
                   f'style="background:{col}"></i><b>{E(str(rid))}</b></span><span class="lab">{E(r.get("label", ""))}</span>'
                   f'{tags}<span class="t">{_ns(t)}</span></div>'
                   f'<div class="hrow">{grid}<ul class="facts">' + "".join(f"<li>{f}</li>" for f in facts) + '</ul></div>'
                   f'<div class="pipe-w">{_pipe_svg(p["nodes"], col)}</div>'
                   f'<div class="pipe-n">{_pipe_svg(p["nodes"], col, narrow=True)}</div></div>')
    return "".join(out)


def _approaches_html(req, res, slots):
    rows = _approaches(req, res)
    head = ("config", "changes vs baseline", "time", "speedup", "verdict", "bottleneck", "")
    body = []
    for rid, ch, t, sp, v, bn, tags in rows:
        tag = "".join(f'<span class="tag">{E(x)}</span> ' for x in tags.split(", ") if x)
        body.append(f'<tr class="appr"><td><i class="sw" style="background:{_col(slots.get(rid))}"></i>{E(str(rid))}</td>'
                    f"<td>{E(ch)}</td><td>{E(t)}</td><td>{E(sp)}</td><td>{E(v)}</td><td>{E(bn)}</td><td>{tag}</td></tr>")
    return ('<h2>Approaches</h2><div class="card wrap"><table><tr>' + "".join(f"<th>{E(c)}</th>" for c in head)
            + "</tr>" + "".join(body) + "</table></div>")


def _bars(res, best_id, slots):
    """Horizontal time bars (ns) with low..high whiskers, one per config in its color; the recommended one is marked."""
    rows = [(rid, r) for rid, r, _ in _rows(res) if r.get("valid")]
    if not rows:
        return ""
    W, lab, right, bh, gap = 920, 260, 150, 22, 14
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
        w = max(2.0, x1 - lab)
        r4 = min(4, w / 2)
        sp = _speedup(r)
        val = f'{t["nominal"]:,.0f}' + (f" · {sp:.2f}x" if rid != "baseline" and sp is not None else "")
        name = _fit(str(rid) + (" (best)" if rid == best_id else ""), lab - 16)
        tip = f"{rid}: {r['label']} | {t['nominal']:,.0f} ns ({t['low']:,.0f} .. {t['high']:,.0f})"
        out.append(f'<g class="bar"><title>{E(tip)}</title>'
                   f'<text x="{lab - 10}" y="{y + bh / 2 + 4}" text-anchor="end" class="v{" b" if rid == best_id else ""}">'
                   f'{E(name)}</text>'
                   f'<path d="M{lab} {y} h{w - r4:.1f} q{r4:.1f} 0 {r4:.1f} {r4:.1f} v{bh - 2 * r4:.1f} '
                   f'q0 {r4:.1f} {-r4:.1f} {r4:.1f} h{-(w - r4):.1f} z" fill="{_col(slots.get(rid))}"/>'
                   f'<line x1="{sx(t["low"]):.1f}" x2="{sx(t["high"]):.1f}" y1="{y + bh / 2}" y2="{y + bh / 2}" '
                   f'stroke="var(--ink2)" stroke-width="1.5"/>'
                   f'<text x="{sx(t["high"]) + 8:.1f}" y="{y + bh / 2 + 4}" class="v">{val}</text></g>')
    out.append("</svg>")
    return "".join(out)


def _roles_cmp(entries):
    """Small multiples, one panel per measure (busy / starved / backpressure / waiting on NoC); in each, every role
    has one bar per config (grouped, the config's color) against 100% of the run. Not stacked: a compute thread's
    unpack/math and pack stages overlap, so shares can sum past 100%."""
    roles = []
    for _, r, _ in entries:
        roles += [k for k in (r.get("roles") or {}) if k not in roles]
    if not roles:
        return ""
    n = len(entries)
    W, lab, bh, ig, gg, top = 920, 110, (12 if n > 1 else 16), 2, 14, 22
    gh = n * bh + (n - 1) * ig
    pw = (W - lab) / len(ROLE_PARTS)
    H = top + len(roles) * (gh + gg)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of the run per role and measure, per config">']
    for i, role in enumerate(roles):
        y = top + i * (gh + gg)
        th = {(r.get("roles") or {}).get(role, {}).get("threads") for _, r, _ in entries} - {None}
        out.append(f'<text x="{lab - 10}" y="{y + gh / 2 + 4}" text-anchor="end" class="v">{E(role)}'
                   f'{f" x{th.pop()}" if len(th) == 1 else ""}</text>')
    for k, (key, name) in enumerate(ROLE_PARTS):
        x0 = lab + k * pw
        span = pw - 56
        out.append(f'<text x="{x0:.1f}" y="12" class="v">{name}</text>')
        for i, role in enumerate(roles):
            for j, (rid, r, slot) in enumerate(entries):
                v = (r.get("roles") or {}).get(role)
                if v is None:
                    continue
                y = top + i * (gh + gg) + j * (bh + ig)
                f = min(1.0, max(0.0, v.get(key, 0.0)))
                w = span * f
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{span:.1f}" height="{bh}" rx="3" fill="var(--track)"/>')
                if w >= 1:
                    r4 = min(3.0, w / 2)
                    out.append(f'<path d="M{x0:.1f} {y} h{w - r4:.1f} q{r4:.1f} 0 {r4:.1f} {r4:.1f} v{bh - 2 * r4:.1f} '
                               f'q0 {r4:.1f} {-r4:.1f} {r4:.1f} h{-(w - r4):.1f} z" fill="{_col(slot)}">'
                               f'<title>{E(str(rid))} {E(role)} x{v.get("threads", "?")}: {name} {f:.0%} of the run</title></path>')
                out.append(f'<text x="{x0 + w + 6:.1f}" y="{y + bh / 2 + 4}" class="v">{f:.0%}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{_col(slot)}"></i>{E(str(rid))}</span>' for rid, _, slot in entries)
    return "".join(out) + (f'<div class="legend">{legend}</div>' if n > 1 else "")


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
    across charts. A legend entry per series; end labels when they do not collide."""
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
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" stroke="var(--axis)" stroke-width="1.5"/>')
    for v in xt:
        out.append(f'<text x="{sx(v):.1f}" y="{H - 12}" text-anchor="middle">{v:,g}</text>')
    for sid, k, s in series:
        col = _col(k)
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
    ys = sorted([sy(yval(ids[0], s[-1])) for ids, s in groups] + ([sy(ref)] if ref else []))
    if all(b2 - a2 >= 14 for a2, b2 in zip(ys, ys[1:])):  # else the legend alone names the lines
        if ref:
            out.append(f'<text x="{W - r + 6}" y="{sy(ref) + 4:.1f}">baseline</text>')
        for ids, s in groups:
            out.append(f'<text x="{sx(s[-1]["value"]) + 10:.1f}" y="{sy(yval(ids[0], s[-1])) + 4:.1f}" class="v">'
                       f'{E(" = ".join(ids))}</text>')
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{_col(k)}"></i>{E(sid)}</span>' for sid, k, _ in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def _sensitivity_charts(sens, slots, req):
    """Time (and speedup vs the baseline) over the swept parameter: one line per series, colored like its config."""
    ids = list(sens["series"])
    free = iter(k for k in range(64) if k not in set(slots.values()))
    series = [(sid, slots[sid] if sid in slots else next(free), sens["series"][sid]) for sid in ids[:MAX_SLOTS]]
    out = _linechart(series, sens["param"], lambda sid, p: p.get("time_ns"), lambda v: f"{v:,.0f}",
                     f"Time vs {sens['param']}")
    base = {p["value"]: p.get("time_ns") for p in sens["series"].get("baseline", [])}
    cands = [x for x in series if x[0] != "baseline"]
    if cands and base:
        sp = lambda sid, p: (base[p["value"]] / p["time_ns"]) if p.get("time_ns") and base.get(p["value"]) else None  # noqa: E731
        out += (f'<h3 class="note" style="margin:16px 0 4px">Speedup vs baseline (above the line = faster)</h3>'
                + _linechart(cands, sens["param"], sp, lambda v: f"{v:.2f}x", "Speedup vs baseline", ref=1.0))
    missing = [c.get("id") for c in req.get("candidates", []) if c.get("id") and c.get("id") not in sens["series"]]
    if missing:
        out += (f'<p class="note">Not swept: {E(", ".join(missing))}. This op\'s sensitivity response carries only '
                f'the series shown.</p>')
    if len(ids) > MAX_SLOTS:
        out += f'<p class="note">{len(ids) - MAX_SLOTS} more series are in the table below.</p>'
    return out


def to_html(req, res):
    prof = res.get("profiles", {}).get("costs", {})
    title = E(f"{(req.get('workload') or {}).get('op', 'qperf')} {req.get('action', 'estimate')} report")
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{title}</title><style>{CSS}</style></head><body><main>'
            f'<h1>qperf: {E(str(req.get("action")))}</h1><p class="sub">{E(_workload(req))} · '
            f'{E(prof.get("name", ""))}@{E(prof.get("revision", ""))}</p>')
    return head + body_html(req, res) + "</main></body></html>"


def body_html(req, res, uid=""):
    """One result's report body (everything below the page title): to_html's page content, and one request's section
    of a bundle page (qperf/bundle.py). uid: suffix that keeps SVG pattern ids unique on a page of several results."""
    if res["status"] not in ("ok", "partial"):
        return f'<div class="card"><b>{E(res["status"])}</b>: {E(res.get("message", ""))}</div>'
    if res.get("survey"):
        return _survey_html(req, res, uid)
    b = res["baseline"]
    best = _best(res)
    slots = _slots(res)
    tiles = [("Baseline time", _ns(b["time_ns"]["nominal"]), b["label"]),
             ("Bottleneck", b["bottleneck"]["resource"], f"busiest role: {b['bottleneck'].get('busiest_role', '-')}")]
    if best and best[2]["comparison"]["verdict"] == "faster":
        tiles.insert(1, ("Best alternative", f"{best[0]:.2f}x faster", best[1]))
    elif best or res.get("search"):
        tiles.insert(1, ("Best alternative", "none faster",
                         f"best other: {best[1]} {best[0]:.2f}x" if best else "the baseline is the best config"))
    parts = ['<div class="tiles">']
    parts += [f'<div class="card tile"><div class="l">{E(l)}</div><div class="v">{E(v)}</div><div class="d">{E(d)}</div></div>'
              for l, v, d in tiles]
    parts.append('</div><div class="card"><h2 style="margin-top:0">Key points</h2><ul class="kp">')
    parts += [f"<li>{E(p)}</li>" for p in key_points(req, res)]
    parts.append("</ul></div>")
    parts.append(_how_html(req, res, slots))
    best_id = best[1] if best and best[2]["comparison"]["verdict"] == "faster" else None
    if len(_rows(res)) > 1:
        parts.append(_approaches_html(req, res, slots))
        parts.append('<h2>Predicted time per configuration (ns)</h2><div class="card chart">' + _bars(res, best_id, slots) +
                     '<p class="note">Bar: nominal estimate, with the speedup vs the baseline. Whisker: all costs at their '
                     'low / high scenario. (best): the alternative that is faster in every scenario.</p></div>')
    if res.get("sensitivity"):
        parts.append(f'<h2>Sensitivity to {E(res["sensitivity"]["param"])}</h2><div class="card chart">'
                     + _sensitivity_charts(res["sensitivity"], slots, req) + "</div>")
    entries = [("baseline", b, slots.get("baseline", 0))]
    if best and best[2].get("roles"):
        entries.append((best[1], best[2], slots.get(best[1])))
    title = "Where the time goes" + (f": baseline vs {best[1]}" if len(entries) > 1 else " (baseline)")
    if b.get("roles"):
        parts.append(f'<h2>{E(title)}, share of the run</h2><div class="card chart">{_roles_cmp(entries)}</div>')
    md = to_markdown(req, res)
    table = md.split("## Configurations", 1)[1].split("\n## ", 1)[0].strip().splitlines()
    head_cells = [c.strip() for c in table[0].strip("|").split("|")]
    body = "".join("<tr>" + "".join(f"<td>{E(c.strip())}</td>" for c in row.strip("|").split("|")) + "</tr>" for row in table[2:])
    parts.append('<h2>All numbers</h2><div class="card wrap"><table><tr>' + "".join(f"<th>{E(c)}</th>" for c in head_cells)
                 + f"</tr>{body}</table></div>")
    parts.append(f'{_scope_notes(res)}<details><summary>Request</summary><pre>'
                 f'{E(json.dumps(req, indent=1))}</pre></details>')
    return "".join(parts)


def _scope_notes(res):
    """The platform (what the simulator times) and the op's accuracy (how close the model is), as notes."""
    out = f'<p class="note">{E(res.get("platform_scope", ""))}</p>'
    if res.get("accuracy"):
        out += f'<p class="note accuracy"><b>Accuracy:</b> {E(res["accuracy"])}</p>'
    return out


# ------------------------------------------------------------------------------------------------ survey

def _sx(shape):
    return "x".join(str(d) for d in shape) if shape else "-"


def _survey_order(sv):
    """Approach ids, best geomean first."""
    return sorted(sv["approaches"], key=lambda k: -(sv["approaches"][k]["speedup"]["geomean"] or 0))


def _evaluated(a):
    c = a["counts"]
    return c["faster"] + c["neutral"] + c["slower"]


def _counts_txt(a):
    c = a["counts"]
    return f"{c['faster']} / {c['neutral']} / {c['slower']}" + (f" (+{c['invalid']} invalid)" if c["invalid"] else "")


def _regs_txt(a):
    return ", ".join(f"{_sx(r['shape'])} {r['speedup']:.2f}x" for r in a["regressions"]) or "none"


def _cmp_speedup(cmp, aid):
    """An approach's speedup {geomean, min, max} in survey.compare, or {}."""
    return ((cmp or {}).get("approaches", {}).get(aid) or {}).get("speedup") or {}


def _geo_txt(sp):
    return "-" if sp.get("geomean") is None else f"{sp['geomean']:.2f}x"


def _range_txt(sp):
    return "-" if sp.get("min") is None else f"{sp['min']:.2f}..{sp['max']:.2f}x"


def _untimed(sv, i, j):
    """True when an approach's comparison at [i][j] involves a side the profile does not time (layout choices)."""
    return any((a.get("untimed") or [[None]])[i][j] if a.get("untimed") else False for a in sv["approaches"].values())


def _outside(sv, i, j):
    g = sv.get("outside_calibrated_range")
    return bool(g and g[i][j])


def _headline(sv, res):
    """(id, approach) of the best general approach: the first recommendation, else the best geomean."""
    rid = next((r["id"] for r in res.get("recommendations", []) if r.get("id")), None)
    if rid is None:
        order = _survey_order(sv)
        rid = order[0] if order else None
    return rid, sv["approaches"].get(rid) if rid else None


def _survey_md(req, res):
    sv = res["survey"]
    L = ["## Key points", ""] + [f"- {x}" for x in sv["summary"]]
    for r in res.get("recommendations", []):
        L.append(f"- Recommend {r['label']}: {r.get('reason', '')}")
    L += [f"- Warning: {w}" for w in res.get("warnings", [])]
    cmp = sv.get("compare")
    me = (res.get("profiles", {}).get("costs") or {}).get("name", "this profile")
    ccols = f" {me} min..max | {cmp['profile']['name']} geomean | {cmp['profile']['name']} min..max |" if cmp else ""
    L += ["", "## Approaches", "",
          f"| approach | overrides | {me + ' ' if cmp else ''}geomean | best shape | worst shape | "
          f"faster / neutral / slower | regressions |{ccols}",
          "|---|---|---|---|---|---|---|" + ("---|---|---|" if cmp else "")]
    for aid in _survey_order(sv):
        a = sv["approaches"][aid]
        ov = describe(a["overrides"]) or "-"
        g = a["speedup"]["geomean"]
        L.append(f"| {aid}{' (hypothetical)' if a['hypothetical'] else ''}"
                 f"{' (untimed on ' + str(a['untimed_cells']) + ')' if a.get('untimed_cells') else ''} | {ov} | "
                 f"{'-' if g is None else f'{g:.2f}x'} | "
                 f"{_sx(a['best']['shape']) + ' ' + format(a['best']['speedup'], '.2f') + 'x' if a['best'] else '-'} | "
                 f"{_sx(a['worst']['shape']) + ' ' + format(a['worst']['speedup'], '.2f') + 'x' if a['worst'] else '-'} | "
                 f"{_counts_txt(a)} | {a['counts']['slower']} |" +
                 (f" {_range_txt(a['speedup'])} | {_geo_txt(_cmp_speedup(cmp, aid))} | "
                  f"{_range_txt(_cmp_speedup(cmp, aid))} |" if cmp else ""))
    if cmp:
        L += ["", f"## On {cmp['profile']['name']} (compare_profile)", "",
              f"**Platform:** {cmp.get('platform_scope', '-')}  ", f"**Accuracy:** {cmp.get('accuracy', '-')}", ""]
        L += [f"- {x}" for x in cmp["summary"]]
    L += ["", "## Regressions", ""]
    for aid in _survey_order(sv):
        a = sv["approaches"][aid]
        if a["regressions"]:
            more = a["counts"]["slower"] - len(a["regressions"])
            L.append(f"- **{aid}**: slower on {a['counts']['slower']}/{_evaluated(a)}"
                     + (f" ({a['loses_where']})" if a.get("loses_where") else "") + f": {_regs_txt(a)}"
                     + (f" and {more} more" if more > 0 else ""))
    if L[-1] == "":
        L.append("- none: no approach is slower than the baseline beyond the noise threshold at any shape")
    hs, ws = sv["heights"], sv["widths"]
    L += ["", "## Best approach per shape", "", "| H \\ W | " + " | ".join(str(w) for w in ws) + " |",
          "|---|" + "---|" * len(ws)]
    for i, h in enumerate(hs):
        L.append(f"| {h} | " + " | ".join((sv["best_per_shape"][i][j] or "skipped") + (" *" if _outside(sv, i, j) else "")
                                          + (" †" if _untimed(sv, i, j) else "") for j in range(len(ws))) + " |")
    if any(_outside(sv, i, j) for i in range(len(hs)) for j in range(len(ws))):
        rng = sv["calibrated_range"]
        L += ["", f"\\* extrapolated: outside the calibrated range on {rng['profile']} ({accuracy_range_txt(rng)})"]
    if any(_untimed(sv, i, j) for i in range(len(hs)) for j in range(len(ws))):
        L += ["", "† untimed: the profile does not time a side of this comparison; not meaningful"]
    if sv["skipped"]:
        L += ["", "## Skipped shapes", ""] + [f"- {_sx(s['shape'])}: {s['reason']}" for s in sv["skipped"]]
    L += ["", "## Request", "", "```json", json.dumps(req, indent=1), "```", ""]
    return L


HM_CELL, HM_H, HM_GAP, HM_LAB, HM_TOP = 56, 28, 2, 52, 22


def _hm_frame(hs, ws, cw, idx):
    """SVG head (width labels, hatch pattern) and geometry for a heights x widths grid of cells cw wide."""
    W = HM_LAB + len(ws) * (cw + HM_GAP)
    H = HM_TOP + len(hs) * (HM_H + HM_GAP)
    out = [f'<defs><pattern id="hatch{idx}" width="6" height="6" patternUnits="userSpaceOnUse" '
           f'patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--surface)"/>'
           f'<line x1="0" y1="0" x2="0" y2="6" stroke="var(--grid)" stroke-width="3"/></pattern>'
           f'<pattern id="xhatch{idx}" width="5" height="5" patternUnits="userSpaceOnUse" '
           f'patternTransform="rotate(-45)"><line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink)" '
           f'stroke-opacity=".28" stroke-width="1.2"/></pattern></defs>',
           f'<text x="{HM_LAB - 8}" y="14" text-anchor="end">H \\ W</text>']
    for j, w in enumerate(ws):
        out.append(f'<text x="{HM_LAB + j * (cw + HM_GAP) + cw / 2}" y="14" text-anchor="middle">{w}</text>')
    for i, h in enumerate(hs):
        out.append(f'<text x="{HM_LAB - 8}" y="{HM_TOP + i * (HM_H + HM_GAP) + HM_H / 2 + 4}" text-anchor="end">{h}</text>')
    return W, H, out


XTIP = " (extrapolated: outside the calibrated range)"


def _x_overlay(x, y, cw, idx):
    """The extrapolation hatch over an evaluated cell (a shape beyond the calibrated range)."""
    return (f'<rect class="xhatch" x="{x}" y="{y}" width="{cw}" height="{HM_H}" rx="4" fill="url(#xhatch{idx})" '
            f'pointer-events="none"/>')


def _skip_cell(x, y, cw, idx, tip):
    return (f'<g class="cell skipped"><title>{E(tip)}</title><rect x="{x}" y="{y}" width="{cw}" height="{HM_H}" rx="4" '
            f'fill="url(#hatch{idx})" stroke="var(--grid)"/></g>')


def _heat_fill(s, noise):
    """Diverging fill centred at 1.0: wins blend --win, regressions --loss into the neutral --mid, by |log2 s| up to
    4x either way; within the noise band the neutral midpoint."""
    if abs(s - 1) <= noise:
        return "var(--mid)"
    t = min(1.0, abs(math.log2(s)) / 2)
    return f"color-mix(in srgb, var({'--win' if s > 1 else '--loss'}) {15 + 55 * t:.0f}%, var(--mid))"


UTIP = " (untimed: the profile does not time a side of this comparison; not meaningful)"


def _heatmap(sv, aid, idx, noise):
    a = sv["approaches"][aid]
    ug = a.get("untimed")
    hs, ws, cw = sv["heights"], sv["widths"], HM_CELL
    skipped = {tuple(s["shape"]): s["reason"] for s in sv["skipped"]}
    W, H, out = _hm_frame(hs, ws, cw, idx)
    for i, h in enumerate(hs):
        for j, w in enumerate(ws):
            x, y = HM_LAB + j * (cw + HM_GAP), HM_TOP + i * (HM_H + HM_GAP)
            s, bt = a["grid"][i][j], sv["baseline_ns"][i][j]
            if bt is None:
                out.append(_skip_cell(x, y, cw, idx, f"{h}x{w}: {skipped.get((h, w), 'not evaluated')}"))
            elif s is None:
                out.append(f'<g class="cell invalid"><title>{E(aid)} at {h}x{w}: the shape cannot take these knobs '
                           f'(baseline {bt:,.0f} ns)</title><rect x="{x + .5}" y="{y + .5}" width="{cw - 1}" '
                           f'height="{HM_H - 1}" rx="4" fill="var(--surface)" stroke="var(--axis)" stroke-dasharray="3 2"/>'
                           f'<text x="{x + cw / 2}" y="{y + HM_H / 2 + 4}" text-anchor="middle">n/a</text></g>')
            else:
                verdict = "faster" if s > 1 + noise else "slower" if s < 1 - noise else "neutral"
                xo = _outside(sv, i, j)
                un = bool(ug and ug[i][j])
                tip = (f"{aid} at {h}x{w}: {s:.3f}x, {verdict} (baseline {bt:,.0f} ns -> {bt / s:,.0f} ns)"
                       + (XTIP if xo else "") + (UTIP if un else ""))
                if un:
                    out.append(f'<g class="cell {verdict}{" extrapolated" if xo else ""} untimed" data-shape="{h}x{w}" '
                               f'data-speedup="{s}"><title>{E(tip)}</title>'
                               f'<rect x="{x + .5}" y="{y + .5}" width="{cw - 1}" height="{HM_H - 1}" rx="4" '
                               f'fill="var(--surface)" stroke="var(--ink2)" stroke-dasharray="2 2"/>'
                               f'<text x="{x + cw / 2}" y="{y + HM_H / 2 + 4}" text-anchor="middle">{s:.2f}'
                               f'{"*" if xo else ""}†</text></g>')
                    continue
                out.append(f'<g class="cell {verdict}{" extrapolated" if xo else ""}" data-shape="{h}x{w}" '
                           f'data-speedup="{s}"><title>{E(tip)}</title>'
                           f'<rect x="{x}" y="{y}" width="{cw}" height="{HM_H}" rx="4" style="fill:{_heat_fill(s, noise)}"/>'
                           + (_x_overlay(x, y, cw, idx) if xo else "") +
                           f'<text x="{x + cw / 2}" y="{y + HM_H / 2 + 4}" text-anchor="middle">{s:.2f}'
                           f'{"*" if xo else ""}</text></g>')
    return (f'<svg class="heatmap" data-approach="{E(aid)}" viewBox="0 0 {W} {H}" style="width:{W}px;height:{H}px;max-width:none" role="img" '
            f'aria-label="Speedup of {E(aid)} vs the baseline per shape">' + "".join(out) + "</svg>")


def _heat_legend(noise, extrapolated=False, untimed=False):
    stops = [(0.25, "0.25x"), (0.5, "0.5x"), (0.8, "0.8x"), (1.0, f"1x ±{noise:.0%}"), (1.25, "1.25x"), (2.0, "2x"),
             (4.0, "4x+")]
    items = "".join(f'<span><i style="background:{_heat_fill(v, noise)}"></i>{E(t)}</span>' for v, t in stops)
    xh = ('<span><i class="xhatch"></i>* extrapolated: outside the calibrated range</span>' if extrapolated else "")
    if untimed:
        xh += ('<span><i style="border:1px dashed var(--ink2)"></i>† untimed: the profile does not time a side; '
               'not meaningful</span>')
    return (f'<div class="legend">{items}<span><i class="hatch"></i>skipped</span>{xh}'
            f'<span><i style="border:1px dashed var(--axis)"></i>n/a: knobs invalid at the shape</span></div>')


def _best_map(sv, idx):
    hs, ws = sv["heights"], sv["widths"]
    cw = 92
    freq = {}
    for row in sv["best_per_shape"]:
        for b in row:
            if b and b != "baseline":
                freq[b] = freq.get(b, 0) + 1
    order = [k for k in _survey_order(sv) if k in freq]
    order.sort(key=lambda k: -freq[k])
    slot = {k: n for n, k in enumerate(order)}
    skipped = {tuple(s["shape"]): s["reason"] for s in sv["skipped"]}
    W, H, out = _hm_frame(hs, ws, cw, idx)
    for i, h in enumerate(hs):
        for j, w in enumerate(ws):
            x, y = HM_LAB + j * (cw + HM_GAP), HM_TOP + i * (HM_H + HM_GAP)
            b = sv["best_per_shape"][i][j]
            if b is None:
                out.append(_skip_cell(x, y, cw, idx, f"{h}x{w}: {skipped.get((h, w), 'not evaluated')}"))
                continue
            s = 1.0 if b == "baseline" else sv["approaches"][b]["grid"][i][j]
            fill = "var(--mid)" if b == "baseline" else f"color-mix(in srgb, {_col(slot[b])} 60%, var(--surface))"
            tip = f"{h}x{w}: {b}" + ("" if b == "baseline" else f" {s:.2f}x") + \
                  (f" (baseline {sv['baseline_ns'][i][j]:,.0f} ns)" if sv["baseline_ns"][i][j] is not None else "")
            xo = _outside(sv, i, j)
            tip += XTIP if xo else ""
            un = _untimed(sv, i, j)
            tip += UTIP if un else ""
            out.append(f'<g class="cell best{" extrapolated" if xo else ""}{" untimed" if un else ""}" data-best="{E(b)}">'
                       f'<title>{E(tip)}</title>'
                       f'<rect x="{x}" y="{y}" width="{cw}" height="{HM_H}" rx="4" style="fill:{fill}"'
                       + (' stroke="var(--ink2)" stroke-dasharray="2 2"' if un else "") + '/>'
                       + (_x_overlay(x, y, cw, idx) if xo else "") +
                       f'<text x="{x + cw / 2}" y="{y + HM_H / 2 + 4}" text-anchor="middle">'
                       f'{E(_fit(b, cw - 8 - (8 if un else 0), 6.2))}{"†" if un else ""}</text></g>')
    legend = "".join(f'<span><i style="background:{_col(slot[k])}"></i>{E(k)} ({freq[k]})</span>' for k in order)
    n_base = sum(b == "baseline" for row in sv["best_per_shape"] for b in row)
    legend += f'<span><i style="background:var(--mid)"></i>baseline: nothing faster ({n_base})</span>'
    n_x = (sv.get("stats") or {}).get("outside_calibrated_range")
    if n_x:
        legend += f'<span><i class="xhatch"></i>extrapolated: outside the calibrated range ({n_x})</span>'
    if any(_untimed(sv, i, j) for i in range(len(hs)) for j in range(len(ws))):
        legend += ('<span><i style="border:1px dashed var(--ink2)"></i>† untimed: the profile does not time a '
                   'side; not meaningful</span>')
    return (f'<svg class="bestmap" viewBox="0 0 {W} {H}" style="width:{W}px;height:{H}px;max-width:none" role="img" '
            f'aria-label="Fastest approach per shape">' + "".join(out) + f'</svg><div class="legend">{legend}</div>')


def _survey_html(req, res, uid=""):
    from .api import NOISE

    sv = res["survey"]
    order = _survey_order(sv)
    hid, ha = _headline(sv, res)
    rec = next((r for r in res.get("recommendations", []) if r.get("id")), None)
    tiles = []
    if ha and ha["speedup"]["geomean"] is not None:
        tiles.append(("Best general approach", hid, (rec or {}).get("reason") or "highest geomean speedup (not recommended)"))
        tiles.append(("Its geomean speedup", f"{ha['speedup']['geomean']:.2f}x",
                      f"faster on {ha['counts']['faster']} of {_evaluated(ha)} shapes"))
        w = ha["worst"]
        tiles.append(("Its worst shape", f"{w['speedup']:.2f}x" if w else "-",
                      f"at {_sx(w['shape'])}" + (f"; slower on {ha['counts']['slower']}" if ha["counts"]["slower"] else
                                                 "; never slower") if w else ""))
    else:
        tiles.append(("Best general approach", "baseline", "no approach is faster"))
    st = sv.get("stats") or {}
    out = ['<div class="tiles">']
    out += [f'<div class="card tile"><div class="l">{E(l)}</div><div class="v">{E(str(v))}</div><div class="d">{E(d)}</div>'
            f'</div>' for l, v, d in tiles]
    out.append('</div><div class="card"><h2 style="margin-top:0">Key points</h2><ul class="kp">')
    out += [f"<li>{E(x)}</li>" for x in sv["summary"]]
    out += [f"<li>Recommend {E(r['label'])}: {E(r.get('reason', ''))}</li>" for r in res.get("recommendations", [])]
    out += [f"<li>{E(w[0].upper() + w[1:])}.</li>" for w in res.get("warnings", [])]
    out.append(f'</ul><p class="note">{st.get("evaluated", "?")} of {st.get("shapes", "?")} shapes evaluated with '
               f'{st.get("approaches", len(order))} approaches; {st.get("plans_evaluated", "?")} distinct plans timed '
               f'({st.get("plans_reused", "?")} reused) in {st.get("eval_ms", "?")} ms. {E(sv.get("speedup_basis", ""))}.'
               f'</p></div>')
    cmp = sv.get("compare")
    me = (res.get("profiles", {}).get("costs") or {}).get("name", "")
    head = ("approach", "overrides", f"{me} geomean" if cmp else "geomean", "best shape", "worst shape",
            "faster / neutral / slower", "regressions")
    if cmp:
        cn = cmp["profile"]["name"]
        head += (f"{me} min..max", f"{cn} geomean", f"{cn} min..max")
    body = []
    for aid in order:
        a = sv["approaches"][aid]
        g = a["speedup"]["geomean"]
        tag = ' <span class="tag">hypothetical</span>' if a["hypothetical"] else ""
        if a.get("untimed_cells"):
            tag += f' <span class="tag">untimed on {a["untimed_cells"]}</span>'
        extra = "".join(f"<td>{E(t)}</td>" for t in (_range_txt(a["speedup"]), _geo_txt(_cmp_speedup(cmp, aid)),
                                                       _range_txt(_cmp_speedup(cmp, aid)))) if cmp else ""
        body.append(f'<tr class="surv"><td><b>{E(aid)}</b>{tag}</td>'
                    f'<td>{E(describe(a["overrides"]) or "-")}</td>'
                    f'<td>{"-" if g is None else f"{g:.2f}x"}</td>'
                    f'<td>{E(_sx(a["best"]["shape"]) + " · " + format(a["best"]["speedup"], ".2f") + "x") if a["best"] else "-"}</td>'
                    f'<td>{E(_sx(a["worst"]["shape"]) + " · " + format(a["worst"]["speedup"], ".2f") + "x") if a["worst"] else "-"}</td>'
                    f'<td>{E(_counts_txt(a))}</td><td>{E(_regs_txt(a))}</td>{extra}</tr>')
    out.append('<h2>Approaches</h2><div class="card wrap"><table><tr>' + "".join(f"<th>{E(c)}</th>" for c in head)
               + "</tr>" + "".join(body) + "</table>"
               + (f'<p class="note"><b>{E(cmp["profile"]["name"])}:</b> {E(cmp.get("platform_scope", ""))} '
                  f'<b>Accuracy:</b> {E(cmp.get("accuracy", ""))}</p><ul class="kp">'
                  + "".join(f"<li>{E(x)}</li>" for x in cmp["summary"]) + "</ul>" if cmp else "") + "</div>")
    n_x = st.get("outside_calibrated_range")
    n_u = sum(a.get("untimed_cells", 0) for a in sv["approaches"].values())
    out.append(f'<h2>Speedup vs baseline per shape</h2><p class="note">Rows: tensor rows H; columns: width W. Blue: '
               f'faster than the baseline, red: slower, gray: within ±{NOISE:.0%}. Hover a cell for the times.'
               + (f' Hatched with *: {n_x} shapes outside the calibrated range (extrapolated).' if n_x else "")
               + (' Dashed with †: untimed, not meaningful on this profile.' if n_u else "") +
               f'</p><div class="card">{_heat_legend(NOISE, bool(n_x), bool(n_u))}</div><div class="hm-grid">')
    for k, aid in enumerate(order):
        a = sv["approaches"][aid]
        g = a["speedup"]["geomean"]
        where = "; ".join(x for x in (a.get("wins_where") and f"wins {a['wins_where']}",
                                      a.get("loses_where") and f"loses {a['loses_where']}") if x)
        out.append(f'<div class="card hm" data-approach="{E(aid)}"><h3>{E(aid)}'
                   f'{" <span class=tag>hypothetical</span>" if a["hypothetical"] else ""}'
                   f'{" <span class=tag>untimed</span>" if a.get("untimed_cells") else ""}</h3>'
                   f'<p class="note">{E(describe(a["overrides"]) or "no overrides")} · '
                   f'{"-" if g is None else f"{g:.2f}x geomean"} · {E(_counts_txt(a))} faster / neutral / slower'
                   f'{" · " + E(where) if where else ""}</p><div class="hx">{_heatmap(sv, aid, f"{uid}{k}", NOISE)}</div></div>')
    out.append("</div>")
    out.append('<h2>Best approach per shape</h2><div class="card"><p class="note" style="margin-top:0">The fastest '
               f'approach at each shape when it beats the baseline by more than {NOISE:.0%}; otherwise baseline. '
               f'Legend counts are shapes won.</p><div class="hx">{_best_map(sv, f"{uid}{len(order)}")}</div></div>')
    if sv["skipped"]:
        out.append('<h2>Skipped shapes</h2><div class="card"><ul class="kp">'
                   + "".join(f"<li>{E(_sx(s['shape']))}: {E(s['reason'])}</li>" for s in sv["skipped"]) + "</ul></div>")
    out.append(f'{_scope_notes(res)}<details><summary>Request</summary><pre>'
               f'{E(json.dumps(req, indent=1))}</pre></details>')
    return "".join(out)
