"""Several analyze() requests answered as one report (a bundle): `qperf analyze --request a.json --request b.json ...`
or a bundle file {"title": str, "requests": [request, ...]}.

On top of each request's own section (the single-request report: How it runs, heatmaps, charts, numbers, accuracy),
the bundle states, across requests:
- one summary line per op: today's factory and limiter, the best change and the worst regression;
- two ranking tables that are never merged: kernel changes (hypothetical survey approaches / tune or compare setups)
  and layout choices (shard / reshard options: they change what the op and the next op receive). Rows are ranked by
  the ttsim-perf geomean speedup over one shape grid (the grid is stated; rows on another grid or a single shape come
  after); craq-sim rows use a survey's compare_profile ttsim-perf when present;
- the caps: what the assumed DRAM bandwidth of silicon-hypothesis does to the top rows when a request ran it, else a
  note that DRAM is untimed.
Nothing is recomputed from the numbers: every value comes from the results (plus one compare per survey at a
mid-grid shape, to draw how the baseline and the top approach run).
"""

from __future__ import annotations

import copy
import html
import json
import math

from .api import analyze
from .survey import (DEFAULT_HEIGHTS, DEFAULT_RESHARD, DEFAULT_SHARD, DEFAULT_WIDTHS, SHARD_OPS, describe, kind_of,
                     shape_label)

RANK_PROFILE = "ttsim-perf"  # the profile the rankings use (it times L1 NoC transfers; craq-sim flatters small requests)
DRAM_PROFILE_HINT = "silicon-hypothesis"
E = html.escape


class BundleError(Exception):
    pass


def load(paths):
    """(title or None, [request]) from request files and / or bundle files ({"title", "requests": [...]})."""
    title, reqs = None, []
    for p in paths:
        with open(p) as f:
            doc = json.load(f)
        if isinstance(doc, dict) and isinstance(doc.get("requests"), list):
            title = title or doc.get("title")
            reqs += doc["requests"]
        elif isinstance(doc, dict):
            reqs.append(doc)
        else:
            raise BundleError(f"{p}: a request object or a bundle {{\"title\", \"requests\": [...]}}")
    if not reqs:
        raise BundleError("no requests")
    return title, reqs


def is_bundle(paths):
    """True when the paths need the bundle path: several files, or one bundle file."""
    if len(paths) != 1 or paths[0] == "-":
        return len(paths) > 1
    with open(paths[0]) as f:
        doc = json.load(f)
    return isinstance(doc, dict) and isinstance(doc.get("requests"), list)


# ------------------------------------------------------------------------------------------------- running

def _mid_shape(sv):
    """An evaluated grid shape near the middle of the grid (log scale), or None."""
    hs, ws = sv["heights"], sv["widths"]
    hm, wm = hs[len(hs) // 2], ws[len(ws) // 2]
    cells = [(abs(math.log2(h / hm)) + abs(math.log2(w / wm)), i, j) for i, h in enumerate(hs) for j, w in enumerate(ws)
             if sv["baseline_ns"][i][j] is not None]
    if not cells:
        return None
    _, i, j = min(cells)
    return hs[i], ws[j]


def _top_approach(sv):
    """The approach with the best geomean among those without untimed cells (any, if all are untimed), or None."""
    aps = sv["approaches"]
    timed = [k for k in aps if not aps[k].get("untimed_cells") and aps[k]["speedup"]["geomean"] is not None]
    pool = timed or [k for k in aps if aps[k]["speedup"]["geomean"] is not None]
    return max(pool, key=lambda k: aps[k]["speedup"]["geomean"]) if pool else None


def _example(req, res, detail):
    """For a survey: one compare at a mid-grid shape (baseline vs the top approach), to draw how they run."""
    sv = res.get("survey")
    if not sv or res["status"] not in ("ok", "partial"):
        return None
    shape = _mid_shape(sv)
    if not shape:
        return None
    w = dict(req["workload"], shape=[1, 1, *shape])
    if w["op"] in SHARD_OPS and not w.get("shard") and not {"shard_strategy", "shard_cores"} <= set(req.get("baseline") or {}):
        w["shard"] = dict(DEFAULT_SHARD)
    if w["op"] == "reshard" and not w.get("reshard") and "reshard" not in (req.get("baseline") or {}):
        w["reshard"] = copy.deepcopy(DEFAULT_RESHARD)
    top = _top_approach(sv)
    q = {"schema_version": req.get("schema_version", "1"), "action": "compare", "profiles": req["profiles"],
         "workload": w, "baseline": req.get("baseline") or {},
         "candidates": [{"id": top, "overrides": sv["approaches"][top]["overrides"]}] if top else [],
         "analysis": {"detail": detail}}
    if req.get("what_if"):
        q["what_if"] = req["what_if"]
    r = analyze(q)
    if r["status"] not in ("ok", "partial"):
        return None
    return {"shape": list(shape), "approach": top, "request": q, "result": r}


def run(reqs, title=None, detail="compact"):
    """Analyze every request; the bundle dict (results, summary, rankings, caps)."""
    results = []
    for q in reqs:
        r = analyze(q)
        results.append({"request": q, "result": r, "example": _example(q, r, detail)})
    b = {"title": title or _default_title(results), "results": results}
    b["ops"] = _ops(results)
    b["grid"] = _grid(results)
    b["rankings"] = _rankings(results, b["grid"])
    b["summary"] = _summary(results, b["rankings"])
    b["caps"] = _caps(results, b["rankings"])
    return b


def _op(q):
    return (q.get("workload") or {}).get("op", "?")


def _prof(q, r=None):
    p = ((r or {}).get("profiles") or {}).get("costs") or {}
    return p.get("name") or (q.get("profiles") or {}).get("costs", "?")


def _default_title(results):
    ops = []
    for x in results:
        if _op(x["request"]) not in ops:
            ops.append(_op(x["request"]))
    return "qperf: " + ", ".join(ops)


def _ops(results):
    """Ops in first-seen order (each op keeps its color slot on the page)."""
    out = []
    for x in results:
        if _op(x["request"]) not in out:
            out.append(_op(x["request"]))
    return out


def _grid(results):
    """The reference grid: the one most survey requests use (default when none)."""
    grids = {}
    for x in results:
        sv = x["result"].get("survey")
        if sv:
            key = (tuple(sv["heights"]), tuple(sv["widths"]))
            grids[key] = grids.get(key, 0) + 1
    hs, ws = max(grids, key=grids.get) if grids else (tuple(DEFAULT_HEIGHTS), tuple(DEFAULT_WIDTHS))
    default = list(hs) == DEFAULT_HEIGHTS and list(ws) == DEFAULT_WIDTHS
    return {"heights": list(hs), "widths": list(ws), "shapes": len(hs) * len(ws), "default": default,
            "text": f"heights {hs[0]}..{hs[-1]} ({len(hs)}) x widths {ws[0]}..{ws[-1]} ({len(ws)}) = "
                    f"{len(hs) * len(ws)} shapes" + (" (the default survey grid)" if default else "")}


# ------------------------------------------------------------------------------------------------- rankings

def _wkey(q):
    """The workload a survey row belongs to, without the profile: a silicon-hypothesis run of the same request
    matches its ttsim-perf row."""
    w = {k: v for k, v in (q.get("workload") or {}).items() if k != "shape"}
    return json.dumps([w, q.get("baseline") or {}, (q.get("survey") or {}).get("heights"),
                       (q.get("survey") or {}).get("widths")], sort_keys=True)


def dram_sided(q, overrides=None):
    """Does the request (with a candidate's overrides) move data to or from DRAM? An L1-only option cannot be capped
    by DRAM bandwidth."""
    w = q.get("workload") or {}
    ov = overrides or {}
    op = w.get("op")
    if op == "reshard":
        r = {**(w.get("reshard") or DEFAULT_RESHARD), **(ov.get("reshard") or {})}
        return any((r.get(k) or {}).get("memory", "l1") == "dram" for k in ("from", "to"))
    pl = w.get("placements") or {}
    if op in ("add", "multiply"):
        return "dram" in pl.values()
    return (ov.get("input") or (q.get("baseline") or {}).get("input") or pl.get("a", "dram")) == "dram"


def _n(counts):
    return counts["faster"] + counts["neutral"] + counts["slower"]


def _survey_rows(x, grid, idx):
    """Ranking rows of one survey result: per approach, on RANK_PROFILE when the result has it (own profile or
    compare_profile), else on the result's profile."""
    q, r = x["request"], x["result"]
    sv = r["survey"]
    me = _prof(q, r)
    cmp = sv.get("compare")
    use_cmp = me != RANK_PROFILE and cmp and cmp["profile"]["name"] == RANK_PROFILE
    on_grid = sv["heights"] == grid["heights"] and sv["widths"] == grid["widths"]
    rows, omitted = [], []
    for aid, a in sv["approaches"].items():
        kind = a.get("kind") or kind_of(_op(q))
        if kind == "kernel_change" and not a["hypothetical"]:
            omitted.append(aid)  # the factory's own plan at every shape: not a change
            continue
        src = cmp["approaches"][aid] if use_cmp else a
        rows.append({
            "op": _op(q), "kind": kind, "id": aid, "change": describe(a["overrides"]) or "no overrides",
            "hypothetical": a["hypothetical"], "factory": a.get("factory"),
            "geomean": src["speedup"]["geomean"], "best": src.get("best"), "worst": src.get("worst"),
            "regressions": src["counts"]["slower"], "evaluated": _n(src["counts"]), "invalid": src["counts"]["invalid"],
            "shapes": len(sv["heights"]) * len(sv["widths"]), "on_grid": on_grid,
            "grid": "reference grid" if on_grid else f"{len(sv['heights'])}x{len(sv['widths'])} grid",
            "profile": cmp["profile"]["name"] if use_cmp else me, "untimed_cells": src.get("untimed_cells", 0),
            "dram_capped": a.get("dram_capped") if not use_cmp else src.get("dram_capped"),
            "wkey": _wkey(q), "source": "survey", "req": idx,
            "dram_side": dram_sided(q) or dram_sided(q, a["overrides"])})
    return rows, omitted


def _point_rows(x, idx):
    """Ranking rows of a compare / tune result at one shape: hypothetical setups (kernel changes) and, for the
    sharding ops and reshard, the candidates (layout choices)."""
    q, r = x["request"], x["result"]
    op = _op(q)
    shape = (q.get("workload") or {}).get("shape") or []
    H, W = (math.prod(shape[:-1]), shape[-1]) if len(shape) >= 2 else (None, None)
    rows = []
    for c in list(r.get("candidates") or []) + list(r.get("top") or []):
        if not c.get("valid") or not c.get("comparison"):
            continue
        hyp = bool((c.get("config") or {}).get("hypothetical"))
        kind = "kernel_change" if hyp else ("layout_choice" if kind_of(op) == "layout_choice" else None)
        if kind is None or c.get("label") == (r.get("baseline") or {}).get("label"):
            continue
        s = c["comparison"]["speedup_vs_baseline"]["nominal"]
        ov = next((k.get("overrides") for k in q.get("candidates") or [] if k.get("id") == c.get("id")), None)
        pt = {"shape": [H, W], "speedup": s} if H else None
        rows.append({"op": op, "kind": kind, "id": c.get("id") or c.get("label"),
                     "change": describe(ov) if ov else c.get("label", ""), "hypothetical": hyp,
                     "factory": (c.get("config") or {}).get("factory"), "geomean": s, "best": pt, "worst": pt,
                     "regressions": int(s < 0.98), "evaluated": 1, "invalid": 0, "shapes": 1, "on_grid": False,
                     "grid": f"1 shape ({shape_label(H, W)})" if H else "1 shape", "profile": _prof(q, r),
                     "untimed_cells": int(c["comparison"].get("verdict") == "inconclusive"),
                     "dram_capped": int((c.get("bottleneck") or {}).get("resource") == "dram_global")
                     if (r.get("baseline") or {}).get("bounds_ns") else None,
                     "wkey": None, "source": q.get("action"), "req": idx,
                     "dram_side": dram_sided(q) or dram_sided(q, ov)})
    return rows


def _has_dram_bound(r):
    sv = r.get("survey") or {}
    return "dram_capped" in (sv.get("baseline") or {})


def _rankings(results, grid):
    """{"kernel_changes": [...], "layout_choices": [...], "omitted": {...}, "profile", "grid"}: rows ranked by
    RANK_PROFILE geomean on the reference grid first; a silicon-hypothesis survey of the same workload fills a row's
    DRAM columns instead of adding rows."""
    rows, omitted, dram_runs = [], {}, {}
    for idx, x in enumerate(results):
        r = x["result"]
        if r["status"] not in ("ok", "partial"):
            continue
        if r.get("survey"):
            if _has_dram_bound(r):
                dram_runs[(_op(x["request"]), _wkey(x["request"]))] = x
            got, om = _survey_rows(x, grid, idx)
            rows += got
            if om:
                omitted.setdefault(_op(x["request"]), []).extend(om)
        else:
            rows += _point_rows(x, idx)
    # attach silicon-hypothesis surveys to the matching rows of another profile, drop their own rows there
    keep = []
    for row in rows:
        dx = dram_runs.get((row["op"], row["wkey"])) if row["wkey"] else None
        if dx and row["profile"] != _prof(dx["request"], dx["result"]):
            a = dx["result"]["survey"]["approaches"].get(row["id"])
            if a:
                row["dram"] = {"profile": _prof(dx["request"], dx["result"]), "geomean": a["speedup"]["geomean"],
                               "capped": a.get("dram_capped", 0), "evaluated": _n(a["counts"]),
                               "baseline_capped": dx["result"]["survey"]["baseline"].get("dram_capped", 0)}
        keep.append(row)
    matched = {(r["op"], r["wkey"]) for r in keep if r.get("dram")}
    keep = [r for r in keep if not (r["wkey"] and (r["op"], r["wkey"]) in matched and
                                    (r["op"], r["wkey"]) in dram_runs and
                                    r["profile"] == _prof(dram_runs[(r["op"], r["wkey"])]["request"],
                                                          dram_runs[(r["op"], r["wkey"])]["result"]))]
    for r in keep:
        if "dram" not in r and r.get("dram_capped") is not None:  # a row of the DRAM-bounded profile itself
            r["dram"] = {"profile": r["profile"], "geomean": r["geomean"], "capped": r["dram_capped"],
                         "evaluated": r["evaluated"], "baseline_capped": None}
        r.pop("wkey", None)
        r.pop("dram_capped", None)

    def order(r):
        timed = not (r["untimed_cells"] and r["untimed_cells"] >= r["evaluated"])
        return (not r["on_grid"], r["profile"] != RANK_PROFILE, not timed, -(r["geomean"] or 0))

    out = {"profile": RANK_PROFILE, "grid": grid["text"], "omitted": omitted}
    for kind, key in (("kernel_change", "kernel_changes"), ("layout_choice", "layout_choices")):
        out[key] = sorted((r for r in keep if r["kind"] == kind), key=order)
        for i, r in enumerate(out[key]):
            r["rank"] = i + 1
    return out


# ------------------------------------------------------------------------------------------------- summary, caps

def _limiters_txt(lim, n):
    return ", ".join(f"{k} on {v}/{n}" for k, v in sorted(lim.items(), key=lambda kv: -kv[1]))


def _primary(results, op):
    """The result that speaks for an op: a survey on RANK_PROFILE, else any survey, else the first result."""
    xs = [x for x in results if _op(x["request"]) == op and x["result"]["status"] in ("ok", "partial")]
    for test in (lambda x: x["result"].get("survey") and _prof(x["request"], x["result"]) == RANK_PROFILE,
                 lambda x: x["result"].get("survey") and not _has_dram_bound(x["result"]),
                 lambda x: x["result"].get("survey"), lambda x: True):
        hit = [x for x in xs if test(x)]
        if hit:
            return hit[0]
    return None


def _summary(results, rankings):
    """One line per op."""
    out = []
    for op in _ops(results):
        x = _primary(results, op)
        if x is None:
            bad = next(y for y in results if _op(y["request"]) == op)["result"]
            out.append({"op": op, "text": f"{op}: {bad['status']}: {bad.get('message', '')}"})
            continue
        q, r = x["request"], x["result"]
        prof = _prof(q, r)
        sv = r.get("survey")
        if sv:
            base = sv.get("baseline") or {}
            n = sum((base.get("limiters") or {}).values())
            today = (f"today {base.get('factory') or '?'} factory" + (" (also " + ", ".join(
                f for f in base.get("factories", {}) if f != base.get("factory")) + ")"
                if len(base.get("factories", {})) > 1 else "") +
                     (f", limited by {_limiters_txt(base['limiters'], n)} shapes" if base.get("limiters") else ""))
            if (kind_of(op) == "layout_choice") and q["workload"].get("op") in SHARD_OPS:
                sh = q["workload"].get("shard") or DEFAULT_SHARD
                today += f" (baseline {sh['strategy']} x{sh['cores']}, interleaved side " \
                         f"{(q['workload'].get('placements') or {}).get('a', 'dram')})"
            elif op == "reshard":
                today += f" (baseline {describe({'reshard': q['workload'].get('reshard') or DEFAULT_RESHARD})})"
            key = "kernel_changes" if kind_of(op) == "kernel_change" else "layout_choices"
            mine = [row for row in rankings[key] if row["req"] == results.index(x)]
            timed = [row for row in mine if not row["untimed_cells"]]
            word = "change" if key == "kernel_changes" else "choice"
            if timed:
                b = max(timed, key=lambda row: row["geomean"] or 0)
                best = (f"best {word} {b['id']} {b['geomean']:.2f}x geomean"
                        + (f" (up to {b['best']['speedup']:.2f}x at {shape_label(*b['best']['shape'])})" if b["best"] else ""))
            elif mine:
                best = f"best {word}: none timed (every option has an untimed side on {prof}: not meaningful)"
            else:
                best = f"no {word} modeled"
            regs = [(row["worst"]["speedup"], row) for row in mine if row["worst"] and row["worst"]["speedup"] < 0.98
                    and not row["untimed_cells"]]
            if regs:
                s, w = min(regs, key=lambda t: t[0])
                worst = f"worst regression {w['id']} {s:.2f}x at {shape_label(*w['worst']['shape'])}"
            else:
                worst = "no regression" if timed else "regressions not timed"
            text = f"{op} ({prof}): {today}; {best}; {worst}"
            dx = next((y for y in results if _op(y["request"]) == op and y["result"].get("survey")
                       and _has_dram_bound(y["result"])), None)
            if dx:
                dsv = dx["result"]["survey"]
                text += (f"; {_prof(dx['request'], dx['result'])}: the DRAM bound caps the baseline at "
                         f"{dsv['baseline']['dram_capped']}/{dsv['stats']['evaluated']} shapes")
        else:
            b = r.get("baseline") or {}
            text = (f"{op} ({prof}, {q.get('action')}): {b.get('label', '')}, limited by "
                    f"{(b.get('bottleneck') or {}).get('resource', '?')}")
            from .report import _best

            best = _best(r)
            if best:
                text += f"; best alternative {best[1]} {best[0]:.2f}x ({best[2]['comparison']['verdict']})"
        out.append({"op": op, "text": text})
    return out


def _caps(results, rankings):
    """What the assumed DRAM bandwidth does to the top rows (when a request ran a DRAM-bounded profile), else a note
    that DRAM is untimed."""
    ran = [x for x in results if x["result"].get("survey") and _has_dram_bound(x["result"])]
    point = [x for x in results if (x["result"].get("baseline") or {}).get("bounds_ns", {}).get("dram_global")]
    items = []
    for key in ("kernel_changes", "layout_choices"):
        top = [r for r in rankings[key] if r.get("dram")][:3]
        for r in top:
            d = r["dram"]
            if d["profile"] == r["profile"]:
                items.append(f"{r['id']} ({r['op']}): {r['geomean']:.2f}x on {d['profile']}, DRAM bound sets its time at "
                             f"{d['capped']}/{d['evaluated']} shapes")
            else:
                items.append(f"{r['id']} ({r['op']}): {r['geomean']:.2f}x on {r['profile']} -> "
                             f"{d['geomean']:.2f}x on {d['profile']} (DRAM bound sets its time at {d['capped']}/"
                             f"{d['evaluated']} shapes; the baseline's at {d['baseline_capped']})")
    if ran or point:
        names = sorted({_prof(x["request"], x["result"]) for x in ran + point})
        done = {_op(x["request"]) for x in ran + point}
        not_run = [op for op in _ops(results) if op not in done and
                   any(dram_sided(x["request"]) for x in results if _op(x["request"]) == op)]
        note = (f"{', '.join(names)} assumes a DRAM bandwidth (a hypothesis, no silicon): a DRAM-bound shape gains "
                f"less from any change")
        if not_run:
            note += f"; not run there: {', '.join(not_run)} (their DRAM side stays untimed)"
        return {"dram_profile_ran": True, "items": items, "note": note}
    return {"dram_profile_ran": False, "items": [],
            "note": f"DRAM is untimed: ttsim-perf times L1 NoC transfers, not DRAM, and craq-sim times neither; no "
                    f"request ran {DRAM_PROFILE_HINT}, so these rankings ignore DRAM bandwidth. Large shapes that are "
                    f"DRAM-bound on silicon gain less; add a {DRAM_PROFILE_HINT} survey to see the cap."}


# ------------------------------------------------------------------------------------------------- formatting

def _sp(x):
    return "-" if x is None else f"{x:.2f}x"


def _pt(p):
    return f"{_sp(p['speedup'])} @ {shape_label(*p['shape'])}" if p and p.get("shape") and p["shape"][0] else (
        _sp(p["speedup"]) if p else "-")


def _dram_txt(r):
    d = r.get("dram")
    if not d:
        return "not run" if r.get("dram_side", True) else "no DRAM side"
    if d["profile"] == r["profile"]:
        return f"{d['capped']}/{d['evaluated']} capped"
    return f"{d['capped']}/{d['evaluated']} capped, {_sp(d['geomean'])}"


def _geo_cell(r):
    g = _sp(r["geomean"])
    if r["untimed_cells"] and r["untimed_cells"] >= r["evaluated"]:
        return g + " (untimed)"
    if r["untimed_cells"]:
        return g + f" ({r['untimed_cells']} untimed)"
    return g


HEAD = ("#", "op", "change", "geomean", "best shape", "worst shape", "regressions", "DRAM-capped?", "profile", "shapes")


def _cells(r):
    shapes = f"{r['evaluated']}/{r['shapes']}" + ("" if r["on_grid"] else f" ({r['grid']})")
    hyp = " (hypothetical)" if r["hypothetical"] and r["kind"] != "kernel_change" else ""
    return (str(r["rank"]), r["op"], f"{r['id']}: {r['change']}{hyp}",
            _geo_cell(r), _pt(r["best"]), _pt(r["worst"]), str(r["regressions"]), _dram_txt(r), r["profile"], shapes)


TABLES = (("kernel_changes", "Kernel changes (hypothetical)",
           "changes to the factory's kernels (survey approaches / setups marked hypothetical: not in the pinned "
           "kernels, timed with the factory's calibrated costs)"),
          ("layout_choices", "Layout choices (change what the next op receives)",
           "shard / reshard options: the pinned factories with another shard layout; they change what the op and the "
           "next op receive, so rank them against each other, not against kernel changes"))


def _basis(b):
    return (f"Ranked by {b['rankings']['profile']} geomean speedup vs each op's baseline over the same grid: "
            f"{b['grid']['text']}; geomean over each row's evaluated shapes (shapes column: evaluated / grid; the rest "
            f"invalid for the option or skipped for the baseline). Rows on another grid or a single shape come last.")


def units_note(b):
    """One line when the bundle mixes simulator cycles (craq-sim, ttsim-perf) with real ns (silicon-hypothesis), else
    None."""
    by = {}
    for x in b["results"]:
        u = x["result"].get("time_unit")
        if u:
            by.setdefault(u, set()).add(_prof(x["request"], x["result"]))
    if len(by) < 2:
        return None
    cyc, ns = sorted(by["cycles"]), sorted(by["ns"])
    return (f"Times: {', '.join(cyc)} report{'s' if len(cyc) == 1 else ''} simulator cycles (cyc; 1 cycle = 1 ns by "
            f"convention), {', '.join(ns)} real ns at {'its' if len(ns) == 1 else 'their'} assumed clock; do not compare "
            f"times across them (speedups are unitless).")


def _section_title(x, i):
    q, r = x["request"], x["result"]
    return f"{i}. {_op(q)} {q.get('action', 'estimate')} on {_prof(q, r)}"


def to_text(b):
    from .cli import _text

    L = [f"# {b['title']}", f"{len(b['results'])} requests; " + _basis(b)] + [n for n in [units_note(b)] if n]
    L += ["", "Summary by op"]
    L += [f"  {s['text']}" for s in b["summary"]]
    for key, name, what in TABLES:
        rows = b["rankings"][key]
        L += ["", f"{name}: {what}"]
        if not rows:
            L.append("  none in these requests")
            continue
        cells = [HEAD] + [_cells(r) for r in rows]
        widths = [max(len(c[k]) for c in cells) for k in range(len(HEAD))]
        widths[2] = min(widths[2], 64)
        for c in cells:
            L.append("  " + "  ".join((c[k][:widths[k]] if k == 2 else c[k]).ljust(widths[k]) for k in range(len(HEAD))).rstrip())
    om = b["rankings"]["omitted"]
    if om:
        L.append("  not ranked (the factory's own plan at every shape): " +
                 "; ".join(f"{op}: {', '.join(v)}" for op, v in om.items()))
    L += ["", "Caps: " + b["caps"]["note"]] + [f"  {t}" for t in b["caps"]["items"]]
    for i, x in enumerate(b["results"], 1):
        L += ["", f"== {_section_title(x, i)} =="]
        ex = x.get("example")
        if ex:
            from .report import _how_text, _plan, _shown

            L.append(f"How it runs at {shape_label(*ex['shape'])}:")
            L += [f"  {rid}: {_how_text(_plan(ex['request'], r))}" for rid, r in _shown(ex["result"])]
        L.append(_text(x["result"]))
    return "\n".join(L)


def to_markdown(b):
    from .report import _how_text, _plan, _shown
    from .report import to_markdown as md1

    L = [f"# {b['title']}", "", f"{len(b['results'])} requests. {_basis(b)}", ""]
    L += [units_note(b), ""] if units_note(b) else []
    L += ["## Summary by op", ""]
    L += [f"- {s['text']}" for s in b["summary"]]
    for key, name, what in TABLES:
        rows = b["rankings"][key]
        L += ["", f"## {name}", "", f"{what[0].upper() + what[1:]}.", ""]
        if not rows:
            L.append("None in these requests.")
            continue
        L += ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
        L += ["| " + " | ".join(c.replace("|", "/") for c in _cells(r)) + " |" for r in rows]
    om = b["rankings"]["omitted"]
    if om:
        L += ["", "Not ranked (the factory's own plan at every shape): " +
              "; ".join(f"{op}: {', '.join(v)}" for op, v in om.items()) + "."]
    L += ["", "## Caps", "", b["caps"]["note"] + ("" if not b["caps"]["items"] else ":"), ""]
    L += [f"- {t}" for t in b["caps"]["items"]]
    for i, x in enumerate(b["results"], 1):
        L += ["", f"## {_section_title(x, i)}", ""]
        if x["result"].get("accuracy"):
            L += [f"**Accuracy:** {x['result']['accuracy']}", ""]
        ex = x.get("example")
        if ex:
            L += [f"### How it runs at {shape_label(*ex['shape'])}", ""]
            L += [f"- **{rid}**: {_how_text(_plan(ex['request'], r))}" for rid, r in _shown(ex["result"])] + [""]
        body = md1(x["request"], x["result"]).split("\n")
        for ln in body[1:]:  # the request's own report, headings one level down
            L.append("#" + ln if ln.startswith("#") else ln)
    return "\n".join(L)


BUNDLE_CSS = """
:root{--o1:#6f4fd8;--o2:#0e8c96;--o3:#b0602a;--o4:#55741f;--o5:#c0397a;--o6:#50637a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--o1:#9a80f0;--o2:#22aab4;--o3:#d07a40;
--o4:#8aab45;--o5:#e2629b;--o6:#8ea2b8}}
:root[data-theme="dark"]{--o1:#9a80f0;--o2:#22aab4;--o3:#d07a40;--o4:#8aab45;--o5:#e2629b;--o6:#8ea2b8}
.opsw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:0}
ul.sum{margin:0;padding-left:0;list-style:none}ul.sum li{margin:6px 0;padding-left:16px;text-indent:-16px}
table.rank td:nth-child(3){min-width:220px}table.rank td.num{text-align:right;white-space:nowrap}
.caps{border-left:3px solid var(--s4)}details.req{background:var(--surface);border:1px solid var(--ring);
border-radius:10px;margin:10px 0;padding:0 16px}details.req>summary{cursor:pointer;padding:12px 0;list-style:none}
details.req>summary::-webkit-details-marker{display:none}details.req>summary h2{display:inline;margin:0;font-size:16px}
details.req>summary .note{display:block;margin-top:2px}details.req[open]{padding-bottom:12px}
details.req>summary::before{content:"\\25B8";display:inline-block;width:16px;color:var(--ink2)}
details.req[open]>summary::before{content:"\\25BE"}
details.req h2{font-size:15px}details.req .card{background:var(--page)}
.muted{color:var(--muted)}.mini{display:inline-block;position:relative;width:64px;height:10px;vertical-align:-1px;
margin-right:6px;background:linear-gradient(var(--track),var(--track)) 0 4px/100% 2px no-repeat}
.mini::after{content:"";position:absolute;left:31.5px;top:0;width:1px;height:10px;background:var(--ink2)}
.mini b{position:absolute;top:1px;height:8px;border-radius:2px}.mini .w{background:var(--win)}
.mini .l{background:var(--loss)}.mini .n{background:var(--axis)}.mini .u{background:var(--deemph)}
"""


def _mini_bar(r):
    """A small bar of the geomean speedup on a log scale centred at 1x (0.25x .. 4x): blue faster, red slower, gray
    when untimed."""
    g = r["geomean"]
    if not g:
        return ""
    mid = 32
    x = max(-1.0, min(1.0, math.log2(g) / 2)) * mid
    col = "u" if r["untimed_cells"] else "w" if g > 1.02 else "l" if g < 0.98 else "n"
    x0, bw = (mid, max(x, 1.5)) if x >= 0 else (mid + x, -x)
    return f'<i class="mini" aria-hidden="true"><b class="{col}" style="left:{x0:.1f}px;width:{bw:.1f}px"></b></i>'


def _op_col(b, op):
    k = b["ops"].index(op) if op in b["ops"] else None
    return f"var(--o{k % 6 + 1})" if k is not None else "var(--deemph)"


def _dot(b, op):
    return f'<i class="opsw" style="background:{_op_col(b, op)}"></i>'


def to_html(b):
    from .report import CSS, _how_html, _slots, bin_css, body_html, defs

    profs = sorted({_prof(x["request"], x["result"]) for x in b["results"]})
    parts = [f'<h1>{E(b["title"])}</h1><p class="sub">{len(b["results"])} requests · profiles '
             f'{E(", ".join(profs))} · grid {E(b["grid"]["text"])}</p>']
    note = units_note(b)
    if note:
        parts.append(f'<p class="note units">{E(note)}</p>')
    parts.append('<div class="card"><h2 style="margin-top:0">Summary by op</h2><ul class="sum">')
    parts += [f'<li>{_dot(b, s["op"])}{E(s["text"])}</li>' for s in b["summary"]]
    parts.append("</ul></div>")
    for key, name, what in TABLES:
        rows = b["rankings"][key]
        parts.append(f'<h2>{E(name)}</h2><p class="note">{E(what[0].upper() + what[1:])}. {E(_basis(b))}</p>')
        if not rows:
            parts.append('<div class="card note">None in these requests.</div>')
            continue
        body = []
        for r in rows:
            c = _cells(r)
            tags = ('<span class="tag">hypothetical</span>' if r["hypothetical"] and key != "kernel_changes" else "") + \
                   ('<span class="tag">untimed</span>' if r["untimed_cells"] else "")
            dim = ' class="muted"' if (not r["on_grid"] or r["profile"] != b["rankings"]["profile"] or
                                      (r["untimed_cells"] and r["untimed_cells"] >= r["evaluated"])) else ""
            body.append(f'<tr{dim} data-op="{E(r["op"])}" data-id="{E(r["id"])}"><td class="num">{c[0]}</td>'
                        f'<td>{_dot(b, r["op"])}{E(r["op"])}</td><td><b>{E(r["id"])}</b> {tags}<br>'
                        f'<span class="note">{E(r["change"])}</span></td>'
                        + f'<td class="num">{_mini_bar(r)}{E(c[3])}</td>'
                        + "".join(f'<td class="num">{E(v)}</td>' for v in c[4:7])
                        + f'<td>{E(c[7])}</td><td>{E(c[8])}</td><td class="num">{E(c[9])}</td></tr>')
        parts.append(f'<div class="card wrap"><table class="rank" data-table="{key}"><tr>'
                     + "".join(f"<th>{E(h)}</th>" for h in HEAD) + "</tr>" + "".join(body) + "</table></div>")
    om = b["rankings"]["omitted"]
    if om:
        parts.append('<p class="note">Not ranked (the factory\'s own plan at every shape): '
                     + E("; ".join(f"{op}: {', '.join(v)}" for op, v in om.items())) + ".</p>")
    caps = b["caps"]
    parts.append(f'<div class="card caps"><h2 style="margin-top:0">Caps</h2><p class="note" style="margin-top:0">'
                 f'{E(caps["note"])}</p>' + ('<ul class="kp">' + "".join(f"<li>{E(t)}</li>" for t in caps["items"])
                                             + "</ul>" if caps["items"] else "") + "</div>")
    parts.append(f'<h2>Requests</h2><p class="note">Each request\'s own report: accuracy, how it runs, heatmaps or '
                 f'charts, all numbers. Click to expand.</p>')
    for i, x in enumerate(b["results"], 1):
        q, r = x["request"], x["result"]
        op = _op(q)
        from .report import _workload

        acc = r.get("accuracy") or ""
        parts.append(f'<details class="req" data-op="{E(op)}" style="border-left:3px solid {_op_col(b, op)}">'
                     f'<summary><h2>{_dot(b, op)}{E(_section_title(x, i))}</h2>'
                     f'<span class="note">{E(_workload(q))}</span>'
                     + (f'<span class="note accuracy"><b>Accuracy:</b> {E(acc)}</span>' if acc else "")
                     + "</summary>")
        ex = x.get("example")
        if ex:
            how = _how_html(ex["request"], ex["result"], _slots(ex["result"]))
            label = (f"How it runs at {shape_label(*ex['shape'])}: baseline"
                     + (f" vs {ex['approach']}" if ex["approach"] else ""))
            parts.append(how.replace("<h2>How it runs</h2>", f"<h2>{E(label)}</h2>", 1))
        parts.append(body_html(q, r) + "</details>")
    parts.append("</main></body></html>")
    body = "".join(parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{E(b["title"])}</title>'
            f'<style>{CSS}{BUNDLE_CSS}{bin_css(body)}</style></head><body>{defs(body)}<main>' + body)


def to_json(b):
    return {k: b[k] for k in ("title", "grid", "summary", "rankings", "caps")} | {
        "results": [{"request": x["request"], "result": x["result"],
                     **({"example": {k: x["example"][k] for k in ("shape", "approach", "result")}}
                        if x.get("example") else {})} for x in b["results"]]}
