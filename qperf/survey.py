"""survey: bench approaches across a grid of shapes and say, per approach, where it wins and where it regresses.

An approach is a set of knob overrides applied unchanged at every shape (request candidates, or DEFAULT_APPROACHES:
shape-independent policies over the hypothetical layout design knobs of tilize Default / untilize MultiCore). Each
shape [H, W] (H = rows = prod(shape[:-1])) is timed for the baseline (the request's baseline knobs, default the
factory) and every approach; speedup = baseline time / approach time (nominal costs). An approach whose knobs resolve
to the baseline's plan at a shape is neutral there (speedup exactly 1.0); equal plans are timed once per shape.
Shapes the factory routes elsewhere (untilize wide -> Block, use_multicore false -> SingleCore) or that the profile's
simulator cannot run (craq-sim tilize wider than 64 tiles) are skipped with a reason.
"""

from __future__ import annotations

import math
import time

from . import accuracy
from .api import NOISE
from .layout_api import CRAQ_SMALL_REQUESTS, LayoutError, _costs, _result, _workload
from .ops.layout import DESIGN_FACTORIES, TILE, estimate, plan, resolve_design
from .profiles import load

DEFAULT_HEIGHTS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_WIDTHS = [64, 128, 256, 512, 1024, 2048]
# Shape-independent policies; "batch" is the factory's batched DM thread (tilize write_batch, untilize read_batch).
DEFAULT_APPROACHES = [
    ("cols_all", {"split": "cols", "cores": 32}),
    ("grid2d", {"split": "2d"}),
    ("batch4", {"batch": 4}),
    ("batch8", {"batch": 8}),
    ("dfb2", {"dfb_blocks": 2}),
    ("dfb2_batch8", {"dfb_blocks": 2, "batch": 8}),
    ("grid2d_dfb2_batch8", {"split": "2d", "dfb_blocks": 2, "batch": 8}),
    ("half_blocks", {"tiles_per_block": "half"}),
]
SURVEY_OPS = ("tilize", "untilize")
HYPOTHETICAL_WARNING = ("hypothetical layout designs (the approaches marked hypothetical): not in the pinned kernels; "
                        "timed with the factory's calibrated per-request costs")


def shape_label(h, w):
    return f"{h}x{w}"


def _region(cells, valid, heights, widths):
    """The set of shapes `cells` as one rectangle of the grid ('H<=128, W>=1024'), when it is exactly every valid
    shape in its bounding box and not the whole grid; else None. cells, valid: sets of (H, W)."""
    if not cells:
        return None
    hs, ws = sorted({h for h, _ in cells}), sorted({w for _, w in cells})
    box = {(h, w) for h in heights if hs[0] <= h <= hs[-1] for w in widths if ws[0] <= w <= ws[-1]} & valid
    if box != cells or cells == valid:
        return None

    def axis(name, lo, hi, values):
        if lo == values[0] and hi == values[-1]:
            return None
        if lo == hi:
            return f"{name}={lo}"
        if lo == values[0]:
            return f"{name}<={hi}"
        if hi == values[-1]:
            return f"{name}>={lo}"
        return f"{name} {lo}..{hi}"

    parts = [p for p in (axis("H", hs[0], hs[-1], heights), axis("W", ws[0], ws[-1], widths)) if p]
    return ", ".join(parts) or None


def _approaches(req, batch_knob):
    """[(id, overrides)] with the generic "batch" mapped to the factory's batch knob."""
    src = [(c.get("id"), c.get("overrides") or {}) for c in req.get("candidates") or []] or DEFAULT_APPROACHES
    out, seen = [], set()
    for aid, ov in src:
        ov = {(batch_knob if k == "batch" else k): v for k, v in ov.items()}
        aid = aid or " ".join(f"{k}={v}" for k, v in ov.items()) or "factory"
        if aid in seen or aid == "baseline":
            raise LayoutError("invalid", f"survey approach ids must be unique and not 'baseline' ({aid!r})")
        seen.add(aid)
        out.append((aid, ov))
    return out


def run_survey(req, base_knobs, profile, what_if, analysis, t_start, compare=True):
    """The survey part of the response: {"survey", "recommendations", "warnings", "status", "accuracy"}. With
    survey.compare_profile (and compare), the same survey on that profile goes to survey.compare."""
    w = req["workload"]
    op = w["op"]
    if op not in SURVEY_OPS:
        raise LayoutError("unsupported", f"survey compares approaches over the hypothetical layout design knobs of "
                                         f"tilize Default and untilize MultiCore across shapes; {op} has none. A binary_ng "
                                         f"survey over a num_tiles axis is not modeled yet (use sensitivity num_tiles).")
    spec = req.get("survey") or {}
    heights = [int(h) for h in spec.get("heights") or DEFAULT_HEIGHTS]
    widths = [int(x) for x in spec.get("widths") or DEFAULT_WIDTHS]
    batch_knob = DESIGN_FACTORIES[(op, "Default" if op == "tilize" else "MultiCore")]
    base_knobs = {(batch_knob if k == "batch" else k): v for k, v in base_knobs.items()}
    approaches = _approaches(req, batch_knob)
    if "time_budget_s" in analysis:
        budget_s = analysis["time_budget_s"]
    elif "time_budget_ms" in analysis:
        budget_s = analysis["time_budget_ms"] / 1e3
    else:
        budget_s = 30
    costs = _costs(profile, "nominal", what_if.get("cost_overrides"), what_if.get("compute_service_scale", 1.0))
    ghz = profile["clock"]["GHz"]
    t_eval = time.perf_counter()
    cache = {}
    n_eval = n_reused = 0

    def resolve(H, W, knobs):
        """(workload, plan signature, hypothetical); raises LayoutError / ValueError when the shape cannot take it."""
        wl = _workload(dict(w, shape=[H, W]), knobs)
        f = plan(wl["op"], wl["H"], wl["W"], wl["dtype"], wl["mem"], wl["use_multicore"])[0]
        d = resolve_design(op, f, H, W, wl["dtype"], wl["design"]) if wl.get("design") else None
        hyp = bool(d and d["hypothetical"])
        small = hyp and d["tiles_per_block"] < math.ceil(W / TILE)  # shorter stick reads / row writes
        return wl, (wl["mem"], wl["use_multicore"], f, d["sig"] if hyp else "factory"), (hyp, small), f

    def time_ns(wl, sig):
        nonlocal n_eval, n_reused
        if sig in cache:
            n_reused += 1
            return cache[sig]
        n_eval += 1
        cache[sig] = estimate(wl["op"], wl["H"], wl["W"], costs, wl["dtype"], wl["mem"], wl["use_multicore"],
                              design=wl.get("design"))["span"] / ghz
        return cache[sig]

    nH, nW = len(heights), len(widths)
    base_grid = [[None] * nW for _ in heights]
    grids = {aid: [[None] * nW for _ in heights] for aid, _ in approaches}
    hyp_any = {aid: False for aid, _ in approaches}
    shrinks = {aid: False for aid, _ in approaches}  # a plan with shorter NoC requests than the factory's (craq-sim)
    outside = [[False] * nW for _ in heights]  # beyond the factory's calibrated shape range on this profile
    rng = accuracy.calibrated_range(profile, op, factory_name(op))
    invalid = {aid: 0 for aid, _ in approaches}
    best = [[None] * nW for _ in heights]
    skipped, not_evaluated, base_warnings = [], [], None
    out_of_time = False
    for i, H in enumerate(heights):
        for j, W in enumerate(widths):
            if out_of_time or time.perf_counter() - t_start > budget_s:
                out_of_time = True
                not_evaluated.append([H, W])
                continue
            cache.clear()  # signatures are per shape
            try:
                bwl, bsig, _, factory = resolve(H, W, base_knobs)
            except (LayoutError, ValueError) as e:
                skipped.append({"shape": [H, W], "reason": f"baseline: {getattr(e, 'message', e)}"})
                continue
            if (op, factory) not in DESIGN_FACTORIES:
                skipped.append({"shape": [H, W], "reason": f"{op} routes to the {factory} factory here (no design knobs)"})
                continue
            if op == "tilize" and math.ceil(W / TILE) > 64 and profile["name"] == "craq-sim":
                skipped.append({"shape": [H, W], "reason": "craq-sim cannot run tilize wider than 64 tiles"})
                continue
            if base_warnings is None:  # profile checks (needs_input, fp32 not calibrated, known gaps) on one shape
                base_warnings = [x for x in _result(bwl, profile, what_if)["warnings"] if not x.startswith("hypothetical")]
            bt = time_ns(bwl, bsig)
            base_grid[i][j] = round(bt, 1)
            outside[i][j] = accuracy.outside(rng, H, W)
            row = []
            for aid, ov in approaches:
                try:
                    wl, sig, hyp, _ = resolve(H, W, dict(base_knobs, **ov))
                except (LayoutError, ValueError):
                    invalid[aid] += 1
                    continue
                s = 1.0 if sig == bsig else bt / time_ns(wl, sig)
                hyp_any[aid] |= hyp[0]
                shrinks[aid] |= hyp[1]
                grids[aid][i][j] = round(s, 3)
                row.append((round(s, 3), len(ov), aid))
            best[i][j] = _best_of(row)
    eval_ms = round((time.perf_counter() - t_eval) * 1e3, 1)
    skip_set = {tuple(s["shape"]) for s in skipped} | {tuple(s) for s in not_evaluated}
    evaluated = [(H, W) for H in heights for W in widths if (H, W) not in skip_set]

    out = {}
    for aid, ov in approaches:
        g = grids[aid]
        vals = [(g[i][j], H, W) for i, H in enumerate(heights) for j, W in enumerate(widths) if g[i][j] is not None]
        a = {"overrides": ov, "hypothetical": hyp_any[aid], "grid": g,
             "counts": {"faster": sum(s > 1 + NOISE for s, _, _ in vals), "slower": sum(s < 1 - NOISE for s, _, _ in vals),
                        "invalid": invalid[aid]}}
        a["counts"]["neutral"] = len(vals) - a["counts"]["faster"] - a["counts"]["slower"]
        if vals:
            lo, hi = min(vals, key=lambda v: v[0]), max(vals, key=lambda v: v[0])
            a["speedup"] = {"geomean": round(math.exp(sum(math.log(s) for s, _, _ in vals) / len(vals)), 3),
                            "min": lo[0], "max": hi[0]}
            a["worst"] = {"shape": [lo[1], lo[2]], "speedup": lo[0]}
            a["best"] = {"shape": [hi[1], hi[2]], "speedup": hi[0]}
        else:
            a["speedup"] = {"geomean": None, "min": None, "max": None}
            a["worst"] = a["best"] = None
        regs = sorted((v for v in vals if v[0] < 1 - NOISE), key=lambda v: v[0])
        a["regressions"] = [{"shape": [H, W], "speedup": s} for s, H, W in regs[:5]]
        valid = {(H, W) for _, H, W in vals}
        a["wins_where"] = _region({(H, W) for s, H, W in vals if s > 1 + NOISE}, valid, heights, widths)
        a["loses_where"] = _region({(H, W) for _, H, W in regs}, valid, heights, widths)
        out[aid] = a

    def line(aid, a):
        c, n = a["counts"], a["counts"]["faster"] + a["counts"]["neutral"] + a["counts"]["slower"]
        if not n:
            return f"{aid}: invalid at every evaluated shape"
        if c["faster"] == c["slower"] == 0 and a["speedup"]["min"] == a["speedup"]["max"] == 1.0:
            return f"{aid}: same plan as the baseline at every shape (neutral on {n}/{n})"
        s = (f"{aid}: {a['speedup']['geomean']:.2f}x geomean, faster on {c['faster']}/{n} shapes"
             + (f" ({a['wins_where']})" if a["wins_where"] else ""))
        if c["slower"]:
            s += (f", worst {a['worst']['speedup']:.2f}x at {shape_label(*a['worst']['shape'])}, slower on "
                  f"{c['slower']}/{n}" + (f" ({a['loses_where']})" if a["loses_where"] else ""))
        else:
            s += ", never slower"
        if c["invalid"]:
            s += f", invalid on {c['invalid']}"
        return s

    order = sorted(out, key=lambda k: -(out[k]["speedup"]["geomean"] or 0))
    summary = [line(k, out[k]) for k in order]
    recs = _recommend(out, order, analysis.get("top_k", 3))
    warnings = list(base_warnings or [])
    if any(hyp_any.values()):
        warnings.append(HYPOTHETICAL_WARNING)
    n_out = sum(outside[i][j] for i in range(nH) for j in range(nW))
    if n_out:
        note = (f"{n_out} of {len(evaluated)} evaluated shapes are outside the calibrated range for {op} "
                f"{factory_name(op)} on {accuracy.range_on(profile, rng)} ({accuracy.range_txt(rng)}): extrapolated "
                f"(survey.outside_calibrated_range; hatched in the report)")
        summary.append(note)
        warnings.append(note)
    small = [aid for aid, _ in approaches if shrinks[aid]]
    if small and profile["name"] == "craq-sim":
        warnings.append(CRAQ_SMALL_REQUESTS.format(ids=", ".join(small)))
    if skipped:
        reasons = sorted({s["reason"] for s in skipped})
        warnings.append(f"{len(skipped)} of {nH * nW} shapes skipped: " + "; ".join(reasons))
    if out_of_time:
        warnings.append(f"time budget ({budget_s:g} s) reached: {len(not_evaluated)} shapes not evaluated")
    if not evaluated and not out_of_time:
        raise LayoutError("unsupported", "survey: no shape can be evaluated: " +
                          "; ".join(sorted({s["reason"] for s in skipped})))
    fac, dtype = factory_name(op), w.get("dtype", "bf16")
    acc = accuracy.layout(profile, op, fac, dtype, False, bool(what_if.get("cost_overrides")))
    if any(hyp_any.values()) and not accuracy.source_of(profile)[1]:
        acc += "; approaches marked hypothetical: " + accuracy.hypothetical_note(profile, op, fac, dtype).removeprefix(
            "hypothetical setup: ")
    elif any(hyp_any.values()):  # a derived profile
        acc += "; the approaches marked hypothetical are not in the kernels either: no simulator measurement"
    res = {
        "status": "partial" if out_of_time else "ok",
        "warnings": warnings,
        "recommendations": recs,
        "accuracy": acc,
        "survey": {"heights": heights, "widths": widths, "skipped": skipped, "baseline_ns": base_grid,
                   "approaches": out, "best_per_shape": best, "summary": summary,
                   "outside_calibrated_range": outside, "calibrated_range": rng,
                   "stats": {"shapes": nH * nW, "evaluated": len(evaluated), "skipped": len(skipped),
                             "not_evaluated": not_evaluated, "plans_evaluated": n_eval, "plans_reused": n_reused,
                             "approaches": len(approaches), "out_of_time": out_of_time, "time_budget_s": budget_s,
                             "eval_ms": eval_ms, "outside_calibrated_range": n_out},
                   "speedup_basis": f"nominal costs; faster / slower beyond {NOISE:.0%} (the compare verdict "
                                    f"threshold), neutral within it"},
    }
    other = spec.get("compare_profile")
    if other and compare:
        res["survey"]["compare"] = cmp = _compare(req, base_knobs, other, what_if, analysis, t_start, res)
        if cmp["status"] == "partial":
            res["status"] = "partial"
        res["warnings"] += [f"compare_profile {other}: {x}" for x in cmp["warnings"]]
    return res


def factory_name(op):
    """The factory a survey times (the one with the design knobs)."""
    return "Default" if op == "tilize" else "MultiCore"


def _compare(req, base_knobs, name, what_if, analysis, t_start, primary):
    """survey.compare: the same survey (grid, approaches, baseline) on profile `name`; per approach its speedup
    geomean / min / max, extremes and counts there, next to the requested profile's."""
    prof, pid = load("costs", name)
    try:
        r = run_survey(req, base_knobs, prof, what_if, analysis, t_start, compare=False)
    except LayoutError as e:
        raise LayoutError(e.status, f"survey.compare_profile {name}: {e.message}")
    sv = r["survey"]
    return {"profile": pid, "status": r["status"], "platform_scope": prof.get("platform_scope", ""),
            "accuracy": r["accuracy"],
            "approaches": {aid: {k: a[k] for k in ("speedup", "counts", "worst", "best")}
                           for aid, a in sv["approaches"].items()},
            "summary": sv["summary"], "warnings": [x for x in r["warnings"] if x not in primary["warnings"]],
            "stats": {k: sv["stats"][k] for k in ("evaluated", "skipped", "plans_evaluated", "eval_ms",
                                                  "outside_calibrated_range")}}


def _best_of(row):
    """A shape's winner: "baseline" unless an approach is faster beyond NOISE; among the approaches within NOISE of
    the fastest, the one with the fewest overrides (then the faster), as tune prefers the simpler of tied plans."""
    if not row or max(row)[0] <= 1 + NOISE:
        return "baseline"
    top = max(row)[0]
    return min((n, -s, aid) for s, n, aid in row if top / s - 1 <= NOISE)[2]


def _recommend(out, order, top_k):
    """Approaches faster on most shapes and never slower (by geomean); else the best geomean with its regressions."""
    def label(aid):
        ov = out[aid]["overrides"]
        return f"{aid} ({', '.join(f'{k}={v}' for k, v in ov.items()) or 'no overrides'})"

    def n(a):
        return a["counts"]["faster"] + a["counts"]["neutral"] + a["counts"]["slower"]

    good = [k for k in order if n(out[k]) and out[k]["counts"]["faster"] > n(out[k]) / 2 and not out[k]["counts"]["slower"]]
    recs = []
    for k in good:
        a = out[k]
        recs.append({"id": k, "label": label(k), "speedup_nominal": a["speedup"]["geomean"],
                     "speedup_range": [a["speedup"]["min"], a["speedup"]["max"]],
                     "reason": f"faster on {a['counts']['faster']}/{n(a)} shapes, never slower; geomean "
                               f"{a['speedup']['geomean']:.2f}x" +
                               (f"; invalid on {a['counts']['invalid']}" if a["counts"]["invalid"] else "")})
    if recs:
        return recs[:top_k]
    k = next((k for k in order if (out[k]["speedup"]["geomean"] or 0) > 1 + NOISE), None)
    if k is None:
        return [{"id": None, "label": "keep baseline", "reason": f"no approach has a geomean speedup above "
                                                                 f"{1 + NOISE:.2f}x over the surveyed shapes"}]
    a = out[k]
    regs = ", ".join(f"{shape_label(*r['shape'])} ({r['speedup']:.2f}x)" for r in a["regressions"])
    return [{"id": k, "label": label(k), "speedup_nominal": a["speedup"]["geomean"],
             "speedup_range": [a["speedup"]["min"], a["speedup"]["max"]],
             "reason": f"best geomean {a['speedup']['geomean']:.2f}x, faster on {a['counts']['faster']}/{n(a)} shapes"
                       + (f" ({a['wins_where']})" if a["wins_where"] else "")
                       + (f", but slower on {a['counts']['slower']}" + (f" ({a['loses_where']})" if a["loses_where"] else "")
                          + f": {regs}" if regs else ", neutral on the rest, never slower")}]
