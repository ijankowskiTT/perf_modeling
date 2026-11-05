"""survey: bench approaches across a grid of shapes and say, per approach, where it wins and where it regresses.

Two kinds of approaches, never mixed in one survey:
- kernel changes (tilize, untilize; "kind": "kernel_change"): a set of knob overrides applied unchanged at every shape
  (request candidates, or DEFAULT_APPROACHES: shape-independent policies over the hypothetical layout design knobs of
  tilize Default / untilize MultiCore).
- layout choices (interleaved_to_sharded, sharded_to_interleaved, reshard; "kind": "layout_choice"): shard options
  (SHARD_CHOICES, RESHARD_CHOICES or candidates) that change what the op (and the next op) receives; they run the
  pinned factories, so they are never hypothetical. A shard option that would leave cores with empty TILE shards is
  invalid at that shape. A profile that does not time one side of an option (ttsim-perf: DRAM; craq-sim: reshard
  bytes) marks those cells untimed: their speedups are not meaningful.

Each shape [H, W] (H = rows = prod(shape[:-1])) is timed for the baseline (the request's baseline knobs, default the
factory / height x8) and every approach; speedup = baseline time / approach time (nominal costs). An approach whose
knobs resolve to the baseline's plan at a shape is neutral there (speedup exactly 1.0); equal plans are timed once per
shape. Shapes the factory routes elsewhere (untilize wide -> Block, use_multicore false -> SingleCore), that the
profile's simulator cannot run (craq-sim tilize wider than 64 tiles) or that the baseline cannot take are skipped with a
reason. Per shape the baseline's limiter (busiest role, or a global L1 / DRAM bound) and, on a profile with an assumed
DRAM bandwidth, where the DRAM bound caps a plan are recorded too.
"""

from __future__ import annotations

import math
import time

from . import accuracy
from .api import NOISE
from .layout_api import CRAQ_SMALL_REQUESTS, LayoutError, _costs, _result, _workload
from .ops.layout import DESIGN_FACTORIES, DESIGN_KNOBS, SHARD_OPS, TILE, estimate, plan, resolve_design
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
SURVEY_OPS = ("tilize", "untilize")  # kernel changes (hypothetical design knobs)
CHOICE_OPS = SHARD_OPS + ("reshard",)  # layout choices (shard options)
KINDS = {"kernel_change": "kernel changes (hypothetical layout designs of the factory's kernels)",
         "layout_choice": "layout choices (shard options: what the op, and the next op, receives)"}
HYPOTHETICAL_WARNING = ("hypothetical layout designs (the approaches marked hypothetical): not in the pinned kernels; "
                        "timed with the factory's calibrated per-request costs")
# interleaved_to_sharded / sharded_to_interleaved: the shard layout of the baseline when workload.shard is absent
DEFAULT_SHARD = {"strategy": "height", "cores": 8}
SHARD_CHOICES = [
    ("height_x16", {"shard_strategy": "height", "shard_cores": 16}),
    ("height_x32", {"shard_strategy": "height", "shard_cores": 32}),
    ("width_x16", {"shard_strategy": "width", "shard_cores": 16}),
    ("block_x16", {"shard_strategy": "block", "shard_cores": 16}),
    ("width_x8", {"shard_strategy": "width", "shard_cores": 8}),
]


def _rs(strategy, cores=None, grid=None, memory="l1"):
    s = {"strategy": strategy, "memory": memory}
    s.update({"grid": grid} if grid else {"cores": cores})
    return s


# reshard: the baseline when workload.reshard is absent (an L1 height x8 -> height x8 reshard), and its options
DEFAULT_RESHARD = {"from": _rs("height", 8), "to": _rs("height", 8)}
RESHARD_CHOICES = [
    ("h16_to_h16", {"reshard": {"from": _rs("height", 16), "to": _rs("height", 16)}}),
    ("h32_to_h32", {"reshard": {"from": _rs("height", 32), "to": _rs("height", 32)}}),
    ("h8_to_h16", {"reshard": {"from": _rs("height", 8), "to": _rs("height", 16)}}),
    ("h8_to_h32", {"reshard": {"from": _rs("height", 8), "to": _rs("height", 32)}}),
    ("h8_to_h4", {"reshard": {"from": _rs("height", 8), "to": _rs("height", 4)}}),
    ("w8_to_w8", {"reshard": {"from": _rs("width", 8), "to": _rs("width", 8)}}),
    ("b4x4_to_b4x4", {"reshard": {"from": _rs("block", grid=[4, 4]), "to": _rs("block", grid=[4, 4])}}),
]
SHARD_DRAM_NOTE = ("ttsim-perf does not time DRAM transfers: the interleaved DRAM side of {op} is nearly free here, so "
                   "its times and the speedups between shard options are not meaningful (interleaved side in L1 "
                   "compares on ttsim-perf; silicon-hypothesis assumes a DRAM bandwidth)")
UNTIMED_NOTE = ("{n} of {m} compared cells involve a side {profile} does not time: their speedups are not meaningful "
                "(survey grids mark them untimed; reports hatch them)")


def shape_label(h, w):
    return f"{h}x{w}"


def kind_of(op):
    """The survey kind of an op: kernel_change (tilize / untilize) or layout_choice (shard ops, reshard)."""
    return "kernel_change" if op in SURVEY_OPS else "layout_choice" if op in CHOICE_OPS else None


def _notes(op, profile_name, wl):
    """(warnings, untimed) of one plan: what the profile cannot time (reshard: ops/reshard.py notes; sharding ops:
    the interleaved DRAM side on ttsim-perf)."""
    if op == "reshard":
        from .ops import reshard

        return reshard.notes(profile_name, wl["mem"])
    if op in SHARD_OPS and profile_name == "ttsim-perf" and wl["placement"] == "dram":
        return [SHARD_DRAM_NOTE.format(op=op)], True
    return [], False


def _empty_shards(op, wl, H, W):
    """Why a shard option cannot hold this shape without empty TILE shards (fewer tile rows / columns than shards; the
    model splits TILE shards evenly), or None."""
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)

    def need(kind, n, gx=None, gy=None, what=""):
        if kind == "hs" and ht < n:
            return f"{what}height x{n} needs >= {n} tile rows (H >= {n * TILE})"
        if kind == "ws" and wt < n:
            return f"{what}width x{n} needs >= {n} tile columns (W >= {n * TILE})"
        if kind == "bs" and (ht < gy or wt < gx):
            return f"{what}block {gy}x{gx} needs >= {gy} tile rows and >= {gx} tile columns"
        return None

    if op in SHARD_OPS:
        mem = wl["mem"]
        if mem.endswith("r"):  # ROW_MAJOR: only cores with data run (shard_plan clips)
            return None
        n = int("".join(ch for ch in mem[2:] if ch.isdigit()))
        k = math.isqrt(n)
        why = need(mem[:2], n, k, k)
    elif op == "reshard":
        from .ops.reshard import parse_mem

        spec = parse_mem(wl["mem"], H, W)
        if spec["layout"] != "TILE":
            return None
        why = next((x for x in (need(spec[s]["kind"], spec[s]["n"], spec[s]["gx"], spec[s]["gy"], f"{s}: ")
                                for s in ("from", "to")) if x), None)
    else:
        return None
    return why and why + ": fewer would leave cores with empty shards (not modeled)"


def _limiter(e):
    """The limiter of one estimate() result, as layout_api._result names it: the DRAM or L1 bound when it sets the time,
    else the busiest role of the busiest core."""
    if e.get("dram_bound") and e["dram_bound"] >= e["span"] - 1e-9:
        return "dram_global"
    if e.get("l1_bound") and e["l1_bound"] >= e["span"] - 1e-9:
        return "noc_l1_global"
    run = max(e["loads"], key=lambda x: x[2]["makespan"])[2]
    return max(run["roles"], key=lambda k: run["roles"][k]["busy"])


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


def _approaches(req, batch_knob, defaults=DEFAULT_APPROACHES):
    """[(id, overrides)] with the generic "batch" mapped to the factory's batch knob."""
    src = [(c.get("id"), c.get("overrides") or {}) for c in req.get("candidates") or []] or defaults
    out, seen = [], set()
    for aid, ov in src:
        ov = {(batch_knob if k == "batch" and batch_knob else k): v for k, v in ov.items()}
        aid = aid or " ".join(f"{k}={v}" for k, v in ov.items()) or "factory"
        if aid in seen or aid == "baseline":
            raise LayoutError("invalid", f"survey approach ids must be unique and not 'baseline' ({aid!r})")
        seen.add(aid)
        out.append((aid, ov))
    return out


def _spec_txt(s):
    """A reshard side as text: 'height x8 L1', 'block 4x4 DRAM'."""
    n = "x".join(str(v) for v in s["grid"]) if s.get("grid") else s.get("cores", "?")
    return f"{s.get('strategy', '?')} {'' if s.get('grid') else 'x'}{n} {str(s.get('memory', 'l1')).upper()}"


def describe(ov):
    """Overrides as short text: 'k=v, ...'; a reshard spec as 'height x8 L1 -> height x16 L1', shard knobs as
    'width x16'."""
    ov = dict(ov)
    parts = []
    if isinstance(ov.get("reshard"), dict):
        r = ov.pop("reshard")
        sides = [_spec_txt(r[k]) if isinstance(r.get(k), dict) else "same" for k in ("from", "to")]
        parts.append(" -> ".join(sides) + (f", {r['layout']}" if r.get("layout") else ""))
    if "shard_strategy" in ov and "shard_cores" in ov:
        parts.append(f"{ov.pop('shard_strategy')} x{ov.pop('shard_cores')}")
    parts += [f"{k}={v}" for k, v in ov.items()]
    return ", ".join(parts)


def _top(counter):
    """The most frequent key of a {key: count} dict (ties: the first seen), or None."""
    return max(counter, key=counter.get) if counter else None


def run_survey(req, base_knobs, profile, what_if, analysis, t_start, compare=True):
    """The survey part of the response: {"survey", "recommendations", "warnings", "status", "accuracy"}. With
    survey.compare_profile (and compare), the same survey on that profile goes to survey.compare."""
    w = req["workload"]
    op = w["op"]
    kind = kind_of(op)
    if kind is None:
        raise LayoutError("unsupported", f"survey compares approaches across shapes: the hypothetical layout design "
                                         f"knobs of tilize Default and untilize MultiCore (kernel changes), or shard "
                                         f"options of interleaved_to_sharded, sharded_to_interleaved and reshard (layout "
                                         f"choices); {op} has none. A binary_ng survey over a num_tiles axis is not "
                                         f"modeled yet (use sensitivity num_tiles).")
    kernel = kind == "kernel_change"
    spec = req.get("survey") or {}
    heights = [int(h) for h in spec.get("heights") or DEFAULT_HEIGHTS]
    widths = [int(x) for x in spec.get("widths") or DEFAULT_WIDTHS]
    defaults_applied = []
    if kernel:
        batch_knob = DESIGN_FACTORIES[(op, "Default" if op == "tilize" else "MultiCore")]
        defaults = DEFAULT_APPROACHES
    else:
        batch_knob = None
        if op == "reshard":
            defaults = RESHARD_CHOICES
            if not w.get("reshard") and "reshard" not in base_knobs:
                w = dict(w, reshard=DEFAULT_RESHARD)
                defaults_applied.append("survey baseline reshard height x8 L1 -> height x8 L1 (default)")
        else:
            defaults = SHARD_CHOICES
            if not w.get("shard") and not ("shard_strategy" in base_knobs and "shard_cores" in base_knobs):
                w = dict(w, shard=DEFAULT_SHARD)
                defaults_applied.append("survey baseline shard height x8 (default)")
    base_knobs = {(batch_knob if k == "batch" and batch_knob else k): v for k, v in base_knobs.items()}
    approaches = _approaches(req, batch_knob, defaults)
    if "time_budget_s" in analysis:
        budget_s = analysis["time_budget_s"]
    elif "time_budget_ms" in analysis:
        budget_s = analysis["time_budget_ms"] / 1e3
    else:
        budget_s = 30
    costs = _costs(profile, "nominal", what_if.get("cost_overrides"), what_if.get("compute_service_scale", 1.0))
    ghz = profile["clock"]["GHz"]
    dram_bounded = bool(costs.get("dram.total_bytes_per_cycle")) and costs.get("dram.bound_layout", 0.0) >= 0.5
    t_eval = time.perf_counter()
    cache = {}
    n_eval = n_reused = 0
    notes = []  # the profile's notes on untimed sides, once each

    def resolve(H, W, knobs):
        """(workload, plan signature, (hypothetical, smaller requests), factory, untimed); raises LayoutError /
        ValueError when the shape cannot take it."""
        wl = _workload(dict(w, shape=[H, W]), knobs)
        f = plan(wl["op"], wl["H"], wl["W"], wl["dtype"], wl["mem"], wl["use_multicore"])[0]
        if not kernel:
            why = _empty_shards(op, wl, H, W)
            if why:
                raise LayoutError("invalid", why)
            warn, untimed = _notes(op, profile["name"], wl)
            notes.extend(x for x in warn if x not in notes)
            return wl, (wl["mem"], f), (False, False), f, untimed
        d = resolve_design(op, f, H, W, wl["dtype"], wl["design"]) if wl.get("design") else None
        hyp = bool(d and d["hypothetical"])
        small = hyp and d["tiles_per_block"] < math.ceil(W / TILE)  # shorter stick reads / row writes
        return wl, (wl["mem"], wl["use_multicore"], f, d["sig"] if hyp else "factory"), (hyp, small), f, False

    def time_ns(wl, sig):
        """(time ns, limiter, DRAM-capped) of one plan."""
        nonlocal n_eval, n_reused
        if sig in cache:
            n_reused += 1
            return cache[sig]
        n_eval += 1
        e = estimate(wl["op"], wl["H"], wl["W"], costs, wl["dtype"], wl["mem"], wl["use_multicore"],
                     design=wl.get("design"))
        cache[sig] = (e["span"] / ghz, _limiter(e), bool(e.get("dram_bound") and e["dram_bound"] >= e["span"] - 1e-9))
        return cache[sig]

    nH, nW = len(heights), len(widths)
    base_grid = [[None] * nW for _ in heights]
    grids = {aid: [[None] * nW for _ in heights] for aid, _ in approaches}
    untimed_grids = {aid: [[None] * nW for _ in heights] for aid, _ in approaches}
    hyp_any = {aid: False for aid, _ in approaches}
    shrinks = {aid: False for aid, _ in approaches}  # a plan with shorter NoC requests than the factory's (craq-sim)
    facs = {aid: {} for aid, _ in approaches}  # factory -> shapes, per approach
    capped = {aid: 0 for aid, _ in approaches}  # shapes where the assumed DRAM bandwidth sets the approach's time
    differs = {aid: False for aid, _ in approaches}  # a plan other than the baseline's at some shape
    base_info = {"factories": {}, "limiters": {}, "dram_capped": 0, "untimed": 0}
    outside = [[False] * nW for _ in heights]  # beyond the baseline factory's calibrated shape range on this profile
    ranges = {}
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
                bwl, bsig, _, factory, b_untimed = resolve(H, W, base_knobs)
            except (LayoutError, ValueError) as e:
                skipped.append({"shape": [H, W], "reason": f"baseline: {getattr(e, 'message', e)}"})
                continue
            if kernel and (op, factory) not in DESIGN_FACTORIES:
                skipped.append({"shape": [H, W], "reason": f"{op} routes to the {factory} factory here (no design knobs)"})
                continue
            if op == "tilize" and math.ceil(W / TILE) > 64 and profile["name"] == "craq-sim":
                skipped.append({"shape": [H, W], "reason": "craq-sim cannot run tilize wider than 64 tiles"})
                continue
            if base_warnings is None:  # profile checks (needs_input, fp32 not calibrated, known gaps) on one shape
                base_warnings = [x for x in _result(bwl, profile, what_if)["warnings"] if not x.startswith("hypothetical")]
            bt, neck, bcap = time_ns(bwl, bsig)
            base_grid[i][j] = round(bt, 1)
            base_info["factories"][factory] = base_info["factories"].get(factory, 0) + 1
            base_info["limiters"][neck] = base_info["limiters"].get(neck, 0) + 1
            base_info["dram_capped"] += bcap
            base_info["untimed"] += b_untimed
            if factory not in ranges:
                ranges[factory] = accuracy.calibrated_range(profile, op, factory)
            outside[i][j] = accuracy.outside(ranges[factory], H, W)
            row = []
            for aid, ov in approaches:
                try:
                    wl, sig, hyp, f, untimed = resolve(H, W, dict(base_knobs, **ov))
                except (LayoutError, ValueError):
                    invalid[aid] += 1
                    continue
                if sig == bsig:
                    s, cap = 1.0, bcap
                else:
                    t, _, cap = time_ns(wl, sig)
                    s = bt / t
                    differs[aid] = True
                hyp_any[aid] |= hyp[0]
                shrinks[aid] |= hyp[1]
                facs[aid][f] = facs[aid].get(f, 0) + 1
                capped[aid] += cap
                grids[aid][i][j] = round(s, 3)
                untimed_grids[aid][i][j] = untimed or b_untimed
                row.append((round(s, 3), len(ov), aid))
            best[i][j] = _best_of(row)
    eval_ms = round((time.perf_counter() - t_eval) * 1e3, 1)
    skip_set = {tuple(s["shape"]) for s in skipped} | {tuple(s) for s in not_evaluated}
    evaluated = [(H, W) for H in heights for W in widths if (H, W) not in skip_set]
    base_fac = _top(base_info["factories"]) or (factory_name(op) if kernel else None)
    rng = ranges.get(base_fac) if base_fac in ranges else accuracy.calibrated_range(profile, op, base_fac or "")

    out = {}
    for aid, ov in approaches:
        g = grids[aid]
        vals = [(g[i][j], H, W) for i, H in enumerate(heights) for j, W in enumerate(widths) if g[i][j] is not None]
        a = {"overrides": ov, "kind": kind, "hypothetical": hyp_any[aid], "grid": g,
             "factory": _top(facs[aid]) or base_fac,
             "counts": {"faster": sum(s > 1 + NOISE for s, _, _ in vals), "slower": sum(s < 1 - NOISE for s, _, _ in vals),
                        "invalid": invalid[aid]}}
        a["counts"]["neutral"] = len(vals) - a["counts"]["faster"] - a["counts"]["slower"]
        if not kernel:
            a["untimed"] = untimed_grids[aid]
            a["untimed_cells"] = sum(bool(x) for row in untimed_grids[aid] for x in row)
            if len(facs[aid]) > 1:
                a["factories"] = facs[aid]
        if dram_bounded:
            a["dram_capped"] = capped[aid]
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
            if differs[aid]:
                return f"{aid}: another plan, same time as the baseline at every shape (neutral on {n}/{n})"
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
        if a.get("untimed_cells"):
            s += f"; untimed on {a['untimed_cells']}/{n}: not meaningful on {profile['name']}"
        return s

    order = sorted(out, key=lambda k: -(out[k]["speedup"]["geomean"] or 0))
    summary = [line(k, out[k]) for k in order]
    recs = _recommend(out, order, analysis.get("top_k", 3), profile["name"])
    warnings = list(base_warnings or [])
    warnings += [x for x in notes if x not in warnings]
    n_untimed = sum(a.get("untimed_cells", 0) for a in out.values())
    if n_untimed:
        n_cells = sum(_evaluated_n(a) for a in out.values())
        warnings.append(UNTIMED_NOTE.format(n=n_untimed, m=n_cells, profile=profile["name"]))
    if not kernel:  # known gaps of the factories the options run (the baseline's are in base_warnings)
        from .ops import registry

        gaps = getattr(registry.module_for(op), "KNOWN_GAPS", {})
        for f in sorted({f for a in facs.values() for f in a} - {base_fac}):
            gap = gaps.get((profile["name"], f))
            if gap:
                ids = ", ".join(aid for aid in out if f in facs[aid])
                warnings.append(f"{op} {f} on {profile['name']} ({ids}): known gap, {gap:.0%} mean error over the "
                                f"calibration runs (target 3%); treat as indicative")
    if any(hyp_any.values()):
        warnings.append(HYPOTHETICAL_WARNING)
    n_out = sum(outside[i][j] for i in range(nH) for j in range(nW))
    if n_out:
        note = (f"{n_out} of {len(evaluated)} evaluated shapes are outside the calibrated range for {op} "
                f"{base_fac} on {accuracy.range_on(profile, rng)} ({accuracy.range_txt(rng)}): extrapolated "
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
    dtype = w.get("dtype", "bf16")
    acc = accuracy.layout(profile, op, base_fac or "?", dtype, False, bool(what_if.get("cost_overrides")))
    if any(hyp_any.values()) and not accuracy.source_of(profile)[1]:
        acc += "; approaches marked hypothetical: " + accuracy.hypothetical_note(profile, op, base_fac, dtype).removeprefix(
            "hypothetical setup: ")
        # approaches whose only knob has simulator runs of a kernel variant (e.g. tilize write_batch): say so
        by_knob = {}
        for aid, ov in approaches:
            knobs = set(ov) & set(DESIGN_KNOBS)
            if hyp_any[aid] and len(knobs) == 1 and knobs == set(ov):
                by_knob.setdefault(next(iter(knobs)), []).append(aid)
        for knob, aids in sorted(by_knob.items()):
            v = accuracy.validation(profile, op, base_fac, knob)
            if v:
                acc += f" (except {', '.join(aids)}: {accuracy.validation_txt(profile, knob, v)})"
    elif any(hyp_any.values()):  # a derived profile
        acc += "; the approaches marked hypothetical are not in the kernels either: no simulator measurement"
    others = sorted({f for a in facs.values() for f in a} - {base_fac})
    if others and not kernel:  # options that run another factory carry that factory's error
        acc += "; options on other factories: " + "; ".join(
            accuracy.layout(profile, op, f, dtype, False, bool(what_if.get("cost_overrides"))) for f in others)
    base_info = {k: v for k, v in base_info.items() if k != "dram_capped" or dram_bounded}
    if kernel:
        base_info.pop("untimed")
    res = {
        "status": "partial" if out_of_time else "ok",
        "warnings": warnings,
        "recommendations": recs,
        "accuracy": acc,
        "survey": {"heights": heights, "widths": widths, "skipped": skipped, "baseline_ns": base_grid,
                   "approaches": out, "best_per_shape": best, "summary": summary,
                   "outside_calibrated_range": outside, "calibrated_range": rng,
                   "kind": kind, "baseline": dict(base_info, factory=base_fac),
                   "stats": {"shapes": nH * nW, "evaluated": len(evaluated), "skipped": len(skipped),
                             "not_evaluated": not_evaluated, "plans_evaluated": n_eval, "plans_reused": n_reused,
                             "approaches": len(approaches), "out_of_time": out_of_time, "time_budget_s": budget_s,
                             "eval_ms": eval_ms, "outside_calibrated_range": n_out},
                   "speedup_basis": f"nominal costs; faster / slower beyond {NOISE:.0%} (the compare verdict "
                                    f"threshold), neutral within it"},
    }
    if defaults_applied:
        res["applied_defaults"] = defaults_applied
    other = spec.get("compare_profile")
    if other and compare:
        res["survey"]["compare"] = cmp = _compare(req, base_knobs, other, what_if, analysis, t_start, res)
        if cmp["status"] == "partial":
            res["status"] = "partial"
        res["warnings"] += [f"compare_profile {other}: {x}" for x in cmp["warnings"]]
    return res


def _evaluated_n(a):
    c = a["counts"]
    return c["faster"] + c["neutral"] + c["slower"]


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
            "approaches": {aid: {k: a[k] for k in ("speedup", "counts", "worst", "best", "dram_capped", "untimed_cells")
                                 if k in a}
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


def _recommend(out, order, top_k, profile_name=""):
    """Approaches faster on most shapes and never slower (by geomean); else the best geomean with its regressions.
    Options with untimed cells (a side the profile does not time) are never recommended."""
    timed = [k for k in order if not out[k].get("untimed_cells")]
    if len(timed) < len(order) and not timed:
        return [{"id": None, "label": "inconclusive", "reason": f"every option compares against a side {profile_name} "
                                                                f"does not time (untimed cells): rerun with the "
                                                                f"interleaved side in L1, or on silicon-hypothesis"}]
    order = timed

    def label(aid):
        return f"{aid} ({describe(out[aid]['overrides']) or 'no overrides'})"

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
