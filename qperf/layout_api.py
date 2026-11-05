"""analyze() for Quasar data-movement requests (qperf/ops/layout.py and the registered op modules). Same response
shape as binary_ng.

Workload: op tilize | untilize | tilize_with_val_padding | untilize_with_unpadding, dtype, shape [..., H, W] = the
logical (unpadded) tensor, leading dims fold into rows; the padding variants pad to / unpad from the next tile
multiple. placements.a = the tensor's memory (dram | l1 interleaved, borrowed = height-sharded L1, tilize only).
Knobs: use_multicore (default true), input (overrides placements.a); tilize Default / untilize MultiCore also take the
hypothetical design knobs (cores, split, tiles_per_block, dfb_blocks, write_batch | read_batch; ops/layout.py
resolve_design). Actions: estimate, compare, tune (over the design knobs, DEFAULT_LAYOUT_SPACE), sensitivity (param
num_rows | num_cols; baseline and every candidate), survey (qperf/survey.py: approaches across a grid of shapes).
"""

from __future__ import annotations

import itertools
import math
import time

from . import ENGINE_VERSION, SCHEMA_VERSION, accuracy
from .api import NOISE, RequestError, _compact, _compare_rows, _ranking, _recommend
from .ops import registry
from .ops.layout import (DESIGN_FACTORIES, DESIGN_KNOBS, GRID, SHARD_OPS, STRATEGY, TILE, estimate, family_of, plan,
                         prefix_of, resolve_design, structure)
from .profiles import SCENARIOS, _pick, load, sha256

BASE_OPS = ("tilize", "untilize", "tilize_with_val_padding", "untilize_with_unpadding", "interleaved_to_sharded",
            "sharded_to_interleaved")
MEM_OF = {"dram": "dram", "l1": "l1", "borrowed": "hs"}
# tune's default space; "batch" is the factory's batched DM thread (tilize write_batch, untilize read_batch), None =
# the factory default, "half" = half the core's chunk. Combos the shape cannot take are dropped, equal plans deduped.
DEFAULT_LAYOUT_SPACE = {"cores": [8, 16, 32], "split": ["rows", "cols", "2d"], "tiles_per_block": [None, "half"],
                        "dfb_blocks": [1, 2], "batch": [1, 4, 8]}
# craq-sim's DM costs are per request with little per byte: plans with shorter requests than the factory's gain more
# there than on ttsim-perf (format: ids = the setups that shrink requests)
CRAQ_SMALL_REQUESTS = ("craq-sim charges each NoC request a fixed cost with little per-byte cost, so setups that shrink "
                       "the requests (split cols / 2d or fewer tiles_per_block: shorter stick reads / row writes; here "
                       "{ids}) look better here than on ttsim-perf, which times L1 NoC transfers; ttsim-perf gives the "
                       "more conservative number")


def _shrinks_requests(r, W):
    """A hypothetical row whose blocks are narrower than the tensor row (shorter stick reads / row writes)."""
    c = r.get("config") or {}
    return bool(c.get("hypothetical")) and c.get("tiles_per_block", math.inf) < math.ceil(W / TILE)


def _craq_warning(profile, rows, W):
    """CRAQ_SMALL_REQUESTS naming (up to 3 of) the rows whose plans shrink the requests, on craq-sim only."""
    ids = [r["id"] for r in rows if r.get("valid") and _shrinks_requests(r, W)]
    if not ids or profile["name"] != "craq-sim":
        return []
    return [CRAQ_SMALL_REQUESTS.format(ids=", ".join(ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else ""))]


def layout_ops():
    """Every op this API answers: the base layout ops plus registered op modules."""
    return BASE_OPS + tuple(op for m in registry.MODULES for op in m.OPS)


def _own_module(op):
    """The op module that builds its own memory string (move, reshard: no use_multicore / input knobs), or None."""
    m = registry.module_for(op)
    return m if m and hasattr(m, "mem_of") else None


class LayoutError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def _costs(profile, scenario, overrides, compute_scale=1.0):
    c = {k: _pick(p, scenario) for k, p in profile["params"].items()}
    for k, v in (overrides or {}).items():
        c[k] = _pick(v, scenario)
    if compute_scale != 1.0:  # what_if.compute_service_scale: scale the compute TRISC costs
        for k in c:
            term = k.split(".", 1)[-1].removeprefix("f32_")  # fp32 overrides scale as their bf16 terms
            if term.startswith(("cu_", "um_", "pk_")) and not term.endswith(("cu_first", "cu_tail")):
                c[k] *= compute_scale
    return c


def _workload(w, knobs):
    op, shape = w["op"], w.get("shape")
    if not shape or len(shape) < 2:
        raise LayoutError("invalid", f"{op} needs workload.shape [..., H, W]")
    base = {"op": op, "dtype": w.get("dtype", "bf16"), "H": math.prod(shape[:-1]), "W": shape[-1]}
    design = {k: knobs[k] for k in DESIGN_KNOBS if knobs.get(k) is not None}
    if design:
        if op not in ("tilize", "untilize"):
            raise LayoutError("unsupported", f"layout design knobs ({', '.join(sorted(design))}) are modeled for tilize "
                                             f"Default and untilize MultiCore only, not {op}")
        base["design"] = design
    m = _own_module(op)
    if m:
        try:
            mem, label = m.mem_of(w, knobs)
        except ValueError as e:
            raise LayoutError("invalid", str(e))
        return dict(base, mem=mem, placement=(w.get("placements") or {}).get("a", "dram"), shard=label,
                    use_multicore=True)
    place = knobs.get("input") or (w.get("placements") or {}).get("a", "dram")
    if op not in SHARD_OPS:
        return dict(base, mem=MEM_OF[place], placement=place, use_multicore=knobs.get("use_multicore", True))
    # the L1 shard layout and the interleaved side (placements.a: dram | l1)
    sh = w.get("shard") or {}
    strategy = knobs.get("shard_strategy") or sh.get("strategy")
    cores = knobs.get("shard_cores") or sh.get("cores")
    if not strategy or not cores:
        raise LayoutError("invalid", f"{op} needs workload.shard {{strategy: height|width|block, cores: N}}")
    if place not in ("dram", "l1"):
        raise LayoutError("invalid", f"{op}: placements.a is the interleaved side, dram or l1")
    if strategy == "block" and math.isqrt(cores) ** 2 != cores:
        raise LayoutError("unsupported", "block sharding is modeled on a square core grid (4, 9, 16, ...)")
    layout = w.get("layout", "tile")  # tile: even tile shards; row_major: sticks, the last shard may be partial
    if layout not in ("tile", "row_major"):
        raise LayoutError("invalid", f"{op}: workload.layout is tile or row_major")
    rm = layout == "row_major"
    mem = STRATEGY[strategy] + str(cores) + ("l" if place == "l1" else "") + ("r" if rm else "")
    return dict(base, mem=mem, placement=place, shard=f"{strategy} x{cores}" + (", row_major" if rm else ""),
                use_multicore=True)


def _result(wl, profile, what_if):
    op, H, W = wl["op"], wl["H"], wl["W"]
    design = wl.get("design")
    try:
        factory = plan(op, H, W, wl["dtype"], wl["mem"], wl["use_multicore"], design)[0]
        d = resolve_design(op, factory, H, W, wl["dtype"], design) if design else None
    except ValueError as e:
        raise LayoutError("unsupported", str(e))
    hyp = bool(d and d["hypothetical"])
    p = prefix_of(op, factory)
    if op in SHARD_OPS:
        label = f"{op} ({wl['shard']}, interleaved side {wl['placement']})"
    elif wl.get("shard"):
        label = f"{op} {factory} ({wl['shard']})"
    elif hyp:
        label = f"{op} {factory} hypothetical ({', '.join(d['changed'])}; input {wl['placement']})"
    else:
        label = f"{op} {factory} ({'multicore' if wl['use_multicore'] else 'single core'}, input {wl['placement']})"
    mod = registry.module_for(op)
    config = {"factory": factory} if _own_module(op) else \
        {"factory": factory, "use_multicore": wl["use_multicore"], "input": wl["placement"]}
    res = {"label": label, "config": config, "valid": True, "errors": [], "warnings": []}
    if hyp:
        config.update({"hypothetical": True, "cores": d["cores"], "split": d["split"],
                       "tiles_per_block": d["tiles_per_block"], "dfb_blocks": d["dfb_blocks"], d["batch_knob"]: d["batch"]})
        res["warnings"].append(f"hypothetical layout ({', '.join(d['changed'])}): not in the pinned kernels; timed with "
                               f"the factory's calibrated per-request costs")
    if hasattr(mod, "notes"):  # what this profile cannot time for the op
        warn, res["untimed"] = mod.notes(profile["name"], wl["mem"])
        res["warnings"] += warn
    gap = getattr(mod, "KNOWN_GAPS", {}).get((profile["name"], factory))
    if gap:  # a factory a simulator times worse than the target (mean error over the calibration runs)
        res["warnings"].append(f"{op} {factory} on {profile['name']}: known gap, {gap:.0%} mean error over the calibration "
                               f"runs (target 3%); treat as indicative and confirm with one simulator run")
    if p is None or f"{p}.start_reader" not in profile["params"]:  # every calibrated factory has its kernel starts
        raise LayoutError("needs_input", f"cost profile '{profile['name']}' has no costs for {op} {factory}; calibrate it "
                                         f"(calibration/fit_layout.py)")
    if wl["dtype"] != "bf16" and not any(k.startswith(f"{p}.f32_") for k in
                                         list(profile["params"]) + list(what_if.get("cost_overrides") or {})):
        res["warnings"].append(f"fp32 is not calibrated for {op} {factory} on this profile: costs are the bf16 fit")
    if family_of(op) == "tilize" and math.ceil(W / TILE) > 64 and profile["name"] == "craq-sim":
        res["warnings"].append("craq-sim cannot run tilize wider than 64 tiles (mop_expander); this is an extrapolation")
    knobs = {ch.split("=")[0] for ch in d["changed"]} if hyp else None  # the design knobs this row changes
    res["accuracy"] = accuracy.layout(profile, op, factory, wl["dtype"], hyp, bool(what_if.get("cost_overrides")), knobs)
    out_of_range = accuracy.range_warning(profile, op, factory, H, W)
    if out_of_range:
        res["warnings"].append(out_of_range)
        res["outside_calibrated_range"] = True
    ghz = profile["clock"]["GHz"]
    costs = {sc: _costs(profile, sc, what_if.get("cost_overrides"), what_if.get("compute_service_scale", 1.0))
             for sc in SCENARIOS}
    runs = {sc: estimate(op, H, W, costs[sc], wl["dtype"], wl["mem"], wl["use_multicore"], trace=3, design=design)
            for sc in SCENARIOS}
    nominal = runs["nominal"]
    busiest_cfg, _, run = max(nominal["loads"], key=lambda x: x[2]["makespan"])
    span = run["makespan"]
    roles = {}
    for name, rr in run["roles"].items():
        roles[name] = {"threads": rr["threads"], "useful_busy_frac": round(rr["busy"] / span, 3),
                       "starved_frac": round(rr["wait_data"] / span, 3),
                       "backpressure_frac": round(rr["wait_space"] / span, 3),
                       "transfer_wait_frac": round(rr["wait_transfer"] / span, 3),
                       "start_ns": round(rr["start_min"] / ghz, 1)}
    busiest = max(roles, key=lambda k: roles[k]["useful_busy_frac"])
    frac, wrole, wkind = max([(r["starved_frac"], k, "starved") for k, r in roles.items()] +
                             [(r["backpressure_frac"], k, "backpressure") for k, r in roles.items()])
    tiles = math.ceil(H / TILE) * math.ceil(W / TILE)
    # the shared network to / from L1 binds, not a thread; or (a profile with an assumed DRAM bandwidth) DRAM
    l1_binds = nominal.get("l1_bound") and nominal["l1_bound"] >= nominal["span"] - 1e-9
    dram_binds = nominal.get("dram_bound") and nominal["dram_bound"] >= nominal["span"] - 1e-9
    res.update({
        "result_kind": "finite_buffer_estimate",
        "time_ns": {k: round(r["span"] / ghz, 1) for k, r in runs.items()},
        "time_cycles": round(nominal["span"], 1),
        "per_tile_ns": round(nominal["span"] / ghz / tiles, 3),
        "bottleneck": {"resource": "dram_global" if dram_binds else "noc_l1_global" if l1_binds else busiest,
                       "busiest_role": busiest,
                       "dominant_wait": {"role": wrole, "kind": wkind, "fraction": frac} if frac > 0.05 else None},
        "roles": roles,
        "work_split": [{"blocks_per_core": c["blocks"], "tiles_per_block": c["wt"], "cores": n} for c, n, _ in nominal["loads"]],
    })
    if "dram_bound" in nominal:  # reported as binary_ng reports its global DRAM bound
        res["lower_bound_ns"] = round(nominal["dram_bound"] / ghz, 1)
        res["bounds_ns"] = {"dram_global": round(nominal["dram_bound"] / ghz, 1)}
        if dram_binds:
            res["bottleneck"]["bound_share_of_estimate"] = 1.0
            res["warnings"].append(f"bound by the {accuracy.assumed_dram(profile)}: all cores' DRAM traffic "
                                   f"({nominal['dram_bytes'] / 2**20:.1f} MiB) cannot move faster; the busiest core alone "
                                   f"would take {span / ghz:.0f} ns")
    st = structure(op, factory, H, W, nominal["plan"], busiest_cfg, costs["nominal"])
    if st:
        res["structure"] = st
    if factory == "SingleCore" and wl["use_multicore"] is False:
        res["warnings"].append(f"single core: one of {GRID} cores does all the work")
    return res


def _design_id(wl, base_wl):
    """A tune row's id: the resolved design (every knob, factory defaults included) plus input / use_multicore when
    they differ from the baseline."""
    op, H, W = wl["op"], wl["H"], wl["W"]
    factory = plan(op, H, W, wl["dtype"], wl["mem"], wl["use_multicore"])[0]
    extra = (f" input={wl['placement']}" if wl["placement"] != base_wl["placement"] else "") + \
            (f" use_multicore={str(wl['use_multicore']).lower()}" if wl["use_multicore"] != base_wl["use_multicore"] else "")
    if (op, factory) not in DESIGN_FACTORIES:
        return f"{op} {factory}" + extra
    d = resolve_design(op, factory, H, W, wl["dtype"], wl.get("design"))
    return (f"cores={d['cores']} split={d['split']} tiles_per_block={d['tiles_per_block']} dfb_blocks={d['dfb_blocks']} "
            f"{d['batch_knob']}={d['batch']}" + extra)


def _simplicity(r, changed):
    c = r["config"]
    return (bool(c.get("hypothetical")), len(changed.get(r["id"], ())), c.get("cores", 0))


def _tune(req, base_knobs, base_wl, base, costs, what_if, spec, analysis, detail, t_start):
    """Exhaustive search over the hypothetical design knobs: every combination of the space (DEFAULT_LAYOUT_SPACE
    merged with tune.space) is resolved to a plan, invalid ones (knobs the factory or shape cannot take) are counted
    and dropped, equal plans are evaluated once, then every distinct plan gets the full model, the baseline first."""
    op = base_wl["op"]
    factory = base["config"]["factory"]
    batch_knob = DESIGN_FACTORIES.get((op, factory))
    if batch_knob is None:
        raise LayoutError("unsupported", f"tune explores the hypothetical layout design knobs of tilize Default and "
                                         f"untilize MultiCore; {op} {factory} has none (use compare: use_multicore, input)")
    space = {(batch_knob if k == "batch" else k): v for k, v in DEFAULT_LAYOUT_SPACE.items()}
    space.update(spec.get("space") or {})
    budget_ms = analysis.get("time_budget_ms", 20000)
    max_c = spec.get("max_candidates")
    top_k = analysis.get("top_k", 3)
    keys = list(space)
    seen, plans = {}, []
    enumerated = invalid = 0
    for combo in itertools.product(*[space[k] if isinstance(space[k], list) else [space[k]] for k in keys]):
        enumerated += 1
        knobs = dict(base_knobs, **{k: v for k, v in zip(keys, combo) if v is not None})
        try:
            wl = _workload(req["workload"], knobs)
            f = plan(wl["op"], wl["H"], wl["W"], wl["dtype"], wl["mem"], wl["use_multicore"], wl.get("design"))[0]
            d = resolve_design(wl["op"], f, wl["H"], wl["W"], wl["dtype"], wl.get("design")) if wl.get("design") else None
        except (LayoutError, ValueError):
            invalid += 1
            continue
        sig = (wl["mem"], wl["use_multicore"], f, d["sig"] if d and d["hypothetical"] else "factory")
        if sig in seen:
            continue
        seen[sig] = True
        plans.append((wl, d["changed"] if d and d["hypothetical"] else []))
    plans.sort(key=lambda p: (bool(p[1]), len(p[1])))
    truncated = max_c is not None and len(plans) > max_c
    cands = plans[:max_c]
    base_id = _design_id(base_wl, base_wl)
    evaluated, out_of_time, n_changed = [], False, {}
    base_row = dict(base, id=base_id, comparison=_compare_rows(base, base))
    evaluated.append(base_row)
    for wl, changed in cands:
        rid = _design_id(wl, base_wl)
        if rid == base_id:
            continue
        if (time.perf_counter() - t_start) * 1e3 > budget_ms:
            out_of_time = True
            break
        try:
            r = _result(wl, costs, what_if)
        except LayoutError:
            invalid += 1
            continue
        r["id"] = rid
        n_changed[rid] = changed
        r["comparison"] = _compare_rows(base, r)
        if base.get("untimed") or r.get("untimed"):
            r["comparison"]["verdict"] = "inconclusive"
        evaluated.append(r)
    simple = lambda r: _simplicity(r, n_changed)  # noqa: E731
    evaluated.sort(key=lambda r: (r["time_ns"]["nominal"], simple(r)))
    best = evaluated[0]["time_ns"]["nominal"]
    # among plans indistinguishable from the best, prefer the simpler one (the factory's, then fewer changed knobs)
    tied = sorted([r for r in evaluated if r["time_ns"]["nominal"] <= best * (1 + NOISE)], key=simple)
    picks = (tied + [r for r in evaluated if r not in tied])[:top_k]
    return {
        "baseline": _compact(base, detail),
        "top": [_compact(r, detail) for r in picks],
        "search": {"enumerated": enumerated, "invalid": invalid, "valid": len(plans), "considered": len(cands),
                   "evaluated": len(evaluated), "truncated_by_max_candidates": truncated, "out_of_time": out_of_time,
                   "exhaustive": not truncated and not out_of_time,
                   "order": "every distinct valid plan (equal plans deduped) gets the full model, the baseline first, "
                            "then the factory-like plans before those changing more knobs; a budget stop (out_of_time) "
                            "leaves the rest unevaluated, not proven worse",
                   "evaluated_ids": [r["id"] for r in evaluated]},
        "recommendations": _recommend([_compact(r, "compact") for r in evaluated if r is not base_row], top_k),
        "warnings": _craq_warning(costs, [r for r in evaluated if r is not base_row], base_wl["W"]),
    }


def analyze_layout(req, t_start):
    prof = req.get("profiles", {})
    if "costs" not in prof:
        raise RequestError("needs_input", "profiles.costs is required (e.g. craq-sim, ttsim-perf)")
    _, arch_id = load("architecture", prof.get("architecture", "quasar"))
    _, kernel_id = load("kernel", "layout-quasar")
    costs, costs_id = load("costs", prof["costs"])
    action = req["action"]
    analysis = req.get("analysis", {})
    detail = analysis.get("detail", "compact")
    what_if = dict(req.get("what_if", {}))
    base_knobs = dict(req.get("baseline") or {})
    op = req["workload"]["op"]
    own = _own_module(op)
    result = {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "ok",
              "profiles": {"architecture": arch_id, "kernel": kernel_id, "costs": costs_id},
              "applied_defaults": [] if "use_multicore" in base_knobs or own else ["use_multicore=true (default)"],
              "assumptions": [getattr(registry.module_for(op), "ASSUMPTION", None) or
                              f"{op}: one reader, one compute, one writer thread per core (the factory's kernels)"],
              "warnings": [], "result_kind": "finite_buffer_estimate"}
    if costs.get("platform_scope"):
        result["platform_scope"] = costs["platform_scope"]
    try:
        if action == "survey":
            from .survey import run_survey

            sres = run_survey(req, base_knobs, costs, what_if, analysis, t_start)
            result["applied_defaults"] += sres.pop("applied_defaults", [])
            result.update(sres)
            wl = {"workload": {k: v for k, v in req["workload"].items() if k != "shape"}, "baseline": base_knobs,
                  "survey": {k: result["survey"][k] for k in ("heights", "widths")},
                  "approaches": {k: a["overrides"] for k, a in result["survey"]["approaches"].items()}}
            if "compare" in result["survey"]:
                wl["compare_profile"] = result["survey"]["compare"]["profile"]
        else:
            wl = _workload(req["workload"], base_knobs)
            base = _result(wl, costs, what_if)
            result["baseline"] = _compact(base, detail)
            result["accuracy"] = base["accuracy"]
        if action == "compare":
            rows = []
            for c in req.get("candidates", []):
                knobs = dict(base_knobs, **c.get("overrides", {}))
                try:
                    r = _result(_workload(req["workload"], knobs), costs, what_if)
                except LayoutError as e:
                    r = {"label": str(knobs), "config": {}, "valid": False, "errors": [e.message]}
                r["id"] = c.get("id", r["label"])
                r["comparison"] = _compare_rows(base, r)
                if r.get("valid") and (base.get("untimed") or r.get("untimed")):  # the profile cannot rank them
                    r["comparison"]["verdict"] = "inconclusive"
                rows.append(_compact(r, detail))
            result["candidates"] = rows
            result["recommendations"] = _recommend(rows, analysis.get("top_k", 3))
            result["warnings"] += _craq_warning(costs, rows, wl["W"])
        elif action == "tune":
            result.update(_tune(req, base_knobs, wl, base, costs, what_if, req.get("tune") or {}, analysis, detail,
                                t_start))
            if not result["search"]["exhaustive"]:
                result["status"] = "partial"
        elif action == "sensitivity":
            spec = req.get("sensitivity") or {}
            param = spec.get("param")
            if param not in ("num_rows", "num_cols"):
                raise LayoutError("invalid", f"{wl['op']} sensitivity param must be num_rows or num_cols")
            configs = [("baseline", base_knobs)] + [(c.get("id", str(c.get("overrides"))),
                                                     dict(base_knobs, **c.get("overrides", {})))
                                                    for c in req.get("candidates", [])]
            series = {cid: [] for cid, _ in configs}
            for v in spec.get("values", []):
                for cid, knobs in configs:
                    try:
                        cw = _workload(req["workload"], knobs)
                        r = _result(dict(cw, **{"H" if param == "num_rows" else "W": int(v)}), costs, what_if)
                        pt = {"value": v, "valid": True, "time_ns": r["time_ns"]["nominal"],
                              "bottleneck": r["bottleneck"]["resource"]}
                    except LayoutError:
                        if cid == "baseline":
                            raise
                        pt = {"value": v, "valid": False, "time_ns": None, "bottleneck": None}
                    series[cid].append(pt)
            crossings, shifts = _ranking(series, param)
            result["sensitivity"] = {"param": param, "series": series, "ranking_changes": crossings,
                                     "bottleneck_shifts": shifts}
    except LayoutError as e:
        return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": e.status,
                "message": e.message, "result_kind": None, "runtime_ms": round((time.perf_counter() - t_start) * 1e3, 2)}
    result["resolved_input_sha256"] = sha256({"workload": wl, "action": action, "profiles": result["profiles"]})
    result["cache_hits"] = 0
    result["runtime_ms"] = round((time.perf_counter() - t_start) * 1e3, 2)
    return result
