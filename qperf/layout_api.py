"""analyze() for Quasar data-movement requests (qperf/ops/layout.py and the registered op modules). Same response
shape as binary_ng.

Workload: op tilize | untilize | tilize_with_val_padding | untilize_with_unpadding, dtype, shape [..., H, W] = the
logical (unpadded) tensor, leading dims fold into rows; the padding variants pad to / unpad from the next tile
multiple. placements.a = the tensor's memory (dram | l1 interleaved, borrowed = height-sharded L1, tilize only).
Knobs: use_multicore (default true), input (overrides placements.a). Actions: estimate, compare, sensitivity (param
num_rows | num_cols).
"""

from __future__ import annotations

import math
import time

from . import ENGINE_VERSION, SCHEMA_VERSION
from .api import RequestError, _compact, _compare_rows, _recommend
from .ops import registry
from .ops.layout import GRID, SHARD_OPS, STRATEGY, TILE, estimate, family_of, plan, prefix_of
from .profiles import SCENARIOS, _pick, load, sha256

BASE_OPS = ("tilize", "untilize", "tilize_with_val_padding", "untilize_with_unpadding", "interleaved_to_sharded",
            "sharded_to_interleaved")
MEM_OF = {"dram": "dram", "l1": "l1", "borrowed": "hs"}


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
    try:
        factory = plan(op, H, W, wl["dtype"], wl["mem"], wl["use_multicore"])[0]
    except ValueError as e:
        raise LayoutError("unsupported", str(e))
    p = prefix_of(op, factory)
    if op in SHARD_OPS:
        label = f"{op} ({wl['shard']}, interleaved side {wl['placement']})"
    elif wl.get("shard"):
        label = f"{op} {factory} ({wl['shard']})"
    else:
        label = f"{op} {factory} ({'multicore' if wl['use_multicore'] else 'single core'}, input {wl['placement']})"
    mod = registry.module_for(op)
    config = {"factory": factory} if _own_module(op) else \
        {"factory": factory, "use_multicore": wl["use_multicore"], "input": wl["placement"]}
    res = {"label": label, "config": config, "valid": True, "errors": [], "warnings": []}
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
    ghz = profile["clock"]["GHz"]
    runs = {sc: estimate(op, H, W, _costs(profile, sc, what_if.get("cost_overrides"),
                                          what_if.get("compute_service_scale", 1.0)),
                         wl["dtype"], wl["mem"], wl["use_multicore"], trace=3) for sc in SCENARIOS}
    nominal = runs["nominal"]
    run = max(nominal["loads"], key=lambda x: x[2]["makespan"])[2]
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
    # the shared network to / from L1 binds, not a thread
    l1_binds = nominal.get("l1_bound") and nominal["l1_bound"] >= nominal["span"] - 1e-9
    res.update({
        "result_kind": "finite_buffer_estimate",
        "time_ns": {k: round(r["span"] / ghz, 1) for k, r in runs.items()},
        "time_cycles": round(nominal["span"], 1),
        "per_tile_ns": round(nominal["span"] / ghz / tiles, 3),
        "bottleneck": {"resource": "noc_l1_global" if l1_binds else busiest, "busiest_role": busiest,
                       "dominant_wait": {"role": wrole, "kind": wkind, "fraction": frac} if frac > 0.05 else None},
        "roles": roles,
        "work_split": [{"blocks_per_core": c["blocks"], "tiles_per_block": c["wt"], "cores": n} for c, n, _ in nominal["loads"]],
    })
    if factory == "SingleCore" and wl["use_multicore"] is False:
        res["warnings"].append(f"single core: one of {GRID} cores does all the work")
    return res


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
    try:
        wl = _workload(req["workload"], base_knobs)
        if action == "tune":
            raise LayoutError("unsupported", f"{wl['op']} has no TTNN_QSR knobs to tune; use compare (use_multicore, input)")
        base = _result(wl, costs, what_if)
        own = _own_module(wl["op"])
        result = {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "ok",
                  "profiles": {"architecture": arch_id, "kernel": kernel_id, "costs": costs_id},
                  "applied_defaults": [] if "use_multicore" in base_knobs or own else ["use_multicore=true (default)"],
                  "assumptions": [getattr(registry.module_for(wl["op"]), "ASSUMPTION", None) or
                                  f"{wl['op']}: one reader, one compute, one writer thread per core (the factory's kernels)"],
                  "warnings": [], "result_kind": "finite_buffer_estimate", "baseline": _compact(base, detail)}
        if costs.get("platform_scope"):
            result["platform_scope"] = costs["platform_scope"]
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
        elif action == "sensitivity":
            spec = req.get("sensitivity") or {}
            param = spec.get("param")
            if param not in ("num_rows", "num_cols"):
                raise LayoutError("invalid", f"{wl['op']} sensitivity param must be num_rows or num_cols")
            series = {"baseline": []}
            for v in spec.get("values", []):
                r = _result(dict(wl, **{"H" if param == "num_rows" else "W": int(v)}), costs, what_if)
                series["baseline"].append({"value": v, "valid": True, "time_ns": r["time_ns"]["nominal"],
                                           "bottleneck": r["bottleneck"]["resource"]})
            result["sensitivity"] = {"param": param, "series": series, "ranking_changes": []}
    except LayoutError as e:
        return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": e.status,
                "message": e.message, "result_kind": None, "runtime_ms": round((time.perf_counter() - t_start) * 1e3, 2)}
    result["resolved_input_sha256"] = sha256({"workload": wl, "action": action, "profiles": result["profiles"]})
    result["cache_hits"] = 0
    result["runtime_ms"] = round((time.perf_counter() - t_start) * 1e3, 2)
    return result
