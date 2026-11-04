"""analyze() for Quasar tilize / untilize requests (qperf/ops/layout.py). Same response shape as binary_ng.

Workload: op tilize | untilize | tilize_with_val_padding | untilize_with_unpadding, dtype, shape [..., H, W] = the
logical (unpadded) tensor, leading dims fold into rows; the padding variants pad to / unpad from the next tile
multiple. placements.a = the tensor's memory (dram | l1 interleaved, borrowed = height-sharded L1, tilize only). Knobs: use_multicore (default true),
input (overrides placements.a). Actions: estimate, compare, sensitivity (param num_rows | num_cols).
"""

from __future__ import annotations

import math
import time

from . import ENGINE_VERSION, SCHEMA_VERSION
from .ops import registry
from .ops.layout import GRID, TILE, estimate, family_of, plan, prefix_of
from .profiles import SCENARIOS, _pick, load, sha256

BASE_OPS = ("tilize", "untilize", "tilize_with_val_padding", "untilize_with_unpadding", "interleaved_to_sharded",
            "sharded_to_interleaved")


def layout_ops():
    """Every op this API answers: the base layout ops plus registered op modules."""
    return BASE_OPS + tuple(op for m in registry.MODULES for op in m.OPS)


LAYOUT_OPS = BASE_OPS  # kept for callers; use layout_ops() to include registered modules
SHARD_OPS = ("interleaved_to_sharded", "sharded_to_interleaved")
MEM_OF = {"dram": "dram", "l1": "l1", "borrowed": "hs"}


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
            if k.split(".", 1)[-1].startswith(("cu_", "um_", "pk_")) and not k.endswith(("cu_first", "cu_tail")):
                c[k] *= compute_scale
    return c


def _workload(w, knobs):
    m = registry.module_for(w["op"])
    if m and hasattr(m, "mem_of"):
        shape = w.get("shape")
        if not shape or len(shape) < 2:
            raise LayoutError("invalid", f"{w['op']} needs workload.shape [..., H, W]")
        try:
            mem, label = m.mem_of(w, knobs)
        except ValueError as e:
            raise LayoutError("invalid", str(e))
        return {"op": w["op"], "dtype": w.get("dtype", "bf16"), "H": math.prod(shape[:-1]), "W": shape[-1], "mem": mem,
                "placement": (w.get("placements") or {}).get("a", "dram"), "shard": label, "use_multicore": True}
    shape = w.get("shape")
    if not shape or len(shape) < 2:
        raise LayoutError("invalid", f"{w['op']} needs workload.shape [..., H, W]")
    rows = math.prod(shape[:-1])
    W = shape[-1]
    place = knobs.get("input") or (w.get("placements") or {}).get("a", "dram")
    if w["op"] in SHARD_OPS:  # the L1 shard layout and the interleaved side (placements.a: dram | l1)
        sh = dict(w.get("shard") or {})
        strategy = knobs.get("shard_strategy") or sh.get("strategy")
        cores = knobs.get("shard_cores") or sh.get("cores")
        if not strategy or not cores:
            raise LayoutError("invalid", f"{w['op']} needs workload.shard {{strategy: height|width|block, cores: N}}")
        if place not in ("dram", "l1"):
            raise LayoutError("invalid", f"{w['op']}: placements.a is the interleaved side, dram or l1")
        if strategy == "block" and math.isqrt(cores) ** 2 != cores:
            raise LayoutError("unsupported", "block sharding is modeled on a square core grid (4, 9, 16, ...)")
        mem = {"height": "hs", "width": "ws", "block": "bs"}[strategy] + str(cores) + ("l" if place == "l1" else "")
        return {"op": w["op"], "dtype": w.get("dtype", "bf16"), "H": rows, "W": W, "mem": mem, "placement": place,
                "shard": f"{strategy} x{cores}", "use_multicore": True}
    return {"op": w["op"], "dtype": w.get("dtype", "bf16"), "H": rows, "W": W, "mem": MEM_OF[place],
            "placement": place, "use_multicore": knobs.get("use_multicore", True)}


def _result(wl, profile, what_if):
    op, H, W = wl["op"], wl["H"], wl["W"]
    try:
        factory, loads = plan(op, H, W, wl["dtype"], wl["mem"], wl["use_multicore"])
    except ValueError as e:
        raise LayoutError("unsupported", str(e))
    p = prefix_of(op, factory)
    if op in SHARD_OPS or wl.get("shard"):
        label = f"{op} ({wl['shard']}, interleaved side {wl['placement']})" if op in SHARD_OPS else f"{op} ({wl['shard']})"
    else:
        label = f"{op} {factory} ({'multicore' if wl['use_multicore'] else 'single core'}, input {wl['placement']})"
    res = {"label": label, "config": {"factory": factory, "use_multicore": wl["use_multicore"], "input": wl["placement"]},
           "valid": True, "errors": [], "warnings": []}
    if p is None or f"{p}.start_reader" not in profile["params"]:  # every calibrated factory has its kernel starts
        raise LayoutError("needs_input", f"cost profile '{profile['name']}' has no costs for {op} {factory}; calibrate it "
                                         f"(calibration/fit_layout.py)")
    if wl["dtype"] != "bf16":
        res["warnings"].append("fp32 is not calibrated for tilize/untilize: costs are the bf16 fit")
    if family_of(op) == "tilize" and math.ceil(W / TILE) > 64 and profile["name"] == "craq-sim":
        res["warnings"].append("craq-sim cannot run tilize wider than 64 tiles (mop_expander); this is an extrapolation")
    ghz = profile["clock"]["GHz"]
    times, nominal = {}, None
    for sc in SCENARIOS:
        r = estimate(op, H, W, _costs(profile, sc, what_if.get("cost_overrides"), what_if.get("compute_service_scale", 1.0)),
                     wl["dtype"], wl["mem"],
                     wl["use_multicore"], trace=3)
        times[sc] = r["span"]
        if sc == "nominal":
            nominal = r
    cfg, cores, run = max(nominal["loads"], key=lambda x: x[2]["makespan"])
    span = run["makespan"]
    roles = {}
    for name, rr in run["roles"].items():
        roles[name] = {"threads": rr["threads"], "useful_busy_frac": round(rr["busy"] / span, 3),
                       "starved_frac": round(rr["wait_data"] / span, 3),
                       "backpressure_frac": round(rr["wait_space"] / span, 3),
                       "transfer_wait_frac": round(rr["wait_transfer"] / span, 3),
                       "start_ns": round(rr["start_min"] / ghz, 1)}
    busiest = max(roles, key=lambda k: roles[k]["useful_busy_frac"])
    waits = [(r["starved_frac"], k, "starved") for k, r in roles.items()] + \
            [(r["backpressure_frac"], k, "backpressure") for k, r in roles.items()]
    frac, wrole, wkind = max(waits)
    tiles = math.ceil(H / TILE) * math.ceil(W / TILE)
    if nominal.get("l1_bound") and nominal["l1_bound"] >= nominal["span"] - 1e-9:
        busiest = "noc_l1_global"  # the shared network to / from L1 binds, not a thread
    res.update({
        "result_kind": "finite_buffer_estimate",
        "time_ns": {k: round(v / ghz, 1) for k, v in times.items()},
        "time_cycles": round(times["nominal"], 1),
        "per_tile_ns": round(times["nominal"] / ghz / tiles, 3),
        "bottleneck": {"resource": busiest, "busiest_role": max(roles, key=lambda k: roles[k]["useful_busy_frac"]),
                       "dominant_wait": {"role": wrole, "kind": wkind, "fraction": frac} if frac > 0.05 else None},
        "roles": roles,
        "work_split": [{"blocks_per_core": c["blocks"], "tiles_per_block": c["wt"], "cores": n} for c, n, _ in nominal["loads"]],
    })
    if factory == "SingleCore" and wl["use_multicore"] is False:
        res["warnings"].append(f"single core: one of {GRID} cores does all the work")
    return res


def _compare(base, cand):
    from .api import _compare_rows

    return _compare_rows(base, cand)


def analyze_layout(req, t_start):
    from .api import RequestError, _compact, _recommend

    prof = req.get("profiles", {})
    if "costs" not in prof:
        raise RequestError("needs_input", "profiles.costs is required (e.g. craq-sim, ttsim-perf)")
    arch, arch_id = load("architecture", prof.get("architecture", "quasar"))
    kernel, kernel_id = load("kernel", "layout-quasar")
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
        result = {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "ok",
                  "profiles": {"architecture": arch_id, "kernel": kernel_id, "costs": costs_id},
                  "applied_defaults": [] if "use_multicore" in base_knobs else ["use_multicore=true (default)"],
                  "assumptions": [f"{wl['op']}: one reader, one compute, one writer thread per core (the factory's kernels)"],
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
                r["comparison"] = _compare(base, r)
                rows.append(_compact(r, detail))
            result["candidates"] = rows
            result["recommendations"] = _recommend(base, rows, analysis.get("top_k", 3))
        elif action == "sensitivity":
            spec = req.get("sensitivity") or {}
            param = spec.get("param")
            if param not in ("num_rows", "num_cols"):
                raise LayoutError("invalid", f"{wl['op']} sensitivity param must be num_rows or num_cols")
            series = {"baseline": []}
            for v in spec.get("values", []):
                w2 = dict(wl, **({"H": int(v)} if param == "num_rows" else {"W": int(v)}))
                r = _result(w2, costs, what_if)
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
