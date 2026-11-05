"""analyze(request) -> result. The one engine behind the CLI, the tests and the LLM skill.

No network, no repository scan, no hardware: a request plus versioned profiles in, a JSON-able dict out.
"""

from __future__ import annotations

import copy
import itertools
import math
import time

from . import ENGINE_VERSION, SCHEMA_VERSION
from .bounds import cluster_bounds, global_dram_bound
from .cache import LRU
from .constraints import check
from .pipeline import Deadlock, evaluate
from .profiles import SCENARIOS, ProfileError, load, param_provenance, resolve_costs, sha256
from .rule import is_sfpu, resolve_native_config, split_work_to_cores
from .traffic import cluster_traffic
from .validate import validate_request

KNOBS = ("reader_threads", "compute_threads", "writer_threads", "dm_batch", "tiles_per_cycle", "entries_per_thread")
SHORT = {"reader_threads": "R", "compute_threads": "C", "writer_threads": "W", "dm_batch": "dm_batch",
         "tiles_per_cycle": "tiles_per_cycle", "entries_per_thread": "entries_per_thread"}
# Hypothetical schedule variants (not in the pinned kernels): knob -> pinned default.
VARIANTS = {"barriers": "cluster", "work_split": "static"}
VARIANT_NOTE = {"barriers": "per-thread NoC barriers (transaction IDs)",
                "work_split": "a shared work queue for the DM threads"}
TILE_BYTES = {"bf16": 2048, "fp32": 4096}
NOISE = 0.02  # relative difference below which two estimates are "indistinguishable" (no meaningful gain)
# The pipeline is evaluated tile by tile up to this many tiles per cluster and extrapolated beyond from two samples
# whole periods apart. Max-plus steady state is periodic, so the extrapolation is exact (0 ppm against tile-by-tile
# evaluation of 200k tiles) at a fraction of the cost.
EXACT_TILES = 2048

_cache = LRU(256)


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ----------------------------------------------------------------------------------------------- workload

def resolve_workload(w, arch):
    out = dict(w)
    applied = []
    dtype = w.get("dtype", "bf16")
    if dtype not in TILE_BYTES:
        raise RequestError("unsupported", f"dtype {dtype} not modeled (bf16, fp32)")
    tiles = w.get("num_tiles")
    if "shape" in w:
        shp = list(w["shape"])
        if len(shp) < 2:
            raise RequestError("invalid", "shape needs at least 2 dims")
        st = math.prod(shp[:-2]) * math.ceil(shp[-2] / 32) * math.ceil(shp[-1] / 32)
        if tiles is not None and tiles != st:
            raise RequestError("invalid", f"num_tiles={tiles} disagrees with shape {shp} -> {st} padded tiles")
        tiles = st
    if not tiles or tiles < 1:
        raise RequestError("needs_input", "workload needs num_tiles or shape")
    clusters = w.get("clusters")
    if clusters is None:
        clusters = arch["clusters_max"]["value"]
        applied.append(f"workload.clusters={clusters} (architecture worker grid)")
    if clusters > arch["clusters_max"]["value"]:
        raise RequestError("invalid", f"clusters={clusters} > {arch['clusters_max']['value']}")
    scope = w.get("tiles_scope", "operation")
    total = tiles * clusters if scope == "cluster" else tiles
    pl = w.get("placements") or {}
    plc = [pl.get("a", "dram"), pl.get("b", "dram"), pl.get("out", "dram")]
    for name in "ab":
        if name not in pl:
            applied.append(f"workload.placements.{name}=dram (default)")
    for v in plc:
        if v not in ("dram", "l1", "borrowed"):
            raise RequestError("invalid", f"placement {v}: use dram | l1 | borrowed")
    op = w.get("op", "add")
    variant = w.get("kernel_variant")
    sf = is_sfpu(op, dtype) if variant is None else (variant == "sfpu")
    if sf is None:
        raise RequestError("unsupported", f"routing of op '{op}' ({dtype}) is not recorded in the kernel profile; "
                                          f"set workload.kernel_variant to fpu|sfpu from the program log")
    out.update(dtype=dtype, num_tiles_total=total, clusters=clusters, placements_list=plc, op=op, sfpu=sf,
               variant=f"{'sfpu' if sf else 'fpu'}:{op}:{dtype}")
    return out, applied


def routing_for(wl, cfg_r):
    plc = wl["placements_list"]
    tb = TILE_BYTES[wl["dtype"]]
    # L1-resident operands (borrowed shards, L1-interleaved pages) occupy each cluster's L1 for their share.
    share = math.ceil(wl["num_tiles_total"] / min(wl["clusters"], wl["num_tiles_total"]))
    resident = share * tb * sum(1 for p in plc if p in ("l1", "borrowed"))
    return {"in_moves": [plc[0] != "borrowed", plc[1] != "borrowed"], "out_moves": plc[2] != "borrowed",
            "placements": plc, "sfpu": wl["sfpu"], "dst32": wl["dtype"] == "fp32", "tile_bytes": [tb, tb, tb],
            "resident_l1_bytes": resident}


def resolve_config(wl, knobs):
    plc = wl["placements_list"]
    moves_in = plc[0] != "borrowed" or plc[1] != "borrowed"
    moves_out = plc[2] != "borrowed"
    any_dram = "dram" in plc
    C = knobs.get("compute_threads") or 4
    share = 0
    if "borrowed" not in plc and (plc[0] == "dram" or plc[1] == "dram"):
        per_cluster = math.ceil(wl["num_tiles_total"] / min(wl["clusters"], wl["num_tiles_total"]))
        share = math.ceil(per_cluster / C)
    moves = {"inputs": moves_in, "output": moves_out, "dram": any_dram, "sfpu": wl["sfpu"],
             "dst32": wl["dtype"] == "fp32", "tiles_per_compute_thread": share,
             "batch_pays": not (wl["op"] == "divide" and wl["dtype"] == "fp32"),
             "dram_batch": knobs.get("dram_batch", True)}
    cfg = resolve_native_config(moves, knobs)
    applied = [f"{SHORT[k]}={cfg[_ck(k)]} (rule)" for k in KNOBS if not knobs.get(k)]
    if not moves_in and knobs.get("reader_threads") not in (None, 1):
        applied.append("R forced to 1: both inputs borrowed")
    if not moves_out and knobs.get("writer_threads") not in (None, 1):
        applied.append("W forced to 1: output borrowed")
    if not moves_in:
        cfg["R"] = 1
    if not moves_out:
        cfg["W"] = 1
    for k, default in VARIANTS.items():
        if knobs.get(k, default) != default:
            cfg[k] = knobs[k]
    return cfg, applied


def _ck(k):
    return {"reader_threads": "R", "compute_threads": "C", "writer_threads": "W"}.get(k, k)


def config_label(cfg):
    return (f"R{cfg['R']} C{cfg['C']} W{cfg['W']} dm_batch={cfg['dm_batch']} tiles_per_cycle={cfg['tiles_per_cycle']}"
            f" entries_per_thread={cfg['entries_per_thread']}" + "".join(f" {k}={cfg[k]}" for k in VARIANTS if k in cfg))


# ----------------------------------------------------------------------------------------------- evaluation

def _period(cfg):
    p = 1
    for v in (cfg["R"], cfg["C"], cfg["W"], cfg["dm_batch"], cfg["tiles_per_cycle"], cfg["entries_per_thread"]):
        p = p * v // math.gcd(p, v)
    return p * max(cfg["R"], cfg["C"], cfg["W"])


def eval_cluster(cfg, routing, costs, n, cap, trace=0):
    """Pipeline makespan for n tiles; beyond cap tiles, linear extrapolation from two periodic samples."""
    pcfg = dict(cfg, in_moves=routing["in_moves"], out_moves=routing["out_moves"], tile_bytes=routing["tile_bytes"])
    if n <= cap:
        r = evaluate(pcfg, costs, n, trace)
        r["extrapolated"] = False
        return r
    P = _period(cfg)
    rem = n % P
    m2 = max(2, (cap - rem) // P)
    m1 = max(1, m2 // 2)
    n1, n2 = rem + m1 * P, rem + m2 * P
    r1 = evaluate(pcfg, costs, n1)
    r2 = evaluate(pcfg, costs, n2, trace)
    slope = (r2["makespan"] - r1["makespan"]) / (n2 - n1)
    r2["makespan_sampled"] = r2["makespan"]
    r2["makespan"] = r2["makespan"] + slope * (n - n2)
    r2["extrapolated"] = True
    r2["samples"] = [n1, n2]
    r2["slope"] = slope
    return r2


def effective_costs(costs, routing):
    """Resolve placement-specific transfer keys (suffix .dram / .l1) into the generic keys the pipeline and the
    bounds read: read latency = slowest moving input's source, write latency = the output's destination, and one
    shared cluster NoC bandwidth that charges each operand's bytes at its own placement's rate."""
    c = dict(costs)
    pl = routing["placements"]
    moving_in = [pl[i] for i in range(2) if routing["in_moves"][i]]

    def get(key, place):
        return c.get(f"{key}.{place}", c.get(key))

    if moving_in:
        c["noc.read_latency"] = max(get("noc.read_latency", p) or 0.0 for p in moving_in)
        c["_rd_service"] = tuple(get("noc.read_service", p) or 0.0 for p in moving_in)
        thr = [get("noc.thread_bytes_per_cycle", p) for p in moving_in]
        if any(thr):
            c["noc.thread_bytes_per_cycle"] = min(t for t in thr if t)
    if routing["out_moves"]:
        c["noc.write_latency"] = get("noc.write_latency", pl[2]) or 0.0
        c["noc.write_service"] = get("noc.write_service", pl[2]) or 0.0
    tb = routing["tile_bytes"]
    parts = [(tb[i], get("noc.cluster_rw_shared_bytes_per_cycle", pl[i])) for i in range(2) if routing["in_moves"][i]]
    if routing["out_moves"]:
        parts.append((tb[2], get("noc.cluster_rw_shared_bytes_per_cycle", pl[2])))
    # An operand whose placement has no rate (DRAM on ttsim-perf) costs nothing on the shared port; dropping the
    # whole bound for it would hide the L1 operands' share in mixed layouts.
    if parts and any(bw for _, bw in parts):
        total = sum(b for b, _ in parts)
        c["noc.cluster_rw_shared_bytes_per_cycle"] = total / sum(b / bw for b, bw in parts if bw)
    return c


def evaluate_config(wl, cfg, arch, kernel, cost_profile, what_if, trace=False, bounds_only=False, scenarios=SCENARIOS):
    """All metrics for one resolved configuration. Pure function of its inputs (cached). bounds_only skips the
    pipeline: tune's internal pre-screen, never returned to the caller."""
    routing = routing_for(wl, cfg)
    routing["knob_limits_override"] = what_if.get("_knob_limits")
    errors, warnings, l1 = check(cfg, routing, arch, kernel)
    warnings = warnings + [f"hypothetical kernel change: {VARIANT_NOTE[k]}; the pinned kernels do not implement it, "
                           f"so no simulator run can confirm this estimate" for k in VARIANTS if k in cfg]
    res = {"config": dict(cfg), "label": config_label(cfg), "valid": not errors, "errors": errors,
           "warnings": warnings, "l1": l1}
    if errors:
        return res
    groups = split_work_to_cores(wl["clusters"], wl["num_tiles_total"])
    ghz = cost_profile["clock"]["GHz"]
    times, lbs, detail = {}, {}, None
    traffic = {n: cluster_traffic(cfg, routing, n) for n, _ in groups}
    for sc in scenarios:
        costs = effective_costs(resolve_costs(cost_profile, wl["variant"], what_if.get("cost_overrides"), sc,
                                              what_if.get("compute_service_scale", 1.0)), routing)
        worst, worst_group, worst_lb = -1.0, None, 0.0
        per_group = []
        for n, ncores in groups:
            tr = traffic[n]
            bnd = cluster_bounds(cfg, routing, tr, costs)
            if bounds_only:
                pipe = None
                t = max(v["finish_lb"] for v in bnd.values()) + costs.get("end.drain", 0.0)
            else:
                try:
                    pipe = eval_cluster(cfg, routing, costs, n, EXACT_TILES, trace=16 if trace else 0)
                except Deadlock as e:
                    res.update(valid=False, errors=[f"deadlock: {e}"])
                    return res
                t = pipe["makespan"]
            # Shared bandwidth inside the cluster is not arbitrated by the recurrence: apply as a bound.
            bw_keys = [k for k in bnd if k.startswith("noc_") or k == "l1_ports"]
            bw_finish = max((bnd[k]["finish_lb"] for k in bw_keys), default=0.0) + costs.get("end.drain", 0.0)
            lb = max(v["finish_lb"] for v in bnd.values())
            t_final = max(t, bw_finish)
            per_group.append({"tiles_per_cluster": n, "clusters": ncores, "time": t_final, "pipeline": pipe,
                              "bounds": bnd, "traffic": tr, "bw_bound_binds": bw_finish > t})
            if t_final > worst:
                worst, worst_group, worst_lb = t_final, per_group[-1], lb
        dram = global_dram_bound(sum(g["traffic"]["dram_bytes"] * g["clusters"] for g in per_group), costs)
        if dram:
            worst = max(worst, dram["finish_lb"] + costs.get("end.drain", 0.0))
            worst_lb = max(worst_lb, dram["finish_lb"])
        times[sc] = worst
        lbs[sc] = worst_lb
        if sc == "nominal":
            detail = (worst_group, dram)
    g, dram = detail
    n_total = wl["num_tiles_total"]
    t_ns = {k: v / ghz for k, v in times.items()}
    res.update({
        "time_ns": {k: round(v, 1) for k, v in t_ns.items()},
        "time_cycles": round(times["nominal"], 1),
        "lower_bound_ns": round(lbs["nominal"] / ghz, 1),
        "per_tile_ns": round(t_ns["nominal"] / n_total, 3),
        "throughput_tiles_per_us": round(n_total / (t_ns["nominal"] / 1000.0), 3) if t_ns["nominal"] else None,
        "critical_cluster": {"tiles": g["tiles_per_cluster"], "clusters_with_this_share": g["clusters"]},
        "work_split": [{"tiles_per_cluster": n, "clusters": c} for n, c in groups],
    })
    res["bottleneck"], res["near_binding"] = _bottleneck(g, dram, times["nominal"])
    if g["pipeline"]:
        p = g["pipeline"]
        span = p.get("makespan_sampled", p["makespan"])
        roles = {}
        for name, r in p["roles"].items():
            frac = lambda key: round(r[key] / (r["threads"] * span), 3) if span else None  # noqa: E731
            roles[name] = {"threads": r["threads"], "useful_busy_frac": frac("busy"), "starved_frac": frac("wait_data"),
                           "backpressure_frac": frac("wait_space"), "transfer_wait_frac": frac("wait_transfer"),
                           "start_ns": round(r["start_min"] / ghz, 1)}
        res["roles"] = roles
        res["extrapolated"] = p.get("extrapolated", False)
        if p.get("extrapolated"):
            res["steady_state_ns_per_tile_per_cluster"] = round(p["slope"] / ghz, 3)
        if "threads" in p:
            res["threads"] = p["threads"]
        res["result_kind"] = "finite_buffer_estimate"
    else:
        res["roles"] = None
        res["result_kind"] = "lower_bound"
    tr = g["traffic"]
    res["traffic_per_cluster"] = {k: tr[k] for k in ("tiles", "noc_read_requests", "noc_write_requests",
                                                       "noc_read_bytes", "noc_write_bytes", "dram_bytes",
                                                       "l1_logical_bytes", "reader_barriers", "writer_barriers",
                                                       "compute_dfb_ops")}
    res["bounds_ns"] = {k: round(v["finish_lb"] / ghz, 1) for k, v in g["bounds"].items()}
    if dram:
        res["bounds_ns"]["dram_global"] = round(dram["finish_lb"] / ghz, 1)
    return res


def _bottleneck(g, dram, t):
    cands = {k: v["finish_lb"] for k, v in g["bounds"].items()}
    if dram:
        cands["dram_global"] = dram["finish_lb"]
    top = max(cands, key=cands.get)
    near = sorted([k for k, v in cands.items() if k != top and v >= 0.9 * cands[top]], key=lambda k: -cands[k])
    gap = (t - cands[top]) / t if t else 0.0
    b = {"resource": top, "bound_share_of_estimate": round(cands[top] / t, 3) if t else None}
    if gap > 0.15:
        b["note"] = (f"estimate exceeds the largest resource bound by {gap:.0%}: time is lost to dependencies "
                     f"(fill/drain, batch publication, finite rings), not to one saturated resource")
    p = g.get("pipeline")
    if p:
        worst = max(p["roles"].items(), key=lambda kv: kv[1]["busy"] / kv[1]["threads"])
        b["busiest_role"] = worst[0]
        span = p.get("makespan_sampled", p["makespan"])
        waits = []
        for role, r in p["roles"].items():
            for kind, key in (("starved", "wait_data"), ("backpressure", "wait_space"), ("transfer_latency", "wait_transfer")):
                waits.append((r[key] / (r["threads"] * span) if span else 0.0, role, kind))
        frac, role, kind = max(waits)
        if frac > 0.05:
            b["dominant_wait"] = {"role": role, "kind": kind, "fraction": round(frac, 3)}
    return b, near


# ----------------------------------------------------------------------------------------------- actions

def _cached_eval(wl, cfg, ctx, bounds_only=False):
    key = sha256({"wl": {k: wl[k] for k in ("variant", "num_tiles_total", "clusters", "placements_list", "dtype")},
                  "cfg": cfg, "trace": ctx["trace"], "bo": bounds_only, "wi": ctx["what_if"], "ph": ctx["profile_hashes"], "v": ENGINE_VERSION})
    hit = _cache.get(key)
    if hit is not None:
        ctx["cache_hits"] += 1
        return copy.deepcopy(hit)
    r = evaluate_config(wl, cfg, ctx["arch"], ctx["kernel"], ctx["costs"], ctx["what_if"], ctx["trace"], bounds_only)
    _cache.put(key, copy.deepcopy(r))
    return r


def _compare_rows(base, cand):
    if not (base.get("valid") and cand.get("valid")):
        return None
    sp = {sc: base["time_ns"][sc] / cand["time_ns"][sc] for sc in base["time_ns"]}
    lo, hi = min(sp.values()), max(sp.values())
    if lo > 1 + NOISE:
        verdict = "faster"
    elif hi < 1 - NOISE:
        verdict = "slower"
    elif abs(sp["nominal"] - 1) <= NOISE and lo >= 1 - NOISE and hi <= 1 + NOISE:
        verdict = "no_meaningful_gain"
    else:
        verdict = "inconclusive"
    return {"speedup_vs_baseline": {k: round(v, 3) for k, v in sp.items()}, "verdict": verdict}


def _compact(r, detail):
    if detail == "full":
        return r
    keep = ("label", "config", "valid", "errors", "warnings", "result_kind", "time_ns", "lower_bound_ns",
            "per_tile_ns", "bottleneck", "near_binding", "roles", "l1", "extrapolated", "comparison", "id",
            "steady_state_ns_per_tile_per_cluster", "bounds_ns", "structure")
    out = {k: r[k] for k in keep if k in r}
    if "l1" in out:
        out["l1"] = {"ring_bytes": r["l1"]["ring_bytes"], "fits": r["l1"]["fits"]}
    return out


def analyze(request):
    t_start = time.perf_counter()
    try:
        return _analyze(request, t_start)
    except RequestError as e:
        return _status(e.status, str(e), t_start)
    except ProfileError as e:
        return _status(e.code, str(e), t_start)


def _status(status, msg, t_start):
    return {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": status,
            "result_kind": None, "message": msg, "runtime_ms": round((time.perf_counter() - t_start) * 1e3, 2)}


def _analyze(req, t_start):
    errs = validate_request(req)
    if errs:
        raise RequestError("invalid", "; ".join(errs))
    from .layout_api import analyze_layout, layout_ops

    if req["workload"]["op"] in layout_ops():
        return analyze_layout(req, t_start)
    action = req["action"]
    if action == "survey":
        raise RequestError("unsupported", f"survey compares approaches over the hypothetical layout design knobs of "
                                          f"tilize Default and untilize MultiCore across shapes; {req['workload']['op']} "
                                          f"(binary_ng) has none. A binary_ng survey over a num_tiles axis is not modeled "
                                          f"yet (use sensitivity with param num_tiles).")
    prof = req.get("profiles", {})
    arch, arch_id = load("architecture", prof.get("architecture", "quasar"))
    kernel, kernel_id = load("kernel", prof.get("kernel", "binary-ng-native"))
    if "costs" not in prof:
        raise RequestError("needs_input", "profiles.costs is required: a cost profile says which platform the "
                                          "times describe (e.g. ttsim-perf, craq-sim, silicon-hypothesis)")
    costs, costs_id = load("costs", prof["costs"])
    analysis = req.get("analysis", {})
    detail = analysis.get("detail", "compact")
    budget_ms = analysis.get("time_budget_ms", 250 if action != "tune" else 20000)
    what_if = dict(req.get("what_if", {}))
    hyp = req.get("hypothetical") or {}
    if hyp.get("knob_limits"):
        what_if["_knob_limits"] = hyp["knob_limits"]
    wl, applied = resolve_workload(req["workload"], arch)
    ctx = {"arch": arch, "kernel": kernel, "costs": costs, "what_if": what_if, "cache_hits": 0, "trace": detail == "full",
           "profile_hashes": [arch_id["sha256"], kernel_id["sha256"], costs_id["sha256"]]}
    base_knobs = dict(req.get("baseline") or {})
    base_cfg, base_applied = resolve_config(wl, base_knobs)
    applied += [f"baseline {a}" for a in base_applied]
    resolved_input = {"action": action, "workload": {k: wl[k] for k in ("op", "dtype", "num_tiles_total", "clusters",
                                                                        "placements_list", "variant")},
                      "baseline": base_cfg, "what_if": what_if,
                      "profiles": {"architecture": arch_id, "kernel": kernel_id, "costs": costs_id}}
    result = {"schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION, "status": "ok",
              "profiles": resolved_input["profiles"], "applied_defaults": applied,
              "assumptions": _assumptions(costs, what_if), "warnings": []}
    if costs.get("platform_scope"):
        result["platform_scope"] = costs["platform_scope"]
    if hyp:
        result["hypothetical"] = hyp
        result["assumptions"].append(f"HYPOTHETICAL program change, not the pinned factory: {hyp}")

    base = _cached_eval(wl, base_cfg, ctx)
    result["result_kind"] = base.get("result_kind")
    if action == "estimate":
        result["baseline"] = _compact(base, detail)
        if detail == "full":
            result["parameters"] = param_provenance(costs, wl["variant"], sorted(resolve_costs(
                costs, wl["variant"], what_if.get("cost_overrides")).keys()))
    elif action == "compare":
        result["baseline"] = _compact(base, detail)
        rows = []
        for c in req.get("candidates", []):
            cfg, _ = resolve_config(wl, dict(base_knobs, **c.get("overrides", {})))
            r = _cached_eval(wl, cfg, ctx)
            r["id"] = c.get("id", config_label(cfg))
            r["comparison"] = _compare_rows(base, r)
            rows.append(_compact(r, detail))
        result["candidates"] = rows
        result["recommendations"] = _recommend(rows, analysis.get("top_k", 3))
    elif action == "tune":
        result.update(_tune(wl, base_knobs, base, ctx, req.get("tune", {}), analysis, detail, budget_ms, t_start))
        if result.get("search", {}).get("exhaustive") is False:
            result["status"] = "partial"
    elif action == "sensitivity":
        result.update(_sensitivity(req, wl, base_knobs, ctx, budget_ms, t_start))
        result["baseline"] = _compact(base, detail)
    if not base.get("valid"):
        result["status"] = "invalid"
        result["message"] = "baseline configuration is invalid: " + "; ".join(base["errors"])
    result["resolved_input_sha256"] = sha256(resolved_input)
    if detail == "full":
        result["resolved_input"] = resolved_input
    result["cache_hits"] = ctx["cache_hits"]
    result["runtime_ms"] = round((time.perf_counter() - t_start) * 1e3, 2)
    return result


def _assumptions(costs, what_if):
    a = list(costs.get("assumptions", []))
    a.append("fixed schedule from the kernels: no dynamic arbitration; shared NoC/L1/DRAM bandwidth applied as bounds "
             "(full-overlap rule), not as contention in the recurrence")
    a.append("time ranges are scenarios (all costs at low / nominal / high), shared by baseline and candidates; "
             "not statistical confidence intervals")
    shown = {k: v for k, v in what_if.items() if not k.startswith("_")}
    if shown:
        a.append(f"what-if applied (not a measurement): {shown}")
    return a


def _recommend(rows, top_k):
    good = [r for r in rows if r.get("comparison") and r["comparison"]["verdict"] == "faster"]
    good.sort(key=lambda r: r["time_ns"]["nominal"])
    out = []
    for r in good[:top_k]:
        out.append({"id": r["id"], "label": r["label"], "speedup_range": [r["comparison"]["speedup_vs_baseline"]["low"],
                                                                            r["comparison"]["speedup_vs_baseline"]["high"]],
                    "speedup_nominal": r["comparison"]["speedup_vs_baseline"]["nominal"],
                    "bottleneck_after": r["bottleneck"]["resource"]})
    if not out:
        out.append({"id": None, "label": "keep baseline",
                    "reason": "no candidate is faster in every cost scenario by more than "
                              f"{NOISE:.0%}; differences are inside the model's uncertainty"})
    return out


DEFAULT_SPACE = {"reader_threads": [1, 2, 4], "compute_threads": [1, 2, 4], "writer_threads": [1, 2, 4],
                 "dm_batch": [1, 2, 4, 8], "tiles_per_cycle": [1, 2, 4, 8], "entries_per_thread": [None]}


def _simplicity(cfg):
    return (sum(k in cfg for k in VARIANTS), cfg["R"] + cfg["C"] + cfg["W"], cfg["dm_batch"] + cfg["tiles_per_cycle"],
            cfg["entries_per_thread"])


def _tune(wl, base_knobs, base, ctx, spec, analysis, detail, budget_ms, t_start):
    space = dict(DEFAULT_SPACE)
    space.update(spec.get("space", {}))
    max_c = spec.get("max_candidates")
    top_k = analysis.get("top_k", 3)
    keys = list(space)
    seen, valid = set(), []
    enumerated = invalid = 0
    for combo in itertools.product(*[space[k] for k in keys]):
        knobs = dict(base_knobs, **{k: v for k, v in zip(keys, combo) if v is not None})
        cfg, _ = resolve_config(wl, knobs)
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        enumerated += 1
        # 1) validity first: cheap, and an invalid config must not use the candidate budget
        rt = routing_for(wl, cfg)
        rt["knob_limits_override"] = ctx["what_if"].get("_knob_limits")
        errs, _, _ = check(cfg, rt, ctx["arch"], ctx["kernel"])
        if errs:
            invalid += 1
            continue
        valid.append(cfg)
    valid.sort(key=_simplicity)
    truncated = max_c is not None and len(valid) > max_c
    cands = valid[:max_c]
    # 2) bound-based estimate for every candidate: only the ORDER of the pipeline evaluations, so a budget stop
    # keeps the most promising ones
    scored = []
    for cfg in cands:
        r = _cached_eval(wl, cfg, ctx, bounds_only=True)
        if r["valid"]:
            scored.append((r["time_ns"]["nominal"], _simplicity(cfg), cfg))
    scored.sort(key=lambda x: (x[0], x[1]))
    # 3) the pipeline model for every candidate, the baseline first, then by bound
    to_refine = [c for _, _, c in scored]
    if base.get("valid"):
        to_refine = [base["config"]] + [c for c in to_refine if c != base["config"]]
    evaluated, out_of_time = [], False
    for cfg in to_refine:
        if (time.perf_counter() - t_start) * 1e3 > budget_ms:
            out_of_time = True
            break
        r = _cached_eval(wl, cfg, ctx)
        if not r["valid"]:
            continue
        r["id"] = config_label(cfg)
        r["comparison"] = _compare_rows(base, r)
        evaluated.append(r)
    evaluated.sort(key=lambda r: (r["time_ns"]["nominal"], _simplicity(r["config"])))
    picks = []
    if evaluated:
        # among configs indistinguishable from the best, prefer the simpler one
        best = evaluated[0]["time_ns"]["nominal"]
        tied = sorted([r for r in evaluated if r["time_ns"]["nominal"] <= best * (1 + NOISE)],
                      key=lambda r: _simplicity(r["config"]))
        picks = (tied + [r for r in evaluated if r not in tied])[:top_k]
    return {
        "baseline": _compact(base, detail),
        "top": [_compact(r, detail) for r in picks],
        "search": {"enumerated": enumerated, "invalid": invalid, "valid": len(valid), "considered": len(cands),
                   "evaluated": len(evaluated), "truncated_by_max_candidates": truncated, "out_of_time": out_of_time,
                   "exhaustive": not truncated and not out_of_time,
                   "order": "every valid candidate gets the pipeline model, baseline first, then by bound-based "
                            "estimate; a budget stop (out_of_time) leaves the rest unevaluated, not proven worse",
                   "evaluated_ids": [r["id"] for r in evaluated]},
        "recommendations": _recommend([_compact(r, "compact") for r in evaluated], top_k),
    }


def _sensitivity(req, wl, base_knobs, ctx, budget_ms, t_start):
    spec = req.get("sensitivity") or {}
    param = spec.get("param")
    values = spec.get("values") or []
    if not param or not values:
        raise RequestError("needs_input", "sensitivity needs {param, values}")
    configs = [("baseline", base_knobs)]
    for c in req.get("candidates", []):
        configs.append((c.get("id", str(c.get("overrides"))), dict(base_knobs, **c.get("overrides", {}))))
    series = {cid: [] for cid, _ in configs}
    partial = False
    for v in values:
        if (time.perf_counter() - t_start) * 1e3 > budget_ms:
            partial = True
            break
        wi = copy.deepcopy(ctx["what_if"])
        if param == "compute_service_scale":
            wi["compute_service_scale"] = v
        elif param != "num_tiles":
            wi.setdefault("cost_overrides", {})[param] = v
        sub = dict(ctx, what_if=wi)
        w2 = dict(wl, num_tiles_total=int(v)) if param == "num_tiles" else wl
        for cid, knobs in configs:
            cfg, _ = resolve_config(w2, knobs)
            r = _cached_eval(w2, cfg, sub)
            series[cid].append({"value": v, "time_ns": r.get("time_ns", {}).get("nominal") if r["valid"] else None,
                                "bottleneck": r.get("bottleneck", {}).get("resource"), "valid": r["valid"]})
    crossings, shifts = _ranking(series, param)
    return {"sensitivity": {"param": param, "series": series, "ranking_changes": crossings,
                            "bottleneck_shifts": shifts, "partial": partial}}


def _ranking(series, param):
    """Where two series swap order (beyond NOISE) and where a series' bottleneck changes."""
    crossings = []
    for a, b in itertools.combinations(series, 2):
        prev = None
        for pa, pb in zip(series[a], series[b]):
            if pa["time_ns"] is None or pb["time_ns"] is None:
                continue
            s = (pa["time_ns"] < pb["time_ns"] * (1 - NOISE)) - (pb["time_ns"] < pa["time_ns"] * (1 - NOISE))
            if prev is not None and s != 0 and prev != 0 and s != prev:
                crossings.append({"between": [a, b], "param": param, "at_about": pa["value"]})
            if s != 0:
                prev = s
    shifts = []
    for cid, pts in series.items():
        for p0, p1 in zip(pts, pts[1:]):
            if p0["bottleneck"] != p1["bottleneck"]:
                shifts.append({"config": cid, "from": p0["bottleneck"], "to": p1["bottleneck"], "at_about": p1["value"]})
    return crossings, shifts
