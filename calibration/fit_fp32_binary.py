"""Fit the fp32 binary_ng compute variants (sfpu:add:fp32, sfpu:multiply:fp32) on one simulator's data.

`qperf calibrate fit` refits keys of a variant the profile already has; a new variant needs a seed, so this script
copies the bf16 SFPU variant (sfpu:multiply:bf16), registers the profile in memory (nothing under profiles/ is
written), fits the compute keys by least squares on log(predicted / measured) over the fit runs (every `src` except
the holdout `B`) and reports fit and holdout errors. --write saves {"compute_variants": {...}} for the lead to merge.

fp32 runs batch at most 2 tiles (SFPU kernel, 32-bit DST), so a chunk holds 1 or 2 tiles: cu.fixed and cu.per_tile
are fitted and cu.fixed_n1 is dropped (two chunk sizes cannot separate three constants). Keys outside cu/cm/cp are
fp32-specific DM terms, written into the variant (a variant key overrides the shared one).

    # craq-sim: compute keys + the 4 KB write completion, on every layout (craq-sim times no L1/DRAM transfer)
    python calibration/fit_fp32_binary.py --data calibration/data/fp32binary-craq-sim.jsonl --sim craq-b702fd90 \
        --costs craq-sim --keys cu.fixed,cu.per_tile,cu.first_chunk_extra,cm.fixed,cm.per_tile,cp.fixed,cp.per_tile,\
noc.write_latency --start-compute 891 --write calibration/out/fp32binary-craq-sim.json   # scipy: tt-metal venv
    # ttsim-perf: compute keys on hhh/ddd (DRAM untimed), then the L1 transfer service on lll
    python calibration/fit_fp32_binary.py --data calibration/data/fp32binary-ttsim-perf.jsonl --sim perf \
        --costs ttsim-perf --compute-layouts hhh,ddd --l1-keys noc.read_service.l1,noc.write_service.l1 \
        --start-compute 1333 --write calibration/out/fp32binary-ttsim-perf.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qperf import profiles as P  # noqa: E402
from qperf.calibration import load_records, predict, summarize  # noqa: E402

SEED = "sfpu:multiply:bf16"
KEYS = ["cu.fixed", "cu.per_tile", "cu.first_chunk_extra", "cm.fixed", "cm.per_tile", "cp.fixed", "cp.per_tile"]
BOUNDS = {"cu.fixed": (0, 3000), "cu.per_tile": (0, 1000), "cu.first_chunk_extra": (0, 4000), "cm.fixed": (0, 3000),
          "cm.per_tile": (0, 2000), "cp.fixed": (0, 3000), "cp.per_tile": (0, 1000),
          "noc.read_service.l1": (0, 1000), "noc.write_service.l1": (0, 1000), "noc.read_latency.l1": (0, 5000),
          "noc.write_latency.l1": (0, 5000), "rd.tc_first_touch": (0, 2000), "wr.tc_first_touch": (0, 2000),
          "noc.read_latency": (0, 3000), "noc.write_latency": (0, 3000), "rd.issue": (0, 500), "wr.issue": (0, 500),
          "rd.fixed": (0, 1000), "wr.fixed": (0, 1000), "rd.fixed_b1": (0, 1000), "wr.fixed_b1": (0, 1000),
          "start.reader": (0, 8000), "start.writer": (0, 8000), "end.drain": (0, 1000),
          "noc.thread_bytes_per_cycle": (5, 5000), "noc.cluster_rw_shared_bytes_per_cycle.l1": (5, 2000)}
RIDGE = 0.02  # weight of a pull toward the seed (log space): keeps keys the data cannot separate near the bf16 values


def overlay(base, variants, name):
    prof = copy.deepcopy(base)
    prof["name"] = name
    prof.setdefault("compute_variants", {}).update(copy.deepcopy(variants))
    P.register("costs", prof)
    return prof


def errors(recs, name):
    rows = predict(recs, name)
    return rows, summarize(rows)


def fit_variant(base, recs, variant, v, keys):
    """Fit `keys` of variant dict v in place; a key the variant lacks starts from the profile's shared value (a
    variant key overrides the shared one in resolve_costs)."""
    from scipy.optimize import least_squares

    for k in keys:
        if k not in v:
            v[k] = {"value": float(P._pick(base["params"][k], "nominal")) if k in base["params"] else 100.0}
    name = base["name"] + "__fp32fit"
    # a zero start has no relative finite-difference step
    x0 = [float(v[k]["value"]) if float(v[k]["value"]) > 0 else 30.0 for k in keys]
    lo = [BOUNDS[k][0] for k in keys]
    hi = [BOUNDS[k][1] for k in keys]
    x0 = [min(max(x, a + 1e-6), b - 1e-6) for x, a, b in zip(x0, lo, hi)]

    def resid(x):
        for k, val in zip(keys, x):
            v[k]["value"] = float(val)
        overlay(base, {variant: v}, name)
        rows = predict(recs, name)
        r = [math.log(row["predicted"] / row["measured"]) if row["predicted"] else 1.0 for row in rows]
        r += [RIDGE * math.log((val + 10.0) / (x00 + 10.0)) for val, x00 in zip(x, x0)]
        return r

    sol = least_squares(resid, x0, bounds=(lo, hi), diff_step=1e-3, x_scale="jac", max_nfev=400)
    for k, val in zip(keys, sol.x):
        v[k]["value"] = round(float(val), 3)
    P.unregister("costs", name)
    return v, sol


DM_NOTE = ("keys outside cu/cm/cp are fp32-specific data-movement terms: a variant key overrides the shared one "
           "(resolve_costs), and the shared DM costs were fitted on 2 KB bf16 tiles")


def emit(v, fitted, variant, sim, n_fit, holdout, rng, data_name):
    today = time.strftime("%Y-%m-%d")
    op = variant.split(":")[1]
    src = f"fitted {today} on {data_name} ({sim}, {n_fit} {op} fp32 runs; B-holdout {holdout:.1%} mean)"
    out = {"_source": "eltwise_binary_sfpu_no_bcast_dfb.cpp with fp32: UnpackToDest copy_tile of both operands, "
                      "SFPU op on the math TRISC, 1 pack; at most 2 tiles per DST section (32-bit DST)"}
    if any(k.split(".")[0] not in ("cu", "cm", "cp", "start") for k in fitted):
        out["_dm_note"] = DM_NOTE
    for k, p in v.items():
        if k.startswith("_"):
            continue
        val = p["value"]
        if k == "start.compute":
            out[k] = {"value": val, "class": "measurement",
                      "source": f"TRISC0 KERNEL start, median of the fp32 {op} runs ({sim})"}
        elif k in fitted:
            out[k] = {"value": val, "low": round(val * (1 - rng), 3), "high": round(val * (1 + rng), 3),
                      "class": "measurement-fit", "source": src,
                      "range_note": f"+-{rng:.0%} scenario range (holdout p90 error)"}
        else:
            out[k] = dict(p, source=f"copied from {SEED}: " + p.get("source", ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--ops", default="add,multiply")
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--compute-layouts", default="hhh,ddd,lll",
                    help="layouts the compute keys are fitted on (ttsim-perf: hhh,ddd, as it times L1 transfers)")
    ap.add_argument("--l1-keys", default="",
                    help="second stage: fp32-specific transfer keys fitted on the lll runs with the compute keys fixed")
    ap.add_argument("--start-compute", type=float,
                    help="TRISC0 KERNEL start (cycles); measured medians: craq-sim 891, ttsim-perf 1333")
    ap.add_argument("--raw", help="raw sweep jsonl with role times, to measure start.compute instead")
    ap.add_argument("--write")
    a = ap.parse_args(argv)

    base, _ = P.load("costs", a.costs)
    seed = base["compute_variants"][SEED]
    keys = a.keys.split(",")
    allrecs = [r for r in load_records(a.data, a.sim) if r["dtype"] == "fp32"]
    variants, report = {}, {}
    for op in a.ops.split(","):
        variant = f"sfpu:{op}:fp32"
        recs = [r for r in allrecs if r["op"] == op]
        fit_set = [r for r in recs if r.get("src") != "B"]
        hold = [r for r in recs if r.get("src") == "B"]
        sc = a.start_compute if a.start_compute is not None else seed["start.compute"]["value"]
        if a.raw:
            starts = []
            for line in open(a.raw):
                r = json.loads(line)
                if r.get("error") or r.get("sim") != a.sim or r["op"] != op or r["dtype"] != "fp32":
                    continue
                starts += [t[0] for role, t in r["roles"].items() if role.endswith("TRISC0")]
            sc = round(statistics.median(starts))
        v = {k: copy.deepcopy(p) for k, p in seed.items() if k != "cu.fixed_n1"}
        v["start.compute"] = {"value": sc}
        lays = a.compute_layouts.split(",")
        v, sol = fit_variant(base, [r for r in fit_set if r["layout"] in lays], variant, v, keys)
        l1_keys = [k for k in a.l1_keys.split(",") if k]
        if l1_keys:
            v, sol = fit_variant(base, [r for r in fit_set if r["layout"] == "lll"], variant, v, l1_keys)
        name = base["name"] + "__fp32eval"
        overlay(base, {variant: v}, name)
        rows_f, s_f = errors(fit_set, name)
        rows_h, s_h = errors(hold, name)
        P.unregister("costs", name)
        rng = max(0.03, round(s_h.get("p90_abs_rel_err", 0.05) + 0.005, 2)) if hold else 0.05
        variants[variant] = emit(v, keys + l1_keys, variant, a.sim, len(fit_set), s_h.get("mean_abs_rel_err", float("nan")), rng,
                                 os.path.basename(a.data))
        by_layout = {lay: {"fit": summarize([r for r, rec in zip(rows_f, fit_set) if rec["layout"] == lay]),
                           "holdout": summarize([r for r, rec in zip(rows_h, hold) if rec["layout"] == lay])}
                     for lay in sorted({r["layout"] for r in recs})}
        report[variant] = {"fit": s_f, "holdout": s_h, "by_layout": by_layout,
                           "values": {k: v[k]["value"] for k in keys + l1_keys},
                           "start.compute": sc, "cost": round(float(sol.cost), 6),
                           "worst": sorted(((round(r["rel_err"], 4), r["case"]) for r in rows_f + rows_h
                                            if r["rel_err"] is not None), key=lambda t: -abs(t[0]))[:6]}
    print(json.dumps(report, indent=1))
    if a.write:
        os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
        with open(a.write, "w") as f:
            json.dump({"compute_variants": variants}, f, indent=1)
            f.write("\n")
        print(a.write)


if __name__ == "__main__":
    main()
