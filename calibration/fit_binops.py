"""Calibrate the compute variants of the binary_ng ops beyond add / multiply (subtract, divide, and multiply on the
FPU kernel with fast_and_approximate_mode) on one simulator's data (calibration/data/binops-<profile>.jsonl).

The native factory runs one reader, compute and writer kernel for every op: only the compute kernel's op (the FPU
math instruction, or the SFPU tile op BINARY_SFPU_OP and its init) changes. Each new variant has a calibrated variant
on the same kernel and dtype (SEEDS):
- SAME: the op runs cycle for cycle as the seed's (subtract as add; bf16 multiply on the FPU as add). The variant is
  written as {"_same_as": seed}: it resolves to the seed's terms, so its predictions equal add's on the same
  configuration, as the simulators measure. The script checks it: every run of the variant whose configuration
  (layout, tiles, knobs) the seed op also ran (calibration/data/{,fp32binary-}<profile>.jsonl) must have the same
  span; a mismatch fits the variant instead.
- otherwise a copy of the seed with its math TRISC terms (cm.fixed, cm.per_tile) fitted, plus --extra-keys (e.g.
  cu.first_chunk_extra: divide's SFPU init before the first tile). Unpack, pack and every data-movement term stay the
  seed's: add / multiply terms are never changed.

Fit runs: every `src` except the holdout `B` (every 4th run). On ttsim-perf the compute keys are fitted on the DRAM /
borrowed runs (--compute-layouts ddd,hhh): it times L1 transfers, whose known gap (add L1 ~8%) is not a compute term.

    python calibration/fit_binops.py --data calibration/data/binops-craq-sim.jsonl --sim craq-b702fd90 \
        --costs craq-sim --write calibration/out/binops-craq-sim.json            # scipy: tt-metal venv
    # ttsim-perf: divide's SFPU init delays the first tile by ~3000 cycles (cu.first_chunk_extra); craq-sim shows none
    python calibration/fit_binops.py --data calibration/data/binops-ttsim-perf.jsonl --sim perf \
        --costs ttsim-perf --compute-layouts ddd,hhh \
        --extra-keys sfpu:divide:bf16=cu.first_chunk_extra,sfpu:divide:fp32=cu.first_chunk_extra \
        --write calibration/out/binops-ttsim-perf.json
    python calibration/merge_params.py calibration/out/binops-craq-sim.json craq-sim   # into compute_variants
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fit_fp32_binary import fit_variant, overlay  # noqa: E402
from qperf import profiles as P  # noqa: E402
from qperf.calibration import load_records, predict, summarize  # noqa: E402
from qperf.rule import native_routing  # noqa: E402

# new variant -> the calibrated variant of the same kernel and dtype
SEEDS = {"fpu:subtract:bf16": "fpu:add:bf16", "fpu:multiply:bf16": "fpu:add:bf16",
         "sfpu:divide:bf16": "sfpu:multiply:bf16", "sfpu:subtract:fp32": "sfpu:add:fp32",
         "sfpu:divide:fp32": "sfpu:multiply:fp32"}
SAME = ("fpu:subtract:bf16", "fpu:multiply:bf16", "sfpu:subtract:fp32")
KEYS = ["cm.fixed", "cm.per_tile"]
KERNEL = {"fpu": "eltwise_binary_no_bcast_dfb.cpp (FPU)", "sfpu": "eltwise_binary_sfpu_no_bcast_dfb.cpp (SFPU)"}
PAIR_DATA = ("{}.jsonl", "fp32binary-{}.jsonl")  # the seed ops' runs (add, multiply), per profile


def variant_of(rec):
    sf, why = native_routing(rec["op"], rec["dtype"], rec.get("kernel_variant"))
    assert why is None, why
    return f"{'sfpu' if sf else 'fpu'}:{rec['op']}:{rec['dtype']}"


def evaluate(base, recs, variant, v):
    name = base["name"] + "__binops_eval"
    overlay(base, {variant: v}, name)
    try:
        return predict(recs, name)
    finally:
        P.unregister("costs", name)


def split(rows, recs, pred):
    return summarize([r for r, rec in zip(rows, recs) if pred(rec)])


def pairs(recs, seed, profile, sim):
    """(identical, differing) runs: the variant's runs whose layout, tiles and knobs the seed op also ran."""
    sop, sdt = seed.split(":")[1:]
    seen = {}
    for pat in PAIR_DATA:
        path = os.path.join(ROOT, "calibration", "data", pat.format(profile))
        for r in load_records(path, sim):
            if r["op"] == sop and r["dtype"] == sdt and not r.get("kernel_variant"):
                seen[(r["layout"], r["tpc"], r["knobspec"])] = r["span_med"]
    same, diff = 0, []
    for r in recs:
        s = seen.get((r["layout"], r["tpc"], r["knobspec"]))
        if s is None:
            continue
        if abs(s / r["span_med"] - 1) <= 0.005:
            same += 1
        else:
            diff.append((r["layout"], r["tpc"], r["knobspec"], r["span_med"], s))
    return same, diff


def describe(variant):
    kind, op, dtype = variant.split(":")
    fast = " with fast_and_approximate_mode" if kind == "fpu" and op == "multiply" else ""
    return f"{KERNEL[kind]}: {op}{fast} {dtype}"


def emit_same(variant, seed, n_same, n_runs, sim, data_name):
    return {"_same_as": seed,
            "_source": f"{describe(variant)}; runs cycle for cycle as {seed}: {n_same} of {n_runs} runs in "
                       f"{data_name} ({sim}) repeat a configuration of the {seed.split(':')[1]} calibration data, "
                       f"every one with the same span, so the variant uses {seed}'s terms"}


def emit_fit(v, seed_name, seed, fitted, variant, sim, n_fit, hold_mean, rng, data_name):
    today = time.strftime("%Y-%m-%d")
    out = {"_source": f"{describe(variant)}; the kernels of {seed_name} with only the op changed, so its unpack / "
                      f"pack terms are copied and {', '.join(fitted)} fitted"}
    src = f"fitted {today} on {data_name} ({sim}, {n_fit} {variant} runs; B-holdout {hold_mean:.1%} mean)"
    for k, p in v.items():
        if k.startswith("_"):
            continue
        if k in fitted:
            val = p["value"]
            out[k] = {"value": val, "low": round(val * (1 - rng), 3), "high": round(val * (1 + rng), 3),
                      "class": "measurement-fit", "source": src,
                      "range_note": f"+-{rng:.0%} scenario range (holdout p90 error)"}
        else:
            out[k] = dict(seed[k], source=f"copied from {seed_name}: " + seed[k].get("source", ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--variants", default=",".join(SEEDS))
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--extra-keys", default="", help="variant=key1+key2,... fitted beside --keys for that variant")
    ap.add_argument("--start", default="sfpu:divide:bf16/cm.per_tile=1000,sfpu:divide:fp32/cm.per_tile=1000",
                    help="variant/key=value,...: start values (a math term below the unpack stage has no gradient)")
    ap.add_argument("--compute-layouts", default="ddd,lll,hhh")
    ap.add_argument("--write")
    a = ap.parse_args(argv)

    base, _ = P.load("costs", a.costs)
    keys = a.keys.split(",")
    extra, start = {}, {}
    for part in filter(None, a.extra_keys.split(",")):
        var, _, ks = part.partition("=")
        extra[var] = ks.split("+")
    for part in filter(None, a.start.split(",")):
        vk, _, val = part.partition("=")
        var, _, key = vk.partition("/")
        start.setdefault(var, {})[key] = float(val)
    allrecs = load_records(a.data, a.sim)
    lays = a.compute_layouts.split(",")
    data_name = os.path.basename(a.data)
    variants, report = {}, {}
    for variant in a.variants.split(","):
        seed_name = SEEDS[variant]
        seed = base["compute_variants"][seed_name]
        recs = [r for r in allrecs if variant_of(r) == variant]
        if not recs:
            continue
        fit_set = [r for r in recs if r.get("src") != "B"]
        hold = [r for r in recs if r.get("src") == "B"]
        v = {k: copy.deepcopy(p) for k, p in seed.items()}
        n_same, diff = pairs(recs, seed_name, a.costs, a.sim)
        same = variant in SAME and n_same > 0 and not diff
        fk = [] if same else keys + extra.get(variant, [])
        if not same:
            for k, val in start.get(variant, {}).items():
                v[k] = dict(v.get(k, {}), value=val)
            v, _ = fit_variant(base, [r for r in fit_set if r["layout"] in lays], variant, v, fk)
        rows_f, rows_h = evaluate(base, fit_set, variant, v), evaluate(base, hold, variant, v)
        s_h = summarize(rows_h)
        if same:
            variants[variant] = emit_same(variant, seed_name, n_same, len(recs), a.sim, data_name)
        else:
            rng = max(0.03, round(s_h.get("p90_abs_rel_err", 0.05) + 0.005, 2)) if hold else 0.05
            variants[variant] = emit_fit(v, seed_name, seed, fk, variant, a.sim, len(fit_set),
                                         s_h.get("mean_abs_rel_err", float("nan")), rng, data_name)
        report[variant] = {
            "seed": seed_name, "same_as_seed": same, "paired_runs": {"same_span": n_same, "differing": diff},
            "all": summarize(rows_f + rows_h), "fit": summarize(rows_f), "holdout": s_h,
            "by_layout": {lay: {"fit": split(rows_f, fit_set, lambda r, lay=lay: r["layout"] == lay),
                                "holdout": split(rows_h, hold, lambda r, lay=lay: r["layout"] == lay)}
                          for lay in sorted({r["layout"] for r in recs})},
            "values": {k: v[k]["value"] for k in fk},
            "worst": sorted(((round(r["rel_err"], 4), r["case"]) for r in rows_f + rows_h if r["rel_err"] is not None),
                            key=lambda t: -abs(t[0]))[:6]}
    print(json.dumps(report, indent=1))
    if a.write:
        os.makedirs(os.path.dirname(a.write) or ".", exist_ok=True)
        with open(a.write, "w") as f:
            json.dump({"compute_variants": variants}, f, indent=1)
            f.write("\n")
        print(a.write)


if __name__ == "__main__":
    main()
