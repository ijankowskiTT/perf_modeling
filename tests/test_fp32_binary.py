"""fp32 binary_ng add/multiply: the fitted compute variants (calibration/out/fp32binary-<profile>.json), overlaid on
the cost profile, give estimates within the calibration targets on calibration/data/fp32binary-<profile>.jsonl."""

import copy
import json
import os

import pytest

from qperf import profiles as P
from qperf.api import analyze
from qperf.calibration import load_records, predict, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMS = {"craq-sim": ("craq-b702fd90", 0.02), "ttsim-perf": ("perf", 0.03)}  # profile: (sim tag, holdout target)


def overlaid(profile):
    base, _ = P.load("costs", profile)
    prof = copy.deepcopy(base)
    prof["name"] = profile + "__fp32test"
    with open(os.path.join(ROOT, "calibration", "out", f"fp32binary-{profile}.json")) as f:
        prof["compute_variants"].update(json.load(f)["compute_variants"])
    P.register("costs", prof)
    return prof["name"]


def request(costs, op, placement, knobs=None, tiles=64):
    return {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
            "workload": {"op": op, "dtype": "fp32", "num_tiles": tiles, "tiles_scope": "cluster", "clusters": 32,
                         "placements": {"a": placement, "b": placement, "out": placement}},
            "baseline": knobs or {}, "analysis": {"detail": "compact"}}


@pytest.mark.parametrize("profile", sorted(SIMS))
@pytest.mark.parametrize("op", ["add", "multiply"])
def test_fp32_estimate_succeeds_with_the_overlay(profile, op):
    name = overlaid(profile)
    try:
        for placement in ("borrowed", "l1", "dram"):
            res = analyze(request(name, op, placement))
            assert res["status"] == "ok", res
            assert res["baseline"]["valid"] and res["baseline"]["time_ns"]["nominal"] > 0
    finally:
        P.unregister("costs", name)


def test_fp32_rule_and_dst_capacity():
    # The factory logs tiles_per_cycle=2 for the SFPU kernel in a 32-bit DST and refuses 4 (TT_FATAL).
    name = overlaid("craq-sim")
    try:
        res = analyze(request(name, "add", "l1"))
        cfg = res["baseline"]["config"]
        assert (cfg["R"], cfg["C"], cfg["W"], cfg["dm_batch"], cfg["tiles_per_cycle"], cfg["entries_per_thread"]) == \
            (4, 4, 2, 8, 2, 16)
        assert analyze(request(name, "add", "borrowed"))["baseline"]["config"]["tiles_per_cycle"] == 2
        bad = analyze(request(name, "multiply", "borrowed", {"compute_threads": 4, "tiles_per_cycle": 4}))
        assert not bad["baseline"]["valid"]
        assert any("DST section capacity 2" in e for e in bad["baseline"]["errors"])
    finally:
        P.unregister("costs", name)


@pytest.mark.parametrize("profile", sorted(SIMS))
@pytest.mark.parametrize("op", ["add", "multiply"])
def test_fp32_accuracy_on_the_calibration_data(profile, op):
    sim, target = SIMS[profile]
    recs = [r for r in load_records(os.path.join(ROOT, "calibration", "data", f"fp32binary-{profile}.jsonl"), sim)
            if r["op"] == op]
    assert recs and all(r["max_abs_err"] <= 1e-3 for r in recs)  # fp32 outputs are exact to fp32 tolerance
    name = overlaid(profile)
    try:
        rows = predict(recs, name)
    finally:
        P.unregister("costs", name)
    assert all(r["predicted"] for r in rows)
    hold = summarize([x for x, r in zip(rows, recs) if r["src"] == "B"])
    assert hold["mean_abs_rel_err"] <= target, hold
    # ttsim-perf times L1 transfers, and its L1 runs are a known gap (bf16 L1: 10% mean): checked on their own.
    l1_timed = profile == "ttsim-perf"
    fit = summarize([x for x, r in zip(rows, recs) if r["src"] != "B" and not (l1_timed and r["layout"] == "lll")])
    assert fit["mean_abs_rel_err"] <= target, fit
    if l1_timed:
        l1 = summarize([x for x, r in zip(rows, recs) if r["layout"] == "lll"])
        assert l1["mean_abs_rel_err"] <= 0.08 and l1["max_abs_rel_err"] <= 0.16, l1
