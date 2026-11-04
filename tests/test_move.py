"""move (ttnn.experimental.quasar.move): factory selection, API contract, and accuracy against the simulator runs in
calibration/data, with the terms calibration/fit_layout.py emitted (calibration/out/move-*.json) overlaid on the cost
profile (they are not merged into profiles/costs yet)."""

import copy
import json
import os
import statistics

import pytest

from qperf import analyze
from qperf.ops.layout import estimate, plan
from qperf.ops.move import multicast_regions, split_work
from qperf.profiles import _pick, load, register, unregister
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def overlaid(profile):
    prof = copy.deepcopy(load("costs", profile)[0])
    with open(os.path.join(ROOT, "calibration", "out", f"move-{profile}.json")) as f:
        prof["params"].update(json.load(f))
    return prof


@pytest.fixture
def move_profile():
    prof = overlaid("craq-sim")
    prof["name"] = "craq-sim-move-test"
    register("costs", prof)
    yield prof["name"]
    unregister("costs", prof["name"])


def req(costs, shape=(1, 1, 1024, 1024), place="dram", **kw):
    w = {"op": "move", "dtype": "bf16", "shape": list(shape), "placements": {"a": place}}
    w.update(kw.pop("workload", {}))
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs}, "workload": w}
    r.update(kw)
    return r


def test_factory_selection_and_split():
    assert plan("move", 1024, 1024, mem="dram_ov")[0] == "Overlap"
    assert plan("move", 1024, 512, mem="l1_ov_rm")[0] == "OverlapRowMajor"
    for mem in ("dram", "l1_rm", "hs8"):  # MoveProgramFactory (rejected on Quasar), MoveSharded
        with pytest.raises(ValueError, match="not modeled yet"):
            plan("move", 1024, 1024, mem=mem)
    with pytest.raises(ValueError, match="fits_in_cb"):
        plan("move", 32768, 8192, mem="dram_ov")
    # split_work_to_cores over the 8x4 grid: group 1 (with the controller) gets one more page
    assert split_work(33) == (32, [(2, 1), (1, 31)]) and split_work(1024) == (32, [(32, 32)])
    # multicast regions: 1 core none; one column: the cells below; full columns: + right; a partial column: + 1
    assert [multicast_regions(n) for n in (1, 4, 5, 8, 9, 32)] == [0, 1, 2, 2, 3, 2]
    fac, loads = plan("move", 1056, 32, mem="dram_ov")
    assert [(c["pages"], n) for c, n in loads] == [(2, 1), (1, 31)]


def test_estimate_conforms(move_profile):
    for place, layout in (("dram", "tile"), ("l1", "row_major")):
        r = analyze(req(move_profile, place=place, workload={"layout": layout}))
        assert r["status"] == "ok" and validate_response(r) == [], r
        t = r["baseline"]["time_ns"]
        assert 0 < t["low"] <= t["nominal"] <= t["high"]


def test_more_pages_is_slower(move_profile):
    r = analyze(req(move_profile, action="sensitivity", sensitivity={"param": "num_rows", "values": [256, 1024, 4096]}))
    assert validate_response(r) == []
    t = [p["time_ns"] for p in r["sensitivity"]["series"]["baseline"]]
    assert t == sorted(t)


@pytest.mark.parametrize("request_, status", [
    (req("craq-sim-move-test", workload={"output": "separate"}), "unsupported"),  # MoveProgramFactory: TT_FATAL
    (req("craq-sim-move-test", place="borrowed"), "invalid"),  # sharded move
])
def test_unsupported_and_missing_costs_carry_no_numbers(move_profile, request_, status):
    r = analyze(request_)
    assert r["status"] == status and "baseline" not in r and validate_response(r) == []


@pytest.mark.parametrize("profile,data,limit", [
    ("craq-sim", "move-craq-sim.jsonl", 0.025),
    ("ttsim-perf", "move-ttsim-perf.jsonl", 0.03),
])
def test_accuracy_per_factory(profile, data, limit):
    c = {k: _pick(v, "nominal") for k, v in overlaid(profile)["params"].items()}
    errs = {}
    for line in open(os.path.join(ROOT, "calibration", "data", data)):
        r = json.loads(line)
        p = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
        errs.setdefault(p["factory"], []).append(abs(p["span"] / r["span_max"] - 1))
    means = {k: round(statistics.mean(v), 4) for k, v in errs.items()}
    assert set(means) == {"Overlap", "OverlapRowMajor"} and all(m <= limit for m in means.values()), means


def test_move_answers_from_the_merged_profile():
    r = analyze(req("craq-sim"))
    assert r["status"] == "ok" and validate_response(r) == []
