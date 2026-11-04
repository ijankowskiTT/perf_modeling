"""tilize / untilize: API contract, schedule invariants, and accuracy against the simulator runs in calibration/data."""

import json
import os
import statistics

import pytest

from qperf import analyze
from qperf.ops.layout import estimate, plan, sub_block_width
from qperf.profiles import _pick, load
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def req(op="tilize", shape=(1, 1, 2048, 512), place="dram", costs="craq-sim", **kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
         "workload": {"op": op, "dtype": "bf16", "shape": list(shape), "placements": {"a": place}}}
    r.update(kw)
    return r


@pytest.mark.parametrize("op", ["tilize", "untilize"])
def test_estimate_conforms_and_is_positive(op):
    r = analyze(req(op))
    assert r["status"] == "ok" and validate_response(r) == []
    t = r["baseline"]["time_ns"]
    assert 0 < t["low"] <= t["nominal"] <= t["high"]


def test_single_core_is_slower_and_compare_conforms():
    r = analyze(req("untilize", action="compare", candidates=[{"id": "sc", "overrides": {"use_multicore": False}}]))
    assert validate_response(r) == []
    assert r["candidates"][0]["comparison"]["verdict"] == "slower"


def test_sensitivity_wider_is_slower():
    r = analyze(req(action="sensitivity", sensitivity={"param": "num_cols", "values": [64, 256, 1024, 2048]}))
    assert validate_response(r) == []
    t = [p["time_ns"] for p in r["sensitivity"]["series"]["baseline"]]
    assert t == sorted(t)


@pytest.mark.parametrize("request_, status", [
    (req("untilize", place="borrowed"), "unsupported"),
    (req(action="tune"), "unsupported"),
    (req(costs="silicon-hypothesis"), "needs_input"),
])
def test_unsupported_and_missing_costs_carry_no_numbers(request_, status):
    r = analyze(request_)
    assert r["status"] == status and "baseline" not in r and validate_response(r) == []


def test_factory_selection_and_split():
    assert plan("tilize", 3136, 64)[0] == "Default"
    fac, loads = plan("tilize", 3136, 64)
    assert [(c["blocks"], n) for c, n in loads] == [(4, 24), (2, 1)]  # 98 tile-rows: 4 per core on 25 cores, cliff 2
    assert plan("tilize", 2048, 64, mem="hs")[0] == "Sharded"
    assert plan("untilize", 256, 512, multicore=False)[0] == "SingleCore"
    assert [sub_block_width(w, 2) for w in (4, 8, 12, 13, 32)] == [4, 8, 6, 1, 8]


@pytest.mark.parametrize("profile,data,limit", [
    ("craq-sim", "layout-craq-sim.jsonl", 0.025),
    ("ttsim-perf", "layout-ttsim-perf.jsonl", 0.03),
])
def test_accuracy_per_factory(profile, data, limit):
    prof = load("costs", profile)[0]
    c = {k: _pick(v, "nominal") for k, v in prof["params"].items()}
    errs = {}
    for line in open(os.path.join(ROOT, "calibration", "data", data)):
        r = json.loads(line)
        if r["op"] not in ("tilize", "untilize"):  # padding variants: collected, not modeled yet
            continue
        p = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
        errs.setdefault((r["op"], p["factory"]), []).append(abs(p["span"] / r["span_max"] - 1))
    means = {k: round(statistics.mean(v), 4) for k, v in errs.items()}
    assert len(means) == 5 and all(m <= limit for m in means.values()), means
