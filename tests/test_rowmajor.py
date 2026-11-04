"""interleaved_to_sharded / sharded_to_interleaved, ROW_MAJOR: work split, API contract, and accuracy against the
simulator runs in calibration/data/rowmajor-<profile>.jsonl. The fitted terms live in calibration/out/rowmajor-<profile>.json
until they are merged into profiles/costs; the tests overlay them on the profile's params."""

import copy
import json
import os
import statistics

import pytest

from qperf import analyze
from qperf.ops.layout import estimate, plan
from qperf.profiles import _pick, load, register, unregister
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = ("craq-sim", "ttsim-perf")


def overlay(profile):
    """The profile with the emitted ROW_MAJOR terms on top (a no-op once they are merged)."""
    prof = copy.deepcopy(load("costs", profile)[0])
    path = os.path.join(ROOT, "calibration", "out", f"rowmajor-{profile}.json")
    prof["params"].update(json.load(open(path)))
    return prof


@pytest.fixture(params=PROFILES)
def profile(request):
    register("costs", overlay(request.param))
    yield request.param
    unregister("costs", request.param)


def req(op="interleaved_to_sharded", shape=(1, 1, 1024, 512), place="dram", strategy="height", cores=8,
        layout="row_major", **kw):
    w = {"op": op, "dtype": "bf16", "shape": list(shape), "placements": {"a": place},
         "shard": {"strategy": strategy, "cores": cores}}
    if layout:
        w["layout"] = layout
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": "craq-sim"}, "workload": w}
    r.update(kw)
    return r


def test_work_split():
    # height: ceil(H/n) rows per shard, only cores with data run, the last shard clipped
    fac, loads = plan("sharded_to_interleaved", 250, 512, mem="hs8r")
    assert fac == "RowMajor" and [(c["rows"], c["width"], n) for c, n in loads] == [(32, 1024, 7), (26, 1024, 1)]
    fac, loads = plan("interleaved_to_sharded", 900, 40, mem="hs32r")
    assert [(c["rows"], n) for c, n in loads] == [(29, 31), (1, 1)]
    # i2s from DRAM: rows that are not whole 64 B go through the scratch (unaligned), from L1 never
    assert plan("interleaved_to_sharded", 256, 512, mem="hs8r")[0] == "RowMajor"
    assert plan("interleaved_to_sharded", 256, 520, mem="hs8r")[0] == "RowMajorUnaligned"
    assert plan("interleaved_to_sharded", 256, 520, mem="hs8lr")[0] == "RowMajor"
    assert plan("sharded_to_interleaved", 256, 520, mem="hs8r")[0] == "RowMajor"
    # width: shard width rounded up to 16 B, the source window starts at curr_idx_w rounded down to 64 B
    # 64x80 over 4: 20 elements -> 24 (48 B); offsets 0, 48, 96, 144 -> windows 48, 96, 80, 32 B (last shard 16 B)
    fac, loads = plan("interleaved_to_sharded", 64, 80, mem="ws4r")
    assert fac == "RowMajorUnaligned"
    assert [(c["width"], c["abytes"]) for c, _ in loads] == [(48, 48), (48, 96), (48, 80), (16, 32)]
    # block: k x k grid of ceil(H/k) x ceil(W/k) (width rounded to 16 B), right / bottom shards clipped
    fac, loads = plan("interleaved_to_sharded", 1000, 1000, mem="bs9r")
    assert sum(n for _, n in loads) == 9 and loads[0][0]["rows"] == 334 and loads[0][0]["width"] == 672
    assert sorted({(c["rows"], c["width"]) for c, _ in loads}) == [(332, 656), (332, 672), (334, 656), (334, 672)]
    with pytest.raises(ValueError):  # the ops reject shard rows that are not whole 16 B
        plan("interleaved_to_sharded", 128, 100, mem="hs4r")
    with pytest.raises(ValueError):
        plan("interleaved_to_sharded", 128, 128, mem="bs8r")


def test_tile_default_is_unchanged():
    assert plan("interleaved_to_sharded", 1024, 256, mem="hs8")[0] == "Default"
    r = analyze(req(layout=None))
    assert r["status"] == "ok" and "row_major" not in r["baseline"]["label"]


def test_estimate_conforms(profile):
    for op in ("interleaved_to_sharded", "sharded_to_interleaved"):
        for place, shape, strategy, cores in [("dram", (1, 1, 1024, 512), "height", 8), ("l1", (2, 256, 520), "width", 4),
                                              ("dram", (1, 1, 1000, 1000), "block", 9)]:
            r = analyze(req(op, shape, place, strategy, cores, profiles={"costs": profile}))
            assert r["status"] == "ok" and validate_response(r) == [], r
            assert "row_major" in r["baseline"]["label"]
            t = r["baseline"]["time_ns"]
            assert 0 < t["low"] <= t["nominal"] <= t["high"]


def test_unaligned_is_slower_and_compare_conforms(profile):
    a = analyze(req(shape=(1, 1, 1024, 512), profiles={"costs": profile}))
    b = analyze(req(shape=(1, 1, 1024, 520), profiles={"costs": profile}))
    assert b["baseline"]["time_ns"]["nominal"] > a["baseline"]["time_ns"]["nominal"]
    r = analyze(req(action="compare", profiles={"costs": profile},
                    candidates=[{"id": "32", "overrides": {"shard_cores": 32}}]))
    assert r["status"] == "ok" and validate_response(r) == []
    assert r["candidates"][0]["comparison"]["verdict"] == "faster"


@pytest.mark.parametrize("request_, status", [
    (req(layout="rowmajor"), "invalid"),  # not in the schema enum
    (req(shape=(1, 1, 128, 100), cores=4), "unsupported"),  # 200 B shard rows
])
def test_invalid_and_unsupported_carry_no_numbers(request_, status):
    r = analyze(request_)
    assert r["status"] == status and "baseline" not in r and validate_response(r) == []


@pytest.mark.parametrize("profile_,limit", [("craq-sim", 0.025), ("ttsim-perf", 0.03)])
def test_accuracy_per_factory(profile_, limit):
    c = {k: _pick(v, "nominal") for k, v in overlay(profile_)["params"].items()}
    errs = {}
    for line in open(os.path.join(ROOT, "calibration", "data", f"rowmajor-{profile_}.jsonl")):
        r = json.loads(line)
        assert r["max_abs_err"] == 0.0  # data movement: outputs exact
        p = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
        errs.setdefault((r["op"], p["factory"]), []).append(abs(p["span"] / r["span_max"] - 1))
    means = {k: round(statistics.mean(v), 4) for k, v in errs.items()}
    assert len(means) == 3 and all(m <= limit for m in means.values()), means
