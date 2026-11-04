"""fp32 tilize / untilize / padding variants / sharding moves: API, fp32 overrides, and accuracy against the simulator
runs in calibration/data/fp32layout-<profile>.jsonl, with calibration/out/fp32layout-<profile>.json (the fitted
<p>.f32_* terms) laid over the profile until they are merged into it (calibration/merge_params.py)."""

import json
import os
import statistics

import pytest

from qperf import analyze
from qperf.ops.layout import dtype_costs, estimate, plan, prefix_of, sub_block_width
from qperf.profiles import _pick, load
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def overlay(profile):
    path = os.path.join(ROOT, "calibration", "out", f"fp32layout-{profile}.json")
    return json.load(open(path))


def costs(profile, with_fp32=True):
    c = {k: _pick(v, "nominal") for k, v in load("costs", profile)[0]["params"].items()}
    if with_fp32:
        c.update({k: _pick(v, "nominal") for k, v in overlay(profile).items()})
    return c


def req(op, shape, place="dram", profile="craq-sim", **kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": profile},
         "workload": {"op": op, "dtype": "fp32", "shape": list(shape), "placements": {"a": place}}}
    r.update(kw)
    return r


@pytest.mark.parametrize("profile", ["craq-sim", "ttsim-perf"])
@pytest.mark.parametrize("op,shape", [("tilize", (1, 1, 2048, 512)), ("untilize", (1, 1, 2048, 512)),
                                      ("untilize", (1, 1, 256, 4096)), ("tilize_with_val_padding", (1, 1, 3000, 500)),
                                      ("untilize_with_unpadding", (1, 1, 250, 1000))])
def test_fp32_estimate_conforms(profile, op, shape):
    ovl = overlay(profile)
    r = analyze(req(op, shape, profile=profile, what_if={"cost_overrides": ovl}))
    assert r["status"] == "ok" and validate_response(r) == []
    t = r["baseline"]["time_ns"]
    assert 0 < t["low"] <= t["nominal"] <= t["high"]
    assert not any("fp32 is not calibrated" in w for w in r["baseline"].get("warnings", []))
    # the fitted fp32 terms are what the estimate uses: without them it is the bf16 fit (and says so, until merged)
    plain = analyze(req(op, shape, profile=profile))
    prof_has = any(".f32_" in k for k in load("costs", profile)[0]["params"])
    warned = any("fp32 is not calibrated" in w for w in plain["baseline"].get("warnings", []))
    assert warned != prof_has


def test_fp32_overrides_leave_bf16_alone():
    c = costs("craq-sim")
    base = costs("craq-sim", with_fp32=False)
    for op, H, W, mem in [("tilize", 1024, 512, "dram"), ("untilize", 1024, 512, "l1"), ("untilize", 256, 4096, "dram"),
                          ("interleaved_to_sharded", 1024, 512, "hs8")]:
        assert estimate(op, H, W, c, "bf16", mem)["span"] == estimate(op, H, W, base, "bf16", mem)["span"]
    fac, _ = plan("tilize", 1024, 512, "fp32")
    p = prefix_of("tilize", fac)
    v = dtype_costs(c, p, 4)
    assert v[f"{p}.cu_per_tile"] == c[f"{p}.f32_cu_per_tile"] and dtype_costs(c, p, 2) is c


def test_fp32_dest_holds_half_the_tiles():
    assert [sub_block_width(w, 4) for w in (2, 4, 8, 12, 13, 32)] == [2, 4, 4, 4, 1, 4]
    fac, loads = plan("untilize", 1024, 256, "fp32")
    assert fac == "MultiCore" and loads[0][0]["elem"] == 4 and loads[0][0]["row_bytes"] == 1024


@pytest.mark.parametrize("profile,limit", [("craq-sim", 0.025), ("ttsim-perf", 0.03)])
def test_fp32_accuracy_per_factory(profile, limit):
    c = costs(profile)
    errs = {}
    for line in open(os.path.join(ROOT, "calibration", "data", f"fp32layout-{profile}.jsonl")):
        r = json.loads(line)
        assert r["dtype"] == "fp32" and r["max_abs_err"] == 0.0  # fp32 data moves are exact
        p = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
        errs.setdefault((r["op"], p["factory"]), []).append(abs(p["span"] / r["span_max"] - 1))
    means = {k: round(statistics.mean(v), 4) for k, v in errs.items()}
    assert len(means) == 13 and all(m <= KNOWN.get((profile,) + k, limit) for k, m in means.items()), means


# known gap (docs/CALIBRATION.md): narrow, writer-bound rows of untilize_with_unpadding SingleCore are under-predicted
# by ~5% in bf16 too; the fp32 fit adds wide rows (4 sub-blocks per row) over-predicted by up to 10%
KNOWN = {("craq-sim", "untilize_with_unpadding", "SingleCore"): 0.03}
