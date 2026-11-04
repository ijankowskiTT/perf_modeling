"""reshard: API contract, factory selection and work split, accuracy against the simulator runs in calibration/data."""

import copy
import json
import os
import statistics

import pytest

from qperf import analyze, profiles, validate
from qperf.ops.layout import estimate
from qperf.ops.reshard import _map_for_reshard, parse_mem, plan, select_factory
from qperf.profiles import _pick, load
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Proposed schema fragment: request.schema.json workload.properties.reshard (until the schema carries it).
RESHARD_SIDE = {"type": "object", "additionalProperties": False, "required": ["strategy"],
                "properties": {"strategy": {"enum": ["height", "width", "block"]},
                               "cores": {"type": "integer", "minimum": 1},
                               "grid": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                               "memory": {"enum": ["l1", "dram"]}}}
RESHARD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["from", "to"],
                  "properties": {"layout": {"enum": ["tile", "row_major"]}, "from": RESHARD_SIDE, "to": RESHARD_SIDE}}


def _overlay(profile):
    """The profile with calibration/out/reshard-<profile>.json merged in, registered in memory."""
    prof = copy.deepcopy(load("costs", profile)[0])
    path = os.path.join(ROOT, "calibration", "out", f"reshard-{profile}.json")
    prof["params"].update(json.load(open(path)))
    prof["name"] = f"{profile}-reshard-test"
    return prof


@pytest.fixture
def api():
    validate._validate({}, "request.schema.json")  # load and cache the schema
    schema = validate._schemas["request.schema.json"]
    saved = copy.deepcopy(schema)
    schema["properties"]["workload"]["properties"].setdefault("reshard", RESHARD_SCHEMA)
    for name in ("craq-sim", "ttsim-perf"):
        profiles.register("costs", _overlay(name))
    yield
    validate._schemas["request.schema.json"] = saved
    for name in ("craq-sim", "ttsim-perf"):
        profiles.unregister("costs", f"{name}-reshard-test")


def req(frm, to, shape=(1, 1, 1024, 1024), layout="tile", costs="craq-sim-reshard-test", **kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
         "workload": {"op": "reshard", "dtype": "bf16", "shape": list(shape),
                      "reshard": {"layout": layout, "from": frm, "to": to}}}
    r.update(kw)
    return r


H8 = {"strategy": "height", "cores": 8, "memory": "l1"}
H4 = {"strategy": "height", "cores": 4, "memory": "l1"}
H2D = {"strategy": "height", "cores": 2, "memory": "dram"}


@pytest.mark.parametrize("frm,to,layout", [(H8, H4, "tile"), (H2D, H8, "tile"), (H8, H2D, "tile"),
                                           ({"strategy": "width", "cores": 4, "memory": "l1"},
                                            {"strategy": "width", "cores": 8, "memory": "l1"}, "row_major")])
def test_estimate_conforms_and_is_positive(api, frm, to, layout):
    for costs in ("craq-sim-reshard-test", "ttsim-perf-reshard-test"):
        r = analyze(req(frm, to, layout=layout, costs=costs))
        assert r["status"] == "ok" and validate_response(r) == [], r
        t = r["baseline"]["time_ns"]
        assert 0 < t["low"] <= t["nominal"] <= t["high"]


def test_more_cores_is_faster_and_compare_conforms(api):
    r = analyze(req(H8, {"strategy": "height", "cores": 16, "memory": "l1"}))
    slow = analyze(req(H8, {"strategy": "height", "cores": 2, "memory": "l1"}))
    assert r["baseline"]["time_cycles"] < slow["baseline"]["time_cycles"]


@pytest.mark.parametrize("request_, status", [
    (req(H8, H4, costs="craq-sim"), "needs_input"),  # a profile without reshard costs carries no numbers
    (req(H8, {"strategy": "height", "cores": 4, "memory": "dram"}), "invalid"),  # 4 DRAM banks: there are 2
])
def test_errors_carry_no_numbers(api, request_, status):
    r = analyze(request_)
    assert r["status"] == status and "baseline" not in r and validate_response(r) == []


def test_factory_selection():
    sel = lambda m, H=1024, W=1024: select_factory(parse_mem(m, H, W), W, 2)  # noqa: E731
    assert sel("t_hs8l_hs4l") == "SameWidth" and sel("t_hs2d_hs8l") == "SameWidth"
    assert sel("t_hs8l_hs2d") == "SameWidthWrite"
    assert sel("r_ws4l_ws8l") == "SameHeight" and sel("r_ws8l_ws2d") == "SameHeightWrite"
    assert sel("r_ws4l_ws8l", W=1000) == "Generic"  # padded shards
    assert sel("t_ws4l_ws8l") == "Generic" and sel("t_hs8l_ws8l") == "Generic" and sel("t_bs2x2l_bs4x4l") == "Generic"
    assert sel("t_hs2d_ws2d") == "CopyPages" and sel("t_bs2x2l_hs2d") == "CopyLocal"


def test_work_split_mirrors_the_factories():
    # SameWidth: 8 -> 4 L1 cores, 1024x1024 TILE: each output core reads 2 remote shards of 128 tiles: one segment
    # per kernel (64 tiles each... the kernel split is div_up(256, 2) = 128 units = one remote shard each)
    fac, loads = plan("reshard", 1024, 1024, "bf16", "t_hs8l_hs4l")
    cfg, n = loads[0]
    assert sum(n for _, n in loads) == 4 and cfg["k0"] == [128 * 2048] and cfg["k1"] == [128 * 2048]
    # ttsim-perf NoC: core 0 is paced by its own port (factor 1); core 3's responses cross the link x6 -> x5 that
    # carries 4 flows (512 KB) against its 256 KB inbound: factor 2
    assert cfg["noc"] == (1.0, 0.5) and loads[-1][0]["noc"] == (1.0, 2.0)
    # 4 -> 8: each kernel reads half of one output shard from one input shard
    cfg = plan("reshard", 1024, 1024, "bf16", "t_hs4l_hs8l")[1][0][0]
    assert cfg["k0"] == [64 * 2048] and cfg["k1"] == [64 * 2048]
    # SameHeight 4 -> 8 cores, 256 rows: one segment per core; k0 128 rows, k1 128 rows, one read per row
    cfg = plan("reshard", 256, 1024, "bf16", "r_ws4l_ws8l")[1][0][0]
    assert cfg["k0"] == [256] * 128 and cfg["seg0"] == [1]
    # CopyPages: 1024 tiles over 32 cores
    fac, loads = plan("reshard", 1024, 1024, "bf16", "t_hs2d_ws2d")
    assert [(c["pages"], n) for c, n in loads] == [(32, 32)]


def test_page_runs_follow_create_map_for_reshard():
    a, b = ((0, 0), 0), ((1, 0), 0)
    # one run of 4 consecutive pages on one core
    runs = lambda seq: [x[:3] for x in _map_for_reshard(seq)]  # noqa: E731  (stride size, strides, skip)
    assert runs([((0, 0), i) for i in range(4)]) == [(4, 1, False)]
    # padding gap, then a run
    assert runs([None, None, ((0, 0), 0), ((0, 0), 1)]) == [(2, 1, True), (2, 1, False)]
    # runs of 2 with a data stride: the kernel's stride check groups two runs per PageStride
    seq = [((0, 0), r * 8 + c) for r in range(4) for c in range(2)]
    assert runs(seq) == [(2, 2, False), (2, 2, False)]
    assert _map_for_reshard([a, b]) == [(1, 2, False, (0, 0), (1, 0))]  # same page, next core: a core stride


@pytest.mark.parametrize("profile,data,limit", [
    ("craq-sim", "reshard-craq-sim.jsonl", 0.025),
    ("ttsim-perf", "reshard-ttsim-perf.jsonl", 0.03),
])
def test_accuracy_per_factory(profile, data, limit):
    prof = _overlay(profile)
    c = {k: _pick(v, "nominal") for k, v in prof["params"].items()}
    errs = {}
    for line in open(os.path.join(ROOT, "calibration", "data", data)):
        r = json.loads(line)
        assert r["max_abs_err"] < 0.05  # every output was exact
        p = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
        errs.setdefault(p["factory"], []).append(abs(p["span"] / r["span_max"] - 1))
    means = {k: round(statistics.mean(v), 4) for k, v in errs.items()}
    assert len(means) >= 5 and all(m <= limit for m in means.values()), means
