"""Hypothetical tilize / untilize designs (cores, split, tiles_per_block, dfb_blocks, write_batch / read_batch): the
factory defaults stay the factory plan, variants are comparable, tune / sensitivity / structure contracts."""

import json
import math
import os

import pytest

from qperf import analyze
from qperf.ops.layout import DESIGN_FACTORIES, estimate, plan, resolve_design
from qperf.profiles import _pick, load
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYP = "hypothetical layout ("
KEYS = {"factory", "grid", "cores_used", "split", "row_cores", "col_chunks", "source", "dest", "per_core", "reader",
        "compute", "writer", "dfb"}


def req(op="tilize", shape=(1, 1, 64, 2048), costs="craq-sim", **kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
         "workload": {"op": op, "dtype": "bf16", "shape": list(shape), "placements": {"a": "dram"}}}
    r.update(kw)
    return r


@pytest.mark.parametrize("profile", ["craq-sim", "ttsim-perf"])
def test_factory_defaults_resolve_to_the_factory_plan(profile):
    prof = load("costs", profile)[0]
    c = {k: _pick(v, "nominal") for k, v in prof["params"].items()}
    n = 0
    for line in open(os.path.join(ROOT, "calibration", "data", f"layout-{profile}.jsonl")):
        r = json.loads(line)
        args = (r["op"], r["H"], r["W"])
        try:
            factory = plan(*args, r["dtype"], r["mem"], r["multicore"])[0]
        except ValueError:
            continue
        if (r["op"], factory) not in DESIGN_FACTORIES:
            continue
        explicit = {"cores": 32, "split": "rows", "tiles_per_block": 10 ** 6, "dfb_blocks": 1 if r["op"] == "tilize" else 2,
                    DESIGN_FACTORIES[(r["op"], factory)]: 1}
        assert not resolve_design(r["op"], factory, r["H"], r["W"], r["dtype"], explicit)["hypothetical"]
        ref = estimate(*args, c, r["dtype"], r["mem"], r["multicore"])["span"]
        assert estimate(*args, c, r["dtype"], r["mem"], r["multicore"], design=explicit)["span"] == ref
        n += 1
    assert n > 20


def test_defaults_in_the_api_are_not_hypothetical():
    a = analyze(req())
    b = analyze(req(baseline={"cores": 32, "split": "rows", "write_batch": 1, "dfb_blocks": 1}))
    assert a["baseline"]["time_ns"] == b["baseline"]["time_ns"] and a["baseline"]["config"] == b["baseline"]["config"]
    assert "hypothetical" not in a["baseline"]["config"] and not any(HYP in w for w in b["baseline"]["warnings"])


def test_cols_and_2d_use_more_cores_and_are_faster_on_a_wide_short_tensor():
    r = analyze(req(action="compare", candidates=[{"id": "cols", "overrides": {"split": "cols"}},
                                                  {"id": "2d", "overrides": {"split": "2d", "cores": 16}}]))
    assert r["status"] == "ok" and validate_response(r) == []
    base = r["baseline"]
    assert base["structure"]["cores_used"] == 2  # 64 rows = 2 tile-rows: the factory uses 2 cores
    for c in r["candidates"]:
        assert c["structure"]["cores_used"] > base["structure"]["cores_used"]
        assert c["structure"]["col_chunks"] > 1 and c["comparison"]["verdict"] == "faster"
        assert c["config"]["hypothetical"] is True and any(HYP in w for w in c["warnings"])
        assert "not in the pinned kernels; timed with the factory's calibrated per-request costs" in c["warnings"][-1]


@pytest.mark.parametrize("op,shape", [("tilize", (1, 1, 64, 2048)), ("tilize", (1024, 1024)), ("untilize", (1024, 1024))])
def test_tune_ranks_distinct_plans(op, shape):
    r = analyze(req(op, shape, action="tune", analysis={"top_k": 5}))
    assert r["status"] == "ok" and validate_response(r) == []
    assert r["runtime_ms"] < 2000
    s = r["search"]
    assert s["exhaustive"] and s["enumerated"] == 3 * 3 * 2 * 2 * 3 and 1 < s["valid"] < s["enumerated"]
    assert len(set(s["evaluated_ids"])) == len(s["evaluated_ids"])  # equal plans deduped
    top = r["top"]
    assert len(top) == 5 and len({t["id"] for t in top}) == 5
    t = [x["time_ns"]["nominal"] for x in top]
    assert t[0] < r["baseline"]["time_ns"]["nominal"] and t[1:] == sorted(t[1:])
    for x in top:
        assert "comparison" in x and "structure" in x
        if x["config"].get("hypothetical"):
            assert any(w.startswith(HYP) for w in x["warnings"])
    rec = r["recommendations"][0]
    assert rec["id"] == top[0]["id"] and rec["speedup_nominal"] > 1


def test_tune_space_override_and_unsupported_factories():
    r = analyze(req(action="tune", tune={"space": {"cores": [32], "split": ["cols"], "tiles_per_block": [None],
                                                   "dfb_blocks": [1], "write_batch": [1, 2]}}))
    assert r["status"] == "ok" and r["search"]["enumerated"] == 2
    assert {x["config"].get("write_batch") for x in r["top"]} >= {1, 2}
    r = analyze(req(action="tune", tune={"max_candidates": 3}))
    assert r["status"] == "partial" and r["search"]["truncated_by_max_candidates"]
    for q in (req("untilize", (64, 2048), action="tune"),  # the Block factory
              req(action="tune", baseline={"use_multicore": False}),
              req("tilize_with_val_padding", (100, 1000), action="tune")):
        out = analyze(q)
        assert out["status"] == "unsupported" and "tilize Default and untilize MultiCore" in out["message"]


@pytest.mark.parametrize("over,msg", [({"use_multicore": False, "cores": 4}, "not tilize SingleCore"),
                                      ({"read_batch": 4}, "read_batch is not a knob of tilize"),
                                      ({"input": "borrowed", "split": "cols"}, "not tilize Sharded")])
def test_knobs_rejected_where_not_modeled(over, msg):
    r = analyze(req(action="compare", candidates=[{"id": "x", "overrides": over}]))
    c = r["candidates"][0]
    assert not c["valid"] and msg in c["errors"][0]
    r = analyze(req(baseline=over))
    assert r["status"] == "unsupported" and msg in r["message"]


def test_sensitivity_has_candidate_series():
    r = analyze(req(action="sensitivity", sensitivity={"param": "num_cols", "values": [512, 2048, 4096]},
                    candidates=[{"id": "cols", "overrides": {"split": "cols"}},
                                {"id": "wb8", "overrides": {"write_batch": 8}}]))
    assert r["status"] == "ok" and validate_response(r) == []
    series = r["sensitivity"]["series"]
    assert set(series) == {"baseline", "cols", "wb8"}
    for pts in series.values():
        assert [p["value"] for p in pts] == [512, 2048, 4096] and all(p["valid"] and p["time_ns"] for p in pts)
    assert all(c < b for c, b in zip((p["time_ns"] for p in series["cols"]), (p["time_ns"] for p in series["baseline"])))


def _check_structure(st, cfg):
    assert set(st) == KEYS and st["grid"] == [8, 4]
    pc = st["per_core"]
    assert pc["tiles"] == pc["blocks"] * pc["tiles_per_block"]
    assert 1 <= st["cores_used"] <= 32 and st["row_cores"] * st["col_chunks"] >= st["cores_used"] >= st["row_cores"]
    tilize = st["compute"]["kind"] == "tilize_block"
    rows, tiles = (st["reader"], st["writer"]) if tilize else (st["writer"], st["reader"])
    assert rows["requests"] == pc["blocks"] * 32 and rows["barriers"] == pc["blocks"]
    assert rows["bytes_per_request"] == pc["tiles_per_block"] * 32 * 2
    batch = cfg.get("write_batch" if tilize else "read_batch", 1)
    assert tiles["requests"] == pc["tiles"] and tiles["barriers"] == pc["blocks"] * math.ceil(pc["tiles_per_block"] / batch)
    if tilize:
        assert st["compute"]["steps"] == pc["blocks"] and st["compute"]["tiles_per_step"] == pc["tiles_per_block"]
    else:
        assert st["compute"]["steps"] * st["compute"]["tiles_per_step"] == pc["tiles"]
    if cfg.get("hypothetical"):
        assert st["dfb"]["in_entries"] == cfg["dfb_blocks"] * pc["tiles_per_block"] == st["dfb"]["out_entries"]
        assert st["split"] == cfg["split"] and st["cores_used"] == cfg["cores"]
    assert st["dfb"]["entry_bytes"] == 2048


@pytest.mark.parametrize("op,shape", [("tilize", (1, 1, 64, 2048)), ("tilize", (4096, 2048)), ("untilize", (1024, 1024)),
                                      ("untilize", (96, 1024))])
def test_structure_is_consistent(op, shape):
    r = analyze(req(op, shape, action="tune", analysis={"top_k": 10}))
    assert r["status"] == "ok"
    for row in [r["baseline"]] + r["top"]:
        _check_structure(row["structure"], row["config"])
    st = r["baseline"]["structure"]
    assert st["split"] == "rows" and st["col_chunks"] == 1 and st["source"] == st["dest"] == "dram"


def test_structure_for_other_factories():
    cases = [("tilize", (3136, 64), {"input": "borrowed"}, "Sharded", 0),
             ("tilize", (2048, 512), {"use_multicore": False}, "SingleCore", 2048),
             ("untilize", (256, 4096), {}, "Block", 64)]
    for op, shape, knobs, factory, reqs in cases:
        r = analyze(req(op, shape, baseline=knobs))
        assert r["status"] == "ok" and validate_response(r) == []
        st = r["baseline"]["structure"]
        assert set(st) == KEYS and st["factory"] == factory
        assert st["reader"]["requests"] == reqs
    st = analyze(req("untilize", (256, 4096)))["baseline"]["structure"]
    assert st["split"] == "2d" and st["row_cores"] * st["col_chunks"] == st["cores_used"] == 16


def test_batches_that_do_not_divide_the_block_run():
    # 6-tile blocks, 4-tile barriers: groups of 4 + 2 per block, no deadlock, fewer barriers than per-tile
    a = analyze(req("tilize", (32 * 8, 32 * 6), baseline={"cores": 4, "write_batch": 4}))
    b = analyze(req("tilize", (32 * 8, 32 * 6), baseline={"cores": 4}))
    assert a["status"] == b["status"] == "ok"
    assert a["baseline"]["structure"]["writer"]["barriers"] == 2 * 2 and b["baseline"]["structure"]["writer"]["barriers"] == 12
    assert a["baseline"]["time_ns"]["nominal"] < b["baseline"]["time_ns"]["nominal"]
    u = analyze(req("untilize", (32 * 8, 32 * 6), baseline={"cores": 4, "read_batch": 4, "tiles_per_block": 3,
                                                           "dfb_blocks": 3}))
    assert u["status"] == "ok" and u["baseline"]["structure"]["dfb"]["in_entries"] == 9


@pytest.mark.parametrize("profile", ["craq-sim", "ttsim-perf"])
def test_batched_writer_runs_within_acceptance(profile):
    """tilize Default with a real batched writer kernel (calibration/data/tilize-wbatch-*.jsonl, write_batch 4 and 8,
    12 shapes each, plus the pinned writer at the same shapes): mean error <= 2% per batch, and the batching speedup
    over the pinned writer within 2% on average; the accuracy table states the same numbers (hypothetical_validation)."""
    prof = load("costs", profile)[0]
    c = {k: _pick(v, "nominal") for k, v in prof["params"].items()}
    meas, pred = {}, {}
    for line in open(os.path.join(ROOT, "calibration", "data", f"tilize-wbatch-{profile}.jsonl")):
        r = json.loads(line)
        k = (r["design"]["write_batch"], r["H"], r["W"])
        meas[k] = r["span_max"]
        pred[k] = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"], design=r["design"])["span"]
    shapes = sorted({k[1:] for k in meas})
    assert len(shapes) == 12 and set(meas) == {(b,) + s for b in (1, 4, 8) for s in shapes}
    errs = []
    for b in (1, 4, 8):
        e = [abs(pred[(b,) + s] / meas[(b,) + s] - 1) for s in shapes]
        assert sum(e) / len(e) <= 0.02, (profile, b, sum(e) / len(e))
        if b > 1:
            errs += e
            sp = [abs((pred[(1,) + s] / pred[(b,) + s]) / (meas[(1,) + s] / meas[(b,) + s]) - 1) for s in shapes]
            assert sum(sp) / len(sp) <= 0.02, (profile, b, sum(sp) / len(sp))
            assert all(pred[(b,) + s] < pred[(1,) + s] for s in shapes)  # batching never predicted slower here
    v = json.load(open(os.path.join(ROOT, "profiles", "accuracy.json")))["hypothetical_validation"]
    v = v["tilize Default write_batch"][profile]
    assert v["runs"] == len(errs) == 24 and abs(v["mean"] - sum(errs) / len(errs)) <= 0.001


@pytest.mark.parametrize("profile", ["craq-sim", "ttsim-perf"])
@pytest.mark.parametrize("name,designs,n_shapes", [
    ("tilize-dfb", ({"dfb_blocks": 2}, {"dfb_blocks": 2, "write_batch": 8}), 18),
    ("tilize-cols", ({"split": "cols"},), 14),
])
def test_dfb_and_cols_runs_within_acceptance(profile, name, designs, n_shapes):
    """tilize Default kernel variants of calibration/variants/tilize_dfb_cols.patch: dfb_blocks 2 (alone and with
    write_batch 8) over 18 shapes, split cols over 14 (2 tall shapes where cols resolves to rows), each file with the
    pinned kernels at the same shapes. Mean error <= 2% per design, the speedup over the pinned kernels within 2% on
    average, gains predicted where measured (> 3%), and the accuracy table states the same numbers."""
    prof = load("costs", profile)[0]
    c = {k: _pick(v, "nominal") for k, v in prof["params"].items()}
    meas, pred = {}, {}
    for line in open(os.path.join(ROOT, "calibration", "data", f"{name}-{profile}.jsonl")):
        r = json.loads(line)
        assert r["max_abs_err"] == 0
        d = None if r["kernel_variant"] == "pinned" else json.dumps(r["design"], sort_keys=True)
        meas[(d, r["H"], r["W"])] = r["span_max"]
        pred[(d, r["H"], r["W"])] = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"],
                                             design=None if d is None else r["design"])["span"]
    shapes = sorted({k[1:] for k in meas})
    keys = [json.dumps(d, sort_keys=True) for d in designs]
    assert len(shapes) == n_shapes and set(meas) == {(d,) + s for d in [None] + keys for s in shapes}
    v = json.load(open(os.path.join(ROOT, "profiles", "accuracy.json")))["hypothetical_validation"]
    for d, k in zip(designs, keys):
        e = [abs(pred[(k,) + s] / meas[(k,) + s] - 1) for s in shapes]
        assert sum(e) / len(e) <= 0.02, (profile, k, sum(e) / len(e))
        ms = {s: meas[(None,) + s] / meas[(k,) + s] for s in shapes}
        ps = {s: pred[(None,) + s] / pred[(k,) + s] for s in shapes}
        sp = [abs(ps[s] / ms[s] - 1) for s in shapes]
        assert sum(sp) / len(sp) <= 0.02, (profile, k, sum(sp) / len(sp))
        assert all(ps[s] > 1.01 for s in shapes if ms[s] > 1.03), (profile, k)
        stored = v["tilize Default " + "+".join(sorted(d))][profile]
        assert stored["runs"] == n_shapes and abs(stored["mean"] - sum(e) / len(e)) <= 0.001
