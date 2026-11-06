"""API / CLI contract: statuses, validity rejection, determinism, CLI == API, scaling invariants."""

import copy
import json
import subprocess
import sys
import os

import pytest

from qperf import analyze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def req(**kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": "ttsim-perf"},
         "workload": {"op": "add", "dtype": "bf16", "num_tiles": 128, "tiles_scope": "cluster", "clusters": 32,
                      "placements": {"a": "dram", "b": "dram", "out": "dram"}},
         "baseline": {"reader_threads": 4, "compute_threads": 4, "writer_threads": 2, "dm_batch": 1,
                      "tiles_per_cycle": 1, "entries_per_thread": 2}}
    r.update(kw)
    return r


def test_ok_estimate_has_contract_fields():
    r = analyze(req())
    assert r["status"] == "ok"
    for k in ("schema_version", "engine_version", "profiles", "resolved_input_sha256", "assumptions", "result_kind"):
        assert k in r
    b = r["baseline"]
    assert b["time_ns"]["low"] <= b["time_ns"]["nominal"] <= b["time_ns"]["high"]
    assert b["lower_bound_ns"] <= b["time_ns"]["nominal"]
    for p in r["profiles"].values():
        assert len(p["sha256"]) == 64


def test_missing_cost_profile_is_needs_input():
    r = req()
    del r["profiles"]
    assert analyze(r)["status"] == "needs_input"


def test_unknown_profile_and_revision_mismatch():
    assert analyze(req(profiles={"costs": "no-such"}))["status"] == "unsupported"
    assert analyze(req(profiles={"costs": "ttsim-perf@1999-01-01"}))["status"] == "unsupported"


def test_uncalibrated_variant_is_needs_input_not_a_guess():
    from qperf import profiles as P

    prof = copy.deepcopy(P.load("costs", "ttsim-perf")[0])
    prof["name"] = "ttsim-perf__no_divide"
    del prof["compute_variants"]["sfpu:divide:bf16"]
    P.register("costs", prof)
    try:
        r = req(profiles={"costs": prof["name"]})
        r["workload"]["op"] = "divide"  # SFPU, no compute costs in this profile
        out = analyze(r)
    finally:
        P.unregister("costs", prof["name"])
    assert out["status"] == "needs_input" and "sfpu:divide:bf16" in out["message"]
    assert "time_ns" not in json.dumps(out)


def test_unknown_routing_is_unsupported():
    r = req()
    r["workload"]["op"] = "logaddexp"
    out = analyze(r)
    assert out["status"] == "unsupported" and "Metal 2.0" in out["message"]


def test_schema_rejects_unknown_fields():
    r = req()
    r["baseline"]["readers"] = 4
    assert analyze(r)["status"] == "invalid"


@pytest.mark.parametrize("knobs,frag", [
    ({"compute_threads": 3}, "C=3"),
    ({"reader_threads": 4, "writer_threads": 4}, "R+W"),
    ({"reader_threads": 3}, "STRIDED"),
    ({"tiles_per_cycle": 8, "dm_batch": 8, "entries_per_thread": 8}, "double-buffer"),
    ({"dm_batch": 4, "entries_per_thread": 6}, "divide"),
    ({"entries_per_thread": 128}, "255"),
    ({"tiles_per_cycle": 2, "reader_threads": 4, "compute_threads": 2, "writer_threads": 2, "entries_per_thread": 4}, "R<=C"),
])
def test_invalid_configs_are_rejected(knobs, frag):
    r = req()
    r["baseline"].update(knobs)
    out = analyze(r)
    assert out["status"] == "invalid"
    assert frag in json.dumps(out["baseline"]["errors"])


def test_l1_budget_rejects_oversized_rings():
    # 2048 tiles per cluster of three L1-resident bf16 operands = 12 MiB > a 4 MiB cluster L1
    r = req()
    r["workload"].update(num_tiles=2048, placements={"a": "l1", "b": "l1", "out": "borrowed"})
    out = analyze(r)
    assert out["status"] == "invalid" and "L1" in json.dumps(out["baseline"]["errors"])


def test_deterministic_and_cache_transparent():
    a = analyze(req())
    b = analyze(req())
    for k in ("runtime_ms", "cache_hits"):
        a.pop(k)
        b.pop(k)
    assert a == b


def test_cli_and_api_agree(tmp_path):
    r = req(action="compare", candidates=[{"id": "b8", "overrides": {"dm_batch": 8, "tiles_per_cycle": 8,
                                                                       "entries_per_thread": 16}}])
    p = tmp_path / "r.json"
    p.write_text(json.dumps(r))
    out = subprocess.run([sys.executable, "-m", "qperf", "analyze", "--request", str(p)], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    cli = json.loads(out.stdout)
    api = analyze(r)
    assert cli["baseline"]["time_ns"] == api["baseline"]["time_ns"]
    assert cli["candidates"][0]["time_ns"] == api["candidates"][0]["time_ns"]
    assert cli["resolved_input_sha256"] == api["resolved_input_sha256"]


def test_compare_reports_no_gain_for_identical_candidate():
    r = req(action="compare", candidates=[{"id": "same", "overrides": {}}])
    out = analyze(r)
    assert out["candidates"][0]["comparison"]["verdict"] == "no_meaningful_gain"
    assert out["recommendations"][0]["label"] == "keep baseline"


def test_more_clusters_saturate_on_shared_dram():
    base = req(what_if={"cost_overrides": {"dram.total_bytes_per_cycle": 64.0}})
    base["workload"].update(num_tiles=4096, tiles_scope="operation")
    times = []
    for g in (1, 4, 16, 32):
        r = copy.deepcopy(base)
        r["workload"]["clusters"] = g
        times.append(analyze(r)["baseline"]["time_ns"]["nominal"])
    # floor: 4096 tiles * 6144 B / 64 B per cycle
    floor = 4096 * 6144 / 64.0
    assert times[-1] >= floor
    assert times[1] / times[-1] < 8  # 8x the clusters do not give anywhere near 8x


def test_rule_defaults_are_reported():
    r = req()
    r["baseline"] = {}
    out = analyze(r)
    assert any("(rule)" in a for a in out["applied_defaults"])
    # HEAD rule beside DRAM: share = 128/4 = 32 tiles per compute thread -> batch 8 (bit_floor(min(8, 32)))
    assert out["baseline"]["config"]["dm_batch"] == 8 and out["baseline"]["config"]["tiles_per_cycle"] == 8


def test_tune_is_bounded_and_honest():
    r = req(action="tune", tune={"max_candidates": 40}, analysis={"top_k": 3, "time_budget_ms": 5000})
    out = analyze(r)
    s = out["search"]
    assert s["considered"] <= 40 and s["evaluated"] <= s["considered"]
    assert out["status"] in ("ok", "partial")
    if s["truncated_by_max_candidates"]:
        assert s["exhaustive"] is False and out["status"] == "partial"
    assert len(out["top"]) <= 3


def test_sensitivity_finds_bottleneck_shift():
    r = req(action="sensitivity", sensitivity={"param": "compute_service_scale", "values": [0.25, 0.5, 1, 2, 4]})
    out = analyze(r)
    pts = out["sensitivity"]["series"]["baseline"]
    times = [p["time_ns"] for p in pts]
    assert times == sorted(times)  # more compute work never makes it faster


def test_knob_ceiling_and_hypothetical_override():
    r = req()
    r["baseline"].update(dm_batch=16, tiles_per_cycle=8, entries_per_thread=32)
    out = analyze(r)
    assert out["status"] == "invalid" and "knob ceiling" in json.dumps(out["baseline"]["errors"])
    r["hypothetical"] = {"knob_limits": {"dm_batch": 64}}
    out = analyze(r)
    assert out["status"] == "ok" and out["hypothetical"]
    assert any("HYPOTHETICAL" in a for a in out["assumptions"])


def test_single_evaluation_mode():
    # analysis.mode is gone: there is one evaluation (the pipeline model, exact extrapolation past 2,048 tiles).
    r = analyze(req(analysis={"mode": "quick"}))
    assert r["status"] == "invalid" and "mode" in r["message"]
    full = analyze(req(analysis={"detail": "full"}))
    assert full["result_kind"] == "finite_buffer_estimate" and full["baseline"]["threads"]
    assert "threads" not in analyze(req())["baseline"]


def test_extrapolation_matches_tile_by_tile():
    from qperf import api
    small = analyze(req(workload=dict(req()["workload"], num_tiles=6000)))["baseline"]
    assert small["extrapolated"]
    old = api.EXACT_TILES
    try:
        api.EXACT_TILES = 1 << 20
        api._cache = api.LRU(256)
        exact = analyze(req(workload=dict(req()["workload"], num_tiles=6000)))["baseline"]
    finally:
        api.EXACT_TILES = old
        api._cache = api.LRU(256)
    assert not exact["extrapolated"]
    assert abs(small["time_ns"]["nominal"] - exact["time_ns"]["nominal"]) <= 1e-6 * exact["time_ns"]["nominal"]


def test_tune_evaluates_every_candidate():
    out = analyze(req(action="tune", tune={}, analysis={"time_budget_ms": 60000}))
    s = out["search"]
    assert s["exhaustive"] and out["status"] == "ok"
    assert s["evaluated"] == s["considered"] == s["valid"]


# ---- hypothetical schedule variants

L1 = {"a": "l1", "b": "l1", "out": "l1"}


def _variant_times(costs, placements, overrides, bases):
    out = []
    for base in bases:
        r = analyze(req(action="compare", profiles={"costs": costs}, baseline=base,
                        workload=dict(req()["workload"], placements=placements),
                        candidates=[{"id": "v", "overrides": overrides}]))
        out.append((r["baseline"]["time_ns"]["nominal"], r["candidates"][0]))
    return out


BASES = [{"reader_threads": r, "compute_threads": 4, "writer_threads": w, "dm_batch": b, "tiles_per_cycle": b,
          "entries_per_thread": 2 * b} for r, w, b in ((4, 2, 1), (2, 2, 2), (4, 2, 4), (1, 4, 8))]


def test_per_thread_barriers_never_slower_on_timed_transfers():
    for base_t, cand in _variant_times("ttsim-perf", L1, {"barriers": "per_thread"}, BASES):
        assert cand["time_ns"]["nominal"] <= base_t * (1 + 1e-9)
        assert "barriers=per_thread" in cand["label"]
        assert any("hypothetical kernel change" in w for w in cand["warnings"])


def test_per_thread_barriers_change_nothing_when_transfers_are_untimed():
    for base_t, cand in _variant_times("craq-sim", req()["workload"]["placements"], {"barriers": "per_thread"}, BASES):
        assert cand["time_ns"]["nominal"] == base_t


def test_dynamic_split_pays_its_claims():
    w = req()["workload"]
    times = [analyze(req(baseline=dict(BASES[2], work_split="dynamic"), workload=w,
                         what_if={"cost_overrides": {"dq.claim": v}}))["baseline"]["time_ns"]["nominal"]
             for v in (0, 30, 300)]
    assert times[0] <= times[1] < times[2]


def test_variant_extrapolation_matches_tile_by_tile():
    from qperf import api
    for knobs in ({"barriers": "per_thread"}, {"work_split": "dynamic"}):
        r = req(profiles={"costs": "silicon-hypothesis"}, baseline=dict(BASES[2], **knobs),
                workload=dict(req()["workload"], num_tiles=5000, clusters=1))
        small = analyze(r)["baseline"]
        old = api.EXACT_TILES
        try:
            api.EXACT_TILES = 1 << 20
            api._cache = api.LRU(256)
            exact = analyze(r)["baseline"]
        finally:
            api.EXACT_TILES = old
            api._cache = api.LRU(256)
        assert small["extrapolated"] and not exact["extrapolated"]
        assert abs(small["time_ns"]["nominal"] / exact["time_ns"]["nominal"] - 1) < 1e-3, knobs
