"""Every response shape (each action, status and detail level) conforms to schema/response.schema.json."""

import glob
import json
import os

import pytest

from qperf import analyze
from qperf.validate import validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = {"op": "add", "dtype": "bf16", "num_tiles": 4096, "placements": {"a": "dram", "b": "dram", "out": "dram"}}


def req(**kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": "ttsim-perf"}, "workload": W, "baseline": {}}
    r.update(kw)
    return r


CASES = {
    "estimate": req(),
    "estimate-full": req(analysis={"detail": "full"}),
    "full": req(analysis={"detail": "full"}, workload=dict(W, num_tiles=256)),
    "multiply": req(workload=dict(W, op="multiply")),
    "unknown-op": req(workload=dict(W, op="divide")),
    "no-costs": req(profiles={}),
    "bad-profile": req(profiles={"costs": "nope"}),
    "invalid-baseline": req(baseline={"compute_threads": 3}),
    "l1": req(workload=dict(W, placements={"a": "l1", "b": "l1", "out": "l1"})),
    "borrowed": req(workload=dict(W, num_tiles=256, placements={"a": "borrowed", "b": "borrowed", "out": "borrowed"})),
    "sens-tiles": req(action="sensitivity", sensitivity={"param": "num_tiles", "values": [64, 1024, 8192]},
                      candidates=[{"id": "x", "overrides": {"dm_batch": 1}}]),
    "sens-compute": req(action="sensitivity", sensitivity={"param": "compute_service_scale", "values": [0.5, 1, 2]}),
    "sens-cost": req(action="sensitivity", sensitivity={"param": "noc.read_latency", "values": [0, 500]},
                     profiles={"costs": "silicon-hypothesis"}),
    "sens-missing": req(action="sensitivity"),
    "hypothetical": req(hypothetical={"knob_limits": {"dm_batch": 16}}, action="compare",
                        candidates=[{"id": "b16", "overrides": {"dm_batch": 16}}]),
    "compare-invalid-cand": req(action="compare", candidates=[{"overrides": {"dm_batch": 16}}]),
    "tune": req(action="tune", tune={"max_candidates": 4}),
    "tune-budget": req(action="tune", analysis={"time_budget_ms": 1}),
    "garbage": {"bogus": 1},
    "variants": req(action="compare", workload=dict(W, placements={"a": "l1", "b": "l1", "out": "l1"}),
                    candidates=[{"id": "pt", "overrides": {"barriers": "per_thread"}},
                                {"id": "dq", "overrides": {"work_split": "dynamic"}}]),
    "tune-variants": req(action="tune", tune={"space": {"barriers": ["cluster", "per_thread"], "reader_threads": [2, 4],
                                                        "compute_threads": [4], "writer_threads": [2]}}),
    "what-if": req(profiles={"costs": "craq-sim"},
                   what_if={"cost_overrides": {"wr.issue": 20}, "compute_service_scale": 0.5}),
}
for f in sorted(glob.glob(os.path.join(ROOT, "skills", "quasar-perf", "examples", "*.json"))):
    CASES["example-" + os.path.basename(f)] = json.load(open(f))


@pytest.mark.parametrize("name", sorted(CASES))
def test_response_conforms_to_schema(name):
    out = analyze(CASES[name])
    assert validate_response(out) == [], out.get("status")


def test_schema_catches_a_drifted_field():
    out = analyze(CASES["estimate"])
    out["baseline"]["speedup"] = 2.0
    out["status"] = "great"
    errs = validate_response(out)
    assert any("unknown field 'speedup'" in e for e in errs) and any("'great'" in e for e in errs)
