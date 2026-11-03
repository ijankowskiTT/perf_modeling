"""The skill's example requests stay valid and produce honest results (no fabricated numbers on failure)."""

import glob
import json
import os

import pytest

from qperf import analyze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "skills", "quasar-perf", "examples", "*.json")))


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_example_runs_ok(path):
    req = json.load(open(path))
    out = analyze(req)
    assert out["status"] in ("ok", "partial"), out.get("message")
    assert out["runtime_ms"] < 5000
    assert out["profiles"]["costs"]["sha256"]
    if req["action"] == "tune":
        assert len(out["top"]) <= req.get("analysis", {}).get("top_k", 3)
        assert "exhaustive" in out["search"]


def test_skill_doc_points_at_existing_things():
    skill = open(os.path.join(ROOT, "skills", "quasar-perf", "SKILL.md")).read()
    assert "schema/request.schema.json" in skill and os.path.exists(os.path.join(ROOT, "schema", "request.schema.json"))
    for name in ("ttsim-perf", "craq-sim", "silicon-hypothesis"):
        assert name in skill
        assert os.path.exists(os.path.join(ROOT, "profiles", "costs", name + ".json"))


@pytest.mark.parametrize("mutate,status", [
    (lambda r: r["profiles"].update(costs="nope"), "unsupported"),
    (lambda r: r["workload"].update(op="atan2"), "unsupported"),
    (lambda r: r["workload"].update(op="subtract"), "needs_input"),
    (lambda r: r["baseline"].update(compute_threads=3), "invalid"),
])
def test_failures_carry_no_numbers(mutate, status):
    req = json.load(open(EXAMPLES[0]))
    mutate(req)
    out = analyze(req)
    assert out["status"] == status
    if status != "invalid":
        assert "baseline" not in out and "time_ns" not in json.dumps(out)
