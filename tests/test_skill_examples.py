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


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_example_reports_render(path):
    from qperf.report import to_html, to_markdown

    req = json.load(open(path))
    out = analyze(req)
    md, page = to_markdown(req, out), to_html(req, out)
    assert md.startswith("# qperf report") and "## Key points" in md
    assert page.startswith("<!doctype html>") and "<svg" in page and "<script" not in page
    from qperf.cli import _text

    assert _text(out).startswith("status: ok")
    assert out["baseline"]["label"] in page


def test_report_of_a_failed_request_has_no_numbers():
    from qperf.report import to_html, to_markdown

    req = json.load(open(EXAMPLES[0]))
    req["workload"]["op"] = "atan2"
    out = analyze(req)
    assert " ns" not in to_markdown(req, out) and "<svg" not in to_html(req, out)


def test_skill_and_agent_point_at_existing_things():
    import re
    from qperf.profiles import load

    skill = open(os.path.join(ROOT, "skills", "quasar-perf", "SKILL.md")).read()
    agent = open(os.path.join(ROOT, "agents", "qperf-runner.md")).read()
    assert "qperf:qperf-runner" in skill and "agents/qperf-runner.md" in skill
    assert re.search(r"^model: (haiku|sonnet|opus)", agent, re.M)
    for name in ("ttsim-perf", "craq-sim", "silicon-hypothesis"):
        assert name in agent and os.path.exists(os.path.join(ROOT, "profiles", "costs", name + ".json"))
    params = load("costs", "silicon-hypothesis")[0]["params"]
    for key in re.findall(r"(noc\.\w+\.dram|dram\.\w+)", agent):
        assert key in params, key
    # the agent's request template is a valid request that runs
    tmpl = agent[agent.index('{"schema_version"'):]
    tmpl = json.loads(tmpl[:tmpl.index("}}}") + 3])
    assert analyze(tmpl)["status"] == "ok"


@pytest.mark.parametrize("mutate,status", [
    (lambda r: r["profiles"].update(costs="nope"), "unsupported"),
    (lambda r: r["workload"].update(op="atan2"), "unsupported"),
    (lambda r: r["workload"].update(op="maximum"), "unsupported"),
    (lambda r: r["workload"].pop("num_tiles"), "needs_input"),
    (lambda r: r["baseline"].update(compute_threads=3), "invalid"),
])
def test_failures_carry_no_numbers(mutate, status):
    req = json.load(open(EXAMPLES[0]))
    mutate(req)
    out = analyze(req)
    assert out["status"] == status
    if status != "invalid":
        assert "baseline" not in out and "time_ns" not in json.dumps(out)
