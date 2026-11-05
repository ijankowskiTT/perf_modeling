"""survey: approaches benched across a grid of shapes, per approach where it wins and where it regresses; the response
contract, plan dedupe, skipped shapes, the time budget and the reports. Kernel changes (tilize / untilize design knobs)
and layout choices (shard options of the sharding ops and reshard, with untimed sides marked)."""

import json
import re

import pytest

from qperf import analyze
from qperf.api import NOISE
from qperf.cli import _text
from qperf.report import to_html, to_markdown
from qperf.survey import DEFAULT_APPROACHES, DEFAULT_HEIGHTS, DEFAULT_WIDTHS, RESHARD_CHOICES, SHARD_CHOICES
from qperf.validate import validate_request, validate_response


def req(op="tilize", costs="craq-sim", place="dram", **kw):
    r = {"schema_version": "1", "action": "survey", "profiles": {"costs": costs},
         "workload": {"op": op, "dtype": "bf16", "placements": {"a": place}}}
    r.update(kw)
    return r


_cache = {}


def run(op="tilize", costs="craq-sim"):
    if (op, costs) not in _cache:
        _cache[(op, costs)] = analyze(req(op, costs))
    return _cache[(op, costs)]


def cells(sv, grid):
    return {(h, w): grid[i][j] for i, h in enumerate(sv["heights"]) for j, w in enumerate(sv["widths"])}


@pytest.mark.parametrize("op", ["tilize", "untilize"])
@pytest.mark.parametrize("costs", ["craq-sim", "ttsim-perf"])
def test_default_survey_runs_and_validates(op, costs):
    r = run(op, costs)
    assert r["status"] == "ok" and validate_response(r) == []
    sv = r["survey"]
    assert sv["heights"] == DEFAULT_HEIGHTS and sv["widths"] == DEFAULT_WIDTHS
    assert list(sv["approaches"]) == [a for a, _ in DEFAULT_APPROACHES]
    assert len(sv["summary"]) == len(DEFAULT_APPROACHES) + bool(sv["stats"]["outside_calibrated_range"])
    assert r["runtime_ms"] < 15000
    assert sum("hypothetical layout designs" in w for w in r["warnings"]) == 1
    assert r["platform_scope"] and r["recommendations"]
    batch = "write_batch" if op == "tilize" else "read_batch"
    assert sv["approaches"]["batch8"]["overrides"] == {batch: 8}  # the generic "batch" maps to the factory's knob


@pytest.mark.parametrize("op", ["tilize", "untilize"])
def test_counts_regressions_and_extremes_agree_with_the_grid(op):
    sv = run(op)["survey"]
    base = cells(sv, sv["baseline_ns"])
    evaluated = [k for k, v in base.items() if v is not None]
    assert sv["stats"]["evaluated"] == len(evaluated) == len(base) - len(sv["skipped"])
    for aid, a in sv["approaches"].items():
        c = a["counts"]
        g = cells(sv, a["grid"])
        assert c["faster"] + c["neutral"] + c["slower"] + c["invalid"] == len(evaluated)
        vals = {k: v for k, v in g.items() if v is not None}
        assert set(vals) <= set(evaluated) and len(vals) == len(evaluated) - c["invalid"]
        assert c["faster"] == sum(v > 1 + NOISE for v in vals.values())
        assert c["slower"] == sum(v < 1 - NOISE for v in vals.values())
        regs = sorted((v, k) for k, v in vals.items() if v < 1 - NOISE)
        assert [r["speedup"] for r in a["regressions"]] == [v for v, _ in regs[:5]]
        assert all(g[tuple(r["shape"])] == r["speedup"] for r in a["regressions"])
        assert a["speedup"]["min"] == min(vals.values()) == a["worst"]["speedup"] == g[tuple(a["worst"]["shape"])]
        assert a["speedup"]["max"] == max(vals.values()) == a["best"]["speedup"] == g[tuple(a["best"]["shape"])]
    best = cells(sv, sv["best_per_shape"])
    for k, b in best.items():
        if base[k] is None:
            assert b is None
        elif b == "baseline":
            assert all((cells(sv, a["grid"])[k] or 0) <= 1 + NOISE for a in sv["approaches"].values())
        else:
            top = max(cells(sv, a["grid"])[k] or 0 for a in sv["approaches"].values())
            assert cells(sv, sv["approaches"][b]["grid"])[k] > 1 + NOISE and top / cells(sv, sv["approaches"][b]["grid"])[k] - 1 <= NOISE


def test_a_policy_equal_to_the_factory_is_neutral_everywhere():
    r = analyze(req(candidates=[{"id": "factory", "overrides": {"cores": 32, "split": "rows", "write_batch": 1,
                                                                 "dfb_blocks": 1}},
                                {"id": "cols", "overrides": {"split": "cols"}}]))
    assert r["status"] == "ok" and validate_response(r) == []
    a = r["survey"]["approaches"]["factory"]
    n = r["survey"]["stats"]["evaluated"]
    assert a["hypothetical"] is False and a["counts"] == {"faster": 0, "neutral": n, "slower": 0, "invalid": 0}
    assert all(v == 1.0 for row in a["grid"] for v in row)
    assert a["regressions"] == [] and a["speedup"] == {"geomean": 1.0, "min": 1.0, "max": 1.0}
    assert "same plan as the baseline" in next(s for s in r["survey"]["summary"] if s.startswith("factory:"))
    assert r["survey"]["approaches"]["cols"]["hypothetical"] is True
    # untilize already double-buffers: dfb_blocks 2 is its factory plan
    u = run("untilize")["survey"]["approaches"]["dfb2"]
    assert u["hypothetical"] is False and u["counts"]["neutral"] == run("untilize")["survey"]["stats"]["evaluated"]


def test_plans_equal_across_approaches_are_timed_once():
    st = run()["survey"]["stats"]
    n = st["evaluated"] * (1 + len(DEFAULT_APPROACHES))
    assert st["plans_evaluated"] + st["plans_reused"] < n  # neutral approaches never re-time the baseline plan
    assert st["plans_reused"] > 0


def test_cols_all_wins_wide_short_and_does_not_help_tall_narrow():
    sv = run()["survey"]
    g = cells(sv, sv["approaches"]["cols_all"]["grid"])
    assert g[(64, 2048)] > 1.5  # 2 tile-rows: the factory uses 2 cores, cols_all all 32
    assert g[(8192, 64)] <= 1 + NOISE  # 256 tile-rows already fill the grid: nothing to gain
    assert sv["approaches"]["cols_all"]["wins_where"] == "H<=512"
    assert sv["best_per_shape"][DEFAULT_HEIGHTS.index(64)][DEFAULT_WIDTHS.index(2048)] != "baseline"


def test_half_blocks_regresses_and_is_named_in_the_summary():
    sv = run()["survey"]
    a = sv["approaches"]["half_blocks"]
    assert a["counts"]["slower"] > 0 and len(a["regressions"]) == 5
    line = next(s for s in sv["summary"] if s.startswith("half_blocks:"))
    w = a["worst"]
    assert f"worst {w['speedup']:.2f}x at {w['shape'][0]}x{w['shape'][1]}" in line and "slower on" in line


def test_recommendations_are_never_slower_on_most_shapes():
    r = run()
    sv = r["survey"]
    for rec in r["recommendations"]:
        a = sv["approaches"][rec["id"]]
        n = a["counts"]["faster"] + a["counts"]["neutral"] + a["counts"]["slower"]
        assert a["counts"]["slower"] == 0 and a["counts"]["faster"] > n / 2
    assert "half_blocks" not in {x["id"] for x in r["recommendations"]}


def test_untilize_wide_routes_to_block_and_is_skipped():
    sv = run("untilize")["survey"]
    skipped = {tuple(s["shape"]): s["reason"] for s in sv["skipped"]}
    assert (32, 2048) in skipped and "Block" in skipped[(32, 2048)]
    j = sv["widths"].index(2048)
    assert sv["baseline_ns"][0][j] is None and sv["best_per_shape"][0][j] is None
    assert all(a["grid"][0][j] is None for a in sv["approaches"].values())


def test_craq_sim_skips_tilize_wider_than_64_tiles_custom_grid():
    r = analyze(req(survey={"heights": [32, 1024], "widths": [512, 4096]},
                    candidates=[{"id": "b4", "overrides": {"write_batch": 4}}, {"id": "half", "overrides": {"tiles_per_block": "half"}}]))
    assert validate_request(req(survey={"heights": [32]}, candidates=[{"overrides": {"tiles_per_block": "half"}}])) == []
    assert r["status"] == "ok" and validate_response(r) == []
    sv = r["survey"]
    assert {tuple(s["shape"]) for s in sv["skipped"]} == {(32, 4096), (1024, 4096)}
    assert "64 tiles" in sv["skipped"][0]["reason"]
    assert sv["approaches"]["b4"]["overrides"] == {"write_batch": 4}
    t = analyze(req(costs="ttsim-perf", survey={"heights": [32], "widths": [4096]}))
    assert t["status"] == "ok" and t["survey"]["skipped"] == []


def test_time_budget_returns_partial():
    r = analyze(req(analysis={"time_budget_s": 0}))
    assert r["status"] == "partial" and validate_response(r) == []
    st = r["survey"]["stats"]
    assert st["out_of_time"] and len(st["not_evaluated"]) == 54 - st["evaluated"] - st["skipped"]
    assert any("time budget" in w for w in r["warnings"])
    to_html(req(), r), to_markdown(req(), r), _text(r)


@pytest.mark.parametrize("op", ["add", "tilize_with_val_padding", "untilize_with_unpadding"])
def test_unsupported_ops(op):
    r = analyze(req(op))
    assert r["status"] == "unsupported" and validate_response(r) == []
    assert "survey" in r["message"] and "binary_ng" in r["message"]


def test_all_skipped_is_unsupported():
    r = analyze(req(baseline={"use_multicore": False}))
    assert r["status"] == "unsupported" and "SingleCore" in r["message"]


def test_reports_render_a_heatmap_per_approach_and_the_best_map():
    q, r = req(), run()
    html = to_html(q, r)
    n = len(r["survey"]["approaches"])
    assert len(re.findall(r'<svg class="heatmap"', html)) == n
    assert all(f'data-approach="{aid}"' in html for aid in r["survey"]["approaches"])
    assert html.count('class="bestmap"') == 1 and "Best approach per shape" in html
    assert "--win" in html and "--loss" in html and "prefers-color-scheme:dark" in html
    assert html.count('class="cell') >= n * 54
    assert "Best general approach" in html and r["recommendations"][0]["id"] in html
    assert r["platform_scope"].split(" ")[0] in html and "<summary>Request</summary>" in html
    md = to_markdown(q, r)
    assert "## Approaches" in md and "## Regressions" in md and "half_blocks" in md.split("## Regressions")[1]
    u = run("untilize")
    assert "hatch" in to_html(req("untilize"), u) and "## Skipped shapes" in to_markdown(req("untilize"), u)
    txt = _text(r)
    assert "faster/neutral/slower/invalid" in txt and all(s in txt for s in r["survey"]["summary"])


# ------------------------------------------------------------------------------------------------ layout choices

def choice(op, costs="ttsim-perf", place="dram"):
    key = (op, costs, place, "choice")
    if key not in _cache:
        r = req(op, costs, place)
        if op == "reshard":
            r["workload"].pop("placements")
        _cache[key] = analyze(r)
    return _cache[key]


def test_survey_kinds():
    assert run("tilize", "ttsim-perf")["survey"]["kind"] == "kernel_change"
    assert all(a["kind"] == "kernel_change" for a in run("tilize", "ttsim-perf")["survey"]["approaches"].values())
    for op in ("interleaved_to_sharded", "sharded_to_interleaved", "reshard"):
        sv = choice(op)["survey"]
        assert sv["kind"] == "layout_choice"
        assert all(a["kind"] == "layout_choice" and a["hypothetical"] is False for a in sv["approaches"].values())


@pytest.mark.parametrize("op", ["interleaved_to_sharded", "sharded_to_interleaved"])
def test_sharding_survey_defaults_skips_and_untimed_dram_side(op):
    r = choice(op)
    assert r["status"] == "ok" and validate_response(r) == []
    assert r["runtime_ms"] < 3000
    sv = r["survey"]
    assert sv["heights"] == DEFAULT_HEIGHTS and sv["widths"] == DEFAULT_WIDTHS
    assert list(sv["approaches"]) == [a for a, _ in SHARD_CHOICES]
    assert "survey baseline shard height x8 (default)" in r["applied_defaults"]
    assert sv["baseline"]["factory"] == "Default" and sv["baseline"]["untimed"] == sv["stats"]["evaluated"]
    # height x8 needs 8 tile rows: H < 256 is skipped with the reason; width x16 needs W >= 512
    assert {tuple(s["shape"]) for s in sv["skipped"]} == {(h, w) for h in (32, 64, 128) for w in DEFAULT_WIDTHS}
    assert "empty shards" in sv["skipped"][0]["reason"]
    w16 = cells(sv, sv["approaches"]["width_x16"]["grid"])
    assert w16[(1024, 256)] is None and w16[(1024, 512)] is not None
    # ttsim-perf does not time DRAM: every compared cell is untimed and no option is recommended
    for a in sv["approaches"].values():
        ug = cells(sv, a["untimed"])
        assert {k for k, v in ug.items() if v} == {k for k, v in cells(sv, a["grid"]).items() if v is not None}
        assert a["untimed_cells"] == a["counts"]["faster"] + a["counts"]["neutral"] + a["counts"]["slower"]
    assert any("not meaningful" in w and "DRAM" in w for w in r["warnings"])
    assert any("compared cells involve a side ttsim-perf does not time" in w for w in r["warnings"])
    assert r["recommendations"][0]["id"] is None and r["recommendations"][0]["label"] == "inconclusive"
    assert "untimed on" in sv["summary"][0]
    assert r["accuracy"].startswith(f"{op} Default on ttsim-perf")


def test_sharding_survey_from_l1_is_timed_and_recommends():
    r = choice("interleaved_to_sharded", place="l1")
    sv = r["survey"]
    assert r["status"] == "ok" and r["runtime_ms"] < 3000
    assert all(a["untimed_cells"] == 0 for a in sv["approaches"].values())
    assert not any("not meaningful" in w for w in r["warnings"])
    assert r["recommendations"][0]["id"] in ("height_x32", "width_x16", "height_x16", "block_x16")
    # width x8 runs another plan with the same per-core tiles as height x8: neutral, but not "the same plan"
    assert "another plan, same time" in next(s for s in sv["summary"] if s.startswith("width_x8"))


def test_reshard_survey_options_factories_and_accuracy():
    r = choice("reshard")
    assert r["status"] == "ok" and validate_response(r) == [] and r["runtime_ms"] < 3000
    sv = r["survey"]
    assert list(sv["approaches"]) == [a for a, _ in RESHARD_CHOICES]
    assert "survey baseline reshard height x8 L1 -> height x8 L1 (default)" in r["applied_defaults"]
    assert sv["baseline"]["factory"] == "SameWidth" and sv["baseline"]["untimed"] == 0
    aps = sv["approaches"]
    assert aps["w8_to_w8"]["factory"] == "Generic" and aps["b4x4_to_b4x4"]["factory"] == "Generic"
    assert aps["h8_to_h16"]["factory"] == "SameWidth"
    assert "Generic" in r["accuracy"] and any("known gap" in w and "w8_to_w8" in w for w in r["warnings"])
    h4 = aps["h8_to_h4"]["counts"]
    assert h4["slower"] == h4["faster"] + h4["neutral"] + h4["slower"]  # fewer output cores: slower everywhere
    assert aps["h32_to_h32"]["speedup"]["geomean"] > 1.5
    assert {x["id"] for x in r["recommendations"]} <= {"h32_to_h32", "h16_to_h16", "b4x4_to_b4x4"}
    assert "height x32 L1 -> height x32 L1" in r["recommendations"][0]["label"]


def test_reshard_survey_on_craq_sim_is_untimed():
    r = choice("reshard", "craq-sim")
    assert all(a["untimed_cells"] > 0 for a in r["survey"]["approaches"].values())
    assert any("per request" in w for w in r["warnings"]) and r["recommendations"][0]["label"] == "inconclusive"


def test_silicon_hypothesis_survey_counts_dram_capped_shapes():
    t = run("tilize", "silicon-hypothesis")["survey"]
    assert t["baseline"]["dram_capped"] > 0 and all("dram_capped" in a for a in t["approaches"].values())
    assert "dram_capped" not in run("tilize", "ttsim-perf")["survey"]["baseline"]
    s = choice("interleaved_to_sharded", "silicon-hypothesis")["survey"]
    assert s["baseline"]["dram_capped"] > 0 and s["baseline"]["untimed"] == 0
    assert s["baseline"]["limiters"].get("dram_global") == s["baseline"]["dram_capped"]


def test_choice_reports_mark_untimed_cells():
    q, r = req("interleaved_to_sharded", "ttsim-perf"), choice("interleaved_to_sharded")
    html = to_html(q, r)
    assert html.count(' untimed"') >= r["survey"]["stats"]["evaluated"] and "†" in html
    assert "untimed: the profile does not time a side" in html and "<span class=tag>untimed</span>" in html
    md = to_markdown(q, r)
    assert "(untimed on" in md and "† untimed" in md and "height x16" in md
    page = to_html(req("reshard", "ttsim-perf"), choice("reshard"))
    assert "height x8 L1 -&gt; height x16 L1" in page and ' untimed"' not in page
