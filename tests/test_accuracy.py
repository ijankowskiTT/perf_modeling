"""How answers state their accuracy and limits: the op-specific accuracy field (profiles/accuracy.json kept current),
extrapolation beyond the calibrated shape range, survey.compare_profile and the craq-sim small-request warning, and
the assumed-DRAM-bandwidth bound of data-movement ops on silicon-hypothesis."""

import copy
import json
import os
import re
import sys

import pytest

from qperf import accuracy, analyze, profiles
from qperf.cli import _text
from qperf.report import to_html, to_markdown
from qperf.validate import validate_request, validate_response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "calibration"))
G2D = {"split": "2d", "dfb_blocks": 2, "write_batch": 8}  # the survey's grid2d_dfb2_batch8


def layout_req(op="tilize", shape=(2048, 512), costs="ttsim-perf", place="dram", **kw):
    r = {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
         "workload": {"op": op, "dtype": "bf16", "shape": list(shape), "placements": {"a": place}}}
    r.update(kw)
    return r


def binary_req(op="add", costs="ttsim-perf", a="dram"):
    return {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs},
            "workload": {"op": op, "dtype": "bf16", "num_tiles": 4096, "placements": {"a": a, "b": "dram", "out": "dram"}}}


def survey_req(costs="ttsim-perf", **kw):
    r = {"schema_version": "1", "action": "survey", "profiles": {"costs": costs},
         "workload": {"op": "tilize", "dtype": "bf16", "placements": {"a": "dram"}}}
    r.update(kw)
    return r


# ------------------------------------------------------------------------------------------------ accuracy table

def test_accuracy_table_is_current():
    """profiles/accuracy.json == calibration/make_accuracy.py recomputed now (errors within 0.1 pt, ranges exact)."""
    from make_accuracy import compute

    stored = json.load(open(os.path.join(ROOT, "profiles", "accuracy.json")))["profiles"]
    fresh = compute()
    assert set(stored) == set(fresh)
    for prof, t in fresh.items():
        assert stored[prof]["revision"] == t["revision"] == profiles.load("costs", prof)[0]["revision"]
        assert set(stored[prof]["factories"]) == set(t["factories"]), prof
        for key, f in t["factories"].items():
            s = stored[prof]["factories"][key]
            assert {k: s[k] for k in ("runs", "max_h", "max_w", "max_elems")} == \
                   {k: f[k] for k in ("runs", "max_h", "max_w", "max_elems")}, (prof, key)
            assert set(s["error"]) == set(f["error"])
            for dt, e in f["error"].items():
                assert s["error"][dt]["runs"] == e["runs"] and abs(s["error"][dt]["mean"] - e["mean"]) <= 0.001, \
                    (prof, key, dt)
    from make_accuracy import validation

    stored = json.load(open(os.path.join(ROOT, "profiles", "accuracy.json")))["hypothetical_validation"]
    fresh = validation()
    assert set(stored) == set(fresh) and "tilize Default write_batch" in fresh
    for key, per in fresh.items():
        assert set(stored[key]) == set(per)
        for prof, v in per.items():
            s = stored[key][prof]
            assert (s["runs"], s["values"], s["data"]) == (v["runs"], v["values"], v["data"]), (key, prof)
            assert abs(s["mean"] - v["mean"]) <= 0.001 and abs(s["speedup_mean"] - v["speedup_mean"]) <= 0.001


def test_platform_scope_carries_no_op_error():
    for name in ("craq-sim", "ttsim-perf", "silicon-hypothesis"):
        scope = profiles.load("costs", name)[0]["platform_scope"]
        assert "mean error" not in scope and not re.search(r"\d%", scope), name


@pytest.mark.parametrize("req_,want,not_want", [
    (layout_req(), "tilize Default on ttsim-perf: 1.2% mean error over 56 bf16", "8.5%"),
    (layout_req("untilize", costs="craq-sim"), "untilize MultiCore on craq-sim: 0.4% mean error", "2.1%"),
    (layout_req("untilize", (256, 8192)), "untilize Block on ttsim-perf: 0.7%", "8.5%"),
    (dict(layout_req(), workload=dict(layout_req()["workload"], dtype="fp32")), "1.4% mean error over 56 fp32", "1.2%"),
    (binary_req("add", "craq-sim"), "binary_ng add bf16 on craq-sim: 2.1%", "1.4%"),
    (binary_req("multiply", "craq-sim"), "binary_ng multiply bf16 on craq-sim: 1.4%", "2.1%"),
    (binary_req("add", "ttsim-perf", "l1"), "8.5% mean error with an L1-interleaved operand, as here", None),
    (binary_req("add", "ttsim-perf"), "1.4% mean error with DRAM / borrowed operands, as here", None),
    (binary_req("add", "silicon-hypothesis"), "silicon-hypothesis is a hypothesis, no measurement", None),
])
def test_accuracy_is_op_specific(req_, want, not_want):
    r = analyze(req_)
    assert r["status"] == "ok" and validate_response(r) == []
    assert want in r["accuracy"], r["accuracy"]
    assert not_want is None or not_want not in r["accuracy"]
    assert f"accuracy: {r['accuracy']}" in _text(r)
    assert f"**Accuracy:** {r['accuracy']}" in to_markdown(req_, r)


def test_hypothetical_rows_say_so():
    r = analyze(layout_req(action="compare", candidates=[{"id": "g", "overrides": G2D}, {"id": "sc", "overrides":
                                                                                         {"use_multicore": False}}]))
    assert validate_response(r) == []
    g, sc = r["candidates"]
    # dfb_blocks has no simulator runs (split 2d resolves to rows at 2048x512); write_batch has: partly validated
    assert g["accuracy"] == ("hypothetical setup, partly validated: write_batch validated: the factory's own kernels "
                             "with only write_batch changed (4, 8) ran 24 times on ttsim-perf over 12 shapes "
                             "(H 32-4096, W 256-2048): 0.8% mean error, 1.0% on the speedup over write_batch=1; "
                             "dfb_blocks: no simulator measurement; built from costs with 1.2% "
                             "error on the factory (tilize Default, ttsim-perf)")
    assert sc["accuracy"].startswith("tilize SingleCore on ttsim-perf: 1.1% mean error")
    assert r["baseline"]["accuracy"] == r["accuracy"]
    assert f"accuracy: {g['accuracy']}" in _text(r)


# -------------------------------------------------------------------------------------------------- extrapolation

def test_shape_beyond_the_calibrated_range_is_flagged():
    r = analyze(layout_req(shape=(8192, 8192)))
    assert validate_response(r) == []
    b = r["baseline"]
    assert b["outside_calibrated_range"] is True
    assert ("shape 8192x8192 is outside the calibrated range for tilize Default on ttsim-perf (max 4096x4096, "
            "8.4M elements, 112 runs): extrapolated") in b["warnings"]
    for shape in ((4096, 2048), (5120, 1024), (2048, 5120), (2048, 512)):  # inside, or within the 1.25x tolerance
        b = analyze(layout_req(shape=shape))["baseline"]
        assert "outside_calibrated_range" not in b and not any("calibrated range" in w for w in b["warnings"]), shape
    # elements alone: 4096 x 2560 rows / width inside, 10.5M elements > 1.25 x 8.4M
    assert analyze(layout_req(shape=(4096, 2600)))["baseline"].get("outside_calibrated_range")


def test_range_check_covers_registered_ops_and_derived_profiles():
    rng = accuracy.calibrated_range(profiles.load("costs", "ttsim-perf")[0], "move", "Overlap")
    assert rng and rng["runs"] > 0
    move = {"schema_version": "1", "action": "estimate", "profiles": {"costs": "ttsim-perf"},
            "workload": {"op": "move", "dtype": "bf16", "shape": [4096, 1024], "placements": {"a": "dram"}}}
    b = analyze(move)["baseline"]
    assert b.get("outside_calibrated_range") and "for move Overlap on ttsim-perf" in " ".join(b["warnings"])
    b = analyze(layout_req(shape=(8192, 8192), costs="silicon-hypothesis"))["baseline"]
    assert "on ttsim-perf, whose costs silicon-hypothesis uses" in " ".join(b["warnings"])


def test_survey_marks_extrapolated_cells_once():
    r = analyze(survey_req())
    sv = r["survey"]
    prof = profiles.load("costs", "ttsim-perf")[0]
    rng = accuracy.calibrated_range(prof, "tilize", "Default")
    assert sv["calibrated_range"] == rng
    want = [[accuracy.outside(rng, h, w) for w in sv["widths"]] for h in sv["heights"]]
    assert sv["outside_calibrated_range"] == want
    n = sum(map(sum, want))
    assert n == sv["stats"]["outside_calibrated_range"] > 0
    note = [x for x in sv["summary"] if "outside the calibrated range" in x]
    assert len(note) == 1 and note[0].startswith(f"{n} of 54 evaluated shapes")
    assert sum("outside the calibrated range" in w for w in r["warnings"]) == 1
    html = to_html(survey_req(), r)
    heat = "".join(re.findall(r'<div class="heatmap hmap".*?</div>', html))
    cells = [c for c in re.findall(r'<i class="([^"]*)"', heat) if "x" in c.split()]
    assert len(cells) == n * len(sv["approaches"]) and "xhatch" in html
    assert " *" in to_markdown(survey_req(), r).split("## Best approach per shape")[1]
    assert f"{n} extrapolated" in _text(r)


# ------------------------------------------------------------------------------------------ survey compare_profile

def test_survey_compare_profile():
    q = survey_req(survey={"compare_profile": "craq-sim"})
    assert validate_request(q) == []
    r = analyze(q)
    assert r["status"] == "ok" and validate_response(r) == []
    assert r["runtime_ms"] < 2000  # both default tilize surveys
    cmp = r["survey"]["compare"]
    alone = analyze(survey_req("craq-sim"))["survey"]
    assert cmp["profile"]["name"] == "craq-sim" and set(cmp["approaches"]) == set(r["survey"]["approaches"])
    for aid, a in alone["approaches"].items():
        assert cmp["approaches"][aid]["speedup"] == a["speedup"]
    # craq-sim charges per request: small-request setups look better there than on ttsim-perf
    g = "grid2d_dfb2_batch8"
    assert cmp["approaches"][g]["speedup"]["max"] > 1.5 * r["survey"]["approaches"][g]["speedup"]["max"]
    assert any(w.startswith("compare_profile craq-sim: craq-sim charges each NoC request") for w in r["warnings"])
    txt = _text(r)
    assert "craq-sim geomean, min..max" in txt and "compare_profile craq-sim@" in txt
    md = to_markdown(q, r)
    assert "craq-sim geomean" in md and "## On craq-sim (compare_profile)" in md
    assert "craq-sim min..max" in to_html(q, r)
    assert analyze(survey_req(survey={"compare_profile": "no-such"}))["status"] == "unsupported"


def test_craq_small_request_warning():
    def warned(r):
        return [w for w in r["warnings"] if "craq-sim charges each NoC request" in w]

    w = warned(analyze(survey_req("craq-sim")))
    ids = re.search(r"; here (.*)\) look better", w[0]).group(1).split(", ") if w else []
    assert len(w) == 1 and ids == ["cols_all", "grid2d", "grid2d_dfb2_batch8", "half_blocks"]
    assert "ttsim-perf gives the more conservative number" in w[0]
    assert not warned(analyze(survey_req("ttsim-perf")))
    assert not warned(analyze(survey_req("craq-sim", candidates=[{"id": "b8", "overrides": {"write_batch": 8}}])))
    tune = analyze(layout_req(shape=(64, 2048), costs="craq-sim", action="tune"))
    assert validate_response(tune) == [] and len(warned(tune)) == 1
    cmp = layout_req(shape=(64, 2048), costs="craq-sim", action="compare",
                     candidates=[{"id": "cols", "overrides": {"split": "cols", "cores": 32}},
                                 {"id": "b8", "overrides": {"write_batch": 8}}])
    w = warned(analyze(cmp))
    assert len(w) == 1 and "here cols)" in w[0]
    assert not warned(analyze(dict(cmp, profiles={"costs": "ttsim-perf"})))


# ------------------------------------------------------------------------------- silicon-hypothesis DRAM bound

@pytest.fixture(scope="module")
def no_bound():
    """silicon-hypothesis without the layout DRAM bound (dram.bound_layout removed), in memory."""
    p = copy.deepcopy(profiles.load("costs", "silicon-hypothesis")[0])
    p["name"] = "silicon-hypothesis-no-layout-bound"
    del p["params"]["dram.bound_layout"]
    profiles.register("costs", p)
    yield p["name"]
    profiles.unregister("costs", p["name"])


def compare(shape, costs):
    return analyze(layout_req(shape=shape, costs=costs, action="compare", candidates=[{"id": "g", "overrides": G2D}]))


def test_dram_bandwidth_caps_large_layout_ops_on_silicon_hypothesis(no_bound):
    sim, hyp, off = compare((8192, 2048), "ttsim-perf"), compare((8192, 2048), "silicon-hypothesis"), \
        compare((8192, 2048), no_bound)
    for r in (sim, hyp, off):
        assert r["status"] == "ok" and validate_response(r) == []
    sp = lambda r: r["candidates"][0]["comparison"]["speedup_vs_baseline"]["nominal"]  # noqa: E731
    assert sp(sim) > 2 and sp(hyp) < sp(sim) and sp(hyp) < sp(off)
    for row in (hyp["baseline"], hyp["candidates"][0]):
        assert row["bottleneck"]["resource"] == "dram_global"
        assert row["lower_bound_ns"] == row["time_ns"]["nominal"] == row["bounds_ns"]["dram_global"]
        assert "assumed DRAM bandwidth" in row["accuracy"]
        assert any(w.startswith("bound by the assumed DRAM bandwidth") for w in row["warnings"])
    # 32 MiB read + 32 MiB written at 648.1 B/cycle x 0.75, 1.35 GHz: >= 102 us
    assert hyp["baseline"]["time_ns"]["nominal"] >= 64 * 2**20 / (648.1 * 0.75) / 1.35
    assert hyp["baseline"]["time_ns"]["nominal"] > off["baseline"]["time_ns"]["nominal"]
    # a small shape is not DRAM-bound: identical to the profile without the bound
    small, small_off = compare((256, 512), "silicon-hypothesis"), compare((256, 512), no_bound)
    for a, b in ((small["baseline"], small_off["baseline"]), (small["candidates"][0], small_off["candidates"][0])):
        assert a["time_ns"]["nominal"] == b["time_ns"]["nominal"] and a["bottleneck"]["resource"] != "dram_global"
        assert a["lower_bound_ns"] < a["time_ns"]["nominal"]
    assert "assumed DRAM bandwidth" in small["accuracy"]


def test_simulator_profiles_have_no_layout_dram_bound():
    for name in ("craq-sim", "ttsim-perf"):
        p = profiles.load("costs", name)[0]["params"]
        assert "dram.bound_layout" not in p and "dram.total_bytes_per_cycle" not in p
        b = analyze(layout_req(shape=(8192, 2048), costs=name))["baseline"]
        assert "lower_bound_ns" not in b and b["bottleneck"]["resource"] != "dram_global"


def test_silicon_hypothesis_carries_ttsim_layout_terms():
    from make_silicon_hypothesis import layout_prefixes

    t = profiles.load("costs", "ttsim-perf")[0]
    s = profiles.load("costs", "silicon-hypothesis")[0]
    fam = tuple(f"{p}." for p in layout_prefixes())
    terms = {k: v for k, v in t["params"].items() if k.startswith(fam)}
    assert len(terms) > 500 and all(s["params"].get(k) == v for k, v in terms.items())
    assert s["clock"]["GHz"] == 1.35 and s["params"]["dram.total_bytes_per_cycle"]["value"] == 648.1
    assert s["params"]["dram.bound_layout"]["value"] == 1
    assert s["calibration"]["derived_from"] == f"ttsim-perf@{t['revision']}"


@pytest.mark.parametrize("op", ["interleaved_to_sharded", "sharded_to_interleaved"])
def test_dram_bound_covers_the_interleaved_side_of_sharding_ops(op):
    """silicon-hypothesis bounds the sharding ops by the DRAM bytes of their interleaved side (whole tiles for TILE,
    the logical rows for ROW_MAJOR); an L1 interleaved side and the simulator profiles have no DRAM bound."""
    from qperf.layout_api import _costs
    from qperf.ops.layout import estimate

    def est(costs, place="dram"):
        q = layout_req(op, (8192, 2048), costs, place)
        q["workload"]["shard"] = {"strategy": "height", "cores": 8}
        return analyze(q)["baseline"]

    b = est("silicon-hypothesis")
    assert b["bottleneck"]["resource"] == "dram_global" and b["bounds_ns"]["dram_global"] == b["time_ns"]["nominal"]
    assert any("32.0 MiB" in w for w in b["warnings"])  # 8192 x 2048 bf16 moved once to / from DRAM
    for costs, place in (("silicon-hypothesis", "l1"), ("ttsim-perf", "dram"), ("craq-sim", "dram")):
        assert "bounds_ns" not in est(costs, place)
    c = _costs(profiles.load("costs", "silicon-hypothesis")[0], "nominal", None)
    tile = estimate(op, 1000, 256, c, "bf16", "hs8")  # TILE: 32 tile rows = 1024 padded rows
    rm = estimate(op, 1000, 256, c, "bf16", "hs8r")
    assert tile["dram_bytes"] == 1024 * 256 * 2 and rm["dram_bytes"] == 1000 * 256 * 2
