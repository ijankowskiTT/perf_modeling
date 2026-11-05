"""Bundle reports (qperf analyze with several --request, or a bundle file): one summary line per op, two ranking tables
that never mix kernel changes with layout choices, one stated grid, the caps, and every request's own section; the
single-request outputs stay what they were."""

import contextlib
import hashlib
import io
import json
import os
import re

import pytest

from qperf import bundle
from qperf.cli import main
from qperf.survey import DEFAULT_HEIGHTS, DEFAULT_WIDTHS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_OPS, CHOICE_OPS = {"tilize", "untilize"}, {"interleaved_to_sharded", "reshard"}


def survey(op, costs="ttsim-perf", **kw):
    r = {"schema_version": "1", "action": "survey", "profiles": {"costs": costs},
         "workload": {"op": op, "dtype": "bf16", "placements": {"a": "dram"}}}
    r.update(kw)
    return r


REQS = [survey("tilize", survey={"compare_profile": "craq-sim"}), survey("untilize"), survey("interleaved_to_sharded"),
        {"schema_version": "1", "action": "survey", "profiles": {"costs": "ttsim-perf"},
         "workload": {"op": "reshard", "dtype": "bf16"}},
        survey("tilize", "silicon-hypothesis")]
TITLE = "Quasar layout ops"
_b = {}


def B():
    if "b" not in _b:
        _b["b"] = bundle.run([json.loads(json.dumps(q)) for q in REQS], TITLE, detail="full")
    return _b["b"]


def cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["analyze", *args])
    return code, buf.getvalue()


def files(tmp_path, reqs=REQS):
    out = []
    for i, q in enumerate(reqs):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(q))
        out.append(str(p))
    return out


MS = re.compile(r"\d+(\.\d+)? ms")


def masked(s):
    return MS.sub("<T> ms", s)


# ------------------------------------------------------------------------------------------------ rankings

def test_two_rankings_never_mix_kernel_changes_and_layout_choices():
    rk = B()["rankings"]
    kc, lc = rk["kernel_changes"], rk["layout_choices"]
    assert kc and lc
    assert {r["op"] for r in kc} == KERNEL_OPS and all(r["kind"] == "kernel_change" and r["hypothetical"] for r in kc)
    assert {r["op"] for r in lc} == CHOICE_OPS and all(r["kind"] == "layout_choice" and not r["hypothetical"] for r in lc)
    # survey approaches equal to the factory's plan everywhere (untilize dfb2) are not kernel changes
    assert "dfb2" in rk["omitted"]["untilize"] and not any(r["op"] == "untilize" and r["id"] == "dfb2" for r in kc)


def test_rows_share_one_stated_grid_and_rank_by_ttsim_perf_geomean():
    b = B()
    g = b["grid"]
    assert g["heights"] == DEFAULT_HEIGHTS and g["widths"] == DEFAULT_WIDTHS and g["shapes"] == 54 and g["default"]
    for key in ("kernel_changes", "layout_choices"):
        rows = b["rankings"][key]
        assert all(r["on_grid"] and r["shapes"] == 54 and r["profile"] == "ttsim-perf" for r in rows)
        timed = [r for r in rows if not r["untimed_cells"]]
        assert [r["geomean"] for r in timed] == sorted((r["geomean"] for r in timed), reverse=True)
        assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    # each row's geomean is its survey's ttsim-perf geomean over its evaluated shapes
    tz = next(x for x in b["results"] if x["request"]["workload"]["op"] == "tilize")["result"]["survey"]
    row = next(r for r in b["rankings"]["kernel_changes"] if r["op"] == "tilize" and r["id"] == "cols_all")
    a = tz["approaches"]["cols_all"]
    assert row["geomean"] == a["speedup"]["geomean"] and row["evaluated"] == 54 and row["regressions"] == a["counts"]["slower"]
    assert row["best"] == a["best"] and row["worst"] == a["worst"]


def test_silicon_hypothesis_fills_the_dram_column_and_the_caps():
    b = B()
    tz = [r for r in b["rankings"]["kernel_changes"] if r["op"] == "tilize"]
    assert all(r["dram"]["profile"] == "silicon-hypothesis" and r["dram"]["capped"] > 0 for r in tz)
    assert not any(r["profile"] == "silicon-hypothesis" for r in tz)  # attached to the ttsim-perf rows, not ranked
    sil = next(x for x in b["results"] if x["request"]["profiles"]["costs"] == "silicon-hypothesis")["result"]["survey"]
    top = tz[0]
    assert top["dram"]["geomean"] == sil["approaches"][top["id"]]["speedup"]["geomean"]
    assert top["dram"]["capped"] == sil["approaches"][top["id"]]["dram_capped"]
    caps = b["caps"]
    assert caps["dram_profile_ran"] and caps["items"] and top["id"] in caps["items"][0]
    assert "silicon-hypothesis" in caps["items"][0] and "untilize" in caps["note"]  # untilize ran on ttsim-perf only
    assert "reshard" not in caps["note"]  # L1 -> L1: no DRAM side to cap
    assert all(bundle._dram_txt(r) == "no DRAM side" for r in b["rankings"]["layout_choices"] if r["op"] == "reshard")


def test_untimed_layout_choices_are_marked_and_never_best():
    b = B()
    i2s = [r for r in b["rankings"]["layout_choices"] if r["op"] == "interleaved_to_sharded"]
    assert i2s and all(r["untimed_cells"] == r["evaluated"] for r in i2s)
    assert all("(untimed)" in bundle._geo_cell(r) for r in i2s)
    lc = b["rankings"]["layout_choices"]
    assert lc[0]["op"] == "reshard"  # timed rows rank before untimed ones
    line = next(s["text"] for s in b["summary"] if s["op"] == "interleaved_to_sharded")
    assert "none timed" in line


def test_summary_one_line_per_op_with_factory_limiter_best_and_worst():
    b = B()
    assert [s["op"] for s in b["summary"]] == ["tilize", "untilize", "interleaved_to_sharded", "reshard"]
    tz = b["summary"][0]["text"]
    assert "Default factory" in tz and "limited by" in tz and "best change" in tz and "worst regression" in tz
    assert "silicon-hypothesis" in tz  # the DRAM cap of the baseline
    rs = b["summary"][3]["text"]
    assert "SameWidth factory" in rs and "best choice" in rs and "worst regression h8_to_h4" in rs


def test_no_dram_profile_says_dram_is_untimed():
    b = bundle.run([REQS[1], REQS[3]], None)
    assert not b["caps"]["dram_profile_ran"] and "DRAM is untimed" in b["caps"]["note"]
    assert b["title"] == "qperf: untilize, reshard"


# ------------------------------------------------------------------------------------------------ formats

def _table(text, name):
    """The rows of one ranking table in the text output."""
    part = text.split(name, 1)[1]
    part = part.split("\n\n", 1)[0]
    return [ln for ln in part.splitlines()[2:] if ln.startswith("  ") and ln.strip()[0].isdigit()]


def test_text_md_html_carry_both_tables_the_grid_and_every_section():
    b = B()
    text, md, page = bundle.to_text(b), bundle.to_markdown(b), bundle.to_html(b)
    grid = b["grid"]["text"]
    for out in (text, md, page.replace("&#x27;", "'")):
        assert "Kernel changes (hypothetical)" in out and "Layout choices (change what the next op receives)" in out
        assert grid in out and "Summary by op" in out and "Caps" in out
    for i, x in enumerate(b["results"], 1):
        title = bundle._section_title(x, i)
        assert f"== {title} ==" in text and f"## {title}" in md and title in page
    kt = _table(text, "Kernel changes (hypothetical):")
    lt = _table(text, "Layout choices (change what the next op receives):")
    assert len(kt) == len(b["rankings"]["kernel_changes"]) and len(lt) == len(b["rankings"]["layout_choices"])
    assert all(ln.split()[1] in KERNEL_OPS for ln in kt) and all(ln.split()[1] in CHOICE_OPS for ln in lt)
    mk = md.split("## Kernel changes (hypothetical)")[1].split("## Layout choices")[0]
    ml = md.split("## Layout choices (change what the next op receives)")[1].split("## Caps")[0]
    assert not any(op in mk for op in CHOICE_OPS) and not any(f"| {op} |" in ml for op in KERNEL_OPS)


def test_html_page_tables_sections_components_and_themes():
    b = B()
    page = bundle.to_html(b)
    assert page.startswith("<!doctype html>") and "<script" not in page
    for key, ops in (("kernel_changes", KERNEL_OPS), ("layout_choices", CHOICE_OPS)):
        table = page.split(f'data-table="{key}"')[1].split("</table>")[0]
        assert set(re.findall(r'<tr[^>]* data-op="([^"]+)"', table)) == ops
    assert page.count('<details class="req"') == len(REQS)
    # each section reuses the single-request components: accuracy, how it runs, heatmaps, the best map
    assert page.count('class="note accuracy"') >= len(REQS) and page.count("How it runs at") == len(REQS)
    assert page.count('class="bestmap hmap"') == len(REQS) and page.count('<div class="heatmap hmap"') >= 30
    assert page.count("<i") > 1000 and '<i class="u"' in page  # untimed: i2s cells on ttsim-perf
    ids = re.findall(r' id="([^"]+)"', page)
    assert len(ids) == len(set(ids)) and page.count("<defs>") == 1  # shared SVG defs once per page
    assert "prefers-color-scheme:dark" in page and ':root[data-theme="dark"]' in page and "--o1" in page
    assert "max-width:960px" in page and "overflow-x:auto" in page  # phone width: tables scroll inside cards
    # one color per op, the same in the summary, both tables and the section border
    assert page.count('style="background:var(--o1)"') > 10 and "border-left:3px solid var(--o4)" in page


def test_html_page_weight_and_mixed_units_note():
    b = B()
    page = bundle.to_html(b)
    # 5 requests, 36 heatmaps: ~1 MB with per-cell SVG markup; colour classes, CSS-grid cells and shared defs
    assert len(page.encode()) < 350_000
    assert page.count(' title="') > 2000 and "<title>" not in page.split("Speedup vs baseline per shape", 1)[1].split(
        "Best approach per shape", 1)[0]  # one tooltip per heatmap cell, as an attribute
    assert not re.search(r'<i [^>]*style="fill:color-mix', page) and ".hmap .w" in page  # bins as classes
    # ttsim-perf (cycles) and silicon-hypothesis (ns) in one bundle: one line says so in every format
    note = bundle.units_note(b)
    assert note.startswith("Times: ttsim-perf reports simulator cycles") and "silicon-hypothesis real ns" in note
    assert note in bundle.to_text(b) and note in bundle.to_markdown(b) and note.replace("'", "&#x27;") in page
    assert bundle.units_note(bundle.run([REQS[1], REQS[3]], None)) is None  # all on ttsim-perf


def test_cli_bundle_forms_agree_and_json_carries_everything(tmp_path):
    paths = files(tmp_path)
    code, a = cli(*sum((["--request", p] for p in paths), []), "--title", TITLE, "--format", "text")
    assert code == 0
    bf = tmp_path / "bundle.json"
    bf.write_text(json.dumps({"title": TITLE, "requests": REQS}))
    code, b = cli("--request", str(bf), "--format", "text")
    assert code == 0 and masked(a) == masked(b) and a.startswith(f"# {TITLE}")
    code, j = cli("--request", str(bf), "--format", "json")
    doc = json.loads(j)
    assert set(doc) >= {"title", "grid", "summary", "rankings", "caps", "results"} and len(doc["results"]) == len(REQS)
    out = tmp_path / "b.html"
    code, msg = cli("--request", str(bf), "--format", "html", "--out", str(out))
    assert code == 0 and msg.strip() == str(out) and out.read_text().count('<details class="req"') == len(REQS)
    code, md = cli("--request", paths[2], "--request", paths[3], "--format", "md")
    assert code == 0 and md.startswith("# qperf: interleaved_to_sharded, reshard")


def test_bundle_with_a_failing_request_still_renders(tmp_path):
    bad = {"schema_version": "1", "action": "survey", "profiles": {"costs": "ttsim-perf"},
           "workload": {"op": "move", "dtype": "bf16"}}
    paths = files(tmp_path, [REQS[3], bad])
    code, text = cli(*sum((["--request", p] for p in paths), []), "--format", "text")
    assert code == 2 and "move: unsupported" in text and "Layout choices" in text


def test_compare_and_tune_rows_join_the_right_table():
    tune = {"schema_version": "1", "action": "tune", "profiles": {"costs": "ttsim-perf"},
            "workload": {"op": "tilize", "dtype": "bf16", "shape": [1, 1, 512, 2048], "placements": {"a": "l1"}}}
    shard = json.load(open(os.path.join(ROOT, "skills", "quasar-perf", "examples", "shard_cores.json")))
    b = bundle.run([tune, shard], None)
    kc, lc = b["rankings"]["kernel_changes"], b["rankings"]["layout_choices"]
    assert kc and all(r["op"] == "tilize" and r["hypothetical"] and r["grid"] == "1 shape (512x2048)" for r in kc)
    assert {r["id"] for r in lc} == {"32_cores", "width_sharded"} and all(not r["on_grid"] for r in lc)
    assert all(bundle._dram_txt(r) == "no DRAM side" for r in kc)  # tilize from L1


# ------------------------------------------------------------------------------------------------ single request

# sha256 of the single-request CLI outputs (timings masked); regenerated deliberately for the survey / report polish
# (cores per shape, cycles vs ns labels, scenario-dependent verdicts, lighter HTML). A deliberate change of the
# single-request reports or a refit regenerates them: python tests/test_bundle.py
SINGLE = {
    "reshard_compare_ttsim.html": "f000a9a434b0bbbca3b9c02ba50da837172d2df3406e7ef0a02128a8d06fcf6a",
    "reshard_compare_ttsim.md": "2a3636a8b1d51a778ca1a0f00c9637652495c6c4eb5d630284960eeef85f14fd",
    "reshard_compare_ttsim.text": "910168e329d8b31009581997e3509c77317ede2753944d2f64b11b23018b1f45",
    "shard_cores.html": "867140c2f2d6bf00804aed271ad6f75ff744b3f7d5abb29bcabfc23cef214f52",
    "shard_cores.md": "5640ab41f4dc9b8f1c6ae0eaa2b873bdc3221d0c16085454b74a32b529a7999f",
    "shard_cores.text": "3b6d7652fb37f5fbca0b4ebae02356ee11df4cb07767136a82931b164f0647cc",
    "tilize_compare.html": "c195639278e6abfda321c143a99a1ef541f6bbdd5dafcaf5cf2e5997857e01b1",
    "tilize_compare.md": "aad1ab4f0692377dca7e5ae079bc69b77b158167da4064a7807772dc74e97eb4",
    "tilize_compare.text": "2cda32b4d2f8fee00fa135fbffe160859b67c1534703a22601bf2130d6cb9f76",
    "tilize_survey_ttsim.html": "b3f231565323436ee48dea3cf15f1f7fd56f3231c16e6a9f30adfe012ddca0ed",
    "tilize_survey_ttsim.md": "d87e55ad98657993d16598288180f5732456fcbccaeae1693f34b926f80aaf7b",
    "tilize_survey_ttsim.text": "7954b331568a323d98f8382795564e97efc4e14dc74fc0c2a402bc08b30b984b",
}


def single_requests():
    ex = os.path.join(ROOT, "skills", "quasar-perf", "examples")
    return {
        "tilize_survey_ttsim": survey("tilize"),
        "tilize_compare": json.load(open(os.path.join(ex, "tilize_compare.json"))),
        "shard_cores": json.load(open(os.path.join(ex, "shard_cores.json"))),
        "reshard_compare_ttsim": {
            "schema_version": "1", "action": "compare", "profiles": {"costs": "ttsim-perf"},
            "workload": {"op": "reshard", "dtype": "bf16", "shape": [1, 1, 1024, 1024],
                         "reshard": {"from": {"strategy": "height", "cores": 8, "memory": "l1"},
                                     "to": {"strategy": "height", "cores": 4, "memory": "l1"}}},
            "candidates": [{"id": "to16", "overrides": {"reshard": {"to": {"strategy": "height", "cores": 16,
                                                                             "memory": "l1"}}}}]},
    }


def single_hashes(tmp):
    out = {}
    for k, q in single_requests().items():
        p = os.path.join(tmp, k + ".json")
        with open(p, "w") as f:
            json.dump(q, f)
        for fmt in ("text", "md", "html"):
            out[f"{k}.{fmt}"] = hashlib.sha256(masked(cli("--request", p, "--format", fmt)[1]).encode()).hexdigest()
    return out


def test_single_request_outputs_are_unchanged(tmp_path):
    assert single_hashes(str(tmp_path)) == SINGLE


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        print(json.dumps(single_hashes(d), indent=4, sort_keys=True))
