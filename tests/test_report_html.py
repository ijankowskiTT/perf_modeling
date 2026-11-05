"""HTML / markdown reports: 'How it runs' (core grid + per-core pipeline), the approaches table and charts that compare
every config. Fixture responses carry the layout `structure` contract; real analyze() responses check that every
action renders with or without it."""

import copy
import json
import os
import re

import pytest

from qperf import analyze
from qperf.report import to_html, to_markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(os.path.join(ROOT, "skills", "quasar-perf", "examples", f)
                  for f in os.listdir(os.path.join(ROOT, "skills", "quasar-perf", "examples")))


def _roles(r, c, w):
    return {k: {"threads": 1, "useful_busy_frac": v, "starved_frac": 0.1, "backpressure_frac": 0.05,
                "transfer_wait_frac": 0.0, "start_ns": 0.0} for k, v in (("reader", r), ("compute", c), ("writer", w))}


def _row(label, t, neck, roles, structure, config=None, **kw):
    r = {"label": label, "config": dict(config or {"factory": "Default", "use_multicore": True, "input": "dram"}),
         "valid": True, "errors": [], "warnings": [], "time_ns": {"nominal": t, "low": t * 0.95, "high": t * 1.05},
         "bottleneck": {"resource": neck, "busiest_role": neck}, "roles": roles, "structure": structure}
    r.update(kw)
    return r


STRUCT = {"factory": "Default", "grid": [8, 4], "cores_used": 32, "split": "rows", "row_cores": 32, "col_chunks": 1,
          "source": "dram", "dest": "dram", "per_core": {"blocks": 4, "tiles_per_block": 64, "tiles": 256},
          "reader": {"requests": 128, "bytes_per_request": 4096, "barriers": 4},
          "compute": {"steps": 4, "tiles_per_step": 64, "kind": "tilize_block"},
          "writer": {"requests": 256, "bytes_per_request": 2048, "barriers": 4},
          "dfb": {"in_entries": 2, "out_entries": 2, "entry_bytes": 131072}}


def layout_fixture():
    s16 = dict(STRUCT, cores_used=16, split="2d", row_cores=8, col_chunks=2,
               per_core={"blocks": 8, "tiles_per_block": 32, "tiles": 256},
               reader={"requests": 256, "bytes_per_request": 2048, "barriers": 8},
               writer={"requests": 64, "bytes_per_request": 8192, "barriers": 2})
    base = _row("tilize Default (multicore, input dram)", 50000.0, "reader", _roles(0.9, 0.3, 0.5), STRUCT)
    c1 = _row("tilize Default (16 cores, 2d)", 40000.0, "writer", _roles(0.6, 0.35, 0.8), s16, id="cores16_2d",
              comparison={"speedup_vs_baseline": {"nominal": 1.25, "low": 1.24, "high": 1.26}, "verdict": "faster"})
    c2 = _row("tilize Default (write_batch 4)", 48000.0, "reader", _roles(0.88, 0.3, 0.4), STRUCT,
              config={"factory": "Default", "use_multicore": True, "input": "dram", "write_batch": 4, "hypothetical": True},
              id="wb4", warnings=["hypothetical kernel change: batched writes"],
              comparison={"speedup_vs_baseline": {"nominal": 1.04, "low": 1.03, "high": 1.05}, "verdict": "faster"})
    req = {"schema_version": "1", "action": "compare", "profiles": {"costs": "craq-sim"},
           "workload": {"op": "tilize", "dtype": "bf16", "shape": [1, 1, 8192, 2048], "placements": {"a": "dram"}},
           "baseline": {}, "candidates": [{"id": "cores16_2d", "overrides": {"cores": 16, "split": "2d"}},
                                          {"id": "wb4", "overrides": {"write_batch": 4}}]}
    res = {"status": "ok", "profiles": {"costs": {"name": "craq-sim", "revision": "test"}}, "platform_scope": "fixture",
           "baseline": base, "candidates": [c1, c2]}
    return req, res


def _cards(page):
    return dict(re.findall(r'<div class="card how" data-config="([^"]+)">(.*?)(?=<div class="card how"|<h2>)', page, re.S))


def _wide(card):
    return card.split('<div class="pipe-w">', 1)[1].split('<div class="pipe-n">', 1)[0]


def test_how_it_runs_grid_and_pipeline_from_structure():
    req, res = layout_fixture()
    page = to_html(req, res)
    cards = _cards(page)
    assert list(cards) == ["baseline", "cores16_2d", "wb4"]
    assert cards["baseline"].count('class="on"') == 32 and cards["baseline"].count('class="off"') == 0
    assert cards["cores16_2d"].count('class="on"') == 16 and cards["cores16_2d"].count('class="off"') == 16
    assert "32 of 32 cores</b>, rows split" in cards["baseline"]
    assert "16 of 32 cores</b>, 2d split (8 row cores x 2 column chunks)" in cards["cores16_2d"]
    wide = _wide(cards["baseline"])
    for label in ("source", "DRAM", "reader x1", "128 x 4,096 B", "4 barriers", "input DFB", "2 x 131,072 B",
                  "compute x1", "4 x 64 tiles", "tilize_block", "output DFB", "writer x1", "256 x 2,048 B",
                  "destination", "90% busy", "bottleneck"):
        assert f">{label}</text>" in wide, label
    # the bottleneck role is outlined in the config's color: reader for the baseline, writer for cores16_2d
    hot = r'<g class="node role"><title>{}[^<]*bottleneck</title><rect class="bx"[^>]*style="stroke:var\(--{}\);'
    assert re.search(hot.format("reader", "s1"), wide)
    assert re.search(hot.format("writer", "s2"), _wide(cards["cores16_2d"]))
    assert "hypothetical" in cards["wb4"] and "hypothetical" not in cards["cores16_2d"]
    assert page.count('<svg class="pipe"') == 6  # wide and narrow layout per config


def test_approaches_table_lists_changes_and_tags():
    req, res = layout_fixture()
    page = to_html(req, res)
    rows = re.findall(r'<tr class="appr">(.*?)</tr>', page, re.S)
    assert len(rows) == 3
    cells = [re.findall(r"<td>(.*?)</td>", r, re.S) for r in rows]
    assert "(baseline)" in rows[0]
    assert "cores 16" in cells[1][1] and "split 2d" in cells[1][1] and "1.25x" in rows[1]
    assert "write_batch 4" in cells[2][1] and "hypothetical" in rows[2] and "hypothetical" not in rows[1]
    md = to_markdown(req, res)
    assert "| cores16_2d | cores 16, split 2d | 40,000 ns | 1.25x | faster | writer |  |" in md
    assert "| wb4 | write_batch 4 | 48,000 ns | 1.04x | faster | reader | hypothetical |" in md


def test_markdown_how_it_runs_line():
    req, res = layout_fixture()
    md = to_markdown(req, res)
    block = md.split("## How it runs", 1)[1].split("\n## ", 1)[0]
    line = next(x for x in block.splitlines() if x.startswith("- **baseline**"))
    for part in ("32 of 32 cores, rows split", "per core 4 blocks x 64 tiles = 256 tiles", "source DRAM",
                 "reader x1 128 x 4,096 B, 4 barriers (busy 90%, bottleneck)", "input DFB 2 x 131,072 B",
                 "compute x1 4 x 64 tiles, tilize_block", "writer x1 256 x 2,048 B, 4 barriers", "destination DRAM"):
        assert part in line, part
    assert "8 row cores x 2 column chunks" in block


def test_charts_compare_every_config():
    req, res = layout_fixture()
    page = to_html(req, res)
    bars = page.split("Predicted time per configuration", 1)[1].split("</svg>", 1)[0]
    assert bars.count('<g class="bar">') == 3 and "cores16_2d (best)" in bars
    assert all(f'fill="var(--s{k})"' in bars for k in (1, 2, 3))
    roles = page.split("Where the time goes", 1)[1]
    assert "baseline vs cores16_2d" in page
    legend = roles.split('<div class="legend">', 1)[1].split("</div>", 1)[0]
    assert "baseline" in legend and "cores16_2d" in legend
    # grouped bars: each role x measure has one track per config, and a bar in the config's color when non-zero
    chart = roles.split("</svg>", 1)[0]
    assert chart.count('fill="var(--track)"') == 3 * 4 * 2
    assert chart.count('fill="var(--s2)"><title>cores16_2d') == 3 * 3  # transfer_wait is 0 in the fixture


def test_without_structure_falls_back_to_the_row():
    req, res = layout_fixture()
    for r in [res["baseline"]] + res["candidates"]:
        del r["structure"]
    res["baseline"]["work_split"] = [{"blocks_per_core": 4, "tiles_per_block": 2, "cores": 24},
                                     {"blocks_per_core": 2, "tiles_per_block": 2, "cores": 1}]
    page = to_html(req, res)
    cards = _cards(page)
    assert cards["baseline"].count('class="on"') == 25 and "25 of 32 cores" in cards["baseline"]
    # the half-loaded tail core is lighter; consecutive cores with one load share a tooltip
    assert "--o:0.70" in cards["baseline"] and 'title="core 24: 2 blocks x 2 tiles"' in cards["baseline"]
    assert 'title="cores 0-23: 4 blocks x 2 tiles"' in cards["baseline"]
    assert "of 32 cores" not in cards["cores16_2d"]  # no structure and no work_split: no grid, no invented count
    wide = _wide(cards["cores16_2d"])
    assert ">reader x1</text>" in wide and "barriers" not in wide and ">destination</text>" in wide
    assert "## How it runs" in to_markdown(req, res)


def test_sensitivity_has_a_line_and_legend_entry_per_series():
    req, res = layout_fixture()
    req = dict(req, action="sensitivity", sensitivity={"param": "num_cols", "values": [64, 256, 1024]})
    res = copy.deepcopy(res)
    res["sensitivity"] = {"param": "num_cols", "ranking_changes": [], "series": {
        sid: [{"value": v, "valid": True, "time_ns": t * f, "bottleneck": "reader"} for v, t in ((64, 1e3), (256, 3e3), (1024, 9e3))]
        for sid, f in (("baseline", 1.0), ("cores16_2d", 0.8), ("wb4", 0.95))}}
    page = to_html(req, res)
    sens = page.split("<h2>Sensitivity", 1)[1].split("<h2>", 1)[0]
    time_chart = sens.split("Speedup vs baseline", 1)[0]
    legend = time_chart.split('<div class="legend">', 1)[1].split("</div>", 1)[0]
    assert re.findall(r"</i>([^<]+)</span>", legend) == ["baseline", "cores16_2d", "wb4"]
    strokes = re.findall(r'<path d="[^"]+" fill="none" stroke="([^"]+)"', time_chart)
    assert strokes == ["var(--s1)", "var(--s2)", "var(--s3)"]  # same color as the config everywhere else
    assert "Not swept" not in sens


def test_layout_sensitivity_notes_unswept_candidates():
    req = json.load(open(os.path.join(ROOT, "skills", "quasar-perf", "examples", "untilize_width.json")))
    req["candidates"] = [{"id": "single_core", "overrides": {"use_multicore": False}}]
    res = analyze(dict(req, analysis={"detail": "full"}))
    page = to_html(req, res)
    if "single_core" not in res["sensitivity"]["series"]:
        assert "Not swept: single_core" in page
    assert '<div class="legend"><span>' in page  # a legend even for one series


LAYOUT_REQS = [
    {"action": "estimate", "workload": {"op": "tilize", "dtype": "bf16", "shape": [1, 1, 3136, 64], "placements": {"a": "dram"}}},
    {"action": "compare", "workload": {"op": "untilize", "dtype": "bf16", "shape": [1, 1, 2048, 512], "placements": {"a": "dram"}},
     "candidates": [{"id": "sc", "overrides": {"use_multicore": False}}]},
    {"action": "sensitivity", "workload": {"op": "tilize", "dtype": "bf16", "shape": [1, 1, 3136, 64], "placements": {"a": "dram"}},
     "sensitivity": {"param": "num_rows", "values": [1024, 3136, 8192]}},
    {"action": "estimate", "workload": {"op": "move", "dtype": "bf16", "shape": [1, 1, 1024, 1024], "placements": {"a": "dram"}}},
    {"action": "compare", "workload": {"op": "reshard", "dtype": "bf16", "shape": [1, 1, 1024, 1024], "reshard": {
        "layout": "tile", "from": {"strategy": "height", "cores": 8, "memory": "l1"},
        "to": {"strategy": "height", "cores": 16, "memory": "l1"}}},
     "candidates": [{"id": "to4", "overrides": {"reshard": {"to": {"strategy": "height", "cores": 4, "memory": "l1"}}}}]},
]


def _requests():
    for path in EXAMPLES:
        yield os.path.basename(path), json.load(open(path))
    for i, r in enumerate(LAYOUT_REQS):
        yield f"layout{i}_{r['workload']['op']}_{r['action']}", dict(r, schema_version="1", profiles={"costs": "craq-sim"}, baseline={})
    yield "binary_estimate", {"schema_version": "1", "action": "estimate", "profiles": {"costs": "craq-sim"}, "baseline": {},
                              "workload": {"op": "add", "dtype": "bf16", "num_tiles": 4096, "clusters": 32,
                                           "placements": {"a": "dram", "b": "dram", "out": "dram"}}}


@pytest.mark.parametrize("detail", ["compact", "full"])
@pytest.mark.parametrize("name,req", list(_requests()), ids=[n for n, _ in _requests()])
def test_every_action_renders(name, req, detail):
    res = analyze(dict(req, analysis=dict(req.get("analysis", {}), detail=detail)))
    assert res["status"] in ("ok", "partial"), res.get("message")
    page, md = to_html(req, res), to_markdown(req, res)
    assert "<h2>How it runs</h2>" in page and "## How it runs" in md and "<script" not in page
    cards = _cards(page)
    assert "baseline" in cards and len(cards) <= 4
    if detail == "full":  # work_split is there: the grid shades exactly the cores (clusters) that carry work
        rows = {"baseline": res["baseline"], **{c.get("id"): c for c in res.get("candidates", [])}}
        for rid, card in cards.items():
            if rid in rows and rows[rid].get("work_split"):
                ws = rows[rid]["work_split"]
                n = sum(e.get("cores", e.get("clusters", 0)) for e in ws)
                assert card.count('class="on"') == min(n, 32), rid
    n_rows = 1 + len(res.get("candidates", [])) + len(res.get("top", []))
    assert page.count('<tr class="appr">') == (n_rows if n_rows > 1 else 0)
    if res.get("top"):
        assert all(f">top{i + 1}" in page for i in range(len(res["top"])))
    if res.get("sensitivity"):
        legend = page.split("<h2>Sensitivity", 1)[1].split('<div class="legend">', 1)[1].split("</div>", 1)[0]
        assert len(re.findall(r"<span>", legend)) == min(len(res["sensitivity"]["series"]), 6)
