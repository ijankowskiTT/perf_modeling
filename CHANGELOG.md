# Changelog

## 1.4.0

- Bundles: several requests in one call render one page (summary per op, kernel changes and layout choices ranked
  separately, caps from assumed DRAM, a section per request).
- `survey` for interleaved_to_sharded, sharded_to_interleaved and reshard (layout choices); untimed cells marked.

## 1.3.0

- Every answer states its op's own accuracy (`accuracy`); no profile note carries another op's error.
- Shapes outside the calibrated range are flagged as extrapolated (marked in survey heatmaps).
- `survey.compare_profile`: both profiles side by side; craq-sim warns when setups shrink requests.
- silicon-hypothesis has data-movement costs and an assumed DRAM bandwidth cap: large-shape gains are bounded there.

## 1.2.0

- **`survey`** (tilize, untilize): each approach over a grid of tensor sizes, no size needed. Geomean, best and worst
  shape, regressions, best approach per shape; reports draw a heatmap per approach.

## 1.1.0

- **Design search** for tilize / untilize multicore: cores, rows / cols / 2d split, block size, buffering, batching
  (`tune`, hypothetical). Sensitivity sweeps plot every option.
- **Reports** show how each setup runs (core grid, per-core reader -> compute -> writer flow, bottleneck) and compare
  every option in the charts.

## 1.0.0

- **binary_ng** `add`, `multiply` (bf16, fp32). Covers thread allocation, DM batch, tiles_per_cycle, ring depth,
  compute / DRAM / L1 what-ifs and hypothetical kernel changes (per-thread barriers, dynamic work queue).
- **Data movement** (bf16, fp32):
  - tilize, untilize and their padding variants;
  - interleaved_to_sharded and sharded_to_interleaved (TILE, ROW_MAJOR);
  - move (overlap path);
  - reshard (SameWidth, SameHeight and their write variants, Generic, CopyPages).
- **Accuracy**: <= 2.5% mean per factory on craq-sim and <= 3.3% on ttsim-perf. Exceptions, flagged in answers:
  ttsim-perf reshard SameHeight / Generic, and binary_ng L1 (~8%).
- **Plugin**: the `quasar-perf` skill hands each question to a Haiku runner (~$0.14 per question). Follow-ups keep
  context. Reports with charts are optional.
- **`check-drift --ref main | pr/N`**: compares the pinned kernels with GitHub. Comment, include and formatting
  changes count as cosmetic.
