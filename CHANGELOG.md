# Changelog

## 1.7.0

- binary_ng `subtract`, `divide` and bf16 FPU `multiply` (652 simulator runs). Every other binary op and broadcast is
  answered `unsupported` with the reason (other factory, or doesn't build on Quasar).
- Simulator-validated tilize knobs: `dfb_blocks` 2, `split` cols and their combination with `write_batch`, as real
  kernel variants (`calibration/variants/tilize_dfb_cols.patch`), within 1% after refit. Tilize compute now runs as
  two stages through DEST; factory accuracy improved (craq-sim 1.21% -> 0.75%).

## 1.6.0

- Quasar DRAM channel model on silicon-hypothesis (`qperf/dram.py`). It uses ttsim-perf's Blackhole DRAM structure
  (per-channel streaming, idle-read latency, time-ordered queuing across cores) with Quasar's 2 channels and assumed
  rates. It shows whether batching hides DRAM latency, single-bank shards using one channel, and latency-bound
  readers. craq-sim and ttsim-perf results are unchanged.
- Results report per-channel DRAM busy share. Sensitivity sweeps get a 1 s default budget.

## 1.5.2

- `check-drift --ref pr/N` reports `included` when the PR is already merged into the pinned code (its head branch may
  differ in unrelated files), instead of `drift`.
- Runner: batching schemes set both `dm_batch` and `tiles_per_cycle`; the factory rule is PR #62954's DRAM batching.
  qperf reproduces that PR's add result at 256 tiles per cluster (2.19x vs 2.10x measured on a newer craq-sim).

## 1.5.1

- Tested 13 edge-case prompts through the plugin; fixes from them: data-movement answers default to ttsim-perf;
  fusion / new-kernel ideas get an upper bound instead of a refusal; validated knobs say exactly what was run;
  layout-op overviews run a fixed preset (`presets/layout-ops.json`) with kernel changes and layout choices apart.
- `docs/EXAMPLES.md`: checked prompts and what to expect.

## 1.5.0

- First simulator check of a hypothetical setup: tilize with a real batched writer (48 runs). Refit on them: batched
  predictions within 0.8-1.6%, speedups within 1%; answers say `write_batch` is validated.
- Surveys show cores used per shape; times are labelled cycles (simulators) or ns (silicon-hypothesis); verdicts that
  change across cost scenarios are flagged; report pages are about 3x lighter.

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
