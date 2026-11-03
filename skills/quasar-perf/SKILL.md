---
name: quasar-perf
description: Answer performance questions about Quasar binary_ng eltwise (reader -> compute -> writer) with the deterministic qperf model - R/C/W thread allocation, DM batch, tiles_per_cycle, ring depth (entries_per_thread), compute-time what-ifs, "would X help this add?", "which 3 configs should I benchmark?". Use instead of estimating cycles yourself. Read-only: never runs hardware, simulators or kernel changes.
---

# quasar-perf

You write the request and explain the result; the numbers come only from qperf. Write the request to a file
(scratch directory) and run, with an absolute path:

```bash
PYTHONPATH="${CLAUDE_PLUGIN_ROOT}" python3 -m qperf analyze --request /abs/path/req.json --format json
```

`--format text` gives a quick view. Python 3, standard library only; nothing to install.

## Request
Start from `${CLAUDE_PLUGIN_ROOT}/skills/quasar-perf/examples/`; schema in
`${CLAUDE_PLUGIN_ROOT}/schema/request.schema.json`.
- `action`: `estimate` (one config), `compare` (baseline + candidates, put every alternative in one call),
  `tune` (all valid configs, `analysis.top_k`), `sensitivity` (one cost key, `compute_service_scale` or
  `num_tiles`).
- Knobs: `reader_threads`, `compute_threads`, `writer_threads`, `dm_batch`, `tiles_per_cycle`,
  `entries_per_thread` (= `TTNN_QSR_*`). `baseline: {}` = the factory's rule.
- Kernel-change what-ifs: `barriers: per_thread` (transaction IDs), `work_split: dynamic` (shared work queue).
  Say they are hypothetical: no simulator can confirm them.
- `profiles.costs`: `craq-sim` (default for every question, DRAM included), `ttsim-perf` (only when an operand
  is in L1 or the question compares L1 vs DRAM: it times L1 NoC transfers; neither simulator times DRAM
  transfers), `silicon-hypothesis` (assumed clock, NoC and DRAM latency/bandwidth; call it a hypothesis). Plain
  ttsim has no profile; do not substitute one.
- One or two calls answer most questions; do not loop to hunt for a speedup.

## Answer
- Bottleneck, the alternative worth trying, its speedup range and verdict, the main limitation. Map configs
  back to `TTNN_QSR_*` and show the request so the user can rerun it.
- `no_meaningful_gain` / `inconclusive`: no clear winner; prefer the simpler config and suggest one simulator run.
- Ranges are cost scenarios, not confidence intervals; `lower_bound_ns` is not achievable; times are simulator
  cycles, not silicon.
- Accuracy: craq-sim ~1-2% mean (R1C1W1 up to ~7%); ttsim-perf DRAM ~1.4%, L1 ~8% (up to ~40% at dm_batch=1).

## Report
When the user wants a report, a summary to share, or charts: rerun the same request with
`--format html --out <scratch>/qperf-report.html` (one page: headline, key points, time per config, time split
per role, sensitivity lines, all numbers) and publish it with the Artifact tool if available, else give the path.
`--format md --out <file>.md` writes the same content as markdown. Add your explanation in chat, not in the file.

## Never guess
`needs_input`, `unsupported`, `invalid`: report the reason verbatim. Unknown latencies: use `sensitivity` over a
range. If `check-drift --tree <tt-metal>` (same command form) reports `drift`, say the predictions describe the
pinned kernels.
