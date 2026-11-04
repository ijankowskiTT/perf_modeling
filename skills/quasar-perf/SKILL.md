---
name: quasar-perf
description: Answer performance questions about Quasar binary_ng eltwise (reader -> compute -> writer) with the deterministic qperf model - R/C/W thread allocation, DM batch, tiles_per_cycle, ring depth (entries_per_thread), compute or DRAM/L1 what-ifs, kernel-change ideas, "would X help this add?", "which configs should I benchmark?". Use instead of estimating cycles yourself. Read-only: never runs hardware, simulators or kernel changes.
---

# quasar-perf

Delegate: spawn the `qperf:qperf-runner` agent (Haiku by default; pass model "sonnet" for multi-part questions
such as a PR analysis or many layouts) with the user's question and any context it needs
(PR summary, sizes, layouts). Do not run qperf or read its files yourself. Relay its answer as is (do not redo or expand it); if it returns a
report path, publish that HTML with the Artifact tool when available, else give the path.

Only if the Agent tool is unavailable: follow `${CLAUDE_PLUGIN_ROOT}/agents/qperf-runner.md` yourself.
