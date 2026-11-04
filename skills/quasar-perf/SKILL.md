---
name: quasar-perf
description: >-
  Answer performance questions about Quasar ops with the deterministic qperf model - binary_ng eltwise (R/C/W thread allocation, DM batch, tiles_per_cycle, ring depth, compute or DRAM/L1 what-ifs, kernel-change ideas) and data movement - tilize / untilize and their padding variants, interleaved_to_sharded / sharded_to_interleaved, move, reshard (factory choice, single vs multi core, shard strategy and core count, how time scales with height and width). "would X help this add?", "how long does this tilize take?", "is resharding to 16 cores faster?", "which configs should I benchmark?". Use instead of estimating cycles yourself. Read-only: never runs hardware, simulators or kernel changes.
---

# quasar-perf

Delegate: spawn the `qperf:qperf-runner` agent (Haiku by default; pass model "sonnet" for multi-part questions
such as a PR analysis or many layouts) with the user's question and any context it needs
(PR summary, sizes, layouts). Do not run qperf or read its files yourself. Relay its answer as is (do not redo or expand it); if it returns a
report path, publish that HTML with the Artifact tool when available, else give the path.

Follow-ups ("and with dm_batch 4?", "what about L1?"): continue the same runner with SendMessage so it keeps the
workload and request files; if that is not possible, spawn a new runner and pass the previous request file path and
the workload in one line.

Only if the Agent tool is unavailable: follow `${CLAUDE_PLUGIN_ROOT}/agents/qperf-runner.md` yourself.
