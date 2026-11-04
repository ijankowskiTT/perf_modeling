---
name: quasar-perf
description: >-
  Answer performance questions about Quasar ops with the deterministic qperf model - binary_ng eltwise (R/C/W thread allocation, DM batch, tiles_per_cycle, ring depth, compute or DRAM/L1 what-ifs, kernel-change ideas) and data movement - tilize / untilize and their padding variants, interleaved_to_sharded / sharded_to_interleaved, move, reshard (factory choice, single vs multi core, shard strategy and core count, how time scales with height and width). "would X help this add?", "how long does this tilize take?", "is resharding to 16 cores faster?", "which configs should I benchmark?". Use instead of estimating cycles yourself. Read-only: never runs hardware, simulators or kernel changes.
---

# quasar-perf

Spawn the `qperf:qperf-runner` agent with the user's question and the context it needs (PR summary, sizes,
layouts). It runs on Haiku; pass model "sonnet" for multi-part questions such as a PR analysis. Do not run qperf
or read its files yourself.

- Relay the runner's answer as is. If it returns a report path, publish the HTML with the Artifact tool if you have
  it; otherwise give the path.
- Follow-ups ("and with dm_batch 4?"): continue the same runner with SendMessage. If you can't, spawn a new runner
  with the previous request file path.
- No Agent tool: follow `${CLAUDE_PLUGIN_ROOT}/agents/qperf-runner.md` yourself.
