# qperf

Performance model of Quasar ops (binary_ng eltwise, tilize/untilize and their padding variants) for Claude Code and
other agents. Ask a performance question in plain words; the answer is backed by a model of the real kernel
schedule, in seconds, without a simulator or hardware.

## Install

Needs read access to this repo and GitHub credentials on your machine (SSH key, or `gh auth login` plus
`export CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`).

```bash
claude plugin install qperf --marketplace ijankowskiTT/perf_modeling
# Claude Code before 2.1.292: claude plugin marketplace add ijankowskiTT/perf_modeling && claude plugin install qperf@perf-modeling
```

Update with `claude plugin update qperf@perf-modeling`, or turn on auto-update in `/plugin`.

## Use

Ask in any Claude Code session. Claude picks the skill up from the question; to force it, start with
"Use the quasar-perf skill" or type `/qperf:quasar-perf <question>`.

```
Would dm_batch 16 help a 128k-tile bf16 add from DRAM?
For a 64k-tile add from DRAM, would 2 more reader threads or 30% faster compute help?
Is it worth implementing per-thread NoC barriers? When would it help?
Which 3 configs should I benchmark for a 4k-tile multiply in L1?
How long does tilize of a 3136x64 tensor take, and would height-sharding it help?
How does untilize time grow with width? Where does it stop being writer-bound?
Is it worth sharding this 1024x256 tensor over 32 cores instead of 8?
How does it change if DRAM latency doubles? Does a deeper ring hide it?
Use the quasar-perf skill on PR <link>: compare main vs the PR's batching and other options.
```

Add "make a report with charts" to get a shareable page: headline, time per config, where each role's time goes,
sensitivity lines and all numbers.

Comparisons with "main" use the kernels pinned in the model (re-verified against tt-metal main on 2026-11-04). For a
question about main, a branch or a PR, Claude first checks on GitHub whether those kernel files changed and says so.

Each answer gives the bottleneck, the options with time, speedup and verdict (`faster`, `no_meaningful_gain`,
`inconclusive`), the best choice and why, and its limits. Ranges are cost scenarios, not confidence intervals.
Unsupported cases are reported as such, never guessed.

**Other agents**: `agents/qperf-runner.md` is self-contained (request format, command, rules); any agent that can
run Python 3 can follow it. No packages to install.

## How it works

```
you / an agent  ->  your Claude session  ->  qperf-runner agent (Haiku)  ->  qperf model (Python, no LLM)
```

The skill hands each question to the runner on a cheap model (Sonnet for multi-part questions such as a PR
analysis); the main model only routes and relays. qperf replays the kernel's exact schedule (loops, rings,
barriers, work split) with step costs fitted to 600+ simulator runs.

Token use per question (`python bench/tokens.py`):

| Setup | Cost | Tokens |
|---|---|---|
| Main model alone | $0.26 | 232k |
| Haiku runner (default) | $0.15 | 95k |

## Coverage

| | Covered | Not yet |
|---|---|---|
| Ops | binary_ng `add`, `multiply` (bf16, fp32); `tilize` (Default, SingleCore, Sharded), `untilize` (MultiCore, SingleCore, Block), `tilize_with_val_padding` (Default, SingleCore), `untilize_with_unpadding` (Interleaved, BlockInterleaved, SingleCore), `interleaved_to_sharded` / `sharded_to_interleaved` (TILE, height/width/block); bf16 | `subtract`, `rsub`, `divide` (no costs yet); move, reshard (next); addalpha, activations, broadcast, matmul |
| Layout | each operand in DRAM, L1 (interleaved) or borrowed (resident shard), any mix | tail rings of uneven slices |
| Knobs | reader/compute/writer threads, `dm_batch`, `tiles_per_cycle`, `entries_per_thread` (`TTNN_QSR_*`) | |
| Kernel changes | per-thread NoC barriers, dynamic work queue (hypothetical: no simulator can confirm) | other schedules |
| Platform | craq-sim (binary_ng ~1-2%, tilize/untilize <=2% held out), ttsim-perf (binary_ng DRAM 1.4%, L1 ~8%), silicon as a hypothesis | silicon, DRAM timing in either simulator; craq-sim cannot run tilize wider than 64 tiles |

No Quasar silicon exists yet: times are simulator cycles. Confirm close calls (<3%) with one simulator run.

## Findings

For `add`/`multiply` from DRAM on 32 clusters, craq-sim unless noted:

| Question | Answer |
|---|---|
| Best threads, batch, tiles_per_cycle? | The factory's default is within 2% of the best (except a 1k-tile `add`: dm_batch 4, +3.5%) |
| Larger DM batch? | dm_batch 16: +8% at 32k tiles, +11% at 128k, nothing at 8k |
| Faster compute? | `add` is writer-bound: 2x faster compute gives 2%. `multiply` is compute-bound: 1.85x |
| Deeper rings? | No effect even at 2000-cycle DRAM latency; a larger batch hides latency |
| Dynamic work queue? | No gain: the static split is balanced to 1 tile |
| Per-thread barriers? | Nothing from DRAM; 1.16x from L1 (ttsim-perf) |

## More

[docs/MODEL.md](docs/MODEL.md) (what is modeled) · [docs/CALIBRATION.md](docs/CALIBRATION.md) (accuracy,
refitting) · `python -m pytest -q tests`
