# qperf

Performance model of Quasar ops for Claude Code and other agents. Ask a performance question in plain words and get
an answer in seconds, without a simulator or hardware. The answer comes from replaying the real kernel schedule.

## Install

Needs read access to this repo and GitHub credentials: an SSH key, or `gh auth login` plus
`export CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`.

```bash
claude plugin install qperf --marketplace ijankowskiTT/perf_modeling
# before Claude Code 2.1.292: claude plugin marketplace add ijankowskiTT/perf_modeling && claude plugin install qperf@perf-modeling
```

To update, run `claude plugin update qperf@perf-modeling` or turn on auto-update in `/plugin`.

## Use

Ask in any Claude Code session. To force the skill, start with "Use the quasar-perf skill:" or type
`/qperf:quasar-perf <question>`. Add "make a report with charts" for a shareable page.

**Example: full picture of an op.** Prompt:

```
Use the quasar-perf skill: give me the full picture of tilize on Quasar. How does it run today, which multicore
approaches are worth it across tensor sizes, where does each regress, and what limits it? Make a report with charts.
```

Output: [tilize survey report](https://claude.ai/artifact/VJrBHzpCKU5mwCyVeJLk7P?sk=ubpNNdqu4o1YJkd7hUFmlA) (Tenstorrent
only). It covers 8 approaches over 54 tensor sizes, a regression heatmap for each, and the limiters, with assumed DRAM.

**Example: wins across several ops.** Prompt:

```
Use the quasar-perf skill: where are the biggest performance wins in Quasar's layout ops (tilize, untilize, sharding,
reshard)? For each op, show how it runs today, what limits it, which changes help across tensor sizes, where they
regress, and what caps them on real hardware. Rank the improvements by expected gain. Make a report with charts.
```

Output: [layout ops report](https://claude.ai/artifact/Sfj2kJZLYg65LfE2SsTzfC?sk=id4mHCl0N8wNsmEaQi3QMw) (Tenstorrent
only). One page with a summary per op and two rankings over one size grid: kernel changes (hypothetical) and layout
choices (which change what the next op receives). It also shows the DRAM caps, with a section per op.

**Example: a merged PR against other designs.** Prompt:

```
Use the quasar-perf skill on PR https://github.com/tenstorrent/tt-metal/pull/62954: it batches binary_ng beside DRAM by
each Neo's share (from 4 tiles, largest power of 2, capped by a DEST section). Compare that rule with other batching
schemes (no batching, fixed dm_batch 4 / 8 / 16, batch by the full share without the DEST cap) for bf16 add and multiply
from DRAM across tile counts. Where does each win or regress? Make a report with charts.
```

Output: [batching report](https://claude.ai/artifact/UoWwykpLuKVMJXBR9m3qJ8?sk=k947PDf0x5kEOHkJV1E9Ng) (Tenstorrent only).
It compares six schemes from 1 to 64K tiles per cluster. qperf reproduces the PR's own result (add at 256 tiles:
2.19x, against 2.10x measured). The gain comes from compute batching, and the DEST cap is required. Where the PR rule
leaves time behind: batch 2 for multiply at 8 tiles (5.8%), and DM batch 16 for large adds (4-11%, on craq-sim with no
DRAM time).

More checked examples, including edge cases (unsupported parts, invalid requests, extrapolation, fusion ideas,
confidence questions): [docs/EXAMPLES.md](docs/EXAMPLES.md).

| Use case | Prompt |
|---|---|
| Design a multicore op | Bench the multicore approaches for bf16 untilize across tensor sizes. Which is best, and where does each regress? |
| Best setup for one shape | What is the best multicore setup for tilize of a 64x2048 tensor: cores, split, batching, buffering? |
| Review a PR | Use the quasar-perf skill on PR \<link\>: compare main vs the PR's batching and other options. |
| Tune an eltwise op | Which 3 configs should I benchmark for a 4k-tile bf16 multiply in L1? |
| Is a kernel idea worth it | Is it worth implementing per-thread NoC barriers? When would it help? |
| What limits it | For a 64k-tile add from DRAM, would 2 more reader threads or 30% faster compute help? |
| Hardware what-if | How does a 128k-tile add change if DRAM latency doubles? Does a larger DM batch hide it? |
| Shard layout | Is it worth height-sharding a 1024x256 tensor over 32 cores instead of 8? Which strategy is fastest? |
| Reshard plan | Is resharding a 1024x1024 tensor from 8 to 16 height shards in L1 faster than to 4? Which factory runs? |
| Real-hardware ceiling | How far can tilize of an 8192x2048 tensor go on real hardware once DRAM bandwidth counts? |

Each answer gives:
- the bottleneck;
- each option's time, speedup and verdict (`faster`, `no_meaningful_gain`, `inconclusive`);
- the best choice;
- the limits.

Reports show how each setup runs: the core grid it uses and each core's reader -> compute -> writer flow, with the
bottleneck marked, next to charts that compare every option.

Ranges are cost scenarios, not confidence intervals. Unsupported cases are reported as such, never guessed.

Questions about main, a branch or a PR first check on GitHub whether the modeled kernel files changed.

Other agents: `agents/qperf-runner.md` is self-contained and needs only Python 3.

## How it works

```
you -> your Claude session -> qperf-runner (Haiku) -> qperf model (Python, no LLM)
```

The main model only routes questions to the runner. qperf replays the kernel's schedule (loops, rings, barriers,
work split) with step costs fitted to ~3,800 simulator runs. Cost per question: ~$0.14, against $0.26 for the main
model alone (`python bench/tokens.py`).

## Coverage

Mean error per factory against the simulators, bf16 and fp32:

| Op | Factories / paths | craq-sim | ttsim-perf |
|---|---|---|---|
| binary_ng `add`, `multiply` | native; any threads, batch, tiles_per_cycle, ring depth | ~1-2% | DRAM ~1.4%, L1 ~8% |
| `tilize` | Default, SingleCore, Sharded | <=1.6% | <=1.4% |
| `untilize` | MultiCore, SingleCore, Block | <=2.2% | <=1.4% |
| `tilize_with_val_padding` | Default, SingleCore | <=2.4% | <=2.8% |
| `untilize_with_unpadding` | Interleaved, BlockInterleaved, SingleCore | <=2.6% | <=2.5% |
| `interleaved_to_sharded`, `sharded_to_interleaved` | TILE, ROW_MAJOR; height / width / block | <=1.7% | <=2.3% |
| `move` | overlapping output (the only path Quasar runs) | <=0.4% | <=1.5% |
| `reshard` | SameWidth(Write), SameHeight(Write), Generic, CopyPages | <=1.9% | <=3.1%; SameHeight 16%, Generic 10% (flagged) |

Details are in [docs/CALIBRATION.md](docs/CALIBRATION.md).

Not covered (answered as `unsupported`, never with a number):
- other ops;
- move with a separate output, and sharded move;
- reshard CopyLocal and Generic from DRAM;
- dtype conversion in sharding ops.

No Quasar silicon exists yet: times are simulator cycles. Confirm close calls (<3%) with one simulator run.

DRAM: neither simulator times Quasar DRAM. The `silicon-hypothesis` profile adds a Quasar DRAM channel model: 2
channels with per-channel queuing and idle-read latency, using ttsim-perf's Blackhole structure at assumed rates. It
shows when an op is DRAM-bound and whether batching hides DRAM latency. It is a hypothesis until it can be
calibrated.

## Findings

For `add`/`multiply` from DRAM on 32 clusters (craq-sim):

| Question | Answer |
|---|---|
| Best threads, batch, tiles_per_cycle? | Factory default is within 2% of the best |
| Larger DM batch? | dm_batch 16: +8% at 32k tiles, +11% at 128k, nothing at 8k |
| Faster compute? | `add` is writer-bound (2x compute gives 2%); `multiply` is compute-bound (1.85x) |
| Deeper rings? | No effect, even at 2000-cycle DRAM latency |
| Dynamic work queue? | No gain: the static split is balanced to 1 tile |
| Per-thread barriers? | Nothing from DRAM; 1.16x from L1 (ttsim-perf) |

## More

[docs/MODEL.md](docs/MODEL.md) explains what is modeled, [docs/CALIBRATION.md](docs/CALIBRATION.md) covers accuracy
and refitting, and `python -m pytest -q tests` runs the tests.
