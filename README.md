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

Ask in any Claude Code session. To force the skill, type `/qperf:quasar-perf <question>`.

```
Would dm_batch 16 help a 128k-tile bf16 add from DRAM?
For a 64k-tile add from DRAM, would 2 more reader threads or 30% faster compute help?
Is it worth implementing per-thread NoC barriers? When would it help?
How long does tilize of a 3136x64 tensor take, and would height-sharding it help?
Is resharding a 1024x1024 tensor from 8 to 16 height shards in L1 faster than to 4?
What is the best multicore setup for tilize of a 64x2048 tensor: cores, split, batching, buffering?
Use the quasar-perf skill on PR <link>: compare main vs the PR's batching and other options.
```

Each answer gives:
- the bottleneck;
- each option's time, speedup and verdict (`faster`, `no_meaningful_gain`, `inconclusive`);
- the best choice;
- the limits.

Reports show how each setup runs: the core grid it uses and each core's reader -> compute -> writer flow, with the
bottleneck marked, next to charts that compare every option.

Ranges are cost scenarios, not confidence intervals. Unsupported cases are reported as such, never guessed. Add
"make a report with charts" to get a shareable page.

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
