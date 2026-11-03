# qperf

Performance model of Quasar binary_ng eltwise (`add`, `multiply`). For an op size, placement and `TTNN_QSR_*`
config it predicts the time, names the bottleneck and finds the best config, in milliseconds, without a
simulator. Python 3 standard library only.

## Install (Claude Code plugin)

Needs read access to this private repo and git credentials for GitHub on your machine (SSH key, or
`gh auth login` plus `export CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`).

```bash
claude plugin install qperf --marketplace ijankowskiTT/perf_modeling     # Claude Code 2.1.292+
# older: claude plugin marketplace add ijankowskiTT/perf_modeling && claude plugin install qperf@perf-modeling
```

Then ask Claude Code a question (below). Update: `claude plugin marketplace update perf-modeling && claude plugin
update qperf@perf-modeling`, or enable auto-update for the marketplace in `/plugin`.

## Example

```bash
python -m qperf analyze --request skills/quasar-perf/examples/shortlist_tune.json --format text
```
```text
  baseline   R4 C4 W2 dm_batch=8 tiles_per_cycle=8 entries_per_thread=16
      time 4832 ns  [4529 .. 5143]  bound 3485  bottleneck writer_issue
      reader: busy 50% starved 0%   compute: busy 29% starved 28%   writer: busy 63% starved 24%
  top2       R4 C4 W2 dm_batch=8 tiles_per_cycle=4 entries_per_thread=16
      time 5137 ns  [4804 .. 5480]   speedup 0.94x  -> slower
  search: 432 enumerated, 180 invalid, 252 of 252 valid evaluated, exhaustive=True
  recommend: keep baseline  no candidate is faster in every cost scenario by more than 2%
```

| Field | Meaning |
|---|---|
| `time` | simulator cycles (shown as ns); `[low .. high]` = all costs at their low/high scenario |
| `bound` | lower bound no schedule can beat |
| `bottleneck` | limiting resource: `reader_issue`, `writer_issue`, `compute_unpack_math`, `compute_pack`, bandwidth |
| `busy / starved` | per role: working / waiting for input (`backpressure`: waiting for space) |
| verdict | `faster`/`slower` in every scenario, `no_meaningful_gain` (<2%), `inconclusive` (run both) |

Invalid configs return the factory check they fail; unsupported ops return `needs_input`/`unsupported`, never a
guess. `--format json` gives the full response (`schema/response.schema.json`).

## Usage

**Claude Code** (plugin installed): ask in plain words ("which 3 configs should I benchmark for a 64k-tile add from
DRAM?"); the `quasar-perf` skill builds the request and explains the answer.

**CLI**: copy an example from `skills/quasar-perf/examples/` and run it:

```bash
python -m qperf analyze --request my.json --format text
python -m qperf analyze --request my.json --set workload.num_tiles=65536
python -m qperf analyze --request my.json --format html --out report.html   # charts; --format md for markdown
```

```json
{"schema_version": "1", "action": "compare", "profiles": {"costs": "craq-sim"},
 "workload": {"op": "add", "dtype": "bf16", "num_tiles": 4096,
              "placements": {"a": "dram", "b": "dram", "out": "dram"}},
 "baseline": {},
 "candidates": [{"id": "C2", "overrides": {"compute_threads": 2}}]}
```

- `action`: `estimate`, `compare`, `tune` (every valid config), `sensitivity` (sweep a cost or `num_tiles`).
- Knobs: `reader_threads`, `compute_threads`, `writer_threads`, `dm_batch`, `tiles_per_cycle`,
  `entries_per_thread`. `{}` = the factory's default rule.
- Placements: `dram`, `l1`, `borrowed`.
- Hypothetical kernel changes, compared like knobs: `"barriers": "per_thread"` (transaction-ID barriers) and
  `"work_split": "dynamic"` (shared work queue for the DM threads). Results carry a warning: no simulator can
  confirm them.

**Python**: `from qperf import analyze; analyze(request)`.

## Profiles

| `profiles.costs` | Use for | Holdout error |
|---|---|---|
| `craq-sim` | default | ~1-2% |
| `ttsim-perf` | L1 vs DRAM placement, NoC (craq-sim does not time transfers) | DRAM 1.4%, L1 7-9% |
| `silicon-hypothesis` | silicon what-ifs under assumed clock, NoC, DRAM | n/a |

No Quasar silicon exists yet: times are simulator cycles. Confirm close calls (<3%) with one simulator run.

## Findings

For `add`/`multiply` from DRAM on 32 clusters (craq-sim unless noted); each line has a request to rerun.

| Question | Answer | Request |
|---|---|---|
| Best R/C/W, batch, tiles_per_cycle? | The factory's default is within 2% of the best config (except small `add`, 1k tiles: dm_batch 4, +3.5%) | `shortlist_tune.json` |
| Larger DM batch? | dm_batch 16 is +8% at 32k tiles, +11% at 128k, nothing at 8k | `dm_batch16.json` |
| Faster compute? | `add` is writer-bound: 2x faster compute gives 2%. `multiply` is compute-bound: 1.85x | `compute_whatif.json` (`--set workload.op=multiply`) |
| Deeper rings? | No effect, even at 2000-cycle DRAM latency (silicon-hypothesis, 1 cluster): the barrier after each batch limits requests in flight; a larger batch hides latency | `ring_depth_latency.json` |
| Dynamic work queue? | No gain: the static split is balanced to 1 tile; claims cost 1-5% | `schedule_variants.json` |
| Per-thread barriers (transaction IDs)? | Nothing from DRAM; 1.16x from L1 (ttsim-perf), where the cluster-wide barrier stalls every reader | `schedule_variants.json` |

## More

- [docs/MODEL.md](docs/MODEL.md): what is modeled
- [docs/CALIBRATION.md](docs/CALIBRATION.md): accuracy, data, refitting
- `python -m pytest -q tests`; `python -m qperf check-drift --tree <tt-metal>` flags kernel changes
