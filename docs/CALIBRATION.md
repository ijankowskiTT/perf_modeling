# Calibration

There is no Quasar silicon yet; profiles are fitted to simulator runs at tt-metal `0fff850f3c6`.

| Profile | Simulator | Times | Holdout mean error |
|---|---|---|---|
| `craq-sim` | craq-sim, branch `quasar` @`b702fd90` | DM RISC-V, Tensix | add 2.1% (L1 0.9%), multiply 1.4% |
| `ttsim-perf` | ttsim-perf (`ENABLE_PERF=1`) | DM RISC-V, Tensix, L1 NoC transfers | DRAM 1.4%, L1 7.3% (all L1 runs 8.5%, up to 41% at dm_batch=1) |
| `silicon-hypothesis` | none: ttsim-perf costs + assumed clock, NoC, DRAM | | n/a |

Neither simulator times DRAM transfers; craq-sim times no transfers, so L1 and DRAM placements predict the same.
Plain ttsim has no profile: its cycle counts are not physically consistent (4 compute threads run 6.4x faster
than 1). The model picks the measured-best configuration in every group, except one small L1 group on ttsim-perf
(`lll` at 32 tiles: 23% regret). Known misses: R1C1W1 (+7%),
and the default config at 256+ tiles per cluster (-3%).

### tilize / untilize and the padding variants

Fitted per factory with `calibration/fit_layout.py` (least squares on the span and the DM threads' end times; every
4th run held out). Mean error over all runs (held-out mean in brackets):

| Factory | craq-sim | ttsim-perf |
|---|---|---|
| tilize Default | 1.3% (2.1%) | DRAM 0.7%, L1 1.2% (1.1%) |
| tilize SingleCore | 1.6% (1.0%) | 1.1% (1.4%) |
| tilize Sharded | 0.5% (0.7%) | 0.4% (0.4%) |
| untilize MultiCore | 0.4% (0.8%) | DRAM 0.8%, L1 3.8% (2.5%) |
| untilize SingleCore | 0.8% (0.7%) | 2.3% (1.9%) |
| tilize_with_val_padding Default | 0.7% (0.9%) | 0.4% (0.6%) |
| untilize_with_unpadding Interleaved | 0.6% (0.6%) | 0.2% (0.2%) |
| untilize Block / untilize_with_unpadding BlockInterleaved (shared kernels) | 0.5% / 1.3% (1.4%) | 0.7% / 0.7% (1.0%) |

Known gaps: untilize from L1 on ttsim-perf (up to 11%), single-tile-high untilize SingleCore on ttsim-perf (up to
14%), widths with a prime tile count (13 tiles: 5%). craq-sim cannot run tilize wider than 64 tiles.

## Data

`calibration/data/<profile>.jsonl` (binary_ng) and `layout-<profile>.jsonl` (tilize/untilize, from
`calibration/run_dm_case.py`), one simulator run per line. binary_ng records: `layout` (`ddd`/`lll`/`hhh`/`lld`/`ddl`: a, b,
out in DRAM, L1 or borrowed), `op`, `tpc` (tiles per cluster), `knobspec`, `span_med` (cycles), `src` (campaign;
`B` is the holdout).

## Check or refit

```bash
python -m qperf calibrate report --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim --rank
python -m qperf calibrate fit --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim \
    --keys rd.issue,rd.fixed,wr.issue,wr.fixed,cu.per_tile --revision <new> --filter op=add
python calibration/make_silicon_hypothesis.py        # after refitting ttsim-perf
python calibration/fit_layout.py --data calibration/data/layout-craq-sim.jsonl --sim craq-b702fd90 --op tilize \
    --write craq-sim                                  # tilize/untilize (scipy: use the tt-metal venv)
```

## After a simulator update

Simulator builds change timing (craq-sim #480 moved some runs by up to 21%). Re-run the dataset's cases on the
new build with `calibration/sweep.sh` (needs `TREE`, `QPERF_SIM_DIR`), give the build its own `sim` tag, and refit
if `calibrate report` drifts.
