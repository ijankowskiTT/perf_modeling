# Calibration

There is no Quasar silicon yet; profiles are fitted to simulator runs at tt-metal `0fff850f3c6`.

| Profile | Simulator | Times | Holdout mean error |
|---|---|---|---|
| `craq-sim` | craq-sim, branch `quasar` @`b702fd90` | DM RISC-V, Tensix | add 2.1% (L1 0.9%), multiply 1.4% |
| `ttsim-perf` | ttsim-perf (`ENABLE_PERF=1`) | DM RISC-V, Tensix, L1 NoC transfers | DRAM 1.4%, L1 9% (up to 32% at dm_batch=1) |
| `silicon-hypothesis` | none: ttsim-perf costs + assumed clock, NoC, DRAM | | n/a |

Neither simulator times DRAM transfers; craq-sim times no transfers, so L1 and DRAM placements predict the same.
Plain ttsim has no profile: its cycle counts are not physically consistent (4 compute threads run 6.4x faster
than 1). The model picks the measured-best configuration in every group, except two small L1 groups on ttsim-perf
(`lld`/`lll` at 32 tiles: 16-19% regret). Known misses: R1C1W1 (+7%),
and the default config at 256+ tiles per cluster (-3%).

## Data

`calibration/data/<profile>.jsonl`, one simulator run per line: `layout` (`ddd`/`lll`/`hhh`/`lld`/`ddl`: a, b,
out in DRAM, L1 or borrowed), `op`, `tpc` (tiles per cluster), `knobspec`, `span_med` (cycles), `src` (campaign;
`B` is the holdout).

## Check or refit

```bash
python -m qperf calibrate report --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim --rank
python -m qperf calibrate fit --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim \
    --keys rd.issue,rd.fixed,wr.issue,wr.fixed,cu.per_tile --revision <new> --filter op=add
python calibration/make_silicon_hypothesis.py        # after refitting ttsim-perf
```

## After a simulator update

Simulator builds change timing (craq-sim #480 moved some runs by up to 21%). Re-run the dataset's cases on the
new build with `calibration/sweep.sh` (needs `TREE`, `QPERF_SIM_DIR`), give the build its own `sim` tag, and refit
if `calibrate report` drifts.
