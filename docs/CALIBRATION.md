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

### binary_ng fp32 (add, multiply)

fp32 add and multiply run the SFPU kernel (UnpackToDest, 32-bit DST: at most 2 tiles per batch) and take identical
cycles on both simulators. `calibration/fit_fp32_binary.py` fits `sfpu:{add,multiply}:fp32` from the bf16 SFPU seed
into `calibration/out/fp32binary-<profile>.json`; data `calibration/data/fp32binary-<profile>.jsonl`. The variants
carry fp32-specific DM terms: craq-sim `noc.write_latency` (a 4 KB write ends ~100 cycles later), ttsim-perf
`noc.read_service.l1`/`noc.write_service.l1`. Mean error (fit / holdout): craq-sim 1.5% / 1.4% (max 4%, at 32
tiles); ttsim-perf borrowed and DRAM 0.5-0.8% / 0.4-0.6%, L1 9% / 5% (max 16%).

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

### fp32 (tilize / untilize, padding variants, sharding moves)

The bf16 fit stays; `calibration/fit_layout.py --dtype fp32` fits only `<p>.f32_<term>` overrides (compute,
per-byte, L1 and kernel-start terms; `FP32_TERMS`) on `calibration/data/fp32layout-<profile>.jsonl` (the bf16
datasets' 888 cases rerun in fp32, all outputs exact) and emits them to `calibration/out/fp32layout-<profile>.json`.
Fit mean (held-out mean in brackets):

| Factory | craq-sim | ttsim-perf |
|---|---|---|
| tilize Default | 0.7% (1.4%) | 1.3% (1.8%) |
| tilize SingleCore | 1.6% (1.4%) | 0.7% (0.5%) |
| tilize Sharded | 0.7% (0.4%) | 1.0% (1.0%) |
| tilize_with_val_padding Default | 0.6% (0.8%) | 0.5% (0.4%) |
| tilize_with_val_padding SingleCore | 0.6% (1.1%) | 0.8% (2.5%) |
| untilize MultiCore | 0.4% (0.4%) | 1.4% (1.2%) |
| untilize SingleCore (single-buffered DFBs, as the factory) | 1.0% (0.3%) | 2.0% (1.5%) |
| untilize Block / untilize_with_unpadding BlockInterleaved | 2.3% (2.1%) | 0.5% (1.0%) |
| untilize_with_unpadding Interleaved | 1.4% (2.1%) | 0.3% (0.8%) |
| untilize_with_unpadding SingleCore | 2.5% (2.8%) | 2.0% (2.6%) |
| interleaved_to_sharded | 1.4% (1.9%) | 1.5% (2.6%) |
| sharded_to_interleaved | 1.5% (2.2%) | 1.8% (2.8%) |

Reproduce (each call merges into the emitted file):

```bash
for op in tilize tilize_with_val_padding interleaved_to_sharded sharded_to_interleaved untilize,untilize_with_unpadding; do
  python calibration/fit_layout.py --data calibration/data/fp32layout-craq-sim.jsonl --sim craq-b702fd90 --op $op \
      --dtype fp32 --emit calibration/out/fp32layout-craq-sim.json     # and --sim perf / ttsim-perf
done
# craq-sim untilize_with_unpadding Interleaved: its fp32 per-request DM costs differ too
python calibration/fit_layout.py --data calibration/data/fp32layout-craq-sim.jsonl --sim craq-b702fd90 \
    --op untilize,untilize_with_unpadding --dtype fp32 --only uwp --emit calibration/out/fp32layout-craq-sim.json \
    --terms cu_fixed,cu_per_tile,cu_first,cu_tail,um_fixed,um_per_tile,pk_fixed,pk_per_tile,rd_cpb,wr_cpb,rd_cpb_l1,wr_cpb_l1,l1_gbw,rd_issue,wr_issue,rd_fixed,wr_fixed
python calibration/merge_params.py calibration/out/fp32layout-craq-sim.json craq-sim   # into the profile
```

Known fp32 gaps: untilize_with_unpadding SingleCore on craq-sim (narrow writer-bound rows -5%, as in bf16; rows
of 4 sub-blocks +10%), untilize Block on craq-sim with single-block DFBs (S = 15, 16: +10-13%), sharding moves
with the interleaved side in L1 on ttsim-perf (bank conflicts are linear in the conflicting cores; up to 17%), wide
tilize from L1 on ttsim-perf (-12%).

## Data

`calibration/data/<profile>.jsonl` (binary_ng) and `layout-<profile>.jsonl` (tilize/untilize, from
`calibration/run_dm_case.py`; `fp32layout-<profile>.jsonl` the same cases in fp32), one simulator run per line. binary_ng records: `layout` (`ddd`/`lll`/`hhh`/`lld`/`ddl`: a, b,
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
