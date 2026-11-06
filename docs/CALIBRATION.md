# Calibration

There is no Quasar silicon yet. Profiles are fitted to simulator runs at tt-metal `0fff850f3c6`.

| Profile | Simulator | Times |
|---|---|---|
| `craq-sim` | craq-sim, branch `quasar` @`b702fd90` | DM RISC-V, Tensix; no transfers (L1 = DRAM) |
| `ttsim-perf` | ttsim-perf (`ENABLE_PERF=1`) | DM RISC-V, Tensix, L1 NoC transfers |
| `silicon-hypothesis` | none: ttsim-perf costs plus an assumed clock, NoC and DRAM | n/a |

Neither simulator times DRAM. Plain ttsim has no profile: its cycle counts are inconsistent (4 compute threads run
6.4x faster than 1).

## Accuracy

Mean error over all runs. Held-out runs (every 4th) fall within the same bounds.

**binary_ng** (`add`, `subtract`, `multiply`, `divide`):

| Case | craq-sim | ttsim-perf |
|---|---|---|
| add bf16, and subtract / FPU multiply bf16 (same kernels and cycles as add) | 2.1-2.6% | DRAM 1.4%, L1 8.5% (one thread per role on L1: up to 2x off) |
| multiply bf16 (SFPU) | 1.4% | DRAM 1.4%, L1 8.5% |
| add / subtract / multiply fp32 | 1.5% | DRAM / borrowed <= 0.8%, L1 9-11% |
| divide bf16 / fp32 | 0.7% / 0.9% | 2.3% / 2.5% |

`calibration/data/binops-<profile>.jsonl`: 652 runs, fitted by `calibration/fit_binops.py`. Only divide has its own
math cost (about 5x multiply, so always compute-bound). Ops the native factory doesn't run are answered `unsupported`
with the reason:
- `rsub` bf16, maximum, minimum, logaddexp, pow, logical ops and others run on another factory;
- comparisons, atan2, xlogy and remainder don't build on Quasar;
- broadcast runs only on the other factory.

The model picks the measured-best config in every group but one: on ttsim-perf, all three operands in L1 at 32
tiles is 23% off the best.

**Data movement**, bf16 / fp32:

| Op | Factory | craq-sim | ttsim-perf |
|---|---|---|---|
| tilize | Default | 1.3 / 0.9% | 1.2 / 1.4% |
| | SingleCore | 1.6 / 1.5% | 1.1 / 0.7% |
| | Sharded | 0.5 / 0.6% | 0.4 / 1.0% |
| tilize_with_val_padding | Default | 0.7 / 0.7% | 0.8 / 0.4% |
| | SingleCore | 2.4 / 0.7% | 2.8 / 1.3% |
| untilize | MultiCore | 0.4 / 0.4% | 0.8 / 1.4% |
| | SingleCore | 0.8 / 0.9% | 1.0 / 0.6% |
| | Block | 0.5 / 2.2% | 0.7 / 0.6% |
| untilize_with_unpadding | Interleaved | 0.6 / 1.5% | 0.4 / 0.4% |
| | BlockInterleaved | 1.3 / 2.4% | 0.7 / 0.9% |
| | SingleCore | 2.4 / 2.6% | 2.5 / 2.2% |
| interleaved_to_sharded | TILE | 1.5 / 1.5% | 1.8 / 2.3% |
| | ROW_MAJOR (aligned / unaligned) | 0.1 / 0.6% (bf16) | 1.9 / 0.9% (bf16) |
| sharded_to_interleaved | TILE | 1.7 / 1.7% | 1.9 / 2.0% |
| | ROW_MAJOR | 0.2% (bf16) | 0.8% (bf16) |
| move | TILE / ROW_MAJOR | 0.4 / 0.2% | 1.4 / 1.5% |
| reshard (bf16) | SameWidth | 1.7% | 3.1% |
| | SameWidthWrite, SameHeightWrite | <= 0.2% | 0.0% |
| | SameHeight | 0.0% | **16%** |
| | Generic | 0.5% | **9.5%** |
| | CopyPages | 1.9% | 1.4% |

Known gaps:

| Profile | Gap |
|---|---|
| ttsim-perf | reshard SameHeight and Generic: many small L1 responses move faster than 64 B/cycle. Answers warn, and tests cap them. |
| ttsim-perf | untilize from L1: up to 11%. fp32 sharding ops with the interleaved side in L1: up to 17%. |
| craq-sim | Can't run tilize wider than 64 tiles. Reshard spans under 250 cycles: up to 5%. |

## Validated hypothetical setups

The design knobs are hypothetical until a simulator runs them. Checked so far:

| Knob | How | Runs | Mean error | Speedup error |
|---|---|---|---|---|
| tilize `write_batch` 4, 8 | real kernel variant (`calibration/variants/tilize_writer_batched.cpp`), 12 shapes | 24 per profile | craq-sim 1.6%, ttsim-perf 0.8% | 0.4%, 1.0% |

Before refitting on these runs the model over-predicted the batching gain on ttsim-perf (4096x2048: 1.39x predicted,
1.30x measured; 3.2% mean speedup error). The joint fit separates the per-write and per-flush writer costs and adds
`tz.wr_bissue` (extra cost per write in a batch; 28 cycles on craq-sim, 0 on ttsim-perf). Answers say which knobs are
validated. Not validated yet: `cores`, `split`, `tiles_per_block`, `dfb_blocks`, `read_batch`.

## Data

`calibration/data/*.jsonl` has one simulator run per line:

| File | Content |
|---|---|
| `<profile>.jsonl` | binary_ng. `layout` is a, b, out in DRAM / L1 / borrowed (e.g. `ddd`); also `tpc` (tiles per cluster), `knobspec`, `span_med`; `src` `B` is the holdout. |
| `fp32binary-<profile>.jsonl` | binary_ng fp32 |
| `layout-`, `fp32layout-`, `rowmajor-`, `move-`, `reshard-<profile>.jsonl` | data-movement ops, from `calibration/run_dm_case.py` or `calibration/runners/<op>.py` |

## Check or refit

```bash
# binary_ng
python -m qperf calibrate report --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim --rank
python -m qperf calibrate fit --data calibration/data/craq-sim.jsonl --sim craq-b702fd90 --costs craq-sim \
    --keys rd.issue,rd.fixed,wr.issue,wr.fixed,cu.per_tile --revision <new> --filter op=add
python calibration/fit_fp32_binary.py              # fp32 SFPU terms
python calibration/make_silicon_hypothesis.py      # after refitting ttsim-perf

# data movement (scipy: use the tt-metal venv). Per op; ttsim-perf: --sim perf
python calibration/fit_layout.py --data calibration/data/layout-craq-sim.jsonl --sim craq-b702fd90 --op tilize \
    --emit calibration/out/layout-craq-sim.json    # --dtype fp32: only <p>.f32_* overrides; --only <prefix>, --terms a,b
python calibration/merge_params.py calibration/out/layout-craq-sim.json craq-sim
```

## After a simulator update

A new build changes timing: craq-sim #480 moved some runs by up to 21%.
1. Re-run the cases with `calibration/sweep.sh` (needs `TREE` and `QPERF_SIM_DIR`; set-up in `calibration/setup/`).
2. Give the build its own `sim` tag.
3. Refit if `calibrate report` drifts.
