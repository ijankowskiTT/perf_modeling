# Model

qperf replays the real schedule of one Quasar program against a cost profile. Each op in `qperf/ops/` builds threads
and rings from the pinned kernels (`profiles/kernels/`); one engine (`qperf/engine.py`) evaluates them.

## Engine

- Threads run their kernels' own loops over DFB tile counters, with finite rings and a half-sync DST.
- Barriers are cluster-wide, as in the firmware; timed transfers complete through one per-cluster queue.
- Shared NoC, L1 and DRAM bandwidth are bounds.
- Past 2048 tiles per cluster the result is extrapolated exactly from the periodic steady state.
- A configuration that cannot progress is invalid.

## Ops

**binary_ng** (`add`, `multiply`; `qperf/ops/binary_ng.py`): no broadcast, TILE layout, operands in DRAM, L1 or
borrowed. Default config and work split mirror the factory (`rule.py`); its checks are in `constraints.py`, exact
traffic counts in `traffic.py`.

**tilize / untilize and padding variants** (`qperf/ops/layout.py`):
- One reader, compute and writer thread per core. Work is split by tile-rows, and the factory is chosen as the
  device op chooses it.
- Each DM thread's DMA moves one transfer at a time, ~56 B/cycle on craq-sim. A DM kernel has a prologue, and a
  producer exits only once its DFB is drained.
- untilize: unpack and pack run as two stages over DEST sections of <= 8 tiles. tilize: one step per tile-row.
- Padding: pad fill by DMA zero-fill or a store loop. Block factories use S x S-tile squares.

**interleaved_to_sharded / sharded_to_interleaved**:
- Only the interleaved side moves data, with a barrier every `thr` reads.
- ROW_MAJOR: one request per shard row. From an unaligned DRAM source: a scratch read plus a local copy per row,
  4 in flight. Uneven last shards are clipped.

**move** (`qperf/ops/move.py`): the overlapping-output path only. That is all Quasar runs: MoveProgramFactory's
kernels are rejected, and an output at the input's address runs no program.
1. Each core reads its pages one at a time, with a barrier each.
2. All cores handshake through the controller core.
3. Then they write.

**reshard** (`qperf/ops/reshard.py`): factory selection mirrors `ReshardDeviceOperation::select_program_factory`.

| Case | Factory |
|---|---|
| HEIGHT→HEIGHT | SameWidth (output cores read); SameWidthWrite (DRAM output: input cores write) |
| ROW_MAJOR WIDTH→WIDTH | SameHeight; SameHeightWrite (DRAM output) |
| Other cases with the output in L1 | Generic (strided page runs, as `create_map_for_reshard`) |
| DRAM→DRAM | CopyPages (32 cores, page by page through a 1-entry DFB) |

Both DM kernels of a core split the segments: one request per segment and one barrier at the end. On ttsim-perf,
each NoC port and each directed link moves 64 B/cycle. A core's reads end when the busiest port or link on its
paths has moved its bytes.

**fp32**: same kernels, twice the bytes; untilize sub-blocks hold <= 4 tiles. `<p>.f32_<term>` overrides
`<p>.<term>` (compute, per-byte, kernel starts). For sharding ops with the interleaved side in L1, cores that start on
the same L1 bank queue there (`<p>.l1_conf_cpb`).

## Simulator rules

| Simulator | Rule |
|---|---|
| craq-sim | No NoC latency or contention, so L1 and DRAM time the same. Reshard transfers cost per request, not per byte, so reshard comparisons on it are `inconclusive`. |
| ttsim-perf | Times L1 NoC transfers, not DRAM. A core's L1 reads and writes share one completion queue, and many cores share a global L1 bandwidth. Comparisons with a DRAM reshard side are `inconclusive`. |

## Hypothetical variants

These are not in the kernels, and results say so. They reuse the factory's calibrated per-request, per-byte,
per-block and per-tile costs, so they compare fairly with the factory, but no simulator has run them.
- binary_ng `barriers: per_thread`: each DM thread waits only on its own transfers.
- binary_ng `work_split: dynamic`: DM threads claim batches from one queue, at `dq.claim` (30 cycles assumed) per batch.
- tilize Default / untilize MultiCore design knobs (`tune` searches them by default; a setup equal to the factory's
  is not hypothetical):

| Knob | Values | Meaning |
|---|---|---|
| `cores` | 1-32 | cores used |
| `split` | rows, cols, 2d | whole tile-rows per core (factory); tile-rows cut into column chunks; the best rows x chunks grid |
| `tiles_per_block` | divisor of the chunk | block width: more blocks, more barriers, smaller DFBs |
| `dfb_blocks` | 1-3 | DFB depth in blocks (DFBs must fit half of L1) |
| `write_batch` / `read_batch` | 1, 4, 8 | tiles per writer (tilize) / reader (untilize) barrier |

Every tilize / untilize result carries `structure`: the core grid, the split and the busiest core's requests,
bytes, barriers, compute steps and DFB sizes. Reports draw it.

## Not modeled

- NoC routing (except reshard on ttsim-perf), most L1 bank conflicts, dynamic arbitration, launch skew.
- move with a separate output, and sharded move.
- reshard CopyLocal (wrong output on both simulators), Generic from DRAM, unaligned HEIGHT→HEIGHT rows, row-major
  width changes.

## Costs

Every parameter in `profiles/costs/*.json` has `value`, `low`, `high`, `class` and `source`. `low` and `high` move
all costs together. They are scenarios, not confidence intervals. A missing cost gives `needs_input`, never zero.
