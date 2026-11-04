# Model

qperf evaluates the real schedule of one Quasar program (today binary_ng native, `kernels_qsr/`, pinned in
`profiles/kernels/binary-ng-native.json`) against a cost profile. Each op is a description in `qperf/ops/`
that builds threads and rings; one generic engine evaluates them.

**Scope**: no-broadcast tensor-tensor ops, TILE layout, each operand in DRAM, L1 (interleaved) or borrowed
(resident shard). Calibrated: `add`, `multiply` (bf16). Anything else returns `needs_input` or `unsupported`.

| Layer | File | Content |
|---|---|---|
| Rule | `qperf/rule.py` | the factory's default config, work split (mirror of the C++) |
| Validity | `qperf/constraints.py` | the factory's checks (threads, STRIDED ratio, DST, ring, L1 budget) |
| Traffic | `qperf/traffic.py` | exact counts of tiles, batches, barriers, NoC requests, bytes |
| Bounds | `qperf/bounds.py` | per-resource lower bounds |
| Op description | `qperf/ops/binary_ng.py` | the kernels' thread programs: DM batches, compute chunks, rings |
| Engine | `qperf/engine.py` | generic fixed-schedule recurrence over any op description |

**tilize / untilize** (`qperf/ops/layout.py`): one reader, compute and writer thread per core, work split by
tile-rows (`split_blocks_for_tilize`), the factory chosen as the device op does. Rules the simulator runs showed:
each DM thread's DMA moves one transfer at a time at ~56 B/cycle (craq-sim); a DM kernel has a prologue before its
first DFB wait; a producer kernel exits only once its DFB is drained; untilize's unpack and pack TRISCs run as two
stages over DEST sections (sub-blocks of <= 8 tiles), tilize's as one step per tile-row. On ttsim-perf, L1 reads and writes of a
core share one completion queue and a barrier waits for either kind; many cores moving L1 data at once are bound by
a global network bandwidth. The padding variants model the pad fill (DMA zero-fill, or a CPU store loop with a fixed
cost per padded row); untilize Block / BlockInterleaved split the tensor into S x S-tile squares as the device op does.
interleaved_to_sharded / sharded_to_interleaved: only the interleaved side moves data (the L1 shard is borrowed), the
reader batching barriers every `thr` reads as the kernel does.

**Pipeline**: reader, compute and writer threads run their kernels' own loops over the DFB's strided tile
counters with finite rings and a half-sync DST. Per batch: fixed cost, one request per tile, one barrier.
Barriers are cluster-wide, as in the firmware; timed transfers complete through one per-cluster queue.
Shared NoC, L1 and DRAM bandwidth are applied as bounds. Past 2048 tiles per cluster the result is extrapolated
exactly from the periodic steady state (`extrapolated: true`). A configuration that cannot progress is invalid.

**Hypothetical variants** (not in the kernels; results say so): `barriers: per_thread` gives each DM thread a
barrier on its own transfers only (transaction IDs); `work_split: dynamic` lets the DM threads of a role claim
batches from one shared queue, paying `dq.claim` (assumed 30 cycles) per batch. The completion queue stays shared
and serves transfers in time order, so variants and the pinned schedule are evaluated the same way.

**Not modeled**: NoC routing and link contention, L1 bank conflicts, dynamic arbitration, launch skew.

**Costs**: every parameter in `profiles/costs/*.json` has `value`, `low`, `high`, `class` and `source`.
`low`/`high` move all costs together (scenarios, shared by baseline and candidates), not confidence intervals.
A missing cost is `needs_input`, never zero.
