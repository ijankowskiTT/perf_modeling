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
reader batching barriers every `thr` reads as the kernel does. ROW_MAJOR: one request per shard row
(aligned path), or a transaction-ID scratch read plus a local loopback per row with 4 in flight (unaligned DRAM
source); uneven last shards clipped.

**move** (`qperf/ops/move.py`, registered in `qperf/ops/registry.py`): interleaved DRAM or L1, TILE or ROW_MAJOR,
the overlapping-output path (MoveOverlap), which is the only interleaved path Quasar runs: a non-overlapping output
selects MoveProgramFactory, whose legacy copy kernels Quasar rejects (`DataMovementKernel is not supported on
Quasar`), and an output at the input's address runs no program. Pages split by `split_work_to_cores`; per core the
reader reads page by page (one barrier each), then all cores handshake through the controller core (one semaphore
multicast per region), and only then the writer writes. The controller sets the span; the slowest other core's reads
are evaluated alongside it. On both simulators a page read costs the same for any page size, writes are nearly free;
on ttsim-perf L1 reads of 32 cores cost ~6x DRAM reads. Sharded move is not modeled yet.

**reshard** (`qperf/ops/reshard.py`): sharded to sharded, TILE or ROW_MAJOR, height / width / block, L1 or the 2
DRAM banks on either side. Factory selection mirrors `ReshardDeviceOperation::select_program_factory`:
HEIGHT->HEIGHT SameWidth (the output cores read) or SameWidthWrite (DRAM output: the input cores write); ROW_MAJOR
WIDTH->WIDTH SameHeight / SameHeightWrite; other L1-output cases Generic (each output core gathers strided page
runs, grouped as `create_map_for_reshard` does); DRAM->DRAM CopyPages (all 32 cores, page by page through a 1-entry
DFB). Both DM kernels of a core run the same source on their half of the segments, one request per segment (per row
for SameHeight, per stride run for Generic) and one barrier at the end. craq-sim charges a transfer per request, not
per byte (answers on it say so and comparisons are inconclusive); ttsim-perf does not time DRAM (same). On
ttsim-perf L1 reads are paced by NoC contention: each core's NoC port and each directed link moves 64 B/cycle, and a
core's reads end when the busiest port or link on its response paths has moved its bytes. Not modeled: CopyLocal (L1
-> DRAM with a layout change; its output is wrong on both simulators), Generic from a DRAM input (neither simulator
runs it), unaligned HEIGHT->HEIGHT rows, row-major width changes.

**fp32** runs the same kernels and loops: tiles are 4096 B and rows twice as many bytes, the compute unpacks straight
to DEST (`UnpackToDest`) with fp32 DEST, so untilize sub-blocks hold <= 4 tiles (half of bf16's 8), and tilize uses
the standard (lossless) path. Request counts are unchanged; the fp32 runs measured that craq-sim's DM time per
request barely grows with bytes (fitted per-byte terms ~half of bf16's) while ttsim-perf times the extra bytes, and
that compute per tile differs (ttsim-perf: ~1.8x slower per tile for tilize). A factory's `<p>.f32_<term>` overrides
`<p>.<term>` for fp32 (`dtype_costs`): compute terms, per-byte terms and kernel starts; every other term is the bf16
fit. Sharding moves with the interleaved side in L1 also pay a bank-conflict term: cores that start their shards on
the same L1 bank (pages round-robin over the 32 banks) walk them in lockstep and queue there (`<p>.l1_conf_cpb` per
other core on the bank; fitted on fp32 runs only, 0 for bf16).

**Pipeline**: reader, compute and writer threads run their kernels' own loops over the DFB's strided tile
counters with finite rings and a half-sync DST. Per batch: fixed cost, one request per tile, one barrier.
Barriers are cluster-wide, as in the firmware; timed transfers complete through one per-cluster queue.
Shared NoC, L1 and DRAM bandwidth are applied as bounds. Past 2048 tiles per cluster the result is extrapolated
exactly from the periodic steady state (`extrapolated: true`). A configuration that cannot progress is invalid.

**Hypothetical variants** (not in the kernels; results say so): `barriers: per_thread` gives each DM thread a
barrier on its own transfers only (transaction IDs); `work_split: dynamic` lets the DM threads of a role claim
batches from one shared queue, paying `dq.claim` (assumed 30 cycles) per batch. The completion queue stays shared
and serves transfers in time order, so variants and the pinned schedule are evaluated the same way.

**Not modeled**: NoC routing and link contention (except reshard on ttsim-perf), L1 bank conflicts (except the
sharding moves' lockstep conflict above), dynamic arbitration, launch skew.

**Costs**: every parameter in `profiles/costs/*.json` has `value`, `low`, `high`, `class` and `source`.
`low`/`high` move all costs together (scenarios, shared by baseline and candidates), not confidence intervals.
A missing cost is `needs_input`, never zero.
