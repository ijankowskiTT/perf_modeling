# Model

qperf evaluates the real schedule of one binary_ng Quasar-native program (`kernels_qsr/`, pinned in
`profiles/kernels/binary-ng-native.json`) against a cost profile.

**Scope**: no-broadcast tensor-tensor ops, TILE layout, each operand in DRAM, L1 (interleaved) or borrowed
(resident shard). Calibrated: `add`, `multiply` (bf16). Anything else returns `needs_input` or `unsupported`.

| Layer | File | Content |
|---|---|---|
| Rule | `qperf/rule.py` | the factory's default config, work split (mirror of the C++) |
| Validity | `qperf/constraints.py` | the factory's checks (threads, STRIDED ratio, DST, ring, L1 budget) |
| Traffic | `qperf/traffic.py` | exact counts of tiles, batches, barriers, NoC requests, bytes |
| Bounds | `qperf/bounds.py` | per-resource lower bounds |
| Pipeline | `qperf/pipeline.py` | fixed-schedule recurrence over the thread programs and rings |

**Pipeline**: reader, compute and writer threads run their kernels' own loops over the DFB's strided tile
counters with finite rings and a half-sync DST. Per batch: fixed cost, one request per tile, one barrier.
Barriers are cluster-wide, as in the firmware; timed transfers complete through one per-cluster queue.
Shared NoC, L1 and DRAM bandwidth are applied as bounds. Past 2048 tiles per cluster the result is extrapolated
exactly from the periodic steady state (`extrapolated: true`). A configuration that cannot progress is invalid.

**Not modeled**: NoC routing and link contention, L1 bank conflicts, dynamic arbitration, launch skew.

**Costs**: every parameter in `profiles/costs/*.json` has `value`, `low`, `high`, `class` and `source`.
`low`/`high` move all costs together (scenarios, shared by baseline and candidates), not confidence intervals.
A missing cost is `needs_input`, never zero.
