# Examples

These prompts were run through the plugin (qperf 1.5.0) and their answers checked against the model. Each one shows a
behaviour to rely on. Times are simulator cycles unless the answer says ns.

## Overviews

**Wins across the layout ops**
```
Use the quasar-perf skill: where are the biggest performance wins in Quasar's layout ops (tilize, untilize, sharding,
reshard)? Rank them by expected gain. Make a report with charts.
```
- Two separate rankings over 54 sizes.
  - Kernel changes: tilize 2D split + double buffer + batch 8 is best at 1.31x; untilize tops out at 1.06x.
  - Layout choices: reshard on 32 cores is 2.26x; resharding to another spec is 0.25–0.73x.
- The bottleneck per op, and the DRAM caps.
- Report: [layout ops](https://claude.ai/artifact/Sfj2kJZLYg65LfE2SsTzfC?sk=id4mHCl0N8wNsmEaQi3QMw) (Tenstorrent only).

**Full picture of one op**
```
Use the quasar-perf skill: give me the full picture of tilize on Quasar. How does it run today, which multicore
approaches are worth it across tensor sizes, where does each regress, and what limits it? Make a report with charts.
```
- 8 approaches over 54 sizes, a heatmap for each, and the cores used per size.
- On assumed DRAM, large tensors are DRAM-bound.
- Report: [tilize survey](https://claude.ai/artifact/VJrBHzpCKU5mwCyVeJLk7P?sk=ubpNNdqu4o1YJkd7hUFmlA) (Tenstorrent only).

**No size given**
```
Use the quasar-perf skill: is untilize on Quasar fast enough, or should we change it?
```
- Surveys all sizes itself.
- Untilize uses 1 core at H=32 and all 32 from H=1024.
- The best change is 1.06x, and only at heights up to 512. Half-size blocks regress on 45 of 49 sizes. Verdict: don't rework it.

**fp32 vs bf16**
```
Use the quasar-perf skill: across tensor sizes, do the multicore approaches for untilize help fp32 as much as bf16?
```
- The factory's multicore gains more in fp32 (4.8x vs 4.2x over single core): fp32 is compute-bound on one core.
- The proposed variants gain about the same in both.

## Honest limits

**Part of the question is unsupported**
```
Use the quasar-perf skill: compare how long a 1024x1024 bf16 matmul and a tilize of the same tensor take on Quasar.
```
- Gives the tilize time (4,960 cycles, 1.2% error) and states that matmul isn't modeled. No number is guessed.

**Invalid request**
```
Use the quasar-perf skill: how long does resharding a 1024x1024 bf16 tensor from 8 height shards in L1 to 4 height
shards in DRAM take?
```
- Rejects it: Quasar has 2 DRAM banks.
- Answers the closest valid case (2 banks, SameWidthWrite, ~1,300 cycles) and warns that the DRAM side isn't timed.

**Outside the calibrated range**
```
Use the quasar-perf skill: how long does tilize of a 16384x8192 fp32 tensor from L1 take on Quasar?
```
- 1.63M cycles, NoC-bound.
- Flagged as extrapolated: 134M elements against a calibrated maximum of 8.4M.

**Kernel idea the model doesn't have**
```
Use the quasar-perf skill: would fusing tilize into the following binary add kernel speed things up?
```
- Says fusion isn't modeled, then bounds it from the separate ops.
- At 1024x1024, tilize costs 11,714 cycles and the add 2,543, so fusion saves at most ~82%. Labelled an upper bound.

**Placement that can't fit**
```
Use the quasar-perf skill: for a 1M-tile bf16 multiply with a in L1 and b in DRAM, which matters most: thread counts,
dm_batch or tiles_per_cycle?
```
- Catches that `a` doesn't fit in L1.
- Answers the closest setup that runs: all in DRAM, DRAM-bound, none of the three knobs matter.

## Trust and code

**How confident is it**
```
Use the quasar-perf skill: how confident should I be that batching tilize's writes 8 at a time actually helps? Has it
been checked?
```
- write_batch was validated: 24 simulator runs over 12 sizes, 0.8% error.
- Batch 8 alone gains only ~4% on small tensors.
- The bigger gains come from block size and buffering, which are unvalidated, and it says so.

**Is the model current with main**
```
Use the quasar-perf skill: has tilize or untilize changed on tt-metal main since the model was pinned?
```
- Runs `check-drift` against GitHub main: no file of the layout kernel set has changed. It names the pinned commit
  and main's commit.

## What-ifs and plans

**Hardware what-if**
```
Use the quasar-perf skill: if Quasar's DRAM bandwidth were twice what you assume, how much would batched writes and
double buffering help tilize of an 8192x2048 tensor?
```
- At the assumed bandwidth the op is DRAM-bound and nothing helps (1.00x).
- At 2x bandwidth the writer becomes the limit: either change gives 1.20x, and combining them adds nothing.

**Reshard plan**
```
Use the quasar-perf skill: for a 2048x2048 bf16 tensor height-sharded over 8 cores in L1, what is the cheapest reshard
target among 4, 16 and 32 height shards, and which factory runs?
```
- All three run SameWidth. 16 and 32 tie at ~27,400 cycles; 4 shards is 2.4x slower.
- The shape is flagged as extrapolated.
