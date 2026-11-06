# Phase 2 design: from a few ops to all of Quasar

Status: proposal for review. Owner: qperf. Date: 2026-11-06.

## Goal

qperf should estimate any Quasar op, and whole models, at the accuracy it has today (1-3% against the simulators),
with a new op costing days instead of weeks.

Non-goals: replacing the simulators, and silicon accuracy before silicon exists.

## Why today's approach does not scale

Each factory is mirrored by hand: its work split, each kernel's loops, and its own cost terms (prefixes `tz`, `utz`,
`rsw` and so on). The results are good, about 1% after validation, but the cost is high:
- about 1-3 weeks per op;
- terms fitted per factory never transfer to the next op;
- Quasar matmul alone has 7 factories (the two multicast ones are ~13k lines).

## Proposal

### A. Primitive cost layer

Cost the steps all ops share once per profile, instead of once per factory:

| Primitive | Cost terms |
|---|---|
| NoC read / write request | issue, per byte, latency per placement, completion queue (exists in the engine) |
| Barrier, flush | fixed |
| DFB reserve / push / wait / pop | fixed, credit latency |
| Unpack, math, pack per tile | per LLK op and dtype, plus a fixed cost per step |
| DEST section handoff | fixed |
| Kernel prologue / epilogue | per kernel source (see risk 3) |

An op then differs only in its LLK compute op (add, matmul tile, tilize block, ...) and its schedule (B).

**Feasibility test (M1):** refit all existing calibration data (~4,500 runs, every op) with shared primitive terms.
Accept if each factory stays within +0.5 pt of its current error.

### B. Kernel templates, composed from the program spec

- **Templates:** model each kernel source file once. A template is a schedule over the kernel's runtime and
  compile-time args: its loops, DFB operations and NoC operations, expressed in the engine's primitives (Thread,
  DMBatch, Ring). Kernels are shared across factories and ops (e.g. `reader_unary_*`, `writer_unary_interleaved_*`,
  `reader_bmm_tile_layout_*`), so the library grows slower than the op list.
- **Composition:** the Metal2 `ProgramSpec` and `ProgramRunArgs` say which kernels run on which cores, with which
  args, DFB sizes and tensor bindings. qperf builds the engine program from them, so a new factory whose kernels are
  in the library needs no qperf code.
- **Capture:** `ProgramSpec` has no serializer. Add a debug-only JSON dump in tt-metal: an env var at program
  creation writes the spec and run args. One craq-sim run of the op then gives qperf its structure. Fallback if the
  hook is not accepted: keep a Python mirror of the factory's work split, as today, but compose it from templates.
- **Checking a template:** profiler zones on craq-sim (DFB push / pop timestamps per thread) are compared with the
  template's event order automatically. `calibration/dm_common.py` already reads the profiler log.

### C. Whole-model estimates

- `ttnn.graph.begin_graph_capture` gives the op sequence with shapes, dtypes and memory configs.
- qperf estimates each op it supports, lists the ones it does not (never a guessed number), and adds a per-op
  dispatch term.
- Output: total time, a ranked table of where the time goes, the layout changes between ops, and the unsupported ops
  with their share unknown.
- The question it answers: "which op should we optimise next for model X".

## Matmul first

1. **Unified factory:** Quasar's placement-first Metal 2.0 matmul, one program for every placement (~475 lines).
   Template its reader, writer and `unified_matmul_compute` kernels.
2. **Campaign:** M, K, N sweeps, interleaved and sharded placements, bf16 and fp32, on both simulators, every 4th run
   held out.
3. **Then:** the multicast 1D / 2D factories, if Quasar selects them for real workloads. Their sender / receiver
   kernels need semaphore and multicast primitives in the engine.

## Milestones

| | Work | Effort | Done when |
|---|---|---|---|
| M1 | Primitive cost layer, refit on all existing data | 1 week | every factory within +0.5 pt |
| M2 | Spec capture hook (or the fallback), kernel template library for today's kernels | 1-2 weeks | current results reproduced through templates |
| M3 | Matmul unified: templates, campaign, fit, API, survey | 2 weeks | <= 3% mean on both simulators |
| M4 | Whole-model estimate from graph capture, on one target model | 1 week | per-op table for that model |
| M5 | softmax, reduction, transpose on templates | 1 week each, less as the library grows | <= 3% each |

## Risks

1. **The hook may not be accepted upstream.** Mitigation: a local patch on the calibration tree is enough for
   capture, or the Python-mirror fallback.
2. **Complex kernels** (multicast, semaphores, ring all-gather) need new engine primitives. Mitigation: start with
   the unified matmul; add primitives only when a selected factory needs them.
3. **Primitive costs may not transfer.** RISC-V overhead per loop iteration differs per kernel. Mitigation: allow one
   overhead term per kernel source, fitted from a few runs. M1 measures how much is needed.
4. **Simulator ground truth.** It is unchanged by this design. DRAM stays a hypothesis until ttsim times Quasar DRAM.

## Decisions needed

1. The target model for M4 (e.g. Llama decode): it sets which ops come after matmul.
2. Whether a debug `ProgramSpec` dump can go into tt-metal, and who reviews it.
3. Whether Quasar workloads use the multicast matmul factories or only the unified one.
4. Long-term ownership of the kernel template library.
