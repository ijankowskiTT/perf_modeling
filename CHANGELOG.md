# Changelog

## 1.0.0

- binary_ng eltwise (`add`, `multiply`): thread allocation, DM batch, tiles_per_cycle, ring depth, compute / DRAM / L1
  what-ifs, kernel-change variants (per-thread barriers, dynamic work queue). ~1-2% on craq-sim.
- Data movement on a generic engine (`qperf/engine.py`): tilize, untilize, tilize_with_val_padding,
  untilize_with_unpadding (Default / MultiCore / SingleCore / Sharded / Block factories), interleaved_to_sharded,
  sharded_to_interleaved (TILE and ROW_MAJOR), move (overlap path, the only one Quasar runs); bf16 and fp32.
  <= 2.5% mean per factory on craq-sim, <= 3% on ttsim-perf.
- binary_ng fp32 (add, multiply): ~1.4% held-out on craq-sim.
- Claude Code plugin: the `quasar-perf` skill delegates to a Haiku runner agent (~$0.15 per question); follow-ups keep
  context; HTML / markdown reports with charts.
- `check-drift --ref main | pr/N`: compares the mirrored kernel sources with GitHub; comment / include / formatting
  changes count as cosmetic.
