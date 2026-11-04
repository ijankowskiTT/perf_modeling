# Changelog

## 1.0.0

- **binary_ng** `add`, `multiply` (bf16, fp32). Covers thread allocation, DM batch, tiles_per_cycle, ring depth,
  compute / DRAM / L1 what-ifs and hypothetical kernel changes (per-thread barriers, dynamic work queue).
- **Data movement** (bf16, fp32):
  - tilize, untilize and their padding variants;
  - interleaved_to_sharded and sharded_to_interleaved (TILE, ROW_MAJOR);
  - move (overlap path);
  - reshard (SameWidth, SameHeight and their write variants, Generic, CopyPages).
- **Accuracy**: <= 2.5% mean per factory on craq-sim and <= 3.3% on ttsim-perf. Exceptions, flagged in answers:
  ttsim-perf reshard SameHeight / Generic, and binary_ng L1 (~8%).
- **Plugin**: the `quasar-perf` skill hands each question to a Haiku runner (~$0.14 per question). Follow-ups keep
  context. Reports with charts are optional.
- **`check-drift --ref main | pr/N`**: compares the pinned kernels with GitHub. Comment, include and formatting
  changes count as cosmetic.
