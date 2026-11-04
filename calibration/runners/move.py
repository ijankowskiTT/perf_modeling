# One simulator run of ttnn.experimental.quasar.move; prints one JSON line prefixed "JSON " (calibration/dm_common.py).
# Usage (sweep.sh calls it): python move.py move <dtype> <H> <W> <mem> <multicore>
#   mem = <dram|l1>[_ov][_rm]: interleaved buffer type (same in and out); _ov = overlapping output (MoveOverlap
#   factory), else the output lands in a freed same-size dummy (MoveProgramFactory: the copy SameMemoryConfig page
#   copy); _rm = ROW_MAJOR (pages are rows), else TILE. multicore is ignored (move always splits over the grid).
# move frees its input before allocating the output: without a freed dummy below (DRAM, bottom-up) or above (L1,
# top-down) the input, the output gets the input's address and move returns without running a program.
import os
import sys

import torch
import ttnn

op, dtype, H, W, mem = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dm_common import profiler_log, report  # noqa: E402

parts = mem.split("_")
buf, overlap, rm = parts[0], "ov" in parts, "rm" in parts
prof = profiler_log()
dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[dtype]
tdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
layout = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
memcfg = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[buf]
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(H * 7 + W)
    x = torch.rand((1, 1, H, W), dtype=torch.float32).to(tdt)
    # overlap: a one-page ROW_MAJOR dummy (as tests/.../test_move.py TINY_DUMMY), so the output starts just below
    # (DRAM) / above (L1) the input and overlaps it; else a dummy of the input's size, so the output does not overlap
    dshape, dlayout = ((1, 1, 1, 8), ttnn.ROW_MAJOR_LAYOUT) if overlap else ((1, 1, H, W), layout)
    dummy = ttnn.from_torch(torch.zeros(dshape, dtype=tdt), dtype=dt, layout=dlayout, device=device, memory_config=memcfg)
    t = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=memcfg)
    in_addr = t.buffer_address()
    ttnn.deallocate(dummy)
    out = ttnn.experimental.quasar.move(t, memory_config=memcfg)
    out_addr = out.buffer_address()
    got = ttnn.to_torch(out).float()
    max_err = (got - x.float()).abs().max().item()
    ttnn.ReadDeviceProfiler(device)
finally:
    ttnn.close_device(device)

report(op, dtype, H, W, mem, True, max_err, prof, extra={"in_addr": in_addr, "out_addr": out_addr})
