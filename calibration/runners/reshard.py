# One simulator run of ttnn.experimental.quasar.reshard; prints one JSON line prefixed "JSON ".
# Usage (sweep.sh, simulator env sourced): python runners/reshard.py reshard <dtype> <H> <W> <mem> <knobs>
#   mem = <layout>_<from>_<to>: layout t (TILE) | r (ROW_MAJOR); from / to = <kind><cores><place>
#     kind  hs | ws (cores on the 8x4 grid row-wise, or DRAM banks 0..cores-1) | bs<rows>x<cols> (block, L1 only)
#     place l (L1) | d (DRAM, sharded over the 2 DRAM banks)
#   e.g. t_hs8l_hs4l: TILE, height-sharded over 8 L1 cores -> 4 L1 cores; r_ws2d_ws8l: row-major, width-sharded
#   over the 2 DRAM banks -> 8 L1 cores.
#   Shards are ceil-divided (TILE: whole tiles); the shard spec is the one qperf.ops.reshard.shard_spec computes.
import os
import sys

import torch
import ttnn

op, dtype, H, W, mem = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dm_common import profiler_log, report  # noqa: E402
from qperf.ops.reshard import parse_mem  # noqa: E402  (pure Python: the model's own shard specs)

prof = profiler_log()
dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[dtype]
spec = parse_mem(mem, H, W, {"bf16": 2, "fp32": 4}[dtype])
LAYOUT = {"TILE": ttnn.TILE_LAYOUT, "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT}[spec["layout"]]
STRAT = {"hs": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "ws": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
         "bs": ttnn.TensorMemoryLayout.BLOCK_SHARDED}


def memcfg(s):
    if s["place"] == "dram":
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(s["n"] - 1, 0))})
        btype = ttnn.BufferType.DRAM
    else:
        btype = ttnn.BufferType.L1
        if s["kind"] == "bs":
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(s["gx"] - 1, s["gy"] - 1))})
        else:
            grid = ttnn.num_cores_to_corerangeset(s["n"], ttnn.CoreCoord(8, 4), row_wise=True)
    shard = ttnn.ShardSpec(grid, [s["sh"], s["sw"]], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(STRAT[s["kind"]], btype, shard)


device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(H * 7 + W)
    x = torch.rand((1, 1, H, W), dtype=torch.float32).to({"bf16": torch.bfloat16, "fp32": torch.float32}[dtype])
    t = ttnn.from_torch(x, dtype=dt, layout=LAYOUT, device=device, memory_config=memcfg(spec["from"]))
    out = ttnn.experimental.quasar.reshard(t, memcfg(spec["to"]))
    got = ttnn.to_torch(out).float()[..., :H, :W]
    max_err = (got - x.float()).abs().max().item()
    ttnn.ReadDeviceProfiler(device)
finally:
    ttnn.close_device(device)

report(op, dtype, H, W, mem, True, max_err, prof)
