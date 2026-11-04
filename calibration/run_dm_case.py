# One simulator run of one Quasar data-movement op; prints one JSON line prefixed "JSON ".
# Usage (from a worktree, simulator env sourced): python run_dm_case.py <op> <dtype> <H> <W> <mem> <multicore>
#   op: tilize | untilize | tilize_with_val_padding | untilize_with_unpadding | interleaved_to_sharded |
#       sharded_to_interleaved | move
#   mem: dram | l1 (interleaved, input and output) | hs (height-sharded over the grid, L1; tilize/untilize only)
#        sharding ops: <hs|ws|bs><cores>[d|l], e.g. hs8 = height-sharded over 8 cores (L1), interleaved side in DRAM
#        (suffix l: interleaved side in L1); bs needs a square core count (2x2, 3x3, ...); a trailing r = ROW_MAJOR
#        tensors (e.g. hs8r, ws4lr): shard shape ceil(H/n) x W etc. in elements, the last shard may be partial
#   multicore: 1 | 0 (use_multicore)
# The record keeps every core's span (the op ends with its slowest core) and the kernel files the program used,
# so the factory is read from the program, not inferred.
import os, sys
import torch, ttnn

op, dtype, H, W, mem, mc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6] == "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_common import profiler_log, report  # noqa: E402

prof = profiler_log()
dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[dtype]
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(H * 7 + W)
    shape = (1, 1, H, W)
    x = torch.rand(shape, dtype=torch.float32).to({"bf16": torch.bfloat16, "fp32": torch.float32}[dtype])
    if mem == "hs":
        g = device.compute_with_storage_grid_size()
        ncores = min(g.x * g.y, H // 32)
        grid = ttnn.num_cores_to_corerangeset(ncores, g, row_wise=True)
        memcfg = ttnn.create_sharded_memory_config((H // ncores, W), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
                                                   use_height_and_width_as_shard_shape=True)
    elif mem in ("dram", "l1"):
        memcfg = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[mem]
    Q = ttnn.experimental.quasar
    if op in ("interleaved_to_sharded", "sharded_to_interleaved"):
        kind, rest = mem[:2], mem[2:]
        rm = rest.endswith("r")  # ROW_MAJOR tensors: shard shapes in elements, uneven last shards allowed
        rest = rest.rstrip("r")
        side = ttnn.L1_MEMORY_CONFIG if rest.endswith("l") else ttnn.DRAM_MEMORY_CONFIG
        n = int(rest.rstrip("dl"))
        g = device.compute_with_storage_grid_size()
        div = (lambda a, b: -(-a // b)) if rm else (lambda a, b: a // b)  # ROW_MAJOR: ceil, the last shard partial
        # ROW_MAJOR shard rows must be whole 16 B (the ops' L1 page alignment check): round a split width up
        wdiv = (lambda a, b: -(-div(a, b) * x.element_size() // 16) * 16 // x.element_size()) if rm else div
        if kind == "bs":
            k = int(round(n ** 0.5))
            grid = ttnn.CoreGrid(y=k, x=k)
            strategy, shard = ttnn.ShardStrategy.BLOCK, (div(H, k), wdiv(W, k))
        else:
            grid = ttnn.num_cores_to_corerangeset(n, g, row_wise=True)
            strategy = ttnn.ShardStrategy.HEIGHT if kind == "hs" else ttnn.ShardStrategy.WIDTH
            shard = (div(H, n), W) if kind == "hs" else (H, wdiv(W, n))
        shcfg = ttnn.create_sharded_memory_config(shard, core_grid=grid, strategy=strategy,
                                                  use_height_and_width_as_shard_shape=True)
        lay = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
        if op == "interleaved_to_sharded":
            t = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=side)
            out = Q.interleaved_to_sharded(t, shcfg)
        else:
            t = ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=shcfg)
            out = Q.sharded_to_interleaved(t, side)
        got = ttnn.to_torch(out)[..., :H, :W].float()
    elif op == "move":  # TILE, interleaved, same memory config in and out (non-overlapping buffers: page copy)
        t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
        out = Q.move(t, memory_config=memcfg)
        got = ttnn.to_torch(out).float()
    elif op in ("tilize", "tilize_with_val_padding"):
        t = ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memcfg)
        if op == "tilize":
            out = Q.tilize(t, memory_config=memcfg, dtype=dt, use_multicore=mc)
        else:
            out = Q.tilize_with_val_padding(t, [1, 1, -(-H // 32) * 32, -(-W // 32) * 32], 0.0, memory_config=memcfg,
                                            use_multicore=mc)
        got = ttnn.to_torch(out)[..., :H, :W].float()
    else:
        t = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
        if op == "untilize":
            out = Q.untilize(t, memory_config=memcfg, use_multicore=mc)
        else:
            out = Q.untilize_with_unpadding(t, [0, 0, H - 1, W - 1], memory_config=memcfg, use_multicore=mc)
        got = ttnn.to_torch(out).float()
    max_err = (got - x.float()).abs().max().item()
    ttnn.ReadDeviceProfiler(device)
finally:
    ttnn.close_device(device)

report(op, dtype, H, W, mem, mc, max_err, prof)
