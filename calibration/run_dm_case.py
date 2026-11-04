# One simulator run of one Quasar data-movement op; prints one JSON line prefixed "JSON ".
# Usage (from a worktree, simulator env sourced): python run_dm_case.py <op> <dtype> <H> <W> <mem> <multicore>
#   op: tilize | untilize | tilize_with_val_padding | untilize_with_unpadding
#   mem: dram | l1 (interleaved, input and output) | hs (height-sharded over the grid, L1; tilize/untilize only)
#   multicore: 1 | 0 (use_multicore)
# The record keeps every core's span (the op ends with its slowest core) and the kernel files the program used,
# so the factory is read from the program, not inferred.
import collections, csv, json, os, pathlib, re, statistics, sys
import torch, ttnn

op, dtype, H, W, mem, mc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6] == "1"
prof = pathlib.Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"
if prof.exists():
    prof.unlink()
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
    else:
        memcfg = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[mem]
    Q = ttnn.experimental.quasar
    if op in ("tilize", "tilize_with_val_padding"):
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

rows = list(csv.DictReader(prof.read_text().splitlines()[1:], skipinitialspace=True))
start = collections.defaultdict(lambda: 1 << 62)
end = collections.defaultdict(int)
for r in rows:
    if "KERNEL" not in r["zone name"].strip():
        continue
    key = ((int(r["core_x"]), int(r["core_y"])), r["RISC processor type"].strip())
    t = int(r["time[cycles since reset]"])
    if r["type"].strip() == "ZONE_START":
        start[key] = min(start[key], t)
    else:
        end[key] = max(end[key], t)
cores = {}
for core in sorted({k[0] for k in start}):
    keys = [k for k in start if k[0] == core and k in end]
    t0 = min(start[k] for k in keys)
    cores[core] = {"span": max(end[k] for k in keys) - t0,
                   "roles": {k[1]: [start[k] - t0, end[k] - t0] for k in sorted(keys)}}
spans = [c["span"] for c in cores.values()]
kernels = (pathlib.Path(os.environ["TT_METAL_LOGS_PATH"]) / "generated" / "inspector" / "kernels.yaml").read_text()
files = sorted({os.path.basename(m) for m in re.findall(r"[\w/.-]+\.cpp", kernels)})
print("JSON " + json.dumps({
    "op": op, "dtype": dtype, "H": H, "W": W, "mem": mem, "multicore": mc, "cores": len(spans),
    "span_max": max(spans), "span_med": statistics.median(spans), "span_min": min(spans),
    "spans": sorted(spans), "kernels": files, "max_abs_err": max_err,
    "core_detail": {f"{c[0]},{c[1]}": v for c, v in list(cores.items())[:4]}}))
