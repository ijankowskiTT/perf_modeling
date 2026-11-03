# One simulator run of one binary_ng op on 32 clusters; prints one JSON line prefixed "JSON ".
# Usage (from a worktree, simulator env sourced): python run_case.py <op> <dtype> <tpc> <layout>
#   layout: ddd = a, b, out DRAM-interleaved (NoC path both ways, no anchor)
#           lll = L1-interleaved with 32*tpc+1 tiles, so no bank slice divides and nothing is borrowed (NoC path)
#           hhh = height-sharded over all 32 clusters, tpc tiles each: everything borrowed (compute only)
#           any other mix of d/l per operand (a, b, out), e.g. lld or ddl: the lll shape, each operand in its own
#           memory, to separate L1-NoC read and write costs
# Knobs come from the environment (TTNN_QSR_*); the record keeps them and the factory's own config log lines,
# so the routing is read from the program, not inferred.
import collections, csv, json, os, pathlib, statistics, sys
import torch, ttnn

op, dtype, tpc, layout = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prof = pathlib.Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"
if prof.exists():
    prof.unlink()
dt = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}[dtype]
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(tpc)
    g = device.compute_with_storage_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (g.x - 1, g.y - 1))})
    if layout == "hhh":
        shape = (1, 1, 32 * 32, 32 * tpc)
        mem = ttnn.create_sharded_memory_config((32, 32 * tpc), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
                                                use_height_and_width_as_shard_shape=True)
    elif layout == "lll":
        shape, mem = (1, 1, 32, 32 * (tpc * 32 + 1)), ttnn.L1_MEMORY_CONFIG
    elif layout == "ddd":
        shape, mem = (1, 1, 1024, 32 * tpc), ttnn.DRAM_MEMORY_CONFIG
    elif len(layout) == 3 and set(layout) <= {"d", "l"}:
        shape, mem = (1, 1, 32, 32 * (tpc * 32 + 1)), None
    else:
        raise SystemExit(f"unknown layout {layout}")
    mems = [mem or {"d": ttnn.DRAM_MEMORY_CONFIG, "l": ttnn.L1_MEMORY_CONFIG}[x] for x in layout]
    cfg = dict(dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.rand(*shape) * 2 + 0.5, memory_config=mems[0], **cfg)
    b = ttnn.from_torch(torch.rand(*shape) * 2 + 0.5, memory_config=mems[1], **cfg)
    out = getattr(ttnn.experimental.quasar, op)(a, b, memory_config=mems[2])
    got = ttnn.to_torch(out).float()
    ref = {"add": torch.add, "multiply": torch.mul, "subtract": torch.sub}[op](
        ttnn.to_torch(a).float(), ttnn.to_torch(b).float())
    max_err = (got - ref).abs().max().item()
    ttnn.ReadDeviceProfiler(device)
finally:
    ttnn.close_device(device)

rows = list(csv.DictReader(prof.read_text().splitlines()[1:], skipinitialspace=True))
start = collections.defaultdict(lambda: 1 << 62)
end = collections.defaultdict(int)
for r in rows:
    if "KERNEL" not in r["zone name"].strip():
        continue
    key = ((r["core_x"].strip(), r["core_y"].strip()), r["RISC processor type"].strip())
    t = int(r["time[cycles since reset]"])
    if r["type"].strip() == "ZONE_START":
        start[key] = min(start[key], t)
    else:
        end[key] = max(end[key], t)
spans, firsts, lasts = [], [], []
roles = collections.defaultdict(lambda: ([], []))
for core in sorted({k[0] for k in start}):
    keys = [k for k in start if k[0] == core and k in end]
    t0 = min(start[k] for k in keys)
    t1 = max(end[k] for k in keys)
    spans.append(t1 - t0)
    firsts.append(t0)
    lasts.append(t1)
    for k in keys:
        roles[k[1]][0].append(start[k] - t0)
        roles[k[1]][1].append(end[k] - t0)
role_times = {r: [statistics.median(s), statistics.median(e)] for r, (s, e) in sorted(roles.items())}
zone_tot = collections.defaultdict(list)
for r in rows:
    if r["type"].strip() == "ZONE_TOTAL":
        zone_tot[r["zone name"].strip() + "@" + r["RISC processor type"].strip()].append(int(r["data"]))
zones = {k: statistics.median(v) for k, v in sorted(zone_tot.items())}
kernels = (pathlib.Path(os.environ["TT_METAL_LOGS_PATH"]) / "generated" / "inspector" / "kernels.yaml").read_text()
tree = "qsr" if "kernels_qsr/" in kernels else "dfb" if "kernels_dfb/" in kernels else "other"
knobs = {k: v for k, v in os.environ.items() if k.startswith("TTNN_QSR_")}
print("JSON " + json.dumps({
    "layout": layout, "op": op, "dtype": dtype, "tpc": tpc, "clusters": len(spans), "tree": tree, "knobs": knobs,
    "span_med": statistics.median(spans), "span_max": max(spans), "span_min": min(spans),
    "op_span": max(lasts) - min(firsts), "max_abs_err": max_err, "roles": role_times, "zones": zones}))
