"""Quasar tilize / untilize (ttnn.experimental.quasar) as engine Programs, one core each.

Every factory runs one reader, one compute and one writer thread per core (no TTNN_QSR knobs). Work is split by
split_blocks_for_tilize: a block is one tile-row (32 sticks x the full width, Wt tiles); `per_core` blocks per core,
the last core may get fewer. Mirrored kernels (calibration tree 0fff850f3c6, unchanged on main f3ef8f22aaa):

tilize Default   reader_unary_stick_layout_split_rows_multicore: per block reserve(Wt), 32 stick reads of W*elem bytes,
                 1 barrier, push(Wt); in/out DFBs hold Wt entries (single-buffered); compute tilize_block(Wt);
                 writer_unary_interleaved_start_id: per tile wait(1), 1 write, flush, pop(1).
tilize SingleCore same, one core; blocks are tiles_per_block wide (the row or its largest divisor that fits L1).
tilize Sharded   both DFBs borrow the shard: no NoC traffic, compute only.
untilize MultiCore reader_unary_start_id_metal2: per tile reserve(1), 1 read, barrier, push(1); DFBs hold Wt entries
                 (2*Wt with more than one block per core); compute untilize(Wt);
                 writer ..._split_rows_multi_core_metal2: per block wait(Wt), 32 row writes, 1 barrier, pop(Wt).
untilize SingleCore same reader; writer ..._split_rows_single_core: per block 32 row writes, 1 barrier.

Cost keys (cycles; profiles/costs/*.json), one prefix per factory because each runs its own kernels:
<p>.start_{reader,compute,writer}, <p>.rd_{setup,fixed,issue,first,tail}, tilize <p>.cu_{fixed,per_tile} per block; untilize <p>.um_{fixed,per_tile} (unpack+math per DEST section) and
<p>.pk_{fixed,per_tile} (pack); <p>.cu_{first,tail}, <p>.pk_offset,
<p>.wr_{setup,fixed,issue,first,tail}, <p>.{rd,wr}_cpb (cycles per byte moved), <p>.drain; p = tz (tilize Default), tzs (tilize SingleCore), tzh (tilize Sharded), utz (untilize MultiCore),
utzs (untilize SingleCore). *_setup is a DM kernel's prologue before its first DFB wait, *_tail its cost after the last batch.
"""

from __future__ import annotations

import math

from ..engine import INF, DMBatch, Group, Program, Ring, Server, ThreadStats, Thread, TransferLog

TILE = 32


def sub_block_width(wt, elem):
    """untilize_helpers: a block wider than the DEST limit (8 tiles bf16, 4 with fp32) is untilized in sub-blocks of the
    largest divisor of wt that fits."""
    limit = 8 if elem == 2 else 4
    return wt if wt <= limit else max(d for d in range(1, limit + 1) if wt % d == 0)


def split_blocks(ncores_max, nblocks):
    """split_blocks_for_tilize: (per_core, ncores, cliff blocks on the last core or 0)."""
    per_core = math.ceil(nblocks / ncores_max)
    ncores = math.ceil(nblocks / per_core)
    return per_core, ncores, nblocks % per_core


def _noc(c, p, kind, place, nbytes):
    """(latency, gap, svc, timed) of one request of nbytes to/from `place` (dram | l1). gap = nbytes x the factory's
    data-movement cycles per byte (<p>.rd_cpb / <p>.wr_cpb: the DM thread moves ~56 B/cycle on craq-sim), plus, for L1
    on a profile that times L1 NoC transfers, <p>.{rd,wr}_cpb_l1 extra cycles per byte and <p>.{rd,wr}_lat_l1. The
    op's own terms are used, not binary_ng's per-tile NoC costs: these kernels move rows and blocks."""
    d = "rd" if kind == "read" else "wr"
    gap = nbytes * c.get(f"{p}.{d}_cpb", 0.0)
    lat = 0.0
    if place == "l1":
        gap += nbytes * c.get(f"{p}.{d}_cpb_l1", 0.0)
        lat = c.get(f"{p}.{d}_lat_l1", 0.0)
    return lat, gap, 0.0, bool(lat or gap)


def build(op, cfg, c):
    """One core of a tilize/untilize factory. cfg: blocks (on this core), wt (tiles per block), elem (bytes),
    row_bytes (bytes per stick request), in_place / out_place (dram | l1 | borrowed), factory."""
    p = PREFIX[(op, cfg["factory"])]
    fam = FAMILY[op]  # tilize or untilize: which compute and base DM loops the op's kernels share
    nb, wt, elem = cfg["blocks"], cfg["wt"], cfg["elem"]
    rows = cfg.get("rows") or [TILE] * nb  # valid rows per block (the last tile-row of a padded tensor is mixed)
    tile_bytes = TILE * TILE * elem
    n = nb * wt
    borrowed = cfg["in_place"] == "borrowed"
    cap_in = cap_out = cfg.get("cap") or wt
    if op == "untilize" and nb > 1 and not cfg.get("cap"):
        cap_in = cap_out = 2 * wt
    rin = Ring("in", 1, n + 1 if borrowed else cap_in, n)
    rout = Ring("out", 1, n + 1 if borrowed else cap_out, n)
    logs, lat_cr = [TransferLog(), TransferLog()], c.get(f"{p}.credit_latency", 0.0)  # own terms only, not binary_ng's
    blocks = [list(range(b * wt, (b + 1) * wt)) for b in range(nb)]
    st_r = ThreadStats("reader", 0, start=c[f"{p}.start_reader"])
    st_c = ThreadStats("compute", 0, start=c[f"{p}.start_compute"])
    st_w = ThreadStats("writer", 0, start=c[f"{p}.start_writer"])

    if borrowed:  # the shard backs both DFBs: publish all, compute, pop all
        rd = [("busy", c.get(f"{p}.rd_fixed", 0.0), True), ("push", rin, range(n), 0.0)]
        wr = [("wait", "data", rout, range(n), True), ("busy", c.get(f"{p}.wr_fixed", 0.0), True), ("pop", rout, range(n), 0.0)]
    elif op == "tilize_with_val_padding":
        # reader_unary_pad_dims_split_rows_multicore (pad 0): per block reserve(Wt); per valid row 1 read of the row's
        # bytes + 1 zero-fill of the pad columns; 1 zero-fill of the missing rows; zero-fill and read barriers; push.
        lat, gap, svc, timed = _noc(c, p, "read", cfg["in_place"], cfg["row_bytes"])
        z = c.get(f"{p}.rd_zcpb", 0.0)
        rd = [DMBatch(b, "space", rin, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                      [(r, lat, gap, svc, "rd"), (r, 0.0, cfg["pad_cols_bytes"] * z, 0.0, "rd"),
                       (1, 0.0, (TILE - r) * cfg["padded_row_bytes"] * z, 0.0, "rd")], 0, True, "push", lat_cr, dma=True)
              for b, r in zip(blocks, rows)]
        lat, gap, svc, timed = _noc(c, p, "write", cfg["out_place"], tile_bytes)
        wr = [DMBatch([i], "data", rout, c[f"{p}.wr_fixed"], 0.0, (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"],
                      [(1, lat, gap, svc, "wr")], 1, timed, "pop", lat_cr, dma=True) for i in range(n)]
    elif op == "untilize_with_unpadding" or cfg["factory"] == "Block":
        # reader (..._interleaved_start_id or ..._wh_multicore): per tile 1 read + barrier; writer
        # (..._split_rows_multicore or ..._wh_multicore): per block wait(Wt), each valid row 1 write + barrier, pop(Wt);
        # padding tile-rows are wait + pop only. Block / BlockInterleaved: a block = one tile-row of the core's square.
        lat, gap, svc, timed = _noc(c, p, "read", cfg["in_place"], tile_bytes)
        rd = [DMBatch([i], "space", rin, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                      [(1, lat, gap, svc, "rd")], 0, timed, "push", lat_cr, dma=True) for i in range(n)]
        lat, gap, svc, timed = _noc(c, p, "write", cfg["out_place"], cfg["row_bytes"])
        wr = []
        for b, r in zip(blocks, rows):
            wr.append(("wait", "data", rout, b, True))
            if r == 0:
                wr.append(("pop", rout, b, lat_cr))
            for k in range(r):
                last = k == r - 1
                wr.append(DMBatch(b if last else [], "data", rout, c[f"{p}.wr_fixed"] if k == 0 else 0.0, 0.0,
                                  (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"], [(1, lat, gap, svc, "wr")],
                                  1, True, "pop", lat_cr, dma=True))
    elif op == "tilize":
        lat, gap, svc, timed = _noc(c, p, "read", cfg["in_place"], cfg["row_bytes"])
        rd = [DMBatch(b, "space", rin, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                      [(TILE, lat, gap, svc, "rd")], 0, timed, "push", lat_cr, dma=True) for b in blocks]
        lat, gap, svc, timed = _noc(c, p, "write", cfg["out_place"], tile_bytes)
        wr = [DMBatch([i], "data", rout, c[f"{p}.wr_fixed"], 0.0, (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"],
                      [(1, lat, gap, svc, "wr")], 1, timed, "pop", lat_cr, dma=True) for i in range(n)]
    else:
        lat, gap, svc, timed = _noc(c, p, "read", cfg["in_place"], tile_bytes)
        rd = [DMBatch([i], "space", rin, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                      [(1, lat, gap, svc, "rd")], 0, timed, "push", lat_cr, dma=True) for i in range(n)]
        lat, gap, svc, timed = _noc(c, p, "write", cfg["out_place"], cfg["row_bytes"])
        wr = [DMBatch(b, "data", rout, c[f"{p}.wr_fixed"], 0.0, (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"],
                      [(TILE, lat, gap, svc, "wr")], 1, timed, "pop", lat_cr, dma=True) for b in blocks]

    # kernel prologue before the first DFB wait (args, address generators) and epilogue after the last batch
    # a producer kernel exits only once its consumer has popped everything it pushed (the DFB drains)
    rd = ([("busy", c.get(f"{p}.rd_setup", 0.0), True)] + rd + [("wait", "popped", rin, range(n), False),
                                                                ("busy", c.get(f"{p}.rd_tail", 0.0), True)])
    wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True)] + wr + [("busy", c.get(f"{p}.wr_tail", 0.0), True)]
    um, pk = [], []
    if fam == "tilize":
        # tilize_block(Wt) as one step per block: unpack pops the row only after its last tile, so the stages move
        # together (a two-stage split fits the runs no better).
        for k, b in enumerate(blocks):
            t = c[f"{p}.cu_fixed"] + c[f"{p}.cu_per_tile"] * wt + (c.get(f"{p}.cu_first", 0.0) if k == 0 else 0.0)
            pk += [("wait", "data", rin, b, True), ("wait", "space", rout, b, True), ("busy", t, True),
                   ("push", rout, b, lat_cr), ("pop", rin, b, lat_cr), ("count", wt, 1)]
    else:
        # untilize: the unpack(+math) TRISC and the pack TRISC run as two stages over DEST sections (sub-blocks of
        # sub_block_width tiles), half-sync = 2 sections. Unpack pops each sub-block as soon as it is unpacked; pack
        # reserves the block's output, packs each section, pushes the block.
        sw = sub_block_width(wt, elem)
        dst = Ring("dst", 1, 2, n // sw)
        sec = 0
        for b in blocks:
            pk.append(("wait", "space", rout, b, True))
            for j in range(0, wt, sw):
                part = b[j:j + sw]
                first = c.get(f"{p}.cu_first", 0.0) if sec == 0 else 0.0
                um += [("wait", "data", rin, part, True), ("wait", "space", dst, [sec], False),
                       ("busy", c[f"{p}.um_fixed"] + c[f"{p}.um_per_tile"] * sw + first, True), ("push", dst, [sec], 0.0),
                       ("pop", rin, part, lat_cr)]
                pk += [("wait", "data", dst, [sec], False), ("busy", c[f"{p}.pk_fixed"] + c[f"{p}.pk_per_tile"] * sw, False),
                       ("pop", dst, [sec], 0.0)]
                sec += 1
            pk += [("push", rout, b, lat_cr), ("count", wt, 1)]
    pk += [("wait", "popped", rout, range(n), False), ("busy", c.get(f"{p}.cu_tail", 0.0), False)]  # drain, teardown
    t_um = Thread(st_c, um, sets_end=False)
    t_pk = Thread(st_c, pk, t0=st_c.start + c.get(f"{p}.pk_offset", 0.0))
    t_rd, t_wr = Thread(st_r, rd), Thread(st_w, wr)
    return Program(groups=[Group([t_rd], dm=True), Group([t_um, t_pk]), Group([t_wr], dm=True)], logs=logs,
                   queues={"rd": Server(), "wr": Server()}, stats=[st_r, st_c, st_w], drain=c.get(f"{p}.drain", 0.0),
                   first_output=rout)


PREFIX = {("tilize", "Default"): "tz", ("tilize", "SingleCore"): "tzs", ("tilize", "Sharded"): "tzh",
          ("untilize", "MultiCore"): "utz", ("untilize", "SingleCore"): "utzs", ("untilize", "Block"): "utzb",
          ("tilize_with_val_padding", "Default"): "tvp", ("untilize_with_unpadding", "Interleaved"): "uwp",
          ("untilize_with_unpadding", "Block"): "utzb"}  # BlockInterleaved runs the untilize Block kernels
FAMILY = {"tilize": "tilize", "tilize_with_val_padding": "tilize", "untilize": "untilize",
          "untilize_with_unpadding": "untilize"}
GRID = 32  # Quasar worker cores (clusters) in the compute grid
L1_HALF = 3670016 // 2  # bytes SingleCore factories may give their two DFBs (half of the usable L1)


def plan(op, H, W, dtype="bf16", mem="dram", multicore=True):
    """Factory selection and work split (subset of the device ops' select logic: Default / MultiCore, SingleCore,
    tilize Sharded). Returns the factory name and one per-core cfg per distinct core load: [(cfg, cores)]."""
    elem = {"bf16": 2, "fp32": 4}[dtype]
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    tile_bytes = TILE * TILE * elem
    place = {"dram": "dram", "l1": "l1", "hs": "borrowed"}[mem]
    if mem == "hs":
        if op != "tilize":
            raise ValueError("height-sharded input is modeled for tilize only")
        ncores = min(GRID, ht)
        return "Sharded", [(dict(blocks=ht // ncores, wt=wt, elem=elem, row_bytes=0, in_place=place, out_place=place), ncores)]
    if op in ("tilize_with_val_padding", "untilize_with_unpadding"):
        if not multicore:
            raise ValueError(f"{op} SingleCore (a barrier per row) is not modeled yet")
        if op == "untilize_with_unpadding" and uses_block_factory(GRID, ht, wt):
            return "Block", block_plan(H, W, elem, place)  # BlockInterleaved: the untilize Block kernels
        return padded_plan(op, H, W, elem, place)
    if op == "untilize" and multicore and uses_block_factory(GRID, ht, wt):
        return "Block", block_plan(H, W, elem, place)
    if not multicore:
        tpb = wt
        while 2 * tpb * tile_bytes > L1_HALF and tpb > 1:
            tpb = max(d for d in range(1, tpb) if wt % d == 0)
        cfg = dict(blocks=ht * (wt // tpb), wt=tpb, elem=elem, row_bytes=tpb * TILE * elem, in_place=place, out_place=place)
        return "SingleCore", [(cfg, 1)]
    per_core, ncores, cliff = split_blocks(GRID, ht)
    base = dict(wt=wt, elem=elem, row_bytes=W * elem, in_place=place, out_place=place)
    loads = [(dict(base, blocks=per_core), ncores - (1 if cliff else 0))]
    if cliff:
        loads.append((dict(base, blocks=cliff), 1))
    return ("Default" if op == "tilize" else "MultiCore"), loads


def compute_ncores(grid, nblocks):
    """work_split compute_ncores: (cores, blocks per core)."""
    per = math.ceil(nblocks / grid)
    return math.ceil(nblocks / per), per


def _closest_square_larger_than(b, width, height, ref):
    r = math.isqrt(b)
    if r * r == b and math.ceil(width / r) * math.ceil(height / r) < ref:
        return b, r
    for cand in range(r + 1, width * height + 1):
        if math.ceil(width / cand) * math.ceil(height / cand) < ref:
            return cand * cand, cand
    return 1, 1


def uses_block_factory(grid, ht, wt):
    """The untilize device ops' 2-D (Block) selection: wider than 32 tiles, and a 2-D split of square blocks would use
    more cores than splitting by tile-rows (compute_ncores_wh in work_split_tilize.hpp)."""
    if not (wt > 32 and (ht > 32 or wt > ht)):
        return False
    per = math.ceil(ht * wt / grid)
    _, side = _closest_square_larger_than(per, wt, ht, grid)
    ncores_wh = math.ceil(wt / side) * math.ceil(ht / side)
    return compute_ncores(grid, ht)[0] < ncores_wh


def block_plan(H, W, elem, place):
    """untilize Block / untilize_with_unpadding BlockInterleaved (split_blocks_for_tilize_wh): each core untilizes an
    S x S-tile square (cliff regions on the right / bottom edges), one tile-row of the square at a time; both DFBs hold
    lcm(region widths) entries. The last tile-row may be mixed (H mod 32 rows) and the right edge's writes are clipped
    to the unpadded width."""
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    _, S = _closest_square_larger_than(math.ceil(ht * wt / GRID), wt, ht, GRID)
    fr, cr = wt // S, wt - (wt // S) * S  # full cores per row, cliff width
    fc, cc = ht // S, ht - (ht // S) * S  # full cores per column, cliff height
    widths = ([S] if fr and fc else []) + ([cr] if cr else []) + ([S] if cc and fr else [])
    cap = 1
    for w_ in widths:
        cap = cap * w_ // math.gcd(cap, w_)
    last_rows = H - (ht - 1) * TILE
    groups = {}
    for i in range(math.ceil(ht / S)):  # every core's square, grouped by (width, height, right edge, bottom edge)
        for j in range(math.ceil(wt / S)):
            sw, sh = min(S, wt - j * S), min(S, ht - i * S)
            key = (sw, sh, (j + 1) * S >= wt, (i + 1) * S >= ht)
            groups[key] = groups.get(key, 0) + 1
    loads = []
    # largest square first: loads[0] is a full core, as the first core the profiler reports
    for (sw, sh, right, bottom), n in sorted(groups.items(), key=lambda kv: (-kv[0][0] * kv[0][1], kv[0][2], kv[0][3])):
        cols = (W - (wt - sw) * TILE) if right else sw * TILE  # unpadded columns written per row
        rows = [TILE] * sh
        if bottom:
            rows[-1] = last_rows
        loads.append((dict(blocks=sh, wt=sw, elem=elem, row_bytes=cols * elem, rows=rows, cap=cap, in_place=place,
                           out_place=place), n))
    return loads


def padded_plan(op, H, W, elem, place):
    """tilize_with_val_padding Default / untilize_with_unpadding Interleaved: tile-rows split as tilize does; every
    tile-row has 32 valid rows except the last (H mod 32), which sits on the last core."""
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    last_rows = H - (ht - 1) * TILE
    per_core, ncores, cliff = split_blocks(GRID, ht)
    base = dict(wt=wt, elem=elem, row_bytes=W * elem, pad_cols_bytes=(wt * TILE - W) * elem,
                padded_row_bytes=wt * TILE * elem, in_place=place, out_place=place)
    last_blocks = cliff or per_core
    loads = []
    if ncores > 1:
        loads.append((dict(base, blocks=per_core, rows=[TILE] * per_core), ncores - 1))
    loads.append((dict(base, blocks=last_blocks, rows=[TILE] * (last_blocks - 1) + [last_rows]), 1))
    return ("Default" if op == "tilize_with_val_padding" else "Interleaved"), loads


def estimate(op, H, W, costs, dtype="bf16", mem="dram", multicore=True, trace=0):
    """Busiest core's span (cycles) of one tilize/untilize, with the factory and per-load results."""
    from ..engine import run

    factory, loads = plan(op, H, W, dtype, mem, multicore)
    for cfg, _ in loads:
        cfg["factory"] = factory
    results = [(cfg, cores, run(build(op, cfg, costs), trace)) for cfg, cores in loads if cfg["blocks"]]
    span = max(r["makespan"] for _, _, r in results)
    return {"factory": factory, "span": span, "loads": results}
