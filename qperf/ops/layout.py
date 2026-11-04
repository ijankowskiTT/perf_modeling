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
    nb, wt, elem = cfg["blocks"], cfg["wt"], cfg["elem"]
    tile_bytes = TILE * TILE * elem
    n = nb * wt
    borrowed = cfg["in_place"] == "borrowed"
    cap_in = cap_out = wt
    if op == "untilize" and nb > 1:
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
    if op == "tilize":
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
          ("untilize", "MultiCore"): "utz", ("untilize", "SingleCore"): "utzs"}
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


def estimate(op, H, W, costs, dtype="bf16", mem="dram", multicore=True, trace=0):
    """Busiest core's span (cycles) of one tilize/untilize, with the factory and per-load results."""
    from ..engine import run

    factory, loads = plan(op, H, W, dtype, mem, multicore)
    for cfg, _ in loads:
        cfg["factory"] = factory
    results = [(cfg, cores, run(build(op, cfg, costs), trace)) for cfg, cores in loads if cfg["blocks"]]
    span = max(r["makespan"] for _, _, r in results)
    return {"factory": factory, "span": span, "loads": results}
