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
fp32: the same kernels and loops (tilize_block per tile-row, untilize sub-blocks of <= 4 tiles: fp32 DEST holds half
the tiles), unpacked straight to DEST; <p>.f32_<term> overrides <p>.<term> for fp32 (dtype_costs).

Hypothetical designs (not in the pinned kernels; tilize Default and untilize MultiCore): plan(design=...) re-splits the
work (cores, split rows | cols | 2d), narrows blocks (tiles_per_block), deepens the DFBs (dfb_blocks) and batches the
per-tile DM thread's barriers (tilize write_batch, untilize read_batch), timed with the same loops and the factory's
own prefix (resolve_design, design_loads). structure() describes a core's work for reports.
"""

from __future__ import annotations

import math

from ..engine import DMBatch, Group, Program, Ring, Server, ThreadStats, Thread, TransferLog, run

TILE = 32
ELEM = {"bf16": 2, "fp32": 4}  # bytes per element
GRID_X, GRID_Y = 8, 4  # Quasar worker cores (clusters) in the compute grid
GRID = GRID_X * GRID_Y
L1_ALIGN, DRAM_ALIGN = 16, 64  # buffer alignment (bytes)
SHARD_OPS = ("interleaved_to_sharded", "sharded_to_interleaved")
STRATEGY = {"height": "hs", "width": "ws", "block": "bs"}  # shard strategy -> mem string kind


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
    lat = svc = 0.0
    if place == "l1":
        gap += nbytes * c.get(f"{p}.{d}_cpb_l1", 0.0)
        lat = c.get(f"{p}.{d}_lat_l1", 0.0)
        svc = c.get(f"{p}.{d}_svc_l1", 0.0)  # per-request service in the core's shared L1 completion queue
        share = c.get(f"{p}.l1_share_bw", 0.0)  # fair share of the global L1 network among the cores moving L1 data
        if share:
            gap = max(gap, nbytes * c.get("_active_l1_cores", 1) / share)
    return lat, gap, svc, bool(lat or gap or svc)


def _batches(c, p, d, ring, units, count, place, nbytes, lat_cr):
    """One DM batch per unit (a tile or a block of tiles), `count` transfers of nbytes each (None: one per tile of the
    unit): the reader's reserve / reads / barrier / push (d = "rd") or the writer's wait / writes / flush / pop (d = "wr")."""
    rd = d == "rd"
    lat, gap, svc, timed = _noc(c, p, "read" if rd else "write", place, nbytes)
    return [DMBatch(u, "space" if rd else "data", ring, c[f"{p}.{d}_fixed"], 0.0, (c.get(f"{p}.{d}_first", 0.0), 0.0, 0),
                    c[f"{p}.{d}_issue"], [(count or len(u), lat, gap, svc, d)], 0 if rd else 1, timed,
                    "push" if rd else "pop", lat_cr, dma=True) for u in units]


def _groups(blocks, k):
    """Tiles per DM barrier: groups of k tiles that never span two blocks (the last group of a block may be smaller);
    k = 1 is the kernels' per-tile batch."""
    return [b[j:j + k] for b in blocks for j in range(0, len(b), k)]


def _queues(shared):
    q = Server()
    return {"rd": q, "wr": q} if shared else {"rd": q, "wr": Server()}


def dm_program(st_r, rd, st_w, wr, logs, shared, drain):
    """One core running a reader and a writer DM thread, no compute."""
    return Program(groups=[Group([Thread(st_r, rd)], dm=True), Group([Thread(st_w, wr)], dm=True)], logs=logs,
                   queues=_queues(shared), stats=[st_r, st_w], drain=drain, first_output=None)


def _build_shard_move(op, cfg, c, p):
    """interleaved_to_sharded / sharded_to_interleaved (TILE, L1 shard borrowed): only the interleaved side moves data.
    i2s reader_unary_sharded_blocks_interleaved_start_id: reserve(shard), shard tile reads with a barrier every `thr`
    reads and one at the end, push(shard); writer_unary_sharded: wait + pop. s2i reader_unary_sharded: push(shard);
    writer_unary_sharded_blocks_interleaved_start_id: wait(shard), one write per tile, one barrier, pop(shard).

    ROW_MAJOR (cfg["rm"]): a DFB entry is one row (stick) of the shard, `rows` x `width` bytes on this core (clipped on
    the last shards).
    i2s reader_unary_stick_layout_sharded_blocks_interleaved_start_id: reserve(rows); aligned (source in L1, or DRAM
      with the core's width offset and align(unit, 16) both multiples of 64): one read of `width` bytes per row, one
      barrier, push(rows). Unaligned: per row a TRID-tagged read source -> scratch of `abytes` (width + offset into the
      64 B-aligned source window), then a local loopback read scratch -> shard of `width` bytes (an L1 NoC read); up to
      4 TRID slots in flight, each polled until flushed; push(rows). writer_unary_sharded: wait + pop.
    s2i reader_unary_sharded: push(rows); writer_unary_stick_layout_sharded_blocks_interleaved_start_id: wait(shard),
      one write of `width` bytes per row, one barrier, pop."""
    rm = cfg.get("rm", False)
    n, elem = (cfg["rows"] if rm else cfg["tiles"]), cfg["elem"]
    tile_bytes = TILE * TILE * elem
    ring = Ring("shard", 1, n + 1, n)
    allt = list(range(n))
    logs, lat_cr = [TransferLog(), TransferLog()], 0.0
    st_r = ThreadStats("reader", 0, start=c[f"{p}.start_reader"])
    st_w = ThreadStats("writer", 0, start=c[f"{p}.start_writer"])
    # interleaved pages in L1: cores that walk their shards in lockstep and start on the same bank queue on it
    # (<p>.l1_conf_cpb: extra cycles per byte per other core on the bank; 0 where not fitted)
    conf = tile_bytes * c.get(f"{p}.l1_conf_cpb", 0.0) * (cfg.get("bank_conflict", 1) - 1)
    if rm and op == "interleaved_to_sharded":
        rd = [("busy", c.get(f"{p}.rd_setup", 0.0), True), ("wait", "space", ring, allt, True)]
        lat, gap, svc, _ = _noc(c, p, "read", cfg["in_place"], cfg["abytes"])
        if cfg["aligned"]:
            groups, inflight = [(n, lat, gap, svc, "rd")], 0
        else:
            # issue order of the TRID loop: 4 source reads, then each slot's loopback once its source read is flushed,
            # then the slot's next source read once its loopback is flushed: max_inflight=4 encodes both dependencies
            llat, lgap, lsvc, _ = _noc(c, p, "read", "l1", cfg["width"])
            groups, inflight = [], 4  # num_trids
            for j in range(0, n, inflight):
                k = min(inflight, n - j)
                groups += [(k, lat, gap, svc, "rd"), (k, llat, lgap, lsvc, "rd")]
        rd += [DMBatch(allt, "space", ring, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0),
                       c[f"{p}.rd_issue"], groups, 0, True, "push", lat_cr, max_inflight=inflight, dma=True),
               ("wait", "popped", ring, allt, False), ("busy", c.get(f"{p}.rd_tail", 0.0), True)]
        wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True), ("wait", "data", ring, allt, True),
              ("busy", c.get(f"{p}.wr_fixed", 0.0), True), ("pop", ring, allt, 0.0), ("busy", c.get(f"{p}.wr_tail", 0.0), True)]
    elif op == "interleaved_to_sharded":
        lat, gap, svc, _ = _noc(c, p, "read", cfg["in_place"], tile_bytes)
        gap += conf if cfg["in_place"] == "l1" else 0.0
        thr = cfg["thr"] or n
        rd = [("busy", c.get(f"{p}.rd_setup", 0.0), True), ("wait", "space", ring, allt, True)]
        for j in range(0, n, thr):
            last = j + thr >= n
            rd.append(DMBatch(allt if last else [], "space", ring, c[f"{p}.rd_fixed"] if j == 0 else 0.0, 0.0,
                              (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                              [(min(thr, n - j), lat, gap, svc, "rd")], 0, True, "push", lat_cr, dma=True))
        rd += [("wait", "popped", ring, allt, False), ("busy", c.get(f"{p}.rd_tail", 0.0), True)]
        wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True), ("wait", "data", ring, allt, True),
              ("busy", c.get(f"{p}.wr_fixed", 0.0), True), ("pop", ring, allt, 0.0), ("busy", c.get(f"{p}.wr_tail", 0.0), True)]
    else:
        lat, gap, svc, _ = _noc(c, p, "write", cfg["out_place"], cfg["width"] if rm else tile_bytes)
        gap += conf if cfg["out_place"] == "l1" else 0.0
        rd = [("busy", c.get(f"{p}.rd_setup", 0.0), True), ("push", ring, allt, 0.0), ("wait", "popped", ring, allt, False),
              ("busy", c.get(f"{p}.rd_tail", 0.0), True)]
        wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True),
              DMBatch(allt, "data", ring, c[f"{p}.wr_fixed"], 0.0, (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"],
                      [(n, lat, gap, svc, "wr")], 1, True, "pop", lat_cr, dma=True),
              ("busy", c.get(f"{p}.wr_tail", 0.0), True)]
    return dm_program(st_r, rd, st_w, wr, logs, False, c.get(f"{p}.drain", 0.0))


def shard_plan(op, H, W, elem, mem):
    """mem = <hs|ws|bs><cores>[d|l][r]: shard strategy, cores, interleaved side in DRAM (default) or L1. TILE: even
    shards. Trailing r = ROW_MAJOR: shards as the runner and ttnn build them, ceil(H/n) x W (height), H x ceil(W/n)
    (width) or ceil(H/k) x ceil(W/k) on a k x k grid (block), in elements; only cores with data run, the last shard row /
    column is clipped. Per core, as the factory's loop computes it: rows, width bytes, and for i2s from DRAM whether the
    stick read is aligned (curr_idx_w % 64 == 0 and align(unit, 16) % 64 == 0) or goes through the scratch (then
    abytes = curr_idx_w % 64 + width). Factories: RowMajor (s2i; i2s with every core aligned), RowMajorUnaligned (i2s
    with any core unaligned). One load per distinct core, core 0 (a full shard at offset 0) first."""
    kind, rest = mem[:2], mem[2:]
    rm = rest.endswith("r")
    rest = rest.rstrip("r")
    side = "l1" if rest.endswith("l") else "dram"
    ncores = int(rest.rstrip("dl"))
    if rm:
        i2s = op == "interleaved_to_sharded"
        if kind == "bs":
            k = math.isqrt(ncores)
            if k * k != ncores:
                raise ValueError("block sharding is modeled on a square core grid (4, 9, 16, ...)")
            sh, sw = math.ceil(H / k), math.ceil(W / k)
        else:
            sh, sw = (math.ceil(H / ncores), W) if kind == "hs" else (H, math.ceil(W / ncores))
        if kind != "hs":  # the runner rounds a split width up to whole 16 B (the ops' shard page alignment check)
            sw = -(-sw * elem // 16) * 16 // elem
        unit, row = sw * elem, W * elem  # shard row bytes, tensor row bytes
        if unit % 16:
            raise ValueError("ROW_MAJOR shard rows must be a multiple of 16 B (the ops reject other shard widths)")
        padded = -(-unit // 16) * 16  # shard row stride in L1 (keep_l1_aligned, L1 alignment 16 B)
        groups = {}
        for i in range(math.ceil(H / sh)):
            for j in range(math.ceil(row / unit)):
                rows, width, off = min(sh, H - i * sh), min(unit, row - j * unit), j * unit
                aligned = not i2s or side == "l1" or (off % 64 == 0 and padded % 64 == 0)  # DRAM alignment 64 B
                key = (rows, width, aligned, width if aligned else off % 64 + width)
                g = groups.setdefault(key, [0, False])
                g[0] += 1
                g[1] = g[1] or (i == 0 and j == 0)
        loads = []
        for (rows, width, aligned, abytes), (n, first) in sorted(groups.items(),
                                                                 key=lambda kv: (not kv[1][1], -kv[0][0] * kv[0][1])):
            loads.append((dict(rm=True, rows=rows, width=width, aligned=aligned, abytes=abytes, elem=elem, blocks=1,
                               wt=rows, tiles=rows * width / (TILE * TILE * elem),  # tile units: estimate's L1 bound
                               in_place=side if i2s else "borrowed",
                               # the unaligned reader's loopback copy into the shard is an L1 NoC read
                               out_place=("borrowed" if aligned else "l1") if i2s else side), n))
        if i2s and not all(cfg["aligned"] for cfg, _ in loads):
            return "RowMajorUnaligned", loads
        return "RowMajor", loads
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    if kind == "hs":
        sh, sw = math.ceil(ht / ncores), wt
        first = [i * sh * wt for i in range(ncores)]  # each core's first interleaved page (tile id)
    elif kind == "ws":
        sh, sw = ht, math.ceil(wt / ncores)
        first = [j * sw for j in range(ncores)]
    else:
        k = math.isqrt(ncores)
        sh, sw = math.ceil(ht / k), math.ceil(wt / k)
        first = [i * sh * wt + j * sw for i in range(k) for j in range(k)]
    # bank conflict: most cores whose first page sits on the same L1 bank (pages round-robin over GRID banks); with
    # equal shards they stay aligned page after page
    banks = {}
    for f in first:
        banks[f % GRID] = banks.get(f % GRID, 0) + 1
    tile_bytes = TILE * TILE * elem
    thr = (512 // ncores * 1152) // tile_bytes  # reader barrier every thr reads (0: only the final barrier)
    cfg = dict(tiles=sh * sw, elem=elem, thr=thr, blocks=1, wt=sh * sw, bank_conflict=max(banks.values()),
               in_place=side if op == "interleaved_to_sharded" else "borrowed",
               out_place="borrowed" if op == "interleaved_to_sharded" else side)
    return "Default", [(cfg, ncores)]


def dtype_costs(c, p, elem):
    """fp32 (elem 4) runs the same kernels with UnpackToDest and fp32 DEST (half the DEST tiles): a factory's
    <p>.f32_<term> replaces <p>.<term> where the fp32 runs measured it (compute and per-byte terms); every other
    term is the bf16 one."""
    if elem != 4:
        return c
    pre = p + ".f32_"
    over = {p + "." + k[len(pre):]: v for k, v in c.items() if k.startswith(pre)}
    return {**c, **over} if over else c


def _ext(op):
    from . import registry

    return registry.module_for(op)


def build(op, cfg, c):
    """One core of a tilize/untilize factory. cfg: blocks (on this core), wt (tiles per block), elem (bytes),
    row_bytes (bytes per stick request), in_place / out_place (dram | l1 | borrowed), factory."""
    m = _ext(op)
    if m:
        return m.build(op, cfg, c, m.PREFIX[(op, cfg["factory"])])
    p = PREFIX[(op, cfg["factory"])]
    c = dtype_costs(c, p, cfg["elem"])
    if op in SHARD_OPS:
        return _build_shard_move(op, cfg, c, p)
    fam = FAMILY[op]  # tilize or untilize: which compute and base DM loops the op's kernels share
    nb, wt, elem = cfg["blocks"], cfg["wt"], cfg["elem"]
    rows = cfg.get("rows") or [TILE] * nb  # valid rows per block (the last tile-row of a padded tensor is mixed)
    tile_bytes = TILE * TILE * elem
    n = nb * wt
    borrowed = cfg["in_place"] == "borrowed"
    cap_in = cap_out = cfg.get("cap") or wt
    if op == "untilize" and nb > 1 and not cfg.get("cap"):
        # MultiCore double-buffers with more than one block; SingleCore's DFBs hold one block (factory), but craq-sim
        # behaves as if they held two (<p>.dfb_blocks, per profile)
        dfb_blocks = 2 if cfg["factory"] == "MultiCore" else c.get(f"{p}.dfb_blocks", 1)
        cap_in = cap_out = int(dfb_blocks) * wt
    rin = Ring("in", 1, n + 1 if borrowed else cap_in, n)
    rout = Ring("out", 1, n + 1 if borrowed else cap_out, n)
    logs, lat_cr = [TransferLog(), TransferLog()], c.get(f"{p}.credit_latency", 0.0)  # own terms only, not binary_ng's
    # A profile that measured it (ttsim-perf: noc.barrier_all_kinds, noc.rw_serialized): L1 reads and writes share one
    # completion queue and a barrier waits for either kind, as for binary_ng. Only when an operand is in L1.
    shared_l1 = "l1" in (cfg["in_place"], cfg["out_place"]) and c.get("noc.barrier_all_kinds", 0.0) >= 0.5
    if shared_l1:
        logs = [logs[0], logs[0]]
    blocks = [list(range(b * wt, (b + 1) * wt)) for b in range(nb)]
    tiles = [[i] for i in range(n)]
    st_r = ThreadStats("reader", 0, start=c[f"{p}.start_reader"])
    st_c = ThreadStats("compute", 0, start=c[f"{p}.start_compute"])
    st_w = ThreadStats("writer", 0, start=c[f"{p}.start_writer"])

    if borrowed:  # the shard backs both DFBs: publish all, compute, pop all
        rd = [("busy", c.get(f"{p}.rd_fixed", 0.0), True), ("push", rin, range(n), 0.0)]
        wr = [("wait", "data", rout, range(n), True), ("busy", c.get(f"{p}.wr_fixed", 0.0), True), ("pop", rout, range(n), 0.0)]
    elif op == "tilize_with_val_padding" and cfg["factory"] == "SingleCore":
        # reader_unary_pad_dims_split_rows: per block reserve; per valid row 1 read + (pad columns: CPU store loop) +
        # barrier; missing rows filled by a CPU store loop; push. <p>.rd_pcpb = CPU fill cycles per byte, <p>.rd_pfix =
        # fixed fill cost per row that has pad columns.
        lat, gap, svc, _ = _noc(c, p, "read", cfg["in_place"], cfg["row_bytes"])
        pc = c.get(f"{p}.rd_pcpb", 0.0)
        rd = []
        for b, r in zip(blocks, rows):
            rd.append(("wait", "space", rin, b, True))
            for k in range(r):
                last = k == r - 1
                # a row with pad columns runs the fill code: a fixed part (branch, byte loop to 4 B alignment) + per byte
                post = ((c.get(f"{p}.rd_pfix", 0.0) + cfg["pad_cols_bytes"] * pc) if cfg["pad_cols_bytes"] else 0.0) + \
                    ((TILE - r) * cfg["padded_row_bytes"] * pc if last else 0.0)
                rd.append(DMBatch(b if last else [], "space", rin, c[f"{p}.rd_fixed"], post,
                                  (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"], [(1, lat, gap, svc, "rd")],
                                  0, True, "push", lat_cr, dma=True))
        wr = _batches(c, p, "wr", rout, tiles, 1, cfg["out_place"], tile_bytes, lat_cr)
    elif op == "tilize_with_val_padding":
        # reader_unary_pad_dims_split_rows_multicore (pad 0): per block reserve(Wt); per valid row 1 read of the row's
        # bytes + 1 zero-fill of the pad columns; 1 zero-fill of the missing rows; zero-fill and read barriers; push.
        lat, gap, svc, _ = _noc(c, p, "read", cfg["in_place"], cfg["row_bytes"])
        z = c.get(f"{p}.rd_zcpb", 0.0)
        rd = [DMBatch(b, "space", rin, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                      [(r, lat, gap, svc, "rd"), (r, 0.0, cfg["pad_cols_bytes"] * z, 0.0, "rd"),
                       (1, 0.0, (TILE - r) * cfg["padded_row_bytes"] * z, 0.0, "rd")], 0, True, "push", lat_cr, dma=True)
              for b, r in zip(blocks, rows)]
        wr = _batches(c, p, "wr", rout, tiles, 1, cfg["out_place"], tile_bytes, lat_cr)
    elif op == "untilize_with_unpadding" or cfg["factory"] == "Block":
        # reader (..._interleaved_start_id or ..._wh_multicore): per tile 1 read + barrier; writer
        # (..._split_rows_multicore or ..._wh_multicore): per block wait(Wt), each valid row 1 write + barrier, pop(Wt);
        # padding tile-rows are wait + pop only. Block / BlockInterleaved: a block = one tile-row of the core's square.
        rd = _batches(c, p, "rd", rin, tiles, 1, cfg["in_place"], tile_bytes, lat_cr)
        lat, gap, svc, _ = _noc(c, p, "write", cfg["out_place"], cfg["row_bytes"])
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
        rd = _batches(c, p, "rd", rin, blocks, TILE, cfg["in_place"], cfg["row_bytes"], lat_cr)
        # write_batch k > 1 (design): wait(k), k writes at DFB offsets, one flush, pop(k) per group. Simulator runs of
        # that kernel (calibration/variants/tilize_writer_batched.cpp, k = 4 and 8; tilize-wbatch-*.jsonl) gain from 4 to
        # 8 tiles 1.6x (ttsim-perf) to 2.4x (craq-sim) what a per-flush cost alone gives against the per-tile kernel:
        # each write of the batched loop costs <p>.wr_bissue more than the pinned kernel's (offset address, inner loop)
        # while the flush (<p>.wr_fixed + the last write's transfer) is paid once per group.
        k = cfg.get("batch", 1)
        cw = dict(c, **{f"{p}.wr_issue": c[f"{p}.wr_issue"] + c.get(f"{p}.wr_bissue", 0.0)}) if k > 1 else c
        wr = _batches(cw, p, "wr", rout, _groups(blocks, k), None, cfg["out_place"], tile_bytes, lat_cr)
    else:
        rd = _batches(c, p, "rd", rin, _groups(blocks, cfg.get("batch", 1)), None, cfg["in_place"], tile_bytes, lat_cr)
        wr = _batches(c, p, "wr", rout, blocks, TILE, cfg["out_place"], cfg["row_bytes"], lat_cr)

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
                   queues=_queues(shared_l1), stats=[st_r, st_c, st_w], drain=c.get(f"{p}.drain", 0.0),
                   first_output=rout)


PREFIX = {("tilize", "Default"): "tz", ("tilize", "SingleCore"): "tzs", ("tilize", "Sharded"): "tzh",
          ("untilize", "MultiCore"): "utz", ("untilize", "SingleCore"): "utzs", ("untilize", "Block"): "utzb",
          ("tilize_with_val_padding", "Default"): "tvp", ("untilize_with_unpadding", "Interleaved"): "uwp",
          ("untilize_with_unpadding", "Block"): "utzb",
          ("tilize_with_val_padding", "SingleCore"): "tvps", ("untilize_with_unpadding", "SingleCore"): "uwps",
          ("interleaved_to_sharded", "Default"): "i2s", ("sharded_to_interleaved", "Default"): "s2i",
          ("interleaved_to_sharded", "RowMajor"): "i2sr", ("interleaved_to_sharded", "RowMajorUnaligned"): "i2sru",
          ("sharded_to_interleaved", "RowMajor"): "s2ir"}  # BlockInterleaved runs the untilize Block kernels
FAMILY = {"tilize": "tilize", "tilize_with_val_padding": "tilize", "untilize": "untilize",
          "untilize_with_unpadding": "untilize", "interleaved_to_sharded": "move", "sharded_to_interleaved": "move"}
L1_HALF = 3670016 // 2  # bytes SingleCore factories may give their two DFBs (half of the usable L1)


def plan(op, H, W, dtype="bf16", mem="dram", multicore=True, design=None):
    """Factory selection and work split (subset of the device ops' select logic: Default / MultiCore, SingleCore,
    tilize Sharded). Returns the factory name and one per-core cfg per distinct core load: [(cfg, cores)].

    design: hypothetical layout knobs (DESIGN_KNOBS) for tilize Default / untilize MultiCore; a design that resolves to
    the factory's own plan returns that plan unchanged (see resolve_design / design_loads)."""
    factory, loads = _factory_plan(op, H, W, dtype, mem, multicore)
    design = {k: v for k, v in (design or {}).items() if v is not None}
    if not design:
        return factory, loads
    d = resolve_design(op, factory, H, W, dtype, design)
    if not d["hypothetical"]:
        return factory, loads
    return factory, design_loads(H, W, ELEM[dtype], loads[0][0]["in_place"], d)


def _factory_plan(op, H, W, dtype, mem, multicore):
    m = _ext(op)
    if m:
        return m.plan(op, H, W, dtype, mem, multicore)
    elem = ELEM[dtype]
    if op in SHARD_OPS:
        return shard_plan(op, H, W, elem, mem)
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    tile_bytes = TILE * TILE * elem
    place = {"dram": "dram", "l1": "l1", "hs": "borrowed"}[mem]
    if mem == "hs":
        if op != "tilize":
            raise ValueError("height-sharded input is modeled for tilize only")
        ncores = min(GRID, ht)
        return "Sharded", [(dict(blocks=ht // ncores, wt=wt, elem=elem, row_bytes=0, in_place=place, out_place=place), ncores)]
    if op in ("tilize_with_val_padding", "untilize_with_unpadding"):
        if not multicore:  # one core, a block = the whole padded row (fits L1 up to ~440 tiles wide)
            cfg = dict(blocks=ht, wt=wt, elem=elem, row_bytes=W * elem, pad_cols_bytes=(wt * TILE - W) * elem,
                       padded_row_bytes=wt * TILE * elem, in_place=place, out_place=place,
                       rows=[TILE] * (ht - 1) + [H - (ht - 1) * TILE])
            return "SingleCore", [(cfg, 1)]
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
        if op == "untilize" and elem == 4:
            # untilize_single_core_program_factory gives both DFBs one block (no double buffering). The bf16 fit was
            # made with build()'s 2 blocks and is kept until refit; fp32 (fitted here) uses the factory's sizes.
            cfg["cap"] = tpb
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
    fr, cr = divmod(wt, S)  # full cores per row, cliff width
    fc, cc = divmod(ht, S)  # full cores per column, cliff height
    cap = math.lcm(*([S] if fr and fc else []), *([cr] if cr else []), *([S] if cc and fr else []))
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


# ------------------------------------------------------------------------------------------- hypothetical design
# Knobs the pinned factories do not expose (tilize Default, untilize MultiCore only). Each variant runs the factory's
# own kernel loops and fitted cost prefix (tz / utz): costs are per request, per byte, per barrier, per block and per
# tile, so a different split, block width, DFB depth or barrier batch is timed with the same calibrated terms.
DESIGN_KNOBS = ("cores", "split", "tiles_per_block", "dfb_blocks", "write_batch", "read_batch")
DESIGN_FACTORIES = {("tilize", "Default"): "write_batch", ("untilize", "MultiCore"): "read_batch"}
SPLITS = ("rows", "cols", "2d")


def _largest_divisor_le(n, k):
    return max(d for d in range(1, min(n, k) + 1) if n % d == 0)


def _split(split, cores, ht, wt):
    """(row_cores, tile-rows per core, col_chunks, tiles per chunk) of the busiest core. rows: split_blocks over
    tile-rows; cols: the same row split, each tile-row cut into k = cores // row_cores column chunks (at most one per
    tile column, widths ceil(wt / k) so every chunk has work); 2d: row_cores x col_chunks <= cores with the fewest
    tiles on the busiest core (ties: fewer column chunks = longer stick reads, then fewer row cores)."""
    if split == "2d":
        best = None
        for r in range(1, min(cores, ht) + 1):
            per_r = math.ceil(ht / r)
            r_eff = math.ceil(ht / per_r)
            for k in range(1, min(wt, cores // r_eff) + 1):
                chunk = math.ceil(wt / k)
                key = (per_r * chunk, math.ceil(wt / chunk), r_eff)
                if best is None or key < best[0]:
                    best = (key, (r_eff, per_r, math.ceil(wt / chunk), chunk))
        return best[1]
    per_r, r, _ = split_blocks(cores, ht)
    k = min(wt, max(1, cores // r)) if split == "cols" else 1
    chunk = math.ceil(wt / k)
    return r, per_r, math.ceil(wt / chunk), chunk


def resolve_design(op, factory, H, W, dtype, design):
    """The plan a set of design knobs resolves to, against the factory's own (every knob at its default: cores = GRID,
    split rows, a block = the core's whole chunk, the factory's DFB depth, per-tile barriers). Raises ValueError for a
    factory without the knobs or a value the shape cannot take. tiles_per_block snaps to the largest divisor of the
    chunk width <= the request ("half": half the chunk); dfb_blocks and the batch clamp to what the core's work can use,
    so knobs that change nothing resolve to the factory plan (hypothetical False)."""
    design = {k: v for k, v in (design or {}).items() if v is not None}
    batch_knob = DESIGN_FACTORIES.get((op, factory))
    if batch_knob is None:
        raise ValueError(f"layout design knobs ({', '.join(sorted(design))}) are modeled for tilize Default and untilize "
                         f"MultiCore only, not {op} {factory}")
    other = "read_batch" if batch_knob == "write_batch" else "write_batch"
    if other in design:
        raise ValueError(f"{other} is not a knob of {op} {factory}: its batched DM thread takes {batch_knob}")
    elem = ELEM[dtype]
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    cores, split = design.get("cores", GRID), design.get("split", "rows")
    if not isinstance(cores, int) or not 1 <= cores <= GRID:
        raise ValueError(f"cores must be 1..{GRID}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    r, per_r, k, chunk = _split(split, cores, ht, wt)
    tpb = design.get("tiles_per_block") or chunk
    if tpb == "half":
        tpb = max(1, chunk // 2)
    if not isinstance(tpb, int) or tpb < 1:
        raise ValueError('tiles_per_block must be a positive integer or "half"')
    tpb = _largest_divisor_le(chunk, tpb)
    blocks = per_r * (chunk // tpb)
    rule_dfb = min(1 if op == "tilize" else 2, blocks)  # the factory's DFB depth for this many blocks
    dfb = design.get("dfb_blocks", rule_dfb)
    if dfb not in (1, 2, 3):
        raise ValueError("dfb_blocks must be 1, 2 or 3")
    dfb = min(dfb, blocks)
    batch = design.get(batch_knob, 1)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError(f"{batch_knob} must be a positive integer")
    batch = min(batch, tpb)
    per_f, n_f, _ = split_blocks(GRID, ht)
    sig = (r, per_r, k, chunk, tpb, dfb, batch)
    hyp = sig != (n_f, per_f, 1, wt, wt, min(1 if op == "tilize" else 2, per_f), 1)
    if hyp and 2 * dfb * tpb * TILE * TILE * elem > L1_HALF:
        raise ValueError(f"DFBs of {dfb} x {tpb} tiles ({2 * dfb * tpb * TILE * TILE * elem} B for in + out) exceed the "
                         f"{L1_HALF} B budget; use fewer tiles_per_block or dfb_blocks")
    changed = []
    if r * k != n_f:
        changed.append(f"cores={r * k}")
    if k > 1:
        changed.append(f"split={split} ({r}x{k})")
    if tpb != chunk:
        changed.append(f"tiles_per_block={tpb}")
    if dfb != rule_dfb:
        changed.append(f"dfb_blocks={dfb}")
    if batch > 1:
        changed.append(f"{batch_knob}={batch}")
    if hyp and not changed:
        changed.append(f"cores={r * k} ({per_r} tile-rows each)")
    return {"split": "rows" if k == 1 else split, "cores": r * k, "row_cores": r, "rows_per_core": per_r,
            "col_chunks": k, "chunk": chunk, "tiles_per_block": tpb, "blocks": blocks, "dfb_blocks": dfb,
            "batch_knob": batch_knob, "batch": batch, "sig": sig, "hypothetical": hyp, "changed": changed}


def design_loads(H, W, elem, place, d):
    """Per-core loads of a resolved design. Only the busiest core (full tile-rows x a full chunk) is run; the others
    (cliff tile-rows, the narrower last chunk: less work on the same loops) carry their tile counts for the cross-core
    L1 bound and the core count (skip=True)."""
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    tpb = d["tiles_per_block"]
    base = dict(elem=elem, in_place=place, out_place=place)
    whole_row = d["col_chunks"] == 1 and tpb == wt  # a stick read covers the logical row, as the factory's
    busiest = dict(base, blocks=d["blocks"], wt=tpb, row_bytes=(W if whole_row else tpb * TILE) * elem, cap=d["dfb_blocks"] * tpb,
                   batch=d["batch"], split=d["split"], row_cores=d["row_cores"], col_chunks=d["col_chunks"])

    def groups(full, n, total):
        last = total - (n - 1) * full
        return [(full, n)] if last == full else [(full, n - 1), (last, 1)]

    loads = []
    for nr, cr in groups(d["rows_per_core"], d["row_cores"], ht):
        for nc, cc in groups(d["chunk"], d["col_chunks"], wt):
            if not cr * cc:
                continue
            if nr == d["rows_per_core"] and nc == d["chunk"]:
                loads.insert(0, (busiest, cr * cc))
            else:
                loads.append((dict(base, blocks=nr, wt=nc, row_bytes=nc * TILE * elem, tiles=nr * nc, skip=True), cr * cc))
    return loads


def structure(op, factory, H, W, loads, cfg, c):
    """Compact description of one core (cfg, the busiest of loads) and the grid, for reports. None for the ops it does
    not describe (sharding moves, registered modules). Counts follow build()'s loops."""
    if _ext(op) or op in SHARD_OPS:
        return None
    fam = FAMILY[op]
    nb, wt, elem = cfg["blocks"], cfg["wt"], cfg["elem"]
    tile_bytes = TILE * TILE * elem
    n = nb * wt
    rows = cfg.get("rows") or [TILE] * nb
    cores = sum(cnt for _, cnt in loads)
    if "split" in cfg:
        split, rc, kc = cfg["split"], cfg["row_cores"], cfg["col_chunks"]
    elif factory == "Block":  # S x S-tile squares
        side = loads[0][0]["wt"]
        split, rc, kc = "2d", math.ceil(math.ceil(H / TILE) / side), math.ceil(math.ceil(W / TILE) / side)
    else:
        split, rc, kc = "rows", cores, 1
    tile_req = {"requests": n, "bytes_per_request": tile_bytes, "barriers": nb * math.ceil(wt / cfg.get("batch", 1))}
    borrowed = cfg["in_place"] == "borrowed"
    if borrowed:  # the shard backs both DFBs: no NoC traffic
        reader = {"requests": 0, "bytes_per_request": 0, "barriers": 0}
        writer = dict(reader)
    elif fam == "tilize":
        per_row = op == "tilize_with_val_padding" and factory == "SingleCore"  # a barrier after every row read
        reader = {"requests": sum(rows), "bytes_per_request": cfg["row_bytes"], "barriers": sum(rows) if per_row else nb}
        writer = tile_req
    else:
        reader = tile_req
        writer = {"requests": sum(rows), "bytes_per_request": cfg["row_bytes"], "barriers": sum(1 for r in rows if r)}
    if fam == "tilize":
        compute = {"steps": nb, "tiles_per_step": wt, "kind": "tilize_block"}
    else:
        sw = sub_block_width(wt, elem)
        compute = {"steps": nb * (wt // sw), "tiles_per_step": sw, "kind": "untilize"}
    cap = cfg.get("cap") or wt
    if op == "untilize" and nb > 1 and not cfg.get("cap"):  # as build()
        p = PREFIX[(op, factory)]
        cap = (2 if factory == "MultiCore" else int(dtype_costs(c, p, elem).get(f"{p}.dfb_blocks", 1))) * wt
    entries = n if borrowed else cap
    return {"factory": factory, "grid": [GRID_X, GRID_Y], "cores_used": cores, "split": split, "row_cores": rc,
            "col_chunks": kc, "source": cfg["in_place"], "dest": cfg["out_place"],
            "per_core": {"blocks": nb, "tiles_per_block": wt, "tiles": n},
            "reader": reader, "compute": compute, "writer": writer,
            "dfb": {"in_entries": entries, "out_entries": entries, "entry_bytes": tile_bytes}}


def prefix_of(op, factory):
    """Cost prefix of (op, factory), from this module or a registered one."""
    m = _ext(op)
    return (m.PREFIX if m else PREFIX).get((op, factory))


def family_of(op):
    m = _ext(op)
    return (m.FAMILY if m else FAMILY).get(op)


def estimate(op, H, W, costs, dtype="bf16", mem="dram", multicore=True, trace=0, design=None):
    """Busiest core's span (cycles) of one tilize/untilize, with the factory, per-load results and the whole plan
    (every load, including those not run)."""
    factory, loads = plan(op, H, W, dtype, mem, multicore, design)
    for cfg, _ in loads:
        cfg["factory"] = factory
    active = sum(n for cfg, n in loads if "l1" in (cfg.get("in_place"), cfg.get("out_place")))
    if active:
        costs = dict(costs, _active_l1_cores=active)
    results = [(cfg, cores, run(build(op, cfg, costs), trace)) for cfg, cores in loads
               if cfg["blocks"] and not cfg.get("skip")]
    span = max(r["makespan"] for _, _, r in results)
    out = {"factory": factory, "span": span, "loads": results, "plan": loads}
    # Global bound: all cores' NoC traffic to / from L1 shares the network (<p>.l1_gbw bytes/cycle, fitted where L1
    # runs exist; ttsim-perf times it): with many cores moving L1 data at once the op cannot finish sooner.
    p = prefix_of(op, factory)
    if p:
        costs = dtype_costs(costs, p, ELEM[dtype])
    gbw = costs.get(f"{p}.l1_gbw", 0.0) if p else 0.0
    if gbw:
        total = sum(n * (cfg.get("tiles") or cfg["blocks"] * cfg["wt"]) * (TILE * TILE * ELEM[dtype]) *
                    ((cfg["in_place"] == "l1") + (cfg["out_place"] == "l1")) for cfg, n in loads)
        if total:
            bound = costs.get(f"{p}.start_reader", 0.0) + total / gbw + costs.get(f"{p}.drain", 0.0)
            out["l1_bound"] = bound
            out["span"] = max(span, bound)
    # Whole-op DRAM bound, only where the profile assumes a DRAM bandwidth and opts in (dram.bound_layout: the
    # simulators do not time DRAM, so craq-sim / ttsim-perf have neither): every core's DRAM reads and writes share
    # dram.total_bytes_per_cycle x dram.efficiency, after one DRAM read latency.
    bw = costs.get("dram.total_bytes_per_cycle", 0.0) * costs.get("dram.efficiency", 1.0)
    if bw and costs.get("dram.bound_layout", 0.0) >= 0.5:
        nbytes = dram_bytes(op, H, W, ELEM[dtype], loads)
        if nbytes:
            bound = (costs.get(f"{p}.start_reader", 0.0) + costs.get("noc.read_latency.dram", 0.0) + nbytes / bw
                     + costs.get(f"{p}.drain", 0.0))
            out["dram_bound"], out["dram_bytes"] = bound, nbytes
            out["span"] = max(out["span"], bound)
    return out


def dram_bytes(op, H, W, elem, loads):
    """Bytes the whole op reads from and writes to DRAM: per DRAM side, the tile-padded tensor for a tiled side and the
    logical rows for a row-major side (tilize: rows in, tiles out; untilize: tiles in, rows out; TILE sharding ops: the
    interleaved side's tiles); the other ops move the logical tensor (a lower bound of their traffic)."""
    cfg = loads[0][0]
    tiled = math.ceil(H / TILE) * math.ceil(W / TILE) * TILE * TILE * elem
    rows = H * W * elem
    fam = family_of(op)
    side = {"in": rows, "out": rows}
    if op in SHARD_OPS and not cfg.get("rm"):  # TILE sharding: the interleaved side holds whole tiles
        side = {"in": tiled, "out": tiled}
    elif fam == "tilize":
        side["out"] = tiled
    elif fam == "untilize":
        side["in"] = tiled
    return (side["in"] if cfg.get("in_place") == "dram" else 0) + (side["out"] if cfg.get("out_place") == "dram" else 0)
