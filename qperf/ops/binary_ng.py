"""binary_ng Quasar-native (kernels_qsr/, no broadcast) as an engine Program.

Readers and writers run the kernels' counter-major batch walk (reader_no_bcast_dfb.cpp / writer_no_bcast_dfb.cpp);
compute thread c takes tiles {c, c+C, ...} in chunks of tiles_per_cycle, each chunk an unpack+math stage then a pack
stage with a half-sync DST of 2 sections. Rings are the DFB's STRIDED layout. A borrowed ring is the resident shard:
the reader publishes each counter once, the writer pops it once.

Hypothetical schedule variants (not in the pinned kernels):
- barriers = "per_thread": each DM thread's barrier waits only for its own transfers (transaction IDs); the
  completion queue stays shared, since the NIU still serves the cluster's transfers one after another.
- work_split = "dynamic": the DM threads of a role take batches from one shared queue (in tile order) instead of a
  fixed stride; the idle thread that is free first claims the next batch and pays dq.claim cycles for it.
"""

from __future__ import annotations

from .. import dram
from ..engine import INF, DMBatch, Group, Program, Ring, Server, ThreadStats, Thread, TransferLog
from ..pipeline import compute_chunks, compute_scale, num_tcs, reader_batch_fixed, reader_batches


def build(cfg, c, n_tiles):
    """cfg: R, C, W, dm_batch, tiles_per_cycle, entries_per_thread, in_moves [a, b], out_moves, tile_bytes, and
    optionally barriers / work_split. c: resolved costs (cycles), see profiles/costs/*.json."""
    R, C, W = cfg["R"], cfg["C"], cfg["W"]
    B, BC, K = cfg["dm_batch"], cfg["tiles_per_cycle"], cfg["entries_per_thread"]
    in_moves, out_moves = cfg["in_moves"], cfg["out_moves"]
    n_in = sum(1 for m in in_moves if m)
    reader_active = n_in > 0
    if not reader_active:
        R = 1
    if not out_moves:
        W = 1
    rin = Ring("in", max(R, C), K if reader_active else n_tiles + 1, n_tiles)
    rout = Ring("out", max(C, W), K if out_moves else n_tiles + 1, n_tiles)

    # DM kernel start: a role base plus a cost per tile counter the thread initializes, plus a penalty for threads on
    # DM slots 5-6 (the larger role takes the first slots).
    rd_tcs_ring = num_tcs(R, C, True)
    rd_tcs = rd_tcs_ring * (n_in if reader_active else 2)
    wr_tcs = num_tcs(C, W, False)
    order = (["reader"] * R + ["writer"] * W) if R >= W else (["writer"] * W + ["reader"] * R)
    slot_of = {"reader": [], "writer": []}
    for slot, role in enumerate(order):
        slot_of[role].append(slot)
    late_slots = c.get("start.dm_early_slots", 4)

    def dm_start(role, t):
        if role == "reader":
            base = c["start.reader"] + c.get("start.reader_per_tc", 0.0) * rd_tcs
        else:
            base = c["start.writer"] + c.get("start.writer_per_tc", 0.0) * wr_tcs
        return base + (c.get("start.dm_late", 0.0) if slot_of[role][t] >= late_slots else 0.0)

    lat_r, lat_w = c.get("noc.read_latency", 0.0), c.get("noc.write_latency", 0.0)
    bw = c.get("noc.thread_bytes_per_cycle", INF) or INF
    q_max = int(c.get("noc.max_inflight", 0) or 0)
    rd_svc = c.get("_rd_service") or (c.get("noc.read_service", 0.0),)
    wr_svc = c.get("noc.write_service", 0.0)
    wr_q = "rd" if c.get("noc.rw_serialized", 0.0) >= 0.5 else "wr"
    tb_in, tb_out = cfg["tile_bytes"][0], cfg["tile_bytes"][2]
    gap_in = tb_in / bw if bw != INF else 0.0
    gap_out = tb_out / bw if bw != INF else 0.0
    lat_cr = c.get("dfb.credit_latency", 0.0)
    rd_pub = min(1.0, max(0.0, c.get("rd.publish_frac", 0.3)))
    wr_rel = min(1.0, max(0.0, c.get("wr.release_frac", 0.5)))
    rd_fixed = reader_batch_fixed(c, B, n_in)
    wr_fixed = c["wr.fixed_b1"] if B == 1 else c["wr.fixed"]
    per_thread = cfg.get("barriers", "cluster") == "per_thread"
    dynamic = cfg.get("work_split", "static") == "dynamic"
    # DRAM channels (dram.model = channels, with the op's cluster layout from the API, cfg["dram"]): this cluster
    # stands for every cluster (bases: each one's first tile) and its DRAM operands' requests go to the channels; an
    # L1 operand keeps its own latency
    dd = cfg.get("dram") if dram.enabled(c) else None
    chans, sides, lat_in = None, [None, None, None], [lat_r, lat_r]
    if dd:
        chans = dram.Channels(c)
        cores = [dram.worker_noc(k) for k in range(len(dd["bases"]))]
        sides = [dram.Side(chans, i < 2, cores, dd["bases"]) if dd["placements"][i] == "dram" else None
                 for i in range(3)]
        lat_in = dd["lat_in"]
    moving = [i for i in range(2) if in_moves[i]]

    # Barrier logs: one per cluster (firmware; reads and writes share one with noc.barrier_all_kinds), or per thread.
    rd_timed = lat_r > 0 or bw != INF or any(rd_svc)
    wr_timed = lat_w > 0 or bw != INF or wr_svc > 0
    logs = []

    def new_log():
        logs.append(TransferLog())
        return len(logs) - 1

    if per_thread:
        rd_log = [new_log() for _ in range(R if reader_active else 0)]
        wr_log = [new_log() for _ in range(W if out_moves else 0)]
    else:
        shared = new_log()
        wlog = shared if c.get("noc.barrier_all_kinds", 0.0) >= 0.5 else new_log()
        if wlog == shared:  # one quiet condition: an untimed kind still waits for the other kind's transfers
            rd_timed = wr_timed = rd_timed or wr_timed
        rd_log, wr_log = [shared] * R, [wlog] * W

    stats, groups = [], []
    # ---- readers
    rd_threads = []
    if reader_active:
        for t in range(R):
            st = ThreadStats("reader", t, start=dm_start("reader", t))
            steps = []
            for b in reader_batches(n_tiles, R, t, B, rd_tcs_ring):
                n = len(b)
                grp = [(n, lat_in[moving[i]], gap_in, rd_svc[i % len(rd_svc)], "rd") for i in range(n_in)]
                if dd:
                    grp = [g + ((dram.xfer(sides[moving[i]], [sides[moving[i]].counts(k) for k in b], tb_in),)
                                if sides[moving[i]] else ()) for i, g in enumerate(grp)]
                steps.append(DMBatch(b, "space", rin, rd_fixed * (1.0 - rd_pub), rd_fixed * rd_pub,
                                     (c.get("rd.first_batch_extra", 0.0), c.get("rd.tc_first_touch", 0.0), rd_tcs_ring),
                                     c["rd.issue"], grp, rd_log[t], rd_timed, "push", lat_cr, q_max))
            rd_threads.append(Thread(st, steps))
        rd_group = Group(rd_threads, dm=True, claim=c.get("dq.claim", 0.0) if dynamic else 0.0)
        if dynamic:
            rd_group.work = sorted((s for th in rd_threads for s in th.steps), key=lambda s: s.tiles[0])
            for th in rd_threads:
                th.steps = []
    else:
        # Borrowed inputs: one reserve_back + push_back per tile counter of each borrowed ring at kernel start.
        st = ThreadStats("reader", 0, start=dm_start("reader", 0))
        pub = c.get("rd.borrow_publish", 0.0) * min(C, max(n_tiles, 1)) * 2
        stats.append(st)
        groups.append(Group([Thread(st, [("busy", pub, True), ("push", rin, range(n_tiles), 0.0)])]))
        rd_group = None

    # ---- compute: per Neo an unpack+math thread and a pack thread sharing one ThreadStats; DST = 2 sections
    cu_threads, cu_stats = [], []
    sc = compute_scale(c, C)

    def lin(prefix, n):
        f = c[prefix + ".fixed"]
        if n == 1 and (prefix + ".fixed_n1") in c:
            f = c[prefix + ".fixed_n1"]
        return (f + c[prefix + ".per_tile"] * n) * sc

    for t in range(C):
        st = ThreadStats("compute", t, start=c["start.compute"])
        chunks = compute_chunks(n_tiles, C, t, BC)
        dst = Ring("dst", 1, 2, len(chunks))
        um, pk = [], []
        for k, tiles in enumerate(chunks):
            n = len(tiles)
            u = lin("cu", n) + (c.get("cu.first_chunk_extra", 0.0) if k == 0 else 0.0)
            m = lin("cm", n)
            um += [("wait", "space", dst, [k], False), ("wait", "data", rin, tiles, True), ("busy", u, True),
                   ("pop", rin, tiles, lat_cr), ("busy", max(u, m) - u, True), ("push", dst, [k], 0.0)]
            pk += [("wait", "data", dst, [k], False), ("wait", "space", rout, tiles, True), ("busy", lin("cp", n), False),
                   ("push", rout, tiles, lat_cr), ("pop", dst, [k], 0.0), ("count", n, 1)]
        cu_threads += [Thread(st, um, sets_end=False), Thread(st, pk, t0=st.start + c.get("start.pack_offset", 0.0))]
        cu_stats.append(st)

    # ---- writers
    wr_threads = []
    if out_moves:
        for t in range(W):
            st = ThreadStats("writer", t, start=dm_start("writer", t))
            so = sides[2]
            steps = [DMBatch(b, "data", rout, wr_fixed * (1.0 - wr_rel), wr_fixed * wr_rel,
                             (c.get("wr.first_batch_extra", 0.0), c.get("wr.tc_first_touch", 0.0), wr_tcs),
                             c["wr.issue"], [(len(b), lat_w, gap_out, wr_svc, wr_q)
                                             + ((dram.xfer(so, [so.counts(k) for k in b], tb_out),) if so else ())],
                             wr_log[t], wr_timed, "pop", lat_cr, q_max)
                     for b in reader_batches(n_tiles, W, t, B, wr_tcs)]
            wr_threads.append(Thread(st, steps))
    else:
        # Borrowed output: the writer pops each counter's whole share once it is computed.
        st = ThreadStats("writer", 0, start=dm_start("writer", 0))
        allt = range(n_tiles)
        wr_threads.append(Thread(st, [("wait", "data", rout, allt, True),
                                      ("busy", c.get("wr.borrow_pop", 0.0) * min(C, max(n_tiles, 1)), True),
                                      ("pop", rout, allt, 0.0)]))
    wr_group = Group(wr_threads, dm=True, claim=c.get("dq.claim", 0.0) if dynamic else 0.0)
    if dynamic and out_moves:
        wr_group.work = sorted((s for th in wr_threads for s in th.steps), key=lambda s: s.tiles[0])
        for th in wr_threads:
            th.steps = []

    if rd_group is not None:
        groups.append(rd_group)
        stats += [th.stats for th in rd_threads]
    groups += [Group(cu_threads), wr_group]
    stats += cu_stats + [th.stats for th in wr_threads]
    return Program(groups=groups, logs=logs, queues={"rd": Server(), "wr": Server()}, stats=stats,
                   drain=c.get("end.drain", 0.0), first_output=rout, dram=chans)
