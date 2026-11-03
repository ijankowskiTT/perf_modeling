"""Fixed-schedule finite-buffer evaluation of one cluster's reader -> compute -> writer pipeline.

This is a recurrence (max-plus) evaluation, not a simulator: every thread runs a FIXED program derived
from the kernels (reader/writer counter-major batch walk, compute chunks), every ring is the DFB's STRIDED
layout (tile i of the cluster lives in tile counter i mod max(P, C), capacity entries_per_thread per counter),
and the waits are data dependencies (an entry pushed, a slot popped) and NoC barriers. Apart from the NoC
completion queue below there is no arbitration between threads; other shared bandwidth is a bound (bounds.py).

NoC barriers are cluster-wide, as in the Quasar firmware: the issued counters (noc_reads_num_issued, the
nonposted-write counters) are shared globals of all DM threads and the NIU response counters are per cluster,
so a read barrier returns only at a moment when NO read of any reader thread is outstanding (writes alike;
with noc.barrier_all_kinds = 1, as measured on ttsim-perf, no transfer of either kind). Each transfer is logged
as its outstanding interval (issue end, completion); a barrier is released at the first quiet moment at or
after it is reached. Timed transfers of one kind also complete one after another per cluster
(noc.read_service / noc.write_service cycles each, per placement), as the RD_BAR zone shows on ttsim-perf: the
wait after the last issue grows with the reads in flight cluster-wide. With noc.rw_serialized = 1 reads and
writes share that queue. Queue order is evaluation order, which the barriers keep close to time order. When
transfers take no time (latency 0, no service, no per-thread bandwidth: DRAM on ttsim-perf) nothing is
outstanding and a barrier returns at once.

Times are in cycles of the cost profile's clock. The evaluation order is "advance every thread as far as
its dependencies are known"; when nothing can advance, the parked barrier with the earliest quiet moment is
released (every other pending event depends on some barrier, so none can happen earlier). A pass with no
progress and no parked barrier means the schedule deadlocks (an invalid program).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf


@dataclass
class Ring:
    """One STRIDED DFB: tile i -> counter i % counters; capacity entries per counter."""

    name: str
    counters: int
    capacity: int  # entries per counter (entries_per_thread); borrowed rings hold the whole share
    n: int
    push_t: list = field(default_factory=list)
    pop_t: list = field(default_factory=list)

    def __post_init__(self):
        self.push_t = [None] * self.n
        self.pop_t = [None] * self.n

    def slot_free_time(self, tiles):
        """Earliest time the slots for `tiles` are free: the entry `capacity` positions earlier in the same
        counter must be popped. None if not yet known."""
        t = 0.0
        step = self.capacity * self.counters
        for i in tiles:
            j = i - step
            if j >= 0:
                p = self.pop_t[j]
                if p is None:
                    return None
                if p > t:
                    t = p
        return t

    def data_time(self, tiles):
        t = 0.0
        for i in tiles:
            p = self.push_t[i]
            if p is None:
                return None
            if p > t:
                t = p
        return t


@dataclass
class ThreadStats:
    role: str
    index: int
    start: float = 0.0
    end: float = 0.0
    busy: float = 0.0  # own service time (instructions, DFB ops, issue)
    wait_data: float = 0.0  # blocked on an input entry (starvation)
    wait_space: float = 0.0  # blocked on a free slot (backpressure)
    wait_transfer: float = 0.0  # blocked in a barrier on outstanding NoC transfers
    batches: int = 0
    tiles: int = 0

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def reader_batches(n_tiles, threads, thread_id, dm_batch, num_tcs):
    """The reader/writer kernel's batch walk (reader_no_bcast_dfb.cpp / writer_no_bcast_dfb.cpp): lists of tile
    indices, one list per reserve/push (or wait/pop)."""
    out = []
    if dm_batch == 1:
        for k in range(thread_id, n_tiles, threads):
            out.append([k])
        return out
    tile_step = num_tcs * threads
    round_span = dm_batch * tile_step
    batch_span = (dm_batch - 1) * tile_step
    full_limit = n_tiles - batch_span if n_tiles > batch_span else 0
    round_base = thread_id
    while round_base < n_tiles:
        first = round_base
        c = 0
        while c < num_tcs and first < n_tiles:
            n = dm_batch
            if first >= full_limit:
                n = (n_tiles - first + tile_step - 1) // tile_step
            out.append([first + j * tile_step for j in range(n)])
            c += 1
            first += threads
        round_base += round_span
    return out


def compute_chunks(n_tiles, threads, thread_id, tiles_per_cycle):
    """Compute thread c consumes {c, c+C, ...} in chunks of tiles_per_cycle, last chunk short."""
    mine = list(range(thread_id, n_tiles, threads))
    return [mine[i : i + tiles_per_cycle] for i in range(0, len(mine), tiles_per_cycle)]


def num_tcs(producers, consumers, is_producer):
    if is_producer:
        return consumers // producers if consumers >= producers else 1
    return producers // consumers if producers >= consumers else 1


def reader_batch_fixed(c, dm_batch, n_in_moving):
    """Per-batch fixed reader cost; profiles give it for two moving inputs, one moving input takes a fraction."""
    f = c["rd.fixed_b1"] if dm_batch == 1 else c["rd.fixed"]
    return f if n_in_moving == 2 else f * c.get("rd.one_operand_frac", 0.6)


def compute_scale(c, C):
    """Compute service scale: several Neos share the cluster (C > 1)."""
    return c.get("compute.multi_neo_scale", 1.0) if C > 1 else 1.0


class Deadlock(Exception):
    pass


class TransferLog:
    """Outstanding intervals (issue end, completion) of the transfers one barrier waits for, across all DM threads
    of a cluster."""

    def __init__(self):
        self.iv = []

    def add(self, a, b):
        if b > a:
            self.iv.append((a, b))

    def quiet_at(self, t):
        """Earliest time >= t at which no logged transfer is outstanding."""
        while True:
            m = t
            for a, b in self.iv:
                if a <= t and b > m:
                    m = b
            if m == t:
                return t
            t = m

    def prune(self, before):
        """Forget transfers completed before `before` (no later barrier is reached that early)."""
        self.iv = [x for x in self.iv if x[1] > before]


def evaluate(cfg, costs, n_tiles, trace_limit=0):
    """Evaluate one cluster processing n_tiles output tiles.

    cfg: dict with R, C, W, dm_batch, tiles_per_cycle, entries_per_thread, in_moves (list of bool for a, b),
         out_moves (bool), tile_bytes (per operand list [a, b, out]).
    costs: resolved cost dict (cycles), see profiles/costs/*.json for the keys.
    Returns dict: makespan, per-thread stats, per-role aggregates.
    """
    R, C, W = cfg["R"], cfg["C"], cfg["W"]
    B, BC, K = cfg["dm_batch"], cfg["tiles_per_cycle"], cfg["entries_per_thread"]
    in_moves = cfg["in_moves"]
    out_moves = cfg["out_moves"]
    n_in_moving = sum(1 for m in in_moves if m)
    reader_active = n_in_moving > 0
    if not reader_active:
        R = 1
    if not out_moves:
        W = 1

    in_counters = max(R, C)
    out_counters = max(C, W)
    # A borrowed ring is the resident shard: every entry exists up front, so its capacity never binds.
    in_cap = K if reader_active else n_tiles + 1
    out_cap = K if out_moves else n_tiles + 1
    rin = Ring("in", in_counters, in_cap, n_tiles)
    rout = Ring("out", out_counters, out_cap, n_tiles)

    c = costs
    t0_cu = c["start.compute"]
    # DM kernel start: a role base plus a cost per tile counter the thread initializes (num_tcs per ring it
    # touches), plus a penalty for threads placed on DM slots 5-6 (the larger role takes the first slots).
    rd_rings = n_in_moving if reader_active else 2
    rd_tcs_ring = num_tcs(R, C, True)
    rd_tcs = rd_tcs_ring * rd_rings
    wr_tcs = num_tcs(C, W, False)
    order = (["reader"] * R + ["writer"] * W) if R >= W else (["writer"] * W + ["reader"] * R)
    late_slots = c.get("start.dm_early_slots", 4)
    slot_of = {"reader": [], "writer": []}
    for slot, role in enumerate(order):
        slot_of[role].append(slot)

    def dm_start(role, t):
        if role == "reader":
            base = c["start.reader"] + c.get("start.reader_per_tc", 0.0) * rd_tcs
        else:
            base = c["start.writer"] + c.get("start.writer_per_tc", 0.0) * wr_tcs
        return base + (c.get("start.dm_late", 0.0) if slot_of[role][t] >= late_slots else 0.0)

    t0_dm = dm_start("reader", 0)
    per_dm = 0.0
    lat_r, lat_w = c.get("noc.read_latency", 0.0), c.get("noc.write_latency", 0.0)
    bw_thr = c.get("noc.thread_bytes_per_cycle", INF) or INF
    q_max = int(c.get("noc.max_inflight", 0) or 0)
    # cluster completion service of one read per moving input (in issue order), and of one write
    rd_svc = c.get("_rd_service") or (c.get("noc.read_service", 0.0),)
    wr_svc = c.get("noc.write_service", 0.0)
    done_q = {"rd": -INF, "wr": -INF}  # last completion in each kind's cluster queue
    wr_key = "rd" if c.get("noc.rw_serialized", 0.0) >= 0.5 else "wr"
    tb_in = cfg["tile_bytes"][0]
    tb_out = cfg["tile_bytes"][2]

    threads = []  # (stats, generator-state)

    # ---- reader threads ----
    reader_progs = []
    if reader_active:
        tcs = num_tcs(R, C, True)
        for t in range(R):
            st = ThreadStats("reader", t, start=dm_start("reader", t))
            reader_progs.append((st, reader_batches(n_tiles, R, t, B, tcs)))
    else:
        # Borrowed inputs: one publish per counter at kernel start; entries are resident.
        st = ThreadStats("reader", 0, start=t0_dm)
        # one reserve_back + push_back per tile counter of each borrowed ring
        pub = t0_dm + c.get("rd.borrow_publish", 0.0) * min(C, max(n_tiles, 1)) * 2
        for i in range(n_tiles):
            rin.push_t[i] = pub
        st.end = pub
        st.busy = pub - t0_dm
        threads.append(st)

    # ---- writer threads ----
    writer_progs = []
    if out_moves:
        tcs = num_tcs(C, W, False)
        for t in range(W):
            st = ThreadStats("writer", t, start=dm_start("writer", t))
            writer_progs.append((st, reader_batches(n_tiles, W, t, B, tcs)))
    else:
        st = ThreadStats("writer", 0, start=dm_start("writer", 0))
        # Borrowed output: the writer pops each counter's whole share once it is computed.
        writer_progs.append((st, None))

    # ---- compute threads ----
    compute_progs = []
    for t in range(C):
        st = ThreadStats("compute", t, start=t0_cu)
        compute_progs.append((st, compute_chunks(n_tiles, C, t, BC)))

    neo_scale = compute_scale(c, C)

    def lin(prefix, n):
        f = c[prefix + ".fixed"]
        if n == 1 and (prefix + ".fixed_n1") in c:
            f = c[prefix + ".fixed_n1"]
        return (f + c[prefix + ".per_tile"] * n) * neo_scale

    # Per-thread mutable state; "bar" holds a batch parked at its NoC barrier.
    rd_state = [{"pc": 0, "t": st.start, "bar": None} for st, _ in reader_progs]
    wr_state = [{"pc": 0, "t": st.start, "bar": None} for st, _ in writer_progs]
    rlog = TransferLog()
    wlog = rlog if c.get("noc.barrier_all_kinds", 0.0) >= 0.5 else TransferLog()
    # Untimed transfers leave nothing outstanding: barriers return at once, no parking needed.
    rd_timed = lat_r > 0 or bw_thr != INF or any(rd_svc)
    wr_timed = lat_w > 0 or bw_thr != INF or wr_svc > 0
    if wlog is rlog:  # one quiet condition: an untimed kind still waits for the other kind's transfers
        rd_timed = wr_timed = rd_timed or wr_timed
    # Compute: unpack+math stage (UM) and pack stage (P), DST half-sync = 2 sections.
    cu_state = [{"um_pc": 0, "p_pc": 0, "um_t": st.start, "p_t": st.start + c.get("start.pack_offset", 0.0),
                 "um_end": [], "p_end": []} for st, _ in compute_progs]

    rd_fixed = reader_batch_fixed(c, B, n_in_moving)
    wr_fixed = c["wr.fixed_b1"] if B == 1 else c["wr.fixed"]
    lat_cr = c.get("dfb.credit_latency", 0.0)
    rd_pub = min(1.0, max(0.0, c.get("rd.publish_frac", 0.3)))
    wr_rel = min(1.0, max(0.0, c.get("wr.release_frac", 0.5)))

    def step_reader(i):
        st, prog = reader_progs[i]
        s = rd_state[i]
        if s["bar"] is not None or s["pc"] >= len(prog):
            return False
        tiles = prog[s["pc"]]
        free = rin.slot_free_time(tiles)
        if free is None:
            return False
        n = len(tiles)
        t = s["t"]
        t_rsv = max(t, free)
        st.wait_space += t_rsv - t
        # per-batch fixed cost (loop, reserve(s), barrier, push(es)): the part before the first request is the
        # reserve side, the rest follows the barrier and delays publication.
        warm = c.get("rd.first_batch_extra", 0.0) if st.batches == 0 else 0.0
        if st.batches < rd_tcs_ring:
            warm += c.get("rd.tc_first_touch", 0.0)  # first batch on each of the thread's tile counters
        t = t_rsv + rd_fixed * (1.0 - rd_pub) + warm
        busy = rd_fixed * (1.0 - rd_pub) + warm
        reqs = n * n_in_moving
        comp_prev = -INF
        issue = t
        completes = []
        for j in range(reqs):
            if q_max and j >= q_max:
                issue = max(issue, completes[j - q_max])
            issue_end = issue + c["rd.issue"]
            comp = max(issue_end + lat_r, comp_prev + (tb_in / bw_thr if bw_thr != INF else 0.0))
            svc = rd_svc[(j // n) % len(rd_svc)]  # the kernel reads all of a's tiles, then all of b's
            if svc:
                comp = max(comp, done_q["rd"] + svc)
                done_q["rd"] = comp
            completes.append(comp)
            rlog.add(issue_end, comp)
            comp_prev = comp
            issue = issue_end
        busy += c["rd.issue"] * reqs
        st.busy += busy
        s["bar"] = (issue, tiles)
        if not rd_timed:
            finish_reader(i, issue)
        return True

    def finish_reader(i, t_bar):
        """Release reader i's parked batch from its read barrier at t_bar, then publish."""
        st, prog = reader_progs[i]
        s = rd_state[i]
        issue, tiles = s["bar"]
        s["bar"] = None
        n = len(tiles)
        st.wait_transfer += t_bar - issue
        t = t_bar + rd_fixed * rd_pub
        busy = 0.0
        busy += rd_fixed * rd_pub
        for k in tiles:
            rin.push_t[k] = t + lat_cr
        st.busy += busy
        st.batches += 1
        st.tiles += n
        s["t"] = t
        s["pc"] += 1
        if s["pc"] == len(prog):
            st.end = t

    def step_writer(i):
        st, prog = writer_progs[i]
        s = wr_state[i]
        if prog is None:
            # borrowed output: wait for every tile, pop per counter
            if s["pc"]:
                return False
            ready = rout.data_time(range(n_tiles))
            if ready is None:
                return False
            t = max(s["t"], ready)
            st.wait_data += t - s["t"]
            t += c.get("wr.borrow_pop", 0.0) * min(C, max(n_tiles, 1))
            st.busy += t - max(s["t"], ready)
            for k in range(n_tiles):
                rout.pop_t[k] = t
            s["t"] = t
            s["pc"] = 1
            st.end = t
            return True
        if s["bar"] is not None or s["pc"] >= len(prog):
            return False
        tiles = prog[s["pc"]]
        ready = rout.data_time(tiles)
        if ready is None:
            return False
        n = len(tiles)
        t = max(s["t"], ready)
        st.wait_data += t - s["t"]
        warm = c.get("wr.first_batch_extra", 0.0) if st.batches == 0 else 0.0
        if st.batches < wr_tcs:
            warm += c.get("wr.tc_first_touch", 0.0)
        t += wr_fixed * (1.0 - wr_rel) + warm
        busy = wr_fixed * (1.0 - wr_rel) + warm
        comp_prev = -INF
        issue = t
        completes = []
        for j in range(n):
            if q_max and j >= q_max:
                issue = max(issue, completes[j - q_max])
            issue_end = issue + c["wr.issue"]
            comp = max(issue_end + lat_w, comp_prev + (tb_out / bw_thr if bw_thr != INF else 0.0))
            if wr_svc:
                comp = max(comp, done_q[wr_key] + wr_svc)
                done_q[wr_key] = comp
            completes.append(comp)
            wlog.add(issue_end, comp)
            comp_prev = comp
            issue = issue_end
        busy += c["wr.issue"] * n
        st.busy += busy
        s["bar"] = (issue, tiles)
        if not wr_timed:
            finish_writer(i, issue)
        return True

    def finish_writer(i, t_bar):
        """Release writer i's parked batch from its write barrier at t_bar, then free the output slots."""
        st, prog = writer_progs[i]
        s = wr_state[i]
        issue, tiles = s["bar"]
        s["bar"] = None
        n = len(tiles)
        st.wait_transfer += t_bar - issue
        t = t_bar + wr_fixed * wr_rel
        st.busy += wr_fixed * wr_rel
        for k in tiles:
            rout.pop_t[k] = t + lat_cr
        st.batches += 1
        st.tiles += n
        s["t"] = t
        s["pc"] += 1
        if s["pc"] == len(prog):
            st.end = t

    def step_compute(i):
        st, chunks = compute_progs[i]
        s = cu_state[i]
        progressed = False
        # UM stage: needs inputs pushed and a free DST section (pack of chunk k-2 done).
        while s["um_pc"] < len(chunks):
            k = s["um_pc"]
            if k >= 2 and len(s["p_end"]) < k - 1:
                break
            tiles = chunks[k]
            ready = rin.data_time(tiles)
            if ready is None:
                break
            dst_free = s["p_end"][k - 2] if k >= 2 else 0.0
            start = max(s["um_t"], ready, dst_free)
            st.wait_data += max(0.0, ready - max(s["um_t"], dst_free))
            n = len(tiles)
            u = lin("cu", n) + (c.get("cu.first_chunk_extra", 0.0) if k == 0 else 0.0)
            m = lin("cm", n)
            end = start + max(u, m)
            for kk in tiles:
                rin.pop_t[kk] = start + u + lat_cr  # the unpack TRISC pops once its unpack loop is done
            st.busy += max(u, m)
            s["um_t"] = end
            s["um_end"].append(end)
            s["um_pc"] += 1
            progressed = True
        # P stage: needs UM of this chunk and free output slots.
        while s["p_pc"] < s["um_pc"]:
            k = s["p_pc"]
            tiles = chunks[k]
            free = rout.slot_free_time(tiles)
            if free is None:
                break
            start = max(s["p_t"], s["um_end"][k], free)
            st.wait_space += max(0.0, free - max(s["p_t"], s["um_end"][k]))
            n = len(tiles)
            end = start + lin("cp", n)
            for kk in tiles:
                rout.push_t[kk] = end + lat_cr
            s["p_t"] = end
            s["p_end"].append(end)
            s["p_pc"] += 1
            st.tiles += n
            st.batches += 1
            progressed = True
        if s["p_pc"] == len(chunks) and chunks:
            st.end = s["p_end"][-1]
        elif not chunks:
            st.end = st.start
        return progressed

    pending = True
    while pending:
        progressed = False
        for i in range(len(reader_progs)):
            while step_reader(i):
                progressed = True
        for i in range(len(compute_progs)):
            if step_compute(i):
                progressed = True
        for i in range(len(writer_progs)):
            while step_writer(i):
                progressed = True
        done = (
            all(s["pc"] >= len(p) for s, (_, p) in zip(rd_state, reader_progs))
            and all(s["p_pc"] >= len(ch) for s, (_, ch) in zip(cu_state, compute_progs))
            and all((s["pc"] >= (len(p) if p is not None else 1)) for s, (_, p) in zip(wr_state, writer_progs))
        )
        pending = not done
        if pending and not progressed:
            # Release the parked barrier with the earliest quiet moment (ties: all of them).
            parked = [(rlog.quiet_at(s["bar"][0]), 0, i) for i, s in enumerate(rd_state) if s["bar"] is not None]
            parked += [(wlog.quiet_at(s["bar"][0]), 1, i) for i, s in enumerate(wr_state) if s["bar"] is not None]
            if not parked:
                raise Deadlock("schedule makes no progress: the configuration would hang")
            t_min = min(p[0] for p in parked)
            for t_q, kind, i in parked:
                if t_q == t_min:
                    (finish_reader if kind == 0 else finish_writer)(i, t_q)
            low = min([s["bar"][0] if s["bar"] is not None else s["t"] for s in rd_state + wr_state] or [t_min])
            rlog.prune(low)
            if wlog is not rlog:
                wlog.prune(low)

    for (st, p), s in zip(reader_progs, rd_state):
        if not p:
            st.end = st.start
        threads.append(st)
    for (st, _), s in zip(compute_progs, cu_state):
        threads.append(st)
    for (st, p), s in zip(writer_progs, wr_state):
        if p is not None and not p:
            st.end = st.start
        threads.append(st)

    makespan = max(st.end for st in threads) + c.get("end.drain", 0.0)
    roles = {}
    for st in threads:
        r = roles.setdefault(st.role, {"threads": 0, "busy": 0.0, "wait_data": 0.0, "wait_space": 0.0,
                                       "wait_transfer": 0.0, "end_max": 0.0, "start_min": INF})
        r["threads"] += 1
        r["busy"] += st.busy
        r["wait_data"] += st.wait_data
        r["wait_space"] += st.wait_space
        r["wait_transfer"] += st.wait_transfer
        r["end_max"] = max(r["end_max"], st.end)
        r["start_min"] = min(r["start_min"], st.start)
    for r in roles.values():
        span = makespan
        r["utilization"] = r["busy"] / (r["threads"] * span) if span > 0 else 0.0
    out = {"makespan": makespan, "roles": roles, "first_output": rout.push_t[0] if n_tiles else None}
    if trace_limit:
        out["threads"] = [st.as_dict() for st in threads[:trace_limit]]
    return out
