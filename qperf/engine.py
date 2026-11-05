"""Generic fixed-schedule evaluator: an op description (qperf/ops/*) builds a Program of rings and threads, and run()
evaluates it as a max-plus recurrence. Nothing here knows about a particular op.

A Program has:
- rings: STRIDED dataflow buffers (tile i -> counter i mod counters, `capacity` entries per counter);
- logs: transfer logs a NoC barrier waits on (one per cluster and kind, or one per thread);
- queues: per-cluster completion queues that serve timed transfers one after another, in time order;
- groups: threads in evaluation order. A DM group may own a shared work queue (`work`): its idle threads claim the
  next batch, paying `claim` cycles.

Thread steps:
- ("wait", "space"|"data"|"popped", ring, tiles, counted)   until slots are free / entries pushed / entries popped
                                                    ("popped": a producer's kernel exits only once its DFB is drained)
- ("busy", cycles, counted)                         own service time
- ("push"|"pop", ring, tiles, delay)                publish entries / free slots, visible after `delay`
- ("count", tiles, batches)
- DMBatch                                           wait, issue NoC transfers, barrier, publish or release

Evaluation advances every thread as far as its known dependencies allow; a DM batch parks at its barrier. When nothing
can advance, the parked barrier with the earliest quiet moment is released (every other pending event depends on some
barrier). No progress and no parked barrier is a deadlock: the configuration would hang.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INF = math.inf


class Deadlock(Exception):
    pass


def _latest(times, tiles):
    """Latest of times[i] over tiles, or None if one is not known yet."""
    t = 0.0
    for i in tiles:
        p = times[i]
        if p is None:
            return None
        if p > t:
            t = p
    return t


@dataclass
class Ring:
    """One STRIDED DFB: tile i -> counter i % counters; capacity entries per counter."""

    name: str
    counters: int
    capacity: int
    n: int
    push_t: list = field(default_factory=list)
    pop_t: list = field(default_factory=list)

    def __post_init__(self):
        self.push_t = [None] * self.n
        self.pop_t = [None] * self.n

    def slot_free_time(self, tiles):
        """Earliest time the slots for `tiles` are free (the entry `capacity` positions earlier in the same counter
        is popped), or None if not yet known."""
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

    def popped_time(self, tiles):
        """Time the consumer has popped all of `tiles`, or None if not yet known."""
        return _latest(self.pop_t, tiles)

    def data_time(self, tiles):
        """Time all of `tiles` are pushed, or None if not yet known."""
        return _latest(self.push_t, tiles)


class TransferLog:
    """Outstanding intervals (issue end, completion) of the transfers a barrier waits for."""

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
        self.iv = [x for x in self.iv if x[1] > before]


class Server:
    """A completion queue: each transfer takes `svc` cycles of service and finishes no earlier than `ready`; it is
    served in the earliest free slot ending at or after `ready`, so the result does not depend on evaluation order."""

    def __init__(self):
        self.busy = []  # sorted, disjoint (start, end)

    def serve(self, ready, svc):
        s = ready - svc
        k = 0
        while k < len(self.busy) and self.busy[k][1] <= s:
            k += 1
        while k < len(self.busy) and self.busy[k][0] < s + svc - 1e-6:  # an exact fit fits, despite rounding
            s = max(s, self.busy[k][1])
            k += 1
        self.busy.insert(k, (s, s + svc))
        return s + svc

    def prune(self, before):
        self.busy = [x for x in self.busy if x[1] > before]


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


@dataclass
class DMBatch:
    """One DM batch: wait for slots (reader) or data (writer), fixed cost, NoC transfers, barrier, fixed cost, publish
    (push) or release (pop) the batch's entries.

    groups: [(count, latency, gap, svc, queue)] in issue order; gap = per-request bytes / thread bandwidth, svc = the
    completion queue's service per request (0 = untimed by the queue). post is the part of `fixed` after the barrier."""

    tiles: list
    wait: str  # "space" | "data"
    ring: Ring
    pre: float
    post: float
    warm: tuple  # (extra on the thread's first batch, extra on each of its first `tcs` batches, tcs)
    issue: float
    groups: list
    log: int
    timed: bool
    release: str  # "push" | "pop"
    delay: float
    max_inflight: int = 0
    dma: bool = False  # True: transfers queue on the thread's DMA engine, each taking `gap`, across batches


class Thread:
    def __init__(self, stats, steps, t0=None, sets_end=True):
        self.stats = stats
        self.steps = steps
        self.t = stats.start if t0 is None else t0
        self.pc = 0
        self.bar = None  # parked batch: (barrier arrival time, DMBatch)
        self.sets_end = sets_end
        self.ran = False
        self.dma_free = -INF  # when the thread's DMA engine finishes its last transfer (DMBatch.dma)

    @property
    def done(self):
        return self.bar is None and self.pc >= len(self.steps)


@dataclass
class Group:
    threads: list
    dm: bool = False
    work: list = None  # shared work queue of batches (dynamic split), or None
    claim: float = 0.0


@dataclass
class Program:
    groups: list
    logs: list
    queues: dict
    stats: list  # ThreadStats in output order
    drain: float = 0.0
    first_output: Ring = None


def _step(th, prog):
    """Run thread th's next step. True if it advanced (a parked batch counts), False if blocked or done."""
    if th.bar is not None or th.pc >= len(th.steps):
        return False
    s = th.steps[th.pc]
    st = th.stats
    if isinstance(s, DMBatch):
        return _start_batch(th, s, prog)
    op = s[0]
    if op == "wait":
        _, kind, ring, tiles, counted = s
        r = (ring.slot_free_time(tiles) if kind == "space" else ring.data_time(tiles) if kind == "data"
             else ring.popped_time(tiles))
        if r is None:
            return False
        if r > th.t:
            if counted:
                if kind == "space":
                    st.wait_space += r - th.t
                else:
                    st.wait_data += r - th.t
            th.t = r
    elif op == "busy":
        th.t += s[1]
        if s[2]:
            st.busy += s[1]
    elif op == "push":
        for k in s[2]:
            s[1].push_t[k] = th.t + s[3]
    elif op == "pop":
        for k in s[2]:
            s[1].pop_t[k] = th.t + s[3]
    elif op == "count":
        st.tiles += s[1]
        st.batches += s[2]
    th.pc += 1
    th.ran = True
    if th.sets_end and th.pc == len(th.steps):
        st.end = th.t
    return True


def _start_batch(th, b, prog):
    st = th.stats
    ready = b.ring.slot_free_time(b.tiles) if b.wait == "space" else b.ring.data_time(b.tiles)
    if ready is None:
        return False
    t = max(th.t, ready)
    if b.wait == "space":
        st.wait_space += t - th.t
    else:
        st.wait_data += t - th.t
    first, touch, tcs = b.warm
    warm = (first if st.batches == 0 else 0.0) + (touch if st.batches < tcs else 0.0)
    t += b.pre + warm
    st.busy += b.pre + warm
    log = prog.logs[b.log]
    issue, comp_prev, completes, j = t, -INF, [], 0
    for count, lat, gap, svc, q in b.groups:
        for _ in range(count):
            if b.max_inflight and j >= b.max_inflight:
                issue = max(issue, completes[j - b.max_inflight])
            issue_end = issue + b.issue
            if b.dma:  # the DMA engine moves one transfer at a time: start when issued and the engine is free
                th.dma_free = max(issue_end, th.dma_free) + gap
                comp = th.dma_free + lat
            else:
                comp = max(issue_end + lat, comp_prev + gap)
            if svc:
                comp = prog.queues[q].serve(comp, svc)
            completes.append(comp)
            log.add(issue_end, comp)
            comp_prev = comp
            issue = issue_end
            j += 1
    st.busy += b.issue * j
    th.t = issue
    th.bar = (issue, b)
    th.ran = True
    if not b.timed:
        _finish_batch(th, issue)
    return True


def _finish_batch(th, t_bar):
    """Release th's parked batch from its barrier at t_bar, then publish or release its entries."""
    issue, b = th.bar
    th.bar = None
    st = th.stats
    st.wait_transfer += t_bar - issue
    t = t_bar + b.post
    st.busy += b.post
    ring_t = b.ring.push_t if b.release == "push" else b.ring.pop_t
    for k in b.tiles:
        ring_t[k] = t + b.delay
    st.batches += 1
    st.tiles += len(b.tiles)
    th.t = t
    th.pc += 1
    if th.sets_end and th.pc == len(th.steps):
        st.end = t


def _claim(g):
    """Dynamic split: hand the next queued batch to the idle thread that is free first, unless a thread parked at a
    barrier could become free earlier."""
    got = False
    while g.work:
        idle = [th for th in g.threads if th.done]
        if not idle:
            break
        th = min(idle, key=lambda x: x.t)
        if any(o.bar is not None and o.bar[0] < th.t for o in g.threads):
            break
        th.steps.append(("busy", g.claim, True))
        th.steps.append(g.work.pop(0))
        got = True
    return got


def _run_group(g, prog):
    moved = False
    while True:
        m = _claim(g) if g.work is not None else False
        for th in g.threads:
            while _step(th, prog):
                m = True
        if not m or not g.dm:
            return moved or m
        moved = True


def run(prog, trace_limit=0):
    """Evaluate a Program. Returns makespan, per-role aggregates, first output time and optional thread traces."""
    dm_threads = [th for g in prog.groups if g.dm for th in g.threads]
    while True:
        progressed = False
        for g in prog.groups:
            if _run_group(g, prog):
                progressed = True
        if all(th.done for g in prog.groups for th in g.threads) and not any(g.work for g in prog.groups):
            break
        if progressed:
            continue
        parked = [(prog.logs[th.bar[1].log].quiet_at(th.bar[0]), th) for th in dm_threads if th.bar is not None]
        if not parked:
            raise Deadlock("schedule makes no progress: the configuration would hang")
        t_min = min(p[0] for p in parked)
        for t_q, th in parked:
            if t_q == t_min:
                _finish_batch(th, t_q)
        low = min([th.bar[0] if th.bar is not None else th.t for th in dm_threads] or [t_min])
        for log in prog.logs:
            log.prune(low)
        for q in prog.queues.values():
            q.prune(low - 1e6)
    for g in prog.groups:
        for th in g.threads:
            if th.sets_end and not th.ran:
                th.stats.end = th.stats.start
    makespan = max(st.end for st in prog.stats) + prog.drain
    roles = {}
    for st in prog.stats:
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
        r["utilization"] = r["busy"] / (r["threads"] * makespan) if makespan > 0 else 0.0
    ring = prog.first_output
    out = {"makespan": makespan, "roles": roles, "first_output": ring.push_t[0] if ring is not None and ring.n else None}
    if trace_limit:
        out["threads"] = [st.as_dict() for st in prog.stats[:trace_limit]]
    return out
