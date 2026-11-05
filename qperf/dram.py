"""Quasar DRAM channel model, used where a cost profile sets dram.model = "channels" (silicon-hypothesis).

No simulator times Quasar DRAM (craq-sim times no transfer, ttsim-perf times L1 NoC transfers only). The structure is
ttsim-perf's Blackhole DRAM model (src/noc_perf.h, noc_dram_access, compiled only for TT_ARCH_VERSION == 1), with
Quasar's channel count and positions and assumed rates:
- a channel streams dram.channel_bytes_per_cycle, shared by every core that reaches it: an access of b bytes holds the
  channel b / rate cycles in the earliest free window at or after its arrival (noc_reserve: first fit, so an access
  can fill a gap left between later ones);
- a read that finds the channel idle for more than dram.idle_cycles waits dram.idle_latency more before it streams
  (NOC_DRAM_IDLE / NOC_DRAM_LATENCY); a write never does (ttsim-perf's v1 write path: the channel takes the write
  when its data arrives, and the ack leaves once it has streamed);
- an interleaved buffer's page p sits in bank p mod banks, one bank per channel (tt-metal: NUM_DRAM_BANKS = the soc
  descriptor's dram views, get_bank_index in dataflow_api_addrgen.h); a DRAM-sharded buffer's shard k sits in bank k;
- a request reaches the channel dram.{read,write}_fixed_latency / 2 + hops x noc.hop_latency after it is issued and
  completes as long after the channel has streamed it (hops: X + Y distance on the mesh, QSR noc_hops).

Cores of one load (identical programs) are evaluated once, by a representative core (the mean field of the op):
every one of its requests stands for the same request of each core of the load, so it holds each channel for the
copies whose page maps there (count x b / rate in one window: the channel's throughput is exact) and completes with
the middle copy of the slowest channel (cores drift apart, so each sees the copies' mean wait, not always the
last). All loads of an op (and both cluster groups of binary_ng) share the same channels in one evaluation, so
requests queue per channel in time order across cores: channels saturate, camp (every core on the same bank at the
same time) or idle between batches, and batching and more threads hide the idle latency or do not.
"""

from __future__ import annotations

import bisect
import math

GRID_X, GRID_Y = 8, 4  # Quasar workers: an 8 x 4 grid at NoC x 2..9, y 2..5 (quasar_32_arch_ttsim.yaml)
WORKER_X0, WORKER_Y0 = 2, 2
BOUND_UTIL = 0.85  # a channel busy at least this share of the op's DRAM-active window is saturated: DRAM-bound
PRUNE_MARGIN = 1e5  # cycles of reservations kept behind the evaluation front (idle checks look back dram.idle_cycles)


def enabled(c):
    """True when the resolved costs time DRAM transfers with the channel model."""
    return c.get("dram.model") == "channels"


def worker_noc(k):
    """NoC coordinates of logical worker k, row-wise on the 8 x 4 grid."""
    return (WORKER_X0 + k % GRID_X, WORKER_Y0 + (k // GRID_X) % GRID_Y)


def logical_noc(xy):
    """NoC coordinates of a logical (x, y) worker."""
    return (WORKER_X0 + xy[0], WORKER_Y0 + xy[1])


class Channels:
    """The DRAM channels of one evaluation: per-channel reservations, idle-read latency, busy cycles and bytes."""

    def __init__(self, c):
        self.n = max(1, int(c["dram.channels"]))
        pos = c.get("dram.channel_noc") or []
        self.pos = [tuple(pos[i % len(pos)]) for i in range(self.n)] if pos else [None] * self.n
        self.rate = float(c["dram.channel_bytes_per_cycle"])
        self.idle_lat = float(c.get("dram.idle_latency", 0.0))
        self.idle = float(c.get("dram.idle_cycles", 0.0))
        self.hop = float(c.get("noc.hop_latency", 0.0))
        self.fixed = {True: float(c.get("dram.read_fixed_latency", 0.0)),
                      False: float(c.get("dram.write_fixed_latency", 0.0))}
        self.starts = [[] for _ in range(self.n)]  # sorted, disjoint, merged busy windows per channel
        self.ends = [[] for _ in range(self.n)]
        self.busy = [0.0] * self.n
        self.nbytes = [0.0] * self.n
        self.first = [math.inf] * self.n
        self.last = [0.0] * self.n
        self.idle_reads = 0
        self.reads = 0

    def mean_hops(self, cores):
        """Mean X + Y hops from NoC coordinates `cores` to each channel (0 where a position is unknown)."""
        out = []
        for p in self.pos:
            if p is None or not cores:
                out.append(0.0)
            else:
                out.append(sum(abs(x - p[0]) + abs(y - p[1]) for x, y in cores) / len(cores))
        return out

    def access(self, ch, t, nbytes, read, copies=1, take=True):
        """`copies` accesses of nbytes arriving at channel ch at t, served back to back: returns when the middle one has
        streamed (their mean wait). take=False only looks (a core whose traffic another load already accounts for)."""
        st, en = self.starts[ch], self.ends[ch]
        if not en or t >= en[-1]:  # after every window (the common case): no search
            if read:
                self.reads += take
                if not en or t > en[-1] + self.idle:
                    t += self.idle_lat
                    self.idle_reads += take
            length = copies * nbytes / self.rate
            done, mid = t + length, t + (copies + 1) / 2.0 * nbytes / self.rate
            if take:
                self.busy[ch] += length
                self.nbytes[ch] += copies * nbytes
                if t < self.first[ch]:
                    self.first[ch] = t
                if done > self.last[ch]:
                    self.last[ch] = done
                if en and en[-1] >= t - 1e-9:
                    en[-1] = done
                else:
                    st.append(t)
                    en.append(done)
            return mid
        i = bisect.bisect_right(st, t)  # windows starting at or before t: [0, i)
        if read:
            self.reads += take
            if i == 0 or t > en[i - 1] + self.idle:  # idle for more than dram.idle_cycles: the read waits
                t += self.idle_lat
                self.idle_reads += take
                i = bisect.bisect_right(st, t)
        length = copies * nbytes / self.rate
        s = t
        k = i - 1 if i and en[i - 1] > s else i  # the first window that ends after t
        while k < len(st):
            if st[k] >= s + length - 1e-9:
                break
            if en[k] > s:
                s = en[k]
            k += 1
        done = s + length
        mid = s + (copies + 1) / 2.0 * nbytes / self.rate  # the mean copy's end
        if not take:
            return mid
        self.busy[ch] += length
        self.nbytes[ch] += copies * nbytes
        if s < self.first[ch]:
            self.first[ch] = s
        if done > self.last[ch]:
            self.last[ch] = done
        # insert [s, done) at k (windows before k end at or before s), merging touching neighbours
        e = done
        if k and en[k - 1] >= s - 1e-9:
            k -= 1
            s = st[k]
            del st[k], en[k]
        if k < len(st) and st[k] <= e + 1e-9:
            e = max(e, en[k])
            del st[k], en[k]
        st.insert(k, s)
        en.insert(k, e)
        return mid

    def prune(self, before):
        """Drop windows that ended long before `before` (no access can arrive there any more)."""
        cut = before - PRUNE_MARGIN
        for ch in range(self.n):
            en = self.ends[ch]
            j = bisect.bisect_left(en, cut)
            if j:
                del self.starts[ch][:j], en[:j]

    def stats(self, span, scale=1.0):
        """{"channels": [{"noc", "busy_frac", "bytes"}], "bound", "idle_read_frac"}: busy share of the op's span and
        bytes per channel (x scale: an extrapolated run reports the whole op), bound = some channel busy at least
        BOUND_UTIL of the window from the first to the last DRAM access."""
        first, last = min(self.first), max(self.last)
        window = last - first if last > first else 0.0
        chans = [{"noc": list(p) if p else None, "busy_frac": round(min(1.0, b / span), 3) if span else 0.0,
                  "bytes": int(round(nb * scale))} for p, b, nb in zip(self.pos, self.busy, self.nbytes)]
        bound = bool(window) and max(self.busy) / window >= BOUND_UTIL
        return {"channels": chans, "bound": bound,
                "idle_read_frac": round(self.idle_reads / self.reads, 3) if self.reads else 0.0}


class Side:
    """One DRAM-sided tensor of one load as its representative core sees it: the channels, read or write, the latency
    to each channel from the load's cores (mean hops), and how the load's cores spread over the banks (hist[r]:
    cores whose page base is r mod channels; take=False for a core another load already accounts for)."""

    def __init__(self, ch, read, cores, bases=(0,), take=True):
        self.ch, self.read, self.take = ch, read, take
        n = ch.n
        hops = ch.mean_hops(cores)
        half = ch.fixed[read] / 2.0
        self.pre = [half + h * ch.hop for h in hops]
        self.post = list(self.pre)
        hist = [0] * n
        for b in bases:
            hist[b % n] += 1
        # counts per channel for a relative page g: rot[g % n][ch] = cores whose page base + g maps to ch
        self.rot = [tuple(hist[(c - g) % n] for c in range(n)) for g in range(n)]

    def counts(self, g):
        """Copies per channel of a request for the representative's relative page g (interleaved buffer)."""
        return self.rot[g % self.ch.n]

    def bank_counts(self, banks):
        """Copies per channel of a request whose copies go to explicit banks ({bank: cores}; a sharded buffer)."""
        out = [0] * self.ch.n
        for b, k in banks.items():
            out[b % self.ch.n] += k
        return tuple(out)

    def complete(self, t, counts, nbytes):
        """Completion of one request leaving its core at t, `counts` copies per channel: the slowest copy."""
        end = 0.0
        ch = self.ch
        for c, k in enumerate(counts):
            if k:
                e = ch.access(c, t + self.pre[c], nbytes, self.read, k, self.take) + self.post[c]
                if e > end:
                    end = e
        return end


class Pages:
    """A load's DRAM-side requests as relative page ids (page = representative's base + relative id): tile pages of a
    core's chunk of `chunk` tiles per tile-row in a tensor `stride` tiles wide (chunk 0: consecutive), row pages of
    blocks `nbr` per tile-row."""

    def __init__(self, side, chunk=0, stride=0, nbr=1, tile=32):
        self.side, self.chunk, self.stride, self.nbr, self.tile = side, chunk, stride, max(1, nbr), tile

    def tiles(self, ts):
        c, s = self.chunk, self.side
        if c:
            return [s.counts((t // c) * self.stride + t % c) for t in ts]
        return [s.counts(t) for t in ts]

    def rows(self, block, ks):
        base = (block // self.nbr) * self.tile
        return [self.side.counts(base + k) for k in ks]


def xfer(side, counts, nbytes, ack=None):
    """The 6th element of a DMBatch transfer group: the channel timing of its requests (one counts tuple each). ack:
    flushed writes (the batch waits until they leave), the log index of the kernel's final write barrier."""
    return (side, counts, nbytes) if ack is None else (side, counts, nbytes, ack)


def mean_trip(c):
    """Request + response hops of an average worker to an average channel, in cycles (2 x mean hops x hop latency)."""
    pos = c.get("dram.channel_noc") or []
    workers = [worker_noc(k) for k in range(GRID_X * GRID_Y)]
    if not pos:
        return 0.0
    hops = sum(abs(x - p[0]) + abs(y - p[1]) for p in pos for x, y in workers) / (len(pos) * len(workers))
    return 2.0 * hops * float(c.get("noc.hop_latency", 0.0))


def apply_overrides(c, overrides):
    """What-ifs on the aggregate DRAM keys reach the channel model (resolved costs c, in place): a new total bandwidth
    or efficiency sets the per-channel rate (total x efficiency / channels), a new DRAM read / write latency the fixed
    request latency (an isolated read from an average core keeps taking noc.read_latency.dram; below the idle latency
    plus the hops, the idle latency gives way first). Overriding the channel keys themselves wins."""
    if not overrides or not enabled(c):
        return c
    o = set(overrides)
    if o & {"dram.total_bytes_per_cycle", "dram.efficiency"} and "dram.channel_bytes_per_cycle" not in o:
        c["dram.channel_bytes_per_cycle"] = (c["dram.total_bytes_per_cycle"] * c.get("dram.efficiency", 1.0)
                                             / max(1, int(c["dram.channels"])))
    trip = mean_trip(c)
    if "noc.read_latency.dram" in o and "dram.read_fixed_latency" not in o:
        rest = c["noc.read_latency.dram"] - trip
        if "dram.idle_latency" not in o:
            c["dram.idle_latency"] = min(c.get("dram.idle_latency", 0.0), max(0.0, rest))
        c["dram.read_fixed_latency"] = max(0.0, rest - c.get("dram.idle_latency", 0.0))
    if "noc.write_latency.dram" in o and "dram.write_fixed_latency" not in o:
        c["dram.write_fixed_latency"] = max(0.0, c["noc.write_latency.dram"] - trip)
    return c


def aggregate_bound(c, total_bytes):
    """Lower bound of the DRAM time of total_bytes over every channel at its full rate (cycles, no start)."""
    return total_bytes / (max(1, int(c["dram.channels"])) * float(c["dram.channel_bytes_per_cycle"]))
