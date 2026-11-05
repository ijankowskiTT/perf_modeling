"""Quasar reshard (ttnn.experimental.quasar.reshard) as engine Programs, one core each.

mem string (the API builds it from workload.reshard, see mem_of): <layout>_<from>_<to>
  layout  t = TILE | r = ROW_MAJOR
  from/to <kind><cores><place>: kind hs | ws | bs<rows>x<cols>; place l (L1) | d (DRAM, banks 0..cores-1, <= 2)
  e.g. t_hs8l_hs4l, r_ws4l_ws8l, t_hs2d_hs8l, t_bs2x2l_bs4x4l. Shards are ceil-divided (TILE: whole tiles); L1 cores
  are taken row-wise on the 8x4 grid (bs: a rows x cols rectangle at (0,0)).

Factory selection mirrors ReshardDeviceOperation::select_program_factory (calibration tree 0fff850f3c6):
  legacy path when both specs are 2-D and (same layout, not BLOCK, >= 1 L1 buffer) or the output is in L1:
    HEIGHT->HEIGHT                         SameWidth (output in L1: the output cores read) / SameWidthWrite (input
                                           cores write to the DRAM output)
    ROW_MAJOR WIDTH->WIDTH                 SameHeight (output L1, no padding) / SameHeightWrite (output DRAM);
                                           output L1 with padded shards -> Generic
    anything else                          Generic (output cores gather strided page runs)
  otherwise DRAM->DRAM                     CopyPages; L1 -> DRAM                CopyLocal

Mirrored kernels (both DM kernels of a core run the same source; k0 = "reader" config, k1 = "writer" config):
  SameWidth       reshard_same_width_reader: the core's units split div_up(n,2) / rest; per kernel one read per
                  segment (a run of units inside one remote shard), one read barrier at the end.
  SameWidthWrite  reshard_same_width_writer: same split and segments, one write per segment, one write barrier.
  SameHeight      reshard_same_height_reader: segments = overlaps of the local shard's columns with remote shards;
                  k0 takes floor(rows/2) rows, k1 the rest; per segment per row one read; one barrier at the end.
  SameHeightWrite reshard_same_height_writer: the same with writes.
  Generic         reshard_reader: per output core the page runs (create_map_for_reshard: consecutive pages of one
                  input core, grouped in strides); k0 takes the first div_up(ranges,2) ranges; one read per stride
                  run; one barrier at the end.
  CopyPages       nd_reshard_copy_pages_{reader,writer}: all 32 cores, pages split evenly (first cores +1); per page
                  reserve(1) / read / barrier / push, wait / write / barrier / pop on a 1-entry DFB.
  CopyLocal       nd_reshard_copy_local_shards (is_reader): shard k on core k's first kernel (the second kernel's
                  shard k+N does not exist); per page one write per transfer_size piece and a flush; one barrier.
                  Not modeled yet (plan raises): its output is wrong on both simulators. Generic from a DRAM input
                  is not modeled either (neither simulator runs it); unaligned HEIGHT->HEIGHT rows (scratch path)
                  and Generic with differing page sizes (row-major width changes) neither.

Cost keys (<p>.*, cycles; p per factory, PREFIX): start_{reader,writer}; {rd,wr}_setup (prologue before the first
request), {rd,wr}_first (once per kernel that has work), {rd,wr}_fixed (per segment / range: vararg loads, address
set-up), {rd,wr}_issue (per NoC request), {rd,wr}_cpb (DMA cycles per byte), {rd,wr}_tail (epilogue after the
barrier), drain, dram_setup (both kernels, DRAM input), wr_bank_mod (CopyPages: writer per page with 2 output banks);
rd_{lat,svc}_l1 where a profile times L1 transfers (ttsim-perf), with the NoC port / link rates rd_{port,link}_cpb
(default NOC_CPB) stretched by _contention. rd_* belong to the kernel source that reads, wr_* to the one that writes;
the two DM threads of a core run one source, so k0 pays rd_setup / rd_tail and k1 wr_setup / wr_tail of it.
"""

from __future__ import annotations

import functools
import math

from .. import dram
from ..engine import DMBatch, Ring, ThreadStats, TransferLog
from .layout import DRAM_ALIGN, ELEM, GRID, GRID_X, L1_ALIGN, STRATEGY, TILE, dm_program

LINK_UNSTALLED = 0.8  # share of a link a response to a single-source core holds (_contention)

OPS = ("reshard",)
PREFIX = {("reshard", "SameWidth"): "rsw", ("reshard", "SameWidthWrite"): "rsww", ("reshard", "SameHeight"): "rsh",
          ("reshard", "SameHeightWrite"): "rshw", ("reshard", "Generic"): "rgen", ("reshard", "CopyPages"): "rcp",
          ("reshard", "CopyLocal"): "rcl"}
FAMILY = {"reshard": "move"}
# ttsim-perf noc_perf.h (QSR): an NIU moves a 256-byte flit into or out of L1 every NOC_L1_CYCLES_UNHASHED = 4 cycles
NOC_CPB = 4 / 256
FIT_INIT = {"dram_setup": 10.0, "wr_bank_mod": 5.0}  # initial values of this module's own terms (fit_layout.py)


# ----------------------------------------------------------------------------------------------- shard specs

def _side(tok, H, W, layout):
    place = {"l": "l1", "d": "dram"}.get(tok[-1])
    body = tok[:-1]
    kind = body[:2]
    if place is None or kind not in ("hs", "ws", "bs"):
        raise ValueError(f"reshard: bad shard spec '{tok}' (<hs|ws|bs><cores><l|d>)")
    rest = body[2:]
    if kind == "bs":
        gy, gx = (int(v) for v in rest.split("x")) if "x" in rest else (math.isqrt(int(rest)),) * 2
        n = gx * gy
    else:
        n = int(rest)
        gx, gy = (n, 1) if place == "dram" else (min(n, GRID_X), math.ceil(n / GRID_X))
    if place == "dram" and (kind == "bs" or n > 2):
        raise ValueError("reshard: DRAM shards are modeled height/width-sharded over <= 2 banks")
    if n < 1 or (place == "l1" and n > GRID):
        raise ValueError(f"reshard: {n} cores do not fit the 8x4 grid")
    ht, wt = math.ceil(H / TILE), math.ceil(W / TILE)
    if layout == "TILE":
        if kind == "hs":
            sh, sw = math.ceil(ht / n) * TILE, wt * TILE
        elif kind == "ws":
            sh, sw = ht * TILE, math.ceil(wt / n) * TILE
        else:
            sh, sw = math.ceil(ht / gy) * TILE, math.ceil(wt / gx) * TILE
    else:  # row-major: whole rows; widths in multiples of 32 elements keep every page aligned
        q = 32
        if kind == "hs":
            sh, sw = math.ceil(H / n), W
        elif kind == "ws":
            sh, sw = H, math.ceil(math.ceil(W / n) / q) * q
        else:
            sh, sw = math.ceil(H / gy), math.ceil(math.ceil(W / gx) / q) * q
    if place == "l1":
        if kind == "bs":
            cores = [(x, y) for y in range(gy) for x in range(gx)]
        else:
            cores = [(k % GRID_X, k // GRID_X) for k in range(n)]
    else:
        cores = [(k, 0) for k in range(n)]
    # shard grid as the device sees it (first core range: num_shard_cores in the padding check)
    first_range_x = gx if kind == "bs" else (n if place == "dram" else min(n, GRID_X))
    return dict(kind=kind, place=place, cores=cores, n=n, gx=gx, gy=gy, sh=sh, sw=sw, first_range_x=first_range_x)


def parse_mem(mem, H, W, elem=2):
    """mem string -> {layout, from, to}: shard specs as the runner creates them (calibration/runners/reshard.py)."""
    try:
        lay, a, b = mem.split("_")
    except ValueError:
        raise ValueError(f"reshard: mem '{mem}' is not <t|r>_<from>_<to>")
    layout = {"t": "TILE", "r": "ROW_MAJOR"}.get(lay)
    if layout is None:
        raise ValueError(f"reshard: layout '{lay}' is t (TILE) or r (ROW_MAJOR)")
    return {"layout": layout, "from": _side(a, H, W, layout), "to": _side(b, H, W, layout)}


def select_factory(spec, W, elem):
    a, b = spec["from"], spec["to"]
    legacy = (a["kind"] == b["kind"] and a["kind"] != "bs" and "l1" in (a["place"], b["place"])) or b["place"] == "l1"
    if legacy:
        if a["kind"] == "hs" and b["kind"] == "hs":
            return "SameWidth" if b["place"] == "l1" else "SameWidthWrite"
        if spec["layout"] == "ROW_MAJOR" and a["kind"] == "ws" and b["kind"] == "ws":
            if b["place"] == "l1":
                pad = a["first_range_x"] * a["sw"] > W or b["first_range_x"] * b["sw"] > W
                return "Generic" if pad else "SameHeight"
            return "SameHeightWrite"
        return "Generic"
    if a["place"] == "dram" and b["place"] == "dram":
        return "CopyPages"
    return "CopyLocal"


# ----------------------------------------------------------------------------------------------- per-core work

def _same_width(spec, H, W, elem, local_is_output):
    """ReshardSameWidthFactory: per local core, per kernel, the segment sizes in bytes."""
    loc, rem = (spec["to"], spec["from"]) if local_is_output else (spec["from"], spec["to"])
    if spec["layout"] == "TILE":
        unit = TILE * TILE * elem
        loc_per, rem_per = loc["sh"] * loc["sw"] // (TILE * TILE), rem["sh"] * rem["sw"] // (TILE * TILE)
        num_units = math.ceil(H / TILE) * math.ceil(W / TILE)
    else:
        unit = loc["sw"] * elem
        loc_per, rem_per = loc["sh"], rem["sh"]
        num_units = H
    for s in (loc, rem):
        if unit % (L1_ALIGN if s["place"] == "l1" else DRAM_ALIGN):
            raise ValueError("reshard: unaligned HEIGHT->HEIGHT rows (scratch path) not modeled yet")
    rem_rem, ri = rem_per, 0
    left = num_units
    out = []
    for _core in loc["cores"]:
        per_core = min(left, loc_per)
        left -= per_core
        per_kernel = math.ceil(per_core / 2)
        ks = []
        for _ in range(2):
            n = min(per_core, per_kernel)
            segs = []
            while n > 0:
                if rem_rem == 0:
                    rem_rem, ri = rem_per, ri + 1
                u = min(rem_rem, n)
                segs.append((u * unit, rem["cores"][ri]))
                per_core -= u
                n -= u
                rem_rem -= u
            ks.append(segs)
        out.append((ks[0], [1] * len(ks[0]), ks[1], [1] * len(ks[1])))
    return out


def _width_segments(loc, rem, nloc, nrem, elem):
    """compute_width_sharding_reshard_segments: per local core, the segments (bytes, remote core)."""
    out = []
    lo = ro = ri = 0
    for li in range(nloc):
        segs = []
        while lo < loc["sw"]:
            final = li >= nloc - 1 and ri >= nrem - 1
            t = (rem["sw"] - ro) if final else min(loc["sw"] - lo, rem["sw"] - ro)
            segs.append((t * elem, rem["cores"][ri]))
            lo += t
            ro += t
            if ro == rem["sw"]:
                ri += 1
                ro = 0
            if final:
                break
        lo = 0
        out.append(segs)
    return out


def _same_height(spec, elem, local_is_output):
    """ReshardSameHeightFactory: segments per local core; k0 does floor(rows/2) rows of every segment, k1 the rest.
    Returns per core (k0 requests, k0 segment starts, k1 requests, k1 segment starts)."""
    loc, rem = (spec["to"], spec["from"]) if local_is_output else (spec["from"], spec["to"])
    segs = _width_segments(loc, rem, len(loc["cores"]), len(rem["cores"]), elem)
    r0 = loc["sh"] // 2
    r1 = loc["sh"] - r0
    return [([b for b in s for _ in range(r0)], [1] * len(s), [b for b in s for _ in range(r1)], [1] * len(s))
            for s in segs]


def _page_grid(s, H, W, layout, elem):
    """A sharded tensor as pages: (rows of pages, pages per row, shard rows, shard cols in pages, page bytes)."""
    if layout == "TILE":
        return math.ceil(H / TILE), math.ceil(W / TILE), s["sh"] // TILE, s["sw"] // TILE, TILE * TILE * elem
    return H, math.ceil(W / s["sw"]), s["sh"], 1, s["sw"] * elem


def _page_owner(s, shr, shc, r, c):
    """(core, device page) holding page (r, c)."""
    if s["kind"] == "hs":
        k, dp = r // shr, (r % shr) * shc + c
    elif s["kind"] == "ws":
        k, dp = c // shc, r * shc + c % shc
    else:
        k, dp = (r // shr) * s["gx"] + c // shc, (r % shr) * shc + c % shc
    return s["cores"][k], dp


def _map_for_reshard(seq):
    """create_map_for_reshard (reshard_program_factory_generic.cpp) for one output core: its device pages in order as
    (input core, input device page) or None (padding) -> PageStrides [(stride_size, num_strides, skip, start core,
    core stride)]."""
    out = []
    i, end = 0, len(seq)
    while i < end:
        if seq[i] is None:
            j = i + 1
            while j < end and seq[j] is None:
                j += 1
            out.append((j - i, 1, True, None, (0, 0)))
            i = j
            continue
        if i + 1 == end:
            out.append((1, 1, False, seq[i][0], (0, 0)))
            break
        last = i
        j = i + 1
        while j < end and seq[j] is not None:
            if seq[last][0] != seq[j][0] or seq[last][1] + 1 != seq[j][1]:
                break
            last = j
            j += 1
        size = last - i + 1
        st = i + size  # stride_it
        last_st = i
        score, sdata = (0, 0), 0
        if st != end and st != i and seq[st] is not None:
            nxt, prv = seq[st], seq[last_st]
            if nxt[0] == prv[0] and nxt[1] > prv[1]:
                sdata = nxt[1] - prv[1] - size
                score = (0, 0)
            elif nxt[0] != prv[0] and nxt[0][0] >= seq[i][0][0] and nxt[0][1] >= seq[i][0][1] and nxt[1] == seq[i][1]:
                sdata = 0
                score = (nxt[0][0] - seq[i][0][0], nxt[0][1] - seq[i][0][1])
            else:
                out.append((size, 1, False, seq[i][0], (0, 0)))
                i = st
                continue
        n = 1
        while st != end and seq[st] is not None:
            incomplete = False
            k, lk = st + 1, st
            for _ in range(size - 1):
                if k >= end or seq[k] is None or seq[k][0] != seq[lk][0] or seq[k][1] != seq[lk][1] + 1:
                    incomplete = True
                    break
                lk = k
                k += 1
            if incomplete:
                break
            n += 1
            last_st = k - 1
            st = k
            if st == end or seq[st] is None:
                break
            nxt, cur = seq[st], seq[last_st]
            if (nxt[0][0] - cur[0][0] != score[0] or nxt[0][1] - cur[0][1] != score[1] or
                    abs(nxt[1] - cur[1]) != sdata):
                break
        out.append((size, n, False, seq[i][0], score))
        i = st
    return out


def _generic(spec, H, W, elem):
    """ReshardGenericFactory (same page size): per output core, k0 / k1 read sizes (bytes)."""
    a, b = spec["from"], spec["to"]
    lay = spec["layout"]
    pa, pb = _page_grid(a, H, W, lay, elem), _page_grid(b, H, W, lay, elem)
    if pa[4] != pb[4]:
        raise ValueError("reshard: Generic with different input / output page sizes (row-major width change) "
                         "not modeled yet")
    pr, pc, _, _, page = pb  # same page grid on both sides
    seqs = {core: {} for core in b["cores"]}
    for r in range(pr):
        for c in range(pc):
            oc, dp = _page_owner(b, pb[2], pb[3], r, c)
            seqs[oc][dp] = _page_owner(a, pa[2], pa[3], r, c)
    out = []
    for core in b["cores"]:
        d = seqs[core]
        seq = [d.get(k) for k in range(max(d) + 1)] if d else []
        strides = _map_for_reshard(seq)
        h = math.ceil(len(strides) / 2)
        reads = [[(size * page, (c0[0] + k * cs[0], c0[1] + k * cs[1]))
                  for size, n, skip, c0, cs in part if not skip for k in range(n)]
                 for part in (strides[:h], strides[h:])]
        out.append((reads[0], [1] * h, reads[1], [1] * (len(strides) - h)))  # one loop pass per range
    return out


def _copy_pages(spec, H, W, elem):
    a = spec["from"]
    pr, pc, _, _, page = _page_grid(a, H, W, spec["layout"], elem)
    per, rem = divmod(pr * pc, GRID)
    loads = {}
    for k in range(GRID):
        p = per + (1 if k < rem else 0)
        loads.setdefault(p, []).append(k)
    return page, [(p, ks) for p, ks in sorted(loads.items(), key=lambda kv: kv[1][0])]


def _noc_of(xy):
    """NoC coordinates of a logical (x, y) worker (QSR: workers at NoC (x + 2, y + 2))."""
    return (xy[0] + 2, xy[1] + 2)


def _bank_hists(reqs):
    """Per request position j, {bank: cores} over the cores' request lists [(bytes, remote (bank, 0))]."""
    out = []
    for j in range(max((len(r) for r in reqs), default=0)):
        h = {}
        for r in reqs:
            if j < len(r):
                b = r[j][1][0]
                h[b] = h.get(b, 0) + 1
        out.append(h)
    return out


def _copy_pg(spec, H, W, elem, p, ks):
    """CopyPages on the DRAM channels: per page j of the load's cores (core k copies pages start_k ..), {bank: cores}
    of the page's input shard and of its output shard (shard k of a DRAM-sharded buffer sits in bank k)."""
    a, b = spec["from"], spec["to"]
    pa, pb = _page_grid(a, H, W, spec["layout"], elem), _page_grid(b, H, W, spec["layout"], elem)
    per, rem = divmod(pa[0] * pa[1], GRID)
    starts = [k * per + min(k, rem) for k in ks]

    def hists(s, pg):
        out = []
        for j in range(p):
            h = {}
            for st in starts:
                r, cc = divmod(st + j, pg[1])
                bank = _page_owner(s, pg[2], pg[3], r, cc)[0][0]
                h[bank] = h.get(bank, 0) + 1
            out.append(h)
        return out

    return {"noc": [_noc_of((k % GRID_X, k // GRID_X)) for k in ks], "banks": [hists(a, pa), hists(b, pb)]}


def _copy_local(spec, H, W, elem):
    """NdReshardCopyLocalShardFactory<local_is_input>: shard k on core k (first kernel); writes per page."""
    a, b = spec["from"], spec["to"]
    pa, pb = _page_grid(a, H, W, spec["layout"], elem), _page_grid(b, H, W, spec["layout"], elem)
    xfer = math.gcd(pa[4], pb[4])
    out = []
    for core in a["cores"]:
        pages = sum(_page_owner(a, pa[2], pa[3], r, c)[0] == core for r in range(pa[0]) for c in range(pa[1]))
        out.append((pages, pa[4] // xfer, xfer))
    return out


# ----------------------------------------------------------------------------------------------- plan / build

def plan(op, H, W, dtype="bf16", mem="t_hs8l_hs4l", multicore=True):
    """Factory and per-core work: [(cfg, cores)], the first load being logical core (0,0) (the profiler's first
    core). cfg: factory work lists, in_place / out_place (the remote side's placement for the NoC terms)."""
    if op != "reshard":
        raise ValueError(f"{op} is not reshard")
    return _plan(H, W, dtype, mem)


@functools.lru_cache(maxsize=512)
def _plan(H, W, dtype, mem):  # pure in its arguments; the fitter plans every run many times
    elem = ELEM[dtype]
    spec = parse_mem(mem, H, W, elem)
    fac = select_factory(spec, W, elem)
    a, b = spec["from"], spec["to"]
    if fac == "CopyLocal":
        raise ValueError("reshard CopyLocal (L1 -> DRAM, layout change) not modeled yet: its output is wrong on both "
                         "simulators at 0fff850f3c6")
    if fac == "Generic" and a["place"] == "dram":
        raise ValueError("reshard Generic from DRAM not modeled yet: the simulators cannot run it (coord_to_tile)")
    if fac in ("SameWidth", "SameWidthWrite", "SameHeight", "SameHeightWrite", "Generic"):
        out_local = fac in ("SameWidth", "SameHeight", "Generic")
        remote = a if out_local else b
        if fac.startswith("SameWidth"):
            work = _same_width(spec, H, W, elem, out_local)
        elif fac.startswith("SameHeight"):
            work = _same_height(spec, elem, out_local)
        else:
            work = _generic(spec, H, W, elem)
        kind = "read" if out_local else "write"
        local = spec["to"] if out_local else spec["from"]
        share = _contention(work, local["cores"]) if kind == "read" and remote["place"] == "l1" else {}
        groups = {}
        for i, (k0, s0, k1, s1) in enumerate(work):
            key = (tuple(b for b, _ in k0), tuple(s0), tuple(b for b, _ in k1), tuple(s1),
                   share.get(local["cores"][i], (1.0, 0.0)))
            groups.setdefault(key, []).append(i)
        keys = sorted(groups, key=lambda k: groups[k][0])  # logical core 0 first
        loads = []
        for k in keys:
            k0, s0, k1, s1, noc = k
            nbytes = sum(k0) + sum(k1)
            cfg = dict(kind=kind, k0=list(k0), k1=list(k1), seg0=list(s0), seg1=list(s1), noc=noc,
                       blocks=1, wt=len(k0) + len(k1), tiles=nbytes / (TILE * TILE * elem), elem=elem,
                       in_place=remote["place"] if out_local else "borrowed",
                       out_place="borrowed" if out_local else remote["place"])
            if remote["place"] == "dram":
                # DRAM channels: shard k of a DRAM-sharded buffer sits in bank k; per request of each kernel, how many
                # of the load's cores send it to each bank (the load's cores share sizes, not banks)
                ids = groups[k]
                cfg["_pg"] = {"noc": [_noc_of(local["cores"][i]) for i in ids],
                              "banks": [_bank_hists([work[i][w] for i in ids]) for w in (0, 2)]}
            loads.append((cfg, len(groups[k])))
        return fac, loads
    if fac == "CopyPages":
        page, groups = _copy_pages(spec, H, W, elem)
        loads = [(dict(kind="copy", pages=p, page=page, out_banks=b["n"], blocks=1, wt=p, tiles=p * page / (TILE * TILE * elem),
                       elem=elem, in_place="dram", out_place="dram", _pg=_copy_pg(spec, H, W, elem, p, ks)), len(ks))
                 for p, ks in groups if p]
        return fac, loads
    work = _copy_local(spec, H, W, elem)
    groups = {}
    for w in work:
        groups[w] = groups.get(w, 0) + 1
    keys = [work[0]] + [k for k in groups if k != work[0]]
    loads = [(dict(kind="local", pages=p, per_page=t, xfer=x, blocks=1, wt=p, tiles=p * t * x / (TILE * TILE * elem),
                   elem=elem, in_place="borrowed", out_place=b["place"]), groups[(p, t, x)]) for p, t, x in keys]
    return fac, loads


def _noc(c, p, kind, place, nbytes, noc=(1.0, 0.0)):
    """(latency, gap, svc) of one request: gap = bytes x <p>.{rd,wr}_cpb (the thread's DMA), plus the L1 terms when
    the remote side is L1 on a profile that times L1 transfers: <p>.{rd,wr}_cpb_l1, latency, and the core's NoC port
    queue (both kernels' requests served in time order): <p>.{rd,wr}_svc_l1 + bytes x max(niu x <p>.rd_port_cpb,
    link x <p>.rd_link_cpb), niu / link = the busiest NIU / link on the core's response paths over its own inbound
    bytes (_contention)."""
    d = "rd" if kind == "read" else "wr"
    gap = nbytes * c.get(f"{p}.{d}_cpb", 0.0)
    lat = svc = 0.0
    if place == "l1":
        gap += nbytes * c.get(f"{p}.{d}_cpb_l1", 0.0)
        # NoC rates: a profile that times L1 transfers for this factory (has <p>.rd_lat_l1) runs at the simulator's
        # NOC_CPB unless it overrides <p>.rd_port_cpb / <p>.rd_link_cpb; one that does not (craq-sim) has none
        timed = f"{p}.{d}_lat_l1" in c
        rate = max(noc[0] * c.get(f"{p}.rd_port_cpb", NOC_CPB if timed else 0.0),
                   noc[1] * c.get(f"{p}.rd_link_cpb", NOC_CPB if timed else 0.0))
        svc = c.get(f"{p}.{d}_svc_l1", 0.0) + nbytes * rate
        # the queue places a service window ending at the completion time: the latency includes the service, so a
        # transfer streams after its request arrives, never before
        lat = c.get(f"{p}.{d}_lat_l1", 0.0) + svc
    return lat, gap, svc


def _route(src, dst):
    """Directed links a read response from src to dst crosses: X first along src's row, then Y (QSR NOC, 8x4 workers
    at NOC (x+2, y+2): a contiguous mesh)."""
    (x, y), (tx, ty) = src, dst
    links = []
    while x != tx:
        nx = x + (1 if tx > x else -1)
        links.append(("link", (x, y), (nx, y)))
        x = nx
    while y != ty:
        ny = y + (1 if ty > y else -1)
        links.append(("link", (x, y), (x, ny)))
        y = ny
    return links


def _contention(work, cores):
    """ttsim-perf times L1 reads flit by flit: an NIU moves a 256-byte flit into or out of its L1 every 4 cycles
    (64 B/cycle), and a response packet holds its VC on every link of its path until its tail has drained into the
    destination's L1, so a directed link passes only about as much. With every core reading at once, a core's reads
    end when its busiest shared resource has moved all its bytes. A response to a core that reads from one source
    only is never stalled at its destination and holds the links less: it counts LINK_UNSTALLED of its bytes (read
    off the ttsim-perf runs: fan-out reshards pass ~80 B/cycle per link, fan-in ones 64). Returns {dst: (busiest
    NIU load, busiest link load) on dst's response paths, each over dst's own inbound bytes}."""
    load = {}
    paths = {}
    for core, w in zip(cores, work):
        reqs = w[0] + w[2]
        lw = 1.0 if len({src for _, src in reqs}) > 1 else LINK_UNSTALLED
        for b, src in reqs:
            path = paths.setdefault((core, src), [("out", src), ("in", core)] + _route(src, core))
            for r in path:
                load[r] = load.get(r, 0) + (b * lw if r[0] == "link" else b)
    niu, link = {}, {}
    for (dst, src), path in paths.items():
        niu[dst] = max(niu.get(dst, 0), max(load[r] for r in path if r[0] != "link"))
        link[dst] = max([link.get(dst, 0)] + [load[r] for r in path if r[0] == "link"])
    return {d: (niu[d] / load[("in", d)], link[d] / load[("in", d)]) for d in niu}


def build(op, cfg, c, p):
    """One core of a reshard factory: two DM threads (k0 = reader config, k1 = writer config)."""
    st_r = ThreadStats("reader", 0, start=c[f"{p}.start_reader"])
    st_w = ThreadStats("writer", 0, start=c[f"{p}.start_writer"])
    shared = c.get("noc.barrier_all_kinds", 0.0) >= 0.5 and "l1" in (cfg["in_place"], cfg["out_place"])
    logs = [TransferLog()] if shared else [TransferLog(), TransferLog()]
    li = lambda kind: 0 if shared or kind == "read" else 1  # noqa: E731  (barriers count per kind, cluster-wide)
    drain = c.get(f"{p}.drain", 0.0)
    kind = cfg["kind"]
    chans = cfg.get("_chans")  # DRAM channels, when the estimate times DRAM with them
    pg = cfg.get("_pg") or {}

    def side(read):
        return dram.Side(chans, read, pg["noc"]) if chans is not None else None

    if kind in ("read", "write"):
        d = "rd" if kind == "read" else "wr"
        place = cfg["in_place"] if kind == "read" else cfg["out_place"]
        dummy = Ring("none", 1, 1, 0)
        threads = []
        sd = side(kind == "read") if place == "dram" else None
        for w, (reqs, segs, who) in enumerate(((cfg["k0"], cfg["seg0"], "rd"), (cfg["k1"], cfg["seg1"], "wr"))):
            # a DRAM input costs both reading kernels a fixed set-up (BankView over DRAM banks), not a per-request
            # cost; the writing factories always write DRAM (folded into their set-up)
            from_dram = kind == "read" and place == "dram"
            setup = c.get(f"{p}.{who}_setup", 0.0) + (c.get(f"{p}.dram_setup", 0.0) if from_dram else 0.0)
            steps = [("busy", setup, True)]
            if reqs:
                groups, extra = [], []  # extra: per group its bytes and, on the DRAM channels, each request's banks
                for j, nb in enumerate(reqs):
                    g = (1,) + _noc(c, p, kind, place, nb, cfg["noc"]) + (d,)
                    cnt = sd.bank_counts(pg["banks"][w][j]) if sd is not None else None
                    if groups and groups[-1][1:] == g[1:] and extra[-1][0] == nb:
                        groups[-1] = (groups[-1][0] + 1,) + g[1:]
                        extra[-1][1].append(cnt)
                    else:
                        groups.append(g)
                        extra.append((nb, [cnt]))
                if sd is not None:  # one request per segment: its bytes, from / to its shard's bank
                    groups = [g + (dram.xfer(sd, cnt, nb),) for g, (nb, cnt) in zip(groups, extra)]
                # per segment (Generic: per range) loop overhead, paid before its requests: folded into pre
                pre = c.get(f"{p}.{d}_fixed", 0.0) * len(segs)
                # <p>.{rd,wr}_first: once per kernel that has work (its first request and the barrier that waits for
                # outstanding transfers; a kernel without work only runs set-up and tail)
                steps.append(DMBatch([], "space", dummy, pre, 0.0, (c.get(f"{p}.{d}_first", 0.0), 0.0, 0),
                                     c[f"{p}.{d}_issue"], groups, li(kind), True, "push", 0.0, dma=True))
            steps.append(("busy", c.get(f"{p}.{who}_tail", 0.0), True))
            threads.append(steps)
        return dm_program(st_r, threads[0], st_w, threads[1], logs, shared, drain)
    if kind == "copy":  # per page: reserve / read / barrier / push; wait / write / barrier / pop (1-entry DFB)
        n, page = cfg["pages"], cfg["page"]
        ring = Ring("cb", 1, 1, n)
        sr, sw = side(True), side(False)

        def dx(s, w, i):  # page i's channel timing: its input (w = 0) / output (w = 1) shard's bank per core
            return () if s is None else (dram.xfer(s, [s.bank_counts(pg["banks"][w][i])], page),)

        lat, gap, svc = _noc(c, p, "read", "dram", page)
        rd = [("busy", c.get(f"{p}.rd_setup", 0.0), True)]
        rd += [DMBatch([i], "space", ring, c[f"{p}.rd_fixed"], 0.0, (c.get(f"{p}.rd_first", 0.0), 0.0, 0), c[f"{p}.rd_issue"],
                       [(1, lat, gap, svc, "rd") + dx(sr, 0, i)], li("read"), True, "push", 0.0, dma=True)
               for i in range(n)]
        rd += [("wait", "popped", ring, range(n), False), ("busy", c.get(f"{p}.rd_tail", 0.0), True)]
        lat, gap, svc = _noc(c, p, "write", "dram", page)
        wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True)]
        # the writer's page -> bank selection costs <p>.wr_bank_mod per page when the output spans two banks (one
        # bank: folded away); the reader's lookup is hidden by the lock-step with the writer
        wfix = c[f"{p}.wr_fixed"] + (c.get(f"{p}.wr_bank_mod", 0.0) if cfg["out_banks"] > 1 else 0.0)
        wr += [DMBatch([i], "data", ring, wfix, 0.0, (c.get(f"{p}.wr_first", 0.0), 0.0, 0), c[f"{p}.wr_issue"],
                       [(1, lat, gap, svc, "wr") + dx(sw, 1, i)], li("write"), True, "pop", 0.0, dma=True)
               for i in range(n)]
        wr += [("busy", c.get(f"{p}.wr_tail", 0.0), True)]
        return dm_program(st_r, rd, st_w, wr, logs, shared, drain)
    # local: k0 writes its shard page by page (flush per page), k1 has no shard; both end with a write barrier
    n, per, x = cfg["pages"], cfg["per_page"], cfg["xfer"]
    dummy = Ring("none", 1, 1, 0)
    lat, gap, svc = _noc(c, p, "write", cfg["out_place"], x)
    rd = [("busy", c.get(f"{p}.rd_setup", 0.0), True)]
    rd += [DMBatch([], "space", dummy, c[f"{p}.wr_fixed"], 0.0, (0.0, 0.0, 0), c[f"{p}.wr_issue"],
                   [(per, lat, gap, svc, "wr")], li("write"), True, "push", 0.0, dma=True) for _ in range(n)]
    rd += [("busy", c.get(f"{p}.rd_tail", 0.0), True)]
    wr = [("busy", c.get(f"{p}.wr_setup", 0.0), True), ("busy", c.get(f"{p}.wr_tail", 0.0), True)]
    return dm_program(st_r, rd, st_w, wr, logs, shared, drain)


def l1_fit_keys(op, factory, sim):
    """L1 transfer terms the fitter may change: none on craq-sim (it times no NoC transfer, so the runs cannot
    identify them); on ttsim-perf the read terms and the NoC port rate (no global L1 bound: _contention places the
    shared NoC resources per core). Only the reading factories move data out of L1."""
    if sim.startswith("craq") or factory not in ("SameWidth", "SameHeight", "Generic"):
        return []
    return ["rd_lat_l1", "rd_svc_l1"]  # the NoC rates are the simulator's constants (NOC_CPB), not fitted


def fit_keys(op, factory):
    common = ["rd_setup", "wr_setup", "rd_tail", "wr_tail", "drain"]
    if factory == "Generic":
        return common + ["rd_fixed", "rd_issue", "rd_first", "rd_cpb"]
    if factory in ("SameWidth", "SameHeight"):
        return common + ["rd_fixed", "rd_issue", "rd_first", "rd_cpb", "dram_setup"]
    if factory in ("SameWidthWrite", "SameHeightWrite"):
        return common + ["wr_fixed", "wr_issue", "wr_first", "wr_cpb"]
    return common + ["rd_fixed", "rd_issue", "rd_first", "rd_cpb", "wr_fixed", "wr_issue", "wr_first", "wr_cpb",
                     "wr_bank_mod"]


def _spec_token(s):
    strategy, place = s.get("strategy"), s.get("memory", "l1")
    if strategy not in STRATEGY or place not in ("l1", "dram"):
        raise ValueError("reshard: workload.reshard.from / to need strategy height|width|block and memory l1|dram")
    if strategy == "block":
        grid = s.get("grid")
        if grid:
            body = f"bs{grid[0]}x{grid[1]}"
        else:
            n = s.get("cores")
            if not n or math.isqrt(n) ** 2 != n:
                raise ValueError("reshard: block sharding needs grid [rows, cols] or a square cores count")
            body = f"bs{math.isqrt(n)}x{math.isqrt(n)}"
    else:
        n = s.get("cores")
        if not n:
            raise ValueError("reshard: workload.reshard.from / to need cores")
        body = f"{STRATEGY[strategy]}{n}"
    return body + ("l" if place == "l1" else "d")


ASSUMPTION = "reshard: two data-movement kernels per core running the same source, no compute (the factory's kernels)"


def notes(profile_name, mem):
    """What a profile cannot time for this reshard -> (warnings, untimed). untimed: comparisons with it are
    inconclusive."""
    if profile_name == "craq-sim":
        return ["craq-sim times a reshard transfer per request, not per byte (a 512 KB segment costs about what a 2 KB "
                "one does): it ranks request counts only; use ttsim-perf for size, core-count and placement questions"], True
    if profile_name == "ttsim-perf" and "d_" in mem + "_":
        return ["ttsim-perf does not time DRAM transfers: the DRAM side of this reshard is nearly free here, so L1 vs "
                "DRAM comparisons are not meaningful (only L1 -> L1 specs compare on ttsim-perf)"], True
    return [], False


# Mean error over the calibration runs where a simulator is timed worse than the 3% target (tests/test_reshard.py
# bounds them): ttsim-perf L1 reads of many small responses (see _contention).
KNOWN_GAPS = {("ttsim-perf", "SameHeight"): 0.16, ("ttsim-perf", "Generic"): 0.095}


def mem_of(workload, knobs):
    """workload.reshard = {"layout": "tile"|"row_major" (default tile), "from": {strategy, cores | grid, memory},
    "to": {...}} -> (mem string, label)."""
    r = {**(workload.get("reshard") or {}), **((knobs or {}).get("reshard") or {})}  # a candidate may replace from / to
    if "from" not in r or "to" not in r:
        raise ValueError("reshard needs workload.reshard {from: {strategy, cores, memory}, to: {...}}")
    shape = workload["shape"]
    H, W = math.prod(shape[:-1]), shape[-1]
    lay = {"tile": "t", "row_major": "r"}.get(r.get("layout", "tile"))
    if lay is None:
        raise ValueError("reshard: layout is tile or row_major")
    mem = f"{lay}_{_spec_token(r['from'])}_{_spec_token(r['to'])}"
    parse_mem(mem, H, W)
    return mem, mem.replace("_", " ", 1).replace("_", " -> ")
