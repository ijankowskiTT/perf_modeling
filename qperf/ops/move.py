"""ttnn.experimental.quasar.move (interleaved, same buffer type in and out) as engine Programs, one core each.

move frees its input, allocates the output and picks a factory from where the output landed (move.cpp):
- same address: no program runs (the data is already in place);
- overlapping, and the per-core share fits L1 (fits_in_cb): MoveOverlap - modeled here (TILE: factory "Overlap",
  ROW_MAJOR: "OverlapRowMajor");
- not overlapping (or another buffer type, or too big for L1): MoveProgramFactory, which delegates to the legacy copy
  SameMemoryConfig descriptor (DataMovementKernel). Quasar rejects it (kernel.hpp TT_FATAL "DataMovementKernel is not
  supported on Quasar"), so plan() raises;
- sharded input: MoveSharded (local L1 copy backwards in chunks of dst - src) - not modeled yet.

MoveOverlap (move_overlap_program_factory.cpp; kernels move_[stick_layout_]interleaved_with_overlap[_writer].cpp,
calibration tree 0fff850f3c6). Pages (tiles, or rows for ROW_MAJOR) are split by split_work_to_cores over the 8x4
grid, column-major; logical core (0,0) is the controller and always in the group with more pages. Per core:
  reader (DM2): reserve(n); per page 1 read + read barrier; handshake (others: sem.up to the controller, wait for its
  multicast; controller: wait for every other core's up, then one semaphore multicast per region, 0-3 regions);
  push(n); full barrier.
  writer (DM3): wait_front(n); per page 1 write + write barrier; pop(n).
Writes start only after every core has read (the overlap-safety handshake). The scratch DFB holds all n pages and
implicit DFB sync is off, so the reader never waits for space and exits without waiting for the drain.
The controller finishes last (it multicasts before it pushes), so each load is modeled as the controller would run
it: an upper bound for the other cores, exact for the core that sets the span.

Cost keys (cycles; prefix mvo = Overlap, mvr = OverlapRowMajor): <p>.start_{reader,writer}; <p>.rd_setup, rd_fixed (per
page), rd_cpb (DMA cycles per byte), rd_tail; the controller's handshake waits until the slowest other core's
last read + <p>.sync_up (its semaphore increment arriving; that core is evaluated alongside), then costs
<p>.sync_core x (cores - 1) + mc x regions (its fixed part is in rd_setup); <p>.wr_setup, wr_fixed (per page), wr_cpb, wr_tail. L1 operands: <p>.{rd,wr}_cpb_l1, {rd,wr}_lat_l1,
rd_cpb_l1g: cycles per byte per reading core (every core reads from all L1 banks at once, so the
network's share per core shrinks with the core count). ROW_MAJOR pages whose aligned size (64 B DRAM, 16 B L1) is
not a power of two: + <p>.rd_npow2_setup before the first read, + rd_npow2 per read, + wr_npow2 per write.
"""

from __future__ import annotations

import math

from .. import dram
from ..engine import DMBatch, Group, Program, Ring, Server, Thread, ThreadStats, TransferLog
from .layout import DRAM_ALIGN, ELEM, GRID, GRID_X, GRID_Y, L1_ALIGN, TILE, _pages

OPS = ("move",)
PREFIX = {("move", "Overlap"): "mvo", ("move", "OverlapRowMajor"): "mvr"}
FAMILY = {"move": "move"}
L1_USABLE = 3670016  # bytes of L1 a core can give the scratch DFB (approximation of l1_size - base allocator address)
FIT_INIT = {"sync_up": 100.0, "sync_core": 1.0, "mc": 30.0, "rd_cpb_l1g": 0.001, "rd_npow2_setup": 50.0,
            "rd_npow2": 5.0, "wr_npow2": 1.0}  # initial values of this module's own terms (fit_layout.py)


def split_work(pages):
    """split_work_to_cores(grid, pages) column-major: (cores, [(pages, cores)] group 1 first)."""
    ncores = min(GRID, pages)
    n1, extra = divmod(pages, ncores)
    if extra == 0:
        return ncores, [(n1, ncores)]
    return ncores, [(n1 + 1, extra), (n1, ncores - extra)]


def corerangeset(ncores):
    """num_cores_to_corerangeset(ncores, 8x4, row_wise=false): full columns, then a partial column; the full grid
    when ncores >= 32. Ranges as (x_end, y_end) from their start column."""
    if ncores >= GRID:
        return [((0, 0), (GRID_X - 1, GRID_Y - 1))]
    ranges, left, x = [], ncores, 0
    if left > GRID_Y:  # strict, as in work_split.cpp: exactly one column is a 'partial' range
        full = left // GRID_Y
        ranges.append(((0, 0), (full - 1, GRID_Y - 1)))
        left -= full * GRID_Y
        x = full
    if left > 0:
        ranges.append(((x, 0), (x, left - 1)))
    return ranges


def multicast_regions(ncores):
    """get_multicast_regions: the controller's range splits into the columns right of it and the cells below it
    (either may be empty); a second range is one more region."""
    ranges = corerangeset(ncores)
    (_, _), (ex, ey) = ranges[0]
    return (ex > 0) + (ey > 0) + (len(ranges) - 1)


def _align(nbytes, a):
    return -(-nbytes // a) * a


def parse_mem(mem):
    parts = mem.split("_")
    return parts[0], "ov" in parts, "rm" in parts


def plan(op, H, W, dtype="bf16", mem="dram_ov", multicore=True):
    """mem = <dram|l1>[_ov][_rm] (see calibration/runners/move.py). Returns the factory and [(cfg, cores)], the
    load holding the controller (group 1) first."""
    elem = ELEM[dtype]
    buf, overlap, rm = parse_mem(mem)
    if buf not in ("dram", "l1"):
        raise ValueError(f"move with {buf} input (sharded: MoveSharded) is not modeled yet")
    if not overlap:
        raise ValueError("move to a non-overlapping output runs MoveProgramFactory (legacy copy DataMovementKernels), "
                         "which Quasar rejects (TT_FATAL); not modeled yet")
    if rm:
        pages, page_bytes = H, W * elem
    else:
        pages, page_bytes = math.ceil(H / TILE) * math.ceil(W / TILE), TILE * TILE * elem
    per_bank = math.ceil(pages / GRID) * _align(page_bytes, L1_ALIGN)  # calculate_bank_size_spread over the L1 banks
    # fits_in_cb: the scratch DFB (one bank's share) must fit L1; with an L1 output it must also sit below the output
    if per_bank * (2 if buf == "l1" else 1) > L1_USABLE:
        raise ValueError("move: the per-core share does not fit L1 (fits_in_cb false), so the device op would pick "
                         "MoveProgramFactory, which Quasar rejects; not modeled yet")
    ncores, groups = split_work(pages)
    regions = multicast_regions(ncores)
    tile_bytes = TILE * TILE * elem
    # the controller waits for the other cores' semaphore increments: the slowest other core has group 1's pages if
    # group 1 has another core, else group 2's
    others = 0 if ncores == 1 else (groups[0][0] if groups[0][1] > 1 else groups[1][0])
    # the address generators scale page ids by the buffer's aligned page size: a shift when it is a power of two, else
    # a multiply (and a larger kernel that starts later) - ROW_MAJOR only, a tile page is always a power of two
    a = _align(page_bytes, DRAM_ALIGN if buf == "dram" else L1_ALIGN)
    npow2 = a & (a - 1) != 0
    loads = [(dict(pages=n, page_bytes=page_bytes, ncores=ncores, regions=regions, other_pages=others, npow2=npow2,
                   in_place=buf, out_place=buf, blocks=n, wt=1, tiles=n * page_bytes / tile_bytes, elem=elem), cores)
             for n, cores in groups]
    # DRAM channels: split_work_to_cores gives cores consecutive pages, group 1 (one page more) first, column-major
    k = first = 0
    for cfg, cores in loads:
        idx = list(range(k, k + cores))
        cfg["_pg"] = {"noc": [(2 + i // GRID_Y, 2 + i % GRID_Y) for i in idx],
                      "tile": [first + j * cfg["pages"] for j in range(cores)]}
        k, first = k + cores, first + cores * cfg["pages"]
    return ("OverlapRowMajor" if rm else "Overlap"), loads


def _noc(c, p, d, place, nbytes, ncores):
    """(latency, DMA gap, queue service) of one page transfer; every page waits for its own barrier, so one transfer
    is in flight per thread and no queue service applies."""
    gap = nbytes * c[f"{p}.{d}_cpb"]
    lat = 0.0
    if place == "l1":
        gap += nbytes * c[f"{p}.{d}_cpb_l1"]
        if d == "rd":
            gap += nbytes * ncores * c[f"{p}.rd_cpb_l1g"]
        lat = c[f"{p}.{d}_lat_l1"]
    return lat, gap, 0.0


def _dx(pg, i, nbytes):
    """Page i's DRAM channel timing (dram.xfer) when the channels time this side, else nothing."""
    return () if pg is None else (dram.xfer(pg.side, pg.tiles([i]), nbytes),)


def _reads(c, p, cfg, ring, n, log, queue, pg=None):
    lat, gap, svc = _noc(c, p, "rd", cfg["in_place"], cfg["page_bytes"], cfg["ncores"])
    np2 = cfg["npow2"]
    fixed = c[f"{p}.rd_fixed"] + (c[f"{p}.rd_npow2"] if np2 else 0.0)
    return [("busy", c[f"{p}.rd_setup"] + (c[f"{p}.rd_npow2_setup"] if np2 else 0.0), True)] + [
        DMBatch([], "space", ring, fixed, 0.0, (0.0, 0.0, 0), 0.0,
                [(1, lat, gap, svc, queue) + _dx(pg, i, cfg["page_bytes"])], log, True, "push", 0.0, dma=True)
        for i in range(n)]


def build(op, cfg, c, p):
    n = cfg["pages"]
    ring = Ring("scratch", 1, n + 1, n)  # holds every page of the core: never full
    allp = list(range(n))
    st_r = ThreadStats("reader", 0, start=c[f"{p}.start_reader"])
    st_w = ThreadStats("writer", 0, start=c[f"{p}.start_writer"])
    # DRAM channels (None unless the estimate times DRAM with them): this core stands for every core of its load; the
    # other core the controller waits for only looks at the channels (its load accounts for its traffic)
    prd, pwr = _pages(cfg, True, False), _pages(cfg, False, False)
    rd = _reads(c, p, cfg, ring, n, 0, "rd", prd)
    groups, logs, queues = [], [TransferLog(), TransferLog()], {"rd": Server(), "wr": Server()}
    if cfg["other_pages"]:
        # the slowest other core's reader (same start and costs, its own NoC queue), whose semaphore increment reaches
        # the controller <p>.sync_up after its last read; the controller's wait ends at the later of the two
        sem = Ring("sem", 1, 2, 1)
        m = cfg["other_pages"]
        ghost = _reads(c, p, cfg, Ring("other", 1, m + 1, m), m, 2, "other", _pages(cfg, True, False, take=False)) + \
            [("push", sem, [0], c[f"{p}.sync_up"])]
        groups.append(Group([Thread(ThreadStats("other_reader", 0, start=c[f"{p}.start_reader"]), ghost)], dm=True))
        logs.append(TransferLog())
        queues["other"] = Server()
        rd.append(("wait", "data", sem, [0], True))
    rd += [("busy", c[f"{p}.sync_core"] * (cfg["ncores"] - 1) + c[f"{p}.mc"] * cfg["regions"], True),
           ("push", ring, allp, 0.0), ("busy", c[f"{p}.rd_tail"], True)]
    lat, gap, svc = _noc(c, p, "wr", cfg["out_place"], cfg["page_bytes"], cfg["ncores"])
    wr = [("busy", c[f"{p}.wr_setup"], True), ("wait", "data", ring, allp, True)]
    wfix = c[f"{p}.wr_fixed"] + (c[f"{p}.wr_npow2"] if cfg["npow2"] else 0.0)
    wr += [DMBatch(allp if i == n - 1 else [], "data", ring, wfix, 0.0, (0.0, 0.0, 0), 0.0,
                   [(1, lat, gap, svc, "wr") + _dx(pwr, i, cfg["page_bytes"])], 1, True, "pop", 0.0, dma=True)
           for i in range(n)]
    wr += [("busy", c[f"{p}.wr_tail"], True)]
    groups = [Group([Thread(st_r, rd)], dm=True)] + groups + [Group([Thread(st_w, wr)], dm=True)]
    return Program(groups=groups, logs=logs, queues=queues, stats=[st_r, st_w], drain=0.0, first_output=None)


# L1 transfer terms the fitter adds for L1 runs (calibration/fit_layout.py); no whole-op L1 bound (l1_gbw): reads and
# writes run in separate phases, and the reads' shared-network cost is rd_cpb_l1g
L1_KEYS = ["rd_cpb_l1", "wr_cpb_l1", "rd_lat_l1", "wr_lat_l1"]  # one transfer in flight: no queue service terms


def fit_keys(op, factory):
    keys = ["rd_setup", "rd_fixed", "rd_cpb", "sync_up", "sync_core", "mc", "rd_tail", "wr_setup", "wr_fixed", "wr_cpb",
            "wr_tail", "rd_cpb_l1g"]
    return keys + (["rd_npow2_setup", "rd_npow2", "wr_npow2"] if factory == "OverlapRowMajor" else [])


def mem_of(workload, knobs):
    """API workload -> (mem, label). placements.a (or knobs.input) = dram | l1, input and output; workload.layout =
    tile (default) | row_major; workload.output = overlap (default: the output overlaps the input, the only interleaved
    case Quasar runs) | separate (non-overlapping: MoveProgramFactory, rejected on Quasar)."""
    place = knobs.get("input") or (workload.get("placements") or {}).get("a", "dram")
    if place not in ("dram", "l1"):
        raise ValueError("move: placements.a must be dram or l1 (interleaved); sharded move is not modeled yet")
    layout = workload.get("layout", "tile")
    out = workload.get("output", "overlap")
    mem = place + ("_ov" if out == "overlap" else "") + ("_rm" if layout == "row_major" else "")
    return mem, f"{layout}, {place}, {'overlapping' if out == 'overlap' else 'separate'} output"
