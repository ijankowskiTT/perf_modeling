"""binary_ng schedule helpers shared by the op description (ops/binary_ng.py), traffic.py and bounds.py, and
evaluate(): one cluster's reader -> compute -> writer pipeline, evaluated by the generic engine (engine.py)."""

from __future__ import annotations

from .engine import Deadlock, run  # noqa: F401  (Deadlock is part of this module's interface)


def evaluate(cfg, costs, n_tiles, trace_limit=0):
    """Evaluate one cluster processing n_tiles output tiles: makespan, per-role aggregates, first output time.

    cfg: R, C, W, dm_batch, tiles_per_cycle, entries_per_thread, in_moves [a, b], out_moves, tile_bytes, and optionally
    barriers / work_split. costs: resolved costs (cycles), see profiles/costs/*.json."""
    from .ops.binary_ng import build

    return run(build(cfg, costs, n_tiles), trace_limit)

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
