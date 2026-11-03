"""Exact work and data-movement accounting for one cluster (no timing).

Counts follow the kernels in kernels_qsr/: one NoC request per tile per moving operand, one barrier per
reader/writer batch, reserve+push per moving input per reader batch, wait+pop per writer batch, and per
compute chunk 2x wait_front + 2x pop_front (unpack TRISC) and reserve_back + push_back (pack TRISC).
"""

from __future__ import annotations

from .pipeline import num_tcs


def count_batches(n_tiles, threads, thread_id, dm_batch, tcs):
    """Number of reserve/push (wait/pop) batches of one DM thread: reader_batches without building the lists."""
    if dm_batch == 1:
        return len(range(thread_id, n_tiles, threads))
    tile_step = tcs * threads
    count = 0
    # counter c of this thread holds tiles thread_id + c*threads + k*tile_step; dm_batch of them per batch
    for c in range(tcs):
        first = thread_id + c * threads
        if first < n_tiles:
            count += -(-len(range(first, n_tiles, tile_step)) // dm_batch)
    return count


def cluster_traffic(cfg, routing, n_tiles):
    R, C, W = cfg["R"], cfg["C"], cfg["W"]
    B, BC = cfg["dm_batch"], cfg["tiles_per_cycle"]
    in_moves, out_moves = routing["in_moves"], routing["out_moves"]
    tb = routing["tile_bytes"]
    n_in = sum(1 for m in in_moves if m)
    rd_batches = []
    if n_in:
        tcs = num_tcs(R, C, True)
        rd_batches = [count_batches(n_tiles, R, t, B, tcs) for t in range(R)]
    wr_batches = []
    if out_moves:
        tcs = num_tcs(C, W, False)
        wr_batches = [count_batches(n_tiles, W, t, B, tcs) for t in range(W)]
    chunks = [-(-len(range(t, n_tiles, C)) // BC) for t in range(C)]
    noc_read_bytes = n_tiles * sum(tb[i] for i in range(2) if in_moves[i])
    noc_write_bytes = n_tiles * tb[2] if out_moves else 0
    sfpu = routing["sfpu"]
    unpack_reads = n_tiles * (tb[0] + tb[1])  # both operands are unpacked once (FPU: binary unpack; SFPU: 2 copies)
    pack_writes = n_tiles * tb[2]
    l1_traffic = noc_read_bytes + unpack_reads + pack_writes + noc_write_bytes
    dram_bytes = n_tiles * (sum(tb[i] for i in range(2) if routing["placements"][i] == "dram")
                            + (tb[2] if routing["placements"][2] == "dram" else 0))
    return {
        "tiles": n_tiles,
        "noc_read_requests": n_tiles * n_in,
        "noc_write_requests": n_tiles if out_moves else 0,
        "noc_read_bytes": noc_read_bytes,
        "noc_write_bytes": noc_write_bytes,
        "dram_bytes": dram_bytes,
        "l1_logical_bytes": l1_traffic,
        "reader_batches_per_thread": rd_batches,
        "reader_barriers": sum(rd_batches),
        "reader_reserve_push_ops": 2 * n_in * sum(rd_batches),
        "writer_batches_per_thread": wr_batches,
        "writer_barriers": sum(wr_batches),
        "writer_wait_pop_ops": 2 * sum(wr_batches),
        "compute_chunks_per_thread": chunks,
        "compute_dfb_ops": 6 * sum(chunks),
        "unpack_passes": n_tiles * (2 if sfpu else 1),
        "pack_passes": n_tiles,
        "tiles_per_thread": {
            "reader": [len(range(t, n_tiles, R)) for t in range(R)] if n_in else [],
            "compute": [len(range(t, n_tiles, C)) for t in range(C)],
            "writer": [len(range(t, n_tiles, W)) for t in range(W)] if out_moves else [],
        },
    }
