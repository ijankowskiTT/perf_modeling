"""Resource lower bounds (generalized roofline) for one cluster and for the whole operation.

T_LB = max(start_r + D_r / Q_r) over resources r. A bound is never an achievable time by itself; it is reported
next to the pipeline estimate so that a gap between the two can be read as waiting (dependencies, buffers).
"""

from __future__ import annotations

import math

from .pipeline import compute_scale, reader_batch_fixed


def _lin(c, prefix, n, scale=1.0):
    f = c[prefix + ".fixed"]
    if n == 1 and (prefix + ".fixed_n1") in c:
        f = c[prefix + ".fixed_n1"]
    return (f + c[prefix + ".per_tile"] * n) * scale


def _dm_demand(work, batches, cfg, c):
    """Busiest thread's work; with a dynamic split the threads share it evenly, plus one claim per batch."""
    if not work:
        return 0.0
    if cfg.get("work_split") == "dynamic":
        return (sum(work) + sum(batches) * c["dq.claim"]) / len(work)
    return max(work)


def cluster_bounds(cfg, routing, traffic, c):
    """Per-resource busy demand in cycles of the cost profile's clock, with the resource's start offset."""
    B = cfg["dm_batch"]
    n_in = sum(1 for m in routing["in_moves"] if m)
    out = {}
    if n_in:
        per_batch = reader_batch_fixed(c, B, n_in)
        nbs = traffic["reader_batches_per_thread"]
        work = [nb * per_batch + nt * n_in * c["rd.issue"] for nb, nt in zip(nbs, traffic["tiles_per_thread"]["reader"])]
        out["reader_issue"] = {"start": c["start.reader"], "demand": _dm_demand(work, nbs, cfg, c)}
    if routing["out_moves"]:
        per_batch = c["wr.fixed_b1"] if B == 1 else c["wr.fixed"]
        nbs = traffic["writer_batches_per_thread"]
        work = [nb * per_batch + nt * c["wr.issue"] for nb, nt in zip(nbs, traffic["tiles_per_thread"]["writer"])]
        out["writer_issue"] = {"start": c["start.writer"], "demand": _dm_demand(work, nbs, cfg, c)}
    BC = cfg["tiles_per_cycle"]
    sc = compute_scale(c, cfg["C"])
    um, pk = [], []
    for nt in traffic["tiles_per_thread"]["compute"]:
        full, rem = divmod(nt, BC)
        u = full * max(_lin(c, "cu", BC, sc), _lin(c, "cm", BC, sc)) + (
            max(_lin(c, "cu", rem, sc), _lin(c, "cm", rem, sc)) if rem else 0)
        p = full * _lin(c, "cp", BC, sc) + (_lin(c, "cp", rem, sc) if rem else 0)
        um.append(u)
        pk.append(p)
    out["compute_unpack_math"] = {"start": c["start.compute"], "demand": max(um) if um else 0.0}
    out["compute_pack"] = {"start": c["start.compute"], "demand": max(pk) if pk else 0.0}
    rd_bw = c.get("noc.cluster_read_bytes_per_cycle", math.inf) or math.inf
    wr_bw = c.get("noc.cluster_write_bytes_per_cycle", math.inf) or math.inf
    shared = c.get("noc.cluster_rw_shared_bytes_per_cycle", math.inf) or math.inf
    if traffic["noc_read_bytes"] and rd_bw != math.inf:
        out["noc_read_port"] = {"start": c["start.reader"], "demand": traffic["noc_read_bytes"] / rd_bw}
    if traffic["noc_write_bytes"] and wr_bw != math.inf:
        out["noc_write_port"] = {"start": c["start.reader"], "demand": traffic["noc_write_bytes"] / wr_bw}
    if shared != math.inf and (traffic["noc_read_bytes"] + traffic["noc_write_bytes"]):
        out["noc_port_shared"] = {"start": c["start.reader"],
                                  "demand": (traffic["noc_read_bytes"] + traffic["noc_write_bytes"]) / shared}
    l1_bw = c.get("l1.cluster_bytes_per_cycle", math.inf) or math.inf
    if l1_bw != math.inf:
        out["l1_ports"] = {"start": min(c["start.reader"], c["start.compute"]), "demand": traffic["l1_logical_bytes"] / l1_bw}
    for v in out.values():
        v["finish_lb"] = v["start"] + v["demand"]
    return out


def global_dram_bound(total_dram_bytes, c):
    bw = c.get("dram.total_bytes_per_cycle", math.inf) or math.inf
    if not total_dram_bytes or bw == math.inf:
        return None
    eff = c.get("dram.efficiency", 1.0)
    return {"start": c["start.reader"], "demand": total_dram_bytes / (bw * eff),
            "finish_lb": c["start.reader"] + total_dram_bytes / (bw * eff)}
