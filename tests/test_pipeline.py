"""Hand-calculated cases and invariants of the fixed-schedule pipeline (no profiles involved)."""

import itertools
import math

import pytest

from qperf.pipeline import Deadlock, compute_chunks, evaluate, num_tcs, reader_batches

# A deliberately simple cost set: every fixed part zero unless a test sets it.
BASE = {
    "start.compute": 0.0, "start.reader": 0.0, "start.writer": 0.0, "start.dm_late": 0.0,
    "rd.issue": 0.0, "rd.fixed": 0.0, "rd.fixed_b1": 0.0, "rd.publish_frac": 1.0,
    "wr.issue": 0.0, "wr.fixed": 0.0, "wr.fixed_b1": 0.0, "wr.release_frac": 1.0,
    "cu.fixed": 0.0, "cu.per_tile": 0.0, "cm.fixed": 0.0, "cm.per_tile": 0.0, "cp.fixed": 0.0, "cp.per_tile": 0.0,
}


def cfg(R=1, C=1, W=1, B=1, T=1, E=2, moves=(True, True), out=True):
    return {"R": R, "C": C, "W": W, "dm_batch": B, "tiles_per_cycle": T, "entries_per_thread": E,
            "in_moves": list(moves), "out_moves": out, "tile_bytes": [2048, 2048, 2048]}


def costs(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


@pytest.mark.parametrize("n,threads,B", list(itertools.product([1, 7, 16, 33, 100], [1, 2, 4], [1, 2, 4, 8])))
def test_batch_walk_covers_every_tile_once(n, threads, B):
    for C in (1, 2, 4):
        if max(threads, C) % min(threads, C):
            continue
        tcs = num_tcs(threads, C, True)
        seen = []
        for t in range(threads):
            for batch in reader_batches(n, threads, t, B, tcs):
                assert 1 <= len(batch) <= B
                # a batch comes from ONE tile counter of the ring max(threads, C)
                assert len({i % max(threads, C) for i in batch}) == 1
                seen += batch
        assert sorted(seen) == list(range(n))


@pytest.mark.parametrize("n,threads,B", list(itertools.product([1, 7, 33, 100], [1, 2, 4], [1, 2, 8])))
def test_batch_count_matches_walk(n, threads, B):
    from qperf.traffic import count_batches

    for C in (1, 2, 4):
        if max(threads, C) % min(threads, C):
            continue
        tcs = num_tcs(threads, C, True)
        for t in range(threads):
            assert count_batches(n, threads, t, B, tcs) == len(reader_batches(n, threads, t, B, tcs))


def test_extrapolation_matches_exact_evaluation():
    from qperf.api import eval_cluster

    c = costs(**{"rd.fixed": 150.0, "rd.fixed_b1": 90.0, "rd.issue": 34.0, "wr.fixed": 100.0, "wr.fixed_b1": 50.0,
                 "wr.issue": 45.0, "cu.fixed": 360.0, "cu.per_tile": 5.0, "cp.fixed": 250.0, "cp.per_tile": 30.0,
                 "start.reader": 4000.0, "start.compute": 1100.0, "noc.read_latency": 470.0})
    for k in ({"R": 4, "C": 4, "W": 2, "dm_batch": 1, "tiles_per_cycle": 1, "entries_per_thread": 2},
              {"R": 2, "C": 4, "W": 2, "dm_batch": 4, "tiles_per_cycle": 4, "entries_per_thread": 8}):
        routing = {"in_moves": [True, True], "out_moves": True, "tile_bytes": [2048] * 3}
        exact = eval_cluster(k, routing, c, 9001, 10 ** 9)["makespan"]
        approx = eval_cluster(k, routing, c, 9001, 1024)
        assert approx["extrapolated"]
        assert approx["makespan"] == pytest.approx(exact, rel=1e-3)


@pytest.mark.parametrize("n,C,T", list(itertools.product([1, 5, 16, 33], [1, 2, 4], [1, 2, 8])))
def test_compute_chunks_cover_every_tile_once(n, C, T):
    seen = [i for c in range(C) for ch in compute_chunks(n, C, c, T) for i in ch]
    assert sorted(seen) == list(range(n))


def hand_recurrence(n, sR, sC, sW, K_in, K_out):
    """The teaching model of the project brief, section 6 step D (single servers, tile granularity)."""
    FR, FC, FW = {}, {}, {}
    g = lambda d, i: d.get(i, 0.0)  # noqa: E731
    for i in range(1, n + 1):
        FR[i] = sR + max(g(FR, i - 1), g(FC, i - K_in))
        FC[i] = sC + max(g(FC, i - 1), FR[i], g(FW, i - K_out))
        FW[i] = sW + max(g(FW, i - 1), FC[i])
    return FW[n]


@pytest.mark.parametrize("sR,sC,sW,K", [(10, 5, 3, 2), (3, 10, 4, 2), (2, 3, 12, 2), (7, 7, 7, 1), (5, 9, 2, 4)])
def test_single_server_matches_hand_recurrence(sR, sC, sW, K):
    # R=C=W=1, batch 1: reader cost per tile = fixed_b1, compute = cu.fixed_n1 (unpack) with pack free,
    # writer = fixed_b1. The model frees an input slot when unpack finishes (compute completion here).
    n = 12
    c = costs(**{"rd.fixed_b1": sR, "cu.fixed": sC, "wr.fixed_b1": sW})
    # Pack free and DST not binding: compute completion == unpack completion, as in the hand model.
    got = evaluate(cfg(E=K), c, n)["makespan"]
    want = hand_recurrence(n, sR, sC, sW, K, K)
    assert got == pytest.approx(want)


def test_infinite_buffers_approach_slowest_stage():
    n = 400
    c = costs(**{"rd.fixed_b1": 4.0, "cu.fixed": 9.0, "wr.fixed_b1": 6.0})
    t = evaluate(cfg(E=1000), c, n)["makespan"]
    assert t / n == pytest.approx(9.0, rel=0.01)


def test_zero_compute_is_bounded_by_dm():
    n = 64
    c = costs(**{"rd.fixed_b1": 10.0, "wr.fixed_b1": 3.0})
    t = evaluate(cfg(E=2), c, n)["makespan"]
    assert t >= 10.0 * n


@pytest.mark.parametrize("scale", [1.0, 0.75, 0.5, 0.25, 0.0])
def test_faster_compute_never_slows_a_fixed_schedule(scale):
    n = 96
    base = costs(**{"rd.fixed": 50.0, "rd.issue": 6.0, "wr.fixed": 30.0, "wr.issue": 7.0, "cu.fixed": 80.0,
                    "cu.per_tile": 9.0, "cp.fixed": 40.0, "cp.per_tile": 5.0, "start.reader": 300.0})
    cf = cfg(R=2, C=4, W=2, B=4, T=4, E=8)
    t_full = evaluate(cf, base, n)["makespan"]
    fast = dict(base)
    for k in ("cu.fixed", "cu.per_tile", "cp.fixed", "cp.per_tile"):
        fast[k] *= scale
    assert evaluate(cf, fast, n)["makespan"] <= t_full + 1e-9


def test_reader_bound_compute_speedup_is_negligible():
    n = 128
    c = costs(**{"rd.fixed_b1": 40.0, "cu.fixed": 20.0, "wr.fixed_b1": 5.0})
    t1 = evaluate(cfg(E=2), c, n)["makespan"]
    c2 = dict(c, **{"cu.fixed": 10.0})
    t2 = evaluate(cfg(E=2), c2, n)["makespan"]
    assert (t1 - t2) / t1 < 0.01


def test_writer_bound_backpressure_reaches_compute():
    n = 128
    c = costs(**{"rd.fixed_b1": 2.0, "cu.fixed": 3.0, "wr.fixed_b1": 20.0})
    r = evaluate(cfg(E=2), c, n)
    assert r["roles"]["compute"]["wait_space"] > 0
    assert r["roles"]["reader"]["wait_space"] > 0  # the reader eventually blocks on input slots too


def test_larger_batch_delays_first_output():
    c = costs(**{"rd.fixed": 30.0, "rd.fixed_b1": 30.0, "rd.issue": 10.0, "cu.fixed": 5.0, "wr.fixed": 5.0,
                 "wr.fixed_b1": 5.0})
    # with one tile the batch size cannot matter; with 8 tiles the first publication waits for the whole batch
    r1 = evaluate(cfg(B=1, E=16), c, 8)
    r8 = evaluate(cfg(B=8, E=16), c, 8)
    assert r8["first_output"] > r1["first_output"]  # publication waits for the whole batch
    assert r8["makespan"] < r1["makespan"]  # while the amortized fixed cost still wins overall


def test_batch_larger_than_capacity_deadlocks():
    with pytest.raises(Deadlock):
        evaluate(cfg(B=4, E=2), costs(), 16)


def test_borrowed_everything_is_compute_only():
    c = costs(**{"cu.fixed": 10.0, "rd.fixed_b1": 1000.0, "wr.fixed_b1": 1000.0})
    r = evaluate(cfg(C=4, moves=(False, False), out=False), c, 64)
    assert r["makespan"] == pytest.approx(16 * 10.0)


def test_transfer_latency_hidden_by_outstanding_requests():
    # With latency L and B requests before a barrier, per-batch cost ~ fixed + B*issue + L: larger batches
    # amortize L (Little's law: more requests in flight).
    c = costs(**{"rd.fixed": 10.0, "rd.fixed_b1": 10.0, "rd.issue": 5.0, "noc.read_latency": 500.0,
                 "cu.fixed": 1.0, "wr.fixed": 1.0, "wr.fixed_b1": 1.0})
    t1 = evaluate(cfg(B=1, E=2), c, 64)["makespan"]
    t8 = evaluate(cfg(B=8, E=16), c, 64)["makespan"]
    assert t8 < t1 / 4
    assert not math.isinf(t8)
