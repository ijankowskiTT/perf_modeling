"""The DRAM channel model (qperf/dram.py, dram.model = channels on silicon-hypothesis): idle-read latency only after
the idle window, per-channel serialisation, first-fit reservations, page interleaving, the representative core's
copies, flushed writes in the engine, and its use by binary_ng and the data-movement ops (craq-sim / ttsim-perf have
no channel model and are unchanged: tests/test_golden.py and the pinned report hashes in tests/test_bundle.py)."""

import pytest

from qperf import analyze, dram, profiles
from qperf.engine import DMBatch, Group, Program, Ring, Server, Thread, ThreadStats, TransferLog, run, run_merged
from qperf.validate import validate_response

C = {"dram.model": "channels", "dram.channels": 2, "dram.channel_noc": [[2, 7], [4, 0]],
     "dram.channel_bytes_per_cycle": 100.0, "dram.idle_latency": 50.0, "dram.idle_cycles": 8.0,
     "noc.hop_latency": 0.0, "dram.read_fixed_latency": 0.0, "dram.write_fixed_latency": 0.0}


def chans(**kw):
    return dram.Channels(dict(C, **kw))


# ------------------------------------------------------------------------------------------------- channels

def test_idle_latency_only_after_the_idle_window():
    ch = chans()
    assert ch.access(0, 0.0, 1000, True) == 50.0 + 10.0  # idle channel: + idle latency, then 1000 B / 100 B/cycle
    assert ch.access(0, 60.0 + 8.0, 1000, True) == 68.0 + 10.0  # within dram.idle_cycles of the last end: no wait
    assert ch.access(0, 78.0 + 8.5, 1000, True) == 86.5 + 50.0 + 10.0  # idle 8.5 > 8 cycles: waits again
    assert ch.access(0, 140.0, 1000, True) == 146.5 + 10.0  # arrives while the channel streams: queues, no idle wait
    assert ch.idle_reads == 2 and ch.reads == 4


def test_writes_never_pay_the_idle_latency():
    ch = chans()
    assert ch.access(1, 0.0, 1000, False) == 10.0
    assert ch.access(1, 500.0, 1000, False) == 510.0
    assert ch.idle_reads == 0


def test_a_channel_serialises_its_accesses_and_channels_run_in_parallel():
    ch = chans(**{"dram.idle_latency": 0.0})
    a = ch.access(0, 0.0, 2000, True)
    b = ch.access(0, 0.0, 2000, True)  # same channel, same time: after the first
    c = ch.access(1, 0.0, 2000, True)  # the other channel: in parallel
    assert (a, b, c) == (20.0, 40.0, 20.0)
    s = ch.stats(40.0)
    assert [x["bytes"] for x in s["channels"]] == [4000, 2000]
    assert [x["busy_frac"] for x in s["channels"]] == [1.0, 0.5] and s["bound"]


def test_reservations_fill_the_first_gap_that_fits():
    ch = chans(**{"dram.idle_latency": 0.0})
    assert ch.access(0, 100.0, 1000, False) == 110.0
    assert ch.access(0, 0.0, 500, False) == 5.0  # fits before the later window: order of evaluation does not matter
    assert ch.access(0, 95.0, 1000, False) == 120.0  # does not fit in [95, 100): after the window
    assert ch.starts[0] == [0.0, 100.0] and ch.ends[0] == [5.0, 120.0]  # touching windows merge


def test_copies_hold_the_channel_for_all_and_complete_with_the_middle_one():
    ch = chans(**{"dram.idle_latency": 0.0})
    assert ch.access(0, 0.0, 1000, True, copies=4) == 25.0  # 4 x 10 cycles held, the mean copy done at 25
    assert ch.access(0, 0.0, 1000, True) == 50.0  # the next request waits for all four
    assert ch.access(0, 0.0, 1000, True, take=False) == 60.0 and ch.busy[0] == 50.0  # a look reserves nothing


def test_mean_hops_and_latency_to_each_channel():
    ch = chans(**{"noc.hop_latency": 7.0, "dram.read_fixed_latency": 100.0})
    assert ch.mean_hops([(2, 2)]) == [5.0, 4.0]  # (2, 7): 0 + 5; (4, 0): 2 + 2
    s = dram.Side(ch, True, [(2, 2)])
    assert s.pre == [50.0 + 35.0, 50.0 + 28.0] and s.post == s.pre


# ------------------------------------------------------------------------------------------------ interleaving

def test_pages_interleave_over_the_banks_by_page_id():
    ch = chans()
    camp = dram.Side(ch, True, [(2, 2)] * 4, bases=[0, 2, 4, 6])  # every core's page base even: all on one bank
    assert camp.counts(0) == (4, 0) and camp.counts(1) == (0, 4) and camp.counts(7) == (0, 4)
    spread = dram.Side(ch, True, [(2, 2)] * 4, bases=[0, 1, 2, 3])
    assert spread.counts(0) == spread.counts(1) == (2, 2)
    assert spread.bank_counts({0: 3, 3: 1}) == (3, 1)  # a sharded buffer: shard k in bank k
    pg = dram.Pages(camp, chunk=4, stride=10)  # a core's 4-tile chunk of a 10-tile-wide tensor
    assert pg.tiles([0, 3, 4]) == [camp.counts(0), camp.counts(3), camp.counts(10)]
    assert dram.Pages(camp, nbr=2).rows(3, [0, 1]) == [camp.counts(32), camp.counts(33)]


def test_camping_on_one_bank_is_slower_than_spreading():
    """Four cores with one request each in flight: all on the same bank at the same time (camping: 4 copies of 10
    cycles, the mean one done after 25) take longer than spread over both banks (2 copies per bank, 15, then the
    next request waits for its bank's previous pair: 20 per request)."""
    def span(bases):
        ch = chans(**{"dram.idle_latency": 0.0})
        s = dram.Side(ch, True, [(2, 2)] * 4, bases=bases)
        t = 0.0
        for g in range(8):
            t = s.complete(t, s.counts(g), 1000)
        return t
    assert span([0, 2, 4, 6]) == 8 * 25.0 and span([0, 1, 2, 3]) == 15.0 + 7 * 20.0


# ------------------------------------------------------------------------------------------------ engine

def _dm_core(ch, n, ack=False, bases=(0,)):
    """One DM thread reading n pages from DRAM, one barrier per page (or, ack: writing them, flushed, with one write
    barrier at the end)."""
    side = dram.Side(ch, not ack, [(2, 2)], bases=bases)
    ring = Ring("r", 1, n + 1, n)
    logs = [TransferLog(), TransferLog()]
    steps = [DMBatch([i], "space", ring, 0.0, 0.0, (0.0, 0.0, 0), 1.0,
                     [(1, 0.0, 0.0, 0.0, "rd", dram.xfer(side, [side.counts(i)], 1000, 1 if ack else None))],
                     0, True, "push", 0.0) for i in range(n)]
    if ack:
        steps.append(DMBatch([], "data", ring, 0.0, 0.0, (0.0, 0.0, 0), 0.0, [], 1, True, "push", 0.0))
    st = ThreadStats("reader", 0)
    return Program(groups=[Group([Thread(st, steps)], dm=True)], logs=logs, queues={"rd": Server()}, stats=[st], dram=ch)


def test_engine_times_dram_requests_on_the_channels():
    ch = chans()
    r = run(_dm_core(ch, 4))
    # each read: issue 1, idle channel (50) + 10 streaming, its barrier before the next
    assert r["makespan"] == pytest.approx(4 * 61.0)
    assert r["dram"]["idle_read_frac"] == 1.0 and [c["bytes"] for c in r["dram"]["channels"]] == [2000, 2000]


def test_flushed_writes_wait_for_departure_and_the_final_barrier_for_the_acks():
    ch = chans(**{"dram.write_fixed_latency": 400.0})
    r = run(_dm_core(ch, 4, ack=True))
    # each flush only waits for the write to leave (issue 1 cycle): they leave at 1..4 and reach the banks 200 cycles
    # later, alternating; bank 1 streams pages 1 and 3 (10 cycles each) until 222; the final barrier waits for that
    # ack, 200 cycles back: 422 (one barrier per write, as the reads, would take ~4 x 410)
    assert r["makespan"] == pytest.approx(422.0)


def test_merged_cores_share_the_channels():
    alone = run(_dm_core(chans(**{"dram.idle_latency": 0.0}), 8, bases=(0, 2, 4, 6)))["makespan"]
    ch = chans(**{"dram.idle_latency": 0.0})
    outs = run_merged([_dm_core(ch, 8, bases=(0, 2, 4, 6)), _dm_core(ch, 8, bases=(0, 2, 4, 6))], ch)
    assert all(o["makespan"] > alone for o in outs)  # two loads of four cores queue on the same banks
    assert sum(c["bytes"] for c in outs[0]["dram"]["channels"]) == 2 * 4 * 8 * 1000


# ------------------------------------------------------------------------------------------------ ops

def binary(n=65536, pl=("dram", "dram", "dram"), costs="silicon-hypothesis", **kw):
    q = {"schema_version": "1", "action": "compare", "profiles": {"costs": costs},
         "workload": {"op": "add", "dtype": "bf16", "num_tiles": n, "placements": dict(zip(("a", "b", "out"), pl))},
         "candidates": [{"id": f"b{b}", "overrides": {"dm_batch": b}} for b in (1, 16)]}
    q.update(kw)
    return analyze(q)


def test_binary_ng_dram_add_queues_on_the_channels():
    r = binary()
    assert r["status"] == "ok" and validate_response(r) == []
    b = r["baseline"]
    total = sum(ch["bytes"] for ch in b["dram"]["channels"])
    assert total == pytest.approx(3 * 65536 * 2048, rel=1e-3)  # a, b and out once each, extrapolated share
    assert b["dram"]["bound"] and b["bottleneck"]["resource"] == "dram_global"
    assert b["time_ns"]["nominal"] >= b["bounds_ns"]["dram_global"]  # the channels' full rate: a lower bound
    b1 = next(c for c in r["candidates"] if c["id"] == "b1")
    assert b1["time_ns"]["nominal"] > b["time_ns"]["nominal"]  # one tile per barrier leaves the channels idle
    assert "DRAM: channel model (2 channels" in r["accuracy"]


def test_binary_ng_channels_carry_only_dram_operands():
    mixed = binary(4096, ("dram", "l1", "l1"))["baseline"]
    assert sum(ch["bytes"] for ch in mixed["dram"]["channels"]) == 4096 * 2048
    l1 = binary(4096, ("l1", "l1", "l1"))["baseline"]
    assert "dram" not in l1


@pytest.mark.parametrize("costs", ["craq-sim", "ttsim-perf"])
def test_simulator_profiles_have_no_channel_model(costs):
    assert "dram.model" not in profiles.load("costs", costs)[0]["params"]
    r = binary(costs=costs)
    assert all("dram" not in row for row in [r["baseline"]] + r["candidates"])


def test_a_cluster_layout_without_the_channel_model_changes_nothing():
    """binary_ng's build only uses the channels when the costs enable them: the same layout on ttsim-perf costs gives
    the same pipeline number for number."""
    from qperf.api import effective_costs, resolve_config, resolve_workload, routing_for
    from qperf.pipeline import evaluate
    from qperf.profiles import resolve_costs

    arch, _ = profiles.load("architecture", "quasar")
    wl, _ = resolve_workload({"op": "add", "dtype": "bf16", "num_tiles": 256, "tiles_scope": "cluster", "clusters": 32}, arch)
    cfg, _ = resolve_config(wl, {})
    rt = routing_for(wl, cfg)
    costs = effective_costs(resolve_costs(profiles.load("costs", "ttsim-perf")[0], wl["variant"]), rt)
    pcfg = dict(cfg, in_moves=rt["in_moves"], out_moves=rt["out_moves"], tile_bytes=rt["tile_bytes"])
    with_layout = dict(pcfg, dram={"bases": list(range(0, 32 * 256, 256)), "placements": ["dram"] * 3,
                                   "lat_in": [0.0, 0.0]})
    assert evaluate(pcfg, costs, 256) == evaluate(with_layout, costs, 256)


def layout(op, shape, costs="silicon-hypothesis", action="estimate", **w):
    return analyze({"schema_version": "1", "action": action, "profiles": {"costs": costs},
                    "workload": dict({"op": op, "dtype": "bf16", "shape": list(shape), "placements": {"a": "dram"}},
                                     **w)})


def test_layout_ops_report_channel_use():
    big = layout("tilize", (8192, 2048))["baseline"]
    assert big["dram"]["bound"] and big["bottleneck"]["resource"] == "dram_global"
    small = layout("tilize", (256, 512))["baseline"]
    assert not small["dram"]["bound"] and small["dram"]["idle_read_frac"] > 0.5  # sparse reads find idle channels
    mv = layout("move", (2048, 2048))["baseline"]
    assert sum(ch["bytes"] for ch in mv["dram"]["channels"]) == 2 * 2048 * 2048 * 2  # read once, written once
    rs = analyze({"schema_version": "1", "action": "estimate", "profiles": {"costs": "silicon-hypothesis"},
                  "workload": {"op": "reshard", "dtype": "bf16", "shape": [1024, 1024],
                               "reshard": {"from": {"strategy": "height", "cores": 2, "memory": "dram"},
                                           "to": {"strategy": "height", "cores": 8, "memory": "l1"}}}})["baseline"]
    assert [ch["bytes"] for ch in rs["dram"]["channels"]] == [1024 * 1024] * 2  # one DRAM shard per bank
    l1 = analyze({"schema_version": "1", "action": "estimate", "profiles": {"costs": "silicon-hypothesis"},
                  "workload": {"op": "reshard", "dtype": "bf16", "shape": [1024, 1024],
                               "reshard": {"from": {"strategy": "height", "cores": 8, "memory": "l1"},
                                           "to": {"strategy": "height", "cores": 4, "memory": "l1"}}}})["baseline"]
    assert "dram" not in l1


def test_reports_show_the_channels():
    from qperf.cli import _text
    from qperf.report import to_html, to_markdown

    q = {"schema_version": "1", "action": "compare", "profiles": {"costs": "silicon-hypothesis"},
         "workload": {"op": "tilize", "dtype": "bf16", "shape": [8192, 2048], "placements": {"a": "dram"}},
         "candidates": [{"id": "b8", "overrides": {"write_batch": 8}}]}
    r = analyze(q)
    assert "DRAM channels: (2,7) busy " in _text(r) and "; saturated" in _text(r)
    assert "DRAM channels busy " in to_markdown(q, r)
    html = to_html(q, r)
    assert html.count('aria-label="DRAM channel use"') == 2 and "DRAM bars: share of the run" in html


def test_what_ifs_on_the_aggregate_dram_keys_reach_the_channels():
    """The runner's DRAM what-ifs (noc.read_latency.dram, dram.total_bytes_per_cycle, dram.efficiency) move the
    channel parameters: the per-channel rate and the fixed latency follow, and the times with them."""
    from qperf.profiles import resolve_costs

    p = profiles.load("costs", "silicon-hypothesis")[0]
    base = resolve_costs(p, "fpu:add:bf16")
    c = resolve_costs(p, "fpu:add:bf16", {"dram.total_bytes_per_cycle": 1000.0, "noc.read_latency.dram": 1000.0})
    assert c["dram.channel_bytes_per_cycle"] == pytest.approx(1000.0 * 0.75 / 2)
    trip = dram.mean_trip(c)
    assert trip == pytest.approx(2 * 6.375 * 7)
    assert c["dram.read_fixed_latency"] + c["dram.idle_latency"] + trip == pytest.approx(1000.0)
    assert c["dram.idle_latency"] == base["dram.idle_latency"]
    low = resolve_costs(p, "fpu:add:bf16", {"noc.read_latency.dram": 100.0})  # below idle + hops: idle gives way
    assert low["dram.read_fixed_latency"] == 0.0 and low["dram.idle_latency"] == pytest.approx(100.0 - trip)

    def t(what_if, n=65536, clusters=32):
        return analyze({"schema_version": "1", "action": "estimate", "profiles": {"costs": "silicon-hypothesis"},
                        "workload": {"op": "add", "dtype": "bf16", "num_tiles": n, "clusters": clusters,
                                     "placements": {"a": "dram", "b": "dram", "out": "dram"}},
                        "what_if": what_if})["baseline"]["time_ns"]["nominal"]
    assert t({"cost_overrides": {"dram.total_bytes_per_cycle": 1000.0}}) < 0.7 * t({})  # bandwidth-bound: faster
    assert t({"cost_overrides": {"noc.read_latency.dram": 2000.0}}, 256, 1) > \
        t({}, 256, 1) + 1000  # one cluster, latency-bound: slower


def test_channel_model_stays_fast():
    s = analyze({"schema_version": "1", "action": "survey", "profiles": {"costs": "silicon-hypothesis"},
                 "workload": {"op": "tilize", "dtype": "bf16", "placements": {"a": "dram"}}})
    assert s["status"] == "ok" and s["runtime_ms"] < 6000  # target < 3 s
    assert binary()["runtime_ms"] < 2000  # target < 1 s
