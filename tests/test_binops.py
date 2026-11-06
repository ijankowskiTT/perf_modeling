"""binary_ng ops beyond add / multiply: subtract, divide and multiply on the FPU kernel (fast_and_approximate_mode).

The native factory runs these four ops only (matches_quasar_native_slice); every other binary op, and every subtile
broadcast, runs on the Metal 2.0 (DFB) factory or fails on Quasar, and is answered as unsupported with the reason.
Calibration: calibration/data/binops-<profile>.jsonl (calibration/fit_binops.py); the per-variant errors live in
profiles/accuracy.json binary_ng (kept current by tests/test_accuracy.py)."""

import json
import os

import pytest

from qperf import analyze
from qperf import profiles as P
from qperf.calibration import load_records, predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = {"craq-sim": "craq-b702fd90", "ttsim-perf": "perf"}
ACC = json.load(open(os.path.join(ROOT, "profiles", "accuracy.json")))["binary_ng"]


def req(op, dtype="bf16", placement="dram", tiles=256, costs="craq-sim", knobs=None, kernel=None, action="estimate"):
    w = {"op": op, "dtype": dtype, "num_tiles": tiles, "tiles_scope": "cluster", "clusters": 32,
         "placements": {"a": placement, "b": placement, "out": placement}}
    if kernel:
        w["kernel_variant"] = kernel
    return {"schema_version": "1", "action": action, "profiles": {"costs": costs}, "workload": w,
            "baseline": knobs or {}, "analysis": {"detail": "compact"}}


def cfg_of(res):
    c = res["baseline"]["config"]
    return c["R"], c["C"], c["W"], c["dm_batch"], c["tiles_per_cycle"], c["entries_per_thread"]


# ------------------------------------------------------------------------------------------------ accuracy

# (profile, variant): (mean over all runs, holdout mean, mean over the DRAM / borrowed runs, mean over the L1 runs).
# subtract and FPU multiply run cycle for cycle as add, so they carry add's model error over a wider config set:
# on craq-sim single-thread L1 with tiles_per_cycle 4 is +10-13%; on ttsim-perf the L1 runs inherit add's L1
# transfer gap, worst for R=C=W=1 (up to 2x).
CAPS = {
    ("craq-sim", "sfpu:divide:bf16"): (0.015, 0.015, 0.015, 0.02),
    ("craq-sim", "sfpu:divide:fp32"): (0.015, 0.015, 0.015, 0.02),
    ("craq-sim", "sfpu:subtract:fp32"): (0.02, 0.02, 0.02, 0.025),
    ("craq-sim", "fpu:subtract:bf16"): (0.03, 0.035, 0.025, 0.035),
    ("craq-sim", "fpu:multiply:bf16"): (0.03, 0.035, 0.025, 0.035),
    ("ttsim-perf", "sfpu:divide:bf16"): (0.03, 0.03, 0.03, 0.03),
    ("ttsim-perf", "sfpu:divide:fp32"): (0.03, 0.03, 0.02, 0.05),
    ("ttsim-perf", "sfpu:subtract:fp32"): (0.04, 0.045, 0.01, 0.12),
    ("ttsim-perf", "fpu:subtract:bf16"): (0.08, 0.09, 0.02, 0.22),
    ("ttsim-perf", "fpu:multiply:bf16"): (0.08, 0.09, 0.02, 0.22),
}


@pytest.mark.parametrize("profile,variant", sorted(CAPS))
def test_per_op_accuracy_on_the_calibration_runs(profile, variant):
    e = ACC[profile][variant]
    mean, hold, db, l1 = CAPS[(profile, variant)]
    assert e["runs"] >= 60 and e["holdout"] <= hold and e["mean"] <= mean, e
    assert e["dram_borrowed"]["mean"] <= db and e["by_placement"]["l1"]["mean"] <= l1, e


@pytest.mark.parametrize("profile", sorted(SIM))
def test_every_run_is_native_and_predicted(profile):
    recs = load_records(os.path.join(ROOT, "calibration", "data", f"binops-{profile}.jsonl"), SIM[profile])
    assert len(recs) >= 300 and {r["op"] for r in recs} == {"subtract", "divide", "multiply"}
    assert sum(r["src"] == "B" for r in recs) == len(recs) // 4  # every 4th run held out
    assert all(r.get("kernel_variant") == "fpu" for r in recs if r["op"] == "multiply")
    rows = predict(recs[::7], profile)
    assert all(r["predicted"] for r in rows)


@pytest.mark.parametrize("profile", sorted(SIM))
def test_subtract_and_fpu_multiply_run_as_add(profile):
    """Every binops run whose configuration add also ran has add's span, and the model gives add's time."""
    add = {(r["layout"], r["tpc"], r["knobspec"], r["dtype"]): r["span_med"]
           for f in (f"{profile}.jsonl", f"fp32binary-{profile}.jsonl")
           for r in load_records(os.path.join(ROOT, "calibration", "data", f), SIM[profile]) if r["op"] == "add"}
    recs = load_records(os.path.join(ROOT, "calibration", "data", f"binops-{profile}.jsonl"), SIM[profile])
    paired = [(r, add[k]) for r in recs if r["op"] in ("subtract", "multiply")
              for k in [(r["layout"], r["tpc"], r["knobspec"], r["dtype"])] if k in add]
    assert len(paired) >= 60 and all(r["span_med"] == s for r, s in paired)
    for op, dtype, kernel in (("subtract", "bf16", None), ("subtract", "fp32", None), ("multiply", "bf16", "fpu")):
        for pl in ("dram", "l1", "borrowed"):
            a = analyze(req("add", dtype, pl, tiles=128, costs=profile))
            b = analyze(req(op, dtype, pl, tiles=128, costs=profile, kernel=kernel))
            assert b["status"] == "ok" and b["baseline"]["time_ns"] == a["baseline"]["time_ns"], (op, dtype, pl)


def test_same_as_variant_resolves_to_the_seed_terms():
    for name in ("craq-sim", "ttsim-perf", "silicon-hypothesis"):
        prof = P.load("costs", name)[0]
        for var, seed in (("fpu:subtract:bf16", "fpu:add:bf16"), ("fpu:multiply:bf16", "fpu:add:bf16"),
                          ("sfpu:subtract:fp32", "sfpu:add:fp32")):
            assert prof["compute_variants"][var]["_same_as"] == seed
            for sc in ("nominal", "low", "high"):
                assert P.resolve_costs(prof, var, scenario=sc) == P.resolve_costs(prof, seed, scenario=sc)
        # divide: the seed's unpack / pack terms, its own math terms
        for dt, seed in (("bf16", "sfpu:multiply:bf16"), ("fp32", "sfpu:multiply:fp32")):
            d, m = P.resolve_costs(prof, f"sfpu:divide:{dt}"), P.resolve_costs(prof, seed)
            assert d["cm.per_tile"] > 3 * m["cm.per_tile"]
            assert all(d[k] == m[k] for k in ("cu.per_tile", "cp.fixed", "cp.per_tile", "start.compute"))


def test_accuracy_statement_names_the_op():
    r = analyze(req("divide"))
    assert r["accuracy"].startswith("binary_ng divide bf16 (SFPU) on craq-sim: 0.7% mean error over 68 simulator runs")
    r = analyze(req("subtract", placement="l1", costs="ttsim-perf"))
    assert "with an L1-interleaved operand, as here" in r["accuracy"] and "runs cycle for cycle as add" in r["accuracy"]
    r = analyze(req("multiply", kernel="fpu", costs="ttsim-perf"))
    assert "fast_and_approximate_mode" in r["accuracy"] and "DRAM / borrowed operands, as here" in r["accuracy"]
    r = analyze(req("divide", "fp32", costs="silicon-hypothesis"))
    assert "silicon-hypothesis is a hypothesis" in r["accuracy"] and "divide fp32" in r["accuracy"]


# ------------------------------------------------------------------------------------------------ rule

@pytest.mark.parametrize("op,dtype,kernel,l1_rule,mtpc", [
    ("subtract", "bf16", None, (4, 4, 2, 8, 1, 16), 8),   # FPU: one DST section of 8, batched at R=C=W=1 only
    ("multiply", "bf16", "fpu", (4, 4, 2, 8, 1, 16), 8),
    ("divide", "bf16", None, (4, 4, 2, 8, 4, 16), 4),     # SFPU: 2 DST slots per tile
    ("divide", "fp32", None, (4, 4, 2, 8, 2, 16), 2),     # SFPU in a 32-bit DST
    ("subtract", "fp32", None, (4, 4, 2, 8, 2, 16), 2),
])
def test_dst_slots_and_l1_rule(op, dtype, kernel, l1_rule, mtpc):
    r = analyze(req(op, dtype, "l1", tiles=64, kernel=kernel))
    assert cfg_of(r) == l1_rule
    one = analyze(req(op, dtype, "l1", tiles=64, kernel=kernel,
                      knobs={"reader_threads": 1, "compute_threads": 1, "writer_threads": 1}))
    assert cfg_of(one)[4] == mtpc  # R=C=W=1: both kernels batch a full DST section
    if mtpc < 8:  # 8 is also the TTNN_QSR knob ceiling
        over = analyze(req(op, dtype, "borrowed", tiles=64, kernel=kernel, knobs={"tiles_per_cycle": 2 * mtpc}))
        assert any(f"DST section capacity {mtpc}" in e for e in over["baseline"]["errors"])


@pytest.mark.parametrize("op,dtype,dram_rule", [
    ("divide", "bf16", (4, 4, 2, 4, 4, 8)),    # share batches up to the SFPU DST section
    ("subtract", "bf16", (4, 4, 2, 8, 8, 16)),
    ("subtract", "fp32", (4, 4, 2, 2, 2, 4)),
    ("divide", "fp32", (4, 4, 2, 1, 1, 2)),    # batch_pays: fp32 divide keeps batch 1 beside DRAM
])
def test_dram_batching_rule(op, dtype, dram_rule):
    assert cfg_of(analyze(req(op, dtype, "dram", tiles=256))) == dram_rule


# ------------------------------------------------------------------------------------------------ unsupported

@pytest.mark.parametrize("op,dtype,kernel,why", [
    ("rsub", "bf16", None, "Metal 2.0 (DFB) factory"),
    ("maximum", "bf16", None, "Metal 2.0 (DFB) factory"),
    ("minimum", "fp32", None, "Metal 2.0 (DFB) factory"),
    ("logaddexp", "bf16", None, "Metal 2.0 (DFB) factory"),
    ("squared_difference", "bf16", None, "Metal 2.0 (DFB) factory"),
    ("rsub", "fp32", None, "rsub_binary_tile"),
    ("gt", "bf16", None, "not ported"),
    ("atan2", "bf16", None, "not ported"),
    ("divide", "bf16", "fpu", "native gate refuses it"),
    ("subtract", "bf16", "sfpu", "runs only the FPU kernel"),
    ("multiply", "fp32", "fpu", "runs only the SFPU kernel"),
    ("where", "bf16", None, "not an op qperf models"),
])
def test_unsupported_ops_say_why(op, dtype, kernel, why):
    r = analyze(req(op, dtype, kernel=kernel))
    assert r["status"] == "unsupported" and why in r["message"], r
    assert "baseline" not in r and "time_ns" not in json.dumps(r)


def test_divide_is_compute_bound_and_slower_than_multiply():
    d, m = analyze(req("divide", tiles=2048)), analyze(req("multiply", tiles=2048))
    assert d["baseline"]["bottleneck"]["resource"].startswith("compute")
    assert d["baseline"]["time_ns"]["nominal"] > 4 * m["baseline"]["time_ns"]["nominal"]


def test_routing_matches_the_simulator_runs():
    """calibration/data/binops-routing-craq-sim.jsonl: each binary op (and subtile broadcast) run once on craq-sim,
    with the factory that ran it. Native runs are modeled; the others are unsupported with the matching reason."""
    path = os.path.join(ROOT, "calibration", "data", "binops-routing-craq-sim.jsonl")
    rows = [json.loads(line) for line in open(path)]
    assert {r["factory"] for r in rows} == {"native", "metal_v2", "none"}
    for r in rows:
        if r.get("bcast"):
            assert r["factory"] == "metal_v2"  # the native gate refuses every subtile broadcast
            continue
        fpu = r.get("fast_and_approximate_mode") and r["dtype"] == "bf16" and r["op"] == "multiply"
        out = analyze(req(r["op"], r["dtype"], kernel="fpu" if fpu else None))
        if r["factory"] == "native":
            assert out["status"] == "ok", r
        elif r.get("fast_and_approximate_mode"):  # bf16 divide in fast mode: an FPU divide, refused by the gate
            assert analyze(req(r["op"], r["dtype"], kernel="fpu"))["status"] == "unsupported", r
        else:
            assert out["status"] == "unsupported", r
            assert ("Metal 2.0" if r["factory"] == "metal_v2" else "not ported") in out["message"], (r, out["message"])
