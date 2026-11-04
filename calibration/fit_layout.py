"""Fit tilize/untilize cost terms (qperf/ops/layout.py) to simulator runs of calibration/run_dm_case.py.

  python calibration/fit_layout.py --data runs.jsonl --sim craq-b702fd90 --op tilize [--holdout H=1280] [--write craq-sim]

Coordinate descent (golden section per key, 4 sweeps) on mean squared log(pred/meas) over the fit runs; prints
fit and holdout error per factory. --write stores the values in profiles/costs/<profile>.json.
"""
import argparse
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qperf.ops.layout import estimate  # noqa: E402

FIT_KEYS = ["rd_cpb", "wr_cpb", "rd_setup", "wr_setup", "rd_fixed", "rd_issue", "rd_first", "rd_tail", "um_fixed", "um_per_tile", "pk_fixed", "pk_per_tile", "cu_first", "wr_fixed", "wr_issue",
            "wr_first", "wr_tail", "cu_tail", "drain"]
def keys_for(op, fac):
    """Cost terms a factory's kernels actually have (others stay at their initial value)."""
    from qperf.ops import registry
    from qperf.ops.layout import family_of

    m = registry.module_for(op)
    if m and hasattr(m, "fit_keys"):
        return list(m.fit_keys(op, fac))
    if family_of(op) == "move":  # no compute; only the interleaved side moves data
        common = ["rd_setup", "rd_tail", "wr_setup", "wr_fixed", "wr_tail", "drain"]
        if op == "interleaved_to_sharded":
            return common + ["rd_fixed", "rd_issue", "rd_first", "rd_cpb"]
        return common + ["wr_issue", "wr_first", "wr_cpb"]
    cu = ["cu_fixed", "cu_per_tile"] if family_of(op) == "tilize" else ["um_fixed", "um_per_tile", "pk_fixed", "pk_per_tile"]
    if fac == "Sharded":  # no NoC traffic: publish all, compute, pop all
        return ["rd_setup", "wr_setup", "rd_fixed", "rd_tail"] + cu + ["cu_first", "cu_tail", "wr_fixed", "wr_tail", "drain"]
    # padding fill: DMA zero-fill (multicore) or a CPU store loop (SingleCore)
    extra = (["rd_pcpb", "rd_pfix"] if fac == "SingleCore" else ["rd_zcpb"]) if op == "tilize_with_val_padding" else []
    return [k for k in FIT_KEYS if not k.startswith(("um_", "pk_", "cu_fixed", "cu_per_tile"))] + cu + extra


INIT = {  # realistic starting values (cycles): a flat start lets least squares settle in poor local minima
    "rd_zcpb": 0.01, "rd_pcpb": 0.25, "rd_pfix": 30, "rd_cpb": 0.018, "wr_cpb": 0.018, "rd_setup": 100, "wr_setup": 500, "rd_fixed": 20, "rd_issue": 25,
    "rd_first": 0, "rd_tail": 50, "cu_fixed": 100, "cu_per_tile": 50, "um_fixed": 100, "um_per_tile": 5, "pk_fixed": 50,
    "pk_per_tile": 10, "cu_first": 0, "wr_fixed": 20, "wr_issue": 35, "wr_first": 0, "wr_tail": 50, "cu_tail": 20,
    "drain": 10}


# fp32 (--dtype fp32): the bf16 fit stays; only the terms fp32 changes are fitted, as <p>.f32_<term> overrides
# (qperf.ops.layout.dtype_costs), starting from the bf16 values. Compute: UnpackToDest and fp32 DEST (half the DEST
# tiles); per-byte terms: rows and tiles twice as many bytes, the request counts unchanged.
FP32_TERMS = ["cu_fixed", "cu_per_tile", "cu_first", "cu_tail", "um_fixed", "um_per_tile", "pk_fixed", "pk_per_tile",
              "rd_cpb", "wr_cpb", "rd_zcpb", "rd_pcpb", "rd_pfix", "rd_cpb_l1", "wr_cpb_l1", "l1_gbw", "l1_conf_cpb"]


def load(path, sim, op, dtype="bf16"):
    rs = [json.loads(l) for l in open(path)]
    return [r for r in rs if not r.get("error") and r["sim"] == sim and r["op"] == op and r["max_abs_err"] < 0.05
            and r.get("dtype", "bf16") == dtype]


def predict(r, c):
    return estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"], trace=3)


ROLE_OF = {"reader": "QUASAR_DM2", "writer": "QUASAR_DM3"}
# The pack TRISC's kernel end is consistent only where compute ends last (borrowed DFBs, e.g. tilize Sharded).
ROLE_OF_SHARDED = dict(ROLE_OF, compute="QUASAR_NEO0_TRISC2")


def role_ends(r):
    """Measured end of each role on a fully loaded core (relative to the core's first kernel start)."""
    core = next(iter(r.get("core_detail", {}).values()), None)
    roles = ROLE_OF_SHARDED if r["mem"] == "hs" else ROLE_OF
    return {k: core["roles"][v][1] for k, v in roles.items() if core and v in core["roles"]}


def objective(rs, c, role_weight=0.5):
    """Mean squared log error of the busiest core's span, plus each role's end on a full core: the role ends tie
    every cost term to the thread that pays it."""
    e = []
    for r in rs:
        p = predict(r, c)
        e.append(math.log(p["span"] / r["span_max"]) ** 2)
        model_ends = {s["role"]: s["end"] for s in p["loads"][0][2]["threads"]}
        for role, meas in role_ends(r).items():
            if meas > 0 and model_ends.get(role, 0) > 0:
                e.append(role_weight * math.log(model_ends[role] / meas) ** 2)
    return sum(e) / len(e)


def golden(f, lo, hi, it=24):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(it):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
    return (a + b) / 2


def residuals(rs, c, role_weight=0.5):
    out = []
    for r in rs:
        p = predict(r, c)
        out.append(math.log(p["span"] / r["span_max"]))
        model_ends = {s["role"]: s["end"] for s in p["loads"][0][2]["threads"]}
        for role, meas in role_ends(r).items():
            if meas > 0 and model_ends.get(role, 0) > 0:
                out.append(math.sqrt(role_weight) * math.log(model_ends[role] / meas))
    return out


def fit(rs, c, keys, sweeps=8):
    """Bounded least squares on the log residuals (scipy, as in the tt-metal venv); coordinate descent without it."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return fit_cd(rs, c, keys, sweeps)
    c = dict(c)
    hi = [0.2 if "_cpb" in k else 5000.0 for k in keys]

    def f(x):
        c.update(zip(keys, x))
        return residuals(rs, c)

    best = None
    for start in ([min(c[k], h) for k, h in zip(keys, hi)], [h * 0.1 for h in hi]):  # two starts: initial values, flat
        res = least_squares(f, start, bounds=([0.0] * len(keys), hi), x_scale=[h / 50 for h in hi], diff_step=1e-3)
        if best is None or res.cost < best.cost:
            best = res
    c.update(zip(keys, best.x))
    return c


def fit_cd(rs, c, keys, sweeps=8):
    c = dict(c)
    for _ in range(sweeps):
        for k in keys:
            def f(v):
                c[k] = v
                return objective(rs, c)
            c[k] = golden(f, 0.0, 0.1 if k.endswith("_cpb") else max(3 * c[k], 2000.0))
    return c


def report(name, rs, c, worst=0):
    by, rows = {}, []
    for r in rs:
        p = predict(r, c)
        err = p["span"] / r["span_max"] - 1
        by.setdefault(p["factory"], []).append(abs(err))
        rows.append((err, f"{r['mem']} {r['H']}x{r['W']} {'mc' if r['multicore'] else 'sc'} meas {r['span_max']} pred {p['span']:.0f}"))
    for fac, e in sorted(by.items()):
        print(f"  {name:8s} {fac:11s} n={len(e):3d} mean {100 * statistics.mean(e):5.2f}%  max {100 * max(e):5.2f}%")
    for err, txt in sorted(rows, key=lambda x: -abs(x[0]))[:worst]:
        print(f"      {100 * err:+6.1f}%  {txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="one or more jsonl files, comma-separated")
    ap.add_argument("--sim", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--holdout", default="every4", help="every4, or key=value (e.g. H=1280)")
    ap.add_argument("--write", help="store the fitted terms in profiles/costs/<profile>.json")
    ap.add_argument("--emit", help="write the fitted terms (profile param entries) to this JSON file instead")
    ap.add_argument("--dtype", default="bf16", help="bf16, or fp32: fit <p>.f32_* overrides on top of the profile")
    ap.add_argument("--terms", help="fp32: comma-separated terms to fit (default: the factory's terms in FP32_TERMS)")
    ap.add_argument("--only", help="comma-separated cost prefixes to fit (default: every prefix in the runs)")
    a = ap.parse_args()
    f32 = a.dtype == "fp32"
    ops = a.op.split(",")  # several ops when their factories share kernels (untilize Block, ..._unpadding BlockInterleaved)
    rs = []
    for path in a.data.split(","):
        for op in ops:
            rs += load(path, a.sim, op, a.dtype)

    def modeled(r):
        try:
            plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])
            return True
        except ValueError:
            return False

    from qperf.ops.layout import plan
    skipped = [r for r in rs if not modeled(r)]
    rs = [r for r in rs if modeled(r)]
    if skipped:
        print(f"  ({len(skipped)} runs use factories not modeled yet, skipped)")
    if a.holdout == "every4":  # every 4th run of each factory, in a fixed order
        from qperf.ops.layout import plan
        rs.sort(key=lambda r: (plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])[0], r["mem"], r["H"], r["W"]))
        seen, hold = {}, []
        for r in rs:
            fac = plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])[0]
            seen[fac] = seen.get(fac, 0) + 1
            if seen[fac] % 4 == 2:
                hold.append(r)
    else:
        hk, hv = a.holdout.split("=")
        hold = [r for r in rs if str(r[hk]) == hv]
    fitset = [r for r in rs if r not in hold]
    from qperf.ops.layout import family_of, plan, prefix_of
    # group by cost prefix: factories that share kernels share one set of cost terms
    pre_of = lambda r: prefix_of(r["op"], plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])[0])  # noqa: E731
    fac_of = lambda r: plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])[0]  # noqa: E731
    # profile-wide flags the layout model reads (e.g. noc.barrier_all_kinds: shared L1 queue and barrier): fit with
    # the same values the predictions will use
    prof_name = a.write or {"craq-b702fd90": "craq-sim", "perf": "ttsim-perf"}.get(a.sim)
    flags = {}
    if prof_name:
        params = json.load(open(os.path.join(ROOT, "profiles", "costs", prof_name + ".json")))["params"]
        flags = {k: v["value"] for k, v in params.items() if k in ("noc.barrier_all_kinds",) or k.endswith(".dfb_blocks")}
        if f32:  # fp32 overrides sit on the whole bf16 profile
            flags = {k: v["value"] for k, v in params.items()}
    c = {}
    for p in sorted({pre_of(r) for r in rs}):
        if a.only and p not in a.only.split(","):
            continue
        fac = fac_of(next(r for r in rs if pre_of(r) == p))
        op0 = next(r for r in rs if pre_of(r) == p)["op"]
        fs = [r for r in fitset if pre_of(r) == p]
        hs = [r for r in hold if pre_of(r) == p]
        if not fs:  # nothing to fit on: fit on all of this factory's runs
            fs, hs = hs, []
        # kernel starts read off the profiler (median role start on a full core, relative to its first kernel)
        starts = {"reader": [], "compute": [], "writer": []}
        for r in fs:
            for core in list(r.get("core_detail", {}).values())[:1]:
                roles = core["roles"]
                starts["reader"].append(roles.get("QUASAR_DM2", [0])[0])
                starts["writer"].append(roles.get("QUASAR_DM3", [0])[0])
                starts["compute"].append(roles.get("QUASAR_NEO0_TRISC0", [0])[0])
        cf = dict(flags)
        sp = f"{p}.f32_" if f32 else f"{p}."
        cf.update({f"{sp}start_{k}": statistics.median(v) for k, v in starts.items()})
        from qperf.ops import registry
        mod = registry.module_for(op0)
        if not f32:
            cf.update({f"{p}.{k}": v for k, v in INIT.items()})
            cf.update({f"{p}.{k}": v for k, v in getattr(mod, "INIT", {}).items()})  # an op module's own terms
            cf.update({f"{p}.{k}": v for k, v in getattr(mod, "FIT_INIT", {}).items()})
        keys = keys_for(op0, fac)
        def l1_sides(r):  # which directions move data over the NoC from / to L1 (the plan's placements)
            cfg = plan(r["op"], r["H"], r["W"], r["dtype"], r["mem"], r["multicore"])[1][0][0]
            return cfg["in_place"] == "l1", cfg["out_place"] == "l1"

        rd_l1 = any(l1_sides(r)[0] for r in fs)
        wr_l1 = any(l1_sides(r)[1] for r in fs)
        in_l1 = lambda r: any(l1_sides(r))  # noqa: E731
        if rd_l1 or wr_l1:  # L1 transfer terms only where the runs can identify them
            l1_init = {}
            for d, on in (("rd", rd_l1), ("wr", wr_l1)):
                if on:
                    l1_init.update({f"{d}_cpb_l1": 0.01, f"{d}_lat_l1": 50.0, f"{d}_svc_l1": 20.0})
            # global L1 network: a hard bound plus a fair share per core moving L1 data (gradual contention)
            l1_init.update({"l1_gbw": 500.0, "l1_share_bw": 500.0})
            allowed = getattr(mod, "L1_KEYS", None)  # a module may leave some out
            l1 = [k for k in l1_init if allowed is None or k in allowed]
            if mod and hasattr(mod, "l1_fit_keys"):  # or say which L1 terms a simulator can identify
                l1 = list(mod.l1_fit_keys(op0, fac, a.sim))
            keys += l1
            if not f32:
                cf.update({f"{p}.{k}": l1_init[k] for k in l1 if k in l1_init})  # others: the module's FIT_INIT
        if f32:  # override only what fp32 changes, where the bf16 profile has the term; start from the bf16 value
            allowed = a.terms.split(",") if a.terms else FP32_TERMS
            if family_of(op0) == "move" and any(in_l1(r) for r in fs):  # L1 bank conflicts (0 in the bf16 fit)
                keys.append("l1_conf_cpb")
                cf.setdefault(f"{p}.l1_conf_cpb", 0.0)
            keys = [f"f32_{k}" for k in keys if k in allowed and f"{p}.{k}" in cf]
            cf.update({f"{p}.{k}": cf[f"{p}.{k[4:]}"] for k in keys})
        cf = fit(fs, cf, [f"{p}.{k}" for k in keys])
        print(f"{op0} {fac} ({p}) on {a.sim}: {len(fs)} fit runs, {len(hs)} holdout ({a.holdout})")
        shown = [k for k in cf if k.startswith(sp) and (".start_" in k or k.split(".", 1)[1] in keys)] if f32 else \
            [k for k in cf if k.startswith(p + ".")]
        print("  " + ", ".join(f"{k.split('.')[1]}={cf[k]:.4g}" + (f" (bf16 {cf[p + '.' + k.split('.')[1][4:]]:.4g})"
                                                                   if f32 and ".f32_" in k else "") for k in shown))
        report("fit", fs, cf, worst=3)
        if hs:
            report("holdout", hs, cf, worst=2)
        c.update({k: v for k, v in cf.items() if (k not in flags or f32) and k.startswith(sp) and
                  (".start_" in k or k.split(".", 1)[1] in keys)})
    if a.emit:  # for parallel work: merge into a profile later (calibration/merge_params.py)
        sig = lambda x: float(f"{x:.6g}")  # noqa: E731
        out = {k: {"value": sig(v), "low": sig(v * 0.95), "high": sig(v * 1.05),
                   "class": "measurement" if ".start_" in k else "measurement-fit",
                   "source": "profiler: median kernel start" if ".start_" in k else
                   f"fitted on {a.op} runs ({a.sim}), calibration/fit_layout.py"} for k, v in c.items()}
        old = json.load(open(a.emit)) if os.path.exists(a.emit) else {}
        old.update(out)
        json.dump(old, open(a.emit, "w"), indent=1)
        print("  emitted", len(out), "terms to", a.emit)
    if a.write:
        path = os.path.join(ROOT, "profiles", "costs", a.write + ".json")
        prof = json.load(open(path))
        for k, v in c.items():
            src = "profiler: median kernel start" if ".start_" in k else f"fitted on {a.op} runs ({a.sim}), calibration/fit_layout.py"
            sig = lambda x: float(f"{x:.6g}")  # noqa: E731  (cycles-per-byte terms are ~0.007: keep 6 significant digits)
            prof["params"][k] = {"value": sig(v), "low": sig(v * 0.95), "high": sig(v * 1.05),
                                 "class": "measurement" if ".start_" in k else "measurement-fit", "source": src}
        json.dump(prof, open(path, "w"), indent=1)
        open(path, "a").write("\n")
        print("  wrote", path)


if __name__ == "__main__":
    main()
