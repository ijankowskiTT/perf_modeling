"""Measurement import, prediction-error reports and explicit fitting of cost profiles.

Nothing here runs implicitly: `qperf calibrate report` compares a profile with a dataset, `qperf calibrate fit`
writes a NEW profile revision next to the old one, with the fit set, the residuals and the dataset recorded.
Records come from calibration/run_case.py (one JSON line per simulator run, knobs fully specified).
"""

from __future__ import annotations

import copy
import json
import math
import os
import statistics

from .api import analyze

LAYOUT_PLACEMENTS = {"ddd": ("dram", "dram", "dram"), "lll": ("l1", "l1", "l1"), "hhh": ("borrowed",) * 3,
                     "lld": ("l1", "l1", "dram"), "ddl": ("dram", "dram", "l1")}
KNOB_KEYS = {"R": "reader_threads", "C": "compute_threads", "W": "writer_threads", "B": "dm_batch",
             "T": "tiles_per_cycle", "E": "entries_per_thread"}


def load_records(path, sim=None):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") or (sim and r.get("sim") != sim):
                continue
            if r["layout"] not in LAYOUT_PLACEMENTS:
                continue
            out.append(r)
    return out


def knobs_of(spec):
    k = {}
    if spec and spec != "-":
        for part in spec.split(","):
            if part[0] in KNOB_KEYS:
                k[KNOB_KEYS[part[0]]] = int(part[1:])
    return k


def request_for(rec, costs_ref, profile_override=None):
    pa, pb, po = LAYOUT_PLACEMENTS[rec["layout"]]
    return {"schema_version": "1", "action": "estimate", "profiles": {"costs": costs_ref},
            "workload": {"op": rec["op"], "dtype": rec["dtype"], "num_tiles": rec["tpc"], "tiles_scope": "cluster",
                         "clusters": rec.get("clusters", 32), "placements": {"a": pa, "b": pb, "out": po}},
            "baseline": knobs_of(rec.get("knobspec")), "analysis": {"detail": "compact"}}


def tag(rec):
    return f"{rec['layout']}:{rec['op']}:{rec['tpc']}:{rec.get('knobspec')}"


def predict(records, costs_ref, what_if=None):
    rows = []
    for rec in records:
        req = request_for(rec, costs_ref)
        if what_if:
            req["what_if"] = what_if
        res = analyze(req)
        b = res.get("baseline", {})
        pred = b.get("time_ns", {}).get("nominal") if b.get("valid") else None
        meas = rec["span_med"]
        rows.append({"case": tag(rec), "measured": meas, "predicted": pred,
                     "rel_err": (pred - meas) / meas if pred else None,
                     "bottleneck": b.get("bottleneck", {}).get("resource"),
                     "output_correct": rec.get("max_abs_err", 0) <= 0.1, "status": res["status"],
                     "errors": b.get("errors")})
    return rows


def summarize(rows):
    errs = [abs(r["rel_err"]) for r in rows if r["rel_err"] is not None]
    if not errs:
        return {"n": 0}
    errs.sort()
    return {"n": len(errs), "mean_abs_rel_err": round(statistics.mean(errs), 4),
            "median_abs_rel_err": round(statistics.median(errs), 4),
            "p90_abs_rel_err": round(errs[int(0.9 * (len(errs) - 1))], 4), "max_abs_rel_err": round(errs[-1], 4),
            "unpredicted": sum(1 for r in rows if r["rel_err"] is None)}


def ranking_quality(rows, records):
    """Within each (layout, op, size) group, does the model pick the measured-best config? Reports regret."""
    groups = {}
    for r, rec in zip(rows, records):
        if r["predicted"] is None:
            continue
        groups.setdefault((rec["layout"], rec["op"], rec["tpc"]), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        if len(rs) < 3:
            continue
        best_meas = min(rs, key=lambda r: r["measured"])
        chosen = min(rs, key=lambda r: r["predicted"])
        # Kendall tau
        conc = disc = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a = (rs[i]["measured"] - rs[j]["measured"]) * (rs[i]["predicted"] - rs[j]["predicted"])
                conc += a > 0
                disc += a < 0
        out.append({"group": ":".join(map(str, g)), "n": len(rs), "kendall_tau": round((conc - disc) / max(1, conc + disc), 3),
                    "chosen": chosen["case"], "best_measured": best_meas["case"],
                    "regret": round(chosen["measured"] / best_meas["measured"] - 1, 4)})
    return out


# --------------------------------------------------------------------------------------------- fitting

def _objective(records, profile, variant_keys):
    from . import profiles as P

    P.register("costs", profile)
    rows = predict(records, profile["name"])
    e = [math.log(r["predicted"] / r["measured"]) ** 2 for r in rows if r["predicted"]]
    miss = sum(1 for r in rows if not r["predicted"])
    return (sum(e) / max(1, len(e))) + miss * 1.0


def _get(profile, key, variant):
    if key in profile["params"]:
        return profile["params"][key]
    return profile["compute_variants"][variant][key]


WIDE = {"dfb.credit_latency": (0.0, 800.0), "end.drain": (0.0, 600.0),
        "noc.read_latency.l1": (0.0, 3000.0), "noc.write_latency.l1": (0.0, 3000.0),
        "noc.cluster_rw_shared_bytes_per_cycle.l1": (5.0, 400.0), "cm.fixed": (0.0, 400.0), "noc.thread_bytes_per_cycle.l1": (2.0, 400.0),
        "noc.read_service.l1": (0.0, 400.0), "noc.write_service.l1": (0.0, 400.0)}


def fit(records, base_profile, keys, variant=None, sweeps=3, rel_span=0.4):
    """Coordinate descent (golden section per key) on mean squared log error. Writes a temporary profile under
    the base name while searching, so callers must restore or rename. Returns (profile, history)."""
    prof = copy.deepcopy(base_profile)
    prof["name"] = base_profile["name"] + "__fit_tmp"
    hist = []
    best = _objective(records, prof, keys)
    hist.append(best)
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(sweeps):
        for k in keys:
            p = _get(prof, k, variant)
            v0 = p["value"]
            lo = max(0.0, v0 * (1 - rel_span)) if v0 else 0.0
            hi = v0 * (1 + rel_span) if v0 else 50.0
            if k.endswith("_frac"):
                lo, hi = 0.0, 1.0
            elif k in WIDE:
                lo, hi = WIDE[k]
            a, b = lo, hi
            c, d = b - gr * (b - a), a + gr * (b - a)

            def f(x):
                p["value"] = x
                return _objective(records, prof, keys)

            fc, fd = f(c), f(d)
            for _ in range(12):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = f(d)
            x = (a + b) / 2
            fx = f(x)
            if fx <= best:
                best, p["value"] = fx, round(x, 3)
            else:
                p["value"] = v0
            hist.append(best)
    from . import profiles as P

    P.unregister("costs", prof["name"])
    prof["name"] = base_profile["name"]
    return prof, hist
