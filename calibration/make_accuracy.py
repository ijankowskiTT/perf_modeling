"""Build profiles/accuracy.json: per cost profile and data-movement factory, the model's mean error against the
simulator runs in calibration/data (per dtype) and the calibrated shape range (largest rows, width and elements).

qperf/accuracy.py reads it to state an op-specific accuracy on every answer and to flag shapes beyond the calibrated
range as extrapolated. hypothetical_validation: a hypothetical design knob run on the simulators as a kernel variant
(calibration/data/<VALIDATION>-<profile>.jsonl, records with a `design` and a non-pinned kernel_variant): the model's
mean error on those runs and on their speedup over the pinned kernel at the same shape; answers whose only
non-default knob is that one say it is validated. tests/test_accuracy.py recomputes the table and fails when it is stale. Rerun after a refit or
new calibration runs:
    python calibration/make_accuracy.py
"""

import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qperf.layout_api import _costs  # noqa: E402
from qperf.ops.layout import estimate  # noqa: E402
from qperf.profiles import load  # noqa: E402

PROFILES = ("craq-sim", "ttsim-perf")
FAMILIES = ("layout", "fp32layout", "rowmajor", "move", "reshard")  # calibration/data/<family>-<profile>.jsonl
VALIDATION = ("tilize-wbatch",)  # calibration/data/<name>-<profile>.jsonl: kernel-variant runs of a design knob
OUT = os.path.join(ROOT, "profiles", "accuracy.json")


def compute():
    """{profile: {"revision", "factories": {"<op> <factory>": {"runs", "max_h", "max_w", "max_elems",
    "error": {dtype: {"mean", "runs"}}}}}}; mean = mean |model / simulator - 1| at nominal costs."""
    out = {}
    for prof in PROFILES:
        p, pid = load("costs", prof)
        c = _costs(p, "nominal", None)
        errs, rng = {}, {}
        for fam in FAMILIES:
            for line in open(os.path.join(ROOT, "calibration", "data", f"{fam}-{prof}.jsonl")):
                r = json.loads(line)
                try:
                    e = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"])
                except ValueError:  # a factory not modeled (answered as unsupported)
                    continue
                key = f"{r['op']} {e['factory']}"
                errs.setdefault((key, r["dtype"]), []).append(abs(e["span"] / r["span_max"] - 1))
                g = rng.setdefault(key, {"runs": 0, "max_h": 0, "max_w": 0, "max_elems": 0})
                g["runs"] += 1
                g["max_h"], g["max_w"] = max(g["max_h"], r["H"]), max(g["max_w"], r["W"])
                g["max_elems"] = max(g["max_elems"], r["H"] * r["W"])
        facs = {}
        for key in sorted(rng):
            facs[key] = dict(rng[key], error={dt: {"mean": round(statistics.mean(v), 4), "runs": len(v)}
                                              for (k, dt), v in sorted(errs.items()) if k == key})
        out[prof] = {"revision": pid["revision"], "factories": facs}
    return out


def validation():
    """{"<op> <factory> <knob>": {profile: {"mean", "max", "runs", "speedup_mean", "values", "data"}}}: mean
    |model / simulator - 1| over the kernel-variant runs of one design knob, and over each run's speedup against the
    pinned kernel's run of the same shape (same file), at nominal costs."""
    out = {}
    for prof in PROFILES:
        p, _ = load("costs", prof)
        c = _costs(p, "nominal", None)
        for name in VALIDATION:
            path = f"calibration/data/{name}-{prof}.jsonl"
            rs = [json.loads(line) for line in open(os.path.join(ROOT, path))]
            key = lambda r: (r["op"], r["dtype"], r["mem"], r["multicore"], r["H"], r["W"])  # noqa: E731
            pinned = {key(r): r for r in rs if r.get("kernel_variant", "pinned") == "pinned"}
            groups = {}
            for r in rs:
                d = {k: v for k, v in (r.get("design") or {}).items() if v is not None}
                if r.get("kernel_variant", "pinned") == "pinned" or len(d) != 1:
                    continue
                (knob, value), = d.items()
                e = estimate(r["op"], r["H"], r["W"], c, r["dtype"], r["mem"], r["multicore"], design=d)
                g = groups.setdefault(f"{r['op']} {e['factory']} {knob}", {"err": [], "sp": [], "values": set()})
                g["err"].append(abs(e["span"] / r["span_max"] - 1))
                g["values"].add(value)
                b = pinned.get(key(r))
                if b:
                    e0 = estimate(b["op"], b["H"], b["W"], c, b["dtype"], b["mem"], b["multicore"])
                    g["sp"].append(abs((e0["span"] / e["span"]) / (b["span_max"] / r["span_max"]) - 1))
            for k, g in groups.items():
                out.setdefault(k, {})[prof] = {
                    "mean": round(statistics.mean(g["err"]), 4), "max": round(max(g["err"]), 4), "runs": len(g["err"]),
                    "speedup_mean": round(statistics.mean(g["sp"]), 4) if g["sp"] else None,
                    "values": sorted(g["values"]), "data": path}
    return out


def main():
    table = {"generated_by": "calibration/make_accuracy.py",
             "basis": "mean |model / simulator - 1| at nominal costs over calibration/data/{layout,fp32layout,rowmajor,"
                      "move,reshard}-<profile>.jsonl (all runs, holdouts included); range = the largest rows (H), "
                      "width (W) and H x W among the factory's runs",
             "profiles": compute(),
             "hypothetical_validation": validation()}
    with open(OUT, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(OUT, {k: len(v["factories"]) for k, v in table["profiles"].items()})


if __name__ == "__main__":
    main()
