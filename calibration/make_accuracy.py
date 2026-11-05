"""Build profiles/accuracy.json: per cost profile and data-movement factory, the model's mean error against the
simulator runs in calibration/data (per dtype) and the calibrated shape range (largest rows, width and elements).

qperf/accuracy.py reads it to state an op-specific accuracy on every answer and to flag shapes beyond the calibrated
range as extrapolated. tests/test_accuracy.py recomputes the table and fails when it is stale. Rerun after a refit or
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


def main():
    table = {"generated_by": "calibration/make_accuracy.py",
             "basis": "mean |model / simulator - 1| at nominal costs over calibration/data/{layout,fp32layout,rowmajor,"
                      "move,reshard}-<profile>.jsonl (all runs, holdouts included); range = the largest rows (H), "
                      "width (W) and H x W among the factory's runs",
             "profiles": compute()}
    with open(OUT, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)
        f.write("\n")
    print(OUT, {k: len(v["factories"]) for k, v in table["profiles"].items()})


if __name__ == "__main__":
    main()
