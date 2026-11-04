"""Snapshot of pipeline evaluations over a spread of workloads, configs, profiles and schedule variants, so an engine
or op-description refactor can be checked number for number. Regenerate only when a model change is intended:
    python tests/make_golden.py
Writes the full snapshot (makespan, per-role stats, 16 thread traces; ~60 MB, outside the repo, $QPERF_GOLDEN) and a
1-in-50 sample (makespan, first output, per-role busy) to tests/golden.json, checked by tests/test_golden.py.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qperf.api import DEFAULT_SPACE, effective_costs, resolve_config, resolve_workload, routing_for  # noqa: E402
from qperf.constraints import check  # noqa: E402
from qperf.pipeline import Deadlock, evaluate  # noqa: E402
from qperf.profiles import load, resolve_costs  # noqa: E402

OUT = os.environ.get("QPERF_GOLDEN", "/localdev/ijankowski/qperf-golden/golden_pipeline.json")
PLACEMENTS = [("dram",) * 3, ("l1",) * 3, ("borrowed",) * 3, ("borrowed", "borrowed", "l1"), ("borrowed", "dram", "dram"),
              ("l1", "l1", "dram"), ("dram", "dram", "l1")]


def cases():
    arch, _ = load("architecture", "quasar")
    kernel, _ = load("kernel", "binary-ng-native")
    space = [dict(zip(DEFAULT_SPACE, v)) for v in itertools.product(*DEFAULT_SPACE.values())]
    for prof in ("craq-sim", "ttsim-perf", "silicon-hypothesis"):
        cost_profile, _ = load("costs", prof)
        for op, plc, n in itertools.product(("add", "multiply"), PLACEMENTS, (1, 7, 64, 300)):
            wl, _ = resolve_workload({"op": op, "dtype": "bf16", "num_tiles": n, "tiles_scope": "cluster", "clusters": 1,
                                      "placements": dict(zip(("a", "b", "out"), plc))}, arch)
            for k, knobs in enumerate(space):
                if (k + n + len(op)) % 7:  # a deterministic 1-in-7 sample of the knob space
                    continue
                for variant in ({}, {"barriers": "per_thread"}, {"work_split": "dynamic"}):
                    cfg, _ = resolve_config(wl, {**{a: b for a, b in knobs.items() if b}, **variant})
                    rt = routing_for(wl, cfg)
                    if check(cfg, rt, arch, kernel)[0]:
                        continue
                    for sc in ("nominal", "low"):
                        costs = effective_costs(resolve_costs(cost_profile, wl["variant"], None, sc), rt)
                        yield {"profile": prof, "op": op, "placements": plc, "n": n, "cfg": cfg, "scenario": sc}, cfg, rt, costs


def evaluate_case(cfg, rt, costs, n):
    pcfg = dict(cfg, in_moves=rt["in_moves"], out_moves=rt["out_moves"], tile_bytes=rt["tile_bytes"])
    try:
        r = evaluate(pcfg, costs, n, trace_limit=16)
    except Deadlock:
        return {"deadlock": True}
    return {"makespan": r["makespan"], "first_output": r["first_output"], "roles": r["roles"], "threads": r["threads"]}


SMALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")


def summary(out):
    if "deadlock" in out:
        return out
    return {"makespan": round(out["makespan"], 6), "first_output": out["first_output"] and round(out["first_output"], 6),
            "busy": {r: round(v["busy"], 6) for r, v in out["roles"].items()}}


if __name__ == "__main__":
    rows = [dict(key, out=evaluate_case(cfg, rt, costs, key["n"])) for key, cfg, rt, costs in cases()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), separators=(",", ":"))
    small = [dict({k: v for k, v in r.items() if k != "out"}, out=summary(r["out"])) for r in rows[::50]]
    json.dump(small, open(SMALL, "w"), separators=(",", ":"))
    print(len(rows), "cases ->", OUT, os.path.getsize(OUT) // 1024, "KB;", len(small), "->", SMALL,
          os.path.getsize(SMALL) // 1024, "KB")
