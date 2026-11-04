"""The engine reproduces the snapshot in tests/golden.json (tests/make_golden.py) for every sampled case."""

import json
import os

from make_golden import SMALL, cases, evaluate_case, summary


def test_engine_matches_golden_sample():
    golden = json.load(open(SMALL))
    want = {json.dumps({k: v for k, v in g.items() if k != "out"}, sort_keys=True): g["out"] for g in golden}
    seen = set()
    for key, cfg, rt, costs in cases():
        k = json.dumps(key, sort_keys=True)
        if k not in want:
            continue
        got, ref = summary(evaluate_case(cfg, rt, costs, key["n"])), want[k]
        if "deadlock" in ref:
            assert got == ref, key
        else:
            assert abs(got["makespan"] - ref["makespan"]) <= 1e-6 * ref["makespan"], key
            assert got["first_output"] == ref["first_output"] or abs(got["first_output"] - ref["first_output"]) <= 1e-6, key
            for role, b in ref["busy"].items():
                assert abs(got["busy"][role] - b) <= 1e-6 * max(1.0, b), (key, role)
        seen.add(k)
    assert seen == set(want)
