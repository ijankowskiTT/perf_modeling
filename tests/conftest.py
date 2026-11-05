"""Shared test set-up: NO_DM_COSTS, an in-memory cost profile with binary_ng terms only (no data-movement factory
costs), for the "a profile without the op's costs gives needs_input, never a number" checks."""

import copy

from qperf import profiles

NO_DM_COSTS = "test-no-data-movement-costs"


def _register():
    p = copy.deepcopy(profiles.load("costs", "ttsim-perf")[0])
    p["name"] = NO_DM_COSTS
    p["params"] = {k: v for k, v in p["params"].items()
                   if k.split(".", 1)[0] in ("start", "end", "rd", "wr", "compute", "dfb", "dq", "noc")}
    profiles.register("costs", p)


_register()
