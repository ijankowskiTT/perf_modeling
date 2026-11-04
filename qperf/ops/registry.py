"""Op modules beyond qperf/ops/layout.py. Each module defines:

  OPS            tuple of op names it models
  PREFIX         {(op, factory): cost prefix}
  FAMILY         {op: family}          (e.g. "move": no compute)
  plan(op, H, W, dtype, mem, multicore) -> (factory, [(cfg, cores)])   cfg has blocks, wt, in_place, out_place
  build(op, cfg, costs, prefix)        -> engine.Program for one core
  fit_keys(op, factory)                -> cost terms the fitter may change (optional)
  mem_of(workload, knobs)              -> the `mem` string plan() takes, from an API workload (optional)

Register a module by adding its import below; layout.plan / build / estimate, the API and the fitter pick it up.
"""

MODULES = []


def register(module):
    MODULES.append(module)
    return module


def module_for(op):
    for m in MODULES:
        if op in m.OPS:
            return m
    return None


# --- registered op modules (one line each) ---
