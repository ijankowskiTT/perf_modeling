"""Shared measurement for simulator case runners (calibration/run_dm_case.py, calibration/runners/<op>.py).

A runner: parse argv (op dtype H W mem multicore), call `profiler_log()` before opening the device, run the op,
check the output, `ttnn.ReadDeviceProfiler(device)`, close the device, then `report(...)` prints the JSON line
sweep.sh collects: every core's span (the op ends with its slowest core), role start/end, the kernel files used.
"""
import collections
import csv
import json
import os
import pathlib
import re
import statistics


def profiler_log():
    """Remove a previous profile log so this run's zones are the only ones; returns its path."""
    prof = pathlib.Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"
    if prof.exists():
        prof.unlink()
    return prof


def report(op, dtype, H, W, mem, mc, max_err, prof, extra=None):
    rows = list(csv.DictReader(prof.read_text().splitlines()[1:], skipinitialspace=True))
    start = collections.defaultdict(lambda: 1 << 62)
    end = collections.defaultdict(int)
    for r in rows:
        if "KERNEL" not in r["zone name"].strip():
            continue
        key = ((int(r["core_x"]), int(r["core_y"])), r["RISC processor type"].strip())
        t = int(r["time[cycles since reset]"])
        if r["type"].strip() == "ZONE_START":
            start[key] = min(start[key], t)
        else:
            end[key] = max(end[key], t)
    cores = {}
    for core in sorted({k[0] for k in start}):
        keys = [k for k in start if k[0] == core and k in end]
        t0 = min(start[k] for k in keys)
        cores[core] = {"span": max(end[k] for k in keys) - t0,
                       "roles": {k[1]: [start[k] - t0, end[k] - t0] for k in sorted(keys)}}
    spans = [c["span"] for c in cores.values()]
    kernels = (pathlib.Path(os.environ["TT_METAL_LOGS_PATH"]) / "generated" / "inspector" / "kernels.yaml").read_text()
    files = sorted({os.path.basename(m) for m in re.findall(r"[\w/.-]+\.cpp", kernels)})
    rec = {
        "op": op, "dtype": dtype, "H": H, "W": W, "mem": mem, "multicore": mc, "cores": len(spans),
        "span_max": max(spans), "span_med": statistics.median(spans), "span_min": min(spans),
        "spans": sorted(spans), "kernels": files, "max_abs_err": max_err,
        "core_detail": {f"{c[0]},{c[1]}": v for c, v in list(cores.items())[:4]}}
    rec.update(extra or {})
    print("JSON " + json.dumps(rec))
