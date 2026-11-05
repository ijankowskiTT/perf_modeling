"""Benchmark of the tool itself (not of Quasar): latency of compare requests, bounded tune, CLI cold start, memory."""

from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

from .api import _cache, analyze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _req(action, n_tiles=4096, **kw):
    r = {"schema_version": "1", "action": action, "profiles": {"costs": "ttsim-perf"},
         "workload": {"op": "add", "dtype": "bf16", "num_tiles": n_tiles, "clusters": 32,
                      "placements": {"a": "dram", "b": "dram", "out": "dram"}},
         "baseline": {"reader_threads": 4, "compute_threads": 4, "writer_threads": 2, "dm_batch": 1,
                      "tiles_per_cycle": 1, "entries_per_thread": 2},
         "analysis": {"detail": "compact"}}
    r.update(kw)
    return r


EIGHT = [{"id": f"c{i}", "overrides": o} for i, o in enumerate([
    {"dm_batch": 8, "tiles_per_cycle": 8, "entries_per_thread": 16},
    {"dm_batch": 4, "tiles_per_cycle": 4, "entries_per_thread": 8},
    {"dm_batch": 2, "tiles_per_cycle": 2, "entries_per_thread": 4},
    {"reader_threads": 2, "writer_threads": 2},
    {"reader_threads": 2, "writer_threads": 4},
    {"reader_threads": 1, "writer_threads": 1},
    {"compute_threads": 2, "reader_threads": 2, "writer_threads": 2},
    {"dm_batch": 8, "tiles_per_cycle": 1, "entries_per_thread": 16},
])]


def _time(fn, repeat, clear_cache=True):
    ts = []
    for _ in range(repeat):
        if clear_cache:
            _cache.d.clear()
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"p50_ms": round(statistics.median(ts), 2), "p95_ms": round(ts[int(0.95 * (len(ts) - 1))], 2),
            "max_ms": round(ts[-1], 2)}


def run(repeat=20):
    analyze(_req("estimate"))  # load profiles once, as the targets say "after profiles are loaded"
    out = {"platform": {"python": platform.python_version(), "machine": platform.machine(),
                        "cpu": platform.processor() or platform.machine()}}
    out["estimate_4096_tiles"] = _time(lambda: analyze(_req("estimate")), repeat)
    out["compare_8_candidates_4096_tiles"] = _time(lambda: analyze(_req("compare", candidates=EIGHT)), repeat)
    big = _req("compare", n_tiles=32 * 100000, candidates=EIGHT)
    out["compare_8_candidates_3.2M_tiles_extrapolated"] = _time(lambda: analyze(big), max(3, repeat // 4))
    tune = _req("tune", tune={}, analysis={"time_budget_ms": 10000})
    out["tune_full"] = _time(lambda: analyze(tune), max(3, repeat // 4))
    out["compare_cached_repeat"] = _time(lambda: analyze(_req("compare", candidates=EIGHT)), repeat, clear_cache=False)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_req("compare", candidates=EIGHT), f)
        path = f.name
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        subprocess.run([sys.executable, "-m", "qperf", "analyze", "--request", path], cwd=ROOT, check=True,
                       capture_output=True)
        ts.append((time.perf_counter() - t) * 1e3)
    os.unlink(path)
    out["cli_cold_compare_8"] = {"p50_ms": round(statistics.median(ts), 1), "max_ms": round(max(ts), 1)}
    out["max_rss_MiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    out["targets"] = {"compare (8) p95 < 250 ms": out["compare_8_candidates_4096_tiles"]["p95_ms"] < 250,
                      "CLI cold start <= ~1 s": out["cli_cold_compare_8"]["p50_ms"] <= 1000,
                      "full tune ~2 s": out["tune_full"]["p95_ms"] <= 2000,
                      "engine RSS < 64 MiB": out["max_rss_MiB"] < 64}
    return out
