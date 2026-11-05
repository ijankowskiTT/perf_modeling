"""Versioned profiles: load by 'name@revision', hash the content, resolve cost scenarios with provenance."""

from __future__ import annotations

import hashlib
import json
import os

ROOT = os.environ.get("QPERF_PROFILES", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles"))
_DIRS = {"architecture": "architecture", "kernel": "kernels", "costs": "costs"}
_cache = {}
_memory = {}  # name -> profile dict registered in memory (calibration search); never written to disk


def register(kind, data):
    _memory[(kind, data["name"])] = (data, sha256(data))


def unregister(kind, name):
    _memory.pop((kind, name), None)


class ProfileError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def available(kind):
    d = os.path.join(ROOT, _DIRS[kind])
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def load(kind, ref):
    """ref = 'name' or 'name@revision'. A revision that does not match the file is an error, not a fallback."""
    if not ref:
        raise ProfileError("needs_input", f"no {kind} profile given; available: {available(kind)}")
    name, _, rev = ref.partition("@")
    if (kind, name) in _memory:
        data, _ = _memory[(kind, name)]
        return data, {"name": name, "revision": data.get("revision"), "sha256": sha256(data)}
    path = os.path.join(ROOT, _DIRS[kind], name + ".json")
    if not os.path.exists(path):
        raise ProfileError("unsupported", f"{kind} profile '{name}' not found; available: {available(kind)}")
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _cache:
        with open(path) as f:
            data = json.load(f)
        _cache[key] = (data, sha256(data))
    data, digest = _cache[key]
    if rev and rev != data.get("revision"):
        raise ProfileError("unsupported",
                           f"{kind} profile '{name}' is at revision {data.get('revision')}, request pins {rev}")
    return data, {"name": name, "revision": data.get("revision"), "sha256": digest}


SCENARIOS = ("nominal", "low", "high")


def _pick(p, scenario):
    if not isinstance(p, dict):
        return float(p)
    if scenario == "low" and "low" in p:
        return float(p["low"])
    if scenario == "high" and "high" in p:
        return float(p["high"])
    return float(p["value"])


def resolve_costs(profile, variant, overrides=None, scenario="nominal", compute_scale=1.0):
    """Flatten a cost profile into {key: cycles} for one compute variant and scenario.

    Missing variant -> ProfileError(needs_input). Overrides replace a key in every scenario (a what-if is a
    point value unless it gives low/high). compute_scale multiplies the cu/cm/cp keys (what-if only).
    """
    out = {k: _pick(p, scenario) for k, p in profile["params"].items()}
    variants = profile.get("compute_variants", {})
    if variant not in variants:
        if not (overrides and all(k in overrides for k in ("cu.fixed", "cu.per_tile", "cm.fixed", "cm.per_tile",
                                                               "cp.fixed", "cp.per_tile"))):
            raise ProfileError("needs_input",
                               f"cost profile '{profile['name']}' has no compute costs for '{variant}'; known: "
                               f"{sorted(variants)}. Supply cu/cm/cp .fixed/.per_tile in what_if.cost_overrides "
                               f"or calibrate this variant.")
    else:
        out.update({k: _pick(p, scenario) for k, p in variants[variant].items() if not k.startswith("_")})
    for k, v in (overrides or {}).items():
        out[k] = _pick(v, scenario)
    if compute_scale != 1.0:
        for k in list(out):
            if k.split(".")[0] in ("cu", "cm", "cp"):
                out[k] *= compute_scale
    return out


def param_provenance(profile, variant, keys):
    rows = {}
    allp = dict(profile["params"])
    allp.update({k: v for k, v in profile.get("compute_variants", {}).get(variant, {}).items() if not k.startswith("_")})
    for k in keys:
        p = allp.get(k)
        if isinstance(p, dict):
            rows[k] = {kk: p[kk] for kk in ("value", "low", "high", "class", "source") if kk in p}
    return rows
