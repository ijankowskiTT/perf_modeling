"""Profile hygiene: provenance on every parameter, consistent scenario ranges, revisions present."""

import glob
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTS = sorted(glob.glob(os.path.join(ROOT, "profiles", "costs", "*.json")))
CLASSES = {"specification", "measurement", "measurement-fit", "derivation", "assumption", "convention"}


def params(d):
    for k, p in d["params"].items():
        yield k, p
    for v, ps in d.get("compute_variants", {}).items():
        for k, p in ps.items():
            if not k.startswith("_"):
                yield f"{v}/{k}", p


@pytest.mark.parametrize("path", COSTS, ids=os.path.basename)
def test_every_cost_has_provenance_and_a_consistent_range(path):
    d = json.load(open(path))
    assert d.get("revision") and d.get("clock", {}).get("GHz")
    for k, p in params(d):
        assert p.get("class") in CLASSES, (k, p)
        assert p.get("source"), k
        if "low" in p and "high" in p:
            lo, hi = p["low"], p["high"]
            if "bytes_per_cycle" in k or k.endswith("efficiency"):
                lo, hi = hi, lo  # capacities: the fast scenario has the larger value
            assert lo <= p["value"] <= hi, (k, p)


def test_kernel_profile_pins_a_commit():
    d = json.load(open(os.path.join(ROOT, "profiles", "kernels", "binary-ng-native.json")))
    assert d["revision"] and d["knob_limits"]["dm_batch"] >= 1


def test_check_drift_detects_a_changed_source(tmp_path):
    import subprocess
    import sys

    d = json.load(open(os.path.join(ROOT, "profiles", "kernels", "binary-ng-native.json")))
    files = d["source_blobs"]["files"]
    repo = os.environ.get("QPERF_TT_METAL", "")
    if not os.path.isdir(repo):
        pytest.skip("set QPERF_TT_METAL to a tt-metal checkout")
    for path in files:
        blob = subprocess.run(["git", "-C", repo, "show", f"{d['source_blobs']['commit']}:{path}"],
                              capture_output=True, check=True).stdout
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(blob)
    run = lambda: subprocess.run([sys.executable, "-m", "qperf", "check-drift", "--tree", str(tmp_path)],  # noqa: E731
                                 cwd=ROOT, capture_output=True, text=True)
    assert run().returncode == 0
    first = tmp_path / next(iter(files))
    first.write_bytes(first.read_bytes() + b"// edit\n")
    out = run()
    assert out.returncode == 3 and json.loads(out.stdout)["changed"] == [next(iter(files))]


def test_cosmetic_changes_are_recognized():
    from qperf.cli import code_tokens as t

    a = '#include "x.h"\nint f(int a) { // add\n  return a+1; /* c */ }\n'
    b = '#include <y.hpp>\n#include "x.h"\n\nint f( int a )\n{\n    return a + 1;\n}\n'
    assert t(a) == t(b)  # includes, comments and formatting only
    assert t(a) != t("int f(int b) { return b+1; }")  # a rename is a real change
    assert t(a) != t("int f(int a) { return a+2; }")
