"""Merge fitted terms emitted by calibration/fit_layout.py --emit into a cost profile.

  python calibration/merge_params.py calibration/out/reshard-craq-sim.json craq-sim
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, profile = sys.argv[1], sys.argv[2]
path = os.path.join(ROOT, "profiles", "costs", profile + ".json")
prof = json.load(open(path))
new = json.load(open(src))
prof["params"].update(new)
json.dump(prof, open(path, "w"), indent=1)
open(path, "a").write("\n")
print(f"merged {len(new)} terms from {src} into {path}")
