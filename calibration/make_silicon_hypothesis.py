"""Build profiles/costs/silicon-hypothesis.json from the ttsim-perf profile plus ASSUMED transfer parameters.

Instruction-level costs (DM loops, DFB ops, unpack/math/pack) come from the cycle-approximate simulator; the
simulator does not time DRAM transfers, so every latency/bandwidth below is an assumption with its source and a
wide scenario range. 'low'/'high' are the values used in the fast/slow scenario (for a bandwidth, the fast
scenario has the HIGHER bandwidth). Rerun after recalibrating ttsim-perf.
"""

import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GHZ = 1.35  # assumed AICLK (HAS preliminary: 0.76 / 1.33 / 1.55 GHz NoC; Tensix ~1.4 GHz)


def cyc(ns):
    return round(ns * GHZ, 1)


def main():
    src = json.load(open(os.path.join(ROOT, "profiles", "costs", "ttsim-perf.json")))
    d = copy.deepcopy(src)
    d["name"] = "silicon-hypothesis"
    d["revision"] = f"{src['revision']}+hyp1"
    d["platform"] = "Quasar silicon (HYPOTHESIS): no measurement exists"
    d["platform_scope"] = ("conditional prediction: instruction costs from ttsim-perf at an assumed 1.35 GHz AICLK; "
                           "DRAM/NoC latency and bandwidth are assumptions with wide ranges. Not a measurement.")
    d["clock"] = {"GHz": GHZ, "domain": "AICLK (assumed)", "class": "assumption",
                  "source": "HAS Table 9 preliminary (RES:1601); use sensitivity to test 0.76-1.55 GHz"}
    d["calibration"] = {"derived_from": f"ttsim-perf@{src['revision']}", "script": "calibration/make_silicon_hypothesis.py"}
    d["assumptions"] = [
        "DM/Tensix instruction costs in AICLK cycles equal ttsim-perf cycles (cycle-approximate Rocket/Tensix)",
        "transfer latency/bandwidth are assumptions (no Quasar silicon or DRAM-timed simulator data)",
        "ttsim-perf start-up (DM kernels ~4 us after Neo) kept as is; it may be a simulator artifact",
    ]
    p = d["params"]
    for k in list(p):
        if k.startswith("noc.") or k.startswith("dram."):
            del p[k]
    A = {"class": "assumption"}
    p["noc.read_latency.dram"] = dict(A, value=cyc(350), low=cyc(200), high=cyc(800),
                                      source="~350 ns DRAM read: Aether-study estimate carried in RES:1003-1004, not measured")
    p["noc.write_latency.dram"] = dict(A, value=cyc(300), low=cyc(150), high=cyc(700),
                                       source="write-ack round trip assumed slightly below read latency")
    p["noc.read_latency.l1"] = dict(A, value=cyc(110), low=cyc(60), high=cyc(300),
                                    source="on-chip NoC round trip across an 8x4 mesh: assumption")
    p["noc.write_latency.l1"] = dict(A, value=cyc(110), low=cyc(60), high=cyc(300), source="same")
    p["noc.thread_bytes_per_cycle"] = dict(A, value=62, low=62, high=24,
                                           source="DM issue rate <=62 B/cyc per core, ~24 measured (O2O study, BH-calibrated; RES:1199)")
    p["noc.cluster_rw_shared_bytes_per_cycle"] = dict(A, value=256, low=512, high=128,
                                                      source="NOC_PAYLOAD_WIDTH 2048 b = 256 B/cyc per port (RES:916); sharing of read and write assumed")
    p["dram.total_bytes_per_cycle"] = dict(A, value=round(875e9 / (GHZ * 1e9), 1),
                                           low=round(1500e9 / (GHZ * 1e9), 1), high=round(256e9 / (GHZ * 1e9), 1),
                                           source="QSR1.A1 875 GB/s nominal; fast = QSR3.A2 1.5 TB/s, slow = bring-up 256 GB/s (RES:1594-1600)")
    p["dram.efficiency"] = dict(A, value=0.75, low=0.75, high=0.5, source="eta max 75% (O2O study)")
    out = os.path.join(ROOT, "profiles", "costs", "silicon-hypothesis.json")
    json.dump(d, open(out, "w"), indent=1)
    print(out, d["revision"])


if __name__ == "__main__":
    main()
