"""Build profiles/costs/silicon-hypothesis.json from the ttsim-perf profile plus ASSUMED transfer parameters.

Instruction-level costs (DM loops, DFB ops, unpack/math/pack) come from the cycle-approximate simulator; the
simulator does not time DRAM transfers, so every latency/bandwidth below is an assumption with its source and a
wide scenario range. 'low'/'high' are the values used in the fast/slow scenario (for a bandwidth, the fast
scenario has the HIGHER bandwidth). Rerun after recalibrating ttsim-perf.

DRAM is timed by the channel model (dram.model = channels, qperf/dram.py): channel count and NoC positions from the
Quasar soc descriptor (TT_METAL_TREE, default /localdev/ijankowski/wt-perf-main), the idle-read latency / window and
the hop latency from ttsim-perf's noc_perf.h (TTSIM_NOC_PERF); both fall back to the recorded values when absent.
"""

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qperf.ops import layout, registry  # noqa: E402

GHZ = 1.35  # assumed AICLK (HAS preliminary: 0.76 / 1.33 / 1.55 GHz NoC; Tensix ~1.4 GHz)
BH_GHZ = 1.35  # Blackhole AICLK the ttsim-perf DRAM constants are in cycles of (p150: 1350 MHz)
TT_METAL = os.environ.get("TT_METAL_TREE", "/localdev/ijankowski/wt-perf-main")
SOC = os.path.join(TT_METAL, "tt_metal", "soc_descriptors", "quasar_32_arch_ttsim.yaml")
NOC_PERF = os.environ.get("TTSIM_NOC_PERF", "/localdev/ijankowski/ttsim-perf/src-repo/src/noc_perf.h")


def cyc(ns):
    return round(ns * GHZ, 1)


# binary_ng instruction terms (the compute terms are in compute_variants)
BINARY_PREFIXES = ("start", "end", "rd", "wr", "compute", "dfb", "dq")


def layout_prefixes():
    """Cost prefixes of every data-movement factory: qperf/ops/layout.py and the registered op modules (move,
    reshard). Their terms are instruction-level ttsim-perf fits, copied as the binary_ng ones."""
    return sorted(set(layout.PREFIX.values()) | {p for m in registry.MODULES for p in m.PREFIX.values()})


def soc_dram():
    """(DRAM channel NoC positions, banks, worker NoC positions) from the Quasar soc descriptor: one bank per DRAM view
    (tt-metal l1_banking_allocator: num_dram_channels = get_num_dram_views(); interleaved page p -> bank p mod banks,
    get_bank_index in dataflow_api_addrgen.h)."""
    if os.path.exists(SOC):
        import yaml

        d = yaml.safe_load(open(SOC))
        xy = lambda s: [int(v) for v in str(s).split("-")]  # noqa: E731  ("2-7" -> [2, 7])
        chans = [xy(c[0] if isinstance(c, list) else c) for c in d["dram"]]
        return chans, len(d["dram_views"]), [xy(w) for w in d["functional_workers"]], os.path.basename(SOC)
    print(f"warning: {SOC} not found; using the recorded Quasar DRAM layout")
    return [[2, 7], [4, 0]], 2, [[x, y] for y in range(2, 6) for x in range(2, 10)], "quasar_32_arch_ttsim.yaml (recorded)"


def ttsim_constants():
    """ttsim-perf's DRAM and hop constants (src/noc_perf.h): Blackhole's NOC_DRAM_* (TT_ARCH_VERSION == 1) and QSR's
    NOC_HOP (the second definition)."""
    rec = {"NOC_DRAM_LATENCY": 142, "NOC_DRAM_RATE_X16": 759, "NOC_DRAM_IDLE": 8, "NOC_HOP": [9, 7]}
    if not os.path.exists(NOC_PERF):
        print(f"warning: {NOC_PERF} not found; using the recorded ttsim-perf constants")
        return rec
    import re

    txt = open(NOC_PERF).read()
    out = {k: int(re.search(rf"#define {k} (\d+)", txt).group(1)) for k in ("NOC_DRAM_LATENCY", "NOC_DRAM_RATE_X16",
                                                                           "NOC_DRAM_IDLE")}
    out["NOC_HOP"] = [int(v) for v in re.findall(r"#define NOC_HOP (\d+)", txt)]  # [BH, QSR]
    return out


def channel_model(p):
    """dram.model = channels (qperf/dram.py): ttsim-perf's Blackhole DRAM model structure with Quasar's channels and
    assumed rates. The per-channel rate splits the assumed total (dram.total_bytes_per_cycle x dram.efficiency, the
    old aggregate bound) over the channels in each scenario; the idle-read latency and window are ttsim-perf's
    Blackhole constants at the clock ratio; the fixed request latency is what is left of the assumed DRAM read / write
    latency (noc.{read,write}_latency.dram) for a core at the mean distance once the hops and (reads) the idle latency
    are counted, so an isolated read from an average core takes the old assumption."""
    chans, banks, workers, soc = soc_dram()
    k = ttsim_constants()
    A = {"class": "assumption"}
    S = lambda key, sc: p[key][sc] if sc in p[key] else p[key]["value"]  # noqa: E731
    p["dram.model"] = dict(A, value="channels",
                           source="qperf/dram.py: per-channel streaming rate, idle-read latency, page interleaving and "
                                  "hop latency, structure from ttsim-perf's Blackhole DRAM model (noc_dram_access); "
                                  "no simulator times Quasar DRAM, not calibrated on Quasar")
    p["dram.channels"] = {"value": banks, "class": "specification",
                          "source": f"{soc}: {len(chans)} DRAM channels, {banks} dram_views = banks (one NoC endpoint each)"}
    p["dram.channel_noc"] = {"value": chans, "class": "specification", "source": f"{soc}: dram NoC coordinates"}
    rate = {sc: round(S("dram.total_bytes_per_cycle", sc) * S("dram.efficiency", sc) / banks, 2)
            for sc in ("value", "low", "high")}
    p["dram.channel_bytes_per_cycle"] = dict(A, **rate, source=(
        f"total assumed DRAM bandwidth x efficiency / {banks} channels: 875 GB/s x 0.75 nominal, fast 1.5 TB/s x 0.75, "
        f"slow 256 GB/s x 0.5, at {GHZ} GHz (ttsim-perf Blackhole: {k['NOC_DRAM_RATE_X16']}/16 = "
        f"{k['NOC_DRAM_RATE_X16'] / 16:.1f} B/cycle per channel)"))
    scale = GHZ / BH_GHZ
    lo, hi = 200 / 350, 800 / 350  # the read latency assumption's range (noc.read_latency.dram 200 / 350 / 800 ns)
    idle = k["NOC_DRAM_LATENCY"] * scale
    p["dram.idle_latency"] = dict(A, value=round(idle, 1), low=round(idle * lo, 1), high=round(idle * hi, 1), source=(
        f"ttsim-perf Blackhole NOC_DRAM_LATENCY {k['NOC_DRAM_LATENCY']} cycles (fitted to BH silicon) x {GHZ}/{BH_GHZ} "
        f"GHz (BH AICLK); fast / slow x {lo:.2f} / {hi:.2f} as the read latency range"))
    p["dram.idle_cycles"] = dict(A, value=round(k["NOC_DRAM_IDLE"] * scale, 1), source=(
        f"ttsim-perf Blackhole NOC_DRAM_IDLE {k['NOC_DRAM_IDLE']} cycles x {GHZ}/{BH_GHZ} GHz: a read that finds its "
        f"channel idle longer waits dram.idle_latency more"))
    hop = k["NOC_HOP"][-1]
    p["noc.hop_latency"] = {"value": hop, "class": "assumption",
                            "source": f"ttsim-perf noc_perf.h QSR NOC_HOP {hop} (designers' NoC mesh bench): a router "
                                      f"to the next, NoC = AI clock"}
    mean_hops = sum(abs(wx - cx) + abs(wy - cy) for cx, cy in chans for wx, wy in workers) / (len(chans) * len(workers))
    trip = 2 * mean_hops * hop
    rd = {sc: round(S("noc.read_latency.dram", sc) - S("dram.idle_latency", sc) - trip, 1)
          for sc in ("value", "low", "high")}
    wr = {sc: round(S("noc.write_latency.dram", sc) - trip, 1) for sc in ("value", "low", "high")}
    p["dram.read_fixed_latency"] = dict(A, **rd, source=(
        f"noc.read_latency.dram - dram.idle_latency - 2 x {mean_hops:g} mean hops (32 workers to the {len(chans)} "
        f"channels) x noc.hop_latency: half before the channel, half after"))
    p["dram.write_fixed_latency"] = dict(A, **wr, source=(
        f"noc.write_latency.dram - 2 x {mean_hops:g} mean hops x noc.hop_latency (writes pay no idle latency, as "
        f"ttsim-perf's v1 write path)"))


def main():
    src = json.load(open(os.path.join(ROOT, "profiles", "costs", "ttsim-perf.json")))
    d = copy.deepcopy(src)
    d["name"] = "silicon-hypothesis"
    d["revision"] = f"{src['revision']}+hyp2"  # hyp2: DRAM channel model (hyp1: aggregate DRAM bound)
    d["platform"] = "Quasar silicon (HYPOTHESIS): no measurement exists"
    d["platform_scope"] = ("conditional prediction: instruction costs from ttsim-perf at an assumed 1.35 GHz AICLK; "
                           "DRAM/NoC latency and bandwidth are assumptions with wide ranges; DRAM timed by a channel "
                           "model (2 channels). Not a measurement.")
    d["clock"] = {"GHz": GHZ, "domain": "AICLK (assumed)", "class": "assumption",
                  "source": "HAS Table 9 preliminary (RES:1601); use sensitivity to test 0.76-1.55 GHz"}
    d["calibration"] = {"derived_from": f"ttsim-perf@{src['revision']}", "script": "calibration/make_silicon_hypothesis.py"}
    d["assumptions"] = [
        "DM/Tensix instruction costs in AICLK cycles equal ttsim-perf cycles (cycle-approximate Rocket/Tensix)",
        "transfer latency/bandwidth are assumptions (no Quasar silicon or DRAM-timed simulator data)",
        "ttsim-perf start-up (DM kernels ~4 us after Neo) kept as is; it may be a simulator artifact",
        "DRAM: channel model (qperf/dram.py): the soc descriptor's 2 channels, interleaved pages round-robin over them, "
        "an assumed per-channel rate and ttsim-perf's Blackhole idle-read latency; not calibrated on Quasar",
    ]
    # ttsim-perf's binary_ng and data-movement (layout family) instruction terms; its measured NoC terms are replaced
    # by the assumptions below. Any other term is an error: decide whether it belongs here first.
    fam = tuple(layout_prefixes())
    p = d["params"] = {}
    for k, v in src["params"].items():
        pre = k.split(".", 1)[0]
        if pre in ("noc", "dram"):
            continue
        if pre not in BINARY_PREFIXES + fam:
            raise SystemExit(f"ttsim-perf term {k}: neither a binary_ng nor a layout-family term")
        p[k] = copy.deepcopy(v)
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
    channel_model(p)
    out = os.path.join(ROOT, "profiles", "costs", "silicon-hypothesis.json")
    json.dump(d, open(out, "w"), indent=1)
    print(out, d["revision"])


if __name__ == "__main__":
    main()
