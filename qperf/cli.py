"""qperf command line. Every subcommand goes through the same analyze() as the Python API.

  qperf analyze --request case.json [--format json|text|md|html] [--out report.html] [--set analysis.detail=full]
  qperf calibrate report --data calibration/data/ttsim-perf.jsonl --sim perf --costs ttsim-perf [--rank]
  qperf calibrate fit --data ... --sim perf --costs ttsim-perf --keys k1,k2 --revision NEW [--filter layout=ddd]
  qperf bench [--repeat 20]
  qperf profiles
  qperf check-drift --tree /path/to/tt-metal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _set(req, assignment):
    path, _, val = assignment.partition("=")
    try:
        v = json.loads(val)
    except json.JSONDecodeError:
        v = val
    node = req
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = v


def _text(res):
    out = [f"status: {res['status']}  kind: {res.get('result_kind')}  runtime: {res.get('runtime_ms')} ms"]
    if res.get("message"):
        out.append(f"message: {res['message']}")
    if "profiles" in res:
        out.append("profiles: " + ", ".join(f"{k}={v['name']}@{v['revision']}" for k, v in res["profiles"].items()))
    if res.get("platform_scope"):
        out.append(f"platform: {res['platform_scope']}")

    def row(r, tag):
        if not r.get("valid"):
            return f"  {tag:14s} {r.get('label', '')}  INVALID: {'; '.join(r.get('errors', []))}"
        t = r["time_ns"]
        s = f"  {tag:14s} {r['label']}\n      time {t['nominal']:.0f} ns  [{t['low']:.0f} .. {t['high']:.0f}]"
        s += f"  bound {r['lower_bound_ns']:.0f}  bottleneck {r['bottleneck']['resource']}"
        if r.get("comparison"):
            c = r["comparison"]
            s += f"\n      speedup {c['speedup_vs_baseline']['nominal']:.2f}x [{c['speedup_vs_baseline']['low']:.2f}..{c['speedup_vs_baseline']['high']:.2f}]  -> {c['verdict']}"
        if r.get("roles"):
            s += "\n      " + "  ".join(f"{k}: busy {v['useful_busy_frac']:.0%} starved {v['starved_frac']:.0%} "
                                      f"backpressure {v['backpressure_frac']:.0%}" for k, v in r["roles"].items())
        return s

    if "baseline" in res:
        out.append(row(res["baseline"], "baseline"))
    for c in res.get("candidates", []):
        out.append(row(c, c.get("id", "candidate")))
    for i, c in enumerate(res.get("top", [])):
        out.append(row(c, f"top{i + 1}"))
    if "search" in res:
        s = res["search"]
        out.append(f"  search: {s['enumerated']} enumerated, {s['invalid']} invalid, {s['evaluated']} of {s['valid']} valid evaluated, "
                   f"exhaustive={s['exhaustive']}")
    if "sensitivity" in res:
        sens = res["sensitivity"]
        out.append(f"  sensitivity of {sens['param']}:")
        for cid, pts in sens["series"].items():
            out.append(f"    {cid}: " + ", ".join(f"{p['value']}->{p['time_ns'] and round(p['time_ns'])} ({p['bottleneck']})" for p in pts))
        for x in sens["ranking_changes"]:
            out.append(f"    ranking change {x['between']} at ~{x['at_about']}")
    for r in res.get("recommendations", []):
        out.append(f"  recommend: {r.get('label')} {r.get('speedup_range', '')} {r.get('reason', '')}")
    for a in res.get("applied_defaults", []):
        out.append(f"  default: {a}")
    return "\n".join(out)


def cmd_analyze(a):
    from .api import analyze

    req = json.load(open(a.request)) if a.request != "-" else json.load(sys.stdin)
    for s in a.set or []:
        _set(req, s)
    res = analyze(req)
    if a.format in ("md", "html"):
        from .report import to_html, to_markdown

        if a.format == "html":
            res = analyze(dict(req, analysis=dict(req.get("analysis", {}), detail="full")))
        out = (to_html if a.format == "html" else to_markdown)(req, res)
    else:
        out = json.dumps(res, indent=1) if a.format == "json" else _text(res)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
        print(a.out)
    else:
        print(out)
    return 0 if res["status"] in ("ok", "partial") else 2


def _filtered(recs, filters):
    for f in filters or []:
        k, _, v = f.partition("=")
        recs = [r for r in recs if str(r.get(k)) == v]
    return recs


def cmd_report(a):
    from .calibration import load_records, predict, ranking_quality, summarize

    recs = _filtered(load_records(a.data, a.sim), a.filter)
    rows = predict(recs, a.costs)
    out = {"summary": summarize(rows), "by_layout": {}}
    for lay in sorted({r["layout"] for r in recs}):
        out["by_layout"][lay] = summarize([x for x, r in zip(rows, recs) if r["layout"] == lay])
    if a.rank:
        out["ranking"] = ranking_quality(rows, recs)
    if a.rows:
        out["rows"] = rows
    print(json.dumps(out, indent=1))
    return 0


def cmd_fit(a):
    from . import profiles as P
    from .calibration import fit, load_records, predict, summarize

    recs = _filtered(load_records(a.data, a.sim), a.filter)
    base, _ = P.load("costs", a.costs)
    keys = a.keys.split(",")
    prof, hist = fit(recs, base, keys, variant=a.variant, sweeps=a.sweeps)
    prof["revision"] = a.revision
    prof["calibration"] = dict(prof.get("calibration", {}), fitted_keys=keys, fit_dataset=a.data,
                               fit_filter=a.filter, fit_sim=a.sim, fit_objective="mean squared log(pred/meas)",
                               fit_history=[round(h, 6) for h in hist], fitted_at=time.strftime("%Y-%m-%d"))
    for k in keys:
        node = prof["params"].get(k) or prof["compute_variants"][a.variant][k]
        node["class"] = "measurement-fit"
        node["source"] = f"fitted {time.strftime('%Y-%m-%d')} on {os.path.basename(a.data)} ({a.sim}, {len(recs)} runs)"
        if k.endswith("_frac"):
            node["low"], node["high"] = max(0.0, node["value"] - 0.2), node["value"]
        else:
            node["low"], node["high"] = round(node["value"] * (1 - a.range), 3), round(node["value"] * (1 + a.range), 3)
        node["range_note"] = f"+-{a.range:.0%} scenario range for a fitted value (set --range from holdout error)"
    out = a.out or os.path.join(P.ROOT, "costs", base["name"] + ".json")
    with open(out, "w") as f:
        json.dump(prof, f, indent=1)
    P._cache.clear()
    rows = predict(recs, prof["name"])
    print(json.dumps({"written": out, "revision": a.revision, "fit": summarize(rows),
                      "values": {k: (prof["params"].get(k) or prof["compute_variants"][a.variant][k])["value"] for k in keys}},
                     indent=1))
    return 0


def cmd_bench(a):
    from .api import analyze
    from .bench import run

    print(json.dumps(run(a.repeat), indent=1))
    return 0


def cmd_check_drift(a):
    """Compare the sources the kernel profile mirrors with a tt-metal tree (git blob sha1, no git needed)."""
    import hashlib

    from . import profiles as P

    kernel, kid = P.load("kernel", a.kernel)
    blobs = kernel.get("source_blobs", {})
    changed, missing = [], []
    for path, want in blobs.get("files", {}).items():
        f = os.path.join(a.tree, path)
        if not os.path.exists(f):
            missing.append(path)
            continue
        data = open(f, "rb").read()
        got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if got != want:
            changed.append(path)
    out = {"kernel_profile": kid, "pinned_commit": blobs.get("commit"), "tree": a.tree, "changed": changed,
           "missing": missing, "status": "ok" if not (changed or missing) else "drift"}
    if changed or missing:
        out["advice"] = ("the mirrored code moved: re-read the changed files, update qperf/rule.py, constraints.py or "
                         "pipeline.py if the schedule or rule changed, then recalibrate (docs/CALIBRATION.md)")
    print(json.dumps(out, indent=1))
    return 0 if out["status"] == "ok" else 3


def cmd_profiles(a):
    from . import profiles as P

    print(json.dumps({k: P.available(k) for k in ("architecture", "kernel", "costs")}, indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="qperf", description="Quasar binary_ng eltwise performance model")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze")
    s.add_argument("--request", required=True)
    s.add_argument("--format", choices=["json", "text", "md", "html"], default="json")
    s.add_argument("--out", help="write the output to this file (md/html reports)")
    s.add_argument("--set", action="append", help="override a request field, e.g. analysis.detail=full")
    s.set_defaults(fn=cmd_analyze)
    c = sub.add_parser("calibrate")
    csub = c.add_subparsers(dest="ccmd", required=True)
    r = csub.add_parser("report")
    f = csub.add_parser("fit")
    for x in (r, f):
        x.add_argument("--data", required=True)
        x.add_argument("--sim", required=True)
        x.add_argument("--costs", required=True)
        x.add_argument("--filter", action="append")
    r.add_argument("--rank", action="store_true")
    r.add_argument("--rows", action="store_true")
    r.set_defaults(fn=cmd_report)
    f.add_argument("--keys", required=True)
    f.add_argument("--variant", default="fpu:add:bf16")
    f.add_argument("--revision", required=True)
    f.add_argument("--sweeps", type=int, default=3)
    f.add_argument("--out")
    f.add_argument("--range", type=float, default=0.05, help="relative scenario range given to fitted values")
    f.set_defaults(fn=cmd_fit)
    b = sub.add_parser("bench")
    b.add_argument("--repeat", type=int, default=20)
    b.set_defaults(fn=cmd_bench)
    d = sub.add_parser("check-drift", help="are the sources the model mirrors unchanged in a tt-metal tree?")
    d.add_argument("--tree", required=True)
    d.add_argument("--kernel", default="binary-ng-native")
    d.set_defaults(fn=cmd_check_drift)
    pr = sub.add_parser("profiles")
    pr.set_defaults(fn=cmd_profiles)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
