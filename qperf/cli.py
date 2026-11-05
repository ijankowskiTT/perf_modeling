"""qperf command line. Every subcommand goes through the same analyze() as the Python API.

  qperf analyze --request case.json [--format json|text|md|html] [--out report.html] [--set analysis.detail=full]
  qperf calibrate report --data calibration/data/ttsim-perf.jsonl --sim perf --costs ttsim-perf [--rank]
  qperf calibrate fit --data ... --sim perf --costs ttsim-perf --keys k1,k2 --revision NEW [--filter layout=ddd]
  qperf bench [--repeat 20]
  qperf profiles
  qperf check-drift --tree /path/to/tt-metal | --ref main | --ref <branch|sha> | --ref pr/63305 [--repo tenstorrent/tt-metal]
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
    out = [f"status: {res['status']}  kind: {res.get('result_kind')}  model computed in {res.get('runtime_ms')} ms (not the op's time)"]
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
        s += (f"  bound {r['lower_bound_ns']:.0f}" if r.get('lower_bound_ns') is not None else "") + f"  bottleneck {r['bottleneck']['resource']}"
        if r.get("comparison"):
            c = r["comparison"]
            s += f"\n      speedup {c['speedup_vs_baseline']['nominal']:.2f}x [{c['speedup_vs_baseline']['low']:.2f}..{c['speedup_vs_baseline']['high']:.2f}]  -> {c['verdict']}"
        if r.get("roles"):
            s += "\n      " + "  ".join(f"{k}: busy {v['useful_busy_frac']:.0%} starved {v['starved_frac']:.0%} "
                                      f"backpressure {v['backpressure_frac']:.0%}" for k, v in r["roles"].items())
        s += "".join(f"\n      warning: {w}" for w in r.get("warnings", []))
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
    if "survey" in res:
        out += _survey_text(res["survey"])
    for r in res.get("recommendations", []):
        out.append(f"  recommend: {r.get('label')} {r.get('speedup_range', '')} {r.get('reason', '')}")
    for w in res.get("warnings", []) if "survey" in res else []:
        out.append(f"  warning: {w}")
    for a in res.get("applied_defaults", []):
        out.append(f"  default: {a}")
    return "\n".join(out)


def _survey_text(sv):
    """Survey: the summary lines, then one row per approach (geomean, min / max with their shape, counts)."""
    st = sv.get("stats") or {}
    out = [f"  survey: {len(sv['heights'])} heights x {len(sv['widths'])} widths, {st.get('evaluated', '?')} shapes "
           f"evaluated, {len(sv['skipped'])} skipped; {st.get('plans_evaluated', '?')} plans timed "
           f"({st.get('plans_reused', '?')} reused) in {st.get('eval_ms', '?')} ms"]
    out += [f"    {line}" for line in sv["summary"]]
    shape = lambda x: "x".join(str(d) for d in x["shape"]) if x else "-"  # noqa: E731
    out.append(f"    {'approach':20s} {'geomean':>8s}  {'min @ shape':18s} {'max @ shape':18s} faster/neutral/slower/invalid")
    for aid, a in sorted(sv["approaches"].items(), key=lambda kv: -(kv[1]["speedup"]["geomean"] or 0)):
        sp, c = a["speedup"], a["counts"]
        if sp["geomean"] is None:
            out.append(f"    {aid:20s} {'-':>8s}  {'-':18s} {'-':18s} {c['faster']}/{c['neutral']}/{c['slower']}/{c['invalid']}")
            continue
        out.append(f"    {aid:20s} {sp['geomean']:7.2f}x  {_at(a['worst'], shape):18s} {_at(a['best'], shape):18s} "
                   f"{c['faster']}/{c['neutral']}/{c['slower']}/{c['invalid']}")
    for s in sv["skipped"]:
        out.append(f"    skipped {shape(s)}: {s['reason']}")
    return out


def _at(x, shape):
    return f"{x['speedup']:.2f}x @ {shape(x)}" if x else "-"


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
    node_of = lambda k: prof["params"].get(k) or prof["compute_variants"][a.variant][k]  # noqa: E731
    for k in keys:
        node = node_of(k)
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
                      "values": {k: node_of(k)["value"] for k in keys}},
                     indent=1))
    return 0


def cmd_bench(a):
    from .bench import run

    print(json.dumps(run(a.repeat), indent=1))
    return 0


def _github(path):
    """GET a GitHub API path: `gh api` when available (uses the user's auth), else anonymous HTTPS."""
    import subprocess
    import urllib.request

    try:
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return json.loads(r.stdout)
        if "Not Found" in r.stdout + r.stderr:
            return None
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        with urllib.request.urlopen("https://api.github.com/" + path, timeout=30) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _remote_blobs(repo, ref, paths):
    """Blob sha of each path at ref (branch, sha, or pr/<n> = the PR's head) in a GitHub repo."""
    label = ref
    if ref.startswith("pr/"):
        pr = _github(f"repos/{repo}/pulls/{ref[3:]}")
        if not pr:
            raise SystemExit(f"cannot read PR {ref[3:]} of {repo}")
        repo, ref = pr["head"]["repo"]["full_name"], pr["head"]["sha"]
        label = f"PR #{pr['number']} head {ref[:11]}"
    commit = _github(f"repos/{repo}/commits/{ref}")
    if not commit:
        raise SystemExit(f"cannot resolve {ref} in {repo} (gh auth login, or check the name)")
    sha = commit["sha"]
    out = {}
    for path in paths:
        meta = _github(f"repos/{repo}/contents/{path}?ref={sha}")
        out[path] = meta.get("sha") if isinstance(meta, dict) else None
    return out, f"{repo}@{label} ({sha[:11]}, {commit['commit']['committer']['date'][:10]})"


def code_tokens(text):
    """C/C++ source without comments, #include lines and formatting, as a token list: two versions with equal tokens
    differ only cosmetically. Conservative: a rename or any code change still differs."""
    import re

    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)  # block comments
    text = re.sub(r"//[^\n]*", " ", text)  # line comments
    text = re.sub(r"^[ \t]*#[ \t]*include[^\n]*", " ", text, flags=re.M)  # include lines
    return re.findall(r"\w+|[^\w\s]", text)


def _blob_text(repo, sha):
    """Content of a git blob (by its sha) from GitHub, or None."""
    import base64

    meta = _github(f"repos/{repo}/git/blobs/{sha}")
    if not isinstance(meta, dict) or "content" not in meta:
        return None
    return base64.b64decode(meta["content"]).decode("utf-8", "replace")


def cmd_check_drift(a):
    """Compare the sources the kernel profile mirrors with a tt-metal tree or a GitHub ref (git blob sha1)."""
    import hashlib

    from . import profiles as P

    kernel, kid = P.load("kernel", a.kernel)
    blobs = kernel.get("source_blobs", {})
    files = blobs.get("files", {})
    if a.ref:
        got_all, where = _remote_blobs(a.repo, a.ref, list(files))
    elif a.tree:
        got_all, where = {}, a.tree
        for path in files:
            f = os.path.join(a.tree, path)
            if os.path.exists(f):
                data = open(f, "rb").read()
                got_all[path] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    else:
        raise SystemExit("give --tree <tt-metal checkout> or --ref <main|branch|sha|pr/N>")
    changed = [p for p, want in files.items() if got_all.get(p) and got_all[p] != want]
    missing = [p for p in files if not got_all.get(p)]
    # Changes that are only comments, includes or formatting do not change the schedule the model mirrors.
    cosmetic = []
    for path in changed:
        old = _blob_text(a.repo, files[path])
        if a.ref:
            new = _blob_text(a.repo, got_all[path])
        else:
            new = open(os.path.join(a.tree, path), encoding="utf-8", errors="replace").read()
        if old is not None and new is not None and code_tokens(old) == code_tokens(new):
            cosmetic.append(path)
    changed = [p for p in changed if p not in cosmetic]
    out = {"kernel_profile": kid, "pinned_commit": blobs.get("commit"), "compared_with": where, "changed": changed,
           "cosmetic_changes": cosmetic, "missing": missing, "status": "ok" if not (changed or missing) else "drift"}
    if cosmetic and not (changed or missing):
        out["note"] = "only comments, includes or formatting changed: the model still matches this code"
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
    d.add_argument("--tree", help="a local tt-metal checkout")
    d.add_argument("--ref", help="GitHub ref instead of a checkout: main, a branch, a sha, or pr/<number>")
    d.add_argument("--repo", default="tenstorrent/tt-metal")
    d.add_argument("--kernel", default="binary-ng-native")
    d.set_defaults(fn=cmd_check_drift)
    pr = sub.add_parser("profiles")
    pr.set_defaults(fn=cmd_profiles)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
