"""Token / cost benchmark of the plugin: runs fixed questions through `claude -p` and reports per run the tokens,
cost, turns, models used, whether qperf actually ran and whether the answer has the expected fact.

  python bench/tokens.py                      # all questions, modes delegated + inline
  python bench/tokens.py --modes delegated --questions dm16 --repeat 3 --out runs/tokens.jsonl

delegated = the skill hands the work to the qperf-runner agent (cheaper model); inline = the Agent tool is disabled,
so the main model does the work itself. Costs real API usage (about $0.05-0.40 per run).
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = {  # question, regex an answer backed by the model must contain
    "dm16": ("Would dm_batch 16 help a 128k-tile bf16 add from DRAM on Quasar?", r"1\.1\d\s*[x×]|11\s*%"),
    "readers": ("For a 64k-tile bf16 add on Quasar from DRAM, would 2 more reader threads or 30% faster compute help?",
                r"writer"),
    "barriers": ("Is it worth implementing per-thread (transaction-ID) NoC barriers for a 4k-tile bf16 add on Quasar, "
                 "with tensors in DRAM vs L1?", r"1\.1[5-7]\s*[x×]|1[5-7]\s*%"),
    "tilize": ("How long does tilize of a 1x1x3136x64 bf16 tensor from DRAM take on Quasar, and would height-sharded "
               "input be faster?", r"4[.,]\d\s*µs|4,?6\d\d|3(\.\d)?\s*[x×]|9[.,]6\s*µs|9,?6\d\d"),  # craq-sim | ttsim-perf
    "untilize_wide": ("On Quasar, which untilize factory runs for a 256x4096 bf16 tensor in DRAM, and what limits it?",
                      r"[Bb]lock"),
    "padding": ("How long does tilize_with_val_padding of a 3000x1000 bf16 tensor from DRAM take on Quasar?",
                r"\d[\d,.]*\s*(ns|µs|cycles)"),
    "shard_cores": ("Is it worth height-sharding a 1024x256 bf16 tensor from DRAM onto 32 cores instead of 8 on Quasar?",
                    r"2[.,]\d\s*[x×]|1[.,]3\d\s*[x×]|faster"),  # craq-sim 2.4x | ttsim-perf 1.35x
    "reshard": ("On Quasar, is resharding a 1024x1024 bf16 tensor from 8 to 16 height shards in L1 faster than to 4? "
                "Which factory runs?", r"SameWidth"),
    "move": ("How long does ttnn move of a 1024x1024 bf16 tile tensor in DRAM take on Quasar?", r"\d[\d,.]*\s*(ns|µs|cycles)"),
    "unsupported": ("How long does a Quasar matmul of 1024x1024 by 1024x1024 take?", r"(not|un)\s*supported|cannot|no model"),
}
TOOLS = ["Bash(PYTHONPATH=*)", "Bash(python3 *)", "Write", "Read", "Skill", "Agent"]


def run(qid, mode, model, budget):
    prompt, expect = QUESTIONS[qid]
    cmd = ["claude", "-p", prompt, "--plugin-dir", ROOT, "--output-format", "stream-json", "--verbose",
           "--max-budget-usd", str(budget), "--allowedTools", *TOOLS]
    if mode == "inline":
        cmd += ["--disallowedTools", "Agent"]
    if model:
        cmd += ["--model", model]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as cwd:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900).stdout
    tools, qperf_runs, agents, res = [], 0, 0, {}
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    tools.append(c["name"])
                    agents += c["name"] in ("Agent", "Task")
                    qperf_runs += "qperf analyze" in json.dumps(c.get("input", {}))
        elif e.get("type") == "result":
            res = e
    usage = res.get("usage", {})
    models = {m: {"in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0),
                  "cache_read": u.get("cacheReadInputTokens", 0), "cache_write": u.get("cacheCreationInputTokens", 0),
                  "cost": round(u.get("costUSD", 0.0), 4)} for m, u in (res.get("modelUsage") or {}).items()}
    tok = sum(v["in"] + v["out"] + v["cache_read"] + v["cache_write"] for v in models.values())
    answer = res.get("result", "") or ""
    return {"question": qid, "mode": mode, "main_model": model or "default", "ok": res.get("subtype") == "success",
            "cost_usd": round(res.get("total_cost_usd", 0.0), 4), "tokens_total": tok,
            "tokens_out": sum(v["out"] for v in models.values()), "turns": res.get("num_turns"),
            "tool_calls": len(tools), "qperf_runs": qperf_runs, "agent_spawns": agents,
            "expected_fact": bool(re.search(expect, answer)), "seconds": round(time.time() - t0, 1), "answer": answer[:3000],
            "models": models, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=",".join(QUESTIONS))
    ap.add_argument("--modes", default="delegated,inline")
    ap.add_argument("--model", default="", help="main session model (default: your Claude Code default)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--budget", type=float, default=1.0, help="max USD per run")
    ap.add_argument("--out", help="append one JSON line per run")
    a = ap.parse_args()
    rows = []
    for qid in a.questions.split(","):
        for mode in a.modes.split(","):
            for _ in range(a.repeat):
                r = run(qid, mode, a.model, a.budget)
                rows.append(r)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(r) + "\n")
    print("| question | mode | cost $ | tokens | out tok | turns | tools | qperf runs | agent | fact | s | models |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ms = ", ".join(f"{m.split('-')[1] if '-' in m else m} ${v['cost']}" for m, v in r["models"].items())
        print(f"| {r['question']} | {r['mode']} | {r['cost_usd']} | {r['tokens_total']:,} | {r['tokens_out']:,} | "
              f"{r['turns']} | {r['tool_calls']} | {r['qperf_runs']} | {r['agent_spawns']} | "
              f"{'yes' if r['expected_fact'] else 'NO'} | {r['seconds']} | {ms} |")
    for mode in a.modes.split(","):
        rs = [r for r in rows if r["mode"] == mode]
        if rs:
            print(f"\n{mode}: mean ${sum(r['cost_usd'] for r in rs) / len(rs):.3f}, "
                  f"{sum(r['tokens_total'] for r in rs) / len(rs):,.0f} tokens, "
                  f"facts {sum(r['expected_fact'] for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
