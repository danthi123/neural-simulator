#!/usr/bin/env python3
"""Aggregate a bake-off batch into per-model, per-role quality and cost (docs: tools/model_bakeoff/README.md).

Quality: command/exact tasks score 1 or 0; rubric tasks score the judges' mean (0-1). A disqualified run (audit
flag) scores 0 and is counted separately. Cost per run is estimated from Hermes's token counts and OpenRouter's
public list prices (uncached input, cache reads, cache writes, output); the batch's real OpenRouter spend
(key_usage.json, from the key's own usage counter) is printed beside the estimate as a check. The local model
costs $0 here (electricity and the GPU time it takes from research runs are not priced).

  python3 report.py --batch screen1 [--out report.md]
"""
import argparse
import glob
import json
import os
import urllib.request

from run import RUNS, load_json


def prices():
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
        data = json.load(r)["data"]
    out = {}
    for m in data:
        p = m.get("pricing") or {}
        f = lambda k: float(p.get(k) or 0)
        out[m["id"]] = {"in": f("prompt"), "out": f("completion"),
                        "cache_read": f("input_cache_read") or f("prompt"), "cache_write": f("input_cache_write") or f("prompt")}
    return out


def run_cost(res, pr):
    t = res.get("tokens") or {}
    p = pr.get(res["model"])
    if not p or res.get("tier") == "local":
        return 0.0
    return (t.get("input", 0) * p["in"] + t.get("cache_read", 0) * p["cache_read"]
            + t.get("cache_write", 0) * p["cache_write"] + t.get("output", 0) * p["out"])


def score(res):
    if res.get("disqualified"):
        return 0.0
    g = res.get("grade") or {}
    if "pass" in g:
        return 1.0 if g["pass"] else 0.0
    return g.get("score")          # None while a rubric run is unjudged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = [load_json(p) for p in sorted(glob.glob(os.path.join(RUNS, a.batch, "*", "*", "result.json")))]
    pr = prices()
    by_model, roles = {}, set()
    for r in results:
        r["_cost"] = run_cost(r, pr)
        r["_score"] = score(r)
        by_model.setdefault(r["model"], []).append(r)
        roles.add(r.get("role") or "?")
    roles = sorted(roles)
    lines = ["# Model bake-off: batch `%s`" % a.batch, "",
             "Scores are 0-1 (pass/fail tasks 1 or 0; rubric tasks the two judges' mean). DQ = runs disqualified by the "
             "network/outside-path audit (scored 0). Cost is estimated from tokens at OpenRouter list prices.", "",
             "| Model | Tier | " + " | ".join(roles) + " | All | Runs | DQ | Unjudged | Mean min/run | Est. $/run | Est. $ total |",
             "|---|---|" + "---|" * len(roles) + "---|---|---|---|---|---|---|"]
    summary = []
    for m, rs in sorted(by_model.items(), key=lambda kv: -sum((x["_score"] or 0) for x in kv[1]) / max(1, len(kv[1]))):
        cells = []
        for role in roles:
            sc = [x["_score"] for x in rs if (x.get("role") or "?") == role and x["_score"] is not None]
            cells.append("%.2f (%d)" % (sum(sc) / len(sc), len(sc)) if sc else "–")
        allsc = [x["_score"] for x in rs if x["_score"] is not None]
        cost = sum(x["_cost"] for x in rs)
        mins = sum((x.get("wall_s") or 0) for x in rs) / max(1, len(rs)) / 60
        row = {"model": m, "tier": rs[0].get("tier"), "score": round(sum(allsc) / len(allsc), 3) if allsc else None,
               "runs": len(rs), "dq": sum(1 for x in rs if x.get("disqualified")),
               "unjudged": sum(1 for x in rs if x["_score"] is None), "mean_min": round(mins, 1),
               "cost_per_run": round(cost / max(1, len(rs)), 3), "cost_total": round(cost, 2),
               "by_role": dict(zip(roles, cells))}
        summary.append(row)
        lines.append("| %s | %s | %s | %s | %d | %d | %d | %.1f | %.3f | %.2f |" % (
            m, row["tier"], " | ".join(cells), "%.2f" % row["score"] if row["score"] is not None else "–",
            row["runs"], row["dq"], row["unjudged"], mins, row["cost_per_run"], cost))
    ku = os.path.join(RUNS, a.batch, "key_usage.json")
    if os.path.exists(ku):
        k = load_json(ku)
        lines += ["", "OpenRouter's own spend counter for this batch: $%s (estimate above: $%.2f)." % (
            "%.2f" % k["spent"] if k.get("spent") is not None else "unknown", sum(x["_cost"] for x in results))]
    adj = [r for r in results if (r.get("grade") or {}).get("needs_adjudication")]
    if adj:
        lines += ["", "Needs adjudication (judges disagree by > 0.25, or a judge failed): " +
                  ", ".join("%s/%s" % (r["model"], r["task"]) for r in adj)]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)
        json.dump(summary, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
