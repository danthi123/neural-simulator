# Capability: Episodic recall
# Record: brain/research/2026-11-21-replay-priority-prereg.md (REPORT ONLY, written 11-22 after the PR-l leak read, not a bar); stage-1 read
#   brain/experiments/results/2026-11-21-stage1-registered-test-1.5b.md (first-person echo, 150 of 311)
# Expected-fail: none (host-only post-hoc read of a results JSON; echoes are expected to stay common in B)
"""First-person echo read-out from a life results JSON: battery probe rows (res['probes']) whose item has a telling,
scored with the strict scorer's own echo rule (brain/experiments/strict_scorer.is_echo: a first-person reply that
restates the told sentence, e.g. "My cat is called Pixel." to "What is my cat called?"). Per probe night and arm:
told-item rows, rows scored right, echoes (counted only on rows NOT scored right, as the stage-1 read did) and the
key-hit echoes (echo rows that still hold the key). No new scoring rule; only the existing scorer is called.

  python -m brain.eval.life_echo RESULTS.json [--nights 10,15]"""
from __future__ import annotations

import json
import sys

from brain.experiments.strict_scorer import has_key, is_echo

ARMS = ("intact", "hippocampus-off")


def echo_table(res: dict) -> dict:
    """{(night, arm): {"n", "ok", "echo", "echo_key"}} over told-item battery rows."""
    led = res["ledger"]
    out = {}
    for pr in res.get("probes", []):
        arm = pr["arm"]
        if arm not in ARMS:
            continue
        t = out.setdefault((pr["night"], arm), {"n": 0, "ok": 0, "echo": 0, "echo_key": 0})
        for r in pr["rows"]:
            e = led.get(r.get("item") or "")
            if not e or not e.get("telling"):
                continue
            t["n"] += 1
            if r.get("ok"):
                t["ok"] += 1
                continue
            keys = e.get("keys") or []
            if any(is_echo(r["brain"], k, e["telling"]) for k in keys):
                t["echo"] += 1
                if any(has_key(r["brain"], k) for k in keys):
                    t["echo_key"] += 1
    return out


def pooled(tab: dict, nights, arms=ARMS) -> dict:
    s = {"n": 0, "ok": 0, "echo": 0, "echo_key": 0}
    for (n, a), t in tab.items():
        if n in nights and a in arms:
            for k in s:
                s[k] += t[k]
    s["share"] = s["echo"] / s["n"] if s["n"] else None
    return s


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    nights = (10, 15)
    if "--nights" in argv:
        i = argv.index("--nights")
        nights = tuple(int(x) for x in argv[i + 1].split(","))
        del argv[i:i + 2]
    res = json.load(open(argv[0]))
    tab = echo_table(res)
    print(f"seed {res.get('seed')} told-item battery rows; echo counted on rows not scored right")
    print("| arm | night | n | ok | echo | echo with key | echo share |")
    print("|---|---|---|---|---|---|---|")
    for (n, a), t in sorted(tab.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"| {a} | {n} | {t['n']} | {t['ok']} | {t['echo']} | {t['echo_key']} | {t['echo'] / t['n'] if t['n'] else 0:.3f} |")
    for a in ARMS:
        print(f"pooled nights {list(nights)}, {a}: {pooled(tab, nights, (a,))}")


if __name__ == "__main__":
    main()
