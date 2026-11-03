# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-sall-live-bar-refit.md (pre-registration this implements)
"""Refit the S_all familiarity bar / margin on cues asked of the LIVE brain (dev facts told in conversation, slept).

  score : one fit seed. Tell 12 shuffled DEV_FACTS36 statements (live_brain_probe.tell_day), restart sleeps once, then a
          fresh Brain(key="S_all") gives the strength-weighted S_all scores of every stored episode for every cue.
          python -m brain.experiments.sall_live_bar_fit score --home <dir> --seed 42 --out <json>
  fit   : pool score JSONs, grid FAMILIARITY x MARGIN, maximise J (pre-registration). python -m ... fit <json>...
  check : top-1 score on a probe's slept TEST home vs the probe's recorded similarity (fidelity), plus the full rows.
          python -m ... check --home <probe home>/slept --probe <seedN.json> --out <json>
"""
import argparse
import json
import random

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments import live_brain_probe as lbp
from brain.experiments.facts_bank import DEV_FACTS36, DEV_NEVER_TOLD

BARS = [round(-0.10 + 0.005 * i, 3) for i in range(181)]
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def cue_rows(home, cortex, cues):
    """cues: [(text, kind, right_episode_text or None)] -> rows with every episode's weighted S_all score."""
    b = Brain(home, cortex=cortex, sleep_on_restart=False, key="S_all")
    assert b.key == "S_all" and b.meta["sleeps"] == 1, b.meta
    eps = [e.text for e in b.hippocampus.episodes]
    rows = []
    for q, kind, right in cues:
        s = b.hippocampus.scores(b.cortex, b.base, q).tolist()  # what recall() competes on; no side effects
        rows.append({"q": q, "kind": kind, "right": right, "scores": [round(x, 5) for x in s]})
    return eps, rows


def score_seed(a):
    c = Cortex(model_path=a.model, device="cpu")
    facts = list(DEV_FACTS36)
    random.Random(a.seed).shuffle(facts)
    told, untold = facts[:12], facts[12:]
    day1 = lbp.tell_day(a.home, c, [f[0] for f in told], a.seed, a.steps)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(a.home, cortex=c, sleep_steps=a.steps, sleep_lr=lbp.SLEEP_LR, sleep_seed=a.seed)  # the night
    assert b.meta["sleeps"] == 1, b.meta
    b.release()
    cues = [(f[1], "orig", f"User: {f[0]}") for f in told] + [(f[2], "para_a", f"User: {f[0]}") for f in told] + \
           [(f[3], "para_b", f"User: {f[0]}") for f in told] + [(f[1], "untold", None) for f in untold] + \
           [(q, "never", None) for q in DEV_NEVER_TOLD]
    eps, rows = cue_rows(a.home, c, cues)
    json.dump({"seed": a.seed, "model": a.model, "steps": a.steps, "day1": day1, "episodes": eps, "rows": rows},
              open(a.out, "w"), indent=1)


def decide(scores, eps, bar, margin):
    """Hippocampus._compete for a question cue: (fired, winner text)."""
    o = np.argsort(-np.asarray(scores), kind="stable")
    top = scores[o[0]]
    run = scores[o[1]] if len(o) > 1 else -1.0
    fired = top >= bar and top - run >= margin
    return fired, eps[o[0]] if fired else None


def rates(runs, bar, margin):
    out = {}
    for kind in ("orig", "para_a", "para_b", "untold", "never"):
        rs = [(r, run["episodes"]) for run in runs for r in run["rows"] if r["kind"] == kind]
        d = [decide(r["scores"], e, bar, margin) for r, e in rs]
        out[kind] = {"fire": sum(f for f, _ in d) / len(d),
                     "sel": sum(f and w == r["right"] for (f, w), (r, _) in zip(d, rs)) / len(d)}
    told = (out["orig"]["sel"] + out["para_a"]["sel"]) / 2  # equal counts per kind: pooled mean
    false = (out["untold"]["fire"] * 24 + out["never"]["fire"] * 12) / 36
    out["J"] = told - false
    return out


def fit(paths):
    runs = [json.load(open(p)) for p in paths]
    assert sorted(r["seed"] for r in runs) == [42, 43, 44], [r["seed"] for r in runs]
    best = None
    for bar in BARS:
        for m in MARGINS:
            J = rates(runs, bar, m)["J"]
            key = (round(J, 9), bar, m)  # ties to the larger bar, then the larger margin
            if best is None or key > best:
                best = key
    J, bar, m = best
    print(json.dumps({"fit": {"familiarity": bar, "margin": m, "J": round(J, 4)},
                      "at_fit": rates(runs, bar, m), "at_offline_bar": rates(runs, 0.185, 0.05),
                      "at_S0_default_bar": rates(runs, 0.12, 0.02),
                      "forced": rates(runs, -1e9, 0.0)}, indent=1))


def check(a):
    c = Cortex(model_path=a.model, device="cpu")
    pr = json.load(open(a.probe))
    rows = pr["arms"]["a_Sall"]["rows"]
    eps, out = cue_rows(a.home, c, [(r["q"], r["kind"], None) for r in rows])
    diffs = [abs(max(o["scores"]) - r["similarity"]) for o, r in zip(out, rows)]
    print("max |top1 - probe similarity|", max(diffs))
    json.dump({"seed": pr["seed"], "bars": pr["bars"], "episodes": eps, "rows": out, "max_diff": max(diffs)},
              open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "fit", "check"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--home")
    ap.add_argument("--probe")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out")
    a = ap.parse_args()
    {"score": score_seed, "fit": lambda a: fit(a.paths), "check": check}[a.mode](a)


if __name__ == "__main__":
    main()
