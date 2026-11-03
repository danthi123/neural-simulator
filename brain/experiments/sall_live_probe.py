# Capability: Memory selection (pattern completion and separation)
# Record: brain/experiments/results/2026-11-03-meaning-selection-realpath.md (S_all lead; live-brain test asked for)
"""Live-brain test of the S_all meaning key: the same lived day and the same night, recall with key S0 vs S_all.

Day 1: the live Brain (default key) is told the 12 FACTS in conversation (live_brain_probe.tell_day), saved, not
closed; the home is copied before the restart (no-sleep control); the restart sleeps once. Storage does not depend on
the key (an episode is its matrix-memory delta), so both keys answer from byte-identical homes: the only difference
between paired arms is the pattern-completion key (and its bar) the brain opens the home with, Brain(key=...).
Arms (fresh Brain per question, sampled reply seeded per question, same seed across arms):
  a_S0 / a_Sall       slept home, intact, key S0 / S_all (primary)
  b_S0 / b_Sall       never-slept copy, intact (hippocampus alone), key S0 / S_all (secondary)
  c_hippo_off         slept home, hippocampus off (lesion; key-independent)
Questions: 12 originals, 12 PARA (reworded), 16 never-told (sleep_probe.NEVER_TOLD 4 + facts_bank.NEVER_TOLD 12).
Scorer: strict_scorer.score (primary, as the 12-fact probe); correct_answer reported. Selection read-outs: the
recalled episode is the asked fact's telling; never-told fire = any episode recalled on a never-told question.

  python -m brain.experiments.sall_live_probe --home <dir> --seed 42 --out <json>
"""
import argparse
import json
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments import live_brain_probe as lbp
from brain.experiments.facts import FACTS, PARA
from brain.experiments.facts_bank import NEVER_TOLD as BANK_NEVER
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.experiments.strict_scorer import correct_answer, has_key, score

NEVER = NEVER_TOLD + BANK_NEVER
ARMS = {"a_S0": ("slept", "S0", False), "a_Sall": ("slept", "S_all", False),
        "b_S0": ("nosleep", "S0", False), "b_Sall": ("nosleep", "S_all", False),
        "c_hippo_off": ("slept", "S0", True)}


def ask_arm(home, cortex, key, hip_off, seed, steps):
    keys = [k for _, _, k in FACTS]
    qs = [(q, s, k, "told") for s, q, k in FACTS] + [(p, f[0], f[2], "para") for p, f in zip(PARA, FACTS)] + \
         [(q, None, None, "never") for q in NEVER]
    rows = []
    for q, told, k, kind in qs:
        np.random.seed(seed)
        torch.manual_seed(seed)
        b = Brain(home, cortex=cortex, sleep_on_restart=False, sleep_steps=steps, sleep_lr=lbp.SLEEP_LR,
                  sleep_seed=seed, key=key)
        assert b.key == key
        b.set_lesions(hippocampus_off=hip_off)
        reply, info = b.hear(q)
        r = {"q": q, "kind": kind, "reply": reply, "recalled": info["recalled"], "similarity": info["similarity"]}
        if k is None:
            r.update(false_recall=any(has_key(reply, x) for x in keys), fired=info["recalled"] is not None)
        else:
            r.update(score(reply, k, q), correct=correct_answer(reply, k, q, told),
                     selected=info["recalled"] == f"User: {told}", fired=info["recalled"] is not None)
        rows.append(r)
    n = lambda kind, f: sum(bool(r[f]) for r in rows if r["kind"] == kind)
    return {"recall": n("told", "hit"), "paraphrase": n("para", "hit"), "false_recall": n("never", "false_recall"),
            "recall_correct": n("told", "correct"), "paraphrase_correct": n("para", "correct"),
            "sel_told": n("told", "selected"), "sel_para": n("para", "selected"), "never_fired": n("never", "fired"),
            "echo": n("told", "echo") + n("para", "echo"), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = Cortex(model_path=a.model, device="cpu")
    homes = {k: f"{a.home}/{k}" for k in ("slept", "nosleep")}
    day1 = lbp.tell_day(homes["slept"], c, [s for s, _, _ in FACTS], a.seed, a.steps)
    shutil.rmtree(homes["nosleep"], ignore_errors=True)
    shutil.copytree(homes["slept"], homes["nosleep"])
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = Brain(homes["slept"], cortex=c, sleep_steps=a.steps, sleep_lr=lbp.SLEEP_LR, sleep_seed=a.seed)  # the night
    assert b.meta["sleeps"] == 1, b.meta
    res = {"model": a.model, "seed": a.seed, "steps": a.steps, "day1": day1, "sleep_seconds": round(time.time() - t0, 1),
           "bars": dict(Brain.KEY_BARS), "arms": {}}
    for arm, (hk, key, hip_off) in ARMS.items():
        r = ask_arm(homes[hk], c, key, hip_off, a.seed, a.steps)
        res["arms"][arm] = r
        print(a.seed, arm, {k: v for k, v in r.items() if k != "rows"}, flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
