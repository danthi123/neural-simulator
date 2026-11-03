# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md
"""Honest live-brain consolidation test: 12 told facts, one restart (= sleep), five arms, strict scorer.

Day 1: the live Brain is told the 12 FACTS in conversation (hear), saved, not closed. The restart is a new Brain on
the same home. Arms, each a fresh Brain per question (so no conversation carry-over), sampled replies seeded per question:
  a_intact_sleep         restart sleeps; hippocampus on, consolidated synapses on
  b_intact_nosleep       home copied before the restart, sleep_on_restart=False; both stores... hippocampus on, no adapters
  c_hippo_off            slept home, hippocampus off
  d_hippo_off_consol_off slept home, hippocampus off, consolidated synapses off (pretrained cortex)
  e_shuffled_hippo_off   day 1 told with answer words swapped between facts (as sleep_probe), slept, hippocampus off;
                         scored against the ORIGINAL keys (a pass here means the test is broken)
Questions per arm: 12 originals, 12 PARA, 4 NEVER_TOLD (false recall = any fact key in the reply).
Scorer: brain/experiments/strict_scorer.py (key as a word, reply not first-person); the lenient substring count is kept.

  python -m brain.experiments.live_brain_probe --home <dir> --seed 42 --out <json>
"""
import argparse
import json
import random
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments.facts import FACTS, PARA
from brain.experiments.sleep_probe import NEVER_TOLD
from brain.experiments.strict_scorer import has_key, score

ARMS = {  # arm -> (home key, hippocampus_off, consolidated_off)
    "a_intact_sleep": ("slept", False, False),
    "b_intact_nosleep": ("nosleep", False, False),
    "c_hippo_off": ("slept", True, False),
    "d_hippo_off_consol_off": ("slept", True, True),
    "e_shuffled_hippo_off": ("shuffled", True, False)}


def shuffled_facts(seed):
    names = [k for _, _, k in FACTS]
    perm = names[:]
    rng = random.Random(seed)
    while any(x == y for x, y in zip(perm, names)):
        rng.shuffle(perm)
    out = []
    for (s, _, k), wrong in zip(FACTS, perm):
        i = s.lower().find(k)
        out.append(s[:i] + wrong.capitalize() + s[i + len(k):])
    return out


def tell_day(home, cortex, sentences, seed, steps):
    shutil.rmtree(home, ignore_errors=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    b = Brain(home, cortex=cortex, sleep_steps=steps, sleep_seed=seed)
    log = [{"you": s, "brain": b.hear(s)[0]} for s in sentences]
    b.save()  # not closed: the restart itself must sleep
    return log


def ask_arm(home, cortex, hip_off, con_off, seed, steps):
    keys = [k for _, _, k in FACTS]
    rows = []
    qs = [(q, k, "told") for _, q, k in FACTS] + [(p, f[2], "para") for p, f in zip(PARA, FACTS)] + \
         [(q, None, "never") for q in NEVER_TOLD]
    for q, key, kind in qs:
        np.random.seed(seed)
        torch.manual_seed(seed)
        b = Brain(home, cortex=cortex, sleep_on_restart=False, sleep_steps=steps, sleep_seed=seed)
        b.set_lesions(hippocampus_off=hip_off, consolidated_off=con_off)
        reply, info = b.hear(q)
        if key is None:
            r = {"false_recall": any(has_key(reply, k) for k in keys),
                 "false_lenient": any(k in reply.lower() for k in keys)}
        else:
            r = score(reply, key, q)
        rows.append({"q": q, "kind": kind, "reply": reply, "recalled": info["recalled"], **r})
    n = lambda kind, f: sum(bool(r[f]) for r in rows if r["kind"] == kind)
    return {"recall": n("told", "hit"), "paraphrase": n("para", "hit"), "false_recall": n("never", "false_recall"),
            "echo": n("told", "echo") + n("para", "echo"), "recall_lenient": n("told", "lenient"),
            "paraphrase_lenient": n("para", "lenient"), "false_lenient": n("never", "false_lenient"), "rows": rows}


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
    homes = {k: f"{a.home}/{k}" for k in ("slept", "nosleep", "shuffled")}
    day1 = tell_day(homes["slept"], c, [s for s, _, _ in FACTS], a.seed, a.steps)
    shutil.rmtree(homes["nosleep"], ignore_errors=True)
    shutil.copytree(homes["slept"], homes["nosleep"])   # same pre-sleep day-1 state, never slept
    day1_shuf = tell_day(homes["shuffled"], c, shuffled_facts(a.seed), a.seed, a.steps)
    res = {"model": a.model, "seed": a.seed, "steps": a.steps, "day1": day1, "day1_shuffled": day1_shuf, "arms": {}}
    for key in ("slept", "shuffled"):  # the restart = a night
        t1 = time.time()
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        b = Brain(homes[key], cortex=c, sleep_steps=a.steps, sleep_seed=a.seed)
        assert b.meta["sleeps"] == 1, b.meta
        res[f"sleep_seconds_{key}"] = round(time.time() - t1, 1)
        res[f"adapter_norm_{key}"] = b.cortex.plastic.adapter_norm()
    for arm, (hk, hip_off, con_off) in ARMS.items():
        r = ask_arm(homes[hk], c, hip_off, con_off, a.seed, a.steps)
        res["arms"][arm] = r
        print(a.seed, arm, {k: v for k, v in r.items() if k != "rows"}, flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
