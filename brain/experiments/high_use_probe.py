# Capability: Sleep consolidation
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md
"""Does use strengthen consolidation? 12 told facts, 4 each told k=1, 3, 10 times; one sleep; hippocampus OFF.

Design choice: a fact told k times is ONE episode with strength k. sleep() samples episodes in proportion to strength,
which is exactly the same replay distribution as k identical copies, and it is the mechanism the brain uses (the
hippocampus adds strength per use). Which facts get which k is rotated by seed (a seeded permutation), so no fact is
always the high-use one: the per-fact in-context difficulty (0.1B ceiling 6/12) is spread over the k groups.
Arms: no_sleep; sleep_use (strength = k); sleep_equal (CONTROL: identical steps, all strengths 1).
Read-outs per arm: recall and paraphrase recall by k, false recall (NEVER_TOLD), drift (KL on ordinary text).

  python -m brain.experiments.high_use_probe --model <path.pth> --steps 240 --seed 42 --out <json>
"""
import argparse
import json
import random
import time

import torch

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts import FACTS
from brain.experiments.sleep_probe import GENERIC, evaluate
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep

KS = (1, 3, 10)


def by_k(items, ks):
    return {str(k): [x for x, kk in zip(items, ks) if kk == k] for k in KS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    keys_all = [k for _, _, k in FACTS]
    order = list(range(len(FACTS)))
    random.Random(a.seed).shuffle(order)
    ks = [0] * len(FACTS)
    for pos, i in enumerate(order):
        ks[i] = KS[pos // 4]
    texts = [f"User: {s}\n\nAssistant: Okay, I'll remember that.\n\n" for s, _, _ in FACTS]
    arms = {"no_sleep": None,
            "sleep_use": [(t, float(k)) for t, k in zip(texts, ks)],
            "sleep_equal": [(t, 1.0) for t in texts]}
    result = {"model": a.model, "seed": a.seed, "steps": a.steps, "kl_weight": a.kl_weight, "lr": a.lr,
              "k_of_fact": {FACTS[i][2]: ks[i] for i in range(len(FACTS))}, "arms": {}}
    for arm, eps in arms.items():
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        if eps is not None:
            sleep(pc, base, eps, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight, seed=a.seed,
                  log=lambda m: print(arm, m, flush=True))
        r = evaluate(pc, base, keys_all)
        r["recall_by_k"] = {k: sum(v) for k, v in by_k(r["recall_items"], ks).items()}
        r["paraphrase_by_k"] = {k: sum(v) for k, v in by_k(r["paraphrase_items"], ks).items()}
        r["adapter_norm"] = pc.adapter_norm()
        result["arms"][arm] = r
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    result["elapsed_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
