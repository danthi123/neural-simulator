# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-03-high-use-facts/README.md (next step 1)
"""Does use, counted INSIDE the hippocampus, make a reachable fact reliably consolidated?

12 facts the cortex can hold in context (rule: the first 12, in pool order, whose ORIGINAL question is answered
in-context at this model size; from ceiling_probe json). 4 facts told k=1, 4 told k=3, 4 told k=10 times, each telling
passing through UseHippocampus.store (re-telling raises the one episode's strength; nothing is set by hand). The k of
each fact is a seeded permutation. One sleep, hippocampus off at test.
Arms: no_sleep; sleep_use (replay in proportion to the strength the hippocampus earned); sleep_equal (CONTROL: same
steps and episodes, all strengths 1).

  python -m brain.experiments.use_strength_probe --model <path.pth> --ceiling <ceiling json> --seed 42 --out <json>
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.ceiling_probe import POOL
from brain.experiments.sleep_probe import GENERIC, NEVER_TOLD, ask
from brain.hippocampus_use import UseHippocampus
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep

KS = (1, 3, 10)
N_FACTS = 12


def reachable_facts(ceiling_json):
    rows = json.load(open(ceiling_json))["rows"]
    keep = [p for p, r in zip(POOL, rows) if r["original"]]
    para_ok = {r["key"]: r["para"] for r in rows}
    return keep[:N_FACTS], para_ok


def evaluate(pc, state, facts, keys_all):
    rec = [k in ask(pc, state, q).lower() for _, q, k, _ in facts]
    par = [k in ask(pc, state, p).lower() for _, _, k, p in facts]
    false = [any(k in ask(pc, state, q).lower() for k in keys_all) for q in NEVER_TOLD]
    with torch.no_grad():
        drift = 0.0
        for g in GENERIC:
            ids = pc.cortex.pipe.encode(g)
            on = F.log_softmax(pc(ids, state)[0], -1)
            pc.enabled = False
            off = F.log_softmax(pc(ids, state)[0], -1)
            pc.enabled = True
            drift += float(F.kl_div(on, off, log_target=True, reduction="batchmean"))
    return {"recall": sum(rec), "paraphrase": sum(par), "false_recall": sum(false), "generic_kl": drift / len(GENERIC),
            "recall_items": rec, "paraphrase_items": par}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ceiling", required=True)
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
    facts, para_ok = reachable_facts(a.ceiling)
    keys_all = [f[2] for f in facts]
    order = list(range(len(facts)))
    random.Random(a.seed).shuffle(order)
    ks = [0] * len(facts)
    for pos, i in enumerate(order):
        ks[i] = KS[pos // 4]
    # tell each fact k times through the hippocampus, round-robin so tellings are spread over the "day"
    hip = UseHippocampus()
    for rnd in range(max(KS)):
        for i, (s, _, _, _) in enumerate(facts):
            if rnd < ks[i]:
                hip.store(c, base, s)
    assert len(hip.episodes) == len(facts), "re-telling must merge into one episode per fact"
    strengths = [e.strength for e in hip.episodes]
    assert strengths == [float(k) for k in ks], (strengths, ks)
    suffix = "\n\nAssistant: Okay, I'll remember that.\n\n"
    texts = [e.text + suffix for e in hip.episodes]
    arms = {"no_sleep": None,
            "sleep_use": [(t, e.strength) for t, e in zip(texts, hip.episodes)],
            "sleep_equal": [(t, 1.0) for t in texts]}
    result = {"model": a.model, "seed": a.seed, "steps": a.steps, "kl_weight": a.kl_weight, "lr": a.lr,
              "facts": [f[2] for f in facts], "k_of_fact": {f[2]: k for f, k in zip(facts, ks)},
              "learned_strength": {f[2]: s for f, s in zip(facts, strengths)},
              "para_reachable_in_context": {f[2]: para_ok[f[2]] for f in facts}, "arms": {}}
    for arm, eps in arms.items():
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        if eps is not None:
            sleep(pc, base, eps, GENERIC, steps=a.steps, lr=a.lr, kl_weight=a.kl_weight, seed=a.seed,
                  log=lambda m: print(arm, m, flush=True))
        r = evaluate(pc, base, facts, keys_all)
        r["recall_by_k"] = {str(k): sum(x for x, kk in zip(r["recall_items"], ks) if kk == k) for k in KS}
        r["paraphrase_by_k"] = {str(k): sum(x for x, kk in zip(r["paraphrase_items"], ks) if kk == k) for k in KS}
        r["adapter_norm"] = pc.adapter_norm()
        result["arms"][arm] = r
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    result["elapsed_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
