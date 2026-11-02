# Capability: Sleep consolidation
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md
"""Does sleep move told facts into the cortex, so they are recalled with the hippocampus OFF?

Arms (same seed, same steps): no-sleep; sleep on the true episodes; sleep on SHUFFLED episodes (names swapped
between facts: same exposure, wrong content, so a pass there would mean the test is broken). Read-outs, all with the
hippocampus off: recall on the original questions and on paraphrases; false recall on questions about things never
told; drift on ordinary text (KL to the pretrained cortex). Expected ceiling: the cortex's own in-context recall at
this size (0.1B: 6/12, 0.4B: 11/12; brain/experiments/results/2026-11-02-scale-ladder.txt).

  python -m brain.experiments.sleep_probe --model <path.pth> --steps 240 --seed 7 --out <json>
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts import CHAT, FACTS, PARA
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep

NEVER_TOLD = ["What is my horse called?", "Where does my uncle live?", "What is my daughter's name?",
              "What instrument does my mother play?"]
GENERIC = [CHAT, "User: What's a good way to spend a rainy afternoon?\n\nAssistant: Reading a book with tea is nice.\n\n",
           "User: Can you count to five?\n\nAssistant: One, two, three, four, five.\n\n"]


def ask(pc, state, q, max_tokens=16):
    """Greedy answer from the plastic cortex (hippocampus off)."""
    ids = pc.cortex.pipe.encode(f"User: {q}\n\nAssistant:")
    out = []
    with torch.no_grad():
        logits, st = pc(ids, state)
        lg = logits[-1]
        for _ in range(max_tokens):
            tok = int(torch.argmax(lg))
            if tok == 0:
                break
            out.append(tok)
            text = pc.cortex.pipe.decode(out)
            if "\n\n" in text:
                break
            lg, st = pc([tok], st)
            lg = lg[-1]
    return pc.cortex.pipe.decode(out).split("\n\n")[0].strip()


def evaluate(pc, state, keys_all):
    rec = [k in ask(pc, state, q).lower() for _, q, k in FACTS]
    par = [k in ask(pc, state, q).lower() for q, (_, _, k) in zip(PARA, FACTS)]
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
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    c = Cortex(model_path=a.model, device=a.device)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    keys_all = [k for _, _, k in FACTS]
    told = [(f"User: {s}\n\nAssistant: Okay, I'll remember that.\n\n", 1.0) for s, _, _ in FACTS]
    rng = random.Random(a.seed)
    names = [k for _, _, k in FACTS]
    perm = names[:]
    while any(x == y for x, y in zip(perm, names)):
        rng.shuffle(perm)
    shuffled = []
    for (s, _, k), wrong in zip(FACTS, perm):  # same sentence, another fact's answer word
        i = s.lower().find(k)
        shuffled.append((f"User: {s[:i]}{wrong.capitalize()}{s[i + len(k):]}\n\nAssistant: Okay, I'll remember that.\n\n", 1.0))
    result = {"model": a.model, "seed": a.seed, "steps": a.steps, "arms": {}}
    for arm, eps in (("no_sleep", None), ("sleep_true", told), ("sleep_shuffled", shuffled)):
        torch.manual_seed(a.seed)
        pc = PlasticCortex(c)
        losses = []
        if eps is not None:
            losses = sleep(pc, base, eps, GENERIC, steps=a.steps, seed=a.seed, log=lambda m: print(arm, m, flush=True))
        r = evaluate(pc, base, keys_all)
        r["final_replay_loss"] = losses[-1][0] if losses else None
        r["adapter_norm"] = pc.adapter_norm()
        result["arms"][arm] = r
        print(arm, {k: v for k, v in r.items() if not k.endswith("_items")}, flush=True)
    result["elapsed_s"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
