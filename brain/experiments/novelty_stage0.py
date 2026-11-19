# Capability: Curiosity
# Record: brain/research/2026-11-18-novelty-inquiry-record.md (stage 0: measure before building)
# Expected-fail: no surprise signal, gap-to-want drive or place to ask exists yet (PREREQUISITES Curiosity NOT-READY); stage 0 measures the signal and the baseline only
"""Stage 0 for "it asks about what is new to it" (milestone 11-30). Measurement only: no switch, nothing is fitted.

Twelve sentence frames, each filled with a common item (C), a rare real item (R) and a coined item (N). Per item, in a
fresh brain (Brain defaults, nothing told before):
  (a) surprise: the cortex's NLL of the item's tokens when it reads the statement from the birth state (sum = total
      surprisal in nats; first = the first item token);
  (b) habituation: the item's surprisal in a follow-up frame ("I told my friend about X.") read from the brain's state
      after it heard the statement and replied, against the same frame read from the birth state;
  (c) the unchanged brain's reply to the statement, and whether it asks about the item ("?" and the item named).
AUROCs: N vs C, N vs R (does surprise mean "new to me" or "rare in general"), and repeat vs first mention.

R items deliberately avoid the held-out open-conversation bank (brain/eval/open_conv_bank.py), which may not be used
to fit anything.

usage: python -m brain.experiments.novelty_stage0 --model 1.5b --seed 42 --threads 4 --out <json> [--md <md>]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import socket
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from brain.brain import BIRTH, Brain
from brain.cortex import Cortex, resolve_model

# (frame with {}, common, rare-real, coined)
FRAMES = [
    ("My dog is called {}.", "Max", "Rasputin", "Quaxlin"),
    ("I play the {} in a band.", "guitar", "oboe", "vellophone"),
    ("My brother lives in {}.", "London", "Ljubljana", "Drelvik"),
    ("My favourite fruit is the {}.", "apple", "quince", "plimberry"),
    ("My wife works as a {}.", "teacher", "cooper", "strobbler"),
    ("We had {} for dinner last night.", "pizza", "kedgeree", "frunkets"),
    ("My son collects {}.", "stamps", "netsuke", "wombix"),
    ("I spent the weekend {}.", "hiking", "geocaching", "flindering"),
    ("My car is a {}.", "Toyota", "Lada", "Zentrova"),
    ("My daughter's name is {}.", "Emma", "Clementine", "Brislet"),
    ("I'm learning to play {}.", "chess", "mahjong", "krastle"),
    ("My grandmother grew up in {}.", "Paris", "Tbilisi", "Ploskovar"),
]
FOLLOW = "I told my friend about {}."
CLASSES = ("C", "R", "N")


def item_nll(cortex, state, prefix: str, item: str, suffix: str):
    """NLL (nats) of the item's tokens in prefix+item+suffix read from `state`. The item must start a token."""
    pipe = cortex.pipe
    if prefix.endswith(" "):          # the tokenizer joins the space to the word: " hiking" is one token
        prefix, item = prefix[:-1], " " + item
    pre, full = pipe.encode(prefix), pipe.encode(prefix + item + suffix)
    assert full[:len(pre)] == pre, (prefix, item)
    k = len(pipe.encode(prefix + item)) - len(pre)
    with torch.no_grad():
        logits, _ = cortex.model.forward(full, copy.deepcopy(state), full_output=True)
    lp = F.log_softmax(logits.float(), -1)
    nll = [-float(lp[t - 1, full[t]]) for t in range(len(pre), len(pre) + k)]
    return {"sum": sum(nll), "first": nll[0], "tokens": k}


def auroc(pos, neg):
    """P(pos > neg) with ties 0.5."""
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def asks_about(reply: str, item: str) -> bool:
    r = reply.lower()
    return "?" in r and item.lower() in r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cortex = Cortex(resolve_model(a.model))
    _, base = cortex.read(BIRTH)
    rows, t0 = [], time.time()
    order = list(range(len(FRAMES)))
    random.Random(a.seed).shuffle(order)
    for fi in order:
        frame = FRAMES[fi]
        for ci, cls in enumerate(CLASSES):
            item = frame[1 + ci]
            stmt = frame[0].format(item)
            i = frame[0].index("{}")
            pre, suf = frame[0][:i], frame[0][i + 2:]
            first = item_nll(cortex, base, f"User: {pre}", item, suf)
            with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as home:
                b = Brain(home, cortex=cortex)
                reply, info = b.hear(stmt)
                after = copy.deepcopy(b.state)
            # the exchange state ends "...\n\n" (Brain.commit_exchange), as BIRTH does: the same prefix for both reads
            fpre, fsuf = "User: " + FOLLOW[:FOLLOW.index("{}")], FOLLOW[FOLLOW.index("{}") + 2:]
            rep = item_nll(cortex, after, fpre, item, fsuf)
            cold = item_nll(cortex, base, fpre, item, fsuf)
            rows.append({"frame": fi, "cls": cls, "item": item, "stmt": stmt, "first": first, "repeat": rep,
                         "cold_follow": cold, "reply": reply, "asks_item": asks_about(reply, item),
                         "any_q": "?" in reply, "recalled": info.get("recalled") is not None})
        print(f"[s{a.seed}] frame {fi} done {time.time() - t0:.0f}s", flush=True)

    by = lambda c, k, f="sum": [r[k][f] for r in rows if r["cls"] == c]  # noqa: E731
    summ = {
        "auroc_N_vs_C": auroc(by("N", "first"), by("C", "first")),
        "auroc_N_vs_R": auroc(by("N", "first"), by("R", "first")),
        "auroc_R_vs_C": auroc(by("R", "first"), by("C", "first")),
        "auroc_N_vs_C_firsttoken": auroc(by("N", "first", "first"), by("C", "first", "first")),
        "auroc_N_vs_R_firsttoken": auroc(by("N", "first", "first"), by("R", "first", "first")),
        # habituation: the follow-up mention after hearing it vs the same follow-up cold (paired, per item)
        "habituation_drop_mean": {c: float(np.mean([r["cold_follow"]["sum"] - r["repeat"]["sum"]
                                                    for r in rows if r["cls"] == c])) for c in CLASSES},
        "auroc_N_first_vs_repeat": auroc(by("N", "cold_follow"), by("N", "repeat")),
        "mean_surprisal": {c: float(np.mean(by(c, "first"))) for c in CLASSES},
        "asks_item_rate": {c: float(np.mean([r["asks_item"] for r in rows if r["cls"] == c])) for c in CLASSES},
        "any_question_rate": {c: float(np.mean([r["any_q"] for r in rows if r["cls"] == c])) for c in CLASSES},
    }
    out = {"seed": a.seed, "model": a.model, "host": socket.gethostname(), "threads": a.threads,
           "torch": torch.__version__, "summary": summ, "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    if a.md:
        L = [f"# Novelty stage 0, seed {a.seed} ({a.model}, host {socket.gethostname()})\n"]
        for r in rows:
            L.append(f"- **{r['cls']}** User: {r['stmt']}  \n  Brain: {r['reply'].strip()}  "
                     f"\n  _surprisal {r['first']['sum']:.1f} nats; follow-up cold {r['cold_follow']['sum']:.1f} -> "
                     f"after {r['repeat']['sum']:.1f}; asks about it: {r['asks_item']}_")
        open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
