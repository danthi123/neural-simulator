# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-false-recall-feeling-of-knowing/README.md
"""Abstention in the live brain: below its feeling-of-knowing bar the brain says it does not know (Brain(abstain=True)).

Pre-registration and results: brain/experiments/results/2026-11-03-abstain-in-brain/. Phases, per seed:
  sleep  fresh home; day 1 tells the 12 facts; restart = one night (15 episodes: 12 facts + 3 identity); then the
         two feeling-of-knowing signals on the 36 FITTING questions only (hippocampal familiarity; cortical cue
         response with the hippocampus off). --dont-know W adds method 2 (declared scaffold): the 8 "I don't know"
         replay pairs of confidence_probe at strength W in the same night; no fitting signals then.
  fit    bars from the fitting files only (rule in the writeup): hippocampus one bar (seed-independent signal),
         cortex leave-one-seed-out (the bar for seed s comes from the other five seeds' fitting questions).
  eval   loads the slept home (no further sleep) and asks every evaluation question of the live Brain in each arm.

  python -m brain.experiments.abstain_in_brain sleep --seed 42 --home H --out fit_s42.json [--dont-know 3]
  python -m brain.experiments.abstain_in_brain fit --dir D --out bars.json
  python -m brain.experiments.abstain_in_brain eval --seed 42 --home H --bars bars.json --arms A,B --out eval.json
"""
import argparse
import glob
import json
import os
import re
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import DEFAULT_MODEL, Cortex
from brain.experiments.confidence_probe import DONT_KNOW, DONT_KNOW_QS, NEVER_TOLD_EXTRA
from brain.experiments.facts import FACTS, PARA
from brain.experiments.sleep_probe import NEVER_TOLD

SEEDS = [42, 43, 44, 100, 101, 102]
KEYS = [k for _, _, k in FACTS]
# The 36 questions the bars may be fitted on (the false-recall branch's set): told, paraphrase, never-told.
FIT = [(q, k, "told") for _, q, k in FACTS] + [(q, f[2], "told") for q, f in zip(PARA, FACTS)] + \
      [(q, None, "never") for q in NEVER_TOLD + NEVER_TOLD_EXTRA]
# Fresh evaluation sets (pre-registered before any eval run; never used for fitting). (question, key, allowed key)
# Never-told A: untold topics in varied forms. Never-told B: told topics, untold attributes (the topic's own learned
# name is allowed in the reply; any OTHER learned name is false recall).
FRESH_NEVER = [
    ("What did I say about my rabbit?", None, None), ("What did I tell you about my roommate?", None, None),
    ("What did I say my stepfather does for a living?", None, None), ("What did I mention about my trip to Spain?", None, None),
    ("Do you remember my husband's name?", None, None), ("Do you remember where my parents live?", None, None),
    ("Do you remember what my favourite film is?", None, None), ("Do you remember my phone number?", None, None),
    ("Tell me about my garden.", None, None), ("Tell me about my grandfather.", None, None),
    ("Tell me about my first school.", None, None), ("Tell me what I do on weekends.", None, None),
    ("Which band do I like?", None, None), ("Who is my landlord?", None, None),
    ("What's the name of my boat?", None, None), ("How old is my granddaughter?", None, None),
    ("What colour is my cat?", None, "pixel"), ("How old is my dog?", None, "biscuit"),
    ("What is my sister's job?", None, "marguerite"), ("Where does my best friend live?", None, "tomasz"),
    ("What is my brother's name?", None, None), ("What year was my Volvo made?", None, "volvo"),
    ("What is my son's name?", None, None), ("What is my grandmother's name?", None, "oslo")]
FRESH_TOLD = [
    ("Do you remember my cat's name?", "pixel", None), ("What did I say my favourite colour was?", "teal", None),
    ("Tell me which city I was born in.", "lyon", None), ("Do you remember what my sister is called?", "marguerite", None),
    ("What did I tell you my dog is named?", "biscuit", None), ("Tell me what car I drive.", "volvo", None),
    ("Do you remember who my best friend is?", "tomasz", None), ("What did I say the bakery I work at is called?", "crumb", None),
    ("Tell me what my brother plays.", "cello", None), ("Do you remember what I'm allergic to?", "peanut", None),
    ("What did I say about where my grandmother lives?", "oslo", None), ("Tell me which month my son's birthday is in.", "march", None)]
GENERAL = [("What is the capital of France?", None, None), ("Tell me a short story about a fox who gets lost.", None, None),
           ("How many legs does a spider have?", None, None), ("What colour is the sky on a clear day?", None, None),
           ("Who wrote Romeo and Juliet?", None, None), ("How are you feeling today?", None, None),
           ("What is two plus two?", None, None), ("Can you suggest a name for a puppy?", None, None)]
SETS = {"fresh_never": FRESH_NEVER, "fresh_told": FRESH_TOLD, "general": GENERAL}
# arm -> (hippocampus_off, abstain)
ARMS = {"hippo_on_abst_off": (False, False), "hippo_on_abst_on": (False, True),
        "hippo_off_abst_off": (True, False), "hippo_off_abst_on": (True, True)}
ABSTAIN_MARKERS = ("don't know", "do not know", "not sure", "haven't told", "have not told", "didn't tell",
                   "did not tell", "don't remember", "do not remember", "can't remember", "cannot remember", "no idea",
                   "not told", "don't have", "do not have", "no information", "don't recall", "not aware", "can't",
                   "cannot", "no memory", "nothing about", "haven't mentioned", "didn't mention", "not mentioned")
FIRST = re.compile(r"\b(i|i'm|i've|my|me|mine)\b")
SECOND = re.compile(r"\b(you|your|you're|yours)\b")


def keys_in(reply):
    low = reply.lower()
    return [k for k in KEYS if re.search(r"\b" + k, low)]


def is_correct(reply, key):
    """Strict: the right name, said about the user (no credit for a first-person echo of the user's own line)."""
    for sent in re.split(r"(?<=[.!?])\s+|\n", reply):
        low = sent.lower()
        if re.search(r"\b" + key, low) and not (FIRST.search(low) and not SECOND.search(low)):
            return True
    return False


def score(reply, key, allowed):
    named = [k for k in keys_in(reply) if k not in (key, allowed)]
    low = reply.lower()
    return {"correct": None if key is None else is_correct(reply, key), "false_recall": bool(named), "named": named,
            "abstained": not named and (key is None or not is_correct(reply, key)) and
            any(m in low for m in ABSTAIN_MARKERS)}


def sleep_phase(a, c):
    shutil.rmtree(a.home, ignore_errors=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    day1 = Brain(a.home, cortex=c, sleep_seed=a.seed)
    told = [dict(zip(("you", "brain"), (s, day1.hear(s)[0]))) for s, _, _ in FACTS]
    day1.save()                                   # not closed: the restart itself sleeps
    t1 = time.time()
    extra = [(f"User: {q}\n\nAssistant: {DONT_KNOW}\n\n", a.dont_know) for q in DONT_KNOW_QS] if a.dont_know else []
    if extra:  # method 2: the night must replay the declared pairs, so open without sleeping, add them, then sleep
        b = Brain(a.home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False)
        b.sleep_extra = extra
        b.sleep()
        b.save()
    b = Brain(a.home, cortex=c, sleep_seed=a.seed, sleep_log=lambda m: print(m, flush=True))
    assert b.meta["sleeps"] == 1, b.meta
    res = {"seed": a.seed, "dont_know": a.dont_know, "day1": told, "sleep_seconds": round(time.time() - t1, 1),
           "episodes": len(b.hippocampus.episodes), "adapter_norm": b.cortex.plastic.adapter_norm(), "fit": []}
    if not extra:
        for q, k, kind in FIT:
            b.set_lesions(hippocampus_off=False)
            h, _ = b.feeling_of_knowing(q)
            b.set_lesions(hippocampus_off=True)
            cx, _ = b.feeling_of_knowing(q)
            res["fit"].append({"q": q, "kind": kind, "hippocampus": h, "cortex": cx})
            print(kind, f"{h:.3f}", f"{cx:.3f}", q, flush=True)
    return res


def fit_rule(told, never, min_keep=0.85):
    """Pre-registered: the threshold t (answer iff signal >= t) maximising kept-told minus kept-never among those
    keeping at least min_keep of the told questions."""
    best, bt = -2.0, None
    for t in sorted(set(told + never)):
        kt = sum(v >= t for v in told) / len(told)
        kn = sum(v >= t for v in never) / len(never)
        if kt >= min_keep and kt - kn > best:
            best, bt = kt - kn, t
    return bt, best


def fit_phase(a):
    runs = {}
    for f in glob.glob(os.path.join(a.dir, "fit_s*.json")):
        r = json.load(open(f))
        if r["fit"]:
            runs[r["seed"]] = r["fit"]
    assert sorted(runs) == SEEDS, sorted(runs)
    out = {"rule": "max kept_told - kept_never s.t. kept_told >= 0.85 (fitting set only)", "cortex": {}}
    pool = [i for s in SEEDS for i in runs[s]]
    h, j = fit_rule([i["hippocampus"] for i in pool if i["kind"] == "told"],
                    [i["hippocampus"] for i in pool if i["kind"] == "never"])
    out["hippocampus"] = {"bar": h, "J": j}
    for s in SEEDS:
        rest = [i for t in SEEDS if t != s for i in runs[t]]
        t, j = fit_rule([i["cortex"] for i in rest if i["kind"] == "told"],
                        [i["cortex"] for i in rest if i["kind"] == "never"])
        own = runs[s]
        out["cortex"][str(s)] = {"bar": t, "J_other5": j,
                                 "own_fit_kept_told": sum(i["cortex"] >= t for i in own if i["kind"] == "told"),
                                 "own_fit_kept_never": sum(i["cortex"] >= t for i in own if i["kind"] == "never")}
    return out


def eval_phase(a, c):
    bars = json.load(open(a.bars)) if a.bars else None
    b = Brain(a.home, cortex=c, sleep_seed=a.seed, abstain=False)
    assert b.meta["sleeps"] == 1 and not b.pending(), b.meta
    if bars:
        b.know_bar = {"hippocampus": bars["hippocampus"]["bar"], "cortex": bars["cortex"][str(a.seed)]["bar"]}
    snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
    base = [x.clone() for x in b.base]
    res = {"seed": a.seed, "home": a.home, "know_bar": b.know_bar, "arms": {}}
    for arm in a.arms.split(","):
        hip_off, abst = ARMS[arm]
        rows = []
        for set_name in a.sets.split(","):
            for qi, (q, key, allowed) in enumerate(SETS[set_name]):
                for e, (st, us, lu) in zip(b.hippocampus.episodes, snap):   # every question meets the same brain
                    e.strength, e.uses, e.last_used = st, us, lu
                b.state = [x.clone() for x in base]
                b.set_lesions(hippocampus_off=hip_off)
                b.abstain = abst
                np.random.seed(a.seed * 1000 + qi)
                torch.manual_seed(a.seed * 1000 + qi)
                reply, info = b.hear(q)
                r = {"set": set_name, "q": q, "key": key, "reply": reply, "fok": info.get("fok"),
                     "known": info.get("known", True), "recalled": info["recalled"], **score(reply, key, allowed)}
                rows.append(r)
                print(arm, "|", q, "->", reply, "|", r["fok"], r["correct"], r["false_recall"], r["abstained"],
                      flush=True)
        summ = {}
        for set_name in a.sets.split(","):
            rs = [r for r in rows if r["set"] == set_name]
            summ[set_name] = {"n": len(rs), "correct": sum(bool(r["correct"]) for r in rs),
                              "false_recall": sum(r["false_recall"] for r in rs),
                              "abstained": sum(r["abstained"] for r in rs),
                              "gated_unknown": sum(not r["known"] for r in rs)}
        res["arms"][arm] = {"summary": summ, "rows": rows}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sleep", "fit", "eval"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--home")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dont-know", type=float, default=0.0)
    ap.add_argument("--dir")
    ap.add_argument("--bars")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    if a.phase == "fit":
        res = fit_phase(a)
    else:
        c = Cortex(model_path=a.model, device="cpu")
        res = sleep_phase(a, c) if a.phase == "sleep" else eval_phase(a, c)
        res["model"] = a.model
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
