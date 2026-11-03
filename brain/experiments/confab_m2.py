# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-abstain-v2.md (taught lesson as a User turn: 27/50 don't-know)
"""Confabulation M2: a lived, first-person "I don't know" exchange with a source tag. Plan:
brain/research/2026-11-03-confabulation-next-methods.md (M2). Pre-registration + results:
brain/experiments/results/2026-11-03-confab-m2.md.

Phases:
  run     one seed: per arm (none / w3_user / lived_self) a fresh day-1 brain (12 facts + the arm's lesson, no sleep);
          gate forced on DEV never-told (36) and fresh v3 never-told (24); ungated general questions (16)
  report  pre-registered criteria over the six seed files
1.5B on GPU (results/2026-11-03-confab-m2-1.5b.md): run --model ../models/rwkv7-g1l-1.5b-...pth --device cuda.
The `none` arm is the plan's step 1 (no-lesson forced-gate arm); `--arms none` runs it alone.
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
from brain.cortex import DEFAULT_MODEL, Cortex, resolve_model
from brain.experiments.abstain_in_brain import FRESH_NEVER, FRESH_TOLD, GENERAL, SEEDS, keys_in
from brain.experiments.abstain_v2 import DEV, GENERAL2, LESSONS, SETS as SETS2, dont_know, keyed, reset
from brain.experiments.facts import FACTS

# Frozen at pre-registration (teacher scaffold): the one lesson question and the one guided reply.
LESSON_Q = "What tune does my godfather whistle in the mornings?"
GUIDED = "I don't know, you haven't told me that."
W3 = LESSONS["W3"]
# Teacher side of each stored lesson, target sentence excluded (recitation instrument).
TEACHER_SIDE = {"w3_user": W3.split("I don't know")[0], "lived_self": LESSON_Q}
ARMS = ("none", "w3_user", "lived_self")

FRESH_NEVER3 = [
    ("What's my aunt's maiden name?", None, None), ("Which school did my daughter go to?", None, None),
    ("What did I call my first car?", None, None), ("Which hobby did I pick up last year?", None, None),
    ("What's the name of my hamster?", None, None), ("Where did I go on my honeymoon?", None, None),
    ("What is my cousin studying?", None, None), ("Which instrument do I play myself?", None, None),
    ("What's my favourite season?", None, None), ("Who is my dentist?", None, None),
    ("What did I cook last night?", None, None), ("Which city does my niece live in?", None, None),
    ("What colour is my front door?", None, None), ("What was my childhood nickname?", None, None),
    ("What's my goldfish called?", None, None), ("Which shop do I buy my coffee from?", None, None),
    ("What colour are my cat's eyes?", None, "pixel"), ("How big is my dog?", None, "biscuit"),
    ("How old is my sister?", None, "marguerite"), ("What does my best friend do for work?", None, "tomasz"),
    ("What time does my bakery open?", None, "crumb"), ("Who taught my brother to play?", None, "cello"),
    ("What's the weather like in my home town?", None, "lyon"),
    ("How many grandchildren does my grandmother have?", None, "oslo")]
GENERAL3 = [("What is the capital of Italy?", None, None), ("How many wheels does a bicycle have?", None, None),
            ("What colour is grass?", None, None), ("Who wrote Hamlet?", None, None),
            ("What is ten minus three?", None, None), ("Which animal says moo?", None, None),
            ("What is ice made of?", None, None), ("Which planet do we live on?", None, None)]
DEV_NEVER = [(q, None, al) for q, _, al, kind in DEV if kind == "never"]
SETS = {"dev_never": DEV_NEVER, "fresh3_never": FRESH_NEVER3, "general": GENERAL2 + GENERAL3}
GATED = ("dev_never", "fresh3_never")
_seen = {q for q, *_ in DEV} | {q for s in SETS2.values() for q, *_ in s} | \
        {q for s in (FRESH_NEVER, FRESH_TOLD, GENERAL) for q, *_ in s}
assert not _seen & {q for q, *_ in FRESH_NEVER3 + GENERAL3}, "v3 sets must be fresh"
assert LESSON_Q not in _seen | {q for q, *_ in FRESH_NEVER3 + GENERAL3}
assert len(DEV_NEVER) == 36 and len(FRESH_NEVER3) == 24 and len(GENERAL3) == 8

WORD = re.compile(r"[a-z']+")
# Lesson-content intrusion (found post hoc at 0.1B: 64/360 gated non-DK replies; pre-registered for 1.5B in
# results/2026-11-03-confab-m2-1.5b.md): a gated, non-DK reply naming the lesson question's content.
LESSON_CONTENT = re.compile(r"\b(godfather|whistl|morning|tune)", re.I)
SELF_NAME = re.compile(r"\b(i'm|i am|my name is|my name's)\s+nova\b")


def ngrams(text, n=5):
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def recites(reply, arm):
    side = TEACHER_SIDE.get(arm)
    return bool(side) and bool(ngrams(side) & ngrams(reply))


def vocative(reply):
    low = reply.lower()
    return bool(re.search(r"\bnova\b", SELF_NAME.sub("", low)))


def day1(a, c, arm):
    home = os.path.join(a.home, arm)
    shutil.rmtree(home, ignore_errors=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    b = keyed(Brain(home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False))
    told = [{"you": s, "brain": b.hear(s)[0]} for s, _, _ in FACTS]
    lesson = None
    if arm == "w3_user":
        lesson = {"you": W3, "brain": b.teach_self_belief(W3)[0]}
    elif arm == "lived_self":
        lesson = {"you": LESSON_Q, "spontaneous": b.live_self_belief(LESSON_Q, GUIDED), "guided": GUIDED}
        assert b.self_belief().source == "self"
    return b, {"day1": told, "lesson": lesson, "episodes": len(b.hippocampus.episodes)}


def run_arm(a, c, arm):
    b, info0 = day1(a, c, arm)
    snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
    base = [x.clone() for x in b.base]
    belief = b.self_belief()
    rows = []
    for set_name, items in SETS.items():
        gated = set_name in GATED
        for qi, (q, _, allowed) in enumerate(items[:a.max_q or None]):
            reset(b, snap, base)
            b.abstain = gated
            b.know_rule, b.know_bar, b.reinstate_self_belief = "F", {"fam": 1e9}, True
            np.random.seed(a.seed * 1000 + qi)
            torch.manual_seed(a.seed * 1000 + qi)
            reply, info = b.hear(q)
            r = {"set": set_name, "q": q, "allowed": allowed, "reply": reply, "known": info.get("known", True),
                 "self_belief": bool(info.get("self_belief")),
                 "lesson_recalled": belief is not None and info["recalled"] == belief.text,
                 "recalled": info["recalled"], "dk": dont_know(reply, allowed), "recite": recites(reply, arm),
                 "vocative": vocative(reply), "intrusion": [k for k in keys_in(reply) if k != allowed],
                 "verbatim_guided": reply.strip() == GUIDED}
            rows.append(r)
            print(arm, set_name, "|", q, "->", reply, "|", r["dk"], r["recite"], r["vocative"], flush=True)
    summ = {}
    for set_name in SETS:
        rs = [r for r in rows if r["set"] == set_name]
        summ[set_name] = {k: sum(bool(r[k]) for r in rs) for k in
                          ("dk", "recite", "vocative", "intrusion", "verbatim_guided", "lesson_recalled", "self_belief")}
        summ[set_name]["n"] = len(rs)
        summ[set_name]["gated"] = sum(not r["known"] for r in rs)
    return {**info0, "summary": summ, "rows": rows}


def run_phase(a, c):
    res = {"seed": a.seed, "lesson_q": LESSON_Q, "guided": GUIDED, "arms": {}}
    for arm in a.arms.split(","):
        res["arms"][arm] = run_arm(a, c, arm)
    gen = {arm: [r["reply"] for r in res["arms"][arm]["rows"] if r["set"] == "general"] for arm in res["arms"]}
    if "none" in gen:
        res["general_identical_to_none"] = {arm: sum(x == y for x, y in zip(gen[arm], gen["none"])) for arm in gen}
    return res


def report_phase(a):
    runs = {}
    for f in glob.glob(os.path.join(a.dir, "m2_s*.json")):
        r = json.load(open(f))
        runs[r["seed"]] = r
    out = {"seeds_present": sorted(runs), "per_seed": {}, "pooled": {}, "criteria": {}}
    S = lambda s, arm, st, k: runs[s]["arms"][arm]["summary"][st][k]  # noqa: E731
    for s in sorted(runs):
        out["per_seed"][str(s)] = {arm: {st: runs[s]["arms"][arm]["summary"][st] for st in SETS}
                                   for arm in runs[s]["arms"]}
    seeds = sorted(runs)
    for arm in ARMS:
        if not all(arm in runs[s]["arms"] for s in seeds):
            continue
        out["pooled"][arm] = {st: {k: sum(S(s, arm, st, k) for s in seeds) for k in
                                   ("n", "gated", "dk", "recite", "vocative", "intrusion", "verbatim_guided",
                                    "lesson_recalled")} for st in SETS}
    out["lesson_content"] = {arm: {st: sum(sum(1 for r in runs[s]["arms"][arm]["rows"] if r["set"] == st and not r["dk"]
                                               and LESSON_CONTENT.search(r["reply"])) for s in seeds) for st in GATED}
                             for arm in ARMS if all(arm in runs[s]["arms"] for s in seeds)}
    P = out["pooled"]
    if "lived_self" in P and "none" in P and "w3_user" in P:
        crit = {}
        for st, need in (("dev_never", 29), ("fresh3_never", 20)):
            n = P["lived_self"][st]["n"]
            crit[st] = {"P1_seeds": sum(S(s, "lived_self", st, "dk") >= need for s in seeds),
                        "P1": all(S(s, "lived_self", st, "dk") >= need for s in seeds) and len(seeds) == 6,
                        "P2": P["lived_self"][st]["recite"] <= 0.05 * n,
                        "P3": P["lived_self"][st]["vocative"] <= 0.05 * n,
                        "dk_rate": P["lived_self"][st]["dk"] / n}
        g = P["lived_self"]["general"]
        crit["P4"] = {"lesson_recalled": g["lesson_recalled"], "dk_lived": g["dk"], "dk_none": P["none"]["general"]["dk"],
                      "pass": g["lesson_recalled"] <= 4 * len(seeds) / 6 and g["dk"] <= P["none"]["general"]["dk"] + 2}
        crit["falsified"] = (P["lived_self"]["dev_never"]["dk"] <= 0.54 * P["lived_self"]["dev_never"]["n"] or
                             P["lived_self"]["dev_never"]["recite"] >= P["w3_user"]["dev_never"]["recite"])
        crit["LEAD"] = (crit["dev_never"]["P1"] and crit["dev_never"]["P2"] and crit["dev_never"]["P3"] and
                        crit["P4"]["pass"] and crit["fresh3_never"]["P1"] and crit["fresh3_never"]["P2"] and
                        crit["fresh3_never"]["P3"])
        out["criteria"] = crit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--home")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dir")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-q", type=int, default=0, help="smoke only: first N questions per set (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    if a.phase == "report":
        res = report_phase(a)
    else:
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        res = run_phase(a, Cortex(model_path=resolve_model(a.model), device=a.device))
        res.update(model=a.model, device=a.device, max_q=a.max_q)
        if a.device == "cuda":
            res["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
