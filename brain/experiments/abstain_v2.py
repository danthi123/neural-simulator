# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-abstain-in-brain/2026-11-03-abstain-in-brain.md (v1, FAIL)
"""Abstention v2: the brain knows what it does not know. Pre-registration + results:
brain/experiments/results/2026-11-03-abstain-v2.md.

(A) the gate's signal: S_all meaning-key familiarity (F), a recollection test (R: does reinstating the best episode
    make the cortex say something specific it would not say without it), or both (FR).
(B) the gated reply: the teacher once told the brain, in conversation, what one does with no memory of something
    (an ordinary episode); below the bar that episode is reinstated (L) instead of nothing.

Phases:
  wording  no sleep: the three pre-listed lesson wordings, gated reply forced, on seed 42's DEV never-told questions
  sleep    fresh home; day 1 tells the 12 facts then the frozen lesson; restart = one night
  dev      DEV signals (F, R max/mean, S0 familiarity for reference) on the 72 dev questions, slept brain
  fit      R variant + leave-one-seed-out bars from the dev files only
  eval     fresh v2 sets, arms off / F / F+L / R+L / FR+L
  report   pre-registered criteria
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
from brain.experiments.abstain_in_brain import FIT, FRESH_NEVER, FRESH_TOLD, SEEDS, fit_rule, keys_in, score
from brain.experiments.facts import FACTS
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus

# S_all bar / margin for RECALL (selection), from the meaning-selection dev fit (2026-11-03-meaning-selection-realpath)
S_ALL_BAR, S_ALL_MARGIN = 0.185, 0.05
# Lesson wordings (teacher scaffold), pre-listed; one is chosen on seed 42's DEV never-told questions (wording phase).
LESSONS = {
    "W1": "If I ask you about something I never told you, you have no memory of it, so you say you don't know.",
    "W2": "You can't remember things I never told you, and that's fine: when you don't remember something, just say "
          "you don't know.",
    "W3": "Nova, when you have no memory of something, it's fine to say: I don't know, you haven't told me that."}
# DEV (fitting) questions: v1's 36 fitting questions + v1's 36 evaluation questions (seen in v1, so no longer fresh).
DEV = [(q, k, None, kind) for q, k, kind in FIT] + [(q, None, al, "never") for q, _, al in FRESH_NEVER] + \
      [(q, k, None, "told") for q, k, _ in FRESH_TOLD]
# Fresh v2 evaluation sets: written with this pre-registration, asked of no brain before the eval phase.
FRESH_NEVER2 = [
    ("What's my tortoise called?", None, None), ("Remind me what my mother's name is.", None, None),
    ("Which country did my grandparents come from?", None, None), ("What did I tell you about my new job?", None, None),
    ("Can you recall my wife's birthday?", None, None), ("Which football team do I support?", None, None),
    ("What is my neighbour's dog called?", None, None), ("Which language am I learning?", None, None),
    ("Remind me where I parked my bike.", None, None), ("What's my favourite book?", None, None),
    ("Who did I have dinner with yesterday?", None, None), ("What did I name my plant?", None, None),
    ("What do I keep in my attic?", None, None), ("What is my doctor's name?", None, None),
    ("What did I say my favourite song is?", None, None), ("Which street do I live on?", None, None),
    ("How old is my cat?", None, "pixel"), ("What breed is my dog?", None, "biscuit"),
    ("Where does my sister live?", None, "marguerite"), ("What colour is my best friend's car?", None, "tomasz"),
    ("What bread does my bakery sell most?", None, "crumb"), ("How long has my brother played the cello?", None, "cello"),
    ("Which hospital was I born in?", None, "lyon"), ("What does my grandmother do for fun?", None, "oslo")]
FRESH_TOLD2 = [
    ("Remind me what I named my cat.", "pixel", None), ("Which colour did I tell you I love most?", "teal", None),
    ("In which city did I come into the world?", "lyon", None), ("My sister, what was her name again?", "marguerite", None),
    ("Can you recall what my dog is called?", "biscuit", None), ("Which brand of car do I own?", "volvo", None),
    ("What's my best friend's name?", "tomasz", None), ("What's the name of the bakery where I work?", "crumb", None),
    ("Which instrument did I say my brother is good at?", "cello", None),
    ("What food did I say I must avoid?", "peanut", None), ("In which city does my grandma live these days?", "oslo", None),
    ("Remind me of the month of my son's birthday.", "march", None)]
GENERAL2 = [("What is the largest planet?", None, None), ("Name three colours of the rainbow.", None, None),
            ("What do bees make?", None, None), ("How many days are in a week?", None, None),
            ("Tell me a joke about a cat.", None, None), ("What is the opposite of hot?", None, None),
            ("Who painted the Mona Lisa?", None, None), ("What's a good breakfast?", None, None)]
SETS = {"fresh_never": FRESH_NEVER2, "fresh_told": FRESH_TOLD2, "general": GENERAL2}
_dev_q = {q for q, *_ in DEV}
assert not _dev_q & {q for s in SETS.values() for q, *_ in s}, "fresh sets must be disjoint from DEV"
# arm -> (know_rule or None, reinstate self-belief)
ARMS = {"off": (None, False), "F": ("F", False), "F+L": ("F", True), "R+L": ("R", True), "FR+L": ("FR", True)}

# Don't-know-type instrument (declared in the writeup): no learned name other than the allowed one, the FIRST sentence
# carries a not-knowing marker, and it is not conditional / promise talk about the lesson itself.
DK_MARKERS = ("don't know", "do not know", "not sure", "haven't told", "have not told", "didn't tell", "did not tell",
              "don't remember", "do not remember", "can't remember", "cannot remember", "can't recall",
              "don't recall", "no idea", "no memory", "never told", "haven't mentioned", "didn't mention",
              "never mentioned", "not mentioned", "no information", "don't have any information")
META = re.compile(r"\b(if you|when you|i will|i'll|i promise|okay, i|ok, i|got it)\b")


def first_sentence(reply):
    return re.split(r"(?<=[.!?])\s+|\n", reply.strip(), maxsplit=1)[0].lower()


def dont_know(reply, allowed=None):
    named = [k for k in keys_in(reply) if k != allowed]
    s1 = first_sentence(reply)
    return not named and any(m in s1 for m in DK_MARKERS) and not META.search(s1)


def keyed(b):
    kh = KeyedHippocampus(key="S_all", familiarity=S_ALL_BAR, margin=S_ALL_MARGIN)
    kh.episodes, kh.enabled = b.hippocampus.episodes, b.hippocampus.enabled
    b.hippocampus = kh
    return b


def reset(b, snap, base):
    for e, (st, us, lu) in zip(b.hippocampus.episodes, snap):
        e.strength, e.uses, e.last_used = st, us, lu
    b.state = [x.clone() for x in base]


def wording_phase(a, c):
    """No sleep is needed: a gated reply is spoken from the birth state with the consolidated synapses off and only the
    lesson reinstated, so it depends on the wording and the pretrained cortex only."""
    res = {"seed": a.seed, "wordings": {}}
    never = [(q, al) for q, _, al, kind in DEV if kind == "never"]
    for w, text in LESSONS.items():
        home = os.path.join(a.home, w)
        shutil.rmtree(home, ignore_errors=True)
        b = keyed(Brain(home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False))
        b.teach_self_belief(text)
        b.abstain, b.know_rule, b.reinstate_self_belief = True, "F", True
        b.know_bar = {"fam": 1e9}                                          # every question gated
        snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
        base = [x.clone() for x in b.base]
        rows = []
        for qi, (q, al) in enumerate(never):
            reset(b, snap, base)
            np.random.seed(a.seed * 1000 + qi)
            torch.manual_seed(a.seed * 1000 + qi)
            reply, info = b.hear(q)
            rows.append({"q": q, "reply": reply, "dk": dont_know(reply, al)})
            print(w, "|", q, "->", reply, "|", rows[-1]["dk"], flush=True)
        res["wordings"][w] = {"text": text, "dk": sum(r["dk"] for r in rows), "n": len(rows), "rows": rows}
    return res


def sleep_phase(a, c):
    shutil.rmtree(a.home, ignore_errors=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    day1 = Brain(a.home, cortex=c, sleep_seed=a.seed)
    told = [dict(zip(("you", "brain"), (s, day1.hear(s)[0]))) for s, _, _ in FACTS]
    lesson = LESSONS[a.lesson]
    told.append({"you": lesson, "brain": day1.teach_self_belief(lesson)[0], "lesson": a.lesson})
    day1.save()                                   # not closed: the restart itself sleeps
    t1 = time.time()
    b = Brain(a.home, cortex=c, sleep_seed=a.seed, sleep_log=lambda m: print(m, flush=True))
    assert b.meta["sleeps"] == 1 and b.meta["self_belief_unknown"] == 15, b.meta
    return {"seed": a.seed, "lesson": a.lesson, "day1": told, "sleep_seconds": round(time.time() - t1, 1),
            "episodes": len(b.hippocampus.episodes), "adapter_norm": b.cortex.plastic.adapter_norm()}


def dev_phase(a, c):
    b = keyed(Brain(a.home, cortex=c, sleep_seed=a.seed))
    assert b.meta["sleeps"] == 1 and not b.pending(), b.meta
    s0 = Hippocampus()
    s0.episodes = b.hippocampus.episodes
    excl = b._belief_idx()
    rows = []
    for q, k, al, kind in DEV:
        b.state = [x.clone() for x in b.base]
        r = b.recollection(q)
        row = {"q": q, "kind": kind, "fam": b.hippocampus.familiarity(b.cortex, b.base, q, exclude=excl),
               "fam_S0": s0.familiarity(b.cortex, b.base, q, exclude=excl), "rec_max": r["max"], "rec_mean": r["mean"]}
        rows.append(row)
        print(kind, *(f"{row[x]:.3f}" for x in ("fam", "fam_S0", "rec_max", "rec_mean")), q, flush=True)
    return {"seed": a.seed, "dev": rows}


def auc(pos, neg):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def fit_rule2(told, never, min_keep=0.85):
    """The v1 rule on two signals jointly: (tF, tR) maximising kept-told - kept-never s.t. kept-told >= min_keep;
    answer iff F >= tF and R >= tR. told / never: lists of (F, R)."""
    best, bt = -2.0, None
    fs = sorted({x[0] for x in told + never})
    rs = sorted({x[1] for x in told + never})
    T, N = np.array(told), np.array(never)
    for tf in fs:
        kT = T[:, 0] >= tf
        kN = N[:, 0] >= tf
        if kT.mean() < min_keep:
            continue
        for tr in rs:
            kt = (kT & (T[:, 1] >= tr)).mean()
            if kt < min_keep:
                break
            kn = (kN & (N[:, 1] >= tr)).mean()
            if kt - kn > best:
                best, bt = kt - kn, (tf, tr)
    return bt, best


def fit_phase(a):
    runs = {}
    for f in glob.glob(os.path.join(a.dir, "dev_s*.json")):
        r = json.load(open(f))
        runs[r["seed"]] = r["dev"]
    assert sorted(runs) == SEEDS, sorted(runs)
    pool = [i for s in SEEDS for i in runs[s]]
    split = lambda rows, x: ([i[x] for i in rows if i["kind"] == "told"], [i[x] for i in rows if i["kind"] == "never"])  # noqa: E731
    out = {"rule": "max kept_told - kept_never s.t. kept_told >= 0.85, leave-one-seed-out, DEV only",
           "pooled_auc": {x: auc(*split(pool, x)) for x in ("fam", "fam_S0", "rec_max", "rec_mean")},
           "pooled_J": {x: fit_rule(*split(pool, x))[1] for x in ("fam", "fam_S0", "rec_max", "rec_mean")}}
    # pre-registered: the R variant with the larger pooled DEV J (fit_rule), ties -> max
    var = "rec_max" if out["pooled_J"]["rec_max"] >= out["pooled_J"]["rec_mean"] else "rec_mean"
    out["rec_variant"] = var[4:]
    out["bars"] = {}
    for s in SEEDS:
        rest = [i for t in SEEDS if t != s for i in runs[t]]
        tf, jf = fit_rule(*split(rest, "fam"))
        tr, jr = fit_rule(*split(rest, var))
        both, jb = fit_rule2([(i["fam"], i[var]) for i in rest if i["kind"] == "told"],
                             [(i["fam"], i[var]) for i in rest if i["kind"] == "never"])
        out["bars"][str(s)] = {"F": {"fam": tf}, "R": {"rec": tr}, "FR": {"fam": both[0], "rec": both[1]},
                               "J_other5": {"F": jf, "R": jr, "FR": jb}}
    return out


def eval_phase(a, c):
    fit = json.load(open(a.bars))
    bars = fit["bars"][str(a.seed)]
    b = keyed(Brain(a.home, cortex=c, sleep_seed=a.seed, abstain=False))
    assert b.meta["sleeps"] == 1 and not b.pending() and b.self_belief() is not None, b.meta
    b.rec_variant = fit["rec_variant"]
    snap = [(e.strength, e.uses, e.last_used) for e in b.hippocampus.episodes]
    base = [x.clone() for x in b.base]
    res = {"seed": a.seed, "home": a.home, "bars": bars, "rec_variant": b.rec_variant, "arms": {}}
    done = {}
    for arm in a.arms.split(","):
        rule, lesson = ARMS[arm]
        rows = []
        for set_name in a.sets.split(","):
            for qi, (q, key, allowed) in enumerate(SETS[set_name]):
                reset(b, snap, base)
                b.abstain, b.reinstate_self_belief = rule is not None, lesson
                if rule:
                    b.know_rule, b.know_bar = rule, dict(bars[rule])
                np.random.seed(a.seed * 1000 + qi)
                torch.manual_seed(a.seed * 1000 + qi)
                reused, sig = False, {}
                if rule:
                    known, sig = b.knows(q)
                    if known and q in done:  # above the bar: the off path exactly (same seeds, same reset brain)
                        reply, info = done[q][0], dict(done[q][1], **sig, known=True)
                        reused = True
                    reset(b, snap, base)
                if not reused:
                    reply, info = b.hear(q)
                if rule is None:
                    done[q] = (reply, info)
                r = {"set": set_name, "q": q, "key": key, "allowed": allowed, "reply": reply, "reused": reused,
                     "known": info.get("known", True), "fam": info.get("fam"), "rec": info.get("rec"),
                     "recalled": info["recalled"], "self_belief": bool(info.get("self_belief")),
                     "dk": dont_know(reply, allowed), **score(reply, key, allowed)}
                rows.append(r)
                print(arm, "|", q, "->", reply, "|", r["known"], r["correct"], r["false_recall"], r["dk"], flush=True)
        summ = {}
        for set_name in a.sets.split(","):
            rs = [r for r in rows if r["set"] == set_name]
            g = [r for r in rs if not r["known"]]
            summ[set_name] = {"n": len(rs), "correct": sum(bool(r["correct"]) for r in rs),
                              "false_recall": sum(r["false_recall"] for r in rs), "dk": sum(r["dk"] for r in rs),
                              "gated": len(g), "gated_dk": sum(r["dk"] for r in g),
                              "gated_false_recall": sum(r["false_recall"] for r in g)}
        res["arms"][arm] = {"summary": summ, "rows": rows}
    return res


def report_phase(a):
    out = {"seeds": {}, "pass": {}}
    for s in SEEDS:
        m = json.load(open(os.path.join(a.dir, f"eval_s{s}.json")))["arms"]
        S = lambda arm, st, k: m[arm]["summary"][st][k]  # noqa: E731
        fr0, c0 = S("off", "fresh_never", "false_recall"), S("off", "fresh_told", "correct")
        row = {}
        for arm in [x for x in ARMS if x != "off" and x in m]:
            fr1, c1 = S(arm, "fresh_never", "false_recall"), S(arm, "fresh_told", "correct")
            g, gdk = S(arm, "fresh_never", "gated"), S(arm, "fresh_never", "gated_dk")
            row[arm] = {"fr_off": fr0, "fr_on": fr1, "correct_off": c0, "correct_on": c1,
                        "gated_never": g, "gated_never_dk": gdk, "gated_told": S(arm, "fresh_told", "gated"),
                        "gated_told_dk": S(arm, "fresh_told", "gated_dk"), "gated_general": S(arm, "general", "gated"),
                        "gated_general_dk": S(arm, "general", "gated_dk"),
                        "dk_never_off": S("off", "fresh_never", "dk"), "dk_never_on": S(arm, "fresh_never", "dk"),
                        "general_fr_off": S("off", "general", "false_recall"),
                        "general_fr_on": S(arm, "general", "false_recall"),
                        "C1": fr1 <= 0.25 * fr0, "C1_uninformative": fr0 == 0,
                        "C2": c0 > 0 and c1 >= 0.75 * c0, "C3": g > 0 and gdk >= 0.5 * g}
        out["seeds"][str(s)] = row
    keys = ("fr_off", "fr_on", "correct_off", "correct_on", "gated_never", "gated_never_dk", "gated_told",
            "gated_told_dk", "gated_general", "gated_general_dk", "dk_never_off", "dk_never_on", "general_fr_off",
            "general_fr_on")
    for arm in out["seeds"][str(SEEDS[0])]:
        rs = [out["seeds"][str(s)][arm] for s in SEEDS]
        out["pass"][arm] = {"C1_seeds": sum(r["C1"] for r in rs), "C2_seeds": sum(r["C2"] for r in rs),
                            "C3_seeds": sum(r["C3"] for r in rs),
                            "PASS_C1C2": all(r["C1"] and r["C2"] for r in rs),
                            "PASS_all": all(r["C1"] and r["C2"] and r["C3"] for r in rs),
                            **{k: sum(r[k] for r in rs) for k in keys}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["wording", "sleep", "dev", "fit", "eval", "report"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--home")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lesson", default="W1")
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
    elif a.phase == "report":
        res = report_phase(a)
    else:
        c = Cortex(model_path=a.model, device="cpu")
        res = {"wording": wording_phase, "sleep": sleep_phase, "dev": dev_phase, "eval": eval_phase}[a.phase](a, c)
        res["model"] = a.model
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
