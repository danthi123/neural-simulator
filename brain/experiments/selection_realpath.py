# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-02-record-brief-consolidation-and-selection.md (B4 selection benchmark);
#         brain/experiments/results/2026-11-03-meaning-selection.md (the Gram-matrix probe this one replaces)
"""Meaning-based selection through the REAL KeyedHippocampus path (store -> scores -> recall -> reinstate -> answer).

The 11-03 probe (selection_probe.py) recomputed similarities from Gram matrices and never called recall(). Here every
number comes from KeyedHippocampus: `scores()` (strength-weighted centred cosine, what recall competes on) and
`recall()` (the default Hippocampus competition: familiarity bar, margin, use boost), then `reinstate()` of the stored
fp16 delta at the default gain, then the cortex answers (greedy, 16 tokens; hit = answer word in the reply).

Two banks, never mixed: DEV_FACTS36 + DEV_NEVER_TOLD fit the bar and margin per key and model; FACTS48 + NEVER_TOLD
are the test (split per seed exactly as selection_probe: Random(seed) shuffle, 24 stored, 12 untold). Pass 1 records
scores on both banks for all seeds; the bar is then fit on DEV only (pooled over seeds); pass 2 runs recall() on TEST
with the arms: forced (bar and margin off: top-1), default (S0's 0.12 / 0.02), refit (dev-fit). Each recall() runs on
a fresh strength (strength / uses / last_used restored after the cue), so the cue order does not matter.

  python -m brain.experiments.selection_realpath --model <pth> --out <json> [--seeds ...] [--threads 3]
"""
import argparse
import json
import math
import os
import random
import time

import torch

from brain.brain import BIRTH
from brain.cortex import Cortex, DEFAULT_MODEL
from brain.experiments.facts_bank import DEV_FACTS36, DEV_NEVER_TOLD, FACTS48, NEVER_TOLD
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import KeyedHippocampus, episode_bytes

CUES = ("orig", "para_a", "para_b")
KEYS = ("S0", "S_all")
# Pre-registered (writeup, before the run): bar grid and objective.
BARS = [round(-0.10 + 0.005 * i, 3) for i in range(181)]          # -0.10 .. 0.80
MARGINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
OFF = -math.inf


def split(seed, n):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order[:24], order[24:36]


def cues_for(bank, never, stored, untold):
    """[(kind, cue, text, correct_index_in_store or None)]"""
    out = [("told", cue, bank[i][1 + j], n) for j, cue in enumerate(CUES) for n, i in enumerate(stored)]
    out += [("untold", "untold_fact", bank[i][1], None) for i in untold]
    out += [("untold", "never_told", q, None) for q in never]
    return out


def fresh(h):
    """Every stored episode is as fresh as at its telling (age 0, strength as stored): scores do not drift with the
    wall-clock time the probe takes."""
    now = time.time()
    for e in h.episodes:
        e.last_used = now


def decide(top, runner, bar, margin):
    return top >= bar and top - runner >= margin


def fit(rows):
    """J = P(fire and correct | told) - P(fire | untold), pooled. Ties: larger bar, then larger margin."""
    told = [r for r in rows if r["kind"] == "told"]
    untold = [r for r in rows if r["kind"] == "untold"]
    best = None
    for bar in BARS:
        for m in MARGINS:
            tp = sum(decide(r["top"], r["runner"], bar, m) and r["top_i"] == r["correct"] for r in told) / len(told)
            fp = sum(decide(r["top"], r["runner"], bar, m) for r in untold) / len(untold)
            cand = (tp - fp, bar, m, tp, fp)
            if best is None or cand[:3] > best[:3]:
                best = cand
    return {"J": best[0], "familiarity": best[1], "margin": best[2], "told_fire_correct": best[3], "untold_fire": best[4]}


def auroc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c = Cortex(model_path=a.model)
    _, base = c.read(BIRTH, None)
    base = [s.float() for s in base]
    banks = {"dev": (DEV_FACTS36, DEV_NEVER_TOLD), "test": (FACTS48, NEVER_TOLD)}

    def build(key, bank, stored, fam=None, mar=None):
        h = KeyedHippocampus(key=key, familiarity=fam, margin=mar)
        for i in stored:
            h.store(c, base, bank[i][0])
        return h

    # ---- pass 1: scores on both banks (no recall, nothing mutated) ----
    rows = {k: {"dev": [], "test": []} for k in KEYS}
    mem = {}
    for seed in a.seeds:
        for bname, (bank, never) in banks.items():
            stored, untold = split(seed, len(bank))
            for key in KEYS:
                h = build(key, bank, stored)
                mem.setdefault(key, episode_bytes(h.episodes[0]))
                for kind, cue, q, corr in cues_for(bank, never, stored, untold):
                    fresh(h)
                    s = h.scores(c, base, q)
                    o = torch.argsort(s, descending=True)
                    rows[key][bname].append({"seed": seed, "kind": kind, "cue": cue, "q": q, "correct": corr,
                                             "top_i": int(o[0]), "top": float(s[o[0]]), "runner": float(s[o[1]])})
            print(f"pass1 seed {seed} {bname} ({time.time() - t0:.0f}s)", flush=True)
    fits = {k: fit(rows[k]["dev"]) for k in KEYS}
    print("dev fits", fits, flush=True)

    # ---- pass 2: real recall() on TEST, then reinstate and answer ----
    answers = {}

    def answer(q, ep):
        k = f"{q}||{ep.text if ep is not None else ''}"
        if k not in answers:
            st0 = Hippocampus().reinstate(c, base, ep) if ep is not None else base
            lg, st = c.read(f"User: {q}\n\nAssistant:", st0)
            answers[k] = c.speak(lg, st, max_tokens=a.max_tokens, temperature=0)[0]
        return answers[k]

    arms = [(k, "forced", OFF, OFF) for k in KEYS]
    arms += [(k, "default", Hippocampus.FAMILIARITY, Hippocampus.MARGIN) for k in KEYS]
    arms += [(k, "refit", fits[k]["familiarity"], fits[k]["margin"]) for k in KEYS]
    result = {"model": a.model, "n_layers": c.n_layers, "seeds": {}, "fits": fits, "bytes_per_episode_after": mem,
              "delta_bytes": None, "bars": {"grid": [BARS[0], BARS[-1], 0.005], "margins": MARGINS}}
    bank, never = banks["test"]
    for seed in a.seeds:
        stored, untold = split(seed, 48)
        cues = cues_for(bank, never, stored, untold)
        res = {"stored": stored, "untold": untold, "arms": {}}
        hs = {k: build(k, bank, stored) for k in KEYS}
        result["delta_bytes"] = sum(d.numel() * d.element_size() for d in hs["S0"].episodes[0].delta if d is not None)
        srow = {k: [r for r in rows[k]["test"] if r["seed"] == seed] for k in KEYS}
        for key, arm, fam, mar in arms:
            h = hs[key]
            h.FAMILIARITY, h.MARGIN = fam, mar
            per = {}
            for (kind, cue, q, corr), r in zip(cues, srow[key]):
                assert r["q"] == q
                fresh(h)
                snap = [(e.strength, e.uses, e.last_used) for e in h.episodes]
                s2 = h.scores(c, base, q)
                o2 = torch.argsort(s2, descending=True)
                top, runner, top_i = float(s2[o2[0]]), float(s2[o2[1]]), int(o2[0])
                ep, sim = h.recall(c, base, q)
                for e, (s_, u_, l_) in zip(h.episodes, snap):
                    e.strength, e.uses, e.last_used = s_, u_, l_
                # P6: recall() decides exactly as the offline rule on scores()
                fired = ep is not None
                assert fired == decide(top, runner, fam, mar), (key, arm, q)
                assert not fired or h.episodes.index(ep) == top_i
                assert top_i == r["top_i"] and abs(top - r["top"]) < 1e-4  # pass-1 scores reproduce
                d = per.setdefault(cue, {"n": 0, "fire": 0, "sel": 0, "rec": 0})
                d["n"] += 1
                d["fire"] += fired
                if kind == "told":
                    d["sel"] += fired and top_i == corr
                    i = stored[corr]
                    d["rec"] += any(w in answer(q, ep).lower() for w in bank[i][4])
            res["arms"][f"{key}_{arm}"] = {cue: {m: v / d["n"] for m, v in d.items() if m != "n"} for cue, d in per.items()}
            print(seed, key, arm, {cue: round(v["rec"], 2) if cue in CUES else round(v["fire"], 2)
                                   for cue, v in res["arms"][f"{key}_{arm}"].items()}, flush=True)
        for ref in ("oracle", "none"):
            per = {}
            for kind, cue, q, corr in cues:
                if kind != "told":
                    continue
                i = stored[corr]
                ep = hs["S0"].episodes[corr] if ref == "oracle" else None
                per.setdefault(cue, []).append(any(w in answer(q, ep).lower() for w in bank[i][4]))
            res["arms"][ref] = {cue: {"rec": sum(v) / len(v)} for cue, v in per.items()}
        for key in KEYS:
            told = [r["top"] for r in srow[key] if r["kind"] == "told"]
            un = [r["top"] for r in srow[key] if r["kind"] == "untold"]
            res[f"auroc_{key}"] = auroc(told, un)
        result["seeds"][str(seed)] = res
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    result["scores"] = rows
    result["answers"] = answers
    result["elapsed_s"] = time.time() - t0
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
