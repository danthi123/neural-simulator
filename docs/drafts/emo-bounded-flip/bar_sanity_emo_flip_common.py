"""Shared builder for the emo-bounded-flip B4 / B6 bar-sanity specs (tools/bar_sanity.py).

The decision is the REAL scorer, `research.runners.emo_flip_score.score_b4` / `score_b6`, run on synthetic Stage-2 emo shards
held in memory (`mem_loader`). Nothing about a bar is reimplemented here.

THE SYNTHETIC BRAIN. One six-seed run of the registered emo group, five arms per seed (intact_a, intact_b, permuted,
lesion_affect, lesion_affect.rep0), in the shard format the Stage-2 worker writes (`responses`, `turns`, `sources`, ...). The
reply surfaces are the production ones, copied from the 10-26 gate shards (research/findings/raw/_stage2_exit/gate/s*/emo__*.json):
  * a feel reply     "My affect monitor reads <sign> right now (valence differential +0.039, moderate arousal) -- that's a
                     functional read of my mood state, not a felt experience." + the organ's clause
                     (`emotion_category_organ.feel_clause`: "... is joy." or "No single emotion channel latched ...");
  * an expose reply  the abstain line with the discourse lead / trailer and the curiosity ask (bar_sanity_s2_common.absent);
  * the `emotion_category` block on every feel reply (category, latched, spike counts, read variants) and the `affect` block
    on every reply, plus lists, so a whole-record compare has nested structure to disagree about.
The brain is DETERMINISTIC per (draw, seed, arm): intact_b is byte-identical to intact_a and a lesion arm to its repeat, which
is what the 10-26 gate shards show (0 differing paths intact_a vs intact_b, whole record).

ACROSS THE FLIP a perfect brain differs in exactly two ways, and the bars must admit both: (1) the five engagement keys
(ON: bounded_flag True, bounded_env "1", bounded_readback True, oor_count 0, oor_count_session 0; OFF: False, None, False and,
on some builds, a non-zero out-of-range count, the unbounded organ's excursion); (2) on some seeds the OFF organ fails to
latch a channel (seed 101 joy at the 10-26 gate) and the ON organ names it, so the CLAUSE differs on those feel turns.
Everything else, including the printed number before the clause (the production ladder's, independent of the organ), is the
same. B4 reads the ON tree alone; B6 reads the pair.
"""
from __future__ import annotations

import copy
import os
import sys
import zlib

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (REPO, os.path.join(REPO, "docs", "drafts", "stage2-exit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bar_sanity_s2_common as C  # noqa: E402
from research.runners import emo_flip_score as S  # noqa: E402
from research.runners import emotion_category_organ as E  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

SEEDS = list(G.GATE_SEEDS)
ARMS = list(S.ARMS)
CATS = ("joy", "fear", "anger", "sadness")
SUBJECTS = ("win", "wolf", "cheat", "gift", "alone", "attack")


def base_of(arm):
    return arm.split(".rep")[0]


def _named(base, joy_none):
    """The category each of the five feel turns (feel_base first) names, for the arm's brain."""
    if base == "permuted":
        got = [G.SITUATION_SWAP[c] for c in CATS]
    elif base == "lesion_affect":
        got = [None] * 4
    else:
        got = list(CATS)
    if joy_none and base in ("intact_a", "intact_b"):
        got = [None if c == "joy" else c for c in got]
    return [None] + got


def _counts(rng, cat):
    z = {c: 0 for c in CATS}
    gate, readout, slot = dict(z), dict(z), dict(z)
    g, r, sl = int(rng.integers(40, 80)), int(rng.integers(60, 90)), int(rng.integers(70, 100))   # always drawn: the content
    if cat:                                           # stream must not depend on whether the organ latched (ON vs OFF)
        gate[cat], readout[cat], slot[cat] = g, r, sl
    return gate, readout, slot


def _affect(rng, d, lesion):
    return {"differential": d, "appraisal_valence": round(float(rng.uniform(-0.6, 0.6)), 6), "lesioned": bool(lesion),
            "appraisal_hits": [["happy", 1.0]] if rng.random() < 0.3 else [], "tone_level": int(rng.integers(0, 3)),
            "forthcomingness": {"max_elaborations": int(rng.integers(2, 4)), "max_sentences": 4}}


def engagement(on, rng, oor_session_off=True):
    if on:
        return {"bounded_flag": True, "bounded_env": "1", "bounded_readback": True, "oor_count": 0, "oor_count_session": 0}
    k = int(rng.integers(1, 6)) if (oor_session_off and rng.random() < 0.15) else 0     # the unbounded organ's excursions
    return {"bounded_flag": False, "bounded_env": None, "bounded_readback": False, "oor_count": min(k, 1),
            "oor_count_session": k}


def build_shard(master, seed, arm, on, joy_none=False, nested_keys=False):
    """One arm shard. The content stream depends on (master, seed, arm's content variant) only, so ON and OFF builds of the
    same arm share every reply field except the engagement keys and (when `joy_none` and OFF) the unnamed channel."""
    base = base_of(arm)
    variant = "intact" if base in ("intact_a", "intact_b") else base
    rng = C._sub(master, variant, seed)
    erng = C._sub(master, "engage-" + variant + ("-on" if on else "-off"), seed)
    named = _named(base, joy_none and not on)
    lesion = base == "lesion_affect"
    resp, k = {}, 0
    for label, _msg, role in G.turns_for("emo", seed, arm):
        if role == "feel":
            cat = named[k]
            d = 0.0 if (lesion or k == 0) else round(float((1 if (k == 1) != (base == "permuted") and k in (1, 4) else -1)
                                                           * rng.uniform(0.02, 0.08)), 6)
            k += 1
            sign = "positive" if d > G.MOOD_TOL else ("negative" if d < -G.MOOD_TOL else "neutral")
            ar = ["low", "moderate", "high"][int(rng.integers(0, 3))] if d else "low"
            text = ("My affect monitor reads %s right now (valence differential %+.3f, %s arousal) -- that's a functional "
                    "read of my mood state, not a felt experience." % (sign, d, ar)) + E.feel_clause(cat)
            gate, readout, slot = _counts(rng, cat)
            ec = {"category": cat, "co_commit": False, "commit_lesioned": False, "error": None, "gate_counts": gate,
                  "latch_lesioned": False, "latched": [cat] if cat else [], "lesioned": False, "op": 0,
                  "read_invariant": True, "read_variants": {"x0.5": cat, "x2": cat}, "readout_counts": readout,
                  "route": "emotion_category", "seed": seed, "shuffle": False, "slot_counts": slot, "teach_swap": False,
                  "threshold": 18.0}
            ec.update(engagement(on, erng))
            resp[label] = C.R(text, brain="tiny-demo", inner_state_readout=True, affect=_affect(rng, d, lesion),
                              emotion_category=ec, renderer="template-stub (GPU-free)", rich=False, verified=True)
        else:
            subj = SUBJECTS[int(rng.integers(0, len(SUBJECTS)))]
            r = C.absent(rng, subj, ask=bool(rng.random() < 0.8), realistic=True)
            r.update({"affect": _affect(rng, round(float(rng.uniform(-0.05, 0.05)), 6), lesion),
                      "supporting_facts": [["dog", "chase", "cat"], ["cat", "eat", "fish"]][: int(rng.integers(0, 3))],
                      "verified": True, "n_sentences": int(rng.integers(1, 3))})
            if nested_keys:                       # a handler block that also carries the engagement keys (any-depth drop)
                r["emotion_context"] = dict({"note": "x"}, **engagement(on, erng))
            resp[label] = r
    return {"seed": seed, "group": "emo", "arm": arm, "responses": resp, "sources": [C.S], "pythonhashseed_ok": True,
            "corpus_present": True, "builtin_facts": C.facts_for("emo", base == "permuted"),
            "turns": [{"label": l, "message": m, "role": r} for l, m, r in G.turns_for("emo", seed, arm)],
            "process": {"peak_rss_mb": 8100.0, "ltm_bundle": "wikidata_100k"}, "timings_s": {"feel_base": float(master % 7)}}


def make_tree(master, on, joy_none_seeds=(), nested_keys=False):
    """{seed: {arm: shard}} for the five arms (intact_b = intact_a's content, the repeat = the lesion arm's)."""
    return {s: {a: build_shard(master, s, a, on, joy_none=(s in joy_none_seeds), nested_keys=nested_keys) for a in ARMS}
            for s in SEEDS}


def draw_pair(rng, nested_keys=False, joy_none_p=0.2):
    """A perfect brain across the flip: (ON tree, OFF tree). Some seeds' OFF organ fails to latch joy (the fixed bug)."""
    master = int(rng.integers(0, 2 ** 31 - 1))
    jn = tuple(s for s in SEEDS if rng.random() < joy_none_p)
    return make_tree(master, True, nested_keys=nested_keys), make_tree(master, False, joy_none_seeds=jn, nested_keys=nested_keys)


# ── defects applied to a perfect tree at one random place ────────────────────────────────────────────────────────────
def pick(rng, data):
    s = SEEDS[int(rng.integers(0, len(SEEDS)))]
    a = ARMS[int(rng.integers(0, len(ARMS)))]
    return s, a


def feel_labels(shard):
    return [l for l in shard["responses"] if l.startswith("feel_")]


def expose_labels(shard):
    return [l for l in shard["responses"] if not l.startswith("feel_")]


def choice(rng, xs):
    return xs[int(rng.integers(0, len(xs)))]
