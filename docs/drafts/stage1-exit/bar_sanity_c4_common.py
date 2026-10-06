"""Shared builder for the Stage-1 C4 bar-sanity specs (tools/bar_sanity.py).

The decision is the REAL scorer: `research.runners.stage1_exit_gate.aggregate` run over synthetic shards written in
the shard format the workers write (`shard_path`, `responses`, `patch`, `sources`, ...), reading the `val_heldout`
group verdict. Nothing about the bar is reimplemented: the 0.55 intact bar, the lesion clause, the reproduction
clause, the null clause and the "scrambled-map r < 0.20 on every seed" clause are all the scorer's own.

The synthetic brains are made from the REAL learned map (`research/findings/raw/_affect_learned_valence_map.json`,
the DR-2 leave-one-out values of the 24 held-out words) and the REAL norms (`HELDOUT_WORDS`). A brain's reply
differential for a word is a fixed linear function of the value it holds for that word, plus Gaussian read noise:
    perfect (realistic): noise scaled so r(read, held value) is drawn from U(0.93, 0.96), the range attempt 1's
                         scrambled arms read (research/findings/2026-10-04-stage1-exit-gate-RESULT.md);
    perfect_exact:       no read noise (r = 1 up to the 3-decimal print).
In the scrambled arm the brain holds the value the control assigned to that word (the draw), in the intact arm its own.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import stage1_exit_gate as G  # noqa: E402

_MAP = json.load(open(os.path.join(REPO, "research", "findings", "raw", "_affect_learned_valence_map.json")))["map"]
WORDS = [w for w, _n in G.HELDOUT_WORDS]
NORMS = [n for _w, n in G.HELDOUT_WORDS]
VALS = [float(_MAP[w][0]) for w in WORDS]
K = 0.02                                   # reply differential per unit of learned valence above the neutral point 5
SD_A = float(np.std([K * (v - 5.0) for v in VALS]))
N_SEEDS = len(G.GATE_SEEDS)                # 6
SOURCES = [G.EXPECTED_SOURCE]


def derangement_old(n, seed):
    """The registered control BEFORE RULING s1-c4-control (research/runners/stage1_exit_gate.py at 8c56ff97b):
    the first derangement of the fixed RNG default_rng(seed + 7), no constraint on its correlation with the norms."""
    rng = np.random.default_rng(int(seed) + 7)
    while True:
        p = list(rng.permutation(n))
        if all(p[i] != i for i in range(n)):
            return [int(x) for x in p]


def derangement_constrained(n, seed):
    """The registered control AFTER the ruling: the scorer module's own `_scrambled_valence_draw` (|r| <= its
    SCRAMBLE_R_MAX against the human ratings)."""
    perm, _r, _within = G._scrambled_valence_draw(VALS, NORMS, seed)
    return [int(x) for x in perm]


def _answer(d):
    sign = "positive" if d > G.MOOD_TOL else ("negative" if d < -G.MOOD_TOL else "neutral")
    return ("My affect monitor reads %s right now (valence differential %+.3f, moderate arousal) -- that's a "
            "functional read of my mood state, not a felt experience." % (sign, d))


def _responses(labels, reads):
    """{label: reply}: the feel turns print `reads[word]`; every other turn is a fixed neutral reply."""
    out = {}
    for lab in labels:
        if lab.endswith("_feel") and lab[:-5] in reads:
            out[lab] = {"answer": _answer(reads[lab[:-5]]), "abstained": False}
        else:
            out[lab] = {"answer": "(turn %s)" % lab, "abstained": False}
    return out


def _shard(seed, arm, responses, patch=None):
    return {"seed": seed, "arm": arm, "responses": responses, "sources": SOURCES, "pythonhashseed_ok": True,
            "corpus_present": True, "patch": patch}


def _reads(rng, held, rho):
    """Reply differentials for a brain holding the values `held` (word order = WORDS): K*(v-5) + Gaussian read noise
    sized so r(read, held) ~ rho (rho None: exact)."""
    d = np.array([K * (v - 5.0) for v in held])
    if rho is not None:
        d = d + rng.normal(0.0, SD_A * np.sqrt(1.0 / rho ** 2 - 1.0), size=len(d))
    return {w: float(x) for w, x in zip(WORDS, d)}


def make_data(rng, draw, *, rho="realistic", control="follows", lesion="works", intact="follows"):
    """One six-seed run of the C4 primary group.
    draw     : function (n, seed) -> permutation (the control under test)
    rho      : "realistic" (U(0.93, 0.96) per arm), None (exact), or a float
    control  : "follows" (scrambled arm reads the assigned values) | "ignores" (reads its own unscrambled values,
               the inert control of attempt 1)
    lesion   : "works" (every feel read neutral) | "inert" (the lesion arm reads like the intact one)
    intact   : "follows" (own learned values) | "random" (reads independent of the map)"""
    data = {}
    for _ in range(N_SEEDS):
        seed = int(rng.integers(0, 10 ** 6))
        labels = [t[0] for t in G.turns_for("val_heldout", seed, "intact_a")]

        def r_():
            return float(rng.uniform(0.93, 0.96)) if rho == "realistic" else rho

        if intact == "random":
            intact_reads = {w: float(x) for w, x in zip(WORDS, rng.normal(0.0, SD_A, size=len(WORDS)))}
        else:
            intact_reads = _reads(rng, VALS, r_())
        perm = list(draw(len(WORDS), seed))
        held = [VALS[perm[i]] for i in range(len(WORDS))] if control == "follows" else VALS
        scr_reads = (_reads(rng, held, r_()) if intact != "random" else
                     {w: float(x) for w, x in zip(WORDS, rng.normal(0.0, SD_A, size=len(WORDS)))})
        les_reads = ({w: 0.0 for w in WORDS} if lesion == "works" else intact_reads)
        ra = _responses(labels, intact_reads)
        rl = _responses(labels, les_reads)
        data[seed] = {
            "intact_a": _shard(seed, "intact_a", ra), "intact_b": _shard(seed, "intact_b", ra),
            "lesion_affect": _shard(seed, "lesion_affect", rl), "lesion_affect.rep0": _shard(seed, "lesion_affect.rep0", rl),
            "permuted": _shard(seed, "permuted", _responses(labels, scr_reads),
                               {"patch": "permuted_map", "n_words": len(WORDS), "perm": perm}),
        }
    return data


_TMP = None


def decide(data):
    """The real scorer's C4 (val_heldout) verdict over these six seeds' shards is PASS."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="bar_sanity_c4_")
    seeds = sorted(data)
    for seed in seeds:
        for arm, shard in data[seed].items():
            p = G.shard_path(_TMP, seed, "val_heldout", arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump(shard, fh)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):   # attributable_to prints
        agg = G.aggregate(_TMP, seeds=seeds, informational=False)
    return agg["groups"]["val_heldout"]["verdict"] == "PASS"


def cleanup():
    global _TMP
    if _TMP:
        shutil.rmtree(_TMP, ignore_errors=True)
        _TMP = None
