"""bar-sanity spec: s2-c4-build DEV battery bar (organ level), decided by the real code.

decide = `research.runners.emotion_category_dev.battery_verdict` over eight dev-seed records in `run_seed`'s format:
PASS iff every one of the ten cases (intact, permuted, affect/word/latch/commit lesions, shuffle build, teach-swap
build, ladder-only probe, baseline) reads the expected category on all eight dev seeds, with a threshold-invariant read
wherever a category is named.

Brains (each draws eight seeds):
  perfect                       the capability: the category is read off the held ladder by the commit
  null_host_lookup              the category is a lexicon lookup of the last heard affect words (ignores the ladder)
  null_lookup_gated_by_ladder   the lookup, printed only while the ladder holds something (the A3 residual brain)
  null_lesion_inert             the ladder brain, but the affect_out lesion does not reach the read
  null_no_hold                  a brain that names only while the words are present (no held state)
  null_host_argmax              the ladder's readout, decided by a host argmax (no commit: the commit lesion is inert)
  random                        a random category (or none) per read, random invariance
  info_one_flaky_case           the ladder brain with one random case wrong on one seed with p 0.3 (reported)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import emotion_category_dev as D  # noqa: E402

NAME = "s2-c4-build dev battery"
DESCRIPTION = ("Organ-level dev bar for the emotion-category organ: all ten cases right on dev seeds 1-8. Nulls include "
               "the host lexicon lookup and the ladder-gated lookup that the Stage-2 reply-level bar cannot separate.")
SCORER_FILES = ["research/runners/emotion_category_dev.py"]
CATS = D.CATS
LEX = {c: c for c in CATS}           # what a lexicon lookup of a situation's affect words returns


def decide(data):
    return bool(D.battery_verdict(data)["pass"])


def _rec(cat, inv=True):
    return {"category": cat, "read_invariant": bool(inv), "co_commit": False}


def _seed(seed, fn):
    cases = {}
    for case, want in D.EXPECT.items():
        cases[case] = {k: _rec(*fn(case, k, w)) for k, w in want.items()}
    return {"seed": seed, "op": 0, "cases": cases}


def _brain(fn):
    return [_seed(s, fn) for s in D.DEV_SEEDS]


def perfect(rng):
    return _brain(lambda case, k, w: (w, True))


def _lookup(case, k):
    """What a lexicon lookup of the words heard most recently names (None when no affect word was heard)."""
    if case == "ladder_only":
        return None
    if case == "baseline":
        return None if k == "fresh" else "joy"
    if case == "permuted":
        return D.SITUATION_SWAP[k]
    return LEX[k]


def null_host_lookup(rng):
    return _brain(lambda case, k, w: (_lookup(case, k), True))


def null_lookup_gated_by_ladder(rng):
    def fn(case, k, w):
        if case in ("affect_lesion", "word_lesion", "latch_lesion"):
            return (None, True)                 # the ladder holds nothing (or is unread): the gated lookup is silent
        return (_lookup(case, k), True)
    return _brain(fn)


def null_lesion_inert(rng):
    return _brain(lambda case, k, w: ((k if case == "affect_lesion" else w), True))


def null_no_hold(rng):
    def fn(case, k, w):
        if case == "ladder_only":
            return (None, True)
        return (None, True) if w is not None else (w, True)
    return _brain(fn)


def null_host_argmax(rng):
    return _brain(lambda case, k, w: ((k if case == "commit_lesion" else w), True))


def random(rng):
    opts = list(CATS) + [None]
    return _brain(lambda case, k, w: (opts[int(rng.integers(len(opts)))], bool(rng.integers(2))))


def info_one_flaky_case(rng):
    data = perfect(rng)
    if rng.random() < 0.3:
        s = int(rng.integers(len(data)))
        case = sorted(D.EXPECT)[int(rng.integers(len(D.EXPECT)))]
        k = sorted(data[s]["cases"][case])[0]
        w = data[s]["cases"][case][k]["category"]
        data[s]["cases"][case][k]["category"] = ("joy" if w != "joy" else "fear")
    return data
