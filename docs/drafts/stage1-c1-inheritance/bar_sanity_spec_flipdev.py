"""bar-sanity spec: the C1 flip DEV bar (docs/drafts/stage1-c1-inheritance/PLAN.md step B; HANDOFF.md).

THE BAR. research/runners/c1_flip_dev_check.py: per dev seed 1-8, flag-on vs flag-off through the real handler on the
Stage-1 groups cat_disc, cat_told, word, lived, val_novel. A seed passes iff cat_disc flag-on names `worm` on the direct
probe AND reply content (abstained / recalled_svo / hypothesis_svo) is equal on every other turn, with the only
allowed change on cat_disc / cat_told `p_canary_eat` being a FLAGGED guess naming worm where flag-off abstained. Dev
bar PASS iff 8/8 seeds pass (UNDEFINED if any seed is invalid or the flag did not engage).

THE DECISION IS THE REAL ONE. Each synthetic brain is written as shard files in the runner's own output format to a
temporary directory; `c1_flip_dev_check.score` loads, validates and scores them and `decide` reads its verdict.

BRAINS (per seed; noise = the answer TEXT differs on random turns, which the bar must ignore because the discourse
preamble of later turns depends on what earlier turns said).
  perfect            every seed: the canary guess `worm` on cat_disc, nothing else changes; text noise.
  perfect_told       as perfect, and on a random half of the seeds cat_told's canary probe also guesses worm.
  null               no inheritance on any seed (flag-on equals flag-off).
  null_one_leak      perfect on 7 seeds; on one random seed flag-on also guesses on a random moat (word / lived).
  null_one_lost      perfect on 7 seeds; on one random seed flag-on abstains where flag-off answered (lived recall).
  null_as_fact       every seed states worm as a RECALLED fact, not a flagged guess (the honesty boundary).
  null_wrong         on one random seed the canary guesses bug.
  random             each seed draws a kind uniformly from the above.
"""
import os
import sys
import tempfile
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import c1_flip_dev_check as F  # noqa: E402

NAME = "c1-flip-dev"
DESCRIPTION = "C1 category-inheritance flip dev bar: 8/8 dev seeds correct + content-specific vs flag-off"
SCORER_FILES = ["research/runners/c1_flip_dev_check.py", "research/runners/stage1_exit_gate.py"]

_KINDS = ("perfect", "perfect_told", "null", "leak_moat", "answer_lost", "as_fact", "wrong_category")


def _text_noise(cells, rng):
    for g, shard in cells.items():
        for lab, r in shard["responses"].items():
            if rng.random() < 0.2:
                r["answer"] = "On %s, then -- %s" % (lab, r.get("answer"))
    return cells


def _seed_cells(kind, rng):
    on, off = F.synthetic_seed(kind)
    return _text_noise(on, rng), _text_noise(off, rng)


def decide(data):
    with tempfile.TemporaryDirectory() as td:
        for seed, (on, off) in data.items():
            for side, cells in (("on", on), ("off", off)):
                for g, shard in cells.items():
                    p = F.shard_path(td, seed, side, g)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as f:
                        json.dump(shard, f)
        return F.score(td, F.DEV_SEEDS)["verdict"] == "PASS"


def _brain(kinds_by_seed, rng):
    return {s: _seed_cells(kinds_by_seed[s], rng) for s in F.DEV_SEEDS}


def perfect(rng):
    return _brain({s: "perfect" for s in F.DEV_SEEDS}, rng)


def perfect_told(rng):
    return _brain({s: ("perfect_told" if rng.random() < 0.5 else "perfect") for s in F.DEV_SEEDS}, rng)


def null(rng):
    return _brain({s: "null" for s in F.DEV_SEEDS}, rng)


def _one_bad(rng, bad):
    k = {s: "perfect" for s in F.DEV_SEEDS}
    k[int(rng.choice(F.DEV_SEEDS))] = bad
    return k


def null_one_leak(rng):
    return _brain(_one_bad(rng, "leak_moat"), rng)


def null_one_lost(rng):
    return _brain(_one_bad(rng, "answer_lost"), rng)


def null_as_fact(rng):
    return _brain({s: "as_fact" for s in F.DEV_SEEDS}, rng)


def null_wrong(rng):
    return _brain(_one_bad(rng, "wrong_category"), rng)


def random(rng):
    return _brain({s: _KINDS[int(rng.integers(len(_KINDS)))] for s in F.DEV_SEEDS}, rng)
