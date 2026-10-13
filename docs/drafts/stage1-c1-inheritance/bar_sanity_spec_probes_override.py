"""bar-sanity spec: the C1 OVERRIDE probe dev bar (RULING c1-moat-ruled-1013; PROBES-PLAN.md section 3).

THE BAR. research/runners/c1_generalization_probes.py, decided by its real `score_shards(...)["override"]`: per dev
seed, after `the bat swims in the pond` the direct organ read of (bat, swim) decides pond (O1) and the reply names
pond and never lake (O2); the same for the eel and river when the eel's pre-override guess reached lake (O3, O4;
otherwise vacuous). PASS iff all 8 dev seeds are defined and pass.

BRAINS (per seed; `synthetic_shard` profiles: the guess lake 0.03-0.08 before, the told object 0.15-0.4 after).
  perfect                 the told fact wins on every seed (bat and eel).
  perfect_eel_vacuous     as perfect, and on a random half of the seeds the eel never guessed (vacuous, not failed).
  null_template           the told fact never wins: lake stays the organ's pick on every seed.
  null_one_template       perfect on 7 seeds; one seed keeps lake.
  null_one_tie            perfect on 7 seeds; on one the told object and lake tie (the organ abstains).
  null_one_reply_lake     perfect on 7 seeds; on one the reply also names lake.
  null_one_eel_template   perfect on 7 seeds; on one the eel keeps lake after the told river.
  random                  each seed draws a kind uniformly from the above.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import c1_generalization_probes as P  # noqa: E402

NAME = "c1-probe-override"
DESCRIPTION = "C1 override probe (dev 1-8): the told specific fact beats the inherited lake guess, organ and reply"
SCORER_FILES = ["research/runners/c1_generalization_probes.py"]
_KINDS = ("perfect", "template", "override_tie", "reply_lake", "eel_template", "eel_vacuous")


def _brain(rng, base="perfect", one=None, half=None):
    seeds = list(P.DEV_SEEDS)
    bad = seeds[int(rng.integers(len(seeds)))] if one else None
    out = {}
    for s in seeds:
        kind = one if s == bad else base
        if half and rng.random() < 0.5:
            kind = half
        out[s] = P.synthetic_seed(kind, rng)
    return out


def decide(data):
    return P.score_shards(data)["override"]["verdict"] == "PASS"


def perfect(rng):
    return _brain(rng)


def perfect_eel_vacuous(rng):
    return _brain(rng, half="eel_vacuous")


def null(rng):
    return _brain(rng, base="template")


def null_one_template(rng):
    return _brain(rng, one="template")


def null_one_tie(rng):
    return _brain(rng, one="override_tie")


def null_one_reply_lake(rng):
    return _brain(rng, one="reply_lake")


def null_one_eel_template(rng):
    return _brain(rng, one="eel_template")


def random(rng):
    return {s: P.synthetic_seed(_KINDS[int(rng.integers(len(_KINDS)))], rng) for s in P.DEV_SEEDS}
