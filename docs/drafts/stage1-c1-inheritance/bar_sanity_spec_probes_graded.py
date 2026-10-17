"""bar-sanity spec: the C1 GRADED probe dev bar (RULING c1-moat-ruled-1013; PROBES-PLAN.md section 3).

THE BAR. research/runners/c1_generalization_probes.py, decided by its real `score_shards(...)["graded"]`: per dev
seed, the lake rate in the organ trace of `what does the <target> swim` rises strictly with the shared traits, newt
(1) < toad (2) < eel (3), with three told facts per target. PASS iff all 8 dev seeds are defined and pass.

BRAINS (per seed; `synthetic_shard` profiles).
  perfect                 each shared trait adds 0.004-0.03 to the lake rate (newt 0.015-0.03).
  null_flat               one level for every target (the strength does not see the evidence: a template).
  null_count_only         the strength follows the NUMBER of told facts, constant here: one level plus jitter.
  null_inverted           strength falls with the shared traits.
  null_one_top_tie        perfect on 7 seeds; on one, toad and eel tie.
  random                  each seed draws a kind uniformly from the above.
  info_noisy_grading      REPORTED, not thresholded: a graded brain whose slot heterogeneity is comparable to the
                          step (step 0.01 per trait, per-target noise sd 0.004, quantized to 1/600): how often a
                          real but noisy grading passes an every-seed strict bar.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import c1_generalization_probes as P  # noqa: E402

NAME = "c1-probe-graded"
DESCRIPTION = "C1 graded probe (dev 1-8): the inherited guess's rate rises strictly with shared traits 1 < 2 < 3"
SCORER_FILES = ["research/runners/c1_generalization_probes.py"]
_KINDS = ("perfect", "graded_flat", "graded_count_only", "graded_inverted", "graded_top_tie")


def _brain(rng, base="perfect", one=None):
    seeds = list(P.DEV_SEEDS)
    bad = seeds[int(rng.integers(len(seeds)))] if one else None
    return {s: P.synthetic_seed(one if s == bad else base, rng) for s in seeds}


def decide(data):
    return P.score_shards(data)["graded"]["verdict"] == "PASS"


def perfect(rng):
    return _brain(rng)


def null(rng):
    return _brain(rng, base="graded_flat")


def null_count_only(rng):
    return _brain(rng, base="graded_count_only")


def null_inverted(rng):
    return _brain(rng, base="graded_inverted")


def null_one_top_tie(rng):
    return _brain(rng, one="graded_top_tie")


def random(rng):
    return {s: P.synthetic_seed(_KINDS[int(rng.integers(len(_KINDS)))], rng) for s in P.DEV_SEEDS}


def info_noisy_grading(rng):
    data = _brain(rng)
    for s in data:
        g = data[s]["graded"]
        base = float(rng.uniform(0.02, 0.05))
        for a, k in P.SHARED_TRAITS.items():
            r = round(round((base + 0.01 * k + float(rng.normal(0.0, 0.004))) * 600) / 600, 4)
            r = max(r, 0.0)
            for rd in ((g["responses"][P.GRADED_PROBE[a]].get("category_inherit") or {}).get("read"),
                       g["direct_reads"][P.GRADED_PROBE[a]]["%s/swim" % a]):
                rd["admitted"] = [["lake", r]]
    return data
