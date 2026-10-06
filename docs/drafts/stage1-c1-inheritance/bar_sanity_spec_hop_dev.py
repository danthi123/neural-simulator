"""bar-sanity spec: the C1 HOP-ATTENUATION dev bar (organ level, dev seeds 1-8; RULING c1-hop-first).

THE BAR. research/runners/category_inherit_organ_dev.py --hop, decided by its `hop_dev_verdict`: a dev seed passes
iff, with BRAIN_CATEGORY_INHERIT_HOP=1, the C1 canary (`the canary flies in the sky`) still reaches worm on `eat`, the
permuted canary reaches bug, the read lesion abstains with the cut held, the C2/C3 moat shapes abstain, and in BOTH
bridge orders (bat heard before / after the canary) the canary's `swim` abstains AND its `eat` still reaches worm.
PASS iff all 16 dev seeds (1-16) pass; UNDEFINED if a seed is missing or a row ran without the option.

THE DECISION IS THE REAL ONE: `decide` hands synthetic rows in run_seed's format to `hop_dev_verdict`.

BRAINS (per seed; rates are drawn around the measured ranges, and `trout/fly` is free: one-member support is reported,
not barred).
  perfect               the fix as intended on every seed.
  perfect_trout_sky     as perfect, and on a random half of the seeds the trout's `fly` reaches sky (not barred).
  null_no_attenuation   the 09-30 organ as measured (dev 8/8): both bridge orders reach lake on swim, and the
                        canary's eat ties with bug on about half of the bridge reads.
  null_inert_lesion     the read lesion leaves worm standing (the cut does not hold) on every seed.
  null_one_lake         perfect on 7 seeds; one seed reaches lake in one bridge order.
  null_overattenuated   the hop is cut so hard the inheritance itself is gone: C1 canary abstains on every seed.
  null_one_lost_worm    perfect on 7 seeds; on one seed the bridge read of the canary's eat ties (abstains).
  null_moat_leak        perfect on 7 seeds; on one seed a C2/C3 moat shape answers.
  random                each seed draws a kind uniformly from the above.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.runners import category_inherit_organ_dev as D  # noqa: E402

NAME = "c1-hop-dev"
DESCRIPTION = "C1 hop attenuation (organ, dev 1-16): bridge canary swim abstains, canary eat keeps worm, controls hold"
SCORER_FILES = ["research/runners/category_inherit_organ_dev.py"]
SEEDS = D.HOP_DEV_SEEDS
_KINDS = ("perfect", "trout_sky", "no_attenuation", "inert_lesion", "one_lake", "overattenuated", "lost_worm",
          "moat_leak")


def _read(patient, rng, other=None, reason=None):
    win = round(float(rng.uniform(0.04, 0.12)), 4)
    adm = [] if patient is None else [[patient, win]]
    if other:
        adm.append([other, round(float(rng.uniform(0.0, win / 1.3)), 4)])
    return {"patient": patient, "reason": reason or ("reached" if patient else "below_floor"), "admitted": adm,
            "lesion": None, "lesion_held": None}


def _row(seed, kind, rng):
    def bridge(lake=False, tie=False):
        eat = ({"patient": None, "reason": "tie", "admitted": [["worm", 0.1], ["bug", 0.09]]} if tie
               else _read("worm", rng, "bug"))
        return {"canary/eat": eat, "canary/swim": _read("lake" if lake else None, rng),
                "trout/fly": _read("sky" if (kind == "trout_sky" and rng.random() < 0.5) else None, rng)}
    row = {"seed": seed, "hop": True,
           "intact": {"canary/eat": _read(None if kind == "overattenuated" else "worm", rng, "bug")},
           "permuted": {"canary/eat": _read("bug", rng, "worm")},
           "read_lesion_all": {"canary/eat": {"patient": None, "reason": "below_floor", "admitted": [["worm", 0.0]],
                                              "lesion": "all", "lesion_held": True}},
           "stage1_c2_c3_shapes": {q: _read(None, rng) for q in ("otter/chase", "clam/eat", "wolf/chase", "bear/eat")},
           "bridge_bat": bridge(), "bridge_bat_canary_first": bridge()}
    if kind == "no_attenuation":
        row["bridge_bat"] = bridge(lake=True, tie=rng.random() < 0.5)
        row["bridge_bat_canary_first"] = bridge(lake=True, tie=rng.random() < 0.5)
    elif kind == "inert_lesion":
        row["read_lesion_all"]["canary/eat"].update(patient="worm", reason="reached", lesion_held=False)
    elif kind == "one_lake":
        row[("bridge_bat", "bridge_bat_canary_first")[int(rng.integers(2))]] = bridge(lake=True)
    elif kind == "lost_worm":
        row[("bridge_bat", "bridge_bat_canary_first")[int(rng.integers(2))]] = bridge(tie=True)
    elif kind == "moat_leak":
        row["stage1_c2_c3_shapes"]["otter/chase"] = _read("clam", rng)
    return row


def decide(data):
    return D.hop_dev_verdict(data, seeds=SEEDS)["verdict"] == "PASS"


def _brain(kinds, rng):
    return [_row(s, kinds[s], rng) for s in SEEDS]


def _one_bad(rng, bad):
    k = {s: "perfect" for s in SEEDS}
    k[int(rng.choice(SEEDS))] = bad
    return k


def perfect(rng):
    return _brain({s: "perfect" for s in SEEDS}, rng)


def perfect_trout_sky(rng):
    return _brain({s: "trout_sky" for s in SEEDS}, rng)


def null_no_attenuation(rng):
    return _brain({s: "no_attenuation" for s in SEEDS}, rng)


def null_inert_lesion(rng):
    return _brain({s: "inert_lesion" for s in SEEDS}, rng)


def null_one_lake(rng):
    return _brain(_one_bad(rng, "one_lake"), rng)


def null_overattenuated(rng):
    return _brain({s: "overattenuated" for s in SEEDS}, rng)


def null_one_lost_worm(rng):
    return _brain(_one_bad(rng, "lost_worm"), rng)


def null_moat_leak(rng):
    return _brain(_one_bad(rng, "moat_leak"), rng)


def random(rng):
    return _brain({s: _KINDS[int(rng.integers(len(_KINDS)))] for s in SEEDS}, rng)
