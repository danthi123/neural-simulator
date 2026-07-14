"""SCALE de-risk for the graded-confidence console (does the 3-level read + moat HOLD as the concept inventory grows?):
14 members across 4 emergent categories (bird/fish/mammal/insect), 4 taught properties (each taught to ONE member of its
category), 2 category-ambiguous members, and a never-observed concept. The mission question: as the number of COMPETING
properties grows, does the graded margin read still (a) generalize CONFIDENTLY to HELD-OUT category members (a property
taught to robin is inherited confidently by eagle/wren), (b) HEDGE on genuinely category-ambiguous members, (c) ABSTAIN
across categories, (d) keep the unknown-concept moat? Reuse-by-import of `GradedExperientialConsole`; NO `sim/` edit.

WHY: the small wire-in (`_emerge_graded_confidence_console_derisk`, 12-seed GO) proved the mechanism on a handful of
concepts. Production open-domain conversation needs it to survive a larger inventory with more competing properties
(more chances for a spurious CONFIDENT / a collapsed HEDGE). This scales members 3->14, properties 1->4.

GO (6-seed standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): HELD-OUT members (eagle/wren/bass/wolf/ant, never
taught the property themselves) inherit CONFIDENTLY+correct via the learned category, AND both ambiguous members HEDGE,
AND cross-category queries ABSTAIN, AND the unknown concept hits the MOAT, with PERMUTED (every member co-occurs equally
with all contexts) destroying the held-out confidence AND LESION (coincidence off) driving all-abstain, on >=5/6 in BOTH
sets.

Run: SIM_BACKEND=numpy python -m research.runners._emerge_graded_confidence_console_scale_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._emerge_graded_confidence_console_derisk import GradedExperientialConsole

# 4 emergent categories (context = the shared co-occurring cue); each member co-occurs with its category context.
CATEGORIES = {
    "nest":  ["robin", "sparrow", "eagle", "wren"],       # bird -> fly
    "river": ["trout", "pike", "bass"],                   # fish -> swim
    "den":   ["mole", "fox", "wolf"],                     # mammal -> walk
    "hive":  ["bee", "ant"],                              # insect -> sting
}
TEACH = [("robin", "fly"), ("trout", "swim"), ("mole", "walk"), ("bee", "sting")]  # ONE member per category
AMBIG = [("bat", ["nest", "den"]), ("penguin", ["nest", "river"])]                 # category-ambiguous
ALL_CTX = list(CATEGORIES.keys())

# probes: HELD-OUT members (property taught to a DIFFERENT member of their category) -> expect CONFIDENT+correct
HELD_OUT = [("eagle", "fly"), ("wren", "fly"), ("bass", "swim"), ("wolf", "walk"), ("ant", "sting")]
AMBIG_Q = [("bat", "fly"), ("penguin", "swim")]                                    # -> expect HEDGED
CROSS_Q = [("trout", "fly"), ("eagle", "sting")]                                   # -> expect ABSTAIN (wrong category)
MOAT_Q = ("griffin", "fly")                                                        # never observed -> MOAT


def _build(seed, epochs, arm):
    c = GradedExperientialConsole(seed=seed, epochs=epochs, capacity=84, coincidence=(arm != "lesion"))
    if arm != "nolearn":
        if arm == "permuted":                                                      # every member <-> ALL contexts
            for members in CATEGORIES.values():
                for m in members:
                    c.observe_mixed(m, ALL_CTX)
            for (m, _ctx) in AMBIG:
                c.observe_mixed(m, ALL_CTX)
        else:
            for ctx, members in CATEGORIES.items():
                for m in members:
                    c.observe(m, ctx)
            for (m, ctxs) in AMBIG:
                c.observe_mixed(m, ctxs)
    for (m, p) in TEACH:
        c.learn_can(m, p)
    return c


def _run_arm(seed, arm, epochs):
    """Each probe asks graded_answer(member, its_true_category_property); we keep only the label."""
    c = _build(seed, epochs, arm)
    return {"arm": arm,
            "held": {m: c.graded_answer(m, p)[0] for (m, p) in HELD_OUT},
            "amb": {m: c.graded_answer(m, p)[0] for (m, p) in AMBIG_Q},
            "cross": {f"{m}_{p}": c.graded_answer(m, p)[0] for (m, p) in CROSS_Q},
            "moat": c.graded_answer(*MOAT_Q)[0]}


def run(seed, epochs):
    htm = _run_arm(seed, "htm", epochs)
    perm = _run_arm(seed, "permuted", epochs)
    les = _run_arm(seed, "lesion", epochs)
    held_confident = all(v == "CONFIDENT" for v in htm["held"].values())        # held-out members inherit confidently
    amb_hedged = all(v == "HEDGED" for v in htm["amb"].values())                # category-ambiguous -> hedge
    cross_abstain = all(v == "ABSTAIN" for v in htm["cross"].values())          # wrong-category property -> abstain
    moat_ok = htm["moat"] == "MOAT"                                             # unknown concept -> moat
    all_levels = held_confident and amb_hedged and cross_abstain and moat_ok
    permuted_breaks = any(v != "CONFIDENT" for v in perm["held"].values())      # held-out confidence must degrade
    lesion_breaks = all(v == "ABSTAIN" for v in les["held"].values())
    go = bool(all_levels and permuted_breaks and lesion_breaks)
    n_conf = sum(v == "CONFIDENT" for v in htm["held"].values())
    print(f"[graded-scale seed={seed}] held_confident={n_conf}/{len(htm['held'])} amb={list(htm['amb'].values())} "
          f"cross={list(htm['cross'].values())} moat={htm['moat']} | perm_breaks={permuted_breaks} "
          f"lesion_breaks={lesion_breaks} -> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "held": htm["held"], "amb": htm["amb"], "cross": htm["cross"],
            "moat": htm["moat"], "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    res = [run(s, a.epochs) for s in a.seeds]
    print(f"[graded-scale] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
