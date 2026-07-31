"""gap#4: does DEEP CREDIT learn when the forward is REPRESENTABLE? THE experiment, with the deep arms in.

WHY THIS FILE EXISTS, given `_gap4_credit_on_expanded_forward_derisk.py` already ran. That file answered a
NARROWER question and says so in its own docstring (its "SCOPE OF THE LEARNER" block): its learner is `fit_lin`,
a ONE-LAYER softmax regression trained by its own exact gradient. It is neither DEEP nor the bio-plausible
credit rule whose POWERED NO-GO this arc exists to re-open. Its 6-seed result --

    raw 0.2778 -> expanded 0.5864, shuffle 0.2901, no-credit 0.3333, lesion 0.3333, memctrl-leak 0.0000,
    codon reproducibility 1.0000, 6/6 seeds expanded > raw AND > the 0.333 majority floor
    (research/findings/raw/gap4/credit_on_expanded_6seed.json)

-- is a FORWARD-REPRESENTABILITY result: a linear read-out decodes held-out inheritance better from the
plateau-expanded forward. It is NOT evidence that deep credit learns. This file swaps the deep arms in, which
is what that docstring says must happen before any deep-credit claim.

THE ONE VARIABLE: the forward representation. Same task, same arms, same depth, same seeds, same budget.
  arm `raw`      -- the task features straight in (the frozen forward; the banked POWERED NO-GO's condition)
  arm `expanded` -- the SAME features through PlateauExpander first (the 2026-07-25 surpassed forward)

Everything else is imported unmodified from `_semantic_inheritance_deep_credit_derisk`: the task, the
depth-genuineness Stage 0, and the Stage-1 arms (microcircuit / KP-learned / plain-FA) with their full
anti-cheat set (wrong-sign alignment, permuted-label, apical lesion, 1-hidden floor, no-weight-transport,
same-init-as-oracle, memorization control). No validated code is edited. The task knobs and hyperparameters
are that runner's OWN DEFAULTS, deliberately: Stage 0's `depth_separating` criterion was calibrated there, and
changing the task to suit this probe would move a second variable inside a one-variable experiment.

⛔ THE GUARD THAT DECIDES WHETHER THIS EXPERIMENT MEANS ANYTHING -- and the reason the fit_lin result cannot be
read as a deep-credit win. That result IS the warning: a LINEAR read-out went from 0.2778 to 0.5864 on the
expanded forward. If the expansion has made the task substantially linearly solvable, then on that forward the
task NO LONGER REQUIRES DEPTH -- and a deep net "learning" a shallow task says nothing whatever about deep
credit. So Stage 0 (the depth-genuineness oracle: 0/1/2/3-hidden fenced backprop) is re-run ON EACH FORWARD,
and if the expanded forward reports `depth_separating=False` the deep-credit verdict is **UNDEFINED**, not a
GO -- the expansion solved the task instead of enabling credit. This is the wall-reframe applied to our own
instrument: the depth requirement is a property of the TASK-PLUS-FORWARD, and it was silently assumed constant
while the forward was the thing being changed.

GO GATE (pre-registered, 6 seeds). Every clause must hold:
  1. the expanded forward is still DEPTH-SEPARATING (Stage 0) -- else UNDEFINED, above
  2. the best deep arm on `expanded` beats the best deep arm on `raw`
  3. that arm also beats the 1-hidden floor ON `expanded` (depth is used, not bypassed)
  4. it beats the majority-class rate on the scored split
  5. permuted-label at/below the floor, apical-lesion below the arm, no-weight-transport TRUE,
     deepest-layer alignment > 0
KILL CRITERION: if the deep arms do NOT improve on the expanded forward while Stage 0 still says the task is
depth-required, then representability was never what blocked credit; the tonic-pinned diagnosis is incomplete
and the next lever is the graded-state escape (lever (b) of the 2026-07-24 root-cause finding). Record it; do
not retune.

ANTI-CHEATS BEYOND THE INHERITED SET, each earned by a specific retraction in this project's record:
  * expander-lesion (all-zero codon of the SAME width) -- proves any lift comes from the expansion, not from a
    wider input layer. Without it "200 features beat 40 features" reads as a credit result.
  * the lever must ENGAGE: raw and expanded accuracies asserted to differ (failure class 1, 10 incidents;
    `_emerge6` has three arms agreeing to sixteen digits).
  * instrument verification BEFORE any verdict: the codon must be reproducible across repeat reads and must
    VARY across rows. A dead or unreliable codon makes this a measurement of noise.
  * a floor of max(1/k, majority-class rate) -- the no-credit and lesion arms reach the majority rate with no
    learning at all, so 1/k alone is not the floor. It can only ever BLOCK a verdict, never manufacture one.

NOTE ON THE IMPORT: `_gap4_plateau_expander_probe` runs its full 6-seed experiment at MODULE level, so
importing `PlateauExpander` from it executes that experiment and prints its results. Its stdout is captured
here so this runner's own output stays readable. That file is not ours to edit.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import logging                                               # noqa: E402
import numpy as np                                           # noqa: E402
logging.disable(logging.INFO)

from tools.lab import lever                                  # noqa: E402

_probe_banner = io.StringIO()
with contextlib.redirect_stdout(_probe_banner):              # the probe RUNS on import; keep its output out of ours
    from research.runners._gap4_plateau_expander_probe import (          # noqa: E402
        PlateauExpander, topk_active, TOPK as PROBE_TOPK)
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, stage1_deep_credit)

# The deep-credit runner's OWN defaults. Changing these moves a second variable (see the docstring).
TASK_KWARGS = dict(n_super=24, n_members=8, held_per_super=3, n_prop=3,
                   member_id_dim=3, n_obs=14, noise=0.02, feature_seed=0)
HP = dict(hidden=96, epochs=250, lr=0.3, batch=128, deep_layers=2)
DEEP_ARMS = ("test_fixed", "test_learned", "plain_fa")


def _expand(exp, X, topk):
    """Feature vector -> plateau codon, reading active features exactly as the validated probe does
    (`topk_active`: a FIXED active-count, so the codon is not also encoding how many features cleared a
    threshold -- the third defect the fit_lin runner had to fix)."""
    return np.asarray([exp.codon(a) for a in topk_active(X, topk)], dtype=np.float64)


def _expand_graded(exp, X, topk):
    """The SAME expansion read GRADED instead of thresholded: `codon_graded` returns the standardized
    continuous apical voltage where `codon` returns `(v_apical > FLOOR)`.

    WHY THIS ARM EXISTS (added after the seed-42 smoke, which is the whole reason to run a smoke). The
    thresholded codon raised the LINEAR floor 0.204 -> 0.611 and simultaneously collapsed the 2-hidden
    ORACLE ceiling 1.000 -> 0.667, leaving every depth at ~0.6-0.7: the expansion bought decodability by
    DESTROYING the structure depth needs. `FLOOR` is a CONSTANT we substituted for a graded biological
    quantity, and the wall-reframe says to suspect the substituted constant before the mechanism. This arm
    holds the expansion fixed and removes only the binarization, so it separates 'the expansion is lossy'
    from 'the THRESHOLD is lossy'. They imply completely different next levers."""
    return np.asarray([exp.codon_graded(a) for a in topk_active(X, topk)], dtype=np.float64)


def _codon_reproducibility(exp, X, topk, n=8):
    """Read the same rows TWICE; report the fraction identical. The expander exists because a rate-settle read
    collapsed to reproducibility 0.07 -- a verdict off an unreliable codon is a verdict about noise."""
    acts = topk_active(X[:n], topk)
    a = np.asarray([exp.codon(s) for s in acts])
    b = np.asarray([exp.codon(s) for s in acts])
    return float(np.mean([float((a[i] == b[i]).all()) for i in range(len(acts))]))


def _best_deep(s1):
    """Best held-out-INHERITANCE accuracy across the deep arms, with the arm that produced it."""
    vals = [(s1[a]["inherit_heldout"], a) for a in DEEP_ARMS if a in s1]
    vals = [(v, a) for v, a in vals if v == v]                          # drop NaN
    return max(vals) if vals else (float("nan"), None)


def _forward_block(name, task, idx, k, seed):
    """Stage 0 (depth genuineness) + Stage 1 (deep-credit arms) on ONE forward representation."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                               # bridge/oracle chatter, not our output
        s0 = stage0_depth_genuineness(task, idx, k, HP["hidden"], HP["epochs"], HP["lr"], HP["batch"], seed)
        s1 = stage1_deep_credit(task, idx, k, HP["hidden"], HP["epochs"], HP["lr"], HP["batch"], seed,
                                rule="microcircuit", feedback="fixed", deep_layers=HP["deep_layers"])
    best, arm = _best_deep(s1)
    return {"forward": name, "stage0": s0, "stage1": s1, "best_deep": best, "best_deep_arm": arm}


def run_seed(seed, n_col, topk):
    task_full = make_task_semantic_inheritance(seed, **TASK_KWARGS)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = int(meta["k_classes"])
    inh = idx["inh_idx"]
    yh = yte[inh]
    majority = float(np.bincount(yh, minlength=k).max() / len(yh))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                               # bridge init prints to stdout
        exp = PlateauExpander(Xtr.shape[1], n_col, seed)
        Etr, Ete = _expand(exp, Xtr, topk), _expand(exp, Xte, topk)
        Gtr, Gte = _expand_graded(exp, Xtr, topk), _expand_graded(exp, Xte, topk)
        repro = _codon_reproducibility(exp, Xtr, topk)

    raw_task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    exp_task = ((Etr, ytr, Ltr), (Ete, yte, Lte))
    grd_task = ((Gtr, ytr, Ltr), (Gte, yte, Lte))       # same expansion, threshold removed
    # expander LESION: the same width, carrying nothing. Separates "the expansion" from "a wider input layer".
    les_task = ((np.zeros_like(Etr), ytr, Ltr), (np.zeros_like(Ete), yte, Lte))

    out = {"seed": seed, "k": k, "n_train": int(Xtr.shape[0]), "n_scored_inherit": int(len(inh)),
           "n_features_raw": int(Xtr.shape[1]), "n_features_expanded": int(Etr.shape[1]),
           "majority_class_rate": round(majority, 4),
           "codon_reproducibility": round(repro, 4),
           "codon_row_variance": round(float(Etr.std(0).mean()), 6),
           "codon_sparsity": round(float(Etr.mean()), 4)}
    for name, t in (("raw", raw_task), ("expanded", exp_task), ("expanded_graded", grd_task),
                    ("expander_lesion", les_task)):
        out[name] = _forward_block(name, t, idx, k, seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-col", type=int, default=200, help="expander columns (the probe's N_COL)")
    ap.add_argument("--topk", type=int, default=PROBE_TOPK, help="active features per row (the probe's TOPK)")
    ap.add_argument("--out", default="research/findings/raw/gap4/deep_credit_on_expanded.json")
    a = ap.parse_args()

    rows = [run_seed(s, a.n_col, a.topk) for s in a.seeds]
    m = lambda f: float(np.mean([r[f] for r in rows]))                              # noqa: E731
    g = lambda fwd, *path: float(np.mean([_dig(r[fwd], path) for r in rows]))       # noqa: E731

    def _dig(d, path):
        for p in path:
            d = d[p]
        return float(d)

    raw_best, exp_best, grd_best, les_best = (
        g(f, "best_deep") for f in ("raw", "expanded", "expanded_graded", "expander_lesion"))
    sep = {f: float(np.mean([r[f]["stage0"]["depth_separating"] for r in rows]))
           for f in ("raw", "expanded", "expanded_graded")}
    raw_sep, exp_sep, grd_sep = sep["raw"], sep["expanded"], sep["expanded_graded"]
    # THE VERDICT MUST BE READ OFF A FORWARD THAT IS STILL DEPTH-REQUIRED. A variant whose Stage 0 says the
    # task went shallow cannot carry a DEEP-credit verdict at all, however well its arms score.
    cands = [(g(f, "best_deep"), f) for f in ("expanded", "expanded_graded") if sep[f] >= 1.0]
    best_val, best_fwd = max(cands) if cands else (float("nan"), None)
    majority, chance = m("majority_class_rate"), 1.0 / rows[0]["k"]
    floor = max(chance, majority)

    print("  %-26s %s" % ("seeds", a.seeds))
    print("  %-26s n_train=%d  scored=%d held-out-INHERITANCE rows  raw_feats=%d  expanded_feats=%d"
          % ("split", rows[0]["n_train"], rows[0]["n_scored_inherit"],
             rows[0]["n_features_raw"], rows[0]["n_features_expanded"]))
    print("  --- STAGE 0: is the task still DEPTH-REQUIRED on each forward? ---")
    for f in ("raw", "expanded", "expanded_graded"):
        print("  %-26s linear=%.3f l1=%.3f l2=%.3f deep_best=%.3f gap=%+.3f  depth_separating=%.2f of 1"
              % (f, g(f, "stage0", "linear_inherit_heldout"), g(f, "stage0", "l1_inherit_heldout"),
                 g(f, "stage0", "l2_inherit_heldout"),
                 g(f, "stage0", "deep_best_inherit_heldout"), g(f, "stage0", "depth_gap"), sep[f]))
    print("  --- STAGE 1: the DEEP-CREDIT arms (held-out inheritance) ---")
    for f in ("raw", "expanded", "expanded_graded", "expander_lesion"):
        print("  %-26s %s" % (f, "  ".join("%s=%.3f" % (arm.replace("test_", ""), g(f, "stage1", arm,
                                                                                   "inherit_heldout"))
                                           for arm in DEEP_ARMS)))
    for key, path in (("single-hidden floor", ("stage1", "single_layer", "inherit_heldout")),
                      ("permuted-label ctrl", ("stage1", "permuted", "inherit_heldout")),
                      ("apical-lesion ctrl", ("stage1", "apical_lesion", "inherit_heldout")),
                      ("oracle ceiling", ("stage1", "oracle", "inherit_heldout")),
                      ("memctrl leak", ("stage1", "test_fixed", "memctrl_heldout")),
                      ("deepest alignment", ("stage1", "test_fixed", "deepest_layer_alignment"))):
        print("  %-26s raw=%.4f  expanded=%.4f  graded=%.4f"
              % (key, g("raw", *path), g("expanded", *path), g("expanded_graded", *path)))
    for key in ("codon_reproducibility", "codon_row_variance", "codon_sparsity"):
        print("  %-26s %.4f" % (key, m(key)))

    lever("forward representation (raw -> thresholded codon), DEEP arms", raw_best, exp_best, required=False)
    lever("threshold removed (thresholded -> graded codon), DEEP arms", exp_best, grd_best, required=False)
    identical = abs(raw_best - exp_best) < 1e-9 and abs(raw_best - grd_best) < 1e-9
    dead_expander = m("codon_row_variance") <= 0.0 or m("codon_reproducibility") < 1.0
    below_floor = max(raw_best, exp_best, grd_best) < floor
    # BOTH expanded variants must lose depth-separation before the verdict is blocked on that ground: if the
    # GRADED read keeps the task depth-required, the verdict is available and is read off THAT forward.
    shallow_after_expansion = best_fwd is None
    fwd = best_fwd or "expanded"                                  # what the remaining clauses are read from
    nwt = all(r[fwd]["stage1"][arm]["no_weight_transport"] for r in rows for arm in DEEP_ARMS)
    align_ok = g(fwd, "stage1", "test_fixed", "deepest_layer_alignment") > 0.0
    beats_floor_arm = best_val > g(fwd, "stage1", "single_layer", "inherit_heldout")
    perm_ok = g(fwd, "stage1", "permuted", "inherit_heldout") <= floor + 1e-9
    lesion_ok = g(fwd, "stage1", "apical_lesion", "inherit_heldout") < best_val
    undefined = dead_expander or below_floor or identical or shallow_after_expansion
    go = (not undefined and best_val > raw_best and beats_floor_arm and best_val > floor
          and nwt and align_ok and perm_ok and lesion_ok)

    print("-" * 100)
    if dead_expander:
        print("  ⛔ EXPANDER INSTRUMENT FAILED (row-variance %.6f, reproducibility %.3f). Verdict UNDEFINED:"
              % (m("codon_row_variance"), m("codon_reproducibility")))
        print("     this measures the instrument, not the hypothesis.")
    elif below_floor:
        print("  ⛔ EVERY DEEP ARM BELOW THE FLOOR (%.3f) — task/label wiring broken. UNDEFINED, NOT a NO-GO."
              % floor)
    elif identical:
        print("  ⛔ ARMS IDENTICAL — the expansion never engaged. UNDEFINED, not negative.")
    elif shallow_after_expansion:
        print("  ⛔ NEITHER EXPANDED FORWARD IS STILL DEPTH-SEPARATING (thresholded %.2f, graded %.2f; raw %.2f)."
              % (exp_sep, grd_sep, raw_sep))
        print("     The expansion SOLVED the task rather than enabling credit, so a deep net succeeding on it")
        print("     is not a deep-credit result. DEEP-CREDIT VERDICT UNDEFINED — report as representability.")
    else:
        print("  => %s (read off `%s`)"
              % ("GO: deep credit learns MORE on the expanded forward, on a task Stage 0 still finds "
                 "depth-required — representability WAS the blocker"
                 if go else
                 "⛔ NO-GO: the expanded forward does not help the DEEP arms while the task remains "
                 "depth-required. KILL CRITERION FIRES — representability was not what blocked credit; "
                 "next lever is the GRADED-STATE escape, not more forward work.", fwd))
    print("     floor=%.3f (1/k=%.3f, majority=%.3f) | raw=%.4f | thresholded=%.4f | graded=%.4f | lesion=%.4f"
          % (floor, chance, majority, raw_best, exp_best, grd_best, les_best))
    print("     depth_separating raw=%.2f thresholded=%.2f graded=%.2f | no_weight_transport=%s | align>0=%s"
          % (raw_sep, exp_sep, grd_sep, nwt, align_ok))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"config": {"seeds": a.seeds, "n_col": a.n_col, "topk": a.topk,
                          "task_kwargs": TASK_KWARGS, "hyperparams": HP, "rule": "microcircuit",
                          "feedback": "fixed"},
               "runner": "research/runners/_gap4_deep_credit_on_expanded_forward_derisk.py",
               "one_variable": "forward representation (raw vs plateau-expanded, thresholded vs graded), "
                               "DEEP credit arms",
               "supersedes_scope_of": "research/findings/raw/gap4/credit_on_expanded_6seed.json (fit_lin, "
                                      "one-layer softmax — a forward-representability result, not deep credit)",
               "gate": "best depth-separating expanded forward > raw_best AND > its single-hidden floor AND > "
                       "max(1/k, majority) AND no_weight_transport AND deepest_alignment>0 AND permuted<=floor "
                       "AND apical_lesion<arm. If NEITHER expanded forward is depth-separating the verdict is "
                       "UNDEFINED — a deep net succeeding on a shallow task is not a deep-credit result",
               "kill": "if the deep arms do not improve while Stage 0 still says depth-required, "
                       "representability was not the blocker; go to the graded-state escape (lever b of the "
                       "2026-07-24 root-cause finding)",
               "verdict_read_off": best_fwd,
               "per_seed": rows,
               "means": {"raw_best_deep": raw_best, "expanded_best_deep": exp_best,
                         "expanded_graded_best_deep": grd_best, "expander_lesion_best_deep": les_best,
                         "raw_depth_separating": raw_sep, "expanded_depth_separating": exp_sep,
                         "expanded_graded_depth_separating": grd_sep,
                         "majority_class_rate": majority, "chance_1_over_k": chance, "floor": floor,
                         "codon_reproducibility": m("codon_reproducibility"),
                         "codon_row_variance": m("codon_row_variance")},
               "arms_identical": identical, "below_floor": below_floor, "dead_expander": dead_expander,
               "shallow_after_expansion": shallow_after_expansion,
               "no_weight_transport": bool(nwt), "verdict_undefined": bool(undefined),
               "go": bool(go)}, open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
