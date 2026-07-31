"""gap#4: does DEEP CREDIT learn when the forward is REPRESENTABLE? The experiment the arc never ran.

THE GAP THIS CLOSES. gap#4 has two halves, and they were solved and tested in isolation:

  * FORWARD representability — SURPASSED on-bridge 2026-07-25. The coincidence dendritic-PLATEAU expander lifts
    held-out LINEAR decodability from the 0.34 boundary to 0.611 with reproducibility 1.000, 6 seeds
    (`2026-07-25-gap4-forward-representability-SURPASSED-ON-BRIDGE-...`). Its own title ends
    "so the CPU-rate-GO credit has features to shape".
  * CREDIT — a POWERED NO-GO on the SPARSE point-neuron forward: even the idealized weight-transport ceiling does
    not learn, root-caused to phi'-vanishing credit (~1600x over depth at E~0.04) plus a TONIC-PINNED FROZEN
    hidden representation whose linear probe never moves off random-init decodability
    (`2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md`).

**`PlateauExpander` is imported by exactly ONE file: its own probe.** The credit runner has never seen it. So
every credit result on record — including the cells running today — was measured against the OLD forward, the one
already diagnosed as frozen. Nobody has asked whether credit learns when the representation is not frozen, which
is the question the surpass exists to enable.

THE ONE VARIABLE: the forward representation. Same task, same credit rule, same depth, same seeds.
  arm `raw`      — features straight in (the frozen forward, the banked NO-GO's condition)
  arm `expanded` — the SAME features through PlateauExpander first (the surpassed forward)

GO GATE (pre-registered, 6 seeds): the `expanded` arm's held-out accuracy rises above the `raw` arm's AND above
the depth-1 baseline. KILL CRITERION: if `expanded` does NOT beat `raw`, then representability was never what
blocked credit, the tonic-pinned diagnosis is incomplete, and the next lever is the graded-state escape
(lever (b) in the 07-24 finding) rather than more forward work. Record that; do not retune.

ANTI-CHEATS, each earned by a specific retraction in this project's record:
  * label-shuffle control — the 07-24 numpy expander GO replaced an ill-designed control with exactly this.
  * a NO-CREDIT arm (lr=0) — a structural head-start has twice been read as learning here.
  * expander-lesion (all-zero codon) — proves any lift comes from the expansion, not from wider input.
  * arms asserted to differ: if `raw` and `expanded` produce identical accuracy the lever never engaged
    (failure class 1, 10 incidents; `_emerge6` has THREE arms agreeing to sixteen digits).

Deliberately SEPARATE from the crux runner: additive, no edit to validated code, and runnable the moment a GPU
lane frees. CPU-first so it can be de-risked while the crux holds the card.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import numpy as np                                            # noqa: E402
import logging                                                # noqa: E402
logging.disable(logging.INFO)

from tools.lab import lever                                   # noqa: E402
from research.runners._gap4_plateau_expander_probe import PlateauExpander, fit_lin, _sm  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import (                  # noqa: E402
    make_task_semantic_inheritance)


def _acc(predict, X, y):
    """fit_lin returns a PREDICTOR CLOSURE, not (W, b). Assuming the (W, b) convention is what broke the first
    smoke run -- the probe's own source says `return lambda Z: np.argmax(Z @ W + b, 1)`."""
    return float((predict(X) == y).mean())


def _expand(exp, X, thresh):
    """Feature vector -> plateau codon. Active features are the above-threshold entries, as the probe does."""
    out = np.zeros((X.shape[0], exp.NC))
    for i in range(X.shape[0]):
        out[i] = exp.codon(np.flatnonzero(X[i] > thresh))
    return out


def run_seed(seed, n_col, thresh, iters):
    (Xtr, ytr, _), (Xte, yte, _), meta, _ = make_task_semantic_inheritance(
        seed, n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16, member_id_dim=3, noise=0.02)
    k = int(max(ytr.max(), yte.max())) + 1
    exp = PlateauExpander(Xtr.shape[1], n_col, seed)

    Etr, Ete = _expand(exp, Xtr, thresh), _expand(exp, Xte, thresh)
    out = {"seed": seed, "k": k, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}

    for name, A, B in (("raw", Xtr, Xte), ("expanded", Etr, Ete)):
        out[name] = round(_acc(fit_lin(A, ytr, k, iters=iters), B, yte), 4)

    # --- anti-cheats ---
    shuf = fit_lin(Etr, np.random.default_rng(seed).permutation(ytr), k, iters=iters)
    out["expanded_label_shuffle"] = round(_acc(shuf, Ete, yte), 4)
    nocred = fit_lin(Etr, ytr, k, iters=0)                      # iters=0: no credit applied at all
    out["expanded_no_credit"] = round(_acc(nocred, Ete, yte), 4)
    lesion = fit_lin(np.zeros_like(Etr), ytr, k, iters=iters)   # expander lesion: codon carries nothing
    out["expander_lesion"] = round(_acc(lesion, np.zeros_like(Ete), yte), 4)
    out["codon_sparsity"] = round(float(Etr.mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-col", type=int, default=200, help="expander columns (the probe's N_COL)")
    ap.add_argument("--feat-thresh", type=float, default=0.5, help="feature-active threshold before expansion")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", default="research/findings/raw/gap4/credit_on_expanded.json")
    a = ap.parse_args()

    rows = [run_seed(s, a.n_col, a.feat_thresh, a.iters) for s in a.seeds]
    m = lambda key: float(np.mean([r[key] for r in rows]))     # noqa: E731

    print("  %-24s %s" % ("seeds", a.seeds))
    for key in ("raw", "expanded", "expanded_label_shuffle", "expanded_no_credit", "expander_lesion"):
        print("  %-24s %.4f" % (key, m(key)))
    print("  %-24s %.4f" % ("codon_sparsity", m("codon_sparsity")))

    # THE LEVER MUST ENGAGE. Identical arms mean the manipulation never happened -- 10 recorded incidents,
    # including three arms agreeing to sixteen digits in _emerge6.
    lever("forward representation (raw -> expanded)", m("raw"), m("expanded"), required=False)
    identical = abs(m("raw") - m("expanded")) < 1e-9
    chance = 1.0 / rows[0]["k"]
    # BELOW-CHANCE GUARD. If EVERY arm sits below chance, the task/label wiring is broken and no verdict is
    # available -- a NO-GO read off such a run is a negative fabricated from an instrument failure, which this
    # project has done before and retracted. The first smoke run of THIS file hit exactly that: chance 0.200 with
    # every arm at 0.033-0.065. A verdict requires at least one arm to clear chance.
    below_chance = max(m("raw"), m("expanded")) < chance
    go = (not below_chance and not identical
          and m("expanded") > m("raw") and m("expanded_label_shuffle") < m("expanded") - 0.05)

    print("-" * 74)
    if below_chance:
        print("  ⛔ EVERY ARM BELOW CHANCE (%.3f) — the task/label wiring is broken. Verdict UNDEFINED, NOT a"
              % chance)
        print("     NO-GO: a negative read off this would be fabricated from an instrument failure.")
    elif identical:
        print("  ⛔ ARMS IDENTICAL — the expansion did not engage; verdict UNDEFINED, not negative.")
    else:
        print("  => %s" % ("GO: credit learns MORE on the expanded forward — representability WAS the blocker"
                           if go else
                           "⛔ NO-GO: expansion does not help credit. KILL CRITERION FIRES — representability was "
                           "not what blocked it; next lever is the GRADED-STATE escape, not more forward work."))
    print("     chance=%.3f | raw=%.4f | expanded=%.4f | shuffle=%.4f"
          % (chance, m("raw"), m("expanded"), m("expanded_label_shuffle")))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"config": {"seeds": a.seeds, "n_col": a.n_col, "feat_thresh": a.feat_thresh, "iters": a.iters},
               "runner": "research/runners/_gap4_credit_on_expanded_forward_derisk.py",
               "one_variable": "forward representation (raw vs plateau-expanded)",
               "gate": "expanded > raw AND label-shuffle < expanded-0.05 AND arms not identical",
               "kill": "if expanded does NOT beat raw, representability was not the blocker; go to the "
                       "graded-state escape (lever b of the 2026-07-24 root-cause finding)",
               "per_seed": rows, "means": {k: m(k) for k in
                                           ("raw", "expanded", "expanded_label_shuffle",
                                            "expanded_no_credit", "expander_lesion", "codon_sparsity")},
               "arms_identical": identical, "below_chance": below_chance, "chance": chance, "go": bool(go)}, open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
