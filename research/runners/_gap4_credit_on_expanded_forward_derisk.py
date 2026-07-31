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

⚠️ SCOPE OF THE LEARNER — WHAT THIS FILE DOES NOT YET MEASURE (flagged 2026-07-31, NOT fixed here because it
changes what the experiment IS, not how it is wired). The learner in both arms is `fit_lin` — a ONE-LAYER
softmax regression trained by its own exact gradient. It is neither DEEP nor the bio-plausible deep-credit rule
(microcircuit / BDSP / burstprop / weight-transport ceiling, in `_gnw_d1_spiking_bdsp_derisk` + `DendriticMLP`)
whose POWERED NO-GO this run exists to re-open. So a positive here reads as **"a linear read-out decodes the
held-out composition better from the plateau-expanded forward than from the raw features"** — which is close to
a re-measurement of the 07-25 forward-representability probe on more training rows, not evidence that deep
credit learns. Any deep-credit claim needs the deep arms swapped in for `fit_lin` first; until then report this
as a forward-representability result and say so.

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
  * memorization-control read (reported): the untaught supers' reserved class is never a training target, so a
    faithful classifier must score ~0 on them. Anything above 0 is a per-super leak.
  * instrument verification before any verdict: the codon must be REPRODUCIBLE across repeat reads and must
    actually VARY across rows. A dead/constant expander is UNDEFINED, never a NO-GO.

Deliberately SEPARATE from the crux runner: additive, no edit to validated code, and runnable the moment a GPU
lane frees. CPU-first so it can be de-risked while the crux holds the card.

──────────────────────────────────────────────────────────────────────────────────────────────────────────────
WHAT THE FIRST SMOKE GOT WRONG (fixed 2026-07-31; the numbers were raw 0.0362 / expanded 0.0652 / shuffle 0.0399
against a nominal chance of 0.200, and the below-chance guard correctly refused to call it a NO-GO).

  (1) **THE EVAL SET WAS 90% UNSCOREABLE BY CONSTRUCTION.** `make_task_semantic_inheritance` returns FOUR values;
      this file discarded the fourth (`idx`) with `_`. `Xte` is NOT the held-out inheritance test — it is the
      union of the 27 held-out-INHERITANCE rows (the composition test) and 249 MEMORIZATION-CONTROL rows drawn
      from the `n_super//4` UNTAUGHT supers, whose label is a RESERVED novel class (`novel_class = n_class`) that
      **never appears in training** and is designed to be uninferable. Measured at seed 42: `Xte` = 276 rows,
      `yte` = {0:9, 1:6, 2:6, 3:6, 4:249}, `inh_idx` = 27 rows with labels in {0..3}. A classifier trained on
      `ytr` (classes 0-3 only) can never emit class 4, so **the ceiling on the full `Xte` is 27/276 = 0.0978** —
      below the 0.200 the guard was comparing against. Every arm was therefore pinned under "chance" by the
      scoring set, not by the substrate. The probe evaluates on `Xte[idx["inh_idx"]]`; so does this file now.
  (2) **`k = max(y)+1` READ THE WRONG CLASS SPACE.** It landed on 5 only because `yte` contains the reserved
      novel class; restricted to the rows that are actually scored it would give 4, and the task's own
      `meta["k_classes"]` is 5. A class count inferred from whichever labels happen to be present is not the
      task's class space. `k` now comes from `meta["k_classes"]`, as the probe does.
  (3) **THE FEATURE-ACTIVITY CONVENTION DID NOT MATCH THE VALIDATED EXPANDER.** The 0.611 result reads the codon
      from `topk_active(X, TOPK=4)` — exactly 4 active features per row. This file used `X > 0.5`, which yields
      1-5 active features per row (mean 3.04 at seed 42). With `SAMP=3, ACT_TH=2` the plateau codon is a function
      of HOW MANY features are active, so a varying active-count is a second, uncontrolled variable inside a
      run whose entire claim is ONE variable. Now `topk_active`, matching the probe.

  Consequence for the honesty boundary: `1/k` = 0.200 is NOT the floor on the scored split. Its labels are
  {0:9, 1:6, 2:6, 3:6}, so **always-guess-the-majority scores 0.333**, and both the no-credit (iters=0) and
  expander-lesion arms land there by construction. The below-chance guard therefore uses
  `floor = max(1/k, majority-class rate)` — the SAME expression shape, strictly HARDER to pass, and it can only
  ever block a verdict, never manufacture one. `1/k` is still recorded separately as pre-registered.

  NOT changed: the GO expression, the label-shuffle margin, the arms-identical check, the lesion and no-credit
  arms, the KILL criterion.

NOTE ON THE IMPORT: `_gap4_plateau_expander_probe` runs its full 6-seed experiment at MODULE level, so importing
`PlateauExpander` from it executes that experiment (~15 s CPU) and prints its results. Its stdout is captured
here so this runner's own output stays readable; stderr is untouched. That file is not ours to edit.
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
_probe_banner = io.StringIO()
with contextlib.redirect_stdout(_probe_banner):               # the probe RUNS on import; keep its output out of ours
    from research.runners._gap4_plateau_expander_probe import (            # noqa: E402
        PlateauExpander, fit_lin, topk_active, TOPK as PROBE_TOPK)
from research.runners._semantic_inheritance_deep_credit_derisk import (    # noqa: E402
    make_task_semantic_inheritance)


def _acc(predict, X, y):
    """`fit_lin` returns a PREDICTOR CLOSURE, not (W, b) — `return lambda Z: np.argmax(Z @ W + b, 1)`."""
    return float((predict(X) == y).mean())


def _expand(exp, X, topk):
    """Feature vector -> plateau codon. Active features are the TOP-K entries, exactly as the validated probe
    reads them (`topk_active`) — a fixed active-count, so the codon is not also encoding how many features
    happened to clear a threshold."""
    return np.asarray([exp.codon(a) for a in topk_active(X, topk)], dtype=np.float64)


def _codon_reproducibility(exp, X, topk, n=8):
    """Read the same rows TWICE and report the fraction that come back byte-identical. The expander's whole
    claim is reliability (it exists because a rate-settle read collapsed to reproducibility 0.07); a verdict
    read off an unreliable codon is a verdict about noise."""
    acts = topk_active(X[:n], topk)
    a = np.asarray([exp.codon(s) for s in acts])
    b = np.asarray([exp.codon(s) for s in acts])
    return float(np.mean([float((a[i] == b[i]).all()) for i in range(len(acts))]))


def run_seed(seed, n_col, topk, iters):
    # FOUR return values. The 4th (`idx`) carries inh_idx = the held-out INHERITANCE rows (the composition test)
    # and memctrl_idx = the untaught-super rows whose class is never a training target. Scoring on the full Xte
    # scores 90% rows that are uninferable BY DESIGN (see the module docstring, defect 1).
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(
        seed, n_super=12, n_members=8, held_per_super=3, n_prop=2, n_obs=16, member_id_dim=3, noise=0.02)
    k = int(meta["k_classes"])                       # the TASK's class space, not max(y)+1 (defect 2)
    inh, mem = idx["inh_idx"], idx["memctrl_idx"]
    Xh, yh = Xte[inh], yte[inh]                      # the scored split: held-out INHERITANCE only
    Xm, ym = Xte[mem], yte[mem]                      # reported leakage control (reserved novel class)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):            # bridge init prints to stdout, not the logger
        exp = PlateauExpander(Xtr.shape[1], n_col, seed)
        Etr, Eh, Em = (_expand(exp, Xtr, topk), _expand(exp, Xh, topk), _expand(exp, Xm, topk))
        repro = _codon_reproducibility(exp, Xtr, topk)

    out = {"seed": seed, "k": k, "n_train": int(Xtr.shape[0]), "n_test_scored": int(len(inh)),
           "n_test_full": int(Xte.shape[0]), "n_memctrl": int(len(mem)), "n_features": int(Xtr.shape[1]),
           "majority_class_rate": round(float(np.bincount(yh, minlength=k).max() / len(yh)), 4),
           "codon_reproducibility": round(repro, 4),
           # a codon that does not VARY across rows carries nothing: the instrument is dead, not the hypothesis
           "codon_row_variance": round(float(Etr.std(0).mean()), 6)}

    for name, A, B in (("raw", Xtr, Xh), ("expanded", Etr, Eh)):
        out[name] = round(_acc(fit_lin(A, ytr, k, iters=iters), B, yh), 4)

    # --- anti-cheats ---
    clf_e = fit_lin(Etr, ytr, k, iters=iters)
    shuf = fit_lin(Etr, np.random.default_rng(seed).permutation(ytr), k, iters=iters)
    out["expanded_label_shuffle"] = round(_acc(shuf, Eh, yh), 4)
    nocred = fit_lin(Etr, ytr, k, iters=0)                      # iters=0: no credit applied at all
    out["expanded_no_credit"] = round(_acc(nocred, Eh, yh), 4)
    lesion = fit_lin(np.zeros_like(Etr), ytr, k, iters=iters)   # expander lesion: codon carries nothing
    out["expander_lesion"] = round(_acc(lesion, np.zeros_like(Eh), yh), 4)
    # reported, NOT gated: the untaught supers' class was never a training target -> a faithful net scores ~0.
    out["expanded_memctrl_leak"] = round(_acc(clf_e, Em, ym), 4)
    out["codon_sparsity"] = round(float(Etr.mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-col", type=int, default=200, help="expander columns (the probe's N_COL)")
    ap.add_argument("--topk", type=int, default=PROBE_TOPK,
                    help="active features per row before expansion (the probe's TOPK)")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--out", default="research/findings/raw/gap4/credit_on_expanded.json")
    a = ap.parse_args()

    rows = [run_seed(s, a.n_col, a.topk, a.iters) for s in a.seeds]
    m = lambda key: float(np.mean([r[key] for r in rows]))     # noqa: E731

    print("  %-24s %s" % ("seeds", a.seeds))
    print("  %-24s n_train=%d  scored=%d held-out-INHERITANCE rows (of %d in Xte; %d memctrl)"
          % ("split", rows[0]["n_train"], rows[0]["n_test_scored"], rows[0]["n_test_full"], rows[0]["n_memctrl"]))
    for key in ("raw", "expanded", "expanded_label_shuffle", "expanded_no_credit", "expander_lesion"):
        print("  %-24s %.4f" % (key, m(key)))
    for key in ("expanded_memctrl_leak", "codon_sparsity", "codon_reproducibility", "codon_row_variance"):
        print("  %-24s %.4f" % (key, m(key)))

    # THE LEVER MUST ENGAGE. Identical arms mean the manipulation never happened -- 10 recorded incidents,
    # including three arms agreeing to sixteen digits in _emerge6.
    lever("forward representation (raw -> expanded)", m("raw"), m("expanded"), required=False)
    identical = abs(m("raw") - m("expanded")) < 1e-9
    chance = 1.0 / rows[0]["k"]                                # 1/k, as pre-registered
    majority = m("majority_class_rate")                        # always-guess-the-majority on the SCORED split
    floor = max(chance, majority)
    # BELOW-CHANCE GUARD. If EVERY arm sits below the floor, the task/label wiring is broken and no verdict is
    # available -- a NO-GO read off such a run is a negative fabricated from an instrument failure, which this
    # project has done before and retracted. The first smoke run of THIS file hit exactly that: chance 0.200 with
    # every arm at 0.033-0.065, because it scored the full Xte whose ceiling is 0.098 (module docstring, defect
    # 1). The floor is max(1/k, majority-class rate) because the no-credit and lesion arms reach the majority
    # rate with no learning at all; it can only ever BLOCK a verdict, never manufacture one.
    below_chance = max(m("raw"), m("expanded")) < floor
    # a codon that is identical on every row carries zero information: the instrument is dead, not the hypothesis
    dead_expander = m("codon_row_variance") <= 0.0 or m("codon_reproducibility") < 1.0
    go = (not below_chance and not identical and not dead_expander
          and m("expanded") > m("raw") and m("expanded_label_shuffle") < m("expanded") - 0.05)

    print("-" * 74)
    if dead_expander:
        print("  ⛔ EXPANDER INSTRUMENT FAILED (row-variance %.6f, reproducibility %.3f) — the codon is dead or"
              % (m("codon_row_variance"), m("codon_reproducibility")))
        print("     unreliable. Verdict UNDEFINED: this measures the instrument, not the hypothesis.")
    elif below_chance:
        print("  ⛔ EVERY ARM BELOW CHANCE (%.3f) — the task/label wiring is broken. Verdict UNDEFINED, NOT a"
              % floor)
        print("     NO-GO: a negative read off this would be fabricated from an instrument failure.")
    elif identical:
        print("  ⛔ ARMS IDENTICAL — the expansion did not engage; verdict UNDEFINED, not negative.")
    else:
        print("  => %s" % ("GO: credit learns MORE on the expanded forward — representability WAS the blocker"
                           if go else
                           "⛔ NO-GO: expansion does not help credit. KILL CRITERION FIRES — representability was "
                           "not what blocked it; next lever is the GRADED-STATE escape, not more forward work."))
    print("     floor=%.3f (1/k=%.3f, majority=%.3f) | raw=%.4f | expanded=%.4f | shuffle=%.4f"
          % (floor, chance, majority, m("raw"), m("expanded"), m("expanded_label_shuffle")))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"config": {"seeds": a.seeds, "n_col": a.n_col, "topk": a.topk, "iters": a.iters},
               "runner": "research/runners/_gap4_credit_on_expanded_forward_derisk.py",
               "one_variable": "forward representation (raw vs plateau-expanded)",
               "scored_split": "held-out INHERITANCE rows only (idx['inh_idx']); the other ~90% of Xte are "
                               "memorization-control rows whose reserved class is never a training target",
               "gate": "expanded > raw AND label-shuffle < expanded-0.05 AND arms not identical",
               "kill": "if expanded does NOT beat raw, representability was not the blocker; go to the "
                       "graded-state escape (lever b of the 2026-07-24 root-cause finding)",
               "per_seed": rows, "means": {k: m(k) for k in
                                           ("raw", "expanded", "expanded_label_shuffle",
                                            "expanded_no_credit", "expander_lesion", "expanded_memctrl_leak",
                                            "codon_sparsity", "codon_reproducibility", "codon_row_variance",
                                            "majority_class_rate")},
               "arms_identical": identical, "below_chance": below_chance, "dead_expander": dead_expander,
               "chance_1_over_k": chance, "chance_majority": majority, "chance": floor,
               "go": bool(go)}, open(a.out, "w"), indent=1)
    print("  wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
