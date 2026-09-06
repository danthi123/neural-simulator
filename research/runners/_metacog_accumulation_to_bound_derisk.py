"""DE-RISK (wall-reframe follow-on to scaffold-retirement backlog rank 9's PARTIAL verdict,
research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md): does an ACCUMULATION-TO-BOUND
confidence read (`RFPhasorComposer._spiking_margin_accum`) out-discriminate the FIXED-ENDPOINT snapshot
(`_spiking_margin`) specifically in the AMBIGUOUS MIDDLE BAND the PARTIAL finding characterized as a residual
(host-margin-agreement 50% there vs 97.6% on unambiguous cases, NOT resolved by a drive/window-size sweep)?

THE WALL-REFRAME HYPOTHESIS (CLAUDE.md's "what companion process did we replace with a constant?"): the PARTIAL's
own sweep only ever changed how LONG the fixed window was, still reading a SINGLE SNAPSHOT at the end of it. The
brain's own recall competition already RUNS a `_cleanup_window`-step Izhikevich WTA deliberation
(`_spiking_margin`'s existing loop) -- the companion process replaced with a constant is the SEQUENTIAL-SAMPLING /
accumulation-to-bound read a real decision-confidence circuit performs over that deliberation (Ratcliff
drift-diffusion; Reddi & Carpenter 2000 LATER; Pleskac & Busemeyer 2010 two-stage DDM for confidence): evidence
integrates over TIME, and both the accumulated TRAJECTORY and the TIME-TO-BOUND carry information a fixed-endpoint
read discards -- especially for a genuinely borderline item, whose signature is a SLOW, low-drift, late-or-never
bounding accumulation.

THIS SCRIPT (CPU/numpy smoke, cost-routed per the mouth-training GPU lock): reuses the validated rank-9 PARTIAL's
own composer/capture machinery UNCHANGED (`build_composer`, `capture_raw_scores`, `_host_mrc`, `FACTS`, `VOCAB` --
reuse-by-import, no duplicated logic / no re-derivation of an already-validated harness), but where the PARTIAL
drew exactly ONE noise realization per sigma level, this runs a BATTERY of independent noise draws per sigma
across a sigma band chosen to straddle the host's own confident/hedge transition (`SIGMAS_AMBIG`, informed by the
PARTIAL's own per-seed crossing table) -- enough trials to get a genuine MIX of correct and incorrect recalls (the
type-2 ground truth), not just a mix of sigma LEVELS.

For every trial, on the SAME captured per-role raw score arrays (one query, one capture -- never a separate
simulation per arm), computes BOTH:
  (a) `snapshot_mrc`  -- the mean over roles of `_spiking_margin_accum(...)['final_margin']`, IDENTICAL in value
      to what `_spiking_margin` itself would return (bit-exact, see tests/test_spiking_margin_accum.py) -- the
      OLD single-snapshot arm.
  (b) `accum_mrc`     -- the mean over roles of `_spiking_margin_accum(...)['mean_trajectory_margin']` -- the NEW
      accumulation-to-bound arm's primary confidence read.
plus `frac_roles_bounded` / `mean_steps_to_bound` (the time-to-bound diagnostic).

VERDICT: per-seed and pooled type-2 AUC (does confidence predict TRUE recall correctness -- `ans == the stored
patient`, not agreement with the host formula, which the PARTIAL already measured) for BOTH arms, split ALL vs.
AMBIGUOUS-band-only (host mrc inside ROLE_CONF_LO/HI) -- the split the PARTIAL's own residual lives in. Uses
`tools.lab.undefined_if_empty` rather than fabricating a score when a class is missing from a small battery.

Usage: python -m research.runners._metacog_accumulation_to_bound_derisk --out <path.json> [--seeds 42 43]
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["BRAIN_METACOG_SPIKING_MARGIN"] = "1"   # populate margin_spiking trace fields -> raw scores reachable

import numpy as np

from research.runners._metacog_spiking_recall_margin_derisk import (
    build_composer, capture_raw_scores, _host_mrc, FACTS, VOCAB,
)
import research.runners.rf_phasor_composer as _rfp
from research.runners.metacog_production_organ import ROLE_CONF_LO, ROLE_CONF_HI
from research.runners._stageA_foundation_honesty_arbiter_derisk import _auc
from research.runners._emergent_graceful_degradation_derisk import _noise
from tools.lab import undefined_if_empty

EXPECTED_PATIENT = "spikes"   # FACTS[0] = ("brain", "use", "spikes"); the smoke query is query_patient("brain","use")
SEEDS_SMOKE = [42, 43]        # the SMALLEST config that tests the direction (CPU smoke; full 6-seed deferred, see
                              # the finding's ready-to-queue GPU command)
# targets the host confidence transition band directly (the PARTIAL finding's own per-seed noise-sweep table:
# seed 42/43 cross into the host's hedge zone around sigma 1.5, seed 100/101 around sigma 2.0-2.5) -- a battery
# spanning this band gives trials whose HOST mrc genuinely lands inside ROLE_CONF_LO/HI, not just a mix of sigma
# labels, and a genuine mix of correct/incorrect recalls (the type-2 ground truth this de-risk needs that the
# PARTIAL's own agreement-rate metric did not).
SIGMAS_AMBIG = [1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5, 3.0]
N_TRIALS_PER_SIGMA = 10


def _spiking_mrc_from_raw(comp, raw_scores, lesion=False):
    """Mean over roles of BOTH the single-snapshot (`final_margin`) and accumulation-to-bound
    (`mean_trajectory_margin`) reads, off the SAME captured per-role raw score arrays -- one call per role, no
    separate simulation per arm (both reads come out of the ONE `_spiking_margin_accum` call)."""
    snaps, accums, bounded_flags, steps = [], [], [], []
    for s in raw_scores:
        r = comp.comp._spiking_margin_accum(s, lesion=lesion)
        snaps.append(r["final_margin"])
        accums.append(r["mean_trajectory_margin"])
        bounded_flags.append(bool(r["bounded"]))
        if r["steps_to_bound"] is not None:
            steps.append(r["steps_to_bound"])
    return {
        "snapshot_mrc": float(np.mean(snaps)) if snaps else None,
        "accum_mrc": float(np.mean(accums)) if accums else None,
        "frac_roles_bounded": float(np.mean(bounded_flags)) if bounded_flags else None,
        "mean_steps_to_bound": float(np.mean(steps)) if steps else None,
    }


def run_seed(seed, sigmas, n_trials):
    comp = build_composer(seed)
    base_conns = list(comp.store_conns)
    trials = []
    trial_idx = 0
    for sigma in sigmas:
        for _rep in range(n_trials):
            # a FRESH, unique rng per trial (independent draws, not one stream re-walked -- genuine repeated
            # sampling of the noise distribution at this sigma, not path-dependent successive perturbations).
            rng = np.random.default_rng(90000 + seed * 1009 + trial_idx)
            noised = _noise(base_conns, sigma, rng)
            ans, trace, raw = capture_raw_scores(comp, noised)
            host = _host_mrc(trace)
            reads = _spiking_mrc_from_raw(comp, raw)
            correct = bool(ans == EXPECTED_PATIENT)
            ambiguous = bool(host is not None and ROLE_CONF_LO < host < ROLE_CONF_HI)
            trials.append({"seed": seed, "sigma": sigma, "trial": trial_idx, "answer": ans,
                          "correct": correct, "abstained": ans is None, "host_mrc": host,
                          "ambiguous": ambiguous, **reads})
            trial_idx += 1
    return trials


def _type2_auc(trials, key, subset=None):
    rows = trials if subset is None else [t for t in trials if t[subset]]
    rows = [t for t in rows if t[key] is not None]
    if not rows:
        return None, 0
    scores = [t[key] for t in rows]
    labels = [t["correct"] for t in rows]
    return _auc(np.asarray(scores), np.asarray(labels)), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS_SMOKE)
    ap.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS_AMBIG)
    ap.add_argument("--n-trials-per-sigma", type=int, default=N_TRIALS_PER_SIGMA)
    args = ap.parse_args()

    all_trials = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        rows = run_seed(seed, args.sigmas, args.n_trials_per_sigma)
        all_trials.extend(rows)
        n_correct = sum(1 for r in rows if r["correct"])
        n_ambig = sum(1 for r in rows if r["ambiguous"])
        n_ambig_correct = sum(1 for r in rows if r["ambiguous"] and r["correct"])
        print(f"  {len(rows)} trials, {n_correct} correct, {n_ambig} ambiguous-band "
              f"({n_ambig_correct} of those correct)", flush=True)

    verdict = {}
    for label, subset in (("all", None), ("ambiguous", "ambiguous")):
        auc_snap, n_snap = _type2_auc(all_trials, "snapshot_mrc", subset)
        auc_accum, n_accum = _type2_auc(all_trials, "accum_mrc", subset)
        # `frac_roles_bounded` -- the TIME-TO-BOUND diagnostic (fraction of a trial's roles whose competition
        # reached `_margin_accum_count_bound` within the window) -- as its OWN candidate confidence signal, not
        # folded into `accum_mrc`. Measured (not presupposed) alongside the planned primary
        # `mean_trajectory_margin` read; reported honestly even though it was not the read this de-risk set out
        # to privilege.
        auc_bounded, n_bounded = _type2_auc(all_trials, "frac_roles_bounded", subset)
        verdict[f"auc_snapshot_{label}"] = undefined_if_empty(
            f"type2 AUC snapshot ({label})", n_snap, auc_snap, n_snap)
        verdict[f"n_{label}_snapshot"] = n_snap
        verdict[f"auc_accum_{label}"] = undefined_if_empty(
            f"type2 AUC accum ({label})", n_accum, auc_accum, n_accum)
        verdict[f"n_{label}_accum"] = n_accum
        verdict[f"auc_frac_bounded_{label}"] = undefined_if_empty(
            f"type2 AUC frac_roles_bounded ({label})", n_bounded, auc_bounded, n_bounded)
        verdict[f"n_{label}_frac_bounded"] = n_bounded
        if verdict[f"auc_snapshot_{label}"] is not None and verdict[f"auc_accum_{label}"] is not None:
            verdict[f"auc_delta_{label}"] = verdict[f"auc_accum_{label}"] - verdict[f"auc_snapshot_{label}"]
        else:
            verdict[f"auc_delta_{label}"] = None
        if verdict[f"auc_snapshot_{label}"] is not None and verdict[f"auc_frac_bounded_{label}"] is not None:
            verdict[f"auc_delta_frac_bounded_{label}"] = (
                verdict[f"auc_frac_bounded_{label}"] - verdict[f"auc_snapshot_{label}"])
        else:
            verdict[f"auc_delta_frac_bounded_{label}"] = None

    # per-seed ambiguous-band AUCs (all three signals) -- the consistency-across-seeds table the finding cites.
    per_seed_auc = []
    for seed in args.seeds:
        seed_ambig = [t for t in all_trials if t["seed"] == seed and t["ambiguous"]]
        row = {"seed": seed, "n_ambiguous": len(seed_ambig),
              "n_correct": sum(1 for t in seed_ambig if t["correct"]),
              "n_incorrect": sum(1 for t in seed_ambig if not t["correct"])}
        for key, out_key in (("snapshot_mrc", "auc_snapshot"), ("accum_mrc", "auc_accum"),
                             ("frac_roles_bounded", "auc_frac_bounded")):
            a, n = _type2_auc(seed_ambig, key)
            row[out_key] = a
            row[out_key + "_n"] = n
        per_seed_auc.append(row)
    verdict["per_seed_ambiguous_auc"] = per_seed_auc

    # secondary, sample-size-robust diagnostic: mean confidence by correctness in the ambiguous band (does NOT
    # need a full ROC, just enough of each class to report a mean) -- reported alongside the AUC, not in place of
    # it, since a mean split can look separated even when the AUC (which uses every pairwise comparison) is not.
    ambig = [t for t in all_trials if t["ambiguous"]]
    for key in ("snapshot_mrc", "accum_mrc", "frac_roles_bounded"):
        corr = [t[key] for t in ambig if t["correct"] and t[key] is not None]
        wrong = [t[key] for t in ambig if not t["correct"] and t[key] is not None]
        verdict[f"ambiguous_mean_{key}_correct"] = float(np.mean(corr)) if corr else None
        verdict[f"ambiguous_mean_{key}_incorrect"] = float(np.mean(wrong)) if wrong else None
        verdict[f"ambiguous_n_correct_{key}"] = len(corr)
        verdict[f"ambiguous_n_incorrect_{key}"] = len(wrong)

    verdict["n_total_trials"] = len(all_trials)
    verdict["n_ambiguous_trials"] = len(ambig)
    verdict["seeds"] = args.seeds
    verdict["sigmas"] = args.sigmas

    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2, default=str))

    out = {"seeds": args.seeds, "sigmas": args.sigmas, "n_trials_per_sigma": args.n_trials_per_sigma,
          "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI,
          "margin_accum_count_bound": _rfp.RFPhasorComposer(seed=1)._margin_accum_count_bound,
          "trials": all_trials, "verdict": verdict}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
