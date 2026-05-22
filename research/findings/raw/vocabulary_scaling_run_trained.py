"""Vocabulary scaling on a TRAINED G.20 sparse substrate -- the
corrected re-run of the 64-concept vocabulary-scaling test.

WHY THIS RUNNER EXISTS
----------------------
The 64-concept vocabulary-scaling decisive run
(`vocabulary_scaling_run.py`) was a NEGATIVE. The capture-drive probe
arc (g20_capture_drive_probe_v1/v2/v3.py; findings
2026-05-22-vocabulary-scaling-capture-drive-probe-...) diagnosed the
cause precisely: the decisive run captured activity from a freshly-built,
UNTRAINED G.20 sparse bridge. On a fresh bridge the
`language_input -> shared_concept_pool` pathway is random and
non-selective, so a concept's orthogonal language-input drive does not
preferentially evoke its K-of-N pattern; the strong fast-spiking-
interneuron winner-take-all loop then collapses the broad, non-selective
activity to near-silence (about 0.008 of the pool active, vs the ~0.075
of the validated 16-concept substrate the pipeline passed on). The
grounded symbols, derived from near-silent noise-dominated activity, did
not compose.

The vocabulary-scaling design doc specified the project's VALIDATED
(i.e. trained) G.20 sparse substrate. The decisive run, by a setup gap,
used a fresh untrained one. This runner corrects that gap: it inserts
the validated G.20 encoding -- the sparse topographic prior plus the
validated per-concept training -- BEFORE the activity capture, then runs
the SAME biologized grounded-composition pipeline against the SAME
frozen 0.80 compositional bar.

This is NOT config-cranking a NEGATIVE. The 0.80 bar is frozen and
unchanged; the loads {2,3,5}, the seeds, the FHRR dimension, the
recognition settings -- everything downstream of the substrate -- are
imported byte-unchanged from `vocabulary_scaling_run.py`. The ONLY
change is that the substrate is exercised into the state the design doc
specified before it is read.

WHAT IS REUSED, BYTE-UNCHANGED
------------------------------
- The 64-concept G.20 sparse bridge builder + patterns (Task 1).
- The validated G.20 encoding: `apply_sparse_topographic_prior` and
  `train_concept_sparse`, imported from the validated G.20 module
  `concept_pool_sparse_distributed` (NOT modified).
- The entire biologized grounded-composition pipeline -- activity
  capture, recognition, grounding, resonate-and-fire FHRR, attractor
  clean-up, the multi-seed aggregate + verdict -- imported from
  `vocabulary_scaling_run.py` (which was adversarially reviewed CLEAR).

The genuinely-new code here is ONE function, `train_substrate`, that
orchestrates the validated prior + training calls (the orchestration
loop mirrors `concept_pool_sparse_distributed.main()`'s encoding stage;
the prior and per-concept training step it calls are reused unchanged).

PRE-REGISTERED reading (fixed; never tuned):
- PASS: integrated multi-seed mean >= 0.80 at all loads {2,3,5}. The
  biologized grounded compositional capability scales to a 64-concept
  vocabulary on the validated (trained) G.20 sparse substrate; proceed
  to the 160/320-concept ensemble.
- NEGATIVE: integrated below 0.80 at some load. The honest finding is
  that the activity-grounded pipeline needs a denser substrate than the
  G.20 sparse pool provides even when trained -- and grounding the
  symbol in the G.20 sparse K-of-N pattern itself (the concept's clean
  code) is weighed honestly against whether that is still
  substrate-grounded or closer to an oracle lookup.

`--smoke` runs a reduced-scale grounding check (tiny bridge, tiny vocab
subset, few training events) -- toy numbers, NOT propagated as a result.
A kill-safe per-seed activity cache (numeric-only .npz) lets a re-run
skip completed seeds. No protected/frozen/moat module modified. No
automatic differentiation. Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Task 1's 64-concept G.20 sparse bridge builder + patterns.
from research.findings.raw.vocabulary_scaling_substrate import (
    build_64_concept_sparse_bridge,
    sixty_four_concept_sparse_patterns,
    DEFAULT_N_LANG_INPUT,
    DEFAULT_N_SHARED_POOL,
    DEFAULT_N_SHARED_FS,
    DEFAULT_PATTERN_SIZE,
)
# The validated G.20 encoding -- reused by import, byte-unchanged.
from research.runners.concept_pool_sparse_distributed import (
    apply_sparse_topographic_prior,
    train_concept_sparse,
)
# The biologized grounded-composition pipeline + the multi-seed test
# harness -- imported byte-unchanged from the decisive runner (which was
# adversarially reviewed CLEAR). The trained-substrate runner reuses the
# pipeline exactly; only the substrate-preparation step is new.
from research.findings.raw.vocabulary_scaling_run import (
    N_CONCEPTS, BAR, LOADS, SEEDS,
    N_DIM, K_RECOG, K_VOCAB, N_TRIALS, M_OBS, SPARSITY,
    STIM_STEPS, RESET_STEPS, DRIVE_PA, TEACHER_PA,
    SMOKE_N_LANG_INPUT, SMOKE_N_SHARED_POOL, SMOKE_N_SHARED_FS,
    SMOKE_PATTERN_SIZE, SMOKE_VOCAB, SMOKE_M_OBS, SMOKE_K,
    SMOKE_TRIALS, SMOKE_STIM_STEPS,
    capture_concept_activity, recognition_accuracy, run_pipeline,
    build_smoke_bridge, smoke_sparse_patterns,
    _save_cache, _load_cache,
)

# =====================================================================
# Validated G.20 encoding parameters (the defaults of the validated G.20
# runner `concept_pool_sparse_distributed`; reused verbatim, NOT tuned).
# Task 0's test pins these so they cannot drift.
# =====================================================================
N_TRAIN_EVENTS = 400         # validated G.20 default: events per concept
TOPOGRAPHIC_FACTOR = 10.0    # validated G.20 default: prior boost factor
OFF_TARGET_FACTOR = 0.1      # validated G.20 default: prior dampen factor
TRAIN_TEACHER_PA = 500.0     # validated G.20 default: training teacher pA

# Smoke-scale training -- a few events for the fast end-to-end check.
SMOKE_TRAIN_EVENTS = 20

# Trained-substrate activity cache (kept separate from the untrained
# decisive run's cache -- the captured activity is a different quantity).
TRAINED_CACHE_DIR = "research/findings/raw/vocabulary_scaling_trained_cache"


# ---------------------------------------------------------------------
# The substrate-preparation step -- the one genuinely-new function.
# ---------------------------------------------------------------------
def train_substrate(bridge, sparse_patterns, n_lang_input, n_concepts,
                    seed, n_train_events, sparsity,
                    n_words_for_orthogonal=N_CONCEPTS, verbose=True):
    """Run the validated G.20 sparse encoding on a freshly-built bridge.

    Exercises the substrate into the state the vocabulary-scaling design
    doc specified: applies the sparse topographic prior, then runs the
    validated interleaved per-concept training. After this the
    `language_input -> shared_concept_pool` pathway is selective, so a
    concept's orthogonal language-input drive evokes its own K-of-N
    pattern rather than the whole pool.

    The orchestration here mirrors `concept_pool_sparse_distributed`'s
    `main()` encoding stage; the two validated units it calls --
    `apply_sparse_topographic_prior` and `train_concept_sparse` -- are
    reused by import byte-unchanged. The plasticity gates are opened for
    training and frozen again afterwards, so the subsequent activity
    capture is pure observation (no further weight change).

    `sparsity` is the orthogonal-drive sparsity; it MUST equal the
    capture's sparsity so the prior/training shape the weights for the
    same active language-input set the capture drives. `n_concepts` is
    the number of concepts to train; `n_words_for_orthogonal` is the
    full-vocabulary stride for the orthogonal codes (so a smoke subset
    keeps non-colliding codes), matching how the capture is invoked.
    """
    apply_sparse_topographic_prior(
        bridge=bridge, n_concepts=n_concepts, n_lang_input=n_lang_input,
        sparse_patterns=sparse_patterns, sparsity=sparsity,
        topographic_factor=TOPOGRAPHIC_FACTOR,
        off_target_factor=OFF_TARGET_FACTOR,
        n_words_for_orthogonal=n_words_for_orthogonal, verbose=verbose)

    # Open the two plastic pathways the validated encoding trains.
    bridge.set_plasticity_gate("language_input_to_shared", 1.0)
    bridge.set_plasticity_gate("shared_to_language_output", 1.0)

    # Interleaved training -- the validated G.20 schedule: every epoch
    # presents all concepts in a fresh shuffled order.
    rng = np.random.RandomState(seed)
    interleaved = []
    for _ in range(n_train_events):
        order = list(range(n_concepts))
        rng.shuffle(order)
        interleaved.extend(order)

    t0 = time.time()
    for evt_idx, i in enumerate(interleaved):
        train_concept_sparse(
            bridge=bridge, word_idx=i, sparse_pattern=sparse_patterns[i],
            n_lang_input=n_lang_input, n_lang_output=n_lang_input,
            sparsity=sparsity, n_words_for_orthogonal=n_words_for_orthogonal,
            teacher_pA=TRAIN_TEACHER_PA)
        if verbose and (evt_idx + 1) % 2000 == 0:
            print(f"    train event {evt_idx + 1}/{len(interleaved)} "
                  f"({int(time.time() - t0)}s)", flush=True)

    # Freeze: the activity capture must be pure observation.
    bridge.set_plasticity_gate("language_input_to_shared", 0.0)
    bridge.set_plasticity_gate("shared_to_language_output", 0.0)
    if verbose:
        print(f"    [train_substrate] {len(interleaved)} events in "
              f"{time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------
# Per-seed: build -> TRAIN -> capture. Cached (kill-safe at seed grain).
# ---------------------------------------------------------------------
def capture_seed_trained(seed, smoke, m_obs, n_train_events, verbose=True):
    """Build the 64-concept G.20 sparse bridge for `seed`, run the
    validated encoding on it, then capture per-neuron concept activity.

    Cached to disk: a re-run loads the cache and skips the (expensive)
    build + train + capture for that seed -- the kill-safe resume unit.
    """
    tag = "trained_smoke" if smoke else "trained_full"
    path = os.path.join(TRAINED_CACHE_DIR, f"{tag}_seed{seed}.npz")
    if os.path.exists(path):
        acts, words, patterns = _load_cache(path)
        if verbose:
            print(f"  [seed {seed}] loaded cached TRAINED activity "
                  f"({len(words)} concepts, "
                  f"{acts[words[0]].shape[0]} obs/concept)", flush=True)
        return acts, words, patterns

    t0 = time.time()
    if smoke:
        bridge, all_words = build_smoke_bridge(seed)
        all_patterns = smoke_sparse_patterns(seed)
        words = list(all_words[:SMOKE_VOCAB])
        patterns = [all_patterns[i] for i in range(SMOKE_VOCAB)]
        n_lang_input = SMOKE_N_LANG_INPUT
        stim_steps = SMOKE_STIM_STEPS
        n_concepts_trained = SMOKE_VOCAB
    else:
        bridge, words = build_64_concept_sparse_bridge(
            seed=seed, n_lang_input=DEFAULT_N_LANG_INPUT,
            n_shared_pool=DEFAULT_N_SHARED_POOL,
            n_shared_fs=DEFAULT_N_SHARED_FS,
            pattern_size=DEFAULT_PATTERN_SIZE, verbose=verbose)
        patterns = sixty_four_concept_sparse_patterns(
            seed, n_shared_pool=DEFAULT_N_SHARED_POOL,
            pattern_size=DEFAULT_PATTERN_SIZE)
        n_lang_input = DEFAULT_N_LANG_INPUT
        stim_steps = STIM_STEPS
        n_concepts_trained = N_CONCEPTS

    # --- THE CORRECTION: exercise the substrate before capture. -------
    if verbose:
        print(f"  [seed {seed}] training the G.20 sparse substrate "
              f"({n_concepts_trained} concepts x {n_train_events} "
              f"events) ...", flush=True)
    train_substrate(
        bridge, patterns, n_lang_input=n_lang_input,
        n_concepts=n_concepts_trained, seed=seed,
        n_train_events=n_train_events, sparsity=SPARSITY,
        n_words_for_orthogonal=N_CONCEPTS, verbose=verbose)

    if verbose:
        print(f"  [seed {seed}] capturing {m_obs} activity "
              f"observations/concept for {len(words)} concepts ...",
              flush=True)
    acts = capture_concept_activity(
        bridge, words, patterns, m_obs=m_obs,
        n_lang_input=n_lang_input, n_words_for_orthogonal=N_CONCEPTS,
        stim_steps=stim_steps, verbose=verbose)

    _save_cache(path, acts, words, patterns)
    if verbose:
        captured = float(np.mean([np.mean(acts[w] > 0.0) for w in words]))
        print(f"  [seed {seed}] trained + captured + cached in "
              f"{time.time() - t0:.1f}s "
              f"(captured pool density {captured:.4f})", flush=True)
    return acts, words, patterns


def run_one_seed_trained(seed, smoke=False):
    """Capture trained-substrate 64-concept activity for one seed and run
    the biologized grounded-composition pipeline on it.

    The recognition readout and the pipeline are imported byte-unchanged
    from `vocabulary_scaling_run.py`; only the substrate state differs
    (trained here vs untrained in the decisive run)."""
    print(f"\n--- seed {seed} ---", flush=True)
    m_obs = SMOKE_M_OBS if smoke else M_OBS
    k_recog = SMOKE_K if smoke else K_RECOG
    k_vocab = SMOKE_K if smoke else K_VOCAB
    n_trials = SMOKE_TRIALS if smoke else N_TRIALS
    n_train_events = SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS

    acts, words, _patterns = capture_seed_trained(
        seed, smoke, m_obs, n_train_events)

    # Recognition -- reported separately, never folded into composition.
    d_act = acts[words[0]].shape[1]
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    rec_per_obs, rec_avg = recognition_accuracy(
        acts, words, consolidated, k_recog, np.random.default_rng(seed + 7))

    per_load = run_pipeline(seed, acts, words, LOADS, n_trials,
                            k_recog, k_vocab)

    for load in LOADS:
        e = per_load[load]
        print(f"  L={load}: integrated acc={e['integrated_accuracy']:.4f} "
              f"| composition-only acc={e['composition_only_accuracy']:.4f} "
              f"(n={e['n_composition_only']})", flush=True)
    print(f"  [seed {seed}] recognition (reported separately): "
          f"per-observation={rec_per_obs:.4f}, "
          f"temporally-averaged={rec_avg:.4f}", flush=True)

    return {
        "seed": seed,
        "smoke": bool(smoke),
        "trained_substrate": True,
        "n_train_events": int(n_train_events),
        "n_concepts_captured": len(words),
        "activity_dim": int(d_act),
        "m_obs": int(m_obs),
        "recognition_per_observation": rec_per_obs,
        "recognition_temporally_averaged": rec_avg,
        "per_load": per_load,
    }


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Vocabulary scaling on a TRAINED 64-concept G.20 "
                    "sparse substrate -- the corrected re-run, against "
                    "the frozen 0.80 compositional bar.")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced-scale grounding check (tiny bridge + "
                         "tiny vocab subset + few training events) -- "
                         "toy numbers, NOT propagated as a result")
    args = ap.parse_args()

    smoke = bool(args.smoke)
    seeds = [42] if smoke else SEEDS

    print("=== vocabulary scaling: TRAINED 64-concept G.20 sparse "
          "substrate -> biologized grounded-composition pipeline ===",
          flush=True)
    if smoke:
        print("  *** SMOKE MODE: reduced-scale bridge + tiny vocab "
              "subset + few training events, toy numbers, NOT a "
              "result ***", flush=True)
    print(f"concepts={N_CONCEPTS}; FHRR N_dim={N_DIM}; "
          f"train events/concept={SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS}; "
          f"recognition K={SMOKE_K if smoke else K_RECOG}; loads={LOADS}; "
          f"bar={BAR}; seeds={seeds}; grounding=meancenter; "
          f"substrate=TRAINED (validated G.20 encoding)", flush=True)

    seed_results = [run_one_seed_trained(s, smoke=smoke) for s in seeds]

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][load]["integrated_accuracy"]
                    for r in seed_results]
        comp_accs = [r["per_load"][load]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        valid_comp = [c for c in comp_accs if c == c]
        mean_comp = float(np.mean(valid_comp)) if valid_comp else float("nan")
        agg[load] = {"mean_integrated": mean_int,
                     "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed="
              f"{['%.3f' % a for a in int_accs]} mean={mean_int:.4f} "
              f"({'>=' if mean_int >= BAR else '<'} {BAR}) | "
              f"composition-only mean={mean_comp:.4f}", flush=True)

    rec_per_obs = float(np.mean([r["recognition_per_observation"]
                                 for r in seed_results]))
    rec_avg = float(np.mean([r["recognition_temporally_averaged"]
                             for r in seed_results]))
    print(f"\nrecognition (reported separately + honestly): "
          f"per-observation mean={rec_per_obs:.4f}, "
          f"temporally-averaged mean={rec_avg:.4f}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE run -- toy numbers, not propagated as a result.",
              flush=True)
    elif all_pass:
        verdict = "VOCABULARY_SCALING_64CONCEPT_TRAINED_PASS"
        print("  The biologized grounded-composition pipeline clears the "
              "frozen 0.80 bar multi-seed at all loads on a TRAINED "
              "64-concept G.20 sparse substrate -- the compositional "
              "capability scales beyond the 16-concept vocabulary. "
              "Proceed to the 160/320-concept ensemble.", flush=True)
    else:
        verdict = "VOCABULARY_SCALING_64CONCEPT_TRAINED_BELOW_BAR"
        print("  Integrated multi-seed mean is below 0.80 at some load "
              "even on the trained substrate. The honest finding: the "
              "activity-grounded pipeline needs a denser substrate than "
              "the G.20 sparse pool provides; weigh pattern-grounded "
              "symbols honestly against oracle-adjacency.", flush=True)

    out = {
        "seeds": seeds, "n_concepts": N_CONCEPTS, "n_dim": N_DIM,
        "n_train_events": SMOKE_TRAIN_EVENTS if smoke else N_TRAIN_EVENTS,
        "k_recog": SMOKE_K if smoke else K_RECOG, "loads": LOADS,
        "n_trials": SMOKE_TRIALS if smoke else N_TRIALS, "bar": BAR,
        "grounding": "meancenter", "substrate": "trained", "smoke": smoke,
        "topographic_factor": TOPOGRAPHIC_FACTOR,
        "off_target_factor": OFF_TARGET_FACTOR,
        "train_teacher_pA": TRAIN_TEACHER_PA,
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "recognition_per_observation_mean": rec_per_obs,
        "recognition_temporally_averaged_mean": rec_avg,
        "verdict": verdict,
    }
    tag = "smoke" if smoke else "full"
    out_path = f"research/findings/raw/vocabulary_scaling_run_trained_{tag}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
