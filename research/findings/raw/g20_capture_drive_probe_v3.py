"""G.20 sparse capture-drive probe v3 -- does priming the substrate with
the validated G.20 topographic prior fix the near-silent capture?

THE DIAGNOSTIC CHAIN SO FAR
---------------------------
The 64-concept vocabulary-scaling decisive run was a NEGATIVE: the
captured G.20 sparse activity was near-silent (0.0077 pool-nonzero,
recomputed from the recording), about ten times sparser than the
validated v14/v16 substrate (0.075) the biologized grounded-composition
pipeline was validated on.

- Probe v1 swept the teacher current on a reduced-scale bridge and
  claimed "100 pA was too weak". A smell-test FALSIFIED that: v1's
  numbers were a scale artifact of its 1000-neuron pool.
- Probe v2 ran at the decisive run's EXACT full scale (8192 / 2000 /
  300 / K=100) and found that ALL THREE drive conditions -- teacher
  only, lang+teacher, lang only -- are near-silent (~0.003-0.004
  pool-nonzero). The lang_input drive is not the suppressor. A stronger
  teacher (2000 pA) does recover density, but only by force-firing the
  K pattern itself -- which makes the captured "activity" little more
  than the pattern, edging toward the oracle-symbol shortcut the
  biologization arc exists to remove.

So the cause is the substrate itself, freshly built and UNTRAINED. On a
fresh bridge the `language_input -> shared_concept_pool` pathway is
random and non-selective: no concept's drive evokes its K pattern
preferentially, and the strong FS WTA loop clamps the result to
near-silence.

WHAT v3 TESTS
-------------
The validated G.20 substrate is the EXERCISED one. The G.20 builder
ships `apply_sparse_topographic_prior`: it boosts each concept's
lang_input -> K-pattern edges by a topographic factor and dampens the
off-pattern edges. That prior is the structural selectivity a fresh
bridge lacks. v3 asks the pre-registered question: with that validated
prior applied, does a concept's natural lang_input drive evoke its K
pattern at a density comparable to v14/v16 AND selectively (its own
pattern much more than other concepts')?

If yes: the vocabulary-scaling NEGATIVE was capturing from an unprimed
substrate; the fix is a cheap prior-insert before capture, and the
64-concept test should be re-run on a primed substrate.
If no: the G.20 sparse pool's dynamics are intrinsically too quiet for
an activity-grounded readout at this scale -- route to pattern-grounded
symbols (with the honest oracle-adjacency caveat) or the full validated
training stage.

PRE-REGISTERED reading (fixed; never tuned):
- PRIMED_SUBSTRATE_GROUNDABLE: with the prior applied, a concept's
  lang+teacher capture (the decisive run's drive) reaches pool density
  >= 0.04 nonzero AND the driven concept's own K-pattern recruitment is
  >= 2x the mean recruitment of other concepts' patterns. Route:
  re-run the pre-registered 64-concept vocabulary-scaling test
  capturing from a primed G.20 sparse bridge.
- PRIMED_SUBSTRATE_STILL_TOO_SPARSE: even with the prior the primed
  capture stays below 0.04 nonzero. The G.20 sparse pool is
  intrinsically too quiet for an activity-grounded readout at this
  scale.

Reuses Task 1's 64-concept builder and the validated G.20
`apply_sparse_topographic_prior` + `orthogonal_drive_pattern` by import,
byte-unchanged. No protected/frozen/moat module modified. No automatic
differentiation. Plain ASCII.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_substrate import (
    build_64_concept_sparse_bridge,
    sixty_four_concept_sparse_patterns,
    DEFAULT_N_LANG_INPUT,
    DEFAULT_N_SHARED_POOL,
    DEFAULT_N_SHARED_FS,
    DEFAULT_PATTERN_SIZE,
)
# The validated G.20 topographic prior -- reused by import, byte-unchanged.
from research.runners.concept_pool_sparse_distributed import (
    apply_sparse_topographic_prior,
)

SEED = 42
N_CONCEPTS = 64
N_CONCEPTS_TESTED = 6        # concepts captured for the density readout
M_OBS = 4                    # observations per (concept, condition)

# Capture parameters -- identical to the decisive vocabulary-scaling run.
DRIVE_PA = 200.0
TEACHER_PA = 100.0
SPARSITY = 0.01
STIM_STEPS = 100
RESET_STEPS = 20

# Pre-registered reading thresholds (fixed; never tuned).
V16_DENSITY = 0.075
TARGET_DENSITY_LOW = 0.04    # "comparable to v14/v16" lower bound
SELECTIVITY_MIN = 2.0        # own-pattern recruitment / off-pattern mean


def _capture(bridge, pool_idx, ext, cp, to_host):
    """Drive the bridge with a fixed external-input vector and return
    the per-neuron firing-rate vector over the shared pool."""
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    counts = cp.zeros(pool_idx.shape[0], dtype=cp.float64)
    for _ in range(STIM_STEPS):
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += bridge.cp_firing_states[pool_idx]
    bridge.cp_external_input_current[:] = 0.0
    return np.asarray(to_host(counts), dtype=np.float64) / float(STIM_STEPS)


def _measure(bridge, pool_idx, lang_idx, shared_local, patterns, cp,
             to_host, orthogonal_drive_pattern, use_lang, teacher_pA):
    """Capture pool density + K-pattern selectivity for N_CONCEPTS_TESTED
    concepts under one drive condition.

    Selectivity: for a driven concept, the fraction of its OWN K-pattern
    neurons that fired vs the mean fraction over OTHER concepts' patterns.
    A groundable concept symbol needs own >> off."""
    n_total = int(bridge.cp_external_input_current.shape[0])
    pool_fz, own_rec, off_rec = [], [], []
    for cidx in range(N_CONCEPTS_TESTED):
        ext = cp.zeros(n_total, dtype=cp.float32)
        if use_lang:
            drive_in = orthogonal_drive_pattern(
                cue_idx=cidx, n_cues=N_CONCEPTS,
                n_neurons=DEFAULT_N_LANG_INPUT,
                drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
            ext[lang_idx] = cp.asarray(drive_in, dtype=cp.float32)
        if teacher_pA:
            pat_global = [shared_local[i] for i in patterns[cidx]]
            ext[cp.asarray(pat_global, dtype=cp.int64)] = teacher_pA
        for _ in range(M_OBS):
            rates = _capture(bridge, pool_idx, ext, cp, to_host)
            pool_fz.append(float(np.mean(rates > 0.0)))
            own = np.mean(rates[np.asarray(patterns[cidx])] > 0.0)
            others = [j for j in range(N_CONCEPTS) if j != cidx][:16]
            off = np.mean([np.mean(rates[np.asarray(patterns[j])] > 0.0)
                           for j in others])
            own_rec.append(float(own))
            off_rec.append(float(off))
    return (float(np.mean(pool_fz)), float(np.mean(own_rec)),
            float(np.mean(off_rec)))


def main():
    print("=== G.20 sparse capture-drive probe v3 "
          "(does the validated topographic prior fix the near-silence?) ===")
    print(f"seed {SEED}; full-scale bridge: lang={DEFAULT_N_LANG_INPUT}, "
          f"pool={DEFAULT_N_SHARED_POOL}, FS={DEFAULT_N_SHARED_FS}, "
          f"K={DEFAULT_PATTERN_SIZE}; v14/v16 density {V16_DENSITY}")

    from sim.backend import get_backend, to_host
    from sim.text_embeddings import orthogonal_drive_pattern
    cp, _ = get_backend()

    bridge, words = build_64_concept_sparse_bridge(
        seed=SEED, n_lang_input=DEFAULT_N_LANG_INPUT,
        n_shared_pool=DEFAULT_N_SHARED_POOL,
        n_shared_fs=DEFAULT_N_SHARED_FS,
        pattern_size=DEFAULT_PATTERN_SIZE, verbose=True)
    patterns = sixty_four_concept_sparse_patterns(
        SEED, n_shared_pool=DEFAULT_N_SHARED_POOL,
        pattern_size=DEFAULT_PATTERN_SIZE)

    rm = bridge.region_manager
    lang_idx = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    shared_local = list(rm.indices("shared_concept_pool"))
    pool_idx = cp.asarray(shared_local, dtype=cp.int64)

    results = {}

    # --- Stage 1: fresh (unprimed) substrate, the decisive-run drive ---
    fz, own, off = _measure(bridge, pool_idx, lang_idx, shared_local,
                            patterns, cp, to_host, orthogonal_drive_pattern,
                            use_lang=True, teacher_pA=TEACHER_PA)
    results["fresh_lang_plus_teacher"] = {
        "pool_frac_nonzero": fz, "own_recruit": own, "off_recruit": off}
    print(f"  fresh  lang+teacher : pool nonzero={fz:.4f}  "
          f"own-pattern recruit={own:.4f}  off-pattern recruit={off:.4f}")

    # --- Apply the validated G.20 topographic prior --------------------
    print("  applying validated G.20 topographic prior "
          "(boost x10 / dampen x0.1) ...")
    apply_sparse_topographic_prior(
        bridge, n_concepts=N_CONCEPTS, n_lang_input=DEFAULT_N_LANG_INPUT,
        sparse_patterns=patterns, sparsity=SPARSITY,
        n_words_for_orthogonal=N_CONCEPTS, verbose=True)

    # --- Stage 2: primed substrate, lang-only (cleanest -- no teacher) -
    fz, own, off = _measure(bridge, pool_idx, lang_idx, shared_local,
                            patterns, cp, to_host, orthogonal_drive_pattern,
                            use_lang=True, teacher_pA=0.0)
    results["primed_lang_only"] = {
        "pool_frac_nonzero": fz, "own_recruit": own, "off_recruit": off}
    print(f"  primed lang-only    : pool nonzero={fz:.4f}  "
          f"own-pattern recruit={own:.4f}  off-pattern recruit={off:.4f}")

    # --- Stage 3: primed substrate, the decisive-run lang+teacher drive
    fz, own, off = _measure(bridge, pool_idx, lang_idx, shared_local,
                            patterns, cp, to_host, orthogonal_drive_pattern,
                            use_lang=True, teacher_pA=TEACHER_PA)
    results["primed_lang_plus_teacher"] = {
        "pool_frac_nonzero": fz, "own_recruit": own, "off_recruit": off}
    print(f"  primed lang+teacher : pool nonzero={fz:.4f}  "
          f"own-pattern recruit={own:.4f}  off-pattern recruit={off:.4f}")

    # --- Pre-registered verdict ----------------------------------------
    pt = results["primed_lang_plus_teacher"]
    dense = pt["pool_frac_nonzero"] >= TARGET_DENSITY_LOW
    sel = (pt["own_recruit"] >= SELECTIVITY_MIN * max(pt["off_recruit"],
                                                      1e-9))
    print(f"\n=== VERDICT ===")
    if dense and sel:
        verdict = "PRIMED_SUBSTRATE_GROUNDABLE"
        print(f"  With the validated G.20 topographic prior applied, the "
              f"primed substrate's lang+teacher capture reaches "
              f"{pt['pool_frac_nonzero']:.4f} pool-nonzero (>= "
              f"{TARGET_DENSITY_LOW}, comparable to v14/v16's "
              f"{V16_DENSITY}) and is selective -- own-pattern "
              f"recruitment {pt['own_recruit']:.4f} vs off-pattern "
              f"{pt['off_recruit']:.4f}. The vocabulary-scaling NEGATIVE "
              f"was capturing from an UNPRIMED (untrained) substrate. "
              f"Route: re-run the pre-registered 64-concept "
              f"vocabulary-scaling test capturing from a primed G.20 "
              f"sparse bridge -- a cheap prior-insert before capture.")
    elif not dense:
        verdict = "PRIMED_SUBSTRATE_STILL_TOO_SPARSE"
        print(f"  Even with the validated topographic prior the primed "
              f"capture reaches only {pt['pool_frac_nonzero']:.4f} "
              f"pool-nonzero, below {TARGET_DENSITY_LOW}. The G.20 sparse "
              f"pool is intrinsically too quiet for an activity-grounded "
              f"readout at this scale -- route to pattern-grounded "
              f"symbols (honest oracle-adjacency caveat) or the full "
              f"validated training stage.")
    else:
        verdict = "PRIMED_DENSE_BUT_NOT_SELECTIVE"
        print(f"  The primed capture is dense "
              f"({pt['pool_frac_nonzero']:.4f}) but not selective "
              f"(own {pt['own_recruit']:.4f} vs off "
              f"{pt['off_recruit']:.4f}, < {SELECTIVITY_MIN}x). The "
              f"prior raises activity but does not separate concepts -- "
              f"the full validated training stage is needed before a "
              f"clean activity symbol can be grounded.")

    out = {
        "seed": SEED,
        "scale": {"n_lang_input": DEFAULT_N_LANG_INPUT,
                  "n_shared_pool": DEFAULT_N_SHARED_POOL,
                  "n_shared_fs": DEFAULT_N_SHARED_FS,
                  "pattern_size": DEFAULT_PATTERN_SIZE},
        "n_concepts_tested": N_CONCEPTS_TESTED, "m_obs": M_OBS,
        "v16_density": V16_DENSITY,
        "target_density_low": TARGET_DENSITY_LOW,
        "selectivity_min": SELECTIVITY_MIN,
        "stages": results,
        "verdict": verdict,
    }
    out_path = "research/findings/raw/g20_capture_drive_probe_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
