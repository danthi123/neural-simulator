"""G.20 sparse capture-drive probe v2 -- isolating the cause of the
vocabulary-scaling NEGATIVE's near-silent captured activity.

WHY v2 SUPERSEDES v1's NARRATIVE
--------------------------------
v1 (g20_capture_drive_probe.py) swept the teacher current on a
reduced-scale bridge and reported "DRIVE_GAP_RECOVERABLE: 100 pA was too
weak, re-run with 1000 pA". A smell-test of v1's own data falsifies that
narrative: v1 at the decisive run's exact 100 pA teacher current already
recorded 0.0787 pool-nonzero -- comparable to the validated v14/v16
substrate (0.075) and about ten times DENSER than the decisive run's
captured activity (0.0077 nonzero, recomputed from the recording). v1
and the decisive run cannot both be measuring the same thing at the
same teacher current. The reason: v1 dropped the lang_input drive. The
decisive run drove BOTH lang_input AND the teacher current; v1 drove
only the teacher. So the variable that changed was never teacher
strength -- it was the lang_input drive.

THE HYPOTHESIS v2 TESTS
-----------------------
On a FRESH, UNTRAINED G.20 sparse bridge the `language_input ->
shared_concept_pool` pathway is random and non-selective (density 0.30,
weight_mean 3.0). An orthogonal lang_input drive therefore excites the
WHOLE shared pool broadly, not the concept's K-of-N pattern
selectively. The pool's strong WTA feedback inhibition (shared_pool ->
shared_FS weight 1.0; shared_FS -> shared_pool weight 4.0) collapses
that broad, non-selective excitation to near-silence. The teacher
current alone is perfectly selective (it lands only on the K pattern)
and does NOT trigger the collapse.

If true: the vocabulary-scaling NEGATIVE was a capture-setup gap, but
NOT the teacher-strength gap v1 named -- it is the untrained,
non-selective lang_input drive collapsing the pool. The fix is to
capture from a TRAINED G.20 sparse bridge (where the lang_input->pool
pathway is selective -- the validated substrate the design doc named),
or to capture with the teacher current alone.

WHAT v2 DOES
------------
Builds ONE full-scale G.20 sparse bridge at the decisive run's EXACT
scale (8192 lang_input, 2000 shared pool, 300 shared_FS, K=100) and, for
a handful of concepts, captures shared-pool AND shared_FS activity
density under three drive conditions:
  A. teacher-only       -- teacher current on the K pattern, no lang
  B. lang+teacher       -- the decisive run's EXACT capture drive
  C. lang-only          -- orthogonal lang_input drive, no teacher
and, under condition B, sweeps the teacher current [100, 600, 2000] pA
to settle whether a stronger teacher can rescue the lang-collapsed
capture (v1's recommendation).

PRE-REGISTERED reading (fixed; never tuned):
- ISOLATED: if teacher-only (A) reaches v14/v16-comparable pool density
  (>= 0.04 nonzero) AND lang+teacher (B) reproduces the decisive run's
  near-silence (< 0.025 nonzero -- about 3x the recomputed 0.0077) AND
  shared_FS fires substantially harder under B than under A, the cause
  is isolated to the untrained, non-selective lang_input drive
  collapsing the pool through the FS WTA loop. Teacher strength was
  never the bottleneck. Routes to: re-run the 64-concept test capturing
  from a TRAINED G.20 sparse bridge.
- NOT_REPRODUCED: if lang+teacher (B) does NOT reproduce the
  near-silence (>= 0.04 nonzero), the cache and this probe disagree --
  the near-silence is a seed or scale artifact and needs a fresh look.

Standalone; reuses Task 1's 64-concept G.20 sparse builder and the
validated orthogonal_drive_pattern by import. No protected/frozen/moat
module modified. No automatic differentiation. Plain ASCII.
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

SEED = 42
N_CONCEPTS_TESTED = 6        # a handful of concepts is enough for density
M_OBS = 4                    # observations per (concept, condition)

# The decisive run's EXACT capture parameters (vocabulary_scaling_run.py).
DRIVE_PA = 200.0             # lang_input orthogonal drive
TEACHER_PA = 100.0           # teacher bias on the sparse K pattern
SPARSITY = 0.01              # orthogonal-drive sparsity (64-concept tier)
STIM_STEPS = 100
RESET_STEPS = 20
N_WORDS_FOR_ORTHOGONAL = 64  # full-vocabulary orthogonal stride

# Teacher sweep under the lang+teacher (decisive-run) drive.
TEACHER_SWEEP = [100.0, 600.0, 2000.0]

# Pre-registered reading thresholds (fixed; never tuned).
# v14/v16, the validated substrate the pipeline passed on: 0.075 nonzero.
V16_DENSITY = 0.075
TARGET_DENSITY_LOW = 0.04    # "comparable to v14/v16" lower bound
# The decisive run's captured density, recomputed from the recording
# (research/findings/raw/vocabulary_scaling_cache/full_seed42.npz):
# 0.00765 nonzero. "Near-silent" = within ~3x of that.
DECISIVE_RUN_DENSITY = 0.00765
NEAR_SILENT_MAX = 0.025


def _density(bridge, idx, ext, cp, to_host, stim_steps, reset_steps):
    """Drive the bridge with a fixed external-input vector `ext` and
    return (fraction of `idx` neurons that fired, mean per-neuron rate)
    accumulated over `stim_steps`. A reset window precedes the drive."""
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    counts = cp.zeros(idx.shape[0], dtype=cp.float64)
    for _ in range(stim_steps):
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += bridge.cp_firing_states[idx]
    bridge.cp_external_input_current[:] = 0.0

    rates = np.asarray(to_host(counts), dtype=np.float64) / float(stim_steps)
    return float(np.mean(rates > 0.0)), float(np.mean(rates))


def _capture_condition(bridge, pool_idx, fs_idx, ext, cp, to_host):
    """Capture pool + FS density once under a fixed drive `ext`.

    The pool and FS densities are captured in two separate passes so
    each gets a clean reset window -- the bridge's per-step firing state
    is read for whichever population the pass accumulates."""
    pool_fz, pool_mr = _density(bridge, pool_idx, ext, cp, to_host,
                                STIM_STEPS, RESET_STEPS)
    fs_fz, fs_mr = _density(bridge, fs_idx, ext, cp, to_host,
                            STIM_STEPS, RESET_STEPS)
    return pool_fz, pool_mr, fs_fz, fs_mr


def main():
    print("=== G.20 sparse capture-drive probe v2 "
          "(isolating the near-silence cause) ===")
    print(f"seed {SEED}; FULL-scale decisive-run bridge: "
          f"lang_input={DEFAULT_N_LANG_INPUT}, pool={DEFAULT_N_SHARED_POOL}, "
          f"FS={DEFAULT_N_SHARED_FS}, K={DEFAULT_PATTERN_SIZE}")
    print(f"decisive run captured density (recomputed from recording): "
          f"{DECISIVE_RUN_DENSITY:.5f} nonzero; v14/v16 substrate: "
          f"{V16_DENSITY}")

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
    lang_local = list(rm.indices("language_input"))
    shared_local = list(rm.indices("shared_concept_pool"))
    fs_local = list(rm.indices("shared_FS"))
    lang_idx = cp.asarray(lang_local, dtype=cp.int64)
    pool_idx = cp.asarray(shared_local, dtype=cp.int64)
    fs_idx = cp.asarray(fs_local, dtype=cp.int64)
    n_total = int(bridge.cp_external_input_current.shape[0])

    def make_ext(drive_gpu, pattern_arr, teacher_pA):
        """Build a fixed external-input vector: lang_input gets the
        orthogonal code `drive_gpu` (or nothing if None); the K pattern
        gets `teacher_pA` (or nothing if 0)."""
        ext = cp.zeros(n_total, dtype=cp.float32)
        if drive_gpu is not None:
            ext[lang_idx] = drive_gpu
        if teacher_pA:
            ext[pattern_arr] = teacher_pA
        return ext

    # --- Conditions A / B / C at the decisive-run teacher current ------
    conditions = {
        "A_teacher_only": dict(use_lang=False, teacher=TEACHER_PA),
        "B_lang_plus_teacher": dict(use_lang=True, teacher=TEACHER_PA),
        "C_lang_only": dict(use_lang=True, teacher=0.0),
    }
    cond_results = {}
    for cname, spec in conditions.items():
        pool_fz, pool_mr, fs_fz, fs_mr = [], [], [], []
        for cidx in range(N_CONCEPTS_TESTED):
            pattern_global = [shared_local[i] for i in patterns[cidx]]
            pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)
            drive_gpu = None
            if spec["use_lang"]:
                drive_in = orthogonal_drive_pattern(
                    cue_idx=cidx, n_cues=N_WORDS_FOR_ORTHOGONAL,
                    n_neurons=DEFAULT_N_LANG_INPUT,
                    drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
                drive_gpu = cp.asarray(drive_in, dtype=cp.float32)
            ext = make_ext(drive_gpu, pattern_arr, spec["teacher"])
            for _ in range(M_OBS):
                pfz, pmr, ffz, fmr = _capture_condition(
                    bridge, pool_idx, fs_idx, ext, cp, to_host)
                pool_fz.append(pfz); pool_mr.append(pmr)
                fs_fz.append(ffz); fs_mr.append(fmr)
        cond_results[cname] = {
            "pool_frac_nonzero": float(np.mean(pool_fz)),
            "pool_mean_rate": float(np.mean(pool_mr)),
            "fs_frac_nonzero": float(np.mean(fs_fz)),
            "fs_mean_rate": float(np.mean(fs_mr)),
        }
        r = cond_results[cname]
        print(f"  {cname:>22}: pool nonzero={r['pool_frac_nonzero']:.4f}  "
              f"pool rate={r['pool_mean_rate']:.5f}  | "
              f"FS nonzero={r['fs_frac_nonzero']:.4f}  "
              f"FS rate={r['fs_mean_rate']:.5f}")

    # --- Teacher sweep under the lang+teacher (decisive-run) drive -----
    print(f"\n  teacher sweep under the lang+teacher drive "
          f"(decisive-run capture):")
    sweep = {}
    for teacher in TEACHER_SWEEP:
        pool_fz = []
        for cidx in range(N_CONCEPTS_TESTED):
            pattern_global = [shared_local[i] for i in patterns[cidx]]
            pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)
            drive_in = orthogonal_drive_pattern(
                cue_idx=cidx, n_cues=N_WORDS_FOR_ORTHOGONAL,
                n_neurons=DEFAULT_N_LANG_INPUT,
                drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
            drive_gpu = cp.asarray(drive_in, dtype=cp.float32)
            ext = make_ext(drive_gpu, pattern_arr, teacher)
            for _ in range(M_OBS):
                pfz, _ = _density(bridge, pool_idx, ext, cp, to_host,
                                  STIM_STEPS, RESET_STEPS)
                pool_fz.append(pfz)
        sweep[teacher] = float(np.mean(pool_fz))
        print(f"    teacher={teacher:>6.0f} pA + lang drive: "
              f"pool nonzero={sweep[teacher]:.4f}")

    # --- Pre-registered verdict ----------------------------------------
    a = cond_results["A_teacher_only"]
    b = cond_results["B_lang_plus_teacher"]
    teacher_only_ok = a["pool_frac_nonzero"] >= TARGET_DENSITY_LOW
    lang_collapses = b["pool_frac_nonzero"] < NEAR_SILENT_MAX
    fs_harder = b["fs_mean_rate"] > a["fs_mean_rate"]
    sweep_stays_silent = all(v < TARGET_DENSITY_LOW for v in sweep.values())

    print(f"\n=== VERDICT ===")
    if teacher_only_ok and lang_collapses and fs_harder:
        verdict = "NEAR_SILENCE_ISOLATED_TO_UNTRAINED_LANG_DRIVE"
        print(f"  Teacher-only capture reaches "
              f"{a['pool_frac_nonzero']:.4f} pool-nonzero -- comparable "
              f"to the validated v14/v16 substrate ({V16_DENSITY}). The "
              f"decisive run's lang+teacher capture reproduces the "
              f"near-silence at {b['pool_frac_nonzero']:.4f} nonzero, and "
              f"shared_FS fires harder under it ({b['fs_mean_rate']:.5f} "
              f"vs {a['fs_mean_rate']:.5f}). The vocabulary-scaling "
              f"NEGATIVE's near-silence is the untrained, non-selective "
              f"lang_input drive broadly exciting the pool and the FS "
              f"WTA loop collapsing it -- NOT a too-weak teacher "
              f"current. The teacher sweep "
              f"({'stays near-silent' if sweep_stays_silent else 'recovers'}) "
              f"under the lang drive confirms a stronger teacher "
              f"{'cannot' if sweep_stays_silent else 'can'} rescue it. "
              f"Route: re-run the 64-concept test capturing from a "
              f"TRAINED G.20 sparse bridge (selective lang_input->pool "
              f"pathway), the validated substrate the design doc named.")
    elif not lang_collapses:
        verdict = "NEAR_SILENCE_NOT_REPRODUCED"
        print(f"  The lang+teacher capture did NOT reproduce the "
              f"near-silence ({b['pool_frac_nonzero']:.4f} nonzero >= "
              f"{TARGET_DENSITY_LOW}). The cache and this probe disagree "
              f"-- the decisive run's near-silence is a seed or scale "
              f"artifact and needs a fresh look.")
    else:
        verdict = "INCONCLUSIVE"
        print(f"  Mixed signature: teacher_only_ok={teacher_only_ok}, "
              f"lang_collapses={lang_collapses}, fs_harder={fs_harder}. "
              f"The simple untrained-lang-drive mechanism is not cleanly "
              f"confirmed; report honestly and investigate further.")

    out = {
        "seed": SEED,
        "scale": {"n_lang_input": DEFAULT_N_LANG_INPUT,
                  "n_shared_pool": DEFAULT_N_SHARED_POOL,
                  "n_shared_fs": DEFAULT_N_SHARED_FS,
                  "pattern_size": DEFAULT_PATTERN_SIZE},
        "n_concepts_tested": N_CONCEPTS_TESTED, "m_obs": M_OBS,
        "stim_steps": STIM_STEPS, "drive_pA": DRIVE_PA,
        "v16_density": V16_DENSITY,
        "decisive_run_density_recomputed": DECISIVE_RUN_DENSITY,
        "target_density_low": TARGET_DENSITY_LOW,
        "near_silent_max": NEAR_SILENT_MAX,
        "conditions": cond_results,
        "teacher_sweep_under_lang_drive": {str(t): v
                                           for t, v in sweep.items()},
        "verdict": verdict,
    }
    out_path = "research/findings/raw/g20_capture_drive_probe_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
