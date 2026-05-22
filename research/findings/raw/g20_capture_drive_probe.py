"""G.20 sparse capture-drive probe -- diagnosing the vocabulary-scaling
NEGATIVE.

*** RETRACTED 2026-05-22 -- this probe's DRIVE_GAP_RECOVERABLE verdict
is a SCALE ARTIFACT and must NOT be acted on. This probe ran on a
reduced-scale bridge (1000-neuron pool); the full decisive-run bridge
has a 2000-neuron pool whose pool->interneuron->pool feedback-inhibition
loop has a much higher gain, so it behaves completely differently. At
the decisive run's exact 100 pA teacher current this probe recorded
0.0787 pool-nonzero; the full-scale bridge records 0.0026. Superseded by
g20_capture_drive_probe_v2.py (controlled, full scale) and _v3.py
(topographic prior). See research/findings/2026-05-22-vocabulary-
scaling-capture-drive-probe-near-silence-diagnosed-to-untrained-
substrate.md ***

The 64-concept vocabulary-scaling decisive run was a NEGATIVE, diagnosed
to a near-silent captured activity: the G.20 sparse shared pool fired at
only 0.5% nonzero (mean rate 0.00015), about fifteen times sparser than
the v14/v16 substrate (7.5% nonzero) the biologized grounded-composition
pipeline was validated on. The grounded symbols derived from near-silent
Poisson-noise-dominated activity could not compose.

That decisive run drove each concept with a teacher current of 100 pA on
its K-of-N sparse pattern neurons. The honest question this probe
answers: was 100 pA simply too weak, and does a stronger teacher current
bring the captured activity to a density comparable to the validated
v14/v16 substrate?

A G.20 sparse concept is a K-of-N pattern -- K of the N shared-pool
neurons. At the validated tier K/N is about 5% (K=100 in N=2000). So a
concept whose pattern neurons all fire would give an activity vector
about 5% nonzero -- close to the v14/v16 7.5%. The decisive run reached
only 0.5%, so only about a tenth of the K pattern fired. This probe
sweeps the teacher current and measures whether a stronger drive
recruits the full K pattern.

Pipeline: build ONE G.20 sparse bridge, then for a handful of concepts
capture the shared-pool activity at a range of teacher-current strengths
and stimulation-window lengths, and report the resulting activity
density (fraction of pool neurons nonzero, mean rate) and how much of
each concept's K pattern was recruited.

PRE-REGISTERED reading (fixed; never tuned):
- If a teacher current within a plausible range brings the captured
  activity to a density comparable to the validated v14/v16 substrate
  (toward 5-8% nonzero) and recruits most of each concept's K pattern,
  the vocabulary-scaling NEGATIVE was a capture-drive setup gap: re-run
  the pre-registered vocabulary-scaling test with the corrected drive.
- If no plausible teacher current recruits the K pattern -- the pool
  stays near-silent -- the G.20 sparse pool's dynamics are intrinsically
  too quiet for an activity-grounded readout, an honest substrate
  finding that routes to either a trained bridge or pattern-grounded
  symbols.

Standalone, reuses Task 1's 64-concept G.20 sparse builder and the
vocabulary-scaling capture path by import. No protected/frozen/moat
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
)

SEED = 42
# Reduced-scale bridge -- the drive -> density relationship is a local
# pool-dynamics property, so a smaller pool answers it cheaply.
N_LANG_INPUT = 1024
N_SHARED_POOL = 1000
N_SHARED_FS = 120
PATTERN_SIZE = 50            # K/N = 5%, the validated G.20 sparse ratio
N_CONCEPTS_TESTED = 6        # a handful of concepts is enough for density
M_OBS = 4
TEACHER_SWEEP = [100.0, 300.0, 600.0, 1000.0, 2000.0]   # pA on the K pattern
STIM_STEPS = 100
RESET_STEPS = 20
# The validated v14/v16 substrate the pipeline passed on: 7.5% nonzero.
V16_DENSITY = 0.075
TARGET_DENSITY_LOW = 0.04    # pre-registered: "comparable" lower bound


def capture_density(bridge, pool_idx, pattern_arr, teacher_pA, cp,
                    to_host, stim_steps, reset_steps):
    """Drive one concept's K pattern with a teacher current and return
    (fraction of pool neurons that fired at all, mean per-neuron rate,
    fraction of the concept's own K-pattern neurons recruited)."""
    n_total = int(bridge.cp_external_input_current.shape[0])
    ext = cp.zeros(n_total, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    counts = cp.zeros(pool_idx.shape[0], dtype=cp.float64)
    for _ in range(stim_steps):
        ext.fill(0)
        ext[pattern_arr] = teacher_pA
        bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += bridge.cp_firing_states[pool_idx]
    bridge.cp_external_input_current[:] = 0.0

    rates = np.asarray(to_host(counts), dtype=np.float64) / float(stim_steps)
    frac_nonzero = float(np.mean(rates > 0.0))
    mean_rate = float(np.mean(rates))
    # K-pattern recruitment: of the concept's own pattern neurons, how
    # many fired. pattern_arr holds global indices; map to pool-local.
    pool_set = {int(x): i for i, x in enumerate(np.asarray(to_host(pool_idx)))}
    pat_local = [pool_set[int(g)] for g in np.asarray(to_host(pattern_arr))
                 if int(g) in pool_set]
    pat_recruit = (float(np.mean(rates[pat_local] > 0.0))
                   if pat_local else float("nan"))
    return frac_nonzero, mean_rate, pat_recruit


def main():
    print("=== G.20 sparse capture-drive probe ===")
    print(f"seed {SEED}; reduced bridge pool={N_SHARED_POOL}, "
          f"K={PATTERN_SIZE} (K/N={PATTERN_SIZE/N_SHARED_POOL:.1%}); "
          f"teacher sweep={TEACHER_SWEEP} pA; "
          f"target density >= {TARGET_DENSITY_LOW} (v14/v16 is "
          f"{V16_DENSITY})")

    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    bridge, words = build_64_concept_sparse_bridge(
        seed=SEED, n_lang_input=N_LANG_INPUT, n_shared_pool=N_SHARED_POOL,
        n_shared_fs=N_SHARED_FS, pattern_size=PATTERN_SIZE, verbose=False)
    patterns = sixty_four_concept_sparse_patterns(
        SEED, n_shared_pool=N_SHARED_POOL, pattern_size=PATTERN_SIZE)

    rm = bridge.region_manager
    shared_local = list(rm.indices("shared_concept_pool"))
    pool_idx = cp.asarray(shared_local, dtype=cp.int64)

    grid = {}
    for teacher in TEACHER_SWEEP:
        densities, rates, recruits = [], [], []
        for cidx in range(N_CONCEPTS_TESTED):
            pattern_global = [shared_local[i] for i in patterns[cidx]]
            pattern_arr = cp.asarray(pattern_global, dtype=cp.int64)
            for _ in range(M_OBS):
                fz, mr, pr = capture_density(
                    bridge, pool_idx, pattern_arr, teacher, cp, to_host,
                    STIM_STEPS, RESET_STEPS)
                densities.append(fz)
                rates.append(mr)
                recruits.append(pr)
        d = float(np.mean(densities))
        r = float(np.mean(rates))
        pr = float(np.nanmean(recruits))
        grid[teacher] = {"frac_nonzero": d, "mean_rate": r,
                         "k_pattern_recruited": pr}
        print(f"  teacher={teacher:>6.0f} pA: pool nonzero={d:.4f}  "
              f"mean rate={r:.5f}  K-pattern recruited={pr:.3f}")

    best_teacher = max(TEACHER_SWEEP,
                       key=lambda t: grid[t]["frac_nonzero"])
    best_density = grid[best_teacher]["frac_nonzero"]

    print(f"\n=== VERDICT ===")
    if best_density >= TARGET_DENSITY_LOW:
        verdict = "DRIVE_GAP_RECOVERABLE"
        print(f"  A teacher current of {best_teacher:.0f} pA brings the "
              f"captured activity to {best_density:.4f} nonzero -- "
              f"comparable to the validated v14/v16 substrate "
              f"({V16_DENSITY}). The vocabulary-scaling NEGATIVE was a "
              f"capture-drive setup gap: re-run the pre-registered "
              f"64-concept test with the corrected teacher current.")
    else:
        verdict = "G20_POOL_INTRINSICALLY_TOO_SPARSE"
        print(f"  Even at {best_teacher:.0f} pA the captured activity "
              f"reaches only {best_density:.4f} nonzero, far below the "
              f"validated v14/v16 density ({V16_DENSITY}). The G.20 "
              f"sparse pool is intrinsically too quiet for an activity-"
              f"grounded readout -- routes to a trained bridge or "
              f"pattern-grounded symbols.")

    out = {
        "seed": SEED, "n_shared_pool": N_SHARED_POOL,
        "pattern_size": PATTERN_SIZE, "teacher_sweep": TEACHER_SWEEP,
        "stim_steps": STIM_STEPS, "v16_density": V16_DENSITY,
        "target_density_low": TARGET_DENSITY_LOW,
        "grid": {str(t): grid[t] for t in TEACHER_SWEEP},
        "best_teacher_pA": best_teacher,
        "best_density": best_density,
        "verdict": verdict,
    }
    with open("research/findings/raw/g20_capture_drive_probe.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/g20_capture_drive_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
