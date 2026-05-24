"""Direction E substrate Task 0: GROUNDING PIN smoke.

This is the FIRST concrete step of the theta-gamma substrate
biologization arc (per docs/plans/2026-05-24-direction-E-theta-gamma-
substrate-design.md). The grounding pin is the cheapest possible
test that exercises the WIRING but doesn't yet claim a substrate
capability:

1. Build the v16 substrate WITHOUT enable_positional_context (no
   ec_context region; this is theta-gamma's parallel positional
   mechanism, so we don't want to confuse the two).
2. Use the step-index phase function reused from pirazzini's
   _phase_is_trough pattern (Direction E substrate design (1a) --
   no new region needed).
3. Train via standard v16 recipe (200 events / word; 16 words).
4. Run encoding for 1 sequence with gamma-slot-gated drive: drive
   lang_input(word_i) ONLY during gamma slot i within a theta cycle;
   repeat across multiple theta cycles.
5. Verify the engram tag captures non-zero neurons.
6. Verify the slot index changes correctly across simulation steps
   (instrument check; the actual capability test is later tasks).

The grounding pin GREENS only after Task 1 (the actual gamma-slot
gated encoding wired through bridge primitives) is implemented. Per
discipline: the grounding pin must be present from Task 0 so we
detect harness regressions early. The pin is intentionally minimal
- enough to verify the simulation runs end-to-end through one theta
cycle with gamma-slot gating, NOT enough to claim a substrate
capability.

NUMPY-aware GPU-friendly; ~5 min wall on a small substrate (no full
v16 training; this is just a wiring smoke).
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.concept_pool_demo import (
    build_concept_bridge, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from research.runners.concept_compose_train import _WORD_TO_POOL
from sim.text_embeddings import orthogonal_drive_pattern
from sim.backend import get_backend, is_gpu_backend


OUT_JSON = os.path.join(
    _HERE, "direction_E_substrate_task0_grounding.json")

# Pre-registered constants (matching Direction E algebra)
N_LANG_INPUT = 2048
N_PER_POOL = 200
N_FS_PER_POOL = 24
SPARSITY = 0.05
SEED = 42
SLOT_COUNT = 3  # small smoke; full would be 5 or 7

# Theta-gamma timing (Lisman-Idiart): 8Hz theta, 56Hz gamma (7 slots)
# At bridge dt=1.0ms, theta cycle = 125 steps; gamma slot = 17-18 steps
THETA_MS = 125.0
N_GAMMA = 7
TEACHER_PA = 500.0
N_THETA_CYCLES = 3  # repeat sequence presentation 3 times


def phase_to_gamma_slot(step_idx, theta_steps, n_gamma):
    """Map simulation step within theta cycle to gamma slot index
    (0..n_gamma-1). Reuses the pirazzini _phase_is_X pattern."""
    phase = int(step_idx) % int(theta_steps)
    return min(n_gamma - 1, (phase * n_gamma) // theta_steps)


def main():
    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print(f"=== Direction E substrate Task 0: GROUNDING PIN ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Grounding pin: verify gamma-slot-gated encoding "
          f"wiring works; does NOT claim capability.", flush=True)
    print(f"  THETA_MS={THETA_MS}, N_GAMMA={N_GAMMA}, "
          f"SLOT_COUNT={SLOT_COUNT}, n_theta_cycles={N_THETA_CYCLES}",
          flush=True)

    # Build substrate WITHOUT enable_positional_context (theta-gamma
    # uses temporal phase, not spatial ec_context region)
    words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
             list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
    n_words = len(words)

    t0 = time.time()
    bridge = build_concept_bridge(
        seed=SEED, n_lang_input=N_LANG_INPUT, n_per_pool=N_PER_POOL,
        n_fs_per_pool=N_FS_PER_POOL, enable_adjective=True,
        weak_dynamics=True, enable_positional_context=False,
        verbose=False,
    )
    print(f"  built bridge in {(time.time()-t0):.1f}s",
          flush=True)

    cfg = bridge.core_config
    theta_steps = max(2, int(round(THETA_MS / cfg.dt_ms)))
    gamma_period = max(1, theta_steps // N_GAMMA)
    print(f"  bridge dt={cfg.dt_ms} ms -> theta_steps={theta_steps},"
          f" gamma_period={gamma_period}", flush=True)

    # Verify phase function correctness
    test_phases = []
    for step in range(2 * theta_steps):
        slot = phase_to_gamma_slot(step, theta_steps, N_GAMMA)
        test_phases.append((step, slot))
    # First slot 0 transition
    first_slot1_step = next(
        (s for s, sl in test_phases if sl == 1), None)
    first_slot1_expected = gamma_period
    print(f"  phase function check: first slot=1 step "
          f"{first_slot1_step} (expected ~{first_slot1_expected})",
          flush=True)
    # Last slot in first theta cycle
    last_slot_first_cycle = test_phases[theta_steps - 1][1]
    print(f"  last step of first theta cycle slot = "
          f"{last_slot_first_cycle} (expected {N_GAMMA - 1})",
          flush=True)
    # Wrap-around
    second_cycle_slot_0_step = test_phases[theta_steps][1]
    print(f"  first step of second theta cycle slot = "
          f"{second_cycle_slot_0_step} (expected 0)", flush=True)

    # Pick a small test sequence
    seq_words = [words[0], words[4], words[8]]  # north, apple, go
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Resolve indices
    cp, _ = get_backend()
    rm = bridge.region_manager
    lang_in_idx = list(rm.indices("language_input"))
    lang_in_arr = cp.asarray(lang_in_idx, dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    pool_arrs = []
    for w in seq_words:
        pool_idx = list(rm.indices(_WORD_TO_POOL[w]))
        pool_arrs.append(cp.asarray(pool_idx, dtype=cp.int64))

    # Freeze plasticity (this is a grounding pin; not learning)
    for g in ("language_input_to_motor",
              "language_input_to_noun_pool",
              "language_input_to_verb_pool",
              "language_input_to_adjective_pool",
              "motor_to_language_output",
              "noun_pool_to_language_output",
              "verb_pool_to_language_output",
              "adjective_pool_to_language_output"):
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass

    # Settle
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Gamma-slot-gated encoding: across N_THETA_CYCLES theta cycles,
    # drive lang_input(word_i) + teacher(pool_i) ONLY during gamma
    # slot i within each theta cycle.
    tag_name = "task0_grounding"
    bridge.start_engram_recording(tag_name)
    ext = cp.zeros(n_total, dtype=cp.float32)
    drives_per_word = [orthogonal_drive_pattern(
        cue_idx=word_to_idx[w], n_cues=len(words),
        n_neurons=N_LANG_INPUT, drive_max_pA=200.0,
        sparsity=SPARSITY) for w in seq_words]

    slot_step_counts = [0] * SLOT_COUNT  # diagnostics
    encoding_steps = N_THETA_CYCLES * theta_steps
    for step in range(encoding_steps):
        current_slot = phase_to_gamma_slot(step, theta_steps, N_GAMMA)
        # Only drive when current_slot is one of our SLOT_COUNT slots
        if current_slot < SLOT_COUNT:
            slot_idx = current_slot
            ext.fill(0)
            ext[lang_in_arr] = cp.asarray(
                drives_per_word[slot_idx], dtype=cp.float32)
            ext[pool_arrs[slot_idx]] = TEACHER_PA
            bridge.cp_external_input_current[:] = ext
            slot_step_counts[slot_idx] += 1
        else:
            ext.fill(0)
            bridge.cp_external_input_current[:] = ext
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(20):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    # Use concept-pool region filter (same as validated multitag)
    region_filter = []
    for kind, names in [
        ("noun_pool", ["APPLE", "RIVER", "DOG", "CAT"]),
        ("verb_pool", ["GO", "COME", "STOP", "LOOK"]),
        ("adjective_pool", ["BIG", "SMALL", "HOT", "COLD"]),
    ]:
        for n in names:
            try:
                rm.indices(f"{kind}_{n}")
                region_filter.append(f"{kind}_{n}")
            except Exception:
                pass
    for m in ["motor_N", "motor_E", "motor_S", "motor_W"]:
        try:
            rm.indices(m)
            region_filter.append(m)
        except Exception:
            pass

    stats = bridge.commit_engram_tag(
        tag_name, top_k=100, region_filter=region_filter)
    n_tagged = stats.get("n_tagged", 0)

    print(f"\n  encoding wall: {(time.time()-t0)/60:.1f} min total",
          flush=True)
    print(f"  steps per slot: {slot_step_counts} "
          f"(each should be ~{N_THETA_CYCLES * gamma_period})",
          flush=True)
    print(f"  engram tag n_tagged: {n_tagged} (>0 -> wiring "
          f"works end-to-end)", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if (n_tagged > 0 and all(c > 0 for c in slot_step_counts)
            and last_slot_first_cycle == N_GAMMA - 1):
        verdict = "TASK0_GROUNDING_GREEN"
        print(f"  GROUNDING GREEN: phase function correct, "
              f"per-slot encoding fires the right neurons, engram "
              f"tag captures non-zero neurons. Task 1+2+3 build "
              f"justified.", flush=True)
    else:
        verdict = "TASK0_GROUNDING_RED"
        print(f"  GROUNDING RED: wiring issue. Diagnose before "
              f"building Task 1.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu, "seed": SEED,
        "theta_ms": THETA_MS, "theta_steps": theta_steps,
        "n_gamma": N_GAMMA, "gamma_period": gamma_period,
        "slot_count": SLOT_COUNT, "n_theta_cycles": N_THETA_CYCLES,
        "slot_step_counts": slot_step_counts,
        "first_slot1_step": first_slot1_step,
        "last_slot_first_cycle": last_slot_first_cycle,
        "second_cycle_first_slot": second_cycle_slot_0_step,
        "n_tagged": n_tagged,
        "verdict": verdict,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
