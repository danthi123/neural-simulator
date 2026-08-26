---
type: plan
status: live
date: 2026-04-20
---

# Design: G3 — Persistence across sessions

**Date:** 2026-04-20 (sketched while G2 probe runs)
**Status:** Draft — activate only if G2 closes GO
**Scope:** Gate G3 — train a brain, checkpoint it mid-training, restart the process, resume from the checkpoint, and verify the learning curve picks up *where it left off* rather than restarting.

---

## 1. Context

G1 is GO (71.3% reservoir + logreg). G2 is either showing STDP moves the needle or it's not (probe in flight). Assuming G2 is GO or PARTIAL, G3 tests whether we can checkpoint the sim state + plasticity state + RNG state and resume cleanly.

## 2. What needs to persist

**Sim state (bridge):**
- `cp_connections` (CSR)
- `cp_membrane_potential_v`, `cp_recovery_variable_u`, `cp_adex_w` (model-specific)
- `cp_conductance_g_e`, `cp_conductance_g_i`
- `cp_firing_states`, `cp_prev_firing_states`
- `cp_refractory_timers`, `cp_traits`, `cp_neuron_positions_3d`
- `cp_stp_u`, `cp_stp_x`
- `cp_last_spike_time`, `cp_eligibility_trace`
- `cp_ou_current`
- `cp_synapse_plastic_mask` ← **NEW: G3 requires this**
- `cp_synapse_pulse_timers`, `cp_synapse_pulse_progress`
- `_synapse_count`, `_synapse_capacity`
- `runtime_state.current_time_step`, `current_time_ms`, `actual_seed_used`

Most of this is already in `save_checkpoint` / `load_checkpoint`. Adds: `cp_synapse_plastic_mask`.

**Runner state (G2+ wrapper):**
- Current epoch number
- Training-example RNG state (for reproducible shuffles)
- Accumulated per-epoch metrics (so resumed run appends to the same log)
- Random seed used for fresh LogReg per epoch

**Dataset:** stays on disk; path recorded in checkpoint metadata.

## 3. Architecture

### 3.1 Bridge additions
- Extend `save_checkpoint` to write `cp_synapse_plastic_mask` when non-None.
- Extend `load_checkpoint` to restore it (skip if absent, for back-compat).

### 3.2 Runner additions
- Wrap state in a `G3Checkpoint` dataclass: `epoch`, `rng_state`, `per_epoch_history`, `core_config_snapshot`.
- Serialize to JSON alongside the sim `.simstate.h5`.
- `run_g3(start_from=None, save_every=N_epochs)` — if `start_from` is not None, load bridge + runner state and resume at `checkpoint.epoch + 1`.

### 3.3 Test harness

The G3 runner test is **two process invocations**:

1. Process A: `run_g3(seed=42, n_epochs=6, save_after=3)` → trains 3 epochs, saves `g3_ckpt_seed42_ep3.simstate.h5` + `g3_ckpt_seed42_ep3.json`.
2. Process B: `run_g3(seed=42, n_epochs=6, start_from='g3_ckpt_seed42_ep3')` → loads, trains 3 more, saves final.

Verify:
- Final epoch's test_acc is similar to a clean 6-epoch run's final test_acc (within ±5 pp).
- Intermediate epoch (4) weights match clean-run epoch 4 weights (given same RNG).

Actually easier: **compare trajectories.**
- Run A: 6 epochs in one process, record history.
- Run B: 3 epochs, save, load, 3 more, record history.
- For a deterministic run, B's trajectory should match A's from epoch 3 onwards.

## 4. Success criteria

**GO:**
- After load+resume, test accuracy at final epoch within 3 pp of a clean run at the same final epoch (with the same seed).
- Plasticity state (weights, last_spike_time) within numerical tolerance between clean-run epoch K and resumed-run epoch K.

**NO-GO:**
- Large trajectory divergence (>5 pp at final epoch).
- Crash on load.

**PARTIAL:**
- Resume works but the curve doesn't match exactly due to RNG state not captured — document and fix RNG persistence specifically.

## 5. Risks

| Risk | Mitigation |
|------|------------|
| RNG state for the example-shuffle generator not captured → different orderings, non-deterministic | Explicitly save `np.random.Generator.bit_generator.state` |
| `cp_synapse_plastic_mask` missing from existing checkpoint path | Add save/load; add a test that saves+loads a masked bridge and asserts the mask is preserved |
| OU noise RNG state in the sim isn't captured | Check whether `cp_ou_current` alone is enough (probably yes — it's a stateful current, no separate RNG state) |
| CuPy RNG state for stimulus Poisson draws isn't captured | Accept non-determinism on stimulus spikes; our gates measure average behaviour, not bit-exact reproducibility |

## 6. Deliverables

- `sim/bridge.py`: plastic-mask save/load.
- `tests/test_plastic_mask_checkpoint.py`: round-trip test.
- `research/runners/g3_runner.py` with `run_g3(..., start_from=...)`.
- `tests/test_g3_resume.py`: two-phase test (train-then-resume vs clean-run trajectory match).
- `research/findings/2026-04-20-g3.md`.

## 7. Branch decision

Stays on `main`. Pure extension of existing `save_checkpoint` / `load_checkpoint`.
