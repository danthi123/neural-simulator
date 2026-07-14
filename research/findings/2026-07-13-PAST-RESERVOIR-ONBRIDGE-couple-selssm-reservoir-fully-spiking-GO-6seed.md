# ON-BRIDGE (fully-spiking) COUPLING GO (6-seed): a recurrent SPIKING reservoir + the on-bridge SELECTIVE channel, co-resident on ONE SimulationBridge, lifts the spiking reservoir past its long-range conjunction bound — on real spikes, transport-free

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_onbridge_couple_selssm_reservoir_derisk.py` · CI `tests/test_reslm_onbridge_couple_selssm_reservoir.py` (1/1) · raw `research/findings/raw/_onbridge_couple/`. NO further `sim/` edit (uses the Rung-4b-ii `enable_selective_ssm_state` mechanism).
**Status:** ✅ GO 6/6. The emergence-bar realization (fully spiking, one brain) of the numpy coupling.

## Why

The numpy coupling was validated frozen (6/6) and co-trained (joint 6/6), transport-free, adversarially verified. The owner's core directive is **fully spiking on one brain** — so the mission step is to realize the coupling on a real `SimulationBridge`: a recurrent SPIKING reservoir + the on-bridge selective channel co-resident, read-out over both. All pieces were already on-bridge-validated separately (the spiking reservoir = EMERGE-82's `internal_density` recurrence + Izhikevich + conductance synapses; the selective channel = Rung 4b-iii-a byte-equivalent to numpy, 4b-iii-b learns end-to-end on-bridge). This combines them on ONE bridge.

## Setup (single variable = the on-bridge selective channel; fully spiking; transport-free)

ONE bridge with two regions: a **recurrent reservoir** (`internal_density=0.2` → fixed-random recurrent Izhikevich synapses) on `res_idx`, and a disjoint **ssm** region on `ssm_idx` carrying the selective channel (`enable_selective_ssm_state`). Per token: drive the reservoir via `cp_external_input_current` (`W_in @ onehot + bias`) + set the selective channel's `cp_ssm_inject`/`cp_ssm_shunt` on the ssm slice; step the bridge `T_STEP=6` times (the bridge's real conductance-synapse + Izhikevich loop); read `h_t` = the reservoir slice's spike-rate over the token AND `c_t` = `cp_ssm_state[ssm_idx]`. At the QUERY step, read-out over `[h_t, c_t]`, train the read-out (delta) + the gate (forward-mode eligibility × **FIXED RANDOM FEEDBACK** `Bc` — no BPTT, no weight transport). Bridge washed to post-init between sequences. Task: `[KEY, filler×10, QUERY] → rule[KEY, QUERY]` (K=6). Arms: **res_only** (read-out over the spiking reservoir `h_t` only) · **res_plus_sel** (read-out over `[h_t, c_t]`).

## Result — 6-seed GO (accuracy; chance 1/6 = 0.167)

| seed | res_plus_sel | res_only | GO |
|---|---|---|---|
| 42 | 0.413 | 0.120 | GO |
| 43 | 0.360 | 0.133 | GO |
| 44 | 0.340 | 0.140 | GO |
| 100 | 0.387 | 0.167 | GO |
| 101 | 0.407 | 0.193 | GO |
| 102 | 0.387 | 0.213 | GO |

- **res_plus_sel mean ~0.382** (2.3× chance) vs **res_only mean ~0.161** (≈ chance) — on 6/6 seeds the spiking reservoir ALONE sits at chance (its fading Izhikevich memory cannot hold the distal KEY across 10 filler tokens), and adding the on-bridge selective channel lifts it to ~2.3× chance. The selective channel HOLDS the distal KEY on real spikes.
- The gate is trained fully transport-free (fixed random feedback, no BPTT) — the corrected discipline from the numpy coupling's adversarial-verify, applied here from the start.

## ⇒ interpretation

The selective-SSM long-range mechanism — validated in numpy (Rungs 1–4b), scaled (Rung-3 trajectory), coupled into the emergent generator (frozen + joint, transport-free, adversarially verified) — now runs on the **fully-spiking substrate, one brain**: a recurrent Izhikevich reservoir and the `cp_ssm_state` selective channel co-resident on ONE `SimulationBridge`, the selective channel lifting the spiking reservoir past a long-range conjunction its fading membrane memory cannot do. This is the emergence-bar realization of the coupling.

## Honest scope
- Synthetic gated-conjunction task (the tractable on-bridge long-range probe, as in Rungs 2/4b), not real-text LM (a per-token corpus on-bridge LM is far more bridge-stepping; the numpy joint result is the real-text validation, and Rung 4b-iii-a's byte-equivalence guarantees the selective channel transfers).
- Absolute accuracy ~0.38 (2.3× chance) at DEPTH=10 — the on-bridge spiking read (finite spike-count over `T_STEP=6`) is noisier than the numpy graded read; larger `T_STEP` / population averaging would raise it (the finite-spike-read cost, not a mechanism limit).
- The read-out argmax is host-computed here; the validated spiking FS-WTA read-out (`_reslm_spiking_readout`) is the drop-in to make the READ spiking too (a bounded follow-on).

## Next
- Wire the spiking FS-WTA read-out so the argmax is on spikes too (fully-spiking end-to-end).
- The real-text on-bridge LM (expensive; the numpy joint result + byte-equivalence already cover the mechanism).
- raw `research/findings/raw/_onbridge_couple/seed*.json`.
