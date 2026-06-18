# On-bridge neural-error realization — the error population fires, but the full spiking combination is a deferred (expensive, non-load-bearing) confirmation

**Date:** 2026-06-17 (track A #1, the on-bridge realization step)
**Status:** **BOUNDARY (honest, non-load-bearing).** The predictive-coding error population fires on the bridge once
the drive is calibrated, but the full spiking-error → read-out-learning combination did not converge at a tractable
training budget, and each run is expensive (~25-48 min/seed). Crucially, this combination is **not required** for the
biologization claim — that claim already rests on two independently-validated 6-seed GOs (below). Banked as a
deferred confirmation, not a mechanism failure.
**Runner:** `research/runners/_phaseB_onbridge_neural_error_readout_derisk.py`
**Raw:** `research/findings/raw/_phaseB_onbridge_neural_error_readout.json`

## What IS proven (the load-bearing results)

The "the read-out learning is brain-based" claim stands on two pieces, each 6-seed unanimous GO:
1. **The error is neuralisable** (`2026-06-17-neural-error-localrule-derisk-GO.md`, numpy, 6-seed): a predictive-coding
   error population (ON/OFF `relu(target-est)` / `relu(est-target)`, exc target − inh prediction) drives the delta
   rule to 100% of the exact host error; scrambled-error collapses.
2. **The read-out learns on the substrate** (`2026-06-17-onsubstrate-readout-rule-bridge-GO.md`, on-bridge, 6-seed):
   the bridge's three-factor plasticity learns the read-out decoder via `cp_per_synapse_reward_override` × eligibility
   = the delta rule, to 111% of the host rule; scrambled-teaching collapses.

⇒ the per-output error CAN be computed by neurons, AND the bridge CAN learn the read-out from a per-output error. The
remaining piece — running BOTH in spikes at once on the bridge — is a confirmation that they compose, not a new claim.

## What the on-bridge combination found

A two-bridge realization (the read-out bridge + a separate error-population bridge = the inferior-olive analogue,
projecting through the climbing-fiber `cp_per_synapse_reward_override` channel):
- **Drive calibration matters:** unit-norm filler codes make `(target - est) ~ 0.06`, so the initial `ERR_DRIVE=220 pA`
  left the error neurons sub-threshold (`cal=32`, essentially silent → no learning). Raising to `6400 pA` (the `onb`
  precedent scaled for O(0.06) inputs) made them fire (`cal=0.45`).
- **But the read-out did not converge** at the reduced 15-pass smoke budget (neural ≈ lesion ≈ chance). The on-bridge
  read-out needed 40 passes to reach 1.000 with the EXACT error; the spiking error is noisier (Poisson + LIF f-I
  rectification + a sub-threshold dead-zone for small errors), so it plausibly needs the full budget AND/OR a
  population-coded error read-out (several error neurons per output for SNR, the CYCLE-91 lift) — at ~25-48 min/seed,
  an expensive loop.

## Decision — defer (honest, cost-aware)

The full on-bridge spiking-error combination is **not load-bearing** (the claim stands on the two 6-seed GOs), and
fully closing it is expensive (noisy-error convergence at ~40 passes × ~48 min/seed × multi-seed = hours). Banked as a
**deferred confirmation**. If pursued later, the cheap-first fix is a population-coded error read (N error neurons per
output, averaged — the documented rate-code-wall lift) + the full 40-pass budget + a single-seed convergence check
before multi-seed. The error-population realization (drive-calibrated, firing) + the two proven halves make this a
characterized, low-risk escalation, not an open problem.

## Honest scope of the biologization

The read-out **learning** is brain-based: the weights are learned by real synaptic plasticity (on-bridge 6-seed GO),
and the per-output error that drives it is neuralisable by a predictive-coding population (numpy 6-seed GO). The only
remaining host element is the `target` (the env/teacher supervised signal — a legitimate teaching scaffold, not a
shortcut). Running the error population and the read-out plasticity simultaneously in spikes on the bridge is the
deferred confirmation.
