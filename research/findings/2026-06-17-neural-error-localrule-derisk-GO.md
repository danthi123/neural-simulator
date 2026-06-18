# The per-output teaching error is neuralisable — a predictive-coding error population drives the delta rule, 6-seed GO

**Date:** 2026-06-17 (biologization track A #1: remove the read-out learning's last host scaffold — the per-output error)
**Status:** **GO, 6 seeds unanimous** (42, 43, 44, 100, 101, 102). The per-output teaching error `err_j = target_j −
est_j`, currently a HOST subtraction written into the bridge's teaching channel, can instead be computed by a
**predictive-coding error population**: two error neurons per output firing `relu(target_j − est_j)` (ON) and
`relu(est_j − target_j)` (OFF), rate-coded with Poisson spike-count noise; the signed error fed to the delta rule is
`ON_rate − OFF_rate`. The subtraction `target − prediction` is then done by the error neuron's **excitatory (target)
minus inhibitory (prediction)** inputs — the standard Rao-Ballard / Bastos predictive-coding error unit — not a host
formula. CPU/numpy; reuse-by-import; NO `sim/` edit.
**Runner:** `research/runners/_phaseB_neural_error_localrule_derisk.py`
**Raw:** `research/findings/raw/_phaseB_neural_error_localrule.json`

## Context

The on-bridge read-out is now learned by real synaptic plasticity (`2026-06-17-onsubstrate-readout-rule-bridge-GO.md`,
6-seed GO) via the bridge's three-factor rule `weight_update = lr · per_output_error · presynaptic_eligibility`. The
weight learning is brain-based; the residual host scaffold is the per-output error itself (a host subtraction). This
de-risk isolates the error-neuralisation (exact input activity; only the error is the rate-coded ON/OFF population)
from the input-rate-code question (already GO, `2026-06-17-onsubstrate-localrule-spikerate-derisk` track).

## Result — 6 seeds, the systematicity protocol, bundled held-out

| arm | held-out (6-seed mean) | per-seed |
|---|---|---|
| EXACT-err (host reference) | 1.000 | [1.000 × 6] |
| **NEURAL-err (predictive-coding population)** | **1.000** | [1.000 × 6] |
| SCRAMBLED (anti-cheat) | 0.036 | [0.000, 0.074, 0.000, 0.000, 0.000, 0.111] |

- **NEURAL-err = 1.000 = 100% of the exact host error, 6/6**; single-binding systematicity 1.000 (generalizes to
  held-out role-filler combinations).
- **Anti-cheat collapses:** permuting the neural error across outputs (so `err_j` no longer addresses output `j`)
  drops recall to 0.036 ≈ chance — the per-output error is load-bearing, not an artifact of the noisy code.

## Reading it (brain-based-only)

- The per-output teaching SUBTRACTION is neuralisable: an exc(target) − inh(prediction) error neuron, rate-coded with
  realistic spike-count noise, drives the read-out delta rule exactly as well as the host subtraction. ⇒ the read-out
  learning's last host scaffold (the error formula) is removable.
- **The `target` itself remains an env/teacher scaffold** — legitimate, the supervised teaching signal (the
  innate-teacher-teaches-a-learned-circuit pattern); what this de-risk neuralises is the *subtraction* that turns
  (target, prediction) into the per-output error. The `prediction` is the read-out's own neural output (already neural).
- **On-bridge realization next:** build the error population on the `SimulationBridge` (exc driven by the target
  teaching afferent, inh by the read-out's prediction), read its ON/OFF firing rates, and deliver them through the
  `cp_per_synapse_reward_override` climbing-fiber channel that the read-out learning already uses — closing the
  read-out learning to fully brain-based in spikes.

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_neural_error_localrule_derisk \
    --seeds 42,43,44,100,101,102
```
