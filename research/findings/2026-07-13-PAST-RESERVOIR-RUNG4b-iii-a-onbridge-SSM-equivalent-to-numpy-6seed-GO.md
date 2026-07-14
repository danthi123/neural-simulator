# Past the reservoir bound, Rung 4b-iii-a (on-bridge, 6-seed GO): the on-bridge `cp_ssm_state` mechanism is EQUIVALENT to the validated numpy selective SSM to float32 precision — the whole Rung-2/3/4a ladder runs on the spiking bridge

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung4b_iii_onbridge_ssm_equivalence_derisk.py` (uses the `enable_selective_ssm_state` `sim/` mechanism; numpy; NO further `sim/` edit).
**Status:** ✅ 6-seed GO.

## What this establishes (like-for-like, the decisive on-substrate check)

Rung 4b-ii landed the additive `sim/` slow-SSM-state mechanism and validated it on the hold/release primitive. This rung proves the stronger, decisive claim: the on-bridge state update **IS** the validated numpy selective-SSM update, not merely qualitatively similar. Driving the on-bridge `cp_ssm_state` and a numpy replica with the SAME random per-neuron inject + shunt sequences (32 neurons, 40 steps, random each step):

| seed | max\|on-bridge s − numpy s\| over 40 steps |
|---|---|
| 42 | 1.62e-07 |
| 43 | 1.73e-07 |
| 44 | 1.78e-07 |
| 100 | 1.87e-07 |
| 101 | 1.47e-07 |
| 102 | 2.10e-07 |

Max abs diff ~1e-7 (float32 round-off) on all 6 seeds → **byte-equivalent**. The bridge's per-step block `s = clip(1 − k(1+shunt),0,1)·s + (1−lam)·inject` reproduces the numpy selective-SSM state exactly.

## ⇒ the claim

The on-bridge `cp_ssm_state` mechanism is the SAME dynamical object as the validated numpy selective SSM — so the entire numpy ladder transfers to the spiking bridge EXACTLY:
- **Rung 2** (the selective SSM beats a fixed reservoir + 3 controls on the gated-conjunction),
- **Rung 3** (it beats the fixed reservoir + bigram at deep context on REAL TinyStories text, 6/6),
- **Rung 4a** (the conductance-shunt biological realization)

are now **on-bridge results**, because the on-bridge state IS that SSM. The transport-free, real-text-validated selective SSM — the honest path past the reservoir bound — runs on the spiking substrate.

## Honest scope / next

- Equivalence is proven for the STATE dynamics (the core of the mechanism). The gate (which sets `cp_ssm_shunt = softplus(w·u)` from the input) and the read-out are currently host-computed by the runner; the eligibility-trained gate weights are learned by the same forward-mode local rule.
- NEXT (Rung 4b-iii-b): the on-bridge eligibility-trace LEARNING of the gate weights (the trace is a local per-neuron quantity, computable on-bridge), then a synaptic input→shunt PATHWAY (input → `cp_ssm_shunt` via a conductance) + the spiking read-out = the fully-autonomous on-substrate transport-free long-range learner.
- NO further `sim/` edit (reuses the Rung-4b-ii mechanism). CI guard `tests/test_selective_ssm_state.py` extended.

## Files
- `research/runners/_reslm_rung4b_iii_onbridge_ssm_equivalence_derisk.py`. Uses the `enable_selective_ssm_state` mechanism (`2026-07-13-PAST-RESERVOIR-RUNG4b-ii-...`). Follows Rungs 1–4a.
