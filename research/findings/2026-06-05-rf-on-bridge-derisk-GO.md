---
type: finding
status: contributing
date: 2026-06-05
mechanism: fhrr
---

# Resonate-and-fire ON the SimulationBridge — minimal de-risk → GO — 2026-06-05

**Verdict: GO. The SimulationBridge can natively host resonate-and-fire (RF) phasor neurons that compute FHRR
composition at parity with the numpy reference.** The owner-FUNDED FHRR-on-bridge pivot (Option A; the structural
escape from the opponency rate-coded SNR wall, `2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`) is
de-risked. The first protected `sim/` edit of the consolidation arc is in, minimal, additive, and regression-free.

## What was built (the first protected `sim/` edit — additive, guarded, flagged)
Per `docs/plans/2026-06-05-rf-on-bridge-derisk-design.md`:
- **`sim/enums.py`:** `+ NeuronModel.RESONATE_AND_FIRE` (1 enum value).
- **`sim/bridge.py`:** `+` one `elif` branch in `_run_one_simulation_step` (after AdEx) that holds a complex state
  `Z = re + i·im` (reusing `v=re`, `u=im`), rotates it each step by `exp(λ+iω)` and fires on the upward Im
  zero-crossing (the spike step encodes the kick's phase); `+ rf_kick()` / `rf_read_phases()`. The Izhikevich /
  HH / AdEx branches are **byte-unchanged**; the RF branch is inert unless `neuron_model_type==RESONATE_AND_FIRE`.
  Ported verbatim from `research/runners/resonate_fire_fhrr.py` (Izhikevich 2001; Frady & Sommer 2019, PNAS).

## The three TDD gates — all GO (`tests/test_rf_on_bridge.py`, 3/3 pass)
1. **Phase readout:** kick 10 phases → recovered as spike timing at mean circular error **< 0.02**.
2. **Bind / unbind / bundle:** `phasor_a·phasor_b` → phase(a)+phase(b); `phasor_c·conj(a)` → phase(c)−phase(a);
   `Σ phasor` → phase of the complex sum — all at circular error **< 0.03** on the bridge.
3. **The composer task (the de-risk GATE):** the project's compositional task (vocab 8×8, loads 2/3/5, frozen bar
   0.80) with EVERY resonate routed through the bridge's RF step.

### De-risk GATE result (`_rf_on_bridge_probe.py`, D=256, 15 trials, period=1000)
| load | accuracy | groundable sim min | ungroundable sim max | abstention separates |
|---|---|---|---|---|
| 2 | **1.0000** | 0.594 | 0.127 | ✅ |
| 3 | **1.0000** | 0.439 | 0.130 | ✅ |
| 5 | **1.0000** | 0.339 | 0.123 | ✅ |

**Perfect accuracy at every load with clean abstention — at PARITY with the numpy reference** (spiking-phasor
self-test: 1.0/1.0/1.0, g_min 0.597/0.454/0.303). The bridge's own neurons, in the bridge's own step, compute the
FHRR bind→bundle→unbind→cleanup chain identically. The opponency simply does not exist in this algebra.

## Regression (the protected edit is clean)
- Composer `test_recovery_rate_clears_frozen_bar` PASSES on GPU **with** the edit (unchanged).
- `tests/test_determinism.py` PASSES.
- The composer's numpy-backend 0/6 is the **documented spiking-bind degeneracy** (identical on the clean tree with
  the edit stashed) — NOT this edit.

## Honest scope / what remains (the full-feature arc)
This de-risk proves the **RF dynamics + phase readout + the full composition chain** run on the bridge's neurons.
It does NOT yet:
- realize the **complex-synapse bind** (here the kick `phasor_a·phasor_b` is computed and injected via `rf_kick`;
  the production bind would have the synapse carry the operand phasor — the next feature layer);
- recode the **production composer** (`core_sim_composition.py` / `brain_conversational_agent.py`) from rate codes
  to phase/timing, nor re-validate the full capability matrix (who/what/abstain/negation/clauses/dialogue) on it.

Those are the months-scale full FHRR-on-bridge feature (separate design → plan → subagent-driven). **This de-risk
clears the gate to start it:** the bridge substrate is proven; the representation is proven (perfect at all loads);
the only remaining risk is engineering (the recoding + re-validation), not feasibility.

## Artifacts
- `sim/enums.py`, `sim/bridge.py` (protected edits, commit bbd62ce8 + this commit)
- `tests/test_rf_on_bridge.py` (3 gates), `research/findings/raw/_rf_on_bridge_probe.py` (+ `.json`)
- design `docs/plans/2026-06-05-rf-on-bridge-derisk-design.md`; reference `research/runners/resonate_fire_fhrr.py`
- Backend: CPU (numpy) for the RF probe (the RF rotation is backend-agnostic; GPU verified for the composer
  regression). Frozen bars / no-confab moat untouched.
