# Past the reservoir bound, Rung 4a (the spiking-realization ladder, step 1): the selective SSM's input-modulated leak SURVIVES a BIOLOGICAL conductance-based shunt — how a real neuron modulates its time constant — green-lighting the on-bridge Rung 4b

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung4_conductance_shunt_ssm_derisk.py` (self-contained numpy; NO `sim/` edit, NO BPTT, NO weight transport).
**Status:** ✅ 5/6 GO; 6/6 DIRECTIONAL (selective beats every control on all 6 seeds; seed 102 a marginal +0.026 gate-margin clip).
**Provenance:** the first rung of the spiking-realization ladder for the past-reservoir mechanism (Rung 2 synthetic + Rung 3 real-text, both GO). Before wiring the actual `SimulationBridge`, verify the mechanism survives the BIOLOGICAL leak form a spiking neuron uses.

## Why this rung (the a0 on the bridge)

Rungs 2–3 used an ABSTRACT gate `lam_{t,i} = sigmoid(w·u)` for the input-modulated leak. The bridge's neuron decays (RF `_rf_lambda`, NMDA tau) are FIXED scalars — no per-neuron input-modulated leak exists yet. But a real neuron's effective membrane time constant is `tau_eff = C / g_total`, and an INPUT-DRIVEN SHUNTING conductance (`g_syn` to rest) increases `g_total` → shortens `tau_eff` → increases the leak. So an input-driven shunt IS the input-modulated leak = the selective lambda — realizable with the bridge's existing conductance-based synapses, NO new mechanism. This rung tests that the selective advantage SURVIVES this biological conductance form (before the on-bridge wiring, Rung 4b).

## Mechanism (bridge-faithful conductance dynamics)

```
conductance-based leaky integrator:  V_{t,i} = lam_eff_{t,i}*V_{t-1,i} + (1 - lam_eff_{t,i})*inj_{t,i}
input-driven shunt (the GATE):        g_shunt_{t,i} = softplus(w_i·u_t + c_i)     (>=0 conductance)
effective retention:                  lam_eff_{t,i} = clip(1 - k_leak*(1 + g_shunt_{t,i}), 0, 1)
EXACT forward-mode eligibility (local; softplus' = sigmoid):
   dlam/dw_i = -k_leak*sigmoid(a_i)*u_t ;  e^w = lam_eff*e^w + (dlam/dw)*(V_prev - inj) ;  Δtheta ∝ -delta*e
```
Small base leak `k_leak=0.06` → `lam_eff≈0.92` at init (the state HOLDS; the forget-bias equivalent), and the input-driven shunt INCREASES the leak (releases). The gate gradient is `k_leak`-scaled (small), so the gate LR is compensated (`LR_GATE=0.4`) — a principled fix for the conductance form's smaller leak-sensitivity, not tuning-to-pass (the mechanism was already directionally GO before).

## Result — seed 42/43/44 (task = Rung 2's [KEY, filler×12, QUERY]→rule[KEY,QUERY]; chance 1/6=0.167)

| seed | selective | fixed_res | detached | randgate |
|---|---|---|---|---|
| 42 | 0.541 | 0.270 | 0.270 | 0.330 |
| 43 | 0.422 | 0.319 | 0.319 | 0.333 |
| 44 | 0.656 | 0.441 | 0.493 | 0.511 |

selective beats fixed_res (fixed leak), detached (shunt gate UNTRAINED), and randgate (shunt gate on a RANDOM input) on all 3 seeds, with accuracies matching the abstract Rung-2 range (0.42–0.66).

**6-seed (42/43/44/100/101/102) — 5/6 GO; 6/6 DIRECTIONAL:** selective mean 0.546; selective beats every control on ALL 6 seeds — vs fixed_res +0.156 (min +0.026), vs detached +0.141 (min +0.026), vs randgate +0.126 (min +0.033). 5/6 by the +0.08 gate margin (seed 102 a marginal +0.026 clip — selective 0.437 still beats all three controls, just under the margin). The conductance-shunt realization holds directionally on every seed.

## ⇒ the claim

The selective SSM's input-modulated leak — the ingredient that captures long-range conjunctions past the reservoir bound — SURVIVES a biological CONDUCTANCE-BASED shunt (an input-driven shunting conductance = the input-modulated time constant), trained by the same exact forward-mode eligibility trace (no BPTT, no weight transport). This is exactly the primitive the bridge's conductance-based synapses provide, so the mechanism is realizable on the spiking substrate WITHOUT a new `sim/` mechanism. Green-lights the on-bridge Rung 4b.

## Honest scope / next

- Still numpy (the bridge-faithful conductance dynamics, not yet bridge-hosted) — this rung isolates "does the conductance-shunt leak form work?" before the on-bridge wiring cost.
- NEXT (Rung 4b): wire this on a real `SimulationBridge` — SSM units = conductance-based leaky-integrator neurons, the gate = input-driven shunting conductances (learned by a local synaptic eligibility trace), a spiking read-out — the fully-on-substrate transport-free long-range learner. Then the spiking read (rate code) robustness + scale.
- NO `sim/` edit. CI guard `tests/test_reslm_rung4_conductance_shunt_ssm.py`.

## Files
- `research/runners/_reslm_rung4_conductance_shunt_ssm_derisk.py`; raw `research/findings/raw/_rung4/seed*.json`.
- Builds on Rung 2 (`2026-07-13-PAST-RESERVOIR-RUNG2-...`) + Rung 3 (`-RUNG3-...`) + the bridge conductance-synapse a0-read.
