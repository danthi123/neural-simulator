# Past the reservoir bound, Rung 4b-i (on-bridge, informative negative): the raw Izhikevich membrane leaks too fast to hold the SSM state — the input-modulated shunt is swamped by the intrinsic leak → the realization needs a SLOW per-neuron state (the precise next `sim/` mechanism)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung4b_onbridge_shunt_primitive_derisk.py` (real `SimulationBridge`, numpy; NO `sim/` edit).
**Status:** Informative NEGATIVE — the naive membrane realization does NOT hold; it MAPS the exact next mechanism (a slow-state input-modulated leak). Per the mission, an honest negative that launches the next mechanism is a first-class deliverable.

## What was tested

Rung 4a validated the selective SSM's input-modulated leak with a biological conductance-shunt, and the a0 established the on-bridge path: a sub-threshold neuron's membrane `V` is a graded leaky integrator toward `V_rest`, and an inhibitory conductance `g_i` with reversal `E_i = V_rest` is a PURE SHUNT (`g_i·(V_rest − V)`, a leak term) — so a low `g_i` should HOLD the integrated value and a high `g_i` RELEASE it. This rung tests that primitive on a real bridge: inject (raise V), then filler, then vary `g_i` — HOLD (low `g_i`) vs RELEASE (high `g_i`).

## Result — the membrane leaks too fast (the shunt is swamped)

Mean membrane above rest, per step (inject steps 0–3, filler 4–15):
```
HOLD    (g_i=0.02): 1.5 2.7 3.8 4.8 | 4.2 3.8 3.4 3.0 2.6 2.3 2.1 1.8 1.6 1.4 1.3 1.1
RELEASE (g_i=0.8):  1.5 2.7 3.8 4.8 | 4.2 3.7 3.3 2.9 2.5 2.2 1.9 1.7 1.5 1.3 1.1 1.0
```
- **The intrinsic membrane leak dominates:** V decays 4.8 → ~1.1 mV over 12 filler steps REGARDLESS of the shunt — the Izhikevich membrane time constant is far too fast to HOLD a value across the filler.
- **The shunt is swamped:** g_i=0.8 (RELEASE) barely changes the trajectory vs g_i=0.02 (HOLD) — end 1.0 vs 1.1 mV. The intrinsic leak already dominates, so the input-modulated shunt is NOT the controlling leak term. Not a GO (hold−release ≈ +0.1 mV).

## ⇒ the precise next mechanism (the honest negative → the launch)

For the on-bridge selective SSM to work, the state must be a **SLOW per-neuron leaky integrator** where (a) the intrinsic leak is slow enough to HOLD across the filler, and (b) the input-modulated SHUNT is the DOMINANT leak term (so the learned gate controls the retention). The raw Izhikevich membrane is neither. Two biology-grounded realizations map the exact next step:

1. **A minimal additive `sim/` mechanism (justified per the anti-cheat-not-anti-biology rule):** a per-neuron SLOW leaky-integrator state (a dendritic plateau / calcium-like graded variable, slow base decay) whose leak is modulated by an input-driven shunt conductance — the faithful "graded integrator with shunting-modulated time constant." Additive, default-off, byte-identical when off. This is the cleanest realization (a true linear SSM state with an input-modulated leak).
2. **A slow existing conductance as the state:** the slow NMDA conductance (tau ~150 ms, the D3 register's persistent-slot substrate) HOLDS across the filler; the open piece is making its decay input-MODULATED (a shunt on the slow state) — the same input-modulated-leak requirement, on the slow conductance.

The mechanism is validated (Rung 4a); the on-bridge realization needs a slow state, which the raw membrane isn't — a clean, informative substrate boundary, not a wall.

## Honest scope / next

- This is the FIRST on-bridge rung; it correctly falsified the naive membrane-as-SSM-state realization + localized WHY (intrinsic leak too fast, shunt swamped). Rung 4b-ii = the slow-state realization (the minimal additive `sim/` slow-integrator-with-input-modulated-leak, OR the modulatable-slow-conductance route), then the on-bridge eligibility-trace learning.
- The mechanism itself (Rungs 1–4a) stands: a transport-free, real-text-validated selective SSM, spiking-realizable via conductance-shunt — the on-bridge realization just needs the correct (slow) state variable.
- NO `sim/` edit in this rung (the finding IS the map to the justified next edit).

## Files
- `research/runners/_reslm_rung4b_onbridge_shunt_primitive_derisk.py`. Follows Rung 4a (`2026-07-13-PAST-RESERVOIR-RUNG4a-...`).
