# Past the reservoir bound, Rung 4b-ii (ON THE BRIDGE, 6-seed GO): a minimal additive `sim/` SLOW-SSM-STATE mechanism realizes the input-modulated-leak HOLD/RELEASE the raw membrane could not — the on-substrate core of the transport-free long-range learner

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_rung4b_ii_onbridge_slow_ssm_state_derisk.py`. **`sim/` edit:** ONE additive mechanism (`enable_selective_ssm_state`), config.py + bridge.py.
**Status:** ✅ 6-seed GO (identical across seeds — the mechanism is deterministic; no connectivity/noise). The FIRST `sim/` edit of this arc (all Rungs 1–4a were reuse-by-import).

## What this closes

Rung 4b-i (on-bridge, informative negative) showed the raw Izhikevich membrane leaks too fast to hold the SSM state, and the shunt is swamped by the intrinsic leak — the realization needs a SLOW per-neuron state where the input-modulated shunt is the DOMINANT leak. That is exactly a faithful biological mechanism (a dendritic-plateau / calcium graded integrator whose leak is set by an input-driven shunting conductance = a shunting-modulated membrane time constant), so per "the protected-module caution is anti-CHEAT, not anti-biology" it is a JUSTIFIED minimal additive `sim/` edit.

## The `sim/` mechanism (additive, default-off, byte-identical-when-off, verified)

- **`config.py`:** `enable_selective_ssm_state: bool = False` + `ssm_k_leak: float = 0.06`.
- **`bridge.py`:** three per-neuron arrays `cp_ssm_state` / `cp_ssm_inject` / `cp_ssm_shunt` (None by default); allocated in `_initialize_simulation_data` ONLY when the flag is set; a self-contained per-step block at the top of `_run_one_simulation_step`:
  ```
  if self.cp_ssm_state is not None:
      lam_eff = clip(1 - ssm_k_leak*(1 + cp_ssm_shunt), 0, 1)
      cp_ssm_state = lam_eff*cp_ssm_state + (1 - lam_eff)*cp_ssm_inject
  ```
  The block reads `cp_ssm_inject`/`cp_ssm_shunt` and writes `cp_ssm_state` — it touches NO other bridge state, so when the flag is OFF (`cp_ssm_state is None`) the block is skipped and the step is BYTE-IDENTICAL.
- **Verified:** with the flag off, a bridge builds with `cp_ssm_state is None`, steps identically, and 11/11 existing bridge/EMERGE tests pass (non-regression). Mirrors the `enable_two_compartment_dap` / `enable_htm_learning` additive pattern.

The runner sets `cp_ssm_inject` + `cp_ssm_shunt` each step (the world/body interface writing the drive + gate) and reads `cp_ssm_state` — the on-bridge realization of the selective-diagonal-SSM lambda (Rungs 1–4a, transport-free, real-text-validated).

## Result — on-bridge HOLD/RELEASE (6-seed GO)

Inject a value (steps 0–3), then filler (steps 4–15, inject=0), varying the filler shunt:

| arm | slow-state `s` after 12 filler steps |
|---|---|
| **HOLD** (low shunt → lam_eff ~0.94) | **0.362** |
| **RELEASE** (high shunt → lam_eff ~0.58) | **0.001** |

hold − release = 0.361 on all 6 seeds (identical, deterministic). The slow state HOLDS the injected value across the filler under low shunt and RELEASES it (→0) under high shunt — exactly the input-modulated leak the raw membrane could NOT do (Rung 4b-i: hold ≈ release ≈ 1 mV, both leaked). The mechanism math checks out: inject phase drives `s`→~0.76 (lam_inject~0.7 over 4 steps), HOLD retains `0.76·0.94^12 ≈ 0.36`, RELEASE `0.76·0.58^12 ≈ 0`.

## ⇒ the claim

The past-reservoir mechanism's core primitive — a per-neuron slow graded state with an INPUT-MODULATED LEAK (the selective λ) — now runs ON the spiking bridge as a minimal additive, byte-identical-when-off `sim/` mechanism (a faithful graded-integrator-with-shunting-modulated-time-constant). This unblocks the fully-on-substrate selective SSM (the transport-free long-range learner): the state is on-bridge; the remaining rungs wire the input-driven SHUNT as a synaptic conductance (input → shunt), the read-out, and the on-bridge eligibility-trace LEARNING of the gate.

## Honest scope / next

- This rung validates the STATE mechanism (hold/release). The gate (which sets the shunt from the token input) is still set by the runner; the on-bridge eligibility LEARNING of the gate weights (Rung 4b-iii) + a synaptic input→shunt pathway + the spiking read-out are the remaining rungs to the fully-on-substrate learner.
- ONE additive `sim/` edit (`enable_selective_ssm_state`), default-off, byte-identical-when-off, non-regression-verified.
- CI guard `tests/test_selective_ssm_state.py` (mechanism hold/release + byte-identity when off).

## Files
- `sim/config.py` (+flag/param), `sim/bridge.py` (+arrays/init/per-step block); `research/runners/_reslm_rung4b_ii_onbridge_slow_ssm_state_derisk.py`; raw `research/findings/raw/_rung4bii/`.
- Follows Rung 4b-i (`2026-07-13-PAST-RESERVOIR-RUNG4b-i-...`) + Rungs 1–4a.
