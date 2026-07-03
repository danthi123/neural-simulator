# EMERGE-70 — the whole flagship EMERGE conversation CO-EXECUTES **FULLY-SPIKING IN ONE PROCESS** (the master-directive ONE BRAIN step) — **GO** (3-seed)

**2026-07-03 (autonomous).** EMERGE-69 wired the fully-spiking A→W render into the flagship console and shipped it GO, but named an honest residual constraint (`2026-07-03-emerge69-console-fully-spiking-GO.md` line 11): *"`sim.bridge` binds one backend at import (module-global `cp`); the numpy EMERGE-52/54 reasoner and the cupy A→W read-out cannot co-execute in one process; validated component-wise."* EMERGE-70 PROBES that residual head-on: **can the EMERGE reasoner + the cupy A→W read-out co-execute fully-spiking in ONE process?** The honest answer is **GO — and the residual was TINY** (3 host→device write lines fixable with a one-liner), so the whole flagship now runs fully-spiking in one process.

## The question (cheap-first, mechanical)

Under `SIM_BACKEND=cupy`, can the SAME process run BOTH (1) the EMERGE-52/54 **reasoner** (`PerDimensionConsole` — the stacked HTM pooler discovers a taxonomy → multi-level inheritance + per-dimension Collins-Quillian cancellation + no-confab moat) AND (2) the EMERGE-67/68 **A→W read-out** (`UnifiedNeuralSpell` — content on BRIDGE-A, function on BRIDGE-F, decoded from `language_output` spikes), so a full flagship turn (reason → gate decision → spiking A→W render) co-executes fully-spiking in one process?

## Why the constraint existed — the PRECISE residual (cited, quantified)

`sim.backend.get_backend()` is a full numpy/cupy abstraction with `from_host`/`to_host` marshals — but the reasoner's **on-bridge writes don't route through it**. They use `xp = bridge.xp if hasattr(bridge, "xp") else np`, and **`SimulationBridge` has NO `.xp` attribute** (grep `self\.xp`/`\.xp` over `sim/bridge.py` → zero hits), so `xp` is **always numpy**. Under cupy that assigns a HOST numpy array into a DEVICE cupy `cp_*` array, which cupy rejects:

```
ValueError: non-scalar numpy.ndarray cannot be used for fill      (== EMERGE-69's named error, confirmed verbatim)
```

The reasoner uses **only two committed helpers** that touch the bridge, so the whole residual is **3 distinct host→device write LINES** (probe verified — `--probe` reports the crash unshimmed):
- `_emerge14_stageC_onbridge_learning_derisk.apply_kernel_update:115` — the teaching write `bridge.cp_connections.data[:] = <numpy>`
- `_emerge12_stageB2_bridge_tm_derisk._prime_from_winners:203,204` — the inference priming writes `cp_prev_firing_states[:]` / `cp_external_input_current[:] = <numpy>`
- `_emerge12_stageB2_bridge_tm_derisk.reset_soma:155` — `cp_membrane_potential_v[:] = <numpy>` (invoked by `_prime_from_winners`)

Not on the residual: `_clear_apical`/`reset_state` use `xp.float32(scalar)` (a **scalar** fill — cupy accepts it); `present_and_predict`/`reset_state` are **not** on the reasoner's read path (`_drive → _prime_from_winners` only); the EMERGE-44 `_competitive_pool` the reasoner imports is **pure host numpy** (no bridge writes) so it is already backend-agnostic. **⇒ the entire residual is 3 write lines in 2 research-runner helpers.**

The fix is the ONE-LINER `from_host(...)` per site (numpy passthrough ⇒ **byte-identical**; cupy H→D copy) — **exactly** the fix EMERGE-69 already used in `_emerge61._restore_state` ("a backend-compat bug found + SURPASSED with a byte-identical fix"). Not a wall — a 3-line SURPASS.

## The probe (cheap-first, one variable at a time; committed runners UNTOUCHED)

The `from_host` fix is applied as a **probe-scoped monkeypatch shim** (`install_from_host_shim`) of the two research-runner helpers — so the committed numpy-default runners stay byte-identical; folding `from_host` into the committed helpers is the trivial follow-on (**EMERGE-71**, byte-identical on numpy so all EMERGE CI stays green).

- **(a) reasoner-on-cupy UNSHIMMED** → builds fine (bridge init OK, 484 neurons, 32320 synapses) but **crashes at the first teach** with the named `non-scalar numpy.ndarray cannot be used for fill` — the constraint is real (confirmed every seed).
- **(b) reasoner-on-cupy WITH the `from_host` shim** → runs on cupy; its EMERGE-54 answers are **IDENTICAL to the numpy reference** (per-dimension cancellation + inheritance + sibling-discrimination + moat).
- **(c) ONE-PROCESS CO-EXECUTION** → build the reasoner (shimmed, cupy) AND `UnifiedNeuralSpell` (cupy, BRIDGE-A + BRIDGE-F caches) **in the same process**; run a full flagship turn: **reason [cupy] → gate decision → spiking A→W render [cupy]**.
- **(d) gate-first MOAT** → an ABSTAIN (unknown token) invokes the A→W read-out **0 times** — the render is never reached — so the no-confab moat holds by construction, unchanged.

## De-risk — **GO** (3-seed 42/43/44, GPU/cupy, 99.6 s)

| gate | value | bar |
|---|---|---|
| unshimmed crash == the named residual (`non-scalar … cannot be used for fill`) | **1** every seed | 1 |
| reasoner-on-cupy answers == numpy reference (per-dim cancel + inherit + sibling + moat) | **1** every seed | 1 |
| co-execute a full flagship turn in ONE cupy process (reason → spiking render) | **1** every seed | 1 |
| gate-first MOAT — A→W spell calls on ABSTAIN | **0** every seed | 0 |

Sample turn (ONE cupy process, both bridges co-resident):
- `can a penguin fly?` → reasoner **"No, a penguin walks"** (LOCOMOTION overridden) → spike-render **"the penguin walks"**
- `can a penguin breathe?` → reasoner **"Yes"** (RESPIRATION inherited — the EMERGE-54 fix) → spike-render **"the penguin can breathe"**
- `can a zzz fly?` → **ABSTAIN** → A→W read-out **never invoked** (0 spell calls) — moat by construction

## Verdict

**GO.** The **whole flagship EMERGE conversation co-executes fully-spiking in ONE process** — the master-directive ONE BRAIN step. Under `SIM_BACKEND=cupy`, the same process runs BOTH the EMERGE-52/54 reasoner (structure discovery + multi-level inheritance + per-dimension cancellation + moat) AND the EMERGE-67/68 A→W read-out (content on BRIDGE-A, function on BRIDGE-F, decoded from `language_output` spikes), and a full turn co-executes (reason → gate → spiking render). The reasoner's answers on cupy are byte-for-byte the numpy reference; the gate-first no-confab moat holds by construction (0 A→W invocations on abstain). **The constraint EMERGE-69 named is RESOLVED**: the residual was 3 host→device write lines in 2 research-runner helpers that bypassed the `sim.backend` abstraction (used `xp=bridge.xp`-or-`np` where the bridge has no `.xp`), each fixed by the one-liner `from_host` — the same fix EMERGE-69 used in `_emerge61`. Applied here as a probe-scoped shim (committed runners untouched); **NO `sim/` edit**. 3-seed.

⇒ structure discovery → reasoning → spiking render all co-execute **fully-spiking in ONE process** on cupy. The emergent brain now discovers categories from experience → reasons → and speaks its grounded answers on spikes **all in one process** — the true one-brain milestone for the EMERGE conversation.

## The trivial follow-on (EMERGE-71)

Fold `from_host` into the two committed helpers (`_emerge14.apply_kernel_update` + `_emerge12.reset_soma`/`_prime_from_winners`) directly — byte-identical on numpy, so all EMERGE-9..69 numpy CI stays green (verified: EMERGE-68/69 CI 11 passed / 2 GPU-skip on numpy, unchanged by the probe). That retires the shim and makes the reasoner cupy-clean permanently, so the flagship console (EMERGE-60 `SpikingBrocaConsole` with a live reasoner + the `neural_spell` A→W) can run its whole turn under one `SIM_BACKEND=cupy` process with no probe scaffolding.

## Files
- `research/runners/_emerge70_one_brain_single_backend_probe_derisk.py` — the probe: `install_from_host_shim` (the probe-scoped `from_host` fix of the 2 committed helpers, byte-identical on numpy), `_build_reasoner` (reasoner on cupy), `_emerge_turn`/`_one_process_coexecute` (the full flagship turn: reason → gate → spiking A→W render), `_probe_one`/`_derisk`; `--probe`/`--derisk`.
- `tests/test_emerge70_one_brain_single_backend.py` — 5 CPU-safe (residual documented / shim installs+uses `from_host` / `from_host` byte-identical on numpy / gate-first-moat turn logic / numpy-reference == EMERGE-54 ground truth) + 1 GPU smoke (skip unless the process is `SIM_BACKEND=cupy` and both caches exist; PASSED on GPU).
- `research/findings/raw/_emerge70_one_brain_single_backend.json` — the 3-seed de-risk (go=true; named_residual_confirmed=true).
- Caches reused verbatim: `bridges/emerge67_aw/aw_content.simstate.h5` (BRIDGE-A), `bridges/emerge68_aw/aw_func.simstate.h5` (BRIDGE-F) — local, `.h5` gitignored, regenerable via EMERGE-67/68 `--train`.
