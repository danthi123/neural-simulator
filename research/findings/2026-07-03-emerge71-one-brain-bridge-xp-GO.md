# EMERGE-71 — the TRUE ONE BRAIN, production-clean: `SimulationBridge.xp` retires the shim so the flagship reasoner runs on cupy with NO scaffolding — **GO**

**2026-07-03 (autonomous).** EMERGE-70 proved the whole flagship can co-execute fully-spiking in one process, via a probe-scoped `from_host` shim on 3 host→device write lines. EMERGE-71 replaces that shim with the **single, general, design-intended fix**: the reasoner's on-bridge helpers were written as `xp = bridge.xp if hasattr(bridge, "xp") else np` — expecting a `bridge.xp` backend accessor that `SimulationBridge` never had (grep: 0 hits), so `xp` always fell back to numpy → a host array assigned into a device `cp_*` array → `ValueError: non-scalar numpy.ndarray cannot be used for fill` on cupy. Adding the missing accessor fixes **all** such sites at once — no per-write shim.

## The fix (one additive property)

`sim/bridge.py` — a byte-identical `@property xp` on `SimulationBridge` returning the module-global active backend `cp` (cupy on GPU / numpy on CPU — the module the bridge's `cp_*` arrays live in):

```python
class SimulationBridge:
    @property
    def xp(self):
        return cp   # the active backend module (cupy / numpy)
```

- **On numpy:** `cp` IS numpy, so `bridge.xp` == the prior `else np` fallback → **byte-identical**.
- **On cupy:** `bridge.xp` == cupy → `xp.asarray(host_val)` yields a device array → the assignment-fill is device-correct.

This is the ONLY `sim/` edit — additive, guarded by nothing (a pure accessor), completing the backend abstraction the helpers already assumed. It fixes the ~7 `xp = bridge.xp if hasattr(bridge,"xp") else np` sites in `_emerge12`/`_emerge14` in one place (vs the EMERGE-70 shim's 3 per-write patches).

## Verification (controller-direct)

- **Byte-identical on numpy:** the EMERGE reasoner + on-bridge learning + determinism CI = **22 passed** unchanged; `test_regions` = **4 failed / 38 passed both WITH and WITHOUT** the edit (identical — the 4 are pre-existing cupy-path failures, a *different* class [host numpy arrays into cupy kernels] that this accessor does not touch; `sim/` was unedited across all of EMERGE-56..70, so they predate this work — flagged as a separate task).
- **Fixes cupy co-execution WITHOUT the shim:** `SIM_BACKEND=cupy` — the reasoner (`PerDimensionConsole`) builds + teaches + answers on cupy, byte-identical to the numpy reference: `ask_can(penguin, fly)` → **"No, a penguin walks."**, `ask_can(owl, fly)` → **"Yes, an owl can fly."** No `ValueError`; `bridge.xp is cupy` on the cupy backend.
- **`bridge.xp is numpy` on the numpy backend** + `hasattr(bridge, "xp")` True — the accessor the helpers expected.

## Verdict

**GO.** With the `bridge.xp` accessor, the flagship reasoner (EMERGE-52/54) runs natively on cupy, co-resident with the EMERGE-67/68 A→W read-out, so the **whole flagship EMERGE conversation runs its turn in one `SIM_BACKEND=cupy` process with NO scaffolding** — structure discovery + reasoning + fully-spiking render, one brain, one backend, one process. Byte-identical on numpy (CPU portability preserved). The EMERGE-70 probe shim is retired (superseded by this general fix). The gate-first moat is untouched. ⇒ the master-directive ONE BRAIN, production-clean.

## Files
- `sim/bridge.py` — the additive `SimulationBridge.xp` property (the only `sim/` edit).
- Honest note: `tests/test_regions.py` has 4 **pre-existing** cupy-path failures (host arrays into cupy kernels — a distinct class), unrelated to this accessor and unchanged by it; tracked as a separate task.
