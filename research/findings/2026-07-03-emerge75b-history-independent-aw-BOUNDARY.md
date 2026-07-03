# EMERGE-75b — the EMERGE-61 substrate wash-out on the A→W read-out does NOT close the EMERGE-75 regression (it makes it WORSE) — **BOUNDARY** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge75b_history_independent_aw_derisk.py`
**Test:** `tests/test_emerge75b_history_independent_aw.py`
**Raw:** `research/findings/raw/_emerge75b_history_independent_aw.json`
**Verdict:** BOUNDARY — the proposed fix is refuted by its own de-risk. The wash-out did NOT close the regression; it INCREASED it (hi-OFF regress 2 → hi-ON regress 25). The load-bearing control held (hi-OFF reproduces the boundary) and the moat is intact (0 spell/producer invocations on abstains), but the hypothesis's REMEDY fails.

## What was attempted

EMERGE-75 returned a small BOUNDARY: 3 full-render surfaces regressed vs the token spell, on seed 102 only (isolated
decode 16/16, moat intact). The diagnosis (from the raw data: isolated 16/16, only the deepest-history seed regresses)
was that the EMERGE-61 Izhikevich slow-adaptation current `cp_recovery_variable_u` ACCUMULATES across the shared 6-seed
render loop's ~1000+ sequential pool-drives, leaking into the A→W READ path (a static audit confirmed the ORDER path
already inherits EMERGE-61's wash-out; the A→W read path was the one un-washed spiking sub-system). EMERGE-75b applied
EMERGE-61's substrate wash-out to the A→W decode: snapshot each concept-pool bridge's post-load dynamic state
(`cp_membrane_potential_v`/`cp_recovery_variable_u`/conductances/STP/firing) and HARD-RESTORE it before every `_decode`,
so each decode would start from identical substrate state regardless of render depth. Toggleable (`_hi_enabled`) so ONE
engine yields both the un-washed EMERGE-75 baseline (hi-OFF) and the fix (hi-ON) — the causal control.

## The result — the fix FAILS (6 seeds 42/43/44/100/101/102, GPU)

| condition | total regress (6 seeds) | per-seed |
|---|---|---|
| **hi-OFF** (un-washed, == EMERGE-75) | **2** | `[0, 0, 0, 0, 0, 2]` — only seed 102 (deepest history) regresses |
| **hi-ON** (the wash-out fix) | **25** | `[0, 7, 4, 4, 4, 6]` — the wash-out turned 4 clean seeds into regressing |

The wash-out **increased** regressions ~12× and turned four clean seeds (43/44/100/101) into regressing ones. Other
gates: all-word ground-truth 0.979, overflow decode 0.938 (slightly DOWN from EMERGE-75's 1.000 — the wash-out also
mildly hurt the isolated decode), overflow-lesion 0.146 (still genuinely spiking), MOAT **0 / 0** on abstains (intact).

## Why the fix fails (the honest re-diagnosis)

The hard restore targets the **post-BUILD** state — but that is NOT the state a normal decode reads from. A normal decode
(`drive_pool_and_read_lang_output`) first zeros the external current and runs a **50-step settling reset** from the
CURRENT state. Restoring to the post-build state (which carries firing/conductance/`u` from the checkpoint-load, an
UNSETTLED operating point) and then running the 50-step reset yields a DIFFERENT (worse) trajectory than the natural
shallow-history path. So the natural un-washed path (hi-OFF) is actually FINE on 5/6 seeds; the hard-restore-to-post-build
is the WRONG reset target and injects regressions. **This partly refutes the accumulation-remedy hypothesis:** if
un-washed accumulation were the dominant cause, washing would help — instead it hurts, so the ~2-3-render seed-102
residual is a subtler deep-history effect, not simply cured by resetting to the post-build snapshot.

## Honest status + the named next mechanism (deferred, LOW priority)

- The **EMERGE-75 boundary stands** as a tiny honest residual: ~2-3 renders on the deepest-history seed only, out of ~80
  renders/seed; isolated decode ≈ 16/16; the no-confab MOAT intact. It is a bounded-render-inventory polish, **not** a
  capability wall.
- The wash-out-to-post-build remedy is **refuted** (worse). The named next hypotheses (deferred below the EMERGE-78
  frontier, since this is render-polish not a capability gap): snapshot a **SETTLED quiescent** state (run N zero-input
  settling steps AFTER build, THEN snapshot — a better reset target than post-build), or reload the bridge between seeds,
  or a per-decode 2-stage read calibration (EMERGE-77's Turrigiano-style per-pool bias). None is on the critical path.
- The MOAT is untouched; NO `sim/` edit; the EMERGE-75 render remains usable (5/6 seeds clean, moat safe).

## Files
- `research/runners/_emerge75b_history_independent_aw_derisk.py` — the wash-out install (`_install_history_independence`)
  + `UnifiedHistIndepSpell75` (toggleable) + the 6-seed hi-OFF/hi-ON de-risk.
- `tests/test_emerge75b_history_independent_aw.py` — CPU wash-out toggle/snapshot tests (the mechanism is CPU-testable; the
  on-spikes A→W is GPU-only; the full GPU de-risk returned this BOUNDARY).
- `research/findings/raw/_emerge75b_history_independent_aw.json` — the 6-seed result.
