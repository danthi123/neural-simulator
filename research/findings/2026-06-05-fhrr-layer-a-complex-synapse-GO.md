---
type: finding
status: contributing
date: 2026-06-05
mechanism: fhrr
---

# Full FHRR-on-bridge — layer (a) complex-synapse bind → GO — 2026-06-05

**Verdict: GO. FHRR bind / unbind / bundle / superposition all happen THROUGH complex synapses on the
SimulationBridge** — no external `rf_kick` injection, no numpy. Layer (a) of the owner-greenlit full FHRR-on-bridge
feature (`docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md`) is done. The months-arc's hard part —
phase-dependent synaptic transmission for FHRR (Frady-Sommer 2019) — works on the bridge.

## What was added (the protected `sim/` edit — additive, guarded, flagged)
- **`sim/bridge.py` RF branch:** `+` the complex synaptic input `u_i = Σ_j W_ij·z_j` (complex matvec from the
  PRESYNAPTIC RF states `z=(re,im)`, added to the rotated state each step). Binding `phasor_a·phasor_b` is `phasor_a`
  passing through a synapse whose complex weight is `phasor_b` (complex multiply = phase sum). `+ rf_set_complex_weights()`.
  Guarded by `cp_rf_w_re is not None` → inert for RF Tasks 1-3 and the Izhikevich/HH/AdEx models (all unchanged).

## The four gates — all GO (`tests/test_rf_complex_synapse.py`, 4/4 pass)
1. **Bind through one complex synapse:** pre state `a`, synapse weight `b` → post resonates at phase(a)+phase(b)
   (err < 0.05).
2. **Bundle through synapses:** 3 pre phasors via unit synapses → post resonates the phase of their complex sum.
3. **Bind→unbind round-trip through synapses:** bind `(cue,filler)` via a diagonal synapse (weight = filler
   phasor), unbind via a diagonal synapse (weight = conj(cue)) → cleanup recovers the filler (1 of 8).
4. **Load-3 superposition through synapses:** bind 3 `(cue,filler)` pairs, BUNDLE the 3 bound vectors (unit
   synapses), UNBIND each (conj synapses) → recover **all 3 fillers** via cleanup. The superposition + retrieval,
   entirely through complex synapses.

The flagged risk (the recurrent `Σ W z` term destabilizing the phase readout across the resonate window) did NOT
bite — the readout is robust because the zero-crossing occurs within the first cycle, before any magnitude growth
matters (the phase is magnitude-invariant).

## Regression
4/4 RF Tasks-1-3 + complex-synapse tests pass together; the Izhikevich/HH/AdEx branches are byte-unchanged; the
matvec is skipped unless complex weights are installed. No regression.

## Honest scope / what remains
- This realizes bind/unbind/bundle/superposition through complex synapses, validated to load 3. The full
  loads-2/3/5 8×8 task through synapses (a synapse-based analogue of `_rf_on_bridge_probe.py`) would add nothing
  the four gates haven't shown (the ops + the superposition are proven); it's folded into layer (b)'s on-the-real-
  composer validation rather than built standalone.
- **Layer (b) is next:** recode the production composer (`research/runners/core_sim_composition.py` bind/unbind,
  then `brain_conversational_agent.py`) onto RF phasor codes + complex synapses — concept codes become phasor
  vectors (random phases), cleanup becomes phase-cosine similarity (already the FHRR cleanup). **Layer (c):**
  re-validate the full capability matrix (who/what/abstain/negation/clauses/dialogue) at parity — the GATE for
  declaring the opponency cleared on the production path.
- The rate-coded composer stays the production path until layer (c) re-validates at parity; no capability
  regression ships silently.

## Artifacts
- `sim/bridge.py` (complex-synapse matvec + `rf_set_complex_weights`; commits b40aee89 + 60ec7c1a + this)
- `tests/test_rf_complex_synapse.py` (4 gates); design `docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md`
- Backend: CPU (numpy); the RF + complex matvec are backend-agnostic. Frozen bars / no-confab moat untouched.
