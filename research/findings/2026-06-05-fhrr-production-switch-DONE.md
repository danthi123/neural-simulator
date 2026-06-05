# FHRR-on-bridge — PRODUCTION SWITCH DONE: the conversational agent runs opponency-free — 2026-06-05

**The `BrainConversationalAgent` now defaults to the FHRR-on-bridge `RFPhasorComposer`.** The opponency rate-coded
SNR wall — the blocker that started this arc — is cleared on the conversational production path. Owner-signed-off
production switch (the final step of the owner-chosen "full 320 production escape"). Validated at parity; the
rate-coded composer remains an explicit opt-in.

## The switch
- `research/runners/brain_conversational_agent.py`: `__init__` gains `composer_kind` (default **'rf'**). When no
  explicit `composer` is passed, the agent builds an `RFPhasorComposer(seed, D=128, vocab=..., period=200)` instead
  of the rate-coded `CoreSimComposer`. `composer_kind='rate'` keeps the legacy composer as an opt-in (needs the
  denoise64 cache). An explicit `composer=` instance still overrides.
- `research/runners/rf_phasor_composer.py`: added "come" to `DEFAULT_VOCAB` (the agent's test set uses it) + a
  duck-typed `_is_clause()` so the composer recognizes BOTH its own `Clause` AND `core_sim_composition.Clause` (the
  agent passes the latter; they are distinct namedtuple classes, so `isinstance` missed across them).

## Re-validation at parity (GPU)
- **The agent's full EXISTING suite passes with the RF default** (`tests/test_brain_conversational_agent.py`, 7
  tests, unchanged): comprehend→store→who/what Q&A + abstention, voice-invariant comprehension, negation/yes-no,
  embedded clauses, dialogue planning (`elaborate`), generation (`describe`), and the dlPFC cache-invalidation
  guard. The agent's tests were written for the rate composer; they pass verbatim on RF → behavioral parity.
- The RF composer suite (`tests/test_rf_phasor_composer.py`, 22 tests) still green. Combined re-validation **29
  passed** (218 s GPU). No test was weakened; the no-confab moat holds throughout.

## What the opponency escape delivered (the arc, end to end)
- Opponency (`onoff(bon−boff)`) confirmed a FUNDAMENTAL rate-coded SNR wall — 3 independent spiking mechanisms
  NEGATIVE (`2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED.md`); biology removes the common mode in the
  analog stage before spiking (Kandel Ch 22).
- Pivot to FHRR phasors (no common mode, no small signed difference). RF-on-bridge de-risk GO; layer (a)
  complex-synapse bind GO; layer (b) the RF phasor composer with the full capability matrix multi-seed; layer (c)
  320-correctness-validated + optimized (sparse complex-matvec + dedicated `rf_resonate_steps` loop).
- The conversational composer is now FULLY on the FHRR-on-bridge substrate (resonate-and-fire phasor neurons +
  complex synapses) — the opponency does not exist in the phasor algebra. **Bonus available:** the F=3
  two-attribute resonator (which the ±1 scheme provably can't do) can now lift the documented 2-attribute K=5
  boundary — a follow-on.

## Honest residuals / follow-ons
- The 320-concept retrieval pipeline (`compose_flatdist320` etc.) is a SEPARATE system; this switch is scoped to
  `BrainConversationalAgent`. RF is 320-correctness-validated (8/8/8) so a future larger-vocab agent is supported.
- Performance: the RF composer is correctness-first; per-op latency is still seconds-scale at 320 (D=512). Remaining
  incremental optimization (batched per-fact ops, fused RF kernels, shorter period — the sweep showed period=80
  holds) for response latency. Not blocking.
- GPU: the agent path (parser + dlPFC) is GPU-validated (numpy-backend issues in those components, not the RF ops).

## Protected `sim/` edits this arc (all additive/guarded/flagged, zero regression)
`NeuronModel.RESONATE_AND_FIRE` + the RF step branch / `_rf_advance_one` / `rf_resonate_steps` + `rf_kick` /
`rf_read_phases` / `rf_set_complex_weights` (sparse). Izhikevich/HH/AdEx byte-unchanged; determinism + the existing
agent clean. Frozen bars / no-confab moat never weakened.
