# gap#5 one-brain MERGE — the PRODUCTION conversational agent (OneBrainComposer) survives a WAKE→SLEEP→WAKE phase-switch cycle, 6-SEED GO: store + recall + the no-confab MOAT are identical before/after the AdEx sleep phase — the merge is validated on the REAL agent (2026-07-25)

## Headline
The one-brain replay merge is now validated on the **actual production conversational agent**. The `OneBrainComposer`
bridge is Izhikevich/dt1.0 (the RF composer runs as masked complex-synapse ops on a slice, NOT a global RF neuron model),
so the validated wake/sleep phase-switch (Izh/dt1.0 → AdEx/dt0.1 → Izh/dt1.0) applies directly. Running the switch cycle
around a live composer: its **store, recall, AND no-confab moat are IDENTICAL before and after the AdEx sleep phase**,
6-seed GO. The conversational memory (RF complex synapses `cp_rf_w_re/im` + the Izhikevich parser weights `cp_connections`)
survives the neuron-model switch untouched.

## Result — 6-SEED GO (seeds 42/43/44/100/101/102)
Store 3 SVO facts (dog→go→north, cat→come→east, bird→look→south); query them (WAKE 1); run the phase-switch sleep cycle
(→ AdEx/dt0.1, frozen STDP, a 60-step SWR/sleep window, → Izhikevich/dt1.0); re-query (WAKE 2):
- **recall preserved 6/6:** pre-sleep `['north', 'east', 'south']` == post-sleep `['north', 'east', 'south']` every seed.
- **no-confab moat intact 6/6:** two never-stored cues (apple→swim, river→stop) abstain (`None`) both before AND after
  the sleep cycle.
- ⇒ **VERDICT: GO** — the production agent's full conversational capability survives the wake/sleep phase-switch.

## Why it works (and why the memory is untouched)
The phase-switch (`switch_to_adex_sleep` / `switch_to_izhikevich_wake`) swaps only the neuron-model state (v / adex_w / u
/ cfg model+dt + cached synaptic decays); it never touches the synaptic weight arrays. The composer's memory lives in
those weight arrays:
- the RF fact-store + bind/bundle in the **complex synapses** `cp_rf_w_re/im` (separate arrays, model-agnostic);
- the parser's comprehension in `cp_connections` (preserved byte-identical, as the round-trip finding showed).
During sleep, the composer is quiescent (no ops issued), the RF neurons are AdEx-typed but idle, and the reset of v/u is
harmless (the RF composer is stateless-per-op, re-kicked each op — its memory is in the synapses, not v/u). On wake, the
Izhikevich dynamics + RF ops resume against the preserved weights.

## What this closes
This is the final MECHANISM piece of the one-brain replay merge, now proven on the REAL production agent (not a stand-in):
nav/conversational Izhikevich brain + an AdEx CA3-replay slice can co-reside on ONE bridge; a SWR/sleep phase switches to
AdEx/dt0.1+frozen-STDP to run the CA3 traveling replay (validated 6-seed, `1bdcc5a4`), and the production conversational
memory + moat survive the cycle (this finding). The merge sidesteps the per-region-neuron-model + dt-stiffness walls via
the temporal separation of replay and conversation — exactly as in the brain.

## HONEST SCOPE / remaining
- This validates the composer's SURVIVAL across the sleep cycle. Wiring the actual CA3 replay slice co-resident ON the
  composer's bridge (so the SWR window runs a real replay, not just an idle AdEx window) is the last assembly step — the
  co-resident round-trip (`1bdcc5a4`) already proved a conversational slice + replay slice work together, and the composer
  survives the switch (this finding); combining them is plumbing, not new de-risk.
- A `sim/` first-class `switch_neuron_model_phase()` + `reset_transient_state()` capability (additive, default-off,
  CI-guarded) would replace the runner-side switch replication — a clean follow-on.
- The neural imaginative-replay READER (the Bayesian decode is a measurement instrument) is the other remaining gap#5
  closure item, naturally built alongside replay-driven consolidation.
- Runner-import fix (shipped here): `_gap5_wake_sleep_phase_switch.py` + `_gap5_wake_sleep_roundtrip.py` had unguarded
  `__main__` blocks (importing them ran their full 6-seed suites); both are now `if __name__ == "__main__"`-guarded so the
  switch helpers are cleanly importable.

## Provenance
`research/runners/_gap5_onebrain_sleepcycle_merge.py` (log `onebrain_sleepcycle{,6}.log`). Reuses the phase-switch helpers
(`_gap5_wake_sleep_phase_switch`, `_gap5_wake_sleep_roundtrip`) + the production `OneBrainComposer`. NO `sim/` edit. GPU
(needs the concept-code cache + CuPy). Builds on the full gap#5 replay/learned-band/merge arc (`d6e140bf`, `a051d84d`,
`6ed6f0a2`, `50255443`, `1bdcc5a4`).
