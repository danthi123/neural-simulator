---
type: plan
status: live
date: 2026-06-05
---

# FHRR-on-bridge layer (b) — recode the production composer onto RF phasor + complex synapses — design — 2026-06-05

> **For Claude:** layer (b) of the owner-greenlit full FHRR-on-bridge feature
> (`docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md`), after layer (a) GO
> (`research/findings/2026-06-05-fhrr-layer-a-complex-synapse-GO.md`). De-risk-driven, TDD, reuse-by-import the RF +
> complex-synapse substrate already on the bridge (NO further `sim/` edits expected for layer b). Frozen bars /
> no-confab moat never weakened; the rate-coded composer stays production until layer (c) re-validates at parity.

**Goal:** a composer that performs the full conversational capability (store / who-what Q&A / abstention / negation /
clauses / dialogue) using FHRR phasor codes + the bridge's RF complex-synapse bind/unbind — so the opponency is gone
(the phasor algebra has no common mode) on the real conversational path.

## Approach — a PARALLEL RF phasor composer (safe; no in-place rewrite until parity)
Build `research/runners/rf_phasor_composer.py` (`RFPhasorComposer`) exposing the SAME API the
`BrainConversationalAgent` uses (`store`, who/what queries, `unbind`, abstention), but FHRR-phasor-based on the RF
bridge. `core_sim_composition.CoreSimComposer` stays the production path; layer (c) validates the RF composer against
the identical capability tests, then the agent switches. A regression at layer (c) is a reportable finding (the
measured cost), not hidden.

### Representation map (rate-coded ±1 Hadamard → FHRR phasor)
| current (CoreSimComposer) | RF phasor composer |
|---|---|
| `self.concepts[w]` = denoise64 real code (D-dim) | phasor code: phases `θ_w ∈ [0,1)^D` (deterministic per seed) |
| `self.roles[r]` = ±1 code | role phasor: phases `φ_r ∈ [0,1)^D` |
| bind = ±1 Hadamard via `hadamard_spiking` (rate) | bind = `phasor_cue ⊙ phasor_filler` via DIAGONAL complex synapses on the RF bridge (layer-a Gate 1/3) |
| superposition + `onoff(bon−boff)` opponency (the WALL) | bundle = sum of bound phasors via UNIT complex synapses (layer-a Gate 2/4) — NO opponency exists |
| unbind = rate Hadamard with role | unbind = `phasor_bound ⊙ conj(phasor_cue)` via conj diagonal synapses (layer-a Gate 3) |
| cleanup = numpy argmax / NEF spiking | cleanup = phase-cosine similarity `mean(cos(2π(rec−code)))` argmax |
| abstention (no-confab moat) = None when no agent matches | SAME: abstain when max phase-similarity < threshold (set between groundable-min and ungroundable-max, layer-a/de-risk showed a clean gap) |
| negation/yes-no = bound polarity tag | SAME: a bound AFFIRM/NEGATE phasor tag |
| clauses = recursive bound vector as filler | SAME: a clause's bound phasor is a filler phasor |

**Why phasor codes are random (not the denoise64 codes):** FHRR robustness is the unit-magnitude phase code (SNR ≈
2N/M, a dimension dial). The denoise64 codes are the rate substrate's; the RF composer uses fresh per-seed phasor
codes. (A later refinement could derive phasor phases from the substrate's own activity — out of layer-b scope.)

## De-risk-driven TDD sequence (each gate GO before the next)
- **b.1 minimal RF composer (the de-risk GATE for layer b):** store 2-3 SVO facts; `who <action> <patient>?` →
  agent; `what did <agent> <action>?` → patient; an absent-cue query → ABSTAIN (the no-confab moat). All through the
  RF complex-synapse bind/bundle/unbind on the bridge. GATE: correct retrieval + correct abstention, multi-seed.
- **b.2 negation / yes-no:** a bound polarity tag; `is <fact>?` → YES/NO; negated facts abstain on positive query.
- **b.3 one-attribute + clauses:** attribute role-tag (the documented 1-attribute RESOLVES boundary carries over);
  a clause as a filler (recursive).
- **b.4 dialogue (`elaborate`):** reuse the dlPFC content-selection over the agent's RF-stored facts.

Each gate mirrors an existing `tests/test_core_sim_composition.py` / `test_brain_conversational_agent.py` test so
parity is measured against the current bars.

## Layer (c) — re-validate at parity, then switch
Run the RF composer against the FULL capability matrix at the production scale (the same multi-seed bars the
rate-coded composer meets). GO at parity → switch `BrainConversationalAgent` to the RF composer; the opponency is
cleared on the production path + the F=3 two-attribute resonator becomes available (the ±1 scheme's documented
2-attribute K=5 boundary may lift — a bonus to test). NEGATIVE/partial → report the measured cost; keep the
rate-coded composer; the RF composer stays an opt-in validated alternative.

## Scope / risk
- Performance: per-op resonate windows (period+8 steps) are slower than the rate composer's read window; the RF
  composer is correctness-first. A later optimization pass (shorter period, batched ops, GPU complex matvec) follows
  parity. The de-risk used period=1000; b.1 can use a shorter period if it holds parity.
- Sparse complex weights: layer-a used dense N×N for the de-risk; the RF composer at production D needs a sparse
  complex matrix (`cupyx.scipy.sparse` / `scipy.sparse` two-real-part form) — a contained addition if dense is too
  heavy.
- NO further `sim/` edits expected (the RF + complex-synapse substrate is in). If one proves necessary, flag it.
