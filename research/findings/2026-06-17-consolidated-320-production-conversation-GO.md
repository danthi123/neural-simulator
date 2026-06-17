# Consolidation GO — the production conversational agent talks with the codes it LEARNED FROM CONVERSATION (320-concept stream cortex), end-to-end on one agent

**Date:** 2026-06-17
**Status:** **GO.** 3/3 seeds on the host-readout stream codes + **3/3 on the fully-brain-based neural-readout
codes** (seeds 42/43/44; the third-seed hardening completed 2026-06-17 once the neural codes finished streaming).
The loop closes: learn word meanings from a conversation stream → converse using them through the production
agent. No new mechanism — this is the assembly of validated pieces the CYCLE-119 plan named as the reuse-heavy
runner-up frontier.

## What was separate, and is joined here

Two halves of the conversational system had never been run together:

1. **The 320-concept cortex** (`_phaseB_onbridge_stream_conversation_derisk.py`) learns each word's meaning from
   a conversation stream by population-Hebbian co-occurrence, reads it out with two real cortical gain-control
   operations (per-hub spike-frequency adaptation + per-concept feedforward inhibition, the `--readout-norm
   neural` path), and stores a 320×300 concept-code matrix per seed. It had been validated only through a numpy
   HRR who/what + abstention pipeline — **not** through the production agent.
2. **The production conversational agent** (`brain_conversational_agent.py` → `rf_phasor_composer.py`) parses
   sentences by word-position × voice, binds role-filler facts on resonate-and-fire phasor neurons, answers
   who/what, ABSTAINS when no fact matches (the no-confab moat), confirms yes/no over a bound polarity tag, and
   — with `enable_neural_render` — produces a described sentence's word ORDER from a spiking competitive-queuing
   serial-order generator. It had always run on codes the **composer self-generates**, never on the codes the
   cortex learned from conversation.

`research/runners/consolidated_320_conversation_demo.py` joins them: it feeds the 320 stream-learned cortex
codes into the production agent as its concept vocabulary (via the fixed complex grounding projection
`angle(M @ code)/2π`, the same map the step-3 perception arc used —
`_step3_grounded_codes_production_composer_derisk.py`), then drives the agent through a multi-turn conversation.

## Result

```
[neural read-out, fully brain-based]  seed 42: recall 1.00 | abstain 1.00 (0 false-accepts) | yes/no ok
                                                | describe 'dog eat apple' | elaborate 'apple'  ==> GO
[host read-out, multi-seed]           seed 42/43/44: recall 1.00 | abstain 1.00 (0 false-accepts) | yes/no ok
                                                | describe 'dog eat apple' | elaborate 'apple'  ==> GO 3/3
```

The agent hears eight natural child-corpus statements (`dog eat apple`, `cat play ball`, `girl run park`, …,
plus one explicitly NEGATED `fish eat cake`), then:

- **Recall** (who AND what): every stored fact's patient and agent recovered — 1.00, 16/16 queries per seed.
- **No-confab moat**: every unstored (agent, action) and (action, patient) cue ABSTAINS (returns None) — 0
  false-accepts at every seed. The moat is the production RELATIONAL host check, which abstains on whether the
  fact was stored, not on code geometry — so it is structurally safe regardless of how correlated the codes are.
- **Yes/no** over the bound polarity tag: affirmed fact → "yes", negated fact → "no", unstored → "unknown"
  (never affirms an unstored triple).
- **Describe** (generation): `describe("dog")` → `"dog eat apple"`, word ORDER produced by the spiking
  competitive-queuing serial-order generator; `describe("frog")` (no stored fact) → None (no confabulation).
- **Elaborate** (dialogue planning): `elaborate("dog")` → `"apple"`, an on-topic associate chosen by the dlPFC
  spiking content-selection Control over the agent's own association graph.

## The interesting science (and an honest scope note)

The cortex codes are SEMANTICALLY STRUCTURED — they carry category similarity (mean grounded phase-cosine
+0.10–0.13, max +0.42–0.56 among the demo words), which is exactly what lets the cortex generalize. The
production binder prefers decorrelated codes, but the role-binding decorrelates the cross-terms (tolerant to
code-similarity up to ~0.98, `_step3_correlated_percept_boundary.py`), so recall stayed perfect and
within-category-error was 0 at every seed.

**Honest scope.** With 8 clean facts there were no recall errors, so the predicted "within-category
generalization signature" (a recall error, if any, lands on a same-category neighbor: dog→cat) is the metric the
runner is set up to measure but did NOT need to fire here — clean recall is the GO. Eliciting that signature
(many same-category agents, denser KB) is a bounded follow-on. The moat is the production relational host check;
the LEARNED familiarity gate is validated alongside it at V=320 separately (`familiarity_gate_v320_validation.py`)
and is not weakened here. This is a CONSOLIDATION (assembly of already-validated pieces into one production
agent), not a new capability.

## Why it matters

It is the end-to-end closure of the conversational loop on ONE agent: the same system that learned what 320
words mean from hearing a conversation now uses those learned meanings to parse, store, recall, abstain, negate,
generate (neural word order), and plan what to say next — with the no-confabulation guarantee intact. In the
CYCLE-119 fork this is the conversational PRODUCT in the most-likely (dendritic-NEGATIVE) branch, and a clean
deliverable in every branch.

## Reproduce

```bash
# fully-brain-based neural read-out codes (seed 42):
SIM_BACKEND=numpy python -m research.runners.consolidated_320_conversation_demo --seeds 42 --readout neural
# multi-seed host read-out codes:
SIM_BACKEND=numpy python -m research.runners.consolidated_320_conversation_demo --seeds 42 43 44 --readout host
# regression guard (skips if the stream-code cache is absent):
SIM_BACKEND=numpy python -m pytest tests/test_consolidated_320_conversation.py -q
```

No `sim/` edit. Reuse-by-import: `BrainConversationalAgent`, `RFPhasorComposer`, `TAXONOMY_40x8`, the step-3
grounding map.
