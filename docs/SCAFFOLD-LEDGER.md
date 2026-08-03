# Scaffold Ledger

Last rewritten: 2026-08-03.

This ledger names temporary shortcuts that are useful for research but are not
the final brain. A scaffold is allowed only when it helps us build or measure the
real mechanism and when its replacement path is visible.

## Rule

Every scaffold needs:

- a plain name;
- what it currently helps with;
- why it is not final;
- the biological or brain-native replacement;
- a trigger for removing or demoting it.

Evaluation tools are allowed to inspect the brain from the outside. The problem
is when an evaluation tool or host helper becomes the thing doing cognition.

## Current Scaffolds

| Scaffold | What it helps with | Why it is not final | Replacement path | Burn-down trigger |
|---|---|---|---|---|
| Conventional language-model training | Gives language experiments enough sequence structure to test larger circuits. | It learns text continuation outside grounded lived interaction. | Grounded speech-as-action learned from perception, action, affect, memory, and contingent social feedback. | A brain-native language path can choose and render simple utterances from internal state in the live loop. |
| Host-side query parsing and discourse planning | Makes current conversation demos usable and testable. | The host decides much of what the brain should interpret or say. | Neural comprehension, preverbal message formation, and speech planning on the shared substrate. | Removing the parser/planner does not collapse basic grounded dialogue in the live loop. |
| Exact source metadata floor | Prevents current known-fact recalls from confidently asserting answers that disagree with the exact stored source fact. | It reads a Python fact record, which is database-like source access. | Independent neural/source-memory agreement, then learned source-monitoring tied to self-schema confidence. | The independent source-memory path and learned monitor cover the same failure battery with source metadata disabled or permuted. |
| Independent RF source-memory echo | Gives the honesty hook a separate RF/FHRR memory trace to check whether a candidate answer fits the cue. | The echo is deliberately written at store time with engineered codes; it is not yet learned developmental source monitoring. | Plastic source tags and a learned ACC/aPFC/self-schema monitor that treats source disagreement as uncertainty. | Six-seed familiar-wrong/source-mismatch tests pass when the echo is learned or self-organized rather than written as a parallel engineered trace. |
| Hand-designed concept and role codes | Makes binding, recall, and early language experiments tractable. | The code geometry is assigned by the host instead of learned from perception/action/affect. | Self-organized grounded assemblies with overlapping sensory, motor, affective, and word representations. | New concepts can be acquired from interaction and used by memory/language without manually assigned codes. |
| Fixed grammar frames and render templates | Allows simple SVO-style speech and question answering. | Word order and phrasing are supplied by host structure rather than emerging from a language circuit. | Neural message-to-word-order production and later articulation-like motor output. | Simple speech remains coherent after templates are removed or used only as evaluation labels. |
| Host-computed novelty, appraisal, confidence, or learning-progress scalars | Speeds de-risking of curiosity, affect, and honesty circuits. | The host computes the psychological signal instead of the brain. | Spiking familiarity, prediction-error, appraisal, and metacognitive circuits feeding neuromodulators and self-schema. | Lesionable brain signals replace the host scalar and preserve behavior across seeds. |
| Hand-set pathway weights or fixed region wiring | Lets us test whether a circuit shape is worth pursuing. | Development should tune and wire useful pathways through local learning and growth. | Local plasticity, developmental wiring rules, homeostasis, replay, and growth. | The same behavior survives with learned/self-organized weights and appropriate collapse controls. |
| AI teacher/caregiver | Provides early social interaction, correction, and curriculum. | The teacher is an external scaffold and may smuggle intelligence into training. | Gradual handoff to human interaction and simpler world feedback, with the brain's own learning doing the work. | The brain continues learning useful language/behavior from natural interaction after teacher support is reduced. |
| Measurement readouts and artifact scripts | Make science auditable. | They are outside observers and should not drive behavior unless explicitly modeled as sensory feedback. | Keep them as instruments only; move behavioral decisions into neural circuits. | Reports and gates can observe the brain, but production behavior no longer depends on the measurement script. |

## Workflow

When adding or promoting a mechanism:

1. State its role in the whole brain.
2. Name any scaffold it introduces or relies on.
3. Add or update the ledger entry.
4. Define the replacement and burn-down trigger.
5. Run tests that would fail if the scaffold silently became the capability.

This file is intentionally compact. The detailed evidence remains in
`research/findings/`.
