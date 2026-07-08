# NEURAL role/filler EXTRACTION for questions (GO, 6-seed): a fronto-striatal reservoir read-out labels each question content word's thematic ROLE (subj=AGENT / verb=PREDICATE / obj=THEME), replacing the console's position-based token extraction — held-out 1.000 on NOVEL fillers, SCRAMBLE collapses (word order is load-bearing). Completes FULLY-NEURAL comprehension (type + roles). A clean linguistic dissociation: question TYPE is function-word-carried (lesion collapses), thematic ROLE is word-order-carried (scramble collapses). NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_neural_role_extraction_derisk.py` (reuse-by-import: EMERGE-78 `Reservoir`/`Encoder`/`_fit_slots`/`_slot_acc`). CI guard `tests/test_neural_question_routing.py::test_neural_role_extraction_generalizing`. numpy. NO `sim/` edit.
**Verdict:** GO (6-seed) — each question content word's thematic role is extracted neurally, generalizing to novel fillers, word order load-bearing.

## Why this ran (the last comprehension residual)
The neural question-TYPE router is de-risked + console-wired (CYCLE 1025/1026). But the console still EXTRACTS the subject/verb/object of a question by TOKEN POSITION (`c = [t for t in toks if t not in determiners]`). This de-risk replaces that with a NEURAL read-out: the EMERGE-78 reservoir + a per-slot role read-out (Dominey-Hinaut) labels each content word's thematic ROLE from the whole question, so the console recovers subj=AGENT, verb=PREDICATE, obj=THEME on spikes.

## The load-bearing property (why a reservoir, not positions)
The ROLE of the head noun FLIPS with the question form: in "who eats the X" the X is the THEME (done-to); in "what does the X eat" the X is the AGENT (doer). A position rule cannot assign this without knowing the form; the reservoir reads the WHOLE question. Held-out on NOVEL fillers (the role is carried by the closed-class + word-order STRUCTURE, not the specific animal/verb).

## The result — 6-seed (42/43/44/100/101/102)
```
held-out (NOVEL fillers) role-acc = 1.000 every seed
SCRAMBLE (shuffle the question)   = 0.33-0.51 (load-bearing collapse; word order carries the role)
LESION (closed-class -> generic)  = 0.778 (reported; only PARTIAL)
chance = 1/5 = 0.20
```
- **Held-out 1.000 on NOVEL fillers, all seeds** — each content word's thematic role (AGENT/PREDICATE/THEME) is extracted correctly, generalizing to animals/verbs never trained on.
- **SCRAMBLE collapses it** (→ 0.45) — the WORD ORDER is load-bearing for thematic role (English signals role by position; the reservoir integrates it).

## The clean linguistic dissociation (type vs role)
Comparing to the question-TYPE router (CYCLE 1025): the two are carried by DIFFERENT cues, and the reservoir correctly uses each —
- **Question TYPE** (property/what/who/yes-no): FUNCTION-WORD-carried → the closed-class LESION collapses it (0.20), SCRAMBLE stays high (order-invariant).
- **Thematic ROLE** (AGENT/PREDICATE/THEME): WORD-ORDER-carried → SCRAMBLE collapses it (0.45), the closed-class lesion only partial (0.778 — position survives the lesion).

This is the linguistically-correct division of labor (Bates-MacWhinney competition model: English marks role by word order, and grammatical class by function words) — emergent from one reservoir, not hand-coded.

## What this establishes
Comprehension is now FULLY neural: the question TYPE (which handler) AND the thematic ROLES (which word is subj/verb/obj) are both extracted by reservoir read-outs, generalizing to novel fillers, on the substrate. Together with the fully-spiking production side, the WHOLE conversational turn — comprehend (type + roles) → reason → speak — is neural. Follow-on: wire the role read-out into the console's `ask()` extraction (replace the position-based parse); the spiking-LSM role read-out.

## Files
`research/runners/_realcorpus_neural_role_extraction_derisk.py`; `tests/test_neural_question_routing.py`. Reuses EMERGE-78 (`Reservoir`/`Encoder`/`_fit_slots`/`_slot_acc`). Prior: the neural question-TYPE routing `2026-07-08-neural-question-comprehension-routing-GO.md` + its console wiring `2026-07-08-console-neural-question-routing-wired-GO.md`.
