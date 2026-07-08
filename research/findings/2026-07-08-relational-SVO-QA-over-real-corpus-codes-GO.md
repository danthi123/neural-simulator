# RELATIONAL/SVO question-answering over the brain's OWN real-corpus codes (GO, 6-seed): the emergent talkable brain STORES + ANSWERS "what did &lt;subject&gt; &lt;verb&gt;? → the object" by binding role-fillers over its DISCOVERED co-occurrence codes, and ABSTAINS on an unstored relation (perfect no-confab moat). A genuinely new knowledge dimension (events/relations) — the SAME codes that do property inheritance/cancellation. NO `sim/` edit.

**Date:** 2026-07-08
**Runners:** `research/runners/_realcorpus_svo_compose_probe.py` (the compose probe) + `_realcorpus_svo_qa_derisk.py` (the QA capability). Self-contained numpy FHRR over the breadth discovery. NO `sim/` edit.
**Verdict:** GO — relational SVO Q&A over real-corpus codes, 6-seed, moat perfect.

## Why this ran (a new knowledge dimension beyond property-inheritance)
The breadth→knowledge arc reasoned about category PROPERTIES (inheritance + cancellation). Real conversation also involves EVENTS/RELATIONS ("the dog chased the cat"). This tests whether the emergent talkable brain's OWN discovered concept codes support relational SVO facts via the project's validated FHRR (Fourier Holographic Reduced Representation) composition — the open question being whether the CORRELATED real-corpus co-occurrence codes compose with acceptable SNR (the compose-perceived arc showed the algebra tolerates code-correlation up to ~0.98).

## Step 1 — the compose PROBE (6-seed GO): the codes bind/unbind cleanly
Each concept's real-corpus code → a unit phasor (fixed random complex projection); roles AGENT/VERB/PATIENT = fixed random phasors; bind a fact = AGENT·z_s + VERB·z_v + PATIENT·z_o; unbind the object = fact·conj(PATIENT) → cleanup.
```
6-seed: obj_acc=1.000, subj_acc=1.000 | permuted(wrong-role unbind)=0.000 | margin stored 0.71 vs unstored 0.01 (~70x) | chance 0.0039
```
The correlated real-corpus codes compose cleanly at D=512 (the role-binding decorrelates cross-terms, √D SNR); unbinding with the WRONG role recovers nothing (the binding is load-bearing).

## Step 2 — the relational Q&A capability (6-seed GO): store + answer-by-cue + moat
A persistent `SVOStore` binds N facts (superposed role-fillers, kept as a list); `answer_patient(subj, verb)` SCANS the stored facts, finds the one whose agent-slot cleans up to &lt;subj&gt; AND verb-slot to &lt;verb&gt; (both above a confidence margin), and reads its patient-slot — NO stored label is read (recovery is purely by unbind+cleanup). An UNSTORED (subj,verb) matches no fact → "I don't know" (gate-first moat).
```
6-seed (K=256, V=256, 12 facts): answer_acc=1.000 | MOAT abstain=1.000 | permuted(wrong-verb cue)=0.000 | chance=0.0039
answer>0.75 all=True | moat>0.9 all=True | beats_permuted all=True  -> GO
```
- **answer_acc 1.0**: every stored fact's object recovered by cue (256× chance).
- **MOAT abstain 1.0**: every unstored relation → abstain (no confabulation) — the moat requires the FULL cue (agent AND verb) to match a SINGLE stored fact.
- **permuted 0.0**: asking with the correct subject but a WRONG verb → no fact matches → abstain — the verb is load-bearing (not just subject-matching).

## Adversarial self-checks (inline)
- The answer is recovered by unbind+cleanup only (the ground-truth (s,v,o) labels are used solely for scoring). 
- The moat is genuine: an unstored (subj,verb) requires BOTH slots to match the SAME fact, which random cues don't → abstain (1.0, all seeds).
- Permuted (correct subj, wrong verb) collapses to 0 → the relation, not just the entity, is what's matched.

## Honest scope
- Rate-level (numpy FHRR) — the project's RFPhasorComposer (spiking resonate-and-fire + complex synapses) is the validated SPIKING realization; feeding these real-corpus grounded codes into it is the step-3 compose-perceived pattern (a reuse-by-import follow-on).
- Distinct-concept facts (subj/verb/obj distinct); scaling to more facts + repeated concepts (a concept as agent in several facts) is a bounded next step (the composer's superposition-capacity scaling).
- The facts are TAUGHT (bound explicitly); the point is that the brain's OWN discovered codes support the relational algebra + moat, not that relations are mined from the corpus.

## What this establishes
The emergent talkable brain's discovered concept codes are rich enough for RELATIONAL knowledge, not just category properties: the SAME codes that inherit/cancel properties also store + answer SVO relational facts ("what did X verb? → the object") with a perfect no-confab moat. One emergent code set, two knowledge dimensions — broadening what the brain can talk about, transformer-free. Follow-on: the spiking realization (RFPhasorComposer + grounded codes); spoken relational answers (the A→W); scaling + repeated-concept facts.

## Files
`research/runners/_realcorpus_svo_compose_probe.py`, `_realcorpus_svo_qa_derisk.py`; per-seed `research/findings/raw/_svo_s*.json` + `_svoqa_s*.json`. Prior: the property-inheritance rungs (breadth→knowledge); the compose-perceived arc (grounded codes into the composer); RFPhasorComposer (the spiking FHRR composer).
