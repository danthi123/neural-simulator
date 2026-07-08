# SPOKEN relational SVO Q&A (GO): the talkable brain ANSWERS "what does the fish eat? → mouse" by RECOVERING the object (FHRR unbind over its own real-corpus codes) and SPEAKING it ON SPIKES, with a genuinely relational no-confab moat ("what does the cat eat?" → "I don't know" — cat is a stored object, not a subject). NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_spoken_svo_qa_derisk.py` (reuse-by-import: `SVOStore` [the relational Q&A GO] + the breadth concept-pool A→W `ConceptFrameSpeaker`; numpy FHRR reasoner + numpy A→W in one process). Requires `SIM_BACKEND=numpy`. NO `sim/` edit.
**Verdict:** GO — the relational answer is recovered from the brain's own codes and spoken on spikes, moat intact.

## Why this ran
The relational SVO Q&A (CYCLE 988, 6-seed GO) answered over the brain's own codes but returned a concept index (rate-level). This ties it to SPEECH: store speakable facts (animal-eat-animal), ask a relational question, recover the object by FHRR unbind, and SPEAK it ON SPIKES via the A→W — making the relational dimension part of the actual spoken conversation.

## The result — seed 42 (K=256)
```
STORED facts: 'the fish eats mouse', 'the bird eats dog', 'the frog eats cat'
ask 'what does the fish eat?' -> "mouse"        -> SPOKE OBJECT ON SPIKES [correct]
ask 'what does the bird eat?' -> "dog"          -> SPOKE OBJECT ON SPIKES [correct]
ask 'what does the cat eat?'  -> "I don't know"  -> [MOAT: cat is a stored OBJECT, not a subject]
ask 'what does the zzzqqx eat?' -> "I don't know" -> [MOAT: unknown word]
VERDICT: GO (2/2 correct + moat)
```
The brain recovers the object by FHRR unbind over its OWN real-corpus codes and SPEAKS it on spikes (the A→W). The moat is genuinely RELATIONAL: "what does the cat eat?" abstains because no stored fact has `cat` as the agent (cat appears only as an object) — not merely an unknown-word check.

## 6-seed (K=256, verb "eat")
**All 6 seeds 2/2 correct + moat 1.0** — every stored subject's object is recovered (FHRR unbind) and spoken correctly on spikes; every unstored/unknown relation abstains.
```
seed 42/43/44/100/101/102: 2/2 correct | moat True   -> GO all 6
```
**A→W spell fix (systematic-debugging).** A first pass had per-seed spell misses (e.g. seed 44 "bear eats fish" spoke "sleep") because the A→W speaker bridge was built with the RUN's seed, not the checkpoint's. `save_checkpoint` does not persist firing thresholds (the CLAUDE.md gotcha), so a mismatched build seed perturbs the decode. Building the speaker at the checkpoint's seed (42), decoupled from the reasoner's seed, restored 2/2 on every seed — a general lesson for any A→W speaker loaded from a checkpoint (the same fix applies to the spoken-cancellation runners).

## Honest scope
- numpy FHRR reasoner + numpy A→W co-execute in one process (the same one-backend pattern as the fully-spiking cancellation conversation). The FHRR is rate-level; the spiking realization is the RFPhasorComposer (follow-on).
- Facts are animal-VERB-animal over spellable animals present in the discovered vocab (so the object is speakable); "fish eats mouse" etc. are mechanism-demo pairings (the relational algebra + moat is what's validated, not the facts' plausibility).
- The question is a fixed "what does &lt;subj&gt; &lt;verb&gt;?" template; the neural interrogative parse is a separate validated piece (a follow-on to wire in).

## What this establishes
The emergent talkable brain now SPEAKS relational answers: it stores SVO facts over its own discovered codes, recovers the object by role-unbinding, speaks it on spikes, and abstains on an unstored relation (a genuinely relational moat). Combined with the property (inherit/cancel) speech, the brain converses across TWO knowledge dimensions — category properties AND relations — spoken on spikes, transformer-free, moat intact. Follow-on: an interactive console routing property-vs-relational questions; the spiking FHRR realization; scaling + repeated-concept facts.

## Files
`research/runners/_realcorpus_spoken_svo_qa_derisk.py`; per-seed `research/findings/raw/_spk_svoqa_s*.log`. Prior: the relational Q&A `2026-07-08-relational-SVO-QA-over-real-corpus-codes-GO.md`; the frame speech `2026-07-08-full-frame-fluent-speech-on-spikes-GO.md`.
