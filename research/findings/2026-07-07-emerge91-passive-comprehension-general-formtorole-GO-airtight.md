# EMERGE-91 — the emergent reservoir comprehension LEARNS PASSIVES: a GENERAL form→role mechanism (canonical + object-relative + passive) in the conversational pipeline, airtight 6-seed GO (position-read necessity + reservoir-silence load-bearing both clean)

**Date:** 2026-07-07
**Runner:** `research/runners/_emerge91_passive_comprehends_composer_answers_derisk.py` (reuse-by-import EMERGE-90's comprehender + the objrel emergent read-out + `RFPhasorComposer`; NO `sim/` edit).
**Verdict:** GO, 6-seed (42/43/44/100/101/102), all controls clean; controller-verified inline (the load-bearing reservoir-silence lesion added + confirmed).
**Builds on:** `2026-07-07-emerge90-objrel-comprehension-in-conversation-CAPABILITY-GO-mechanism-marginal.md` (objrel in the pipeline) + the objrel end-to-end close.

## The result (6-seed)
The brain HEARS a PASSIVE ("the ball was chased by the dog" → agent=dog[by-phrase], patient=ball[surface subject], action=chased), COMPREHENDS it ON SPIKES (the emergent per-role Dale-legal read-out + content-lexeme slotting), STORES the fact, ANSWERS "what did the dog chase?" → "ball", moat intact:
- **passive parse (EXACT-fact) = 1.00, passive recall = 1.00, canonical-not-broken = 1.00 — all 6 seeds.**
- **EMERGENT: pre-learning parse = 0.00 → learned 1.00** all 6 (the delta-rule read-out learns passive; not warm-started). Genuinely spiking, Dale-legal.
- **no-confab moat = 0.00 FA** all 6.
- **NECESSITY (position-read) = 0.00** all 6: a strict surface-order read (slot0→AGENT) misreads the passive (assigns agent=surface-subject=the patient) → the LEARNED reservoir comprehension is necessary. (The CORRECTED necessity control — a POSITION read, not a canonical-LEARNED read which generalizes, per the EMERGE-90 lesson.)
- **LOAD-BEARING (reservoir-silence lesion) = 0.00** all 6: silencing the reservoir feature (zero `final_state`, keep only the +1 bias) collapses passive recall to 0.00 → the reservoir comprehension is genuinely load-bearing for the pipeline turn. (The airtight lesion; the encoder-lesion = 1.00 all 6, a documented-weak EMERGE-78/c2 control because passive's "was"/"by" are OOV — disregarded.)

## The finding: the emergent reservoir comprehension is a GENERAL form→role mechanism
Passive is a genuinely NOVEL slot-role pattern — over the sorted content slots: canonical `[AGENT,PRED,THEME]`, object-relative `[THEME,AGENT,PRED]`, **passive `[THEME,PRED,AGENT]`** (slot2=AGENT never appears at slot2 in canonical or objrel). The read-out NEEDED passive training (the canonical+objrel-only read-out gets passive 0.00) and LEARNED it from experience (train set grows to include passives; `DP._train_dopamine` unchanged; the function words "was"/"by" excluded by the same content-lexeme filter as "that" — no new hand rule; the roles come entirely from the emergent read-out). ⇒ the fronto-striatal reservoir + a learned Dale-legal spiking read-out + content-lexeme slotting comprehends CANONICAL + OBJECT-RELATIVE + PASSIVE — three distinct form→role mappings — all emergent, spiking, moat, in the fully-spiking conversational pipeline on one co-resident brain. This is the anti-whack-a-mole comprehension goal generalizing (the reservoir retires the hand form→role labeler across construction families, not one branch at a time).

## Honest scope
Validated at 3 construction families (canonical/objrel/passive), 6-seed, held-out content. The load-bearing controls are the position-read necessity + the reservoir-silence lesion (both clean); the closed-class encoder-lesion is a documented-weak control (doesn't collapse for OOV-function-word constructions) and is not relied upon. NEXT: more constructions (datives, etc.) → the reservoir comprehender as the DEFAULT parse path in the flagship console (retire the hand `label_sentence_ext`) — now well-motivated (3 families handled). NO `sim/` edit.

## Files
`research/runners/_emerge91_passive_comprehends_composer_answers_derisk.py`; `research/findings/raw/_emerge91_passive_seed{42,43,44,100,101,102}.json` (6-seed, both lesions reported).
