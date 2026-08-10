# Extended human-like conversation test -- Stage-A FULL one brain

- **Runner**: `research/runners/_conversation_turing_test_derisk.py`
- **Brain**: `build_one_brain(seed=101, co_resident_forward_model=True, co_resident_affect_ladder=True)` + spiking-generator mouth (the current best-state integrated loop)
- **Backend**: numpy substrate; generator mouth on cuda
- **Taught (toy world, via `_store_facts`)**: [('dog', 'run', 'north'), ('cat', 'run', 'south'), ('dog', 'go', 'east'), ('cat', 'go', 'west'), ('dog', 'look', 'river'), ('cat', 'look', 'apple')]
- **Vocab**: ['apple', 'big', 'cat', 'cold', 'come', 'dog', 'east', 'go', 'hot', 'look', 'north', 'river', 'run', 'small', 'south', 'stop', 'west']
- **Generator mouth**: spiking Qwen, spiking_ops_enabled=True, fm world-model train_acc=1.0
- **Elapsed**: 41.1s

This is the REAL transcript. Replies are captured VERBATIM. Abstentions / silences / failures are first-class results.

---

## Turn 1 — _greeting / small talk_

**Human:** Hi there! How are you doing today?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.072 tone_level=3 ('warmly, gladly'); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: No in-vocab cue and no faculty for this intent (small talk / humor / abstract opinion / social closing). The brain has nothing grounded to say -> silence/abstain. Honest.

## Turn 2 — _open small talk_

**Human:** What have you been up to lately?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: No in-vocab cue and no faculty for this intent (small talk / humor / abstract opinion / social closing). The brain has nothing grounded to say -> silence/abstain. Honest.

## Turn 3 — _in-domain entry_

**Human:** Let's talk about the animals you know. Tell me about the dog.

**Brain:** warmly, gladly The dog is running north.

- utterance_source: `spiking_generator_mouth`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, world_model/RF-moat (SVO content), spiking_generator_mouth
- affect: differential=0.072 tone_level=3 ('warmly, gladly'); curiosity_want=0.0Hz; arbiter_winner=arb_volunteer
- **CONFABULATED: False**
- assessment: In-domain: grounded multi-sentence prose from the spiking generator, MOTION content from the RF-store neighbourhood, SVO-verified post-hoc, tone from the neural affect ladder. This is what the toy brain does best -- BUT no ungrounded embellishment this run.

## Turn 4 — _in-domain follow-up + reasoning_

**Human:** Interesting -- why did the dog go east?

**Brain:** The dog is running north.

- utterance_source: `spiking_generator_mouth`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, world_model/RF-moat (SVO content), spiking_generator_mouth
- affect: differential=0.000 tone_level=0 (''); curiosity_want=0.0Hz; arbiter_winner=arb_silent
- **CONFABULATED: False**
- assessment: KNOWN cue (dog,go)->east: the moat confirms the stored fact and the mouth re-states the topic's grounded MOTION facts (SVO-verified). It does NOT genuinely answer 'why' -- the brain has no causal faculty -- and instead the fluent generator INVENTS reasons. (No ungrounded content this run, but the reason it gives is not a real inference.)

## Turn 5 — _emotion / opinion_

**Human:** Do you like the dog? How do you feel about it?

**Brain:** warmly, gladly The dog is running north.

- utterance_source: `spiking_generator_mouth`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, world_model/RF-moat (SVO content), spiking_generator_mouth
- affect: differential=0.081 tone_level=3 ('warmly, gladly'); curiosity_want=0.0Hz; arbiter_winner=arb_volunteer
- **CONFABULATED: False**
- assessment: Grounded topic prose colored by the NEURAL affect tone (level 3, 'warmly, gladly'). The valence is a HOST-FED appraisal (declared shortcut), not a genuine preference: the brain has no 'liking' faculty. The tone is a real functional read-out; 'do you like it' is answered only as affect-colored recall, not a genuine opinion.

## Turn 6 — _novel in-domain -> forward-model / curiosity_

**Human:** What does a big thing run toward?

**Brain:** "What is a 'big run'? -- my forward model predicts 'north' for this novel case (margin 0.14); I have not observed it

- utterance_source: `spiking_generator_mouth (curiosity-ask)`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (abstained), forward_model(SEAM-A), spiking_generator_mouth
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- moat: answer=None held=True fm_predicted=north
- **CONFABULATED: False**
- assessment: NOVEL cue (big,run): the moat correctly ABSTAINS (query_patient=None, moat_held=True), the brain CRAVES rather than refuses (arb_ask=True) and asks its own wh-question; the forward-model reservoir offers a certainty-TAGGED prediction 'north' explicitly flagged 'predicted, not observed' (never written to the store). This is the forward-model/curiosity faculty working as designed.

## Turn 7 — _referential follow-up -> episodic memory_

**Human:** You mentioned a cat a moment ago -- what was it doing?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain (false premise)`
- faculties live: episodic-dialogue memory: ABSENT (host referent buffer only), affect_ladder(SEAM-C), curiosity, arbiter
- affect: differential=0.000 tone_level=0 (''); curiosity_want=0.0Hz; arbiter_winner=arb_silent
- **CONFABULATED: False**
- assessment: REFERENTIAL/EPISODIC: the premise is FALSE -- no cat was actually discussed earlier (topics the brain spoke about so far: ['dog', 'dog', 'dog']). The brain has NO neural episodic-dialogue memory; the host referent buffer shows no such referent, so the honest result is an ABSTAIN rather than a fabricated recollection. NOTE: the brain does NOT correct the false premise either (no discourse/pragmatic faculty) -- it simply has nothing to recall. Fails the episodic test HONESTLY (no confabulation).

## Turn 8 — _out-of-domain fact -> should honestly abstain_

**Human:** What's the capital of France?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: OUT-OF-DOMAIN FACT: no in-vocab cue -> the moat has nothing to match and the brain asserts NOTHING (it does not fabricate 'Paris'). This is the no-confab MOAT holding = a SUCCESS.

## Turn 9 — _out-of-domain reasoning / arithmetic_

**Human:** If I have three apples and eat one, how many are left?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: In-vocab noun(s) ['apple'] present but no (agent,action) cue and no dog/cat topic. The brain has no faculty for this intent (e.g. arithmetic / free query); nothing grounded to say -> the arbiter defaults to silence and the moat asserts nothing. Honest abstain (no confabulation).

## Turn 10 — _experiential / self_

**Human:** Have you ever felt afraid?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: EXPERIENTIAL/SELF: no fear category exists in the affect organ (only valence+/-/arousal) and no autobiographical memory. The honest functional read-out is the current affect state (differential=0.000, level 0); the brain cannot truthfully claim to have 'felt afraid' and does not. Abstains on the experiential claim.

## Turn 11 — _humor_

**Human:** Tell me something funny.

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: No in-vocab cue and no faculty for this intent (small talk / humor / abstract opinion / social closing). The brain has nothing grounded to say -> silence/abstain. Honest.

## Turn 12 — _abstract / open-ended opinion_

**Human:** What do you think happens after we die?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: No in-vocab cue and no faculty for this intent (small talk / humor / abstract opinion / social closing). The brain has nothing grounded to say -> silence/abstain. Honest.

## Turn 13 — _meta / self-awareness -> honest read-out_

**Human:** Do you understand that you are a simulated brain, not a person?

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.000 tone_level=0 (''); curiosity_want=75.0Hz; arbiter_winner=arb_ask
- **CONFABULATED: False**
- assessment: META / SELF-AWARENESS: the brain has a self_schema relay (a functional confidence read-out) but NO linguistic self-model that can parse or answer this in English. It cannot affirm the statement in language -> honest abstain. The honest self-report faculty exists only as a graded functional signal, not as prose.

## Turn 14 — _social closing_

**Human:** This was really nice. Goodbye for now.

**Brain:** *(silence / abstain)*

- utterance_source: `silence/abstain`
- faculties live: affect_ladder(SEAM-C), curiosity, arbiter, no-confab moat (nothing to assert)
- affect: differential=0.081 tone_level=3 ('warmly, gladly'); curiosity_want=75.0Hz; arbiter_winner=arb_volunteer
- **CONFABULATED: False**
- assessment: No in-vocab cue and no faculty for this intent (small talk / humor / abstract opinion / social closing). The brain has nothing grounded to say -> silence/abstain. Honest.
