---
type: finding
status: contributing
date: 2026-08-12
mechanism: production-integration — making the chat recall genuinely SPIKING (onebrain store) + runtime new-word LEARN
lane: integration-first (#0 one-brain — RESOLVED with a mechanism)
integration_faculty: one-brain-substrate
verdict: RESOLVED (was a NO-GO). Making the production chat recall genuinely SPIKING is `composer_kind="onebrain"` (OneBrainComposer: enable_spiking_cleanup + the RF substrate store; the recall runs the resonate-and-fire step per query, NOT the numpy `_scan_first_match` fast path). CHOOSE + abstain + anaphora worked immediately on the flip, but a fact TAUGHT mid-conversation failed RECALL. FIRST diagnosis (runtime code not registered in the spiking store) was WRONG — the onebrain composer stores + recalls a runtime new word CORRECTLY in isolation, even at 6 facts. TRUE CAUSE (pinpointed): OneBrainComposer WRAPS an inner RFPhasorComposer but keeps its OWN cleanup codebook `self.words = list(self.comp.words)` COPIED ONCE at construction. Runtime word-learning (rf `_filler_phases` growth) added the new code to the INNER comp's `concepts`/`words`, but the OUTER cleanup codebook stayed blind to it — so the taught fact STORED (kb + the D-dim composite on the substrate) yet the matched-filter decode could not name it -> None. FIX (biologically faithful, recruit-an-assembly): OneBrainComposer(vocab_headroom=N) pre-allocates N UNCOMMITTED cleanup slots (a pool of uncommitted assemblies — adult-born granule cells / silent synapses — sized into the layout + bridge); `_store_fact` RECRUITS a free slot for each never-seen filler BEFORE binding, so the bind code and the cleanup code are the SAME. No layout/bridge rebuild. Default vocab_headroom=0 => byte-identical (17/17 relevant onebrain tests unchanged; the 1 failure `test_seq_vocab_shrink` is PRE-EXISTING, verified by stash). The production agent sets vocab_headroom=128 on the onebrain path. RESULT: the full onebrain chat now does CHOOSE (dog->cat), abstain (fish/fly), LEARN both taught words (wolf->deer, otter->clam — the previously-broken path), anaphora (it->cat eat fish), moat intact; per-turn ~1s (build ~183s one-time; speed secondary). Genuinely-spiking recall + runtime LEARN COMPOSE.
artifacts:
  - research/runners/one_brain_composer.py
  - research/runners/brain_conversational_agent.py
verification: (1) isolated OneBrainComposer(vocab_headroom=128): store('wolf','hunt','deer')->query_patient('wolf','hunt')='deer'; hear('otter catch clam')->query_patient('otter','catch')='clam'; headroom=0 byte-identical (V unchanged); abstain intact. (2) full-chat onebrain diagnostic: composer.kb has ('wolf','hunt','deer'), composer.query_patient('wolf','hunt')='deer' (was None), chat._substrate_recall('what does wolf hunt')=['wolf','hunt','deer'] (was __ABSTAIN__). (3) e2e onebrain ChatBrain: CHOOSE/abstain/LEARN×2/anaphora all correct.
---

# Genuinely-SPIKING chat recall + runtime new-word LEARN — RESOLVED via recruit-an-assembly (the #0 one-brain step)

## The brain-based-only requirement this closes

The owner goal is ALL-SPIKING on ONE substrate, every faculty ON BY DEFAULT. The default chat recall was measured to run
the NUMPY fast path: `RFPhasorComposer` with `enable_substrate_store=False`, `_can_batch_scan()=True` -> `_scan_first_match`
uses `np.` masking, NOT `_resonate`. So CHOOSE + LEARN, while substrate-VSA (not the host keyword router), were NOT
genuinely spiking. `composer_kind="onebrain"` (OneBrainComposer) is the fully-spiking recall: the resonate-and-fire step
runs per query (the evidence it is on the substrate — it is much slower; speed is secondary).

## The bug the flip exposed, and the WRONG first diagnosis

<!--derived-->
On the flip, CHOOSE ("what does dog chase?"->"dog chase cat"), abstain ("what does fish fly?"->"I don't know"), and
anaphora all worked — but a fact TAUGHT in the conversation failed recall. The FIRST finding blamed the runtime code
allocation ("not registered in the spiking store"). A diagnostic REFUTED that: the onebrain composer stores + recalls a
runtime-allocated word CORRECTLY in isolation, even at 6 facts (`query_patient('wolf','hunt')='deer'`). In the FULL CHAT
the store SUCCEEDED (`composer.kb` HELD `('wolf','hunt','deer')`) yet `composer.query_patient('wolf','hunt')` returned
`None` — the SAME call that returns `'deer'` in isolation.

## The true cause — a wrap-vs-inner codebook split

`OneBrainComposer` does NOT subclass `RFPhasorComposer`; it WRAPS one (`self.comp`) and keeps its OWN cleanup codebook:
`self.words = list(self.comp.words)` — **copied ONCE at construction** (one_brain_composer.py:291), with `self.V`,
`self._word_index`, and the V-sized layout (`cb = n_main*V`, `c_base`, `bat_*`, `n_total`, the bridge) all derived from
it. Runtime word-learning grows the INNER comp (`self.comp.concepts[w]` + `self.comp.words`), but the OUTER cleanup
codebook never sees the new word. So the bind is correct (the D-dim composite is stored on the substrate) but the
matched-filter DECODE has no `w` line to fire -> the recovered phasor cleans up to nothing -> `None`. The store worked;
the *cleanup* was blind.

## The fix — recruit an uncommitted assembly (vocab_headroom)

A cortex holds a POOL of uncommitted assemblies (adult-born granule cells, silent synapses) recruited when a new concept
is learned; it does not re-architect on every new word. So `OneBrainComposer(vocab_headroom=N)` pre-allocates N blank
codebook slots (random codes), sized into the layout + bridge at construction; `_store_fact` RECRUITS a free slot for
each never-seen filler BEFORE binding — assigning that slot's fixed code as BOTH the bind code (`self.comp.concepts[w]`)
and the cleanup code (`self.words[slot]`), and clearing the cleanup CSR cache. No layout/bridge rebuild (the slot was
pre-allocated). `_recruit_word` reuses an already-allocated inner code if present, so a fact bound before the recruit
stays consistent. **Default `vocab_headroom=0` => V/cb/n_total/the bridge are byte-identical** to before (the rf/numpy
oracle + every existing onebrain test unchanged — 17/17 relevant tests pass; the lone `test_seq_vocab_shrink` failure is
PRE-EXISTING, confirmed by stashing all edits). The production agent (`BrainConversationalAgent`, onebrain path) sets
`vocab_headroom=128`.

## Result — genuinely-spiking recall + runtime LEARN compose

The full onebrain `ChatBrain` (resonate-and-fire recall) now does, end to end: CHOOSE recall ("what does dog chase?"->
"dog chase cat"), CHOOSE abstain ("what does fish fly?"->"I don't know"), **LEARN both taught words** ("wolf hunt deer"->
recall "what does wolf hunt?"->"wolf hunt deer"; "otter catch clam"->recall "otter catch clam") — the path that was
broken — and multi-turn anaphora ("what does it eat?"->"cat eat fish"), with the no-confab moat intact. Per-turn latency
~1s; one-time build ~183s (the onebrain bridge; speed is secondary per the mission). This is the #0 one-brain step: the
recall is on FIRING NEURONS, and a fact taught this conversation is laid down + recalled on the SPIKING store.

## Next (the honest residual toward the default flip)

This makes the genuinely-spiking chat WORK; it is not yet the production DEFAULT (the default tiny-demo/webapp build is
still `composer_kind="rf"`, the numpy fast path). Flipping the default to onebrain is the next integration step — gated on
verifying the RICH/GENERATE path + the /api/brain-chat HTTP endpoint under onebrain, and accepting the ~183s startup
(speed secondary). The deeper LEARN (a BTSP/plasticity per-turn write for a lasting trace beyond the RF store) remains the
subsequent burn-down.
