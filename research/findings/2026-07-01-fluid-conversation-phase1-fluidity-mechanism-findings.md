# Fluid conversation — Phase 1: the fluidity mechanism space (v1 NEGATIVE → the broadened-veto + sampling fix)

**2026-07-01 (autonomous night; owner's fluid-conversation priority).** Phase 0 established that the ~21M TinyStories
generator, behind the per-token grounded veto, is **grounded + non-vacuous + moat-intact** (SCALE-CONFIDENT). This doc
opens Phase 1 — the *fluidity* gap (the owner's north star: fluid, LLM-like, multi-sentence conversation, not a rigid
single proposition) — and reports the first (negative) attempt + the precise diagnosis that reframes the mechanism.

## v1 attempt (NEGATIVE, informative): constrained core + free continuation
`_fluidconv_phase1_grounded_continuation_derisk.py` — GATE → **constrained core** (Phase-0 render of the gated
proposition) → **free continuation** (unconstrained 21M) → **VERIFY** (re-parse full text, flag ungrounded
known-entity SVOs). Result 3 seeds: fluid+grounded **0/5**, drift-caught 3/3, untaught-abstain 3/3, and — inverted from
expectation — the grounded path asserted MORE ungrounded facts than the free baseline.

## The diagnosis (from reading the actual generated text — the decisive step)
Two distinct failures, both fundamental, neither a mere tuning bug:

1. **The Phase-0 constrained decode is INHERENTLY terse — grounded ≠ fluent.** The veto allows only *the one
   proposition's content words* + function words. Once "dog eat meat" is emitted, the ONLY legal next tokens are those
   3 words + function words, so any continuation is FORCED to loop: cores came out `"and meat and meat and meat…"`,
   `"the fox chase the fox … very very very very"`. **Phase 0 PASSED its bars anyway** because "grounded + ≥2 distinct
   on-proposition content words" does NOT require fluency — *Phase 0 measured grounding, not fluency*. The fluency
   claim in the Phase-0 finding came from SEPARATE free temp-0.8 samples, not the constrained path. So constrained
   decode is fluent only *within* one short proposition; it cannot elaborate.
2. **Free (unconstrained) continuation is fluent but ungrounded — it hallucinates or falls to canned openings.** The
   free 21M continued into `"the fox chase tree"` (ungrounded), `"the cat eats the meat too"` (wrong fact), or bailed
   to its highest-frequency training sequence `"Once upon a time, there was a little girl named Lily…"`. And greedy
   decoding (the harness default, for reproducibility) *loops* (`"They are just different, but they are not mean."`×N).
   VERIFY correctly FLAGGED the hallucinations (drift-caught 3/3) — the moat-as-a-plus works — but that means the fluid
   output would be *withheld*, so it is not usable as-is.

**The root cause in one line:** the constrained veto's vocabulary (one proposition) is too SMALL to say anything
fluent beyond that proposition, and removing the veto entirely loses grounding. A raw TinyStories *completion* model
also cannot do Q&A (it is not instruction-tuned) — it only continues narrative.

## The reframe → v2 (the fix, no new model, no fine-tune)
**Broaden the grounded veto from ONE proposition to the queried subject's ENTIRE retrieved knowledge set (all the
brain's facts about it), and SAMPLE with temperature + a repetition penalty instead of greedy.** Then the generator can
fluidly *weave multiple real facts* — e.g. for the subject "dog" (facts: eat→meat, chase→cat, like→bone, is→big) it can
produce *"The big dog ate the meat and chased the cat."* — **fluent** (sampling, broad vocabulary, no forced loop) AND
**grounded** (every content word is drawn from the subject's real facts; the veto still forbids ungrounded entities).
This is precisely the **retrieval-augmentation** frame from the roadmap (GAP B): the brain retrieves the grounded
knowledge; the generator is conditioned on (here: veto-restricted to) exactly that knowledge; abstention stays the
honest breadth boundary. The moat is preserved by construction (the veto vocabulary IS the grounded set) AND checked
post-hoc by VERIFY.

**v2 mechanism (building next):** allow_text = ∪(all facts about the queried subject) + FUNCTION_WORDS; decode with
temperature (~0.7–0.9) + a repetition penalty over the constrained logits; measure (a) FLUENCY (distinct-token /
low-repetition, reads as natural multi-fact prose), (b) GROUNDED (VERIFY: every asserted known-entity SVO ∈ the store),
(c) the moat (untaught subject → empty knowledge set → nothing to say → abstain), (d) load-bearing (unconstrained
sampling drifts ungrounded; the broadened veto does not). ≥3 seeds.

**Honest open question v2 must answer:** does a broadened veto + sampling produce genuinely *natural* multi-fact prose,
or does the tiny 21M model still need a small **fact-rendering/dialogue fine-tune** (the roadmap's "brain-train it"
lever) to learn the conversational rendering register? If v2 is still stilted, the fine-tune is the next lever (still a
small, minimized, brain-trained generator — not the Qwen fallback). Either outcome is a real finding.

## v2 RESULT (NEGATIVE) + the decisive reframe
`_fluidconv_phase1_broadened_veto_derisk.py` — broadened the veto to the subject's whole knowledge set + temperature
sampling + repetition penalty. Result 3 seeds: grounded+multifact **0/4**, and the veto output is **word-salad**:
*"the cat and his chase them to a very chase them at their yes of the chase to the chase and the chase and the
chase…"*. Even the subject's whole knowledge is only ~7 content words; restricting the lexicon that hard cannot form
grammatical English (the model loops on the few legal words + emits `yes`/`-` artifacts). **Second confirmation that a
per-token grounded veto is fundamentally incompatible with fluency** (the vocabulary is too small) — the Phase-0
honest ceiling ("constrained decode TRADES fluency for faithfulness BY DESIGN") is now doubly verified.

**The decisive contrast that reframes the whole approach — the FREE (unvetoed) baseline:** *"the bird started to fly
away. It followed it back and forth until it finally stopped. The bird was so happy because now it could see the
sky."* and *"the cat started to run away. It ran as fast as it could, but soon came across a big tree and decided not
to stop running…"* — **genuinely fluent, coherent multi-sentence prose, and it asserts ZERO false known-entity facts**
(free-path ungrounded = 0). The free 21M generation is fluent AND non-hallucinatory (it stays in
narrative-description register — fly, run, happy, sky — without making false SVO claims about the animals). **It
simply does not render the SPECIFIC grounded fact on command** (a base completion model is not instruction-tuned).

**⇒ The consolidated Phase-1 verdict:** the fluency is already there and it does not hallucinate facts; the only gap is
"render the specific grounded fact fluently." Two families of failure are now closed (hard-veto: word-salad; free-gen:
off-fact). The path forward is NOT a third veto variant — it is one of:
1. **Prompt-conditioning + post-hoc VERIFY (cheap, no training) — TEST NEXT (v3):** give the free generator a NATURAL
   fact-lead (the grounded fact stated as an opening — via the brain's own neural serial-order render / `describe()`,
   biology-based word ordering, NOT a host template) and let it continue fluently; VERIFY re-parses the whole thing
   and rejects any NEW ungrounded assertion. The free-baseline evidence (fluent + non-hallucinatory) predicts this can
   work with zero training.
2. **The retrieval-augmented render fine-tune (the roadmap's "brain-train it" lever) — if v3 is insufficient:** a small
   fine-tune of the 21M on (fact-in-prompt → fluent sentence) + (question + fact → grounded answer) pairs over a BROAD
   vocabulary, so it learns the render/answer FORMAT generally (generalizing, not memorizing specific facts). Still a
   minimized, brain-trained, brain-gated generator — the honest sweet spot, not the Qwen fallback.

## v3 RESULT — **GO**: prompt-conditioned free-gen + post-hoc VERIFY = fluid AND grounded (no fine-tune, no veto)
`_fluidconv_phase1_conditioned_freegen_derisk.py` — the reframe realized: state the gated fact as a natural lead (the
brain's neural word order + a minimal surface scaffold), let the 21M continue FREELY (fluent, unvetoed), and re-parse
the whole text post-hoc (moat-as-a-plus). **Result 3 seeds: fluid+grounded 15/15, drift-caught 3/3,
untaught-abstain 3/3 — GO.** Samples are genuinely fluent grounded prose:
- *"The dog eats meat. It is not in the stick! It is a thick stick for Max to sleep. He licks Tom and Lily. 'Look, it
  is a bone! Let's eat it!' Lily says."*
- *"The bird eats seed. It is not in the pot. 'Yes, it is. But I am brave. And I can do it.' Tom says. He takes a
  piece of bread from his pocket."*

Each starts with the brain's grounded fact and continues fluently WITHOUT asserting a false animal-fact; adversarial
wrong-fact leads ("The cat eats meat.") are FLAGGED by post-hoc VERIFY; untaught cues abstain (the GATE gives no
lead). One initial "miss" (seed 43) was root-caused to a VERIFY extractor **false positive** — the multi-fact
extractor bound `fish · make · cat` ACROSS three sentence boundaries ("fish. They make a big mess. …her cat"); the fix
made `_extract_all_svos` **sentence-boundary-aware** (an SVO must be within one sentence) → 15/15. That is a strict
strengthening of VERIFY (fewer false positives, no real within-sentence assertion missed).

**⇒ Phase-1 fluidity mechanism SETTLED:** fluent grounded RENDERING is achievable NOW, cheaply, with the ~21M
generator, via **prompt-conditioning (not a veto) + free (fluent) generation + post-hoc VERIFY** — the moat kept as a
post-hoc PLUS. This is the honest fluid-grounded substrate the owner's north star needs.

**Honest scope + what's still open for "talk to it like an LLM":** v3 does fluent grounded *rendering* (state a fact +
fluent narrative), NOT (a) focused conversational **Q&A** (the base model *continues a story*, it does not *answer a
question* — it rambles on-topic-but-unfocused, e.g. "…Sam is sad. He does not like the fish."), (b) **multi-turn**
coherence, or (c) open-topic **breadth**. The next high-leverage lever is the roadmap's **retrieval-augmented
render/QA fine-tune** (teach the 21M the Q + retrieved-fact → focused grounded answer FORMAT over a broad synthetic
vocabulary, generalizing not memorizing) — still a minimized, brain-trained, brain-gated generator. Then assemble the
full turn: free-text question → brain comprehend + retrieve/abstain → fluent focused grounded reply → VERIFY.

**Artifacts:** `_fluidconv_phase1_grounded_continuation_derisk.py` (v1 + the sentence-aware `_extract_all_svos`);
`_fluidconv_phase1_broadened_veto_derisk.py` (v2); `_fluidconv_phase1_conditioned_freegen_derisk.py` (v3, GO);
results `_fluidconv_phase1_{grounded_continuation,broadened_veto,conditioned_freegen}.json`. NO `sim/` edit;
reuse-by-import. The two negatives + the GO together map the fluid-grounded mechanism space: per-token veto ✗ (kills
fluency), free-gen alone ✗ (off-fact), prompt-condition + post-hoc-verify ✓.
