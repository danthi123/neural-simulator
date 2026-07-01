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

**Artifacts:** `research/runners/_fluidconv_phase1_grounded_continuation_derisk.py` (v1, negative);
`research/findings/raw/_fluidconv_phase1_grounded_continuation.json`. NO `sim/` edit. Reuse-by-import.
