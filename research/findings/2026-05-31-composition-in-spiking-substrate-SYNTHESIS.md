# Composition in a spiking brain-analogue substrate — synthesis + biology-translatable insights (2026-05-31)

**Goal frame:** artificial life with a proper brain analogue; biology-translatable insights are the
scientific deliverable; conversational capability is instrumental; honest negatives under strict
biology are real results. This document synthesizes one autonomous arc against that frame.

**One line:** compositional binding — the operation a brain needs to represent structured thought
and language — runs IN the project's spiking substrate, computed by spiking neurons, validated
multi-seed and adversarially reviewed; and the arc yielded a set of biology-translatable insights
about how (and how far) a brain-analogue substrate can compose.

## The capability ladder built this arc (all multi-seed unless noted, all on real concept codes)

1. **Spiking bind/unbind** (role (x) filler, by threshold coincidence neurons): RESOLVES, capacity
   K=4 (clean-AND operating point) extending to K=6 (higher firing rate). Adversarial reviewer CLEAR
   (7 exploit classes). Finding: `2026-05-31-in-substrate-spiking-bind-unbind-VALIDATED.md`.
2. **Relational fact-memory** (a queryable subject/verb/object knowledge base): multi-seed 3/3,
   scales to ~12 facts (vocab-limited). Answers "what does dog chase?" via spiking unbind + cleanup.
3. **End-to-end from LIVE TEXT**: drive each word through the trained concept-pool bridge -> live
   concept-pool activity -> spiking bind -> relational query -> answer. Multi-seed (42,43,44) all
   1.000 (single/relational/control), matching the cached-code baseline; front-end recognition
   15-16/16 -- the bind is robust to the 0-1 recognition mislabel per seed.
4. **Parser representational gate**: voice-invariant role assignment ("dog chases cat" ≡ "cat is
   chased by dog", same agent) requires conjunctive position×voice coding (cheap-first).

Owner-facing demos: `compose_spiking_bind_demo.py`, `compose_relational_memory_demo.py`,
`compose_live_text_kb_demo.py`. Capability widget: pillar n=111.

## Biology-translatable insights (the deliverable)

1. **The compositional bind is coincidence detection.** role (x) filler (the ±1 Hadamard / VSA
   binding) is computed EXACTLY by a spiking neuron firing only when it receives BOTH an active role
   AND an active filler, with a tonic hyperpolarizing bias setting the AND threshold. Binding does
   not need a special mechanism — it is the dendritic/somatic coincidence operation cortex already
   performs. Graded gating (rate ∝ filler magnitude when the role gates it) makes it multiplicative
   gain-modulation, also well-documented.

2. **ON/OFF opponency = the project's mean-centering, and it is load-bearing.** Representing signed
   values as firing-above vs below baseline (retinal/thalamic opponency) is what lets ≥0 spiking
   operations realize a signed algebra; re-canonicalizing the superposed bound through opponency
   (lateral inhibition between ON and OFF channels) is the common-mode removal that keeps superposed
   bindings legible.

3. **Binding capacity is set by firing-rate resolution, not readout time.** A longer readout window
   does NOT raise the number of simultaneously-bound items (a falsified prediction — corrected
   honestly); a higher coincidence firing rate (more dynamic range) does, lifting capacity from ~4 to
   ~6 superposed bindings (the Miller 7±2 range). Translatable: working-memory binding capacity
   should track the dynamic range / gain of the binding population, not integration duration.

4. **Separate storage is the universal mechanism for structure; flat superposition is not.** Both
   multi-fact memory AND hierarchy (nesting) FAIL when packed into one superposed vector (the
   multi-hop / SNR wall: a relational query over superposed facts ~0.48; flat nested "phrase as
   filler" descent ~chance). The SAME tasks succeed when each structured item is a SEPARATE bound
   ensemble retrieved by cue (relational query 1.000; KB scales to a dozen facts). Translatable:
   the brain should store distinct facts/phrases as distinct ensembles bound by association, not sum
   them — consistent with hippocampal pattern-separated episodes.

5. **The bind is robust to recognition errors because it uses the distributed code, not the label.**
   End-to-end live text recovers at 1.000 even when the concept-pool "recognition" readout mislabels
   1/16 words: the bind operates on the full distributed population vector, which stays separable
   even where the winner-take-all pool label is wrong. Translatable: downstream composition can be
   more reliable than the categorical readout that sits on the same population.

6. **Syntactic role parsing requires conjunctive position×voice coding (mixed selectivity).**
   Position alone — or position + voice added — cannot represent the active↔passive role flip (an
   interaction); only the conjunction can. Translatable: role assignment in language cortex needs
   mixed-selectivity neurons conjoining word-position with syntactic-voice cues, not a positional
   readout — and the substrate's distributed codes already are conjunctive-capable.

## Honest scope and boundaries

- Fixed-wiring composition: generalizes by VSA construction (no training); the validated learning
  rules are reused where learning is involved, with no new autograd anywhere.
- Roles are SUPPLIED, not yet parsed from raw input (the parser is the next arc; its representational
  requirement is settled, its spiking integration is pending).
- 16-word vocabulary, canonical subject/verb/object; cue-based retrieval, not open-ended reasoning.
- The linear inter-phase memory (superposition sum, opponency) is captured-rate arithmetic, each
  step itself a linear/lateral-inhibition operation realizable in-substrate.

## What is next (teed up)

The learned role-filler parser: STDP-acquire the conjunctive (position×voice)→role mapping from
example sentences (voice detected by function-word PRESENCE + relative position — both tractable
features, NOT the substrate's bounded ordered-sequence processing), then wire parsed roles into the
validated bind for voice-invariant understanding end-to-end. All component mechanisms (coincidence
for the conjunction, STDP for the small mapping, the bind) are validated; the next arc is their
integration plus an honest test of whether STDP acquires the conjunction and it composes.
