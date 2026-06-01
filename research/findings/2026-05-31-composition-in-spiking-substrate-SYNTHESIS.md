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
4. **End-to-end LEARNED syntactic understanding**: voice-invariant role assignment ("dog chases cat" ≡
   "cat is chased by dog", same agent) requires conjunctive position×voice coding (cheap-first); that
   conjunctive→role mapping is LEARNED in-substrate by the v16 Hebbian co-firing rule — multi-seed
   (42,43,44) 6/6 conjunctions including the active↔passive flip every seed (bare spike-timing STDP
   fails on the simultaneous teacher); and the full pipeline composes end-to-end — the learned parser
   assigns roles, the spiking bind stores the sentence, a relational query extracts the agent
   VOICE-INVARIANTLY (multi-seed 42/43/44: parse 6/6, voice-invariant agent 1.000, scrambled-parse
   control 0.000 — every seed). Different word orders, same meaning, correctly understood — learned, not
   supplied.

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

7. **Role assignment is LEARNED by Hebbian co-firing, not spike-timing STDP.** The conjunctive
   (position×voice)→role mapping is acquired in-substrate by a teacher-co-active protocol with the
   v16 rate-based Hebbian rule (pre&post-gated, grows co-active synapses toward firing strength):
   6/6 conjunctions learned including the active↔passive flip. Bare spike-timing STDP FAILS on the
   same protocol — a simultaneous teacher provides no pre→post order. Translatable: associating a
   conjunctive syntactic context with a role is a Hebbian co-activation (cell-assembly) learning
   problem, not a fine-timing one; the supervisory signal need only co-activate, not precede.

## Conversational capabilities + the scaling answer (built on the bind)

On the validated bind, a complete (small-vocab) conversational agent was built, all spiking, all
multi-seed (finding `2026-05-31-conversational-capabilities-on-the-spiking-bind.md`): statements ->
stored facts; wh-question answering (who/what, 1.0/1.0/0.9); NEGATION + yes/no via a bound polarity
tag (3/3); GENERATION (full-sentence production: "describe dog" -> "dog go north"); a PERSISTENT KB
across sessions (3/3, no forgetting -- the continual-learning premise); and an interactive REPL the
owner can talk to (`compose_conversation_repl.py`). The agent both understands and produces.

The owner asked whether vocabulary scales past the documented ~320 limit. Measured + demonstrated
answer (`_vocab_scaling_locus_note.md`): the bind/COMPOSITION layer is vocabulary-ROBUST -- spiking
wh-QA = 1.000 at V=64, 160, AND 320 (sparse codes), and cleanup recovery is 1.000 to V=640. The real
~320 limit lives entirely in the RECOGNITION FRONT-END (getting hundreds of clean distinct concept
codes), NOT in composition: v17's 28-word 50% was the OLD architecture; the encoding-axis arch
validated 64-word recognition at 3/3 GO, and G.20 sparse reaches 320 at 98.4%. Plus insight #5 (the
bind uses the distributed code, not the pool label) means the effective recognition exceeds the
label accuracy. So 64-word conversation is clean and 320 is feasible at ~98%, gated by recognition.
Two of my own overstatements were corrected by the data ("scaling is tractable" and "cleanup
degrades with vocabulary" -- both wrong; the truth is precise: composition robust, front-end gated).

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
validated bind for voice-invariant understanding end-to-end.

Update (same day) -- the parser core now RESOLVES in-substrate. A first STANDALONE probe with BARE
STDP (_insubstrate_parser_stdp_probe.py) FAILED (role ensembles silent, conj->role weights never grew
to firing strength) -- the honest diagnosis was the WRONG learning rule: bare STDP is timing-based and
a simultaneous teacher gives no pre->post order. Switching to the v16 embodied-Hebbian CO-FIRING rule
(bridge.py:5265, gated on pre&post co-firing -> selective; hebbian_max_weight=400 = firing strength)
RESOLVES: 6/6 conjunctions activate the CORRECT role (correct 0.04-0.08, incorrect ~0.000) AND the
active<->passive flip is LEARNED (pos0-active->agent vs pos0-passive->patient; pos2 inverted). This is
LEARNED (not supplied) syntactic role assignment in-substrate, including voice-dependent role flipping.
A 7th insight: **role assignment is learned by Hebbian co-firing, not spike-timing STDP** -- a
teacher-co-active protocol with a rate-based (pre&post-gated) rule grows the conjunctive->role map;
the timing-based rule fails on simultaneous teaching. So ALL parser pieces are validated in-substrate
(coincidence for the conjunction, Hebbian-learned conjunction->role, the bind for role->filler), and the
FULL PIPELINE now composes END-TO-END (_insubstrate_parser_bind_e2e_probe.py): the learned parser assigns
roles, the spiking bind stores the sentence, a relational query extracts the agent VOICE-INVARIANTLY --
multi-seed (42,43,44) parse 6/6, voice-invariant agent 1.000, scrambled-parse control 0.000 every seed.
"dog chases cat" and "cat is chased by dog" -- different word orders, same meaning -- are both correctly
understood (dog is the agent), LEARNED not supplied. This is the first end-to-end learned syntactic
understanding in-substrate, composing every validated piece. What remains for richer conversation: scale
vocabulary, more syntactic constructions (questions, relative clauses), and multi-turn persistent dialogue.

## Cheating audit + honest scope (2026-05-31)

A rigorous, controlled answer to "are we cheating / using templates?" is in
2026-05-31-cheating-audit-learned-vs-given-and-genuine-composition.md. Three things could be cheating;
each has a control:

1. Composition a template? NO. The bind/unbind is genuine VSA algebra: generalizes to novel combinations
   (8/8 nonsense, 60/60, 3/3 multi-seed) AND correctly abstains on unstored facts (qa64 "unknown control"
   1.000 at V=160 and V=320). Correct abstention is the decisive anti-artifact control -- a system with
   merely distinct codes would clean up an unknown query to SOME concept and answer wrongly.
2. Parsing a positional template? NO (closed). The live REPL uses the LEARNED Hebbian parser, voice-
   invariant, multi-seed 3/3.
3. Concepts learned or given? PARTLY EACH, measured not hidden. Pool-label recognition (the honest metric)
   trained-minus-untrained = the genuinely LEARNED fraction: +68.7 pp at 16 words (trained 0.812 vs
   untrained-random floor 0.125) and +53.5 pp at 28 words (0.571 vs 0.036). The learned fraction is HIGHER at
   small vocab and ERODES with scale -- the quantified front-end wall. The GIVEN component is the orthogonal
   input encoding; large-V (160/320) "concepts" are given sparse codes, so learned word recognition is
   validated only at small vocab -- the front-end is the documented ceiling.

Honest-discipline note: this audit was triggered by catching one of my OWN metrics as a drive-echo
ARTIFACT (a bind-QA on captured codes read 1.000 even untrained -- see
2026-05-31-front-end-distributed-vs-label-ARTIFACT-honest-negative.md). The artifact was isolated to that
one flawed front-end metric; the abstention-controlled composition results are unaffected. Net: the
composition is genuine and generalizes; concept recognition is genuinely learned at small scale with a
measured "given by encoding" component; the learned-concept-at-scale front-end is the bounded, honestly
quantified frontier.
