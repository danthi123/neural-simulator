# Roadmap: genuine conversation + thinking on the sim (2026-06-27 deep-research synthesis)

Synthesis of a 4-front read-only deep-research effort (the project's standing-practice gate at a new direction;
owner-requested). Each front reviewed the biology catalog (`sim-catalog/references/feature-catalog.md`) + Kandel 6e
+ current literature, grounded in the sim's VERIFIED current state (read from code, not docs). Cluster docs:
- `2026-06-27-conv-thinking-research-comprehension-representation.md` (front 1)
- `2026-06-27-conv-thinking-research-discourse-pragmatics.md` (front 2)
- `2026-06-27-conv-thinking-research-reasoning-thinking.md` (front 3)
- `2026-06-27-conv-thinking-research-generation-control.md` (front 4)

(Housekeeping: `glossary.md` is absent from the catalog dir; all four fronts substituted literature. The catalog's
"Sim status" lines are core-sim-dated — each front verified current runner state directly in code.)

## The diagnosis: the bottleneck is REPRESENTATION, not parsing or wording

The four fronts converge on ONE root. The sim's PARSING front-end is the most-built part of the whole stack
(voice-invariant comprehension, multi-frame word order 6/6, Bates-MacWhinney multi-cue robustness, embedded
clauses) and WORDING is now handled (Path B's grounded fluency-LLM, GO). The gap is what they operate ON:

1. **Bare, type-keyed SVO triples.** The composer's entire relational alphabet is
   `(agent, action, patient, polarity, attribute, attribute2)`; extraction collapses all obliques into one slot,
   discarding the preposition. So there is no verb argument structure ("go TO the park"), no tense/aspect, no events.
2. **Types, not tokens.** Only the generic concept "boy" exists — no entity instances — so "which boy?" is
   literally unrepresentable.
3. **Storage fused with the active buffer.** The working-memory region is sized PER-VOCABULARY-ITEM (the balloon
   bug, `content_selection_spiking.py:307,359`: `n=max(600, 60*len(vocab))`) instead of a fixed ~4±1 pointer
   buffer. Biology separates unbounded LTM storage from the small PFC buffer; the sim fuses them.

(2) and (3) are the same modeling error as (1) at different layers: **the representation is a skeleton.** This is
why the owner's three chat complaints — "the boy goes park" (no GOAL role), "where does the boy go" (no wh-parse),
"which boy?" (no instances) — are all the same root, not three unrelated bugs.

## The convergence (this de-risks the plan)

Four independent fronts landed on two keystones:
- **Verb-frame argument structure** — fronts 1 (comprehension) AND 4 (generation) BOTH ranked it #1, both
  low-risk + reuse-by-import.
- **Entity instances** — fronts 1 (representation) AND 2 (discourse) BOTH named it the keystone everything
  social/referential is downstream of.

And the machinery is largely ALREADY in the codebase: the FHRR role-binding; the SHIPPED D.14 engram "barcode"
API; `OrderedPositionWM` (a fixed-slot pointer buffer); the biased-competition WTA buffer; the Bogacz-Brown
familiarity gate; DG/CA3 pattern separation/completion (validated). **Most of the near-term roadmap is WIRING
validated pieces, not new mechanism.**

## Biology — the buildable frameworks

- **Hagoort MUC** (Memory-Unification-Control): each verb's structural FRAME lives in temporal cortex (Memory);
  Broca/LIFG binds fillers into it (Unification); N400=Memory/semantic, P600=Unification/reanalysis. The
  argument-structure target IS the Memory store.
- **Hippocampal episodic-index "barcode"** (Quian-Quiroga concept cells; eLife 2024): the type→token mechanism.
  The sim's D.14 engram API is a barcode.
- **Fixed-capacity PFC WM** (Cowan ~4±1; Lisman-Idiart ~7): a small buffer of POINTERS into unbounded cortical
  storage — kills the balloon by construction.
- **Mentalizing network ≈ DMN dorsomedial** (rTPJ + dmPFC + precuneus + pSTS; Kandel Ch 56): the discourse/ToM
  apex of Hasson's temporal-receptive-window hierarchy, fed by hippocampal binding + the ATL hub.
- **VSA analogy** (Eliasmith SPA solves Raven's; Komer-Stewart A:B::C:?): unbind→transform→apply — the composer
  already exposes these ops.

## The tiered roadmap

### Tier 0 — quick unlocks (days; reuse-by-import; directly fixes the owner's chat experience)
- **0.1 Verb-frame argument structure** [fronts 1+4 #1]. Extend the role alphabet beyond (agent,action,patient);
  a per-verb-class frame lexicon (go→GOAL-PP, give→THEME+RECIPIENT); keep the preposition in extraction; expand
  the bare triple into ordered (content + closed-class) slots fed to the validated FrameCQ serial-order engine.
  → "the boy goes TO THE park." Anti-cheat: ablate the function-word pool → telegraphic agrammatic output
  (reproduces Broca's aphasia — a signature an artifact can't fake).
- **0.2 Fixed-capacity WM** [front 4 #2]. Replace the per-concept attractor bank with the in-codebase
  `OrderedPositionWM` (fixed-slot, vocabulary-independent pointer buffer). Kills the freeze by construction;
  houses 0.1's frame slots.
- **0.3 Wh-questions as a filler-gap dependency** [front 1 #3]. "where/what/who does X V?" → hold the wh-filler
  in the (now-fixed) WM; the verb-frame (0.1) says which role is gapped; query it. Reuses the embedded-clause
  WM-hold + the multi-frame parser. → natural "where does the boy go."
- **0.4 Clarification-on-failure** [front 2 #2]. Route the EXISTING abstention/familiarity-gate signal to an
  informative "I don't know X" / "which X?" instead of silent abstention. Turns the moat into graceful
  degradation. (Full disambiguation needs Tier 1, but the TRIGGER is free now.)

### Tier 1 — the keystone (the representational unlock)
- **1.1 Entity-instance / discourse-referent layer** [fronts 1+2 #1]. Per-instance hippocampal engrams (the D.14
  barcode): an indefinite ("a boy") ALLOCATES a token; a definite/pronoun ("the boy"/"it") PATTERN-COMPLETES to
  the held token (DG/CA3, validated). A Discourse-Representation-Theory file-card maps surface refs → tokens.
  Turns "knows the type boy" into "tracks THIS boy" → "which boy?" answerable; grounds reference, common ground,
  person concepts, and any ToM-about-a-specific-person. Capacity ~7 (Lisman-Idiart — a biology-faithful limit,
  not a defect). Anti-cheat: two same-type instances must stay separable (pattern-separation) + a pronoun
  resolves to the right one (the biased-competition WTA, already de-risked GO).

### Tier 2 — reasoning, thinking, discourse (built ON the new representation)
- **2.1 Spiking analogy A:B::C:?** [front 3 #1] on the existing FHRR composer (unbind→transform→apply). Converts
  retrieval→reasoning; redeems nothing it didn't earn (held-out + permuted anti-cheat).
- **2.2 Associative chain-of-thought** [front 3 #2] — self-cued attractor hops: the agent picks the NEXT relation
  by learned association strength, not a caller-supplied plan. The structural heart of "thinking" + the
  communicable-brain north-star. Anti-cheat: lesion the association → the chain collapses.
- **2.3 Transitive inference via a learned ordinal map** [front 3 #3] — the REDEMPTION of the retracted
  2026-05-14 chaining; the biologically-correct cognitive-map geometry (Eichenbaum/Park 2020); gated by the
  symbolic-distance-effect signature (an artifact can't fake it).
- **2.4 Minimal common-ground** [front 2 #3] — a shared-vs-private fact tag → audience design (the cheapest ToM slice).
- **2.5 Tense/aspect + events; PBWM gating** [front 4 #3] — the next representation + control layers (BG
  disinhibition/`transmission_gate`/DA-RPE re-targeted as a WM input gate, O'Reilly-Frank; an ACC conflict-margin
  read-out over the WTA cleanup for abstain/repair).

### Tier 3 — deep walls (deferred; research-gated; named honestly)
- **Full theory-of-mind / false-belief + Gricean implicature/RSA** — recursive agent-modeling; genuine substrate
  walls (only minimal slices tractable on point neurons).
- **Productive recursive / LEARNED grammar + schemas** — the hand-authored frame lexicons in Tier 0/1 are
  SCAFFOLDS (variety, not learned grammar); the learned version is the ~134K-param BPTT-SNN scale wall + the
  dendritic-substrate frontier.
- **Turn-taking** — clean reuse of the commit-burst accumulator, but only relevant for a streaming/spoken
  interface, not the typed console.

### Parallel / ongoing (the DEPTH axis, orthogonal to the representation work above)
- **Code/association quality** (the noisy PPMI neighbours, e.g. "world"→calvados — a rare-word PPMI bias): a
  frequency-floor on the neighbour-picker (quick) + a bigger/richer corpus (the breadth lever).
- **More facts** (raise the console's fact count, validated to 300 @ 96%); **Path A** (pure brain-generative LM)
  as the long-term LLM-free generation.

## Honest framing

Tier 0 is mostly ASSEMBLY of validated parts — and the owner's exact chat complaints ("goes park", "where does
the boy go", the "which boy" trigger) all fall here. Tier 1 (entity instances) is the keystone unlock, and the
core machinery (the D.14 barcode + DG/CA3) is already shipped. Tier 2 is real but reuse-heavy. Tier 3 are the
genuine walls (recursion, full ToM, learned grammar) — named, deferred, research-gated. **NO dendritic rewrite is
required for Tiers 0-2.** The no-confab moat is preserved throughout: clarification REPLACES silent abstention, it
never fabricates; schema/instance prediction stays VERIFY-gated.

The one-line version: **our brain parses and words well, but it thinks in skeletons. The roadmap is to give it a
richer skeleton — typed verb frames + entity instances — almost entirely by wiring parts we've already built and
validated, before reaching for the genuine walls.**
