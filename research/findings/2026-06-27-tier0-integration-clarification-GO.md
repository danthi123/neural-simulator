# Tier 0 integration + 0.4 clarification — GO (2026-06-27)

**Tier 0.1 (verb-frame argument structure, `ArgStructureComposer`) and 0.3 (wh-questions, `wh_question_parser`)
were BUILT + verified (commits `1dc4e1ec`, `d0bbfef7`) but DARK on the deployed first-chat console** — the console
built the plain `RFPhasorComposer`, so typed roles never bound and "where does the boy go?" abstained. This arc
WIRES them in (lights up Tier 0 end-to-end on the real brain) and adds **0.4 clarification-on-failure**. Plus a
linguistic correctness fix in the typed-roles extractor (ditransitive direct objects → THEME).

**Verdict: GO.** End-to-end on the real `brain3000pos_w7000` brain (3000 concepts, D=128, single-bridge
`ArgStructureComposer`), CPU/numpy: "where does the boy go?" → **"the boy goes to the park"** (the typed GOAL now
lights up, NOT abstain); "what does the mom give?" → **"the mom gives the hug"** (THEME); moat 0 FA throughout;
every rendered answer re-parses to its stored typed fact; clarification fires informatively on an abstain. The
DEFAULT path (no `--argstructure`) rubric is **10/10, moat 0** (regression GREEN). **NO `sim/` edit.**

## What was built (reuse-by-import; additive; default unchanged)

### 1. Typed facts (`research/findings/raw/_facts3000_typed.json`, 917 facts)
`_corpus_svo_extract --typed-roles` over TinyStories (175,347 sentences), restricted to the `brain3000pos` 3000
vocab. Role distribution: **GOAL 226, patient 654, THEME 33, LOCATION 4**. Top fact: `(boy, go, {GOAL: park})`
×169. Every fact corpus-attested (≥2 occurrences, logged source sentence) — host CURRICULUM preprocessing, the
brain still stores/recalls via binding (BRAIN-BASED-ONLY compliant: this is "preparing the syllabus").

**Linguistic fix in the extractor** (`_corpus_svo_extract.py`, `--typed-roles` path only; default output
byte-unchanged): a DIRECT object (`prep=None`) now fills the verb-FRAME's first internal-argument content role
(Bock & Levelt: the verb lemma projects its argument frame), not always the bare `patient`. So a ditransitive
verb's direct object ("mom gave a hug") → THEME (give's frame has no `patient` slot), while a transitive verb's
direct object stays `patient`. Without this, the 33 `give`/`put` direct-object facts were stored as `patient`,
which the give/put frame cannot render — so "what does X give?" would have had nothing to answer.

### 2. Console integration (`first_chat_console.py`)
- `--argstructure` flag (default off): builds an `ArgStructureComposer(seed, D, vocab, grounded_codes)` instead of
  the plain `RFPhasorComposer`, on the SAME learned codes. Single-bridge (`--shards 1`; multi-bridge ArgStructure
  is a follow-on — a `note` prints if `--shards>1`). Requires `--facts-json` with typed facts (clean `SystemExit`
  otherwise).
- `_load_typed_facts(...)`: reads the typed-role facts, dedup to one object per (agent,action)/(action,filler) (the
  unambiguous-cue moat rule), stores each via `ArgStructureComposer.store_fact`. **Double-binding for the flat-SVO
  pipeline**: a typed object filler is bound to BOTH its typed role (GOAL/THEME/…) AND `patient`, so
  `query_role("GOAL", boy, go)` AND `query_patient(boy, go)` both return the filler. The verb-frame `render` emits
  ONLY the typed unit (the redundant `patient` is invisible to the frame), so "the boy goes to the park" is clean
  while the DiscursiveTurn / proposer / `audit_moat` (all SVO-shaped) see a genuinely stored, recallable
  `(agent, action, filler)`. 24/24 stored facts recall correctly via `query_patient`.
- The wh-route (`respond` → `_wh_response` → `answer_wh(comp, …)`) was ALREADY composer-agnostic; on the
  `ArgStructureComposer` it now takes the FULL typed-role filler-gap path (`query_role`). For "what does X V" where
  the verb's frame realizes a TYPED object (give→THEME), the route now goes through the wh path so the answer
  renders via the verb FRAME ("the girl gives the ball"); for a plain `patient` verb it stays on the existing
  `_WHAT_DOES_RE` discuss path (the rich discuss-while-answering is preserved).
- A **copula guard**: "what is X?" / "who is X?" (subject-form, verb ∈ {is,are,was,were,be,am}) no longer get
  consumed by the wh-route — they fall through to the `_ABOUT_RE` route (which gives the right unknown-word
  clarification on the REAL word X). This fixes a latent pre-existing bug where "what is X" bypassed the nicer
  `_ABOUT_RE` clarification and hit the wh bare-fallback.

### 3. 0.4 Clarification-on-failure (`first_chat_console.py`)
Routes the EXISTING abstain/familiarity signal to an INFORMATIVE reply instead of a bare canned line — never
fabricates (clarification REPLACES silent abstention):
- **unknown word** → `_clarify_unknown(x)`: "I don't know the word \"x\" yet — it's not in what I've learned."
  Fired from the `_ABOUT_RE` route, the wh-route abstain (when the named referent is out-of-vocab), and the
  bare-topic fallback (an unknown content-shaped word).
- **known-but-factless topic** → the existing grounded PPMI hedge (already built in `_render` / `_wh_response`):
  names the topic's real PPMI neighbours, framed as association-not-fact.
- **referential / under-specified** ("which boy?", "which one …?") → `_clarify_underspecified`: a generic, honest
  "I'm not sure which boy you mean — I track the idea of \"boy\" but not specific ones yet. Can you say more?" The
  FULL which-X disambiguation needs Tier-1 entity instances; this is the honest generic TRIGGER (free now). A
  `_WHICH_RE` detector checked before the wh/about routes.

## The end-to-end chat (the payoff)

Real brain (`brain3000pos_w7000.npz_seed42.npz`, ArgStructureComposer, 24 typed facts), CPU/numpy:

```
[where]  YOU: where does the boy go?
         BRAIN: the boy goes to the park
         [intent=wh_question role=GOAL filler=park certain=1 moat=OK]
         VERIFY: re-parse('the boy goes to the park') -> stored typed fact? True; query_role(GOAL,boy,go)==park? True

[theme]  YOU: what does the mom give?
         BRAIN: the mom gives the hug
         [intent=wh_question role=THEME filler=hug certain=1 moat=OK]
         VERIFY: re-parse('the mom gives the hug') -> stored typed fact? True; query_role(THEME,mom,give)==hug? True

[what_does] YOU: what does boy go?        (the old probe still works)
            BRAIN: Good question. The boy goes park.   [intent=question certain=1 moat=OK]

[phatic] YOU: hi there!
         BRAIN: Hi! Ask me about what I know, or what I think about something.   [intent=phatic moat=OK]

[clarify_unknown] YOU: what is florbglax?
                  BRAIN: I don't know the word "florbglax" yet -- it's not in what I've learned.   [intent=unknown_word moat=OK]

[clarify_which]  YOU: which boy?
                 BRAIN: I'm not sure which boy you mean -- I track the idea of "boy" but not specific ones yet. Can you say more?   [intent=underspecified moat=OK]

[moat_where] YOU: where does the dog give?    (frame-licensed but unstored -> abstain)
             BRAIN: I don't have a stored fact answering that, but dog tends to come up alongside breed, alzheimer, and dogs -- I'd be guessing past that.   [certain=0 moat=OK]

  MOAT leaks across probes: 0   (CLEAN)
  render-reparse failures:  0  (CLEAN)
```

`--moat-test` on the argstructure console: **0 leaks** (world/music abstain honestly; an untaught cue engages via
adjacency without fabricating).

## Anti-cheats

- **Moat 0 FA** across every probe + the `--moat-test`. The integration never weakens it; clarification replaces
  silent abstention (never fabricates).
- **VERIFY (moat-on-the-render):** every rendered GOAL/THEME answer re-parses (`argstructure_composer.reparse_to_fact`)
  to its stored typed fact — content-mismatch would reject. 0 failures.
- **Regression — DEFAULT (no `--argstructure`) rubric = 10/10, moat 0, MIXED, VERDICT PASS.** Note: 0.4 changed two
  default-rubric abstention STRINGS (the bare "I don't have a grounded answer … rather not guess" → the more
  informative neighbour-naming hedge / "I don't know X yet" clarification) — this is the INTENDED 0.4 effect (the
  moat made graceful), a strict improvement, and both still pass + moat 0. The default COMPOSER-construction path is
  byte-unchanged (still `RFPhasorComposer`); the typed-facts/argstructure path is fully gated behind `--argstructure`.

## Honest scope / notes

- The typed-role assignment is the host-side verb-frame SCAFFOLD (a single oblique role per verb-frame): "go for a
  walk" / "go with mom" map to GOAL because `go`'s frame has one oblique slot (GOAL). The headline forms
  (go→GOAL:park, give→THEME) are correct; finer oblique disambiguation (COMITATIVE "with", PURPOSE "for") is a
  bounded follow-on (more frame entries + prep→role rows). This is variety, not learned grammar (Tier 3).
- `--argstructure` is single-bridge. A `RoutedComposer` of `ArgStructureComposer`s (typed roles + sharded
  deep-knowledge scaling) is a follow-on.
- The opinion/discuss path on a stored agent does not always lead with a certain fact (recall is lossy at D=128) —
  pre-existing DiscursiveTurn behavior, moat-safe (hedged), not introduced here.

## Files

- `research/runners/_corpus_svo_extract.py` — ditransitive direct-object → frame's first internal-arg role (typed-roles path only)
- `research/runners/first_chat_console.py` — `--argstructure`, `_load_typed_facts`, the typed-fact store + flat
  projection, the wh-route typed-`what` routing + copula guard, 0.4 clarification (`_clarify_unknown`,
  `_clarify_underspecified`, `_WHICH_RE`, the unknown-content-word fallback)
- `research/findings/raw/_facts3000_typed.json` — 917 typed-role corpus facts (brain3000pos vocab)

NO `sim/` edit. CPU/numpy. The composer/parser/console are research runners (reuse-by-import).
