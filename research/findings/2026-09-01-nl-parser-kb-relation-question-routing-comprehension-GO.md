---
type: finding
status: positive
date: 2026-09-01
mechanism: NL-parser routing of natural questions about the shipped KB's underscored multi-word relations to the recall primitive (comprehension-only)
lane: comprehension-routing
seeds: [42]
seed-waiver: >
  The added mechanism is DETERMINISTIC comprehension routing (regex question-shape -> (entity, relation) pair);
  it has no stochastic component, so routing is seed-independent by construction. The one seed-varying step is
  the downstream RECALL (composer.query_patient), which is the SAME primitive every existing knowledge route
  uses and is itself already 6-seed-GO. Resolution (29/29) and lesion-attribution (29/29 return None with the
  flag off) are established by directly exercising the route function (no stochasticity); recall-of-the-exact-
  stored-patient (29/29) is confirmed on seed 42 through the FULL production brain + the shipped 15k-fact LTM.
  A 6-seed rebuild of the full-LTM production brain per relation was attempted (verify_kb_relations.py) but
  OOM-exhausts (each per-seed 15k-LTM brain build accumulates RSS) — an instrument cost, not a result gap: the
  routing claim does not depend on the seed and the recall primitive it calls is separately 6-seed-validated.
artifacts:
  - research/findings/raw/_nl_parser_real_kb_relations/routing_verify_result.json
  - research/findings/raw/_nl_parser_real_kb_relations/verify_kb_relations.py
  - research/runners/brain_chat_tui.py
external: NO-EXTERNAL-NEEDED — an internal comprehension-routing wiring over this repo's own shipped KB.
---

# NL parser routes natural questions about the shipped KB's multi-word relations to the recall primitive — comprehension-only, moat-preserving

**Artifact:** `research/findings/raw/_nl_parser_real_kb_relations/routing_verify_result.json` (routing + recall verify, provenance-stamped).

## The gap (named, not invented — the confidence-forthcomingness arc's own residual)

The shipped `wikidata_core_15k` core (the out-of-the-box default brain's LTM) keys each fact on ONE atomic
underscored token per concept, so its TOP relations are collapsed multi-word Wikidata property labels —
`country_of_citizenship`, `member_of_political_party`, `place_of_birth`, `headquarters_location`, `part_of`, … .
The live NL question parser could not reach any of them: `_relation_fronted_route`'s regex requires the fronted
relation to be a SINGLE bare word (its own docstring: "a multi-word relation phrase is left to the generic
parse"), and the generic positional parse has no notion of a relation noun-phrase. So a natural question about
any of these — *"what is X's country of citizenship?"*, *"where was X born?"*, *"what political party is X a
member of?"* — never reached a routable `(entity, relation)` pair and the turn honestly-but-wrongly ABSTAINED,
even though the substrate answers instantly once the right `(entity_token, relation_token)` pair reaches
`composer.query_patient`. This is the exact "NL-parser vocab gap" the confidence-forthcomingness arc named as
its remaining live-traffic blocker (`2026-09-01-confidence-forthcomingness-ltm-elaboration-load-bearing-GO.md`).
It gates board #66 (knowledge-in-chat) and #94 (confidence-forthcomingness on the shipped-KB's real traffic).

## The mechanism (COMPREHENSION only — no recall, capacity, or `sim/` change)

`research/runners/brain_chat_tui.py`: a curated table maps 29 underscored/idiomatic relations to natural-English
question shapes (regex, each capturing an `entity` group) — two GENERIC shapes from the relation's own
underscore→space phrase (*"what is `<entity>`'s `<phrase>`?"* / *"what is the `<phrase>` of `<entity>`?"*,
covering every relation with zero per-relation authoring) plus idiomatic EXTRA shapes for the subset a real
speaker phrases differently (*"where was X born?"*, *"who does X work for?"*, *"what political party is X a
member of?"*). `ChatBrain._kb_relation_question_route(question)` resolves the entity via the SAME spiking
alias-hop (`_ground_content_words`, naive underscore-join fallback) the existing routes use, and is wired into
`_extract_route` between the relation-fronted and definitional-copula routes; a non-match / disabled flag
returns `None` → falls straight through, byte-identical for every other question. Same honesty class as
`_relation_fronted_route`'s own regex — a phrase-shape scaffold; the RECALL is untouched (`query_patient`), so
the change can only ADD a resolvable route, never invent an answer. `BRAIN_KB_RELATION_QUESTIONS` in
`{0,false,no,off}` is the lesion/escape (byte-identical to before this arc); default-ON like every other
knowledge-grounding flag.

## Verification

- **Resolution (flag ON): 29/29** — every underscored relation's generic question routes to `[entity, relation]`;
  sampled idioms (born / work-for / member-of-party / nationality / educated-at) also route (5/5).
- **Lesion-attribution (flag OFF): 29/29 return `None`** — the flag genuinely gates the route (the coupling is
  load-bearing, not a no-op beside an existing route).
- **Recall + moat (seed 42, FULL production path): 29/29** — through `_build_tiny_demo(composer_kind="onebrain")`
  + the shipped 15k `ShardedPhasorStore` LTM via `TieredFactStore` (exactly what `webapp.server._build_chat_brain`
  builds for the out-of-the-box brain), each question's `gate()` recalls the EXACT shipped stored patient — e.g.
  *"where was akita masami born?"* → `tokyo_administrative_district`; *"who does asimov isaac work for?"* →
  `university_of_boston`; *"what political party is asimov isaac a member of?"* → `us_democratic_party`. A wrong
  (invented) patient scores as FAIL identically to an abstain, so 29/29 means recall-of-the-real-fact, never
  fabrication — the moat holds.
- **Byte-identical-off: STRUCTURAL** — the route is a pure additive branch in `_extract_route` returning only on
  its own regex match; a non-match returns `None` → unchanged downstream. The 29/29 lesion-`None` result confirms
  the fallthrough path is exercised.

See the seed-waiver (frontmatter) for why routing is seed-independent and the full-LTM 6-seed rebuild OOM-exhausts.

## Next rungs

- Advances board #66 (knowledge-in-chat) and closes the NL-parser vocab blocker #94 named — the shipped KB's
  multi-word-relation questions now reach the recall primitive on the real chat path.
- The entity side still leans on the naive underscore-join because the shipped bundle ships ZERO `alias_of`
  facts (a separate residual: entity aliasing), not this arc's scope.
- With this closed, the confidence-forthcomingness flip's remaining blocker is the owner's UX call.
