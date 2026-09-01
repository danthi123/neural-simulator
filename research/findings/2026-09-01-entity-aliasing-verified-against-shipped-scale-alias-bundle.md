---
type: finding
status: positive
date: 2026-09-01
mechanism: entity ALIAS-form question resolution+recall against the already-built alias-extended wikidata_core_15k_grounded_v1 bundle (30,804 alias_of facts), through the live _kb_relation_question_route / _relation_fronted_route / _definitional_copula_route alias-hop wiring
lane: comprehension-routing
seeds: [42]
seed-waiver: >
  ROUTING (Part A) is DETERMINISTIC comprehension (regex question-shape + a genuine but stochasticity-free
  query_patient alias-hop over a fixed, loaded-once store) -- seed-independent by construction, verified via the
  unbound route methods against a mock `self` (no brain build; per GAP_CLOSURE_MISSION.md's 2026-09-01 OPS LESSON,
  commit e4680f4e3, on avoiding a per-case full-brain rebuild). Part B's full-production `gate()` confirmation
  needs a genuinely-spiking substrate build (composer_kind="onebrain", the LARGER 37,837-word alias-extended
  vocab); ONE seed (42), ONE build, is used -- matching this session's own prior precedent
  (2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md's own seed-waiver: the recall primitive
  `query_patient` it calls is separately already 6-seed-GO, 2026-08-25-fhrr-decode-rate-at-scale.md). A SECOND
  (LESIONED) full-gate() build was budgeted but not separately completed under this session's CPU contention
  (repeated ~500s+ builds); the identical mechanism-level lesion (grounding ON vs OFF changing `entity_final`)
  IS empirically confirmed for all 29 relations in Part A, using the SAME `_kb_relation_question_route` `gate()`
  itself calls -- see "What was not separately re-run" below for the honest accounting.
artifacts:
  - research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.json
  - research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.py
  - research/findings/raw/_kb_entity_aliasing_verify/part_a_transcript.txt
  - research/findings/raw/_kb_entity_aliasing_verify/part_b_intact_transcript.txt
  - research/findings/raw/_kb_entity_aliasing_verify/single_word_collision_probe.json
  - research/runners/brain_chat_tui.py
  - research/runners/_knowledge_core_curate.py
external: NO-EXTERNAL-NEEDED -- an internal verification of this repo's own already-built alias mechanism against its own already-built alias-extended knowledge bundle.
---

# Entity aliasing verified: alias-form questions now resolve + recall through the live chat pipeline (board #94/#143 residual)

**Artifact:** `research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.json`.

## The residual (named, not invented)

`2026-09-01-nl-parser-kb-relation-question-routing-comprehension-GO.md`'s own "Next rungs" named it explicitly:
"The entity side still leans on the naive underscore-join because the shipped bundle ships ZERO `alias_of`
facts (a separate residual: entity aliasing)." A question naming an entity by a common alias ("chelsea", "isac
asimof") rather than the exact canonical token ("chelsea_fc", "asimov_isaac") could miss even though the KB
holds the fact — because the DEFAULT shipped `wikidata_core_15k` bundle (what `_default_ltm_bundle_dir()`
resolves to) carries zero `alias_of` facts for the alias-hop to read.

## VERIFY-FIRST: the mechanism already existed in full (not re-derived)

Both halves of "the build" this task named were **already built**, before this arc started:

1. **Build-time alias-table generation** — `research/runners/_knowledge_core_curate.py`'s `build_alias_facts`
   (+ `_all_other_aliases` + `_alias_quality`) already emits `{agent: alias_token, action: "alias_of", patient:
   canonical_token}` facts straight from the KB's own raw Wikidata alias/redirect lists (host code building the
   alias table from KB data — exactly this arc's brief), with an honest ambiguity policy (an alias claimed by
   ≥2 distinct concepts, or colliding with an existing canonical word, is DROPPED, never guessed).
2. **Query-time spiking alias-hop** — `research/runners/brain_chat_tui.py`'s `_alias_hop`/`_ground_content_words`
   already resolve a surface form to its canonical token via ONE MORE genuine `composer.query_patient(candidate,
   "alias_of")` hop (the same brain-based primitive `compositional_chain_route.py`'s reasoning hops already
   count as), wired **eagerly** (`min_span=1`) into all three entity-capturing routes
   (`_kb_relation_question_route`, `_relation_fronted_route`, `_definitional_copula_route`), not merely a
   last-resort fallback.
3. **A full production-scale alias-extended bundle was already built** (2026-08-26,
   `2026-08-26-knowledge-grounding-natural-language.md`): `~/Projects/sim-data/knowledge_bundles/
   wikidata_core_15k_grounded_v1`, the SAME curation parameters as the shipped `wikidata_core_15k`
   (8000 entities/40 relations/15000 facts/seed 42) **plus 30,804 `alias_of` facts** (44 ambiguous aliases
   correctly dropped), verified 60/60-correct after a save/reload round-trip.

**What was genuinely missing** (this finding's job): that bundle's alias facts had never been exercised with an
actual BATTERY of alias-FORM natural-language questions through `_kb_relation_question_route` — a route that did
not exist yet when the 2026-08-26 finding ran its own (2-example, smoke-scale) alias-grounding check. This
finding closes that verification gap. **No new alias-generation mechanism, no `sim/` change, no production-
default flip** — `webapp/server.py`'s `_default_ltm_bundle_dir()` is untouched; the alias-extended bundle is
loaded here via an explicit path override, exactly as the 2026-08-26 finding's own verify did. Swapping the
shipped default remains the owner's call (named open in board task #143: "OPEN DECISION for the owner: swap
as-is (+~1s/turn), or first reduce the alias count / shard the alias vocabulary").

## Method

Per `GAP_CLOSURE_MISSION.md`'s 2026-09-01 OPS LESSON (commit `e4680f4e3`): verify deterministic ROUTING via the
**unbound route method + a mock `self`** (pure parsing, no brain build) against a **loaded-once** real
`ShardedPhasorStore` (genuine `query_patient`, not a host dict); confirm end-to-end RECALL separately through a
real production `ChatBrain` build — not per test case (the prior OOM lesson).

- **Part A — mock-self routing** (`research/findings/raw/_kb_entity_aliasing_verify/verify_entity_aliasing.py`,
  `part_a_mock_routing`): `ChatBrain._kb_relation_question_route`/`_relation_fronted_route`/
  `_definitional_copula_route` called UNBOUND against a trivial mock `self` whose `.inner.composer` is the real,
  loaded-once `ShardedPhasorStore` over the alias-extended bundle. All 29 `_KB_UNDERSCORED_RELATIONS`, sampled
  with a REAL fact whose agent has ≥1 real alias (drawn from the bundle's own `facts.json`, not hand-typed).
- **Part B — full production `gate()`** (`part_b_full_production`): `_build_tiny_demo(42, composer_kind=
  "onebrain")` + the alias-extended LTM attached via `TieredFactStore` (mirrors `webapp.server._build_chat_brain`
  / `_nl_parser_real_kb_relations/verify_kb_relations.py` exactly) — ONE build, 8 relations + a canonical-form
  regression + a fabricated-alias moat probe, through the real `gate()` call (`/api/brain-chat`'s own code path).
  A second (LESIONED) full-gate() build was budgeted but not separately completed this session (CPU-contention
  wall-clock cost of a second ~500s+ build) — see "What was not separately re-run" below.

## Results

- **Part A resolution: 29/29 KB-relation alias-form questions** resolve to `[canonical_entity, relation]`
  (e.g. `"where was isac asimof educated?"` → `['asimov_isaac', 'educated_at']`; `"what is bill blythe iv's
  position held?"` → `['bill_clinten', 'position_held']` — real Wikidata aliases, including several that read
  as genuine nicknames/typos in the source data, not synthetic examples).
- **Load-bearing: 29/29** — with `BRAIN_KNOWLEDGE_GROUNDING=0`, the SAME question's `entity_final` reverts to
  the naive underscore-join of the ALIAS phrase (≠ the canonical token in every case) — the alias-hop, not the
  naive join, is what makes these resolve.
- **`_relation_fronted_route` (pre-existing 2026-08-27 route): 2/2** alias-form questions resolve
  (`"what country is jackson mississippi from?"` → `['jackson_ms', 'country']`).
- **`_definitional_copula_route` (pre-existing 2026-08-26 route): 5/5** alias-form questions resolve.
- **Part A moat: 3/3** adversarial phrases correctly fail to alias-hop (abstain): `"chelsea"` (a genuinely
  ambiguous common alias in THIS core — chelsea_fc / chelsea_kensington / chelsea_middlesex all present),
  `"purple elephant bicycle"`, `"zorblexia quintar"` (invented nonsense).
- **Part B (full `gate()`, seed 42): 8/8** sampled alias-form KB-relation questions recall the EXACT stored
  patient (never a wrong/invented one) through the real production pipeline over the alias-extended bundle —
  e.g. `"where is ens paris headquartered?"` → `['ens_ulm', 'headquarters_location', 'sport_in_paris']`;
  `"what borders jamhuri ya kenya?"` → `['etymology_of_kenya', 'shares_border_with', 'people_of_uganda']`.
- **Canonical-form regression:** `"what is the language of work or name of margaret movie?"` (the LITERAL
  canonical form, no alias needed) → identical answer — the alias-hop is purely additive.
- **Moat, full pipeline:** `"what is the glorble house of nonexistence's country of origin?"` (a fabricated
  5-word nonsense entity) → `gate()` → `None` — an honest abstain through the complete production path, never a
  fabricated fact.

## What was not separately re-run (an honest accounting, not a silent gap)

Part B's design (see the cited `.py`) includes a SECOND full-brain build with `BRAIN_KNOWLEDGE_GROUNDING=0` to
re-confirm load-bearing at the `gate()` level. Under this session's CPU contention, each onebrain+alias-vocab
build over the LARGER (37,837-word) alias-extended bundle took 200–900s (three attempts were needed to land Part
A cleanly and Part B's INTACT arm — see the `.json` artifact's `provenance_note`); a second build for the
LESIONED re-confirmation was not completed. This is **not a gap in the underlying claim**: `gate()`'s recall
step (`_substrate_recall` → `_extract_route` → `_kb_relation_question_route`) is the EXACT SAME function Part A
already lesions for all 29 relations (`BRAIN_KNOWLEDGE_GROUNDING=0` → `entity_final` reverts to the naive join,
≠ canonical in 29/29 cases) — a mock `self` and a real `ChatBrain` call the identical code path, so the
load-bearing property IS empirically established for the mechanism `gate()` depends on; only the redundant
full-gate()-level re-observation of that same fact was traded away for wall-clock cost. Named here rather than
silently dropped from the original design.

*(Full totals + every question/route/gate pair are in the cited JSON artifact and its two transcript
companions.)*

## An honest residual surfaced by this verification (not fixed here)

Exercising the alias-hop against REAL, full-scale alias data for the first time (the shipped default bundle has
zero alias facts, so this code path was previously dead in production) surfaced a genuine, **pre-existing**
collision risk distinct from the multi-word moat check above: a **single ordinary English word** that happens to
also be a genuine (non-ambiguous within this KB) raw Wikidata alias collides outright. `"what is house?"` →
`_definitional_copula_route` resolves to `['domestic_architecture', 'isa']` (not an abstain, not a literal
"house" lookup) — `"house"` is a real, unambiguous Wikidata alias for a `domestic_architecture` concept in this
core. The same collision reaches `_relation_fronted_route` and `_kb_relation_question_route` (all three sites
use `min_span=1`). This is **structurally different** from the multi-word nonsense case verified safe above: a
partial (>1-token) grounding is rejected by the `len(grounded) == 1` guard at every call site and falls back to
the naive join (which then finds no matching canonical entity → abstain); a **single**-word span has no
"partial" state for that guard to catch — if that one word IS a real alias, it resolves. Full repro:
`research/findings/raw/_kb_entity_aliasing_verify/single_word_collision_probe.json`. **Scope note:** this is a
property of the ALREADY-SHIPPED 2026-08-26/2026-08-27 alias-hop wiring, not introduced by this arc — this arc's
verification is simply the first time it has run against real production-scale alias data. Not fixed here (a fix
needs a collision policy for the three `min_span=1` sites — e.g. a common-word stoplist, or reusing
`_extract_route`'s own `known_words` in-session guard — a real design decision, out of this arc's comprehension/
recall verification scope). Named as the next rung below.

## Honesty: scaffold vs. substrate (unchanged from 2026-08-26; not re-derived)

Identical honesty classification as the 2026-08-26 finding this one verifies (see that finding for the full
argument): build-time raw-alias extraction + query-time phrase segmentation are host scaffolding (the same class
already accepted for `_definitional_copula_route`'s regex and `compositional_chain_route.py`'s role-noun table);
the alias RESOLUTION itself is a genuine stored fact recalled via the same spiking `query_patient` op the rest
of the system already counts as brain-based — an unresolved alias abstains exactly like an unknown agent.

## Next rungs

- **Owner UX call (unchanged, restated):** swap `BRAIN_LTM_BUNDLE`'s default to the alias-extended bundle (cost:
  ~1.13s/recall vs ~0.2-0.4s on the un-aliased core, per the 2026-08-26 finding's own latency measurement — not
  re-measured here), or re-curate the shipped bundle in place with the vocab fix, or reduce/shard the alias
  vocabulary first. This finding does not resolve that decision, only removes "we don't know if the alias
  mechanism actually works at scale" as a reason to defer it.
- **The single-common-word collision** named above: a collision policy for the three `min_span=1` entity-capture
  sites (stoplist, or an in-session `known_words` guard mirroring `_extract_route`'s own).
- **Rung 3 (unchanged, restated from 2026-08-26):** a fully LEARNED entity-linking/synonymy mechanism acquiring
  alias↔canonical associations from co-occurrence in running text, rather than a teacher-curated Wikidata alias
  file — what would let the brain ground a genuinely NOVEL entity's surface forms it was never given an alias
  list for.
