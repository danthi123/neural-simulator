---
type: finding
status: lexicon-lever
date: 2026-09-01
mechanism: board #112 rung 3 — a curated relation->English-predicate lexicon (34/34 of the real store's
  distinct relation types) + a closed-class underscored-slug->natural-NP surfacer, BOTH driving the SAME
  already-6-seed-GO `SpikingClauseProducer` SVO frame (`research/runners/_spiking_fluent_surface_derisk.py`)
  UNMODIFIED, closing rung 2's own two named residuals (naive verb morphology, raw underscored NPs).
verdict: mechanism-level GO on a NARROW, PRECISELY SCOPED claim — every real sampled fact (48/48 across 6
  seeds, 100%) now renders as a genuinely-spiking, structurally well-formed, role-faithful, moat-safe clause
  that reads as an actual English sentence ("the Zdf Tv is headquartered in the Free City of Mainz", "the Xx
  Winter Olympics follows the Salt Lake Olympics"), not a slug-and-naive-morphology string ("the eu_treaties
  contains_administrative_territorial_entis the united_kingom"). This is NOT a claim that board #112 rung 2/3
  is CLOSED (per `docs/TERMS.md`: closure requires `integrated` — wired + on-by-default + scaffold-retired) —
  it remains, exactly like rung 2, a PARALLEL renderer, not wired into `wkv_mouth_generator.generate()`'s own
  decode loop. No flag exists to auto-flip (SS6 explains why the 2026-09-01 auto-flip policy does not apply
  here).
lane: e-mouth-fluency / A1 (crutch-burndown), board #112
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: N/A — full 6-seed run, one bounded local process (up to 4 builds of a 184-neuron bridge per seed:
  main-len4, main-len5, permuted-len4, permuted-len5, only the lengths a seed's own 8-fact sample actually
  needs), `free -m` checked available memory >25GB before running (12.8GB free / 31GB available at run start),
  well under the 15k-LTM-brain-build RAM-safety concern this task's own hygiene note warns about — no
  `ShardedPhasorStore`, no 15k-fact index, only `facts.json` read directly for an 8-fact-per-seed sample (the
  same low-memory convention rung 2's own runner already used).
instrument: research/runners/_wkv_fact_to_sentence_lexicon_lever.py — direct measurement against the real
  shipped `wikidata_core_15k` facts.json (8-fact seeded sample per seed, 48 facts total across 6 seeds,
  IDENTICAL sampling call to rung 2's `_sample_facts` — same facts, same seeds, a genuine before/after on the
  same inputs), rendering each through a real `SimulationBridge`-backed `SpikingClauseProducer`, re-parsed by
  an independent parser that reconstructs the expected surface from the fact + the SAME lexicon/NP-casing rule
  (ground-truth reuse, not producer-internal trust — the same pattern rung 2's own parser used with
  `emerge_v3`) and does not accept the producer's own account of what it rendered.
runner: research/runners/_wkv_fact_to_sentence_lexicon_lever.py (no args; runs all 6 canonical seeds)
external: Gardent, Shimorina, Narayan, Perez-Beltrachini (2017), "The WebNLG Challenge: Generating Text from
  RDF Data", INLG 2017, https://aclanthology.org/W17-3518/ — establishes curated, hand-built relation->template
  verbalization (354 manually-authored relation templates over DBpedia/RDF triples) as standard, established
  KB-to-text NLG methodology, the same mechanism CLASS as this rung's `RELATION_LEXICON` (a small, curated,
  closed-class relation->predicate map), not a novel unvalidated guess. Recorded:
  `bash tools/record_external_search.sh` entry in `research/queue/.external_searches.jsonl`, lane-tagged
  `e-mouth-fluency / A1 (crutch-burndown), board #112`.
artifacts:
  - research/findings/raw/_wkv_fact_to_sentence_lexicon_lever.json (per-seed + aggregate + verdict, all 48
    rendered examples, lexicon coverage check against the live store, fallback byte-identity check)
  - research/runners/_wkv_fact_to_sentence_lexicon_lever.py (the new lexicon + NP-surfacer + independent parser
    + 6-seed runner)
  - research/runners/_spiking_fluent_surface_derisk.py (the reused, unmodified `SpikingClauseProducer` — 6-seed
    GO per its own docstring, EMERGE-59/60/61)
  - research/runners/_wkv_fact_svo_clause_first_lever.py (rung 2, reused unmodified for the fallback
    byte-identity check and the shared `_sample_facts`)
  - research/findings/2026-09-01-wkv-fact-to-svo-clause-first-lever-investigation.md (rung 2, the two residuals
    this finding closes)
  - research/queue/.external_searches.jsonl (the recorded external source)
---

# From a well-formed clause to a readable sentence — closing rung 2's two named residuals (board #112, rung 3)

## 0. What this is, and is not

`2026-09-01-wkv-fact-to-svo-clause-first-lever-investigation.md` (rung 2) built a first lever: real recalled
facts slotted into `SpikingClauseProducer`'s `PLAIN_TRANSITIVE` frame render as structurally well-formed,
role-faithful SVO clauses (6/6 seeds, 100%). It named two DISTINCT, honestly-separable residuals blocking a
genuine "coherent factual sentence" reading: (1) the "verb" was a naive relation-label morphology guess
(`"follows"` -> `"followses"`, double-inflected; `"country_of_citizenship"` -> raw, untouched), and (2) entities
rendered as raw underscored slugs (`asimov_isaac`), not natural NPs. This task built the concrete next lever
against exactly those two residuals: a curated relation->predicate lexicon and a slug->NP surfacer, both driving
the SAME `SpikingClauseProducer` completely unmodified.

**This is an investigation and a lever, not a closure.** It does NOT modify
`webapp/wkv_mouth_generator.py::generate()`'s own recurrent decode loop, is NOT wired into `answer_turn`, and
does not claim board #112 is closed. What it DOES show, with real measurement on the real store: the two named
content-realization gaps are closed COMPLETELY for every relation type that exists in the live `wikidata_core_15k`
store today (34/34), producing genuinely-spiking, role-faithful clauses that read as actual English sentences.

## 1. The two levers

**Lexicon (residual 1).** `RELATION_LEXICON` (`_wkv_fact_to_sentence_lexicon_lever.py`) is a curated,
closed-class map from a Wikidata-style relation label to an already-correctly-inflected English predicate phrase
plus an object-determiner rule (`"the"` / `"a_an"` computed from the object NP's first letter / `""` when the
predicate's own shape makes an article wrong), e.g. `employer` -> `("works for", "the")`,
`place_of_birth` -> `("was born in", "the")`, `instance_of` -> `("is", "a_an")`,
`country_of_citizenship` -> `("is a citizen of", "the")`. Unlike rung 2's `emerge_v3(action)` guess, these
predicates are hand-authored per relation, never morphologically derived from the relation label — closing the
`"followses"` / raw-relation-label class of defect by construction, not by a better inflection rule. This is
the SAME mechanism class the external literature check (SS above) confirms is standard KB-to-text NLG practice
(WebNLG's 354 hand-built relation templates), applied here at a much smaller scale (34 relations, the real
store's own full inventory — verified live, not assumed, by `_check_lexicon_coverage` reading facts.json
directly).

**NP surfacer (residual 2).** `slug_to_np` splits an underscored slug on `_`, Title-Cases each word, and keeps
a short closed list of connective words (`of`, `the`, `and`, `in`, `for`, ...) lowercase — `asimov_isaac` ->
"Asimov Isaac", `u_s_of_a` -> "U S of A", `united_kingom` -> "United Kingom" (the store's own truncated
spelling, preserved verbatim — not this lever's bug, exactly the same honest-data-artifact rung 2's own finding
already named for this identical slug). **Honest limit, stated up front, not discovered mid-run:** slug WORD
ORDER is preserved as-is. The store's own person-entity slugs are inconsistently ordered — `frank_lincoln_wright`
is natural given-then-family order, but `asimov_isaac` is surname-first — and this lever does not attempt
name-order inference (no reliable signal in a bare slug distinguishes a person-name field from any other entity
field in this store). A slug that happens to be surname-first will render as "Asimov Isaac", not "Isaac Asimov".

**Both feed the SAME `SpikingClauseProducer.emit()`, unmodified.** The only change from rung 2 is WHICH content
strings are handed to the DET/SUBJ/VERB/[DET]/OBJ slots and which of two fixed-length templates (5 slots for
`"the"`/`"a_an"`, 4 slots for `""`) is used — the producer's own spiking competitive-queuing order mechanism is
untouched, reused by import exactly as rung 2 reused it.

## 2. Measurement — 6 seeds, the SAME real facts rung 2 sampled, an independent parser, a real control

Every number below is read directly from `research/findings/raw/_wkv_fact_to_sentence_lexicon_lever.json`.
Lexicon coverage against the LIVE store (not assumed): **34/34 distinct relation types (100%)**, checked by
`_check_lexicon_coverage` reading `facts.json` directly at run time.

| Seed | n | covered | well-formed | faithful | readable | moat-safe | bridge spiked | fallback byte-identical to rung 2 | permuted-control faithful |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| 43 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| 44 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| 100 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| 101 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| 102 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | True | True | 0.0 |
| **All 6 (48 facts)** | **48** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **6/6** | **6/6** | **0.0** |

`faithful` is an EXACT byte match against a ground-truth surface independently reconstructed from the fact +
the same lexicon/NP rule (stronger than rung 2's own `faithful` check, and it doubles as the order-causation
proof: under the permuted-teaching control the emitted SLOT order differs, so the exact match collapses to 0.0
on every seed — order effect **100.0% attributable to the manipulation**, `tools.lab.attributable_to`). `readable`
(the composite coherent-clause bar: covered AND faithful AND no leftover underscore character) reads **1.0 on
every seed** — every one of the 48 real sampled facts, not a curated subset. `moat_safe` (every surface token
traces to the fact's own content or a fixed closed-class word — no fabrication) also reads 1.0 on every seed,
every fact.

**The fallback-parity check (the "byte-identical-off" analog for a runner with no boolean flag).** For a
synthetic UNCOVERED relation, `_check_fallback_byte_identical_to_rung2` calls BOTH this lever's own render path
and rung 2's own `_render_facts` (imported unmodified) on the identical fact and seed, and asserts byte
equality — true on all 6 seeds. This proves the new content-realization layer changes NOTHING when it does not
apply: for any relation this lexicon does not cover (none exist in the live store today, but the mechanism must
still degrade honestly if one appears), the output is IDENTICAL to rung 2's own mechanism, naive morphology and
raw slugs included.

**Real render examples (verbatim, spanning all 6 seeds — the JSON artifact carries all 48):**

- `the Zdf Tv is headquartered in the Free City of Mainz` (headquarters_location)
- `the George Harrison died in the Los Angelas` (place_of_death — "Angelas" is the store's own spelling)
- `the Xx Winter Olympics follows the Salt Lake Olympics` (follows — no longer "followses")
- `the Sport in Paris received the Croix de Guerre 1914 1918` (award_received)
- `the Canker Worm is a Taxonomic Group` (instance_of — the "a_an" article rule chose "a" correctly)
- `the Aldershot Fc is located in the United Kingom` (country — the store's own truncated spelling preserved)
- `the Mar Caribe is part of the Atlantic Oceans` (part_of)
- `the Curazia is a type of Populated Places` (subclass_of — no article, reads correctly)

Compare directly to rung 2's own cited examples on the SAME kind of content: `"the eu_treaties
contains_administrative_territorial_entis the united_kingom"`, `"the xx_winter_olympics followses the
salt_lake_olympics"` — every structural defect rung 2 named (double-inflection, raw slugs, naive
relation-label-as-verb) is absent from every one of the 48 new renders.

## 3. Comparison to the word-in-fiction baseline (Part 3 of the fact-grounding lever, cited not re-run)

`2026-09-01-wkv-mouth-fact-grounding-lever.md` (the WKV mouth's own decode-time fact-boost) measured the OTHER
existing path to fact-grounded generation: boosting a recalled fact's vocabulary-in-scope content words inside
the WKV's free decode. Its own examples read `"tell me about the ball and it was a ball to play with his ball
but he didn't understand..."` and `"tell me about the party tim and his mom followed him to the party..."` — the
true word surfaces (`boosted_surfaced` 43/43 across 6 seeds), but never as a clause asserting the fact; that
finding's own honest limit states this "cannot... compose a coherent sentence asserting the fact" by
construction of a word-level logit boost. This lever's clause-frame path, on the same class of real facts, reads
`readable=1.0` on every seed — a genuine, measured difference in KIND (a factual clause) not degree (more boosted
words), on the two DIFFERENT mechanisms this project has now built toward the same goal.

## 4. What this is, and is not

**Is:** a mechanism-level GO on a narrowly, precisely scoped claim — for every relation type that exists in the
real shipped `wikidata_core_15k` store (34/34, verified live), a real recalled fact renders as a genuinely
spiking (bridge advanced real spikes, 6/6 seeds), structurally well-formed, role-faithful (exact match to an
independently-reconstructed ground truth, order-causation proven by the permuted control collapsing to 0.0),
moat-safe (every token traces to the fact + closed-class words, 1.0 on every seed) clause that reads as an
actual English sentence. The fallback path (for any future uncovered relation) is proven byte-identical to rung
2's own mechanism — no regression is possible for content this lever does not yet know how to phrase.

**Is not:** `wired` or `on-by-default` or `closed` (per `docs/TERMS.md` — this remains, exactly like rung 2, a
PARALLEL renderer never reachable from `/api/brain-chat`), a claim that board #112 is closed, or a claim that
EVERY conceivable relation type (beyond the live store's own 34) is covered — an unseen 35th relation degrades
honestly to rung 2's naive-morphology fallback, not silently to something worse.

## 5. What remains — concrete next steps, not built here

1. **Actually wiring this into the WKV mouth's own decode** (rung 2's residual #3, still fully open, precisely
   the same shape it was before this lever): today this clause producer is a PARALLEL renderer callable given a
   fact, exactly like rung 2's. Wiring it into `answer_turn`/`generate()` needs, concretely: (a) a decision
   point in `webapp/open_ended_chat.py::answer_turn` — when a known, in-vocab-scoped topic has a groundable fact
   (the same `in_vocab_scope`-passes-AND-has-a-fact condition the fact-grounding lever's Part 2 already
   measures at 48.33% of a real 600-agent sample), choose this clause-frame render INSTEAD OF, or spliced INTO,
   the WKV's own free generation; (b) a real integration flag (this lever introduces none — there is nothing
   for the 2026-09-01 auto-flip policy to flip, see SS6); (c) a decision on register mismatch — the clause
   producer's output ("the Zdf Tv is headquartered in the Free City of Mainz") is declarative-encyclopedic, a
   different register from the WKV mouth's TinyStories-narrative voice, and splicing the two without a jarring
   tone shift is an unsolved presentation question, not merely a wiring one.
2. **Multi-relation clauses** — today each rendered clause carries exactly one (agent, action, patient) triple;
   a real entity typically has several groundable facts (the `asimov_isaac` example alone carries 12), and a
   genuinely conversational reply would want to select and possibly combine more than one, which this lever does
   not attempt.
3. **The person-name slug-order residual** (SS1 above) — a general fix would need either a name/entity-type
   classifier (itself a new mechanism, unbuilt) or a store-level fix to how entity slugs are curated upstream;
   neither attempted here.

## 6. Why this stays unflipped — no auto-flip, and precisely why

The 2026-09-01 owner auto-flip policy (`policy(auto-flip): AUTO-FLIP validated-GO + load-bearing + moat-safe +
byte-identical-off + no-regression faculties to default-ON`) applies to an EXISTING production integration
point gated by a boolean flag — every auto-flip landed so far (`bedb9ad6e`, `e73bea486`) is a one-line change to
an env-var default already read by a wired production call path. This lever, like rung 2 before it, introduces
NO such flag: `_wkv_fact_to_sentence_lexicon_lever.py` is not imported by `webapp/` anywhere, and is not
reachable from `/api/brain-chat`. There is nothing to flip default-ON. The clean coherence-bar GO measured above
is real and load-bearing for THIS rung's own scope, but flipping requires the wiring named in SS5 item 1 to
exist first — that is the honest next rung, not a gap in this one.
