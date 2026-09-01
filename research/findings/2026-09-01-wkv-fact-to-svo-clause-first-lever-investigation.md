---
type: finding
status: first-lever
date: 2026-09-01
mechanism: board #112 rung 2 investigation + first lever — slotting a real recalled fact directly into the
  already-6-seed-GO `SpikingClauseProducer` SVO frame (`research/runners/_spiking_fluent_surface_derisk.py`),
  reusing its vocabulary-agnostic spiking-order mechanism instead of the WKV mouth's closed-vocabulary decode
verdict: mechanism-level GO on a NARROW, PRECISELY SCOPED claim — genuinely spiking (bridge spiked on 6/6
  seeds), structurally well-formed (DET SUBJ VERB DET OBJ, 1.0 on every seed, n=8/seed, 48 facts total) and
  role-faithful (1.0 on every seed) clauses from real `wikidata_core_15k` facts, with a permuted-teaching
  control collapsing well-formedness to 0.0 (proving the order is genuinely caused by the learned spiking
  primacy, not a host constant). This is NOT a claim of natural, semantically-fluent English (SS4's honest
  residual), NOT a wiring into `wkv_mouth_generator.generate()`'s own decode loop (SS0/SS5), and NOT a closure
  of board #112 rung 2.
lane: e-mouth-fluency / A1 (crutch-burndown), board #112, rung 2 (investigation + first lever, per this task's
  own instruction: "an HONEST NEGATIVE mapping exactly why the word won't cohere into a sentence yet is a
  first-class deliverable")
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: N/A — full 6-seed run, all in one bounded local process (12 builds of a 184-neuron bridge: 6 seeds
  x {main, permuted-control} arm), `free -m` checked >25GB available before running, well under the 15k-LTM-
  brain-build RAM-safety concern this task's own hygiene note warns about (no `ShardedPhasorStore`, no 15k-fact
  index — only `facts.json` read for an 8-fact-per-seed sample, the same low-memory retrieval convention the
  fact-grounding finding's own runner uses).
instrument: research/runners/_wkv_fact_svo_clause_first_lever.py — direct measurement against the real shipped
  `wikidata_core_15k` facts.json (8-fact seeded sample per seed, 48 facts total across 6 seeds), rendering each
  through a real `SimulationBridge`-backed `SpikingClauseProducer`, re-parsed by an independent structural
  parser that does not trust the producer.
runner: research/runners/_wkv_fact_svo_clause_first_lever.py (no args; runs all 6 canonical seeds)
external: NO-EXTERNAL-NEEDED — this reuses an ALREADY-6-seed-GO, already-cited mechanism
  (`_spiking_fluent_surface_derisk.py`, EMERGE-59/60/61) verbatim for the spiking-order part; the only new code
  is the fact->slot mapping and a structural re-parser, not a new method needing literature support.
artifacts:
  - research/findings/raw/_wkv_fact_svo_clause_first_lever.json (per-seed + aggregate + verdict, all examples)
  - research/runners/_spiking_fluent_surface_derisk.py (the reused, unmodified `SpikingClauseProducer` +
    `PLAIN_TRANSITIVE` frame — 6-seed GO per its own docstring, EMERGE-59/60/61)
  - research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md (this rung's own next-steps item 4, the
    finding that named "a structural (not decode-time) grounding lever")
  - research/findings/2026-09-01-wkv-mouth-invocab-scope-leadin-loophole-fix.md (rung 1, landed earlier this
    session on the same branch)
---

# From a boosted word to a grounded clause — investigation + first lever (board #112, rung 2)

## 0. What this is, and is not

`research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md` named the deeper residual behind board #112's
"clean unlock": the fact-boost lever (already landed) raises the odds the WKV mouth's free generation contains
the TRUE recalled word, but that word lands inside ordinary TinyStories fiction ("tim saw the ball in the
basket"), never as a clause asserting the fact. This task's own instruction pointed at a specific reuse path:
`_spiking_fluent_surface_derisk.py`'s `SpikingClauseProducer` / SVO frames — a DIFFERENT, already-6-seed-GO
spiking mechanism (EMERGE-59/60/61) that renders grammatical clauses by ordering abstract slot pools via real
spiking competitive queuing, "not a host template."

**This finding is an investigation, a first lever, and an honest map — explicitly not a claim that board #112
rung 2 is closed.** It does NOT modify `webapp/wkv_mouth_generator.py::generate()`'s own recurrent decode loop,
is NOT wired into `answer_turn`, and does NOT claim the resulting clauses read as natural English. What it DOES
show, with real measurement: the clause-FRAME machinery can take a genuinely recalled fact and render it as a
structurally well-formed, role-faithful, genuinely-spiking-ordered SVO clause — a different, parallel path
toward "coherent clause, not word-in-fiction" than biasing the WKV's own free decode, worth banking as a first
concrete step.

## 1. Why this mechanism, specifically — the investigation

`SpikingClauseProducer` (already 6-seed GO, `research/runners/_spiking_fluent_surface_derisk.py`) renders a
clause by ordering a FIXED SET of 6 abstract slot pools (180 neurons total: DET/SUBJ/VERB/OBJ + spares) via
real spiking rate-coded competitive queuing on a genuine `SimulationBridge`, then realizes each slot by calling
an IDENTITY `spell=str` callback on whatever payload string is supplied. Per that module's own docstring, "the
mechanism does not depend on which specific words fill the roles" — it is vocabulary-agnostic BY DESIGN. This
is the exact opposite of the WKV mouth's closed V=1000 embedding table (the wall Part 1 of the fact-grounding
finding quantified: ~74% of real facts have zero content-word overlap with that checkpoint's vocabulary at
all). A real fact's agent/action/patient words — however rare, however Wikidata-specific — can be slotted
DIRECTLY into this frame, without needing a trained embedding for them. That is the concrete reason this
mechanism, not the WKV's own decode, is where a first lever toward a genuine clause is cheapest to test.

## 2. The lever

For each real AFFIRM fact `(agent, action, patient)` (8 per seed, sampled the same way the fact-grounding
finding's own Part 3 sampled real store facts): `subject=agent`, `verb_3sg=emerge_v3(action)` (the SAME regular
3rd-person-singular morphology `_emerge59`'s production frames already use), `object=patient` — each kept as
ONE literal underscored token (the codebase's own existing `StubRenderer` convention, per `research/FAILURE_
LOG.md`'s 2026-09-01 `TieredFactStore` entry: "The asimov_isaac employers university_of_boston."). These fill
`PLAIN_TRANSITIVE`'s `(DET,"the"),(SUBJ,None),(VERB,None),(DET,"the"),(OBJ,None)` template, rendered through
`SpikingClauseProducer.emit()` UNMODIFIED — no change to the reused mechanism's own code.

## 3. Measurement — 6 seeds, real facts, an independent parser, a real control

Every number below is read directly from `research/findings/raw/_wkv_fact_svo_clause_first_lever.json`.

| Seed | n | well-formed | faithful | bridge spiked | permuted-control well-formed |
|---|---|---|---|---|---|
| 42 | 8 | 1.0 | 1.0 | True | 0.0 |
| 43 | 8 | 1.0 | 1.0 | True | 0.0 |
| 44 | 8 | 1.0 | 1.0 | True | 0.0 |
| 100 | 8 | 1.0 | 1.0 | True | 0.0 |
| 101 | 8 | 1.0 | 1.0 | True | 0.0 |
| 102 | 8 | 1.0 | 1.0 | True | 0.0 |
| **All 6 (48 facts)** | **48** | **1.0** | **1.0** | **6/6** | **0.0** |

An INDEPENDENT structural parser (`parse_plain_transitive`, which does not trust the producer) confirms every
one of the 48 rendered clauses is `the <SUBJ> <VERB-3sg> the <OBJ>` with the CORRECT agent/action/patient in
the correct roles. The permuted-teaching control (the SAME anti-cheat class the original 6-seed GO used)
collapses well-formedness to 0.0 on every seed — the DET-SUBJ-VERB-DET-OBJ order is genuinely CAUSED by the
learned spiking primacy gradient, not a host constant, exactly reproducing the original mechanism's own
load-bearing claim on this new (real-fact, not pseudo-word) content.

## 4. The honest residual, precisely — why this is not yet a "sentence"

Real render examples (verbatim, from the seed-44 and seed-101 artifacts):

- `the eu_treaties contains_administrative_territorial_entis the united_kingom`
- `the xx_winter_olympics followses the salt_lake_olympics`
- `the columbus_crew_soccer_club sports the association_football_club`
- `the aldershot_fc countries the united_kingom`

Two DISTINCT, honestly-separable gaps remain, neither closed by this lever:

1. **The vocabulary is raw Wikidata slugs, not natural English NPs.** `eu_treaties`, `united_kingom` (the
   store's own truncated slug — not this lever's bug, verified by checking the raw `facts.json` field) render
   as single underscored tokens because the frame's SUBJ/OBJ slots assume one lexical item per role; a real
   multi-word entity name ("United Kingdom") would need NP-internal structure this 5-slot frame does not have.
2. **The "verb" is a naive relation-label-as-verb morphological guess, not real semantic composition.**
   `emerge_v3` is a REGULAR 3sg-suffix rule with no knowledge of what a relation MEANS: `"follows"` (already
   arguably verb-shaped) becomes `"followses"` (double-inflected — its `-s`-ending base isn't in the small
   known-verb table, so the "already 3sg" auto-detect misses it); `"contains_administrative_territorial_enti"`
   (itself store-truncated) becomes `"...entis"`; `"sport"` happens to land passably as `"sports"` ("the X
   sports the Y" reads almost like real English) purely by morphological accident, not because the mechanism
   understood the relation. There is no mapping from a Wikidata RELATION TYPE (employer, country,
   headquarters_location, followed_by, sport, ...) to the right ENGLISH PREDICATE ("works for", "is located
   in", "is headquartered in", "was followed by", "plays") anywhere in this lever — that is a genuine
   relation-to-predicate GENERATION problem, structurally identical to the gap the KB-relation NL-parser
   direction of this project already works on `COMPREHENSION`-side, now named here for the `GENERATION` side.

Both gaps are STRUCTURAL, not tuning questions: no amount of more spiking-order training closes them, because
neither the multi-word-NP problem nor the relation-to-predicate problem is an ORDERING problem — they are
CONTENT-REALIZATION problems the current mechanism was never built to solve (it only orders abstract slots and
spells whatever string it is handed). This is the honest map the task asked for: the clause-frame lever closes
the "coherent-STRUCTURE-vs-embedded-in-fiction" gap (SS3's GO) but leaves the "coherent-ENGLISH-vs-slug-and-
naive-morphology" gap fully open, named precisely rather than hidden behind a passing structural score.

## 5. What remains — concrete next steps, not built here

1. **Relation-to-predicate mapping** (the load-bearing next lever): a small learned or curated map from each of
   the store's ~handful of distinct relation types to a real English verb phrase (e.g. `sport -> "plays"`,
   `employer -> "works for"`, `country -> "is located in"`) — this is what would turn `"the X sports the Y"`
   into `"the X plays Y"`. Whether this mapping itself should be BRAIN-BASED (a learned association, per the
   project's own brain-based-only standard) or a bounded closed-class lookup (analogous to `_FUNCTION_WORDS`,
   `_LEADIN_WORDS`, `_LEADINS` — all already-accepted small closed-class host tables in this exact codebase) is
   an open design call for whoever picks this up.
2. **Multi-word NP support** — either widen the frame to accept a short word LIST per SUBJ/OBJ slot (rendering
   each in sequence, still governed by the SAME spiking order for the higher-level SUBJ/VERB/OBJ positions), or
   pre-split the slug ("united_kingom" -> ["united","kingom"]) and adapt the parser accordingly.
3. **Actually wiring this into the WKV mouth's own decode** (not attempted here, and a materially different,
   larger lever than either of the above): today this is a PARALLEL renderer callable given a fact; connecting
   it to `answer_turn`/`generate()` so a known-topic turn can choose between free WKV generation and a rendered
   fact-clause (or splice one into the other) is unbuilt, un-scoped, and explicitly out of this investigation.
