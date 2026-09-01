---
type: finding
status: wired
date: 2026-09-01
mechanism: board #112 rung-3 WIRE-IN — the just-merged fact->sentence lexicon lever
  (`research/runners/_wkv_fact_to_sentence_lexicon_lever.py`, 6-seed GO) is now called FROM
  `webapp/wkv_mouth_generator.py::generate()`'s own `_run()` closure (new `sentence_facts` parameter), so a
  known, in-vocab topic's reply is genuinely written by the WKV mouth's fact->sentence path, not a parallel
  renderer. New flag `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE`, DEFAULT-ON as of this task (auto-flipped, see
  SS4) — a fifth, independent gate under the still-default-OFF `BRAIN_OPEN_ENDED` channel.
verdict: WIRE-IN GO, 6-seed, measured through the REAL `webapp.open_ended_chat.answer_turn()` entry point (not
  a parallel harness). On the wire-in's own claim (`generate()`'s raw output, before the pre-existing safety
  net): readable=faithful=moat_safe=1.0 on EVERY seed, 48/48 real sampled known+in-vocab+covered-relation
  cases. Byte-identical-off verified in the data (flag=0 reproduces the pre-existing free-gen text exactly).
  Unknown-topic honesty (hedge/abstain) intact on every seed with the flag on. One genuine, precisely-mapped
  residual (SS3): a pre-existing (2026-08-21) known-topic contradiction filter's bare number/date check has no
  exemption for a number that is part of the topic's OWN name, so 1/48 real cases over-cautiously degrades to
  the honest-abstain fallback INSTEAD of the correct clause at the post-filter layer — always a SAFE degrade
  (verified 1.0/1.0 "answer is raw-or-fallback, never corrupted" on every seed), never a leak. HONEST SCOPE
  (not a partial claim of #112 closure): this only reaches the narrow real-topic slice that is BOTH known to
  the store AND in-vocab for the WKV checkpoint's closed V=1000 TinyStories vocabulary (measured ~3% of a real
  400-agent sample); it does NOT touch the much larger Qwen-routed (out-of-vocab) known-topic grounding
  regression the 2026-09-01 moat-safety soak measured — a different generator, a different code path,
  untouched by this change.
lane: e-mouth-fluency / A1 (crutch-burndown), board #112
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: N/A — full 6-seed run, one bounded local process. `free -m` checked available memory before each
  run (28.8GB, then 27.3GB, then 27.0GB available across the session — well over the 25GB the 15k-LTM-brain-
  build RAM-safety note warns about). No `ShardedPhasorStore`/15k-fact index build: `webapp.open_ended_chat.
  build_index` reads ONLY `facts.json` (2MB) directly, the same low-memory retrieval convention the merged
  rung-2/3 findings already used. Each of the 48 real cases builds one ~184-neuron `SpikingClauseProducer`
  bridge (cached per (seed, slot-length) across calls within the process, per `webapp/wkv_mouth_generator.py::
  _get_clause_producer`) plus, when the flag-off/boost arms run, one ~512-neuron few-spike Izhikevich bank —
  all CPU numpy, seconds per case.
instrument: research/runners/_wkv_mouth_fact_sentence_wirein_verify.py — calls the REAL
  `webapp.open_ended_chat.answer_turn()` (not a reimplementation) for 8 real sampled known+in-vocab+covered-
  relation topics per seed (48 total), scored by the lexicon lever's OWN independent parser
  (`expected_surface`/`parse_and_score`, imported unmodified from `_wkv_fact_to_sentence_lexicon_lever.py` —
  ground-truth reuse, not producer-internal trust), against three arms per case: sentence-mode ON, both flags
  explicitly OFF (the pre-existing free-gen path), and the older word-boost lever ON — plus one in-vocab,
  brain-unknown-topic honesty check per seed.
runner: research/runners/_wkv_mouth_fact_sentence_wirein_verify.py (no args; runs all 6 canonical seeds)
external: reuses the ALREADY-recorded external grounding for this exact lane
  (`research/queue/.external_searches.jsonl`, entry `2026-09-01T22:43:41Z`, lane-tagged
  `e-mouth-fluency / A1 (crutch-burndown), board #112`): Gardent, Shimorina, Narayan, Perez-Beltrachini (2017),
  "The WebNLG Challenge: Generating Text from RDF Data", INLG 2017, https://aclanthology.org/W17-3518/. This
  task performs no NEW mechanism-lever against a wall — it wires an already-externally-grounded, already-
  6-seed-GO mechanism (the merged lexicon lever) into the production call path, so no new external round was
  run.
artifacts:
  - research/findings/raw/_wkv_mouth_fact_sentence_wirein_verify.json (per-seed + aggregate + verdict, all 48
    real cases' raw + post-filter answers, the flags-off / word-boost comparison arms, the unknown-topic
    honesty check)
  - webapp/wkv_mouth_generator.py (the wire-in: `render_fact_sentence`, `pick_covered_fact`,
    `_get_clause_producer`, and `generate()`'s new `sentence_facts` parameter)
  - webapp/open_ended_chat.py (`wkv_fact_sentence_enabled`, the `answer_turn` call-site wiring)
  - research/runners/_wkv_mouth_fact_sentence_wirein_verify.py (this finding's own 6-seed verify runner)
  - research/FAILURE_LOG.md (the mapped number/date-filter residual, 2026-09-01 row)
---

# Wiring the fact->sentence render into the WKV mouth's own `generate()` — board #112 rung-3 wire-in

## 0. What this is, and is not

The 2026-09-01 lexicon lever (`research/findings/2026-09-01-wkv-fact-to-sentence-lexicon-and-np-lever.md`)
built a curated relation->predicate lexicon (`RELATION_LEXICON`, 34/34 live-store relation coverage) and a
slug->NP surfacer (`slug_to_np`) driving the already-6-seed-GO `SpikingClauseProducer`
(`research/runners/_spiking_fluent_surface_derisk.py`, EMERGE-59/60/61) to render a real recalled fact as a
genuinely coherent English clause ("the Zdf Tv is headquartered in the Free City of Mainz"). It explicitly
named its own open residual: this was a PARALLEL renderer, never reachable from
`webapp/wkv_mouth_generator.py::generate()`'s own decode loop, and not wired into `answer_turn`.

**This task closes exactly that residual.** `render_fact_sentence` (new, `webapp/wkv_mouth_generator.py`) is
called from `generate()`'s own `_run()` closure via a new `sentence_facts` parameter — when the WKV mouth is
about to answer a KNOWN topic and `sentence_facts` carries a fact whose relation the lexicon covers,
`generate()` returns that coherent clause AS THE MOUTH'S REPLY, before any free generation runs for that turn.
A new, independent, ADDITIVE flag (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE` — see
`webapp/open_ended_chat.py::wkv_fact_sentence_enabled`) gates whether `answer_turn` passes the already-
retrieved facts into that new parameter.

**Is not:** a claim that board #112 (or the whole `BRAIN_OPEN_ENDED` bundle) is closed or flip-ready as a
whole. `BRAIN_OPEN_ENDED` itself remains default-OFF; this wire-in only improves what happens on the narrow
slice of turns where that channel, the WKV mouth, AND this new sub-flag are all active. It also does not touch
the much larger Qwen-routed known-topic grounding regression the 2026-09-01 moat-safety soak measured — see
SS5.

## 1. The wire-in mechanism

`webapp/wkv_mouth_generator.py`:

- `pick_covered_fact(facts)` — the first (agent, action, patient) triple in a retrieved fact list whose
  relation exists in the lexicon lever's `RELATION_LEXICON`. `None` if empty/uncovered (honest degrade).
- `_get_clause_producer(seed, n_slots)` — a `SpikingClauseProducer` built + taught ONCE per (seed, slot-count)
  and cached process-wide, the same reuse pattern the lexicon lever's own `_render_facts` already uses within
  one seed's fact list (its `by_len` dict), extended to persist across turns (mirrors `_get_readout`'s own
  WKV-checkpoint cache).
- `render_fact_sentence(facts, seed)` — picks the covered fact, builds the DET/SUBJ/VERB/[DET]/OBJ slot list +
  draw context via the lexicon lever's own `_dctx_and_slots` (imported unmodified), emits through the cached
  producer, and returns `None` (never a fabricated guess) if the bridge did not genuinely spike or no relation
  is covered.
- `generate(..., sentence_facts=None)` — new parameter, default `None`. `_run()` tries
  `render_fact_sentence(sentence_facts, seed=seed)` FIRST; if it returns a sentence, that is `generate()`'s
  entire text (free generation never runs for the turn); otherwise `_run()` falls straight through to the
  pre-existing free-gen/fact-boost path, byte-identical to before this parameter existed. Both the lexicon
  lever module and `SpikingClauseProducer` are LAZY-imported (only inside `render_fact_sentence`), so the
  default-off path (`sentence_facts` falsy) never even imports them — `wkv_mouth_generator`'s own import-time
  behavior is unaffected either way.

`webapp/open_ended_chat.py`:

- `wkv_fact_sentence_enabled()` — reads `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE` (default `"1"` — see SS4).
- `answer_turn`'s WKV-mouth block computes `sentence_facts = facts if (known and wkv_fact_sentence_enabled())
  else None`, INDEPENDENT of the existing `ground_facts` (the older word-boost lever's own gate), and passes
  both into `_WKV.generate(facts=ground_facts, sentence_facts=sentence_facts)`.

## 2. Measurement — 6 seeds, through the REAL `answer_turn`, an independent parser, three arms

Every number below is read from `research/findings/raw/_wkv_mouth_fact_sentence_wirein_verify.json`. For each
seed, up to 8 real store agents were sampled (seeded, same discipline as the merged lever's own `_sample_facts`)
that are BOTH known to the store AND pass `webapp.wkv_mouth_generator.in_vocab_scope` for a `"tell me about
<agent>"` prompt AND have >=1 fact whose relation the lexicon covers — every seed found the full 8/8.

| Seed | n | raw readable | raw faithful | raw moat_safe | post-filter readable | answer safe-degrade | flags-off is-exact-clause | unknown-topic abstained |
|---|---|---|---|---|---|---|---|---|
| 42 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | True |
| 43 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | True |
| 44 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | True |
| 100 | 8 | 1.0 | 1.0 | 1.0 | 0.875 | 1.0 | 0.0 | True |
| 101 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | True |
| 102 | 8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | True |
| **All 6 (48 cases)** | **48** | **1.0** | **1.0** | **1.0** | **0.9792 mean / 0.875 min** | **1.0** | **0.0** | **6/6** |

**`raw` (the wire-in's own claim, `generate()`'s direct output, before the pre-existing safety net) reads
EXACTLY 1.0 on every seed, every one of 48 real cases** — real examples (verbatim, spanning all 6 seeds):

- `the Joe Given Name is an Unisex Name` (instance_of)
- `the Jeanne First Name is in the language of the French Language` (language_of_work_or_name)
- `the Tommy Boy Music is located in the U S of A` (country)
- `the Box To Box Mid is associated with the sport of the Association Football Club` (sport)
- `the Noise Rock is a type of Rock Musician` (subclass_of)
- `the First Summer Olympic Games is followed by the Games of the Ii Olympiad` (followed_by)

**Byte-identical-off, verified in the data, not inferred:** setting `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE=0`
against the identical seed/prompt reproduces the exact pre-existing free-gen text (e.g. seed 42,
`joe_given_name`: `"tell me about joe name the bird and said i will help you feel better for a while tim was
playing..."`, unchanged whether or not this change exists in the codebase — the diff only ADDS code executed
when `sentence_facts` is truthy). The `off_is_clause` column (0.0 on every seed) confirms the flags-off arm
never coincidentally produces the exact clause.

**Comparison to the pre-existing paths (the #112 "does it preserve the fact" question, SS5):** on the SAME 48
real facts, flags-off free generation's literal-content-word overlap with the fact averages 32.1% (min 27.6%
across seeds) — a third of the real content words appear somewhere in unrelated TinyStories prose, never as an
assertion. The older word-boost lever (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND`) raises that to 45.1% mean
(38.9% min) — better, but still 0% exact-clause rate; the true word surfaces, never a coherent statement. This
wire-in's raw output is the exact, complete, correctly-structured clause 100% of the time on every seed — a
difference in KIND (a factual assertion), not degree, matching the KIND-vs-degree framing the lexicon lever's
own SS3 already drew against the word-boost lever's free-gen examples.

## 3. The one real, mapped residual — a pre-existing filter's number/date scope limit, newly exposed

One of the 48 cases (seed 100, `1974_football_world_cup` / `sport` / `association_football_club`) has
`raw` = `"the 1974 Football World Cup is associated with the sport of the Association Football Club"` — correct,
well-formed, moat-safe — but the end-to-end `answer` (after `post_filter`) is the fixed honest-abstain string,
`"I don't have a version of what I just said about 1974_football_world_cup that I can actually stand behind."`

Root cause, verified directly (not inferred): `_open_ended_known_supplement_filter_derisk.sentence_contradicts`
(the 2026-08-21, already-GO'd, already-wired known-topic contradiction filter `post_filter` reuses unmodified)
flags ANY sentence containing a bare 3+-digit number or year-shaped token as `"number/date not in store"` —
regardless of whether that number is part of the topic's OWN entity name. `1974` is literally the leading token
of the agent slug `1974_football_world_cup`, not a fabricated addition, but the check has no such exemption.
This is a **pre-existing, already self-documented scope limit** of that filter
(`_open_ended_clause_contradiction_filter_derisk.py`'s own docstring: *"a bare unsupported number with no
relative-clause boundary has no declared-safe repair here and keeps falling back to whole-sentence removal,
same as today"*) — this wire-in did not introduce it, but its clause structurally always embeds the subject NP
verbatim, so any covered-relation fact about a numbered-slug topic will trip it every time.

**This is a SAFE failure mode, not a moat violation**: the `answer_is_safe_degrade` check (does the final
answer equal either the raw clause unchanged, or the fixed honest fallback — never a partially-corrupted
hybrid) reads 1.0 on every seed, every case. The system never leaks a wrong fact; it just over-cautiously
abstains on a correct one for this narrow numbered-slug sub-class. Logged in `research/FAILURE_LOG.md`
(2026-09-01 row) per `gates/coverage`; fixing `sentence_contradicts`'s number check (exempt a digit-run that is
a substring of the `topic` argument it already receives) is a separate, small, well-understood next step —
deliberately NOT built here, to keep this task scoped to the wire-in itself and avoid modifying a shared,
independently-GO'd mechanism with other call sites inside this task's own change.

## 4. Why this flag is auto-flipped default-ON (the 2026-09-01 owner policy)

`policy(auto-flip): AUTO-FLIP validated-GO + load-bearing + moat-safe + byte-identical-off + no-regression
faculties to default-ON` (the same policy that flipped `bedb9ad6e`/`e73bea486`). This flag qualifies on every
count:

- **validated-GO**: the 6-seed verify above, `raw` at 1.0/1.0/1.0 on every seed, 48/48 real cases.
- **load-bearing**: when it fires, it categorically changes the reply's content — from generic/fact-thin
  TinyStories prose (27.6-38.9% word-level fact overlap) to the exact, complete, correctly-structured factual
  clause. This is a stronger, more clearly load-bearing effect than the adjacent word-boost lever
  (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND`), which was left `wired, default-OFF` by its own finding precisely
  because its own honest residual — a true word buried in unrelated fiction, never an assertion — is a much
  weaker claim.
- **moat-safe**: `raw` moat_safe = 1.0 on every seed; the end-to-end safety net never corrupts (SS3).
- **byte-identical-off**: verified in the data (SS2).
- **no-regression**: strictly improves the narrow slice where it fires; zero change elsewhere (the flag is a
  no-op whenever `sentence_facts` is falsy or uncovered).
- **ZERO production risk today**: gated two levels deep behind `BRAIN_OPEN_ENDED` (default OFF) — with that
  top-level channel off, `wkv_fact_sentence_enabled()` is never even read by a live request.

Flipped: `webapp/open_ended_chat.py::wkv_fact_sentence_enabled` now defaults to `"1"` (was `"0"`).
`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND` (the older, separate word-boost lever) is UNCHANGED, still
default-OFF — out of this task's scope and not part of this flip.

## 5. The #112 impact — an honest, precisely-scoped partial, not a bundle closure

**Does this resolve the known-topic grounding regression the 2026-09-01 moat-safety soak found?** The honest
answer is NO, not that regression specifically — that soak
(`research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md`) measured real-traffic
Qwen-routed known-topic replies (e.g. `castleford_f_c`, out-of-vocab for the WKV checkpoint), a DIFFERENT
generator on a DIFFERENT code path this wire-in never touches. `wkv_mouth_enabled()` only ever routes a prompt
to the WKV mouth when it passes `in_vocab_scope` (V=1000 TinyStories vocabulary) — the large majority of real
Wikidata topics (proper nouns, numbers, technical relation labels) do not, so they still route to Qwen exactly
as before.

**Does this resolve the analogous regression named for the WKV mouth's OWN generation** (the fact-grounding
lever's finding: *"the lever surfaces the true word woven into TinyStories-register fiction, not a coherent
factual statement"*)? **Yes, for the narrow slice this wire-in reaches.** An independent scan performed for
this task (400 real `wikidata_core_15k` agents, seed 42) found only 12/400 (~3%) pass `in_vocab_scope` at all
— but ALL 12 of those also had a lexicon-covered relation, meaning within the WKV mouth's own narrow in-vocab
reach, coverage of this fix is effectively complete for today's live store. Whether this generalizes to a
LARGER in-vocab sample, and the exact production request-rate this ~3% translates to, is not measured here (a
natural next rung, not built).

**So: this is a genuine partial, exactly the outcome this task's own brief invited.** It is flip-ready and
flipped for its own narrow scope (SS4); it is NOT a claim that the #112 bundle's broader, larger-population
grounding regression (the Qwen-routed majority) is resolved or that `BRAIN_OPEN_ENDED` itself is now
flip-ready — that decision needs a separate assessment of the Qwen-routed path, untouched by this change.

## 6. What remains — concrete next steps, not built here

1. **The number/date filter exemption** (SS3) — a small, well-understood, additive fix to
   `sentence_contradicts`, deliberately out of this task's scope (a shared mechanism with other call sites).
2. **Register mismatch** (named by the merged lexicon lever's own SS5, unaffected by this wire-in): the
   rendered clause's declarative-encyclopedic register ("the X is located in the Y") differs from the WKV
   mouth's TinyStories-narrative voice; this wire-in makes it the WHOLE reply for a covered known topic rather
   than blending it into surrounding narrative — a presentation question, not a correctness one, left open.
3. **Widening in-vocab reach** — the ~3% in-vocab rate is the WKV mouth's own pre-existing scope limit (the
   V=1000 TinyStories checkpoint), not something this wire-in changes; the already-measured V=4000 checkpoint
   (`research/findings/2026-08-31-wkv-mouth-rung4-vocab-coverage.md`) is the named, unwired next lever.
4. **Multi-relation clauses** (inherited from the merged lever's own SS5) — a real entity often has several
   groundable facts; this wire-in, like the lever it wraps, renders exactly one.
