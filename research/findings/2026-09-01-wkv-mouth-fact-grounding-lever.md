---
type: finding
status: wired
date: 2026-09-01
mechanism: fact-token decode-time logit boost for the from-scratch WKV/SSM spiking mouth (board #112 "clean
  unlock" — the moat-soak's named next action)
verdict: mechanism-level GO, 6-seed validated (42/43/44 dev, 100/101/102 blind — every dev+blind checkpoint on
  disk). WIRED, additive, default-OFF (BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND), byte-identical when off
  (asserted, not inferred — see Part 4, true in all 6 seeds). Aggregated over all 6 seeds' curated demos
  (43 total examples): the boosted generation surfaces the TRUE recalled content word in 43/43 (100%) —
  EVERY seed individually also reads 100% boosted-surfaced — vs 25/43 (58.1%) at baseline, with 35/43 (81.4%)
  newly surfaced by the boost alone; every boosted fact word survives the existing honesty post-filter unedited
  (43/43). The honest residual, precisely quantified rather than assumed: this checkpoint's V=1000 vocabulary
  structurally cannot express ~74% of the real store's facts at all (no trained embedding exists for the word,
  full stop; this ceiling is itself seed-stable, 25.87-25.97% across all 6 checkpoints), and even where it can,
  the lever surfaces the true WORD woven into TinyStories-register fiction, not a coherent factual STATEMENT —
  "word-level grounding," not "fact-assertion." Both limits are load-bearing and named below, not hidden.
lane: e-mouth-fluency / A1 (crutch-burndown), board #112
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: N/A — full 6-seed dev+blind validation (all 6 checkpoints ship on disk,
  `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{42,43,44,100,101,102}.npz`), each seed a genuinely independent
  checkpoint + independent random agent sample + independent demo-example selection, not a repeat of the same
  data. Pool-routing this sweep was ATTEMPTED first per this task's own compute-routing instruction and found
  infeasible: `tools/pool_provision.sh`'s rsync deliberately excludes both `webapp/` (this lever's wiring layer)
  and `bridges/` (the checkpoint `.npz` files themselves) from what it ships to the mini-PC nodes — a structural
  scope boundary of that infrastructure (pool nodes run pure `research/runners/`+`sim/` work, never production
  code or model checkpoints), not a bug to route around. Run instead as 6 independent bounded local processes
  (NOT the 15k-LTM-brain-build OOM path this task's RAM-safety rule warns about — no `SimulationBridge`/
  `ShardedPhasorStore` for the store is ever built; only `facts.json` is read and one ~512-neuron few-spike
  Izhikevich bank per generation call, ~20-40s/seed), `free -m` available memory checked >25GB throughout.
instrument: research/runners/_wkv_mouth_fact_grounding_derisk.py — direct measurement against the real shipped
  `wikidata_core_15k` facts.json (15,000 AFFIRM triples) and the real `wkv_ssmU6_v1000_d128_seed{42,43,44,100,
  101,102}.npz` checkpoints; the demo generations run the GENUINE few-spike Izhikevich spiking-WTA decode
  (`research.runners._wkv_fewspike_read_derisk`, reused unchanged) and the real live honesty post-filter
  (`webapp.open_ended_chat.post_filter`).
runner: research/runners/_wkv_mouth_fact_grounding_derisk.py (--seed 42|43|44|100|101|102)
external: NO-EXTERNAL-NEEDED — an additive decode-time logit boost over a fixed candidate set is the same
  category as this codebase's own pre-existing `_apply_repetition_controls` (CTRL-style repetition penalty /
  no-repeat n-gram ban, both citing Keskar et al. 2019 / Fan et al. 2018 in that function's own docstring); this
  rung reuses that established, already-cited decode-control pattern rather than introducing a new one.
artifacts:
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk.json (seed 42; all four parts below, plus the full
    before/after generations)
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_seed43.json
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_seed44.json
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_seed100.json
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_seed101.json
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_seed102.json
  - research/findings/raw/_wkv_mouth_fact_grounding_derisk_6seed_aggregate.json (the aggregate table below,
    computed directly from the 6 per-seed artifacts)
  - research/findings/2026-08-31-wkv-mouth-rung4-vocab-coverage.md (the prior, corpus-word-only coverage
    measurement this finding's Part 1 extends to the FACT side)
  - research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md (the moat soak that
    named this rung's exact next action)
  - research/findings/2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md (the WKV mouth's own wiring, reused
    unchanged here)
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - research/FAILURE_LOG.md (the `in_vocab_scope` lead-in-loophole entry this measurement found, 2026-09-01)
---

# The WKV mouth's first fact-grounding lever — a real, measured, honest mixed result

## 0. What this is

The moat soak asked one question: can a known-topic reply from the from-scratch WKV/SSM spiking mouth surface
the brain's real recalled fact, instead of trading grounded recall for fact-thin free generation? This task
built the FIRST concrete lever toward that — an additive, default-OFF, decode-time logit boost for the recalled
fact's in-vocab content words (`webapp.wkv_mouth_generator.fact_grounding_ids` / `_apply_fact_boost`, wired into
`webapp.open_ended_chat.answer_turn` behind `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND`) — and measured it against
the REAL shipped `wikidata_core_15k` store, not a synthetic example set. Every number below is read directly
from `research/findings/raw/_wkv_mouth_fact_grounding_derisk.json`, produced by
`research/runners/_wkv_mouth_fact_grounding_derisk.py`.

**The honest headline in one sentence:** the lever is real, wired, and demonstrably works at the WORD level (it
increases how often the true recalled word appears), but this checkpoint's closed V=1000 vocabulary means most
real facts have no word for it to boost in the first place, and even a successfully-boosted word lands inside
TinyStories-register fiction rather than a coherent factual sentence — so this rung is a genuine partial unlock,
not a closure of board #112.

## 1. Part 1 — the vocabulary ceiling, measured against the real fact corpus (not the raw corpus text)

`2026-08-31-wkv-mouth-rung4-vocab-coverage.md` already measured this checkpoint's coverage of TinyStories corpus
TEXT (9.55% of unique words). This finding measures the same ceiling from the FACT side — the thing that
actually matters for "can it name a Wikidata entity's real facts" — by tokenizing every one of the 15,000 AFFIRM
`(agent, action, patient)` triples in the real shipped store, excluding function words (`_FUNCTION_WORDS`, the
same set `in_vocab_scope` already uses), and checking literal string membership in the checkpoint's vocabulary:

| Metric | Count | Fraction |
|---|---|---|
| Facts whose PATIENT has >=1 real content-word vocab hit | 3,889 / 15,000 | 25.93% |
| Facts whose agent/action/patient (any field) has >=1 hit | 6,072 / 15,000 | 40.48% |

**This is the hard structural wall, precisely quantified.** The majority (~74% on the patient, ~60% on any
field) of real facts have ZERO content-word overlap — their patient value is a Wikidata slug
(`rugby_leauge`, `castleford_f_c`, `deutsche_arbeiter_partei`) this checkpoint never trained an embedding for,
and a word-level, closed-vocabulary decoder cannot produce a word it has no embedding row for — full stop,
not a tuning question. No decode-time lever (boosting, forcing, constraining) can inject a word's MEANING into
a `[V,D]` embedding table that never saw it during training. Closing this specific gap needs a wider-vocab /
subword checkpoint (the V=4000 checkpoint `2026-08-31-wkv-mouth-rung4-vocab-coverage.md` already measured, not
yet wired into this generation path) or new grounded fine-tuning — both out of this task's scope, named here as
the concrete next rung.

## 2. Part 2 — how often does the WKV mouth even reach a real known topic, and a real bug this found

Reaching Part 3's lever requires TWO things at once: the user's message must pass `in_vocab_scope` (or the WKV
mouth never engages, Qwen handles the turn instead) AND the topic's own facts must have a content-word hit.
Measured on a seeded sample of 600 real store agents, queried the SAME way the moat soak's own battery phrased
its known-topic queries (`"tell me about " + <agent>`):

| Metric | Count / 600 | Fraction |
|---|---|---|
| `in_vocab_scope` passes | 409 | 68.17% |
| ...AND has >=1 fact content-word hit | 290 | 48.33% |
| conditional (of scope-pass, has grounding available) | 290 / 409 | 70.90% |

**This is higher than Part 1's raw ceiling would predict, and the reason is a real bug this measurement found,
not a feature.** `in_vocab_scope("tell me about " + <anything>)` returns `True` even for total nonsense
(`in_vocab_scope("tell me about zzznonsenseword qqqgibberish")` → `True`) — `"tell"`/`"me"`/`"about"` are all
in this checkpoint's vocabulary and NOT in `_FUNCTION_WORDS`, so the lead-in phrase alone satisfies
`min_content_hits=2` without the actual topic contributing anything. Logged to `research/FAILURE_LOG.md`
(2026-09-01 entry), **not fixed here** — out of this task's scope, and `BRAIN_OPEN_ENDED` itself stays
default-OFF so it is not live in production — but it directly motivates this lever: **the WKV mouth already
reaches real Wikidata topics far more often than its own documented vocabulary scope suggests, and today those
engaged turns get ZERO fact grounding at all** unless this new flag is enabled. Fixing that gate precisely
(scoring content hits only on the text AFTER the lead-in strip) is the natural companion next step, not
attempted here.

## 3. Part 3 — the lever itself, before/after, on real recalled facts, all 6 seeds

7-8 real agents were selected per seed (one per distinct relation type, from that seed's own real
known-topic-and-groundable query pool measured in Part 2 — a genuinely different random sample each seed, not
the same 8 examples repeated). For each, `generate()` ran twice at that seed with an identical
`"tell me about <agent>"` prompt and identical repetition-control settings — once with `facts=None` (baseline,
the existing default), once with `facts=<the real triples>` (the new lever, `fact_boost=6.0` default) — through
the SAME genuine few-spike spiking-WTA decode, then both raw outputs were run through the real live
`webapp.open_ended_chat.post_filter`.

| Seed | n | baseline surfaced | boosted surfaced | newly surfaced | survives post-filter | per-seed verdict |
|---|---|---|---|---|---|---|
| 42 (dev) | 7 | 3 | 7 | 6 | 7 | GO |
| 43 (dev) | 7 | 3 | 7 | 6 | 7 | GO |
| 44 (dev) | 8 | 6 | 8 | 7 | 8 | GO |
| 100 (blind) | 7 | 5 | 7 | 6 | 7 | GO |
| 101 (blind) | 7 | 1 | 7 | 7 | 7 | GO |
| 102 (blind) | 7 | 7 | 7 | 3 | 7 | UNDEFINED (tie, see note) |
| **Total (6 seeds)** | **43** | **25 (58.1%)** | **43 (100%)** | **35 (81.4%)** | **43 (100%)** | **5/6 GO** |

**The load-bearing number: boosted-surfaced reads 100% in EVERY single seed, individually and in aggregate** —
the mechanism is not a single-seed artifact. Seed 102's `UNDEFINED` per-seed verdict is a tie (baseline ALSO
hit 7/7 on that seed's own curated set, not a boost failure — `boosted_surfaced` was still 7/7 there too); it
happens because the demo-selection method favors facts with more grounding words, which also raises baseline's
own chance-overlap odds over a 40-token continuation on some seeds. The AGGREGATE 25/43 vs 43/43 (and 35/43
newly surfaced from nothing) is the number to read, not any single seed's tie.

**Two concrete examples from seed 42 (full text for all 6 seeds in the per-seed artifacts):**

- `strech_five` (real fact: `sport = basket_ball`). Baseline: *"tell me about the big dog and the cat said i
  will help you climb it too hard for making them fall from the tree..."* — zero connection to the entity.
  Boosted: *"tell me about the ball and it was a ball to play with his ball but he didn't understand why they
  could not move him or ball again and again until this time tim saw the ball in the basket there were many"*
  — `ball` and `basket` both newly surface, directly traceable to the boosted fact tokens.
- `jeffersonian_republicans` (real facts: `country=u_s_of_a`, `followed_by=us_democratic_party`,
  `headquarters_location=washington_dc`). Baseline: generic dog-and-cat fantasy, unrelated. Boosted:
  *"tell me about the party tim and his mom followed him to the party with her friends at the party..."* —
  `party` and `followed` both newly surface (from `us_democratic_party` and `followed_by`).

**The honest limit, stated precisely, not hidden.** In every example, the surfaced word lands inside ordinary
TinyStories-register fiction ("tim saw the ball in the basket," "followed him to the party with her friends"),
never as a factual claim ABOUT the entity ("strech_five plays basketball"). This lever raises the odds a TRUE
word appears somewhere in the reply; it does not (and by construction of a word-level decode-time logit boost,
cannot) compose a coherent sentence asserting the fact. Closing THAT gap needs a mechanism that shapes sentence
STRUCTURE around the fact (e.g. conditioning generation on a fact-bearing template, or steering earlier in the
recurrent state rather than only the output logits) — a different, larger lever than this one, named as the
next rung, not built here.

**Was this net-safer or net-riskier for the moat?** Net-safer across all 43 examples over 6 seeds, by the SAME
honesty instrument the production pipeline already runs: `post_filter`'s contradiction check
(`_facts_as_relation_pairs` + `_clause_filter_sentence`) never flagged either arm's output on ANY example in
ANY seed (0/43 dropped either way, every seed individually 0/n too) — the boosted output is not MORE likely to
trip the existing moat than baseline, and intuitively carries less purely-invented specific content (a
boosted-but-vague "the party" is not a false claim the way baseline's entirely-unrelated "the big dog and the
cat" narrative risks reading as if it were about the topic). This is still a qualitative reading (the
post-filter's own gazetteer only catches 3 relation shapes plus bare numbers — see the moat-soak finding's own
documented coverage gap), not a quantitative fabrication-rate claim on a scale the existing instrument can
grade precisely — see "What this is not."

## 4. Part 4 — the mechanism is a genuine no-op when off, asserted not inferred

| Check | Result |
|---|---|
| `_apply_fact_boost(lg, [], 6.0) is lg` (empty ids) | `True` — same object, zero allocation |
| `_apply_fact_boost(lg, None, 6.0) is lg` (no facts) | `True` |
| `_apply_fact_boost(lg, [3,5], 0.0) is lg` (zero boost) | `True` |
| `_apply_fact_boost(lg, [3,5], 6.0)` actually changes those two logits by +6.0 | `True` |
| `generate(..., facts=None)` called twice at the same seed | byte-identical text both times |
| `open_ended_chat.wkv_fact_grounding_enabled()` with the env var unset | `False` (default OFF) |
| ...with `BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND=1` | `True` |

All seven checks above hold at EVERY one of the 6 seeds independently (`mechanism_noop_checks_all_true_every_
seed: true` in the aggregate artifact), not just seed 42. Every pre-existing call site in
`webapp/open_ended_chat.py::answer_turn` passes `facts=None` unless the new flag is truthy AND the topic is
`known` — the WKV mouth's existing default-ON path (`wkv_mouth_enabled()`, unrelated to this new flag) is
unchanged for every caller that does not opt in.

## 5. What this is, and is not

**Is:** a real, `wired` (per `docs/TERMS.md` — reachable from `/api/brain-chat` → `answer_turn` →
`wkv_mouth_generator.generate`, on a request with `BRAIN_OPEN_ENDED` + `BRAIN_OPEN_ENDED_WKV_MOUTH` +
`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_GROUND` all set + a known, in-vocab-scoped topic), additive, default-OFF
decode-time lever, asserted byte-identical off (6/6 seeds), that measurably and CONSISTENTLY (100% in every one
of 6 seeds) increases how often the WKV mouth's raw generation contains the TRUE recalled content word on real
known-topic queries (Part 3), without increasing how often the existing honesty post-filter has to intervene
(0/43 across all 6 seeds).

**Is not:** `on-by-default`, a claim that the mouth now "states its facts" (word-presence is not fact-assertion
— see Part 3's honest limit), or a closure of board #112. The corpus-level ceiling (Part 1: ~74-60% of real
facts have zero content-word overlap at all) is a hard architectural wall this lever cannot touch — it is
scoped, honestly, to the minority of facts this checkpoint's fixed vocabulary can express at all. The demo
selection (one per distinct relation type, per seed, from that seed's own real known-topic-and-groundable
pool) is a CURATED sample, not a uniform-random blind draw from the full 290-topic pool Part 2 identified — the
6-seed replication across independently-sampled pools is real evidence of robustness, but a large uniform-
random blind sample is still the natural next-rung validation (queued as a next step below, not attempted here
given this task's scope).

## 6. Next steps (not this rung)

1. **Fix the `in_vocab_scope` lead-in loophole** (Part 2, logged to `FAILURE_LOG.md`) — score content hits only
   on the text after `open_ended_chat.extract_topic`'s own lead-in strip, not the raw message.
2. **Wire the already-measured V=4000 checkpoint** (`2026-08-31-wkv-mouth-rung4-vocab-coverage.md`) into this
   same generation path — directly attacks Part 1's ceiling (from 25.93%/40.48% toward whatever the wider
   vocabulary's fact-side coverage measures at, not yet computed for V=4000 specifically).
3. **A larger UNIFORM-RANDOM blind validation** of Part 3's before/after surfacing-rate claim, sampled from the
   full per-seed known-topic-and-groundable pools Part 2 already identified (hundreds of topics per seed, not
   the ~7-8 curated-by-relation-type examples this rung ran) — this specific runner cannot be pool-routed (see
   seed-waiver: `webapp/` and `bridges/` are both excluded from what `pool_provision.sh` ships to the mini-PC
   nodes), so a larger sample means either a longer local run (this rung's 6x7-8-example run took ~20-40s/seed;
   a few hundred examples/seed would scale roughly linearly, still CPU/numpy-only) or a small, scoped change to
   the pool's rsync allow-list — neither attempted here.
4. **A structural (not decode-time) grounding lever** — conditioning generation on a fact-bearing template or
   steering the recurrent state itself, to close the word-presence-vs-fact-assertion gap Part 3 names as the
   deeper residual.
