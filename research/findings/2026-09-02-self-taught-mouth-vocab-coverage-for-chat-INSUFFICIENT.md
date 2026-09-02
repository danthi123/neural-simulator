---
status: measurement-complete
lane: e-mouth-fluency / board #99, #112, #199 (open-ended-mouth wire-in gate, question 1 of 2)
type: finding
date: 2026-09-02
seed-waiver: measurement-only (a fixed-checkpoint vocabulary coverage profile, not a stochastic capability
  GO/NO-GO -- there is nothing to seed: the V=1000 vocabulary is the SAME 999-word set on every seeded
  checkpoint `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz` ships (frequency-rank-ordered, trained
  independently per seed but over the identical TinyStories corpus -- verified for seed 42 here; a 6-seed
  repeat would measure the same vocabulary set, not new information)
instrument: direct tokenization + real-code coverage measurement -- `webapp.wkv_mouth_generator._get_readout`
  (the actual production vocab loader) and `webapp.wkv_mouth_generator.in_vocab_scope` (the actual, CURRENT
  production accept/reject gate, imported and called unmodified -- never re-implemented) against a probe
  corpus of genuine chat-register utterances and real chat topics
runner: research/runners/_wkv_mouth_chat_topic_vocab_coverage_derisk.py
artifacts:
  - research/findings/raw/_wkv_mouth_chat_topic_vocab_coverage.json (V=1000, the production-default
    checkpoint, seed 42, learned-head ON per the current default)
  - research/findings/raw/_wkv_mouth_chat_topic_vocab_coverage_v4000.json (V=4000 comparison checkpoint,
    same probe corpus, same code path, `BRAIN_WKV_MOUTH_CKPT` override)
  - research/findings/2026-08-31-wkv-mouth-rung4-vocab-coverage.md (prior measurement -- TinyStories
    TRAINING-corpus text only, not a chat-topic probe; this finding is the genuinely different measurement)
  - research/findings/2026-09-01-wkv-mouth-fact-grounding-lever.md (prior measurement -- Wikidata FACT
    triples specifically, not general chat register; this finding's group 3 extends that with a larger
    seeded sample and adds groups 1-2, which that finding did not measure)
  - webapp/wkv_mouth_generator.py (the vocab loader + the live `in_vocab_scope` gate, unmodified)
---

# The self-taught WKV mouth's vocabulary against TYPICAL CHAT TOPICS — measured INSUFFICIENT, precisely quantified

## 0. The question, and why the two prior measurements do not answer it

Board #99/#112/#199 name wiring the from-scratch WKV/SSM spiking mouth into the `BRAIN_OPEN_ENDED` free-talk
channel (replacing Qwen there) as gated on two open questions. This finding answers the FIRST: **does this
checkpoint's own vocabulary cover typical chat topics?** (The second — whether the recurrence/memory was
trained end-to-end by a genuinely local brain-like rule — is a separate task, not attempted here.)

Two measurements already exist and are reused, not repeated, but **neither is a "typical chat" probe**:

- `2026-08-31-wkv-mouth-rung4-vocab-coverage.md` measured coverage against the checkpoint's OWN TinyStories
  TRAINING corpus text — self-referential by construction (a model trivially covers a large fraction of the
  text it was trained on; this cannot tell us whether it covers what a *user* would say).
- `2026-09-01-wkv-mouth-fact-grounding-lever.md` measured coverage against Wikidata FACT triples specifically
  (the `(agent, action, patient)` structure the knowledge store uses internally) — the right question for
  "can it state a fact," but narrower than "typical chat topics," and it never touched ordinary
  conversational REGISTER (greetings, small talk, opinions, humor) at all.

This finding assembles a genuinely different probe: real conversational-register utterances plus real
topic-style queries (both everyday-famous and from the live production knowledge store), all reused
verbatim from material already in this repo, and measures coverage with the CURRENT production code path —
not a re-implementation of the gate (the 2026-08-31 measurement's own script re-implemented an OLDER version
of `in_vocab_scope` that predates the 2026-09-01 `_LEADIN_WORDS` fix; this runner imports the live function
instead, so its numbers reflect what today's code actually does).

## 1. The vocabulary itself

Production default: `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz`, loaded via
`webapp.wkv_mouth_generator._get_readout(seed)` (pure `np.load`, no `SimulationBridge`, no RNG effect).

- **Size**: 1000 vocabulary slots = **999 real words + 1 `<unk>` sentinel** (verified directly:
  `words[-1] == "<unk>"`, exactly one such sentinel, no other special tokens).
- **Tokenization**: **word-level, closed vocabulary.** `WKVReadout.words` is a flat list of whole words with
  no subword/BPE merge table anywhere in the checkpoint (`research/runners/_wkv_fewspike_read_derisk.py`'s
  `WKVReadout.__init__` reads `W["words"]` directly as the ID→string table). A word not in this exact list
  cannot be partially expressed via subword pieces — it is simply absent, full stop, by construction.
  Frequency-rank-ordered: the top-40 entries are dominated by English function words ("the, and, a, to,
  was, they, he, it, ..."), consistent with training-corpus frequency statistics, not curated for topical
  breadth.
- A wider checkpoint also ships (`bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz`, V=4000, same
  word-level/closed-vocab design) but is **not the production default** — `BRAIN_WKV_MOUTH_CKPT` must be
  overridden to reach it. Used below only as a comparison point for the "next step" quantification.

## 2. The probe corpus — three groups, all reused verbatim from material already in this repo

| Group | n | Source | What it represents |
|---|---|---|---|
| `conversational_register` | 14 | `research/runners/_conversation_turing_test_derisk.HUMAN_TURNS` (the project's own Turing-style stress-test battery: greeting, small talk, in-domain entry, emotion, forward-model curiosity, referential/episodic, out-of-domain fact, arithmetic, self/experiential, humor, abstract opinion, meta self-awareness, social closing) | topic-agnostic everyday chat REGISTER |
| `everyday_real_world_topics` | 10 | `research/runners/_open_ended_state_driven_generation_derisk._QWEN_KNOWN_STORE_UNKNOWN` (paris, python, shakespeare, coffee, jupiter, beethoven, tokyo, everest, photosynthesis, gravity), each wrapped as `"Tell me about X."` (a lead-in `webapp/open_ended_chat.py::_LEADINS` already accepts) | famous, ordinary-conversation TOPICS a real user names |
| `wikidata_known_agents` | 100 | a seeded sample (seed=42) of real agents from the LIVE production knowledge-store bundle `wikidata_core_15k` (its `facts.json`, resolved the SAME way `webapp.open_ended_chat.build_index` resolves it — see `_bundle_dir`/`_sample_known_topics`'s own candidate-root search), drawn with `research/runners/_open_ended_bundle_moat_safety_soak._sample_known_topics` (reused unchanged), each slug rendered as a natural noun phrase via `research/runners/_wkv_fact_to_sentence_lexicon_lever.slug_to_np` and wrapped as `"Tell me about X."` | topics the brain's own knowledge store actually holds |

**124 probe utterances total.** Nothing here was invented or downloaded — every list and template is an
existing, already-committed project artifact; the knowledge-store bundle lives in the project's local data
lake outside this git repo (the SAME source `_wkv_mouth_fact_grounding_derisk.py` already reads from, not a
network fetch) — its resolved directory is recorded in this task's own raw artifact rather than hardcoded
into this prose (see `research/findings/raw/_wkv_mouth_chat_topic_vocab_coverage.json`'s
`groups.wikidata_known_agents.rows[*].subtype` for the exact 100 sampled agent slugs).

## 3. Measurement, at the production-default V=1000 checkpoint (seed 42)

Two independent metrics, both computed directly from the CURRENT `webapp/wkv_mouth_generator.py` code:

- **content-word OOV rate** — tokenize with `WKV._WORD_RE`, exclude `_FUNCTION_WORDS` (the same set the
  codebase's own `fact_grounding_ids`/`_content_hits` already use as "content"), check literal membership
  in the checkpoint's vocab set.
- **`gate_pass`** — the live `in_vocab_scope(text, seed=42)` call, unmodified: the actual accept/reject
  decision `answer_turn` makes today (True → routes to the WKV mouth; False → falls back to Qwen).

| Group | n | content-word OOV | fully-in-vocab (all content words present) | `in_vocab_scope` gate PASS |
|---|---|---|---|---|
| `conversational_register` | 14 | 25.0% (15/60) | 42.9% (6/14) | **78.6% (11/14)** |
| `everyday_real_world_topics` | 10 | 25.0% (10/40) | 0.0% (0/10) | **0.0% (0/10)** |
| `wikidata_known_agents` | 100 | 42.6% (233/547) | 1.0% (1/100) | **2.0% (2/100)** |
| **Overall** | **124** | **39.9% (258/647)** | **5.65% (7/124)** | **10.48% (13/124)** |

**The registers diverge sharply.** Generic small-talk phrasing (`conversational_register`) fares reasonably
— most such turns are plain, simple sentences close to TinyStories' own register, and 11/14 clear the
production gate. But the moment a chat turn NAMES A TOPIC — which is what "chat topic coverage" is actually
asking about — the gate collapses: **0% and 2%** pass for the two topic-naming groups. Two concrete gate
failures from `conversational_register` itself: *"What's the capital of France?"* (`capital`, `france`,
`what's` all OOV) and *"Do you understand that you are a simulated brain, not a person?"* (`brain`,
`person`, `simulated` all OOV) — ordinary adult-register words, not obscure ones.

**Why `everyday_real_world_topics` reads exactly 0%, precisely, not approximately.** Each probe here is
`"Tell me about X."` where X is the ONE content word after the lead-in strip (`in_vocab_scope`'s own
`_LEADIN_WORDS` correctly excludes "tell/me/about" from counting as content, closing the exact loophole the
2026-09-01 finding logged). All 10 topic words (paris, python, shakespeare, coffee, jupiter, beethoven,
tokyo, everest, photosynthesis, gravity) are OOV, so `min_content_hits=2` can never be met — the gate
correctly, deterministically refuses every one of these ten ordinary "ask about a famous thing" queries.

**`wikidata_known_agents` — concrete examples.** Of 100 sampled real store topics, only 2 pass the gate
(`"Tell me about Lake Fly."`, `"Tell me about Park Barn Estate."` — and even the second still has one OOV
word, `estate`). Representative failures: `"Tell me about Ac Le Havre."` (ac/le/havre all OOV — French club
name), `"Tell me about Aomori Prefecture."` (aomori/prefecture OOV), `"Tell me about Art Students League."`
(art/students/league OOV), `"Tell me about Baltimore Colts 1947 50."` (baltimore/colts OOV). The 233 missing
content-word instances (214 distinct words) are dominated by two buckets, both clearly named, neither a
surprise given the checkpoint's TinyStories provenance:
  1. **Geographic/institutional proper nouns** — city/country/region names (tokyo, osaka, kansas, helsinki,
     brussels, ukraine, portugal, aomori, prefecture...) and organization vocabulary (college×5, football×4,
     university, team, national, olympic, league, club, records×4, history×4, fc/ac/bsc/ec abbreviations).
  2. **Person names** — beethoven, shakespeare, harper, coleman, livingstone, and similar.
  A minor (~6%, 14/233) share of the OOV instances are single-letter club-name abbreviation fragments
  (`f`, `c`, `ac`, `fc` from slugs like `manchester_united_f_c`) — an artifact of literal underscore-split
  rendering, not a meaningful vocabulary gap on its own, but it does not change the overall picture: even
  discounting it entirely, content-word OOV for this group stays above 41%.

## 4. Comparison: the already-measured wider checkpoint (V=4000), same probe corpus, same code path

The V=4000 checkpoint (`bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz`) is not new — it was already
identified in `2026-08-31-wkv-mouth-rung4-vocab-coverage.md` as covering more of the TinyStories corpus text
— but it had never been measured against a genuine chat-topic probe before. Same 124 utterances, same
`in_vocab_scope` gate, `BRAIN_WKV_MOUTH_CKPT` pointed at the V=4000 file (learned-head disabled — that
override is V=1000/D=128-shaped and fails safe on a shape mismatch, irrelevant to vocabulary):

| Group | content-word OOV (V=1000 → V=4000) | gate PASS (V=1000 → V=4000) |
|---|---|---|
| `conversational_register` | 25.0% → 10.0% | 78.6% → 85.7% |
| `everyday_real_world_topics` | 25.0% → 22.5% | 0.0% → 0.0% |
| `wikidata_known_agents` | 42.6% → 35.8% | 2.0% → 10.0% |
| **Overall** | **39.9% → 32.6%** | **10.48% → 17.74%** |

A real, measurable improvement (content OOV drops ~7 points, gate-pass nearly doubles) — but it is a partial
mitigation, not a fix: `wikidata_known_agents` gate-pass only reaches 10%, and `everyday_real_world_topics`
stays at exactly 0% (single-content-word famous-entity queries are structurally unaffected by vocab size
alone when the specific proper noun is still absent — 9/10 of those exact ten words are still OOV even at
V=4000).

## 5. Verdict

**INSUFFICIENT** for typical chat, against the concrete bar this finding states up front: a chat-topic query
should clear the ACTUAL production accept gate (`in_vocab_scope`) at a rate that would make the WKV mouth a
credible general replacement for Qwen in free-talk, not just an occasional lucky hit. **10.48% overall
gate-pass (13/124)**, with the two topic-naming groups (the part of "typical chat topics" that is literally
about naming a topic) at **0% and 2%**, is far below that bar by any reasonable reading. Content-word OOV of
**39.9% overall** (25.9-40.5% was the FACT-side ceiling the 2026-09-01 finding already found; this
REGISTER-and-topic-side measurement is consistent with, not contradicted by, that number) confirms this is
not a measurement artifact of one particular query template — the underlying word coverage itself is the
constraint.

**The one honest bright spot, stated precisely so it is not lost:** topic-agnostic small talk
(`conversational_register`, 78.6% gate-pass) is meaningfully better covered than topic-specific chat. If the
open-ended channel's actual traffic mix skews toward register (greetings, opinions, chit-chat) rather than
named-entity lookups, the practical impact of this gap is smaller than the blended 10.48% number alone
suggests — but board #112's own stated value proposition for this mouth (per the fact-grounding finding) is
specifically KNOWN-TOPIC recall, which is exactly the register this measurement finds weakest.

## 6. The precise residual, and the concrete next step (a wall defers a METHOD, not the capability)

**What's missing, precisely:** not obscure jargon — ordinary geographic/institutional proper nouns (city,
country, and organization names: tokyo, kansas, university, football, college, records, history...), person
names (beethoven, shakespeare, harper...), and a handful of everyday abstract/adult-register words even
outside the topic-naming groups (capital, brain, person, simulated, interesting, mentioned, lately). This is
the direct fingerprint of a checkpoint trained on TinyStories (a children's-story corpus about dogs, cats,
apples, and forests) being asked to name real-world topics — an expected, not surprising, mismatch, but one
this finding now quantifies exactly rather than assuming.

**Named next steps, not a stopping point:**
1. **Wire the already-measured, already-partially-verified V=4000 checkpoint** — Section 4 shows it roughly
   doubles gate-pass (10.48% → 17.74%) with zero new training; still not sufficient alone, but the correct
   first lever given it already exists on disk and was never previously checked against a chat-topic probe.
2. **A genuinely wider-vocabulary or subword-capable checkpoint** — this checkpoint's word-level, closed
   design means an OOV word can NEVER be partially expressed; a subword/BPE tokenizer would let it compose
   novel proper nouns and domain terms from familiar pieces instead of requiring every whole word to have
   been seen at training time. Not attempted here — a training-time, not decode-time, change.
3. **Tail-learning / vocabulary extension on the live knowledge store's own vocabulary** — the store's own
   932 real agent names (and their relation/patient vocabulary) are known in advance; a targeted
   fine-tune/extension pass over exactly that vocabulary (rather than a generic wider corpus) would directly
   target the `wikidata_known_agents` group's 42.6% OOV, the largest and most load-bearing gap measured here.
4. **Morphology-aware matching** (plurals, simple derivational forms) was NOT the driver of any failure
   measured here — spot-checking the OOV lists found no case where a missing word was a simple morphological
   variant of an in-vocab word — so this lever is named for completeness, not because this measurement found
   evidence it would help.

## 7. What this is not

Not a verdict on the recurrence/memory training-provenance question (board's second open question) — untouched
here. Not a claim that the WKV mouth is unusable — `conversational_register`'s 78.6% shows real, useful
coverage for generic chat register. Not a re-measurement of the fact-side ceiling (`2026-09-01`'s 25.9-40.5%
stands, corroborated rather than superseded by this finding's independent 39.9% register+topic-side number).
Not a 6-seed capability GO/NO-GO — see `seed-waiver` above for why that does not apply to a fixed-checkpoint
vocabulary-membership measurement.
