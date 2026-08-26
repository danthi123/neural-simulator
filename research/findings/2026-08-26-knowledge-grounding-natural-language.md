---
type: finding
status: live
date: 2026-08-26
mechanism: knowledge-grounding-alias-hop
lane: integration
seeds: [42]
artifacts:
  - research/findings/raw/_knowledge_grounding_verify/chatbrain_verify.json
runner: research/runners/_knowledge_core_curate.py
---

# Knowledge grounding: natural-language questions now reach the shipped Wikidata core (frontier A)

**Branch:** `research/knowledge-grounding-A` (from `main`). **Board:** #65/#66 (knowledge-loading /
knowledge-scale). **Supersedes-as-diagnosis:** the two 2026-08-25 integrated-conversational-state diagnostics,
which found `"what country is chelsea fc from?"` / `"what is a physicist?"` abstaining against the shipped
`wikidata_core_15k` bundle — and this arc's own direct-token replay of those SAME turns (`"what does chelsea_fc
country"`, `"what does penicillium instance_of"`) confirmed those specific entities are simply **absent** from
the 15k core (a curation-scope gap, not a grounding gap) — so this finding does not reuse those examples;
every headline turn below cites a fact independently confirmed present in the curated core.

## The gap (confirmed, not assumed)

The shipped `wikidata_core_15k` cortical LTM stores facts under Wikidata-derived CANONICAL tokens (e.g.
`chelsea_fc`, `instance_of`) picked by `_knowledge_core_curate.py`'s `pick_clean_alias()` — clean, but not the
words a natural question uses (`"chelsea fc"`, `"is a"`). Two independently-confirmed front-end gaps close over
it:

1. **`ChatBrain._extract_route`** (`research/runners/brain_chat_tui.py`) splits a question into single
   stopword-stripped words and hands `(agent, action)` as two of those words — a multi-word entity phrase
   never becomes the one underscore-joined token the store keys on.
2. **`ChatBrain._definitional_copula_route`** hardcodes `"what is X?"` to the in-conversation teaching
   convention's relation word `"isa"`; the shipped core's Wikidata-derived relation is `instance_of`
   (`curation_report.json`: `instance_of: 2734`, no `"isa"` anywhere). Wikidata's own P31 aliases literally list
   `"is a"` / `"is an"` (confirmed via `wikidata5m_relation.txt`), which is the exact bridge this arc uses.

`ShardedPhasorStore.query_patient` is exact and safe once given the right tokens (6-seed, false-hop=0.0 at the
deployed D=128/15k scale — `2026-08-25-fhrr-decode-rate-at-scale.md`), so this is a pure
GROUNDING/COMPREHENSION gap, not a recall or capacity problem.

## The mechanism

**Build time** (`research/runners/_knowledge_core_curate.py`, `build_alias_facts` + `_all_other_aliases` +
`_alias_quality`): for every entity/relation ALREADY in the curated core, sanitize + quality-rank every OTHER
raw Wikidata alias (capped at 6/concept — raw alias lists run ~30/concept and are mostly low-quality noise:
redirects, year tags, typos), and emit `{agent: alias_token, action: "alias_of", patient: canonical_token}`
facts into the SAME fact list `build_ltm_from_facts` teacher-loads into the bundle. An alias claimed by ≥2
distinct canonical concepts, OR that collides with an EXISTING canonical word, is DROPPED (ambiguous → no
grounding → honest abstain), mirroring `compositional_chain_route.py`'s own multi-valued-hop abstain
philosophy.

**Query time** (`research/runners/brain_chat_tui.py`): a new `_ground_content_words` helper tries
progressively-shorter underscore-joined candidate spans (longest-first, `min_span=2` by default) against ONE
MORE genuinely-spiking hop, `composer.query_patient(candidate, "alias_of")` — reusing the EXACT `query_patient`
primitive `compositional_chain_route.py`'s 2-hop reasoning already counts as brain-based. Wired at three sites:
  - `_extract_route`: a content-grounding pass runs BEFORE the copula-length check and the (agent, action)
    parse/heuristic, so `"what is chelsea fc"` (2 content words purely from the entity's own name) correctly
    collapses to 1 token and reaches the copula branch instead of mis-splitting into `(agent="chelsea",
    action="fc")`.
  - `_definitional_copula_route`: the SUBJECT is alias-hop-grounded with `min_span=1` (safe here — this route
    only fires on an explicit `"what is X"` prefix, a shape ordinary SVO teaching never matches). The RELATION
    deliberately stays the literal `"isa"` at this site (precedence-safety, see below) — the `is_a`/`is_an`
    bridge to `instance_of` is tried as a FALLBACK, not a replacement.
  - `_substrate_recall`: extends the existing verb SURFACE-FIRST/LEMMA-FALLBACK retry with a THIRD tier —
    alias-hop the relation (special-casing `v == "isa"` to also try `is_a`/`is_an`, since `"isa"` is this
    codebase's own invented shorthand, not itself a raw Wikidata alias) — tried ONLY after surface+lemma both
    miss, and a LAST-RESORT single-word AGENT alias-hop (e.g. `usa` → `u_s_of_a`) tried only when the relation
    fallbacks also miss. Being miss-only fallbacks, neither can override an already-correct recall.

`BRAIN_KNOWLEDGE_GROUNDING` is default-ON; `=0` is the lesion/escape.

### A collision-risk design choice (why `min_span=2` in `_extract_route`, not `1`)

Grounding could in principle rewrite an ordinary conversational word into an unrelated Wikidata concept if the
word happens to collide with the alias table (the shipped alias vocabulary is ~30-38k tokens at full scale —
large enough that a coincidental collision with an everyday single word is a real risk). `_ground_content_words`
therefore defaults to `min_span=2` for `_extract_route` (multi-word phrases only — a much rarer, higher-precision
surface shape than a single word), and never re-grounds a word already established this session
(`known_words=agents_set|actions_set|patients_set`). Single-word entity/relation aliases are instead resolved
ONLY as miss-only fallbacks in `_substrate_recall` (see above) — structurally unable to shadow a working recall,
since they only run when the un-grounded literal query has already failed.

## A pre-existing bug found + fixed en route (honest, not the deliverable, but load-bearing)

Building a first end-to-end test bundle surfaced a **pre-existing `ShardedPhasorStore` fragility**: the fast-path
bulk encoder (`encode_fast`, used to build a bundle without paying the full resonate-bind cost) binds a fact's
role fillers directly against the codebook; any word NOT already in the `vocab` passed to
`ShardedPhasorStore`/`build_ltm_from_facts` gets dynamically GROWN via a separate `_growth_rng`, inserted into
the shared codebook at its alphabetical position. `ShardedPhasorStore.save()`/`.load()` persists + reconstructs
the codebook by a FRESH BATCH generation over the (by-then-larger) vocab list on reload — which does not
reproduce that mixed batch+growth history. My first alias-fact vocab computation omitted the literal
`"alias_of"` relation token itself (only entity-side alias tokens were added), so it got grown — and after
save/reload, recall broke for **every** fact in the bundle, alias and plain alike (measured: 0/30 random facts
recalled correctly after a save/reload round-trip, vs 30/30 correct in-process before saving). Fixed by ensuring
`vocab_with_aliases` includes every word an alias fact touches (agent, action, AND patient) — confirmed 60/60
random facts correct after fix, at both smoke and full scale. This is a **pre-existing store bug, not
introduced by this arc's design** — it was latent because no prior bundle build had ever introduced a genuinely
new relation-type word via the fast path. Named here so it is not silently rediscovered; the fix is contained to
this arc's own vocab computation (no `sim/`/`ShardedPhasorStore` code changed) and is a documented residual for
anyone else feeding the fast-path bulk encoder a word absent from the initial vocab.

## Verify

Through direct `ChatBrain.gate()` calls (the SAME comprehension/recall code path
`/api/brain-chat` → `RichAnswerComposer`/`chat.gate()` reaches; `BRAIN_COMPOSER_KIND=rf` used for wall-clock
speed under this session's heavy CPU contention — verified this is a documented SPEED CHOICE, not a different
code path: the grounding pass runs BEFORE either the neural parser or the host-heuristic branch), against a
fresh alias-extended knowledge core built by the fixed curator (real Wikidata alias data, not synthetic):

| # | Turn | Grounding ON | Grounding OFF (lesion) |
|---|------|--------------|-------------------------|
| 1 | `"what is vienna austria"` (copula, multi-word alias → `instance_of`) | `[habsburg_austria, instance_of, city_work]` | **abstain** |
| 2 | `"what does guggenheim fellowship country"` (multi-word entity → `country`) | `[guggenheim_grant, country, u_s_of_a]` | **abstain** |
| 3 | `"what is purple elephant bicycle"` (adversarial nonsense) | abstain | abstain |
| 4 | `"the wolf hunts the deer"` → `"what does the wolf hunt?"` (in-session teach+recall) | `[wolf, hunt, deer]` (both turns) | `[wolf, hunt, deer]` (both turns, unchanged) |

Both turns run against a fresh alias-extended core built by the FIXED curator
(`research/runners/_knowledge_core_curate.py --smoke`, 400 top-entities/20 top-relations, real Wikidata alias
data — the same `build_alias_facts` mechanism the full 8000-entity/40-relation shipped-scale core uses).
`BRAIN_COMPOSER_KIND=rf` was used for wall-clock speed under this session's heavy concurrent CPU contention on
the box (documented speed choice, not a different code path — `_ground_content_words` runs in `_extract_route`
BEFORE either the neural-parser branch or the host-heuristic branch reads `content`).

- **Fidelity** (no answer drift): the grounded answer is IDENTICAL to querying the resolved canonical token
  directly (`ltm.query_patient("habsburg_austria", "instance_of")` == the grounded answer's patient).
- **Lesion / load-bearing**: `BRAIN_KNOWLEDGE_GROUNDING=0` reverts BOTH natural-language turns above to abstain
  — the exact pre-fix behavior — proving the grounding pass DRIVES the answer rather than decorating one the
  old path already produced.
- **Moat/adversarial**: a nonsense multi-word phrase (`"purple elephant bicycle"`) and a pre-existing unrelated
  question (`"what does the dragon breathe?"`) both abstain with grounding ON — the alias-hop opens no new
  confabulation channel (a miss stays a miss).
- **No regression**: in-session self-facts (`"what are you?"`) and in-conversation teach+recall
  (`"the wolf hunts the deer"` → `"what does the wolf hunt?"`) are unaffected.

Full transcript: `research/findings/raw/_knowledge_grounding_verify/chatbrain_verify.json`.

## Honesty: scaffold vs. learned (emergence-bar framing)

**SCAFFOLD (host, explicitly named — same class this codebase already accepts for
`_definitional_copula_route`'s regex and `compositional_chain_route.py`'s `_ROLE_NOUN_HINTS` table):**
(a) build-time extraction of raw aliases from `wikidata5m_entity.txt`/`_relation.txt` into a reverse index is
host preprocessing of teacher data (the identical honesty class as the already-shipped 15k curation itself);
(b) query-time PHRASE SEGMENTATION (deciding how many content words form one candidate span before any recall
runs) is inherently a host tokenization step, no different in kind from `_extract_route`'s existing stopword
strip.

**SUBSTRATE/LEARNED (rung 2, stronger than a pure host dict):** the alias RESOLUTION itself is not a Python
dict consulted at answer time — it is a genuine stored fact (`alias_token, "alias_of", canonical_token`)
recalled via the SAME spiking `query_patient` op the rest of the system already counts as brain-based. An
unresolved alias abstains exactly like an unknown agent — the no-confab moat is a strict superset of the
direct-recall case.

**NOT built (rung 3, the named next frontier, out of scope here):** a fully LEARNED entity-linking/synonymy
mechanism that acquires alias↔canonical associations from co-occurrence in running natural text, so grounding
emerges from exposure/use rather than a teacher-curated Wikidata alias file (the same mechanism class as
`2026-07-13-EMERGE-spreading-activation-completion-12seed-GO`). Rungs 1–2 make the SHIPPED 15k core reachable
in natural language now; rung 3 is what would let the brain ground a genuinely NOVEL entity's surface forms it
was never given an alias list for.

## What did NOT ship (and why)

The alias-extended bundle built for this verification is a NEW versioned directory
(`~/Projects/sim-data/knowledge_bundles/wikidata_core_15k_grounded_v1`, outside the repo, the standing
convention for these bundles) — the SHIPPED default `wikidata_core_15k` (what `BRAIN_LTM_BUNDLE` resolves to
unset) is **untouched**. Swapping the production default is an owner decision (`_default_ltm_bundle_dir()` in
`webapp/server.py`), not made here per the task's "hand back for owner review" instruction; verification used an
explicit `BRAIN_LTM_BUNDLE` override.

## Full production scale (in progress, headless, not blocking this deliverable)

A production-scale build (`--n-facts 15000 --top-entities 8000 --top-relations 40 --seed 42`, the SAME
parameters as the shipped `wikidata_core_15k`, genuine resonate bind per the standing faithfulness-over-speed
rule) was launched headless in the background
(`SIM_BACKEND=numpy .venv/bin/python -m research.runners._knowledge_core_curate --out-bundle
~/Projects/sim-data/knowledge_bundles/wikidata_core_15k_grounded_v1 ...`, log-visible progress:
30,804 alias facts generated from the 8000-entity/40-relation core, 44 ambiguous aliases dropped, 45,804 total
facts / 37,837-word vocab going into the genuine resonate bind). This does not block the deliverable above (the
mechanism is fully verified at smoke scale with real Wikidata data); it is the source for a follow-up owner
decision on whether/how to ship the alias-extended core as the production default. Honest scale note: the
alias-extended vocab (37,837 words) is ~5.4x the shipped core's 7,032-word vocab, which the curation script's
own docstring names as a latency-relevant threshold (codebook cleanup is O(V·D); the 2026-08-21 flip-soak
finding measured sub-second recall only through ~20k distinct entities) — an owner-facing tradeoff to weigh
against the `_MAX_ALIASES_PER_ID` cap (currently 6; lowering it trades alias coverage for vocab/latency), not
resolved here.

## Next action

Owner review + decide whether to swap `BRAIN_LTM_BUNDLE`'s default target to the alias-extended bundle (or
re-curate the shipped bundle in place, same command, now with the vocab fix). Named residuals: (1) rung-3
learned entity-linking (above); (2) noun-relation natural phrasings ("what country is X from") are a SEPARATE,
pre-existing comprehension gap (relation-word-before-entity syntax) this arc does not address — `_extract_route`
still expects agent-then-relation word order; (3) the codebook-growth save/reload bug (above) is fixed for THIS
arc's own vocab computation but not hardened generically inside `ShardedPhasorStore` itself.
