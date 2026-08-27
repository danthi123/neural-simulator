---
type: finding
status: live
date: 2026-08-27
mechanism: sharded-ltm-key-routing-alias-fallback
lane: E-language / knowledge-in-chat
artifacts:
  - research/findings/raw/_ltm_key_routing_alias_fix/verify_store_level.py
  - research/findings/raw/_ltm_key_routing_alias_fix/verify_e2e.py
  - research/findings/raw/_ltm_key_routing_alias_fix/verify_moat_synthetic.py
  - research/findings/raw/_ltm_key_routing_alias_fix/build_verdict.py
  - research/findings/raw/_ltm_key_routing_alias_fix/verify_verdict.json
runner: research/findings/raw/_ltm_key_routing_alias_fix/build_verdict.py (+verify_store_level.py, +verify_e2e.py, +verify_moat_synthetic.py)
---

# The `_portal`/`_core` LTM key-routing residual is closed: a bare surface form now resolves to its suffix-keyed entity, additively, moat intact

**One-line:** `research/runners/sharded_phasor_store.py`'s `ShardedPhasorStore` gained a lazily-built,
retrieval-time-only alias fallback (`build_alias_index` / `_resolve_alias`, wired into `query_patient`,
`ask_yes_no`, `render_fact`, `query_agent`): on a direct-key miss, it strips a known curation-artifact suffix
(`_portal`, `_core`) from the query key and retries once. A user typing the bare name (`"canada"`) now retrieves
the fact the shipped bundle stored under `"canada_portal"`. No bundle rebuild, no key rename, no change to any
already-working lookup.

## The residual this closes

Named as a follow-up in `2026-08-27-ltm-exempt-production-flip-knowledge-answers-live-by-default.md`'s "Honest
residual" section: the shipped `wikidata_core_15k` bundle keys 11 entities as `<name>_portal` / `<name>_core`
(`ballet, berlin, brandenburg, cambodia, canada, comic, dorset, lgbt, portugal, schleswig_holstein` -> `_portal`;
`ska` -> `_core`) rather than the bare `<name>` a user actually types, so a genuinely-held fact could still miss
on retrieval even with the GNW-consensus LTM-exemption flip live -- independent of that flip, and independent of
the comprehension/consensus fixes in `2026-08-27-knowledge-in-live-chat-veto-...md`.

## Root cause (WHY the suffixing happens -- a curation-time picker bug, not a store bug)

`research/runners/_knowledge_core_curate.py`'s `pick_clean_alias` scores each raw Wikidata alias for an entity
and keeps the highest scorer as the canonical token: `+2` for a 2-4 word phrase, `+1.5` for a capitalized first
letter, `-0.15*i` for lateness in the alias list. Wikidata's crowd-sourced alias dump for several entities
includes Wikipedia-**namespace** artifacts alongside the genuine name -- confirmed directly against
`wikidata5m_entity.txt`: entity **Q16 (Canada)**'s alias line contains both `"Canada"` (1 word) and `"Canada
portal"` (2 words, capitalized -- from the `Portal:Canada` Wikipedia page). Under the scoring rule, `"Canada
portal"` scores `2 + 1.5 = 3.5` against `"Canada"`'s `1.5` (no multi-word bonus), so the picker keeps the
*wrong* alias and `sanitize()` turns it into `"canada_portal"`. `_CRUFT` already rejects `"wikiproject"`,
`"template"`, `"category"`, etc. but not `"portal"` -- so this specific Wikipedia-namespace class slipped
through. The deep fix is re-curating (add a portal/namespace reject to `_CRUFT`, rebuild the bundle -- ~13
CPU-min for 15k facts); this finding is the additive retrieval-time mitigation, not that rebuild.

## The fix (additive, retrieval-time only)

`ShardedPhasorStore` (research/runners/sharded_phasor_store.py):
- `_KNOWN_KEY_SUFFIXES = ("_portal", "_core")`, `_strip_known_suffix(key)`.
- `build_alias_index()`: lazily scans every fact currently in the store for agent/patient strings ending in a
  known suffix, and maps `bare -> suffixed_key` **only when**:
  1. the bare form is **not itself already a stored key** (a real, distinctly-keyed entity is never shadowed);
  2. the suffix-strip is **unambiguous** (exactly one distinct stored key strips to that bare form -- an
     ambiguous case resolves to nothing rather than guessing).
- `query_patient`, `ask_yes_no`, `render_fact`, `query_agent` each try the literal key **first**, unchanged
  from before this change, for every already-working lookup; **only on a miss** do they consult the alias index
  and retry once with the resolved key. No environment flag was added -- per the task's own guidance, this is a
  pure fallback that only ever fires when the literal lookup already failed AND a resolvable alias exists, so
  default-on carries no regression risk to any existing behavior (verified below, not just argued).

## Verification (through the REAL retrieval path, unmocked, the shipped bundle)

**Store level** (`verify_store_level.py`, `SIM_BACKEND=numpy`, the actual `wikidata_core_15k` bundle,
`ShardedPhasorStore.load` + `query_patient`):

| check | result |
|---|---|
| `query_patient('canada', 'member_of')` | `'united_nations'` -- bare form now retrieves |
| `query_patient('berlin', 'country')` | `'federal_republic_of_germany'` -- bare form now retrieves |
| `query_patient('dorset', 'country')` | `'united_kingom'` -- bare form now retrieves |
| `query_patient('ska', 'instance_of')` (`_core` suffix) | `'genre_of_music'` -- bare form now retrieves |
| `query_patient('chelsea_fc', 'country')` | `'united_kingom'` -- **unchanged** (never touches the alias path) |
| `query_patient('definitely_not_real_xyz', 'country')` | `None` -- **moat holds**: nonexistent entity abstains |
| `query_patient('canada', 'definitely_not_a_real_relation_xyz')` | `None` -- moat holds: real entity, fake relation, abstains |
| `query_patient('canada_portal', 'member_of')` | `'united_nations'` -- direct suffixed-key lookup unaffected |
| `build_alias_index()` size | 11 (exactly the 11 known-suffixed entities; every one uniquely resolves) |

**Full live-chat pipeline** (`verify_e2e.py`, `ChatBrain.gate` -- comprehension `_relation_fronted_route` ->
`_substrate_recall` -> `TieredFactStore` -> `ShardedPhasorStore`, the exact call graph
`webapp/server.py`'s `/api/brain-chat` uses):

| question | `chat.gate(...)` | |
|---|---|---|
| `"what country is berlin in"` | `['berlin', 'country', 'federal_republic_of_germany']` | the canada-type case now COMMITS in chat |
| `"what country is chelsea fc from"` | `['chelsea_fc', 'country', 'united_kingom']` | unaffected (regression check, matches the 2026-08-27 veto finding) |
| `"what country is definitely not real xyz in"` | `None` | abstain -- moat holds end-to-end |

**Adversarial synthetic moat checks** (`verify_moat_synthetic.py`, cases NOT present in the real bundle, to
prove the two safety invariants hold in general):
- **Ambiguous strip**: a synthetic store holding both `foo_portal` and `foo_core` (both strip to `foo`) ->
  `build_alias_index()` does NOT map `foo` -> anything; `query_patient('foo', ...)` returns `None`. An
  ambiguous bare form abstains rather than guessing which of two entities the user meant.
- **Shadowing**: a synthetic store holding both a real `bar` entity and a `bar_portal` entity -> `build_alias_index()`
  does NOT map `bar` (it's already a real key); `query_patient('bar', 'rel2')` returns **`bar`'s own fact**
  (`'z'`), never `bar_portal`'s (`'y'`). A real, distinctly-keyed entity is never shadowed by a same-bare-form
  suffixed one.

All 15 checks above are also registered as `tools.verdict.Verdict` preconditions
(`research/findings/raw/_ltm_key_routing_alias_fix/build_verdict.py`) and re-measured live against the real
store + the full `chat.gate()` pipeline in one run: **`status: GO`**, 15/15 preconditions `ok: true`, zero
unmeasured, zero unmet -- `research/findings/raw/_ltm_key_routing_alias_fix/verify_verdict.json`.

## Scope / what this does not touch

`query_chain` / `chain_of_thought`'s multi-hop reasoning reuse `query_patient` internally for each hop, so a
chain STARTING or PASSING THROUGH a suffix-keyed concept as a mid-chain cue benefits transparently; the
self-cued `chain_of_thought`'s own relation-selection step (`shard_for(x)._relation_assoc()`) still routes on
the literal `x` and was not touched (a self-generated-thought edge case, not the live-chat retrieval path this
residual was about). `TieredFactStore` and `RFPhasorComposer` were not modified -- this is scoped entirely to
`ShardedPhasorStore`, the LTM tier's own class.

Functional read-outs only; no phenomenal-experience claim.
