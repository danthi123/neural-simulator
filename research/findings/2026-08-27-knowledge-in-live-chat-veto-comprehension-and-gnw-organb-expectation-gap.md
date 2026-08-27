---
type: finding
status: live
date: 2026-08-27
mechanism: knowledge-in-live-chat-wrongful-veto — comprehension word-order bug (FIXED) + GNW two-organ bus
  organ-B expectation-registry blind spot on the LTM tier (CHARACTERIZED, owner review)
lane: integration
seeds: [42]
artifacts:
  - research/findings/raw/_knowledge_chat_veto/trace_evidence.json
  - research/findings/raw/_knowledge_chat_veto/repro_veto.py
  - research/findings/raw/_knowledge_chat_veto/verify_fix.txt
  - research/findings/raw/_knowledge_chat_veto/byte_identical_check.txt
  - research/findings/raw/_knowledge_chat_veto/full_pipeline_trace.txt
  - research/findings/raw/_knowledge_chat_veto/bus_trace.txt
  - research/findings/raw/_knowledge_chat_veto/full_trace2.txt
runner: research/findings/raw/_knowledge_chat_veto/ (light-path repro scripts, SIM_BACKEND=numpy)
---

# Vikunja #142: "what country is chelsea fc from" -> "I don't know" — TWO independent vetoes traced on the light path; the comprehension one is FIXED, the GNW two-organ-bus one is CHARACTERIZED for owner review

**Bottom line.** The live-chat pipeline has **two sequential, independent bugs** that both must clear before a
plain-English knowledge-core question answers. **Bug 1 (comprehension, `_extract_route`) is FIXED, guarded,
byte-identical-off, and verified.** **Bug 2 (the GNW two-organ consensus bus's organ B) is a genuine architecture
gap — its "expectation" registry structurally EXCLUDES the entire 15,000-fact Wikidata LTM tier, so the
DEFAULT-ON two-organ bus vetoes EVERY knowledge-core recall, not just relation-fronted ones — characterized here
with full evidence, NOT fixed (a judgment call about the anti-crosstalk moat's scope, per the task's explicit
instruction not to weaken honesty unilaterally).** A third, narrower defect in the SAME "reverse-binding verify"
family (`gnw_bus_shadow.py` organ C) was also found while tracing, and a fourth, unrelated latent bug (the
three-organ bus never actually installs despite its own comments claiming default-on) surfaced along the way —
both noted below and in `research/FAILURE_LOG.md`, neither on the critical path for this bug.

## Research-first (what was read before touching code)

- `research/findings/2026-08-11-PRODUCTION-chat-pipeline-is-largely-HOST-not-one-brain-spiking-code-traced-honest-inventory.md`
  — the 2026-08-11 code-trace layer map (now dated: the live pipeline has grown a GNW N-organ bus, a two-organ
  bus, affect/mood, topic-tracking swap-drives, value-choice, and multistep deliberation since then — all
  confirmed present in `webapp/server.py`'s `/api/brain-chat` handler as of this session).
- The task named `research/findings/2026-08-25-reasoning-frontier-chain-routing.md` (task-3) — **this exact
  filename does not exist** on `main`, `research/reasoning-frontier`, `research/reasoning-frontier-hardened`, or
  `research/reasoning-route-decode-rate-measurement` (checked via `git ls-tree` on all four). The closest matches
  (confirmed via the RAG index, top hits) are `research/findings/2026-08-25-reasoning-route-hardened.md` +
  `research/findings/2026-08-25-reasoning-route-moat-audit-hardening-spec.md` (compositional CHAIN-hop routing,
  e.g. "what does the wolf's prey eat" — a different bug class from this single-fact veto) and
  `research/findings/2026-08-25-integrated-conversational-state-diagnostic.md`. None of these documents the
  chelsea-fc-style single-fact veto directly; this finding is the first trace of it.
- `research/reasoning-frontier` / `research/reasoning-frontier-hardened` / `research/reasoning-route-decode-rate-measurement`
  branches checked for existing WIP on this exact veto (`git log`, `git diff --stat` vs `main`): none found —
  their commits are the chain-routing hardening + DA-axis cupy-interop fix, unrelated to this bug.
- RAG (`tools/rag/rag_search.py`, corpus=finding): `"live chat pipeline vetoes correct knowledge answer I don't
  know"` and `"consensus multi-judge abstain vetoes correct answer mood topic tracking"` — top non-generic hits
  were the 2026-08-11 pipeline map and the reasoning-route-hardened family above; nothing pointed at this
  specific comprehension/organ-B mechanism, confirming this is new ground.
- `webapp/server.py`'s `/api/brain-chat` handler (~4007-4900) read in full to map the ACTUAL current layer
  stack: knowledge attach (`_build_chat_brain`'s `tiny-demo +LTM` via `TieredFactStore`) -> GNW N-organ bus
  (`gnw_bus_shadow`) -> GNW two-organ bus (`gnw_two_organ_bus`, DEFAULT-ON) -> GNW three-organ bus
  (`gnw_three_organ_bus`, default-on per its own comment) -> GNW confidence/conflict deliberation
  (`gnw_deliberation`) -> value-driven choice (`value_choice_production_organ`) -> GNW multistep deliberation
  (`gnw_multistep_deliberation`) -> topic-tracking swap-drives (`swap_drives_chat`) -> affect/mood
  (`affect_production_organ`) -> render.

## Reproduction (light path, per the hard constraint: `SIM_BACKEND=numpy`, `tiny-demo` + the shipped LTM, NOT the
GPU server)

Confirmed the shipped `wikidata_core_15k` bundle (`~/Projects/sim-data/knowledge_bundles/wikidata_core_15k`)
holds `{"agent": "chelsea_fc", "action": "country", "patient": "united_kingom"}` (a curation typo in the data,
"kingom" not "kingdom" — irrelevant to this bug, noted for completeness). Built the exact same `tiny-demo +LTM`
ChatBrain `webapp/server.py::_build_chat_brain` builds (`_build_tiny_demo(..., composer_kind="onebrain")` +
`TieredFactStore(buffer, ShardedPhasorStore.load(LTM_BUNDLE))`), on `SIM_BACKEND=numpy`.

**STAGE 0 (direct store query, bypassing every live-turn layer):**
`composer.query_patient('chelsea_fc', 'country') = 'united_kingom'` — correct, confirming the store holds and
can answer the fact in isolation exactly as the bug report says.

## Bug 1 (FIXED): the comprehension layer swaps agent/action on a relation-fronted question

**STAGE 1 (pre-fix):** `chat._extract_route('what country is chelsea fc from')` returned `['country',
'chelsea']` — **backwards**. **STAGE 2:** `chat._substrate_recall(...)` -> `'__ABSTAIN__'` (honestly abstains on
the nonexistent `what_does('country', 'chelsea')` binding). **STAGE 3:** `chat.gate(...)` -> `None` — the exact
"I don't know about that." reproduced on the light path.

**Root cause.** `ChatBrain._extract_route` (`research/runners/brain_chat_tui.py`) tokenizes the question,
strips stopwords, and — for a >=2-content-word question with the on-brain parser present — feeds
`padded = [content[0], content[1], "__q__"]` into `_neural_question_parse`, which assigns POSITION 0 to the
AGENT role and POSITION 1 to the ACTION role (`BrainConversationalAgent.role_of` is purely positional, trained
once at build time on the SVO order "the wolf hunts the deer"). This is correct for the in-conversation
teaching/recall shape ("what does the wolf hunt?" -> entity first, relation second) but is **backwards** for a
Wikidata-style relation-fronted question ("what country is chelsea fc from?" -> relation noun first, entity
second): `content = ['country', 'chelsea', 'fc', 'from']` (note: `_extract_route`'s own LOCAL stopword set is
missing "from", unlike the module-level `QuestionRouter._STOP`, so "from" survives into content — a second,
minor asymmetry that does not change the outcome once the fix below runs on the raw question). Grounding
(`_ground_content_words`) also failed to collapse "chelsea"+"fc" -> "chelsea_fc" here because the entity's own
canonical token was not registered as an alias-of-itself in the curated alias table — a secondary, latent gap
that the fix below sidesteps by grounding on its OWN extracted entity span rather than the generically-stripped
content list.

**Fix.** Added `ChatBrain._relation_fronted_route` (`research/runners/brain_chat_tui.py`), a new comprehension
route mirroring the existing `_definitional_copula_route` pattern: a regex `^what\s+(?P<relation>[a-z]+)\s+
(?:is|are|was|were)\s+(?P<entity>.+?)\s*\??\s*$` matches ONLY the relation-fronted shape (deliberately excludes
`does/do/did`, so it can never fire on the already-working "what does X verb" shape), strips a trailing
preposition ("from"/"in"/"of"/...), grounds the entity phrase (alias hop, falling back to the store's own
underscore-join convention — `"chelsea fc" -> "chelsea_fc"` needs zero alias facts since that phrase already
equals the canonical token), and returns `[entity, relation]` in the CORRECT order. Wired into `_extract_route`
right before the existing `len(content) <= 1` copula check, so it runs on every question but only ever matches
this one syntactic shape. Guarded by `BRAIN_RELATION_FRONTED_QUESTIONS` (default ON; `=0` is the byte-identical
lesion/escape — production convention, matches `_knowledge_grounding_enabled()`'s own shape).

Because `_extract_route` is the SHARED comprehension chokepoint every combination layer calls
(`ChatBrain.gate`'s own `_substrate_recall`, AND `ChatBrain.gate_extract`, which `gnw_bus_shadow.gate_via_bus`
/ `gnw_two_organ_bus.two_organ_gate_via` / `gnw_three_organ_bus` all call for their own (agent, action)
extraction) — this ONE fix reaches every routing layer, not just the plain host path.

**Verification (light path, `SIM_BACKEND=numpy`, tiny-demo+LTM, unwrapped `chat.gate`):**

| check | result |
|---|---|
| `chat.gate('what country is chelsea fc from')` | `['chelsea_fc', 'country', 'united_kingom']` — correct |
| `chat.gate('what country is chelsea fc from?')` (with `?`) | same |
| `chat.gate('what sport is chelsea fc in')` | `['chelsea_fc', 'sport', 'association_football_club']` — correct |
| 8-question no-regression battery (taught-recall, self/identity, definitional-copula, unknown-abstain, teach, does-fronted-out-of-scope) | ALL unchanged |
| **LESION** `BRAIN_RELATION_FRONTED_QUESTIONS=0` on the target question | reverts to `None` (the exact pre-fix abstain) — load-bearing |
| re-enable (unset the lesion) | restores the fix | 

**byte-identical, asserted in the data (docs/TERMS.md's own bar — hash comparison, not code-reading):** 16
pre-existing questions (taught-recall x2, unknown-abstain, self/identity x3, definitional-copula x2, teach,
taught-recall-after-teach, does-fronted-out-of-scope, open-ended-generation x2, yes/no, greeting, relational
"of"-question) run through TWO INDEPENDENT fresh `ChatBrain` builds — one with the fix default-ON, one with
`BRAIN_RELATION_FRONTED_QUESTIONS=0` — and SHA-256-hashed per answer: **0 diffs across all 16.**

Full transcripts: `research/findings/raw/_knowledge_chat_veto/{verify_fix.txt,byte_identical_check.txt}`; machine-readable
summary of every stage's inputs/outputs: `research/findings/raw/_knowledge_chat_veto/trace_evidence.json`.

## Bug 2 (CHARACTERIZED, not fixed — a genuine judgment call): the GNW two-organ consensus bus vetoes EVERY LTM-sourced fact

Running the SAME question through the FULL wrapper stack `webapp/server.py` actually installs on a live turn
(`gnw_bus_shadow.install_bus_gate` -> `gnw_two_organ_bus.install_two_organ_gate` -> (three-organ: did not
install, see Bug 4) -> `gnw_deliberation.install_deliberation_gate` -> `value_choice_production_organ` ->
`gnw_multistep_deliberation.install_multistep_gate`, the exact order `server.py` uses) — **`chat.gate(...)`
still returned `None` after Bug 1's fix.** Dumping every `chat._last_*` trace after the call pinpoints it
exactly:

```
chat._last_two_organ = {'organ_a_recall': 'united_kingom', 'organ_b_confirmed': False,
  'expected': None, 'committed': None, 'ignited': False, 'n_ignited': 0,
  'abstain_reason': 'consensus_veto_organ_b_withheld', 'authored_by': 'two_organ_bus',
  'agent': 'chelsea_fc', 'action': 'country', 'bus_svo': None}
chat._last_gnw_delib     = {'acted': False, 'reason': 'already_abstained'}       # correctly inert (its own scope: only >=2-candidate conflicts)
chat._last_gnw_multistep = {'acted': False, 'reason': 'not_chase_form'}         # correctly inert (not a chase-form question)
```

`organ_a_recall = 'united_kingom'` is CORRECT — the forward composer recall (the actual query) succeeded. The
veto is `organ_b_confirmed = False` with `expected = None`: organ B (`gnw_two_organ_bus.py`, a spiking
"surprise monitor" that must CORROBORATE organ A's recall against its own pre-registered expectation before the
two-organ coincidence can cross the ignition knee — the module's own docstring: "only their coincidence crosses
the ignition knee") had **no expectation registered at all** for `('chelsea_fc', 'country')`.

**Root cause.** `gnw_two_organ_bus._chat_concepts` builds organ B's entire expectation registry `e_b` from
`getattr(chat, "stored_facts", [])` — `ChatBrain.stored_facts`, populated once at construction from
`self.inner.composer.kb`. But the composer is a `TieredFactStore(buffer, ltm)`, and `TieredFactStore.kb`
(attribute delegation, `research/runners/tiered_fact_store.py`) exposes ONLY the small conversational BUFFER's
facts — it never enumerates the LTM tier's contents (confirmed directly: `'chelsea_fc' in (chat.agents_set |
chat.actions_set | chat.patients_set)` is `False` even with the LTM attached). So `e_b.get(('chelsea_fc',
'country'))` is `None` for EVERY one of the 15,000 Wikidata-core facts, unconditionally — not just this one, not
just relation-fronted questions. Since the two-organ bus REQUIRES the coincidence of organ A AND organ B to
ignite, and organ B can structurally never corroborate an LTM-routed concept, **the default-ON two-organ
consensus bus vetoes every single knowledge-core recall**, regardless of correctness, regardless of the
question's syntactic shape. This is a strictly BROADER veto than Bug 1: even a perfectly-parsed, perfectly-
recalled LTM fact never ignites through this layer.

**Why this is characterized, not fixed, per the task's explicit instruction ("never make it answer by DROPPING
the abstain safety").** Organ B is a genuine anti-crosstalk/anti-hallucination corroboration mechanism (a
validated 6-seed-GO de-risk, reused-by-import here) — it was calibrated against a SMALL, explicitly-taught
vocabulary (`N_TRAINED_DEFAULT`-scale cue-addressable blocks, one per distinct stored patient). Closing this gap
is a genuine architecture decision with real trade-offs, not a mechanical bug fix:
1. **Extend organ B's registration to the LTM tier** — architecturally the most faithful, but the LTM holds
   15,000 facts vs. the handful organ B's cue-addressable-block design was sized for; whether the mechanism
   scales (compute, block-collision risk) needs the original de-risk owner's review.
2. **Exempt LTM-routed recalls from organ B's participation** (fall back to organ-A-only, or the underlying
   N-organ bus) — narrows the anti-crosstalk moat's coverage specifically for bulk knowledge; needs the
   original de-risk's crosstalk/false-hop rate re-measured under this carve-out before it can be trusted.
3. **Treat "no registered expectation" as organ-B ABSTAINING from the vote** (distinct from "expectation
   contradicted") rather than a de-facto veto — the most surgical option, but requires touching
   `coincidence_hop`'s ignition threshold math to add a third outcome without changing the calibrated 2-of-2
   knee for the case organ B DOES have an opinion.

None of these was applied. `BRAIN_GNW_2ORGAN=0` is an existing, ALREADY-BUILT escape (byte-identical to the
pre-2026-08-20 path per its own doc), but flipping a repo-wide default that gates a validated 6-seed-GO faculty
is exactly the kind of default-behavior judgment call this task was told not to make unilaterally — it is named
here, not flipped.

Full transcript: `research/findings/raw/_knowledge_chat_veto/{full_pipeline_trace.txt,full_trace2.txt}`.

## Two further discoveries made while tracing (not on the critical path for this bug, logged for completeness)

**Bug 3 — a narrower, LATENT instance of the same failure family.** `webapp/gnw_bus_shadow.py`'s N-organ bus
(the layer BEHIND the two-organ bus; not reached in production today because the two-organ wrapper never
delegates down to it on the normal path — confirmed by `chat._last_gnw_bus` staying `None` throughout the full-
pipeline trace) has its OWN "organ C" reverse-binding check: `cand_C = cand_A if composer.query_agent(action,
cand_A) == agent else None`. Direct trace: `composer.query_agent('country', 'united_kingom') = 'man_city'`
(WRONG — many UK-based entities share `country=united_kingom`; `ShardedPhasorStore.query_agent`'s OWN
docstring documents "Fan out to ALL shards and return the first hit" — an intentionally arbitrary, non-
disambiguating primitive for its OWN documented use case, but organ C reuses it as if it were injective). This
would false-veto the SAME question if the two-organ bus were ever disabled (`BRAIN_GNW_2ORGAN=0`) — noted for
awareness, not fixed (same class of judgment call as Bug 2; `ask_yes_no(agent, action, cand_A)` looks like a
promising exact-triple-match replacement that avoids the many-to-one ambiguity and is still a "genuinely
different substrate read" per organ C's own design intent, but was not implemented or verified here).

**Bug 4 — unrelated, tangential.** `webapp/server.py` gates the three-organ bus install with
`os.environ.get("BRAIN_GNW_3ORGAN", "1")` (defaults ON per its own 2026-08-21 comment: "FLIPPED default-ON"),
but `gnw_three_organ_bus.three_organ_enabled()` — called FIRST INSIDE `install_three_organ_gate` — reads the
SAME env var with a DIFFERENT default: `os.environ.get("BRAIN_GNW_3ORGAN", "")` (empty string, OFF). When the
var is genuinely unset (the common case), the outer server.py check passes (`"1"` is truthy) and calls
`install_three_organ_gate`, whose own FIRST LINE (`if not three_organ_enabled(): return False`) then
immediately declines. **The three-organ consensus bus, believed default-on by its own code comments, never
actually installs in production.** Confirmed empirically: `install_three_organ_gate(chat)` returned `False` in
this session's full-pipeline trace with `BRAIN_GNW_3ORGAN` unset. Not investigated further (unrelated to this
bug — the two-organ veto already fires before three-organ would matter).

## Next steps (flagged, not actioned here)

Bug 2 is the highest-impact follow-up (it blocks 100% of the shipped 15k-fact knowledge core through the
default production path) and is flagged as a background task for owner triage. `research/FAILURE_LOG.md` gets
one line each for Bugs 2, 3, and 4 (NOT-GATEABLE: each needs an architecture decision before a mechanical check
can be written).

## Commands to reproduce

```bash
SIM_BACKEND=numpy /path/to/.venv/bin/python - <<'PY'
import os, sys
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")
sys.path.insert(0, "/path/to/checkout")
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
from research.runners.developed_brain_io import _inner_agent
from research.runners.tiered_fact_store import TieredFactStore
from research.runners.sharded_phasor_store import ShardedPhasorStore

agent, aliases, _ = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
ltm = ShardedPhasorStore.load("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")
inner = _inner_agent(agent); inner.composer = TieredFactStore(inner.composer, ltm)
chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())

print(chat.gate("what country is chelsea fc from"))   # ['chelsea_fc', 'country', 'united_kingom'] post-fix
os.environ["BRAIN_RELATION_FRONTED_QUESTIONS"] = "0"   # lesion -> reverts to None (pre-fix)
print(chat.gate("what country is chelsea fc from"))
PY
```
