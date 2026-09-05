---
type: finding
status: de-risk-GO-6of6
date: 2026-09-05
mechanism: scaffold-retirement de-risk (backlog rank 13) — the on-brain comprehension (BridgeParser.role_of / the definitional-copula comprehension-helper) + the GNW ignition-bus recall combiner extended to self/identity questions and the anaphora-parse-miss, retiring the host QuestionRouter fallback for those two classes when BRAIN_NEURAL_SELFID / BRAIN_NEURAL_ANAPHORA_ABSTAIN are ON (both default OFF)
lane: integration-first (WIRING BACKLOG rank-13)
integration_faculty: content-selection
verdict: GO 6/6 seeds (42/43/44/100/101/102), both flag-gated extensions, on BOTH the plain host `gate()` and the installed GNW-bus `gate()` (the actual unconditional production combiner, `webapp/server.py::brain_reply`). (a) self-referential factual SVO ("what do you use/learn/store?") and (b) bare self-identity ("what are you"/"who are you") both answer correctly AND retire the host router (0 calls to `_gate_router_combine`/`QuestionRouter.match_fact`, instrumented, not inferred). (c) the anaphora-miss ("what does it fly?" after a referent is established) is a REAL, verified host-router confabulation when the flag is OFF (answers `['dog','chase','cat']` — an unrelated fact — instead of abstaining) and ABSTAINS honestly when the flag is ON; legitimate anaphora recall is unaffected. LOAD-BEARING: lesioning the on-brain `BridgeParser.role_of` (the 2026-08-12 CHOOSE-1 recipe, unchanged) collapses the self-factual answer to abstain on 6/6 seeds while the parser-independent recall reflex survives (100% attribution to the manipulation) — this is the ONLY class here with a genuinely-neural mechanism to lesion; the bare-identity and anaphora-miss extensions are host comprehension-helper / control-flow changes (same honesty class as the pre-existing copula/relation-fronted routes), evidenced by the flag-toggle + call-count retirement proof, not a spiking lesion. Both flags stay default OFF — this is a de-risk, not a production flip; the ledger's existing `content-selection` row (`still the self/identity + noisy-anaphora fallback`) remains accurate for the default turn.
artifacts:
  - research/runners/brain_chat_tui.py
  - webapp/gnw_bus_shadow.py
  - research/runners/_selfid_anaphora_scaffold_derisk.py
  - research/findings/raw/_selfid_anaphora_scaffold_derisk/result.json
  - research/findings/raw/_selfid_anaphora_scaffold_derisk/lesion_result.json
verification: |
  SIM_BACKEND=numpy python -u -m research.runners._selfid_anaphora_scaffold_derisk
    (6 seeds x 2 combiners [plain host gate() / installed GNW-bus gate()] x {self_factual, self_identity,
    anaphora_miss, regression}; go=True, status=GO; per-combiner summary: self_factual_correct=True,
    self_factual_retired=True, self_identity_correct=True, self_identity_retired=True,
    anaphora_off_confabulated_on_this_fixture=True, anaphora_on_abstains=True,
    anaphora_first_turn_unaffected=True, no_regression=True (36 checks/combiner) -- IDENTICAL on plain and bus.)
  research/runners/_selfid_anaphora_scaffold_derisk.py::_lesion_selfid, all 6 seeds (onebrain composer):
    all_intact_correct=True, all_lesion_collapses=True, all_reflex_survives=True; attribution 100.0% of the
    effect to the role_of lesion, 0.0% also present in the control, every seed.
---

# Rank-13 de-risk: the on-brain comprehension + GNW bus extended to self/identity + the anaphora-miss

## The residual this targets

The 2026-08-12 CHOOSE-1 integration (`2026-08-12-INTEGRATION-1-CHOOSE-neural-question-parse-router-retired-for-factual.md`)
made a factual-SVO question's (agent, action) comprehension NEURAL — `ChatBrain._neural_question_parse` presents the
question's content words to the on-brain `BridgeParser`, whose (position, voice)→role conjunction fires the role
assignment on Izhikevich neurons — and AUTHORITATIVE: a comprehended parse feeds the recall (since 2026-08-13, the
GNW N-organ ignition-bus, `webapp/gnw_bus_shadow.py`, which `webapp/server.py::brain_reply` installs on
EVERY production turn, unconditionally); a DECLINED parse honestly abstains instead of falling to
`QuestionRouter.match_fact`'s role-blind keyword bag-of-words. That finding's own "Honest scope" named the residual
verbatim: *"the router... still owns self/identity + the anaphora-fallback"* — matching
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `content-selection` row (`host_scaffold_in_default`: *"still the
self/identity + noisy-anaphora fallback"*) and `research/coordination/scaffold_retirement_backlog.md`'s rank-13 entry.

Reading `research/runners/brain_chat_tui.py::ChatBrain._extract_route` confirmed the mechanism of the gap precisely:

- **Self/identity.** `has_self_alias = any(t in self.router.self_aliases for t in content)` unconditionally routes
  around the neural parser whenever ANY content token is a self-alias ("you"/"your"/"it"/...) — even when a genuine
  second content word (the action) is present. The final guard `if a in self.router.self_aliases or v in
  self.router.self_aliases: return None` then hard-blocks the extraction outright. So `"what do you eat?"` NEVER
  reaches `_neural_question_parse`, regardless of whether the brain actually has a matching self-fact.
  `_definitional_copula_route` (the existing `"what is X?" -> [X, "isa"]` comprehension helper) explicitly rejects a
  self-alias subject too, so a bare `"what are you?"` / `"who are you?"` is equally unroutable.
- **Anaphora-miss.** In `gate()`, `if sub == "__ABSTAIN__" and not anaphora_used: return None` (honest abstain) is
  followed by `if sub not in (None, "__ABSTAIN__"): return sub` — meaning an anaphora-resolved query that comes back
  `"__ABSTAIN__"` falls all the way through to `self._gate_router_combine(q)`, the host router, on the reasoning
  ("the WM referent may be noisy, so let the host router try"). `gnw_bus_shadow.gate_via_bus` mirrors the identical
  escape in two places (the 'route'-mode veto-then-anaphora branch, and the 'decline'-mode anaphora branch).

## The extension (reuse, not a new mechanism)

Two flags in `research/runners/brain_chat_tui.py`, both **default OFF** (unset = byte-identical to before this
session):

**`BRAIN_NEURAL_SELFID`**
- *(a) self-referential factual SVO* — `_extract_route` now resolves a self-alias token to `'brain'` (mirroring
  `QuestionRouter._resolve_self`, the host router's own resolution) **before** the `has_self_alias` gate. A question
  like `"what do you eat?"` becomes content `["brain","eat"]` — indistinguishable in shape from any other
  2-content-word factual query — so it flows through the SAME `_neural_question_parse` (genuinely on the on-brain
  `BridgeParser`) and, via the installed GNW bus, the SAME substrate-ignition combiner every other factual-SVO
  question already uses. No new mechanism: `'brain'` is simply a known agent now.
- *(b) bare self-identity* — `_definitional_copula_route` now accepts a self-alias subject, resolving it to
  `'brain'` and returning `['brain', 'isa']` (reusing the identical `"what is X?"` recipe). Because tiny-demo's
  self-facts use `has`/`uses`/`store`/`learn`, not `isa`, a **miss-only candidate-relation retry** — the HOST
  router's own preference order (`has`,`have`,`is`,`uses`,`use`; `QuestionRouter.match_fact`'s `is_identity_q`
  branch, unchanged) — is added at BOTH places that can author the covered-class answer: `_substrate_recall` (the
  plain/non-bus path) and `gnw_bus_shadow.gate_via_bus`'s 'route' branch (the actual production combiner, which
  reads `composer.query_patient` directly and never calls `_substrate_recall` — so mirroring the retry there was
  necessary for this to reach production at all, not merely the non-bus test path). This sub-mechanism is a HOST
  comprehension-helper (regex + a fixed candidate list), the same honesty class as the pre-existing
  copula/relation-fronted/kb-relation routes — **not** a BridgeParser claim. Recall stays on the substrate
  (`what_does` / the bus's organ reads) either way, so the moat is untouched: an unknown self-fact still abstains.

**`BRAIN_NEURAL_ANAPHORA_ABSTAIN`** — when an anaphora-resolved query's substrate/bus recall declines or finds no
fact, `gate()` and both `gate_via_bus` branches now abstain instead of calling `_gate_router_combine` — the same
honesty already applied to the direct-query abstain, extended to the anaphora-miss.

## Verification

### (a)+(b) Correct comprehension + retirement (6 seeds, both combiners)

`research/runners/_selfid_anaphora_scaffold_derisk.py` builds the REAL production `ChatBrain` (`rf` composer,
numpy-CPU, `webapp.server._build_chat_brain`) and drives it two ways: the **plain** host `gate()`, and the
**installed GNW-bus** `gate()` (`gnw_bus_shadow.install_bus_gate` — the same wrapper `webapp/server.py::brain_reply`
installs unconditionally on every real turn). `_gate_router_combine` and `QuestionRouter.match_fact` are wrapped
with call counters (not inferred from the answer) so "retired" means the host router genuinely never ran.

| class | question(s) | flag OFF | flag ON | retired (0 host-router calls) |
|---|---|---|---|---|
| self-factual | "what do you use/learn/store?" | `['brain','use','spikes']` etc. (via the router) | IDENTICAL answer | YES, both combiners |
| self-identity | "what are you" / "who are you" | `['brain','use','spikes']` (via the router) | IDENTICAL answer | YES, both combiners |

Every one of these held on **all 6 seeds x both combiners** (`summary.plain`/`summary.bus`:
`self_factual_correct=True`, `self_factual_retired=True`, `self_identity_correct=True`, `self_identity_retired=True`).

### (c) The anaphora-miss confabulation is real, and the extension retires it

Sequence: teach nothing extra (tiny-demo ships `dog chase cat`); ask `"what does dog chase?"` (establishes the
discourse referent `cat`), then `"what does it fly?"` (cats have no `fly` fact — a well-formed, unanswerable,
anaphora-resolved query). Measured, not assumed:

- **Flag OFF** (today's production): the host router answers `['dog', 'chase', 'cat']` — re-asserting an unrelated
  stored fact in response to a question about flying. This is the exact confabulation shape ("what does fish fly?"
  -> "cat eat fish") the 2026-08-12 direct-query fix already retired, now shown to also occur through the anaphora
  path, which that fix explicitly left open.
- **Flag ON**: abstains (`None`), on both combiners, on 6/6 seeds.
- Legitimate anaphora recall (`"what does dog chase?"` then `"what does it eat?"` -> `['cat','eat','fish']`) is
  **unaffected** by the flag — `anaphora_first_turn_unaffected=True` on every seed/combiner. The extension can only
  add abstentions on an already-unanswerable turn; it never touches a working anaphora recall.

### No regression

36 checks per combiner per run (3 STORED + 2 UNSTORED + 1 anaphora-hit sequence, x 6 seeds) are byte-identical
flag-on vs flag-off on both combiners (`no_regression=True`, `n_regression_checks=36`). The full per-seed,
per-question rows, `go=True`, `status=GO`, 12/12 `Verdict.require` preconditions passed, 0 undefined reasons, and
1 honestly-disabled process (the lesion battery, run separately below) are all in
`research/findings/raw/_selfid_anaphora_scaffold_derisk/result.json`.

### Load-bearing: the on-brain BridgeParser lesion (class (a) only)

Class (a) is the only mechanism here that is genuinely neural (BridgeParser.role_of); (b) and (c) are host
comprehension-helper / control-flow extensions whose "load-bearing" evidence is the flag-toggle + call-count proof
above, not a spiking lesion — conflating the two would be exactly the kind of overclaim `docs/TERMS.md`'s "lesion"
and "fully spiking" entries exist to catch.

Built the `onebrain` composer (the only composer that carries a `.parser`) for each of the 6 seeds, asked
`"what do you use?"` (intact), then monkeypatched `parser.role_of = lambda *a, **k: "junk_role"` (the EXACT
2026-08-12 lesion recipe — a role readout that never resolves to `"agent"`/`"action"`) and asked again:

- **6/6 seeds**: intact answer correct (`['brain','use','spikes']`); lesioned answer collapses to `None` (abstain);
  the parser-INDEPENDENT recall reflex (`composer.query_patient('brain','use')`) still returns `'spikes'`
  (dissociation — the substrate still HAS the fact, only the comprehension that would route to it is gone).
  `attributable_to`: 100.0% of the effect owed to the manipulation, 0.0% also present in the control, every seed.
- Because `gate_via_bus`'s dispatch (`gate_extract` -> `_extract_route`) is the SAME shared extraction call the
  plain path uses — only the recall-COMBINATION differs downstream — this lesion result applies to the bus-installed
  production path by construction (a lesioned parser makes `_extract_route` return `"__DECLINE__"` before either
  combiner is ever reached); it was not re-measured as a separate expensive onebrain build, since the extraction
  step is architecturally identical for both.

Raw: `research/findings/raw/_selfid_anaphora_scaffold_derisk/lesion_result.json`.

## Honest scope (what was NOT covered)

- **Panel size.** 3 self-factual questions, 2 bare-identity phrasings, and ONE hand-constructed anaphora-miss shape
  (mirroring this codebase's existing small hand-picked STORED/SELF/UNSTORED panels) — not a broad or held-out
  sweep. A wider panel (more self-aliases: "my"/"its"/"yourself"; more identity phrasings: "what's your kind";
  more anaphora-miss shapes) is a natural next rung, not attempted here.
- **"what do you do?"** and similarly aux-`do`-as-main-verb questions stay unroutable: `_extract_route`'s stopword
  list strips `"do"` unconditionally (ambiguous between the auxiliary and a genuine main verb), leaving only 1
  content word — a POS-tagging fix would be a NEW mechanism, out of scope for "apply the proven recipe."
- **The candidate-relation retry (b)** is scoped tightly to `agent=='brain'` and reuses the host's OWN fixed
  preference order; it cannot answer an identity question whose defining fact uses a relation outside
  `{isa,has,have,is,uses,use}` (an honest, narrow limitation inherited directly from mirroring the host's own list,
  not a new gap this extension introduces).
- **Both flags stay default OFF.** This is a de-risk of the mechanism, not a production flip; the ledger's
  `content-selection` row and its `retire_status: "BLOCKED:neural-render"` are unaffected and remain accurate
  for the default (flag-off) turn — a separate blocker (the recall-answer surface, not comprehension) this de-risk
  does not touch.

## Bottom line

Does the neural parse retire the host QuestionRouter for self/identity + anaphora, load-bearing? **For the covered
shapes tested — yes, on all 6 seeds, on both the plain and (critically) the actual production GNW-bus combiner —
with the load-bearing/genuinely-neural claim correctly scoped to the ONE sub-class (self-referential factual SVO)
that has a spiking mechanism to lesion.** The residual (aux-`do` identity questions, non-preference-list relations,
broader anaphora-miss shapes) is characterized above, not hidden, and the flags stay default OFF pending a
production-flip decision.
