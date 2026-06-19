# Phase-2 CASE cue → PRODUCTION WIRE-IN into the conversational agent (behind a default-OFF flag)

**Date:** 2026-06-19
**Type:** Phase-2 production wire-in (integration de-risk behind a flag), the "bank the validated mechanism" pattern. Tier 1, item 1 of the owner-accepted conversational-primary roadmap.
**Mechanism de-risk (GO):** `2026-06-19-case-cue-crosslanguage-derisk.md` (spiking case path 5/6 seeds; the cross-linguistic dissociation 6/6; all anti-cheat controls — position-only collapses, case-lesion, no-learning, permuted-case; moat 0 breaches).
**Files:** `research/runners/case_aware_role_parser.py` (the production drop-in), the `enable_case_competition` flag on `research/runners/brain_conversational_agent.py`, `tests/test_case_cue_crosslanguage_agent.py` (CI guard).
**Verdict:** **DONE.** The validated CASE-aware spiking multi-cue role-competition is wired into the production conversational agent behind a default-OFF `enable_case_competition` flag. The agent answers who/what CORRECTLY on a FREE-word-order CASE-MARKED sentence (Japanese-style が/を) where word-position cannot, the install-path cross-linguistic dissociation holds in the production path, canonical case is un-regressed, the no-confab moat holds, and flag-OFF is byte-identical. This is the same "bank a validated mechanism behind a flag" step as the Phase-1 `enable_multicue_competition` wire-in — NOT a production-default flip (that is a separate deliberate step).

---

## 1. What was wired

The Phase-1 wire-in (`enable_multicue_competition`) made the agent robust to **degraded English** (object-fronted / scrambled) by weighting **position + animacy + verb-fit** by their learned validities in a spiking WTA, so **content** overrides misleading position. But English semantics only disambiguate free order because an English patient is typically *inanimate* (apple ≠ dog). In a language with **two animate nouns** ("dog chase wolf"), animacy ties and the symmetric verb is silent — and the **case particle** is the sole reliable role cue. Phase 2 adds that cue.

**`CaseAwareRoleParser`** (`research/runners/case_aware_role_parser.py`) extends the validated `SpikingRoleCompetition` with a **fifth cue — `case`** — whose signed vote (nominative particle → **+1 agent**, accusative → **−1 patient**) joins the position+animacy+verb-fit competition. The cue VALIDITIES are the validated **INSTALL-path case-language weights** (`case 22` dominant, `position 6` low, distractor 2). It exposes the SAME `parse(words, voice) → {agent, action, patient}` shape `BridgeParser`/`MultiCueRoleParser` return, plus `parse_decisive` for the moat content gate. The verb is identified lexically from a caller-supplied known-verb set; each noun's case particle is pulled from the surface tokens via a `case_lexicon` (default Japanese-style `{ga→nom, wo→acc}`; the particle is a clitic immediately following its noun, consumed in the lexical front-end) or supplied explicitly via `markers=`.

**`BrainConversationalAgent(enable_case_competition=True, case_verbs=<set>, case_lexicon=<optional>)`** routes `hear()` (the production turn entry point) through `CaseAwareRoleParser` when ON; there is also an explicit `hear_case()` method. **Default OFF = byte-identical** (the parser is never constructed; the `hear()` branch is skipped). `enable_multicue_competition` takes precedence if BOTH are set (they are alternative comprehension front-ends; case is the case-language one).

**Self-contained co-residence (a real design constraint, solved).** The Phase-1 `SpikingRoleCompetition` reads the module-level `CUES`/`SEMANTIC_CUES` of `_phaseB_multicue_competition_spiking_derisk` both at construction AND at inference (`cue_weights`, `set_cue_weight`, `_semantic_contrast` all iterate them). Permanently adding `case` to those globals (what the de-risk does at import) would **break a co-resident plain `MultiCueRoleParser`** — its moat `_semantic_contrast` would `KeyError` on a missing `ev["case"]`. So every public `CaseAwareRoleParser` method **transiently** swaps the case-extended `CUES`/`_CUE_ID`/`SEMANTIC_CUES` in for the duration of the call and restores them (`_case_cue_context`). The plain Phase-1 path is left byte-identical when not inside a case-parser call — asserted by the CI test's global-isolation guard.

---

## 2. The load-bearing free-word-order win + the dissociation in the production path

The same free-word-order object-fronted CASE-MARKED input, parsed by the agent (observed, CPU/numpy, seed 42):

| Input "wolf wo dog ga chase" (object-fronted, case-marked) | agent | patient | who_does('chase','wolf') | what_does('dog','chase') |
|---|---|---|---|---|
| **position-only read** (surface after particle strip = [wolf, dog]) | **wolf** ✗ | **dog** ✗ | — (would be wrong) | — (would be wrong) |
| **flag ON** (CASE-aware spiking competition) | **dog** ✓ | **wolf** ✓ | **dog** ✓ | **wolf** ✓ |

The case particle (dog+が = nominative → agent, wolf+を = accusative → patient) **overrides** the surface order; the agent's full Q&A then answers correctly, and `who_does('chase','dog')` returns **None** (not the inverted answer). On the position-degrading battery the case parser reads **40/40 = 1.000** free-order object-front role accuracy (re-confirmed this session, install path, seed 42).

**The cross-linguistic dissociation in the production (install-path) path** — three observed facts:
- **case DECIDES** on the case-marked free-order toy (the table above);
- **case is SILENT on English** (no particles) — the SAME case-aware parser falls back to position+semantics, so canonical English "dog eat apple" still reads dog=agent / apple=patient;
- **case is what makes free order solvable** — a case-FREE Japanese-toy item (two animate nouns + a symmetric verb, object-front "wolf dog chase") is content-ambiguous, so a plain position+semantic parser **abstains** (`decisive=False`), while the **identical** item WITH the が/を particles is **decisive and correct**. The case particle's presence flips abstain ↔ decide — the install-path signature of the dissociation.

The full **LEARNED-weight** flip — the SAME three-factor learner driving `w_case` to the FLOOR (0.0) on English and to the TOP (20.0) on the Japanese-style toy, `profile_flips=True` on 6/6 seeds — is the de-risk's headline (`2026-06-19-case-cue-crosslanguage-derisk.md`). The production agent uses the validated **install** path (fixed case-language validities), so the production guard asserts the install-path dissociation above; the on-substrate-learned flip is the deferred follow-on (§4).

---

## 3. The CI guard (all numpy-runnable, 9 tests, 46 s)

`tests/test_case_cue_crosslanguage_agent.py` — uses the **rf composer with an explicit vocab** so the `denoise64` cache is NOT needed (CPU-runnable):

| Test | Asserts |
|---|---|
| `test_case_resolves_free_word_order_object_fronted` | flag ON: free-order object-fronted 'wolf wo dog ga chase' → who/what CORRECT by case (dog=agent, wolf=patient); who_does('chase','dog')=None (not inverted) |
| `test_case_silent_on_english_falls_back_to_position_semantics` | dissociation (English arm): case silent on 'dog eat apple' → position+semantics decide (dog=agent, apple=patient) |
| `test_case_is_what_makes_free_order_solvable` | dissociation (case-load-bearing arm): case-FREE item → decisive=False (abstain); identical item WITH ga/wo → decisive + correct |
| `test_case_no_regression_on_canonical_sov` | flag ON: canonical SOV 'dog ga wolf wo chase' still comprehends correctly |
| `test_case_moat_abstains_on_unstored_fact` | flag ON: an unstored fact → abstain (None); zero confabulation |
| `test_case_moat_unmarked_ambiguous_not_decisive` | flag ON: UNMARKED ambiguous 'dog wolf chase' → decisive=False; case-marked counterpart → decisive=True |
| `test_flag_off_parser_not_built_and_default_path` | flag default-OFF: parser never built, flag OFF, default path answers normally |
| `test_enable_case_requires_verbs` | construction error if the flag is ON without `case_verbs` |
| `test_case_parser_does_not_contaminate_plain_multicue` | global-isolation guard: using the case parser does NOT mutate Phase-1 `CUES`; a co-resident plain `MultiCueRoleParser` reads + abstains correctly afterward |

**Flag-OFF byte-identity:** `tests/test_brain_conversational_agent.py` (the full byte-identity guard) + `tests/test_multicue_competition_agent.py` pass verbatim on numpy (14 passed, 5 GPU-gated skipped) with the new flag present — the default OFF path and the Phase-1 multicue path are untouched.

---

## 4. What is WIRED vs DEFERRED (honest, per the BRAIN-BASED-ONLY directive + the de-risk)

- **WIRED (this step):** the validated CASE-aware spiking role-competition **inference** — the cue populations (now incl. `case`) → plastic cue→role projections → Wong-Wang role accumulators (sel_agent/sel_patient) in mutual inhibition; the WINNER is the spiking WTA settle. The cue VALIDITIES are the validated **INSTALL-path** case-language weights (case dominant, position low — an installed learned parameter, like a pre-trained weight).
- **DEFERRED (documented follow-ons, NOT blockers for the wire-in):**
  1. **Continual ON-SUBSTRATE cue-validity learning** — the three-factor rule (spike-eligibility × reward × vote) that LEARNS the validity spread on the substrate, and which produces the full LEARNED-weight cross-linguistic flip (English `w_case`→floor / Japanese `w_case`→top), is **seed-variable in robustness** at the spiking scale (the shared Phase-1/2 residual; **Tier 1 item 2**, the firm-the-learning follow-on). The install path is the robust deployment.
  2. **Neuralizing the reward** in that learner (the host winner-matched-gold signal → an on-substrate RPE, as for the nav SnC/RPE).
- **HOST front-end (the legitimate token-level lexical boundary, identical to the verb/animacy/verb-fit lexicons of `MultiCueRoleParser`):** the CASE MARKERS are a host-supplied lexicon — which particle TOKENS are nominative vs accusative. This is the **ISOLATING-particle** case (a set-membership check on the particle token, **NO morphological segmentation**), flagged for the eventual learned/neural lexical front-end. The role COMPETITION + the install-path validities are the brain-based win.
- **Phase 3 (DEFERRED — the next tier, NOT built here):** **FUSED/portmanteau case** (Russian -a/-u, Latin -us/-um) needs **sub-word morphological segmentation** (a new representational layer). Phase 2 is the isolating-particle case (Japanese が/を, Korean 이/가·을/를 — a token-level cue), which is what this wire-in covers.
- **Turn-loop integration note (honest):** `hear()` routes through the case competition when the flag is ON. The drop-in's validated scope is the **2-noun transitive** clause; a 1- or 3+-noun input falls back to a surface-order read (the agent's Q&A still abstains on any unstored fact, so the moat is never weakened). Extending the competition to attributed/multi-clause inputs is a bounded follow-on.

---

## 5. Not flipped (deliberate)

No library default is changed. `enable_case_competition` defaults **OFF** everywhere; the rf/onebrain default comprehension path is byte-unchanged. A production-default flip (and which language/agent adopts it) is a separate deliberate step, the same way the Phase-1 multicue wire-in was banked behind a flag first.

---

## 6. Provenance
- Mechanism + GO bar + controls + the dissociation: `2026-06-19-case-cue-crosslanguage-derisk.md`, `2026-06-19-phase2-case-cue-crosslanguage-scoping.md`.
- Reuse substrate: `research/runners/_phaseB_multicue_competition_spiking_derisk.py` (`SpikingRoleCompetition` + `cue_evidence`), `research/runners/multicue_role_parser.py` (the Phase-1 production drop-in this mirrors), `research/runners/brain_conversational_agent.py` (the agent + `BridgeParser`).
- Wire-in template: `enable_multicue_competition` on `BrainConversationalAgent` + `tests/test_multicue_competition_agent.py` (the Phase-1 wire-in, `2026-06-19-multicue-competition-agent-wirein.md`).
- Competition Model + cross-linguistic cue validity: Bates & MacWhinney 1982/1989. Case marking as a thematic-role cue: the canonical Japanese が/を test ("inu ga neko wo oikakeru" — both animate, only the particle tells you who chased whom).
