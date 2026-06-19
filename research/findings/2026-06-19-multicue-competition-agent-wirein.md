# Multi-cue role-competition parser — PRODUCTION WIRE-IN into the conversational agent (behind a default-OFF flag)

**Date:** 2026-06-19
**Type:** Phase-1 production wire-in (integration de-risk behind a flag), the "bank the validated mechanism" pattern.
**Mechanism de-risk (GO):** `2026-06-19-multicue-competition-spiking-derisk.md` (spiking install path 5/6 seeds;
all anti-cheat controls; moat 0 breaches on every seed). Numpy mechanism: `2026-06-19-multicue-competition-derisk.md`.
**Files:** `research/runners/multicue_role_parser.py` (the production drop-in), the `enable_multicue_competition`
flag on `research/runners/brain_conversational_agent.py`, `tests/test_multicue_competition_agent.py` (CI guard).
**Verdict:** **DONE.** The validated SPIKING multi-cue role-competition is wired into the production conversational
agent behind a default-OFF `enable_multicue_competition` flag; the agent answers who/what CORRECTLY on degraded
(object-fronted) English where the default position-only path inverts the roles, clean canonical is un-regressed,
the no-confab moat holds, and flag-OFF is byte-identical. This is the same "bank a validated mechanism behind a
flag" step as conversational #1 (`enable_attributed`) and #2 (`enable_biased_competition`) — NOT a production-
default flip (that is a separate deliberate step).

---

## 1. What was wired

The agent's `hear(sentence)` comprehends an SVO statement into `{agent, action, patient}` and stores it via the
composer. The default `BridgeParser` assigns roles by **word position** (pos0→agent, pos1→action, pos2→patient),
so a degraded order — an **object-fronted** sentence like "apple eat dog" — gets the roles **backwards**
(agent=apple, patient=dog). The validated spiking multi-cue role-competition fixes this: it weights **position +
animacy + verb-selectional-fit** by their learned validities and lets a spiking WTA settle the agent/patient
decision, so the **content** (apple is inanimate + the patient of "eat"; dog is animate) overrides the misleading
position.

**`MultiCueRoleParser`** (`research/runners/multicue_role_parser.py`) is a thin production drop-in that wraps the
validated `SpikingRoleCompetition` (re-pointed `biased_competition_buffer.py` Wong-Wang/Rutishauser WTA over
thematic ROLES + plastic cue→role projections, real `cp_firing_states`) with the validated **INSTALL-path** cue
validities, and exposes the SAME `parse(words, voice) → {agent, action, patient}` shape `BridgeParser.parse`
returns — so it slots straight into the agent `hear()` / composer `store()` path. The verb (action) is identified
lexically from a caller-supplied known-verb set (the legitimate lexical front-end boundary, identical to
`FrameParser` and the buffer's `content_bias_target`); the two nouns' roles are the spiking WTA decision.

**`BrainConversationalAgent(enable_multicue_competition=True, multicue_verbs=<set>)`** routes `hear()` (the
production turn entry point) through `MultiCueRoleParser` instead of the position-only parser. There is also an
explicit `hear_multicue()` method. **Default OFF = byte-identical** (the parser is never even constructed; the
`hear()` branch is skipped; the existing test passes verbatim).

---

## 2. The load-bearing degraded-input win vs the default's failure

The same object-fronted input, parsed two ways (both observed, CPU/numpy):

| Input "apple eat dog" (object-fronted) | agent | patient | who_does('eat','apple') | what_does('dog','eat') |
|---|---|---|---|---|
| **default position-only** (`BridgeParser`, position-by-construction) | **apple** ✗ | **dog** ✗ | — (would be wrong) | — (would be None) |
| **flag ON** (spiking multi-cue competition) | **dog** ✓ | **apple** ✓ | **dog** ✓ | **apple** ✓ |

The default assigns the roles **backwards** (it would store "apple eat dog" as a fact about apple eating dog); the
multi-cue path stores the content-correct "dog eat apple" and the agent's full Q&A then answers correctly. The
contrast is asserted directly in the CI test (`test_default_position_only_parser_inverts_object_fronted` shows the
position-only decision is agent=apple; `test_multicue_resolves_object_fronted_degraded_input` shows the flag-ON
agent answers `who_does('eat','apple')=='dog'`). It is a *different non-None* failure mode, not merely a None: the
default would have made `who_does('eat','dog')` return 'apple' — the flag-ON path returns None there and the
content-correct answer for `who_does('eat','apple')`.

The underlying spiking robustness is the de-risk's (re-confirmed this session on numpy, install path, seed 42):
MULTICUE **0.950** vs POSITION-ONLY **0.225** on the position-degrading battery (object-front 0.950 vs 0.000),
moat 0 breaches.

---

## 3. The CI guard (all numpy-runnable, 7 tests, 11 s)

`tests/test_multicue_competition_agent.py` — uses the **rf composer with an explicit vocab** so the `denoise64`
cache is NOT needed (CPU-runnable):

| Test | Asserts |
|---|---|
| `test_default_position_only_parser_inverts_object_fronted` | the default position-only parser assigns 'apple eat dog' backwards (agent=apple) — the load-bearing failure |
| `test_multicue_resolves_object_fronted_degraded_input` | flag ON: the agent answers who/what CORRECTLY on the degraded object-fronted input (dog=agent, apple=patient) + returns None for the inverted (default-path) answer |
| `test_multicue_no_regression_on_clean_canonical` | flag ON: clean canonical 'cat eat ball' still comprehends correctly (no native-case regression) |
| `test_multicue_moat_abstains_on_unstored_fact` | flag ON: an unstored fact → abstain (None); zero confabulation |
| `test_multicue_moat_ambiguous_sentence_not_decisive` | flag ON: all-ambiguous 'dog chase cat' (two animate + symmetric verb) → `parse_decisive` decisive=False (content gate); decisive counterpart True |
| `test_flag_off_parser_not_built_and_default_path` | flag default-OFF: parser never built, flag OFF, default path answers normally |
| `test_enable_multicue_requires_verbs` | construction error if the flag is ON without `multicue_verbs` |

**Flag-OFF byte-identity:** `tests/test_brain_conversational_agent.py` passes verbatim on numpy (7 passed, 5
GPU-gated skipped) with the new flag present — the default OFF path is untouched.

---

## 4. What is WIRED vs DEFERRED (honest, per the BRAIN-BASED-ONLY directive + the de-risk)

- **WIRED (this step):** the validated SPIKING role-competition **inference** — the cue populations → plastic
  cue→role projections → Wong-Wang role accumulators (sel_agent/sel_patient) in mutual inhibition; the WINNER is
  the spiking WTA settle. The cue VALIDITIES are the validated **INSTALL-path** weights (the robust 5/6-seed GO
  arm; an installed learned parameter, like a pre-trained weight — `position 6 < semantic 20`, distractor 2).
- **DEFERRED (documented follow-ons, NOT blockers for the wire-in):**
  1. **Continual ON-SUBSTRATE cue-validity learning** — the three-factor rule (spike-eligibility × reward × vote)
     that learns the validity spread on the substrate is **seed-variable in robustness** at the spiking scale (an
     honest boundary on the *learning*, not the mechanism; see the de-risk §2b/§5). The install path is the robust
     deployment.
  2. **Neuralizing the reward** in that learner (the host winner-matched-gold signal → an on-substrate RPE, as for
     the nav SnC/RPE).
- **HOST front-end (the legitimate lexical boundary, identical to `FrameParser` + the buffer's
  `content_bias_target`):** the verb (action) is identified from the caller's known-verb set, and the feature
  LEXICONS (animacy, verb-selectional-fit) supply each cue's VALUE for a word. They do NOT supply the role decision
  (that is the learned-weight spiking competition; the de-risk's PERMUTED-CUE + NO-LEARNING controls guard against
  the lexicon doing the discrimination). Conversion target = a learned lexical-feature map.
- **Turn-loop integration note (honest):** `hear()` routes through the multi-cue competition when the flag is ON,
  so the production turn entry point comprehends degraded input. The drop-in's validated scope is the **2-noun
  transitive** clause (the de-risk scope); a 1- or 3+-noun input falls back to a surface-order read (the agent's
  Q&A still abstains on any unstored fact, so the moat is never weakened). Extending the competition to
  attributed/multi-clause inputs is a bounded follow-on.

---

## 5. Not flipped (deliberate)

No library default is changed. `enable_multicue_competition` defaults **OFF** everywhere; the rf/onebrain default
comprehension path is byte-unchanged. A production-default flip (and the choice of which agents/demos adopt it) is
a separate deliberate step, the same way #1/#2 were banked behind flags first.

---

## 6. Provenance
- Mechanism + GO bar + controls: `2026-06-19-multicue-competition-spiking-derisk.md`,
  `2026-06-19-multicue-competition-derisk.md`, `2026-06-19-multicue-competition-parser-scoping.md`.
- Reuse substrate: `research/runners/_phaseB_multicue_competition_spiking_derisk.py` (`SpikingRoleCompetition` +
  `cue_evidence` + `INSTALLED_CUE_WEIGHTS`), `research/runners/biased_competition_buffer.py` (the Wong-Wang /
  Rutishauser WTA + `ANIMACY`/`VERB_SELECTS`), `research/runners/brain_conversational_agent.py` (the agent +
  `BridgeParser`).
- Wire-in template: `enable_biased_competition` on `MultiTurnAgent` + `tests/test_multireferent_biased_competition.py`.
- Competition Model: Bates & MacWhinney 1982/1989. Biased competition: Desimone-Duncan 1995; Wong-Wang 2006.
