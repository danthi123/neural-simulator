---
type: finding
status: contributing
date: 2026-08-13
mechanism: T1-6 OTHER-REPAIR WIRED into the DEFAULT /api/brain-chat turn — a TARGETED conversational clarification on a low-comprehension utterance, replacing the bare D4 abstain. It COMPOSES the already-wired D4 comprehension monitor: on an in-scope transitive the D4 gate would ABSTAIN on (low sel-pool margin), the SAME co-resident SpikingRoleCompetition per-noun agent-evidence (a0,a1 = sel_agent-sel_patient off cp_firing_states, reuse-by-import from research/runners/comprehension_production_organ.py -> the 6/6-GO D4 de-risk) localises WHICH thematic role failed: sign(a0+a1) names the OVER-subscribed role (so the OTHER role is unresolved) and the pair-max magnitude confirms the roles are ACTIVE. The turn then asks a clarification NAMING the unresolved AGENT/PATIENT ("my role-binding didn't resolve the AGENT — which of them is doing the 'carry', the book or the cup?") or the OOV token. Default-ON, moat-safe (a QUESTION, never a fact; the turn stays an abstain), lesion-load-bearing (the D4 spiking signal zeroed -> no target -> the bare abstain), NO sim/ edit.
lane: Gate-B / T1-6 · Other-repair (a targeted clarification on a low-comprehension turn — repair, don't dead-end)
lane_ref: T1-6
verdict: GO / WIRED (production-integration). Single-process synchronous in-process verify on the real /api/brain-chat handler (SIM_BACKEND=numpy, GPU-free stub renderer, tiny-demo, composer=rf). All verify checks pass (research/runners/_gateB_repair_production_verify.py, ALL_OK) + a nine-turn pristine-vs-modified byte-identical regression (stash+re-run): FLAG-OFF identical across every turn; DEFAULT-ON changes ONLY the turns D4 already abstained on. The load-bearing per-noun role signal collapses to zero under the D4 lesion (reproduced numpy-CPU).
seed-waiver: production-INTEGRATION verify of an already-multi-seed-GO faculty (the D4 SpikingRoleCompetition semantic sel-pool read: `_spiking_comprehension_monitor_derisk.py`, a clean type-2 discrimination the cue-lesion collapses to chance, seeds 42/43/44/100/101/102). This doc verifies the deterministic WIRING glue (the per-noun localisation + host clarification template) on the real handler (single process, one seed=42 organ) + reproduces the load-bearing per-noun collapse under lesion. Lesion + flag-off + no-false-repair arms are decisive on the single wired seed; the role-lean robustness (two-inanimate -> AGENT; two-animate near-zero net lean -> generic) was measured across the six D4 seeds pre-wiring.
artifacts:
  - research/findings/raw/_gateB_repair_production_verify.json
---

# Gate-B / T1-6: an honest OTHER-REPAIR clarification on a low-comprehension turn (repair, don't dead-end)

**Status:** GO / WIRED. Genuine conversation is two-way. Before this, a turn the substrate could not parse
ended in a DEAD-END abstain ("my role-binding didn't resolve — I didn't follow that"). Now the brain instead
asks a TARGETED clarification that NAMES what did not resolve — the unresolved thematic ROLE
("...my role-binding didn't resolve the AGENT — which of them is doing the 'carry', the book or the cup?")
or the out-of-vocabulary TOKEN ("...I don't know the words 'wug' or 'glorp' yet — what do they refer to?").
This is a repair SEQUENCE, not just a flag. The DECISION to repair (comprehension failed) and the ROLE TARGET
(which role failed) are both genuinely-spiking reads off `cp_firing_states`; only the clarification wording is
a host language template.

## It COMPOSES the D4 monitor — the target is the SAME spiking read that triggers the abstain

The D4 comprehension organ already settles the two Wong-Wang sel pools (`sel_agent`/`sel_patient`, mutual
inhibition) per noun on the SEMANTIC (animacy+verbfit) cues and reads the per-noun agent-evidence
`a_i = sel_agent_rate - sel_patient_rate` off `cp_firing_states`; the sentence margin `|a0 - a1|` is the D4
abstain trigger. The repair reuses those SAME per-noun reads (a minimal additive extension,
`ComprehensionProductionOrgan.repair_target`) to localise the failure:

- **`sign(a0 + a1)` names the OVER-subscribed role.** A two-INANIMATE transitive ("the ball pushes the rock")
  drives BOTH nouns to PATIENT (animacy says inanimate->patient AND the asymmetric verb's verbfit agrees) ->
  a strongly NEGATIVE net lean -> the **AGENT** slot is the unresolved one. Measured across the 6 D4 seeds:
  two-inanimate net-lean `< -lean_margin` on **72/72** items -> role=AGENT robust. A two-ANIMATE transitive
  ("the wolf watches the owl", symmetric verb) has a near-zero net lean (`mean ~ -0.003`, within <!--derived-->
  `±lean_margin`) -> the substrate genuinely cannot say which role is over-subscribed -> an honest GENERIC
  role-swap clarification ("...which one does the 'watch' and which it happens to — which way round is it?").
- **`max(|a0|,|a1|)` (pair-max) confirms the roles are ACTIVE.** A covered transitive's pair-max is well above
  a build-calibrated floor (well-formed ~0.16, two-inanimate ~0.33, two-animate ~0.08 vs floor ~0.033); <!--derived-->
  under the D4 lesion (learned cue->role synapses zeroed) it collapses to **0.000** -> below the floor -> no <!--derived-->
  target -> the bare abstain. This is the load-bearing lesion behaviour (the WIRED arm is the verify's
  "lesion_collapses_to_bare_abstain" check).

- **OOV token** (a fully out-of-vocabulary transitive): the unknown word is named. This branch is a declared
  HOST-LEXICAL scaffold (the identity of an unknown word is a lexical fact, not a role-competition read),
  exactly the same class as curiosity's host topic extractor — NOT load-bearing on the spiking read.

The two repair thresholds (`role_floor`, `lean_margin`) are a fraction of the well-formed pair-max commitment,
calibrated at build FROM THE SAME battery read the D4 threshold uses — the per-noun read is folded into the
existing threshold loop, so it consumes the substrate RNG identically and the D4 margin is byte-identical.

## Verification (SYNCHRONOUS, numpy-CPU, through the REAL handler)

`research/runners/_gateB_repair_production_verify.py` (composer=rf, stub renderer, tiny-demo) — **6/6 PASS**:

| check | result |
|---|---|
| 2-inanim "the book carries the cup" | role=AGENT clarification, names book+cup, ends "?", loadbearing=spiking_role_evidence |
| 2-animate "the wolf watches the owl" | generic role-swap clarification (net-lean 0.09 within margin), targeted, ends "?" |
| fully-OOV "the wug blickets the glorp" | names 'wug'/'glorp' (kind=oov, host_lexical) |
| comprehensible "the wolf bites the apple" | NO repair key (no false repair — D4 comprehends, no abstain) |
| LESION (`BRAIN_COMPREHENSION_LESION=1`) | 2-inanim -> pair-max collapses below floor -> the BARE abstain (repaired=false) |
| FLAG-OFF (`BRAIN_REPAIR=0`) | 2-inanim -> the BARE abstain, NO repair key |

**Byte-identical regression** (a 9-turn battery covering recall / moat-abstain / anaphora / affect / D2-surprise
/ D4-comprehensible / the repair turns; `git stash` the two tracked edits, re-run pristine, diff):

- **FLAG-OFF (`BRAIN_REPAIR=0`) == pristine byte-identical across ALL 9 turns** (0 differences). The additive
  code adds nothing when off — including the D4 `comprehension.calib` object (the repair thresholds are kept
  OFF `self.calib`) and the D4 margins (the per-noun read reuses the threshold loop, RNG-identical).
- **DEFAULT-ON == pristine on every non-abstain faculty** (recall / moat-abstain+curiosity / anaphora /
  D2-surprise / comprehensible-D4 all byte-identical); it differs ONLY on the turns where **pristine D4 already
  abstained** ("the book carries the cup", "the wug blickets the glorp", and the fully-OOV-parsed
  "i feel wonderful today"), where the bare abstain becomes the targeted clarification — the intended change.

## Moat safety

A clarification is unambiguously a QUESTION. It never asserts or confabulates a fact, never flips the abstain
into an answer, never enters the certainty band — the SAME safety class as curiosity's follow-up question and
the bare abstain it replaces (`abstained` stays True; only the surface text changes). The affect appraisal
still runs (the `affect` field is byte-identical on the affect turn); only the D4-abstain surface text moves.

## Honest residuals (declared — each rides an existing burn-down row)

- **OOV TOKEN branch is a HOST-LEXICAL scaffold, NOT load-bearing** on the spiking read (only the ROLE branch
  is). The identity of an unknown word is a lexical fact; a spiking novelty/unknown-word read is the next rung
  (same class as curiosity's host topic extractor + wh-frame).
- **Two-animate direction is UNDETERMINED by the substrate** — a symmetric-verb two-animate transitive has a
  near-zero net lean, so it gets a GENERIC (still targeted) role-swap question rather than a committed
  AGENT/PATIENT name. This is the honest read, not a fallback for a known answer.
- **The clarification WORDING is a fixed host language template** (a QUESTION frame), like curiosity's wh-frame
  and the body acting on motor output. Only the DECISION + the ROLE TARGET are brain-surfaced.
- **D4-SCOPE INHERITANCE:** the repair fires on exactly the D4-abstain set, so it inherits D4's competence
  scope — including its OOV-transitive edge cases (a 3-content-word feel-statement like "i feel wonderful
  today" parses as fully-OOV feel/wonderful/today and abstains). Tightening D4's scope is a D4 rung, not a
  repair one; the repair faithfully targets whatever D4 abstains on.
- **CO-RESIDENT:** the repair reuses the D4 organ's own SpikingRoleCompetition bridge (no new bridge added) —
  rides on the one-brain merge (burn-down #1), exactly as D4.

FUNCTIONAL CORRELATE, NOT phenomenal: this reports a comprehension-repair correlate (a targeted question shaped
by the role-competition read). It makes NO claim of subjective understanding.

Ledger key: `other-repair`. Finding artifact: `research/findings/raw/_gateB_repair_production_verify.json`.
