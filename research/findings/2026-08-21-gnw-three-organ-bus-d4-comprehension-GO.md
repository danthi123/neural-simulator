---
type: finding
status: live
date: 2026-08-21
mechanism: gnw-workspace
lane: integration
seeds: [42]
seed-waiver: production-WIRING verify (deterministic combine-level, seed 42) of a faculty that composes three ALREADY-6-seed-GO organs — the D4 comprehension monitor (AUC 1.000, 6/6), the N-organ consensus hop (6/6), the 2-organ bus (6/6); this is not a fresh capability claim (mirrors the 2-organ production verify, also 1-seed at the combine level). The 6-seed evidence is the composed faculties' own GOs.
instrument: research/runners/_gnw_three_organ_bus_verify.py — production-wiring verify of the 3rd organ (the 6/6-GO D4 comprehension monitor) added as a genuinely-distinct load-bearing voter on the DEFAULT-ON 2-organ GNW ignition bus, through the REAL production ChatBrain.gate (numpy-CPU, rf recall). The underlying faculties it reuses are 6-seed GO (D4 comprehension, N-organ consensus, 2-organ bus).
runner: research/runners/_gnw_three_organ_bus_verify.py
external: NO-EXTERNAL-NEEDED — an in-repo composition of three already-GO in-repo spiking organs (recall + surprise + comprehension) via the already-GO N-organ consensus hop; the step is production wiring, not a literature question.
artifacts:
  - research/findings/raw/_gnw_three_organ/production_verify_numpy.json
---
# GO: the GNW ignition bus commits a recall only when it RECALLS it, is NOT surprised by it, AND COMPREHENDED it — a THIRD genuinely-distinct organ (D4) added as a load-bearing consensus-veto (default-OFF)

Artifact: `research/findings/raw/_gnw_three_organ/production_verify_numpy.json`

**One line.** The production GNW bus already commits a recall by the COINCIDENCE of organ A (spiking recall) + organ B (the spiking surprise/expectation-violation monitor). This adds a THIRD genuinely-distinct spiking organ — organ C, the 6/6-GO D4 COMPREHENSION monitor (a Wong-Wang `SpikingRoleCompetition` sel-pool WTA read off `cp_firing_states`) — as a real voter behind a NEW default-OFF flag. The workspace now ignites — the brain commits — ONLY when it RECALLS ∧ is NOT surprised ∧ COMPREHENDED the recalled proposition. A recall the brain retrieved and was not surprised by, but whose thematic ROLES its own substrate could not resolve, is now VETOED (abstain) — a decision the 2-organ bus could not make.

## The mechanism (a 3-way AND = a consensus-veto)
- **organ A** — spiking RECALL: `composer.query_patient(agent, action) -> cand` (FHRR phasor unbind).
- **organ B** — spiking EXPECTATION-VIOLATION monitor (`SurpriseProductionOrgan`, an Izhikevich predictive-coding mismatch circuit): confirms `cand` against its OWN learned expectation `e_B[(agent,action)]` by reading `cp_firing_states[surprise]`. CONFIRM (<< threshold) -> vote; SURPRISE (> threshold) -> withhold.
- **organ C** — spiking COMPREHENSION monitor (`ComprehensionProductionOrgan`, the 6/6-GO D4 faculty; a THIRD distinct substrate, a Wong-Wang `SpikingRoleCompetition` whose `sel_agent`/`sel_patient` WTA pools are read off `cp_firing_states`): it reconstructs the transitive `(agent, action, cand)` from the recall candidate (the question "what does {agent} {action}?" is a WH-gap the D4 monitor cannot score directly, so organ C scores the full PROPOSITION the brain is about to commit) and reads the SEMANTIC-cue-driven sel-pool margin `|agentEv(agent) - agentEv(cand)|`. HIGH (roles decisively separate) -> CORROBORATE (vote); LOW (role-ambiguous) -> WITHHOLD (the comprehension VETO). Out of the monitor's cue-lexicon COMPETENCE (a real-but-untabled word the brain knows) -> DEFER (corroborate), so organ C never false-vetoes a recall it is not competent to judge (the D4 declared vocab-ceiling residual).

Each organ writes a SUBTHRESHOLD drive `d_sub` into the shared K-slot GNW workspace via `norgan_hop`; agreeing organs ACCUMULATE on slot(cand). `d_sub = D_SUB_UNANIMITY[3]` is the calibrated Q=3 UNANIMITY drive: on the shared production workspace bridge, 2 votes = 2000 pA stays subthreshold (rate 0.030 < THR 0.167) while 3 votes = 3000 pA ignites (rate 0.333) — so slot(cand) crosses the ignition knee ONLY when ALL THREE organs vote. Any one organ withholding leaves it at <= 2·d_sub -> the workspace ABSTAINS. The AND-over-three-distinct-organs is the neuronal ignition THRESHOLD (WTA + NMDA sustain), not host control flow. <!--derived-->

## What the production verify proves (real `ChatBrain.gate`, numpy-CPU, rf recall — 1 seed, SYNCHRONOUS)
Runner `research/runners/_gnw_three_organ_bus_verify.py`; artifact `research/findings/raw/_gnw_three_organ/production_verify_numpy.json`. Verdict **GO** (all preconditions pass; the low-comprehension abstain is 100% attributable to organ C's veto, intact vs lesion).

- **(A) OFF byte-identical.** With `BRAIN_GNW_3ORGAN` unset, `install_three_organ_gate` is a no-op AND a runtime flag-flip-off makes the wrapper delegate to the DEFAULT-ON 2-organ gate — byte-identical on all 10 panel queries (covered recall + moat + self/identity + open-ended). Flipping the flag changes the MECHANISM, not the off-path behaviour. <!--derived-->
- **(B) ON, load-bearing, SELECTIVE.**
  - HIGH-comprehension stored query (`what does dog eat?` -> `dog eat apple`, a taught well-formed transitive): organ C's spiking margin 0.338 >= threshold 0.249 -> organ C ACTIVELY votes -> 3 votes -> the 3-organ decision == the 2-organ decision (`apple`), no behaviour change. <!--derived-->
  - Out-of-competence stored queries (`brain use spikes`, `brain learn words`, `brain store memory` — real vocabulary outside the toy cue lexicon): organ C DEFERS (corroborates) -> commit unchanged. <!--derived-->
  - LOW-comprehension probes where recall ∧ ¬surprise WOULD commit: `what does dog chase?` (`dog chase cat` — two-animate + a symmetric verb: content cannot say who chases whom; margin 0.108) and `what does cat eat?` (`cat eat fish` — the toy lexicon marks `fish` animate so verbfit fights animacy; margin 0.138), BOTH < threshold 0.249 -> organ C WITHHOLDS -> only 2 votes -> the 3-organ bus ABSTAINS (`consensus_veto_organ_c_low_comprehension`) while the 2-organ bus COMMITS (`cat` / `fish`). This is the new capability the 2-organ bus could not deliver. <!--derived-->
- **(C) LESION severs it.** `BRAIN_GNW_3ORGAN_ORGANC_LESION=1` silences organ C's veto (it corroborates unconditionally, its `cp_firing_states` read bypassed) -> the Q=3 consensus reduces to organ A + organ B (the 2-organ decision) -> BOTH low-comprehension abstains REVERT to the 2-organ commit (`cat` / `fish`, n_votes 2 -> 3). The veto is attributed to organ C's active spiking participation, not a host `if margin < x` (a host branch would still fire when the organ is lesioned). <!--derived-->
- **(D) MOAT preserved.** No unstored (`what does fish fly?`) or inconsistent (`what does cat chase?`) query is committed on EITHER arm — organ A misses -> abstain by construction, on both the 2-organ and 3-organ paths. <!--derived-->

## Why organ C is GENUINELY DISTINCT (the caveat the family tracks)
The N-organ bus's caveat #1 (three reads all from the composer) was closed by organ B (a genuinely different Izhikevich surprise substrate). Organ C is a THIRD distinct substrate and a THIRD distinct property: recall RETRIEVES the patient (FHRR unbind), surprise CHECKS it against a learned Izhikevich expectation, comprehension reads whether the (agent, action, cand) proposition's thematic ROLES RESOLVE via a Wong-Wang sel-pool WTA. On `dog chase cat` the three disagree in kind: recall says "cat", surprise says "matches my expectation", comprehension says "I cannot resolve who chases whom" — the veto is a property neither recall nor surprise captures. The margin is a `cp_firing_states` sel-pool read (the host `_semantic_contrast` dot-product is never called); its SELECTIVITY (why it vetoes `dog chase cat` but votes `dog eat apple`) is caused by the learned cue->role spiking competition — the 6/6-GO D4 de-risk (`2026-08-12-comprehension-production-monitor-wired-into-gate-b`, AUC 1.000, synaptic lesion -> 0.500) proves that discrimination collapses when those learned synapses are zeroed while the host cue VALUES are byte-identical. <!--derived-->

## Honest residuals (declared; ride existing burn-down items)
1. **CO-RESIDENT.** Organ C runs on its OWN `SpikingRoleCompetition` bridge, ALONGSIDE organ A's composer + organ B's surprise circuit + the shared P1.2 workspace — not merged onto ONE bridge. Rides the one-brain merge (burn-down #1), exactly as organ B and the affect organ do.
2. **CUE-LEXICON CEILING.** Organ C's competence + margin are bounded by the toy `ANIMACY`/`VERB_SELECTS` lexicon. `cat eat fish` is vetoed because that lexicon marks `fish` animate (verbfit conflict), not from a deep semantic judgment — a declared D4 residual. On real-but-untabled vocabulary organ C DEFERS (no false veto). A learned/graded cue lexicon is the D4's own mapped next rung.
3. **WH-GAP.** Organ C scores the reconstructed PROPOSITION `(agent, action, cand)`, not the raw WH-question (which the D4 monitor cannot score directly). Scoring the proposition the brain is about to commit is the reasonable design; a comprehension read of the question's own gap is the follow-on.
4. **DEFAULT-OFF + 1-seed WIRING verify.** This is a production-integration GO of an already-6/6-GO faculty (like the 2-organ production verify, which is also 1-seed at the combine level); the 6-seed evidence is the D4 comprehension GO + the N-organ consensus GO + the 2-organ bus GO it composes. A default-ON promotion (owner review) would need the same broad no-regression soak the 2-organ flip earned.

## Files
- Bus: `webapp/gnw_three_organ_bus.py` (`three_organ_combine`, `three_organ_gate_via`, `install_three_organ_gate`, `organc_lesion_on`; reuse-by-import of `gnw_two_organ_bus` + `_gnw_norgan_bus_derisk.norgan_hop` + `comprehension_production_organ` — NO `sim/` edit).
- Wiring: `webapp/server.py::brain_chat` — a flag-guarded (`BRAIN_GNW_3ORGAN`, default-off) `install_three_organ_gate(chat)` after the 2-organ install.
- Verify: `research/runners/_gnw_three_organ_bus_verify.py`; artifact `research/findings/raw/_gnw_three_organ/production_verify_numpy.json`.
- Composes: `2026-08-20-gnw-two-organ-bus-DEFAULT-ON-in-production-both-backends` (organ B), `2026-08-13-gnw-norgan-ignition-bus-substrate-combines-N-organ-reads` (the consensus hop), the 6/6-GO D4 comprehension monitor.
