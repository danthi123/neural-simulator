---
type: finding
status: live
date: 2026-10-26
lane: integration-first (stage exits on the calendar)
mechanism: Stage-2 (CHILD) exit gate on the integrated production brain, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage2_exit_gate.py
artifacts:
  - research/findings/raw/_stage2_exit/gate/aggregate.json
---

# Stage-2 (CHILD) exit gate: result (2026-10-26)

Prereg: research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md (gate revision 43763e637bbda75fddd045c30ebef8ac0c6b4603). Scored only by `python -m research.runners.stage2_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage2_exit/gate/aggregate.json.

**Stage-2 exit verdict: FAIL** (primary criteria: C1 FAIL, C2 FAIL, C3 FAIL, C4 FAIL, C5 PASS)

| group | criterion | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|
| conv | C1 | FAIL | 0/6 | lesion_mouth 6/6 | lesion_mouth 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| infer | C2 | FAIL | 6/6 | lesion_chain_route 6/6 | lesion_chain_route 6/6 | 0/6 | permuted exposure tracked on 0 of 6 seeds (not exposure-driven) |
| ask | C3 | FAIL | 0/6 | lesion_curiosity 6/6, lesion_graded 6/6 | lesion_curiosity 6/6, lesion_graded 6/6 | 0/6 | capability absent on 6 of 6 seeds (correct per order: registered 0 of 6, reversed 0 of 6) |
| emo | C4 | FAIL | 5/6 | lesion_affect 5/6 | lesion_affect 6/6 | 5/6 | capability absent on 1 of 6 seeds |
| inner | C5 | PASS | 6/6 | lesion_workspace 6/6, lesion_selfinit 6/6 | lesion_workspace 6/6, lesion_selfinit 6/6 | 6/6 | - |

Sub-clauses (seeds true / seeds defined; reported, never a pass condition alone):
- conv: sub_transitive 0/6, sub_prepositional 0/6, sub_passive 0/6
- infer: conclusion 6/6, derived_flag 6/6, yesno_derived_moth 0/6
- ask: discriminates 5/6, lp_learnable 0/6, lp_unlearnable 0/6, reversed_all 0/6, reversed_discriminates 0/6, reversed_lp_learnable 0/6, reversed_lp_unlearnable 6/6, all_taught_moats_acquired 0/6
- emo: sign_correct_all 0/6
- inner: private_voices_stored_fact 6/6, inner_terminal_lynx 6/6, chain_hops_internal_ge2 6/6, private_selected_by_state 6/6

Named gaps (amendment A4 reporting rule, NEEDS-CLAUDE #17; printed for a C1 FAIL, whichever sub-clause failed):
- C1 (conv): wh-locative question parse; locative frame storage (the teach echo drops the preposition); discourse elaboration for 'tell me more'

Per-seed curiosity ask-rates (novel-topic gaps vs taught-topic moats; the bar is gap >= 2/3 and gap >= 2x moat; amendment A4: every clause in BOTH orders, a taught moat not acquired is undefined):
- seed 42: gap 0.6666666666666666, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- seed 43: gap 0.6666666666666666, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- seed 44: gap 0.6666666666666666, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- seed 100: gap 0.6666666666666666, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- seed 101: gap 0.6666666666666666, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- seed 102: gap 0.3333333333333333, moat 0.0, learnable False, unlearnable-stops False, moats undefined ['m_goat']; defined
  - reversed order: gap 0.0, moat 0.0, learnable False, unlearnable-stops True, moats undefined ['m_goat']
  - da_lesion (informational, never counted, lesion held on 11 DA traces): gap 1.0, moat 0.0, learnable True, unlearnable-stops False
- C3 known architecture gaps (not the measured cause; read beside the da_lesion lines above): global DA engagement threshold; no per-topic learning-progress signal

Per-seed discrete-emotion read (category named per situation; the 2-D sign/arousal read beside):
- seed 42: named {'joy': 'joy', 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3
- seed 43: named {'joy': 'joy', 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3
- seed 44: named {'joy': 'joy', 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3
- seed 100: named {'joy': 'joy', 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3
- seed 101: named {'joy': None, 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3
- seed 102: named {'joy': 'joy', 'fear': 'fear', 'anger': 'anger', 'sadness': 'sadness'}; signs {'joy': 'positive', 'fear': 'neutral', 'anger': 'negative', 'sadness': 'negative'}; arousal {'joy': 'moderate', 'fear': 'high', 'anger': 'high', 'sadness': 'high'}; distinct 2-D reads 3


## Per-criterion reading (fixed sentences from HANDOFF section 8)

C1 (conv): FAIL, capability absent on 6 of 6 seeds; sub-clauses: conv: sub_transitive 0/6, sub_prepositional 0/6, sub_passive 0/6. The integrated brain does not show this Stage-2 ability at production defaults; the gap named in prereg section 8 applies. Named gaps (amendment A4): C1 (conv): wh-locative question parse; locative frame storage (the teach echo drops the preposition); discourse elaboration for 'tell me more'

C2 (infer): FAIL, not exposure-driven: the permuted exposure was tracked on only 0 of 6 seeds. `infer` informational: qp_lesion conclusion intact on 6/6 (the route did not need the spiking read), transitive_chat derived-moth on 0/6.

C3 (ask): FAIL, capability absent on 6 of 6 seeds; sub-clauses: ask: discriminates 5/6, lp_learnable 0/6, lp_unlearnable 0/6, reversed_all 0/6, reversed_discriminates 0/6, reversed_lp_learnable 0/6, reversed_lp_unlearnable 6/6, all_taught_moats_acquired 0/6. The integrated brain does not show this Stage-2 ability at production defaults; the gap named in prereg section 8 applies. C3 known architecture gaps (not the measured cause; read beside the da_lesion lines in the report block above): global DA engagement threshold; no per-topic learning-progress signal

C4 (emo): FAIL, capability absent on 1 of 6 seeds; sub-clauses: emo: sign_correct_all 0/6. The integrated brain does not show this Stage-2 ability at production defaults; the gap named in prereg section 8 applies. The per-seed discrete-emotion lines are copied verbatim in the report block above.

C5 (inner): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6.

The per-seed curiosity ask-rates (C3 amendment A4) are copied verbatim in the report block above.

## Claude amendment (2026-10-26, after the diagnoses)

The verdict and every number above stand. Three stated reasons are corrected; the evidence is in
research/findings/2026-10-26-stage2-fail-diagnoses.md (each diagnosis survived an independent skeptic).
- C2: the reason "not exposure-driven" is inaccurate. Permuted premise 1 ('the heron hunts the gnat') was never
  acquired. The default-ON D4 comprehension gate refused it as fully out-of-vocabulary on every seed (margin 0.0
  against 0.2; 8 facts stored instead of 9). So the permuted arm measured an acquisition refusal, not whether the
  conclusion follows the exposure. The literal rule still yields FAIL. The `transitive_chat` informational arm is
  void: its flag was already default-ON, and the route is not on the production rich path. The infer content path
  is deterministic, so 6/6 is one measurement repeated, not six replications.
- C3: the global DA gain is the measured cause of every failing clause except the missing per-topic stop. It is not
  "not the measured cause". With the gain pinned (da_lesion), gap is 1.0 and learnable is True on 6/6; lp_unlearnable
  still fails because no per-topic stop exists in the handler.
- C4: the s101 joy miss is a numerical artifact of the 1 ms explicit-Euler step. One low-capacitance appraisal
  neuron (53.6 pF) overshoots under strong inhibitory conductance and emits spurious spikes that recruit the sadness
  quench. The C4 lesion and permuted shortfalls are the same event, not independent failures.

## Claude amendment (2026-10-27): C5 caveat

The verdict and every number above stand; this limits how the C5 PASS may be cited.
- On dev seeds 7 and 8, C5 (inner) and recall verification are decided by the GNW ignition-bus workspace, and that
  workspace's ignition coincides with the numerical runaway of the old integrator. Measured alone, its ignition knee
  is the drive at which all 80 cells of the driven slot leave the valid membrane range, and its ignited rate 1/3 is
  that runaway (research/findings/raw/_stable_integration/bus_ignition_curve_s7.json). By intervention on dev seed 7,
  the bus alone on valid membranes inside an otherwise production brain loses recall verification and fails C5, and
  the bus alone on the old integrator inside an otherwise stable brain keeps both
  (research/findings/raw/_stable_integration/attribution_off_except_bus_vs_off_s7.json,
  attribution_on_except_bus_vs_off_s7.json).
- With valid membranes (SIM_STABLE_INTEGRATION=1, every bridge), C5 fails on dev seeds 7 and 8: the inner scorer
  `all` is true under the production integrator and false on valid membranes
  (research/findings/raw/_stable_integration/compare_s7.json, compare_s8.json).
- With the bus and the thought-swap workspace re-calibrated by their own rules on valid membranes
  (SIM_STABLE_RECAL=1, default OFF), C5 passes again on dev seeds 7 and 8, which are the seeds that re-calibration was
  fitted on (swap 7 and 8, bus 7), so this is an in-sample result
  (research/findings/2026-10-27-stable-recal-dev78.md, research/findings/raw/_stable_recal/dev78/dev_check.json).
- The gate seeds (42, 43, 44, 100, 101, 102) were not measured on valid membranes, with or without the
  re-calibration.
- So the 10-26 C5 PASS is cited only with this caveat: it was scored on an integrator under which the deciding
  workspace's ignition is a numerical runaway, and whether it holds on valid membranes at the gate seeds is unknown.
  Evidence: research/findings/2026-10-26-stable-integration-dev7.md and its raw under
  research/findings/raw/_stable_integration/.
