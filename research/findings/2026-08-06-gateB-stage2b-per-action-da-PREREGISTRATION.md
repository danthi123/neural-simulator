---
type: preregistration
status: preregistered
date: 2026-08-06
mechanism: gateB-stage2b-per-action-compartmentalised-dopamine-continuous-selector
runner: research/runners/_vocal_gateb_stage2b_per_action_da.py
builds-on: 2026-08-06-gateB-stage1-continuous-bg-selector-CONSTRUCTION-GO.md
surpasses: 2026-08-06-gateB-stage2-local-reward-credit-NO-GO.md
reuses-criteria-from: 2026-08-06-gateB-stage2-local-reward-credit-PREREGISTRATION.md
---

# Gate B Stage 2b: per-action compartmentalised dopamine (preregistration)

## Question

Stage 2's single GLOBAL dopamine scalar could not do action-specific credit:
D_contingent == D_yoked on all 6 dev seeds (`STAGE2_NO_GO`). Does the substrate's
per-action compartmentalised DA (Cluster C v2) make reward for action A
potentiate ONLY A's eligible D1 route, so contingent acquisition beats a
reward-count-matched yoked control across the dev seeds, with lesions and
same-brain reversal behaving?

## Mechanism under test (brain-based)

Four per-action DA modulators (`dopamine_{N,E,S,W}`; N->action0, E->action1)
whose `from_action_specific_reward` production fires ONLY for the channel the
body executed (`core_config.last_selected_action` = which motor pool fired, the
body's neural motor read-out, identical to how the nav body moves the agent,
`g11_bg_runner.py:7694`). `str_d1_c` regions carry `action_index=c`, so
`cp_synapse_action_tag` tags their afferent synapses; `compute_per_synapse_da_signal`
routes each channel's DA (conc - baseline) to ONLY its tagged synapses. The
three-factor rule converts `per_synapse_da(action) * eligibility_trace` (neural
coactivity eligibility, scoped to `proposal_c -> str_d1_c`), so only the executed
action's eligible D1 route is credited. No host RPE, no host argmax LABELLING
the credited synapses, no host weight/eligibility edit — the host only reports
the executed action + delivered reward (the body/environment boundary Stage-2
already used for reward).

A neural EXPLORATION process — elevated OU membrane-potential noise on the
proposal AND striatal (D1/D2) populations (background synaptic-bombardment /
intrinsic MSN variability) — de-locks pre-learning action choice so the
disfavoured action is sampled and can earn credit (4/6 Stage-2 dev seeds were
pre-learning seed-locked). It is a substrate noise knob, not host action
randomisation.

## Brain-based boundary (non-negotiable)

Unchanged from the Stage-2 prereg: reward is a DELIVERED environmental scalar
from the body's motor read-out; `last_selected_action` is the SAME body read-out
(the executed action), not a synapse-level credit assignment; credit assignment
is neural (action tag + coactivity eligibility + DA production/decay + the
three-factor conversion).

## Preregistered acceptance criteria (FROZEN — reused verbatim from the Stage-2 prereg)

Metric: bias-free swap differential D = P(a0 | reward a0) - P(a0 | reward a1) on
one brain, plus the reward-count-matched, action-decoupled yoked D_yoked (same
brain, independent noise). A seed is SCOREABLE (exploring) iff pre-learning
P(a0) in [0.20, 0.80]. A seed STEERS iff D_contingent >= 0.30 AND
(D_contingent - D_yoked) >= 0.20.

- **H1/H2 Acquisition + contingency.** GO requires steer on >= 5/6 dev seeds,
  D_contingent_mean(exploring) >= 0.30, and
  (D_contingent - D_yoked)_mean(exploring) >= 0.20.
- **H3 Acquisition lesion.** Neural eligibility OFF (reward delivered
  identically): contingent test target-rate - acq-lesion >= 0.15.
- **H4 Expression lesion.** Restore the D1 routes to symmetric baseline before
  the frozen test: contingent - expr-lesion >= 0.15.
- **H5 Same-brain convention reversal.** Train A then reward B in one brain: end
  P(B) >= 0.60 and P(B) > the A-phase-end P(B).
- **Byte-identical guard.** The reward-OFF build at Stage-1 noise
  (proposal-only OU) is byte-identical (weights + raster hash) to `run_stage1`.
- **Multiseed.** DEV seeds {730601..730606}; lesions + reversal on an exploring
  seed. Held-out {730701..730706} only after a dev GO.

## Verdict rule

`tools.verdict.Verdict`: require the interpretability preconditions (stage1
byte-identical, brain-delivered credit, >= 1 exploring seed); the GO criteria
above feed `decide(go=...)`. Honest NO-GO/UNDEFINED is a first-class deliverable
naming the next mechanism; the frozen criteria are NOT loosened and no host
shortcut is substituted to force a GO.

## Operating point (calibrated single-seed, sealed before scored seeds)

`reward_learning_rate`, reward window/magnitude, and the exploration OU sigma are
calibrated on the construction seed 730501 (both backends); the sealed dev /
held-out seeds are scored at the frozen operating point.
