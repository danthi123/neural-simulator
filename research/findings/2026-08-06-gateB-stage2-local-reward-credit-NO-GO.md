---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2-local-reward-credit-global-DA-three-factor-on-continuous-selector
backend: numpy+cupy
runner: research/runners/_vocal_gateb_stage2_reward_credit.py
prereg: 2026-08-06-gateB-stage2-local-reward-credit-PREREGISTRATION.md
builds-on: 2026-08-06-gateB-stage1-continuous-bg-selector-CONSTRUCTION-GO.md
artifacts:
  - research/findings/raw/gateb_stage2_reward_credit/numpy.json
  - research/findings/raw/gateb_stage2_reward_credit/cupy_confirm.json
---

# Gate B Stage 2: global-dopamine three-factor reward-credit is NOT reward-contingent on the continuous selector

## Verdict

**STAGE2_NO_GO for the global-scalar three-factor method** (earned NO-GO, not
UNDEFINED: preconditions all hold and the criteria are measured and fail). The
plasticity is REAL and lesion-dependent, but it is NOT reward-CONTINGENT:
delivering reward DECOUPLED from the action (yoked) produces behaviourally
IDENTICAL learning to reward CONTINGENT on the action. Equivalence to the Stage-1
selector is byte-identical (weights + raster). Artifact + provenance sidecar:
`research/findings/raw/gateb_stage2_reward_credit/numpy.json`.

## What was built (brain-based, verified)

On the Stage-1 continuous center-surround selector, the two `proposal_c ->
str_d1_c` routes were made plastic and reward-credited via the substrate's
three-factor rule (`sim/bridge.py` ~L9928-10154):
`Delta_w = reward_learning_rate * (current_reward_signal - reward_baseline) *
eligibility_trace`. Eligibility is NEURAL (coactivity on real pre/post spikes,
scoped to the D1 routes; v10 established this tag is action-local). Reward is
delivered as an environmental scalar from the body's motor read-out. No host
assigns credit, edits eligibility/weights, or argmaxes spikes to label the
credited route. The reward-OFF build is asserted byte-identical (weights + raster
hash) to Stage-1 `run_stage1`.

## The plasticity works; the CONTINGENCY does not

Metric: bias-free swap differential D = P(a0 | reward a0) - P(a0 | reward a1) on
one brain (needs >= +0.30 to show reward steers selection). Reward-count-matched,
action-decoupled yoked (same brain, independent noise) gives D_yoked.

- **Plasticity is real and lesion-dependent.** On the exploring seed 730605
  (baseline P(a0)=0.25), contingent training on target 0 raised test P(a0) to
  0.90; the acquisition lesion (neural eligibility OFF) left it at 0.30, and the
  expression lesion (reset the D1 route) dropped it to 0.20. So reward-modulated
  D1 potentiation does change selection, and removing eligibility or the learned
  route abolishes it.
- **But it is NOT reward-contingent.** D_contingent == D_yoked on ALL SIX dev
  seeds (per-seed identical), mean D_contingent = D_yoked = **-0.358** <!--derived--> (floor was
  +0.30). Reward decoupled from the action produced the SAME learning as reward
  contingent on it. steer-criterion passes: **0/6**. Same-brain reversal FAILED:
  after acquiring action 0 (P(A)=0.85), rewarding action 1 left P(B)=**0.00** --
  the brain cannot switch to the disfavoured action.
- **Cross-backend (cupy) confirms it, and rules out exploration as the sole
  cause.** Equivalence is byte-identical on cupy (weights + raster). On cupy seed
  730605 is BALANCED (baseline P(a0)=0.50) yet still collapses to one action
  regardless of reward target: D_contingent = D_yoked = 0.00 (P(a0|r0)=0.00,
  P(a0|r1)=0.00). So even with good exploration the global-DA method shows ZERO
  contingency -- the credit-specificity failure is primary, not just exploration.
- **Exploration is the gating constant.** 4 of 6 dev seeds are seed-LOCKED
  pre-learning (baseline P(a0) in {0.0, 0.0, 1.0, 1.0}); only 1/6 explores both
  actions. A locked selector never emits the disfavoured action, so that action
  never earns eligibility or credit. Per-seed rewarding a0 vs a1 left selection
  driven by the intrinsic bias, not the reward (e.g. 730604 base 1.0: P(a0|r0)
  0.05, P(a0|r1) 1.00 -> D -0.95, and D_yoked identical).

## Metric note (prereg refinement, disclosed)

The prereg scored raw target-selection rate vs a contingent baseline. On seeing
that pre-learning selection is seed-LOCKED (4/6 seeds ~100% one action), raw
target-rate is confounded by intrinsic bias, so the primary contingency metric
was strengthened to the BIAS-FREE swap differential (reward a0 vs a1 in one
brain) plus the reward-count-matched yoked control. This is a stricter test than
the prereg (it cancels intrinsic bias), not a relaxation; the yoked equality
result would fail the prereg's contingent-vs-yoked criterion too.

## Root cause (the companion processes we replaced with constants)

1. **A single GLOBAL dopamine scalar cannot do action-specific credit.** The
   scalar x eligibility term converts BOTH channels' traces (within- and
   cross-trial bleed), so credit is not action-local over trials and the yoked
   (decoupled) control learns identically. The substrate ALREADY documents this
   exact failure (`sim/neuromodulators.py` L11 -> `2026-04-24-session-c.md` sec 4,
   "why eligibility-only modulation fails") and ships the fix (Cluster C v2).
2. **Exploration is a constant, not a process.** Fixed OU noise is a poor proxy
   for the animal's tonic-DA-modulated action variability; without it the
   disfavoured action is never sampled, so it can never be credited.

## Quantified residual

- D_contingent mean (6 seeds) = -0.358; D_yoked mean = -0.358; difference = 0.000 <!--derived-->
  (need contingent - yoked >= 0.20). Exploring-seed subset (1 seed): both -0.10.
- steer passes 0/6; reversal P(B) after B-phase = 0.00 (need >= 0.60).
- The genuine positive: reward-modulated D1 plasticity is intact and
  lesion-gated (contingent 0.90 vs acq-lesion 0.30 vs expr-lesion 0.20). The
  missing capability is action-SPECIFIC contingency, not plasticity itself.

## Exact next mechanism (biology-grounded, in-substrate, no host shortcut)

1. **Per-action compartmentalised dopamine (Cluster C v2, already built).**
   Register per-action DA modulators (`from_action_specific_reward`,
   `sim/neuromodulators.py` L148-164, L1054-1098) + populate `cp_synapse_action_tag`
   on the D1 routes, so `compute_per_synapse_da_signal` converts ONLY the selected
   action's route eligibility -- the fix the codebase built for exactly this
   global-DA failure. Re-run the same swap + yoked + lesion + reversal battery.
2. **A neural exploration/variability process** (tonic-DA-modulated action
   variability during learning) so the disfavoured action is sampled enough to be
   credited -- restoring the companion process the fixed OU noise proxied.

Closure is deferred to a METHOD (the global scalar), NOT the CAPABILITY: Stage-2
reopens with per-action DA + an exploration process on the same one-brain
substrate.
