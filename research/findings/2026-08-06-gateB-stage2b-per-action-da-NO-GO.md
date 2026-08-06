---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2b-per-action-compartmentalised-dopamine-on-continuous-selector
backend: numpy+cupy
runner: research/runners/_vocal_gateb_stage2b_per_action_da.py
prereg: 2026-08-06-gateB-stage2b-per-action-da-PREREGISTRATION.md
builds-on: 2026-08-06-gateB-stage1-continuous-bg-selector-CONSTRUCTION-GO.md
surpasses-method-wall: 2026-08-06-gateB-stage2-local-reward-credit-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2b_per_action_da/numpy.json
  - research/findings/raw/gateb_stage2b_per_action_da/cupy_confirm.json
---

# Gate B Stage 2b: per-action compartmentalised DA fixes the weight-level credit bleed but does NOT pass the behavioural contingency gate

## Verdict

**STAGE2B_NO_GO** (earned: preconditions hold, the criteria are measured and
fail). Per-action compartmentalised dopamine (Cluster C v2) surpasses the exact
Stage-2 METHOD wall it was built for -- it makes reward-driven D1 potentiation
ACTION-LOCAL across trials at the weight level (rewarding action A grows ONLY
A's D1 route; the other stays at construction baseline) -- but that is a PROXY,
not the capability. The behavioural gate does NOT pass: the reward-count-matched
YOKED control learns identically to the contingent condition
(D_contingent == D_yoked), and same-brain reversal fails. A deeper, newly
isolated wall now blocks the GO: appetitive-only per-action DA under the
winner-take-all selector self-reinforces whatever action the brain already
emits. Reward-OFF at Stage-1 noise is byte-identical (weights + raster) to
Stage-1. Artifacts + provenance:
`research/findings/raw/gateb_stage2b_per_action_da/numpy.json`.

## What was built (brain-based, verified)

On the Stage-1 continuous selector, the two `proposal_c -> str_d1_c` routes are
plastic and credited by the substrate's three-factor rule, but the global DA
scalar is replaced by four per-action DA modulators (`dopamine_{N,E,S,W}`;
N->action0, E->action1). Each fires (`from_action_specific_reward`) ONLY for the
channel the body executed (`core_config.last_selected_action` = which motor pool
fired -- the body's neural motor read-out, the same body boundary the nav runner
uses, `g11_bg_runner.py:7694`). `str_d1_c` carries `action_index=c`, so
`cp_synapse_action_tag` tags its afferents; `compute_per_synapse_da_signal`
routes each channel's DA (conc - baseline) to ONLY its tagged synapses. The
three-factor rule converts `per_synapse_da(action) * eligibility` (neural
coactivity eligibility scoped to the D1 routes). No host RPE, no host argmax
LABELLING the credited synapses, no host weight/eligibility edit. A neural
exploration process (elevated OU noise on proposal AND striatal D1/D2
populations) de-locks pre-learning choice. Reward-OFF at Stage-1 noise is
byte-identical to `run_stage1` (weights + raster hash).

## The weight-level credit IS now action-local (the Stage-2 wall, surpassed)

Stage-2's global scalar credited BOTH channels' eligibility across trials
(cross-trial bleed), so the routes could not be told apart. Per-action DA fixes
this. On every scored seed, rewarding action 0 grows ONLY route 0 and leaves
route 1 at its ~40 construction weight; rewarding action 1 does the mirror
(per-item magnitudes, swap control = the opposite action's reward):

- seed 730601: reward0 -> d1(route0,route1)=(133, 40); reward1 -> (40, 56)
- seed 730602: reward0 -> (118, 40);  reward1 -> (40, 87)
- seed 730605: reward0 -> (227, 40);  reward1 -> (40, 88)

Even the two hard-locked seeds show the SAME action-local weight credit (730603:
reward1 -> route1=165, route0=40; 730604: reward0 -> route0=133, route1=40) -- so
credit specificity is independent of exploration; they simply never emit the
other action to be scored behaviourally. Cross-backend (cupy) reproduces both the
byte-identical equivalence and D_contingent == D_yoked (seed 730605: contingent
and yoked reach identical d1 route0=148.9, test target-rate 1.0 = 1.0).

Lesions confirm the acquired preference lives in that eligibility-gated route
(seed 730605, contingent test target-rate 1.0): acquisition lesion (neural
eligibility OFF, reward delivered identically) drops it to 0.55 (Delta 0.45 >=
0.15); expression lesion (restore the D1 routes to symmetric baseline before the
frozen test) drops it to 0.50 (Delta 0.50 >= 0.15). Both lesion criteria PASS.

## But the behavioural capability gate FAILS

- **Contingency (yoked) fails.** D_contingent = D_yoked on all 3 exploring dev
  seeds (per seed D_contingent = D_yoked = 1.0; the frozen GO needs
  D_contingent - D_yoked >= 0.20). steer passes 0/6 dev seeds (3/6 explore).
  Reward DECOUPLED from the action (yoked) collapses to the same action as reward
  contingent on it.
- **Reversal fails (strongest control).** Seed 730605: train action 0 ->
  P(A)=1.0; reverse reward to action 1 and continue -> P(B)=0.0 (need >= 0.60).
  The brain cannot switch to the disfavoured action.
- **Exploration is still gated.** Broadening OU noise to the striatal D1/D2 MSNs
  and raising its amplitude de-locks only the soft seeds (3/6 explore at OU 120);
  2/6 dev seeds stay hard-locked (pre-learning P(a0) in {0.0, 1.0}) at every OU
  amplitude 40..600 -- amplitude is the wrong lever, and very high noise destroys
  clean selection before it flips the hard-locked seeds.

## Root cause (the companion process we replaced with a constant)

Per-action DA credits the EXECUTED action (via `last_selected_action`), and the
DA signal is APPETITIVE-ONLY (`from_action_specific_reward` only raises
concentration above baseline; there is no dip below it). Under the Stage-1
winner-take-all selector with DENSE reward and a small baseline selection bias,
this is a rich-get-richer positive-feedback loop: the marginally-ahead channel is
selected more, so it is credited more, so it is selected still more, until it
wins absolutely. The loop runs identically whether reward is CONTINGENT on the
action or merely DECOUPLED-but-coincident (yoked), because on the frequently-
rewarded trials the brain is doing its biased action either way. Once a channel
locks, the other action is never emitted, never earns eligibility, and can never
be credited -- so reversal is impossible. The missing companion process is the
NEGATIVE arm of the teacher: a reward-EXPECTATION baseline that produces a DA DIP
(negative RPE) when an executed action goes UNREWARDED, depressing the
over-selected route (D1-LTD) and breaking the runaway.

## Quantified residual

- D_contingent - D_yoked (exploring seeds) = 0.000 (need >= 0.20); steer 0/6.
- Reversal P(B) after B-phase = 0.00 (need >= 0.60).
- The genuine advance (banked): weight-level action-local credit (routes above)
  and both lesion criteria (Delta 0.45 / 0.50 >= 0.15). The missing capability is
  behavioural contingency + reversal, gated by appetitive-only self-reinforcement
  under a winner-take-all selector, NOT by weight-level credit specificity.

## Exact next mechanism (biology-grounded, in-substrate, no host shortcut)

1. **Opponent / bidirectional credit via a reward-EXPECTATION baseline (negative
   RPE).** Give the three-factor signal a learned expectation so an UNREWARDED
   execution yields DA BELOW baseline -> D1-LTD on the over-selected route (the
   substrate ships `reward_aversive_scale` + the D1/D2 sign array
   `enable_d1_d2_asymmetry`, both currently OFF; the baseline must be a NEURAL
   value/critic estimate, not a host EMA, to stay brain-based). This is the arm
   that lets contingent DIVERGE from yoked (non-target executions are punished)
   and makes reversal possible (the now-unrewarded dominant action is depressed).
2. **Sustained tonic-DA-modulated exploration** strong enough to keep sampling
   the disfavoured action after a transient lead, so a hard-locked channel can be
   escaped -- amplitude-only OU cannot (measured 40..600 pA).

Closure is deferred to a METHOD (appetitive-only per-action DA under a WTA
selector), NOT the CAPABILITY: Stage-2c reopens action-specific reward credit
with an opponent (negative-RPE) signal + sustained exploration on the same
one-brain substrate.
