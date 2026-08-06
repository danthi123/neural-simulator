---
type: preregistration
status: preregistered
date: 2026-08-06
mechanism: gateB-stage2-local-reward-credit-continuous-selector
runner: research/runners/_vocal_gateb_stage2_reward_credit.py
builds-on: 2026-08-06-gateB-stage1-continuous-bg-selector-CONSTRUCTION-GO.md
supersedes-lesson: 2026-08-03-neural-vocal-credit-gateB-v10-policy-eligibility-UNDEFINED.md
---

# Gate B Stage 2: local reward-credit learning on the continuous BG selector (preregistration)

## Question

Does biological reward-modulated plasticity (neural coactivity eligibility x a
delivered dopamine/reward scalar) on the corticostriatal D1 policy routes make
the Stage-1 continuous center-surround selector ACQUIRE a rewarded action, in a
way that is CONTINGENT on reward (not mere reward exposure), lesion-dependent,
reversible in one brain, and holds across a fresh multiseed development set?

## Why now / why this is not a v10 repeat

v10 was UNDEFINED because its stop-on-commit selector could not preserve one
exclusive action under sustained drive (the loser crossed later in every trial),
so action-local eligibility could not be scored. v10 DID establish the
<!--derived-->
eligibility tag is action-local (selected route 0.0267 vs other 0.0025 for D1);
these two numbers are quoted from the v10 UNDEFINED finding.
Stage 1 replaced the stop-on-commit protocol with a continuously-operating
selector that produces temporally-bounded actions with autonomous return to
tonic (the exclusivity v10 lacked). Stage 2 reopens the reward-credit question on
THAT selector. No host stop-on-winner, no reset.

## Brain-based boundary (non-negotiable)

- Reward is DELIVERED as an environmental scalar (`current_reward_signal`), set
  from the body's motor read-out (legit environment/body). The host reading which
  motor pool fired to decide the reward is the environment evaluating an action.
- CREDIT ASSIGNMENT is neural: eligibility is built from real pre/post spikes
  (`reward_eligibility_from_coactivity`, scoped to the proposal->D1 synapses),
  and the substrate's three-factor rule converts tag x reward into a weight
  change (`sim/bridge.py` ~L9928-10154). No host RPE formula, no host argmax over
  spikes assigning credit, no host-written eligibility or weight edit.
- The winner used for the metric is the neural motor read-out (body). It is NOT
  used to label which synapses receive credit.

## Mechanism under test

On the Stage-1 selector, the two `proposal_c -> str_d1_c` routes are made
plastic; eligibility is scoped to exactly those synapses. During training, each
trial runs one Stage-1 onset window; if the neural winner == the rewarded target
action, a positive reward scalar is delivered for a reward window during the
early gap while eligibility is still high. Three-factor: the SELECTED channel's
D1 route (the one whose D1 actually fired) carries the tag, so reward potentiates
only the selected route. Over trials the rewarded action's corticostriatal drive
strengthens, biasing the center-surround competition toward it.

## Preregistered acceptance criteria (locked before any scored run)

Metric: target-rate = P(neural motor winner == target action) over a FROZEN test
block of `TEST_TRIALS` trials (plasticity gain 0, reward off, no learning).
Chance = 0.5. All conditions share construction weights and symmetric routes at
t=0; only the reward contingency / lesion differs. Reported per dev seed and as
the seed mean.

- **H1 Acquisition.** Contingent test target-rate >= 0.70 (mean over dev seeds)
  AND strictly above the same brain's pre-training baseline target-rate.
- **H2 Contingency (reward-count-matched yoked).** The yoked control receives the
  SAME number of reward deliveries on the SAME trial indices as its contingent
  master, decoupled from its own action. Require contingent target-rate - yoked
  target-rate >= 0.15 (seed mean) AND yoked mean within [0.40, 0.60] (~chance).
- **H3 Acquisition lesion.** Training with the neural eligibility tag disabled
  (`reward_eligibility_from_coactivity=False`; reward delivered identically)
  yields test target-rate within [0.40, 0.60] and < contingent by >= 0.15.
- **H4 Expression lesion.** After contingent training, restoring the proposal->D1
  route weights to their symmetric construction baseline before the frozen test
  collapses test target-rate into [0.40, 0.60] and < trained contingent by
  >= 0.15 (the acquired preference is stored in that route).
- **H5 Same-brain convention reversal (strongest control).** In one brain, train
  target=A (acquire A: P(A) >= 0.65), then reverse reward to B and continue; end
  P(B) >= 0.60 and P(B) > the A-phase-end P(B). Proves the asymmetry is
  reward-assigned and reversible, not fixed wiring.
- **Multiseed.** DEV seeds = {730601..730606} (fresh; disjoint from construction
  730501 and seed-robustness 730501-730504). GO requires H1+H2 on >= 5/6 dev
  seeds and H3+H4+H5 on the construction seed both backends. Held-out seeds
  {730701..730706} are validated only AFTER a dev GO (next step), not here.

## Verdict rule

`tools.verdict.Verdict`: `require` each precondition (exclusivity preserved,
weights immutable outside D1 routes, zero GPi external current, reward brain-
delivered), `floor` H1 vs chance, `control` contingent-vs-yoked (H2),
contingent-vs-acquisition-lesion (H3), trained-vs-expression-lesion (H4),
B-after-reversal vs B-before (H5). `decide(go=...)` -> GO only if all land;
UNDEFINED if any precondition is unmeasured. Honest NO-GO/UNDEFINED (e.g. neural
reward-credit cannot beat yoked) is a first-class deliverable naming the next
biological mechanism; no host shortcut substituted to force a GO.

## Calibration (before scored seeds)

Single-seed (construction 730501), both backends: sweep `reward_learning_rate`,
reward window length, and reward magnitude to a per-reward Delta on the D1 route
that shifts selection without saturating or destabilising the fixed selector.
Calibration is not scored; the sealed dev/held-out seeds are.
