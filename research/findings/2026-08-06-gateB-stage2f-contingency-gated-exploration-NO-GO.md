---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2f-contingency-DeltaP-D1-minus-D2-confidence-gate
backend: numpy
runner: research/runners/_vocal_gateb_stage2f_contingency_gated.py
builds-on: 2026-08-06-gateB-stage2e-directed-novelty-exploration-NO-GO.md
surpasses-method-wall: 2026-08-06-gateB-stage2e-directed-novelty-exploration-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2f_contingency_gated/numpy.json
  - research/findings/raw/gateb_stage2f_contingency_gated/calibrate_numpy.json
---

# Gate B Stage 2f: a NEURAL contingency (DeltaP) gate = the per-action D1-minus-D2 spike contrast rescues the maximally-biased seed Stage-2e could not, but a per-seed-noisy critic + a below-gate credit-variance path leave steer at 4/6

## Verdict
<!--derived-->

**STAGE2F_NO_GO** (steer_seed_passes **4/6**, need >= 5; reward-OFF byte-identical
PASS; same-brain reversal PASS P(B) 0.00 -> 1.00; contingency-DeltaP gate is NEURAL
and load-bearing but per-seed unreliable; lesion contrast FAILS because the frozen
lesion seed 730605 is now itself a failing seed). Stage 2e (value-magnitude conf)
reached steer 4/6 / union-5/6, failing ONLY the maximally-biased seed 730604 (both
variants). Stage 2f replaces the confidence read-out with a NEURAL **DeltaP estimate =
the per-action D1-minus-D2 onset-spike contrast** (Hammond-1980 contingency): the D2
(indirect / NoGo) route, tagged per-action and made plastic here, has cp_d1_d2_sign=-1,
so a DA DIP (reward OMITTED) POTENTIATES str_d2_c and a BURST depresses it -> str_d2_c
becomes a per-action "reward-omitted" read-out (canonical A2A/D2 NoGo learning; Shen
2008; Collins-Frank OpAL). The gate is load-bearing (below) and it UNIQUELY rescues
730604 (Stage-2e's sole double-failure): **730604 D_contingent 1.00, D_yoked -1.00**
(a clean +2.0 steer). But it re-breaks 730605 and leaves 730602 -> steer stays 4/6
(a LATERAL move: which seeds fail changed). Artifacts:
`research/findings/raw/gateb_stage2f_contingency_gated/numpy.json`.

## Per-seed result (dev seeds 730601..730606)
<!--derived-->

| seed | base_p0 | D_cont | D_yoked | steer | dp_cont(t0,t1) | dp_yoked(t0,t1) | conf_y0 |
|------|---------|--------|---------|-------|----------------|-----------------|---------|
| 730601 | 0.57 | +1.00 | +0.00 | PASS | 0.194, 0.702 | 0.563, 0.647 | 1.00 |
| 730602 | 0.22 | +0.00 | +0.00 | FAIL | 0.192, 0.163 | 0.331, 0.377 | 0.66 |
| 730603 | 0.00 | +0.35 | +0.00 | PASS | 0.189, 0.232 | 0.105, 0.025 | 0.01 |
| 730604 | 1.00 | +1.00 | -1.00 | PASS | 0.236, 0.117 | 0.160, 0.054 | 0.17 |
| 730605 | 0.30 | -0.15 | +0.55 | FAIL | 0.208, 0.203 | 0.031, 0.156 | 0.00 |
| 730606 | 0.00 | +0.85 | -0.85 | PASS | 0.520, 0.232 | 0.173, 0.498 | 0.21 |

`D_contingent - D_yoked` mean = **0.725** (mean gate passes) but the per-seed steer
gate is the binding criterion. Two failures, of OPPOSITE character:

- **730602 (D_cont 0.00): contingent never EXPLOITS.** The DeltaP stays low in
  contingent (dp_c0 0.192, conf_c0 0.26) so the equalising drive never fades -> the
  brain keeps sampling both actions -> no commitment. The D1-D2 contrast never rose
  enough to signal "one action is contingently better", even though it is.
- **730605 (D_cont -0.15, D_yoked +0.55): yoked steers the WRONG way, BELOW the
  gate.** Here the gate did its job: conf_y0 **0.00** (the DeltaP correctly read yoked
  as non-contingent, drive stayed ON, sampling equalised). Yet D_yoked = +0.55: the
  reward-count-matched, action-DECOUPLED yoked reward still landed on a noise
  realisation that potentiated one D1 route enough to bias the frozen test. This
  steering is injected in the credit assignment BELOW the exploration gate, so no
  confidence signal can suppress it.

## The DeltaP gate is NEURAL and load-bearing, but a per-seed-noisy P(reward|action) proxy
<!--derived-->

- **Neural.** DeltaP = |net0-net1|/total with net[c] = str_d1_c - str_d2_c onset
  SPIKE counts (read like the motor read-out that moves the body). The D2 route learns
  omission via the substrate's OWN per-action DA dip x cp_d1_d2_sign three-factor rule
  (not a host P(reward|action) counter). **Declared residual (unchanged from Stage-2c):
  the conf->sigma / (1-conf) controller arithmetic and the scalar VALUE_GAIN critic
  gain are host constants** (the abstracted tonic-neuromodulator map). The contingency
  ESTIMATE has no host stand-in; its CONTROLLER does, as in every prior stage.
- **Load-bearing (measured on the lesion seed 730605).** `contingency_lesion` turns the
  D2 subtraction OFF (net = Vd1 alone = the Stage-2e value-magnitude signal): 730605
  goes to **D_yoked 1.00, conf_y0 0.977** (full yoked lock via spurious value-magnitude
  confidence). WITH the D2 subtraction (Stage-2f): conf_y0 **0.977 -> 0.00**, D_yoked
  **1.00 -> 0.55**. The D2 (reward-omitted) arm DOES suppress the spurious-confidence
  lock -- it just cannot remove the residual +0.55 that enters below the gate.
  `novelty_lesion` (directed drive OFF) reproduces the 2d lock (yoked_train_p0 0.975/0.0,
  D_yoked 1.0), so the directed drive is still load-bearing too.
- **Per-seed unreliable.** dp_yoked > dp_contingent on 730601 (0.56 vs 0.19, wrong
  direction) yet its outcome is fine; dp is right-ish on 730605 yet the outcome is
  wrong. The mean lift (0.725) is carried by the extreme seeds (730604, 730606). A
  single global VALUE_GAIN cannot keep the RPE properly signed across seeds'
  heterogeneous striatal firing rates (calibration diagnostic: retuning VALUE_GAIN
  0.010 -> 0.004 flipped the weight-level D1-D2 separation from ~1/3 to ~2/3 of seeds,
  still short), so on some seeds the D2 route grows by execution x omission, not by
  contingency.

## Frozen criteria (unchanged from the Stage-2 preregistration)
<!--derived-->

- Reward-OFF byte-identical to Stage-1 (weights + raster): **PASS** (the D2 plasticity +
  tagging gate on enable_reward, OFF in the equivalence build).
- Same-brain reversal (seed 730605): P(B) 0.00 -> **1.00** (>= 0.60, 1.00 > 0.00) PASS.
- Acquisition/expression lesions: cont 0.40 vs acq 0.45 (delta **-0.05** < 0.15) and vs
  expr 0.45 (**-0.05**) **FAIL** -- but uninterpretable: the frozen lesion seed 730605 is
  now itself a FAILING seed (D_cont -0.15), so its contingent barely clears baseline
  (0.40 vs 0.30) and the lesions cannot reduce what never learned. (Stage-2e's 730605
  was a passing seed.)
- Contingency steer_seed_passes: **4/6 (need >= 5) FAIL** -- passes {601,603,604,606},
  fails {602,605}.

## Quantified residual + exact next mechanism (no-defer)
<!--derived-->

The named surpass (a NEURAL D1-D2 DeltaP gate) IS built and IS load-bearing -- it
rescues the maximally-biased 730604 that no Stage-2e variant could, and it demonstrably
suppresses the spurious-confidence yoked lock (730605 conf_y0 0.977 -> 0.00). But the
per-action D1-D2 contrast estimates **P(reward|action)**, NOT the Hammond DeltaP =
P(reward|action) - P(reward|NO-action): it lacks a neural representation of the reward
rate when the action is WITHHELD, so it cannot subtract the base reward rate that
survives in the yoked credit assignment (730605's +0.55). And a single global critic
gain leaves the RPE mis-signed on some seeds (730602 never exploits).

**Next mechanism (Stage 2g, biology-grounded, in-substrate): a WITHHOLD baseline + an
adaptive critic.** (1) Add NO-ACTION (withhold) trials and a neural context/tonic value
that tracks reward in the ABSENCE of the credited action, so a TRUE contingency
V(action) - V(withhold) is formed neurally -- subtracting the base reward rate removes
the below-gate yoked steering (730605). (2) Replace the scalar VALUE_GAIN with a
homeostatic per-population critic normalisation (target-rate rescaling of the str_d1
value read-out) so the RPE stays properly signed across heterogeneous seeds (730602
exploits). Closure is deferred to a METHOD (a P(reward|action) proxy with a single
global gain), not the CAPABILITY.
