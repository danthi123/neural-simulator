---
type: finding
status: no-go
date: 2026-08-06
mechanism: gateB-stage2h-forced-sampling-epsilon-floor-pushpull-extreme-bias-exploration
backend: numpy
runner: research/runners/_vocal_gateb_stage2h_forced_sampling.py
builds-on: 2026-08-06-gateB-stage2g-hammond-deltap-dev-GO-heldout-NO-GO-critic-saturation.md
artifacts:
  - research/findings/raw/gateb_stage2h_forced_sampling/smoke_numpy_730704.json
  - research/findings/raw/gateb_stage2h_forced_sampling/smoke_numpy_730705.json
---

# Gate B Stage 2h: a proposal-level forced-sampling floor does NOT close the extreme-bias residual — the bottleneck is downstream, and one of the two held-out failures is not a sampling gap at all

## Verdict

**STAGE2H_NO_GO (smoke, method verdict).** The Stage-2g held-out NO-GO (4/6) was
attributed to ONE cause: on maximally-biased seeds (`baseline_p0 ∈ {0,1}`) the brain
never samples both actions. Stage 2h added the prescribed fix — a NEURAL forced-sampling
/ ε-floor that escalates the exploration drive past the 350 pA graded cap until the
under-sampled action fires. A 1-seed smoke on each extreme held-out seed (730704,
730705) REFUTES the method: no proposal-level drive configuration both (a) breaks the
extreme-bias winner-take-all AND (b) leaves the naturally-sampling seeds intact. The
premise is also partly wrong — only ONE of the two failures is a sampling gap.

## What was built (banked; brain-based, additive, 2g byte-preserved)

`_vocal_gateb_stage2h_forced_sampling.py` imports every Stage-2g leaf helper unchanged
(2g untouched) and adds `_ForcedSampler` + `_run_trial_2h`: while an action has < K=3
clean motor samples (after an 8-trial grace on the graded 2g drive), a **push-pull
competition bias** EXCITES the under-sampled action's `proposal_{u}` population and
INHIBITS the incumbent `proposal_{1-u}` — both external currents into proposal pops, a
count-based novelty read-out, exactly the 2e/2g pattern. Stays neural: the host only
biases which proposals compete; the brain's own WTA resolves the winner. Reward is an
env scalar. When both actions reach K the exact 2g graded drive resumes.

## Why it fails (VERIFIED against the substrate, per-trial)

**730705 (`baseline_p0=1.0`, always action 0) — a real sampling gap, but the lock is
DOWNSTREAM of the proposal layer.** Driving `proposal_1` alone to 10000 pA propagates to
the striatum (`str_d1_1` fires 2031 spikes) yet `motor_1` stays at **0** while `motor_0`
is saturated (~857) — the winner-take-all is locked at the reward-POTENTIATED
`str_d1_0 → motor_0` route, not at the proposal input. Adding `proposal_0` inhibition
breaks it ONLY on a FRESH network (motor[0,~860], clean); once the baseline block + a few
reward trials potentiate the action-0 route, no proposal-level push-pull flips it
(`count1` reaches at most 1/40; the single early win is OU-noise luck before potentiation).
Escalating the excitation is COUNTERPRODUCTIVE: above ~1250 pA the driven `proposal_1`
goes silent (`str_d1_1 → 0`, consistent with depolarization block), so the drive is
capped at 1200 pA — at which point it is too weak to break the lock (`count1` = 0, i.e.
identical to 2g).

**730704 (`baseline_p0=0.0`, always action 1) — NOT a sampling gap.** Vanilla 2g already
samples both actions (`count=[13,26]`); its NaN is a training-induced TEST-TIME motor
SILENCING — after training, every test trial reads `motor=[0,0]` despite a healthy
pre-training baseline (`n_clean=13`). Forced sampling is the wrong instrument here: any
engagement fights the network, PREVENTS the natural sampling 2g achieves, and turns the
one clean direction (target0, 2g `n_clean=2`, rate 1.0) into a NaN. The grace period
limits but does not remove this (action 0 is naturally unsampled in the first 8 trials on
this seed, so the floor still engages and disrupts).

## Smoke evidence

FLOOR_ON vs FLOOR_OFF(=2g), contingent, per target, in
`research/findings/raw/gateb_stage2h_forced_sampling/smoke_numpy_730705.json` and
`research/findings/raw/gateb_stage2h_forced_sampling/smoke_numpy_730704.json`: 730705 t1
ON `count(40,0)` = OFF `count(40,0)` (inert); 730704 t0 ON `count(2,9)` test NaN vs OFF
`count(15,24)` test clean rate 1.0 (regressed). `SMOKE_PASS_both_actions_sampled` is not
met on 730705 t1 (`count1`=0); `floor_attribution_of_coverage` = 0 (the floor adds no
coverage over the 2g graded drive on the extreme seeds).

## Next mechanism (no-defer — a method verdict, not a capability abandonment)

The exploration drive lives at the wrong stage. Two distinct residuals, two methods:
(1) **730705 downstream lock** — bias the competition where it is decided: an inhibitory
drive at the incumbent's `str_d1`/GPi (BG output disinhibition) rather than its proposal,
or a homeostatic RESET/anneal of the potentiated D1 route so forced sampling acts before
the lock forms (front-loaded sampling in the first < K·2 trials, before reward potentiates
action 0). (2) **730704 test-silence** — this is a CRITIC/RPE calibration defect, not
exploration: the Hammond-ΔP baseline (`value_est + v_withhold`) over-subtracts on a
saturated seed and net-DEPRESSES the route to silence; the fix is a floor on the net RPE
or a rate homeostat on the motor pool, verified by the pre/post `motor=[0,0]` read-out.
The Stage-2g contingency mechanism remains correct and complete; the forced-sampling
method is banked as insufficient for the extreme-bias coverage residual.
