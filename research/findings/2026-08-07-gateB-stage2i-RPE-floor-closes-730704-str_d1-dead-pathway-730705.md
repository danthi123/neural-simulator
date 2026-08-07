---
type: finding
status: partial
date: 2026-08-07
mechanism: gateB-stage2i-rpe-floor-bounded-critic-value-plus-str_d1-forced-sampling-refuted
backend: numpy
runner: research/runners/_vocal_gateb_stage2i_circuit_fixes.py
builds-on: 2026-08-06-gateB-stage2h-forced-sampling-extreme-bias-NO-GO.md
artifacts:
  - research/findings/raw/gateb_stage2i_circuit_fixes/smoke_numpy_730704.json
  - research/findings/raw/gateb_stage2i_circuit_fixes/smoke_numpy_730704_fixBonly.json
  - research/findings/raw/gateb_stage2i_circuit_fixes/smoke_numpy_730705.json
---

# Gate B Stage 2i: a bounded-critic RPE floor CLOSES the 730704 test-silence residual (steer passes); the 730705 residual is a structural str_d1 dead pathway, NOT a WTA lock, so its named fix is a NEW method

## Verdict

⛔ **FULL-VALIDATION CORRECTION (parent run, 2026-08-07): STAGE2I_NO_GO.** The single-seed smokes below OVERCLAIMED.
The full dev+held-out battery (`dev_fixB.json`, `heldout_fixB.json`) shows FIX B is a NET REGRESSION, not a fix:
**dev 5/6 → 4/6** (730601, 730602 now fail — the RPE floor weakened contingency on two previously-passing dev seeds;
FIX B is NOT inert as the smoke claimed) and **held-out stays 4/6** — 730704's NaN is gone but it STILL fails steer
on the full battery (the smoke's `D_contingent=1.0` did not hold), 730705's dead pathway persists. So FIX B trades a
held-out NaN-fix for dev regressions. NEXT: the RPE floor must be made NON-REGRESSIVE (per-seed/adaptive, engaging
only on the saturated tail) before it can help; then the 730705 dead pathway (Stage-2j) remains. The smoke-based
"CLOSES 730704" claim below is superseded by this full-battery verdict.

## Verdict (original smoke read — SUPERSEDED, see correction above)

**STAGE2I_PARTIAL (smoke, two 1-seed extreme-bias smokes).** Stage 2g held-out failed on
two seeds for two different reasons. **FIX B (a floor on the net RPE) CLOSES 730704**: it
recovers from a frozen NaN to clean actions on both targets with a steer-passing
D_contingent. **FIX A (str_d1 forced-sampling bias) is REFUTED by the substrate** for
730705: the failure is not a reward-potentiated WTA lock but a structural dead striatal
pathway on that seed's heterogeneity draw, and external current into an MSN population is
counterproductive, not a lever. The 730705 residual is precisely isolated and its next
method is named (MSN homeostatic intrinsic excitability -- a substrate change, Stage 2j).
Stage 2g remains byte-reproducible (FIX B is default-on but inert unless value_est is
pathologically inflated; FIX A is default-off, opt-in).

## FIX B -- bounded critic value = a floor on the net RPE (CLOSES 730704)

VERIFIED cause of 730704's post-training motor silence: the homeostatic critic
normalisation can inflate value_est up to VALUE_MAX=1.5, which EXCEEDS REWARD_MAG=1.0.
The Hammond-DeltaP baseline is value_est + v_withhold, so a REWARDED action gets net
RPE = REWARD_MAG - value_est < 0 -- it depresses its own route even when rewarded ->
runaway to silence. Fix: clamp ONLY the self-value component, value_est <=
REWARD_MAG - RPE_FLOOR (= 0.9), so a rewarded action's net RPE >= RPE_FLOOR > 0. Neural
grounding: a critic value cannot exceed the maximum obtainable reward, and the DA burst
to a delivered reward is never fully cancelled by expectation. v_withhold (the base-rate
subtraction) is left untouched, so the contingency and yoked cancellation are unchanged;
the cap only bites when value_est is inflated, so dev seeds with value_est <= 0.9 are
byte-unchanged.

Smoke (FIX B only, FIX A off --
`research/findings/raw/gateb_stage2i_circuit_fixes/smoke_numpy_730704_fixBonly.json`): count_c0 [25,14],
count_c1 [1,38] (both actions sampled, as 2g already did on this seed); test n_clean 20/20
(no freeze), test_rate 1.0/1.0 (no NaN); **D_contingent 1.0, D_yoked 0.0 -> STEER PASSES.**
Vanilla 2g (fixes off) on this seed: n_clean_c1 0, test_rate_c1 NaN, D_contingent NaN
(frozen). NB: running FIX A alongside FIX B on this seed inflated D_yoked to 0.857 <!--derived-->
(0.8571428571428571 in `smoke_numpy_730704.json`) -- FIX A perturbs the yoked runs, which
is why it is default off. FIX B ALONE is the clean fix.

## FIX A -- str_d1 forced-sampling bias (REFUTED by the substrate for 730705)

A per-population diagnostic on a fresh, post-baseline 730705 network refutes the task's
stated diagnosis (a reward-potentiated str_d1_0 WTA lock, breakable by biasing str_d1):

  1. the two proposal->str_d1 routes start SYMMETRIC (mean weight 40.03 vs 40.03) -- the
     baseline_p0=1.0 bias is INTRINSIC (the per-neuron heterogeneity draw), not
     reward-formed, so there is no early lock to anneal;
  2. EXTERNAL current into an str_d1 (MSN) population is COUNTERPRODUCTIVE, not a lever:
     on the normal seed 730601, +200 pA into str_d1_1 DROPS its firing 6 -> 0 and kills
     motor_1 (motor 146 -> 0);
  3. on 730705 str_d1_1 is intrinsically near-UNEXCITABLE -- 0 spikes at every direct
     drive tested (200-3000 pA). So motor_1 can never win normally, AND the
     proposal_1->str_d1_1 route can never gain three-factor eligibility (str_d1_1 never
     co-activates), so no reward can potentiate action 1: the pathway is a structural
     DEAD END on this seed. The only lever that made motor_1 win was inhibiting the
     incumbent MOTOR pool directly (motor_0 -3000 pA -> motor [1,12], winner 1) -- but
     that leaves str_d1_1 at 0, so no learning persists to test.

Consequently FIX A as built (excite under-sampled proposal + inhibit incumbent str_d1)
leaves 730705's counts byte-identical to 2g ([39,1] / [40,0], 32 forced trials, count1
still 0; `smoke_numpy_730705.json`) and D_contingent = 0.

## Quantified residual + next method (no-defer)

730704 is CLOSED by FIX B (structural test-silence -> steer pass). 730705's residual is a
structural str_d1 dead pathway from the heterogeneity draw: 1/12 held-out+dev seeds has an
action whose D1-MSN population does not spike under the network's normal drive. The missing
COMPANION PROCESS (per the wall reframe) is HOMEOSTATIC INTRINSIC EXCITABILITY plasticity:
a real MSN that never fires up-regulates its own excitability toward a target rate (Desai/
Turrigiano intrinsic homeostasis), which we froze as a constant. The next method (Stage 2j)
raises str_d1_1's intrinsic excitability toward a firing set-point by adjusting its
Izhikevich parameters (NOT by injecting current, which the diagnostic shows silences the
MSN), measured against the r0_d1 baseline the runner already records. Expected effect on
held-out: FIX B moves it 4/6 -> 5/6 (730704 recovered); 730705 needs Stage 2j.

## Reproduce (parent's full validation -- FIX B default, FIX A off)

    # dev (byte-repro guard on 2g behaviour where value_est<=0.9):
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._vocal_gateb_stage2i_circuit_fixes \
        --mode seeds --dev-seeds 730601 730602 730603 730604 730605 730606
    # held-out:
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._vocal_gateb_stage2i_circuit_fixes \
        --mode seeds --dev-seeds 730701 730702 730703 730704 730705 730706
    # full verdict (lesions + reversal + dev steer):
    SIM_BACKEND=cupy .venv/bin/python -m research.runners._vocal_gateb_stage2i_circuit_fixes \
        --mode full --dev-seeds 730601 730602 730603 730604 730605 730606
