---
type: finding
status: negative
date: 2026-08-07
mechanism: replay-consolidation-v7-self-calibration-step0
runner: research/runners/_replay_v7_step0_sparsity_probe.py
builds-on: research/findings/2026-08-06-replay-cortical-consolidation-v6-multiseed-NO-GO-operating-point-overfit.md
spec: research/findings/raw/_replay_selfcalibration_scoping.md
artifacts:
  - research/findings/raw/replay_v5_sfa_order/replay_v7_step0_dev.json
  - research/findings/raw/replay_v5_sfa_order/replay_v7_step0_dev.json.prov.json
---

# STEP-0 closability gate is NO-GO: a single label-free regime statistic cannot set the retest competition point

**Verdict: STEP0_NOGO_SINGLE_STATISTIC_INSUFFICIENT.** The proposed replay-consolidation self-calibration (an emergent homeostatic integral controller driving a label-free WTA-sparsity / E-I set-point `S*`, per `_replay_selfcalibration_scoping.md`) does NOT pass its own STEP-0 closability gate. Per the scoping doc's explicit stop rule, the controller (Step 1) was NOT built. This is a verdict on the *label-free-single-statistic METHOD*, not on the interference-control capability.

## What Step 0 tested

The frozen v6 order-STDP runner (byte-identical mechanism/config, INTACT condition) was instrumented additively to log, per development seed (414/415/410), a set of label-free competition-regime statistics `S` of the `cortical_target` population at retest, alongside the scored false-recall. `S` is computed ONLY from the raw per-neuron spike-count vector and the STRUCTURAL assembly SIZE — never assembly identity, seed, correct/wrong labels, or the false-recall metric (asserted in-code by signature check). Candidates: participation ratio (`pr_eff`, `pr_frac`), Gini concentration, top-assembly spike-share, active fraction. Closability requires a monotone relation (concentrated / one-winner regime => low false-recall) across the seeds.

## Result — the regime is ALREADY one-winner, yet false-recall is at floor

| seed | mem | false-recall | correct_rate | wrong_rate | pr_eff | gini | top_conc | active |
|------|-----|-------------|--------------|------------|--------|------|----------|--------|
| 414 | A | **1.000** | 0.000 | 0.072 | 15.36 | 0.698 | 1.000 | 0.333 |
| 414 | B | **0.000** | 0.129 | 0.000 | 15.64 | 0.692 | 1.000 | 0.333 |
| 415 | A | **1.000** | 0.000 | 0.086 | 15.88 | 0.676 | 1.000 | 0.333 |
| 415 | B | **0.000** | 0.115 | 0.000 | 15.76 | 0.684 | 1.000 | 0.333 |
| 410 | A | **0.923** | 0.004 | 0.050 | 17.48 | 0.662 | 0.923 | 0.396 |
| 410 | B | **0.000** | 0.085 | 0.000 | 15.31 | 0.705 | 1.000 | 0.333 |

No label-free candidate is both strongly rank-correlated with false-recall (correct sign) on the probe points and monotone across the seed means; the boundary detector flags a concentrated-but-high-false contradiction on seed 414.

## The precisely-named boundary

**A single label-free regime statistic of the `cortical_target` population is insufficient to set the retest competition point, because the dev-seed failure is not a sparsity/E-I failure — the regime is ALREADY maximally one-winner (gini ~0.68, top-assembly-share ~1.0, pr_eff ~ one assembly, ~1/3 of units active) — it is a WINNER-IDENTITY failure that a label-free statistic cannot resolve.**

The decisive evidence is *within* each seed: the memory-A probe (false-recall ~1.0 — the wrong assembly wins outright, correct_rate 0) and the memory-B probe (false-recall 0.0 — the correct assembly wins) have **essentially identical** label-free `S` (seed 414: pr_eff 15.36 vs 15.64, gini 0.698 vs 0.692, top_conc 1.0 vs 1.0). The catastrophic probe and the perfect probe are label-free-indistinguishable. The discrimination lives entirely in *which* assembly won, which is by definition invisible to any label-free aggregate statistic. The failure pattern is a fixed attractor-dominance asymmetry (memory B's basin captures BOTH cues; A's partial cue falls into B's basin), not a scalar activity level a homeostat could regulate to a set-point.

This is the same class as the v8 source-monitor NO-GO and the gap#3 rate-equalization trap the scoping doc flagged as the real risk: the controllable label-free quantity does not carry the discrimination. A sparsity/E-I integral controller would equalize *total* activity while leaving the winner-identity untouched — it cannot convert an A-cue-lands-in-B's-basin regime into A-wins.

## Consequence for the capability (not deferred)

The interference-control capability is NOT abandoned. What is banked is the *label-free-single-statistic set-point* method. The measurement relocates the residual precisely: the surpass must act on **per-cue basin competition / assembly-identity selectivity** (e.g. cue-specific attractor separation, pattern-separation upstream, or an identity-aware competition mechanism), NOT an aggregate-sparsity homeostat. Whether an identity-aware but still label-free/self-supervised signal exists is the next research question; a scalar regime homeostat is ruled out by this measurement.

## Reproduce

```
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._replay_v7_step0_sparsity_probe \
    --out research/findings/raw/replay_v5_sfa_order/replay_v7_step0_dev.json
```
