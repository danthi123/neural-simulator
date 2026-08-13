---
type: finding
status: contributing
date: 2026-08-13
mechanism: pragmatic-graded-implicature-belief-source
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
instrument: A/B (onehot [leg2_v2 WTA baseline] vs graded [W4 depth-2 RSA soft-competition population rate]) on a SPIKING magnitude-sensitive pragmatic-alignment metric = the neural success-landscape S[t,u]=success_signal(intent=t, belief=belief[u]) read off the Leg-1 coincidence detector, scored as total-variation fidelity to the analytic Frank-Goodman RSA landscape (per intent). Detector A/B = dendritic-coincidence PLATEAU (coincidence=True) vs LINEAR point-soma (coincidence=False). Anti-cheats: normalization-lesion (FS off) + SCRAMBLE (graded mass on wrong intents). Instrument-validity Verdict = GO (belief graded+calibrated; argmax reproduces the 2026-08-11 negative; scramble loses; lesion no spurious win).
artifacts:
  - research/findings/raw/_pragmatic_success/spiking_graded_belief_6seed.json
  - research/findings/raw/_pragmatic_success/spiking_graded_belief_smoke.json
---

# W4 Task#12: the spiking graded-implicature belief read (no host argmax) IS delivered, but the pragmatic-alignment metric still does NOT move — a 6-seed A/B that localizes the 2026-08-11 residual to the coincidence-detector BASE RATE at the mechanistic level, and shows the DEFAULT dendritic plateau does not strip it (the calibrated-k_threshold Rung-0 is the named next lever)

<!--derived-->

**One-line verdict:** the two levers the 2026-08-11 honest negative named — a spiking magnitude-sensitive read
(no host argmax) + the dendritic-plateau detector — were BUILT and A/B-tested on 6 seeds. The belief-side is
sound (a graded population-rate implicature read, moat intact, 12x better calibrated than the one-hot, and the
OLD host-argmax metric is confirmed magnitude-blind). But the pre-registered **metric-move gate FAILS**: over 6
seeds the graded belief does NOT improve (mean −0.035) the magnitude-sensitive fidelity metric, and the default
plateau does not rescue it. A single-cell diagnostic pins WHY: the coincidence detector's **base-rate firing**
already places ~0.27 (normalized) mass on `S[all|some]` for the ONE-HOT belief that has ZERO belief mass there —
so the graded belief's genuine 0.27 fractional mass, being SUB-plateau at the W4-calibrated `K_THR=44`, reads at
the same base rate and is invisible. This **CONFIRMS the 2026-08-11 re-diagnosis** ("the wall lives in the
detector, not the belief") at the mechanistic level, and refutes the assumption that the default dendritic
plateau strips the base rate. NOT a GO; an honest boundary + the specific next lever.

## What is HOST vs SPIKING in W4 (the audit's flag, resolved)

<!--derived-->

The 2026-08-12 faculty audit flagged W4 as "an operating-point read with a HOST argmax". Reading the code, the
host argmax was **in the METRIC, not the belief**: `aligned[t]=argmax_u belief[u][t]` and `succ_opt[t]=argmax_u
S[t,u]` (`_pragmatic_success_readback_leg2_v2_derisk._aligned_utts` + the 2026-08-11 runner) collapse the graded
belief to an argmax to SCORE it — which is (a) a forbidden host shortcut and (b) precisely why a graded
refinement could not move the 2026-08-11 metric. The belief ITSELF was already a substrate soft-competition read
(`_recursive_tom_rsa_derisk._compete` FS divisive-normalization population rate); its only host op is a per-intent
rate normalization (spike-count→rate), a legitimate read-out. **This runner removes the host argmax from the
metric** and reads the pragmatic alignment as a magnitude-sensitive neural coincidence rate.

## The build (both 2026-08-11-named levers, spiking, NO sim/ edit, reuse-by-import)

<!--derived-->

- **Belief (spiking, no host argmax):** `graded_belief_sources` — the depth-2 RSA speaker distribution S1 read
  off the W4 bridge's FS divisive-normalization competition, one step before the operating point's final hard-WTA
  `_compete` would collapse it. A graded population rate over the state assemblies.
- **Metric (spiking, magnitude-sensitive):** the neural success LANDSCAPE `S[t,u]=success_signal(belief=belief[u],
  intent=t)` off the Leg-1 coincidence detector (`_pragmatic_success_coincidence_derisk`). `S[t,u]` is the
  listener's posterior mass on the TRUE intent t, delivered as graded currents and read as a graded coincidence
  RATE — the finding's exact named lever. Alignment score = per-intent total-variation FIDELITY of that neural
  landscape to the analytic Frank-Goodman RSA landscape. The argmax-of-S metric is also computed (to show it stays
  flat). This is a REPLACEMENT of the argmax metric, not a re-sweep.
- **Detector (spiking, the companion process):** the same landscape read through the dendritic-coincidence
  PLATEAU (`coincidence=True`, the engine-native Poirazi/Larkum two-input plateau) vs the LINEAR point-soma sham
  (`coincidence=False`).

## Result — 6 seeds {42,43,44,100,101,102}, CPU numpy

<!--derived-->

| read-out (6-seed mean) | onehot | graded | verdict |
|---|---|---|---|
| **G1** belief implicature margin (SBNA−all) | — | **+0.506** | graded (lesion +0.006 → collapses, 98.9% attributable) |
| **G2** belief calib L1(some)→analytic RSA (lower=better) | 0.500 | **0.041** | 12× better calibrated |
| **G4** OLD host-argmax metric (`argmax_u S[t,u]==t`) | **1.000** | **1.000** | FLAT — reproduces the 2026-08-11 negative (argmax is magnitude-blind) |
| **G3** SPIKING magnitude fidelity (plateau) | **0.653** | **0.618** | **move = −0.035 (1/6 seeds graded>onehot) → FAILS** |
| — fidelity under normalization-LESION (G5a) | 0.653 | 0.587 | lesion < onehot (no spurious win) |
| — fidelity under SCRAMBLE (G5b) | 0.653 | **0.308** | scramble ≪ onehot → the metric is a VALID non-trivial instrument |
| — fidelity, LINEAR point-soma detector | 0.343 | 0.352 | move = +0.009 |
| **G5c** plateau move vs linear move | −0.035 | +0.009 | the plateau does NOT deliver a larger move → FAILS |
| DIAG `S[all|some]` (norm), the some→not-all cell | **0.271** | **0.223** | one-hot ≈ graded — base rate, not belief mass |

Instrument-validity Verdict = **GO** (n≥6, G1, G2, G4, G5a, G5b all hold — the belief is a sound graded spiking
read and the fidelity metric is a valid, non-trivial instrument). The metric-move HYPOTHESIS = **NEGATIVE**.

## Why it does not move — the mechanism (the smoking gun in the raw S)

<!--derived-->

Per-seed the one-hot `S[all,:]` = [none, **some**, all] reads e.g. seed 42 `[0.026, 0.066, 0.075]`, seed 101
`[0.085, 0.051, 0.091]`. The **`some` column is nonzero (0.03–0.07) on every seed** — yet the one-hot
`belief["some"]=[0,1,0]` puts ZERO mass on state "all". It sits at the same level as the true-zero `none` column
(`belief["none"]` also has zero mass on "all"): **both are pure detector base-rate firing.** The graded belief's
genuine 0.27 fractional mass on "all" is SUB-plateau at the W4-calibrated `K_THR=44` (calibrated for a FULL-mass
one-hot coincidence, per the Leg-1 detector), so a 0.27-fraction coincidence does not cross the plateau trigger
and reads at ~base rate too — often LOWER than the one-hot by heterogeneity noise. So the graded refinement is
**invisible to the detector**: `S[all|some]` is 0.271 (onehot) vs 0.223 (graded), a wash dominated by base rate.
This is exactly the 2026-08-11 re-diagnosis ("the succ_opt gap is the DETECTOR, not the belief — a per-utterance
BASE RATE") made mechanistically explicit, and it survives BOTH the magnitude-sensitive metric AND the default
dendritic plateau.

Note the seed-42 SMOKE alone read move = +0.027 (direction-correct); the 6-seed aggregate is −0.035 with only
1/6 seeds positive. **The single-seed smoke was unrepresentative — the 6-seed bar caught a false positive.**

## The residual + the named next lever (per THE LAW: a wall on a METHOD, not the capability)

<!--derived-->

The wall is now localized precisely: the coincidence detector at its **W4/Leg-1 operating point (`K_THR=44`,
`PLATEAU=80`) is calibrated for full-mass (one-hot) coincidence and cannot read a FRACTIONAL (graded) coincidence
above its base rate.** The indicated next lever is a **READOUT-threshold recalibration**, explicitly NOT a
credit-assignment change: two-compartment / dendritic / BDSP / burstprop deep credit is already tested-NEGATIVE
for hidden credit on spikes (`2026-05-17-dendritic-credit-assignment-NEGATIVE.md`,
`2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md`) and is NOT proposed here. What
IS proposed is the same **Rung-0 detector calibration** the 2026-07-08 dendritic dAP READOUT GO required — an
uncalibrated `k_thresh` "NEVER triggered"
(`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`, a read-out nonlinearity, not a learning
rule): set `coincidence_k_threshold` to the per-step coincident drive of a GRADED belief+intent so the plateau
triggers on fractional coincidence and strips the base rate, then re-run this exact A/B. A magnitude-sensitive
learned REWARD (the STEP-2 lever) is downstream — the STEP-1 ceiling shows the belief+metric alone cannot move it
until the detector can see the graded mass. No capability abandoned; the detector-calibration lever is isolated +
quantified.

## External grounding (deep-research gate)

The two design choices are grounded in the external literature, not just our record. (1) The magnitude-sensitive
metric replaces the host argmax because the RSA objective IS a magnitude: the listener posterior L(s|u) is a
graded probability distribution and the speaker objective is expected surprisal, not an argmax — **Frank &
Goodman (2012), Science 336(6084):998, "Predicting Pragmatic Reasoning in Language Games"**. (2) The next-lever
`k_threshold` recalibration is grounded in the dendritic-plateau threshold biology: plateau initiation has a
tunable coinciding-input threshold ("lower the amount of coinciding spikes required to initiate a plateau
potential") — **Larkum (2013), Trends in Neurosciences 36(3):141, "A cellular mechanism for cortical
associations"**; NMDA spike/plateau review (PMC6705260). A detector whose plateau triggers on FRACTIONAL
coincidence is what a graded belief needs; the W4/Leg-1 `K_THR=44` is tuned for full-mass coincidence.

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate. The graded belief is a spiking population-rate read (no host argmax); its
per-intent rate normalization is a read-out op (spike-count→rate), the same footing the existing pipeline uses,
and the graded STRUCTURE is the substrate's FS divisive normalization (collapses under its lesion, 98.9%
attributable). NOT a claim of phenomenal access to another mind; self-report would be a functional read-out.
Plasticity off (STDP/Hebbian/homeostasis/STP/structural/OU/NMDA disabled) — a fixed-operating-point read, as in
the W3/W4/W5/leg2 GOs. This does NOT overturn the 2026-08-11 negative by moving goalposts: the magnitude metric +
dendritic detector are that finding's OWN two named levers, and the scramble control (graded mass on WRONG
intents scores 0.308 ≪ onehot 0.653) keeps the metric honest — it rewards CORRECT implicature content, not
gradedness. numpy-CPU real spiking Izhikevich bridges; additive NEW runner
(`research/runners/_pragmatic_spiking_graded_belief_derisk.py`, reuse-by-import of W4 + Leg-1 + the graded-belief
source); NO `sim/` edit.

Reproducer: `SIM_BACKEND=numpy python -u -m research.runners._pragmatic_spiking_graded_belief_derisk --seeds 42 43
44 100 101 102 --json research/findings/raw/_pragmatic_success/spiking_graded_belief_6seed.json`.
