---
type: finding
status: no-go
claim_check: measured
date: 2026-09-24
lane: B — curiosity (novelty / epistemic-gap crave) x E1 metacognition
mechanism: curiosity-metacog-neuromod-gain v3 (metacog margin-comparator -> frozen point-edge onto curiosity's ASK pool,
  plus a spiking lc_ne population projecting ADDITIVE sub-threshold current onto ASK; both fixed-weight CrossEdges)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_curiosity_metacog_neuromod_gain_derisk.py
prereg: docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md
artifact: research/findings/raw/_curiosity_metacog_neuromod_gain_6seed_combined.json
verdict: NO-GO 1/6 (held-out 0/5). Only the calibration seed 42 passes. On four held-out seeds the lc_ne pathway
  carries less than the pre-registered 20% of the ASK dynamic range (G3); on seed 44 the evidence->ASK coupling is
  not monotone enough (G1). The integrity gates pass on every seed. A verdict on the ADDITIVE, tonic-rate method,
  not on the capability.
---

# Curiosity x metacog, LC-NE modulator: 6-seed NO-GO (1/6, held-out 0/5)

## 1. Result

Artifact: `research/findings/raw/_curiosity_metacog_neuromod_gain_6seed_combined.json` (inputs:
`research/findings/raw/_curiosity_metacog_neuromod_gain_smoke_s42_v3.json` and the five held-out files
`_curiosity_metacog_neuromod_gain_s<seed>.json` beside it).

The pre-registered combined verdict (`--combine` over the six per-seed files, the only output the prereg counts):
**NO-GO, 1 of 6 seeds, 0 of 5 held-out seeds.** Operating point as frozen by the v3 amendment: `LC_N=20`,
`CMP_TO_LC_W=5.0`, `LC_TO_ASK_W=1.0`, G3 floor 0.2.

(Values from `per_seed` in the artifact, rounded to 3 decimals; `gain share` is `attributable_frac.gain`.)

<!--derived-->
| seed | rho(evidence, ASK) | gain share | ASK range combined / gain-lesion (Hz) | failing gate |
|---|---|---|---|---|
| 42 (calibration) | -0.991 | 0.246 | 3.95 / 2.97 | none (GO) |
| 43 | -0.916 | 0.123 | 6.56 / 5.75 | G3 |
| 44 | -0.745 | 0.213 | 5.12 / 4.03 | G1 |
| 100 | -0.855 | 0.084 | 6.99 / 6.40 | G3 |
| 101 | -0.953 | 0.164 | 1.98 / 1.66 | G3 |
| 102 | -0.936 | 0.109 | 6.37 / 5.67 | G3 |

What holds on every seed: the lc_ne population's rate is graded by the metacog evidence (G10, rho between -0.80
and -0.99), the relay lesion abolishes the coupling (G8), class swap stays monotone (G7), the fresh-process
determinism hash matches (G6), metacog's balance/threshold/raster are byte-equal across arms (G5), the restore is
exact and the base connectivity is byte-identical with the organ off. So the circuit is wired and the lc_ne signal
is the right signal; it just does not move the ASK pool enough.

Two further facts from the artifact: the ASK pool never reaches the production threshold on any seed
(`s1_reaches_production_threshold` false; peak about 2-7 Hz against a threshold near 22 Hz), and the edge alone
owns the coupling (`attributable_frac.edge` 1.0, the edge-lesion arm reads 0 Hz). The operating point was chosen on
seed 42, which turned out to be the seed with the largest gain share; the held-out seeds sit well below it.

## 2. What the real system runs that this method replaced

The method delivers the lc_ne signal as a tonic, rate-graded, additive current. Two sources say that is the form
of LC input least able to widen a target's dynamic range:

- (Aston-Jones & Cohen, 2005), Annu Rev Neurosci 28:403-450, doi:10.1146/annurev.neuro.28.061604.135709: LC works in <!--derived-->
  two modes. Phasic bursts are locked to the outcome of a decision and facilitate the ensuing response; tonic
  activity goes with disengagement and exploration. Frontal monitors of utility (ACC, OFC) set the mode.
- (Fan, To & Sciolino, 2026), Cell Rep 45(9):117990, doi:10.1016/j.celrep.2026.117990: optogenetic PHASIC LC <!--derived-->
  activation expanded the dynamic range of a cortical population's stimulus representation, which the authors
  attribute largely to multiplicative gain modulation; TONIC activation engaged fewer neurons and did not expand
  any attribute axis.

The runner's own honesty note already says the lc_ne input is additive current, not a multiplicative gain. The
measured gain share (8-25%) is what an additive offset on top of the edge's drive would give.

## 3. Next method (per the LAW: the capability stays open)

Rebuild the lc_ne -> ASK path as (a) phasic: lc_ne bursts time-locked to the metacog comparator's low-margin
event rather than a sustained rate, and (b) a gain on ASK's response to the edge, not an added current. Candidate
spiking realisations to check against the record first (`bash tools/before_you_build.sh "lc-ne multiplicative gain on
a target pool"`): a modulatory conductance on ASK neurons, or the level of balanced background excitation plus
inhibition onto ASK, which in cortical neurons sets the gain of the response to excitatory drive without a
signal-independent rate offset (raising it is divisive; Chance, Abbott & Reyes, 2002, Neuron 35:773-782,
doi:10.1016/s0896-6273(02)00820-6). A new prereg is needed; this one's operating point is not re-tuned. <!--derived-->

## 4. Honesty

Functional read-outs only. "Curiosity" and "metacognition" here name spiking populations and their measured rates;
nothing in this finding asserts a felt state.
