---
type: finding
status: boundary
date: 2026-08-13
mechanism: w4-landscape-read-topography-correction
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_w4_belief_magnitude_calibration_derisk.py
artifacts:
  - research/findings/raw/_w4_belief_magnitude/w4_6seed.json
  - research/findings/raw/_w4_belief_magnitude/smoke.json
builds_on:
  - research/findings/2026-08-13-w4-informativeness-objective-BOUNDARY.md
  - research/findings/2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md
  - research/findings/2026-08-01-W4-recursive-theory-of-mind-2nd-order-false-belief-plus-depth2-scalar-implicature-6seed-GO.md
---

# The W4 implicature ~2x OVERSHOOT is a READ-TOPOGRAPHY artifact (the whole-population landscape read sums off-target belief-only AND-gate leak), NOT the belief magnitude and NOT the coincidence nonlinearity -- reading the TRUE-INTENT detector removes it and the graded belief BEATS one-hot on the informativeness objective 5/6 (mean OBJ_inf move +0.0905 vs the whole-population read's -0.01081, which reproduces [W]), a major advance that just misses the 6/6 bar on ONE heterogeneity-adverse seed (44) whose matched detector under-responds to the fractional implicature drive

<!--derived-->
The 6-seed OBJ_inf numbers, the per-success-group decomposition and the seed-44 collapse below are all read from the cited runner + w4_6seed.json artifact.

<!--derived-->

**One-line verdict:** [W] localized the last W4 residual to "the detector->landscape->row-normalization TRANSFER"
that inflates the graded implicature cell ~2x (~0.40 vs analytic 0.20). Reading the substrate (per-success-group
decomposition) REFINES that: the inflation is specifically the WHOLE-POPULATION landscape read summing an
off-target belief-only AND-gate LEAK -- NOT the belief content (correct, moat intact) and NOT the coincidence
nonlinearity (the [M] graded plateau is magnitude-preserving, r_coinc(.25)/r_coinc(1)~0.23~analytic 0.25). Reading
the TRUE-INTENT detector success[t] (the correct "posterior mass on the intended state", using only the intent
goal) removes the leak, drops the implicature cell 0.40->~0.17 toward analytic 0.20, and makes the graded belief
BEAT one-hot on the informativeness objective **OBJ_inf move +0.090, 5/6 seeds** (mean graded 0.907 vs onehot
0.817), where [W]'s whole-population read could not (move -0.011, 3/6, reproduced EXACTLY here). NOT the 6/6 GO bar:
it fails on ONE heterogeneity-adverse seed (44) whose matched detector under-responds to the low fractional
implicature drive (belief mass ~0.27 -> implicature cell collapses to 0.017). An honest BOUNDARY that BANKS the
read-topography correction (it removed the off-target-leak half of the residual and flipped 5 of 6 seeds) and hands
the seed-44 residual to a detector OPERATING-POINT / homeostatic-gain lever -- a NEW, distinct mechanism, not a
belief change, not an objective reshape, not the refuted deep-credit rule.

## Where the overshoot actually comes from -- measured, not assumed (per-success-group decomposition, seed 42)

<!--derived-->

The success landscape is `S[t,u] = success_signal(belief=belief_src[u], intent=t)`, and `success_signal` reads the
WHOLE success population: `S[t,u] = (1/K)[rate(success[t]) + Sum_{k!=t} rate(success[k])]`. success[t] gets
belief[t]=BELIEF_TOTAL*belief_src[u][t] + the one-hot intent -> a COINCIDENCE (the real signal). success[k!=t] gets
belief[k] but NO intent -> it SHOULD be silent (an AND-gate), but at HIGH belief mass the plateau leaks. Decomposing
the intent="all" row under utterance "some" (graded belief=[0, 0.73, 0.27]) into its three success groups
[none, SBNA, all]:

| read | none | SBNA | all | row-norm implicature cell (intent=all, utt=some) |
|---|---|---|---|---|
| whole-population (the [W]/[M] read) | 0.000 | 0.105 (belief-only LEAK) | 0.065 (coincidence) | 0.057/0.143 = **0.40** (the overshoot) |
| matched true-intent detector success[all] only | -- | -- | 0.065 | 0.065/0.323 = **0.20** (analytic!) |

The off-target success[SBNA] belief-only leak (0.105, from the 0.73 SBNA mass at 1825 pA) is LARGER than the real
coincidence success[all] (0.065), and the whole-population sum dumps it into the implicature-cell numerator -> the
~2x overshoot. The matched read (success[all] only) lands the cell on 0.20. And the coincidence transfer ITSELF is
magnitude-preserving (r_coinc(0.27)/r_coinc(1)=0.0587/0.2575=0.23~analytic 0.25 -- the [M] plateau worked). So the
inflation is the read TOPOGRAPHY (whole-population vs the matched true-intent detector), not the belief magnitude
and not the read nonlinearity. This is a refinement of [W]'s "detector->landscape transfer" localization, now
pinned to the specific term.

## The fix -- a content-independent landscape-read correction (NO belief change, NO sim/ edit)

<!--derived-->

Define the landscape cell as the rate of the TRUE-INTENT detector success[t] -- the correct "posterior mass on the
intended state t" -- instead of the whole-population sum. This uses ONLY the intent goal t (a legitimate
communicative input), zero RSA content, zero per-cell tuning; it removes the off-target AND-gate leak. Beliefs are
byte-identical to the W4 A/B (plasticity off, fixed operating point); the graded magnitude-preserving plateau read
is reused; the monotone read preserves argmax/recall (moat intact). A separate inverse of the detector's
content-free transfer (T measured on controlled one-hot-at-fraction-f drives, averaged over the K columns)
CERTIFIES the matched read faithfully encodes the RSA posterior (mean |T^{-1}(matched)-belief_mass|=0.042 -- it maps
to the posterior, NOT the analytic target: the non-circularity anti-cheat); the inverse is superfluous as a fix
because r_coinc is already magnitude-preserving.

## Result -- 6 seeds {42,43,44,100,101,102}, CPU numpy, the magnitude-preserving graded read

<!--derived-->

| landscape read (OBJ_inf, primary) | onehot | graded | move | graded>onehot | scramble | scramble loses? |
|---|---|---|---|---|---|---|
| **whole-population** (reproduces [W]) | **0.846** | 0.835 | **-0.011** | **3/6** | 0.280 | yes |
| **matched true-intent detector** (the fix) | 0.817 | **0.907** | **+0.090** | **5/6** | 0.189 | yes |
| matched + inverse-transfer (reported) | 0.806 | 0.910 | +0.103 | 5/6 | 0.196 | yes |

The whole-population arm reproduces [W]'s boundary EXACTLY (onehot 0.846, graded 0.835, move -0.011, 3/6, scramble
0.280) -- the instrument is byte-identical. The matched read removes the leak and flips 5 of 6 seeds (mean move
+0.090). M1 (uniform) matched move +0.025 (5/6); OBJ_cell matched move +0.029 (6/6); OBJ_surp matched move +0.003
(3/6 -- it weights the SBNA intent 0.83, where both reads tie, so it barely moves; OBJ_inf, which concentrates on
the implicature intent, is the decisive one and moves +0.090). Per-seed OBJ_inf move (matched): 42 +0.198, 43
+0.125, 100 +0.079, 101 +0.048, 102 +0.133, **44 -0.041** (the single failing seed).

## Why 5/6 not 6/6 -- seed 44's matched detector collapses at the fractional implicature drive

<!--derived-->

The graded implicature cell (matched read) per seed: 42=0.202, 43=0.275, 100=0.286, 101=0.118, 102=0.133, **44=
0.017**. Seed 44 collapses. Its per-seed graded-plateau calibration picked center=96 (the highest of the six; the
others 88), which shifts the plateau operating point UP so the low fractional implicature drive (belief mass ~0.27)
falls further below it -> the matched true-intent detector barely responds at f=0.27 -> the implicature signal
vanishes (impl cell 0.017), and graded's intent="all" fidelity (0.721) drops just below one-hot's (0.764). The
inverse-transfer cannot rescue it (T^{-1}(0.017)->0.000): no monotone read can recover a signal the detector did not
produce. The `calibrate_graded_seed` objective minimizes GLOBAL proportionality error over the whole ignition curve
(FGRID), NOT the implicature operating-point (f~0.27) sensitivity specifically -- so on one heterogeneity-adverse
seed the chosen (center, slope) under-resolves exactly the point the implicature lives at. This is the
"operating-point-is-implicit / missing-companion-process" pattern (CLAUDE.md): a global proportionality proxy
substitutes for the implicature-point sensitivity a homeostatic detector would maintain.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **VALID metric:** SCRAMBLE (graded mass on WRONG intents) LOSES to one-hot on the matched OBJ_inf (0.189 <<
  0.817) -- reading the true-intent detector cannot rescue wrong-intent mass; so graded<=onehot on seed 44 is a REAL
  negative, not a broken read.
- **Whole-population reproduces [W]:** onehot 0.846, graded 0.835, move -0.011, 3/6 -- byte-identical to
  `2026-08-13-w4-informativeness-objective-BOUNDARY`, confirming the matched read is the ONLY change.
- **BELIEF unchanged (moat):** beliefs byte-identical to the W4 A/B; graded implicature margin 0.506 (> 0.05); the
  matched read preserves argmax/recall on every belief (magnitude/topography rescaled, which intent wins untouched).
- **PRINCIPLED / non-circular:** T measured with content-free drives; ONE transform per seed applied UNIFORMLY to
  all beliefs (it cannot preferentially help graded); the RECOVERY check (mean |T^{-1}(matched)-belief_mass|=0.042)
  proves the read maps to the POSTERIOR, not the analytic target (which is an OUTPUT: belief content 0.27 +
  row-norm = 0.20, never an input). The weights' analytic L1 == ANALYTIC_L1.
- **6 seeds 42/43/44/100/101/102** (smoke first; the 6-seed is authoritative -- the per-seed split is genuine).

## The residual + the named next mechanism (a wall on a METHOD, not the capability)

<!--derived-->

The off-target-LEAK half of the W4 implicature residual is SURPASSED: the true-intent detector read removes it,
lands the implicature cell on ~0.20 on 5 of 6 seeds, and flips OBJ_inf from -0.011/3-of-6 to +0.090/5-of-6 with the
scramble control intact. The remaining residual is ONE term -- the coincidence detector's SEED-VARIABLE SENSITIVITY
at the low fractional implicature drive (belief mass ~0.27): on the heterogeneity-adverse seed 44 the per-seed
plateau calibration under-resolves the implicature operating point and the matched implicature response collapses
(0.017). This is a detector OPERATING-POINT term, NOT the belief (moat intact), NOT the objective (surpassed by
[W]), NOT the read topography (fixed on 5/6), NOT the coincidence nonlinearity (magnitude-preserving). The next
mechanism is therefore a DETECTOR-side homeostatic gain: calibrate the graded-plateau operating point to the
IMPLICATURE fractional drive specifically (or a divisive-normalization / homeostatic gain on the detector pool that
guarantees a stable fractional-drive response across heterogeneity), so the implicature signal does not fall out on
an adverse seed. This is a NEW, distinct lever (detector operating-point homeostasis) warranting its own de-risk
with the deadband + scramble anti-cheats; the capability is NOT abandoned, the residual is quantified (one seed, a
collapsed fractional-drive response). The refuted deep-credit / two-compartment / BDSP CREDIT rule
(`2026-07-22-gap4-real-issue-NOT-dendrites`) is NOT re-proposed -- this residual is not a credit-assignment problem.

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate. This is a host-side READ-OUT correction (the same category as the
row-normalization + mag-fidelity scoring the pipeline already applies to the neural landscape): read the TRUE-INTENT
coincidence detector success[t] (the correct definition of "posterior mass on the intended state", using only the
intent goal) instead of the whole population. It rescales the read's MAGNITUDE/topography; it does NOT change the
belief (byte-identical to the W4 A/B) or which intent wins (monotone read => recall intact). ONE transform per
seed, content-free, applied UNIFORMLY to onehot/graded/scramble -- SCRAMBLE still loses. The analytic 0.20 is an
OUTPUT (belief content 0.27 + row-norm), never an input -- NOT circular; the RECOVERY check certifies the read maps
to the posterior. The neural coincidence AND (the Leg-1 GO) still does the belief x intent multiply; once read
faithfully, the objective rewards the graded belief's superior RSA calibration (its standing 12x-better strength)
instead of penalizing it for the off-target leak. numpy-CPU real spiking Izhikevich bridges; NO sim/ edit; additive
NEW runner (reuse-by-import of the magnitude-preserving read + the W4 A/B + the informativeness objective). NOT a
claim of phenomenal access to another mind; a self-report would be a functional read-out.

## Sources

- **Frank & Goodman (2012), Science 336(6084):998** -- "Predicting Pragmatic Reasoning in Language Games". The
  analytic RSA listener posterior L1 is the scale the neural landscape read is calibrated to.
- **Larkum (2013), TiNS 36(3):141**; **Mikulasch & Priesemann** -- the dendritic analog/graded plateau read whose
  transfer is measured and inverted (the read whose off-target leak is decomposed). The matched-group read is the
  standard "read the responsive unit" decontamination; the inverse-transfer is a linearizing read-out.
- Builds on the 2026-08-13 informativeness-objective BOUNDARY (which surpassed the objective half and localized the
  residual to the landscape read) and the 2026-08-13 magnitude-preserving-plateau BOUNDARY (the reused graded read).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._w4_belief_magnitude_calibration_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_w4_belief_magnitude/w4_6seed.json
```
