---
type: finding
status: boundary
date: 2026-08-13
mechanism: rsa-informativeness-weighted-pragmatic-objective
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_w4_informativeness_objective_derisk.py
artifacts:
  - research/findings/raw/_w4_informativeness/w4_6seed.json
  - research/findings/raw/_w4_informativeness/smoke.json
builds_on:
  - research/findings/2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md
  - research/findings/2026-08-13-w4-detector-k-recalibration-BOUNDARY.md
  - research/findings/2026-08-01-W4-recursive-theory-of-mind-2nd-order-false-belief-plus-depth2-scalar-implicature-6seed-GO.md
---

# The RSA-informativeness-weighted pragmatic objective (Frank-Goodman 2012) REMOVES 86.8% of one-hot's advantage on the W4 metric -- confirming the residual WAS the objective aggregation -- but does NOT flip to a 6/6 graded win: the remaining ~13% relocates OFF the objective onto the graded belief's neural landscape MAGNITUDE on the implicature intent (a 6-seed A/B; the objective lever is now exhausted, the residual is belief-side)

<!--derived-->

**One-line verdict:** the 2026-08-13 magnitude-preserving finding localized the W4 pragmatic residual to the
OBJECTIVE/METRIC AGGREGATION (the graded plateau read is magnitude-preserving and the belief reads fine, yet the
intent-AVERAGED M1 favors one-hot because the analytic RSA landscape is mostly one-hot and graded's spurious
off-diagonal mass hurts on the two clean intents). Its named next lever was an informativeness-weighted objective
(Frank & Goodman 2012 -- the RSA speaker maximizes expected informativeness, so weight each intent by how much it
disambiguates). Built + A/B-tested on 6 seeds with weights taken DIRECTLY from the analytic RSA (belief-independent,
zero free parameter). It does what it should: weighting the intents by their RSA informativeness (which
concentrates entirely on the implicature intent -- the only intent with graded recovery) **removes 86.8% of
one-hot's baseline M1 advantage** (M1 gap +0.082 -> informativeness-objective gap +0.011). But it does NOT close
the boundary: on the informativeness objective the graded belief still does not beat one-hot 6/6 -- mean move
**-0.011, 3/6 seeds** (a genuine split, not a win), with the SCRAMBLE control losing on every objective (metrics
VALID). The residual is RELOCATED, and precisely: it is no longer the objective (that half is surpassed), it is
the graded belief's neural landscape MAGNITUDE on the implicature intent -- graded OVERSHOOTS the analytic
implicature after row-normalization while one-hot undershoots, and they roughly tie (3/6). NOT a GO; an honest
boundary that BANKS the objective lever (the aggregation was 87% of the residual, now removed) and hands the
remaining ~13% to a BELIEF-SIDE magnitude-calibration lever -- NOT a further objective reshape and NOT a read-out
(both now exhausted). The refuted deep-credit / two-compartment / BDSP rule is NOT re-proposed.

## The mechanism -- a SCORING-OBJECTIVE change only, weights from the analytic RSA (NOT tuned)

<!--derived-->

The read-out and belief are held fixed at the 2026-08-13 magnitude-preserving operating point (reuse-by-import of
`_magnitude_preserving_plateau_readout_derisk`: the calibrated graded dendritic-plateau read + the exact W4 A/B
belief sources; beliefs byte-identical, plasticity off). ONLY the scoring objective changes. The intents are the
RSA states {none, SBNA, all}; the analytic Frank-Goodman landscape (verified L1 == `ANALYTIC_L1` in-code) is
one-hot on `none` and `SBNA` and graded ONLY on `all` (row-normalized `[0, 0.2, 0.8]` -- the scalar implicature
"some -> not all" lives here). The informativeness weights, all derived from the analytic RSA and independent of
any belief under test:

- **PRIMARY -- per-intent recovery entropy** `w_inf(t) = H(Ideal[t,:])`, the "expected surprisal" the listener
  resolves to recover intent `t`. It is **0 for `none` and `SBNA`** (a single utterance conveys them -- zero
  pragmatic informativeness) and **>0 only for `all`** (recovered from a graded "some"+"all" distribution). So the
  informativeness weighting concentrates the objective on the implicature intent, exactly as Frank-Goodman
  prescribes: `w_inf = [0, 0, 0.500]`.
- **ROBUSTNESS -- per-intent literal surprisal** `w_surp(t) = -Sum_u Ideal[t,u] log L0(t|u)` (keeps SBNA + all):
  `w_surp = [0, 0.693, 0.139]`.
- **ALTERNATIVE -- utterance/cell-level speaker informativeness** `W[t,u] = analytic S1(u|t)` ("the pragmatic
  value of an utterance u for intent t = how much a rational informative speaker uses u to convey t").

The objective is `OBJ_inf = Sum_t w_inf_norm(t) * [1 - 0.5 TV(Snorm[t], Ideal[t])]` -- the SAME per-intent
magnitude-fidelity as M1, only the uniform 1/3 intent average is replaced by the RSA informativeness weight. NO
`sim/` edit; additive NEW runner; the weights are printed + recorded in the artifact.

## Result -- 6 seeds {42,43,44,100,101,102}, CPU numpy (graded dendritic-plateau read = the magnitude-preserving read)

<!--derived-->

| objective (6-seed) | one-hot | graded | move | graded>onehot | scramble | scramble loses? |
|---|---|---|---|---|---|---|
| **M1** intent-averaged (baseline; the wall) | **0.888** | 0.807 | **-0.082** | **0/6** | 0.192 | yes (M1 VALID) |
| **OBJ_inf** informativeness-weighted (PRIMARY) | 0.846 | 0.835 | **-0.011** | **3/6** | 0.280 | yes (VALID) |
| **OBJ_surp** surprisal-weighted (robustness) | 0.881 | 0.871 | -0.011 | 3/6 | 0.282 | yes |
| **OBJ_cell** speaker-S1 utterance-weighted (alt) | 0.898 | 0.812 | -0.086 | 0/6 | 0.220 | yes |

The informativeness weighting collapses one-hot's advantage from **+0.082 to +0.011 (86.8% removed)** -- the
objective aggregation WAS most of the residual, precisely as the magnitude finding predicted. Both principled
per-intent weightings (entropy, surprisal) agree at ~-0.011, 3/6 -- the conclusion is not an artifact of the
weight form. The utterance/cell weighting does NOT help (it weights the diagonal `(all,all)` cell at 0.667, where
one-hot's clean diagonal wins, so it slightly worsens the gap). Per-seed OBJ_inf (the genuine 3/6 split):

| seed | 42 | 43 | 44 | 100 | 101 | 102 |
|---|---|---|---|---|---|---|
| one-hot | 0.952 | 0.807 | 0.810 | 0.820 | 0.885 | 0.800 |
| graded | 0.802 | **0.886** | 0.775 | **0.899** | 0.801 | **0.846** |
| move | -0.150 | +0.079 | -0.035 | +0.079 | -0.084 | +0.046 |

## Why it does not flip -- the residual is the graded belief's landscape magnitude, not the objective

<!--derived-->

The informativeness objective correctly zeroes the two clean intents (`none`, `SBNA`), removing the penalty
graded paid there for spurious off-diagonal mass (graded's per-intent fidelity: `none` 0.708 vs one-hot 0.931 was
its worst loss; that weight is now 0). What remains is the pure implicature-intent comparison, and there graded
does NOT beat one-hot: per-intent fidelity intent=`all` is **graded 0.835 vs one-hot 0.846**, a 3/6 split. The
mechanism is the SAME overshoot the detector-k finding named, now shown to survive the magnitude-preserving read
and to be the LAST term: the graded belief's intent=`all` row, after row-normalization, puts ~0.40 mass on the
"some" (implicature) cell versus the analytic 0.20 -- an OVERSHOOT -- while one-hot puts ~0.15 (an UNDERSHOOT).
The two are roughly equidistant from 0.20, so a symmetric fidelity ties them (and the seed-to-seed jitter in the
overshoot magnitude produces the 3/6 split). The graded belief's *content* is correct (L1(all|some)=0.25, moat
intact); it is the detector->landscape->row-normalization TRANSFER that inflates it ~2x. That is a BELIEF/landscape
term, not an objective term and not a read-out term (the read is verified magnitude-preserving).

Two controls make the relocation unambiguous, both load-bearing:
- **The metric stays VALID:** SCRAMBLE (graded mass on WRONG intents) LOSES to one-hot on EVERY objective
  (OBJ_inf 0.280 << one-hot 0.846) -- so graded<=onehot is a REAL negative, not a broken metric. A scrambled
  belief cannot win the informativeness objective by any weighting tried.
- **The one-hot arm is reproduced** under every objective (its scores are the honest comparison), and the M1 arm
  reproduces the 2026-08-13 wall exactly (move -0.082, 0/6).

## The residual + the named next mechanism (a wall on a METHOD, not the capability)

<!--derived-->

The objective aggregation half of the W4 pragmatic residual is SURPASSED: an informativeness weighting grounded in
Frank-Goodman removes 86.8% of one-hot's M1 advantage with the scramble control intact. The remaining ~13%
(move -0.011) is now isolated to ONE term -- the graded belief's neural-landscape implicature MAGNITUDE overshoots
the analytic RSA (~0.40 vs 0.20 after row-normalization) so it only ties one-hot's undershoot on the implicature
intent. Two levers are now EXHAUSTED for this residual: the detector base rate (detector-k recalibration), the
read-out magnitude-blindness (graded plateau read), and the objective aggregation (this finding). The next
mechanism is therefore BELIEF-SIDE: calibrate the graded belief's landscape magnitude so its implicature cell
lands ON the analytic 0.20 (rather than overshooting to ~0.40) -- e.g. matching the detector's transfer gain to
the analytic RSA target on the ignition-curve instrument, or a divisive-normalization step on the landscape row
that is calibrated content-independently (the SAME instrument discipline used for the read). This is a NEW,
distinct lever (belief/landscape magnitude, not objective or read-out) and warrants its own de-risk with the
deadband + scramble anti-cheats; the capability is NOT abandoned, the residual is quantified (a ~2x implicature-
magnitude overshoot, worth move -0.011 / 3-of-6). The refuted deep-credit / two-compartment / BDSP CREDIT rule
(`2026-07-22-gap4-real-issue-NOT-dendrites`) is NOT re-proposed -- this residual is not a credit-assignment
problem.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **PRINCIPLED weighting, verified:** the analytic L1 the weights derive from matches the `ANALYTIC_L1` RSA
  posterior in-code (gate `weights_principled_L1_matches_analytic`); the weights are the RSA's own informativeness
  (entropy / surprisal / speaker-S1), belief-independent, zero free parameter -- they CANNOT have been tuned to
  make graded win, and indeed they do not (3/6).
- **VALID metric (scramble loses):** the SCRAMBLE control loses to one-hot on all four objectives (OBJ_inf
  scramble 0.280 << one-hot 0.846) -- a scrambled belief must still lose, and does; so graded<=onehot is a real
  negative, not a broken objective.
- **One-hot reproduced + baseline wall reproduced:** the M1 arm reproduces the 2026-08-13 negative EXACTLY
  (move -0.082, 0/6), confirming the informativeness objective SPECIFICALLY changes the aggregation while the
  belief/read are unchanged.
- **Read control:** carried on the ALL-OR-NONE read as well (there the entropy objective reads move +0.006 but
  only 3/6 -- still not a robust win); the conclusion does not depend on the read.
- **6 seeds 42/43/44/100/101/102** (smoke first; the 6-seed is authoritative -- the per-seed split is genuine, not
  a mean artifact).

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate. This changes ONLY the SCORING OBJECTIVE (the informativeness weighting derived
from the analytic Frank-Goodman RSA -- belief-independent, not tuned). Reuse-by-import of the 2026-08-13
magnitude-preserving graded dendritic-plateau read + the W4 A/B belief sources (beliefs byte-identical, plasticity
off, fixed operating point). The SCRAMBLE control keeps the objective honest; the all-or-none read is the read
control. Row-normalization is part of the pre-registered M1 (it controls the intent-drive base rate) and is kept,
NOT changed, to avoid a metric-tuning confound. numpy-CPU real spiking Izhikevich bridges; NO `sim/` edit
(the informativeness objective is host-side scoring over the neural landscape, exactly as M1 is); additive NEW
runner (`research/runners/_w4_informativeness_objective_derisk.py`). NOT a claim of phenomenal access to another
mind; a self-report would be a functional read-out.

## Sources

- **Frank & Goodman (2012), Science 336(6084):998** -- "Predicting Pragmatic Reasoning in Language Games". The RSA
  speaker maximizes EXPECTED INFORMATIVENESS / minimizes surprisal; the pragmatic value of an utterance is weighted
  by how much it disambiguates. This grounds the informativeness weighting (the objective is a graded
  informativeness weighting over intents, not a uniform average). External record logged (lane d-pragmatics,
  2026-08-13).
- Builds on the 2026-08-13 magnitude-preserving-plateau-readout BOUNDARY (which localized the residual to the
  objective aggregation and named this lever) and the 2026-08-13 detector-k-recalibration BOUNDARY (which named the
  ~2x implicature-magnitude overshoot now shown to be the belief-side residual).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._w4_informativeness_objective_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_w4_informativeness/w4_6seed.json
```
