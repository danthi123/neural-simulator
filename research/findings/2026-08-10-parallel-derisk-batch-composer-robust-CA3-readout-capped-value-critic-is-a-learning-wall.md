---
type: finding
status: contributing
date: 2026-08-10
mechanism: multi-frontier-derisk-batch
lane: composer / EPISODIC / D-pragmatics
---

# Parallel de-risk batch (3 frontiers): the composer is REPRESENTATION-robust to correlated codes (the "break" is a readout-offset artifact, verified); the CA3→cortex readout drive is runner-side EXHAUSTED (deep sim wall); and the value-critic wall is the LEARNING stage, not the readout (BOTH readout-SNR duals fail)

Three independent frontiers de-risked concurrently (a workflow fan-out). Each result is teeth-backed; the two
load-bearing REDIRECTS are stated up front.

## 1. Composer under NATURALISTIC (correlated) codes — the "capacity break" is a READOUT-OFFSET artifact, NOT crosstalk (verified)

<!--derived-->

The adversarial-verify caveat was that ALL composer GOs used idealized near-orthogonal codes. New runner
`_teacher_loop_arity_capacity_correlated_derisk.py` adds a correlation knob rho: each primitive code =
`sqrt(1-rho)*independent + sqrt(rho)*(B@g)` with B a SHARED rank-r subspace (rho→1 ⇒ codes collide). I verified the
smoke skeptically (the DC-confound that voided the first arity-capacity finding makes composer "breaks" suspect):
- r=2, rho 0→0.95: corrected shared recall 1.00/1.00/1.00/**0.69**, but **cos-to-true stays 0.988→0.973** and the
  oracle-per-fact offset removal recovers it to 0.81.
- r=1 (codes on a line), rho→0.99: corrected recall craters to **0.44** while **cos-to-true stays 0.993** (perfect
  reconstruction direction) and oracle-DC recovers it.
**A genuine bundling-crosstalk limit REQUIRES the reconstruction to degrade (cos-to-true drops). It does NOT** —
even at extreme correlation the representation reconstructs the true prototype's direction; only the Euclidean
nearest-proto RULER craters, because under correlation the offset becomes fact-DEPENDENT and a constant
mean-centering under-corrects it (and cosine saturates at low rank from collinearity). ⇒ **the composer's
superposition is representation-robust to correlated/naturalistic codes; the apparent naturalistic "capacity break"
is a readout-ruler offset artifact, the SAME class as the retracted arity finding.** Cleanly resolving where bundle
truly needs bind needs an OFFSET-INVARIANT, collinearity-robust readout — the honest open question is a RULER
problem, not a representation limit. (Verdict AMBIGUOUS by the runner's own controls — correctly.)

## 2. CA3→cortex readout DRIVE — runner-side EXHAUSTED; the episodic residual is the deep gap#5 sim wall

<!--derived-->

Runner `_riii_ca3_cortex_readout_drive_sweep_SMOKE.py`, seed 42 at the GO config: sweeping `--ca3-cortex-w`
4→8→16→32 leaves recall IDENTICAL at 0.750 (max_cortex ~0.042 unchanged); baseline reproduces the known positive
(permute 0.000, untrained 0.250 = specific). Both skeptical controls held (specificity did NOT rise with drive).
⇒ the episodic recall residual (0.65, ceiling-bounded) is NOT liftable by runner-side readout weight. **CORRECTION
(2026-08-10, doc-drift fixed — the CA3 de-risk in the same-day parallel push caught this mis-citation):** the
residual is NOT a "functionally-silent recurrents / ~1000×-too-weak transmission" wall — that `2026-07-08-riii-CORRECTION`
claim was REFUTED 3× (`2026-07-17-gap5-ca3-recurrents-NOT-silent-transmission-refuted-*`: the ca3→ca3 recurrents
transmit and scale ~linearly with weight — g_e scales ~30,000× over weight 0→1000; the "0.2 mV" was a weak-drive
floor artifact; re-confirmed 2026-07-25 + a fresh probe this session). The genuine residual is the ATTRACTOR
STRENGTH / SPECIFICITY (does trained recurrent LTP yield pattern-SELECTIVE completion — a tractable weight×density
sweep) OR the dendritic-plateau completion readout (6-seed GO) — a config/runner lever, NOT a sim transmission fix.

## 3. Value-critic — the wall is the LEARNING stage, NOT the readout (BOTH readout-SNR duals fail)

<!--derived-->

Runner `_pragmatic_readback_leg2_v2_ampattractor_derisk.py`: the SIGNAL-AMPLIFICATION dual of the (concurrently
refuted) homeostat — a RECURRENT VALUE ATTRACTOR (within-population self-excitation = winner-take-more) on the
competing utterance assemblies. NEGATIVE: on the controlled-afferent isolation test both mandatory controls indict
the lever (no-lever baseline reproduces the sub-ceiling behavior; the amplifier does not convert the tiny gap into
a robust winner). ⇒ **combined with the homeostat NEGATIVE (`ced73424`), BOTH readout-SNR levers fail — the
value-critic convergence wall is NOT a readout-SNR problem.** The residual is the LEARNING stage: DA credit
assignment must reliably place the highest intent→utterance value on the ALIGNED utterance across seeds (the
critic-argmax is 0.556 because the learned value itself is wrong/noisy, not because a right value can't be read
out). NEXT for "learn to speak" = the DA credit-assignment rule (contingency/eligibility), not the decision stage.

## Housekeeping

<!--derived-->

Composer artifacts (verified this session): `research/findings/raw/_aritycorr_verify_s42.json` (r=2 rho-sweep),
`research/findings/raw/_aritycorr_r1_s42.json` (r=1 rho→0.99). The CA3-readout + value-critic-amp de-risks are the
agents' smokes (controls held, quoted); reproducers are the two runners above. NO `sim/` edit in any de-risk.
SIM_BACKEND=numpy. (A 4th frontier — breadth teacher-loop scaling — was still running at write time; folded in
separately.)
