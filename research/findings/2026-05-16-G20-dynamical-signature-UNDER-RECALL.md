# Dynamical failure signature — idx-12 fails by UNDER-RECALL (mechanism identified)

## TL;DR

After three falsifications localized the failure to the *dynamical*
regime, a direct probe on an existing 320 bridge identifies the
mechanism: the known-failing index fails by **UNDER-RECALL** —
stimulating its engram tag fails to reactivate its own sparse
pattern (~5× weaker self-activation than a robust index). It is
**not** static overlap (falsified) and **not** competitive capture
(the incidental winner is itself weak — the whole response is
depressed).

## Probe result (bridgeA_nouns_sparse64, seed 42)

| Index | self_cum | self_rank | steady_self | winner | mode |
|---|---|---|---|---|---|
| idx-0 `apple` (robust) | **1157** | 1 | 7.85 | self | SELF-WINS |
| idx-12 `ball` (failing) | **213** | 12 | 1.65 | idx21 `baby` (385) | UNDER-RECALL |

idx-12's own pattern fires **5.4× weaker** than the robust index's
(213 vs 1157) and ranks only **12th** among all 64 patterns when its
OWN tag is stimulated. The nominal "winner" idx-21 wins with a feeble
385 — far below the robust self-recall of 1157. So nothing fires
strongly: the engram tag → own-pattern reactivation is the broken
link, not interference from a strong competitor.

## Mechanism + actionable lever (for the flagged recovery task)

The defect is in **engram-tag → pattern reactivation strength** for
certain indices. The committed engram tag for idx-12, when
stimulated, drives too few of idx-12's pattern neurons (or those
neurons are insufficiently excitable post-training) to ignite the
ensemble. This is consistent with everything ruled out:

- NOT static pattern overlap (falsified 3×).
- NOT competitive capture (winner is weak, 385 ≪ 1157).
- IS under-activation of the tagged ensemble.

**Concrete fix hypotheses the flagged recovery should test (in
priority order), instead of overlap-rejection / seed roulette:**
1. **Stronger/longer engram capture for under-recall indices** —
   higher teacher-bias pA or a longer capture window so the
   committed tag spans enough of the pattern to reignite it.
2. **Capture-quality gate** — after commit, probe self-recall; if
   self_cum ≪ median, re-capture that tag with boosted drive
   (a targeted, cheap per-index remediation — no full retrain).
3. Per-bridge seeds remain plausible only insofar as a different
   seed yields patterns whose tags happen to self-recall; there is
   still no a-priori selector (seed-quality predictor was NEGATIVE).

Lever #2 is especially attractive: it is a **post-hoc, per-index**
remediation that does not require changing `generate_sparse_patterns`
(preserving the reproducibility invariant + existing validated
artifacts) — the opposite of the artifact-breaking concern that
kept the recovery flagged.

## Honest scope

- **n = 1 failing index** (idx-12 on bridgeA), vs 1 robust control.
  The signature is strong (5.4× depression, self-rank 12) and
  coherent with the corrected root-cause, but whether ALL weak
  indices (idx-10, idx-42, the 160 seed-43/46 dips) share the
  under-recall signature — vs some being capture — is NOT
  established by n=1. The flagged task should confirm across the
  known weak indices before committing to lever #1/#2.
- Single seed (42), single bridge. Short GPU diagnostic on an
  existing artifact; no retrain, no fix applied here (characterization
  only).

## Files

- `research/runners/g20_dynamical_probe.py`
- `research/findings/raw/g11_bg/g20_dynamical_probe.json`
- Closes the chain: overlap-concept → category [retracted] →
  static-overlap [disconfirmed] → **dynamical UNDER-RECALL
  [identified]**. Hands the flagged recovery a mechanism-grounded,
  artifact-safe fix hypothesis (post-hoc capture-quality gate).
