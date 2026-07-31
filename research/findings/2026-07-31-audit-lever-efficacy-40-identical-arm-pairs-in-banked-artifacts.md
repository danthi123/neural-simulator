---
type: finding
status: live
lane: audit
date: 2026-07-31
mechanism: audit-lever-efficacy
claim_check: synthesis
---

# AUDIT: 40 identical-arm pairs in banked artifacts — lesions and controls that may never have engaged

**The first result of the project-wide audit**, produced by the new gate registry on its first run over the live
corpus. Not a new experiment: a re-reading of results already banked.

## 0. Evidence

`research/findings/raw/_provenance/AUDIT_lever_efficacy.json` — every hit, keyed by artifact.
Gate: `tools/gates/lever_efficacy.py` (failure class 1).

## 1. What was found

**40 identical-arm pairs across 11 banked artifacts.** An "identical pair" means two named arms agreeing on
*every* recorded metric to full float precision.

| artifact | hits |
|---|---|
| `_emerge6_recurrent_microcircuit_seq.json` | 18 |
| `_lge_divnorm_multiseed.json` | 9 |
| `_phase1_composer_routeA_smoke_seed42.json` | 3 |
| `_phase1_composer_routeB_smoke_seed42.json` | 3 |
| 7 others (`_emerge2_selfsup_burst`, `_emerge49_graded_read`, 5 × `_gabor_cifar_*`) | 1 each |

The sharpest case: in `_emerge6_recurrent_microcircuit_seq.json`, **three** arms agree on all three metrics —
`apical_feedback_lesion`, `no_teaching_null` and `untrained` all read `onestep = -0.0698238953499733`. A second
triple in the same file: `eprop_lesion` = `eprop_null` = `eprop_untrained` at `-0.08999957735221398`.

**Two distinct manipulations cannot agree to sixteen significant digits.** They are the same computation run
under different labels.

## 2. Why this is failure class 1, and why it matters

Class 1 — *manipulation-never-engaged* — is the most expensive in the taxonomy (10 prior incidents). Its
signature is exactly this: a lesion, arm or lever that was named, reported, and never actually applied. Prior
instances include a lesion targeting a gate **never declared anywhere** (intact 0.735 vs "lesion" 0.765 — the
same run twice), and the crux `kp` arm, which was gated on a value no arm ever supplied and printed results
byte-identical to `fixed_fa`.

**A lesion that does not engage does not produce a wrong number. It produces the RIGHT number for the WRONG
arm** — and then any conclusion of the form "the intact condition beats the lesion" is comparing a thing with
itself. The tool that catches this (`tools/lab.py::lever`) has existed since 2026-07-29 and is imported by
**2 of 1330 runners**.

## 3. What this does NOT establish

- **It is a flag, not a verdict.** Identical arms are *strong* evidence a manipulation did not engage, but exact
  ties can be legitimate: a frozen control, a hard floor, a degenerate task where several conditions genuinely
  collapse to the same value. The gate already excludes exact 0.0 and 1.0 for that reason.
- **It does not say which conclusions are affected.** That requires reading each artifact's finding and asking
  whether the identical pair was load-bearing for its claim. Eleven artifacts, not forty, need that reading.
- **It does not implicate the banked GOs on the board.** None of the 11 artifacts is one of the headline
  6-seed GOs; they are `_emerge*` and `_gabor_cifar_*` intermediates. That is a statement about which files
  were flagged, not a clearance — the audit has not reached the headline artifacts yet.

## 4. Next

Each of the 11 artifacts needs its owning finding read, and the question asked: **was the identical pair
load-bearing for the claim?** Where it was, the finding is retracted or re-run; where it was incidental, the
artifact gets a note. That is 11 readings, not 1841 — which is the point of triaging by what the gates flag
rather than auditing everything uniformly.

## 5. The lesson this audit already demonstrates

**The record contained this the whole time.** No new compute was spent; the values were sitting in committed JSON,
and a 60-line checker found them in one pass. The gate did not discover a fact about the brain — it discovered
that nobody had ever asked the artifacts a question they could always have answered.
