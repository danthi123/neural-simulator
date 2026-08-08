---
type: finding
status: contributing
date: 2026-08-07
mechanism: stageA-honesty-floor-strengthen-axis-separated
lane: E-language
runner: research/runners/_stageA_honesty_floor_strengthen_derisk.py
builds_on: research/findings/2026-08-07-stageA-foundation-honesty-floor-calibrated-monitor-3way-arbiter-single-seed.md
artifacts:
  - research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed.json
  - research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed.json.prov.json
---

# Stage-A honesty floor STRENGTHENED + axis-separated: familiar-but-wrong is an active CATCH on 4/6, moat-safe on 5/6, a fit-quality guard, and pure-novelty is a characterized boundary

Built ON the 3/6 PARTIAL foundation (`2026-08-07-stageA-foundation-honesty-floor-...`, STRUCTURE 6/6, HONESTY-BEHAVIOR
3/6). Full-size 6-seed run in ONE foreground process, backend numpy, seeds 42/43/44/100/101/102.
Artifact: `research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed.json`. Runner verdict: NEGATIVE-on-strict
(1 seed regresses); the axis-separated reading is a characterized PARTIAL — see below.

## What the foundation left, diagnosed per-seed

The foundation's honesty BEHAVIOR was 3/6 (42/43/101 GO; 44/100 PARTIAL; 102 NEGATIVE). Reading its artifacts, the
three failures were **two different classes**: 44/100 had a GREAT monitor AUC but the deployed band produced only
`A=2` / no-error-in-top-A asserts, so the strict integer `cal_cw < rec_cw` could not fire (0 vs 0) — a small-sample
BAND-COUNT artifact, not a regression. 102 was a genuine bad monitor FIT (its foundation test-AUC fell below the
recall score's). (Foundation numbers per `2026-08-07-stageA-foundation-honesty-floor-...`.)

## What was built (reuse-by-import, additive/default-off, no `sim/` edit, `cfg.seed` set)

- **A STABLE familiar-wrong metric.** Confident-wrong asserts are counted at a **fixed COVERAGE fraction** (top
  ⌈f·N⌉ by the spiking self_schema rate, f∈{0.25,0.333,0.5}) over a **LARGE battery (N=300)**, not the fragile
  per-seed band count. `A` is now driven by coverage (always large), so the 44/100 `A=2` artifact cannot occur.
- **A fit-quality GUARD that validates the DEPLOYED signal.** Fit the calibrated monitor, then on a held-out block
  compare the type-2 AUC of the calibrated **spiking self_schema read** vs the recall self-read. If the monitor does
  not beat recall out-of-sample, REFIT once with 2× calibration data; if it still fails, the guard REFUSES to route
  it and the honesty signal falls back to the recall read (enacted, not just flagged) → graceful degradation.
- **Axis separation.** The MISSION familiar-but-wrong axis (gating) and the PURE-NOVELTY moat-safety axis (a
  characterized boundary, non-gating) are measured and reported distinctly.
- **A battery anti-cheat.** Bootstrap-subsample the large battery at N∈{60,120,300}; report the mean AND std of the
  confident-wrong reduction.

## Result — the MISSION familiar-but-wrong axis (per seed)
<!--derived-->

| seed | outcome | self-read AUC cal / recall | mean confident-wrong: deployed / baseline | note |
|---|---|---:|---:|---|
| 42 | CATCH | 0.871 / 0.727 | 0.058 / 0.117 | large edge |
| 44 | CATCH | 0.882 / 0.678 | 0.052 / 0.163 | **foundation PARTIAL → fixed** |
| 100 | CATCH | 0.817 / 0.682 | 0.093 / 0.153 | **foundation PARTIAL → fixed** |
| 101 | CATCH | 0.859 / 0.692 | 0.134 / 0.240 | large edge |
| 43 | REGRESSION | 0.735 / 0.769 | 0.110 / 0.031 | marginal seed; guard MISSED (see below) |
| 102 | SAFE_FALLBACK | 0.687 / 0.779 | 0.046 / 0.046 | guard caught; degraded to recall (no regression) |

**Active CATCH on 4/6** (42/44/100/101 — every seed where the monitor's self-read edge is large, ~0.14–0.20 AUC).
**Moat-safe (no regression) on 5/6.** The two foundation PARTIALs (44, 100) are now clean active catches — the
strengthening fixed them, confirming they were a band-count artifact, not a monitor failure.

## The two honest residuals (NOT swept)
<!--derived-->

1. **Seed 102 — a genuine per-seed FIT boundary, safely contained, NOT fixed.** Refitting with 2× calibration data
   did NOT rescue it (its self-read AUC stays 0.687 < recall 0.779; validation 0.746 < 0.842). The learned
   correctness monitor cannot beat the raw balance-of-evidence on this seed. The guard's value is robust DETECTION +
   safe degradation (it falls back to recall; deployed confident-wrong 0.046 == baseline 0.046, no regression), NOT a
   fix. So `seed102_fit_fixed_by_refit=False` and the honest per-seed-fit answer is: the guard makes 102 SAFE, the
   underlying fit remains a boundary.
2. **Seed 43 — a marginal-seed regression NO held-out guard can predict.** On seed 43 the calibrated monitor beats
   recall on its calibration AND validation draws (val self-AUC 0.838 > 0.821; at n_val=300, 0.740 > 0.700) but LOSES
   on the independent test draw (0.735 < 0.769) → routing it makes 3.5× more confident-wrong asserts (0.110 vs 0.031).
   More validation data did not flip the guard (it made it MORE confident to route). The regression lives in the TEST
   trial draw, which the guard cannot see without peeking. Seed 44 is the mirror: near-zero validation margin
   (+0.0003) yet a strong test CATCH. So on marginal seeds (monitor-vs-recall edge within ~0.03 AUC) which signal
   wins is within between-draw variance, and no honest guard resolves it. This is the characterized boundary of the
   familiar-wrong axis: robust where the edge is large, a coin-flip where it is small.

## Pure-novelty moat-safety axis — a characterized boundary (reported DISTINCTLY, non-gating)
<!--derived-->
On zero-signal novel trials the learned FAMILIARITY monitor has no reliable edge over raw winner-magnitude (the
isolated finding's point: low signal → low magnitude → abstain, so magnitude is itself informative). Under
coverage-matched operating points the calibrated route is NOT systematically worse (novel-assert ≤ first-order on 5/6;
only the broken-fit seed 102 asserts more, 0.475 vs 0.258). But pure-novelty moat-safety is fundamentally the **hard
cue-match moat's job (475/475 abstains, foundation 6/6)**, not the familiarity monitor's — the monitor is not the moat.
So pure-novelty is a characterized boundary of the monitor (small/absent edge), correctly delegated to the moat, and
is NOT swept under the bigger battery.

## The battery anti-cheat — variance reduction, NOT a faked mean lift
<!--derived-->
Bootstrap-subsampling the large battery at N∈{60,120,300}: on every seed the mean confident-wrong reduction is a
population property that stays ~constant (`mean_shift` 0.0003–0.014 across the N range) while the estimator std falls
sharply (`std_shrink` 0.041–0.054) → `dominant_effect=variance_reduction` on all 6 seeds. The larger battery
STABILIZES the per-seed estimate (which is what fixed the 44/100 spurious PARTIALs — it gives the rate metric a real
denominator); it does NOT manufacture the effect (the monitor is byte-identical at every N). It cannot fake a lift.

## Honest verdict / scope
<!--derived-->
The strengthening DELIVERED: (i) fixed the 44/100 band-count artifact (foundation PARTIAL → active catch); (ii) added
a working fit-quality guard that converts the 102 NEGATIVE into a SAFE fallback (no regression); (iii) separated the
two axes and characterized pure-novelty as a boundary delegated to the moat. It did NOT lift the floor to a clean 6/6:
the familiar-but-wrong axis is an **active CATCH on 4/6** and **moat-safe on 5/6**, with a residual **marginal-seed
regression (43)** that no held-out guard can predict and a **genuine fit boundary (102)** that is contained but not
fixed. So on the KEY question — *is the floor a mission-GO on the familiar-but-wrong axis?* — the honest answer is
**a robust catch on the large-edge seeds (4/6), safe degradation on the clear-cut bad-fit seed, and an irreducible
coin-flip on the marginal seeds**, an improvement in reliability + diagnosis over the 3/6 foundation, NOT a solve.
Reduced-scope caveats: fixed monitor→self_schema relay under STDP/Hebbian/homeostasis/STP/structural/OU DISABLED
(isolation of the mechanism); the affect term is a stub; the moat path is untouched (additive/default-off, no `sim/`
edit). '`The floor lifts to 6/6`' is FALSE here.

## Named next mechanism (do NOT defer the capability)
The marginal-seed instability (43) is the load-bearing residual: the learned monitor and the raw recall score are, on
some seeds, within between-draw variance. The surpass is a monitor with a LARGER, more stable edge over the recall
score — either more/richer ACC/aPFC features so the edge is reliably >0.03 AUC on every seed, or an ENSEMBLE monitor
(average several fits / draws) whose deployed self-read variance is small enough that a held-out guard predicts the
test draw. Both keep the guard's safe-fallback as the floor. This is the next Stage-A honesty build.

## Reproduce
```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._stageA_honesty_floor_strengthen_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 300 --n-novel 120 --calib-robust 192 \
  --out research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed.json
```
