# W→A learning verdict: global scalar feedback fails at biological scale

**Date:** 2026-05-05
**Status:** DEFINITIVE — both classical sign-only AND magnitude-graded
DA fall below the dendritic-learning decision gate. The 18-day
investigation arc has landed.

---

## Headline

**Global scalar feedback (sign-only DA OR magnitude-graded DA) cannot
match supervised gradient at biological-scale W→A learning.**

The bottleneck is the credit-assignment rule, not the architecture:

| Rule | Aligned/N at `tf_with_topo_fs` | Verdict |
|---|---|---|
| Supervised gradient (B3) | 3/3 PERFECT | ✅ task is learnable |
| Classical 3-factor (sign-only DA) | 1/6 | ❌ at noise floor |
| Magnitude-graded 3-factor | 0/6 | ❌ below noise floor |

Same architecture. Same input, hidden, motor regions. Only the
credit-assignment rule changed.

The ~7-8pp "structural noise" excess (best-of-24-permutations beats
true-label by ~8pp) is the same architectural floor we've measured
since 2026-05-03's permuted-label control. It's NOT learning aligned
with task labels.

## Decision gate (from `docs/plans/2026-05-05-dendritic-learning-design.md`)

Both conditions met:

- ✅ `bio_three_factor` (classical 3-factor): **1/6 ≤ 1** at
  `tf_with_topo_fs`
- ✅ `bio_three_factor_graded_da` (magnitude-graded): **0/6 ≤ 2** at
  `tfg_with_topo_fs`

Recommended next direction: **apical-basal dendritic learning** (Bono &
Clopath 2017) OR **predictive coding** (Rao & Ballard 1999).

Estimated scope: 1.5-2 months focused engineering.

## What failed and why

### Classical 3-factor (sign-only DA), 2026-05-04 → 2026-05-05

Result: 1/6 aligned at `tf_with_topo_fs` (seed 101 only — known
architecture-noise winner from random init).

Mechanism: the dopamine signal is a SCALAR ±1 broadcast. Under noisy
selection at biological N=500 motor pools with cortical recurrence +
NMDA bistability, every motor pool gets the same scalar reward. The
cross-projections (`language_input → motor_X`) all see the same DA
signal regardless of which one was responsible for the action.

This is the textbook "credit assignment problem with global feedback."
Schultz 1998 documented dopamine bursts at reward; that signal is
phasic but globally broadcast.

### Magnitude-graded 3-factor (Schultz 1998), 2026-05-05

Result: 0/6 aligned at `tfg_with_topo_fs`.

Mechanism: dopamine magnitude scales with reward magnitude (graded ±1
to ±N). The hypothesis was that sign-only DA washes out small wins;
graded DA preserves them. Empirically: it doesn't help. The bottleneck
isn't reward MAGNITUDE — it's reward SPATIAL specificity. Per-region
error, not per-trial scalar.

Per-seed structure (graded-DA results):

| seed | true | best of 24 perms | best perm | aligned |
|---|---|---|---|---|
| 42 | 25.0% | 35.0% | EWNS | no |
| 43 | 18.0% | 32.0% | SWEN | no |
| 44 | 25.0% | 29.0% | NWES | no |
| 100 | 22.0% | 33.0% | WNES | no |
| 101 | 27.0% | 30.0% | EWNS | no |
| 102 | 27.0% | 32.0% | SEWN | no |

True label means: 24.0%. Best-perm means: 31.8%. Excess: +7.8pp.

Best perms are random across seeds — no structural pattern.

### Why gradient works where 3-factor fails

Gradient (B3 supervised): per-region weight updates are computed from
backprop of the loss with respect to each weight. Each motor pool gets
a SPECIFIC error signal: "you should have fired more / less."

3-factor: every weight gets the same DA × eligibility × pre-post
coincidence. The "pre-post coincidence" carries some specificity, but
not enough — at N=500, the noise floor of which neurons coincided by
chance is too high to identify the right cross-projection.

This is exactly what Frémaux & Gerstner 2016 predicted theoretically.
The empirical confirmation took 18 days of investigation.

## Investigation arc (chronological)

| Date | Hypothesis tested | Result |
|---|---|---|
| 2026-04-26 | Cluster A (closed BG loop) | Helped on 2D nav |
| 2026-04-27 | Perception arc + curriculum | 4.08 ± 0.49 nav best |
| 2026-04-28 | Cluster B (D1/D2 + FSI + TANs) | Mixed |
| 2026-04-29 | Cluster D, F (hippo, cerebellum) | Neutral on nav, broke W→A |
| 2026-04-30 | Single-pool heuristic | 5.02 nav best |
| 2026-05-01 | Cluster K v2 (visual cortex) | 2.97 nav, perception only |
| 2026-05-02 | Text I/O Hebbian decay fix | 28.5% W→A "real" |
| 2026-05-03 | Permuted-label control | **0/25 prior W→A "successes" had aligned mapping** |
| 2026-05-03 | Minimal arch (cascade as cause) | FALSIFIED — minimal still 16.7% |
| 2026-05-04 | Biology fixes (topo + FS) | 0/6 still |
| 2026-05-04 | B3 supervised gradient | 3/3 PERFECT — task IS learnable |
| 2026-05-05 | Classical 3-factor | 1/6 at biological scale |
| 2026-05-05 | Magnitude-graded 3-factor | 0/6 — DEFINITIVE |

The decisive moment was 2026-05-04 evening: when supervised gradient
got 3/3 PERFECT under the same biological canon (N=500 cortical
recurrence + NMDA bistability + Pulvermüller topographic prior + Vogels
PV-FSI lateral inhibition), it definitively proved the architecture
was sufficient. Whatever was broken had to be the credit-assignment
rule.

## What this DOESN'T close

- **2D gridworld navigation works at 2.00 with full biology stack.**
  The Cluster G v2.5 + K v2 result from 2026-05-01 still stands. The
  W→A failure is specific to language→motor association, not all
  learning.
- **Perception works.** Cluster K v2 16×16 gridworld at 2.97 ± 0.12
  closes 4 of 5 cheats with NO heuristic. Visual cortex with Gabor RFs
  + retina rendering is biology-grounded and effective.
- **Real-time gridworld RL works.** A+E + G v2.5 single-goal: 3.31 ±
  0.74. This is a working biology-grounded RL agent.
- **Within a single region, plasticity works.** STDP at single-pool
  scale, R-STDP at single-action scale, both functional.

The W→A failure is specifically about **cross-region credit
assignment with global scalar reward**. That's the open problem.

## Three options for the next direction

### Option 1: Apical-basal dendritic learning (Bono & Clopath 2017)

Multi-compartment Izhikevich neurons (V_basal, V_apical, V_soma).
Apical activity GATES basal STDP. Top-down feedback carries per-region
error signal.

**Scope:** 1.5-2 months focused engineering.

**Open questions** (from design doc):
- Apical teaching signal: reward-modulated drive (cheap), reciprocal
  connectivity (mid), hippocampal replay (rich)?
- Per-region error magnitude: predictive coding from above (cleanest),
  lateral comparison (info-poor), trial-relative comparison (Tobler
  2005, biologically supported)?

**Risk:** if we naively route reward to apical, we reinvent R-STDP. The
real value is per-region error magnitude, which requires either
predictive coding or sophisticated reciprocal architecture.

### Option 2: Predictive coding (Rao & Ballard 1999)

Each region has paired generative + recognition pathways. Errors
propagate as signals through the recognition pathway.
Mathematically equivalent to backprop under certain assumptions
(Whittington & Bogacz 2017).

**Scope:** 2-3 months focused engineering. Different network
organization than what we currently have.

**Risk:** even bigger architectural rewrite. Pairs every region with
a generative twin.

### Option 3: Skip cross-region credit; expand single-region learning

Pivot away from W→A entirely. Focus on:
- 2D gridworld scaling (16×16 → 32×32 → richer environments)
- Real-time visual cortex tasks
- Hippocampal replay during NREM/REM
- Cerebellar timing tasks (Marr-Albus-Ito is partially built)

**Scope:** ongoing, no rewrite needed.

**Risk:** doesn't solve the original W→A goal. But might generate more
publishable results faster.

## Recommendation

Option 1 (apical-basal dendritic learning) is the cheapest of the two
biologically motivated options AND directly addresses the bottleneck
identified. The design doc at
`docs/plans/2026-05-05-dendritic-learning-design.md` already exists
with a 1.5-2 month scope estimate.

The conservative starting point is Week 0:
- Add `cfg.enable_apical_compartment` flag (default False, no-op)
- Allocate `cp_apical_activity` array when flag is on
- Add `target_compartment="apical"` field to RegionPathway

This is reversible, doesn't break anything, and unblocks Week 1-2
(multi-compartment Izhikevich kernel) when the user is ready to commit.

Week 1+ (the kernel rewrite) is a major commit and worth user review
before starting.

## Files

- Verdict doc (this file)
- Design doc: `docs/plans/2026-05-05-dendritic-learning-design.md`
- Aggregator output: `research/findings/2026-05-05-bio-three-factor-graded-da-results.md`
- Classical 3-factor verdict: `research/findings/2026-05-05-three-factor-VERDICT-fails.md`
- Permuted-label control: `research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`
- Minimal-arch falsification: `research/findings/2026-05-04-minimal-isolation-INVERSION.md`
- Bench contamination note: `research/findings/2026-05-05-bench-phase1-contamination.md`

## What I'd tell a future engineer onboarding to this project

1. **Don't try to make global scalar reward work for cross-region
   association.** It's been tried in every reasonable way (sign-only,
   magnitude-graded, surprise-boosted, asymmetric, per-action,
   compartmentalized C v2). All NEGATIVE for W→A. Theoretical reasons
   are clear (Frémaux & Gerstner 2016).
2. **Use the permuted-label control before claiming alignment.**
   25 of 25 prior W→A "successes" failed it. The architecture has
   structural noise that produces ~7-8pp above-chance accuracy on
   SOME mapping; that mapping is random per seed.
3. **Gradient-as-control matters.** When gradient PERFECT-aligns
   under the same architecture, you know the bottleneck is the rule.
4. **Biology canon at N=500 is non-negotiable.** Below that scale
   (e.g. minimal isolation N=80 motor pools), STDP can't form stable
   structure even WITH gradient. Don't waste time on toy
   architectures for production claims.
