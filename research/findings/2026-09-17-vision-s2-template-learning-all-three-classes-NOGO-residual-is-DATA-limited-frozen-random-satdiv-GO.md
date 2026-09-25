---
type: finding
status: verified
date: 2026-09-17
mechanism: close the vision S2-template-learning question by testing the THIRD and last learning class — SUPERVISED
  reward-modulated STDP (R-STDP, label-gated: round-robin class-assigned templates, potentiate-true / depress-wrong,
  Mozafari 2017/2018) — against the frozen-random S2 bank at the satdiv-GO op-point, AND separately sizing the
  frozen-random bank's DATA-scaling (held accuracy vs examples-per-class) to test whether the configural-readout
  residual is data-limited or code-limited
integration_faculty: perception (vision object readout — S2 template code)
lane: perception (vision configural / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO on supervised R-STDP S2-template shaping (the THIRD learning class banked, after unsupervised-local
  BCM and SAILnet), AND the residual is DATA-limited not code-limited. Two 6-seed results at the satdiv-GO op-point:
  (1) supervised R-STDP does NOT beat the frozen-random bank (fewer seeds clear the capability bar than frozen-random
  does, mean held accuracy slightly below; numbers in body), with the templates NOT collapsed and the readout proven
  load-bearing — so it is a genuine method verdict, not an instrument artifact; a naive symmetric depress term DID
  collapse the bank (verify-first caught it; over-depression from the 1-true:N-other class imbalance) and was
  corrected before the 6-seed. (2) the frozen-random bank's held accuracy RISES monotonically with examples-per-class
  and the already-banked capability GO holds from the standard example count upward, degrading below it — the residual
  the whole S2-learning arc chased is DATA, not the template code. Mechanistic close: the frozen unbiased random (JL)
  projection preserves every input direction; ALL three activity/label-driven rules move templates OFF it and discard
  the rare configural directions that MORE DATA lets the readout exploit — so template learning is neither needed
  (frozen-random+satdiv is already GO) nor improvable at this data scale. A wall defers a METHOD (template learning,
  now thrice), never the capability (which is GO). The productive lever is DATA (examples/class), parallel to the
  mouth token-supply finding. Additive, --s2-learn rstdp default-off (byte-identical when off), wires nothing.
runner: research/runners/_vision_lindiscrim_readout_derisk.py (--s2-learn rstdp; --n-ex data-scaling)
artifacts:
  - research/findings/raw/lanes/perception/rstdp_g0.05_d0.1_ep5_6seed.json
  - research/findings/raw/lanes/perception/nex_scaling_ne2_6seed.json
  - research/findings/raw/lanes/perception/nex_scaling_ne4_6seed.json
  - research/findings/raw/lanes/perception/nex_scaling_ne10_6seed.json
  - research/findings/raw/lanes/perception/nex_scaling_ne16_6seed.json
external: EXTERNAL-DONE (DR-gate cleared, recorded) — reward-modulated STDP: Mozafari, Ganjtabesh, Nowzari-Dalini,
  Thorpe & Masquelier 2017 (IEEE TNNLS) + 2018 (Pattern Recognition), R-STDP "extracts task-diagnostic features";
  Fremaux & Gerstner 2016 (three-factor framework); Johnson-Lindenstrauss (random projections preserve pairwise
  distances). The honest negative (a correctly-implemented supervised spiking rule still does not beat the unbiased
  frozen-random bank, and the residual is data) is the deliverable.
builds_on:
  - research/findings/2026-09-17-vision-s2-sail-anti-hebbian-decorrelation-learning-NOGO-underperforms-frozen-random-like-BCM.md
  - research/findings/2026-09-17-vision-satdiv-divisive-norm-is-the-decisive-lever-for-position-invariant-readout-binding-not-load-bearing-GO.md
---

# Vision S2-template learning — all THREE learning classes NO-GO; the residual is DATA-limited, frozen-random+satdiv is GO

The vision position-invariant configural readout is a banked GO with the FROZEN-RANDOM S2 template bank + satdiv
divisive-normalization readout (2026-09-17 satdiv-decisive, 11/12 seeds). The one remaining open question was whether
LEARNING the S2 templates could do better. Two unsupervised-LOCAL rules were already NO-GO (BCM 2026-09-01; SAILnet
2026-09-17), both underperforming frozen-random because a task-blind local rule moves templates off the unbiased
projection. This tests the remaining OPEN class — SUPERVISED / error-guided shaping — and sizes the data-dependence.

## Supervised R-STDP S2 shaping: NO-GO (6-seed) — from research/findings/raw/lanes/perception/rstdp_g0.05_d0.1_ep5_6seed.json
Reward-modulated STDP on the S2 templates (round-robin class-assignment; potentiate winners of the true class toward
the patch, depress winners of other classes; L2-renorm; Mozafari 2017/2018), best explore config (gain 0.05,
depress_scale 0.1, competitive_frac 0.1, 5 epochs), 6 seeds at the satdiv-GO op-point (--n-s2 96 --s2-norm satdiv
--s2-satdiv-n 2.0 --s2-satdiv-sigma 8.0 --s2-satdiv-scale 760.0 --ridge 1.0 --n-glimpses 6 --heldout-position
--scramble-null):
- per-seed capability_go = [True, False, True, False, False, False] = 2/6 (the >=5/6 bar is NOT met).
- LEARNED spiking-WTA held accuracy mean ~0.457 (per seed 0.510/0.438/0.490/0.375/0.479/0.448) vs the frozen-random <!--derived-->
  bank's ~0.477 (GO 5/6) at the SAME op-point — supervised is slightly BELOW, not above. <!--derived-->
- The templates did NOT collapse (mean_pairwise_cosine_abs ~0.59 across seeds, vs the ~0.65 random init) and the
  readout is load-bearing 6/6 (LEARNED spiking-WTA ~0.457 >> RANDOM-readout ~0.238), so the readout works and the <!--derived-->
  anti-cheats are clean (scramble ~0.25, label-shuffle ~0.24, both chance) — this is a genuine METHOD verdict, not a
  broken instrument.

## Verify-first: the naive symmetric depress term COLLAPSED the bank (caught before recording)
<!--derived-->
The first explore used depress_scale 1.0 (depress every other-class winner on every patch). All 8 configs read
byte-identical and BELOW random because the bank COLLAPSED: the per-seed rstdp diag showed mean_pairwise_cosine_abs
~0.985 (near-degenerate) with n_depress:n_potentiate ~3:1 — exactly the 1-true:3-other class imbalance at 4 classes,
i.e. over-depression. That is a METHOD-parameterization artifact (a run whose instrument degenerated), not a
scientific NO-GO; per the verify-instrument-before-a-refutation rule it was corrected (balance the depress term).
The corrected balanced-depress explore did NOT collapse (cosine ~0.45-0.62); 5 of 6 configs still sat below
frozen-random and one (gain 0.05 / depress 0.1) marginally beat frozen-random on seed 42 alone (0.510 vs 0.479) —
which did NOT generalize (the 6-seed above reads 2/6). The function is correct in isolation (unit-tested: it
decorrelates templates and raises class discriminability on a toy problem, and is hyperparameter-sensitive).

## The residual is DATA-limited, not code-limited — from the nex_scaling_ne{2,4,10,16}_6seed.json artifacts
Sizing the FROZEN-RANDOM bank (no learning) vs examples-per-class, 6 seeds, at the same satdiv op-point:
- LEARNED held accuracy rises monotonically: n_ex 2 -> ~0.391 (PARTIAL), 4 -> ~0.471 (PARTIAL), 6 -> ~0.477 <!--derived-->
  (GO 5/6), 10 -> ~0.517 (GO 5/6), 16 -> ~0.543 (GO 5/6). Anti-cheats clean throughout (scramble/label-shuffle at <!--derived-->
  chance; object/position dissociated).
- So the already-banked capability GO holds from the standard example count (6) upward and STRENGTHENS with more
  data, degrading only when examples are cut below it. The frozen-random JL bank is data-limited, not code-limited.

## What it means — the arc closes

Three DIFFERENT S2-template-learning rules now underperform the frozen-random bank at this op-point: BCM
(variance-seeking, unsupervised-local), SAILnet (decorrelation-seeking, unsupervised-local), and R-STDP
(label-driven, supervised). The mechanistic reason is one: a frozen UNBIASED random (Johnson-Lindenstrauss)
projection preserves every input direction, including the rare cross-arrangement configural directions the readout
needs; any activity- or label-driven rule (sharpen toward frequent statistics, spread for coverage, or move toward
class prototypes) systematically moves the templates OFF that unbiased projection and discards exactly those
task-useful directions — and supervised shaping, being the most aggressive, additionally risks collapse. So learning
the S2 templates is neither NEEDED (frozen-random + satdiv is GO) nor IMPROVING (all three classes NO-GO). The
productive lever for the configural readout is DATA (examples/class), directly parallel to the mouth token-supply
finding (capacity/data-limited, not mechanism-limited). `--s2-learn rstdp` is banked default-off alongside
`--s2-learn bcm`/`sail`. A wall defers a METHOD (template learning, now three times), never the capability — the
capability is GO.
