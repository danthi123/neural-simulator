---
type: finding
status: verified
date: 2026-09-17
mechanism: SAILnet/Foldiak three-rule LOCAL unsupervised S2-template learning (Hebbian feedforward + anti-Hebbian
  lateral DECORRELATION + homeostatic threshold, `--s2-learn sail`) — the second attempt to LEARN the vision S2 template
  bank so it beats the frozen-random bank, after unsupervised local BCM was a NO-GO. Explicitly adds a dataset-wide
  pairwise-decorrelation objective (the thing shared-pool BCM lacked)
integration_faculty: perception (vision object readout — S2 template learning)
lane: perception (vision configural / position-invariant readout)
seeds: [42]
verdict: NO-GO at the seed-42 explore (no 6-seed spend earned). At the satdiv-GO operating point (config-B spiking front
  end, held-out-position + scramble-null), SAILnet anti-Hebbian-decorrelation S2 learning UNDERPERFORMS the frozen-random
  bank across ALL 9 explored hyperparameter configs (alpha_l x lca_iters): learned held-out accuracy is well below the
  frozen-random baseline on the same seed (numbers in the body), and below the earlier BCM attempt. So decorrelating the
  templates, done correctly (the lateral matrix grows non-negative and the
  unit-test confirms it reduces pairwise cosine on a toy bars-problem), makes them WORSE for the task, not better. This
  CONFIRMS the mechanistic prediction: an UNBIASED random projection preserves all input directions without bias
  (Johnson-Lindenstrauss); both variance-seeking BCM and coverage-seeking SAILnet decorrelation move the templates AWAY
  from that unbiased projection and discard the rare, task-useful directions the configural readout needs. Banks the
  SECOND unsupervised-LOCAL S2-learning method as insufficient. The remaining open mechanism for S2-template learning is
  SUPERVISED / error-guided template shaping (the class neither BCM nor SAILnet is). Additive, `--s2-learn sail`
  default-off, byte-identical when off; wires nothing.
runner: research/runners/_vision_lindiscrim_readout_derisk.py (--s2-learn sail)
artifacts:
  - research/findings/raw/lanes/perception/vlin_sail_seed42_al0.02_it15.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc760_r1p0_nglim6_heldoutpos_scramblenull_6seed.json
external: EXTERNAL-DONE (perception lane, read + lane-recorded in the design pass) — SAILnet (Zylberberg, Murphy &
  DeWeese 2011), Foldiak 1990 (anti-Hebbian decorrelation), Diehl & Cook 2015 (STDP + lateral inhibition spreads
  prototypes). The honest negative (a correctly-implemented decorrelation rule still underperforms the unbiased random
  bank) is the deliverable.
builds_on:
  - research/findings/2026-09-01-vision-s2-bcm-template-learning-NOGO-collapses-without-competition-underperforms-baseline-with-it.md
---

# Vision S2-SAIL anti-Hebbian-decorrelation learning — NO-GO, underperforms frozen-random (like BCM)

The vision object-readout residual is the FROZEN-RANDOM S2 template bank (the readout side is exhausted; satdiv is the
decisive readout lever, GO). Unsupervised local BCM S2-learning was a NO-GO (2026-09-01) because its shared-pool
templates converged redundant. This tries the biologically-correct fix that BCM lacked — an EXPLICIT dataset-wide
pairwise decorrelation objective (SAILnet/Foldiak anti-Hebbian lateral inhibition + homeostatic threshold).

## Result (from research/findings/raw/lanes/perception/vlin_sail_seed42_al*.json)
<!--derived-->
Seed-42 explore, 9 configs (alpha_l in {0.01,0.02,0.05} x lca_iters in {10,15,20}) at the satdiv-GO op-point
(--n-s2 96 --s2-norm satdiv --ridge 1.0 --n-glimpses 6 --heldout-position --scramble-null):
- LEARNED held-out spiking-WTA accuracy: 0.24-0.35 across all 9 (best 0.354 at alpha_l=0.02/iters=15), every one a
  NO-GO or low PARTIAL.
- Frozen-random baseline (same seed, --s2-learn none): 0.479. So SAIL underperforms frozen-random by ~0.12-0.24 on
  every config, and underperforms the earlier BCM attempt (~0.34) too.
- The mechanism is implemented correctly (the pure-numpy unit test confirms the lateral matrix grows non-negative and
  reduces pairwise cosine on a toy bars-problem; the unbounded spec updates needed internal stability caps). It works as
  designed — it just does not help the task.
- No 6-seed confirm was spent: the pre-registered rule requires beating frozen-random on the explore first, and it
  fails on seed 42.

## What it means — the redirect

Two DIFFERENT unsupervised-LOCAL template-learning rules (BCM variance-seeking; SAILnet coverage/decorrelation-seeking)
now both underperform the frozen-random bank at this op-point. The mechanistic reason is the same for both: a frozen
UNBIASED random projection preserves every input direction without bias, so it keeps the rare, low-frequency,
cross-arrangement directions the CONFIGURAL readout depends on; any local, activity-driven rule (whether it sharpens
toward frequent statistics like BCM or spreads for coverage like SAILnet) systematically moves the templates OFF that
unbiased projection and discards exactly those task-useful directions. Local unsupervised statistics do not carry the
configural (cross-slot-arrangement) signal — the exact limit the 2026-09-01 finding diagnosed, now doubly confirmed. The
remaining OPEN mechanism for S2-template learning is therefore SUPERVISED / error-guided template shaping (a rule that
sees the task/label signal), which is a DIFFERENT class from both banked methods. The frozen-random bank + satdiv readout
(GO, 11/12 seeds) stays the production path; `--s2-learn sail` is banked default-off alongside `--s2-learn bcm`. A wall
defers a METHOD (unsupervised-local template learning, now twice), never the capability (a learned S2 code) — the next
method is supervised.
