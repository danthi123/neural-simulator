# Owner-authorized Option-2 (CIFAR + conv + long) — the load-bearing falsify-cheaply gate ran FIRST and did NOT green-light: a SOUND discriminating conv instrument is not cheaply constructible under the sigmoid-faithfulness constraint (the project-wide boundary, triangulated again in the conv regime); the heavy run is handed to the OWNER eyes-open, NOT autonomously launched

## TL;DR

The owner chose Option 2 a second time: the literal harder GLR-2017 /
Bartunov-2018 discriminating regime (CONVOLUTIONAL net + CIFAR-class
task + long training). The pre-registered design made the
**falsify-cheaply probe the load-bearing gate BEFORE any week-scale
spend** (KEY DESIGN POINT 4: "if even a cheap conv config is
non-discriminating / not soundly constructible -> the heavy run is NOT
launched; if it IS discriminating + V1 sound -> green light"). That
cheap gate has now run, with full anti-cheat discipline, and the honest
result is:

**Durable POSITIVES (real engineering results, reported without
diminishment):**
- **Conv-via-im2col committed-rule faithfulness PROVEN at 8.9e-15.**
  The batched im2col conv-feedback-alignment hidden update EQUALS the
  BYTE-FROZEN dense `sim.dendritic_plasticity.urbanczik_senn_update`
  applied per (sample, spatial-position) im2col patch and summed. The
  crux design risk -- can the committed dense sigmoid-derivative rule be
  applied faithfully under convolution -- is RETIRED.
- **The hand-derived conv-backprop oracle is gradcheck-correct at
  1.05e-9** vs central finite differences (NO autodiff anywhere). The
  measurement/validity path code is sound.

**The cheap gate did NOT green-light (no spin):**
- A **verified-correct true-gradient SIGMOID conv** (the V1 positive
  control) could **not be cheaply trained** on a synthetic conv-
  relational task that a **standard ReLU-MLP true-gradient learner
  solves at 100% heldout** -- across two scale regimes
  (cap16/ep100 lr 1-20; cap32/ep220/bs64 lr 5-200) and a 40x learning-
  rate sweep, oracle stayed at chance (0.25-0.26, 4-class).
- `local_correct` (committed rule) was a **non-robust, config-specific
  artifact**: 0.67 heldout at one cheap config, **0.235 (chance) at
  another**. It was correctly NEVER interpreted as a science result --
  the anti-cheat refusal was vindicated when it collapsed.

**Conclusion, rigorously guarded:** the bottleneck is specifically the
**sigmoid-faithful conv instrument's trainability**. Sigmoid is
HARD-REQUIRED -- the committed rule hard-codes the sigmoid derivative;
a ReLU conv (which trivially solves the task) would break the frozen
faithfulness invariant, so it is not a knob that may be turned. This is
the **same joint-(scale x discriminating-regime) infeasibility boundary
the entire prior arc independently hit**, now triangulated again and
more sharply in the conv regime the owner specifically pointed to --
exactly as the MNIST instrument-broken VOID showed for the dense regime
(the cheap *positive control* itself is not cheaply soundly
constructible).

Per the pre-committed contract and the systematic-debugging iron law
(3 root-caused bounded iterations + a decisive task-validity guard ->
STOP, do not crank fix #4), the week-scale CIFAR+conv run is **NOT
autonomously launched**, and the genuine remaining option is handed to
the OWNER as an eyes-open strategic decision.

## The honest 4-step arc (every step root-caused, not cranked)

1. **Cheap config #1 (imbalanced synthetic task) -> CHEAP-VOID.** Root
   cause found on review: the relational label was class-imbalanced by
   construction (confounds V1/V2). Fixed -> balanced-by-construction
   (quadrant label sampled uniformly, motifs placed consistently;
   conv-hierarchy-required structure + distractors retained).
2. **Cheap config #2 (balanced, cap16/ep100, lr 1-20) -> CHEAP-VOID;
   oracle at chance.** A gradient-check of the hand-derived conv
   backprop initially "FAILED" at an **exact 5.000e-01** worst relative
   error. Root-caused decisively: that exact 0.5 is the N=3 sum-vs-mean
   fingerprint (analytic sum-grad vs a mean-CE harness; rel =
   |num-3num|/(|num|+3|num|) = 0.5 for every entry). Harness fixed
   (apples-to-apples sum-CE) -> **gradcheck PASS at 1.05e-9: the oracle
   code is CORRECT.** So oracle-at-chance is genuine instrument-
   conditioning (vanishing true-gradient through the deep sigmoid conv
   stack), NOT a code bug and NOT "local beats backprop".
3. **Cheap config #3 (mode-agnostic budget/lr conditioning:
   cap32/ep220/bs64, oracle-only V1 calibration over lr {5,20,80,200};
   optimizer + faithfulness invariant + committed rule + frozen verdict
   bars ALL byte-identical; no wrong-sign peek) -> CHEAP-VOID.** Oracle
   still chance across the entire sweep; `local_correct` collapsed
   0.67 -> 0.235 (proving the earlier 0.67 was a non-robust artifact);
   all three modes at chance.
4. **Decisive task-validity guard (against the premature-terminus
   error):** a standard 1-hidden-ReLU-MLP true-gradient learner (fenced
   as validity-only, explicitly NOT the science instrument) achieves
   **1.0000 heldout** on the same `make_relpos` task. The task is
   perfectly well-posed; the failure is genuinely instrument-
   constructibility, so the terminus stands rigorously.

## Honest scope (no overclaim, either direction)

- **NOT** "convolutional / dendritic credit assignment is impossible."
  Bartunov-2018 demonstrates discriminating results -- but at full
  CIFAR/ImageNet conv scale with their specific architectures and long
  training, NOT at a cheap synthetic sigmoid-conv probe.
- **NOT** a proof the heavy CIFAR+conv+long run would fail. A cheap
  synthetic-task sigmoid-conv probe is a deliberately weak proxy; the
  real regime is precisely what the owner authorized BECAUSE there is
  no cheap precursor. The honest, bounded claim: **the falsify-first
  gate could not cheaply construct a sound discriminating conv
  instrument; it provides NO positive precursor for the heavy run; it
  triangulates the same boundary harder.**
- **IS:** under the non-negotiable sigmoid-faithfulness constraint, a
  sound discriminating instrument for biologically-local conv credit
  assignment is not constructible at feasible cheap scale -- the cheap
  positive control (true-gradient sigmoid conv) is not cheaply
  trainable on a verified-learnable task. This converges with, and
  sharpens, every prior triangulation of the same boundary.
- This is an honest **BOUNDARY** propagation (no valid science verdict
  obtainable from the cheap gate), NOT a science FAIL of the rule and
  NOT a PASS. The conv-FA *machinery* (faithfulness + oracle code) is
  sound; a *trained sound discriminating instrument* at cheap scale is
  what is not constructible.

## Why this is NOT config-cranked, and the genuine remaining option (OWNER decision)

The design pre-registered, verbatim, that "a cheap FAIL/VOID is the
honest terminus, NOT a license to escalate" and prescribed the
VOID->fix-instrument->re-run loop *kept cheap*. Three bounded,
root-caused iterations + a decisive task-validity guard exhaust that
loop honestly. The only "fix" that would make the cheap oracle train
(ReLU instead of sigmoid) would violate the BYTE-FROZEN committed-rule
faithfulness invariant -- forbidden. Continuing to crank cap/epochs/
init/normalization toward a desired outcome is exactly the
config-cranking the discipline forbids. So this honest BOUNDARY is
propagated and the genuine remaining option is handed to the OWNER, NOT
taken autonomously:

> The only thing that *might* yield a sound discriminating conv
> instrument is the literal heavy regime the owner already authorized
> (real CIFAR-10, longer training, the conditioning the literature's
> discriminating result actually uses). Honest facts for the decision:
> (a) the cheap falsify-first gate found **NO positive precursor** -- a
> verified-correct true-gradient sigmoid conv could not be cheaply
> trained on a task a standard ReLU-MLP solves at 100%; (b) the
> conv-FA machinery IS sound (faithfulness 8.9e-15, oracle gradcheck
> 1e-9) so a heavy run is *buildable*; (c) it converges on the
> project-wide joint-infeasibility boundary established from many
> independent directions; (d) it is a larger speculative week-scale
> spend with no cheap positive precursor. Authorize it deliberately
> (eyes-open), or accept the boundary. I will not autonomously spend
> the week against a non-constructible cheap signal -- mirroring the
> prior fair-scale VOID's owner-decision handoff.

## What is preserved / validated (unaffected)

The entire cheap gate ran via a single THROWAWAY probe
(`_probe_conv_fa.py`, prefix `_`, never committed, deleted after the
decision) that **imported** the committed rule + backend READ-ONLY and
modified nothing. Byte-UNMODIFIED + green throughout: the
no-confabulation moat (`research/runners/abstention_gate.py` +
`tests/test_abstention_gate.py`), `sim/dendritic_plasticity.py` (the
committed credit-assignment rule), every frozen `*_core` (incl.
`dendritic_fair_core` `_DFAIR_*`), `sim/dendritic_mlp.py`,
`sim/train_checkpoint.py`. NO automatic differentiation anywhere -- the
oracle is a hand-derived numpy/cupy conv backprop, gradcheck-verified,
fenced as measurement/validity only; the ReLU-MLP task-validity guard
is likewise fenced validity-only.

## Anti-cheat discipline (why this BOUNDARY is trustworthy)

The desirable-looking number (`local_correct`=0.67) was scrutinized
HARDEST and explicitly refused interpretation -- and was then proven a
non-robust artifact when it collapsed to chance at another cheap
config. The oracle was gradcheck-verified before any conclusion (bug vs
conditioning, decisively separated; the exact-0.5 fingerprint
diagnosed). The task was validity-checked before any terminus
(ReLU-MLP 100% -> task well-posed, not bugged -- guarding the
premature-terminus error the owner corrected earlier in this arc). V1
was calibrated oracle-only (no wrong-sign peek, so tuning could not be
biased toward a desired V2). Three iterations were each root-caused,
not cranked. The pre-committed either-way contract was honored to the
letter; the heavy-run decision is handed to the owner, not
autonomously escalated. The validated no-confab moat -- the project's
distinctive contribution -- remained byte-identical and green
throughout.

## Files / evidence

- Throwaway probe (deleted post-decision; recorded here for the trail):
  `_probe_conv_fa.py` -- Part A faithfulness 8.882e-15; Part A2 oracle
  gradcheck 1.052e-9; Part B oracle chance across cap16/ep100 lr1-20 +
  cap32/ep220/bs64 lr5-200; `local_correct` 0.67 (cap16) -> 0.235
  (cap32); Part B0 standard ReLU-MLP true-grad heldout 1.0000.
- Supersedes-in-context / triangulates with:
  `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-triangulation.md`
  (fair-scale MNIST sound-instrument VOID),
  `2026-05-18-dendritic-fairscale-glr2017-VOID.md` (instrument-broken
  VOID), `2026-05-17-dendritic-faithful-instrument-TERMINUS.md`.
- Design/plan that pre-registered the falsify-cheaply gate:
  `docs/plans/2026-05-17-dendritic-fairscale-glr2017-{design,implementation}.md`
  + the brainstorm arguments for the owner-authorized CIFAR+conv
  Option-2 escalation.
