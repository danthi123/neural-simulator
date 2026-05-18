# Owner-authorized fair-scale dendritic GLR-2017 MNIST run, SOUND instrument — honest VOID (the STRONGEST triangulation: even with a verified-sound positive control at the literature's own fair scale, the discriminating control does not fail)

## TL;DR

The owner-authorized Option-2 fair-scale run was re-executed after the
design-prescribed instrument fix (the prior VOID was a genuine 128x
batch-sum optimizer bug, honestly surfaced by the V1 gate; fixed
mode-agnostically with sigmoid kept + frozen science bars + moat +
committed rule + every adversarial invariant byte-unchanged; the V1
positive-control independently controller-verified sound at 0.976
heldout on real MNIST before re-launch). The decisive multi-seed GPU
run COMPLETED cleanly (kill-safe; real MNIST 60000/10000
cache-verified; 15 legs).

**Result: GATE = VOID, 3/3 seeds — but for the OPPOSITE reason to
last time, which makes it the strongest triangulation in the whole
arc.** Recomputed from the recorded JSON (mandatory anti-cheat
smell-test, no re-run, no bar-tuning), seeds 42/43/44:

- **V1 (positive control) PASSES, robustly:** true-gradient `oracle`
  heldout = **0.9772 / 0.9770 / 0.9779** (>= 0.95). The instrument is
  SOUND this time -- backprop genuinely trains the deep MLP on real
  MNIST. `biologically_local`=True, `has_controls`=True all seeds.
- **V2 (the discriminating control) FAILS, decisively:** the
  WRONG-SIGN dendritic rule heldout = **0.9487 / 0.9611 / 0.9643**
  (bar <= 0.30). An *inverted* hidden rule generalizes to ~0.95 on
  real MNIST -- as well as, or better than, `local_correct` (which is
  wildly seed-variable: **0.6489 / 0.9527 / 0.1998**).
- The other controls fail correctly: `global_scalar`
  0.0766/0.0868/0.0987 (~chance), `permuted` 0.0974/0.0974/0.0958
  (~chance). So the instrument discriminates the point-neuron analog
  and label-permutation -- but it does NOT discriminate the hidden
  rule's sign/correctness.

The pre-registered THREE-STATE gate correctly returned **VOID** (V2
instrument-validity unmet -> a science PASS/FAIL is meaningless when a
wrong-sign rule also "succeeds"). Working exactly as engineered.

## Why this is the STRONGEST triangulation (and what it means, no spin)

Every prior probe (cheap 1-layer, cheap 3-layer, teach-then-wean, the
prior fair-scale VOID) was either non-discriminating or had a broken
positive control. This run **closes the last escape**: with a
**verified-SOUND** positive control (oracle 0.98) at the **literature's
own canonical discriminating regime** (deep MLP, real MNIST, the exact
Lillicrap-2016 / GLR-2017 feedback-alignment setup), the wrong-sign
control STILL does not fail. The readout-over-rich-hidden-features
confound -- a correctly-trained output layer rescues the task
regardless of the hidden rule's sign/correctness -- is **NOT defeated
even here**, at feasible local scale. `local_correct`'s wild
seed-variance (0.20-0.95) while wrong-sign is consistently ~0.95
*reinforces* this: the readout carries generalization; the hidden
rule's correctness is not what determines it.

Honest scope (no overclaim, no underclaim):
- **NOT** "dendritic credit assignment is impossible." GLR-2017 /
  Sacramento-Senn demonstrate discriminating feedback alignment -- but
  evidently their *discriminating* regime (where wrong-sign genuinely
  fails) requires more than MNIST + a 4-hidden-layer MLP + 60 epochs
  at feasible local scale (harder tasks / conv nets / much longer
  training / specific architectures).
- **IS:** the science question "does the biologically-LOCAL rule's
  correctness genuinely drive learning" is **not validly answerable at
  feasible local scale** -- the discriminating instrument is not
  constructible even at the literature's own fair MNIST scale with a
  verified-sound positive control. This converges with, and is the
  strongest version of, every prior triangulation of the same
  joint-(scale x discriminating-regime) boundary the entire project
  has independently hit.
- This is a VOID (no valid science verdict obtainable here), NOT a
  science FAIL of the rule and NOT a PASS. Reported without spin.

## Why this is NOT config-cranked, and the genuine remaining option (OWNER decision)

The design pre-registered Arch B (CIFAR/conv) and Arch C
(bio-W->A integration) as explicit LATER increments, and that "an
Arch-A FAIL/VOID is the honest terminus, NOT a license to escalate."
The owner authorized Option 2 = the MNIST-FA fair-scale run
specifically. Autonomously escalating to CIFAR/conv/longer-training
now would be config-cranking past the owner-authorized scope toward a
desired outcome -- exactly what the discipline forbids. So this honest
VOID is propagated and the genuine remaining option is handed to the
OWNER as an eyes-open strategic decision, NOT taken autonomously:

> The only thing that *might* make the wrong-sign control genuinely
> fail is the literal harder-than-MNIST GLR-2017 regime (real
> vision/CIFAR, conv architecture, much longer training -- the
> settings where the literature's *discriminating* result lives).
> Honest facts for the decision: (a) NO feasible-local configuration
> tested -- including the literature's own canonical MNIST-deep-MLP FA
> setup with a verified-sound positive control -- produced a
> discriminating instrument; (b) it converges on the project-wide
> joint-infeasibility boundary established from many independent
> directions; (c) it is a further, larger, speculative investment with
> no cheap positive precursor. Authorize it deliberately (eyes-open),
> or accept the boundary. I will not autonomously spend on it against
> a non-discriminating signal at the very regime the literature uses.

## What is preserved / validated (unaffected)

Byte-UNMODIFIED + green across the WHOLE fair-scale build incl. the
instrument fix: the no-confabulation moat (`abstention_gate` +
`tests/test_abstention_gate.py` 7/7), every frozen `*_core` (incl.
`dendritic_fair_core` `_DFAIR_*`), `sim.dendritic_plasticity` (the
committed credit-assignment rule, sigmoid-faithful, 1e-9 exact),
`sim.train_checkpoint`, `sim/bptt_snn*`, `sim/bridge.py`. The
instrument fix was a verified pure mode-agnostic optimizer calibration
(40/40 green; the discriminating power, if it existed at this scale,
would have been preserved/strengthened -- and the controls prove the
gate DOES discriminate global-scalar + permuted; it simply cannot
discriminate the hidden rule's sign at this scale).

## Anti-cheat discipline (why this VOID is trustworthy)

The pre-registered THREE-STATE + V1/V2 instrument-validity-first
design did exactly its job twice: a broken instrument (prior run) ->
VOID; a sound-but-non-discriminating instrument (this run) -> VOID.
Neither was ever fabricated into a science PASS/FAIL. The instrument
fix was diagnosed (128x batch-sum bug), corrected mode-agnostically
with the frozen science bars + committed rule byte-untouched, and the
V1 fix independently controller-verified before re-launch (not trusted
on report). Every number recomputed from the recorded JSON; MNIST
provenance verified real; no bar tuned; no terminated mechanism
re-tread; the further-escalation option handed to the owner rather
than autonomously config-cranked. The validated no-confab moat -- the
project's distinctive contribution -- remained byte-identical and 7/7
green throughout the entire arc.

## Files / evidence

- Result: `research/findings/raw/g11_bg/dendritic_fair_gate.json`
  (GATE VOID; per-seed oracle 0.977x [V1 PASS], wrongsign 0.95x
  [V2 FAIL], global_scalar/permuted ~chance, local_correct
  0.65/0.95/0.20; MNIST 60000/10000 cache-verified).
- Instrument fix (verified sound): commit 1860044
  (`sim/dendritic_mlp.py`, mean-batch+momentum+standardization,
  sigmoid kept); independent V1 recompute 0.9758 heldout.
- Supersedes-in-context: `2026-05-18-dendritic-fairscale-glr2017-
  VOID.md` (prior instrument-broken VOID) and triangulates with
  `2026-05-17-dendritic-faithful-instrument-TERMINUS.md` +
  its teach-then-wean addendum.
- Design/plan: `docs/plans/2026-05-17-dendritic-fairscale-glr2017-
  {design,implementation}.md`.
