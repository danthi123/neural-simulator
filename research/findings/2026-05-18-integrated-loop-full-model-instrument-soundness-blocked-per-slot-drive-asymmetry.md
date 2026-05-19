# Integrated-loop full spiking model: the instrument is built and hardened, but instrument-soundness is blocked on a per-binding drive asymmetry; the decisive run is deliberately not yet run

## Plain-language summary

The goal is a working local proof-of-concept of compositional memory built
the way the brain integrates several systems into one closed loop. A cheap
preliminary simulation already robustly confirmed the load-bearing core:
removing any one of three shared systems (the combinatorial binding step, the
single shared timing rhythm, the fast hippocampal store) destroys both the
working-memory query and the episodic-sequence recall together. The
pre-registered decisive test is the full spiking-network model.

This document records an honest interim outcome: the full-model instrument
has been built and hardened through a deliberately adversarial review, real
faithfulness defects were found and fixed, and the loop now behaves
correctly in several important ways — but it does not yet pass its own
pre-registered instrument-soundness check, for a precisely identified and
biologically meaningful reason. The decisive multi-seed run is therefore
deliberately not run yet (running it now would only return "cannot
conclude" at the soundness stage and waste many GPU-hours; catching this at
the review stage is exactly why the review stage exists).

## What was built and what the adversarial review caught and fixed

The success-criteria / necessity-verdict module was written as a pure,
fixed-threshold, fail-closed scorer with a 16-case adversarial test matrix
(all green). A pre-registration self-consistency defect in its
classification ordering was caught and corrected before any run, with every
numeric threshold left unchanged.

The closed-loop integration runner composes the project's already-validated
subsystems (the 16-pool concept representation, the hippocampal relational
store, the prefrontal working-memory region, the basal-ganglia selective
gate, replay consolidation, the neuromodulatory timing, the trustworthy
answer-only-when-grounded output gate) under one net-new shared
theta-gamma timing controller plus the net-new closed-loop wiring. A
multi-pass adversarial review found and forced strengthen-only fixes for
four real defects, in priority order:

1. **The basal-ganglia selective gate was not causally connected to the
   scored readout.** The runner injected gating current into the prefrontal
   region but no pathway carried that to the concept layer the readout
   measures. Fixed by adding the biologically-correct missing wire (the
   thalamocortical projection from the basal-ganglia output channel onto the
   prefrontal working-memory slot, and the slot's projection back onto the
   concept layer), and by deleting a Python-side shortcut that had been
   selecting the slot directly. Slot selection is now genuinely carried by
   the validated basal-ganglia disinhibition cascade.
2. **The trustworthy grounding gate was out of operating range.** Its
   confidence threshold is calibrated for the production-scale concept
   pools; the minimal slice was far below it, so the gate always abstained.
   Fixed by sizing the slice so a genuinely-bound pool reaches the gate's
   operating range, with the gate and its threshold left byte-unchanged.
3. **The concept layer did not inherit the project's validated selectivity
   recipe.** Without the validated topographic-prior / off-target /
   weak-dynamics / lateral-inhibition / interleaved-training recipe, one
   pool dominated the read-out for every cue. Fixed by faithfully
   inheriting the validated recipe; the drilled binding is now
   role-selective (each cue activates its own bound pool).
4. **The instrument-soundness query used the hard generalization probe.**
   The pre-registered design specifies the soundness baseline as the
   trivial drilled binding, with the novel composed combination reserved
   for the genuine science test. The runner had applied the novel probe to
   the soundness baseline too. Corrected to match the pre-registered
   design; the science task (in the full loop and every lesion) is
   unchanged and exactly as hard.

A separate, important correction during this work: the runner had been
accidentally pinned to the CPU backend (inherited from a minimal-slice
template). It now runs on the GPU for the real and decisive paths, with the
CPU backend used only for the fast smoke test.

## The honest blocking result

After those fixes, verified on the GPU: the loop is role-selective (the
validated concept-pool recipe transferred — different cues activate
different bound pools), and the trustworthy grounding gate is operable and
behaving exactly as designed.

At the minimal two-binding load, however, the closed-loop wiring drives the
two bindings asymmetrically. The first binding's concept population fires
strongly and clears the grounding gate, so its answer is emitted and
correct. The second binding's population fires well below the gate, so the
gate correctly abstains rather than emit a weakly-grounded answer. The
result is that one of the two trivial bindings is confidently grounded and
the other is not, so the soundness score is one-half — below the
pre-registered soundness bar.

The important interpretation: this is the project's distinctive trustworthy
property working correctly. The gate is not failing; it is refusing to
confabulate a binding the loop did not drive strongly enough to be
confidently grounded. The defect is upstream, in the net-new closed-loop
wiring: it under-potentiates later bindings relative to the first, so not
every genuinely-maintained binding reaches the grounding threshold.

## Decision and next step (following the reference biology; not a hand-back, not a declared dead-end)

Because the pre-registered instrument-soundness check is not met, the
decisive multi-seed science run is deliberately not run yet — the review
stage exists precisely to prevent spending many GPU-hours on a
not-yet-sound instrument that could only return "cannot conclude".

This is an honest interim non-success at the instrument-soundness stage,
not a refutation of the integrated-loop hypothesis. The cheap-tier core
finding (compositional memory is non-separable from the integrated loop)
stands unaffected. The prior validated subsystems and the project's honest
boundaries are unaffected. No fixed threshold was moved, and the
trustworthy no-confabulation gate is byte-unchanged.

The next step follows the reference biology, as its own pre-registered
iteration: make the closed-loop wiring drive every maintained binding
symmetrically — per-binding drive symmetry through the
basal-ganglia → prefrontal-slot → concept chain — so that every binding the
loop genuinely maintains is potentiated strongly enough to be confidently
grounded, not only the first. That symmetric-updating discipline is itself
documented in the reference catalog (the basal-ganglia gate updates each
prefrontal slot independently and equivalently; multiple slots are held
simultaneously with comparable strength). This is the named, cited
biology-fidelity gap to close next.

## Honest ceiling (stated; not overstated)

Nothing here claims compositional memory has been demonstrated in the full
model. It claims: the full-model instrument has been built and
adversarially hardened; on the GPU it is role-selective and its trustworthy
gate works; and it is currently blocked at instrument-soundness by a
specific, named, biologically meaningful wiring asymmetry that is the next
iteration target. No claim of fluent language, no claim of a large language
model, no claim the decisive test has passed.

## Files / evidence

- Verdict module: `research/runners/integrated_loop_core.py` (16/16
  adversarial matrix; fixed thresholds; commit `2048750`).
- Integration runner (honest wip state, GPU + role-selective +
  V1-conformance, soundness not yet met): `research/runners/integrated_loop_gate.py`
  (commit `d3a7ac3`).
- Implementation plan + transparent correction logs:
  `docs/plans/2026-05-18-integrated-loop-full-model-implementation.md`.
- Design (pre-registered): `docs/plans/2026-05-18-Q5-integrated-biology-grounded-closed-loop-design.md`.
- Prior cheap-tier finding (core confirmed):
  `research/findings/2026-05-18-integrated-loop-probe-core-confirmed-helpers-need-full-model.md`.
