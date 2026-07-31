---
type: finding
status: contributing
date: 2026-05-19
mechanism: integrated-loop
---

# Integrated-loop iteration 2: per-stripe homeostatic equalization transfers and works; the temporal-credit cold-start break is correctly wired but blocked by the documented zero-initialization precondition

## Plain-language summary

Iteration 2 composed two of the project's own already-validated,
protected subsystems into the closed loop to fix the diagnosed
instrument-soundness blocker: (1) the validated homeostatic
firing-rate regulation, to equalize the prefrontal stripes so multiple
bound items are maintained at comparable strength; and (2) the
validated temporal-credit (eligibility-trace, dopamine-gated) learning
signal, to break the cold-start so the basal-ganglia-gated
prefrontal-slot-to-concept pathway actually potentiates. The result is
a precise, high-confidence, single-root-cause negative that names the
documented next fix. This is the iterate-following-biology process
converging, not thrashing.

## What was verified on the GPU (CuPy, RTX 3090)

Diagnosis-before-change confirmed both pre-existing root causes: the
slot-to-concept efferent weight was at the zero-initialization floor
for both bindings, and the winner-take-most collapse left only the
first binding above the trustworthy grounding gate.

**Homeostatic equalization (mechanism 1) transferred and works as
designed.** Enabling the validated homeostasis (its kernel and bounds
byte-unchanged; only the runner's configuration enables and scopes it,
with rate parameters inside the validated family) released the
suppressed stripes — the previously-silent off-target concept pools
rose by roughly five-fold and per-stripe equalization is visibly
active. This is genuine forward progress and removes the emergent
symmetry-break as a blocker.

**The temporal-credit cold-start break (mechanism 2) is correctly
wired but did not transfer, for a single documented reason.** The
reused temporal-credit idiom is faithful (the validated pattern
byte-unchanged, applied during encoding only, no query-time teaching
or hard-feed, no automatic differentiation, no extra randomness). Yet
the measured learning eligibility on the slot-to-concept efferent
stays exactly zero for both bindings even though the prefrontal slot
fires massively. Root cause: that net-new efferent is initialized at
exactly zero weight. A truly-zero synapse injects no current, so it
never produces the pre-then-post firing that spike-timing plasticity
needs to charge eligibility; the learning update is therefore
learning-rate times error times zero, which is zero regardless of how
correctly the reward is timed. This is exactly the project's already
documented zero-initialization gotcha ("a zero-initialized pathway
carries no current"), the same root cause as the earlier text input
output finding whose fix was non-zero readout-pathway initialization
(biologically grounded as spontaneous baseline cortical weights,
Barlow 1972).

## The named next step (documented project fix; not a new mechanism)

Initialize the net-new slot-to-concept efferent with a small non-zero
prior (a modest mean with jitter), exactly the precondition the
validated reference runners already satisfy for their scored pathways.
With a current-carrying efferent, the now-correctly-timed
temporal-credit reward can actually charge eligibility and potentiate
the pathway for every maintained binding, while the now-working
homeostatic equalization keeps the stripes balanced. This is a single
configuration change on the pathway the runner already adds (its
weight mean and jitter) — not a new mechanism, not a change to any
protected or validated module, not a change to any fixed success
threshold, and not a change to the trustworthy no-confabulation gate.

## Honest status and discipline

This is a faithful negative, propagated honestly; nothing was
committed as a soundness pass. The verified-correct foundation
(both levers wired faithfully, homeostasis working) is preserved as an
honest work-in-progress commit. The cheap-tier core finding
(non-separability) is unaffected; no fixed threshold moved; the
no-confabulation gate is byte-unchanged and behaved exactly as
designed. The diagnosis quality increased markedly relative to
iteration 1 (a precise mechanistic single root cause with a documented
project fix, versus a general symmetry-break observation), which is
why this warrants one focused next iteration applying the documented
fix rather than a step-back. Honest bound: if instrument-soundness
still fails after correctly applying the documented non-zero
initialization on top of the two now-faithful mechanisms, the evidence
would then point to a deeper architectural limit, which would be
surfaced as a genuine architecture question rather than another
configuration iteration.

## Files / evidence

- Integration runner honest-wip (lever 1 + lever 2; acceptance not
  met): `research/runners/integrated_loop_gate.py` (commit `5c27e99`).
- Verdict module (frozen, unchanged): `research/runners/integrated_loop_core.py` (`2048750`).
- Prior iteration-1 diagnosis: `research/findings/2026-05-19-integrated-loop-v1-asymmetry-is-an-emergent-symmetry-break-not-a-wiring-bug.md`.
- Plan + pre-registration trail: `docs/plans/2026-05-18-integrated-loop-full-model-implementation.md`.
