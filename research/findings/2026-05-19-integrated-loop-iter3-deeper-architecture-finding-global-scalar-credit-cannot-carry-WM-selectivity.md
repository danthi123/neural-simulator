---
type: finding
status: contributing
date: 2026-05-19
mechanism: integrated-loop
---

# Integrated-loop iteration 3: the documented zero-initialization fix works (episodic binding now perfect), but a global scalar three-factor signal does not carry role-selective working-memory binding — a deeper, well-triangulated architecture finding

## Plain-language summary

Iteration 3 applied the project's own documented fix for the
zero-initialization precondition (a small non-zero, binding-agnostic
prior on the prefrontal-slot-to-concept pathway). On the GPU this
mechanically did exactly what was predicted: it unblocked current
flow and learning, so the episodic-sequence dimension now binds
perfectly. But role-selective working-memory binding still does not
form. The evidence now precisely localizes the remaining gap and it
converges with a verdict the project already reached on its own. This
is the iterate-following-biology process converging on a structural
finding, not the "it failed, move on" pattern: every faithful part is
preserved and the next step is the project's own validated mechanism
for exactly this gap.

## What iteration 3 established (GPU-verified, high confidence)

- The documented non-zero-initialization fix is correct and worked.
  The prefrontal-slot-to-concept pathway now carries current, so
  spike-timing eligibility charges and learning proceeds. The
  drilled-binding readout scores rose comfortably above the trustworthy
  grounding gate and the episodic-sequence recall dimension is now
  perfect.
- It is not a disguised answer key. The prior was a single uniform,
  binding-agnostic value on every edge; the failure mode is
  non-selective collapse onto one pool, which is the opposite of an
  answer being fed in.
- Role-selective working-memory binding still does not form. With the
  basal-ganglia causal wiring intact, the temporal-credit signal
  faithfully wired, the per-stripe homeostatic equalization working,
  and the documented initialization applied, every cue still collapses
  onto a single concept pool in working memory. The working-memory
  soundness score is zero while the episodic score is perfect.

## The deeper architecture finding (triangulated with the project's own prior verdict)

The integrated loop is now mechanically sound on every dimension
except one, and the remaining gap is precisely identified: a global
scalar three-factor (reward/dopamine, temporal-credit) signal does not
produce role-selective binding in the prefrontal working-memory
dimension of this closed loop, even though the same machinery produces
perfect episodic binding.

This is not a surprise in isolation — it converges with a verdict the
project reached independently months earlier: global scalar feedback
fails to produce selective word-to-action binding at biological scale,
while the architecture itself is sufficient; the credit-assignment
rule is the bottleneck. The same project work then showed the
resolution: a co-firing / Hebbian binding paradigm (with a topographic
prior) produces reliable role-selective binding where the global
scalar signal could not — a large, multi-seed-validated effect (the
validated 16-pool concept-binding substrate reaches high multi-seed
bidirectional binding by exactly this mechanism). The integrated-loop
design already specifies that validated concept-binding substrate as
the concept layer. Iteration 3 has therefore localized the remaining
instrument-soundness gap to precisely the node whose validated
solution already exists in this project.

## Decision (honoring the pre-committed bound; autonomous; following the biology)

A pre-committed bound was in force: a faithful negative after the
documented initialization fix is a deeper architecture question, not
another configuration tweak. That bound is honored — there is no
further configuration iteration. The bound explicitly permitted "a
fundamentally different approach", and the project's own validated
mechanism for this exact selectivity problem is that fundamentally
different approach. Per the standing iterate-following-biology
discipline (no hand-back, no declaring the approach unfit), the next
step proceeds autonomously: carry the working-memory role-selectivity
with the validated embodied co-firing plus topographic-prior binding
mechanism, with the temporal-credit signal relegated to its correct
role (credit and gating, not the binding itself), while keeping every
faithful part already in place (the basal-ganglia causal wiring, the
homeostatic equalization, the documented non-zero initialization, the
episodic binding). This is the design's own concept-layer mandate, not
a new invention.

## Pre-committed bound for the next iteration (stated now, before the run)

If the project's own validated co-firing-plus-topographic-prior
binding mechanism also fails to produce role-selective working-memory
binding inside this integrated loop at the minimal two-binding load,
that is a genuine, program-level refutation of integrated-loop
instrument-soundness at this slice — to be surfaced honestly as a
fundamental finding and a program-level decision, not a further
iteration. Stated in advance so the next outcome cannot be
rationalized.

## Honest status

A faithful negative, propagated honestly; nothing committed as a
soundness pass. The verified-correct current-carrying foundation
(documented non-zero initialization, both faithful levers, episodic
binding perfect) is preserved as an honest work-in-progress commit.
The cheap-tier non-separability core finding is unaffected; no fixed
threshold moved; the no-confabulation gate is byte-unchanged and
behaved exactly as designed. The capability status stays predicted (no
decisive verdict has been produced; this is a structural localization,
not a refutation of the integration hypothesis).

## Files / evidence

- Integration runner honest-wip (documented init applied; acceptance
  not met): `research/runners/integrated_loop_gate.py` (commit `e02f692`).
- Verdict module (frozen, unchanged): `research/runners/integrated_loop_core.py` (`2048750`).
- Iteration-2 finding: `research/findings/2026-05-19-integrated-loop-iter2-homeostasis-works-temporal-credit-blocked-by-zero-init.md`.
- Plan + pre-registration trail: `docs/plans/2026-05-18-integrated-loop-full-model-implementation.md`.
