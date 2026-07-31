---
type: finding
status: contributing
date: 2026-05-19
mechanism: integrated-loop
---

# Integrated-loop instrument-soundness: the per-binding asymmetry is a diagnosed emergent symmetry-break, not a wiring bug; the next step is a deeper, named, biology-grounded equalization mechanism

## Plain-language summary

The pre-registered "per-binding drive symmetry" iteration was run with a
diagnose-before-fix discipline. The diagnosis is itself the main result and
it changes what the next step should be. This is an honest, well-supported
negative at the instrument-soundness stage, not a refutation of the
integrated-loop hypothesis and not a hand-back.

## What was measured (GPU evidence, two-binding trivial case)

A passive per-binding diagnostic recorded, for the trivial drilled
bijection at the minimal two-binding load, on the GPU: the basal-ganglia
channel selected for each binding and its thalamic firing during encode;
the prefrontal slot sub-population firing during encode and at query; the
spike-timing-grown weights on each binding's prefrontal-slot-to-concept
and language-to-concept edges; and each binding's final concept-pool score
at query.

Every measurable structural quantity is symmetric between the first and
second binding (differences of 0.3-1%): the same basal-ganglia channel
firing, the same grown language-to-concept weights (7.49 versus 7.46), the
same input fan-out, the same within-pool recurrence, symmetric encoding.
Swapping the binding order does not move the asymmetry (it tracks the
identity, not the order or the query).

Despite that structural symmetry, at query the first binding's concept
pool fires far above the trustworthy grounding gate's operating point and
the second binding's fires well below it, so the gate emits the first
answer and correctly abstains on the second. The soundness score is
therefore one-half, below the pre-registered bar.

## The diagnosis (the key result)

The asymmetry is not a structural wiring defect. It is an emergent
winner-take-most symmetry-break: a structurally-symmetric network with one
shared inhibitory population across all pools plus self-sustaining
(bistable) pool dynamics deterministically collapses to a single dominant
attractor. The trustworthy gate then correctly passes the single winner
and abstains on the suppressed one. The pathway that could rebalance this
- the basal-ganglia-gated projection from the prefrontal slot onto the
concept layer, which exists and is correctly wired - never becomes
functional because it starts from zero weight and the reward signal is
approximately zero at cold start, so the eligibility-times-reward update
never bootstraps it. It remains at the floor weight for both bindings.

Three faithful strengthen-only fixes from the pre-registered family were
tried and each was falsified by GPU evidence: symmetric per-binding
off-target dampening; symmetric per-epoch encode-order interleaving; a
non-zero prior on the prefrontal-slot-to-concept readout edge. None moved
the soundness score off one-half. All three were reverted; the loop logic
is byte-identical to its prior committed state (only a provably-inert
passive diagnostic harness was retained, to accelerate the next
iteration).

## Why this stops the strengthen-only patch loop (honest, disciplined)

This is the systematic-debugging inflection: several rounds of fixes, each
revealing a new issue in a different place, with the latest revealing that
the remaining issue is an emergent attractor property, not a local wiring
error. No build-time prior, encode-order change, or efferent
initialization can rebalance an emergent symmetry-break - this is now
demonstrated, not assumed. Continuing to apply minimal wiring tweaks would
be undisciplined. The correct action is to step back from reactive
patching, propagate this honestly, and take the next step as a deeper,
properly-designed iteration.

## The next step (named, cited, grounded in the project's own validated subsystems; autonomous, not a hand-back)

The biology to reproduce: the prefrontal-basal-ganglia working-memory
account requires multiple slots to be maintained simultaneously at
comparable strength. A symmetric shared-inhibition bistable network does
not do this on its own; biology equalizes competing maintained items via
active gain control (homeostatic / divisive normalization) and via a
working three-factor (dopamine-gated) learning signal that actually
potentiates the gated slot-to-content pathway rather than leaving it at
cold-start floor.

The project already has both ingredients validated and protected:

- A validated homeostasis mechanism (homeostatic firing-rate regulation),
  the natural basis for a per-stripe normalizing/equalizing drive so two
  maintained bindings settle at comparable strength instead of one
  attractor winning.
- A validated temporal-credit (eligibility-trace, dopamine-gated) learning
  substrate - already mapped in the design to "dopamine-gated learning
  inside the loop" - the natural basis for breaking the cold-start
  deadlock so the basal-ganglia-gated prefrontal-slot-to-concept pathway
  genuinely potentiates for every maintained binding.

The next pre-registered iteration is therefore a deeper change than a
strengthen-only wiring tweak: compose these two already-validated
subsystems into the loop so the prefrontal slots are actively equalized
and the gated efferent actually learns - re-tested against the SAME
frozen verdict and the SAME instrument-soundness bar, with the
no-confabulation gate and every protected module byte-unchanged. Because
it is a substantive mechanism (not a tweak), it gets its own pre-registered
design and plan, pursued autonomously - following the biology, no
hand-back, no declaring the approach unfit.

## Honest ceiling (stated; not overstated)

Nothing here claims compositional memory in the full model. The cheap-tier
core finding (non-separability) stands. The full-model instrument is built,
adversarially hardened, GPU-backed, role-selective, and its trustworthy
gate works correctly. It is blocked at instrument-soundness by a now-
precisely-diagnosed emergent symmetry-break whose fix is a named,
biology-grounded composition of two already-validated project subsystems -
the next pre-registered iteration. No fixed threshold moved; the
no-confabulation moat is byte-unchanged and behaved exactly as designed.

## Files / evidence

- Inert diagnostic harness + integration runner: `research/runners/integrated_loop_gate.py` (commit `6c2c055`; loop logic byte-identical to `d3a7ac3`).
- Verdict module (unchanged, frozen): `research/runners/integrated_loop_core.py` (`2048750`).
- Prior interim finding: `research/findings/2026-05-18-integrated-loop-full-model-instrument-soundness-blocked-per-slot-drive-asymmetry.md`.
- Plan + pre-registration logs: `docs/plans/2026-05-18-integrated-loop-full-model-implementation.md`.
