# Dedicated adversarial review of the FHRR-biologization arc = CLEAR; the arc's claims are genuine and the biologization-arc result is recorded as a capability_status pillar

## Status

The FHRR-biologization arc (replacing the phase-coded composition
layer's three engineered shortcuts with biologically faithful
mechanisms) is a substantial set of load-bearing claims. Per the
project's standing discipline -- scrutinise a result HARDER than a
failure, by a dedicated independent reviewer, before any
capability-status claim -- an independent reviewer (fresh agent, full
tool access, no controller context) ran the exploit-class checks. This
records the outcome.

## The review

The reviewer read the load-bearing files
(`research/runners/resonate_fire_fhrr.py`, the three probe/runner
files, the result JSONs, the findings docs) and RAN the checks rather
than reading only -- reproducing self-tests, recomputing from the real
activity cache, re-running the probes, and diffing the protected set.

## Checks and findings

1. **Resonate-and-fire neuron genuineness.** `rf_resonate` is a genuine
   time-stepped damped complex oscillator with an upward
   imaginary-part zero-crossing spike detector and a magnitude floor --
   not `np.angle` shortcut arithmetic. A hand-traced re-implementation
   of the oscillator dynamics matched the library output exactly. The
   self-test reproduced byte-identically (primitive errors
   0.0019/0.0025/0.0026/0.001; loads 2/3/5 accuracy 1.0000).

2. **Shortcut 3 -- the separated clean-up.** PASS, genuine. The
   familiarity threshold (0.2) is not tuned: it is the same
   phase-similarity quantity measured in the shortcut-1 self-test, and
   0.2 verifiably sits between the ungroundable maximum (0.112) and the
   hardest-load groundable minimum (0.303). The annealed-threshold
   attempt genuinely failed abstention -- not hidden. The attractor is
   a real complex recurrent network; the settle has no internal argmax.

3. **The closed-form fast-path.** `settle_annealed` with `fast=False`
   versus `fast=True`: 60/60 recognition outcomes matched on an
   independent run -- a larger sample than the 16-case check the arc
   claimed.

4. **Shortcut 2 -- the NEGATIVE.** Genuine. The attractor-denoised
   integrated accuracy (~0.25) is confirmed worse than the un-grounded
   baseline (0.33-0.42). Recomputed from the real activity cache: the
   consolidated-symbol mean pairwise similarity is 0.456/0.433/0.466
   (~0.45 as claimed); attractor recognition is exactly 1/16 chance at
   every seed. The pattern-separation probe reproduced byte-identically
   (separation 0.433 -> 0.170, composition 1.000, recognition 0.457).

5. **No automatic differentiation.** No torch / autograd / backward /
   optimizer in the module or any probe file.

6. **Protected set.** The `*_core` and `sim/` and abstention-moat
   files show empty diffs; `spiking_phasor_fhrr.py` is byte-unchanged
   since its creation. The no-confabulation moat test passes 7/7.

7. **Honesty of the findings docs.** No overclaiming. The shortcut-1
   PASS is honestly scoped as a subsystem-level result, not a
   capability claim. The cheap-probe over-optimism (the
   attractor-grounded-symbol probe said reachable while the real test
   was negative) is explicitly recorded.

One minor non-defect was noted: the resonate-and-fire design document
described the spike condition in Frady and Sommer's phrasing rather
than the implementation's equivalent upward zero-crossing form. The
implementation's own docstring is accurate. The design document has
since been corrected to state both forms.

## Verdict: CLEAR

The reviewer found no defect. The biologization arc's claims are
genuine.

## The biologization arc -- recorded synthesis

With the review CLEAR, the arc's outcome is recorded as a
capability_status pillar (status BOUNDARY -- the arc's deliverable is a
precise boundary characterization, not a new capability):

- **Shortcut 1 (the integrator neurons): biologized.** The
  function-first phase-sum / phase-subtraction integrator was replaced
  with the resonate-and-fire neuron, a recognized biological neuron
  model. Subsystem self-test PASS, adversarially reviewed CLEAR.
- **Shortcut 3 (the clean-up): biologized.** The argmax-over-a-stored-
  list was replaced with an attractor settle for identification plus a
  separate familiarity gate for abstention. The structural finding --
  a pure attractor settle confabulates, so abstention cannot be a
  basin-of-attraction property -- is sound. Subsystem self-test PASS,
  adversarially reviewed CLEAR.
- **Shortcut 2 (the symbols): recognition-bounded.** The symbol cannot
  be fully grounded in the substrate's activity by a single transform.
  The oracle lookup's load-bearing function is supplying
  near-orthogonality; pattern separation (the validated dentate-gyrus
  mechanism) supplies it from the substrate's own representations
  (overlap 0.43 reduced to 0.17, composition 1.000) -- but grounding
  then reduces to concept recognition, which is the substrate's own
  ~0.74-0.88 capability. A grounded-symbol pipeline is
  recognition-bounded, exactly as the validated identity-level
  integration already is.

## Honest standing

The phase-coded composition layer can be biologized in its neurons and
its clean-up. Its symbols are groundable via pattern separation, but
the grounded pipeline -- like the oracle-symbol pipeline -- is
recognition-bounded. The whole compositional line converges on one
bound: the substrate's concept-recognition accuracy. The validated
compositional capability stands (the identity-level integration,
multi-seed 0.96-0.99); the biologization arc makes two-thirds of the
composition layer biologically faithful and characterizes the third
precisely. The next arc is the substrate's recognition itself.

## Files / evidence

- Reviewed: `research/runners/resonate_fire_fhrr.py`, the three
  probe/runner files under `research/findings/raw/`, the result JSONs.
- Capability pillar: `webapp/capability_status.json`.
- The arc's findings docs (2026-05-22): resonate-and-fire shortcut 1,
  attractor-cleanup shortcut 3, shortcut 2 NEGATIVE, pattern-separation
  grounding.
