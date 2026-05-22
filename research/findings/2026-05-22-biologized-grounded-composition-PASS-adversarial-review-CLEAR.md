# Dedicated adversarial review of the biologized-grounded-composition PASS = CLEAR; the FHRR-biologization arc closes positively and the capability pillar is corrected

## Status

The fully-biologized grounded compositional pipeline cleared the frozen
0.80 bar at multi-seed 0.98, and that PASS overturned two NEGATIVEs
propagated earlier the same day. A load-bearing PASS that overturns
prior results must be scrutinised by a dedicated independent reviewer
before any capability claim. This records the review outcome.

## The review

An independent reviewer (fresh agent, full tool access, no controller
context) RAN the exploit-class checks -- reproduced the pipeline,
recomputed from the real activity cache, independently rebuilt the
attractor self-identification test, traced the data flow for answer
leakage, and diffed the protected set.

## Checks and findings

1. **Reproduce.** Re-ran the pipeline: integrated multi-seed
   0.988 / 0.981 / 0.979, composition-only 0.999 / 0.994 / 0.990 --
   matches the recorded result within run-to-run query-RNG noise. PASS
   confirmed.

2. **Is mean-centering a cheat?** No -- legitimate. The grounded symbol
   is a deterministic random projection of the substrate's own cached
   activity, with the across-concept common-mode subtracted; it is not
   a random or oracle vector. The common-mode is computed only from
   activity -- no task labels feed it (the reviewer traced the data
   flow: ground-truth labels appear only in the scoring comparison, not
   in the pipeline). Independent recompute: raw concept-symbol overlap
   +0.45, mean-centered -0.05 -- matches. Subtractive normalisation /
   common-mode rejection is a well-characterised cortical computation
   (pooled inhibition); calling it biological is honest.

3. **Recognition.** Raw-activity-space temporal averaging (K=8
   observations, per-pool argmax) -- no label leakage. The cache is
   real captured substrate activity (16 observations of 3200 neurons,
   genuine trial-to-trial noise).

4. **Composition-only.** The reviewer independently built the attractor
   over the mean-centered grounded symbols and settled each clean
   symbol: 15/16, 16/16, 16/16 self-identify, versus 1/16 for the raw
   symbols. The 0.99 composition-only is genuine discrimination.

5. **No automatic differentiation; protected set.** No torch / autograd
   / backward in the runner or helpers. The protected-set diff is empty
   (the validated `spiking_phasor_fhrr.py` is byte-unchanged since its
   creation commit). The no-confabulation moat test passes 7/7.

6. **Is the correction honest?** Yes. Both superseded NEGATIVE
   documents carry a clear correction notice and are kept as the honest
   trail; the notices state plainly that the measurements were real and
   only the conclusions were premature, because common-mode removal was
   an untested transform. The error is named precisely; nothing is
   buried.

## Verdict: CLEAR

The reviewer found no defect. The PASS is genuine, mean-centering is a
legitimate biological operation derived solely from substrate activity,
and the correction of the two prior NEGATIVEs is honest and traceable.

## The FHRR-biologization arc -- final synthesis

With the review CLEAR, the arc's outcome is recorded as the corrected
capability_status pillar (status VALIDATED). The phase-coded
composition layer of the project's validated compositional capability
had three engineered shortcuts; the arc biologized all three:

- **Shortcut 1 -- the integrator neurons.** Replaced with
  resonate-and-fire neurons (Izhikevich 2001; Frady & Sommer 2019), a
  genuine time-stepped damped complex oscillator. PASS.
- **Shortcut 3 -- the clean-up.** Replaced with an attractor settle for
  identification plus a separate familiarity gate for abstention; the
  structural finding that a pure attractor settle confabulates, so
  abstention is a separate signal. RESOLVED.
- **Shortcut 2 -- the symbols.** Grounded in the substrate's own
  activity: the concept's consolidated activity with the across-concept
  common-mode removed (subtractive normalisation). The 0.45 concept
  overlap is almost all common-mode; removing it exposes the
  near-orthogonal concept-specific structure. PASS.

The fully-biologized grounded compositional pipeline -- longer-
integration recognition, common-mode-removed grounded symbols,
resonate-and-fire FHRR composition, attractor clean-up, no oracle
symbol table -- clears the frozen 0.80 compositional bar at integrated
multi-seed 0.98. Every stage is a biological mechanism.

## Honest standing

This is a genuine, multi-seed, adversarially-reviewed result: a
compositional retrieval capability that is biology-grounded end to end.
Honest scope: 3 seeds; the project's small-load compositional task
(loads {2,3,5}); computed from the real substrate activity captured in
the activity cache; the recognition front-end is temporal averaging,
recognition-bounded at about 0.93; it is cue-to-attribute compositional
retrieval, explicitly NOT fluent open-ended language. The biology-
inspired-engineering caveat on the phasor representation still holds --
theta-gamma phase coding is real biology; the FHRR operators are
function-first devices now realized on the resonate-and-fire neuron
model. The validated identity-level integration (oracle symbols)
stands; this arc shows the same capability is reachable with the
symbols grounded in the substrate.

## Files / evidence

- Reviewed: `research/findings/raw/biologized_grounded_composition.py`,
  `..._meancenter.json`, the imported helpers, `resonate_fire_fhrr.py`,
  the activity cache.
- Capability pillar: `webapp/capability_status.json`.
- The PASS finding:
  `2026-05-22-biologized-grounded-composition-PASS-mean-centering-closes-the-arc-and-corrects-the-premature-negatives.md`.
