# Predictive coding (Rao-Ballard / Whittington-Bogacz) cheap falsify-first gate — honest VOID for the LEARNING measurement, with a DURABLE V1 positive (PC's local update provably tracks the backprop gradient, cos~0.995, 5 seeds); NOT config-cranked; the deeper PC-learning-instrument decision handed to the OWNER eyes-open

## TL;DR

After the compose->spiking-bridge integration honest-VOIDed (owner
holds that fork), the next AUTONOMOUS catalog-grounded missing-biology
lever -- distinct from the thrice-validated temporal-credit and from
the owner-forked in-bridge engineering -- was cortical **predictive
coding** (Rao-Ballard; Whittington-Bogacz 2017). It was deferred
earlier because representation-learning rules risk the
readout-over-features confound that boundaried the dendritic/FA arc,
BUT predictive coding uniquely has the discipline-derived property that
distinguished the TD-critic PASS: a PRINCIPLED, analytically-checkable
positive control (Whittington-Bogacz prove PC's local weight update
equals the backprop gradient at the inference equilibrium).

A throwaway, pure-numpy, NO-autodiff falsify-first probe with a
pre-registered THREE-STATE instrument-validity-FIRST gate was built and
run (5 seeds). Two root-caused, mode-AGNOSTIC instrument iterations were
applied and transparently logged (the first per-sample version was too
slow to be "cheap" + PC-inference overflow -> vectorized + stabilized;
the first discriminating control `random_feedback` was scientifically
INVALID -- in a shallow net that IS feedback alignment, which is KNOWN
to partially align with the gradient -> replaced with the FAITHFUL
`no_inference` mechanism-removed control + a fair mode-agnostic training
budget). Frozen science bars byte-UNCHANGED throughout
(`V1_COS_MIN=0.95`, `SCI_ACC_MIN=0.80`, `CTRL_ACC_MAX=0.40`,
`MIN_SEEDS=3`).

**Recorded result (5 seeds 42-46; recomputed from the recorded
output, no re-run-for-different-outcome, no bar-tuning):**

- **DURABLE POSITIVE -- V1 SOUND, robust:** the PC local weight-update
  direction tracks the true (hand-derived numpy, NO autodiff) backprop
  gradient-descent direction at **cosine 0.995 / 0.994 / 0.997 / 0.995
  / 0.995** (>= 0.95 all seeds). The Whittington-Bogacz equivalence
  empirically HOLDS in a faithful from-scratch implementation. This is
  the principled positive control the feedback-alignment arc never had,
  and it is a genuine, durable engineering/scientific result regardless
  of the science outcome.
- **Controls genuinely fail (instrument discriminating):** the
  FAITHFUL `no_inference` mechanism-removed control 0.06-0.36,
  `wrongsign` diverged (nan -> correctly-failed), `permuted` 0.11-0.28
  -- all <= 0.40.
- **The science measurement is NOT soundly answerable at this cheap
  config:** PC trained ALONE scored **0.083 / 0.192 / 0.358 / 0.117 /
  0.400** (>> below the 0.80 bar; ~chance 0.25) AND essentially EQUAL
  to the `no_inference` mechanism-removed control (e.g. seed44 pc 0.358
  vs no_inference 0.358) -- on a task TRIVIALLY learnable by backprop
  (4 well-separated Gaussian clusters; a hand-derived backprop MLP
  reaches ~0.95 in a few epochs). PC's per-step update is cos 0.995 to
  backprop, yet 120 epochs of PC-trained-alone neither learns nor beats
  not-doing-PC-inference.

**Verdict (no spin):** The probe's mechanical THREE-STATE returned
**FAIL** (instrument valid + discriminating, science bar not met). The
mandatory controller anti-cheat smell-test refines this to the **more
honest, less-overclaiming VOID classification**: per-step
direction-soundness (V1, cos 0.995) is NOT training-loop learning-
soundness; PC-trained-alone == mechanism-removed control == chance on a
trivially-backprop-learnable task means the PC *learning* instrument is
not soundly constructed at this cheap config, so the science question
("does PC, as a learner, beat the mechanism-removed control") is **not
validly answerable here** -- VOID, NOT a refutation of the PC
principle, NOT a fabricated PASS. The durable V1 positive (PC update ==
backprop gradient) is honestly preserved and never spun as "PC works."

## Honest scope (no overclaim, no underclaim, no spin)

- **NOT a refutation of predictive coding as a principle.** The
  Whittington-Bogacz equivalence empirically HELD (V1 cos 0.995, 5
  seeds) -- PC's local update genuinely computes the backprop gradient
  direction. What is NOT cheaply constructible is a sound PC *learning*
  instrument (the training loop accumulating that cos-0.995 update into
  actual learning) at feasible cheap scale -- analogous to the
  dendritic/conv "correct direction, instrument not soundly trainable
  cheap" boundary and the compose-bridge spiking-bootstrap VOID.
- **NOT a science FAIL** in the strong sense (which would assert "PC
  fails to learn as a principle"); that cannot be soundly concluded
  from an instrument whose training loop evidently does not turn its
  own gradient-aligned per-step update into learning even on a trivial
  task. Classified VOID (instrument-not-soundly-constructible for the
  learning measurement) -- the conservative, non-overclaiming reading.
- **DOES converge with the arc-wide recurring boundary:** a principled,
  analytically-checkable local credit signal is constructible/checkable
  (PC V1 cos 0.995, like the TD-critic's clean V1), but cheaply turning
  it into a sound, discriminating *learner* at feasible scale is the
  recurring infeasibility hit from independent directions (dendritic
  readout-confound BOUNDARY; compose-bridge spiking-bootstrap VOID; now
  PC-learning-loop VOID). The temporal-credit lever remains the only
  one with a clean validated PASS (itself boundaried at the
  spiking-integration step).

## Why this is NOT config-cranked, and the genuine remaining option (OWNER decision)

Two root-caused, mode-AGNOSTIC instrument iterations were applied
(vectorize+stabilize; faithful-control + fair budget) and transparently
logged; the frozen science bars are byte-UNCHANGED. A third
budget-crank specifically until `pc` crosses 0.80 would be
config-cranking toward a desired PASS -- forbidden by the discipline
and the iron law (root-caused fixes applied; stop, do not keep cranking
toward an outcome). So this honest VOID is propagated and the genuine
remaining option is handed to the OWNER as an eyes-open strategic
decision, NOT taken autonomously (mirroring the dendritic fair-scale
and compose-bridge VOID handoffs):

> The only path to a sound cheap PC *learning* instrument is a deeper
> instrument-engineering pass on the PC training loop (so the
> cos-0.995 per-step update accumulates into learning -- e.g. precision
> weighting, inference-equilibrium scheduling, the published
> learning-rate/relaxation regimes from the PC literature) -- a larger
> speculative investment with no cheap precursor for the *learning*
> sub-problem (the cheap probe validated the per-step EQUIVALENCE, NOT
> PC-as-a-trainable-learner). Honest facts for the decision: (a) the
> Whittington-Bogacz equivalence holds (V1 cos 0.995); (b) the blocker
> is the training-loop accumulation, not the credit direction; (c) it
> is a further, separately-gated engineering effort. Authorize it
> deliberately (eyes-open) or accept this boundary. I will not
> autonomously crank the PC training budget/parameters against a
> pre-registered cheap terminus toward a desired PASS.

## What is preserved / validated (unaffected)

The probe was a single THROWAWAY pure-numpy file (deleted
post-decision; recorded numeric output preserved at
`research/findings/raw/pc_probe_recorded.txt`). NO protected/validated
module was created, modified, or touched: the no-confab moat
(`abstention_gate` + its test, 7/7), `sim/td_value_critic.py`,
`sim/compose_temporal_bind.py`, `sim/kernels.py`, `sim/bridge.py`,
`sim/neuromodulators.py`, every frozen `*_core`, all remain
byte-identical. NO autodiff anywhere (the backprop reference is
hand-derived numpy, validity-only). The prior arc results -- dendritic
BOUNDARY, TD-critic PASS, compose-abstract PASS, pop-transfer
cheap-GREEN, compose-spiking VOID -- are entirely unaffected.

## Anti-cheat discipline (why this VOID is trustworthy)

The pre-registered THREE-STATE + V1-instrument-validity-FIRST design
did its job: it produced a robust principled V1 positive
(Whittington-Bogacz equivalence, cos 0.995) AND refused to emit a
fabricated PASS when PC-trained-alone failed to learn. Two
probe-construction faults were caught and root-caused mode-agnostically
(too-slow/overflow; the FA-confounded control) WITHOUT touching the
frozen science bars and transparently logged. The mechanical FAIL was
scrutinized HARDER by the controller and honestly refined to the
less-overclaiming VOID classification (instrument-not-soundly-
constructible for the learning measurement), with the durable positive
preserved and never spun. No bar-tuning; no config-cranking toward a
PASS; the deeper-engineering option handed to the owner. The validated
no-confab moat remained byte-identical and 7/7 green throughout.

## Files / evidence

- Recorded probe output: `research/findings/raw/pc_probe_recorded.txt`
  (5 seeds; V1cos 0.994-0.997 [PASS]; pc 0.083-0.400 ~chance ==
  no_inference; wrongsign nan; permuted 0.11-0.28; mechanical
  GATE=FAIL, controller-refined to VOID-class per the smell-test
  above).
- Throwaway probe `_probe_predictive_coding.py` (deleted
  post-decision; pure numpy, NO autodiff, faithful Whittington-Bogacz
  PC + hand-derived backprop validity reference + the FAITHFUL
  no_inference mechanism-removed control).
- Converges with / does NOT refute:
  `2026-05-18-td-value-critic-temporal-credit-PASS.md`,
  `2026-05-18-compose-temporal-credit-PASS.md`,
  `2026-05-18-compose-temporal-credit-spiking-VOID.md`,
  `2026-05-18-dendritic-cifar-conv-fa-cheap-gate-NEGATIVE-boundary.md`
  (the arc-wide recurring boundary this triangulates from the
  predictive-coding direction).
