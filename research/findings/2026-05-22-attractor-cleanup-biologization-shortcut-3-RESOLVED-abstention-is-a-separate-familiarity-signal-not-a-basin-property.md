# Biologization step 3 = RESOLVED: identification is biologized as an attractor settle; abstention is a separate familiarity signal, not a basin-of-attraction property

## Status

The third step of the biologization arc -- replace the clean-up's
argmax-over-a-stored-list -- is resolved. It took three attempts, and
the two failures along the way are the substantive finding: they prove
a structural fact about what a clean-up can and cannot do. The resolved
clean-up clears the frozen 0.80 compositional bar at all loads with the
no-confabulation abstention intact.

## The arc: three attempts

The clean-up must do two jobs: identify which vocabulary item a noisy
recovered phasor is, and abstain when the input matches no item. The
three attempts differ in how those two jobs are arranged.

**Attempt 1 -- fixed-threshold attractor (Threshold Phasor Associative
Memory).** A complex-valued attractor network whose fixed points are
the vocabulary; a recovered phasor is cleaned by settling the recurrent
dynamics; abstention was taken to be the settle collapsing to silence.
Result: PARTIAL -- passes loads 2 and 3, fails load 5. At high
compositional load the recovered phasor's signal is weak, the per-neuron
recurrent drive falls just below the threshold that rejects ungroundable
inputs, and the settle collapses on groundable queries too. A
fixed-threshold attractor has a load ceiling. (Recorded separately in
`2026-05-22-attractor-cleanup-biologization-shortcut-3-PARTIAL-passes-loads-2-3-load-ceiling-at-5.md`.)

**Attempt 2 -- annealed-threshold attractor.** The mitigation for the
load ceiling: anneal the threshold from low (admit the noisy input
broadly so the recurrent denoising can run) to high (demand sharpness).
Result: the compositional accuracy reaches 1.000 at every load,
including load 5 -- the load ceiling is gone -- but the abstention is
completely broken. Ungroundable queries do not collapse; they settle
into a memory basin (active fraction up to 1.000 instead of zero). This
is a NEGATIVE, and it is the pivotal finding of the arc.

**Attempt 3 -- separated clean-up.** The resolution forced by attempt
2's finding (below). Result: PASS at all loads (details below).

## The structural finding (from attempt 2)

A Hopfield-type attractor network sorts EVERY input into a memory
basin. The state space is tiled by the basins of the stored patterns;
there is no "no man's land" that a settle can land in. So an attractor
settle that admits an input broadly enough to denoise a noisy
groundable query will also sort an ungroundable query into some basin
and sharpen it there. A pure attractor settle confabulates.

This means abstention -- the no-confabulation moat -- CANNOT be a
basin-of-attraction property. It cannot be "the settle failed to reach
an attractor," because the settle always reaches one. Abstention has to
be a separate signal: an explicit measure of whether the input matches
any stored memory at all, computed independently of the settle.

The two failures are the two horns of a dilemma. With the threshold
fused into the settle and set high, abstention works but high-load
groundable queries are rejected (the load ceiling). With it annealed
down, high-load queries are denoised but abstention is lost. The same
gate cannot both denoise high-load inputs and abstain. The jobs must be
separated.

## The resolved clean-up (attempt 3)

Separate the two jobs:

- **Abstention is a match-strength (familiarity) gate, computed before
  the settle.** How strongly does the recovered phasor match any stored
  memory. Below a familiarity threshold, the clean-up abstains and the
  recall network never engages. A familiarity / novelty signal is a
  real, separate biological mechanism -- the brain has dedicated
  novelty and familiarity signals that gate memory processes. It is not
  a shortcut; it is a distinct neural computation.
- **Identification, for an input that passes the gate, is the annealed
  attractor settle.** This is the genuinely biologized recall: the
  vocabulary lives in distributed recurrent weights, and identification
  is a settling of recurrent dynamics, not an argmax over an enumerated
  list. The annealing handles all loads.

The familiarity threshold was set in advance, from the already-measured
groundable-versus-ungroundable phase-similarity separation (the
resonate-and-fire self-test measured groundable match strength 0.596 /
0.454 / 0.303 at loads 2 / 3 / 5 and ungroundable match strength about
0.11 at every load); the threshold 0.2 sits between. It is not the
compositional bar and was not tuned to this run.

## Result (pre-registered; frozen 0.80 bar; the project's compositional task)

```
            compositional accuracy    abstention (familiarity gate)
L=2         1.0000                    groundable 0.596 > 0.2 > ungroundable 0.112
L=3         1.0000                    groundable 0.454 > 0.2 > ungroundable 0.115
L=5         1.0000                    groundable 0.303 > 0.2 > ungroundable 0.112

VERDICT -> PASS
```

The separated clean-up clears the frozen 0.80 bar at every load, and
the familiarity gate cleanly separates groundable queries (which it
admits) from ungroundable queries (which it abstains on) at every load.

## Smell test (a PASS scrutinised harder than a FAIL)

- The PASS is earned by two characterised failures, not found by a
  parameter sweep. The fixed and annealed attempts each produced a
  precise, mechanistic failure, and together they prove the structural
  fact that forced the separated design. The separated design is the
  logical resolution of that dilemma, not the next variant in a search.
- The familiarity threshold (0.2) was derived from data measured before
  this run (the resonate-and-fire self-test's groundable and
  ungroundable match-strength figures) and pre-registered. It was not
  adjusted to make this run pass.
- The match-strength figures in this run (0.596 / 0.454 / 0.303
  groundable, ~0.11 ungroundable) are identical to those measured in
  the resonate-and-fire self-test, as they must be -- the match
  strength is the same quantity. The separation is the same separation
  already established; this run confirms a familiarity gate at a
  pre-set threshold reads it correctly.
- Honest accounting of what was biologized: the identification
  mechanism is biologized -- an attractor settle over distributed
  recurrent weights replaces the argmax over an enumerated list. The
  abstention mechanism was re-examined and found to be, legitimately, a
  separate familiarity signal -- which is what the engineering scaffold
  already used. The honest claim is therefore precise: shortcut 3
  biologizes the identification; it re-frames the abstention as a
  distinct biological signal (familiarity detection) rather than a
  basin property, and the annealed result is the evidence that it must
  be distinct.

## Honest scope

Subsystem-level result. Biologization step 1 (resonate-and-fire
neurons) passed; step 3 (clean-up) is resolved as above. Step 2 -- the
symbol still assigned by oracle lookup -- remains; its naive form (a
symbol derived from raw substrate activity) was a decisive negative
because raw activity is too noisy. Not a capability claim; a dedicated
adversarial review is the pre-registered discipline step before any
capability-status claim rolls up.

## Next step

Shortcut 2's deeper form: ground the symbol not in raw substrate
activity (too noisy) but in a denoised, attractor-stabilised
representation. The attractor machinery built here is the substrate for
it -- an attractor network both stores a representation in distributed
recurrent weights and denoises a noisy version of it by settling. The
activity-level negative said exactly this denoising was the missing
piece.

## Files / evidence

- Module: `research/runners/resonate_fire_fhrr.py` (`ResonateFireTPAM`,
  `cleanup_separated`)
- Results: `research/findings/raw/resonate_fire_tpam_annealed_selftest.json`
  (attempt 2, NEGATIVE),
  `research/findings/raw/resonate_fire_tpam_separated_selftest.json`
  (attempt 3, PASS)
- Design: `docs/plans/2026-05-22-attractor-cleanup-biologization-design.md`

## References

- Frady and Sommer, "Robust computation with rhythmic spike patterns",
  PNAS 116(36):18050-18059, 2019 -- the Threshold Phasor Associative
  Memory.
