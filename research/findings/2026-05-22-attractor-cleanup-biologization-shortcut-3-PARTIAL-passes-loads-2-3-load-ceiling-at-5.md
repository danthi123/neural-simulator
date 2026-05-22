# Biologization step 3 = PARTIAL: the attractor clean-up biologizes the mechanism and passes at loads 2-3, but a fixed-threshold attractor has a load ceiling and fails at load 5

## Status

The third step of the biologization arc: replace the clean-up's
argmax-over-a-stored-list with an attractor network whose fixed points
are the vocabulary. Result: PARTIAL. The attractor clean-up clears the
frozen 0.80 compositional bar at loads 2 and 3 with the abstention moat
intact as a basin-of-attraction property, but at load 5 it fails. The
pre-registered verdict (all loads {2,3,5}) is therefore not met. The
honest finding is a precise one: a fixed-threshold attractor clean-up
has a compositional-load ceiling.

## What was built

`research/runners/resonate_fire_fhrr.py` was extended with a
`ResonateFireTPAM` clean-up -- the Threshold Phasor Associative Memory
of Frady and Sommer (PNAS 2019). The vocabulary is stored in a
recurrent weight matrix (the outer product of the stored phasor
patterns, normalised by the dimension). A noisy recovered phasor is
cleaned by settling the recurrent dynamics: iterate the recurrent
synaptic integration and the resonate-and-fire threshold transfer
(above-threshold neurons re-emit a spike at the phase of their drive;
below-threshold neurons stay silent). Abstention is a basin-of-
attraction property: an ungroundable input lies in no attractor's
basin, the recurrent drive never exceeds threshold, and the state
collapses to silence.

Reuse-by-import only; no protected, frozen, or moat module modified; no
automatic differentiation -- the settling is recurrent attractor
dynamics. The threshold was set in advance from a drive-magnitude
analysis and not tuned afterwards.

## Result (pre-registered; frozen 0.80 bar; the project's compositional task)

```
            compositional accuracy    abstention (settle active fraction)
L=2         1.0000                    groundable 0.908 > ungroundable 0.000
L=3         0.9867                    groundable 0.711 > ungroundable 0.000
L=5         0.1980                    groundable 0.000 ; ungroundable 0.000

VERDICT -> FAIL (load 5 below the bar; the pre-registered verdict
                 requires all loads {2,3,5})
```

At loads 2 and 3 the attractor clean-up passes decisively: it clears
the 0.80 bar, and the abstention signal -- the fraction of neurons
still active after the settle -- cleanly separates groundable queries
(which settle onto a full stored attractor) from ungroundable queries
(which collapse to silence). The no-confabulation moat survives as a
basin-of-attraction property at these loads.

At load 5 the clean-up collapses on groundable queries too: the
groundable settle active fraction drops to zero, and the accuracy
falls to 0.198 -- near chance for an 8-way readout over a collapsed
(silent) state.

## Why -- the mechanism, and the smell test of the failure

The failure is genuine and its mechanism is precise. At load 5 the
composite bundles five facts; unbinding recovers the wanted filler
plus four crosstalk terms. Per dimension the wanted signal has unit
magnitude and the four crosstalk terms sum to roughly twice that, so
the recovered phase is dominated by crosstalk. Averaged over all 512
dimensions the wanted signal still correlates and is recoverable --
but the per-neuron recurrent drive that this averaged correlation
produces is only about 0.4, which sits just below the abstention
threshold of 0.5. So the attractor settle rejects the noisy load-5
groundable queries as if they were ungroundable, and the state
collapses.

The smell test confirms this is a real load ceiling, not a bug:

- The attractor network is mechanically correct: a clean stored
  pattern presented to it is identified correctly and settles with
  almost every neuron active (active fraction 0.998); pure noise
  collapses to silence (active fraction 0.000).
- Loads 2 and 3 pass cleanly with wide abstention separation.
- The load-5 collapse is consistent with the signal-to-crosstalk
  analysis above.
- The decisive control: the same task with the argmax clean-up (no
  threshold) reaches accuracy 1.000 at load 5. So the wanted signal IS
  present in the recovered phasor at load 5 -- the 512-dimensional
  averaged correlation still picks the right filler. What fails is not
  the signal; it is that the fixed abstention threshold, set high
  enough to reject ungroundable queries, also rejects the noisy
  high-load groundable queries.

This is the honest, biology-translatable finding: a fixed-threshold
attractor clean-up faces a genuine tension. The threshold must be high
enough to reject ungroundable inputs (the no-confabulation moat) and at
the same time low enough to admit noisy groundable inputs into the
correct basin. At low compositional load the recovered signal is strong
and one threshold satisfies both. Beyond some load the recovered
signal-to-crosstalk ratio falls far enough that no single fixed
threshold separates groundable from ungroundable -- the basin and the
moat are in conflict.

## What this is and is not

This is not a failure of the attractor clean-up as a biological
mechanism. At loads 2-3 it works, and it converts the no-confabulation
moat from a hand-set similarity cutoff into a basin-of-attraction
property -- which is the goal of the biologization. What it is is a
precise quantitative finding about the limits of a fixed-threshold
attractor: a load ceiling, and the mechanism of that ceiling.

The argmax clean-up has no load ceiling at these loads -- but it also
has no genuine abstention; it always returns a nearest item. The
attractor clean-up has genuine abstention but a load ceiling. The two
honest results together locate the real problem: genuine abstention
and high-load tolerance are in tension under a fixed threshold.

## The mitigation -- next pre-registered step

The mechanism points directly at the fix, and it is biologically
motivated. A fixed threshold is the problem; biology does not use one
-- neural thresholds and gains are dynamic (gain modulation, threshold
adaptation). An attractor settle with an ANNEALED threshold -- low at
first, so a noisy high-load input is admitted into a basin and the
recurrent dynamics get a chance to denoise it, then raised as the state
sharpens toward an attractor -- should admit the load-5 groundable
queries while still rejecting ungroundable ones (an ungroundable input
does not sharpen toward any attractor at any threshold). This is a
structural change to the settle dynamics with its own pre-registered
verdict, not a re-tuning of the fixed threshold.

This also connects to the activity-level negative and to shortcut 2:
both that negative and this load ceiling are about high-noise inputs
needing denoising that a single thresholded pass does not provide. An
annealed (iteratively denoising) attractor settle is the shared answer.

## Honest scope

Subsystem-level result. Biologization step 1 (resonate-and-fire
neurons) passed; step 3 (attractor clean-up) is partial -- the
mechanism is biologized and works at loads 2-3, with a characterised
load ceiling at 5. Not a capability claim. The next pre-registered step
is the annealed-threshold attractor settle.

## Files / evidence

- Module: `research/runners/resonate_fire_fhrr.py` (`ResonateFireTPAM`)
- Result: `research/findings/raw/resonate_fire_tpam_selftest.json`
- Design: `docs/plans/2026-05-22-attractor-cleanup-biologization-design.md`

## References

- Frady and Sommer, "Robust computation with rhythmic spike patterns",
  PNAS 116(36):18050-18059, 2019 -- the Threshold Phasor Associative
  Memory.
