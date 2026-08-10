---
type: finding
status: contributing
date: 2026-08-10
mechanism: cortical-afferent-winner-selection
lane: D-pragmatics / WTA-readout
---

# The firing-rate HOMEOSTAT does NOT rescue the value-critic SNR wall: at the suprathreshold operating point the neurons are pinned at the spike clip, so a spike-threshold homeostat has no graded lever to equalize rates (mechanistic NEGATIVE with teeth)

**Hypothesis tested (untried lever):** the pragmatics Leg-2 "learn to speak" convergence-NEGATIVE is an SNR wall —
the DA-learned value differential (~0.01-0.02) sits BELOW the per-neuron heterogeneity noise (~2-6%). A region-scoped
per-neuron firing-rate HOMEOSTAT (Diehl-Cook adaptive threshold, settling/exposure phase) on the competing
critic/actor populations should EQUALIZE intrinsic excitability → LOWER the heterogeneity noise floor → let the tiny
learned differential separate. This is the same homeostat the WTA gate (`d42fd05d`) named, aimed here at the case
that genuinely has signal-below-noise. **Result: NEGATIVE, mechanistically.**

## The mechanistic negative (with teeth)

<!--derived-->

Runner `research/runners/_pragmatic_readback_leg2_v2_homeostat_derisk.py` (imports the committed v2 runner's helpers;
adds a homeostat-scoped bridge builder + exposure phase + a direct noise measurement; NO `sim/` edit).

- **Faithfulness anchors PASS:** substrate byte-identical between homeo / no-homeo builds; the no-homeo path
  reproduces the committed convergence NEGATIVE exactly (seed 42: actor 0.667 / critic 1.000 on the diagnostic).
- **The homeostat CANNOT reduce the noise:** the assembly-rate CV never drops below the no-homeo baseline (0.0160);
  the HARDER it engages the WORSE it gets (up to 0.0297, with 97% of neurons pinned at the +35 mV spike clip). The
  activity EMA floors at ~0.066 — 3.3× the 0.02 homeostat target — so the controller can never reach target under
  the readout drive.
- **Root cause (measured directly):** at this SUPRATHRESHOLD operating point the rate-vs-threshold curve has a FLOOR
  of ~0.040 (≫ the 0.02 target) and is shallow / non-monotonic near it. A spike-detection-threshold homeostat
  therefore has NO graded, target-reaching lever to equalize per-neuron rates — raising a pinned neuron's threshold
  does not bring its rate down toward target. (A confirmatory with/without convergence sweep was left running; the
  mechanism above is the load-bearing conclusion — if the CV cannot drop, convergence cannot improve.)

## What this closes + redirects

<!--derived-->

The firing-rate homeostat is now a NEGATIVE for BOTH WTA/critic regimes, a clean complete picture:
- where the afferent signal is ABOVE the noise (separable-assembly WTA, `2026-08-10-neural-WTA-separable-*`) — the
  homeostat is UNNEEDED (the winner already follows the afferent);
- where the signal is BELOW the noise (this value-critic case) — the homeostat CANNOT ENGAGE (neurons suprathreshold-
  pinned; a threshold controller has no graded lever at the clip).
⇒ **noise-reduction via an intrinsic threshold homeostat is not the value-critic lever.** The residual redirects to
the DUAL — SIGNAL AMPLIFICATION (a recurrent value attractor / accumulate-to-bound that amplifies the tiny learned
differential; being de-risked concurrently in `_pragmatic_readback_leg2_v2_ampattractor_derisk.py`), and/or moving
the decision to a SUBTHRESHOLD operating point (lower the tonic/readout drive so the neurons are NOT clip-pinned and
a homeostat COULD get a graded lever) as a separate, untested lever.

Artifact (reproduced this session, seed 42): `research/findings/raw/_pragmatic_success/homeostat_mech_s42.json` —
per-condition assembly-rate CV: no_homeo **0.016** (baseline) vs homeostat {a05_long_c30 0.042, a05_long_c35 0.0155,
a2_long_c35 0.017, a2_xlong_c35 0.0297, a2_xlong_c35_cont 0.0297} — NONE meaningfully below baseline; the hardest-
engaging conditions are WORSE (0.0297). Reproducer =
`research/runners/_pragmatic_readback_leg2_v2_homeostat_derisk.py`. NO `sim/` edit. SIM_BACKEND=numpy.
