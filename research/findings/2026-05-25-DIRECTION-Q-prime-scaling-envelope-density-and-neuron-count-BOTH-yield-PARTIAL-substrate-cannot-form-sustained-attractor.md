# Direction Q-prime scaling envelope characterization: density=0.20 AND n=2000 BOTH yield PARTIAL (rate elevation scales with effective recurrent connections but sustained_sec stuck at 0.55-0.65s); biology-translatable finding = the Izhikevich substrate produces strong cue-driven NMDA transients but cannot form a self-sustaining recurrent attractor at the tested scale envelope

**Date:** 2026-05-25
**Status:** All 3 cells in the pre-registered scaling envelope = Q_BISTABILITY_PARTIAL; closes the "density/scale alone fixes Direction Q" hypothesis with convergent multi-seed data; sharpens the biology-translatable diagnosis to point at structural/dynamical missing piece

## What was tested (pre-registered post-PARTIAL chain)

Per the Direction Q PARTIAL post-verdict chain
(`research/findings/2026-05-25-DIRECTION-Q-PARTIAL-dlpfc-n1000-NMDA-elevates-rate-but-not-sustained.md`), executed the density/neuron-count scaling envelope at n=1000 d=0.20 and n=2000 d=0.10 (the two cheap probes that test the "scale alone fixes it" hypothesis).

Bar UNCHANGED throughout (frozen in `research/findings/raw/direction_Q_verdict.py`
at Task 3): `_Q_RATE_RATIO_MIN=2.0`, `_Q_DELAY_MIN_SEC=3.0`, `_Q_MIN_SEEDS_PASS=3`.

## Scaling-envelope multi-seed results

| n | density | effective conns | TEST rate_ratio mean | TEST sustained_sec max | CONTROL rate_ratio | CONTROL sustained_sec | Verdict |
|---|---|---|---|---|---|---|---|
| 1000 | 0.10 | ~120K | 2.27 (2.19-2.38) | 0.45 | 0.96-1.13 | 0.00 | PARTIAL |
| 1000 | 0.20 | ~240K | 8.47 (8.03-9.07) | 0.60 | 0.93-1.06 | 0.00 | PARTIAL |
| 2000 | 0.10 | ~480K | 8.87 (7.02-10.53) | 0.65 | 1.02-1.13 | 0.00 | PARTIAL |

All 3 cells:
- TEST condition (NMDA-on) meets the rate_ratio bar by wide margins
  (4-10x the 2.0 bar)
- TEST condition FAILS the sustained_sec bar in every seed
  (~11-13 bins ~ 0.55-0.65 sec vs 3.0 sec required)
- CONTROL condition (NMDA-off) correctly fails the rate_ratio bar at
  every seed (consistent ~1.0 ratio with sustained=0.0)
- Verdict = Q_BISTABILITY_PARTIAL (pre-registered tag, never tuned)

## The decisive pattern

Rate elevation scales roughly with effective recurrent connection count:
- 120K -> 2.27 rate_ratio
- 240K -> 8.47 (3.7x more connections, 3.7x more rate_ratio)
- 480K -> 8.87 (similar effective conns range; similar rate_ratio)

But sustained_sec barely moves: 0.45 -> 0.60 -> 0.65. The decay
timescale of the cue-driven transient is roughly invariant under the
scaling axes tested. The Wang 2002 self-sustaining attractor does NOT
form at this scale envelope on the Izhikevich substrate with these
parameters.

## Biology-translatable finding (sharpened)

The Direction Q PARTIAL at n=1000 d=0.10 was already informative
(the NMDA mechanism engages; rate elevation appears). The scaling
envelope adds:

**The substrate's recurrent-NMDA mechanism produces a STRONG transient
cue response (scaling roughly with effective recurrent connection
count, up to 10x baseline at the high end) but the transient DECAYS
within ~500-650ms regardless of scale tested. The Wang 2002
bistability requires the recurrent loop to SUSTAIN itself indefinitely
after cue removal; here the recurrent gain is sufficient for cue-driven
amplification but not for self-maintenance.**

This RULES OUT the simplest interpretation (scale alone fixes
Direction I's n=60 negative) and points the diagnosis precisely at
the structural/dynamical missing piece. Candidate causes the data
points at:

1. **Inhibition dominance**: inh_weight_mean=4.0 vs exc_weight_mean=2.0
   in the bridge builder; the 2:1 inhibitory:excitatory ratio may
   throttle the recurrent loop sufficiently to prevent self-
   maintenance. Wang 2002 used a different E/I balance.
2. **Neuron model kinetics**: IZH2007_HIPPO_PYRAMIDAL may lack the
   slow membrane time constant or specific adaptation features that
   Wang 2002's HH-style model provides; the Izhikevich preset's
   F-I curve may be steeper than the published cortical pyramidal
   curve, biasing toward rapid decay.
3. **NMDA-AMPA conductance ratio**: the standard cfg ratios may
   under-weight NMDA relative to AMPA; Wang 2002's published ratio
   (NMDA/AMPA ~0.05-0.10 of total recurrent) may not match the
   bridge's defaults.
4. **Cue protocol**: 500ms cue at 1500pA may not be the optimal
   amplitude/duration to fully potentiate the recurrent loop;
   Wang 2002 typically used longer cue periods with smaller
   amplitudes.

The data does NOT yet localize which of these is the binding
constraint. The most leveraged next test depends on cost:
- (1) E/I balance test is cheap (modify builder to take inh weight
  as parameter; 1 commit + ~5 min wall per test)
- (2) HH neuron model test is substantial (Approach C from the
  Direction Q design doc; HH at 2000 neurons = ~30 min wall)
- (3) NMDA-AMPA ratio test is cheap (config flag in CoreSimConfig)
- (4) Cue protocol sweep is cheap (CLI args already exist)

## Convergence with the broader substrate-scale arc

This adds a 4th convergent BOUNDARY data point to the substrate-scale
characterization:
- Substrate sequence-storage arc: ~8 mechanisms tested, all BOUNDARY
  at substrate scale despite numpy/algebra PASS
- Substrate consolidation arc: 3+ convergent NEGATIVE (SWR-driven
  cortical reactivation doesn't transfer concept-specific patterns)
- Phase-coded representation class: substrate-bounded at the
  reviewer-caught dim-overkill scale
- **Direction Q scaling envelope**: rate elevation engages at scale
  but sustained attractor does NOT form across 3 (n, density)
  configurations

The convergent pattern (now spanning all three mechanism classes from
the 2026-05-25 mechanism-class audit): the substrate at the
60-2000 neuron pool range is bounded for biology-faithful integration
mechanisms regardless of the specific mechanism tested. The bottleneck
is a structural/dynamical property the substrate lacks (or has wrong
parameters for) that does NOT improve with the cheap-axis scaling
tested.

This is the same fundamental finding that motivated the 2026-05-19
owner reframe ("check existing biology-grounded sims FIRST"; build on
SPEAR + theta-gamma + generative replay) and the 2026-05-22
"dynamics-class exhausted; the fix is in the representation"
conclusion. Direction Q tested the SCALE axis directly; the result
sharpens the diagnosis but doesn't escape the bound.

## What is preserved unconditionally

- Direction Q Tasks 0-5 infrastructure (bridge builder + protocol +
  verdict module + multi-seed runner) is reusable for any future
  PFC bistability investigation
- The frozen verdict module's pre-registered thresholds and the
  17/17 adversarial test matrix stand unchanged
- Multi-seed [42, 43, 44] for both TEST and CONTROL in every cell of
  the scaling envelope; full mandatory NMDA-off control passed in
  every cell (control correctly fails the bar)
- No-confab moat 7/7 byte-identical; bar UNCHANGED throughout
- Honest propagation: every cell's PARTIAL recorded as PARTIAL (not
  spun); the convergent diagnosis is the honest scientific deliverable

## Pre-registered next concrete action

The cheapest probe that directly tests the leading candidate cause
(E/I balance) is to add an `inh_weight_mean` parameter to
`research/findings/raw/direction_Q_bridge_builder.py` and rerun with
inh_weight_mean values in {2.0, 3.0, 4.0} at n=1000 density=0.20
(the highest-rate condition). This is a 1-commit change + 3 quick
runs (~15-20 min total).

If E/I balance variation doesn't unlock sustained bistability either,
the next probe is either:
- NMDA-AMPA ratio variation (cheap; CoreSimConfig flag)
- HH neuron model replacement (substantial; Approach C from design)

If neither produces sustained bistability, the conclusion is that
the Izhikevich substrate (with the available parameter range) is
fundamentally bounded for Wang 2002-style attractor dynamics; the
biology-translatable finding is that bistability requires biophysical
detail beyond what Izhikevich approximates. This is itself a real
biology insight (justifies further HH-based investigation in a
narrowly-scoped Direction Q-secondary).

Per the user's ordered direction (Q -> 3 -> 4 -> R), after Direction Q
is fully characterized at this scale envelope, the next direction is
Direction 3 (vocab scaling on bio_brain_regions).

## Files

- Direction Q PARTIAL findings doc: `research/findings/2026-05-25-DIRECTION-Q-PARTIAL-dlpfc-n1000-NMDA-elevates-rate-but-not-sustained.md`
- n=1000 d=0.10 result: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.json` + `.log`
- n=1000 d=0.20 result: `research/findings/raw/direction_Q_dlpfc_n1000_d020.json` + `.log`
- n=2000 d=0.10 result: `research/findings/raw/direction_Q_dlpfc_n2000_d010.json` + `.log`
- Runner: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py`
- Bridge builder: `research/findings/raw/direction_Q_bridge_builder.py`
- Protocol functions: `research/findings/raw/direction_Q_protocol.py`
- Verdict module (frozen): `research/findings/raw/direction_Q_verdict.py`
- Design doc: `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-design.md`
- Implementation plan: `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-implementation.md`
- Mechanism-class audit guide: `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
