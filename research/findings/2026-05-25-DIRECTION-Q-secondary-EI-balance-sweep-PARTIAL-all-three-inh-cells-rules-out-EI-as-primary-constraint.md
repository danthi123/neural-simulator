# Direction Q-secondary E/I balance sweep: all 3 inh values {2.0, 3.0, 4.0} at n=1000 d=0.20 yield PARTIAL (rate_ratio scales monotonically 8.5x -> 30.3x as inh drops, but sustained_sec only stretches 0.6s -> 1.0s; bar 3.0s unmet at every cell); the biology-translatable finding is that E/I balance is NOT the primary constraint on Wang 2002 attractor formation in the Izhikevich substrate

**Date:** 2026-05-25
**Status:** All 3 cells in pre-registered E/I sweep = Q_BISTABILITY_PARTIAL; closes the "E/I balance alone fixes Direction Q" hypothesis; sharpens the biology-translatable diagnosis further toward neuron-model kinetics / NMDA-AMPA conductance ratio / cue protocol
**Wall:** 5.7 min (18 conditions, GPU CuPy backend)

## What was tested (pre-registered post-Q-prime chain)

Per the Direction Q-prime PARTIAL findings doc
(`research/findings/2026-05-25-DIRECTION-Q-prime-scaling-envelope-density-and-neuron-count-BOTH-yield-PARTIAL-substrate-cannot-form-sustained-attractor.md`),
the pre-registered next concrete action was:

> The cheapest probe that directly tests the leading candidate cause
> (E/I balance) is to add an `inh_weight_mean` parameter to
> `research/findings/raw/direction_Q_bridge_builder.py` and rerun
> with inh_weight_mean values in {2.0, 3.0, 4.0} at n=1000 density=0.20
> (the highest-rate condition). This is a 1-commit change + 3 quick
> runs (~15-20 min total).

Direction Q-secondary executes that probe exactly as specified.

Bar UNCHANGED throughout (frozen in
`research/findings/raw/direction_Q_verdict.py` at Task 3):
`_Q_RATE_RATIO_MIN=2.0`, `_Q_DELAY_MIN_SEC=3.0`, `_Q_MIN_SEEDS_PASS=3`.

## E/I sweep multi-seed table (3 inh values x 3 test seeds + 3 control seeds = 18 runs)

All cells at n=1000, dlpfc_density=0.20, seeds {42, 43, 44}, baseline
500ms, cue 500ms at 1500pA / 50% fraction, delay 3000ms / 50ms bins.
Mandatory NMDA-off (AMPA-only) control per cell.

| inh_w | TEST ratio (mean / min-max) | TEST sustained_sec (per-seed) | CTRL ratio max | CTRL sustained max | Verdict |
|---|---|---|---|---|---|
| 2.0 | 30.28 / 27.68-32.57 | [0.85, 1.00, 0.95] | 1.06 | 0.0 | PARTIAL |
| 3.0 | 13.77 / 12.89-14.89 | [0.70, 0.70, 0.70] | 1.06 | 0.0 | PARTIAL |
| 4.0 | 8.47 / 8.03-9.07 | [0.55, 0.55, 0.60] | 1.06 | 0.0 | PARTIAL |

Per-cell observations:
- TEST condition (NMDA-on) meets the rate_ratio bar (>=2.0) by very
  wide margins at every inh value (4x-15x the bar).
- TEST condition FAILS the sustained_sec bar (>=3.0s) in every seed
  at every inh value. Max observed = 1.0s (33% of the bar).
- CONTROL condition (NMDA-off / AMPA-only) correctly fails the
  rate_ratio bar at every seed and every inh value (consistent ~1.0
  ratio with sustained=0.0). No control seed met the bar at any
  inh value (no VOID_CONTROL_ALSO_PASSED branch triggered).
- Verdict = Q_BISTABILITY_PARTIAL (pre-registered tag) at all 3 cells.

## The decisive pattern (sharpening Q-prime's scaling-envelope diagnosis)

Lowering inh_weight_mean from 4.0 (the prior hardcoded value, inh:exc
= 2:1) toward 2.0 (parity with exc_weight_mean=2.0):

| inh_w | inh:exc ratio | TEST rate_ratio mean | sustained_sec mean |
|---|---|---|---|
| 4.0 | 2:1 (inh dominant) | 8.47 | 0.57 |
| 3.0 | 1.5:1 | 13.77 | 0.70 |
| 2.0 | 1:1 (parity) | 30.28 | 0.93 |

- **Rate gain scales strongly**: 3.6x improvement in mean delay
  rate_ratio when inh halves (8.47 -> 30.28). The recurrent NMDA
  loop IS more strongly amplified at lower inhibition, as biophysically
  expected.
- **Decay timescale stretches modestly**: sustained_sec improves
  from 0.57s (inh=4.0) to 0.93s (inh=2.0). 1.6x improvement.
- **No cell crosses the 3.0s bar**: best case (inh=2.0 / seed 43)
  reaches 1.0s, which is 33% of the required sustained-elevation
  duration.
- **Baseline rate barely moves** (~0.76-0.85 Hz at all inh): the
  network's spontaneous firing rate is not the bottleneck.
- **Cue rate scales strongly** (363 Hz at inh=4.0 -> 519 Hz at inh=2.0):
  the cue-driven transient is much larger at lower inh, but the
  POST-cue attractor still decays in ~1 second.

## Cross-validation: byte-identical reproduction of prior Q-prime cell

The inh=4.0 cell of the new E/I sweep is byte-identical to the prior
Direction Q-prime n=1000 d=0.20 result:

| seed | metric | prior Q-prime | new E/I (inh=4.0) | diff |
|---|---|---|---|---|
| 42 | rate_ratio | 8.0325 | 8.0325 | 0.000000 |
| 42 | sustained_sec | 0.5500 | 0.5500 | 0.000000 |
| 42 | baseline | 0.7825 | 0.7825 | 0.000000 |
| 43 | rate_ratio | 9.0745 | 9.0745 | 0.000000 |
| 43 | sustained_sec | 0.5500 | 0.5500 | 0.000000 |
| 43 | baseline | 0.8000 | 0.8000 | 0.000000 |
| 44 | rate_ratio | 8.3067 | 8.3067 | 0.000000 |
| 44 | sustained_sec | 0.6000 | 0.6000 | 0.000000 |
| 44 | baseline | 0.7200 | 0.7200 | 0.000000 |

This confirms:
- The bridge_builder modification (added `inh_weight_mean` parameter
  with default 4.0) is truly default-preserving at the byte level.
- The new E/I runner gives byte-equivalent semantics to the prior
  scale-up runner at the same parameter set.
- Prior Q-prime artifacts (`direction_Q_dlpfc_n1000_d020.json` /
  `.log`) remain valid as the inh=4.0 reference; no recomputation
  needed and no prior commit revised.

## Biology-translatable finding (sharpened further)

The Direction Q-prime scaling envelope ruled out "scale alone"
(density and n_neurons both already tested; both PARTIAL). The
Direction Q-secondary E/I sweep now rules out the second candidate
cause from the Q-prime "candidate causes" list:

**E/I balance is NOT the primary constraint on Wang 2002 attractor
formation in the Izhikevich substrate.** Lowering inhibition from
the 2:1 inh-dominant default toward 1:1 parity DOES strongly amplify
the cue-driven NMDA transient (3.6x in rate_ratio) but only modestly
stretches the post-cue decay timescale (1.6x in sustained_sec).
The attractor's failure to self-maintain is invariant under the
tested E/I axis at the level required to cross the 3.0s bar; the
bottleneck lies elsewhere.

The remaining candidates from the Q-prime list (now narrowed to 2):

1. **Neuron model kinetics (IZH2007_HIPPO_PYRAMIDAL)**: the
   Izhikevich preset's F-I curve and adaptation may not match what
   Wang 2002's HH-style model provides. Specifically, the
   refractory dynamics, the spike-frequency adaptation timescale,
   or the slow afterhyperpolarization may be wrong for sustained
   recurrent activity. Substantial cost (~30 min wall at n=2000 HH
   per the Q design doc Approach C).
2. **NMDA-AMPA conductance ratio**: the standard cfg ratios may
   under-weight NMDA relative to AMPA; Wang 2002's published ratio
   may not match the bridge's defaults. Cheap to test (CoreSimConfig
   flag; one or two runs).

Cue protocol (Q-prime candidate cause #4) is unlikely given the
E/I result: the cue-driven amplification scales strongly with E/I
weakening (519 Hz at inh=2.0), which means cue stimulus delivery is
NOT the limit; the limit is post-cue attractor maintenance.

## Convergence with the broader substrate-scale arc

This adds a 5th convergent BOUNDARY data point to the substrate-
scale characterization (after the Q-prime 3-cell envelope):

- Substrate sequence-storage arc: ~8 mechanisms tested, all BOUNDARY
  at substrate scale despite numpy/algebra PASS
- Substrate consolidation arc: 3+ convergent NEGATIVE
- Phase-coded representation class: substrate-bounded at the
  reviewer-caught dim-overkill scale
- Direction Q scaling envelope: 3 (n, density) cells, all PARTIAL
- **Direction Q-secondary E/I sweep**: 3 inh cells, all PARTIAL;
  recurrent gain scales monotonically with the E/I axis but the
  decay timescale doesn't escape the ~1-second ceiling

The convergent pattern continues to indicate that the substrate at
60-2000 neuron pool range is bounded for biology-faithful integration
mechanisms regardless of the cheap-axis tested. The bottleneck is
not in the E/I balance any more than it was in scale; it's in the
intrinsic neuron/synapse kinetics that the Izhikevich approximation
substitutes for Wang's HH biophysics.

## What is preserved unconditionally

- Direction Q Tasks 0-5 infrastructure (bridge builder + protocol +
  verdict module + multi-seed runner) reusable for any future
  PFC bistability investigation. The bridge_builder modification
  adds ONE keyword parameter with default 4.0 (default-preserving).
- The frozen verdict module's pre-registered thresholds and the
  17/17 adversarial test matrix stand unchanged.
- 30/30 Direction Q test suite passes post-modification (including
  the 3 bridge_builder construction tests and the 7 grounding tests).
- Multi-seed [42, 43, 44] for both TEST and CONTROL in every cell
  of the E/I sweep; full mandatory NMDA-off control passed in every
  cell (control correctly fails the bar at every inh).
- No-confab moat 7/7 byte-identical; bar UNCHANGED throughout.
- The inh=4.0 cell byte-identically reproduces the prior Q-prime
  n=1000 d=0.20 result; no prior committed artifact was modified.
- Honest propagation: every cell's PARTIAL recorded as PARTIAL
  (not spun); the convergent diagnosis sharpening is the honest
  scientific deliverable.

## Pre-registered next concrete action (per verdict)

**All cells = PARTIAL** -> sharpened diagnosis branch.

The cheapest of the 2 remaining candidate causes is NMDA-AMPA
conductance ratio. Testing it requires:

1. Identify the relevant CoreSimConfig flags (likely
   `nmda_g_ratio` or similar in `sim/config.py`).
2. Add the parameter to the bridge_builder analogous to
   `inh_weight_mean`.
3. Sweep at the same n=1000 d=0.20 cell with values bracketing the
   default and Wang 2002's published ratio.
4. If sustained_sec stays bounded at ~1s across the NMDA-AMPA ratio
   sweep, the diagnosis localizes definitively to neuron-model
   kinetics; the conclusion is that the Izhikevich preset (with
   the available parameter range) cannot support Wang 2002-style
   attractor dynamics. This would justify the substantial-cost
   HH-based investigation (Approach C from the Q design).

If neuron-model kinetics is confirmed as the binding constraint,
the biology-translatable finding upgrades from "the Izhikevich
substrate is bounded" to "Wang 2002 bistability requires biophysical
detail beyond what Izhikevich approximates" -- itself a real
biology insight that justifies the HH-based investigation in a
narrowly-scoped Direction Q-tertiary.

**Important context note from the user's ordered direction (Q -> 3 -> 4 -> R):**
Q is now characterized at the n=1000 d=0.20 cell along BOTH the
scaling axis (Q-prime) and the E/I axis (Q-secondary). The next
ordered direction (3, 4, R) takes priority over a Direction Q-tertiary
NMDA-AMPA / HH investigation per the user's pre-stated ordering;
the E/I sweep was queued because it was the cheapest remaining
characterization probe.

## Files

- E/I sweep runner: `research/findings/raw/direction_Q_secondary_ei_balance_runner.py` (new)
- Modified bridge builder: `research/findings/raw/direction_Q_bridge_builder.py`
  (added `inh_weight_mean` kwarg with default 4.0; default-preserving)
- E/I sweep result: `research/findings/raw/direction_Q_secondary_ei_balance_n1000_d020.json` (new)
- E/I sweep log: `research/findings/raw/direction_Q_secondary_ei_balance_n1000_d020.log` (new)
- Prior Direction Q-prime envelope findings:
  `research/findings/2026-05-25-DIRECTION-Q-prime-scaling-envelope-density-and-neuron-count-BOTH-yield-PARTIAL-substrate-cannot-form-sustained-attractor.md`
- Prior Direction Q PARTIAL findings:
  `research/findings/2026-05-25-DIRECTION-Q-PARTIAL-dlpfc-n1000-NMDA-elevates-rate-but-not-sustained.md`
- Frozen verdict module:
  `research/findings/raw/direction_Q_verdict.py` (UNCHANGED;
  byte-identical to prior commits)
- Frozen protocol functions:
  `research/findings/raw/direction_Q_protocol.py` (UNCHANGED)
- Prior reference cell (inh=4.0 byte-identical):
  `research/findings/raw/direction_Q_dlpfc_n1000_d020.json` (UNCHANGED)
