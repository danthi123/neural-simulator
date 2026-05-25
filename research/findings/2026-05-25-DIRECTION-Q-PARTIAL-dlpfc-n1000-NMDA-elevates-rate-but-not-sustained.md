# Direction Q at n=1000 = Q_BISTABILITY_PARTIAL: NMDA-on cleanly elevates dlpfc_wm delay-period rate 2.27x above baseline multi-seed (vs 1.03x AMPA-only control) but the elevation is FRONT-LOADED (high early bins decaying to near-baseline by ~500ms), not sustained 3 sec required for Wang 2002 bistability; biology-translatable: scale (60->1000) engages NMDA mechanism but recurrent attractor doesn't form at density=0.10

**Date:** 2026-05-25
**Status:** Q_BISTABILITY_PARTIAL (pre-registered verdict); pillar candidate NOT promoted (PARTIAL is below PASS bar); honest characterization recorded; pre-registered next direction = scaling envelope probe at n={200, 500, 2000} OR density variation

## What was tested

Direction Q from the 2026-05-25 mechanism-class audit + design:
scale dlpfc_wm from 60 neurons (where Direction I bistability
failed across 4 cheap probes) to 1000 neurons with dense recurrent
NMDA-rich connectivity; test Wang 2002 delayed-response protocol;
measure NMDA-driven persistent activity at biological scale.

Runner: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py`
(Task 4; commit c60fdd8). Built TDD-style via subagent-driven-
development (Tasks 0-5; commits 8715aa3 / 70695cc / 957ac51 /
c93495f / c60fdd8).

Decisive run launched 02:11 EDT (controller-only per Task 6);
completed 02:14 EDT. Total wall 3.59 min on CuPy/RTX 3090.

Config:
- n_dlpfc=1000, dlpfc_density=0.10 (vs Direction I's n=60)
- baseline 500ms; cue 500ms at 1500pA (cue_fraction=0.5);
  delay 3000ms in 50ms bins (60 bins)
- seeds [42, 43, 44] for both NMDA-on (test) and NMDA-off (control)
- Bridge: dlpfc_wm IZH2007_HIPPO_PYRAMIDAL + q_stim_input
  IZH2007_RS_CORTICAL_PYRAMIDAL; 1200 total neurons; 119,629
  synapses; dt=0.5ms; nmda_tau_decay=100.0 (Wang 2002 calibration)
- Bar UNCHANGED from Task 3 frozen module: _Q_RATE_RATIO_MIN=2.0,
  _Q_DELAY_MIN_SEC=3.0, _Q_MIN_SEEDS_PASS=3

## Result: Q_BISTABILITY_PARTIAL

Multi-seed aggregate:

| Seed | Condition | baseline (Hz) | cue (Hz) | mean_delay (Hz) | rate_ratio | sustained_sec |
|---|---|---|---|---|---|---|
| 42 | NMDA-on  | 0.760 | 269.965 | 1.691 | 2.23 | 0.45 |
| 43 | NMDA-on  | 0.760 | 267.555 | 1.663 | 2.19 | 0.35 |
| 44 | NMDA-on  | 0.705 | 259.870 | 1.675 | 2.38 | 0.35 |
| 42 | NMDA-off | 0.677 | 200.570 | 0.649 | 0.96 | 0.00 |
| 43 | NMDA-off | 0.625 | 199.982 | 0.627 | 1.00 | 0.00 |
| 44 | NMDA-off | 0.580 | 195.315 | 0.656 | 1.13 | 0.00 |

**Pre-registered verdict** (computed by frozen `direction_Q_verdict.py`
from recorded per-seed tuples): `Q_BISTABILITY_PARTIAL`
- rate_ratio bar (>= 2.0): PASS multi-seed (3/3 with mean 2.27)
- sustained_sec bar (>= 3.0): FAIL multi-seed (0/3; max 0.45s)
- control bar (must NOT pass either): PASS (all 3 control seeds at
  ratio <= 1.13; sustained = 0.0; correctly distinguishes NMDA from
  baseline)
- VOID branch (control-also-passes): not triggered

## Smell-test PASS (the PARTIAL is genuine, not an artifact)

Recomputed from raw JSON per-seed bin-by-bin trajectories:
- Seed 42 first 3 bins NMDA-on: [14.625, 15.125, 10.6] Hz (cue
  residual); bin 11+ settles to ~0.7-1.0 Hz (slightly above
  baseline 0.76)
- Seed 42 control: similar shape but smaller transient + same
  near-baseline late settling

The pattern is FRONT-LOADED cue-residual decay, not sustained
attractor activity. The 2.27x mean ratio comes entirely from the
first ~10 bins (500ms) when activity hasn't yet decayed. After
~500ms, activity is at or near baseline regardless of NMDA.

Recomputed verdict-module logic: ratio threshold met (all 3 NMDA
seeds >=2.0); sustained threshold NOT met (all 3 << 3.0); control
all below bar; therefore PARTIAL per pre-registered logic. The
verdict matches what the frozen module would output given the same
recorded data; no re-run; bar unchanged; pre-registered tag.

NMDA differential (TEST mean_delay - CONTROL mean_delay across 3
seeds): +1.032 Hz with std ~0.03 across seeds. Reproducible across
seeds; the NMDA mechanism IS engaged at n=1000.

## Biology-translatable insight (load-bearing)

Three findings, all biology-translatable per the project goal:

**1. Scale 60 -> 1000 demonstrably engages the NMDA mechanism.**
Direction I at n=60 found NMDA produced no measurable elevation
above baseline across 4 probes. At n=1000, NMDA-on consistently
elevates delay-period rate 2.27x above baseline + 2.60x above
NMDA-off control. The Wang 2002 mechanism IS active at this scale.

**2. But the recurrent attractor does NOT form at density=0.10.**
The elevation is FRONT-LOADED (a decay transient from the cue),
not sustained. Wang 2002 bistability requires the recurrent loop
to maintain its own activity AFTER the cue is removed; here the
recurrent gain is insufficient and activity returns to near-baseline
within ~500ms.

**3. Direct biology-translatable claim:** for the project's substrate
(Izhikevich + dt=0.5ms + IZH2007_HIPPO_PYRAMIDAL preset + 100ms NMDA
decay), n=1000 neurons at density=0.10 is the FRONT of the
scaling-threshold curve for NMDA bistability - sufficient to engage
the mechanism, insufficient to sustain it. Wang 2002's published
network used density ~0.20; our density=0.10 may be the precise gap.
A follow-up density-sweep at n=1000 (densities 0.10, 0.15, 0.20)
would localize the threshold without scaling neuron count further.

## What this rules in vs out

- **Rules in:** scale matters; NMDA mechanism engages at biological
  scale (n=1000); the project's substrate machinery (NMDA kernel,
  brain-region framework, IZH preset) correctly produces NMDA-driven
  rate elevation at multi-seed margin.
- **Rules out (at this density):** density=0.10 + 1000 neurons +
  the chosen cue protocol is sufficient for cue-driven activation
  but not for self-sustaining attractor formation. The Direction I
  bound is now PARTIALLY closed (the mechanism engages); the
  REMAINING gap is in the recurrent loop gain, not in the substrate's
  basic capacity to express NMDA.
- **Does NOT rule out:** that density=0.20 (Wang 2002 published)
  OR n=2000 OR longer NMDA tau OR stronger recurrent weights might
  produce true sustained bistability at this scale. The next probe
  per the pre-registered chain characterizes this.

## Pre-registered post-PARTIAL chain (from design doc + audit)

**Direction Q-prime (next concrete action; controller-only)**: density
+ scale envelope characterization. Run the same `direction_Q_dlpfc_scale_up_standalone.py`
runner with a small parameter sweep:
- (n=1000, density=0.15)
- (n=1000, density=0.20) (Wang 2002 published density)
- (n=2000, density=0.10)
- (n=2000, density=0.20)

The cheapest single follow-up that tests the density hypothesis:
just rerun with --dlpfc-density 0.20 at n=1000 (~5 min wall;
substantively answers "is density=0.10 the gap?").

If density=0.20 PASSes (sustained >= 3 sec multi-seed): closes the
bound; the Wang 2002 published parameter set works in our substrate;
pillar n=105 candidate via formal adversarial review.

If density=0.20 still PARTIAL: try n=2000; the bound is in neuron
count for the chosen connectivity.

If neither passes: the gap is in the neuron model (IZH2007 preset
may lack the kinetic features needed; Approach C HH replication
becomes the next step).

## What is preserved unconditionally

- Tasks 0-5 + Task 6 of Direction Q completed cleanly; full TDD
  discipline observed.
- Direction Q runner + verdict + bridge builder + protocol modules
  are reusable infrastructure for the scaling envelope probe + any
  future PFC bistability investigation.
- The pre-registered verdict module's 17/17 adversarial tests stand;
  thresholds frozen at 2.0/3.0/3; the PARTIAL output is the verdict
  the frozen module produced from the recorded data, not a tuning.
- The (c) loop NEGATIVE / substrate-scale-bounded arc convergent
  diagnosis stands; this finding ADDS to it (mechanism engages at
  scale but doesn't yet sustain).
- No protected/frozen/moat modification.
- The no-confab moat (abstention_gate 7/7) is unchanged.

## Discipline preserved

- Bar UNCHANGED at 2.0/3.0/3 throughout (frozen in Task 3 module)
- Multi-seed [42, 43, 44] for both TEST and CONTROL (full mandatory
  control)
- Smell-test applied immediately at result (recomputed verdict from
  recorded JSON; per-seed bin trajectories confirm front-loaded
  decay pattern, not sustained activity)
- HONEST PROPAGATION: PARTIAL recorded as PARTIAL (not spun as
  "directional PASS" or "near-PASS"); pre-registered next direction
  identified for further characterization
- Both remotes propagated
- ~5 minutes total decisive wall (much faster than the design doc's
  3-6 hr estimate; the smoke pattern scale-up was accurate)

## Files

- Runner: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.py`
- Bridge builder: `research/findings/raw/direction_Q_bridge_builder.py`
- Protocol functions: `research/findings/raw/direction_Q_protocol.py`
- Verdict module (frozen): `research/findings/raw/direction_Q_verdict.py`
- Result JSON: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.json`
- Log: `research/findings/raw/direction_Q_dlpfc_scale_up_standalone.log`
- Design doc: `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-design.md`
- Implementation plan: `docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-implementation.md`
- Mechanism-class audit guide: `docs/plans/2026-05-25-prior-mechanism-class-audit-direction-selection-guide.md`
- Direction I prior negatives (n=60): `research/findings/2026-05-24-DIRECTION-I-Stage1-CLOSED-PFC-bistability-genuinely-fails-substrate-scale.md`
