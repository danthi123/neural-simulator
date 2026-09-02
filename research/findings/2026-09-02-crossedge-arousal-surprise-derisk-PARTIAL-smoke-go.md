---
type: finding
status: contributing
lane: onebrain-integration-design
date: 2026-09-02
mechanism: crossedge-arousal-surprise-prediction-gain
seeds: [42, 43]
runner: research/runners/_crossedge_arousal_surprise_derisk.py
artifacts:
  - research/findings/raw/_crossedge_arousal_surprise_smoke.json
builds_on:
  - research/findings/2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md
  - research/findings/2026-08-13-affect-lc-arousal-population-GO.md
  - research/findings/2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md
  - research/findings/2026-08-10-NE-LC-gain-vigilance-REAL-SUBSTRATE-does-not-robustly-transfer-3of6.md
---

# Cross-edge #3 (arousal -> D2 surprise, LC-NE prediction-gain) -- 2-seed numpy SMOKE-GO, PARTIAL pending the 6-seed cupy soak

**DE-RISK ONLY. NO production wire-in. NOT a full GO** -- a 2-seed numpy indicator that the rank-#3 cross-edge from
the one-brain integration design (`2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`) is
buildable and behaves as an adaptive-gain modulation. The 6-seed cupy verify is QUEUED (harvested by the controller,
not this agent).

## The mechanism (brain-based, spiking, NOT a host multiplier)

The felt-AROUSAL state (the #81/#84 graded-affect ladder's `appraisal_lad_arousal` in [0,1], carried by the LC-like
salience-integrator population, `2026-08-13-affect-lc-arousal-population-GO.md`) is delivered by a SPIKING
neuromodulatory PROJECTION onto the D2 surprise pool (`surprise_production_organ.py` /
`_spiking_expectation_rpe_derisk.py`). A small `arousal` RS source population, driven at a rate set by the
felt-arousal STATE, projects DIFFUSELY (all-to-all -- the broadcast hallmark of ascending neuromodulation, distinct
from the block-diagonal prediction edges) onto the surprise pool with a FIXED weight (`plastic=False`). When arousal
FIRES, its synapses inject a tonic depolarising conductance that shifts the surprise pool up its f-I curve, so the
SAME asserted-patient input reads as MORE surprising (a lower effective threshold). The gain is carried ENTIRELY by
spiking synaptic transmission; the only host boundary is the felt-arousal STATE delivered as drive to the arousal
SOURCE (exactly the surprise organ's own sensory boundary -- the asserted token is likewise a drive). Grounding:
Aston-Jones & Cohen 2005 (LC-NE adaptive gain, Annu Rev Neurosci 28:403).

## The four de-risk checks (a-d), 2/2 seeds

Config: `arousal_n=24`, `w_arousal=0.30` (diffuse, 6912 arousal->surprise synapses), arousal HIGH=600 pA vs
LOW/silent=0 pA. Numbers (rounded) from `research/findings/raw/_crossedge_arousal_surprise_smoke.json`.

<!--derived-->
| seed | (a) contradict shift lo->hi (Δ) | gain-like ratio (contradictΔ / confirmΔ) | (b) lesion Δ | (c) frac attributable | (d) silent-state byte dev / base-connectivity | borderline flip |
|------|--------------------------------|------------------------------------------|--------------|-----------------------|-----------------------------------------------|-----------------|
| 42 | 5.45 -> 5.79 (+0.340) | 9.4x | +0.000 | 1.00 | 0.0 Hz / identical, only-edge-added | True (2.50 -> 2.90, thr 2.73) |
| 43 | 5.25 -> 5.53 (+0.277) | 11.5x | +0.014 | 0.95 | 0.0 Hz / identical, only-edge-added | True (2.38 -> 2.67, thr 2.56) |

- **(a) SHIFT.** For a FIXED contradict-mismatch input, HIGH arousal raises the spiking surprise Hz vs LOW/silent
  arousal (the control). The shift is **gain-like, not a DC bias**: the CONTRADICT shift is ~9-11x the CONFIRM shift
  (confirm stays ~0.2 Hz, far below the ~2.6-2.7 Hz threshold), i.e. arousal sharpens the already-firing violation
  more than the inhibited match -- the f-I-curve gain signature. (Reported as a ratio + raw per-condition magnitudes;
  NOT claimed as a permuted-control "selective" result.)
- **(b) LESION.** Zeroing the arousal->surprise synapses (`_lesion_arousal`) collapses the high-vs-low shift to ~0.
  The lesion HOLDS at measurement: the arousal->surprise pathway is `plastic=False` and Hebbian learning is off, so
  the zeroed weight cannot regrow.
- **(c) ATTRIBUTABLE.** `attributable_to(intact shift, lesion shift)` = 0.95-1.00: the shift IS the projection, not
  the host drive on the arousal source (which fires identically under lesion).
- **(d) BYTE-OFF.** With the arousal state OFF (source silent), the full circuit (arousal region + edge) reads
  **byte-identically** (0.0 Hz exact, confirm/contradict/novel) to the SHIPPED plain 4-region surprise organ
  (`build_expectation_circuit`); and the with-edge base connectivity is byte-identical to a without-edge pool
  (integration added ONLY the arousal->surprise synapses).
- **Borderline verdict flip (criterion (a)'s "different verdict"):** a FAINT/uncertain assertion (325 pA vs the full
  600) sits just below threshold when the source is silent (NOT surprised) but crosses it under high arousal
  (surprised) -- the SAME input, a DIFFERENT surprise verdict by arousal state. The adaptive-gain story: a subtle
  violation missed when drowsy is noticed when vigilant. See the honest scope on its seed-dependence.

## Why this is NOT the refuted NE-vigilance path

`2026-08-10-NE-LC-gain-vigilance-REAL-SUBSTRATE-does-not-robustly-transfer-3of6.md` found a GLOBAL multiplicative
synaptic-gain modulator (`NeuromodulatorManager.compute_synaptic_gain_multiplier`, scope=all) for a DETECTION d'
task transferring only 3/6, because one global operating point left heterogeneous neurons off the sensitive part of
their f-I curve -- "the operating point is implicit in the animal, held by a homeostatic set-point the idealisation
omitted." Two differences here: (1) this is a GENUINE spiking PROJECTION (arousal neurons fire -> conductance onto
surprise), not a host/global weight-scalar; (2) the D2 surprise pool already carries that missing companion -- the
per-block HOMEOSTATIC prediction-gain equaliser (`_homeostat`, GO 6/6, `2026-08-13-surprise-organ-homeostat-GO.md`)
that places each block at a firing set-point. The hypothesis -- a projection onto an already-homeostatically-held
pool is more seed-robust than a global multiply onto a raw one -- is SUPPORTED by the 2-seed smoke but is NOT yet
established; the 6-seed cupy soak is the test.

## Honest scope (weaker than the plasticity-gate siblings, by design)

- **The arousal->surprise weight is FIXED (a biologically-fixed ascending-modulation weight), NOT learned.** There is
  NO emergence/growth claim (the run_gate emergence arm does not apply). A learned/plastic ascending gain is a
  separate, later rung. This is acceptable per the design brief because the projection is genuinely spiking (the
  lesion zeros the SYNAPSES while the source still fires, and the effect vanishes).
- **The borderline verdict flip is SEED-DEPENDENT** (reported per-seed, not gated). The faint-assertion level (325
  pA) was bracketed to seed 42's threshold; it happened to flip on 43 too, but a fixed level need not flip on every
  seed (each seed's threshold + assert->surprise curve differ). The load-bearing checks (a-d) do NOT depend on it.
- **The de-risk runs on the BASE surprise circuit** (the `_homeostat` precision companion at its default; the queued
  cupy verify can toggle `BRAIN_SURPRISE_HOMEOSTAT`). The load-bearing question -- does the projection shift the read,
  is that shift the SYNAPSE -- is orthogonal to the precision equaliser.
- **Region-presence seam:** adding the arousal region as the LAST region left the 4 core regions' reads byte-clean
  (0.0 Hz), because their per-neuron draws are an identical RNG prefix. A production wire-in onto the shared pool
  would use the per-region threshold seam every merged organ already uses; this de-risk did not exercise that seam
  (it did not need to -- the append-last prefix was already clean here).
- **Instrument note (reported, not gated):** a reused-bridge silent-inertness re-read shows ~0.01 Hz residual = the
  un-reset ADAPTIVE-THRESHOLD drift the NE-vigilance finding characterised (the surprise pool's own firing over the
  intervening reads), NOT the edge -- (d) proves the edge inert when arousal is silent, cleanly, on a pre-contamination
  read.
- Functional read-outs only; no phenomenal-experience claim. 2 seeds is an indicator, not the 6-seed bar.

## Reproduce

```
# numpy smoke (this finding):
SIM_BACKEND=numpy python -m research.runners._crossedge_arousal_surprise_derisk --smoke \
    --out research/findings/raw/_crossedge_arousal_surprise_smoke.json
```

The QUEUED 6-seed cupy verify (controller-harvested, not yet run) is the same runner with
`SIM_BACKEND=cupy --seeds 42,43,44,100,101,102`, writing the basename
`_crossedge_arousal_surprise_6seed.json` under `research/findings/raw/`.

## Files

- Runner: `research/runners/_crossedge_arousal_surprise_derisk.py` (numpy CPU; NO `sim/` edit; additive; reuses the
  surprise circuit + drive/read helpers + `tools.lab.attributable_to` READ-ONLY by import).
- Artifact: `research/findings/raw/_crossedge_arousal_surprise_smoke.json` (+ provenance sidecar).
