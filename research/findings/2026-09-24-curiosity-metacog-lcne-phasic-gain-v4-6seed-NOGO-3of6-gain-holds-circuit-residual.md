---
type: finding
status: no-go
claim_check: measured
date: 2026-09-24
lane: B — curiosity (novelty / epistemic-gap crave) x E1 metacognition
mechanism: curiosity-lcne-phasic-gain v4 (metacog comparator -> frozen point-edge onto ASK, plus a phasic lc_ne population
  with GIRK autoinhibition whose burst silences ask_fb, the relay carrying ASK's own slow GIRK negative feedback; a
  response-gain change on the edge's drive, not an added current; every synapse fixed-weight)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_curiosity_lcne_phasic_gain_derisk.py
prereg: docs/plans/2026-09-24-curiosity-lcne-phasic-gain-PREREG.md
artifact: research/findings/raw/_curiosity_lcne_phasic_gain_6seed_combined.json
verdict: NO-GO 3/6 (v3 was 1/6). The gain the method adds passes its own gate on every seed (G3 gain share 0.37-0.92,
  v3 0.08-0.25). The three failing seeds fail on gates the gain does not control -- seeds 44 (G1) and 100 (G7) read the
  same way with the gain lesioned, and seed 42's multiplicative test (G11) is UNDEFINED because its no-lc response
  rises across only two points of the drive grid. A verdict on this method at this operating point, not on the
  capability.
---

# Curiosity x metacog v4, phasic LC-NE gain: 6-seed NO-GO (3/6), the gain holds and the residual is the base circuit

## 1. Result

Artifact: `research/findings/raw/_curiosity_lcne_phasic_gain_6seed_combined.json` (the pre-registered `--combine` over
`research/findings/raw/_curiosity_lcne_phasic_gain_s42.json` and the five other `_s<seed>.json` files beside it, all at
runner revision `3ff9fe89d`, `git_dirty: false`). The operating point was chosen on dev seeds 7-10 only; all six
evaluation seeds were held out.

(Per-seed values from `per_seed` in the artifact, rounded to 2-3 decimals; "gain-lesioned" is `rho_gain_lesion_arm` /
`rho_swap_gain_lesion_arm`.)

<!--derived-->
| seed | verdict | failing gate | G3 gain share | rho (gain-lesioned) | rho_swap (gain-lesioned) |
|---|---|---|---|---|---|
| 42 | -- | G11 UNDEFINED | 0.463 | -0.991 (-0.991) | -0.909 (-0.909) |
| 43 | GO | none | 0.622 | -0.916 (-0.879) | -0.907 (-0.936) |
| 44 | -- | G1 | 0.503 | -0.691 (-0.755) | -1.000 (-1.000) |
| 100 | -- | G7 | 0.919 | -0.834 (-0.738) | -0.795 (-0.770) |
| 101 | GO | none | 0.371 | -0.953 (-0.953) | -0.909 (-0.909) |
| 102 | GO | none | 0.544 | -0.943 (-0.943) | -0.970 (-0.970) |

Integrity gates pass on every seed.

## 2. Reading it

- **The method's own claim holds on all six seeds.** v3's additive current owned 8-25% of the ASK dynamic range
  (`research/findings/2026-09-24-curiosity-metacog-lcne-modulator-6seed-NOGO-calibration-seed-only.md`); the
  feedback-withdrawal gain owns 37-92%, above the 0.2 floor everywhere.
- **Seeds 44 and 100 fail on the base circuit.** Seed 44's evidence-to-ASK coupling is not monotone enough (rho
  -0.69) and it is no better with the gain lesioned (-0.75); seed 100's class-swap curve (-0.79) is the same with the
  gain lesioned (-0.77). The modulator cannot repair a coupling the edge and ASK pool do not carry.
- **Seed 42's G11 is an instrument limit, not a measured failure.** The gate scores only the rising part of the
  reference (no-lc) response curve. On seed 42 that curve rises between drive 0.9 and 1.0 only, which leaves two points
  and the gate counts it UNDEFINED. The drive grid (0, 0.6-1.2 in 0.1 steps) is too coarse for this seed's steep
  threshold.

## 3. Next (per the LAW: the capability stays open)

1. **Instrument:** a finer drive grid around each seed's own ASK threshold, so G11 is defined on every seed. This is a
   new prereg, not an edit of this one.
2. **Companion process:** the base-circuit failures are an operating-point problem. Each seed's ASK pool sits at a
   different place on its firing curve, and seed 101's response barely clears 1.7 Hz. The real system runs a
   homeostatic set-point alongside such a coupling; build it as neurons and synapses (the D1 operating-point
   stabiliser lane already has a per-seed regulator to adapt), then re-test v4 on top of it.
3. The engine issue the build found (`inject_explicit_wiring` keeps a stale GABA_B routing mask on a second inject) is
   worked around in this runner and needs its own fix.

## 4. Honesty

Functional read-outs only: "curiosity" and "uncertainty" name spiking populations and their measured rates. No felt
state is asserted.
