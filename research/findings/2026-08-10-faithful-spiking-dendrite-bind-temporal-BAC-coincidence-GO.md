---
title: "The composer BIND now runs on a REAL TEMPORAL SPIKING dendrite: a Larkum-BAC coincidence (spike counts, not a host multiply) reproduces the conjunction and recovers zero-shot composition — the deep residual the rate-PARTIAL named is CLOSED"
date: 2026-08-10
type: finding
status: contributing
lane: composer
seeds: [42, 43, 44]
seed-waiver: 3-seed concrete scope. The faithful-mechanism claim (spiking coincidence reproduces the host-product bind within tolerance) is 3/3 on both grids; the stricter absolute recovery gate is 3/3 on 7x7 and 2/3 on 8x8, the one miss being READOUT-bound (see below), not a seed-variance artifact — so a 6-seed rerun bears mainly on the 8x8 tail, not the closure.
---

# The faithful spiking-dendrite BIND: a temporal Larkum-BAC coincidence closes the residual the rate sigma-pi could not

## What this closes

<!--derived-->

The 2026-08-09 dendritic-bind de-risk landed a PARTIAL: a more-biological RATE sigma-pi
(`sim/dendritic_neuron.py:apical_basal_coincidence` -> soma = `phi(basal)*phi(apical)`) recovered zero-shot
composition, but the adversarial verify CAUGHT that it is still ONE HOST MULTIPLY -- no time, no spikes,
`step()`'s membrane dynamics bypassed. The named residual: *"route the two primitive drives through the
two-compartment machinery so a REAL spiking dendrite computes the coincidence IN TIME (basal depolarization must
reach the soma while an apical PLATEAU is active), not a static phi*phi."* **This finding CLOSES it.**

## The mechanism (a genuinely temporal, spiking BAC unit)

<!--derived-->

New additive/default-off method `sim/dendritic_neuron.py:bac_spiking_coincidence` runs a REAL temporal spiking
process and returns per-channel SPIKE COUNTS: the basal drive (a saturating dendritic plateau -> bounded)
leaky-integrates the SOMA membrane (tau_m); a supra-threshold APICAL drive IGNITES a regenerative Ca2+ plateau
(graded, self-sustaining, decays plateau_tau) that injects a sustained depolarizing current into the soma across a
temporal WINDOW; a somatic SPIKE (HARD threshold + reset + refractory) fires ONLY when basal coincides IN TIME with
an active plateau. **The AND is the HARD SPIKE THRESHOLD acting on two individually SUB-THRESHOLD inputs -- the
conjunction a SOFT (sigmoid) soma cannot form** (`step()` sums inside a sigmoid, so basal alone leaks through; a
hard threshold with both inputs sub-threshold does not). theta is set HOMEOSTATICALLY (taught-only, ruler-free):
between the single-input membrane peak and the coincident sum. Signed factors -> biological ON/OFF push-pull. There
is NO host product anywhere in the conjunction path -- the per-channel bind value is a COUNT of threshold crossings.

## Result: GO on the faithful-mechanism claim (3/3), recovery universal at max mixing

<!--derived-->

Runner `research/runners/_teacher_loop_dendritic_bind_spiking_derisk.py`, 4 arms on the ONE frozen Izhikevich
reservoir (readout-only, de-clamped `bdsp_wmax=1e9`), 3 seeds x {7x7, 8x8} x s in {0,0.25,0.5,0.75,1.0}:

- **FAITHFUL reproduction (the core claim) = 3/3, both grids:** the spiking coincidence matches the host-product
  bind within tolerance (`>= product - 0.15`) on EVERY (seed, grid) cell -- it reproduces what the host `*` did.
- **Recovery of the additive break is UNIVERSAL at max mixing:** at s=1.0 (pure conjunction) all 6 (seed,grid)
  cells recover -- held-out recall 0.50-1.00 vs additive superposition 0.00-0.12 (margin +0.38..+1.00 >> the +0.30
  gate; every cell also >= 0.5). 7x7 s=1.0: spk 1.00/0.71/1.00. 8x8 s=1.0: spk 0.88/0.62/0.50.
- **Strict absolute-recovery gate (`>= additive+0.30 AND >= 0.5 at top-s`):** GO 3/3 on 7x7, 2/3 on 8x8.

## The FAITHFULNESS witnesses the PARTIAL could NOT show

<!--derived-->

- **TEMPORAL = 1.00 on every high-s grid-point, all 3 seeds.** Delaying the basal drive PAST the plateau window
  (`basal_onset` beyond `plateau_onset + plateau_tau`) COLLAPSES the coincidence COMPLETELY (temporal_collapse =
  1 - recall_delayed/recall_overlap = 1.00). A static `phi*phi` has NO time -> a delay could not change it; a
  complete collapse PROVES the coincidence is genuinely temporal, computed in the membrane. **This is the decisive
  result the rate PARTIAL was structurally unable to produce.**
- **SPIKE-BASED:** the coincidence is integer SPIKE COUNTS (mean coincidence spikes 0.28-0.53/channel at high s),
  not a continuous product.
- **AND anchor:** coinc(x,0)=coinc(0,x)=0 spikes on every seed (and_max 0.00) -- no somatic spike unless BOTH
  compartments engage, enforced by the MEMBRANE (theta above the single-input peak), not by `phi(0)=0`.
- byte-identical substrate all seeds; `git diff main -- sim/` ADDITIONS-ONLY in `dendritic_neuron.py` (step() and
  the existing methods untouched -> byte-identical when the new method is never called).

## The one honest bound (READOUT, not the coincidence)

<!--derived-->

The single strict-gate miss is seed-44 / 8x8 / s=0.75: spk 0.38 vs additive 0.12 = +0.26 (just under +0.30). A
T=40->80 window doubling left it IDENTICAL (coincSpk 0.28 unchanged) -> **it is NOT spiking quantization.** At that
exact cell the WHOLE family is weak -- readout-product bind 0.25, rate 0.38, additive 0.12 -- i.e. the recoverable
signal itself caps ~0.38 there for every method, and **the faithful spiking bind BEATS the host-product baseline
(0.38 vs 0.25).** So the miss is a bound of the READOUT at the single hardest operating point (8x8, weakest seed,
intermediate mixing), the same 8x8/seed-44 fragility the rate PARTIAL recorded -- not a limit of the temporal
spiking coincidence, which reproduces or beats the product everywhere.

## Sim edit (guarded, verified safe)

<!--derived-->

`sim/dendritic_neuron.py` additive/default-off: one new method `bac_spiking_coincidence` (+~70 lines);
`__init__`/`step()`/`dendritic_plateau`/`apical_basal_coincidence` untouched, so a caller that never invokes it
observes a byte-identical layer (git-diff verified additions-only, asserted in-runner). `cfg.seed` byte-identical
substrate.

Artifacts: `research/findings/raw/teacher_loop_dendritic_bind_spiking_par_s42.json`,
`research/findings/raw/teacher_loop_dendritic_bind_spiking_par_s43.json`,
`research/findings/raw/teacher_loop_dendritic_bind_spiking_par_s44.json` (+ `.prov.json` sidecars).
SIM_BACKEND=numpy.

## Where the composer stands now

<!--derived-->

The VSA composer's BIND is now biologized to a FAITHFUL spiking mechanism: bundle GO (superposition = neural sum),
bind GO (conjunction = a REAL temporal Larkum-BAC spiking coincidence, this finding -- upgraded from the rate
PARTIAL), arity-3 GO (bounded, disjoint channels). ONE named residual remains: the SHARED-CHANNEL arity sweep
(locate the ~1/sqrt(N) bundling-capacity limit where superposition finally needs binding even for
same-attribute-type composition; Plate/Kanerva VSA capacity is the recorded grounding).

NO-EXTERNAL-NEEDED: the two-compartment BAC / Ca2+ plateau is the substrate's own machinery (D2); Larkum 2013 BAC
firing + Larkum/Zhu/Sakmann 1999 Ca2+ plateau are the recorded biology; the faithfulness witnesses (temporal
collapse, spike-count, AND anchor) are the deliverable.
