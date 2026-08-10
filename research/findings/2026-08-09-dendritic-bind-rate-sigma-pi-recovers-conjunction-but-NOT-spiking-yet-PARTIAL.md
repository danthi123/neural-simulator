---
title: "Dendritic-bind: a more-biological RATE sigma-pi (plateau + ON/OFF push-pull) recovers the conjunction, but the adversarial verify CAUGHT that it is still a host multiply — the FAITHFUL spiking-dendrite bind is open"
date: 2026-08-09
type: finding
status: contributing
lane: composer
seeds: [42, 43, 44]
---

# Dendritic-bind is a functional rate dendritic-AND, NOT yet a spiking-membrane coincidence (verify-caught)

## Claim (honest — the adversarial verify overturned the build's "on spikes" headline)

<!--derived-->

Goal: move the composer BIND's conjunction (currently a host `*` on two per-primitive readouts, `2bcf9d13`) INTO a
real two-compartment spiking dendrite so a genuine apical×basal membrane coincidence computes it. **Result: a
more-biological RATE sigma-pi recovers the conjunction, but it is NOT the faithful spiking version — the adversarial
verify CAUGHT the overclaim.** `sim/dendritic_neuron.py:apical_basal_coincidence` returns `{soma: gb*ga}` with
`gb=phi(basal)`, `ga=phi(apical)` and `phi` a saturating plateau (Michaelis-Menten / finite NMDA-Ca); with
`W_basal=B_apical=I` this reduces to `phi(fB)*phi(fA)` — the **same factor readouts** the baseline multiplies with a
plain host `*`. The genuine two-compartment SPIKING dynamics (`DendriticLayer.step()`, membrane leak integration,
plateau *dynamics*) are **bypassed** — no spikes, no time-integration; a static method computes the product. So the
conjunction arithmetic is still a host `gb*ga`. **Verdict: PARTIAL — functional dendritic-AND real, FAITHFUL spiking
coincidence NOT achieved.**

## What IS real (the functional recovery, verify-reproduced)

<!--derived-->

The rate sigma-pi (plateau + ON/OFF push-pull, so it handles signed factors) RECOVERS zero-shot held-out composition
at high mixing, matching/exceeding the readout-product bind at 3/4 high-s grid-points (3-seed):
7×7 s=1.0 dend **1.00** [1,1,1] vs product 0.81 vs additive 0.00 (chance 0.02); 7×7 s=0.75 dend 0.86 vs product 0.57;
8×8 s=0.75 dend 0.71 vs product 0.50; 8×8 s=1.0 dend 0.67 vs product 0.75 (the one point below product, driven by the
weak seed 44). It IS more biological than a plain `*`: AND-anchor holds (`coinc(x,0)=coinc(0,y)=0` — no output unless
BOTH compartments engage), genuinely nonlinear (nl-witness ~0.42, the soma is not a linear multiple of the exact
product), plateau-saturating. Raws: `research/findings/raw/teacher_loop_dendritic_bind_AGG.json` (+ s42/s43/s44).

## The honest residual + why the verify was right to refute

<!--derived-->

The de-risk's HARD RULE was "apical×basal nonlinearity IN SPIKES, not a host `*`". The build met the *nonlinearity*
(plateau) and the *AND* (anchor) but NOT the *in-spikes* part — the coincidence is a rate-model product, `step()`'s
membrane dynamics untouched. Under the brain-based-only standard this is a **more-biological host `*`**, a stepping
stone, not the faithful mechanism. **Honest negatives:** strict all-seed GO = 1/3 (7×7 3/3, but 8×8 seed 44 weak:
0.375@s=0.75, 0.25@s=1.0 — the plateau saturation distorts the rank-1 factor completion on the hardest seed, where
the readout-product ALSO degrades); the plateau costs some precision at intermediate mixing. 3-seed, numpy; 6-seed
not run.

## Sim edit (guarded, verified safe)

`sim/dendritic_neuron.py` +49/−0 — ADDITIVE, DEFAULT-OFF: two new methods (`dendritic_plateau`,
`apical_basal_coincidence`) only; `__init__`/`step()` untouched, so a caller that never invokes the coincidence sees
a byte-identical layer (`sim_additions_only=True`, git-diff verified). cfg.seed byte-identical substrate.

## Next lever (the FAITHFUL spiking-dendrite bind — the real deep residual)

Route the two primitive drives through `DendriticLayer.step()` so the coincidence is computed by REAL membrane
dynamics over time (basal depolarization must reach the soma while an apical PLATEAU is active — a temporal
coincidence in spikes), not a static `phi(fB)*phi(fA)`. THAT is the genuine "conjunction in spikes"; this de-risk
shows the rate approximation of it recovers the composition, which de-risks the target but does not reach it.

NO-EXTERNAL-NEEDED: dendritic plateau / apical-basal coincidence is the substrate's own two-compartment machinery
(D2); the honesty-boundary catch (rate `*` ≠ spiking coincidence) is the deliverable.
