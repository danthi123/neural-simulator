# Phase B — the RETINAL escape BUILT ON THE BRIDGE: the whitening front-end is the wall (cm-pool spike-encoding of the common mode fails), and the bridge's own projection loses the weak/diffuse real structure

**Date:** 2026-06-15 (CYCLE 67). **Status:** ⛔ **NEGATIVE / BOUNDARY** (honest, localized). The numpy-validated retinal escape (analog whitening + ON/OFF + high spike budget → +0.327/gen 0.77 on the real corpus) does **NOT** realize on the bridge's point-neuron substrate: the whitening front-end (a common-mode inhibitory pool) cannot spike-encode the population mean (depol block), and even with the whitening removed the bridge's raw spiking projection already loses the weak/diffuse real structure. NO `sim/` edits (the protected set stayed byte-empty; the build is pure brain-region-framework + the documented `set_pathway_weights` API).

## What was built (the genuine retinal ON/OFF dual pathway, brain-region framework, no sim/ edits)
`research/runners/_phaseB_retinal_cortex.py` — a **6-region** bridge realizing the retina's center-surround whitening + ON/OFF split with NEURONS + SYNAPSES (no host whitening; per the BRAIN-BASED-ONLY standard):
- `hub_e` (EXC input layer) → drives `cortex_on`'s g_e (random projection W_on) + both common-mode pools.
- `hub_i` (INH input layer, identical drive) → `cortex_off`'s inhibition (the negated excitation).
- `cm_i` (INH interneuron pool, FS) — pooled hub excitation → fires ~ the **population mean** (common mode) → INHIBITS `cortex_on` with weights **matched to the hub_e→cortex_on row-sums** → `cortex_on` analog drive = `W_on@drive − popmean·rowsum_on` = the **whitened** drive.
- `cm_e` (EXC interneuron pool, FS) → EXCITES `cortex_off` with matched weights → `cortex_off` drive = the **negated** whitened drive.
- `cortex_on` fires on the POSITIVE whitened drive (ON cells); `cortex_off` on the NEGATIVE (OFF cells). The concept code = `concat(cortex_on, cortex_off)` spike counts.

The matched cm→cortex weights are installed post-build via `bridge.set_pathway_weights` (the documented Gabor-preinit API — NOT a `sim/` edit). The dendritic divisive gain is turned OFF for this build (it suppresses the common mode the cm pool must track — see below).

## The pre-build numpy de-risk (all confirmed; the design is sound IN NUMPY)
| probe | result |
|---|---|
| reference `_phaseB_onoff_whitened_derisk` (real) | ON/OFF-whitened **+0.327 / gen 0.766** at g2000 (host +0.442) ✓ reproduced |
| `_phaseB_whitening_locus_probe` (real) | **axis-1 (population-mean / lateral-inhibition) whitening = +0.324 / gen 0.781** ≈ axis-0 (+0.332); beats the TRUE point control (+0.234) by **+0.090**. So the bridge-realizable lateral-inhibition whitening carries the structure. |
| `_phaseB_cmpool_match_probe` (real) | cm→cortex weight = **hub→cortex row-sum (MATCHED) = +0.323** ≈ ideal; RANDOM/UNIFORM cm weights = +0.288/+0.294 (this is exactly why the CYCLE-61 cm-pool failed: it used random/uniform cm weights → subtracted the wrong direction). |
| dual-pathway check (separate W_on/W_off) | **+0.354 / gen 0.781** — using DIFFERENT random projections for ON and OFF (the bridge's architecture) works as well as the shared-W reference. |

So in numpy the design is validated end-to-end: the matched-cm-pool whitening + ON/OFF dual pathway reaches **~+0.33 / gen ~0.78** on real, clearing the +0.30 bar.

## Where it breaks ON THE BRIDGE (the airtight localization)
**Loss 1 — the whitening front-end: the cm pool cannot spike-encode the population mean (depolarization block).**
- The cm pool's INPUT is correct: cm `g_e` tracks popmean at **+0.71** (the analog common-mode signal arrives).
- But the cm pool's OUTPUT inverts it: cm **spike count ANTI-tracks popmean (−0.45 to −0.83)** across every regime swept (RS and FS neuron type; hub→cm weight 4–400; tonic bias 0–700 pA; hub→cm density 0.05–1.0; cm pool size 8–128). The conductance-based drive `g_e·(V−E_e)` at high g_e clamps the membrane near E_e and the Izhikevich neuron enters **depolarization block** → fires LESS exactly when the common mode is LARGEST. There is no operating window where a single-compartment spiking pool linearly encodes the pooled common mode.
- Consequently the neural whitened drive `g_e − g_i` on `cortex_on` carries **+0.03..+0.06** vs the host-whitened reference **+0.23** — the Step-1 front-end does NOT match the host whitening. The whitening is the wall.

**Loss 2 — the bridge projection itself: even with whitening removed, the raw ON/OFF spike code loses the real structure.**
- The bridge POINT control (single population, no whitening, no ON/OFF) on real = **+0.07** (matching the documented CYCLE-65 bridge spike-code +0.075 on real), vs the numpy point control +0.234. The bridge's spiking hub→cortex projection loses ~3× more structure than numpy's lossless matmul BEFORE any whitening — the bridge's own projection loss on the weak/diffuse real structure that the spec flagged as the boundary risk (real cortex g_e was only +0.175).
- Because the raw projection already sits at +0.07, even a *perfect* whitening front-end could not lift the ON/OFF code to +0.30: the structure is already gone before the ON/OFF split.

**Net:** with the imperfect cm whitening active, the ON/OFF code is **+0.04**, actually BELOW the +0.07 point — the cm pool adds noise (mis-directed inhibition) without removing the common mode. Whitening helps in numpy, hurts on the bridge.

## Why the bridge differs from numpy (the mechanism)
Two independent point-neuron limits compound:
1. **The whitening (Mikulasch-Priesemann).** Removing the common mode requires either (a) an analog per-dimension subtraction (a dendritic/pre-spike computation a point neuron can't do) or (b) a spiking pool that linearly encodes the pooled mean. The bridge's (b) fails to **depol block** — the spiking transfer of the pooled common mode is non-monotone. (The CYCLE-63 probe already showed per-neuron centering of the bridge g_e gives +0.001 — the bridge g_e common mode is per-neuron-entangled differently than numpy's clean `W@Xn`, so even host-side axis-0 whitening of the bridge g_e destroys it.)
2. **The projection loss.** The numpy retinal escape sat on a LOSSLESS projection (matmul + Poisson). The bridge's spiking conductance projection loses the weak/diffuse real structure (point +0.07 vs +0.234) before whitening even enters. The retinal escape is real but it was validated on a substrate the bridge does not match.

## Honest synthesis (the whole arc, faithfully)
- The retinal mechanism (analog center-surround whitening + ON/OFF cells + high spike budget) is **real and numpy-validated** on the real corpus (+0.327). Its **faithful point-neuron spiking realization on the bridge is blocked** by (1) the whitening front-end (the cm pool can't spike-encode the common mode — depol block) and (2) the bridge's own projection loss on the weak/diffuse real structure.
- This is consistent with the CYCLE-65 conclusion (the spiking learned cortex fails on the real weak/diffuse corpus) and re-confirms the *mechanism* (the whitening-vs-magnitude tension is point-neuron-hard). The CYCLE-66 retinal escape cracked it **in numpy** but not on the bridge — the numpy de-risk could not see the bridge's spiking transfer + projection loss.
- Per the BRAIN-BASED-ONLY standard a host-computed whitening is a cheat (ruled out — we required the cm pool to do it neurally, and it can't). The faithful spiking realization of the analog whitening still needs the **dendritic (analog, pre-spike, per-dimension) substrate** — the deferred months-scale piece — OR a richer per-error-unit predictive-coding microcircuit (Jang 2024), which is the same conclusion the arc keeps reaching.

## What's banked (durable; NO sim/ edits anywhere)
- `_phaseB_retinal_cortex.py` — the 6-region retinal ON/OFF bridge builder + the matched-cm-pool whitening (set_pathway_weights) + ON/OFF spike readout + the analog-whitening (g_e−g_i) Step-1 instrument + the full GATE (structure / generalizes / beats-point / permuted). On-device spike accumulation (one D→H sync/concept) so the dense-window GPU readout isn't host-sync-bound.
- `_phaseB_whitening_locus_probe.py` — axis-0 vs axis-1 vs both whitening (settled that the bridge-realizable lateral-inhibition whitening carries the structure in numpy).
- `_phaseB_cmpool_match_probe.py` — proved the cm→cortex weight must be the hub→cortex ROW-SUM (explains the prior cm-pool failure).
- The precise localization: cm g_e tracks popmean +0.71 but cm SPIKE anti-tracks −0.5 (depol block); bridge point +0.07 (projection loss); host-whitened ref +0.23.

The honest NEGATIVE IS the deliverable: it maps the retinal escape's bridge boundary precisely — the whitening front-end (spike-encoding the common mode) and the projection loss are the two point-neuron walls, and the faithful spiking retinal cortex needs the dendritic/predictive-coding substrate, not a point-neuron front-end.
