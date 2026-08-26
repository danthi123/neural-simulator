---
type: finding
status: live
date: 2026-08-26
mechanism: superior-colliculus
---

# Spiking superior colliculus — reflexive visual ORIENTING packaged as a PRODUCTION visuomotor organ + embodied consumer — **GO** (6-seed), flag default-OFF pending the parent flip

**2026-08-26 (autonomous production wire-in).** The N1 spiking superior colliculus (`2026-06-10-N1-spiking-superior-colliculus-CLOSED.md`, 6-seed nav A/B SC/host 0.883 <!--derived-->, scramble regresses 2.4x) existed only INSIDE the nav runner (`g11_bg_runner.py --enable-spiking-sc`). This packages it as a **standalone, process-shared production organ** (`sc_orienting_production_organ.py`) and builds the owner's canonical **cheap visuomotor consumer**: a tiny embodied world where a salient target appears in the retinal field of view, the spiking SC emits an orienting cardinal BY FIRING, and the **body foveates the target**. The spiking SC (not the host `--sc-orienting-reflex` scaffold) drives orienting. **Reuse-by-import; NO `sim/` edit.** Flag `BRAIN_SPIKING_SC_ORIENT` default **OFF** for now — the parent flips it after the pool soak.

## Mechanism (all reuse-by-import of the de-risked machinery)

The organ builds the minimal region scaffold `build_bg_brain_regions` builds for `enable_spiking_sc` (`sc_retina` 2·32·32, retinotopic `sc_map` 16×16, Mexican-hat `sc_fs`, read-out pools `cortex_{N,E,S,W}` + the framework-built surround so `sc_fs` is inhibitory), then calls **`install_spiking_sc_wiring` verbatim** — the SAME production wiring the nav runner uses (retina(ON)→`sc_map` retinotopic 2×2 pooling + short-range recurrent + the signed-ramp `sc_map`→`cortex_{N,E,S,W}` read-out). `orient(agent, goal)` renders the world from the agent's eye (`render_egocentric_goal`), drives `sc_retina` (`image_to_retina_drive`, 2500 pA), steps the bridge, lets the `sc_map` Mexican-hat WTA form a single saliency bump at the blob's retinotopic site, and reads the orienting cardinal off the **winning `cortex_X` pool BY FIRING** (`cp_firing_states`). The consumer (`_sc_orienting_production_organ_verify.py`) then moves the agent one cell in that cardinal — a real body output.

**BRAIN-BASED:** the cardinal is a `cp_firing_states[cortex_X]` read off a topographic spiking read-out of a retinotopic WTA sheet — no host argmax/coordinate enters the decision. Host boundaries are the two the bar permits: the ENVIRONMENT render (world→retinal image) and the BODY (move one cell by which pool fired). The `sc_map` firing-peak SITE is reported as a saliency instrument, not the decision.

## Result — organ verify (CPU, numpy) — **GO**

| check | INTACT (spiking SC) | LESION (`SC_SCRAMBLE`) | host reflex (OFF path) |
|---|---|---|---|
| correct-cardinal battery (12 cases, all bearings, dist 2-3) | **12/12 = 1.000** | **0/12 = 0.000** (chance 0.250) | 12/12 (the scaffold) |
| embodied foveation loop (8 bearings, reach = Chebyshev≤1) | reach **1.00**, ccr 0.88, path-eff 1.25 | reach **0.00**, ccr 0.06 | — |
| load-bearing identity (same (8,8)→(11,8) afferent) | cardinal **E**, bump 177 sp | cardinal **W**, bump 154 sp | — |

The lesion's `sc_map` bump strength is **unchanged** (154 vs 177 sp, ratio 0.87) — the image-only afferent is identical; only the `sc_retina`→`sc_map` retinotopy is permuted — yet the orienting cardinal **decouples from the target location**. This is the decisive proof that the **retinotopic spiking sheet carries the orienting target**, not a re-hidden host read (the de-risk's 2.4x-regression anti-cheat, at the organ level).

## Flip gate — 6-seed soak (`_sc_orienting_flip_soak.py`) — **GO**

`research/findings/raw/sc_orienting/flip_soak.json` (seeds 42-47):

| metric | result | gate |
|---|---|---|
| INTACT correct-cardinal | min **1.000**, mean 1.000 | min ≥ 0.80 |
| LESION correct-cardinal | max **0.250**, mean 0.139 <!--derived--> | max ≤ 0.45 (chance 0.25) |
| INTACT embodied reach | min **1.000**, mean 1.000 | min ≥ 0.80 |
| LESION embodied reach | max **0.125**, mean 0.083 <!--derived--> | max ≤ 0.50 |

All four gates pass at every seed → **FLIP-GATE: GO**. The lesion arm collapses to chance at every seed (load-bearing at each seed, not a single-seed artifact).

## Disciplines

- **Additive + default-OFF:** `BRAIN_SPIKING_SC_ORIENT` defaults OFF. The organ + consumer are NEW files imported by NO existing production path, so flag-OFF is a no-op by construction — there is no existing surface to change. **`off_byte_identical` = N-A for chat** (the faculty is EMBODIED, no chat coupling); the OFF path is the host reflex `sc_orienting_cardinal_from_image` (the scaffold the spiking SC replaces, reported as the comparator: it matches the SC 12/12).
- **Load-bearing, not hollow:** proven via the faculty's OWN de-risk oracle (`SC_SCRAMBLE` permutes the retinotopy). Vary the coupling (intact→scrambled) and the orienting output demonstrably changes (1.00→0.14 correct-cardinal, 1.00→0.08 reach) while the afferent is held fixed; the change vanishes only because the retinotopic coupling was lesioned. Plasticity is disabled (`enable_stdp=False` etc.) and the permutation is structural, so the lesion holds throughout measurement.
- **`GO` scope:** the ORGAN + CONSUMER + 6-seed SOAK pass their own gates. This is a wire-in **GO**, NOT `closed` (per `docs/TERMS.md`, "closed" requires integrated + default-ON + scaffold-retired). The default-ON flip is the parent's call after the pool soak.

## Honest residuals (declared, the named next rungs)

- **RENDER = LINEAR (foveal), FOV ±3 cells.** The organ uses the de-risked LINEAR egocentric render (sharp fovea; the render the CLOSED 6-seed + the probe's 8/8 used). Its retinal FOV is ±3 cells; a target beyond that is off-retina (biologically, un-orientable — you cannot orient to what is not on your retina). The embodied task is therefore "a salient target APPEARS within the FOV → foveate it", the SC orienting reflex's actual function. The `log_polar=True` full-hemifield map (the `2026-06-22` #6 SURPASS) covers far targets but has a larger foveal-null radius (~3-4 cells: a foveated target has ~0 orienting error, so the ramp read-out is ~0 near the sc-centre) — a documented tradeoff, not a substrate limit. Combining a foveal-linear + peripheral-log map into one afferent is a named next rung.
- **READ-OUT = the signed-ramp (default) `sc_map`→`cortex_X` pooling**, not the population-vector cosine decode (`install_spiking_sc_wiring popvector=True`, the #6 build). The ramp is the deployed default the CLOSED finding validated; the popvector decode + divisive-norm is available and is the next rung for eccentric-target margin-SNR.
- **The `sc_map`→`cortex_X` read-out is a FIXED (genetically-specified-style, chemoaffinity/ephrin-Eph) topographic projection**, not a learned map — the same declared status as the innate V1 Gabor RFs (N7). No cognitive quantity is host-computed; it is innate structure the bar permits.
- **CO-RESIDENT on its own SC bridge**, alongside the recall composer — rides on the one-brain merge, as the affect/curiosity/etc. organs do.

## Files

- `research/runners/sc_orienting_production_organ.py` — the production organ (reuse-by-import; NO `sim/` edit).
- `research/runners/_sc_orienting_production_organ_verify.py` — the embodied visuomotor consumer + verify (GO).
- `research/runners/_sc_orienting_flip_soak.py` — the 6-seed flip gate (GO); artifact `research/findings/raw/sc_orienting/flip_soak.json`.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._sc_orienting_production_organ_verify
SIM_BACKEND=numpy python -m research.runners._sc_orienting_flip_soak \
    --seeds 42,43,44,45,46,47 --out research/findings/raw/sc_orienting/flip_soak.json
```
