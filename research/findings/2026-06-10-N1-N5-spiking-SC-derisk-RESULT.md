# Spiking superior-colliculus de-risk (N1 + N5) — RESULT: N1 orienting RESOLVES; N5 approach is real but SNR-limited (needs Option C)

**Date:** 2026-06-10
**Type:** Cheapest-first de-risk (CPU, `SIM_BACKEND=numpy`, no GPU, no nav) — the prove-or-kill gate before building the spiking superior colliculus into navigation, per the deep-research GO (`2026-06-10-N1-N5-spiking-superior-colliculus-research.md`).
**Tool:** `research/runners/sc_map_orienting_probe.py` (a read-only research runner; **no `sim/` edits** — verified).

## What this de-risked

The two remaining host computations between sensation and action in the navigation agent:
- **N1** — the superior-colliculus orienting reflex: `sc_orienting_cardinal_from_image` reads the rendered image (pixels) and returns an orienting cardinal (N/E/S/W) in numpy.
- **N5** — the reward *value*: `reward = sign(Δ retinal-eccentricity)` via `sc_salience_offset_from_image`, a host distance-formula (only the `reward_us` *delivery* of that scalar is spiking).

The deep-research recommendation (Option A+C) is that **one spiking retinotopic superior-colliculus map** closes both — N1 is *where* the activity bump is (orienting), N5 is *how the bump moves toward the foveal centre* (approach). This de-risk built that minimal map on the real `SimulationBridge` and tested behavioural equivalence against the **real host functions**, plus a decisive lesion control.

## Architecture (all runner/probe-side region+pathway wiring; ZERO sim/ edits)

`egocentric retina image` (the world rendered from the agent's eye — a legitimate environment operation, channel-1 of the BRAIN-BASED-ONLY bar) → `retina` (2×32×32 spiking photoreceptors) → **retinotopic** `sc_map` (16×16 spiking sheet) with a **Mexican-hat** surround (`sc_map↔sc_fs` inhibition + short-range recurrent excitation) → a single activity bump at the goal's egocentric retinal site. Two read-outs **by neuron firing**: N1 = weighted-quadrant pooling `sc_map → cortex_{N,E,S,W}` (the winning pool = the orienting cardinal); N5 = foveal-centre pooling `sc_map → approach` (firing rises as the bump nears the centre = the goal gets closer).

## Result (seed 42, CPU)

| Falsifier | INTACT | LESION (scrambled retinotopy) | Verdict |
|---|---|---|---|
| **F1 — N1 orienting** (vs `sc_orienting_cardinal_from_image`, by firing) | **8/8** | **2/8** (collapses to chance) | **RESOLVES** |
| **F2 — N5 approach** (vs host `sign(Δ ecc)`, by `approach`-pool firing) | **3–6/8** (window-dependent) | **2/8** (collapses) | **real but SNR-limited** |

- **N1 RESOLVES decisively.** A spiking 2D retinotopic SC map with Mexican-hat winner-take-all reproduces the host orienting cardinal **8/8 by neuron firing, image-only**, and the **scrambled-retinotopy lesion breaks it (2/8 ≈ chance)** — the signal is genuinely retinotopic, carried by synaptic transmission, not a host argmax. This is a GO to build the spiking superior-colliculus orienting into the nav agent, closing the N1 host cheat.
- **N5 approach is a real, retinotopic signal but SNR-limited under the static foveal-position read.** F2 rises monotonically with the integration window (2/8 @ 35 readout steps → 5/8 @ 100 → 6/8 @ 200; run-variable 3–6/8) and the lesion collapses it to 2/8 — so the approach signal **is** carried by the SC bump position (verified directly: a closer goal puts the bump nearer the foveal centre), but reading a one-cell eccentricity *shift* as a graded firing difference from a sparse spiking bump is near the noise floor. The robust read-out is the research's **Option C** — a slow-channel **temporal-difference** of the rostral-ward bump motion (`nmda_slow` lagged copy + a subtractive comparison), which integrates continuously rather than reading a static position. The machinery (`exc_receptor="nmda_slow"`, `receptor="gaba_b"`) is already merged and runner-enabled (verified in `sim/regions.py`), so Option C is runner-side wiring.

## Bugs found + fixed during the de-risk (load-bearing for the build)

1. **Inhibitory routing requires a non-empty wiring plan.** Declaring an inhibitory pathway with `density=0` and installing it via `set_pathway_weights(add_missing=True)` leaves `inject_explicit_wiring` with "no synapses in plan" → the per-neuron **inhibitory trait mask is never set** → the inhibitory pool's synapses act **excitatory** (they drove the whole map, not a bump). **Fix:** declare the `sc_map↔sc_fs` Mexican-hat pathways with non-zero `density` (the framework builds them and marks `sc_fs` inhibitory); the retinotopic structure is then layered on with `set_pathway_weights`. This is the key gotcha for any future explicit-wired inhibitory region.
2. **Background OU noise drowns a sparse sensory map.** The default `ou_std_current_pA=100` made all 256 SC sites fire uniformly. A sparse retinotopic map needs low spontaneous rate (`ou_std≈6`); the high default is for cortical-circuit realism, not a sensory sheet.
3. **Per-trial hard reset is required** (restore resting v/u, zero conductances/firing/refractory) — otherwise a bump carries over between presentations and the read-out becomes order-dependent (the same clean-reset discipline the nav critic needed).

## Recommendation (the build)

1. **Build the spiking SC orienting (N1) into the nav agent now** — it RESOLVES (8/8, lesion-confirmed). This replaces `sc_orienting_cardinal_from_image` with a spiking superior colliculus, closing the N1 host cheat. Verify with the 6-seed GPU nav A/B (no regression vs the host scaffold) per the standing bar.
2. **Refine N5 with Option C** (the slow-channel temporal-difference read-out) before relying on the spiking approach reward end-to-end; in the interim N5 remains the coordinate-free perceived-approach scaffold (already in place). The SC bump carries the approach signal; only the read-out needs the continuous slow integration.

## Honest scope

This is a single-seed CPU de-risk on hand-set positions (the prove-or-kill gate), not the nav integration. N1's decisiveness (8/8 + lesion-confirmed) is strong; N5's SNR limit is the honest residual that maps the substrate — the static read is marginal, Option C is the principled fix. Per the BRAIN-BASED-ONLY standard, the spiking SC is a genuine conversion (image-only afferent, orienting + approach by neuron firing, lesion-confirmed retinotopic — none of the host argmax/distance), with one residual idealization noted in the research (the `sc_map → cortex_X` topographic read-out is a fixed, genetically-specified-style projection, like the innate V1 Gabors, not a learned map).
