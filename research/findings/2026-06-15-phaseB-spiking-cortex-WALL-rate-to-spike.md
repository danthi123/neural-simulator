# Phase B — the spiking learned-cortex HARD GATE returns an airtight WALL: the L1 rate recipe does NOT realize on the point-neuron spiking bridge (the rate→spike Mikulasch-Priesemann wall)

**Date:** 2026-06-15 (CYCLES 59–62). **Status:** ⛔ **NEGATIVE / WALL** — honest, multiply-confirmed, decision-relevant. This is the deliverable the Phase-B HARD GATE (proposal risk #2) was built to produce: it **maps the rate→spike wall precisely**. NO `sim/` edits anywhere in the whole gate (the protected set stayed byte-empty).

## The one-line result
The L1 learned-cortex recipe — validated comprehensively at the **rate** level (`2026-06-14-L1-learned-cortex-fair-test-GO.md`, +0.545, 5 axes GO) — **does not recover the category structure on the bridge's point-neuron spiking substrate.** Across 6 distinct attempts/probes the trained cortex code is ≈ 0 / slightly negative (best ≈ −0.04), no better than a random projection, while the rate ceiling is +0.89. The structure is lost in the **rate→spike encoding**, which the point-neuron substrate cannot undo.

## What was tried (the gate did its job — cheap-first, zero `sim/` edits)
| # | attempt | result | what it ruled out |
|---|---|---|---|
| 1 | naive hub-drive STDP | cortex silent | **clock bug** — `bridge._run_one_simulation_step()` doesn't advance `current_time_ms` → every spike t=0 → `delta_t≡0` → STDP a no-op (NOT "net-depression"; the "824→169 decay" was the buggy structural plasticity). Fixed runner-level (`_step_with_time`). |
| 2 | C1a: WTA + FAST adaptive-θ homeostasis + non-specific co-fire (Diehl-Cook) | cortex FIRES, structure −0.07 | the silence is curable; the WTA/competition does not recover the structure (the project's own Phase-A finding: the lateral isn't the key). |
| 3 | centering: feedforward subtractive-inhibition cm-pool + synaptic scaling + stronger dendritic gain | WALL, best −0.04 | a single inhibitory pool delivers only **rank-1 (uniform)** inhibition; the common mode is **per-cortex-neuron-varying** (random connectivity) → uniform inhibition removes signal+common-mode together (no Goldilocks). |
| 4 | per-postsynaptic-neuron centering (host-side C1b instrument) | does not recover (−0.15→−0.09) | the **exact L1 op** (`x − mean`) applied to the bridge g_e does not recover it → the wall is **below** the centering rule. |
| 5 | hub-encoding regime (drive strength) | hubs CAN encode (+0.33 at strong drive) but it was a drive artifact | at drive_scale 12 the hubs fired ~0.15 spk/hub (Poisson noise) → the prior 36-cell grid (ds{12,20}) **under-fired**; at ds40, hub code +0.33 (centered) — so the input layer is NOT the fundamental wall. |
| 6 | strong-drive cortex gate (the regime the grid never tried) | WALL, −0.04..−0.07, ≈ random-proj | even with the hubs encoding (+0.33) + centering + strong drive, the **cortex** code does not recover the structure. |

## The precise localization (airtight)
- The drive carries the structure: **log-input cosine +0.891** (centered +0.936 — the L1 op works on the input).
- The **hub spike-rate code** can recover it given enough firing: −0.06 at ds12 (~0.15 spk/hub, Poisson noise) → **+0.33 (centered) at ds40** (~3 spk/hub). So the input spiking is not fundamentally lossy — it just needs enough spikes.
- The **cortex** code (after the hub→cortex projection) loses it: g_e (analog, pre-threshold) cosine ≈ **−0.06 ≈ the spike-code −0.07** ⇒ the spiking threshold is **not** the destroyer; **the common mode dominates the analog cortex drive**, and it is **per-cortex-neuron-varying**, so no point-neuron mechanism (feedforward inhibition, per-neuron centering, WTA, synaptic scaling, the dendritic divisive gain, strong drive) removes it without also removing the signal.

## Root cause — the rate→spike Mikulasch-Priesemann wall
L1's load-bearing op is **common-mode removal (centering / whitening)** — a **per-input-dimension subtraction on a full-precision analog code before the projection**. On the bridge, the projection happens through spikes + conductances; the common mode enters the cortex drive **per-neuron-varyingly** and a point neuron has only **rank-1 (scalar) or threshold-based** tools to remove it — it cannot do the **per-dimension analog whitening**. This is exactly the documented **Mikulasch-Priesemann point-neuron limit** the project has hit 5+ times: *decorrelation/whitening is an analog / pre-spike / dendritic computation a point neuron fundamentally cannot do.* The HARD GATE confirmed it with the bridge in the loop — which the rate-level de-risk could not see.

## What this means (the honest synthesis)
- The L1 mechanism is **real and rate-validated** (the learned cortex reaches the host ceiling at the rate level). Its **faithful spiking, point-neuron realization is blocked** by the Mikulasch-Priesemann wall.
- The L1 GO's conclusion that *"the dendritic D2 build is OFF the critical path"* was a **rate-level** conclusion. The bridge realization **re-opens it**: the **spiking** realization of the centering needs **dendritic (analog, pre-spike, per-dimension) computation** — the deferred, months-scale dendritic-substrate piece.
- Per the BRAIN-BASED-ONLY standard, a host-computed centering is a cheat (ruled out). A simple guarded `sim/` edit (post-triggered STDP, a per-neuron subtractive primitive) is unlikely to suffice — probe #4 shows the wall is **below** the centering rule (the per-dimension analog whitening, not just the LTP/LTD shape).

## The strategic fork (owner's call — the next step is the gated months-scale piece)
- **(A) The dendritic substrate (months-scale).** The only path to a faithful, brain-based, *spiking* learned cortex that does the analog whitening → generalizes. The deep artificial-life / biology-translatable frontier; re-confirmed as necessary by this bridge wall.
- **(B) Accept the point-neuron limit.** Ship the **flat 2,048-concept curated cortex** (the conversational product, delivered) + bank the L1 rate mechanism as validated-but-not-point-neuron-spiking-realizable.
- **(C) A guarded `sim/` edit.** Likely insufficient on its own (the wall is the analog whitening, not the rule shape); blurs into (A).

## What's banked (durable, all pushed both remotes)
- The corrected bridge-STDP **clock fix** (runner-level `_step_with_time`; no `sim/` edit) — a real, reusable correction (`bridge._run_one_simulation_step` must be paired with a `current_time_ms` advance for STDP).
- The **C1a competitive machinery** (WTA via `exc_fraction`+internal density, fast-θ homeostasis override, non-specific co-fire, the cm-pool, synaptic-scaling flag) — all opt-in/additive in `research/runners/spiking_sm_cortex.py`, default-off byte-preserving; reusable for any future bridge competitive-learning work.
- The **precise localization probes** (`_phaseB_c1b_derisk_perneuron_centering.py`, `_phaseB_hub_encoding_regime.py`, `_phaseB_strong_drive_gate.py`) + the deep-research (`2026-06-15-bridge-competitive-stdp-deep-research.md`) + the subagent's WALL write-up (`2026-06-15-phaseB-task3-centering-RESULT.md`).
- A flagged real `sim/` bug (structural-plasticity not resizing `cp_plasticity_rate_gain` → IndexError on gated pathways).
- The protected set stayed **byte-empty** — the entire Phase-B gate was zero-`sim/`-edit.

The honest NEGATIVE IS the deliverable: it maps the rate→spike wall precisely and tells the owner the spiking learned cortex requires the dendritic substrate, saving a months-scale build from the wrong (point-neuron) premise.

---

## ⚠️ REFINEMENT (same night, CYCLE 63) — the wall is the SPIKE-COUNT READOUT of a common-mode-buried weak signal; I over-claimed "the projection needs dendrites." It is a BOUNDARY, not a clean WALL.

A follow-on deep-research (`2026-06-15-spiking-whitening-cheapest-mechanism-research.md`) flagged that the 6 probes centered at/after the **cortex**, whereas L1 centers the **input per-hub before the projection**, and argued the months-scale substrate is likely unnecessary. Four more free probes (no `sim/` edits) localized it precisely and **partly overturn, partly confirm** the WALL:

| measurement (clean bridge, strong drive ds40, untrained random W) | result |
|---|---|
| bridge cortex **g_e (analog conductance)** cosine | **+0.45 to +0.57** — the projection PRESERVES the structure |
| bridge cortex **spike-count** code cosine | **≈ 0** (−0.04..+0.05) across ds{20,40,80,120} × window{150,300}, 60–136 spikes/concept |
| numpy (rate) projection of the hub codes | +0.34 (input-centering ≈ output-centering +0.338 ≈ +0.341 — the **locus does NOT matter**) |
| g_e **per-neuron-centered** | +0.001 (dense) — the g_e structure is **common-mode-correlated / weak**, centering removes it |

**The corrected localization:**
1. The earlier "g_e −0.06" was the **weak-drive (ds12) regime** (hubs under-firing, ~0.15 spk/hub). At strong drive the **analog path is fine** — the hub→cortex projection preserves the structure (g_e +0.45). So the projection does **not** need dendrites — I over-claimed that.
2. But the category signal is a **weak perturbation on a large common mode** (200 common hubs vs 12/category); it sits in the g_e weakly (+0.45 uncentered, ~0 centered), and the **spike-count readout robustly loses it** — the spiking threshold saturates on the common mode, burying the weak category signal. This is the common-mode problem manifesting **at the spike readout**, not the projection.
3. The research's input-vs-output **locus** reframe did **not** rescue it on the bridge (numpy: input ≈ output ≈ +0.34; bridge spike readout: both ≈ 0). So the fix is **not** simply "center at the input."

**Honest status = BOUNDARY** (not the clean WALL above, not "just engineering"): the structure lives in the analog path; transmitting the **common-mode-buried weak category signal through a spike-count code** is the genuine open problem, and removing the common mode cleanly is the point-neuron-hard whitening (the Mikulasch-Priesemann theme — my original WALL was directionally right about the *mechanism*, wrong that it's the projection). The **untested** cheaper-than-dendrites candidate is **predictive coding with per-error-unit interneurons** (Jang et al. 2024, PMC11045951 — demonstrated in single-compartment AdEx POINT neurons, ρ>0.8: a per-dimension prediction-subtraction microcircuit, richer than the rank-1 pool that failed). The FHRR phase-coding escape does **not** apply (different common mode).

**Refined fork for the owner:** (A′) a **predictive-coding microcircuit** (per-dimension common-mode prediction+subtraction in point neurons — Jang 2024; a medium build, cheaper than dendrites, untested here); (B) ship the flat curated cortex (delivered) + bank L1; (C) the minimal single-extra-compartment dendrite (now looks like the *fallback*, not the lead). New localization probes: `_phaseB_input_centering_derisk.py`, `_phaseB_projection_isolation.py`, `_phaseB_cortex_readout.py`.

**Final localization (CYCLE 63, the last probe `_phaseB_homeo_off_readout.py`):** the spike-readout loss is **NOT homeostasis equalization** — with homeostasis OFF the cortex spike code is still ≈ 0 (−0.09..+0.01) while g_e stays +0.40..+0.57. So the loss is robust across drive × window × homeostasis × density × gain (≈ 11 probes total). **The honest, well-localized status:** the category structure lives in the cortex analog g_e (+0.45) but does **not survive the spike-count code**, because the category signal is a *weak perturbation on a large common mode* and removing that common mode **before** the spiking threshold is the point-neuron-hard analog whitening (the Mikulasch-Priesemann theme — my CYCLE-62 instinct about the *mechanism* was right; my claim that it's the *projection*/needs-dendrites was wrong — it is the **spike-count readout of an un-whitened weak signal**). Faithful spike-based transmission needs the common mode removed pre-threshold (whitening) or a richer code/microcircuit (predictive coding, Jang 2024). The analog g_e proves the structure is recoverable *in principle*; the spike transmission is the genuine open boundary. **This is owner-decision territory** (medium build) — the solo cheap-first probing is exhausted.
