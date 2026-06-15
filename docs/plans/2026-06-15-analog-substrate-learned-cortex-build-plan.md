# Analog-substrate learned cortex — build plan (the brain-based escape for the rate→spike wall)

> **Status:** the concrete path forward after the Phase-B arc decisively mapped the rate→spike wall
> (`research/findings/2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md`, ~22 probes + a subagent build).
> Months-scale; owner-strategic. The honest NEGATIVE (point-spiking can't realize L1 on real data) is the
> deliverable that motivates this plan.

## The wall (decisively established)
The L1 learned cortex works at full precision (numpy: +0.48 on the real corpus, generalizes). The
**point-neuron rate-coded spiking** realization FAILS on the weak/diffuse real corpus (+0.06–0.14 vs host
+0.44) due to **compounding spiking losses at every stage**, each one alone insufficient to fix:
1. **Whitening** (common-mode removal — the load-bearing op): a spiking inhibitory pool **cannot linearly
   track the population mean** — depolarization block makes its spike output *anti-track* the mean (cm g_e
   tracks +0.71, but cm SPIKES −0.57; neural whitening +0.14 vs host +0.25).
2. **Projection**: the hub→cortex spiking projection loses the diffuse structure (real cortex g_e only
   +0.175 vs the input +0.50; numpy random projection keeps +0.23).
3. **Readout**: the spike-COUNT code quantizes the weak signal (helped by a huge spike budget — +0.33 at
   ~2000 spikes/unit — but saturates below the full-precision ceiling).
Every cheap+medium escape was exhausted: rate, dense firing, centering (cm-pool/synaptic-scaling/gain/
per-neuron), phase coding (phasors don't preserve continuous similarity — NEGATIVE), ON/OFF retinal (numpy
+0.35 but the *spiking* whitening fails), linear-cm. **The common thread = the Mikulasch-Priesemann limit:
whitening/decorrelation is an analog / pre-spike computation a point neuron cannot do.** The escape is
**faithful analog/graded computation at each stage** — the retina + dendritic substrate.

## The plan (phased; each phase = cheap-first numpy de-risk → bridge build → gate; a NEGATIVE at any phase is the deliverable)

### Phase 1 — graded inhibition (the retina's horizontal cells). **BUILT + verified byte-identical (CYCLE 68); but the cm-POOL architecture it serves whitens on the WRONG AXIS → superseded by Phase 2.**
The graded synaptic transmission mode is SHIPPED on main (both remotes): a default-OFF, byte-reviewed
`RegionPathway(graded=True)` → the per-step inhibitory current uses the source's continuous membrane
`a_cont = clip((v−rest)/scale,0,1)`, not `cp_firing_states` (commits `dec311f4` + `cbcc8f85`; true pre/post
A/B byte-identity proof; 6/6 graded tests pass). It is a real, validated brain mechanism (Kandel retina,
catalog E.05). **BUT** the cm-pool gate on the REAL corpus is NEGATIVE (CYCLE 69): the graded-whitened ON/OFF
cortex code = +0.051 (below the no-whitening control +0.065). **Airtight diagnosis (numpy, real):** the
common-mode POOL does **per-CONCEPT (axis-1)** whitening (it fires ~ each concept's mean over hubs), but the
structure needs **per-FEATURE (axis-0)** centering (subtract each hub's mean across concepts). Even a *perfect*
axis-1 pool caps at +0.255 < the +0.30 bar (`_phaseB_whitening_axis_probe.py`). The graded edit is not wasted
— it is the transmission mode for the corrected Phase-2 mechanism — but the cm-POOL itself is the wrong
architecture for this whitening.

### Phase 2 — per-hub ADAPTATION (axis-0 per-feature centering), NOT a common-mode pool. **Numpy mechanism = 6-seed GO; the bridge realization is the next de-risk.**
The fix for the wrong whitening axis: each hub subtracts its **own** slow running mean (intrinsic spike-
frequency adaptation / slow AHP / synaptic depression = a per-neuron high-pass = the Mikulasch-Priesemann
per-neuron predictive-coding form of whitening) — this is **per-FEATURE (axis-0)** centering, the L1 load-
bearing op, and it's MORE biological than a pool (every real neuron adapts to its own mean).
`_phaseB_perhub_adaptation_derisk.py` (**6 seeds**, real, host +0.442): per-hub adaptation recovers axis-0 =
**+0.311** at a slow rate (α=0.02–0.05; 96–108% of the batch axis-0 ideal; clears +0.30; gen ~0.70), vs the
cm-pool axis-1 +0.246. The slow time-constant is load-bearing (α=0.5 → +0.17 — the adaptation must span many
concept presentations, not one). **The bridge-realization de-risk (next):** realize a *slow per-hub mean* on
the point-neuron bridge — cheap-first via the existing homeostasis (per-neuron rate EMA + threshold) on the
hubs in a streaming protocol; if that doesn't recover axis-0 in spikes, a per-hub slow feedback-inhibition
shadow (1-to-1, reusing the Phase-1 graded mode) or a guarded slow-adaptation `sim/` primitive. **Gate:** the
bridge per-hub-adapted cortex code recovers axis-0 (≈ the numpy +0.31) on real, beats the cm-pool + the point
control + permuted-clean. **Risk:** the slow per-hub mean is itself the Mikulasch-Priesemann slow-analog-
integration challenge on a point substrate (the cm-pool's bridge realization already lost half: host axis-1
+0.246 → neural +0.138), so the spiking realization could lose — a NEGATIVE there maps the wall deeper and is
the deliverable.

### Phase 3 — the learned analog cortex + the real-corpus gate.
With Phases 1–2 (graded whitening + graded/high-precision projection), add the bounded-Hebbian learning and
read the cortex code. **Gate (the decisive GO):** on the real 64-concept corpus, the learned analog cortex
reaches Pearson(cos, S_true) ≥ 0.70×host (≈ +0.31) AND generalizes (held-out ≈ 0.77, the numpy retinal
level) AND beats the point-spiking control + permuted-clean. Then scale toward 320/2048 concepts and plug
into the dual/CLS conversational pipeline (the no-confab moat preserved).

## Honest scope + risks
- This is **months-scale** and graded transmission is a **core `sim/` edit** (the transmission path) — the
  highest-risk change of the project; it must be default-off + byte-reviewed + with a regression test
  (byte-identical when off). Do it with fresh focus, not at the tail of a long arc.
- Even with graded whitening + projection, the real category structure is *moderate* (host +0.44) → the
  learned analog cortex would generalize *moderately* ("cat ~ dog"), not perfectly. Real, but set expectations.
- A NEGATIVE at any phase (graded transmission still loses it) maps the wall further and is the deliverable.

## What this supersedes / parallels
- Supersedes the "point-spiking is impossible" framing (it's the *rate-coded point* substrate that's the
  wall; graded/analog transmission is the brain-canonical escape, not necessarily full multi-compartment
  dendrites — graded inhibition is a smaller first step).
- The flat 2,048-concept curated cortex stays the **shipped conversational product** in parallel; the L1
  rate mechanism (+0.48, generalizes) is the validated target this analog substrate must spike-realize.
- Banked from the arc (all reusable, NO `sim/` edits made): the bridge-STDP clock fix, the dense-firing
  regime + `enable_homeostasis` kwarg, the C1a competitive machinery, the 6-region ON/OFF retinal bridge
  (`_phaseB_retinal_cortex.py`), ~22 localization probes, 3 deep-researches, a flagged `sim/` bug.
