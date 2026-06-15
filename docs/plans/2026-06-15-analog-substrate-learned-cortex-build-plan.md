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

### Phase 1 — graded inhibition (the retina's horizontal cells) for the whitening. **De-risk DONE; needs one guarded `sim/` edit.**
The cm pool's **analog g_e already tracks the population mean (+0.71)** — it's the *spike* readout that
corrupts it (depol block). The retina solves exactly this with **graded (non-spiking) inhibition**
(horizontal cells release transmitter proportional to their *graded* membrane potential, not spikes). Build:
a **graded synaptic transmission mode** — a source region's continuous activity (its normalized g_e /
membrane) drives a graded inhibitory current on the target, bypassing the spike threshold. **`sim/` edit**
(the bridge's transmission is spike-mediated): a default-OFF, byte-reviewed `graded_inhibition` pathway flag
(`RegionPathway(graded=True)` → the per-step inhibitory current uses the source's continuous state, not
`cp_firing_states`). Byte-identical when unused. **Gate:** the graded-inhibition whitened drive matches the
host whitening (neural ≈ host +0.25, vs the spiking cm's +0.14). Brain-grounded: Kandel retina (horizontal/
bipolar graded potentials); catalog E.05 (center-surround).

### Phase 2 — analog/graded projection precision. **De-risk first (numpy), then build.**
The hub→cortex spiking projection loses the diffuse structure (g_e +0.175 vs numpy +0.23). The fix is
higher-precision transmission: either (a) the same graded mode on the feedforward pathway (graded hub
activity → graded cortex drive, full-precision), or (b) a large spike budget + population averaging (the
readout sweep showed this recovers toward +0.33). **De-risk:** does a graded hub→cortex projection's g_e
match the numpy random projection (+0.23) on real? **Gate:** the bridge projection preserves the diffuse
structure to within the numpy reference.

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
