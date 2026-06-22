# Generative-sequence frontier — STEP 0 (C1 consolidation de-risk) = GO: a trained spiking net loads onto the ONE bridge + tracks the trained LIF at 0.92 fidelity (2026-06-22)

**Scope:** the cheap, NO-training C1 feasibility de-risk for the generative-sequence frontier (Spine A). *Does a
TRAINED SPIKING net consolidate onto the one `SimulationBridge` and SPIKE there, reproducing the trained dynamics?*
`research/runners/_genseq_step0_bridge_load_probe.py`, GPU. **NO `sim/` edit.** On `main`.

## Result — GO (`step0: spikes=Y fidelity=0.918 | gen_f_coherent=Y gen_f_ppl=6.1 -> GO`)
- **Install (the C1 path) works:** layer-0 of `cortex_10M_seed42.npz` (a 4-layer LIF net, 66→2048→2048→2048→66)
  installs onto a real `SimulationBridge` as a frozen co-resident slice (~67k synapses) via `inject_explicit_wiring`,
  driven through `cp_external_input_current`, read from real `cp_firing_states`. NO `sim/` edit.
- **Spikes — the dynamics gap measured + closed:** the bridge has no LIF (`enums.py` = Izhikevich/HH/AdEx/RF), so
  AdEx-as-LIF. At native propagation (0.05–1.0) the slice is SILENT (the named LIF↔conductance mismatch); a single
  global synaptic-gain sweep recovers it (0 → 101 → 687 → **947** active hidden at gain 4× / 16× / 64×) — one scalar
  ANN→SNN conversion-calibration, NO `sim/` edit, NO surrogate-grad-on-bridge finetune needed at this stage.
- **Fidelity:** at the calibrated gain the on-bridge per-neuron spike-RATE tracks the off-bridge `bptt_snn` layer-0
  `forward_unroll` output at **Spearman 0.918 / Pearson 0.886 / top-k 0.94** (6 chars).
- **Anti-cheat (specificity) PASSES decisively:** matched-char Spearman **0.918** vs mismatched **0.025**, margin
  **0.893** — the slice computes each char's SPECIFIC trained mapping, not a generic high-firing pattern.
- **Gen-F re-confirm:** coherent fluent TinyStories English (*"...the mole felt so safe with the little girl. She was
  very kind..."*), held-out ppl **6.1** (`2026-05-17-generator-F-...PASS.md`); the `gen_f_ctl` word-shuffle is its
  anti-cheat control.

## What this de-risks
⇒ **C1 ("ends fully-spiking on the ONE bridge") is feasible at the entry point.** A backprop-trained spiking net's
weights load onto the bridge, spike, and reproduce the trained per-neuron computation at 0.92 rank-fidelity via a
single global gain calibration — the scoping's cheapest "conversion-calibration" path, NO `sim/` edit.

## Honest caveats → the next measurements
(a) **Layer-0 only, one-hot input** (no inter-layer error accumulation) — multi-layer fidelity is the next measurement.
(b) **Positive weights only** (pure-excitatory drive test) — signed-weight (E/I) routing is a named downstream
conversion concern. (c) The 64× gain is a **global scalar** — per-layer threshold-balance is the standard refinement.

## Next (Spine A loop)
**Gen-F IS the pretrained non-spiking generator** (~6M, ppl 6.1, coherent TinyStories) — so loop-step 1 (train) is
essentially in hand. The decisive next de-risk = **ANN→SNN-convert Gen-F to a SPIKING generator and confirm the
SPIKING version STILL generates coherent NOVEL text** (held-out ppl + novelty vs the measured 0-novel wall) — note the
spiking-BPTT `cortex_10M` is the *failed* (overfit) generator, so Spine A converts the *working* non-spiking Gen-F
instead. Then consolidate (this de-risked path) → C2 grow + no-forget. The convert (a Transformer→spiking conversion)
is its own sub-problem and gets a focused de-risk before committing the full pipeline.
