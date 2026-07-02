# EMERGE-9c (rung-3b) — GO: the unsupervised high-order sequence mechanism SURVIVES on a SPIKING substrate. LIF somas + a distal-dendrite dAP plateau + per-column WTA reproduce the context-specific branch prediction EXACTLY (spiking == discrete == 1.000, 6 seeds, up to 8 overlapping sequences; dAP-lesion collapses).

**2026-07-02 (autonomous).** Runner `research/runners/_emerge9c_spiking_tm_derisk.py`; results `research/findings/raw/_emerge9c_*.json`. Reuse-by-import (`_emerge9b`); NO `sim/` edit; CPU/numpy; multi-seed; scale runs concurrent.

## The single-variable step
EMERGE-9b proved the faithful HTM Temporal Memory (unsupervised, local, no-teacher) self-organizes robust context-specific high-order prediction as a discrete algorithm. Rung-3b changes **one variable — the substrate**: keep the validated local permanence learning UNCHANGED, but replace the discrete "predictive cell -> active" selection with real SPIKING dynamics. This is the path toward the single spiking substrate the master directive requires.

## Mapping (from the verified Bouhadjar-Diesmann 2022 spiking TM)
- A cell is "predictive" when its distal segment has >= act_th connected synapses from the active cells -> its apical dendrite emits a **dAP PLATEAU** (a sustained depolarizing current to the soma). **The dAP is our confirmed two-compartment neuron's apical compartment.**
- On the next symbol, all cells in the column receive feedforward input; dAP-primed cells (pre-depolarized) cross the LIF threshold FIRST, spike, and drive the column's inhibitory neuron -> per-column **WTA** suppresses the not-yet-fired cells -> sparse, context-specific firing. An unpredicted element has no primed cells -> the column BURSTS (mismatch signal).
- `SpikingTM(HTM)` overrides ONLY inference (LIF membrane `V += dt/tau*(-V) + input + plateau*primed`, threshold 1, per-column WTA cap at k_win). Learning inherited unchanged.

## The decisive fix
First pass gave spiking 0.000 vs discrete 1.000: I capped EVERY column at k_win winners, including the cue's burst — activating the wrong cue cells and breaking the learned chain. The discrete semantics: a predicted column fires SPARSE (WTA over primed cells), an UNPREDICTED column BURSTS (whole column active, so downstream cells see the winner cells among the burst). Fixing this (predicted -> spiking WTA; unpredicted -> burst-all-active) reproduced the discrete result exactly.

## Results

| config | spiking branch (per seed) | discrete | lesion | verdict |
|---|---|---|---|---|
| n_seq=2, L=4 | 1.000 x3 | 1.000 | 0.000 | **GO** |
| 6 seeds | 1.000 x6 | 1.000 | 0.000 | **GO** |
| n_seq=4 | 1.000 x3 | 1.000 | 0.000 | **GO** |
| n_seq=8 | 1.000 x3 | 1.000 | 0.000 | **GO** |

The context-specific high-order prediction is byte-for-byte preserved when the winner selection emerges from spiking competition (dAP-primed cells win) instead of a discrete rule, across seeds and up to 8 overlapping sequences. The dAP-lesion collapses it to chance -> the dendritic plateau is load-bearing. Locality asserted. The unsupervised sequence-learning mechanism now runs on a spiking substrate.

## Next (drive it)
1. **Spiking LEARNING** (rung-3b keeps discrete permanence learning): replace `_learn` with the Bouhadjar THREE-TERM rule driven by SPIKE TIMING — windowed STDP potentiation + constant presynaptic depression + dAP-rate homeostasis (the exact verified equations). Cheap-first numpy, reproduce the GO.
2. **rung-4 = the `sim/` two-compartment port**: a guarded additive `NeuronModel` with an apical compartment generating the dAP plateau (reuse `fused_graded_dendritic_plateau` + Izhikevich soma), distal segments as a plastic `RegionPathway` with the three-term rule, per-column WTA inhibition. `sim/` edits are fair game for faithful biology (additive/default-off/byte-identical when off).
3. **Toward communication:** richer/real sequences (corpus fragments, a real vocabulary) — does allocation scale to real language on one substrate?

## Honest scope
- rung-3b makes INFERENCE spiking; LEARNING is still the (validated) discrete rule — the next rung makes learning spiking (STDP three-term). The `sim/` port is rung-4.
- Unsupervised: no teacher. Anti-cheats: spiking-vs-discrete parity + Markov floor + dAP-lesion collapse + full-context oracle + multi-seed.

## Artifacts
`research/runners/_emerge9c_spiking_tm_derisk.py`, `research/findings/raw/_emerge9c_{spiking_tm,6seed,nseq4,nseq8}.json`. Prior: `2026-07-02-emerge9b-htm-faithful-GO.md`.
