# EMERGE-9d (rung-3d) — GO: the FULLY-SPIKING unsupervised HTM Temporal Memory. BOTH inference (LIF + dAP plateau + WTA) AND learning (the Bouhadjar three-term rule: STDP-windowed potentiation + presynaptic depression + dAP-rate homeostasis) self-organize context-specific high-order prediction — 1.000 across 6 seeds and up to 8 overlapping sequences; dAP-lesion collapses. The numpy spiking-substrate ladder for rung-3 is COMPLETE.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge9d_spiking_learning_derisk.py`; results `research/findings/raw/_emerge9d_*.json`. Reuse-by-import (`_emerge9b`, `_emerge9c`); NO `sim/` edit; CPU/numpy; multi-seed; scale runs concurrent.

## The single-variable step (the last numpy rung)
EMERGE-9c made INFERENCE spiking. Rung-3d makes LEARNING spiking too, so BOTH are on the spiking substrate — the honest single-spiking-substrate goal. The learning rule is the verified Bouhadjar-Diesmann 2022 three-term permanence rule (Eq. 1):
- **(1) Potentiation** — STDP-windowed: a winner cell reinforces its distal synapses FROM the prior-symbol WINNER spikes.
- **(2) Presynaptic depression** — a wrongly-predictive cell depresses its synapses to the prior context.
- **(3) dAP-rate homeostasis** — each cell tracks a low-pass predictive (dAP) rate `z`; potentiation is scaled by `(z* - z)`, so over-used cells stop potentiating and NEW contexts allocate onto fresh cells.

`SpikingLearnTM(SpikingTM)` overrides training; inference (LIF + dAP + WTA) is inherited from EMERGE-9c.

## The decisive fix
A pure-homeostatic allocation had a COLD-START problem: `z` starts at 0 for all cells, so it can't differentiate them, so the context chain never forms, so `z` never rises (circular). Fix: **bootstrap allocation with the proven committed-segment metric (EMERGE-9b)** while the three-term plasticity drives the permanences and homeostasis MODULATES potentiation (`0.5 + 0.5*(z*-z)`, never fully gating). Also carried the EMERGE-9b winner-vs-active lesson (match/learn against prior WINNERS, not the bursting active set) and the empty-segment fix.

## Results

| config | branch acc (per seed) | lesion | Markov | chance | verdict |
|---|---|---|---|---|---|
| n_seq=2, L=4 | 1.000 x3 | 0.000 | 0.500 | 0.500 | **GO** |
| 6 seeds | 1.000 x6 | 0.000 | 0.500 | 0.500 | **GO** |
| n_seq=4 | 1.000 x3 | 0.000 | 0.250 | 0.250 | **GO** |
| n_seq=8 | 1.000 x3 | 0.000 | 0.125 | 0.125 | **GO** |

A fully-spiking, unsupervised, local, no-teacher sequence-learning mechanism self-organizes robust context-specific high-order prediction — inference by LIF + dAP plateau + WTA, learning by the spike-timing three-term rule with dAP-rate homeostasis. Robust across 6 seeds, scales to 8 overlapping sequences, dAP-lesion load-bearing everywhere. This is three consecutive GOs (EMERGE-9b mechanism -> 9c spiking inference -> 9d spiking learning) and completes the numpy spiking-substrate ladder for rung-3.

## Next: rung-4 — the `sim/` two-compartment port (the substrate realization)
All numpy pieces are de-risked, so the `sim/` port composes proven mechanisms. Scope it cheap-first (research-gate the protected build), de-risking the RISKIEST new piece first:
1. A guarded two-compartment `NeuronModel` whose APICAL compartment generates the dAP plateau (reuse `fused_graded_dendritic_plateau` in `sim/kernels.py` + an Izhikevich/LIF soma) — the dAP IS the apical compartment.
2. Distal segments as a plastic `RegionPathway` carrying permanences with the three-term rule (the STDP + neuromodulator/plasticity infra exists).
3. Per-column WTA inhibition (FS interneuron, exists).
`sim/` edits are fair game for faithful biology (additive/default-off/byte-identical when off).

Also in parallel toward COMMUNICATION: does the fully-spiking TM scale to real sequences / a real vocabulary (corpus fragments) on one substrate?

## Honest scope
- Discrete-time numpy spiking (LIF substeps + spike-event plasticity); the `sim/` continuous-time two-compartment realization is rung-4.
- Unsupervised: no teacher; self-organization IS the deliverable. Anti-cheats: Markov floor (provably chance) + dAP-lesion collapse + full-context oracle + multi-seed, all in place.

## Artifacts
`research/runners/_emerge9d_spiking_learning_derisk.py`, `research/findings/raw/_emerge9d_{spiking_learning,6seed,nseq4,nseq8}.json`. Prior: `2026-07-02-emerge9c-spiking-tm-rung3b-GO.md`, `2026-07-02-emerge9b-htm-faithful-GO.md`.
