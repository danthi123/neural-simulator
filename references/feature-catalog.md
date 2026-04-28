# Feature Catalog: Biological Mechanisms in `Principles of Neural Science` (Kandel 6e)

This catalog enumerates biological mechanisms described in Kandel et al., *Principles of Neural Science*, 6th edition (2021), and maps each one onto the simulator's architecture. It is the master input for the biology-buildout roadmap.

**Source:** `references/textbooks/kandel-pns-6e/full-book.pdf` (gitignored — extracted to `full-book.txt` for processing).

**Page-number convention:** "p N" refers to the **textbook printed page**. PDF page = textbook page + ~40 (approximate, varies by chapter location). Where a citation says "Kandel 6e Ch 12 p 254" it means chapter 12, printed page 254.

**Sim-status legend:**
- `implemented` — in the codebase, working, validated.
- `partial` — partially implemented or implemented in a reduced form.
- `missing` — not in the codebase.
- `not-applicable` — biological detail that we deliberately do not model (sub-cellular, molecular, etc. that doesn't bear on functional behavior at our level of abstraction).

**Discrepancy convention:** when the textbook contradicts a project doc, the entry is marked `[discrepancy: …]`.

---

## Cluster framework

The original prompt referred to "Cluster A through E defined in the cheat-5 survey." That doc actually defines three architectural *options* for cheat #5 (structural plasticity, patch-matrix, compartmentalized DA), not five clusters. The cluster framework below is derived from the project's actual architecture (CLAUDE.md, SCIENCE_ROADMAP.md) and extended as new biological systems are encountered in the textbook.

| Cluster | Name | Scope | Current sim state |
|---|---|---|---|
| A | **Closed BG action-selection loop** | Per-action cortex → D1/D2 → GPi/GPe → thalamus → motor disinhibition cascade; same-action-only routing | implemented (g11_bg_runner) |
| B | **Striatal microcircuit & WTA** | MSN lateral inhibition, patch-matrix anatomy, cross-projections, structural pruning | partial (v3 lateral inhib shipped; cross-projections + pruning open) |
| C | **Dopamine & neuromodulation** | DA / NE / 5-HT / ACh concentration dynamics, receptor effects, compartmentalized DA, RPE | partial (NM subsystem framework done; broadcast DA + adaptive DA done; compartmentalized DA + tonic NE / ACh open) |
| D | **Hippocampus & sequence learning** | Place cells, grid cells, pattern separation/completion, DG/CA3/CA1 microcircuit, sleep-replay | partial (place cells via landmark perception; replay infra; DG/CA3/CA1 missing) |
| E | **Sensory perception & cortical encoding** | Beacon/landmark sensors, plastic sensory→cortex, learned perception, cortical columns | partial (8-direction beacon + landmark sensors; column structure missing) |
| F | **Cerebellum & error-correction** | Purkinje cells, climbing fibers, parallel fibers, granule cells, deep cerebellar nuclei | missing (presets exist; no circuit) |
| G | **Working memory / PFC** | Recurrent attractor dynamics, persistent activity, gating | partial (PFC region with recurrent connectivity; gating limited) |
| H | **Motor & spinal output** | α-motoneurons, motor units, central pattern generators, reflex arcs, muscle | partial (motor neuron pools per action; spinal CPGs missing; muscle missing) |
| I | **Channels & intrinsic dynamics** | Ion channels, action potential, passive properties, neuron models (HH/Izh/AdEx) | implemented (per-gate Q10 fix, ~30 region presets) |
| J | **Synapses & plasticity rules** | STDP, LTP, LTD, STP, eligibility traces, NMDA/AMPA/GABA receptors, homeostasis | implemented (most rules; structural pruning being added) |
| K | **Sensory transduction** | Photoreceptors, hair cells, mechanoreceptors, olfactory/gustatory receptors | missing (abstract directional sensors only) |
| L | **Development & critical periods** | Synapse formation/elimination, axon guidance, experience-dependent refinement | partial (per-pathway plasticity gates can model critical periods) |
| M | **Neuromuscular junction** | NMJ-specific machinery: ACh release, end-plate potential, miniature EPPs | missing (NMJ as a model system covered in Ch 12 but the simulator has no muscle output) |
| N | **Sleep & arousal** | NREM/REM cycles, ascending arousal system, sleep-replay | partial (sleep-replay infra; sleep stages missing) |
| O | **Emotion, reward, motivation** | Limbic system, reward circuitry, addiction, fear/anxiety | partial (reward signal abstract; limbic anatomy missing) |
| P | **Disease & neurodegeneration** | Parkinson's, Alzheimer's, Huntington's, schizophrenia, autism, epilepsy | missing (referenced as future direction) |

Clusters are stable identifiers (A, B, C, …) — names may be refined.

---

## Index of catalog sections (by anatomical/functional system)

Sections are appended as chapters are processed. Each section header lists the chapters from which entries were drawn.

- [Synapses & plasticity rules (Cluster J)](#cluster-j--synapses--plasticity-rules) — Ch 11, 12, 13, 14, 15, 16, 53
- [Channels & intrinsic dynamics (Cluster I)](#cluster-i--channels--intrinsic-dynamics) — Ch 8, 9, 10
- [Neuromuscular junction (Cluster M)](#cluster-m--neuromuscular-junction) — Ch 12
- [Development & critical periods (Cluster L)](#cluster-l--development--critical-periods) — Ch 48, 49

---
