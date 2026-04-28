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
| Q | **Glia & neurovascular** | Astrocytes, oligodendrocytes / myelin, Schwann cells, microglia; neurovascular coupling | missing (added 2026-04-28 from Ch 11; entirely absent from the simulator) |

Clusters are stable identifiers (A, B, C, …) — names may be refined.

---

## Index of catalog sections (by anatomical/functional system)

Sections are appended as chapters are processed. Each section header lists the chapters from which entries were drawn.

- [Synapses & plasticity rules (Cluster J)](#cluster-j--synapses--plasticity-rules) — Ch 11, 12, 13, 14, 15, 16, 53
- [Channels & intrinsic dynamics (Cluster I)](#cluster-i--channels--intrinsic-dynamics) — Ch 8, 9, 10
- [Neuromuscular junction (Cluster M)](#cluster-m--neuromuscular-junction) — Ch 12
- [Development & critical periods (Cluster L)](#cluster-l--development--critical-periods) — Ch 48, 49

---

## Cluster J — Synapses & plasticity rules

Entries from Ch 11 (Overview of Synaptic Transmission) onward.

### J.01 Chemical synaptic transmission (general framework)
- **System:** all chemical synapses; CNS and PNS
- **Biological role:** unidirectional signaling via vesicular release of neurotransmitter into a 20–40 nm synaptic cleft, binding to postsynaptic receptors that gate ion flux. Allows amplification (one vesicle → thousands of channels open), modulation, plasticity. The dominant signaling mode in the brain.
- **Sim status:** implemented — `sim/bridge.py` synaptic conductance update + `sim/kernels.py` `fused_conductance_decay_and_current` model the postsynaptic effect; presynaptic AP triggers conductance increment in target. Synaptic cleft, exocytosis, and vesicle pool are abstracted away.
- **Cluster:** J
- **Prerequisites:** I (channels & AP), J.02 (receptor types)
- **Citation:** Kandel 6e Ch 11 p 241–253
- **Behavioral validation:** any working forward simulation already validates this — already covered by E/I balance benchmark (`run_benchmarks.py --benchmark ei-balance`).

### J.02 Ionotropic vs metabotropic receptor distinction
- **System:** all chemical synapses
- **Biological role:** ionotropic receptors are ligand-gated ion channels — fast (ms-scale), direct conductance change. Metabotropic receptors are GPCRs — slow (100 ms–min), trigger second-messenger cascades (cAMP, IP3, DAG → PKA, PKC) that phosphorylate downstream effectors including channels themselves. The two-timescale architecture is what lets the brain combine fast computation with slow modulation.
- **Sim status:** partial — ionotropic receptors are implemented (AMPA-like fast conductance, NMDA with voltage-dep Mg2+ block, GABA with `E_inh = -75 mV`). Metabotropic receptors are *abstracted* via the neuromodulator subsystem (`sim/neuromodulators.py`): we don't model GPCRs / cAMP / kinases literally, but the *functional outcome* — concentration-dependent modulation of synaptic gain, plasticity rate, excitability — is captured. The textbook framing is biophysical; ours is phenomenological. `[discrepancy: project doc treats neuromodulators as a peer mechanism to STDP, while textbook treats them as a *type of receptor signaling* — both views are valid; ours skips the molecular layer.]`
- **Cluster:** J (ionotropic) + C (metabotropic / NM subsystem)
- **Prerequisites:** J.01
- **Citation:** Kandel 6e Ch 11 p 250–251 (Fig 11-9)
- **Behavioral validation:** ionotropic — already in NMDA-Mg2+-block validation. Metabotropic — confirm DA / NE / 5-HT concentration changes alter target metric (e.g. STDP rate doubles when DA concentration doubles, with the right dose-response curve).

### J.03 Active zone and synaptic vesicle pool
- **System:** all chemical synapses
- **Biological role:** transmitter is released from specialized presynaptic boutons containing 100–200 synaptic vesicles each, clustered at "active zones" with voltage-gated Ca²⁺ channels. AP-triggered Ca²⁺ entry triggers vesicle fusion (exocytosis) within ~1 ms.
- **Sim status:** not-applicable at our level of abstraction — vesicle dynamics are absorbed into the short-term plasticity (STP) Tsodyks-Markram model: parameters `stp_U` (release probability), `stp_tau_d` (depletion recovery), `stp_tau_f` (facilitation) phenomenologically capture vesicle-pool depletion and recovery. We do not model individual vesicles or active zones. `[discrepancy: textbook describes vesicle counts and Ca²⁺-triggered fusion as the substrate; project STP fields are the closest functional equivalent. Note in CLAUDE.md: STP fields are global defaults plus per-connection-type variants.]`
- **Cluster:** J
- **Prerequisites:** J.01
- **Citation:** Kandel 6e Ch 11 p 248–249 (Fig 11-7, 11-8)
- **Behavioral validation:** STP paired-pulse ratio benchmark (`run_benchmarks.py --benchmark stp-paired-pulse`) — covers it.

### J.04 Synaptic delay (~1 ms)
- **System:** all chemical synapses
- **Biological role:** the cumulative time for AP arrival, Ca²⁺ entry, vesicle fusion, transmitter diffusion across cleft, receptor binding, and channel gating — "usually 1 ms or less." Constrains how fast spike-based circuits can route information.
- **Sim status:** partial — there is no explicit per-synapse propagation delay in the current code path; the synaptic conductance is incremented in the same step the presynaptic AP fires. With `dt=0.5 ms` (Izh) or `0.05 ms` (HH) the de-facto delay is one step. For most network behaviors at our scale this is acceptable; for tight timing experiments (millisecond-precise STDP windows, gamma phase-locking), per-synapse axonal delay would matter. *Future work:* per-synapse delay buffer (a small ring buffer keyed off current step).
- **Cluster:** J
- **Prerequisites:** J.01
- **Citation:** Kandel 6e Ch 11 p 248 (Fig 11-8)
- **Behavioral validation:** STDP-curve validation already passes at our timescales. To stress-test delay, run a polychrony / synfire-chain experiment and check that the simulated chain produces the expected delay profile.

### J.05 Electrical synapses / gap junctions
- **System:** primarily inhibitory interneurons (cortical FS networks), retina (bipolar / amacrine), olfactory bulb, glia; widespread in immature CNS.
- **Biological role:** direct cytoplasmic coupling via connexon hemichannels (6 connexins each, 1.5–2 nm pore). Bidirectional, near-zero-delay current passage between two cells. Function: synchronize firing of populations (e.g. FS interneurons → gamma oscillations), rapid escape circuits (fish Mauthner cell, crayfish tail-flip), early developmental coupling, glial Ca²⁺ wave propagation.
- **Sim status:** missing. Our gamma-oscillation benchmark (Kandel-cited PING mechanism, peak 27–45 Hz) reproduces gamma without electrical coupling, by E↔I delay tuning. The textbook describes gap-junctional electrical coupling among FS interneurons as a *complementary* mechanism that further sharpens synchrony. Adding gap junctions would shift the benchmark from "PING-only" to "PING+ING+gap-junctional" (closer to the *in vivo* mechanism). **Implementation cost:** moderate — would need a new `cp_gap_junction_coupling` matrix and a `V_post += g_gap * (V_pre - V_post)` term added to the conductance step.
- **Cluster:** J (and indirectly E for retinal circuits, F for cerebellum where they exist on Purkinje axon collaterals)
- **Prerequisites:** I (V dynamics), J.01
- **Citation:** Kandel 6e Ch 11 p 241–248 (Fig 11-1, 11-4, 11-5)
- **Behavioral validation:** gamma-oscillation peak with FS interneurons electrically coupled should be sharper (lower power outside the gamma band) than current PING-only baseline. Replicate Whittington & Traub 2003 in-silico.

### J.06 Glial gap-junction networks (astrocyte Ca²⁺ waves)
- **System:** astrocytes (CNS), Schwann cells (PNS myelin)
- **Biological role:** astrocytes form a gap-junction-coupled syncytium across which Ca²⁺ waves propagate at 1–20 µm/s — orders of magnitude slower than APs. Function still partly unknown but implicated in metabolic coordination, neurovascular coupling, and slow modulation of nearby synapses (gliotransmission of glutamate, ATP, D-serine).
- **Sim status:** missing. We have no glial cells at all. The simulator is purely neuronal.
- **Cluster:** new — *Cluster Q: Glia and neurovascular* (proposed addition; updates the cluster table after this chapter).
- **Prerequisites:** J.05 (gap junctions)
- **Citation:** Kandel 6e Ch 11 p 248
- **Behavioral validation:** would require co-simulation of glia + neurons — currently out of scope. Skipping for now.

---
