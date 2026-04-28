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

## Index of catalog sections

The catalog is organized in two layers:

1. **Section IV equivalent** (Ch 11–16, 48–49, 53) — initial 55 entries written in the foreground session, organized by cluster directly under their cluster H2 sections below.
2. **Catalog additions from Parts II, IV–IX** — 268 entries written by 7 parallel subagents covering the remaining ~1,200 textbook pages. Appended after the Section IV content, grouped by cluster (each cluster has both an "original" Section IV section AND an "additions" section near the end of the file).

Total: **~323 mechanism entries** across 17 clusters (A–Q).

For navigation, search the file for `## Cluster <X>` to find all section headings for a given cluster, and `### <X>.<NN>` for individual entries.

### By cluster — current state (post-merge)

| Cluster | Name | Entries | Sim state |
|---|---|---:|---|
| A | Closed BG action-selection loop | 9 | implemented (g11_bg_runner) |
| B | Striatal microcircuit & WTA | 7 | partial (v3 lateral inhib shipped) |
| C | Dopamine & neuromodulation | 27 | partial (DA fully deployed; NE/5-HT/ACh framework-supported) |
| D | Hippocampus & sequence learning | 20 | partial (place cells via landmarks; DG/CA3/CA1 missing) |
| E | Sensory perception & cortical encoding | 22 | partial (8-dir beacon + landmark; columnar / topographic missing) |
| F | Cerebellum & error-correction | 11 | missing (presets only; no circuit) |
| G | Working memory / PFC / cortical integration | 20 | partial (PFC region; single-compartment) |
| H | Motor & spinal output | 25 | missing-mostly (motor neurons abstract; no muscle / CPGs) |
| I | Channels & intrinsic dynamics | 23 | mostly implemented (HH/Izh/AdEx + per-gate Q10) |
| J | Synapses & plasticity rules | 39 | implemented (most rules); structural pruning under dev |
| K | Sensory transduction | 15 | missing entirely (abstract directional sensors only) |
| L | Development & critical periods | 23 | partial (per-pathway plasticity gates; molecular triggers missing) |
| M | Neuromuscular junction | 4 | missing |
| N | Sleep & arousal | 14 | partial (replay infra; sleep stages partial) |
| O | Emotion, reward, motivation | 19 | partial (DA reward; no amygdala / hypothalamus) |
| P | Disease & neurodegeneration | 37 | missing — most modelable: Parkinson's, schizophrenia, epilepsy |
| Q | Glia & neurovascular | 8 | missing entirely |

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
- **Cluster:** Q (Glia & neurovascular, added 2026-04-28 from this chapter)
- **Prerequisites:** J.05 (gap junctions)
- **Citation:** Kandel 6e Ch 11 p 248
- **Behavioral validation:** would require co-simulation of glia + neurons — currently out of scope. Skipping for now.

### J.07 AMPA receptor — fast excitatory cation channel
- **System:** all glutamatergic synapses in the CNS
- **Biological role:** primary mediator of fast excitatory synaptic transmission. Cation-permeable (Na⁺ in, K⁺ out), reverses near 0 mV. Decay τ ~2–5 ms. GluA1–GluA4 subunits; GluA2-lacking variants are Ca²⁺-permeable. Density / GluA2 editing is the substrate for synaptic-scaling homeostasis.
- **Sim status:** implemented as the *generic* fast excitatory conductance (`E_exc = 0 mV`, exponential decay via `fused_conductance_decay_and_current`). We don't track AMPA explicitly as a named subtype, but the kinetics and reversal correspond. Synaptic scaling (Pillar 2.5 homeostasis) implicitly models AMPA-receptor density adjustment.
- **Cluster:** J
- **Prerequisites:** J.01, J.02
- **Citation:** Kandel 6e Ch 13 p 277–280
- **Behavioral validation:** baseline E/I balance benchmark already covers this — exc rate 1.78 Hz, inh rate 3.25 Hz, CV(ISI) 0.86 — consistent with cortical L2/3 driven by AMPA-like fast conductance.

### J.08 NMDA receptor — voltage-dependent coincidence detector
- **System:** all glutamatergic synapses; especially dense in CA1 hippocampus, neocortical L2/3
- **Biological role:** ligand- AND voltage-gated; Mg²⁺ blocks the pore at resting V, unblocks above ~-40 mV. Highly Ca²⁺-permeable (~10× the AMPA Ca²⁺ flux). Slow kinetics (decay τ ~50–150 ms). The voltage-dependence makes it a coincidence detector for pre/post activity — the cellular substrate for Hebbian LTP. Subunit composition (GluN2A vs GluN2B) modulates kinetics; developmentally regulated. Hypofunction implicated in schizophrenia (NMDA antagonist ketamine produces psychotic symptoms).
- **Sim status:** implemented — `fused_nmda_update_and_current` in `sim/kernels.py` models the voltage-dep Mg²⁺ block. Used as the Ca²⁺ source for STDP/LTP. Slow kinetics captured. Subunit composition (GluN2A/GluN2B) is *not* differentiated.
- **Cluster:** J
- **Prerequisites:** J.07
- **Citation:** Kandel 6e Ch 13 p 281–286
- **Behavioral validation:** STDP timing curve (Bi & Poo 1998) already passes, demonstrating NMDA-driven Ca²⁺-dependent LTP/LTD with correct timing windows.

### J.09 Kainate receptor — third ionotropic glutamate receptor type
- **System:** mossy-fiber → CA3, presynaptic modulation of inhibitory transmission, some cortical inhibitory interneurons
- **Biological role:** ionotropic glutamate receptor with intermediate kinetics. Functions partly redundant with AMPA postsynaptically; more interesting role is presynaptic — kainate-mediated frequency-dependent facilitation/depression of release.
- **Sim status:** missing as a distinct receptor. Subsumed into AMPA-like generic excitatory conductance. *Likely not worth adding* unless we model mossy-fiber-specific phenomena.
- **Cluster:** J
- **Prerequisites:** J.07
- **Citation:** Kandel 6e Ch 13 p 277–278
- **Behavioral validation:** would only be meaningful in a CA3 mossy-fiber-specific experiment.

### J.10 GABA-A receptor — fast inhibitory Cl⁻ channel
- **System:** ubiquitous in CNS — basket, chandelier, Martinotti, neurogliaform interneurons
- **Biological role:** pentameric ligand-gated Cl⁻ channel (most common α1β2γ2). Reversal ~-75 mV in adult neurons (close to but slightly below V_rest), produces hyperpolarizing IPSP in most cells. **Shunting inhibition** is a separate effect: even when ΔV is small, opening Cl⁻ channels increases membrane conductance and divides the EPSP voltage by a factor — works without hyperpolarization. Allosterically modulated by benzodiazepines, barbiturates, alcohol, neurosteroids.
- **Sim status:** implemented — `E_inh = -75 mV` with 0.7× propagation scaling (CLAUDE.md). Shunting inhibition is *implicitly* captured by the increase in membrane conductance during inhibitory current; works correctly in single-compartment but cannot reproduce *compartment-specific* shunting (e.g. perisomatic vs distal-dendritic shunting differ in real pyramidal cells).
- **Cluster:** J
- **Prerequisites:** J.01
- **Citation:** Kandel 6e Ch 13 p 286–289
- **Behavioral validation:** E/I balance benchmark (`run_benchmarks.py --benchmark ei-balance`) covers it. PING gamma benchmark also depends on functional GABA-A.

### J.11 Glycine receptor — fast inhibitory Cl⁻ channel (mostly spinal)
- **System:** brainstem, spinal cord (Renshaw cells, Ia inhibitory interneurons)
- **Biological role:** functionally similar to GABA-A (Cl⁻-selective, fast IPSP) but pharmacologically distinct (strychnine-sensitive). Dominates inhibition in spinal motor circuits.
- **Sim status:** not-applicable. We don't model spinal cord; abstractly any spinal "glycinergic" inhibition would be subsumed under our generic GABA-A inhibitory channel. Worth flagging if we add a spinal CPG cluster.
- **Cluster:** H, J
- **Prerequisites:** J.10
- **Citation:** Kandel 6e Ch 13 p 286–289
- **Behavioral validation:** N/A unless spinal added.

### J.12 Postsynaptic density (PSD) — scaffolding & receptor anchoring
- **System:** all glutamatergic synapses (PSD-95, Homer, Shank, GKAP family proteins)
- **Biological role:** dense protein scaffold under the glutamatergic postsynaptic membrane that anchors AMPA + NMDA receptors at the right density and aligns them with presynaptic active zones. PSD-95 specifically anchors NMDA and stabilizes AMPA via TARPs. Plasticity mechanisms (LTP) rearrange the PSD to insert AMPA receptors. The PSD is one of the *protein bottlenecks* for synapse maintenance and plasticity.
- **Sim status:** not-applicable. The simulator has weights but no scaffolding proteins. AMPA-receptor insertion / removal is captured *phenomenologically* by weight changes (STDP, synaptic scaling). This is the right level of abstraction for circuit-level dynamics; it cannot capture pathologies that disrupt PSD specifically (e.g. SHANK3 mutations in autism — Cluster P).
- **Cluster:** J (with P implications)
- **Prerequisites:** J.07, J.08
- **Citation:** Kandel 6e Ch 13 p 280–281
- **Behavioral validation:** N/A.

### J.13 G-protein-coupled receptors (GPCRs) — metabotropic neurotransmitter receivers
- **System:** every modulatory neurotransmitter system uses GPCRs — DA D1/D2/D3/D4/D5, NE α1/α2/β1/β2, 5-HT (most subtypes except 5-HT3), muscarinic AChR (M1–M5), GABA-B, mGluR1–8, opioid, neuropeptides, odorants, light (rhodopsin)
- **Biological role:** seven-TM receptors that, on transmitter binding, activate a heterotrimeric G protein (Gα + Gβγ). Gα-GTP dissociates and activates a downstream effector (adenylyl cyclase, PLC, ion channel directly, etc.). Hundreds of GPCR genes — the largest receptor family. The **substrate of all neuromodulation in the brain.**
- **Sim status:** not-applicable as molecular machinery. The neuromodulator subsystem (`sim/neuromodulators.py`) abstracts the entire chain "GPCR → G-protein → effector → ion channel modulation" into three target types (`synaptic_gain`, `plasticity_rate`, `excitability_drive`) that shortcut from concentration to functional outcome. The textbook framing is mechanistic; ours is phenomenological. **Trade-off:** we cannot model receptor-subtype heterogeneity (D1 vs D2, β1 vs β2 etc.) at the molecular level, but we *do* model their distinct concentration-effect curves via per-target sensitivity. Discrepancy with project doc: CLAUDE.md treats neuromodulators as a peer mechanism to STDP — the textbook makes clear they are a *different type of receptor*, not a different system.
- **Cluster:** C, J
- **Prerequisites:** J.02
- **Citation:** Kandel 6e Ch 14 p 301–305
- **Behavioral validation:** dose-response curve of any neuromodulator effect (DA→plasticity, NE→excitability) should match published data for that specific receptor subtype.

### J.14 cAMP / PKA pathway
- **System:** Gs-coupled GPCRs (D1, β-adrenergic, 5-HT4/6/7) and Gi (D2, α2, mu-opioid — inhibits cAMP)
- **Biological role:** Gs activates adenylyl cyclase → cAMP → PKA. PKA phosphorylates a *huge* set of substrates: ion channels (K, Ca, HCN), receptors (AMPA Ser845 enhances function — substrate of LTP), CREB (transcription factor for long-term plasticity). cAMP is the most ubiquitous second messenger; PKA-CREB is the canonical "convert short-term experience to long-term memory" pathway (Kandel's Aplysia work).
- **Sim status:** not-applicable. The pathway-specific phosphorylation of AMPA-Ser845 is captured by our STDP weight changes; CREB-driven transcription (the "consolidation" arc, hours timescale) is **missing entirely** — we have no transcriptional state in the simulator. *Implication:* we cannot model the early-LTP / late-LTP distinction (cycloheximide-blockable late phase) without adding a per-synapse "consolidated weight" tier with slow protein-synthesis kinetics.
- **Cluster:** C, J, L (development & long-term plasticity)
- **Prerequisites:** J.13
- **Citation:** Kandel 6e Ch 14 p 305–311
- **Behavioral validation:** would need a long-duration experiment showing weights persist over simulated "hours" with a protein-synthesis dependence — not currently testable.

### J.15 PLC / IP3 / DAG / PKC pathway and intracellular Ca²⁺ release
- **System:** Gq-coupled GPCRs (mGluR1/5, muscarinic M1/M3/M5, α1-adrenergic, 5-HT2)
- **Biological role:** Gq → phospholipase C → cleaves PIP2 into IP3 + DAG. IP3 binds receptors on ER, releases stored Ca²⁺ into cytosol. DAG activates PKC (membrane-bound). The IP3-Ca²⁺ pathway is a *parallel* Ca²⁺ source to NMDA receptors and voltage-gated Ca²⁺ channels — important for forms of LTD (mGluR-LTD in cerebellum, hippocampus).
- **Sim status:** not-applicable. We don't track intracellular Ca²⁺ at all — only the NMDA-derived Ca²⁺ that drives STDP, and that is encoded as an eligibility trace, not an explicit [Ca²⁺] variable. mGluR-LTD specifically would require a separate Ca²⁺ source.
- **Cluster:** C, J, F (cerebellum LTD)
- **Prerequisites:** J.13
- **Citation:** Kandel 6e Ch 14 p 308–311
- **Behavioral validation:** N/A unless cerebellum (Cluster F) is added.

### J.16 Endocannabinoids — retrograde modulators of presynaptic release
- **System:** widespread; especially abundant in hippocampus, cerebellum, cortex
- **Biological role:** when a postsynaptic neuron is strongly depolarized, it synthesizes endocannabinoids (anandamide, 2-AG) on demand, which diffuse *backwards* across the synapse and bind presynaptic CB1 receptors, *reducing* transmitter release. Substrate of DSI (depolarization-induced suppression of inhibition) and DSE (suppression of excitation), and a form of LTD. Retrograde signaling — postsynaptic activity controls presynaptic strength.
- **Sim status:** missing. Our STDP rule is the only mechanism by which postsynaptic activity influences synaptic strength, and it is *Hebbian* (weight-changing) rather than *modulatory* (release-probability changing). Adding eCB-LTD would require a per-synapse retrograde signal driven by postsynaptic Ca²⁺ that scales the effective release probability — touching the STP machinery from the postsynaptic side, which the current architecture doesn't support.
- **Cluster:** J (with C-flavor as a modulatory mechanism)
- **Prerequisites:** J.03 (STP — release probability), J.13 (GPCRs)
- **Citation:** Kandel 6e Ch 14 p 313–315
- **Behavioral validation:** induction of DSE in a hippocampal-like model: prolonged postsynaptic depolarization should suppress IPSCs from connected interneurons for ~10 s.

### J.17 Nitric oxide (NO) — gaseous retrograde messenger
- **System:** NMDAR-coupled, especially in CA1 pyramidals; broader role in vascular coupling and PNS
- **Biological role:** Ca²⁺/calmodulin → nNOS → NO. NO diffuses freely across membranes (gas), enters presynaptic terminal, activates soluble guanylyl cyclase → cGMP → PKG. Originally proposed as the LTP-reinforcing retrograde signal (Hawkins, Schuman). Now thought to be one of *several* retrograde signals; necessity for LTP is debated.
- **Sim status:** missing. Same architectural reasoning as J.16 — postsynaptic→presynaptic signaling is not part of the current pathway.
- **Cluster:** J, Q (neurovascular)
- **Prerequisites:** J.13
- **Citation:** Kandel 6e Ch 14 p 315–316
- **Behavioral validation:** N/A.

### J.18 Long-term gene-expression-dependent plasticity (CREB, late LTP)
- **System:** all neurons; particularly studied in hippocampal CA1 (LTP) and Aplysia sensory-motor synapse (Kandel)
- **Biological role:** repeated or strong activation of cAMP/PKA → PKA enters nucleus → phosphorylates CREB → CREB activates transcription of "immediate early genes" (c-fos, zif268, BDNF) → protein synthesis → structural growth (new spines, new active zones) → late-phase LTP that lasts hours-days-permanently. The "long-term" half of memory consolidation. Cycloheximide / anisomycin block this; early LTP (≤1h) still occurs but late LTP fails.
- **Sim status:** missing. We have no transcriptional state, no protein synthesis, no late-LTP tier of weights. *Implementation cost:* moderate — could be a per-synapse "consolidation" variable that slowly tracks recent plasticity events and modulates a weight floor that resists later LTD. The structural-plasticity machinery currently being added (axon pruning + synaptogenesis) is the closest infrastructure but operates on connectivity not weights. Long-term gene-expression consolidation is a separate axis. *This is one of the more important missing mechanisms for long-horizon memory experiments.*
- **Cluster:** J, L
- **Prerequisites:** J.14
- **Citation:** Kandel 6e Ch 14 p 320–321
- **Behavioral validation:** classical Aplysia long-term sensitization protocol — train for N spaced trials, measure 24h-later behavior; cycloheximide blocks late phase only.

### J.19 Convergence of multiple modulators on same channels
- **System:** all neurons receiving multiple modulatory inputs
- **Biological role:** a single ion channel (e.g. KCNQ / M-current K⁺ channel) can be modulated by ACh (via Gq/PLC), 5-HT, NE, somatostatin, and others — sometimes additively, sometimes occlusively, sometimes with sign reversal. The same channel is the *integration point* for diverse modulatory inputs. This is *not* a redundancy: different modulators carry different *behavioral state* signals (arousal, attention, reward), and the channel sees their union.
- **Sim status:** partial. The neuromodulator subsystem allows multiple modulators to target the same `excitability_drive` or `synaptic_gain` parameter — they are *summed* per default. Occlusive or sign-reversal interactions are not first-class (would require explicit dependent-modulator wiring). Most simple multi-NM experiments would work.
- **Cluster:** C, J
- **Prerequisites:** J.13
- **Citation:** Kandel 6e Ch 14 p 318–319
- **Behavioral validation:** add two NMs targeting the same parameter, verify additive concentration → effect; reproduce a published example like ACh+NE on cortical excitability.

### J.20 Quantal release and the calcium-fourth-power dependence
- **System:** all chemical synapses
- **Biological role:** Katz showed transmitter is released in unitary "quanta" (one vesicle each). The number released per AP is binomially distributed — N release sites × probability p (governed by Ca²⁺ entry). Postsynaptic response at zero-current = mEPSC ≈ "amplitude of one quantum". P_release scales as roughly the **4th power** of presynaptic [Ca²⁺] (because synaptotagmin, the Ca²⁺ sensor, has 5 Ca²⁺-binding sites, ≥4 must bind cooperatively for fusion). High Ca²⁺-affinity synaptotagmin-1/2 mediates synchronous (~1 ms) release; lower-affinity Syt-7 mediates asynchronous release (~tens of ms).
- **Sim status:** partial. STP `stp_U` captures release probability per AP; STP gain is *not* explicitly Ca²⁺-fourth-power. Quantal noise is implicit in spike-driven discrete events (one spike → one conductance increment ≈ one "quantum"-equivalent). Asynchronous release: not modeled.
- **Cluster:** J
- **Prerequisites:** J.03
- **Citation:** Kandel 6e Ch 15 p 326–340 (esp. Katz; Dodge & Rahamimoff)
- **Behavioral validation:** STP paired-pulse benchmark already exists. To validate Ca-4th-power, add Ca-channel blocker analog (scale STP gain) and verify P_release scales as gain^4.

### J.21 SNARE complex (synaptobrevin/VAMP, syntaxin, SNAP-25)
- **System:** every neuron's presynaptic terminal
- **Biological role:** the molecular machine that fuses vesicle with plasma membrane. The four-helix SNARE bundle pulls vesicle and plasma membranes together. Synaptotagmin-1 is the Ca²⁺ sensor that triggers final fusion. Disruption (botulinum toxins cleave SNAREs; tetanus cleaves synaptobrevin) abolishes release.
- **Sim status:** not-applicable. Vesicle fusion is abstracted; we don't model individual SNARE proteins. Toxin experiments (BoNT, TeNT) cannot be reproduced. STP captures *macroscopic* release dynamics; molecular machinery is below our level.
- **Cluster:** J
- **Prerequisites:** J.03, J.20
- **Citation:** Kandel 6e Ch 15 p 340–350
- **Behavioral validation:** N/A.

### J.22 Synaptic vesicle pools (readily releasable, recycling, reserve)
- **System:** every chemical synapse
- **Biological role:** vesicles partition into three pools — RRP (~1% of total, docked at active zone, immediately fusible), recycling (~10–20%), reserve (~80%). Sustained high-frequency stimulation depletes RRP first; recycling pool refills it via endocytosis (clathrin- or kiss-and-run pathways). The pool kinetics dictate the time-course of synaptic depression and recovery — directly the substrate of STP.
- **Sim status:** partial. STP `stp_tau_d` captures the *recovery time constant* of release after depletion. Multi-pool kinetics (RRP vs recycling) is not explicit; one effective "depletion → recovery" timescale per connection type.
- **Cluster:** J
- **Prerequisites:** J.03, J.21
- **Citation:** Kandel 6e Ch 15 p 350–355
- **Behavioral validation:** STP paired-pulse benchmark covers single-pool case. Multi-pool would require sustained-stimulation experiments.

### J.23 Spontaneous miniature EPSCs / IPSCs (mPSCs)
- **System:** every chemical synapse
- **Biological role:** at rest, vesicles spontaneously fuse at low frequency (Hz), producing detectable single-quantum EPSCs / IPSCs. Originally used by Fatt & Katz as proof of quantal release. Recent work suggests spontaneous and evoked release may be partly *independent* (different vesicle pools, different SNARE complexes), with separate roles in homeostatic regulation and developmental signaling.
- **Sim status:** partial. We have OU-noise background drive that *functionally* approximates spontaneous synaptic noise, but not in a per-synapse-event way. True per-synapse mEPSC events are missing — would need to add a low-rate Poisson event trigger per synapse independent of presynaptic AP firing. **Possibly worth adding** because spontaneous release is now thought to drive homeostatic synaptic scaling.
- **Cluster:** J
- **Prerequisites:** J.20, J.22
- **Citation:** Kandel 6e Ch 15 p 326–328
- **Behavioral validation:** record postsynaptic membrane in absence of stimulation, count discrete events per second, verify rate and amplitude match published mEPSC distributions.

---

## Cluster C — Dopamine & neuromodulation (extended)

Entries from Ch 16 (Neurotransmitters). Note: the project already implements a *declarative* neuromodulator framework (`sim/neuromodulators.py`) — adding any of the systems below means writing a `NeuromodulatorConfig` with the right baseline / decay / production rules / receptor target list. The framework exists; specific NMs need to be configured per task.

### C.01 Glutamate — primary fast excitatory transmitter
- **System:** ~80% of cortical synapses; pyramidal cells, granule cells (cerebellum), retinal photoreceptors, etc.
- **Biological role:** the workhorse excitatory transmitter. Acts on AMPA + NMDA + kainate (ionotropic) and mGluR1–8 (metabotropic). Synthesized from glutamine via glutamine synthetase (mostly in astrocytes — glia again). Cleared from cleft by EAAT transporters on astrocytes (the glial Cluster Q again).
- **Sim status:** implemented as the generic excitatory channel (AMPA + NMDA conductances). Glutamate-specific transporters and the astrocyte-mediated glutamate cycle are not modeled. Spillover-mediated mGluR effects (J.15) are missing.
- **Cluster:** J (and C for mGluR)
- **Citation:** Kandel 6e Ch 16 p 360–365
- **Behavioral validation:** every working simulation; covered.

### C.02 GABA — primary fast inhibitory transmitter
- **System:** all CNS — basket / chandelier / Martinotti / SST / VIP interneurons; output projections of striatum (MSNs), cerebellum (Purkinje), thalamic reticular nucleus (TRN)
- **Biological role:** synthesized from glutamate via GAD65/67. Acts on GABA-A (ionotropic Cl⁻; J.10), GABA-C (retina-specific), GABA-B (metabotropic, slow Gi-coupled). Cleared by GAT transporters. Inverted developmentally — in immature neurons GABA is *depolarizing* (Cl⁻ reversal above V_rest because of high intracellular Cl⁻ in immature neurons, KCC2 expression low until ~P14).
- **Sim status:** implemented (`E_inh = -75 mV`). GABA-B (metabotropic, slow IPSP, presynaptic autoreceptor) is **missing** — would need an additional slower inhibitory channel pathway. Developmental Cl⁻ reversal switch is not modeled (we don't have a developmental clock).
- **Cluster:** J, B, A (BG output), F (cerebellum)
- **Citation:** Kandel 6e Ch 16 p 365–367
- **Behavioral validation:** PING gamma + E/I balance benchmarks. GABA-B addition would need a paired-pulse depression assay at high firing rates.

### C.03 Acetylcholine (ACh) — peripheral fast + central modulatory
- **System:** all neuromuscular junctions (peripheral, fast — Cluster M); central cholinergic systems = nucleus basalis of Meynert (basal forebrain → cortex), septal nuclei (→ hippocampus), pedunculopontine (→ thalamus, brainstem). Striatal TANs (tonically active interneurons) are also cholinergic.
- **Biological role (central):** modulates cortical state — high ACh = "attentive / cortical desynchronization", low ACh = "internal state / sleep". Drives slow-wave / REM transitions. Striatal ACh from TANs gates plasticity at corticostriatal synapses. Two receptor types: nAChR (ionotropic, Ch 12) and mAChR (M1–M5, metabotropic, Ch 14).
- **Sim status:** partial. Striatal TANs are present as a region preset (`HH_STRIATAL_TAN`, `IZH2007_STRIATAL_TAN`) but not yet wired into a working ACh→plasticity cycle. Basal forebrain ACh as an *attention* / cortical-state modulator is not deployed in flagship runs but is straightforward to add via the NM framework.
- **Cluster:** C, B (TANs), N (sleep modulation)
- **Citation:** Kandel 6e Ch 16 p 367–371
- **Behavioral validation:** add ACh modulator targeting cortical excitability. Stimulate "basal forebrain" pool, verify cortical desynchronization (decreased low-freq power, increased gamma).

### C.04 Dopamine (DA) — reward / reinforcement / motor / WM
- **System:** SNc → striatum (nigrostriatal motor), VTA → NAc + PFC + amygdala (mesolimbic / mesocortical reward + WM)
- **Biological role:** reward prediction error signal (Schultz 1997) — *phasic* DA encodes RPE; *tonic* DA encodes motivational state. Receptors: D1-like (D1, D5; Gs, increase cAMP) and D2-like (D2, D3, D4; Gi, decrease cAMP). Cortico-striatal LTP requires D1 + glutamate coincidence; D2 gates the indirect-pathway. DA-system pathologies: Parkinson's (SNc death → akinesia), schizophrenia (excess striatal DA → hallucinations), addiction (NAc DA hijacking).
- **Sim status:** **implemented** as the project's central NM. `current_reward_signal` drives plasticity via eligibility traces in `g11_bg_runner`. Asymmetric adaptive DA targeting (Item 4.7 in ROADMAP), surprise-LR-boost (4.8), and adaptive DA EMA gating (per-action DA) are all DA-specific implementations. Compartmentalized DA (per-action) is option 3 in the cheat-5 survey — open. **The DA system is the simulator's most fully developed NM.**
- **Cluster:** C (primary), A, B
- **Citation:** Kandel 6e Ch 16 p 371–376; Ch 43 (Reward / Addiction); Ch 38 (BG)
- **Behavioral validation:** Phase B BG cascade benchmark (4.08 ± 0.49, 6-seed) covers DA-driven action selection. Schultz-type RPE signature: phasic DA on unpredicted reward, dip on omission — currently not directly replicated as a unit test.

### C.05 Norepinephrine (NE) — arousal / vigilance / fight-or-flight
- **System:** locus coeruleus (LC) → diffuse cortical, thalamic, hippocampal, hypothalamic, spinal projections
- **Biological role:** tonic LC firing tracks behavioral arousal (low during sleep, high during stress). Phasic LC bursts on salient stimuli. Receptors: α1 (Gq), α2 (Gi, autoreceptor), β1/β2/β3 (Gs). Increases SNR by simultaneously suppressing background firing and enhancing selective response. Critical for memory consolidation in the hippocampus, attention in PFC.
- **Sim status:** partial. NM framework supports it; one prior session (E.1) tested NE on the silent-motor task and found it insufficient (the silent-motor trap is upstream of NE modulation). **Could be added easily** — has not been deployed in the current flagship config. Yerkes-Dodson curve (inverted-U arousal-performance relationship) would be a natural validation.
- **Cluster:** C
- **Citation:** Kandel 6e Ch 16 p 376–380
- **Behavioral validation:** add NE concentration; vary baseline; measure SNR (signal-induced firing rate change / background CV(ISI)). Should peak at intermediate NE.

### C.06 Serotonin (5-HT) — mood / impulsivity / sleep
- **System:** raphe nuclei → cortex, hippocampus, BG, thalamus
- **Biological role:** receptors are *huge* family — 5-HT1A–F (Gi), 5-HT2A/B/C (Gq), 5-HT3 (ionotropic — the only one), 5-HT4–7 (Gs). Behaviorally: mood (target of SSRIs in depression), impulsivity (low 5-HT → high impulsivity), sleep architecture, satiety. In BG, 5-HT modulates DA function bidirectionally.
- **Sim status:** missing entirely from current flagship. NM framework supports it; just hasn't been deployed.
- **Cluster:** C
- **Citation:** Kandel 6e Ch 16 p 376–380
- **Behavioral validation:** none currently meaningful in our task set; would need a depression / anxiety / decision-making model.

### C.07 Histamine — wakefulness
- **System:** tuberomammillary nucleus (TMN) of hypothalamus → diffuse cortex
- **Biological role:** TMN is the histaminergic equivalent of the LC — promotes wakefulness; H1 antagonists are sedating (older antihistamines).
- **Sim status:** missing. Easy to add; minor priority unless modeling sleep-wake transitions (Cluster N).
- **Cluster:** C, N
- **Citation:** Kandel 6e Ch 16 p 380
- **Behavioral validation:** N/A unless sleep-wake added.

### C.08 Neuropeptides — slow modulators (50+ types)
- **System:** colocalized with classical transmitters in many neurons (e.g. ChAT+VIP, GABA+NPY, GABA+SST, etc.)
- **Biological role:** released at high firing rates (lower release probability than fast NTs; require strong stimulation). Diffuse over longer distances ("volume transmission"). Receptors are GPCRs. Examples: substance P (pain), NPY (feeding inhibition), CRH (stress), vasopressin / oxytocin (social bonding, parturition), enkephalin / dynorphin / β-endorphin (analgesia, reward), somatostatin (cortical inhibitory subtype marker), VIP (cortical disinhibitory subtype).
- **Sim status:** missing as a class. Could be added via NM framework with high firing-rate-gated production rules. Most likely high-priority additions: SST and VIP for cortical interneuron diversity (B.01 follow-on); enkephalin/dynorphin for BG indirect-pathway modulation.
- **Cluster:** C, B
- **Citation:** Kandel 6e Ch 16 p 380–390
- **Behavioral validation:** specific to chosen peptide.

### C.09 Purinergic transmission (ATP and adenosine)
- **System:** widespread; especially active in autonomic ganglia, glia signaling, vascular regulation
- **Biological role:** ATP is co-released with classical NTs from many neurons; acts on P2X (ionotropic) and P2Y (metabotropic) receptors. Hydrolyzed extracellularly to adenosine, which acts on A1 (Gi, inhibitory; coffee blocks A1) and A2A (Gs, BG indirect pathway D2-MSN expressed). Ado/A2A antagonists (caffeine, istradefylline) are clinically useful for Parkinson's. Adenosine accumulation tracks "sleep pressure" — A1 builds up during waking, drops during sleep.
- **Sim status:** missing. NM framework can model adenosine-as-sleep-pressure (production rule: integrate firing rate; decay during sleep gate). Would compose with sleep-replay infrastructure.
- **Cluster:** C, N
- **Citation:** Kandel 6e Ch 16 p 380
- **Behavioral validation:** sleep-pressure model — verify adenosine concentration accumulates during simulated waking, drops during forced sleep replay.

### C.10 Gases (NO, CO) — covered in J.17; cross-listed here
- **System:** see J.17.
- **Citation:** Kandel 6e Ch 16 p 388–389.

---

## Cluster L — Development & critical periods

Entries from Ch 48 (Formation and Elimination of Synapses) and Ch 49 (Experience and the Refinement of Synaptic Connections). Many of these mechanisms are functionally analogous to project infrastructure (per-pathway plasticity gates, structural pruning, curriculum) but operate on different substrates than the textbook (no morphology, no developmental clock).

### L.01 Target recognition / synaptic specificity (cell-adhesion molecules)
- **System:** all developing CNS — visual system, autonomic ganglia, cortex, hippocampus, etc.
- **Biological role:** axons select specific postsynaptic partners using a combinatorial code of cell-surface recognition molecules: cadherins (homophilic adhesion → "like binds like"), protocadherins (~60 isoforms, individually identifying), neurexin (presyn) ↔ neuroligin (postsyn) pairs, LRRTM family, SynCAMs, ephrins/Eph receptors (often *anti-adhesive*, sorting by gradient), semaphorins/plexins (axon guidance with carryover into synaptic targeting). The wiring of which-axon-finds-which-target is *not* random in real brains.
- **Sim status:** missing entirely. Initial connectivity in the simulator is determined by `RegionPathway` declarations + density + spatial connectivity generators (see `sim/connectivity.py`); there is no molecular-recognition layer. This is fine *for a connectivity-as-fixed-prior* model — but in the patch-matrix striatum option (cheat-5 option 2), we'd want subsets of cortical pools to project to specific striatal pools, which is a hand-coded version of this targeting. **A real recognition-code mechanism would be a new system** — probably out of scope unless we want to model developmental wiring errors (autism-related neuroligin mutations).
- **Cluster:** L (and B for the patch-matrix application)
- **Prerequisites:** none (this *precedes* synapse function)
- **Citation:** Kandel 6e Ch 48 p 1182–1192
- **Behavioral validation:** would require a "wiring" benchmark — given a target connectivity matrix and a recognition-code, does our simulation produce the expected adjacency? Not testable in current architecture.

### L.02 Synapse elimination by activity competition
- **System:** NMJ (neonatal multi-innervation → 1:1 by P14), climbing fiber → Purkinje (1:1 in adult), cortex (massive overproduction → pruning to ~50% in adolescence)
- **Biological role:** "use it or lose it" at the synapse level. When a postsynaptic cell is innervated by multiple presynaptic axons, the synapses that fire *coincidently* with the strongest axon (high-correlation = "winners") are stabilized; weaker / poorly-correlated synapses are eliminated. The final 1:1 pattern at NMJ + climbing-fiber emerges through this competition. In cortex, ~50% of overproduced synapses are pruned during adolescence; pruning failure is implicated in schizophrenia and autism.
- **Sim status:** **partial — directly addressed by the structural-pruning option in cheat-5 survey** (option 1 in `docs/plans/2026-04-28-cheat5-real-options-survey.md`). The proposed mechanism — `cp_synapse_alive`, survival-score accumulation, prune when below threshold — is the project's analogue of biological synapse elimination. Once shipped, this entry's status becomes "implemented for the cheat-5 use case; not a general developmental mechanism." Currently **in active development** (see `docs/plans/2026-04-28-structural-plasticity-implementation.md`).
- **Cluster:** L, B (the most likely first deployment)
- **Prerequisites:** J.07–J.10 (synapse function), J.18 (long-term plasticity)
- **Citation:** Kandel 6e Ch 48 p 1198–1205
- **Behavioral validation:** Phase B + structural pruning Tier 2: 3-seed mean sum ≤ 4.5 → cheat #5 closed for real. Currently in Tier 1 / Tier 2.

### L.03 Glia-mediated synapse pruning (complement C1q, C3 → microglia phagocytosis)
- **System:** developing visual cortex, retinogeniculate refinement; ongoing in adult hippocampus
- **Biological role:** astrocytes secrete TGF-β → induces complement protein C1q expression on weaker synapses → C1q tags → microglia recognize C3 → engulf and remove. The classical-immunity complement pathway is repurposed for synapse refinement. Excessive complement-mediated pruning is now thought to contribute to schizophrenia and Alzheimer's (synapse loss).
- **Sim status:** missing. The functional outcome (eliminate weak synapses) is captured by L.02 / structural pruning. The *mechanism* (complement + microglia) is below our level of abstraction. We don't have glia (Cluster Q) at all.
- **Cluster:** Q, L
- **Prerequisites:** L.02, glia infrastructure
- **Citation:** Kandel 6e Ch 48 p 1198–1205
- **Behavioral validation:** N/A (would require glia model).

### L.04 Critical periods (visual, language, social)
- **System:** sensory cortex (V1 ocular dominance ~P21–P35 in mice / ~3 mo in humans), language areas, social-bonding circuits
- **Biological role:** windows of heightened plasticity, after which rewiring becomes much harder. Opening: experience-dependent maturation of GABA-A inhibition (PV-cell maturation) raises the network out of low-inhibition "permissive" state. Closing: perineuronal nets (PNNs) condense around mature PV cells, physically restricting synapse change; myelin-associated inhibitors (Nogo, MAG) up-regulate. Reopening (Hensch et al.): chondroitinase digesting PNNs, or fluoxetine, can re-open critical periods in adult animals.
- **Sim status:** **partial — functionally captured by the curriculum infrastructure**. The 2-phase curriculum in `g11_bg_runner` (warmup with cortex_to_d1 plastic + input layers frozen, then thaw input layers / freeze cortex) is a critical-period analogue — heightened plasticity for a window, then closure. The plasticity-gate substrate (`cp_plasticity_gain`) supports both opening and closing. **The infrastructure is exactly the right shape** for biological critical periods; we just don't model the molecular triggers (PV maturation, PNN deposition).
- **Cluster:** L
- **Prerequisites:** plasticity-gate infrastructure (already implemented)
- **Citation:** Kandel 6e Ch 49 p 1210–1230
- **Behavioral validation:** the curriculum experiments already validate the *functional* shape (warmup window → committed performance). To validate as a *biological* critical-period model would require: (a) PNN-analogue mechanism (slow accumulation of a "lock" variable that resists future plasticity gain), (b) reopening via simulated chondroitinase (zero out the lock variable). Stretch goal.

### L.05 Spontaneous-activity-driven refinement (retinal waves, etc.)
- **System:** developing retina, cochlea, spinal cord, hippocampus — *before* sensory experience
- **Biological role:** even before eyes open or ears function, the developing nervous system generates spontaneous patterned activity (retinal waves: bursts of correlated activity that propagate across the retina at fixed velocities) that drives the refinement of downstream connections via NMDAR-dependent rules. The wave content matters — random noise wouldn't produce ocular dominance maps; coherent waves do. The brain is *self-organizing* its sensory representations before experience arrives.
- **Sim status:** missing as a mechanism. We have OU noise as background; we don't have *patterned* spontaneous activity. Could be added by injecting structured noise patterns during a pretraining phase. **Likely useful** for the plastic-input-layer arc — a "developmental pretraining" via structured retinal-wave-like input might solve the cold-start problem that learned-perception had on 2026-04-26 without curriculum. Worth flagging as a future test.
- **Cluster:** L, E (sensory)
- **Prerequisites:** plasticity-gate infrastructure
- **Citation:** Kandel 6e Ch 49 p 1218–1222
- **Behavioral validation:** generate retinal-wave-like input → train sensory→cortex pathway during pretraining gate-open phase → freeze → verify cortex develops coherent receptive fields (analogue of orientation columns).

### L.06 Activity-dependent refinement is general (NMDAR-dependent)
- **System:** essentially every refinement in every system tested — visual, auditory, somatosensory, motor, BG
- **Biological role:** the *common substrate* of refinement is NMDAR-dependent Hebbian plasticity: coincident pre/post → strengthening; uncorrelated → weakening / pruning. NMDAR antagonists block refinement in all systems. This is precisely the generalization that *Hebb's postulate* predicts.
- **Sim status:** **implemented** (J.08 + STDP). The simulator's STDP is the algorithmic content of this principle. We use it pervasively (Phase B BG cascade, learned perception, hippocampus) — and the simulator's success on the perception arc validates the principle as deployed.
- **Cluster:** L, J
- **Prerequisites:** J.08
- **Citation:** Kandel 6e Ch 49 p 1226–1230
- **Behavioral validation:** STDP timing-curve benchmark (Bi & Poo 1998) — already passes.

---

## Cluster J — Synapses & plasticity rules (continued from Ch 53)

Ch 53 entries continue the J series — plasticity at the synaptic level, the substrate of implicit memory.

### J.24 Habituation (presynaptic depression)
- **System:** Aplysia gill-withdrawal; analogous in vertebrates
- **Biological role:** repeated mild stimulation → progressive decrease in response amplitude. Mechanism: presynaptic Ca²⁺ entry decreases (Ca²⁺ channel inactivation); synaptic depression. Recovers with rest. Operationally indistinguishable in many cases from short-term synaptic depression (J.03 / J.22).
- **Sim status:** partial. STP depression (`stp_tau_d`) captures the time-course. However, *long-term* habituation (after many spaced training sessions, lasts days) requires gene-expression changes (J.18) and is **missing**.
- **Cluster:** J
- **Prerequisites:** J.03
- **Citation:** Kandel 6e Ch 53 p 1314–1320
- **Behavioral validation:** STP paired-pulse depression benchmark covers short-term. Long-term habituation: not currently testable.

### J.25 Sensitization (presynaptic facilitation, 5-HT-mediated)
- **System:** Aplysia (5-HT from interneurons → presyn sensory terminal); analog in vertebrates (NE from LC enhances release at hippocampal afferents)
- **Biological role:** noxious stimulus → 5-HT release → cAMP-PKA in presynaptic terminal → phosphorylation of K⁺ channels → broader AP → more Ca²⁺ entry → more transmitter release. Short-term (minutes); long-term (days) requires CREB-mediated transcription. Kandel's Nobel-winning work.
- **Sim status:** partial. NM framework can implement 5-HT-driven gain modulation of release probability. Not currently deployed. Long-term sensitization (CREB-dependent): missing (J.18).
- **Cluster:** C, J
- **Prerequisites:** J.13, J.14
- **Citation:** Kandel 6e Ch 53 p 1320–1325
- **Behavioral validation:** Aplysia-like sensitization protocol — single shock primes a stronger response to subsequent gentle touch, decays in minutes.

### J.26 Classical conditioning (Aplysia model)
- **System:** Aplysia gill-withdrawal; mammalian eyeblink (cerebellar); fear conditioning (amygdala)
- **Biological role:** activity-dependent presynaptic facilitation — the CS pathway gets selectively *more* facilitated than other pathways because Ca²⁺ entry from CS spike *coincides* with the 5-HT modulator pulse from US. Adenylyl cyclase is the *coincidence detector* (its activity is enhanced by Ca²⁺/calmodulin AND by Gs from 5-HT GPCR — the Gs-Ca²⁺ AND-gate).
- **Sim status:** partial. The associative-conditioning experiment in `experiment/presets.py` does the functional equivalent — paired CS+US with STDP-driven weight change. The *mechanism* is different (we use Hebbian+reward at corticostriatal-like synapses) but the outcome matches.
- **Cluster:** J, O (emotion)
- **Prerequisites:** J.25
- **Citation:** Kandel 6e Ch 53 p 1325–1330
- **Behavioral validation:** existing `run_experiment_headless.py --preset associative` already validates the outcome (CS-on rate increase, weights 0.10 → 0.999, t=11.36).

### J.27 Memory reconsolidation
- **System:** all long-term memory systems
- **Biological role:** reactivating a long-term memory makes it transiently labile — protein-synthesis inhibitors *during retrieval* erase the memory. Suggests retrieval re-stabilizes through the same gene-expression mechanism as initial storage. Therapeutic implication: PTSD treatment via reconsolidation blockade with propranolol.
- **Sim status:** missing. Same dependency as J.18 — no transcriptional state means no reconsolidation. *Implementation cost:* moderate, paired with J.18.
- **Cluster:** J, L
- **Prerequisites:** J.18
- **Citation:** Kandel 6e Ch 53 p 1330–1334
- **Behavioral validation:** N/A.

### J.28 LTP / LTD — long-term potentiation / depression
- **System:** ubiquitous in vertebrate CNS; canonical in CA1 (Schaffer collateral → CA1 NMDAR-LTP), neocortex, BG, cerebellum
- **Biological role:** the workhorse mammalian plasticity rule. **NMDAR-LTP**: high-frequency stim → NMDAR-driven postsynaptic Ca²⁺ → CaMKII activation → AMPA receptor insertion → larger EPSC at the same synapse. **NMDAR-LTD**: low-frequency stim → moderate Ca²⁺ → calcineurin (PP1) → AMPA removal. The Ca²⁺-amplitude switch (high → LTP, moderate → LTD) is the BCM-like substrate. **mGluR-LTD** (cerebellum, hippocampus): a *separate* LTD pathway via mGluR1/5 → Ca²⁺ release from stores → endocannabinoid retrograde signal (cerebellum). Both forms contribute to memory.
- **Sim status:** **implemented** as STDP — the spike-timing version of NMDAR-LTP/LTD. The Ca²⁺ amplitude → sign-of-change mapping is implicit in the STDP kernel (`fused_stdp_weight_update`). Soft-bound LTP captured. mGluR-LTD specifically: missing (J.15).
- **Cluster:** J
- **Prerequisites:** J.08
- **Citation:** Kandel 6e Ch 53 p 1314–1320 (also Ch 54 for hippocampal LTP detail)
- **Behavioral validation:** STDP timing-curve benchmark (Bi & Poo 1998) — already passes for both LTP and LTD.

### J.29 Spike-timing-dependent plasticity (STDP) as the temporal version of LTP/LTD
- **System:** all NMDAR-bearing synapses
- **Biological role:** when presynaptic spike *precedes* postsynaptic spike by ~10 ms → LTP; reverse → LTD. The asymmetric kernel emerged from observing that the post-spike's back-propagating AP unblocks NMDA Mg²⁺ block coincidently with active glutamate, while reverse pairing leaves Mg²⁺ blocked. Bi & Poo 1998 hippocampal cultures.
- **Sim status:** **implemented** — `fused_stdp_weight_update` with asymmetric pre→post / post→pre kernels. STDP is the mainline plasticity rule of the simulator.
- **Cluster:** J
- **Prerequisites:** J.08, J.28
- **Citation:** Kandel 6e Ch 53 p 1318–1320 (the principle); Bi & Poo 1998
- **Behavioral validation:** Bi & Poo benchmark: kernel matches theory to 3e-8, full-sim verified at dt=±5, ±20 ms.

### J.30 Local protein synthesis at synapses (CPEB / prion-like, mRNA in spines)
- **System:** Aplysia and mammalian dendrites
- **Biological role:** Kandel's Lasker / Nobel work. Some forms of long-term facilitation require *local* protein synthesis at the activated synapse — synapse-specific tagging via CPEB (cytoplasmic polyadenylation element binding protein), which has a prion-like self-templating domain. CPEB self-aggregates in the activated spine, becomes a stable mark, and locally drives translation of mRNAs already pre-positioned in dendrites. This is how the cell achieves **synapse specificity** in long-term memory — not all synapses on one cell get the same gene-expression boost.
- **Sim status:** missing. Per-synapse late-LTP tagging would require: (a) per-synapse "synaptic tag" boolean, (b) cell-wide gene-expression product, (c) tag × product → stabilization. Cluster L / J.18 dependency.
- **Cluster:** J, L
- **Prerequisites:** J.18
- **Citation:** Kandel 6e Ch 53 p 1325–1330
- **Behavioral validation:** N/A.

### O.01 Amygdala-mediated threat / fear conditioning
- **System:** lateral amygdala (LA), basolateral amygdala (BLA), central amygdala (CeA)
- **Biological role:** the mammalian implementation of Pavlovian threat conditioning. LA receives convergent CS (e.g., tone, from auditory thalamus + cortex) and US (shock, from somatosensory thalamus). Coincident input → NMDAR-LTP at LA synapses. CeA outputs to brain stem (freezing, autonomic responses). Extinction is *not* erasure — it's new inhibitory learning competing with the stored fear memory.
- **Sim status:** missing. Project has no amygdala. Pavlovian conditioning experiment (`experiment/presets.py`) does the *behavior* (CS-US pairing → CS-driven response) but not the *circuit*. Adding an amygdala region would compose with the existing region framework.
- **Cluster:** O (emotion / reward), J
- **Prerequisites:** L.06, region framework
- **Citation:** Kandel 6e Ch 53 p 1330–1335; Ch 42, 43
- **Behavioral validation:** add amygdala region with CS+US convergence; verify CeA output drives behavior after pairing; verify extinction adds parallel inhibition rather than erasing LA→CeA weights.

---

## Cluster I — Channels & intrinsic dynamics

### I.01 Axon initial segment (AIS) — spike-trigger zone
- **System:** all neurons (the unmyelinated 20–60 µm proximal axon segment)
- **Biological role:** highest density of voltage-gated Na⁺ channels in the cell — orders of magnitude more than the soma — so the AIS has the lowest threshold and is where the AP is initiated. AIS length and Na-channel composition adapts on a timescale of hours (homeostatic plasticity of intrinsic excitability).
- **Sim status:** missing as a distinct compartment. We model each neuron as point-like with one threshold (Izh / HH / AdEx); there is no spatial separation between integration site (soma) and trigger site (AIS). For most circuit-level dynamics this is fine, but for AIS plasticity (slow homeostatic changes to intrinsic excitability) the closest analogue is our `homeostasis_threshold_adapt_rate` parameter.
- **Cluster:** I
- **Prerequisites:** I (Hodgkin-Huxley-like AP machinery; covered in Ch 9-10)
- **Citation:** Kandel 6e Ch 13 p 293–295
- **Behavioral validation:** the homeostasis benchmark validates the *functional* outcome of AIS plasticity (perturbation +200 pA → 50 Hz → recovers to baseline within 1 s). Mechanism is at the threshold-adaptation layer, not the AIS layer.

---

## Cluster G — Working memory / PFC / cortical integration

### G.01 Spatial and temporal summation
- **System:** all neurons; especially relevant for pyramidal cells with thousands of inputs
- **Biological role:** EPSPs / IPSPs from many simultaneous (spatial) or sequential (temporal) inputs sum at the soma and axon initial segment. The cell's spike-or-no-spike output is the integrated summation crossing threshold. Membrane time constant (τ_m) and length constant (λ) determine the windows over which summation is effective.
- **Sim status:** implemented at the single-compartment level. Each neuron integrates all inputs in one membrane equation per step — this is exact spatial summation. Temporal summation is captured by the conductance decay τ. Multi-compartment spatial summation (with passive cable decay across dendrites) is **not** implemented — dendrites are collapsed to a point neuron.
- **Cluster:** G (and fundamental to all of I/J — covered here as the *integration* machinery)
- **Prerequisites:** I (passive cable, AP)
- **Citation:** Kandel 6e Ch 13 p 290–293
- **Behavioral validation:** any working network simulation; covered.

### G.02 Active dendrites — local computation, dendritic spikes
- **System:** especially L5 pyramidal cells (large apical dendrite tree), hippocampal pyramids
- **Biological role:** dendrites contain voltage-gated Na⁺, Ca²⁺, and HCN channels — supporting local AP generation (dendritic spikes), NMDA spikes (NMDAR-driven plateau potentials), and Ca²⁺ spikes at the apical tuft. These produce nonlinear summation rules (e.g. cluster of inputs on one branch ≫ scattered inputs on many branches), gain modulation by apical-basal coincidence (Larkum's two-layer model), and dendritic computation as a substrate for cortical hierarchical processing.
- **Sim status:** missing. Single-compartment everywhere. This is one of the largest abstractions in the simulator. Our `g11_bg_runner` PFC region has recurrent connectivity in cluster G — but PFC neurons are point neurons, not L5 pyramidals. *Implications:* we cannot reproduce experiments where the apical-basal coincidence detection is the substrate (e.g. perceptual inference via L5 apical tuft activity, conscious access models). Compartmental neurons would be a major addition (~10× compute per neuron at minimum).
- **Cluster:** G, I
- **Prerequisites:** I (channel kinetics)
- **Citation:** Kandel 6e Ch 13 p 293–298
- **Behavioral validation:** would require multi-compartment model. Could replicate Larkum BAC firing experiment (basal+apical coincidence → bursts).

---

## Cluster B — Striatal microcircuit & cortical interneuron diversity

### B.01 GABAergic interneuron diversity (basket, chandelier, Martinotti, neurogliaform)
- **System:** cortex (and analogous diversity in hippocampus, striatum)
- **Biological role:** different inhibitory subtypes target different compartments of the postsynaptic pyramidal cell:
  - **Basket cells** (PV+): perisomatic — control spike output, drive gamma
  - **Chandelier cells** (PV+): axon initial segment — gate spike generation directly
  - **Martinotti cells** (SST+): apical dendrites — gate dendritic Ca²⁺ events and feedback
  - **Neurogliaform cells**: volume transmission of GABA, slow IPSPs via GABA-B too
- **Sim status:** partial. We have one inhibitory class via `CORTICAL_FS_INTERNEURON` (PV-like fast-spiking) — corresponds best to basket cells. Chandelier / Martinotti / neurogliaform are not separately modeled. Implications: we cannot replicate experiments that require compartment-specific inhibition (e.g. Martinotti gating dendritic Ca²⁺ spikes). For the BG cascade, the relevant interneurons are TANs (already a preset) and FSIs (basket-equivalent) — both present.
- **Cluster:** B (striatal); also covers cortical microcircuit
- **Prerequisites:** I (interneuron neuron model), J.10 (GABA-A)
- **Citation:** Kandel 6e Ch 13 p 290–292
- **Behavioral validation:** PING gamma benchmark covers basket-equivalent behavior. To validate Martinotti would require active dendrites (missing).

---

## Cluster M — Neuromuscular junction

### M.01 Neuromuscular junction (NMJ) — overall structure
- **System:** α-motor axon → skeletal muscle fiber motor end-plate
- **Biological role:** the canonical "simple" chemical synapse — one motor axon innervates one site (end-plate) in adult mammals; presynaptic boutons sit in primary folds with active zones aligned to postsynaptic junctional folds (~10,000 receptors/µm² packed at fold crests); ~100 nm cleft (wider than CNS cleft) with basal lamina (collagen + AChE). Historically the Rosetta Stone for synaptic biophysics (Katz, 1950+).
- **Sim status:** missing. No muscle output in the simulator; "motor neurons" in `g11_bg_runner` are just abstract spike-emitting populations whose firing rate represents action selection. There is no muscle, no contraction, no force.
- **Cluster:** M (and H — motor & spinal output)
- **Prerequisites:** I (channels & AP)
- **Citation:** Kandel 6e Ch 12 p 254–256 (Fig 12-1)
- **Behavioral validation:** add a 1D Hill-type muscle model fed by motor neuron spike trains; verify that twitch summation, tetanus, and length-tension curves match published data. Stretch goal — only meaningful once we want to model real motor output.

### M.02 Nicotinic acetylcholine receptor (nAChR) — ionotropic
- **System:** NMJ; also autonomic ganglia and CNS (less relevant to this simulator)
- **Biological role:** pentameric ligand-gated cation channel (2α + β + γ + δ in muscle adult form). Two ACh binding sites required for opening; permeable to Na⁺ and K⁺ (and Ca²⁺ at lower conductance). Generates end-plate potential ~75 mV peak — well above the ~20 mV needed to trigger a muscle AP, so NMJ is normally "1:1 reliable" (no failures in healthy muscle). Suprathreshold safety factor distinguishes it from CNS synapses where summation is required.
- **Sim status:** not-applicable directly. The simulator's generic excitatory synapse (`E_exc = 0 mV`) collapses the AMPA/nAChR/NMDA distinction into one phenomenological conductance + reversal pair. ACh as a peripheral fast transmitter is not a separate channel type. *If* we add NMJ + muscle, nAChR would need to be a real channel type (different reversal, kinetics, conductance from AMPA).
- **Cluster:** M, J
- **Prerequisites:** J.01, J.02
- **Citation:** Kandel 6e Ch 12 p 256–270
- **Behavioral validation:** N/A until M.01 added.

### M.03 Acetylcholinesterase (AChE) — synaptic-cleft enzyme
- **System:** NMJ basal lamina; also CNS cholinergic synapses
- **Biological role:** hydrolyzes ACh into acetate + choline within ~1 ms of release, enforcing the "one-hit" rule that lets the postsynaptic conductance return to baseline between APs. Inhibitors (organophosphates, neostigmine — used clinically for myasthenia gravis) prolong receptor activation.
- **Sim status:** not-applicable. The exponential conductance decay (`fused_conductance_decay_and_current`) phenomenologically captures the cumulative effect of unbinding + AChE clearance; we don't track enzymes.
- **Cluster:** M, J
- **Prerequisites:** M.02
- **Citation:** Kandel 6e Ch 12 p 255
- **Behavioral validation:** N/A.

### M.04 End-plate potential / quantal release (foreshadowed; covered in Ch 15)
- **System:** NMJ
- **Biological role:** Katz showed that ACh is released in discrete *quanta* (each quantum ≈ contents of one synaptic vesicle, ~5,000 ACh molecules) — spontaneous miniature end-plate potentials (mEPPs) of ~0.5 mV occur at low frequency without stimulation; AP-evoked EPP equals the sum of N synchronously released quanta. Founded the vesicle hypothesis of release. **Detailed entry deferred to Ch 15.**
- **Sim status:** not-applicable directly; STP `stp_U` parameter captures the "release probability per AP" abstraction.
- **Cluster:** M, J
- **Prerequisites:** J.03
- **Citation:** Kandel 6e Ch 12 p 254 (introduction) → Ch 15 (full treatment)
- **Behavioral validation:** see J.03 entry.

---

---
# Catalog additions from Parts II, IV–IX (merged)

Entries merged from 7 parallel subagent passes. Entries are grouped by cluster; numbering continues from the existing Section IV entries (J.30, M.04, etc.).


## Cluster A — additions

### A.01 Direct pathway — D1 MSN → GPi/SNr disinhibition (action gating)
*[from Part V — Movement (Ch 30-39); renumbered from A.50]*

- **System:** D1-receptor medium spiny neurons (substance P / dynorphin) in striatum → GABAergic projection to internal globus pallidus (GPi) and substantia nigra pars reticulata (SNr).
- **Biological role:** Cortical input excites D1 MSNs → MSN inhibits the GPi/SNr neurons whose tonic 40–80 Hz firing was suppressing thalamus → thalamus disinhibited → cortex / brainstem effector released. This is the "go" pathway. Dopamine via D1 (Gs-coupled) increases MSN excitability.
- **Sim status:** **implemented** — `g11_bg_runner.build_bg_brain_regions` declares per-action `str_d1_X → gpi_X → thal_X → motor_X`. Disinhibitory cascade is the exact mechanism of A.50. cortex→D1 is plastic with `stdp_w_max=30`. **This is the simulator's flagship architecture.**
- **Cluster:** A primary; C (DA) secondary.
- **Prerequisites:** B.* (MSN microcircuit), C.* (DA modulation), I.*, J.*.
- **Citation:** Kandel 6e Ch 38 p 935–943.
- **Behavioral validation:** D1-pool stimulation → GPi pause → thalamus burst → motor selection; matches `g11_bg_runner` cascade probe.

### A.02 Indirect pathway — D2 MSN → GPe → STN → GPi/SNr (action suppression)
*[from Part V — Movement (Ch 30-39); renumbered from A.51]*

- **System:** D2-receptor MSNs (enkephalin) → GABAergic projection to external globus pallidus (GPe) → GPe inhibits STN → STN excites GPi/SNr → increased tonic inhibition of thalamus.
- **Biological role:** "No-go" pathway. Increased D2 activity increases GPi output → suppresses non-selected actions / brakes movement. Dopamine via D2 (Gi-coupled) decreases MSN excitability → disinhibits the indirect pathway less. Imbalance at root of Parkinson (less DA → indirect dominant → bradykinesia) and Huntington (D2 MSN loss → direct dominant → chorea).
- **Sim status:** **implemented** — `g11_bg_runner` declares `str_d2_X → gpe_X → stn (shared) → gpi_X` per action.
- **Cluster:** A primary; C, P (Parkinson/Huntington) secondary.
- **Prerequisites:** A.50, B.*, C.*.
- **Citation:** Kandel 6e Ch 38 p 935–943, p 952–956.
- **Behavioral validation:** D2-pool stimulation → GPi increase → action suppressed; DA depletion → indirect dominant → reduced action initiation.

### A.03 Hyperdirect pathway — cortex → STN → GPi/SNr (rapid global brake)
*[from Part V — Movement (Ch 30-39); renumbered from A.52]*

- **System:** layer V corticofugal neurons → glutamatergic projection directly to STN → STN excites GPi/SNr → fast, broad thalamic inhibition.
- **Biological role:** Provides short-latency global "stop / hold" signal before slower direct/indirect pathway resolution; essential for rapid action cancellation. Nambu's model.
- **Sim status:** **partial** — `g11_bg_runner` includes a shared STN region but no direct cortex → STN pathway is declared. Closing cluster A's "hyperdirect" gap is a small builder change: add cortex → STN `RegionPathway` per action.
- **Cluster:** A primary.
- **Prerequisites:** A.50, A.51.
- **Citation:** Kandel 6e Ch 38 p 941–946 (Nambu et al. 2002).
- **Behavioral validation:** Brief cortical pulse → fast STN burst → GPi burst → thalamic pause within ~10 ms; rapid stop-signal task analog.

### A.04 BG output disinhibition is selective — competitive WTA at GPi/SNr
*[from Part V — Movement (Ch 30-39); renumbered from A.53]*

- **System:** GPi/SNr GABAergic output neurons; tonic 40–80 Hz firing; receive focused inhibition from D1 MSNs of selected action and broad excitation from STN.
- **Biological role:** "Selected" channel = strongest inhibitory input from striatum → GPi/SNr neuron silenced → thalamic / SC target released. Non-selected channels remain inhibited. **Selection is an emergent property of the entire reentrant network.**
- **Sim status:** **implemented** — exactly what the per-action cascade in `g11_bg_runner` does, with v3 MSN lateral inhibition (`--bg-lateral-inhibition`) providing the cross-action competition. **Aligns with textbook.**
- **Cluster:** A primary; B secondary.
- **Prerequisites:** A.50–A.52.
- **Citation:** Kandel 6e Ch 38 p 939–942 (Mink 1996, Redgrave selection model).
- **Behavioral validation:** Two competing cortical inputs → only winning channel's thal pool fires; matches `g11_bg_runner` 6-seed flagship 4.26 ± 0.50.

### A.05 Reentrant cortico-BG-thalamo-cortical loops — parallel channels
*[from Part V — Movement (Ch 30-39); renumbered from A.54]*

- **System:** Alexander/DeLong scheme: motor / oculomotor / dorsolateral prefrontal / lateral orbitofrontal / anterior cingulate loops; topographically segregated through STR → GPi/SNr → thalamus → back to source cortex.
- **Biological role:** Loops are largely segregated but offer relay points where outside information can modulate signal flow. Functional territories preserved: limbic ventromedial → associative middle → sensorimotor dorsolateral gradient in striatum.
- **Sim status:** **partial** — flagship implements 4 sensorimotor channels (one per action). Limbic and associative channels not modeled. **[discrepancy: textbook emphasizes 5 parallel functional loops sharing the same circuit motif; project models only the motor loop].**
- **Cluster:** A primary; G secondary.
- **Prerequisites:** A.50–A.53.
- **Citation:** Kandel 6e Ch 38 p 943–948 (Alexander, DeLong, Strick).
- **Behavioral validation:** Anatomical: stimulating ACC-BG channel modulates orbitofrontal output without affecting motor channel.

### A.06 Cortico-striatal topography — sensorimotor / associative / limbic gradient
*[from Part V — Movement (Ch 30-39); renumbered from A.55]*

- **System:** dorsolateral STR (sensorimotor) ← motor cortex; central STR (associative) ← prefrontal; ventromedial STR (limbic) ← OFC, ACC, amygdala, hippocampus.
- **Biological role:** Each functional zone of cortex maps to a corresponding zone of striatum. Same MSN microcircuit applied to functionally diverse afferents ⇒ basal ganglia perform the *same* selection computation across motor, cognitive, and motivational domains.
- **Sim status:** **partial** — sensorimotor mapping captured by per-action cortex_X→str_X. No associative or limbic stripe.
- **Cluster:** A primary.
- **Prerequisites:** A.54.
- **Citation:** Kandel 6e Ch 38 p 943–948.
- **Behavioral validation:** Lesion dorsolateral STR → motor deficit; lesion ventromedial STR → motivational deficit; lesion associative → cognitive deficit.

### A.07 Subcortical BG loops — superior colliculus, brainstem
*[from Part V — Movement (Ch 30-39); renumbered from A.56]*

- **System:** SNr → superior colliculus (saccades), → MLR / PPN (locomotion), → thalamus → brainstem CPGs.
- **Biological role:** Not all BG output goes to cortex. Direct projections to SC release saccades; to MLR release locomotion. Phylogenetically older — present in lamprey. Argues that BG is fundamentally about action selection, with cortical loops a vertebrate elaboration.
- **Sim status:** **missing** — `g11_bg_runner` GPi → thal → motor is the cortical loop; no SC, no MLR projection.
- **Cluster:** A primary; H (locomotion init) secondary.
- **Prerequisites:** A.50–A.53.
- **Citation:** Kandel 6e Ch 38 p 938–942, McHaffie et al. 2005.
- **Behavioral validation:** SNr → SC inactivation → saccade initiation deficit; analogous to BG → MLR for locomotion.

### A.08 BG conserved across 500 My vertebrate evolution
*[from Part V — Movement (Ch 30-39); renumbered from A.57]*

- **System:** lamprey, fish, amphibian, reptile, bird, mammal — all have STR, GPi-equivalent, SNr-equivalent, dopaminergic input.
- **Biological role:** Architecture and neurotransmitters preserved → BG is a fundamental selection mechanism, not a mammalian elaboration. Justifies BG-as-action-selection-engine view.
- **Sim status:** **n/a** (history note); supports the design choice to use BG as the action-selection backbone.
- **Cluster:** A primary.
- **Prerequisites:** A.50.
- **Citation:** Kandel 6e Ch 38 p 942–945 (Reiner 2010, Grillner).
- **Behavioral validation:** Anatomical conservation evidence.

### A.09 Goal-directed vs habitual control — dorsomedial vs dorsolateral STR
*[from Part V — Movement (Ch 30-39); renumbered from A.58]*

- **System:** dorsomedial striatum (associative, goal-directed) vs dorsolateral striatum (sensorimotor, habitual).
- **Biological role:** Early learning is goal-directed (DMS, action-outcome contingency); with overtraining shifts to habitual (DLS, stimulus-response). Hierarchy: PFC ↔ DMS for novel tasks; DLS for automatized. Devaluation tests dissociate the two.
- **Sim status:** **missing** — no DMS/DLS split, no devaluation paradigm.
- **Cluster:** A primary; G, J secondary.
- **Prerequisites:** A.55.
- **Citation:** Kandel 6e Ch 38 p 950–953 (Yin & Knowlton 2006).
- **Behavioral validation:** After overtraining, devaluing the reward fails to change action selection (habitual); DLS lesion restores goal-directed control.

---

## Cluster B — Striatal microcircuit & WTA

## Cluster B — additions

### B.02 Medium spiny neuron (MSN) — >90% of striatum, GABAergic projection
*[from Part V — Movement (Ch 30-39); renumbered from B.50]*

- **System:** striatum; spiny dendritic morphology; bistable membrane (down-state ~−85 mV, up-state ~−55 mV).
- **Biological role:** Sole projection neuron of striatum. Silent at rest; requires substantial coordinated cortical / thalamic input to reach up-state and fire. Provides a thresholding effect — only strong, consensual inputs drive output.
- **Sim status:** **partial** — `IZH2007_STRIATAL_MSN_D1` and `_D2` presets exist and are used by `g11_bg_runner`. Bistability not explicitly modeled but Izhikevich-2007 captures rest-near-threshold sparsity.
- **Cluster:** B primary; A secondary.
- **Prerequisites:** I.*, J.*.
- **Citation:** Kandel 6e Ch 38 p 933–938.
- **Behavioral validation:** MSN silent at rest; cortex-stim → up-state → spike threshold met only at high coordinated drive.

### B.03 D1 vs D2 MSN segregation — opposing DA modulation
*[from Part V — Movement (Ch 30-39); renumbered from B.51]*

- **System:** D1 MSNs (Gs-coupled, ↑cAMP, substance P / dynorphin) vs D2 MSNs (Gi-coupled, ↓cAMP, enkephalin).
- **Biological role:** ~equal proportions in dorsal striatum. DA increases D1 MSN excitability and decreases D2 MSN excitability. Drives the asymmetric Go/NoGo balance underlying A.50/A.51.
- **Sim status:** **implemented** — `g11_bg_runner` declares separate `str_d1_X` and `str_d2_X` pools per action with appropriate Izhikevich presets.
- **Cluster:** B primary; A, C secondary.
- **Prerequisites:** B.50, C.* (DA).
- **Citation:** Kandel 6e Ch 38 p 935–940 (Surmeier).
- **Behavioral validation:** DA application → D1 firing ↑, D2 firing ↓ in vitro; corresponding behavioral release vs suppression.

### B.04 MSN lateral inhibition — local GABA collaterals (cross-pool WTA)
*[from Part V — Movement (Ch 30-39); renumbered from B.52]*

- **System:** MSN axon collaterals form local GABAergic synapses on neighboring MSNs within striatum.
- **Biological role:** Implements competitive selection within striatum. Anatomically dense; functionally weaker per-synapse than feedforward but collectively shapes which MSN ensemble wins. Combined with same-action-only cortex routing, produces winner-take-all dynamics.
- **Sim status:** **implemented (functional equivalent)** — `--bg-lateral-inhibition` (v3, default since 2026-04-28) adds MSN cross-pool lateral inhibition between per-action D1 pools. 6-seed sum 4.26 ± 0.50, no regression. Closed cheat #5 by design — see `research/findings/2026-04-28-cheat5-v3-results.md`. **[discrepancy: real BG has dense cross-action collaterals AND cross-action cortex inputs; project keeps cortex same-action-only because cross-projections were NEGATIVE in v1, v2, v3.1, v4].**
- **Cluster:** B primary; A secondary.
- **Prerequisites:** B.50, B.51.
- **Citation:** Kandel 6e Ch 38 p 935 (Silberberg & Bolam 2015).
- **Behavioral validation:** Single MSN spike → neighboring MSN IPSP; competitive selection in pool stimulation tests.

### B.05 Cholinergic tonically-active neuron (TAN) — striatal interneuron
*[from Part V — Movement (Ch 30-39); renumbered from B.53]*

- **System:** ~1–2% of striatal neurons; tonic 5–10 Hz firing; broadly arborized.
- **Biological role:** Pause response (~200 ms) to salient sensory cues, gated by thalamic centromedian/parafascicular input. Modulates corticostriatal plasticity via M1/M2 receptor effects on MSNs and DA terminals. Important for behavioral flexibility / set-shifting.
- **Sim status:** **partial** — preset `HH_STRIATAL_TAN` and `IZH2007_STRIATAL_TAN` exist but are not instantiated by `g11_bg_runner`.
- **Cluster:** B primary; C secondary.
- **Prerequisites:** B.50.
- **Citation:** Kandel 6e Ch 38 p 935–938.
- **Behavioral validation:** Salient cue → 200 ms TAN pause → permissive window for cortico-striatal plasticity.

### B.06 Fast-spiking PV+ interneuron — feedforward inhibition in STR
*[from Part V — Movement (Ch 30-39); renumbered from B.54]*

- **System:** ~1% of striatal neurons; parvalbumin+; receives strong cortical input; widely synapses on MSNs.
- **Biological role:** Feedforward GABAergic inhibition of MSNs. Provides another cross-action competition substrate (an MSN can be inhibited by a PV+ FS responding to a different cortical channel). Faster and stronger per-synapse than MSN collaterals.
- **Sim status:** **missing** — no FS interneuron pool in `g11_bg_runner`. (Note: motor-pool lateral inhibition with FS interneurons was tested 2026-04-26 and was MIXED/NEGATIVE; B.54 is the *striatal* FS variant which is biologically more standard.)
- **Cluster:** B primary; A secondary.
- **Prerequisites:** B.50, B.52.
- **Citation:** Kandel 6e Ch 38 p 935.
- **Behavioral validation:** Cortical pulse → PV+ FS spike at ~3 ms latency → MSN IPSP at ~5 ms.

### B.07 Striatal patch / matrix compartments — limbic vs sensorimotor mosaic
*[from Part V — Movement (Ch 30-39); renumbered from B.55]*

- **System:** patch (striosome) ↔ ventral midbrain DA neurons (limbic) vs matrix ↔ thalamus / cortex (sensorimotor); developmentally distinct, identified by μ-opioid receptor / calbindin staining.
- **Biological role:** Different plasticity rules and DA dynamics in patch vs matrix; patch implicated in habit / OCD; matrix in motor learning. Fine structural inhomogeneity within "the striatum."
- **Sim status:** **missing** — no patch/matrix distinction.
- **Cluster:** B primary; A, P secondary.
- **Prerequisites:** B.50.
- **Citation:** Kandel 6e Ch 38 p 935–940.
- **Behavioral validation:** Patch lesion alters habit acquisition curve differently from matrix lesion.

---

## Cluster O — Reward / dopamine

## Cluster C — additions

### C.11 Endogenous opioid descending pain control
*[from Part IV — Perception (Ch 17-29); renumbered from C.50]*

- **System:** PAG → RVM → spinal dorsal horn (substantia gelatinosa)
- **Biological role:** Periaqueductal gray drives raphé / RVM serotonergic + locus coeruleus noradrenergic descending pathways that gate spinal nociceptive transmission via enkephalin / endorphin release onto C-fiber terminals + projection neurons. Mediates stress analgesia and placebo effects.
- **Sim status:** missing — no pain modality, no opioid neuromodulator declared. NM subsystem could host μ-opioid as a target but no spinal substrate exists.
- **Cluster:** C (primary), O (motivation), K (nociception)
- **Prerequisites:** K.58
- **Citation:** Kandel 6e Ch 20 p ~545–550
- **Behavioral validation:** PAG stimulation analgesia; naloxone reversal of placebo.

### C.12 Gate-control theory (Aβ inhibition of C-fiber input)
*[from Part IV — Perception (Ch 17-29); renumbered from C.51]*

- **System:** spinal dorsal horn, substantia gelatinosa
- **Biological role:** Large-diameter Aβ touch afferents excite SG inhibitory interneurons that presynaptically inhibit C-fiber input to projection neurons — "rubbing reduces pain." First instance of competitive sensory gating in the CNS.
- **Sim status:** missing — no spinal dorsal horn microcircuit; spinal profile is single-pool.
- **Cluster:** C (primary), J (presynaptic inhibition), K
- **Prerequisites:** K.56, K.58
- **Citation:** Kandel 6e Ch 20 p ~542–545
- **Behavioral validation:** Aβ stim reduces nociceptive projection-neuron firing; TENS clinical analgesia.

### C.13 Peripheral / central sensitization (hyperalgesia, allodynia)
*[from Part IV — Perception (Ch 17-29); renumbered from C.52]*

- **System:** nociceptor terminals + spinal dorsal horn
- **Biological role:** Tissue damage releases inflammatory soup (bradykinin, prostaglandins, NGF) that lowers nociceptor thresholds (peripheral sensitization). Sustained C-fiber input causes NMDA-dependent LTP-like potentiation in dorsal horn (central sensitization), generating allodynia + hyperalgesia + secondary spread.
- **Sim status:** partial — NMDA + STDP are present (cluster J) but not in any nociceptive pathway.
- **Cluster:** C (primary), J (NMDA-LTP), K
- **Prerequisites:** K.58
- **Citation:** Kandel 6e Ch 20 p ~538–545
- **Behavioral validation:** thermal threshold drop after burn; mechanical allodynia after C-fiber tetanus; MK-801 blocks central sensitization.

---

## Cluster G — Cortical integration touched by perception

### C.14 Locus Coeruleus Norepinephrine System — pontine NE source
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.50]*

- **System:** Pontine nucleus (A6), the sole source of cortical/hippocampal/cerebellar NE.
- **Biological role:** Single small bilateral nucleus (~1500 neurons/side in rat) projects diffusely to virtually the entire forebrain. Fires fastest awake, slows in NREM, near-silent in REM. Phasic bursts gate selective attention; tonic firing relates to vigilance/scanning. NE acts via metabotropic α/β receptors to depolarize cortical pyramidal neurons and switch thalamic firing from burst to single-spike mode.
- **Sim status:** missing — neuromodulator subsystem is framework-supported (`sim/neuromodulators.py`) but no NE modulator is currently instantiated in flagship.
- **Cluster:** C primary, N secondary (arousal).
- **Prerequisites:** C.x (broadcast DA pattern), I.x (channels), J.x (synaptic gain).
- **Citation:** Kandel 6e Ch 40 pp 999-1006.
- **Behavioral validation:** Aston-Jones inverted-U: tonic-mode high baseline → exploration/labile attention; phasic-mode → focused attention with stimulus-locked bursts; very low → drowsy/inattentive.

### C.15 Dorsal/Median Raphe Serotonin System — midline 5-HT source
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.51]*

- **System:** Pontine and midbrain raphe nuclei (B5-B9 groups); medullary B1-B4 separately handle descending pain/autonomic modulation.
- **Biological role:** Forebrain-projecting raphe neurons modulate mood, cognition, arousal. Like LC, fire fastest awake, slow in NREM, silent in REM. 5-HT acts on ≥14 receptor subtypes — same ligand can excite or inhibit depending on target cell receptor expression. SSRI antidepressants suppress REM, providing strong evidence raphe gates REM-on circuitry.
- **Sim status:** missing — framework-supported NeuromodulatorConfig could declare 5-HT but no current deployment.
- **Cluster:** C primary, N secondary (REM gating), O secondary (mood).
- **Prerequisites:** C.50 (parallel monoamine architecture).
- **Citation:** Kandel 6e Ch 40 pp 999-1004; Ch 44 pp 1085-1086.
- **Behavioral validation:** Pharmacological: 5-HT reuptake blockade → reduced REM. Lesion of dorsal raphe DA → ~20% increase in total sleep.

### C.16 VTA / SNc Dopamine Origin Anatomy — A8/A9/A10 cell groups
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.52]*

- **System:** Midbrain dopaminergic cell groups: A8 (retrorubral), A9 (SNc, → dorsal striatum, motor), A10 (VTA, → NAc, PFC, amygdala, hippo, "mesolimbic/mesocortical").
- **Biological role:** A9 supports motor initiation (Parkinson hits A9 first); A10 is the canonical reward circuit. Both populations are heterogeneous — different VTA subpopulations encode reward, aversion, salience, with distinct afferents and targets. Tonic baseline firing supplies continuous DA needed for normal BG function; phasic bursts encode RPE.
- **Sim status:** partial — Phase B BG cascade has a single dopamine pool (A9-like) routed via reward signal; mesolimbic A10 → NAc and mesocortical A10 → PFC are not separately instantiated. [discrepancy: simulator collapses A9/A10 distinction; flagship's "DA" is functionally A9-like — single broadcast scalar — yet drives reward-like learning that biology assigns to A10/NAc.]
- **Cluster:** C primary, A secondary (BG), O secondary.
- **Prerequisites:** A.x (BG cascade), C.x (broadcast DA).
- **Citation:** Kandel 6e Ch 40 pp 999-1003 (Fig 40-11E); Ch 43 pp 1067-1068 (Fig 43-3).
- **Behavioral validation:** Lesion specificity — A9 ablation produces motor symptoms; selective A10/VTA lesion abolishes intracranial self-stimulation reward and conditioned place preference.

### C.17 Tuberomammillary Histamine System — sole brain histamine source
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.53]*

- **System:** Posterior lateral hypothalamus (E1-E5); the only source of brain histamine.
- **Biological role:** Strong wake-promotion. Active during wake, silent in sleep. Antihistamines (H1 antagonists) cause drowsiness — the most direct everyday demonstration. Innervates entire neuraxis from cortex to spinal cord.
- **Sim status:** missing — no histamine modulator declared.
- **Cluster:** C primary, N secondary (arousal flip-flop).
- **Prerequisites:** C.50/C.51 (parallel monoamine framework).
- **Citation:** Kandel 6e Ch 40 pp 999-1003; Ch 44 pp 1083-1085 (Fig 44-3).
- **Behavioral validation:** H1-antagonist administration → reduced cortical arousal, increased NREM propensity.

### C.18 Mesopontine Cholinergic Nuclei (PPT/LDT) — REM-on cholinergic
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.54]*

- **System:** Pedunculopontine (PPT, Ch6) and laterodorsal tegmental (LDT, Ch5) nuclei in the pons.
- **Biological role:** Project to thalamus and basal forebrain. Active in wake AND REM, near-silent in NREM. Cholinergic agonists promote REM; this is the classical pontine REM cholinergic mechanism. Also drive thalamic switch from burst → single-spike mode (via mAChR), supporting cortical readiness during wake.
- **Sim status:** missing — no ACh subsystem.
- **Cluster:** C primary, N secondary (REM-on driver).
- **Prerequisites:** C.50 (broadcast modulator pattern), N.x (sleep stage).
- **Citation:** Kandel 6e Ch 40 pp 1003-1006; Ch 44 pp 1083-1088 (Fig 44-3, 44-5).
- **Behavioral validation:** Cholinergic agonist injection into pontine reticular formation triggers REM-like state; muscarinic antagonist suppresses REM.

### C.19 Basal Forebrain Cholinergic + GABAergic Arousal — cortical drive
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.55]*

- **System:** Medial septum, diagonal band (Ch1-3), nucleus basalis of Meynert (Ch4); plus interleaved GABAergic and glutamatergic populations.
- **Biological role:** Diffuse cortical/hippocampal/amygdala innervation; ACh enhances responsiveness of cortical pyramidals; GABAergic component disinhibits cortex by inhibiting cortical inhibitory interneurons. Selective attention amplification beyond global arousal. Bilateral lesion of basal forebrain produces coma — i.e. this nucleus is on the *essential* parabrachial → basal forebrain → cortex pathway.
- **Sim status:** missing — flagship has no basal forebrain analog. Selective attention is approximated by motor exploration noise + sensory perception arc rather than gain-modulated attention.
- **Cluster:** C primary, N secondary, G secondary (PFC interaction).
- **Prerequisites:** C.54 (parallel ACh population), G.x (PFC).
- **Citation:** Kandel 6e Ch 40 pp 1003-1006 (Fig 40-15).
- **Behavioral validation:** Optogenetic activation of basal forebrain ACh/GABA/glutamate populations → cortical desynchronization (arousal); bilateral lesion → coma despite intact monoamines.

### C.20 Tonic vs Phasic Monoamine Firing — pacemaker baseline + bursts
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.56]*

- **System:** Generic property of monoaminergic neurons (LC, raphe, VTA, TMN).
- **Biological role:** Intrinsic pacemaker currents drive regular tonic firing in vivo (~1-5 Hz). Tonic delivery ensures continuous low-level neuromodulator availability at distant targets. Phasic bursts above tonic encode behaviorally relevant transients (RPE for DA, salience for LC). Many terminals are non-synaptic (volume transmission to many targets at once via diffusion).
- **Sim status:** partial — current flagship DA is a *scalar reward signal* updated only when reward changes; there is no continuous tonic baseline + transient burst structure. [discrepancy: project's `current_reward_signal` is event-triggered, not pacemaker + burst. The neuromodulator subsystem's `concentration_min/max` and decay-tau infrastructure could model this but isn't currently used to generate tonic baselines.]
- **Cluster:** C primary, J secondary (modulation of plasticity).
- **Prerequisites:** C.52 (DA), I.x (pacemaker channels).
- **Citation:** Kandel 6e Ch 40 pp 1001-1002 (Fig 40-12).
- **Behavioral validation:** In-vitro slice recording: monoaminergic neurons spontaneously pacemaker-fire at ~1-5 Hz with characteristic AHP + slow depolarization to next spike. In-vivo: tonic baseline + phasic event-locked bursts.

### C.21 Volume-Transmission Neuromodulation — non-synaptic diffuse release
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.57]*

- **System:** Monoamine + neuropeptide co-release from boutons-en-passant.
- **Biological role:** Many monoaminergic axon terminals do NOT form conventional synapses; transmitter spills into extracellular space and acts on G-protein-coupled receptors at micrometer-to-millimeter distances. This is fundamentally different from point-to-point glutamatergic/GABAergic transmission and is what justifies modeling neuromodulators as scalar fields rather than per-synapse signals.
- **Sim status:** implemented (architectural fit) — `sim/neuromodulators.py` already models neuromodulators as global concentration scalars with target-type effects (`synaptic_gain`, `plasticity_rate`, `excitability_drive`), which is the right abstraction for volume transmission.
- **Cluster:** C primary.
- **Prerequisites:** C.50 (LC) — canonical example.
- **Citation:** Kandel 6e Ch 40 pp 1001-1002.
- **Behavioral validation:** Microdialysis vs. fast-scan cyclic voltammetry — extrasynaptic monoamine concentration tracks population firing rather than individual release events.

### C.22 Dopamine Reward Prediction Error (Schultz RPE) — phasic DA encodes δ
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.58]*

- **System:** A10/VTA → NAc + PFC + amygdala + hippocampus.
- **Biological role:** **Project-critical.** Schultz monkey experiments (1997, 2016): unexpected reward → DA burst; cue-predicted reward → burst shifts to cue, no burst at reward; predicted reward omitted → DA *dip* below baseline at expected reward time. This is the canonical TD-learning prediction-error signal. Drives selective strengthening of synapses on coactive eligibility traces (three-factor rule).
- **Sim status:** partial — flagship implements broadcast DA = reward signal driving eligibility-trace × DA → weight update. The 2026-04-26 surprise-LR-boost variant explicitly amplifies LR by `(1 + α × |reward - reward_ema|)`, which IS an RPE-flavored mechanism. The 2026-04-26 adaptive-DA targeting uses reward EMA gating. **However, project doesn't model the burst↔cue transfer** — the cue itself doesn't acquire DA-burst-evoking power as it does in Schultz's data; reward only releases DA at delivery time. [discrepancy: textbook RPE includes cue-shift and omission-dip; project models pos-RPE-amplification but not the cue-shift dynamic that is the canonical RPE signature.]
- **Cluster:** C primary, A secondary (BG), J secondary (plasticity).
- **Prerequisites:** C.52 (VTA), C.56 (tonic/phasic), J.x (eligibility, STDP).
- **Citation:** Kandel 6e Ch 43 pp 1068-1069 (Fig 43-2); Schultz, Dayan, Montague 1997.
- **Behavioral validation:** Three-trial paradigm: (a) unexpected reward → burst at reward; (b) trained CS+R → burst at CS, no burst at R; (c) trained CS but R omitted → dip at R-expected time. Currently only (a) is faithfully reproduced.

### C.23 Heterogeneous DA Subpopulations — reward, aversion, salience VTA cells
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.59]*

- **System:** Subpopulations within A10/VTA distinguished by afferents, targets, and response polarity.
- **Biological role:** Not all DA neurons are RPE encoders. Some respond to BOTH reward and aversion (salience); some preferentially to reward; some to aversion only; some show inverted (reward-suppressed, aversion-activated) profiles. Anatomically these correspond to distinct projection targets — e.g. medial VTA → mPFC tends salience-coded; lateral VTA → NAc lateral shell tends reward-coded.
- **Sim status:** missing — flagship has a single homogeneous DA population. The cheat #5 v3.1 / v4 cross-projection failures may be related: a single broadcast DA cannot supply the differentiated teaching signals that biology distributes across subpopulations.
- **Cluster:** C primary, A secondary, O secondary.
- **Prerequisites:** C.52 (VTA), C.58 (RPE).
- **Citation:** Kandel 6e Ch 43 pp 1068-1069.
- **Behavioral validation:** Single-unit recording in identified VTA cells with retrograde tracing → diversity of stimulus-response profiles correlated with projection target.

### C.24 Dopamine in Aversion — DA also encodes salience and warning
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.60]*

- **System:** Subpopulation of VTA DA neurons activated by aversive stimuli (foot shock, air puff).
- **Biological role:** Counter-evidence to the simple "DA = pleasure" / "DA = reward only" view. Aversion-activated DA neurons may signal *salience* rather than reward valence. Dopamine-depleted rodents (6-OHDA) and DA-synthesis-knockout mice still show hedonic taste reactions to sucrose — so DA is NOT a hedonic signal, it's a learning/teaching signal.
- **Sim status:** partial — current scheme uses signed scalar reward (positive and negative). Negative reward decreases weights via STDP × DA, which IS a form of aversion encoding. But the "salience-only" subpopulation (responds to both valences) is not modeled.
- **Cluster:** C primary, O secondary.
- **Prerequisites:** C.58, C.59.
- **Citation:** Kandel 6e Ch 43 pp 1068-1069.
- **Behavioral validation:** Recording during foot-shock or aversive Pavlovian conditioning shows DA increase in subset of VTA neurons, dip in others.

### C.25 NAc cAMP-CREB Pathway Adaptation — chronic-DA homeostasis
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.61]*

- **System:** Nucleus accumbens medium spiny neurons; cAMP → PKA → CREB intracellular cascade.
- **Biological role:** Repeated drug exposure (or chronic strong reward) acutely *suppresses* cAMP via Gi-linked D2/μ-opioid/CB1 receptors. Cells adaptively *upregulate* adenylyl cyclase and CREB to restore baseline activity (tolerance). On drug removal, the upregulated pathway is unopposed → withdrawal hyperactivity. This is the molecular substrate of **reward tolerance**.
- **Sim status:** missing — flagship has no second-messenger modeling. Synaptic scaling (homeostasis) provides a coarse functional analog (sets activity setpoints) but operates on firing rate, not on RPE setpoint. [Could matter for long-horizon RL: project agents may not show realistic tolerance/sensitization to persistent reward.]
- **Cluster:** C primary, J secondary, O secondary.
- **Prerequisites:** C.58 (DA), J.x (homeostasis).
- **Citation:** Kandel 6e Ch 43 pp 1074-1075 (Fig 43-5).
- **Behavioral validation:** Repeated morphine → reduced cAMP/PKA acutely, gradually restored despite continued drug, then *elevated* on naloxone — measurable as PKA-dependent phosphorylation timecourse.

### C.26 ΔFosB Sensitization Memory — long-lived transcription factor
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.62]*

- **System:** NAc MSNs and other reward areas; truncated splice product of FosB gene.
- **Biological role:** Unique among Fos family in stability — accumulates over weeks of repeated drug/reward exposure. Mediates **reward sensitization** (escalating response, increased self-administration, relapse). Functionally it's a *long-timescale memory variable* of repeated exposure that biases behavior weeks after the last exposure.
- **Sim status:** missing — no homologous slow-accumulator state variable. Project lacks a "trait that integrates reward history over hours-to-weeks." This may matter for any long-running learning experiments where habituation/sensitization should occur.
- **Cluster:** C primary, J secondary.
- **Prerequisites:** C.61 (downstream cascade).
- **Citation:** Kandel 6e Ch 43 pp 1074-1075.
- **Behavioral validation:** Weeks-long viral overexpression of ΔFosB in NAc → enhanced cocaine self-administration and relapse-like behavior.

### C.27 Wanting vs Liking Dissociation — Berridge two-system reward
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.63]*

- **System:** NAc and ventral pallidum hedonic hotspots vs. mesolimbic DA "wanting" system.
- **Biological role:** "Liking" (hedonic taste reaction) is mediated by μ-opioid + endocannabinoid signaling in small NAc/VP hotspots and survives DA depletion. "Wanting" (incentive motivation, approach behavior) is DA-mediated. Amphetamine elevates wanting without elevating liking. Dopamine is *not* the pleasure signal — it's the incentive-salience / wanting signal.
- **Sim status:** missing — flagship has only one reward axis (DA-driven). No separate hedonic / consummatory pathway. [Architecturally important: the project's `current_reward_signal` does both jobs that biology splits — this is fine as a simplification but worth documenting.]
- **Cluster:** C primary, O secondary.
- **Prerequisites:** C.58.
- **Citation:** Kandel 6e Ch 43 p 1068; Ch 41 p 1038 (Berridge incentive motivation).
- **Behavioral validation:** Sucrose hedonic taste reactions intact in DA-knockout mice; "wanting"-type lever pressing is abolished.

---

## Cluster N — Sleep, Arousal, Replay (project-critical for Ch 44)

## Cluster D — additions

### D.01 Episodic memory — encoding / storage / retrieval / consolidation cycle
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.50]*

- **System:** Medial temporal lobe (hippocampus + perirhinal + parahippocampal + entorhinal) interfacing with frontoparietal networks and association cortices.
- **Biological role:** Binds multimodal items into events and events into episodes via temporal/spatial context (Tulving 1972; Eichenbaum/Cohen relational memory). Consolidation transforms labile traces into durable, distributed cortical representations through hippocampal–neocortical dialogue; re-encoding may follow each retrieval (reconsolidation).
- **Sim status:** missing as a system — sleep-replay infrastructure exists (NREM scaffolding) but no episodic encoder, no separate "labile vs consolidated" trace bookkeeping, no relational binding API. Phase-tagged plasticity gating (`set_plasticity_gate`) could express consolidation phases but no runner uses it.
- **Cluster:** D primary, N secondary (sleep-driven consolidation), G secondary.
- **Prerequisites:** D.51 (HC microcircuit), D.55 (place cells), N.* (replay).
- **Citation:** Kandel 6e Ch 52 pp 1296–1302.
- **Behavioral validation:** Anterograde amnesia for new associations after MTL lesion with preserved working/skill memory (H.M.); retrieval-time hippocampal–cortical reactivation (iEEG word-pair studies).

### D.02 Relational binding / "memory space" — Eichenbaum–Cohen model
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.51]*

- **System:** Hippocampus proper (CA1 output), with perirhinal item streams and parahippocampal context streams converging.
- **Biological role:** Stores events as items-in-context, episodes as temporal sequences of events, and networks via overlapping events allowing flexible inference (e.g., transitive). Distinguishes overlapping episodes that share elements (same restaurant, different visits) without interference.
- **Sim status:** missing — no relational binding primitive. Place-cell-like encoding from learned-perception of landmark sensors is content-only, not item+context; no episode boundary detection or sequence-of-events memory.
- **Cluster:** D primary, G secondary.
- **Prerequisites:** D.55 (place cells), D.56 (sequence learning).
- **Citation:** Kandel 6e Ch 52 pp 1301–1302; Eichenbaum/Cohen 2014.
- **Behavioral validation:** Inference on overlapping experiences (transitive inference); selective deficit on configural learning after dorsal-HC lesion.

### D.03 Trisynaptic pathway — EC layer II → DG → CA3 → CA1 (indirect)
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.52]*

- **System:** Entorhinal cortex layer II → perforant path → dentate gyrus granule cells → mossy fiber → CA3 pyramidal → Schaffer collateral → CA1 pyramidal → subiculum + EC deep layers (loop closure).
- **Biological role:** Three sequential excitatory stages with distinct functional properties at each: DG sparsifies, CA3 completes, CA1 outputs. Returns to deep EC for cortical broadcast.
- **Sim status:** missing — `sim/regions.py` allows declaring DG/CA3/CA1 as separate `BrainRegion`s and pathways with `density`, `weight_mean`, `plastic`, but no runner builds the trisynaptic loop. Current "hippocampus" is a single recurrent pool with place-cell-like place fields from landmark sensors.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** none — uses existing region/pathway primitives.
- **Citation:** Kandel 6e Ch 54 pp 1340–1342, Fig 54-1.
- **Behavioral validation:** Selective lesion at each stage produces distinct deficits (pattern separation, completion, output binding).

### D.04 Direct entorhinal pathway (temporoammonic) — EC layer III → CA1
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.53]*

- **System:** EC layer III axons targeting *distal* apical dendrites of CA1 (parallel to trisynaptic input arriving at proximal Schaffer dendrites).
- **Biological role:** Provides direct sensory context to CA1 in parallel with the indirect path. Distal/proximal segregation enables CA1 to compare current input against CA3-recalled pattern (a "match/mismatch" or novelty-detection function in some theories).
- **Sim status:** missing — would require multi-compartment CA1 or distinct excitatory pathways with different dendritic-zone effects (currently CA1 single compartment can only sum inputs).
- **Cluster:** D primary, I (channels — needs dendritic compartments).
- **Prerequisites:** D.52, multi-compartment neuron support.
- **Citation:** Kandel 6e Ch 54 p 1340.
- **Behavioral validation:** Direct-pathway lesion impairs novelty detection but spares pattern completion.

### D.05 CA3 recurrent collaterals — autoassociative attractor substrate
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.54]*

- **System:** CA3 pyramidal cells with extensive recurrent excitatory connections among themselves; LTP-modifiable.
- **Biological role:** Implements pattern completion: partial cue activates an attractor that converges on the full stored pattern. Marr (1971) autoassociator. Pathologically prone to seizure (runaway recurrent excitation).
- **Sim status:** partial — `RegionPathway` from CA3 to CA3 with `internal_density>0` would create the recurrent substrate, but no runner does this and no test verifies attractor convergence on cue completion.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** D.52.
- **Citation:** Kandel 6e Ch 54 pp 1342, 1360–1361.
- **Behavioral validation:** Partial-cue retrieval: stored "ABCDE" reactivated by partial "AB__" cue; lesion of CA3 recurrents impairs partial-cue recall but spares full-cue recall.

### D.06 Place cells — hippocampal spatial code (O'Keefe 1971)
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.55]*

- **System:** CA1 and CA3 pyramidal cells; one-or-few place fields per cell; rate code.
- **Biological role:** Represent the animal's location in an environment. Fields tile the environment via the population. Remap completely between environments (orthogonalization). Field size grades along dorsoventral axis (small dorsal → large ventral). Stable for days when animal attends to space.
- **Sim status:** partial — `g11_bg_runner --learned-perception --landmarks` produces place-cell-like activations from landmark-distance sensors via STDP, but cells are sensor-driven not allocentric, not validated for remapping, and the population is undifferentiated (no DG/CA3/CA1 distinction).
- **Cluster:** D primary, E (sensors).
- **Prerequisites:** D.52, learned-perception input layer.
- **Citation:** Kandel 6e Ch 54 pp 1361–1366, Figs 54-12, 54-13, 54-15.
- **Behavioral validation:** (a) Stable place fields across sessions in same environment; (b) global remapping when room changes; (c) larger fields ventrally; (d) place-field instability after CaMKII-inhibitor or NMDAR-NR1 KO.

### D.07 Grid cells — medial entorhinal cortex periodic spatial code (Mosers 2005)
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.56]*

- **System:** Medial EC layers II/III, dorsoventrally organized in 4 discrete modules with grid-spacing 40–100+ cm.
- **Biological role:** Each grid cell fires at vertices of a triangular lattice tiling the entire environment, regardless of context, landmarks, or darkness. Provides a context-invariant Cartesian metric. Updated by self-motion (path integration). Discrete modules suggest hierarchical positional code.
- **Sim status:** missing — no path-integration substrate, no periodic spatial firing. `--landmarks` perception is purely allocentric-from-vision, not self-motion-derived.
- **Cluster:** D primary, E (proprioception/self-motion sensors).
- **Prerequisites:** velocity/heading sensors, attractor-network or oscillatory-interference grid generator.
- **Citation:** Kandel 6e Ch 54 pp 1361–1364, Figs 54-12C, 54-13A.
- **Behavioral validation:** Hexagonal autocorrelation of firing field; persistence in darkness; modular grid-spacing distribution; coherent rescaling under environment deformation.

### D.08 Head-direction cells — allocentric heading code
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.57]*

- **System:** Presubiculum + medial EC; conjunctive grid+HD cells exist.
- **Biological role:** Fire selectively when animal faces a particular compass direction, independent of location. Internal compass; integrates angular vestibular velocity.
- **Sim status:** missing — no heading variable in `g11` agent state passed to neurons; agent does have heading via action history but it's not encoded by a population.
- **Cluster:** D primary, E secondary.
- **Prerequisites:** vestibular/heading sensor channel.
- **Citation:** Kandel 6e Ch 54 pp 1362–1364, Fig 54-14A.
- **Behavioral validation:** Polar tuning curve to head direction; cue-rotation aligns ensemble.

### D.09 Border / object-vector cells
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.58]*

- **System:** Medial EC; intermingled with grid + HD cells.
- **Biological role:** Border cells fire when animal approaches an environment edge (any wall) — anchor the grid metric to the local geometry. Object-vector cells encode distance+direction relative to specific landmarks.
- **Sim status:** partial — landmark sensors in `g11` provide distance + direction to landmarks, which is functionally object-vector encoding at the sensor stage; no grid alignment to borders since no grid cells.
- **Cluster:** D primary, E primary.
- **Prerequisites:** landmark sensor channel.
- **Citation:** Kandel 6e Ch 54 pp 1362–1364, Fig 54-14B.
- **Behavioral validation:** Border cell tracks deformed wall; object-vector cell follows displaced landmark.

### D.10 Speed cells — running-speed code
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.59]*

- **System:** Medial EC.
- **Biological role:** Firing rate ∝ running speed, location- and direction-independent. Together with HD cells, supplies grid network with the velocity vector for path integration.
- **Sim status:** missing — agent has speed but no neuron population encodes it.
- **Cluster:** D primary, E secondary.
- **Prerequisites:** speed sensor channel.
- **Citation:** Kandel 6e Ch 54 pp 1362–1364, Fig 54-14C.
- **Behavioral validation:** Linear firing-rate-vs-speed regression.

### D.11 Time cells — temporal sequence code in CA1
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.60]*

- **System:** CA1 pyramidal cells during structured delays (e.g., between trial events).
- **Biological role:** Sequentially active cells tile the delay period (analog of place cells in time). Underpin temporal organization of episodes.
- **Sim status:** missing — no sequential cell ensembles tied to delay periods.
- **Cluster:** D primary, G secondary (working memory bridge).
- **Prerequisites:** D.52.
- **Citation:** Kandel 6e Ch 54 (referenced; primary lit MacDonald/Eichenbaum 2011, Pastalkova 2008).
- **Behavioral validation:** Sequential firing tiling fixed delay; remapping when delay duration changes.

### D.12 Pattern separation — DG sparsifies overlapping inputs
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.61]*

- **System:** DG granule cells (very sparse activity, ~2–5%) receive convergent EC perforant input on a much larger population than EC layer II.
- **Biological role:** Marr "expansion recoding" — divergence onto a larger sparse population orthogonalizes similar inputs. Makes nearby episodes (same restaurant, two visits) distinct in the DG output. Adult neurogenesis in DG specifically supports fine pattern separation; new granule cells are particularly important.
- **Sim status:** missing — no DG region with high-density divergence + sparse-coding inhibitory drive. Adult neurogenesis not modeled (structural plasticity exists but is not progenitor-cell-based and not DG-localized).
- **Cluster:** D primary, J (sparse coding via inhibition), L (neurogenesis).
- **Prerequisites:** D.52 (trisynaptic), strong feedforward inhibition for sparsification.
- **Citation:** Kandel 6e Ch 54 pp 1357–1360.
- **Behavioral validation:** Discrimination of similar contexts in fear conditioning; selective DG silencing impairs near-pair discrimination but spares far-pair.

### D.13 Pattern completion — CA3 recurrents reconstruct full pattern from partial cue
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.62]*

- **System:** CA3 recurrent collateral network (autoassociator) + Schaffer output.
- **Biological role:** Partial or noisy retrieval cue activates a stored attractor that converges to the full pattern. Trade-off with separation: too much completion → confused episodes; too little → no generalization.
- **Sim status:** missing — see D.54. No runner builds a CA3 recurrent attractor with stored patterns and tests partial-cue retrieval.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** D.54 (recurrent CA3 collaterals + LTP).
- **Citation:** Kandel 6e Ch 54 pp 1357, 1360–1361.
- **Behavioral validation:** CA3-NMDA-KO mice fail when 2 of 4 visual cues are removed (partial cue) but succeed with all 4 cues.

### D.14 Engram cells — sparse activity-tagged ensembles store specific memories (Tonegawa)
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.63]*

- **System:** DG granule + CA1 pyramidal cells active during a specific experience; tagged via c-Fos/Arp-driven optogenetic actuators (ChR2 / iC1C2).
- **Biological role:** Reactivation of the tagged ensemble alone (in a neutral context) elicits the recall behavior (e.g., freezing). Inhibition of the ensemble blocks recall. Pairing context-A engram with shock in context-B creates a *false memory* of fear in A.
- **Sim status:** missing — no per-cell activity-history "tag" that can be later reactivated as a unit. Could be approximated by recording high-firing-rate neurons during a phase, then driving them with a stimulus channel during recall.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** ensemble identification + stimulus injection (already in `experiment/stimulus.py`).
- **Citation:** Kandel 6e Ch 54 pp 1357–1359, Fig 54-11.
- **Behavioral validation:** (a) Light-triggered freezing in neutral context after engram tag during conditioning; (b) optogenetic inhibition during natural cue blocks freezing; (c) false-memory paradigm.

### D.15 CA2 — social memory module
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.64]*

- **System:** Small region between CA3 and CA1; receives strong vasopressin/oxytocin input from hypothalamus; outputs to ventral CA1.
- **Biological role:** Recognition memory for conspecifics. Genetic CA2 silencing impairs social novelty recognition while sparing object/place memory. Vasopressin stimulation prolongs social memory durably.
- **Sim status:** missing — no agent currently has conspecifics; no neuropeptide modulators (NM framework supports declaring them but not currently used).
- **Cluster:** D primary, O (vasopressin / oxytocin neuromodulation).
- **Prerequisites:** social-agent extension; declared `NeuromodulatorConfig` for vasopressin/oxytocin with `target_type=excitability_drive` scoped to CA2.
- **Citation:** Kandel 6e Ch 54 p 1360.
- **Behavioral validation:** Three-chamber social novelty test; CA2 silencing.

### D.16 Place-field stability requires attention + D1/D5 dopamine + late-LTP
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.65]*

- **System:** CA1 place cells; D1/D5 receptors on pyramidal cells; PKA cascade for late-LTP.
- **Biological role:** Inattentive exploration → fields form but degrade in 3–6 hours. Attended exploration (e.g., goal-directed running) → fields stable for days. PKA-inhibitor transgene mimics inattentive phenotype.
- **Sim status:** partial — DA broadcast and adaptive DA exist; no PKA cascade, no late-LTP phase distinction. Stable-vs-unstable place-field-equivalent not yet a metric.
- **Cluster:** D primary, C (DA), J (late-LTP).
- **Prerequisites:** J.55, attention proxy (e.g., reward-engagement gating).
- **Citation:** Kandel 6e Ch 54 pp 1366–1367, Fig 54-16.
- **Behavioral validation:** D1 antagonist disrupts long-term place-field stability with intact short-term formation.

### D.17 Hippocampal remapping — orthogonal codes for distinct environments
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.66]*

- **System:** Population-level CA1/CA3 firing patterns.
- **Biological role:** Same neuron has uncorrelated place fields in different rooms ("global remapping") or same field with rate change ("rate remapping"). Minimizes interference among stored episodes. Sometimes triggered by minor sensory or motivational changes.
- **Sim status:** missing — single environment in `g11`; no test of cross-environment field decorrelation. Orthogonalization machinery (DG sparsification) not present.
- **Cluster:** D primary.
- **Prerequisites:** D.61 (pattern separation), multi-context runner.
- **Citation:** Kandel 6e Ch 54 p 1365, Fig 54-15.
- **Behavioral validation:** Population-vector correlation across environments ≈ 0; correlation within environment >> 0 across days.

### D.18 Theta rhythm — 4–12 Hz oscillation organizes encoding/retrieval phases
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.67]*

- **System:** Septal pacemaker → HC; modulates LFP and gates CA1 firing phase. Phase-precession of place cells (firing phase advances as animal traverses field).
- **Biological role:** Different theta phases bias toward encoding (peak) vs retrieval (trough); compresses behavioral sequences into compressed firing sequences within one theta cycle (theta sequences).
- **Sim status:** missing — no oscillatory pacemaker drive; no phase-precession; no theta-gated plasticity. Could be added via NM framework with sinusoidal `excitability_drive`.
- **Cluster:** D primary, I (oscillations), J (phase-gated plasticity).
- **Prerequisites:** rhythmic drive injection.
- **Citation:** Kandel 6e Ch 54 (referenced); also Buzsáki 2005.
- **Behavioral validation:** 4–12 Hz LFP power during locomotion; theta-phase precession of place cells.

### D.19 Sharp-wave ripples (SWRs) — replay in quiet wakefulness + NREM
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.68]*

- **System:** CA3 self-organized population burst → CA1 ripple (140–200 Hz) propagating to deep EC → neocortex. Coordinated with cortical slow oscillations + thalamic spindles during NREM.
- **Biological role:** Compressed (~20×) replay of waking firing sequences. Forward replay primes upcoming trajectories; reverse replay during reward consolidates path-to-reward. SWR disruption impairs spatial memory consolidation. The mechanism that drives hippocampal–cortical dialogue for systems consolidation.
- **Sim status:** partial — sleep-replay infrastructure exists in `bridge.py` (NREM scaffolding) but replay *content* is the named bottleneck; no SWR detection, no compressed sequential replay of waking trajectories, no coupling to cortical slow oscillation.
- **Cluster:** N primary, D primary, J (replay-driven LTP).
- **Prerequisites:** sequence storage during waking (theta sequences D.67), replay generator with compression, ripple-band oscillation in CA3.
- **Citation:** Kandel 6e Ch 54 pp 1365–1366, p 1250 (reference); also Buzsáki, Wilson/McNaughton replay literature.
- **Behavioral validation:** (a) Detected ripple bursts (140–200 Hz); (b) replay sequences match recent waking trajectories at 10–20× compression; (c) closed-loop ripple disruption during sleep impairs next-day spatial memory.

### D.20 Reactivation supports retrieval — cortical patterns recur during recall
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.69]*

- **System:** HC retrieval-time activity drives reactivation of original cortical encoding patterns; iEEG word-pair recall shows HC-temporal-cortex coupling that re-instantiates encoding-time patterns.
- **Biological role:** Retrieval is partial reinstatement of the encoding state, mediated by HC-cued reactivation of distributed cortical traces. Closes the encoding-storage-retrieval loop with the same circuit.
- **Sim status:** missing — no per-event encoding pattern stored as a labeled vector; no retrieval-cue-triggered reinstantiation.
- **Cluster:** D primary, G secondary.
- **Prerequisites:** D.63 (engram tagging primitive).
- **Citation:** Kandel 6e Ch 52 p 1299–1300.
- **Behavioral validation:** Encoding-pattern multivariate similarity peaks at retrieval relative to baseline (RSA / pattern-similarity analysis).

## Cluster E — additions

### E.01 Sensory coding — labeled-line / modality specificity
*[from Part IV — Perception (Ch 17-29); renumbered from E.50]*

- **System:** general sensory pathways
- **Biological role:** Each modality is carried by a distinct receptor class projecting through a private dorsal-column / spinothalamic / cranial pathway and thalamic relay (VPL, VPM, LGN, MGN). Modality identity is preserved by *which* line carries the signal, not by spike pattern alone (Müller's law).
- **Sim status:** partial — beacon and landmark are labeled lines into separate plastic cortical pathways. No modality multiplex.
- **Cluster:** E
- **Prerequisites:** any K.* transducer
- **Citation:** Kandel 6e Ch 17 p ~449–453
- **Behavioral validation:** electrical stimulation of a line evokes its modality regardless of stimulus.

### E.02 Receptive field — fundamental encoding unit
*[from Part IV — Perception (Ch 17-29); renumbered from E.51]*

- **System:** any sensory neuron (peripheral or central)
- **Biological role:** Each neuron responds to stimuli only in a restricted region of sensory space (skin patch, retinal area, frequency band). RF size + structure (excitatory/inhibitory subregions) determines selectivity. Hierarchically widens centrally.
- **Sim status:** partial — beacon sensors have implicit 8-direction RFs, but no sub-structure (no center-surround, no oriented subfields).
- **Cluster:** E
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 17 p ~451–458
- **Behavioral validation:** mapping responses across stimulus space.

### E.03 Population coding & vector averaging
*[from Part IV — Perception (Ch 17-29); renumbered from E.52]*

- **System:** cortical sensory + motor populations
- **Biological role:** A stimulus parameter (orientation, direction, location, frequency) is represented by the *distribution of activity* across many broadly-tuned neurons; downstream vector sum or Bayesian decoding extracts the value. Robust to noise and single-neuron loss.
- **Sim status:** partial — output motor uses pool-level voting (per-action populations), but sensory side has no proper population code over a continuous variable.
- **Cluster:** E (primary), G (decoding)
- **Prerequisites:** E.50
- **Citation:** Kandel 6e Ch 17 p ~458–464; Ch 25 (motion population)
- **Behavioral validation:** decode angle from population vector; tuning-curve width vs. discrimination acuity.

### E.04 Topographic / somatotopic / retinotopic / tonotopic maps
*[from Part IV — Perception (Ch 17-29); renumbered from E.53]*

- **System:** S1, V1, A1 cortices and their thalamic relays
- **Biological role:** Adjacent receptors map to adjacent cortical neurons, producing organized maps (homunculus, retinotopic V1 mirror, cochleotopic A1). Maps are warped by behavioral importance (cortical magnification — fingertips, fovea).
- **Sim status:** missing — declared regions have no spatial coordinate within them; connectivity is ER-random / patch-matrix, not topographic.
- **Cluster:** E
- **Prerequisites:** E.50
- **Citation:** Kandel 6e Ch 17 p ~460–462; Ch 19 p ~510–515; Ch 23 p ~559; Ch 28 p ~677
- **Behavioral validation:** systematic shift of RF center across cortical surface; magnification factor.

### E.05 Lateral inhibition & center-surround antagonism
*[from Part IV — Perception (Ch 17-29); renumbered from E.54]*

- **System:** retina (bipolar / ganglion), DCN, cochlear nucleus
- **Biological role:** Inhibitory horizontal connections sharpen contrast: center-on cells fire to bright spot in center + dark surround. Implements edge / change detection, decorrelates output.
- **Sim status:** partial — no explicit center-surround, but MSN lateral inhibition (cluster B) is the *same algorithmic motif* used for action WTA.
- **Cluster:** E (primary), B (algorithmic kin)
- **Prerequisites:** E.51
- **Citation:** Kandel 6e Ch 22 p ~588–593
- **Behavioral validation:** Mach bands; contrast-response with surround; difference-of-Gaussians fits.

### E.06 Parallel pathways (ON/OFF, magno/parvo/konio)
*[from Part IV — Perception (Ch 17-29); renumbered from E.55]*

- **System:** retina → LGN → V1
- **Biological role:** Visual stream split into ON-center / OFF-center channels (push-pull luminance), then magnocellular (motion, low-contrast, transient), parvocellular (form, color red-green), koniocellular (color blue-yellow). Channels remain segregated through LGN laminae and V1 input layers.
- **Sim status:** missing — no parallel sensory channels of any kind.
- **Cluster:** E
- **Prerequisites:** K.50, E.54
- **Citation:** Kandel 6e Ch 22 p ~590–595; Ch 23 p ~556–562
- **Behavioral validation:** selective lesion → loss of motion vs. color; contrast sensitivity functions.

### E.07 Ganglion cell encoding & spike-rate code at optic nerve
*[from Part IV — Perception (Ch 17-29); renumbered from E.56]*

- **System:** retinal ganglion cells (~20 types)
- **Biological role:** Final retinal output: each RGC type encodes one feature (sustained ON, transient OFF, direction, etc.) with center-surround RF. Sole channel from eye to brain — dimensionality reduction from ~125M photoreceptors to ~1M axons.
- **Sim status:** missing
- **Cluster:** E
- **Prerequisites:** K.50, E.54
- **Citation:** Kandel 6e Ch 22 p ~593–598
- **Behavioral validation:** RGC RF maps; type-specific tuning.

### E.08 V1 simple cells — oriented bar detectors
*[from Part IV — Perception (Ch 17-29); renumbered from E.57]*

- **System:** primary visual cortex layer 4
- **Biological role:** Orientation-tuned, position-specific receptive fields built from aligned LGN center-surround inputs (Hubel-Wiesel). Linear filter + threshold approximation — Gabor-like RFs. Foundation of all downstream form processing.
- **Sim status:** missing — no oriented filter / Gabor / V1 analog. The "perception arc" stops at beacon→cortex pathway.
- **Cluster:** E
- **Prerequisites:** E.51, E.54, E.55
- **Citation:** Kandel 6e Ch 22 p ~595–598; Ch 23 p ~559–564
- **Behavioral validation:** orientation tuning curve (~30° HWHH); spatial frequency tuning.

### E.09 V1 complex cells — phase-invariant orientation
*[from Part IV — Perception (Ch 17-29); renumbered from E.58]*

- **System:** V1 layers 2/3, 5
- **Biological role:** Pool simple-cell outputs to give orientation tuning that is spatially invariant within the RF (responds to a bar anywhere in RF). Builds first stage of position invariance.
- **Sim status:** missing
- **Cluster:** E
- **Prerequisites:** E.57
- **Citation:** Kandel 6e Ch 23 p ~561–566
- **Behavioral validation:** orientation tuning preserved as bar shifts within RF.

### E.10 Cortical columns & ocular dominance / orientation pinwheels
*[from Part IV — Perception (Ch 17-29); renumbered from E.59]*

- **System:** V1 (and S1 barrels, A1 frequency stripes)
- **Biological role:** Vertical columns share a feature (orientation, eye preference, frequency); horizontal organization tiles all values. Provides modular wiring substrate for plasticity and recurrent computations.
- **Sim status:** missing — declared regions are flat populations with no internal columnar organization.
- **Cluster:** E (primary), L (development)
- **Prerequisites:** E.57
- **Citation:** Kandel 6e Ch 23 p ~562–569; Ch 17 p ~462–464
- **Behavioral validation:** electrode penetration shows constant feature; tangential traverse cycles features.

### E.11 Color opponency (red-green, blue-yellow, luminance)
*[from Part IV — Perception (Ch 17-29); renumbered from E.60]*

- **System:** retina cone wiring → LGN parvo/konio → V1 blobs → V4
- **Biological role:** Cone signals recombined into opponent channels: L–M (red-green), S–(L+M) (blue-yellow), L+M (luminance). Explains color afterimages, unique hues, color constancy.
- **Sim status:** missing
- **Cluster:** E
- **Prerequisites:** K.50, E.55
- **Citation:** Kandel 6e Ch 22 p ~593–598; Ch 24 p ~578
- **Behavioral validation:** opponent afterimages; isoluminant chromatic gratings.

### E.12 Ventral "what" stream — object recognition (V1 → V2 → V4 → IT)
*[from Part IV — Perception (Ch 17-29); renumbered from E.61]*

- **System:** occipitotemporal cortex
- **Biological role:** Hierarchical buildup from edges (V1) → contours/textures (V2) → shapes/color (V4) → object/face/category (IT). Each stage increases RF size and feature complexity; IT cells fire to specific objects across viewpoint.
- **Sim status:** missing
- **Cluster:** E (primary), G (object→memory)
- **Prerequisites:** E.57, E.58
- **Citation:** Kandel 6e Ch 24 p ~568–587
- **Behavioral validation:** IT lesions → agnosia; face cells in FFA; viewpoint invariance.

### E.13 Dorsal "where/how" stream — spatial vision & action (V1 → MT → MST → PPC)
*[from Part IV — Perception (Ch 17-29); renumbered from E.62]*

- **System:** occipitoparietal cortex
- **Biological role:** Encodes location, motion, and visuomotor transformations for reaching/grasping. MT/V5 — motion direction; MST — optic flow / self-motion; PPC — egocentric coordinates. Lesions cause optic ataxia (can see but can't act).
- **Sim status:** missing — there is no separate "where" stream; navigation uses direct (gx, gy) cheats or beacon proxy.
- **Cluster:** E (primary), H (action), G (spatial WM)
- **Prerequisites:** E.57
- **Citation:** Kandel 6e Ch 24 p ~582–587; Ch 25 p ~593–600
- **Behavioral validation:** MT microstim biases motion judgment; PPC lesion → optic ataxia.

### E.14 Motion energy / direction selectivity (MT)
*[from Part IV — Perception (Ch 17-29); renumbered from E.63]*

- **System:** middle temporal area (V5/MT)
- **Biological role:** Spatiotemporal filters encode local image velocity (speed × direction). Pattern cells in MT integrate component motions to perceive global direction (aperture problem solution). Population vector decodes motion direction.
- **Sim status:** missing
- **Cluster:** E
- **Prerequisites:** E.57, E.52
- **Citation:** Kandel 6e Ch 23 p ~566; Ch 25 p ~593–598
- **Behavioral validation:** dot-coherence threshold ~5%; MT lesion → akinetopsia.

### E.15 Visual attention — top-down gain modulation
*[from Part IV — Perception (Ch 17-29); renumbered from E.64]*

- **System:** PPC + FEF → V4, MT, IT
- **Biological role:** Attention multiplies firing rates (~20-30%) of neurons tuned to attended location/feature, sharpens tuning, reduces noise correlations. Modeled as gain field on RF response.
- **Sim status:** partial — neuromodulator subsystem can apply scope-targeted gain modulation (synaptic_gain target) but no spatial-attention controller exists.
- **Cluster:** E (primary), G (top-down), C (NM gain)
- **Prerequisites:** E.61, E.62
- **Citation:** Kandel 6e Ch 25 p ~602–612
- **Behavioral validation:** attended-location RT advantage; firing-rate gain in V4.

### E.16 Saccadic suppression & active vision
*[from Part IV — Perception (Ch 17-29); renumbered from E.65]*

- **System:** SC, FEF, V1
- **Biological role:** During saccades (~3/sec), visual input is suppressed and corollary discharge updates spatial maps. Vision is built from snapshots stitched across fixations; perception is *active sampling*.
- **Sim status:** missing — no eye / gaze model; sensors are passive snapshots.
- **Cluster:** E (primary), H (oculomotor)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 25 p ~613–620
- **Behavioral validation:** stimuli flashed mid-saccade not perceived; remapping of RF before saccade.

### E.17 Auditory tonotopy & pitch coding (place + temporal)
*[from Part IV — Perception (Ch 17-29); renumbered from E.66]*

- **System:** A1 / superior temporal gyrus
- **Biological role:** Cortical A1 is tonotopic (low → high freq across surface). Pitch encoded by *both* place (which neurons fire) and temporal phase-locking (when they fire) up to ~3 kHz. Bandpass tuning ~Q10dB ≈ 5.
- **Sim status:** missing — no auditory front-end or tonotopic cortical region.
- **Cluster:** E (primary), K (cochlea)
- **Prerequisites:** K.53, K.54
- **Citation:** Kandel 6e Ch 28 p ~660–675
- **Behavioral validation:** A1 frequency map; pitch discrimination Weber fraction ~0.2%.

### E.18 Sound localization — ITD / ILD circuits
*[from Part IV — Perception (Ch 17-29); renumbered from E.67]*

- **System:** superior olivary complex (MSO/LSO) → IC
- **Biological role:** MSO neurons act as coincidence detectors for interaural time differences (low freq, ~10 µs precision) using delay lines (Jeffress-type). LSO compares interaural intensity differences (high freq). Build a map of azimuthal sound location.
- **Sim status:** missing — no binaural / time-difference computation.
- **Cluster:** E (primary), I (sub-ms timing)
- **Prerequisites:** K.52, K.53
- **Citation:** Kandel 6e Ch 28 p ~659–667
- **Behavioral validation:** spatial-tuning curves in ICX; psychophysical MAA ~1°.

### E.19 Olfactory glomerular map & combinatorial code
*[from Part IV — Perception (Ch 17-29); renumbered from E.68]*

- **System:** olfactory bulb glomeruli, mitral/tufted cells
- **Biological role:** All OSNs expressing the same OR converge onto ~1-2 glomeruli — converts combinatorial OR code into a stereotyped 2D activity map. Mitral cells with lateral granule-cell inhibition decorrelate odor representations.
- **Sim status:** missing — `OLFACTORY_BULB` profile exists but receives no odor input and has no glomerular wiring.
- **Cluster:** E (primary), K (transduction)
- **Prerequisites:** K.59
- **Citation:** Kandel 6e Ch 29 p ~720–725
- **Behavioral validation:** stereotyped glomerular map across animals; pattern decorrelation index.

### E.20 Somatosensory cortex (S1) — area 3b body map & feature columns
*[from Part IV — Perception (Ch 17-29); renumbered from E.69]*

- **System:** S1 (areas 3a, 3b, 1, 2)
- **Biological role:** Area 3b: somatotopic, RF small, builds tactile-feature columns (orientation of edges on skin, like V1 for touch). Areas 1 and 2 build progressively more complex shape/texture/motion representations of object on skin.
- **Sim status:** missing
- **Cluster:** E
- **Prerequisites:** K.56, E.53
- **Citation:** Kandel 6e Ch 19 p ~498–520
- **Behavioral validation:** homunculus mapping; tactile orientation tuning.

### E.21 Constructive / inferential perception (Helmholtz unconscious inference)
*[from Part IV — Perception (Ch 17-29); renumbered from E.70]*

- **System:** all cortical perceptual streams
- **Biological role:** Perception is *inference under prior* — cortex resolves ambiguous stimuli (illusions, occlusion, lighting) by combining sensory data with learned priors. Bayesian framing; explains illusions, ambiguous figures, perceptual filling-in.
- **Sim status:** missing — sim has no generative / predictive perceptual stage.
- **Cluster:** E (primary), G (top-down)
- **Prerequisites:** E.61
- **Citation:** Kandel 6e Ch 21 p ~556–574
- **Behavioral validation:** demonstration illusions (Kanizsa, Adelson checker); cue-combination experiments (Ernst & Banks).

### E.22 Multisensory integration
*[from Part IV — Perception (Ch 17-29); renumbered from E.71]*

- **System:** superior colliculus, STS, parietal cortex
- **Biological role:** Bimodal neurons combine inputs from multiple modalities (visual + auditory + tactile) with super-additive enhancement when sub-threshold and spatially aligned, sub-additive when redundant. Drives orienting and unified percepts (ventriloquism, McGurk).
- **Sim status:** missing — only one modality (location) at a time.
- **Cluster:** E
- **Prerequisites:** ≥2 of E.50 lines
- **Citation:** Kandel 6e Ch 17 p ~462; Ch 25 p ~611
- **Behavioral validation:** super-additive SC neuron response; ventriloquism shift.

---

## Cluster C — Pain modulation / neuromodulation overlap

## Cluster F — additions

### F.01 Purkinje cell — sole output of cerebellar cortex
*[from Part V — Movement (Ch 30-39); renumbered from F.50]*

- **System:** cerebellar cortex molecular + Purkinje layer; massive flat dendritic tree; tonic inhibitory output to deep cerebellar nuclei (DCN).
- **Biological role:** Receives ~150,000 parallel-fiber inputs (PF) plus a single climbing fiber (CF). Tonically fires simple spikes 30–80 Hz; PF input modulates rate. Inhibits DCN; net cerebellar output emerges by disinhibition of DCN as PC activity decreases.
- **Sim status:** **partial** — HH preset `HH_CEREBELLAR_PURKINJE` exists in `sim/enums.py` but no runner instantiates it as part of a circuit.
- **Cluster:** F primary.
- **Prerequisites:** F.51, F.52, F.53.
- **Citation:** Kandel 6e Ch 37 p 918–926.
- **Behavioral validation:** Tonic 30–80 Hz simple spikes when PF input is balanced; CF event evokes characteristic complex spike; brief pause after CF.

### F.02 Granule cell + parallel fiber — divergent code
*[from Part V — Movement (Ch 30-39); renumbered from F.51]*

- **System:** granular layer (>50 billion granule cells in human cerebellum); axon bifurcates into long parallel fibers (PF) running along the folium, perpendicular to PC dendrites.
- **Biological role:** Receives ~4 mossy-fiber inputs each; sparse, high-dimensional combinatorial code. PF passes through dendritic trees of thousands of PCs, contacting each PC with ~1 synapse. Marr-Albus expansion-recoding hypothesis.
- **Sim status:** **partial** — HH preset `HH_CEREBELLAR_GRANULE` exists; no PF wiring code.
- **Cluster:** F primary.
- **Prerequisites:** F.52 (mossy fiber input).
- **Citation:** Kandel 6e Ch 37 p 918–921.
- **Behavioral validation:** Sparse activity (~1–5% active at any moment); sparse-input vs dense-output recoding (distinct mossy patterns → orthogonal PF patterns).

### F.03 Mossy-fiber afferent system — pontine, spinal, vestibular input
*[from Part V — Movement (Ch 30-39); renumbered from F.52]*

- **System:** pontine nuclei (cerebro-cerebellar relay), spinocerebellar tracts, vestibular nuclei → granule layer rosettes.
- **Biological role:** Conveys cortical efference copy + proprioceptive + vestibular state. Excitatory glutamatergic. Branches to DCN (collateral) and granule cells. Each MF excites ~400 granule cells.
- **Sim status:** **missing** — no mossy-fiber pathway in any runner.
- **Cluster:** F primary.
- **Prerequisites:** F.51.
- **Citation:** Kandel 6e Ch 37 p 918–920.
- **Behavioral validation:** Step input (e.g. limb perturbation) drives transient burst across granule layer.

### F.04 Climbing fiber — inferior olive single-cell teaching signal
*[from Part V — Movement (Ch 30-39); renumbered from F.53]*

- **System:** inferior olive (IO) → contralateral cerebellar cortex; one CF per Purkinje cell, wraps dendrites with ~hundreds of synapses.
- **Biological role:** Fires sparsely (~1 Hz) but each spike triggers a Purkinje complex spike (Ca²⁺ plateau). Encodes motor errors / unexpected events. CF coactivation with PF triggers PF→PC LTD — this IS the cerebellar learning rule.
- **Sim status:** **missing** — IO HH preset `HH_INFERIOR_OLIVE` exists, but no 1:1 CF wiring, no PF×CF coincidence-gated plasticity.
- **Cluster:** F primary; J (plasticity) secondary.
- **Prerequisites:** F.50, F.51, F.54.
- **Citation:** Kandel 6e Ch 37 p 920–925.
- **Behavioral validation:** Unexpected perturbation → IO complex spike rate ↑; perturbation becomes predictable → IO rate returns to baseline.

### F.05 PF→PC LTD (Marr-Albus-Ito) — sign-flipped, CF-gated plasticity
*[from Part V — Movement (Ch 30-39); renumbered from F.54]*

- **System:** parallel-fiber → Purkinje cell glutamatergic synapse; postsynaptic mGluR1 + Ca²⁺ from CF.
- **Biological role:** Coincident PF activity and CF complex spike → long-term depression of that PF synapse. Reduces PC simple-spike response to that input. Reduces PC inhibition of DCN → behavior gets stronger / corrected. This is the canonical motor-learning rule.
- **Sim status:** **missing** — `fused_stdp_weight_update` is Hebbian and pre-post-timing-based, not CF-gated. Would need a new fused kernel `fused_pf_pc_ltd` taking (PF spike, CF complex spike) → ΔW < 0, with a separate slow LTP for unpaired PF.
- **Cluster:** F primary; J secondary.
- **Prerequisites:** F.50–F.53.
- **Citation:** Kandel 6e Ch 37 p 922–925 (Marr 1969, Albus 1971, Ito).
- **Behavioral validation:** Eyeblink conditioning (F.57) — paired CS-US → blink prediction; unpaired → no learning.

### F.06 Deep cerebellar nuclei (DCN) — final cerebellar output
*[from Part V — Movement (Ch 30-39); renumbered from F.55]*

- **System:** dentate, interposed, fastigial nuclei; receive inhibitory PC input + excitatory MF/CF collaterals.
- **Biological role:** Tonic firing 40 Hz; PC inhibition silences DCN; release of PC silences DCN releases excitatory drive to thalamus / red nucleus / brainstem. Net cerebellar effect is via DCN disinhibition.
- **Sim status:** **missing** — no DCN region.
- **Cluster:** F primary.
- **Prerequisites:** F.50.
- **Citation:** Kandel 6e Ch 37 p 911–917.
- **Behavioral validation:** PC inhibition → DCN pause → downstream effector burst.

### F.07 Forward / inverse internal models — predictive control
*[from Part V — Movement (Ch 30-39); renumbered from F.56]*

- **System:** cerebro-cerebellar recurrent loops (Strick); Purkinje activity correlates with predicted sensory consequences of motor commands.
- **Biological role:** Cerebellum hosts internal models that predict the sensory consequences of efference copy (forward model) and / or compute motor commands needed to achieve a desired sensory state (inverse model). Used to cancel self-generated input (e.g. tickling), pre-emptively counter interaction torques.
- **Sim status:** **missing** — no efference-copy pathway, no forward-model module.
- **Cluster:** F primary; G (PFC working memory) secondary.
- **Prerequisites:** F.50–F.55.
- **Citation:** Kandel 6e Ch 30 p 720–724 (Box 30-1) and Ch 37 p 921–924.
- **Behavioral validation:** Self-generated stimulus (subject moves) → attenuated cerebellar response; passive stimulus same magnitude → full response (Cullen vestibular paradigm).

### F.08 Eyeblink classical conditioning — canonical cerebellar learning task
*[from Part V — Movement (Ch 30-39); renumbered from F.57]*

- **System:** tone (CS) via mossy fiber → granule → PF; air puff (US) via climbing fiber from IO; blink output via interposed nucleus → red nucleus → motor.
- **Biological role:** Pavlovian timing-precise CR. After paired CS-US trials, animal blinks slightly before US onset. PF→PC LTD on CS-driven PF synapses + DCN plasticity reproduces this. Deep-nuclei lesion abolishes acquired blink; cortical lesion reduces precise timing.
- **Sim status:** **missing** — no canonical task harness; would be the natural smoke-test for cluster F closure.
- **Cluster:** F primary; J, O (reward analog) secondary.
- **Prerequisites:** F.50–F.55, F.54.
- **Citation:** Kandel 6e Ch 37 p 928–932.
- **Behavioral validation:** Acquisition curve (probability and timing of CR vs trials); CS-alone trials probe CR without US; cerebellar lesion abolishes CR.

### F.09 VOR adaptation — gaze stabilization gain learning
*[from Part V — Movement (Ch 30-39); renumbered from F.58]*

- **System:** vestibulocerebellum (flocculus); vestibular MF input + retinal-slip CF input from IO.
- **Biological role:** Vestibulo-ocular reflex keeps retinal image stable during head motion. Magnifying / minimizing glasses produce retinal slip → IO complex spikes → PF→PC LTD → adjusted VOR gain over hours. Floccular lesion abolishes adaptation but spares baseline VOR.
- **Sim status:** **missing**.
- **Cluster:** F primary.
- **Prerequisites:** F.50–F.55.
- **Citation:** Kandel 6e Ch 37 p 925–928.
- **Behavioral validation:** Open-loop VOR gain measured before / after sustained slip → asymptotic gain change in correct direction.

### F.10 Cerebellar timing & motor coordination — interaction-torque compensation
*[from Part V — Movement (Ch 30-39); renumbered from F.59]*

- **System:** cerebellar cortex + interposed nucleus → red nucleus / motor cortex.
- **Biological role:** Cerebellum predicts interaction torques across joints (shoulder torque from elbow swing, etc.) and pre-corrects. Lesion → dysmetria, decomposition of movement, action tremor (Holmes signs).
- **Sim status:** **missing** — no multi-joint plant, no pre-correction.
- **Cluster:** F primary; H secondary.
- **Prerequisites:** F.50–F.55, H.54.
- **Citation:** Kandel 6e Ch 37 p 909–917.
- **Behavioral validation:** Two-joint reach: control compensates interaction torque; "lesioned" model shows shoulder perturbation when only elbow is commanded.

### F.11 Cerebellum-prefrontal nonmotor loops — cognitive role
*[from Part V — Movement (Ch 30-39); renumbered from F.60]*

- **System:** lateral cerebellum ↔ prefrontal / parietal cortex via pontine and thalamic relays (Strick rabies tracing).
- **Biological role:** ~half of cerebellar volume connects to nonmotor cortex. Damage → executive, language, visuospatial, affective deficits. Suggests cerebellar microcircuit performs a generic "predictor" computation applied to whatever input it receives.
- **Sim status:** **missing**; would compose naturally with PFC region once F.50–F.55 exist.
- **Cluster:** F primary; G secondary.
- **Prerequisites:** F.50–F.55, PFC region (existing).
- **Citation:** Kandel 6e Ch 37 p 911–913.
- **Behavioral validation:** Lateral-cerebellum lesion impairs sequence prediction in non-motor working-memory task.

---

## Cluster A — Closed BG action-selection loop (project flagship)

## Cluster G — additions

### G.03 Object-based attention & feature binding
*[from Part IV — Perception (Ch 17-29); renumbered from G.50]*

- **System:** parietal + ventral cortex, gamma-band coherence
- **Biological role:** Once an object is selected, attention spreads across all features belonging to it (not just spatial location). Hypothesized binding via gamma synchrony or a parietal "saliency map" indexing scattered feature patches.
- **Sim status:** missing — no binding mechanism; the gamma-FS profile exists but is not deployed for perceptual binding.
- **Cluster:** G (primary), E
- **Prerequisites:** E.61, E.64
- **Citation:** Kandel 6e Ch 25 p ~607–615
- **Behavioral validation:** same-object advantage; feature-binding errors under attention load (illusory conjunctions).

### G.04 Predictive / corollary discharge for perceptual stability
*[from Part IV — Perception (Ch 17-29); renumbered from G.51]*

- **System:** motor → sensory cortex (efference copy)
- **Biological role:** Self-generated movement (saccade, head turn) produces an efference copy that updates sensory maps so the world appears stable. Failure → vertigo, oscillopsia, schizophrenia hallucinations of "unowned" actions.
- **Sim status:** missing — no efference copy from motor to sensory pathways; no predictive pre-saccadic remapping.
- **Cluster:** G (primary), E, H
- **Prerequisites:** E.65
- **Citation:** Kandel 6e Ch 25 p ~615–620; Ch 29 (sniff-coupled olfaction) ~723
- **Behavioral validation:** push-eye-with-finger → world appears to move (no efference copy); active-vs-passive head movement perception.

---

## Cluster J — Plasticity rules touched

### G.05 Posterior parietal cortex (PPC) — spatial planning, reach intention
*[from Part V — Movement (Ch 30-39); renumbered from G.50]*

- **System:** PPC areas LIP (saccades), PRR (reach), AIP (grasp).
- **Biological role:** Encodes spatial goal in body / world coordinates; persists across delay periods (planning); receives visual input, sends to PMd / FEF.
- **Sim status:** **missing** — no spatial parietal module. Goal-cells region in `g11_bg_runner` is closer to PPC than PFC, but currently labeled "goal" and routed through PFC.
- **Cluster:** G primary; H secondary.
- **Prerequisites:** K.* (vision), G.51.
- **Citation:** Kandel 6e Ch 34 p 826–832.
- **Behavioral validation:** Delay activity in PPC predicts upcoming reach target; lesion disrupts visually guided reach in extrapersonal space.

### G.06 PFC working memory — sustained delay-period activity
*[from Part V — Movement (Ch 30-39); renumbered from G.51]*

- **System:** dorsolateral PFC; recurrent excitation supports persistent firing across delay; modulated by D1.
- **Biological role:** Holds task-relevant information online when stimulus is absent. Substrate for rule-following, task-set maintenance, gating of responses.
- **Sim status:** **partial** — PFC region added 2026-04-27 (60 recurrent neurons, plastic `goal_cells → PFC → cortex_X`). Provides working-memory-like sustained activity for goal context. Sum 4.41 (25% over baseline) when combined with hippo + curriculum.
- **Cluster:** G primary; A secondary.
- **Prerequisites:** I.*, J.*; recurrent excitation in `RegionPathway`.
- **Citation:** Kandel 6e Ch 34 p 827–842.
- **Behavioral validation:** Sustain firing during delay > baseline; lesion impairs delayed-match-to-sample.

### G.07 Pre-SMA / SMA medial premotor — internally generated sequences
*[from Part V — Movement (Ch 30-39); renumbered from G.52]*

- **System:** medial wall of frontal cortex; rostral pre-SMA = abstract sequence; caudal SMA-proper = movement sequence.
- **Biological role:** Activity precedes self-initiated movement (Bereitschaftspotential); damage spares stimulus-cued action but disrupts internally generated sequences; also implicated in action timing.
- **Sim status:** **missing** — see H.68.
- **Cluster:** G primary; H secondary.
- **Prerequisites:** G.51, H.65.
- **Citation:** Kandel 6e Ch 34 p 822–828.
- **Behavioral validation:** Self-paced tap task — pre-movement readiness potential; SMA lesion disrupts.

---

## Cluster K — Sensory transduction (motor-relevant)

### G.08 Working memory in prefrontal cortex — persistent activity for active maintenance
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.50]*

- **System:** Dorsolateral and ventrolateral PFC; supported by parietal verbal/visuospatial stores and ascending DA from VTA/SNc.
- **Biological role:** Maintains transient, goal-relevant representations across delays (seconds). DMS-task PFC neurons hold "what" (object), "where" (location), and "what+where" conjunctions during the delay period (Rainer/Asaad/Miller 1998). Subsystems: phonological loop (rehearsal in Broca's area), visuospatial sketchpad, executive control.
- **Sim status:** partial — `sim/regions.py` `BrainRegion` with `internal_density` recurrent connectivity creates a PFC region used as flagship working-memory pool (60 recurrent neurons, plastic `goal_cells → PFC → cortex_{N,E,S,W}`, gated by `pfc_pathways`). Single-compartment, no DMS-style delay-period mixed selectivity, no explicit "what vs where" coding.
- **Cluster:** G primary, O secondary (DA-modulated rehearsal).
- **Prerequisites:** A.* (BG cascade), C.* (DA gating).
- **Citation:** Kandel 6e Ch 52 pp 1292–1294.
- **Behavioral validation:** Delay-period firing above baseline, content-specific (object vs location vs conjunction); D1-DA dependence of stable maintenance.

### G.09 Imagination / future simulation as constructive memory
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.51]*

- **System:** Default-mode core: medial PFC, posterior cingulate / precuneus, retrosplenial, lateral parietal, lateral temporal, HC.
- **Biological role:** Recombines stored elements to simulate future events. Same network active for "remember last beach trip" and "imagine next beach trip" (Schacter/Addis/Buckner). Adaptive: HC dysfunction degrades both episodic recall *and* novel-scene imagination.
- **Sim status:** missing — sleep-replay can re-run trajectories but no constructive recombination of items into novel hypotheticals; no DMN architecture.
- **Cluster:** G primary, D secondary.
- **Prerequisites:** D.50, D.51, N.* (offline replay).
- **Citation:** Kandel 6e Ch 52 pp 1300–1302.
- **Behavioral validation:** HC-amnesic patients fail "imagine new picnic" task with markedly impoverished scene detail.

### G.10 Language as hierarchical symbolic system — phonemes / morphemes / words / syntax
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.52]*

- **System:** Distributed cortical network, predominantly left hemisphere; anterior (production) and posterior (comprehension) zones plus connecting fasciculi.
- **Biological role:** Finite phoneme inventory combinable into infinitely many morphemes/words/sentences via syntactic rules. Each language has its own phonotactic and syntactic constraints; children acquire them universally by ~age 3.
- **Sim status:** missing — no symbol grounding, no syntactic structure, no language production/comprehension.
- **Cluster:** G primary.
- **Prerequisites:** symbol-grounding extension; well outside current scope.
- **Citation:** Kandel 6e Ch 55 pp 1370–1372.
- **Behavioral validation:** Phonotactic discrimination (e.g., /zb/ illegal in English).

### G.11 Dual-stream model of language — dorsal sensorimotor + ventral semantic
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.53]*

- **System:** Dorsal stream: posterior superior temporal → arcuate fasciculus → Broca's area (sensorimotor mapping for speech production). Ventral stream: superior + middle temporal → semantic interface (sound→meaning). Hickok & Poeppel.
- **Biological role:** Dorsal damage → Broca's / conduction aphasia (production + repetition deficits). Ventral damage → Wernicke's / transcortical sensory aphasia (comprehension deficits, fluent paraphasic speech).
- **Sim status:** missing.
- **Cluster:** G primary, H (motor for production), E (audition for comprehension).
- **Prerequisites:** auditory + motor systems with semantic mapping.
- **Citation:** Kandel 6e Ch 55 pp 1380–1387.
- **Behavioral validation:** Aphasia syndrome dissociation tables (Table 55-2): Broca speech labored/agrammatic; Wernicke fluent but paraphasic; conduction selectively repetition-impaired.

### G.12 Broca's area — speech production + grammatical processing + sensorimotor mapping
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.54]*

- **System:** Left posterior inferior frontal gyrus (pars opercularis + triangularis); subjacent white matter; insula.
- **Biological role:** Maps stored auditory word-forms to motor articulation; supports comprehension of grammatically complex (non-canonical) sentences. Damage → labored, agrammatic speech, retained noun selection, lost function-word/verb use, repetition deficit.
- **Sim status:** missing.
- **Cluster:** G primary, H secondary.
- **Prerequisites:** language model substrate.
- **Citation:** Kandel 6e Ch 55 pp 1382–1384, Fig 55-6.
- **Behavioral validation:** "The girl that the boy is chasing is tall" comprehension fails (grammar-dependent); "The apple the girl ate was green" succeeds (semantically constrained).

### G.13 Wernicke's area — auditory-to-semantic mapping
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.55]*

- **System:** Left posterior superior temporal gyrus + middle temporal gyrus; ventral-stream comprehension.
- **Biological role:** Selects words matching intended meaning; phonemic and semantic paraphasias result from selection failures (e.g., "headman" for "president"). Speech remains fluent and prosodic but unintelligible.
- **Sim status:** missing.
- **Cluster:** G primary, E secondary.
- **Prerequisites:** semantic memory store.
- **Citation:** Kandel 6e Ch 55 pp 1384–1385.
- **Behavioral validation:** Fluent paraphasic output + comprehension deficit; selective conduction-aphasia spares fluency but loses repetition (arcuate fasciculus damage).

### G.14 Left-hemisphere dominance — even for sign language
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.56]*

- **System:** Left-hemisphere language network active for spoken, written, AND signed language in deaf signers; bilingual learners' L2 cortical organization depends on age of acquisition (early L2 colocates with L1; late L2 recruits adjacent territory).
- **Biological role:** Left hemisphere is specialized for *linguistic* structure regardless of modality. Right hemisphere supports paralinguistic prosody (emotional intonation) but linguistic tone (Mandarin, Thai) is left-lateralized.
- **Sim status:** missing.
- **Cluster:** G primary.
- **Prerequisites:** lateralized cortex regions.
- **Citation:** Kandel 6e Ch 55 pp 1382–1383, Fig 55-5.
- **Behavioral validation:** fMRI: Chinese speakers show left planum temporale activation for Chinese tones, Thai speakers for Thai tones; double dissociation by language experience.

---

## Ch 56 — Decision-Making and Consciousness

### G.15 Signal-detection decision rule — threshold on noisy evidence
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.57]*

- **System:** Sensory cortex supplies evidence; decision area applies criterion. Conceptual framework (Weber, Fechner, Green & Swets); ROC curve dissociates sensitivity (signal/noise separation) from policy (criterion).
- **Biological role:** Yes/no detection: compare evidence sample against criterion; criterion encodes prior probability + cost of hits vs misses vs false alarms. Holds for perceptual *and* value-based judgments.
- **Sim status:** partial — BG cascade implements something close to a criterion (winner-take-all over per-action striatum). No explicit ROC analysis.
- **Cluster:** G primary, A secondary.
- **Prerequisites:** A.* (BG cascade).
- **Citation:** Kandel 6e Ch 56 pp 1393–1395, Fig 56-1.
- **Behavioral validation:** ROC curve from yes/no responses across signal strengths; criterion shifts with payoff structure.

### G.16 Drift-diffusion / bounded evidence accumulation — speed–accuracy trade-off
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.58]*

- **System:** Sensory cortex (e.g., MT for motion) supplies noisy momentary evidence; accumulator integrates difference (right − left) over time; decision terminates when accumulator hits ±bound.
- **Biological role:** Explains reaction time distributions, accuracy as f(coherence), and confidence. Bound height trades speed for accuracy. Two anti-correlated accumulators (one per choice) terminate at first-bound-crossing. The dominant model for perceptual decisions in primates.
- **Sim status:** partial — BG cascade with motor-output thresholding is *functionally equivalent to* a bounded accumulator (cortex_X firing rate accumulates DA-modulated evidence; threshold applied at thalamus→motor). No explicit RT analysis or coherence-vs-accuracy curve exists.
- **Cluster:** G primary, A primary, O (DA scaling of drift rate).
- **Prerequisites:** none — already implicit in cascade.
- **Citation:** Kandel 6e Ch 56 pp 1399–1404, Figs 56-6, 56-8.
- **Behavioral validation:** RT distributions and choice accuracy fit by drift-diffusion model with one drift parameter per coherence + shared bound.

### G.17 LIP / parietal accumulator — neural correlate of decision variable
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.59]*

- **System:** Lateral intraparietal area (LIP), medial intraparietal (MIP for reach), anterior intraparietal (AIP for grasp); also dorsolateral PFC. Persistent ramping firing during evidence accumulation; threshold reached just before saccade.
- **Biological role:** Firing rate ≈ accumulated logLR for the choice associated with each cell's response field. Ramps faster when evidence stronger; reaches common bound right before action; the unchosen-direction cell's firing terminates without crossing bound.
- **Sim status:** partial — `cortex_X` per-action pools in BG cascade ramp toward action threshold; not yet validated against ramp-with-coherence signature, no per-trial decoding.
- **Cluster:** G primary, A primary.
- **Prerequisites:** A.* (BG cascade), runner with random-dot-coherence task.
- **Citation:** Kandel 6e Ch 56 pp 1402–1404, Fig 56-7, 56-8.
- **Behavioral validation:** Mean firing rate ramps with coherence; common-threshold-at-saccade signature; dissociates LIP (decision) from MT (sensory).

### G.18 Probabilistic reasoning from symbols — logLR accumulation in LIP
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.60]*

- **System:** Same LIP accumulator, but evidence comes from learned shape→reward associations rather than sensory motion. Each shape contributes its known logLR additively.
- **Biological role:** Demonstrates that LIP's accumulator is not specific to perceptual evidence; it integrates *any* evidence weighted by reliability. Foundation for inferential / Bayesian-like reasoning.
- **Sim status:** missing — no symbol-with-learned-reliability primitive.
- **Cluster:** G primary, O (reward-driven shape→logLR learning).
- **Prerequisites:** G.59, learned-reliability mechanism.
- **Citation:** Kandel 6e Ch 56 pp 1404–1407, Fig 56-9.
- **Behavioral validation:** Choice probability as logistic function of summed shape-logLRs; LIP firing rate tracks running summed logLR.

### G.19 Affordance theory — knowledge as provisional commitment to action (Gibson / Shadlen)
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.61]*

- **System:** Association cortices (parietal, temporal, prefrontal) where source modality (vision, audition) connects to target action system (gaze, reach, grasp).
- **Biological role:** Persistent activity in association areas represents potential behaviors ("I might look there", "I might grasp this") rather than passive sensory features. "Knowing" = provisional decision to embrace a proposition. Reframes consciousness as a decision-like commitment with temporal thickness.
- **Sim status:** not-applicable — philosophical framework, not directly implementable, but informs how action-policy populations should be interpreted.
- **Cluster:** G primary.
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 56 pp 1409–1412, Box 56-1.
- **Behavioral validation:** Hemineglect (right parietal lesion) — patient does not "interrogate" left field, no awareness of deficit (vs hemianopsia where patient knows visual loss).

### G.20 Consciousness — arousal level + content access
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from G.62]*

- **System:** Two dissociable phenomena: (1) level/arousal — brainstem reticular + thalamic systems gate sleep/wake/anesthesia; (2) content/access — selective gating of specific representations to a global workspace (frontoparietal network, Dehaene).
- **Biological role:** Through the decision-making lens: conscious access ≈ a representation winning the threshold-crossing event in the global accumulator, becoming reportable. Nonconscious processing influences behavior without crossing this threshold.
- **Sim status:** missing — no global-workspace gate; no arousal axis (sleep states are NREM-only, no full level-of-consciousness modulation).
- **Cluster:** G primary, N (arousal/sleep), O (NM tone).
- **Prerequisites:** broadcast/gating substrate, arousal NM (NE).
- **Citation:** Kandel 6e Ch 56 pp 1411–1413.
- **Behavioral validation:** Subliminal-priming paradigms; binocular rivalry; report-vs-no-report decoding.

## Cluster H — additions

### H.01 Motor unit (α-motoneuron + muscle fibers it innervates) — Sherrington 1925
*[from Part V — Movement (Ch 30-39); renumbered from H.50]*

- **System:** spinal/brainstem motor nucleus → ventral root → peripheral nerve → skeletal muscle fibers.
- **Biological role:** Elementary unit by which the nervous system controls force. Innervation numbers vary 5 (eye) to ~1,800 (gastrocnemius); small numbers give fine grading. Each muscle fiber is innervated by exactly one α-motoneuron in mature vertebrates; one α-MN drives many fibers via NMJ.
- **Sim status:** **missing** — `sim/regions.py` defines a `motor_X` region as an abstract spike-emitting pool but no muscle fibers, no innervation map, no force output. Closing requires a `MuscleFiber` array attached to each motor pool plus a force-summation readout layer.
- **Cluster:** H primary; M (NMJ) secondary.
- **Prerequisites:** I.* (channels), J.* (synapses); H.51 (muscle force).
- **Citation:** Kandel 6e Ch 31 p 737–740.
- **Behavioral validation:** Single-fiber EMG twitch on suprathreshold MN spike; compound EMG amplitude proportional to active MN count.

### H.02 Twitch contraction & tetanic summation — force as f(spike rate)
*[from Part V — Movement (Ch 30-39); renumbered from H.51]*

- **System:** muscle fiber sarcomere contraction in response to AP-evoked Ca²⁺ release.
- **Biological role:** Single AP → twitch (peak force at contraction time); train of APs → unfused or fused tetanus. Peak tetanic force is a sigmoid in stimulation rate; saturation rate is lower for slow-twitch units. Force grading = recruitment + rate coding.
- **Sim status:** **missing** — motor pools currently emit spikes only. Would need a per-pool low-pass filter (twitch kernel ~50–100 ms slow / ~20 ms fast) feeding a saturating force readout.
- **Cluster:** H primary.
- **Prerequisites:** H.50.
- **Citation:** Kandel 6e Ch 31 p 739–742.
- **Behavioral validation:** Plot force vs MN firing rate → sigmoid; staircase summation as ISI shortens.

### H.03 Henneman size principle (orderly recruitment) — small → large
*[from Part V — Movement (Ch 30-39); renumbered from H.52]*

- **System:** spinal motor pool with mixed slow / fast / fatigable units.
- **Biological role:** Increasing synaptic drive recruits motor units in order of increasing axon size (and force, fatigability). Slow type-I units fire first; fast type-II later. Provides smooth gradation and energy economy at low force.
- **Sim status:** **missing** — current motor pools are homogeneous Izhikevich/HH presets without size-dependent input thresholds. Trivially closed with per-neuron threshold heterogeneity sorted by motor-unit "size" parameter.
- **Cluster:** H primary; L (developmental sorting) secondary.
- **Prerequisites:** H.50.
- **Citation:** Kandel 6e Ch 31 p 745–747 (Henneman 1957).
- **Behavioral validation:** Ramp current to motor pool, log spike order; sequence should match assigned size index.

### H.04 Slow-twitch (type I) vs fast-twitch (type IIa/IIx) muscle fibers
*[from Part V — Movement (Ch 30-39); renumbered from H.53]*

- **System:** myosin ATPase / mitochondrial / glycolytic biochemistry of muscle fiber.
- **Biological role:** Type I = low ATPase, oxidative, fatigue-resistant, small force. Type IIa = fast oxidative-glycolytic, moderately fatigable. Type IIx = fast glycolytic, large force, highly fatigable. Endurance training shifts IIx→IIa; resistance training increases cross-section.
- **Sim status:** **missing** — no muscle layer at all.
- **Cluster:** H primary.
- **Prerequisites:** H.50, H.51.
- **Citation:** Kandel 6e Ch 31 p 741–744.
- **Behavioral validation:** Long-train fatigue: type I sustains, type IIx force decays.

### H.05 Force-length and force-velocity relations — sarcomere mechanics
*[from Part V — Movement (Ch 30-39); renumbered from H.54]*

- **System:** actin-myosin cross-bridge geometry; series-elastic tendon.
- **Biological role:** Force is bell-shaped in fiber length (optimal overlap) and decreasing in shortening velocity (Hill curve). Joint torque also depends on muscle moment arm. Sets the plant for every motor controller.
- **Sim status:** **missing** — would be needed for any closed-loop limb / Hill-type plant.
- **Cluster:** H primary.
- **Prerequisites:** H.50, H.51.
- **Citation:** Kandel 6e Ch 31 p 752–757.
- **Behavioral validation:** Isometric force vs imposed length; isotonic force vs velocity match Hill hyperbola.

### H.06 Stretch reflex (monosynaptic Ia → α-MN) — myotatic reflex
*[from Part V — Movement (Ch 30-39); renumbered from H.55]*

- **System:** muscle spindle Ia afferent → spinal α-motoneuron of homonymous muscle.
- **Biological role:** Resists muscle lengthening; basis of tendon-tap reflex. Latency ~30 ms. Stabilizes joint angle and provides mechanical impedance.
- **Sim status:** **missing** — no spindle, no Ia afferent, no spinal interneurons.
- **Cluster:** H primary; K (proprioceptive transduction) secondary.
- **Prerequisites:** H.50, K.* (spindle transduction).
- **Citation:** Kandel 6e Ch 32 p 762–765 (Sherrington / Liddell).
- **Behavioral validation:** Tendon tap → ipsilateral homonymous burst at monosynaptic latency; no response with cut dorsal root.

### H.07 Reciprocal inhibition (Ia inhibitory interneuron) — antagonist relaxation
*[from Part V — Movement (Ch 30-39); renumbered from H.56]*

- **System:** Ia afferent → glycinergic Ia-IN → α-MN of antagonist muscle.
- **Biological role:** Co-activation of antagonists is wasteful and rigid; Ia-IN suppresses antagonist α-MN during stretch reflex and voluntary movement. Modulated by descending input.
- **Sim status:** **missing** — no antagonist pairing, no Ia-IN class.
- **Cluster:** H primary; B (inhibitory selection circuits) secondary.
- **Prerequisites:** H.55.
- **Citation:** Kandel 6e Ch 32 p 765–768.
- **Behavioral validation:** Tendon tap to flexor → simultaneous suppression of extensor EMG.

### H.08 Renshaw cell recurrent inhibition — gain control on α-MN
*[from Part V — Movement (Ch 30-39); renumbered from H.57]*

- **System:** α-MN axon collateral → glycinergic Renshaw cell → same and synergist α-MNs.
- **Biological role:** Negative feedback on motoneuron firing rate; limits saturation, sharpens recruitment differences, modulated by descending tracts.
- **Sim status:** **missing** — no recurrent feedback class on motor pools. v3 lateral inhibition in `g11_bg_runner` is at striatal level, not motoneuron level — different mechanism.
- **Cluster:** H primary; B secondary.
- **Prerequisites:** H.50.
- **Citation:** Kandel 6e Ch 32 p 768–770.
- **Behavioral validation:** Direct ventral-root antidromic stimulation suppresses ongoing MN firing.

### H.09 Golgi tendon organ Ib reflex — autogenic inhibition
*[from Part V — Movement (Ch 30-39); renumbered from H.58]*

- **System:** Ib afferent (force-sensitive, in tendon) → Ib interneuron → α-MN of homonymous muscle.
- **Biological role:** Inhibits the muscle producing high tension; protects against rupture; contributes to load-compensation during locomotion.
- **Sim status:** **missing** — no force sensor, no Ib pathway.
- **Cluster:** H primary; K secondary.
- **Prerequisites:** H.51 (force), H.55.
- **Citation:** Kandel 6e Ch 32 p 770–772.
- **Behavioral validation:** High muscle tension → autogenic α-MN inhibition; reflex reverses sign during locomotion (state-dependent).

### H.10 γ-motoneuron / fusimotor drive — spindle sensitivity gain
*[from Part V — Movement (Ch 30-39); renumbered from H.59]*

- **System:** γ-MN → intrafusal muscle fibers (bag / chain) inside spindle.
- **Biological role:** Adjusts spindle sensitivity by contracting intrafusal fibers, keeping spindle taut at all extrafusal lengths. α-γ co-activation maintains responsiveness during voluntary movement.
- **Sim status:** **missing** — no spindle model.
- **Cluster:** H primary; K secondary.
- **Prerequisites:** H.55.
- **Citation:** Kandel 6e Ch 32 p 766–768.
- **Behavioral validation:** Dynamic γ-MN stimulation increases Ia phasic response to stretch.

### H.11 Flexion (withdrawal) reflex + crossed-extensor — polysynaptic protective
*[from Part V — Movement (Ch 30-39); renumbered from H.60]*

- **System:** cutaneous nociceptor → spinal interneuron network → ipsilateral flexor MNs (withdraw) + contralateral extensor MNs (support).
- **Biological role:** Removes limb from noxious stimulus while preserving balance via crossed extensor support. Polysynaptic, ~50–100 ms latency, integrates multimodal cutaneous input.
- **Sim status:** **missing** — no nociception, no crossed connectivity.
- **Cluster:** H primary; K secondary.
- **Prerequisites:** H.50, K.* (nociception).
- **Citation:** Kandel 6e Ch 32 p 762–763, 770–773.
- **Behavioral validation:** Noxious heel stimulus → ipsi-flexor + contra-extensor co-activation.

### H.12 Spasticity / hyperreflexia after CNS lesion — descending pathway loss
*[from Part V — Movement (Ch 30-39); renumbered from H.61]*

- **System:** corticospinal / reticulospinal tract lesion → loss of descending inhibition on spinal reflex circuits.
- **Biological role:** Removal of supraspinal balance leaves stretch reflexes hyperexcitable → velocity-dependent rigidity, clonus. Diagnostic of UMN lesion.
- **Sim status:** **missing** — no spinal reflex circuits to disinhibit.
- **Cluster:** H primary; P (disease) secondary.
- **Prerequisites:** H.55, H.57.
- **Citation:** Kandel 6e Ch 32 p 776–780.
- **Behavioral validation:** Velocity-dependent stretch resistance after corticospinal ablation; clonus on sustained stretch.

### H.13 Central pattern generator (CPG) for locomotion — Brown 1911
*[from Part V — Movement (Ch 30-39); renumbered from H.62]*

- **System:** spinal interneuron network (lamprey: ~100 segments; mammal: lumbar enlargement).
- **Biological role:** Intrinsic spinal circuit generates the rhythm and pattern of stepping/swimming without descending input or sensory feedback. Demonstrated in deafferented decerebrate cat, fictive locomotion in paralyzed spinal cat, isolated lamprey cord.
- **Sim status:** **missing** — no rhythm-generating spinal circuit. Project's motor "actions" are step-by-step discrete choices, not continuous rhythmic output. Closing this would be a substantial new region. Highly project-relevant for any continuous-time motor control work.
- **Cluster:** H primary; A (selection of gait) secondary.
- **Prerequisites:** H.50, J.* (plasticity not required for basic CPG).
- **Citation:** Kandel 6e Ch 33 p 783–793 (Brown 1911, Grillner).
- **Behavioral validation:** Isolated spinal cord with neuroactive drug bath produces alternating flexor / extensor bursts; rhythm persists after deafferentation.

### H.14 Half-center model — mutually inhibiting flexor / extensor pools
*[from Part V — Movement (Ch 30-39); renumbered from H.63]*

- **System:** two reciprocally inhibitory interneuron pools with intrinsic adaptation / "fatigue."
- **Biological role:** Brown's mechanism for rhythm generation: when half is active, it inhibits the other; adaptation releases the inhibition, switching state. Modern view splits this into separate rhythm-generating layer + pattern-shaping layer (Rybak).
- **Sim status:** **missing** — no half-center inhibition between motor pools. Could be prototyped with two AdEx pools + adaptation + reciprocal `RegionPathway` density.
- **Cluster:** H primary; B secondary.
- **Prerequisites:** H.62; AdEx adaptation already exists (`sim/kernels.py` `fused_adex_dynamics_update`).
- **Citation:** Kandel 6e Ch 33 p 790–793.
- **Behavioral validation:** Pair of AdEx pools with strong cross-inhibition + spike-frequency adaptation → spontaneous alternating bursts at adaptation timescale.

### H.15 Mesencephalic locomotor region (MLR) / pedunculopontine nucleus (PPN) — locomotion initiation
*[from Part V — Movement (Ch 30-39); renumbered from H.64]*

- **System:** midbrain MLR (cuneiform + PPN) → reticulospinal tract → spinal CPG.
- **Biological role:** Tonic glutamatergic drive from MLR initiates locomotion and grades speed (low → walk; high → trot/gallop). PPN cholinergic. Receives input from BG output (SNr/GPi) — disinhibited gait initiation. DBS target in Parkinson freezing-of-gait.
- **Sim status:** **missing** — no MLR region, no continuous "go-faster" drive signal feeding motor pools.
- **Cluster:** H primary; A (BG → MLR), C (cholinergic) secondary.
- **Prerequisites:** H.62, A.* (BG output).
- **Citation:** Kandel 6e Ch 33 p 798–806.
- **Behavioral validation:** Graded MLR stimulation → graded locomotor frequency; BG output ablation removes gait initiation.

### H.16 Voluntary motor cortex (M1) somatotopic map — Penfield homunculus
*[from Part V — Movement (Ch 30-39); renumbered from H.65]*

- **System:** primary motor cortex layer V Betz cells → corticospinal tract.
- **Biological role:** Microstimulation evokes muscle / movement; ordered face → arm → leg from lateral to medial. Map is plastic on training, lesion, amputation. M1 codes muscle activation patterns and movement parameters at single-cell level.
- **Sim status:** **partial** — `g11_bg_runner` has per-action `cortex_X` pools (4 actions × 25 neurons) that are functionally equivalent to a coarse M1 map. No somatotopy continuum, no microstimulation API.
- **Cluster:** G primary; A secondary.
- **Prerequisites:** A.* (cortex → BG), H.50.
- **Citation:** Kandel 6e Ch 34 p 815–825.
- **Behavioral validation:** Stim cortex_X → motor_X firing → "action X" emitted; lesion cortex_X disables action X.

### H.17 Population vector coding (Georgopoulos) — distributed direction code
*[from Part V — Movement (Ch 30-39); renumbered from H.66]*

- **System:** M1 single-unit recordings during 2D / 3D arm reaching.
- **Biological role:** Each M1 neuron has a preferred direction with cosine tuning. The population vector (sum of preferred-direction vectors weighted by firing rate) predicts reach direction on a trial-by-trial basis. Foundational for BMI decoders.
- **Sim status:** **partial** — `g11_bg_runner` motor pools are categorical (one pool per action) rather than vector-tuned. Could be tested by adding cosine-tuned input layer; would naturally yield population vector readout.
- **Cluster:** G primary; H secondary.
- **Prerequisites:** H.65.
- **Citation:** Kandel 6e Ch 34 p 825–840 (Georgopoulos 1986).
- **Behavioral validation:** Compute Σ rᵢ · θᵢ across motor pool → matches commanded angle within ~10°.

### H.18 Mirror neurons (premotor F5 / parietal) — observation = execution
*[from Part V — Movement (Ch 30-39); renumbered from H.67]*

- **System:** ventral premotor cortex F5 + inferior parietal lobule.
- **Biological role:** Cells fire both when the monkey performs a goal-directed action and when it observes another agent perform the same action. Implicated in action understanding, imitation learning, possibly social cognition.
- **Sim status:** **missing** — no observational input pathway, no premotor region beyond what `cortex_X` represents.
- **Cluster:** G primary; H secondary.
- **Prerequisites:** H.65, K.* (visual input).
- **Citation:** Kandel 6e Ch 34 p 833–845.
- **Behavioral validation:** Same neurons fire on (a) executing reach to object, (b) observing reach by another agent.

### H.19 Premotor / SMA — sequential & rule-based action
*[from Part V — Movement (Ch 30-39); renumbered from H.68]*

- **System:** dorsal/ventral premotor cortex + supplementary motor area (medial wall).
- **Biological role:** Pre-SMA / SMA encode action sequences, internally generated movement, motor rules; dorsal PMd encodes instructed reach plans during delay periods; ventral PMv encodes grasp configuration.
- **Sim status:** **partial** — PFC region (60 recurrent neurons) added 2026-04-27 is the closest analog (working memory); does not yet encode sequence / rule, no separation of pre-SMA vs PMd vs PMv.
- **Cluster:** G primary; H secondary.
- **Prerequisites:** H.65; PFC region (existing).
- **Citation:** Kandel 6e Ch 34 p 822–835.
- **Behavioral validation:** Delay-period activity in PMd predicts upcoming reach; SMA lesion disrupts internally cued sequences but not stimulus-cued ones.

---

## Cluster F — Cerebellum (currently cell presets only — circuit MISSING)

**What "cluster F closure" means for this project:** the simulator's `sim/enums.py` ships
HH presets `HH_CEREBELLAR_PURKINJE` and `HH_CEREBELLAR_GRANULE` and a profile
`CEREBELLAR_CORTEX_SIMPLE`, but **no module wires the canonical microcircuit** (mossy
→ granule → parallel-fiber → Purkinje, plus climbing fiber from inferior olive, plus
Purkinje → deep nuclei → output). To close cluster F, build a `cerebellum_runner.py`
that:
1. Declares regions: `granule_layer`, `purkinje_layer`, `deep_nuclei`, `inferior_olive`.
2. Declares pathways: `mossy → granule` (sparse), `granule → purkinje` via parallel-fiber
   axon (massive fan-out), `inferior_olive → purkinje` (1:1 or 1:few climbing fiber),
   `purkinje → deep_nuclei` (inhibitory), `deep_nuclei → output`.
3. Implements **PF→PC LTD** gated by climbing-fiber complex spikes (sign-flipped STDP
   variant, anti-Hebbian: PF active + CF active → ↓ weight). Existing
   `fused_stdp_weight_update` would need a CF-gated variant or a per-pathway sign flag.
4. Validates against eyeblink conditioning or VOR adaptation as the smoke-test behavior.

### H.20 Brain-machine interface (BMI) decoder — population vector or Kalman
*[from Part V — Movement (Ch 30-39); renumbered from H.69]*

- **System:** multi-electrode array in M1 / PMd / PPC; offline / online decoder maps spike rates → cursor / arm trajectory.
- **Biological role:** Demonstrates that population activity (esp. tuning + variance) can be inverted to estimate intended movement at sub-second latency. Closed-loop adaptation (subject + decoder co-adapt) approaches natural-arm performance.
- **Sim status:** **partial** — `experiment/readout.py` has `ReadoutEngine` for population rates / spike counts / PSD. Could implement a population-vector BMI decoder on motor pools as an analysis layer; would require no engine change.
- **Cluster:** H primary; G secondary.
- **Prerequisites:** H.66.
- **Citation:** Kandel 6e Ch 39 p 957–980 (Shenoy, Donoghue).
- **Behavioral validation:** Decoder applied to motor-pool spike train tracks intended action with > chance accuracy; performance improves with co-adaptation.

### H.21 Deep brain stimulation (DBS) — STN / GPi pulse train as Parkinson therapy
*[from Part V — Movement (Ch 30-39); renumbered from H.70]*

- **System:** chronic implanted electrode at STN (most common) or GPi; ~130 Hz pulse train.
- **Biological role:** High-frequency stimulation paradoxically silences pathological synchronous bursting in parkinsonian BG, restoring function. Mechanism still debated (depolarization block, antidromic activation, jamming).
- **Sim status:** **missing** but trivially testable — inject high-frequency current into STN region in `g11_bg_runner` after DA-depletion lesion; observe whether action initiation recovers.
- **Cluster:** H primary; A, P secondary.
- **Prerequisites:** A.51, A.52, P.50.
- **Citation:** Kandel 6e Ch 39 p 970–980.
- **Behavioral validation:** Simulate Parkinson (P.50) → bradykinesia; add 130 Hz STN stim → recovery of action initiation rate.

---

## Cluster H — Posture (Ch 36)

### H.22 Postural control — equilibrium + orientation, multi-modal integration
*[from Part V — Movement (Ch 30-39); renumbered from H.71]*

- **System:** somatosensory + vestibular + visual integration in brainstem (vestibular nuclei, reticular formation), cerebellum, BG, cortex.
- **Biological role:** Maintains upright stance against gravity and perturbations. Two interrelated goals: (a) center-of-mass over base of support (equilibrium); (b) body-segment alignment (orientation). Distributed control across spinal antigravity (basic), brainstem+cerebellum (sensory integration), BG (adaptation), cortex (anticipatory).
- **Sim status:** **missing** — no body, no gravity, no balance metric.
- **Cluster:** H primary; F, K secondary.
- **Prerequisites:** H.50–H.62, K.52.
- **Citation:** Kandel 6e Ch 36 p 883–905.
- **Behavioral validation:** Platform-translation perturbation → automatic postural response within 100–150 ms; anticipatory postural adjustment precedes voluntary arm raise.

### H.23 Anticipatory postural adjustment (APA) — efference-copy-driven
*[from Part V — Movement (Ch 30-39); renumbered from H.72]*

- **System:** SMA / PPC → spinal postural muscles in advance of voluntary movement.
- **Biological role:** Before voluntary arm raise, postural muscles (legs, trunk) pre-activate to counter the upcoming inertial torque. Latency: ~50–100 ms before primary mover. Lost in cerebellar / BG damage (Parkinson "freezing"). Demonstrates motor system uses forward model (F.56).
- **Sim status:** **missing**.
- **Cluster:** H primary; F secondary.
- **Prerequisites:** H.71, F.56.
- **Citation:** Kandel 6e Ch 36 p 887–893.
- **Behavioral validation:** EMG of leg muscles fires *before* arm-raise EMG; cerebellar lesion abolishes APA precedence.

---

## Cluster H — Eye movement control (Ch 35)

### H.24 Saccade generator — pontine reticular formation burst circuit
*[from Part V — Movement (Ch 30-39); renumbered from H.73]*

- **System:** paramedian pontine reticular formation (horizontal saccades) + rostral interstitial MLF (vertical); excitatory burst neurons (EBN), inhibitory burst neurons (IBN), omnipause neurons (OPN).
- **Biological role:** Saccade is initiated when OPNs are silenced, releasing EBNs to drive a high-velocity burst on extraocular MNs. Pulse-step waveform; pulse drives the saccade, step holds gaze. Burst duration determines amplitude.
- **Sim status:** **missing** — no eye plant.
- **Cluster:** H primary; A secondary (BG → SC → saccade).
- **Prerequisites:** H.50; A.56.
- **Citation:** Kandel 6e Ch 35 p 868–880.
- **Behavioral validation:** OPN silence → EBN burst → MN burst → step displacement; saccade main sequence (peak velocity vs amplitude).

### H.25 Superior colliculus saccade map — topographic motor map
*[from Part V — Movement (Ch 30-39); renumbered from H.74]*

- **System:** intermediate / deep SC; topographic representation of saccade target relative to fovea.
- **Biological role:** Stimulating a SC site evokes a saccade of fixed amplitude/direction matching that site. SC integrates visual + auditory + cognitive inputs into a "where to look next" decision; output → pontine reticular saccade generator (H.73). Receives BG (SNr) tonic inhibition; selection by SNr disinhibition (A.56).
- **Sim status:** **missing**.
- **Cluster:** H primary; A secondary.
- **Prerequisites:** A.56.
- **Citation:** Kandel 6e Ch 35 p 875–882.
- **Behavioral validation:** Microstim of SC site → fixed saccade vector; SC lesion → contralateral neglect for saccades.

---

## Cluster J — Plasticity rules specific to motor system (cross-references)

## Cluster I — additions

### I.02 Hodgkin–Huxley action potential — canonical Na+/K+ kinetics
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.50]*

- **System:** All excitable neurons; squid giant axon canonical model.
- **Biological role:** Depolarization opens fast voltage-gated Na+ channels (regenerative inward current, rising phase), then inactivates them; delayed rectifier K+ channels open more slowly to drive repolarization. Two-gate (m,h for Na; n for K) ODE system fully reconstructs AP waveform, threshold, and refractory behavior.
- **Sim status:** **implemented** — `sim/kernels.py:fused_hodgkin_huxley_dynamics_update` (line 36) implements the m/h/n kinetics with per-gate Q10 (`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`); 20+ regional HH presets in `sim/enums.py:DefaultHodgkinHuxleyParams`.
- **Cluster:** I (primary).
- **Prerequisites:** none (foundational).
- **Citation:** Kandel 6e Ch 10 p 211–222.
- **Behavioral validation:** AP waveform peak ≈ +40 mV, width ~1–2 ms at 37 °C; ENa reversal demonstrable by Na+ substitution; refractory period emerges from h-gate recovery time constant.

### I.03 Voltage-gated Na+ channel inactivation (h-gate / IFM motif)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.51]*

- **System:** All Na+-spiking neurons; especially refractory dynamics.
- **Biological role:** During sustained depolarization, the cytoplasmic linker between domains III and IV (IFM motif) docks against the inner pore mouth, occluding it. This "ball-and-chain" inactivation enforces absolute and relative refractory periods, limits maximum firing rate, and gives rise to use-dependent block by local anesthetics.
- **Sim status:** **implemented** — h-gate in `fused_hodgkin_huxley_dynamics_update` provides functional inactivation. AdEx uses adaptation variable `w` for similar effect; Izhikevich uses recovery variable `u`. Soft refractoriness emerges naturally.
- **Cluster:** I.
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 217–224 (Fig 10-8, 10-10).
- **Behavioral validation:** Paired-pulse INa amplitude ratio < 1 at 5–15 ms intervals; recovery curve fittable to single exponential τ ≈ 5–10 ms.

### I.04 Voltage-gated Na+ channel α-subunit diversity (Nav1.1–1.9)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.52]*

- **System:** Distinct expression in interneurons, axon initial segments, nociceptors, cardiac/skeletal muscle.
- **Biological role:** Nav1.1 dominates GABAergic interneurons (loss → Dravet epilepsy); Nav1.2/1.6 are central excitatory; Nav1.6 enriched at nodes of Ranvier and AIS; Nav1.7/1.8/1.9 in peripheral pain neurons; Nav1.4/1.5 in muscle. Different kinetics tune cell-class excitability.
- **Sim status:** **missing** — `fused_hodgkin_huxley_dynamics_update` uses a single homogeneous Na+ conductance; no isoform-level expression. HH presets parametrize global g_Na but do not differentiate Nav1.x kinetics. [discrepancy: many regional presets exist (`HH_CORTICAL_FS_INTERNEURON`, `HH_NODE`-equivalents) but they vary g_Na/g_K rather than channel identity.]
- **Cluster:** I.
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 224–227.
- **Behavioral validation:** Knock-out simulation of Nav1.1 in interneurons should reduce inhibitory firing rate while sparing pyramidal AP threshold.

### I.05 Delayed-rectifier K+ channels (Kv1, Kv2)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.53]*

- **System:** Universal repolarizing current; squid axon canonical Kv.
- **Biological role:** Slowly activating, non-inactivating K+ channels open during the falling phase of the AP, driving repolarization toward EK. Kv1 activates at near-threshold voltages and contributes to AP threshold; Kv2 activates during repolarization and stays open through the AHP.
- **Sim status:** **implemented** — n-gate of HH model captures the lumped delayed-rectifier kinetics; Izhikevich/AdEx subsume into adaptation variable.
- **Cluster:** I.
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 227–229 (Fig 10-13).
- **Behavioral validation:** TEA application (parameter sweep g_K → 0) should broaden APs and abolish AHP.

### I.06 Kv3 fast-repolarizing channels — fast-spiking phenotype
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.54]*

- **System:** Cortical FS GABAergic interneurons, cerebellar Purkinje neurons, auditory brainstem.
- **Biological role:** Activate only at high depolarizations near AP peak with very fast kinetics; close rapidly after repolarization. This produces narrow APs (~0.3 ms) and minimal AHP, enabling sustained firing up to 500 Hz characteristic of FS interneurons.
- **Sim status:** **partial** — FS phenotype exists at the level of preset firing rates (`HH_CORTICAL_FS_INTERNEURON`, `IZH2007_FS_CORTICAL_INTERNEURON`) but does not arise from an explicit Kv3 kinetic. AP-width differences between RS and FS in HH simulation are tuned via Cm and uniform g_K.
- **Cluster:** I.
- **Prerequisites:** I.53.
- **Citation:** Kandel 6e Ch 10 p 228–231.
- **Behavioral validation:** FS neurons should sustain ~300+ Hz firing without rate adaptation under steady current; AP half-width < 0.5 ms.

### I.07 A-type transient K+ current (Kv4)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.55]*

- **System:** Dendrites and soma of pyramidal neurons; modulates AP threshold and back-propagation.
- **Biological role:** Activates fast on depolarization but inactivates over ms–tens of ms. Steady-state inactivation by small subthreshold depolarizations means hyperpolarizing inputs de-inactivate IA, raising threshold and producing delayed firing (Fig 10-15B). Important for delay-line coding and dendritic compartmentalization.
- **Sim status:** **missing** — no IA implemented. Fast K+ in HH lumps as delayed rectifier; transient component absent. Adaptation behavior in cortex presets comes from u/w recovery variables, not IA.
- **Cluster:** I.
- **Prerequisites:** I.53.
- **Citation:** Kandel 6e Ch 10 p 227–228, 231 (Fig 10-15B).
- **Behavioral validation:** Hyperpolarizing pre-pulse should produce a delay-to-first-spike that scales with pre-pulse duration.

### I.08 M-current / Kv7 / KCNQ — slow non-inactivating subthreshold K+
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.56]*

- **System:** Cortical pyramidal cells, hippocampus, sympathetic ganglia.
- **Biological role:** Slowly activating, non-inactivating, open near rest. Dampens excitability and produces spike-frequency adaptation across hundreds of ms. Suppressed by muscarinic ACh receptor activation (hence "M-current"), increasing excitability under cholinergic drive.
- **Sim status:** **implemented** — `sim/kernels.py:fused_hh_m_current_update` (line 108) provides Kv7-like persistent K+ with parameter `g_M_max` and `tau_m_ms`. Modulation by neuromodulators is supported via the declarative neuromodulator framework (target_type="excitability_drive").
- **Cluster:** I (primary), C (neuromodulation gating).
- **Prerequisites:** I.53.
- **Citation:** Kandel 6e Ch 10 p 228 (Fig 10-13, 10-15).
- **Behavioral validation:** Spike-frequency adaptation across 200–500 ms during a step current; ACh (muscarinic agonist) → reduced adaptation, increased steady-state rate.

### I.09 Ih / HCN hyperpolarization-activated cyclic-nucleotide-gated channel
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.57]*

- **System:** Cortical layer-5 pyramidal dendrites, thalamocortical relay, SAN-like pacemakers.
- **Biological role:** Mixed Na+/K+ permeability, reversal ~−40 mV. Slowly activated by hyperpolarization; produces a depolarizing "sag" that returns membrane toward rest after IPSPs. Drives pacemaker rhythms; modulated by intracellular cAMP, linking neuromodulator state to intrinsic rhythmicity.
- **Sim status:** **implemented** — `sim/kernels.py:fused_hh_h_current_update` (line 145) implements Ih. Used in thalamocortical and pacemaker presets.
- **Cluster:** I (primary), N (sleep/arousal pacemaking).
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 228 (Fig 10-15D).
- **Behavioral validation:** Voltage sag in response to hyperpolarizing step (V approaches steady-state ~5–10 mV less negative than peak); rebound depolarization after IPSP termination.

### I.10 Persistent Na+ current (INaP)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.58]*

- **System:** Cortical pyramidal cells, SCN pacemaker neurons, dopaminergic SNc.
- **Biological role:** Small (1–3% of transient INa), non-inactivating Na+ current activating at voltages as negative as −70 mV. Amplifies subthreshold EPSPs, drives slow pacemaker depolarizations (e.g., circadian rhythm-related firing in suprachiasmatic nucleus), and contributes to bursting.
- **Sim status:** **implemented** — `sim/kernels.py:fused_hh_NaP_current_update` (line 159). Available in pacemaker presets (e.g., `HH_GPE_PACEMAKER`, `HH_DOPAMINE_SNC`).
- **Cluster:** I (primary), N (circadian/sleep).
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 230–231 (Fig 10-15E).
- **Behavioral validation:** Spontaneous firing in absence of synaptic input at 1–5 Hz; pharmacological block (g_NaP → 0) should silence pacemaker activity.

### I.11 High-voltage-activated Ca2+ channels (L-type Cav1, P/Q Cav2.1, N Cav2.2)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.59]*

- **System:** Soma, dendrites, presynaptic terminals.
- **Biological role:** Open during APs at depolarized voltages; mediate Ca2+ influx that triggers vesicle release (P/Q, N at terminals), drives plateau potentials and slow afterdepolarizations (L-type, soma/dendrite), and signals to Ca2+-activated channels and second messengers.
- **Sim status:** **partial** — Ca2+ entry as a separate ion species is NOT modelled in the HH kernel. Synaptic transmission uses event-driven release, not residual-Ca-dependent. No L-type plateau dynamics. [discrepancy: many findings in `research/findings/` discuss Ca2+ dynamics conceptually but the kernel lacks an explicit cytoplasmic Ca pool.]
- **Cluster:** I (primary), J (presynaptic release coupling).
- **Prerequisites:** I.50.
- **Citation:** Kandel 6e Ch 10 p 227 (Cav nomenclature).
- **Behavioral validation:** L-type plateau potentials should appear in MSN-like neurons (D1/D2) under sustained depolarization; abolished by nifedipine analog (g_Cav1 → 0).

### I.12 T-type Ca2+ channels (Cav3, LVA) — rebound bursting
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.60]*

- **System:** Thalamic relay (TC) neurons, thalamic reticular (TRN), subset of cortex.
- **Biological role:** Activate near −65 mV, inactivate at rest. Hyperpolarization (e.g., from TRN GABAergic input) de-inactivates them; subsequent return to rest produces a rebound Ca2+ spike that can trigger a burst of Na+ APs. Substrate of thalamic burst-firing in sleep and absence epilepsy.
- **Sim status:** **implemented** — `sim/kernels.py:fused_hh_CaT_current_update` (line 127). Used in `HH_THALAMIC_RELAY_TBURST` and `HH_TRN_BURST_INHIB` presets.
- **Cluster:** I (primary), N (sleep oscillations).
- **Prerequisites:** I.50, I.59.
- **Citation:** Kandel 6e Ch 10 p 227, 231.
- **Behavioral validation:** Rebound burst (3–7 Na+ spikes) after release from −90 mV hyperpolarization in TC presets.

### I.13 Calcium-activated K+ channels — BK and SK (mAHP/sAHP)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.61]*

- **System:** Hippocampal pyramidal, cortical pyramidal, midbrain dopamine, cerebellar Purkinje.
- **Biological role:** BK channels (large conductance, voltage- and Ca-dependent) open at AP peak with Ca2+ influx, contributing to fast repolarization. SK channels (small conductance, Ca-only, voltage-independent) gate slowly and accumulate over many spikes, producing the slow afterhyperpolarization (sAHP) that strongly limits sustained firing rate.
- **Sim status:** **missing** — neither BK nor SK is implemented as an explicit channel. Spike-frequency adaptation in current presets is captured phenomenologically by Izhikevich's u variable / AdEx's w / homeostatic threshold adaptation. [discrepancy: behavioral phenotype (adaptation) is present but not via Ca2+-K coupling, so adaptation can't respond to Ca-buffering manipulations.]
- **Cluster:** I (primary), C (apamin-sensitive sAHP modulated by NA/5HT).
- **Prerequisites:** I.59 (need Ca2+ pool first).
- **Citation:** Kandel 6e Ch 10 p 229.
- **Behavioral validation:** Apamin (SK block) simulation should abolish sAHP and increase steady-state firing; BK block should broaden AP and reduce fast AHP.

### I.14 Resting / leak K+ channels (K2P, inward rectifiers)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.62]*

- **System:** All neurons; especially glia (Kir4.1).
- **Biological role:** Always-open K+ pathways that set resting membrane potential near EK and supply input conductance (R_input). Inward rectifiers (Kir) pass K+ inward more readily than outward (cytosolic Mg2+/polyamine block). K2P (TWIK, TASK, TREK) are gated by temperature, mechanical force, anesthetics.
- **Sim status:** **implemented** (lumped) — `g_L * (V - E_L)` leak term in HH and AdEx. Single equivalent leak conductance, not split by Kir / K2P / Cl-.
- **Cluster:** I.
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 8 p 178–180; Ch 9 p 191–197.
- **Behavioral validation:** RMP scales correctly with [K+]o (Goldman / Nernst); halothane (TREK opener) simulation should hyperpolarize and reduce excitability.

### I.15 Resting membrane potential — Goldman/Nernst equilibrium
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.63]*

- **System:** All cells.
- **Biological role:** RMP arises from differential permeability across Na+/K+/Cl-. With PK ≫ PNa, PCl, the membrane sits near EK ≈ −75 mV but is pulled positive by finite Na+ leak. Goldman–Hodgkin–Katz equation quantifies. Maintained against passive leak by the electrogenic Na+/K+-ATPase (3 Na+ out / 2 K+ in).
- **Sim status:** **implemented** — equilibrium emerges from the integrated dynamics. E_inh = -75 mV with 0.7× propagation scaling matches biological E_K and inhibitory drive. No explicit Na+/K+ ATPase, but resting potential is stable due to leak parametrization.
- **Cluster:** I.
- **Prerequisites:** I.62.
- **Citation:** Kandel 6e Ch 9 p 191–199.
- **Behavioral validation:** Setting g_Na → 0 should drive Vm to ≈ −75 mV (E_K).

### I.16 Membrane capacitance Cm — passive integration time constant
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.64]*

- **System:** Universal; lipid bilayer property ~1 µF/cm².
- **Biological role:** Cm with input resistance Rin gives the membrane time constant τ_m = Rin·Cm (typically 5–30 ms in cortex). Sets temporal window for synaptic integration and rate of voltage change in response to current. Larger τ_m → more temporal summation; smaller τ_m → more coincidence detection.
- **Sim status:** **implemented** — `C_param` in Izhikevich-2007 and `C_m`/`C` in HH/AdEx kernels parametrize membrane capacitance per neuron.
- **Cluster:** I.
- **Prerequisites:** I.62.
- **Citation:** Kandel 6e Ch 9 p 199–203.
- **Behavioral validation:** Step-current voltage trajectory should rise as 1 − exp(−t/τ); FS neurons (low τ) should be coincidence detectors, RS (high τ) integrators.

### I.17 Cable equation, length constant λ — passive electrotonic spread
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.65]*

- **System:** Dendrites and unmyelinated axons.
- **Biological role:** λ = √(rm/ra). Voltage decays as exp(−x/λ) along a passive cable. Sets how far synaptic input propagates before attenuation. Diameter, channel density, and myelination control λ; thicker / better-insulated processes have longer λ.
- **Sim status:** **not-applicable** — simulator is single-compartment. No dendrites, no spatial extent, no λ. Synaptic inputs sum at the soma without distance-dependent attenuation.
- **Cluster:** I.
- **Prerequisites:** I.64.
- **Citation:** Kandel 6e Ch 9 p 203–207.
- **Behavioral validation:** N/A in current sim. Would require multi-compartment cable model to test.

### I.18 Axon initial segment (AIS) — high-density Nav, AP trigger zone
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.66]*

- **System:** AIS of all projection neurons; ~20–50 µm from soma.
- **Biological role:** Nav1.6 density at the AIS is ~50× greater than soma membrane, making it the lowest-threshold spike-trigger zone. Anchored by ankyrin-G; co-clustered with Kv1, Kv7. Plastic over hours-days (length, position) — homeostatic adjustment of intrinsic excitability.
- **Sim status:** **missing / not-applicable** — single-compartment model has no AIS; spike threshold is determined by global m/h kinetics. Homeostatic threshold adaptation in `cp_homeostatic_thresholds` provides a phenomenological analog of AIS plasticity.
- **Cluster:** I (primary), L (AIS structural plasticity = a critical-period-like phenomenon).
- **Prerequisites:** I.50, I.52.
- **Citation:** Kandel 6e Ch 7 p 153 (Fig 7-16); Ch 10 p 231.
- **Behavioral validation:** N/A in current sim. Would require either an explicit AIS compartment or a per-neuron threshold that adapts to mean activity (~the existing homeostasis).

### I.19 Saltatory conduction — myelinated axon AP propagation
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.67]*

- **System:** All myelinated axons (PNS Schwann, CNS oligodendrocyte).
- **Biological role:** Myelin reduces internodal Cm and increases rm, so passive electrotonic spread between nodes is fast. APs are regenerated only at the Na+-rich nodes of Ranvier (~50× node Na+ density). Conduction velocity scales with axon diameter and myelination, up to 100 m/s in humans.
- **Sim status:** **not-applicable** — no spatial axon, no propagation delay model. Synaptic events are delivered with `synaptic_delay_ms` parameter as a lumped scalar; saltatory effects are absorbed into that delay.
- **Cluster:** I (primary), Q (myelinating glia).
- **Prerequisites:** I.50, I.65.
- **Citation:** Kandel 6e Ch 7 p 151–153; Ch 9 p 207–208.
- **Behavioral validation:** N/A. Demyelination simulation would require multi-compartment axon.

### I.20 Refractory periods — absolute and relative
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.68]*

- **System:** All spiking neurons.
- **Biological role:** Absolute refractory: ~1–2 ms after AP, no second AP possible because Nav h-gates remain in the inactivated state. Relative refractory: 5–15 ms after AP, increased K+ conductance and partial Na+ inactivation make spike threshold higher; second AP requires stronger drive.
- **Sim status:** **implemented** — emerges naturally from HH h-gate / Izh u-recovery / AdEx w-adaptation. Acts as a hard upper bound on firing rate (~500 Hz at 37 °C).
- **Cluster:** I.
- **Prerequisites:** I.51.
- **Citation:** Kandel 6e Ch 10 p 220–221 (Fig 10-8).
- **Behavioral validation:** Paired-pulse spike probability < 1 at 2–10 ms ISI; recovery curve sigmoid.

### I.21 Intrinsic firing patterns — RS, FS, IB, CH (regular, fast, intrinsic-bursting, chattering)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.69]*

- **System:** Cortex layer 2/3, layer 5; subcortical.
- **Biological role:** Cortical pyramidal cells fire in adapting "regular spiking" pattern; FS interneurons fire fast, non-adapting; intrinsically bursting cells (IB) fire an initial high-frequency doublet/triplet then settle; chattering cells (CH) fire repetitive 30–80 Hz bursts. These patterns reflect distinct channel complements (Kv3, Kv4, Kv7, Cav, etc.).
- **Sim status:** **implemented** — Izhikevich and AdEx presets explicitly named `IZH2007_RS_CORTICAL_PYRAMIDAL`, `IZH2007_FS_CORTICAL_INTERNEURON`, `ADEX_RS`, `ADEX_FS`, `ADEX_IB`, `ADEX_CH` reproduce the four canonical patterns. HH presets use g_K / g_Na ratios + slower kinetics for RS, fast kinetics for FS.
- **Cluster:** I (primary), G (cortical processing).
- **Prerequisites:** I.50, I.54, I.55.
- **Citation:** Kandel 6e Ch 10 p 229–231 (Fig 10-14).
- **Behavioral validation:** F-I curve linearity (FS) vs adapting (RS); initial-burst signature (IB); 30+ Hz oscillation in CH cells under steady drive.

### I.22 Pacemaking via INaP + leak Na+ (e.g., SCN circadian)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.70]*

- **System:** Suprachiasmatic nucleus, SNc dopamine, GPe, locus coeruleus.
- **Biological role:** Subthreshold persistent Na+ leak depolarizes neuron toward AP threshold; firing without synaptic input. SCN expresses higher leak Na+ density during the day, producing day/night firing-rate differences that gate circadian behavior.
- **Sim status:** **partial** — INaP kernel exists (I.58) and is used in pacemaker presets; rhythmic firing emerges. No explicit circadian modulation of g_NaP, but neuromodulator framework could drive it via excitability_drive target.
- **Cluster:** I (primary), N (sleep / arousal / circadian), C (neuromodulator-driven excitability).
- **Prerequisites:** I.58.
- **Citation:** Kandel 6e Ch 10 p 230–231 (Fig 10-15E).
- **Behavioral validation:** Spontaneous firing 1–5 Hz with no input; firing rate scales with g_NaP parameter.

### I.23 Channelopathies — single-gene channel mutations cause disease
*[from Part II — Cells & Channels (Ch 7-10); renumbered from I.71]*

- **System:** Brain, heart, muscle, peripheral nerve.
- **Biological role:** Loss-of-function in Nav1.1 → Dravet epilepsy; Kv7 (KCNQ2/3) mutations → benign familial neonatal seizures; Cav1.4 → congenital stationary night blindness; SCN5A → cardiac long-QT. Demonstrates that intrinsic dynamics depend on specific channel proteins, not generic conductances.
- **Sim status:** **partial** — channel-level perturbations can be simulated by parameter sweeps over g_Na, g_K, g_M, g_NaP, etc. No explicit mutation library or per-isoform genes. A future "channelopathy mode" would require I.52 (isoform identity).
- **Cluster:** I (primary), P (disease).
- **Prerequisites:** I.52, I.56.
- **Citation:** Kandel 6e Ch 10 p 226–227.
- **Behavioral validation:** Reducing g_Na in interneuron presets only should produce E/I imbalance and runaway pyramidal firing, mimicking Dravet phenotype.

---

## Cluster Q — Glia & neurovascular

## Cluster J — additions

### J.31 Synaptic LTP/LTD as substrate for perceptual learning
*[from Part IV — Perception (Ch 17-29); renumbered from J.50]*

- **System:** sensory cortices (V1, S1, A1)
- **Biological role:** Repeated exposure to a stimulus reorganizes RFs and maps via the same NMDA-dependent LTP/LTD machinery used in hippocampus. Underlies tactile expansion in Braille readers, auditory map expansion after tone training, perceptual learning effects.
- **Sim status:** implemented at the rule level (STDP, NMDA, homeostasis, plasticity gating) but not deployed in any cortical sensory map (because no maps exist).
- **Cluster:** J (primary), E, L
- **Prerequisites:** E.53; STDP/NMDA already present
- **Citation:** Kandel 6e Ch 19 p ~519–522; Ch 24 p ~586
- **Behavioral validation:** training-induced cortical magnification; stimulus-specific RF sharpening.

---

## High-level coverage summary

- **Cluster K (sensory transduction):** entirely absent. K.50–K.61 are all gaps. Adding even one transducer (most tractable: K.56 mechanoreceptor classes, since they map to existing AdEx neurons with adapted RC + threshold) would let beacon/landmark be replaced by a real receptor channel.
- **Cluster E (cortical encoding):** RF, topographic maps, columns, V1/MT/IT hierarchy, ON/OFF & opponent channels are all missing. Sim has only 8-direction beacon + landmark plastic pathways into a flat cortex.
- **Cluster C/J overlaps:** simulator has the *machinery* (NMDA, STDP, NM gain, lateral inhibition) but it is not wired into any perceptual or nociceptive substrate.
- **Discrepancy notes:** project doc treats `--learned-perception` as the perception arc; Kandel-grade perception involves transduction → parallel channels → topographic maps → hierarchical RFs → multisensory binding → predictive inference. The current arc compresses all of this to a single plastic projection from beacon→cortex.

### J.32 PF→PC LTD vs corticostriatal LTP/LTD — opposite-sign learning rules
*[from Part V — Movement (Ch 30-39); renumbered from J.50]*

- **System:** cerebellum (F.54) and striatum (O.51).
- **Biological role:** Cerebellum decreases responsivity to error-correlated input (anti-Hebbian via CF teaching signal). Striatum changes corticostriatal weight in DA-direction-dependent manner (D1 LTP / D2 LTD on phasic DA burst). Two complementary learning systems for motor refinement.
- **Sim status:** **partial** — corticostriatal three-factor rule implemented (J.50 / O.51); cerebellar LTD missing (F.54).
- **Cluster:** J primary; F, O secondary.
- **Prerequisites:** F.54, O.51.
- **Citation:** Kandel 6e Ch 37 p 924–928 + Ch 38 p 947–950.
- **Behavioral validation:** see F.57 (eyeblink) and O.50 (DA RPE).

---

# Summary

**Entry count:** 47 entries (numbered .50–.74 within Cluster H, plus entries in F, A, B, C, G, J, K, L, O, P).

**Cluster distribution:**
- **Cluster H (motor & spinal):** 23 entries — H.50–H.74. **Massive gap** — almost everything textbook-canonical for spinal motor output (motor unit, twitch, recruitment, stretch reflex, Renshaw, CPG, MLR, posture, saccade generator) is missing from the simulator.
- **Cluster F (cerebellum):** 11 entries — F.50–F.60. Gap is well-defined: cell presets exist, no circuit. Closing requires a new runner (~`cerebellum_runner.py`) that wires mossy / granule / parallel-fiber / Purkinje / climbing-fiber / DCN, plus a CF-gated PF→PC LTD kernel.
- **Cluster A (BG action selection):** 9 entries — A.50–A.58. **Project flagship is well-aligned** with textbook for direct + indirect + selective disinhibition. Gaps: hyperdirect cortex→STN explicit pathway, parallel cognitive/limbic loops, subcortical SC / MLR loops, goal-directed vs habitual split. **One discrepancy noted:** real BG has dense cross-action cortex projections; project keeps cortex same-action-only because cross-projections were NEGATIVE in v1/v2/v3.1/v4 (cheat #5 closed by design 2026-04-28).
- **Cluster B (striatal microcircuit):** 6 entries. v3 lateral inhibition ships and aligns with B.52; D1/D2 segregation aligned (B.51); FS PV+ interneurons (B.54) and patch/matrix (B.55) absent.
- **Cluster O (reward):** 3 entries — DA RPE and three-factor rule mostly aligned; goal-directed → habitual transition missing.
- **Cluster P (disease):** 4 entries — Parkinson, Huntington, OCD/Tourette, schizophrenia. None implemented but Parkinson and Huntington are *trivially testable* by ablating DA / D2 pools in `g11_bg_runner`.
- **Cluster G (PFC / association):** 3 entries — PFC partial via 2026-04-27 region; PPC and SMA missing.
- **Cluster K, L, J:** scattered support entries.

**Top 3 most actionable additions for the simulator:**

1. **Implement Parkinson / Huntington smoke tests on `g11_bg_runner` (P.50, P.51).** Drop `current_reward_signal` to zero or ablate `str_d2_X` pools; measure action-initiation rate and perseveration. Tests whether the cascade reproduces the textbook prediction. Zero new infrastructure required — pure runner experiment. Would close P cluster meaningfully and confirm A cluster fidelity.

2. **Add hyperdirect cortex → STN pathway in `g11_bg_runner` (A.52).** Current implementation has STN in indirect pathway only; cortex projects only via D1/D2 striatum. A small `RegionPathway(from_region="cortex_X", to_region="stn", density=0.3, weight_mean=...)` per action closes the well-known Nambu hyperdirect circuit and provides a fast stop-signal substrate. Could be tested on a stop-signal task analog. **One-day feature.**

3. **Build the canonical cerebellum microcircuit (F.50–F.55) as a new runner.** Highest-leverage missing feature in the codebase — currently ALL of cluster F is presets-only. Required pieces: (a) declare regions for granule / Purkinje / DCN / inferior olive; (b) wire mossy → granule (sparse), granule → Purkinje (dense PF), IO → Purkinje (1:1 CF), Purkinje → DCN (inhibitory); (c) implement a new fused kernel `fused_pf_pc_ltd` with `(PF_pre × CF_gate) → ΔW < 0` semantics; (d) port eyeblink conditioning as the validation behavior. This is a *major* feature (≥1 week) but unlocks an entire dimension of motor learning experiments and would compose with the existing BG flagship (cerebellum + BG + PFC = the canonical motor-control trinity).

**Discrepancies flagged:**
- A.52 (hyperdirect): present in textbook, partial in simulator.
- A.54 (parallel functional loops): textbook describes 5 channels (motor / oculomotor / dlPFC / lateral OFC / ACC) with same circuit motif; simulator implements only the motor channel.
- B.52 (MSN lateral inhibition + cross-action cortex routing): real BG is anatomically dense in cross-action wiring at both cortex and striatum; project's same-action-only routing is a *principled choice* given v1/v2/v3.1/v4 NO-GO findings on cross-projections, but is anatomically a simplification.

### J.33 Implicit memory taxonomy — habituation / sensitization / classical / operant / priming / skill
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.50]*

- **System:** Distributed: cerebellum (motor adaptation), BG (habit, stimulus-response), amygdala (fear conditioning), neocortex sensory areas (priming), Aplysia gill-withdrawal circuit (sensitization).
- **Biological role:** Knowledge that guides behavior without conscious recall. Incremental, repetition-driven, often reinforcement-shaped. Multiple subsystems with distinct neural substrates and characteristic timescales.
- **Sim status:** partial — STDP/Hebbian/reward-modulated plasticity (covered in Ch 53) supports operant-style RL; cerebellar adaptation and habit-vs-goal arbitration absent.
- **Cluster:** J primary, A secondary, F (cerebellum, missing).
- **Prerequisites:** Ch 53 J entries.
- **Citation:** Kandel 6e Ch 52 pp 1303–1306.
- **Behavioral validation:** Double dissociation: amnesic patients (H.M.) acquire mirror-tracing skill but cannot verbally recall sessions.

### J.34 Memory imperfections as features — schemas, gist, false memory
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.51]*

- **System:** Distributed cortical schemas; HC-amygdala interactions for emotional gist.
- **Biological role:** Errors (gist over verbatim, suggestibility, intrusion) reflect adaptive prioritization of generalizable structure over episodic detail. Reconsolidation makes retrieval inherently editable.
- **Sim status:** not-applicable for current scope — could be modeled with structural plasticity + gated reconsolidation but no project goal yet.
- **Cluster:** J primary, D secondary.
- **Prerequisites:** D.51.
- **Citation:** Kandel 6e Ch 52 pp 1306–1308.
- **Behavioral validation:** DRM false-recall paradigm; misinformation effect.

---

## Ch 54 — The Hippocampus and the Neural Basis of Explicit Memory Storage (PROJECT-CRITICAL)

### J.35 Schaffer collateral LTP — NMDA-dependent, postsynaptic, associative
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.52]*

- **System:** CA3→CA1 synapse on proximal CA1 dendrites; AMPA + NMDA receptors; CaMKII / PKC / NO retrograde messenger; AMPA insertion at silent synapses.
- **Biological role:** Hebbian coincidence detection: NMDAR Mg²⁺ block requires postsynaptic depolarization + presynaptic glutamate. Ca²⁺ influx triggers CaMKII → AMPAR phosphorylation + insertion. Silent synapses (NMDAR-only) "AMPAfied" by LTP.
- **Sim status:** implemented (Ch 53 J entries) — STDP fused kernel, NMDA voltage-dependent Mg²⁺ block (`fused_nmda_update_and_current`), soft-bound w_max. Silent-synapse AMPAfication not explicitly modeled but functionally subsumed by STDP weight growth from near-zero.
- **Cluster:** J primary, I (NMDA receptors).
- **Prerequisites:** I.* (NMDA channels).
- **Citation:** Kandel 6e Ch 54 pp 1342–1347, Figs 54-3, 54-4.
- **Behavioral validation:** APV blockade abolishes LTP at Schaffer; CA1-NR1 KO mice show place fields but unstable across sessions.

### J.36 Mossy fiber LTP — presynaptic, NMDA-independent, nonassociative
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.53]*

- **System:** DG granule → CA3 mossy fiber terminals; large boutons, presynaptic Ca²⁺-AC-cAMP-PKA cascade.
- **Biological role:** Strengthening here is purely presynaptic (more glutamate per spike), does not require postsynaptic activity. Provides high-gain "detonator" input to CA3 that can drive sparse CA3 firing on novel events.
- **Sim status:** missing — current STDP rule is exclusively postsynaptic-NMDA Hebbian; no presynaptic-cAMP plasticity rule. Detonator behavior could be approximated with very high `weight_mean` and short STP recovery.
- **Cluster:** J primary, D secondary.
- **Prerequisites:** none for approximation; full implementation needs presynaptic plasticity rule.
- **Citation:** Kandel 6e Ch 54 pp 1342–1343, Fig 54-2C.
- **Behavioral validation:** APV does NOT block; PKA inhibitor H-89 does; LTP visible in EPSC under voltage-clamp.

### J.37 Direct perforant path LTP — mixed NMDAR + L-type Ca²⁺ channel
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.54]*

- **System:** EC layer III → CA1 distal dendrites; partial APV block, completed by APV+nifedipine.
- **Biological role:** Distinct induction biophysics — voltage-gated Ca²⁺ entry contributes alongside NMDAR. Reflects the distal dendritic location where local depolarization is weaker.
- **Sim status:** missing — no L-type Ca²⁺ channel kernel; HH preset list (`HH_*`) doesn't include L-type-only-driven plasticity coupling.
- **Cluster:** J primary, I secondary.
- **Prerequisites:** L-type Ca channel addition.
- **Citation:** Kandel 6e Ch 54 p 1343, Fig 54-2B.
- **Behavioral validation:** APV partial block; APV+nifedipine full block.

### J.38 Late-LTP — protein synthesis, CREB, PKMζ, structural
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.55]*

- **System:** Same CA3→CA1 synapse; ≥3 trains of 100 Hz tetanus; cAMP→PKA→MAPK→CREB-1 transcription; PKMζ translation; growth of new synaptic contacts.
- **Biological role:** Converts hours-scale early-LTP into days/weeks-scale late-LTP via gene expression and structural change. Required for long-term spatial memory; PKA inhibitor selectively abolishes late-LTP and place-field stability beyond ~1 hour.
- **Sim status:** missing — STDP weight changes are immediate and persistent in `cp_synaptic_weights`; no early/late phase distinction, no protein-synthesis dependence, no structural growth tied to repeated tetanus. Structural plasticity exists but not phase-gated.
- **Cluster:** J primary, L (structural), D secondary.
- **Prerequisites:** time-windowed potentiation accumulator.
- **Citation:** Kandel 6e Ch 54 pp 1346–1347, Fig 54-5.
- **Behavioral validation:** Anisomycin or PKA-inhibitor transgene blocks late-LTP and destabilizes place fields after ~1 hour while sparing initial formation.

### J.39 LTD in hippocampus — flexibility / saturation prevention
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from J.56]*

- **System:** Low-frequency stimulation (1 Hz) of Schaffer collaterals; modest, prolonged Ca²⁺ rise activates phosphatases (PP1) instead of kinases.
- **Biological role:** Prevents LTP saturation, enables remapping when environment changes. LTD-deficient mice cannot relearn new platform locations in Morris water maze (perseverate at old location).
- **Sim status:** implemented — STDP rule already produces depression for post-before-pre pairings; soft-bound `stdp_w_max` prevents saturation. Frequency-dependent LTD (1 Hz LFS) not separately tested.
- **Cluster:** J primary.
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 54 pp 1356–1357.
- **Behavioral validation:** Reversal learning in water maze; PP1 inhibition blocks LTD and reversal.

## Cluster K — additions

### K.01 Phototransduction cascade — rods & cones (G-protein cascade)
*[from Part IV — Perception (Ch 17-29); renumbered from K.50]*

- **System:** retinal photoreceptors (outer segments)
- **Biological role:** Photons isomerize 11-cis-retinal in opsin → activates transducin (Gαt) → PDE6 hydrolyzes cGMP → cyclic-nucleotide-gated (CNG) channels close → photoreceptor *hyperpolarizes* (sign-inverted vs. typical receptor cells). High amplification (1 photon ≈ 10^5 cGMP hydrolyzed) and ~100 ms response.
- **Sim status:** missing — no photoreceptor model, no light input, no graded hyperpolarizing transducer cells. Sim has only abstract "beacon sensors."
- **Cluster:** K (primary), E (secondary — feeds bipolar/ganglion encoding)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 22 p ~579–585
- **Behavioral validation:** intensity-response curve (Naka-Rushton), light/dark adaptation (~10^9 dynamic range), ~10 ms latency at high intensity vs ~300 ms scotopic.

### K.02 Phototransduction adaptation (Ca2+ feedback)
*[from Part IV — Perception (Ch 17-29); renumbered from K.51]*

- **System:** photoreceptor outer segment
- **Biological role:** When CNG channels close, Ca2+ influx drops; low Ca2+ disinhibits guanylate cyclase (via GCAPs) and modulates recoverin/rhodopsin kinase, accelerating shutoff and resetting cGMP. Implements the Weber-law-like background adaptation.
- **Sim status:** missing — no Ca2+-driven gain control on any sensor.
- **Cluster:** K
- **Prerequisites:** K.50
- **Citation:** Kandel 6e Ch 22 p ~585–588
- **Behavioral validation:** Weber's law for incremental thresholds; recovery half-time after bleach.

### K.03 Hair cell mechanotransduction — tip-link / MET channel
*[from Part IV — Perception (Ch 17-29); renumbered from K.52]*

- **System:** cochlea (auditory) and vestibular hair cells
- **Biological role:** Stereocilia deflection toward tallest row tensions tip-links (cadherin-23 / protocadherin-15), gating mechanoelectrical transduction (MET) channels (TMC1/2). K+ influx from endolymph depolarizes; sub-millisecond kinetics enable phase-locking to acoustic frequency. Adaptation via myosin-1c slow + Ca2+ fast pathways.
- **Sim status:** missing — no mechanotransducer model; no endolymphatic K+ battery.
- **Cluster:** K (primary), I (channels)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 26 p ~602–610; Ch 27 p ~636–640
- **Behavioral validation:** sub-µm displacement threshold; phase-locking up to ~3 kHz; rapid (~ms) and slow (~10 ms) adaptation.

### K.04 Cochlear traveling wave & tonotopy
*[from Part IV — Perception (Ch 17-29); renumbered from K.53]*

- **System:** basilar membrane / organ of Corti
- **Biological role:** Frequency-dependent mechanical traveling wave on basilar membrane: high freq peaks at base (stiff/narrow), low freq at apex. Establishes place code converted to neural tonotopy in spiral ganglion → cochlear nuclei → auditory cortex.
- **Sim status:** missing — no cochlear filterbank front-end.
- **Cluster:** K (primary), E (tonotopic maps)
- **Prerequisites:** K.52
- **Citation:** Kandel 6e Ch 26 p ~599–605
- **Behavioral validation:** characteristic frequency map; psychophysical critical bands.

### K.05 Outer hair cell electromotility (cochlear amplifier)
*[from Part IV — Perception (Ch 17-29); renumbered from K.54]*

- **System:** outer hair cells (prestin)
- **Biological role:** Voltage-driven length changes via prestin protein boost basilar-membrane vibration ~100-1000×, producing ~40 dB threshold reduction and sharp tuning. Loss = ~50 dB hearing loss + flat tuning curves.
- **Sim status:** missing
- **Cluster:** K
- **Prerequisites:** K.53
- **Citation:** Kandel 6e Ch 26 p ~614–618
- **Behavioral validation:** tuning curve sharpness Q10dB; otoacoustic emissions; compressive nonlinearity.

### K.06 Vestibular hair-cell directional sensitivity
*[from Part IV — Perception (Ch 17-29); renumbered from K.55]*

- **System:** semicircular canals + otolith organs (utricle, saccule)
- **Biological role:** Hair cells in canals encode angular acceleration (cupula deflection by endolymph inertia); otoliths encode linear acceleration / gravity (otoconia mass loading). Bidirectional response (depolarize / hyperpolarize) about resting discharge.
- **Sim status:** missing — no inertial / gravitational input.
- **Cluster:** K (primary), H (gaze/posture downstream)
- **Prerequisites:** K.52
- **Citation:** Kandel 6e Ch 27 p ~630–640
- **Behavioral validation:** VOR gain ~1.0; sinusoidal head rotation phase; tilt → tonic firing change.

### K.07 Mechanoreceptor types & adaptation classes (Pacinian / Meissner / Merkel / Ruffini)
*[from Part IV — Perception (Ch 17-29); renumbered from K.56]*

- **System:** glabrous + hairy skin
- **Biological role:** Four cutaneous afferent classes split by adaptation × receptive-field size: SA1 (Merkel, slow, small RF — pressure/edges), SA2 (Ruffini, slow, large — skin stretch), RA1 (Meissner, fast, small — flutter ~5–50 Hz), RA2/PC (Pacinian, fast, large — vibration ~100–300 Hz). Together span ~0.4 Hz to ~500 Hz.
- **Sim status:** missing — no skin / mechanoreceptor abstraction; no force or vibration input modality.
- **Cluster:** K (primary), E (somatosensory cortex)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 18 p ~454–462; Ch 19 p ~498–510
- **Behavioral validation:** two-point discrimination ≈1 mm fingertip; vibration JND; tactile frequency tuning by class.

### K.08 Piezo1 / Piezo2 mechanosensitive channels
*[from Part IV — Perception (Ch 17-29); renumbered from K.57]*

- **System:** somatosensory + visceral
- **Biological role:** Trimeric cation channels gated by membrane tension; Piezo2 is the primary transducer for light touch (Merkel-cell complex) and proprioception. Loss-of-function abolishes touch and proprioception.
- **Sim status:** missing — no membrane-tension-gated channel.
- **Cluster:** K (primary), I (channels)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 18 p ~458–462
- **Behavioral validation:** touch/proprioception loss in PIEZO2 KO; rapidly adapting current response to indentation ramp.

### K.09 Nociceptor classes & TRP channel transduction
*[from Part IV — Perception (Ch 17-29); renumbered from K.58]*

- **System:** Aδ + C fibers in skin / viscera
- **Biological role:** Free nerve endings express TRPV1 (heat >43°C, capsaicin), TRPM8 (cold, menthol), TRPA1 (irritants, cold, mustard oil), ASICs (acid). Transducer currents depolarize bare endings to threshold. Polymodal C-fibers integrate thermal + chemical + mechanical noxious input.
- **Sim status:** missing — no temperature / chemical / damage modality.
- **Cluster:** K (primary), I (TRP channels), O (pain → motivation)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 18 p ~470–474; Ch 20 p ~531–537
- **Behavioral validation:** thermal threshold ~43°C; capsaicin desensitization; QST psychophysics.

### K.10 Olfactory transduction (GPCR cAMP cascade)
*[from Part IV — Perception (Ch 17-29); renumbered from K.59]*

- **System:** olfactory sensory neurons (OE)
- **Biological role:** Each OSN expresses ONE odorant receptor (OR, ~400 in human) → Golf → adenylyl cyclase III → cAMP gates CNG channel → Ca2+-activated Cl- amplification. Combinatorial code: each odorant activates a *pattern* of ORs.
- **Sim status:** missing — no chemical / odor modality.
- **Cluster:** K (primary), E (glomerular map)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 29 p ~715–722
- **Behavioral validation:** combinatorial activation maps in olfactory bulb glomeruli; concentration-response curves.

### K.11 Gustatory transduction (5 modalities, GPCR + ion-channel)
*[from Part IV — Perception (Ch 17-29); renumbered from K.60]*

- **System:** taste receptor cells, taste buds
- **Biological role:** Sweet (T1R2/T1R3), umami (T1R1/T1R3), bitter (T2R family) — GPCR/PLCβ2/IP3/TRPM5 cascade releasing ATP onto afferents. Salty — ENaC. Sour — proton-sensing channels (Otop1). Each modality typed to distinct cell, but afferents may sum.
- **Sim status:** missing
- **Cluster:** K (primary), O (hedonic valence)
- **Prerequisites:** none
- **Citation:** Kandel 6e Ch 29 p ~725–733
- **Behavioral validation:** modality-selective lesions; T1R/T2R KO loss of taste class.

### K.12 Proprioceptors (muscle spindle, GTO, joint receptors)
*[from Part IV — Perception (Ch 17-29); renumbered from K.61]*

- **System:** muscle / tendon / joint
- **Biological role:** Muscle spindle (Ia + II afferents) encodes muscle length + rate; Golgi tendon organ (Ib) encodes tendon force; joint receptors signal extreme angles. Proprioceptive afferents are essential for limb position sense and motor control loops.
- **Sim status:** missing — no muscle/skeleton model. Spinal cord profile exists but receives no afferent input.
- **Cluster:** K (primary), H (motor)
- **Prerequisites:** K.57
- **Citation:** Kandel 6e Ch 18 p ~462–470
- **Behavioral validation:** stretch reflex (Ia → α-MN monosynaptic); force feedback (Ib di-synaptic inhibition).

---

## Cluster E — Sensory perception & cortical encoding

### K.13 Muscle spindle Ia/II afferent — proprioception
*[from Part V — Movement (Ch 30-39); renumbered from K.50]*

- **System:** intrafusal fiber capsule with primary (Ia) and secondary (II) afferent endings.
- **Biological role:** Ia: dynamic + static length; II: static length only. Source signal for stretch reflex (H.55) and proprioceptive sense of limb position. γ-MN tunes sensitivity (H.59).
- **Sim status:** **missing** — no spindle / Ia / II afferent class.
- **Cluster:** K primary; H secondary.
- **Prerequisites:** H.50, H.54.
- **Citation:** Kandel 6e Ch 32 p 763–770.
- **Behavioral validation:** Ramp-and-hold stretch → Ia phasic+tonic, II tonic only.

### K.14 Golgi tendon organ Ib — force / tension
*[from Part V — Movement (Ch 30-39); renumbered from K.51]*

- **System:** encapsulated mechanoreceptor in series with collagen at musculotendinous junction.
- **Biological role:** Senses muscle force; activates Ib autogenic inhibition (H.58). Force feedback for grip control and fall protection.
- **Sim status:** **missing**.
- **Cluster:** K primary; H secondary.
- **Prerequisites:** H.51.
- **Citation:** Kandel 6e Ch 32 p 770–772.
- **Behavioral validation:** Force step → Ib firing rate proportional to active tension.

### K.15 Vestibular afferents — head linear / angular acceleration
*[from Part V — Movement (Ch 30-39); renumbered from K.52]*

- **System:** semicircular canals (angular), otolith organs (linear); hair-cell mechanoreceptors → vestibular ganglion.
- **Biological role:** Drives VOR (F.58), postural reflexes, sense of orientation. Fundamental input for cerebellar prediction of self-motion (forward-model attenuation, F.56).
- **Sim status:** **missing**.
- **Cluster:** K primary; F secondary.
- **Prerequisites:** I.*.
- **Citation:** Kandel 6e Ch 35 p 884–887.
- **Behavioral validation:** Head rotation → afferent firing modulation, drives compensatory eye movement.

---

## Cluster O / Cluster L — Misc

## Cluster L — additions

### L.07 Spinal CPG developmental specification — V0/V1/V2/V3 interneuron classes
*[from Part V — Movement (Ch 30-39); renumbered from L.50]*

- **System:** dorso-ventral patterning genes (Pax / Lim / Sim1 / Shox2) define interneuron identity in developing spinal cord.
- **Biological role:** V0 commissural (left-right alternation); V1 ipsilateral inhibitory (Renshaw, Ia-IN); V2a glutamatergic excitatory (rhythm); V3 commissural excitatory. Genetic deletion of each class produces specific gait deficits (e.g. V0 KO → bunny-hopping mouse).
- **Sim status:** **missing** — no developmental specification module; CPG itself missing (H.62).
- **Cluster:** L primary; H secondary.
- **Prerequisites:** H.62, H.63.
- **Citation:** Kandel 6e Ch 33 p 793–800 (Kiehn).
- **Behavioral validation:** V0 KO → loss of left-right alternation; V2a perturbation → loss of speed control.

---

## Cluster H (continued) — BMI / record-stimulate

### L.08 Neural induction by the Spemann organizer — BMP-inhibitor signaling
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.60]*

- **System:** embryonic ectoderm; dorsal blastopore lip / node organizer
- **Biological role:** Default ectodermal fate is epidermis; neural fate emerges only when BMP signaling is locally inhibited by organizer-secreted antagonists (noggin, chordin, follistatin). The "default model" — neural is what ectoderm becomes when BMP is silenced.
- **Sim status:** not-applicable — simulator initializes adult/postnatal-like network state directly; there is no ectoderm-equivalent substrate or default-fate machinery.
- **Cluster:** L (development)
- **Prerequisites:** none — predates anything we model
- **Citation:** Kandel 6e Ch 45 pp 1107-1110
- **Behavioral validation:** ectopic organizer transplant induces secondary neural axis (Spemann & Mangold 1924)

### L.09 Rostrocaudal patterning by Wnt / FGF / RA gradients — caudalization
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.61]*

- **System:** neural plate / early neural tube
- **Biological role:** Posteriorizing morphogen gradients (Wnt, FGF, retinoic acid from paraxial mesoderm) convert anterior-default neural tissue into progressively more caudal identities. Hox gene expression boundaries are read out of these gradients.
- **Sim status:** not-applicable — region identity is a name string in `BrainRegion(name=...)`, not a gradient computation. We do not model morphogen diffusion or Hox cascades.
- **Cluster:** L
- **Prerequisites:** L.60
- **Citation:** Kandel 6e Ch 45 pp 1110-1117
- **Behavioral validation:** RA exposure during gastrulation truncates rostral structures; Hox gene knockouts homeotically transform rhombomere identity

### L.10 Dorsoventral patterning by SHH / BMP opposing gradients
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.62]*

- **System:** neural tube; floor plate (SHH source) vs roof plate (BMP source)
- **Biological role:** Counter-gradients of Sonic hedgehog (ventral) and BMPs (dorsal) divide the neural tube into 11+ progenitor domains, each defined by a transcription-factor code that determines neuronal subtype (e.g. ventral motor neurons, dorsal sensory interneurons).
- **Sim status:** not-applicable — neuron *type* in our framework is a `NeuronType` enum slot assigned per region at construction. No spatial morphogen reading; no progenitor-domain combinatorics.
- **Cluster:** L
- **Prerequisites:** L.61
- **Citation:** Kandel 6e Ch 45 pp 1117-1126
- **Behavioral validation:** floor-plate ablation eliminates motor neurons; ectopic SHH ventralizes dorsal cord

### L.11 Hindbrain rhombomere segmentation — Hox + Eph/ephrin repulsion
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.63]*

- **System:** rhombencephalon; transient rhombomeres r1-r8
- **Biological role:** Krox20 turns on EphA4 in r3/r5 and ephrinB3 in r4; mutual repulsion at boundaries prevents cell mixing and locks in segment-specific Hox codes. Each rhombomere is a developmental compartment with distinct cranial-nerve outputs.
- **Sim status:** not-applicable — no embryonic compartmentalization. Region boundaries are index-slice boundaries, enforced trivially by `RegionManager`.
- **Cluster:** L
- **Prerequisites:** L.61, L.62
- **Citation:** Kandel 6e Ch 45 pp 1115-1117
- **Behavioral validation:** Hox paralog group knockouts collapse rhombomere identity; Krox20 KO loses r3/r5

### L.12 Neurogenesis from radial-glia stem cells — symmetric vs asymmetric division
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.64]*

- **System:** ventricular / subventricular zones of the developing neural tube
- **Biological role:** Radial glia act as neural stem cells, undergoing symmetric (population-expanding) then asymmetric (one stem + one neuron) divisions. The expanded subventricular zone in primates is the major contributor to cortical thickening.
- **Sim status:** not-applicable — `num_neurons` is fixed at simulation start; no proliferation, no division, no progenitor pool.
- **Cluster:** L
- **Prerequisites:** L.62
- **Citation:** Kandel 6e Ch 46 pp 1131-1140
- **Behavioral validation:** Notch pathway perturbation collapses progenitor pool prematurely (early neuron differentiation, microcephaly)

### L.13 Neuronal migration along radial glia — laminar cortex assembly
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.65]*

- **System:** developing neocortex; radial glia as scaffold
- **Biological role:** Newborn pyramidal neurons climb radial glial fibers in inside-out order — earliest-born settle in deep layers (L6), latest-born pass them and settle superficially (L2/3). Reelin signaling from Cajal-Retzius cells stops migration at the marginal zone.
- **Sim status:** not-applicable — laminar position is encoded only insofar as `BrainRegion` instances are named (e.g. `CORTEX_L23_RS_FS`, `CORTEX_L5_DEEP_OUTPUT`). No physical migration step.
- **Cluster:** L
- **Prerequisites:** L.64
- **Citation:** Kandel 6e Ch 46 pp 1140-1146
- **Behavioral validation:** *reeler* mouse: cortical layers inverted; lissencephaly in human reelin/DCX/LIS1 mutations

### L.14 Tangential interneuron migration — ganglionic-eminence origin
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.66]*

- **System:** medial / caudal / lateral ganglionic eminences → cortex, hippocampus, olfactory bulb
- **Biological role:** Inhibitory interneurons are born subcortically and migrate tangentially into cortex, guided by slit/semaphorin/ephrin repellents and motogenic cues. A separate rostral migratory stream feeds olfactory bulb interneurons throughout adult life in many mammals.
- **Sim status:** not-applicable — E/I split is set at config time via `BrainRegion.exc_fraction`; inhibitory indices are selected deterministically by `RegionManager`. No tangential migration.
- **Cluster:** L
- **Prerequisites:** L.64, L.65
- **Citation:** Kandel 6e Ch 46 pp 1138-1141
- **Behavioral validation:** Dlx1/2 KO loses ganglionic-eminence-derived interneurons; cortex becomes hyperexcitable

### L.15 Neurotrophic-factor hypothesis — target-derived survival
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.67]*

- **System:** developing PNS / CNS, NGF / BDNF / NT-3 / NT-4 + Trk receptors
- **Biological role:** Roughly half of all neurons born during development die. Survival depends on retrograde-transported neurotrophin captured at axon terminals — neurons that fail to reach (or compete for limited) target supply undergo apoptosis. Matches neuron number to target size.
- **Sim status:** missing — we have no analogue. There is no neuron-survival score, no apoptosis pathway, no target-supply competition. Activity-dependent *synaptic* survival exists (structural plasticity / pruning, Cluster L close to project work) but cell-level death does not. Adding a per-neuron trophic budget (driven by post-synaptic activity at downstream targets) would be the closest analogue and is genuinely interesting for self-organizing population sizing.
- **Cluster:** L (primary), O (reward — trophic = persistent reward signal)
- **Prerequisites:** L.65, L.66
- **Citation:** Kandel 6e Ch 46 pp 1146-1158
- **Behavioral validation:** anti-NGF antibodies in neonates ablate sympathetic ganglia; Bax KO rescues ~all developmental death

### L.16 Programmed cell death (apoptosis) — caspase cascade & Bcl-2 family
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.68]*

- **System:** all developing neurons
- **Biological role:** Cell death is an active program: Bax/Bak permeabilize mitochondria → cytochrome c release → Apaf-1 / caspase-9 / caspase-3 cascade → orderly self-disassembly. Trophic-factor signaling suppresses the program; removal triggers it. Conserved from C. elegans (ced-3/4/9).
- **Sim status:** not-applicable / missing — no neuron is ever removed at runtime. (Synapse death exists via structural-plasticity pruning; cell death does not.) Only meaningful if we add L.67.
- **Cluster:** L
- **Prerequisites:** L.67
- **Citation:** Kandel 6e Ch 46 pp 1156-1162
- **Behavioral validation:** caspase-3 KO mice have dramatic embryonic CNS hyperplasia

### L.17 Growth cone guidance — netrins, semaphorins, ephrins, slits
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.69]*

- **System:** all extending axons during development
- **Biological role:** A small toolkit of conserved cue families (netrin/DCC + UNC5; semaphorin/plexin; ephrin/Eph; slit/Robo) provides attractive and repulsive guidance. Combined with intermediate targets (e.g. floor plate, optic chiasm) they wire long-distance projections. Ephrin gradients on tectum + retina set up topographic visual maps (chemoaffinity hypothesis, Sperry).
- **Sim status:** not-applicable — connectivity is declarative: `RegionPathway(from_region, to_region, density, weight_mean, ...)`. There are no axons, no growth cones, no guidance gradients. Topographic projections, if needed, would have to be constructed by a connectivity generator that places synapses according to declared spatial coordinates.
- **Cluster:** L
- **Prerequisites:** L.62, L.65
- **Citation:** Kandel 6e Ch 47 pp 1165-1196
- **Behavioral validation:** netrin/DCC KO: commissural axons fail to cross midline; EphA KO: temporal retinal axons mis-target rostral tectum

### L.18 Topographic mapping by Eph/ephrin gradients (chemoaffinity)
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.70]*

- **System:** retinotectal projection (canonical); also somatosensory, auditory tonotopy
- **Biological role:** Continuous gradients of Eph receptors (on RGC axons) read continuous gradients of ephrin ligands (on tectum) to produce a smooth topographic map. Activity-dependent refinement (Hebbian) sharpens the initial coarse map.
- **Sim status:** not-applicable for the gradient step; partial for refinement — STDP-driven map sharpening is supported, so if a coarse topographic projection were *constructed at config time*, plasticity could refine it. We currently build pathways with random-within-density connectivity, no topography.
- **Cluster:** L (primary), G (Hebbian/STDP for refinement)
- **Prerequisites:** L.69
- **Citation:** Kandel 6e Ch 47 pp 1183-1196
- **Behavioral validation:** ephrin-A2/A5 double KO: distorted retinotectal map; activity blockade prevents map sharpening

### L.19 Critical period for ocular dominance — activity-dependent refinement
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.71]*

- **System:** primary visual cortex (V1) layer IV; LGN inputs
- **Biological role:** During an early postnatal window, monocular deprivation shifts cortical territory toward the open eye via STDP/Hebbian competition between thalamocortical inputs. Window closure depends on inhibitory (PV interneuron) maturation and ECM (perineuronal nets, Nogo).
- **Sim status:** **partial — this is the project's closest analogue.** Per-pathway plasticity gates (`RegionPathway.plasticity_gate`, `cp_plasticity_gain`, `bridge.set_plasticity_gate`) are exactly the runtime knob a critical period needs. The 2026-04-27 curriculum (cortex_to_d1 plastic + inputs frozen → cortex frozen + inputs thawed) is functionally a staged critical period. Missing: the *biology of closure* — PV maturation, perineuronal nets, intrinsic mechanisms that ratchet the window shut without external command.
- **Cluster:** L (primary), G (plasticity)
- **Prerequisites:** L.69, L.70
- **Citation:** Kandel 6e Ch 49 (cross-ref) pp 1248-1258 (covered separately); see also Ch 47 for the connectivity substrate
- **Behavioral validation:** monocular deprivation in kittens during weeks 4-12 → permanent amblyopia; chondroitinase digestion of perineuronal nets reopens the window in adult rat (Pizzorusso 2002)

### L.20 Adult neurogenesis — dentate gyrus & subventricular zone
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.72]*

- **System:** hippocampal DG (granule cells); SVZ → rostral migratory stream → olfactory bulb interneurons
- **Biological role:** Two persistent neurogenic niches in mammals. New DG granule cells integrate over weeks and contribute to pattern separation and possibly to forgetting / memory clearance. SVZ neurons feed olfactory bulb in many species (sparse in adult human).
- **Sim status:** missing — `num_neurons` is fixed and there's no insertion mechanism. The closest infrastructure is structural plasticity (synapse birth/death) — extending it to *neuron* birth/death is non-trivial because GPU arrays are pre-allocated. A capped-pool design (some neurons start "silent" and get recruited) would be a tractable approximation. Likely Cluster D (memory) or new Cluster L sub-feature, given DG-specific role in pattern separation.
- **Cluster:** L (primary), D (memory — DG pattern separation)
- **Prerequisites:** L.64
- **Citation:** Kandel 6e Ch 46 pp 1137-1138, 1144-1146
- **Behavioral validation:** ablation of adult hippocampal neurogenesis impairs context discrimination and pattern separation tasks; running and enriched environment increase DG neurogenesis

### L.21 Neural-crest derivation of the PNS
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.73]*

- **System:** neural crest → DRG sensory neurons, sympathetic & parasympathetic ganglia, enteric NS, Schwann cells, melanocytes, craniofacial skeleton
- **Biological role:** A multipotent, migratory stem-cell population originating at the dorsal neural tube boundary. Migration paths through somite anterior-half compartments (ephrinB-regulated) determine ultimate fate.
- **Sim status:** not-applicable — no peripheral nervous system in the simulator at all. PNS is out-of-scope.
- **Cluster:** L
- **Prerequisites:** L.62
- **Citation:** Kandel 6e Ch 46 pp 1140-1144
- **Behavioral validation:** Hirschsprung disease (RET / EDNRB mutations) — failed enteric neural crest migration

## Cluster Q — Glia & repair

### L.22 Hormonal organization of sexually dimorphic nuclei
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from L.74]*

- **System:** SRY → testis → testosterone (perinatal surge) → aromatized to estradiol in CNS → masculinized circuits (e.g. SDN-POA, BNST, song-control nuclei in birds)
- **Biological role:** Gonadal steroids exert *organizational* effects during a critical perinatal window — permanently shaping circuit volume, cell number, and connectivity — and *activational* effects later (e.g. menstrual-cycle modulation).
- **Sim status:** not-applicable — no body, no gonads, no genome. Sex differentiation is fully out of scope.
- **Cluster:** L
- **Prerequisites:** L.67, L.68 (perinatal organizational changes are largely apoptosis-based)
- **Citation:** Kandel 6e Ch 51 pp 1260-1295
- **Behavioral validation:** neonatal castration of male rats shrinks SDN-POA to female size; perinatal testosterone in females masculinizes mounting behavior

---

## Notes on project relevance

The catalog above is mostly `not-applicable` — appropriately, since the
simulator deliberately starts at a postnatal/adult level of abstraction.
Three entries do bear on active project work:

1. **L.67 Neurotrophic-factor / target-derived survival** — currently
   `missing`. A per-neuron trophic budget driven by downstream activity
   would parallel the existing structural-plasticity machinery and could
   self-tune population sizes. Worth filing as a future experiment.
2. **L.71 Critical-period activity-dependent refinement** — `partial`,
   and arguably the closest biological analogue of the project's
   curriculum + plasticity-gate infrastructure. The 2026-04-27 curriculum
   IS a staged critical period; what's missing is biological *closure*
   (PV maturation, perineuronal nets, Nogo) — currently we just call
   `set_plasticity_gate` from the runner.
3. **L.72 Adult neurogenesis (DG)** — `missing`. Pattern separation in
   DG depends on ongoing neurogenesis; if we ever model a hippocampus
   capable of pattern separation beyond what static recurrence provides,
   a capped-pool "neuron recruitment" mechanism may be necessary.
4. **Q.62 CNS regeneration failure** — `not-applicable` directly, but
   provides biological cover for the simulator's design choice that
   connectivity is config-time and plasticity is runtime. Worth
   citing in design docs when defending the no-regrowth model.

The remaining ~16 entries (patterning, migration, axon guidance, sex
differentiation) document the developmental machinery that produces
the *initial conditions* of our simulated network — a network we
declare via `BrainRegion`/`RegionPathway` rather than grow.

### L.23 Critical period for language — perceptual narrowing in first year
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from L.50]*

- **System:** Auditory cortex + frontal speech areas; experience-dependent pruning of phonetic discriminability.
- **Biological role:** Infants <6 mo discriminate all human-language phonemes ("universalist"); by 12 mo, discrimination of non-native contrasts drops dramatically. After puberty, second-language acquisition rarely achieves native phonology.
- **Sim status:** missing — generic critical-period concept matches `--curriculum` and plasticity-gate infrastructure (e.g., `--curriculum-phase2-cortex-gain`), but no language-specific application.
- **Cluster:** L primary, G secondary.
- **Prerequisites:** plasticity gating (already implemented).
- **Citation:** Kandel 6e Ch 55 pp 1372–1376.
- **Behavioral validation:** Mismatch-negativity ERP for non-native contrasts attenuates between 6 and 12 months.

## Cluster N — additions

### N.01 Ascending Arousal System Architecture — parabrachial / PPT / basal forebrain
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.50]*

- **System:** Glutamatergic neurons in dorsolateral pons (parabrachial + PPT) → basal forebrain → cortex; supplemented by monoamines (LC, raphe, TMN), orexin, ACh.
- **Biological role:** **The wake circuit.** Bilateral lesion of parabrachial-PPT or basal forebrain causes coma; lesions of *any single monoamine group alone* do NOT abolish wakefulness, indicating the parabrachial-basal-forebrain glutamatergic pathway is the essential backbone, monoamines are augmentation.
- **Sim status:** missing — no arousal-state variable beyond time-of-day. Sleep-replay infrastructure exists but is triggered programmatically, not by an arousal-state circuit.
- **Cluster:** N primary, C secondary (monoamine augmentation), O secondary.
- **Prerequisites:** C.50-C.55 (monoamines).
- **Citation:** Kandel 6e Ch 40 pp 1003-1006 (Fig 40-15); Ch 44 pp 1083-1085 (Fig 44-3).
- **Behavioral validation:** Animals with parabrachial-PPT lesions enter persistent coma; same animals with isolated LC or raphe lesions remain awake, just less alert.

### N.02 VLPO Sleep-Promoting Nucleus — GABA + galanin sleep ON-switch
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.51]*

- **System:** Ventrolateral preoptic nucleus + median preoptic nucleus (anterior hypothalamus).
- **Biological role:** GABA + galanin co-release inhibits the entire ascending arousal system. Fires slowest awake, fastest in deep sleep — opposite of monoamines. Mutual inhibition with arousal nuclei produces a flip-flop bistable. von Economo's encephalitis lethargica observation: anterior hypothalamic damage → severe insomnia.
- **Sim status:** missing — sleep stages are scheduled, not generated by a competing ON-switch. The flip-flop dynamic itself (rapid bistable transitions, low time spent in intermediate states) is also missing — important behaviorally because real animals don't dwell in drowsy states.
- **Cluster:** N primary, O secondary (homeostasis).
- **Prerequisites:** N.50 (arousal targets).
- **Citation:** Kandel 6e Ch 44 pp 1085-1087 (Fig 44-4).
- **Behavioral validation:** Lesion of VLPO → fragmented sleep with up to 50% loss of total sleep time; correlation with sleep fragmentation in elderly (postmortem VLPO galanin neuron count predicts antemortem fragmentation).

### N.03 Wake-Sleep Flip-Flop Switch — mutual-inhibition bistable
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.52]*

- **System:** VLPO/MNPO ↔ ascending arousal system mutual inhibition.
- **Biological role:** Bistable mutual-inhibition circuit forces near-binary state transitions. Animals spend almost all time clearly awake or clearly asleep, very little in drowsy/transitional state — this is *adaptive* (vulnerability minimization). Orexin neurons stabilize the wake side (loss → narcolepsy with state instability).
- **Sim status:** missing — sleep stages are explicit scheduled phases, not emergent from competing populations. Could be valuable for biologically realistic transition dynamics, especially if the project ever models attention lapses or drowsiness.
- **Cluster:** N primary.
- **Prerequisites:** N.50, N.51.
- **Citation:** Kandel 6e Ch 44 pp 1085-1087 (Fig 44-4B).
- **Behavioral validation:** Hypnogram measurement: healthy adults spend <5% of nighttime in transitional states.

### N.04 REM Flip-Flop (Subceruleus / vlPAG) — second pontine bistable
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.53]*

- **System:** Subceruleus / sublaterodorsal area (REM-on glutamatergic) ↔ ventrolateral periaqueductal gray (REM-off GABA) — a separate flip-flop nested inside non-REM sleep.
- **Biological role:** Generates 90-min NREM↔REM cycle. REM-on cells in subceruleus drive both EEG desynchronization (via PPT/LDT/basal-forebrain) and motor atonia (via ventromedial medulla → spinal motor inhibition). REM-off vlPAG GABAergic cells gate this. Monoaminergic LC/raphe inhibit REM-on cells, which is why SSRI/NRI antidepressants suppress REM.
- **Sim status:** partial — sleep-replay stage alternation exists; biological flip-flop generator does not. Project could potentially model 90-min cycle as a stochastic bistable for more realistic replay scheduling.
- **Cluster:** N primary, C secondary (monoamine inhibition).
- **Prerequisites:** N.51 (flip-flop pattern), C.50/C.51 (monoamine REM suppression).
- **Citation:** Kandel 6e Ch 44 pp 1086-1088 (Fig 44-5).
- **Behavioral validation:** Subceruleus lesion → REM atonia abolished (REM behavior disorder model); pontine cholinergic agonist → forced REM.

### N.05 Slow Oscillation (Up/Down States) — NREM cortical 0.5-1 Hz
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.54]*

- **System:** Cortex-intrinsic; survives in isolated cortical slab.
- **Biological role:** During NREM, cortical pyramidals oscillate between Up (depolarized, firing) and Down (hyperpolarized, silent) states at ~0.5-1 Hz, producing EEG slow waves. This is generated by recurrent excitatory + inhibitory circuitry within cortex, not by thalamic input. Drives the synchronized "frame" within which spindles and ripples are coordinated.
- **Sim status:** missing — flagship doesn't generate Up/Down state alternation. Cortical replay scheduling is event-triggered. **Project-actionable:** add a NREM-mode that biases cortical baseline drive into a slow up/down regime and frames replay events on Up-state onsets.
- **Cluster:** N primary, J secondary (replay framing).
- **Prerequisites:** N.51 (NREM detection).
- **Citation:** Kandel 6e Ch 44 pp 1081-1083 (Fig 44-2A).
- **Behavioral validation:** Intracortical recording or surface EEG: 0.5-1 Hz slow waves with paired multi-unit Up/Down structure. Persists even with thalamus inactivated.

### N.06 Sleep Spindles (10-16 Hz) — thalamocortical reticular ↔ relay
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.55]*

- **System:** Thalamic reticular nucleus (TRN) ↔ thalamocortical relay; T-type Ca²⁺ low-threshold spike mechanism.
- **Biological role:** Stage N2 hallmark. TRN burst hyperpolarizes relay neurons, de-inactivating T-type Ca²⁺ channels; rebound burst when hyperpolarization wanes; resulting Ca²⁺ spike triggers Na⁺ burst that re-excites TRN — closing 100-ms cycle that recurs 12-14×/sec for ~1-2 sec spindle. **Strongly correlated with overnight motor-memory consolidation** (Stickgold).
- **Sim status:** missing — flagship has no thalamic relay vs. reticular distinction; no T-type Ca²⁺ channel; no spindle generator. **Project-actionable for replay:** spindles "open windows" for cortical-hippocampal coordination — adding even a coarse spindle phase variable could nest hippocampal replay events for biologically plausible consolidation timing.
- **Cluster:** N primary, J secondary (memory consolidation), I secondary (T-type channel).
- **Prerequisites:** N.54 (Up state framing), I.x (channels).
- **Citation:** Kandel 6e Ch 44 pp 1081-1083 (Fig 44-2B).
- **Behavioral validation:** EEG: 10-16 Hz waxing/waning spindle bursts in N2; spindle density correlates with motor sequence task improvement after sleep.

### N.07 Hippocampal Sharp-Wave Ripples (SWRs) — NREM replay events ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.56]*

- **System:** Hippocampal CA3/CA1 high-frequency (140-250 Hz) population bursts during NREM and quiet wakefulness.
- **Biological role:** **Most directly relevant to project's replay implementation.** SWRs compress sequences of place cells in correct or reverse order at ~10-20× speed. Disrupting SWRs during sleep impairs spatial-memory consolidation. SWRs co-occur with cortical Up states / spindle troughs, allowing hippocampus → cortex transfer of compressed sequences.
- **Sim status:** partial — sleep-replay infrastructure exists, but Kandel-style time-compressed sequence replay isn't explicitly modeled. **Top-3 actionable:** the existing replay infra could be upgraded to (a) generate compressed sequences from recent active place-cell trajectories, (b) phase-lock replay events to NREM-stage windows, (c) add reverse-replay variant. Biologically grounded and would test the hypothesis that *replay quality, not replay quantity, is the bottleneck*.
- **Cluster:** N primary, D secondary (hippocampus), J secondary (consolidation).
- **Prerequisites:** D.x (hippocampus), N.54/N.55 (NREM framing), J.x (plasticity replay).
- **Citation:** Kandel 6e Ch 44 pp 1090-1092 (text on Stickgold consolidation; SWR mechanism is fully covered in Ch 54 of Part VIII but referenced here).
- **Behavioral validation:** Closed-loop SWR disruption (electrical stim triggered on detected ripple) during post-task sleep → impaired next-day spatial memory. Forward AND reverse replay observed in CA1.

### N.08 Adenosine Sleep Pressure — humoral homeostat ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.57]*

- **System:** Brain-wide extracellular adenosine (ATP→ADP→AMP→adenosine accumulation during waking metabolism); A1 receptors inhibitory on wake-promoting neurons; A2A receptors excitatory on VLPO.
- **Biological role:** **Process-S** of two-process sleep model. Adenosine accumulates monotonically during wake (ATP dephosphorylation from sustained metabolism), inhibits LC/TMN/orexin via A1, and *activates* VLPO via A2A in NAc-shell projection. Caffeine works by blocking A1/A2A. Provides the *homeostatic* drive distinct from circadian.
- **Sim status:** missing — no homeostatic sleep-pressure variable. Sleep stages are scheduled rather than triggered by accumulated drive.
- **Cluster:** N primary, O secondary (homeostasis).
- **Prerequisites:** N.50 (arousal targets), N.51 (VLPO).
- **Citation:** Kandel 6e Ch 44 pp 1086-1087.
- **Behavioral validation:** Sleep-deprivation rebound: animals/humans deprived of sleep pay back missed sleep with deeper N3; caffeine antagonizes this.

### N.09 Two-Process Model (Process-S × Process-C) — homeostatic × circadian
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.58]*

- **System:** Sleep pressure (Process-S, adenosine-driven) interacts with circadian wake drive (Process-C, SCN-driven via DMH).
- **Biological role:** S rises monotonically during wake, falls during sleep. C is sinusoidal 24-hr peak in late-day to dip in pre-dawn. Sleep onset = C drops below S's threshold; waking = S falls below C threshold. Mid-afternoon dip in C produces siesta tendency. Pre-dawn rise of C-promoting-sleep prevents waking before homeostatic recovery is complete.
- **Sim status:** missing.
- **Cluster:** N primary.
- **Prerequisites:** N.57 (Process-S), N.59 (Process-C).
- **Citation:** Kandel 6e Ch 44 pp 1087-1088 (Fig 44-6A).
- **Behavioral validation:** Forced-desynchrony protocol: subjects on non-24-hr schedules show separable circadian and homeostatic components in alertness and sleep propensity.

### N.10 Suprachiasmatic Nucleus Master Clock — BMAL1/CLOCK/PER/CRY loop
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.59]*

- **System:** SCN (anterior hypothalamus, above optic chiasm); melanopsin retinal ganglion cell input.
- **Biological role:** ~20K GABAergic neurons; intracellular transcriptional-translational feedback loop with BMAL1+CLOCK as positive limb (heterodimer binds E-box, drives Per1/2 + Cry1/2 transcription); PER+CRY proteins return to nucleus and disrupt BMAL1/CLOCK dimer (negative limb); cycle period ~24.1 hr. SCN-cell-to-SCN-cell coupling (gap junctions + GABAergic) synchronizes population. Drives wake-sleep via SCN → subparaventricular zone → DMH → orexin/VLPO.
- **Sim status:** missing.
- **Cluster:** N primary, O secondary.
- **Prerequisites:** N.50 (arousal targets).
- **Citation:** Kandel 6e Ch 44 pp 1087-1090 (Fig 44-6B, 44-7).
- **Behavioral validation:** Constant-dim-light free-run experiment: humans drift ~25.2 hr period; familial advanced sleep-phase syndrome (PER or CK1δ mutations) → ~22 hr period.

### N.11 Orexin / Hypocretin Stabilization — narcolepsy gene
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.60]*

- **System:** Posterior lateral hypothalamus (~70K neurons in human); two peptides from same precursor.
- **Biological role:** Excite all monoaminergic + cholinergic arousal nuclei AND inhibit REM-on cells (via vlPAG REM-off). Function as **state stabilizer** rather than wake-promoter per se — loss does not reduce total wake time but produces fragmented states (narcolepsy: easy daytime sleep onset, frequent night awakenings, REM intrusion into wake = cataplexy). Autoimmune-mediated loss in narcolepsy.
- **Sim status:** missing.
- **Cluster:** N primary, C secondary.
- **Prerequisites:** N.50, N.52.
- **Citation:** Kandel 6e Ch 44 pp 1093-1095 (Fig 44-9).
- **Behavioral validation:** Orexin-knockout mice + narcolepsy patients: same total sleep amount, but with frequent state transitions and REM-onset within minutes of sleep onset.

### N.12 Sleep-Dependent Memory Consolidation — Stickgold/Tononi ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.61]*

- **System:** Cortical synapses; bidirectional change during sleep stages.
- **Biological role:** Two complementary theories. (a) **Active replay/consolidation** (Stickgold): hippocampal replay during NREM SWRs + REM creative recombination → consolidates declarative + procedural memory. Subjects trained at night and tested next morning out-perform same-day 12-hour-no-sleep group. (b) **Synaptic homeostasis** (Tononi/Cirelli, SHY): wake produces net synaptic potentiation; sleep down-scales weak synapses while preserving strong ones, restoring signal-to-noise.
- **Sim status:** partial — synaptic scaling/homeostasis IS implemented. Replay infrastructure exists. Together they cover both SHY and active-replay theories at coarse level. Bottleneck: replay content quality (which traces get replayed).
- **Cluster:** N primary, J secondary (homeostasis), D secondary (hippo replay).
- **Prerequisites:** N.51, N.56 (SWR), J.x (homeostasis).
- **Citation:** Kandel 6e Ch 44 pp 1090-1092.
- **Behavioral validation:** Sleep-vs-wake-after-training paradigms; sleep selective for task type (REM helps perceptual; N2 helps motor sequence); post-sleep PSD changes track baseline sleep.

### N.13 Glymphatic Clearance During Sleep — ECF expansion
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.62]*

- **System:** Brain-wide CSF/ISF exchange via aquaporin-4-positive astrocytic endfeet around vessels.
- **Biological role:** During sleep, brain extracellular space expands (~60% increase), allowing CSF to perfuse and clear metabolic waste (notably β-amyloid). Sleep deprivation reduces β-amyloid clearance — proposed mechanism linking poor sleep to Alzheimer pathology.
- **Sim status:** missing — no metabolic clearance modeling. [discrepancy: largely irrelevant to current project but worth flagging if disease modeling (Cluster P) ever activates.]
- **Cluster:** N primary, P secondary (Alzheimer link), Q secondary (glia).
- **Prerequisites:** Q.x (glia).
- **Citation:** Kandel 6e Ch 44 pp 1095-1096.
- **Behavioral validation:** Two-photon imaging in mice: ECF volume fraction expands during NREM; tracer clearance halved by sleep deprivation.

---

## Cluster O — Emotion, Reward, Motivation (project-critical for Ch 41-43)

### N.14 Hippocampal–neocortical dialogue — systems consolidation
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from N.50]*

- **System:** SWRs (HC) ↔ slow oscillations + spindles (neocortex); fronto-parietal targets receive HC-replay-locked input during NREM.
- **Biological role:** Repeated coordinated reactivation gradually transfers memory from HC-dependent (recent) to neocortex-dependent (remote) state. Standard consolidation theory; multiple-trace theory adds episodic detail stays HC-dependent.
- **Sim status:** missing — neither SWR generator nor coordinated-thalamocortical-oscillation infrastructure is present.
- **Cluster:** N primary, D primary, G secondary.
- **Prerequisites:** D.68, cortical slow-oscillation generator, thalamic spindle population.
- **Citation:** Kandel 6e Ch 52 p 1299, Ch 54 p 1366; Buzsáki 1989 / Wilson-McNaughton 1994 (review).
- **Behavioral validation:** Time-graded retrograde amnesia after HC lesion (spares remote memories that have been consolidated).

## Cluster O — additions

### O.02 Phasic dopamine = reward prediction error (Schultz) — TD learning signal
*[from Part V — Movement (Ch 30-39); renumbered from O.50]*

- **System:** ventral tegmental area (VTA) and substantia nigra pars compacta (SNc) DA neurons → striatum, NAcc, PFC.
- **Biological role:** Tonic ~5 Hz firing. Phasic burst on unexpected reward; transfer to predictive cue with learning; **dip below baseline on reward omission**. Quantitatively matches temporal-difference RPE.
- **Sim status:** **partial** — broadcast DA from `current_reward_signal`; `--adaptive-da` adds asymmetric per-action gating with slow-positive / fast-negative tau (Schultz 1998 phasic asymmetry). RPE is not computed inside the simulator from a value function — externally supplied as `reward - baseline`.
- **Cluster:** O primary; C secondary.
- **Prerequisites:** A.50, J.*.
- **Citation:** Kandel 6e Ch 38 p 949–953 (Schultz 2007).
- **Behavioral validation:** Conditioning paradigm: DA burst initially on reward → transfers to CS; reward omission produces dip — see project's reward_ema_pre / surprise-LR-boost machinery.

### O.03 DA modulation of corticostriatal plasticity — three-factor rule
*[from Part V — Movement (Ch 30-39); renumbered from O.51]*

- **System:** glutamatergic cortex→MSN synapse + DA terminal at same dendritic spine.
- **Biological role:** PF/PK pre-post coincidence + DA presence determines LTP vs LTD direction. Three-factor learning rule (Hebb × DA). Different sign at D1 vs D2 MSNs.
- **Sim status:** **implemented** — eligibility-trace × `current_reward_signal` × STDP machinery in `sim/bridge.py`. `cortex→D1` plastic with `stdp_w_max=30`. Aligns with textbook three-factor rule.
- **Cluster:** O primary; J secondary.
- **Prerequisites:** O.50, J.*.
- **Citation:** Kandel 6e Ch 38 p 947–950 (Surmeier 2009).
- **Behavioral validation:** Reward-paired action → cortex→D1 LTP; punishment-paired → LTD or D2 LTP.

### O.04 Goal-directed → habitual transition — DA + DLS plasticity
*[from Part V — Movement (Ch 30-39); renumbered from O.52]*

- **System:** dopamine-driven plasticity migrating from dorsomedial (early) to dorsolateral (late) striatum with overtraining.
- **Biological role:** Same DA RPE signal, applied to different striatal compartments at different stages of learning. Substrate for skill automatization and OCD-like rigidity.
- **Sim status:** **missing** — no separate DMS / DLS, see A.58.
- **Cluster:** O primary; A secondary.
- **Prerequisites:** A.58, O.50.
- **Citation:** Kandel 6e Ch 38 p 950–956.
- **Behavioral validation:** Habit / devaluation paradigm.

---

## Cluster P — Disease (Parkinson, Huntington, Tourette, OCD)

### O.05 Hypothalamic Homeostatic Architecture — afferent/effector/feedback ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.50]*

- **System:** Hypothalamus as 3-zone (preoptic/anterior, tuberal, posterior) coordination hub for survival behaviors.
- **Biological role:** Implements ~24 distinct sensor-controller-effector loops (temperature, osmolarity, blood pressure, glucose, fat stores, etc.). Each loop has: detector (e.g. SFO osmoreceptors), integrator (e.g. PVN neurons), effector (autonomic output, hormone release, behavior). Multiple loops yield emergent settling-points rather than hard setpoints.
- **Sim status:** missing — no homeostatic-drive variables. Reward is goal-state-defined externally rather than defended around a setpoint.
- **Cluster:** O primary, C secondary (modulation).
- **Prerequisites:** Q.x (could fold "homeostasis cluster" here).
- **Citation:** Kandel 6e Ch 41 pp 1011-1013 (Table 41-1).
- **Behavioral validation:** Lesion studies: any single-zone lesion produces specific homeostatic dysregulation (e.g. preoptic → thermoregulation; PVN → autonomic; arcuate POMC → obesity).

### O.06 Arcuate POMC / AgRP / MC4R Feeding Loop — hunger/satiety ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.51]*

- **System:** Arcuate nucleus → PVN → parabrachial nucleus.
- **Biological role:** **Two antagonistic populations.** AgRP neurons (hunger-promoting, GABA + NPY + AgRP) inhibit POMC and PVN-MC4R. POMC neurons (satiety, release α-MSH which agonizes MC4R). PVN-MC4R "satiety neurons" project to lateral parabrachial. Leptin (long-term fat signal) excites POMC, inhibits AgRP. Ghrelin (pre-meal) excites AgRP. CCK (postprandial vagal) → satiety. AgRP-stim in sated mouse → ravenous eating; AgRP-ablation → starvation.
- **Sim status:** missing — no feeding/satiety circuit. **Top-3 actionable:** if hunger were modeled as a slow-changing state variable that *modulates the reward value* of food-cue inputs (incentive motivation, see O.55), this would naturally produce richer goal-switching behavior than fixed external reward.
- **Cluster:** O primary, C secondary (homeostatic NM).
- **Prerequisites:** O.50.
- **Citation:** Kandel 6e Ch 41 pp 1031-1037 (Fig 41-14).
- **Behavioral validation:** Optogenetic AgRP stim → immediate feeding; chemogenetic POMC stim → satiety; MC4R-knockout → severe obesity.

### O.07 Leptin / Insulin Long-Term Energy Signals — adipose-feedback hormones
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.52]*

- **System:** Adipose-derived leptin + pancreatic insulin → arcuate POMC/AgRP receptors + NTS receptors.
- **Biological role:** Long-timescale "fat-store report" that biases the AgRP/POMC tug-of-war. Leptin asymmetry: defends against starvation strongly, weakly resists obesity (leptin saturates above set-point — explains why obese humans aren't leptin-corrected by their own elevated leptin).
- **Sim status:** missing.
- **Cluster:** O primary.
- **Prerequisites:** O.51.
- **Citation:** Kandel 6e Ch 41 pp 1033-1037.
- **Behavioral validation:** ob/ob leptin-deficient mice → severe obesity, reversed by exogenous leptin. Diet-induced obese mice fail to reduce intake despite high endogenous leptin (leptin resistance).

### O.08 SFO/OVLT Thirst Circuit — osmolarity sensing
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.53]*

- **System:** Subfornical organ + organum vasculosum laminae terminalis (circumventricular organs lacking BBB) → median preoptic + PVN.
- **Biological role:** Direct osmoreceptor neurons (mechanically gated TRPV1-related channels respond to cell-volume changes). Hyperosmolality → activate SFO → thirst behavior + vasopressin release. Anticipatory feedforward inhibition of vasopressin/thirst when drinking begins, *before* osmolarity changes.
- **Sim status:** missing.
- **Cluster:** O primary, K secondary (transduction).
- **Prerequisites:** O.50.
- **Citation:** Kandel 6e Ch 41 pp 1027-1031.
- **Behavioral validation:** SFO optogenetic stim → sated mice begin drinking within seconds; lesion → adipsia.

### O.09 HPA Axis (CRH→ACTH→cortisol) — stress hormone cascade
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.54]*

- **System:** PVN parvocellular CRH neurons → anterior pituitary corticotrophs (ACTH) → adrenal cortex (cortisol).
- **Biological role:** Cardinal stress axis. Cortisol negative-feedback on PVN, hippocampus, anterior pituitary. Chronic activation produces hippocampal atrophy and cognitive impairment. Circadian-modulated (peak at waking, trough at sleep onset).
- **Sim status:** missing.
- **Cluster:** O primary, C secondary, P secondary (chronic stress disease).
- **Prerequisites:** O.50.
- **Citation:** Kandel 6e Ch 41 pp 1027-1029 (anterior pituitary endocrine control).
- **Behavioral validation:** Dexamethasone suppression test; cortisol elevation to acute stressor; cortisol-circadian-blunting in major depression.

### O.10 Incentive Motivation Theory — deficiency adjusts reward value
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.55]*

- **System:** AgRP neurons modulate reward value of food/cues; analogous logic for thirst, thermoregulation.
- **Biological role:** Berridge/Toates: deprivation does not generate behavior directly — it *amplifies the reward value* (incentive salience) of the relevant goal stimuli. Sated mouse with AgRP optogenetic stim → behaves as fasted (high food reward). Cues predicting food acquire reward value through learned association.
- **Sim status:** missing — flagship reward is fixed external scalar; agent state doesn't modulate per-stimulus reward weights.
- **Cluster:** O primary, C secondary.
- **Prerequisites:** O.51, C.58.
- **Citation:** Kandel 6e Ch 41 pp 1037-1039 (Fig 41-15A).
- **Behavioral validation:** Place-preference paradigm: AgRP-stim in sated mouse + paired location → conditioned preference; demonstrates reward-value amplification.

### O.11 Drive Reduction Theory — deficiency is aversive ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.56]*

- **System:** AgRP neuron activation is itself aversive; eating reduces aversion → "drive reduction" reward.
- **Biological role:** Sternson group: optogenetic AgRP stim in sated mice produces conditioned place *aversion*. In food-deprived mice (high endogenous AgRP), animals show place preference for context that previously *suppressed* AgRP activity — i.e., they "want to escape hunger." Provides distinct theoretical account from incentive motivation: hunger is the negative reinforcer, and food relieves it.
- **Sim status:** missing.
- **Cluster:** O primary.
- **Prerequisites:** O.51, O.55.
- **Citation:** Kandel 6e Ch 41 pp 1038-1039 (Fig 41-15B).
- **Behavioral validation:** Two-chamber CPP test with optogenetic AgRP control demonstrates both the aversion (sated) and relief (fasted) signs.

### O.12 Amygdala Fear-Learning Circuit (LA/BLA/CeA) ⭐
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.57]*

- **System:** Lateral amygdala (LA) receives sensory CS+US convergence from thalamus + cortex; basal/basolateral (BLA) for context; central amygdala (CeA) drives output → PAG (freezing), hypothalamus (autonomic), parabrachial (respiratory). Plus internal LA→BLA→CeA + intercalated cell circuitry.
- **Biological role:** **Fastest fear-conditioning substrate.** Direct thalamic shortcut to LA allows fear response before cortex completes processing ("low road"). LA pyramidal neurons show classical Hebbian LTP at CS-US convergence — site of fear-memory storage. Damage abolishes Pavlovian fear conditioning to learned cues. Patient S.M. with bilateral amygdala lesion: no fear of external threats.
- **Sim status:** missing — flagship has no fear/aversive-valence-dedicated structure. Reward-side-only architecture.
- **Cluster:** O primary, J secondary (Hebbian LTP at convergence).
- **Prerequisites:** D.x (hippo for context), J.x (LTP).
- **Citation:** Kandel 6e Ch 42 pp 1083-1099 (Fig 42-5).
- **Behavioral validation:** Pavlovian tone-shock conditioning → freezing response abolished by LA lesion. Cued (LA-dependent) vs contextual (BLA + hippo-dependent) fear dissociates.

### O.13 Amygdala Reward / Appetitive Role — amygdala beyond fear
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.58]*

- **System:** LA/BLA also encode appetitive stimulus-reward associations.
- **Biological role:** Distinct circuitry within amygdala mediates appetitive Pavlovian conditioning. Single-neuron recording: amygdala neurons rapidly modulate firing as visual images become reward-predictive. Lesion impairs both punishment and reward learning. Activation of appetitive amygdala ensembles can elicit positive emotional behavior. The amygdala is therefore better described as a **valence-and-arousal map** than a "fear center."
- **Sim status:** missing — but adding this with O.57 as a single "valence module" would be principled.
- **Cluster:** O primary, C secondary (DA interaction).
- **Prerequisites:** O.57.
- **Citation:** Kandel 6e Ch 42 pp 1097-1099.
- **Behavioral validation:** Reactivation of appetitive-tagged amygdala neurons elicits approach behavior even in absence of CS.

### O.14 Interoceptive Fear (CO₂ Panic) — independent of amygdala
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.59]*

- **System:** Carotid body chemoreceptors + brainstem CO₂-sensing → presumably insula and brainstem alarm pathways (NOT amygdala).
- **Biological role:** Patient S.M. (bilateral amygdala lesion) shows **no** fear of external threats, but **intense panic** when made to breathe CO₂. Demonstrates the amygdala is not the only fear substrate; interoceptive panic is mediated by an amygdala-independent pathway. Important for any "emotion module" architecture: fear has multiple substrates.
- **Sim status:** missing.
- **Cluster:** O primary.
- **Prerequisites:** O.57.
- **Citation:** Kandel 6e Ch 42 pp 1095-1097 (Fig 42-6).
- **Behavioral validation:** CO₂ inhalation in amygdala-lesion patients → panic equal-to-or-greater than controls.

### O.15 vmPFC ↔ Amygdala Top-Down Regulation — fear extinction
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.60]*

- **System:** Ventromedial PFC + infralimbic cortex → BLA + intercalated cells.
- **Biological role:** vmPFC drives fear *extinction* by inhibiting CeA output. Cognitive-behavioral therapy mechanism. PTSD = failure of vmPFC inhibition; conditioned fear "reappears" without prefrontal override. Bidirectional: amygdala → PFC affects how PFC-mediated decisions weight emotional information (loss-aversion framing in fMRI).
- **Sim status:** missing — but project does have a PFC region (G.x cluster, 60-neuron recurrent). A PFC ↔ amygdala interaction would map naturally if O.57 is added.
- **Cluster:** O primary, G secondary (PFC).
- **Prerequisites:** O.57, G.x (PFC).
- **Citation:** Kandel 6e Ch 42 pp 1099-1102.
- **Behavioral validation:** Fear-extinction protocols: vmPFC inactivation → no extinction; vmPFC stimulation → accelerated extinction.

### O.16 NAc Reward Hub Architecture — VTA + cortical + hippo + amyg convergence
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.61]*

- **System:** Nucleus accumbens (ventral striatum) MSNs receive VTA DA + glutamate from mPFC, hippocampus, BLA, lateral habenula, thalamus.
- **Biological role:** Central reward integrator. DA (VTA) provides teaching signal; mPFC provides goal/context; hippocampus provides spatial/episodic context; amygdala provides emotional valence; lateral habenula provides aversion / negative-RPE. NAc shell vs core distinction: shell more associated with hedonic hotspots and motivational learning, core with goal-directed action selection.
- **Sim status:** partial — flagship BG cascade includes per-action striatal D1/D2 pools, VTA-like DA pool, plus PFC and hippocampus regions. NAc is not separately distinguished from dorsal striatum, and the hedonic-vs-motivational shell-core split is absent.
- **Cluster:** O primary, A secondary, C secondary, D secondary, G secondary.
- **Prerequisites:** A.x (BG), C.52 (VTA), D.x (hippo), G.x (PFC).
- **Citation:** Kandel 6e Ch 43 pp 1067-1068 (Fig 43-3).
- **Behavioral validation:** NAc-shell vs core lesions dissociate Pavlovian-instrumental transfer (shell) from action-outcome learning (core).

### O.17 Brain Stimulation Reward (Olds-Milner) — self-stimulation as reward proxy
*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from O.62]*

- **System:** Medial forebrain bundle electrode → activates VTA DA + adjacent reward fibers.
- **Biological role:** Animals will press lever for hours, cross electrified grids, forego food/water for self-stimulation. Demonstrates the reward circuitry exists as a neural substrate — and that activating it is rewarding independently of any sensory stimulus. Cocaine/nicotine *reduce* the stimulation threshold (= "augment reward").
- **Sim status:** implementable as test paradigm — could validate the BG cascade by adding a "phantom reward channel" that delivers reward when agent emits a designated lever-press action; agent should learn to press repeatedly. Would be a useful regression test that reward circuit functions as expected.
- **Cluster:** O primary, A secondary.
- **Prerequisites:** A.x (BG), C.58 (RPE).
- **Citation:** Kandel 6e Ch 43 pp 1066-1067 (Fig 43-1).
- **Behavioral validation:** Lever-press rate increases monotonically with stimulation frequency, drug administration shifts rate-frequency curve leftward.

---

## Summary

**Entry count:** 35 entries across 3 clusters.
- **Cluster C (Neuromodulation):** 14 entries (C.50-C.63)
- **Cluster N (Sleep & Arousal):** 13 entries (N.50-N.62)
- **Cluster O (Emotion / Reward / Motivation):** 13 entries (O.50-O.62)

**3 most actionable additions (★ marks above):**

1. **Hippocampal SWR-based replay (N.56):** The project's existing replay infrastructure can be upgraded to generate compressed sequences from recent place-cell trajectories, phase-locked to a NREM stage variable. This directly addresses the documented "replay content quality is the bottleneck" issue. Minimal new GPU code — extends existing replay scheduler with sequence-extraction logic. Also see N.55 spindles + N.54 slow oscillation for nesting events in biologically realistic time windows.

2. **Hypothalamic homeostatic drives (O.50, O.51, O.55, O.56):** Add hunger / thirst as slow-changing internal state variables that *modulate per-stimulus reward weights* (incentive motivation, O.55) AND act as aversive negative reinforcers (drive reduction, O.56). This produces dynamic goal-switching that fixed external rewards cannot. The neuromodulator subsystem already supports the right abstraction — declare an `AgRP-like` modulator with `excitability_drive` target on a "hunger" group + `synaptic_gain` on food-cue pathways. Closes the "agent has no internal homeostatic state" gap and would naturally test the project's sensed-reward architecture against a non-spatial reward-modulation regime.

3. **Amygdala valence module (O.57 + O.58):** A single LA/BLA/CeA module that pairs *fear/threat* AND *appetitive valence* (per O.58, the amygdala does both). Provides emotional valence beyond pure RPE — particularly important for any aversive learning the project might add. Two-population architecture (CS-US convergence + output) would be small (~50-100 neurons) and composes cleanly with the PFC region for vmPFC-amygdala extinction (O.60). Closes the major gap that the flagship has no aversive-side architecture; right now negative reward just produces negative DA, but biology routes threats through a separate substrate that interacts with reward differently.

**Notable textbook ↔ project discrepancies:**
- C.52: simulator collapses A9/A10 distinction into a single DA pool.
- C.56: project's reward signal is event-triggered scalar, not pacemaker tonic + phasic burst.
- C.58: project's RPE-flavored mechanisms (surprise-LR-boost, adaptive-DA) implement positive RPE amplification but NOT the canonical cue-shift dynamic where the DA burst transfers from reward to predictive cue across learning.
- O.50/O.51: complete absence of hypothalamic homeostatic drives; reward is exogenously defined rather than defended around setpoints.

### O.18 Reward-modulated stimulus–outcome learning — striatum/HC interaction
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from O.50]*

- **System:** Ventral striatum (NAcc) for incremental S-R, hippocampus for configural/contextual variants; coupled by reciprocal HC↔striatum projections.
- **Biological role:** Trial-by-trial probabilistic learning of cue→outcome (e.g., weather-prediction task). Striatum dominates with simple cue-reward; HC engages when configural representation needed (Duncan et al. 2018).
- **Sim status:** partial — DA-modulated cortex→D1 STDP in `g11_bg_runner` does the striatum side; HC↔striatum cooperation/competition not explicitly modeled though hippocampus region exists.
- **Cluster:** O primary, A secondary, D secondary.
- **Prerequisites:** A.* (BG), D.51, C.* (DA).
- **Citation:** Kandel 6e Ch 52 pp 1303–1305.
- **Behavioral validation:** Configural-vs-elemental strategy switch correlates with HC–NAcc functional coupling (fMRI).

### O.19 Value-based decisions — vmPFC / OFC encode subjective value
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from O.51]*

- **System:** Ventromedial PFC + orbitofrontal cortex represent expected subjective value; project to striatum and LIP/PFC accumulators. Value modulates drift rate of accumulator.
- **Biological role:** Decisions about preferences (which menu item, which apartment) reduce to evidence accumulation where each option's evidence is its subjective-value samples. Same drift-diffusion math as perceptual decisions.
- **Sim status:** partial — DA-modulated cortex→D1 implements value learning; no separate vmPFC/OFC region encodes scalar value across actions independently of the action selector.
- **Cluster:** O primary, G primary, A secondary.
- **Prerequisites:** explicit value-coding region declaration.
- **Citation:** Kandel 6e Ch 56 pp 1406–1409.
- **Behavioral validation:** vmPFC fMRI BOLD ∝ subjective value; lesions cause reversal-learning deficits and intransitive preferences.

## Cluster P — additions

### P.01 Parkinson disease — SNc DA neuron loss → indirect-pathway dominance
*[from Part V — Movement (Ch 30-39); renumbered from P.50]*

- **System:** progressive degeneration of nigrostriatal DA neurons; α-synuclein Lewy pathology.
- **Biological role:** Loss of DA → D1 underdriven (less Go), D2 disinhibited (more NoGo) → bradykinesia, rigidity, resting tremor. Levodopa replenishes DA precursor; DBS of STN or GPi reduces overactive output. Freezing of gait targets PPN.
- **Sim status:** **missing** — could be tested by ablating `current_reward_signal` or zeroing nigrostriatal weights; matches textbook prediction (action initiation deficit). **A natural smoke-test for the BG cascade.**
- **Cluster:** P primary; A, O secondary.
- **Prerequisites:** A.50–A.52, O.50.
- **Citation:** Kandel 6e Ch 38 p 952–956.
- **Behavioral validation:** Drop simulated DA → action initiation rate ↓, perseveration ↑; restore DA → recovery.

### P.02 Huntington disease — D2 MSN loss → direct-pathway dominance
*[from Part V — Movement (Ch 30-39); renumbered from P.51]*

- **System:** CAG-repeat expansion in HTT → preferential degeneration of D2 indirect-pathway MSNs early; later D1 MSNs and cortex.
- **Biological role:** Loss of D2 MSNs → indirect "NoGo" weakened → uncontrolled action release (chorea, motor disinhibition). Late stages: bradykinesia + dementia as both pathways and cortex degenerate.
- **Sim status:** **missing** — could be tested by ablating `str_d2_X` pools.
- **Cluster:** P primary; A secondary.
- **Prerequisites:** A.51.
- **Citation:** Kandel 6e Ch 38 p 953–956.
- **Behavioral validation:** Inactivate D2 MSN pools → involuntary action emission, reduced suppression of competitors.

### P.03 OCD / Tourette — pathologically dominant or intrusive option
*[from Part V — Movement (Ch 30-39); renumbered from P.52]*

- **System:** abnormal cortico-BG-thalamo-cortical loop activity; OCD = dominant compulsive option that wins selection repeatedly; Tourette = intrusion of nonselected motor option (tic).
- **Biological role:** Failure of selection mechanism. OCD: orbitofrontal-caudate hyperactivity. Tourette: striatal "matrisome" disinhibition. DBS of GPi or anterior limb of internal capsule effective.
- **Sim status:** **missing** — interesting future test: persistently high gain on one action's D1 pool should reproduce OCD-like perseveration.
- **Cluster:** P primary; A secondary.
- **Prerequisites:** A.50–A.55.
- **Citation:** Kandel 6e Ch 38 p 954–956.
- **Behavioral validation:** Bias one action's MSN excitability → behavior locks; mirrors compulsion.

### P.04 Schizophrenia — failure to suppress nonselected options
*[from Part V — Movement (Ch 30-39); renumbered from P.53]*

- **System:** hyperdopaminergic mesolimbic + hypodopaminergic mesocortical; striatal DA dysregulation impairs filtering.
- **Biological role:** Salience-attribution to irrelevant stimuli (positive symptoms = hallucinations, delusions) interpreted as "selection of inappropriate options"; antipsychotics block D2 to restore filtering.
- **Sim status:** **missing**.
- **Cluster:** P primary; O secondary.
- **Prerequisites:** A.50–A.55, O.50.
- **Citation:** Kandel 6e Ch 38 p 954–956.
- **Behavioral validation:** Excess DA noise on STR → spurious action selection at random.

---

## Cluster G — Prefrontal / association cortex

### P.05 Hippocampal disorders of autobiographical memory
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from P.50]*

- **System:** AD (Aβ → synaptic dysfunction → HC atrophy); schizophrenia (HC-PFC desynchrony, place-field over-rigidity, CA2 PV-interneuron loss); reduced Aβ rescues plasticity in mouse models.
- **Biological role:** HC dysfunction underlies memory loss in AD and the working-memory + social-memory deficits in schizophrenia. Place cells are over-rigid in schizophrenia mouse models (failure to remap).
- **Sim status:** missing — no disease module.
- **Cluster:** P primary, D secondary.
- **Prerequisites:** P module.
- **Citation:** Kandel 6e Ch 54 p 1367.
- **Behavioral validation:** Aβ-rescue restores LTP in mouse model; HC-PFC coherence reduced in schizophrenia mice.

---

## Ch 55 — Language

### P.06 Agnosias as failures of interrogation, not sensation
*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from P.51]*

- **System:** Hemineglect (right parietal), prosopagnosia (fusiform face area), achromatopsia (V4/V8), Wernicke aphasia (auditory-semantic).
- **Biological role:** Patient does not perceive a *lack* of information — the apparatus that *asks the question* is broken. Distinguished from primary sensory loss (hemianopsia) where the patient knows what's missing and compensates.
- **Sim status:** missing — diagnostic only.
- **Cluster:** P primary, G secondary.
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 56 pp 1410–1411, Fig 56-10.
- **Behavioral validation:** Clock-drawing test (left half omitted); denial of paretic limb ownership.

---

## Coverage summary

- **Total entries:** 38 (D ×17, G ×13, J ×7, O ×3, N ×2, L ×1, P ×2, with overlapping secondary clusters).
- **Cluster D additions (most actionable):** trisynaptic + direct pathways (D.52–D.53), CA3 recurrents (D.54), DG pattern separation (D.61), CA3 pattern completion (D.62), engram tagging (D.63), grid/HD/border/speed/time cells (D.56–D.60), theta (D.67), SWR replay (D.68), remapping (D.66).
- **Cluster G additions:** working memory (G.50), drift-diffusion accumulator (G.58), LIP decision variable (G.59), affordance framework (G.61).
- **Cluster N additions:** SWR-driven hippocampal-cortical dialogue / systems consolidation (N.50, plus D.68).
- **Cluster J additions:** mossy-fiber presynaptic LTP (J.53), late-LTP / CREB / PKMζ (J.55), perforant-path L-type-Ca²⁺ LTP (J.54).

## Top 3 most actionable additions for the simulator

### 1. Build the trisynaptic loop as 3 declared `BrainRegion`s (D.52, D.61, D.62, D.54)
Replace the single hippocampus pool with **DG → CA3 → CA1** declared regions in `g11_bg_runner` (or a sibling runner). Concrete:
- **DG**: large population (e.g. 4× CA3 size), strong feedforward inhibition target rate ~2–5%, plastic perforant-path input. This *is* the pattern-separator (Marr expansion recoding); divergence + sparseness comes free from the existing connectivity primitives plus inhibitory drive.
- **CA3**: smaller population with `internal_density` recurrent collaterals (autoassociator); plastic. This *is* the pattern-completer.
- **CA1**: output region, plastic Schaffer input.
- Use existing `RegionPathway.plasticity_gate` to phase encoding vs retrieval.
- **Acceptance metrics:** (a) DG output orthogonality between similar EC inputs > raw EC orthogonality (pattern separation); (b) CA3 recovers full stored pattern from 50% partial cue (pattern completion); (c) CA1 places fields stable across sessions in same env, decorrelated across envs (D.66).
- This is the single biggest "biology-grounded" upgrade available with existing infrastructure — no new GPU code required, just topology and inhibition tuning.

### 2. Implement SWR-driven replay with sequential trajectory content (D.68, N.50)
The named bottleneck is *replay content*. Concrete plan:
- During waking (online phase), record the temporal sequence of CA1 active-cell ids in a circular buffer per trajectory.
- During NREM phase, generate a "ripple event" by injecting a brief (~50–100 ms) high-frequency excitation pulse into CA3 via `StimulusManager`; gate it to play back recorded sequences at 10–20× compression by sequentially driving the recorded cell ids with short current pulses.
- Couple this to a cortical "slow-oscillation" surrogate: alternate up/down state via excitability-drive NM target with ~0.5–1 Hz sinusoid; ripples should fire during up-states (already a known SWR–slow-oscillation coupling biomarker).
- Existing plasticity (STDP) running during replay automatically performs the consolidation transfer. Validate by training a reward-related trajectory in HC, sleeping, then probing whether cortex-only readout has acquired the path.
- This addresses both the SWR mechanism (D.68) and systems consolidation (N.50) in one infrastructure addition.

### 3. Engram-tagging primitive — record + reactivate ensembles (D.63)
A small bridge addition with high research leverage:
- API: `bridge.tag_active_ensemble(name, threshold_hz=20, window_ms=500)` — records the set of neuron ids whose firing rate exceeded `threshold_hz` over the window into a named tag.
- API: `bridge.stimulate_tag(name, current_pA, duration_ms)` — drives that ensemble via `StimulusManager`, emulating optogenetic ChR2 activation.
- Optional inverse `silence_tag` via inhibitory drive.
- This unlocks **causal-recall experiments**: tag the engram during phase-1 navigation, then in phase-2 a different environment, drive the tag and check whether the agent emits phase-1-appropriate behavior (analog of Tonegawa freezing-on-tag-stimulation).
- Also provides the substrate for pattern-completion validation in D.62 (drive partial subset of a tagged ensemble and check whether CA3 recurrents reactivate the rest).
- Cheap to build (~50 LOC bridge + thin runner glue), opens a whole class of episodic-memory experiments aligned with the "biology-grounded over engineered" project ethos.

### P.07 Myasthenia Gravis — autoimmune attack on muscle nicotinic AChR
*[from Part IX — Diseases (Ch 57-64); renumbered from P.50]*

- **System:** Neuromuscular junction (skeletal muscle)
- **Biological role / Pathology:** Anti-nAChR antibodies cross-link and internalize postsynaptic AChR, flatten junctional folds, and reduce mEPP amplitude until each EPP fails to reach threshold. Decremental response on repetitive stimulation; weakness worsens with use, recovers with rest; reversed by anti-AChE (neostigmine).
- **Sim status:** missing-needs-new-infrastructure — simulator has no NMJ / muscle layer; closest analog is reducing AMPA conductance per synapse on a designated motor pool.
- **Cluster:** P, F (motor unit), L (postsynaptic receptor density)
- **Prerequisites:** F.NMJ (motor-unit muscle coupling), L.receptor-density-as-state (per-synapse AChR count separable from `weight`)
- **Citation:** Kandel 6e Ch 57 p 1437–1444
- **Behavioral validation:** Repetitive 3 Hz stimulation of motor pool produces decremental EPSP train (1st > 5th); restored when receptor-density variable is increased mid-trial (AChE-inhibitor analog).

### P.08 Lambert-Eaton Myasthenic Syndrome — presynaptic VGCC autoimmunity
*[from Part IX — Diseases (Ch 57-64); renumbered from P.51]*

- **System:** Presynaptic motor terminal
- **Biological role / Pathology:** Antibodies against P/Q-type Cav2.1 reduce Ca²⁺ entry → reduced quantal content. Opposite-direction modulation from MG: weakness *improves* with sustained activity (residual Ca²⁺ accumulates).
- **Sim status:** missing-needs-new-infrastructure — no presynaptic Ca²⁺ pool; STP `U` parameter is the closest analog (lower release probability).
- **Cluster:** P, K (presynaptic release), I (channels)
- **Prerequisites:** K.presynaptic-Ca-pool, then VGCC-density variable per pathway
- **Citation:** Kandel 6e Ch 57 p 1442
- **Behavioral validation:** Lower `stp_U` on motor synapse → reduced first-pulse EPSP, but residual facilitation grows with high-frequency train (opposite of MG).

### P.09 Botulinum / Tetanus Toxin — SNARE cleavage at motor terminal
*[from Part IX — Diseases (Ch 57-64); renumbered from P.52]*

- **System:** Presynaptic vesicle release
- **Biological role / Pathology:** Botulinum cleaves SNAP-25 / VAMP / syntaxin → flaccid paralysis (no ACh release). Tetanus retrograde-transports to spinal interneurons and blocks GABA/glycine release → spastic paralysis.
- **Sim status:** missing-needs-new-infrastructure — no SNARE machinery; functionally model as setting per-pathway release probability to 0.
- **Cluster:** P, K (vesicle release), B (inhibition for tetanus)
- **Prerequisites:** per-pathway "release_enabled" flag (already trivially possible via weight=0 + plasticity gate)
- **Citation:** Kandel 6e Ch 57 p 1442–1444
- **Behavioral validation:** Setting motor-pathway weight=0 produces silent motor; selectively zeroing inhibitory pathways onto motoneurons produces tonic firing analog of tetanus.

### P.10 Amyotrophic Lateral Sclerosis (ALS) — progressive motor-neuron death
*[from Part IX — Diseases (Ch 57-64); renumbered from P.53]*

- **System:** Upper + lower motor neurons
- **Biological role / Pathology:** SOD1, TDP-43, FUS, C9ORF72 mutations cause selective motor-neuron degeneration via excitotoxicity, protein aggregation, axonal-transport failure. Surviving units sprout to reinnervate orphaned fibers (giant motor units, fasciculations) until reserve exhausts.
- **Sim status:** missing-but-modelable (partial) — can implement as time-progressive ablation of motor-pool neurons + structural sprouting toward orphaned downstream targets.
- **Cluster:** P, F (motor pool), G (excitotoxicity), structural plasticity
- **Prerequisites:** time-progressive neuron-ablation API; structural-plasticity already exists (`struct_plast_activity_bias`)
- **Citation:** Kandel 6e Ch 57 p 1445–1452 (Table 57-2)
- **Behavioral validation:** Gradually delete motor neurons in `motor_X` pools → motor output declines; surviving units' fan-in grows via structural plasticity until critical fraction lost → output collapses.

### P.11 Charcot-Marie-Tooth & Demyelinating Neuropathies — myelin loss
*[from Part IX — Diseases (Ch 57-64); renumbered from P.54]*

- **System:** Peripheral nerve axons
- **Biological role / Pathology:** PMP22 dup/del, MPZ, Cx32 mutations cause demyelination → conduction slowing + block. Saltatory conduction fails; spatial buffering of K⁺ disrupted.
- **Sim status:** missing-needs-new-infrastructure — point-neurons lack axons, myelin, conduction delay distinct from synaptic delay.
- **Cluster:** P, glia (Q), I (channels redistributed)
- **Prerequisites:** axon model with conduction-delay state, glial Schwann/oligodendrocyte layer
- **Citation:** Kandel 6e Ch 57 p 1452–1458 (Table 57-3)
- **Behavioral validation:** Increase per-pathway conduction delay → temporal precision of spike-coincidence falls → STDP windows mistime → learning collapses.

### P.12 Spinal Muscular Atrophy (SMN1 deletion) — α-motoneuron death
*[from Part IX — Diseases (Ch 57-64); renumbered from P.55]*

- **System:** Lower motor neurons (esp. spinal)
- **Biological role / Pathology:** Loss-of-function in *SMN1* (compensated partly by *SMN2*) impairs snRNP biogenesis; selective motor-neuron loss in infancy.
- **Sim status:** missing-needs-new-infrastructure — analogous to ALS but developmental; no mRNA-splicing layer.
- **Cluster:** P, M (development), F (motor)
- **Prerequisites:** developmental neuron-survival module
- **Citation:** Kandel 6e Ch 57 p 1450
- **Behavioral validation:** Suppress spawn of motor-pool neurons during construction → reduced motor output from birth.

### P.13 Focal-Onset Seizures — local hyperexcitable focus
*[from Part IX — Diseases (Ch 57-64); renumbered from P.56]*

- **System:** Cortex (often hippocampus / temporal lobe)
- **Biological role / Pathology:** "Paroxysmal depolarizing shift" (PDS) — synchronous large depolarization driven by NMDA + recurrent AMPA, followed by long AHP. Surround inhibition normally contains the focus; when GABA fails or recurrent excitation dominates, activity spreads.
- **Sim status:** **missing-but-modelable (high)** — simulator has E/I balance, NMDA, recurrent connectivity, AHP currents. PDS reproducible by raising recurrent excitatory weight or weakening local inhibition in one region.
- **Cluster:** P, J (E/I balance), I (channels), B (interneurons)
- **Prerequisites:** none — current architecture sufficient.
- **Citation:** Kandel 6e Ch 58 p 1456–1465
- **Behavioral validation:** In a single brain region, scale `inh_weight_mean` → 0.3× and `exc_weight_mean` → 1.5×; observe synchronized population bursts (PDS-like) and propagation to neighboring regions through pathways.

### P.14 Generalized Seizures (Absence, Tonic-Clonic) — thalamocortical 3 Hz oscillation
*[from Part IX — Diseases (Ch 57-64); renumbered from P.57]*

- **System:** Thalamocortical loop
- **Biological role / Pathology:** Absence epilepsy involves T-type Ca²⁺ channels in TRN/TC neurons producing rhythmic 3 Hz spike-wave discharges; pathological hypersynchrony of cortex via TC relay. Ethosuximide blocks T-type.
- **Sim status:** partial — simulator has TC and TRN HH presets (`HH_THALAMIC_RELAY_TBURST`, `HH_TRN_BURST_INHIB`) including T-type-like burst behavior. Modelable by tuning T-current strength + TC/TRN connectivity to provoke 3 Hz rhythm.
- **Cluster:** P, I (T-type Ca²⁺), B (TRN inhibition), recurrent loop dynamics
- **Prerequisites:** thalamic relay + TRN regions wired in cortico-thalamic loop (extension of brain-region framework)
- **Citation:** Kandel 6e Ch 58 p 1465–1472
- **Behavioral validation:** Build TC↔TRN↔cortex loop, observe 3 Hz spike-wave discharge in EEG-analog (population rate FFT) when T-current gain is increased.

### P.15 Channelopathies — SCN1A/Dravet, KCNQ2, GABRG2 — monogenic epilepsies
*[from Part IX — Diseases (Ch 57-64); renumbered from P.58]*

- **System:** Cortical/hippocampal interneurons (Dravet) or pyramidal neurons
- **Biological role / Pathology:** SCN1A loss-of-function selectively impairs PV+ FS interneurons (which depend on Nav1.1) → disinhibition. KCNQ2 (M-current) loss → reduced AHP → hyperexcitability. GABRG2 mutations → altered GABA-A kinetics.
- **Sim status:** missing-but-modelable — Nav1.1 / M-current selectively can be lowered in interneuron subset via per-region HH parameter overrides.
- **Cluster:** P, I (channels), B (interneuron subtypes)
- **Prerequisites:** per-cell-type Nav / M-current scaling (extension of HH preset system)
- **Citation:** Kandel 6e Ch 58 p 1462–1465
- **Behavioral validation:** Set Nav peak conductance to 50% in `cortical_FS_interneuron` pool only → FS firing drops → pyramidal disinhibition → spontaneous bursts.

### P.16 Kindling — activity-induced epileptogenesis
*[from Part IX — Diseases (Ch 57-64); renumbered from P.59]*

- **System:** Limbic / hippocampal circuits
- **Biological role / Pathology:** Repeated subthreshold stimulation progressively lowers seizure threshold via LTP-like + structural changes (mossy fiber sprouting, NMDA upregulation). Permanent network change.
- **Sim status:** missing-but-modelable — combine LTP (already present) with structural plasticity (already present, activity-biased) and homeostatic threshold regulation. Kindling = LTP + sprouting at same site over many trials.
- **Cluster:** P, J (LTP), structural plasticity, D (hippocampus)
- **Prerequisites:** none — composes existing primitives.
- **Citation:** Kandel 6e Ch 58 p 1474–1480
- **Behavioral validation:** Repeated focal Poisson stimulus to a hippocampal region; track whether spontaneous synchronous bursts emerge in unstimulated test windows after N kindling sessions.

### P.17 Status Epilepticus & Excitotoxicity — sustained seizure → cell death
*[from Part IX — Diseases (Ch 57-64); renumbered from P.60]*

- **System:** Hippocampus, cortex
- **Biological role / Pathology:** >30 min continuous seizure overactivates NMDA → Ca²⁺ overload → mitochondrial failure → neurodegeneration. CA1 and CA3 pyramidals especially vulnerable.
- **Sim status:** missing-needs-new-infrastructure — no Ca²⁺ overload / cell-death state; survival is binary (allocated or not).
- **Cluster:** P, J (NMDA), G (Ca²⁺ dynamics), Q (mitochondria)
- **Prerequisites:** intracellular Ca²⁺ pool with overload threshold → ablation
- **Citation:** Kandel 6e Ch 58 p 1480–1483
- **Behavioral validation:** With Ca²⁺ pool added: drive prolonged firing >30 sim-min in CA1; observe progressive ablation of high-firing-rate pyramidals.

### P.18 Conversion / Functional Neurological Disorder — psychogenic motor/sensory deficit
*[from Part IX — Diseases (Ch 57-64); renumbered from P.61]*

- **System:** Cortical "control" circuits without primary lesion
- **Biological role / Pathology:** Genuine motor/sensory deficits with no detectable structural pathology; fMRI shows altered prefrontal-motor/sensory coupling. Historically termed hysteria.
- **Sim status:** not-applicable — phenomenon is at level of explicit/implicit cognitive control we don't yet simulate.
- **Cluster:** P, top-down attention/control (currently absent)
- **Prerequisites:** explicit attention/gating subsystem above motor cascade
- **Citation:** Kandel 6e Ch 59 p 1474–1488
- **Behavioral validation:** N/A at current level of abstraction.

### P.19 Schizophrenia — Dopamine Hypothesis (excess striatal D2 signaling)
*[from Part IX — Diseases (Ch 57-64); renumbered from P.62]*

- **System:** Mesolimbic DA projection (VTA → ventral striatum)
- **Biological role / Pathology:** Hyperactive DA at D2 receptors in ventral striatum drives positive symptoms (delusions, hallucinations); all effective antipsychotics block D2. PET shows elevated striatal DA synthesis capacity in patients.
- **Sim status:** **implemented (directly modelable)** — neuromodulator subsystem supports tonic DA elevation; BG cascade has D1/D2 pathways. Simply raise DA `baseline` or add manual production rule.
- **Cluster:** P, C (dopamine), A (BG), action-selection
- **Prerequisites:** none — currently shipping infrastructure.
- **Citation:** Kandel 6e Ch 60 p 1497–1505
- **Behavioral validation:** In flagship g11 BG runner, set `neuromodulator.DA.baseline` 2× normal → expect indirect-pathway suppression (D2 inhibited) → motor disinhibition / spurious action selection (analog of "stimulus-driven false percepts").

### P.20 Schizophrenia — NMDA Hypofunction Hypothesis (glutamate)
*[from Part IX — Diseases (Ch 57-64); renumbered from P.63]*

- **System:** Cortical / hippocampal glutamatergic synapses
- **Biological role / Pathology:** NMDA antagonists (ketamine, PCP) recreate positive *and* negative symptoms; NMDA-R hypofunction on PV+ FS interneurons is hypothesized to produce cortical disinhibition + reduced gamma synchrony. Genetic risk variants converge on NMDA / synaptic genes.
- **Sim status:** **implemented (directly modelable)** — `fused_nmda_update_and_current()` exists; NMDA conductance can be scaled globally or per pathway/cell-type.
- **Cluster:** P, J (NMDA), B (PV+ interneurons), gamma oscillations
- **Prerequisites:** per-cell-type NMDA scaling (small extension)
- **Citation:** Kandel 6e Ch 60 p 1505–1515
- **Behavioral validation:** Reduce NMDA conductance to 30% on FS interneurons only → measure cortical gamma power (40–80 Hz band) via existing `run_benchmarks.py --benchmark gamma-oscillations` → expect reduced gamma + altered E/I (project benchmark already exists for gamma!).

### P.21 Schizophrenia — Synaptic Pruning Hypothesis (adolescence)
*[from Part IX — Diseases (Ch 57-64); renumbered from P.64]*

- **System:** Cortical synapse density (esp. PFC)
- **Biological role / Pathology:** Excessive adolescent synaptic pruning (C4 complement gene, microglia) leaves PFC under-connected → working-memory deficits. Patient post-mortem shows reduced spine density.
- **Sim status:** missing-but-modelable — structural plasticity exists; an "over-pruning" parameter on PFC pathways during a developmental window is a 1-line config.
- **Cluster:** P, M (development), structural plasticity, glia (microglia)
- **Prerequisites:** developmental schedule for structural-plasticity rates
- **Citation:** Kandel 6e Ch 60 p 1515–1525
- **Behavioral validation:** In runs with PFC, increase pruning rate during a "developmental" pre-eval window → observe reduced PFC connectivity → working-memory tasks (delay match) degrade.

### P.22 Major Depressive Disorder — monoamine + HPA + structural
*[from Part IX — Diseases (Ch 57-64); renumbered from P.65]*

- **System:** Limbic + monoamine systems (5-HT, NE, DA), HPA axis, hippocampus
- **Biological role / Pathology:** Reduced monoamine availability + chronic HPA-axis hyperactivity (high cortisol) + reduced hippocampal volume / reduced BDNF / impaired neurogenesis. SSRIs work over weeks (monoamine elevation alone insufficient — implies downstream plasticity).
- **Sim status:** missing-but-modelable (low fidelity) — neuromodulator subsystem can declare 5-HT/NE; reduced baseline + reduced reward-EMA tracking yields anhedonia analog. No HPA / cortisol / neurogenesis layer.
- **Cluster:** P, C (monoamines), D (hippocampus), reward, structural plasticity
- **Prerequisites:** declarative serotonin neuromodulator; eventually HPA + neurogenesis layers
- **Citation:** Kandel 6e Ch 61 p 1510–1520
- **Behavioral validation:** Lower `5HT.baseline` and `DA.baseline` in g11 → reward-tracking (sum-of-reward) degrades; restore via SSRI-analog (raise 5-HT) over many sim-minutes.

### P.23 Bipolar Disorder — mood-state instability
*[from Part IX — Diseases (Ch 57-64); renumbered from P.66]*

- **System:** Same as MDD but bistable
- **Biological role / Pathology:** Alternation between mania and depression; lithium effective (Wnt/GSK-3β, IP3 modulation). Strong heritability.
- **Sim status:** missing-needs-new-infrastructure — no slow bistable mood-state variable; would need a meta-modulator with hysteresis.
- **Cluster:** P, C (monoamine), slow neuromodulator dynamics
- **Prerequisites:** bistable / hysteretic neuromodulator dynamics
- **Citation:** Kandel 6e Ch 61 p 1520–1525
- **Behavioral validation:** Add bistable DA / 5-HT toggle on multi-hour timescale; observe alternating phases of high reward-pursuit (mania) vs anhedonia (depression).

### P.24 Anxiety Disorders — amygdala fear-learning hyperactivity
*[from Part IX — Diseases (Ch 57-64); renumbered from P.67]*

- **System:** Amygdala + PFC top-down inhibition
- **Biological role / Pathology:** Heightened amygdala fear-conditioning + impaired PFC extinction; benzodiazepines potentiate GABA-A. PTSD = persistent fear memory + impaired extinction.
- **Sim status:** missing-but-modelable (low fidelity) — declare amygdala region; pair CS with US punishment; vary GABA-A conductance.
- **Cluster:** P, B (GABA-A), reward (negative valence), D (memory)
- **Prerequisites:** amygdala region + fear-conditioning experiment preset (associative pairing already supported)
- **Citation:** Kandel 6e Ch 61 p 1525–1535
- **Behavioral validation:** Run associative-conditioning preset on amygdala region; lower GABA-A conductance → CS-conditioned avoidance response over-trains and resists extinction.

### P.25 Autism — Synaptic Adhesion Genes (Neurexin / Neuroligin / SHANK)
*[from Part IX — Diseases (Ch 57-64); renumbered from P.68]*

- **System:** Cortical excitatory & inhibitory synapses
- **Biological role / Pathology:** Mutations in trans-synaptic adhesion molecules (NRXN, NLGN3/4, SHANK1/2/3) disrupt synapse formation, maturation, E/I balance. Likely converge on PSD scaffolding and target recognition.
- **Sim status:** missing-needs-new-infrastructure — connection generation in `sim/connectivity.py` uses spatial / WS / motif rules without molecular target-recognition layer; PSD-organization is implicit in `weight`.
- **Cluster:** P, L (target recognition / PSD), J (E/I balance), M (development)
- **Prerequisites:** L.target-recognition (per-pathway compatibility code); PSD scaffolding as a separable variable
- **Citation:** Kandel 6e Ch 62 p 1531–1545
- **Behavioral validation:** Long-term — randomize a fraction of cortex→cortex pathway "compatibility codes" → reduced wiring specificity → reduced behavioral plasticity / sensory-specific responses.

### P.26 Fragile X Syndrome — FMR1 loss → exaggerated mGluR-LTD
*[from Part IX — Diseases (Ch 57-64); renumbered from P.69]*

- **System:** Cortical/hippocampal synapses
- **Biological role / Pathology:** Loss of FMRP (translational repressor) → unchecked dendritic protein synthesis → exaggerated mGluR5-dependent LTD, immature dendritic spines, altered E/I.
- **Sim status:** missing-needs-new-infrastructure — no mGluR pathway, no local protein synthesis.
- **Cluster:** P, J (mGluR), L (spine maturation), M (development)
- **Prerequisites:** mGluR / metabotropic plasticity layer
- **Citation:** Kandel 6e Ch 62 p 1535–1538
- **Behavioral validation:** Set LTD rate >> LTP rate globally → networks lose acquired patterns rapidly → Hebbian formation tasks fail.

### P.27 Rett Syndrome — MECP2 loss → broad transcriptional deregulation
*[from Part IX — Diseases (Ch 57-64); renumbered from P.70]*

- **System:** Whole-brain (esp. cortex) — neuronal maturation halted
- **Biological role / Pathology:** Loss of methyl-CpG-binding protein 2 → broad gene-expression changes; near-normal early development followed by regression. Female-predominant (X-linked).
- **Sim status:** missing-needs-new-infrastructure — no transcriptional layer; closest analog is freezing structural plasticity at a developmental waypoint.
- **Cluster:** P, M (development), Q (gene-regulation)
- **Prerequisites:** transcriptional / gene-state layer
- **Citation:** Kandel 6e Ch 62 p 1535
- **Behavioral validation:** Freeze all structural plasticity + reduce synaptic-scaling rate after a "regression point" → behavioral capacities acquired earlier are lost.

### P.28 Tuberous Sclerosis (TSC1/TSC2) — mTOR-pathway dysregulation
*[from Part IX — Diseases (Ch 57-64); renumbered from P.71]*

- **System:** Cortical neurons (with comorbid epilepsy + autism)
- **Biological role / Pathology:** Loss-of-function in TSC1/TSC2 → constitutive mTOR signaling → cortical tubers, dysmorphic neurons, hyperexcitability. Treatable with rapamycin/everolimus.
- **Sim status:** missing-needs-new-infrastructure — no protein-synthesis / mTOR signaling.
- **Cluster:** P, M (development), I (excitability), J (E/I balance)
- **Prerequisites:** signaling / growth-factor layer
- **Citation:** Kandel 6e Ch 58 p 1469 (and Ch 62)
- **Behavioral validation:** N/A directly; phenocopy via increased excitability + heterotopic-position neurons in cortex.

### P.29 Parkinson's Disease — SNc dopamine neuron death
*[from Part IX — Diseases (Ch 57-64); renumbered from P.72]*

- **System:** Nigrostriatal DA pathway
- **Biological role / Pathology:** Selective loss of SNc DA neurons (often α-synuclein Lewy pathology) → striatal DA depletion → indirect-pathway dominance (D2 disinhibited) → bradykinesia, rigidity, tremor. L-DOPA partially rescues.
- **Sim status:** **implemented (directly modelable, FLAGSHIP MATCH)** — Phase B BG cascade has cortex → D1 / D2 → GPi/GPe → thal → motor + a dopamine pool. Lesioning the dopamine pool / setting baseline DA = 0 reproduces the lesion exactly.
- **Cluster:** P, A (BG cascade), C (DA), F (motor)
- **Prerequisites:** none — already the simulator's strongest disease-modeling vector.
- **Citation:** Kandel 6e Ch 63 p 1545–1560
- **Behavioral validation:** In `g11_bg_runner`, set DA region size → 0 (or `nm_mgr.set_concentration("DA", 0)` + freeze production). Expect: indirect-pathway pools (str_D2_X, GPe → STN → GPi) dominate → motor pools fire less → moving-goal task reward sum collapses. Add L-DOPA analog (manual DA injection) → partial recovery.

### P.30 Alzheimer's Disease — amyloid + tau, hippocampal synapse loss
*[from Part IX — Diseases (Ch 57-64); renumbered from P.73]*

- **System:** Hippocampus, entorhinal cortex, eventually neocortex
- **Biological role / Pathology:** Aβ plaques (APP processing) + intracellular tau tangles → synapse loss in CA1/EC → episodic-memory failure progressing to global cognitive decline. Cholinergic basal-forebrain neurons especially vulnerable (rationale for AChE inhibitors).
- **Sim status:** partial — hippocampus presets exist; structural pruning can ablate synapses. Memory-loss validation feasible. No proteinopathy / staged-progression layer.
- **Cluster:** P, D (hippocampus), L (synapse loss), C (cholinergic)
- **Prerequisites:** stage-scheduled structural pruning targeted at hippocampus → cortex; cholinergic neuromodulator
- **Citation:** Kandel 6e Ch 63 p 1560–1571; Ch 64 p 1565–1572
- **Behavioral validation:** Train CA3 → CA1 associative memory; then progressively prune CA1 synapses + reduce ACh baseline → recall accuracy declines monotonically (analog of staged AD).

### P.31 Huntington's Disease — striatal MSN death (esp. D2 indirect-pathway)
*[from Part IX — Diseases (Ch 57-64); renumbered from P.74]*

- **System:** Striatum (MSNs), then cortex
- **Biological role / Pathology:** Autosomal-dominant CAG expansion in *HTT* → mutant huntingtin aggregates → selective MSN loss, indirect-pathway MSNs first → loss of inhibition on thalamus → chorea (involuntary movements), later cognitive + dementia.
- **Sim status:** **missing-but-modelable (high)** — BG cascade has explicit `str_D1_X` and `str_D2_X` pools; ablating a fraction of D2 MSNs preferentially is a direct test.
- **Cluster:** P, A (BG), F (motor), G (excitotoxicity / aggregation)
- **Prerequisites:** none for selective-pool ablation; aggregation dynamics would need new infrastructure.
- **Citation:** Kandel 6e Ch 63 p 1571–1577
- **Behavioral validation:** Progressively delete neurons in `str_D2_X` pools → indirect pathway weakens → involuntary motor activations (motor pools fire spontaneously without cortical drive). Quantify as motor-pool firing rate without sensory input.

### P.32 Familial ALS / Frontotemporal Dementia — TDP-43, C9ORF72, FUS
*[from Part IX — Diseases (Ch 57-64); renumbered from P.75]*

- **System:** Motor neurons + frontal-temporal cortex
- **Biological role / Pathology:** RNA-binding-protein dysfunction → cytoplasmic aggregates → neuronal death. C9ORF72 GGGGCC repeat causes both ALS + FTD (overlap spectrum).
- **Sim status:** missing-needs-new-infrastructure — no protein homeostasis. Phenocopy = staged ablation of motor + frontal pools.
- **Cluster:** P, F (motor), PFC (working memory), G (proteinopathy)
- **Prerequisites:** RNA / protein-aggregation state per neuron
- **Citation:** Kandel 6e Ch 63 p 1577–1582
- **Behavioral validation:** Combined ablation of motor pools + PFC region in g11 runner over time → behavioral disinhibition + motor failure.

### P.33 Prion Diseases — propagating misfolded protein
*[from Part IX — Diseases (Ch 57-64); renumbered from P.76]*

- **System:** Variable (cortex for CJD, thalamus for FFI, cerebellum for kuru)
- **Biological role / Pathology:** PrPᶜ → PrPˢᶜ template-conversion + spread → spongiform encephalopathy. Genuinely transmissible without nucleic acid.
- **Sim status:** missing-needs-new-infrastructure — no protein-conformation / spatial-spread layer.
- **Cluster:** P, M (cell-cell propagation), G (proteinopathy)
- **Prerequisites:** spatial diffusion of an "infection" state across coupled neurons
- **Citation:** Kandel 6e Ch 63 p 1582–1588
- **Behavioral validation:** N/A at current architecture.

### P.34 Spinocerebellar Ataxias / Polyglutamine Diseases — CAG-expansion family
*[from Part IX — Diseases (Ch 57-64); renumbered from P.77]*

- **System:** Cerebellum (SCAs), basal ganglia (HD), spinal cord (SBMA)
- **Biological role / Pathology:** Expansions of CAG (polyQ) in disease-specific genes cause aggregation-related selective neuronal death in distinct circuits.
- **Sim status:** missing-needs-new-infrastructure — same proteinopathy gap. Phenocopy: targeted region ablation.
- **Cluster:** P, F (motor), F.cerebellum, G (proteinopathy)
- **Prerequisites:** cerebellum region + Purkinje preset (HH preset exists; full circuit not built)
- **Citation:** Kandel 6e Ch 63 p 1577 (Table)
- **Behavioral validation:** Build cerebellar region; ablate Purkinje cells progressively → loss of motor-timing precision (use existing motor task).

### P.35 The Aging Brain — slow synapse loss, glial reactivity, neurogenesis decline
*[from Part IX — Diseases (Ch 57-64); renumbered from P.78]*

- **System:** Whole brain (PFC, hippocampus most affected)
- **Biological role / Pathology:** Mild cognitive decline largely attributable to synapse loss + impaired plasticity, *not* widespread neuron death. Hippocampal neurogenesis declines. Microglial activation rises. Vascular contributions important.
- **Sim status:** missing-but-modelable (low fidelity) — synapse-pruning rate can be raised over sim-time; plasticity rates can be reduced. No glial / vascular / neurogenesis state.
- **Cluster:** P, D (hippocampus), L (synapse density), J (plasticity decline)
- **Prerequisites:** time-decaying plasticity + structural-pruning rate
- **Citation:** Kandel 6e Ch 64 p 1561–1572
- **Behavioral validation:** Linearly decay `reward_learning_rate` and increase pruning rate over a long run → previously fast-learning agent gradually slows in re-adapting to new goals.

### P.36 Stroke / Vascular Insult — focal ischemic lesion
*[from Part IX — Diseases (Ch 57-64); renumbered from P.79]*

- **System:** Anywhere in vascular territory
- **Biological role / Pathology:** Acute ischemia → ATP depletion → glutamate excitotoxicity → cell death in core; penumbra at risk. Recovery via cortical reorganization.
- **Sim status:** missing-but-modelable (low fidelity) — instantaneous ablation of a contiguous spatial region of neurons + structural plasticity in surround = lesion + recovery analog.
- **Cluster:** P, G (excitotoxicity), structural plasticity (recovery), Q (vascular)
- **Prerequisites:** spatial-region neuron deletion API; structural plasticity exists.
- **Citation:** Kandel 6e Ch 64 p 1572–1574 (vascular contributions to aging)
- **Behavioral validation:** Delete all neurons within a sphere in motor cortex during a trained run → observe behavioral deficit + partial recovery as surround sprouts (existing structural plasticity).

### P.37 Excitotoxicity (general mechanism) — NMDA-mediated cell death
*[from Part IX — Diseases (Ch 57-64); renumbered from P.80]*

- **System:** Any glutamatergic circuit under metabolic stress
- **Biological role / Pathology:** Excessive glutamate / Ca²⁺ entry via NMDA → mitochondrial Ca²⁺ overload → caspase activation → death. Common pathway across stroke, status epilepticus, ALS, HD, AD.
- **Sim status:** missing-needs-new-infrastructure — generic Ca²⁺ overload + cell-death trigger (used across multiple disease models above).
- **Cluster:** P, J (NMDA), G (Ca²⁺), Q (mitochondria)
- **Prerequisites:** intracellular [Ca²⁺] state with overload threshold → ablation
- **Citation:** Kandel 6e Ch 58 p 1480–1483 (also recurs through Ch 63)
- **Behavioral validation:** Generic substrate: deliver sustained NMDA current to test region; verify monotonic ablation as duration grows. Then re-use this across P.60, P.72, P.73, P.74, P.79.

---

## Notes on cross-cluster prerequisites (recurring gaps)

Six pieces of new infrastructure would unlock a large block of disease entries currently marked "missing-needs":
1. **Per-cell-type channel / receptor scaling** (used by P.51, P.58, P.63) — extension of existing HH preset system.
2. **Intracellular [Ca²⁺] pool with overload-triggered ablation** (P.60, P.72-progressive, P.79, P.80).
3. **Time-progressive / staged neuron-or-synapse ablation API** (P.53, P.55, P.73, P.74, P.78).
4. **Cholinergic + serotonergic neuromodulator declarations** (P.65, P.67, P.73 — neuromodulator framework is already in place; just declare).
5. **Axon / conduction-delay distinct from synaptic delay** (P.54).
6. **Protein-aggregation / proteinopathy state with spatial spread** (P.75, P.76, P.77).

Items 1, 3, and 4 are small extensions of existing infrastructure and would unlock 8+ disease entries. Items 2 and 6 are larger but central to neurodegeneration.

`[discrepancy: none material — Kandel's Parkinson description matches the simulator's BG cascade unusually well; the cascade was built independent of disease-modeling intent but produces a textbook-faithful PD lesion model out of the box.]`

## Cluster Q — additions

### Q.01 Oligodendrocyte / Schwann myelination
*[from Part II — Cells & Channels (Ch 7-10); renumbered from Q.50]*

- **System:** CNS (oligodendrocyte) and PNS (Schwann); white matter and peripheral nerve.
- **Biological role:** Wraps concentric layers of glial membrane around axons to insulate and increase rm/decrease cm internodally, enabling saltatory conduction. One oligodendrocyte myelinates up to 30 axons; Schwann cells one segment per cell. Myelin composition: 70% lipid + MBP / PLP / P0 / PMP22.
- **Sim status:** **missing** — no glial population, no axonal compartment. Conduction velocity is a lumped synaptic-delay parameter.
- **Cluster:** Q (primary), I (saltatory propagation depends on this).
- **Prerequisites:** I.67.
- **Citation:** Kandel 6e Ch 7 p 151–154.
- **Behavioral validation:** N/A in single-compartment model. Would require axonal compartments to validate conduction-velocity scaling.

### Q.02 Astrocyte K+ buffering and glutamate uptake
*[from Part II — Cells & Channels (Ch 7-10); renumbered from Q.51]*

- **System:** Gray-matter neuropil; tripartite synapse.
- **Biological role:** Astrocyte membranes are heavily Kir4.1-positive; take up K+ released by active neurons (preventing extracellular K+ accumulation during sustained firing) and redistribute it via gap-junction syncytium. EAAT1/2 transporters clear synaptic glutamate (preventing excitotoxicity), converting it to glutamine for return to neurons.
- **Sim status:** **missing** — extracellular ion concentrations are constant; no glial pool, no glutamate clearance dynamics. Synaptic transmission assumes instant cleanup.
- **Cluster:** Q (primary), J (glutamate dynamics).
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 7 p 154–156.
- **Behavioral validation:** Sustained high-frequency firing should NOT accumulate extracellular K+ in current sim (it can't); a future validation would require [K+]o tracking.

### Q.03 Microglia — surveillance, synaptic pruning, immune response
*[from Part II — Cells & Channels (Ch 7-10); renumbered from Q.52]*

- **System:** Resident myeloid cells throughout CNS.
- **Biological role:** Resting microglia continuously survey neuropil with fine processes; on injury, transform to amoeboid, phagocytose debris, release cytokines. Developmentally, prune weak synapses via complement-tagging (C1q/C3). Implicated in schizophrenia, Alzheimer.
- **Sim status:** **missing** — no microglia, no immune signalling. Structural plasticity in `struct_plast_activity_bias` provides activity-dependent synapse formation/elimination but is not microglia-mediated.
- **Cluster:** Q (primary), L (developmental synaptic pruning), P (neurodegeneration).
- **Prerequisites:** none.
- **Citation:** Kandel 6e Ch 7 p 156–161.
- **Behavioral validation:** N/A in current sim. Pruning could be validated by an inactivity-tagged removal rule.

### Q.04 Astrocyte-mediated tripartite synapse (gliotransmission)
*[from Part II — Cells & Channels (Ch 7-10); renumbered from Q.53]*

- **System:** Neuropil; especially hippocampus, cortex.
- **Biological role:** Astrocytes express NMDA-like and metabotropic glutamate receptors; glutamate spillover triggers astrocyte Ca2+ waves, propagated via gap junctions, that release ATP/D-serine/glutamate ("gliotransmitters") back onto neurons. Modulates LTP/LTD and slow-wave sleep dynamics.
- **Sim status:** **missing** — no astrocyte compartment, no D-serine, no Ca2+ waves. NMDA-block formula is purely neuronal-side.
- **Cluster:** Q (primary), J (plasticity gating), N (sleep oscillations).
- **Prerequisites:** I.59 (Ca2+ pool needed first).
- **Citation:** Kandel 6e Ch 7 p 154–156.
- **Behavioral validation:** Blocking gliotransmission should reduce LTP magnitude in a CA3→CA1 protocol.

---

End of Part II catalog (24 entries: 22 in Cluster I, 4 in Cluster Q;
some entries cross-tagged to N, L, P, G, J, C).

### Q.05 Wallerian degeneration — distal-stump axon self-destruction
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from Q.60]*

- **System:** any severed axon (PNS and CNS); SARM1 / NMNAT2 axis
- **Biological role:** Distal axon segments degenerate within 1-3 days post-axotomy via an active self-destruct program (SARM1 activated when NMNAT2 supply is cut → NAD+ collapse → calcium overload). Not passive starvation — the *Wlds* mutant (stable NMNAT) delays degeneration by weeks.
- **Sim status:** not-applicable — no axons as physical entities; no axotomy concept; "synapse loss" via structural plasticity is independent of axonal-segment biology.
- **Cluster:** Q (glia/repair)
- **Prerequisites:** none from this part
- **Citation:** Kandel 6e Ch 50 pp 1238-1242
- **Behavioral validation:** *Wlds* mutant mice show dramatically slowed distal-stump degeneration (weeks vs hours)

### Q.06 Schwann-cell-mediated PNS regeneration
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from Q.61]*

- **System:** peripheral nerve; Schwann cells transition to repair phenotype after injury
- **Biological role:** Distal Schwann cells dedifferentiate, proliferate, clear myelin debris, secrete NGF/BDNF/GDNF, and form Bands of Büngner that guide regenerating proximal-stump axons back to targets. Why PNS regenerates and CNS does not is largely an extrinsic (glial) question, not an intrinsic (neuron) one.
- **Sim status:** not-applicable — no glia, no axon regrowth, no nerve injury model. Recovery in our framework is via plasticity reweighting rather than physical regrowth.
- **Cluster:** Q
- **Prerequisites:** Q.60
- **Citation:** Kandel 6e Ch 50 pp 1240-1244
- **Behavioral validation:** transected sciatic nerve recovers function over weeks; same lesion in optic nerve does not

### Q.07 CNS regeneration failure — myelin inhibitors (Nogo, MAG, OMgp) & glial scar
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from Q.62]*

- **System:** central nervous system white matter; oligodendrocyte-derived inhibitors
- **Biological role:** Adult CNS axons fail to regenerate not because neurons can't grow but because the environment actively repels them. Myelin-associated Nogo-A, MAG, and OMgp all bind NgR/PirB on growth cones to collapse them. Reactive astrocytes form a chondroitin-sulfate-proteoglycan-rich glial scar. This is *why* mature connectivity is effectively fixed in mammals — it provides indirect biological support for the simulator's design choice that wiring is set at config and only plasticity (not regrowth) operates at runtime.
- **Sim status:** not-applicable directly, but **conceptually relevant** — the project's structural-plasticity / pruning model assumes connections are gained/lost at the synapse level, not by axon regrowth. CNS-regen failure is the textbook justification for this.
- **Cluster:** Q (primary), L (development — same molecules act as guidance cues during development; their persistence into adulthood inhibits regen)
- **Prerequisites:** Q.60, L.69
- **Citation:** Kandel 6e Ch 50 pp 1242-1252
- **Behavioral validation:** Nogo-A KO or anti-Nogo Ab modestly improves spinal-cord-injury recovery in rodent models; chondroitinase ABC dissolves perineuronal-net component of glial scar and improves recovery

### Q.08 Reactive gliosis & glial scar — astrocyte response to CNS injury
*[from Part VII extras — Development (Ch 45-47, 50-51); renumbered from Q.63]*

- **System:** astrocytes & microglia at any CNS lesion site
- **Biological role:** Astrocytes hypertrophy, upregulate GFAP, and proliferate around damage; microglia clear debris but slowly compared to PNS macrophages. The scar walls off damage but obstructs regrowth.
- **Sim status:** not-applicable — no glia, no reactive states.
- **Cluster:** Q
- **Prerequisites:** Q.60
- **Citation:** Kandel 6e Ch 50 pp 1239-1245
- **Behavioral validation:** GFAP/vimentin double KO reduces scar density, modestly improves regeneration

## Cluster L — sex differentiation (placed here as developmental)
