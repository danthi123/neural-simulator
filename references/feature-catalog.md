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

- [Cluster J — Synapses & plasticity rules](#cluster-j--synapses--plasticity-rules) — Ch 11, 12, 13 (so far); Ch 14, 15, 16, 53 pending
- [Cluster I — Channels & intrinsic dynamics](#cluster-i--channels--intrinsic-dynamics) — Ch 13 (AIS); Ch 8-10 pending
- [Cluster G — Working memory / PFC / cortical integration](#cluster-g--working-memory--pfc--cortical-integration) — Ch 13
- [Cluster B — Striatal microcircuit & cortical interneuron diversity](#cluster-b--striatal-microcircuit--cortical-interneuron-diversity) — Ch 13
- [Cluster M — Neuromuscular junction](#cluster-m--neuromuscular-junction) — Ch 12
- (pending) Cluster L — Development & critical periods — Ch 48, 49

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
