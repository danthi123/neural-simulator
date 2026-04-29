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

## Cluster A — Closed BG action-selection loop

**9 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

### A.01 Direct pathway — D1 MSN → GPi/SNr disinhibition (action gating)

*[from Part V — Movement (Ch 30-39); renumbered from A.50]*

- **System:** D1-receptor medium spiny neurons (substance P / dynorphin) in striatum → GABAergic projection to internal globus pallidus (GPi) and substantia nigra pars reticulata (SNr).
- **Biological role:** Cortical input excites D1 MSNs → MSN inhibits the GPi/SNr neurons whose tonic 40–80 Hz firing was suppressing thalamus → thalamus disinhibited → cortex / brainstem effector released. This is the "go" pathway. Dopamine via D1 (Gs-coupled) increases MSN excitability.
- **Sim status:** **implemented** — `g11_bg_runner.build_bg_brain_regions` declares per-action `str_d1_X → gpi_X → thal_X → motor_X`. Disinhibitory cascade is the exact mechanism of A.50. cortex→D1 is plastic with `stdp_w_max=30`. **This is the simulator's flagship architecture.**
- **Cluster:** A primary; C (DA) secondary.
- **Prerequisites:** B.* (MSN microcircuit), C.* (DA modulation), I.*, J.*.
- **Citation:** Kandel 6e Ch 38 p 935–943.
- **Behavioral validation:** D1-pool stimulation → GPi pause → thalamus burst → motor selection; matches `g11_bg_runner` cascade probe.
- **Supplemental:** D1 MSN axons innervate the proximal somatodendritic compartment of GPi/SNr output neurons in a *basket-like* pattern, with large boutons selectively concentrated on the proximal regions (Bolam-2000 p 535, "In the output nuclei, pallidal neurons give rise to large synaptic boutons that selectively innervate the proximal regions of basal ganglia output neurons, often in a basket-like manner"). The corticostriatal terminals that drive D1 MSNs synapse on the *heads* of dendritic spines, while DA terminals contact the *necks* of the same spines — placing DA in an ideal position to gate cortical input to each spine independently (Bolam-2000 pp 529–531, Fig 2B,D). The current single-compartment MSN model collapses spine-level head/neck modulation; faithful reproduction would require dendritic compartments with separate AMPA/NMDA-bearing spine heads and DA-receptor-bearing necks.
- **Supplemental:** PBR-160 ch 16 (McGinty pp 273–280) clarifies the neuropeptide phenotype of the *direct* (striatonigral) pathway: D1 MSNs co-release **dynorphin + substance P (PPD/SP)** alongside GABA. Dynorphin acts on presynaptic κ-opioid receptors (KORs) on glutamatergic and dopaminergic terminals to *suppress* DA + glutamate release — a homeostatic brake against runaway D1 activity (McGinty Fig 5, p 280). Substance P acts on NK-1 receptors expressed predominantly by cholinergic interneurons, *increasing* striatal ACh release. Net effect: D1-MSN activation does not just inhibit GPi/SNr — it also closes a local DA + Glu auto-regulatory loop and excites cholinergic interneurons. **Sim implication:** the simulator's D1 cascade currently models the GABAergic disinhibition only; modeling the dynorphin/SP arms would require neuromodulator subsystem entries for KOR (suppressive) and NK-1 (cholinergic-excitatory). Maps to declarative neuromodulator framework (sim/neuromodulators.py).
- **Supplemental (anatomy):** PBR-160 ch 9 (Deniau et al. pp 158–160) — striatonigral conduction velocity is **the slowest of all long-range BG GABAergic projections**, ~1.4 m/s, antidromic latency ~10 ms. Pallidonigral conduction velocity is ~4 m/s (~3× faster). The simulator's `cortex_X → str_d1_X → gpi_X` cascade implicitly assumes uniform conduction; adding axonal-delay differentiation between striatal vs pallidal inputs to GPi/SNr would let phasic GPe input arrive *before* the slower D1 gate, matching the in-vivo three-phase response sequence (early STN excitation → striatal-mediated inhibition → late STN excitation; PBR-160 ch 7 Fig 6).

### A.02 Indirect pathway — D2 MSN → GPe → STN → GPi/SNr (action suppression)

*[from Part V — Movement (Ch 30-39); renumbered from A.51]*

- **System:** D2-receptor MSNs (enkephalin) → GABAergic projection to external globus pallidus (GPe) → GPe inhibits STN → STN excites GPi/SNr → increased tonic inhibition of thalamus.
- **Biological role:** "No-go" pathway. Increased D2 activity increases GPi output → suppresses non-selected actions / brakes movement. Dopamine via D2 (Gi-coupled) decreases MSN excitability → disinhibits the indirect pathway less. Imbalance at root of Parkinson (less DA → indirect dominant → bradykinesia) and Huntington (D2 MSN loss → direct dominant → chorea).
- **Sim status:** **implemented** — `g11_bg_runner` declares `str_d2_X → gpe_X → stn (shared) → gpi_X` per action.
- **Cluster:** A primary; C, P (Parkinson/Huntington) secondary.
- **Prerequisites:** A.50, B.*, C.*.
- **Citation:** Kandel 6e Ch 38 p 935–943, p 952–956.
- **Behavioral validation:** D2-pool stimulation → GPi increase → action suppressed; DA depletion → indirect dominant → reduced action initiation.
- **Supplemental:** Bolam emphasizes the indirect pathway is more complex than the canonical D2→GPe→STN→GPi triad. Single-cell labelling (Kita & Kitai 1994; Bevan et al. 1998) shows that *individual* GPe neurons collateralize into multiple BG nuclei: a typical GPe neuron innervates GPe locally (~92–294 boutons), STN (~41–274 boutons), GPi/EP (~108–130 boutons), AND SNr/SNc (Bolam-2000 Fig 4 legend, p 537). A subset (~25%) also projects back to striatum (see new entry A.10). The simulator's `gpe_X → stn` is therefore one of several collateral targets per GPe cell, not a dedicated projection. **[discrepancy: per-action GPe pool with single downstream target oversimplifies; real GPe neurons broadcast to all caudal BG nuclei simultaneously].**
- **Supplemental:** PBR-160 ch 16 (McGinty pp 273–276) fixes the *indirect* (striatopallidal) pathway peptide phenotype: D2 MSNs co-release **enkephalin** (PPE-derived) alongside GABA. Enkephalin binds δ-opioid receptors (DORs) expressed by *cholinergic interneurons* and µ-opioid receptors (MORs) clustered in striosomes. Enkephalin acts via DORs to *increase* DA release and *inhibit* ACh release — opposing the dynorphin/D1 effect. **Sim implication:** the D1/D2 split in `g11_bg_runner` captures only the GABA arm; the opposing neuropeptide arms (D1: KOR↓DA, NK-1↑ACh; D2: DOR↑DA, MOR various) provide a slower, second-order contrast that may help stabilize action selection without explicit homeostatic terms.
- **Supplemental:** PBR-160 ch 7 (Kita pp 113–115) — striatopallidal axons make ~100–250 boutons each (small/medium, ~<1 µm), distributed sparsely along GPe dendrites. Each individual stria→GPe axon evokes only **<10 pA unitary IPSCs** at the GPe soma (ch 7 Fig 4). The Str:GPe neuron ratio is **~60:1** (rat). Together this means GPe inhibition only emerges when *many* synchronously-firing striatal MSNs converge — directly validating the simulator's per-action-pool synchronization design but suggesting the per-pool MSN count should be larger (currently 25/action) to capture the threshold-like cooperative gating.

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
- **Supplemental:** Bolam-2000 Fig 3E and pp 535–537 demonstrate that individual GPi/SNr/STN output neurons receive *convergent synaptic input from both globus pallidus (GP, sensorimotor/associative) AND ventral pallidum (VP, limbic)* in zones of overlap. Output dendrites are oriented to cross functional boundaries defined by these inputs. The selection at GPi is therefore not a pure motor competition — limbic and motor-associative information are integrated at the level of individual output neurons. This composes naturally with the project's v3 MSN lateral inhibition (cheat #5 closure) but suggests a future extension where motor-channel GPi neurons also receive limbic inputs from a ventral-pallidum analog.
- **Supplemental — intrinsic pacemaker biophysics (PBR-160 ch 9, Deniau et al. pp 157–158):** SNr GABAergic projection neurons are *autonomous pacemakers* — they fire 40–80 Hz tonically *with all fast synaptic transmission blocked*. The pacemaker rests on three mechanisms: (1) a **slowly-inactivating, voltage-dependent TTX-sensitive Na⁺ current** that drives subthreshold depolarization toward AP threshold; (2) a **TTX-insensitive cation current** (partly Na⁺-mediated) contributing to the pacemaker drive; (3) **SK channels** (small-conductance Ca²⁺-activated K⁺) coupled to o-conotoxin-GVIA-sensitive Cav2.2 currents — these set the AHP and thereby the *precision* (regularity) of firing. SK blockade by apamin reduces firing precision. SNr also expresses Ih and can generate slow Ca²⁺ spikes from below −60 mV. **Sim implication:** the project's `HH_GPI_OUTPUT` preset uses a generic HH framework. Faithful 40–80 Hz tonic SNr/GPi behavior would benefit from explicit persistent Na (`hh_NaP`) + SK channel + Ih in the preset. The simulator already exposes `fused_hh_NaP_current_update` and `fused_hh_h_current_update` kernels — these should be enabled for SNr/GPi.
- **Supplemental — striatal vs pallidal targeting on GPi/SNr (PBR-160 ch 8 Nambu pp 137–139, Fig 2):** synaptic terminals on individual GPi neurons differ systematically by *origin*: striatal GABA terminals contact **distal dendrites** (~70% of all inputs), GPe GABA terminals contact **soma + proximal dendrites** (~15%), STN glutamate terminals are **uniformly distributed on soma + dendrites** (~10%). Same pattern in SNr (PBR-160 ch 9 p 159). This is exactly opposite of the simulator's single-compartment assumption (all inputs sum at the soma). The functional consequence is that GPe-mediated inhibition has **much stronger somatic veto power** per synapse than striatal inhibition — under normal conditions, GPe disynaptic disinhibition (cortex → GPe → GPi/SNr) can override the slower striatal direct pathway. The simulator's flagship cascade omits GPe→GPi/SNr altogether; adding `gpe_X → gpi_X` with high relative weight (perisomatic-equivalent) and `str_d1_X → gpi_X` with lower per-synapse weight (distal-dendrite-equivalent) would replicate the distal/proximal asymmetry. **[discrepancy: single-compartment GPi can't represent compartment-specific input strengths; current cascade models distal striatal pathway only].**

### A.05 Reentrant cortico-BG-thalamo-cortical loops — parallel channels

*[from Part V — Movement (Ch 30-39); renumbered from A.54]*

- **System:** Alexander/DeLong scheme: motor / oculomotor / dorsolateral prefrontal / lateral orbitofrontal / anterior cingulate loops; topographically segregated through STR → GPi/SNr → thalamus → back to source cortex.
- **Biological role:** Loops are largely segregated but offer relay points where outside information can modulate signal flow. Functional territories preserved: limbic ventromedial → associative middle → sensorimotor dorsolateral gradient in striatum.
- **Sim status:** **partial** — flagship implements 4 sensorimotor channels (one per action). Limbic and associative channels not modeled. **[discrepancy: textbook emphasizes 5 parallel functional loops sharing the same circuit motif; project models only the motor loop].**
- **Cluster:** A primary; G secondary.
- **Prerequisites:** A.50–A.53.
- **Citation:** Kandel 6e Ch 38 p 943–948 (Alexander, DeLong, Strick).
- **Behavioral validation:** Anatomical: stimulating ACC-BG channel modulates orbitofrontal output without affecting motor channel.
- **Supplemental:** Bolam-2000 (pp 535–537) and Joel & Weiner (1994, 1997) cited therein argue against strictly segregated parallel loops. Anatomical evidence shows convergence at multiple levels: (a) PV+ FSIs receive integrated motor + sensory cortical input (~25–50% of FSIs apposed by both motor and sensory terminals — Hanley/Deniau/Bolam unpublished, Bolam-2000 Fig 3A,B, p 533); (b) GPe neurons project to all caudal BG nuclei plus the striatum, providing inter-channel coupling; (c) GPi/SNr/SNc dendrites integrate VP + GP inputs at the cell level. This *deepens* the existing discrepancy note: textbook (Kandel/Alexander) describes 5 parallel loops; the BG anatomy literature shows the loops are *extensively cross-coupled by the local microcircuitry*. The project's same-action-only routing is an even larger simplification than originally noted.

### A.06 Cortico-striatal topography — sensorimotor / associative / limbic gradient

*[from Part V — Movement (Ch 30-39); renumbered from A.55]*

- **System:** dorsolateral STR (sensorimotor) ← motor cortex; central STR (associative) ← prefrontal; ventromedial STR (limbic) ← OFC, ACC, amygdala, hippocampus.
- **Biological role:** Each functional zone of cortex maps to a corresponding zone of striatum. Same MSN microcircuit applied to functionally diverse afferents ⇒ basal ganglia perform the *same* selection computation across motor, cognitive, and motivational domains.
- **Sim status:** **partial** — sensorimotor mapping captured by per-action cortex_X→str_X. No associative or limbic stripe.
- **Cluster:** A primary.
- **Prerequisites:** A.54.
- **Citation:** Kandel 6e Ch 38 p 943–948.
- **Behavioral validation:** Lesion dorsolateral STR → motor deficit; lesion ventromedial STR → motivational deficit; lesion associative → cognitive deficit.
- **Supplemental:** Bolam-2000 (p 529, citing Kincaid et al. 1998) provides a concrete corticostriatal convergence rule that constrains wiring: an *individual* cortical neuron makes only 1–2 synapses with an *individual* MSN, but there is high convergence onto each MSN, AND **close-neighbour MSNs do not share common cortical inputs**. The implication for the simulator: inside a per-action pool, the cortex→MSN matrix should be sparse + decorrelated (each cortical axon contacts a few MSNs, neighbouring MSNs sample disjoint cortical sources). Current `g11_bg_runner` uses dense per-action all-to-all wiring (`density` ~0.3–0.5) which violates this. A sparser, decorrelated wiring would more faithfully replicate the "noisy decorrelation" that Bolam suggests is the substrate for striatal ensemble formation.

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

### A.10 Pallidostriatal feedback — GPe → striatal interneurons (selective FSI/NOS targeting)
- **System:** ~25% of GPe neurons issue collateral projections back to striatum (Kita & Kitai 1994; Bevan et al. 1998). Each pallidostriatal axon contributes ~790 boutons within striatum; 44 ± 18% of those boutons selectively contact PV+ FSI interneurons; 3–32% contact NOS+/NPY+ LTS interneurons. Quantitative model (Bolam-2000 Table 1, p 534): each striatal FSI receives ~7 GPe neurons' input, totalling ~48 GPe boutons; predominantly perisomatic / proximal-dendrite targeting.
- **Biological role:** GPe powerfully gates striatal output via feedforward FSI inhibition rather than direct MSN contact. GPe is monosynaptically activated by cortex (or rapidly disynaptically via STN), so the loop is corticostriatal → STR → GPe → striatal FSI → MSN — a fast feedback path that *shunts* cortical excitation of MSNs and may phase-lock or prevent action potential generation in the next round. Functionally the GPe is in a position to control nearly the whole striatum (Bolam-2000 pp 534, 538).
- **Sim status:** **missing** — `g11_bg_runner` has `gpe_X` regions but no GPe → striatum projection. Adding a `gpe_X → str_fsi_X` pathway (once an FSI pool exists per B.04 augmentation) is a small builder change with strong biological backing.
- **Cluster:** A primary; B secondary.
- **Prerequisites:** A.02 (GPe), B.06 (FSI population — currently missing).
- **Citation:** Bolam-2000 pp 533–538, Table 1; Kita & Kitai 1994; Bevan et al. 1998; TK-2017 p 161 ("FSIs receive GABAergic input from at least two different neuron populations in the GPe, PV+ and PV−").
- **Behavioral validation:** GPe-stim → 5–10 ms FSI spike → MSN IPSP at 8–15 ms; pallidal lesion / inactivation → MSN hyperexcitability and altered timing of striatal output.
- **Supplemental:** PBR-160 ch 7 (Kita pp 112–114, Fig 2C) confirms and extends the Bolam-2000 finding: in monkey GPe, **the GPe→Str projection arises predominantly from PV-negative GPe neurons** (sparse-spiny dendrites, often immunoreactive for preproenkephalin mRNA), forming about ~1/3 of all GPe cells. PV-positive GPe neurons (large aspiny, discoidal dendrites parallel to Str/GPe border) form the canonical GPe→GPi/STN/SNr projection. **In monkey** (but not rat), the two cell types appear to be more strictly segregated: PV-negative GPe→Str neurons rarely collateralize, while PV-positive cells do. This is the anatomical antecedent of the now-canonical **prototypic (PV+) vs arkypallidal (PV−)** classification that Mallet et al. 2008 later formalized — Kita 2007 already shows the morphological + neurochemical split. **Sim implication:** if the project adds GPe→Str feedback (currently missing in `g11_bg_runner`), splitting the GPe pool into a PV+ subpool projecting to STN/GPi/SNr and a PV− subpool projecting only to striatal FSIs would be more biologically grounded than a single mixed pool.

### A.11 Pallidal convergence on BG output — GP+VP integration on individual SNr/STN/SNc neurons
- **System:** Single SNr, STN, and SNc neurons receive convergent input from BOTH globus pallidus (GP, sensorimotor/associative — via dorsal striatum) AND ventral pallidum (VP, limbic — via nucleus accumbens). Double anterograde tracing reveals topographically segregated fields with overlapping zones; in the overlap zones, individual output neurons are contacted by GP and VP boutons simultaneously, often perisomatically (Bolam-2000 pp 535–537, Fig 3E).
- **Biological role:** Provides anatomical substrate for synaptic integration of *functionally diverse* (motor + limbic) information at the BG output stage. Output dendrites are often oriented to cross GP/VP territory boundaries. This enables motivational state (limbic) to gate motor selection (sensorimotor) at the level of individual GPi/SNr cells — not via separate parallel loops as Alexander/DeLong scheme suggests.
- **Sim status:** **missing** — the project models only the motor channel; no VP or limbic-stripe equivalent. Adding a VP region with `vp → gpi_X / stn / snc` projections would let limbic value modulate motor selection at the cell level.
- **Cluster:** A primary; G secondary (working-memory / limbic integration).
- **Prerequisites:** A.05 (parallel loops), C.* (DA — for SNc target).
- **Citation:** Bolam-2000 pp 535–537, Fig 3E (Bevan et al. 1996, 1997).
- **Behavioral validation:** Single-cell GPi recordings show responses modulated by both arm-movement (motor) and reward-context (limbic) variables; lesion of either GP or VP shifts response selectivity.

### A.12 Sparse, decorrelated cortico-striatal convergence — Kincaid wiring rule
- **System:** Cortico-striatal terminals; corticostriatal axons make asymmetric synapses on MSN spine heads. Each individual cortical neuron contacts an MSN with only 1–2 boutons; convergence onto each MSN is high (thousands of cortical axons), but **close-neighbour MSNs do not share common cortical inputs** (Kincaid et al. 1998, reviewed Bolam-2000 p 529).
- **Biological role:** A "decorrelation rule" — cortical drive to neighbouring MSNs is statistically independent, which is essential for MSN ensembles to encode distinct features. With shared inputs, MSNs would co-fire and lose discriminative power; with totally independent inputs, no ensemble structure forms. The 1–2 contacts per axon + non-overlapping neighbour rule is the substrate for ensemble decorrelation in striatum.
- **Sim status:** **missing** — `g11_bg_runner` uses dense per-action `cortex_X → str_d1_X` wiring (density ~0.3–0.5, no per-axon contact-count constraint). MSNs within a pool effectively share cortical drive. A switch to sparse, per-axon-bounded, neighbour-decorrelated wiring would let intra-pool MSN ensembles emerge.
- **Cluster:** A primary; B, J (plasticity) secondary.
- **Prerequisites:** A.06.
- **Citation:** Bolam-2000 p 529 citing Kincaid, Zheng, Wilson 1998 J. Neurosci. 18:4722–4731.
- **Behavioral validation:** In vivo paired MSN recordings show low spike correlation between neighbours despite shared cortical region of origin; manipulations that increase cortical input sharing reduce MSN ensemble discriminability.

---

### A.13 GPe cell-type heterogeneity — prototypic (PV+) vs arkypallidal (PV−), two firing modes
- **System:** GPe contains at least two morphologically + neurochemically + physiologically distinct projection neuron subtypes (Kita & Kitai 1994; Kita 1996; Cooper & Stanford 2000; reviewed PBR-160 ch 7 Kita pp 111–114).
  - **PV+ "prototypic" GPe neurons:** ~2/3 of GPe; large aspiny soma with discoidal dendritic field oriented parallel to the Str/GPe border; project to STN, GPi, SNr (multi-target collateralization). High-frequency tonic firing (~30–80 Hz) interspersed with spontaneous pauses ("HFD-pause" pattern in vivo). Membrane: weak Ih, no rebound Ca²⁺ spike, no spike accommodation (PBR-160 ch 7 p 112).
  - **PV− "arkypallidal" GPe neurons:** ~1/3 of GPe; sparsely-spiny radiating dendrites, often preproenkephalin-mRNA-positive. **Project predominantly back to striatum** (pallidostriatal feedback — see A.10), targeting striatal interneurons (FSI, SOM/NOS) selectively. Low-frequency, bursty firing in vivo. Membrane: prominent Ih, prominent rebound Ca²⁺ spike, no spike accommodation.
  - In monkey, the two subtypes are **strictly segregated by projection target** — PV− cells exclusively → Str, PV+ cells exclusively → STN/GPi/SNr (Kita et al. 1999, ch 7 p 111). In rat, segregation is partial (~25% of GPe cells collateralize to Str on the way to STN/GPi).
  - This is the anatomical antecedent of the now-canonical Mallet et al. 2008 "prototypic vs arkypallidal" classification, with the same projection pattern but additional in-vivo firing dynamics across cortical UP/DOWN states.
- **Biological role:** PV+ pool is the canonical indirect-pathway relay (Str-D2→GPe→STN→GPi). PV− pool implements *pallidostriatal feedback inhibition* — fast disynaptic gating of striatal output through FSI/LTS interneurons (A.10). The two subtypes have **opposite cortical-state-dependent firing**: in awake/cortical-desync states PV+ fires, in slow-wave/cortical-sync states PV− fires (Mallet et al. 2008, postdating Kita 2007 but consistent with the morphological/intrinsic split documented here).
- **Sim status:** **missing.** `g11_bg_runner` GPe regions are single uniform pools driven by D2 MSNs; no PV+/PV− split. Adding two GPe subpools per action with different downstream targets (PV+_X → STN/GPi_X; PV−_X → str_fsi_X — the latter requires a striatal FSI pool, currently also missing per B.04 augmentation) would restore the bidirectional cortex–striatum–GPe loop.
- **Cluster:** A primary; B secondary.
- **Prerequisites:** A.10 (pallidostriatal feedback), B.06 (FSI pool). Possibly new HH preset for arkypallidal cell with prominent Ih + rebound Ca²⁺.
- **Citation:** PBR-160 ch 7 (Kita) pp 111–114, Fig 2; ch 6 p 92 (Wilson summary). Kita & Kitai 1994 J. Comp. Neurol. 351:519–533; Mallet et al. 2008 J. Neurosci. 28:4795 (later confirmation).
- **Behavioral validation:** Optogenetic activation of PV+ GPe → motor inhibition; activation of PV− GPe → MSN disinhibition. Asymmetric responses to dopamine depletion match Parkinson's clinical pattern.

### A.14 SNr + GPi receive perisomatic GPe inhibition vs distal striatal inhibition — input compartmentalization
- **System:** Synaptic terminals on individual GPi/SNr neurons are systematically segregated by anatomical origin: **GPe GABAergic boutons cluster perisomatically + on proximal dendrites** (~15% of synapses but high local potency); **striatal D1-MSN GABAergic boutons populate distal dendrites** (~70% of synapses, low per-synapse potency); **STN glutamatergic boutons distribute uniformly across soma and dendrites** (~10%) (PBR-160 ch 8 Nambu Fig 2 p 138; ch 9 Deniau et al. p 159). GPe boutons are larger, contain pleomorphic vesicles + multiple mitochondria; striatal boutons are smaller with fewer mitochondria.
- **Biological role:** GPe inhibition has *much stronger somatic veto* per synapse than striatal inhibition because of perisomatic targeting — a single GPe bouton can reset GPi spike timing (PBR-160 ch 7 Fig 5; ch 8 p 137–139). The functional consequence is that the GPi/SNr pacemaker (40–80 Hz tonic, A.04 supplemental) is preferentially gated *first* by GPe input, *then* modulated by distal striatal input. The classical "direct pathway opens the gate by inhibiting GPi/SNr" picture is anatomically incomplete — a stronger and faster gating arrives via cortex → STN → GPe → GPi (Nambu's three-phase response sequence in PBR-160 ch 7 Fig 6: early STN excitation → 50–100 ms striatal inhibition → late STN-mediated late excitation).
- **Sim status:** **partial — direct path implemented, GPe→GPi/SNr missing.** `g11_bg_runner` per-action `str_d1_X → gpi_X → thal_X → motor_X` lacks the parallel `gpe_X → gpi_X` projection that should have *higher unitary weight* than the direct striatonigral path. Adding `gpe_X → gpi_X` with `weight_mean` ~3× the str_d1→gpi value would replicate the perisomatic/distal asymmetry under the single-compartment approximation. Single-compartment can't fully reproduce the spatial effect, but per-pathway weight scaling is an adequate functional surrogate.
- **Cluster:** A primary; B secondary.
- **Prerequisites:** A.02 (GPe), A.04 (BG output).
- **Citation:** PBR-160 ch 8 (Nambu) pp 137–139, Fig 2; ch 9 (Deniau et al.) p 159; Smith et al. 1994; Shink & Smith 1995.
- **Behavioral validation:** Brief GPe stim → ~3 ms GPi IPSP that can reset spike phase (ch 7 Fig 5); striatal stim → 8–15 ms IPSP with weaker spike-resetting power.

### A.15 GABA-A subunit composition is region-specific in BG — α1β2γ2 dominant, α2/α3 in striatum
- **System:** PBR-160 ch 13 (Boyes & Bolam) Tables 1–2 pp 232–233 enumerate GABA_A receptor subunit expression across all BG nuclei in rat (in-situ hybridization + immunohistochemistry):
  - **Striatum (MSNs):** primarily **α2 + α3** (medium spiny neurons specifically); β2/β3 + γ2 throughout. PV+ FSI and CR interneurons express **α1**. Cholinergic interneurons express **α3**.
  - **GPe + GPi/EP:** strongly **α1 + β2/β3 + γ2** (the "BZ-I subtype" — high-affinity for benzodiazepines). Very little α2, β1, or γ1.
  - **STN:** **α1β2/β3γ2** dominates. (Same as the most common combination in mammalian brain.)
  - **SNr:** strong **α1β2/β3γ2**, similar to GP.
  - **SNc DA neurons:** **α3 dominates**, with α4 and γ3 also present — *no α1*. This is the only BG nucleus where α1 is *not* the dominant α subunit.
  - **GABA_A receptors are concentrated at synaptic specializations** with ~220–440× enrichment over extrasynaptic sites (Fujiyama 2000, ch 13 p 233).
- **Biological role:** Subunit composition controls IPSC kinetics (α1 → fastest decay, ~5 ms; α2/α3 → slower, ~15–30 ms) and benzodiazepine pharmacology. The MSN α2/α3 → slower decay pattern means striatal IPSCs have ~3× longer duration than pallidal IPSCs at otherwise identical conductance — a substantial effect for STDP timing windows. SNc α3 + α4 + γ3 (without α1) means SNc DA neurons have distinct sensitivity to GABA_A modulators (clinically relevant — e.g., benzodiazepines have weaker effect on DA cells).
- **Sim status:** **missing — uniform GABA_A kinetics across all regions.** `CoreSimConfig` has a single `gaba_a_decay_tau` field; per-region overrides are not currently exposed. Adding per-region `gaba_a_decay_tau_per_region` (MSN ~15 ms, GPe/GPi/SNr ~5 ms, SNc ~20 ms) would substantially change phase relationships in the cascade — particularly STDP timing in cortex→MSN learning, since the slower MSN IPSC widens the post-pre window.
- **Cluster:** A primary; B, J (plasticity) secondary.
- **Prerequisites:** none (small config extension).
- **Citation:** PBR-160 ch 13 (Boyes & Bolam) pp 232–235, Tables 1–2; Pirker 2000; Fritschy & Möhler 1995; Schwarzer 2001; Waldvogel 1999.
- **Behavioral validation:** α1-selective agonist zolpidem differentially modulates GP/SNr (strong) vs MSN (weak) firing; consistent with subunit map.

### A.16 STN intrinsic biophysics — pacemaker + Cav3 short rebound + Cav1.2/1.3 long plateau
- **System:** STN projection neurons fire autonomously at 5–15 Hz in slice without synaptic input (PBR-160 ch 10 Bevan et al. pp 175–176). The pacemaker depends on persistent + resurgent Na⁺ currents. Spike precision is set by **SK channels** (small-conductance Ca²⁺-activated K⁺) coupled to **Cav2.2** (ω-conotoxin-GIVA-sensitive). On hyperpolarization to ~−80 mV, STN neurons exhibit **two distinct rebound modes**:
  - **~75% of STN cells:** short rebound burst (<100 ms) driven by **Cav3 (T-type) low-threshold Ca²⁺ spike** with riding Na⁺ APs.
  - **~25% of STN cells:** long rebound burst (several hundred ms) driven by additional **Cav1.2/1.3 (L-type) plateau potential** — dihydropyridine-sensitive.
  - Both classes have a **hyperpolarization sag** mediated by **Ih (HCN)** that restores autonomous activity after sustained inhibition.
- **Biological role:** Combined with GPe→STN GABAergic input (perisomatic, fast — see Bevan 1998), the STN architecture creates a **rebound-burst gate**: inhibitory input from GPe doesn't simply silence STN — by deinactivating Cav3 + Cav1.2/1.3, GPe input *transiently amplifies* STN response to subsequent excitatory cortical inputs (PBR-160 ch 10 abstract p 173 and pp 176–178). This is a non-trivial paradoxical mechanism that the canonical D2→GPe→STN→GPi indirect pathway model omits.
- **Sim status:** **missing — only generic AdEx/Izh STN preset.** The simulator has `IZH2007_STN_BURST` (parameter set tuned for fast bursts) but does not implement the dual Cav3 + Cav1.2/1.3 + Ih mechanism. Adding the existing `fused_hh_h_current_update`, `fused_hh_NaP_current_update`, and a new `fused_hh_caT_current_update` (T-type Ca²⁺) for an HH-STN preset would produce post-inhibitory rebound bursts. Functionally this changes the indirect pathway from a pure suppressive route to a *delayed-amplification* route — phase-2 readaptation behavior (currently the Achilles heel of cross-projection variants per cheat-5 attempts) might benefit from STN rebound dynamics that reset cleanly after goal change.
- **Cluster:** A primary; I (intrinsic neuron model) secondary.
- **Prerequisites:** I.* extension for T-type Ca²⁺.
- **Citation:** PBR-160 ch 10 (Bevan et al.) pp 175–177, Fig 3; Beurrier et al. 1999; Hallworth et al. 2003; Bevan et al. 2002; Otsuka et al. 2001.
- **Behavioral validation:** GPe inactivation → STN spike rate falls but rebound bursts emerge; cortical pulse during/after GPe burst → STN response amplified vs same pulse without preceding GPe input.

---

## Cluster B — Striatal microcircuit & WTA

### B.08 Striatal LTS interneuron — NPY/SOM/NOS, beta resonance, slow inhibition
- **System:** ~0.55–0.8% of striatal neurons (Rymar et al. 2004); coexpress somatostatin (SOM), NPY, and nNOS. Medium soma (~15 µm), 3–5 aspiny dendrites, very long sparsely branching axon extending up to 1 mm with infrequent bouquet-like terminations. Originally called PLTS but the plateau potential turned out to be a whole-cell artifact; renamed LTS in 2018 (Tepper-2018 pp 8–9).
- **Biological role:** Spontaneously active in vitro (~91% of cells), low firing rate. Beta-band (10–20 Hz) intrinsic spiking and membrane resonance — distinct from the gamma-band FSI (Beatty et al. 2015). LTS→MSN connection probability is low in blind paired recordings (~3%) but ~14% within axonal field; weak, conventional fast GABA_A IPSCs. Receives strong cortical input but minimal thalamic (parafascicular) input — opposite of FSI/NGF/CIN/THIN. Excited by D1/D5 dopamine and nicotinic ACh; releases NO, NPY, GABA, and SOM into the striatum (volume neuromodulation in addition to point-to-point synapses). Inhibited by THINs (Tepper-2018 p 10).
- **Sim status:** **missing** — no LTS-equivalent population. Could be added as a single sparse pool projecting weakly to all MSNs with beta resonance for slow rhythmic gating.
- **Cluster:** B primary; C (DA) secondary.
- **Prerequisites:** I.* (interneuron model with LTS — needs T-current Ca²⁺ + plateau machinery beyond current AdEx/IZH).
- **Citation:** TK-2017 pp 164–168 §IV; Tepper-2018 pp 8–9, Table 1; Kawaguchi 1993; Ibáñez-Sandoval et al. 2011.
- **Behavioral validation:** Beta-rhythmic LTS spiking phase-locks to cortical beta; selective ablation alters slow modulation of MSN firing without affecting fast gamma timing (FSI domain).

### B.09 Striatal NPY-neurogliaform (NGF) interneuron — GABA_A-slow inhibition
- **System:** ~25% of striatal NPY-expressing interneurons (revealed only via NPY-GFP transgenic mice; Ibáñez-Sandoval et al. 2011). 5–9 short branched aspiny dendrites forming dense compact field <200 µm, dense axon >400 µm. Hyperpolarized RMP (~−85 mV), low input R (~140 MΩ), no spontaneous activity, no LTS or plateau. Electrotonically coupled to other NGFs via gap junctions, AND heterosynaptically coupled to FAIs and THINs (Tepper-2018 p 3).
- **Biological role:** **Mediates GABA_A-slow inhibition** — IPSC rise time ~10 ms, decay τ ~120 ms (≈10× slower than conventional fast GABA_A) (Ibáñez-Sandoval et al. 2011; English et al. 2012). NGF→MSN connection probability ≥85% (very high). Driven by parafascicular thalamic input (suprathreshold) and by Type-II nicotinic input from ChIs (English et al. 2012) — i.e., the substrate for the disynaptic ChI→NGF→MSN inhibition that follows ChI rebound bursts. Adds a slow blanket of GABA_A-mediated inhibition that complements fast FSI-mediated inhibition. Unlike LTS, NGFs respond mostly subthreshold to cortex, supra-threshold to thalamus — a perfect "thalamic salience gate".
- **Sim status:** **missing** — no GABA_A-slow inhibition modelled, no NGF-equivalent. Adding NGFs would let the simulator capture ChI-rebound salience inhibition (a candidate mechanism for behavioral pause/redirect).
- **Cluster:** B primary; A, C secondary.
- **Prerequisites:** new GABA_A-slow synapse type (decay ~120 ms vs current ~5 ms); new "parafascicular thalamus" region.
- **Citation:** TK-2017 pp 167–168 §V; Tepper-2018 pp 2–4; Ibáñez-Sandoval et al. 2011 J. Neurosci. 31:16757–16769; English et al. 2012 Nat. Neurosci. 15:123–130.
- **Behavioral validation:** Optogenetic activation of striatal ChIs evokes prolonged (>200 ms) IPSC barrage in MSNs, blocked by GABA_A antagonist + Type II nicotinic antagonist; abolished by NGF-selective ablation.

### B.10 Striatal TH+ interneuron (THIN) — non-dopaminergic GABAergic
- **System:** Tyrosine-hydroxylase-expressing striatal interneurons identified via TH-EGFP / TH-Cre transgenic mice. Four electrophysiologically distinct subtypes: Type I (60–80%, strong spike-frequency adaptation → complete spike failure ~100 ms in to depolarization, spontaneously active ~5 Hz), Type II + III (rare, FSI-like), Type IV (~20%, LTS-like). Medium-sized soma, dense local axon. Despite expressing TH (the rate-limiting enzyme for DA synthesis), THINs do NOT express VMAT2 or DAT and do NOT release DA — they are GABAergic interneurons (Xenias et al. 2015).
- **Biological role:** Receive monosynaptic glutamatergic input from cortex and from parafascicular thalamus; receive DA and ACh modulation (D1/D5 → plateau potentials via TRPM2/I_CAN current). THIN→MSN GABA_A IPSCs (~15% blind connection probability, 100% of MSNs respond to optogenetic ensemble activation → upper bound 3:1 convergence). Uniquely among striatal interneurons, **MSNs synapse back onto Type I THINs** (6/18 pairs in Ibáñez-Sandoval et al. 2010) — a feedback loop never observed for FSIs. THINs also inhibit LTS interneurons (and thereby gate slow inhibition) and contact CINs (Tepper-2018 p 10). Numbers transiently increase ~30% after 6-OHDA DA depletion (compensatory hypothesis).
- **Sim status:** **missing** — no TH-IN equivalent. The compensatory "DA-depletion driven THIN upregulation" is a candidate Parkinson model (cluster P) once added.
- **Cluster:** B primary; C, P secondary.
- **Prerequisites:** I.* (interneuron model with TRPM2 / I_CAN — currently missing); reciprocal MSN→IN synapse (currently absent in the bridge for any interneuron class).
- **Citation:** TK-2017 pp 168–171 §VI; Tepper-2018 pp 9–11; Ibáñez-Sandoval et al. 2010 J. Neurosci. 30:6999–7016; Xenias et al. 2015 J. Neurosci. 35:6584–6599.
- **Behavioral validation:** Optogenetic THIN activation produces inhibition in 100% of MSNs tested, blocking depolarization-evoked spikes; THIN-selective ablation increases MSN firing variability.

### B.11 Striatal fast-adapting interneuron (FAI) — facilitating MSN inhibition
- **System:** Htr3a-Cre-targeted GABAergic interneuron, ~7% of Htr3a+ population. Medium-sized, 3–5 aspiny varicose dendrites, axonal field overlapping dendrites. Depolarized RMP (~−66 mV), high input resistance (~362 MΩ), pronounced spike-frequency adaptation (gives the cell its name) but no depolarization block, no spontaneous activity. Electrotonically coupled to NGFs (Tepper-2018 p 3).
- **Biological role:** Receives powerful suprathreshold nicotinic input from striatal ChIs (sometimes mecamylamine-sensitive Type III, sometimes DHβE-sensitive Type II — pharmacologically heterogeneous). FAI→MSN connection probability ~50%. **Uniquely among striatal GABAergic synapses, the FAI→MSN IPSC exhibits short-term FACILITATION** (~2× growth from 1st to 3rd spike at 50 Hz; sometimes 100% failure on 1st spike, large IPSC on 3rd) — every other characterized striatal interneuron synapse depresses (TK-2017 pp 171, "in contrast to all other inhibitory GABAergic synapses in striatum previously observed that display short-term depression"). Likely targets distal MSN dendrites.
- **Sim status:** **missing**. The facilitating IPSC is mechanistically distinct from current STP machinery (Tsodyks-Markram with depression default); requires per-pathway STP override or facilitating-only `stp_U_per_type` configuration.
- **Cluster:** B primary; C (ACh) secondary.
- **Prerequisites:** STP framework with per-connection-type facilitation (already present via `stp_U_per_type`, `stp_tau_f_per_type`).
- **Citation:** TK-2017 pp 171, Fig 8.5; Tepper-2018 pp 4–5, Fig 3; Faust et al. 2015 Eur. J. Neurosci. 42:1764–1774.
- **Behavioral validation:** Train of 3–5 ChI spikes → progressively larger MSN IPSCs blocked by bicuculline; selective FAI silencing reduces ChI-evoked MSN inhibition.

### B.12 Striatal spontaneously active bursty interneuron (SABI) — interneuron-selective IN
- **System:** Htr3a-Cre-targeted GABAergic interneuron distinct from FAI (Assous et al. 2018). Medium soma, sparse axonal arborization (mostly local + occasional extended sparse axons). High input resistance (>600 MΩ), depolarized RMP (~−50 mV), spontaneously active in highly irregular long bursts (25–125 spikes at 100–300 Hz) separated by long silent periods. Driven by Type-III nicotinic ACh input.
- **Biological role:** **First demonstrated interneuron-selective interneuron in the striatum** — paired recordings show only ~4% connection probability with MSNs (vs ~50–86% for FSI / NGF / FAI). Optogenetic *inhibition* of SABIs evokes large IPSC barrages in MSNs, demonstrating SABI normally inhibits another (still unidentified) GABAergic interneuron population that itself inhibits MSNs — a disinhibitory motif. Suggests a hierarchical interneuron-to-interneuron control layer in striatum (Tepper-2018 pp 6–7, Fig 4G).
- **Sim status:** **missing**. Adding SABIs would let the project test "disinhibitory release of MSN ensembles" as an alternative WTA mechanism — orthogonal to v3 lateral inhibition and FSI feedforward inhibition.
- **Cluster:** B primary.
- **Prerequisites:** at least one downstream IN class (NGF, LTS, or THIN) for SABI to inhibit.
- **Citation:** Tepper-2018 pp 5–7, Fig 4; Assous et al. 2018 J. Neurosci. 38:5688–5699.
- **Behavioral validation:** Halorhodopsin silencing of Htr3a-Cre+ neurons → IPSC barrages in 70%+ of recorded MSNs.

### B.13 Calretinin (CR) interneurons — multiple subtypes, primate-dominant
- **System:** ~0.8% of striatal neurons in rodent (Rymar et al. 2004); medium soma, smooth aspiny dendrites. **In primates and humans, CR+ neurons are 3–4× more numerous than PV+ or NPY+ interneurons** (Wu & Parent 2000) and split into morphologically distinct subtypes. In rodent, three subtypes have been distinguished by CR + secretagogin (Scgn) + Sp8 + Lhx7 combinations (Garas et al. 2017): Type I small monopolar spiny (rostro-dorsal), Type II + III medium multipolar aspiny (mid-caudal). Human striatum has a fourth large CR+ class that co-expresses ChAT (Petryszyn et al. 2016).
- **Biological role:** Almost entirely uncharacterized electrophysiologically — first in vivo recordings (Garas et al. 2017) show variable firing during cortical slow waves, tonic activity during cortical desynchronization. At least Type I CR evokes GABA_A IPSCs on MSNs. The primate-specific dominance + the human-specific cholinergic CR class suggest CR neurons may carry primate-/human-specific computational roles not present in rodent BG. **[discrepancy: rodent-derived BG models — including this simulator — likely miss a major primate interneuron class that, in humans, outnumbers PV+ FSIs and NPY/SOM cells combined].**
- **Sim status:** **missing**. Low priority for rodent-task simulations; high priority if scaling to primate / human striatal ratios is ever a goal.
- **Cluster:** B primary.
- **Prerequisites:** none unique.
- **Citation:** TK-2017 pp 171–173 §VIII; Tepper-2018 pp 11; Bennett & Bolam 1993; Wu & Parent 2000; Garas et al. 2017 J. Comp. Neurol. 526:877–898.
- **Behavioral validation:** species-comparison data; lacking single-cell physiology in primate.

### B.14 MSN GABA_A reversal is depolarized — shunting inhibition + KCC2 dependence
- **System:** Direct measurement of GABA_A reversal in striatal MSNs (gramicidin perforated patch, preserving native [Cl⁻]ᵢ): **E_GABA = −60 mV in mature rat MSN** (PBR-160 ch 6 Wilson p 104, citing Kööset al. 2004 + Gustafson et al. 2005). This is *positive* to RMP (−85 mV) and *negative* to threshold (−50 mV). MSN GABA_A IPSPs are therefore **depolarizing at rest** (at RMP, GABA opens Cl⁻ channels, Cl⁻ flows out, membrane depolarizes by ~10–15 mV) and **hyperpolarizing-by-shunt near threshold**. Critically, E_GABA aligns with the *peak of MSN dendritic input resistance* (the −60 mV minimum-K⁺-current voltage) so GABA_A IPSPs in dendrites preferentially *linearize* the I-V curve and shrink the electrotonic length constant (PBR-160 ch 6 Fig 8).
- **Biological role:** The seemingly anomalous depolarizing GABA in MSN was historically interpreted as immature (developmental) or pathological. Wilson 2007 establishes it as a **functional design feature**: dendritic Sp-Sp inhibition (a) *escapes* the KIR2 hyperpolarized voltage trap by depolarizing toward −60 mV, (b) *removes* the Up/Down nonlinearity locally, (c) *shunts* the dendrite preventing AP back-propagation, and (d) *shortens* the dendritic length constant so distal inputs become more effective. This is fundamentally different from the cortical model where GABA_A is purely hyperpolarizing.
- **Sim status:** **missing — `cfg.E_inh = -75 mV` global default.** The simulator's `CoreSimConfig` uses a uniform inhibitory reversal of −75 mV across all regions; striatal MSNs need −60 mV, cortical pyramids need −75 mV, SNc DA needs ~−55 mV (next entry B.15). One-line per-region override `E_inh_per_region` would correctly model MSN shunting inhibition. **This is a small change with large downstream effects:** STDP windows depend on actual postsynaptic voltage trajectory, which is qualitatively different under shunting vs hyperpolarizing inhibition.
- **Cluster:** B primary; A, J (plasticity) secondary.
- **Prerequisites:** none (config extension).
- **Citation:** PBR-160 ch 6 (Wilson) pp 104–106, Figs 6, 8; Köös et al. 2004 J. Neurosci. 24:7916; Gustafson et al. 2005 J. Neurophysiol. 95:737; Misgeld et al. 1982; Bracci & Panzeri 2006.
- **Behavioral validation:** MSN whole-cell IPSC reversal at −60 mV; GABA_A application produces depolarizing IPSP at RMP that switches to hyperpolarizing at threshold.

### B.15 SNc DA neurons lack KCC2 — depolarized E_Cl, GABA disinhibition cascade
- **System:** SNc dopaminergic neurons **do not express KCC2**, the K⁺/Cl⁻ co-transporter that sets a low [Cl⁻]ᵢ in mature CNS neurons (Gulácsi et al. 2003; PBR-160 ch 11 Tepper & Lee p 199). As a consequence, ECl in SNc DA neurons is significantly more depolarized than in SNr GABA neurons or in MSNs — closer to ~−55 mV, near AP threshold.
- **Biological role:** GABA_A IPSPs in DA neurons are *only weakly hyperpolarizing* and frequently *shunting near threshold*. The canonical "striatum inhibits SNr/GPi which disinhibits thalamus" picture works because SNr cells *do* express KCC2 (E_Cl ~−75 mV) and are strongly inhibited by GABA. But **DA cells with depolarized E_Cl are remarkably resistant to direct striatal/pallidal GABA inhibition**. Instead, the dominant route by which BG modulates DA firing is **disynaptic disinhibition via SNr→SNc axon collaterals**: striatum inhibits SNr GABA cells → SNr collateral release on DA cells decreases → DA cells *burst*. PBR-160 ch 11 pp 192–195 documents this through GPe lesion / GABA_A blockade experiments: blocking SNr GABA_A → DA cells switch to burst mode regardless of starting pattern. This is the principal mechanism for **phasic DA bursts during cue-evoked reward signals** (overlaid on glutamate from PPN/STN that triggers individual bursts).
- **Sim status:** **missing — DA region in `g11_bg_runner` has no SNr→SNc collateral disinhibition pathway.** Currently DA in the cascade is driven by reward signal injection. Adding an explicit `snr_X → snc` GABAergic projection that, when SNr is inhibited by D1 MSNs, releases SNc → produces DA burst, would create *biologically grounded* phasic DA from action selection — without manual reward-signal injection. This composes with the plastic-input-layer arc: cortex → str_d1_X → snr_X (inhibition) → snc (disinhibition) → DA burst → eligibility trace converted to weight change *exactly when the action wins*, automatically.
- **Cluster:** B primary (mechanism); A, C (DA) secondary.
- **Prerequisites:** per-region E_inh override (B.14 prerequisite); SNr→SNc projection in `build_bg_brain_regions`.
- **Citation:** PBR-160 ch 11 (Tepper & Lee) pp 192–195, 199; Gulácsi et al. 2003 J. Neurosci. 23:8237; Tepper et al. 1995 J. Neurosci. 15:3092; Paladini et al. 1999.
- **Behavioral validation:** Local SNr GABA_A blockade in vivo → DA cells switch to bursting; salient-cue → striatal D1 burst → SNr pause → DA burst (the reward prediction error pathway).

### B.16 Striatonigral conduction is the slowest in BG — 1.4 m/s vs 4 m/s pallidonigral
- **System:** PBR-160 ch 11 (Tepper & Lee) pp 191–192 quantifies conduction velocities of the major BG GABAergic projections measured by antidromic activation:
  - **Striatonigral (D1 MSN → SNr/GPi):** ~1.4 m/s, antidromic latency ~10 ms (Ryan et al. 1986). Slowest in the BG.
  - **Pallidonigral (GPe → SNr/SNc):** ~4 m/s, latency ~1 ms. ~3× faster.
  - **Nigronigral collaterals (SNr → SNc):** ~3–4 m/s, latency ~1–2 ms.
  - **Striatum spontaneous firing rate:** <1 Hz (very low); striatonigral neurons fire only during cortical Up states.
  - **GPe spontaneous rate:** ~50 Hz tonic (PV+ subtype).
  - **SNr spontaneous rate:** ~30 Hz tonic.
- **Biological role:** The 10-ms striatonigral latency vs 1-ms pallidonigral latency means **GPe input to SNr/GPi arrives ~9 ms before striatal input** during a cortical pulse. Combined with the perisomatic targeting of GPe (vs distal targeting of striatal — A.14), GPe gating actually dominates *first*; striatal D1 input arrives later but on already-modulated SNr neurons. The classical Albin/DeLong direct/indirect-pathway model treats both as instantaneous; the in-vivo three-phase response to cortical stimulation (early excitation → inhibition → late excitation; PBR-160 ch 7 Fig 6) is the actual time-course.
- **Sim status:** **missing — no axonal conduction delays on cross-region projections.** The simulator's `RegionPathway` has no `conduction_delay_ms` parameter; all pathway transmission is one-step. Adding a per-pathway delay (1 ms for pallidonigral, 10 ms for striatonigral, etc.) would let the cascade reproduce the in-vivo three-phase response.
- **Cluster:** B primary; A secondary.
- **Prerequisites:** Engine-level support for axonal delays on `RegionPathway`.
- **Citation:** PBR-160 ch 11 (Tepper & Lee) pp 191–192; Ryan et al. 1986; Kita & Kitai 1991; Celada et al. 1999; Deniau et al. 1978.
- **Behavioral validation:** Cortical electrical pulse → in-vivo SNr response: 4–6 ms early excitation (via STN), 10–20 ms inhibition (striatal D1), 30–60 ms late excitation (indirect path via STN); the simulator currently produces a single combined response with no temporal structure.

### B.17 Sp-Sp dendritic inhibition removes Up/Down nonlinearity — voltage-dependent dendritic linearization
- **System:** Wilson's central insight (PBR-160 ch 6 pp 96, 105–107) is that the principal *functional* role of MSN-MSN recurrent collateral inhibition is **not WTA** (B.04 confirms this is weak) but **dendritic linearization**. The mechanism (Fig 7, 8):
  - Sp-Sp synapses are made on the *spiny region* of dendrites (40% on shafts, 48% on spine necks; only 12% on soma) — exactly the locations where KIR2 and Kv-2 K⁺ currents make the dendrite electrotonically nonlinear.
  - Each individual Sp-Sp IPSP is small at the soma (<0.5 mV) but **~10 mV at the synapse** (Fig 7).
  - The reversal potential of Sp-Sp inhibition (~−60 mV; B.14) sits exactly at the *peak* of MSN dendritic input resistance (the voltage where K⁺ currents are minimized).
  - Therefore Sp-Sp synaptic conductance, when turned on, *exactly cancels* the voltage-dependent K⁺ contribution, **flattening the I-V curve and removing the Up/Down state nonlinearity** (Fig 8).
  - Quantitatively: ~9 quanta distributed across one 220-µm dendrite (~20% of the local Sp-Sp synapses active simultaneously) is enough to make the entire dendrite electrotonically compact (<2 length constants over its whole voltage range).
- **Biological role:** This is a **second style of synaptic integration** in striatum, complementary to the canonical sparse phasic mode. When few MSNs are active → Sp-Sp inhibition is sparse → dendrites are nonlinear → MSN integrates sparsely + thresholds via Up/Down → discrete activity episodes. When many MSNs are active → Sp-Sp inhibition saturates → dendrites linearize → MSN integrates continuously → graded firing of the entire population. Two regimes from one circuit. Wilson concludes (p 107): "the striatal network may possess two different styles of synaptic integration; one characterized by very sparse activity ... and one in which many neurons are engaged, and firing of individual neurons is continuously distributed over a wide range."
- **Sim status:** **missing — single-compartment MSN cannot exhibit dendritic linearization.** The simulator's `IZH2007_STRIATAL_MSN_D1/D2` presets are point neurons with no dendrite. Capturing the regime-switching behavior would require either (a) explicit multi-compartment MSN with KIR2 + Kv2 + dendritic GABA_A, or (b) a phenomenological switch that increases effective gain when total MSN activity exceeds a threshold (cheaper but less mechanistic). This is one of the more interesting *predictions* of the Wilson 2007 chapter that the simulator could test.
- **Cluster:** B primary; A secondary; I (multi-compartment).
- **Prerequisites:** multi-compartment MSN model (currently absent); KIR2 + Kv-2 fused kernels.
- **Citation:** PBR-160 ch 6 (Wilson) pp 96, 100–107, Figs 7, 8; Plenz 2003 Trends Neurosci 26:436; Bracci & Panzeri 2006 J. Neurophysiol. 95:1285.
- **Behavioral validation:** Population recording of MSNs during sparse vs dense behavioral epochs; sparse-mode MSNs show bimodal Up/Down distributions, dense-mode MSNs show continuous voltage distributions. Pharmacological MSN-MSN GABA_A blockade should *increase* the bimodality even in dense regimes.

## Cluster B — additions

---

## Cluster B — Striatal microcircuit & WTA

**7 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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
- **Supplemental — IMPORTANT CORRECTION:** the current entry uses *cortical* interneuron taxonomy (basket / chandelier / Martinotti / neurogliaform) and labels it "cortex (and analogous diversity in hippocampus, striatum)". This is **incorrect for striatum**. The striatum has its own non-isomorphic GABAergic interneuron taxonomy (TK-2017 ch. 8; Tepper-2018 §"Functional Significance"): no chandelier-equivalent has been identified, no Martinotti-equivalent, and the striatal "neurogliaform" is a recently characterized NPY-expressing class (NPY-NGF) that mediates GABA_A-slow currents distinct from cortical NGF (Tepper-2018 pp 2–4). The correct striatal taxonomy as of 2018 lists at least **eight distinct GABAergic interneuron classes**: PV-FSI, NPY-LTS (formerly P-LTS), NPY-NGF, CR, TH (THIN, four subtypes), FAI (fast-adapting), SABI (spontaneously-active-bursty), plus the cholinergic ChI/TAN and putative recurrent / disinhibitory IN classes (Tepper-2018 Table 1, pp 11–12). This warrants splitting B.01 into "cortical interneuron diversity" and a separate "striatal interneuron diversity" entry, with the latter pointing to new B entries B.08–B.13 below. **[discrepancy: textbook applies cortical interneuron categories to striatum; specialty literature treats the two as separate, non-isomorphic taxonomies].** (TK-2017 pp 157–158, 174; Tepper-2018 pp 1–2, 11–12.)

## Cluster M — Neuromuscular junction

### B.02 Medium spiny neuron (MSN) — >90% of striatum, GABAergic projection

*[from Part V — Movement (Ch 30-39); renumbered from B.50]*

- **System:** striatum; spiny dendritic morphology; bistable membrane (down-state ~−85 mV, up-state ~−55 mV).
- **Biological role:** Sole projection neuron of striatum. Silent at rest; requires substantial coordinated cortical / thalamic input to reach up-state and fire. Provides a thresholding effect — only strong, consensual inputs drive output.
- **Sim status:** **partial** — `IZH2007_STRIATAL_MSN_D1` and `_D2` presets exist and are used by `g11_bg_runner`. Bistability not explicitly modeled but Izhikevich-2007 captures rest-near-threshold sparsity.
- **Cluster:** B primary; A secondary.
- **Prerequisites:** I.*, J.*.
- **Citation:** Kandel 6e Ch 38 p 933–938.
- **Behavioral validation:** MSN silent at rest; cortex-stim → up-state → spike threshold met only at high coordinated drive.
- **Supplemental:** MSN dendritic spines have a stereotyped two-input architecture: corticostriatal terminals form *asymmetric* synapses on the spine *head* (AMPA + NMDA, AMPA GluR2/3 and NMDA NR1 colocalized within the synaptic specialization — Bolam-2000 Fig 2B,C, p 530), while DA terminals from SNc form *symmetric* synapses on the spine *neck* with D1 or D2 receptors localized both at the synapse and extrasynaptically (Bolam-2000 Fig 2D–F, pp 530–531). GABAergic terminals are also found on spine necks. Each MSN spine is thus a gated three-input integration unit (cortex × DA × GABA). The current single-compartment MSN cannot reproduce per-spine modulation; a future multi-compartment MSN would expose per-spine DA gating as an experimental knob.
- **Supplemental — quantitative synaptic budget (PBR-160 ch 6 Wilson, pp 92–95):** an individual MSN receives:
  - **~10,000 asymmetric (glutamatergic) synapses** — ~95% on dendritic spine heads, mostly cortical + intralaminar thalamic (parafascicular thalamic input is the exception — terminates on dendritic *shafts*, not spines).
  - **~2,500 symmetric (mostly GABAergic) synapses**, of which ~325 are dopaminergic (Roberts 2002) and ~300 cholinergic; the remaining **~1,875 are GABAergic** distributed across FSI, LTS/SOM/NOS, and Sp-Sp recurrent collaterals.
  - Of those GABAergic synapses, the **MSN-MSN (recurrent) synapses are quantitatively dominant**: each MSN receives ~1,200 Sp-Sp synapses arising from ~400 different nearby Sp cells (i.e., each upstream MSN contributes ~3 synapses on average — small bouton count is why each is functionally weak). FSI→MSN synapses are far fewer (~6 per FSI on the target MSN soma, ~2–4 FSIs converging on each MSN — total ~12–24 FSI synapses) but individually 6× stronger.
  - Excitation/inhibition ratio measured during the *Up state* in vivo is **~2–5× excitation-dominant** (reversal potential of Up state is between −10 and −20 mV vs ECl at ~−60 mV; Wilson & Kawaguchi 1996, ch 6 Fig 5). This contrasts with cortical pyramidal cells where Up-state excitation/inhibition is roughly equal. **Implication:** striatal Up states are excitation-driven (cortical+thalamic synchrony lifts the cell over KIR2's voltage barrier); inhibition in striatum is *not* the gate for entering Up state, it is the modulator of dendritic electrotonic structure and AP timing.
- **Supplemental — KIR2 + Up/Down state mechanism (PBR-160 ch 6 Wilson pp 100–104):** MSN bistability is generated by *two voltage-dependent K⁺ currents*, not by recurrent inhibition: (1) **KIR2 (Kv-inwardly-rectifying)** active at hyperpolarized potentials clamps RMP to −80 to −95 mV with input resistance 20–60 MΩ; (2) **Kv-1.2/Kv-2.1** subthreshold-activated K⁺ currents active above −60 mV. Both deactivate around −60 mV → **input resistance peaks ~6× higher (~150–300 MΩ) at −60 mV** (ch 6 Fig 6). Membrane time constant similarly peaks at −60 mV. KIR2 is *developmentally late* — not fully expressed until **postnatal day 25–28** in rat (ch 6 p 104, citing Tepper et al. 1998), so juvenile-slice MSNs and adult-MSN computational models without KIR2 cannot reproduce Up/Down bistability. **Sim implication:** the project's `IZH2007_STRIATAL_MSN` preset uses a 9-parameter Izh-2007 model that approximates the rest-near-threshold sparsity, but does *not* implement explicit KIR2 + Kv-2 voltage-dependent leak. Adding a KIR2-equivalent fused kernel would make MSN dendritic effectiveness voltage-dependent (the Wilson key insight: at −60 mV the dendrite is electrotonically compact, so inhibition acts everywhere; at hyperpolarized or threshold-near voltages, dendrites elongate to 2–4 length constants and only proximal synapses count).
- **Supplemental — GABA_A reversal potential is depolarizing-but-shunting (PBR-160 ch 6 pp 104–106, ch 11 pp 192–193):** measured GABA_A reversal in striatal MSNs (gramicidin perforated patch) is ~−60 mV — *positive* to RMP (−85 mV) but *negative* to spike threshold (~−50 mV). This means GABA_A IPSPs in MSNs are **depolarizing IPSPs** when measured from rest, and only become hyperpolarizing near AP threshold. The functional consequence: GABA inhibition in striatum is primarily *shunting* (conductance increase) rather than hyperpolarizing — and remarkably, the GABA reversal sits exactly at the **point of maximum dendritic input resistance** (ch 6 Fig 6/8), so inhibition selectively *removes the KIR2-induced nonlinearity* and linearizes the dendrite. **Sim implication:** the project's `E_inh = -75 mV` (CoreSimConfig default) is reasonable for cortical pyramids but **wrong for striatal MSNs** — striatal MSNs should use `E_inh ≈ -60 mV`. This is a one-line config override per region and would produce qualitatively different striatal dynamics (depolarizing IPSPs near rest, shunting near threshold). The same applies to dopaminergic SNc neurons (ch 11 p 199): DA neurons lack the **KCC2 chloride exporter**, so their ECl is even more depolarized than MSNs — closer to −55 mV — making GABA_A in DA neurons depolarizing or even excitatory at rest.

### B.03 D1 vs D2 MSN segregation — opposing DA modulation

*[from Part V — Movement (Ch 30-39); renumbered from B.51]*

- **System:** D1 MSNs (Gs-coupled, ↑cAMP, substance P / dynorphin) vs D2 MSNs (Gi-coupled, ↓cAMP, enkephalin).
- **Biological role:** ~equal proportions in dorsal striatum. DA increases D1 MSN excitability and decreases D2 MSN excitability. Drives the asymmetric Go/NoGo balance underlying A.50/A.51.
- **Sim status:** **implemented** — `g11_bg_runner` declares separate `str_d1_X` and `str_d2_X` pools per action with appropriate Izhikevich presets.
- **Cluster:** B primary; A, C secondary.
- **Prerequisites:** B.50, C.* (DA).
- **Citation:** Kandel 6e Ch 38 p 935–940 (Surmeier).
- **Behavioral validation:** DA application → D1 firing ↑, D2 firing ↓ in vitro; corresponding behavioral release vs suppression.
- **Supplemental:** Bolam-2000 (p 529) confirms the D1 (substance P / dynorphin / projects to GPi+SNr) vs D2 (enkephalin / projects to GPe) segregation. Kincaid 1998 and Bolam emphasize that corticostriatal terminals contact spines giving rise to both pathways (a single cortical axon's terminals can be presynaptic to D1 or D2 MSN spines — the two populations sample the *same* cortical pool, not different ones). This validates the simulator's per-action `cortex_X → str_d1_X / str_d2_X` arrangement where the same cortex pool projects to both pools.
- **Supplemental — neuropeptide co-transmission (PBR-160 ch 16 McGinty pp 273–280, Fig 1):** the D1/D2 dichotomy extends beyond DA receptor signaling to a complete neurochemical program:
  - **D1 MSNs:** preprodynorphin (PPD → dynorphin) + preprotachykinin (PPT → substance P, neurokinin A); project to GPi/SNr (direct pathway).
  - **D2 MSNs:** preproenkephalin (PPE → met/leu-enkephalin); project to GPe (indirect pathway).
  - Dynorphin → KORs on cortical Glu + nigrostriatal DA terminals → presynaptic *suppression* of Glu and DA (negative feedback against D1 activation). Substance P → NK-1 on cholinergic + somatostatin/GABA interneurons → *increases* ACh release.
  - Enkephalin → DORs on cholinergic interneurons → *decreases* ACh release. MORs (in striosomes/patches) → various.
  - **Functional asymmetry:** D1 activation triggers SP/dynorphin-mediated ACh-release *increase* + DA-release *suppression*; D2 activation triggers enkephalin-mediated ACh-release *decrease*. Net result: D1 vs D2 push the local cholinergic tone in opposite directions, on top of the canonical Go/NoGo GABA balance. **Sim implication:** if/when the simulator instantiates a TAN/ChI population (B.05 currently has presets but unused), declaring NK-1 (excitation from D1 SP) and DOR (inhibition from D2 enk) as neuromodulator targets on the ChI pool would create a biologically grounded D1/D2 → ChI dynamic without ad-hoc terms. Maps directly to Cluster C.08 (neuropeptides).

### B.04 MSN lateral inhibition — local GABA collaterals (cross-pool WTA)

*[from Part V — Movement (Ch 30-39); renumbered from B.52]*

- **System:** MSN axon collaterals form local GABAergic synapses on neighboring MSNs within striatum.
- **Biological role:** Implements competitive selection within striatum. Anatomically dense; functionally weaker per-synapse than feedforward but collectively shapes which MSN ensemble wins. Combined with same-action-only cortex routing, produces winner-take-all dynamics.
- **Sim status:** **implemented (functional equivalent)** — `--bg-lateral-inhibition` (v3, default since 2026-04-28) adds MSN cross-pool lateral inhibition between per-action D1 pools. 6-seed sum 4.26 ± 0.50, no regression. Closed cheat #5 by design — see `research/findings/2026-04-28-cheat5-v3-results.md`. **[discrepancy: real BG has dense cross-action collaterals AND cross-action cortex inputs; project keeps cortex same-action-only because cross-projections were NEGATIVE in v1, v2, v3.1, v4].**
- **Cluster:** B primary; A secondary.
- **Prerequisites:** B.50, B.51.
- **Citation:** Kandel 6e Ch 38 p 935 (Silberberg & Bolam 2015).
- **Behavioral validation:** Single MSN spike → neighboring MSN IPSP; competitive selection in pool stimulation tests.
- **Supplemental:** Specialty literature reports MSN-MSN lateral inhibition is functionally weak: paired recordings show low connection probability (~14–25% within axonal field), small unitary IPSPs (<0.5 mV), high failure rates, and short-term depression (TK-2017 pp 160–162 citing Czubayko & Plenz 2002, Tunstall et al. 2002, Koós et al. 2004, Tecuapetla et al. 2009). FSI→MSN feedforward IPSPs are *significantly larger and more reliable* than MSN→MSN feedback IPSPs under the same conditions, due to FSI-MSN synapses being more proximal and more numerous (TK-2017 p 163). **The implication for v3 lateral inhibition (`--bg-lateral-inhibition`): the dominant biological substrate of cross-pool WTA is feedforward FSI inhibition, NOT MSN collateral inhibition.** v3 currently encodes the WTA as direct MSN→MSN inhibition. This is a *functional* equivalent (closes cheat #5 by design) but is anatomically backwards. A future v3-bis: add a per-action FSI pool that is excited by cortex_X and inhibits str_d1_Y / str_d2_Y (Y ≠ X) — this would match the dominant WTA substrate in the literature. (TK-2017 pp 161–163; Tepper-2018 pp 8–9.)
- **Supplemental — Wilson 2007 quantitative dissection (PBR-160 ch 6 pp 96–103):** the entire chapter is essentially a re-evaluation of the WTA hypothesis for striatum. Key quantitative findings strengthening the prior B.04 correction:
  - **Connection probability** for MSN→MSN within an axonal field (~400 µm diameter): ~**16–17%** (Tunstall 2002, Taverna 2004, Venance 2004; ch 6 p 99).
  - **Reciprocal connections are at chance level** (1/36 in random distribution; observed 1/38 — Taverna 2004) — *no* preferred bidirectional coupling that would let competing groups form.
  - **Quantal analysis (Koós et al. 2004):** unitary FSI→MSN and MSN→MSN synapses have *identical* per-quantum conductance (~630 pS) and quantal release probability (>0.5). The strength asymmetry comes entirely from (a) **release-site count** — FSI axon makes 6 (avg) up to 18 sites per target MSN; Sp-Sp axon averages 3 sites — and (b) **synapse location** — FSI axon makes clusters on the MSN *soma*, Sp-Sp axon makes scattered en-passant contacts on **proximal dendritic spines** (40%) and spiny dendrites (48%). Soma-located inhibition has 2–3× larger somatic IPSP amplitude than dendritic.
  - **Wilson's strong conclusion (p 107):** "Mutual inhibition does not produce strong competitive interactions that could be responsible for low background activity, phasic firing, or sparse striatal movement-related activity." Rather, MSN-MSN inhibition acts in the *dendrite* — locally *very* large IPSP (~10 mV at synapse, decays to ~0.5 mV at soma) — and primarily **counteracts the KIR2 electrotonic nonlinearity** rather than implementing WTA per se.
  - **Implication for v3 lateral inhibition:** v3 (`--bg-lateral-inhibition`) shipped as the WTA mechanism in `g11_bg_runner` flagship. The Wilson data confirm that v3 is a *functional* equivalent (2026-04-28 6-seed result 4.26 ± 0.50, GO) but reinforces that the *anatomical* WTA substrate is FSI feedforward, not Sp-Sp feedback. A future v3-bis: per-action FSI pool (excited by cortex_X, inhibits str_d1_Y for Y≠X) targeting MSN soma directly. Wilson Fig 6/7/8 + p 100 also rules out "connectivity groups" (preassigned competing subnetworks) as a way to rescue the WTA hypothesis — for an average connectivity of 1/6, even 6 disjoint groups only raise across-group connectivity to 1/5, asymptotically approaching 1/6 as group count grows.

### B.05 Cholinergic tonically-active neuron (TAN) — striatal interneuron

*[from Part V — Movement (Ch 30-39); renumbered from B.53]*

- **System:** ~1–2% of striatal neurons; tonic 5–10 Hz firing; broadly arborized.
- **Biological role:** Pause response (~200 ms) to salient sensory cues, gated by thalamic centromedian/parafascicular input. Modulates corticostriatal plasticity via M1/M2 receptor effects on MSNs and DA terminals. Important for behavioral flexibility / set-shifting.
- **Sim status:** **partial** — preset `HH_STRIATAL_TAN` and `IZH2007_STRIATAL_TAN` exist but are not instantiated by `g11_bg_runner`.
- **Cluster:** B primary; C secondary.
- **Prerequisites:** B.50.
- **Citation:** Kandel 6e Ch 38 p 935–938.
- **Behavioral validation:** Salient cue → 200 ms TAN pause → permissive window for cortico-striatal plasticity.
- **Supplemental:** TK-2017 and Tepper-2018 add that ChIs/TANs are not just DA/plasticity gates: they drive a **disynaptic inhibition of MSNs** via nicotinic excitation of GABAergic interneurons. Specifically: ChI spike → α4β2-nicotinic excitation of NPY-NGF interneurons → GABA_A-slow IPSC on MSN (decay τ ≈ 120 ms — see new B.09 below); and ChI spike → nicotinic excitation of FAIs → fast facilitating IPSC on MSN (new B.11). The classic "TAN pause → cortico-striatal plasticity window" framing is therefore incomplete — the pause also lifts a *long-lasting GABA-A-slow blanket inhibition* of MSNs. (TK-2017 pp 167, 171–172; Tepper-2018 pp 2–6.) Implication: the simulator's existing TAN preset, if instantiated, would need an output pathway via at least an NGF-equivalent population to capture the disynaptic ChI→IN→MSN inhibition, not a direct ChI→MSN connection.
- **Supplemental — substance P / NK-1 driver (PBR-160 ch 16 McGinty Fig 1, pp 274–278):** ChIs in the striatum are **the dominant NK-1-receptor-expressing population** (Kaneko 1993, cited McGinty p 274). Direct-pathway D1 MSNs release substance P from local axon collaterals → SP binds NK-1 on neighboring ChIs → ChIs depolarize and release ACh. This forms an **open-loop excitation chain D1 → ChI** that operates on a slower timescale (peptide volume transmission) than the canonical Glu→ChI cortical drive. McGinty Fig 5 (p 280) and pp 277–278 also implicate ChI involvement in dopamine-induced locomotor/sensitization behavior — knockout of ChIs (Kaneko 2000 immunotoxin study) abolishes both D1 SP-mediated PPD induction *and* D2 antagonist (eticlopride)-mediated PPE induction, indicating ChIs are an obligatory relay for *both* D1- and D2-driven peptide gene expression. **Sim implication:** the existing `IZH2007_STRIATAL_TAN` preset is currently un-used in `g11_bg_runner`. To capture the full direct-pathway dynamics, instantiating a TAN pool driven by `str_d1_X → tan_X` (NK-1 mediated, slow) in addition to the canonical thalamic CM/Pf input would be a self-contained extension. The runner-side change is small; the new biology is the SP→NK-1→ACh delay loop modulating striatal plasticity gating from within the indirect-pathway side.

### B.06 Fast-spiking PV+ interneuron — feedforward inhibition in STR

*[from Part V — Movement (Ch 30-39); renumbered from B.54]*

- **System:** ~1% of striatal neurons; parvalbumin+; receives strong cortical input; widely synapses on MSNs.
- **Biological role:** Feedforward GABAergic inhibition of MSNs. Provides another cross-action competition substrate (an MSN can be inhibited by a PV+ FS responding to a different cortical channel). Faster and stronger per-synapse than MSN collaterals.
- **Sim status:** **missing** — no FS interneuron pool in `g11_bg_runner`. (Note: motor-pool lateral inhibition with FS interneurons was tested 2026-04-26 and was MIXED/NEGATIVE; B.54 is the *striatal* FS variant which is biologically more standard.)
- **Cluster:** B primary; A secondary.
- **Prerequisites:** B.50, B.52.
- **Citation:** Kandel 6e Ch 38 p 935.
- **Behavioral validation:** Cortical pulse → PV+ FS spike at ~3 ms latency → MSN IPSP at ~5 ms.
- **Supplemental:** Quantitative parameters from specialty literature (TK-2017 pp 159–164; Tepper-2018 pp 7–9):
  - **Proportion:** PV+ FSIs are only **0.7%** of striatal neurons by unbiased stereology (Rymar et al. 2004), not 1–2%. Despite their low number, FSI:MSN convergence is 2–4 FSIs per MSN; one FSI's axonal field contains hundreds of MSNs.
  - **Intrinsic:** input resistance 86 ± 38 MΩ, RMP ≈ −80 mV, no spontaneous activity at rest, narrow APs (0.29 ± 0.04 ms half-width), gamma-range subthreshold oscillations during depolarized periods, gap-junction coupling to other FSIs (3–20% coupling ratio).
  - **Synapse:** unitary FSI→MSN IPSP averages >0.4 mV at hyperpolarized MSN, >1 mV near threshold, short bursts produce compound IPSPs up to 7 mV; failure rate <1%; perisomatic targeting (50% of boutons). FSI→MSN connection probability ~50% within 250 µm.
  - **Two FSI subtypes:** in rat and primate (not mouse), FSIs split into Scgn+ (preferentially innervate D1/direct-pathway MSN) and Scgn− (preferentially innervate D2/indirect-pathway MSN) (Garas et al. 2016, reviewed Tepper-2018 pp 8–9). This is the first demonstrated FSI subtype specialization for the direct vs indirect pathway. If the simulator adds an FSI pool (recommended in B.04 above), splitting into Scgn+/Scgn− subpools that selectively gate D1 vs D2 MSNs would be a small but biologically grounded refinement.

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
- **Supplemental:** Bolam-2000 does not give patch/matrix detailed treatment but confirms the developmentally distinct compartments and notes their differential dopaminergic innervation (patch ↔ ventral midbrain DA → limbic; matrix ↔ thalamus / cortex → sensorimotor). TK-2017 and Tepper-2018 don't add patch/matrix detail beyond what Kandel covers — the specialty BG literature treats striatal microcircuitry as relatively patch/matrix-agnostic at the interneuronal level (i.e., FSIs, LTS, NGFs, etc. are found in both compartments). No new constraints for the simulator beyond the existing entry.
- **Supplemental — striosomes also project to SNr (PBR-160 ch 9 Deniau et al. p 160):** the long-standing dogma that striatonigral projections originate "exclusively from striatal matrix" has been corrected. Single-cell labeling and neurotoxic targeting of striosomal neurons reveal that **striosomes contribute substantial direct input to SNr in addition to their canonical SNc target**. A subset of D1 MSNs in striosomes co-projects to both SNc (DA cells, the classical striosomal projection) and SNr (GABA output cells). Functionally this means striosomes can directly bias BG output — not just modulate DA via SNc — as well as inject limbic information into the SNr-thalamic motor channel. PBR-160 ch 11 p 191 (Tepper & Lee) further notes that **major input to SNc dopaminergic neurons arises from striatal patch (striosome) compartment** while **GABAergic SNr neurons receive input from striatal matrix** — i.e., the patch/matrix split aligns with the SNc/SNr split at the output level. **Sim implication:** if/when the simulator adds a striosomal pool (currently missing), declaring it as projecting to both SNc and SNr (with limbic input source) creates the substrate for the "limbic gates the motor channel" effect that A.11 already flags as missing.
- **Supplemental — SNc DA neuron has *depolarized* GABA_A reversal (PBR-160 ch 11 Tepper & Lee pp 192–193, 199):** SNc dopaminergic neurons **lack KCC2** (the chloride exporter that maintains hyperpolarizing E_Cl in mature CNS neurons; Gulácsi et al. 2003 cited p 199). Their ECl is consequently several mV more *depolarized* than SNr GABA neurons. This creates a paradoxical asymmetry: stimulation of striatum or pallidum *inhibits* SNr GABA neurons (canonical) but produces only *weak inhibition* of SNc DA neurons; meanwhile, blocking SNr GABA_A receptors triggers DA *burst firing* via release of SNr→SNc tonic disinhibition (axon collaterals of SNr projection cells onto SNc dopamine neurons; Tepper et al. 1995). The **major drive to spontaneous DA burst firing in vivo is the SNr→SNc axon-collateral inhibition**, and disinhibition through GPe→SNr is the dominant route by which pallidal activity controls phasic DA. **Sim implication:** the simulator's `dopamine` region in `g11_bg_runner` currently uses generic excitatory drive; modeling SNr_X → DA disynaptic disinhibition (via SNr collaterals) would let DA bursts emerge *naturally* from the cascade rather than being injected manually. Maps to Cluster C (DA prediction error).

## Cluster O — Reward / dopamine

### O.20 Generalized Policy Iteration (GPI) — the unifying control structure
- **System:** Mathematical / algorithmic. S&B Ch 4.6 (pp. 104–106) and Ch 15.1 "The Unified View" (pp. 303–305). GPI is the alternation between **policy evaluation** (improving `V` to match the current policy `π`) and **policy improvement** (improving `π` to be greedy w.r.t. the current `V`). Almost all RL methods (DP, MC, TD, actor-critic, Q-learning) are instances of GPI; they differ in *how* they implement the two halves and in their backup style (full vs. sample).
- **Biological role:** S&B Ch 15.1 argues GPI captures "any model of intelligence" that maintains an approximate value function and an approximate policy and continually tries to improve each on the basis of the other. In the BG, this maps onto the iterative interaction between (i) DA-RPE-driven update of the value function (critic side; see C.30) and (ii) DA-RPE-driven update of cortico-striatal synaptic weights that determine the policy (actor side). The two interact every trial, not every episode — making the BG a continuous-online-GPI implementation rather than a batch one.
- **Sim status:** **partially implemented (actor side only).** The BG cascade improves the policy via DA-gated STDP every step (`policy improvement`); but with no critic (C.30), the `policy evaluation` half is missing, replaced by the implicit "evaluation" of the EMA reward baseline. Adding a critic (C.30) makes the simulator a full GPI implementation; until then it is **policy-improvement-only**, which can converge to local optima that an evaluator would have moved past.
- **Cluster:** O primary, C secondary, A secondary.
- **Prerequisites:** C.30 (actor-critic), A.* (BG cascade).
- **Citation:** Sutton & Barto Ch 4.6 (pp. 104–106), Ch 15.1 (pp. 303–305), Fig. 15.1 (referenced p. 304).
- **Behavioral validation:** **Sutton's "policy improvement theorem" test:** with a critic in place and a frozen policy, run policy evaluation alone for K steps; the value function `V` should converge (RMSE → 0). Then run a single policy improvement step (greedify w.r.t. V); the new policy's value should be ≥ the old policy's value at every state. The flagship currently cannot run this test because there is no separable `V` to evaluate.

### O.21 Average-reward formulation — undiscounted continuing tasks
- **System:** Mathematical / algorithmic. S&B Ch 11.3 "R-Learning and the Average-Reward Setting" (pp. 260–262). For non-episodic tasks (continuing, no terminal state, no natural γ < 1 discount), classical TD with γ → 1 is unstable. The fix: replace the absolute reward with a **relative reward** `R − R̄` where `R̄ = lim_n→∞ (1/n) Σ E[R_t]` is the long-run average reward under policy π. Updates: `δ = R − R̄ + max_a Q(S′, a) − Q(S, A); R̄ ← R̄ + βδ` if greedy.
- **Biological role:** This is the **algorithmic homologue of two project mechanisms** previously described independently: (a) the project's `--adaptive-da` reward-EMA-gating (relative reward EMA serves the same function as `R̄`); (b) Cluster C.25's NAc cAMP/CREB tolerance (slow molecular tracking of average dopamine activity restores baseline). All three — adaptive-DA, R-learning, and CREB tolerance — share the same algorithmic insight: in a continuing task with no episode boundary, *only the deviation from the long-run average matters for credit assignment*, not the absolute reward magnitude. The simulator's `g11_bg_runner --moving-goal` task is exactly an undiscounted continuing task — there is no terminal state; the goal moves; the agent learns continuously. This is precisely the regime R-learning was designed for.
- **Sim status:** **implicitly implemented** by the EMA reward baseline subtraction in `--adaptive-da`. Could be made explicit by declaring a `R̄` neuromodulator with `from_reward` production rule (slow EMA of reward) and subtracting it from `current_reward_signal` before plasticity; this would unify the EMA-gating with the average-reward RL formalism and provide a principled tau (the R-learning literature suggests β ≪ α, i.e. R̄ updates slower than action values).
- **Cluster:** O primary, C secondary (NM-implementable).
- **Prerequisites:** existing NM subsystem; existing reward-EMA mechanism in `--adaptive-da`.
- **Citation:** Sutton & Barto §11.3 (pp. 260–262, esp. Fig. 11.2 algorithm box); Schwartz (1993) for original R-learning derivation, cited S&B p. 264.
- **Behavioral validation:** Run g11 moving-goal task with three reward-baseline configurations: (i) absolute reward (no subtraction; baseline `--no-adaptive-da`); (ii) `--adaptive-da` symmetric EMA; (iii) explicit R-learning `R̄` (slow EMA, β ≪ α). Acceptance: under continuing-task setup (no episode boundaries), config (iii) should match or exceed (ii) on summed reward across goal transitions. Currently (ii) is in flagship; (i) is the documented baseline.

### O.22 Striatal action-value coding — strict definition, A_left vs A_right neurons
*[from Schultz 2016 J. Neural Transm. §§"Action value", "Pure reward", "Conjoint reward and action" pp. 686–688]*

- **System:** Caudate nucleus + putamen + nucleus accumbens, primarily phasically active medium spiny neurons.
- **Biological role:** Schultz16-JNT documents that the striatum is **the** structure that biologically realizes the **machine-learning concept of action value**. Strict definition (from Sutton & Barto 1998, reproduced in Schultz16-JNT p. 687): action value `Q(s, a)` for a specific action `a` "needs to be coded for each action by separate neurons irrespective of the action being chosen." Striatal neurons satisfy this empirically (Samejima et al. 2005; Lau & Glimcher 2008; Ito & Doya 2009; Kim et al. 2009; Seo et al. 2012, all cited Schultz16-JNT pp. 687–688): subgroups of MSNs **fire selectively for the value of one specific action (e.g. left arm movement) regardless of which action the animal actually chooses**. Schultz16-JNT's basic decision model (p. 688): the dopamine RPE signal updates synaptic weights at active cortex→striatum synapses (via the three-factor rule with eligibility traces); striatal action-value neurons feed a downstream "competitive decision mechanism" (= argmax-with-noise selector). Striatum also encodes **subjective rather than objective value** (Cromwell et al. 2005), reward delays (Roesch et al. 2009), and previous-trial outcomes (Histed et al. 2009). And **social reward**: a fraction of striatal neurons fire only when *a specific agent* (self vs conspecific) acts to deliver a reward (Báez-Mendoza et al. 2013).
- **Sim status:** **partial — the structure is there but not the explicit Q(s,a) readout.** The project's BG cascade has per-action `str_D1_X` populations (X ∈ {N, E, S, W} for moving-goal). These are functionally action-coding because each pool is recipient of a specific cortex→D1 subset. **However**, the project does not enforce or measure the Sutton-Barto strict-action-value criterion: that an `A_left` neuron should fire to the value of leftward action *whether or not the animal chose left*. The project's per-action readout is from spike counts during the choice phase, conflating action-value and action-execution. Lateral inhibition (cheat-5 v3 `--bg-lateral-inhibition`) further entangles action values across pools.
- **Cluster:** O primary, A secondary (BG), G secondary (decisions).
- **Prerequisites:** A.x (striatum), C.22 (RPE), C.32 (two-component DA), O.20 (GPI control structure).
- **Citation:** Schultz16-JNT pp. 686–688 §§"Pure reward", "Conjoint reward and action", "Action value", "Reward learning"; Samejima et al. 2005 (Fig. 4c in Schultz16-JNT p. 687); Báez-Mendoza et al. 2013 (Fig. 5, social reward).
- **Behavioral validation:** Implement a **forced-choice trial** variant in `g11_bg_runner`: on a fraction of trials, inject motor exploration noise heavily into one specific motor pool to force a specific action choice independent of cortex→D1 weights. Measure firing of `str_D1_left` during forced-right trials. The strict action-value criterion says `str_D1_left` should still encode the *would-be value* of left, even though the animal went right. Currently the project does not produce that signal — `str_D1_X` activity is largely chosen-action activity, not value-of-X-irrespective-of-choice. This is a measurable gap and a concrete validation criterion.

### O.23 Three reward functions — reinforcer, goal/value, emotion
*[from Schultz 2016 NRN p. 1 Introduction]*

- **System:** Behavioral / functional taxonomy applied to the same reward stimulus.
- **Biological role:** A reward has three distinct behavioral functions, dissociable in experiment (Schultz16-NRN p. 1):
  1. **Positive reinforcer** — induces learning. Operationalized by reinforcement-learning paradigms (operant conditioning, Pavlovian transfer). Substrate: DA × eligibility-trace plasticity.
  2. **Goal object for approach + economic choice** — assigns value to options for selection. Operationalized by binary-choice tasks revealing transitive subjective preferences and certainty equivalents. Substrate: DA-encoded utility (C.34) + striatum action-value (O.22) + OFC/vmPFC scalar value (O.19).
  3. **Emotion** — produces pleasure / desire. Operationalized weakly in animals (taste reactivity, place preference) but not quantitatively. Substrate: NOT phasic DA (DA is the teaching signal, not the hedonic signal — see C.24, C.27). Hedonic "liking" depends on opioid/cannabinoid hotspots in NAc (Berridge).
  Schultz16's key methodological claim (p. 1): functions 1 and 2 can be quantitatively studied via behavioral tasks, function 3 cannot, so neuroscience of reward should focus on 1 and 2. Function 3 is the "wanting vs liking" / hedonic axis covered in C.27.
- **Sim status:** **partial.** Project cleanly implements function 1 (positive reinforcer via `current_reward_signal × eligibility_trace × STDP`). Partially implements function 2 (per-action eligibility gating gives action-value-like behavior, but no separable subjective-value readout for choice; argmax + WTA approximate the choice mechanism). Does not implement function 3 (no hedonic axis, no opioid hotspot machinery; this is appropriate for the project's scope but worth noting in a feature catalog).
- **Cluster:** O primary, C secondary.
- **Prerequisites:** C.04 (DA), C.27 (wanting vs liking), O.19 (subjective value).
- **Citation:** Schultz16-NRN p. 1 Introduction; cross-references C.27 (Berridge) for function 3.
- **Behavioral validation:** N/A as a single mechanism — this is a taxonomic frame for grouping the C/O entries above. Useful as a project-level documentation point: "the simulator implements reward function 1 fully, function 2 partially, and explicitly does not address function 3."

---

---

## Cluster C — additions

### C.28 TD error as the algorithmic form of phasic dopamine — δ = r + γV(s′) − V(s)
- **System:** Mathematical / algorithmic. The TD-error is the bracketed update term in TD(0) (S&B §6.1, p. 144), in actor-critic (S&B §11.1, p. 258), and in Q-learning (S&B §6.5, p. 157).
- **Biological role:** Schultz98 (Eq. 6, p. 12; Eq. 6a–6c, pp. 12–13) and Schultz, Dayan & Montague (1997) established that phasic VTA/SNc DA encodes exactly this quantity. The three signatures (Schultz98 Fig. 2, p. 4): (a) burst on unpredicted reward [δ > 0 because `r > 0, P=0`], (b) no response to predicted reward [δ ≈ 0 because `r = P`], (c) dip on omitted reward [δ < 0 because `r = 0, P > 0`]. The cue-shift transfer (Schultz98 Fig. 3, p. 5) emerges naturally because `P(t)` is itself learned over trials.
- **Sim status:** **partial — gap is measurable.** Project's `current_reward_signal = r(t)` *not* `r(t) + γV(s′) − V(s)`. The `--adaptive-da` mechanism EMAs `r(t)` to subtract a baseline (≈ Rescorla-Wagner one-step), but never bootstraps from a learned `V(s′)`, so it cannot produce the cue-shift signature. Closing this requires a critic population (see C.30).
- **Cluster:** C primary, O primary, J secondary (drives plasticity).
- **Prerequisites:** O.50 (DA), J.* (eligibility/STDP), regions framework for critic population.
- **Citation:** Sutton & Barto Ch 6.1 (TD prediction) and Ch 6.4–6.5 (Sarsa, Q-learning); Schultz98 pp. 11–14 §"Using the dopamine reward prediction error signal"; HS98 entire paper.
- **Behavioral validation:** Single-trial Pavlovian conditioning paradigm with a fixed delay between CS and US. Acceptance metric: across N=20 conditioning trials, the mean phasic dopamine response time should *shift monotonically* from US-onset to CS-onset (correlation r > 0.7 between trial number and time-of-peak). On reward omission (CS but no US), the dopamine pool should show a *depression* at the predicted US time. Currently failing.

### C.29 Eligibility traces and TD(λ) — credit assignment over time
- **System:** Mathematical / algorithmic. S&B Ch 7 entirely (pp. 167–195) defines eligibility traces; Eq. 7.8–7.11 (pp. 177–178) are the backward-view (mechanistic) updates.
- **Biological role:** An eligibility trace `e_t(s)` is a memory variable per state (or per state-action pair) that records "recent visitation," decaying each step by `γλ` and incremented by 1 (accumulating) or set to 1 (replacing) when the state is visited. The TD error `δ_t` then drives weight updates proportional to `e_t(s)` for **all** states recently visited, not just the immediately preceding one (S&B Eq. 7.11, p. 178: `ΔV(s) = αδ_t e_t(s)`). λ ∈ [0,1] interpolates: λ=0 ⇒ TD(0) (only most recent state); λ=1 ⇒ Monte Carlo (all visited states equally credited). Intermediate λ usually wins (S&B Fig. 7.2, p. 172). Schultz98 (Eq. 9, p. 15) names this the "synaptic eligibility trace" and proposes Ca²⁺ + CaMKII as the biological substrate.
- **Sim status:** **implemented.** `cp_eligibility_trace` array decays geometrically per step and is gated multiplicatively by `current_reward_signal` (DA) and `cp_plasticity_gain` (per-pathway gate). This is **TD(λ) with accumulating traces** in everything but name. The project's λ is implicit in the trace decay constant. Two refinements possible: (1) **replacing traces** (S&B p. 184: trace set to 1 on visitation rather than incremented) often outperform accumulating in noisy settings; (2) **dutch traces** (`e ← (1−α)e + 1`) are theoretically equivalent to a real-time forward-view λ-return (S&B Fig. 7.6, p. 176). Both are 1-line changes in `fused_eligibility_trace_decay()`.
- **Cluster:** C primary, J primary (synaptic plasticity machinery).
- **Prerequisites:** O.50 (DA), J.* (STDP).
- **Citation:** Sutton & Barto Ch 7 (entire), esp. Fig. 7.6 (p. 176) for trace-type comparison and §7.10 (p. 190) for conclusions; Schultz98 §"Postsynaptic plasticity together with synaptic eligibility trace" (p. 15) for biological grounding.
- **Behavioral validation:** Random-walk-19-states benchmark (S&B Example 7.1, p. 171 and Fig. 7.2 p. 172): RMSE of `V(s)` after 10 episodes should be lowest at intermediate λ ≈ 0.5–0.9, *worse* at λ=0 (TD(0)) and worse again at λ=1 (Monte Carlo). The simulator could validate its eligibility trace implementation by sweeping the trace decay constant and confirming the U-shape.

### C.30 Actor-critic architecture — separable policy and value with shared TD error
- **System:** Mathematical / algorithmic. S&B Ch 11.1 (pp. 257–259) and Fig. 11.1 (p. 258). The earliest combination of TD with trial-and-error learning (Barto, Sutton & Anderson 1983, cited S&B p. 24 and Schultz98 p. 14).
- **Biological role:** Critic learns `V(s)` (or `Q(s,a)`); its TD error `δ_t = R + γV(s′) − V(s)` is broadcast as a **single scalar** that updates both itself (improving `V`) and the actor's action preferences `H(s,a) ← H(s,a) + αδ_t` (S&B Eq., p. 259). Because the critic and actor both consume the same δ, learning is on-policy and consistent. **Anatomical mapping** (Schultz98 Fig. 9C, p. 13; Houk, Adams & Barto 1995): VTA/SNc DA = critic δ output; striosome-patch (limbic striatum) = critic state-value; striatal matrix (sensorimotor striatum) = actor preferences; corticostriatal synapses on matrix = actor weights modified by δ. **The actor-critic mapping is the cleanest available account of the BG-DA system in RL terms.**
- **Sim status:** **partial — actor implemented, critic missing.** Phase-B BG cascade is an actor-only architecture: per-action `cortex_X → str_D1_X / str_D2_X → gpi_X → thal_X → motor_X` is the actor `H(s,a)` distributed across action-specific pools, with `current_reward_signal` substituted directly for δ. There is **no separable population that outputs a learned `V(s)`**, and consequently no bootstrapping. The Direct (D1) / Indirect (D2) pathway distinction in the project also maps onto a partial actor-critic in another way: D1 = "Go" = positive-affect actor, D2 = "NoGo" = negative-affect actor (both learn from δ, but with opposite-signed plasticity rules). This is a **two-actor, no-critic** architecture, more closely matching Frank's BG models than canonical actor-critic.
- **Cluster:** C primary, O primary, A primary (BG).
- **Prerequisites:** A.* (BG cascade, already in flagship), regions/pathways framework, eligibility trace machinery (C.29, already implemented).
- **Citation:** Sutton & Barto Ch 11.1 (pp. 257–259) + Ch 11.2 (eligibility traces for actor-critic, pp. 259–260); Schultz98 §"Neurobiological implementations of temporal difference learning" (pp. 13–14, esp. Fig. 9C); Barto 1995 ("Adaptive critics and the basal ganglia"), cited Schultz98 p. 14.
- **Behavioral validation:** Two acceptance metrics. (a) **Cue-shift signature** — see C.28; with critic added, dopamine pool firing should shift from US to CS over conditioning trials. (b) **Reward-omission dip** — with critic, omission of predicted reward should produce δ < 0 in the dopamine pool. Currently the flagship reproduces neither without external scaffolding.

### C.31 Bootstrapping vs. Monte Carlo backups — why phasic DA cannot be Monte Carlo
- **System:** Mathematical / algorithmic. S&B Ch 6.1 (pp. 143–148) defines bootstrapping as "updating a guess from a guess." S&B Fig. 7.1 (p. 169) shows the n-step spectrum from 1-step TD (full bootstrapping) to Monte Carlo (no bootstrapping).
- **Biological role:** A Monte Carlo (MC) backup waits until episode end (the actual return `G_t`) before updating any state estimate; a TD backup updates *immediately* using `R_{t+1} + γV(S_{t+1})` as a proxy for the unknown remainder of the return. Schultz98 (Eq. 6a, p. 12) and Hollerman-Schultz 1998 establish that DA bursts occur **on a single trial, with no episode-end wait**, and shift from US to CS over consecutive trials at the rate fitted by TD(0) — not by MC, which would not transfer until many trials had completed. The empirical signature is therefore **direct evidence of bootstrapping in biology**: the brain is not waiting for trial outcomes to update predictions; it is updating from one moment to the next using its current estimates as the target. This is one of the few cases where a deep computational architectural choice (bootstrapping) is *required* by the empirical data.
- **Sim status:** **trivially "implemented"** in the degenerate sense that `current_reward_signal` is a 1-step signal applied immediately to eligibility traces — but the project does not bootstrap a *value estimate* (see C.28, C.30). The current architecture is closer to a *windowed Monte Carlo* in which the window is the eligibility-trace decay length: weight updates accumulate over a ~1s window without any predictive value computation.
- **Cluster:** C primary, J secondary.
- **Prerequisites:** C.28 (TD error), C.29 (eligibility traces).
- **Citation:** Sutton & Barto §6.1 "TD Prediction" (pp. 143–148); §7.1 "n-Step TD Prediction" (pp. 168–172); §7.10 "Conclusions" (p. 190 — "eligibility traces in conjunction with TD errors provide an efficient, incremental way of shifting between Monte Carlo and TD"); HS98 entire paper as the empirical case for bootstrapping.
- **Behavioral validation:** N/A — this is a theoretical entry that documents a mapping. The empirical validation is C.28's cue-shift criterion, which a non-bootstrapping architecture cannot satisfy.

### C.32 Two-component DA response — detection (Component 1) + utility-RPE (Component 2)
*[from Schultz 2016 NRN review, integrated with Schultz 2016 J. Neural Transm.]*

- **System:** Same A8/A9/A10 midbrain DA cell groups as in C.04, C.16, C.22 — but parsed as a **temporally compound burst** rather than a single phasic event.
- **Biological role:** The phasic dopamine burst on a salient/rewarding stimulus consists of **two sequential components**:
  - **Component 1 (detection / salience):** latency 60–90 ms, duration 50–100 ms. Unselective — fires to any sufficiently intense, novel, generalizable, or contextually-rewarded stimulus, including aversive stimuli, conditioned inhibitors, and unrewarded probes. Graded by physical intensity, reward generalization (similarity to known reward cues), reward context (rewarded vs unrewarded session), and novelty. Implements the salience term of the Pearce-Hall (1980) attentional learning rule. Functionally: amplifies learning rate and downstream sensory gain on any "potentially important" event before identification.
  - **Component 2 (value / utility RPE):** latency 150–300 ms (clearly separated in demanding tasks like dot-motion discrimination, Schultz16-JNT Fig. 1a). Graded by **subjective utility** (not raw reward) — follows the inflected convex-then-concave utility function measured by certainty-equivalent fractile procedure on equiprobable risky gambles. Tracks reward, risk-discounting, delay-discounting, and arithmetic sum of positive + negative outcomes. Implements `δ = r̂(t) − P(t−1)` over a learned utility function `u(r)`.
  - The two components operate on a **10-ms-precision temporal structure**: the value information of Component 2 cannot be read out before Component 1 finishes, and stimulant drugs that prolong extrasynaptic DA *smear* Component 1 into Component 2 and create a "false value signal" — proposed mechanism for stimulant addiction and psychiatric pathology of DA.
- **Sim status:** **partial — accidental implementation across two CLI flags.** The project's `--surprise-lr-boost` (LR multiplier × `(1 + α|RPE|)`) and `--adaptive-da --adaptive-da-ema-decay-negative 0.7` (asymmetric per-action eligibility gating) are functionally Component-1 and Component-2 analogs respectively. They were discovered empirically and currently treated as alternatives in the recommended-config table; biology says they should compose. The combination has not been validated and the project notes that "combining adaptive DA with WTA, or adaptive DA with LR boost, doesn't compose well" (CLAUDE.md "Other refinement variants") — but that's an empirical interaction with shared reward EMA; the principled biological prescription is that Component 1 should drive a global LR with the *raw* surprise (no EMA), while Component 2 should drive eligibility with the EMA-tracked utility error. Rewriting `--adaptive-da` to share the EMA only for Component 2 and leaving Component 1 raw-surprise-driven would be a one-flag-rewrite test.
- **Cluster:** C primary, O secondary.
- **Prerequisites:** C.04 (DA), C.22 (RPE), C.20 (tonic-phasic).
- **Citation:** Schultz16-NRN pp. 4–11 §§"The initial component: detection", "The main component: valuation"; Schultz16-JNT pp. 681–682 §"Two phasic response components", Fig. 1a.
- **Behavioral validation:** Replicate Nomoto et al. 2010's dot-motion discrimination paradigm in `g11_bg_runner` — vary stimulus discriminability so Component 1 stays constant (detection) while Component 2 varies with reward probability (value). Currently impossible because the simulator has only one DA channel. **Specific instrumentation criterion:** measure the firing rate of the dopamine pool (or `current_reward_signal` proxy) at 60–90 ms after stimulus onset (should be uniform across reward conditions) vs at 150–300 ms (should track reward probability monotonically). Project does not currently emit a time-resolved DA signal — adding one would be the first concrete step.

### C.33 Pedunculopontine nucleus (PPN) — sensory + reward input to DA neurons
*[from Schultz 2016 J. Neural Transm. §"Pedunculopontine nucleus" pp. 684–686]*

- **System:** Brainstem (mesopontine) cholinergic + glutamatergic nucleus. Major projection: **PPN → SNc DA, SNr (substantia nigra pars reticulata), GPi, STN**. Inputs: cortex (via internal globus pallidus and STN), thalamus, cerebellum, spinal cord, contralateral PPN. Already partially indexed in **C.18 Mesopontine Cholinergic Nuclei (PPT/LDT) — REM-on cholinergic** but the *reward* function is distinct and not covered there.
- **Biological role:** PPN neurons show heterogeneous activity: some fire to sensory stimuli, some to saccadic eye movements, some to reward-predicting stimuli (sustained activations until reward delivery), some to reward delivery itself. PPN neurons differentiate reward magnitude (higher firing for larger predicted reward) but **do NOT fire bidirectional reward prediction errors** the way DA neurons do. PPN reward responses are not depressed by reward omission. **Functional role inferred from anatomy + lesion:** PPN supplies "components" of the DA RPE signal — electrical PPN stimulation activates 20–40% of DA neurons; PPN inactivation by local anesthetic in behaving rats reduces DA prediction-error responses to conditioned cues (Pan & Hyland 2005, cited Schultz16-JNT p. 685). Both glutamate and acetylcholine are involved in PPN→DA excitation. Latencies of stimulus responses are slightly shorter in PPN than in DA neurons — consistent with PPN being a **driver / contributor**, not a follower, of the early DA detection component (Component 1).
- **Sim status:** **missing.** The simulator's BG cascade in `g11_bg_runner.build_bg_brain_regions` does not include PPN. There is no upstream sensory/reward driver of the dopamine pool other than `current_reward_signal` event-injection. **Implication:** the project cannot model the *dynamics* of how a reward-predicting cue becomes able to drive DA — biologically PPN is a candidate substrate of that learned drive (along with striatum and STN). Adding a small PPN region (e.g. 30–50 neurons) projecting to the dopamine pool, receiving sensory cue inputs, with plastic synapses gated by reward delivery, would let the project model the **cue-shift transfer** dynamic that is the canonical Schultz signature still missing from the project (see C.22 supplemental, "Currently the project reproduces only sign (a)…").
- **Cluster:** C primary, A secondary (BG).
- **Prerequisites:** C.16 (VTA/SNc), C.18 (PPT/LDT — anatomical overlap), C.22 (RPE).
- **Citation:** Schultz16-JNT pp. 684–686 §"Pedunculopontine nucleus"; Kobayashi & Okada 2007; Okada et al. 2009; Pan & Hyland 2005; Hong & Hikosaka 2014 (all cited by Schultz16-JNT).
- **Behavioral validation:** PPN inactivation should **specifically degrade the cue-evoked DA burst** (Component 1 + transferred Component 2) without abolishing the unconditioned-reward DA burst. In `g11_bg_runner`, this is testable: simulate a PPN ablation by zeroing PPN→DA projection, run a Pavlovian conditioning protocol, expect normal initial-trial reward-evoked DA but failure of cue-DA acquisition across trials.

### C.34 DA codes economic utility u(x) — nonlinear, inflected, risk-sensitive
*[from Schultz 2016 NRN §§"Subjective reward value", "Utility" pp. 6–9; Schultz 2016 J. Neural Transm. §"Formal economic utility" pp. 684–685]*

- **System:** Same DA neurons as C.04/C.22 — but the *response amplitude curve* is the focus.
- **Biological role:** When physical reward amount is varied, DA Component 2 amplitude does **not** increase linearly — it follows the **inflected utility function u(x)** the same animal reveals through behavioral choices on equiprobable risky gambles (Stauffer et al. 2014). Concretely: u(x) is **convex (progressively increasing)** at small reward amounts (animal shows risk-seeking on small gambles), **concave (progressively flattening)** at large amounts (animal shows risk-aversion on large gambles), with an inflection point in between. DA Component 2 reproduces this shape (Schultz16-NRN Fig. 4d). Tested directly with binary equiprobable gambles, the higher of the two gamble outcomes elicits non-monotonically varying positive RPE responses that match u(x). Schultz16's strong claim: "the phasic dopamine reward prediction-error response can be specified as a **utility prediction-error signal** … dopamine responses seem to represent a physical correlate for utility" (Schultz16-NRN p. 9).
- **Sim status:** **missing as a feature, irrelevant for current tasks.** Project tasks are binary-reward (moving-goal, BG cascade). For these, DA-codes-utility vs DA-codes-reward is observationally identical. **Becomes critical** for any future risk-sensitive choice tasks (binary-gamble selection, two-arm bandit with variance differences, delay-discounting tasks). One-line implementation: insert a `utility_fn(reward)` callable in `current_reward_signal` computation. Default `lambda r: r` preserves current behavior; `lambda r: np.sign(r) * np.power(np.abs(r), α)` with α<1 gives the concave (risk-averse) regime, two-piece convex/concave gives the full inflected curve.
- **Cluster:** C primary, O secondary (decisions).
- **Prerequisites:** C.22 (RPE), O.19 (subjective value).
- **Citation:** Schultz16-NRN pp. 6–9 §§"Subjective reward value", "Utility"; Schultz16-JNT pp. 684–685 §"Formal economic utility"; Stauffer et al. 2014 (cited in both); Caraco et al. 1980; Kagel et al. 1995.
- **Behavioral validation:** Run a binary-gamble selection task in `g11_bg_runner`. Measure DA pool activation amplitude as a function of reward delivered. Plot DA-amplitude vs reward-amount. With current linear `reward = +1 / -1`, this is degenerate. With a graded reward (e.g. `reward ∝ inverse_distance_to_goal`) this should reveal whatever utility curve the project implicitly imposes — currently linear, biology says inflected.

### C.35 Stimulant drugs smear Component 1 into Component 2 — drug-action mechanism
*[from Schultz 2016 NRN p. 10 BOX 2; Schultz 2016 NRN §"Correct behaviour based on late component"]*

- **System:** Pharmacological action of dopamine reuptake inhibitors (cocaine, amphetamine, methylphenidate) on the temporal structure of phasic DA bursts.
- **Biological role:** The two-component DA response operates on a "narrow timescale that requires unaltered, precise processing in the 10 ms range" (Schultz16-NRN p. 9). Stimulant drugs prolong extrasynaptic DA concentration after a phasic burst (DAT block → impaired reuptake → DA persists in extracellular space for hundreds of ms instead of <100 ms). Schultz16's mechanistic proposal: this prolongation causes **Component 1 (detection) to overlap with Component 2 (value)**, so postsynaptic plasticity machinery cannot distinguish "detected something" from "this was rewarding." The downstream MSNs receive a fused signal that they treat as Component-2-strength value information for any salient stimulus, biasing learning toward whatever cue triggered Component 1 — including conditioned-aversion cues, novelty cues, and physical-impact cues that biology designs Component 1 to *not* commit value to. This is offered as a candidate **mechanism for stimulant addiction** distinct from the simpler "high tonic DA → euphoria" account.
- **Sim status:** **missing — and currently impossible to model.** The project's `current_reward_signal` is event-triggered with no internal Component 1/Component 2 temporal structure to smear. The `NeuromodulatorConfig.decay_tau_ms` field could in principle model DA reuptake — setting a longer tau would produce a smeared signal — but without distinct C1 and C2 production rules there's no smearing of one onto the other. This is therefore a future-feature that depends on first implementing C.32 (two-component DA response) as distinct mechanisms.
- **Cluster:** C primary, P secondary (disease — addiction).
- **Prerequisites:** C.32 (two-component DA), C.21 (volume transmission).
- **Citation:** Schultz16-NRN p. 10 §"Correct behaviour based on late component" + BOX 2 "Drug actions".
- **Behavioral validation:** With C.32 implemented, model a "stimulant" condition by extending `decay_tau_ms` of the salience-Component-1 modulator. Predicted phenotype: increased weight changes for novel/physically-intense-but-unrewarded stimuli; impaired cue-shift transfer (because Component 1 cannot decouple from Component 2 across learning); preserved behavior on already-learned tasks (Component 2 still functions, but driven by smeared Component 1).

---

---

## Cluster C — Dopamine & neuromodulation

**27 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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
- **Supplemental:** Schultz98 frames DA as a *3-component* signal: (1) a prediction-independent **alerting** component to salient onsets, (2) a learned **reward-prediction** component that transfers from primary reward to the earliest reward-predicting cue across trials, and (3) the **prediction error** (Schultz98 pp. 1–4, "Summary 2: effective stimuli"; Eqs. 1–2 p. 11). Of these, the simulator implements only (3), and only positively (the cue-shift dynamic of (2) is the canonical Schultz signature; see C.22). Schultz98 also distinguishes **two functions** of DA on different timescales: phasic burst-encoded RPE (10s–100s of ms) and tonic ambient ~5–10 nM extracellular concentration that *enables* striatal/cortical processing on a 10s–100s of seconds scale (Schultz98 pp. 19–21, "Dopamine reward signal vs. parkinsonian deficits"). The project's `current_reward_signal` collapses both into one event-triggered scalar — closing this gap is feasible with the existing NM subsystem (`baseline + decay_tau_ms + concentration_min/max`) without any new GPU code.
- **Supplemental — the two-component DA framework (Schultz16-NRN):** Schultz's 2016 review **revises** the 3-component scheme of Schultz98 into a cleaner **two-component** picture, which is what the project should target for biological fidelity:
  - **Component 1 — initial detection / unselective activation.** Latency 60–90 ms, duration 50–100 ms; "unselectively detects any potential reward (including stimuli that turn out to be aversive or neutral)" (Schultz16-NRN p. 183 Abstract; pp. 4–6 §"The initial component: detection"; Schultz16-JNT p. 681 §"Two phasic response components"). Sensitivity is graded by **physical intensity, reward context, reward generalization, and novelty** — i.e. it is a multi-flavored salience signal, not a value signal. Boosted by stimulus intensity (Schultz16-NRN Fig. 3a, p. 4); by being presented in a rewarded context (Schultz16-NRN Fig. 3b, p. 4 "increasing the probability that the animal will receive a reward… increases the incidence of dopamine activations to unrewarded stimuli"); by physical resemblance to known rewards (Fig. 3c, p. 4); and by novelty (Fig. 3d, p. 5). Component 1 is **transient**: "lasts only until the subsequent value component conveys the accurate reward value information" (Schultz16-NRN p. 9).
  - **Component 2 — value / utility prediction error.** Begins during Component 1 but extends to ~150–300 ms latency in demanding tasks (Schultz16-NRN Fig. 2c–e; Schultz16-JNT Fig. 1a "second dopamine response component begins later, at latencies around 250 ms… varies with reward value"). This is the canonical Schultz RPE — "the fully developed main response component codes utility… the phasic dopamine reward prediction-error response can be specified as a utility prediction-error signal" (Schultz16-NRN p. 9, "The main component: valuation").
  - **Why this matters for the project — direct mapping to the existing flagship config:**
    - The project's `--surprise-lr-boost` mechanism (`(1 + α × |reward - reward_ema|) × reward_learning_rate`) **is a Component-1 analog**: an unselective salience-driven scalar that scales the global plasticity rate on any large RPE magnitude, regardless of valence. This is what Component 1 is biologically — multisensory, valence-blind, salience-graded. The mechanism's robustness across slow- and fast-change tasks (`research/findings/2026-04-26-surprise-lr-boost.md`) is consistent with Component 1's role: rapid detection that does not commit to value before identification.
    - The project's `--adaptive-da --adaptive-da-ema-decay-negative 0.7` mechanism **is a Component-2 analog**: it gates eligibility per-action based on a slow positive / fast negative reward EMA, which mirrors the Component-2 utility-tracking dynamics. The asymmetry (slow positive τ~10, fast negative τ~3) is biology-grounded: phasic depressions on omission are sharper and shorter than activations on acquisition — see Schultz16-NRN p. 9 "the second response component persists throughout the resulting behaviour" and the omission-dip data preserved from Schultz98.
  - **Open work:** the project does NOT currently model the two components as distinct mechanisms running in parallel — `current_reward_signal` is a single scalar. A faithful Schultz16 implementation would have a fast-onset salience pulse (Component 1) gating learning rate, and a slower value-tracking pulse (Component 2) gating eligibility — composing the two existing CLI flags in a principled way rather than as alternatives. See new entry **C.32 Two-component DA response** for the dedicated mechanism.
- **Supplemental — three reward functions (Schultz16-NRN p. 1):** rewards have three behavioral functions: (1) **positive reinforcers** that induce learning, (2) **goal objects** for approach behavior and economic choice, (3) **emotion** (pleasure/desire). Functions 1 and 2 are quantitatively assessable in animals; function 3 is hard to test. The project's `current_reward_signal` exclusively implements function 1, partially implements function 2 (via per-action eligibility), and does not address function 3.

### C.05 Norepinephrine (NE) — arousal / vigilance / fight-or-flight

- **System:** locus coeruleus (LC) → diffuse cortical, thalamic, hippocampal, hypothalamic, spinal projections
- **Biological role:** tonic LC firing tracks behavioral arousal (low during sleep, high during stress). Phasic LC bursts on salient stimuli. Receptors: α1 (Gq), α2 (Gi, autoreceptor), β1/β2/β3 (Gs). Increases SNR by simultaneously suppressing background firing and enhancing selective response. Critical for memory consolidation in the hippocampus, attention in PFC.
- **Sim status:** partial. NM framework supports it; one prior session (E.1) tested NE on the silent-motor task and found it insufficient (the silent-motor trap is upstream of NE modulation). **Could be added easily** — has not been deployed in the current flagship config. Yerkes-Dodson curve (inverted-U arousal-performance relationship) would be a natural validation.
- **Cluster:** C
- **Citation:** Kandel 6e Ch 16 p 376–380
- **Behavioral validation:** add NE concentration; vary baseline; measure SNR (signal-induced firing rate change / background CV(ISI)). Should peak at intermediate NE.
- **Supplemental:** Schultz98 (pp. 18–19, "Comparisons with other projection systems / Noradrenaline") contrasts LC-NE with VTA/SNc-DA: NE neurons respond to a much wider range of arousing stimuli (including aversive ones), discriminate poorly between appetitive and neutral events, and **track familiarity / change rather than reward valence**. Critically, NE responses appear and disappear faster than DA (Aston-Jones et al. 1997: NE neurons reverse target preference *before* behavioral reversal completes, whereas DA tracks the RPE itself). This is the empirical grounding for the project's existing **surprise-LR-boost** mechanism (`(1 + α × |RPE|) × reward_learning_rate`, see C.22 supplement) — by amplifying the global plasticity rate on unexpected outcomes regardless of valence, surprise-LR-boost is functionally an LC-NE signal layered on top of the DA RPE channel, not a reshaping of DA itself. Adding a declared NE modulator with `from_error_persistence` production rule and `plasticity_rate` target type would make this explicit rather than implicit.
- **Supplemental — Component-1 DA shares mechanism with LC-NE:** Schultz16-NRN connects DA's initial unselective component to the broader monoaminergic salience-detection family (pp. 4–7, §"Salience"). Both DA Component 1 and LC-NE phasic bursts respond to physical, novelty, and motivational salience without committing to valence. The Pearce-Hall attentional learning rule (Schultz16-NRN p. 6, Pearce & Hall 1980) — "surprise salience derived from reward prediction errors enhances the learning rate" — is given by Schultz as the formal model that **both** Component-1 DA and LC-NE bursts plausibly implement. This justifies the project's existing identification of `--surprise-lr-boost` as functionally an LC-NE signal layered onto DA: biologically these are **distinct populations carrying overlapping salience information**, and modeling them as separate `NeuromodulatorConfig` entries (one DA-Component-1, one NE) with shared `from_error_persistence` production rules and `plasticity_rate` target type would correctly factor what the simulator currently fuses into a single LR multiplier.

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
- **Supplemental:** Schultz98 §"Dopamine reward signal vs. parkinsonian deficits" (pp. 19–21) is the locus classicus for the **tonic-phasic dual function**: (a) phasic short bursts (50–110 ms latency, ~200 ms duration; Schultz98 p. 6 "Homogeneous character of responses") report RPE; (b) tonic 1–5 Hz pacemaker firing maintains a 5–10 nM ambient extracellular DA concentration that activates high-affinity D2 receptors in their high-affinity state and is **necessary for striatal plasticity itself** (Calabresi et al. 1992a, 1997 cited p. 21 — D2 antagonists or D2 knockout abolish posttetanic depression). Implication: a project that wants Schultz-grade fidelity must run *both* a tonic concentration variable that maintains plasticity competence *and* a phasic burst that scales eligibility-trace × DA. The project currently fuses both into one transient scalar — explicitly, in the `current_reward_signal` design any plasticity that requires "DA presence" is conditioned on the same event that delivers the teaching signal, conflating Schultz's two axes.
- **Supplemental — phasic structure is itself two-tier (Schultz16-NRN, Schultz16-JNT):** Schultz's 2016 work **further decomposes the phasic burst** into the two-component sequence (Component 1 detection 60–90 ms, Component 2 value 150–300 ms; Schultz16-JNT p. 681). The "phasic burst" is therefore not a unitary event but a two-sub-event temporal structure operating "on a narrow timescale that requires unaltered, precise processing in the 10 ms range" (Schultz16-NRN p. 9). Stimulant drugs that prolong DA concentrations cause Component 1 to **smear into and overlap** Component 2, generating "a false value signal for postsynaptic neurons" — Schultz16-NRN proposes this as a mechanistic substrate for stimulant addiction and psychiatric disorder dynamics (p. 10, BOX 2). For the project: any future model of psychotropic-drug effects on the DA system should preserve the two sub-phases, because their *temporal separation* is the carrier of value information.

### C.21 Volume-Transmission Neuromodulation — non-synaptic diffuse release

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.57]*

- **System:** Monoamine + neuropeptide co-release from boutons-en-passant.
- **Biological role:** Many monoaminergic axon terminals do NOT form conventional synapses; transmitter spills into extracellular space and acts on G-protein-coupled receptors at micrometer-to-millimeter distances. This is fundamentally different from point-to-point glutamatergic/GABAergic transmission and is what justifies modeling neuromodulators as scalar fields rather than per-synapse signals.
- **Sim status:** implemented (architectural fit) — `sim/neuromodulators.py` already models neuromodulators as global concentration scalars with target-type effects (`synaptic_gain`, `plasticity_rate`, `excitability_drive`), which is the right abstraction for volume transmission.
- **Cluster:** C primary.
- **Prerequisites:** C.50 (LC) — canonical example.
- **Citation:** Kandel 6e Ch 40 pp 1001-1002.
- **Behavioral validation:** Microdialysis vs. fast-scan cyclic voltammetry — extrasynaptic monoamine concentration tracks population firing rather than individual release events.
- **Supplemental:** Schultz98 (pp. 8–11, Fig. 8 and §"Processing in striatal neurons") provides the anatomic bookkeeping that justifies modeling DA as a global scalar: ~10,000 cortical terminals + 1,000 DA varicosities per MSN dendrite; each DA axon has ~500,000 terminal varicosities; nigrostriatal divergence factor 300–400 in macaque. Phasic burst → "short puff of dopamine that is released from extrasynaptic sites or diffuses rapidly from synapses into the juxtasynaptic area. Dopamine quickly reaches regionally homogenous concentrations likely to influence the dendrites of probably all striatal and many cortical neurons" (Schultz98 p. 9). This is the empirical argument for `compute_synaptic_gain_multiplier()` and the `excitability_drive` target type being computed from *concentration* rather than per-synapse signaling — the abstraction in `sim/neuromodulators.py` (scalar concentration with diffuse target effect) **is** Schultz's "rather global reinforcement signal" implemented.

### C.22 Dopamine Reward Prediction Error (Schultz RPE) — phasic DA encodes δ

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.58]*

- **System:** A10/VTA → NAc + PFC + amygdala + hippocampus.
- **Biological role:** **Project-critical.** Schultz monkey experiments (1997, 2016): unexpected reward → DA burst; cue-predicted reward → burst shifts to cue, no burst at reward; predicted reward omitted → DA *dip* below baseline at expected reward time. This is the canonical TD-learning prediction-error signal. Drives selective strengthening of synapses on coactive eligibility traces (three-factor rule).
- **Sim status:** partial — flagship implements broadcast DA = reward signal driving eligibility-trace × DA → weight update. The 2026-04-26 surprise-LR-boost variant explicitly amplifies LR by `(1 + α × |reward - reward_ema|)`, which IS an RPE-flavored mechanism. The 2026-04-26 adaptive-DA targeting uses reward EMA gating. **However, project doesn't model the burst↔cue transfer** — the cue itself doesn't acquire DA-burst-evoking power as it does in Schultz's data; reward only releases DA at delivery time. [discrepancy: textbook RPE includes cue-shift and omission-dip; project models pos-RPE-amplification but not the cue-shift dynamic that is the canonical RPE signature.]
- **Cluster:** C primary, A secondary (BG), J secondary (plasticity).
- **Prerequisites:** C.52 (VTA), C.56 (tonic/phasic), J.x (eligibility, STDP).
- **Citation:** Kandel 6e Ch 43 pp 1068-1069 (Fig 43-2); Schultz, Dayan, Montague 1997.
- **Behavioral validation:** Three-trial paradigm: (a) unexpected reward → burst at reward; (b) trained CS+R → burst at CS, no burst at R; (c) trained CS but R omitted → dip at R-expected time. Currently only (a) is faithfully reproduced.
- **Supplemental — the single most important RPE augment:** HS98 is the **direct experimental validation criterion** the simulator could replicate. Two monkeys, novel-picture-pair learning trials. (i) **Cue-shift across learning**: 50% of DA neurons activated by reward during initial learning trials; activations dropped to 12% in familiar trials *only after the 2-of-4 learning criterion was reached*. Mean reward activation peaks at ~193% above baseline in trials 1–2 with novel pictures, declining toward 90–110% as task is learned (HS98 pp. 305–306, Figs. 3–5). The transfer is *graded* with learning rate, *not binary* — slow-learned pairs retain reward responses for tens of trials, fast-learned pairs lose them within ~2 trials. (ii) **Reward-omission dip**: 70% of neurons (28/40) showed a depression at 99 ± 29 ms after the time the reward would have been delivered, lasting 401 ± 36 ms (HS98 p. 305, Fig. 6a). (iii) **Temporal prediction error**: when reward delivered 0.5 s late, neurons depressed at the *original* reward time AND activated at the *new* reward time; when delivered 0.5 s early, neurons activated at the new time but did NOT depress at the original time (the early arrival cancels the prediction). HS98 pp. 305–306, Fig. 6b. **Validation criterion for the simulator**: in a `g11_bg_runner` variant, instrument the dopamine pool firing rate against a 2-cue Pavlovian schedule. **Currently the project reproduces only sign (i, partially) — it lacks the cue-shift transfer** because the predictive cue itself never acquires DA-burst-evoking power. This is the canonical signature; reproducing it requires the value-function critic of an actor-critic architecture (see new entry C.30 Actor-Critic mapping). Schultz98 §"Temporal difference learning" (Eq. 6, p. 12) gives the math: the cue-shift falls out of `r̂(t) = r(t) + γP(t) − P(t−1)` automatically once a learned prediction `P(t−1)` exists.
- **Supplemental — algorithmic mapping for adaptive-DA and surprise-LR-boost:** Schultz98 Eq. 6a (p. 12) is `r̂(t) = r(t) − P(t−1)`, the effective reinforcement at reward time. The project's `--adaptive-da --adaptive-da-ema-decay-negative 0.7` sets up an **asymmetric per-action `P(t−1)`** by EMA: the slow-positive (τ~10) tau is the analog of `P(t−1)` settling toward `r(t)` after consistent delivery (= r̂→0, "commit"); the fast-negative (τ~3) tau is the analog of `P(t−1)` collapsing fast when `r(t)` drops (= negative r̂, "explore"). This is **functionally the Rescorla-Wagner / TD(0) update with an asymmetric learning rate on the value estimate**. The asymmetry is biology-grounded: Schultz98 p. 6 reports DA depressions on omission are sharper and more transient than the buildup of activations on acquisition. Spelling this out in the catalog gives the AI/RL audience the algebraic isomorphism they need.
- **Supplemental — the RPE is specifically a UTILITY prediction error (Schultz16-NRN, Schultz16-JNT):** the canonical refinement that Schultz16 makes to the 1997/1998 RPE statement is that the second component codes **economic utility u(x)**, not raw reward magnitude. Concretely, the dopamine Component 2 follows the **nonlinear curvature of the behavioral utility function** measured by certainty-equivalent fractile procedures with binary equiprobable gambles (Schultz16-NRN Fig. 4c–d, p. 8; Schultz16-JNT Fig. 2, p. 684). The function is **convex (risk-seeking) at low amounts, concave (risk-averse) at high amounts** — an inflected utility function (Stauffer et al. 2014 cited Schultz16-JNT p. 685). DA tracks this curve, not a linear reward axis. Mapping to the project: the simulator's `current_reward_signal` is a linear scalar — there is no nonlinear utility transformation between physical reward and RPE. For most simulator tasks this is fine because rewards are binary or near-binary. But if the project ever runs **risk-sensitive choice tasks** (the obvious next step beyond moving-goal navigation), modeling DA as RPE-on-`reward_amount` will systematically underestimate phasic activations at low amounts and overestimate them at high amounts; a `utility_fn(reward)` bottleneck before computing the RPE would be the principled fix and is a one-line bridge change.
- **Supplemental — temporal-prediction-error component (Schultz16-NRN p. 4):** the initial detection component is itself "sensitive to the time of stimulus occurrence and thus codes a temporal-event prediction error" (Schultz16-NRN p. 4). This is the **temporal** axis Schultz98 §"Temporal difference learning" derives mathematically. The project's `current_reward_signal` event-trigger contains no model of *when* a reward should arrive, so a temporally-late-but-otherwise-correct reward generates the same teaching signal as a temporally-perfect one. A phase-dependent reward expectation (e.g. the dwell-time in `g11_bg_runner.moving_goal`) could supply this for free — the omission-dip and late-arrival depression-then-activation patterns from Schultz98 Fig. 6b would then be reproducible.

### C.23 Heterogeneous DA Subpopulations — reward, aversion, salience VTA cells

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.59]*

- **System:** Subpopulations within A10/VTA distinguished by afferents, targets, and response polarity.
- **Biological role:** Not all DA neurons are RPE encoders. Some respond to BOTH reward and aversion (salience); some preferentially to reward; some to aversion only; some show inverted (reward-suppressed, aversion-activated) profiles. Anatomically these correspond to distinct projection targets — e.g. medial VTA → mPFC tends salience-coded; lateral VTA → NAc lateral shell tends reward-coded.
- **Sim status:** missing — flagship has a single homogeneous DA population. The cheat #5 v3.1 / v4 cross-projection failures may be related: a single broadcast DA cannot supply the differentiated teaching signals that biology distributes across subpopulations.
- **Cluster:** C primary, A secondary, O secondary.
- **Prerequisites:** C.52 (VTA), C.58 (RPE).
- **Citation:** Kandel 6e Ch 43 pp 1068-1069.
- **Behavioral validation:** Single-unit recording in identified VTA cells with retrograde tracing → diversity of stimulus-response profiles correlated with projection target.
- **Supplemental — Schultz16 narrows the diversity claim:** Schultz16-NRN §"Dopamine diversity" (pp. 12–13) argues that the **phasic** RPE responses of dopamine neurons are remarkably similar across the population, with only **graded — not categorical — differences** between medial/lateral and dorsal/ventral midbrain subgroups (Schultz16-NRN p. 12; Schultz16-JNT p. 681 "70–90% of dopamine neurons, is very similar in latency across the dopamine neuronal population, and shows only graded rather than categorical differences"). Apparent categorical differences (e.g. Matsumoto & Hikosaka's reward vs salience subpopulations) are reinterpreted as **varying sensitivities to Component 1 drivers** (physical intensity, reward generalization, reward context) rather than as anatomically distinct value vs salience encoders. **Diversity in DA neurons is real, but it is overwhelmingly in the slow / tonic / non-phasic dimensions — morphology, neurochemistry, projection target, slow ramping with risk or reward proximity** — not in the phasic teaching signal. Implication for the project's cheat-5 v3.1 / v4 cross-projection failures: a single broadcast phasic DA may actually be the **biologically faithful** abstraction; the missing differentiation is plausibly in tonic-DA + projection-target-specific receptor effects, not in the phasic RPE itself.

### C.24 Dopamine in Aversion — DA also encodes salience and warning

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.60]*

- **System:** Subpopulation of VTA DA neurons activated by aversive stimuli (foot shock, air puff).
- **Biological role:** Counter-evidence to the simple "DA = pleasure" / "DA = reward only" view. Aversion-activated DA neurons may signal *salience* rather than reward valence. Dopamine-depleted rodents (6-OHDA) and DA-synthesis-knockout mice still show hedonic taste reactions to sucrose — so DA is NOT a hedonic signal, it's a learning/teaching signal.
- **Sim status:** partial — current scheme uses signed scalar reward (positive and negative). Negative reward decreases weights via STDP × DA, which IS a form of aversion encoding. But the "salience-only" subpopulation (responds to both valences) is not modeled.
- **Cluster:** C primary, O secondary.
- **Prerequisites:** C.58, C.59.
- **Citation:** Kandel 6e Ch 43 pp 1068-1069.
- **Behavioral validation:** Recording during foot-shock or aversive Pavlovian conditioning shows DA increase in subset of VTA neurons, dip in others.
- **Supplemental:** Schultz98 (pp. 3–4, "Activation by primary appetitive stimuli", "Activation-depression with response generalization", and §"Effective stimuli for dopamine neurons" pp. 6–7 Fig. 5) reports only ~14% of DA neurons activated by primary aversive stimuli and only 11% activated by conditioned aversive stimuli, and even those tend to be the same neurons that respond to rewards. Generalization-based activations are smaller in magnitude than reward-conditioned-stimulus activations, and dopamine activations to aversive cues *fail to generalize when behavior is avoidance* (Mirenowicz & Schultz 1996). The interpretation in Schultz98 is that DA is **predominantly an appetitive/alerting signal**, not a salience signal — the residual aversive activations reflect physical similarity to appetitive cues rather than valence-free salience. This contradicts a strict "DA = salience" reading and supports the project's signed-reward design (positive vs negative `current_reward_signal`) **with the caveat** that the magnitude asymmetry biology reports (much stronger appetitive than aversive responses) is not currently modeled — phasic responses to negative reward should be *weaker* than phasic responses to positive reward of equal magnitude, not symmetric.
- **Supplemental — Schultz16 re-argues this strongly: aversive activations are likely PHYSICAL IMPACT, not aversiveness:** Schultz16-NRN §"Unlikely aversive activation" (pp. 11–12) and Schultz16-JNT §"Confounded aversive activations" (pp. 682–683) sharpen the case made in Schultz98. The strongest evidence: Fiorillo et al. 2013a,b independently varied physical intensity and aversiveness of bitter solutions (denatonium). At low concentration (1 mM), denatonium produced **substantial dopamine activation**; at 10× concentration (10 mM, much more aversive) the activation was **replaced by depression** — the depression "undercutting" the activation reflects negative value, while the activation itself reflects physical liquid-impact (Schultz16-JNT Fig. 1c, p. 682). Concretely: positive correlations of DA with physical intensity, **negative correlations with aversiveness**. Schultz16's interpretation: "phasic dopamine activations by aversive stimuli seem to constitute the initial, unselective dopamine response component driven by physical impact, and possibly boosted by reward context and reward generalization, rather than reflecting a straightforward aversive response" (Schultz16-NRN p. 12). For the project's signed-reward design: if you want biologically faithful aversive coding, the **negative `current_reward_signal` should produce a phasic DEPRESSION below tonic baseline, not a sign-flipped activation** — these are observably different at the postsynaptic plasticity level (a depression below baseline produces opposite-sign LTD via D2 vs the LTP-via-D1 produced by an activation). Currently the simulator's signed-scalar reward conflates these.

### C.25 NAc cAMP-CREB Pathway Adaptation — chronic-DA homeostasis

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from C.61]*

- **System:** Nucleus accumbens medium spiny neurons; cAMP → PKA → CREB intracellular cascade.
- **Biological role:** Repeated drug exposure (or chronic strong reward) acutely *suppresses* cAMP via Gi-linked D2/μ-opioid/CB1 receptors. Cells adaptively *upregulate* adenylyl cyclase and CREB to restore baseline activity (tolerance). On drug removal, the upregulated pathway is unopposed → withdrawal hyperactivity. This is the molecular substrate of **reward tolerance**.
- **Sim status:** missing — flagship has no second-messenger modeling. Synaptic scaling (homeostasis) provides a coarse functional analog (sets activity setpoints) but operates on firing rate, not on RPE setpoint. [Could matter for long-horizon RL: project agents may not show realistic tolerance/sensitization to persistent reward.]
- **Cluster:** C primary, J secondary, O secondary.
- **Prerequisites:** C.58 (DA), J.x (homeostasis).
- **Citation:** Kandel 6e Ch 43 pp 1074-1075 (Fig 43-5).
- **Behavioral validation:** Repeated morphine → reduced cAMP/PKA acutely, gradually restored despite continued drug, then *elevated* on naloxone — measurable as PKA-dependent phosphorylation timecourse.
- **Supplemental — RL theory mapping:** S&B Ch 11.3 "R-Learning and the Average-Reward Setting" (pp. 260–262) defines the **average-reward MDP formulation** in which value functions are computed *relative to* an estimated long-run average reward `r̄(π)`: `v_π(s) = E[Σ(R_{t+k} − r̄(π))]`. Updates: `δ = R − R̄ + max_a Q(S′,a) − Q(S,A); R̄ ← R̄ + βδ` when greedy. **This is the algorithmic homologue of NAc cAMP/CREB tolerance**: subtracting a slow-tracked baseline reward level from instantaneous reward is exactly what the CREB-mediated AC upregulation does at the molecular level. The "withdrawal hyperactivity" phenomenon is the signature of `R̄` over-tracking when the baseline reward source is removed. If the project ever wants to model long-horizon reward homeostasis without molecular machinery, R-learning's `R̄` provides a one-line abstraction: declare a `reward_baseline_ema` neuromodulator with `manual` production driven by a tau~minutes EMA of `current_reward_signal`, and subtract it from reward before plasticity. This unifies C.25 (biology) with average-reward RL (algorithm).

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
- **Supplemental — Schultz16 reconciliation with incentive-salience theory:** Schultz16-NRN §"Salience" (pp. 12–13) explicitly addresses Berridge's incentive-salience hypothesis. Schultz argues the two views **are not mutually exclusive**: "incentive salience concerns dopamine's influences on behaviour, whereas prediction-error coding concerns the properties of the dopamine prediction-error signal itself, which can have many functions. Indeed, a prediction-error signal can support both learning and efficient performance" (Schultz16-NRN p. 13). The Component-1/Component-2 split clarifies the dispute: Component 1 (detection / salience) is what Berridge-style incentive-salience theories grasp at, Component 2 (value / utility RPE) is what TD-learning theories grasp at, and both are real properties of the same neuronal burst. The project does not currently model "wanting" as separable from "learning" — both are mediated by the same `current_reward_signal × eligibility_trace` machinery. Adding a Component-1-style detection pulse that drives transient performance enhancement (membrane depolarization, attentional gain) without writing eligibility traces would be a clean implementation of the Berridge "wanting" dimension on top of the current RPE substrate.

## Cluster N — Sleep, Arousal, Replay (project-critical for Ch 44)

### N.15 Theta-gamma cross-frequency coupling — multiplexed cell-assembly buffer
- **System:** Hippocampal CA1/CA3 + neocortex; theta carrier at 4–12 Hz with gamma (40–100 Hz) amplitude phase-locked to the theta cycle. Each theta cycle hosts 7–9 nested gamma cycles, each carrying a distinct cell-assembly.
- **Biological role:** Bz Cycle 12 (pp. 350–353, Fig. 12.6, esp. Lisman & Idiart 1995) proposes that the theta cycle + nested gammas implements a **time-multiplexed working-memory buffer**: each gamma cycle carries one item-assembly, the theta period sets the buffer span (≈7±2 items, matching Miller's "magical number"), and re-firing on subsequent theta cycles maintains the items in active memory. Shifting (rather than repeating) the assembly sequence on successive theta cycles encodes a sequence into episodic memory. This is the mechanistic substrate proposed for short-term memory's classical capacity limit *and* the proposed bridge from STM to episodic LTM. **Important discrepancy with Kandel framing:** Kandel Ch 52 treats working memory and episodic memory as separate cluster-G and cluster-D phenomena; Bz/Lisman frame them as the *same* theta-gamma multiplex, distinguished only by whether the gamma sequence repeats (STM) or shifts (episodic encoding). For a sim that builds both PFC working memory (project already has) and hippocampal episodic encoding, this means a single theta-gamma generator could in principle drive both.
- **Sim status:** missing. The project has neither theta nor gamma generators in the locale path. Adding nested oscillators is straightforward in the NM framework (sinusoidal `excitability_drive` at theta + a faster modulator with theta-phase-modulated amplitude). Validation would be measuring assembly count per theta cycle and showing capacity ~7±2.
- **Cluster:** D primary, G primary (working memory), I secondary (oscillations), J secondary (sequence encoding).
- **Prerequisites:** D.18 (theta), I.* (gamma — currently exists as the FS interneuron gamma in the existing sim), G.* (PFC working memory).
- **Citation:** Bz (2006) Cycle 12 pp. 350–353, Fig. 12.6 p. 352; Lisman & Idiart (1995); Bragin et al. (1995); Chrobak & Buzsáki (1998); Cycle 6 (synchronization) for the math.
- **Behavioral validation:** (a) Recover Lisman-style multiplex by counting distinct gamma-cycle-locked assemblies per theta cycle in a working-memory task — should be ≤9; (b) capacity-limit test: introduce 5, 7, 9, 11 stimuli to PFC and measure recall accuracy — should drop sharply past the gamma/theta ratio.

### N.16 Sharp-wave–ripple as self-organized hippocampal event (developmental + intrinsic)
- **System:** CA3 recurrent network — *not* dependent on extra-hippocampal input. The same network can produce SWRs in vitro, in transplants, and in the developing brain before EC inputs are mature.
- **Biological role:** Bz Cycle 12 (p. 344, citing Leinekugel et al. 2002) makes a distinction the catalog currently misses: SWRs are not "an NREM sleep replay event" generated by the sleep network — they are an **intrinsic CA3 self-organized event** that *also* happens to occur preferentially during NREM. Specifically, sharp waves are present in fetal/neonatal hippocampus before any sleep-stage architecture exists; they persist in transplanted hippocampi cut off from all afferents; and they emerge in CA3 slices in vitro. The NREM-sleep timing is selection-by-context, not generation-by-context. **Project implications:** (1) the existing project framing of replay as "scheduled during NREM phase" inverts the causality — biologically, SWRs would emerge spontaneously from a CA3 recurrent network whenever its drive crosses a self-organization threshold, and the role of NREM is to *gate* (via slow oscillation up-states + spindle troughs) which SWRs have downstream effect; (2) this means the simulator's SWR generator should live in the CA3 region's intrinsic dynamics, not in a sleep-stage scheduler. Specifically: a CA3 recurrent network with sufficient density and adaptation should produce population bursts spontaneously; gating those bursts by an NREM phase variable produces the empirical timing distribution.
- **Sim status:** missing — and the framing matters more than the implementation. Currently the project's `bridge.py` schedules replay events; biologically the events should be intrinsic with NREM as the gate.
- **Cluster:** N primary, D primary, J secondary.
- **Prerequisites:** D.05 (CA3 recurrents) — must be a real recurrent attractor, not just a sparse pool. N.05 (slow oscillation) for gating.
- **Citation:** Bz (2006) Cycle 12 pp. 343–351, esp. p. 344 on developmental + transplant evidence (Leinekugel et al. 2002; Buzsáki et al. 1987; Buzsáki 1986, 1989, 1996, 1998).
- **Behavioral validation:** (a) Disable the NREM scheduler; the CA3 recurrent network should still produce intermittent sharp-wave-like population bursts during quiet rest. (b) Re-enable the slow-oscillation gate; bursts should preferentially fall on Up-state troughs *without* the scheduler imposing this — the gating should be passive (the burst probability depends on local network excitability, which already follows the slow oscillation).

### N.17 Awake replay during behavioral pauses — online deliberation, not just consolidation
- **System:** Same CA3-CA1 ripple machinery as NREM SWRs, but occurring during **quiet wakefulness** at choice points, reward sites, and brief immobility periods.
- **Biological role:** Bz Cycle 12 (pp. 348–351, citing the Foster & Wilson 2006 reverse-replay literature and Pfeiffer & Foster 2013 forward-replay-of-future-trajectory) reports that ~50% of all SWRs occur during waking immobility, not sleep. The content of awake SWRs is biased toward (a) the trajectory just completed (reverse replay, often after reward — proposed credit-assignment role), and (b) candidate trajectories *about to be taken* (forward replay, proposed deliberative-planning role). Awake replay's behavioral relevance is now better-supported than NREM replay for short-term spatial decision-making — a major reframe vs the catalog's current "replay = consolidation during sleep" framing inherited from Stickgold/Wilson 1990s work.
- **Sim status:** missing as a project-distinct mechanism. The sleep-replay infra is gated to NREM phases; awake-replay would require firing the *same* generator on the same agent during waking immobility (e.g., when reward is delivered or `g11` agent reaches a goal). This is one of the higher-value additions if the project wants replay to influence online behavior, not just offline weight changes — and it directly addresses the "replay-content quality bottleneck" in a *behavioral* loop: forward-replay before action + R-STDP on the actual outcome could provide dramatically better credit assignment than waiting for sleep.
- **Cluster:** N primary, D primary, B (BG action selection — replay biases choice), C (DA — reward-triggered reverse replay).
- **Prerequisites:** N.16 (intrinsic SWR generator), D.24 (theta sequences during waking, supplying replay bias).
- **Citation:** Bz (2006) Cycle 12 pp. 348–351; Foster & Wilson (2006) reverse replay; Pfeiffer & Foster (2013) forward trajectory replay; Karlsson & Frank (2009).
- **Behavioral validation:** (a) Detect SWR-like population bursts in CA3 during `g11` agent immobility at goal sites (post-reward); they should preferentially replay the just-completed trajectory in reverse. (b) Detect SWR-like bursts at choice points during waking; they should preferentially replay candidate forward trajectories. (c) Disabling awake-SWRs (gate them off in waking, leave NREM SWRs intact) should impair choice-point performance specifically — the most decisive behavioral test of an awake-replay function.

### N.18 NREM hierarchical nesting — slow oscillation > spindle > ripple
- **System:** Whole-brain NREM rhythm hierarchy: cortical slow oscillation (0.5–1 Hz) > thalamocortical spindle (10–16 Hz) > hippocampal SWR (140–200 Hz). Each rhythm phase-modulates the next-faster rhythm's amplitude.
- **Biological role:** Bz Cycle 12 (pp. 343–353) and the supporting cross-frequency literature establish the canonical NREM consolidation frame: cortical Up-states permit spindles (which require interneuron drive); spindle troughs permit hippocampal SWRs (the briefly hyperpolarized cortical state allows the SWR-driven CA3→cortex packet to land on a ready-to-receive cortex). This three-level nesting is tightly correlated with overnight memory consolidation, and disrupting any one level (slow-osc disruption via TMS, spindle disruption pharmacologically, or SWR disruption electrically) impairs consolidation by similar magnitudes — strong evidence each level is necessary, not just correlated. **Discrepancy with Kandel framing:** Kandel discusses each rhythm separately (slow oscillation in N.05, spindles in N.06, ripples in N.07) without explicit nesting. Bz frames them as a single mechanism with three levels.
- **Sim status:** missing as an integrated structure. Each component is also missing individually (per current N.05–N.07 entries). However, the *integrated* framing matters for the project: a single phase-modulated drive (slow → spindle → ripple, each gating the next) is architecturally simpler than three independent generators with cross-coupling rules. This is the natural sim implementation if the project decides to build NREM properly.
- **Cluster:** N primary, J secondary (consolidation), I secondary (oscillations).
- **Prerequisites:** N.05, N.06, N.07 / N.16.
- **Citation:** Bz (2006) Cycle 12 pp. 343–353, Fig. 12.6 p. 352; Sirota et al. (2005); Mölle et al. (2002); Siapas & Wilson (1998) for hippocampal-cortical spindle-ripple coupling.
- **Behavioral validation:** (a) During simulated NREM, measure the joint phase distributions: SWR probability should peak at the trough of the spindle, which itself peaks at the trough/early-up of the slow oscillation. (b) Selectively suppress one level at a time (e.g., zero spindles by disabling TRN T-type rebound; leave slow-osc and SWRs intact) — overnight memory-consolidation-equivalent should drop with each individual disruption.

### N.19 Gamma binding-by-synchrony — ING vs PING mechanisms
- **System:** Cortical gamma (40–100 Hz). Two computational variants: **ING** (interneuron-network gamma) — pure interneuron oscillation driven by tonic excitation, requires no pyramidal participation; **PING** (pyramidal-interneuron gamma) — pyramidal cells lead by a few ms, driving interneurons that in turn pace the population. Both are real and coexist; the dominant mode varies with state and brain region.
- **Biological role:** Bz Cycle 9 (pp. 231–261) is the definitive treatment. Beyond the catalog's existing gamma entry (Cluster I), Bz makes three additions central to cognitive function: (a) gamma's frequency is set by the **decay time of GABAA inhibition** (~10–25 ms → 40–100 Hz; Bz pp. 248–250), explaining why gamma is roughly the same frequency everywhere in cortex despite enormously varied excitatory architecture; (b) gamma synchrony provides a **temporal window for feature binding** — neurons firing within the same gamma cycle are "co-grouped," neurons offset by a half-cycle are segregated (Bz pp. 250–260, esp. Fig. 9.3 p. 248 on STDP-window match); (c) the **gamma cycle and the STDP window are matched in duration** — this is not coincidence but the substrate that lets gamma-bound assemblies actually form synaptic links. **Discrepancy with Kandel framing:** Kandel discusses gamma as a measurement / signature of attention; Bz frames it as the *computational mechanism* by which transient cell assemblies are bound and made eligible for plasticity-mediated storage.
- **Sim status:** partial. The existing simulator's gamma validation (Bi & Poo / `gamma-oscillations` benchmark) confirms frequency-band production in the FS interneuron network. The functional role — gamma-bound cell assemblies + STDP-window matching — is not specifically tested. Adding a binding-task validation (transient gamma-coherent groups with within-group STDP potentiation > between-group) would close this entry.
- **Cluster:** I primary, J primary (STDP-window match), D secondary (theta-gamma multiplex via N.15).
- **Prerequisites:** I.* (existing FS interneurons + GABAA), J.* (STDP).
- **Citation:** Bz (2006) Cycle 9 pp. 231–261; Whittington et al. (1995, 2000); Wang & Buzsáki (1996); Traub et al. (1996, 1999); Bibbig et al. (2001) for STDP-window match; Kopell (2000) for math.
- **Behavioral validation:** (a) Confirm decay-time-of-inhibition controls oscillation frequency (already tested in `gamma-oscillations` benchmark); (b) drive two non-overlapping pyramidal sub-populations into separate gamma cycles; STDP should strengthen within-group synapses but not between-group; (c) under PING regime, pyramidal lead-time ahead of interneurons should be ~2–5 ms (Bz citing Csicsvari et al. 1998).

---

## Cluster D — additions

### D.21 Cognitive-map theory — hippocampus as Euclidean spatial framework for episodic binding
- **System:** Whole-hippocampus framework hypothesis (not a single circuit). Locale system = HC + perforant path + EC; outputs via subiculum and fornix to motor and arousal targets.
- **Biological role:** O&N's central claim (1978, Introduction pp. 1–4 + Ch 13–14): the hippocampus implements an *a priori* Euclidean spatial framework — an absolute, allocentric, observer-independent metric onto which sensory items and events are mapped. Items inhabit *places*; places interrelate via the framework, not via shared sensory features. This is what makes the system flexible: novel inferences (shortcut taking, transitive choices, latent learning) drop out of the framework's geometry, not out of stored sensorimotor associations. In humans, the same machinery extends to non-spatial *temporal* contexts (events-at-times) and is hypothesized to underpin episodic memory (Ch 14, pp. 384–390). The locale system is contrasted with **taxon systems** (extra-hippocampal, egocentric, route- and stimulus-response based) which the catalog treats separately in D.22.
- **Sim status:** missing as a *theory-level* commitment. The project has place-cell-like activations (D.06) but no Euclidean framework primitive, no evidence of allocentric remapping, and no "items located in places" relational primitive. D.21 is what *all of D.06–D.20 are predicting jointly*: a successful trisynaptic-loop + theta + SWR sim should produce O&N's cognitive-map signatures (allocentric place fields, remap on environment change, shortcut behavior, partial-cue completion), and D.21 is the rubric for evaluating whether it has.
- **Cluster:** D primary, G secondary (working memory bridge for episodic).
- **Prerequisites:** D.03 (trisynaptic loop), D.06 (place cells), D.17 (remapping), D.18 (theta), D.19 (SWRs).
- **Citation:** O&N (1978) Introduction pp. 1–4; Ch 1 pp. 5–61 (philosophical background); Ch 2 pp. 62–101 (cognitive-map model); Ch 13 pp. 374–380 (locale long-term memory); Ch 14 pp. 381–411 (human extension). Bz (2006) Cycle 11 pp. 277–333 endorses and updates.
- **Behavioral validation:** A successful D.21 implementation should reproduce, on a single sim instance, *all* of: (1) allocentric place fields that survive cue removal/rotation (D.06 supplemental); (2) global remapping between two distinct environments (D.17); (3) shortcut taking through never-traversed Euclidean paths after free exploration; (4) latent learning (no-reward exploration produces map; map drives behavior on later reward introduction). The shortcut and latent-learning tests do not exist in the current `g11` runner and would be the strongest D.21 validation.

### D.22 Locale vs taxon systems — dual-system architecture for navigation and learning
- **System:** Locale = hippocampus + EC + subicular complex. Taxon = striatum (S-R habits, aligns with current Phase B BG cascade), parietal/temporal cortex (egocentric/object-feature recognition), neocortical association areas. Two systems run in parallel and compete for behavioral control.
- **Biological role:** O&N (Ch 2.3, pp. 89–101) propose that learned behavior is supported by two *qualitatively different* systems with distinct properties (summarized in O&N Table 2, p. 91): the **locale** system is map-based, allocentric, all-or-none on encoding (one-trial), context-dependent, flexibly recombinable, and not subject to extinction; the **taxon** system is route- and response-based, egocentric, incremental on encoding, context-bound, inflexible, and decays with disuse. After hippocampal damage, animals retain taxon-system performance and lose locale-system flexibility — predicting a precise behavioral profile that the lesion literature (O&N Chs 5–13) largely confirmed. This is the *original* dual-system theory of memory (predating Squire's declarative/procedural by years).
- **Sim status:** **partial — and surprisingly aligned with the project's existing architecture.** The Phase B BG cascade (per-action cortex → D1/D2 → GPi → thal → motor) is functionally a **taxon system**: stimulus-response, incremental via STDP+R-STDP, action-bound, not flexibly recombinable. The "hippocampus" stub is a placeholder for the **locale system**. The project's current best flagship ("PFC + hippocampus + perception arc + curriculum") is doing what O&N predicted: locale-system signals provide context that biases taxon-system action selection, with hippocampal lesion (= disabling `--hippocampus` flag) collapsing performance to a flatter taxon-only baseline. **Strong project-relevant insight:** the catalog's current framing treats the BG cascade and the hippocampus as additive perception+action modules; O&N would frame them as *competing* memory systems, and the most informative experiments would be ones where they *disagree* (e.g., goal location changes mid-trial — locale should re-acquire fast, taxon should re-acquire slow).
- **Cluster:** D primary, B (striatum) primary, G secondary.
- **Prerequisites:** D.21 (cognitive map), Phase B BG cascade.
- **Citation:** O&N Ch 2.3 pp. 89–101 (esp. Table 2 p. 91); Ch 13 pp. 374–380 (locale long-term storage); Chs 5–13 lesion review for predicted dissociations.
- **Behavioral validation:** Mid-trial environment change on `g11_bg_runner --moving-goal --hippocampus` should produce *faster* re-acquisition than `--moving-goal` (no hippocampus), exactly mirroring the rat literature on hippocampal vs caudate lesions. A failure to find this dissociation is evidence the project's hippocampus stub isn't yet a *locale* system in O&N's sense.

### D.23 Misplace system — hippocampal novelty detection driving exploration
- **System:** O&N propose CA1 "displace/misplace" units that fire when stimuli expected in a place are absent or new stimuli appear. Output via fornix → septum → brain-stem motor programs activates exploration patterns (and at extreme drives fear/freezing).
- **Biological role:** O&N Ch 2 (pp. 96–101) and Ch 4.7.2 (pp. 195–209) describe two operations of the locale system: the **place system** (fires when stimuli in a place match the stored map) and the complementary **misplace system** (fires when stimuli mismatch — novel object, missing object, novel arrangement). Misplace output drives investigative exploration, *reciprocally* updating the map with the new arrangement via one-trial Hebbian-LTP-mediated incorporation (O&N pp. 230, 244–247). This is the original hippocampal novelty-detection theory and the conceptual ancestor of subsequent novelty-N400 and CA1-comparator literature.
- **Sim status:** missing. The project has no novelty signal, no hippocampal-driven exploration bonus, no map-update-on-mismatch mechanism. Could be implemented as a CA1 region whose excitatory drive is modulated by the *negative correlation* between current sensor input and CA3-recalled pattern; high CA1 firing would gate a `noise-injection` or `exploration-bonus` neuromodulator. **Discrepancy with Kandel framing:** Kandel attaches novelty detection to the perirhinal cortex / EC-III "match-mismatch" comparator; O&N attach it firmly to CA1 itself. The sim community has been split on this for decades — the simulator is in a position to test both.
- **Cluster:** D primary, O secondary (exploration neuromodulator), C secondary (DA novelty bonus).
- **Prerequisites:** D.04 (EC-III direct path), D.05 (CA3 autoassociator).
- **Citation:** O&N Introduction p. 3; Ch 2.3 pp. 89–101; Ch 4.7.2 pp. 195–209 (single-unit data); Ch 4.8 pp. 217–230 (proposed mechanism). Cf. Kandel Ch 54 attribution to perirhinal/EC-III.
- **Behavioral validation:** (a) Single-cell: identify simulated CA1 cells whose firing inversely correlates with cue-pattern stability in the current place; (b) population: removing one of N landmarks should transiently elevate this CA1 sub-population without affecting baseline place cells; (c) behavioral: agent should preferentially approach the changed-cue region after such manipulation (latent novelty-driven exploration).

### D.24 Theta-paced sequence compression — behavioral seconds → STDP milliseconds
- **System:** CA1 + CA3 place-cell ensembles during theta-associated locomotion; the same neurons that produce phase precession (D.18 supplemental).
- **Biological role:** Within a single ~125-ms theta cycle, a population of place cells with overlapping fields fires in the same temporal order they will be visited in the next several seconds of behavior, but compressed ~20× (Bz Cycle 11 pp. 316–323, esp. Fig. 11.14 p. 317 and Dragoi & Buzsáki 2006). This compression brings non-adjacent positions on a route into the spike-timing-dependent-plasticity window (~10–40 ms), so STDP can store *higher-order* sequence relationships, not just immediate-neighbor ones — an essential precondition for SWR replay to have content (D.19, N.07 supplemental).
- **Sim status:** missing; this is a candidate **most-impactful** addition for the project's hippocampus arc. The project has STDP and would have place cells in a real trisynaptic loop. Adding a theta drive (sinusoidal `excitability_drive` from a `septum` neuromodulator at 8 Hz) and an asymmetric input scheme on perforant-path inputs (advancing-phase drive as the agent enters a field — established mechanism class in Bz pp. 318–322) would produce phase precession + theta sequences as emergent behavior. This in turn provides the bias structure for SWR replay during quiet rest (per AUGMENT D.19 supplemental), addressing the project's named "replay content quality" bottleneck without changing the replay generator.
- **Cluster:** D primary, J primary (it's about STDP-window plasticity), G secondary.
- **Prerequisites:** D.03 (trisynaptic), D.06 (place cells), D.18 (theta drive), J.* (STDP, already implemented).
- **Citation:** Bz (2006) Cycle 11 pp. 313–323, Figs. 11.13–11.14; Dragoi & Buzsáki (2006); originally O'Keefe & Recce (1993) for phase precession; Skaggs et al. (1996) for theta sequences.
- **Behavioral validation:** (a) Phase precession of simulated place cells against septal-theta LFP proxy; (b) within-theta-cycle ordering of spikes from cells with adjacent fields matches the order of the agent's next several positions; (c) measure pairwise STDP-window overlap between non-adjacent place-cell pairs before and after theta-paced exploration — should rise with theta on, stay near baseline with theta off; (d) SWR replay content quality (sequence-similarity to recent trajectories) should be *higher* when the prior waking phase had theta enabled.

---

---

## Cluster D — Hippocampus & sequence learning

**20 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

### D.01 Episodic memory — encoding / storage / retrieval / consolidation cycle

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.50]*

- **System:** Medial temporal lobe (hippocampus + perirhinal + parahippocampal + entorhinal) interfacing with frontoparietal networks and association cortices.
- **Biological role:** Binds multimodal items into events and events into episodes via temporal/spatial context (Tulving 1972; Eichenbaum/Cohen relational memory). Consolidation transforms labile traces into durable, distributed cortical representations through hippocampal–neocortical dialogue; re-encoding may follow each retrieval (reconsolidation).
- **Sim status:** missing as a system — sleep-replay infrastructure exists (NREM scaffolding) but no episodic encoder, no separate "labile vs consolidated" trace bookkeeping, no relational binding API. Phase-tagged plasticity gating (`set_plasticity_gate`) could express consolidation phases but no runner uses it.
- **Cluster:** D primary, N secondary (sleep-driven consolidation), G secondary.
- **Prerequisites:** D.51 (HC microcircuit), D.55 (place cells), N.* (replay).
- **Citation:** Kandel 6e Ch 52 pp 1296–1302.
- **Behavioral validation:** Anterograde amnesia for new associations after MTL lesion with preserved working/skill memory (H.M.); retrieval-time hippocampal–cortical reactivation (iEEG word-pair studies).
- **Supplemental:** O&N (Ch 13–14, pp. 374–411) frame episodic memory as the *human* extension of the rat locale system: the same map machinery, but now indexed by *time* in addition to space. The rat hippocampus stores places; the human hippocampus stores places-at-times, which O&N argue is the substrate of Tulving's episodic memory (O&N pp. 384–390, esp. footnote on p. 390 noting parallel with Tulving 1972). Bz (Cycle 11, pp. 314–316) explicitly endorses this O&N-1978 reading and notes that O'Keefe later partially walked it back (O'Keefe 1999) — Bz disputes the walk-back and treats hippocampal *sequence coding* as the primitive that underlies both rat dead-reckoning and human episodic recall. Useful for the project: episodic memory = "place + time-on-theta-cycle"; the same simulator machinery (theta-organized place cells + SWR replay) potentially serves both.

### D.02 Relational binding / "memory space" — Eichenbaum–Cohen model

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.51]*

- **System:** Hippocampus proper (CA1 output), with perirhinal item streams and parahippocampal context streams converging.
- **Biological role:** Stores events as items-in-context, episodes as temporal sequences of events, and networks via overlapping events allowing flexible inference (e.g., transitive). Distinguishes overlapping episodes that share elements (same restaurant, different visits) without interference.
- **Sim status:** missing — no relational binding primitive. Place-cell-like encoding from learned-perception of landmark sensors is content-only, not item+context; no episode boundary detection or sequence-of-events memory.
- **Cluster:** D primary, G secondary.
- **Prerequisites:** D.55 (place cells), D.56 (sequence learning).
- **Citation:** Kandel 6e Ch 52 pp 1301–1302; Eichenbaum/Cohen 2014.
- **Behavioral validation:** Inference on overlapping experiences (transitive inference); selective deficit on configural learning after dorsal-HC lesion.
- **Supplemental:** O&N's locale system (Ch 2, pp. 89–101; Ch 13, pp. 374–380) is the *direct ancestor* of the Eichenbaum–Cohen relational-memory framework — items located in places, places linked into a unitary map, novel inferences supported by traversal of the map. O&N pp. 96–101 spell out the algebra ("place hypotheses" can be tested without reactivating any specific stimulus that was originally present). The catalog's current framing treats relational binding as primarily Eichenbaum's — supplemental note: O&N's "map" already provides this binding architecturally (each place node is an item-set within a spatial frame), so a faithful sim of D.02 may not need a separate "relational store" beyond the locale-system substrate of D.06/D.18.

### D.03 Trisynaptic pathway — EC layer II → DG → CA3 → CA1 (indirect)

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.52]*

- **System:** Entorhinal cortex layer II → perforant path → dentate gyrus granule cells → mossy fiber → CA3 pyramidal → Schaffer collateral → CA1 pyramidal → subiculum + EC deep layers (loop closure).
- **Biological role:** Three sequential excitatory stages with distinct functional properties at each: DG sparsifies, CA3 completes, CA1 outputs. Returns to deep EC for cortical broadcast.
- **Sim status:** missing — `sim/regions.py` allows declaring DG/CA3/CA1 as separate `BrainRegion`s and pathways with `density`, `weight_mean`, `plastic`, but no runner builds the trisynaptic loop. Current "hippocampus" is a single recurrent pool with place-cell-like place fields from landmark sensors.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** none — uses existing region/pathway primitives.
- **Citation:** Kandel 6e Ch 54 pp 1340–1342, Fig 54-1.
- **Behavioral validation:** Selective lesion at each stage produces distinct deficits (pattern separation, completion, output binding).
- **Supplemental:** O&N Ch 3 (pp. 102–140) and Ch 4.8 (pp. 217–230) propose specific computational roles for *each* trisynaptic stage that map cleanly onto current modeling: DG = sparse re-coder of EC inputs (O&N pp. 116–122 on perforant path imbrication and pp. 219–222 on the lamellar place-coding scheme); CA3 = autoassociative store with theta-driven sequential readout (pp. 222–227, esp. Fig. 29); CA1 = match/mismatch comparator between CA3-recalled pattern and direct EC-III drive (pp. 228–230). **Discrepancy with Kandel:** Kandel Ch 54 attributes "pattern completion" to CA3 generically; O&N argue more specifically that CA3's autoassociator is *sequential* — theta paces successive completions of adjacent places along a trajectory, not just one-shot pattern recall. This matters for the project's T1.A roadmap: a sequential CA3 attractor (theta-paced) is a different sim target from a Hopfield-style point attractor.

### D.04 Direct entorhinal pathway (temporoammonic) — EC layer III → CA1

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.53]*

- **System:** EC layer III axons targeting *distal* apical dendrites of CA1 (parallel to trisynaptic input arriving at proximal Schaffer dendrites).
- **Biological role:** Provides direct sensory context to CA1 in parallel with the indirect path. Distal/proximal segregation enables CA1 to compare current input against CA3-recalled pattern (a "match/mismatch" or novelty-detection function in some theories).
- **Sim status:** missing — would require multi-compartment CA1 or distinct excitatory pathways with different dendritic-zone effects (currently CA1 single compartment can only sum inputs).
- **Cluster:** D primary, I (channels — needs dendritic compartments).
- **Prerequisites:** D.52, multi-compartment neuron support.
- **Citation:** Kandel 6e Ch 54 p 1340.
- **Behavioral validation:** Direct-pathway lesion impairs novelty detection but spares pattern completion.
- **Supplemental:** O&N Ch 4.8.1(d) (pp. 228–230) is the original explicit proposal that CA1 *compares* CA3-recalled content (proximal Schaffer input) against direct EC-III input (distal apical dendrite). The "imbricated" termination pattern of EC-III axons across CA1 dendrites combined with theta-phase-shift along the apical trunk is presented as the mechanism by which CA1 selects which dendritic patch is "open" at any given theta phase — a biophysical match/mismatch operation (O&N Fig. 30, p. 229). This is *very* close to the project's existing plasticity-gate concept and suggests CA1 in the simulator might be implemented as two pathways sharing a single CA1 region, with phase-gated competition between them rather than two separate compartments.

### D.05 CA3 recurrent collaterals — autoassociative attractor substrate

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.54]*

- **System:** CA3 pyramidal cells with extensive recurrent excitatory connections among themselves; LTP-modifiable.
- **Biological role:** Implements pattern completion: partial cue activates an attractor that converges on the full stored pattern. Marr (1971) autoassociator. Pathologically prone to seizure (runaway recurrent excitation).
- **Sim status:** partial — `RegionPathway` from CA3 to CA3 with `internal_density>0` would create the recurrent substrate, but no runner does this and no test verifies attractor convergence on cue completion.
- **Cluster:** D primary, J secondary.
- **Prerequisites:** D.52.
- **Citation:** Kandel 6e Ch 54 pp 1342, 1360–1361.
- **Behavioral validation:** Partial-cue retrieval: stored "ABCDE" reactivated by partial "AB__" cue; lesion of CA3 recurrents impairs partial-cue recall but spares full-cue recall.
- **Supplemental:** O&N (pp. 222–227, Fig. 29) is the original autoassociator proposal *specifically* for CA3 — predating both Marr (1971, who treated it more abstractly) and the later Treves–Rolls modeling. Two non-obvious O&N features the catalog should track: (1) the autoassociator is *theta-paced* — successive theta cycles read out successive places along a path, not a single static attractor (pp. 224–225); (2) the recurrent network is proposed to learn via Hebbian LTP on co-active recurrents during exploration (p. 230, citing Bliss & Lømo) — a prediction made *before* the connectivity-specific LTP literature confirmed it. Bz (Cycle 11, pp. 296–301) extends this with experimental data on CA3 autoassociator dynamics. **Project-relevant:** the canonical "CA3 autoassociator + LTP" target of the T1.A roadmap should treat sequence-attractor (not point-attractor) as the validation goal.

### D.06 Place cells — hippocampal spatial code (O'Keefe 1971)

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.55]*

- **System:** CA1 and CA3 pyramidal cells; one-or-few place fields per cell; rate code.
- **Biological role:** Represent the animal's location in an environment. Fields tile the environment via the population. Remap completely between environments (orthogonalization). Field size grades along dorsoventral axis (small dorsal → large ventral). Stable for days when animal attends to space.
- **Sim status:** partial — `g11_bg_runner --learned-perception --landmarks` produces place-cell-like activations from landmark-distance sensors via STDP, but cells are sensor-driven not allocentric, not validated for remapping, and the population is undifferentiated (no DG/CA3/CA1 distinction).
- **Cluster:** D primary, E (sensors).
- **Prerequisites:** D.52, learned-perception input layer.
- **Citation:** Kandel 6e Ch 54 pp 1361–1366, Figs 54-12, 54-13, 54-15.
- **Behavioral validation:** (a) Stable place fields across sessions in same environment; (b) global remapping when room changes; (c) larger fields ventrally; (d) place-field instability after CaMKII-inhibitor or NMDAR-NR1 KO.
- **Supplemental:** The discovery paper is O'Keefe & Dostrovsky (1971), but the *theoretical foundation* is O&N Ch 4.7 (pp. 190–217), which establishes (a) the operational definition of a place unit (firing tied to *location in absolute space*, not to any single sensory modality, p. 192), (b) the distinction between place cells and "displace cells" (= theta cells / interneurons, p. 195), (c) that place fields are *allocentric* — they survive cue removal, darkness, and rotation of the rat, but follow rotation of the room frame (pp. 200–209), (d) that fields are remembered across days *and* across overlapping environments (pp. 209–215), and (e) the prediction that fields will *remap* completely between distinct environments (p. 213). **Discrepancy with current sim status:** the project's `--learned-perception --landmarks` place-cell-like activations are *sensor-driven* (drop the landmark sensor → no firing); a true O&N place cell should still fire on subsequent traversals of the same location even after some cues are removed. This is testable in the existing `g11` runner and could be a useful validation gate for D.06.

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
- **Supplemental:** O&N (pp. 209–215, "missing stimulus" experiments) argued for partial-cue completion *before* it had a name — the same place unit fires when 1–2 of 4 original cues are removed, but stops firing when 3+ are removed. O&N pp. 224–227 attribute this directly to CA3 recurrent collaterals via Hebbian-strengthened intra-CA3 synapses (1978, six years before the term "pattern completion" entered general use). **Reframe for catalog:** D.13 is not a 1990s computational-neuroscience contribution — it is the *original* O&N hypothesis backed by 1970s single-unit data, with the modern CA3-NMDA-KO mouse experiments (cited in current entry) as confirmation.

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
- **Supplemental:** Bz Cycle 11 (pp. 313–323) is the definitive treatment of theta-paced sequence coding. Beyond the encoding/retrieval-phase distinction the catalog already has, Bz adds: (a) **phase precession** (O'Keefe & Recce 1993; Bz pp. 314–316) — place-cell spike phase advances ~360° as the rat traverses a field, *independent* of running speed and firing rate, with slope set only by field size; (b) **theta sequences** (Bz pp. 316–323, Fig. 11.14) — place cells with successive fields fire in temporal order *within a single theta cycle*, compressing real seconds-of-traversal into ~120 ms of theta-cycle time, which is the LTP/STDP window — this is the *mechanism* by which behavioral sequences become storable in synaptic weights. (c) **Discrepancy with Kandel framing:** Kandel attributes encoding/retrieval to peak vs trough; Bz/O&N treat theta as a *temporal compression* mechanism whose primary function is to bring distant-in-real-time events into the synaptic-plasticity window. The compression view is more directly actionable for the simulator: a theta carrier with phase-advancing place-cell drive would let recent trajectories self-encode via existing STDP, no new plasticity rule needed.

### D.19 Sharp-wave ripples (SWRs) — replay in quiet wakefulness + NREM

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.68]*

- **System:** CA3 self-organized population burst → CA1 ripple (140–200 Hz) propagating to deep EC → neocortex. Coordinated with cortical slow oscillations + thalamic spindles during NREM.
- **Biological role:** Compressed (~20×) replay of waking firing sequences. Forward replay primes upcoming trajectories; reverse replay during reward consolidates path-to-reward. SWR disruption impairs spatial memory consolidation. The mechanism that drives hippocampal–cortical dialogue for systems consolidation.
- **Sim status:** partial — sleep-replay infrastructure exists in `bridge.py` (NREM scaffolding) but replay *content* is the named bottleneck; no SWR detection, no compressed sequential replay of waking trajectories, no coupling to cortical slow oscillation.
- **Cluster:** N primary, D primary, J (replay-driven LTP).
- **Prerequisites:** sequence storage during waking (theta sequences D.67), replay generator with compression, ripple-band oscillation in CA3.
- **Citation:** Kandel 6e Ch 54 pp 1365–1366, p 1250 (reference); also Buzsáki, Wilson/McNaughton replay literature.
- **Behavioral validation:** (a) Detected ripple bursts (140–200 Hz); (b) replay sequences match recent waking trajectories at 10–20× compression; (c) closed-loop ripple disruption during sleep impairs next-day spatial memory.
- **Supplemental:** Bz Cycle 12 (pp. 343–351) is the most thorough mechanistic treatment available. Several points the current entry doesn't have: (a) SWRs are the *first and only* population pattern in the developing hippocampus and persist when CA3 is *transplanted out of the brain* — they are intrinsic to the CA3 recurrent network, not an entrainment phenomenon (Bz p. 344, citing Leinekugel et al. 2002); (b) ~50,000–100,000 neurons fire in a 100-ms window during a SWR, 5–15% of the local population, an order of magnitude denser than during theta (Bz pp. 345–346); (c) E/I balance during a SWR shows excitation transiently exceeds inhibition by 3–5× — the *only* time in normal hippocampal operation this happens — which is what makes SWRs uniquely effective for plasticity (Bz p. 346); (d) SWR participation of individual cells is *non-random*: a small fraction of pyramidals participate in 40% of successive events, with the bias correlated with that cell's waking firing pattern (Bz pp. 347–348, citing Wilson & McNaughton 1994). **Project-relevant:** the bottleneck for SWR replay-content quality is not the replay generator but the *waking* theta-sequence trace that biases which cells fire together during a subsequent SWR. Improving theta-cycle storage during exploration may matter more than improving the replay rule itself. Bz (pp. 346–347) also presents the alternative "memory-erasure" hypothesis (Crick & Mitchison 1983; Colgin et al. 2004) — sharp waves *scramble* hippocampal weights between days. Currently the project's sleep-replay infra implicitly assumes the consolidation-not-erasure view; a configurable scramble-vs-replay flag would let the project test both.

### D.20 Reactivation supports retrieval — cortical patterns recur during recall

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from D.69]*

- **System:** HC retrieval-time activity drives reactivation of original cortical encoding patterns; iEEG word-pair recall shows HC-temporal-cortex coupling that re-instantiates encoding-time patterns.
- **Biological role:** Retrieval is partial reinstatement of the encoding state, mediated by HC-cued reactivation of distributed cortical traces. Closes the encoding-storage-retrieval loop with the same circuit.
- **Sim status:** missing — no per-event encoding pattern stored as a labeled vector; no retrieval-cue-triggered reinstantiation.
- **Cluster:** D primary, G secondary.
- **Prerequisites:** D.63 (engram tagging primitive).
- **Citation:** Kandel 6e Ch 52 p 1299–1300.
- **Behavioral validation:** Encoding-pattern multivariate similarity peaks at retrieval relative to baseline (RSA / pattern-similarity analysis).
- **Supplemental:** Bz Cycle 12 (pp. 343–351) frames hippocampal-cortical reactivation as the *neocortical readout* of an SWR — the synchronous CA3→CA1→Sub→EC-deep→neocortex chain (Bz Fig. 12.3, p. 345) in which 5–15% of HC neurons co-fire and the ripple-bound spike train is propagated through the parahippocampal output pathway. This is *not* a generic "HC drives cortex" story — it requires the specific 100-ms compressed SWR packet to overcome neocortical input thresholds. Implication for the simulator: bidirectional reinstatement-during-recall (currently missing) requires the *output* arm of the SWR cascade (CA1 → Sub → EC-deep → cortex), which is absent from the current `g11` hippocampus stub.

## Cluster E — additions

---

## Cluster E — Sensory perception & cortical encoding

**22 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

### F.12 Codon representation — sparse expansion recoding via granule layer
- **System:** mossy fibre → granule cell glomerulus; combinatorial
  sparse coding from ~7,000 MFs/PC into ~200,000 GCs/PC.
- **Biological role:** Each GC fires only when ≥R of its 4–5 MF claws
  are active (R = "codon size"). Different MF input patterns map to
  *overlapping but proportionally less-overlapping* GC subsets:
  pattern overlap X scales as (W/L)^R, so codons separate similar
  inputs geometrically. <1/20 of GCs active at any time (Marr's <5%
  sparsity prediction). The codon code is what makes a single PC
  perceptron-classifier viable — without expansion recoding, the
  raw MF input space is too low-dimensional to be linearly separable
  by a single output cell.
- **Sim status:** **missing**. Existing connectivity generators
  (`sim/connectivity.py`) support spatial / WS / motif wiring but no
  expansion-recoding generator that controls (a) MF→GC random
  4–5-claw sampling and (b) GC firing threshold tuned to a target
  codon size R. Roadmap T2.A should add `build_mf_gc_codon_layer(
  n_mf, n_gc, claws=4, target_codon_size=R)`.
- **Cluster:** F primary; G (working memory / sparse coding) secondary.
- **Prerequisites:** F.02, F.03.
- **Citation:** Marr 1969 §3.0 p 444; §3.1–3.3 p 445–449; Albus 1971
  §IV.A p 41–42 (Expansion-Recoder Perceptron).
- **Behavioral validation:** Two MF inputs overlapping by W/L = 0.5
  produce GC populations overlapping by (0.5)^R (R=4: 0.06; R=5:
  0.03). Smoke test: present similar MF patterns → GC overlap ratio
  matches Marr's Table 1 (p 444).

### F.13 Golgi-cell codon-size regulator — adaptive sparsity control
- **System:** Golgi cell, 1 per 9–10 PCs; upper dendrites in molecular
  layer (driven by PFs), lower dendrites in glomeruli (driven by MFs);
  axon inhibits ~4,500 GCs in its region via glomerular inhibition.
- **Biological role:** Adjusts GC firing threshold so that the number
  of active GCs (and thus codon size R) tracks the number of active
  MFs L. Without this, large MF inputs swamp the GC layer (Marr §4.1
  p 449–450; Albus §IV.B): if L jumps from 20 to 2,000 active MFs at
  fixed threshold, GC activity explodes. The Golgi cell measures L
  via its lower dendrites (and PF activity via its upper dendrites)
  and feeds back inhibition that scales R appropriately. Marr §8.2
  predicts the Golgi cell is driven by the *greater* of upper vs
  lower dendritic inputs (max-pooling, not summation).
- **Sim status:** **missing**. Even when granule-layer wiring is
  added, an adaptive-threshold Golgi feedback is non-trivial. Could
  be implemented with a homeostatic mechanism scaled by global MF
  rate, but Marr's max-pool prediction (lower-vs-upper dendrite)
  makes it a one-off architectural choice.
- **Cluster:** F primary; J (homeostasis) secondary.
- **Prerequisites:** F.12.
- **Citation:** Marr 1969 §4 p 449–453; §8.2 p 469; Albus 1971 §II.C
  p 27–29 (Golgi anatomy + overlap).
- **Behavioral validation:** Sweep number of active MFs from 20 to
  2,000; with Golgi feedback, fraction of active GCs stays bounded
  (Marr's <5% target); without, fraction rises monotonically.

### F.14 Maintenance-reflex / learned conditional reflex — postural learning
- **System:** spinal portion of inferior olive (Brodal 1954, Armstrong
  et al. 1968) — IO cells driven by *receptors* rather than cerebral
  command-fibre collaterals; cerebellar cortex; effector closes the
  loop via stabilising negative feedback.
- **Biological role:** A second, distinct cerebellar input-output
  regime alongside learned movements (F.07/F.10). Each receptor →
  olivary cell → PC → effector → environment forms a stabilising
  negative-feedback loop *iff* the PC fires; the cerebellar cortex
  learns *which contexts* (MF-encoded postural state) enable that
  loop. Worked example: child learning to stand. Posture context
  (vestibular + proprioceptive MF input) → trained PC pause → DCN
  disinhibition → corrective limb movement → reduced imbalance →
  reduced IO firing → loop closed. The PCs effectively store an
  inverse-model lookup table: "in postural context X, suppressing
  PC Y stabilises imbalance Z".
- **Sim status:** **missing**. Distinct from learned-movement F.07 —
  requires (a) IO cells driven by sensor input rather than command
  efference, (b) a closed receptor-effector loop external to the
  network. A 2D inverted-pendulum balance task would be a natural
  smoke test.
- **Cluster:** F primary; H (motor plant) secondary.
- **Prerequisites:** F.04, F.05, F.06; receptor + effector models.
- **Citation:** Marr 1969 §7.2 p 466–467; §8.3 p 469.
- **Behavioral validation:** Pendulum-balance benchmark: cerebellar
  cortex learns to stabilise pendulum from vestibular-like state
  input; IO firing rate drops as posture is held; cortical lesion
  abolishes acquired stability but baseline reflex remains (DCN
  short-loop preserved).

### F.15 Inhibition-sampling readout — alternative PC readout regime
- **System:** PC simple-spike firing modulated by background tonic
  inhibition from stellate / basket cells; downstream readout
  monitors PC response to a CF probe.
- **Biological role:** Alternative to the standard "PC fires when
  context recognised" readout. In inhibition-sampling, basal PC
  inhibition is high (PCs near silent); the rest of the brain probes
  whether the current MF context has been learned by observing the
  *amplitude* of PC response to a uniform weak CF probe. Recognised
  context → large PC response → strong DCN inhibition → effector
  pause; unrecognised → small response. Marr 1969 §7.1 (p 465)
  attributes this idea to Eccles, Ito & Szentágothai 1967 p 177 and
  argues it is especially natural if the IO command pathway IS the
  cortico-olivo-PC command circuit (rather than a collateral).
- **Sim status:** **missing**. The current `g11_bg_runner`-style
  output extraction (population firing rate via `ReadoutEngine`) is
  closer to standard readout. Inhibition-sampling would require
  active baseline inhibition + CF-probe gating.
- **Cluster:** F primary; G (readout / decoding) secondary.
- **Prerequisites:** F.01, F.04, F.05.
- **Citation:** Marr 1969 §7.1 p 465; Eccles, Ito & Szentágothai
  1967 p 177.
- **Behavioral validation:** Two PCs trained on same MF patterns,
  one in standard readout regime, one inhibition-sampling — both
  reach similar discrimination accuracy on novel MF probes; the
  latter yields graded-amplitude rather than binary responses.

### F.16 Variable inhibitory synapses (PF → basket / stellate-b) — Albus's bidirectional rule
- **System:** parallel-fibre synapses on basket cell + stellate-b cell
  dendrites (in molecular layer); ~5% PF contact rate × ~100 cells per
  PC ≈ same convergence as direct PF→PC.
- **Biological role:** Albus 1971 §IV.D–E (p 46–48) argues that the
  same CF-gated weakening rule that governs PF→PC LTD also governs
  PF→stellate/basket synapses (CF axon collaterals contact basket /
  stellate somata). Function: gives the PC effective bidirectional
  weight adjustment despite Dale's law (a single PF cannot directly
  excite some PCs and inhibit others). When CF + PF coincide, the
  PF→PC excitation weakens (PC pauses on this pattern) AND the
  PF→basket→neighbour-PC inhibition also weakens (neighbouring PCs
  *disinhibit* on this pattern). Net: a single CF teaching event
  produces a spatial pattern of PC pauses + adjacent-PC excitation
  in the transverse direction — the cerebellar analog of a perceptron
  learning both positive and negative weights. Albus's stability +
  capacity arguments (§IV.C–F) require this; a PF→PC-LTD-only system
  asymptotically silences and saturates.
- **Sim status:** **missing**. Even when F.05's PF→PC LTD is added,
  the basket / stellate-b plasticity is rarely modelled. Important
  for stability — without it, all PCs in a runner converge to silent.
- **Cluster:** F primary; J secondary.
- **Prerequisites:** F.01, F.04, F.05.
- **Citation:** Albus 1971 §IV.D–F p 46–49 (and Fig. 9 p 49).
- **Behavioral validation:** Long-run training without basket/stellate
  plasticity → PCs collapse to silent over hours; with it, PC firing
  rates remain in physiological 30–80 Hz range while still acquiring
  pattern-specific pauses.

### F.17 Intrinsic Purkinje-cell timer — non-LTD substrate for adaptive CR timing
- **System:** PC dendritic membrane; mGluR1-coupled slow Ca²⁺
  cascades; Ca²⁺-activated K⁺ currents; possibly other slow molecular
  "timer" mechanisms intrinsic to the PC.
- **Biological role:** Hesslow 2013 §3–4 (p 85–86) argues — based on
  recordings (Jirenhed & Hesslow 2011a/b) — that the adaptive timing
  of the conditioned PC pause does NOT come from a granule-cell
  delay-line (no temporal patterning in GC responses) and does NOT
  come from PF→PC LTD alone (no PF depression in conditioned PCs).
  Instead, training selects from a "family of timer units" intrinsic
  to the PC that, once triggered by a brief PF input, run a
  predetermined hyperpolarising response with a learned latency and
  duration. Evidence: a single brief MF pulse evokes a normally-timed
  ~200 ms PC CR; a 50 Hz 400 ms train yields the same CR — i.e. the
  CR shape is determined by the PC, not the input train. Two
  CS-US intervals can coexist as a double-peak CR in a single PC.
  CR latency does not change gradually with CS-US shifts — old CR
  extinguishes, new CR is acquired independently.
- **Sim status:** **missing**. Single-compartment HH does not include
  mGluR1-coupled slow cascades. Implementation would require an mGluR
  state variable (slow Ca²⁺), an SK / KCa channel coupled to it, and
  CF-gated potentiation of the timer's latency setpoint. Substantial
  but tractable extension to `sim/kernels.py`.
- **Cluster:** F primary; J (plasticity), I (channel kinetics) secondary.
- **Prerequisites:** F.01, F.04.
- **Citation:** Hesslow 2013 §3 p 85–86 (proposed); Fiala, Grossberg &
  Bullock 1996; Steuber & Willshaw 2004 (computational mGluR models).
- **Behavioral validation:** A trained PC produces a brief-input → full-
  duration adaptive CR (single MF pulse → normally-timed pause); two
  CS-US intervals trained in alternation yield double-peaked CR;
  pharmacological mGluR1 block abolishes timing but not CR amplitude.

### F.18 Nucleo-olivary feedback loop — DCN inhibition of inferior olive
- **System:** anterior interpositus nucleus (AIN) of DCN sends
  inhibitory GABAergic projection back to inferior olive (the
  "nucleo-olivary pathway", Hesslow 2013 Fig. 1 caption p 82).
- **Biological role:** Once a CR is learned and the PC pauses on the
  CS, the DCN is released from PC inhibition (transient burst), and
  this DCN burst drives both (a) the motor effector (blink) and
  (b) inhibition of the IO. The latter cuts off the CF teaching
  signal once the cerebellum is correctly predicting the US, closing
  the loop and explaining why CS-alone trials extinguish the CR
  (no CF teaching signal arrives because IO is suppressed by the
  PC-pause-driven DCN burst that the now-reliable CR causes itself).
  This is the "predictor as its own teacher off-switch" — a key
  architectural feature absent from a feed-forward LTD-only story.
  Hesslow & Ivarsson 1996 demonstrate IO suppression during CRs in
  decerebrate ferret.
- **Sim status:** **missing**. The current cluster-F closure plan
  (line 1786–1793 of feature-catalog.md) wires `inferior_olive →
  purkinje` and `purkinje → deep_nuclei` but does NOT wire
  `deep_nuclei → inferior_olive` inhibition. Without this loop, an
  eyeblink runner will continue to fire CFs throughout training and
  the PF→PC weights will drift indefinitely.
- **Cluster:** F primary; J (plasticity stability) secondary.
- **Prerequisites:** F.04, F.06.
- **Citation:** Hesslow 2013 Fig. 1 caption p 82; Hesslow & Ivarsson
  1996, Exp. Brain Res. 110: 36–46.
- **Behavioral validation:** With nucleo-olivary feedback, IO firing
  rate decreases as CR is acquired; CS-alone trials produce
  extinction because no CF teaching event is delivered. Without
  feedback, IO continues firing at baseline regardless of CR
  acquisition and PF→PC weights diverge.

### F.19 Brainstem eyeblink reflex circuit — UR substrate the CR converges onto
- **System:** trigeminal afferents (Vp / Vo / Vi / Vc subdivisions) →
  premotor "blink" area (rostral Vo + caudal Vp + reticular) → o.o.
  motoneurons in facial (VIIth) nucleus + r.b. motoneurons in accessory
  abducens (AccVI) + l.p. inhibition via contralateral oculomotor (III).
  Two parallel pathways: Path 1 (disynaptic, R1 ~6–7 ms) and Path 2
  (polysynaptic via Vi/Vc/upper spinal cord, R2 >15 ms).
- **Biological role:** Generates the protective eyeblink/NM
  unconditioned response (UR) to corneal/periocular stimulation. The
  CR converges onto this same final common path via cerebellar AIP →
  red nucleus → premotor blink area / motoneurons (Holstege & Tan 1988).
  Lateralised: ipsilateral CR; corneal US transferred to contralateral
  eye after unilateral cerebellar lesion conditions normally on that
  side (McCormick et al. 1982a). The premotor blink area coordinates
  o.o. closure + r.b. retraction + l.p. inhibition simultaneously to
  produce a complete blink.
- **Sim status:** **missing**. No brainstem reflex module exists. T2.A
  cerebellar microcircuit should declare a small `brainstem_blink`
  region (5–10 motoneurons + premotor pool) with cerebellar AIP
  projections AND direct trigeminal-afferent CS/US drives, so that
  pre-conditioning UR baseline can be measured.
- **Cluster:** F primary; H (motor plant) secondary.
- **Prerequisites:** F.06 (DCN), F.04 (climbing fibre via IO), and a
  trigeminal afferent input region.
- **Citation:** Hesslow & Yeo 2002 §"Eyelid Blink and NMR Response" pp
  89–94; §"Premotor Blink Area" pp 91–94; §"Lesions of Cerebellar
  Efferent Pathways" p 109.
- **Behavioral validation:** Pre-conditioning, periocular shock evokes
  a UR with two EMG components R1 (6–7 ms) and R2 (>15 ms). After
  conditioning, the CR latency is intermediate (~CS-US interval before
  US onset). Cerebellar AIP lesion abolishes CR but UR survives with
  slightly depressed amplitude (CR/UR dissociation gate).

### F.20 Reversible cerebellar inactivation — gold-standard learning-vs-performance assay
- **System:** AIP cold-block / muscimol / lidocaine / CNQX infusion;
  alternatively TTX in brachium conjunctivum (cerebellar efferent
  block).
- **Biological role:** Decisive methodology that distinguishes
  *learning* sites from *performance* sites. If conditioning trials
  given during inactivation produce no CR after the inactivation is
  lifted → the inactivated structure was necessary for *acquisition*.
  If CRs appear immediately after lifting → the structure was only
  necessary for performance. Critical control: showing that
  *extinction* is also blocked during inactivation rules out
  state-dependent learning. Krupa & Thompson 1995 dissociation: AIP
  somata inactivation prevents acquisition; cerebellar efferent
  (brachium conjunctivum) inactivation does not — places the learning
  site within the cerebellum, not downstream.
- **Sim status:** **missing**. The simulator's `set_plasticity_gate`
  infrastructure already supports per-pathway freeze; what's missing
  is a runner-side abstraction "mute region X for K trials, then
  unmute and resume" that mirrors a reversible-inactivation experiment.
  Trivial extension to `g11_bg_runner` style.
- **Cluster:** F primary; J (plasticity validation) secondary.
- **Prerequisites:** F.06, F.08, plus existing plasticity-gate
  infrastructure.
- **Citation:** Hesslow & Yeo 2002 §"Reversible Cerebellar Inactivations"
  pp 116–117; Welsh & Harvey 1991 (negative); Clark et al. 1992; Nordholm
  et al. 1993; Krupa et al. 1993; Krupa & Thompson 1995; Ramnani & Yeo
  1996; Hardiman et al. 1996; Attwell et al. 1999a/b (CNQX cortical block).
- **Behavioral validation:** Mute AIP for first 100 conditioning trials,
  then unmute for next 100. Acquisition curve over the second block
  should track an animal that started fresh (no carry-over learning).
  Compare with cerebellar-efferent mute (downstream-of-AIP region) for
  same number of trials → second-block acquisition should be *fast*
  (carry-over learning intact). The dissociation is the validation.

### F.21 Microzone — olivo-cortico-nucleo-olivary processing module
- **System:** parasagittal strip of cerebellar cortex (~few hundred PCs,
  parallel to PC dendritic plane, transverse to folium) sharing a
  common climbing-fibre input from one olivary subregion + a common
  output to one DCN subregion + an inhibitory return projection from
  that DCN subregion to the same olivary subregion.
- **Biological role:** The canonical cerebellar processing unit.
  Voogd's zones A/B/C/D/Y subdivide laterally; each zone splits into
  microzones with shared climbing-fibre receptive fields. Hesslow & Yeo
  2002 propose every cerebellar function — motor, autonomic, cognitive
  — uses the same olivo-cortico-nucleo-olivary module replicated across
  the cerebellum. Microzones are NOT independent: parallel fibres run
  *across* microzones along the folium, so PF→PC LTD in one microzone
  can affect downstream PCs in adjacent microzones. But CF teaching
  is microzone-private. This gives the architecture an interesting
  asymmetry: shared substrate, private supervision. For eyeblink: AIP
  receives convergent input from C1, C3, Y zones; eye-blink microzones
  in HVI cluster ipsilaterally with face-receptive-field CFs from
  rostral DAO + rostral PO.
- **Sim status:** **missing**. The cluster F closure plan does not
  declare microzones explicitly. T2.A should structure the cerebellar
  cortex region as `[microzone_0, microzone_1, ..., microzone_N]` each
  with:
  - own IO subgroup (say 5–10 IO cells) → CF projection to its PCs;
  - own AIP subgroup (5–10 DCN cells) ← PC inhibition from its PCs;
  - GABAergic AIP→IO return projection within the microzone;
  - shared GC pool (parallel fibres cross all microzones in a folium).
- **Cluster:** F primary.
- **Prerequisites:** F.01, F.04, F.06, F.18.
- **Citation:** Hesslow & Yeo 2002 §"Cerebellar Cortex: Zones and
  Microzones" pp 100–101; §"Olivo-Cortico-Nuclear Module" pp 101–103;
  Andersson & Hesslow 1987a (microzone-respecting nucleo-olivary).
- **Behavioral validation:** Two adjacent microzones controlling
  antagonistic muscles should NOT inhibit each other through the
  nucleo-olivary loop (Andersson & Hesslow 1987a — IO cells projecting
  to a microzone are inhibited by *that* microzone's AIP cells, not by
  adjacent antagonist-microzone AIP cells). Test: train microzone A on
  CS₁→US₁; verify microzone B's IO firing during US₂ is unaffected.

### F.22 Trace conditioning — hippocampus-dependent CS-US bridging
- **System:** hippocampus (CA1 + CA3) + entorhinal cortex →
  pontine nuclei → mossy fibre → cerebellum. Adds a sustained CS-bridge
  signal during the CS-free trace interval.
- **Biological role:** When CS terminates before US onset (trace gap >
  0), the cerebellum must associate a CS-driven signal with a US
  delivered hundreds of ms later. The cerebellum can *itself* bridge
  up to ~500 ms (F.17 intrinsic PC timer + Svensson & Ivarsson 1999 —
  single 0.2 ms MF pulse evokes normally-timed CR). For longer
  traces, hippocampectomised rabbits fail entirely (Moyer et al. 1990
  — 500 ms trace abolishes learning; 300 ms learns normally; H.M.-class
  amnesics fail traces but learn delay normally). Berger et al.
  1976/1980/1983 documented hippocampal CR-correlated activity that
  *precedes* the overt CR by hundreds of ms — consistent with the
  hippocampus generating a sustained CS-trace signal that is then fed
  into the cerebellum via pontine nuclei.
- **Sim status:** **missing**. Bridges Cluster F and Cluster D
  (hippocampus). Implementation requires a hippocampal region that
  receives CS input and produces sustained activity for a learnable
  trace-window duration, projecting to pontine MF.
- **Cluster:** F primary; D (hippocampus) secondary.
- **Prerequisites:** F.08 (delay eyeblink), hippocampal region (existing).
- **Citation:** Hesslow & Yeo 2002 §"Trace Conditioning" p 133;
  Solomon et al. 1986; Moyer et al. 1990; Weiskrantz & Warrington 1979.
- **Behavioral validation:** CS-US gap parameter sweep: with intact
  hippocampus, learning succeeds for traces 0–800 ms; with
  hippocampus muted, learning succeeds for 0–~500 ms then collapses.
  Delay conditioning (no gap, US during CS) is unaffected by hippo
  ablation. (Two-axis validation: delay vs trace × hippo vs no-hippo
  → 2×2 factorial with sharp predictions.)

### F.23 Hippocampus-dependent classical-conditioning paradigms — six-pack
- **System:** hippocampus + pontine MF channel + cerebellar CCC module.
  Each paradigm requires an intact hippocampus on top of the cerebellar
  CCC for normal learning.
- **Biological role:** Six classical-conditioning phenomena where
  hippocampal lesions impair acquisition or extinction even though
  delay eyeblink itself is intact (Hesslow & Yeo 2002 pp 132–135):
  (1) **Trace conditioning** (CS terminates before US, gap > ~500 ms);
  (2) **Discrimination reversal** (CS+ ↔ CS- contingencies swapped
  mid-training; hippo subjects fail to drop responses to new CS-);
  (3) **Latent inhibition** (CS-alone pre-exposures slow subsequent
  acquisition; absent in hippo lesions — Solomon & Moore 1975);
  (4) **Conditional discrimination** (CS reinforced *only if* preceded
  by gating stimulus S; lost in hippo lesions — Ross et al. 1984; Daum
  et al. 1991); (5) **Sensory preconditioning** (S₁ paired with S₂
  pre-training, then S₂ paired with US; S₁ subsequently elicits CR
  via S₁→S₂→US chain; lost without hippo — Port et al. 1987);
  (6) **Blocking** (Kamin: prior CS₁→US training prevents CS₂
  acquisition in CS₁+CS₂ compound; lost without hippo — Solomon 1977).
  The shared theme: hippocampus is required when the task involves
  more than one contingency, contextual gating, or pre-association
  among stimuli — a "preprocessing" role that delivers a refined CS
  signal to the cerebellum.
- **Sim status:** **missing**. Each is a distinct behavioural validation
  for a hippocampus + cerebellum combined model. Blocking is the most
  mechanistically grounded — Kim et al. 1998 confirmed nucleo-olivary
  GABA pathway, but Solomon 1977 shows hippocampus also required,
  unresolved interaction.
- **Cluster:** F primary; D (hippocampus) secondary; O (associative /
  contextual learning) tertiary.
- **Prerequisites:** F.08, F.18, F.22, hippocampal region.
- **Citation:** Hesslow & Yeo 2002 §"Role of the Hippocampus" pp 132–135;
  Solomon & Moore 1975 (latent inhibition); Solomon 1977 (blocking);
  Ross et al. 1984 (conditional discrimination); Port et al. 1987
  (sensory preconditioning); Orr & Berger 1985 (discrimination
  reversal); Kim et al. 1998 (blocking via nucleo-olivary).
- **Behavioral validation:** Each of the six paradigms is a separate
  validation gate. Minimum viable test: blocking. Train CS₁→US to
  asymptote; switch to CS₁+CS₂→US compound for N trials; probe CS₂
  alone — intact model: CR rate to CS₂ near zero (blocked); hippo or
  IO-GABA-disabled model: CR rate to CS₂ near baseline acquisition
  rate.

### F.24 Adaptive CR latency from frequency-coded CS — cerebellar after-MF timing
- **System:** cerebellar cortex (specifically the intrinsic PC timing
  mechanism, F.17) operating on MF/PF inputs delivered at variable
  frequency.
- **Biological role:** When stimulation frequency of either a peripheral
  CS or direct MCP/MF stimulation increases from 50 Hz to 100 Hz, the
  CR latency *immediately shortens* (well before any plasticity could
  re-tune); with continued training at the new frequency, latency
  gradually re-adapts to the original CS-US-anchored value (Svensson et
  al. 1997). Same effect for peripheral and MF-direct stimulation —
  proves the timing transformation occurs *after* the MF stage, in the
  cortex. This is a direct prediction of an intrinsic PC timer model
  (F.17): higher input frequency → larger Ca²⁺ load on the timer →
  faster timer trigger; slow plasticity then retunes the latency
  setpoint.
- **Sim status:** **missing**. Requires F.17 intrinsic timer + a runner
  that sweeps CS frequency mid-session.
- **Cluster:** F primary; J (plasticity timing) secondary.
- **Prerequisites:** F.17 (intrinsic PC timer), F.08.
- **Citation:** Hesslow & Yeo 2002 §"Information Processing in the CS
  Pathway" p 127; Svensson et al. 1997.
- **Behavioral validation:** Train to asymptotic 50 Hz CS-evoked CR
  with peak ~250 ms post-CS-onset (CS-US = 300 ms). Switch to 100 Hz
  CS without changing CS-US interval. Predict: first-block CRs peak
  ~150–200 ms (latency shortened); after 100+ trials at 100 Hz, peak
  re-tunes back toward 250 ms. Reverse switch should produce mirror
  effect.

### F.25 Cortico-rubral plasticity (Tsukahara) — alternative non-cerebellar substrate
- **System:** cerebral peduncle → corticorubral fibres → red nucleus.
  Tsukahara's leg-flexion conditioning paradigm uses sub-threshold
  cerebral-peduncle stimulation (CS) paired with peripheral forelimb
  shock (US); over ~7–10 days, the threshold for evoking forelimb
  flexion via CS drops, attributed to *axonal sprouting* at corticorubral
  synapses (Tsukahara et al. 1981; Oda et al. 1988 — fast-rising EPSP
  develops; Pananceau et al. 1996 — interposito-rubral plasticity in
  variant paradigm).
- **Biological role:** A non-cerebellar plasticity that can support an
  associative learning task with similar surface phenomenology to
  classical conditioning (acquisition + extinction). Hesslow & Yeo
  2002 flag this as evidence that *some* skeletal conditioning
  paradigms recruit non-cerebellar substrates, but with three
  important differences from eyeblink: (a) no adaptive timing (CR has
  same latency as CS-evoked response), (b) much slower acquisition
  (~120 trials/day × 7–10 days vs <200 trials total for eyeblink),
  (c) plasticity mechanism is structural (axonal sprouting), not
  synaptic. Inclusion in the catalog is partly cautionary: NOT every
  conditioning result implicates the cerebellum.
- **Sim status:** **missing**. Probably out of scope for T2.A
  cerebellar microcircuit, but documented as the alternative against
  which "cerebellar-essential" claims should be validated.
- **Cluster:** F primary (as a *contrast*); H (motor plant) secondary.
- **Prerequisites:** N/A — separate substrate.
- **Citation:** Hesslow & Yeo 2002 §"Conditioning-Induced Plasticity in
  Other Motor Structures" pp 130–132; Tsukahara et al. 1981; Oda et
  al. 1988; Pananceau et al. 1996.
- **Behavioral validation:** A simulator runner using Tsukahara
  cortico-rubral plasticity (axonal sprouting analogue) on a leg-flexion
  task should reproduce: (1) acquisition over ~1000 trials; (2) no
  adaptive timing — CR latency = CS-evoked-response latency; (3) same
  paradigm on eyeblink should fail (or behave differently).

---

End of patch.

---
- **Supplemental:** Hesslow & Yeo 2002 §"Nucleo-Olivary Inhibition" pp
  130–131 develops the nucleo-olivary loop as more than just a
  CR-extinction mechanism — it also explains **Kamin blocking**. After
  training to CS₁, the AIP-driven CR inhibits the IO (Hesslow & Ivarsson
  1996 — direct measurement: US-elicited IO excitation is inversely
  proportional to the size of a preceding CR). When CS₂ is added in
  compound CS₁+CS₂ trials, the established CR₁ now inhibits IO before
  the US arrives, so no CF teaching signal reaches the cerebellum, and
  CS₂ never acquires PF→PC association. Kim et al. 1998 confirmed
  mechanistically: **GABA-receptor blocker injection into the IO
  abolishes blocking** — the nucleo-olivary GABAergic projection is
  *the* substrate. **For the simulator:** an eyeblink runner with
  nucleo-olivary feedback enabled should automatically reproduce
  Kamin blocking on a CS₁→CS₁+CS₂ paradigm; without the feedback, CS₂
  will acquire normally. This is a quantitative behavioural validation.
  `[note: Solomon 1977 reports hippocampal lesions ALSO abolish blocking,
  suggesting a higher-order behavioural-control mechanism may interact
  with the IO substrate; Hesslow & Yeo flag this as unresolved p 131.]`
- **Supplemental:** Hesslow & Yeo 2002 §"Nucleo-Olivary Inhibition" p
  130 explicit positive-feedback argument: a hypothetical *excitatory*
  nucleo-olivary projection would close a positive-feedback loop
  (CF→PC depression → DCN disinhibition → ↑ excite IO → ↑ CF). Positive
  feedback is unstable; the GABAergic (negative) projection is required
  for closed-loop stability. **For the simulator:** if the simulator's
  default neuromodulator subsystem accidentally inverted the sign on
  this projection, the CR-acquisition runner would diverge unboundedly
  in IO firing rate. Worth a unit test: assert
  `assert sign(nucleo_olivary_weight) < 0` after wiring.

## Cluster F — Cerebellum & error-correction

**11 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

### F.01 Purkinje cell — sole output of cerebellar cortex

*[from Part V — Movement (Ch 30-39); renumbered from F.50]*

- **System:** cerebellar cortex molecular + Purkinje layer; massive flat dendritic tree; tonic inhibitory output to deep cerebellar nuclei (DCN).
- **Biological role:** Receives ~150,000 parallel-fiber inputs (PF) plus a single climbing fiber (CF). Tonically fires simple spikes 30–80 Hz; PF input modulates rate. Inhibits DCN; net cerebellar output emerges by disinhibition of DCN as PC activity decreases.
- **Sim status:** **partial** — HH preset `HH_CEREBELLAR_PURKINJE` exists in `sim/enums.py` but no runner instantiates it as part of a circuit.
- **Cluster:** F primary.
- **Prerequisites:** F.51, F.52, F.53.
- **Citation:** Kandel 6e Ch 37 p 918–926.
- **Behavioral validation:** Tonic 30–80 Hz simple spikes when PF input is balanced; CF event evokes characteristic complex spike; brief pause after CF.
- **Supplemental:** Marr 1969 (p 442) gives concrete number: each PC
  receives ~200,000 spine synapses with parallel fibres on a flat
  fan-shaped dendritic tree, and ~7,000 distinct mossy fibres converge
  via the granule layer (Marr 1969 §3.1 p 445). Albus 1971 §II.D
  confirms 200,000 PF-PC spine synapses. Hesslow 2013 §1.1 (p 81)
  emphasises the convergence is what makes PCs ideal associative-learning
  loci. Cerminara & Rawson (2004), cited by Hesslow §2.1 p 82, show PCs
  have an *intrinsic* simple-spike generator firing 30–80 Hz even with
  AMPA blocked — i.e. background firing is NOT driven by PF excitation.
  This is important for the simulator: a single-compartment PC with
  intrinsic pacemaker current (Ih + persistent Na) reproduces tonic
  output without PF drive, which `HH_CEREBELLAR_PURKINJE` should match.
- **Supplemental:** Hesslow & Yeo 2002 §"Neuronal Architecture" pp 97–99
  give the canonical numbers used throughout the 2002 chapter: PCs fire
  simple spikes 50–100 Hz with a strong intrinsic spike-generating
  mechanism (PCs fire at high rates "even when inputs are totally absent"
  — p 100). Each climbing-fibre burst is one PC complex spike (CS) of
  1–4 spikes followed by a 15–30 ms simple-spike pause (the "inactivation
  response" of Granit & Phillips 1956); CS peak Ca²⁺ plateau lasts up to
  several hundred ms in the dendrites (p 99). **For the simulator:** the
  simple-spike pause must outlive the immediate CS by at least 15–30 ms
  for any downstream readout to register; if an HH PC preset under-shoots
  this pause length, downstream DCN disinhibition will be too brief to
  drive a behavioural CR.

### F.02 Granule cell + parallel fiber — divergent code

*[from Part V — Movement (Ch 30-39); renumbered from F.51]*

- **System:** granular layer (>50 billion granule cells in human cerebellum); axon bifurcates into long parallel fibers (PF) running along the folium, perpendicular to PC dendrites.
- **Biological role:** Receives ~4 mossy-fiber inputs each; sparse, high-dimensional combinatorial code. PF passes through dendritic trees of thousands of PCs, contacting each PC with ~1 synapse. Marr-Albus expansion-recoding hypothesis.
- **Sim status:** **partial** — HH preset `HH_CEREBELLAR_GRANULE` exists; no PF wiring code.
- **Cluster:** F primary.
- **Prerequisites:** F.52 (mossy fiber input).
- **Citation:** Kandel 6e Ch 37 p 918–921.
- **Behavioral validation:** Sparse activity (~1–5% active at any moment); sparse-input vs dense-output recoding (distinct mossy patterns → orthogonal PF patterns).
- **Supplemental:** Marr 1969 §3.0 (p 444) introduces the **codon
  representation**: the MF→GC relay encodes each MF input pattern as a
  small subset (codon) of active GCs. Codon size R = number of active
  MFs required to fire a GC (= GC threshold). Pattern-overlap formula
  X = (W/L)^R (eq. 1, p 444), where L = active MFs, W = MFs shared by
  two patterns, R = codon size — shows that increasing R sharpens
  pattern separation roughly geometrically. Marr's specific anatomical
  predictions: 4–5 claws/granule cell, codon size R adjustable by Golgi
  cells (§4), <1/20 of GCs active at any time (§8.2 p 469). Albus 1971
  §IV.A (p 41–42) calls this **expansion recoding** and treats the
  MF→GC stage as the front-end of an "Expansion-Recoder Perceptron"
  (Albus Fig. 6 p 39): the GC-PF layer expands a low-dimensional MF
  pattern into a sparse high-dimensional code that a single linear
  Perkinje "perceptron" can then classify. This is the original
  formal articulation of what is now called *kernel expansion* or
  *random-feature reservoir computing*.

### F.03 Mossy-fiber afferent system — pontine, spinal, vestibular input

*[from Part V — Movement (Ch 30-39); renumbered from F.52]*

- **System:** pontine nuclei (cerebro-cerebellar relay), spinocerebellar tracts, vestibular nuclei → granule layer rosettes.
- **Biological role:** Conveys cortical efference copy + proprioceptive + vestibular state. Excitatory glutamatergic. Branches to DCN (collateral) and granule cells. Each MF excites ~400 granule cells.
- **Sim status:** **missing** — no mossy-fiber pathway in any runner.
- **Cluster:** F primary.
- **Prerequisites:** F.51.
- **Citation:** Kandel 6e Ch 37 p 918–920.
- **Behavioral validation:** Step input (e.g. limb perturbation) drives transient burst across granule layer.
- **Supplemental:** Albus 1971 §II.A (p 26–27) details that MFs carry
  three distinct stream types: vestibular/reticular, cortico-pontine
  (efference copy), and spinocerebellar (proprioception via dorsal
  spinocerebellar tract from muscle spindles + Golgi tendon organs;
  ventral SCT signalling whole-limb contraction state). Tonic baseline
  10–30 Hz even at rest. Each MF branches in 2+ folia and produces
  20–50 rosettes per branch, several hundred rosettes total. **For the
  simulator:** these are three separable input channels — efference
  copy / vestibular / proprioception — and a faithful runner should
  declare them as three separate `mossy_*` source regions, not one
  monolithic pool.

### F.04 Climbing fiber — inferior olive single-cell teaching signal

*[from Part V — Movement (Ch 30-39); renumbered from F.53]*

- **System:** inferior olive (IO) → contralateral cerebellar cortex; one CF per Purkinje cell, wraps dendrites with ~hundreds of synapses.
- **Biological role:** Fires sparsely (~1 Hz) but each spike triggers a Purkinje complex spike (Ca²⁺ plateau). Encodes motor errors / unexpected events. CF coactivation with PF triggers PF→PC LTD — this IS the cerebellar learning rule.
- **Sim status:** **missing** — IO HH preset `HH_INFERIOR_OLIVE` exists, but no 1:1 CF wiring, no PF×CF coincidence-gated plasticity.
- **Cluster:** F primary; J (plasticity) secondary.
- **Prerequisites:** F.50, F.51, F.54.
- **Citation:** Kandel 6e Ch 37 p 920–925.
- **Behavioral validation:** Unexpected perturbation → IO complex spike rate ↑; perturbation becomes predictable → IO rate returns to baseline.
- **Supplemental:** Marr 1969 §1 (p 438–439) gives the canonical 1:1
  IO→PC mapping with very few exceptions; CF makes "synaptic contact
  almost everywhere" on PC dendrites (§2 p 443). Albus 1971 §II.E
  (p 30) and §IV.C (p 44) describe the *inactivation response*: each
  CF burst triggers one PC axon spike followed by 15–30 ms full pause
  (Granit & Phillips), recovery over 100–300 ms. Albus identifies this
  as the unconditioned response in a classical conditioning frame:
  CF-burst = US, PC pause = UR, MF pattern at the time of CF burst =
  CS, learned PC pause = CR (Albus 1971 §IV.C p 44). Marr 1969 §1
  proposes a different framing — IO encodes "elemental movement
  commands" (cerebral command-fibre collaterals); learned PCs replace
  IO by implementing the same elemental movement themselves once the
  context is recognised. **For the simulator:** Albus's framing maps
  cleanly onto our existing reward/training engine — CS = MF pattern,
  US = CF burst, CR = PC pause; the IO-as-error-signal framing is
  what most modern eyeblink simulations use.
- **Supplemental:** Hesslow & Yeo 2002 §"Afferent Systems" p 99 quantifies:
  IO firing is normally "around 1 Hz, with a maximal rate of about 10 Hz,
  but only for very brief periods". Each CF makes "as many as 10 branches,
  each of which contacts a single Purkinje cell" — confirms the 1:CF ↔ ~10
  PCs structure used in Marr's geometric arguments. The CF→PC contact is
  "extensive synaptic contacts" producing a complex spike that is a
  *massive* depolarisation, not just a single EPSP — important for any
  simulator that models the CF as a single point-synapse: it must be
  amplified ~10-fold or modelled as a multi-synapse cluster. §"Olivary
  Lesion" pp 128–129 documents that olivary inactivation/lesion *also*
  causes a rise in tonic PC simple-spike firing and a virtual shutdown of
  cerebellar output (Colin et al. 1980; Montarolo et al. 1982) — i.e. the
  CF normally has a *general tonic suppressive influence* on PC excitability
  on top of its phasic teaching role. Removing IO ≠ just removing the
  teaching signal; it disinhibits PCs and silences DCN downstream. This is
  why olive lesions immediately abolish CRs (Yeo et al. 1986; Welsh & Harvey
  1998) rather than producing gradual extinction. **For the simulator:**
  the IO must contribute a tonic excitability term to PCs, not only spike-
  triggered teaching events; otherwise IO ablation will fail to reproduce
  the observed immediate CR loss.
- **Supplemental:** Hesslow & Yeo 2002 §"IO oscillation" p 100 cites
  Llinás & Yarom 1981 / Llinás & Welsh 1993 noting IO neurons are
  electrotonically coupled by gap junctions (Sotelo et al. 1974) AND have
  intrinsic ~10 Hz oscillatory membrane properties. Functional implication:
  the IO can synchronise small assemblies of PCs in a phase-coordinated
  manner, not just deliver independent error signals. Largely orthogonal
  to eyeblink conditioning but relevant for any cerebellar timing module.

### F.05 PF→PC LTD (Marr-Albus-Ito) — sign-flipped, CF-gated plasticity

*[from Part V — Movement (Ch 30-39); renumbered from F.54]*

- **System:** parallel-fiber → Purkinje cell glutamatergic synapse; postsynaptic mGluR1 + Ca²⁺ from CF.
- **Biological role:** Coincident PF activity and CF complex spike → long-term depression of that PF synapse. Reduces PC simple-spike response to that input. Reduces PC inhibition of DCN → behavior gets stronger / corrected. This is the canonical motor-learning rule.
- **Sim status:** **missing** — `fused_stdp_weight_update` is Hebbian and pre-post-timing-based, not CF-gated. Would need a new fused kernel `fused_pf_pc_ltd` taking (PF spike, CF complex spike) → ΔW < 0, with a separate slow LTP for unpaired PF.
- **Cluster:** F primary; J secondary.
- **Prerequisites:** F.50–F.53.
- **Citation:** Kandel 6e Ch 37 p 922–925 (Marr 1969, Albus 1971, Ito).
- **Behavioral validation:** Eyeblink conditioning (F.57) — paired CS-US → blink prediction; unpaired → no learning.
- **Supplemental:** **Sign discrepancy.** Marr 1969 §5.1 (p 455–456) and
  §8.1 (p 468) explicitly predict the synapse is *facilitated* (LTP) by
  conjunctive PF + CF activity: "the efficacy of that synapse is
  increased towards some fixed maximum value", coincidence window
  ~50–100 ms. Albus 1971 §IV.C (p 44–46) reversed this to **depression**
  on three explicit grounds: (1) Marr's "all weights only go up"
  Perceptron is asymptotically saturated and cannot un-learn (Albus
  p 45–46); (2) Perceptron capacity is ~2× the number of weights only
  if both signs are allowed — Marr's LTP-only theory caps each PC at
  ~200 patterns; with bidirectional, ~200,000 (Albus p 46); (3) using
  the inactivation-response pause as UR (F.04) requires the PF→PC
  weights driving spontaneous firing to be *weakened* by the CF
  teaching signal, not strengthened (Albus p 44–45). Ito & Kano 1982
  (cited by Hesslow 2013 p 82) confirmed Albus's sign empirically.
  `[discrepancy: Marr 1969 originally predicted LTP; Albus 1971 §IV.C
  reversed to LTD on stability grounds; Ito 1982 confirmed Albus.]`
- **Supplemental:** **Albus's explicit weight-update rule** (Albus 1971
  p 44, eq. (4) and surrounding text): on each CF burst, every active
  PF synapse has its weight decreased by an amount proportional to
  *how strongly that synapse was exciting the PC at the time of the
  error signal*. In Perceptron terms: `Δw_i = -η · pf_i · cf_burst`
  for active PF i. Notably Albus also predicts variable PF synapses on
  *basket and stellate b cells* with the same CF-gated rule (Albus
  §IV.C–E p 46–48): the CF teaches both excitatory PF→PC weights and
  inhibitory PF→Basket→PC weights symmetrically. **For the simulator:**
  the new `fused_pf_pc_ltd` kernel proposed in cluster-F-closure should
  also be applied to PF→stellate / PF→basket synapses if those regions
  exist; otherwise PC firing collapses asymptotically (Albus's
  "stability" argument, p 45–46).
- **Supplemental:** **Hesslow 2013 — LTD-as-sole-mechanism is
  contested.** §2 (p 82–86) reviews four challenges: (i) §2.1 p 82 —
  PCs have intrinsic pacemaker firing (Cerminara & Rawson 2004), so
  removing PF excitation cannot drive a PC below baseline; the
  conditioned-response *pause* requires active inhibition, not just
  loss of excitation. (ii) §2.2 p 82–84 — LTD induced in vitro is
  strongest at zero or short PF-CF intervals, but eyeblink conditioning
  fails entirely below 100 ms CS-US (Gormezano & Moore 1969); rates of
  in-vitro LTD induction (~10 min, 100–600 trials) are 10–100×
  faster than behavioural conditioning (2–3 hours). (iii) §2.3 p 84 —
  LTD on a single synapse cannot account for adaptive timing; the
  classical "delay-line" rescue (different GCs firing at different
  times) is contradicted by data (Jirenhed & Hesslow 2011b) showing
  no temporal patterning in GC responses. (iv) §2.4 p 84–85 — direct
  recording during conditioning shows PF responses in conditioned PCs
  are *not* depressed outside the CR window (Fig. 3). Schonewille
  et al. 2011 GluR2D7/K882A LTD-knockout mice condition normally,
  Welsh et al. 2005 pharmacological LTD block does not impair timing.
  Hesslow proposes intrinsic-PC "timer units" (slow molecular
  cascades, possibly mGluR1-coupled or Ca²⁺-activated K⁺) as the
  actual timing substrate. `[discrepancy: PF→PC LTD is necessary but
  not sufficient — see Hesslow 2013 §2; intrinsic PC mechanisms
  likely also involved.]`
- **Supplemental:** Marr's **specific PC capacity calculation** (§5.3
  p 458, Table 6): if 70% of PF synapses are facilitated and each
  learned event uses n active PFs, capacity x = largest integer with
  (1 - n/200000)^x > 0.3. For n=500: x ≈ 480 patterns; n=1000: 240;
  n=2000: 119; n=5000: 47. **For the simulator:** if a runner uses a
  small reservoir (e.g. 200 PCs × 200 events ≈ 40K patterns), this
  validates that one PC can hold ~200 distinct CR memories.
- **Supplemental:** Hesslow & Yeo 2002 §"Conclusions" p 136 explicitly
  notes the *direct* evidence for PF→PC LTD as the conditioning substrate
  is weak. Of the four CCC-model assumptions, they conclude (point 4, p
  136): "The US pathway may be through the climbing fibers. This central
  assumption of the CCC model is **less well supported** than the others.
  It is based on theoretical considerations and also on the fact that
  climbing fiber activation can induce plasticity in coactivated parallel
  fiber-Purkinje cell synapses. At present, there is little direct evidence
  that the US is transmitted by climbing fibers." The 2013 paper
  compresses this to a one-line caveat; the 2002 chapter spends pp 128–130
  reviewing why olive lesion/inactivation evidence (which would seem to
  test the CF=US hypothesis directly) is in fact uninformative because IO
  inactivation removes both the putative teaching signal AND the tonic
  excitability drive to PCs. `[discrepancy: the canonical "CF carries the
  US, PF→PC LTD encodes the CR" story is widely simulated but the direct
  empirical evidence — beyond in-vitro LTD demonstrations — is weaker
  than commonly assumed.]`

### F.06 Deep cerebellar nuclei (DCN) — final cerebellar output

*[from Part V — Movement (Ch 30-39); renumbered from F.55]*

- **System:** dentate, interposed, fastigial nuclei; receive inhibitory PC input + excitatory MF/CF collaterals.
- **Biological role:** Tonic firing 40 Hz; PC inhibition silences DCN; release of PC silences DCN releases excitatory drive to thalamus / red nucleus / brainstem. Net cerebellar effect is via DCN disinhibition.
- **Sim status:** **missing** — no DCN region.
- **Cluster:** F primary.
- **Prerequisites:** F.50.
- **Citation:** Kandel 6e Ch 37 p 911–917.
- **Behavioral validation:** PC inhibition → DCN pause → downstream effector burst.
- **Supplemental:** Hesslow & Yeo 2002 §"Lesions of the Cerebellar Nuclei
  Abolish NMR Conditioning" pp 105–106 establishes the **anterior
  interpositus nucleus (AIP)** specifically — not the dentate, fastigial,
  or posterior interpositus — as the critical DCN region for eyeblink/NMR
  conditioning. Yeo et al. 1985a tested all four nuclei with discrete
  lesions; only AIP lesion abolishes CRs to both auditory and visual CSs.
  AIP lesions are confirmed effective with kainic-acid (fibre-sparing)
  lesions (Lavond et al. 1985) — i.e. it is AIP somata, not fibres of
  passage, that matter. **For the simulator:** the cerebellar microcircuit
  closure in T2.A should explicitly partition the DCN into AIP / posterior
  interpositus / dentate / fastigial regions, and connect only AIP to the
  red nucleus → blink-control pathway; collapsing the DCN into a single
  pool will obscure the lesion specificity that defines the model.
- **Supplemental:** Hesslow & Yeo 2002 §"Lesions of Cerebellar Efferent
  Pathways" p 109 specifies the AIP→red-nucleus→rubrobulbar-tract
  pathway as essential for CR expression (McCormick et al. 1982b;
  Rosenfield & Moore 1983; Rosenfield et al. 1985); the alternative
  fastigial→vestibular/reticular and dentate→ventrolateral-thalamus→motor-
  cortex routes are NOT required. Mauk & Thompson 1987 confirmed via
  rostral decerebration: thalamus and forebrain are not necessary. **For
  the simulator:** the minimal cerebellar→motor wiring is
  `AIP → red_nucleus → facial_nucleus_VII / accessory_abducens_VI`.
- **Supplemental:** Hesslow & Yeo 2002 §"Cerebellar Lesions and
  Excitability Changes" pp 108–109 gives the **CR / UR dissociation**
  argument: AIP lesions remove tonic excitatory drive on the eyeblink
  circuitry and slightly DEPRESS the UR (Welsh & Harvey 1989; Bracha et
  al. 1994), whereas cortical lobule HVI lesions remove PC inhibition of
  AIP, DISINHIBIT the eyeblink circuitry, and *increase* UR amplitudes
  (Yeo & Hardiman 1992; Gruart & Yeo 1995) — yet HVI lesions also abolish
  the CR. UR up + CR down with cortical lesions is strong evidence
  *against* the "performance deficit only" hypothesis: if the cerebellum
  only modulated reflex excitability without storing memory, raising UR
  excitability should also raise CR excitability, but it doesn't. **For
  the simulator:** an honest cerebellar runner should reproduce this
  double dissociation — AIP ablation should drop both CR and UR; cortical
  ablation should drop CR and *raise* UR amplitude; only the BG-style
  fully-shared-engine model would fail this test.
- **Supplemental:** Mauk's two-site (cortical + nuclear) plasticity
  model, presented in Hesslow & Yeo 2002 pp 108–109, 113: cortical
  lesions in trained subjects abolish the long-latency adaptive CR but
  unmask a **short-latency, CS-driven response** that cannot be
  extinguished and whose timing does not adapt to the CS-US interval
  (Perrett et al. 1993; Perrett & Mauk 1995). H&Y interpret this as
  evidence for a slower MF→AIP synaptic plasticity that learns the CS-US
  contingency (without timing), normally masked by Purkinje-cell
  inhibition. The cerebellar cortex then *sculpts* the temporal envelope
  of this short-latency response into a timed CR. Medina & Mauk 1999
  formalize this as a two-site model. **For the simulator:** the T2.A
  microcircuit should include MF collateral → AIP synapses with their
  own slower plasticity rule (e.g. mossy-fibre LTP gated by an AIP-level
  reinforcement signal) in addition to PF→PC LTD, and the runner's
  ablation tests should produce short-latency unadaptive responses when
  the cortex is removed.

### F.07 Forward / inverse internal models — predictive control

*[from Part V — Movement (Ch 30-39); renumbered from F.56]*

- **System:** cerebro-cerebellar recurrent loops (Strick); Purkinje activity correlates with predicted sensory consequences of motor commands.
- **Biological role:** Cerebellum hosts internal models that predict the sensory consequences of efference copy (forward model) and / or compute motor commands needed to achieve a desired sensory state (inverse model). Used to cancel self-generated input (e.g. tickling), pre-emptively counter interaction torques.
- **Sim status:** **missing** — no efference-copy pathway, no forward-model module.
- **Cluster:** F primary; G (PFC working memory) secondary.
- **Prerequisites:** F.50–F.55.
- **Citation:** Kandel 6e Ch 30 p 720–724 (Box 30-1) and Ch 37 p 921–924.
- **Behavioral validation:** Self-generated stimulus (subject moves) → attenuated cerebellar response; passive stimulus same magnitude → full response (Cullen vestibular paradigm).
- **Supplemental:** Marr 1969 §7.3 (p 467–468) anticipates the modern
  "internal model" framing: cerebellum becomes a "sophisticated and
  interpretive buffer language between [cerebrum] and muscle", letting
  cerebrum "handle movements and situations in a symbolic way without
  having continually to make the retranslation". The IO-as-cerebral-
  command-collateral hypothesis (§7.1 p 463–464) IS an early forward-
  model proposal: each elemental movement command produces both an
  efferent motor signal and an efference copy via the IO that trains
  the PC to predict the sensorimotor context that will accompany that
  movement.

### F.08 Eyeblink classical conditioning — canonical cerebellar learning task

*[from Part V — Movement (Ch 30-39); renumbered from F.57]*

- **System:** tone (CS) via mossy fiber → granule → PF; air puff (US) via climbing fiber from IO; blink output via interposed nucleus → red nucleus → motor.
- **Biological role:** Pavlovian timing-precise CR. After paired CS-US trials, animal blinks slightly before US onset. PF→PC LTD on CS-driven PF synapses + DCN plasticity reproduces this. Deep-nuclei lesion abolishes acquired blink; cortical lesion reduces precise timing.
- **Sim status:** **missing** — no canonical task harness; would be the natural smoke-test for cluster F closure.
- **Cluster:** F primary; J, O (reward analog) secondary.
- **Prerequisites:** F.50–F.55, F.54.
- **Citation:** Kandel 6e Ch 37 p 928–932.
- **Behavioral validation:** Acquisition curve (probability and timing of CR vs trials); CS-alone trials probe CR without US; cerebellar lesion abolishes CR.
- **Supplemental:** Hesslow 2013 §1.1–1.2 (p 81–82) gives the modern
  protocol details: CS = tone/light/skin stim (often direct MF
  stimulation in lab); US = corneal air puff (often direct CF/IO
  stimulation); CS-US interval typically 150–500 ms (must be ≥100 ms
  for any learning); 2–3 trials per minute, 1.67–2 hours minimum to
  achieve criterion CR; intertrial interval ≥4 s required (10 s
  reliable). PC CRs recorded in C3 zone of cerebellar cortex — a
  specific PC pause in simple-spike firing develops with paired
  CS-US, mirrors the overt CR's adaptive timing (peak just before
  US onset), extinguishes with CS-alone, reacquires fast on
  re-pairing (Jirenhed et al. 2007). Stimulation of the relevant C3
  zone PCs *suppresses* the behavioural CR (Hesslow 1994a/b) — i.e.
  the PC pause is causally upstream of the blink, confirming the
  microcircuit's basic logic: PC pause → DCN disinhibition → red
  nucleus → motor neuron → blink. **For the simulator:** a faithful
  smoke-test must use CS-US ≥100 ms, intertrial ≥4 s, and measure
  both the overt blink CR *and* a PC pause in the recorded PC
  population — both should track each other in latency and
  amplitude.
- **Supplemental:** Hesslow §3 (p 85) summarises the evidence that
  the cerebellar cortex is the *primary* memory locus (Attwell, Cooke
  & Yeo 2002; Kellett et al. 2010 — cortical pharmacological
  inactivation prevents CR consolidation) — but PC simple-spike CR
  drives the overt CR (Hesslow 1994a/b stimulation result). This
  matters for our implementation: the simulator's choice of *where*
  to put the plastic synapse (PF→PC vs DCN nuclear plasticity) is a
  live debate. Implementing both and ablating one or the other in the
  runner would be a publishable contribution.
- **Supplemental:** Hesslow & Yeo 2002 §"Eyelid Blink and NMR Response
  in Rabbits" pp 89–94 documents the **brainstem reflex circuitry**
  underlying the UR (and the substrate the CR converges onto). The
  external eyelid blink is mediated by the orbicularis oculi (o.o.)
  muscle innervated by the *facial (VIIth) nucleus*; the NM response
  is driven by retraction of the eyeball via the retractor bulbi (r.b.)
  muscle innervated by the *accessory abducens (AccVI) nucleus*. Upper
  eyelid position is the *balance* of o.o. (VIIth nerve) closure vs
  levator palpebrae (l.p.) muscle innervated by the contralateral III
  oculomotor nucleus — so a complete blink requires both excitation of
  VIIth + AccVI motoneurons AND inhibition of contralateral III
  motoneurons. Trigeminal afferents enter via Vp (principal) and Vsp
  (spinal: Vo / Vi / Vc subdivisions); van Ham & Yeo 1996a found
  periorbital afferents project mainly to Vc + dorsal-horn C1; corneal
  afferents to ventral Vi + caudal Vc. **For the simulator:** the
  minimal reflex efferent stage is `(Vp/Vi/Vc) → (premotor blink area
  in VpVo) → (VII + AccVI motoneurons)` plus a contralateral inhibitory
  branch to III. Collapsing this into a single "motor neuron" loses the
  lateralisation the chapter treats as essential.
- **Supplemental:** Hesslow & Yeo 2002 §"Premotor Blink Area" pp 91–94
  identify a discrete **premotor "blink" area** in rostral Vo + adjacent
  Vp + reticular formation (van Ham & Yeo 1996b) that simultaneously
  drives o.o. + AccVI motoneurons + inhibits l.p. Two pathways are
  distinguished: **Path 1** (R1 component, ~6–7 ms latency, disynaptic)
  and **Path 2** (R2 component, polysynaptic, >15 ms latency through
  caudal Vi/Vc/upper spinal). The cerebellum (via red nucleus → blink
  area, Holstege & Tan 1988) modulates *both* pathways. **For the
  simulator:** if reproducing both UR and CR EMG records, model two
  parallel routes (a fast disynaptic and a slow polysynaptic) so that
  cerebellar gain modulation reproduces the differential UR latency
  effects.
- **Supplemental:** Hesslow & Yeo 2002 §"CS and US Pathways" pp 123–127
  document the **mossy-fibre stimulus substitution** results that prove
  the CS pathway. Steinmetz et al. 1986a/1989 — pontine nuclei or middle
  cerebellar peduncle (MCP) electrical stimulation can substitute as CS;
  unpaired pontine CS produces extinction, confirming the response is a
  true CR. Hesslow et al. 1999 (in decerebrate ferret) demonstrated
  *immediate transfer* — after firmly establishing a forelimb CS, MCP
  stimulation immediately elicits a normally-timed CR. Crucially, with
  lidocaine block of the MCP ventral to the stimulation site,
  forelimb-CS CRs disappear but MCP-CS CRs persist — ruling out
  antidromic activation of pontine inputs as the mechanism. **For the
  simulator:** the MF → cerebellum path is the privileged CS channel;
  the simulator's `mossy_*` source regions can be driven by either a
  peripheral sensory channel or directly with a synthetic spike train
  and the cerebellar microcircuit should produce equivalent CRs.
- **Supplemental:** Hesslow & Yeo 2002 §"Information Processing in CS
  Pathway" pp 126–127 documents the **cerebellar temporal bridge**:
  Svensson & Ivarsson 1999 showed that after training to a 300-ms
  forelimb CS, replacing with a *single 0.2 ms MF-stimulation pulse*
  evokes a normally-timed CR adaptively timed to the US — the cerebellum
  by itself bridges the CS-US interval. The *initial* part of the CS is
  sufficient. Trace-conditioning experiments show this bridging
  capacity has a ceiling around 500 ms of CS-US gap; beyond that, the
  hippocampus is required (Moyer et al. 1990 — hippocampectomised rabbits
  fail trace conditioning when trace > 500 ms but learn fine when <300
  ms). **For the simulator:** an honest cerebellar microcircuit should
  generate adaptive timing for CS-US 100–500 ms from a brief CS pulse;
  the F.17 intrinsic-PC-timer mechanism is the leading candidate substrate.
- **Supplemental:** Hesslow & Yeo 2002 §"Information Processing in CS
  Pathway" p 127 documents **frequency-coded CR latency**: Svensson et al.
  1997 — increasing the stimulation frequency of either a peripheral or
  MCP CS from 50 Hz to 100 Hz produces an *immediate reduction* in CR
  latency; with continued training at the new frequency, latency
  gradually re-adapts back to the originally-timed envelope. Same effect
  for peripheral and MCP CS, proving the timing is computed *after* the
  MF stage. **For the simulator:** a runner that switches CS frequency
  in the middle of a session should reproduce the immediate-shift +
  slow-readapt pattern; this is a validation gate stronger than simple
  acquisition of a CR.
- **Supplemental:** Hesslow & Yeo 2002 §"Cerebellar Cortex Lesions" pp
  106–107 documents the **bilateral vs unilateral lesion** dissociation
  (Gruart & Yeo 1995): unilateral HVI cortical lesions in highly
  overtrained rabbits produce CR deficits that recover with retraining,
  but bilateral HVI cortical lesions produce sustained low CR rates
  with no recovery, AND the few residual CRs are highly variable in
  latency-to-peak — i.e. timing accuracy never recovers. This pairs with
  Ivarsson & Hesslow 1993's electrophysiology showing eyeblink control
  via *decussating* output pathways from both hemispheres. **For the
  simulator:** mirroring the cerebellar microcircuit on both sides (with
  contralateral motor projections) is required to reproduce the
  unilateral-recoverable / bilateral-permanent dissociation.
- **Supplemental:** Hesslow & Yeo 2002 §"Reversible Cerebellar
  Inactivations" pp 116–117 — the **reversible-inactivation methodology**
  is the strongest single line of evidence that the cerebellum is the
  *learning* site, not merely a performance site. Conditioning trials
  given during AIP cold-block / lidocaine / muscimol inactivation
  (Clark et al. 1992; Nordholm et al. 1993; Krupa et al. 1993; Ramnani
  & Yeo 1996; Hardiman et al. 1996) produce no learning: when the block
  is lifted, animals must learn from scratch. The Welsh & Harvey 1991
  contrary report failed to replicate. Critically, *extinction* is also
  blocked by AIP inactivation (Ramnani & Yeo 1996; Hardiman et al. 1996),
  ruling out simple performance / state-dependent-learning explanations.
  Krupa & Thompson 1995 — TTX inactivation of the cerebellar EFFERENTS
  in the brachium conjunctivum *does not* prevent learning; only
  inactivation of the cerebellum itself does. **For the simulator:**
  reversible-inactivation is the gold-standard validation pattern for any
  cerebellar runner — the runner should support muting AIP (or PCs, or
  IO) during training and showing that no plasticity accumulates, then
  un-muting and showing that learning starts from baseline.

### F.09 VOR adaptation — gaze stabilization gain learning

*[from Part V — Movement (Ch 30-39); renumbered from F.58]*

- **System:** vestibulocerebellum (flocculus); vestibular MF input + retinal-slip CF input from IO.
- **Biological role:** Vestibulo-ocular reflex keeps retinal image stable during head motion. Magnifying / minimizing glasses produce retinal slip → IO complex spikes → PF→PC LTD → adjusted VOR gain over hours. Floccular lesion abolishes adaptation but spares baseline VOR.
- **Sim status:** **missing**.
- **Cluster:** F primary.
- **Prerequisites:** F.50–F.55.
- **Citation:** Kandel 6e Ch 37 p 925–928.
- **Behavioral validation:** Open-loop VOR gain measured before / after sustained slip → asymptotic gain change in correct direction.
- **Supplemental:** Marr 1969 §7.2 (p 466–467) gives the conceptual
  template: VOR adaptation is the **learned conditional reflex**, in
  which the IO is driven by a *receptor* (here retinal slip) whose
  activity is *reduced* by the consequences of the corresponding PC
  firing (here adjusted VOR gain → less retinal slip). This is the
  cerebellum implementing a stabilising negative-feedback loop whose
  "context" (head-velocity MF input) is learned. Marr's framing
  predicts: lesion the IO retinal-slip pathway → no gain adaptation;
  shut the cerebellar cortex → baseline VOR preserved (loop closes
  through DCN). Both predictions are confirmed empirically.

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
- **Supplemental:** Hesslow & Yeo 2002 §"Cerebellar Cortex: Zones and
  Microzones" + §"Olivo-Cortico-Nuclear Module" pp 100–103 establish the
  modern microzone view that EVERY cerebellar function — motor and
  non-motor — uses the same processing module: an *olivo-cortico-nucleo-
  olivary loop*. Voogd's original four zones (A medial / B / C / D
  lateral) project to fastigial / vestibular / interpositus / dentate
  respectively; in lobules IV–V the C zone subdivides into C1, C2, C3
  with additional Cx and Y. **The C1 + C3 + Y zones receive climbing
  fibres from the dorsal accessory olive (DAO) and project to anterior
  interpositus.** Each cortical zone is further subdivided into
  microzones (parasagittal strips, parallel to PC dendrites) — each
  microzone contains PCs that all receive the *same olivary input* with
  *the same somatosensory receptive field*, controlling a single muscle
  or muscle group. This gives a topographic motor map at the microzone
  level. **For the simulator:** the cluster F microcircuit should
  declare microzones explicitly: each microzone is a small (~hundred)
  population of PCs with a shared CF source (one IO subzone) and a
  shared AIP target subzone. Cross-microzone connectivity through PFs
  is geometrically defined: PFs run *across* microzones along the
  folium, but PCs within a microzone share CF input — i.e. PFs cross
  many microzones, CFs do not. This is what makes the cerebellum a
  large set of independent perceptrons sharing a parallel-fibre
  substrate.

## Cluster A — Closed BG action-selection loop (project flagship)

## Cluster G — additions

---

## Cluster G — Working memory / PFC / cortical integration

**20 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster H — Motor & spinal output

**25 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster I — Channels & intrinsic dynamics

**23 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster J — Synapses & plasticity rules

**39 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster K — Sensory transduction

**15 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster L — Development & critical periods

**23 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster M — Neuromuscular junction

**4 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster N — Sleep & arousal

**14 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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
- **Supplemental:** Bz Cycle 7 (pp. 175–205) is the in-depth treatment. Two project-relevant additions: (a) **gamma is suppressed in down states** because gamma requires sufficient interneuron drive, which is absent during the down state (Bz Cycle 12 pp. 350–351) — so a sim that toggles between Up and Down NREM regimes naturally produces correct gamma envelopes; (b) **delta vs slow oscillation are distinct** — true slow oscillation is 0.5–1 Hz and cortex-intrinsic, while 1.5–4 Hz delta is thalamocortical (T-type Ca²⁺ rebound, Bz Cycle 7 p. 199) — the catalog conflates them and a sim adding NREM should choose which. Slow oscillation is closer to the experimental "Up state framing of SWRs" referenced in N.06/N.07.

### N.06 Sleep Spindles (10-16 Hz) — thalamocortical reticular ↔ relay

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.55]*

- **System:** Thalamic reticular nucleus (TRN) ↔ thalamocortical relay; T-type Ca²⁺ low-threshold spike mechanism.
- **Biological role:** Stage N2 hallmark. TRN burst hyperpolarizes relay neurons, de-inactivating T-type Ca²⁺ channels; rebound burst when hyperpolarization wanes; resulting Ca²⁺ spike triggers Na⁺ burst that re-excites TRN — closing 100-ms cycle that recurs 12-14×/sec for ~1-2 sec spindle. **Strongly correlated with overnight motor-memory consolidation** (Stickgold).
- **Sim status:** missing — flagship has no thalamic relay vs. reticular distinction; no T-type Ca²⁺ channel; no spindle generator. **Project-actionable for replay:** spindles "open windows" for cortical-hippocampal coordination — adding even a coarse spindle phase variable could nest hippocampal replay events for biologically plausible consolidation timing.
- **Cluster:** N primary, J secondary (memory consolidation), I secondary (T-type channel).
- **Prerequisites:** N.54 (Up state framing), I.x (channels).
- **Citation:** Kandel 6e Ch 44 pp 1081-1083 (Fig 44-2B).
- **Behavioral validation:** EEG: 10-16 Hz waxing/waning spindle bursts in N2; spindle density correlates with motor sequence task improvement after sleep.
- **Supplemental:** Bz Cycle 7 (pp. 195–205) confirms the catalog mechanism (TRN ↔ relay T-type rebound) but adds the cross-frequency framing critical for replay scheduling: spindles *nest inside* slow-oscillation Up states, and SWRs *nest inside* spindle troughs — a triple hierarchy slow-osc(0.5–1 Hz) > spindle(10–16 Hz) > ripple(140–200 Hz) that is the canonical NREM-consolidation frame (Bz pp. 343–351). The project's current "schedule a replay event" approach skips two levels of nesting; if the sim added even a coarse Up-state phase variable, replay-event timing could naturally fall on its trough and reproduce the empirical hierarchy without explicit scheduling code.

### N.07 Hippocampal Sharp-Wave Ripples (SWRs) — NREM replay events ⭐

*[from Part VI — Emotion/Reward (Ch 40-44); renumbered from N.56]*

- **System:** Hippocampal CA3/CA1 high-frequency (140-250 Hz) population bursts during NREM and quiet wakefulness.
- **Biological role:** **Most directly relevant to project's replay implementation.** SWRs compress sequences of place cells in correct or reverse order at ~10-20× speed. Disrupting SWRs during sleep impairs spatial-memory consolidation. SWRs co-occur with cortical Up states / spindle troughs, allowing hippocampus → cortex transfer of compressed sequences.
- **Sim status:** partial — sleep-replay infrastructure exists, but Kandel-style time-compressed sequence replay isn't explicitly modeled. **Top-3 actionable:** the existing replay infra could be upgraded to (a) generate compressed sequences from recent active place-cell trajectories, (b) phase-lock replay events to NREM-stage windows, (c) add reverse-replay variant. Biologically grounded and would test the hypothesis that *replay quality, not replay quantity, is the bottleneck*.
- **Cluster:** N primary, D secondary (hippocampus), J secondary (consolidation).
- **Prerequisites:** D.x (hippocampus), N.54/N.55 (NREM framing), J.x (plasticity replay).
- **Citation:** Kandel 6e Ch 44 pp 1090-1092 (text on Stickgold consolidation; SWR mechanism is fully covered in Ch 54 of Part VIII but referenced here).
- **Behavioral validation:** Closed-loop SWR disruption (electrical stim triggered on detected ripple) during post-task sleep → impaired next-day spatial memory. Forward AND reverse replay observed in CA1.
- **Supplemental:** See full mechanistic detail in AUGMENT D.19 above. Critical additions specific to the N-cluster framing: (a) SWRs occur during *both* NREM and quiet wakefulness — calling them an "NREM replay event" understates their behavioral role. Awake SWRs at choice points (rest periods on the maze) carry forward-trajectory replay that *predicts* the next route taken, suggesting SWRs serve online deliberation in addition to offline consolidation (Bz Cycle 12 pp. 348–351). For the project's sleep-replay infra, this means there's a second-natural-place to fire SWR-like events: at *behavioral pauses inside* the waking task (e.g., when `g11` agent reaches a goal and rests), not only during programmed NREM phases. (b) Closed-loop SWR disruption during sleep impairs next-day spatial memory — Girardeau et al. 2009 (referenced in Bz pp. 347–348) — provides the cleanest causal-test paradigm to validate any SWR module the project adds.

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
- **Supplemental:** Bz Cycle 12 (pp. 343–351) lays out the **two-stage memory model** (Buzsáki 1989) explicitly: (1) waking theta-sequenced encoding stores experience-dependent CA3 recurrent weight changes; (2) sleep SWR-replay drives the *same* sequences (now compressed ~20× further than theta-sequence compression) into neocortex, where late-LTP / synaptic-tag mechanisms convert them into durable cortical traces (Bz pp. 346–347, Frey & Morris 1997 synaptic-tag). This places the project's existing infrastructure squarely within a well-defined two-stage architecture: stage-1 = waking theta storage (mostly missing); stage-2 = sleep replay (partially implemented). The Tononi SHY view sits *alongside*, not against, this — Bz treats SHY as a complementary normalization process operating during the same NREM windows, not a competing theory.

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
- **Supplemental:** Bz Cycle 12 Fig. 12.3 (p. 345) gives the explicit anatomical chain: CA3 burst → CA1 ripple → subiculum → parasubiculum → EC deep layers → widespread neocortex. This is *not* "cortex receives a fuzzy hippocampal signal" — it is a discrete 100-ms compressed packet propagated through a defined output cascade (the output limb that closes the trisynaptic loop on the cortex side). If the project ever builds out N.14, the sim should declare these 5 regions as a chain of `BrainRegion`s downstream of the hippocampus stub, not a single pathway, because each stage in the chain has been shown to gate (sub) or amplify (EC-deep) the SWR packet differently.

## Cluster O — additions

---

## Cluster O — Emotion, reward, motivation

**19 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

### O.02 Phasic dopamine = reward prediction error (Schultz) — TD learning signal

*[from Part V — Movement (Ch 30-39); renumbered from O.50]*

- **System:** ventral tegmental area (VTA) and substantia nigra pars compacta (SNc) DA neurons → striatum, NAcc, PFC.
- **Biological role:** Tonic ~5 Hz firing. Phasic burst on unexpected reward; transfer to predictive cue with learning; **dip below baseline on reward omission**. Quantitatively matches temporal-difference RPE.
- **Sim status:** **partial** — broadcast DA from `current_reward_signal`; `--adaptive-da` adds asymmetric per-action gating with slow-positive / fast-negative tau (Schultz 1998 phasic asymmetry). RPE is not computed inside the simulator from a value function — externally supplied as `reward - baseline`.
- **Cluster:** O primary; C secondary.
- **Prerequisites:** A.50, J.*.
- **Citation:** Kandel 6e Ch 38 p 949–953 (Schultz 2007).
- **Behavioral validation:** Conditioning paradigm: DA burst initially on reward → transfers to CS; reward omission produces dip — see project's reward_ema_pre / surprise-LR-boost machinery.
- **Supplemental — explicit RL-theory mapping:** S&B Ch 6.1 (p. 144) gives TD(0) as `V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) − V(S_t)]`. The bracketed quantity is the **TD error δ_t = R_{t+1} + γV(S_{t+1}) − V(S_t)** — this is exactly the dopamine RPE. Schultz98 derives the same formula independently (Eq. 6, p. 12) as the "effective reinforcement signal" of a TD algorithm with discount factor γ. The project's `current_reward_signal` is currently *not* TD-error: it is `r(t)` directly (no `V(s)` learned, no bootstrapped `V(s′)`). This is why the project reproduces sign (a) — burst on unexpected reward — but not signs (b) cue-transfer and (c) omission-dip of the canonical Schultz triplet (see C.22). **To close this gap algorithmically requires a separable critic**: a population whose readout is `V(s)` and whose pathways learn from the same TD-error δ that drives the actor (see new entry C.30 Actor-Critic). The simplest version is a single linear readout from a "value cortex" that is updated by the same `δ` used for STDP gating; this is one bridge change, no new kernels.
- **Supplemental — Rescorla-Wagner mapping:** S&B Ch 14 (Psychology, in this in-progress edition placeholder, but the math is given in Ch 6.1 and developed historically in Schultz98 p. 11): the Rescorla-Wagner rule `ΔV = αβ(λ − V)` is mathematically identical to TD with discount factor γ = 0 collapsed to a single trial — the "λ − V" error term is `R − V(s)`, an immediate reward minus the current prediction. The project's reward-baseline asymmetric EMA in `--adaptive-da` IS Rescorla-Wagner-with-asymmetric-learning-rates. This is the cleanest classical-conditioning-to-RL bridge and should be cited explicitly when documenting the adaptive-DA mechanism.
- **Supplemental — utility-PE upgrade and two-component framework (Schultz16-NRN, Schultz16-JNT):** Schultz's 2016 reviews update the canonical Schultz97 statement in two important ways that the project should be aware of:
  1. The phasic burst is a **two-component sequence**: Component 1 (60–90 ms latency, unselective detection / salience), Component 2 (~150–300 ms latency, utility-coded value RPE). Schultz16-NRN Figs. 2 & 5; Schultz16-JNT Fig. 1a. These map onto the project's `--surprise-lr-boost` (Component-1 analog) and `--adaptive-da --adaptive-da-ema-decay-negative 0.7` (Component-2 analog) mechanisms — see C.04 Supplemental for the full mapping argument.
  2. The error is on **utility u(x) not raw reward r(x)**: Schultz16-NRN Fig. 4c–d shows DA Component 2 follows the inflected (convex-then-concave) utility curve measured behaviorally with risky-reward fractile procedures. For the moving-goal task this distinction is invisible (binary reward), but any risk-sensitive task in the project's future should insert a utility transform.
- **Supplemental — Pearce-Hall attentional learning rate ↔ surprise-LR-boost:** Schultz16-NRN p. 6 explicitly identifies the initial DA component as the biological substrate of the **Pearce-Hall (1980)** attentional learning rule, in which "surprise salience derived from reward prediction errors enhances the learning rate." This is the textbook label for what the project's `--surprise-lr-boost` does. Citing Pearce & Hall 1980 directly (alongside the Schultz16 reinterpretation) gives the mechanism a 45-year theoretical lineage that pre-dates the project's empirical discovery of its utility.

### O.03 DA modulation of corticostriatal plasticity — three-factor rule

*[from Part V — Movement (Ch 30-39); renumbered from O.51]*

- **System:** glutamatergic cortex→MSN synapse + DA terminal at same dendritic spine.
- **Biological role:** PF/PK pre-post coincidence + DA presence determines LTP vs LTD direction. Three-factor learning rule (Hebb × DA). Different sign at D1 vs D2 MSNs.
- **Sim status:** **implemented** — eligibility-trace × `current_reward_signal` × STDP machinery in `sim/bridge.py`. `cortex→D1` plastic with `stdp_w_max=30`. Aligns with textbook three-factor rule.
- **Cluster:** O primary; J secondary.
- **Prerequisites:** O.50, J.*.
- **Citation:** Kandel 6e Ch 38 p 947–950 (Surmeier 2009).
- **Behavioral validation:** Reward-paired action → cortex→D1 LTP; punishment-paired → LTD or D2 LTP.
- **Supplemental:** Schultz98 §"Possible learning mechanisms using the dopamine signal" (pp. 14–17) gives the **two canonical implementations** of the 3-factor rule that the project's eligibility-trace machinery faithfully reproduces. (i) `Δw = η · r̂ · i · o` — postsynaptic plasticity gated by DA at coincident pre+post activity (Schultz98 Eq. 8, p. 14). (ii) `Δw = η · r̂ · h(i,o)` where `h(i,o)` is an **eligibility trace of conjoint pre/post activity that outlasts the events themselves** (Schultz98 Eq. 9, p. 15). This is exactly the project's design — STDP traces decay over ~1 s and are gated by `current_reward_signal` × `cp_plasticity_gain`. Schultz98 names "prolonged calcium concentration changes, CaMKII formation, and sustained striatal/cortical activity" as the candidate biological substrates (p. 15). Worth citing in the catalog so the AI audience sees the mapping `current_reward_signal × eligibility_trace × STDP` ↔ `r̂ × h(i,o)` is not coincidental but Schultz's own proposal. Sutton & Barto (1981, cited Schultz98 p. 15) is the source of the "eligibility trace" concept itself; S&B Ch 7 (pp. 167–195) is the modern treatment.
- **Supplemental — D1 vs D2 differential plasticity confirmed (Schultz16-NRN p. 10):** the three-factor rule the project implements is biologically substantiated by **opposite-sign plasticity at D1 vs D2 MSNs** under positive vs negative DA. Schultz16-NRN p. 10 cites optogenetic dissociation: "dopamine prolongs transitions to excitatory membrane up states in D1 receptor-expressing striatal direct pathway neurons, but reduces membrane up states and prolongs membrane down states in D2 receptor-expressing striatal indirect pathway neurons" (Schultz16-NRN p. 10, citing Hernandez-Lopez et al. 2000, Surmeier et al. 2007). The project's BG cascade includes both `str_D1_X` and `str_D2_X` populations per action; per-pathway plasticity gating already exists for `cortex_to_d1`. **Open infrastructure:** the symmetric `cortex_to_d2` pathway should be wired with **opposite-sign DA modulation** of plasticity — currently both D1 and D2 cortex-to-striatum projections receive the same sign of `current_reward_signal × eligibility_trace`, but biologically D2 LTP requires DA *dip*, D1 LTP requires DA *burst*. This is one bridge change (sign flip on D2-pathway STDP gating) that would add biological fidelity at near-zero cost.
- **Supplemental — striatum has its own reward neurons (Schultz16-JNT pp. 685–688):** Schultz16-JNT documents that striatal MSNs themselves carry rich reward-related signaling beyond the DA-driven plasticity story: "All groups of striatal neurons process reward information without reflecting sensory stimulus components or movements" (Schultz16-JNT p. 686 §"Pure reward"); some striatal neurons code subjective value (preference-driven) rather than objective amount; some fire to reward prediction errors directly. This means the cortico-striatal three-factor rule is operating in a *target population that is itself reward-tuned*, not a value-neutral substrate. The project's `str_D1_X` populations are currently value-neutral readouts of cortex; a faithful upgrade would have a small fraction of striatal neurons with intrinsic reward-prediction-error sensitivity in addition to the cortex-driven action coding. See new entry **O.22 Striatal action-value coding**.

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
- **Supplemental:** Schultz98 §"Electrical stimulation of dopamine neurons as unconditioned stimulus" (pp. 16–17) lists three principled differences between electrical self-stimulation and natural DA activation that bear on whether self-stim is a clean smoke-test for the BG cascade. (1) Natural rewards activate DA *plus several parallel non-DA reward systems* (NAc, cortex, amygdala — see Schultz98 p. 17–18 §"Cooperation between reward signals"); electrical stim activates DA alone. (2) Electrical stim is unconditional reinforcement — no RPE involved. (3) Electrical stim is delivered *after the action*, not at the predictive cue. Implication for the project's proposed self-stim regression test (currently in O.17): a "phantom reward channel" that just delivers reward when the agent emits action-A is an **unconditional reinforcement** test, not an RPE test, and the BG cascade should pass it trivially via `cortex_A → str_D1_A → motor_A` Hebbian potentiation alone. To make it a genuine RPE test, deliver the phantom reward at the time of a learned predictive cue, then measure whether the dopamine pool's firing-time shifts from cue to reward across trials (the Schultz cue-shift signature, C.22).

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
- **Supplemental:** S&B Ch 11.1 (pp. 257–259) gives the actor-critic architecture (Fig. 11.1, p. 258) that maps directly onto the BG: the **critic** is a state-value function whose TD-error δ_t = R_{t+1} + γV(S_{t+1}) − V(S_t) drives all learning; the **actor** is a policy `π(a|s) = e^{H(s,a)} / Σ_b e^{H(s,b)}` whose preferences are updated by `H(s,A) ← H(s,A) + αδ`. Schultz98 Fig. 9C (p. 13) and Houk, Adams & Barto (1995, cited Schultz98 p. 14) make the **anatomical mapping explicit**: VTA/SNc DA = critic output δ; striatal striosomes (limbic striatum) = critic state-value `V`; striatal matrix (sensorimotor striatum) = actor `H(s,a)`. The project's BG cascade has the matrix side fully (per-action D1/D2 → GPi → thal → motor) but **lacks the explicit `V(s)` representation in striosomes**. Adding a 50-neuron striosome population that learns to output a scalar `V(s)` from cortical input, and routing the resulting `δ = r + γV(s′) − V(s)` to the eligibility-trace gate, is the single highest-leverage architectural upgrade for closing the cue-shift gap (C.22) without abandoning the existing flagship configuration.

### O.19 Value-based decisions — vmPFC / OFC encode subjective value

*[from Part VIII — Memory/Cognition (Ch 52, 54-56); renumbered from O.51]*

- **System:** Ventromedial PFC + orbitofrontal cortex represent expected subjective value; project to striatum and LIP/PFC accumulators. Value modulates drift rate of accumulator.
- **Biological role:** Decisions about preferences (which menu item, which apartment) reduce to evidence accumulation where each option's evidence is its subjective-value samples. Same drift-diffusion math as perceptual decisions.
- **Sim status:** partial — DA-modulated cortex→D1 implements value learning; no separate vmPFC/OFC region encodes scalar value across actions independently of the action selector.
- **Cluster:** O primary, G primary, A secondary.
- **Prerequisites:** explicit value-coding region declaration.
- **Citation:** Kandel 6e Ch 56 pp 1406–1409.
- **Behavioral validation:** vmPFC fMRI BOLD ∝ subjective value; lesions cause reversal-learning deficits and intransitive preferences.
- **Supplemental — DA also codes subjective value, not just OFC (Schultz16-NRN pp. 6–8; Schultz16-JNT pp. 683–685):** the canonical reading of O.19 is that vmPFC/OFC owns the subjective-value representation. Schultz16-NRN/JNT show that **dopamine neurons themselves code subjective value**, evidenced by:
  - Higher DA activations to the **preferred** of two equal-amount juices (Schultz16-NRN p. 7, blackcurrant vs orange juice in monkeys).
  - DA tracks **risk-discounted** subjective value (risk-seekers show enhanced DA to risky cues, risk-avoiders show reduced DA — Schultz16-JNT p. 684, Lak et al. 2014).
  - DA tracks **delay-discounted** subjective value (Schultz16-NRN Fig. 4a; Schultz16-JNT pp. 683–684, hyperbolic decay).
  - DA tracks the **arithmetic sum** of positive and negative subjective values when reward is mixed with aversive (Schultz16-NRN Supplementary S2; Schultz16-JNT p. 684).
  Implication for the project: subjective value is **not** localized to one region — it is computed by the DA system itself and broadcast. The current flagship's reliance on a single `current_reward_signal` scalar is therefore not just a TD-error stand-in; it is also performing subjective-value encoding. If the project ever instantiates a vmPFC/OFC region for explicit value coding (as roadmapped), it should receive DA *as input* (Component-2 utility signal), not just exist as a parallel value computation.

## Cluster P — additions

---

## Cluster P — Disease & neurodegeneration

**37 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

## Cluster Q — Glia & neurovascular

**8 entries.** Numbering reflects discovery order across the textbook chapters where this cluster's mechanisms appear.

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

---

