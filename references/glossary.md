# Canonical Biological Terminology Glossary

**Source:** `references/feature-catalog.md` (Kandel 6e + supplemental texts: Bolam-2000, TK-2017, Tepper-2018, PBR-160, Bz 2006, Schultz98/16, Marr 1969, Albus 1971, Hesslow & Yeo 2002, Sutton & Barto, O'Keefe & Nadel 1978).
**Purpose:** Authoritative reference for terminology used in the neural-simulator
codebase, comments, and documentation. Drives the terminology audit at
`references/terminology-survey.md`.

**Last updated:** 2026-04-29 (initial extraction).

## Conventions

- **canonical_name**: the form to use in NEW prose, comments, and documentation.
- **project_identifier**: stable code shorthand (e.g. `gpi_X`, `str_D1_X`,
  `IZH2007_DOPAMINE`). Kept for backward compatibility with sidecars,
  saved JSONs, and historical findings; NEW code can use the canonical name
  in comments alongside.
- **incorrect_uses**: terms that should be flagged in audits and replaced
  when encountered.
- **cluster**: catalog cluster (A–Q) where the term is primarily defined.

## Cluster index (for cross-reference)

| Cluster | Name |
|---|---|
| A | Closed BG action-selection loop |
| B | Striatal microcircuit & WTA |
| C | Dopamine & neuromodulation |
| D | Hippocampus & sequence learning |
| E | Sensory perception & cortical encoding |
| F | Cerebellum & error-correction |
| G | Working memory / PFC / cortical integration |
| H | Motor & spinal output |
| I | Channels & intrinsic dynamics |
| J | Synapses & plasticity rules |
| K | Sensory transduction |
| L | Development & critical periods |
| M | Neuromuscular junction |
| N | Sleep & arousal |
| O | Emotion, reward, motivation |
| P | Disease & neurodegeneration |
| Q | Glia & neurovascular |

---

## Anatomy — Basal ganglia & midbrain

### Striatum (caudate, putamen, NAc)
- **canonical:** "striatum" (whole structure); "dorsal striatum" / "ventral striatum (NAc)" for subdivisions
- **accepted:** "neostriatum" (older), "STR" (abbreviation only)
- **project:** `str_D1_X`, `str_D2_X`, `str_FS_X`, `str_patch_X` (per-action pools, X ∈ {N, E, S, W})
- **scope:** project models a unified dorsal striatum split into per-action D1/D2 pools; no separate dorsomedial (DMS) / dorsolateral (DLS) compartments; no separate NAc shell/core
- **category:** anatomy
- **cluster:** A, B
- **notes:** catalog flags `dorsomedial vs dorsolateral STR` (A.09) and `patch/matrix` (B.07) splits as missing in project

### Globus pallidus externus (GPe)
- **canonical:** "GPe" or "globus pallidus externus"
- **accepted:** "external globus pallidus"; in rodents "GP" (without distinction) appears in older literature
- **project:** `gpe_X` (prototypic / PV+); `gpe_arky_X` (arkypallidal / PV− per A.13)
- **scope:** indirect-pathway relay; project models prototypic + arkypallidal split as of 2026-04-29
- **category:** anatomy
- **cluster:** A
- **notes:** GPe cell heterogeneity (PV+ prototypic vs PV− arkypallidal) is documented in A.13; the project's `gpe_X` historically meant the prototypic pool — alias preserved

### Globus pallidus internus / SNr (BG output complex)
- **canonical:** "GPi/SNr" or "BG output complex" (the catalog uses both — GPi is dominant in primates, SNr in rodents)
- **accepted:** "globus pallidus internus" (= GPi = entopeduncular nucleus EP in rodents); "substantia nigra pars reticulata" (SNr); GPi/EP
- **project:** `gpi_X` regions; `IZH2007_GPI_OUTPUT` and `HH_GPI_OUTPUT` presets
- **scope:** project models a single output pool per action; in rodents this is predominantly SNr (GPi proper is small), in primates GPi
- **category:** anatomy
- **cluster:** A
- **notes:** project shorthand `gpi_X` covers both GPi and SNr; A.04 supplemental documents perisomatic GPe inputs vs distal striatal inputs (B.16, A.14) — single-compartment project model collapses these

### Substantia nigra pars compacta (SNc)
- **canonical:** "SNc"
- **accepted:** "A9 cell group" (ventral midbrain DA classification, C.16)
- **project:** `dopamine` region (single pool; A9 + A10 collapsed); `IZH2007_DOPAMINE`, `HH_DOPAMINE_SNC` presets
- **scope:** project's dopamine pool collapses A9 (SNc, motor) and A10 (VTA, mesolimbic) — flagged as `[discrepancy]` in C.16
- **category:** anatomy
- **cluster:** C
- **notes:** SNc DA neurons lack KCC2, have depolarized E_Cl (B.15) — current single-compartment project ignores this

### Ventral tegmental area (VTA)
- **canonical:** "VTA"
- **accepted:** "A10 cell group"
- **project:** collapsed into the same `dopamine` region as SNc
- **scope:** mesolimbic / mesocortical DA source; not separately modeled
- **category:** anatomy
- **cluster:** C
- **notes:** flagship's DA = A9-like behavior (broadcast scalar); reward learning is biologically A10-driven — `[discrepancy]` per C.16

### Subthalamic nucleus (STN)
- **canonical:** "STN"
- **accepted:** "subthalamic nucleus"; "Luys' body" (rare)
- **project:** `stn` (single shared region across actions); `IZH2007_STN_BURST`, `HH_STN_BURST` presets
- **scope:** glutamatergic; project models a single shared pool (not per-action)
- **category:** anatomy
- **cluster:** A
- **notes:** dual-mode rebound dynamics (Cav3 short, Cav1.2/1.3 long plateau) per A.16 — not modeled in current Izh/AdEx STN preset

### Thalamus (motor / relay nuclei)
- **canonical:** "thalamus" (functional name); "thalamic relay nucleus" or "TC" (thalamocortical) for cell-type
- **accepted:** "VL/VA" for motor thalamus, "ventroanterior/ventrolateral nuclei"
- **project:** `thal_X` (per-action); `IZH2007_THALAMIC_RELAY`, `HH_THALAMIC_RELAY_TBURST` presets
- **scope:** generic thalamic relay used for BG output; no specific nucleus identification
- **category:** anatomy
- **cluster:** A

### Thalamic reticular nucleus (TRN)
- **canonical:** "TRN"
- **accepted:** "thalamic reticular nucleus", "nucleus reticularis thalami"
- **project:** `IZH2007_THALAMIC_RETICULAR`, `HH_TRN_BURST_INHIB` presets (not currently instantiated in flagship)
- **scope:** GABAergic shell around thalamus; spindle generator (N.06)
- **category:** anatomy
- **cluster:** N

### Pedunculopontine nucleus (PPN / PPT)
- **canonical:** "PPN" (in motor / BG context); "PPT" (pedunculopontine tegmental, sleep / arousal context)
- **accepted:** PPN = PPT; both names refer to the same nucleus; LDT (laterodorsal tegmental) is its partner
- **project:** missing — not instantiated
- **scope:** sensory + reward driver of DA neurons (C.33); REM-on cholinergic (C.18); locomotion initiation via MLR (H.15)
- **category:** anatomy
- **cluster:** C, H, N

### Mesencephalic locomotor region (MLR)
- **canonical:** "MLR"
- **accepted:** "cuneiform + PPN complex" (anatomical components)
- **project:** missing
- **scope:** tonic glutamatergic drive initiating locomotion
- **category:** anatomy
- **cluster:** H

### Inferior olive (IO)
- **canonical:** "IO" or "inferior olive"
- **accepted:** "olivary nucleus"
- **project:** `HH_INFERIOR_OLIVE` preset (no circuit instantiated)
- **scope:** climbing fiber source; teaching signal for cerebellar PF→PC LTD
- **category:** anatomy
- **cluster:** F

---

## Anatomy — Cortex

### Primary motor cortex (M1)
- **canonical:** "M1" or "primary motor cortex"
- **accepted:** "Brodmann area 4", "precentral gyrus"
- **project:** `cortex_X` per-action pools (4 actions × 25 neurons) approximate a coarse M1 map
- **scope:** project's `cortex_X` is functionally an M1 analog; no continuous somatotopy
- **category:** anatomy
- **cluster:** H, G

### Prefrontal cortex (PFC)
- **canonical:** "PFC"; subdivisions "dlPFC" (dorsolateral) / "vmPFC" (ventromedial) / "OFC" (orbitofrontal)
- **accepted:** "frontal association cortex"
- **project:** `pfc` region (60 recurrent neurons); `HH_PFC_PYRAMIDAL` preset
- **scope:** generic working-memory / sustained-activity pool; no dlPFC vs vmPFC vs OFC differentiation
- **category:** anatomy
- **cluster:** G

### Posterior parietal cortex (PPC)
- **canonical:** "PPC"; subdivisions LIP / MIP / AIP
- **accepted:** "parietal association cortex"
- **project:** missing as a region; `goal_cells` region in g11 is closer to PPC than PFC despite naming
- **scope:** spatial planning, reach intention, decision accumulator (LIP)
- **category:** anatomy
- **cluster:** G, E

### Entorhinal cortex (EC)
- **canonical:** "EC"; layers "EC-II", "EC-III", "EC deep" referenced separately
- **accepted:** "entorhinal area", "Brodmann 28"
- **project:** `ec` region (Cluster D hippocampus, opt-in)
- **scope:** project models a single EC pool feeding the trisynaptic loop; layer-specific (II vs III) pathways missing
- **category:** anatomy
- **cluster:** D

### Hippocampus subregions: DG / CA3 / CA1 / CA2 / Subiculum
- **canonical:** "DG" (dentate gyrus), "CA3", "CA1", "CA2", "subiculum"
- **accepted:** "Ammon's horn" subregions (CA1-CA3); "fascia dentata" (= DG)
- **project:** `dg` (with `dg_fs` PV-FSI subpool), `ca3`, `ca1` (Cluster D opt-in via `--enable-cluster-d-hippocampus`); `place_cells` and `goal_cells` from older `--hippocampus` flag
- **scope:** project models the trisynaptic loop (EC → DG → CA3 → CA1) when Cluster D is enabled; CA2 missing (D.15); subiculum missing
- **category:** anatomy
- **cluster:** D
- **notes:** older `--hippocampus` flag uses generic `place_cells` + `goal_cells` regions; new Cluster D flag uses canonical DG/CA3/CA1 names

### Cerebellum subregions
- **canonical:** "cerebellar cortex" (granular + Purkinje + molecular layers); "DCN" (deep cerebellar nuclei: dentate, interposed = anterior interpositus AIP + posterior, fastigial)
- **accepted:** "neocerebellum", "spinocerebellum", "vestibulocerebellum" (functional zones)
- **project:** `HH_CEREBELLAR_PURKINJE`, `HH_CEREBELLAR_GRANULE` presets; no circuit
- **scope:** presets exist but no runner instantiates the cerebellar microcircuit
- **category:** anatomy
- **cluster:** F
- **notes:** F.06 specifies AIP (anterior interpositus) as critical for eyeblink conditioning; project should partition DCN if it ever instantiates the circuit

### Locus coeruleus (LC)
- **canonical:** "LC"
- **accepted:** "A6 cell group"
- **project:** missing — no NE modulator deployed in flagship
- **scope:** sole source of cortical/hippocampal NE
- **category:** anatomy
- **cluster:** C

### Raphe nuclei
- **canonical:** "raphe nuclei"; "dorsal raphe" (DRN), "median raphe" (MRN) for forebrain-projecting populations
- **accepted:** "B5–B9 cell groups" (rostral 5-HT); "B1–B4" (medullary)
- **project:** missing — no 5-HT modulator deployed
- **scope:** midline serotonergic source
- **category:** anatomy
- **cluster:** C

### Tuberomammillary nucleus (TMN)
- **canonical:** "TMN"
- **accepted:** "E1–E5 cell groups"; "histaminergic posterior hypothalamus"
- **project:** missing
- **scope:** sole source of brain histamine
- **category:** anatomy
- **cluster:** C, N

### Basal forebrain (BF)
- **canonical:** "basal forebrain"; "nucleus basalis of Meynert" (NBM, Ch4 cholinergic)
- **accepted:** "Ch1–Ch4 cholinergic groups"; "medial septum + diagonal band of Broca"
- **project:** missing
- **scope:** cortical cholinergic + GABAergic arousal driver
- **category:** anatomy
- **cluster:** C, N

### Suprachiasmatic nucleus (SCN)
- **canonical:** "SCN"
- **accepted:** "circadian master clock"
- **project:** missing
- **scope:** ~20K GABAergic cells; circadian pacemaker (BMAL1/CLOCK/PER/CRY loop)
- **category:** anatomy
- **cluster:** N

### Ventrolateral preoptic nucleus (VLPO)
- **canonical:** "VLPO"
- **accepted:** "ventrolateral preoptic area"; "MNPO" (median preoptic) often grouped with VLPO
- **project:** missing
- **scope:** GABA + galanin sleep-promoting nucleus
- **category:** anatomy
- **cluster:** N

### Amygdala (LA / BLA / CeA)
- **canonical:** "amygdala"; "LA" (lateral), "BLA" (basolateral), "CeA" (central) for nuclei
- **accepted:** "amygdaloid complex"
- **project:** missing — flagship has no amygdala
- **scope:** Pavlovian threat conditioning, valence-and-arousal map
- **category:** anatomy
- **cluster:** O

### Hypothalamus subregions
- **canonical:** "hypothalamus"; sub-nuclei "PVN" (paraventricular), "arcuate", "LH" (lateral), "VMH" (ventromedial), "DMH" (dorsomedial), "preoptic", "TMN", "SFO", "OVLT"
- **accepted:** standard anatomical names
- **project:** missing
- **scope:** homeostatic drives, feeding, thirst, HPA axis
- **category:** anatomy
- **cluster:** O

### Nucleus accumbens (NAc)
- **canonical:** "NAc"; subdivisions "shell" / "core"
- **accepted:** "ventral striatum"; "accumbens"
- **project:** not separately distinguished from dorsal striatum in flagship
- **scope:** central reward hub (O.16); cAMP-CREB pathway adaptation (C.25)
- **category:** anatomy
- **cluster:** O

### Periaqueductal gray (PAG)
- **canonical:** "PAG"; subdivisions "vlPAG" (ventrolateral)
- **accepted:** "midbrain central gray"
- **project:** missing
- **scope:** descending pain modulation (C.11), REM-off (N.04)
- **category:** anatomy
- **cluster:** C, N, O

### Superior colliculus (SC)
- **canonical:** "SC"; "intermediate / deep SC" for motor map
- **accepted:** "tectum" in non-mammals; "optic tectum"
- **project:** missing
- **scope:** topographic saccade motor map; orienting
- **category:** anatomy
- **cluster:** A, H

### Spinal cord
- **canonical:** "spinal cord"; "ventral horn", "dorsal horn", "substantia gelatinosa"
- **project:** `HH_SPINAL_MOTOR`, `HH_SPINAL_INTERNEURON` presets; no circuit
- **scope:** motor pool / spinal CPG / reflex circuits all missing
- **category:** anatomy
- **cluster:** H

### Olfactory bulb (OB)
- **canonical:** "olfactory bulb"; "glomerulus" for the glomerular layer
- **project:** `OLFACTORY_BULB` profile, `HH_OLFACTORY_MITRAL` preset; no input
- **category:** anatomy
- **cluster:** E, K

---

## Cell types — Striatum

### Medium spiny neuron (MSN)
- **canonical:** "MSN" or "medium spiny neuron"; subspecies "D1 MSN" (direct pathway) / "D2 MSN" (indirect pathway)
- **accepted:** "spiny projection neuron" (SPN) — newer literature; "Sp" (striatal projection)
- **project:** `IZH2007_STRIATAL_MSN`, `IZH2007_STRIATAL_MSN_D1`, `IZH2007_STRIATAL_MSN_D2`, `HH_STRIATAL_MSN`, `HH_STRIATAL_MSN_D1`, `HH_STRIATAL_MSN_D2`, `ADEX_MSN` presets
- **scope:** >90% of striatum, GABAergic projection neurons; project uses single-compartment Izh-2007 D1/D2 split
- **category:** cell-type
- **cluster:** B
- **notes:** B.02 emphasizes Up/Down state bistability via KIR2 + Kv-2 (project does not explicitly model)

### D1 MSN (direct pathway / striatonigral)
- **canonical:** "D1 MSN" or "direct-pathway MSN"
- **accepted:** "striatonigral MSN"; "Go-pathway MSN"; PPD/SP (preprodynorphin / substance P) co-releaser
- **project:** `str_D1_X` regions per action; `IZH2007_STRIATAL_MSN_D1`
- **scope:** project models the GABAergic arm via `str_D1_X → gpi_X`; neuropeptide co-release (dynorphin, substance P) opt-in via `--enable-bg-neuropeptides`
- **category:** cell-type
- **cluster:** A, B
- **notes:** STDP soft-bound gotcha — `cortex→D1` weight_mean=25 requires `cfg.stdp_w_max=30`

### D2 MSN (indirect pathway / striatopallidal)
- **canonical:** "D2 MSN" or "indirect-pathway MSN"
- **accepted:** "striatopallidal MSN"; "NoGo-pathway MSN"; PPE (preproenkephalin) co-releaser
- **project:** `str_D2_X` regions per action; `IZH2007_STRIATAL_MSN_D2`
- **scope:** GABAergic arm via `str_D2_X → gpe_X`; enkephalin co-release opt-in
- **category:** cell-type
- **cluster:** A, B
- **notes:** O.03 supplemental flags that biology requires opposite-sign DA modulation of plasticity at D2 vs D1; current project uses same sign for both (open issue)

### PV-FSI (parvalbumin fast-spiking interneuron, striatal)
- **canonical:** "PV-FSI" or "PV-positive fast-spiking interneuron"
- **accepted:** "FSI" (in striatal context); "PV+ basket-equivalent"
- **project:** `str_FS_X` regions; `IZH2007_FS_CORTICAL_INTERNEURON` preset (shared with cortical FSIs)
- **scope:** 1 of 8 striatal GABAergic classes per Tepper-2018; project models PV-FSI specifically
- **category:** cell-type
- **cluster:** B
- **notes:** B.06 — NOT a generic "all striatal interneuron" pool; other 7 classes (NPY-LTS, NPY-NGF, CR, TH/THIN, FAI, SABI, ChI/TAN) not modeled. ~0.7% of striatal neurons (Rymar 2004)

### TAN / ChI (cholinergic tonically-active neuron / cholinergic interneuron)
- **canonical:** "ChI" (anatomical / molecular literature) or "TAN" (electrophysiology / behaving-animal literature) — same cell
- **accepted:** "striatal cholinergic interneuron"; "tonically active neuron"
- **project:** `IZH2007_STRIATAL_TAN`, `HH_STRIATAL_TAN` presets (not currently instantiated)
- **scope:** ~1–2% of striatal neurons; tonic 5–10 Hz; pause response; M1/M2 modulation
- **category:** cell-type
- **cluster:** B
- **notes:** B.05 — preset exists but unused; would gate corticostriatal plasticity

### Striatal LTS interneuron (NPY-LTS)
- **canonical:** "LTS interneuron" or "NPY-LTS" — co-expresses SOM, NPY, nNOS
- **accepted:** older name "PLTS" (plateau-LTS) deprecated 2018 (artifact)
- **project:** missing
- **scope:** beta-resonant slow-inhibition class
- **category:** cell-type
- **cluster:** B
- **notes:** B.08 — formerly called PLTS (incorrect — plateau was whole-cell artifact); use "LTS" or "NPY-LTS"

### Striatal NGF interneuron (NPY-NGF)
- **canonical:** "NGF" or "NPY-NGF" or "neurogliaform"
- **accepted:** "GABA_A-slow interneuron"
- **project:** missing
- **scope:** mediates GABA_A-slow inhibition (decay τ ~120 ms); driven by parafascicular thalamus + Type-II nicotinic from ChIs
- **category:** cell-type
- **cluster:** B
- **notes:** B.09 — *cortical* NGF and *striatal* NGF are not isomorphic; B.01 supplemental flags this discrepancy

### Striatal TH+ interneuron (THIN)
- **canonical:** "THIN" or "TH+ interneuron"
- **accepted:** "tyrosine-hydroxylase-positive striatal interneuron"; explicitly NOT dopaminergic
- **project:** missing
- **scope:** GABAergic despite TH expression; receives reciprocal MSN input
- **category:** cell-type
- **cluster:** B
- **notes:** B.10 — incorrect to call THIN "dopaminergic"; THIN does NOT express VMAT2/DAT and does NOT release DA

### Striatal FAI (fast-adapting interneuron)
- **canonical:** "FAI"
- **accepted:** "Htr3a+ FAI"
- **project:** missing
- **cluster:** B (B.11)

### Striatal SABI (spontaneously active bursty interneuron)
- **canonical:** "SABI"
- **project:** missing
- **cluster:** B (B.12)

### Calretinin interneuron (CR)
- **canonical:** "CR interneuron"
- **project:** missing
- **scope:** primate-dominant; rare in rodent
- **cluster:** B (B.13)

### Striosomal MSN (patch)
- **canonical:** "striosomal MSN" or "patch MSN"
- **accepted:** "patch compartment MSN"
- **project:** `str_patch_X` (R3.11 per-action subpool)
- **scope:** project models a small striosomal subpool per action
- **category:** cell-type
- **cluster:** B

---

## Cell types — Pallidum

### GPe prototypic (PV+) neuron
- **canonical:** "PV+ GPe neuron" or "prototypic GPe neuron"
- **accepted:** "PV-positive pallidal projection neuron"
- **project:** `gpe_X` (canonical pool); `IZH2007_GPE_PACEMAKER`, `HH_GPE_PACEMAKER` presets
- **scope:** ~2/3 of GPe; HFD-pause firing; targets STN/GPi/SNr
- **category:** cell-type
- **cluster:** A
- **notes:** A.13 — older "GPe" without PV+/− distinction is the prototypic pool by convention

### GPe arkypallidal (PV−) neuron
- **canonical:** "arkypallidal neuron" or "PV− GPe neuron"
- **accepted:** "preproenkephalin-mRNA+ GPe neuron"
- **project:** `gpe_arky_X` (R3.7)
- **scope:** ~1/3 of GPe; pallidostriatal feedback projection target
- **category:** cell-type
- **cluster:** A

---

## Cell types — Other (cortical / hippocampal / cerebellar / sensory)

### Cortical pyramidal (RS)
- **canonical:** "pyramidal neuron"; "L5 pyramidal" / "L2/3 pyramidal" for layers
- **accepted:** "regular spiking" (RS) — electrophysiological; "Betz cell" (giant L5 in M1)
- **project:** `IZH2007_RS_CORTICAL_PYRAMIDAL`, `HH_L5_CORTICAL_PYRAMIDAL_RS`, `ADEX_RS_CORTICAL_PYRAMIDAL`, `RS_EXCITATORY_LEGACY` presets
- **category:** cell-type
- **cluster:** G, I

### Cortical FS interneuron (PV+ basket)
- **canonical:** "PV+ FS interneuron" or "cortical fast-spiking interneuron"
- **accepted:** "basket cell" (perisomatic targeting); FS-PV
- **project:** `IZH2007_FS_CORTICAL_INTERNEURON`, `HH_CORTICAL_FS_INTERNEURON`, `ADEX_FS_CORTICAL_INTERNEURON`, `FS_INHIBITORY_LEGACY`
- **scope:** project's FS preset is shared between cortical and striatal FSI use; biologically distinct cell types
- **category:** cell-type
- **cluster:** B, I
- **notes:** B.01 — Martinotti, chandelier, and neurogliaform classes missing entirely

### Chandelier cell
- **canonical:** "chandelier cell" or "axo-axonic cell"
- **scope:** PV+, axon-initial-segment-targeting
- **project:** missing
- **cluster:** B

### Martinotti cell
- **canonical:** "Martinotti cell"
- **accepted:** "SST+ apical-dendrite-targeting interneuron"
- **project:** missing
- **cluster:** B

### Neurogliaform cell (cortical)
- **canonical:** "neurogliaform cell" (cortical)
- **scope:** volume-transmission GABA, slow IPSP
- **project:** missing
- **cluster:** B
- **notes:** distinct from striatal NPY-NGF (see B.01 supplemental)

### Intrinsic bursting (IB), chattering (CH), low-threshold spiking (LTS) cortical phenotypes
- **canonical:** "IB", "CH", "LTS"
- **project:** `IB_EXCITATORY_LEGACY`, `CH_EXCITATORY_LEGACY`, `LTS_INHIBITORY_LEGACY`, `ADEX_IB_BURSTING`, `ADEX_CH_CHATTERING`, `ADEX_LTS_LOW_THRESHOLD`
- **cluster:** I

### Hippocampal pyramidal neuron
- **canonical:** "CA1 pyramidal" / "CA3 pyramidal"
- **project:** `IZH2007_HIPPO_PYRAMIDAL`, `HH_CA1_PYRAMIDAL_BURST`, `HH_CA3_PYRAMIDAL_BURST` presets
- **cluster:** D

### Dentate granule cell (DG)
- **canonical:** "granule cell" or "DG granule cell"
- **project:** part of `dg` region (Cluster D)
- **cluster:** D

### Place cell
- **canonical:** "place cell"
- **scope:** CA1/CA3 pyramidal cell with location-specific firing field; allocentric (D.06 supplemental)
- **project:** `place_cells` region in older `--hippocampus` flag; sensor-driven (catalog flags this as not strictly allocentric per O&N criteria)
- **category:** cell-type / phenomenon
- **cluster:** D

### Grid cell
- **canonical:** "grid cell"
- **scope:** medial EC; periodic hexagonal lattice firing; modular grid spacing
- **project:** missing
- **cluster:** D (D.07)

### Head-direction cell, border cell, object-vector cell, speed cell, time cell
- **canonical:** as named
- **project:** mostly missing; landmark sensors functionally encode object-vector at sensor stage
- **cluster:** D (D.08–D.11)

### Engram cell
- **canonical:** "engram cell"
- **scope:** sparse activity-tagged ensemble storing a specific memory (Tonegawa)
- **project:** missing
- **cluster:** D (D.14)

### Purkinje cell (PC)
- **canonical:** "Purkinje cell"
- **project:** `HH_CEREBELLAR_PURKINJE` preset (no circuit)
- **cluster:** F

### Cerebellar granule cell
- **canonical:** "granule cell" (cerebellum); axon → "parallel fiber" (PF)
- **project:** `HH_CEREBELLAR_GRANULE` preset
- **cluster:** F

### Climbing fiber (CF)
- **canonical:** "climbing fiber"
- **scope:** IO axon → 1:1 PC; teaching signal
- **project:** missing
- **cluster:** F (F.04)

### Mossy fiber (MF)
- **canonical:** "mossy fiber"
- **scope:** in cerebellum: pontine + spinocerebellar + vestibular afferents → granule layer; in hippocampus: DG granule axon → CA3
- **project:** missing in both contexts
- **cluster:** F, D

### Mitral cell, tufted cell
- **canonical:** "mitral cell" / "tufted cell" (olfactory bulb)
- **project:** `HH_OLFACTORY_MITRAL` preset; no circuit
- **cluster:** E, K

### Photoreceptors (rods, cones)
- **canonical:** "rod" / "cone"
- **project:** missing
- **cluster:** K (K.01)

### Hair cell (cochlear, vestibular)
- **canonical:** "hair cell"; "outer hair cell" (OHC) / "inner hair cell" (IHC)
- **project:** missing
- **cluster:** K (K.03–K.06)

### Mechanoreceptors (Pacinian, Meissner, Merkel, Ruffini)
- **canonical:** as named; afferent classes "SA1" / "SA2" / "RA1" / "RA2(PC)"
- **project:** missing
- **cluster:** K (K.07)

### Nociceptor (Aδ, C-fiber)
- **canonical:** "nociceptor"; afferent "Aδ" (myelinated, fast) / "C-fiber" (unmyelinated, slow)
- **project:** missing
- **cluster:** K (K.09)

### α-motoneuron, γ-motoneuron
- **canonical:** "α-motoneuron" (alpha) / "γ-motoneuron" (gamma)
- **accepted:** "α-MN" / "γ-MN"
- **project:** abstract `motor_X` pools; `HH_SPINAL_MOTOR` preset
- **cluster:** H, K

### Renshaw cell
- **canonical:** "Renshaw cell"
- **scope:** spinal recurrent inhibitory interneuron
- **project:** missing; `HH_SPINAL_INTERNEURON` is generic
- **cluster:** H (H.08)

### Muscle spindle Ia/II afferent, Golgi tendon organ Ib afferent
- **canonical:** "Ia afferent" / "II afferent" (spindle); "Ib afferent" (GTO)
- **project:** missing
- **cluster:** K (K.13–K.14)

### V0 / V1 / V2 / V3 spinal interneurons
- **canonical:** as named (developmental class)
- **project:** missing
- **cluster:** L (L.07)

### Astrocyte, oligodendrocyte, microglia, Schwann cell
- **canonical:** as named
- **project:** missing entirely (no glia)
- **cluster:** Q

### Cajal-Retzius cell, radial glia, neural crest cell
- **canonical:** as named (developmental cell types)
- **project:** missing
- **cluster:** L

---

## Receptors

### AMPA receptor
- **canonical:** "AMPA receptor" or "AMPAR"; subunits "GluA1–GluA4"
- **accepted:** "α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid receptor"
- **project:** generic fast excitatory conductance (`E_exc = 0 mV`)
- **scope:** project does not track AMPA explicitly as a named subtype
- **category:** receptor
- **cluster:** J (J.07)

### NMDA receptor
- **canonical:** "NMDA receptor" or "NMDAR"; subunits "GluN1", "GluN2A", "GluN2B"
- **accepted:** "N-methyl-D-aspartate receptor"
- **project:** modeled via `fused_nmda_update_and_current` with voltage-dep Mg²⁺ block
- **category:** receptor
- **cluster:** J (J.08)

### Kainate receptor
- **canonical:** "kainate receptor"; subunits "GluK1–5"
- **project:** missing as distinct receptor (subsumed into AMPA-generic)
- **cluster:** J (J.09)

### GABA-A receptor
- **canonical:** "GABA_A receptor"; subunits e.g. "α1β2γ2"
- **accepted:** "GABAA"; "GABA_A R"
- **project:** modeled with `E_inh = -75 mV` global; per-region overrides for striatum (~−60 mV), SNc DA (~−55 mV) per B.14, B.15
- **category:** receptor
- **cluster:** J (J.10), B
- **notes:** A.15 documents region-specific subunit composition; project treats GABA_A as uniform

### GABA-B receptor
- **canonical:** "GABA_B receptor"
- **scope:** metabotropic, slow, presynaptic autoreceptor
- **project:** missing
- **cluster:** J (C.02)

### Glycine receptor
- **canonical:** "glycine receptor" or "GlyR"
- **scope:** spinal Cl⁻ channel; strychnine-sensitive
- **project:** not modeled (spinal cord absent)
- **cluster:** J (J.11)

### mGluR (metabotropic glutamate receptors)
- **canonical:** "mGluR1–8"; groups I (mGluR1/5), II (mGluR2/3), III (mGluR4/6/7/8)
- **project:** abstracted via NM subsystem
- **cluster:** J, C

### Dopamine receptors
- **canonical:** "D1-like" (D1, D5; Gs-coupled, ↑cAMP) / "D2-like" (D2, D3, D4; Gi-coupled, ↓cAMP)
- **accepted:** specific subtypes "D1", "D2", "D3", "D4", "D5"
- **project:** D1 / D2 modeled functionally via per-pathway DA modulation; specific receptor subunits not differentiated
- **category:** receptor
- **cluster:** C (C.04)
- **notes:** project models the D1/D2 dichotomy; D3/D4/D5 not separately modeled

### Adrenergic receptors
- **canonical:** "α1" (Gq), "α2" (Gi, autoreceptor), "β1/β2/β3" (Gs)
- **project:** missing (NE not deployed)
- **cluster:** C (C.05)

### 5-HT receptors
- **canonical:** "5-HT1A–F" (Gi), "5-HT2A/B/C" (Gq), "5-HT3" (ionotropic — only one), "5-HT4–7" (Gs)
- **project:** missing
- **cluster:** C (C.06)

### Muscarinic ACh receptors (mAChR)
- **canonical:** "M1–M5"; M1/M3/M5 Gq; M2/M4 Gi
- **project:** missing
- **cluster:** C (C.03)

### Nicotinic ACh receptors (nAChR)
- **canonical:** "nAChR"; "α7", "α4β2" (CNS); "muscle-type" (NMJ)
- **project:** missing
- **cluster:** M (M.02), C

### Histamine receptors (H1, H2, H3)
- **canonical:** "H1", "H2", "H3"
- **project:** missing
- **cluster:** C (C.07)

### Opioid receptors
- **canonical:** "μ" (mu, MOR), "δ" (delta, DOR), "κ" (kappa, KOR)
- **project:** abstracted in `--enable-bg-neuropeptides` (dynorphin → KOR; enkephalin → DOR)
- **cluster:** C (C.08, A.01)

### Cannabinoid receptors
- **canonical:** "CB1", "CB2"
- **project:** missing
- **cluster:** J (J.16)

### Adenosine receptors
- **canonical:** "A1" (Gi, inhibitory), "A2A" (Gs, on D2-MSN)
- **project:** missing
- **cluster:** C (C.09)

### Substance P / NK-1 receptor
- **canonical:** "NK-1 receptor"
- **project:** modeled via `--enable-bg-neuropeptides` (substance_p modulator)
- **cluster:** C, B (B.05, A.01)

### MC4R (melanocortin-4 receptor)
- **canonical:** "MC4R"
- **scope:** PVN satiety; AgRP / α-MSH ligands
- **project:** missing
- **cluster:** O (O.06)

### TRP channels (sensory transduction)
- **canonical:** "TRPV1", "TRPM8", "TRPA1"
- **project:** missing
- **cluster:** K (K.09), I

### Piezo1 / Piezo2
- **canonical:** as named
- **scope:** mechanosensitive cation channels
- **project:** missing
- **cluster:** K (K.08), I

---

## Pathways

### Direct pathway
- **canonical:** "direct pathway" — D1 MSN → GPi/SNr disinhibition (action gating)
- **accepted:** "Go pathway"
- **project:** `cortex_X → str_D1_X → gpi_X → thal_X → motor_X` cascade in `g11_bg_runner`
- **category:** pathway
- **cluster:** A (A.01)

### Indirect pathway
- **canonical:** "indirect pathway" — D2 MSN → GPe → STN → GPi/SNr (action suppression)
- **accepted:** "NoGo pathway"
- **project:** `cortex_X → str_D2_X → gpe_X → stn → gpi_X` cascade
- **category:** pathway
- **cluster:** A (A.02)

### Hyperdirect pathway
- **canonical:** "hyperdirect pathway" — cortex → STN → GPi/SNr (rapid global brake)
- **project:** partial — STN exists but no direct cortex→STN pathway declared by default
- **category:** pathway
- **cluster:** A (A.03)

### Pallidostriatal feedback
- **canonical:** "pallidostriatal feedback" — GPe → striatal interneurons (especially FSI)
- **project:** missing — no `gpe_X → str_FS_X` projection
- **cluster:** A (A.10)

### Cortico-BG-thalamo-cortical loops
- **canonical:** "cortico-BG-thalamo-cortical loops" or "Alexander/DeLong loops"
- **scope:** 5 parallel functional loops (motor / oculomotor / dlPFC / lateral OFC / ACC)
- **project:** flagship implements only the motor loop
- **cluster:** A (A.05)

### Mesolimbic, mesocortical, nigrostriatal DA pathways
- **canonical:** "nigrostriatal" (SNc → dorsal striatum), "mesolimbic" (VTA → NAc + amygdala + hippocampus), "mesocortical" (VTA → PFC)
- **project:** collapsed into single `dopamine` pool
- **cluster:** C (C.16)

### Trisynaptic pathway (hippocampus)
- **canonical:** "trisynaptic pathway" — EC-II → DG → CA3 → CA1
- **accepted:** "indirect hippocampal path"
- **project:** declared in Cluster D (`--enable-cluster-d-hippocampus`); regions `ec`, `dg`, `ca3`, `ca1`
- **category:** pathway
- **cluster:** D (D.03)

### Direct entorhinal pathway (temporoammonic)
- **canonical:** "temporoammonic pathway" or "direct EC-III → CA1"
- **project:** missing
- **cluster:** D (D.04)

### Perforant path
- **canonical:** "perforant path"
- **scope:** EC-II → DG axons
- **project:** part of `ec → dg` declared pathway
- **cluster:** D

### Schaffer collateral
- **canonical:** "Schaffer collateral"
- **scope:** CA3 → CA1 axons
- **project:** part of `ca3 → ca1` pathway
- **cluster:** D, J (J.35)

### Mossy fiber pathway (hippocampal)
- **canonical:** "mossy fiber pathway" or "DG → CA3 mossy fiber"
- **project:** part of `dg → ca3` pathway
- **cluster:** D, J (J.36)

### Spinothalamic tract, dorsal column-medial lemniscus
- **canonical:** as named
- **project:** missing
- **cluster:** E, K

### Corticospinal tract
- **canonical:** "corticospinal tract"
- **scope:** M1 layer V Betz cells → spinal motor neurons
- **project:** abstract `cortex_X → motor_X` pathway is the project's analog
- **cluster:** H (H.16)

### Reticulospinal, vestibulospinal, rubrospinal tracts
- **canonical:** as named
- **project:** missing
- **cluster:** H

### Ventral / dorsal visual streams
- **canonical:** "ventral 'what' stream" (V1→V2→V4→IT) / "dorsal 'where/how' stream" (V1→MT→MST→PPC)
- **project:** missing
- **cluster:** E (E.12, E.13)

### Magnocellular / parvocellular / koniocellular pathways
- **canonical:** "magnocellular" / "parvocellular" / "koniocellular"
- **scope:** retina → LGN → V1
- **project:** missing
- **cluster:** E (E.06)

### HPA axis
- **canonical:** "HPA axis" — hypothalamus → pituitary → adrenal cortex
- **scope:** CRH → ACTH → cortisol stress cascade
- **project:** missing
- **cluster:** O (O.09)

---

## Plasticity rules

### LTP / LTD
- **canonical:** "LTP" (long-term potentiation), "LTD" (long-term depression)
- **accepted:** specific forms: "early LTP" (≤1h, no protein synthesis), "late LTP" (protein-synthesis-dependent, CREB)
- **project:** implemented via STDP (Hebbian). Late-LTP (J.18, J.38) missing
- **category:** plasticity
- **cluster:** J (J.28, J.38)

### STDP (spike-timing-dependent plasticity)
- **canonical:** "STDP"
- **accepted:** "Bi-Poo STDP", "Hebbian STDP"
- **project:** `fused_stdp_weight_update` GPU kernel; soft-bound rule with `stdp_w_max`
- **category:** plasticity
- **cluster:** J (J.29)
- **notes:** STDP soft-bound w_max gotcha — collapses weights silently if `weight_mean > stdp_w_max`

### Hebbian / anti-Hebbian
- **canonical:** "Hebbian" (positive correlation strengthens), "anti-Hebbian" (sign flipped, e.g. PF→PC LTD)
- **project:** STDP is Hebbian; anti-Hebbian PF→PC LTD missing (F.05)
- **cluster:** J, F

### Three-factor (R-STDP) rule
- **canonical:** "three-factor learning rule" or "R-STDP"
- **scope:** pre × post × neuromodulator (DA) coincidence
- **project:** implemented via `eligibility_trace × current_reward_signal × STDP`
- **cluster:** O (O.03), J

### Eligibility trace
- **canonical:** "eligibility trace"
- **accepted:** "TD(λ) trace"; "synaptic eligibility" (Schultz 1998)
- **project:** `cp_eligibility_trace` array; `fused_eligibility_trace_decay` kernel
- **cluster:** C (C.29), J

### Heterosynaptic / homosynaptic plasticity
- **canonical:** "heterosynaptic plasticity" / "homosynaptic plasticity"
- **project:** mostly homosynaptic via STDP
- **cluster:** J

### Homeostatic / synaptic scaling
- **canonical:** "synaptic scaling" or "homeostatic plasticity"
- **scope:** activity-dependent global scaling that maintains setpoint firing rate
- **project:** `fused_homeostasis_update` kernel; EMA alpha + threshold adapt rate
- **cluster:** J

### BCM rule
- **canonical:** "BCM rule" (Bienenstock–Cooper–Munro)
- **scope:** sliding modification threshold
- **project:** not explicitly implemented; threshold homeostasis is the closest analog
- **cluster:** J

### STP (short-term plasticity, Tsodyks-Markram)
- **canonical:** "STP" or "short-term plasticity"; subtypes "facilitation" / "depression"
- **accepted:** "Tsodyks-Markram model"
- **project:** `fused_stp_decay_recovery` kernel; `stp_U`, `stp_tau_d`, `stp_tau_f` parameters
- **category:** plasticity
- **cluster:** J (J.03)

### Structural plasticity
- **canonical:** "structural plasticity"; subtypes "synaptogenesis" / "synapse elimination" / "axon pruning"
- **project:** `struct_plast_activity_bias` parameter
- **cluster:** J, L (L.02, L.03)

### PF→PC LTD (Marr-Albus-Ito)
- **canonical:** "PF→PC LTD" or "parallel-fiber → Purkinje LTD"
- **scope:** anti-Hebbian, CF-gated cerebellar plasticity
- **project:** missing
- **cluster:** F (F.05)
- **notes:** Marr 1969 originally predicted LTP; Albus 1971 reversed to LTD; Ito 1982 confirmed Albus — `[discrepancy: Marr 1969 originally predicted LTP, corrected to LTD]`

### Reconsolidation
- **canonical:** "memory reconsolidation"
- **project:** missing
- **cluster:** J (J.27)

### Habituation, sensitization, classical conditioning, operant conditioning
- **canonical:** as named
- **scope:** Aplysia gill-withdrawal model is the canonical experimental substrate
- **project:** classical conditioning at the behavioral level (Pavlovian preset); cell-level sensitization missing
- **cluster:** J (J.24–J.26, J.33)

### TD learning / actor-critic / GPI
- **canonical:** "TD learning" (temporal difference); "actor-critic"; "GPI" (generalized policy iteration)
- **project:** project's reward signaling is partial — no separable critic (C.30)
- **cluster:** O, C (C.28–C.30, O.20)

### Rescorla-Wagner rule
- **canonical:** "Rescorla-Wagner rule"; mathematically TD with γ=0
- **project:** functionally implemented via `--adaptive-da` reward EMA
- **cluster:** O, C (C.28)

### R-learning (average-reward RL)
- **canonical:** "R-learning" or "average-reward formulation"
- **project:** implicitly approximated by `--adaptive-da`
- **cluster:** O (O.21)

### Reservoir computing / echo state
- **canonical:** "reservoir computing"
- **project:** earlier silent-motor-trap arc used reservoir + readout; replaced by BG cascade
- **cluster:** F (Marr-Albus expansion-recoding, F.02, F.12)

---

## Neuromodulators / neurotransmitters

### Glutamate
- **canonical:** "glutamate"
- **project:** generic excitatory channel (AMPA + NMDA)
- **category:** neuromodulator (transmitter)
- **cluster:** C (C.01), J

### GABA
- **canonical:** "GABA" (γ-aminobutyric acid)
- **project:** generic inhibitory channel (`E_inh = -75 mV`)
- **cluster:** C (C.02), J

### Dopamine (DA)
- **canonical:** "dopamine" or "DA"
- **project:** `dopamine` neuromodulator (default config); central NM driving plasticity
- **scope:** project's most fully developed NM; broadcast and per-action variants
- **category:** neuromodulator
- **cluster:** C (C.04)
- **notes:** project's `current_reward_signal` is the DA scalar; biology splits into Component 1 (detection) and Component 2 (utility RPE) — see C.32

### Norepinephrine (NE) / noradrenaline
- **canonical:** "NE" or "norepinephrine"; "noradrenaline" (NA, equivalent in older / European literature)
- **project:** missing — NM framework supports it but not deployed
- **cluster:** C (C.05, C.14)

### Serotonin (5-HT)
- **canonical:** "5-HT" or "serotonin"
- **project:** missing
- **cluster:** C (C.06, C.15)

### Acetylcholine (ACh)
- **canonical:** "ACh" or "acetylcholine"
- **project:** `acetylcholine` neuromodulator default config (opt-in); striatal TANs (cholinergic) preset exists but unused
- **cluster:** C (C.03, C.18, C.19)

### Histamine
- **canonical:** "histamine"
- **project:** missing
- **cluster:** C (C.07, C.17)

### Adenosine
- **canonical:** "adenosine"
- **scope:** sleep pressure / Process-S
- **project:** missing
- **cluster:** C (C.09), N (N.08)

### Endocannabinoids (anandamide, 2-AG)
- **canonical:** "endocannabinoid" or "eCB"; ligands "anandamide" / "2-AG"
- **project:** missing
- **cluster:** J (J.16)

### Nitric oxide (NO)
- **canonical:** "NO" or "nitric oxide"
- **project:** missing
- **cluster:** J (J.17)

### Dynorphin
- **canonical:** "dynorphin"; "PPD" (preprodynorphin precursor)
- **scope:** D1 MSN co-release with GABA + substance P; KOR ligand
- **project:** `dynorphin` neuromodulator (opt-in, `--enable-bg-neuropeptides`)
- **cluster:** C, A (A.01, B.03)

### Enkephalin
- **canonical:** "enkephalin" (met-/leu-); "PPE" (preproenkephalin precursor)
- **scope:** D2 MSN co-release with GABA; DOR ligand
- **project:** `enkephalin` neuromodulator (opt-in)
- **cluster:** C, A (A.02, B.03)

### Substance P
- **canonical:** "substance P"; precursor "PPT" (preprotachykinin)
- **scope:** D1 MSN co-release with GABA + dynorphin; NK-1 ligand
- **project:** `substance_p` neuromodulator (opt-in)
- **cluster:** C (B.05, A.01, C.08)

### β-endorphin / endogenous opioids
- **canonical:** "β-endorphin"; family "endogenous opioids"
- **project:** missing
- **cluster:** C (C.08, C.11)

### Oxytocin, vasopressin
- **canonical:** as named
- **project:** missing (would be needed for CA2 social memory, D.15)
- **cluster:** C (C.08), D

### NPY (neuropeptide Y)
- **canonical:** "NPY"
- **project:** missing as a modulator
- **cluster:** C (C.08), B

### Somatostatin (SST / SOM)
- **canonical:** "SST" or "somatostatin"
- **project:** missing
- **cluster:** C (C.08), B

### CRH / cortisol
- **canonical:** "CRH" (corticotropin-releasing hormone) / "cortisol"
- **project:** missing
- **cluster:** O (O.09)

### Leptin, insulin, ghrelin, AgRP, α-MSH, POMC, CCK
- **canonical:** as named
- **scope:** feeding / energy balance hormones
- **project:** missing
- **cluster:** O (O.06, O.07)

### Orexin / hypocretin
- **canonical:** "orexin" or "hypocretin"
- **project:** missing
- **cluster:** N (N.11)

### Galanin
- **canonical:** "galanin"
- **scope:** VLPO sleep-promoting cotransmitter with GABA
- **project:** missing
- **cluster:** N (N.02)

### ATP / purinergic
- **canonical:** "ATP" / "purinergic transmission"
- **project:** missing
- **cluster:** C (C.09)

### D-serine, glycine (as NMDAR co-agonists)
- **canonical:** "D-serine"; "glycine"
- **scope:** astrocyte gliotransmission
- **project:** missing
- **cluster:** Q (Q.04), J

---

## Phenomena / oscillations / states

### Up state / Down state
- **canonical:** "Up state" / "Down state"; "slow oscillation" (0.5–1 Hz, NREM)
- **project:** missing — flagship doesn't generate Up/Down alternation
- **cluster:** N (N.05), B (B.02 supplemental — MSN bistability)

### Theta rhythm
- **canonical:** "theta rhythm"; band 4–12 Hz; septal-driven
- **project:** missing
- **cluster:** D (D.18)

### Gamma oscillation
- **canonical:** "gamma" or "gamma oscillation"; band 40–100 Hz
- **accepted:** "ING" (interneuron-network gamma) / "PING" (pyramidal-interneuron gamma)
- **project:** validated via `gamma-oscillations` benchmark (PING via FS interneurons)
- **cluster:** I, N (N.19), J

### Sleep spindle
- **canonical:** "sleep spindle"; band 10–16 Hz; thalamocortical (TRN ↔ relay)
- **project:** missing
- **cluster:** N (N.06)

### Sharp-wave ripple (SWR)
- **canonical:** "SWR" or "sharp-wave–ripple"; ripple band 140–200 Hz
- **accepted:** "ripple" alone (in CA1 LFP context)
- **project:** partial — sleep-replay infrastructure exists; SWR detection / sequence content missing
- **cluster:** N (N.07, N.16), D (D.19)

### Theta-gamma coupling
- **canonical:** "theta-gamma cross-frequency coupling"
- **project:** missing
- **cluster:** N (N.15)

### NREM / REM sleep
- **canonical:** "NREM" (non-REM, stages N1/N2/N3); "REM" (rapid eye movement)
- **project:** sleep-replay infrastructure has scheduled phases; biological flip-flop generators missing
- **cluster:** N

### Replay (forward / reverse / awake)
- **canonical:** "replay"; "forward replay" / "reverse replay"; "awake replay" (during quiet wakefulness)
- **project:** sleep-replay infra exists; content quality / sequence compression / awake replay missing
- **cluster:** N (N.07, N.17), D

### Phase precession
- **canonical:** "phase precession"
- **scope:** place cell spike phase advances within theta cycle
- **project:** missing
- **cluster:** D (D.18 supplemental)

### Pattern separation / pattern completion
- **canonical:** "pattern separation" (DG); "pattern completion" (CA3)
- **project:** missing as validated dynamic; would emerge from Cluster D trisynaptic implementation
- **cluster:** D (D.12, D.13)

### Remapping (global / rate)
- **canonical:** "global remapping" / "rate remapping"
- **scope:** place-cell ensemble decorrelation between environments
- **project:** missing
- **cluster:** D (D.17)

### Reward prediction error (RPE)
- **canonical:** "RPE" or "reward prediction error"; "TD error" (algorithmic)
- **project:** partial — `current_reward_signal` is `r(t)` not `r(t) + γV(s′) − V(s)`; no critic
- **cluster:** C (C.22, C.28)
- **notes:** Schultz16 refines to "utility prediction error" — DA codes inflected u(x) (C.34)

### Two-component DA response
- **canonical:** "Component 1" (detection / salience, 60–90 ms) / "Component 2" (value / utility RPE, 150–300 ms)
- **project:** partial — `--surprise-lr-boost` is a Component 1 analog; `--adaptive-da` is a Component 2 analog
- **cluster:** C (C.32, C.20)

### Wanting vs liking
- **canonical:** "wanting" (DA-mediated incentive salience) vs "liking" (μ-opioid / endocannabinoid hedonic hotspots)
- **project:** missing — only one reward axis
- **cluster:** O (C.27)

### Eyeblink classical conditioning, VOR adaptation
- **canonical:** as named
- **scope:** canonical cerebellar learning paradigms
- **project:** missing (cluster F closed-circuit not built)
- **cluster:** F (F.08, F.09)

### Drift-diffusion model
- **canonical:** "drift-diffusion model" (DDM); "bounded evidence accumulation"
- **project:** functionally approximated by BG cascade
- **cluster:** G (G.16, G.17)

### Cognitive map
- **canonical:** "cognitive map" (O'Keefe & Nadel 1978)
- **scope:** allocentric Euclidean spatial framework
- **project:** missing as theory-level commitment
- **cluster:** D (D.21)

### Locale vs taxon systems
- **canonical:** "locale system" (hippocampus, allocentric, map-based) vs "taxon system" (striatum, egocentric, route-based)
- **project:** Phase B BG cascade ≈ taxon; hippocampus stub ≈ locale
- **cluster:** D, B (D.22)

### Place field
- **canonical:** "place field"
- **scope:** location-specific firing region of a place cell
- **cluster:** D

### Population coding / population vector
- **canonical:** "population coding"; "population vector" (Georgopoulos 1986)
- **project:** partial via per-action pools
- **cluster:** E (E.03), H (H.17)

### Receptive field (RF)
- **canonical:** "receptive field"
- **cluster:** E (E.02)

### Topographic / somatotopic / retinotopic / tonotopic maps
- **canonical:** as named
- **project:** Cluster E v1 partial; flat populations otherwise
- **cluster:** E (E.04)

### Cortical column / ocular dominance / orientation pinwheel
- **canonical:** as named
- **project:** missing
- **cluster:** E (E.10)

### Lateral inhibition / center-surround
- **canonical:** "lateral inhibition"; "center-surround antagonism"
- **project:** MSN lateral inhibition (`--bg-lateral-inhibition`) is the same algorithmic motif used for action WTA
- **cluster:** E (E.05), B

### Saltatory conduction
- **canonical:** "saltatory conduction"
- **scope:** myelinated axon AP propagation
- **project:** missing (no axonal compartments)
- **cluster:** I (I.19)

### Refractory period (absolute, relative)
- **canonical:** "refractory period"
- **cluster:** I (I.20)

### Action potential
- **canonical:** "action potential" or "AP"; "spike"
- **project:** modeled in Izh / HH / AdEx
- **cluster:** I (I.02)

### Neuromuscular junction (NMJ)
- **canonical:** "NMJ" or "neuromuscular junction"
- **project:** missing
- **cluster:** M (M.01)

### Quantal release
- **canonical:** "quantal release"; "miniature EPSP/EPSC" (mEPP/mEPSC)
- **scope:** Katz; Ca²⁺-fourth-power dependence
- **project:** abstracted into spike-driven discrete events
- **cluster:** J (J.20, J.23, M.04)

### Critical period
- **canonical:** "critical period"; "ocular dominance critical period" specifically
- **project:** functionally captured by curriculum + plasticity gates
- **cluster:** L (L.04, L.19)

### Adult neurogenesis
- **canonical:** "adult neurogenesis"
- **scope:** DG granule cells + SVZ → olfactory bulb
- **project:** missing
- **cluster:** L (L.20)

### Engram, false memory, schema
- **canonical:** as named
- **project:** missing
- **cluster:** D (D.14), J (J.34)

### CPG (central pattern generator)
- **canonical:** "CPG" or "central pattern generator"
- **project:** missing
- **cluster:** H (H.13)

### Mirror neuron
- **canonical:** "mirror neuron"
- **project:** missing
- **cluster:** H (H.18)

### Henneman size principle
- **canonical:** "Henneman size principle" or "orderly recruitment"
- **project:** missing (motor pools homogeneous)
- **cluster:** H (H.03)

### Saccade, VOR (vestibulo-ocular reflex)
- **canonical:** as named
- **project:** missing (no eye plant)
- **cluster:** H (H.24, F.09)

### Stretch reflex, reciprocal inhibition, Renshaw inhibition
- **canonical:** as named
- **project:** missing
- **cluster:** H (H.06–H.08)

### Volume transmission
- **canonical:** "volume transmission"
- **scope:** non-synaptic diffuse release of monoamines / neuropeptides
- **project:** modeled (architecturally fit) — `sim/neuromodulators.py` global concentration scalars
- **cluster:** C (C.21)

### Tripartite synapse, gliotransmission
- **canonical:** "tripartite synapse"; "gliotransmission" (astrocyte ATP/D-serine/glutamate release)
- **project:** missing
- **cluster:** Q (Q.04), J

### Sleep-dependent memory consolidation
- **canonical:** "memory consolidation"; subtypes "active replay" (Stickgold) / "synaptic homeostasis" (SHY, Tononi)
- **project:** SHY-like synaptic scaling implemented; replay infra exists
- **cluster:** N (N.12), J

### Two-process model (sleep)
- **canonical:** "two-process model"; "Process-S" (homeostatic) × "Process-C" (circadian)
- **project:** missing
- **cluster:** N (N.09)

### Glymphatic clearance
- **canonical:** "glymphatic system" / "glymphatic clearance"
- **project:** missing
- **cluster:** N (N.13), Q

### Olds-Milner brain stimulation reward
- **canonical:** "Olds-Milner self-stimulation"
- **scope:** medial forebrain bundle electrical reward
- **project:** could be implementable as a regression test
- **cluster:** O (O.17)

### Berridge incentive salience theory
- **canonical:** "incentive salience"
- **scope:** "wanting" half of wanting/liking (C.27)
- **cluster:** O

### Pearce-Hall attentional learning
- **canonical:** "Pearce-Hall attentional learning rule" (Pearce & Hall 1980)
- **scope:** surprise-driven learning rate boost
- **project:** functionally implemented as `--surprise-lr-boost`
- **cluster:** C (C.04 / C.32 supplemental)

### Inactivation response (cerebellar)
- **canonical:** "inactivation response" (Granit & Phillips 1956)
- **scope:** PC simple-spike pause after CF complex spike
- **cluster:** F (F.04)

### Codon representation (cerebellar)
- **canonical:** "codon" (Marr 1969)
- **scope:** sparse expansion-recoding via granule cell layer
- **project:** missing (Marr-Albus codon machinery not built)
- **cluster:** F (F.12)

### NMR (nictitating membrane response)
- **canonical:** "NMR" or "nictitating membrane response"
- **scope:** rabbit eyeblink-conditioning model
- **cluster:** F (F.06, F.08)

---

## Disease (Cluster P)

### Parkinson's disease (PD)
- **canonical:** "Parkinson's disease" or "PD" — possessive form preferred per catalog
- **accepted:** "Parkinson disease" (without possessive); "PD"
- **scope:** SNc DA neuron loss → indirect-pathway dominance → bradykinesia / rigidity / tremor; α-synuclein Lewy pathology
- **project:** missing as deployed disease module; testable via DA ablation in BG cascade
- **cluster:** P (P.01, P.29)

### Huntington's disease (HD)
- **canonical:** "Huntington's disease" or "HD"
- **scope:** CAG-repeat expansion in HTT; D2 MSN loss preferentially
- **project:** missing; testable via D2 pool ablation
- **cluster:** P (P.02, P.31)

### Schizophrenia
- **canonical:** "schizophrenia"
- **scope:** hyperdopaminergic mesolimbic + hypodopaminergic mesocortical; positive/negative/cognitive symptoms
- **project:** missing
- **cluster:** P (P.04, P.19, P.20, P.21)

### OCD / Tourette syndrome
- **canonical:** "OCD" (obsessive-compulsive disorder) / "Tourette syndrome"
- **project:** missing
- **cluster:** P (P.03)

### Alzheimer's disease (AD)
- **canonical:** "Alzheimer's disease" or "AD"
- **scope:** β-amyloid + tau; hippocampal synapse loss
- **project:** missing
- **cluster:** P (P.30)

### ALS / amyotrophic lateral sclerosis
- **canonical:** "ALS"; "amyotrophic lateral sclerosis"
- **project:** missing
- **cluster:** P (P.10, P.32)

### Myasthenia gravis (MG), Lambert-Eaton (LEMS)
- **canonical:** "myasthenia gravis" / "Lambert-Eaton myasthenic syndrome"
- **project:** missing
- **cluster:** P (P.07, P.08), M

### Botulism / tetanus
- **canonical:** "botulinum toxin" / "tetanus toxin"; SNARE cleavage
- **project:** missing
- **cluster:** P (P.09)

### Charcot-Marie-Tooth, demyelinating neuropathies
- **canonical:** "Charcot-Marie-Tooth disease" or "CMT"
- **cluster:** P (P.11)

### Spinal muscular atrophy (SMA)
- **canonical:** "SMA"; SMN1 deletion
- **cluster:** P (P.12)

### Epilepsy / seizures (focal, generalized, absence)
- **canonical:** "epilepsy"; subtypes "focal-onset seizure", "absence seizure", "tonic-clonic seizure"
- **scope:** thalamocortical 3 Hz oscillation in absence
- **project:** missing
- **cluster:** P (P.13–P.17)

### Channelopathies (Dravet, KCNQ2, GABRG2)
- **canonical:** as named; gene symbols italic in formal use
- **cluster:** P (P.15), I (I.23)

### Major depressive disorder (MDD), bipolar disorder, anxiety disorders
- **canonical:** as named
- **project:** missing
- **cluster:** P (P.22–P.24)

### Autism (ASD), Fragile X, Rett, tuberous sclerosis
- **canonical:** "autism spectrum disorder" or "ASD"; "Fragile X syndrome"; "Rett syndrome"; "tuberous sclerosis complex" (TSC)
- **scope:** synaptic-adhesion (NRXN/NLGN/SHANK), FMR1 → mGluR-LTD, MECP2, mTOR
- **project:** missing
- **cluster:** P (P.25–P.28)

### PTSD
- **canonical:** "PTSD"; "post-traumatic stress disorder"
- **scope:** failed vmPFC inhibition of amygdala
- **cluster:** O (O.15), P

### Stroke
- **canonical:** "stroke"; "ischemic stroke"
- **cluster:** P (P.36)

### Excitotoxicity
- **canonical:** "excitotoxicity"
- **scope:** NMDA-mediated cell death
- **cluster:** P (P.37)

### Prion disease, spinocerebellar ataxia, FTD/C9ORF72
- **canonical:** "prion disease" / "spinocerebellar ataxia" (SCA) / "frontotemporal dementia" (FTD)
- **cluster:** P (P.32–P.34)

### Narcolepsy, REM behavior disorder
- **canonical:** as named
- **scope:** narcolepsy = orexin loss
- **cluster:** N (N.11), P

### Aphasias (Broca, Wernicke, conduction)
- **canonical:** "Broca's aphasia" / "Wernicke's aphasia" / "conduction aphasia"
- **cluster:** G (G.10–G.13), P

### Agnosias (prosopagnosia, achromatopsia, hemineglect)
- **canonical:** as named
- **cluster:** P (P.06)

---

## Drugs / pharmacology (selected referenced)

### Levodopa
- **canonical:** "levodopa" or "L-DOPA"
- **scope:** DA precursor; PD therapy
- **cluster:** P

### DBS (deep brain stimulation)
- **canonical:** "DBS"; targets "STN-DBS" / "GPi-DBS"
- **cluster:** H (H.21)

### SSRI / NRI / antidepressants
- **canonical:** "SSRI" (selective serotonin reuptake inhibitor) / "NRI"
- **cluster:** C, N

### Caffeine, cocaine, amphetamine, methylphenidate
- **canonical:** as named
- **scope:** A1/A2A blocker (caffeine); DAT/DA reuptake inhibitors (cocaine, amphetamine — C.35 stimulant smearing)
- **cluster:** C (C.09, C.35)

### Anti-NMDA (ketamine)
- **canonical:** "ketamine"; class "NMDA antagonist"
- **cluster:** C, P

### Strychnine, bicuculline, picrotoxin
- **canonical:** as named
- **scope:** GlyR antagonist (strychnine); GABA-A antagonists (bicuculline, picrotoxin)
- **cluster:** J

---

## Mathematical / RL (project-specific terms used in catalog)

### TD(λ), Q-learning, SARSA
- **canonical:** "TD(λ)"; "Q-learning"; "SARSA"
- **scope:** Sutton & Barto canonical algorithms
- **cluster:** O, C

### Actor-critic
- **canonical:** "actor-critic"; biologically: striatal matrix = actor, striosome = critic
- **project:** actor implemented; critic missing
- **cluster:** O, C (C.30)

### Bootstrapping (RL)
- **canonical:** "bootstrapping"
- **scope:** updating a guess from a guess; distinguishes TD from Monte Carlo
- **cluster:** C (C.31)

### Drift-diffusion (DDM)
- See "Phenomena" section above

### Generalized Policy Iteration (GPI)
- See "Plasticity rules" section

---

## Common incorrect / deprecated terms (audit flags)

The following terms appear in older literature or were specifically flagged by the catalog:

- **"PLTS" (plateau-LTS striatal interneuron)**: deprecated 2018 — plateau was a whole-cell artifact. Use "LTS" or "NPY-LTS" (B.08).
- **"all striatal interneurons"** as a single class: incorrect — at least 8 distinct GABAergic classes (Tepper-2018) plus ChI/TAN. Use specific class names (B.01 supplemental).
- **Treating cortical-interneuron taxonomy (basket / chandelier / Martinotti / neurogliaform) as applying to striatum**: incorrect; striatum has its own non-isomorphic taxonomy (B.01 supplemental).
- **Calling THIN "dopaminergic"**: incorrect — TH+ but not VMAT2/DAT-positive; releases GABA, not DA (B.10).
- **"GP" without disambiguation between GPe and GPi**: ambiguous; in rodents older literature used "GP" alone but modern usage requires GPe / GPi explicit. Project's `gpi_X` covers GPi/SNr collectively which is acceptable as project shorthand.
- **Calling DA "the pleasure signal" or "the hedonic signal"**: incorrect — DA is the *teaching/incentive* signal; hedonic "liking" is opioid/cannabinoid (Berridge). C.27, C.24.
- **"DA = salience" only**: incomplete — Schultz16 argues DA is predominantly appetitive with weak / context-dependent aversive activations. C.24.
- **"Phasic DA = single homogeneous burst"**: incomplete — Schultz16 documents two-component temporal structure (Component 1 detection at 60–90 ms, Component 2 utility at 150–300 ms). C.32.
- **"PF→PC LTP"** (Marr's original 1969 prediction): superseded by LTD (Albus 1971, confirmed Ito 1982). F.05 — `[discrepancy]`.
- **"NREM = SWR generator"**: causally inverted — SWRs are intrinsic to CA3 recurrent network; NREM is the gate, not the source (N.16).
- **"Striatum = single homogeneous structure"**: misses dorsomedial / dorsolateral functional split (A.09) and patch / matrix anatomical split (B.07).
- **Conflating "delta" (1.5–4 Hz, thalamocortical T-type rebound) with "slow oscillation" (0.5–1 Hz, cortex-intrinsic)**: distinct mechanisms (N.05 supplemental).
- **"DG NGF"** vs **"cortical NGF"**: not isomorphic — distinct cell types with overlapping name (B.01, B.09).
- **"Place cells are sensor-driven"**: incomplete per O&N — true place cells should be allocentric (fire on subsequent traversals after sensory cues are removed). Project's current `--learned-perception --landmarks` falls short of this criterion (D.06 supplemental).
- **"Replay = sleep-only consolidation"**: incomplete — ~50% of SWRs occur during quiet wakefulness and serve online deliberation (N.17).
- **Treating eyeblink CR purely as "PF→PC LTD"**: contested — Hesslow 2013 §2 lists four challenges; intrinsic PC timer mechanisms also involved (F.05 supplemental). `[discrepancy]`.

---

## Ambiguous / [NEEDS-REVIEW] entries

- **`[NEEDS-REVIEW]` GPi vs SNr in project shorthand**: project's `gpi_X` collectively means "BG output complex" (functionally GPi/SNr). Catalog notes this is biologically reasonable since rodent SNr ≈ primate GPi. Glossary keeps `gpi_X` as covering both; auditor should not flag instances where `gpi_X` actually represents SNr-equivalent in rodent context.
- **`[NEEDS-REVIEW]` "FSI" disambiguation**: project uses `IZH2007_FS_CORTICAL_INTERNEURON` for both cortical and striatal FSIs (`str_FS_X` and `cortex_FS_X`). Biologically these are distinct populations. Acceptable as engineering shortcut, but audit should note when "FSI" appears without anatomical qualifier.
- **`[NEEDS-REVIEW]` "DA" vs "current_reward_signal"**: project's `current_reward_signal` is functionally a single DA scalar that conflates phasic/tonic, Component-1/Component-2, A9/A10. Catalog (C.04, C.20, C.32) flags this as a major simplification. Audit should not flag every use of `current_reward_signal` but should note when biological distinctions matter.
- **`[NEEDS-REVIEW]` "place cell" usage**: project documents place-cell-like activations, not strictly allocentric place cells per O&N (1978). Audit may distinguish "place-cell-like" (sensor-driven) from "true place cell" (allocentric).
- **`[NEEDS-REVIEW]` "PFC" subdivisions**: project's `pfc` region is generic; biology distinguishes dlPFC / vmPFC / OFC. Audit should not require subdivision in current code but may flag for future clarity.
- **`[NEEDS-REVIEW]` "hippocampus" without subregion**: older `--hippocampus` flag uses generic `place_cells` + `goal_cells`; new `--enable-cluster-d-hippocampus` uses canonical DG/CA3/CA1. Audit should note when "hippocampus" appears without subregion specification — both forms valid in their respective contexts.
- **`[NEEDS-REVIEW]` "neurotransmitter" vs "neuromodulator"**: catalog J.13 flags that the project's NM subsystem abstracts both. Glutamate / GABA = transmitters; DA / NE / 5-HT / ACh / histamine = modulators (also use receptors that are GPCRs). Project's `NeuromodulatorConfig` handles both classes; "neuromodulator" in project usage is broader than strict biological definition.
