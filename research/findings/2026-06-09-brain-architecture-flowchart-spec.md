# Simulated-brain architecture flowchart — complete as-implemented SPEC

**Date:** 2026-06-09
**Type:** read-only deep-extraction / layout plan (no code changed, no jobs run)
**Purpose:** the complete structured inventory + Graphviz layout plan for the ENTIRE
simulated brain — every region, every distinct pathway, signal direction, receptor /
nature, plasticity, gating — AS ACTUALLY IMPLEMENTED in the code, with an honesty layer
distinguishing faithful models from documented shortcuts. A visual-design agent renders
this into the diagram.

## How to read this spec

The brain is not one fixed network — it is **assembled per config** from a declarative
region/pathway grammar (`sim/regions.py`). Two builder functions assemble almost the
entire architecture, gated by ~80 opt-in flags:

- **`build_bg_brain_regions`** (`research/runners/g11_bg_runner.py:131`) — the
  **NAVIGATION** brain: BG action-selection cascade, spiking-SNc actor-critic + neural
  value critic, hippocampus, cerebellum, superior-colliculus orienting/commit, thalamus +
  TRN, visual ventral stream, dlPFC working memory, plus a text-I/O bolt-on.
- **`build_biological_brain_regions`** (`research/runners/text_minimal_isolation.py:173`)
  — the **CONVERSATIONAL** brain: language_input/output, Wernicke (multi-pool), semantic
  cortex, Broca + motor_speech, concept pools (noun/verb/adjective/motor), dlPFC verb WM,
  hippocampal consolidation, episodic context, visual ventral stream, multimodal hub.
- (`build_minimal_brain_regions`, same file:59, is a strict subset of the conversational
  builder — language_input + 4 motor_X + optional motor_FS — used for isolation tests. Not
  a separate architecture; do NOT draw it separately.)

Both run on the SAME core `SimulationBridge` engine (the per-region slices, the conductances,
the plasticity). They are **alternative configurations of one substrate**, not two brains.
The diagram should show them as two large config-scoped supergroups sharing the common
mechanism layer, with a few genuinely shared region TYPES (hippocampal trisynaptic loop,
visual ventral stream, language_input/output) appearing in both.

**Per-action replication:** the navigation BG cascade is built as a **×4 template** replicated
over `ACTION_NAMES = ["N","E","S","W"]` (`g11_bg_runner.py:68`). Draw the template ONCE with
a "×4 (N/E/S/W)" annotation rather than 4 copies. Same for concept pools (×4 per kind) and
cerebellar Purkinje/DCN (×4). This is called out explicitly in §5.

---

## 1. SUBSYSTEM CLUSTERS (layout groups)

Cluster → regions, with which config builds it. `[nav]` = navigation builder, `[conv]` =
conversational builder, `[both]` = built (as the same region type) by both.

| # | Cluster (canonical) | Regions (sim-ids; ×4 = per-action template) | Config |
|---|---|---|---|
| C1 | **Visual ventral stream** | retina, cortex_v1_simple, cortex_v1_complex, cortex_v2, cortex_it | [both] (opt-in `--enable-visual-cortex`) |
| C2 | **Sensorimotor cortex (M1 channels)** | cortex_N/E/S/W (×4), cortex_FS_N/E/S/W (×4, opt) | [nav] |
| C3 | **Basal ganglia — direct** | str_D1_X (×4), gpi_X (×4) | [nav] |
| C3i | **Basal ganglia — indirect** | str_D2_X (×4), gpe_X (×4), gpe_arky_X (×4, opt), stn | [nav] |
| C3h | **Basal ganglia — hyperdirect** | stn (shared), cortex_X→stn | [nav] (opt `--enable-cluster-a-closed-loop`) |
| C3f | **Striatal interneurons** | str_PV_FSI_X (×4, opt), str_striosome_X (×4) | [nav] |
| C4 | **Thalamus + TRN** | thal_X (×4), thal_FS_X (×4, opt = TRN) | [nav] |
| C5 | **Superior colliculus / orienting (accumulate→commit)** | sel_X (×4, opt), sel_FS_X (×4, opt), commit_X (×4, opt), commit_OPN (opt) | [nav] (`--readout-source spiking_wta`) |
| C6 | **Reward & neuromodulation (midbrain DA)** | snc, striosome_value (opt), vs_place_context (opt) + the DA/ACh/peptide modulators (edges, not nodes) | [nav] |
| C7 | **Hippocampal formation** | ec, dg, dg_pv_basket, ca3, ca1, ec_context (opt) | [both] |
| C7p | **Perception readouts (nav)** | sensor_place_readout, ppc_goal_input, sensory, beacon_sensors, landmark_sensors | [nav] (perception arc) |
| C8 | **Cerebellum (Marr-Albus-Ito)** | mossy_state, granule, purkinje_X (×4), dcn_aip_X (×4), inferior_olive | [nav] (opt `--enable-cluster-f-cerebellum`) |
| C9 | **dlPFC / working memory** | dlpfc_wm [nav]; dlpfc_verb [conv] | both (separate regions) |
| C10 | **Motor output** | motor_N/E/S/W (×4) [both]; motor_FS_X (×4, opt) [both]; motor_pop_θ (×8, opt) [nav]; motor_speech [conv] | both |
| C11 | **Language / conversational core** | language_input, language_output [both]; wernicke / wernicke_pool_i + wernicke_fs_pool_i [conv]; semantic_cortex (+semantic_fs) [conv]; broca [conv]; multimodal_hub [conv] | [conv] |
| C12 | **Concept pools** | {noun,verb,adjective}_pool_NAME (×4 each) + _fs (opt) [conv] | [conv] |

**Co-existence:** C1, C7, C9, C10, C11(language_input/output) appear in BOTH configs. The BG
cascade (C2–C6, C8, C7p) is navigation-only. The semantic/Broca/concept-pool stack (C11–C12)
is conversational-only. A "combined" brain (the project's long-term unification target) would
union them; today they are built by separate runners.

---

## 2. NODE INVENTORY (every region)

Columns: canonical | sim-id | cluster | cell class (Izhikevich-2007 preset) | E/I | role |
faithful vs shortcut | config. **E/I** is derived from `exc_fraction` (0.0 = pure inhibitory,
1.0 = pure excitatory, 0.8 = cortical 80/20, 0.05/0.85/0.95 = mostly-one-type). All neurons
are **Izhikevich-2007 point neurons** unless a run selects HH/AdEx (see honesty layer SH-2).

### Visual ventral stream (C1) — [both], opt-in

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| Retina (ON/OFF) | `retina` | RS pyramidal | exc | 2×32×32=2048 ON/OFF photoreceptor proxy; drive-injected from rendered image | SHORTCUT: rendered by host (the environment); rate proxy, not real phototransduction |
| V1 simple | `cortex_v1_simple` | RS pyramidal | exc | orientation/freq/position Gabor cells (Hubel-Wiesel) | FAITHFUL form (Gabor RF init); weights host-installed |
| V1 complex | `cortex_v1_complex` | RS pyramidal | exc | phase-pooled (orientation invariance) | SHORTCUT: rate-avg approximates max-pooling |
| V2 | `cortex_v2` | RS pyramidal | 0.8 | higher-order feature combinations; plastic recurrent | FAITHFUL |
| IT (inferotemporal) | `cortex_it` | RS pyramidal | 0.8 | object/category "what" stream; plastic recurrent | FAITHFUL (position-invariant ventral code) |

### Sensorimotor cortex (C2) — [nav]

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| Motor cortex channel ×4 | `cortex_{N,E,S,W}` | RS pyramidal | exc | M1-equivalent per-action input pool; drives D1/D2 channel | SHORTCUT: labeled-line per-action split is phenomenological (stands in for learned cortex→striatum weights); uniform drive (`_position_to_cortex_drive` returns equal pA) |
| Cortical FS ×4 | `cortex_FS_{N,E,S,W}` | FS interneuron | inh | per-pool WTA cross-inhibition | FAITHFUL (PV-FS microcircuit), opt-in |

### Basal ganglia (C3 / C3i / C3h / C3f) — [nav]

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| D1 MSN (direct) ×4 | `str_D1_{X}` | MSN-D1 (E_GABA=−60) | inh | "go": inhibits gpi_X → disinhibits thal | FAITHFUL |
| D2 MSN (indirect) ×4 | `str_D2_{X}` | MSN-D2 (E_GABA=−60) | inh | "no-go": → gpe → stn → gpi | FAITHFUL |
| PV-FSI ×4 | `str_PV_FSI_{X}` | FS interneuron | inh | feedforward cross-action WTA broadcast | FAITHFUL, opt-in |
| Striosome/patch ×4 | `str_striosome_{X}` | MSN-D1 (E_GABA=−60) | inh (exc_frac 0.05) | patch compartment → SNc (drives DA) + → gpi | FAITHFUL; limbic input proxied by cortex_X (SHORTCUT: vmPFC/amygdala source absent) |
| GPe prototypic ×4 | `gpe_{X}` | GPe pacemaker | inh | PV+ proto: → stn | FAITHFUL |
| GPe arkypallidal ×4 | `gpe_arky_{X}` | GPe pacemaker | inh | PV− arky: "stop" → striatal FSIs | FAITHFUL, opt-in (needs FSIs) |
| GPi/SNr ×4 | `gpi_{X}` | GPi output | inh | BG output; tonic 40–80 Hz → thal (the gate); → snc collateral | FAITHFUL; SHORTCUT: GPi and SNr collapsed into one pool (rodent/primate naming) |
| STN | `stn` | STN burst | exc | diffuse excitation to all gpi (hyperdirect/indirect convergence) | FAITHFUL |

### Thalamus + TRN (C4) — [nav]

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| Thalamic relay ×4 | `thal_{X}` | thalamic relay (TC) | exc | relay nucleus; released by GPi silence → drives motor + cortex feedback | FAITHFUL (tonic mode only; no T-type burst/tonic switch) |
| TRN ×4 | `thal_FS_{X}` | FS interneuron | inh | reticular-nucleus reciprocal inhibition between relays (WTA) | FAITHFUL form, opt-in; SHORTCUT: modeled as cortical-FS-style, not true TRN bursting |

### Superior colliculus / orienting (C5) — [nav], `--readout-source spiking_wta`

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| Selection/accumulator ×4 | `sel_{X}` | RS pyramidal (NMDA on) | exc | Wang-2002 NMDA-slow evidence accumulator; soft-WTA α<1 | FAITHFUL (decision attractor) |
| Selection FS ×4 | `sel_FS_{X}` | FS interneuron | inh | structured cross-pool WTA (Rutishauser) | FAITHFUL |
| Commit burst ×4 | `commit_{X}` | RS pyramidal | exc | SC saccade-burst analogue; all-or-none commit (Lo-Wang) | FAITHFUL |
| Omnipause (OPN) | `commit_OPN` | FS interneuron | inh | tonic gate holding all commit pools silent until a winner ramps | FAITHFUL form; default tonic drive OFF (rate-coded rebound instability — documented) |

### Reward & neuromodulation (C6) — [nav]

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| SNc dopamine | `snc` | dopamine (E_GABA=−55, no KCC2) | exc | A9 DA neuron pool; FIRING = RPE; broadcasts via `dopamine` modulator | FAITHFUL (spiking-SNc); SHORTCUT: A9+A10/VTA collapsed into one; broadcast is a global scalar (SH-5) |
| Striosome value critic | `striosome_value` | MSN-D1 (E_GABA=−60) | inh | dedicated state-value V(s) critic; → snc via GABA_B subtraction (r−V at membrane) | FAITHFUL, opt-in (`--enable-neural-critic`); per-region homeostasis to reach firing range |
| VS place context | `vs_place_context` | RS pyramidal | exc | dense grid-32 Gaussian place-code afferent feeding ONLY the critic | FAITHFUL form; SHORTCUT: place code rendered from (x,y) by host, drive-injected |

### Hippocampal formation (C7) + perception readouts (C7p) — [both / nav]

| canonical | sim-id | cell class | E/I | role | faithful/shortcut | cfg |
|---|---|---|---|---|---|---|
| Entorhinal cortex | `ec` | RS pyramidal | 0.8 | perceptual entry; perforant path + direct CA1 bypass | FAITHFUL | both |
| Dentate gyrus | `dg` | hippo pyramidal | 0.95 | pattern separation (FFi-driven sparsity) | FAITHFUL | both |
| DG PV basket | `dg_pv_basket` | FS interneuron | inh | strong feedforward inhibition → DG sparsity | FAITHFUL | both |
| CA3 | `ca3` | hippo pyramidal | 0.85 | recurrent autoassociator; pattern completion; SWR replay | FAITHFUL (recurrence via internal_density or SWR-gated explicit self-loop) | both |
| CA1 | `ca1` | hippo pyramidal | 0.85 | readout; EC+CA3 integration; → consolidation targets | FAITHFUL | both |
| Episodic context EC | `ec_context` | RS pyramidal | exc | positional/time-cell drive → DG (word×position binding) | FAITHFUL form; SHORTCUT: positional code host-generated | conv (opt) |
| Place readout | `sensor_place_readout` | hippo pyramidal | exc | sparse Gaussian place cells (agent x,y) | SHORTCUT: sensor-driven readout, not allocentric place cells (glossary); host-rendered | nav |
| PPC goal input | `ppc_goal_input` | hippo pyramidal | exc | goal (gx,gy) context | SHORTCUT: PPC-like, host-fed coordinate | nav |
| Sensory (dx,dy) | `sensory` | RS pyramidal | exc | 7×7 relative-position tuned input | SHORTCUT: host-encoded position | nav (opt) |
| Beacon sensors | `beacon_sensors` | RS pyramidal | exc | 8 directional beacon-intensity sensors | SHORTCUT: host-computed cue | nav (opt) |
| Landmark sensors | `landmark_sensors` | RS pyramidal | exc | 8 (distance,bearing)-to-landmark sensors | SHORTCUT: host-computed | nav (opt) |

### Cerebellum (C8) — [nav], opt-in `--enable-cluster-f-cerebellum`

| canonical | sim-id | cell class | E/I | role | faithful/shortcut |
|---|---|---|---|---|---|
| Mossy/state input | `mossy_state` | RS pyramidal | exc | MF input pool (state) | FAITHFUL form (single MF stream) |
| Granule | `granule` | RS pyramidal | exc | sparse expansion code (Marr codon, ~3-5% active) | FAITHFUL form; SHORTCUT: 250 cells vs ~50M (breaks Albus LTD calibration) |
| Purkinje ×4 | `purkinje_{X}` | FS interneuron | inh | per-action PC; PF input modulates rate; → DCN | SHORTCUT: FS preset stands in for Purkinje (no HH Purkinje at nav dt) |
| DCN / AIP ×4 | `dcn_aip_{X}` | RS pyramidal | exc | deep nucleus; PC pause → disinhibition → motor | FAITHFUL |
| Inferior olive | `inferior_olive` | RS pyramidal | exc | climbing-fiber teaching signal (Δd>0 trigger) | FAITHFUL form; SHORTCUT: no 1:1 PC:CF, CF event host-triggered |

### dlPFC / working memory (C9), Motor output (C10), Language core (C11), Concept pools (C12)

| canonical | sim-id | cell class | E/I | role | faithful/shortcut | cfg |
|---|---|---|---|---|---|---|
| dlPFC working memory | `dlpfc_wm` | hippo pyramidal (NMDA opt) | 0.8 | recurrent persistent activity (Wang-2002 bistability); goal WM | FAITHFUL | nav (opt) |
| dlPFC verb WM | `dlpfc_verb` | RS pyramidal (NMDA opt) | 0.8 | holds verb context ~500ms for 2-word phrases | FAITHFUL | conv (opt) |
| Motor pool ×4 | `motor_{N,E,S,W}` | RS pyramidal (NMDA opt) | exc / 0.8 | motor output; agent acts on which fires | FAITHFUL (pure-exc in nav; E/I cortical canon in conv) | both |
| Motor FS ×4 | `motor_FS_{X}` | FS interneuron | inh | PV-FS WTA between motor pools | FAITHFUL; (nav variant DEPRECATED — biology is spinal Renshaw, not cortical-FS) | both (opt) |
| Distributed motor ×8 | `motor_pop_{E,NE,N,NW,W,SW,S,SE}` | RS pyramidal (NMDA opt) | exc | 45°-spaced cosine-tuned population vector (Georgopoulos) | FAITHFUL form | nav (opt, replaces 4-pool) |
| Motor speech | `motor_speech` | RS pyramidal | 0.85 | 4-slot articulation output | FAITHFUL form (reduced) | conv (opt) |
| Language input (Wernicke) | `language_input` | RS pyramidal | 0.8 | token-embedding input region; plastic recurrent | FAITHFUL form; SHORTCUT: token embedding host-supplied | both |
| Language output (Broca-ish) | `language_output` | RS pyramidal | 0.8 | word-pattern output (A→W readout) | FAITHFUL form | both (opt) |
| Wernicke | `wernicke` | RS pyramidal | 0.8 | lang↔semantic bridge (single-pool variant) | FAITHFUL | conv (opt) |
| Wernicke pool i | `wernicke_pool_{i}` | RS pyramidal | 0.8 | per-concept Wernicke sub-pool | FAITHFUL | conv (opt multi-pool) |
| Wernicke FS pool i | `wernicke_fs_pool_{i}` / `wernicke_fs` | FS interneuron | inh | cross-pool WTM inhibition (sparse codes) | FAITHFUL | conv (opt) |
| Semantic cortex (ATL) | `semantic_cortex` | RS pyramidal | 0.85 | sparse distributed concept store; plastic recurrent attractor | FAITHFUL | conv (opt) |
| Semantic FS | `semantic_fs` | FS interneuron | inh | winner-take-most among concept sub-pops | FAITHFUL | conv (opt) |
| Broca | `broca` | RS pyramidal | 0.8 | syntactic composition; sentence WM (plastic recurrent) | FAITHFUL | conv (opt) |
| Multimodal hub (ATL) | `multimodal_hub` | RS pyramidal | 0.8 | auditory(wernicke)+visual(IT) convergence (hub-and-spoke) | FAITHFUL | conv (opt) |
| Lang output pool i | `lang_output_pool_{i}` | RS pyramidal | 0.8 | per-concept output pool | FAITHFUL | conv (opt) |
| Lang output FS pool i | `lang_output_fs_pool_{i}` | FS interneuron | inh | output-layer WTM | FAITHFUL | conv (opt) |
| Concept pool (noun/verb/adj) | `{noun,verb,adjective}_pool_{NAME}` | RS pyramidal (NMDA opt) | 0.8 | dedicated per-concept pool (Pulvermüller distributed word ensembles) | FAITHFUL | conv (opt) — ×4 names per kind |
| Concept pool FS | `{kind}_pool_{NAME}_fs` | FS interneuron | inh | within-kind cross-inhibition (cross-kind omitted for composition) | FAITHFUL | conv (opt) |

**Node count (distinct region TYPES, counting ×N templates once):** ~58 region types. Fully
expanded with all per-action ×4 / per-kind ×4 / ×8 replicas under maximal flags: ~150+ region
instances. The diagram should draw the **types** (with ×N annotations), not the instances.

---

## 3. EDGE INVENTORY (every distinct pathway)

Columns: from → to | nature | receptor | plastic | gate | function | faithful/shortcut |
replica. **Nature legend:** EXC = glutamatergic excitatory (g_e, AMPA+NMDA-share, E_e≈0);
GABA_A = fast Cl⁻ inhibition (g_i, E_inh≈−75, MSN override −60); GABA_B = slow GIRK K⁺ (g_gabab,
E_K=−90); NMDA-recur = the recurrent self-excitation flagged enable_nmda (still rides g_e but
with NMDA-slow dynamics on that region's mask); MOD-DA / MOD-ACh / MOD-peptide = neuromodulatory
(not a synapse — a concentration scalar affecting plasticity/excitability). **Inhibitory vs
excitatory of a pathway is auto-derived from the SOURCE region's exc_fraction** by the bridge:
a pathway out of a pure-inhibitory region (exc_fraction=0) is GABAergic; out of an excitatory
region it is glutamatergic. The `receptor` field on `RegionPathway` only switches GABA_A↔GABA_B.

### 3A. Perception → cortex (navigation input) — [nav]

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `sensory → cortex_{X}` | EXC | yes | `sensory_to_cortex` | learned position→action | F | ×4 |
| `sensor_place_readout → cortex_{X}` | EXC | yes | `place_goal_to_cortex` | place→action | F | ×4 |
| `ppc_goal_input → cortex_{X}` | EXC | yes | `place_goal_to_cortex` | goal→action | F | ×4 |
| `beacon_sensors → ppc_goal_input` | EXC | yes | `beacon_to_goal` | beacon→goal-cell | F | 1 |
| `landmark_sensors → sensor_place_readout` | EXC | yes | `landmark_to_place` | landmark→place self-org | F | 1 |
| `dlpfc_wm → cortex_{X}` | EXC | yes | `dlpfc_wm_pathways` | WM-driven action bias | F | ×4 |
| `ppc_goal_input → dlpfc_wm` | EXC | yes | `dlpfc_wm_pathways` | goal into WM | F | 1 |
| `cortex_it → cortex_{X}` | EXC | yes | `visual_cortex_action` | visual→action (K v2; zero-init, gated post-warmup) | F | ×4 |

### 3B. Basal ganglia cascade (the heart of navigation) — [nav], ×4 template

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `cortex_{X} → str_D1_{X}` | EXC | yes | `corticostriatal` | corticostriatal direct (LEARNING site; DA-gated) | F | ×4 same-action |
| `cortex_{X} → str_D2_{X}` | EXC | yes | `corticostriatal` | corticostriatal indirect | F | ×4 same-action |
| `cortex_{X} → str_D1/D2_{Y≠X}` | EXC | yes | `corticostriatal_cross` | cross-projections (cheat-5, opt; weak, patch-matrix sparse) | F (opt, mostly NEGATIVE) | ≤24 |
| `cortex_{X} → str_striosome_{X}` | EXC | yes | `corticostriatal` | limbic-proxy → patch | S (limbic source absent) | ×4 |
| `str_D1_{X} → gpi_{X}` | GABA_A | no | — | DIRECT pathway "go" (disinhibition) | F | ×4 |
| `str_D2_{X} → gpe_{X}` | GABA_A | no | — | INDIRECT pathway start | F | ×4 |
| `gpe_{X} → stn` | GABA_A | no | — | indirect: pallido-subthalamic | F | ×4→1 |
| `str_D2_{X} → gpe_arky_{X}` | GABA_A | no | — | drive arkypallidal | F | ×4 |
| `gpe_arky_{X} → str_PV_FSI_{Y}` | GABA_A | no | — | arky "stop" broadcast to striatal FSIs | F (opt) | ×16 |
| `stn → gpi_{X}` | EXC | no | — | diffuse excitation (hyperdirect/indirect convergence; bias against premature select) | F | 1→×4 |
| `gpi_{X} → thal_{X}` | GABA_A | no | — | BG OUTPUT gate (tonic inhibition; D1 silence releases) | F | ×4 |
| `str_striosome_{X} → snc` | GABA_A | no | — | striosome→SNc (drives phasic DA) | F | ×4→1 |
| `str_striosome_{X} → gpi_{X}` | GABA_A | no | — | secondary striosome→SNr | F | ×4 |
| `gpi_{X} → snc` | GABA_A | no | — | SNr→SNc collateral disinhibition (DA burst substrate) | F | ×4→1 |
| `cortex_{X} → stn` | EXC | no | — | HYPERDIRECT (Nambu; fast global stop) | F (opt, Cluster A) | ×4→1 |
| `thal_{X} → cortex_{X}` | EXC | no | — | thalamo-cortical feedback (closes the loop) | F (opt, Cluster A) | ×4 |
| `thal_{X} → motor_{X}` | EXC | no | — | thalamus drives motor output | F | ×4 |

### 3C. Striatal interneuron microcircuits — [nav], opt-in

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `cortex_{X} → str_PV_FSI_{X}` | EXC | no | — | drive same-action FSI | F | ×4 |
| `str_PV_FSI_{X} → str_D1/D2_{Y≠X}` | GABA_A | no | — | cross-action feedforward WTA (dominant striatal inhibition) | F | ×24 |
| `str_D1/D2_{X} → str_D1/D2_{Y≠X}` | GABA_A | no | — | MSN-MSN lateral collaterals (weaker; `enable_bg_lateral_inhibition`) | F (opt) | ×24 |

### 3D. Spiking value critic (Stage B) — [nav], opt-in `--enable-neural-critic`

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `vs_place_context → striosome_value` | EXC | yes | `value_input` | learn V(s) from place code (trained by SNc DA delta) | F | 1 |
| `striosome_value → snc` | **GABA_B** | no | (transmission gate `critic_snc_window`) | value subtraction r−V at SNc membrane (slow GIRK K⁺) | F (the GABA_B fix) | 1 |

### 3E. Superior colliculus / accumulate-commit — [nav], `--readout-source spiking_wta`

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `thal_{X} → sel_{X}` | EXC | no | — | clean relay drives accumulator (READ-ONLY tap, no back-projection) | F | ×4 |
| `sel_{X} → sel_{X}` (internal) | NMDA-recur | no | — | Wang-2002 NMDA-slow self-excitation (evidence integration) | F | ×4 |
| `sel_{X} → sel_FS_{X}` | EXC | no | — | winner recruits its interneuron | F | ×4 |
| `sel_FS_{X} → sel_{Y≠X}` | GABA_A | no | — | cross-pool WTA suppression | F | ×12 |
| `sel_{X} → commit_{X}` | EXC | no | — | ramped accumulator fires the burst | F | ×4 |
| `commit_{X} → commit_{X}` (internal) | EXC-recur | no | — | all-or-none burst regeneration | F | ×4 |
| `commit_OPN → commit_{X}` | GABA_A | no | — | tonic omnipause holds bursts silent until a winner | F | 1→×4 |
| `thal_{X} → thal_FS_{X}` | EXC | no | — | relay collateral drives TRN | F (opt, alt) | ×4 |
| `thal_FS_{X} → thal_{Y≠X}` | GABA_A | no | — | TRN reciprocal inhibition (WTA on relay) | F (opt) | ×12 |

### 3F. Cerebellum (Marr-Albus-Ito) — [nav], opt-in

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `{state sources} → mossy_state` | EXC | no | — | state into mossy fibers (place/goal/sensory or cortex proxy) | F | union |
| `mossy_state → granule` | EXC | no | — | sparse expansion (Marr codon, ~3 MF/granule) | F | 1 |
| `granule → purkinje_{X}` | EXC | yes | `cerebellum_pf_pc` | parallel-fiber→PC (THE LEARNING SITE; reward-mod STDP) | F | ×4 |
| `purkinje_{X} → dcn_aip_{X}` | GABA_A | no | — | PC inhibits DCN (pause → disinhibition) | F | ×4 |
| `dcn_aip_{X} → motor_{X}` | EXC | no | — | cerebellar drive additive to BG (alongside thal) | F | ×4 |
| `inferior_olive → purkinje_{X}` | EXC | no | — | climbing-fiber teaching (complex spike) | F | 1→×4 |

### 3G. Hippocampal trisynaptic loop + consolidation — [both]

| from → to | nature | plastic | gate | function | F/S | cfg |
|---|---|---|---|---|---|---|
| `language_input → ec` (conv) / `sensory|landmark → ec` (nav) | EXC | yes | `lang_to_ec` / `sensory_to_ec` | cortex→hippo entry | F | both |
| `ec → dg` | EXC | yes | `ec_to_dg` | perforant path | F | both |
| `ec_context → dg` | EXC | yes | `ec_context_to_dg` | positional binding (word×position) | F | conv (opt) |
| `ec → dg_pv_basket` | EXC | no | — | FFi recruitment | F | both |
| `dg_pv_basket → dg` | GABA_A | no | — | strong feedforward inhibition → DG sparsity | F | both |
| `ec → ca1` | EXC | yes | `ec_to_ca1` | direct cortical bypass | F | both |
| `dg → ca3` | EXC | yes | `dg_to_ca3` | mossy fibers (sparse, strong) | F | both |
| `ca3 → ca3` (internal or explicit) | EXC | yes | `ca3_swr_burst` | recurrent autoassociator (SWR-gated) | F | both |
| `ca3 → ca1` | EXC | yes | `ca3_to_ca1` | Schaffer collaterals | F | both |
| `ca1 → sensor_place_readout` (nav) | EXC | no | — | hippo readout into perception arc | F | nav (opt) |
| `ca1 → motor_{X}` (conv) | EXC | yes | `ca1_to_motor` | sleep-replay consolidation to cortex | F | conv ×4 |
| `ca1 → language_output` (conv) | EXC | yes | `ca1_to_lang_out` | consolidation of bindings | F | conv |
| `ca1 → semantic_cortex` (conv) | EXC | yes | `ca1_to_semantic` | engrams → durable cortical meaning (KEY bridge) | F | conv |
| `ca1 → lang_output_pool_{i}` (conv) | EXC | yes | `ca1_to_lang_pool_{i}` | per-concept naming consolidation | F | conv |

### 3H. Conversational comprehension/production + concept pools — [conv]

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `language_input → motor_{X}` | EXC | yes | `language_input_to_motor` | word→action (Tier 1 embodied binding) | F | ×4 |
| `motor_{X} → language_output` | EXC | yes | `motor_to_language_output` | action→word (A→W readout; reciprocal) | F | ×4 |
| `language_input → wernicke[_pool_i]` | EXC | yes | `lang_to_wernicke` / `lang_to_wernicke_pool_{i}` | phonological→Wernicke | F | ×N pools |
| `wernicke_pool_i → semantic_cortex` | EXC | yes | `{pool}_to_semantic` | comprehension → meaning | F | ×N |
| `semantic_cortex → wernicke_pool_i` | EXC | yes | `semantic_to_{pool}` | production (weaker) | F | ×N |
| `wernicke_pool_i → wernicke_fs_pool_i` | EXC | yes | `{pool}_to_fs` | drive own FS | F | ×N |
| `wernicke_fs_pool_i → wernicke_pool_{j≠i}` | GABA_A | no | — | cross-pool WTM | F | ×N(N-1) |
| `wernicke_pool_i → language_output` / `→ lang_output_pool_i` | EXC | yes | `{pool}_to_lang_out` / `{pool}_to_lang_pool_{i}` | naming output | F | ×N |
| `wernicke → broca` | EXC | yes | `wernicke_to_broca` | semantic content → syntax | F | 1 |
| `semantic_cortex → broca` | EXC | yes | `semantic_to_broca` | meaning constraints on syntax | F | 1 |
| `broca → broca` (internal) | EXC-recur | yes(int) | — | sentence-level working memory | F | 1 |
| `broca → motor_speech` | EXC | yes | `broca_to_motor_speech` | articulation drive | F | 1 |
| `broca → ec_context` | EXC | yes | `broca_to_ec_context` | Broca drives positional context during production | F | 1 (opt) |
| `semantic_cortex → semantic_fs` / `semantic_fs → semantic_cortex` | EXC / GABA_A | yes/no | `semantic_to_fs` / `fs_to_semantic` | selective-attractor WTM | F | 1/1 |
| `language_input → {kind}_pool_{NAME}` | EXC | yes | `language_input_to_{kind}_pool` | word→concept pool (Tier-1 recipe) | F | ×4/kind |
| `{kind}_pool_{NAME} → language_output` | EXC | yes | `{kind}_pool_to_language_output` | concept→word (A→W) | F | ×4/kind |
| `{kind}_pool_{NAME} → {kind}_pool_{NAME}_fs` | EXC | no | — | drive own FS | F | ×4/kind |
| `{kind}_pool_{NAME}_fs → {kind}_pool_{OTHER}` | GABA_A | no | — | within-kind WTM (cross-kind omitted) | F | ×12/kind |
| `verb_pool_{V} → motor_{X}` | EXC | yes | `verb_to_motor_direct` | v16 direct composition (zero-init until compose) | F (BOUNDARY result) | ×16 |
| `verb_pool_{V} → dlpfc_verb` / `dlpfc_verb → verb_pool_{V}` | EXC | yes | `verb_pool_to_dlpfc` / `dlpfc_to_verb_pool` | v12 PFC integration (bidirectional) | F (NEGATIVE) | ×N |
| `verb_pool_{V} → dlpfc_verb` + `dlpfc_verb → motor_{X}` | EXC | yes | `verb_pool_to_dlpfc_uni` + `dlpfc_verb_to_motor_uni` | v15 unidirectional gating | F (NEGATIVE) | ×N+×4 |
| `{pool} → {other pool}` (all-to-all) | EXC | yes | `cross_pool_concept` | v18 cross-pool association (zero-init) | F (NEGATIVE) | ×N² |
| `language_input → dlpfc_verb` | EXC | yes | `language_input_to_dlpfc_verb` | verb into WM | F | 1 (opt) |

### 3I. Visual ventral stream pathways — [both], opt-in

| from → to | nature | plastic | gate | function | F/S |
|---|---|---|---|---|---|
| `retina → cortex_v1_simple` | EXC | yes | `visual_cortex_v1` | Gabor RF (host-installed init) | F (Gabor) |
| `cortex_v1_simple → cortex_v1_complex` | EXC | no | — | phase pooling (orientation invariance) | S (rate-avg ≈ max-pool) |
| `cortex_v1_complex → cortex_v2` | EXC | yes | `visual_cortex_v2` | higher-order features | F |
| `cortex_v2 → cortex_it` | EXC | yes | `visual_cortex_it` | object/category | F |
| `cortex_it → language_output` (conv) | EXC | yes | `it_to_language_output` | image→word | F |
| `cortex_it → multimodal_hub` (conv) | EXC | yes | `it_to_hub` | visual semantic content | F |
| `wernicke_pool_i → multimodal_hub` | EXC | yes | `wernicke_pool_{i}_to_hub` | auditory semantic content | F |
| `multimodal_hub → lang_output_pool_i` | EXC | yes | `hub_to_lang_pool_{i}` | visual recognition → naming | F |
| `cortex_it → cortex_{X}` (nav) | EXC | yes | `visual_cortex_action` | (already in 3A) visual→action | F |

### 3J. Motor lateral inhibition + distributed motor + cross-coupling — [both/nav]

| from → to | nature | plastic | gate | function | F/S | replica |
|---|---|---|---|---|---|---|
| `motor_{X} → motor_FS_{X}` | EXC | no | — | drive own FS | F | ×4 |
| `motor_FS_{X} → motor_{Y≠X}` | GABA_A | no | — | WTA cross-pool (DEPRECATED in nav) | F | ×12 |
| `thal_{X} → motor_pop_{θ}` | EXC | no | — | cosine-tuned distributed drive (Georgopoulos) | F | ≤×24 |
| `language_input → motor_pop_{θ}` | EXC | yes | `language_input_to_motor` | distributed word→motor | F | ×8 |
| `motor_{X} → motor_{Y}` (adjacent 90°) | EXC | no | — | overlapping somatotopy cross-coupling | F (opt) | ×8 |

### 3K. Internal recurrence (within-region; drawn as self-loops or implied)

Every region with `internal_density>0` has internal connectivity (mixed E/I from
`_build_region_internal`). Notable ones to mark as self-loops: `ca3` (autoassociator),
`dlpfc_wm` / `dlpfc_verb` (WM persistence), `semantic_cortex` / `broca` / `multimodal_hub`
(plastic attractors), `cortex_v2` / `cortex_it`, `language_input` / `language_output`,
`sel_X` / `commit_X` (decision recurrence), all conv pools (cortical canon 0.10).

### 3L. Neuromodulatory edges (NOT synapses — concentration scalars) — [nav]

These are global/scoped modulator broadcasts, drawn as colored dotted edges from the source
to the affected targets (or as a "neuromodulator bus"). Source = production rule; target =
`ModulatorTarget`. See §4 for the production rules.

| modulator | produced by | affects (target_type/scope) | nature | F/S |
|---|---|---|---|---|
| `dopamine` | SNc firing (`from_region_firing_signed` over `snc`) — RPE | plasticity_rate, scope=all (gates ALL STDP) | MOD-DA | F (spiking-SNc); SHORTCUT: global scalar |
| `dopamine` (Cluster C v1) | `from_reward` (host reward signal) | plasticity_rate, all | MOD-DA | SHORTCUT: host reward |
| `dopamine_{N,E,S,W}` (Cluster C v2) | `from_action_specific_reward` | plasticity_rate, scope=action:idx | MOD-DA | per-action, opt; NEGATIVE |
| `acetylcholine_tan` | TAN `pause_on_reward` | plasticity_window_gate, all | MOD-ACh | F form (opt); NULL on nav |
| `dynorphin` | D1 firing (`from_region_firing`) | plasticity_rate −0.4, all | MOD-peptide | F (opt) |
| `substance_p` | D1 firing | excitability_drive +20pA, all | MOD-peptide | F (opt) |
| `enkephalin` | D2 firing | plasticity_rate +0.3, all | MOD-peptide | F (opt) |

**Edge count (distinct pathway TYPES, ×N templates counted once):** ~95 pathway types across
all configs + 7 modulatory broadcasts. Fully expanded (all ×4/×N replicas, max flags): several
hundred edges. The diagram draws TYPES with ×N labels.

---

## 4. MECHANISMS LAYER (legend / annotation — not nodes)

These operate uniformly across regions on the bridge. Render as a legend box + annotations,
NOT as graph nodes.

### Neuron models (`sim/kernels.py`, selected by `cfg.neuron_model_type`)
- **Izhikevich-2007** (`fused_izhikevich2007_dynamics_update`): the DEFAULT for ~all runs.
  `C·dv/dt = k(v−vr)(v−vt) − u + I`; `du/dt = a(b(v−vr) − u)`. Point neuron. Per-region preset
  (MSN-D1/D2, FS, GPe, GPi, STN, DA, TC, hippo pyramidal, RS).
- **Hodgkin-Huxley** (`fused_hodgkin_huxley_dynamics_update` + extended currents
  M/CaT/Ih/NaP): full biophysics, per-gate Q10, dt=0.05ms. Available, used selectively.
- **AdEx** (`fused_adex_dynamics_update`): adaptive exponential I&F. Available.
- Integration: **forward Euler** (gating vars use exact analytic first-order update).

### Synaptic conductances (per step, `_run_one_simulation_step`)
- **AMPA / excitatory** `g_e`: instantaneous rise, single-exp decay τ≈5ms, E_e≈0
  (`fused_conductance_decay_and_current`).
- **GABA_A / fast inhibition** `g_i`: single-exp τ≈10ms, E_inh=−75mV (MSN/striosome/SNc
  override −60/−55). Driving-force compensated by `inhibitory_propagation_strength=0.105`.
- **GABA_B / slow GIRK** `g_gabab` (`fused_gabab_decay_and_current`, opt-in `enable_gabab`):
  τ≈150ms, E_K=−90mV — independent of Cl⁻; strongly hyperpolarizes KCC2-lacking DA cells. Used
  by `striosome_value → snc` (the value subtraction). Default ratio 0 = byte-identical.
- **NMDA** `g_nmda` (`fused_nmda_update_and_current`, opt-in per-region `enable_nmda`):
  dual-exponential + Jahr-Stevens Mg²⁺ block `B(V)=1/(1+[Mg]/3.57·exp(−0.062V))`, E_nmda≈0.
  Rides the same excitatory drive scaled by `nmda_ratio` (AMPA+NMDA share one presynaptic
  event). Gives bistability/slow integration to dlPFC, sel_X, NMDA-tagged pools.

### Plasticity (the learning rules)
- **STDP** (`fused_stdp_weight_update`, Bi-Poo 1998): soft-bound, asymmetric. LTP
  `A_plus·(w_max−w)·exp(−Δt/τ+)`; LTD `−A_minus·(w−w_min)·exp(Δt/τ−)`. On all `plastic=True`
  pathways. (Gotcha: soft-bound clips at `stdp_w_max`.)
- **Three-factor / reward-modulated eligibility**: STDP eligibility trace
  (`fused_eligibility_trace_decay`) × dopamine signal → weight change. The `reward_learning_rate`
  is multiplied by the dopamine `plasticity_rate` multiplier.
- **STP / Tsodyks-Markram** (`fused_stp_decay_recovery`): u (facilitation) + x (depression),
  per-connection-type U/τ_d/τ_f.
- **Homeostasis** (`fused_homeostasis_update`): EMA-of-activity → threshold adaptation
  (τ≈5s). Per-region opt-in (`enable_homeostasis`) used by the value critic.
- **Hebbian** + **synaptic scaling**: available (Hebbian usually OFF for the language runners
  per the decay gotcha).
- **Plasticity gating** (`cp_plasticity_rate_gain`, `set_plasticity_gate`): freezes weight
  UPDATES per named gate (curriculum / critical periods). Does NOT stop current.
- **Transmission gating** (`cp_transmission_gain`, `set_transmission_gate`): scales synaptic
  CURRENT in [0,1] per named gate (thalamocortical dynamical gating). Used by `critic_snc_window`.

### Neuromodulator production rules (`sim/neuromodulators.py`)
`from_reward`, `from_region_firing` (one-sided), `from_region_firing_signed` (two-sided, the
spiking-SNc RPE), `from_action_specific_reward`, `pause_on_reward` (TAN), `from_error_persistence`,
`from_surprise`, `manual`. Targets: `synaptic_gain`, `plasticity_rate`, `excitability_drive`,
`plasticity_gate`, `plasticity_window_gate`.

---

## 5. PER-ACTION / TEMPLATE REPLICATION (collapse these for the designer)

Draw each ONCE with a "×N" badge; do NOT draw N copies.

- **BG cascade ×4 (N/E/S/W):** `cortex_X, cortex_FS_X, str_D1_X, str_D2_X, str_PV_FSI_X,
  str_striosome_X, gpe_X, gpe_arky_X, gpi_X, thal_X, thal_FS_X, motor_X, sel_X, sel_FS_X,
  commit_X` — all replicated per action. The cross-action pathways (FSI→MSN_Y, sel_FS→sel_Y,
  cross-projections) connect template instances; show one representative cross-arc labeled
  "to other 3 actions". `stn`, `snc`, `commit_OPN` are SHARED (single, all-action).
- **Cerebellum ×4:** `purkinje_X, dcn_aip_X` per action; `granule, mossy_state, inferior_olive`
  shared.
- **Concept pools ×4 per kind:** `noun_pool_{APPLE,RIVER,DOG,CAT}`, `verb_pool_{GO,COME,STOP,LOOK}`,
  `adjective_pool_{BIG,SMALL,HOT,COLD}` + their `_fs`. Draw one pool box per KIND with "×4 names".
- **Wernicke / lang-output pools ×N:** `wernicke_pool_i`, `lang_output_pool_i` (N configurable).
- **Distributed motor ×8:** `motor_pop_{E,NE,N,NW,W,SW,S,SE}` (alternative to the 4-pool motor).
- **Per-action DA ×4:** `dopamine_{N,E,S,W}` modulators (Cluster C v2).

---

## 6. HONESTY LAYER (faithful vs shortcut vs experimental/optional)

Three orthogonal axes the legend should encode:

### Axis A — Faithful model vs documented SHORTCUT (cite `2026-06-08-...-shortcuts-audit.md`)
Substrate-level reductions that apply to (nearly) every node/edge — annotate globally:
- **SH-1 GABA_A-only** by default (no GABA_B/GIRK except the opt-in `striosome_value→snc`).
  Every GABA_A T-bar in the chart is the fast Cl⁻ arm only.
- **SH-2 Point neurons** everywhere (no dendrites). Every node is single-compartment.
- **SH-3 Uniform 1-step delay** (no axonal conduction delays). Every edge has the same latency.
- **SH-4 Single-exp AMPA/GABA_A** (NMDA is dual-exp; AMPA+NMDA share one presynaptic event).
- **SH-5 Neuromodulators = global scalars** (no volume gradient, receptor subtypes collapsed
  into per-target sensitivity). Every MOD edge is a broadcast scalar.
- **SH-6 Izhikevich default** (phenomenological, not biophysical) — "biology-grounded" results
  below the spike level are Izhikevich-level, not HH.
- **SH-7 AHP = M-current** (no Ca²⁺-gated SK/BK). **SH-8 STP** lacks Ca⁴ release/async.
- **SH-9 Forward Euler.** **SH-10 No glia.** **SH-11 Compressed homeostasis clocks.**
- **SH-12 No late-LTP/transcription tier.** **SH-13 static Jahr-Stevens Mg block.**
- **SH-14 inhib driving-force via gain scalar** (absolute conductances calibrated, not physical).

Node/edge-LOCAL shortcuts (mark the specific node/edge):
- `retina`, `sensor_place_readout`, `ppc_goal_input`, `sensory`, `beacon_sensors`,
  `landmark_sensors`, `ec_context`, `vs_place_context` — inputs **rendered/encoded by HOST**
  (legitimate: the environment renders sensory input; but the codes are not generated by
  upstream neural processing). Mark with a "host-rendered input" icon.
- `cortex_X` per-action labeled-line split + uniform drive — phenomenological stand-in for
  learned cortex→striatum differentiation.
- `gpi_X` collapses GPi+SNr; `snc` collapses A9+A10/VTA.
- `str_striosome_X` limbic input proxied by `cortex_X` (real vmPFC/amygdala/vHipp absent).
- `granule` at 250 cells (vs ~50M) — breaks Albus LTD scale.
- `purkinje_X` uses an FS preset (no HH Purkinje at nav dt).
- `cortex_v1_complex` phase pooling = rate-average approximation of max-pooling.
- The token embedding into `language_input` and the word readout from `language_output` are
  host-supplied/decoded (the I/O boundary).

The **BRAIN-BASED-ONLY** meta-note (CLAUDE.md owner directive): even where the host computation
is biologically correct (a host RPE, a host reward, an argmax over spikes, a host-rendered place
code), it is a SHORTCUT because the *brain* is not doing it. The spiking-SNc + neural value
critic + spiking-WTA commit are the project's in-progress conversions of exactly these host
pieces into neural mechanisms. Mark the host-computed cognitive pieces distinctly from the
host-rendered ENVIRONMENT/BODY pieces (the latter are legitimate per the boundary).

### Axis B — Implemented-default vs experimental/optional
- **Always built** (no flag): `language_input` + `motor_X` + `language_input→motor_X` (conv);
  `cortex_X`, `str_D1/D2_X`, `str_striosome_X`, `gpe_X/arky_X`, `gpi_X`, `stn`, `thal_X`,
  `snc`, BG cascade pathways (nav). These are the CORE.
- **Opt-in (flag-gated):** everything else — hippocampus, cerebellum, visual cortex, SC
  commit stack, TRN, neural critic, semantic/Broca/concept-pool stack, dlPFC, FSIs, cross-
  projections, neuromodulator arms. Mark with a dashed/ghosted style or a flag badge.
- **DEPRECATED:** nav `--motor-lateral-inhibition` (motor_FS WTA; biology is spinal Renshaw).
- **Result-quality flags (for honesty in prose, not structure):** the v12/v15/v18 composition
  pathways (`verb_pool→dlpfc_verb`, cross_pool_concept) are **NEGATIVE** results; v16
  `verb_to_motor_direct` is a **BOUNDARY** (weak, seed-dependent); Cluster C v2 per-action DA,
  Cluster F cerebellum, TAN ACh are NEUTRAL/NULL. The pathways exist structurally but mark them
  as "present, not a working capability" so the chart isn't read as "this all works."

### Axis C — Config scope
`[nav]` / `[conv]` / `[both]` per §1–§2. The two builders are alternative configs of one
substrate; only a handful of region TYPES are shared.

---

## 7. RECOMMENDED LAYOUT (for the Graphviz design agent)

### 7.1 Overall structure — two strategies, pick per audience
- **Master + detail (RECOMMENDED).** One high-level "master" graph showing the 12 clusters as
  big boxes with the main signal-flow arteries between them; then per-cluster detail subgraphs
  (BG cascade, hippocampal loop, SC commit, conversational stack) expanded. At full detail a
  single flat graph is ~150 nodes / several-hundred edges — unreadable. Cluster + collapse is
  mandatory.
- Use **`cluster_*` subgraphs** for each C1–C12 with a labeled border and the cluster color.
- Split the canvas into two macro-regions: **NAVIGATION** (left/top) and **CONVERSATIONAL**
  (right/bottom), with the **shared mechanism legend** + **shared region types** (hippocampus,
  visual stream, language_input/output) bridging them.

### 7.2 Main signal-flow axes (the arteries — make these the visual spine)
1. **Navigation perception→action:** `retina→V1→V2→IT` and `place/goal/sensory readouts`
   → `cortex_X` → **BG: str_D1/D2 → gpe/gpi (+stn) → thal** → `sel_X→commit_X` (or motor) →
   `motor_X` → (body acts). Lay this LEFT→RIGHT or TOP→BOTTOM as the dominant rank.
2. **Reward loop:** `(perceived state) → striosome/striosome_value → snc → dopamine (broadcast)
   → gates corticostriatal STDP`. Draw `snc` central with the DA broadcast as a distinct colored
   bus reaching the corticostriatal synapses + the critic. The `striosome_value →(GABA_B)→ snc`
   value-subtraction edge is a signature mechanism — highlight it.
3. **Cerebellar side-loop:** `state → mossy → granule → purkinje → dcn → motor` (parallel to BG,
   converging on motor). IO teaching as a distinct colored edge into Purkinje.
4. **Hippocampal loop:** `EC → DG →(−DG basket) CA3 ⟲ → CA1 → {consolidation targets}` — a tight
   sub-cluster, shared by both configs. ec_context feeds DG.
5. **Conversational comprehend→produce:** `language_input → wernicke(pool) → semantic_cortex →
   broca → motor_speech` (comprehension/production); reciprocal `semantic→wernicke→language_output`;
   parallel `language_input → concept pools → language_output`; `IT + wernicke → multimodal_hub`.
6. **Working-memory taps:** dlpfc_wm (nav) and dlpfc_verb (conv) as side recurrent boxes feeding
   cortex_X / motor_X / verb pools.

### 7.3 Edge-style encoding (THE visual grammar — define in the legend)
| Signal nature | Graphviz style |
|---|---|
| **EXC (glutamatergic AMPA+NMDA-share)** | solid line, **standard arrowhead** (`arrowhead=normal`) |
| **NMDA-recurrent self-excitation** | solid **double** line or `arrowhead=normalnormal` / heavier penwidth; or a "⟲" self-loop with an NMDA tag |
| **GABA_A (fast Cl⁻)** | solid line, **T-bar** (`arrowhead=tee`) |
| **GABA_B (slow GIRK K⁺)** | **dashed** line, **T-bar** (`arrowhead=tee, style=dashed`) — visually distinct slow inhibition |
| **MOD-dopamine** | **dotted**, colored (e.g. gold/orange), open diamond head; or a labeled "DA bus" |
| **MOD-ACh** | dotted, teal |
| **MOD-peptide (dyn/SP/enk)** | dotted, purple, thin |
| **plastic pathway** | normal weight + a small "STDP" / open-circle decoration (vs `plastic=False` = thin/gray) |
| **gated (transmission/plasticity gate)** | add a small gate glyph ⊟ on the edge with the gate name |
| **host-rendered input edge** | gray/ghosted with a "host" tag entering from outside the brain boundary |

Reserve **penwidth** for weight magnitude tiers (e.g. the strong `gpi→thal` w=8, `thal→motor`
w=20, `cortex→MSN` w≈125 vs weak w≈1–2) if you want a second channel; otherwise keep uniform.

### 7.4 Color-by-subsystem (node fill)
Assign one hue per cluster C1–C12 (e.g. visual=blue, sensorimotor cortex=slate, BG=amber/brown
family with direct=green-tint, indirect=red-tint, hyperdirect=orange, thalamus=teal, SC=violet,
reward/DA=gold, hippocampus=green, cerebellum=cyan, dlPFC=indigo, motor=red, language=sky,
concept pools=sky-variant). Use a **darker border for inhibitory** regions (exc_fraction≤0.2)
and lighter for excitatory, so cell-type reads at a glance independent of cluster hue.

### 7.5 Readability tactics at this scale
- **Collapse ×N templates** (§5) — one box, "×4 (N/E/S/W)" badge, one representative cross-arc.
- **Ghost the opt-in regions/edges** (Axis B) so the always-built CORE reads as the backbone and
  the experimental machinery recedes.
- **Two-page option:** page 1 = navigation brain; page 2 = conversational brain; shared legend +
  hippocampus/visual cross-referenced. (Cleaner than one mega-graph.)
- **Rank hints:** force the perception→cortex→BG→thal→motor spine into ordered ranks
  (`rank=same` for the four-action template members at each stage) so the cascade reads
  top-to-bottom; let the modulatory/feedback edges (thal→cortex, gpi→snc, DA bus) be the
  back-edges.
- **Legend boxes:** (1) edge-nature grammar, (2) cluster colors, (3) the 14 substrate shortcuts
  (SH-1..SH-14) as a footnote block, (4) the "host-rendered = environment/body, legitimate; host-
  computed cognition = shortcut" boundary note.

---

## 8. SUMMARY (counts + highlights)

- **12 subsystem clusters** (visual, sensorimotor cortex, BG direct/indirect/hyperdirect,
  striatal interneurons, thalamus+TRN, superior colliculus, reward/neuromodulation, hippocampal
  formation + perception readouts, cerebellum, dlPFC, motor output, language core, concept pools).
- **~58 distinct region TYPES** (~150+ instances fully expanded with ×4/×N/×8 replicas under
  maximal flags). All Izhikevich-2007 point neurons by default; per-region cell-class presets
  (MSN-D1/D2, FS, GPe pacemaker, GPi output, STN burst, DA, TC relay, hippo pyramidal, RS).
- **~95 distinct pathway TYPES + 7 neuromodulatory broadcasts** (several hundred edges fully
  expanded). Nature breakdown: most are glutamatergic EXC; the BG output (str→gpi/gpe→thal,
  FSI→MSN, TRN, motor-WTA, SC-WTA) are GABA_A; ONE signature GABA_B edge (`striosome_value→snc`,
  the value subtraction); NMDA-recurrence on dlPFC/sel_X/NMDA-tagged pools; 7 modulatory scalars
  (dopamine + per-action DA + ACh-TAN + dynorphin/substance_P/enkephalin).
- **Two config-scoped supergroups** (navigation vs conversational) on ONE shared bridge engine;
  a handful of shared region types (hippocampal loop, visual ventral stream, language_input/output,
  motor pools).
- **Honesty highlights:** 14 substrate-wide shortcuts (GABA_A-only, point-neurons, 1-step delay,
  global-scalar NM, Izhikevich-default, etc. — all from the 2026-06-08 audit); node-local
  shortcuts on every host-rendered sensory input + the per-action cortex labeling + GPi/SNr and
  A9/A10 collapses + the reduced cerebellum; and a result-quality caveat that several composition
  pathways (v12/v15/v18) are structurally present but are NEGATIVE/BOUNDARY results, not working
  capabilities. The CORE (always-built) BG cascade + Tier-1 language→motor binding are the
  faithful, validated backbone; almost everything else is opt-in.
- **Recommended layout:** master cluster-graph + per-cluster detail (or two pages: nav | conv),
  perception→cortex→BG→thalamus→motor as the dominant ranked spine, reward→SNc→DA-bus as the
  highlighted modulatory loop, ×N templates collapsed, opt-in machinery ghosted, edge-nature
  grammar (solid-arrow EXC / T-bar GABA_A / dashed-T-bar GABA_B / dotted-colored MOD / NMDA 2nd
  style) + cluster color + cell-type border-weight as the three visual channels.

---

### Source files read (for provenance)
- `sim/regions.py` (wiring grammar: BrainRegion + RegionPathway, receptor/gates/topography)
- `research/runners/g11_bg_runner.py:131-2233` (`build_bg_brain_regions`, full) + `:3543-3617`
  (neuromodulator config) + `:68` (ACTION_NAMES)
- `research/runners/text_minimal_isolation.py:59-1661` (`build_minimal_brain_regions`,
  `build_biological_brain_regions`, `_add_concept_kind`, full)
- `sim/visual_cortex.py` (retina/V1 Gabor, rendering)
- `sim/neuromodulators.py` (targets, production rules, default DA/ACh/peptide configs)
- `sim/kernels.py` (neuron models, conductances incl. GABA_B + NMDA, plasticity)
- `sim/config.py` (E_inh=−75, E_K=−90 GABA_B, NMDA ratio, reward params)
- `sim/enums.py` (Izhikevich-2007 cell-class presets)
- `references/glossary.md` (canonical names ↔ sim-ids)
- `research/findings/2026-06-08-sim-biological-accuracy-shortcuts-audit.md` (the 14 shortcuts)
- `sim-catalog/references/feature-catalog.md` (biological grounding + per-feature sim-status)
