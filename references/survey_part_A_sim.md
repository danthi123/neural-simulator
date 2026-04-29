# Terminology Survey — Part A: `sim/`

**Audit against:** `references/glossary.md` (228 canonical entries).
**Tier 1+2 findings should be fixed.** Tier 3 flagged for later renaming policy.

## Summary

- Files scanned: 9 (`bridge.py`, `config.py`, `connectivity.py`, `data_bus.py`, `enums.py`, `kernels.py`, `neuromodulators.py`, `profiles.py`, `regions.py`)
- Tier 1 findings: 18 (pure prose, safe)
- Tier 2 findings: 7 (symbol-in-prose, safe)
- Tier 3 findings: 9 (identifiers, deferred)
- Total fixes recommended (T1+T2): 25

`data_bus.py` had no biological terminology and no findings. `connectivity.py`
is dominated by spatial / Watts-Strogatz / sampling code with only generic
biophysical jargon and no findings. `regions.py` is mostly framework code with
two minor flags noted under Tier 1.

---

## Tier 1 findings — pure prose

### `sim/kernels.py:266-267`
- **Current:**
  ```
  Implements classical asymmetric STDP window:
  - delta_t > 0 (post-before-pre): LTP (potentiation)
  - delta_t < 0 (pre-before-post): LTD (depression)
  ```
- **Issue:** The parentheticals are reversed. With `delta_t = t_post − t_pre`,
  `delta_t > 0` means post fires AFTER pre (i.e. **pre-before-post**, which is
  the LTP direction in canonical Hebbian STDP per glossary §STDP "Bi-Poo
  STDP"). Line 282 elsewhere in the same kernel docstring states correctly
  "post fired after pre -> strengthen synapse", confirming the convention; the
  parentheticals on lines 266-267 are wrong.
- **Canonical replacement:**
  ```
  - delta_t > 0 (pre-before-post): LTP (potentiation)
  - delta_t < 0 (post-before-pre): LTD (depression)
  ```

### `sim/profiles.py:95`
- **Current:** `"description": "Striatal network with ~95% inhibitory MSNs and a small FS interneuron population."`
- **Issue:** Strictly fine — MSNs are GABAergic (per glossary §MSN). However,
  the trait fraction of 0.05 is "FS interneurons" and the glossary §PV-FSI
  notes striatal FSI is ~0.7% of striatal neurons (Rymar 2004). 5% is a
  modeling shortcut, not a biological figure — flag for clarity if rewriting.
- **Canonical replacement:** Acceptable as-is, but consider noting `~0.7%
  biologically; 5% used as engineering allowance` for truthfulness.
- **Notes:** Borderline — only flag if rewriting nearby prose.

### `sim/profiles.py:100`
- **Current:** `# MSNs modeled as RS-like inhibitory`
- **Issue:** MSNs (medium spiny neurons) are GABAergic projection neurons,
  not "RS-like" — RS (regular spiking) is a cortical electrophysiology
  taxonomy that the glossary lists for cortical pyramidals, not striatal
  cells. The trait_definition reuses `IZH2007_RS_CORTICAL_PYRAMIDAL` here,
  which is itself a Tier 3 issue (see below); but the comment label
  "RS-like inhibitory" is biologically backwards because RS pyramidals are
  excitatory. Also, the glossary canonical for MSN's electrophysiology is
  the dedicated `IZH2007_STRIATAL_MSN` preset (not RS pyramidal).
- **Canonical replacement:** `# MSNs (GABAergic projection neurons) approximated using RS pyramidal preset; consider IZH2007_STRIATAL_MSN for fidelity`

### `sim/enums.py:46`
- **Current:** `HH_STRIATAL_MSN_D1 = "HH_STRIATAL_MSN_D1"   # Direct pathway MSN (DA D1+ sensitive)`
- **Issue:** "DA D1+ sensitive" is unusual. Glossary §"D1 MSN (direct
  pathway / striatonigral)" describes D1 MSNs as having D1-class receptors
  (D1, D5; Gs-coupled, ↑cAMP) which are DA-activated. The "+" notation here
  is shorthand for "expresses D1 receptor".
- **Canonical replacement:** `# Direct-pathway MSN (D1 receptor expressing; LTP-biased under +DA)`

### `sim/enums.py:47`
- **Current:** `HH_STRIATAL_MSN_D2 = "HH_STRIATAL_MSN_D2"   # Indirect pathway MSN (DA D2- sensitive)`
- **Issue:** "D2- sensitive" is incorrect shorthand. D2 MSNs express D2
  receptors (Gi-coupled, ↓cAMP); they are NOT "D2-negative". Glossary §D2
  MSN: "indirect pathway / striatopallidal".
- **Canonical replacement:** `# Indirect-pathway MSN (D2 receptor expressing; LTD-biased under +DA)`

### `sim/enums.py:48`
- **Current:** `HH_STRIATAL_TAN = "HH_STRIATAL_TAN"          # Tonically Active Cholinergic Interneuron`
- **Issue:** Glossary §"TAN / ChI" notes that ChI (anatomical) and TAN
  (electrophysiology) refer to the same cell. Comment is accurate but
  could note the ChI alias for searchability.
- **Canonical replacement:** `# Tonically Active Neuron / Cholinergic Interneuron (TAN ≡ ChI)`
- **Notes:** Minor — only if revising the line.

### `sim/enums.py:484`
- **Current:**
  ```
  # Slightly more KIR inward rectifier than D1 (modeled by stronger M-current)
  ```
- **Issue:** Redundant — KIR ("inwardly rectifying K⁺ channel") already
  contains "inward rectifier". Glossary §MSN B.02 references "KIR2".
- **Canonical replacement:** `# Slightly more KIR2 (inward-rectifier K⁺) than D1 (approximated by stronger M-current)`

### `sim/enums.py:619-620`
- **Current:**
  ```
  # Medium spiny neuron — D1/D2 striatal projection neurons.
  ```
- **Issue:** Fine — fully canonical per glossary.
- **Canonical replacement:** No fix needed; flagged here only as positive
  example of correct usage.
- **Notes:** Skip.

### `sim/enums.py:680-681`
- **Current:**
  ```
  # Tonically Active Neuron (cholinergic interneuron). Spontaneously
  # firing 2-10 Hz with strong AHP between spikes.
  ```
- **Issue:** Per glossary, TAN ≡ ChI, both are correct. AHP = "afterhyperpolarization"
  is biophysics not in the glossary's audit list. Acceptable.
- **Notes:** Skip.

### `sim/profiles.py:227`
- **Current:** `"description": "Mitral/tufted cells with strong granule cell inhibition. High E/I ratio drives gamma/theta oscillations."`
- **Issue:** Mostly correct — glossary §"Mitral cell, tufted cell"
  confirms canonical names. Note "granule cell" in olfactory-bulb context
  refers to OB granule cells (different from cerebellar granule cells); the
  description is unambiguous in context.
- **Canonical replacement:** No fix needed.
- **Notes:** Skip.

### `sim/profiles.py:246`
- **Current:** `"description": "Midbrain dopamine circuit: autonomous pacemaker DA neurons (65%) with GABAergic interneurons (35%)."`
- **Issue:** SNc has ~70% DA / 30% non-DA in vivo per Schultz literature;
  65/35 is a modeling allowance. Glossary §SNc notes A9/A10 collapse into
  the project's `dopamine` region — accept this as project shorthand.
- **Notes:** Skip.

### `sim/profiles.py:265`
- **Current:** `"description": "Inhibition-dominated network for studying gamma (30-80 Hz) oscillations driven by PV+ FS interneurons."`
- **Issue:** Glossary §"Gamma oscillation" canonical band is "40-100 Hz",
  with "ING" / "PING" sub-types; this profile says 30-80 Hz which is the
  legacy 1990s band. Defensible if intent is to capture low-gamma, but a
  pedant would note 40-100 is canonical.
- **Canonical replacement:** `"... studying gamma (40-100 Hz, low end ~30 Hz acceptable in some models) oscillations driven by PV+ FS interneurons (PING regime)."`
- **Notes:** Mild — flag only if rewriting.

### `sim/profiles.py:284`
- **Current:** `"description": "Olivary neurons with CaT/Ih-driven subthreshold oscillations. Note: gap junctions not modeled."`
- **Issue:** Fine. Glossary §"Inferior olive (IO)" canonical, gap junction
  caveat is honest.
- **Notes:** Skip.

### `sim/bridge.py:233`
- **Current:** `# BrainRegion.syn_reversal_potential_i_override (e.g., striatal MSNs`
- **Issue:** Fine — striatal MSNs is canonical; matches glossary §MSN and
  §"GABA-A receptor" notes (project models region-specific E_Cl).
- **Notes:** Skip.

### `sim/bridge.py:945-947`
- **Current:**
  ```
  # (e.g., striatal MSNs use −60 mV per PBR-160 ch 6; SNc DA uses
  # ~−55 mV per ch 11). The fused conductance kernel broadcasts this
  # array element-wise against per-neuron membrane potential.
  ```
- **Issue:** Canonical. Glossary §"GABA-A receptor" explicitly references
  this region asymmetry (B.14, B.15). Comment is accurate.
- **Notes:** Skip.

### `sim/bridge.py:1377`
- **Current:** `# neuron type (striatum_D1 uses MSN_D1, GPe uses GPE_PACEMAKER, etc.).`
- **Issue:** Comment uses `striatum_D1` as a region-name placeholder, but
  the actual project convention per glossary §"Striatum" is `str_D1_X`
  (per-action pools). Slight prose-symbol mismatch; not wrong but not
  matching the deployed naming.
- **Canonical replacement:** `# neuron type (str_D1_X uses MSN_D1, gpe_X uses GPE_PACEMAKER, etc.).`

### `sim/bridge.py:1206-1207`
- **Current:**
  ```
  # e.g. striatum_D1 region use IZH2007_STRIATAL_MSN_D1 while
  # motor region uses IZH2007_RS_CORTICAL_PYRAMIDAL.
  ```
- **Issue:** Same as above — `striatum_D1` should be `str_D1_X`. The "motor
  region uses IZH2007_RS_CORTICAL_PYRAMIDAL" is biologically inaccurate per
  glossary §"α-motoneuron, γ-motoneuron" — motor neurons are not RS
  cortical pyramidals. The project uses cortical pyramidals as a stand-in
  for `motor_X` per a documented modeling shortcut, but the comment should
  acknowledge that.
- **Canonical replacement:**
  ```
  # e.g. str_D1_X region uses IZH2007_STRIATAL_MSN_D1 while
  # motor_X uses IZH2007_RS_CORTICAL_PYRAMIDAL as an abstract motor pool
  # (HH_SPINAL_MOTOR is the canonical α-motoneuron preset).
  ```

### `sim/bridge.py:1953-1957`
- **Current:**
  ```
  # Biological grounding: developmental staging (sensory cortex matures
  # before association cortex), critical periods (visual cortex ocular
  # dominance plasticity closes via PV interneuron maturation), and
  # neuromodulator-gated plasticity windows.
  ```
- **Issue:** Canonical. "PV interneuron" matches glossary §"Cortical FS
  interneuron (PV+ basket)"; "ocular dominance critical period" matches
  glossary §"Critical period". Good prose.
- **Notes:** Skip — included as positive reference.

### `sim/bridge.py:4424-4426`
- **Current:**
  ```
  # pause_on_reward -> ACh pause -> plasticity_window_gate opens) AT
  # THE SAME STEP. This is required for fast-dynamics gates
  # (TAN/ACh) where the pause and the reward are the same event.
  ```
- **Issue:** "TAN/ACh" canonical per glossary §"TAN / ChI" and §"Acetylcholine".
- **Notes:** Skip.

### `sim/neuromodulators.py:50-52`
- **Current:**
  ```
  Biological grounding: critical-period closure via PV
  interneuron maturation, DA-gated corticostriatal plasticity,
  ACh-gated cortical attention plasticity.
  ```
- **Issue:** Canonical per glossary §"Critical period", §"Cortical FS
  interneuron", §"Dopamine", §"Acetylcholine". Good prose.
- **Notes:** Skip.

### `sim/neuromodulators.py:736-737`
- **Current:**
  ```
  Models tonically active cholinergic interneurons that pause briefly on
  salient events (reward, novelty), opening a transient corticostriatal
  plasticity window.
  ```
- **Issue:** Canonical. "Tonically active cholinergic interneurons" matches
  glossary §"TAN / ChI"; "corticostriatal plasticity window" matches
  glossary §"D1 MSN" notes. Good prose.
- **Notes:** Skip.

### `sim/neuromodulators.py:769-776`
- **Current:** D1/D2 neuropeptide commentary uses canonical anatomy
  ("D1 MSNs co-release dynorphin + substance P with GABA"; "DOR receptor
  effects").
- **Issue:** Canonical per glossary §"Dynorphin", §"Enkephalin",
  §"Substance P", §"Opioid receptors". Good prose.
- **Notes:** Skip.

### `sim/regions.py:46-48`
- **Current:**
  ```
  exc_fraction:
      Fraction excitatory (rest inhibitory). 0.8 matches cortical
      layer 2/3 (Markram et al. 2015).
  ```
- **Issue:** Canonical — cortical L2/3 is approximately 80/20 E/I per
  Markram, matches glossary §"Cortical pyramidal (RS)" and §"Cortical FS
  interneuron".
- **Notes:** Skip.

### `sim/regions.py:90-97`
- **Current:**
  ```
  # Per-region GABA_A reversal potential override in mV. None = use global
  # cfg.syn_reversal_potential_i. Used to model regions with different
  # chloride homeostasis (e.g., striatal MSNs ~−60 mV per PBR-160 ch 6;
  # SNc DA ~−55 mV per ch 11). MSNs lack the deep negative ECl seen in
  # cortical pyramidals: gramicidin perforated patch measurements give
  # ~-60 mV, producing shunting (depolarizing-near-rest, hyperpolarizing-
  # near-threshold) inhibition. SNc DA neurons lack KCC2 entirely.
  ```
- **Issue:** Canonical — explicit references to glossary §"GABA-A receptor"
  notes (B.14, B.15) and §"SNc" cell-type details. Excellent prose.
- **Notes:** Skip.

### `sim/profiles.py:189`
- **Current:** `"description": "Layer 5 corticofugal output circuit: thick-tufted pyramidal tract (PT) neurons with burst-firing properties."`
- **Issue:** Canonical — "L5 PT neurons" / "thick-tufted" / "corticofugal"
  match glossary §"Cortical pyramidal (RS)" notes ("L5 pyramidal /
  Betz cell").
- **Notes:** Skip.

### `sim/enums.py:594-597`
- **Current:** Comment block on FS interneuron `d_increment`:
  ```
  # d_increment must be POSITIVE for FS interneurons (Izhikevich 2007, Table 2).
  # ... Value of 25 pA gives the characteristic non-adapting, high-frequency
  # firing pattern of PV+ basket cells.
  ```
- **Issue:** Canonical — "PV+ basket cells" matches glossary §"Cortical FS
  interneuron (PV+ basket)".
- **Notes:** Skip.

### `sim/enums.py:407` (CORTICAL_FS_INTERNEURON header)
- **Current:** `# Cortical fast-spiking PV+ interneuron (Erisir et al. 1999, Wang & Buzsaki 1996)`
- **Issue:** Canonical per glossary §"Cortical FS interneuron". Good prose.
- **Notes:** Skip.

### `sim/enums.py:444-453` (GPI_OUTPUT block)
- **Current:**
  ```
  # BG output gate: GPi (and SNr, which is functionally similar — primary BG
  # output to thalamus, suppressing motor activity at rest, releasing it on
  # action selection via direct-pathway disinhibition). Higher tonic firing
  # than GPe, modest g_NaP.
  ```
- **Issue:** Canonical. Matches glossary §"Globus pallidus internus / SNr"
  (`gpi_X` covers both per project shorthand). Good prose.
- **Notes:** Skip.

### `sim/profiles.py:113`
- **Current:** `"description": "Thalamic relay (TC) and reticular (TRN) network with excitatory-inhibitory recurrence and bursting."`
- **Issue:** Canonical — TC and TRN match glossary §"Thalamus" and §"TRN".
- **Notes:** Skip.

### `sim/profiles.py:193-194`
- **Current:** Trait definitions assign IZH2007_RS_CORTICAL_PYRAMIDAL to
  L5 PT excitatory trait (line 193) and IZH2007_FS_CORTICAL_INTERNEURON to
  inhibitory trait (line 194).
- **Issue:** RS preset stands in for L5 PT — "thick-tufted pyramidal tract
  neurons" ≠ "RS pyramidal" canonically (PT cells are typically more
  bursty/IB per glossary §"Cortical pyramidal" and §"IB"). Acceptable as
  modeling shortcut but worth noting in the description.
- **Canonical replacement:** Add a comment noting the simplification:
  `# Note: PT cells are typically IB-like (intrinsic bursting); using RS preset as a fallback. Consider IB_EXCITATORY_LEGACY or ADEX_IB_BURSTING for fidelity.`
- **Notes:** Borderline — skip unless revising.

### `sim/enums.py:47-48` (D2 sensitivity description, see also above):
Already covered — duplicate. Skip.

---

## Tier 2 findings — symbol-in-prose

### `sim/bridge.py:1377` (also flagged Tier 1)
- **Current:** `# (striatum_D1 uses MSN_D1, GPe uses GPE_PACEMAKER, etc.).`
- **Issue:** Symbol `striatum_D1` is non-canonical; project uses `str_D1_X`.
  See Tier 1 above for the recommended replacement. Tier 2 flag for the
  prose-symbol alignment.

### `sim/bridge.py:1206`
- **Current:** `# e.g. striatum_D1 region use IZH2007_STRIATAL_MSN_D1 while`
- **Issue:** Same — `striatum_D1` non-canonical. Replace with `str_D1_X`.

### `sim/neuromodulators.py:782-789` (`_default_dynorphin_config`)
- **Current:** Comment refers to `str_D1_N`, `str_D1_E`, etc., as
  source_regions. These are canonical project identifiers per glossary
  §"Striatum".
- **Issue:** None — correctly using canonical identifiers.
- **Notes:** Skip; positive example.

### `sim/neuromodulators.py:980-983` (`_default_enkephalin_config`)
- **Current:** Source regions list `str_D2_N`, `str_D2_E`, etc.
- **Issue:** Canonical. Skip.

### `sim/regions.py:101-110` (action_index docstring)
- **Current:**
  ```
  # When a region is action-specific (cortex_X, str_D1_X, str_D2_X,
  # gpi_X, thal_X, motor_X, etc), this is the action index in [0, N-1]
  ```
- **Issue:** All identifiers canonical per glossary. Good prose-symbol
  alignment.
- **Notes:** Skip; positive example.

### `sim/bridge.py:4505`
- **Current:** `per_action_names = ["dopamine_N", "dopamine_E", "dopamine_S", "dopamine_W"]`
- **Issue:** Per-action DA modulator names — these are project identifiers
  not in the canonical glossary section but are documented in
  `neuromodulators.py:933` `_default_per_action_dopamine_config`. Acceptable
  as project shorthand.
- **Notes:** Skip.

### `sim/profiles.py:170`
- **Current:** `"description": "Subthalamic nucleus (STN) and globus pallidus externus (GPe) excitatory-inhibitory loop."`
- **Issue:** Canonical per glossary §STN and §GPe. STN is "glutamatergic"
  per glossary; calling it "excitatory" is correct functional shorthand.
- **Notes:** Skip; positive example.

### `sim/bridge.py:1248-1251` (region_manager use)
- **Current:** Region names referenced via `region.name` symbolically;
  variable name `inh_indices_concat`.
- **Issue:** Inhibitory canonical. Skip.

### `sim/regions.py:411-413` (internal connectivity docstring)
- **Current:**
  ```
  """Sparse Erdős-Rényi internal connectivity for a region.

  Each ordered (pre, post) pair (pre != post) within the region is
  included with probability `region.internal_density`.
  ```
- **Issue:** Erdős-Rényi is a graph theory term, not in glossary;
  acceptable.
- **Notes:** Skip.

### `sim/connectivity.py:743-748` (`generate_watts_strogatz_3d`)
- **Current:** `Creates a small-world network with high clustering and short
  path lengths`
- **Issue:** Watts-Strogatz / small-world are graph theory canonical, not
  flagged in glossary. Skip.

---

## Tier 3 findings — identifiers (FLAGGED for policy)

### `sim/profiles.py:100`
- **Identifier:** `IZH2007_RS_CORTICAL_PYRAMIDAL` used as MSN preset (in `BASAL_GANGLIA_STRIATUM`)
- **Issue:** Glossary §MSN canonical preset is `IZH2007_STRIATAL_MSN`
  (which exists in the codebase). The profile reuses the cortical pyramidal
  enum. This is functional cross-typing.
- **Risk:** Switching to `IZH2007_STRIATAL_MSN` would change the dynamics of
  the legacy `BASAL_GANGLIA_STRIATUM` profile and break any saved JSONs that
  reference it.
- **Suggested action:** Keep current behavior; add a docstring comment in
  `profiles.py` to document the cross-typing. Migration would be a separate
  config-versioning task.

### `sim/enums.py:14-26` (Izhikevich 2007 enum names)
- **Identifier:** `IZH2007_*` prefix on every preset
- **Issue:** Glossary §"Izhikevich 2007 Presets" treats these as canonical
  project identifiers. No issue.
- **Risk:** N/A.
- **Suggested action:** Keep.

### `sim/enums.py:47` `HH_STRIATAL_MSN_D2`
- **Identifier:** `HH_STRIATAL_MSN_D2`
- **Issue:** Glossary §"D2 MSN" canonical is "D2 MSN" / "indirect-pathway
  MSN" / "striatopallidal MSN"; the project identifier omits "indirect" but
  is documented as canonical. No issue.
- **Suggested action:** Keep.

### `sim/enums.py:23` `IZH2007_GPI_OUTPUT`
- **Identifier:** `IZH2007_GPI_OUTPUT`
- **Issue:** Glossary §"Globus pallidus internus / SNr" notes
  `IZH2007_GPI_OUTPUT` and `HH_GPI_OUTPUT` cover both GPi and SNr (project
  shorthand acknowledged in `[NEEDS-REVIEW]` section). No issue.
- **Suggested action:** Keep.

### `sim/enums.py:26` `IZH2007_DOPAMINE` and `sim/enums.py:41` `HH_DOPAMINE_SNC`
- **Identifier:** `IZH2007_DOPAMINE`, `HH_DOPAMINE_SNC`
- **Issue:** Glossary §SNc / §VTA notes the project's `dopamine` region
  collapses A9 (SNc) and A10 (VTA). The HH preset suffix `_SNC` is more
  specific than the Izh `_DOPAMINE` suffix. Inconsistent naming but
  documented as project shorthand.
- **Suggested action:** Keep; if renaming, use `*_SNC_VTA` to make collapse
  explicit.

### `sim/enums.py:25` `IZH2007_HIPPO_PYRAMIDAL`
- **Identifier:** `IZH2007_HIPPO_PYRAMIDAL`
- **Issue:** Glossary §"Hippocampal pyramidal neuron" canonical is "CA1
  pyramidal" / "CA3 pyramidal"; project's Izh enum collapses both, while
  HH has separate `HH_CA1_PYRAMIDAL_BURST` and `HH_CA3_PYRAMIDAL_BURST`.
- **Suggested action:** Keep; document in enum comment that this is a single
  preset for CA1 + CA3 hippocampal pyramidals.

### `sim/config.py:187-203` `current_reward_signal` / `last_selected_action`
- **Identifier:** `current_reward_signal`, `last_selected_action`
- **Issue:** Glossary §"Two-component DA response" / `[NEEDS-REVIEW]` flag
  both — `current_reward_signal` conflates Component-1 / Component-2 / A9 /
  A10 DA responses. The code already documents this on lines 188-197.
  Glossary explicitly says "Audit should not flag every use of
  `current_reward_signal` but should note when biological distinctions
  matter."
- **Suggested action:** Keep; the existing comment block (lines 188-197) is
  excellent self-documentation. No change.

### `sim/regions.py:78-89` `izh_neuron_type`, `hh_neuron_type`, `adex_neuron_type`
- **Identifier:** Per-region neuron-type override fields
- **Issue:** Field naming uses model abbreviation (`izh`, `hh`, `adex`).
  Acceptable; these are config field names, not biological terminology.
- **Suggested action:** Keep.

### `sim/profiles.py:9` `NEURAL_STRUCTURE_PROFILES`
- **Identifier:** Profile keys (e.g., `BASAL_GANGLIA_STRIATUM`,
  `HIPPOCAMPUS_CA1_RS_FS`).
- **Issue:** Profile keys reuse cortical RS+FS structure even for
  non-cortical regions (`HIPPOCAMPUS_CA1_RS_FS`, `BASAL_GANGLIA_STRIATUM`).
  This is a functional shortcut documented in glossary
  `[NEEDS-REVIEW]` for FSI disambiguation.
- **Risk:** Renaming would invalidate saved profiles in
  `simulation_profiles/`.
- **Suggested action:** Keep + comment explaining cross-typing in profiles.

---

## Items NOT flagged (intentional shorthand)

- **`gpi_X` for GPi/SNr (project)**: Glossary `[NEEDS-REVIEW]` says auditor
  should not flag this conflation. Both `IZH2007_GPI_OUTPUT` and
  `HH_GPI_OUTPUT` and the `gpi_X` region naming are accepted shorthand for
  "BG output complex".
- **`IZH2007_FS_CORTICAL_INTERNEURON` used for striatal FSIs**: Glossary
  `[NEEDS-REVIEW]` says this is acceptable as engineering shortcut. The
  preset is shared across `cortex_FS_X` and `str_FS_X` use.
- **`current_reward_signal` as DA scalar**: Glossary `[NEEDS-REVIEW]` says
  auditor should not flag every use; the existing config comment block
  (config.py:188-197) acknowledges the simplification.
- **`pfc` region without dlPFC/vmPFC/OFC subdivision**: Glossary
  `[NEEDS-REVIEW]` accepts generic PFC for current code; future-clarity
  flag only.
- **`place_cells` / `goal_cells` (older `--hippocampus` flag)**: Glossary
  `[NEEDS-REVIEW]` accepts both forms; canonical DG/CA3/CA1 used in
  Cluster D.
- **`neuromodulator` as catch-all term**: Glossary `[NEEDS-REVIEW]` notes
  the framework abstracts both transmitters and modulators; this broader
  usage is project-canonical.
- **`HH_OLFACTORY_MITRAL`**: Glossary §"Mitral cell, tufted cell" canonical;
  preset name is fine.
- **`HH_INFERIOR_OLIVE`**: Glossary §IO canonical; preset name fine.
- **`HH_PFC_PYRAMIDAL` (no subdivision)**: Acceptable per glossary
  `[NEEDS-REVIEW]`.
- **`IZH2007_THALAMIC_RELAY` (no specific nucleus)**: Glossary §Thalamus
  notes "generic thalamic relay used for BG output; no specific nucleus
  identification" — accepted.
- **`OLFACTORY_BULB`, `INFERIOR_OLIVE`, `DOPAMINERGIC_MIDBRAIN`,
  `CORTEX_GAMMA_FS_NETWORK` profile names**: Project-canonical.
- **`enable_d1_d2_asymmetry` and `cp_d1_d2_sign`**: Project identifiers,
  documented in CLAUDE.md (Cluster B.1).
- **`enable_bg_neuropeptides`**: Project identifier, documented.
- **`hh_q10_m`, `hh_q10_h`, `hh_q10_n`**: Per-gate Q10 fields,
  biologically-grounded per Mainen & Sejnowski 1996; not canonical glossary
  but standard biophysics.
- **`fused_*` kernel function naming**: GPU kernel convention, not
  biological terminology.
- **`ou_seed`, `ou_tau_ms`, `ou_std_current_pA`**: Ornstein-Uhlenbeck
  process parameters; not in glossary but canonical biophysics.

---

## Notes on scope

- Skipped kernel arithmetic (lines without docstrings/comments) per
  instructions.
- Skipped `bridge.py` HDF5 / queue / threading internals — not biological.
- Skipped `connectivity.py` Watts-Strogatz / spatial-binning math beyond
  spot-checks.
- Re-reading the glossary, the conventions section explicitly accepts the
  project's `gpi_X`, `str_D1_X`, `IZH2007_*` style as project-canonical, so
  the bulk of identifiers in `enums.py` and `regions.py` aren't flagged.
- All Tier 1 findings cluster around small docstring inaccuracies (D1/D2
  receptor descriptions, the STDP delta_t convention parenthetical) and
  prose-symbol alignment (`striatum_D1` should be `str_D1_X`).
