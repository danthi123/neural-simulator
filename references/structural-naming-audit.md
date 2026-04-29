# Structural Naming Audit

**Audit against:** `references/glossary.md` (228 canonical entries) and the catalog cluster hierarchy in `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (~323 entries across clusters A–Q).

**Purpose:** Identify code identifiers (regions, pathways/gates, flags, presets, modulators, bridge-state fields) whose names don't accurately reflect the biology they model. The output is the input to the post-cheat-5 renaming pass.

**Date:** 2026-04-29.

---

## Headline findings

The simulator's identifier surface accreted across ~12 months of incremental research arcs (sessions D, E, F, G, H, I plus the BG cascade refactor and Cluster A–E rollouts). Names were chosen at the moment a feature shipped, often *before* the textbook catalog was systematized, and reflect three recurring failure modes that the recent biology-buildout audit can now correct:

1. **Umbrella-name overclaiming.** Several flags adopt anatomical umbrellas (`--enable-pfc`, `--hippocampus`, `--landmarks`) when the implementation captures only a single feature beneath that umbrella. `--enable-pfc` adds a 60-neuron recurrent pool that models *one* aspect (working-memory persistent activity, catalog G.06/G.08) of a structure that biology subdivides into dlPFC / vmPFC / OFC / ACC. `--hippocampus` predates the canonical Cluster D trisynaptic implementation and uses only sensor-driven `place_cells` + `goal_cells` with no DG / CA3 / CA1. `--landmarks` is a sensor abstraction, not the cluster D landmark-cell biology.

2. **Plasticity-gate mislabeling.** The gate `cortex_to_d1` is the worst case: the runner uses the *same* gate string on both `cortex_X → str_D1_X` and `cortex_X → str_D2_X` (and the patch pathway), so freezing/thawing it actually controls all three. The name describes the first pathway only and silently misleads anyone reading the curriculum logic. Other gates (`hippo_to_cortex`, `cortex_to_str_fs`, `sensory_to_cortex`) use ad-hoc shorthand that loosely follows `<source>_to_<target>` but mixes anatomical (cortex / sensory) with functional (hippo / pfc-pathways) terminology.

3. **Conflated cell-type taxonomies.** All `*_FS_*` regions (cortical `cortex_FS_*`, striatal `str_FS_*`, motor `motor_FS_*`) share the same `IZH2007_FS_CORTICAL_INTERNEURON` preset, which is biologically a cortical PV+ basket-cell analog. Striatal FSI is a distinct PV-FSI class (B.06; ~0.7% of striatum, distinct intrinsic biophysics, NOT one of the chandelier/Martinotti/NGF cortical taxonomy per B.01 supplemental). The runner's striatal FSI section already adds a long comment acknowledging this; the *name* doesn't reflect the distinction. Motor `motor_FS_*` is a project invention with no direct biological referent (real motor output is via α-motoneurons + spinal interneurons, not a cortical-style basket microcircuit).

A fourth, less severe pattern: **per-action shorthand for diffuse brain regions.** `dopamine` is a single A9+A10 collapse (glossary flags this); `stn` is one shared pool (correct since real STN is also a single small nucleus); `goal_cells` and `place_cells` are pre-Cluster-D legacy names whose anatomical analog is closer to PPC / parahippocampal than to canonical place-cell biology. The recent Cluster D rollout introduced canonical `ec / dg / dg_fs / ca3 / ca1` names — these are clean.

The audit identifies **~75 distinct identifiers** across six categories; **~20 are flagged as high-priority renames** with concrete suggestions, and the rest are either fine, acceptable engineering shorthand, or genuinely ambiguous.

---

## Summary table

| Category | identifiers audited | match | partial / acceptable shorthand | mismatch / overclaiming | rename priority high |
|---|---:|---:|---:|---:|---:|
| Regions | 22 | 7 | 10 | 5 | 3 |
| Pathways / Plasticity gates | 13 | 5 | 4 | 4 | 3 |
| CLI flags | 25 | 12 | 6 | 7 | 5 |
| Neuron presets | 12 (selected) | 9 | 2 | 1 | 0 |
| Neuromodulators | 7 | 5 | 1 | 1 | 1 |
| Bridge state fields | 7 (selected) | 4 | 2 | 1 | 1 |
| **Total** | **86** | **42** | **25** | **19** | **13** |

Notes:
- "match" means the name maps cleanly onto the canonical biology.
- "partial / acceptable shorthand" means the name is loose but unambiguous-in-context (e.g., `cortex_X` for M1).
- "mismatch / overclaiming" means the name actively misleads (e.g., `cortex_to_d1` gating D2 too).
- "rename priority high" is a subset of the mismatch column where the misleading effect is severe enough to warrant action.

---

## A. Regions

### Region: `cortex_{N,E,S,W}`
- **What it models:** per-action motor-cortex column (~25 RS pyramidal neurons) that drives the BG cascade. The line comment in the runner already acknowledges this is "primary motor cortex (M1-equivalent) pools".
- **Catalog umbrella:** primary motor cortex (M1) — Cluster H "Motor & spinal output" + Cluster A "Closed BG action-selection loop" + Cluster E (cortical encoding).
- **Canonical feature:** "M1 layer-5 RS pyramidal pool per action channel" (no canonical glossary entry for an "M1 per-action column" — closest are E.04 topographic maps and the corticospinal tract H.16).
- **Current correctness:** **partial / acceptable shorthand** — `cortex_X` is generic but the codebase doesn't model any *other* cortex pool (PFC is separate, sensory is separate). Unambiguous-in-context.
- **Suggested canonical name:** `m1_{N,E,S,W}` or `cortex_m1_{N,E,S,W}` (preferred) when a renaming pass is run. **Defer until other cortex regions are added** (PMd, pre-SMA from G.07, S1 from E.20).
- **Rename blast radius:** large — used in 32 pathway declarations, every action runner, every test, every sidecar JSON, dozens of findings docs. Curriculum logic, gate-name strings, sleep-replay refers to "cortex".
- **Rename priority:** **medium** (defer until additional cortical regions force disambiguation).
- **Catalog ref:** Cluster H, Cluster E, Cluster A.

### Region: `cortex_FS_{N,E,S,W}`
- **What it models:** cortical fast-spiking interneuron pool per action; supports motor-cortex WTA microcircuit (`--cortex-wta` opt-in).
- **Catalog umbrella:** cortical PV+ basket cells (B.01 cortical interneuron diversity).
- **Canonical feature:** "cortical PV+ FS basket cell" — perisomatic-targeting; gamma driver per N.19. Glossary "Cortical FS interneuron (PV+ basket)" entry.
- **Current correctness:** **partial** — `FS` is the electrophysiological class, accurate; `cortex_FS` is acceptable shorthand. Doesn't disambiguate from chandelier / Martinotti / NGF cortical interneuron classes (all currently absent).
- **Suggested canonical name:** keep `cortex_FS_X` for now; document as "cortical PV+ basket-equivalent" in the runner comment. When chandelier / Martinotti are added, rename to `cortex_PV_basket_X`.
- **Rename blast radius:** moderate — the cortical WTA flag is opt-in, so not in flagship.
- **Rename priority:** low.
- **Catalog ref:** B.01.

### Region: `str_D1_{N,E,S,W}`
- **What it models:** direct-pathway D1-receptor-expressing medium spiny neurons per action channel; uses `IZH2007_STRIATAL_MSN_D1` preset; `E_inh = -60 mV` override (B.14).
- **Catalog umbrella:** D1-MSN / striatonigral / direct pathway (Cluster A.01, B.03).
- **Canonical feature:** "D1 MSN" or "direct-pathway MSN".
- **Current correctness:** **match** — exactly the canonical name.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** A.01, B.03.

### Region: `str_D2_{N,E,S,W}`
- **What it models:** indirect-pathway D2-receptor-expressing MSNs per action channel.
- **Catalog umbrella:** D2-MSN / striatopallidal / indirect pathway (A.02, B.03).
- **Canonical feature:** "D2 MSN" or "indirect-pathway MSN".
- **Current correctness:** **match**.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** A.02, B.03.

### Region: `str_FS_{N,E,S,W}`
- **What it models:** striatal fast-spiking interneurons per action — implemented as cross-action feedforward inhibitors (Cluster B.2). Uses `IZH2007_FS_CORTICAL_INTERNEURON` preset (engineering shortcut).
- **Catalog umbrella:** striatal PV-FSI (B.06).
- **Canonical feature:** "PV-FSI" or "PV-positive striatal fast-spiking interneuron". Glossary entry: ~0.7% of striatum (Rymar 2004); 8 distinct striatal interneuron classes per Tepper-2018, of which PV-FSI is just one.
- **Current correctness:** **partial / acceptable shorthand** — `FS` matches the electrophysiology, but elides that this is the **PV-FSI** specifically, not a generic "all striatal interneurons" pool. Catalog flags this as a `[NEEDS-REVIEW]` item.
- **Suggested canonical name:** `str_PV_FSI_X` or `str_pv_fsi_X`. Existing comment in runner already says "this is the **PV-FSI** class" — make the name reflect it.
- **Rename blast radius:** small — `--enable-striatal-fsis` is opt-in, region only present when flag set.
- **Rename priority:** **medium** — would be high if other striatal interneuron classes (NPY-LTS, NPY-NGF, FAI, SABI, CR, THIN, ChI/TAN) get added; defer until then.
- **Catalog ref:** B.06; glossary `[NEEDS-REVIEW] "FSI" disambiguation`.

### Region: `str_patch_{N,E,S,W}`
- **What it models:** striosomal MSN compartment per action; projects to dopamine (canonical) and gpi (secondary). Uses `IZH2007_STRIATAL_MSN_D1` preset.
- **Catalog umbrella:** striosome / patch compartment (B.07).
- **Canonical feature:** "striosomal MSN" (modern term) or "patch MSN" (older).
- **Current correctness:** **partial / acceptable shorthand** — "patch" is the older but still-accepted term; "striosome" is modern. Glossary explicitly lists `str_patch_X` as the project identifier.
- **Suggested canonical name:** consider `str_striosome_X` for newer literature alignment, but `str_patch_X` is fine — catalog uses both terms interchangeably.
- **Rename priority:** low.
- **Catalog ref:** B.07.

### Region: `gpe_{N,E,S,W}`
- **What it models:** PV+ prototypic GPe per action channel; projects to STN, GPi (R3.7 split).
- **Catalog umbrella:** GPe — globus pallidus externus (canonical).
- **Canonical feature:** "GPe prototypic neuron" or "PV+ GPe neuron" per A.13.
- **Current correctness:** **partial / acceptable shorthand** — `gpe` is the umbrella; doesn't disambiguate from PV− arkypallidal (which exists separately as `gpe_arky_X`). The runner comment notes "prototypic (PV+); existing alias preserved" — the alias is intentional. Glossary's note: "older `gpe_X` without PV+/− distinction is the prototypic pool by convention".
- **Suggested canonical name:** consider `gpe_proto_X` or `gpe_pv_X` for symmetry with `gpe_arky_X`. **Backward-compat caveat:** the unqualified `gpe_X` is referenced in many findings docs and the existing-alias comment; renaming would require an alias map.
- **Rename priority:** **medium** (paired with `gpe_arky_X` symmetry).
- **Catalog ref:** A.13.

### Region: `gpe_arky_{N,E,S,W}`
- **What it models:** arkypallidal (PV−, preproenkephalin-mRNA+) GPe per action channel; projects back to striatum (broadcasts to FSIs).
- **Catalog umbrella:** GPe arkypallidal (A.13).
- **Canonical feature:** "arkypallidal neuron" or "PV− GPe neuron".
- **Current correctness:** **match** — `arky` is the canonical specificity tag.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** A.13.

### Region: `gpi_{N,E,S,W}`
- **What it models:** BG output complex per action channel — GPi/SNr collectively (the project doesn't separately model SNr from GPi).
- **Catalog umbrella:** GPi/SNr / BG output complex.
- **Canonical feature:** "GPi/SNr" — both names apply since rodent GPi proper is small and SNr carries most output work. Glossary `[NEEDS-REVIEW] GPi vs SNr in project shorthand` explicitly accepts the unified name.
- **Current correctness:** **partial / acceptable shorthand** — `gpi_X` is project shorthand for GPi/SNr collectively. Glossary explicitly does not flag this as wrong.
- **Suggested canonical name:** keep `gpi_X`; document in code comments that this is GPi/SNr collectively.
- **Rename priority:** none (catalog and glossary explicitly accept this).
- **Catalog ref:** A.04, glossary `[NEEDS-REVIEW]`.

### Region: `stn`
- **What it models:** subthalamic nucleus, single shared pool across actions.
- **Catalog umbrella:** STN (Cluster A).
- **Canonical feature:** "STN" or "subthalamic nucleus".
- **Current correctness:** **match**. STN is anatomically a single small nucleus; not having per-action subdivisions is biologically correct.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** A.16.

### Region: `thal_{N,E,S,W}`
- **What it models:** thalamic relay nucleus per action channel; uses `IZH2007_THALAMIC_RELAY` preset.
- **Catalog umbrella:** motor thalamus (VL/VA per Cluster A).
- **Canonical feature:** "thalamic relay nucleus" or "VA/VL motor thalamus" per glossary.
- **Current correctness:** **partial / acceptable shorthand** — `thal_X` is generic; biology is specifically VL/VA motor thalamus.
- **Suggested canonical name:** `va_vl_X` or `thal_motor_X`. Defer until other thalamic nuclei (TRN, intralaminar / parafascicular) are added.
- **Rename priority:** low.
- **Catalog ref:** A.05, glossary "Thalamus (motor / relay nuclei)".

### Region: `motor_{N,E,S,W}`
- **What it models:** abstract motor output pool per action; uses `IZH2007_RS_CORTICAL_PYRAMIDAL` preset (NOT spinal motoneuron).
- **Catalog umbrella:** corticospinal output (H.16) functionally; cortex+motoneuron mix.
- **Canonical feature:** ambiguous — biology has α-motoneuron (H.x), spinal cord (H.y), and corticospinal tract (H.16) as distinct entities. The project's `motor_X` is a behavioral abstraction.
- **Current correctness:** **partial / acceptable shorthand** — name is generic but the project doesn't model any of: α-motoneurons, spinal cord, muscle. Flagging the abstraction in the runner comment would help.
- **Suggested canonical name:** keep `motor_X` until muscle/spinal-CPG work begins (Tier 3 T3.C). At that point rename to `motor_output_X` (project abstraction) and add separate `alpha_mn_X` / `gamma_mn_X` for biological motoneurons.
- **Rename priority:** low (defer).
- **Catalog ref:** H.16, glossary "α-motoneuron".

### Region: `motor_FS_{N,E,S,W}`
- **What it models:** FS interneuron pool per motor pool, mediates motor WTA (`--motor-lateral-inhibition` opt-in).
- **Catalog umbrella:** none directly — biology routes motor WTA through spinal Renshaw inhibition / reciprocal inhibition (H.07, H.08), not via cortical-style FS interneurons.
- **Canonical feature:** project invention; closest biological analog is reciprocal-inhibition spinal microcircuit (H.07).
- **Current correctness:** **mismatch / underclaim** — the name suggests cortical-FS-like inhibition, but the biology of motor pool WTA is spinal (Renshaw / reciprocal). The flag was tested 2026-04-26 and was MIXED/NEGATIVE; not in flagship.
- **Suggested canonical name:** since the flag is non-recommended and may be removed, **defer**. If kept, rename to `motor_wta_FS_X` to flag it as a non-biological abstraction.
- **Rename priority:** low (flag is opt-in NEGATIVE).
- **Catalog ref:** H.07, H.08.

### Region: `dopamine`
- **What it models:** single shared DA pool — A9 (SNc) + A10 (VTA) collapsed; broadcast modulator. Per-action variants `dopamine_{N,E,S,W}` exist with `--enable-compartmentalized-da` (Cluster C v2).
- **Catalog umbrella:** SNc + VTA / midbrain DA (Cluster C, C.16).
- **Canonical feature:** "SNc" (A9, motor, nigrostriatal) and "VTA" (A10, mesolimbic+mesocortical) are biologically distinct. Glossary explicitly flags collapse as `[discrepancy]` per C.16.
- **Current correctness:** **mismatch** — `dopamine` is a transmitter name being used as a region name; biologically the *region* is SNc + VTA. Glossary flags this.
- **Suggested canonical name:** rename region to `snc` (since the runner comment explicitly says "this is the project's A9-equivalent — SNc dopaminergic neurons"). Keep the *neuromodulator* named `dopamine` (correct — DA is the transmitter). When VTA / mesolimbic arc is added (Tier 2), add a separate `vta` region.
- **Rename blast radius:** moderate — `dopamine` region appears in pathway declarations, sidecar logs, neuromodulator default helpers, several findings docs.
- **Rename priority:** **medium-high** — because this is a single-word identifier conflating transmitter and region, it's particularly misleading.
- **Catalog ref:** C.16, glossary "Substantia nigra pars compacta (SNc)".

### Region: `place_cells`
- **What it models:** opt-in (`--hippocampus`) sparse Gaussian-tuned cells encoding agent (x, y); plastic to all 4 cortex pools.
- **Catalog umbrella:** CA1/CA3 place cells (D.06).
- **Canonical feature:** "place cell" — but glossary and catalog both flag that the project's implementation is **sensor-driven**, not strictly allocentric per O'Keefe & Nadel 1978. Glossary `[NEEDS-REVIEW] "place cell" usage`.
- **Current correctness:** **overclaiming** — the name implies a true allocentric place cell; the implementation is closer to a sensor-driven readout that becomes place-cell-like only when paired with `--landmarks`. Catalog explicitly flags this as a discrepancy.
- **Suggested canonical name:** `pseudo_place_cells` or `sensor_place_cells` to flag the sensor-driven origin. Alternatively when the new Cluster D `ca1` exists, the readout role moves to `ca1` and `place_cells` becomes legacy/deprecated.
- **Rename blast radius:** small — `--hippocampus` flag is opt-in legacy; new `--enable-cluster-d-hippocampus` uses canonical `dg / ca3 / ca1`.
- **Rename priority:** **high** if `--hippocampus` flag is kept; **defer** if the legacy flag is being phased out in favor of Cluster D.
- **Catalog ref:** D.06, glossary `[NEEDS-REVIEW]`.

### Region: `goal_cells`
- **What it models:** opt-in (`--hippocampus`) sparse Gaussian-tuned cells encoding goal (gx, gy); plastic to all 4 cortex pools and (with `--pfc`) to PFC.
- **Catalog umbrella:** **NOT a hippocampal cell type.** Closer biological analog is PPC / parahippocampal goal-direction signals or PFC goal-encoding neurons. Glossary explicitly notes "`goal_cells` region in g11 is closer to PPC than PFC despite naming".
- **Canonical feature:** none in standard nomenclature — "goal cell" is not a canonical place-cell-system term. Catalog has D.09 "object-vector cells" (allocentric, goal-vector-like) as the closest match.
- **Current correctness:** **mismatch / overclaiming** — `goal_cells` invokes hippocampal place-cell semantics but is anatomically PPC- or PFC-like. The fact that the new `--pfc` flag wires `goal_cells → pfc → cortex` underscores this — they're already being treated as upstream inputs to PFC, suggesting they belong in PPC.
- **Suggested canonical name:** `ppc_goal_X` or `pfc_goal_input` (depending on which framing is adopted). Catalog D.09 "object-vector cell" is the closest canonical match if the cells are intended as allocentric goal-vector encoders.
- **Rename blast radius:** moderate — pathway declarations, the `--goal-silence` flag, multiple PFC-related findings.
- **Rename priority:** **high** — name is genuinely misleading.
- **Catalog ref:** glossary "Posterior parietal cortex (PPC)"; D.09 object-vector cells.

### Region: `beacon_sensors`
- **What it models:** 8 directional sensors with preferred bearings; activation = beacon intensity × cosine alignment. `--beacon-perception` arc, replaces direct (gx, gy) goal access.
- **Catalog umbrella:** abstract sensor; closest biological analog is K (sensory transduction) or E.04 topographic maps, but the implementation is purely an engineering input layer.
- **Canonical feature:** project abstraction; no clean catalog match.
- **Current correctness:** **partial / acceptable shorthand** — `beacon` is project terminology for the simulated environment cue; not a standard biological term but unambiguous within the project.
- **Suggested canonical name:** keep — flagging this as a project-specific abstraction in code comments is sufficient. When real sensory transduction (Cluster K) is added, this can be retired.
- **Rename priority:** low.
- **Catalog ref:** Cluster K (where this would eventually live).

### Region: `landmark_sensors`
- **What it models:** 8 directional sensors tied to a fixed-position landmark; activation depends on (distance, bearing) to landmark. `--landmarks` arc, drives plastic learning of place cells.
- **Catalog umbrella:** abstract sensor; closest biological analog is allothetic (landmark-based) navigation input — D.09 object-vector cells.
- **Canonical feature:** project abstraction.
- **Current correctness:** **partial / acceptable shorthand** — sensor abstraction; the *flag* `--landmarks` is more problematic (claims a whole landmark-cell biology, see D section).
- **Suggested canonical name:** keep region name; address the umbrella overclaim at the flag layer.
- **Rename priority:** low (region); see Flag section for `--landmarks`.
- **Catalog ref:** D.09.

### Region: `pfc`
- **What it models:** 60 recurrent neurons (`internal_density=0.20`); plastic recurrence; uses `IZH2007_HIPPO_PYRAMIDAL` preset (line comment notes "can switch to dedicated PFC preset HH_PFC_PYRAMIDAL"). Models working-memory persistent activity (G.06, G.08).
- **Catalog umbrella:** PFC — entire prefrontal cortex (dlPFC, vmPFC, OFC, ACC).
- **Canonical feature:** "PFC working memory persistent activity" (G.06 / G.08) — specifically dlPFC.
- **Current correctness:** **overclaiming** — `pfc` claims the whole PFC umbrella; we model one feature (working memory). Glossary `[NEEDS-REVIEW] "PFC" subdivisions` accepts this for now but flags for future clarity.
- **Suggested canonical name:** `dlpfc_wm` or `pfc_wm` (preferred, balances brevity and accuracy). Aligns with G.06.
- **Rename blast radius:** moderate — pathway declarations (`goal_to_pfc_weight`, `pfc_to_cortex_weight`), CLI flag `--pfc`, plasticity gate `pfc_pathways`, sidecar logs.
- **Rename priority:** **high** — actively misleading; flag, region, and gate all need coordinated rename.
- **Catalog ref:** G.06, G.08, glossary `[NEEDS-REVIEW]`.

### Region: `sensory`
- **What it models:** opt-in (`--learned-perception`) 7×7 grid of position-tuned neurons (49 cells) feeding cortex. Each neuron is tuned to a (dx, dy) ∈ [−3, 3]² position offset.
- **Catalog umbrella:** generic sensory cortex; closest biological analog is S1 (E.20) or topographic-map encoding (E.04).
- **Canonical feature:** project abstraction; sensory-input layer.
- **Current correctness:** **partial / acceptable shorthand** — `sensory` is generic; biology has many sensory cortices (S1, V1, A1) that this stands in for.
- **Suggested canonical name:** `pos_sensory` or `sensory_pos_grid` to flag what's being encoded; alternatively `s1_pos` if the analog is S1.
- **Rename priority:** low.
- **Catalog ref:** E.04, E.20.

### Region: `ec`
- **What it models:** entorhinal cortex stub (80 neurons) — bridges perception → hippocampus trisynaptic loop (`--enable-cluster-d-hippocampus`).
- **Catalog umbrella:** entorhinal cortex (D.03, D.04).
- **Canonical feature:** "EC" (whole) — biology distinguishes EC layers II / III / deep. Project models a single pool.
- **Current correctness:** **partial / acceptable shorthand** — `ec` is the umbrella; biology has layer-II vs layer-III pathways. Glossary documents this.
- **Suggested canonical name:** keep; consider `ec_ii` or `ec_iii` if/when layer-specific pathways are added.
- **Rename priority:** low.
- **Catalog ref:** D.03, D.04, glossary "Entorhinal cortex (EC)".

### Region: `dg`, `dg_fs`, `ca3`, `ca1`
- **What they model:** canonical Cluster D trisynaptic loop — DG granule cells, DG fast-spiking interneurons (FFi), CA3 (autoassociator with `internal_density=0.30`), CA1 (readout).
- **Catalog umbrella:** hippocampus subregions (D.03, D.05, D.12, D.13).
- **Canonical feature:** "DG", "CA3", "CA1" — exact canonical names. `dg_fs` follows the `*_FS_*` cortical-FSI shortcut — biologically these are PV+ basket cells in DG.
- **Current correctness:** **match** for `dg`, `ca3`, `ca1`; **partial** for `dg_fs` (same FSI conflation as `cortex_FS`/`str_FS`).
- **Suggested canonical name:** keep `dg`, `ca3`, `ca1`. For `dg_fs`, consider `dg_pv` to align with future hippocampal interneuron taxonomy.
- **Rename priority:** none for `dg/ca3/ca1`; low for `dg_fs`.
- **Catalog ref:** D.03, D.05, D.12, D.13.

---

## B. Pathways / Plasticity gates

The runner uses `RegionPathway.plasticity_gate` strings to tag pathways for runtime gating. Below are all 10 gate names found.

### Gate: `cortex_to_d1`
- **What it gates:** `cortex_X → str_D1_X` AND `cortex_X → str_D2_X` AND `cortex_X → str_patch_X` (per-action, plus cross-projections when `--bg-cross-projections` is on, which use `bg_cross_projections` instead). The line in the runner is:
  - line 793: `gate = "cortex_to_d1"` for same-action D1 path
  - lines 802–803, 807–812: applied to BOTH `str_D1_X` AND `str_D2_X` paths
  - line 962: applied to `str_patch_X` path too
- **Catalog umbrella:** corticostriatal plasticity (Cluster B + J.07).
- **Canonical name:** "corticostriatal" (covers cortex→D1 + cortex→D2 + cortex→patch).
- **Current correctness:** **mismatch / overclaiming narrowness** — gate string says D1-only, gates D1+D2+patch.
- **Suggested canonical name:** `corticostriatal` (preferred) or `cortex_to_msn` (covers D1+D2; would need a separate gate or comment for the patch addition).
- **Rename blast radius:** moderate — the curriculum logic uses this gate string to thaw cortex→striatum at phase 2. Used in 3 pathway declarations + the curriculum freeze/thaw API. Several plan documents and findings reference `cortex_to_d1`.
- **Rename priority:** **high** — name is genuinely misleading; anyone reading curriculum code might think they're freezing only the direct pathway.
- **Catalog ref:** J.07, A.01, A.02.

### Gate: `bg_cross_projections`
- **What it gates:** opt-in (`--bg-cross-projections`) cortex_X → str_D1_Y / str_D2_Y plasticity for X ≠ Y.
- **Catalog umbrella:** corticostriatal plasticity (subset).
- **Canonical name:** "corticostriatal cross-projections" or "cortex-to-msn-cross".
- **Current correctness:** **partial** — `bg_cross_projections` is colloquial; doesn't specify *which* BG projections cross. The actual implementation is cortex→str cross-action (not BG-internal cross).
- **Suggested canonical name:** `corticostriatal_cross` or `cortex_to_msn_cross`.
- **Rename priority:** medium — `--bg-cross-projections` is opt-in NEGATIVE in current evaluation; if kept, rename for clarity.
- **Catalog ref:** A.06 cortico-striatal topography.

### Gate: `sensory_to_cortex`
- **What it gates:** plastic sensory → cortex_{N,E,S,W} pathway (`--learned-perception`).
- **Catalog umbrella:** thalamocortical / sensory-to-cortical plasticity (Cluster E).
- **Canonical name:** acceptable as `sensory_to_cortex` since `sensory` and `cortex` are existing region names.
- **Current correctness:** **partial / acceptable shorthand** — name follows `<source>_to_<target>` convention with project shorthand for both ends.
- **Suggested canonical name:** keep; rename if/when `sensory` is renamed to `s1_pos` (then rename to `s1_to_m1`).
- **Rename priority:** low.
- **Catalog ref:** E.04, E.20.

### Gate: `hippo_to_cortex`
- **What it gates:** `place_cells → cortex_X` and `goal_cells → cortex_X` plasticity (`--hippocampus` flag).
- **Catalog umbrella:** hippocampo-cortical plasticity (D.04, D.20).
- **Canonical name:** "hippocampo-cortical" or `hippocampus_to_cortex`.
- **Current correctness:** **partial / acceptable shorthand** — `hippo` is informal abbreviation for hippocampus. Functionally accurate but the source regions (`place_cells`, `goal_cells`) are not strictly hippocampal in the project (see Region section).
- **Suggested canonical name:** `place_goal_to_cortex` or `hippocampo_cortical` (more formal).
- **Rename priority:** medium — depends on `place_cells`/`goal_cells` rename.
- **Catalog ref:** D.20.

### Gate: `beacon_to_goal`
- **What it gates:** `beacon_sensors → goal_cells` plasticity (`--beacon-perception`).
- **Catalog umbrella:** sensory-to-association plasticity (project abstraction).
- **Canonical name:** acceptable as project shorthand.
- **Current correctness:** **partial / acceptable shorthand**.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** project-specific.

### Gate: `landmark_to_place`
- **What it gates:** `landmark_sensors → place_cells` plasticity (`--landmarks`).
- **Catalog umbrella:** sensory-to-association plasticity.
- **Canonical name:** acceptable.
- **Current correctness:** **partial / acceptable shorthand**.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** project-specific.

### Gate: `pfc_pathways`
- **What it gates:** `goal_cells → pfc` and `pfc → cortex_X` plasticity (`--pfc`).
- **Catalog umbrella:** PFC working-memory pathways (G.06).
- **Canonical name:** if region is renamed `dlpfc_wm`, gate should become `dlpfc_wm_pathways`.
- **Current correctness:** **partial / overclaiming** — `pfc` umbrella overclaim (see Region section); `_pathways` suffix is lazily generic for "the cluster of plastic pathways involving this region".
- **Suggested canonical name:** `dlpfc_wm_io` (covers both input and output pathways) or split into `goal_to_dlpfc` and `dlpfc_to_m1`.
- **Rename priority:** **medium** — coordinated with `pfc` region rename.
- **Catalog ref:** G.06.

### Gate: `pfc_internal`
- **What it gates:** PFC recurrent connections (`internal_density>0`) plasticity. Not currently visible in the grep output but present in plan docs.
- **Catalog umbrella:** PFC recurrent attractor (G.06, G.08).
- **Canonical name:** `dlpfc_wm_recurrent` if region is renamed.
- **Rename priority:** medium (paired).
- **Catalog ref:** G.06.

### Gate: `cortex_to_str_fs`
- **What it gates:** `cortex_X → str_FS_X` (cluster B.2 cortex-to-striatal-FSI drive). Static (`plastic=False`) so the gate isn't actively used; reserved for future plasticity.
- **Catalog umbrella:** cortico-FSI drive (B.06).
- **Canonical name:** `cortico_pv_fsi` if FSI region is renamed; `cortex_to_pv_fsi` is also acceptable.
- **Current correctness:** **partial** — gate is reserved; name follows `<source>_to_<target>` with `str_fs` shorthand.
- **Suggested canonical name:** keep, paired with `str_FS` rename if it happens.
- **Rename priority:** low.
- **Catalog ref:** B.06.

### Gate: `str_fs_to_msn`
- **What it gates:** `str_FS_X → str_D1/D2_Y` cross-action plasticity (cluster B.2). Static.
- **Catalog umbrella:** PV-FSI feedforward inhibition (B.06).
- **Canonical name:** `pv_fsi_to_msn` if FSI region is renamed.
- **Rename priority:** low.
- **Catalog ref:** B.06.

### Gates: `sensory_to_ec`, `ec_to_dg`, `ec_to_ca1`, `dg_to_ca3`, `ca3_to_ca1`
- **What they gate:** Cluster D trisynaptic-loop plastic pathways (`--enable-cluster-d-hippocampus`). Each follows `<source>_to_<target>` with canonical region names.
- **Catalog umbrella:** D.03 trisynaptic, D.04 temporoammonic, D.05 CA3 recurrent.
- **Canonical name:** match. These are the cleanest gates in the codebase.
- **Current correctness:** **match**.
- **Suggested canonical name:** keep all five.
- **Rename priority:** none.
- **Catalog ref:** D.03–D.05.

---

## C. CLI flags

### Flag: `--enable-pfc` / `--pfc`
- **What it does:** adds the `pfc` region with `n_pfc=60` recurrent neurons, `internal_density=0.20`. Models working-memory persistent activity.
- **Catalog umbrella:** Cluster G "Working memory / PFC".
- **Canonical feature:** "PFC working memory persistent activity" (G.06, G.08).
- **Current correctness:** **overclaiming** — `pfc` is the whole PFC; we model only WM persistent activity.
- **Suggested canonical name:** `--enable-dlpfc-wm` or `--enable-pfc-wm`. Catalog G.06 maps to dlPFC specifically.
- **Rename blast radius:** **large** — flagship recipe in CLAUDE.md, multiple findings (PFC working memory, perception arc), webapp preset definitions, sidecar logs.
- **Rename priority:** **high** — single most-overclaimed flag.
- **Catalog ref:** G.06, G.08.

### Flag: `--hippocampus`
- **What it does:** adds `place_cells` + `goal_cells` regions with sparse Gaussian tuning. Pre-Cluster-D legacy; not the trisynaptic loop.
- **Catalog umbrella:** hippocampus (Cluster D).
- **Canonical feature:** none — implementation is sensor-driven place-cell-like activations + non-canonical goal cells. Glossary explicitly notes this is the older flag with non-canonical regions.
- **Current correctness:** **overclaiming** — `--hippocampus` claims the whole structure; we model two non-canonical regions.
- **Suggested canonical name:** `--enable-place-goal-readout` or `--enable-pseudo-place-cells`. Newer `--enable-cluster-d-hippocampus` is more accurate.
- **Rename blast radius:** large — flagship recipe, many findings.
- **Rename priority:** **high** — but consider phasing out the legacy flag entirely once Cluster D ships in flagship.
- **Catalog ref:** D.06, glossary `[NEEDS-REVIEW] "hippocampus" without subregion`.

### Flag: `--enable-cluster-d-hippocampus`
- **What it does:** adds `ec`, `dg`, `dg_fs`, `ca3`, `ca1` regions and the trisynaptic loop pathways.
- **Catalog umbrella:** Cluster D trisynaptic core (D.03–D.05).
- **Canonical feature:** "trisynaptic pathway" (D.03).
- **Current correctness:** **partial / acceptable shorthand** — flag name says "Cluster D" + "hippocampus" — somewhat redundant but unambiguous. The flag *correctly* names what it implements (the cluster).
- **Suggested canonical name:** keep, or shorten to `--enable-trisynaptic-loop` (which is the canonical name for the actual circuit added).
- **Rename priority:** low.
- **Catalog ref:** D.03–D.05.

### Flag: `--enable-cluster-a-closed-loop`
- **What it does:** adds hyperdirect (cortex → STN) + thalamocortical feedback (thal_X → cortex_X) pathways.
- **Catalog umbrella:** Cluster A closed BG loop.
- **Canonical feature:** "hyperdirect pathway" (A.03) + "cortico-BG-thalamo-cortical loops" (A.05).
- **Current correctness:** **match** — flag uses cluster identifier; underlying biology is correctly named.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** A.03, A.05.

### Flag: `--enable-cluster-e-topography`
- **What it does:** adds 2D coordinates to cortex / D1 / D2 / patch regions; enables distance-dependent connection probability.
- **Catalog umbrella:** Cluster E topographic maps (E.04).
- **Canonical feature:** "topographic maps".
- **Current correctness:** **match**.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** E.04.

### Flag: `--enable-tans`
- **What it does:** registers acetylcholine neuromodulator (TAN-pause-on-reward); when fully wired, would gate corticostriatal plasticity.
- **Catalog umbrella:** Cluster B / C — TAN/ChI biology (B.05, C.18).
- **Canonical feature:** "TAN" or "ChI" — both names refer to the same cell.
- **Current correctness:** **match** — TAN is the canonical electrophysiological-literature name.
- **Suggested canonical name:** keep. (Glossary: ChI is the anatomy/molecular literature name; TAN is electrophysiology — both accepted.)
- **Rename priority:** none.
- **Catalog ref:** B.05.

### Flag: `--enable-d1-d2-asymmetry`
- **What it does:** adds opposite-sign DA modulation of D1 vs D2 plasticity (Cluster B.1).
- **Catalog umbrella:** D1/D2 segregation (B.03, O.03).
- **Canonical feature:** "D1 vs D2 MSN segregation — opposing DA modulation".
- **Current correctness:** **match**.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** B.03.

### Flag: `--enable-striatal-fsis`
- **What it does:** adds `str_FS_X` regions and FSI cross-action feedforward inhibition.
- **Catalog umbrella:** PV-FSI (B.06).
- **Canonical feature:** "PV-FSI" or "PV+ basket-equivalent".
- **Current correctness:** **partial** — `striatal-fsis` doesn't disambiguate from other striatal interneuron classes (NPY-LTS, NGF, etc.). Catalog flags this is just one of 8.
- **Suggested canonical name:** `--enable-striatal-pv-fsi` or `--enable-striatal-pv-basket`.
- **Rename priority:** medium (pair with `str_FS` region rename).
- **Catalog ref:** B.06, B.01 supplemental.

### Flag: `--enable-tonic-da` / `--enable-compartmentalized-da`
- **What they do:** `--enable-tonic-da` registers the default DA neuromodulator with tonic baseline; `--enable-compartmentalized-da` (Cluster C v2) registers per-action `dopamine_X` modulators.
- **Catalog umbrella:** Cluster C (DA).
- **Canonical feature:** "tonic DA" (C.20) and "compartmentalized DA" (catalog framing).
- **Current correctness:** **match**.
- **Suggested canonical name:** keep both.
- **Rename priority:** none.
- **Catalog ref:** C.20, C.32.

### Flag: `--enable-bg-neuropeptides`
- **What it does:** registers `dynorphin`, `substance_p`, `enkephalin` neuromodulators (R3.6).
- **Catalog umbrella:** Cluster A.01–A.02 D1/D2 co-release; Cluster C.08 neuropeptides.
- **Canonical feature:** D1 co-release of dynorphin + substance P; D2 co-release of enkephalin.
- **Current correctness:** **match** — neuropeptides is the right umbrella for the three peptides.
- **Suggested canonical name:** keep; possibly more precise as `--enable-msn-co-release` or `--enable-d1-d2-peptides`.
- **Rename priority:** low.
- **Catalog ref:** A.01, A.02, C.08.

### Flag: `--bg-lateral-inhibition`
- **What it does:** adds `str_D1_X → str_D1_Y` and `str_D2_X → str_D2_Y` cross-pool inhibition for X≠Y. **GO** in flagship.
- **Catalog umbrella:** MSN lateral inhibition (B.04).
- **Canonical feature:** "MSN lateral inhibition" — but catalog supplemental flags that this is *anatomically backwards*: real cross-pool WTA in striatum is FSI feedforward, not MSN-MSN feedback (B.04 supplemental, Wilson 2007 PBR-160 ch 6).
- **Current correctness:** **partial / mismatch** — the *flag name* claims "BG lateral inhibition", which is broad and could imply pallidal lateral inhibition. The implementation is specifically *MSN* lateral inhibition. Functionally correct, biologically backwards (FSI is the real substrate).
- **Suggested canonical name:** `--enable-msn-lateral-inhibition` (more specific). The B.04 supplemental flags this as a known discrepancy.
- **Rename priority:** **medium** — name is broader than the implementation; biology is documented as imperfect.
- **Catalog ref:** B.04, B.04 supplemental.

### Flag: `--motor-lateral-inhibition`
- **What it does:** adds `motor_FS_X` regions and motor WTA microcircuit. NEGATIVE in evaluation.
- **Catalog umbrella:** none directly biological — see `motor_FS_*` Region entry.
- **Canonical feature:** project invention.
- **Current correctness:** **mismatch** — name suggests cortical-style FS WTA, but real motor WTA is spinal (Renshaw, reciprocal inhibition).
- **Suggested canonical name:** `--enable-motor-pool-wta` (descriptive without overclaiming biology) or remove flag entirely if no longer used.
- **Rename priority:** low (NEGATIVE flag).
- **Catalog ref:** H.07–H.08.

### Flag: `--cortex-wta`
- **What it does:** adds `cortex_FS_X` regions and cortex WTA pathways.
- **Catalog umbrella:** cortical PV+ basket WTA (B.01, gamma circuits N.19).
- **Canonical feature:** PING gamma circuits.
- **Current correctness:** **partial** — name uses "WTA" (winner-take-all) which is computational, not biological. Biology is "perisomatic basket inhibition driving gamma".
- **Suggested canonical name:** `--enable-m1-pv-basket` (biology-grounded) or `--enable-cortex-pv-fsi` (matching the cortex/striatal naming). Or keep `--cortex-wta` as a deliberate computational descriptor.
- **Rename priority:** low.
- **Catalog ref:** B.01, N.19.

### Flag: `--landmarks`
- **What it does:** adds `landmark_sensors` region + `landmark_sensors → place_cells` pathway.
- **Catalog umbrella:** Cluster D landmark cells (D.09 object-vector cells); Cluster E sensor-driven encoding.
- **Canonical feature:** the implementation is *sensor* abstraction, not the D.09 landmark-cell biology (which is downstream of multimodal cortex).
- **Current correctness:** **overclaiming** — `--landmarks` invokes a whole class of cells (landmark cells, object-vector cells); the implementation is just a sensor abstraction.
- **Suggested canonical name:** `--enable-landmark-sensor` (descriptive of the sensor abstraction).
- **Rename priority:** **high** — name overclaims biology.
- **Catalog ref:** D.09.

### Flag: `--beacon-perception`
- **What it does:** adds `beacon_sensors` region + `beacon_sensors → goal_cells` pathway. Project-specific environmental cue.
- **Catalog umbrella:** sensor abstraction (closest: K sensory transduction or E.04 topographic maps).
- **Canonical feature:** project abstraction.
- **Current correctness:** **partial / acceptable shorthand** — beacon is project terminology.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** project-specific.

### Flag: `--cue-reflex`
- **What it does:** adds direct cue-→ -motor reflex when `--cue-reflex-replaces-heuristic` is on. Bypass for the navigation heuristic.
- **Catalog umbrella:** none — engineering bypass.
- **Canonical feature:** project abstraction.
- **Current correctness:** **partial / acceptable shorthand**.
- **Suggested canonical name:** keep.
- **Rename priority:** low.

### Flag: `--learned-perception`
- **What it does:** adds `sensory` region (49-cell position-tuned grid) + plastic `sensory → cortex_X` pathway.
- **Catalog umbrella:** Cluster E sensory-to-cortical encoding.
- **Canonical feature:** "S1 position encoding" (E.04, E.20) — closest.
- **Current correctness:** **partial / acceptable shorthand** — name describes the *function* (perception is learned via STDP) rather than the *biology* (position-tuned sensory cortex).
- **Suggested canonical name:** keep `--learned-perception` for the functional aspect; pair with biological-region rename if `sensory` becomes `s1_pos`.
- **Rename priority:** low.
- **Catalog ref:** E.04, E.20.

### Flag: `--sensed-reward`
- **What it does:** computes reward from beacon-intensity gradient instead of distance to goal. Closes a "cheat" by removing direct goal-coordinate access.
- **Catalog umbrella:** Cluster K (sensory transduction) + O (reward).
- **Canonical feature:** project abstraction (gradient-following).
- **Current correctness:** **partial / acceptable shorthand**.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** project-specific.

### Flag: `--surprise-lr-boost`
- **What it does:** when |reward − reward_ema_pre| is high, multiply `reward_learning_rate` by `(1 + α × |RPE|)`. Models NE-like fast meta-modulation. Pearce-Hall analog.
- **Catalog umbrella:** Cluster C two-component DA (C.32) — Component 1 (salience-blind detection); Pearce-Hall attentional learning (catalog and glossary).
- **Canonical feature:** "Pearce-Hall attentional learning rule" (catalog and glossary explicitly state "functionally implemented as `--surprise-lr-boost`").
- **Current correctness:** **match (functional)** — the flag's behavior matches Pearce-Hall; the name `surprise-lr-boost` is descriptive but not biologically anchored.
- **Suggested canonical name:** keep, optionally add `--enable-pearce-hall` alias.
- **Rename priority:** low.
- **Catalog ref:** C.32 supplemental, "Pearce-Hall attentional learning".

### Flag: `--adaptive-da` / `--adaptive-da-ema-decay-negative`
- **What it does:** asymmetric reward-EMA-gated per-action DA targeting; slow positive ramp / fast negative dip.
- **Catalog umbrella:** Cluster C two-component DA (C.32) — Component 2 (utility / RPE).
- **Canonical feature:** "Component 2 utility-RPE DA" / "asymmetric phasic DA dip vs ramp".
- **Current correctness:** **match (functional)** — flag is descriptive, not biologically anchored, but matches Schultz 1998/2016 asymmetric phasic DA.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** C.32, glossary "Two-component DA response".

### Flag: `--curriculum`
- **What it does:** runs phase-1 with frozen perception layers + plastic cortex→striatum, then thaws perception at phase-2.
- **Catalog umbrella:** Cluster L critical periods (L.04, L.19).
- **Canonical feature:** "critical period" — glossary "functionally captured by curriculum + plasticity gates".
- **Current correctness:** **match (functional)**.
- **Suggested canonical name:** keep, optionally `--enable-critical-period-curriculum`.
- **Rename priority:** none.
- **Catalog ref:** L.04, L.19.

### Flag: `--developmental-pretraining`
- **What it does:** opt-in NO-GO experiment — pre-trains cross-projections during a 5K-trial critical period, then freezes for eval.
- **Catalog umbrella:** Cluster L critical periods.
- **Canonical feature:** "critical period" emulation.
- **Current correctness:** **match** — name is biology-grounded.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** L.04.

### Flag: `--enable-structural-pruning`
- **What it does:** opt-in structural-plasticity pruning of weak synapses (Cluster B Option 1).
- **Catalog umbrella:** structural plasticity (Cluster J, L.03).
- **Canonical feature:** "synapse elimination" / "axon pruning".
- **Current correctness:** **match**.
- **Suggested canonical name:** keep.
- **Rename priority:** none.
- **Catalog ref:** L.03.

### Flag: `--per-action-da` / `--da-gated-wta` / `--rpe-scaled-reward`
- **What they do:** various opt-in DA / WTA / RPE refinement variants. Mostly NEGATIVE in evaluation.
- **Catalog umbrella:** Cluster C variants.
- **Canonical feature:** none with clean catalog mapping (these are project-specific algorithmic variants).
- **Current correctness:** **partial / acceptable shorthand**.
- **Suggested canonical name:** keep.
- **Rename priority:** low.
- **Catalog ref:** project-specific.

### Flag: `--informed-init` / `--informed-init-alpha`
- **What it does:** initializes `sensory → cortex` weights with a hand-coded position-to-action heuristic (`α=8.0`). Engineering bypass for cold-start.
- **Catalog umbrella:** none (engineering shortcut).
- **Canonical feature:** project abstraction.
- **Current correctness:** **partial / acceptable shorthand**.
- **Rename priority:** low.

### Flag: `--goal-schedule`
- **What it does:** selects between {default 2-goal, multi 4-goal, curriculum 1-flip} task schedules.
- **Catalog umbrella:** task definition.
- **Canonical feature:** project abstraction.
- **Current correctness:** **partial / acceptable shorthand**.
- **Rename priority:** low.

---

## D. Neuron presets

### Preset: `IZH2007_RS_CORTICAL_PYRAMIDAL`
- **What it models:** regular-spiking cortical pyramidal neuron (Izh-2007 9-parameter).
- **Canonical:** "RS pyramidal" or "L2/3 / L5 RS pyramidal".
- **Current correctness:** **match**.
- **Rename priority:** none.

### Preset: `IZH2007_FS_CORTICAL_INTERNEURON`
- **What it models:** fast-spiking cortical PV+ basket interneuron.
- **Canonical:** "PV+ FS interneuron" or "cortical fast-spiking interneuron".
- **Current correctness:** **match** — preset name is correct. **Concern:** this preset is *also used* by `str_FS_*` and `motor_FS_*` regions (engineering shortcut; biologically distinct). Glossary acknowledges this `[NEEDS-REVIEW]`.
- **Rename priority:** none for the preset itself; the issue is shared use across biologically distinct populations (covered in Region section).
- **Catalog ref:** B.01, B.06.

### Preset: `IZH2007_STRIATAL_MSN_D1` / `_D2`
- **What it models:** D1/D2 medium spiny neurons.
- **Canonical:** match.
- **Rename priority:** none.

### Preset: `IZH2007_STRIATAL_MSN`
- **What it models:** generic MSN (no D1/D2 distinction).
- **Canonical:** "MSN" — accepted.
- **Current correctness:** **partial / acceptable shorthand** — generic; D1/D2-specific variants exist.
- **Rename priority:** none (deprecated in flagship; D1/D2-specific variants used).

### Preset: `IZH2007_STRIATAL_TAN`
- **What it models:** cholinergic tonically-active interneuron (= ChI).
- **Canonical:** "TAN" or "ChI" — both accepted.
- **Current correctness:** **match**.
- **Rename priority:** none.
- **Catalog ref:** B.05.

### Preset: `IZH2007_GPE_PACEMAKER` / `IZH2007_GPI_OUTPUT` / `IZH2007_STN_BURST`
- **What they model:** GPe (prototypic by convention), GPi/SNr output, STN.
- **Canonical:** match.
- **Rename priority:** none. (Note: the GPE preset doesn't disambiguate prototypic vs arkypallidal — but the comment notes default is prototypic by convention.)

### Preset: `IZH2007_THALAMIC_RELAY` / `IZH2007_THALAMIC_RETICULAR`
- **What they model:** TC neurons; TRN.
- **Canonical:** "thalamic relay nucleus / TC" and "TRN".
- **Current correctness:** **match**.
- **Rename priority:** none.

### Preset: `IZH2007_HIPPO_PYRAMIDAL`
- **What it models:** generic CA1/CA3 pyramidal (IB-like). Used by `place_cells`, `goal_cells`, `pfc`, `dg`, `ca3`, `ca1` regions in the project.
- **Canonical:** "CA1 pyramidal" / "CA3 pyramidal" or "hippocampal pyramidal" — generic.
- **Current correctness:** **match (preset)** — `IZH2007_HIPPO_PYRAMIDAL` is correct for the preset itself. **Engineering concern:** preset is used by non-hippocampal regions (`pfc`, `goal_cells`); see Region section.
- **Rename priority:** none for preset.

### Preset: `IZH2007_DOPAMINE`
- **What it models:** SNc/VTA DA neuron.
- **Canonical:** "DA neuron" or "SNc/VTA DA".
- **Current correctness:** **match**.
- **Rename priority:** none.

### Preset: `HH_DOPAMINE_SNC` (and other `HH_*` presets)
- **What they model:** Hodgkin-Huxley variants of the above.
- **Canonical:** match. `SNC` suffix is correct (doesn't claim VTA).
- **Rename priority:** none.

### Preset: `HH_PFC_PYRAMIDAL`
- **What it models:** PFC pyramidal (HH variant); not currently used in flagship `pfc` region (which uses `IZH2007_HIPPO_PYRAMIDAL` preset).
- **Canonical:** "PFC pyramidal" — generic; doesn't disambiguate dlPFC / vmPFC / OFC.
- **Current correctness:** **partial** — same overclaim as the `pfc` region.
- **Suggested canonical name:** `HH_DLPFC_PYRAMIDAL`.
- **Rename priority:** low (preset rename has saved-profile compatibility cost; defer until region rename).
- **Catalog ref:** G.06.

**Backward-compat caveat for all preset renames:** preset enum names appear in saved JSON profiles in `simulation_profiles/`. Any rename needs an alias map (`HH_PFC_PYRAMIDAL` → `HH_DLPFC_PYRAMIDAL`) maintained in `sim/enums.py` for at least one release cycle.

---

## E. Neuromodulators

### Modulator: `dopamine`
- **What it models:** broadcast DA scalar driving plasticity rate (`plasticity_rate sensitivity=+1.0 scope=all`); production from reward signal.
- **Catalog umbrella:** DA (Cluster C; A9 nigrostriatal + A10 mesolimbic).
- **Canonical feature:** "dopamine" as a transmitter — correct.
- **Current correctness:** **partial** — the *transmitter* name is correct; the implementation collapses A9 + A10 (catalog flags as `[discrepancy]` C.16). Glossary `[NEEDS-REVIEW] "DA" vs "current_reward_signal"`.
- **Suggested canonical name:** keep `dopamine` for the modulator; rename the **region** to `snc` (see Region section). When VTA is added, register a separate `dopamine_meso` modulator.
- **Rename priority:** low (modulator); medium (region disambiguation).
- **Catalog ref:** C.04, C.16, C.32.

### Modulator: `dopamine_{N,E,S,W}` (Cluster C v2)
- **What it models:** per-action DA channels (`--enable-compartmentalized-da`); each targets only synapses with matching `action_index`.
- **Catalog umbrella:** compartmentalized DA (Tier 2 T2.D in roadmap).
- **Canonical feature:** "compartmentalized DA" — project term; biological evidence in Schultz / Berke per-DA-axon specificity.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Modulator: `acetylcholine`
- **What it models:** TAN-pause-on-reward; opens corticostriatal plasticity window.
- **Catalog umbrella:** Cluster C.18 ACh.
- **Canonical feature:** "ACh" or "acetylcholine" — correct transmitter; implementation is specifically the TAN-pause arm.
- **Current correctness:** **partial** — `acetylcholine` claims the whole transmitter; we model the TAN-pause behavior only. Brain has multiple ACh sources (basal forebrain Ch1–4, brainstem PPN/LDT — all missing).
- **Suggested canonical name:** `acetylcholine_tan` (specifies that this is the striatal-TAN-driven population). When basal-forebrain ACh is added (Cluster C/N), it should be a separate modulator.
- **Rename priority:** **medium** — rename improves clarity for future BF-ACh addition.
- **Catalog ref:** C.18, B.05.

### Modulator: `dynorphin`
- **What it models:** D1-driven KOR ligand; suppresses corticostriatal plasticity (sensitivity = −0.4).
- **Catalog umbrella:** Cluster A.01 / B.03 / C.08 D1 co-release.
- **Canonical feature:** "dynorphin" — correct.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Modulator: `enkephalin`
- **What it models:** D2-driven DOR ligand; raises plasticity (mirrors DA).
- **Catalog umbrella:** A.02, B.03, C.08 D2 co-release.
- **Canonical feature:** "enkephalin" — correct.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Modulator: `substance_p`
- **What it models:** D1-driven NK-1 ligand; raises ACh; modeled as `excitability_drive scope=all` (with note about avoiding double-modulation with ACh).
- **Catalog umbrella:** B.05, C.08, A.01.
- **Canonical feature:** "substance P" — correct.
- **Current correctness:** **match** — naming is correct; underscore in identifier is conventional for code.
- **Rename priority:** none.

### Production rule names — `manual`, `from_reward`, `from_error_persistence`, `pause_on_reward`, `from_region_firing`, `from_action_specific_reward`
- **What they model:** internal rule-type strings driving NM concentration changes.
- **Catalog umbrella:** project framework; rough biological analogs:
  - `from_reward` ≈ phasic DA from RPE
  - `from_error_persistence` ≈ sustained tonic NE/DA from prediction error
  - `pause_on_reward` ≈ TAN pause response
- **Current correctness:** **match (functional)** — names are descriptive.
- **Rename priority:** none.

---

## F. Bridge state fields (selected)

### Field: `current_reward_signal` (CoreSimConfig)
- **What it represents:** scalar reward `r(t)`; gates eligibility-trace × STDP.
- **Catalog umbrella:** Cluster C — phasic DA scalar / RPE.
- **Canonical feature:** glossary "DA" + "RPE"; flagged as `[NEEDS-REVIEW] "DA" vs "current_reward_signal"`.
- **Current correctness:** **partial** — name describes the role (reward signal) accurately; conflation with phasic/tonic DA, A9/A10, Component-1/Component-2 is a separate biological concern (catalog `[discrepancy]`).
- **Suggested canonical name:** keep — the name is functionally accurate. Code comments should clarify that this is the project's "DA scalar" abstraction.
- **Rename priority:** low.
- **Catalog ref:** C.04, glossary `[NEEDS-REVIEW]`.

### Field: `reward_baseline` (CoreSimConfig)
- **What it represents:** expected reward baseline for prediction error.
- **Catalog umbrella:** Cluster C TD-error / RPE.
- **Canonical feature:** "Component 2" baseline / TD baseline.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Field: `reward_aversive_scale` (CoreSimConfig)
- **What it represents:** scaling factor for negative reward (asymmetric scaling per Schultz98/16).
- **Catalog umbrella:** C.32 two-component DA, asymmetric burst/dip.
- **Canonical feature:** "appetitive vs aversive DA asymmetry".
- **Current correctness:** **match**.
- **Rename priority:** none.

### Field: `cp_eligibility_trace`
- **What it represents:** synaptic eligibility trace.
- **Catalog umbrella:** C.29, J — three-factor learning.
- **Canonical:** "eligibility trace" — exact match.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Field: `cp_synapse_action_tag`
- **What it represents:** per-synapse action index (Cluster C v2 compartmentalized DA).
- **Catalog umbrella:** project framework for action-specific DA (T2.D).
- **Canonical:** project term.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Field: `cp_d1_d2_sign`
- **What it represents:** per-synapse sign for D1 vs D2 plasticity asymmetry (Cluster B.1).
- **Catalog umbrella:** B.03, O.03.
- **Canonical:** project term.
- **Current correctness:** **match**.
- **Rename priority:** none.

### Field: `cp_plasticity_gain` / `cp_plasticity_window_gate`
- **What they represent:** per-synapse multipliers gating STDP/eligibility/Hebbian (`cp_plasticity_gain`); per-synapse window gate driven by ACh/TANs (`cp_plasticity_window_gate`).
- **Catalog umbrella:** Cluster J plasticity rules; Cluster L critical periods.
- **Canonical:** project framework.
- **Current correctness:** **match**.
- **Concern:** `plasticity_gain` and `plasticity_window_gate` sound similar; their distinction (one is a real-valued multiplier on rate; the other is a binary-ish gate driven by the ACh/TAN system) is documented but easily confused. Code comments should disambiguate.
- **Rename priority:** **medium** — rename `plasticity_gain` to `plasticity_rate_gain` to differentiate from `plasticity_window_gate`.

### Field: `enable_brain_region_framework` / `enable_neuromodulator_subsystem`
- **What they enable:** opt-in for the region/neuromodulator subsystems.
- **Canonical:** project framework.
- **Current correctness:** **match**.
- **Rename priority:** none.

---

## Systematic issues spanning multiple identifiers

### Issue 1: Umbrella-name overclaim for project-specific features
**Pattern.** Several flags adopt anatomical umbrella names when the implementation captures only one feature beneath that umbrella:
- `--enable-pfc` / `pfc` region → only WM persistent activity, not whole PFC
- `--hippocampus` → only `place_cells` + `goal_cells`, not the trisynaptic loop
- `--landmarks` → sensor abstraction, not landmark-cell biology

**Fix.** Add specificity suffix: `--enable-pfc-wm`, `--enable-place-goal-readout`, `--enable-landmark-sensor`. Coordinated rename of the region + flag + gate triple.

**Affected identifiers:** `--enable-pfc`, `pfc` region, `pfc_pathways` gate, `pfc_internal` gate, `--hippocampus`, `place_cells`, `goal_cells`, `hippo_to_cortex` gate, `--landmarks`, `landmark_to_place` gate.

### Issue 2: `cortex_to_d1` gate gates D1, D2, AND patch
**Pattern.** Single gate string `"cortex_to_d1"` is applied to `cortex_X → str_D1_X`, `cortex_X → str_D2_X`, and `cortex_X → str_patch_X` pathways. Anyone reading curriculum logic that calls `bridge.set_plasticity_gate("cortex_to_d1", ...)` would reasonably assume only the direct pathway is affected; in fact all three paths are.

**Fix.** Rename to `corticostriatal` (covers all three) — the single rename addresses all three conflated uses. Update curriculum docstrings.

**Affected identifiers:** `cortex_to_d1` gate (3 pathway declarations), curriculum logic, plan documents.

### Issue 3: `*_FS_*` taxonomy conflation across cortical and striatal
**Pattern.** All `cortex_FS_*`, `str_FS_*`, `motor_FS_*`, `dg_fs` regions share the `IZH2007_FS_CORTICAL_INTERNEURON` preset, which is biologically a cortical PV+ basket cell. Striatal FSI (B.06) is a distinct cell-type family from cortical FSI (B.01); the shared preset is acknowledged in the runner comment as an engineering shortcut but the *names* don't reflect the distinction. Glossary explicitly flags this `[NEEDS-REVIEW]`.

**Fix.** Either:
- Add cell-class qualifier: `cortex_PV_basket_X`, `str_PV_FSI_X`, `dg_PV_basket`. (Preserves single-preset engineering shortcut but flags biology.)
- Or split presets and use class-specific names downstream. (Bigger change.)

**Affected identifiers:** `cortex_FS_X`, `str_FS_X`, `motor_FS_X`, `dg_fs`, plus the `IZH2007_FS_CORTICAL_INTERNEURON` preset (which is correctly named for cortical use but reused).

### Issue 4: Generic `dopamine` region collapses A9 + A10
**Pattern.** Single `dopamine` region collapses SNc (A9, motor / nigrostriatal) and VTA (A10, mesolimbic / mesocortical). Glossary flags this as a `[discrepancy]`.

**Fix.** Rename region to `snc` (since runner comment says "this is the project's A9-equivalent — SNc DA neurons"); keep transmitter modulator named `dopamine`. Add separate `vta` region when mesolimbic arc is built (T2.E amygdala / Cluster O).

**Affected identifiers:** `dopamine` region (only); the `dopamine` *modulator* name is correct.

### Issue 5: Pre-Cluster-D legacy regions (`place_cells`, `goal_cells`) clash with new canonical names
**Pattern.** The older `--hippocampus` flag uses non-canonical region names (`place_cells`, `goal_cells`); the new `--enable-cluster-d-hippocampus` uses canonical `dg / ca3 / ca1`. Both can be active simultaneously in flagship (place_cells receives ca1 output when both flags are on).

**Fix.** Either:
- Phase out `--hippocampus` legacy flag once Cluster D is fully integrated into flagship.
- Rename `place_cells` → `pseudo_place_cells` or `sensor_place_readout` to flag the sensor-driven origin.
- Rename `goal_cells` → `ppc_goal_X` (catalog explicitly says these are PPC-like, not hippocampal).

**Affected identifiers:** `--hippocampus`, `place_cells`, `goal_cells`, `hippo_to_cortex` gate.

### Issue 6: `*_FS_*` motor and `motor_FS_*` are project inventions
**Pattern.** The motor WTA microcircuit (`--motor-lateral-inhibition` + `motor_FS_X`) has no clean biological referent. Real motor-pool WTA is via spinal Renshaw cells / reciprocal inhibition (H.07–H.08), not via cortical-style FS basket cells.

**Fix.** Either retire the flag (NEGATIVE in evaluation) or rename to something that flags the abstraction (`motor_pool_wta`).

**Affected identifiers:** `--motor-lateral-inhibition`, `motor_FS_X` regions.

---

## Recommended renaming sequence

If a renaming pass were run in priority order, here are the top renames with concrete migration paths.

| # | Current | Suggested | Rationale | Migration |
|---|---|---|---|---|
| 1 | gate `"cortex_to_d1"` | `"corticostriatal"` | Gates D1+D2+patch; name implies D1-only | Rename in 3 pathway sites + curriculum API; add temporary alias in `set_plasticity_gate` for one release cycle |
| 2 | `--enable-pfc` / `pfc` region / `pfc_pathways` gate | `--enable-dlpfc-wm` / `dlpfc_wm` / `dlpfc_wm_pathways` | "PFC" overclaims; we model only WM persistent activity (G.06) | Coordinated triple rename; CLI alias for `--enable-pfc`; CLAUDE.md flagship recipe update; preset audit (`HH_PFC_PYRAMIDAL` could rename to `HH_DLPFC_PYRAMIDAL` later) |
| 3 | `--hippocampus` / `place_cells` / `goal_cells` / `hippo_to_cortex` gate | Phase out legacy flag; rename `place_cells` → `sensor_place_readout`, `goal_cells` → `ppc_goal_X`, gate → `place_goal_to_cortex` | Legacy flag is sensor-driven, not canonical hippocampus; `goal_cells` are PPC-like per glossary | Big rename; needs alias map for sidecar files; consider deprecation cycle |
| 4 | `--landmarks` flag | `--enable-landmark-sensor` | "Landmarks" overclaims D.09 landmark-cell biology; implementation is sensor abstraction | Single flag rename + alias |
| 5 | `dopamine` region | `snc` | "Dopamine" is the transmitter, not the region. Region is A9 (SNc) | Region rename; keep `dopamine` transmitter modulator name; small migration |
| 6 | `--bg-lateral-inhibition` | `--enable-msn-lateral-inhibition` | Implementation is MSN-MSN, not generic BG lateral inhibition; biology is FSI feedforward (B.04 supplemental flags discrepancy) | Single flag rename + alias |
| 7 | `--enable-striatal-fsis` / `str_FS_X` | `--enable-striatal-pv-fsi` / `str_PV_FSI_X` | "FS" doesn't disambiguate from other striatal interneuron classes (B.01 supplemental: 8 distinct classes) | Coordinated rename; small (opt-in flag) |
| 8 | `acetylcholine` modulator | `acetylcholine_tan` | Implementation is striatal-TAN-driven only; brain has multiple ACh sources | Modulator rename; one default config function |
| 9 | `cp_plasticity_gain` field | `cp_plasticity_rate_gain` | Confused with `cp_plasticity_window_gate`; one is a multiplier, the other a gate | Bridge field rename; update docstrings; small |
| 10 | `motor_FS_X` regions / `--motor-lateral-inhibition` | Either retire (NEGATIVE) or rename to `motor_pool_wta_X` / `--enable-motor-pool-wta` | No biological referent (real motor WTA is spinal); flag is NEGATIVE in evaluation | Optional — retire is cleaner |

**Lower priority (defer):**
- `cortex_X` → `m1_X` (defer until other cortex regions added)
- `thal_X` → `va_vl_X` (defer until other thalamic nuclei added)
- `motor_X` → `motor_output_X` (defer until muscle/spinal Tier 3)
- `gpe_X` → `gpe_proto_X` for symmetry with `gpe_arky_X` (low priority; alias works)
- `HH_PFC_PYRAMIDAL` preset → `HH_DLPFC_PYRAMIDAL` (defer due to saved-profile compatibility)

**Backward compatibility caveats:**
- Preset enum names appear in saved JSON profiles (`simulation_profiles/`). Any preset rename needs an alias map in `sim/enums.py`.
- Region names and gate strings appear in sidecar JSONs in `research/findings/raw/g11_bg/`. Aliases on the runner side preserve readability of historical results.
- CLI flags appear in CLAUDE.md flagship recipes, plan documents, and findings docs. CLI aliases (`--pfc → --dlpfc-wm`) preserve historical commands; documentation should use new names going forward.

---

## Genuinely ambiguous biology (catalog doesn't give a clean name)

The following are flagged as not having a clean canonical name from the catalog — additional research / catalog enrichment is needed before any rename.

1. **`goal_cells`** — biology has goal-vector / goal-direction cells in PPC, MEC (some interpretations), and prefrontal cortex; canonical naming depends on theoretical framing. D.09 "object-vector cells" is the closest single-named referent but doesn't capture the project's "goal as separate input" usage. **Suggestion:** `ppc_goal_X` if framing as PPC, `pfc_goal_input` if framing as PFC input. Decision required.

2. **`motor_X`** — abstract motor output pool. Without muscle / spinal CPGs, this is neither M1 (cortical) nor α-motoneuron (spinal). **Suggestion:** retain `motor_X` as a project abstraction; clarify in docstrings.

3. **`beacon_sensors`, `landmark_sensors`** — engineering sensor abstractions; no canonical name. Catalog Cluster K (sensory transduction) has biological referents but the project's beacons/landmarks are environmental cues, not sensory transducers. **Suggestion:** keep as project terminology; flag as abstractions in docstrings.

4. **`current_reward_signal`** — scalar `r(t)` driving plasticity; conflates phasic/tonic DA, A9/A10, Component-1/Component-2 (catalog `[discrepancy]`). The *name* is fine (it's a reward signal); the *biology* is partial. **Suggestion:** keep name; document the partial-biology disclaimer.

5. **`pfc_internal` gate** — gates PFC recurrent connectivity. Biology has multiple "internal" PFC plasticity rules (homosynaptic STDP, NMDA-dependent recurrent, neuromodulator-gated). **Suggestion:** if PFC region is renamed `dlpfc_wm`, gate becomes `dlpfc_wm_recurrent` to be specific.

6. **`stp_U` etc.** — STP parameters use Tsodyks-Markram convention (`U`, `tau_d`, `tau_f`); these are the canonical parameter names from the original paper, even though they're cryptic without the paper open. **Suggestion:** keep — adding biological-meaning comments next to the parameter is sufficient.

---

## Out-of-scope (deliberately unaddressed)

- **`g11_bg_runner.py` filename** — runner names follow `g{N}_*` for research-gate identifiers; this is project-organizational, not biological.
- **Test file names** — `tests/test_regions.py` etc. follow code structure, not biology.
- **HDF5 attribute keys in checkpoints** — preserved for compatibility; not part of the user-facing identifier surface.
- **Internal CSR-array structure naming** (`cp_connections`, `cp_synapse_*`) — implementation detail; not directly biological.
