# Rename master plan — structural identifier pass

**Date:** 2026-04-29
**Status:** DEFERRED — pending cheat-5 / biology buildout stopping point
**Inputs:** glossary.md (228 entries), 6 prose surveys (A/B/C/D/E/F), structural-naming-audit.md (86 IDs across 6 categories), reference-coverage-audit.md
**Reference library:** `E:\Documents\Projects\sim-catalog\references\textbooks\` (16 PDFs, all locally readable)

---

## Executive summary

This plan is the authoritative input for the post-cheat-5 structural-rename pass.
It folds together one glossary, six prose surveys, one structural-naming audit
(86 identifiers categorized), and one reference-coverage audit into a single
ordered work list. After cheat-5 / biology buildout reaches a stopping point,
the rename executor should be able to follow this document top-to-bottom
without re-reading the eight inputs.

**In scope (Tier 3 structural identifiers):**
- Region names (`pfc`, `place_cells`, `goal_cells`, `dopamine`, `cortex_X`, `motor_FS_X`, etc.)
- Pathway / plasticity-gate strings (`cortex_to_d1`, `pfc_pathways`, `bg_cross_projections`, etc.)
- CLI flags (`--enable-pfc`, `--hippocampus`, `--landmarks`, `--bg-lateral-inhibition`, etc.)
- Neuromodulator names where they overclaim biology (`acetylcholine` for the
  TAN-only arm)
- Bridge-state field names where they confuse readers
  (`cp_plasticity_gain` vs `cp_plasticity_window_gate`)
- Glossary `[NEEDS-REVIEW]` items where the audit decision is "rename, with
  alias for compat"

**Out of scope (already done or further-deferred):**
- Tier 1+2 prose fixes — DONE 2026-04-29 at commit `8aa2fcb` (15 high-impact prose
  fixes: STDP docstring direction, Bi & Poo 2001→1998, Q10 tooltip, D1/D2 MSN
  shorthand, "striatum_D1" placeholder → "str_D1_X", "Connections" log →
  "Synapses", "trisynaptic loop" → "trisynaptic pathway" in active plans, etc.)
- HH preset enum-name renames (saved-profile JSONs in `simulation_profiles/`
  reference these by exact string) — defer to a separate config-versioning task
  with explicit alias map
- Behavioral parameter changes masquerading as renames (theta-band 4-8 vs 4-12 Hz
  in `experiment/readout.py:222` is a behavioral choice, not a rename)
- `current_reward_signal` rename — explicitly exempted by glossary `[NEEDS-REVIEW]`
  per surveys A/C/F; documented simplification, not misleading
- CSR array name `cp_connections` — the underlying data structure is fine;
  surveys recommend fixing only the user-facing prose ("connections" → "synapses"),
  which is Tier 1 and already done

**Recommended sequencing:**

The rename pass proceeds in three waves, ordered to maximize signal-per-effort
and minimize blast radius compounding:

1. **Wave 1 (high priority, ~13 renames):** the "actively misleading" set —
   rename everything that makes a reader form an incorrect mental model.
   Tightest blast radius: pathway gates and CLI flags that appear in ≤30
   places. Each rename is an independent commit with an alias map preserving
   backward compat for one release cycle.

2. **Wave 2 (medium priority, ~13 renames):** the "underspecified but not wrong"
   set — names that work in current context but will break when the catalog's
   missing structures (Cluster A arkypallidal split, Cluster F cerebellum,
   Cluster O amygdala/NAc) are added. Most of Wave 2 can be deferred until
   those features ship; rename when they do.

3. **Wave 3 (low / deferred-further, ~10 renames):** saved-profile-coupled
   identifiers (HH/Izh/AdEx preset enum names) and project abstractions with
   no clean biological referent (`goal_cells`, `motor_X`, `beacon_sensors`).
   These need either an alias-map infrastructure pass *or* a human policy
   decision; do not rename them in the same session as Waves 1-2.

**Acid test:** at the end of the rename pass, a 6-seed cheat-5 baseline must
reproduce the documented 4.08 ± 0.49 with no regression. Functional behavior
must be unchanged; only names and aliases change.

---

## Top-priority renames (Wave 1 — do first)

These are the "actively misleading" set. Each rename's mismatch causes
readers to form an incorrect mental model of either *what* is being modeled
or *what* a function call will do. Fix order is by independent blast radius
(small first), so failures don't compound.

### 1. Gate `"cortex_to_d1"` → `"corticostriatal"`
- **Current name** (`research/runners/g11_bg_runner.py:793, 802-803, 807-812, 962`): the gate string `"cortex_to_d1"` is applied to **three** pathways: `cortex_X → str_D1_X`, `cortex_X → str_D2_X`, AND `cortex_X → str_patch_X`.
- **Proposed canonical name:** `"corticostriatal"`
- **What it actually models:** plasticity gating across the entire corticostriatal projection (D1 + D2 + patch arms).
- **Why current is wrong:** name claims D1-only; gates D1, D2, AND patch. Anyone reading `bridge.set_plasticity_gate("cortex_to_d1", 0.0)` in curriculum logic will reasonably assume only the direct pathway is being frozen. In fact all three are.
- **Blast radius:** small/medium — 3 pathway sites in g11_bg_runner + curriculum freeze/thaw API + ~5 plan/findings doc references. ~8-10 places total.
- **Backward-compat strategy:** add a temporary alias in `bridge.set_plasticity_gate()`: `"cortex_to_d1"` → `"corticostriatal"` (warn-on-use for one release cycle).
- **Verification PDF:** none needed — pathway-gate naming is project-internal infrastructure, not biological. Cluster J.30 (corticostriatal STDP) confirms canonical name is "corticostriatal" — see `sim-catalog/references/textbooks/kandel-pns-6e/full-book.pdf` Ch 38 if questioned.
- **Glossary citation:** J.30, A.01, A.02 (direct + indirect pathways).

### 2. Region `pfc` + flag `--enable-pfc` + gate `pfc_pathways` (coordinated triple)
- **Current names** (`research/runners/g11_bg_runner.py` various, `CLAUDE.md` flagship recipe, `webapp/server.py:241-278` preset list): `pfc` region (60 recurrent neurons), `--enable-pfc` flag, `pfc_pathways` plasticity gate.
- **Proposed canonical names:** `dlpfc_wm` region, `--enable-dlpfc-wm` flag, `dlpfc_wm_pathways` (or split into `goal_to_dlpfc` + `dlpfc_to_m1`) gate.
- **What it actually models:** a single recurrent attractor implementing dlPFC working-memory persistent activity (catalog G.06 / G.08). Not vmPFC, not OFC, not ACC, not goal-encoding.
- **Why current is wrong:** "PFC" is the entire prefrontal cortex (dlPFC + vmPFC + OFC + ACC). The implementation captures one feature (working memory) of one subregion (dlPFC). Glossary `[NEEDS-REVIEW] "PFC" subdivisions` (entry §"Prefrontal cortex (PFC)") explicitly flags this.
- **Blast radius:** **large** — flagship recipe in CLAUDE.md (line ~498), webapp preset list, sidecar JSONs in `research/findings/raw/g11_bg/`, ~15 findings docs reference `--pfc`. Coordinated rename of region + flag + gate across runner, webapp, plans, and findings.
- **Backward-compat strategy:**
  - CLI: `--enable-pfc` → `--enable-dlpfc-wm` with `--pfc` as deprecated alias for one release.
  - Region: rename in g11_bg_runner.py; alias map in any code that reads region names from sidecars.
  - Gate: rename + alias in `set_plasticity_gate()`.
  - Preset: defer `HH_PFC_PYRAMIDAL` → `HH_DLPFC_PYRAMIDAL` to Wave 3 (saved-profile coupled).
- **Verification PDF:** Kandel 6e Ch 52-56 (prefrontal cortex). `sim-catalog/references/textbooks/kandel-pns-6e/full-book.pdf`. Cluster G is Kandel-self-contained per reference-coverage-audit; no specialty PDF re-read required.
- **Glossary citation:** §"Prefrontal cortex (PFC)" (catalog G.06, G.08).

### 3. Region `dopamine` → `snc`
- **Current name** (`research/runners/g11_bg_runner.py:551-562` — single shared pool; `dopamine_{N,E,S,W}` per-action variants exist with `--enable-compartmentalized-da`): `dopamine` region.
- **Proposed canonical name:** `snc` (region); the *neuromodulator* `dopamine` keeps that name (DA is the transmitter — correct).
- **What it actually models:** the runner comment explicitly says "this is the project's A9-equivalent — SNc dopaminergic neurons". A9 = SNc per Cluster C.16.
- **Why current is wrong:** "dopamine" is a transmitter name being used as a region name. Biologically the *region* is SNc (A9, motor / nigrostriatal) + VTA (A10, mesolimbic / mesocortical) collapsed; the implementation matches A9 broadcast behavior. Glossary §SNc and §VTA flag the collapse as `[discrepancy]` per C.16.
- **Blast radius:** moderate — `dopamine` region appears in pathway declarations (~10 sites), default neuromodulator helper functions, several findings docs, sidecar logs.
- **Backward-compat strategy:** rename region; keep neuromodulator named `dopamine` (transmitter is correct). Sidecar logs containing `"region": "dopamine"` need a translation step on load. When VTA / mesolimbic arc is added (Tier 2 buildout, not yet started), a separate `vta` region can be added cleanly.
- **Verification PDF:** Schultz papers in `sim-catalog/references/textbooks/schultz-dopamine/` (Schultz 1998, Schultz 2016 NRN). Glossary entry §SNc explicitly cites C.16 `[discrepancy]`.
- **Glossary citation:** §"Substantia nigra pars compacta (SNc)", §"Ventral tegmental area (VTA)" — both flag A9/A10 collapse.

### 4. Flag `--hippocampus` → `--enable-place-goal-readout` (or phase out entirely)
- **Current name** (`research/runners/g11_bg_runner.py` ~236-260, `CLAUDE.md` flagship recipe): `--hippocampus` flag adds `place_cells` + `goal_cells` regions with sparse Gaussian tuning; sensor-driven, not the trisynaptic loop.
- **Proposed canonical name:** `--enable-place-goal-readout` OR phase out the legacy flag entirely once Cluster D (`--enable-cluster-d-hippocampus`) ships in flagship.
- **What it actually models:** two non-canonical regions (place_cells, goal_cells) with sensor-driven activations. Per glossary, place_cells are not strictly allocentric per O'Keefe & Nadel 1978 criteria; goal_cells are closer to PPC than hippocampus.
- **Why current is wrong:** "--hippocampus" claims the whole structure (DG + CA3 + CA1 + EC + subiculum at minimum). The implementation is two abstract sensor-driven readout pools.
- **Blast radius:** **large** — flagship recipe in CLAUDE.md, USER_GUIDE.md, README.md (~3 mentions each), ~8 findings docs.
- **Backward-compat strategy:** CLI alias `--hippocampus` → `--enable-place-goal-readout` for one release. Coordinate with rename #5 (place_cells) and #6 (goal_cells) — same triple.
- **Verification PDF:** `sim-catalog/references/textbooks/okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.pdf` Ch 4.7 pp 190-217 (allocentric criterion). Glossary §"Place cell" cites this directly.
- **Glossary citation:** §"Place cell" `[NEEDS-REVIEW]`, §"Hippocampus subregions" notes both legacy and Cluster D forms.
- **Decision needed (human policy):** retire `--hippocampus` entirely once Cluster D enters flagship, OR keep both forms with the new name. **Recommendation:** rename + alias, document deprecation, retire when Cluster D becomes the flagship default for sequence-aware learning (per SCIENCE_ROADMAP §552).

### 5. Region `place_cells` → `sensor_place_readout` (or `ca1_place_cells` — see Wave 3)
- **Current name:** `place_cells` region (paired with `--hippocampus` flag).
- **Proposed canonical name (Wave 1):** `sensor_place_readout` — flags the sensor-driven origin without claiming canonical place-cell biology.
- **Proposed canonical name (Wave 3, requires policy):** `ca1_place_cells` — but this collides with the new Cluster D `ca1` region.
- **What it actually models:** sparse Gaussian-tuned cells encoding agent (x, y); plastic to all 4 cortex pools.
- **Why current is wrong:** "place_cells" implies allocentric tuning per O'Keefe & Nadel 1978; the implementation is sensor-driven. Catalog D.06 supplemental: a true allocentric place cell should still fire on subsequent traversals after some sensor cues are removed — the project's place_cells fail this test by construction.
- **Blast radius:** moderate — `--hippocampus` is opt-in legacy; `place_cells` region appears in 8-10 pathway declarations, sleep-replay logs, several findings docs.
- **Backward-compat strategy:** rename region; in pathway declarations + sidecar logs use the new name. Alias not needed at the CLI layer (the flag itself is renamed in #4).
- **Verification PDF:** `sim-catalog/references/textbooks/okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.pdf` Ch 4 (allocentric criterion, definition of place cell).
- **Glossary citation:** §"Place cell" `[NEEDS-REVIEW]`.

### 6. Region `goal_cells` → `ppc_goal_X` (per-action) or `ppc_goal_input`
- **Current name:** `goal_cells` region (paired with `--hippocampus` flag; also feeds into `pfc` when both flags on).
- **Proposed canonical name:** `ppc_goal_X` (per-action variants if matching the action structure) OR `ppc_goal_input` (single pool).
- **What it actually models:** sparse Gaussian-tuned cells encoding goal (gx, gy); plastic to cortex_X and PFC.
- **Why current is wrong:** "goal_cells" invokes hippocampal place-cell-system semantics, but goal-encoding is anatomically PPC-like (or PFC for goal-context). Glossary §"Posterior parietal cortex (PPC)" explicitly says: "missing as a region; `goal_cells` region in g11 is closer to PPC than PFC despite naming." The fact that the new `--pfc` flag wires `goal_cells → pfc → cortex` underscores this — goal_cells are upstream of PFC, suggesting they belong in PPC.
- **Blast radius:** moderate — pathway declarations, the `--goal-silence` flag, multiple PFC-related findings, 5+ plan documents.
- **Backward-compat strategy:** rename region; alias in any sidecar reader.
- **Verification PDF:** Kandel 6e Ch 17-29 (parietal cortex), no specialty PDF needed.
- **Glossary citation:** §"Posterior parietal cortex (PPC)"; D.09 "object-vector cells" is the closest single-named referent if you want to frame as MEC-side.
- **Genuine ambiguity (see "non-acceptance" section):** goal_cells has no clean biological analogue. Catalog D.09 "object-vector cells" is closest if framing as allocentric goal-vector encoders, but the project usage is "goal as separate input" — closer to PPC. **Decision required:** is the canonical framing PPC (parietal goal-direction signal) or MEC (object-vector cell)? Recommendation: PPC.

### 7. Flag `--landmarks` → `--enable-landmark-sensor`
- **Current name:** `--landmarks` flag adds `landmark_sensors` region + `landmark_sensors → place_cells` pathway.
- **Proposed canonical name:** `--enable-landmark-sensor` (descriptive of the sensor abstraction).
- **What it actually models:** 8 directional sensors tied to a fixed-position landmark; activation depends on (distance, bearing) to landmark. Sensor abstraction, not landmark-cell biology.
- **Why current is wrong:** "--landmarks" invokes a whole class of cells (landmark cells, object-vector cells per D.09), but the implementation is a sensor abstraction. The downstream region is rightly named `landmark_sensors`; the flag should match.
- **Blast radius:** small — opt-in flag, ~5 references.
- **Backward-compat strategy:** CLI alias `--landmarks` → `--enable-landmark-sensor` for one release.
- **Verification PDF:** Kandel 6e Ch 53 (parahippocampal / hippocampal navigation). No specialty PDF needed.
- **Glossary citation:** D.09 "object-vector cell"; project's K (sensory transduction).

### 8. Flag `--bg-lateral-inhibition` → `--enable-msn-lateral-inhibition`
- **Current name:** `--bg-lateral-inhibition` adds `str_D1_X → str_D1_Y` and `str_D2_X → str_D2_Y` cross-pool inhibition for X≠Y. Permanent default in flagship.
- **Proposed canonical name:** `--enable-msn-lateral-inhibition`.
- **What it actually models:** specifically *MSN* lateral inhibition (cross-pool D1↔D1, D2↔D2). Catalog B.04 supplemental flags that this is anatomically backwards: real cross-pool WTA in striatum is FSI feedforward, not MSN-MSN feedback (Wilson 2007 PBR-160 ch 6).
- **Why current is wrong:** "BG lateral inhibition" is broad; could imply pallidal lateral inhibition or thalamic lateral inhibition. The implementation is specifically MSN-MSN, and the biology is documented as imperfect (FSI is the real substrate).
- **Blast radius:** moderate — flagship recipe in CLAUDE.md (line ~498), several findings docs, webapp preset.
- **Backward-compat strategy:** CLI alias `--bg-lateral-inhibition` → `--enable-msn-lateral-inhibition` for one release.
- **Verification PDF:** `sim-catalog/references/textbooks/basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf` ch 6 (Wilson — MSN lateral inhibition vs FSI feedforward).
- **Glossary citation:** B.04, B.04 supplemental.

### 9. Flag `--enable-striatal-fsis` + region `str_FS_X` → `--enable-striatal-pv-fsi` + `str_PV_FSI_X`
- **Current names:** `--enable-striatal-fsis` flag, `str_FS_X` regions. Uses `IZH2007_FS_CORTICAL_INTERNEURON` preset (engineering shortcut).
- **Proposed canonical names:** `--enable-striatal-pv-fsi` flag, `str_PV_FSI_X` regions (or `str_pv_fsi_X` for case-consistency with existing `str_d1`/`str_d2` lowercase style — see policy decision below).
- **What it actually models:** PV+ parvalbumin-fast-spiking interneurons specifically — one of EIGHT distinct striatal GABAergic classes catalogued in Tepper-2018 (the others: NPY-LTS, NPY-NGF, CR, TH/THIN, FAI, SABI, ChI/TAN). The runner already has an exemplary comment at lines 425-433 acknowledging this; the *name* doesn't reflect the comment.
- **Why current is wrong:** "FS" is the electrophysiology class; "FSI" suggests the only striatal interneuron, eliding 7 other classes. When future work adds NPY-LTS or other classes, `str_FS_X` will need to be disambiguated anyway.
- **Blast radius:** small — `--enable-striatal-fsis` is opt-in, `str_FS_X` regions only present when flag is set.
- **Backward-compat strategy:** CLI alias + region-name alias for one release.
- **Verification PDF:** `sim-catalog/references/textbooks/basal-ganglia-reviews/Tepper-2018-StriatalGABAergic-Heterogeneity.pdf` (covers 8-class taxonomy); `sim-catalog/references/textbooks/basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf` pp 157-158, 174.
- **Glossary citation:** B.06 "PV-FSI", B.01 supplemental (8-class taxonomy).
- **Policy decision:** case style. The existing `str_D1_X` uses uppercase D1; `str_FS_X` uses uppercase FS. Proposed `str_PV_FSI_X` follows that precedent. Alternative: `str_pv_fsi_X` (lowercase) — more typical of project shorthand. **Recommendation:** keep uppercase (matches current style).

### 10. Modulator `acetylcholine` → `acetylcholine_tan`
- **Current name** (`sim/neuromodulators.py` `_default_acetylcholine_config`, `tests/test_tans.py`): the `acetylcholine` neuromodulator implements TAN-pause-on-reward → opens corticostriatal plasticity window.
- **Proposed canonical name:** `acetylcholine_tan` (specifies that this is the striatal-TAN-driven population, not the basal-forebrain or brainstem ACh source).
- **What it actually models:** ACh from striatal cholinergic interneurons (TANs / ChIs) only. The brain has multiple ACh sources: basal forebrain (Ch1-Ch4), brainstem (PPN/LDT) — none of which are modeled.
- **Why current is wrong:** "acetylcholine" claims the whole transmitter; we model one source (striatal TAN). When basal-forebrain ACh is added (future Cluster C/N work), a separate modulator will be needed; the current name forecloses that namespace.
- **Blast radius:** small/medium — modulator name appears in `_default_acetylcholine_config()`, `--enable-tans` registration in g11_bg_runner.py, ~3 test files, `_default_substance_p_config()` cross-reference (substance P boosts ACh per B.05).
- **Backward-compat strategy:** rename in `sim/neuromodulators.py`; add deprecated alias in `_default_acetylcholine_config()` factory.
- **Verification PDF:** Kandel 6e Ch 38, 41 (ACh sources). `sim-catalog/references/textbooks/basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf` pp 167, 171-172 (TAN biology, B.05 disynaptic).
- **Glossary citation:** §"TAN / ChI" (B.05); §"Acetylcholine" (C.18).

### 11. Region `motor_FS_X` + flag `--motor-lateral-inhibition` → retire OR rename to `motor_wta_FS_X` / `--enable-motor-pool-wta`
- **Current names:** `motor_FS_X` regions (FS interneuron pool per motor pool), `--motor-lateral-inhibition` flag. Both NEGATIVE in 2026-04-26 evaluation.
- **Proposed canonical names:** `motor_wta_FS_X` / `--enable-motor-pool-wta` (descriptive, flags the abstraction) — OR retire entirely.
- **What it actually models:** project invention — no clean biological referent. Real motor-pool WTA is via spinal Renshaw cells / reciprocal inhibition (H.07-H.08), not via cortical-style FS basket cells.
- **Why current is wrong:** name implies cortical-FS-like inhibition; biology is fundamentally different (spinal Renshaw is the canonical motor-WTA substrate).
- **Blast radius:** small — opt-in flag, NEGATIVE in evaluation.
- **Backward-compat strategy:** **recommend retiring** since the flag is NEGATIVE. If kept for archival reproducibility, rename with descriptive prefix.
- **Verification PDF:** Kandel 6e Ch 35 (spinal motor microcircuit, H.07-H.08).
- **Glossary citation:** §"Renshaw cell" (H.08), §"α-motoneuron, γ-motoneuron".
- **Policy decision:** retire vs rename. **Recommendation:** retire the flag; document in CHANGELOG that any future motor-WTA work should explicitly model spinal Renshaw inhibition, not a cortical-FS abstraction.

### 12. Bridge field `cp_plasticity_gain` → `cp_plasticity_rate_gain`
- **Current name:** `cp_plasticity_gain` per-synapse multiplier on STDP/eligibility/Hebbian/synaptic-scaling rates.
- **Proposed canonical name:** `cp_plasticity_rate_gain` (specifies that this is a *rate* multiplier, distinct from the binary-ish `cp_plasticity_window_gate`).
- **What it actually models:** real-valued multiplier on plasticity rate (gates STDP updates, eligibility traces, Hebbian rules, synaptic scaling). NOT to be confused with `cp_plasticity_window_gate` (binary window gate driven by ACh/TANs).
- **Why current is wrong:** "plasticity_gain" sounds nearly identical to "plasticity_window_gate" — readers easily confuse the two. The distinction matters: one is a continuous rate scaler set by curriculum; the other is a binary on/off window driven by the neuromodulator system.
- **Blast radius:** moderate — bridge state field, used in 4 plasticity kernels (`fused_stdp_weight_update`, eligibility, Hebbian, synaptic scaling), test files, plan docs.
- **Backward-compat strategy:** rename in bridge.py; if any sidecar/checkpoint reads the field name, add alias. (Most checkpoint code uses positional / dict-key access; rename is local to bridge.)
- **Verification PDF:** none — purely internal infrastructure naming.
- **Glossary citation:** Cluster J plasticity rules; project framework.

### 13. Flag `--enable-pfc` (paired with rename #2 above)
- See item #2 — the CLI flag is part of the `--enable-pfc` / `pfc` / `pfc_pathways` triple.

---

## Medium-priority renames (Wave 2 — do next, paired with biology buildout)

These are "underspecified but not actively wrong" — they work in current
context but will need rename when the missing biology is added. Defer most
until then.

### 14. Region `gpe_X` → `gpe_proto_X` (paired with existing `gpe_arky_X`)
- **Current name:** `gpe_X` (single pool — PV+ prototypic by convention; runner comment notes this).
- **Proposed canonical name:** `gpe_proto_X` (or `gpe_pv_X`) — for symmetry with `gpe_arky_X`.
- **What it actually models:** PV+ prototypic GPe neurons per action channel; projects to STN, GPi (R3.7 split).
- **Blast radius:** moderate — many pathway declarations, findings docs.
- **Backward-compat strategy:** alias map. Glossary already accepts both.
- **Verification PDF:** `sim-catalog/references/textbooks/basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf` ch 7 (Kita pp 111-114; PV+ prototypic vs PV- arkypallidal split).
- **Glossary citation:** A.13.
- **Defer until:** GPe arkypallidal work generates enough findings that the unqualified `gpe_X` becomes ambiguous in plan docs. Until then, the runner comment "prototypic by convention" is sufficient.

### 15. Region `cortex_X` → `m1_X` (or `cortex_m1_X`)
- **Current name:** `cortex_X` per-action motor-cortex (M1-equivalent) pools.
- **Proposed canonical name:** `m1_X` or `cortex_m1_X`.
- **What it actually models:** M1 layer-5 RS pyramidal pool per action channel. Runner comment at line 343-345 already acknowledges this.
- **Blast radius:** **very large** — 32 pathway declarations, every action runner, every test, every sidecar JSON, dozens of findings docs.
- **Backward-compat strategy:** would need exhaustive alias map at sidecar reader.
- **Defer until:** other cortex regions (PMd, pre-SMA from G.07; S1 from E.20) are added, forcing disambiguation.

### 16. Region `thal_X` → `va_vl_X` or `thal_motor_X`
- **Current name:** `thal_X` per-action thalamic relay pools (motor thalamus, VL/VA).
- **Proposed canonical name:** `va_vl_X` or `thal_motor_X`.
- **Blast radius:** large — pathway declarations, findings docs.
- **Defer until:** other thalamic nuclei (TRN, intralaminar/parafascicular) are added.

### 17. Region `motor_X` → `motor_output_X` (project-abstraction tag)
- **Current name:** `motor_X` — abstract motor output pool (uses `IZH2007_RS_CORTICAL_PYRAMIDAL`, NOT spinal motoneuron).
- **Proposed canonical name:** `motor_output_X` (project abstraction).
- **What it actually models:** behavioral abstraction; not M1, not α-motoneuron.
- **Blast radius:** very large — every runner, every findings doc.
- **Defer until:** muscle/spinal-CPG work begins (Tier 3 T3.C). At that point: rename + add `alpha_mn_X` / `gamma_mn_X` for biological motoneurons.

### 18. Region `sensory` → `pos_sensory` or `sensory_pos_grid` or `s1_pos`
- **Current name:** `sensory` (49-cell position-tuned grid; plastic `sensory → cortex_X` pathway).
- **Proposed canonical name:** `pos_sensory` (descriptive) or `s1_pos` (claims S1 analog).
- **Blast radius:** small — `--learned-perception` is opt-in.
- **Defer until:** other sensory-cortex regions (V1, A1) are added.

### 19. Gate `bg_cross_projections` → `corticostriatal_cross` or `cortex_to_msn_cross`
- **Current name:** opt-in cortex_X → str_D1_Y / str_D2_Y plasticity gate name.
- **Proposed canonical name:** `corticostriatal_cross` (more specific — the implementation is cortex→striatum cross-action, not BG-internal cross).
- **Blast radius:** moderate — `--bg-cross-projections` is opt-in NEGATIVE in current eval; if kept, rename.
- **Defer until:** cheat-5 cross-projection work resumes (currently ON HOLD pending biology buildout per CLAUDE.md).

### 20. Gate `hippo_to_cortex` → `place_goal_to_cortex` or `hippocampo_cortical`
- **Current name:** plasticity gate for `place_cells → cortex_X` and `goal_cells → cortex_X`.
- **Proposed canonical name:** `place_goal_to_cortex` (more accurate per renames #5 + #6) or `hippocampo_cortical` (formal).
- **Blast radius:** moderate.
- **Backward-compat strategy:** alias.
- **Pair with:** renames #4, #5, #6 (the `--hippocampus` triple).

### 21. Gate `pfc_internal` → `dlpfc_wm_recurrent`
- **Current name:** PFC recurrent-connection plasticity gate.
- **Proposed canonical name:** `dlpfc_wm_recurrent` (paired with rename #2).
- **Blast radius:** small — gate is currently reserved (`internal_density>0` not yet used in flagship).
- **Pair with:** rename #2.

### 22. Region `dg_fs` → `dg_pv` or `dg_pv_basket`
- **Current name:** DG fast-spiking interneuron pool (FFi for DG sparsity).
- **Proposed canonical name:** `dg_pv` or `dg_pv_basket` (canonical: DG basket cells are PV+).
- **Blast radius:** small — Cluster D opt-in.
- **Backward-compat strategy:** alias.
- **Verification PDF:** Kandel 6e Ch 54 (hippocampal interneurons).
- **Glossary citation:** §"Hippocampus subregions" + B (PV+ basket nomenclature).

### 23. Region `cortex_FS_X` → `cortex_pv_basket_X`
- **Current name:** cortical fast-spiking interneuron pool per action.
- **Proposed canonical name:** `cortex_pv_basket_X`.
- **Blast radius:** small — `--cortex-wta` is opt-in.
- **Defer until:** chandelier / Martinotti / NGF cortical interneurons are added (B.01 expansion).

### 24. Flag `--cortex-wta` → `--enable-m1-pv-basket` (or `--enable-cortex-pv-fsi`)
- **Current name:** "WTA" (winner-take-all) is computational; biology is "perisomatic basket inhibition driving gamma" (PING).
- **Proposed canonical name:** `--enable-m1-pv-basket` (biology-grounded) or `--enable-cortex-pv-fsi` (consistent with rename #9 naming pattern).
- **Blast radius:** small — opt-in flag.
- **Defer until:** rename #9 (`str_FS` → `str_PV_FSI`) lands and naming convention is established.

### 25. Flag `--enable-bg-neuropeptides` → `--enable-msn-co-release` or `--enable-d1-d2-peptides`
- **Current name:** registers `dynorphin`, `substance_p`, `enkephalin` modulators.
- **Proposed canonical name:** `--enable-msn-co-release` (more precise — D1 co-releases dynorphin + substance P, D2 co-releases enkephalin) or `--enable-d1-d2-peptides`.
- **Blast radius:** small — opt-in flag.
- **Defer until:** lower-priority polish pass; current name is acceptable.

### 26. CLI `--enable-cluster-d-hippocampus` → `--enable-trisynaptic-loop` (or keep)
- **Current name:** flag adds the canonical Cluster D trisynaptic loop.
- **Proposed canonical name:** `--enable-trisynaptic-loop` (named for the actual circuit added).
- **Blast radius:** small — opt-in flag.
- **Backward-compat strategy:** alias.
- **Note:** glossary uses "trisynaptic pathway" not "trisynaptic loop"; the prose Tier 1 fix (commit `8aa2fcb`) updated active plans. Flag rename should match: `--enable-trisynaptic-pathway`. **Recommendation:** keep `--enable-cluster-d-hippocampus` (flag name describes scope clearly); rename only if flag-name proliferation becomes a usability issue.

---

## Low-priority / deferred-further (Wave 3)

These have either huge blast radius (saved-profile JSON files), genuine
biological ambiguity, or both. Do not bundle with Waves 1-2.

### 27. Preset `HH_PFC_PYRAMIDAL` → `HH_DLPFC_PYRAMIDAL`
- **Current name:** HH preset for PFC pyramidal neurons.
- **Why deferred:** preset enum names appear in saved JSON profiles in `simulation_profiles/`. Any rename needs a *full* alias-map infrastructure pass.
- **Pair with:** rename #2, but only after a config-versioning task.

### 28. Preset `IZH2007_HIPPO_PYRAMIDAL` → split `IZH2007_CA1_PYRAMIDAL` / `IZH2007_CA3_PYRAMIDAL`
- **Current name:** generic CA1/CA3 pyramidal Izh preset.
- **Why deferred:** HH already has split presets (`HH_CA1_PYRAMIDAL_BURST`, `HH_CA3_PYRAMIDAL_BURST`); Izh-side migration would benefit but breaks saved profiles.
- **Verification PDF:** Kandel 6e Ch 54.
- **Glossary citation:** §"Hippocampal pyramidal neuron".

### 29. Preset `IZH2007_DOPAMINE` / `HH_DOPAMINE_SNC` → `IZH2007_SNC_VTA` / `HH_DA_SNC_VTA`
- **Current names:** Izh and HH variants for SNc/VTA DA neurons.
- **Why deferred:** saved-profile coupled. Glossary `[NEEDS-REVIEW]` accepts current naming; rename when VTA region is added.

### 30. Region `ec` → `ec_ii` / `ec_iii` (layer-specific)
- **Current name:** generic EC stub (single pool).
- **Why deferred:** project models a single EC pool; biology has layer II / III / deep distinctions. Only relevant if layer-specific pathways are added (D.04 temporoammonic vs perforant path).

### 31. Region `str_patch_X` → `str_striosome_X` (or keep `patch`)
- **Current name:** striosomal MSN compartment per action.
- **Why deferred:** glossary explicitly accepts both "patch" and "striosome" — modern literature uses both. Cosmetic rename only.

### 32. Region `stn` → `stn_burst` (or keep generic)
- **Current name:** subthalamic nucleus, single shared pool.
- **Why deferred:** STN is anatomically a single small nucleus; current name is correct. Rename only if dual-mode rebound dynamics (Cav3 short, Cav1.2/1.3 long per A.16) are explicitly modeled.

### 33. Profile keys `BASAL_GANGLIA_STRIATUM`, `HIPPOCAMPUS_CA1_RS_FS` → ?
- **Current names:** `NEURAL_STRUCTURE_PROFILES` dict keys; reuse cortical RS+FS structure for non-cortical regions.
- **Why deferred:** keys appear in saved JSON profiles in `simulation_profiles/` (47 files). Renaming requires alias map. Survey-A:470-479 explicitly recommends keep + comment.

### 34. Region `beacon_sensors`, `landmark_sensors` (keep — project abstractions)
- **Status:** keep as project terminology. No clean biological referent (these are environmental cues, not sensory transducers).
- **Defer further:** retire if/when real Cluster K sensory transduction is added.

### 35. CSR array `cp_connections` → `cp_synapses`
- **Status:** project's CSR sparse-matrix data structure. Surveys explicitly recommend leaving as identifier-shorthand; only fix prose ("connections" → "synapses" in user-facing strings, which is Tier 1 and DONE).
- **Defer further:** never; current naming is fine for the data structure.

### 36. Bridge field `current_reward_signal` (keep — explicitly exempted)
- **Status:** glossary `[NEEDS-REVIEW]` explicitly says auditor should not flag every use. Current naming is functional ("it's a reward signal"); biological-conflation issue is a separate documentation concern, not a rename.
- **Defer further:** never. Add documentation only.

---

## Cross-cutting structural patterns

These are systematic naming issues that span multiple individual renames.
The fix sequencing below ensures coordinated renames across the family.

### Pattern P1: Umbrella overclaim
Multiple flags adopt anatomical umbrella names when the implementation
captures one feature beneath that umbrella.

**Instances:**
- `--enable-pfc` / `pfc` region → only WM persistent activity (rename #2)
- `--hippocampus` → only `place_cells` + `goal_cells` (rename #4)
- `--landmarks` → sensor abstraction, not landmark-cell biology (rename #7)
- `--bg-lateral-inhibition` → MSN-MSN only (rename #8)

**Consistent fix-pattern:** add specificity suffix that names *what* is implemented:
- `--enable-pfc` → `--enable-dlpfc-wm` (the feature)
- `--hippocampus` → `--enable-place-goal-readout` (the abstraction)
- `--landmarks` → `--enable-landmark-sensor` (the sensor)
- `--bg-lateral-inhibition` → `--enable-msn-lateral-inhibition` (the substrate)

**Handling:** address all four in Wave 1; coordinated commit per family (flag + region + gate triples).

### Pattern P2: FS taxonomy conflation
All `*_FS_*` regions share the `IZH2007_FS_CORTICAL_INTERNEURON` preset (a
cortical PV+ basket-cell analog). Striatal PV-FSI is a distinct cell-type
family (Tepper-2018 8-class taxonomy); motor `motor_FS_X` is a project
invention with no clean biological referent.

**Instances:**
- `cortex_FS_X` (cortical PV+ basket) — name is acceptable shorthand
- `str_FS_X` (striatal PV-FSI, Tepper-2018) — needs rename (#9)
- `motor_FS_X` (project invention, NEGATIVE) — retire or rename (#11)
- `dg_fs` (DG PV+ basket) — needs rename (#22)

**Consistent fix-pattern:** add cell-class qualifier:
- `cortex_FS_X` → `cortex_pv_basket_X` (Wave 2)
- `str_FS_X` → `str_PV_FSI_X` (Wave 1)
- `motor_FS_X` → retire (Wave 1)
- `dg_fs` → `dg_pv` or `dg_pv_basket` (Wave 2)

**Note:** the *preset name* `IZH2007_FS_CORTICAL_INTERNEURON` is correct for
its cortical use; the issue is the shared use across biologically distinct
populations. Do not rename the preset itself (saved-profile coupled, Wave 3).

### Pattern P3: Pre-Cluster-D legacy region clash
The older `--hippocampus` flag uses non-canonical region names
(`place_cells`, `goal_cells`); the new `--enable-cluster-d-hippocampus` uses
canonical `dg / ca3 / ca1`. Both can be active simultaneously in flagship
(place_cells receives ca1 output when both flags are on per the perception arc).

**Instances:**
- `--hippocampus` flag (rename #4)
- `place_cells` region (rename #5)
- `goal_cells` region (rename #6)
- `hippo_to_cortex` gate (rename #20)

**Phasing-out plan:**
1. Wave 1 renames: `--hippocampus` → `--enable-place-goal-readout`,
   `place_cells` → `sensor_place_readout`, `goal_cells` → `ppc_goal_X`,
   `hippo_to_cortex` → `place_goal_to_cortex`.
2. Mark all four as deprecated with a 1-release alias cycle.
3. Once Cluster D matures into flagship (per SCIENCE_ROADMAP §552), retire
   the legacy flag entirely. Document the migration path in CHANGELOG.

### Pattern P4: Transmitter-as-region naming
Single instance: `dopamine` region (rename #3). Glossary flags as
`[discrepancy]` per C.16 (A9 + A10 collapse).

**Fix:** rename region to `snc`; keep the *transmitter* modulator named
`dopamine` (correct). When VTA / mesolimbic arc is added, register a
separate `vta` region. (Wave 1.)

### Pattern P5: Bridge-field name confusability
Single instance: `cp_plasticity_gain` vs `cp_plasticity_window_gate`
(rename #12). Names sound nearly identical; semantics differ (rate
multiplier vs binary window gate).

**Fix:** rename `cp_plasticity_gain` → `cp_plasticity_rate_gain` to make
the distinction unambiguous in code. (Wave 1.)

### Pattern P6: Modulator-source overclaim
Single instance: `acetylcholine` modulator (rename #10). Implements only
the striatal-TAN arm; brain has multiple ACh sources.

**Fix:** rename to `acetylcholine_tan` to leave namespace for future
basal-forebrain / brainstem ACh modulators. (Wave 1.)

---

## Acceptance criteria

The rename pass is complete when:

1. **All Tier 1 prose fixes applied** — DONE 2026-04-29 at commit `8aa2fcb`.
   No new prose issues introduced by the rename.
2. **All Wave 1 renames landed (13 items)** with backward-compat aliases for
   one release cycle.
3. **Glossary updated** to mark renamed identifiers as canonical: most are
   already documented; Wave 1 renames should update the `project:` field of
   each affected glossary entry.
4. **Saved-profile JSON files have alias maps** where needed (only for
   Wave 1 renames that affect region/gate names appearing in sidecar JSONs;
   none of the Wave 1 renames touch HH/Izh/AdEx preset enum names so
   `simulation_profiles/` is not affected).
5. **Test suite passes** — `pytest tests/` should run cleanly. Tests
   reference identifiers by their new names; aliases ensure backward
   compatibility for archival sidecars only.
6. **Acid test:** a 6-seed cheat-5 baseline run reproduces the documented
   4.08 ± 0.49 with no regression. Functional behavior unchanged; only
   names + aliases change.
7. **CHANGELOG updated** with one section per Wave 1 rename, listing the
   alias and the deprecation timeline (1 release cycle = 1 deprecation
   warning, then alias removal).
8. **CLAUDE.md flagship recipe** updated to use new names (preserving the
   alias-warning compatibility for users who copy old recipes).
9. **Webapp preset list** (`webapp/server.py:241-278`) updated to use new
   flag names.

---

## Non-acceptance (out-of-scope for this rename pass)

The following are explicitly NOT part of this rename. They may be addressed
in separate sessions or remain as-is.

1. **HH/Izh/AdEx preset enum-name renames** — saved-profile coupled; defer
   to a separate config-versioning task with proper alias-map infrastructure.
   Includes:
   - `HH_PFC_PYRAMIDAL` → `HH_DLPFC_PYRAMIDAL`
   - `IZH2007_HIPPO_PYRAMIDAL` → split CA1/CA3
   - `IZH2007_DOPAMINE` / `HH_DOPAMINE_SNC` → `*_SNC_VTA`
   - `BASAL_GANGLIA_STRIATUM`, `HIPPOCAMPUS_CA1_RS_FS` profile keys

2. **Theta-band 4-8 vs 4-12 Hz** (`experiment/readout.py:222-228`) — this
   is a behavioral parameter decision (rodent-hippocampal vs human-EEG
   convention), not a rename. Defer to a separate behavioral-policy session.

3. **"connections" CSR array name** (`cp_connections`) — surveys explicitly
   recommend leaving the data-structure name; only fix the user-facing
   prose ("connections" → "synapses"), which is Tier 1 and DONE at `8aa2fcb`.

4. **Glossary additions for project-internal terms** (silent-motor trap,
   perception arc, cheat #5, finalQ, recent_dist, beacon perception,
   cue-following reflex, sensed reward) — these are project terminology
   without canonical biology referents. Add only if future audits keep
   flagging them; otherwise leave as project-internal.

5. **`current_reward_signal` rename** — explicitly exempted by glossary
   `[NEEDS-REVIEW]`. The name is functionally accurate; the biological
   conflation issue is a separate documentation concern.

6. **Hardcoded UI baselines** (`webapp/static/app.js:927-928, 1083`:
   "baseline 5.88, flagship 4.08") — engineering staleness, not a rename.
   Update via a separate webapp pass.

7. **`R-STDP`, `WTA`, "reservoir", "perceptron"** algorithmic shorthand —
   accepted per glossary; not subject to rename.

8. **Historical findings docs** (`research/findings/*.md`, ~75 files) —
   reference-only per surveys B/E. Do not edit historical records;
   new findings going forward use new names.

9. **`g11_bg_runner.py` filename** — runner names follow `g{N}_*` for
   research-gate identifiers; this is project-organizational, not biological.

10. **Per-gate Q10 fields** (`hh_q10_m`, `hh_q10_h`, `hh_q10_n`) and
    `_BASE_HH_TEMP=6.3` — biophysics canonical, not subject to rename.

11. **Behavioral / numerical constants** — `HEURISTIC_DRIVE_PA=800.0`,
    sleep-replay rate ~150-250 Hz vs glossary 140-200 Hz, gamma test
    tolerance 25-55 Hz vs glossary 40-100 Hz — these are tuning constants /
    test tolerances, not renames.

---

## Source PDFs needed for borderline calls

When a Wave 1 or Wave 2 rename's biological grounding is questioned during
the rename pass, open the corresponding PDF for source-level verification.
All paths are absolute: `E:\Documents\Projects\sim-catalog\references\textbooks\`.

| Rename | Question | PDF path | Section / pp |
|---|---|---|---|
| #1 cortex_to_d1 | corticostriatal canonical name? | `kandel-pns-6e/full-book.pdf` | Ch 38 (basal ganglia) |
| #2 pfc → dlpfc_wm | dlPFC vs vmPFC vs OFC distinction | `kandel-pns-6e/full-book.pdf` | Ch 52-56 (PFC) |
| #3 dopamine → snc | A9 vs A10 split | `schultz-dopamine/Schultz-2016-NRN-RPE-twocomponent.pdf` | full paper (~13 pp) |
| #4 hippocampus, #5 place_cells | allocentric criterion | `okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.pdf` | Ch 4.7 pp 190-217 |
| #6 goal_cells → ppc_goal | PPC goal-encoding biology | `kandel-pns-6e/full-book.pdf` | Ch 17-29 (parietal) |
| #7 landmarks | object-vector cells (D.09) | `kandel-pns-6e/full-book.pdf` | Ch 53 (parahippocampal) |
| #8 bg-lateral-inhibition | MSN-MSN vs FSI feedforward | `basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf` | ch 6 (Wilson) |
| #9 str_FS → str_PV_FSI | 8-class taxonomy | `basal-ganglia-reviews/Tepper-2018-StriatalGABAergic-Heterogeneity.pdf` | full paper |
| #9 (extended) | FSI cross-action wiring | `basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf` | pp 157-158, 174 |
| #10 acetylcholine → acetylcholine_tan | TAN biology, B.05 disynaptic | `basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf` | pp 167, 171-172 |
| #11 motor_FS retire | spinal Renshaw inhibition | `kandel-pns-6e/full-book.pdf` | Ch 35 (spinal motor) |
| #14 gpe_X → gpe_proto_X | PV+ prototypic vs PV- arkypallidal | `basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf` | ch 7 (Kita) pp 111-114 |
| #28 IZH_HIPPO split | CA1 vs CA3 distinction | `kandel-pns-6e/full-book.pdf` | Ch 54 |

**For Schultz / dopamine renames:** also accessible:
- `schultz-dopamine/Schultz-1998-JNeurophysiol-PredictiveReward.pdf`
- `schultz-dopamine/Hollerman-Schultz-1998-NatNeuro.pdf`
- `schultz-dopamine/Schultz-2016-JNeuralTransm-RewardFunctionsBG.pdf`

**For Bolam-2000 BG synaptic organization:**
- `basal-ganglia-reviews/Bolam-2000-JAnat-SynapticOrgBG.pdf`

**For Buzsáki rhythms (sleep replay, theta, gamma) — referenced for completeness, not Wave 1:**
- `buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.pdf`

**For Marr / Albus / Hesslow cerebellum (Cluster F, all in Wave 3):**
- `cerebellum-marr/Marr-1969-cerebellar-cortex.pdf`
- `cerebellum-albus/Albus-1971-cerebellar-function.pdf`
- `cerebellum-marr/Hesslow-2013-classical-conditioning-motor.pdf`
- `cerebellum-marr/Moore-ed-2002-NeuroscientistsGuide-ClassicalConditioning.pdf`

---

## Session log

*To be filled in when the rename pass starts.*

### Date | Wave # | Rename(s) landed | Commit | Notes
- (empty)

---

## Cross-references

- **Glossary:** `E:\Documents\Projects\sim\references\glossary.md` (228 entries, 11 categories)
- **Structural audit:** `E:\Documents\Projects\sim\references\structural-naming-audit.md` (86 IDs)
- **Reference-coverage audit:** `E:\Documents\Projects\sim\references\reference-coverage-audit.md`
- **Surveys:**
  - `references\survey_part_A_sim.md` — sim/ engine
  - `references\survey_part_B_research.md` — research/ runners + probes
  - `references\survey_part_C_tests_exp_viz_ui.md` — tests + experiment + viz + ui
  - `references\survey_part_D_docs.md` — docs/plans/ + docs/
  - `references\survey_part_E_webapp_toplevel.md` — webapp + top-level docs
  - `references\survey_part_F_runtime_strings.md` — log/error strings + neural-simulator.py
- **Tier 1+2 prose fixes:** committed at `8aa2fcb` on 2026-04-29 (15 high-impact prose fixes)
- **Catalog (canonical biology source):** `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (5640 lines, ~323 entries across clusters A-Q)
- **Reference textbook library:** `E:\Documents\Projects\sim-catalog\references\textbooks\` (16 PDFs, all locally readable per reference-coverage-audit upgrade 2026-04-29 17:45)
