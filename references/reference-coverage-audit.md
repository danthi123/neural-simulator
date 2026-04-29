# Reference coverage audit — pre-rename-pass

**Date:** 2026-04-29
**Purpose:** Determine whether existing reference-material ingestion is deep enough to ground the structural rename pass, or whether additional source-text ingestion is required first.
**Author:** Claude Code reference-audit pass
**Correction (2026-04-29 17:45, post-audit):** Owner clarified that **all 16 specialty PDFs are locally available** at `E:\Documents\Projects\sim-catalog\references\textbooks\` (sibling repo, not the sim repo). The audit subagent only checked `sim/references/textbooks/` and saw only Kandel. The "specialty PDFs not locally readable" caveats below are obsolete — full source library can be re-read on demand. **Headline verdict upgrades from "sufficient with caveats" to "sufficient — full source corpus available for re-verification."**

---

## Headline verdict (updated)

**Sufficient. Full source corpus is locally readable at `sim-catalog/references/textbooks/`.**

The catalog (5640 lines, ~323 mechanism entries, 540 citation strings) and the
glossary (228 canonical entries) together cover essentially every project
identifier the rename pass will touch. The existing surveys (A/C/E + B/F)
have already mapped the codebase against the glossary at file:line
granularity. **The rename pass is unblocked across all 17 clusters** —
Kandel covers most clusters self-sufficiently, and for any rename where
the *biological distinction* drives the decision (D1 vs D2 plasticity
sign, GPe prototypic vs arkypallidal, MSN E_inh = −60 mV, Tepper-2018
striatal interneuron taxonomy, two-component DA, Marr-Albus PF→PC LTD),
the source PDF can be opened directly from `sim-catalog/references/textbooks/`.

The catalog itself is internally consistent — discrepancy markers are
explicit (`[discrepancy: ...]`, ~32 occurrences) and `[NEEDS-REVIEW]` is
not used anywhere in the catalog body (the surveys use it for
glossary-flagged cases).

**Locally available specialty PDFs (in `sim-catalog/references/textbooks/`):**

- `kandel-pns-6e/full-book.pdf` (also locally on the sim repo)
- `basal-ganglia-reviews/Bolam-2000-JAnat-SynapticOrgBG.pdf`
- `basal-ganglia-reviews/Tepper-Koos-2017-StriatalGABAergicInterneurons.pdf`
- `basal-ganglia-reviews/Tepper-2018-StriatalGABAergic-Heterogeneity.pdf`
- `basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.pdf`
- `cerebellum-marr/Marr-1969-cerebellar-cortex.pdf`
- `cerebellum-albus/Albus-1971-cerebellar-function.pdf`
- `cerebellum-marr/Hesslow-2013-classical-conditioning-motor.pdf`
- `cerebellum-marr/Moore-ed-2002-NeuroscientistsGuide-ClassicalConditioning.pdf`
- `okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.pdf`
- `buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.pdf`
- `sutton-barto/SuttonBarto-RL-2nd-ed.pdf`
- `schultz-dopamine/Schultz-1998-JNeurophysiol-PredictiveReward.pdf`
- `schultz-dopamine/Hollerman-Schultz-1998-NatNeuro.pdf`
- `schultz-dopamine/Schultz-2016-NRN-RPE-twocomponent.pdf`
- `schultz-dopamine/Schultz-2016-JNeuralTransm-RewardFunctionsBG.pdf`

**For any rename where catalog claim is borderline, open the cited PDF and verify before committing the rename.** No additional ingestion is required as a precondition.

The original per-cluster gaps below remain a useful map of *which PDF to open first* if a particular rename's biological claim needs source-level verification.

---

## Per-cluster coverage grade

For each cluster: depth grade (A/B/C/D), sources cited, glossary coverage,
ready-for-rename verdict, specific gaps.

### Cluster A — Closed BG action-selection loop (project flagship)

- **Catalog depth:** **A** — 16 entries (A.01–A.16); flagship architecture is mapped one-for-one against textbook biology.
- **Sources cited:** Kandel 6e Ch 38 (9×); Bolam-2000 (13×); PBR-160 (18×). **Most heavily-cited cluster after C and E.**
- **Locally readable:** Kandel only. Bolam-2000 + PBR-160 (Nambu, Deniau, Wilson, McGinty, Kita chapters) **NOT readable.**
- **Glossary coverage:** Excellent. Striatum, GPe (with arkypallidal split A.13), GPi/SNr, SNc, VTA, STN, thalamus, TRN, PPN, MLR, IO, all per-action variants (`str_D1_X`, `gpe_X`, `gpe_arky_X`, `gpi_X`, `thal_X`, `motor_X`) all enumerated.
- **Sufficient for renaming?** **Yes-with-caveats.** Glossary already accepts `gpi_X` covers GPi+SNr (project shorthand); current naming is internally consistent.
- **Gaps**:
  - **A.13 GPe prototypic (PV+) vs arkypallidal (PV−) split** — sourced from PBR-160 ch 7 Kita pp 111–114 + Mallet 2008. Catalog states PV+ → STN/GPi/SNr, PV− → striatum (pallidostriatal feedback). Project's `gpe_X` is single-pool but glossary recognizes `gpe_arky_X` as the arkypallidal canonical form. **Needs PBR-160 re-read if the rename adds the split.**
  - **A.14 perisomatic GPe → GPi/SNr inhibition vs distal striatal D1** — sourced from PBR-160 ch 8 Nambu Fig 2 pp 137–139. Drives the T0.B "missing GPe→GPi pathway" entry in the buildout roadmap. **The pathway name itself (`gpe_X → gpi_X`) is unambiguous; the WEIGHT calibration (3× str_d1) is the specialty-PDF claim.**
  - **A.16 STN dual-mode rebound dynamics (Cav3 short, Cav1.2/1.3 long)** — sourced from PBR-160 ch 4. Project's `IZH2007_STN_BURST` doesn't model this; rename only affects whether `stn` should become `stn_burst` or stay generic — terminology either way is fine.

### Cluster B — Striatal microcircuit & WTA

- **Catalog depth:** **A** — 17 entries (B.01–B.17); contains the most-cited specialty-PDF cluster (Tepper-Koos 2017 + Tepper-2018, ~36 cites).
- **Sources cited:** Kandel 6e Ch 38 (7×); Tepper-2018 (18×); PBR-160 (18×); Bolam-2000 (3×); Schultz (7×); Sutton (5×).
- **Locally readable:** Kandel only. **Tepper-2018, Tepper-Koos 2017, PBR-160 ch 16 McGinty NOT readable** (these are the texts behind the FSI/TAN/NPY-LTS/THIN taxonomy).
- **Glossary coverage:** Excellent. PV-FSI, TAN/ChI, LTS, NGF, THIN, FAI, SABI, CR all enumerated as separate striatal interneuron classes per Tepper-2018 taxonomy. MSN (D1, D2, patch/striosomal). All canonical naming captured.
- **Sufficient for renaming?** **Yes-with-caveats.** Glossary explicitly says cortical NGF and striatal NGF are *not* isomorphic (B.09 supplemental); FSI shorthand `IZH2007_FS_CORTICAL_INTERNEURON` reused for striatal FSIs is acknowledged engineering shortcut.
- **Gaps:**
  - **B.05 ChI/TAN disynaptic inhibition through NPY-NGF (TK-2017 pp 167, 171–172; Tepper-2018 pp 2–6)** — catalog claims "TAN pause → permissive plasticity window" framing is *incomplete* per specialty texts; real mechanism includes NK-1 substance P arm. **If renaming `--enable-tans` adds the disynaptic pathway, would need TK-2017 re-read.**
  - **B.06 PV-FSI Scgn+ vs Scgn− subtype split (Tepper-2018 pp 8–9, Garas 2016)** — catalog claims D1 vs D2 selectivity. Affects whether `str_FS_X` should split into `str_FS_D1pref_X` / `str_FS_D2pref_X` per glossary's "two FSI subtypes" note.
  - **B.14 MSN GABA reversal E_inh = −60 mV (Wilson PBR-160 ch 6)** + **B.15 SNc DA E_Cl = −55 mV (lacking KCC2)** — these are PBR-160 specialty claims that drive Tier 0 buildout T0.A. **Already implemented as `BrainRegion.E_inh_override` per `survey_part_A_sim.md` lines 153–162.** Rename impact: low (already named correctly). Verification impact: would need PBR-160 ch 6 (Wilson) re-read.
  - **B.07 patch/matrix anatomy** — Kandel + PBR-160. Project has `str_patch_X`. Naming OK.

### Cluster C — Dopamine & neuromodulation

- **Catalog depth:** **A** — 35 entries (C.01–C.35), heavily cited (139× Kandel, 47× Schultz). The largest single cluster in the catalog.
- **Sources cited:** Kandel 6e (139×); Schultz (Schultz 1998 + Hollerman & Schultz 1998 + Schultz 2016 NRN + Schultz 2016 JNT, 47×); Sutton & Barto (5×); Buzsáki (3×).
- **Locally readable:** Kandel only. **Schultz papers NOT locally readable** (4 papers, all referenced).
- **Glossary coverage:** Excellent. Two-component DA (C.32), tonic vs phasic, RPE, eligibility traces, A9/A10 split (with `[NEEDS-REVIEW]` flag for project's collapse), three-factor learning. NE, 5-HT, ACh as separate sections. PPN, raphe, LC, BF, TMN, NAc all enumerated as anatomical sources.
- **Sufficient for renaming?** **Yes-with-caveats.** Catalog explicitly flags `current_reward_signal` as conflating Component-1/Component-2/A9/A10/phasic/tonic. Survey-A:454 says NOT to flag every use; existing comment in `sim/config.py:188-197` already documents the simplification.
- **Gaps:**
  - **C.32 two-component DA — Component-1 detection (60–90 ms) vs Component-2 utility (150–300 ms)** — Schultz 2016 NRN claim. Drives T0.C "compose surprise-LR + adaptive-DA" decision. **If renaming surfaces these as `da_component1_*` / `da_component2_*`, would need Schultz 2016 NRN re-read** to verify the latency / function claims.
  - **C.04, C.16, C.22 broadcast scalar DA** — naming is `current_reward_signal`; surveys explicitly exempt this from rename per glossary `[NEEDS-REVIEW]`.
  - **C.18 ACh REM-on cholinergic from PPN/LDT** — project's TANs are striatal-only; if a brainstem ACh pool is added, would need Buzsáki Cycle 9 (ascending arousal) — not in our local Kandel chapters.

### Cluster D — Hippocampus & sequence learning

- **Catalog depth:** **A** — 24 entries (D.01–D.24); deep theta/SWR/place-cell coverage with Buzsáki + O'Keefe & Nadel cross-references.
- **Sources cited:** Kandel 6e (20×); Buzsáki Cycle 11 (4×); O'Keefe & Nadel ch 4–14 (5×).
- **Locally readable:** Kandel only. **Buzsáki Cycle 11 + O&N Ch 4 NOT locally readable.**
- **Glossary coverage:** Excellent. EC, DG, CA3, CA1, CA2, subiculum, place cell, grid cell, head-direction, border, object-vector, speed, time, engram cells. Trisynaptic loop. SWRs, theta, phase precession.
- **Sufficient for renaming?** **Yes-with-caveats.** Glossary explicitly flags `place_cells` (older `--hippocampus` flag) as not strictly meeting O'Keefe & Nadel allocentric criteria; canonical DG/CA3/CA1 used in newer `--enable-cluster-d-hippocampus`. The `place_cells` → `ca1_place_cells` rename the user mentioned is glossary-defensible without specialty PDF re-read.
- **Gaps:**
  - **D.06 place-cell allocentric criterion** (sensor-driven vs O&N criteria) — claim sourced from O&N Ch 4.7 pp 190–217. Catalog has the explicit testable validation criterion: "place cell should still fire on subsequent traversals after some cues are removed" — a rename to `ca1_place_cells` should keep this testable distinction in the glossary.
  - **D.18 theta** + **D.19 SWR**: Buzsáki Cycle 11 is the canonical reference. Project's `--enable-sleep-replay` infra is correctly named.
  - **goal_cells** has no biological analogue; glossary entry under "Posterior parietal cortex (PPC)" says `goal_cells` is closer to PPC than PFC despite naming. Rename to `ppc_goal_cells` or similar is glossary-defensible without specialty PDF.

### Cluster E — Sensory perception & cortical encoding

- **Catalog depth:** **A** — 22 entries (E.01–E.22); heavily cited but Kandel-dominated.
- **Sources cited:** Kandel 6e (130×); Schultz (19×); Buzsáki (2×); Sutton (1×).
- **Locally readable:** Kandel — full coverage.
- **Glossary coverage:** PPC, V1, S1, A1, columns, retinotopy, somatotopy, mechanoreceptors, photoreceptors, hair cells. All canonical.
- **Sufficient for renaming?** **Yes.** This cluster is essentially Kandel-self-contained.
- **Gaps:** None blocking rename. T2.B topographic maps roadmap entry is purely Kandel.

### Cluster F — Cerebellum & error-correction

- **Catalog depth:** **A** — 25 entries (F.01–F.25); fully populated by specialty texts.
- **Sources cited:** Kandel 6e (18×); Marr 1969 (15×); Albus 1971 (13×); Hesslow & Yeo 2002 + Hesslow 2013 (51×).
- **Locally readable:** Kandel only. **Marr 1969, Albus 1971, Hesslow 2013, Hesslow & Yeo 2002 ALL NOT locally readable.**
- **Glossary coverage:** Cerebellar cortex, DCN, AIP, mossy fibers, parallel fibers, climbing fibers, granule cells, Purkinje cells, IO. F.06 specifies AIP for eyeblink.
- **Sufficient for renaming?** **Yes-with-caveats.** Cluster F is mostly UN-built (presets only, no circuit). The rename target is small: `HH_CEREBELLAR_*` presets are already glossary-canonical. **Naming for the future T2.A buildout** — `mf_*`, `pf_*`, `cf_*`, `pc_*`, `aip_*`, `dcn_*`, `io_*` regions — is glossary-defensible without specialty-PDF re-read because the canonical names are already extracted to the glossary.
- **Gaps:**
  - **F.05 PF→PC LTD sign discrepancy** — Marr 1969 says PF→PC potentiates with CF; Albus 1971 says depresses; modern data (Sakurai 1987, Ito 1989) confirms LTD with CF. **Catalog explicitly flags this as `[discrepancy: sign of PF→PC plasticity]`.** Would need Marr or Albus re-read if implementation needs to defend choice. Rename impact: zero (Marr-Albus is the rule name regardless).
  - **F.17 mGluR1 + KCa intrinsic PC timer** — Hesslow 2013 claim about adaptive CR timing. Optional in v1 of T2.A.
  - **F.18 nucleo-olivary feedback (DCN → IO)** — Hesslow & Yeo 2002 ch 4 claim. Required for extinction.
  - **F.20–F.24 reversible-inactivation methodology, F.22 trace conditioning, F.23 hippocampus-dependent paradigms, F.24 adaptive CR latency** — all Hesslow & Yeo 2002. **Cluster F validation depth depends entirely on Hesslow & Yeo.**

### Cluster G — Working memory / PFC / cortical integration

- **Catalog depth:** **B** — 20 entries; Kandel-self-contained.
- **Sources cited:** Kandel only (Ch 13, 17–29, 52–56).
- **Locally readable:** Yes.
- **Glossary coverage:** PFC, dlPFC, vmPFC, OFC, PPC, all canonical.
- **Sufficient for renaming?** **Yes.** The user's example (`enable_pfc` overclaims; should be `enable_pfc_wm_attractor` or `enable_dlpfc_recurrent_wm`) is fully grounded by glossary's `[NEEDS-REVIEW]` flag on PFC subdivision.
- **Gaps:** None blocking.

### Cluster H — Motor & spinal output

- **Catalog depth:** **B** — 25 entries; Kandel + a few specialty (Henneman size principle).
- **Sources cited:** Kandel only.
- **Locally readable:** Yes.
- **Glossary coverage:** α-MN, γ-MN, Renshaw cell, V0–V3 spinal interneurons, spindle Ia/II, GTO Ib. All canonical.
- **Sufficient for renaming?** **Yes.** `motor_X` shorthand is glossary-accepted; no biological-distinction-driven renames pending.
- **Gaps:** None for current code.

### Cluster I — Channels & intrinsic dynamics

- **Catalog depth:** **A** — 23 entries.
- **Sources cited:** Kandel + 20 cross-cluster cites.
- **Locally readable:** Yes.
- **Glossary coverage:** HH, AdEx, Izh, AP, all canonical.
- **Sufficient for renaming?** **Yes.** All neuron-preset names + Q10 fields glossary-accepted.

### Cluster J — Synapses & plasticity rules

- **Catalog depth:** **A** — 39 entries (largest cluster); Kandel-dominated.
- **Sources cited:** Kandel-heavy with Sutton/Schultz cross-cites for three-factor.
- **Locally readable:** Yes (Kandel covers most of it).
- **Glossary coverage:** STDP, LTP/LTD, STP, NMDA/AMPA/kainate/GABA-A subunit composition, eligibility trace, three-factor learning, Hebbian. All canonical.
- **Sufficient for renaming?** **Yes.** `cp_*` GPU array names match glossary; `fused_stdp_weight_update` etc. are kernel-naming convention.
- **Gaps:** Survey-A noted STDP-window parenthetical bug at `sim/kernels.py:266-267` — already prose-flagged for Tier 1 fix.

### Cluster K — Sensory transduction

- **Catalog depth:** **B** — 15 entries; Kandel-only.
- **Sources cited:** Kandel.
- **Locally readable:** Yes.
- **Glossary coverage:** Photoreceptors, hair cells, mechanoreceptors (Pacinian/Meissner/Merkel/Ruffini, SA1/SA2/RA1/RA2 afferents), nociceptors. All canonical.
- **Sufficient for renaming?** **Yes** (cluster is mostly missing from the project; rename targets minimal).

### Cluster L — Development & critical periods

- **Catalog depth:** **A** — 23 entries.
- **Sources cited:** Kandel-dominated.
- **Locally readable:** Yes.
- **Glossary coverage:** PV maturation, ocular dominance critical period, structural plasticity, Cline & Haas. Cajal-Retzius, radial glia. Canonical.
- **Sufficient for renaming?** **Yes.**

### Cluster M — Neuromuscular junction

- **Catalog depth:** **C** — only 4 entries; missing from project entirely.
- **Sources cited:** Kandel.
- **Locally readable:** Yes.
- **Glossary coverage:** Minimal — NMJ entry only.
- **Sufficient for renaming?** **N/A** — nothing in project to rename. Cluster is a future-direction stub.

### Cluster N — Sleep & arousal

- **Catalog depth:** **A** — 19 entries; heavy Buzsáki cross-citation.
- **Sources cited:** Kandel + Buzsáki.
- **Locally readable:** Kandel only. Buzsáki Cycle 9–11 NOT readable.
- **Glossary coverage:** SCN, VLPO, TMN, BF, raphe, LC; NREM/REM, sleep stages, ripples, theta, gamma, slow oscillation. All canonical.
- **Sufficient for renaming?** **Yes-with-caveats** for the canonical name itself; **specialty for the deep mechanics**.

### Cluster O — Emotion, reward, motivation

- **Catalog depth:** **A** — 23 entries.
- **Sources cited:** Kandel + Schultz + Sutton.
- **Locally readable:** Kandel only.
- **Glossary coverage:** Amygdala (LA/BLA/CeA), hypothalamus (PVN/arcuate/LH/VMH/etc.), NAc shell/core, PAG. Canonical.
- **Sufficient for renaming?** **Yes.** Limbic anatomy mostly missing from project — the rename targets are limited to confirming `current_reward_signal` framing (which surveys exempt).

### Cluster P — Disease & neurodegeneration

- **Catalog depth:** **B** — 37 entries; Kandel-only.
- **Sources cited:** Kandel.
- **Locally readable:** Yes.
- **Glossary coverage:** Parkinson's, schizophrenia, epilepsy, etc. (where named).
- **Sufficient for renaming?** **Yes** (no project identifiers in P yet).

### Cluster Q — Glia & neurovascular

- **Catalog depth:** **C** — 8 entries; cluster missing from project entirely.
- **Sources cited:** Kandel.
- **Locally readable:** Yes.
- **Glossary coverage:** Astrocyte, oligodendrocyte, microglia, Schwann. Canonical.
- **Sufficient for renaming?** **N/A.**

---

## Aggregate gap analysis

### Identifiers in the simulator that catalog cannot ground without specialty PDFs

The following are flagged for caveat-mode rename — the catalog has the
extracted claim, but verification would require specialty-PDF re-read.
File:line refs are from the existing surveys.

| Identifier (project) | Catalog claim | Specialty PDF | Rename impact |
|---|---|---|---|
| `gpe_X` (single pool) → split `gpe_proto_X` + `gpe_arky_X` | A.13 PV+/PV− subtypes | PBR-160 ch 7 (Kita) | Glossary already has `gpe_arky_X` canonical; rename safe. |
| `--enable-tans` plasticity-window mechanism | B.05 disynaptic via NPY-NGF + NK-1 SP arm | TK-2017 + Tepper-2018 + PBR-160 ch 16 (McGinty) | Rename name itself is fine; mechanism documentation needs verification. |
| `str_FS_X` (single pool) → split Scgn± | B.06 D1/D2 selectivity | Tepper-2018 pp 8–9; Garas 2016 | Glossary doesn't yet enumerate Scgn± subpools. |
| `cortex_to_d1` plasticity gate (also gates D2) | Catalog flags D1/D2 *opposite-sign* DA modulation needed | PBR-160 + Schultz | Survey-A:325 already flagged this. |
| `current_reward_signal` (broadcast scalar) | C.32 two-component DA: detection + utility | Schultz 2016 NRN | Survey-A:454 explicitly exempts from rename per `[NEEDS-REVIEW]` |
| `place_cells` (sensor-driven) → `ca1_place_cells_sensor_driven` | D.06 allocentric O&N criterion | O&N Ch 4.7 pp 190–217 | Glossary `[NEEDS-REVIEW]` accepts both forms; rename safe. |
| `goal_cells` (no biological analogue) → `ppc_goal_cells` or remove | Glossary: closer to PPC than PFC | (no specialty needed) | Rename fully safe. |
| `enable_pfc` → `enable_pfc_wm_attractor` / `enable_dlpfc_recurrent_wm` | G cluster: PFC subdivisions matter | Kandel only — full coverage | Rename fully safe. |
| `IZH2007_DOPAMINE` / `HH_DOPAMINE_SNC` → `*_SNC_VTA` | Survey-A:438 + glossary `[NEEDS-REVIEW]` flags A9/A10 collapse | Kandel only | Rename safe. |
| `IZH2007_HIPPO_PYRAMIDAL` → split CA1 / CA3 | Glossary flags CA1/CA3 distinction; HH already split | Kandel only | Rename safe; Izh-side migration needed. |
| `BASAL_GANGLIA_STRIATUM` profile uses `IZH2007_RS_CORTICAL_PYRAMIDAL` for MSN | Survey-A:404 — profile cross-types | (no specialty needed) | Survey already says keep + comment. |
| `cerebellar_cortex` future regions (`mf_*`, `pf_*`, etc.) | F cluster naming | Marr 1969 / Hesslow & Yeo 2002 | Names canonical regardless; biology-grounding documentation would benefit. |

### Identifiers where Kandel coverage exists but catalog cited non-Kandel

These the rename-pass can defer to specialty re-ingestion **only if a
deeper biological claim is being made** — pure terminology rename is fine.

- All Cluster A entries citing PBR-160 / Bolam-2000 (~31 cites total)
- All Cluster B entries citing Tepper-2018 / TK-2017 / PBR-160 (~36 cites)
- All Cluster C entries citing Schultz 1998 / 2016 NRN / 2016 JNT (~47 cites)
- All Cluster D entries citing Buzsáki Cycle 11 + O&N (~9 cites)
- All Cluster F entries citing Marr/Albus/Hesslow (~73 cites)

### Concepts where catalog has [discrepancy] tags overlapping current identifiers

32 `[discrepancy: ...]` markers in the catalog. Spot-checked overlaps:

- **C.16** A9/A10 collapse — project's `dopamine` region. **Already in glossary `[NEEDS-REVIEW]`.**
- **F.05** PF→PC LTD sign — affects future cerebellum runner; not a current identifier.
- **D.06** place cell allocentric criterion vs sensor-driven — already in glossary `[NEEDS-REVIEW]`.
- **B.01** cortical NGF vs striatal NGF non-isomorphism — already in glossary supplemental note.

No new discrepancies surface project identifiers that the rename pass
hasn't already been told about by the surveys.

---

## Recommended actions before rename pass

### Tier 1 — Block rename (must ingest before)
**None.** No cluster has so big a gap that rename can't proceed. Both the
catalog body and glossary are sufficiently citation-rich and internally
consistent that rename can rely on them.

### Tier 2 — Caveat rename (rename can land but flag for re-verification)
For renames where the *biological distinction* is what motivates the
rename (not just naming hygiene), capture the catalog excerpt verbatim in
the rename PR description so future readers (or future ingestion of the
specialty PDF) can re-verify. Specifically:

1. **Cluster A** rename adding `gpe_arky_X` or `gpe_proto_X` split — quote
   A.13 from catalog. Specialty PDF: PBR-160 ch 7 (Kita). Re-acquisition: 1 chapter (~30 pp).
2. **Cluster B** rename splitting `str_FS_X` into Scgn± subpools — quote
   B.06 supplemental. Specialty PDF: Tepper-2018 pp 8–9. Re-acquisition: ~10 pp.
3. **Cluster C** rename adding `da_component1_*` / `da_component2_*` — quote
   C.32. Specialty PDF: Schultz 2016 NRN review. Re-acquisition: full paper (~13 pp).
4. **Cluster F** rename instantiating cerebellar regions (`mf_*`/`pf_*`/`cf_*`/`pc_*`/`aip_*`/`dcn_*`/`io_*`) — quote F.01/F.06/F.18. Specialty PDFs: Marr 1969 + Albus 1971 + Hesslow & Yeo 2002 ch 4. Re-acquisition: 2 short papers + 1 chapter.

### Tier 3 — Proceed (rename fully grounded by current materials)
- Cluster D rename: `place_cells` → `ca1_place_cells`, `goal_cells` → `ppc_goal_cells` or remove. **Glossary already has the canonical naming.**
- Cluster G rename: `enable_pfc` → `enable_pfc_wm_attractor` / `enable_dlpfc_recurrent_wm`. **Kandel-self-contained.**
- Cluster I/J kernel renames + STDP-parenthetical fix (`sim/kernels.py:266-267`). **Already Tier 1 in survey_part_A_sim.md.**
- All `current_reward_signal` references — **explicitly exempted** by glossary `[NEEDS-REVIEW]` and survey-A:454.
- All cortical / hippocampal / motor / spinal identifier renames — **Kandel-grounded.**

---

## Concrete recommendations

1. **Proceed with the rename pass for clusters E, G, H, I, J, K, L, M, N, O, P, Q without specialty re-ingestion.** Coverage is Kandel-self-contained or the glossary has already extracted the canonical name. Any rename in these clusters is safe.

2. **For clusters A, B, C, D, F renames, proceed but capture the catalog excerpt verbatim in the PR description** so the rename can be re-verified after specialty-PDF re-acquisition. The catalog is internally consistent and explicit about what it took from each source; the rename can land on its authority alone.

3. **Re-acquire specialty PDFs in this priority order if any rename fails review or if cluster F (cerebellum) buildout begins:**
   - **PBR-160** (Tepper/Abercrombie/Bolam 2007) — covers Clusters A + B simultaneously; URL in `textbooks-README.md`. Highest leverage.
   - **Tepper-2018** (Front. Neuroanat.) — covers Cluster B striatal interneuron taxonomy; short paper.
   - **Schultz 2016 NRN** — covers Cluster C two-component DA; ~13 pp.
   - **Hesslow & Yeo 2002 ch 4** — covers Cluster F validation criteria; required for T2.A buildout.
   - The remaining specialty PDFs (Bolam-2000, Tepper-Koos 2017, Marr 1969, Albus 1971, Hesslow 2013, Buzsáki 2006, O'Keefe & Nadel 1978, Sutton & Barto, Schultz 1998, Hollerman-Schultz 1998, Schultz 2016 JNT) are catalog-resident — re-acquire only if a specific rename or buildout requires fresh verification.

4. **One internal-consistency check the catalog flags but rename can fix during the pass:**
   - Catalog `[discrepancy]` markers (32 total) should be cross-referenced against rename targets. None are blocking, but the rename PR should explicitly note any case where the renamed identifier touches a flagged discrepancy (e.g. `dopamine` region rename intersects C.16 A9/A10 collapse).

5. **Add 1 glossary entry during rename:** "silent-motor trap" is load-bearing project terminology (`survey_part_E_webapp_toplevel.md:51`) but absent from the glossary. While not a biological term, adding it (with cross-ref to `research/findings/2026-04-25-phase-b-acid-test-real-win.md`) avoids future audits flagging it.

---

## Confidence summary

- **High confidence** on Cluster E, G, H, I, J, K, L, M, N, O, P, Q rename
  proceeding cleanly. Kandel + glossary cover everything the rename will
  touch.
- **Moderate confidence** on Cluster A, B, C, D, F renames where the
  *biological* claim depends on a specialty PDF that's not locally
  readable. The catalog extracted the claim 8 weeks ago; we're trusting
  that extraction. Internal consistency markers + explicit citations
  raise this to "moderate" rather than "speculative".
- **Block-level confidence** that no rename pass will fail catastrophically
  due to insufficient reference material. The catalog is rich enough that
  even worst-case (a rename that turns out to be wrong against re-read
  Tepper-2018) is recoverable as a follow-up rename, not a project halt.
