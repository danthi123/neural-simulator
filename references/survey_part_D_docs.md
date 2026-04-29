# Terminology Survey — Part D: `docs/`

**Audit against:** `references/glossary.md` (228 canonical entries).
**Tier 1+2 should be fixed in ACTIVE plans** (those describing in-progress or future work).
**Historical/completed plans are reference-only.**

## Summary

- Files scanned: 39 (1 SCIENCE_ROADMAP.md + 38 docs/plans/*.md)
- T1 (active): 14
- T2 (active): 11
- T3 (deferred): 12
- Reference-only (historical plans): 38

### Files classified

| Class | Count | Files |
|---|---|---|
| ACTIVE — SCIENCE_ROADMAP.md | 1 | `docs/SCIENCE_ROADMAP.md` |
| ACTIVE — this-week plans (2026-04-27 onwards) | 16 | `docs/plans/2026-04-27-perception-arc-plan.md`, all `2026-04-28-*.md` (12), all `2026-04-29-*.md` (5) |
| Reference-only — older plans | 22 | `2026-04-06-*` (2), `2026-04-20-*` (5), `2026-04-21-*` (2), `2026-04-24-*` (2), `2026-04-25-*` (3) |

## Heuristic

- Files dated 2026-04-27 onwards (this week) describe in-progress or future
  work and are subject to active terminology fixes.
- Files dated 2026-04-26 or earlier describe historical decisions (most are
  marked DONE / completed). Issues are listed but **not flagged for edit** —
  they're a record of how language evolved, including pre-glossary usage.

---

## Tier 1 — pure prose (active plans / SCIENCE_ROADMAP)

These are plain-prose terminology issues in active plans where the
prose describes biology and could/should use the canonical form.

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:1`
- **Current:** "Cluster D — Hippocampus Trisynaptic Loop + Replay Design"
- **Issue:** Glossary canonical for the EC-II → DG → CA3 → CA1 sequence is
  **"trisynaptic pathway"** (entry "Trisynaptic pathway (hippocampus)").
  "Trisynaptic loop" is colloquial — common in literature but not the
  glossary canonical form.
- **Canonical:** "trisynaptic pathway"
- **Tier:** 1
- **Notes:** also at lines 9 ("canonical Cajal 'trisynaptic loop'"), 26
  ("minimal trisynaptic loop"), 109 ("trisynaptic-loop core"). 4
  occurrences in this file alone. Highest-impact T1 fix in the survey.

### `docs/plans/2026-04-29-catalog-remediation-pass.md:62`
- **Current:** "informs Cluster D (hippocampal trisynaptic loop, T1.A,
  month 1)"
- **Issue:** Same "trisynaptic loop" → "trisynaptic pathway".
- **Canonical:** "trisynaptic pathway"
- **Tier:** 1

### `docs/SCIENCE_ROADMAP.md:286`
- **Current:** "Phase B: BG-style action selection (2026-04-25, GO)"
- **Issue:** "BG-style" is project shorthand. Glossary canonical for
  the BG anatomy is "basal ganglia"; for the algorithmic loop see
  "cortico-BG-thalamo-cortical loops" / "Alexander/DeLong loops"
  (Pathways section).
- **Canonical:** "basal-ganglia-style action selection" or "BG action
  selection" (initialism alone is acceptable per Anatomy section
  precedent).
- **Tier:** 1
- **Notes:** also "BG cascade" appears in lines 21, 252, 290, 430. The
  initialism is fine; only "BG-style" is mildly non-canonical (the "-style"
  suffix is informal). LOW priority.

### `docs/SCIENCE_ROADMAP.md:551-554`
- **Current:**
  ```
  - Cluster B: striatal microcircuit (D1/D2 asymmetry + FSIs + TANs)
  - Cluster A: closed BG loop (thalamo-cortical feedback + hyperdirect)
  - Cluster C: DA system completeness (tonic + compartmentalized)
  - Cluster D: sequence-aware learning (hippo/PFC → striatum + replay)
  ```
- **Issue:** "hippo/PFC" — "hippo" is informal shorthand for hippocampus.
  Glossary canonical is **"hippocampus"** (or specific subregion: DG, CA3,
  CA1). All-caps "PFC" is canonical. The "FSIs" / "TANs" plural forms
  are fine.
- **Canonical:** "hippocampus / PFC"
- **Tier:** 1

### `docs/plans/2026-04-29-cluster-c-tonic-da-design.md:74`
- **Current:** "register `_default_dopamine_config()` and enable
  subsystem (cumulative with --enable-tans / --enable-bg-neuropeptides)"
- **Issue:** Prose "cumulative with --enable-tans" is OK (flag name).
  But surrounding prose discusses "B.3 TANs" (line 6) and bare "TANs"
  is acceptable since glossary entry "TAN / ChI" treats "TAN" as
  canonical for the electrophysiology context.
- **Notes:** Not actually a T1 issue — included for completeness. NOT
  FLAGGED.

### `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md:30`
- **Current:** "Thalamic relay nuclei (VA/VL) send glutamatergic
  projections back to motor / premotor cortex."
- **Issue:** Glossary entry "Thalamus (motor / relay nuclei)" lists the
  accepted form as **"VL/VA"** (ventroanterior/ventrolateral) — order
  swapped. Both forms widely used in the literature.
- **Canonical:** "VL/VA" per glossary
- **Tier:** 1
- **Notes:** Also at line 55 ("VA/VL nuclei"). Cosmetic — both orderings
  used in published anatomy.

### `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md:42`
- **Current:** "Sparse hyperdirect; per Nambu 2002 ~30% of cortex
  pyramids contact STN"
- **Issue:** Glossary canonical for the route is **"hyperdirect
  pathway"** (Pathways section, A.03). "Hyperdirect" alone as adjective
  is acceptable when context is clear, which it is here.
- **Notes:** NOT FLAGGED — adjectival use is fine.

### `docs/SCIENCE_ROADMAP.md:288-292`
- **Current:** "**Architecture:** `research/runners/g11_bg_runner.py`
  builds a 30-region cascade — per-action `cortex_X → str_D1_X /
  str_D2_X → gpi_X → thal_X → motor_X` with disinhibition gating, plus
  shared STN and dopamine."
- **Issue:** "disinhibition gating" — accepted prose; glossary doesn't
  contain a "disinhibition" entry but it's standard neuroscience
  terminology and matches the Pathways → Direct pathway entry's "D1
  MSN → GPi/SNr disinhibition (action gating)" wording.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:34`
- **Current:** "DG fast-spiking interneurons providing strong feedforward
  inhibition"
- **Issue:** Glossary canonical for striatal FSI is "PV-FSI" (with PV+
  qualification). Hippocampal DG basket-cell FSIs are anatomically
  distinct from striatal FSIs but use the same shorthand. The
  `[NEEDS-REVIEW]` "FSI disambiguation" entry flags this exact concern.
  Since DG is hippocampus, "DG fast-spiking interneurons" is acceptable;
  "fast-spiking" is canonical for the electrophysiology, and the anatomical
  qualifier "DG" disambiguates from striatal FSI.
- **Notes:** NOT FLAGGED — anatomical qualifier present.

### `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md:5`
- **Current:** "Real BG has compartmentalized DA — DA axons targeting
  striatal patches matched by action selectivity."
- **Issue:** "striatal patches" is correct (glossary "Striosomal MSN
  (patch)" — patch ≡ striosome). Acceptable.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-c-tonic-da-design.md:33-34`
- **Current:**
  ```
  ProductionRule(rule_type="from_reward", sensitivity=+1.0, threshold=0.0),
  # Positive reward → DA activation above baseline
  # Negative reward → DA depression below baseline
  ```
- **Issue:** Per glossary "Two-component DA response" entry and the
  "Common incorrect / deprecated terms" note: phasic DA biology is
  two-component (Component 1 detection 60–90ms / Component 2 utility
  150–300ms). Calling negative-reward response "DA depression" is
  shorthand; technically biology has phasic depression below tonic for
  *aversive prediction errors* (Schultz98/16), not just any negative
  reward. Prose simplifies this.
- **Canonical:** prose is acceptable for design-doc level; the
  technical paragraph could note "phasic depression below tonic" (per
  catalog R2.4 in the remediation pass) if precision matters here.
- **Tier:** 1 (low priority — prose is descriptive, not normative)
- **Notes:** Cluster C v1 design plan — fix optional during update pass.

### `docs/plans/2026-04-29-cluster-c-tonic-da-design.md:54`
- **Current:** "da_signal = self.neuromodulator_manager.get_concentration("dopamine") - dopamine_baseline"
- **Issue:** code identifier — Tier 3 (NOT FLAGGED, deferred).

### `docs/plans/2026-04-28-cluster-b3-tans-implementation.md:5`
- **Current:** "Real BG TANs are tonically active (~5 Hz baseline) but
  pause briefly on salient events"
- **Issue:** "TANs" is canonical per glossary "TAN / ChI" entry —
  electrophysiology context.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b3-tans-implementation.md:9`
- **Current:** "Real BG TANs are tonically active (~5 Hz baseline) but
  pause briefly on salient events (reward, novel stimuli)"
- **Issue:** "salient events" is informal; glossary "Two-component DA
  response" uses "salience" only as a Component 1 attribute. Acceptable
  prose.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md:50`
- **Current:** "~1% of striatal cells; parvalbumin-positive; very fast
  firing (>200 Hz transient bursts)"
- **Issue:** "parvalbumin-positive" — glossary canonical is "PV+" or
  "PV-positive" (PV-FSI entry). "parvalbumin-positive" = PV-positive,
  the long form is acceptable.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md:71`
- **Current:** "ACh release at corticostriatal synapses modulates both
  LTP and LTD via M1/M4 muscarinic receptors."
- **Issue:** Receptor names "M1/M4" canonical per glossary "Muscarinic
  ACh receptors (mAChR)" entry (M1–M5).
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md:7`
- **Current:** "synapses terminating on `str_D1_*` get sign=+1 (LTP under
  +DA) and synapses terminating on `str_D2_*` get sign=-1 (LTP under -DA,
  LTD under +DA — the inverted D2 rule)"
- **Issue:** Identifier-style ("str_D1_*"). Tier 3 deferred. The prose
  "LTP under +DA / LTD under +DA" matches glossary D2 MSN entry note:
  "biology requires opposite-sign DA modulation of plasticity at D2 vs
  D1".
- **Tier:** 3 deferred (identifier).
- **Notes:** NOT FLAGGED for prose — biology summary is correct.

### `docs/plans/2026-04-28-cheat5-real-options-survey.md:51`
- **Current:** "v3 MSN lateral inhibition + same-action-only routing
  achieves the equivalent functional outcome of biological winner-take-all
  in our reduced model"
- **Issue:** "winner-take-all" — glossary "Lateral inhibition /
  center-surround" entry mentions "MSN lateral inhibition...used for
  action WTA". The "winner-take-all" longform is acceptable; "WTA"
  initialism is also canonical.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md:7`
- **Current:** "Real BG handles this via:
  - MSN-MSN GABAergic collaterals (within and between action pools)
  - Striatal FS interneurons (strong feed-forward inhibition)
  - Center-surround organization in pallidum"
- **Issue:** "MSN-MSN" is OK as a project shorthand. "Striatal FS
  interneurons" — should be "PV-FSI" or "striatal PV+ FSI" per glossary
  entry. "FS interneurons" is acceptable when context is clear (and the
  word "Striatal" qualifies it).
- **Tier:** 1 (mild — improve to "PV-FSI" / "striatal PV+ FSI" if
  precision matters)
- **Notes:** Also at line 51 ("FS interneurons + pallidal center-
  surround"). LOW priority.

### `docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md:264`
- **Current:** "the cascade currently relies on the indirect path
  (D2→GPe→STN→GPi excitation) for cross-action suppression"
- **Issue:** Glossary canonical "indirect pathway" (Pathways section,
  A.02). "indirect path" is shorthand.
- **Canonical:** "indirect pathway"
- **Tier:** 1
- **Notes:** Also at lines 264 ("real lateral inhibition is fast"),
  multiple cheat5 docs use "direct path" / "indirect path" similarly.

### `docs/plans/2026-04-28-cheat5-curriculum-staged-bg-cross.md:238`
- **Current:** "In real BG, the indirect path (cortex → D2 → GPe → STN
  → GPi) cancels the bias from the direct path."
- **Issue:** Same as above — "indirect path" / "direct path" should be
  "indirect pathway" / "direct pathway" per glossary canonical.
- **Canonical:** "indirect pathway" / "direct pathway"
- **Tier:** 1

### `docs/plans/2026-04-28-cheat5-curriculum-staged-bg-cross.md:243`
- **Current:** "Make cross-projections go ONLY to str_D2_Y (indirect,
  inhibitory net effect on action Y)"
- **Issue:** Adjective-form "indirect" alone is fine; following sentence
  uses "the indirect path" which is the issue (line 244-ish if any).
- **Notes:** NOT FLAGGED for line 243 itself.

### `docs/plans/2026-04-28-cluster-b3-tans-implementation.md:27`
- **Current:** "we want pause from a tonic baseline, not absolute
  negative."
- **Issue:** "tonic baseline" is canonical (glossary "Component 1
  / Component 2"; "tonic" is widely used and matches catalog R2.4 framing).
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md:34`
- **Current:** "Net effect: D1 LTPs under +DA / LTDs under −DA. D2
  LTPs under −DA / LTDs under +DA."
- **Issue:** "LTPs" / "LTDs" used as verbs (LTP-as-verb). Common usage
  in modern lit. Glossary "LTP / LTD" lists nominal forms, but the
  verb-style is widespread and informal.
- **Notes:** NOT FLAGGED — informal usage acceptable in design doc.

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:22`
- **Current:** "**Sharp-wave ripples (SWRs):** intrinsic CA3 events
  (R3.12 catalog framing) — population bursts arising from recurrent
  excitation + adaptation thresholds."
- **Issue:** Glossary canonical for the event is "SWR" or
  "sharp-wave–ripple" (with em-dash). "Sharp-wave ripples" with hyphen
  is also acceptable per glossary "accepted" line.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:44`
- **Current:** "ec → dg | 0.40 | 6.0 | True (perforant path)"
- **Issue:** "perforant path" canonical per glossary entry. CORRECT.
- **Notes:** NOT FLAGGED — already canonical.

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:50`
- **Current:** "ca3 → ca1 | 0.30 | 4.0 | True (Schaffer collaterals)"
- **Issue:** "Schaffer collaterals" canonical per glossary "Schaffer
  collateral" entry (singular form preferred but plural acceptable for
  the bundle).
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md:21`
- **Current:** "For synapses targeting `str_D1_X / str_D2_X / cortex_X
  / motor_X / thal_X / gpi_X / gpe_X / etc`: tag = action index"
- **Issue:** Identifier-style usage — Tier 3 deferred.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-e-topographic-maps-design.md:48`
- **Current:** "cortex_X regions: 2D coordinates (each X gets a corner:
  N=(0,1), E=(1,1), S=(1,0), W=(0,0) of unit square)"
- **Issue:** Identifier-style. Tier 3 deferred.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:24`
- **Current:** "**2026-04-27 (mid)** PFC working memory composing (4.41,
  p=0.018)"
- **Issue:** "working memory" canonical per glossary cluster G ("Working
  memory / PFC / cortical integration"); "PFC" canonical.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:260`
- **Current:** "E.2 — Brain-region framework (`sim/regions.py`):
  declarative `BrainRegion` + `RegionPathway` for multi-region
  simulations on a single bridge. PFC + Motor + Striatum + Thalamus
  etc. each own a contiguous index slice."
- **Issue:** "Striatum" / "Thalamus" capitalized as proper-noun region
  names — fits the project's region naming convention; glossary lists
  "striatum" lowercase as anatomical reference but capitalization in
  region-list context is fine.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:418-435`
- **Current:** "GO. Adding a recurrent prefrontal region on top of the
  hippocampus + sensory + curriculum stack..." [PFC working memory
  description]
- **Issue:** Prose throughout uses "PFC", "working memory", "recurrent"
  — all canonical or close to canonical.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:454-463`
- **Current:**
  ```
  Goal at (gx,gy) emits beacon  →  8 directional beacon sensors (cosine tuning)
     →  plastic beacon → goal_cells (curriculum-gated)
     →  cue-following reflex → cortex_X drive (replaces heuristic)
  Landmark at fixed (lx,ly) emits cue  →  8 directional landmark sensors
     →  plastic landmark → place_cells (curriculum-gated, self-organizes)
  ```
- **Issue:** "place_cells" / "goal_cells" — identifier style. The
  glossary `[NEEDS-REVIEW]` entry on "place cell" notes the project's
  usage doesn't strictly match O&N allocentric criteria; "place-cell-
  like" is a valid hedge. "goal_cells" is project-specific (no
  matching biology term).
- **Tier:** 2 (symbol-in-prose).
- **Notes:** Code-style identifiers in prose — keep as-is in code/diagrams,
  but doc prose references could use "place cells (project-specific
  region)" if precision matters.

---

## Tier 2 — symbol-in-prose (active plans / SCIENCE_ROADMAP)

These mix code identifiers with prose. The identifiers themselves are
canonical project shorthand per the glossary "project_identifier"
column, so the prose mixing is a stylistic concern more than a
correctness issue.

### `docs/SCIENCE_ROADMAP.md:290`
- **Current:** "per-action `cortex_X → str_D1_X / str_D2_X → gpi_X →
  thal_X → motor_X`"
- **Issue:** Heavy code-identifier usage in prose. Glossary entries
  list these as `project` shorthand for D1 MSN, D2 MSN, GPi, thalamic
  relay, motor pool — all canonical project-form.
- **Tier:** 2
- **Notes:** Acceptable per glossary's "project_identifier" convention.

### `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md:9-15`
- **Current:** ASCII-art cascade `sensory → cortex_X → str_D1_X → gpi_X
  → thal_X → motor_X / etc`
- **Issue:** Code identifiers in prose-diagram. Same as above.
- **Tier:** 2

### `docs/plans/2026-04-29-cluster-d-hippocampus-design.md:11-15`
- **Current:** ASCII-art trisynaptic loop diagram with `EC → DG → CA3
  → CA1`
- **Issue:** Anatomical abbreviations in prose; all canonical per
  glossary "Hippocampus subregions" entry.
- **Tier:** 2 (NOT FLAGGED — fully canonical).

### `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md:13-17`
- **Current:** description of "Single `dopamine` neuromodulator", "4
  per-action DA modulators: `dopamine_N`, `dopamine_E`, ..."
- **Issue:** Identifier-style "dopamine_N" in prose; project-specific
  per-action shorthand.
- **Tier:** 2

### `docs/plans/2026-04-29-cluster-c-tonic-da-design.md:14-38`
- **Current:** Code block declaring `_default_dopamine_config()`.
- **Issue:** Code-style. Tier 3 deferred (identifier).
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-28-cluster-b3-tans-implementation.md:62-68`
- **Current:** "ModulatorTarget(target_type='plasticity_window_gate',
  scope='all'), production_rules=[ProductionRule(rule_type='pause_on_reward',
  ...)]"
- **Issue:** Identifier-style in code blocks. Tier 3 deferred.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:471-479`
- **Current:** Recipe block `python -m research.runners.g11_bg_runner
  --moving-goal --hippocampus --learned-perception --pfc ...`
- **Issue:** CLI flags are identifiers; not prose-terminology issues.
- **Tier:** 3 deferred.
- **Notes:** NOT FLAGGED.

### `docs/SCIENCE_ROADMAP.md:561-572`
- **Current:** Same recipe block, second occurrence with --bg-lateral-inhibition added
- **Issue:** Same — CLI identifiers. Tier 3 deferred.
- **Notes:** NOT FLAGGED.

### `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md:42`
- **Current:** "cortex_X → stn | 0.10 | 3.0 | False"
- **Issue:** Pathway-table identifier. Glossary canonical for the route
  is "hyperdirect pathway"; the identifier `cortex_X → stn` is
  project-form for that pathway.
- **Tier:** 2
- **Notes:** Pathway notation is unavoidable in implementation tables.

### `docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md:158-160`
- **Current:** "D1 MSNs LTP under +DA / LTD under -DA; D2 MSNs invert
  both signs."
- **Issue:** "+DA" / "-DA" symbols in prose; "D1 MSNs" / "D2 MSNs" are
  canonical per glossary "D1 MSN" / "D2 MSN" entries.
- **Tier:** 2
- **Notes:** Symbol mixing acceptable; canonical otherwise.

### `docs/plans/2026-04-28-cluster-b3-tans-implementation.md:151-160`
- **Current:** "Effective plasticity gain = `1 - (acetylcholine_concentration
  / baseline)`"
- **Issue:** "acetylcholine_concentration" symbol-style; "ACh" canonical.
- **Tier:** 2 (mild).

---

## Tier 3 — identifiers (FLAGGED only)

These are pure code identifier references; not subject to terminology
fixes per Part D scope. Listed for completeness so reviewers can
confirm Tier 3 = identifier-only.

- `docs/plans/2026-04-29-cluster-c-tonic-da-design.md`: `_default_dopamine_config`,
  `NeuromodulatorConfig`, `ModulatorTarget`, `ProductionRule`, `bridge.core_config.current_reward_signal`,
  `effective_reward_lr`, `eligibility[:n]`, `dopamine_baseline` — all
  pure code.
- `docs/plans/2026-04-29-cluster-d-hippocampus-design.md`: `place_cells`, `goal_cells`,
  `BrainRegion`, `RegionPathway` — code.
- `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md`:
  `cp_synapse_action_tag`, `dopamine_N`, `dopamine_E`, `dopamine_S`,
  `dopamine_W`, `from_action_specific_reward` — code.
- `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md`: pathway
  tables (cortex_X → stn etc).
- `docs/plans/2026-04-29-cluster-e-topographic-maps-design.md`: dataclass
  fields, `cp_neuron_coords`.
- `docs/plans/2026-04-28-*.md` (most files): all CLI flags
  (--enable-d1-d2-asymmetry, --enable-striatal-fsis, --enable-tans,
  --enable-tonic-da, --bg-cross-projections, etc), kwargs, GPU array
  names, etc.
- `docs/plans/2026-04-28-cluster-b3-tans-implementation.md`: `pause_on_reward`,
  `plasticity_window_gate`, `compute_plasticity_window_gate_multiplier`.
- `docs/SCIENCE_ROADMAP.md`: many CLI flags, `cp_plasticity_gain`,
  `RegionPathway.plasticity_gate`, etc.
- `docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md`:
  `str_FS_X`, `cortex_to_fs_weight`, `fs_to_msn_weight`.
- `docs/plans/2026-04-28-cheat5-v4-implementation.md`: `_run_pretraining_phase`,
  `_PRETRAINING_THAWED_GATES`, etc.
- `docs/plans/2026-04-28-structural-plasticity-implementation.md`:
  `cp_synapse_alive`, `cp_synapse_survival`, `update_pruning`,
  `pruning_alpha`, `pruning_threshold`, `pruning_weight_floor`.
- `docs/plans/2026-04-28-perf-*-design.md`: `cp_firing_states`,
  `motor_idx_per_action`, etc.

---

## Reference-only — historical plans (DO NOT FIX)

These are historical records of decisions made at specific dates. New
plans going forward should use canonical terminology. Issues listed
for the record only.

### `docs/plans/2026-04-25-phase-b-bg-action-selection.md`

**Status:** historical. The plan describes Phase B BG cascade design.
Phase B was implemented and validated 2026-04-25 (per CLAUDE.md
"Phase B BG Action Selection Module" section, status GO).

| Line | Current | Issue | Canonical |
|---|---|---|---|
| 27 | "VTA/SNc DA neurons (5-10 cells)" | OK — both VTA and SNc canonical per glossary "VTA" and "SNc" entries | — |
| 32-37 | "striatum_D1[N,E,S,W] / striatum_D2[N,E,S,W] / GPi[N,E,S,W] / GPe[N,E,S,W] / thalamus_VL[N,E,S,W]" | Note "striatum_D1" was the proposed identifier — actual implementation used `str_D1_X` per CLAUDE.md and glossary | — (historical) |
| 35 | "direct path / indirect path" | "direct pathway" / "indirect pathway" canonical | (historical) |
| 87 | "cortical-striatal plasticity is the LEARNING site" | Glossary canonical "corticostriatal" (no hyphen). | "corticostriatal" |
| 150 | "D1 pathway: DA enhances direct path response" | "direct pathway" canonical | (historical) |
| 153 | "D2 pathway: DA suppresses indirect path response" | "indirect pathway" canonical | (historical) |
| 175 | "DA neurons fire at ~3-5 Hz (tonic)" | OK — "DA neurons" canonical shorthand for SNc/VTA neurons | — |
| 198 | "Cortico-striatal weights to correct action grow" | "Corticostriatal" canonical | (historical) |
| 256 | "PFC working memory" | OK — canonical | — |

### `docs/plans/2026-04-25-session-g-motor-exploration.md`

**Status:** historical. Implementation complete (smoke tested per the
file's status header). Findings doc exists.

| Line | Current | Issue |
|---|---|---|
| 1-12 | "silent-motor trap that defeated [Routes A/C]..." | project-internal terms. OK | — |
| 26-36 | ASCII-art with `cortex_X → motor` | OK | — |
| 115 | "first-spike WTA" | "WTA" canonical | — |
| 119 | "Faster decay (`stdp_tau` smaller)" | code identifier in prose | (historical) |
| 131 | "could DA modulate the exploration rate (tonic vs. phasic)? Future work." | OK | — |

### `docs/plans/2026-04-25-session-g-action-attribution.md`

**Status:** historical contingency, did not get launched (V3 didn't
fail). Marked "Drafted as contingency" in header.

| Line | Current | Issue |
|---|---|---|
| 7-12 | "V1+V2+V3 isolated three layers..." | project-internal. OK | — |

### `docs/plans/2026-04-24-neuromodulator-subsystem.md`

**Status:** historical. E.1 framework DONE per CLAUDE.md.

| Line | Current | Issue |
|---|---|---|
| 5 | "each hormone (dopamine, noradrenaline, etc.)" | "noradrenaline" is glossary `accepted` (canonical "NE" or "norepinephrine") — both forms acceptable per glossary. | NOT FIX |
| 13, 49, 179, 388, 402, 422, 426, 432, 469, 613, 619, 1253, 1276, 1311 | "noradrenaline" as the modulator name | Per glossary "Norepinephrine (NE) / noradrenaline" entry, "noradrenaline" is in the `accepted` list. The codebase uses "noradrenaline" as the module/registration name (and probably should keep — the API is now stable). | (historical, accepted form) |

### `docs/plans/2026-04-24-brain-region-framework.md`

**Status:** historical. E.2 framework DONE per CLAUDE.md.

| Line | Current | Issue |
|---|---|---|
| 5 | "(PFC, BG, hippocampus, etc.)" | All canonical | — |
| 19 | "Full PFC working memory tuning" | canonical | — |
| 20 | "Hippocampus / striatum / amygdala" | canonical | — |

### `docs/plans/2026-04-21-g6-2d-gridworld-design.md`

**Status:** historical. G6 PARTIAL per CLAUDE.md.

No biology-terminology issues. Mostly RL / engineering terms.

### `docs/plans/2026-04-21-g5v3-signed-perceptron-design.md`

**Status:** historical. G5 GO per CLAUDE.md.

| Line | Current | Issue |
|---|---|---|
| 11 | "the sim's eligibility trace stores `|Δw|` (unsigned)" | "eligibility trace" canonical per glossary | — |

### `docs/plans/2026-04-20-g1-implementation-plan.md`, `docs/plans/2026-04-20-g1-encoder-decoder-loss-design.md`, `docs/plans/2026-04-20-g2-sim-local-learning-design.md`, `docs/plans/2026-04-20-g3-persistence-design.md`, `docs/plans/2026-04-20-g5-sensorimotor-design.md`

**Status:** historical. G1, G3, G5 GO; G2 NO-GO per CLAUDE.md.

No significant biology-terminology issues. Most prose discusses
encoder-decoder pipeline / RL gates / engineering concerns. References
to "STDP", "STP", "homeostasis", "Izhikevich" etc are all canonical
per glossary.

### `docs/plans/2026-04-06-interactive-exploration-and-experiment-prototyping-design.md`, `docs/plans/2026-04-06-phase1-module-split.md`

**Status:** historical. Module split DONE.

No biology-terminology issues. Discusses module architecture, UI,
threading. Engineering only.

---

## Items NOT flagged (intentional shorthand)

These were considered but not flagged because they fall into glossary-
permitted shorthand or `accepted` forms:

1. **"BG cascade" / "BG output complex"** — glossary's GPi/SNr entry
   uses "BG output complex" canonical; "BG cascade" is project shorthand
   for the per-action chain.
2. **"DA" alone** — glossary canonical "dopamine" or "DA"; both
   acceptable.
3. **"NE" alone vs "noradrenaline"** — both `canonical` and `accepted`
   per glossary; old plan files use "noradrenaline" because that's
   the actual module/string name in the implementation.
4. **"hippo"** in SCIENCE_ROADMAP §Cluster D bullet — informal
   shorthand. FLAGGED above as Tier 1 fix; debatable whether the
   bulletpoint terseness justifies "hippo".
5. **"BG-style"** — informal. FLAGGED above as Tier 1 (low priority).
6. **"FSIs"** in active plan prose with anatomical context (e.g.
   "Striatal FSIs", "DG FSIs") — qualifier disambiguates from the
   glossary `[NEEDS-REVIEW]` "FSI disambiguation" concern.
7. **CLI flag names** like `--bg-lateral-inhibition`,
   `--enable-tans`, `--enable-d1-d2-asymmetry` — Tier 3 deferred,
   identifiers not prose terminology.
8. **"corticostriatal"** vs "cortico-striatal" — glossary writes
   "corticostriatal" (no hyphen) consistently. Active plans use the
   non-hyphenated form correctly. Historical plans (2026-04-25
   phase-b-bg-action-selection.md) use "cortical-striatal" /
   "Cortico-striatal" — flagged in historical-only section.
9. **"GPe" / "GPi" / "STN" / "thalamus"** — all canonical.
10. **"trisynaptic loop"** vs "trisynaptic pathway" — flagged as the
    top Tier 1 fix below; present in 4 active-plan locations.
11. **"VA/VL" vs "VL/VA"** — order-swap; both forms used in lit.
    LOW priority Tier 1 cosmetic.

---

## Top-3 most-impactful Tier 1 fixes (priority order)

1. **"trisynaptic loop" → "trisynaptic pathway"** — 4 occurrences in
   active plans (`docs/plans/2026-04-29-cluster-d-hippocampus-design.md`
   lines 1, 9, 26, 109; `docs/plans/2026-04-29-catalog-remediation-pass.md`
   line 62). The glossary canonical is "trisynaptic pathway"; "trisynaptic
   loop" is colloquial. Easiest one-shot fix with measurable canonical
   alignment.
2. **"direct path" / "indirect path" → "direct pathway" /
   "indirect pathway"** — multiple occurrences in active cheat5 docs
   (`docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md:264`,
   `docs/plans/2026-04-28-cheat5-curriculum-staged-bg-cross.md:238,
   243`). Glossary canonical names the pathway, not the path. ~6
   occurrences total in active docs.
3. **"hippo" / "hippo/PFC" → "hippocampus" / "hippocampus / PFC"** —
   `docs/SCIENCE_ROADMAP.md:554` (Cluster D bullet). Single occurrence
   but in a high-visibility doc that documents the biology buildout
   plan.

---

## Policy questions

1. **"trisynaptic loop" treatment.** Glossary uses "trisynaptic pathway"
   as canonical, but "trisynaptic loop" is widely used in modern
   literature (and is the term Cajal originally used). Should the
   glossary update to accept "trisynaptic loop" as `accepted`, or should
   docs uniformly switch to "trisynaptic pathway"? Recommend the
   latter — keep glossary as authority, update docs.

2. **Per-action region identifiers in prose.** Files like
   `docs/plans/2026-04-29-cluster-c-v2-compartmentalized-da-design.md`
   mix `cp_synapse_action_tag`, `dopamine_N`, `str_D1_X` etc into prose.
   This is unavoidable for design docs that describe code. Suggest
   policy: **identifier-style in code blocks = Tier 3 (deferred);
   in prose paragraphs = Tier 2 (acceptable per glossary
   `project_identifier` convention)**. Adopt this convention going
   forward.

3. **Historical-plan terminology.** Plans dated >2 weeks old
   (2026-04-06 / 2026-04-20 / 2026-04-21 / 2026-04-24 / 2026-04-25)
   describe completed work. Their terminology reflects pre-glossary
   usage. Recommend leaving them as-is (per the "historical artifacts"
   policy in this survey's brief). Exception: if a plan file is
   actively being referenced from current work, it could be updated
   on the next ALL-CAPS-required-update pass.

4. **"goal_cells" / "place_cells" project-specific identifiers.** Per
   glossary `[NEEDS-REVIEW]` entries on "place cell usage" and
   "hippocampus without subregion", these regions are project-specific
   and don't strictly match biology canonicals (place cells should be
   allocentric per O&N 1978; project's are sensor-driven). Active plan
   prose could optionally hedge with "place-cell-like" or "project's
   `place_cells` region (sensor-driven, not allocentric)". LOW priority
   — `[NEEDS-REVIEW]` flag in glossary already documents the concern.

5. **D1/D2 plasticity-asymmetry framing.** Glossary D2 MSN entry notes:
   "biology requires opposite-sign DA modulation of plasticity at D2
   vs D1; current project uses same sign for both (open issue)". This
   is an *implementation* discrepancy, not terminology. The docs/plans
   describing Cluster B.1 (`docs/plans/2026-04-28-cluster-b1-d1d2-
   asymmetry-implementation.md`) explicitly address closing this gap.
   No action needed for terminology audit.
