# Terminology Survey — Part E: `webapp/` + top-level docs

**Audit against:** `references/glossary.md` (228 canonical entries).

## Scope

Files scanned:
- `webapp/server.py` (~1205 LOC) — FastAPI server
- `webapp/static/app.js` (1481 LOC) — main dashboard JS
- `webapp/static/world.js` (1139 LOC) — World tab JS
- `webapp/static/charts.js` (411 LOC) — chart helpers
- `webapp/static/ui.js` (203 LOC) — UI helpers
- `webapp/static/index.html` (282 LOC) — markup
- `webapp/README.md` (~205 LOC)
- `CLAUDE.md` (~720 LOC) — project guide for Claude Code
- `README.md` (~648 LOC) — top-level user/dev docs
- `CHANGELOG.md` (~600+ LOC; treated as historical reference-only)
- `USER_GUIDE.md` (~619 LOC)
- `QUICKSTART.md` (~119 LOC)
- `CONTRIBUTING.md` (~426 LOC)

## Summary
- Files scanned: 13
- T1 (active prose): **42**
- T2 (symbol-in-prose): **9**
- T3 (deferred identifiers): **11**
- Reference-only (past CHANGELOG entries): **18**
- NOT-flagged intentional shorthand: noted at bottom

---

## Tier 1 — active prose

### `CLAUDE.md`

#### `CLAUDE.md:9`
- **Issue:** "biologically-inspired neuron models" — minor; the project's own preferred phrasing elsewhere is "biology-grounded" or "biologically grounded" (no hyphen). Pick one form for consistency.
- **Glossary:** style consistency (no specific entry). Compare with `README.md:3` which uses "biologically-grounded".
- **Recommendation:** "biologically-grounded neuron models" (matches `README.md`).

#### `CLAUDE.md:104`
- **Issue:** "Inhibitory reversal: `E_inh = -75mV`, propagation scaled 0.7x for driving force compensation".
- **Glossary:** `GABA_A receptor` entry (line ~600) explicitly notes **per-region overrides for striatum (~−60 mV), SNc DA (~−55 mV) per B.14, B.15**. This bullet implies a single global E_inh, which is now incorrect after the 2026-04-29 R1.1 catalog-pass change. The per-region overrides are mentioned later in the file but not referenced here.
- **Recommendation:** Add a parenthetical "(global default; per-region overrides for striatum and SNc DA — see R1.1 below)".

#### `CLAUDE.md:165–175`
- **Issue:** The list of `NEURAL_STRUCTURE_PROFILES` includes `BASAL_GANGLIA_STN_GPE`, `THALAMUS_TC_TRN`. These project identifiers are fine, but glossary entries clarify GPe = "globus pallidus externus", TC = "thalamocortical relay", TRN = "thalamic reticular nucleus". The CLAUDE.md prose around this could note the canonical expansion in a one-line gloss, since CLAUDE.md is read by LLM agents who may not know the abbreviation hierarchy.
- **Recommendation:** Optional one-line explanation "GPE/STN, TC/TRN, etc." Lower priority.

#### `CLAUDE.md:319` — "silent-motor trap"
- **Issue:** Project-specific term. Glossary does not list it. It IS load-bearing project terminology, however.
- **Recommendation:** No change. Project-specific term established by usage. Could optionally add a glossary entry "silent-motor trap" pointing to `research/findings/2026-04-25-phase-b-acid-test-real-win.md`.

#### `CLAUDE.md:329` — "tonic dopamine driving spontaneous striatal/cortical activity (Schultz 2007)"
- **Issue:** Tonic vs phasic DA distinction is correct here. Glossary `Two-component DA response` entry (line ~1098) gives the canonical Component 1 / Component 2 framing — this prose conflates "tonic" with "Component 1", but Schultz 2007's tonic-phasic-burst framework is distinct from the 2016 two-component decomposition. Either is acceptable; just note that "tonic dopamine" is a specific biological state, not a generic descriptor.
- **Recommendation:** Keep. (Tonic dopamine is a canonical glossary entry under `Tonic DA / phasic DA` if added; for now just usage is fine.)

#### `CLAUDE.md:355` — "shared 200-neuron reservoir + argmax readout"
- **Issue:** Reservoir is a glossary-listed term: "**reservoir computing**". Lowercase is fine. No issue.
- **Recommendation:** No change.

#### `CLAUDE.md:357` — `cortex_X → str_D1_X / str_D2_X → gpi_X → thal_X → motor_X`
- **Issue:** All project identifiers; consistent with glossary's `D1 MSN` and `D2 MSN` and `gpi_X` entries. "thal_X" is glossary-canonical. **No issue**.

#### `CLAUDE.md:362`
- **Issue:** "n_cortex=400 over-drove D1 to ~220 Hz (saturated, unphysiological)". Glossary's `MSN` entry notes biological MSN firing range; "unphysiological" is the right framing. No issue.

#### `CLAUDE.md:385` — "real critical periods close gradually"
- **Issue:** "critical period" is a glossary-canonical term. Correct usage. No issue.

#### `CLAUDE.md:434` — "BG cross-projections"
- **Issue:** Used informally throughout — no glossary entry, but consistent with project usage. The glossary documents `cortex_X → str_D1_Y` style as project shorthand.
- **Recommendation:** No change.

#### `CLAUDE.md:443–451`
- **Issue:** "Adaptive DA", "asymmetric adaptive DA", "broadcast DA" — DA is glossary-canonical for dopamine. No issue.

#### `CLAUDE.md:496` — "MSN cross-pool lateral inhibition"
- **Issue:** Glossary has `Lateral inhibition / center-surround` entry. "Lateral inhibition" is canonical. The project's `--bg-lateral-inhibition` is consistent. **No issue**.

#### `CLAUDE.md:526` — "Cluster B.1 (D1/D2 asymmetry, `--enable-d1-d2-asymmetry`) — PARTIAL SIGNAL"
- **Issue:** "D1/D2 asymmetry" is project-internal naming. Glossary entries for `D1 MSN` and `D2 MSN` use the canonical names. The biological asymmetry refers to **opposite-sign DA modulation of plasticity at D1 vs D2** (per glossary D2 entry, citing O.03). Prose is consistent.
- **Recommendation:** No change; possibly add "DA-modulation sign asymmetry" once for clarity.

#### `CLAUDE.md:548` — "Tepper-2018"
- **Issue:** Citation form — could be more explicit, e.g. "Tepper et al. 2018 (TK-2017 ch. 8)". Minor.
- **Recommendation:** Keep as-is; it matches the convention used in `references/feature-catalog.md`.

#### `CLAUDE.md:599` — "Uses heuristic + direct goal coords + distance-based reward"
- **Issue:** Three project-specific items. Glossary lists "`--learned-perception` (standalone, REPLACES heuristic)". The "heuristic" terminology is project-internal — represents a hand-coded direction-towards-goal nudge. Not glossary-listed.
- **Recommendation:** No change. Consistent with project usage.

#### `CLAUDE.md:640` — "great on 42-44, bad on 100-102, pooled 6-seed mean 5.23 ± 1.90"
- **Issue:** "pooled" usage is fine (statistical term). No issue.

#### `CLAUDE.md:641` — "phasic DA biology — dips are sharper than ramps (Schultz 1998)"
- **Issue:** "phasic DA" is canonical (paired with "tonic DA"). "Dips" / "ramps" are informal but recognizable.
- **Recommendation:** No change.

### `README.md`

#### `README.md:3`
- **Issue:** "biologically-grounded models (Izhikevich 2007, Hodgkin–Huxley, AdEx)". Good. "Hodgkin–Huxley" with en-dash matches glossary (where "Hodgkin and Huxley 1952" is the canonical citation). No issue.

#### `README.md:9` — "**Project status (2026-04-28):**"
- **Issue:** Date stamp is now historical (today is 2026-04-28 itself, but recent CLAUDE.md edits include 2026-04-29 changes). Roll forward to 2026-04-29.
- **Glossary:** N/A (date freshness, not terminology).
- **Recommendation:** Update to "Project status (2026-04-29)" or accept that README is one-day-stale.

#### `README.md:11` — "Phase B basal-ganglia action selection — silent-motor trap resolved"
- **Issue:** "basal-ganglia" hyphenated; glossary entries use "basal ganglia" without hyphen as the canonical noun. As an attributive adjective ("basal-ganglia action selection") the hyphen is grammatically defensible but unusual. The README also uses "BG" elsewhere as the abbreviation.
- **Recommendation:** Consider "basal ganglia action selection" (no hyphen) for consistency with glossary's noun form.

#### `README.md:14` — "biology-grounded version (4.08, p=0.00045, 30.6% over baseline) BEATS cheats-allowed (4.41)"
- **Issue:** "cheats-allowed" is project shorthand. Not glossary-listed. **Specific, fine.**
- **Recommendation:** No change.

#### `README.md:155` — "Phase B — Basal-Ganglia Action Selection (resolved 2026-04-25)"
- **Issue:** Same hyphenation question as line 11. Glossary canonical form: "basal ganglia". When used as a heading attribute ("Basal-Ganglia Action Selection") the hyphen is acceptable.
- **Recommendation:** Either form is OK; lean toward "Basal Ganglia" (no hyphen) for consistency with glossary.

#### `README.md:208` — "Selection emerges from independent disinhibition gates, not a shared argmax."
- **Issue:** "Disinhibition" is glossary-canonical (Direct pathway entry). "Argmax" is informal but RL-canonical. Both fine. No issue.

#### `README.md:230` — "**Reward-modulated STDP** (three-factor learning): eligibility traces × dopamine signal."
- **Issue:** Three-factor learning is glossary-canonical (line ~821). "Eligibility traces × dopamine signal" matches glossary's project description. No issue.

#### `README.md:252` — "Declarative concentration dynamics for dopamine / NE / 5-HT / etc."
- **Issue:** All glossary-canonical neuromodulator names. No issue.

#### `README.md:434–445` — "Hippocampus (`--hippocampus`) — place + goal cells with sparse Gaussian tuning"
- **Issue:** Glossary's `[NEEDS-REVIEW] "hippocampus" without subregion` entry (line ~1482) explicitly flags this: older `--hippocampus` flag uses generic `place_cells` + `goal_cells`. README mentions both conventions in nearby lines but doesn't disambiguate "hippocampus" vs the new `--enable-cluster-d-hippocampus`.
- **Recommendation:** Add one line: "(legacy generic place+goal cells; for canonical DG/CA3/CA1 trisynaptic loop see `--enable-cluster-d-hippocampus`)".

#### `README.md:438` — "Beacon perception ... 8 directional sensors detecting beacon, replaces direct goal coords"
- **Issue:** "Beacon" is project terminology (the goal-emitted intensity field used as a perception cue). Not glossary-listed but functionally specific. Project-internal, fine.
- **Recommendation:** No change. Glossary entry for `Beacon perception` would help future audits — consider adding to glossary or documenting in the perception arc finding.

### `webapp/README.md`

#### `webapp/README.md:3` — "decoupled from the existing DearPyGUI app"
- **Issue:** "DearPyGUI" — capitalize as in glossary? Glossary doesn't list it but the codebase uses both `DearPyGUI` (per repo CLAUDE.md / config) and `DearPyGui` (per docs). Inconsistent capitalization across the project but not a glossary issue.
- **Recommendation:** Not a glossary issue.

#### `webapp/README.md:14`
- **Issue:** "the brain-region framework, neuromodulator subsystem, plasticity gates, perception arc, or research runners" — all are project-internal feature names. Consistent with the rest of docs.
- **Recommendation:** No change.

#### `webapp/README.md:43–46` — KPI / activity feed
- **Issue:** Pure UI/UX terminology. No biological terms. No issue.

### `USER_GUIDE.md`

#### `USER_GUIDE.md:74`
- **Issue:** "**Brain-region framework** (`sim/regions.py`, opt-in): declarative multi-region simulations with `BrainRegion` + `RegionPathway`." — Project-internal feature names; consistent.
- **Recommendation:** No change.

#### `USER_GUIDE.md:251` — "**THALAMUS_TC_TRN** – Thalamocortical and reticular nuclei"
- **Issue:** "Thalamocortical" should be "thalamocortical relay nucleus" or "TC neuron" per glossary. "Reticular nuclei" should be "thalamic reticular nucleus (TRN)" per glossary. The current phrasing ("Thalamocortical and reticular nuclei") is *understandable* but reads like the profile contains two separate "thalamocortical" and "reticular" structures, when really it's the canonical TC + TRN complex.
- **Glossary:** `Thalamus (motor / relay nuclei)` and `Thalamic reticular nucleus (TRN)` entries.
- **Recommendation:** "Thalamocortical relay (TC) + thalamic reticular nucleus (TRN)".

#### `USER_GUIDE.md:253` — "**BASAL_GANGLIA_STN_GPE** – Subthalamic nucleus and globus pallidus externa"
- **Issue:** "globus pallidus externa" — glossary canonical is **"globus pallidus externus"** (masculine). "Externa" is grammatically incorrect Latin (should agree with "globus" which is masculine). Also the glossary entry uses "GPe" or "globus pallidus externus".
- **Glossary:** "GPe" entry (line ~57): "canonical: 'GPe' or 'globus pallidus externus'".
- **Recommendation:** **"globus pallidus externus"**. *Tier 1 fix — unambiguous typo.*

#### `USER_GUIDE.md:316–318` — STDP parameters
- **Issue:** "**A+, A−**: Magnitude of potentiation and depression." Cleanly aligned with glossary's STDP entry.
- **Recommendation:** No change.

#### `USER_GUIDE.md:319` — "**Reward-Modulated Plasticity / Three-factor learning: combines STDP with an external reward signal (e.g., dopamine).**"
- **Issue:** Excellent — uses glossary canonical "three-factor learning" + names dopamine explicitly.
- **Recommendation:** No change.

#### `USER_GUIDE.md:336` — "Cline & Haas (2008) model of activity-dependent neurite outgrowth"
- **Issue:** Glossary `Structural plasticity` entry mentions "Cline & Haas 2008 style" — consistent. No issue.

#### `USER_GUIDE.md:535` — "dopamine / NE / 5-HT modeling"
- **Issue:** Glossary canonical. No issue.

#### `USER_GUIDE.md:557–567` — "Phase B: BG-Style Action Selection"
- **Issue:** Architecture description uses "cortex → str_D1 / str_D2 → GPi / GPe → STN → thalamus → motor" — fine, glossary-aligned.
- **Recommendation:** No change.

#### `USER_GUIDE.md:561` — "D1 inhibits GPi (direct path)"
- **Issue:** "direct path" is glossary-canonical. No issue.

### `QUICKSTART.md`

#### `QUICKSTART.md:35` — "**Innate cue-following reflex** (direction-only, like phototaxis in real animals — replaces hand-coded heuristic)"
- **Issue:** "Phototaxis" is well-chosen biological framing (light-following innate behavior). Not a glossary entry but appropriate. No issue.
- **Recommendation:** No change.

#### `QUICKSTART.md:34` — "**Hippocampal place + goal cells** (self-organize from sensors, not coordinates)"
- **Issue:** Glossary `Place cell` entry (line ~472) carries a `[NEEDS-REVIEW]` flag noting that **project's place cells are sensor-driven, not strictly allocentric per O&N criteria**. QUICKSTART.md says "self-organize from sensors, not coordinates" — this is *more accurate* than calling them "place cells" without qualification, but still implies they meet the place-cell criterion.
- **Recommendation:** Optional clarification: "place-cell-like + goal-cell populations" or footnote referencing the catalog's allocentric criterion. **Low priority** — the project usage is consistent.

### `CONTRIBUTING.md`

#### `CONTRIBUTING.md:289` — "Additional neuron models (LIF, multi-compartment)"
- **Issue:** "LIF" = leaky integrate-and-fire — not in glossary explicitly but standard neuroscience term. No issue.
- **Recommendation:** No change.

#### `CONTRIBUTING.md:298` — "Additional plasticity rules"
- **Issue:** Already-aligned with glossary's `Plasticity rules` section.
- **Recommendation:** No change.

### `webapp/server.py` (docstrings + comments + UI strings)

#### `webapp/server.py:241–245` — preset description comment
- **Issue:** "Biology-grounded minimal flagship (R-pass + Cluster B + Cluster A + E)." — All project-internal naming. No biological-terminology issue.
- **Recommendation:** No change.

#### `webapp/server.py:180–192` — `final_quarter_mean_distance` and `finalQ`
- **Issue:** "finalQ" is project shorthand for the metric. Used in glossary'd findings docs but not glossary-listed itself. Naming is internally consistent.
- **Recommendation:** No change. Could be glossary-added.

### `webapp/static/app.js` (UI strings — user-visible)

#### `webapp/static/app.js:299` — "Manhattan distance to goal"
- **Issue:** "Manhattan distance" is correct. No issue.

#### `webapp/static/app.js:320` — "Reward (50-step moving avg)"
- **Issue:** UI string — clear. No issue.

#### `webapp/static/app.js:331` — "Agent visit heatmap — count time spent in each cell across the run."
- **Issue:** UI prose, no terminology issue.

#### `webapp/static/app.js:933` — `Sum_finalQ distribution across ${real.length} runs (green=flagship 4.08, yellow=baseline 5.88)`
- **Issue:** Hard-coded baselines (4.08, 5.88). These are correct as of 2026-04-28 but will drift as the flagship improves. The 2026-04-29 biology-grounded recipe is now 3.31. **Stale UI hardcode — not a glossary issue but a freshness concern.**
- **Recommendation:** Either accept the hard-coding (small UI cost) or thread the values through as config.

#### `webapp/static/app.js:1083–1085` — "vs flagship 4.08"
- **Issue:** Same — hardcoded "4.08" reference. Will go stale when 2026-04-29 biology-grounded sum 3.31 becomes the flagship.
- **Recommendation:** Same as above.

### `webapp/static/index.html` (UI strings — user-visible)

#### `index.html:138` — "**Goal beacon** — green dot with halo; emits intensity field falling off with distance"
- **Issue:** Clear UI string. "Beacon" is project-internal but well-defined here. No issue.

#### `index.html:142` — "**Beacon intensity** — green radial gradient around the goal (what beacon sensors detect)"
- **Issue:** Good — explains what the visualization corresponds to in the model. No issue.

#### `index.html:146` — "**Landmark** — blue ring; fixed reference cue (only relevant if run used `--landmarks`)"
- **Issue:** Good. "Landmark" is project usage; matches the perception-arc terminology. No issue.

#### `index.html:220–227` — preset descriptions (user-facing dropdown)
- **Issue:** "flagship — 4-cheats-closed (recommended)" — UI label for the dropdown. Phrasing is project-internal but consistent across docs. No issue.

#### `index.html:266` — "Inspect run details (per-trial finalQ, motor counts, phase stats)"
- **Issue:** "motor counts" is project-internal — refers to per-action firing of `motor_X` regions. Consistent with glossary's `motor_X` project shorthand.
- **Recommendation:** No change.

### `webapp/static/world.js` (UI strings + comments)

#### `world.js:7` — "Goal beacon (with intensity-falloff halo)"
- **Issue:** Same UI vocabulary. No issue.

#### `world.js:399` — `Agent distance from goal — rolling 100-step mean (yellow dots = goal moved)`
- **Issue:** Clear UI label. No issue.

#### `world.js:402` — comment "max Manhattan on 8x8 grid is 14"
- **Issue:** Geometric fact, not glossary. No issue.

#### `world.js:872` — `function actionName(idx) { return ["N", "E", "S", "W"][idx] ?? "?"; }`
- **Issue:** Action labels match the per-action pool naming in glossary (`X ∈ {N, E, S, W}`). No issue.

### `webapp/static/ui.js` (categorize function)

#### `ui.js:171` — `if (e.includes("baseline") || e === "default") return { category: "baseline", color: "#9aa3ad" };`
- **Issue:** "Baseline" is project-internal benchmark name. No issue.

#### `ui.js:177–178` — `category: "perception arc"`
- **Issue:** "Perception arc" is project-internal milestone label. No issue.

#### `ui.js:179` — `category: "PFC working memory"`
- **Issue:** "PFC" is glossary-canonical (line ~153). "Working memory" is glossary-canonical (cluster G label). Good.

#### `ui.js:181` — `category: "Phase C / curriculum"`
- **Issue:** Project milestone label. No issue.

#### `ui.js:183` — `category: "sleep replay"`
- **Issue:** "Sleep replay" — glossary `Replay (forward / reverse / awake)` entry covers this terminology. Good.

#### `ui.js:186` — `category: "Phase B refinement"`
- **Issue:** Project milestone label. No issue.

---

## Tier 2 — symbol-in-prose

### `CLAUDE.md:104`
- "`E_inh = -75mV`" — symbol in prose, units format. Mostly fine; glossary sometimes uses `−75 mV` (Unicode minus + space). Project allows `-75mV` in identifiers. *Not flagged.*

### `CLAUDE.md:108`
- "`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`" — symbol-style. Fine. *Not flagged.*

### `CLAUDE.md:109` — STDP soft-bound formula
- "`Δw_LTP = A_plus * (w_max - w) * exp(...)`" — uses Δ (capital delta), A_plus is a code identifier. Standard. *Not flagged.*

### `README.md:230`
- "**Reward-modulated STDP** (three-factor learning)" — STDP is glossary-canonical abbreviation. Good.

### `README.md:234` — "**NMDA receptors**: voltage-dependent Mg²⁺ block"
- "Mg²⁺" — glossary uses both "Mg2+" (in identifiers) and "Mg²⁺" (in prose). Both forms appear in repo. *Not flagged*; both are acceptable. (Glossary `NMDA receptor` uses Mg²⁺ in prose.)

### `README.md:241`
- "Realistic E_inh = −75 mV (Cl⁻ Nernst at 37°C)" — uses Unicode minus and Unicode superscript-2; matches glossary. Good.

### `USER_GUIDE.md:182`
- "**hh_q10_factor**" — single uniform Q10 — but as of 2026-04-25 the model uses **per-gate Q10** (m=3.0, h=1.5, n=1.5). UI prose only mentions a single Q10 factor.
- **Issue:** Stale documentation — UI panel description does not reflect the per-gate Q10 introduced 2026-04-25.
- **Recommendation:** Update to "**hh_q10_m / hh_q10_h / hh_q10_n** (per-gate Q10 scaling, replacing the older uniform Q10 — m=3.0, h=n=1.5 default)".

### `USER_GUIDE.md:347–354` — "**NMDA Receptors** ... Voltage-dependent Mg²⁺ block adds biological realism..."
- Symbol form Mg²⁺ — fine. No issue.

### `USER_GUIDE.md:611–615` — Troubleshooting "HH presets don't fire at 37°C" — uses canonical hh_q10_m / hh_q10_h / hh_q10_n. Aligns with glossary. Good.

---

## Tier 3 — identifiers (FLAGGED only)

These are code identifiers that appear in prose. Per task spec, only listed; no fix proposed.

### `CLAUDE.md:312` — `cp_synaptic_gain_modulator`
- Mentioned in past context ("shelved `cp_synaptic_gain_modulator`"). Identifier-level, no glossary issue.

### `CLAUDE.md:386–390` — `cp_plasticity_gain`, `set_plasticity_gate`, `target_type="plasticity_gate"`, `scope="gate:<name>"`
- Code identifiers — match glossary's mention of project plasticity-gate infrastructure. No fix needed.

### `CLAUDE.md:534` — `cp_d1_d2_sign`
- Mentioned in CHANGELOG too. Identifier-level. No glossary issue.

### `CLAUDE.md:563–570` — `pause_on_reward`, `plasticity_window_gate`, `_default_acetylcholine_config()`
- Code identifiers, glossary does not list them but they're project-specific.

### `CLAUDE.md:707` — `RuntimeState.actual_seed_used`
- Code identifier. Fine.

### `README.md:497–509` — Code example using `BrainRegion`, `RegionPathway`, `IZH2007_STRIATAL_MSN`, `IZH2007_THALAMIC_RELAY`
- All glossary-canonical project identifiers. No issue.

### `webapp/server.py:189` — `final_quarter_mean_distance`
- Project-specific metric name. Used consistently across server, app.js, world.js. Could be glossary-added.

### `webapp/server.py:241–278` — preset arrays use flag identifiers like `--enable-d1-d2-asymmetry`, `--enable-striatal-fsis`, `--enable-cluster-a-closed-loop`, `--enable-cluster-e-topography`
- Match the runner's CLI surface (and glossary mentions). No issue.

### `webapp/static/world.js:399, 404` — `recent_dist`
- Project metric name. Consistent with runner output. No issue.

### `webapp/static/app.js:606–609` — heavy field names: `motor_counts`, `distance_log`, `trajectory`, `spike_counts`, `place_cell_log`, `goal_cell_log`, `raw_phase1_motor_counts`
- All match runner output JSON keys. No glossary issue.

### `webapp/static/ui.js:172` — `e.includes("v3lateral") || e.includes("v3.1")`
- Filename-suffix experiment grouping. Internal naming convention. No issue.

---

## Reference-only — historical CHANGELOG (DO NOT FIX)

Per task spec, the CHANGELOG is treated as historical. Items listed for completeness but **NOT proposed for fix**:

1. `CHANGELOG.md:61` — "GPe is heterogeneous" — uses "PV+/PV−" canonical.
2. `CHANGELOG.md:66` — "Schultz98 / Schultz16" — citation style consistent with feature-catalog.
3. `CHANGELOG.md:79` — "topology-luck signal" — informal but clear.
4. `CHANGELOG.md:85` — "FSIs broadcast inhibition before agent commits to winner" — "FSI" used without "PV+" qualifier; **glossary `[NEEDS-REVIEW]` entry** flags this exact ambiguity. In CHANGELOG context, accept as historical.
5. `CHANGELOG.md:87` — "real TAN function requires tonic DA-driven plasticity" — "TAN" canonical; "tonic DA" canonical.
6. `CHANGELOG.md:88` — "step-order bug fix" — engineering, not biology.
7. `CHANGELOG.md:108` — "Phase-0 cortex_N/E activations reinforce cross-projections to all D1 pools, locking in motor bias the agent can't unlearn." — Glossary-aligned.
8. `CHANGELOG.md:114` — "**Stage 1: Goal-beacon perception** — replaces direct (gx, gy) goal cell access with 8 directional sensors detecting beacon strength × cosine alignment." — clear biological framing.
9. `CHANGELOG.md:115` — "Models innate phototaxis-like wiring." — appropriate biological analogy.
10. `CHANGELOG.md:116` — "Stage 2: Landmark-based place cell self-organization" — "place cell" used without `[NEEDS-REVIEW]` allocentric qualifier; matches project usage.
11. `CHANGELOG.md:121` — "PFC working memory region (Item 3)" — glossary-canonical.
12. `CHANGELOG.md:131` — "Real curriculum learning" — project terminology.
13. `CHANGELOG.md:133` — "**Sleep-replay infrastructure** — NREM trajectory replay (logged successful (place, goal) tuples) + REM random replay alternation. Mechanism works; current task structure doesn't reward consolidation." — NREM/REM are glossary-canonical.
14. `CHANGELOG.md:153` — "phase 1 finalQ 1.76 avg vs G9 baseline 6.74 (74% improvement)" — historical metric.
15. `CHANGELOG.md:160` — "AdEx preset library (`DefaultAdExParamsManager`): RS, FS, IB, CH, LTS, MSN, DOPAMINE" — all glossary-canonical phenotype labels.
16. `CHANGELOG.md:162` — "**Per-gate Q10 temperature scaling** for Hodgkin–Huxley" — glossary-aligned.
17. `CHANGELOG.md:174–193` — Brain-region framework + neuromodulator subsystem (Session E.1/E.2) — feature description aligned with glossary.
18. `CHANGELOG.md:283–286` — "**Inhibitory Reversal Potential** ... E_inh changed from -70mV to -75mV (matches Cl- Nernst potential at 37°C)" — Glossary-aligned. Note: should be "Cl⁻" but historical entry is fine.

---

## Items NOT flagged (intentional shorthand)

The following appear repeatedly across the audited files but are accepted as project-internal shorthand or already glossary-aligned:

- **`cortex_X`, `str_D1_X`, `str_D2_X`, `gpi_X`, `gpe_X`, `gpe_arky_X`, `stn`, `thal_X`, `motor_X`, `dopamine`, `str_FS_X`, `str_patch_X`, `dg`, `dg_fs`, `ca3`, `ca1`, `ec`, `pfc`, `place_cells`, `goal_cells`** — all glossary-listed project identifiers.
- **`IZH2007_*`, `HH_*`, `ADEX_*` preset names** — all glossary-canonical (per `Izhikevich 2007 Presets`, `Hodgkin-Huxley Presets`, `AdEx Presets` sections in CLAUDE.md).
- **`E_inh`, `E_exc`, `E_Na`, `E_K`, `E_L`, `g_Na`, `g_K`, `g_L`, `g_NaP`, `g_M`, `g_h`, `g_CaT`** — channel/conductance variable names, glossary-aligned.
- **`stp_U`, `stp_tau_d`, `stp_tau_f`** — Tsodyks-Markram parameters, glossary-canonical.
- **`stdp_w_max`, `A_plus`, `A_minus`, `tau_plus`, `tau_minus`** — STDP rule parameters, glossary-canonical.
- **`current_reward_signal`** — `[NEEDS-REVIEW]` glossary entry explicitly says auditor should not flag every use; only when biological distinctions matter. None of the prose in scoped files crosses that threshold.
- **`finalQ`, `sum_finalQ`, `recent_dist`** — project-internal metric names with stable usage.
- **"BG", "DA", "RPE", "STDP", "STP", "MSN", "FSI", "TAN", "GPe", "GPi", "STN", "PFC", "PPC", "DG", "CA1", "CA3"** — glossary-canonical abbreviations used throughout.
- **"flagship", "baseline", "smoke", "perception arc", "Phase A/B/C", "Cluster A/B/C/D/E", "cheat #5", "v3 lateral inhibition"** — project-internal milestone/variant names.
- **"motor exploration", "motor exploration noise", "silent-motor trap", "ε-greedy", "Boltzmann exploration"** — project-internal arc names; "ε-greedy" and "Boltzmann exploration" are RL-canonical.
- **Webapp UI strings**: "Live mode", "Trash", "Compare N", "Pause / Resume", "Latest", etc. — pure UX vocabulary, not biological.

---

## Top-3 most-impactful Tier 1 fixes

### 1. **`USER_GUIDE.md:253`** — `BASAL_GANGLIA_STN_GPE` description: "globus pallidus externa" → **"globus pallidus externus"**
**Severity:** Tier 1 — unambiguous Latin grammatical error in a user-facing reference doc. Glossary canonical entry confirms correct form.

### 2. **`USER_GUIDE.md:182`** (Hodgkin-Huxley parameters panel reference) — single `hh_q10_factor` → per-gate Q10
The UI guide describes a single uniform Q10, but the project uses per-gate Q10 (`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`) since 2026-04-25. The doc is *more than three days* stale on this critical fix that's prominently noted in CLAUDE.md and CHANGELOG. Update the parameter description.

### 3. **`README.md:434–445`** — disambiguate the legacy `--hippocampus` flag from `--enable-cluster-d-hippocampus`
Glossary's `[NEEDS-REVIEW]` entry explicitly notes this ambiguity. README mentions both forms but doesn't make the relationship explicit. Recommended one-line addition: "(legacy generic place_cells + goal_cells regions; for canonical DG/CA3/CA1 trisynaptic loop see `--enable-cluster-d-hippocampus`)".

---

## Policy questions

1. **Hard-coded numeric baselines in webapp UI** (e.g. `app.js:927–928` "baseline 5.88, flagship 4.08", and `app.js:1083` "vs flagship 4.08").
   The 2026-04-29 deterministic single-goal biology recipe just achieved sum 3.31 ± 0.74, which beats 4.08 by 19%. The hardcoded UI references will go stale once the new flagship is documented. Should the webapp:
   (a) Accept staleness (small UX cost; updated next webapp pass)
   (b) Read flagship/baseline values from a config file or `/api/info` endpoint
   (c) Compute "best in current data" dynamically (ignores documented flagship designation)
   This is engineering-side, not glossary-side, but worth flagging.

2. **`README.md:9` "Project status (2026-04-28)"** is one day stale (per CLAUDE.md showing 2026-04-29 catalog R-pass + cluster scaffolding). Roll forward, or accept that README updates lag CLAUDE.md by one cycle?

3. **`webapp/server.py:916`** — `"phase": "1 (research dashboard, runner launcher, findings browser)"` does not match `webapp/README.md` which describes Phase 1 + 2 + 2.5. Inconsistent — should the `info` endpoint return the more accurate "1+2+2.5" string?

4. **CLAUDE.md uses both "biologically-inspired" (line 9) and "biology-grounded" (line 411 etc.)** — pick one for project-wide consistency? "Biology-grounded" matches README.md and most recent docs.

5. **Glossary deferral question:** Should the glossary add entries for project-internal terms used heavily in audit-scope prose? Candidates: "silent-motor trap", "perception arc", "cheat #5", "flagship", "biology-grounded", "finalQ", "recent_dist", "beacon perception", "cue-following reflex", "sensed reward". Currently these are project terminology with stable usage but not in glossary, so future audits will keep flagging them.
