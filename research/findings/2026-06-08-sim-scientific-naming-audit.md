# Sim scientific-naming audit (biologist-readability)

**Date:** 2026-06-08
**Auditor task:** Check whether the simulator still uses internal jargon / shorthand
instead of true scientific names + descriptions, against the owner's bar:
*"if a computational biologist looked at the code, they'd understand what things are
doing and why, based on how things are named, and easily draw connections to real
biology/neurology."*

**Sources of truth used:** `references/glossary.md` (228 canonical entries, the project's
own canonical-terminology reference), `sim-catalog/references/feature-catalog.md`
(~323 entries, clusters A–Q), and the prior `references/structural-naming-audit.md`
(2026-04-29) which executed a comprehensive identifier audit and drove the Wave-1/2/3
renames.

**Scope covered:** all primary `sim/` files (`regions.py`, `bridge.py`, `kernels.py`,
`config.py`, `neuromodulators.py`, `enums.py`, `connectivity.py`, `visual_cortex.py`);
the secondary biological builders in `research/runners/` (`g11_bg_runner.py` BG +
hippocampus + cerebellum + SC-readout + visual builders; `text_minimal_isolation.py`
conversational builders). Read-only; no code changed, no jobs run.

---

## 1. Executive summary + VERDICT

**VERDICT: biologist-readable, with only minor nice-to-have gaps. The sim is in a
genuinely strong state on scientific naming — this is NOT a jargon-heavy codebase.**

A computational neuroscientist opening `sim/` would, in the overwhelming majority of
cases, immediately recognize what each biological entity is and map it to real
neurobiology from the name plus the adjacent docstring — and would usually also find a
primary-literature citation telling them *why*. The substrate (`sim/`) is the strongest
area and is **clean of experiment-codename jargon in actual identifiers** (no region,
field, function, or array is named "cluster_X", "tier_N", "cheat5", "G_v2", "N5", etc.).

Three structural reasons the sim scores well:

1. **There is a canonical glossary** (`references/glossary.md`) that explicitly defines a
   two-tier convention: a **canonical_name** for prose/comments and a stable
   **project_identifier** (`gpi_X`, `str_D1_X`, …) kept for backward compatibility with
   checkpoints/sidecars. So the terse region tags are *sanctioned, documented shorthand
   with an authoritative mapping*, not undocumented jargon. This is exactly the practice
   the owner's standard asks for.
2. **A prior dedicated audit already ran** (`references/structural-naming-audit.md`,
   2026-04-29) and its high-priority renames were applied: `dopamine`→`snc`,
   `str_FS_X`→`str_PV_FSI_X`, `pfc`→`dlpfc_wm`, `place_cells`→`sensor_place_readout`,
   `goal_cells`→`ppc_goal_input`, `dg_fs`→`dg_pv_basket`, `str_patch_X`→`str_striosome_X`,
   and the worst-case mislabeled gate `cortex_to_d1` (which actually gated D1+D2+patch)
   → `corticostriatal` / `corticostriatal_cross`. Old names are preserved as
   deprecation-warned aliases (`_DEPRECATED_REGION_NAMES` in `regions.py`).
3. **Citations are pervasive in the biological code.** Neuron presets, channel kernels,
   plasticity rules, and region builders routinely cite primary literature (Hodgkin &
   Huxley 1952, Jahr & Stevens 1990, Bi & Poo 1998, Brette & Gerstner 2005, Wilson &
   Kawaguchi 1996, Wang 2002, Lo-Wang 2006, Hubel & Wiesel 1962, Kandel chapters).

The genuine findings that remain are almost all **class D (missing biological context /
a docstring would help)** or low-severity **class B (decodable-but-terse)**. I found
**no class-C jargon leakage into the substrate's identifiers** and **no must-fix
biologist-would-be-actively-misled case** in `sim/`. The single recurring class-C-ish
pattern is cosmetic: a few module/file **docstring titles lead with the research gate
codename** ("G11: …", "Cluster K v1 — …") before giving the real biology in the next
line.

This is a "polish the last 5%," not a "fix a jargon problem," situation.

---

## 2. Where it's STRONG (class A — well-named + well-cited)

Being specific here, because an honest read is that most of the biological surface is
already exemplary:

- **`sim/kernels.py`** — model. Every kernel is the real mechanism with the canonical
  citation: `fused_hodgkin_huxley_dynamics_update` (per-gate Q10 m/h/n with the
  temperature-bug rationale), `fused_nmda_update_and_current` (Jahr & Stevens 1990 Mg²⁺
  block formula written out), `fused_stdp_weight_update` (Bi & Poo 1998 asymmetric
  window, soft-bound LTP/LTD), `fused_stp_decay_recovery` (Tsodyks-Markram u/x),
  `fused_homeostasis_update`, named extended currents `fused_hh_{m,CaT,h,NaP}_current_update`
  (M-current/Kv7, T-type Ca²⁺, I_h, persistent Na⁺). A biologist needs no extra context.
- **`sim/enums.py`** — every HH/Izh-2007/AdEx preset names the real cell type and cites
  the source: `CEREBELLAR_PURKINJE` (Khaliq 2003, De Schutter & Bower 1994),
  `STRIATAL_MSN` (Wilson & Kawaguchi 1996, with a catalog note on the KIR2/Kv2 bistability
  it does *not* yet capture), `DOPAMINE_SNC` (Drion 2011, Putzier 2009),
  `CORTICAL_FS_INTERNEURON` (Erisir 1999, Wang & Buzsáki 1996, "NO adaptation — defining
  feature of FS"), AdEx phenotypes (Brette & Gerstner 2005 Table 1). The conductance
  re-tunes carry the biophysical reasoning (e.g. STN g_NaP 0.8→0.15 per Bevan & Wilson
  1999 density).
- **`sim/neuromodulators.py`** — production rules and default configs are biologically
  annotated: `pause_on_reward` (BG TAN pause, Aosaki 1994 / Morris 2004),
  `from_region_firing_signed` (spiking-SNc negative-RPE dip, Schultz 1998),
  D1/D2 neuropeptide co-release configs (`dynorphin`/`substance_p`/`enkephalin`, McGinty
  PBR-160 ch 16, with KOR/DOR/NK-1 receptor targets). The ACh modulator is correctly
  scoped to its source population (`acetylcholine_tan`, with a docstring noting other ACh
  sources — basal forebrain Ch1–Ch4 — are *not* modeled).
- **`sim/regions.py`** — `BrainRegion`/`RegionPathway` fields are documented with the
  biology they model (exc_fraction 0.8 = cortical L2/3 per Markram 2015;
  `syn_reversal_potential_i_override` cites striatal MSN ~−60 mV gramicidin measurements
  and SNc lacking KCC2; `transmission_gate` cites Logiaco-Abbott-Escola 2021
  thalamocortical dynamical gating; per-region `enable_nmda` cites Wang 2002 NR2B). The
  `_DEPRECATED_REGION_NAMES` map documents each rename with its catalog/glossary
  justification.
- **`sim/config.py`** — biologically-meaningful fields cite sources inline (NMDA Jahr &
  Stevens; STP Tsodyks-Markram; STDP Bi & Poo / Song 2000; synaptic scaling Turrigiano;
  heterogeneity Marder & Goaillard 2006; OU background drive; GABA_A E_Cl −75 mV). Note
  especially the **honest self-documentation of `current_reward_signal`** as a signed
  scalar that "conflates two biologically distinct DA responses" — this is the project
  flagging its own host-computed shortcut in the exact spirit of the BRAIN-BASED standard.
- **`research/runners/g11_bg_runner.py`** — the BG builder header gives the
  direct/indirect/hyperdirect ASCII schematic, D1/D2 Gs/Gi receptor coupling (Kandel ch
  43), and refs (Frank 2005, Schroll & Hamker 2013). The SC-readout block names the
  oculomotor decision circuit correctly: `sel_X` (LIP/SC evidence accumulator, Wang 2002
  NMDA integrator), `commit_X` (SC burst / saccade-generator EBN, Lo-Wang 2006,
  catalog H.24/H.25), `commit_OPN` (omnipause neurons). Region names are now canonical:
  `snc`, `dlpfc_wm`, `ec`/`dg`/`dg_pv_basket`/`ca3`/`ca1`, `inferior_olive`, `stn`,
  `sensor_place_readout`, `ppc_goal_input`.
- **`research/runners/text_minimal_isolation.py`** — conversational regions map onto the
  catalog with citations: `wernicke` (G.13 Wernicke's area, auditory→semantic),
  `broca` (G.12, speech production + grammar, Kandel ch 55), `semantic_cortex` (ventral
  semantic interface), `motor_speech` (articulation), `ec_context` (D.01/D.02/D.11
  positional/time cells). Region-name strings are the full `language_input`/
  `language_output` (the `lang_*` forms are only kwarg names, not identifiers).
- **`sim/visual_cortex.py`** — V1 simple/complex with Gabor RFs, phase-pooling, the
  retina→V1→V2→IT hierarchy, Hubel & Wiesel 1962 / Felleman & Van Essen 1991 / Tanaka
  1996 IT. Content is fully canonical (only the docstring *title* leads with "Cluster K
  v1 —"; see §3/§4).

---

## 3. Findings, ranked by impedance to biological understanding

Ranked most- to least-impeding. All are nice-to-have; none are must-fix. **B** =
cryptic-but-decodable shorthand, **C** = project/experiment jargon where a biological
label belongs, **D** = correct + well-named but missing the docstring/citation that
shows *why*.

### Module/docstring titles + experiment-codename comment tags

| location (file:line) | current name / description | cat | what a biologist would misread | suggested fix |
|---|---|---|---|---|
| `sim/visual_cortex.py:1` | docstring title `"Cluster K v1 — Visual cortex utilities ..."` | C | "Cluster K v1" is a project catalog tag, not biology; harmless because the rest of the line says "Visual cortex (Hubel-Wiesel 1962...)" | Lead with the biology: `"Visual cortex utilities (V1 simple/complex, Gabor RFs) — Hubel & Wiesel 1962 ... [project: catalog Cluster K]"`. Pure docstring edit, runner-side, zero blast radius. |
| `research/runners/g11_bg_runner.py:1` | docstring title `"G11: Basal ganglia action selection module."` | C | "G11" is a research-gate codename; immediately followed by the real biology so a reader is not misled | Optional: reorder to `"Basal ganglia action-selection module (research gate G11)."` Cosmetic. |
| `sim/bridge.py`, `regions.py`, `config.py`, `neuromodulators.py` — many comment tags `"Cluster C v2 (2026-04-29): ..."`, `"Cluster B.1 ..."`, `"Cluster G v2 ..."`, `"cheat-5 option-1"` | C (comments only) | These are **provenance/date tags in comments**, each followed by the actual biological explanation. They are lab-notebook breadcrumbs, not identifiers. | Acceptable as-is (they aid trail-following). If desired, a one-line key mapping "Cluster A–Q → catalog cluster" already exists in the glossary; nothing in code needs changing. **No identifier carries these tags.** |

### Regions / pathways (runner-side)

| location | current name | cat | what a biologist would misread | suggested fix |
|---|---|---|---|---|
| `g11_bg_runner.py:1535` etc. | `sel_{N,E,S,W}` (+ `sel_FS_X`) | B | "sel" is terse; full meaning ("evidence-accumulation selection pool, LIP/SC-like, Wang-2002 NMDA integrator") lives in the adjacent comment, not the name | Decodable in context; keep. If ever renamed, `accum_X` or `lip_accum_X` is closer to the cited biology. Low priority. |
| `g11_bg_runner.py:1059` | `motor_speech` | A/D | clear (articulatory motor output); no citation on the region itself | Fine; could cite Kandel ch 55 articulation in the region docstring (nice-to-have). |
| `g11_bg_runner.py` | `motor_{N,E,S,W}`, `cortex_{N,E,S,W}` | B (acceptable shorthand) | `cortex_X` is generic M1; `motor_X` is an abstract per-action pool, not literally α-motoneurons | Already flagged in the prior audit as "defer until other cortical regions force disambiguation (PMd/pre-SMA/S1)". The glossary documents `cortex_X` ≈ M1 and `motor_X` ≈ abstract α-MN pool. Keep; revisit only when more cortex is added. |
| `g11_bg_runner.py` | `motor_FS_{N,E,S,W}` | D | a cortical-style FS basket microcircuit on the *motor output* pool has no direct biological referent (real motor WTA is spinal Renshaw recurrent inhibition) | The code comment already says this is an opt-in WTA microcircuit. Nice-to-have: a one-line docstring noting "biologically this stands in for spinal Renshaw recurrent inhibition (Kandel ch 35), not a cortical basket". |

### Neuron models / receptors / channels

| area | assessment |
|---|---|
| HH presets, Izh-2007 presets, AdEx presets (`enums.py`) | **Class A across the board.** Each names the real cell type + cites the source. No findings. |
| Channel currents (`kernels.py`) | **Class A.** Named ion currents (Na/K/L/M/CaT/I_h/NaP), Jahr & Stevens Mg²⁺ block. No findings. |
| Receptors | Modeled generically (AMPA = fast exc conductance, GABA_A = `E_inh`). The glossary explicitly records this scope ("project does not track AMPA explicitly as a named subtype"). Not a naming defect — a documented modeling-scope choice. |

### Plasticity / neuromodulators

| area | assessment |
|---|---|
| STDP, STP, homeostasis, eligibility-trace, three-factor (`kernels.py`/`config.py`) | **Class A**, all cite canonical sources. |
| Neuromodulator production rules + default configs (`neuromodulators.py`) | **Class A.** dopamine/ACh-TAN/dynorphin/substance_p/enkephalin all biologically annotated with receptor targets + refs. |
| `current_reward_signal` (`config.py:210`) | **Class A (honest).** Already self-documented as a host-computed signed-scalar DA shortcut that conflates Schultz-98 activation vs depression; points to the `dopamine` neuromodulator + `reward_aversive_scale` as the real-biology path. This is the BRAIN-BASED-standard shortcut flagged in-code, exactly as desired — not a naming defect. |

### Config / bridge state fields

| area | assessment |
|---|---|
| `cp_*` GPU arrays (`bridge.py`) | Transparent: `cp_gating_variable_{m,h,n}`, `cp_hh_CaT_{m,h}`, `cp_hh_NaP_activation`, the 9 `cp_izh_*` params, `cp_d1_d2_sign`, `cp_syn_reversal_potential_i_per_neuron`, `cp_graded_lateral_M`. No findings. |
| `graded_lateral_*` config block (`config.py`) | **Class A.** Documented as the analog pre-spike LGN/retina whitening (variance equalization) with the design-doc reference. |

---

## 4. Experiment-jargon-leakage finding (is the substrate clean?)

**Yes — the substrate is clean of experiment jargon in identifiers.** This is the most
important reassurance for the owner's standard.

- A direct identifier scan for regions/fields/functions/arrays named with
  `cluster|tier|cheat|gate_|g_v2|n5|n6` returns **zero hits in `sim/` and zero
  region-name strings in the runners**. No biological entity is *named* after an
  experiment codename.
- The matches that exist are: (a) **comment provenance tags** ("Cluster C v2
  (2026-04-29): …") that are always followed by the real biology; (b) **two docstring
  titles** that lead with a codename ("G11: …", "Cluster K v1 — …") before the biology;
  (c) **legitimate domain vocabulary** that merely shares a substring — `next_tier`/
  `arch_for_tier` in `auto_growth.py` is the *vocabulary-tier ladder* (a real concept in
  the conversational scaling work), and synapse `tiering` is the standard CS storage-tier
  term, not the "Tier 1/2.1" experiment labels; (d) **test names** (`test_cluster_f_*`),
  which are bookkeeping, not substrate.
- The research-gate / cluster / "G v2.5" / "Tier" / "cheat #5" vocabulary is therefore
  **confined to lab-notebook bookkeeping** (findings docs, file/runner titles, comments)
  — which is acceptable, expected research shorthand — and has **not leaked into the
  biological code in a way that obscures the biology**. The one cosmetic exception is the
  two docstring titles in §3.

---

## 5. Prioritized remediation recommendation (top items)

Ranked by readability gain per unit effort. **All are low-risk.** None require a rename
in protected `sim/` (the substrate's identifiers are already canonical); the only `sim/`
touches suggested are **docstring/comment-only** edits, which do not affect imports or
checkpoints. Genuine `sim/` *renames* are **not** recommended — they would risk
breaking imports/sidecars/checkpoints for negligible benefit since the glossary already
documents the mapping.

1. **(`sim/visual_cortex.py:1`, docstring-only)** Reword the module title to lead with the
   biology, demoting "Cluster K" to a parenthetical. Highest single readability win; zero
   risk. *(sim/, but comment-only — no byte-level behavior change.)*
2. **(`research/runners/g11_bg_runner.py:1`, docstring-only)** Same treatment for the
   "G11:" title. Runner-side, zero risk.
3. **(runner-side, docstring add)** Add a one-line "biological referent" note to
   `motor_FS_X` (spinal Renshaw recurrent inhibition, Kandel ch 35) and to `motor_speech`
   (Kandel ch 55 articulation). Class-D gap-fill.
4. **(docs, optional)** In the glossary or a short `references/` note, add the explicit
   "research-codename → catalog-cluster" key (e.g. "G11 = BG action-selection;
   Cluster K = visual cortex; Tier 1/2.1 = vocab-scaling milestones") so a newcomer can
   decode the comment provenance tags without hunting findings docs. Pure documentation.
5. **(no action — record as intentional)** Keep `sel_X`, `cortex_X`, `motor_X`,
   `gpi_X`, `str_D1_X` etc. as the sanctioned `project_identifier` shorthand the glossary
   already defines. Renaming them is **not** recommended (large blast radius across
   pathways/tests/sidecars/checkpoints; the glossary mapping already satisfies the
   biologist-readability bar).
6. **(no action — already correct)** The prior audit's high-priority renames
   (`snc`, `dlpfc_wm`, `str_PV_FSI`, `dg_pv_basket`, `corticostriatal` gate, etc.) are
   applied; deprecation aliases are in place. Nothing to redo.

**Bottom line for the owner:** the sim already meets the "a computational biologist could
read it and map it to real biology" bar. The remaining work is two docstring-title
rewordings and a handful of optional "why" citations — polish, not a jargon cleanup. The
substrate carries no misleading biological names and no experiment-codename identifiers.
