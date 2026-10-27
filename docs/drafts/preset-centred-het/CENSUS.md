# Census: per-region neuron presets overwritten by globally-centred parameter heterogeneity

Date 2026-10-20. Branch `claude-build/preset-centred-het-1020`. Read-only census, taken on the unedited bridge
(`sim/bridge.py` at 1f8d30fc7) before the `preset_centred_parameter_heterogeneity` flag was added.

## The defect

`CoreSimConfig.enable_parameter_heterogeneity` defaults True (`sim/config.py:190`). During
`SimulationBridge._initialize_simulation_data`, `_apply_per_region_neuron_types` writes each region's Izhikevich
preset (`sim/enums.py` `DefaultIzhikevichParamsManager.PARAMS`) or its HH overrides. Then
`_apply_parameter_heterogeneity` overwrites `cp_izh_a`, `cp_izh_b`, `cp_izh_d_increment`, `cp_izh_C` (and for HH
`cp_hh_C_m`, `cp_hh_g_Na_max`, `cp_hh_g_K_max`, `cp_hh_g_L`) for every neuron with draws whose centre is the GLOBAL
`cfg.izh_*_val` / `cfg.hh_*` value (`_get_default_heterogeneity_distributions`: the RS cortical defaults a 0.03,
b -2, d 100, C 100; HH L5 defaults C_m 1.0, g_Na 50, g_K 12, g_L 0.1). The opt-in region-scoped redraw
(`_overwrite_region_scoped_parameter_heterogeneity`) and the global-OFF per-region mask
(`BrainRegion.enable_heterogeneity=True`) use the same globally-centred distributions. Only k, vr, vt, vpeak and
c_reset (Izhikevich) and E_Na, E_K, E_L (HH) keep the preset.

So every non-RS preset region in a heterogeneity-on bridge runs as an RS-centred hybrid: its own k/vr/vt/vpeak/c with
RS a/b/d/C. Sign flips are common: b is +1 (GPe), +2 (GPi), +5 (hippocampal pyramidal), +15 (thalamic relay) in
the presets and about -2 after the overwrite.

First noted 2026-06-18 (`research/findings/2026-06-18-organ-lift-homeo-generalize-derisk.md`, about line 128) and
worked around in that one organ by turning heterogeneity off; never fixed or logged. Found again in the adversarial
review of the bg-tonic-loop packet draft (2026-10-20).

## Method

1. Static scan of every `.py` under `research/runners/`, `webapp/` and `sim/` (excluding `sim/bridge.py`,
   `sim/config.py`, `sim/enums.py`) for per-region type assignment (`izh_neuron_type=` with a value, HH
   `hh_*_override=`) and for `enable_parameter_heterogeneity` assignments. 133 files set per-region Izhikevich
   types; none outside `tests/` sets HH overrides.
2. Reachability: for each `on_by_default: YES` row of `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`, the import closure of
   the row's organ modules, intersected with the builders from step 1.
3. Dynamic confirmation (numpy, one seed): a recorder wrapping `_initialize_simulation_data` built each candidate
   builder and dumped, per region, the preset vs the effective (post-heterogeneity) mean of a/b/d/C.
4. Attempted end-to-end: the default `/api/brain-chat` brain (regression-battery worker, all 26 probe turns) under
   the recorder at the 4 GB memcap. It was OOM-killed (rc 137) during the base build, before any turn ran:
   `rf_phasor_composer._build_rf_bridge` had built 53,471 bridges by then. The base-build records it did capture are
   in table B. The lazily built organs were therefore confirmed per builder (step 3), not through the live turn.

A region whose preset is RS (`IZH2007_RS_CORTICAL_PYRAMIDAL`) is unaffected in substance: its preset equals the
global centre. Only non-RS presets are listed.

## A. Production-default paths affected (ledger on_by_default YES; confirmed by build + dump)

Effective = the region mean after the build, one seed (selector seed 42, others as noted). The draws are
RS-centred, so effective a/b/d/C are about 0.03 / -2 / 100 / 100 whatever the preset.

| Ledger row(s) | Organ module -> builder | Affected regions (preset) | Preset a / b / d / C | Effective a / b / d / C |
|---|---|---|---|---|
| bg-action-selection | `bg_action_selection_production_organ` -> `_vocal_action_selector_gate.build_selector_bridge` | `selector_stn` (STN_BURST) | 0.005 / -1 / 75 / 80 | 0.029 / -2.16 / 100.9 / 100.8 |
| | | `str_d1_*` (MSN_D1) | 0.01 / -20 / 150 / 50 | 0.032 / -1.99 / 106.1 / 98.8 |
| | | `str_d2_*` (MSN_D2) | 0.01 / -25 / 180 / 50 | 0.034 / -1.93 / 105.4 / 98.3 |
| | | `gpe_*` (GPE_PACEMAKER) | 0.05 / 1 / 20 / 60 | 0.031 / -1.93 / 100.6 / 97.9 |
| | | `gpi_*` (GPI_OUTPUT) | 0.05 / 2 / 25 / 60 | 0.030 / -1.90 / 110.9 / 101.7 |
| | | `thal_*` (THALAMIC_RELAY) | 0.01 / 15 / 10 / 200 | 0.031 / -1.70 / 99.8 / 98.4 |
| | | `selector_reset`, `str_fsi_*`, `commit_fs_*` (FS) | 0.2 / -2 / 25 / 20 | 0.029 / -1.92 / 98.6 / 101.9 |
| affect-marker-spiking-wta, affect-drives-response | `webapp/affect_drives_chat` -> `_affect_marker_wta_derisk._build_bridge` (seed 42 / 43) | `aff_val_fsi*`, `aff_aro_fsi*` (FS) | 0.2 / -2 / 25 / 20 | 0.029 / -2.02 / 99.3 / 99.4 |
| da-write-gain-spiking, da-gated-encoding | `webapp/da_encoding_drives_chat` -> `_da_write_gain_spiking_derisk._build_write_gain_bridge` | `write_gain` (HIPPO_PYRAMIDAL) | 0.01 / 5 / 50 / 100 | 0.030 / -2.01 / 103.2 / 100.8 |
| comprehension-monitor, comprehension-learned-animacy-cue, comprehension-learned-verb-selects, other-repair | `comprehension_production_organ` -> `_phaseB_multicue_competition_spiking_derisk.SpikingRoleCompetition` | `sel_FS_*` (FS) | 0.2 / -2 / 25 / 20 | 0.029 / -2.12 / 101.9 / 98.2 |
| prospective-memory | `prospective_memory_production_organ` -> `_pmem_intention_latch_derisk.ProspectiveMemory` family | `cortex_ctx` (HIPPO_PYRAMIDAL) | 0.01 / 5 / 50 / 100 | 0.031 / -2.00 / 98.7 / 98.8 |
| anaphora-wm | `multi_turn_agent.MultiTurnAgent._build_wm` -> `content_selection_spiking.SpikingLoopContextBuffer` -> `build_loop_wm_bridge` | `cortex_ctx` (HIPPO_PYRAMIDAL) | 0.01 / 5 / 50 / 100 | 0.032 / -1.93 / 101.7 / 100.8 |
| selective-attention-biased-competition | `multi_turn_agent.MultiTurnAgent` (the default brain is built with `use_multiturn=True`) -> `biased_competition_buffer.BiasedCompetitionContextBuffer` | `cortex_ctx` (HIPPO_PYRAMIDAL) | 0.01 / 5 / 50 / 100 | 0.031 / -1.97 / 99.2 / 98.9 |
| | | `sel_FS_*` (FS) | 0.2 / -2 / 25 / 20 | 0.034 / -1.92 / 101.4 / 99.5 |
| curiosity-followup, curiosity-graded-novelty | `curiosity_production_organ` -> `_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge(..., per_region_heterogeneity=True)` | `striosome_value` (MSN_D1). Global heterogeneity is OFF here, but the region opts in through the per-region mask and the region-scoped redraw, both RS-centred | 0.01 / -20 / 150 / 50 | 0.032 / -2.04 / 94.2 / 100.5 |

Reached from default-on rows, affected when built, but not confirmed as built on the default turn (import depth >= 2,
or behind a sub-flag):

- `_affect_marker_wta_derisk._build_bridge(seed+1009, n, "d6focus")` (FS `d6focus_fsi*`, same builder as the
  affect marker), built by `d6_multiref_wm_production_organ._focus_wta_rates` only when `BRAIN_MULTIREF_FOCUS_BIND`
  is set (anchor `_MULTIREF_FOCUS_BIND_DEFAULT_ON = False`).
- `lexicon_spiking_frame_category.build_circuit` (FS `IN*`: effective 0.030 / -1.83 / 94.8 / 102.9), built by the
  d6 organ only when `BRAIN_LEARNED_REFERENT_LEXICON` is set (default OFF).
- `affect_learned_vocabulary.build_circuit` (FS `IP*`: same effective values, same builder pattern), built by
  `affect_production_organ` only when `BRAIN_AFFECT_LEARNED_VOCAB` is set (default OFF).
- `content_selection_spiking.build_dlpfc_context_bridge` (HIPPO `dlpfc_wm`: effective 0.031 / -1.98 / 102.7 /
  100.5). Imported by the base agent modules; not traced to a default-turn build.
- `_da_composer_salience_cleanup_derisk._build_snc_bridge` (DOPAMINE `snc`, preset 0.01 / 1 / 5 / 100, effective
  0.035 / -2.05 / 96.9 / 104.1). Depth 2 from da-gated-encoding.
- `_riii_ca3_coincidence_completion_derisk._build` (FS), depth 2 from self-initiated-utterance; and
  `_emerge12_stageB2_bridge_tm_derisk.build_bridge_from_tm` (FS), depth 2 from vision-identity-spiking-hmax.
  Statically affected, not built here.

## B. Production builds that are NOT affected

Records from the default-brain base build (step 4) plus the merged organs:

| Builder | Heterogeneity | Per-region types | Why unaffected |
|---|---|---|---|
| `onebrain_merge_production.ensure_built` (merged pool: `_spiking_expectation_rpe_derisk.build_expectation_circuit`, `_affective_world_model_derisk.build_world_model_circuit`) | OFF | FS `patient_expected`, `pred_pos`, `pred_neg` | effective == preset (0.2 / -2 / 25 / 20) |
| `rf_phasor_composer._build_rf_bridge`, `one_brain_composer.build_coresident_bridge`, `attributed_parser`, `_D_sparse_heteroassoc.build`, `_d3_spiking_attractor_derisk.build_fswta_score_bridge` | ON | none | no region preset (see C2 for the trait-typed variant) |

## C. Other research builders (static scan)

Classified by the `enable_parameter_heterogeneity` assignments in the file itself. "Default" means the file never
assigns it, so the bridge runs with heterogeneity ON unless its config comes from another module.

- **Non-RS presets, heterogeneity default/ON (48 files, all affected when built):** `_affect_marker_wta_derisk`,
  `_b1_v1_selforg_columnar_wta_derisk`, `_b1_v1_selforg_onbridge_derisk`, `_da_composer_salience_cleanup_derisk`,
  `_da_write_gain_spiking_derisk`, `_emerge11_stageB_wta_dap_derisk`, `_emerge12_stageB2_bridge_tm_derisk`,
  `_emerge41_fs_wta_kwinners_derisk`, `_laneD_visual_identity_homeostasis_gate`, `_learned_spiking_segmentation_derisk`,
  `_msn_synaptic_drive_probe`, `_phaseB_biased_competition_wta_derisk`, `_phaseB_multicue_competition_spiking_derisk`,
  `_phaseB_retinal_cortex`, `_pmem_intention_latch_derisk`, `_replay_cortical_consolidation_gate_v2/_v3/_v5`,
  `_replay_dg_pattern_separation_bridge/_gate/_homeostat/_popsetpoint`, `_riii_ca3_coincidence_completion_derisk`,
  `_riii_ca3_cortical_episodic_wta_derisk`, `_transitive_ordinal_map_derisk`, `_vocal_action_selector_gate`,
  `_vocal_gateb_stage1_selector`, `affect_learned_vocabulary`, `biased_competition_buffer`, `content_selection_spiking`,
  `gated_compose_bg_genuine_demo`, `gated_compose_bg_learned_demo`, `hebbian_plausibility_organ`,
  `lexicon_spiking_frame_category`, `nmda_compositional_consolidation`, `sc_approach_td_probe`, `sc_map_orienting_probe`,
  `sc_n5_rpe_probe`, `sc_orienting_production_organ`, `snc_pavlovian_probe`, `snc_stageb_critic_probe`,
  `snc_stageb_critic_probe_place`, `text_minimal_isolation`, `text_visual_grounding`; explicitly ON:
  `_laneC_source_provenance_opponent_derisk`, `_rubicon_delayed_credit_derisk`, `_stageA_full_integration_derisk`,
  `_wta_afferent_winner_homeostat_derisk`.
- **Conditional (a parameter or module constant decides; 9 files):** `_affect_appraisal_emotion_reappraisal_derisk`,
  `_affective_world_model_derisk`, `_curiosity_reward_omission_veto_derisk`, `_curiosity_seek_learn_onbridge_derisk`,
  `_limbic_core_rpe_battery_derisk`, `_magnitude_preserving_plateau_readout_derisk`,
  `_shortcut9_trace_conditioning_bridge_probe`, `_spiking_expectation_rpe_derisk`, `nav_conv_merged_bridge` (its
  `td_striosome` MSN-D1 critic opts in through `enable_heterogeneity=True` with global OFF: affected).
- **Non-RS presets, heterogeneity OFF (28 files, unaffected):** e.g. `g11_bg_runner`, `coincidence_n9_derisk`,
  `n9_*`, `snc_stageb_critic_probe_navfaithful`, `_vocal_action_credit_gate_v13_tonic_output`, the affect-ladder and
  laneC source-monitor gates.
- **RS-only presets (45 files):** unaffected in substance.

## C2. Related variants the flag does not cover

1. **Trait-typed populations.** With `num_traits > 1` (the `CoreSimConfig` default is 5), the Izhikevich init
   assigns each neuron one of every defined IZH2007 preset by trait (`sim/bridge.py` `use_trait_based`), and the
   global draw then overwrites a/b/d/C with RS-centred values. This hits neurons outside any typed region, including
   production base-brain bridges (`_d3_spiking_attractor_derisk.build_fswta_score_bridge`,
   `_D_sparse_heteroassoc.build`). The flag keeps these neurons on the global centre, on purpose: the requirement is
   that neurons with no REGION preset stay identical with the flag on and off.
   Follow-up 2026-10-27 (research/stable-mechc2): `cfg.trait_centred_parameter_heterogeneity` and `SIM_TRAIT_CENTRED_HET=1`
   (both default OFF) centre these neurons on their own trait preset, on a neuron set disjoint from the region flag's. On dev
   seed 7 it removes the association graph's cells below -110 mV (110 down to 4) and changes the replies and the organs' own
   gates for the worse: `research/findings/2026-10-27-trait-centred-het-dev7.md`.
2. **Default type vs global centre.** `cfg.izh_*_val` / `cfg.hh_*` are not synced with `default_neuron_type_izh` /
   `default_neuron_type_hh`. A non-RS default Izhikevich type, or any HH default type other than the L5 values (for
   example `HH_EXCITATORY_DEFAULT_LEGACY`: g_Na 120, g_K 36, g_L 0.3), is recentred on the global values whenever
   heterogeneity is on.
3. **HH per-region overrides.** `hh_C_m_override` / `hh_g_Na_max_override` / `hh_g_K_max_override` /
   `hh_g_L_override` are overwritten the same way: on a heterogeneity-on test bridge, an override of g_Na 88 reads
   57.5 mean (lognormal centred on 50). No builder in `research/runners/`, `webapp/` or `sim/` sets HH overrides
   today (only tests, all with heterogeneity OFF), so no production path is affected. The flag applies the same
   preset-centring rule to these four HH parameters.

## Artifacts

The census scripts (recorder, static scan, reachability closure) and their outputs were run from the session
scratchpad and are not committed. The facts above are reproducible with the warning added in this branch: a build
that overwrites region presets now logs one WARNING ("parameter heterogeneity is centred on the global defaults
... and overwrites the presets of N region(s): ...") naming the regions and the parameters.

## Opting in (after this branch)

An organ keeps its presets by setting the flag on the config it builds, before `_initialize_simulation_data`:

```python
cfg.enable_parameter_heterogeneity = True            # unchanged: heterogeneity stays on
cfg.preset_centred_parameter_heterogeneity = True    # jitter a/b/d/C around each region's own preset
```

or, for a builder that takes config overrides (`build_selector_bridge(seed, core_config_updates=...)`):

```python
build_selector_bridge(seed, core_config_updates={"preset_centred_parameter_heterogeneity": True})
```

The draws, their order and the RNG state after the build are unchanged; only region-preset neurons move (value =
ratio x preset, ratio = sample / centre). Flipping it on for an organ changes that organ's dynamics, so each flip
needs its own six-seed gate; no organ is flipped in this branch.
