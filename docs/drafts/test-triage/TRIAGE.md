# Full-suite test triage — 2026-10-01 (runbook section 5, idle)

Run: `SIM_NO_PROVENANCE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider --continue-on-collection-errors --ignore=tests/test_game_pause_resume.py --junitxml=_scratch/suite.xml -rfE tests` from `.claude/worktrees/draft-test-triage` (research/draft-test-triage, from main).

Result: **175 failed, 4 errors, 7258 passed, 497 skipped, 3 xfailed** — 13587.5s (3:46:27). The 179 failed/errored tests group into **76 distinct first-error lines** (appendix: `grouped-first-error-lines.txt`).

## Classification (counts sum to 179)

| class | tests | meaning |
|---|---:|---|
| ENVIRONMENT | 66 | GPU hidden (CUDA_VISIBLE_DEVICES empty per section 5), container (no systemd/cupy/torch/dearpygui/rsync), or worktree artifact (gitignored `data/` not checked out; hardcoded /home/dant123/Projects/sim paths absent on S). |
| POLLUTION | 33 | Single-process full-suite artifact: import-time mutation of `concept_pool_demo`'s module-level vocabs by other runners (v2..v5/set2..set5/multitag_eval_v17/compose_engram_demo_v2 all do `cpd_v1.NOUN_VOCAB = ...`). |
| REAL | 28 | Verified FAIL in isolation (or fails on the main checkout too). |
| UNVERIFIED | 52 | Grouped by first error line only; no isolation run. |

## ENVIRONMENT (66)

GPU hidden (CUDA_VISIBLE_DEVICES empty per section 5), container (no systemd/cupy/torch/dearpygui/rsync), or worktree artifact (gitignored `data/` not checked out; hardcoded /home/dant123/Projects/sim paths absent on S). Not code regressions.

- **[21]** FileNotFoundError: [Errno 2] No such file or directory: '/home/dant123/Projects/neural-simulator/.claude/worktrees/draft-test-triage/data/corpus/tinystories.txt'
    - `tests.test_comprehension_learned_animacy_cue::test_flag_on_extends_coverage_load_bearing`
    - `tests.test_comprehension_learned_animacy_cue::test_flag_on_lesioned_reverts_byte_identical`
    - `tests.test_comprehension_learned_animacy_cue::test_learned_lexicon_abstains_off_graph`
    - `tests.test_comprehension_learned_animacy_cue::test_learned_lexicon_classifies_held_out_word`
    - `tests.test_comprehension_learned_animacy_cue::test_learned_lexicon_lesion_collapses_to_abstain`
    - `tests.test_comprehension_learned_animacy_cue::test_moat_genuinely_oov_still_abstains_flag_on`
    - `tests.test_comprehension_learned_animacy_cue::test_unset_now_defaults_to_on`
    - `tests.test_comprehension_learned_cues_joint::test_held_noun_coverage_extends_with_both_on`
    - `tests.test_comprehension_learned_cues_joint::test_held_verb_coverage_extends_with_both_on`
    - `tests.test_comprehension_learned_cues_joint::test_joint_doubly_held_out_oov_naming_shrinks_to_empty`
    - `tests.test_comprehension_learned_cues_joint::test_lesion_animacy_only_reverts_noun_spares_verb`
    - `tests.test_comprehension_learned_cues_joint::test_lesion_both_full_revert`
    - `tests.test_comprehension_learned_cues_joint::test_lesion_verb_only_reverts_verb_spares_noun`
    - `tests.test_comprehension_learned_cues_joint::test_moat_holds_both_on`
    - `tests.test_comprehension_learned_verbselects_cue::test_flag_on_extends_coverage_load_bearing`
    - `tests.test_comprehension_learned_verbselects_cue::test_flag_on_lesioned_reverts_byte_identical`
    - `tests.test_comprehension_learned_verbselects_cue::test_learned_lexicon_abstains_off_graph`
    - `tests.test_comprehension_learned_verbselects_cue::test_learned_lexicon_classifies_held_out_verb`
    - `tests.test_comprehension_learned_verbselects_cue::test_learned_lexicon_lesion_collapses_to_abstain`
    - `tests.test_comprehension_learned_verbselects_cue::test_moat_genuinely_oov_still_abstains_flag_on`
    - `tests.test_comprehension_learned_verbselects_cue::test_unset_now_defaults_to_on`

- **[13]** ModuleNotFoundError: No module named 'cupy'
    - `::tests.test_determinism`
    - `tests.test_bridge_text_io::test_set_token_drive_activates_correct_neurons`
    - `tests.test_bridge_text_io::test_set_token_drive_deterministic`
    - `tests.test_fast_spike_reset::test_fast_spike_reset_handles_high_firing`
    - `tests.test_fast_spike_reset::test_fast_spike_reset_handles_no_firings`
    - `tests.test_fast_spike_reset::test_fast_spike_reset_numerically_equivalent_to_legacy[123]`
    - `tests.test_fast_spike_reset::test_fast_spike_reset_numerically_equivalent_to_legacy[42]`
    - `tests.test_fast_spike_reset::test_fast_spike_reset_numerically_equivalent_to_legacy[7]`
    - `tests.test_regions::test_freeze_thaw_cycle`
    - `tests.test_regions::test_frozen_pathway_blocks_stdp_weight_changes`
    - `tests.test_regions::test_set_plasticity_gate_updates_gain`
    - `tests.test_regions::test_thawed_pathway_allows_stdp_weight_changes`
    - `tests.test_surrogate_grad::test_module_loads`

- **[7]** ModuleNotFoundError: No module named 'dearpygui'
    - `::tests.test_state_manager`
    - `tests.test_figure_export.TestFigureExport::test_comparison_figure`
    - `tests.test_figure_export.TestFigureExport::test_comparison_figure_no_significance`
    - `tests.test_figure_export.TestFigureExport::test_freq_response_figure`
    - `tests.test_figure_export.TestFigureExport::test_sweep_figure`
    - `tests.test_figure_export.TestFigureExport::test_sweep_figure_t_statistic_metric`
    - `tests.test_figure_export.TestFigureExport::test_sweep_figure_with_errors`

- **[7]** ModuleNotFoundError: No module named 'torch'
    - `::tests.test_tiny_transformer`
    - `::tests.test_tiny_transformer_train_smoke`
    - `tests.test_generator_f_noharm::test_generator_f_does_not_pull_song_g1_core`
    - `tests.test_linattn_readout_parity.TestExactParityAgainstTorchForward::test_state_dict_key_layout_matches_documented_contract`
    - `tests.test_wkv_readout_multilayer.TestMultiLayerNumericalCorrectness::test_state_dict_key_layout_matches_documented_contract`
    - `tests.test_wkv_spiking_forward::test_full_runner_reuses_rf_caches_across_all_seeds`
    - `tests.test_wkv_spiking_forward::test_rf_graded_read_keeps_float32_weight_casts_alive`

- **[4]** FileNotFoundError: [Errno 2] No such file or directory: 'rsync'
    - `tests.test_pool_sync_safety::test_chmod_f_minus_x_strips_the_executable_bit`
    - `tests.test_pool_sync_safety::test_extension_allowlist_blocks_py_files_but_allows_result_types`
    - `tests.test_pool_sync_safety::test_safe_links_rejects_a_symlink_escaping_the_source_tree`
    - `tests.test_pool_sync_safety::test_without_safe_links_the_escaping_symlink_would_have_been_pulled`

- **[3]** AssertionError: ⛔ no /home/dant123/Projects/sim/tools/pool_queue.sh
    - `tests.test_b2b_queue_next_wave_reason_check::test_a_redo_reason_for_a_different_amendment_2_cell_is_also_accepted`
    - `tests.test_b2b_queue_next_wave_reason_check::test_amendment1_and_redo_lines_together_are_both_accepted`
    - `tests.test_b2b_queue_next_wave_reason_check::test_amendment1_line_alone_is_well_formed`

- **[2]** FileNotFoundError: [Errno 2] No such file or directory: 'data/tinyshakespeare.txt'
    - `tests.test_distill_subword_lm_train_smoke::test_tiny_cpu_distill_reduces_loss_and_is_resumable`
    - `tests.test_scaled_subword_lm_train_smoke::test_tiny_cpu_train_reduces_loss_and_is_resumable`

- **[1]** AssertionError:   workstation: UNREACHABLE (skipped)
    - `tests.test_sim_runtime_trust_boundary::test_a_reachable_host_without_a_main_results_dir_still_has_its_revisions_pulled`

- **[1]** AssertionError: ('  pool41: UNREACHABLE (skipped)
    - `tests.test_sim_runtime_trust_boundary::test_runtime_pull_stops_and_fails_loudly_at_the_stage_cap`

- **[1]** AssertionError: assert ('would pull 1 file(s)' in '  pool41: UNREACHABLE (skipped)\npool_sync: (dry-run) 0 file(s) would be pulled from [pool41 ]\n')
    - `tests.test_sim_runtime_trust_boundary::test_runtime_dry_run_writes_nothing`

- **[1]** AssertionError: assert None is not None
    - `tests.test_generator_f_noharm::test_torch_is_preexisting_dependency`

- **[1]** AssertionError: redo line wrongly flagged malformed: ⛔ no /home/dant123/Projects/sim/tools/pool_queue.sh
    - `tests.test_b2b_queue_next_wave_reason_check::test_amendment2_redo_line_is_accepted_not_flagged_as_malformed`

- **[1]** FileNotFoundError: [Errno 2] No such file or directory: '/home/dant123/Projects/sim-worktrees/gate-b-v2-clean/research/specs/v13_backend_state_transplant.json'
    - `tests.test_v13_backend_state_transplant::test_loads_exact_authoritative_spec_path_and_rejects_stale_digest`

- **[1]** FileNotFoundError: [Errno 2] No such file or directory: '/sys/devices/system/cpu/cpu16/cpufreq/scaling_governor'
    - `tests.test_run_v14_stageA_performance_v3::test_gpu_clocks_and_profile_are_restored`

- **[1]** FileNotFoundError: [Errno 2] No such file or directory: 'systemctl'
    - `tests.test_llm_sh_never_touches_real_systemd::test_llm_on_with_no_local_stub_beyond_the_global_one_never_starts_a_real_unit`

- **[1]** FileNotFoundError: concept cache missing: research/findings/raw/activity_level_integration_cache/denoise64_seed42.npz
    - `tests.test_unified_brain_bridge::test_unified_skeleton_sizes_and_disjoint_slices`

## POLLUTION (33)

Single-process full-suite artifact: import-time mutation of `concept_pool_demo`'s module-level vocabs by other runners (v2..v5/set2..set5/multitag_eval_v17/compose_engram_demo_v2 all do `cpd_v1.NOUN_VOCAB = ...`). Verified: representative tests PASS in isolation (adjective_pool_COLD 1.75s, river 3.39s).

- **[12]** KeyError: 'adjective_pool_COLD'
    - `tests.test_per_regime_monitor_runner::test_calibration_does_not_mutate_compositional_gate_source`
    - `tests.test_per_regime_monitor_runner::test_tiny_synth_calibration_runs_and_returns_well_formed_payload`
    - `tests.test_pool_readout_8th_arc_runner::test_structural_effect_probes_validate_readout_substitution_and_replay`
    - `tests.test_targeted_cue_suppression_replay_runner::test_full_vs_uniform_arms_differ_at_least_on_some_query`
    - `tests.test_targeted_cue_suppression_replay_runner::test_structural_effect_probes_validate_all_three_mechanisms`
    - `tests.test_targeted_cue_suppression_replay_runner::test_tiny_synth_smoke_outputs_expected_json_shape`
    - `tests.test_unified_per_regime_monitor_runner::test_calibrate_direct_v2_returns_full_vocab_separator`
    - `tests.test_unified_per_regime_monitor_runner::test_compositional_encoding_produces_nonzero_engram_tag`
    - `tests.test_unified_per_regime_monitor_runner::test_phase1_caching_skips_retraining`
    - `tests.test_unified_per_regime_monitor_runner::test_substrate_has_hippocampal_regions`
    - `tests.test_unified_per_regime_monitor_runner::test_tiny_synth_end_to_end_well_formed`
    - `tests.test_unified_per_regime_monitor_runner::test_tiny_synth_writes_json_with_out_path`

- **[6]** KeyError: 'adjective_pool_BIG'
    - `tests.test_per_regime_monitor_runner::test_direct_retain_is_a_separate_measurement_from_full_acc`
    - `tests.test_per_regime_monitor_runner::test_tiny_synth_evaluation_runs_end_to_end_and_is_not_structurally_void`
    - `tests.test_per_regime_monitor_runner::test_tiny_synth_evaluation_writes_json_with_out_path`
    - `tests.test_spear_conversational_runner::test_full_vs_rhythm_removed_diverge_in_underlying_readout_smoke`
    - `tests.test_spear_conversational_runner::test_tiny_synth_does_not_raise_with_loads_and_out`
    - `tests.test_spear_conversational_runner::test_tiny_synth_runs_end_to_end_and_is_not_structurally_void`

- **[4]** KeyError: 'adjective_pool_HOT'
    - `tests.test_unified_per_regime_monitor_runner::test_calibrate_mode_writes_compositional_threshold_as_unified_value`
    - `tests.test_unified_per_regime_monitor_runner::test_calibrate_mode_writes_json_with_out_path`
    - `tests.test_unified_per_regime_monitor_runner::test_calibration_mode_runs_direct_gate_calibration`
    - `tests.test_unified_per_regime_monitor_runner::test_cli_direct_calibration_v2_flag_routes_v2`

- **[4]** KeyError: 'river'
    - `tests.test_generative_replay_pfc_frame_runner::test_full_vs_uniform_arms_differ_at_least_on_some_query`
    - `tests.test_generative_replay_pfc_frame_runner::test_tiny_synth_smoke_outputs_expected_json_shape`
    - `tests.test_theta_gamma_mode_unification_runner::test_structural_effect_probe_controls_pass_at_runner_level`
    - `tests.test_theta_gamma_mode_unification_runner::test_structural_effect_probe_runs_via_runner_actual_code_path`

- **[4]** KeyError: 'verb_pool_SPEAK'
    - `tests.test_pool_readout_8th_arc_runner::test_full_vs_uniform_arms_differ_at_least_on_some_query`
    - `tests.test_pool_readout_8th_arc_runner::test_tiny_synth_smoke_outputs_expected_json_shape`
    - `tests.test_theta_gamma_mode_unification_runner::test_full_vs_uniform_arms_differ_at_least_on_some_query`
    - `tests.test_theta_gamma_mode_unification_runner::test_tiny_synth_smoke_outputs_expected_json_shape`

- **[3]** KeyError: 'apple'
    - `tests.test_pirazzini_three_layer_runner::test_ach_only_mechanism_cannot_PASS`
    - `tests.test_pirazzini_three_layer_runner::test_tiny_synth_does_not_raise_with_loads_and_out`
    - `tests.test_pirazzini_three_layer_runner::test_tiny_synth_runs_end_to_end_and_is_not_structurally_void`

## REAL (28)

Verified FAIL in isolation (or fails on the main checkout too). Genuine repo state, no fixes made (section 5).

- **[7]** ValueError: pool #1 composer region D=128 != composer D=256
    - `tests.test_emerge58_unified_fluent_console::test_derisk_go_single_seed`
    - `tests.test_emerge58_unified_fluent_console::test_membership_aware_routing_no_false_denial`
    - `tests.test_emerge60_console_spiking_broca::test_derisk_go_seed42`
    - `tests.test_emerge60_console_spiking_broca::test_membership_routing_and_no_false_denial`
    - `tests.test_emerge66_console_self_organized::test_derisk_go_seed42`
    - `tests.test_emerge66_console_self_organized::test_membership_routing_and_no_false_denial`
    - `tests.test_grounded_wkv_renderer::test_fluidchat_wkv_grounded_and_gatefirst_moat`

- **[5]** ValueError: could not broadcast input array from shape (18,) into shape (13,)
    - `tests.test_emerge61_spiking_broca_order_robustness::test_position_independence_with_reset[100]`
    - `tests.test_emerge61_spiking_broca_order_robustness::test_position_independence_with_reset[101]`
    - `tests.test_emerge61_spiking_broca_order_robustness::test_position_independence_with_reset[42]`
    - `tests.test_emerge61_spiking_broca_order_robustness::test_reset_fixes_sequence_order_on_failing_seeds[100]`
    - `tests.test_emerge61_spiking_broca_order_robustness::test_reset_fixes_sequence_order_on_failing_seeds[101]`

- **[3]** AssertionError: PROTECTED MODIFIED:
    - `tests.test_compose_bind_no_harm::test_protected_byte_untouched_across_range`
    - `tests.test_compose_bridge_no_harm::test_protected_byte_untouched_across_range`
    - `tests.test_td_critic_no_harm::test_protected_byte_untouched_across_td_critic_range`

- **[2]** AssertionError: PROTECTED changed:
    - `tests.test_q2_no_harm::test_protected_byte_empty`
    - `tests.test_q2r_no_harm::test_protected_byte_empty`

- **[2]** TypeError: %o format: an integer is required, not dict
    - `tests.test_bio_three_factor_cli::test_fp16_synapse_state_capsys`
    - `tests.test_bio_three_factor_cli::test_fp16_synapse_state_flag_appears_in_help`

- **[2]** assert 400000 == 101
    - `tests.test_v14_stageB_intrinsic_readiness::test_production_protocol_is_forwarded_to_all_arms_and_bound_in_receipt`
    - `tests.test_v14_stageB_runner::test_production_runner_stops_at_source_bound_101_spikes_and_binds_protocol`

- **[2]** tools.v14_stageB_fast_channel_clamp_analysis.FastChannelClampAnalysisError: numpy provenance sidecar is invalid
    - `tests.test_v14_stageB_fast_channel_clamp_analysis::test_exact_sealed_evidence_produces_structural_no_go`
    - `tests.test_v14_stageB_fast_channel_clamp_analysis::test_source_gate_does_not_allow_aggregate_compensation`

- **[1]** AssertionError:
    - `tests.test_graded_inhibition_pathway::test_byte_identical_when_no_graded_pathway`

- **[1]** AssertionError: PROTECTED set changed:
    - `tests.test_engram_bootstrap_no_harm::test_protected_set_byte_empty_diff`

- **[1]** AssertionError: Protected files diverged from Task 2 commit:
    - `tests.test_generative_replay_soundness::test_reuse_by_import_only`

- **[1]** AssertionError: protected modules were modified during the phase-factored arc: ['research/runners/concept_pool_demo.py', 'research/runners/text_minimal_isolation.py', 'sim/bridge.py', 'sim/kernels.py' ...
    - `tests.test_phase_factored_no_harm::test_protected_set_byte_unchanged_across_the_arc`

- **[1]** ValueError: could not broadcast input array from shape (15780,) into shape (10521,)
    - `tests.test_emerge87_reservoir_coresident::test_seed42_reservoir_goes_coresident_and_is_functionally_isolated`

## UNVERIFIED (52)

Grouped by first error line only; no isolation run. No fixes (section 5).

- **[3]** AssertionError:
    - `tests.test_input_mean_adapt::test_byte_identical_when_off`
    - `tests.test_q2_smoke::test_tiny_smoke_runs`
    - `tests.test_q2r_smoke::test_tiny_smoke_runs`

- **[3]** AssertionError: assert (False)
    - `tests.test_q2_grounding::test_generator_f_artifact_present`
    - `tests.test_q2r_grounding::test_generator_f_artifact_present`
    - `tests.test_sim_runtime_trust_boundary::test_runtime_pull_skips_a_file_over_the_size_cap`

- **[3]** assert (None is not None)
    - `tests.test_webapp_server::test_brain_chat_xedge_selfschema_declarative_reproduces_bespoke_through_real_handler`
    - `tests.test_webapp_server::test_brain_chat_xedge_selfschema_on_reads_live_crossedge_and_lesion_collapses`
    - `tests.test_webapp_server::test_brain_chat_xedge_surprise_episodic_on_reads_live_crossedge_and_lesion_collapses`

- **[2]** AssertionError: assert 2 == 1
    - `tests.test_b2b_queue_next_wave_reason_check::test_a_genuinely_wrong_reason_is_still_flagged`
    - `tests.test_b2b_queue_next_wave_reason_check::test_a_torn_line_missing_the_pinned_cd_prefix_is_still_flagged`

- **[2]** assert False
    - `tests.test_open_ended_generation_fluent::test_render_hypothesis_fluent_flagged_guess_stub`
    - `tests.test_webapp_server::test_brain_chat_xedge_curiosity_d6_ambient_default_is_on`

- **[2]** assert True is False
    - `tests.test_open_ended_generation_fluent::test_render_hypothesis_template_fallback_without_mouth`
    - `tests.test_runner_provenance_snapshot::test_archive_manifest_verification_rejects_tampering_and_extra_source`

- **[1]** AssertionError: /home/dant123/Projects/neural-simulator/.claude/worktrees/draft-test-triage/.venv/bin/python: No module named research.runners.dendritic_wa_gate
    - `tests.test_dendritic_grounding::test_dendritic_wa_gate_pipeline_turns`

- **[1]** AssertionError: I'm holding 2 referents in working memory at once: dog and cat.
    - `tests.test_webapp_server::test_brain_chat_xedge_curiosity_d6_semantic_drop_genuinely_drops_referent`

- **[1]** AssertionError: an artifact with NO run-start source anywhere must fail closed, not pass via mtime
    - `tests.test_gate_corpus_check_required::test_incident_gap4_with_no_run_start_source_at_all_fails_closed`

- **[1]** AssertionError: assert 'hello world there!' == 'Hello world there!'
    - `tests.test_corpus_fetch_smoke::test_clean_text_collapses_whitespace_ascii_only`

- **[1]** AssertionError: assert (<tests.test_pair_production_path._FakeRS object at 0x7f989a9fd310> is <tests.test_pair_production_path._FakeRS object at 0x7f989a9fd310> and 'global' == 42023900)
    - `tests.test_pair_production_path::test_private_rng_legacy_knob_reproduces_the_reseed`

- **[1]** AssertionError: assert (<tests.test_pair_production_path._FakeRS object at 0x7f989af27b10> is not <tests.test_pair_production_path._FakeRS object at 0x7f989af27b10>)
    - `tests.test_pair_production_path::test_private_rng_restores_the_cupy_stream`

- **[1]** AssertionError: assert not True
    - `tests.test_lanea_graded_affect_quench_v1::test_spec_output_paths_are_new_and_phase_separated`

- **[1]** AssertionError: assert {'agrees_with_encoded': True, 'd': 0.9999999902140674, 'encoded_as': 'perceived', 'known': True, ...} is None
    - `tests.test_webapp_server::test_brain_chat_source_provenance_default_off_is_byte_identical`

- **[1]** AssertionError: assert {'candidates'...', ...}], ...} == {'candidates'...', ...}], ...}
    - `tests.test_parameter_research::test_discovery_is_resumable_deduplicated_by_doi_and_preserves_exact_metadata`

- **[1]** AssertionError: backlog.py selftest FAILED:
    - `tests.test_backlog::test_backlog_selftest_passes`

- **[1]** AssertionError: bash: -c: line 6: unexpected EOF while looking for matching `)'
    - `tests.test_seam_contracts.TestSeam4CheckedTokenStripping::test_the_consumers_own_extraction_recovers_the_command_without_the_token`

- **[1]** AssertionError: fused north != host None for cue ('dog', 'bark')
    - `tests.test_seq_fused_fabric::test_fused_equals_host_and_separate_bridge`

- **[1]** AssertionError: legacy separate-bridge != host for ('dog', 'run') -- the fused change perturbed a legacy path
    - `tests.test_seq_fused_fabric::test_off_byte_identical`

- **[1]** AssertionError: moat breach: 1 false-accept(s) on the fused path
    - `tests.test_seq_fused_fabric::test_fused_moat_zero_false_accepts`

- **[1]** AssertionError: shrink != full-V block selection on cue ('cat', 'run')
    - `tests.test_onebrain_integrated_loop_fold::test_seq_vocab_shrink_answer_identical_and_smaller[42]`

- **[1]** AssertionError: the hook must be executable (Claude Code invokes it directly)
    - `tests.test_deny_protected_edits_hook::test_hook_script_exists_and_is_executable`

- **[1]** AssertionError: the real incident artifact is expected to carry no stamp of its own -- did it change?
    - `tests.test_gate_corpus_check_required::test_incident_gap4_c25_ckpt_artifact_passes_via_the_real_0138_entry`

- **[1]** AssertionError: tools/local_llm/llama-swap.yaml's cmd for 'qwen38-27b-iq4nl-mtp' has drifted from llm.sh's profile_cmd():
    - `tests.test_llama_swap_config::test_each_profile_cmd_is_byte_identical_to_llm_sh_profile_cmd`

- **[1]** AssertionError: topographic-bias-factor help should cite the biology source so future users see why 1.5/0.7 not arbitrary.
    - `tests.test_text_embeddings::test_topographic_bias_anti_cheat_ratio_in_biology_range`

- **[1]** AssertionError: vectorized EMAs differ from scalar: max|diff|=6.810e-01
    - `tests.test_gate_coupling_vectorized::test_vectorized_gate_couplings_bit_identical_emas`

- **[1]** AssertionError: {'arm': 'main', 'heldagent_cat_acc': 0.6111111111111112}
    - `tests.test_reservoir_lm_emergence::test_perception_grounded_category_beats_pixel_scramble`

- **[1]** AssertionError: {'arm': 'nolearn', 'heldagent_cat_acc': 0.6111111111111112, 'reversal_acc': 1.0}
    - `tests.test_reservoir_lm_emergence::test_step1_fixed_codon_of_feature_statistics_beats_discovery_scramble`

- **[1]** AssertionError: ⛔ pre-merge-commit: git config core.hooksPath is '/home/dant123/Projects/neural-simulator/.claude/worktrees/draft-test-triage/tools/githooks'.
    - `tests.test_pre_merge_commit_worktree_hookspath::test_current_hook_accepts_the_shared_absolute_hookspath_from_a_worktree`

- **[1]** FileNotFoundError: [Errno 2] No such file or directory: '/opt/data/cache/scratch/pytest-of-hermes/pytest-1/test_revisions_are_listed_even0/projects/neural-simulator/research/findings/raw/gpu.json'
    - `tests.test_sim_runtime_trust_boundary::test_revisions_are_listed_even_when_the_login_shell_wraps_ls`

- **[1]** FileNotFoundError: [Errno 2] No such file or directory: '/opt/data/cache/scratch/pytest-of-hermes/pytest-1/test_runtime_pull_lands_throug0/projects/neural-simulator/research/findings/raw/main.json'
    - `tests.test_sim_runtime_trust_boundary::test_runtime_pull_lands_through_the_stage_and_ignores_a_planted_symlink`

- **[1]** assert "I'm holding ... dog and cat." == "I'm holding ...on right now."
    - `tests.test_webapp_server::test_brain_chat_xedge_curiosity_d6_on_qualifies_reply_and_lesion_collapses_it`

- **[1]** assert (0 > 0)
    - `tests.test_corpus_fetch_smoke::test_fetch_signature_and_offline_degrades_to_shakespeare`

- **[1]** assert (1 == 0)
    - `tests.test_generator_f_gate_smoke::test_help`

- **[1]** assert (1 == 2)
    - `tests.test_generator_f_gate_smoke::test_fewer_than_3_seeds_exit_2`

- **[1]** assert 0.875 == 1.0
    - `tests.test_consolidated_320_conversation::test_production_agent_on_stream_learned_codes_seed42`

- **[1]** assert None is False
    - `tests.test_webapp_server::test_brain_chat_xedge_curiosity_d6_session_isolated`

- **[1]** assert [[0.0, 0.0, 0....0, ...], ...] == [[None, None,...ne, ...], ...]
    - `tests.test_gate_coupling_vectorized::test_vectorized_gate_couplings_identical_gate_writes`

- **[1]** assert not np.True_
    - `tests.test_developmental_vocal_convention::test_coactivity_eligibility_requires_reward_for_weight_change`

- **[1]** assert np.False_
    - `tests.test_v14_stageB_runner::test_compact_trace_mode_writes_lossless_archive_without_inline_vectors`

- **[1]** research.runners.v14_stageB_batched_physiology.StageBBatchedPhysiologyError: batch declaration path must be a regular file
    - `tests.test_v14_stageB_failure_diagnostic::test_real_protocol_authenticates_and_rederives_exact_selection`

- **[1]** tools.v13_legacy_source_package.PackageError: frozen runner cannot start under the execution-package controls: Traceback (most recent call last):
    - `tests.test_v13_legacy_source_package::test_seed_free_execution_probe_uses_package_identity_without_bytecode`

- **[1]** tools.v14_stageB_batch_width_benchmark.StageBBatchWidthBenchmarkError: consumed campaign must be a regular file
    - `tests.test_v14_stageB_batch_width_benchmark::test_filed_benchmark_uses_consumed_candidates_and_fixed_paired_order`

## Notes and deviations

- **No fixes** (section 5). This is grouping only.
- `tools/memcap.sh 12` refuses in this container (no systemd --user, cgroup v2 read-only) — ran uncapped per the script's own documented clause (23-27G free, no live training on S). The later `server-load` ruling (nice -n 10) arrived after this run started; all isolation re-runs used `nice -n 10`.
- Isolation re-runs used the same env (`SIM_NO_PROVENANCE=1 CUDA_VISIBLE_DEVICES=`).
- The 4 errors are collection errors for `test_state_manager` (dearpygui) and `test_tiny_transformer*` (torch) — counted under ENVIRONMENT above (test-id level).
- Root cause of POLLUTION is import-order dependent, so individual tests pass when run file-by-file; only the full suite in one process exposes it.

