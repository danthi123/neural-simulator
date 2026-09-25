# Memory condense: proposed archive list and conflicts

Proposal only. Nothing in `~/.claude/projects/-home-dant123-Projects-sim/memory/` was changed. Every one of the 100
memory files there was read in full (99 indexed in `MEMORY.md` plus one unindexed file,
`project_foundational_curriculum_base_knowledge.md`).

**Result:** 65 memories kept (one line each in `PROPOSED_MEMORY.md`, 5,932 characters versus 19,425 today), 35 proposed
for archiving. "Archive" means: take the file out of the index and move it into a subfolder so it is no longer offered
at session start but can still be read. Suggested commands, for the owner to run after approving:

```bash
cd ~/.claude/projects/-home-dant123-Projects-sim/memory
mkdir -p _archive_2026-09-25
cp MEMORY.md _archive_2026-09-25/MEMORY.md.before-condense
mv -n \
  feedback_close_all_shortcuts_before_capability.md \
  feedback_prioritize_orchestration_overhead.md \
  feedback_probes_match_deployed.md \
  project_actual_goal_artificial_life_brain_analogue.md \
  project_baloo_indexer_memory_hog.md \
  project_cheat5_v3_results.md \
  project_cluster_stacking_falsified.md \
  project_conversational_primary_robust_multicue_parser.md \
  project_deep_knowledge_brain_fluency_build.md \
  project_downtime_compute_2026-07-25.md \
  project_fluid_llm_like_conversation_priority.md \
  project_foundational_curriculum_base_knowledge.md \
  project_gap_closure_mission_active.md \
  project_generative_sequence_frontier.md \
  project_grounded_language_faculty.md \
  project_hermes_local_qwen_fallback_agent.md \
  project_hermes_visible_research_loop.md \
  project_next_priorities.md \
  project_one_brain_integrated_pipeline_and_cleanup.md \
  project_one_brain_substrate_vs_functional.md \
  project_overnight_autonomous_2026_08_21.md \
  project_own_voice_fluency_pursue_fully_2026_09_03.md \
  project_perception_arc_COMPLETE.md \
  project_perception_arc_stage3_BREAKTHROUGH.md \
  project_pfc_working_memory.md \
  project_phase_c_resolved.md \
  project_plasticity_gating_infra.md \
  project_post_conversational_roadmap_tiers.md \
  project_remaining_cheats.md \
  project_replicated_runner_bug.md \
  project_silent_motor_trap.md \
  project_stdp_w_bounds.md \
  project_vllm_pilot_outcome.md \
  project_vocab_target_breadth_vs_depth.md \
  reference_findings_dir.md \
  _archive_2026-09-25/
cp /home/dant123/Projects/sim/.claude/memory_condense/PROPOSED_MEMORY.md MEMORY.md
```

No in-force owner directive is on this list. Four entries are superseded owner directives whose content is carried by
newer kept memories; they are marked **owner: confirm** so you can keep any of them.

## A. Superseded by a newer owner ruling

| Memory | Superseded by | Why |
|---|---|---|
| `project_own_voice_fluency_pursue_fully_2026_09_03.md` | `project_2026_09_19_permanent_llm_mouth_loadbearing_metric` (09-19) | The spiking-mouth fluency arc was closed as falsified; the file itself says it is superseded for the mouth. |
| `project_fluid_llm_like_conversation_priority.md` | `project_dendritic_cortex_for_emergence` (07-01/07-10), `project_genuine_cognition_northstar_pivot` (07-23), 09-19 mouth ruling | The emergence steer the same day says it SUPERSEDES this feature-by-feature mode; "minimize the transformer" is now bounded by the permanent Qwen mouth. |
| `project_deep_knowledge_brain_fluency_build.md` | 09-19 mouth ruling (fluency); `project_knowledge_must_be_learned_not_rag` (09-24, knowledge) | "Brain-native fluency, avoid the LLM" is replaced by the permanent mouth; knowledge must now be written by plasticity, not bulk-extracted. Resumable training lives on in `feedback_pause_on_demand_not_vram_cap`. **owner: confirm** |
| `project_gap_closure_mission_active.md` | `project_genuine_cognition_northstar_pivot` (07-23) | The 5-gap mission became a sub-view of the master roadmap; THE LAW is CLAUDE.md non-negotiable 3. |
| `project_actual_goal_artificial_life_brain_analogue.md` | `project_master_directive_relentless_biological_emergence` (07-01, which says it refines this), the 07-23 north-star, 09-18 fork decision | Its "honest negatives ARE the deliverable" was replaced by "walls are undiscovered mechanisms"; the artificial-life goal is carried by the MASTER directive. **owner: confirm** |
| `feedback_close_all_shortcuts_before_capability.md` | `feedback_end_state_fully_spiking_one_brain_path_by_efficiency` (06-27), `project_2026_09_04_scaffold_retirement_before_learning`, `project_goal_is_integrated_production_default_brain`, `docs/TERMS.md` ("closed") | The "shortcuts before capability" gate was replaced by "end state fixed, path by efficiency" and the 09-04 order; its index line also contradicts its own body (see conflict 15). **owner: confirm** |
| `project_one_brain_integrated_pipeline_and_cleanup.md` | `project_goal_is_integrated_production_default_brain` (08-11), `feedback_close_arcs_to_full_capacity` (keeps numpy as the test oracle) | The 06-18 one-brain arc plan was folded into the 08-11 integration goal. |
| `project_post_conversational_roadmap_tiers.md` | the MASTER ROADMAP (07-23) | The 06-19 tier plan is replaced by the staged faculty roadmap. |
| `project_vocab_target_breadth_vs_depth.md` | 09-19 mouth ruling + 09-24 learned-knowledge ruling | Wording now comes from the mouth; the knowledge axis is governed by 09-24. |
| `project_overnight_autonomous_2026_08_21.md` | CLAUDE.md non-negotiables 9 and 11 + ACTIVE MISSION | A one-night directive; its standing constraints (one GPU brain at a time, cost-routing, resume from the board) are in CLAUDE.md. |
| `project_foundational_curriculum_base_knowledge.md` (unindexed today) | `project_knowledge_must_be_learned_not_rag` (09-24); `feedback_pause_on_demand_not_vram_cap` | The curriculum/first-chat plan is superseded; the 24/7, pausable, crash-resumable training rule is in the pause memory. **owner: confirm** |

## B. Superseded tooling (the local model is now Claude Code via `tools/local_llm/llm.sh claude`, 09-25)

| Memory | Why |
|---|---|
| `project_hermes_local_qwen_fallback_agent.md` | Hermes → OpenHands (09-08) is replaced by `llm claude` (bake-off 09-25). The one-driver-at-a-time rule is now CLAUDE.md non-negotiable 12. |
| `project_hermes_visible_research_loop.md` | Hermes-specific loop. Its general tip (check system AND user systemd units at a wall) is worth one line in a skill if wanted. |
| `project_vllm_pilot_outcome.md` | Hermes-era vLLM pilot; the local model now runs on llama.cpp `llama-server` via `tools/local_llm/`. |

## C. Finished

| Memory | Why |
|---|---|
| `project_baloo_indexer_memory_hog.md` | Owner disabled baloo on 09-17 (index line); isolation of heavy jobs is covered by the memcap memory. |
| `project_downtime_compute_2026-07-25.md` | One weekend's lanes; `tools/aws_gpu.sh` now keeps the key and instance id durably, and the scratch-on-disk memory covers /tmp. |
| `project_generative_sequence_frontier.md` | The loop was demonstrated 06-23; the mouth question was closed 09-19. |
| `project_grounded_language_faculty.md` | 06-23 status of the Qwen2.5-0.5B faculty; superseded by the 09-19 mouth ruling. |
| `project_conversational_primary_robust_multicue_parser.md` | Phases 1-2 completed (recorded in the 06-19 tiers memory). |
| `feedback_prioritize_orchestration_overhead.md` | The CUDA-graph resonate loop landed (`enable_rf_cudagraph` in `sim/config.py`, `sim/bridge.py`); local-first is now the consumer-hardware principle. Verify it is the default before archiving. |
| `project_replicated_runner_bug.md` | Fixed in `research/runners/g11_bg_replicated_runner.py` ("Bug-fix 2026-04-30" comments). |
| `project_phase_c_resolved.md`, `project_pfc_working_memory.md`, `project_perception_arc_stage3_BREAKTHROUGH.md`, `project_perception_arc_COMPLETE.md`, `project_cheat5_v3_results.md`, `project_cluster_stacking_falsified.md`, `project_remaining_cheats.md`, `project_next_priorities.md`, `project_silent_motor_trap.md` | April 2026 navigation arcs, finished; the findings docs they cite hold the results. |

## D. Now carried by the repo (a gate, a tool or a reference doc)

| Memory | Where it lives now |
|---|---|
| `project_stdp_w_bounds.md` | Enforced by `tools.lab.bound_check`; documented in `docs/ENGINE_REFERENCE.md`. |
| `project_plasticity_gating_infra.md` | `docs/ENGINE_REFERENCE.md` (`set_plasticity_gate`). |
| `project_one_brain_substrate_vs_functional.md` | CLAUDE.md non-negotiable 2 + `feedback_move_everything_to_shared_spiking_substrate`. |
| `reference_findings_dir.md` | `gates/doc_type` enforces finding/plan placement. |
| `feedback_probes_match_deployed.md` | The `verify-go` and `neural-simulator` skills' like-for-like and faithful-scale checks. |

## Conflicts found (newer one named; the proposed index follows the newer one)

1. `project_agi_first_fork_openness_2026_09_06`: the index says "NOT yet decided", the file says **DECIDED 2026-09-18**. Newer: the file body.
2. `project_baloo_indexer_memory_hog`: the body says suspend baloo each session; the index says RESOLVED 09-17 (owner disabled it). Newer: the index line.
3. `feedback_no_claude_routines_for_continuity`: the body names a Windows Scheduled Task watchdog as the continuation mechanism; the index line and CLAUDE.md say continuation is MANUAL with an in-session heartbeat, no daemon. Newer: the index line + CLAUDE.md.
4. `feedback_never_stall_autonomous` rule 5 (keep an external re-trigger / watchdog alive) vs CLAUDE.md "no watchdog/daemon" by owner choice. Newer: CLAUDE.md.
5. `feedback_affect_shapes_speech_not_marker_words` ("never flip affect-marker features without owner sign-off") vs the owner ruling on the board, 2026-09-25 ~14:10: RETIRE the prepended marker word (option A). Newer: the board ruling; the memory body should be updated.
6. `project_2026_09_04_scaffold_retirement_before_learning` (continuous learning deferred) vs `project_agi_first_fork_openness_2026_09_06` (09-18: continuous learning as default AGREED), `project_knowledge_must_be_learned_not_rag` (09-24: "I don't know" triggers learning) and `project_remember_what_matters_prioritized_memory` (09-25). Newer: 09-18/09-24/09-25. **Owner: is learning now in scope before scaffold retirement is complete?**
7. `project_2026_08_19_strategic_reframe_continuous_substrate` ("make the brain continuous" first) vs 09-04 (scaffold retirement first). Newer: 09-04 (the file already says so).
8. `project_own_voice_fluency_pursue_fully_2026_09_03` vs 09-19 permanent mouth. Newer: 09-19 (the file already says so).
9. `project_deep_knowledge_brain_fluency_build` and `project_fluid_llm_like_conversation_priority` ("minimize/avoid the LLM") vs 09-19 permanent Qwen mouth. Newer: 09-19.
10. `feedback_end_state_fully_spiking_one_brain_path_by_efficiency` ("END = fully spiking", non-negotiable) vs 09-19 (the mouth is a disclosed permanent exception). Newer: 09-19.
11. `project_goal_is_integrated_production_default_brain` ("done = scaffold-retired") vs 09-19 (load-bearing fraction replaces % scaffold-retired). Newer: 09-19 (the file already says so).
12. `feedback_long_local_runs_ok_confirm_cloud_cause` (07-15: cloud spend is Claude's judgment) vs `project_aws_spend_cap_2026_09_23` (~$50/day cap; above it is an owner ask) and 09-18 (scaling investment gated on research). Newer: 09-23 / 09-18.
13. `project_hardware_upgrade_plan_2026_09` (09-03: add 3090s now) vs the 09-18 finding "do not buy hardware yet" and 09-19 "STOP buying hardware for fluency". Newer: 09-18/09-19. **Owner: is the +3090 plan on hold?**
14. `feedback_parallelize_aggressively_via_subagents` (fan out concurrent subagents by default) vs `feedback_minimize_plan_usage_via_nonclaude_machinery` (08-06) and CLAUDE.md cost-routing (08-19): compute goes to non-Claude machinery, agents only for genuine builds, tiered. Newer: 08-06/08-19.
15. `feedback_close_all_shortcuts_before_capability`: its index line says "closed = converted-to-spiking OR honest-negative", its own body (owner 06-20) says a characterized honest-negative is NOT closed, and `docs/TERMS.md` (newest) says closed = integrated, OR an honest-negative recorded AND the capability explicitly parked. Newest: `docs/TERMS.md`.
16. `feedback_brain_based_only_standard` (owner wants the byte-level diff before a protected `sim/` edit) vs `feedback_dont_gate_on_approval` (06-16/06-20: guarded, reversible `sim/` edits pre-approved, review after). Newer: the latter (it says so).
17. `feedback_autonomous_overnight` ("don't merge to main without approval") vs `project_codex_to_claude_transition` (08-19: main trunk, topic branches merged), `feedback_dont_gate_on_approval` (06-18 standing approval) and the 09-25 ruling that the local model may commit results to main through the gates. Newer: the later practice. **Owner: confirm; this task itself left CLAUDE.md on a branch, unmerged.**
18. `project_generative_sequence_frontier` (06-22: "never weaken the no-confab moat") vs `feedback_moat_not_hard_lossy_memory_ok` (06-17) and the 08-19 reframe (moat softened to a signal). Newest: 08-19.
19. `feedback_pause_on_demand_not_vram_cap` ("cloud only for a genuine >24 GB VRAM wall") vs 07-15 and 09-23. Newer: 09-23.
20. Stale Windows details: `user_role` (E:\ path), `feedback_no_stale_pollers` (Get-CimInstance sweep), `feedback_deep_research_at_roadblocks` (E:\ catalog path) vs `project_linux_migration_state_2026-07-16`. Newer: the Linux migration.
