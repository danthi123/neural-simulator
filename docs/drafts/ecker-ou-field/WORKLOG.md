# ecker-ou-field WORKLOG (T15)

- 2026-09-30 23:28 (local model, Hermes): T15 Ecker CA3 OU-noise field. Confirmed the engine reads
  `cfg.ou_std_current_pA` (default 100 pA, `sim/config.py:370`) and never reads `cfg.ou_noise_sigma_pa`
  (phantom); `enable_ou_process` IS read (`sim/bridge.py`) and defaults True, so the 40 pA runs ran at 100 pA.
- Renamed `ou_noise_sigma_pa` -> `ou_std_current_pA` (same value, T15 comment) in the 8 runners that set only the
  phantom field: `_gap5_ecker_adex_ca3_replay_derisk`, `_gap5_ecker_adex_ca3_stdp_band_derisk`,
  `_gap5_ecker_reactivates_REAL_d5_derisk`, `_gap5_ecker_recurrent_replay`, `_gap5_izh_replay_merge_derisk`,
  `_gap5_learned_band_emergence`, `_gap5_wake_sleep_phase_switch`, `_gap5_wake_sleep_roundtrip`. All 8 `py_compile`d OK.
- `_gap5_onebrain_capstone.py` sets BOTH fields (real `ou_std_current_pA`=20.0 wake / 0.0 sleep + phantom
  `ou_noise_sigma_pa`=40.0 sleep) -> NOT affected (sets the real field); the 40.0 line is dead code. No edit (T15
  rule); flagged to Claude whether it was meant to be 40 pA.
- Affected findings (their own runner is one of the 8; none of the 8 is imported by another runner): 6, all stated
  40 pA / ran 100 pA. Wrote `docs/drafts/ecker-ou-field/AFFECTED.md` (table) + appended a dated
  `## Correction (PROVISIONAL, local model)` paragraph to each (both values, nothing re-interpreted). No runs.
- No re-runs (T15: no runs). Commit on `research/draft-ecker-ou-field` through the gates; no push (RULING
  push-credential). NEEDS-CLAUDE: FAILURE_LOG row + gate (task_8f9171ae); onebrain_capstone dead-line judgment.
- 2026-10-01 04:44 (local model, Hermes): T15 follow-up per the `ecker-ou` ruling (2026-10-01): AFFECTED.md was
  incomplete — 6 runners import `_gap5_ecker_adex_ca3_stdp_band_derisk`'s OU-setting `build_store` and ran under
  the same phantom-field bug (stated 40 pA / ran 100 pA). Listed all 6 importers + the 8 findings that ran them
  in AFFECTED.md (new importer table), added the PROVISIONAL note to each of the 8 findings, and appended the
  onebrain_capstone dead-code sentence. `_gap5_decorrelation_read_ltu_derisk` imports the builder but no finding
  ran it (listed, no note). No re-runs; FAILURE_LOG row + gate PF already on main.
