# T15 — Ecker CA3 OU-noise field: affected findings

**Bug.** Nine `research/runners/_gap5_*` set `cfg.ou_noise_sigma_pa`, a field the engine never reads. The engine
reads `cfg.ou_std_current_pA` (default **100 pA**, `sim/config.py:370`); `ou_noise_sigma_pa` is a phantom. Runs the
runners configured for **40 pA** of OU noise therefore ran at **100 pA**.

**Fix (this branch, `research/draft-ecker-ou-field`).** The 8 runners that set *only* the phantom field had
`ou_noise_sigma_pa` renamed to `ou_std_current_pA` (same value, a comment naming T15). Each was `py_compile`d.
`_gap5_onebrain_capstone.py` sets *both* fields: it sets the real field directly (`ou_std_current_pA=20.0` in the
wake build, `ou_std_current_pA=0.0` in the sleep replay) and also `ou_noise_sigma_pa=40.0` (sleep). Because it sets
the real field, its runs were **not** affected by the phantom bug (wake ran at 20 pA, sleep at 0 pA); the
`ou_noise_sigma_pa=40.0` line is dead code (the engine never reads it). **No edit made** (T15 rule); whether that
dead line was *intended* to be 40 pA is a judgment call — flagged to Claude.

No runs were re-run (T15: no runs). Nothing below re-interprets a finding's conclusion; the table records the
stated-vs-run OU values only.

| finding | runner | stated value | value run |
|---|---|---|---|
| 2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO-closes-the-swr-wall | `_gap5_ecker_adex_ca3_replay_derisk` | 40 pA | 100 pA |
| 2026-08-20-ecker-ca3-forward-band-LEARNED-by-STDP-emergent-swr-replay-6seed-GO | `_gap5_ecker_adex_ca3_stdp_band_derisk` | 40 pA | 100 pA |
| 2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read | `_gap5_ecker_reactivates_REAL_d5_derisk` | 40 pA | 100 pA |
| 2026-07-25-gap5-learned-band-emergence-STDP-directed-traversal-6seed-GO | `_gap5_learned_band_emergence` | 40 pA | 100 pA |
| 2026-07-25-gap5-wake-sleep-phase-switch-6seed-GO | `_gap5_wake_sleep_phase_switch` | 40 pA | 100 pA |
| 2026-07-25-gap5-wake-sleep-roundtrip-coresident-merge-6seed-GO | `_gap5_wake_sleep_roundtrip` | 40 pA | 100 pA |

Not affected (mention the runner in prose, or import a helper, but did not run it): the 2026-08-27 graded-recall and
reverse-edge learn-through-use findings (run `_gap5_graded_recall_*` / `_gap5_reverse_edge_*`), the 2026-07-25
CRITICAL apical-R finding (runs the consolidation runner), the 2026-08-26 cls-interleaved finding (runs
`_cls_interleaved_consolidation_derisk`), the 2026-07-25 consolidation-frontier scoping finding (runs
`nmda_compositional_consolidation`), and the 2026-07-25 onebrain-sleepcycle-merge + 2026-08-26 sleep-replay-wirein
findings (run `_gap5_onebrain_sleepcycle_merge` / `_sleep_replay_flip_soak`, which import the phase-switch/roundtrip
helpers but do not run their OU-setting `build`/`_build` functions). None of the 8 affected runners is imported by
any other runner.
