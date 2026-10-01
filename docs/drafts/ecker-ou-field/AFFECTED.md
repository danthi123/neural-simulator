# T15 — Ecker CA3 OU-noise field: affected findings

**Bug.** Nine `research/runners/_gap5_*` set `cfg.ou_noise_sigma_pa`, a field the engine never reads. The engine
reads `cfg.ou_std_current_pA` (default **100 pA**, `sim/config.py:370`); `ou_noise_sigma_pa` is a phantom. Runs the
runners configured for **40 pA** of OU noise therefore ran at **100 pA**. The bug also reached runners that *import*
an affected builder's OU-setting function (they call `build_store`, which set the phantom field) — see the importer
table below (T15 follow-up, `ecker-ou` ruling 2026-10-01).

**Fix (this branch, `research/draft-ecker-ou-field`).** The 8 runners that set *only* the phantom field had
`ou_noise_sigma_pa` renamed to `ou_std_current_pA` (same value, a comment naming T15). Each was `py_compile`d.
`_gap5_onebrain_capstone.py` sets *both* fields: it sets the real field directly (`ou_std_current_pA=20.0` in the
wake build, `ou_std_current_pA=0.0` in the sleep replay) and also `ou_noise_sigma_pa=40.0` (sleep). Because it sets
the real field, its runs were **not** affected by the phantom bug (wake ran at 20 pA, sleep at 0 pA); the
`ou_noise_sigma_pa=40.0` line is dead code (the engine never reads it). **No edit made** (T15 rule); whether that
dead line was *intended* to be 40 pA is a judgment call — flagged to Claude.

No runs were re-run (T15: no runs). Nothing below re-interprets a finding's conclusion; the tables record the
stated-vs-run OU values only.

## Directly affected (the runner itself set the phantom field)

| finding | runner | stated value | value run |
|---|---|---|---|
| 2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO-closes-the-swr-wall | `_gap5_ecker_adex_ca3_replay_derisk` | 40 pA | 100 pA |
| 2026-08-20-ecker-ca3-forward-band-LEARNED-by-STDP-emergent-swr-replay-6seed-GO | `_gap5_ecker_adex_ca3_stdp_band_derisk` | 40 pA | 100 pA |
| 2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read | `_gap5_ecker_reactivates_REAL_d5_derisk` | 40 pA | 100 pA |
| 2026-07-25-gap5-learned-band-emergence-STDP-directed-traversal-6seed-GO | `_gap5_learned_band_emergence` | 40 pA | 100 pA |
| 2026-07-25-gap5-wake-sleep-phase-switch-6seed-GO | `_gap5_wake_sleep_phase_switch` | 40 pA | 100 pA |
| 2026-07-25-gap5-wake-sleep-roundtrip-coresident-merge-6seed-GO | `_gap5_wake_sleep_roundtrip` | 40 pA | 100 pA |

## Importers of the affected builders (T15 follow-up, `ecker-ou` ruling 2026-10-01)

The same bug reached runners that *import* an affected builder's OU-setting function: each calls `build_store`
(imported from `_gap5_ecker_adex_ca3_stdp_band_derisk`), which — before the T15 rename — set the phantom
`cfg.ou_noise_sigma_pa`. Each such run therefore also used **100 pA** (the engine default) instead of the intended
**40 pA** (the runner's `--ou-sigma` default). `_gap5_ecker_adex_ca3_replay_derisk` and
`_gap5_ecker_reactivates_REAL_d5_derisk` are imported by no other runner (the stdp_band→replay_derisk mention is a
comment, not an import).

| importing runner | builder imported (OU-setting func) | findings that ran it | stated | ran |
|---|---|---|---|---|
| `_gap5_ecker_replay_learn_through_use_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | 2026-08-27-braun-gapcoding-learn-through-use-NOGO, 2026-08-27-btsp-directional-write-learn-through-use-PARTIAL, 2026-08-27-conduction-delay-directional-replay-learn-through-use-PARTIAL, 2026-08-27-ecker-adex-store-learn-through-use-NOGO | 40 pA | 100 pA |
| `_gap5_graded_recall_learn_through_use_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | 2026-08-27-graded-recall-instrument-learn-through-use-NOGO | 40 pA | 100 pA |
| `_gap5_reverse_edge_hetero_depression_ltu_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | 2026-08-27-reverse-edge-heterosynaptic-depression-learn-through-use-NOGO | 40 pA | 100 pA |
| `_gap5_forward_band_homeostatic_scaling_ltu_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | 2026-09-16-gap5-forward-band-homeostatic-scaling-ltu-NO-GO-redirect-read-noise-floor | 40 pA | 100 pA |
| `_gap5_read_noise_floor_ltu_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | 2026-09-17-gap5-read-noise-floor-instrument-lane-is-read-noise-dominated-K18 | 40 pA | 100 pA |
| `_gap5_decorrelation_read_ltu_derisk` | `_gap5_ecker_adex_ca3_stdp_band_derisk` (`build_store`) | (no finding ran it) | 40 pA | 100 pA |

Each of the 8 affected findings above carries the `## Correction (PROVISIONAL, local model)` note dated 2026-10-01.
No re-runs.

`_gap5_onebrain_capstone.py`'s `ou_noise_sigma_pa=40.0` is dead code behind an always-false `hasattr` guard and was
not an intended value (it would reverse the line's own "kill the composer's OU noise" intent); no finding needs a
correction.

## Not affected

Mention the runner in prose, or import a helper, but did not run its OU-setting function: the 2026-07-25 CRITICAL
apical-R finding (runs the consolidation runner), the 2026-08-26 cls-interleaved finding (runs
`_cls_interleaved_consolidation_derisk`), the 2026-07-25 consolidation-frontier scoping finding (runs
`nmda_compositional_consolidation`), and the 2026-07-25 onebrain-sleepcycle-merge + 2026-08-26 sleep-replay-wirein
findings (run `_gap5_onebrain_sleepcycle_merge` / `_sleep_replay_flip_soak`, which import the phase-switch/roundtrip
helpers but do not run their OU-setting `build`/`_build` functions).
