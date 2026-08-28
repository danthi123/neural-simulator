---
type: finding
status: correction
lane: infra-correctness (stale-weight-cache bug class, follow-through from e-language-mouth-read-snr)
date: 2026-08-27
mechanism: repo-wide sweep for OTHER victims of the megakernel-v2 transposed-weight-cache staleness bug fixed in
  22c05f41/d6c375de (sim/bridge.py::mark_weights_edited()) -- any runner that reuses ONE SimulationBridge across
  multiple in-place cp_connections.data edits + reads/steps, on a config satisfying `_step_megakernel_can_dispatch()`
  (GPU backend, Izhikevich, read_only_fast_step, ALL of hebbian/stp/homeostasis/stdp/structural/reward-mod/inhib-stdp
  OFF, no NMDA/GABA_B/coincidence-detection/v_apical/neuromodulator/experiment-engine side channels), WITHOUT calling
  mark_weights_edited() between the edit and the next read.
seeds: [42]
instrument: ad-hoc verification scripts against research/runners/navigate_to_see_then_answer.py,
  research/runners/funcint_perception_to_memory_trained_probe.py, research/runners/funcint_perception_to_memory_probe.py,
  research/runners/_wkv_mouth_readout_eprop_learn_derisk.py
runner: N/A (a sweep + fix pass, not a single runner)
verdict: PARTIAL -- 4 additional genuine victims found and FIXED (mark_weights_edited() added), all empirically
  confirmed (cached-vs-fresh WT-cache divergence measured directly, or an A/B where only the fix toggles). ~150 other
  candidates were systematically cleared via the dispatch guard's own conditions (CPU backend, an always-on plasticity
  flag, NMDA/GABA_B/coincidence-detection/v_apical allocation, or an experiment_engine attached and running). ONE
  fix (research/runners/_wkv_mouth_readout_eprop_learn_derisk.py) changes the headline number of an ALREADY-PUBLISHED
  GO finding (2026-08-14-fluid-mouth-readout-eprop-learned-GO.md, dated after the bug's vulnerable path defaulted
  on) -- REPORTED for controller review, not re-verdicted here. PARTIAL because the sweep, while thorough, is not
  a proof of exhaustiveness over ~150 files inspected by pattern-matching + guard-condition reasoning rather than a
  full per-file empirical run.
artifacts:
  - research/findings/raw/_stale_weight_cache_confound_sweep/evidence.json
  - research/findings/raw/_wkv_readout_eprop_learn_substrate_6seed.json
---

# Stale-weight-cache confound sweep: 4 more victims found + fixed (2 pre-date the bug, 2 don't -- one changes a published GO's headline number)

Artifact: `research/findings/raw/_stale_weight_cache_confound_sweep/evidence.json` (every empirical check below, with
raw numbers) + `research/findings/raw/_wkv_readout_eprop_learn_substrate_6seed.json` (the banked finding's own 6-seed
data, cited for the pattern-match in the last section).

## Context: the bug being swept for

`sim/bridge.py`'s read-only megakernel-v2 transposed-CSR cache (`_ensure_step_v2_transpose`, keyed on
`(id(conn), nnz)`) is invalidated ONLY on a STRUCTURAL change (synapse formation/elimination), NEVER on an in-place
weight edit (`cp_connections.data[...] = ...`). `enable_step_megakernel_v2` (default-on since 2026-07-23,
`33688c37a`) and `read_only_fast_step` are both default-ON, and `_step_megakernel_can_dispatch()` fires whenever a
bridge's config sits in the fully read-only regime (Izhikevich, GPU backend, and every one of
`enable_hebbian_learning` / `enable_short_term_plasticity` / `enable_homeostasis` / `enable_stdp` /
`enable_structural_plasticity` / `enable_reward_modulation` / `enable_inhibitory_stdp` False, plus no NMDA / GABA_B /
coincidence-detection / two-compartment-dAP / neuromodulator-subsystem / transmission-gain / graded-synapse /
divisive-norm / SSM / engram-recording / gate-coupling / data-bus / synapse-store / step-profiler / running
experiment-engine). Any code that writes `cp_connections.data` on such a bridge and then steps it again without
calling `sim/bridge.py::mark_weights_edited()` (added in d6c375de) transmits the STALE, pre-edit weights on every
read after the first. The original instance (the mouth eprop readout's per-step `set_weights`) is already fixed and
merged; this sweep hunts for OTHER instances across `research/runners/`, `sim/`, `webapp/`, `experiment/`.

## What was NOT a risk, and why (narrows a huge candidate set correctly)

Two properties of the underlying caches turned out to matter a lot and are worth recording so a future sweep does
not re-derive them:

1. **`_get_cached_coo()`'s `.data` field is a scipy/cupy sparse VIEW, not a copy**, for `tocoo(copy=False)` on a CSR
   matrix that never changes sparsity pattern -- verified directly on both scipy and cupy in this session
   (`coo.data is csr.data` is `True`; an in-place edit to `csr.data` is visible through the "stale" `coo.data`
   immediately, because they are the SAME array object). So the plain COO cache is NOT actually vulnerable to
   VALUE-only staleness from an in-place edit; every internal `sim/bridge.py` consumer of `_get_cached_coo()` was
   checked and reads `.row`/`.col` (structural, still valid) while reading weight VALUES fresh from
   `cp_connections.data` directly -- confirmed safe by inspection (no internal consumer reads `.data` off the cached
   COO object). The ONLY genuinely copied, disconnected cache is the megakernel-v2 **transposed** CSR (`_step_v2_WT_data`
   etc.) -- a real transpose necessarily reorganizes the data into new arrays, so it cannot be a view. This is why
   `mark_weights_edited()`'s COO-invalidation half is a defensive no-op in the overwhelmingly common case (in-place
   mutation), but matters when a caller REASSIGNS `cp_connections.data = new_array` instead of mutating it in place
   (breaks the view relationship) -- exactly what one of the four fixes below required.
2. **The dispatch guard is narrow on purpose**, and most of the ~150 files with `cp_connections.data[...] =`
   external edits found by grep are excluded by exactly one condition: (a) `SIM_BACKEND=numpy` forced (the whole
   `_emergeNN_*_pooler_derisk.py` / `_emergeNN_*_kernel_derisk.py` family, ~25 files, all forward-declare
   `os.environ.setdefault("SIM_BACKEND", "numpy")`); (b) an always-on plasticity flag for that bridge's whole run
   (`enable_reward_modulation` for `tan_ach_probe.py`'s reward-STDP probe; `enable_stdp`+`enable_reward_modulation`
   for `g11_bg_runner.py`'s DA-gated-WTA fs->motor scaling, verified live -- `cfg.enable_stdp=True` /
   `enable_reward_modulation=True` are set unconditionally at bridge-build time and never toggled off before the
   in-place edit; the PRODUCTION BG organ, `research/runners/bg_action_selection_production_organ.py`, does not
   even import `g11_bg_runner` and never writes `cp_connections.data` at all); (c) `cp_v_apical` allocated
   (`enable_two_compartment_dap=True`) -- blocks dispatch outright and covers the whole D5/BTSP/consolidation family
   (`webapp/continuous_engine.py`'s `consolidate_used_memory`/`consolidate_sleep_replay`, `nmda_compositional_consolidation.py`,
   the `_realcorpus_*_spiking_derisk.py` pair, the `_gap5_*`/`_d5_*`/`_consol_*` reactivate-based runners); (d) an
   `experiment_engine` attached with `is_experiment_running=True` -- covers the whole `g1`..`g9` runner family
   (`g1_v2_runner.py`, `g5_v3_runner.py`, `g6_runner.py`, `g8_runner.py`, `g9_runner.py` all attach a running
   `ExperimentEngine` before their per-step host-computed weight writes; verified live for `g1_v2_runner.py`); (e)
   `cfg.enable_nmda=True` set unconditionally -- covers the whole merged nav+conversation family
   (`nav_conv_merged_bridge.py` line 1470 and everything built on `MergedNavConvAgent`:
   `_grounded_speech_action_loop_derisk.py`, `_developmental_vocal_convention_derisk.py`); or (f) the in-place edit
   happens BEFORE the bridge is ever stepped in the read-only regime (so the WT cache is still unbuilt/`None` and
   the first post-edit step builds it fresh) -- covers `spoken_instruction_nav.py`'s `_lesion_command_route`,
   `funcint_lang_to_action_probe.py`'s `_run_lesion`, and `research/runners/onebrain_merge_production.py`'s
   `cp_connections.data[:] = conn_snap` restore (additionally protected: its per-region `enable_homeostasis=True`
   on the surprise/world-model organs allocates `cp_homeostasis_neuron_mask`, which the guard also checks). Full
   per-candidate reasoning and the exact grep/line evidence is in the session transcript; `evidence.json` carries
   the empirical checks only (the SAFE verdicts above are argued from the dispatch guard's own code, not each
   individually re-run end to end -- flagged honestly as the sweep's residual, not claimed as exhaustively tested).

## The 4 confirmed VICTIMS (all fixed with `mark_weights_edited()`, all empirically verified)

All four share one shape: build a bridge that ends up in the fully read-only regime, STEP it once (which silently
warms the WT cache from the CURRENT weights), THEN externally edit `cp_connections.data` in place for an anti-cheat
lesion or a second weight-set comparison, THEN step again expecting the edit to be live.

**1. `research/runners/navigate_to_see_then_answer.py::_lesion_route`** -- the "(B) perception->memory" milestone's
primary anti-cheat: zero the trained `cortex_it -> language_output` route, expect recall to collapse. `run_episode`
(navigate + perceive + tag) already stepped the bridge in the read-only regime before the lesion. Direct measurement
(`evidence.json`): `_step_megakernel_can_dispatch()` is `True` right before the lesion; the CACHED transposed-weight
array's sum (478196.44) differs from an independently fresh-rebuilt one (458284.05) by exactly the pre-lesion route's
weight sum (19912.39) -- the lesion is invisible to the fused kernel without the fix. Post-fix, the rebuilt cache
matches the independent fresh rebuild exactly. **The banked finding (`2026-06-16-navigate-to-see-then-answer.md`,
GO, LESION recall 0.00 on all 6 seeds) is NOT retroactively invalidated** -- it was produced 2026-06-16, five weeks
before `enable_step_megakernel_v2` existed (let alone defaulted on, 2026-07-23), so that run never touched the fused
path. The code was a live landmine for any FUTURE re-run, now fixed.

**2. `research/runners/funcint_perception_to_memory_trained_probe.py::_run_lesion`** -- the direct ancestor
`navigate_to_see_then_answer.py` "reused VERBATIM" the lesion idiom from. Same shape: `encode_percept_engram` steps
the bridge (warms the cache) before the lesion. Measured: cached WT sum 51653.31 vs fresh 31382.50 pre-fix (stale);
post-fix the lesion correctly collapses recall to 0/4 objects (was previously true in the banked
`2026-06-16-funcint-perception-to-memory-trained-map.md`, ALSO five weeks pre-flip -- not retroactively invalidated).

**3. `research/runners/funcint_perception_to_memory_probe.py::_run_lesion`** -- the "clean labeled-line" sibling of
#2 (no training pass, otherwise identical shape). Measured: cached WT sum 62322.50 vs fresh 31382.50 pre-fix; post-fix
recall collapses to 0/4. Same 2026-06-16 pre-flip banked finding, not retroactively invalidated.

**4. `research/runners/_wkv_mouth_readout_eprop_learn_derisk.py::LearnedReadout.set_weights`** -- the EARLIER
("gap#1/A1") rung of the mouth eprop read-out, a sibling of the already-fixed
`_wkv_mouth_readout_eprop_batched_substrate_derisk.py`. `_build_bridge` sets stdp/hebbian/stp/structural/homeostasis/
reward-mod/nmda all False (the identical vulnerable profile). `_eval_substrate` calls `s.set_weights(W)` for the
LEARNED weights, loops `s.margin_from_h()` (which steps the bridge) over the held-out set, THEN calls
`s.set_weights(hw.copy())` for the COPIED reference head and loops again -- two edits, one intervening step-warmed
cache, no invalidation. This `set_weights` is ALSO worse than a plain in-place edit: it REASSIGNS
`self._b.cp_connections.data = xp.asarray(...)` (a new array object) rather than mutating in place, which additionally
breaks the COO-cache's normal view-safety (see above) -- both caches were stale here, not just the WT transpose.
**This is the one that changes a published number.** `research/findings/2026-08-14-fluid-mouth-readout-eprop-learned-GO.md`
is dated AFTER the 2026-07-23 default-on flip and is NOT protected by the pre-flip exemption above.

## The quantified confound (decisive A/B, only the fix toggles)

Smoke run (`--smoke --seeds 42 --feature host`), current (fixed) code vs a monkeypatched reproduction of the
pre-fix `set_weights` (no `mark_weights_edited()` call, otherwise byte-identical):

| | `sub_learned_recov_mean` | `sub_copied_recov_mean` | `sub_recov_ratio_mean` |
|---|---|---|---|
| fix OFF (reproduces pre-fix code) | 0.8357 | 0.8293 | 1.0077 |
| fix ON (current code) | 0.8357 | **0.9665** | **0.8647** |

`sub_learned_recov_mean` is unaffected (it is always the FIRST `set_weights` call on a freshly-built bridge, so its
own read is never stale). `sub_copied_recov_mean` -- the SECOND call, the one downstream of the un-invalidated cache
-- moves by <!--derived--> +0.137 (0.9665-0.8293, `delta_sub_copied_recov_mean` in `evidence.json`) once the fix is
applied, and the ratio the GO criterion actually gates on
(`sub_recov_ratio_mean >= 0.85`) drops from 1.0077 to 0.8647: still a pass on this single seed, but with a much
narrower margin, and a materially different scientific claim ("the learned read-out matches or beats the copied
reference" vs "the learned read-out recovers ~86% as well as the reference").

**The banked 6-seed data matches the BUGGY pattern, not the corrected one.**
`research/findings/raw/_wkv_readout_eprop_learn_substrate_6seed.json`'s summary reads
`sub_copied_recov_mean=0.8844`, `sub_recov_ratio_mean=1.0018` (min 0.9992) -- a ratio pinned near 1.0 across every
seed, exactly the fix-OFF pattern measured here (1.0077), not the fix-ON pattern (0.8647, copied clearly above
learned). This is strong circumstantial evidence the banked 6-seed run was itself measured through the stale
cache. **This is flagged for controller review, not re-verdicted here**: the fix is applied and verified correct in
isolation (this sweep's job), but re-running the full banked 6-seed suite under the fix -- and deciding whether the
GO verdict, its margin, or its stated numbers need updating -- is a controller-level call on an already-published
finding, consistent with the task's "high-stakes, finding-invalidating result -> report, don't auto-amend" boundary.

## Determinism / byte-identity

`mark_weights_edited()` itself is unchanged (already tested 11/11 in `tests/test_determinism.py` by the original
fix) and byte-identical when weights did not actually change between reads (the caches simply rebuild to identical
data). All four call sites added here are pure ADDITIONS after an edit that was already happening -- no existing
code path's behavior changes except that the edit now actually reaches the substrate. Verified directly for victim
#1 (post-fix rebuilt WT cache sum matches an independently-fresh-rebuilt bridge's WT cache sum exactly, to full
float64 precision) and for victims #2/#3 (post-fix lesion recall collapses to the anti-cheat's expected 0/4, matching
the pre-flip banked findings' own expected behavior).

## Residual / not exhaustively checked

This sweep is thorough but not a proof of completeness. In particular: (a) `sim/bridge.py`'s OWN internal STDP/
Hebbian/BTSP update paths were checked and found self-consistent (they read `cp_connections.data` fresh, never
through a cache, and only run when the SAME flags that gate them also block megakernel-v2 dispatch -- so they
cannot collide with the fused path in one config) but this is an architectural argument, not a re-run of every
internal path; (b) roughly a dozen lower-priority candidates (`_productive_morphology_*`, a handful of
`_laneC_*`/`_laneD_*`/`_i7_*` files that call `_get_cached_coo()` directly) were cleared by the same
always-on-plasticity-flag or `cp_v_apical` reasoning as the bulk of the family they belong to, but not individually
re-run end to end; (c) the sweep covered `research/runners/`, `sim/bridge.py`, `webapp/`, and `experiment/engine.py`
exhaustively for the `cp_connections.data[...]=` grep pattern, but a bridge that goes through a DIFFERENT mutation
API (e.g. a future `set_weights`-style helper not yet written) is obviously not covered by a pattern sweep.
