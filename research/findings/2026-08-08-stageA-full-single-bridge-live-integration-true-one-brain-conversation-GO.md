---
type: finding
status: contributing
date: 2026-08-08
mechanism: stageA-conversation-integration-full-single-bridge
lane: E-language
runner: research/runners/_stageA_full_integration_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/stageA_full_integration_s42.json
---

# Stage-A FULL single-bridge live integration — the four faculties + the no-confab moat CONVERSE on ONE spiking brain, single-seed smoke (GO)

The four adversarially-verified Stage-A faculties — previously MODULAR (each on its own per-faculty
`SimulationBridge`, feeding a shared arbiter via host drive numbers) — are CONSOLIDATED onto ONE co-resident
spiking `SimulationBridge` running a REAL multi-turn conversational loop, per the integration contract in
[`2026-08-07-stageA-conversation-integration-DESIGN.md`](2026-08-07-stageA-conversation-integration-DESIGN.md).
This is the TRUE-ONE-BRAIN proof for Stage-A conversation: honesty, affect, curiosity, the 3-way arbiter, AND the
`CoResidentOneBrainComposer` no-confab moat are all region SLICES of a single bridge object in ONE process.

Runner: `research/runners/_stageA_full_integration_derisk.py`. Artifact:
`research/findings/raw/lanes/stageA/stageA_full_integration_s42.json`.

## The ONE bridge (single-seed 42, numpy/CPU)
<!--derived-->
`composer._merged IS the bridge`; 20 regions, 24961 neurons, ONE process. Region slices: `rf` (the composer VSA
moat substrate) + `workspace/workspace_fs/meta_schema/self_schema` (honesty relay) +
`arb_volunteer/arb_ask/arb_silent/arb_fs` (3-way WTA arbiter) + the P0.3 affect organ
(`affect_vplus/vminus/arousal` + `inh_plus/inh_minus` + `recall_pos/neg` + `speak_acc/silence_acc` + `wta_fs`) +
`cur_ask` (curiosity ask drive). One neuromodulator bus, group-scoped (seam 5): `appraisal_v+/v-/arousal` (affect)
+ `curiosity` from_novelty -> `group:cur_ask` (never scope=all). Per-faculty RNG isolation (seam 7).

## The multi-turn transcript — COMPOSED behavior in one loop on the one bridge
<!--derived-->
- T1 `known_fact` (positive mood): arb_volunteer wins, band=assert -> `"gladly apple big cat ; also big, cat"`
  (honest grounded answer + affect-colored warm tone + volunteered on-topic associates).
- T2 `novel_query`: arb_ask wins, band=MOAT -> `"what does big run ?"` (the brain ASKS its OWN wh-question; crave,
  don't refuse; the moat abstains — answer stays None).
- T3 `known_fact` (mood PERSISTS across the intervening novel turn): arb_volunteer wins, band=assert ->
  `"gladly cold come dog ; also come, dog"` (the positive affect state persists and still colors the answer).
- T4 `novel_query`: arb_ask wins, band=MOAT -> `"what does look run ?"` (curiosity is a standing drive, not a
  one-off). Affect persists across turns (v_state>0 on both known turns).

## Anti-cheat GO-gate (single seed 42, all live)
<!--derived-->
- (a) SINGLE-BRIDGE — every faculty is a slice of ONE bridge object; `composer._merged is bridge` True; all
  faculties present. GO.
- (b) COMPOSES-LIVE — the 4-turn transcript shows honest+colored answers on known turns AND curiosity wh-asks on
  novel turns AND the moat holds, in one loop; affect persists across turns. GO.
- (c) FM4 LIVE — a yoked high-arousal positive affect (read off the shared affect slices) mis-colored tone on
  11/11 below-assert candidates but flipped 0 to assert under the g_eff law; the naive affect-into-confidence path
  flipped 11/11 (the check can fail). The below-assert read is the honesty relay's self_schema spike rate on the
  shared bridge. GO.
- (d) MOAT LIVE 475/475 — the co-resident composer abstained on every unstored cue under a strong positive
  high-arousal mood (v_color 0.057, m_color 0.096); 0 added false-accepts, 0 manufactured answers. GO.
- (e) NO-PIECE-BREAKS-ANOTHER — every pairwise interaction holds under co-residence: FM4 (affect vs honesty),
  one-winner-per-turn (curiosity vs turn-taking), moat intact under affect+curiosity, affect coloring alive,
  curiosity want alive, honesty relay graded-confidence alive. The shared 3-way arbiter arbitrates (3 distinct
  correct winners); mutual-inhibition lesion collapses the winner margin in BOTH genuinely-contested regimes
  (novel_ask 0.772->0.072-class, forthcoming_volunteer 1.0->0.072); the reticent regime drives only silence above
  the ignition knee, so it is a non-contest with no margin to collapse (excluded, honestly). GO.
- (f) DEFAULT-OFF byte-identity — the faculty slices append AFTER the composer rf slice, so the composer neuron
  indices' firing thresholds are byte-identical with vs without the faculty slices (n_composer 24051 -> n_full
  24961). GO.

VERDICT GO (single-seed smoke; the parent runs the 6-seed sweep). Elapsed 165.6 s on numpy/CPU.

## VRAM / feasibility
<!--derived-->
One co-resident bridge at ~24961 neurons on the numpy/CPU backend (host RAM, not VRAM). The design flagged a VRAM
ceiling for 4-5 co-resident slices on GPU; on numpy the ceiling is host RAM and this build is comfortably within
it (the modular composer alone was ~28K neurons; the four faculty slices add only ~910 neurons). No VRAM wall for
Stage-A co-residence on CPU; a GPU port would re-test the ceiling.

## Honest-negatives (declared, not hidden)
- HONESTY SIGNAL SPLIT: the LIVE honesty floor in the loop is the co-resident composer's on-bridge cue-match
  (moat abstain -> MOAT; a cleared cue -> assert), composed under the g_eff LAW. The calibrated ACC/aPFC monitor
  (banked STEP 1) is co-resident as the workspace/meta/self relay and is exercised LIVE for FM4 + a graded
  self_schema confidence read; porting its full calibrated-monitor routing (fit + `_run_report`) onto the shared
  slices — instead of running it on STEP-1's modular bridges — is the remaining honesty consolidation step. The
  FM4 self_schema rate band is compressed on the co-resident substrate (assert/hedge thresholds ~0.055/0.054), <!--derived-->
  which the g_eff law fences off the assertion regardless of band width, but it is a narrower graded read than the
  modular relay's.
- Inherited STEP-2/3 boundaries: HOST-FED appraisal + the BISTABLE good/bad LATCH (binary tone) + HOST RENDER of
  the wh-frame / tone token.
- SHARED GLOBAL CFG: all faculties run under one global (parameter heterogeneity on; OU toggled on only inside
  affect-read windows). The no-piece-breaks-another check measures whether co-residence degraded any faculty vs
  its modular baseline; none degraded on this seed. One observed co-residence effect: affect `m_color` was NOT
  cleanly arousal-graded here (positive-mood arousal 0.3 vs 1.0 read near-identical), so novel turns use a NEUTRAL
  affect (mood 0) to let curiosity win the arbiter, matching the design (novel -> neutral affect -> ask). The
  binary-latch tone (positive vs neutral) is intact; the graded-arousal-forthcomingness axis is the STEP-2
  characterized boundary, not a new regression.

## What this closes / next
This is the Stage-A "one brain" milestone for conversation: the four faculties + the moat are co-resident slices
of a single spiking bridge and COMPOSE in a live multi-turn loop (honest answer + affect tone on known facts;
curiosity wh-ask on novel gaps; moat + honesty floor hold throughout). Next: (1) parent 6-seed sweep; (2) port
the calibrated ACC/aPFC monitor routing onto the shared relay slices (close the honesty-signal split); (3) the
graded circumplex affect (STEP-2 boundary) so tone/forthcomingness are graded, not binary.

## ✅ PARENT-VERIFIED (6-seed) — 6/6 GO
<!--derived-->
Parent 6-seed (42/43/44/100/101/102; aggregate `research/findings/raw/lanes/stageA/stageA_full_integration_6seed_aggregate.json`): ALL 6 seeds GO, composes_live=true every seed, 0 failed anti-cheats — the co-resident single-bridge TRUE-ONE-BRAIN conversation loop (honest + affect-coloured + curiosity-asking + moat-safe, one process) GENERALIZES. The self-driven honest affect-coloured curiosity CORE of open-ended conversation is a 6-seed GO. Honest boundaries unchanged (single-bridge live loop is real; the live honesty floor is the cue-match moat under g_eff with the calibrated monitor co-resident+exercised; bistable-latch affect; host-fed appraisal + host wh-render; scaffold burn-down pending; the maturation to human-like fluency = the next mission phase per 2026-07-01-fluid-conversation-mechanisms-roadmap + the faculty-load-bearing test).
