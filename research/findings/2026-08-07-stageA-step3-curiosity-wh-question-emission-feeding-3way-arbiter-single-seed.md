---
type: finding
status: contributing
date: 2026-08-07
mechanism: stageA-conversation-integration-curiosity-ask
lane: E-language
runner: research/runners/_stageA_step3_curiosity_ask_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/stageA_step3_curiosity_ask_s42.json
---

# Stage-A STEP 3 — the brain emits its OWN wh-questions (crave, don't refuse) feeding the 3-way arbiter, single-seed smoke (GO)

STEP 3 of the Stage-A conversation-integration stack (`research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md`, seam 2 arbiter + seam 4 curiosity). This is the mission-central step for open-ended (not Q&A) conversation: the brain INITIATES — it asks about what it does not know — rather than only answering. Builds on STEP 0/1 (`_stageA_foundation_honesty_arbiter_derisk.py`, the co-resident substrate + honesty floor + the 3-way {volunteer|ask|silent} WTA arbiter) and STEP 2 (`_stageA_step2_affect_coloring_derisk.py`, affect → arb_volunteer/arb_silent). Reuse-by-import, NO `sim/` edit, `SIM_BACKEND=numpy`, `cfg.seed`, additive/default-off.

## What was built (reuse-by-import)
- The ask-DRIVE is the on-bridge DR-1 SPIKING curiosity (`_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge`): `current_novelty_signal` (Bogacz-Brown gate novelty) → the `curiosity` neuromodulator (`from_novelty` rule) → `excitability_drive` scope=group:ask → ASK-pool spikes read off `cp_firing_states[ask]` (Hz). The wanting is a spike-rate, not a host `if novel` flag.
- That spiking want feeds the SHARED 3-way arbiter's `arb_ask` pool (`_stageA_foundation_honesty_arbiter_derisk.build_arbiter_bridge`/`run_arbiter`), CO-RESIDENT with the STEP-2 affect wire-in (`m_color` → arb_volunteer/arb_silent). One winner per turn.
- When `arb_ask` wins on a NOVEL concept, the brain EMITS a wh-question whose CONTENT word is decoded from the on-bridge naming map's WORD-POOL SPIKE COUNTS (`_grounded_message_to_word_onbridge_derisk.name_from_spikes`, the 6-seed PARENT-VERIFIED naming GO), NOT from WKV generation. The wh-frame is a fixed host scaffold; the content is brain-native.
- MOAT INVERTED, not broken: on a NOVEL cue the ACTION becomes ASK instead of a bare refusal, but the answer stays None — the real CoResidentOneBrainComposer no-confab moat still abstains, never confabulates.

## Result — single seed 42, VERDICT GO (all 7 anti-cheats)
<!--derived--> from `research/findings/raw/lanes/stageA/stageA_step3_curiosity_ask_s42.json`:
- (a) CRAVE-ON-SPIKES: corr(epistemic-gap, SPIKING want) = 0.997 (>= 0.9); want mean 84.0 Hz intact → 5.3 Hz under the curiosity-modulator lesion (collapses; a host flag would not). 93.7% of the want is attributable to the modulator (intact vs lesion).
- (b) MOAT INVERTED-NOT-BROKEN: 475/475 unstored cues abstain, 0 confabulated answers, 0 added false-accepts, and 475/475 the brain ASKS instead of refusing — the moat's action-inversion, on the REAL composer.
- (c) WH-TARGETS-THE-GAP: spike-decode target accuracy 1.00; the wrong-gap PERMUTATION control drops to 0.25 (control fails, as required). Emitted transcript: "what is apple ?", "what is seed ?", "what is honey ?", "what is water ?".
- (d) BRAIN-NATIVE WORDS: the content-word pool index comes from `name_from_spikes` (word-pool spike counts); WKV is only the fixed pool→token articulatory alphabet, never invoked to generate the word.
- (e) ARBITER 3-WAY: distinct winners novel→arb_ask, forthcoming→arb_volunteer, reticent→arb_silent; intact min winner-margin 0.362637 vs 0.0710383 under the mutual-inhibition lesion (contention collapses; 80.4% of the margin owed to inhibition). arb_ask competes co-resident with the affect-driven volunteer/silent, one winner/turn.
- (f) DEFAULT-OFF byte-identity: the cur_ask slice appends LAST (350 → 430 neurons); baseline firing thresholds byte-identical.
- Supporting: the LP-max ask SELECTOR (which concept to ask) is GO this seed (7 noisy asks vs novelty-max's 81), reported as a CPU proxy.

## The moat is INVERTED, not broken
On a novel gate read the brain used to (silently) abstain; now it ASKS a wh-question about the gap. The no-confab moat is untouched: `query_patient` still returns None on every unstored cue (475/475), so the brain never manufactures an answer. Asking is the action-inversion the mission asked for (crave, don't refuse), achieved WITHOUT weakening the moat.

## Honest-negatives (declared, not hidden)
- ON-BRIDGE LP MEMORY is FRAGILE (1/6; `2026-07-30-lane-B-curiosity-DR1-onbridge-6seed-GO`). So WHICH concept to ask (the learning-progress SELECTOR) is the LP-MAX CPU PROXY (6-seed GO, `2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO`), which may host-TD-fallback on-bridge. The ask DRIVE (whether the gate craves) IS spiking (anti-cheat a).
- HOST RENDER: the wh-frame "what is ___ ?" is host phrasing (the fixed language scaffold, analogous to the body acting on motor output); only the CONTENT word is the brain's naming-map spike decode. The moat action-inversion frame "what does <agent> <action> ?" reuses cue words the brain already holds.
- MODULAR-BRIDGE SMOKE: the four faculties (curiosity organ, shared arbiter, affect organ, naming map) run on their own numpy spiking bridges and feed one shared 3-way arbiter; the byte-identity test proves the cur_ask slice appends byte-unchanged. Full single-bridge live integration is the parent/next step (matching the STEP-0/1/2 modular-bridge pattern). Single seed; the parent runs the 6-seed sweep.

## Reproduce
```
PYTHONPATH=$PWD SIM_BACKEND=numpy python -m research.runners._stageA_step3_curiosity_ask_derisk \
  --seed 42 --out research/findings/raw/lanes/stageA/stageA_step3_curiosity_ask_s42.json
```

## ✅ PARENT-VERIFIED (6-seed) — 5/6 GO, moat 6/6
<!--derived-->
Parent 6-seed (aggregate `research/findings/raw/lanes/stageA/stageA_step3_curiosity_6seed_aggregate.json`): 5/6 GO (42/43/44/100/101); seed 102 NEGATIVE on the SOLE failed check `e_contention_collapses_on_lesion` (the arbiter mutual-inhibition margin did not collapse cleanly under lesion on that seed — an arbiter-robustness detail). SAFETY holds on ALL 6: moat inverted-not-broken 475/475 (0 confab), crave-on-spikes, brain-native wh-words, wh-targets-gap every seed. Curiosity ask is 6/6-SAFE + 5/6 fully-GO; the arbiter-margin robustness on marginal seeds is the residual (shared with the affect arbiter).
