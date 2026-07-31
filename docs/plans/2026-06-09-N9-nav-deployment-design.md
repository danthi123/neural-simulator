---
type: plan
status: live
date: 2026-06-09
---

# N9 RPE Loop — Nav Deployment Design

**Date:** 2026-06-09 (code-architect pass, controller-reviewed)
**Goal:** port the validated N9 reward-prediction-error loop (self-org place code + FS-PING volley + weighted-coincidence plateau + pair-then-reward value-learning + GABA_B r−V subtraction, validated in `research/runners/n9_place_graded_critic_stage2_derisk.py`) into the online nav loop (`research/runners/g11_bg_runner.py`, `build_bg_brain_regions(enable_neural_critic=True, spiking_snc=True)`), replacing the host-rendered Gaussian `vs_place_context` place code (a BRAIN-BASED-ONLY shortcut). Runner-only; the one sim/ piece (`coincidence_weighted_drive`, `e0818d2d`) is already on main.

## Architecture (after integration)

```
ENVIRONMENT (host — legitimate): grid (x,y,gx,gy) → renders bearing/distance cosines; body ← motor_{N,E,S,W} spikes
BRAIN (all neural):
  landmark_sensors  --[plastic, gate=landmark_to_place]-->  place (IZH2007_HIPPO_PYRAMIDAL, n=200)  <-->  place_fs (FS, n=24)
      (self-org competitive threshold-WTA → sparse distinct fields; FS-PING gated OFF during self-org via transmission_gate=place_fs_gate)
  place  --[plastic, gate=value_input, coincidence_detector=True, coincidence_weighted_drive=True]-->  striosome_value (MSN-D1, n=200)
      (PAIR: place+SNc tonic → up-state + silent eligibility; REWARD: place+SNc burst → DA AFTER pairing → LTP)
  striosome_value  --[receptor=gaba_b, transmission_gate=critic_snc_window]-->  snc (IZH2007_DOPAMINE, n=40)
      (arithmetic GABA_B subtraction → δ = r − V)
  snc spiking → neuromodulator DA → R-STDP on the (existing, unchanged) actor: cortex→str_D1→gpi→thal→motor
```
The host `vs_place_context` Gaussian injection (`g11_bg_runner.py` ~lines 5102–5126) is DELETED. The host then only renders bearing/distance cosines into `landmark_sensors` (legitimate sensory) + reads motor spikes (body).

## Integration gaps + bridging

1. **Place code (remove host Gaussian).** Delete the `vs_place_context` Gaussian injection; keep the existing `landmark_sensors` drive (legitimate). Add a self-org STEP-1 phase in `_run_critic_warmup`: open `landmark_to_place`, close `place_fs_gate`, sweep positions driving `landmark_sensors` ~2000 steps until competitive WTA forms stable fields, then freeze `landmark_to_place` + open `place_fs_gate`. During nav the `place` pool fires from the frozen weights — no host place computation.
2. **Online pair-then-reward.** Restructure `_run_critic_warmup` STEP-2 to the validated pair_then_reward (PAIR: place + SNc tonic, ~100 steps; REWARD: place + SNc burst after `reward_delay_steps`, ~40 steps; reset `g_gabab` + SNc membrane each trial). Online: on goal-reach, hold place active `reward_delay_steps` before the SNc burst (the Yagishita pairing-then-DA).
3. **Wire `place_fs` + coincidence plateau + GABA_B.** In `build_bg_brain_regions` `enable_neural_critic` block: rename `vs_place_context`→`place` (IZH2007_HIPPO_PYRAMIDAL, enable_homeostasis), add `place_fs` (FS, n=24); add `landmark_sensors→place` (plastic, gate `landmark_to_place`), FS-PING reciprocal `place↔place_fs` (the `place_fs→place` arm gets `transmission_gate="place_fs_gate"`); update `place→striosome_value` to `coincidence_detector=True, coincidence_weighted_drive=True`; the GABA_B `striosome_value→snc` is already correct. New params: `n_place_fs=24`, `n_place=200`, `coincidence_threshold` (validated ~12–26 readout / 4 count-train), `selforg_steps=2000`, `selforg_n_positions=40`, `reward_delay_steps=8`.
4. **BRAIN-BASED-ONLY audit.** REMOVE: host Gaussian place code (gap 1) + the position-blind convergent up-state (`enable_convergent_upstate` — the de-risk found it caps grading ~1.2×; hard-gate False when `enable_neural_critic`). REMAINING teacher scaffold: the host injects the SNc burst on goal-reach (the unconditioned-stimulus DA) — biologically the US→VTA; the full-neural follow-on is a `goal_sensor → reward_region → snc` chain (out of scope this pass, flagged). REMAINING legitimate: bearing/distance cosines → landmark_sensors (environment), motor spikes → movement (body).

## Cheap-first integration de-risk (single seed 42, ~20 min, CUBLAS pinned)

- **Stage A (place self-org at nav scale):** FIRE (place ≥5 Hz near goal), PLACE-GRADED (diff-cos ≥0.05 near vs far goal; de-risk got 0.12), ACTOR-NOT-PERTURBED (nav SUM within 10% of `enable_neural_critic=False`). Assert before proceeding.
- **Stage B (value-learning warmup):** LEARNS-V (place→value near ≥1.5× far), GABA_B gap (snc_rate near < far = δ=r−V<0 at the learned goal), lesion (zero g_gabab → gap → ~1.0).
- **Stage C (full nav + anti-cheats):** nav with critic vs without (no >15% regression; look for improvement). Anti-cheats: place-shuffle (permute place→RF-position → improvement collapses), sensor-ablation (zero landmark_sensors → place silent → regress to random walk), GABA_B lesion in nav (zero g_gabab/step → regress if the RPE is real).

## Build sequence (bite-sized, runner-only)

- **Phase 0:** confirm `coincidence_weighted_drive` on main (it is, `e0818d2d`); set `os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"` at the top of g11_bg_runner BEFORE any CuPy import.
- **Phase 1:** add params + the `place`/`place_fs` regions + the pathways (above) in `build_bg_brain_regions`; hard-gate `enable_convergent_upstate` off under neural critic.
- **Phase 2:** refactor `_run_critic_warmup` STEP-1 self-org (open landmark_to_place + close place_fs_gate → sweep → freeze + open place_fs_gate); delete the host Gaussian injection; run Stage A smoke (assert FIRE + GRADED + actor-not-perturbed).
- **Phase 3:** add STEP-2 pair_then_reward warmup; run Stage B smoke (LEARNS-V + GABA_B gap + lesion).
- **Phase 4:** online — `reward_delay_steps` hold before the SNc burst on goal-reach; `_reset_snc_subtraction_state` at each goal-reach reset.
- **Phase 5:** Stage C smoke (single seed + anti-cheats) → 3-seed → 6-seed nav A/B.

## Honest risks
1. Online (continuous) vs discrete-trial regime — the Yagishita window may be violated in fast segments; the `critic_snc_window` sawtooth + `reward_delay_steps` mitigate (medium risk).
2. Place-code self-org at nav scale — the nav landmark formula differs from the de-risk's; Stage A GRADED gate catches insufficient diversity.
3. Actor-critic interaction — the +224 critic neurons feed the global DA; the GABA_B over/under-suppressing SNc tonic miscalibrates actor R-STDP; Stage A actor-not-perturbed catches it.
4. CUBLAS non-determinism at nav scale — the pin is mandatory + must precede `import cupy`.
5. The `_reset_snc_subtraction_state` gap online — residual g_gabab carrying across episodes silently quiets the SNc; the goal-reach reset (Phase 4) is critical.
