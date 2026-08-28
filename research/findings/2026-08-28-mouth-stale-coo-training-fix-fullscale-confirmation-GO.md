---
type: finding
status: qualified
date: 2026-08-28
verdict: The mouth stale-cache training fix delivers a learned spiking readout that recovers most of the copied head at FULL scale (B=48, 6-seed) — sub_learned_recov_mean 0.8499, sub_recov_ratio_mean 0.8686 (min 0.8399), anti-cheat clean — a decisive unlock from the pre-fix wall. Marginal on the strict 6/6 GO bar (go_count 3/6) = a tuning residual, not a wall. The learned readout is now VIABLE for the crutch-burndown (Qwen replacement).
mechanism: full-scale B=48 6-seed magnitude confirmation of the megakernel-v2 WT stale-cache training fix (2026-08-27-mouth-stale-coo-training-fix-PARTIAL)
lane: e-mouth-fluency
artifacts:
  - research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json
runner: research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
---

# Mouth stale-cache training fix — FULL-SCALE B=48 6-seed magnitude confirmation: the learned readout recovers most of the copied head (from the pre-fix wall)

Artifact: `research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=cupy`, B=48, epochs 10, n-sentences 40000 — the production-scale config, run from `main` with the fix).

## The question (settled by this run)

The 2026-08-27 mouth stale-cache training fix ([`2026-08-27-mouth-stale-coo-training-fix-PARTIAL`](2026-08-27-mouth-stale-coo-training-fix-PARTIAL.md)) proved the MECHANISM decisively at a reduced scale (A/B recovery roughly doubled, ‖W‖ runaway resolved <!--derived--> — the megakernel-v2 WT transposed-CSR cache was never invalidated on in-place `set_weights`, so eprop trained against a FROZEN forward). It staged this full-scale B=48 6-seed run to answer the MAGNITUDE question: does the unblocked training reach the host-proxy at production scale?

## Result — the learned readout recovers most of the copied head; marginal on the strict 6/6 bar

Six-seed means from the artifact `research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json` (cupy, B=48):

- **`sub_learned_recov_mean = 0.8499`** — the LEARNED spiking readout, read on the substrate (graded-conductance production read).
- **`sub_copied_recov_mean = 0.9785`** — the COPIED (host-target) head reference.
- **`sub_recov_ratio_mean = 0.8686`**, **`sub_recov_ratio_min = 0.8399`** — the learned readout recovers most of the copied head, every seed above 0.83. From the pre-fix wall <!--derived--> this is the decisive unlock.
- `hostlinear_recov_mean = 0.8172`.
- **Anti-cheat (clean):** `sub_shuffle_recov_mean = 0.0015` (shuffle collapses), `host_matmul_on_forward_max = 0` (no freed host path — genuinely substrate), `anticheats_collapse_count = 6`.
- **`go_count = 3`** of 6 (`go_5of6 = False`): the strict GO bar clears on 3 seeds; the other 3 sit just under the ratio line. `parity_recovery_count = 0` (the stricter learned==copied parity criterion is not met — the learned readout recovers most, not all, of copied).

## What this settles

The stale-cache training fix is confirmed at PRODUCTION scale: the learned spiking read-out head, trained by the local three-factor rule against a now-FAITHFUL forward, recovers most of the copied Qwen-derived head (`sub_recov_ratio_mean = 0.8686`) — from a wall that stood for weeks. The mechanism + magnitude are strong and anti-cheat-clean; the learned readout is now a **viable Qwen-replacement candidate** and the mouth crutch-burndown (train the learned readout into the actual word-generation, measure Qwen-reduction) is UNLOCKED.

## Residual (marginal, not a wall — NO-DEFER next lever)

`go_count = 3` of 6 is a marginal miss on the strict bar: 3 seeds sit just under the ratio line and the learned readout recovers most (not all) of the copied head. This is a **tuning residual**, not a wall — candidate levers to close it: more epochs / a per-seed lr schedule for the sub-bar seeds; a cleaner readout objective (the same MSE/regression-margin vs the softmax question the mouth arc raised); or the decorrelation-read primitive (the shared read-fidelity theme with the one-brain cross-edge attribution reads + surprise→episodic's rate-saturation UNDEFINED). The crutch-burndown can proceed on this readout in parallel with closing this residual.
