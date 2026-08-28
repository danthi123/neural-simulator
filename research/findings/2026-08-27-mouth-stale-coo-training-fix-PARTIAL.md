---
type: finding
status: partial
lane: e-language-mouth-read-snr
date: 2026-08-27
mechanism: invalidate the read-only megakernel-v2 transposed-CSR (WT) transmission cache on an in-place synaptic weight edit, via a new sim/bridge.py::mark_weights_edited() the mouth eprop read-out's per-step set_weights calls -- so the batched substrate forward transmits the CURRENT weights, not the first-loaded ones.
seeds: [42, 43]
instrument: research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
runner: research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
verdict: PARTIAL (mouth-unblocking) -- the ||W||->cap runaway that blocked the learned spiking mouth for weeks was a STALE-CACHE TRAINING ARTIFACT, not a substrate limit. The read-only megakernel-v2 WT cache is keyed on (id(conn), nnz) and was NEVER invalidated on an in-place weight edit, so every substrate forward after the first transmitted the FIRST-loaded weights -- the eprop learning signal was computed against a FROZEN forward. Fixed by mark_weights_edited(). On the config that reproduces the 0.37 baseline (fix OFF: recov 0.3569, ||W|| pinned at the 40 cap), turning the fix ON lifts substrate recovery to 0.7038 and ||W|| converges BELOW the cap (27.1/29.1, toward the host-proxy ~24), host_matmul_on_forward==0 throughout (genuinely substrate, not a freed host path), shuffle collapses. PARTIAL because the decisive full-scale 6-seed (B=48, epochs=10, exact baseline config) is STAGED on gpu_queue, not yet landed; the 2-seed A/B is at a reduced but wall-reproducing scale. Determinism suite green (11/11) and byte-identical on the no-weight-change path.
artifacts:
  - research/findings/raw/_mouth_stale_coo_training_fix/mechanism_verify.json
  - research/findings/raw/_mouth_stale_coo_training_fix/ab_reduced_fixOFF_2seed.json
  - research/findings/raw/_mouth_stale_coo_training_fix/ab_reduced_fixON_2seed.json
  - research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json
---

# The mouth eprop TRAINING ||W||->cap runaway was the SAME stale-cache artifact as the read wall: the substrate forward was frozen. Fixed; recovery ~doubles and ||W|| converges below cap (full-scale 6-seed staged)

Artifact: `research/findings/raw/_mouth_stale_coo_training_fix/mechanism_verify.json` (stale-cache reproduction + byte-identity, seed 42, cupy) + `research/findings/raw/_mouth_stale_coo_training_fix/ab_reduced_fixOFF_2seed.json` + `research/findings/raw/_mouth_stale_coo_training_fix/ab_reduced_fixON_2seed.json` (the decisive A/B: identical reduced config, only the fix toggles) + `research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json` (the prior 0.3744 / ||W||=40 baseline).

## The bug, corrected in one detail from the discovery finding

The 2026-08-27 read-wall discovery blamed `sim/bridge.py::_get_cached_coo()`. The real stale cache is the read-only megakernel-v2 transposed-CSR (`_ensure_step_v2_transpose`, keyed on `(id(conn), nnz)`): `enable_step_megakernel_v2` and `read_only_fast_step` are both default-ON and the read-out config satisfies `_step_megakernel_can_dispatch` (Izhikevich, all plasticity/side-channels off), so transmission runs the fused kernel over the cached WT. An in-place `cp_connections.data` edit leaves id+nnz unchanged, so the WT is NOT rebuilt and the kernel transmits the pre-edit weights. Direct proof (`mechanism_verify.json`, `megakernel_v2_can_dispatch:true`): read a probe A then B on one build -- fix OFF, read B's corr to host-A is 0.5216 while to its OWN host-B is -0.0124 (it transmits A); fix ON, read B's corr to host-B is 0.5197, to host-A 0.015 (it transmits B), matching a fresh build (0.4195).

## The fix (correct + general, byte-identical off the buggy path)

New `sim/bridge.py::mark_weights_edited()` clears `_step_v2_WT_key` (forcing a WT rebuild) and calls `_invalidate_coo_cache()`; the read-out's `set_weights` (research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py) calls it after each in-place edit. It is the weight-only counterpart of the structural `_invalidate_coo_cache`. Byte-identical when weights do NOT change between reads: `mechanism_verify.json` shows a same-weight re-set read is hash-identical with the fix on vs off (m_b `d54acbc86e72dbc3` both). `tests/test_determinism.py` passes 11/11.

## The mouth training result (the point)

Prior baseline (`_wkv_readout_eprop_batched_substrate_6seed.json`): substrate recov 0.3744, ||W||=40.0 pinned at the synaptic-scaling cap every seed, wcos 0.13 -- the "||W||->cap runaway". The A/B reproduces this at a reduced, wall-faithful config (B=16, epochs=4): fix OFF gives recov 0.3569 and ||W||=40.0 pinned (`ab_reduced_fixOFF_2seed.json`). Turning the fix ON at the SAME config (`ab_reduced_fixON_2seed.json`): recov 0.7038 (seed42 0.7456, seed43 0.662), ||W|| 27.1/29.1 -- converged BELOW the cap toward the host-proxy ~24 -- wcos 0.20. Anti-cheat: `host_matmul_on_forward`==0 on both arms (the forward is the substrate read, not a freed host path), shuffle collapses to ~0.0006. So the frozen-forward starvation was the runaway's cause; with a live forward the eprop signal steers W and the cap stops binding.

## What is settled vs staged

Settled: the mouth eprop TRAINING loop shared the read wall's stale-cache bug; a live substrate forward roughly DOUBLES recovery and dissolves the ||W||->cap runaway, at host_matmul==0. Staged (why PARTIAL): the decisive full-scale 6-seed (B=48, epochs=10, exact baseline config) is queued on gpu_queue for the magnitude/GO-bar verdict (does it reach the host-proxy ~0.86 and cross the integrated bar); the A/B here is 2-seed at reduced scale. This unblocks the learned spiking mouth (Qwen-replacing read-out) after weeks -- pending the full-scale confirmation.
