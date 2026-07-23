# General-step megakernel — removing the launch-bound ceiling from the sim's inference loop (design, 2026-07-23)

**Origin:** a clock-offset A/B proved the general `_run_one_simulation_step` loop is LAUNCH-BOUND — steps/sec is flat
(~825-889) across a 10× network-size increase AND did not respond to a GPU clock bump (compute got faster, the loop
didn't). Deep-research (4-angle workflow + synthesis, 2026-07-23) designed the fix. This banks the design; the build is a
focused, high-care follow-on.

## Verdict: MEGAKERNEL (hand-authored `cp.RawKernel`), NOT literal CUDA-graph capture
Two hard walls kill literal CUDA-graph capture (both confirmed against our own RF prototype):
1. **cuSPARSE SpMV cannot be captured** by CuPy — our synaptic drive is a CSR matvec; the exact wall the 2026-06-17 RF
   megakernel hit (`enable_rf_cudagraph` is a documented misnomer — it's a megakernel, not a graph).
2. **Graph capture freezes the OU-noise RNG** — `cp.random.randn` per step would replay identical noise = WRONG.
⇒ the code-gen/megakernel route every serious SNN sim takes (GeNN, NEST-GPU, SpikingJelly, Spike). Reuse the shipped RF
precedent (`_RF_MEGASTEP_SRC` / `_rf_resonate_steps_megakernel`, `sim/bridge.py:5959-6032`).

## The capturable regime (why the win is bounded + safe)
Only the INFERENCE / read-only regime is fusable (learning/STP/homeostasis/structural/neuromod all OFF). There the step
is ~8 ops, and the two per-step device→host syncs (`bool(cp_prev_firing_states.any())` @6080, `bool(spike_count>0)`
@7073) are ALREADY byte-identically removable via the shipped `read_only_fast_step` flag. Net: **~9 launches + 2 syncs
→ 2 launches (randn + megakernel) + 0 syncs**.

## Fuse into ONE RawKernel (one thread per postsynaptic neuron j, all in-place on fixed-shape cp_* arrays)
Exact math to replicate, in order:
1. conductance decay: `g_e *= decay_e; g_i *= decay_i` (cached `_cached_decay_e/_i`).
2. synaptic current: `I_syn = g_e*(E_e - v) + g_i*(E_i - v)` (`fused_conductance_decay_and_current`, E_e=`syn_reversal_potential_e`, E_i per-neuron or scalar).
3. E/I-split transpose matvec ADD from prev-fired spikes: `g_e[j] += Σ_exc WT[j,i]*fired[i]`, `g_i[j] += Σ_inh ...` using a PRE-BUILT transposed CSR `WT=cp_connections.T.tocsr()` (static in-regime) + a per-neuron exc/inh int8 flag; matches the RF kernel's `for k in [indptr,indptr+1)` gather. (Weights carry the `effective_synaptic_strength` scaling; the guard requires no runtime gain/gate change.)
4. (optional NMDA Mg-block current — v2; guard-OFF in v1.)
5. total input + externally-supplied OU noise sample.
6. Izhikevich-2007 dynamics (`fused_izhikevich2007_dynamics_update`): `C_safe=(C==0?1:C); dv=(k*(v-vr)*(v-vt)-u+I)/C_safe; du=a*(b*(v-vr)-u); v+=dv*dt; u+=du*dt` over static `cp_izh_C/k/vr/vt/a/b`.
7. threshold-select: `fired = v_new >= cp_neuron_firing_thresholds`.
8. fast_spike_reset (`fast_spike_reset` path @6886): `v=(fired?cp_izh_c_reset:v); u=(fired?u+cp_izh_d_increment:u)`; refractory `fired→period-1`, else `max(timer-1,0)`.
9. write `cp_firing_states[j]`, `cp_prev_firing_states[j]`.
OUTSIDE the kernel: the OU `cp.random.randn(n)` draw (separate pre-launch each step — keeps RNG bit-faithful, no in-kernel Philox), the device-only `cp.sum(fired)` count, and everything the guard requires ABSENT (STP, all csr-rebuild gates, STDP/Hebbian/BDSP/BTSP, reward mod, structural plasticity [the only CSR-mutating op], homeostasis, recording/data_bus/engram/neuromod/experiment/synapse_store).

## Implementation plan
1. CONFIG: `enable_step_megakernel: bool = False` (default off ⇒ byte-identical; alias the requested `enable_step_cudagraph`).
2. DISPATCH GUARD atop `_run_one_simulation_step`: flag AND gpu AND IZHIKEVICH AND fast_spike_reset AND read_only_fast_step-preconditions AND STP/homeostasis/structural/neuromod OFF AND no per-step csr-rebuild gate (`effective_connections IS cp_connections`) AND no recording/data_bus/engram/synapse_store this step → else the unchanged Python path (byte-identical-when-off AND when-guards-fail).
3. ONE-TIME SETUP (cached instance attrs like `_rf_megastep_kernel`): compile `_STEP_MEGASTEP_SRC`; build static `_step_WT=cp_connections.T.tocsr()` + per-neuron exc/inh flag; invalidate on `cp_connections` id/nnz change.
4. PER-STEP: draw `noise=cp.random.randn(n).astype(float32)` (byte-identical order to @6845); one kernel launch replacing ~9; device-only `cp.sum(firing)`; skip both `bool()` syncs.
5. Structural/guard flip → auto-fallback to the Python path.

## Expected: ~2.5-4× steps/sec at small/medium N (launch-bound regime), tapering to ~1× past ~50-100K where CSR compute dominates. Same class as the RF megakernel (prototype 11×, deployed A5 ~4.3×), more modest (kept OU draw = a 2nd launch, heavier fused body). **Applies to sim INFERENCE only — NOT training runs (they fail the guard + are less launch-bound).**

## Validation (mirrors the RF megakernel discipline)
(A) BYTE-IDENTICAL-WHEN-OFF: determinism + read_only_fast_step suites, hash v/u/firing after N steps == baseline.
(B) ON-PATH EQUIVALENCE: two bridges same `cfg.seed` (the seed gotcha — set `cfg.seed`), flag off vs on, 1000 steps:
    the fired boolean raster IDENTICAL every step (spikes are load-bearing) + max|Δv|,|Δu| within a documented FMA
    residual; OU stream verified bit-identical. New `tests/test_step_megakernel.py`.
(C) BENCHMARK A/B: `benchmark.py` inference config, flag off vs on across N∈{1K,2K,5K,10K}; steps/sec + nsys launch
    counts (~9→2). Reuse the `research/findings/raw/gpubench/` A/B harness.

## Risks
FP/FMA op-order (spike raster must match exactly — a neuron on threshold can flip); config-path coverage (only the
common no-gate E/I-split Izhikevich path in v1, everything else guard-OFF); transpose memory (WT doubles CSR storage
while on; stale-WT invalidation needed); narrow applicability (inference only); maintenance (duplicates 4 kernels' math).
Full synthesis + 4 research angles: workflow `wf_2dda99f4-00d` (task `wy774rl59`).
