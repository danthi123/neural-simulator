# Training-speedup playbook

**Status (updated 2026-05-02):**
- **Tier 1 = PARTIAL.** 1.1 (stim_steps 200→100) and 1.5 (per_type_stp=False) shipped
  and validated. **1.2 (reset_steps 100→50) REVERTED** — caused 300-ep R3+R6 regression
  to 20%/20% from 32.5%/30% baseline (NMDA τ=100ms requires full reset for trial isolation;
  see `2026-05-01-text-io-300ep-tier1-REGRESSION.md`).
- **Tier 2.6 (heterogeneity disable) = NOT SAFE.** Smoke test 2026-05-02: disabling
  `enable_parameter_heterogeneity` (alongside `enable_ou_process`) collapsed correct-moves
  from 30%+ baseline to 2.4%. All-zero language_output spikes; agent always emits "north".
  Pure Izh parameters → pathological synchrony; per-neuron variation is load-bearing for
  breaking lockstep.
- **Tier 1.3 aggressive form (full OU disable) = NOT SAFE.** Same smoke test: disabling
  `enable_ou_process` removed spontaneous activity STDP needs for pre-synaptic spike
  events outside the explicit-input window. Conservative form (gate OU on
  `current_reward_signal != 0 OR cp_external_input_current.any()`) still viable but
  requires a code change in `sim/bridge.py`, not just a config flag.
- Tier 2 (other items) + Tier 3 = documented for future iteration.

**Baseline:** ~5.06 ms wall-clock per sub-step on ~5000 neurons + ~200K
synapses (single RTX 3090, no batching). Empirically: 100-episode
embodied training takes ~75 min; 300-episode takes ~3 hours.

**Per-step composition (rough estimates from `_run_one_simulation_step` profile):**
- ~40% synaptic conductance + STP decay (sparse matrix ops)
- ~20% neuron dynamics (Izhikevich + NMDA fused kernels)
- ~15% STDP + eligibility trace updates
- ~10% Python orchestration (kernel launch latency)
- ~15% misc (homeostasis, structural plasticity arrays, OU noise, recording hooks)

---

## Tier 1 — Easy wins (next to implement)

Target: **1.8-2× speedup**, all low-risk.

### 1.1 Halve `stim_steps_per_pair` (200 → 100)
**File:** `research/runners/text_train_embodied.py`, `text_train.py`, `text_train_contrastive.py`, `text_eval.py`
**Rationale:** 200 sub-steps = 100ms simulated. STDP requires only the spike-pairing window (~20ms). Eligibility trace τ=500ms persists across multiple steps, so 100ms windows still accumulate signal correctly.
**Risk:** Low — but verify accuracy unchanged on smoke test.
**Speedup:** ~1.4× on stim window (62% of total step), ~1.25× overall.

### 1.2 Halve `reset_steps` (100 → 50) — REVERTED 2026-05-02
**File:** same
**Rationale:** NMDA τ=100ms; 50ms is 0.5τ, leaving ~37% of activity at trial start. With per-region NMDA mask (only PFC+cortex_X+motor_X), the residual is minimal.
**Risk:** Medium — might re-introduce some cortex_N bleedover. Test for word-discrimination preservation on smoke.
**Speedup:** ~1.2× on reset window.
**RESULT:** **NEGATIVE.** 300-ep R3+R6 + Tier 1 (with 1.2 active) regressed to 20% / 20%
W→A / I→W vs 32.5% / 30% baseline. Training-phase correct-moves still climbed to
38.5% (visuomotor pathway learned), but language pathway readout came out garbled —
the 50ms reset left ~40% NMDA bleedover, which over 9000 env steps compounded into
systematically scrambled language→cortex weights. Smoke (5 ep × 10 steps) didn't
catch it because contamination only matters at scale and across many trials.
**Reverted.** See `research/findings/2026-05-01-text-io-300ep-tier1-REGRESSION.md`.

### 1.3 Skip OU noise during silent reset windows — DEFERRED (aggressive form unsafe)
**File:** `sim/bridge.py` `_run_one_simulation_step`
**Rationale:** OU noise is computed every step but during reset (zero input, zero reward) we don't care — the goal is just to let dynamics decay. Could gate OU computation on `current_reward_signal != 0 OR cp_external_input_current.any()`.
**Risk:** Low (conservative gated form). HIGH (aggressive form: full disable).
**Speedup:** ~1.05-1.1×.
**RESULT (aggressive form, 2026-05-02):** **NEGATIVE.** Disabling
`enable_ou_process = False` entirely (paired with heterogeneity disable in same
smoke) collapsed correct-moves to 2.4% (from 30%+). OU provides spontaneous
activity STDP needs for pre-synaptic spike events outside the explicit-input
window. Without it, the network has no random fluctuations to seed cortex_X
firing during stim windows where retina input is sparse — STDP eligibility
traces don't form on the right pre-post pairings.
**Conservative gated form remains viable** but requires `sim/bridge.py`
code change (not just config flag). DEFERRED until Tier 2 batch.

### 1.4 Set `OPENGL_AVAILABLE=False` for headless training
**File:** `sim/bridge.py` (already gated by this flag)
**Rationale:** Visualization arrays (`cp_synapse_pulse_timers`, `cp_synapse_pulse_progress`) allocate GPU memory but text training doesn't use them.
**Risk:** Zero — feature-flagged.
**Speedup:** ~1.05×.

### 1.5 `enable_per_type_stp = False` for text training
**File:** text training config
**Rationale:** We have a single STP type (E→E) anyway. Per-type adds a `cp_synapse_conn_type` array lookup per step.
**Risk:** Low.
**Speedup:** ~1.05-1.1×.

### 1.6 Larger `dt_ms` (0.5 → 1.0) for Izhikevich-only training
**File:** text training config
**Rationale:** Izhikevich-2007 model is numerically stable at dt=1ms; the 0.5ms dt is needed only for HH-style multi-current biophysics. Our text training uses Izh.
**Risk:** Medium — verify spike timing isn't degraded.
**Speedup:** **~2× on dynamics** (since fewer dt steps for the same simulated time). Caveat: if we keep simulated time constant (100ms stim window = 200 sub-steps at dt=0.5 OR 100 sub-steps at dt=1.0), this is the SAME as halving stim_steps — they multiply.

### Combined Tier 1 estimate
- Item 1.1 alone: 1.25×
- 1.1 + 1.2: 1.4×
- 1.1 + 1.2 + 1.3-1.5: 1.6×
- All six (with 1.6 dt change): up to 2× (since 1.6 partly subsumes 1.1)

**Test plan:**
1. Smoke run (50 episodes) with all Tier 1 changes; compare W→A accuracy to 100-ep baseline (30%).
2. If accuracy holds (within 2pp), commit.
3. Launch 300-ep with speedups; expected wall time ~90-100 min instead of 180.

---

## Tier 2 — Medium difficulty (multi-day each)

### 2.1 CUDA Graphs for inner stim window
Replay a recorded sequence of CUDA kernel launches. Eliminates Python orchestration overhead and kernel-launch latency for the repetitive inner loop.

**File:** `sim/bridge.py` — wrap `_run_one_simulation_step()` body in a CUDA Graph capture/replay.

**Constraints:**
- Tensor shapes must be invariant across replays (no structural plasticity, no synapse adds)
- Reward signal can change (set externally before replay starts)
- We already disable structural plasticity for text training, so this is feasible

**Speedup estimate:** 2-3× on the stim loop alone (the dominant ~62% of step). Overall ~1.5-1.8×.

**Effort:** 2-3 days. Requires CuPy 13+ CUDA Graphs API.

**Risk:** Medium. Failure modes: silent shape mismatch on replay; CUDA Graph capture-time bugs.

### 2.2 Mixed precision (FP16 weights / FP32 accumulators)
Use FP16 for `cp_connections.data` (weights) but FP32 for `cp_membrane_potential_v`, eligibility traces, and STDP accumulators. RTX 3090 has 35.6 TFLOPS FP16 vs 17.8 TFLOPS FP32.

**File:** `sim/bridge.py`, `sim/kernels.py`.

**Speedup estimate:** 1.5-2× on weight-multiply-heavy ops.

**Risk:** Medium-high. STDP precision is critical; FP16 weights may saturate at boundaries. Need careful test on STDP-driven learning tasks (Bi & Poo timing curve) to verify behavior unchanged.

**Effort:** 3-5 days. Requires extensive correctness testing.

### 2.3 Batched replicas (multi-bridge in one process)
Run 4-8 episodes' worth of bridges in parallel, sharing GPU. Reuses `sim/replicas.py` infrastructure that already exists for replicated wiring.

**File:** new training runner that uses `replicas.py` to run e.g. 6 bridges in parallel for 6-seed validation.

**Speedup estimate:** 4-6× wall-clock for n-seed validation runs (where n bridges run simultaneously).

**Risk:** High. The replicated runner had a known plasticity-rate bug (commit log mentions this), so use with caution.

**Effort:** 1 week.

### 2.4 Numba/Cython for orchestration hot path
Compile the Python wrapping logic around `_run_one_simulation_step` (the env loop, action selection argmax, reward computation). Reduces Python overhead per env step.

**Speedup estimate:** 1.2-1.5×.

**Risk:** Low.

**Effort:** 2 days.

### 2.5 Skip recurrent updates for inactive regions
During stim windows where retina drives V1 (forward sweep), V1's recurrent self-excitation isn't critical for the first few ms. Could batch the recurrent step every N=2 sub-steps for inactive regions.

**Speedup estimate:** 1.2-1.5×.

**Risk:** Medium — affects firing dynamics; needs validation.

**Effort:** 3 days.

### 2.6 Disable per-neuron heterogeneity for text training — NOT SAFE (2026-05-02)
The CV~0.3-0.4 per-neuron parameter heterogeneity adds 4-array memory accesses per neuron per step. Text training may not need this realism.

**Speedup estimate:** 1.05-1.1× (small but free).

**Risk:** ~~Low (just a config flag)~~ **HIGH — empirically falsified.**

**Effort:** ~~trivial~~ DO NOT TRY.

**RESULT (2026-05-02):** **NEGATIVE.** Smoke test with
`enable_parameter_heterogeneity = False` (paired with OU disable in same smoke):
correct-moves collapsed from 30%+ baseline to 2.4%. All-zero language_output
spike counts; agent emits "north" every trial regardless of input.
Diagnosis: pure Izh parameters across the network → pathological synchrony.
Real cortex relies on per-neuron CV ~0.3-0.4 to break lockstep firing patterns;
without it, the BG cascade collapses into one-or-zero global oscillator modes
instead of forming distinct per-action winning pools. Heterogeneity is
load-bearing infrastructure, not optional realism.
**Reverted.** Cautionary comment added in `research/runners/text_train_embodied.py`.

### Combined Tier 2 estimate
On top of Tier 1: another 2-3× → total 4-6× from baseline.

---

## Tier 3 — Architectural / research (week+)

### 3.1 Single fused mega-kernel
Replace the dozens of small `@cp.fuse()` calls with one hand-tuned kernel that runs the entire `_run_one_simulation_step` per call. Eliminates ALL Python overhead and kernel launch latency.

**Tooling:** Triton (PyTorch backend) or raw CUDA-C.

**Speedup estimate:** 3-5× on the simulation loop.

**Effort:** 1-2 weeks. Requires deep CUDA expertise + maintenance burden.

### 3.2 Multi-GPU sharding
Split the simulation across 2 GPUs (e.g., RTX 3090 + RTX 3090). Bridge state synchronizes via NCCL each step.

**Speedup estimate:** 1.5-1.8× (NCCL sync overhead limits ideal 2×).

**Effort:** weeks.

**Cost:** Hardware (already have one 3090).

### 3.3 Hierarchical timestepping
Run V1 at dt=0.5ms (fast retinal dynamics matter), V2 at dt=2.0ms (slower feature integration), IT at dt=5.0ms (object-level integration), motor at dt=0.5ms (precise action timing).

**Speedup estimate:** 2-3× overall.

**Risk:** Medium-high. Need probes to verify accuracy doesn't degrade for tasks that depend on cross-region timing precision.

**Effort:** 1 week + extensive validation.

### 3.4 Rate-coded surrogates for non-critical regions
Replace per-neuron Izhikevich integration with closed-form firing-rate ODEs for regions where individual spike timing doesn't matter (e.g., dopamine baseline, GPe pacemaker).

**Speedup estimate:** 5-10× on those regions; 1.5-2× overall (since spike-timing-critical regions still use Izh).

**Risk:** Medium. Well-trodden path in computational neuroscience (Wilson-Cowan, Tsodyks meanfield).

**Effort:** 2 weeks.

### 3.5 Custom Triton kernel for sparse synaptic conductance
The largest single kernel is the sparse `cp_connections @ cp_firing_states` matrix-vector multiply (CSR sparse matvec). CuPy's stock implementation isn't tuned for our specific sparsity pattern (~5% density, structured by region).

**Speedup estimate:** 2-3× on that op (~40% of step); 1.5× overall.

**Effort:** 2 weeks.

### Combined Tier 3 estimate (with Tier 1 + 2)
Theoretical 10-20× total speedup. 300-ep training in 15 min instead of 180 min.

---

## Decision matrix

| Tier | Effort | Speedup | When to do |
|---|---|---|---|
| **Tier 1** | 1 day | 1.8-2× | Now (next implementation) |
| **Tier 2.1 (CUDA Graphs)** | 2-3 days | 1.5-1.8× more | When Tier 1 speedup isn't enough for next experiment |
| **Tier 2.3 (batched replicas)** | 1 week | 4-6× for n-seed | When 6-seed validation becomes the bottleneck |
| **Tier 3.4 (rate surrogates)** | 2 weeks | 1.5-2× more | When approaching artificial life with 50K+ neurons |
| **Tier 3.1 (mega-kernel)** | 1-2 weeks | 3-5× more | When hardware is the limit, not architecture |

## Files / commits

- `2026-05-01` — this doc (`docs/plans/2026-05-01-training-speedups.md`)
- TBD — Tier 1 implementation commit
- TBD — Tier 1 validation commit (smoke + full run accuracy comparison)
