# Perf-optimization scoping: AdEx, HH, and learning spiking-sim paths (2026-07-23)

Produced by a 4-agent read-only workflow (AdEx / HH / learning analysts + completeness critic) after the Izhikevich
step-megakernel v2 shipped byte-identical + default-on. Question: what perf work is possible for the paths the
megakernel currently EXCLUDES, and is it worth it? Every claim below is grounded in a file:line the agents read;
measurement-vs-reasoning is flagged where it matters.

## Bottom line (ranked by value)
1. **Learning is the biggest opportunity** — it is MORE launch/sync-bound than inference (4-5 device→host syncs/step vs 2), and the cheap win (**branchless/compaction-free STDP+Hebbian, M effort**) removes the 2 nonzero-compaction syncs that a microbench shows make the step **sync-bound (~5ms FLAT from 100K→1M nnz; the all-nnz alternative is 15-45× faster)**. A full learning megakernel (L) is plausibly ≥ the inference 4.3×. **Hard blocker: structural plasticity mutates `cp_connections` shape (bridge.py:8451) → must be guard-forbidden (it already is OFF in the g11 regime).** read_only_fast_step under learning is **impossible (correctness)** — the STDP/structural blocks draw RNG, so forcing the spike-present flags diverges the stream.
2. **AdEx is a clean template extension, but match the tool to density.** AdEx is a structural near-twin of Izhikevich (2-var IF + discrete reset, dt=0.5ms, ~same op count; only new math = one `expf` + one division), so the megakernel template extends cleanly. BUT its ONE real workload (`_genseq_loopstep3_multilayer`, the generative-sequence forward) is **DENSE** (~2048 edges/neuron) → matvec-COMPUTE-bound, not launch-bound. So the win there is the **element-wise @cp.fuse megastep that KEEPS cuSPARSE (M)** or a **dense cuBLAS GEMV** (the critic's catch — the repo already has a dense-weight path, RF `cp_rf_w_dense`), NOT the full matvec-folding RawKernel (which only pays on SPARSE small/medium nets). Free first step: the `_genseq` runner doesn't even set `read_only_fast_step`/`fast_spike_reset` today — flipping them collects the model-agnostic 2-sync removal (though negligible on the dense matvec-bound step).
3. **HH is LOW priority.** It carries none of the production load (conversational/nav/EMERGE are all Izhikevich + resonate-and-fire; HH is biophysical validation, and most HH runs turn learning ON). It already gets read_only_fast_step's 2-sync removal for free. The interesting HH lever isn't a megakernel — it's the critic's: **the gates use exponential-Euler but V uses plain forward Euler (kernels.py:116), forcing dt=0.05ms (10-20× more steps); a stable larger-dt V integrator would cut the STEP COUNT** — a compute reduction, not a launch reduction — but that's a numerical-stability project, not a quick win.

## Per-path detail

### AdEx — near-twin of Izhikevich
- **Bound:** density-dependent. Small/medium SPARSE = launch-bound (would benefit ~4.3× like Izhikevich); the dense `_genseq` = matvec-compute-bound for the SpMV + launch-bound for the ~10 element-wise ops.
- **read_only_fast_step:** already applies (model-agnostic guard) — strips 2 of ~4 syncs; the 1-2 AdEx-RESET syncs (`cp.where(fired)[0]` @7463, boolean-mask refractory decrement @7472) remain (AdEx has no fast-reset path).
- **Optimizations (effort/feasibility):** [S] flip read_only_fast_step on AdEx runners (free, but negligible on the dense workload) · [S] add an AdEx `fast_spike_reset` analog (~10 lines mirroring the Izhikevich cp.where reset @7315-7339 → zero syncs; prerequisite for an AdEx megakernel) · [M] AdEx element-wise @cp.fuse megastep keeping cuSPARSE (safest dense win) · [M] full AdEx RawKernel megakernel folding the matvec (the 4.3×-class win, SPARSE only).

### HH — biophysical, launch-bound per step but dt=0.05ms
- **Bound:** launch-bound per step at 200-50K neurons, but 10-20× more steps than Izhikevich/AdEx.
- **Optimizations:** [S] already gets read_only_fast_step's 2-sync removal · [M] HH @cp.fuse megastep keeping cuSPARSE (~8-10 launches→1) · [M] fuse the 4 extended-current kernels (small, preset-dependent — only bursting presets) · [L] HH matvec-folding RawKernel (4×-class, ~20× amplified per bio-ms, but NARROW audience). Critic's addition: **larger-dt stable V integrator to cut step count** (the real HH lever).

### Learning — the most sync-bound path
- **Bound:** launch/SYNC-bound, MORE than inference (4-5 syncs; the 2 nonzero-compaction syncs are the killer). Empirically: compacting-STDP ~5ms FLAT 100K→1M nnz (sync-bound); branchless all-nnz 15-45× faster.
- **Optimizations:** [M] **branchless STDP+Hebbian** (the cheap big win; must be byte-identical — critic flags the internal `cp.clip(w_new,w_min,w_max)` @kernels.py:468 hazard for a naive all-nnz apply) · [L] full learning megakernel (STDP+eligibility+reward; structural plasticity guard-forbidden) · [S] device-accumulate the plasticity-event stats (removes 1 sync on reward-apply steps; prerequisite for graph capture) · [S/impossible] read_only_fast_step under learning (RNG divergence) · [L/hard-blocker] structural-plasticity megakernel (CSR shape mutation can't live in a fixed-launch kernel).

## Cross-cutting (the critic's highest-value additions)
- **Shared megastep TEMPLATE / front-end fusion.** The three proposed per-model fused kernels (izh done, adex, hh) each RE-DERIVE the identical synaptic front-end (conductance decay + I_syn + matvec-increment). Factor it once → the AdEx/HH kernels become thin dynamics-swaps on a shared front-end. Also: any new RawKernel MUST mirror v2's class-level compiled-kernel caching (`SimulationBridge._step_megastep_kernel` @6368).
- **Counter-based (Philox) in-kernel RNG.** Folding the OU `cp.random.randn(n)` (and HH's 2 conductance-noise draws @7371-7372) INTO the kernel would remove the last separate launch AND is the missing piece for any CUDA-graph route.
- **CUDA-graph capture — nuanced, NOT a free headline.** The project already evaluated + rejected *literal* graph capture for the inference step (design doc `2026-07-23-general-step-megakernel-design.md`: cuSPARSE SpMV is uncapturable + graph capture freezes the OU RNG) and chose the megakernel instead — the right call for the Izhikevich inference path (which is now ONE launch, so a graph adds ~nothing). The critic's angle applies to the MULTI-launch paths NOT yet megakernel'd (learning's separate STDP/eligibility/reward kernels; HH's extended currents): capturing that fixed-topology sequence as one CPU launch is a real win IF the RNG is made capturable (Philox) and any cuSPARSE is folded. So: evaluate CUDA-graph for LEARNING/HH, not inference — but the megakernel route remains the validated approach.
- **Dense GEMV for the dense AdEx `_genseq` matvec** (reuse RF's `cp_rf_w_dense` dense-weight precedent) — the one concrete AdEx workload is dense, so cuBLAS GEMV beats both cuSPARSE and an in-kernel CSR SpMV there.

## Honest caveats the critic caught (don't repeat these in a build)
- **"expf is an nvrtc-header-free device builtin" is UNVERIFIED** and the cited precedent is wrong — NO in-repo RawKernel uses any transcendental. So an AdEx/HH megakernel would be the FIRST transcendental-bearing RawKernel → first-of-kind byte-faithful `exp` verification is added scope (bumps AdEx-full-megakernel above "M").
- The learning branchless-STDP "byte-identical" claim omits the internal-clip hazard; the "reuse `_ensure_step_v2_transpose`" idea under-states that v2 reads a separate float64 transposed-CSR copy of the weights.
- The AdEx "~4 syncs" and boolean-mask-decrement-as-sync counts are reasoned from CuPy semantics, not profiled — confirm with nsys before building.

## Recommendation (what to actually do, in order)
1. **If/when a learning run is the bottleneck: branchless STDP+Hebbian (M)** — cheapest, biggest, byte-identical-able, guard structural-plasticity OFF. This is the single highest-value item.
2. **Factor the shared synaptic front-end template (M)** before writing any AdEx/HH kernel — pays for itself across all three.
3. **AdEx: element-wise @cp.fuse megastep keeping cuSPARSE (M)** for the dense `_genseq` path, or **dense GEMV** — only if AdEx generative-sequence becomes a hot path.
4. **HH + full RawKernel megakernels: defer** — narrow audience, first-transcendental-RawKernel risk, HH better served by a larger-dt integrator.
5. **Everywhere: measure launch-boundedness first** (the clock-offset A/B) — most of these only pay in the small/medium launch-bound regime, and the one real AdEx workload is dense (matvec-bound), where the megakernel does NOT help.
