# Performance audit — making g11_bg navigation de-risk batches iterate faster

**Date:** 2026-06-08
**Scope:** READ-ONLY analysis of the `g11_bg_runner` navigation hot paths + the
multi-seed launcher workflow, with one CPU micro-benchmark. No long GPU jobs run
(GPU busy with owner's runs). Goal: faster ITERATION on the multi-seed de-risk
batches (a single grid-8 1800-step run ≈ 11 min; a 3-seed × 2-condition batch ≈
60–70 min sequential; a 3-seed × 6000-step generalize batch ≈ ~110 min).

**Hard constraint honored:** every recommendation below either preserves biology +
numerics EXACTLY, or is explicitly flagged as an opt-in approximation the owner
must accept. No silent dynamics changes.

---

## 1. Executive summary — top 3 optimizations (ranked by speedup × safety / effort)

| Rank | Optimization | Est. wall-clock impact on a 3-seed×2-cond batch (~66 min) | Safety | Edit site | Effort |
|---|---|---|---|---|---|
| **1** | **Parallel multi-seed (run seeds concurrently, GPU-shared, parallelism=2–3)** | **66 min → ~28–40 min (1.7–2.4×)** | **EXACT** (separate processes, identical math) | **Launcher only** (no `sim/` edit) | **~30 min** (rewrite the `foreach` loops) |
| 2 | **Apply `fast_spike_reset` to nav runs** (add `--fast-spike-reset` CLI → `cfg.fast_spike_reset=True`) | each run ~1.29× → batch −15 to −20% on top of #1 | **EXACT for Izhikevich** (nav's model); test-pinned | tiny protected `sim/` edit (1 line in runner kwargs) + CLI add — OR pure runner edit | ~20 min |
| 3 | **Eliminate the 70 per-substep `.get()` syncs in the readout window** (accumulate motor counts on-device, `.get()` ONCE at window end) | each run ~1.1–1.3× (readout is 70% of substeps); batch −10 to −25% | **EXACT** (same integer counts) | runner only (no `sim/` edit) | ~1–2 hr (careful, hot loop) |

**Single highest-leverage item: #1, parallel multi-seed.** It is the only change
that is simultaneously (a) the biggest win, (b) numerically EXACT — two seeds in
two OS processes compute byte-identically to running them one after another, (c)
zero `sim/` risk (launcher-only), and (d) already supported by shipped infra
(`research/experiment_runner.py` has a `parallelism: N` batch executor;
`bench_parallel_gpu.py` measured ~1.7× effective for 3 concurrent procs). The de-risk
launchers (`research/findings/raw/_run_*.ps1`) currently throw this away by running
seeds in a strict sequential `foreach`.

**One caveat, designed-around (from MEMORY):** over-subscription causes a ~2×
SLOWDOWN, not a speedup — the 2026-05-05 wave-2 doc measured `parallelism=6`
nearly HALVING per-process throughput on this ~20K-neuron net. The sweet spot for
grid-8 is **parallelism=2 (safe) to 3 (measured ceiling)**, never 4+. Recommendation
below pins it at 2–3 with a guard.

---

## 2. Bottleneck analysis — where the time actually goes

### 2a. The sim step itself is ALREADY well-optimized (don't chase it)

`sim/bridge.py::_run_one_simulation_step` (`bridge.py:5126`) is GPU-resident and
clean. Evidence (grepped the full step body 5126→6285):

- **Exactly ONE host↔device sync per step**: `_prev_any = bool(self.cp_prev_firing_states.any())`
  at `bridge.py:5142`, with an explicit comment that it's cached once per step "to
  avoid repeated GPU-CPU sync stalls." No `.get()`, `.item()`, `cp.asnumpy`, or extra
  `.any()`/`bool()` calls anywhere else in the step.
- **The synaptic matvec is already batched**: the exc/inh split is done as a single
  CSR transpose-matvec over a stacked `(n,2)` firing matrix
  (`effective_connections_matrix.T @ fired_2col`, `bridge.py:5304`), reusing one CSR
  index traversal instead of two — a deliberate optimization.
- STP per-type params, the inhibitory mask, and conductance decay factors are all
  **cached** (`_cached_stp_per_type`, `_cached_inhibitory_mask`, `_cached_decay_e/i`)
  so they're computed once, not per step.
- A built-in per-section profiler already exists (`gpu_config.enable_step_profiler`,
  `bridge.py:5134` — `t_init/t_stp/t_syn/t_dyn/t_plast/t_homeo`).

**Conclusion:** the per-step GPU compute is not where the easy wins are. At ~20K
neurons the kernels are *launch-bound* (many tiny elementwise kernels over small
arrays), which is exactly why the existing dt=1.0 + fast_spike_reset (fewer/larger
kernels) help and why **forcing a device sync every substep is so damaging** (it
prevents the launch queue from staying full).

### 2b. The REAL per-action cost: 70 forced device syncs in the readout window

In the main episode loop (`g11_bg_runner.py:4695–4732`), the per-action readout
window calls `bridge.cp_firing_states.get()` **on every readout substep**
(`g11_bg_runner.py:4722`), then does 4 host-side fancy-index sums per action.

Run-length facts (from the runner): `STIMULUS_MS=100`, `dt=1.0` →
`n_stim_steps=100` substeps/action; `READOUT_START_MS=30`, `READOUT_END_MS=100` →
**70 readout substeps/action, each with a `.get()`**.

```
D2H syncs per action (readout window):                       70
+ reward-hold spiking-SNc reads (int(...[idx].sum()), 10):  +10   (only with --spiking-snc)
Total readout D2H syncs per 1800-step run:              126,000
Theoretical minimum (accumulate on-device, .get() once/window):  1,800
                                                  reduction factor:  70×
```

Each `.get()` is a **full device sync** — it blocks the host until the GPU drains,
so the 70-substep readout window runs fully serialized (no host/device overlap).
A CPU micro-benchmark I ran (numpy proxy for the *host-side* bool-cast + 4 sums over
a 20K-element array, 70 reps) measured only **0.58 ms** — i.e. the host *arithmetic*
is negligible. **The cost is the 70 sync stalls themselves**, not the host math. This
is the textbook CuPy bottleneck (small arrays + frequent D2H + forced sync), and it's
the one place in the per-action path it appears.

> Honest note: I could not measure the GPU-side sync-stall cost without the GPU
> (it's busy). The 70× *transfer-count* reduction is exact; the *wall-clock* gain
> from collapsing it (est. 1.1–1.3× per run) is an estimate to be confirmed by the
> profiling plan in §4. It is bounded above by "readout window = 70% of all substeps."

### 2c. Host-side numpy per step is small but non-zero

Once per step (awake, not in sleep), `render_gridworld_to_image` +
`image_to_retina_drive` (`g11_bg_runner.py:4539–4548`) and, when enabled,
`sc_salience_offset_from_image` / `sc_orienting_cardinal_from_image`
(`:4575`, `:4589`) run in host numpy on a 32×32 image. These are once-per-step (not
per-substep), so ~1800 calls/run on tiny images — real but second-order vs the
126,000 readout syncs. Not worth optimizing before #1–#3.

### 2d. Sequential seeds — the biggest structural waste

The de-risk launchers iterate seeds in a strict sequential `foreach`:

```powershell
# _run_bioreward_da_derisk.ps1  (and every other _run_*.ps1 I checked)
foreach ($seed in @(42, 43, 44)) {
  python -m research.runners.g11_bg_runner @common --seed $seed ...   # NEURAL
  python -m research.runners.g11_bg_runner @common --seed $seed ...   # CHEAT
}
```

→ 6 processes run one-at-a-time. At ~20K neurons a single grid-8 process uses only a
*fraction* of the RTX 3090's SMs and VRAM (see §2e), so the GPU sits mostly idle
during each run. This is the single largest reclaimable inefficiency, and it needs
**no code change to the simulator** — only the launcher.

### 2e. VRAM headroom estimate (supports parallelism=2–3)

Network is ~20K neurons; the V1 simple-cell bank dominates count
(`8 orient × 2 freq × 32 × 32 ≈ 16,384`), everything else (per-action BG pools at
10–50 neurons × 4, retina 2048, V2 256, IT 64, hippocampus ~320, granule 250, PFC 60)
is small. Per-neuron state is ~2.4 MB total (≈30 float32 arrays). VRAM is dominated by
the CSR synapse matrix + eligibility/STP per-synapse arrays, which at this scale and
density is on the order of a few hundred MB to low-GB, **not** tens of GB. On a 24 GB
3090 this comfortably fits **3–6 concurrent processes by memory** — so the limit is
*compute throughput / kernel-launch contention*, not VRAM. The measured throughput
ceiling (below) is what caps us at 2–3, not memory.

**Default mempool gotcha for parallelism:** the runner does NOT set
`GPUConfig.memory_pool_limit_fraction`, so each process defaults to grabbing up to
**0.8 of total VRAM** for its pool. Two such processes can collide. When parallelizing,
set `memory_pool_limit_fraction ≈ 0.30` (env or config) per process so N share cleanly.

---

## 3. Ranked optimizations (full detail)

### #1 — Parallel multi-seed launcher (EXACT, launcher-only) ★ highest leverage
- **What:** launch the seeds (and independent conditions) of a de-risk batch as
  concurrent processes sharing the GPU, capped at **parallelism=2 (safe) / 3
  (measured ceiling)**, instead of a sequential `foreach`. Two ready vehicles:
  - **Reuse `research/experiment_runner.py`** — it already runs `conditions × seeds`
    in `parallel-N` batches (`experiment_runner.py:276`, `parallelism` field). Wrap
    the de-risk config as a small YAML/JSON.
  - **Or** convert the `_run_*.ps1` `foreach` into a bounded `Start-Process` /
    `Start-Job` pool of size 2–3 with a `Wait` barrier.
- **Estimated speedup:** **1.7×** effective at 3 procs (measured in
  `2026-05-04-perf-speedup-stack.md`: "~1.7× effective, 3 procs at ~70% efficiency").
  Conservatively **1.7–2.0×** for a 6-run batch at parallelism=2–3 → **66 min →
  ~33–40 min**. A larger 6-seed batch benefits even more (cleaner divisibility).
- **Correctness-safety:** **EXACT.** Separate OS processes → separate CUDA contexts →
  identical RNG seeding and identical math. Parallelism changes *when* a run executes,
  never *what* it computes. (Verify by diffing one seed's output JSON run-solo vs
  run-in-pool — must be byte-identical with `--deterministic`.)
- **Edit site:** **launcher only. No `sim/` edit.**
- **Effort:** ~30 min.
- **Falsifiable validation:** "seed-42 `final_quarter` sums are byte-identical
  solo vs in a parallelism-2 pool, and the 6-run batch finishes in ≤ 0.65× the
  sequential wall clock." Also assert no process exceeds its mempool fraction (no OOM).
- **Risk guard (from MEMORY — the ~2× slowdown trap):** HARD-cap the pool at 3;
  set `memory_pool_limit_fraction≈0.30` per process; ensure the barrier `Wait`s so a
  hung seed can't orphan; never set parallelism≥4 on grid-8 (the wave-2 doc measured
  parallelism=6 → baseline throughput HALVED, i.e. a net slowdown).

### #2 — Apply the existing `fast_spike_reset` flag to nav runs (EXACT for Izhikevich)
- **What:** nav runs leave `cfg.fast_spike_reset = False` (the default;
  `config.py:186`) — grep confirms the g11 runner never sets it and there is **no
  `--fast-spike-reset` CLI flag**. Add the flag (or set it unconditionally in the
  nav builder) so the spike-reset uses the fused `cp.where` masked update
  (`bridge.py:5410`) instead of fancy-index assignment.
- **Estimated speedup:** **1.29×** per run (measured on the minimal arch,
  `2026-05-04-perf-speedup-stack.md`). The gain is largest on small launch-bound
  networks — i.e. exactly this ~20K-neuron nav net.
- **Correctness-safety:** **EXACT for the Izhikevich model**, which is what nav uses.
  The fast path is documented and **test-pinned numerically equivalent for Izhikevich**
  (`tests/test_fast_spike_reset.py`; matches the legacy off-by-one refractory exactly,
  `bridge.py:5419–5432`). It is *not* claimed equivalent for HH/AdEx — nav doesn't use
  those, so this is safe here. **Do not enable globally** for HH/AdEx runners without
  per-model verification.
- **Edit site:** add `--fast-spike-reset` arg in the runner and route to
  `cfg.fast_spike_reset` (pure runner edit, no protected `sim/` file touched), OR set
  `cfg.fast_spike_reset = True` in the nav config block. **No change to `sim/` kernels**
  (the fast path already exists in `bridge.py`).
- **Effort:** ~20 min.
- **Falsifiable validation:** "1-seed nav `final_quarter` sum at seed 42 is identical
  (or within FP-noise; expect IDENTICAL for Izhikevich) with vs without
  `--fast-spike-reset`, and per-step wall clock drops ~20–25%."

### #3 — Collapse the 70 per-substep `.get()` syncs into 1 per window (EXACT)
- **What:** in the readout window (`g11_bg_runner.py:4695–4732`), stop calling
  `cp_firing_states.get()` every substep. Instead keep 4 small **on-device**
  accumulators (one per action) and add `cp_firing_states[motor_idx_cp[a]].sum()`
  into them each substep (device-side, no sync), then call **one** `.get()` (or
  `cp.stack(...).get()` over the 4 totals) AFTER the loop. Same for the optional
  `thal/sel/commit` tallies. This turns 70 sync stalls/action into 1.
- **Estimated speedup:** **1.1–1.3× per run** (the readout window is 70 of 100
  substeps/action; removing its forced serialization lets the GPU pipeline). Estimate;
  confirm via §4.
- **Correctness-safety:** **EXACT.** Integer spike counts are identical — only *when*
  they're transferred changes. The host-side argmax/decision logic downstream is
  unchanged (it consumes the same per-action totals).
- **Caveat — preserve the sample-trace path:** the `_capture` sample-trial branch
  (`g11_bg_runner.py:4715–4719`, only ~3 trials/run) legitimately needs per-substep
  firing for the sel/commit traces. Keep the per-substep `.get()` ONLY on those few
  sample trials; collapse it on the other ~1797. Easy to gate on the existing
  `_capture` flag.
- **Edit site:** runner only. **No `sim/` edit.**
- **Effort:** ~1–2 hr (it's the hot loop; needs care + a count-equality test).
- **Falsifiable validation:** "per-action `motor_counts` dict equals the old code's
  for a fixed seed (assert exact integer equality over a 100-step smoke), and per-run
  wall clock drops measurably."

### #4 — Drop `--deterministic` for SMOKE de-risk only (opt-in approximation)
- **What:** `--deterministic` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  (`g11_bg_runner.py:62`) for reproducible cuBLAS algos; it costs **10–30% wall clock**
  (CLAUDE.md) and tightens the seed-to-seed noise floor from ±3–5 to ±0.7
  (`2026-04-29-overnight-FINAL.md:18`). For an *exploratory* smoke where you just want
  "does this roughly work / not catastrophically regress," you can drop it.
- **Estimated speedup:** 1.1–1.3× per run (the slowdown it adds, reclaimed).
- **Correctness-safety:** **NOT exact — opt-in.** It does not change the *biology*, but
  it widens run-to-run variance, so a smoke without it has **less statistical power**.
  Use ONLY for go/no-go smokes; KEEP `--deterministic` for any result that will be
  reported as a multi-seed verdict (the whole reason it exists is to detect 1–2-sum
  cluster effects below the ±3–5 noise floor).
- **Edit site:** launcher (drop the flag for smoke configs).
- **Effort:** ~0.
- **Falsifiable validation:** N/A (it's a power/speed trade, not a correctness claim).
  Frame explicitly as smoke-vs-production, never silently.

### #5 — Shorten run-length / readout knobs for SMOKES only (opt-in, trades power)
- **What (smoke-only):** for early de-risk where you're checking *direction*, not
  publishing a verdict, the run-length knobs are levers:
  - `n_steps` 1800 → e.g. 900 (halves wall clock; halves statistical settling).
  - `--visual-cortex-action-warmup-steps` 600 — needed for the visual circuit to mature;
    do NOT cut below ~600 for visual-cortex configs or you change what's being tested.
  - The readout window (`READOUT_START_MS=30/END=100`) and `reward_hold_steps=10` are
    biologically/methodologically load-bearing — **leave them alone**; cutting them
    changes the decision dynamics, not just the speed.
- **Estimated speedup:** linear in `n_steps` (1800→900 ≈ 2× on the run).
- **Correctness-safety:** **NOT exact — opt-in, trades statistical power.** A 900-step
  smoke can mis-rank a borderline config. Use to triage, then confirm the survivors at
  full 1800 steps + multi-seed + `--deterministic`.
- **Edit site:** launcher.
- **Effort:** ~0.
- **Falsifiable validation:** "the SIGN of the neural-vs-cheat nav-score gap agrees
  between the 900-step smoke and the 1800-step confirm." If it flips, the smoke is too
  short for that comparison.

### Not recommended / already done
- **dt=1.0:** already applied — the nav runner hardcodes `cfg.dt_ms = 1.0`
  (`g11_bg_runner.py:3131` and `:6366`). No further win available here (going to
  dt>1.0 risks Izhikevich Euler stability — `2026-05-03-dt1ms-speedup-validated.md`
  notes dt=1.0 is already "at the edge").
- **fp16 eligibility (`fp16_synapse_state`):** measured only **1.07×** and on the
  text/synonym path (`2026-05-05-perf-wave2-VERDICT.md`), not nav. Storage-only (compute
  auto-promotes to fp32), so it's *near*-exact but the tiny gain isn't worth changing nav
  state dtype. Skip for nav.
- **Rewriting the step / CSR matvec:** §2a shows the step is already clean and batched.
  No low-effort exact win there.

---

## 4. Concrete profiling plan (run when the GPU is free — measure, don't guess)

All commands assume CWD `E:\Documents\Projects\sim`. Run ONE at a time on an idle GPU.

**(a) Confirm the parallel speedup + find the contention ceiling (validates #1):**
```powershell
# Single-process baseline:
python -m research.runners.bench_parallel_gpu --steps 3000 --tag solo
# Two concurrent (parallelism=2):
Start-Process python "-m research.runners.bench_parallel_gpu --steps 3000 --tag p2a"
Start-Process python "-m research.runners.bench_parallel_gpu --steps 3000 --tag p2b"
# (wait for both; compare each tag's wall clock to solo)
# Then repeat with THREE, then FOUR concurrent. The ceiling is where
# per-proc wall clock x N stops beating solo x N — expect best at 2–3, worse at 4+.
```
Pass criterion for #1: at the chosen N, `(per-proc wall) × ? ` — i.e. N runs finish in
< N× solo. Stop increasing N when total throughput stops improving.

**(b) Per-section step timing (validates §2a "step is clean"):** set
`GPUConfig.enable_step_profiler = True` for a short nav run and read the
`t_init/t_stp/t_syn/t_dyn/t_plast/t_homeo` summary it logs every 500 steps
(`bridge.py:5134`). Confirms whether `t_syn` (CSR matvec) or `t_dyn` (neuron kernels)
dominates the step — if it's evenly spread small numbers, the step is launch-bound and
#2 (fewer kernels) is the right lever.

**(c) Isolate the readout-sync cost (validates #2b/#3):** time one full grid-8 1800-step
run two ways with `cupyx.profiler.benchmark` or a wall-clock `time.perf_counter()`
wrapper around the readout `for s in range(n_stim_steps)` block:
1. current code (70 `.get()`/action),
2. a scratch variant with on-device accumulation + 1 `.get()`/window (#3).
The delta is the sync-stall cost. Also wrap the whole runner in `cProfile` and look at
cumulative time in `cupy.ndarray.get` / `cudaDeviceSynchronize`:
```powershell
python -m cProfile -o nav.prof -m research.runners.g11_bg_runner `
    --moving-goal --goal-schedule multi --deterministic `
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 `
    --grid-size 8 --n-steps 300 --seed 42 --out research/findings/raw/_prof_smoke.json
python -c "import pstats; pstats.Stats('nav.prof').sort_stats('cumulative').print_stats(25)"
```
Use a short `--n-steps 300` so the profile is quick; the *fraction* of time in `.get`
/ sync extrapolates to the full run.

**(d) fast_spike_reset A/B (validates #2):** one seed, `--n-steps 600`, with vs without
`cfg.fast_spike_reset` (once the flag is added). Confirm (i) `final_quarter` sums
IDENTICAL (Izhikevich → exact), (ii) ~20–25% per-step speedup. Cross-check `nvidia-smi
dmon -s u` shows higher SM utilization in the fast path.

**(e) Determinism cost (#4):** same seed, `--n-steps 600`, with vs without
`--deterministic`; record the wall-clock delta (expect 10–30%) to quantify the smoke
trade.

---

## 5. Honest risks — exact/safe vs power-trading (must be opt-in)

**EXACT / safe (enable freely, biology + numerics unchanged):**
- **#1 Parallel multi-seed** — separate processes, identical math; only scheduling
  changes. *Operational* risk only: over-subscription → ~2× slowdown (cap at 2–3,
  mempool 0.30, barrier on the pool). Always diff one seed solo-vs-pooled to prove
  byte-identity.
- **#2 fast_spike_reset** — EXACT **for the Izhikevich model nav uses** (test-pinned).
  Risk: do NOT blanket-enable on HH/AdEx runners; it's only verified for Izhikevich.
- **#3 Readout sync-collapse** — EXACT integer counts; only the transfer timing changes.
  Risk: must preserve the `_capture` per-substep trace on sample trials and ship an
  integer-equality test, or you silently change a diagnostic.

**NOT exact — trades correctness/statistical power; opt-in, smoke-only, never silent:**
- **#4 Drop `--deterministic`** — widens the seed-to-seed noise floor (±0.7 → ±3–5).
  Fine for triage; will mask a 1–2-sum effect. KEEP it for any reported multi-seed
  verdict.
- **#5 Shorter `n_steps` / run-length** — fewer steps = less settling = a borderline
  config can be mis-ranked. Triage only; confirm survivors at full length. Do NOT shorten
  the readout window, `reward_hold_steps`, or visual warmup — those change the
  *dynamics/decision*, not just the speed.

**Net recommended de-risk recipe:** ship **#1 (parallelism=2–3) + #2 (fast_spike_reset)**
as the new default for nav de-risk batches — both EXACT, together ~2.0–2.6× on a
typical 6-run batch (66 min → ~25–33 min) with zero biology change. Add **#3** when there's
an afternoon for the hot-loop edit + test. Use **#4/#5 only** for fast exploratory
triage, explicitly labeled smoke, re-confirmed at full power before any verdict.

---

### Evidence index
- Hot loop / 70 `.get()`/action: `research/runners/g11_bg_runner.py:4695–4732` (`.get()` at `:4722`); run-length consts `:2239–2244`; `cfg.dt_ms=1.0` at `:3131`,`:6366`.
- Clean step / single sync: `sim/bridge.py:5142` (`_prev_any`), batched matvec `:5304`, fast-reset path `:5410`, step profiler `:5134`.
- Config defaults: `fast_spike_reset=False` `config.py:186`; `fp16_synapse_state=False` `config.py:203`; `enable_step_profiler` `config.py:503`.
- Sequential launchers: `research/findings/raw/_run_bioreward_da_derisk.ps1`, `_run_rank2_generalize2.ps1` (strict `foreach` over seeds).
- Parallel infra: `research/experiment_runner.py:276` (`parallelism` batches); `research/runners/bench_parallel_gpu.py`.
- Measured numbers: `2026-05-04-perf-speedup-stack.md` (parallel-3 ~1.7×; fast_spike_reset 1.29×); `2026-05-05-perf-wave2-VERDICT.md` (fp16 1.07×; contention 2.44× baseline inflation = the over-subscription trap); `2026-05-03-dt1ms-speedup-validated.md` (dt=1.0 already default, "at the edge"); `2026-04-29-overnight-FINAL.md:18` (deterministic noise floor ±0.7).
- CPU micro-benchmark (this audit): host-side readout arithmetic = 0.58 ms / 70 substeps (negligible → the cost is the sync stalls, not the math).
