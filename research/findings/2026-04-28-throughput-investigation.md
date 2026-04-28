# GPU throughput investigation (work-in-progress)

**Status:** Phase 1 complete (code analysis). Phase 2/3 scripts staged, waiting for in-flight batch to finish before running on the GPU.

**Background:** 10 concurrent moving-goal runs on a single GPU give ~0.66 step/s each (~7 step/s aggregate). nvidia-smi reports 98% GPU utilization but the small kernels (~14.5K synapses, ~1500 neurons) likely under-saturate SMs — the 98% is "any kernel running," not SM saturation.

## Phase 1 findings — per-step CPU↔GPU sync confirmed

The hypothesis: `--progress-print-interval=1` plus per-trial motor-counting forces a CPU↔GPU sync every step in [g11_bg_runner.py:1672](research/runners/g11_bg_runner.py:1672).

The actual code (lines 1665-1674):

```python
for s in range(n_stim_steps):                      # readout window
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (...)
    if readout_start <= s < readout_end:
        firing = bridge.cp_firing_states.get().astype(bool)   # ← full-array DtoH sync
        for a in ACTION_NAMES:
            motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())
            #                                              ^^^^^^^^^^^^^^^^^^
            #                       motor_idx_per_action[a] is a CONSTANT cupy array
            #                       — recopying it CPU-side every step is wasted work
```

**Two distinct inefficiencies:**

1. `motor_idx_per_action[a].get()` inside the inner loop. The motor indices are set once at network build time ([g11_bg_runner.py:990](research/runners/g11_bg_runner.py:990)) and never change. Pulling them DtoH every readout step is pure overhead — 4 syncs per step × N readout steps per trial × 1800 trials.

2. `bridge.cp_firing_states.get()` copies the whole firing-state array (size = num_neurons) to CPU just to index 4 small subsets. The reduction can run on the GPU and we sync only 4 ints per readout step.

**Fix sketch (zero behavior change):**

```python
# Once at setup:
motor_idx_per_action_np = {a: motor_idx_per_action[a].get() for a in ACTION_NAMES}

# Per readout step (option A — minimal change, keeps numpy reduction):
firing_np = bridge.cp_firing_states.get().astype(bool)  # 1 sync (still full-array)
for a in ACTION_NAMES:
    motor_counts[a] += int(firing_np[motor_idx_per_action_np[a]].sum())

# Per readout step (option B — GPU-side reduction, syncs 4 ints):
firing_gpu = bridge.cp_firing_states  # no sync
for a in ACTION_NAMES:
    motor_counts[a] += int(firing_gpu[motor_idx_per_action[a]].sum().get())
```

Option B is strictly better. We'll measure both vs baseline in Phase 2.

## Phase 2/3 plan (staged, not yet run)

The throughput sweep script is in [scripts/throughput_test.sh](scripts/throughput_test.sh). It measures:

- **A.** Baseline: 1 run, no MPS, `--progress-print-interval 10` (already removes the print-induced sync; isolates the motor-counting sync)
- **C.** MPS on: 1 run, MPS daemon, `--progress-print-interval 10`
- **D.** Concurrency sweep: 1, 4, 8 concurrent under MPS

Script writes a CSV to `research/findings/raw/throughput_test/results.csv`.

### Decision matrix

| Condition pair | Win threshold | Action if win |
|---|---|---|
| A → C (MPS on, 1 run) | ≥1.3× step/s | document MPS startup in CLAUDE.md, add to launch.json startup |
| D 4× vs 8× | 4× aggregate ≥ 8× aggregate × 0.85 | recommend 4 as default concurrency |
| A vs option-B code fix | ≥1.5× step/s | ship the cache + GPU-reduction patch (separate commit) |

### Why NOT to run this now

The user has 10 cheat-5 runs in flight (~30 min remaining). Touching the GPU would corrupt their throughput measurements and risk killing them. Phase 2 fires after the in-flight batch finishes.

## Phase 4 (after Phase 2/3 results land)

If Phase 2 confirms ≥1.3× MPS win:
- Add `nvidia-cuda-mps-control -d` to the recommended startup in CLAUDE.md (Common Commands section).
- Optionally add a "Use MPS" toggle to the webapp launch UI (calls `nvidia-cuda-mps-control -d` if not running).

If Phase 3 confirms 4 concurrent ≈ 8 concurrent aggregate:
- Document recommended concurrency in CLAUDE.md.
- Optionally add a "Max concurrent" input to the webapp launch UI that shows a warning when exceeded.

If the motor-counting code fix wins ≥1.5×:
- Ship as a separate commit titled "perf(g11): cache motor indices CPU-side, reduce on GPU".
- This is a behavior-preserving change; tests should pass unchanged.
- Bumps single-process throughput, so MPS would compose multiplicatively.

## Honest caveats

- nvidia-smi at 98% can still mean SMs are underused. Real metric is **steps/sec** observed.
- Small networks (~1500 neurons) may not benefit from MPS as much as larger ones — kernel launches dominate, and MPS doesn't speed up kernel launches, only their interleaving.
- If MPS gives <1.3× on this workload, **don't ship the daemon-required setup**. The reliability cost of "you must remember to start MPS first" outweighs marginal speedup.
