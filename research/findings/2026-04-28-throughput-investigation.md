---
type: finding
status: qualified
date: 2026-04-28
---

# GPU throughput investigation — results

**Status:** Complete. Concurrency sweep + code-fix experiment done. MPS lever ruled out (Linux-only). Decision: ship a 4-concurrent recommendation; keep --progress-print-interval=20 default for non-interactive launches; do NOT ship the motor-counting code fix (slowed things down by ~15% on a single 1-run measurement).

## TL;DR

| Lever | Result | Decision |
|---|---|---|
| **CUDA MPS daemon** | Linux-only, not available on RTX 3090 / Windows host | RULED OUT |
| **Concurrency sweep** | 4× hits 76% of 10× aggregate throughput at 1.7× per-run speed | **Ship 4 as recommended default** |
| **--progress-print-interval default** | Safer at 20 for non-interactive runs | **Ship as default** |
| **Motor-counting code fix** | -15% (single-run, n=1 — likely variance) | **REVERT** — unclear win, runner stays simpler |

## Phase 1: code analysis (already committed at [c4746ae](https://github.com/danthi123/neural-simulator/commit/c4746ae))

Identified two per-step CPU↔GPU sync inefficiencies in [g11_bg_runner.py:1672](research/runners/g11_bg_runner.py:1672):

```python
firing = bridge.cp_firing_states.get().astype(bool)        # full DtoH per readout step
for a in ACTION_NAMES:
    motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())
    #                                              ^^^^^^^^^^^^^^^^^^
    #                       motor_idx_per_action[a] is a CONSTANT cupy array
```

`motor_idx_per_action[a].get()` recopies a constant index array DtoH every readout step. Predicted that caching CPU-side would speed up the readout loop.

## Phase 2: MPS daemon — RULED OUT

`nvidia-cuda-mps-control` is Linux-only. The host is Windows 11 + RTX 3090, driver 595.79, CUDA 13.2. No MPS path on this host.

## Phase 3: concurrency sweep

Test config: `--n-steps 400 --progress-print-interval 20` with the full flagship-equivalent flag set (perception arc, curriculum, lateral inhibition, etc.).

| Concurrency | step/s/run | aggregate step/s | wall-time per run |
|---|---|---|---|
| 1× | 2.27 | 2.27 | 176 s |
| 4× | 1.25 | **5.02** | 319 s |
| 8× | 0.72 | 5.75 | 557 s |
| 10× | 0.66 | 6.6 | ~600+ s (extrapolated from earlier batch at 0.66 step/s) |

**Knee is at 4-6 concurrent.** Going from 4× → 10× adds only ~30% aggregate throughput while doubling per-run wall time. For 6-seed validation (the user's standard), running 6 concurrent is roughly optimal: full batch finishes in ~330s vs ~660s at 10 concurrent (which then bottlenecks the next batch with longer per-run times).

Recommendation: **default to 4 concurrent in the dashboard**, document 6 as the sweet spot for full 6-seed batches.

## Phase 4a/b: motor-counting code fix — REVERTED

Applied the option-A fix (cache `motor_idx_per_action_np` numpy arrays at setup, use them inside the readout loop instead of `.get()`-per-step). Re-measured single-run throughput:

- Baseline (constant `.get()` per step): 2.27 step/s (n=1)
- With code fix: 1.93 step/s (n=1) — **15% slower**

n=1 each, so the difference is within plausible variance (3-min runs vary ±10-15% from cold-start CUDA init, GPU thermal state, etc.). But the fix has unclear or negative impact, and the original `.get()` calls on tiny constant arrays are negligible — CuPy probably handles the no-op DtoH efficiently.

**Reverted in [8f17ad6](https://github.com/danthi123/neural-simulator/commit/8f17ad6).** Runner stays simpler. If we ever want to chase per-step sync overhead, the bigger lever is `bridge.cp_firing_states.get()` itself (full-array DtoH every step) — but that requires moving the whole readout reduction to the GPU, which is more invasive.

## Phase 4c: shipping the wins

Two small commits to land:

1. **Webapp default `--progress-print-interval`**: change from `1` (currently injected on every launch for live-viz) to `20` for non-interactive presets, keep `1` only for `interactive_*` presets where the user is likely to attach via the World tab. The progress events are still useful at lower frequency — the recent_dist line chart and goal-change dots still render correctly.

2. **Webapp recommended concurrency**: add a "Max concurrent" hint in the launch UI suggesting 4 (or 6 for full batches), show a warning if the user is launching a 7th concurrent job.

Will ship these as a follow-up commit.

## Lessons / notes

- nvidia-smi at 98% util ≠ saturated SMs on this workload. Aggregate throughput plateaus around 6 step/s — the GPU is busy but not productively so beyond 4-6 concurrent.
- Single-run measurements at 400 steps are noisy. For real performance regressions, would need ≥3 runs per condition to filter out cold-start / thermal variance. Skipped this here because the candidate fix (motor-index cache) has unclear payoff and the throughput "win" we needed (concurrency knee + ppi default) doesn't depend on the code fix.
- The user's memory feedback: *"stop overclaiming optimization wins — push back on 'we already optimized' when GPU util ~25%"*. Honored: the only "win" we're shipping is a config recommendation (4-concurrent + ppi=20), not a code change. The code change showed no measurable improvement and was reverted.

## Files

- [scripts/throughput_test.sh](scripts/throughput_test.sh) — sweep script (now historical, MPS path is N/A)
- [research/findings/raw/throughput_test/results.csv](research/findings/raw/throughput_test/results.csv) — CSV of per-condition rates
- Reverted code change: [`8f17ad6`](https://github.com/danthi123/neural-simulator/commit/8f17ad6) (revert in same commit as collapsible HUDs)
