---
type: plan
status: live
date: 2026-04-28
---

# Perf Option B — GPU-side motor reduction

**Status:** design + patch ready to apply behind a flag. Not yet committed.

**Context:** Phase 1 of [throughput investigation](../../research/findings/2026-04-28-throughput-investigation.md) identified two CPU↔GPU sync inefficiencies in the per-trial readout loop. **Option A** (cache motor indices CPU-side, still do full-array `firing.get()`) was implemented, measured (-15% on n=1, likely variance + extra overhead), and **reverted**. **Option B** (keep firing on GPU, do per-action small reductions on GPU, sync only 4 ints per readout step) was never tested.

## The patch

Two files: `research/runners/g11_bg_runner.py` (the runner-side flag + branched logic) and `tests/test_g11_bg_runner_flags.py` (smoke).

### A/B-controllable via flag

Don't replace the existing logic — add the new path behind a flag so we can A/B benchmark with one command.

**Flag:** `--enable-gpu-motor-reduction` (default off → current behavior unchanged).

### Code change

Find the readout block in BOTH the eval loop AND `_run_pretraining_phase`. Both currently look like:

```python
if readout_start <= s < readout_end:
    firing = bridge.cp_firing_states.get().astype(bool)
    for a in ACTION_NAMES:
        motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())
```

Replace with branched logic:

```python
if readout_start <= s < readout_end:
    if enable_gpu_motor_reduction:
        # Option B: keep firing on GPU. Per action, do small index+sum
        # reduction on GPU; sync only 4 ints (one per action) total.
        firing_gpu = bridge.cp_firing_states  # bool array, no sync
        for a in ACTION_NAMES:
            motor_counts[a] += int(firing_gpu[motor_idx_per_action[a]].sum().get())
    else:
        # Option A (current default): full firing-state DtoH per readout step.
        firing = bridge.cp_firing_states.get().astype(bool)
        for a in ACTION_NAMES:
            motor_counts[a] += int(firing[motor_idx_per_action[a].get()].sum())
```

Plumb `enable_gpu_motor_reduction` through `run_moving_goal_episode` kwargs + argparse + main pass-through (mirroring `enable_structural_pruning`).

## A/B benchmark

`scripts/benchmark_motor_reduction.sh`:

```bash
#!/usr/bin/env bash
# A/B benchmark Option A (current default) vs Option B (GPU-side motor reduction).
# Runs both back-to-back, single-process, same seed. Output is wall-clock per run.

set -euo pipefail

OUT_DIR="profile_outputs/motor_reduction_ab"
mkdir -p "$OUT_DIR"

# Identical config; only --enable-gpu-motor-reduction differs.
COMMON_ARGS=(
    --moving-goal --hippocampus --learned-perception --pfc
    --beacon-perception --beacon-replaces-goal
    --cue-reflex --cue-reflex-replaces-heuristic
    --landmarks --landmarks-replace-place
    --sensed-reward --bg-lateral-inhibition
    --adaptive-da --adaptive-da-ema-decay-negative 0.7
    --curriculum --curriculum-warmup-steps 100
    --n-steps 500 --seed 42
    --progress-print-interval 50
)

echo "=== A: Option A (current, full firing.get()) ==="
A_START=$(date +%s.%N)
python -m research.runners.g11_bg_runner "${COMMON_ARGS[@]}" \
    --out "$OUT_DIR/option_a.json" 2>&1 | tail -3
A_END=$(date +%s.%N)
A_ELAPSED=$(echo "$A_END - $A_START" | bc -l)
echo "Option A elapsed: ${A_ELAPSED}s"
echo

echo "=== B: Option B (GPU-side reduction, sync 4 ints) ==="
B_START=$(date +%s.%N)
python -m research.runners.g11_bg_runner "${COMMON_ARGS[@]}" \
    --enable-gpu-motor-reduction \
    --out "$OUT_DIR/option_b.json" 2>&1 | tail -3
B_END=$(date +%s.%N)
B_ELAPSED=$(echo "$B_END - $B_START" | bc -l)
echo "Option B elapsed: ${B_ELAPSED}s"
echo

# Compute speedup
SPEEDUP=$(echo "$A_ELAPSED / $B_ELAPSED" | bc -l)
echo "=== Result ==="
printf "Option A: %.1fs\n" "$A_ELAPSED"
printf "Option B: %.1fs\n" "$B_ELAPSED"
printf "Speedup:  %.2fx (B vs A)\n" "$SPEEDUP"
echo
echo "Decision rule:"
echo "  - B/A >= 1.10: ship B as default (>10% win)"
echo "  - 0.95 <= B/A < 1.10: keep A (within noise; switching adds complexity)"
echo "  - B/A < 0.95: keep A (B regresses; document why for future readers)"
echo
echo "Note: result varies with concurrency. To test at 4-concurrent, launch 4 of each"
echo "      via the webapp and compare aggregate step/s."
```

## Why this might win where A lost

Option A removed 4 small `.get()` calls per readout step but still did one full `firing.get()` (typically ~1500 bytes for the firing-state array). The full DtoH dominates over 4 small index transfers, so removing the small ones has marginal effect.

Option B keeps the full firing array on GPU and does FOUR very small `cupy.sum()` reductions per readout step. Each reduction syncs 4 bytes (1 int32). Total sync per readout step: 16 bytes vs. 1500 bytes for option A. **94% less DtoH bandwidth.**

Risk: 4 small CuPy reductions may have higher kernel-launch overhead than 1 big numpy sum. On RTX 3090 with kernel-launch latency ~5-10 μs per call, 4 launches = 20-40 μs overhead. Compared to a 1500-byte DtoH (~10-15 μs at PCIe 4.0), the launches might not pay off.

The honest answer is: we don't know. Need to measure.

## Decision matrix

After running the A/B benchmark:

| Result | Action |
|---|---|
| B/A ≥ 1.10 (≥10% faster) | ship B as new default; flip the flag default to True |
| B/A in 0.95-1.10 (within noise) | keep A; document B as "no measurable improvement" |
| B/A < 0.95 (B regresses) | keep A; revert the patch; document the negative result |

## Risks

- **Subtle correctness bug**: any mismatch between A and B's `motor_counts[a]` values across a 500-step run would be a real bug. The benchmark script should diff `option_a.json` and `option_b.json` (specifically the action_counts and phase_stats fields) and fail loudly if they differ. Add the diff check to the benchmark script.
- **CuPy kernel launch overhead under high concurrency**: B might win at 1-concurrent and lose at 8-concurrent (where context-switching dominates). Worth re-measuring at 4-concurrent if 1-concurrent shows a win.
- **Cold-start CUDA init bias**: first run pays for CUDA context creation. Run B then A (or vice versa) and compare middle runs of a 3-run sequence to factor out warmup.

## When to apply

- Wait until the v3-baseline-multi-goal batch and option-1 Tier 2 batch both finish (no concurrent GPU work for clean profiling).
- Apply the patch behind the flag.
- Run the A/B benchmark.
- If ≥10% win, ship B as default. Otherwise, leave the flag opt-in for future hardware where it might win.

## What this preserves vs changes

**Preserves:** all existing behavior when the flag is off (the default). Baseline + flagship + structural pruning paths all unchanged.

**Changes:** adds a single new flag + branched readout block + benchmark harness.

## Files

- Patch target: `research/runners/g11_bg_runner.py` (signature + argparse + main + 2 readout blocks)
- Test target: `tests/test_g11_bg_runner_flags.py` (1 smoke test verifying flag is accepted)
- Benchmark: `scripts/benchmark_motor_reduction.sh` (A/B harness)
