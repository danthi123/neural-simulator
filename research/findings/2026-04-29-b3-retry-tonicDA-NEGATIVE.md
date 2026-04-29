# 2026-04-29 — B.3 retry with tonic DA: NEGATIVE on cheat-5

**Run:** `g11_bg_runner.py --moving-goal --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-tans --enable-tonic-da --goal-schedule multi --deterministic`. 6 seeds (42, 43, 44, 100, 101, 102), `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis (going in):** B.3 (TANs) was NULL on cheat-5 in 2026-04-28 because the plasticity_window_gate fires inside the reward-modulation block, which is skipped when reward=0 (between rewards). Cluster C v1 (tonic DA, scaffolded 2026-04-29) registers a tonic dopamine modulator that fires every step regardless of reward, which should let the gate function properly.

## Headline

**NEGATIVE — TANs + tonic DA more than doubles multi-goal cumulative distance vs baseline.**

| Condition | Mean | Std | n | Phase 0 mean | Verdict |
|---|---|---|---|---|---|
| baseline (no clusters, det) | 7.64 | 3.30 | 6 | 1.61 | reference |
| A+E (Cluster A + topographic) | 6.97 | 0.83 | 6 | 1.62 | -75% std, mean within noise |
| **B.3+C v1 (TANs + tonic DA)** | **16.50** | **3.97** | **6** | **5.52** | **NEGATIVE — phase 0 broken** |

Welch's t between baseline and B.3+C v1: ~3.7 (significantly worse).

## Per-seed breakdown

| Seed | Sum | P0 finalQ | P1 | P2 | P3 |
|---|---|---|---|---|---|
| 42 | 19.33 | 5.18 | 2.73 | 2.67 | **8.74** |
| 43 | 11.19 | 3.41 | 4.20 | 1.48 | 2.11 |
| 44 | 13.15 | 4.91 | 3.25 | 1.55 | 3.44 |
| 100 | 21.42 | 5.88 | **8.71** | 1.43 | 5.41 |
| 101 | 18.78 | **7.93** | 4.66 | 1.34 | 4.85 |
| 102 | 15.12 | 5.82 | 3.41 | 1.46 | 4.42 |

**Means:** P0=5.52, P1=4.49, P2=1.66, P3=4.83.

## Why this is NEGATIVE (interpretation)

**Phase 0 broken (5.52 vs baseline 1.61, +243%).** The agent isn't even acquiring the initial goal — it's nearly random across the first 450 steps. This is the diagnostic: tonic DA + TANs is preventing initial learning.

**Likely mechanism (untested):** under multi-goal with sparse positive rewards, tonic DA spends more time *below* baseline (negative reward EMA pulls concentration down). Per the `_default_dopamine_config`, the plasticity_rate target scales as `(concentration - baseline) * sensitivity`. When concentration < baseline, plasticity rate multiplier < 1.0 → plasticity is *damped* exactly when learning is most needed.

Add TANs on top: the plasticity_window_gate further restricts when STDP can fire. Combined effect: the corticostriatal plasticity machinery is gated by *two* mechanisms (DA-modulated rate + ACh-modulated window), both of which trend toward damping under sparse reward. Net result: very little learning happens.

**Secondary signal — P2 OK.** Phase 2 finalQ averages 1.66 — actually *better* than baseline (1.84). After 2 transitions, when the agent has built whatever weak mapping it can, the BG cascade does function. But the gating prevents the initial fast learning that baseline + A+E achieve.

## Comparison to A+E

A+E (Cluster A closed BG loop + Cluster E topographic) at 6.97 ± 0.83 is the operational best for multi-goal det. The architectural improvement (closed BG loop, topographic cortex pools) plus determinism gave the lowest std observed (0.83). Adding B.3+C v1 on top — without first making C v1 interact correctly with sparse-reward tasks — destroys the win.

## Implication for cheat-5 strategy

**Cluster C v1 (`--enable-tonic-da`) requires a separate sparse-reward investigation before being layered into flagship combinations.** The biology probe for tonic DA (PASS 2026-04-29) confirmed the modulator works at the micro level (concentration tracks reward EMA correctly, plasticity_rate multiplier follows). But under realistic multi-goal task structure with sparse positive rewards, the rate damping below baseline is destructive.

Options to investigate (future work, not part of this run):
1. **Asymmetric tonic DA ramps** — use the existing `--adaptive-da-ema-decay-negative` infrastructure to make negative reward windows ramp tonic DA *faster* but ramp positive *slower*, biasing the steady state above baseline.
2. **Tonic DA target = `excitability_drive` instead of `plasticity_rate`** — let tonic DA modulate cortex/MSN drive (motivational/effort-related role, per Schultz 2007) rather than plasticity rate. Removes the damping coupling.
3. **TANs without tonic DA** — accept B.3 (TANs) as NULL until a different gate-fire mechanism is implemented (e.g., gate fires on phasic-DA boundary detection, not inside reward-modulation block).

## Implication for A+C+D test

**Do NOT proceed with A+C+D as currently configured.** Adding C v1 (tonic DA) on top of A+D would likely show the same phase-0 breakage. Recommended pivots:

1. **A+D under multi-goal det** (no C v1) — clean test of A + Cluster D (hippocampus) composition.
2. **A+C v2 + D** — Cluster C v2 (compartmentalized DA, per-action channels) is structurally different from v1 (tonic broadcast). Worth trying since v2 doesn't have the same baseline-damping concern.
3. **A+D first, then add C separately** — establishes whether D adds value before mixing with the failed-mode C v1.

## What this run tested vs cheat-5 closure

This was the third in the planned sequence (after #2 renames and before #1 A+C+D). It tested whether a known-NULL cluster (B.3) could be rescued by adding tonic DA. The answer is **no** — and the failure mode reveals that **C v1 itself has a sparse-reward incompatibility issue** that needs addressing before C v1 is used as a flagship-eligible cluster.

## Provenance

- Code SHA at run time: `9501f26` (after 8 of 13 Wave-1 renames; before the last 4-6 renames).
- Wall-clock: ~35 min per run, 6 runs in parallel → ~35 min total.
- All 6 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_b3_retry_tonicDA.json`.
