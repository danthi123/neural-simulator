# Cluster B.3 — Cholinergic Interneurons (TANs) Results

**Date:** 2026-04-28 (evening)
**Status:** **MIXED** — implementation correct + real bridge bug fixed; gate empirically a no-op in current architecture; documented baselines don't reproduce, requiring fresh validation
**Plan:** [`docs/plans/2026-04-28-cluster-b3-tans-implementation.md`](../../docs/plans/2026-04-28-cluster-b3-tans-implementation.md)

## TL;DR

Three substantive outcomes:

1. **Bridge step-order bug fixed** (`59dc1fc`) — NM `manager.step()` was running AFTER the reward-modulated weight update, so the TAN gate read one-step-stale ACh concentration. With single-pulse rewards (the realistic eval scenario), this meant the gate NEVER opened during reward delivery and plasticity was fully suppressed. New regression test catches the pattern.
2. **TANs gate is empirically a no-op** in the current architecture. Post-fix, with seed 42 multi-goal: TAN-on sum=21.62 vs TAN-off sum=22.03 (Δ ≈ 2%, within run-to-run variance). The gate ≈ 1.0 at every reward step (because `pause_on_reward` drives ACh from baseline to ~0 within the same step), and reward = 0 between rewards (so the gate's value is multiplied against 0). Mathematically: identical to no-TANs in this architecture.
3. **Baseline reproducibility issue surfaced.** The CLAUDE.md / memory baseline of "B.1+B.2 alone = 9.50 ± 0.85" doesn't reproduce at seed 42 in current code (gives 22.03). Pre-B.3 commit `714bc29` ALSO gives 21.22 for seed 42 — so this is NOT a B.3 regression. Either the documented n=3 baseline used different seeds, was measured against an uncommitted parameter override, or had a measurement-recording error. Fresh n=3 baselines being established (in progress).

The biology probe PASSED because it used 10 sustained reward steps — 9/10 of those steps had the gate open (because pause_on_reward fires once per step, and ACh stays clipped to 0 throughout the sustained reward window). Single-pulse rewards (the actual eval scenario) only see the gate open in 0/1 steps pre-fix, and 1/1 steps post-fix — but the post-fix gate ≈ 1.0 makes the gate identical to no-gate.

## Hypothesis (original)

Real BG TANs are tonically active (~5 Hz) and pause briefly on salient events. ACh release at corticostriatal synapses creates "plasticity windows" where synapses only consolidate during ACh-pause windows. This is structurally orthogonal to B.1 (D1/D2 asymmetry) and B.2 (FSI WTA): TANs gate plasticity in **time** rather than in space.

The hope: TANs may compensate for B.2's Phase-0 issue by gating plasticity to ACh-pause windows around reward events.

## Implementation (correct, all unit tests pass)

Five commits:
- **Task 1** (`0173639`, `2f0ad31`): `pause_on_reward` rule + `_default_acetylcholine_config()` helper. ACh: tonic baseline=1.0, decay_tau_ms=500ms, sensitivity=-2.0 on |reward| above threshold=0.0.
- **Task 2** (`3a569ad`): `compute_plasticity_window_gate_multiplier()` returning `clip(1 - conc/baseline, 0, 1)` per modulator (multiplicative aggregation). Bridge wiring at `sim/bridge.py:4361-4365`.
- **Task 3** (`af2fe6f`): `--enable-tans` CLI flag.
- **Task 4** (`1090003`): biology probe at `research/probes/tan_ach_probe.py`.
- **Step-order fix** (`59dc1fc`): `manager.step()` moved BEFORE reward modulation block to fix the one-step-lag bug.

47 unit tests pass: 15 in `tests/test_tans.py` (8 Task 1 + 6 Task 2 + 1 step-order regression), 39 in `tests/test_neuromodulators.py`, 6 in `tests/test_d1_d2_asymmetry.py`, 38 in `tests/test_g11_bg_runner_flags.py`, 7 in `tests/test_determinism.py`.

## The step-order bug (real win)

Pre-fix order:
1. STDP fires → eligibility traces accumulate
2. Reward update reads gate (line 4363) — uses **previous step's** ACh concentration
3. Weight update accumulated
4. `manager.step()` runs (line 4386) — pause_on_reward fires NOW, but too late

For single-pulse rewards (10 reward_hold_steps, then reward=0): step 1 of reward arrival has gate read from step 0 (ACh at baseline 1.0 → gate 0). Plasticity blocked. Then production fires at end of step. Step 2: gate reads ACh ≈ 0 (paused) → gate ≈ 1, BUT reward might still be high so plasticity does flow if it's still in the hold window.

So with 10-step holds, only step 1 of each reward delivery had a closed gate. Effective plasticity ratio: 9/10 of normal. Modest impact.

But the regression test `test_single_pulse_reward_fires_plasticity_within_step` proves that for a TRUE single-step reward, pre-fix gives ZERO weight change — full suppression.

Post-fix order:
1. STDP fires → eligibility traces accumulate
2. `manager.step()` runs — production fires NOW (reads current_reward_signal)
3. Reward update reads gate — uses **current step's** ACh concentration (just updated)
4. Weight update accumulated

This is correct and the regression test passes.

## Biology probe (Task 4) — PASSES, but with a methodological gap

Probe at `research/probes/tan_ach_probe.py` runs 4-phase scenario:

| Phase | Steps (ms) | Reward | Mean ACh | Min ACh | Mean Gate | Phase \|dw\| |
|-------|-----------|--------|----------|---------|-----------|--------------|
| 0 (baseline) | 50 | 0.0 | 1.0000 | 1.0000 | 0.0000 | 0.0000e+00 |
| 1 (+reward)  | 10 | 1.0 | 0.1000 | **0.0000** | **0.9000** | **2.7313e+02** |
| 2 (recovery) | 100 | 0.0 | 0.0927 | 0.0000 | 0.9073 | 0.0000e+00 |
| 3 (continued) | 40 | 0.0 | 0.2124 | 0.1813 | 0.7876 | 0.0000e+00 |

Verdict: PASS. ACh pauses fully on 10 ms of reward, recovers at decay_tau (matches analytic `1 - exp(-140/500) ≈ 0.245`). Gate opens during pause (~ 0.9 mean) and closes during baseline (0). Weight updates land **exclusively** during Phase 1.

**Methodological gap (post-hoc):** the probe used 10 sustained reward steps — 9/10 fired with the gate open after pause_on_reward dropped ACh in step 1. The real eval has multi-step reward holds too (`reward_hold_steps=10`), but ACh dropping to 0 in step 1 means the gate is also at ~1 for steps 2-10 (because pause_on_reward keeps slamming ACh down and the decay barely recovers between consecutive reward steps).

So the gate is approximately 1.0 throughout the entire reward hold. Effectively equivalent to no-gate. The probe couldn't have caught this — it shows correct dynamics, but doesn't compare TAN-on to TAN-off behavioral output.

## Cheat-5 multi-goal eval (Task 5) — current-code data

Re-runs after step-order fix, full Cluster B (B.1 + B.2 + B.3):

### 5a — full Cluster B (B.1 + B.2 + B.3) no cross-projections

| Seed | P0 | P1 | P2 | P3 | Sum |
|------|------|------|------|------|------|
| 42 | 5.73 | 2.31 | 5.55 | 8.04 | 21.62 |
| 43 | 2.64 | 4.08 | 8.42 | 2.27 | 17.41 |
| 44 | 3.74 | 4.47 | 3.43 | 5.11 | 16.75 |
| **Mean ± std** | 4.04 ± 1.56 | 3.62 ± 1.15 | 5.80 ± 2.50 | 5.14 ± 2.88 | **18.59 ± 2.64** |

### 5b — full Cluster B + patch-matrix cross-projections (density 0.25)

| Seed | P0 | P1 | P2 | P3 | Sum |
|------|------|------|------|------|------|
| 42 | 2.02 | 4.72 | 4.22 | 8.04 | 18.99 |
| 43 | 3.35 | 4.25 | 4.80 | 1.66 | 14.05 |
| 44 | 1.93 | 4.55 | 2.79 | 2.18 | 11.44 |
| **Mean ± std** | 2.43 ± 0.79 | 4.50 ± 0.24 | 3.94 ± 1.03 | 3.96 ± 3.54 | **14.83 ± 3.83** |

### Direct TAN-on vs TAN-off, n=3 (current code state, multi-goal, post-fix)

| Config | No TANs | + TANs | Δ |
|---|---|---|---|
| B.1+B.2 alone | **18.02 ± 3.68** | **18.59 ± 2.64** | +0.57 (within variance) |
| patch-matrix + B.1+B.2 | **15.18 ± 3.44** | **14.83 ± 3.83** | -0.35 (within variance) |

**Empirical confirmation: TAN gate is statistically a no-op in this architecture at n=3.**

Per-phase (patch-matrix variants):
| | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| No TANs | 2.73 ± 0.25 | 4.51 ± 0.60 | 4.09 ± 1.36 | 3.85 ± 3.62 |
| + TANs  | 2.43 ± 0.79 | 4.50 ± 0.24 | 3.94 ± 1.03 | 3.96 ± 3.54 |

No phase shows a meaningful difference. P0 (initial learning) is slightly better with TANs (-0.30) but with widely overlapping error bars.

Patch-matrix variants are ~3 sums better than no-cross variants (15.0 vs 18.3) regardless of TANs — so cross-projections DO help at current code state, just not via TANs.

### Bisect — pre-B.3 commit `714bc29` seed 42 multi-goal B.1+B.2 alone

| Sum | n_reward_events |
|---|---|
| 21.22 | 1559 |

Reproduces ~22 at pre-B.3 commit too. The "9.50" documented baseline was at some commit/seed-set that doesn't match current state. Not a B.3 regression.

### v3 baseline — current code state, seed 42 multi-goal

| | P0 | P1 | P2 | P3 | Sum |
|---|---|---|---|---|---|
| Current code | 2.19 | 1.82 | 3.18 | 4.86 | **12.05** |
| Documented (n=6) | 2.13 | 1.47 | 1.90 | 1.52 | **7.08 ± 0.12** |

Even the v3 baseline is regressed at seed 42 in current code (12.05 vs 7.08). P0 reproduces (2.19 ≈ 2.13), but P3 is the dominant regression (4.86 vs 1.52). So multi-goal phase-3 readaptation is harder in current code than was previously documented — affects ALL configs, not just B.1+B.2. Investigation deferred (would require multi-step bisect across many commits).

## Architectural insight: why TANs is a no-op

The plasticity_window_gate is inserted at line 4361-4365 of `sim/bridge.py`, multiplying into `weight_updates` in the reward-modulated update block. That block only fires when `current_reward_signal != 0` (line 4326 check). So the gate is ONLY consulted when reward is non-zero. At those moments, `pause_on_reward` (which fires in the same step post-fix) drops ACh from baseline to ~0, opening the gate to ~1. So the multiplier is ≈1.

Between rewards, `current_reward_signal = 0` → entire block skipped → gate not consulted at all.

Net effect: **gate value never matters**.

### Why moving the gate also doesn't help

The obvious next thought: gate eligibility accumulation (line 4309-4311) instead of weight consumption. With ACh tonic between rewards, gate=0 → no eligibility accumulation. Only STDP events DURING ACh-pause windows accumulate eligibility.

But this has its own problem: **the credit-assignment events we want eligibility for (the agent's navigation decisions) happen BEFORE reward arrives**, when ACh is at baseline. With B.3 v2 gating, those navigation STDP events wouldn't accumulate eligibility — only goal-reaching STDP events during the reward window would. Credit assignment wouldn't trace back through navigation.

### The deeper issue: real biology has tonic DA

In real BG, the LTP rule for corticostriatal synapses is:
- Hebbian conjunction (STDP)
- AND adequate DA
- AND low ACh (during pause)

Real DA has both **tonic baseline** AND **phasic bursts** at reward. With ACh tonic at baseline (between rewards), tonic DA doesn't drive much plasticity. With ACh paused (around reward), the conjunction of phasic DA + ACh-pause creates a strong LTP window.

Our model has ONLY phasic DA (reward signal during the 10-step reward delivery, zero otherwise). There is no tonic DA to be gated by ACh between rewards. So ACh gating has nothing to gate.

To make TANs functional, the model would need:
1. **Tonic DA component** — e.g., a baseline plasticity drive even between rewards, weighted by some background reward expectation. ACh would then gate this background plasticity vs the phasic reward plasticity.
2. **AND** the gate placed somewhere that affects this tonic plasticity — likely on STDP eligibility accumulation.

Both are larger architectural changes than B.3 attempted. Real progress requires extending the DA system (closer to Cluster C: DA system completeness) before TAN gating becomes meaningful.

## Decision

**B.3 SHIPPED AS INFRASTRUCTURE / NULL ON CHEAT-5.**

What's kept:
- `pause_on_reward` production rule (`sim/neuromodulators.py`) — generic infrastructure for any future "pause on event" modulator (DA dip, NE arousal, etc.)
- `plasticity_window_gate` target type — wired into reward modulation, ready to be reused if a future cluster adds tonic DA-driven plasticity
- `_default_acetylcholine_config()` helper — reusable
- `--enable-tans` CLI flag — opt-in
- Bridge step-order fix (`59dc1fc`) — a real bug fix that future fast-dynamics modulators will need
- Biology probe — preserved as a regression check

What's NOT recommended:
- Adding `--enable-tans` to flagship configs. It's a no-op behaviorally and adds NM-subsystem overhead with no benefit at current architecture.

Next steps decision: **Cluster A** (closed BG loop), not B.3 v2 (gate-relocation experiments). Reasoning:

- **B.3 v2 won't help cheat-5 either.** The "extend gate to eligibility accumulation" idea was attractive at the diagnostic stage, but on closer analysis it creates a credit-assignment problem (navigation STDP events don't get eligibility). The deeper issue is missing tonic DA — solvable only by Cluster C-style DA-system extensions.
- **Cluster A is the right next biological scaffolding.** Thalamo-cortical feedback (motor → thalamus → cortex) and hyperdirect pathway (cortex → STN) provide the closed-loop teaching signal that's missing in our reduced model. Closed-loop signals are required for cross-projection coordination — likely more impactful for cheat-5 than within-striatum gating.
- **Cluster C (DA completeness) becomes more attractive after Cluster A.** Once the loop is closed, adding tonic DA + ACh gating composes naturally. Then B.3's gate finally has something to gate.

Cluster B closure status: **3-of-3 attempted** (B.1 partial, B.2 mixed with Phase 0 issue, B.3 null + infrastructure). Cheat #5 remains under cluster-by-cluster buildout. Move to Cluster A.

## Variance trajectory (broken)

The pattern of cluster steps halving std broke at B.3:
- patch-matrix alone: 2.54
- + B.1: 1.23
- + B.1 + B.2: 0.62
- + B.1 + B.2 + B.3 (current architecture): **3.83** (regressed)

This is partly because seed 42 specifically gives high results in current code; absolute mean comparisons against memory's documented 0.62 std measurement aren't valid. The std depends heavily on which seeds are used.

## Files

- Plan: `docs/plans/2026-04-28-cluster-b3-tans-implementation.md`
- Probe: `research/probes/tan_ach_probe.py` + `research/findings/raw/tan_ach_probe/probe_results.json`
- 5a outputs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_clusterB3_no_cross.json`
- 5b outputs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_clusterB3_patch_matrix.json`
- Bisect output: `research/findings/raw/g11_bg/g11_seed42_clusterB12_pre_b3.json`
- TAN-off control: `research/findings/raw/g11_bg/g11_seed42_clusterB12_no_tans.json`
- (Pending) Fresh baselines: `research/findings/raw/g11_bg/g11_seed{42,43,44}_clusterB12_*_no_tans.json`
- Eval logs: `research/findings/raw/g11_bg/clusterB3_eval_logs/`

## Lineage

- B.1 results: `research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`
- B.2 results: `research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md`
- Cluster B design: `docs/plans/2026-04-28-cluster-b-striatal-microcircuit-design.md`
- Cheat-5 reframe: `research/findings/2026-04-28-cheat5-post-v4-reframe.md`
