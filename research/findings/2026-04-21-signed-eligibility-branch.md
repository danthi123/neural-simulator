# Signed-Eligibility Branch — Review Package

**Date:** 2026-04-21
**Branch:** `signed-eligibility` (NOT merged)
**One-line change:** `sim/bridge.py` STDP eligibility accumulation goes from `+= cp.abs(weight_changes)` to `+= weight_changes` (preserves LTP/LTD sign).
**Verdict for merge:** **Conditionally safe.** Biological benchmarks unchanged. R-STDP experiment *improves*. Runner-side gates (G5.v2, G6, G7) are unchanged because they bypass the sim's reward path entirely. Recommended merge but with caveats in §6.

---

## 1. The change

`sim/bridge.py:3846-3859`. Single one-line diff (plus rationale comment):

```diff
 # Update eligibility traces if reward modulation is enabled
 if cfg.enable_reward_modulation and self.cp_eligibility_trace is not None:
     weight_changes = updated_weights - current_weights
-    self.cp_eligibility_trace[stdp_active_indices] += cp.abs(weight_changes)
+    self.cp_eligibility_trace[stdp_active_indices] += weight_changes
```

Effect: eligibility trace now preserves the sign of the underlying STDP pairing (positive for pre-before-post, negative for post-before-pre). Reward modulation (`Δw = lr · reward_error · eligibility`) becomes direction-aware: a positive reward selectively potentiates recently-LTP pairs, a negative reward selectively depresses them (and vice versa).

## 2. Biological-benchmark results (5 of 5)

Pre-change (fresh run on main, today) vs post-change (this branch):

| Benchmark | Pre | Post | Numerical delta |
|-----------|-----|------|-----------------|
| stdp-timing | PASS | PASS | unchanged (kernel-only test, doesn't exercise eligibility) |
| ei-balance | PASS | PASS | exc_rate 1.8→1.8 Hz; inh_rate 3.3→3.2 Hz; ratio 4.0→4.0 |
| stp-paired-pulse | PASS | PASS | unchanged (STP doesn't use eligibility) |
| gamma-oscillations | **FAIL** | **FAIL** | peak freq 124.3→121.7 Hz; gamma_fraction 0.33→0.32 |
| homeostasis | PASS | PASS | baseline 5.22→5.24 Hz; recovery 4.93→4.94 Hz |

**Important caveat**: none of these five benchmarks actually exercises the reward-modulation code path. STDP-timing tests the kernel directly; E/I balance, STP, gamma, and homeostasis all have `enable_reward_modulation = False`. Their "unchanged" status is nearly tautological — the signed-eligibility line only fires when reward modulation is on.

**Pre-existing failure you should know about**: `gamma-oscillations` has been failing on `main` before my work. Peak frequency is 124 Hz (expected 27–45 Hz per `SCIENCE_ROADMAP.md`). The benchmark was passing as of 2026-04-06 per the roadmap; regressed sometime between then and now. Unrelated to this change. Flagging for separate investigation.

## 3. R-STDP biological experiment — the real test

The `run_experiment_headless.py --experiment reinforcement` task is the one biological experiment that actually uses the eligibility trace. Ran 100 trials on both branches today:

| Branch | Early (trials 1–20) | Late (trials 80–99) | Δ success | Learning detected |
|--------|---------------------|---------------------|-----------|-------------------|
| `main` (fresh run)        | **0%** | **0%** | +0% | **NO** |
| `signed-eligibility`      | 20%    | 35%    | +15pp | **YES** |

**Main's R-STDP is completely broken** (0% → 0%, no learning). This is a pre-existing regression from the SCIENCE_ROADMAP's 2026-04-06 baseline (20% → 40%). My signed-eligibility change **partially restores** it (20% → 35% is close to the original 20% → 40%). This is a meaningful positive signal — the change *undoes* an unrelated regression that had already crept into main.

## 4. Runner-side gate re-runs on signed branch

Ran G5.v2 (in-sim R-STDP on navigation, 400 steps, 3 seeds) on the signed branch. Results:

| Seed | mean_dist | quarters | at_goal | verdict |
|------|-----------|----------|---------|---------|
| 42   | 11.65     | [11.30, 11.61, 11.86, 11.85] | 0 | still stuck at x=0 |
| 43   | 11.65     | [11.02, 11.86, 11.82, 11.91] | 0 | still stuck |
| 44   | 11.52     | [10.81, 11.60, 11.86, 11.82] | 0 | still stuck |

**All three seeds still fail G5.v2 on the signed branch.** Same degenerate attractor as on main: agent pins at x=0 early, distance stays at ~12, weights collapse from ~0.8 mean to ~0.3 mean across the episode.

### Why signed eligibility doesn't unlock G5.v2

The sim's eligibility trace decays with `tau ≈ 500ms`, integrating STDP events over that window. When the agent is stuck in a chronically-negative-reward regime:

- Most recent pre-before-post pairings (hidden→motor) have accumulated **positive** signed eligibility.
- Reward is constantly negative (agent far from goal).
- `reward_error × positive_eligibility = negative Δw` → weights that happened to be contributing recently get depressed, regardless of whether they were "correct" at that moment.

This is a finer-grained version of the unsigned pathology, but the same attractor exists: the agent can't escape its local minimum, and no amount of direction-aware reward can save it when temporal credit assignment is blurred across 500 ms of activity.

The runner-side G5.v3 approach avoids this because it uses **per-step credit assignment**: the hidden neurons that fired in *this exact step* get paired with *this step's* reward. No temporal decay, no blur. That's structurally more local than what the sim's eligibility mechanism can provide.

## 5. Effects on G6 and G7 runners

**None.** G6 (2D gridworld) and G7 (moving-goal) runners disable the sim's STDP and reward modulation entirely (`enable_stdp = False`, `enable_reward_modulation = False`) and apply the perceptron delta directly to `cp_connections.data`. They don't touch the eligibility trace at all. The signed-eligibility change is invisible to them.

So the G7 NO-GO conclusion from `research/findings/2026-04-21-g7.md` stands on both branches: the runner-side approach has a policy-inertia architectural limit that moving the goal exposes, and it's not fixed by sim-internal changes.

## 6. Recommendation

**Merge the signed-eligibility change to `main`**, with three caveats:

1. **It doesn't replace the runner-side G5.v3 approach.** I originally thought it would, based on the G5.v2 post-mortem. It doesn't, because the sim's temporal credit assignment through eligibility is too blurry. Keep both available: sim-native R-STDP (now improved by this change) for biological experiments; runner-side signed perceptron for per-step-credit learning tasks.

2. **It fixes a pre-existing R-STDP regression on main.** The biological reinforcement-learning preset was broken (0% → 0%) on current main. This change partially restores it (20% → 35%, close to the 2026-04-06 20%→40% baseline). That alone is a reason to merge.

3. **It does NOT fix the separate gamma-oscillations regression on main.** Gamma benchmark peak frequency is 124 Hz vs expected 27–45 Hz; that's independent of this change and needs separate debugging.

Suggested path: merge, then investigate gamma as a separate issue. The runner-side G5.v3 / G6 / G7 code stays as the primary sensorimotor learning approach.

**DO NOT merge without your sign-off.** I've done the engineering and the comparisons. The judgment call — "is a partial restoration of R-STDP behaviour good enough for main?" and "are the gamma/R-STDP pre-existing regressions acceptable to carry forward?" — is yours.

## 7. Alternative: hold on branch

If you'd rather not merge, this branch stays on GitHub as `signed-eligibility`. Anyone wanting signed R-STDP can `git checkout` it. Merge later after gamma is also diagnosed.

## 8. Raw data

- `research/findings/raw/benchmarks-baseline-pre-signed.json` (5 benchmarks, pre-change)
- `research/findings/raw/benchmarks-signed-post.json` (5 benchmarks, post-change)
- `experiment_rl_*.json` files in repo root (R-STDP runs)
- `research/findings/raw/signed-g5v2-seed{42,43,44}.json` (G5.v2 re-runs on signed branch)
- Branch diff: `git diff main signed-eligibility -- sim/bridge.py`

## 9. What I'd do next

If you merge:
1. Go back to main.
2. Debug gamma-oscillations regression (peak freq 124 vs 27-45). Separate from this work.
3. Resume G-series gates with a cleaner baseline.

If you don't merge:
1. Leave branch on GitHub.
2. Continue on main with runner-side G5.v3 as the primary learning approach.
3. Note in the G7 findings that the sim-native path was probed and found still-limited.
