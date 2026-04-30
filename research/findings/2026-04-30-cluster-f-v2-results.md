# 2026-04-30 — Cluster F v2 (CF-gated anti-Hebbian LTD): NEGATIVE — 3× worse than baseline

**Run:** `g11_bg_replicated_runner.py` multi-goal deterministic, n=6 seeds × 2 conditions = 12 replicas in 2 processes (~34 min total wall-clock via batched-replica framework, vs ~70 min if subprocess-spawned). `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

**Hypothesis (going in):** F v1 was NEUTRAL (per [`2026-04-29-cluster-f-results.md`](2026-04-29-cluster-f-results.md)). The cerebellar pathway was *implemented* — granule cells, parallel fibers, Purkinje cells, climbing fibers — but the PF→PC plasticity was driven by the same global reward signal as everything else. Per Albus 1971 §IV.C eq.4, real cerebellar LTD is **anti-Hebbian and gated by climbing-fiber complex spikes**, not by reward. Decoupling the two was hypothesized to release F's contribution: the cerebellum should learn faster motor-error corrections orthogonal to the reward-driven cortico-striatal trace.

## Headline

**F v2 is decisively NEGATIVE.** Both AFv2 and AEFv2 are ~3× worse than baseline and ~3× worse than the F v1 NEUTRAL counterparts. Welch's t vs baseline +8.4 (AFv2) and +9.3 (AEFv2). 6/6 seeds worse in both conditions.

| Condition | Mean | Std | n | Welch's t vs baseline | Verdict |
|---|---|---|---|---|---|
| baseline (no clusters) | 7.77 | 3.33 | 6 | reference | acid-test |
| AF (F v1, reward-gated PF→PC) | 7.37 | 1.83 | 6 | t=−0.26 | NEUTRAL |
| AEF (F v1) | 8.02 | 1.81 | 6 | t=+0.16 | NEUTRAL |
| **AFv2** (CF-gated anti-Hebbian) | **21.77** | **2.35** | 6 | **t=+8.42** | **NEGATIVE** |
| **AEFv2** (CF-gated + topographic) | **24.88** | **3.07** | 6 | **t=+9.25** | **NEGATIVE** |
| A+E (operational best) | 6.97 | 0.83 | — | — | ★ ceiling |

## Per-seed breakdown

| Seed | baseline | AF (v1) | AEF (v1) | **AFv2** | **AEFv2** |
|---|---|---|---|---|---|
| 42 | 7.42 | 8.56 | 10.35 | **18.94** | **24.03** |
| 43 | 5.93 | 5.00 | 6.48 | **18.75** | **19.41** |
| 44 | 5.06 | 6.47 | 6.27 | **24.16** | **26.06** |
| 100 | 10.95 | 9.63 | 8.01 | **23.52** | **24.97** |
| 101 | 12.70 | 8.65 | 10.09 | **22.90** | **28.51** |
| 102 | 4.58 | 5.88 | 6.93 | **22.38** | **26.31** |

Best F v2 seed (43, AFv2 = 18.75) is still 2.4× worse than the worst F v1 seed (102, AEF = 6.93). No partial overlap.

## Phase-decomposed (AFv2 example, seed 42)

```
P0 (goal=(6,6)): final_quarter_mean_distance = 6.13   ← agent ~6 steps from goal
P1 (goal=(6,1)): final_quarter_mean_distance = 8.19   ← worst phase, near random
P2 (goal=(1,1)): final_quarter_mean_distance = 3.12
P3 (goal=(1,6)): final_quarter_mean_distance = 1.50
```

`distance_log` first/last 10 (seed 42 AFv2):
```
first 10: [9, 10, 9, 8, 7, 6, 6, 7, 6, 7]   ← starting near goal=(6,6), bobbing
last  10: [1, 2, 1, 2, 2, 2, 2, 1, 0, 1]   ← finally converged in P3
```

The agent IS still moving and IS still getting some reward (motor_counts roughly balanced 109/106/80/155, reward sum 14 over 1800 steps). It's not catastrophic — just much noisier convergence. Phases are individually solvable but the agent doesn't transition cleanly between them.

## Why F v2 made it worse — three plausible mechanisms

**1. PF→PC weights drift without anchor.** F v1's reward gating kept LTD episodes correlated with task-meaningful events. F v2 fires LTD whenever the climbing fiber's complex-spike signal arrives — but in the reduced model, the CF signal is just a copy of motor-error, not a sparse "important error" signal. The cerebellum learns from every micro-correction, including noise.

**2. PF→PC LTD is now decoupled from the rest of the credit-assignment chain.** In F v1, every plastic synapse in the model — corticostriatal, cortex→GPe, cerebellum — saw the same reward signal. F v2 carves out the cerebellum into its own learning loop, but the cerebellum's "teaching signal" (CF complex spikes) and the cortico-BG "teaching signal" (DA reward) are not aligned. The cerebellum can be teaching one motor pattern while the BG cascade is reinforcing a different one. They fight.

**3. Albus 1971 anti-Hebbian gain too high for our connectivity.** The biological substrate has ~150K parallel fibers per Purkinje cell; we have 64. Anti-Hebbian LTD scales roughly with PF-firing-rate × CF-firing-rate × n_PF. At our scale, every CF event drives substantial PF→PC weight collapse. Real cerebellum spreads each LTD event across thousands of synapses; our reduced model concentrates it on a handful.

## Decision

**Cluster F v2 is NO-GO.** Tier 3 6-seed unanimous past the >12 NEGATIVE threshold. Not running tier 4. **Recommendation: revert F v2 in flagship configs, keep F v1 (NEUTRAL but harmless and biologically more honest than no cerebellum at all), keep A+E (6.97 ± 0.83) as the operational best.**

The Albus 1971 mechanism is real biology, but the scale gap between the cerebellum's natural connectivity (~150K PF/PC) and our ~64-PF reduced model breaks the LTD calibration. Closing this would require either:
- (a) Scaling up granule cell count by ~100×. Probably a 5-10× wall-clock cost per run. Not worth it for a NEUTRAL-at-best mechanism.
- (b) Adding a CF salience filter that only triggers LTD on "large-error" events. Adds another hyperparameter; biological correlate (climbing-fiber bursting vs single complex spikes) is real (Mathy 2009) but fitting it without overfitting is hard at n=6.
- (c) Constraining the magnitude per LTD event by the biologically expected PF/PC ratio. Cleanest fix; one constant. Worth ~1-2 hours to try if F resurfaces as a candidate.

For now: **F v2 closed NO-GO; do not stack F v2 on flagship.**

## What was learned (positive)

- The **batched-replica framework works**. 6-seed × 2-condition eval finished in ~34 min vs ~70 min for the equivalent 12 subprocess-spawned runs, on the same hardware, same total work. Validates the E.3 design: shared CuPy kernel dispatch beats Python orchestration overhead at this replica count.
- **Per-replica reward modulation works.** `bridge.cp_per_synapse_reward_override` correctly broadcasts each replica's reward to its own block-diagonal synapse range without leaking across replicas.
- **Cluster F v2 implementation is correct** — 47 unit tests pass, biology probe (CF-gated PF→PC LTD with anti-Hebbian sign) verifies correctly. The mechanism just doesn't help here.

## Cluster-stacking ceiling now empirically confirmed at 5 attempts

| Stack | n | Result | Verdict |
|---|---|---|---|
| baseline | 6 | 7.77 ± 3.33 | reference |
| A+D (closed-loop + hippocampus) | 6 | 7.62 ± 1.23 | NEUTRAL |
| A+D+E | 6 | similar | NEUTRAL |
| **A+E (Step 4)** | 6 | **6.97 ± 0.83** | **★ ceiling** |
| A+F (F v1) | 6 | 7.37 ± 1.83 | NEUTRAL |
| A+E+F (F v1) | 6 | 8.02 ± 1.81 | NEUTRAL |
| **A+F v2** | 6 | **21.77 ± 2.35** | **NEGATIVE** |
| **A+E+F v2** | 6 | **24.88 ± 3.07** | **NEGATIVE** |

A+E remains the operational ceiling for cheat-5 multi-goal det. **Five cluster-stacking attempts, no improvement past A+E.** The catalog buildout strategy ([`docs/plans/2026-04-28-cheat5-real-options-survey.md`](../../docs/plans/2026-04-28-cheat5-real-options-survey.md)) is now exhausted on the F branch; remaining unexplored cluster work is D v2 (sharp-wave-ripple replay for offline CA3 cleanup) which has a different mechanism than the per-step plasticity scaffolding F v2 just falsified.

## Files

- Results: [`research/findings/raw/g11_bg/g11_seed*_clusterFv2_AFv2_repl.json`](raw/g11_bg/) and `*_AEFv2_repl.json` (12 files, ~51KB each)
- Implementation: [`research/runners/g11_bg_runner.py:_run_pretraining_phase`](../../research/runners/g11_bg_runner.py) and [`sim/bridge.py:cp_per_synapse_reward_override`](../../sim/bridge.py)
- Test coverage: [`tests/test_cluster_f.py`](../../tests/test_cluster_f.py) (10 tests passing)
- Replicated runner: [`research/runners/g11_bg_replicated_runner.py`](../../research/runners/g11_bg_replicated_runner.py)
- Earlier F v1 finding: [`2026-04-29-cluster-f-results.md`](2026-04-29-cluster-f-results.md)
- Cluster-stacking ceiling discussion: [`docs/SCIENCE_ROADMAP.md`](../../docs/SCIENCE_ROADMAP.md) §4.7
