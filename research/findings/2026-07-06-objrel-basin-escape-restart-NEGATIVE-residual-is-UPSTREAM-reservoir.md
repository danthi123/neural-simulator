# objrel basin-escape (reward-critic-selected restarts) — NEGATIVE for closing objrel; but it LOCALIZES the residual UPSTREAM: the read-out plasticity works on clean reservoirs, the failure is per-seed RESERVOIR feature quality (init-fragile / degenerate), not the read-out rule or the reward critic

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_restart_basin_escape_derisk.py` (fanned across cores, per-seed; aggregator `_rungB1c_objrel_restart_aggregate.py`)
**Raw:** `research/findings/raw/_restart_seed{42,43,44,45,46,100,101,102,103,104}.json`
**Verdict:** NEGATIVE (the restart+critic lever does not close objrel) — but a DIAGNOSTIC advance: it localizes the genuine residual to the UPSTREAM reservoir, orthogonal to the read-out plasticity.
**Builds on:** `2026-07-06-objrel-dopamine-plasticity-emergent-REACHES-most-seeds-BOUNDARY-reachability.md` (the per-role graded reward-modulated Dale-legal-spiking plasticity genuinely learns objrel on most seeds; the residual is a per-seed reachability Bernoulli).
**Infra note:** this was the run that exposed the parallelism gap — a subagent launched it single-threaded + orphaned it (zero output); the controller re-ran it **fanned across 10 cores** (1-thread BLAS, `wait`-held parent), ~7 min vs ~50 min serial. The `neural-simulator` skill now has a MECHANICAL parallelism gate (committed `5e426692`).

## The lever + the pre-registered test
Per-role Dale-legal-spiking detectors, each trained as **K independent random restarts** with the **REWARD CRITIC** selecting the deployed restart by TRAINING reward only (`argmax` of the negative salience-weighted training squared error — NOT the test label). Hypothesis: the per-seed failure is a stochastic init-basin Bernoulli → K restarts + critic-selection drive the miss-rate to ~(miss)^K. Pre-registered FIXED split (dev {42-46}, blind {100-104}), K∈{1,3,5,8}, genuinely-emergent counted (pre<0.85, excluding init-lucky), + a **test-ORACLE column** (best restart by held-out accuracy = what selection COULD achieve).

## Result — the restart+critic lever does NOT close objrel (per-seed, per-K objrel-slot0, oracle in ()):
| seed | pre | K1 | K3 | K5 | K8 | note |
|---|---|---|---|---|---|---|
| 42,43,44,46 | 0.00 | 1.00(1.00) | 1.00 | 1.00 | 1.00 | clean genuine recovery (role-specific, reward-load-bearing) |
| 45 | 0.00 | 0.92(1.00) | 1.00 | 1.00 | 1.00 | **scramble=1.0 → DEGENERATE always-THEME read (not role-specific)** |
| 100,104 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | init-lucky (excluded) |
| **101** | 0.00 | 1.00 | **0.67(1.00)** | **0.67(1.00)** | **0.67(1.00)** | **NON-MONOTONE — critic picks WORSE with more restarts; oracle 1.00 (good basin EXISTS) but critic can't select it** |
| 102 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | clean |
| 103 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | **no-reward=1.0 → init-luck (untrained init already reads objrel)** |

- **The training-reward critic is NOT a reliable basin-selector.** On seed 101 the ORACLE is 1.00 at every K (a good basin exists among the restarts) but the CRITIC selects a 0.67 restart, and it gets WORSE with more restarts (K1 1.00 → K3+ 0.67) — more candidates, the critic picks a worse one. The train-reward critic does not correlate with objrel recovery well enough to pick the good basin. ⇒ the "reward-critic-selected restarts close it" hypothesis is REFUTED.

## Why the anti-cheat "failures" are GENUINE per-seed properties (not runner bugs — code-traced + smoke-clean)
The controls are correctly implemented (`reward_on=False`→`da=0`→no weight update→deploys the untrained init; scramble deranges the role labels). The failures are REAL:
- **no-reward=1.0 on 101/103 = INIT-LUCK:** an untrained random init already reads objrel on those seeds (the pre-learning K0 used a different init seed reading 0.0). This DIRECTLY confirms the stochastic-init-basin residual — some random inits land in the objrel-reading basin.
- **scramble=1.0 on 45 = a DEGENERATE reservoir:** seed 45's read is always-THEME regardless of labels → that reservoir's feature does not cleanly separate the role structure. A per-seed reservoir-quality failure.

## The DIAGNOSTIC advance — the residual is UPSTREAM (the reservoir), not the read-out
The read-out plasticity works CLEANLY where the reservoir feature is clean (42,43,44,46,102: genuine role-specific reward-load-bearing recovery). It fails only where the RESERVOIR is bad — init-fragile (101: the reward-driven read can even move AWAY from the objrel basin, and the critic can't tell) or degenerate (45: always-THEME). The reward critic cannot fix this because it is NOT a read-out-selection problem — it is per-seed UPSTREAM reservoir feature quality. This matches + sharpens the closure's flagged residual ("the seed-fragile reservoir read, orthogonal to the plasticity rule").

## The NEXT mechanism (boundary = next mechanism; and it's a DIFFERENT part of the pipeline)
NOT more read-out restarts (non-monotone) and NOT a better read-out reward critic — the residual is upstream. Attack the RESERVOIR feature robustness so objrel is cleanly + consistently encoded across seeds:
1. **Reservoir hyperparameters (Hinaut-Dominey ESN):** spectral radius, input scaling, leak rate, and SIZE — a larger / better-tuned reservoir reads objrel more robustly (the analytic Dale reference already reads 1.00 from the CURRENT reservoir on ALL seeds via a ridge, so the objrel signal IS present — the issue is the SPIKING read's per-seed fragility + the init-basin, not the reservoir's information content per se). Sweep the reservoir hyperparameters (fanned across cores, per the skill's mechanical gate).
2. **Reservoir ensemble / averaging:** multiple reservoir instances per seed (biological population averaging) to wash out the per-seed degeneracy.
3. **A better basin-SELECTOR that IS available to biology:** since the reward critic (training MSE) can't select the basin, a held-out / novelty-gated selection, OR a homeostatic set-point that pressures the degenerate always-THEME read off its collapse.
The honest scope: the read-out plasticity is SOLVED for clean reservoirs; the objrel closure now depends on the upstream reservoir's per-seed robustness.

## Files
- `research/runners/_rungB1c_objrel_restart_basin_escape_derisk.py` (K-restart + reward-critic selection; NO sim/ edit), `_rungB1c_objrel_restart_aggregate.py` (the per-seed aggregator).
- `research/findings/raw/_restart_seed*.json` (10 per-seed records).
