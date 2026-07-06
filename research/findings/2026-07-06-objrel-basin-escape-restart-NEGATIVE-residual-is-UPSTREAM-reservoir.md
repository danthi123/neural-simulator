# objrel basin-escape (reward-critic-selected restarts) — NEGATIVE for closing objrel; it SPLITS the residual into TWO parts (per-seed analytic-ridge check): genuine RESERVOIR info-absence on some seeds (ridge fails) AND read-out fragility on others where the info IS present (ridge finds it, the spiking learned read doesn't). The read-out plasticity is SOLVED on the clean seeds.

> **Correction note (same-day):** an earlier version of this doc (and its filename) said the residual is "purely UPSTREAM reservoir." A per-seed analytic-ridge check corrected that — see the two-part table below. The read-out is genuinely at fault on 45/101 (ridge OK there); the reservoir is genuinely at fault on 103/104 (ridge fails).

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
- **scramble=1.0 on 45 = a DEGENERATE READ-OUT (not the reservoir):** seed 45's read is always-THEME regardless of labels. Its ridge reads 0.92 (the info IS present in the feature), so this is a spiking READ-OUT collapse (always-THEME), not a reservoir-quality failure. (The genuinely reservoir-degenerate seeds are 103/104, where the ridge itself fails — see the two-part table.)

## The DIAGNOSTIC advance — the residual is TWO DISTINCT failures (per-seed analytic-ridge check; CORRECTS an earlier over-read of this doc)
Checking the analytic ridge (a linear read = "is the objrel signal linearly present in this seed's reservoir feature?") PER SEED splits the residual cleanly (do NOT say "the residual is purely upstream" — that was an over-read; the ridge does NOT read 1.00 on all seeds):
| seed | ridge-objr | spiking K1..K8 | scramble / no-reward | diagnosis |
|---|---|---|---|---|
| 42,43,44,46,102 | **1.00** | 1.00 all K | 0 / 0 | **CLEAN** (info present AND spiking learns it, role-specific, reward-driven) — 5/10 |
| 101 | **1.00** | 1.00→**0.67** | 0 / 1.0 | **READ-OUT learning miss** — info IS present (ridge 1.00), but the spiking learned read + reward critic learn it INCONSISTENTLY (non-monotone; the critic picks a worse restart with more candidates) + init-basin (no-reward init-luck) |
| 45 | 0.92 | 0.92→1.00 | **1.0** / 0 | **READ-OUT degenerate** — info mostly present, but the spiking read is always-THEME (scramble does not collapse) |
| 103, 104 | **0.00 / 0.17** | spurious 1.00 | 0/1.0 ; 0/0 | **RESERVOIR degenerate** — the ridge ALSO fails, so objrel is genuinely ABSENT from the feature; the spiking "1.00" is a degenerate always-THEME artifact |

So it is **two distinct residuals**, not one upstream wall: **(A) genuine RESERVOIR info-absence on 103/104** (the ridge can't read objrel → a real upstream reservoir-capacity issue), and **(B) READ-OUT fragility on 45/101 where the info IS present** (the ridge finds the discriminant, but the spiking learned read + the training-reward critic don't reliably converge to it). The read-out plasticity is genuinely SOLVED on the 5/10 clean seeds. The reward critic can't fix (B) because it doesn't correlate with objrel well enough; it can't fix (A) because the info isn't there.

## The NEXT mechanism (boundary = next mechanism) — attack BOTH residuals, each in its own pipeline stage
NOT more read-out restarts (non-monotone). The read-out plasticity is SOLVED on the 5/10 clean seeds; closing objrel robustly needs BOTH:
- **(A) RESERVOIR info-absence (103/104 — the ridge fails):** the reservoir feature genuinely does not encode objrel on some seeds → a real upstream capacity issue. Sweep the Hinaut-Dominey ESN hyperparameters — spectral radius, input scaling, leak rate, and SIZE — and/or a reservoir ENSEMBLE (population averaging) so the objrel signal is *linearly present* (ridge ≥ 0.9) on every seed. Fanned across cores (the skill's mechanical parallelism gate). This is the genuinely-upstream part.
- **(B) READ-OUT fragility where the info IS present (45/101 — the ridge reads 1.00 but the spiking read doesn't):** the learned spiking read + training-reward critic don't reliably converge to the ridge-discoverable discriminant. Levers: a read-out learner that reaches the ridge solution more reliably (e.g. the analytic-ridge init as a *teacher/warm prior* the plasticity refines — biologically an innate-then-tuned read, distinct from the retracted inert-ridge-warm-start because the plasticity still does refinement work and is measured emergent); a better basin-SELECTOR than training-MSE (held-out / novelty-gated selection); a homeostatic set-point pressuring the degenerate always-THEME read (45) off its collapse.
The honest scope: the read-out plasticity WORKS end-to-end on 5/10 clean seeds; robust closure needs (A) a reservoir that linearly encodes objrel on every seed AND (B) a read-out learner/selector that reliably finds the discriminant when it's present. Research-gate (A) (ESN capacity / spiking-LSM literature) before the sweep.

## Files
- `research/runners/_rungB1c_objrel_restart_basin_escape_derisk.py` (K-restart + reward-critic selection; NO sim/ edit), `_rungB1c_objrel_restart_aggregate.py` (the per-seed aggregator).
- `research/findings/raw/_restart_seed*.json` (10 per-seed records).
