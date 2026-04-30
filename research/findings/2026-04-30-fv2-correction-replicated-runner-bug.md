# 2026-04-30 — CORRECTION: F v2 NEGATIVE finding was a replicated-runner bug

## TL;DR

The Cluster F v2 NO-GO finding from earlier today (`2026-04-30-cluster-f-v2-results.md`)
was based on a 6-seed eval run via `g11_bg_replicated_runner.py`. The results
showed AFv2 21.77 ± 2.35 and AEFv2 24.88 ± 3.07 — 3× worse than baseline.

**Re-running the same conditions on `g11_bg_runner.py` (single runner)
gives AFv2 7.20 ± 2.75 and AEFv2 8.14 ± 3.46 — NEUTRAL relative to A+E
baseline (7.18 ± 1.58).** The 3× discrepancy is a real replicated-runner
bug, not a cluster-F-v2 problem.

The F v2 mechanism itself is not broken. It's just neutral on cheat-5,
not negative. Updates to the synthesis and CLAUDE.md follow.

## Diagnosed bug

The replicated runner's reward-modulation timing differs fundamentally
from the single runner:

**Single runner** (`g11_bg_runner.py`):
- 200 sim steps (the stim window) per env step.
- `bridge.core_config.current_reward_signal = delivered_reward` is set
  AFTER the env step's action+reward, so the next env step's 200 sim
  steps all see that reward in their effective_signal.
- Per env step: ~200 weight-update sim steps.

**Replicated runner** (`g11_bg_replicated_runner.py`):
- 200 sim steps with `cfg.current_reward_signal = 0.0` (reward zeroed
  during stim window — line 443).
- After action+reward: `cfg.current_reward_signal = 1.0` set once,
  `cp_per_synapse_reward_override` set with per-replica rewards, then
  ONE additional sim step runs (line 524-525) for reward modulation.
- Per env step: 1 weight-update sim step.

**Net effect:** the replicated runner does ~200× fewer weight updates
per reward event than the single runner. For plasticity-heavy
experiments (especially F v2's CF-gated LTD which depends on accumulated
PF→PC weight changes), this is catastrophic — the cerebellum barely
learns. For low-plasticity experiments (A baseline), the impact is
muted because there's less learning to accumulate.

This explains why:
- A baseline replicated (n=1, 600 steps): phase 0 = 1.65 (normal,
  matching single-runner ~1.5).
- A+F v2 replicated (n=1, 1800 steps): sum = 23.50 (3× worse than
  single-runner 7.80 for the same seed).
- A+F v2 single (n=6): mean 7.20 (NEUTRAL).

## Verification matrix

| Cond | Runner | n | Mean | Std | Per-seed-42 sum |
|---|---|---|---|---|---|
| A+F v2 | replicated (original 2026-04-30 eval) | 6 | 21.77 | 2.35 | 18.94 |
| A+F v2 | replicated (n=1 debug) | 1 | 23.50 | — | 23.50 |
| **A+F v2** | **single (revisit)** | **6** | **7.20** | **2.75** | **7.80** |
| A+E+F v2 | replicated (original) | 6 | 24.88 | 3.07 | 24.03 |
| **A+E+F v2** | **single (revisit)** | **6** | **8.14** | **3.46** | **10.99** |
| A+F v1 | single (revisit) | 6 | 6.74 | 1.65 | 8.98 |
| A+E baseline | single (earlier) | 6 | 7.18 | 1.58 | 9.86 |

The replicated runner consistently shows 3× higher means than the single
runner for F v2 conditions. F v1 wasn't tested via replicated runner
(was originally evaluated via single runner per `2026-04-29-cluster-f-results.md`),
so its NEUTRAL finding stands.

## Corrected verdict

**Cluster F v2 — NEUTRAL on cheat-5 multi-goal det.**

| Cond | Mean | Std | n | Welch t vs A+E baseline | Verdict (corrected) |
|---|---|---|---|---|---|
| A+E (baseline) | 7.18 | 1.58 | 6 | reference | acid-test |
| A+F v1 | 6.74 | 1.65 | 6 | -0.48 | NEUTRAL (slight improvement, NS) |
| A+F v2 | 7.20 | 2.75 | 6 | +0.02 | NEUTRAL (no effect) |
| A+E+F v2 | 8.14 | 3.46 | 6 | +0.62 | NEUTRAL (slight worsening, NS) |

F v2 doesn't help past A+E, but it's not actively harmful either. The
mechanism (CF-gated anti-Hebbian LTD per Albus 1971 §IV.C eq.4) is
implemented correctly (47 unit tests pass, biology probe verifies the
sign of weight changes). It just doesn't produce a measurable benefit
on the cheat-5 benchmark.

## Updated cluster-stacking synthesis

Pre-correction count: 9 attempts past A+E, 5 NEGATIVE.
**Post-correction count: 9 attempts, 4 NEGATIVE (D-with-sleep, AED+v2,
C v2 multi, C v2 single), 4 NEUTRAL (A+D, A+D+E, A+F, A+E+F, A+F v2,
A+E+F v2), 1 PARTIAL (D v2).**

Wait — that's ten variants now. Let me recount: A+D, A+D+E, A+F, A+E+F,
A+F v2, A+E+F v2, A+E+D-with-sleep, A+E+D+v2, A+E+C v2 multi, A+E+C v2
single. Yes 10 stack variants tested past A+E. Result distribution:

- 5 NEGATIVE (A+E+D-with-sleep, A+E+D+v2, A+E+C v2 multi, A+E+C v2 single)
- Wait that's 4. Let me list again carefully:

| Stack | Verdict |
|---|---|
| A+D | NEUTRAL |
| A+D+E | NEUTRAL |
| A+F | NEUTRAL |
| A+E+F | NEUTRAL |
| **A+F v2** | NEUTRAL (corrected from NEGATIVE) |
| **A+E+F v2** | NEUTRAL (corrected from NEGATIVE) |
| A+E+D (sleep at 1350) | NEGATIVE |
| A+E+D+v2 | PARTIAL (mean barely improves) |
| A+E+C v2 (multi) | NEGATIVE |
| A+E+C v2 (single) | NEGATIVE |

7 NEUTRAL/PARTIAL, 3 NEGATIVE. None GO. The cluster-stacking
falsification still holds, just less dramatic than the F v2 corrections
made it look.

## What this implies for the replicated runner

The replicated runner is **broken for any plasticity-sensitive
experiment**. It was used for:

- F v2 tier-3 (corrected today)
- D v2 tier-3 used SINGLE runner (line in dv2_t3_AED_*.cmd.json shows
  `g11_bg_runner.py`, not replicated). So D v2 finding is intact.
- Earlier replicated experiments need audit.

Recommendation: **Do not use the replicated runner for plasticity evals
until this is fixed.** It's safe for action-selection / readout-only
experiments where weight updates aren't the primary lever.

The fix is straightforward in principle: the replicated runner needs
to either (a) leave `cfg.current_reward_signal = delivered_reward`
during the stim window (matching single-runner behavior), or (b) keep
the override set across all 200 stim steps so the per-synapse reward
signal is correctly applied 200× per env step.

Implementing (b) is cleaner since it preserves the per-replica
isolation. Estimated 1-2 hours to implement + tier-2 validation.

Deferred — not implementing now since the corrected F v2 result is
NEUTRAL and the replicated runner won't be used for plasticity work
until fixed.

## Files

- Re-eval results: `research/findings/raw/g11_bg/fv2_revisit_*.json`
- n=1 debug: `research/findings/raw/g11_bg/repl_debug_*.json`
- Original (incorrect) F v2 finding:
  `research/findings/2026-04-30-cluster-f-v2-results.md` ← needs correction note
- Cluster-stacking synthesis:
  `research/findings/2026-04-30-cluster-stacking-synthesis.md` ← needs update

## Next steps

1. Add correction note at top of `2026-04-30-cluster-f-v2-results.md` ✓
2. Update synthesis to reflect corrected counts
3. Update CLAUDE.md to mark F v2 as NEUTRAL not NO-GO
4. (deferred) Fix the replicated runner's reward-modulation timing
