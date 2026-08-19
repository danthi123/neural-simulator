---
type: finding
status: contributing
date: 2026-08-11
mechanism: continual_forgetting_eval (phase-A word-learning -> phase-B, +/- sleep phase) — a RE-SCOPING of the #7 continual-learning crux
lane: continual-learning / scale-up
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/continual/cfe_cupy_nb16_sleep_s42.json
  - research/findings/raw/continual/cfe_*.json
instrument: 24 `continual_forgetting_eval` runs on the 3090 (cupy) sweeping phase_b_events N in {16,24,32,48,64} x {sleep, no-sleep} x seeds x replay-rounds, reading `metrics.retention_pct`.
---

# `continual_forgetting_eval` retention is a flat 100% across N and sleep — it is NOT the #7 plasticity-scale forgetting crux (an honest re-scope, coordinator-analyzed)

The overnight compute (minipc pool + 3090) was pointed at `continual_forgetting_eval` as "the #7 scale-up crux
(frac_recalled~1/N)". Reading the 24 completed runs shows that choice was MIS-SCOPED, and says so honestly.

## Result (`research/findings/raw/continual/cfe_*.json`, 24 runs)

<!--derived-->

`metrics.retention_pct` is **100.0% on every run** — N=16 (n=8), 24 (n=9), 32 (n=2), 48 (n=3), 64 (n=2), sleep AND
no-sleep alike (range 100-100 in every group). `primary_a_acc` and `primary_b_acc` are 1.0. So this eval's phase-A
word-learning is FULLY retained after learning N phase-B facts, with or without the eval's sleep phase — it does not
exhibit the catastrophic forgetting the #7 scale-up is blocked on, and the retention metric is saturated (no
discriminating power for the crux). The one non-saturated signal is `synonym_b_acc` ~0.63 (generalization), not
retention.

## The honest re-scope

<!--derived-->

The #7 plasticity-scale crux is `frac_recalled ~ 1/N` for facts acquired by the **e-prop weight change** (the
plasticity-learned-facts acquisition), NOT the heteroassociative word-learning `continual_forgetting_eval` measures.
Those are different mechanisms: the word-learning retention here is robust to N=64 (a minor positive — that path
does not forget), while the e-prop acquisition forgetting is the actual bottleneck. The right instrument is the
teacher-loop scaling / e-prop-acquisition forgetting runner (the interleaved-generative-replay arc). Compute
redirected accordingly.

**Takeaway:** `continual_forgetting_eval` is banked as "word-learning retention is not the #7 forgetting
bottleneck (100% retention to N=64)"; the #7 crux is measured elsewhere. Caught by reading the substance of the
accrued sweep (the "read the runner's own numbers, do not assume" discipline).
