---
type: plan
status: live
date: 2026-05-02
---

# Post-distributed-pop experiment runbook

Sequential experiment commands ready to fire when dpop result is in
(~23:00 today). Each scenario has a different first experiment.

## Scenario A: dpop W→A ≥ 35% (BREAKTHROUGH)

Distributed motor pool actually beats the architectural ceiling.
Validate at multiple seeds:

```bash
# Seeds 43, 44, 100, 101, 102 sequentially (~75 min each)
for seed in 43 44 100 101 102; do
    python -m research.runners.text_eval_embodied \
        --n-episodes 100 --steps-per-episode 30 --seed $seed \
        --stim-steps-per-step 200 --reset-steps 100 \
        --enable-distributed-motor-pop \
        --out-stats research/findings/raw/g11_bg/text_eval_dpop_seed${seed}.json
done
```

After all 6 seeds done, run meta-analysis:
```bash
python -m research.runners.text_io_meta_analysis
```

Expected if distributed-pop is real win:
- 6-seed cumulative W→A > 30% with p < 0.01
- Per-direction means > +0.05 across all 4

## Scenario B: dpop W→A 28-35% (within v2 range)

Distributed-pop neither hurts nor helps. Pivot to SWR Phase 3 to test
if consolidation improves on top of v2 baseline:

```bash
python -m research.runners.text_train_curriculum \
    --phase1-episodes 200 --phase2-episodes 100 \
    --phase3-replays 500 \
    --seed 42 \
    --out-stats research/findings/raw/g11_bg/text_eval_curriculum_swr_seed42.json
```

500 replays = ~5x phase 2's plastic events at compressed time. Tests
if consolidation reinforces correct (token, action) pairings beyond
phase 2 STDP alone. ~140 min total.

## Scenario C: dpop W→A < 28% (NEGATIVE)

Distributed-pop hurt. The 28.5% v2 baseline is genuinely the ceiling
under all biology-grounded variations tested. Run SWR consolidation
on v2 baseline (no curriculum, just rehearsal post-training):

Actually the curriculum infrastructure doesn't easily support
"v2 + Phase 3" without phase 1. Use a modified launch:

```bash
# Phase 1 = 0 (skip visuomotor pretraining), Phase 2 = 100 ep, Phase 3 = 500
python -m research.runners.text_train_curriculum \
    --phase1-episodes 0 --phase2-episodes 100 \
    --phase3-replays 500 \
    --seed 42 \
    --out-stats research/findings/raw/g11_bg/text_eval_v2_swr_seed42.json
```

Wait — phase1=0 might break something. Let me check the runner...
Actually `_run_navigation_loop` with `n_episodes=0` should just return
0,0 and skip. Should be fine.

If still problematic, modify to add a `--swr-only` mode that skips
phase 1 entirely.

## Scenario D: BOTH dpop and SWR look promising (ambitious)

Run combined: distributed-pop architecture + SWR consolidation.
Strong test of all the biology-grounded improvements.

```bash
python -m research.runners.text_train_curriculum \
    --phase1-episodes 200 --phase2-episodes 100 \
    --phase3-replays 500 \
    --enable-distributed-motor-pop \
    --seed 42 \
    --out-stats research/findings/raw/g11_bg/text_eval_dpop_swr_seed42.json
```

This composes both. ~145 min. Best chance of beating 28.5%.

## Decision flowchart

```
dpop seed=42 result lands (~23:00)
  |
  ├── W→A ≥ 35% → Scenario A (multi-seed validation, ~6 hr)
  ├── W→A 28-35% → Scenario B (SWR + curriculum, ~140 min)
  ├── W→A < 28% → Scenario C (SWR + v2 only, ~120 min)
  └── (Bonus) → Scenario D (dpop + SWR composed, ~145 min)
```

## Rationale

- **A is the "win" path:** if dpop works, validate it.
- **B is the "compose" path:** if dpop is neutral, layer SWR on top.
- **C is the "fallback" path:** if dpop hurts, SWR is the next mechanism.
- **D is the "everything" test:** if either positive, try composing.

All four scenarios are biology-grounded (Pulvermüller distributed
coding + Wilson-McNaughton SWR consolidation). At least one should
fire immediately on dpop result.

## Post-experiment

After whichever scenario runs, re-run meta-analysis:
```bash
python -m research.runners.text_io_meta_analysis
```

And weight diagnostic on the final checkpoint:
```bash
python -m research.runners.text_weight_diagnostic <checkpoint.h5> --out <diag.json>
```

Compare to v2 weights via:
```bash
python -m research.runners.text_weight_compare \
    v2:<v2_diag> \
    new:<new_diag>
```
