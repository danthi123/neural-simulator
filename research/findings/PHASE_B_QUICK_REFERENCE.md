# Phase B Configuration Quick Reference

Quick lookup for the recommended Phase B BG cascade configuration based on task type. Generated from autonomous overnight session 2026-04-26 + 6-seed corrigendum.

## TL;DR

The Phase B BG cascade architecture (validated 2026-04-25) resolves the silent-motor trap structurally. On top of that, opt-in refinements give modest improvements — but watch out for high seed-variance.

**Important calibration**: The asymmetric adaptive DA "win" (sum=3.53) on seeds 42-44 did NOT generalize to seeds 100-102 (mean 6.94). On a 6-seed pool, asym DA is only 11% better than baseline (within noise). **LR boost is more reliable** — 16% better with lower variance. See [`2026-04-26-six-seed-correction.md`](2026-04-26-six-seed-correction.md).

## Recommended configurations

### Default (no flags) — robust across all task types
```bash
python -m research.runners.g11_bg_runner --moving-goal --seed N --n-steps 1800
```
- Phase B BG cascade with broadcast DA
- Phase 1 finalQ 1.76 (vs G9 baseline 6.74) — 74% improvement
- Multi-goal sum 8.32

### Recommended for general use — most robust
```bash
python -m research.runners.g11_bg_runner --moving-goal --surprise-lr-boost \
    --seed N --n-steps 1800
```
- NE-like surprise amplification of `reward_learning_rate`
- 6-seed mean 4.92 ± 1.07 (16% improvement over baseline 5.88, t=1.31)
- Lower variance than baseline AND than asym DA — most reliable

### Conditional / experimental: asymmetric adaptive DA
```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --seed N --n-steps 1800
```
- Asymmetric adaptive DA (slow positive 0.9, fast negative 0.7 EMA decay)
- 6-seed mean 5.23 ± 1.90 (only 11% improvement, t=0.64, NOT significant)
- **WARNING**: SEED-DEPENDENT. Worked on seeds 42-44 (mean 3.53) but failed on seeds 100-102 (mean 6.94)
- Mechanism is biologically plausible but on-task variance too high to be a reliable recommendation

### Multi-goal (4-corner cycle stress test)
```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi \
    --seed N --n-steps 1800
```
- Default broadcast DA + 4-corner schedule
- Sum 8.32 (best on this task variant)

## All available opt-in flags

| Flag | Purpose | Recommended? |
|---|---|---|
| `--motor-lateral-inhibition` | FS interneuron WTA microcircuit | NO (locks in too much) |
| `--per-action-da` | Hard eligibility-trace gating | NO (use --adaptive-da instead) |
| `--adaptive-da` | Reward-EMA-gated DA targeting (symmetric) | useful for slow-change |
| `--adaptive-da-ema-decay {N}` | Tune EMA decay (default 0.9 = tau~10) | leave default |
| `--adaptive-da-ema-decay-negative {N}` | Asymmetric ramp (recommended: 0.7) | YES with --adaptive-da |
| `--da-gated-wta` | Scale FS→motor weights by gating_strength | NO (still net negative) |
| `--learned-perception` | Plastic sensory→cortex (replaces heuristic) | NO (cold-start fails) |
| `--rpe-scaled-reward` | Amplify reward by RPE | NO (use --surprise-lr-boost) |
| `--rpe-alpha {N}` | RPE scaling magnitude (default 1.0) | leave default |
| `--surprise-lr-boost` | Boost learning rate on surprise | YES for general use |
| `--surprise-lr-alpha {N}` | LR boost magnitude (default 2.0) | leave default |
| `--goal-schedule {default,multi}` | Task variant | task-dependent |

## Full landscape (3-seed averages)

```
                              2-goal sum    multi-goal sum
baseline (no flags)              5.24           8.32   ← always works
asymmetric adaptive DA           3.53 ★        9.97
surprise-boosted LR              4.02           9.11   ← most robust
adaptive DA (sym tau~10)         3.99           —
adaptive DA (sym tau~3)          4.33           —
hard per-action DA               4.65           —
WTA + asymmetric adaptive DA     4.29           —
WTA + adaptive DA                4.41           —
DA-gated WTA + asym DA           4.54           —
WTA only                         4.86           —
RPE-scaled reward only           —              9.62
asym DA + RPE                    —              9.49
LR boost + asym DA combo         4.07           —
learned perception (cold start)  10.85          —     ← random walk
```

★ = best on that task variant

## Findings docs (chronological, 2026-04-26)

For detailed analysis of each variant:

1. `2026-04-26-wta-lateral-inhibition-mixed.md` — WTA partial GO
2. `2026-04-26-per-action-da-mixed.md` — hard DA partial GO
3. `2026-04-26-adaptive-da-targeting.md` — first sum win (3.99)
4. `2026-04-26-asymmetric-adaptive-da.md` — slow-change winner (3.53)
5. `2026-04-26-da-gated-wta.md` — DA-gated WTA NEGATIVE
6. `2026-04-26-learned-perception-cold-start-fail.md` — perception NEGATIVE
7. `2026-04-26-multi-goal-stress-test.md` — asym DA reverses on fast-change
8. `2026-04-26-surprise-lr-boost.md` — most robust variant
9. `2026-04-26-night-summary.md` — overall session overview
10. `PHASE_B_QUICK_REFERENCE.md` — this file

## Architecture changes (all opt-in)

- `sim/regions.py`: `BrainRegion`, `RegionPathway` (unchanged from Phase A)
- `sim/neuromodulators.py`: added `from_surprise` production rule (NE-like RPE phasic firing)
- `research/runners/g11_bg_runner.py`: 8 opt-in flags + builder kwargs
- `tests/test_g11_bg_runner_flags.py`: 12 smoke tests (added)
- `tests/test_neuromodulators.py`: 3 from_surprise tests added (39 total pass)

## Future directions (not pursued autonomously, recommended for next session)

1. **True NE concentration via NeuromodulatorConfig** using the new `from_surprise` rule — declarative config replaces runner-local LR boost
2. **Hybrid heuristic + learned perception** — keep heuristic cortex drive as base, layer plastic refinement on top
3. **Curriculum learning** for sensory→cortex (warm-up on fixed-goal first)
4. **5-HT slow-timescale modulation** — untried
5. **Different task class entirely** — sequential decision, multi-modal sensing, working memory

All 19 commits from the night pushed to main on https://github.com/danthi123/neural-simulator.
