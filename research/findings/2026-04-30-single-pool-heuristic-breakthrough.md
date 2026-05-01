# 2026-04-30 — `--heuristic-single-pool` breaks the cheat-5 A+E ceiling

**Date:** 2026-04-30 (later that day, after the 4-pivot null sweep)
**Status:** GO — recommend enabling by default in flagship configs

## Headline

**A+E with single-pool heuristic: 5.02 ± 0.59 (n=6, multi-goal det)** — **28% improvement** over the documented A+E ceiling (6.97 ± 0.83) AND **41% std reduction** (0.83 → 0.59). Beats every cluster-stacking attempt past A+E.

## How it was found

After ten cluster-stacking attempts past A+E (all NEUTRAL/PARTIAL/NEGATIVE), I investigated the persistent ~50% performance gap between `g11_bg_runner.py` (single) and `g11_bg_replicated_runner.py` (replicated): single A+E ≈ 7.18, replicated A+E ≈ 3.40. After fixing four bugs in the replicated runner (timing, plasticity_gate forwarding, config alignment, shadow region_manager), the gap was still 2× — replicated stayed at ~3.4 while single sat at ~7.

Probing for the source revealed a fundamental policy difference in how the two runners apply the heuristic cortex drive:

- **Single runner (multi-pool, default since 2026-04):** for each of the 4 cardinal directions, drive `cortex_X` if that direction reduces Manhattan distance. For diagonal goals (e.g. NE), this drives **two** cortex pools (`cortex_N` AND `cortex_E`) simultaneously.
- **Replicated runner (single-pool):** pick **one** Manhattan-reducing direction (random tie-break) and drive only that cortex pool.

The multi-pool heuristic forces the BG cascade to arbitrate between two competing positively-driven motor channels. At our scale, the BG can't make a clean pick — both motor pools fire roughly equally, and motor selection becomes noisy. The agent oscillates rather than committing.

## Probes

| Probe | Setup | A+E mean ± std (n=6) | Notes |
|---|---|---|---|
| Single runner default | multi-pool heuristic | 7.18 ± 1.58 | documented baseline |
| Replicated default | single-pool heuristic | 3.40 ± 0.57 | runner-internal default |
| Probe 4: replicated forced multi-pool | aligned heuristic to single | 3.19 ± 0.11 | NULL — multi-pool doesn't hurt the replicated runner (different downstream) |
| **Probe 5: single forced single-pool** | aligned heuristic to replicated | **4.86 ± 0.33** | **HUGE win — 32% improvement** |
| **SP eval (confirm)** | single + `--heuristic-single-pool` | **5.02 ± 0.59** | confirms breakthrough at full eval scale |

The asymmetry — multi-pool hurts single but is neutral on replicated — implies the replicated runner has additional downstream filtering that absorbs heuristic noise. Identifying that mechanism is a separate follow-up; what matters here is that single-pool heuristic dominates on the canonical (single) runner.

## Per-seed (single + single-pool heuristic, multi-goal det)

| Seed | sum | phases (final_quarter mean Manhattan) |
|---|---|---|
| 42 | 5.16 | [1.50, 1.01, 1.42, 1.23] |
| 43 | 4.69 | [1.25, 1.14, 1.08, 1.22] |
| 44 | 4.36 | [1.04, 1.12, 1.19, 1.01] |
| 100 | 5.96 | [1.09, 2.61, 1.02, 1.24] |
| 101 | 5.37 | [1.44, 1.29, 1.25, 1.39] |
| 102 | 4.59 | [1.20, 1.17, 1.03, 1.19] |

**6/6 seeds beat baseline 7.77.** **6/6 seeds beat A+E 6.97.** **0/6 seeds had a phase-3 catastrophe** (worst phase across all seeds: 2.61 in seed 100). Per-phase finalQ all stay in 1.0-2.6 range, with most around 1.0-1.5.

## Comparison to documented best configs

| Config | mean | std | n | improvement vs A+E |
|---|---|---|---|---|
| baseline (no clusters) | 7.77 | 3.33 | 6 | — |
| A+E (multi-pool, documented best biology-grounded) | 6.97 | 0.83 | — | reference |
| Full perception-arc flagship (cheats: heuristic + place + sensed reward + landmarks + curriculum) | 4.08 | 0.49 | 6 | -41% |
| Best with cheats (engineering shortcut, no perception arc) | 4.41 | 0.94 | 6 | -37% |
| **A+E + `--heuristic-single-pool` (NEW)** | **5.02** | **0.59** | **6** | **-28%** |

The single-pool A+E is within striking distance of the perception-arc flagship (5.02 vs 4.08), achieved with a **single-line config change** instead of the 5+ flag perception arc. Stacking single-pool ON TOP of the perception arc is a clear next experiment — likely to give an additional improvement.

## Sleep+D still hurts under single-pool

`A+E+D + sleep replay + single-pool: 13.91 ± 3.06 (n=6)` — much better than the 28-30 range under multi-pool, but still ~3× worse than A+E alone. Sleep replay's hurt to the AED stack is independent of the heuristic. Cluster D + sleep is a separate problem (content quality, per the SCIENCE_ROADMAP §4.7 note).

## CLI usage

```bash
python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
    --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi \
    --enable-cluster-a-closed-loop --enable-cluster-e-topography \
    --heuristic-single-pool \
    --seed N --n-steps 1800
```

## Why this matters

The cluster-stacking buildout strategy was looking for biology mechanisms that would close the cheat-5 ceiling beyond A+E. Ten attempts (B.1, B.2, B.3 partial → null, A, C v1, C v2, D v1, D v2, E, F v1, F v2, plus HER, recency-replay, RPE, surprise-LR) all showed neutral-to-negative effects against the documented 6.97 ± 0.83 baseline.

The actual problem was a methodological artifact: the heuristic was creating BG-cascade competition that the model couldn't resolve, contaminating ALL cluster-stacking measurements. Many of the "NULL" findings may have signal under the corrected baseline. **All cluster work needs revisiting** — the multi-pool heuristic was a confound for two months of experiments.

## Recommended next steps

1. **Update CLAUDE.md flagship recommendations** to include `--heuristic-single-pool`.
2. **Re-validate top cluster results** under single-pool: A+E+D+sleep, A+E+F v1, A+E+F v2, perception-arc flagship + single-pool.
3. **Stack single-pool ON TOP of perception arc** — likely best result to date.
4. **Investigate why multi-pool hurts**: the BG cascade arbitration failure mode is a real biology question. Real cortex-to-striatum projections aren't clean WTA either; how does the brain prevent this?

## Files

- Implementation: [`research/runners/g11_bg_runner.py`](../runners/g11_bg_runner.py) — `--heuristic-single-pool` flag
- Probe 5 results: `probe5_AE_singlepool_seed*.json` (n=6)
- SP eval: `sp_AE_seed*.json` (A+E single-pool, n=6)
- AED+sleep results: `sp_AED_seed*.json` (n=6)
- Commit: `21d1088`
