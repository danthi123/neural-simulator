# 2026-04-29 — Step 3 results: A+E + sensed-reward (n=6)

**Run:** `g11_bg_runner.py` deterministic single-goal-transition (2-phase, 300 + 1500 stable steps), Cluster A (closed BG loop) + Cluster E (topographic maps) + `--sensed-reward` flag, 6 seeds (42, 43, 44, 100, 101, 102), `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## Headline

**3.65 ± 1.47 (n=6)** — essentially equivalent to **A+E without sensed-reward (3.31 ± 0.74, n=6)** documented in `2026-04-29-overnight-FINAL.md`. **Closing cheat #1 (distance-based reward) does NOT degrade A+E architectural win.**

| Condition | Mean | Std | n | Notes |
|---|---|---|---|---|
| **A+E + sensed-reward** (THIS run) | **3.65** | 1.47 | 6 | Step 3 |
| A+E (no sensed) | 3.31 | 0.74 | 6 | overnight FINAL |
| A+E (no sensed) | 3.93 | 1.55 | 12 | tier-4 expanded |
| baseline (no A+E, no sensed) | 4.35 | 2.16 | 6 | overnight FINAL |
| Documented full flagship (cheats allowed) | 4.08 | 0.49 | 6 | 2026-04-27 |

Welch's t between A+E and A+E+sensed is ~0.46 (NOT significant) — sensed-reward + A+E performs the same as A+E alone within noise.

## Per-seed breakdown

| Seed | Sum | Phase 0 finalQ | Phase 1 finalQ |
|---|---|---|---|
| 42 | 3.78 | 2.64 | 1.14 |
| 43 | 6.39 | 3.72 | 2.67 |
| 44 | 2.54 | 0.97 | 1.57 |
| 100 | 2.58 | 1.08 | 1.50 |
| 101 | 3.87 | 2.11 | 1.77 |
| 102 | 2.71 | 0.87 | 1.85 |

**Mean phase 0 (initial goal acquisition): 1.90**
**Mean phase 1 (after goal change at step 300): 1.75** ← improvement, agent IS readapting

Compare to baseline (no A+E, no sensed) phase distribution (overnight FINAL):
- Phase 0: 1.60
- Phase 1: 1.68

A+E+sensed phase 0 is ~19% worse (1.90 vs 1.60) but phase 1 is comparable (1.75 vs 1.68). Sensed-reward might add a small phase-0 cost (agent has to learn what the beacon means) that's recovered by phase 1.

Phase 1 < phase 0 means agent IS readapting after the goal change, which is the key behavioral signature.

## What this confirms

1. **Sensed-reward closure (cheat #1) is achievable without A+E regression.** The flagship config can drop the distance-based reward and use beacon-intensity gradient without losing the A+E architectural improvement.
2. **A+E architectural win is robust to perception arc additions.** Earlier 2026-04-27 night results showed the perception arc (sensed-reward + beacon-perception + cue-reflex + landmarks) sums to 4.56 ± 0.70 (over baseline 5.88, p=0.00819). Adding A+E on top of that should improve further.

## Open questions

- **Does A+E + full perception arc (sensed-reward + beacon + cue-reflex + landmarks) compose successfully?** Step 3 only adds sensed-reward; the full perception arc is more cuts. Worth a follow-up run.
- **Phase 0 cost of sensed-reward (1.90 vs 1.60 baseline phase 0):** is this a curriculum issue (agent needs more steps to learn beacon mapping under sensed-only) or a structural issue (no shaping signal)?

## What's next

Step 4 (multi-goal deterministic, 4 phases, n=12 baseline + n=12 A+E) is in progress and will land in ~30-60 min. That tests whether A+E composes with the harder benchmark where the documented "4.08 single-goal" was earlier.

## Provenance

- Code SHA at run time: `4ec7486` (post structural-naming-audit commit, pre Tier-1 prose pass commit `8aa2fcb`).
- Wall-clock: ~30 min per seed × 6 seeds in parallel; ~30 min total.
- All 6 result JSONs at `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_step3_AE_sensed.json`.
