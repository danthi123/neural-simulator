# 2026-05-01 — Cluster G v2.5: scales perfectly to 16×16 grid

**Context:** After G v2.5 (`--enable-pfc-nmda` with NMDA on PFC + cortex_X +
motor_X) achieved 2.00 ± 0.00 on cheat-5 multi-goal det (8×8), tested
whether the architecture generalizes to a larger gridworld.

## Headline

**G v2.5 on 16×16 grid: 2.00 ± 0.00 (n=3 seeds)** — identical sum-of-final-quarter-mean-distance to the 8×8 result, on a grid with **4× the cell count** and **2× the maximum Manhattan distance**.

| Seed | Phase 0 finalQ | Phase 1 finalQ | Phase 2 finalQ | Phase 3 finalQ | Sum | Mean dist (overall) |
|---|---|---|---|---|---|---|
| 42 | 0.50 | 0.50 | 0.50 | 0.50 | **2.00** | 0.697 |
| 43 | 0.50 | 0.50 | 0.50 | 0.50 | **2.00** | 0.705 |
| 44 | 0.50 | 0.50 | 0.50 | 0.50 | **2.00** | 0.689 |

The 0.50 final-quarter pattern is the same goal-cell oscillation
observed at 8×8: agent reaches the goal then bounces between
(gx, gy) and one adjacent cell. 56 / 113 ≈ 0.4956 vs 57 / 113 ≈ 0.5044.

## Why this matters

16×16 was expected to be substantially harder than 8×8:
- 4× more cells
- 2× larger maximum Manhattan distance
- Heuristic single-pool drives are unchanged in strength but must
  carry the agent further

Yet the agent still:
- Reaches every goal corner ((14,14), (1,14), (1,1), (14,1)) within
  the 450-step phase budget
- Maintains the goal lock once acquired
- Reproduces the same final-quarter oscillation across 3 seeds

This argues that the G v2.5 architecture (closed BG loop A + topographic
cortex E + cortex+motor+PFC NMDA) is **not specifically tuned to the 8×8
geometry** — the heuristic + cascade dynamics scale with grid size.

## Caveats

1. **Heuristic still drives cortex**. The 16×16 robustness is partly the
   heuristic's robustness (it always picks the right cardinal direction).
   The architecture is excellent at *exploiting* the heuristic — but
   testing pure-perception scenarios (no heuristic) on 16×16 is the
   real test.
2. **Same goal schedule as 8×8** — 4 phases × 450 steps = 1800 steps
   total. With grid_size scaling but n_steps unchanged, the agent has
   proportionally less time per cell.
3. **Single phase budget**: only validated at n_steps=3000 (with 4×450
   schedule). Should test larger n_steps to confirm phase-end determinism
   isn't a phase-truncation artifact.

## Frontend implication

The webapp world viewer rendered 16×16 grids at fixed 56px/cell, so the
canvas was 944×944 → overflowed the layout. Fixed in commit a5ae7d1:
cellPx now scales inversely with gridSize so the canvas stays
~512×512 for any grid count, with all icons (agent, goal, halo, trail,
landmark) scaled proportionally. Verified visually with stress_16x16_seed42.

## Files

- Stress runs: `research/findings/raw/g11_bg/stress_16x16_seed{42,43,44}.json`
- Frontend fix: commit `a5ae7d1` (`webapp/static/world.js`)
- G v2.5 design: see `2026-05-01-cluster-g-nmda-breakthrough.md`

## Next steps

1. **Larger grids** — 24×24 or 32×32 to find where the heuristic + cascade
   geometry breaks down.
2. **No-heuristic 16×16** — drop `--heuristic-single-pool`, see how
   pure perception copes at scale. Will likely reveal where learned
   perception plateau vs. heuristic occurs.
3. **Goal-schedule difficulty** — same 4-phase schedule but with goals
   placed at random reachable positions (not fixed corners) on 16×16.
   Tests memory / generalization rather than corner-finding.
