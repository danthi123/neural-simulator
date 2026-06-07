# Perceptual-bootstrap gauge (b): the visual cortex does NOT bootstrap navigation from reward+exploration in a reasonable budget — BLOCKED; (a) needs a stronger perceptual front-end (goal localization), now precisely targeted — 2026-06-06

**Status:** GAUGE BLOCKED (the owner's (b) step of "(b) gauge → (a) if tractable"). Decisive: even with explore-on-losing,
the heuristic-free visual-cortex agent does not learn to reach the goal over 6000 steps. The owner chose to pursue (a)
(the targeted perceptual front-end) on this diagnosis. NO `sim/` edits; the only runner change is the additive
`--goal-schedule single` (one fixed goal, the cold-start test).

## The gauge question

The N1 boundary (`2026-06-06-N1-heuristic-removal-BOUNDARY.md`) found the agent can't navigate without the action
heuristic because the visual cortex's `IT→cortex_X` pathway cold-starts (the heuristic was its implicit teacher). The
gauge: can that pathway be bootstrapped the BIOLOGICAL way — reward-modulated cortico-striatal plasticity + exploration
(the agent reaches the goal, dopamine teaches "see-goal-in-direction-X → move-X") — WITHOUT the heuristic-teacher?

## The decisive table (grid-8, single fixed goal, 6000 steps, seed 42; distance to goal, LOWER better, goal = 0)

| condition | trajectory (mean dist, 6 time-bins) | final-Q | % steps at goal |
|---|---|---|---|
| heuristic-OFF + visual cortex, no explore | [4.81, 5.91, 5.57, 5.63, 4.75, 5.34] | 5.27 | 0.5% |
| heuristic-OFF + visual cortex, **+ explore-on-losing** (`--adaptive-da`) | [5.24, 5.25, 5.75, 5.63, 5.10, 4.48] | 4.89 | **0.5%** |
| heuristic-ON (the target) | ~1.0 throughout | 1.02 | 28.6% |

**Flat at ~5 cells from the goal across all 6000 steps, with or without exploration — no learning trend, and the agent
reaches the goal only 0.5% of the time** (vs 28.6% for the heuristic). (The earlier 300-step smoke's encouraging 10→3
was just the initial drift-in from the corner, not learning.)

## A test-rigor note (caught + corrected)

The first long run lacked exploration, and this runner has no motor-exploration mechanism (only "random action if all
motors are silent," which never fires because the weak visual pull keeps motors active). That created a chicken-and-egg
(the agent stalls ~5 cells out, almost never reaches the goal, so earns almost no teaching reward). To avoid a false
BLOCKED, the gauge was re-run with `--adaptive-da` (explore-on-losing — relaxes the action competition when the agent
isn't reaching the goal). It changed essentially nothing (final-Q 5.27 → 4.89, at-goal still 0.5%). So the chicken-and-egg
was NOT the cause — the BLOCKED is fair.

## The precise diagnosis (what BLOCKED actually means + what (a) needs)

The visual cortex gives the agent a **weak attractive pull** — enough to drift in from the far corner to ~5 cells out —
but **not precise goal localization**: it can sense "the goal is roughly over there" but never pinpoint it well enough
to reach it, and reward+exploration over 6000 steps does not sharpen that. So genuine perception-driven navigation
needs a **stronger perceptual front-end**, not more training:
- **sharper goal localization in IT** (a dedicated goal-cell / salient-object detector that precisely localizes the
  goal in the visual field), OR
- an **innate salient-goal-approach reflex** (a tectum / superior-colliculus orienting response toward the salient
  goal — more biological than the coordinate heuristic) that scaffolds the agent to the goal and provides the teaching
  signal, which the learned `IT→cortex` mapping then refines, OR
- a **critical-period developmental scaffold** — the heuristic (or the orienting reflex) teaches the `IT→cortex` mapping
  during an early period, then is WEANED OFF, leaving the deployed agent navigating from genuinely-learned perception
  (innate scaffolding bootstrapping a learned circuit, then fading — standard developmental biology).

## Net + the owner decision

- **N8 ✅ + N6 ✅** are genuinely biologized (real GPi→thal disinhibition + a real spiking accumulate-then-commit
  decision, beating/≈ the cheats) — those wins stand.
- The **perception layer (N1/N2/N7)** is a characterized boundary: the agent senses the goal's rough direction but
  cannot localize it precisely enough to navigate without a coordinate crutch; the cold-start is real and reward+
  exploration does not bootstrap it in a reasonable budget.
- **Owner chose (a):** pursue the targeted perceptual front-end. First de-risk: the critical-period developmental
  scaffold (heuristic teaches `IT→cortex` during a critical period, then weaned to zero — does the agent navigate from
  the learned perception after weaning?), as the most tractable, biology-grounded test of whether the mapping CAN learn
  precise navigation given a good teacher, before investing in a sharper-IT front-end.

## Artifacts
- `research/findings/raw/_pbgauge_long_heuroff_s42.json` (no-explore), `_pbgauge_long_heuroff_explore_s42.json`
  (+explore), `_pbgauge_long_heuron_s42.json` (control), `_pbgauge_smoke_s42.json` (the 300-step smoke).
- Runner: `--goal-schedule single` (additive, the cold-start test).
- Prior: `2026-06-06-N1-heuristic-removal-BOUNDARY.md`, `2026-06-06-navigation-cheat-audit-and-conversion-plan.md`.
