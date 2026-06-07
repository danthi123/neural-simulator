# N1 critical-period developmental scaffold = TRACTABLE — the heuristic teaches the visual-cortex navigation mapping, then is WEANED to zero, and the deployed agent KEEPS navigating from learned perception (no heuristic). The deepest nav cheat is removable. — 2026-06-06

**Status:** TRACTABLE-IN-PRINCIPLE but FIXED-CONFIG-FRAGILE — **CORRECTED after multi-seed** (the single-seed-42
TRACTABLE result below is real for that seed/config but does NOT generalize at a fixed critical-period length). NO
`sim/` edits; additive default-off wean flags.

> **⚠️ MULTI-SEED CORRECTION (supersedes the single-seed TRACTABLE headline):** the post-wean hold is
> SEED-AND-CONFIG-DEPENDENT and NON-MONOTONIC — **no fixed critical-period length is robust across seeds.**
> Post-wean mean distance (heuristic fully OFF; HOLD ~1-2, COLLAPSE ~5-6):
> | seed | 3000-step crit | 5000-step crit |
> |---|---|---|
> | 42 | **2.14 ✓ holds** | 6.16 ✗ collapses |
> | 43 | 1.69 ✓ holds | 1.68 ✓ holds |
> | 44 | 6.03 ✗ collapses | **1.63 ✓ holds** |
>
> Every seed biologizes at SOME critical-period length (the MECHANISM is real — the visual cortex CAN learn
> self-sufficient navigation), but the sweet-spot LENGTH is seed-dependent and **more teaching can HURT** (the longer
> crit fixed seed 44 but BROKE seed 42 — whack-a-mole). A FIXED critical-period CLOCK is not robust. **The
> biologically-correct fix is ADAPTIVE / activity-gated weaning** — real critical periods close when the circuit is
> READY (neuromodulator / activity-dependent gating), NOT at a fixed age: wean when the learned `IT→cortex` mapping is
> consolidated (measured online — e.g. the agent's recent at-goal rate or the readout-pathway weight magnitude), not
> at a fixed step. That (or a sharper-IT goal-localization front-end) is the path to ROBUST N1 biologization. **Net:
> N1 biologization is demonstrably POSSIBLE (every seed does it at its own sweet spot) but not yet ROBUST at one
> recipe; robust = adaptive weaning, owner decision below.**

---
*(original single-seed write-up follows — accurate for seed 42 @ 3000-crit, but read it through the correction above)*

## The one-line result

The N1 boundary + the perceptual-bootstrap gauge established that the visual cortex's `IT→cortex_X` navigation
pathway COLD-STARTS without the heuristic (the heuristic was its implicit teacher; reward+exploration alone does not
bootstrap it — flat at the ~5-cell floor). This de-risk tested the most tractable biological fix: a CRITICAL-PERIOD
DEVELOPMENTAL SCAFFOLD — let the heuristic TEACH the mapping during an early critical period, then WEAN it to zero,
and test whether the agent navigates from the LEARNED perception after the heuristic is gone. **It does. With the
heuristic completely weaned off, the agent holds at ~2 cells from the goal and stays there — it does NOT collapse
back to the ~5 cold-start floor. The learned perception is self-sufficient. The deepest nav cheat (the action
heuristic) is removable.**

## The decisive table (grid-8, single fixed goal, 9000 steps, seed 42; distance to goal, LOWER better; goal = 0)

Wean schedule: heuristic FULL strength for steps < 3000 (the critical period / teaching), linear fade over
3000–4500, then OFF (= 0) for the rest.

| window | mean distance | % steps at goal |
|---|---|---|
| pre-wean (0–3000, heuristic teaching) | 0.98 | 28.9% |
| during-wean (3000–4500, fading) | 2.81 | 8.0% |
| **POST-WEAN early (4500–6000, heuristic OFF)** | **2.14** | 4.1% |
| **POST-WEAN mid (6000–7500, OFF)** | **2.30** | 3.7% |
| **POST-WEAN late (7500–9000, OFF)** | **1.98** | 4.1% |

Reference points: heuristic-ON-throughout ~1.0 (28.6% at-goal); heuristic-OFF-throughout cold-start floor ~5
(0.5% at-goal). **Post-wean sits at ~2 and HOLDS (early 2.14 → mid 2.30 → late 1.98 — no upward drift toward the
floor), ~2.5× better than the cold-start floor and ~8× the at-goal rate.** The learned `IT→cortex` mapping survives
the removal of its teacher.

## What this means (and the honest scope)

- **The critical-period developmental scaffold biologizes N1.** The deployed, weaned agent navigates from
  genuinely-learned visual perception with NO heuristic. Biologically this is standard development: an innate
  scaffold (here, the heuristic-as-orienting-teacher) bootstraps a learned circuit, then fades — and it
  retroactively legitimizes the heuristic's role (a developmental teacher that's removed, not a permanent crutch).
- **It is realistic, not heuristic-equal.** Post-wean ~2 vs the heuristic's ~1: the learned perception is coarser —
  the agent navigates to within ~2 cells and hovers there (reaching the goal ~4% of steps vs the heuristic's 29%).
  Per the owner-relaxed gate (realistic, not unreasonably slow), ~2 clears the bar — the agent genuinely navigates
  to the goal vicinity from perception alone, far above the ~5 cold-start floor. A sharper-IT goal-localization
  front-end could tighten ~2 → ~1 (a future refinement, not required for "biologized").
- **Single-seed (42); multi-seed (43/44) confirmation IN FLIGHT** (the across-wean hold must be seed-robust before
  the claim is firm — appended below).
- **N1 biologized; N2/N7 are the remaining perception residuals** — the goal is still a rendered stimulus the
  visual cortex sees (N2) and the V1 Gabors are pre-initialized (N7). But the agent now LEARNED to perceive and
  navigate to the goal (the IT→cortex mapping is learned, scaffolded), so the "free goal detection + coordinate
  injection" core of the cheat is gone.

## Net for the nav arc

This is the unlock for the deepest navigation cheat. **N8 ✅ (disinhibition) + N6 ✅ (spiking decision) + N1 ✅
(removable via the developmental scaffold — pending multi-seed)** — the basal-ganglia output stage AND the action
decision AND the action *selection* are now biologized, and the agent navigates from learned perception rather than
a hand-coded coordinate rule. The remaining cheats (N2 goal-rendering, N7 Gabor pre-init, N5 reward, N9 SNc) are
lesser / characterized.

## Production config (the scaffold)
```
... --heuristic-single-pool --heuristic-wean-start 3000 --heuristic-wean-steps 1500 \
    --genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300 --genuine-thal-tonic-pa 750 \
    --readout-source spiking_wta --urgency-max-pa 180 \
    --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --goal-schedule single
```
(The deployed agent, after step ~4500, runs with the heuristic OFF and navigates from learned perception.)

## Artifacts
- `research/findings/raw/_n1_scaffold_wean_s42.json` (the 9000-step across-wean run); `_s43`/`_s44` (multi-seed, in flight).
- Runner: `--heuristic-wean-start` / `--heuristic-wean-steps` (additive, default-off, no `sim/` edit).
- Prior: `2026-06-06-perceptual-bootstrap-gauge-BLOCKED.md`, `2026-06-06-N1-heuristic-removal-BOUNDARY.md`.
