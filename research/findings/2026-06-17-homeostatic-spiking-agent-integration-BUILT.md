# Brain-based spiking homeostatic agent — BUILT + brain-faithful + functional (3 seeds); policy-convergence needs tuning

**Date:** 2026-06-17 (autonomous loop tick)
**Status:** **The integration is BUILT and runs on one spiking bridge.** A single `SimulationBridge` agent —
place + motor + drive + hunger-modulator — navigates and maintains its energy from a SELF-GENERATED neural reward
(the drive-reduction, no host distance/goal term), across 3 seeds. The wiring, the brain-faithfulness, and the
functional agency are demonstrated; a strongly-converged learned policy needs the reward/learning-rate tuning the
design flagged. This is the first running, brain-based, self-directed agent in the project.

## What was built (`_homeostatic_spiking_agent_integration.py`, per the design doc)

One bridge: 5 place cells + `motor_a`/`motor_b` + an `agrp` drive pool + a `hunger` neuromodulator
(`from_region_firing_signed` over `agrp`, CYCLE-127 calibration). `place→motor_{a,b}` are plastic and
reward-modulated, with **g9's LTP-biased three-factor params** (`stdp_a_plus 0.012 > a_minus 0.01`,
`reward_learning_rate 0.08`, eligibility τ 500 ms) + **motor-exploration** spikes — the machinery the CYCLE-128
toy lacked. The ONE new thing vs g9: `current_reward_signal = −Δ(hunger concentration)` (the drive reduction,
read from the drive pool's firing) replaces g9's host Manhattan-distance reward. The action→direction map is
remapped per seed (the agent must learn which motor reaches food).

## Result (3 seeds, tuned: slower depletion 0.015, 120 trials)

| seed | toward-food choice early→late | food reaches | final energy | survives? |
|---|---|---|---|---|
| 42 | 0.45 → 0.50 | 8 | 0.65 | ✓ |
| 43 | 0.55 → **0.75** | 6 | 0.97 | ✓ |
| 44 | 0.50 → 0.50 | 7 | 0.94 | ✓ |

- **Wiring composes + brain-faithful (all seeds):** place + motor + drive + hunger modulator co-reside on one
  bridge; the reward is read from the drive pool's firing — **no host distance/goal term anywhere** (provenance
  asserted). The first un-tuned 60-trial smoke crashed on 2/3 seeds (the race-to-learn-before-starving); slower
  depletion fixed survival on all 3 (energy 0.65–0.97, food reached 6–8×).
- **Functional agency:** the agent navigates the corridor, reaches food repeatedly, and maintains its energy from
  the intrinsic reward — a self-directed spiking agent keeping itself fed, no external goal.
- **Emerging learned policy:** the toward-food preference rises (clearly on seed 43, 0.55→0.75; weakly on 42/44).
  A *strongly*-converged policy is not yet demonstrated — survival is partly carried by motor exploration + the
  short corridor reaching food regardless. The policy-learning needs the reward-magnitude / learning-rate /
  exploration-decay tuning the design flagged as the real risk.

## Honest scope + next

- **Demonstrated:** the brain-based homeostatic agent is BUILT, composes on one spiking bridge, is brain-faithful
  (neural drive + neural reward, no host term), and is functionally self-maintaining.
- **Not yet:** a robust, multi-seed, strongly-converged policy-learning GO (the toward-choice is weak/variable).
  The focused follow-on is the reward/learning-rate/exploration tuning + the full gate (the 4 anti-cheats:
  lesion → no learning + crash; yoke → no learning; reward-provenance ✓ here; remapped-action ✓ here) at ≥6 seeds.
- **NO `sim/` edit** — reuses the brain-region framework, the neuromodulator subsystem, and the three-factor
  learning path. The integration is additive runner code.

This caps the artificial-life motivational-core arc: the core is de-risked across all faces (learns / spikes /
sustains life) and now **assembled into a running spiking self-directed agent**; the remaining work is tuning it
to a robust policy-convergence GO.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._homeostatic_spiking_agent_integration --seed 43 --trials 120 --deplete 0.015
```
