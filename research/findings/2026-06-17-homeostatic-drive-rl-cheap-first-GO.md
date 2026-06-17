# Homeostatic-drive RL — cheapest-first GO (6 seeds): the agent learns to keep itself alive from a self-generated intrinsic reward

**Date:** 2026-06-17
**Status:** **GO, 6 seeds (rate-proxy / algorithm level).** The artificial-life frontier's load-bearing question —
*can the agent learn a policy from a SELF-GENERATED intrinsic drive-reduction reward, with no external goal?* — is
GO. The motivational core is reachable; greenlight the spiking-bridge realization.

## Context — the missing piece

The agent has a competent cognitive engine and a competent body, but (per the frontier scoping,
`2026-06-17-artificial-life-frontier-scoping.md`) **no motivational core**: every behavior is an externally-
triggered episodic task call, and even the navigation reward is an exogenous host Manhattan-distance formula
(`g11_bg_runner.py:3132`). A *living* agent needs a neural internal state that generates its own goals and defines
reward intrinsically. The recommended first capability is a **neural homeostatic drive** (hypothalamic
AgRP=hunger / POMC=satiety two-pool push-pull; Keramati & Gutkin, *eLife* 2014: reward ≡ reduction of a
homeostatic deviation). The scoping flagged the **load-bearing risk** as not the drive→reward→dopamine half (a
proven pattern) but whether the substrate can **learn a policy from the sparse intrinsic reward** — so this
cheapest-first numpy probe falsifies exactly that before any spiking build.

## The loop (`_homeostatic_drive_rl_cheap_first_probe.py`)

Host code is the body + environment only (per the brain-based-only standard); the drive + reward are the "brain"
parts (rate-proxied for the cheap-first):
- **Body:** 1-D energy `E∈[0,1]`, depletes each step; `deficit = set_point − E`. Reaching the resource refills `E`.
- **Drive:** a 2-pool push-pull rate model (AgRP rises with deficit, POMC with surplus, reciprocal inhibition);
  `drive = agrp − pomc` tracks the deficit.
- **Reward (intrinsic):** `r = drive_before − drive_after` (= drive REDUCTION). Eating drops the deficit → drops
  the drive → positive `r`. **No host distance/goal term anywhere.**
- **Learning:** tabular Q-learning over (position) × {two abstract actions}; the agent learns to seek the resource
  from `r`. **The action→direction map is REMAPPED (randomized) per seed**, so the agent cannot default to the
  optimal action — it must *learn* which action moves toward food.

## Result (6 seeds: 42/43/44/100/101/102)

| metric | value | gate |
|---|---|---|
| corr(deficit, drive) | **+0.95 … +0.96** | ≥ +0.90 — the neural drive encodes the body's deficit ✓ |
| real late time-to-resource | **7.4 … 8.1** (optimum ≈ 7) | ✓ near-optimal |
| **lesion** (drive frozen → r=0) late | **32 … 34** | the agent cannot learn → stays slow ✓ |
| **yoke** (drive shuffled → r uninformative) late | **24 … 40** | learns *wrong* → stays slow ✓ |
| learning (real ≤ 0.75× both controls, near optimum) | **6/6 seeds** | **GO** |

**GO, 6/6.** The agent learns an efficient resource-seeking policy (time-to-resource ~7.7, near the ~7 optimum)
**only** from the intrinsic drive-reduction reward — the lesion (no drive → no reward) and yoke (shuffled drive →
noisy reward) controls both stay slow (~30+). A self-generated homeostatic drive **generates the goal** (stay
fed) and the agent **learns to keep itself alive**, with no externally-supplied goal.

## Anti-cheats (all hold)

- **Lesion** the drive → `r=0` → no learning (late ~33, ≈ random). Self-direction collapses without the drive.
- **Yoke** (shuffled drive of matched marginal stats) → uninformative reward → learns wrong (late ~30). The
  *informative* drive-reduction signal is load-bearing, not just any modulation.
- **Remapped action map** (the decisive control): the optimal action is randomized per seed, so the agent cannot
  reach the resource by default — only by learning from `r`. (An earlier version omitted this and the lesion
  reached the resource for free via the argmax-of-zeros default; adding the remap exposed the true GO.)
- **No host goal term:** `r` is the drive reduction, computed from the drive pools — never a distance-to-resource.

## Honest scope

This is the **algorithm-level** cheapest-first: a rate-proxy 2-pool drive + a tabular Q-learner (standing in for
the spiking basal-ganglia cascade). It de-risks the **reward structure** (intrinsic drive-reduction is a learnable
training signal, and the controls collapse) — NOT yet the spiking realization. The next step (per the scoping) is
the **brain-based version**: a 2-pool **spiking** drive region, the neuromodulator subsystem's
`from_region_firing_signed` rule sourcing `r` from the drive pools' firing (`cp_firing_states`, verified to exist,
`sim/neuromodulators.py:131`), and the **existing spiking dopamine RPE** + BG action cascade learning the policy.
Whether the *spiking* substrate learns the policy from this intrinsic reward is the next gate — and an honest
NEGATIVE there would pin the exact wall (a self-generated drive + correct neural reward can be built, but spiking
policy-learning from sparse intrinsic reward is the boundary), itself a high-value deliverable.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._homeostatic_drive_rl_cheap_first_probe --seeds 42 43 44 100 101 102
```

No `sim/` edit. The brain-based realization reuses the neuromodulator subsystem + the existing dopamine RPE / BG
cascade.
