# Robust homeostatic agent: REUSE the validated g11_bg learner (not a fork) via one default-off hook — mechanism GO; the load-bearing config is the SC-reflex perception arc

**Date:** 2026-06-17 (CYCLE 132, autonomous)
**Status:** **Reuse mechanism GO (smoke).** The validated navigation learner
(`run_moving_goal_episode`, the basal-ganglia cascade + value critic + spiking
dopamine) is reused for a homeostatic agent through ONE guarded, default-off hook
— no fork, no re-derivation of the tuned loop. A code-level analysis settled the
correct de-risk design (below). The decisive intact/lesion/yoke convergence run
is in flight; this doc records the landed mechanism + the design finding.

## Why reuse, not fork (the CYCLE-131 hand-off)

CYCLE 130 built the first running brain-based homeostatic agent (place + motor +
AgRP drive + hunger modulator on one bridge), brain-faithful and functional, but
a *minimal* place→motor actor does not robustly converge a learned policy (CYCLE
131: exploration-annealing, short-corridor+long-eligibility, and a TD value
critic each helped only 1/3 seeds). The decisive diagnosis: robust spiking RL
from a sparse intrinsic reward needs BOTH **value bootstrapping** and **clean
basal-ganglia action selection** — exactly the machinery the validated nav loop
`run_moving_goal_episode` already has (the `striosome_value` critic, the BG
cascade's disinhibition winner-take-all, the spiking-SNc dopamine, and a
coordinate-free perceived-approach reward).

So the robust homeostatic agent does **not** need a fork of that ~1500-line
function, nor a re-derivation of its tuned cascade (which the project has
repeatedly shown introduces subtle bugs — e.g. the n_cortex 100-vs-400 probe
mismatch). It needs **one change**: gate that learner's reward by a
self-generated drive.

## The reuse mechanism (one default-off hook)

A single guarded parameter `homeostatic_hook=None` added to
`run_moving_goal_episode` (`research/runners/g11_bg_runner.py`). Default `None` →
`if None is not None` is False → a **no-op for every existing caller**
(byte-identical by inspection; the diff is purely additive + guarded). When set,
the hook is called once per trial *after* the natural reward is finalized:

```
gated_reward, new_goal = homeostatic_hook(reward, x, y, gx, gy, step, dist_after)
```

It (a) **gates the reward by a self-generated hunger drive** (`reward *= hunger`,
the Keramati-Gutkin reward≡drive-reduction principle) and (b) **relocates the
food** (goal) on an eat event (`dist_after == 0`). All homeostatic state (energy,
hunger, food location) lives in the probe's closure
(`research/runners/_homeostatic_g11bg_reuse_probe.py`), not in the protected
function.

## The design finding that fixed the de-risk: the heuristic catch-22

A homeostatic agent is only meaningful if the drive is **load-bearing** for
behaviour. Making it so on g11_bg required resolving a catch-22, established by
code inspection:

- **Coordinate heuristic ON** → the heuristic *directly drives the cortex*
  (`g11_bg_runner.py` ~6270-6294: it writes `cp_external_input_current` for the
  goal-direction pool). Navigation is then **reward-independent** — so lesioning
  the drive could not change behaviour. The drive is not load-bearing.
- **Heuristic simply OFF** → the documented **cold-start NEGATIVE**: with no
  teacher, the learnable perception→action mapping has no selectivity for STDP +
  reward to amplify, so navigation never forms (for *any* drive condition).

The project already resolved exactly this catch-22 for navigation: the **innate
superior-colliculus orienting reflex** — an image-based, coordinate-free teacher
that **weans** — after which navigation is carried by the **reward-LEARNED**
dorsal perception. That is the unique regime where the drive is load-bearing:
lesion the drive → reward → 0 → the learned policy never forms → post-wean
navigation collapses → the agent starves.

So the homeostatic probe uses that perception arc:
`enable_visual_cortex + sc_orienting_reflex + learned_perception_from_vision +
sc_reflex wean + heuristic_strength 0 + perceived_approach_reward`
(coordinate-free image-eccentricity reward), drive-gated by the hook.

## Mechanics — GO (GPU smoke, both configs)

- The hook fires every trial on the full BG-cascade bridge (758 neurons, ~33k
  synapses); the eat+relocate path is exercised; energy depletes/refills as the
  body dynamics specify.
- **Perception-arc smoke (grid 8, 180 steps):** the agent ate **3× after the
  reflex weaned** — i.e. the reward-learned perception navigated post-wean with
  the drive-gated reward. (The tiny smoke also crashed under its aggressive
  depletion; that is the smoke, not the test.)

## The decisive run (in flight) + verdict criteria

Intact at grid 12 / 1800 steps / deplete 0.012, the reflex weaned over
steps ~630→1080 so the final ~40% is pure reward-learned perceptual navigation.

- **If intact SUSTAINS post-wean** (min-energy stays well above the crash floor,
  post-wean eating continues) → run `lesion` (hunger frozen to 0 → reward 0) and
  `yoke` (hunger decorrelated) at the same config; **GO** = intact ≫ lesion,yoke
  on post-wean survival → the drive-gated reward produces load-bearing learning →
  proceed to swap the host hunger proxy for the **neural** AgRP-driven hunger
  modulator (validated on spikes, CYCLE 127) + multi-seed.
- **If intact CRASHES post-wean** (the reward-learned perception did not form
  under the drive-gated, relocating-food reward) → an honest **NEGATIVE/BOUNDARY**
  pinning that the homeostatic agent needs more of the perception-arc machinery
  (the full N1–N9 stack), not just the cascade + critic.

## Honest scope

This is the **convergence-mechanism** de-risk. The hunger here is a host
energy-deficit proxy; the reward is the coordinate-free N5 image reward (already
brain-shaped). The **neural** hunger (AgRP pool → `from_region_firing_signed`
modulator) is the brain-based realization, wired only after this convergence
question answers GO. `run_moving_goal_episode`'s `homeostatic_hook` is the
reusable foundation for that build.

## Reproduce

```bash
python -m research.runners._homeostatic_g11bg_reuse_probe --smoke
python -m research.runners._homeostatic_g11bg_reuse_probe --seed 42 --mode intact \
    --n-steps 1800 --grid-size 12 --deplete 0.012 --refill 0.6
# then --mode lesion / --mode yoke, and:
python -m research.runners._homeostatic_g11bg_reuse_aggregate --seeds 42
```
