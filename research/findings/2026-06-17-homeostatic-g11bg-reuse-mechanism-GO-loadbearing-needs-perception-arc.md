# Robust homeostatic agent: REUSE the validated g11_bg learner (not a fork) via one default-off hook — mechanism GO; the load-bearing config is the SC-reflex perception arc

**Date:** 2026-06-17 (CYCLE 132, autonomous)
**Status:** **Reuse mechanism GO; behavioural convergence = honest BOUNDARY (the
wall is moving-goal perceptual navigation, NOT the drive or the reuse).** The
validated navigation learner (`run_moving_goal_episode`, the basal-ganglia
cascade + value critic + spiking dopamine) is reused for a homeostatic agent
through ONE guarded, default-off hook — no fork, no re-derivation of the tuned
loop. With the corrected (validated Rank-2) config, intact / ungated / lesion all
crash post-wean: the learned perception does not robustly form for RELOCATING
food in a single episode (the perception cold-start re-emerges when the goal
moves on every eat; the Rank-2 6-seed GO was single/fixed-goal). This is the
project's known hardest nav problem re-confirmed in the homeostatic regime — NOT
a drive-gating failure (intact was the best of the three). The FUNCTIONAL
self-directed agent already exists (the CYCLE-130 capstone, 6-seed rate-proxy
GO); this arc characterizes the spiking-perception boundary for moving-goal
homeostasis. See the Results section.

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

## Two diagnostic rounds (systematic debugging)

**Round 1 (a config bug, not a wall).** The first intact run (grid 10, 1500
steps) crashed post-wean (23 eats pre-wean, 6 post, 707/975 crash-steps). Cause,
found by comparing to the validated reference: I omitted the validated N8 (genuine
BG disinhibition) + N6 (spiking-WTA readout) action-selection back-end and weaned
the teaching reflex at step 525 — 4× too early. The validated Rank-2
learned-perception config
(`2026-06-08-Rank2-learned-vision-circuit-and-teacher-correction.md`, 6-seed GO)
needs `--genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300
--genuine-thal-tonic-pa 750 --readout-source spiking_wta --urgency-max-pa 180
--learned-perception --enable-dlpfc-wm --enable-pfc-nmda` AND the schedule
`--sc-reflex-wean-start 2000 --sc-reflex-wean-steps 1000 --n-steps 6000
--grid-size 8`. Corrected the probe to match it exactly.

**Round 2 — the real result (corrected config, grid 8, n_steps 6000, seed 42):**

| mode | post-wean eats | mean energy (post-wean) | crash-steps |
|---|---|---|---|
| intact (drive-gated) | **33** | **0.173** | 3277 |
| ungated (full reward) | 22 | 0.109 | 3576 |
| lesion (no reward) | 23 | 0.109 | 3604 |

**All three crash post-wean — INCLUDING ungated (full reward).** The isolation
control is decisive: this is **NOT** a drive-gating failure (intact, the
drive-gated agent, was actually the BEST of the three: 33 vs 22–23 post-wean eats,
0.173 vs 0.109 mean energy). It is a **perception cold-start**: the
reward-learned perceptual navigation does not robustly form for **relocating**
food within a single episode. Pre-wean (reflex on) every mode ate well (~30–38×,
the innate goal-agnostic reflex handles moving food); post-wean the learned
perception navigated ~4× worse than the reflex (33 eats / 3000 steps ≈ one reach
per ~90 steps, vs the reflex's ~1 per 20) — too weak to sustain. The validated
Rank-2 GO was **single/fixed-goal**; a goal that relocates on every eat is the
harder moving-goal regime, and the learned perception does not consolidate fast
enough against a constantly-moving target in one episode.

## Verdict — honest BOUNDARY

- **GO:** the reuse MECHANISM (the one default-off hook on the validated learner)
  + the drive-gating + eat/energy body dynamics. The drive is even weakly
  load-bearing (intact > both controls on post-wean nav).
- **BOUNDARY:** robust single-episode behavioural convergence, because
  **moving-goal perceptual navigation** (the project's known hardest nav problem)
  does not form fast enough in the relocating-food regime — and this is a
  PERCEPTION wall, not a drive or reuse wall (ungated/full-reward crashes too).
- **The functional self-directed agent already exists** (CYCLE-130 capstone:
  built, brain-faithful, sustains life, 6-seed rate-proxy GO). This arc
  precisely characterizes where the spiking-perception version hits its wall.
- **The scoped fix, if revisited:** a fixed→relocating food CURRICULUM (let the
  learned perception consolidate on a fixed goal first, then relocate) and/or the
  multi-goal-generalization tuning the Rank-2 arc flagged as seed-variable — a
  dedicated multi-cycle nav effort, deliberately NOT pursued now (the owner
  reprioritized to the conversational architecture 2026-06-17, and the functional
  living agent is already demonstrated).

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
