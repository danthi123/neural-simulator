# Limbic-core load-bearing diagnostic on a HIDDEN-GOAL task — honest NEGATIVE (2026-06-19)

## The question

Is the spiking **reward / value / dopamine limbic core BEHAVIORALLY LOAD-BEARING** on a
hidden-goal (Morris-water-maze analogue) task? The 2026-06-19 scoping
(`2026-06-19-next-spikeification-default-on-scoping.md`) flagged that on the standard
gridworld the limbic core is **GREEN_INERT** — validated as spiking, but behaviorally inert,
because the task is *orient-solvable*: the heuristic / SC-orienting can navigate to a
*visible* goal WITHOUT reward, so reward/value/dopamine never change the behavior
(`2026-06-18-merged-neural-reward-QUALIFIED-GO.md`: "nav not regressed host→neural because the
reward is not strongly behaviorally load-bearing").

The proper test (owner standard `feedback_validate_signal_by_its_function`): **make the goal
NOT directly perceivable** so the agent CANNOT orient toward it and MUST learn its location
via reward → value → dopamine → corticostriatal plasticity; then **lesion the reward** and
require the lesion to **collapse the BEHAVIOR (the nav score)**, not merely the SNc/reward-pop
firing.

## Method (cheapest-first, NO `sim/` edit)

Two additive default-OFF params on `run_moving_goal_episode` (+ CLI flags; byte-identical when
unset), committed `e0266017`:

- **`--hidden-goal`**: the goal's coordinates are NOT fed into the brain anywhere — the
  `ppc_goal_input` goal drive (`(gx,gy)` → goal cells) is zeroed each step, while the
  own-position **place drive** (`(x,y)` → `sensor_place_readout`) STAYS (own position is
  legitimate egocentric self-knowledge under BRAIN-BASED-ONLY). Combined with
  `--heuristic-strength 0` (no goal-direction teacher) and no cue-reflex / SC-orienting /
  learned-perception, the ONLY goal-related signal reaching the agent is the **scalar reward**.
- **`--lesion-reward`**: unconditional clamp `reward = 0` each step (after the natural
  computation), so NO learning signal reaches dopamine / the value critic / corticostriatal
  STDP. The load-bearing anti-cheat.

The hidden-goal learner present in the architecture: the agent's own place code drives the
cortex pools through the **plastic** `sensor_place_readout → cortex_{action}` pathway
(`g11_bg_runner.py:1450`), feeding the BG cascade (`cortex → str_D1 → gpi → thal →
sel/commit`, fully-spiking `readout_source="spiking_wta"`). Reward modulates the corticostriatal
STDP. So the substrate to learn place→action from reward IS present.

Probe: `research/runners/_limbic_loadbearing_probe.py` (committed `e2e2e246`). Metric: sum over
phases of `final_quarter_mean_distance` (Manhattan; **LOWER = better**), single static goal
(6,6), start (1,1), grid-8. NOTE: the moving-goal path imports `cupy` directly (GPU-only), so
runs are on the RTX 3090 (the smoke is small: ~140 s / 600-step condition); `SIM_BACKEND=numpy`
is incompatible with this runner.

**Reference floor:** a uniform random-cardinal walk on this geometry (400-trial Monte-Carlo)
gives `final_quarter_mean_distance = 5.52 ± 1.41` (overall mean 5.68). Perfect navigation ≈ 0.

## Results

### 3-condition core (seed 42, grid-8, 600 steps, corticostriatal reward-STDP)

| Condition | goal | sum_finalQ (lower=better) | frac_at_goal | vs random floor 5.52 |
|---|---|---|---|---|
| (iii) control_visible (heuristic ON) | (6,6) | **0.660** | 0.277 | near-perfect ✓ harness OK |
| (i) hidden_reward_ON | (6,6) | **2.873** | 0.035 | below floor |
| (ii) hidden_reward_OFF (lesioned) | (6,6) | **1.640** | 0.062 | below floor |

Two surprises: (a) BOTH hidden conditions are **well below the 5.52 random floor**, and
(b) **reward-OFF (1.64) BEATS reward-ON (2.87)** — lesioning the reward *improved* the score.
Lesioning the reward did NOT collapse the behavior. That already fails the load-bearing test.
But it raises the question: where does the strong below-random goal-bias come from if not from
reward?

### Goal-location anti-cheat — the decisive control (seed 42, hidden + reward OFF, 400 steps)

If the below-random performance were goal knowledge (a leak), the agent would track the goal as
it moves. If it is a **fixed structural bias**, the agent ends at the SAME place regardless of
the goal.

| goal | sum_finalQ | end position | distance of end-pos from goal |
|---|---|---|---|
| (6,6) | **1.67** | (7,6) | ~1 (NEAR goal) |
| (1,6) | **6.20** | (6,7) | ~5 (FAR from goal) |
| (6,1) | **6.50** | (6,6) | ~5 (FAR from goal) |

**The agent drifts to the high-x / high-y (NE) corner ≈ (6–7, 6–7) IRRESPECTIVE of the goal.**
This is conclusive on two points:

1. **The goal is genuinely hidden** — with `--hidden-goal` the agent has NO idea where the goal
   is (it goes to the same corner no matter where the goal is). No leak; the suppression works.
2. **The reward is NOT load-bearing** — behavior is driven entirely by a fixed structural
   corner-drift of the BG cascade (a random-init directional bias in the place→cortex / cascade
   weights), not by the goal location, even with reward fully ON.

The earlier "1.64 looks good" for hidden_reward_OFF @ goal (6,6) was a **goal-coincidence
artifact**: the structural drift corner happens to coincide with the goal corner (6,6). That is
exactly the false-positive the owner standard warns about — the goal-location control unmasked
it. With reward ON the reward-STDP slightly *perturbs* the lucky NE-drift away from (6,6) (→
2.87), which is why reward-OFF "won" at that one goal location.

### Full spiking limbic core confirm (seed 42, hidden + reward ON, 500 steps)

The fuller core — **neural value critic + spiking SNc (dopamine RPE) + spiking reward delivery
(PPN-like `reward_us`)** — was confirmed at two goal locations to check the verdict is robust to
the complete limbic core (not just corticostriatal STDP):

| goal | sum_finalQ | end position | distance of end-pos from goal | tracks goal? |
|---|---|---|---|---|
| (6,6) | 2.072 | (6,7) | ~1 (coincides with NE drift) | NO (drift==goal) |
| (1,6) | 4.928 | (5,6) | ~4 (≈ random floor) | **NO** |

**The full spiking limbic core ALSO fails to track the goal.** At goal (1,6) it ends at (5,6) —
~4 Manhattan from the goal, essentially at the random floor — and at goal (6,6) it lands near the
NE corner by the same structural drift. The verdict is therefore **robust to the complete spiking
limbic core**: the agent does not learn the hidden goal location from reward whether dopamine is
host-computed reward-STDP OR the full spiking SNc + neural value critic + spiking reward delivery.
(The critic-ON score at the *coincident* goal (6,6) is slightly worse than the structural-bias-only
1.67 because the reward/critic perturb the lucky drift without carving a goal-specific policy.)

## Verdict — honest NEGATIVE: the hidden-goal variant (as configured) is STILL not reward-load-bearing

The reward/value/dopamine limbic core does **not** become behaviorally load-bearing simply by
hiding the goal. The lesion (reward → 0) does **not** collapse the behavior; behavior is
dominated by a fixed structural corner-drift of the BG cascade that is independent of the goal
and independent of the reward. The agent never learns the goal location from reward at this
scale.

This is the precise, valuable diagnostic the owner standard is after: a clean negative that
**sharpens the next lever**. The reason it is not load-bearing is now mechanistically
characterized (NOT "the reward signal is bad"):

- **The place→action map never forms.** The substrate (place code → plastic
  `sensor_place_readout → cortex` → BG cascade, reward-modulated STDP) exists, but over a single
  static goal in a few hundred steps the corticostriatal STDP does not overcome the cascade's
  random-init directional bias. The cascade emits a **goal-independent fixed action preference**,
  so the place context is not yet being read into a place-specific action.
- This is the **actor-critic credit-assignment + place-code-selectivity problem**, the SAME
  family the project hit on the conversational side (per-seed structural pool variance dominating
  a weak learning signal; the W→A "global scalar feedback fails at biological scale" verdict).
  A dense per-step distance reward + reward-STDP is not, by itself, enough to carve a
  place→action policy on this point-neuron cascade at this scale.

## What WOULD make a hidden-goal task reward-load-bearing (the next levers, in order)

1. **A place-selective actor with a working value baseline.** The fix is almost certainly NOT
   "more of the same reward" but giving the actor a place-specific substrate the value critic can
   shape: (a) ensure the `place` code is sparse + selective per location (so different positions
   drive different cortex sub-populations — the documented place-cell sparsity), and (b) let the
   **neural critic's V(place)** gate the corticostriatal eligibility (advantage = r − V), so the
   reward only reinforces actions that beat the place's expected value. The TD-value critic
   (`sim/td_value_critic.py`) + the validated N9 spiking SNc RPE are the pieces; the missing link
   is routing the critic's advantage into the actor's place→action STDP. This is the
   actor-critic the project has been building toward — the hidden goal is the task that makes it
   load-bearing.
2. **A longer / curriculum schedule with a goal-stable phase** so the policy has time to
   converge before any goal change (a single 600-step phase is too short for cold-start
   corticostriatal carving from a structural-bias start).
3. **Break the structural corner-drift** (the confound): symmetrize the place→cortex init / add
   the MSN cross-pool WTA + a mild homeostatic balance so the un-learned cascade is near-uniform
   over actions (random-floor, not NE-biased). Then any below-floor performance is attributable
   to learning, and the reward lesion has a clean baseline to collapse to.
4. **A different hardness knob if actor-critic still under-performs:** a *cued* hidden goal
   (a distal landmark the agent must associate with the reward via the value system) rather than
   a pure latent goal — closer to the real Morris water maze, where the animal uses distal cues +
   reward, and known to be hippocampal/striatal-value-dependent.

## Recommended next concrete step

**Build the missing actor-critic link and re-run THIS diagnostic.** Specifically: route the
neural critic's advantage (r − V(place), the N9 spiking-SNc RPE already deployed) into the
`sensor_place_readout → cortex_{action}` (and `cortex → str_D1`) eligibility on a hidden-goal
task with a sparse selective place code and a single long goal-stable phase, with the
structural-bias confound removed (lever 3). Then the load-bearing test is: hidden_reward_ON
solves it (below floor AND **tracks the goal across locations** — the anti-cheat) while
hidden_reward_OFF collapses to the (now random) floor. If that passes 6-seed GPU, the limbic
core is load-bearing and Option B (the harder task redeems it) is validated. The cheap-first
de-risk is a 1-seed CPU/GPU smoke of the advantage-routed actor at one goal location vs its
reward-lesion, exactly as here.

Until then: **on the gridworld (visible OR this latent-goal variant) the spiking limbic core
remains GREEN_INERT / not-load-bearing**, and the honest headline is that hiding the goal is
necessary but NOT sufficient — the actor-critic credit path must also be closed.

## Artifacts

- Runner flags: `research/runners/g11_bg_runner.py` (`hidden_goal`, `lesion_reward`; commit
  `e0266017`).
- Probe: `research/runners/_limbic_loadbearing_probe.py` (commit `e2e2e246`).
- Raw: `research/findings/raw/_limbic_loadbearing_summary.json`,
  `_limbic_{control_visible,hidden_reward_ON,hidden_reward_OFF}_seed42.json`,
  `_limbic_goalctrl_*.json`, `_limbic_critic_*.json`.
