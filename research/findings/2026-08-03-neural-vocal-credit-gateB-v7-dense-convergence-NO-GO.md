---
type: finding
status: no-go
date: 2026-08-03
mechanism: neural-vocal-action-credit-v7-dense-convergence
backend: cupy
seed: 0
---

# Gate B v7: a larger single learned afferent does not create a prediction

## Result

Gate B v7 is retired at its preregistered engagement ladder. On reserved smoke
seed `0` and the RTX 3090, all four trace populations remained active and the
rewarded local route learned, but the MSN-D1 expectation population emitted no
spikes before reward. The required full GABA-B and omission smoke therefore did
not run, and no formal seed was assigned or consumed.

Artifacts, each with a generated provenance sidecar:

- `research/findings/raw/vocal_action_credit_gate_v7/engagement_trace24_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v7/engagement_trace64_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v7/engagement_trace128_seed0_cupy.json`
- `research/findings/raw/vocal_action_credit_gate_v7/engagement_trace200_seed0_cupy.json`

Each artifact identifies the RTX 3090, source commit `59f8537a4`, exact trace
size, and five passing runtime preconditions.

| Trace cells per action | Clean trials, intact / lesion | Rewarded trials, intact / lesion | Rewarded-route mean, before -> after | Late delay expectation, intact / lesion |
|---:|---:|---:|---:|---:|
| `24` | `11 / 11` | `7 / 7` | `0.100 -> 1.307` | `0 / 0` | <!--derived-->
| `64` | `11 / 11` | `6 / 6` | `0.100 -> 1.247` | `0 / 0` | <!--derived-->
| `128` | `8 / 8` | `7 / 6` | `0.100 -> 1.203` | `0 / 0` | <!--derived-->
| `200` | `11 / 11` | `9 / 8` | `0.100 -> 1.553` | `0 / 0` | <!--derived-->

The learning-lesion route remained `0.100 -> 0.100` at every size. No synapse
outside the declared actor and expectation routes changed. The 128-cell point
also missed the fixed 90% action-cleanliness floor; the other three sizes met
that floor and still had zero pre-outcome expectation activity.

## Interpretation

This result retires **population size as the missing lever for the single
plastic trace-to-expectation route**. It does not retire MSN-D1 expectation,
local reward learning, or convergent cortical input in general. Presynaptic
trace activity, reward contingency, route learning, and the lesion control all
engaged, so a silent input or broken learning rule does not explain the null.

The post-result project search recovered an important distinction from the
June N9 work. Its firing-and-learning de-risk used two distinct afferent
populations: a dense fixed route put the MSN near its convergent-excitation
up-state, and a separate plastic route learned context on top of that state.
V7 supplied only the plastic route. Even at 200 cells and 50% density, its
learned mean weight of `1.553` did not bootstrap the postsynaptic cell.

The next experiment therefore cannot extend the v7 size or weight ladders. A
new preregistration may test a **dual-afferent** mechanism in which a fixed,
subthreshold convergent input supplies background state and a separate local
plastic action-context input supplies learned selectivity. The fixed arm must
not make expectation fire by itself; otherwise the learning lesion would cease
to be causal. Any operating-point ladder must be bounded and filed before code
or execution.

## Controls and scope

- Backend was CuPy on `NVIDIA GeForce RTX 3090`; the simulator reported the
  device in every result.
- Only reserved seed `0` ran. Formal execution remained sealed.
- Intact and expectation-learning-lesion arms each completed the fixed 12
  trials at every trace size.
- The presynaptic trace fired before outcome in both arms at every size.
- The lesion prevented route-weight change at every size.
- The null attribution is intentionally reported as undefined because both
  intact and lesion pre-outcome expectation activity were zero. It is not
  presented as a percentage attribution.
- The preregistered stop rule was followed: no downstream output smoke ran
  after all four engagement points failed.

## Sources used before choosing the next hypothesis

- Project finding: `2026-06-08-striatal-value-critic-firing-research.md`.
- Project design: `2026-06-09-N9-faithful-value-cell-design.md`.
- Project result: `2026-06-09-N9-convergent-upstate-derisk.md`.
- Project runner: `research/runners/n9_convergent_upstate_derisk.py`.
- Local catalog entry B.02: striatal MSN up-state and coordinated convergent
  cortical/thalamic input.
- Schultz (1998), predictive dopamine and corticostriatal learning, local copy
  `Schultz-1998-JNeurophysiol-PredictiveReward.txt`.
- Pomata et al. (2008), NMDA gating of striatal information flow,
  [Journal of Neuroscience](https://www.jneurosci.org/content/28/50/13384).

## Decision

`ENGAGEMENT_FAIL / NO-GO`. Retire v7 and do not extend its route-size or route-
weight ladders. Keep the capability open and pass through a new research and
preregistration gate for a biologically distinct dual-afferent mechanism.
