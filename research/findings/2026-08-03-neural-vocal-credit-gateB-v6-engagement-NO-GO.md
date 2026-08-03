---
type: finding
status: no-go
date: 2026-08-03
mechanism: neural-vocal-action-credit-v6-persistent-expectation
backend: cupy
seed: 0
---

# Gate B v6: stronger sparse trace input does not create a prediction

## Result

Gate B v6 is retired before a runner or formal seed manifest was built. The
preregistered four-value implementation ladder exhausted its allowed
trace-to-expectation engagement test on reserved smoke seed `0` and the RTX
3090. The active trace remained intact, but the striatal MSN-D1 expectation
population emitted no spikes during the fixed pre-outcome delay at any tested
weight.

Artifact:
`research/findings/raw/vocal_action_credit_gate_v6/engagement_ladder_seed0_cupy.json`.

| Fixed trace-to-expectation weight | Delay trace spikes, channels 0/1 | Delay expectation spikes, channels 0/1 | Outcome expectation spikes, channels 0/1 |
|---:|---:|---:|---:|
| `0.1` | `27 / 251` | `0 / 0` | `14 / 9` |
| `1.0` | `27 / 251` | `0 / 0` | `14 / 19` |
| `2.0` | `27 / 251` | `0 / 0` | `14 / 24` |
| `4.0` | `27 / 251` | `0 / 0` | `14 / 24` |

The winner was channel `1` in every probe. Learning was frozen and the route
weight was set uniformly, so this was an engagement diagnostic rather than a
claim about acquisition. No scientific seed was assigned or consumed.

## Interpretation

The failure is upstream of GABA-B output. The action trace fires throughout the
delay, but 24 sparse trace cells do not provide enough convergent excitation to
move the MSN-D1 expectation cells into their firing regime. Increasing each
synapse by 40 times changes outcome activity but does not create pre-outcome
activity. Testing stronger values would violate the preregistered ladder and
repeat a parameter-fishing pattern.

The repaired project RAG search recovered the prior diagnosis in
`2026-06-08-striatal-value-critic-firing-research.md`: MSN-D1 value cells are
deliberately difficult to excite because of their KIR-like down state, and the
faithful remedy is many converging active afferents rather than replacing the
cell with an easier generic relay. Its direct diagnostics found only `2.2-5.4
Hz` from one to three active afferents, but `22-49 Hz` from a dedicated dense
afferent with noise disabled. `n9_convergent_upstate_derisk.py` later embodied
that design with 200-cell input populations.

This negative therefore retires the **sparse trace-to-MSN engagement
mechanism**, not learned action-local expectation or the downstream GABA-B
hypothesis. The successor must test dense convergent neural action context,
retain MSN-D1 value cells, and keep the reward association local and plastic.

## Controls and scope

- Backend was CuPy on the RTX 3090; `SIM_NO_PROVENANCE=1` prevented a diagnostic
  sweep from being mistaken for a formal artifact.
- Only reserved seed `0` ran. V5's calibration, development, and held-out seeds
  remain sealed and are not reassigned.
- No source file, threshold, timing, action-selection parameter, or scientific
  seed changed during the ladder.
- This test does not assess GABA-B suppression or omission, because the required
  pre-outcome expectation signal never engaged.

## Sources used before choosing the successor

- Project finding: `2026-06-08-striatal-value-critic-firing-research.md`.
- Project runner: `research/runners/n9_convergent_upstate_derisk.py`.
- Cohen et al. (2012), sustained reward-scaled VTA GABA delay activity,
  [PMC3271183](https://pmc.ncbi.nlm.nih.gov/articles/PMC3271183/).
- Eshel et al. (2015), additive expectation-dependent dopamine suppression,
  [PMC4567485](https://pmc.ncbi.nlm.nih.gov/articles/PMC4567485/).

## Decision

`DYNAMICS_FAIL / NO-GO`. Do not build the named v6 runner and do not extend the
weight ladder. Proceed only under a new preregistration for dense convergent
trace input.
