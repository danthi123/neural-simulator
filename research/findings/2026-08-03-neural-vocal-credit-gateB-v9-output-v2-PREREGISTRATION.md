---
type: preregistration
status: superseded
date: 2026-08-03
mechanism: neural-vocal-action-credit-v9-gabab-output-v2
runner: research/runners/_vocal_action_credit_gate_v9_graded_dendritic.py
---

# Gate B v9 Phase 2 v2: fixed contingent probe blocks

> **Executed once on the reserved CuPy seed and superseded by an undefined
> result.** Every protocol check passed, but no block sampled trained action
> `0`. See the [finding](2026-08-03-neural-vocal-credit-gateB-v9-output-v2-UNDEFINED-no-trained-action.md).

**Filed after the first output protocol was undefined and before replacement
code or execution.** The undefined artifact omitted the passing engagement
protocol's baseline trial, causing every condition to miss the retained 90%
clean-training threshold, and every single probe selected neural action `1`;
action `0` was the only rewarded and learned action. Nothing about reward
suppression or omission was decidable. The preserved
[undefined finding](2026-08-03-neural-vocal-credit-gateB-v9-output-UNDEFINED-action-mismatch.md)
records both protocol failures. Center `2`, seed `0`, closed-output training,
and every mechanism and threshold from Phase 2 remain fixed.

## Locked protocol corrections

1. Restore exactly one pre-learning, no-reward baseline trial before recording
   initial weights and opening training plasticity, matching the passing
   engagement runner. Output and plasticity remain closed during that trial,
   and its expectation firing must be zero. Then run the unchanged 12
   contingent training trials. This is conformance to the original Phase-2
   requirement to retain engagement timing, not a new calibration.
2. Replace each one-probe condition with a fixed block of four probes. Do not
   change any circuit parameter.

- Reward block: deliver reward if and only if the neural action is `0`.
- Omission block: withhold reward for every action.
- Analyze expected reward and expected omission only on probes where the brain
  executes action `0`, because that is the only action with rewarded training
  history.
- Retain action-`1` rows as an internal negative control; do not discard them
  from the artifact.
- Require at least one action-`0` and at least one action-`1` row in every
  reward and omission condition, and require the complete four-action sequence
  to match across all three lesions for each probe type. If either prerequisite
  fails, the result is `UNDEFINED`. Do not extend the block or repeat the seed.

This is not host action selection. The selector runs unchanged on every probe,
all four rows are recorded, and the environment applies a fixed action-
contingent consequence. The correction removes the invalid assumption that one
fixed neural choice would happen to be the rewarded action.

## Locked checks

The restored baseline expectation must be zero and at least 90% of the 12
training trials must contain one clean action in every condition. Failure of
either retained engagement check is `OUTPUT_FAIL`, not `UNDEFINED`.

Apply every outcome check to the action-`0` rows. Use arithmetic means for
dopamine burst, dopamine dip, SNc, LHb, RMTg, pre-outcome expectation, and
GABA-B/GIRK. Because action sequences must match, each lesion contributes the
same number of rows. The calculations are locked as follows:

1. within intact action-`0` rows, mean delay expectation in channel `0` is
   nonzero and at least three times mean channel-`1` expectation, separately
   for reward and omission blocks;
2. the learning lesion removes at least 80% of intact mean channel-`0` delay
   expectation in action-`0` rows, separately for both blocks;
3. the output lesion sets mean pre-outcome SNc GABA-B/GIRK to zero and retains
   mean channel-`0` delay expectation within 20% of intact, separately for both
   blocks;
4. intact reward dopamine burst is at least 20% below both lesions;
5. intact omission recruits LHb and RMTg and creates a dopamine dip, with the
   output and learning lesions removing at least 80% of LHb/RMTg activity and
   at least 50% of the dip;
6. in intact action-`1` rows, mean channel-`1` delay expectation does not exceed
   intact action-`0` rows' mean channel-`0` delay expectation, separately for
   reward and omission blocks;
7. all probe weights remain byte-identical, training changes remain confined,
   and formal phases stay sealed.

If CuPy passes, run this identical fixed-block protocol on NumPy under the
existing agreement rules, then audit independently. If it fails, retire the
v9 output realization without changing center, GABA-B parameters, probe count,
or action policy.
