# Nav close-out R4 — delayed-reward value-load-bearing test — NEGATIVE (2-seed) (2026-06-27)

The R4 delayed-reward 2×2 factorial {value ON/OFF}×{immediate/delayed} tested whether the spiking nav value-critic (N-2, the CYCLE-1B default) is behaviorally LOAD-BEARING. Result (grid-32/1800, reward-delay 12, controller-run):

| seed | imp_immediate | imp_delayed | value×delay interaction | imp_delayed_permuted |
|---|---|---|---|---|
| 42 | −0.005 | −0.004 | +0.001 | −0.041 |
| 43 | +0.007 | −0.112 | −0.118 | −0.228 |

- `helps_on_delayed = False` (both seeds): improvement_delayed ≤ 0 — the value-critic does NOT improve the delayed-reward nav.
- `neutral_on_immediate = True` (both): correct (value isn't needed on immediate-reward).
- `value×delay interaction` inconsistent (+0.001 / −0.118): no reliable delayed-specific value help.
- `permute_control_ok = True` (both): the permuted-contingency control is sound.

⇒ the spiking value-critic, as deployed, is **NOT behaviorally load-bearing** on this delayed-reward nav. Per BRAIN-BASED-ONLY this honest NEGATIVE is the deliverable (it maps a substrate limit). It does NOT regress anything — the critic still runs on spikes; this measured whether the value it computes drives behavior, and it does not.

## The candidates (a deep-research gate reconciles these — NOT just accepted)
1. **Task-design / validate-by-function** (`feedback_validate_signal_by_its_function`): the delayed task still has the actor reach the goal; the delayed reward does not change WHICH action is best, so V may predict the reward without being behaviorally load-bearing. The proper function test is a value-DRIVEN-CHOICE task (pick the higher-value path/goal), not merely delayed reward. **(Suspected primary cause.)**
2. **Weak merged-δ → R3:** the merged value-train δ is ~1.3× (the position-blind up-state floor, `2026-06-18-merged-navcritic-valuetrain-BOUNDARY.md`); the graded dendritic plateau (δ=1.33=host ceiling on-substrate, D2 Stage-1) is the candidate SURPASS — but that's the dendrite (a protected edit, accepted-deep-adjacent).
3. **Spatial actor-critic-credit substrate wall** — the nav-gate flagged R4 as where a real point-neuron wall might appear; possibly the accepted-deep dendritic/credit-assignment family.

NO `sim/` edit in R4 (the delay is an existing `homeostatic_hook` FIFO; the value-OFF is the established `cp_gabab_synapse_mask` lesion). Raw: `research/findings/raw/navcloseout_R4/R4_factorial_seed42.json`, `_seed43.json`.
