# Unified-bridge plasticity isolation — `plastic=False` does NOT isolate under global Hebbian (2026-06-04)

**Context.** Step 1 of the one-bridge unification (`docs/plans/2026-06-04-one-bridge-unification-step1-implementation.md`)
merges the conversational PARSER (`BridgeParser`) and COMPOSER (`CoreSimComposer`) onto ONE
`SimulationBridge`. The parser learns by Hebbian co-firing → it needs `enable_hebbian_learning=True`.
The composer's bind/unbind coincidence wiring is FIXED → it must not drift. On one shared bridge there is
only ONE global `enable_hebbian_learning` flag. The plan sets it `True` (for the parser) and relies on the
composer's wiring population being declared `plastic=False` to keep its fixed weights from drifting.

Task 1 (the load-bearing de-risk) tests exactly that assumption:
`tests/test_unified_brain_bridge.py::test_fixed_population_survives_global_hebbian`.

## Result: the assumption is FALSE.

A FIXED population (`plastic=False`, weight 320) co-resident on a bridge with `enable_hebbian_learning=True`
**drifts** when the simulation steps, even though a *different* (plastic) population is the one being driven:

```
FIXED 'bind' weight: before = 320.0   after = 319.89746   (≈ -0.10 over 300 steps)
```

The control in the same test passed (the PLASTIC 'parse' pair DID change), so the test is non-vacuous —
the drift is a real isolation failure, not a dead synapse.

## Root cause (code, not noise)

`sim/bridge.py`:

- `inject_explicit_wiring` (line ~1954) builds `self.cp_synapse_plastic_mask` from each population's
  `plastic` flag. **But it does NOT allocate `cp_plasticity_rate_gain`** unless a population also declares a
  `plasticity_gate` name (line ~1963: `if any_gated:`). With only `plastic=False` set, `cp_plasticity_rate_gain`
  stays `None`.
- The **Hebbian** update block (`_run_one_simulation_step`, lines ~5382–5416) gates per-synapse ONLY by
  `cp_plasticity_rate_gain` (lines 5396–5397 for potentiation, 5410–5412 for the `hebbian_weight_decay`
  term). It **never consults `cp_synapse_plastic_mask`**. So with the gain array `None`, every synapse —
  including the `plastic=False` ones — receives the ungated Hebbian weight-decay multiply
  `data *= (1 - hebbian_weight_decay)` plus any co-firing potentiation. That is the observed ~-0.1 drift
  (OU noise occasionally co-fires the fixed pre/post pair; the decay term acts every qualifying step).
- The **STDP** block (lines ~5461–5502) DOES honor `cp_synapse_plastic_mask` (line 5483:
  `cp.where(plastic_here, updated_weights, current_weights)`). That is why the `plastic=False` flag *appears*
  to "freeze" populations in the existing G2+ runners — those use STDP, not Hebbian. Under **Hebbian**, the
  flag is silently a no-op for weight freezing.

In short: `plastic=False` isolates under STDP but NOT under global Hebbian learning. The merge cannot rely on
it.

## Fallback adopted (no `sim/` edit) — per-synapse plasticity gate

The plan's specified fallback is implemented and makes the assertion pass: gate the FIXED population's
synapses with the per-synapse plasticity gate `cp_plasticity_rate_gain` set to 0.0.

This is done entirely from outside `sim/` via the existing public API:

1. Tag the fixed population with `"plasticity_gate": "<name>"` in the `inject_explicit_wiring` plan. That
   makes `inject_explicit_wiring` allocate `cp_plasticity_rate_gain` (1.0 everywhere) and register the
   gate→synapse-index map.
2. Call `bridge.set_plasticity_gate("<name>", 0.0)` after wiring. This zeros the gain over exactly the fixed
   synapses.

With gain=0 on the fixed synapses, the Hebbian path multiplies BOTH the potentiation delta (line 5397) AND
the decay term (line 5411) by 0 → the fixed weights are truly frozen. The plastic 'parse' synapse is left at
gain=1.0 (it is not tagged), so the control stays non-vacuous.

After the fallback:
```
FIXED 'bind' weight: before = 320.0   after = 320.0   (np.array_equal == True)
PLASTIC 'parse' weight: changed (control holds)
```

## Implication for the downstream merge (Tasks 2–6)

`UnifiedBrainBridge` and the parameterized `CoreSimComposer` MUST gate the composer's `"bind"` population
with a plasticity gate set to 0.0 (the `plastic=False` flag alone is insufficient on a Hebbian-enabled shared
bridge). The parser's `"parse"` population stays ungated (plastic, gain 1.0). No protected-module (`sim/`)
edit is required — the gate is settable from the runner.

**Verdict:** isolation HOLDS, but via the `cp_plasticity_rate_gain` per-synapse gate, NOT via the
`plastic=False` flag. Documented honestly; the flag's Hebbian no-op is a real characterization of the
substrate, not a workaround to hide.
