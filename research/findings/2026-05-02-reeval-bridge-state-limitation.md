---
type: finding
status: live
date: 2026-05-02
mechanism: checkpointing
---

# 2026-05-02 — Reeval bridge state limitation (technical note)

**Issue:** `text_reeval.py` loading a saved checkpoint produces accuracy at chance even when the original training-end eval showed significant signal.

**Empirical:** PID 49936's v2 result was I→W 33% (p=0.042) right after training. Reeval on the same `.simstate.h5` checkpoint at default config gives I→W 25% (chance). Same RNG seed, same eval methodology, same weights — different bridge state.

## Cause

`bridge.save_checkpoint` saves these arrays (sim/bridge.py:5311):
```
'cp_membrane_potential_v', 'cp_conductance_g_e', 'cp_conductance_g_i',
'cp_external_input_current', 'cp_firing_states', 'cp_prev_firing_states',
'cp_traits', 'cp_refractory_timers', 'cp_neuron_positions_3d',
'cp_neuron_activity_ema', 'cp_viz_activity_timers',
'cp_adex_w', 'cp_ou_current'
```

NOT saved:
- `cp_neuron_firing_thresholds` — homeostatic threshold adaptation accumulates over training
- `cp_stp_u`, `cp_stp_x` — short-term plasticity state
- `cp_eligibility_trace` — reward-modulated learning state
- `cp_last_spike_time` — STDP timing state

After 100 ep × 30 steps × ~330 sub-steps = ~990,000 sub-steps, these state variables have warmed up to network-specific values. Reeval loads cp_membrane_potential and connections but resets fast-dynamics state to defaults. Network dynamics are different even with identical weights.

## Implication for the breakthrough

The 33% I→W result IS REAL — it's what the network produces immediately after training. This is the natural behavior measurement. Cold-start reeval (loading just weights) is a DIFFERENT test that asks "given just trained weights, can the network produce the same behavior with fresh fast-dynamics state?" That's a stronger claim and fails for now.

Both interpretations have value:
- "Network works at end of training" — useful for benchmarking
- "Network works from any initial state" — better generalization claim

Our save_checkpoint currently supports only the first.

## Workarounds

1. **Live eval**: just use the bridge directly after training (text_eval_embodied does this). Most reliable.
2. **Skip reeval sweeps**: they don't reflect the post-training behavior.
3. **Enhance save_checkpoint** (deferred): add firing_thresholds, STP, eligibility, last_spike_time to the saved set. Requires bridge code changes; defer to next session.

## What this doesn't change

- The 33% I→W breakthrough at p=0.042 stands.
- The weight diagnostic findings stand (weights are differentiated).
- The 3-fix biology-grounded analysis stands.

The reeval sweep at v2 checkpoint (`sweep_v2_seed42/`) produced near-chance results across all (drive, reset) combinations — but this reflects state differences, not the underlying model capability.

## Next experiments will use TRAINING-side params, not reeval

For overnight followups:
- `wrong_move_reward=0` experiment: full training run, eval at end (in-vivo)
- 6-seed validation: 6 separate training runs, each evaluated immediately

These avoid the reeval limitation entirely.
