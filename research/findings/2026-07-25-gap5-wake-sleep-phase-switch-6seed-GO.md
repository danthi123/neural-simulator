# gap#5 one-brain merge core mechanism — the WAKE/SLEEP neuron-model PHASE-SWITCH is 6-SEED GO: one bridge runs Izhikevich/dt1.0 (conversation) then switches to AdEx/dt0.1 (SWR/sleep) to run the CA3 replay, the memory band preserved byte-identical, replay == native-AdEx (2026-07-25)

## Headline
The one-brain merge for the AdEx-substrate-specific replay (`_gap5_izh_replay_merge_derisk.py` showed Izhikevich spreads
at every dt) is a temporal **wake/sleep phase-switch** — biologically faithful (hippocampal SWR replay happens during
rest, not active behavior). That switch is now de-risked 6-seed: ONE `SimulationBridge` runs **Izhikevich/dt=1.0** during
a WAKE phase (as the conversational/nav brain does), then switches to **AdEx/dt=0.1** for a SLEEP/SWR phase and runs the
CA3 traveling replay — with the **memory band (cp_connections) preserved byte-identical** across the switch and the
phase-switched replay **identical to a native-AdEx replay**.

## The switch (`switch_to_adex_sleep`, from the runner — NO sim/ edit)
The step loop dispatches neuron dynamics on `cfg.neuron_model_type` each step, so the switch swaps the neuron-model state
while leaving `cp_connections` (the learned memory) untouched:
1. load the ECKER_CA3_PC AdEx preset → `cfg.adex_{C,g_L,E_L,V_T,Delta_T,a,tau_w,b,V_r,V_peak}`;
2. `cfg.neuron_model_type = ADEX`, `cfg.dt_ms = 0.1`;
3. reset the AdEx state: `cp_membrane_potential_v = adex_E_L`, `cp_adex_w = 0` (a fresh sleep state);
4. **recompute the dt-dependent cached synaptic decays** (`_cached_decay_{e,i,nmda,nmda_rise,gabab}` = exp(−dt/τ)) — the
   subtle piece: they are computed once at init for dt=1.0 and would be stale for dt=0.1 (a silent-failure risk if missed);
5. `max_delay_steps = max_synaptic_delay_ms / dt`.
`cp_connections` is never touched → the memory band persists across wake↔sleep.

## Result — 6-SEED GO (seeds 42/43/44/100/101/102)
- **memory band preserved byte-identical across the switch: 6/6** (`np.array_equal(band_before, band_after)`).
- **phase-switched replay travels + decodes: DECODE_r [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], width 0.4, 6/6** — a localized
  directional traveling replay.
- **== native-AdEx reference:** a bridge built as AdEx from scratch gives the same DECODE_r=1.000, width 0.4 → the switch
  reproduces native AdEx exactly (the wake-phase Izhikevich run + the model swap leave the replay unchanged).
⇒ **VERDICT: GO** — the wake/sleep phase-switch is the de-risked, biology-faithful realization of the one-brain replay
merge.

## What this means for the merge
The full merge is now buildable on this foundation: the conversational/nav Izhikevich slices + an AdEx CA3-replay slice
(place-field track + the emergent forward-biased band, `a051d84d`) co-resident on ONE bridge; during WAKE the bridge runs
Izhikevich/dt1.0 (conversation, replay slice quiescent); during a SLEEP/SWR phase it switches to AdEx/dt0.1 and the replay
slice replays (consolidating / imagining), then switches back. The conversational neurons are AdEx-typed during sleep but
quiescent, and their synaptic memory (weights) persists byte-identical. This sidesteps BOTH walls the concurrent merge hit
(the per-region-neuron-model wall + the dt-stiffness wall) by exploiting the fact that replay and conversation are
temporally distinct — exactly as in the brain.

## HONEST SCOPE / next
- This de-risks the SWITCH on a replay-only bridge. The full merge adds the switch to the actual merged nav+conv brain
  (co-resident conversational slices) — the next build: verify the conversational weights/behavior survive a wake→sleep→wake
  cycle unchanged (they should: `cp_connections` is preserved + the conversational neurons are quiescent in sleep), and
  wire a sleep-phase trigger.
- The neural replay-READER (the Bayesian decode is a measurement instrument) is the other remaining gap#5 closure item —
  needed when replay must DRIVE consolidation (connects to the roadmap's ca1→concept consolidation build).
- A `sim/` `switch_neuron_model_phase()` capability (additive, default-off) would make the switch a first-class,
  CI-guarded method rather than a runner-side replication — a clean follow-on once the full merge is validated.

## Provenance
`research/runners/_gap5_wake_sleep_phase_switch.py` (logs `phase_switch{,6}.log`). Reuses the committed replay decoder
(`decode_and_width`, `d6e140bf`) + the ECKER_CA3_PC preset (`d707bf34`). NO `sim/` edit. GPU. Builds on the gap#5 replay
GO (`d6e140bf`), learned-band GO (`a051d84d`), merge de-risk (`6ed6f0a2`).
