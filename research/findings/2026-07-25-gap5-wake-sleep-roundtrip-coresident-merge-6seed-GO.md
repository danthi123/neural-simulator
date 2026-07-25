# gap#5 one-brain MERGE — the full WAKE→SLEEP→WAKE round-trip on a CO-RESIDENT bridge is 6-SEED GO: a conversational slice + the CA3 replay slice on one bridge, the conversational memory survives the cycle byte-identical + the replay travels in sleep, with two precisely-isolated integration requirements (2026-07-25)

## Headline
The one-brain replay merge is now validated **end-to-end on a co-resident bridge**. ONE `SimulationBridge` co-hosts a
CONVERSATIONAL slice (Izhikevich, plastic recurrent — it learns weights during WAKE via STDP) + the CA3 REPLAY slice (the
place-field track + forward-biased band). The full cycle — WAKE(Izh/dt1.0, learn) → SLEEP(AdEx/dt0.1, replay) →
WAKE(Izh/dt1.0) — is **6-seed GO**: the conversational memory (conv-slice weights) survives the round-trip **byte-identical
6/6**, and the CA3 replay **travels in sleep DECODE_r=1.000 6/6**. This completes the merge de-risk: the biology-faithful
wake/sleep phase-switch works with real co-resident conversational memory present.

## Result — 6-SEED GO (seeds 42/43/44/100/101/102)
- **sleep-replay DECODE_r [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]** — the CA3 replay travels in the sleep phase 6/6.
- **conv-memory-preserved 6/6** — the conversational slice's learned weights are byte-identical after the full round-trip
  (the phase-switches never touch `cp_connections`; the reverse AdEx→Izhikevich switch restores v/u + model/dt + cached
  decays).

## Two precisely-isolated integration requirements (the diagnostic ladder, seed 42)
Getting here required isolating two subtle carryover bugs (the diagnostic block reproduces both):
1. **Transient synaptic state must reset on each phase onset.** The phase-switch resets v/adex_w but NOT the synaptic
   conductances (g_e/g_i/NMDA) or STP state; residual conductance from the prior phase leaks into the next. Fix: a
   `reset_transient_synaptic_state` on each onset (biologically, a wake↔sleep transition clears fast transient activity;
   the MEMORY/weights persist). (Necessary but not sufficient.)
2. **The wake STDP must be FROZEN during the sleep/replay phase.** The decisive isolation: an IDLE wake (400 steps, no
   drive) → sleep replay WORKS (DECODE_r 1.000); a DRIVEN wake (conv fires) → sleep replay FAILS (0.000, pc barely
   fires). The band stays intact at 599 throughout (NOT a weight collapse), and freezing STDP before sleep restores the
   replay (0.000→1.000). ⇒ with the wake STDP rule left running during the AdEx replay, prior-wake conv FIRING leaves a
   plasticity-path state that suppresses the sleep replay (mechanism: STDP-active + prior-wake-firing; band unchanged).
   Fix: freeze `enable_stdp` for the sleep phase, thaw on wake — biologically sensible (the replay phase runs a
   controlled consolidation plasticity, not the raw wake STDP). (This is the load-bearing fix.)

## What this establishes for the merge
The full one-brain replay merge is de-risked end-to-end: nav/conversational Izhikevich slices + an AdEx CA3-replay slice
co-resident on ONE bridge; WAKE runs Izhikevich/dt1.0 (conversation, replay slice quiescent), a SLEEP/SWR phase switches
to AdEx/dt0.1 + frozen-STDP and the replay travels (for consolidation/imagination), then WAKE resumes — the conversational
memory preserved byte-identical across the cycle. This sidesteps the per-region-neuron-model + dt-stiffness walls by
exploiting the temporal separation of replay and conversation (as in the brain). The remaining step to full gap#5 closure
is wiring this into the ACTUAL production conversational agent (the OneBrainComposer / MergedNavConvAgent — which adds the
RF composer's neuron model to the mix) + a sleep-phase trigger; the mechanism is proven.

## HONEST SCOPE
- The conversational slice here is a generic Izhikevich associative memory standing in for the real conversational brain
  (parser + RF composer). The mechanism (phase-switch + memory preservation + STDP-freeze-in-sleep) is model-general for
  the Izhikevich↔AdEx switch; the production agent adds RF regions whose model handling is the next integration detail.
- A `sim/` `switch_neuron_model_phase()` + `reset_transient_state()` capability (additive, default-off, CI-guarded) would
  make the wake/sleep switch a first-class method rather than a runner-side replication — a clean follow-on once wired
  into the production agent.

## Provenance
`research/runners/_gap5_wake_sleep_roundtrip.py` (diagnostic ladder A/B/C/G + the 6-seed; logs `roundtrip*.log`). Reuses
`_gap5_wake_sleep_phase_switch` (`switch_to_adex_sleep`, `50255443`) + the replay decoder (`decode_and_width`, `d6e140bf`)
+ the ECKER_CA3_PC preset (`d707bf34`). NO `sim/` edit. GPU. Builds on the gap#5 replay/learned-band/merge arc
(`d6e140bf`, `a051d84d`, `6ed6f0a2`, `50255443`).
