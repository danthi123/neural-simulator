# gap#5 one-brain MERGE de-risk: the localized traveling replay is AdEx-substrate-SPECIFIC — generic Izhikevich (the conversational/nav substrate) produces a SPREADING FRONT at every dt (1.0→0.1), so the merge is a temporal SWR/sleep-phase running an AdEx CA3 slice, NOT concurrent co-hosting (2026-07-25)

## Question
The gap#5 replay + learned-band are 6-seed GO on an ECKER-AdEx bridge at dt=0.1. The remaining closure item is the
one-brain MERGE (co-host the replay with the Izhikevich conversational/nav brain, which runs at dt=1.0). Since verify-go
proved the travel mechanism is **band + spike-reset refractoriness** (the neg-a adaptation is inert), and Izhikevich also
has refractoriness, the cheap-first de-risk asked: does the traveling replay reproduce on **Izhikevich at dt=1.0** (the
conversational substrate's own model + timestep)? If yes → the merge is trivial (same model + dt, no per-region-model
wall). If no → the merge needs the harder path.

## Result — Izhikevich SPREADS at EVERY dt (the model, not the dt, is the blocker)
Forward-biased Gaussian band + interior cue on a 2000-neuron Izhikevich PC track, band-strength sweep, then a dt sweep
(T scaled to hold real-time = 250 ms constant):
- **dt=1.0** (w=80/160): DECODE_r −0.08/−0.25, **width ~23**, **growth +9/+13**, dec_range 85-99 — a spreading front.
- **dt=0.5**: DECODE_r +0.15/+0.20, width ~25, growth +8/+12 — still spreading.
- **dt=0.25**: DECODE_r +0.04/−0.30, width ~22-26, growth +3/+5 — still spreading.
- **dt=0.1** (the AdEx's own dt): DECODE_r +0.04/+0.24, **width ~25**, **growth +5/+10**, dec_range 99 — STILL spreads.
- NO-BAND (dt=0.1): DECODE_r −0.06, width 26 — just noise, no structure.

Izhikevich fires much more readily (F_active 0.006-0.02 vs the AdEx replay's sparse 0.001), so the band's flanks also
fire → the activity broadens into a growing front instead of a narrow moving packet. Even at the AdEx's own dt=0.1 it
spreads → **it is the neuron model, not the coarse dt.**

## Why the AdEx localizes and Izhikevich does not (mechanistic + faithful)
The ECKER CA3-PC AdEx has a **high firing threshold (V_T=−24.4)** and a slow membrane, so only the strongly-driven
leading edge of the bump crosses threshold — the flanks stay sub-threshold and silent → the packet stays razor-narrow
(width 0.8-3.5, F_active 0.001). Generic Izhikevich RS fires at a much lower effective threshold → the flanks fire too →
the bump broadens into a spreading front. ⇒ the localized traveling replay **requires CA3-pyramidal-like high-threshold
sparse-firing dynamics** — a faithful, not incidental, requirement (it is a property of the modeled cell type). This is
consistent with the earlier gap#5 finding that global/host inhibition cannot localize (localization is an intrinsic +
local-inhibition property, not a global knob).

## Verdict + the merge design (per THE LAW — a design finding, not a wall)
- **The replay capability is DONE (6-seed GO, AdEx); the merge is a design choice, and this de-risk sets it:** the merge
  is a **temporal SWR/rest phase** — an AdEx CA3-replay slice, active during a distinct "sleep/rest" phase at dt=0.1,
  temporally separate from the dt=1.0 Izhikevich conversation — **NOT concurrent same-dt co-hosting.** This is
  biologically faithful: hippocampal SWR replay occurs during rest/sleep, not during active behavior.
- **The build hits the per-region-neuron-model wall** (the global-scalar neuron-model kernel; per-region `adex_neuron_type`
  is deferred, `bridge.py:2233`). Options for the merge, cheap-first: (a) a **phase-model-switch** — the merged bridge
  switches the global model+dt to AdEx/0.1 during the replay/sleep phase and back to Izhikevich/1.0 for conversation
  (reuses the existing HH dt-auto-adjust pattern; the replay slice's neurons are allocated but the whole bridge runs
  AdEx during the phase — cheap, but the conversational neurons are AdEx-typed during the phase, which is fine if they're
  quiescent in sleep); (b) the faithful **per-region neuron-model kernel** (additive, default-off — the deferred
  capability); (c) a **separate hippocampal bridge** interacting with the cortical bridge via synaptic projections (two
  bridges = arguably not "one brain," but matches CLS's hippocampus↔cortex separation). Ranked build: (a) is the cheapest
  faithful path (sleep-phase switch), (b) the cleanest long-term.
- **NEXT:** design + build the sleep-phase merge (option a first); the neural replay-reader is the other remaining gap#5
  closure item (the Bayesian decode is a measurement instrument; a downstream neural reader is needed when replay must
  DRIVE consolidation — which connects to the roadmap's ca1→concept consolidation build).

## Provenance
`scratchpad/gap5_izh_replay.py` (logs `izh_replay{,2,3}.log`). Reuses the committed replay decoder (`decode_and_width`,
`d6e140bf`). NO `sim/` edit. GPU. Builds on the gap#5 replay GO (`d6e140bf`) + learned-band GO (`a051d84d`).
