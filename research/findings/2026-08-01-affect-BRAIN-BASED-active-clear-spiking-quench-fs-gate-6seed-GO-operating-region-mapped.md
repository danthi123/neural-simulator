---
type: finding
status: live
date: 2026-08-01
mechanism: affect-state-region
artifacts:
  - research/findings/raw/affect/brain_quench_sweep/_brain_quench_aggregate.json
  - research/findings/raw/affect/brain_quench_sweep/
---

# The affect ratchet is evicted by a BRAIN-BASED spiking clear — a `quench_fs` GABA_A gate, 6-seed GO, operating region + overshoot boundary mapped, NO `sim/` edit

**One-line verdict:** the affect evictor is now done by NEURONS. A dedicated spiking inhibitory pool
(`quench_fs`) fires GABA_A onto the affect pools and collapses the reverberatory latch; the OFF state holds.
6/6 seeds across the operating region, the anti-cheat clean (the quencher is SILENT at readout), and the
engine expressed it runner-only — the host-injected-current shortcut is removed. This closes the affect
evictor as a brain-based mechanism (per brain-based-only); the pre-registered `run_battery` is the remaining
formality.

Aggregate: `research/findings/raw/affect/brain_quench_sweep/_brain_quench_aggregate.json` (42 cells = 7 configs
× 6 seeds {43,44,100,101,102,105}; per-cell data embedded; backend numpy).

## The mechanism (neurons, not host code)

`quench_fs` = 30 FS interneurons (IZH2007_FS_CORTICAL_INTERNEURON) → strong GABA_A `RegionPathway`s onto
affect_vplus/vminus/arousal, behind the `quench_out` transmission gate, recruited during the clear window by a
phasic `quench_drive` neuromodulator (`excitability_drive` on `scope="group:quench_fs"`). The clear is
delivered ENTIRELY as synaptic inhibition from spiking neurons — nothing is written to the affect pools'
external current. **`sim/` is byte-unchanged**: `BrainRegion`, `RegionPathway(receptor="gaba_a",
transmission_gate=…)`, `set_transmission_gate`, and a manual `excitability_drive` neuromodulator on a
region-group are all pre-existing additive abstractions.

## Operating region — 6 seeds, aggregate cited above

| drive pA | ms | GABA_A w | GO | G1 evict | G3 re-ignite | G4 persist | regime |
|---|---|---|---|---|---|---|---|
| 150 / 250 / 400 | 280 | 15 | **6/6** | 0.000 | ~1.0 | 0.642 | GO |
| 250 | 200 / 240 / 280 | 15 | **6/6** | 0.000 | ~1.0 | 0.642 | GO |
| 250 | 280 | 20 | **6/6** | 0.000 | ~1.0 | 0.642 | GO |
| 250 | 280 | 25 | 4/6 | 0.097 | **0.286** | 0.642 | OVERSHOOT → V− |

The GO region is robust and wide: any drive 150–400 pA, window 200–280 ms, GABA_A weight 15–20 gives clean
eviction (G1 0.000), re-ignition surviving (G3 ~1.0), and persistence untouched (G4 0.642, read on episode 0
before any clear).

## The new biology the host clamp could not show — an overshoot boundary

The GABA_A weight is load-bearing. At **w=25** the shunt is too strong: it OVERSHOOTS the neutral OFF basin
and tips the opponent latch into the **V− (negative-valence) attractor** — G3 re-ignition collapses to 0.286
(the pool re-ignites NEGATIVE), 4/6. A calibrated shunt (w 15–20) drains cleanly to neutral. A shunting GABA_A
quench of an opponent latch therefore has an overshoot regime that a raw negative-current clamp lacks — a real
operating-region property that only the brain-based (synaptic, reversal-potential-bounded) mechanism exposes.

## The anti-cheat holds through the conversion

At every read window: affect-pool external current 0.0, `quench_drive` concentration 0.0 (hard-asserted),
`quench_fs` rate 0.0 — while `quench_fs` fires ~332 Hz DURING the clear. The `quench_out` gate is held OPEN
through the read (no gating trick). Held-low with the quencher fully SILENT at read = a genuine basin-switch,
exactly what the host de-risk proved, now with a spiking evictor.

## Scope + what remains
`run_smoke` 6/6 (the LOW/HIGH ratchet protocol), not the full pre-registered `run_battery` — that battery is
the final formality, not a new risk. Amplitudes are on the numpy CPU substrate. This completes the affect
evictor arc: brake class refuted → deep-research reframe (bistability is an asset for a transient clear) →
host-shortcut physics GO → **brain-based spiking GO**. Next: run the full battery, and log the external
sources (Compte-Wang termination; O'Reilly-Frank gated-WM; Durstewitz-Seamans) into `research/biology/`.
