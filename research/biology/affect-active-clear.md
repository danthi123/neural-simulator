---
type: biology
id: affect-active-clear
mechanism: Active-clear / transient quench of a saturated bistable affect attractor — an open-loop basin-switch (a spiking inhibitory gate), NOT a standing outward brake
status: established
last_verified: 2026-08-01
current_finding: research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md
current_status: "BRAIN-BASED GO (run_smoke 6/6, operating region + overshoot mapped; no sim/ edit). The affect 'mood ratchet' is a SATURATED BISTABLE slow-NMDA attractor; every OUTWARD BRAKE fails structurally (GABA_B too weak to cross the basin boundary; STP annihilates AND blocks re-ignition across tau_d 50-4000 x U 0.01-0.15). The mechanism that works is an ACTIVE, TRANSIENT, OPEN-LOOP CLEAR: a spiking quench_fs FS pool fires strong GABA_A onto the affect pools for a window that exceeds the ~180 ms full-drain threshold, the reverberation collapses, and the OFF (down-state) fixed point then HOLDS the state with ZERO standing force. Persistence in normal operation is untouched (the clear is OFF then) and re-ignition survives (the synapses are left recovered, unlike STP). Operating region (6 seeds): drive 150-400 pA x window 200-280 ms x GABA_A weight 15-20 -> G1 evict 0.000, G3 re-ignite ~1.0, G4 persist 0.642. OVERSHOOT boundary at weight 25 (a too-strong shunt tips the opponent latch into the V- attractor) -- a real operating-region property a raw current clamp lacks. LOAD-BEARING ANTI-CHEAT: the quencher is SILENT at the read window (quench_fs rate 0.0, neuromodulator conc 0.0) while firing ~332 Hz DURING the clear -> a genuine basin-switch, not GABA_B-style current subtraction. REMAINING: the full run_battery (running) and 6 seeds (this is run_smoke 6/6)."
sources:
  - path: "doi:10.1093/cercor/10.9.910 (Compte, Brunel, Goldman-Rakic & Wang 2000, Cereb. Cortex — spatial working-memory network model)"
    anchor: "termination by nonspecific excitatory input recruiting feedback inhibition"
    note: "EXTERNAL (recorded for local addition). The canonical termination mechanism of a persistent-activity bump attractor: a strong nonspecific input recruits the network's broad feedback inhibition and drives the selective population back to its spontaneous/down state. Our quench_fs GABA_A pulse is the spiking realization."
  - path: "doi:10.1162/089976606775093909 (O'Reilly & Frank 2006 — PBWM gated working memory)"
    anchor: "a separate phasic gate flushes maintenance rather than a standing force grading it"
    note: "EXTERNAL. Robust-AND-flexible WM comes from a SEPARATE phasic gating signal (BG->thalamus->cortex) that clears the loop, not from a continuous brake — exactly why an open-loop clear works where a standing brake cannot."
  - path: "doi:10.1016/j.biopsych.2008.05.015 (Durstewitz & Seamans 2008, Biol. Psychiatry — dual-state theory of PFC dopamine)"
    anchor: "high vs low energy barrier: robust maintenance vs flexible switching"
    note: "EXTERNAL. The graded/neuromodulatory sibling of the all-or-none clear (mechanism #2 in the deep-research shortlist): the SAME network holds or clears depending on modulatory gain, resolving the persistence-vs-evictability tension by toggling the landscape."
implemented_by:
  - research/runners/_affect_eviction_derisk.py
findings:
  - research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md
  - research/findings/2026-08-01-affect-active-clear-quench-gate-evicts-the-ratchet-6seed-physics-GO-host-shortcut.md
  - research/findings/2026-08-01-affect-ratchet-STP-annihilates-across-all-tau_d-and-U-the-loop-is-bistable-brakes-cannot-grade-evict.md
  - research/findings/2026-07-31-affect-ratchet-the-eviction-processes-were-SWITCHED-OFF-not-missing.md
---

# Active-clear of the affect ratchet — a spiking basin-switch, not a brake

**What is measured:** the affect "mood ratchet" (a saturated bistable slow-NMDA opponent attractor that rises
and never returns to baseline) is cleared by a TRANSIENT, OPEN-LOOP spiking gate — a dedicated `quench_fs` FS
interneuron pool firing strong GABA_A onto the affect pools during post-drive silence. Brain-based (no `sim/`
edit), 6/6 seeds across a mapped operating region.

## Why a brake fails and a clear works (the reframe)

A brake must HOLD a bistable loop down continuously, fighting the attractor; it is either too weak to cross the
ON-basin boundary (no eviction) or crosses it and annihilates (destroying re-ignition). An active clear only
pushes the state across the boundary ONCE; the loop's OTHER stable fixed point (OFF / down-state) then holds it
with zero standing force. Bistability, which killed every brake, is the ASSET that makes the clear hold.

## The two properties that distinguish it from the failed GABA_B brake

1. **Zero standing force at readout** — the quencher is silent at the read window (measured + asserted). A
   held-low read with no standing inhibition is a genuine basin-switch, not current subtraction.
2. **A ~180 ms full-drain threshold + an overshoot ceiling** — the window must exceed the drain time (longer
   than the ~100 ms NMDA decay; residual recovery/adaptation drains too) or the loop re-ignites from OU noise;
   and the GABA_A weight must stay below ~25 or the shunt overshoots the neutral OFF basin into the V-
   attractor. This calibrated window is a real operating-region property the host current clamp could not show.

No `constraints_config` is bound here: the operating region (weight 15-20, window 200-280 ms) is an EMPIRICAL
calibration on this substrate, not a biology-REQUIRED constant, so binding it as a hard config requirement
would over-claim — the same discipline as the deep-credit-on-spikes entry.
