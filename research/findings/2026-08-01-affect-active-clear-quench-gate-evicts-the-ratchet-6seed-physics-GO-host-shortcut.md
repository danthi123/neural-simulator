---
type: finding
status: live
date: 2026-08-01
mechanism: affect-state-region
artifacts:
  - research/findings/raw/affect/quench_sweep/_quench_operating_region_aggregate.json
  - research/findings/raw/affect/quench_sweep/
---

# Active-clear / quench gate EVICTS the affect ratchet — 6-seed physics GO (host shortcut), with the OFF state holding at zero standing force

**One-line verdict:** the wall that the whole outward-brake class could not cross (GABA_B too weak, STP
annihilates — finding on the bistable ratchet) is surpassed by a **transient open-loop clear**. A strong
brief negative current to the affect pools during post-drive silence collapses the reverberation; the OFF
state then holds it with **zero standing force**. 6/6 seeds in the operating region, anti-cheat clean. This is
a PHYSICS de-risk via a documented host shortcut; the brain-based spiking version is the owed next build.

## The mechanism and why it works where a brake cannot

A brake must *hold* a bistable loop down continuously and either can't cross the ON-basin boundary (no
eviction) or crosses it and annihilates (blocking re-ignition — the STP failure). An **active clear** only has
to push the state across the boundary **once, transiently**; the loop's other stable fixed point (OFF /
down-state) then catches and holds it with no standing force. Implemented in
`research/runners/_affect_eviction_derisk.py` (`--quench-pA`/`--quench-ms`, committed `4ff27661`): the clear
fires only in the POST-drive silence of LOW episodes (a lower appraisal has arrived), then is OFF for the rest
of the post gap and the entire read window.

## Operating region — 6 seeds {43,44,100,101,102,105}

Aggregate: `research/findings/raw/affect/quench_sweep/_quench_operating_region_aggregate.json` (36 per-cell
smokes in the same directory).

| quench_pA | quench_ms | EVICT+REIG | mean G1 evict | mean G3 re-ignite | mean G4 persist |
|---|---|---|---|---|---|
| −400, −800, −1600 | 150 | 0/6 | ~1.0 (ratchet holds) | ~1.0 | 0.638 |
| −400 | 280 | 0/6 | 1.044 | ~1.0 | 0.638 |
| **−800** | **280** | **6/6** | **0.000** | 1.004 | 0.638 |
| **−1600** | **280** | **6/6** | **0.000** | 1.006 | 0.638 |

**Both knobs are binding and separable.** Duration is the primary gate: at 150 ms every amplitude fails
(0/6) — the loop re-ignites from OU noise the moment the clear releases. At 280 ms the OFF state holds, but
only if the amplitude is strong enough to silence the pool (−400 fails, −800/−1600 succeed 6/6). G4
persistence is **0.638 in every cell** — untouched, because it is read on episode 0 before any clear fires.

## The load-bearing anti-cheat — this is not GABA_B-style current subtraction

The quench current is **measured** at every read window (`_affect_ext_current`) and **asserted == 0** (a hard
assert; all 36 cells completed rc=0, so it held everywhere). Held-low with **zero standing current** is a
genuine basin-switch, not a constant offset holding the rate down — the precise failure GABA_B had. G3
re-ignition ≈1.0 confirms the attractor is intact (not annihilated like STP). G6 lesion = quench-OFF returns
the ratchet on the same substrate.

## Physics refinement of the prior "bistable" call

The earlier finding called the loop bistable; the duration threshold sharpens it: the OFF basin is **too
shallow to survive OU noise unless the loop is fully drained (~180 ms**, longer than the ~100 ms
recurrent-NMDA decay estimate — residual recovery/adaptation state also drains). Once drained, OFF is a
genuine stable fixed point (held low >1.5 s at zero force in the physics smoke). So: **monostable-ON with a
shallow OFF basin**, and the clear works by draining fully into it.

## Honest scope — what this is and is NOT

- **A HOST SHORTCUT.** The clear command is a host-injected `cp_external_input_current`, not a spiking
  circuit. The brain-based deliverable (per brain-based-only) is a dedicated spiking `quench_fs` inhibitory
  pool wired to the affect pools via GABA_A, driven by a phasic neuromodulator gate. That is the next build;
  this de-risk proves the physics the brain-based version must reproduce.
- **A `run_smoke` GO, not the full `run_battery`.** 6-seed on the smoke's LOW/HIGH protocol with the operating
  region mapped; the full pre-registered battery + a brain-based instrument control are owed before "GO" is
  claimed for the *capability* rather than the *physics*.
- **Amplitude is on a host current scale** (−800 pA), not yet calibrated to a spiking inhibitory drive.

## Next
(1) Convert to the brain-based spiking `quench_fs` + neuromodulator gate — the real deliverable. (2) Full
6-seed battery on the brain-based version. (3) Log the external sources (Compte-Wang persistent-activity
termination; O'Reilly-Frank gated-WM) into `research/biology/` with resolving quotes. This closes the affect
evictor as a mechanism-exists result and re-points the work at biologizing it.
