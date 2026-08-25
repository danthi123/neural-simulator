---
type: finding
status: contributing
date: 2026-08-26
mechanism: four-day-window-negatives-roundup
lane: mixed
claim_check: synthesis
seeds: [42, 43, 44, 100, 101, 102]
---

# Four-day window negatives roundup: each boundary + its next mechanism (no-defer)

Four de-risks came back negative/partial over the free-compute window. Each is a verdict on a METHOD; the next
mechanism is named so no capability is deferred. Aggregate (gate-passing, per-arc source artifacts listed inside):
`research/findings/raw/_harvest_2026_08_26/window_negatives_agg.json`.

## 1. Source-monitoring (#129) — NEGATIVE across variants
The competing mechanisms for tagging WHERE a memory came from all fell short of clearing the source margins AND
the preregistered no-harm control on all 6 seeds:
- `_laneC_plastic_source_memory_derisk_6seed.json` -> NEGATIVE.
- `laneC_source_competitive_decisive.json` -> aggregate 5/6 pass, `go: false` (one
  seed fails the no-harm control — the exact failure the board's CURRENT STATE already flagged for v2 competitive).
- `_laneC_source_monitor_attractor_competition_6seed.json` -> margins do not clear.
Next mechanism: the source-tag must be a CONJUNCTIVE binding (source-context x content) that cannot leak into the
content read — the conjunctive-tag / attractor-joint variants staged on the pool are the next reads; if they also
fail the no-harm control, a dedicated source-context sub-population bound at encoding (not a competition over the
shared content assembly) is the rung.

## 2. Replay order-consolidation (#130) — CALIBRATION_NEEDS_REVISION
`replay_v6_order_stdp_calib.json` -> `calibration_status:
CALIBRATION_NEEDS_REVISION` (still in the calibration phase; it is seed-locked to its calibration seeds). The
gate did not reach a decisive order-vs-shuffled comparison. Remaining host scaffolds it names: wake episode
populations + partial probe cues, opponent inhibitory-channel membership fixed from calibration assemblies, and a
host-scheduled sleep. Next mechanism: revise the calibration (the order-STDP timing window + the assembly
membership) so the order signal survives into a decisive run, THEN run the 6-seed order-vs-shuffled contrast;
demote the host-fixed inhibitory membership to a learned opponent channel.

## 3. Memory self-ignition (gap5) — 1/6, with a positive ignition-sweep lead
`gap5_dg_ignition_6seed.json` -> `GO: false`, n_go 1/6: the DG-detonator ignites on
this substrate (the symmetric positive control fires 2/6) but the DECOUPLED store does not cleanly
ignite-discretely-AND-transition-forward. The POSITIVE lead is the ignition sweep:
`gap5_ignition_sweep_6seed.json` -> `IGNITES-YES` — the readout CAN ignite
discretely at strong drive (26 of 96 configs reach ev>=1), though 0 are "clean" (specific AND low detonator-frac).
Next mechanism: read the igniting configs out of the sweep to set the decoupled-store operating point (the k-of-N
formation floor + drive strength that ignites specifically), then re-run self-ignition at that operating point —
the boundary is an unset operating point, not an absent capability.

## 4. DA-encoding lever-2 homeostat — UNDEFINED
`soak_homeostatic.json` -> `status: UNDEFINED`: the STRESS precondition
(recall_ON >= recall_OFF at EVERY swept sigma — salience redistribution net-neutral-or-positive over the
realistic DA distribution) was not met on all sigmas, so the flip gate is undefined rather than a clean pass or
fail. The lever-2 homeostat (a host multiplicative-scaling clamp) is a PROXY for the homeostatic process real
synapses run alongside potentiation. Next mechanism: replace the host clamp with an on-substrate spiking
synaptic-scaling rule (a real homeostatic-plasticity update on the synapses, Turrigiano-style, emergent from
activity) — a new runner — so the companion process is run by the brain, not proxied by a host bound. The WIP
homeostat is merged default-OFF (byte-identical) as the starting point for that build.
