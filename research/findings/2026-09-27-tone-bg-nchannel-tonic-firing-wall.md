---
type: finding
status: live
lane: load-bearing
date: 2026-09-27
---

# N=5 tonic-firing wall in the N-channel basal-ganglia race (2026-09-27)

**Status:** WALL on a METHOD (the N-way cross-inhibition generalisation), NOT a license to abandon the CAPABILITY
(a spiking N-way winner-selection race). No-defer: the capability is kept; the method is banked and a new spiking
method is the next step.

## The wall

The N-channel generalisation of the two-channel basal-ganglia action selector
(`research/runners/_tone_bg_nchannel_selector.py`, the named next rung of the tone-selection seam —
`webapp.affect_tone_selection.select_by_mood`) commits cleanly at **N=2** (winner-take-all: one motor pool crosses
`commit_threshold_spikes` with the other channel suppressed below the clean-loser ratio) but **NOT at N=5**: every
commit pool fires tonically (~9000 spikes each) regardless of salience, so the race never produces a clean winner
(`winner is None`, `committed is False`). The host-share probe's substitution arm records this directly
(`research/findings/raw/_draft_host_share_tone_probe/dev_seed7_smoke.json`: `n_not_committed: 7` across the v_held
grid, `n_committed: 0`).

## What was ruled out (N=2-vs-N=5 per-region firing comparison)

The tonic floor is NOT driven by any single input the two-channel mechanism uses; each was toggled and the floor
persisted:
- **NMDA** (`commit_enable_nmda` on/off) — no change in the tonic floor.
- **Internal recurrence** (`commit_internal_density`, `commit_internal_weight` → 0) — no change.
- **Thalamic drive** (thalamic tonic) — the floor persists.
- **Salience** (zero salience vs real congruence saliences) — the floor persists.

The differentiator is the **N-way mutual cross-inhibition loop**: at N=2 the single other channel is suppressed
below the clean-loser ratio; at N=5 the cross-inhibition (`commit_fs_c -> commit_{every other channel}`) fails to
suppress ANY channel below the threshold, so no channel wins.

## The N=2 contrast

At N=2 the topology and commit read are byte-identical to the two-channel production selector
(`bg_action_selection_production_organ.py`); the race commits cleanly. The N=5 case is where the faithful N-way
generalisation breaks.

## Next step (no-defer)

A new spiking method for the N-way race. The mutual cross-inhibition weights / commit-pool wiring is the suspect:
the N-way inhibition must be strong enough (or the commit threshold / clean-loser ratio tuned) to suppress N-1
losers simultaneously, not just one. This is a wiring fix, not a capability abandonment. The host-share probe
(`research/runners/_draft_host_share_tone_probe.py`) keeps the wall in view: its substitution arm is recorded but
not gated on, so the moment the N-way race commits cleanly the probe's `n_committed` will rise and the seam's
host-share verdict can be re-read against a neural winner.
