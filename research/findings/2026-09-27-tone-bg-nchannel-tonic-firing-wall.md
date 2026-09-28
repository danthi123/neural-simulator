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

## Tonic-floor sweep: the floor is the saturation driver, but lowering it alone does not close the wall (2026-09-28)

The sweep (`research/runners/_tone_bg_nchannel_tonic_sweep.py`, dev seed 7, numpy/CPU, raw
`research/findings/raw/_tonic_sweep_seed7.log`) scales the tonic floor (gpi_tonic + thalamus_tonic together — the
driver the commit-saturation diagnosis flagged) across 7 levels and reports the commit pool's per-channel firing at
v_held=+0.9 plus how many of the 7 v_held levels commit cleanly:

| scale | gpi / thal (pA) | commit firing at v_held=+0.9 (cand 0..4) | committed |
|-------|-----------------|-------------------------------------------|-----------|
| 1.0   | 1000 / 900      | [9000, 9000, 9000, 9000, 9000]            | 0/7       |
| 0.5   | 500 / 450       | [9000, 9000, 9000, 9000, 9000]            | 0/7       |
| 0.25  | 250 / 225       | [9000, 9000, 9000, 9000, 9000]            | 0/7       |
| 0.1   | 100 / 90        | [9000, 9000, 8995, 9000, 9000]            | 0/7       |
| 0.05  | 50 / 45         | [8075, 8075, 8055, 8101, 8072]            | 0/7       |
| 0.025 | 25 / 22.5       | [0, 0, 0, 0, 0]                           | 0/7       |
| 0.01  | 10 / 9          | [0, 0, 0, 0, 0]                           | 0/7       |

Two findings:

1. **The tonic floor is the saturation driver (mechanism localised).** At scale >= 0.1 the commit pools are pinned at
   exactly 9000 (the 50% tonic ceiling); at scale <= 0.025 they are silent (0). There is a single sub-threshold
   window at scale ~= 0.05 (gpi=50, thal=45) where the pools fire sub-threshold (~8000) and DO differentiate. This
   confirms the commit-saturation diagnosis: the absolute firing level of the commit pools is set by the tonic
   operating point, not by the salience bias.

2. **Lowering the tonic floor alone does NOT close the wall (H3 fails).** In the sole sub-threshold window
   (scale 0.05) the pools differentiate by only 46 spikes (8055-8101, a 0.57% spread) — at the noise level of a
   spiking network — and the apparent max is cand 3 (v=-0.3), NOT the salience-favored cand 1 (v=0.7, closest to
   v_held=+0.9). The race still commits 0/7. So the salience bias (SALIENCE_GAIN_PA) is far too weak to break the
   N-way tie even when the tonic saturation is removed.

This refines (does not contradict) the "What was ruled out" section above: that section localised the N=2-vs-N=5
*difference* to the N-way cross-inhibition (no single input, removed, makes N=5 behave like N=2); the sweep now
localises the absolute *saturation* to the tonic operating point, and shows the cross-inhibition/salience-contrast
problem persists even in the sub-threshold window. (The earlier "thalamic tonic — the floor persists" held within the
range it tested; the sweep extends the tonic toggle to 5% and finds the sub-threshold window.)

## Next step (no-defer)

The untested lever is the **combination**: run the race in the sub-threshold tonic window (scale ~= 0.05, gpi=50,
thal=45) WITH a higher salience gain (the H1 lever, which failed only because at the default tonic the pools were
saturated and the bias had no contrast to act on). A cheap CPU probe sweeping SALIENCE_GAIN_PA x {1200, 2400, 4800,
9600} at scale 0.05 will show whether the salience bias can break the tie once the pools are sub-threshold. If it
does, the no-defer wiring fix is (a) lower the tonic floor into the sub-threshold window + (b) raise the salience
gain. If it does not, the suspect moves to the D1->commit contrast attenuation (the D1 pools differentiate but the
commit pools flatten it — see the D1 diagnosis) and the fix is the D1->commit synapse, not the tonic. Either way the
capability (a spiking N-way winner-selection race) is kept. The host-share probe
(`research/runners/_draft_host_share_tone_probe.py`) keeps the wall in view: its substitution arm is recorded but not
gated on, so the moment the N-way race commits cleanly the probe's `n_committed` will rise and the seam's host-share
verdict can be re-read against a neural winner.
