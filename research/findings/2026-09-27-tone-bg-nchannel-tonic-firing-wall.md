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

## The combination probe + motor-pool correction: the salience gain is NOT the lever (2026-09-28)

The combination probe (`research/runners/_tone_bg_nchannel_combination.py`, dev seed 7) ran the race in the
sub-threshold window (gpi=50, thal=45) across the salience gain (600→19200). It reported the motor pools as
"near-silent (7-16)" — but that was an **artifact of the early break**: `_run_nchannel_trial` stops the run the
instant a channel crosses `commit_threshold_spikes=12`, so it only ever reports the count *at the break point*. The
motor-pool tonic sweep (`research/runners/_tone_bg_nchannel_motor_tonic_sweep.py`, raw
`research/findings/raw/_motor_tonic_sweep_seed7.log`) measures the MOTOR pool (the decision pool) over the FULL run
(no early break) and shows the decision pools actually fire **~1100-2100** spikes at the sub-threshold tonic — not
7-16. The combination probe measured the wrong quantity.

The motor-pool gain sweep (`research/runners/_tone_bg_nchannel_motor_gain_sweep.py`, raw
`research/findings/raw/_motor_gain_sweep_seed7.log`) then tested the salience gain directly on the full-run motor
pool at the sub-threshold tonic (v_held=+0.9, saliences [0.55, 0.9, 0.7, 0.4, 0.2], expected winner cand 1):

| gain (pA) | motor (cand 0..4) | commit (cand 0..4) | argmax | max/2nd |
|-----------|-------------------|--------------------|--------|---------|
| 600       | [2138, 1632, 717, 2035, 1136] | [8075, 8075, 8055, 8101, 8072] | 0 (no) | 1.05 |
| 1200      | [2138, 1632, 717, 2035, 1136] | [8075, 8075, 8055, 8101, 8072] | 0 (no) | 1.05 |
| 2400      | [2149, 1632, 717, 2040, 1136] | [8069, 8075, 8055, 8104, 8072] | 0 (no) | 1.05 |
| 4800      | [2149, 1632, 717, 2040, 1136] | [8069, 8075, 8055, 8104, 8073] | 0 (no) | 1.05 |
| 9600      | [2138, 1632, 717, 2035, 1136] | [8075, 8075, 8055, 8101, 8073] | 0 (no) | 1.05 |
| 19200     | [2138, 1632, 717, 2035, 1136] | [8075, 8075, 8055, 8101, 8073] | 0 (no) | 1.05 |
| 38400     | [2138, 1632, 717, 2035, 1136] | [8075, 8075, 8055, 8101, 8072] | 0 (no) | 1.05 |

Across a **64x salience-gain range** the motor pools vary by at most 11 spikes (0.5%, noise) and the commit pools by
≤9. The motor pools do NOT track the saliences (cand 0, salience 0.55, fires the most; cand 1, salience 0.9, the
expected winner, fires second; cand 2, salience 0.7, fires the least). **The salience gain is decisively not the
lever** — the D1→GPi→thal→commit→motor pathway fully decouples the salience bias from the decision pools.

## Pathway trace: the D1 differentiates perfectly, but the GPi is near its floor (2026-09-28)

The pathway trace (`research/runners/_tone_bg_nchannel_pathway_trace.py`, dev seed 7, raw
`research/findings/raw/_pathway_trace_seed7.log`) measures the full-run firing at EVERY pathway step (D1, GPi,
thalamus, commit, motor) at the sub-threshold tonic (gpi=50, thal=45), v_held=+0.9, default gain, saliences
[0.55, 0.9, 0.7, 0.4, 0.2] (expected winner cand 1):

| step | cand 0 | cand 1 | cand 2 | cand 3 | cand 4 | max/min | argmax |
|------|--------|--------|--------|--------|--------|---------|--------|
| D1 (salience bias applied here) | 452 | **913** | 702 | 417 | 210 | **4.35** | **1 (expected)** |
| GPi (D1 inhibits GPi) | 8 | 17 | 19 | 17 | 28 | 3.50 | 4 |
| thalamus (GPi inhibits thal) | 94 | 83 | 82 | 80 | 87 | 1.18 | 0 |
| commit pool (thal drives commit) | 8075 | 8075 | 8055 | 8101 | 8072 | 1.01 | 3 |
| motor pool (commit drives motor; DECISION) | 2138 | 1632 | 717 | 2035 | 1136 | 2.98 | 0 |

Three findings:

1. **The D1 pool differentiates PERFECTLY by salience.** D1=[452, 913, 702, 417, 210] tracks the saliences
   monotonically (cand 1, sal 0.9 → 913; cand 4, sal 0.2 → 210) and the argmax is cand 1 (the expected winner). The
   salience bias is working correctly at the D1 pool.

2. **The GPi is near its floor (nearly silent), so its differentiation is too small to modulate the thalamus.**
   GPi=[8, 17, 19, 17, 28] — only 8-28 spikes over 600 steps (~0.013-0.047 spikes/step/neuron across 20 neurons).
   The D1 inhibition is strong enough to nearly silence the GPi. The GPi's residual differentiation (8 vs 28) is a
   small modulation of a near-silent pool. The thalamus (thal=[94, 83, 82, 80, 87], max/min=1.18) fires flat at its
   tonic rate (driven by the thalamic tonic, 45 pA), regardless of the GPi. **The GPi→thalamus step is where the
   differentiation vanishes.**

3. **The commit/motor pools are flat (pinned by the thalamic tonic).** commit=[8075, 8075, 8055, 8101, 8072]
   (max/min=1.01) and motor=[2138, 1632, 717, 2035, 1136] (max/min=2.98, but the argmax is cand 0, NOT the expected
   cand 1). The motor pool's "differentiation" is pure noise — it does not track the saliences.

This localises the wall precisely: the salience bias works correctly at the D1 pool, but the **GPi's unfaithful
near-silence** (the real GPi fires at a high tonic rate, ~20-50 Hz, and the D1/D2 pathways modulate it) prevents the
D1 differentiation from propagating to the thalamus/commit/motor. The fix is the **GPi's operating point**, not the
salience gain or the tonic floor.

## Next step (no-defer)

The GPi is near its floor because the D1 inhibition is too strong relative to the GPi's tonic/STN drive. In the real
basal ganglia the GPi fires at a HIGH tonic rate and the D1/D2 pathways modulate it (reduce/increase); the model's
GPi is unfaithfully silent. The lever is to **raise the GPi's tonic rate** (raise `gpi_tonic_pA`, or strengthen the
STN→GPi drive) so the GPi fires at a higher rate and its D1-driven differentiation becomes a larger modulation of the
thalamus. A cheap CPU probe sweeping `gpi_tonic_pA` (50→1000 pA) with the thalamic tonic fixed at 45 pA, measuring
the GPi/thal/commit/motor pools at v_held=+0.9, will show whether a higher GPi tonic propagates the D1
differentiation to the decision pools. If it does, the no-defer wiring fix is to raise the GPi tonic into the
range where the D1 differentiation propagates. If it does not, the suspect moves to the GPi→thalamus synapse weight
(the GPi's inhibition is too weak to modulate the thalamus even when the GPi fires). Either way the capability (a
spiking N-way winner-selection race) is kept. The host-share probe
(`research/runners/_draft_host_share_tone_probe.py`) keeps the wall in view: its substitution arm is recorded but not
gated on, so the moment the N-way race commits cleanly the probe's `n_committed` will rise and the seam's host-share
verdict can be re-read against a neural winner.
