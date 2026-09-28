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

## GPi-tonic sweep: raising the GPi tonic propagates the D1 differentiation to a clean N=5 winner (2026-09-28)

The pathway trace localised the wall to the GPi's unfaithful near-silence (GPi=[8, 17, 19, 17, 28] at the sub-threshold
tonic gpi=50, thal=45). The no-defer lever: raise ONLY the GPi tonic (`gpi_tonic_pA`) with the thalamic tonic fixed at
the sub-threshold value (45 pA), so the D1 inhibition has "room" to modulate a firing GPi. In the real basal ganglia the
GPi fires at a high tonic rate (~20-50 Hz) and the direct (D1) pathway reduces it for the selected action; the model's
GPi at gpi=50 is unfaithfully silent.

The sweep (`research/runners/_tone_bg_nchannel_gpi_tonic_sweep.py`, dev seed 7, numpy/CPU, raw
`research/findings/raw/_gpi_tonic_sweep_seed7.log`) measures the full-run (no early break) firing at every pathway step
across `gpi_tonic` ∈ {50, 100, 200, 400, 600, 800, 1000} pA (thal fixed at 45), v_held=+0.9, saliences
[0.55, 0.9, 0.7, 0.4, 0.2] (expected winner cand 1):

| gpi (pA) | GPi (cand 0..4) | motor (cand 0..4) | winner | max-loser/winner | clean | == expected |
|----------|-----------------|-------------------|--------|------------------|-------|-------------|
| 50       | [8, 17, 19, 17, 28] | [2138, 1632, 717, 2035, 1136] | 0 | 95.2% | no | no |
| 100      | [125, 49, 85, 190, 163] | [150, 275, 0, 0, 0] | 1 | 54.5% | no | yes |
| 200      | [338, 106, 264, 405, 434] | [94, 172, 220, 0, 0] | 2 | 78.2% | no | no |
| **400**  | [668, 323, 520, 737, 806] | [0, 164, 0, 0, 0] | **1** | **0.0%** | **yes** | **yes** |
| **600**  | [941, 505, 769, 1022, 1102] | [0, 167, 0, 0, 0] | **1** | **0.0%** | **yes** | **yes** |
| 800      | [1200, 729, 1002, 1276, 1381] | [0, 0, 0, 0, 0] | — | — | all silent | no |
| 1000     | [1442, 956, 1226, 1509, 1621] | [0, 0, 0, 0, 0] | — | — | all silent | no |

Three findings:

1. **A clean N=5 winner emerges at gpi=400-600 (thal=45), dev seed 7, v_held=+0.9.** The motor pool is [0, 164, 0, 0, 0]
   (gpi=400) and [0, 167, 0, 0, 0] (gpi=600): the expected winner (cand 1) fires cleanly and every loser is silent
   (max-loser/winner = 0.0%, far below the 0.25 clean-loser ratio). The D1 differentiation (perfect at the D1 pool) now
   propagates through GPi → thalamus → commit → motor. This is the first spiking N=5 clean winner: the wall's method
   (the N-way cross-inhibition at the sub-threshold tonic) is replaced by the faithful high-tonic GPi operating point.

2. **The window is bounded on both sides.** Below gpi=400 the GPi is too silent to differentiate cleanly (gpi=50:
   near-silent, motor flat; gpi=100-200: the motor differentiates but the max-loser is 54-78% of the winner, not clean,
   and at gpi=200 the wrong channel wins). Above gpi=600 the GPi tonic over-inhibits the thalamus (gpi=800-1000: the
   thalamus/commit/motor are all silent — no winner at all). The faithful operating point is a window (gpi≈400-600,
   thal=45), not a single value; the fixed point for the six-seed validation is gpi=500 (the mid-window).

3. **The mechanism is the GPi operating point, confirming the pathway-trace diagnosis.** The D1 pool is unchanged across
   the sweep (the salience bias is applied at the D1, independent of the GPi tonic); only the GPi's firing level changes.
   Raising the GPi tonic gives the D1 inhibition room to work, and the differentiation propagates.

## Seed-7 de-risk at the fixed operating point: the race owns the selection; the commit bar is not met (2026-09-28)

The fine sweep validated the clean window only at v_held=+0.9 (the highest-contrast mood). Before the six-seed
generalization, the full de-risk (`research/runners/_tone_bg_nchannel_selector.py::run_seed`, dev seed 7, numpy/CPU,
raw `research/findings/raw/_gpi_fix_seed7_diag.json`) runs all 3 arms (intact + arousal-lesion + direct-path-lesion)
× all 5 v_held × 20 trials (300 races) at the fixed operating point (gpi=500, thal=45):

| arm | commit rate |
|-----|-------------|
| intact | 0.66 (of the committed trials, 0.8 select the host-argmin channel) |
| no-arousal (salience→D1 drive lesioned) | 0.0 |
| no-direct-path (D1→GPi lesioned) | 0.0 |

The modal (most frequent) winner matches the host argmin at four of the five mood levels (v_held −0.9→cand 4,
−0.45→3, 0.0→0, +0.9→1 — the salience-closest candidate); at v_held=+0.45 the modal is cand 1 while the argmin is
cand 2 (the two closest candidates, salience 0.875 vs 0.925).

Two findings:

1. **The commit is fully attributable to the spiking race, not the host (the 2026-09-25 "who owns the computation"
   answer for this seam).** Lesioning EITHER the arousal drive (salience→D1) OR the direct path (D1→GPi) removes the
   entire commit (0.66 → 0.0). No host computation is doing the selection: the spiking N-way race owns it.

2. **The de-risk `go` is false on the commit-rate bar, not the mechanism.** `race_commits_intact` requires ≥0.9 intact
   commit; the de-risk gets 0.66. The load-bearing checks (`arousal_is_load_bearing`, `direct_path_is_load_bearing`)
   are gated on an intact-commit floor (~0.75) and therefore report false even though the underlying lesion effect is
   maximal (0.66 → 0.0). The residual — why 0.66 rather than ~1.0 — is the low-contrast moods: at v_held=±0.45 the
   expected winner's salience advantage over the runner-up is only 0.05 (at +0.45, cand 2 salience 0.925 vs cand 1
   0.875; at −0.45, cand 3 0.925 vs cand 4 0.875), so the race commits less reliably there. The v_held=0.0 mood
   (contrast 0.15) is the most reliable. The modal winner is the salience-correct channel at four of the five moods (all
   but v_held=+0.45).

So the CAPABILITY — a spiking N-way winner-selection race that commits to the salience-correct channel and is fully
attributable to the neural mechanism — is demonstrated on the dev seed. What remains is a refinement (push the intact
commit rate from 0.66 to ≥0.9 by handling the low-contrast moods) and the six-seed generalization (non-negotiable #6),
which is the authoritative check on whether 0.66 holds across seeds.

## Six-seed generalization at gpi=500/thal=45: the capability is present and 100% neural-attributable, but the commit bar is not met (2026-09-28)

The authoritative check (non-negotiable #6): the full de-risk across the six seeds (42/43/44/100/101/102) at the fixed
operating point (gpi=500, thal=45), via `research/runners/_tone_bg_nchannel_gpi_fix_validation.py` (numpy/CPU, raw
`research/findings/raw/_gpi_fix_6seed.json`):

| seed | intact commit | match vs host argmin | no-arousal | no-direct-path |
|------|---------------|----------------------|------------|----------------|
| 42   | 0.45 | 0.6 | 0.0 | 0.0 |
| 43   | 0.51 | 0.8 | 0.0 | 0.0 |
| 44   | 0.15 | 0.6 | 0.0 | 0.0 |
| 100  | 0.30 | 0.8 | 0.0 | 0.0 |
| 101  | 0.69 | 1.0 | 0.0 | 0.0 |
| 102  | 0.28 | 0.8 | 0.0 | 0.0 |
| **mean** | **0.40** | **0.77** | **0.0** | **0.0** |

`n_go = 0/6` (the de-risk `go` fails on every seed).

Three findings, stated honestly:

1. **The dev seed was optimistic; the commit rate does not reach the bar and is seed-variable.** The mean intact commit
   rate is 0.40 (range 0.15–0.69), far below the 0.9 `race_commits_intact` bar. The dev seed 7 (0.66) sat near the top
   of the range; the six seeds show the operating point (gpi=500) is near the edge of the clean window for several of
   them (seed 44 commits only 0.15). This is NOT a closed wall.

2. **The anti-cheat is the robust, consistent result: 100% of the selection is owned by the spiking race, on every seed.**
   On all six seeds, lesioning EITHER the arousal drive OR the direct path drops the commit rate to exactly 0.0. No host
   computation owns any of the selection — the 2026-09-25 "who owns the computation" question is answered decisively for
   this seam: the neural mechanism owns it, fully.

3. **The winner is affect-dependent and mood-congruent, most reliably at the high-contrast moods.** The modal winner is
   the salience-correct channel at v_held=−0.9 (all six seeds → cand 4), v_held=0.0 (all six → cand 0), and v_held=+0.9
   (five of six → cand 1). The errors concentrate at the mid moods (v_held=±0.45), where adjacent candidates' saliences
   are close — e.g. seed 44 picks cand 4 at v_held=−0.45 (expected cand 3) and cand 2 at v_held=+0.45/+0.9 (expected
   cand 1). `winner_tracks_host_argmin` and `winner_is_affect_dependent` are true on all six seeds.

**Verdict (honesty boundary):** the GPi-tonic method REPLACES the wall's method — a spiking N-way race now commits to a
mood-congruent, affect-dependent winner that is fully attributable to the neural mechanism (0% → 0.40 mean commit, clean
and attributable). But it does NOT close the wall to the de-risk bar (0/6 go, mean commit 0.40 < 0.9, seed-variable).
The residual is two-fold: (a) the mid/low-contrast moods, where the salience gap between adjacent candidates is small,
and (b) seed-to-seed sensitivity of the operating point (gpi=500 is mid-window for seed 7 but near the window edge for
others). No-defer: the capability is kept and demonstrated; the GPi-tonic lever is banked as a real (partial) fix; a
further spiking lever is needed to raise the commit rate to the bar.

## Next step (no-defer)

The six-seed generalization is done: 0/6 go, mean commit 0.40 (the capability is present and 100% neural-attributable,
but the commit bar is not met). The next step is a further spiking lever to raise the intact commit rate to ≥0.9,
targeting the two residuals above. Guided by the wall-reframe question ("what else does the real system run alongside
this, that we replaced with a constant?"): the real basal-ganglia selector runs the **STN→GPi competition loop** and a
**post-commit reset** alongside the GPi tonic — the model already has the regions (`selector_stn`, `selector_reset`), so
the first probe is whether the N-channel bridge engages them or leaves them at a constant. That loop is the faithful
mechanism for stabilizing the N-way race (raising the commit rate) and for sharpening the mid-contrast moods, and it is
seed-robust by construction (it does not depend on a hand-tuned tonic operating point). The host-share probe
(`research/runners/_draft_host_share_tone_probe.py`) stays in view: its substitution arm is recorded but not gated on, so
the moment the N-way race commits cleanly the probe's `n_committed` rises and the seam's host-share verdict can be
re-read against a neural winner.
