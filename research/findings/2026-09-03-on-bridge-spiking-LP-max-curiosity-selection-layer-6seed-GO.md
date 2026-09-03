---
type: finding
status: contributing
date: 2026-09-03
mechanism: curiosity-learning-progress-maximizing-selection-onbridge
runner: research/runners/_laneB_curiosity_lp_max_onbridge_derisk.py
builds_on:
  - research/findings/2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md
reuses:
  - research/runners/_affect_marker_wta_derisk.py
  - research/runners/_curiosity_seek_learn_onbridge_derisk.py
artifacts:
  - research/findings/raw/lanes/curiosity/lp_max_onbridge_smoke.json
  - research/findings/raw/lanes/curiosity/lp_max_onbridge_derisk.json
  - research/findings/raw/lanes/curiosity/lp_max_onbridge_near_tie_characterization.json
---

# on-bridge SPIKING LP-max curiosity SELECTION — the neural max, isolated from the LP estimate, is a clean 6-seed GO

<!--derived-->
**One-line verdict.** Realizing the CPU-proxy `2026-08-07` LP-MAX ask-selection thesis on a real
`SimulationBridge` requires an un-built piece: a genuine NEURAL max-selection over options (every existing
on-bridge curiosity runner still picks the ask with a host `max(want[c] ...)`). Following the
`2026-09-03-on-bridge-spiking-LP-max-curiosity-DESIGN.md` design's own cheapest-first prescription, this
build isolates that one piece — an N-channel FSI lateral-inhibition WTA, reused verbatim from the already-GO
`_affect_marker_wta_derisk.py` affective-marker circuit, driven directly by a per-option LP-slope current
instead of a Gaussian-tuned mood value — from the (separately fragile) LP *estimate*, and drives it with
KNOWN synthetic LP vectors instead. Result: **6/6 seeds GO** on all four of the design's own gates
(g_select, g_noisy, g_loadbearing, g_specificity), with clean margins (100% selection accuracy, zero
noisy-option wins, 100% novelty-tracking under the drive-swap lesion, 0% accuracy under the mis-routing
anti-cheat). A supplementary (non-gated) near-tie stress sweep confirms the design's own predicted boundary
case — the circuit degrades by honest *indecision* (a "no clean winner" read), not by confident wrong
answers, as the top-two LP-slopes converge.

## What this tests, and why it is isolated from the LP estimate (re-anchor, design SS1/SS4)

<!--derived-->
The design doc names two host shortcuts in the current LP-max mechanism: (A) the per-option LP estimate is a
host numpy EMA, and (B) the MAX-selection over options is a host `argmax` — present even in the on-bridge
DR-1 curiosity runner (`_curiosity_seek_learn_onbridge_derisk.py` line ~530: `mx = max(want[c] for c in
cands)`), despite that runner's `want`/reward signal already being read from spikes. Shortcut B — genuine
neural max-selection over an option set — has never been built in this repo; this is the build. Per the
design's SS4, the cheapest and most diagnostic first test drives the selection layer with a SYNTHETIC,
KNOWN LP-slope vector rather than the DR-1 spiking reward read, so a failure of the (known-fragile, 1/6-seed
substrate-memory) LP estimate can never be mistaken for a failure of the (previously untested) neural max.
That isolation is what this de-risk executes; the DR-1-fed FULL build is the named next step (below), not
attempted here.

## Mechanism and controls

<!--derived-->
`research/runners/_laneB_curiosity_lp_max_onbridge_derisk.py` builds one `LPMaxWTA` per seed: 6 excitatory
option assemblies (24 RS neurons each) with their own fast-spiking-interneuron cross-inhibition sub-pools
(12 FS neurons each, mutual/reciprocal lateral inhibition), 216 neurons total — reused unmodified via
`_affect_marker_wta_derisk._build_bridge`/`_pool_rates` (same `N_PER`/`N_PER_FSI`/`TO_FSI_WEIGHT`/
`CROSS_INHIB_WEIGHT`/`DEAD_MARGIN` as that already-6/6-GO 6-channel valence circuit — no weights re-derived).
Each trial synthesizes a KNOWN LP-slope vector for 6 options: one clear max-learnable option (LP in
[0.55, 0.95]), several mid options (LP in [0.10, 0.40]), and one option pinned at LP=0.0 — plus a SEPARATE
"would-be novelty" scalar that peaks (0.85-1.0) on that SAME zero-LP option and is NEVER wired into any
option pool's `cp_external_input_current` (novelty stays structurally absent from the selection drive, per
design mitigation #3). `drive_pa[i] = 150 + 1400 * max(0, LP-slope[i])` pA (matching the affect-marker
organ's baseline/gain regime); the winner is read off `cp_firing_states`, requiring the top assembly to
clear the runner-up by the affect-marker's own dead margin (0.05 rate-units) or the circuit honestly reports
"no clean winner" (`None`), same convention as `_affect_marker_wta_derisk`'s "safe no-marker" behavior.

Three conditions per trial, all on the SAME warm bridge:
- **intact**: driven by the true LP-slope vector — tests g_select and g_noisy.
- **lesion**: the SAME circuit driven by the (unwired) novelty vector instead of LP-slope — tests
  g_loadbearing (design: "replace the LP drive with a novelty drive ... the noisy high-novelty option now
  WINS").
- **permuted**: driven by the LP-slope vector re-indexed through a fixed, seeded, non-identity permutation
  (assembly *i* driven by option `perm(i)`'s LP, winner reported by physical/canonical assembly identity,
  never translated back) — tests g_specificity, reusing `_affect_marker_wta_derisk`'s own shuffle anti-cheat
  logic verbatim.

## Frozen 6-seed result

<!--derived-->
Command (deterministic; CPU-only, no GPU):

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_derisk \
  --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/curiosity/lp_max_onbridge_derisk.json
```

50 trials/seed x 6 seeds = 300 trials x 3 conditions = 900 selection reads. Wall time ~10s; peak child RSS
measured at 320MB (well under the 4GB budget).

| seed | g_select (winner==argmax(LP)) | g_noisy (noisy-option wins, raw count) | g_loadbearing (tracks novelty under lesion) | g_specificity (permuted accuracy) | GO |
|---|---:|---:|---:|---:|---|
| 42  | 48/48 evaluable, 100% | 0/48 | 50/50, 100% (0% still tracks LP) | 0/47, 0% | yes |
| 43  | 48/48 evaluable, 100% | 0/48 | 50/50, 100% (0% still tracks LP) | 0/49, 0% | yes |
| 44  | 47/47 evaluable, 100% | 0/47 | 50/50, 100% (0% still tracks LP) | 0/48, 0% | yes |
| 100 | 46/46 evaluable, 100% | 0/46 | 50/50, 100% (0% still tracks LP) | 0/47, 0% | yes |
| 101 | 47/47 evaluable, 100% | 0/47 | 50/50, 100% (0% still tracks LP) | 0/47, 0% | yes |
| 102 | 50/50 evaluable, 100% | 0/50 | 50/50, 100% (0% still tracks LP) | 0/48, 0% | yes |

**Aggregate verdict: 6/6 seeds GO.** All four gates clear their design-specified bars (g_select/g_loadbearing
>=90%, g_specificity <=50% and <=half the intact accuracy, g_noisy a raw zero count) on EVERY seed, with no
close calls: selection accuracy is 100% wherever evaluable, noisy-option wins are exactly zero across 291
evaluable intact trials, and permuted accuracy is exactly zero across 286 evaluable trials (`attributable_to`
reports 100.0% of the "winner tracks max-LP" effect attributable to the intact LP drive on every seed — none
of it survives the novelty-drive lesion). The ~2-8% of trials per seed where the circuit reports "no clean
winner" (46-50 of 50 evaluable) are excluded from the accuracy denominators via `tools.lab.undefined_if_empty`
rather than counted as wrong — an honest indecision, not a miscommitment (see the near-tie characterization
below for why this rate is low: the synthetic trials used a comfortable margin between the max and the
runner-up, as the design's SS4 explicitly recommends for a first pass).

## Supplementary (non-gated): the design's own predicted failure mode, characterized

<!--derived-->
The design names failure mode #1 explicitly: "WTA gives no clean winner / multiple fire when LP-slopes are
close." The 6-seed sweep above never stress-tests this — its "mid" options sit 0.15-0.85 below the max by
construction. A separate runner, `_laneB_curiosity_lp_max_onbridge_near_tie_stress.py`, forces the top-two
options to a swept margin (all 6 seeds, 20 trials/margin/seed = 120 trials/margin) and reports what happens
as the margin narrows:

| margin (LP-slope units) | evaluable rate | accuracy when evaluable |
|---:|---:|---:|
| 0.20 | 98.3% (118/120) | 100.0% |
| 0.10 | 53.3% (64/120) | 100.0% |
| 0.05 | 18.3% (22/120) | 100.0% |
| 0.02 | 6.7% (8/120) | 87.5% |
| 0.01 | 5.8% (7/120) | 57.1% (chance-level, as expected for a genuine near-tie) |

This is exactly the design's own anticipated shape: the circuit degrades by DECLINING to answer (the
dead-margin check correctly reads a true near-tie as "no clean winner"), not by confidently picking the
wrong option — accuracy stays 100% down to a 0.05 margin, and only drops toward chance (57%) once the
margin (0.01) is smaller than what a 216-neuron rate-code can resolve in the 60-step read window. This
matches the design's own SS3b mitigation-in-order-of-preference: mitigation (i), "keep the proxy's eps-greedy
+ tie-break tolerance ... a 'no clean winner' step just picks among near-winners — behaviorally fine," is
sufficient at the margins this sweep produces; mitigation (ii)/(iii) (recurrent self-excitation,
accumulate-to-threshold) are not needed by this evidence.

## Interpretation and honest scope

<!--derived-->
This is a GO on the design's own SS4 gates for the NEURAL MAX in isolation — the genuinely un-built piece
the design identified (Shortcut B). It is deliberately NOT a claim that the full LP-max curiosity loop works
on spikes end-to-end. Four explicit scope limits:

1. **The LP-slope input is synthetic, by design.** This de-risk proves the WTA correctly finds the max of
   WHATEVER per-option scalar drives it; it says nothing about whether the DR-1 spiking reward read
   (`deliver_reward`'s `reward_read`) produces a LP-slope vector with the same well-separated structure the
   synthetic trials used. The lane-B record already carries an open question here (1/6-seed substrate-memory
   promotion of the LP-slope estimate, `2026-08-02` finding) — this de-risk does not resolve it and was not
   designed to.
2. **`--selection bg` (Primitive 2, BG selection-by-disinhibition) is accepted by the CLI but not built.**
   The design itself gates it on Primitive 1 (this WTA) passing first; running `--selection bg` prints that
   deferral and writes a `status: deferred` JSON rather than a fabricated result.
3. **`--fast-tonic` (design SS3a's two-pool phasic/tonic split) is accepted but a documented no-op here.**
   It only matters once the input is a stream of raw progress reads rather than an already-differenced slope
   scalar (this de-risk's input), and the design itself says to add it only if the mastered-vs-noisy
   distinction leaks — untested by this synthetic sweep.
4. **The one-bridge composition (design SS3c) is not built.** The FULL realization — this WTA driven by
   `build_curiosity_bridge` + `deliver_reward`'s spiking `reward_read` on ONE bridge, replacing the synthetic
   vector — is the named next lever (below), not attempted here, precisely because attempting it before this
   layer's own correctness was established would have confounded two untested mechanisms in one result (the
   exact trap the design's SS4 opening paragraph warns against).

## Next step (named, not built)

<!--derived-->
Wire the FULL composition (design SS3c): build a `SimulationBridge` extending
`_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge` with this module's option/FSI regions added to
the SAME `cfg.brain_regions`/`cfg.region_pathways` lists, drive each option pool from `deliver_reward(c,
LP)`'s `reward_read` (instead of the synthetic vector), and re-run this de-risk's four gates on THAT signal.
If the LP-slope estimate's known seed-fragility (1/6 substrate-memory promotion) leaks into the selection
layer's evaluable/accuracy numbers, that failure belongs to the LP-ESTIMATE sub-problem (already flagged,
`2026-08-02` finding), not to the max-selection mechanism this de-risk GO'd — the two must stay
distinguishable in whatever result comes out of that build, which is exactly why this de-risk kept them
apart. If margins in the real (not synthetic) LP-slope signal turn out tighter than 0.05 (this sweep's
100%-accuracy floor), the design's mitigation (ii) — recurrent self-excitation on each option assembly
(Wang 2002) — is the next lever, not a redesign of the WTA topology itself.

## Reproduce

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_derisk \
  --smoke --out research/findings/raw/lanes/curiosity/lp_max_onbridge_smoke.json

env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_derisk \
  --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/curiosity/lp_max_onbridge_derisk.json

env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneB_curiosity_lp_max_onbridge_near_tie_stress \
  --out research/findings/raw/lanes/curiosity/lp_max_onbridge_near_tie_characterization.json
```
