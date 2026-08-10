---
type: finding
status: contributing
date: 2026-08-10
mechanism: cortical-afferent-winner-selection
lane: EPISODIC / WTA-readout
---

# The neural WTA readout is weight-controllable at learnable magnitudes — the "latch negative" is a strong-inhibition + high-tonic operating-point artifact, NOT an intrinsic-strength common-mode; the homeostat surpass is aimed at a non-problem for SEPARABLE assemblies

<!--derived-->

**Verdict: the premise of the same-day homeostat research-gate (`d42fd05d`) is REFRAMED by the cheapest decisive
test.** The DR round's literature analysis was correct (pooled divisive-norm + recall-time subtractive-FFI ARE
refuted; the co-resident case IS a dendritic boundary), but its PREMISE — that a SEPARABLE-assembly WTA carries an
intrinsic-strength common-mode a homeostat must strip — is FALSE. Caught by building the pragmatics afferent-swap
probe, sweeping the operating point, and independently verifying the committed baseline.

## What the build+smoke+verification showed

<!--derived-->

New probe `research/runners/_wta_afferent_winner_homeostat_derisk.py` (fixed intrinsic strength via `cfg.seed`
heterogeneity, swappable afferent advantage `intent[t]->utter[t]=W`, NEURAL winner read = which utterance pool
fires most in `cp_firing_states`, no host argmax). Three results, each with teeth:

1. **The homeostat is NOT the remover.** Region-scoped `enable_homeostasis` adapt-rate has ZERO gradient across
   three orders of magnitude (afferent-follow = 1.00 at the default 0.0005 through 0.5) — the design predicted the
   default would show NO effect, so a 1.00 there means the win is NOT the rate-equalization. It is a threshold-
   regime switch (vpeak→the seeded adaptive range) + the E%-max basket fixing a W=30 over-drive pathology.
2. **The plain latched WTA already FOLLOWS the afferent at learnable magnitudes.** Afferent-magnitude crossover
   (committed v2 build `build_speaker_bridge(oracle=True)`, seeds 42/43/44): W=30 → 0.222 (the documented negative);
   **W=10 / 3 / 1.5 → 1.000.** By the DR gate's OWN criterion ("small crossover = real common-mode; needs-huge-
   afferent = cosmetic"), the crossover is ~1.5× — i.e. essentially NO intrinsic common-mode to remove.
3. **INDEPENDENT VERIFICATION (this session):** the committed v2 oracle-weight acceptance probe at the calibrated
   `W_ORACLE=8` (UTT_FS_W=4, FS_UTT_W=4, tonic=0) scores **mean oracle_weight_acc = 1.0000, 6/6 seeds** (gate 0.85,
   passes; every intent's neural winner is the afferent-advantaged utterance). The WTA readout IS weight-
   controllable — a credit rule CAN steer it.

## Why the latch negative was an operating-point artifact, not a common-mode

<!--derived-->

The v1 negative regime was UTT_FS_W=6, **FS_UTT_W=16 (strong feedback inhibition)**, **tonic=900 (high over-drive)**
→ under strong FS + high tonic the first-igniter latch dominates (intrinsic strength wins). The v2 recalibration
that fixed it (`2026-08-08-pragmatics-readback-leg2-...`, comment at runner L111-118) simply WEAKENED the inhibition
(FS_UTT_W 16→4) and REMOVED the tonic (900→0). That is a latch-strength / drive OPERATING POINT, not a common-mode
removal. Two further tells: the probe uses **DISJOINT** assemblies (no cells shared across utterances), so there is
no `D_i = core_i + unique_i` per-assembly common-mode by construction; and the no-afferent control (W=1.0) → chance
for every arm (the neural read genuinely tracks the afferent, no leak).

## Consequence — for the WTA gate and for episodic task #7

<!--derived-->

- **The homeostat + E%-max surpass (`d42fd05d`) is NOT the build for the SEPARABLE-assembly WTA** (pragmatics,
  episodic-CA3, cortex-wta): those assemblies are disjoint → no common-mode; the readout is already weight-
  controllable once the latch is not over-strong and the afferent is at a learnable magnitude.
- **The genuine WTA common-mode is ONLY the CO-RESIDENT case** (source-monitor: the unique signal summed into ONE
  soma rate on the SAME cells as the core) — already mapped to the DENDRITIC substrate
  (`enable_dendritic_divisive_gain`), out of the homeostat's somatic scope. Do NOT build the homeostat against the
  separable case.
- **Episodic task #7 implication:** the episodic neural-WTA sub-wall (`wta_off == full`) is most likely the SAME
  operating-point issue (over-strong lateral inhibition and/or an afferent ca3→cortex weight below the learnable
  crossover), NOT a common-mode. The cheap next lever is the v2 recipe — weaken the readout lateral inhibition +
  set the ca3→cortex heteroassociative weight into the learnable-crossover band — before any homeostat/dendritic
  machinery.

## What survives from the DR round (not wasted)

<!--derived-->

The DR round correctly REFUTED pooled divisive-norm (rank-preserving) and recall-time subtractive-FFI (rank-1
scalar) so they are not re-proposed, and correctly SCOPED the co-resident case to dendrites. The premise error
(separable assemblies carry a common-mode) was the part the cheapest decisive test was built to check — and it
caught it. The homeostat research-gate is annotated with this reframe (still valid ONLY as the far-future co-
resident/dendritic direction, which is a different substrate).

Artifacts: `research/findings/raw/_wta_afferent/homeostat_smoke_s42.json` (the operating-point sweep + crossover),
`research/findings/raw/_pragmatic_success/_verify_oracle_probe_W8.json` (the 1.0/6 verification). No `sim/` edit.
SIM_BACKEND=numpy.
