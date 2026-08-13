---
type: finding
status: go
date: 2026-08-13
mechanism: w4-detector-operating-point-homeostat
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_w4_detector_operating_point_homeostat_derisk.py
artifacts:
  - research/findings/raw/_w4_op_homeostat/w4_6seed.json
  - research/findings/raw/_w4_op_homeostat/smoke.json
builds_on:
  - research/findings/2026-08-13-w4-belief-magnitude-calibration-BOUNDARY.md
  - research/findings/2026-08-13-w4-informativeness-objective-BOUNDARY.md
  - research/findings/2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md
  - research/findings/2026-08-01-W4-recursive-theory-of-mind-2nd-order-false-belief-plus-depth2-scalar-implicature-6seed-GO.md
---

# The W4 seed-44 implicature collapse was a PER-DETECTOR OPERATING-POINT artifact, not a belief/objective/read-topography residual -- a detector operating-point HOMEOSTAT (calibrate the shared graded-plateau operating point to a content-free per-detector set-point on the WORST detector, not column 0) RESCUES seed 44 and makes the graded belief BEAT one-hot on the informativeness objective 6/6 (up from [B]'s 5/6). The W4 / Task-#12 pragmatic arc CLOSES.

<!--derived-->
The 6-seed OBJ_inf numbers, the per-detector operating-point diagnosis, the seed-44 rescue and every anti-cheat
below are read from the cited runner + w4_6seed.json artifact.

<!--derived-->

**One-line verdict:** [B] `2026-08-13-w4-belief-magnitude-calibration-BOUNDARY` removed the off-target belief-only
AND-gate leak (read the TRUE-INTENT detector, not the whole population), flipping the informativeness objective
OBJ_inf from [W]'s -0.011 / 3-of-6 to +0.090 / 5-of-6, and named the residual: ONE heterogeneity-adverse seed (44)
whose per-seed plateau calibration picked center=96, collapsing the fractional implicature response (implicature
cell 0.017, OBJ_inf move -0.041). This finding LOCATES that residual precisely -- it is a PER-DETECTOR
OPERATING-POINT artifact -- and surpasses it with a detector operating-point homeostat: the graded belief now beats
one-hot on OBJ_inf **6/6 (mean move +0.150)**, with seed 44 RESCUED (-0.041 -> +0.198; implicature cell 0.017 ->
0.202), the col-0 arm reproducing [B]'s 5/6 boundary byte-identically, and the whole-population leak read still
failing. Detector -> read-out -> objective -> OPERATING POINT: all four surpassed.

## Where the seed-44 collapse ACTUALLY comes from -- measured PER DETECTOR COLUMN, not assumed

<!--derived-->

The per-seed graded-plateau calibration (`calibrate_graded_seed`, the [M]/[B] instrument) picks ONE (center, slope)
by minimizing the GLOBAL proportionality error on the ignition curve of ONE detector column (t=0). But under
parameter heterogeneity each success[t] detector has its OWN operating point, and the IMPLICATURE lives on a
DIFFERENT column (t="all", index 2). Reading seed 44's three detectors at the col-0-chosen 96/0.08 (content-free
per-detector ignition drives: solo intent, and the fractional coincidence r(0.27)):

| seed-44 detector @ 96/0.08 | solo_intent | r(0.27) coincidence | state |
|---|---|---|---|
| col 0 (the calibration checks THIS) | 0.000 | 0.079 | clean |
| col 1 | 0.000 | 0.064 | clean |
| **col 2 ("all", the implicature)** | **0.084 (LEAK)** | **0.022 < solo** | **COLLAPSE + INVERSION** |

The implicature detector's solo-intent AND-gate LEAK (0.084) is LARGER than its fractional coincidence (0.022), so
the row-normalized implicature cell inverts to 0.017. At 88/0.08 the SAME seed-44 detectors are ALL clean +
resolved (col 2: solo 0.000, r(0.27)=0.102) -> implicature cell 0.202 (analytic 0.20). So the collapse is a
PER-DETECTOR OPERATING-POINT artifact: the single-column proxy calibration is BLIND to the leak on the detector the
implicature actually uses. This is exactly the "missing-companion-process" pattern (CLAUDE.md): the real detector
population runs per-unit intrinsic-excitability homeostasis so EVERY detector resolves its inputs; we substituted
ONE detector's operating point for the whole pool.

## The mechanism -- a detector OPERATING-POINT HOMEOSTAT (label-free; content-free; NO belief change; NO sim/ edit)

<!--derived-->

Two content-free, label-free components (both host-side READ-OUT corrections, the same category as the
row-normalization the pipeline already applies):

1. **Per-detector operating-point homeostasis (the load-bearing gain).** Re-select the shared graded-plateau
   operating point (center, slope) so the homeostatic set-point -- AND-gate SILENCE (solo drives sub-floor) +
   PROPORTIONAL FRACTIONAL-COINCIDENCE RESOLUTION (r_t(0.27) resolved ABOVE the solo leak) -- holds on the WORST
   detector across ALL K columns, not column 0 alone; among qualifying candidates, minimize the worst-detector
   proportionality error. This is intrinsic-excitability homeostasis (Turrigiano 2008, a FIXED set-point NOT fit to
   the answer): each detector's excitability is set to resolve its own content-free drive; the population
   homeostasis picks the shared point that satisfies the worst detector. It REJECTS seed-44's leaky 96/0.08 and
   selects 88/0.08 where the implicature detector resolves.
2. **Per-detector divisive-normalization read (Carandini & Heeger 2012, "a canonical neural computation").** The
   matched read is divisively normalized per detector by its OWN content-free statistics: Sdn[t,u] = max(0,
   S[t,u]-b_t)/(sigma+g_t), b_t = the detector's solo-drive AND-gate leak floor, g_t = its dynamic range.
   REPORTED (the canonical read-out form + robustness); the operating-point homeostasis is what rescues the
   collapsed detector.

Everything the homeostat uses is measured on CONTENT-FREE controlled drives (solo intent, solo belief, fractional
coincidence) per detector column -- NEVER the RSA answer, the belief content, or which intent wins. Applied
UNIFORMLY to onehot / graded / scramble.

## The clean attribution -- the ONLY change from [B] is the SELECTION CRITERION (proven by a same-grid control)

<!--derived-->

The homeostat searches the IDENTICAL grid as the col-0 / [M] calibration ([80,88,96] x [0.08,0.11,0.14]), so the
ONLY difference from the col-0 arm is the selection criterion (per-detector worst-detector set-point vs col-0
single-column global proportionality). A same-grid control confirmed the CRITERION -- not a finer grid -- is what
rescues seed 44: 88/0.08 (a resolving point) already existed IN the original grid, but col-0's single-column
criterion picked the collapsing 96/0.08 over it. The runner ships the col-0 arm as the in-run LESION (the unfixed
single-column calibration); it reproduces [B]'s boundary byte-identically, isolating the operating-point change.

## Result -- 6 seeds {42,43,44,100,101,102}, CPU numpy, OBJ_inf (informativeness objective, primary)

<!--derived-->

| read (OBJ_inf) | onehot | graded | move | graded>onehot | scramble | scramble loses? |
|---|---|---|---|---|---|---|
| col0 matched (reproduces [B]) | 0.817 | 0.907 | **+0.0905** | **5/6** | 0.19 | yes |
| col0 WHOLE-POP (reproduces [W]) | 0.846 | 0.835 | **-0.0108** | 3/6 | -- | yes |
| **HOMEOSTAT matched (PRIMARY)** | 0.803 | **0.953** | **+0.1499** | **6/6** | 0.137 | yes |
| divnorm (matched + divisive norm) | 0.803 | 0.961 | +0.1705 | 6/6 | -- | yes |
| whole-pop @ homeostat pt (leak control) | 0.807 | 0.827 | +0.0194 | 4/6 | -- | yes |

The col-0 matched arm reproduces [B]'s boundary EXACTLY (onehot 0.81668, graded 0.90718, move +0.0905, 5/6) and the
col-0 whole-population arm reproduces [W]'s (onehot 0.84566, graded 0.83485, move -0.0108, 3/6) -- the instrument is
byte-identical to both prior findings. The homeostat matched read flips ALL SIX seeds. Per-seed homeostat OBJ_inf
move: 42 +0.148, 43 +0.169, **44 +0.198** (the rescued seed, now the STRONGEST mover), 100 +0.079, 101 +0.173, 102
+0.133. The other three objectives ALSO move 6/6 under the homeostat (M1 +0.050 6/6; OBJ_cell +0.050 6/6; OBJ_surp
+0.020 6/6, up from col-0's 3/6).

## Why seed 44 is rescued, and why it is not a blanket threshold lowering

<!--derived-->

Per-seed implicature cell (graded) col0 -> homeostat: 42 0.202->0.253, 43 0.275->0.231, **44 0.017->0.202**, 100
0.287->0.287, 101 0.119->0.227, 102 0.133->0.133. Per-seed operating point col0 -> homeostat: 42 88->88
(slope 0.11->0.08), 43 88->80, **44 96->88**, 100 88->88 (unchanged), 101 88->88 (slope change), 102 96->96
(unchanged). The homeostat SELECTIVELY re-points the leaky detectors: seeds 100 and 102 (already balanced) keep
their operating point; some move up in slope, seed 43 down in center -- it is a per-detector balance, NOT a uniform
threshold lowering. Seed 44's collapsed implicature detector is rescued (96->88, r(0.27) 0.022 -> 0.102 above its
now-silent solo leak).

## Anti-cheats (each a gate that behaved) -- verified with verify-go

<!--derived-->

- **PRIMARY 6/6, all seeds > MOVE_EPS (0.03):** min per-seed move +0.079 (seed 100); the formerly-failing seed 44
  is the strongest (+0.198). Not one lucky seed.
- **LABEL-FREE:** the homeostat reads only content-free per-detector drives (solo intent, solo belief, fractional
  coincidence), never the RSA answer / belief / which intent wins; applied uniformly to onehot/graded/scramble.
- **NOT "just lower all thresholds":** the col-0 LESION arm reproduces [B]'s 5/6 boundary byte-identically (seed 44
  collapses at the col-0 point), and the operating points move selectively (2 of 6 unchanged) -- the per-detector
  criterion is the ONE change (same-grid control).
- **VALID:** SCRAMBLE (graded mass on WRONG intents) LOSES to one-hot under the homeostat matched read (0.137 <
  0.803) AND under the divisive-normalization read -- reading the true-intent detector at any operating point
  cannot rescue wrong-intent mass.
- **LEAK control still fails:** the whole-population (leak) read at the homeostat operating point does NOT reach the
  bar (move +0.0194 <= MOVE_EPS), and the col-0 whole-population arm reproduces [W]'s -0.011 -- the win is REMOVING
  the leak (the matched read, +0.150), not operating-point inflation. HONEST CAVEAT: at the homeostat point the
  leak read gives graded a small non-winning edge (+0.019, vs [W]'s -0.011 at the col-0 point) -- attribution: the
  decisive signal is the matched (leak-removed) read; the whole-pop move (+0.019) is <13% of the matched move.
- **PRINCIPLED / non-circular:** the matched read faithfully encodes the RSA posterior (mean |T^{-1}(matched) -
  belief_mass| = 0.020 < 0.08, content-free transfer); weights' analytic L1 == ANALYTIC_L1. The analytic 0.20 is an
  OUTPUT (belief content ~0.27 + row-norm), never an input.
- **BELIEF unchanged (moat):** beliefs byte-identical between the col-0 and homeostat arms (only the READ operating
  point changed); graded implicature margin 0.506 (> 0.05); argmax/recall preserved.
- **DIAGNOSIS built WITH its lesion:** the fix (per-detector homeostat) and the lesion (col-0 single-column
  calibration) run in the SAME run; a direct per-column measurement (solo_i 0.084 > r(0.27) 0.022 at 96/0.08)
  separates the causes, not a prescribed mechanism.

## What CLOSES, and the honest scope

<!--derived-->

The full W4 / Task-#12 pragmatic residual is SURPASSED end-to-end, 6-seed: the neural coincidence AND (Leg-1 GO)
does the belief x intent multiply; the magnitude-preserving graded plateau ([M]) reads the fraction; the
informativeness objective ([W]) weights the implicature intent; the true-intent detector read ([B]) removes the
off-target leak; and this detector OPERATING-POINT homeostat resolves the last per-detector residual so the graded
implicature belief beats one-hot on the informativeness objective 6/6. There is no remaining W4 residual to hand to
a next mechanism.

A FUNCTIONAL pragmatics correlate. This is a detector-side READ-OUT operating-point correction (re-point the shared
graded-plateau operating point to a content-free per-detector homeostatic set-point + a per-detector divisive-
normalization read), the same category the pipeline already applies to the neural landscape. It re-points the READ
operating point; it does NOT change the belief (byte-identical to the W4 A/B) or which intent wins (recall intact);
SCRAMBLE still loses; the col-0 arm reproduces [B]'s boundary and the col-0 whole-population arm reproduces [W]'s.
numpy-CPU real spiking Izhikevich; NO sim/ edit; additive NEW runner (reuse-by-import of [B]'s matched read + the
W4 A/B + the informativeness objective). NOT a claim of phenomenal access to another mind; a self-report would be a
functional read-out.

## Sources

- **Carandini & Heeger (2012), Nat Rev Neurosci 13:51** -- "Normalization as a canonical neural computation"
  (divisive normalization / gain control -- the per-detector read).
- **Turrigiano (2008), Cell 135:422** -- homeostatic plasticity with a FIXED set-point (intrinsic-excitability
  homeostasis; the operating-point re-selection is content-free, not fit to the answer).
- **Frank & Goodman (2012), Science 336(6084):998** -- the RSA posterior scale the read is calibrated to.
- **Larkum (2013), TiNS 36(3):141**; **Mikulasch & Priesemann** -- the graded plateau's tunable operating point and
  the dendritic analog read whose per-detector operating point is homeostatically balanced.
- Builds on the 2026-08-13 belief-magnitude-calibration BOUNDARY (the matched true-intent read; 5/6; the seed-44
  residual this closes), the informativeness-objective BOUNDARY (the leak control), and the
  magnitude-preserving-plateau BOUNDARY (the graded plateau read + col-0 calibration).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._w4_detector_operating_point_homeostat_derisk \
    --seeds 42 43 44 100 101 102 --json research/findings/raw/_w4_op_homeostat/w4_6seed.json
```
