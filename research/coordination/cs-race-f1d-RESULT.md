# cs-race-f1d RESULT — steps 2-3, measurement M (PROVISIONAL)

**Status:** agent-filed, PROVISIONAL. Per the f1d step-3 handoff, adoption and
any ruling on these readings are Claude's; the driver stops after this commit.
**Provenance:** branch `research/cs-race` @ `b754567f4` (RESULT lands on top);
runner `research/runners/cs_race_f1.py` (worktree); host `83db10ba1ddb`
(server container). `tools/where_ran.py` on a0_dev/a1_dev/c_a5 → `server=3`.
Every run: `SIM_BACKEND=numpy`, `CUDA_VISIBLE_DEVICES=`, wall 42-60 s each
(compute-placement: all well under the 10-minute local limit).
**Blocks:** DEV seeds 1-8 (80 draws/arm), CONFIRM seeds 25-40 (160 draws/arm),
`--controls-only`, baseline grid-row 3, 10 draws/seed.
**Gating (step-2 condition):** A0 reproduced **17/80 ∈ [13,18]** (posctl 10/10
on all 8 seeds) before any arm ran — step 2 proceeded.
**Artifacts:** 69 arm records + `.prov.json` sidecars (argv carries every arm
and steer flag) under `research/findings/raw/_cs_race/f1d/m/`; aggregation in
`research/findings/raw/_cs_race/f1d/step3_stats.json` (+ sidecar), produced
read-only by `/opt/data/cache/scratch/a_s3agg.py`. 48 dev steer runs
(A0/A1/A4/A5 × frog{1,2,4,8} pA × {engram,gate} + vole{+4} × {engram,gate}),
12 confirm runs (4 arms + 8 frog-p4 steers). Identity 1d#2 re-verified PASS 8/8
at spec cue-noise-sd 15.0 (runner annotation now 15.0, same commit series).

## Metric 1-2 — commits / co-commits / none, frog-vole wins, share CI, posctl

```
DEV arm    pass  com  co  none  fro  vol  frog-share [95% CI]   posctl_min
a0         0/8    17  63     0   10    7  0.588 [0.360, 0.784]  10
a1 (gis30) 7/8    78   1     1   46   32  0.590 [0.479, 0.692]  10
a1gs       7/8    77   0     3   43   34  0.558 [0.447, 0.664]  10
a1s (sgi)  0/8    17  63     0   10    7  0.588 [0.360, 0.784]  10
gis20      2/8    59  21     0   36   23  0.610 [0.483, 0.724]  10
gis25      7/8    77   3     0   45   32  0.584 [0.473, 0.688]  10
gis35      7/8    76   0     4   45   31  0.592 [0.480, 0.696]  10
gis40      4/8    70   1     9   39   31  0.557 [0.441, 0.668]  10
a3         0/8    28  52     0   14   14  0.500 [0.326, 0.674]   9
a3n        0/8    29  51     0   14   15  0.483 [0.314, 0.656]   9
a4         6/8    75   4     1   45   30  0.600 [0.487, 0.703]   9
a5         0/8     3   0    77    2    1  0.667 [0.208, 0.939]   0
a5_wie20   0/8     2   0    78    2    0  1.000 [0.342, 1.000]   0
a5_wie80   0/8     3   0    77    2    1  0.667 [0.208, 0.939]   0
CONFIRM arm  pass   com  co  none  fro  vol  frog-share [95% CI]  posctl_min
c_a0         0/16   26 134     0   15   11  0.577 [0.389, 0.745]   9
c_a1        12/16  153   4     3   76   77  0.497 [0.419, 0.575]   9
c_a4        15/16  154   0     6   68   86  0.442 [0.365, 0.520]   9
c_a5         0/16    6   0   154    3    3  0.500 [0.188, 0.812]   0
```

Per-seed frog/vole/com/co/posctl: `step3_stats.json → stats[*].per_seed`.
Full-bar failures are all bar-logic (runner `pass` = com≥9 and both wins in
[2,8]): dev a1/a1gs/gis25 seed 3 = frog 10/0; gis35 seed 3 = 9/1; a4 seeds 3
(10/0) and 4 (com 8); confirm c_a1 seeds 25 (10/0), 30 (vole 9), 31 (com 7),
37 (com 8); c_a4 seed 33 (com 7). Frog shares sit on chance in every arm
(CIs cover 0.50 except c_a4 at 0.442, CI up to 0.520) — no content bias.

## Metric 3 — ignition gap and cross-draw ignition SD

Median |t_ign(vole) − t_ign(frog)| over draws where both gates ignite, and the
count of seeds whose cross-draw SD of min(t_ign) reaches 10 ms (of 8 DEV):

```
arm     med_gap_ms  gap_n  seeds_SD>=10    arm    med_gap_ms gap_n  SD>=10
a0         4         73       2/8          c_a0      5       149     1/8
a1         4         73       2/8          c_a1      5       149     1/8
a3         8         68       2/8          c_a4      8       142     3/8
a3n        4         67       8/8
a4         8         68       2/8          (gis20-40 all med 4, 2/8)
```

## Metric 4 — cue-lock index minus chance baseline

Cue window = [0, 500) ms cue-onset-relative (hold to 600); index = share of
competitor-slot gate spikes inside it. Chance = mean share over 20 windows of
equal length anchored at seeded starts drawn inside the cue window (procedure
frozen here; the ruling left the offset draw unspecified).

```
arm   a0    a1    a1s   a1gs  gis35 gis40 a3    a3n   a4    a5    c_a1  c_a5
idx  1.000 1.000 0.776 0.991 0.988 0.982 1.000 1.000 1.000 0.866 0.997 0.982
base 0.393 0.404 0.703 0.430 0.435 0.470 0.407 0.385 0.410 0.552 0.415 0.554
```

## Metrics 5-7 — unitary EPSP, cue-current ratio, survivors, latch stability

Unitary EPSP peak: 3.07 mV (cue-n 8 arms), 0.488 mV (a3/a4: cue-n 48,
w-cue 7.5); confirm 2.962 / 0.479 mV. Decisive/posctl ratio of pre-ignition
cue-only current: 1.87-2.03 across all arms. Gate-survivor categories:
a0 {none 63, first 6, single 11} vs a1 {first 58, single 18, last 2, none 2}
vs a4 {first 50, single 20, none 5, last 5} (of 80 decisive). First-30 vs
last-30 latch disagreements: 0/80 in a0/a1/a1s/a3/a3n/a4; gis35 3, gis40 3,
a1gs 2, a5-family 1 (of 80); ≤1/160 in every confirm arm — the latch is stable.

## Metric 8 — steer curves (seeds reaching ≥8/10 frog wins; frog totals/80)

```
arm stage pa1     pa2     pa4     pa8     vole+4 target-follows
a0  eng   0/8(11) 0/8(12) 1/8(19) 1/8(29) vole 9/80
a0  gate  0/8(10) 0/8(10) 0/8(11) 0/8(12) vole 7/80
a1  eng   1/8(48) 3/8(56) 5/8(63) 7/8(71) vole 52/80
a1  gate  1/8(47) 1/8(47) 1/8(48) 2/8(51) vole 33/80
a4  eng   2/8(50) 3/8(55) 4/8(60) 7/8(70) vole 50/80
a4  gate  1/8(45) 2/8(46) 2/8(48) 2/8(50) vole 35/80
a5  eng   0/8(2)  0/8(4)  0/8(8)  0/8(14) vole 4/80
a5  gate  0/8(2)  0/8(2)  0/8(2)  0/8(4)  vole 1/80
```

Confirm +4 frog-pA steers (seeds ≥8/10 of 16; totals/160):
c_a1 eng 6/16 (107) vs gate 4/16 (78); c_a4 eng 4/16 (102) vs gate 1/16
(72); c_a0 eng 0/16 (24) vs gate 0/16 (15); c_a5 0/16 both.
The vole +4 checks move their own target (a1: 52/80 vole wins under the
engram vole steer) — the steer follows its target content in both directions,
so the curve is not a frog-biased artifact.

## Rule readings (frozen text; adoption = Claude)

- **(iii) read-out co-commit — NOT CONFIRMED.** Bar = exclusion arm 8/8 DEV
  AND A1 16/16 CONFIRM, posctl intact. A1: **7/8 dev, 12/16 confirm**
  (failures above); A1gs 7/8; A1s 0/8 (bit-identical counts to A0 — W_SGI
  alone is inert at 30; a1gs ≠ a1, so the pathway is wired). Posctl intact
  everywhere. The W_GIS band reaching 7/8 includes both 25 and 35 (response is
  broad, not knife-edge), but no arm reaches 8/8, so the primary bar fails.
  Descriptive: **co-commits 63→1 dev, 134→4 confirm** — W_GIS collapses the
  co-commit regime; the frozen pass bar is missed only by split-band/com-
  minority seeds.
- **(iv) common-volley — NOT CONFIRMED.** Needs all three on one arm: med gap
  ≥8 ms, SD≥10 on ≥6/8 seeds, co cut ≥50%. A3: gap 8 ✓, SD 2/8 ✗, cut 17% ✗.
  A3n: gap 4 ✗, SD 8/8 ✓, cut 18% ✗. Partial patterns only.
- **(v) hair-trigger — NOT CONFIRMED.** A5: Δ decisive = −14 (3 vs 17; need
  ≥ +30) ✗; posctl min 0 (need ≥9/10 on 8/8) ✗; W_IE sensitivity |2−3| = 1
  (need ≥15) ✗. Caveat: A5 silences posctl too (77/80 none, c_a5 erratic
  0-10) — it sits outside a valid operating point, so its data cannot support
  either direction of hair-trigger; the rule reads NOT CONFIRMED and the
  hypothesis stays open pending a valid-point arm (Claude's call).
- **Ownership — NOT ESTABLISHED at the frozen rule** (engram steer with pa ≤
  4 pA moving the latched winner to ≥8/10 on ≥6/8 seeds, gate moving it less):
  a0 eng 1/8 vs gate 0/8; **a1 eng 5/8 vs gate 1/8** (closest, one seed shy);
  a4 eng 4/8 vs gate 2/8; a5 0/8. Descriptive: engram > gate at every pa and
  for both target contents; a1 eng reaches 7/8 at +8 pA (outside the ruled
  range); confirm shares agree in direction (eng vs gate: a1 0.690 vs 0.506,
  a4 0.662 vs 0.477, a0 0.857 vs 0.600).

All four readings are mechanical applications of the frozen rules in
`cs-race-f1d-RULING.md` §step 3 to `step3_stats.json`; no rule text was
changed.

<!-- REFUTER-LENS-BEGIN -->
REQUIRED LENSES (answer each from recorded data, never from inference):
(a) ENGAGEMENT: did each lesion, permutation or flag actually engage, as recorded in the data (a call count, n_words, a changed field, a trace with route / lesioned / error)? An arm with no record is UNDEFINED, never a PASS or a FAIL.
(b) ANSWER-RATE: for any default-on candidate, does flag-on lower the answer rate or raise abstentions ("I don't know about that.") against flag-off on the packet's own probes, on any seed? Run `tools/answer_rate.py` (prefixed or paired mode) or `iw2_score.py B --off-tag`; a FLIP-HURTS-ANSWERS result means the flip needs a Claude ruling (NEEDS line) before it lands.
(c) WHERE-RAN: which host produced each cited result (run `python tools/where_ran.py <result paths>` and quote its host summary), and does validity depend on it (pool/AWS nodes and the workstation differ in what inputs they hold, for example files under research/findings/raw/ that the provisioner does not ship)?
<!-- REFUTER-LENS-END -->

Answers for this RESULT:
- **(a) ENGAGEMENT — answered from records.** Each file's top-level
  `instrument_flags` shows the f1d levers as run (a0 w_gis 0 / a1 w_gis 30 /
  gis35 w_gis 35 / a1s w_sgi 30, cue_n and cue_noise_* likewise); every steer
  file carries `decisive_control.steer_host_probe` {pa, stage, target} and its
  arm's `instrument_flags`; A5's w_xg/w_gs and all CLI arguments are in each
  sibling `.prov.json` `argv`. Outcome moves track the levers (co 63→1 with
  w_gis; frog totals rise monotonically with steer pa; A1s inert while
  a1gs ≠ a1). No arm is unrecorded.
- **(b) ANSWER-RATE — not applicable.** This is a two-fact race harness with
  no default-on candidate and no abstention surface; no default is flipped
  here (adoption is Claude's).
- **(c) WHERE-RAN — answered from `tools/where_ran.py`.** Sampled
  a0_dev/a1_dev/c_a5 → host summary: `83db10ba1ddb (server) 3`; all 69 runs
  share that host (per-run sidecars), CPU numpy, inputs read from this
  worktree — no pool/AWS/workstation input asymmetry applies.
