---
type: finding
status: contributing
date: 2026-09-03
mechanism: vision-lindiscrim-readout / satdiv-divisive-normalization sigma-scale sweep
lane: perception (board #135 / #75)
seeds: [42, 43, 44, 100, 101, 102]   # full decisive 6-seed, every cell
artifacts:
  - research/findings/raw/lanes/perception/satdiv_sig8_sc500_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc771_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig8_sc1200_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig16_sc500_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig16_sc771_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig16_sc1200_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig32_sc500_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig32_sc771_6seed.json
  - research/findings/raw/lanes/perception/satdiv_sig32_sc1200_6seed.json
  - research/findings/raw/lanes/perception/vlin_alpha_readout_6seed.json
  - research/findings/raw/lanes/perception/vision_lindiscrim_opsweep_6seed.json
runner: research/runners/_vision_lindiscrim_readout_derisk.py
builds_on:
  - research/findings/2026-09-01-vision-lindiscrim-opsweep-board135-znorm-dominates-small-nonoverfit-lift-not-capability-crossing.md
  - research/findings/2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md
  - research/findings/2026-09-01-vision-s2-bcm-template-learning-NOGO-collapses-without-competition-underperforms-baseline-with-it.md
  - commit ad0648672 (satdiv scoping + 2-seed smoke)
---

# Board #135's satdiv (sigma, scale) operating-point sweep finds a genuine new best-in-arc cell (sigma=8, scale=771: LEARNED_spkwta_held 0.4722, beats the NO-GO floor 5/6, load-bearing 6/6 — the strongest readout found in the whole arc) that still misses the strict `capability_go` bar on every one of 9 x 6 = 54 seeds by a NARROW, STRUCTURED margin — BORDERLINE, not exhausted: the exact scale value matters far more than the coarse grid tested here, so a finer local sweep is the next well-motivated step, not a fourth confirmation of "readout-exhausted"

## One-line verdict

A full decisive 6-seed sweep of satdiv's (sigma, scale) operating point — 3x3 = 9 cells, sigma in {8, 16, 32} x
scale in {500, 771, 1200}, n=2.0 fixed, the runner's default ridge=0.5 — was pulled from the pool and scored
against the same `capability_go` bar the affine family (z/alpha) already collapsed on. **`capability_go` is 0/6
on every single one of the 9 cells** (54/54 seeds fail it) — the strict multi-criterion capability bar this
lane has used throughout (see `docs/TERMS.md`'s GO condition: "the gate's own verdict is positive", and this
runner's own per-seed `capability_go`, not the looser `overall_verdict`/`task_go_5of6_beat_and_lb` string) is
**not crossed**. But the best cell, (sigma=8, scale=771), is not a small nudge over the affine family's
already-characterized plateau — it is the single strongest cell found anywhere in this arc (see table below),
and its `capability_go` misses are NARROW and STRUCTURED (two specific sub-criteria, close to their thresholds
on several seeds), not a broad collapse like most of the other 8 cells. Combined with a newly-discovered
extreme sensitivity of this operating point to the EXACT scale value (a 0.1-unit change, 771.0 vs 771.1, flips
the beats-NOGO-floor verdict from 5/6 to 1/3 on the seeds both runs share), the honest read is **BORDERLINE**:
the (sigma, scale, ridge) axis is not exhausted the way the earlier ridge-only sweep (fixed at scale=771.1)
concluded — it was under-explored in exactly the dimension (scale precision) that turns out to matter most.

## Why this sweep (context: what was already known)

Board #135's affine s2-norm family (`none`/`submean`/`z`/`alpha`) was already characterized exhausted: the
`vision_lindiscrim_opsweep_6seed.json` 3x4x3 grid found no cell crossing `capability_go` (best cell, z-norm
gain=1.5 ridge=0.5, decisive LEARNED_spkwta_held=0.4531, learned-minus-V1=+0.0347, still under the +0.10
margin), and the follow-on `alpha` graded interpolation's own decisive 6-seed run collapsed to LEARNED_spkwta_held
0.2569 (at/below chance, `learning_load_bearing` 0/6) — the SAME saturation failure signature as the `none`
arm despite a raised RATE ceiling (0.5556). Commit `ad0648672` scoped `satdiv` — Carandini & Heeger's (2012)
actual semi-saturating ratio `R_i = drive_i^n / (sigma^n + sum_j drive_j^n)`, bounded by construction, a
DIFFERENT functional form from the affine `z`/`alpha` family — and its own 2-seed smoke found RATE ceilings
0.55-0.74 but was not itself a GO. Its named next step was a `(sigma, scale, ridge)` exploration.

Two follow-on sessions partially covered that space: a `ridge` sweep {0.05, 0.1, 0.25, 0.5, 1.0} at a FIXED
(sigma=8, scale=771.1), 3 explore seeds only, concluded "readout-exhausted" (best cell: beat 2/3) and handed
off to S2-template learning (BCM); BCM was then built and NO-GO'd at explore (0/3, underperforms the frozen
random baseline). **What had NOT been run: the (sigma, scale) axis itself, at the full 6-seed decisive set.**
That is this sweep — 9 cells, 6 seeds each, 54 runs total, all CPU, dispatched to the mini-PC pool
(`pool41:~/derisk-pool/sim/`) and pulled back via `scp`.

## Result — full 3x3 grid, decisive 6-seed (42/43/44/100/101/102), ridge=0.5, n=2.0

Chance = 0.25; V1-direct floor (mean across cells) approximately 0.42 (varies slightly by seed draw — see per-cell
column); NO-GO floor (config-C fully-spiking) = 0.34; z-normalized affine baseline (published #75) LEARNED=0.4375,
RATE ceiling=0.4653; z op-tuned (opsweep #75a) LEARNED=0.4531; alpha=0.5 (exhausted-affine collapse) LEARNED=0.2569.

| sigma | scale | capability_go | beats NOGO floor | load-bearing | LEARNED_spkwta_held | RATE ceiling | learned-V1 margin |
|---:|---:|:---:|:---:|:---:|---:|---:|---:|
| 8 | 500 | 0/6 | 0/6 | 3/6 | 0.3611 | 0.6215 | -0.0573 |
| **8** | **771** | **0/6** | **5/6** | **6/6** | **0.4722** | **0.6215** | **+0.0538** |
| 8 | 1200 | 0/6 | 0/6 | 0/6 | 0.2569 | 0.6215 | -0.1615 |
| 16 | 500 | 0/6 | 0/6 | 2/6 | 0.3385 | 0.6181 | -0.0799 |
| 16 | 771 | 0/6 | 1/6 | 3/6 | 0.3854 | 0.6181 | -0.0330 |
| 16 | 1200 | 0/6 | 1/6 | 4/6 | 0.3941 | 0.6181 | -0.0243 |
| 32 | 500 | 0/6 | 0/6 | 1/6 | 0.2552 | 0.6129 | -0.1632 |
| 32 | 771 | 0/6 | 0/6 | 1/6 | 0.2882 | 0.6129 | -0.1302 |
| 32 | 1200 | 0/6 | 0/6 | 2/6 | 0.3073 | 0.6129 | -0.1111 |

**All 9/9 cells completed** (no missing/incomplete pool jobs — confirmed by file count and per-seed array length
== 6 in every artifact). **Sigma dominates monotonically**: sigma=8 beats sigma=16 beats sigma=32 on LEARNED_spkwta_held
at every matched scale, and the RATE ceiling also falls slightly as sigma grows (0.6215 -> 0.6181 -> 0.6129) —
larger sigma over-suppresses the semi-saturating ratio, consistent with sigma acting as the ratio's
"how-much-pool-drive-before-suppression-kicks-in" knob (Heeger 1992): more suppression, less signal. Scale is
NON-monotonic within each sigma row (771 beats both 500 and 1200 at sigma=8), consistent with scale setting where
the ratio's dynamic range lands relative to the LIF's non-saturating drive window — too low wastes headroom
(500), too high re-saturates the LIF the same way the affine family did (1200, which reproduces the `alpha=0.5`
collapse almost exactly: LEARNED 0.2569 vs 0.2569, identical to 4 decimals).

## The best cell, seed-by-seed: capability_go misses TWO specific criteria, both marginally

(sigma=8, scale=771) clears `beats_config_c_nogo` on 5/6 seeds and `learning_load_bearing` on 6/6 — both far
better than anything else tried in this arc (the prior best full-6-seed number, k-WTA frac=0.25, was beat=3/6
lb=6/6). Its `capability_go` per-seed breakdown shows exactly which of the 8 AND-ed sub-criteria fail. The
"margin" column below is COMPUTED (per-seed LEARNED minus per-seed A_v1_direct_held, both read from
`research/findings/raw/lanes/perception/satdiv_sig8_sc771_6seed.json`'s per-seed `decode` block) — the artifact
stores each addend, not the pre-subtracted margin:

<!--derived-->

| seed | LEARNED | learned-V1 margin (need >=0.10) | position_pooled_out | capability_go |
|---:|---:|---:|:---:|:---:|
| 42 | 0.4479 | +0.0312 | True | False |
| 43 | 0.4688 | -0.0104 | True | False |
| 44 | 0.4479 | -0.0104 | False (pos leaks: pos_split 0.4167>0.40) | False |
| 100 | 0.4375 | +0.0625 | False (obj_split 0.3958<0.40) | False |
| 101 | 0.5625 | +0.1042 | False (obj_split 0.3958<0.40) | False |
| 102 | 0.4688 | +0.1459 | False (obj_split 0.3750<0.40) | False |

Two of the strict bar's eight sub-criteria are the actual blockers here (scramble-null and label-shuffle-null
pass cleanly on every seed, <=0.34 vs the <=0.40 requirement): the **+0.10-over-V1-direct margin** (met on only
2/6 seeds; the other 4 sit at -0.01 to +0.06, i.e. close but under) and **position-pooled-out** (fails on 4/6
seeds, split between object-decode falling just short of its own 0.40 floor on 3 seeds, and position leaking
just over on 1). Neither is a collapse (compare to sigma=32/scale=500, where LEARNED sits at 0.2552, essentially
random) — both are near-miss margins on a genuinely elevated readout, which is the structural signature of
BORDERLINE rather than exhausted.

## A new instrument finding: this operating point is EXTREMELY sensitive to the exact scale value

A direct comparison against the earlier local ridge-sweep exploration
(`research/findings/raw/lanes/perception/vlin_satdiv_ridge0.5_explore.json`, same sigma=8, ridge=0.5, but
scale=**771.1** not 771.0, seeds {42,43,100} only) on the 3 overlapping seeds:

| seed | local (scale=771.1) LEARNED | this sweep (scale=771.0) LEARNED |
|---:|---:|---:|
| 42 | 0.4479 | 0.4479 (exact match) |
| 43 | 0.4271 | 0.4688 |
| 100 | 0.4167 | 0.4375 |

A **0.1-unit (0.013% <!--derived--> relative) change in scale** flips 2 of 3 overlapping seeds by 0.02-0.04 held accuracy, which is
enough to change the `beats_config_c_nogo` verdict from 1/3 (the local ridge-sweep's own reported number for
ridge=0.5) to what becomes 5/6 at scale=771.0 exactly. This is consistent with — not contradicting — the
arc's own repeatedly-documented spike-quantization sensitivity (the `spkport_cost`/`quantization_gap_rate_minus_spk`
metrics tracked in every prior finding in this lane): a hard-threshold LIF population reading a small,
6-example-per-class training set is inherently sensitive to small shifts in the pre-threshold drive
distribution. **The practical consequence: the earlier ridge sweep's "readout-exhausted" conclusion was drawn
from a scale value that this sweep now shows sits in a locally worse micro-region than a value 0.1 away** — the
readout axis was not fully searched, it was searched at unlucky precision.

## Cross-lane transfer: the SAME mechanism is independently named in the language/mouth lane

This is not an isolated lever. The gap#1 WKV/SSM state-fidelity de-risk
([`2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-...`](2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-6seed-GO-removes-the-non-fading-store-wall.md))
found that the full WKV recurrence's own divisive num/den normalization step is "hard on spikes" and, having
substituted a plain leaky-integrator form for its own spiking-faithful RUNG 2 GO, explicitly named the
divisive normalization as an optional future enhancement "realizable on spikes via FS [fast-spiking] divisive
inhibition" — the SAME Carandini & Heeger (2012) mechanism this vision-lane sweep is de-risking, independently
arrived at from the conversational-fluency side. A working spiking divisive-normalization operating-point
recipe here would transfer directly as prior art for that lane's own still-open lever.

## Honest scope / caveats

- **capability_go is 0/6 everywhere — this is NOT a GO.** The headline improvement is on the looser
  `overall_verdict`/`beats_config_c_nogo`+`learning_load_bearing` bar, not the strict per-seed capability
  criteria docs/TERMS.md and this lane's own prior findings use as the actual GO line.
- The RATE-ceiling gain (0.61-0.62 vs z's 0.4653) could still be partly a dynamic-range/variance artifact on 6
  examples/class (flagged identically in the original satdiv scope and never independently re-checked here).
- `ridge` was held at the runner's default (0.5) for all 9 cells — the earlier 3-seed ridge sweep (at the
  slightly-off scale=771.1) suggested ridge in {0.25, 1.0} might beat ridge=0.5 at that scale; whether that
  still holds at the now-confirmed-better scale=771.0 is untested (see next grid, below).
- `satdiv` remains an additive lever, default `z`, byte-identical unless `--s2-norm satdiv` is passed
  explicitly — this finding adds no production wiring and flips no default.
- External grounding recorded (full citation in Sources, below): Huang, Zhu & Siew (2006) — a fixed random
  hidden layer (this runner's frozen n_s2=96 S2 template bank is architecturally identical in kind) with a
  trained linear readout
  approximates a target function arbitrarily well as hidden-layer WIDTH grows; this corroborates the
  2026-09-01 finding's diagnosis that the residual may ultimately be bank SIZE (capacity), not normalization
  or (as BCM already tested and NO-GO'd) unsupervised template quality — an untried, simpler alternative lever
  if the finer (sigma, scale, ridge) grid below still plateaus.

## Verdict: BORDERLINE — the exact next grid

Not a GO (capability_go 0/6 x 9 cells). Not confidently NO-GO/exhausted either: the best cell is the strongest
readout found anywhere in this arc, its misses are narrow/structured rather than a collapse, and a genuine
instrument finding (extreme scale-precision sensitivity) shows the axis was under-searched in exactly the
dimension that turns out to matter. The next grid should be a FINER, full-6-seed (no explore/decisive split,
given the demonstrated seed-level sensitivity) sweep bracketing the (sigma=8, scale~771) hot zone, re-tuning
ridge AT that zone rather than reusing the default:

- **sigma in {4, 6, 8, 10, 12}** (bracket the confirmed optimum; 16/32 are already ruled out worse, so no need
  to re-test them)
- **scale in {650, 700, 750, 771, 800, 850}** (finer than the 500/771/1200 grid tested here, since a 0.1-unit
  shift already changed the verdict — this range is still coarser than that, but should localize whether 771 is
  near a true local optimum or itself has room to improve)
- **ridge in {0.1, 0.25, 0.5, 1.0}** (re-tune AT this operating point; the earlier ridge sweep only tested this
  axis at scale=771.1, now shown to be a worse point than 771.0)
- **n=2.0 fixed** (Heeger 1992's n~2-4 range; not yet explored, but sigma/scale/ridge should be localized
  first — n is a secondary knob only worth sweeping if the above still plateaus)
- **All 6 seeds (42, 43, 44, 100, 101, 102) directly** — skip the explore/decisive split this time; the
  scale-precision sensitivity found here means a 3-seed explore pass risks picking an op-point that looks good
  on 3 seeds and collapses on the other 3, exactly the failure mode `capability_go`'s per-seed design exists to
  catch.

That is 5 x 6 x 4 = 120 configs x 6 seeds = 720 runs; at ~20-22s per 6-seed run (observed here), this is a
CPU-only pool job on the order of minutes to low tens of minutes depending on core count, not a GPU or
multi-hour commitment.

## Anti-cheats (reused unchanged from the runner; not modified by this sweep)

Every per-seed pass/fail (`capability_go`'s beat-margins, learning-load-bearing vs a random signed readout,
position-pooled-out, scramble/label-shuffle nulls) is the SAME logic every prior finding in this lane already
verified; this sweep is a pure operating-point search over `--s2-satdiv-sigma`/`--s2-satdiv-scale`, adding no
new pass/fail criteria and no new architecture.

## Sources

<!--derived-->

- Carandini, M. & Heeger, D. J. (2012). Normalization as a canonical neural computation. *Nat. Rev. Neurosci.*
  13:51-62. doi:10.1038/nrn3136 (the satdiv ratio itself).
- Heeger, D. J. (1992). Normalization of cell responses in cat striate cortex. *Visual Neuroscience* 9(2):181-197.
  PMID:1504027 (n~2-4 semi-saturation exponent range).
- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme learning machine: theory and applications.
  *Neurocomputing* 70(1-3):489-501. doi:10.1016/j.neucom.2005.12.126 (fixed-random-hidden-layer capacity scales
  with width — recorded via `tools/record_external_search.sh`, lane `perception (board #135 / #75)`, this session).
- Prior on this substrate: the three `builds_on` findings above (opsweep, readout-exhausted, BCM NOGO), commit
  `ad0648672` (satdiv scoping), and the cross-lane
  [`2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-...`](2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-6seed-GO-removes-the-non-fading-store-wall.md)
  finding (divisive normalization independently named as a next mechanism in the mouth/fluency lane).

## Reproduce

```bash
# One cell of this sweep (sigma=8, scale=771, the best cell found) -- this exact command produced the
# committed artifact this finding cites; --out below overwrites it byte-reproducibly (deterministic, per-op
# seeded, see the runner's own ANTI-CHEATS item 4):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 771 --s2-satdiv-n 2.0 \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/satdiv_sig8_sc771_6seed.json
```
