---
type: finding
status: partial
date: 2026-09-03
lane: perception (board #135 / #75)
mechanism: vision-configural-binding-conjunctive-S2.5
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/perception/conjbind_baseline_off_6seed.json
  - research/findings/raw/lanes/perception/conjbind_fixed_min_n192_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n192_6seed.json
  - research/findings/raw/lanes/perception/conjbind_shuffle_n192_6seed.json
  - research/findings/raw/lanes/perception/conjbind_delta0_n192_6seed.json
runner: research/runners/_vision_lindiscrim_readout_derisk.py
builds_on:
  - research/findings/2026-09-03-vision-configural-binding-DESIGN.md
  - research/findings/2026-09-03-vision-satdiv-divisive-norm-readout-BORDERLINE.md
  - research/findings/2026-09-03-satdiv-readout-mostly-in-control-BORDERLINE-refined.md
  - research/findings/2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md
  - research/findings/2026-09-01-vision-s2-bcm-template-learning-NOGO-collapses-without-competition-underperforms-baseline-with-it.md
  - research/findings/2026-08-19-vision-spiking-hierarchy-frontend-holds-configural-readout-quantization-limited.md
  - research/findings/2026-08-19-vision-hmax-hierarchy-composed-pooling-solves-position-invariance-learning-not-load-bearing.md
  - research/biology/coincidence-binding.md
---

# The S2.5 configural-binding conjunctive layer BUILT and RUN: 3/6 seeds cross `capability_go` (below the 5/6 GO bar), but the three binding-specific anti-cheats show the lift is genuinely, if partially, attributable to the RELATIVE-OFFSET binding, not to added width -- a real, no-defer partial signal, not a clean GO

## One-line verdict

The relative-offset conjunctive S2.5 stage from the 2026-09-03 design doc was built (additive, default-off,
byte-identical-off verified in the data) and run at the design's cheapest de-risk point (`--conj-bind fixed
--conj-n 192 --conj-offset-max 4 --conj-mode min`, 6 seeds, CPU). It **does not clear the strict `capability_go`
bar** (3/6 seeds, not >=5/6) -- **this is a NO-GO against the design's own pre-registered gate**, so the honest
top-line status is NOT a capability crossing. But it is not an inert result either: relative to the
byte-identical baseline (0/6, `LEARNED_spkwta_held` mean 0.4375), the mechanism lifts held accuracy to 0.5226
(+0.0851 <!--derived-->) and beat-count from 3/6 to 6/6 -- and, decisively, **both anti-cheats that break the
RELATIVE correspondence (offset-shuffle lesion, Delta=0 degenerate) collapse to within ~0.01-0.02 of the
untouched baseline (86% and 82% of the lift vanishes respectively <!--derived-->)**, while the width-matched
flat control (more random features, no binding at all) retains 57% of the lift <!--derived-->. The three
controls therefore separate cleanly in the predicted order (main > width-control > {shuffle, delta0} approx=
baseline), which is exactly the signature the design's own anti-cheats were built to detect if binding is real:
**part of this lift (the 43% margin between main and the width-matched control) is genuinely attributable to
the relative-offset conjunction, not to added random capacity** <!--derived-->. The mechanism is real but too
weak, on this exact operating point, to cross the strict `capability_go` bar on more than half the seeds -- a
**partial, no-defer result**, not a wall.

## What was built

Per the design's exact hooks (`research/findings/2026-09-03-vision-configural-binding-DESIGN.md` Part 2c/2d),
inserted into `research/runners/_vision_lindiscrim_readout_derisk.py`, no `sim/` edit:

- **`_make_conjunction_bank(n_s2, conj_n, offset_max, delta0_only, seed)`** -- samples the `(template_a,
  template_b, Delta)` triples for the conjunctive bank **ONCE PER SEED** (the design's Part 2b: "sample the
  triples once per seed"), reused UNCHANGED across the train/held/scramble splits exactly as the frozen S2 bank
  `W0` already is. (The design's code sketch derived the sampling RNG from the per-split `base_seed`, which
  differs across the train/held/scramble calls in `run_seed` -- taken literally that would give each split a
  DIFFERENT conjunction bank and silently break the readout; this build follows the design's own PROSE spec
  ["once per seed"] over its sketch's literal per-call wiring.)
- **`_bind_conjunctions(drive, pairs, offsets, mode, shuffle_seed=None)`** -- the S2.5 stage itself:
  `conj_c = MAX_p AND(drive[p,a], drive[p+Delta,b])`, inserted in both `_c2_spike_code` and `_c2_rate_code`
  right after `_apply_s2_norm`/`_kwta_over_templates` and before the existing `.max(axis=1)` C2 pool. `mode`
  'min' (conservative AND, used for this de-risk) or 'prod' (supralinear NMDA-like AND); 'coincidence' (not
  exercised here) additionally raises the LIF gain 1.6x at the spiking S2 layer for the predicted-positive
  spiking arm. `shuffle_seed`, when set, redraws afferent b at an independent permuted grid column instead of
  `p+Delta` (anti-cheat 2, the offset-shuffle lesion) -- freshly seeded inside the function so the SAME
  `shuffle_seed` + grid size reproduces the identical lesion wiring across the train/held/scramble calls of one
  seed.
- **CLI flags** (additive, default-off): `--conj-bind {none,fixed}` (default `none`; the design's speculative
  `learned` selection rung is NOT offered -- it has no implementation, only a Part-2d sketch, and this build did
  not fabricate a fake option), `--conj-n`, `--conj-offset-max`, `--conj-mode {min,prod,coincidence}`,
  `--conj-delta0-only`, `--conj-shuffle-offsets`.
- **A pre-existing shape bug fixed in passing**: the RANDOM-control arm built `Vr` with a hard-coded
  `(a.n_classes, a.n_s2)` shape instead of reading D from the trained readout (`V.shape[1]`) -- harmless while
  every C2 code was exactly `n_s2`-wide, but it would silently mismatch (or crash) the instant the feature count
  changes, which `--conj-bind fixed` does (`n_conj` != `n_s2` in general). Fixed to `V.shape[1]`; a no-op when
  binding is off (`V.shape[1] == a.n_s2` there, unchanged).

All other hooks are reused exactly as the design's table specifies: `_train_linreadout`/`_spiking_class_read`/
`_lin_score_pred` read `D` from the array shape (verified: `r_tr`/`rr_tr` flow through with `n_conj` instead of
`n_s2` columns with zero further edit), the stimulus/positions/GO gate/anti-cheat helpers (`_object_classes`,
`_positions`, `_scramble_images`, `_centroid_decode`, `_within_split_decode`) are untouched imports.

## Byte-identical-off, verified in the data (docs/TERMS.md: asserted, not inferred)

Ran the HEAD (pre-build) version of the runner and this build's version, both with `--conj-bind` at its default
(`none`), same 6 seeds, same `--code count`:

```
by_code EQUAL: True
overall_verdict EQUAL: True   LINDISCRIM-READOUT-PARTIAL-beat3/6-lb6/6
mechanism EQUAL: True
```

The only diff between the two output JSONs is the config echo (`vars(a)` now includes the 6 new `conj_*`
argparse defaults) and the `--out` path -- every decode/verdict/summary field, including all rounded floats, is
identical. Artifact: `research/findings/raw/lanes/perception/conjbind_baseline_off_6seed.json` (this run,
written by the built runner with `--conj-bind` at its default) is also the baseline row used throughout this
finding. Determinism of the NEW code paths (conj sampling + the offset-shuffle lesion's internal RNG) was
separately verified by re-running `--conj-bind fixed --conj-shuffle-offsets --seeds 42` twice and byte-comparing
(minus `elapsed_seconds`/`--out`): identical.

## The de-risk results

6 seeds (42/43/44/100/101/102), CPU/numpy, `--code count` (both the rate ceiling and the spiking readout are
computed in one run). Each run ~17-24s wall, peak RSS ~516 MB for a 1-seed run (measured via
`resource.getrusage`, well under the 4 GB budget for the full 6-seed sweep).

| arm | `LEARNED_spkwta_held` mean | lift over baseline | `capability_go` | `beats_config_c_nogo` | artifact |
|---|---:|---:|---:|---:|---|
| **baseline** (`--conj-bind none`) | 0.4375 | -- | **0/6** | 3/6 | `conjbind_baseline_off_6seed.json` |
| **main** (`fixed`, n=192, min, Delta<=4) | 0.5226 | +0.0851 <!--derived--> | **3/6** | 6/6 | `conjbind_fixed_min_n192_6seed.json` |
| width-matched control (`--n-s2 192`, no binding) | 0.4861 | +0.0486 (57%) <!--derived--> | **1/6** | 4/6 | `conjbind_widthctrl_n192_6seed.json` |
| offset-shuffle LESION | 0.4497 | +0.0122 (14%) <!--derived--> | **1/6** | 3/6 | `conjbind_shuffle_n192_6seed.json` |
| Delta=0 degenerate | 0.4531 | +0.0156 (18%) <!--derived--> | **0/6** | 3/6 | `conjbind_delta0_n192_6seed.json` |

("lift over baseline" and its bracketed percent-of-main's-lift are computed from the table's own means, not
independently re-derived elsewhere.) <!--derived-->

Per-seed `capability_go` for the main mechanism: `[False, False, False, True, True, True]` (seeds
42/43/44 fail; 100/101/102 pass) -- **3/6, below the design's own >=5/6 GO bar.**

### Why 3/6 fails: the +0.10-over-V1-direct margin, not raw accuracy

`capability_go`'s decisive sub-condition on the 3 failing seeds is specifically "beats V1-direct-held by
>=+0.10" (`LEARNED_spkwta_held - A_v1_direct_held >= 0.10`), not overall accuracy:

| seed | `LEARNED_held` | `A_v1_direct_held` | margin (needs >=0.10) | `position_pooled_out` | `capability_go` |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5 | 0.4167 | +0.0833 (fails) <!--derived--> | **False** (pos leak: 0.4375 > chance+0.15=0.40) | False |
| 43 | 0.5312 | 0.4792 | +0.052 (fails) <!--derived--> | True | False |
| 44 | 0.5 | 0.4583 | +0.0417 (fails) <!--derived--> | True | False |
| 100 | 0.5417 | 0.375 | +0.1667 (passes) <!--derived--> | True | True |
| 101 | 0.5729 | 0.4583 | +0.1146 (passes) <!--derived--> | True | True |
| 102 | 0.4896 | 0.3229 | +0.1667 (passes) <!--derived--> | True | True |

For seeds 43/44 the margin-over-V1-direct is the ONLY failing sub-condition; for seed 42, position also leaks
(`position_pooled_out=False`) as a second, independent failure. `A_v1_direct_held` (the V1-direct floor) itself
ranges 0.3229-0.4792 across seeds (a near-2x spread) -- on seeds where that floor happens to sit high (43:
0.4792, 44: 0.4583), even a real, consistent absolute lift from binding struggles to clear a FIXED +0.10 margin
over a NOISY denominator. <!--derived--> This is a measured property of the gate's interaction with this task's
seed-to-seed variance, not a re-interpretation that waives the gate -- `capability_go` is 3/6 and that is the
honest number.

## The three anti-cheats: the lift is genuinely, if partially, binding-attributable

The single biggest risk named in the design (Huang, Zhu & Siew 2006's ELM capacity-lift-from-width confound) is
directly addressed by comparing the three controls' RETAINED FRACTION of the main mechanism's lift over baseline:

1. **Width-matched flat control (`--n-s2 192`, no binding) -- PASSES the anti-cheat (does not clear the bar).**
   `capability_go` = 1/6 (<< the 5/6 bar), `beats_config_c_nogo` = 4/6. It retains 57% of the lift
   (+0.0486 of +0.0851 <!--derived-->) -- width alone is real and non-trivial, but it does NOT spuriously cross
   the strict GO bar the way the design's failure-mode-1 worried it might, and it sits clearly below `main` on
   every metric in the table (GO 1/6 vs 3/6, beat 4/6 vs 6/6, mean 0.4861 vs 0.5226).
2. **Offset-shuffle lesion -- PASSES the anti-cheat (capability collapses toward baseline).** `capability_go`
   drops 3/6 -> 1/6 and, more decisively, the LIFT over baseline collapses from +0.0851 to +0.0122 <!--derived-->
   -- **86% of the lift vanishes** when afferent b is read at an independent, unrelated location
   instead of `p+Delta`. The residual +0.0122 <!--derived--> is smaller than the seed-to-seed noise band visible
   elsewhere in this same arc (e.g. the baseline's own per-seed `LEARNED_spkwta_held` ranges 0.38-0.49, a 0.11
   spread), i.e. statistically indistinguishable from "no binding at all" at this seed count.
3. **Delta=0 degenerate control -- PASSES the anti-cheat (does not clear the bar).** `capability_go` = 0/6.
   Lift over baseline collapses to +0.0156 (18% retained) <!--derived--> -- same-location "AND" (no cross-slot
   relation) contributes almost nothing beyond the width effect, confirming the binding's value is specifically
   in the CROSS-LOCATION relative offset, not in the coincidence nonlinearity by itself.

**The four-way ordering is exactly the predicted signature of genuine (if insufficient) binding**: main
(0.5226) > width-control (0.4861) > {delta0 (0.4531), shuffle (0.4497)} approx= baseline (0.4375). Both controls
that specifically break the RELATIVE correspondence (shuffle, delta0) land close to each other and clearly
BELOW the width-matched control that has the SAME feature count but no AND nonlinearity at all -- if the whole
effect were an ELM capacity artifact, shuffle/delta0 (which still apply an AND nonlinearity, just at the wrong
locations) should have scored AT LEAST as well as the width control; instead they score lower. The
43-percentage-point-of-lift gap between `main` and the width-matched control (0.5226-0.4861=+0.0365, out of a
total lift of +0.0851) is the part of this result attributable to configural binding specifically. <!--derived-->

## Honest verdict

**NO-GO against the design's pre-registered `capability_go` >=5/6 gate.** The mechanism as built and run at this
single, cheapest operating point (`n_conj=192`, `offset_max=4`, `mode=min`) does not establish the capability at
the strict bar this arc has used throughout (satdiv, BCM, k-WTA all NO-GO'd or plateaued against the identical
gate). Unlike those prior levers, though, this one's anti-cheats show the improvement it DOES produce is not a
width/capacity artifact (the width-matched ELM control stays well below both `main` and the strict bar, and the
two lesions that specifically break the relative-offset correspondence collapse toward baseline) -- **this is a
first-class partial positive on the REPRESENTATIONAL diagnosis** (configural binding is a real, measurable,
if under-powered, lever on this task), not a repeat of the readout-side plateau the design was built to escape.
Per the project's standing law, a negative is a verdict on the METHOD (this exact operating point), not a
license to abandon the capability.

**Named next lever (no-defer):** the failure mode is concentrated in the fixed +0.10 margin over a
seed-noisy V1-direct floor, not in raw held accuracy (which is respectably above the NO-GO floor on 6/6 seeds).
Two concrete next steps, both already named by the design as the intended progression:
1. **The design's own deferred `learned` rung** (Part 2d: score the fixed random conjunction bank against train
   labels, keep the top-k most discriminative `(a,b,Delta)` triples) directly targets this -- a supervised,
   sparse, SELECTIVE conjunction code should raise the absolute margin more reliably across seeds than a fully
   random bank, exactly the "discriminative sparse selective S2" the 2026-08-19 spiking finding predicted
   becomes load-bearing.
2. **A wider or larger conjunction bank** (design failure mode 4: pairs may only partially lift a 3-slot
   arrangement; third-order/triple conjunctions, or simply a larger `--conj-n`/`--conj-offset-max` sweep) is the
   cheaper, purely-random-still lever to try before committing to supervised selection.

The spiking (`coincidence` mode) arm and a fuller operating-point sweep were NOT run in this de-risk (the design
scopes the spiking arm as "after rate clears," and this rate/count-code de-risk did not clear at the required
5/6 -- spending the spiking arm's compute on an already-insufficient operating point would not have been
informative; the honest next round is to first find an operating point/selection rule that clears at rate).

## Brain-based status

Unchanged front end: somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory, per-step
membrane noise) at S1, S2, and the readout class populations. The new S2.5 stage's AND (`min`/`prod`) is the
rate-coded proxy for a coincidence-detector soma (`research/biology/coincidence-binding.md`, status:
established); the `coincidence` mode (not exercised here) additionally realizes it as a raised-threshold LIF
soma. FLAGGED scaffolds (same status as the rest of this runner's config B/C front end): retinotopic
weight-sharing + pooling windows; the fixed random S2 bank; and now the fixed random `(a,b,Delta)` conjunction
sampling (explicitly the "fixed-first" design choice, isolating the representational question from any learning
question -- the `learned`/selective rung is the next, not-yet-built, mechanism). No `sim/` edit; `_bind_conjunctions`
and `_make_conjunction_bank` are standalone numpy functions operating on the same dense `(N, n_loc, n_S2)`
drive array the existing S2 norm/k-WTA functions already consume.

## Reproduce

```bash
# byte-identical baseline (unchanged default, --conj-bind none):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --code count --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_baseline_off_6seed.json

# 1. Fixed configural binding (the mechanism):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --code count --conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode min \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_fixed_min_n192_6seed.json

# 2. Width-matched flat ELM control (anti-cheat 1):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --code count --n-s2 192 --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_widthctrl_n192_6seed.json

# 3. Offset-shuffle lesion (anti-cheat 2):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --code count --conj-bind fixed --conj-n 192 --conj-offset-max 4 --conj-mode min --conj-shuffle-offsets \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_shuffle_n192_6seed.json

# 4. Delta=0 degenerate control (anti-cheat 3):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --code count --conj-bind fixed --conj-n 192 --conj-mode min --conj-delta0-only \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_delta0_n192_6seed.json
```

## Sources

- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nature
  Neuroscience* 2:1019-1025.
- Tanaka, K. (1996). Inferotemporal cortex and object vision. *Annual Review of Neuroscience* 19:109-139.
- Tsunoda, K., Yamane, Y., Nishizaki, M. & Tanaka, K. (2001). *Nature Neuroscience* 4:832-838.
- Ullman, S., Vidal-Naquet, M. & Sali, E. (2002). *Nature Neuroscience* 5:682-687.
- von der Malsburg, C. (1981/1999); Treisman, A. & Gelade, G. (1980); Singer, W. & Gray, C. M. (1995); Ghose,
  G. M. & Maunsell, J. (1999) -- the binding-problem / conjunctive-cell literature the design draws on.
- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). *Neurocomputing* 70(1-3):489-501 -- the ELM width-capacity
  confound this de-risk's anti-cheat 1 tests for (and which does NOT explain the observed lift alone).
- The design doc itself: `research/findings/2026-09-03-vision-configural-binding-DESIGN.md` (full diagnosis,
  mechanism spec, hook table, and the anti-cheat rationale this de-risk executes).
