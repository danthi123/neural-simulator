---
type: finding
status: contributing
date: 2026-09-03
mechanism: vision-lindiscrim-readout / satdiv-divisive-normalization reframe-control attribution + scale precision-fragility re-check
lane: perception (board #135 / #75)
seeds: [42, 43, 44, 100, 101, 102]   # full decisive 6-seed on every cell in this finding
artifacts:
  - research/findings/raw/lanes/perception/satdiv_pooltest_sig8_sc771.json           # 1-seed pipe test (seed 42 only) that printed the 93%-in-control read
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc771_rg0.5_6seed.json    # decisive 6-seed reframe base (ridge=0.5)
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc771_rg0.25_6seed.json   # decisive 6-seed reframe base (ridge=0.25)
  - research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7633_rg0.5_6seed.json   # scale 763.3 (-1%), NEW best cell in the arc
  - research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7711_rg0.5_6seed.json   # scale 771.1, ridge=0.5, full 6-seed (was 3-seed only before)
  - research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7711_rg0.25_6seed.json  # scale 771.1, ridge=0.25, full 6-seed
  - research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7787_rg0.5_6seed.json   # scale 778.7 (+1%), full 6-seed
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc700_rg0.25_6seed.json    # 18-cell local refinement grid (sigma bracket), listed individually below
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc700_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc771_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc771_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc850_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig6_sc850_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc700_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc700_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc850_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig8_sc850_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc700_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc700_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc771_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc771_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc850_rg0.25_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ref_sig10_sc850_rg0.5_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ext_sig4_sc700_6seed.json           # 4-cell sigma-extreme grid
  - research/findings/raw/lanes/perception/satdiv_ext_sig4_sc850_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ext_sig12_sc700_6seed.json
  - research/findings/raw/lanes/perception/satdiv_ext_sig12_sc850_6seed.json
runner: research/runners/_vision_lindiscrim_readout_derisk.py
builds_on:
  - research/findings/2026-09-03-vision-satdiv-divisive-norm-readout-BORDERLINE.md   # commit e4ce3c4b3, the original 9-cell sigma x scale sweep + the 3-seed 771.0-vs-771.1 flip
  - research/findings/2026-09-01-vision-readout-side-exhausted-satdiv-plus-ridge-plateau-points-to-S2-template-learning.md
  - docs/TERMS.md   # GO condition: "the gate's own verdict is positive"
---

# The board #135 satdiv "strict-pass" reading was a terminology mix-up (capability_go stayed 0/6 all along, no verdict crossing ever happened to reconcile); the real question — how much of the readout's effect is shared with a bare V1-direct control — resolves to a seed-variable ~11-16% attributable (not the ~7% a single seed suggested), and the precision-fragility claim is real but was overstated by a small-sample read; a new best-in-arc cell (scale=763.3) is named as the next probe point

## One-line verdict

**capability_go (the strict per-seed bar) is 0/6 on every one of the 30 satdiv cells checked to date, including
every sig8/scale771 artifact** — there was never a "STRICT GO" to reconcile; the `LINDISCRIM-READOUT-GO` string
those artifacts carry is the runner's LOOSER `task_go_5of6_beat_and_lb` bar (beats the config-C NO-GO floor +
beats a random-readout control, both >=5/6), a genuinely different and weaker claim that this lane's own prior
BORDERLINE finding already flagged as insufficient for "GO". **The "93% in control" read does not hold at 93%
across 6 seeds** — it was a single (seed 42) data point from a 1-seed pipe test; at the full decisive 6-seed set
the fraction of the readout's held accuracy that is ALSO present in a bare V1-direct centroid control (no S2
template bank, no satdiv, no learned readout at all) averages **~11% attributable to the manipulation / ~89%
shared with the control** at the original hot-zone cell (sigma=8, scale=771), ranging seed-to-seed from **-2.3%
to +31.1%** (2 of 6 seeds: the bare front end actually BEATS the full pipeline). **The scale-precision-fragility
claim (771.0 vs 771.1 flipping beat-NOGO 5/6->1/3) does not replicate at 6 seeds** — the real flip is milder
(5/6->4/6) — but a genuine, smooth LOCAL GRADIENT across a +/-1% scale window is real, and it points to an
UNEXPLORED region below 771: **scale=763.3 is a new best-in-arc cell** (LEARNED_spkwta_held 0.5052, first-ever
`capability_go=True` on a single seed in this whole arc). Verdict: still **BORDERLINE**, now better-characterized
in both directions (the shared-control confound is real but smaller than feared; the precision sensitivity is
real but smaller than feared, and points to a concrete next probe point).

## What `reframe_test` actually is, and where the "93%" number came from

The runner's top-level JSON `reframe_test` field (`_vision_lindiscrim_readout_derisk.py:855-861`) is a fixed
descriptive string ("learning_load_bearing = learned-minus-random spiking-WTA held >= beat_margin, per seed"),
present byte-identically in every artifact this runner produces — it is not a numeric test and is never "empty"
in the sense of missing data. **The actual numeric control-attribution computation is `tools.lab.attributable_to`**
(`tools/lab.py:288`), called three times inside `_summarize()` (runner lines 629-634), one per candidate shared
term:

1. `LEARNED_spkwta_held` vs `RANDOM_spkwta_held` (identical spike-ported architecture, V untrained) — tests
   whether LEARNING itself is load-bearing.
2. `LEARNED_spkwta_held` vs `centroid_spk_held_NOGO_repro` (config-C's centroid readout on the SAME spike code)
   — tests whether the signed-linear-discriminant readout class beats the old centroid readout.
3. `LEARNED_spkwta_held` vs `A_v1_direct_held` (a plain nearest-centroid decode straight off the C1 spiking
   front end, no S2 template bank, no satdiv normalization, no learned readout) — tests whether the WHOLE
   S2+satdiv+readout architecture adds anything over the bare front end. The runner's own summary labels this
   call `"... -> HIERARCHY (vs V1-direct held)"`.

`attributable_to(label, treatment, control)` prints `"X% of the effect is attributable to the manipulation; Y% is
ALSO PRESENT IN THE CONTROL"` where `X = 100*(treatment-control)/treatment`. This print only goes to stdout — it
is NOT written into the JSON artifact — so a pool-dispatched run's console output is typically not captured,
which is what "the 6-seed GO runs have an empty reframe_test" actually meant: nobody had *computed* this fraction
for the decisive 6-seed artifacts, not that a field was missing. It is a pure function of numbers already stored
in every artifact's `decode_means`/per-seed `decode` blocks, so it can be (and was, here) recomputed after the
fact with no new runs required for the base case.

**The "93%" figure is call #3 (HIERARCHY, vs `A_v1_direct_held`) on `satdiv_pooltest_sig8_sc771.json`, a
1-SEED (seed 42 only) pipe test** run to verify the pool dispatch-execute-sync-back path end to end (its own
`POOL_CHECKED_REASON` says exactly this: *"not a research lever, single seed, throwaway file"*). At seed 42:
`LEARNED_spkwta_held=0.4479`, `A_v1_direct_held=0.4167`, `attributable_to = (0.4479-0.4167)/0.4479 = 6.96%`
attributable, `93.04%` shared with the control — this is the exact number the task-setter saw. **It was never a
6-seed number and was never claimed to be one in the artifact itself** (the pooltest's own `config.seeds` is
`[42]`).

<!--derived-->

## Two different controls exist, and the task's framing conflated them

**Control 1 (RANDOM_spkwta_held) — "is learning load-bearing?" This part of the claim is robust and NOT a
confound.** Across every cell checked in this lane (30 cells total: the original 9-cell sweep, the 18-cell
sigma-bracket refinement, the 4-cell sigma-extreme grid, and the 4-cell scale-precision grid built for this
finding), `learning_load_bearing` is 6/6 or very close to it at every configuration that clears the NO-GO floor
at all, with a per-seed attributable-to-manipulation fraction (vs RANDOM) of roughly 35-65% (mean ~48-49% at the
sig8/771 hot zone). Learning genuinely does something beyond the architecture's own random-projection capacity.

**Control 2 (A_v1_direct_held) — "does the whole S2+satdiv+readout architecture beat a bare front-end read?" This
is where the shared-term concern is real, though weaker than the single-seed number suggested.**

<!--derived-->

| seed | LEARNED (rg=0.5) | A_v1_direct | f attributable to manipulation | LEARNED (rg=0.25) | f attributable |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.4479 | 0.4167 | +7.0% | 0.4583 | +9.1% |
| 43 | 0.4688 | 0.4792 | **-2.2%** | 0.4792 | 0.0% |
| 44 | 0.4479 | 0.4583 | **-2.3%** | 0.4479 | **-2.3%** |
| 100 | 0.4375 | 0.3750 | +14.3% | 0.4271 | +12.2% |
| 101 | 0.5625 | 0.4583 | +18.5% | 0.5729 | +20.0% |
| 102 | 0.4688 | 0.3229 | +31.1% | 0.4479 | +27.9% |
| **mean (of means)** | | | **+11.4%** | | **+11.4%** |
| **mean (of per-seed fractions)** | | | +11.1% | | +11.1% |

At **sigma=8, scale=771 (both ridge=0.5 and ridge=0.25)**, the mean fraction of held accuracy attributable to the
whole S2-template + satdiv-normalization + learned-readout apparatus, over what a plain centroid decode on the
raw C1 spiking front end already achieves, is **~11%** — not 93%, but also not small: **~89% of the readout's
held accuracy is already present in the bare front end**, and on 2 of 6 seeds (43, 44) the front end control
actually **beats** the full pipeline (negative attribution). This is not noise-free — the per-seed range is wide
(-2.3% to +31.1%) — but it is a real, seed-consistent-in-direction pattern, not an artifact of one unlucky seed.
It is also exactly the same computation already latent in the BORDERLINE finding's own "learned-V1 margin"
column (mean +0.0538, failing the `capability_go` bar's `>=0.10` requirement on 4/6 seeds) — this finding adds
the fraction/percentage framing and the seed-by-seed breakdown, but does not contradict that finding.

**At the new best cell (sigma=8, scale=763.3, ridge=0.5)**, the picture improves somewhat: mean attributable
fraction rises to **+16.4%** (range -12.2% to +34.0%, still one negative seed). Better, but still majority-shared
with the control, and the negative-seed pattern persists.

## Precision-fragility, re-checked at the full 6-seed set: real, but overstated by the 3-seed read

<!--derived-->

| scale | delta from 771.0 | ridge | LEARNED_spkwta_held | beats-NOGO | capability_go | overall_verdict |
|---:|---:|---:|---:|:---:|:---:|:---|
| 763.3 | -1.0% | 0.5 | **0.5052** | 5/6 | **1/6** | LINDISCRIM-READOUT-GO |
| 771.0 | 0 (baseline) | 0.5 | 0.4722 | 5/6 | 0/6 | LINDISCRIM-READOUT-GO |
| 771.1 | +0.013% | 0.5 | 0.4601 | 4/6 | 0/6 | PARTIAL-beat4/6-lb6/6 |
| 771.1 | +0.013% | 0.25 | 0.4549 | 3/6 | 0/6 | PARTIAL-beat3/6-lb6/6 |
| 778.7 | +1.0% | 0.5 | 0.4375 | 2/6 | 0/6 | PARTIAL-beat2/6-lb6/6 |

The original BORDERLINE finding's headline instrument claim — a 0.1-unit scale change (771.0 -> 771.1) flips
`beats_config_c_nogo` from 5/6 to 1/3 — was measured on only **3 shared seeds** (42, 43, 100) between the earlier
`vlin_satdiv_ridge0.5_explore.json` (scale=771.1) and the 9-cell sweep (scale=771.0). Run at the **full 6-seed
decisive set**, the actual flip at scale=771.1 is **5/6 -> 4/6**, not 5/6 -> 1/3: still a real, monotonic-with-scale
degradation (LEARNED_spkwta_held falls 0.5052 -> 0.4722 -> 0.4601 -> 0.4375 as scale rises from 763.3 to 778.7,
and beats-NOGO falls 5/6 -> 5/6 -> 4/6 -> 2/6), but the 3-seed sample had exaggerated the size of the effect
(small-N seed noise on a 3/6 vs 1/3 boundary swings a lot on 3 points). **This is itself the useful correction**:
the operating point is scale-SENSITIVE (a genuine local gradient, not a random fluke), but it is not the
knife-edge the 3-seed read implied, and — more importantly — **the gradient is monotonically decreasing across
this whole +/-1% window**, meaning the true local optimum was never confirmed to be AT 771; it may sit below
763.3, unexplored by either the original 9-cell sweep (500/771/1200) or the 18-cell refinement grid (700/771/850)
because neither tested the 750-770 range. **scale=763.3 crossing `capability_go=True` on 1/6 seeds is the first
time any cell in this entire satdiv arc has crossed the strict bar on any seed at all** — still nowhere near the
>=5/6 needed for a GO, but evidence the bar is reachable from this operating-point family, not a hard wall.

## Honest verdict

Not a GO (`capability_go` 0/6 at every scale-771 cell, 1/6 at the best cell found, scale=763.3 — still far short
of a >=5/6 bar). Not the confound the task's framing worried about, either: the "93% in control" read was a
single-seed artifact of a throwaway pipe test, not a property of the decisive 6-seed data, which instead shows a
smaller (~11-16%, still majority-shared) and seed-variable attribution to the manipulation. And the
precision-fragility claim is real (a genuine local scale gradient) but was overstated 3-seed-to-6-seed (5/6->1/3
does not replicate; the real number is 5/6->4/6). **BORDERLINE remains the honest characterization** — this
finding narrows the uncertainty on both of its open questions without closing either.

### What owns the ~11-16%-attributable / ~84-89%-shared gap (the companion process this points to)

Per the wall-reframe rule ("what does the real system run alongside this that we replaced with a constant, before
asking what biology surpasses it"): the shared term is **the C1 spiking front end's own linearly-decodable
information content** — a plain nearest-centroid readout directly on C1 spikes, with NO S2 template bank, NO
satdiv normalization, and NO learned linear-discriminant readout, already achieves ~89% of what the entire
downstream apparatus achieves. Two concrete readings of this, both untried here:

1. **Bank-capacity, not normalization or template quality** (already named by the BORDERLINE finding's own
   external citation, Huang, Zhu & Siew 2006 — a fixed-random-hidden-layer's capacity scales with WIDTH): if the
   frozen `n_s2=96` template bank is the bottleneck, no amount of normalization-form or ridge tuning on top of it
   will close the gap; a width sweep (`--n-s2` 96 -> 192/384) is the next, still-untried, simpler lever than
   another normalization pass.
2. **The position-invariance premise may be smaller than assumed at this task's operating point.** The whole
   point of the S2 template-match -> C2 MAX-over-locations hierarchy (over reading C1 directly) is to buy
   POSITION INVARIANCE that `A_v1_direct_held` (a position-SPECIFIC readout) should lack on held positions. That
   `A_v1_direct_held` is competitive (and on 2/6 seeds, superior) suggests the held positions in this task's
   position lattice (train {0,2,4,6}, held {1,3,5,7} of 8 total) are close enough to trained positions that a
   position-specific code already generalizes substantially — i.e., the invariance the hierarchy is built to buy
   may not be the binding constraint at this specific train/held split. A wider position span or a genuinely
   held-out (not interleaved) split would directly test this and is untried.

Per NO-DEFER: this characterizes the satdiv/S2-architecture METHOD's current ceiling on THIS operating point; it
does not close the vision-readout capability, and names two concrete next levers (bank width; interleaved- vs
held-out-position generalization) rather than stopping at "borderline."

## Sources

- `tools/lab.py:288` `attributable_to()` — the exact computation behind every fraction in this finding.
- Carandini, M. & Heeger, D. J. (2012). Normalization as a canonical neural computation. *Nat. Rev. Neurosci.*
  13:51-62 (satdiv itself; already cited by the BORDERLINE finding, re-cited here for the same mechanism).
- Huang, G.-B., Zhu, Q.-Y. & Siew, C.-K. (2006). Extreme learning machine: theory and applications.
  *Neurocomputing* 70(1-3):489-501 (already recorded in the BORDERLINE finding; re-invoked here as the concrete
  next lever named by the ~89%-shared result: bank WIDTH, not normalization).
- `docs/TERMS.md` GO condition ("the gate's own verdict is positive — never a metric lifted out of a run whose
  verdict was negative") — the basis for treating `capability_go`, not the looser `overall_verdict` string, as
  this lane's actual GO bar.

## Reproduce

```bash
# The reframe/control-attribution base case (no new compute needed -- recomputed from existing decode_means):
python3 -c "
import json
d = json.load(open('research/findings/raw/lanes/perception/satdiv_ref_sig8_sc771_rg0.5_6seed.json'))
for r in d['by_code']['count']['per_seed']:
    dec = r['decode']
    L, A = dec['LEARNED_spkwta_held'], dec['A_v1_direct_held']
    print(r['seed'], L, A, round(100*(L-A)/L, 1), '%')
"

# The new best-in-arc cell (scale=763.3, the -1% precision-grid point), on the pool:
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 763.3 --s2-satdiv-n 2.0 --ridge 0.5 \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7633_rg0.5_6seed.json

# The full 6-seed re-check of the original precision-fragility claim (scale=771.1):
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
    research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 771.1 --s2-satdiv-n 2.0 --ridge 0.5 \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/satdiv_precision_sig8_sc7711_rg0.5_6seed.json

# All 4 precision-grid jobs were dispatched to the mini-PC pool (pool41; pool40 was down, skipped
# automatically), not run locally:
bash tools/pool_queue.sh add "<command above>" --checked "<reason>"
bash tools/pool_sync.sh    # or wait for the pool-sync.timer (every 15 min)
```
