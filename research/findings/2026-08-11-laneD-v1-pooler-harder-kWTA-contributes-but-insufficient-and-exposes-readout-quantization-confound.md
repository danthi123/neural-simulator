---
type: finding
status: negative
date: 2026-08-11
mechanism: perception-v1-pooler-trace-invariance
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_recordedreadout_inhib0.00.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_recordedreadout_inhib0.67.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_finereadout_nex8_inhib0.00.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_finereadout_nex8_inhib0.67.json
---

# lane D perception: harder k-WTA (feedback-inhibitory floor) contributes but does NOT open invariant identity, and it exposes that the recorded PARTIAL-2/3 baseline was a readout-quantization artifact

<!--derived-->
**One-line verdict.** The board's named next lever for the V1->OnSubstratePooler trace route was "stronger inhibitory
competition (harder k-WTA / lateral inhibition so winners are position-invariant by SELECTION, not by rescale)". Built it
as an opt-in, default-off `--inhib-frac`: a feedback-inhibitory conductance FLOOR set at `inhib_frac * peak pool drive`, so
a top-k column fires only if its drive clears the floor (O'Reilly kWTA / PV+ feedback inhibition; Foldiak-style lateral
inhibition), pruning ambiguous held-position winners at selection time. It is applied identically in learning, inference,
and every control arm. Result: it does NOT reach the GO bar for invariant identity, but it is a genuine contributor (it
recovers a clean per-seed 5-gate GO under an honest readout), and running it forced a finer decode readout that shows the
recorded **TRACE-ROUTED-PARTIAL-2/3 baseline is largely a 6-held-image quantization artifact** — with 24 held images the
same operating point is **TRACE-ROUTED-NOGO (0/3)**.

## The lever (runner-side, NO sim/ edit)

<!--derived-->
Plain competitive pooling selects winners by a pure rank cut (`np.argsort(-drive)[:k_win]`): a column wins even at
near-zero drive if it ranks k-th, so an ambiguous held-position drive still fills the code with weakly/non-selective
columns. Cortical/hippocampal k-WTA is not a rank cut -- fast feedback inhibition sets an inhibitory floor proportional
to peak pool activity and a column fires only if its excitatory drive clears it. `TraceV1Pooler._select` keeps a top-k
column only if `drive >= inhib_frac * peak`; the peak column always clears its own floor for `inhib_frac <= 1`, so the
code is never empty while any drive is positive. `inhib_frac = 0` is exact legacy behavior (byte-identical to the
recorded run).

## Result 1 -- recorded (coarse) readout: harder k-WTA regresses the coarse verdict

<!--derived-->
Same recorded sidecar operating point (`--position-axis y --complex-norm local_orient_div --n-col 240 --k-win 8
--pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75`), only `--inhib-frac` toggled. `inhib_frac <= 0.34` prunes
nothing (floor never bites; byte-identical to control). At `0.5`-`0.67` the floor engages during learning:

(`inhib_frac=0.34` is omitted from the table: it is byte-identical to the `0.00` control because the floor never
bites there -- documented above, not a separate arm.)

| inhib_frac | overall verdict | seed GO | held-decode mean | trace margin mean | vs-shuffled mean | vs-no-learning mean |
|---:|---|---|---:|---:|---:|---:|
| 0.00 (control) | TRACE-ROUTED-PARTIAL-2/3 | Y . Y | 0.500 | +0.0837 | -0.0022 | +0.0035 |
| 0.50 | TRACE-ROUTED-PARTIAL-1/3 | . . Y | 0.389 | +0.0677 | -0.0022 | +0.0035 |
| 0.67 | TRACE-ROUTED-PARTIAL-1/3 | . . Y | 0.389 | +0.1167 | +0.0095 | +0.0049 |

<!--derived-->
The coarse verdict regresses 2/3 -> 1/3, but the per-seed structure is the tell. At `0.67` the target-failing **seed 43**
improves: its held-to-train margin rises `+0.026 -> +0.073`, its cross-category cosine DROPS `0.635 -> 0.552`, and it now
beats shuffled-temporal (+0.004) and no-learning (+0.004) -- gates it failed in the control. Its only remaining failing
gate is `decode`. The decode is the coarse part: with `n_categories=3`, `n_held_pos=1`, `n_ex=2` there are only 6 held
images, so decode is quantized to steps of 1/6 (0.333, 0.500). A real margin gain cannot register.

## Result 2 -- honest (finer) readout isolates decode quantization from position difficulty

<!--derived-->
Increasing `--n-ex 2 -> 8` (24 held images, decode step 1/24=0.042) holds the position-generalization task fixed while
de-quantizing the readout (control `research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_finereadout_nex8_inhib0.00.json`,
treatment `research/findings/raw/lanes/perception/v1_pooler_trace_hardkwta_finereadout_nex8_inhib0.67.json`):

| n_ex (held images) | inhib_frac | overall verdict | seed GO | held-decode mean | trace margin mean |
|---:|---:|---|---|---:|---:|
| 2 (6) | 0.00 | TRACE-ROUTED-PARTIAL-2/3 | Y . Y | 0.500 | +0.0837 |
| 8 (24) | 0.00 | **TRACE-ROUTED-NOGO** | . . . | 0.375 | +0.0191 |
| 8 (24) | 0.67 | TRACE-ROUTED-PARTIAL-1/3 | Y . . | 0.389 | +0.0321 |

<!--derived-->
Two facts fall out. (1) The **recorded PARTIAL-2/3 baseline is a small-sample artifact**: under a 24-image readout the
identical control operating point is NOGO (0/3) -- seeds 42/44 that read as GO on 6 images have true held-decode
0.29-0.46 and their margins collapse (seed 44 `+0.070 -> -0.004`). The near-degenerate readout the 2026-08-07 finding
flagged did not merely limit the metric; it INFLATED the baseline. (2) On this honest readout, harder k-WTA at
`inhib_frac=0.67` improves the operating point NOGO -> PARTIAL-1/3 and **recovers seed 42 to a clean GO on all five
gates** (decode 0.583, margin +0.112 beats shuffled/V1/no-learning, pixel-scramble 0.375 collapses). Seeds 43/44 still
have near-zero trace margins, so the lever is real but insufficient.

## Interpretation

<!--derived-->
Harder k-WTA is a genuine, biologically-grounded contributor to invariant selection: applied at winner-selection time it
lowers cross-category code similarity on the previously-failing seed and turns one seed into a clean multi-gate GO under
an honest readout, without fighting the `perm > 0.5` connection threshold that sank the homeostatic-scaling lever
(2026-08-07). But it does not by itself open a multi-seed GO for invariant identity: two of three seeds still lack a
trace-specific margin, so the residual is upstream representation/binding, not the selection rule alone.

The larger deliverable is the confound. Every prior verdict on this route was scored on 6 held images. The competition
lever and the "less-degenerate readout/task" lever the board named as alternatives are therefore NOT independent: the
readout must be de-quantized before any competition change can be evaluated honestly, and doing so reveals the true
baseline is NOGO, not PARTIAL-2/3.

## Next mechanism

<!--derived-->
Bank harder k-WTA (`--inhib-frac`) as a real-but-insufficient contributor: keep it on (~0.5-0.67) and re-baseline the
whole route on the HONEST readout (>=24 held images) so verdicts stop being quantization-limited. The residual is now
localized to the seed-43/44 upstream representation -- the trace binding does not produce a position-invariant code for
two of three seeds even with sharpened competition. The next lever is therefore representation-side, not another
selection threshold: (a) a stronger held-position task with more categories/positions so the code has to generalize
across a wider transformation, and (b) learned V1-complex normalization / decorrelation upstream of the pooler (the
2026-08-03 spike-latency NO-GO reached the same conclusion -- "improve the representation or local learning rule that
feeds the selector"). Do not re-run the coarse 6-image readout; it is not decision-useful.

## 6-seed command (for the coordinator)

<!--derived-->
The `--out` is a bare filename here (the raw-lane path is a not-yet-created output, not a cited artifact); land it
under the perception raw lane.

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 100 101 102 --position-axis y --complex-norm local_orient_div \
  --n-col 240 --k-win 8 --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75 \
  --n-ex 8 --inhib-frac 0.67 \
  --out v1_pooler_trace_hardkwta_finereadout_nex8_inhib0.67_6seed.json
```
