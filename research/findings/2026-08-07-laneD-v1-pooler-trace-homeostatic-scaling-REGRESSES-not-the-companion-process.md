---
type: finding
status: contributing
date: 2026-08-07
mechanism: perception-v1-pooler-trace-invariance
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075_homeoCONTROL.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075_homeoTREATMENT.json
---

# lane D perception: Turrigiano homeostatic synaptic scaling on the trace pooler REGRESSES the operating point (not the missing companion process)

<!--derived-->
**One-line verdict.** The board's named next lever for the PARTIAL-2/3 V1->OnSubstratePooler trace route was
"add homeostatic scaling (Turrigiano)". Built it as an opt-in, default-off `--homeo-scale` on the pooler
(multiplicative renormalization of each column's incoming feedforward permanence sum toward the developmental
baseline set-point, after each epoch). Like-for-like at the recorded PARTIAL-2/3 operating point, it does NOT open
a propagate-AND-selective operating point: it **regresses** the route from **TRACE-ROUTED-PARTIAL-2/3 to
PARTIAL-1/3**. This is a verdict on THIS method, not the capability; the trace rule itself remains banked GO on
the competitive pooler, and two un-tried levers (stronger inhibitory competition, a less degenerate held-position
readout) remain open.

## Like-for-like result (same operating point, only `--homeo-scale` toggled)

<!--derived-->
Both arms run in-session on the exact recorded sidecar operating point
(`--position-axis y --complex-norm local_orient_div --n-col 240 --k-win 8 --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75`).
The control arm reproduces the recorded PARTIAL-2/3 baseline byte-for-metric (seed 43 is the pre-existing failure).

```bash
# control (baseline, reproduces the recorded PARTIAL-2/3)
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 --position-axis y --complex-norm local_orient_div \
  --n-col 240 --k-win 8 --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75 \
  --out research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075_homeoCONTROL.json

# treatment (adds --homeo-scale, nothing else changed)
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 --position-axis y --complex-norm local_orient_div \
  --n-col 240 --k-win 8 --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75 --homeo-scale \
  --out research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075_homeoTREATMENT.json
```

| metric | control (no homeo) | +homeo scaling |
|---|---:|---:|
| overall verdict | TRACE-ROUTED-PARTIAL-2/3 | TRACE-ROUTED-PARTIAL-1/3 |
| seed GO count | 2/3 | 1/3 |
| held-position decode mean | 0.500 | 0.389 |
| trace margin mean | +0.0837 | +0.0903 |
| shuffled-temporal margin mean | -0.0022 | +0.0139 |
| no-learning margin mean | +0.0035 | +0.0035 |

Per seed:

| seed | arm | GO | decode | margin | vs-shuffled | vs-V1 | scramble decode | failing gates |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 42 | control | yes | 0.500 | +0.155 | +0.158 | +0.183 | 0.333 | none |
| 42 | +homeo | no | 0.333 | +0.154 | +0.154 | +0.182 | **0.500** | decode |
| 43 | control | no | 0.500 | +0.026 | +0.027 | +0.043 | 0.333 | shuffled, no-learning |
| 43 | +homeo | no | 0.333 | **-0.009** | -0.001 | +0.007 | 0.333 | decode, shuffled, V1, no-learning |
| 44 | control | yes | 0.500 | +0.070 | +0.073 | +0.092 | 0.167 | none |
| 44 | +homeo | yes | 0.500 | +0.126 | +0.077 | +0.148 | 0.333 | none |

## Interpretation

<!--derived-->
Homeostatic synaptic scaling is not the missing companion process at this operating point. Two failure signatures,
both consistent with the same cause: (1) the target failure seed 43 got *worse*, not better — its held-to-train
margin flipped from +0.026 to -0.009 and it now fails every margin gate; (2) seed 42, previously a clean GO, lost
its held-position decode (0.500 -> 0.333, chance) AND leaked its pixel-scramble control (0.333 -> 0.500), i.e. the
code became less position-invariant, not more. The mean trace-vs-shuffled gap did not improve in a usable way (the
shuffled arm's margin also rose, +0.0139), so the extra margin is not trace-specific.

The most likely mechanism is an interaction with the pooler's connectivity threshold: drive counts only CONNECTED
synapses (`perm > 0.5` in `_drive`), so a *multiplicative* rescale toward a single per-column set-point pushes
trace-learned permanences across the 0.5 connection boundary in both directions, corrupting the selective code
rather than stabilizing it. Turrigiano scaling preserves *relative* weights, but this substrate reads a
*thresholded* subset of them, so scale changes are not selectivity-neutral here. Only seed 44 tolerated it.

## Next mechanism

<!--derived-->
Bank homeostatic multiplicative synaptic scaling as tested-NEGATIVE for this route/operating point (do not re-run at
6 seeds; it regresses). Of the board's three named candidates for opening a propagate-AND-selective operating point,
homeostatic scaling is now spent; the two remaining un-tried levers are: (a) **stronger inhibitory competition** in
the pooler (harder k-WTA / lateral inhibition so winners are position-invariant by selection, not by rescale), and
(b) a **less degenerate held-position readout/task** (the decode gate sits at exactly 0.500 on 3 categories in the
control, i.e. the task itself is near-degenerate for the current V1 features). A companion process that does not
fight the `perm > 0.5` threshold — e.g. competition applied at winner-selection time rather than as a post-hoc
weight rescale — is the more promising of the two.
