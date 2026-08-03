---
type: finding
status: contributing
date: 2026-08-02
mechanism: perception-v1-pooler-trace-invariance
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/v1_pooler_trace_smoke.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_3seed_default.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol320_k8_lr08_ld01_td075.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k12_lr08_ld01_td075.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol320_k12_lr08_ld01_td075.json
---

# lane D perception: routing the trace rule through V1 -> OnSubstratePooler is built, but default CPU settings are NOGO

<!--derived-->
**One-line verdict.** The corrected route was implemented: pixels go through the existing Gabor/V1-complex front end,
then an `OnSubstratePooler` is trained with the EMERGE-50 Foldiak trace rule over same-category position-jittered
temporal bouts. That discharges the previous dead-forward V2/IT artifact, but the default routed pass is still
**TRACE-ROUTED-NOGO**: held-position decode stays at chance and the trace margin is too small to beat shuffled/no-learning
controls. The best sidecar operating point reached **TRACE-ROUTED-PARTIAL-2/3**, while larger `n_col`/`k_win` variants
got weaker, so raw pool expansion is not the next useful move.

## Default 3-seed result

<!--derived-->
Command:

```bash
env SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --out research/findings/raw/lanes/perception/v1_pooler_trace_3seed_default.json
```

Aggregate:

| metric | result |
|---|---:|
| verdict | TRACE-ROUTED-NOGO |
| seed GO count | 0/3 |
| chance | 0.333 |
| held-position decode mean | 0.333 |
| trace margin mean | +0.0078 |
| shuffled-temporal margin mean | -0.0017 |
| V1-complex margin mean | -0.0181 |
| no-learning margin mean | +0.0043 |

Per seed:

| seed | GO | decode | trace margin | shuffled | V1 | no-learning | scramble decode | failing gates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 42 | no | 0.333 | +0.010 | 0.000 | -0.011 | +0.027 | 0.333 | decode, shuffled, no-learning |
| 43 | no | 0.333 | +0.000 | 0.000 | -0.019 | +0.001 | 0.333 | decode, shuffled, V1, no-learning |
| 44 | no | 0.333 | +0.013 | -0.005 | -0.023 | -0.016 | 0.500 | decode, shuffled, no-learning, scramble |

## Interpretation

<!--derived-->
The routed path is real enough to produce small positive trace-vs-V1 margins on two seeds, but not a usable invariant
code. It does not justify a 6-seed promotion in this form: the central decode gate is exactly chance, and the learned
pooler does not consistently beat either shuffled temporal order or no-learning by the registered margin. This is a
verdict on the current routed operating point/task settings, not on the trace rule itself, which remains banked GO in
the original competitive-pooler setting.

## Next mechanism

<!--derived-->
Do not repeat this exact default run at larger seed count. The next useful lane-D work is an operating-point or
companion-process change: strengthen the task/readout so V1 features are not already near-degenerate for the
held-position decode, add the normalization/homeostatic controls named by the structural map, or move the trace
eligibility fully on-substrate. The old deployed V2/IT path remains retired; this runner is the path to tune, but it is
not yet a positive invariance result.

## Sidecar operating-point follow-up

<!--derived-->
An opt-in sidecar pass added `--position-axis`, `--complex-norm`, and `--orientation-offset-deg` to the runner and
tested a y-axis held-position task with local orientation divisive normalization:

```bash
.venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 --position-axis y --complex-norm local_orient_div \
  --n-col 240 --k-win 8 --pool-lr-pot 0.08 --pool-lr-depress 0.01 --trace-decay 0.75 \
  --out research/findings/raw/lanes/perception/v1_pooler_trace_sidecar_3seed_yaxis_localdiv_ncol240_k8_lr08_ld01_td075.json
```

This improved the result from 0/3 to **TRACE-ROUTED-PARTIAL-2/3**, not GO:

| metric | default | sidecar |
|---|---:|---:|
| seed GO count | 0/3 | 2/3 |
| held-position decode mean | 0.333 | 0.500 |
| trace margin mean | +0.0078 | +0.0837 |
| shuffled-temporal margin mean | -0.0017 | -0.0022 |
| V1-complex margin mean | -0.0181 | -0.0223 |
| no-learning margin mean | +0.0043 | +0.0035 |

| seed | GO | decode | trace margin | shuffled | V1 | no-learning | scramble decode | failing gates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 42 | yes | 0.500 | +0.155 | -0.003 | -0.029 | -0.003 | 0.333 | none |
| 43 | no | 0.500 | +0.026 | -0.001 | -0.017 | +0.000 | 0.333 | trace-vs-shuffled, trace-vs-no-learning |
| 44 | yes | 0.500 | +0.070 | -0.003 | -0.022 | +0.013 | 0.167 | none |

Interpretation: the route and controls are now live enough to produce a real partial, and the controls were not
weakened: shuffled and no-learning stay near zero, scramble stays collapsed. The remaining problem is seed-43 margin
size, so the next lane-D step is normalization/homeostasis or a stronger held-position task before claiming GO.

## Pool-size and winner-count variants

<!--derived-->
Three cluster follow-ups tested whether the partial sidecar was simply under-provisioned:

```bash
.venv/bin/python -u -m research.runners._laneD_v1_pooler_trace_invariance_derisk \
  --seeds 42 43 44 --position-axis y --complex-norm local_orient_div \
  --n-col {320,240,320} --k-win {8,12,12} --pool-lr-pot 0.08 \
  --pool-lr-depress 0.01 --trace-decay 0.75 --out ...
```

They did not improve the result:

| variant | verdict | per-seed GO | held-position decode mean | trace margin mean | shuffled margin mean | V1 margin mean | no-learning margin mean |
|---|---|---|---:|---:|---:|---:|---:|
| n_col=240, k_win=8 | TRACE-ROUTED-PARTIAL-2/3 | yes, no, yes | 0.5000 | +0.0837 | -0.0022 | -0.0223 | +0.0035 |
| n_col=320, k_win=8 | TRACE-ROUTED-NOGO | no, no, no | 0.2778 | -0.0608 | +0.0091 | -0.0223 | -0.0061 |
| n_col=240, k_win=12 | TRACE-ROUTED-PARTIAL-1/3 | no, yes, no | 0.2778 | -0.0587 | +0.0188 | -0.0223 | +0.0003 |
| n_col=320, k_win=12 | TRACE-ROUTED-NOGO | no, no, no | 0.2222 | -0.0523 | +0.0286 | -0.0223 | -0.0038 |

Interpretation: the best signal is still the smaller sidecar pool. Increasing columns or winners makes the trace code
less stable on this task, probably by spreading already-weak feature evidence instead of solving the missing
normalization/competition problem. The next test should therefore change the biological companion process, not just the
pool size: candidates are on-substrate homeostatic normalization, stronger inhibitory competition, or a less degenerate
held-position readout/task.
