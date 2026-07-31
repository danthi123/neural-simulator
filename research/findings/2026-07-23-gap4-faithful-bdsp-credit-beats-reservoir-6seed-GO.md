---
type: finding
status: live
date: 2026-07-23
mechanism: gap4-credit
---

# gap#4 deep-credit ENABLER — faithful BDSP/FA credit beats a frozen reservoir, 6-seed GO (2026-07-23)

## Result — GO, 6/6 seeds, credit > reservoir at every sparsity + the gap GROWS with sparsity
`_gap4_bdsp_faithful_credit_derisk.py --seeds 42 43 44 100 101 102 --hidden 256 --depth 2 --fracs 1.0 0.1 0.05 --lr 0.03`
(numpy CPU, local, ~minutes, $0; a faithful numpy replica of `sim/kernels.fused_bdsp_update`):
```
sparsity   reservoir   fa_linear   bdsp_nocoinc   bdsp || best-credit gap   seeds credit>res+0.01
1.0        0.785       0.940       0.918          0.802 ||   +0.155           6/6
0.1        0.651       0.843       0.880          0.887 ||   +0.228           6/6
0.05       0.535       0.779       0.834          0.810 ||   +0.299           6/6
```

## What this establishes (the deep-credit enabler for open-prose)
A biological credit-assignment rule that TRAINS THE HIDDEN LAYER — feedback alignment (fixed random feedback, **no
weight transport**) and BDSP (the committed three-term dendritic rule) — BEATS a frozen RESERVOIR (hidden weights
random-frozen, only the readout trained = the credit-INDEPENDENT baseline) at every spiking sparsity, 6/6 seeds. The
advantage GROWS as spiking gets sparser (+0.155 dense → +0.299 at 5%-active) — i.e. directed credit to hidden units
is load-bearing exactly in the BIOLOGICAL sparse-spiking regime where a reservoir's random features are weakest.
This is the enabler evidence for the "brain's own open-prose Broca" (which needs a credit rule, not backprop): the
project HAS a local, transport-free, biology-grounded deep-credit rule that outperforms the no-credit baseline.

## Scope / honesty (per the AWS-scoping audit `2026-07-23-deep-credit-aws-revalidation-spec.md`)
- This is the ENABLER evidence (credit rule > reservoir), NOT the deep-credit-to-ACCURACY-on-the-spiking-bridge claim
  — the board's separate verdict on the on-bridge learn-to-accuracy is a CONFIRMED clean NEGATIVE (credit ≈ lesion ≪
  reservoir), a characterized deprioritized frontier. These are different questions; this firms the enabler, not the
  on-bridge accuracy.
- The 2026-07-17 unseeded-substrate bug does NOT apply here: this runner has no `SimulationBridge` (it seeds via
  `numpy default_rng(seed)` + `DendriticMLP(seed=)`), so the confound the deep-credit-arc audit flagged never touched
  it. The seed bug is separately FIXED + verified (commit 9471908a; `test_determinism.py::TestSubstrateActuallySeeded`).
- `--lr 0.03` (not the runner default 0.3, which drove a dense→chance artifact). numpy CPU — correctly run LOCAL (the
  allocation principle: the bottleneck is the GPU/training, this is CPU + non-contending). NO `sim/` edit.

## Net
The deep-credit enabler (a transport-free biological rule beats a frozen reservoir, robustly, and more so at
biological sparsity) is now 6-seed GO. The open-prose Broca's credit-assignment prerequisite has firm evidence; the
on-bridge learn-to-accuracy remains the characterized-negative deprioritized frontier.
