# TRUE-ONE-BRAIN #5 δ-probe — the self-org place code does NOT lift the value-train δ at nav scale (honest NEGATIVE) (2026-06-18, CYCLE 218)

**Follow-on to** `2026-06-18-merged-neural-place-code-SCOPE-GO.md` (#5 value `a` = the self-org place critic COMPOSES
on the merged bridge, host `vs_place_context` retired — GO + committed `6b9af426`/`41ba1be9`). This is #5 value `b`
(the δ-LIFT hope): does the self-org `place→striosome_value` `coincidence_detector` volley — which fires the MSN-D1
from the LEARNED code WITHOUT the position-blind up-state bootstrap that capped the CYCLE-211/214 value-train δ at
~1.3 — actually LIFT the δ? **Controller-owned run** (the array-disjoint argument: the conv slices have zero
out-edges into the critic, so the standalone self-org δ == the merged δ). **NO `sim/` edit.**

## Result (standalone `g11 --neural-place-selforg --stage-b-smoke --value-train-trials 40 --enable-critic-fs-inhibition --critic-fs-weight 16`, seed 42, GPU)

```
STEP-1 self-org (9s):  place fields cos(near,far)=0.672, sparsity=0.458   <- TOO DENSE / poorly separated
STEP-2 value-train (95s, 40 trials x 4 goals):  w_near 0.200->138.185  w_far 0.203->137.350  (near/far 1.01)
STAGE-B VERDICT:
  [LEARNS-V]            w_near/w_far = 1.01  (>=1.5x => FALSE)        <- V NOT position-specific
  [CRITIC FIRE+GRADE]   critic@near 364.17Hz  critic@far 378.06Hz  (near/far 0.96 => GRADE FALSE; fires MORE at far)
  [GABA_B gap d=r-V]    predicted(NEAR) 500Hz == unpredicted(FAR) 500Hz  -> gap = 1.00  (FLAT, => FALSE)
  [LESION control]      zero 426 GABA_B synapses -> pred 0Hz / unpred 0Hz -> 0.00 (collapses => TRUE)
```

## Verdict — NEGATIVE (value b); the host Gaussian stays the better-δ scaffold

The self-org place value-train δ is **flat (1.00)**, WORSE than the CYCLE-212 `vs_place_context` value-train
(δ ~1.3). The δ-lift hope is NOT realized at nav scale. The CYCLE-212 value-train δ caveat is **not closed** by
the self-org path.

**Root cause (the documented "place-code self-org at nav scale" design risk):** the self-organized place fields at
the nav grid scale are **too dense (sparsity 0.458 = 46% of place cells active per location)** and poorly separated
(cos(near,far) 0.67). So the value-training — which potentiates whatever fires at each of the 4 goal locations —
potentiates the OVERLAPPING cells **uniformly** (w_near ≈ w_far, ratio 1.01) → a uniform V → the critic fires ~equally
near and far (364 vs 378 Hz) → the GABA_B subtraction is flat (δ=1.00). The host Gaussian `vs_place_context` is
position-specific BY CONSTRUCTION (a tuned Gaussian), so its value-train graded (δ ~1.3); the self-org code did not
self-organize into sparse, separable fields at this scale.

**Anti-cheat (the flat δ is a value-GRADING failure, not a wiring failure):** the LESION control collapses the δ
(zero the `striosome_value→snc` GABA_B → pred/unpred both → 0), so the GABA_B route IS wired + load-bearing; the flat
δ is specifically because V is uniform (the place code can't support position-graded value), not because the
subtraction is broken.

## Honest confound + the cheap disambiguation follow-on

The run used the default `stdp_w_max=150` (the value weight saturated at ~138), NOT the CYCLE-212 value-train's
`value_train_stdp_w_max=40` (the soft-bound that keeps the MSN unsaturated). So saturation is a SECONDARY factor.
HOWEVER: `near/far = 1.01` is a field-OVERLAP signal that the cap does not touch (the cap bounds the magnitude, not
the near-vs-far differentiation) — a 40-capped re-run would still see uniform potentiation of the overlapping cells.
So the negative is robust to the cap. Cheap disambiguation if the δ-lift is later prioritized: re-run with
`--value-train-stdp-w-max 40`; the deeper lever is a SPARSER self-org target (lower place sparsity → separable fields),
which is the real boundary (a self-org-quality / nav-scale problem, the design risk the runner's own log flags).

## #5 net result (both halves)

- **value (a) = GO (committed):** the position code is now self-organized spiking `place` cells on the merged "one
  brain" — the host `vs_place_context` Gaussian scaffold is RETIRED, composes, moat-safe, no `sim/` edit. A real
  TRUE-ONE-BRAIN breadth win (a host shortcut removed).
- **value (b) = NEGATIVE:** but the self-org place value-train δ (1.0) UNDERPERFORMS the host Gaussian (1.3) at nav
  scale — the self-org fields are too dense/overlapping to support position-graded value. The self-org place is a
  **validated-but-costly brain-based replacement** (mirroring #4's fully-spiking read-out: genuinely brain-based, but
  underperforms the host shortcut on quality). Per the BRAIN-BASED-ONLY standard, this neural-underperforms-host
  result IS the scientific deliverable (it maps the substrate's cost). The host Gaussian remains the documented
  better-δ scaffold; closing the δ caveat needs sparser self-org fields (depth-tuning) — NOT closed by the
  coincidence-volley alone.

## Reproduce
```bash
SIM_BACKEND=cupy python -m research.runners.g11_bg_runner --moving-goal --goal-schedule multi --deterministic \
  --enable-neural-critic --spiking-reward-us --enable-critic-homeostasis \
  --enable-critic-fs-inhibition --critic-fs-weight 16 --neural-place-selforg --stage-b-smoke \
  --value-train-trials 40 --seed 42 --out research/findings/raw/_n5_selforg_stageb_seed42.json
# (g11 __main__ is a LAUNCHER wrapper that spawns the real run as a detached child -> webapp/runtime/run_<id>.log;
#  read the STAGE-B VERDICT from that log, not the launcher's stdout.)
```
