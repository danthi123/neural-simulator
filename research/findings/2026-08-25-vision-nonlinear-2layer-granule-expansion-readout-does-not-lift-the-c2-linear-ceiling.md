---
type: finding
status: mixed
date: 2026-08-25
lane: perception
board: 75b
mechanism: granule-cell-expansion-2layer-spiking-readout
runner: research/runners/_vision_nonlin2layer_readout_derisk.py
supersedes_method: none (first attempt at board #75b; #75's 1-layer signed-linear readout remains the banked best)
artifacts:
  - research/findings/raw/lanes/perception/vision_nonlin2layer_readout_6seed.json
  - research/findings/raw/lanes/perception/vision_nonlin2layer_opsearch.json
  - research/findings/raw/lanes/perception/vlin_readout_6seed.json
---

# A cerebellum-grounded NONLINEAR 2-layer spiking readout (fixed random K-claw granule expansion + the #75 signed-linear readout) does NOT lift the vision object-"which" ~0.47 linear-separability ceiling (board #75b): the spiking mechanism ties the #75 1-layer baseline (dNONLIN -0.0087, 2/6 seeds positive) rather than beating it, misses the literal task GO bar (capability_go 0/6, anti-cheats-clean 1/6 — new POSITION leakage), and the corrected RATE ceiling shows the expansion itself does not raise the underlying separability (0.3889 vs the #75 1-layer ceiling 0.4653) — across 61 explored operating points spanning 3 architectural variants

**One-line verdict.** Board #75's signed linear-discriminant readout
([`2026-08-25-vision-signed-linear-discriminant-spiking-readout-solves-quantization-wall-relocates-to-feature-ceiling.md`](2026-08-25-vision-signed-linear-discriminant-spiking-readout-solves-quantization-wall-relocates-to-feature-ceiling.md))
named its own next rung: "a 2-layer spiking readout (a hidden layer of LIF conjunction/dendritic-coincidence
units before the class populations) can exceed 0.47." This runner builds that rung as a biologically-grounded
cerebellar GRANULE-CELL EXPANSION (Marr 1969; Albus 1971; Litwin-Kumar et al. 2017) inserted between the
UNCHANGED #75 C2 code and the UNCHANGED #75 readout (reused by import, so the readout itself cannot be a
confound). After an extensive operating-point search (61 configs across three architectural variants — see
below), the decisive 6-seed result is a clean **statistical tie, not a lift**: EXPAND (2-layer, spiking) held
0.4288 vs NOEXPAND (the #75 1-layer readout, reproduced on the identical data) held 0.4375 — dNONLIN **-0.0087**,
2/6 seeds positive, 4/6 negative, no consistent direction. The readout DOES learn (EXPAND beats its own random
control 0.1788, load-bearing 6/6) and DOES clear the pre-existing #72/#75 NO-GO floor (0.34) 6/6 — but both of
those properties are **inherited from the reused #75 readout**, not new to this rung. The task's literal GO bar
(capability_go: beats V1-direct by margin) is **0/6**, and **anti-cheats-clean is only 1/6** — a NEW residual
this rung introduces: the class-population spike code sometimes leaks POSITION (up to 0.48 vs the 0.40
pooled-out threshold on 4/6 seeds), a property #75's simpler 1-layer code did not have. **This is a negative on
the METHOD** (fixed random-positive-K-claw summation + threshold, reading the z-normed C2 code), not a
license to abandon the rung — a decomposition of WHY, and two different named next mechanisms, follow.

## Why this mechanism (built here; no `sim/` edit; C2 front end + the ENTIRE #75 readout REUSED BY IMPORT)

Take the IDENTICAL C2 spike code #75 reads (FIXED config-B spiking front end, FIXED random S2 bank, G=2
glimpses averaged) and project it through a FIXED random sparse "mossy-fiber → granule" connectivity: each of
`n_hidden` granule units samples `k_claw` C2 units with fixed positive excitatory weights (Litwin-Kumar,
Harris, Axel, Sompolinsky & Abbott 2017, *Neuron*: K~4 claws is near-optimal for pattern separation at
realistic mossy-fiber counts). A granule LIF population converts the summed claw drive to spikes; because a
granule cell only crosses threshold when enough of its K claws are jointly active, this is a genuine AND-like
coincidence nonlinearity (emergent from claw-summation + a hard threshold, not a host `np.maximum` standing in
for one). The SAME #75 readout (train-mean/std standardise → ridge signed linear discriminant → spike-port as
E + feedforward-inhibition class populations → spiking WTA), imported unchanged, is then trained on the
EXPANDED code instead of the raw C2 code — isolating the granule expansion as the only new variable.

## The operating-point search (the deliverable is the MAP, not one lucky cell — mirrors #75's own R-STDP sweep)

A smoke test at the naive first-guess gain (0.35) caught **every granule cell saturating** (frac_active=1.000,
mean spike count at the LIF's refractory ceiling) — a real bug, not a scientific result: the claw-summed drive
(~33–67 for k_claw 4–8) vastly exceeds what the LIF's `v_thresh=1.0` needs, so EVERY cell fired at ceiling
regardless of input, destroying all discriminability (held 0.31, worse than the RANDOM control's 0.25). Fixing
the operating point required three explorations on seeds {42,43,100} (leaving {44,101,102} out-of-sample for
the decisive run, exactly #75's own ridge-lambda convention):

| variant | axes swept | n configs | best dNONLIN found |
|---|---|---:|---:|
| global excitatory gain | n_hidden {128,256,384,768,1536} × k_claw {4,8,16} × gc_gain {0.015–0.180} | 48 | **-0.004** (n_hidden=1536, k_claw=8, gain=0.05) |
| per-unit homeostatic gain (intrinsic-excitability calibration, TRAIN-only) | target_mult {0.8,1.2,1.5,2.0,3.0} | 5 | -0.031 |
| signed E+I claws (excitatory mossy-fiber + Golgi-cell-inhibition-grounded) | frac_inhib {0.3,0.5} × gain {0.02–0.10} | 8 | -0.017 |

No configuration in **any** of the three architectures reliably beat NOEXPAND on the 3-seed exploration split.
**A second real instrument bug was caught mid-search**: the RATE-ceiling reference arm's threshold
(`--gc-thresh`, originally a fixed absolute value) was calibrated for the SPIKE code's count scale (0–48) and
was numerically negligible on the z-normed RATE code's much smaller magnitude, so the "ceiling" arm was not
thresholding at all in one version and wildly over-thresholding (collapsing toward chance) in a second, still-
imperfect fix pass (`vision_nonlin2layer_opsearch.json`'s `global_gain_sweep_2_wider_n_hidden` column, an
earlier v_thresh/gc_gain-tied threshold, superseded — see that file's note). The final fix derives the
threshold from each call's OWN claw-drive magnitude (self-normalising, analogous to the S2 z-norm's per-image
statistics — no held-out leak) and gives a physically sane number (0.3889, between the linear-only control
0.4063 and comfortably above chance) on the decisive run. **Both bugs are
recorded because catching them is the "verify the instrument" discipline this project runs on — the pre-fix
numbers would have supported the WRONG conclusion (mechanism totally dead / ceiling catastrophically
destroyed) for an instrument reason, not a mechanism reason.**

The best point found (n_hidden=1536, k_claw=8, gc_gain=0.05, ~16x expansion) became the runner's default and
the decisive-run operating point.

## Result — decisive 6 seeds (42/43/44/100/101/102, held-out positions, count code, chance 0.25)

| quantity | mean | per-seed |
|---|---:|---|
| NOEXPAND (the banked #75 1-layer readout, reproduced here) | **0.4375** | 0.49/0.38/0.46/0.47/0.43/0.41 (byte-matches #75's own 6-seed report) |
| **EXPAND (the mechanism: 2-layer, spike-ported)** | **0.4288** | 0.43/0.45/0.43/0.45/0.48/0.34 |
| EXPAND vs NOEXPAND (dNONLIN) | **-0.0087** | -0.06/+0.07/-0.03/-0.02/+0.05/-0.06 — **2/6 positive, no consistent sign** |
| EXPAND_random (untrained V, spike-ported) | 0.2500 | ≈ chance |
| dLEARN = EXPAND − EXPAND_random | **+0.1788** | learning load-bearing 6/6 (inherited from the reused #75 readout) |
| EXPAND vs the #72/#75 NO-GO floor (0.34) | beats 6/6 raw | (inherited; #75 already established 6/6 at this floor) |
| EXPAND vs V1-direct (capability_go, margin 0.10) | **0/6** | V1-direct 0.42/0.48/0.46/0.38/0.46/0.32 — EXPAND never clears it by margin |
| RATE granule ceiling (threshold-linear, corrected instrument) | 0.3889 | below both the linear-only control (0.4063) and the #75 1-layer RATE ceiling (0.4653) |
| LINEXPAND (same K-claw connectivity, NO threshold — isolates the nonlinearity) | 0.4063 | close to the RATE ceiling — thresholding adds ~nothing |
| object / position split-half decode off the class-pop code | 0.4132 / 0.3785 | **position leaks on 4/6 seeds** (up to 0.48 vs the 0.40 pooled-out bar) |
| anti_cheats_clean (scramble+shuffle+position, ALL three, per seed) | **1/6** | only seed 44 passes cleanly — a NEW residual vs #75's clean 6/6 |
| pixel-scramble / label-shuffle nulls | 0.243 / 0.2448 | ≈ chance — these two remain clean on every seed |
| task GO bar (board #75b, literal) | **0/6 — NOT MET** | capability_go 0/6 AND anti_cheats_clean 1/6 (bar requires ≥5/6 on each) |
| train accuracy (EXPAND) | 1.000 | fits perfectly; ridge + the held numbers above are the honest generalisation |

**Determinism verified**: two independent processes at seed 42 produce byte-identical output (diff only in the
`out` path string and `elapsed_seconds` timing field). This runner uses a standalone numpy LIF pipeline (not
the CoreSimConfig bridge), so `cfg.seed`/`actual_seed_used` do not apply — every RNG is explicitly derived
from the `seed` argument.

## The decomposition — three independent readings converge on the SAME conclusion

1. **The spiking mechanism ties, does not beat, the 1-layer baseline** (dNONLIN -0.0087, no consistent
   sign across seeds) — the headline the task asked to measure.
2. **The corrected RATE ceiling (threshold-linear, no spike-quantization cost) does not exceed, and sits
   BELOW, the #75 1-layer RATE ceiling** (0.389 vs 0.465) — so the residual is not spike-port cost hiding a
   real ceiling lift; the ceiling itself does not move.
3. **The LINEAR-only control (same random K-claw connectivity, no threshold) is close to the thresholded
   RATE arm** (0.406 vs 0.389) — adding the AND-like nonlinearity buys ~nothing over a plain linear
   recombination of the same random subsets.

All three readings agree: **a random K-of-96 positive-weighted subsample, whether summed linearly or
summed-then-thresholded, is a lossy re-basis of the C2 code, never recovering (let alone exceeding) what a
DIRECT signed-linear readout over the FULL 96-dimensional code already achieves.** This is the mathematically
expected outcome when the target's dependence on the base features is *not* strongly nonlinear: #75 itself
already attributed its ~0.47-vs-0.56 gap to a MAGNITUDE/common-mode issue from z-normalisation (its own
lever #1, board #75a), not a missing nonlinear interaction term — this rung's RATE-ceiling evidence (the
underlying linear-separability ceiling is stable at ~0.39–0.47 regardless of expansion architecture) is
**independent confirmation of that diagnosis**, not a new fact. A genuinely nonlinear residual (as gap#4's
own successful random-feature expansion found, held-out linear 0.284 → mlp 0.988 <!--derived--> (quoted from that
finding, not this runner's own artifact), an 0.70 nonlinear GAP) would
have shown the RATE ceiling MOVE with the expansion; here it does not.

**A second, previously-unmeasured residual this rung surfaces**: the class-population code's position
pooling is WEAKER than #75's (anti_cheats_clean 1/6 vs #75's clean 6/6), driven specifically by position
decode exceeding its bar on 4/6 seeds. The random K-claw recombination, unlike the DIRECT full-C2 readout,
appears to inadvertently preserve or introduce position-correlated structure into the class-population code —
plausible because random subsampling of the (per-template MAX-pooled, nominally position-invariant) C2 code
can reintroduce location sensitivity that a globally-optimal signed readout over all 96 dimensions does not.

## Honest residual + the next mechanism (no-defer)

The wall is NOT "the 2-layer spiking port is broken" (it learns cleanly, 6/6 load-bearing, and reproduces
every property #75's own readout has) — it is that **this specific expansion (random positive K-claw
summation of an already z-normed, positive-clipped code) does not add separable structure the direct
96-dimensional signed readout was not already extracting.** Two named, DIFFERENT next mechanisms (this is a
verdict on the method, not the capability):

1. **Combine with #75a first.** #75's own diagnosis (magnitude/common-mode, not nonlinearity, is the ~0.47-
   vs-0.56 gap's cause) is now independently corroborated by this rung's stable RATE ceiling. #75a (a lighter
   `s2_norm`/lower `s2_gain` joint sweep so the READOUT does the common-mode rejection instead of a pre-
   readout constant) is the more likely lever for THIS specific numeric gap, and should be tried before a
   second nonlinear-expansion attempt.
2. **If a nonlinear expansion is retried, it needs a genuinely multiplicative (not summed-then-thresholded)
   conjunction, and/or expand a MAGNITUDE-PRESERVING (less-normalized) C2 code rather than the z-normed one**
   — i.e. run the granule expansion on the SAME lighter-normalized C2 that #75a's op-point sweep would
   produce, so the expansion has real dynamic range to work with instead of an already-rectified, already-
   contrast-normalized input. A true multiplicative NMDA-spike-style two-input coincidence gate (product, not
   sum+threshold) is a categorically stronger nonlinearity than what was tested here and was not tried.
3. **The NEW position-leakage residual (anti_cheats_clean 1/6) needs its own fix independent of (1)/(2)**: a
   competitive/lateral-inhibition step across the granule population (a spiking soft-WTA, rather than
   independent per-cell LIF) before the class-population readout, so the expansion does not reintroduce
   location structure the C2 MAX-pool had already removed.

## Brain-based status

Somata genuinely SPIKE (LIF: leak, hard threshold, reset, absolute refractory, per-step membrane noise) at S1,
S2, the granule/hidden layer, AND the readout class populations. The granule AND-like nonlinearity is an
EMERGENT property of K-claw summation + a hard threshold (not a host function standing in for a neuron).
Common-mode rejection at the readout = feedforward inhibition (Dale-compliant E/I decomposition, unchanged
from #75). Readout weights are a supervised ridge closed-form (exact fixed point of an L2-decayed three-factor
delta rule; a host-computed teacher scaffold, same status as #75/R-STDP). FLAGGED innate developmental
scaffolds (same concessions as config B/C/#75): retinotopic weight-sharing + pooling windows; the fixed random
S2 bank; the fixed random granule (mossy-fiber) connectivity — real cerebellar granule-cell wiring is itself
largely genetically/developmentally specified, not activity-learned, so this is a DEFENDED concession, not a
new one. No live conversational vision consumer exists (#72/#75); scope is the spiking CAPABILITY.

## Reproduce

```bash
# 6-seed decisive (CPU/numpy, ~20s total at the chosen operating point):
SIM_BACKEND=numpy OMP_NUM_THREADS=4 .venv/bin/python -u -m research.runners._vision_nonlin2layer_readout_derisk \
  --seeds 42 43 44 100 101 102 --n-hidden 1536 --k-claw 8 --gc-gain 0.05 \
  --out research/findings/raw/lanes/perception/vision_nonlin2layer_readout_6seed.json
```

## Sources

- Marr, D. (1969). A theory of cerebellar cortex. *J. Physiol.* 202:437-470.
- Albus, J. S. (1971). A theory of cerebellar function. *Math. Biosci.* 10:25-61.
- Litwin-Kumar, A., Harris, K. D., Axel, R., Sompolinsky, H. & Abbott, L. F. (2017). Optimal degree of
  synaptic connectivity. *Neuron* 93:1153-1164.
- Cayco-Gajic, N. A. & Silver, R. A. (2019). Re-evaluating circuit mechanisms underlying pattern separation.
  *Neuron* 101:584-602.
- Brunel, N., Hakim, V., Isope, P., Nadal, J.-P. & Barbour, B. (2004). Optimal information storage and the
  distribution of synaptic weights: perceptron versus Purkinje cell. *Neuron* 43:745-757. (Already #75's
  readout grounding — the Purkinje-cell linear readout of a granule-expanded code.)
- Maass, W., Natschläger, T. & Markram, H. (2002). Real-time computing without stable states. *Neural
  Comput.* 14:2531-2560.
- Fremaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of three-factor learning rules. *Front.
  Neural Circuits* 9:85.
- Prior on this substrate: `2026-08-25-vision-signed-linear-discriminant-spiking-readout-solves-quantization-wall-relocates-to-feature-ceiling.md`
  (#75, this rung's baseline + the named next-mechanism this runner builds);
  `2026-08-26-vision-rstdp-sparse-readout-NOGO-across-the-full-2D-operating-point-sweep-mapped-boundary.md`
  (#75, the earlier mapped dead-end); `2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-numpy-GO-onbridge-next.md`
  (a DIFFERENT gap where a fixed nonlinear expansion DID lift a linear ceiling — contrasted here: that
  task had a large nonlinear gap, mlp 0.988 vs linear 0.284 <!--derived--> (quoted from that finding); this
  task's RATE ceiling does not move with expansion, consistent with a small/absent nonlinear gap).
