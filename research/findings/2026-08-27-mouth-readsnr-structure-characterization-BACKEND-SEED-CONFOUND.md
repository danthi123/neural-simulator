---
type: finding
status: boundary
date: 2026-08-27
verdict: BOUNDARY -- the assigned 6-seed RECODABLE-vs-SUBSTRATE-WALL sweep (CPU-mandated) surfaces a prior-order discovery instead -- SIM_BACKEND is a DIFFERENT RNG (numpy vs cupy), not a speed knob, on cfg.seed; the realized network differs at one nominal seed (thr_hash mismatches 6/6), and the NO-GO's collapse (head_w corr~0.00 vs random~0.95) reproduces 6/6 on cupy but 0/6 on numpy (head_w corr~0.96) -- this CPU run characterized a DIFFERENT network than the wall was found in, so recodability is UNRESOLVED, not RECODABLE. Still informative -- sparsity/IPR shows NO effect; EIGEN-ALIGNMENT is the one real off-wall signal (bottom-1-PC reads mean 0.37 vs top-1-PC 0.94) -- the next lever for a GPU re-run.
mechanism: mouth read-SNR decoder direction -- structure-characterization sweep (interpolation / sparsity / eigen-alignment / anchors) for the 2026-08-27 softmax-confidence NO-GO's structure-selective collapse
lane: E-language-mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_structure_characterization/char_6seed.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s42.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s43.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s44.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s100.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s101.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s102.json
runner: research/runners/_wkv_mouth_readout_structure_characterization_derisk.py
---

# Mouth read-SNR structure-characterization: the CPU-mandated re-run does not see the wall it was sent to characterize -- SIM_BACKEND silently changes which network `cfg.seed=N` builds

Artifact: `research/findings/raw/_wkv_structure_characterization/char_6seed.json` (6-seed, `SIM_BACKEND=numpy`,
B=8, sub-read-window=64 -- the memory-safe scale this task specified).

## The question this arc was sent to answer (do not re-derive)

<!--derived-->
The 2026-08-27 softmax-confidence NO-GO (`2026-08-27-mouth-readsnr-softmax-confidence-weightnorm-NOGO.md`)
established, on `SIM_BACKEND=cupy`: a random/incoherent weight direction reads with corr ~0.95 against the
ideal linear map; the trained decoder's real target direction (`head_w`) reads at corr ~0.00, at every scale
10%-100% of its magnitude -- a structure-selective read-fidelity collapse. This arc's mission: sweep a FAMILY
of structured target directions (interpolation random<->head_w, sparsity, eigen-alignment to the substrate's
own activity subspace) to determine whether corr~0 is a property of `head_w` specifically (RECODABLE -- steer
the learned decoder toward a substrate-readable code) or of ALL structured/low-entropy directions
(SUBSTRATE-WALL -- a deeper read-mechanism problem). Instructed to run CPU-only (`SIM_BACKEND=numpy`, B=8,
read-window=64) to avoid repeating a same-day shared-machine RSS OOM.

## Method: 4 probe families, reuse-by-import only, no `sim/` edit

`research/runners/_wkv_mouth_readout_structure_characterization_derisk.py` reuses `_measure_gain` (the exact
corr instrument) unmodified from the NO-GO's own runner, `BatchedSubstrateReadout`/`_thr_hash` from the
eprop_batched_substrate runner, `_positions`/`WKVReadout`/`_load_eval` from the eprop_learn / fewspike_read
runners. Every probe is rescaled to a FIXED common norm (`||head_w||`, since the NO-GO showed corr is
magnitude-independent over 10-100% of that norm) so only DIRECTION/structure varies:

1. **Interpolation** random<->head_w, alpha in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
2. **Sparsity**: a RANDOM base matrix, per-row top-k-by-|value| kept, k in {1,2,4,8,16,32,64,128=dense} --
   isolates whether low-entropy/sparse structure PER SE (independent of alignment to head_w) degrades the read.
3. **Eigen-alignment** (Schuessler et al. 2023, eLife 12:e93060, "Aligned and oblique dynamics in recurrent
   neural networks" -- a readout direction inside the top-PC subspace of population activity reads as
   ALIGNED/high-correlation, outside it as OBLIQUE/low-correlation): `head_w` projected onto the top-k vs
   bottom-k principal components of the substrate's own input-activation covariance, rank-swept
   k in {1,2,4,8,16,32,64,128=full}.
4. **Anchors**: 3 random controls + `head_w` itself, reproducing the NO-GO's ~0.95 / ~0.00 pair.

6 seeds (42,43,44,100,101,102), `SIM_BACKEND=numpy`, B=8, sub-read-window=64 -- the exact memory-safe
configuration this task specified, matching `_wkv_mouth_readout_init_scale_sweep_derisk.py`'s established-safe
numpy defaults. Build-twice `thr_hash` seed-trap on every seed (`tests/test_determinism.py`'s own discipline).

## THE DECISIVE DISCOVERY -- `cfg.seed` does not survive a backend switch: numpy and cupy build DIFFERENT networks at the SAME nominal seed

<!--derived-->
The smoke test (seed 42) immediately contradicted the NO-GO: `anchor_headw` (== `head_w` itself, unrescaled)
read at corr **0.9528** -- matching random (0.9539), not the ~0.00 the NO-GO reported. Re-running the NO-GO's
OWN diagnosis runner (unmodified) with `SIM_BACKEND=numpy` at the identical seed/B/read-window reproduced this
exactly (`head_w` corr 0.9528, random corr 0.9521) -- ruling out a bug in the new runner. The cause: `cfg.seed`
seeds `cp.random` (`sim/backend.get_backend` aliases `cp` to the real `cupy` module under `SIM_BACKEND=cupy`
and to plain `numpy` under `SIM_BACKEND=numpy`) -- numpy's MT19937/PCG64 and cupy's cuRAND are unrelated RNG
algorithms, so "seed 42" selects a DIFFERENT draw sequence on each backend, both internally self-consistent
(build-twice-same-backend is deterministic -- `tests/test_determinism.py::TestSubstrateActuallySeeded` still
holds) but NOT interchangeable across backends. Confirmed on all 6 seeds, side by side against the NO-GO's own
cupy artifacts (`research/findings/raw/_wkv_softmax_confidence/shorttrain_s*.json`):

| seed | cupy thr_hash | numpy thr_hash | cupy head_w corr | numpy head_w corr | cupy random corr | numpy random corr |
|---|---|---|---|---|---|---|
| 42 | 1d90c97348ccaf4a | a45d2385f84619f0 | 0.0292 | 0.9528 | 0.9527 | 0.9539 |
| 43 | be380e28ac42684d | ecf1aafd6e0abc68 | -0.0133 | 0.9645 | 0.9521 | 0.9569 |
| 44 | d44d5f9a7861d3ee | 71f05e4a50636edf | 0.0010 | 0.9524 | 0.9409 | 0.9407 |
| 100 | 1a18192cd9174748 | 0631f01e792680b2 | -0.0168 | 0.9593 | 0.9374 | 0.9381 |
| 101 | e39b57023a1e076c | 7cabcb320773ceb3 | 0.0100 | 0.9716 | 0.9626 | 0.9639 |
| 102 | a93fa75263bf7be9 | 0b69d841bb382782 | -0.0138 | 0.9439 | 0.9333 | 0.9313 |

Every `thr_hash` differs (6/6); the structure-selective collapse is present in 6/6 cupy realizations and
ABSENT in 6/6 numpy realizations at the same nominal seed values. `random`-direction corr is unaffected by
backend either way (~0.93-0.96 both), so the split is specific to the structured direction, not a generic
GPU-vs-CPU precision gap: `cp_conductance_g_e`/`cp_conductance_g_i` are allocated `dtype=cp.float32`
identically on both backends (`sim/bridge.py:2068-2069`), ruling out a naive float32-vs-float64 explanation.
The remaining candidate is that cupy's cuRAND and numpy's MT19937/PCG64 heterogeneity/OU-noise draws are
DISTRIBUTIONALLY different in some way that makes cupy-realized networks systematically more prone to a
correlated-drive/shared-postsynaptic-pool saturation for the specific `head_w` direction -- root-causing which
specific draw and why is out of this CPU-only task's scope and reach.

**Consequence for the assigned question**: this arc's 6-seed CPU sweep characterizes a set of network
realizations in which the wall being investigated does not exist. It cannot say whether `head_w`-like
structured directions are recodable INSIDE the regime the NO-GO actually measured, because that regime is
cupy-specific and this task was CPU-only by explicit instruction (today's shared-machine RSS OOM). The
RECODABLE-vs-SUBSTRATE-WALL question is UNRESOLVED, not answered, by this run.

## What the CPU sweep DOES show (6-seed means, `char_6seed.json`)

<!--derived-->
In the numpy-realized networks (where nothing collapses), the four families still separate cleanly by
structural axis, which is useful context even off-wall:

- **Interpolation** random(alpha=0)->head_w(alpha=1): flat, 0.947 -> 0.957, NO collapse point anywhere on the
  path -- consistent with the anchors (random 0.9475 mean, head_w 0.9574 mean) already showing no gap to
  interpolate across in this regime.
- **Sparsity** (random base, k=1..128 nonzeros/row): flat, 0.958 (k=1) -> 0.947 (k=128, dense) -- essentially
  NO sparsity effect. Contextual: `head_w`'s own mean inverse-participation-ratio is **45.2** across seeds,
  statistically indistinguishable from a DENSE random gaussian matrix's IPR (**44.1** mean, at sparsity
  k=128) -- `head_w` is not "sparse/low-entropy" in the IPR sense at all, so sparsity is not the axis that
  distinguished it in the NO-GO's cupy measurement.
- **Eigen-alignment, top-k** PCs of the substrate's own activation covariance: flat and high at every k,
  0.942 (k=1, i.e. even RANK-1 confinement to the dominant PC) through 0.958 (k=128, full space).
- **Eigen-alignment, bottom-k** PCs (the weakest/residual activity directions): the ONE real, monotonic signal
  in the whole sweep -- 0.373 mean at k=1 (per-seed range 0.045-0.638), rising smoothly to 0.956 at k=128
  (=full head_w). Bottom-k=1 is the single lowest correlation anywhere in the CPU sweep, in all 6 seeds.

This is a small-magnitude, non-collapsing echo of the Schuessler aligned/oblique principle (alignment to a
population's dominant activity subspace preserves linear readability; confinement to the weak/residual
subspace degrades it) -- present even in the "healthy" numpy regime where the deep collapse itself does not
occur, and it is the only family here with a real gradient to point a target-recoding lever at.

## External source (live literature, this arc)

<!--derived-->
Schuessler, Mastrogiuseppe, Ostojic & Barak (2024/2023), eLife 12:e93060, "Aligned and oblique dynamics in
recurrent neural networks", https://elifesciences.org/articles/93060 -- "If we choose small output weights, we
expect aligned dynamics, because a large correlation is necessary to generate sufficiently large output. If
instead we choose large output weights, we expect oblique dynamics..." (their mechanism is readout-MAGNITUDE
driving co-adapted RNN dynamics; ours is a FIXED substrate's direction-dependent read-fidelity, tested here at
matched magnitude -- a complementary, not identical, use of their aligned/oblique framework, applied to
STRUCTURE rather than magnitude). Recorded via `tools/record_external_search.sh` for the `e-language-mouth-read-snr` lane.

## Verdict + redirect

**BOUNDARY.** The assigned characterization did not resolve RECODABLE-vs-SUBSTRATE-WALL, because the
CPU-mandated protocol builds networks in which the wall under investigation is absent (6/6 seeds). This is a
NEW, load-bearing methodological trap, sibling to CLAUDE.md's existing `cfg.seed` trap: that trap covers
same-backend, cross-build non-determinism (fixed by setting `cfg.seed`); this one covers cross-backend,
SAME-seed non-portability (numpy MT19937/PCG64 vs cupy cuRAND are different RNGs -- there is no config fix,
only the discipline that a same-seed numpy-vs-cupy comparison is a different experiment). Logged to
`research/FAILURE_LOG.md` as `NOT-GATEABLE` (expected RNG behavior, not a bug; the fix is documentation, not a
block). **Next lever**: re-run this exact runner (`_wkv_mouth_readout_structure_characterization_derisk.py`,
already built, additive) on `SIM_BACKEND=cupy` at the SAME memory-safe B=8/read-window=64 scale, one seed at a
time with RSS monitored, to test recodability in the regime the wall actually occupies -- the eigen-alignment
family is the one to prioritize, since it is the only family this CPU pass found a real gradient in, and its
mechanism (confinement to the substrate's own dominant activity-PC subspace) is a concrete, literature-grounded
target-recoding strategy rather than a blind sweep.

## Files

- `research/runners/_wkv_mouth_readout_structure_characterization_derisk.py` -- the 4-family sweep (additive,
  no `sim/` edit, reuse-by-import of `_measure_gain` / `BatchedSubstrateReadout` / `_thr_hash` / `_positions` /
  `WKVReadout` / `_load_eval`, all unmodified).
- `research/FAILURE_LOG.md` -- new row (2026-08-27) for the backend/seed-portability gap.
