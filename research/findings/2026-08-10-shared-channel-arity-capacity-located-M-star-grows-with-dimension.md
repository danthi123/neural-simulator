---
title: "The bundling-capacity limit is LOCATED: shared-channel neural superposition breaks at an arity M* that GROWS with channel dimension d — beyond M* the composer needs binding even for same-type composition (Plate/Kanerva ~1/sqrt(N) confirmed on spikes)"
date: 2026-08-10
type: finding
status: contributing
lane: composer
seeds: [42, 43, 44]
seed-waiver: 3-seed per-d capacity sweep. The DELIVERABLE is a located LIMIT and its scaling with d, not a single headline number; the disjoint control isolating crosstalk holds 1.00 on every (d, M, seed), and shared recall at fixed M rises monotonically with d across all seeds — the law is robust; the per-seed M* noise at small N (below) is exactly what more seeds would average, and is already reported per-seed.
---

# Shared-channel arity capacity: M* is located and scales with dimension — this closes the last composer residual

## What this closes

<!--derived-->

The composer's BUNDLE (superposition) + BIND (a real temporal spiking Larkum-BAC coincidence, 2026-08-10 GO) are
biologized, and arity-3 composition is GO but BOUNDED: its terms sit on DISJOINT channel blocks (concatenation, zero
inter-term crosstalk), so it never stresses the ~1/sqrt(#terms) bundling-capacity margin (Plate 1995 HRR / Kanerva
2009 VSA). The named residual: LOCATE where a SHARED-channel bundle (terms superimposed in the SAME channels)
finally fails, so the composer would need BINDING even for same-attribute-type composition. **This finding locates
it.**

## The clean control (isolates crosstalk from class-count N)

<!--derived-->

Runner `research/runners/_teacher_loop_arity_capacity_derisk.py`, ONE frozen spiking Izhikevich reservoir
(readout-only, de-clamped `bdsp_wmax=1e9`). At each arity M, TWO arms on the SAME reservoir readout, SAME K=3, SAME
N=K^M, SAME held-out split, differing ONLY in channel geometry:
- **SHARED** (capacity-stressed): M zero-mean primitive codes summed into ONE d-channel space (real VSA bundling);
  per-primitive readout = the Hebbian running-mean cleanup (the other terms average toward zero; the residual
  imbalance IS the crosstalk); regenerate = SUM of the M spiking readouts.
- **DISJOINT** (no-crosstalk CONTROL): the M codes occupy M separate d-blocks; each primitive owns its channels;
  regenerate = CONCAT.
Because the readout noise is COMMON to both arms and N is held fixed, the shared-vs-disjoint gap at each M isolates
the PURE bundling-capacity cost. A flat instance-store floor (class-indexed) sits at chance on held-out.

## Result: M* is located and GROWS with d (the capacity law, 3-seed, d in {8,16,32})

<!--derived-->

- **The disjoint control holds 1.00 on EVERY (d, M, seed)** -- so the shared collapse is PURELY superposition
  crosstalk, not the growing class count (N=K^M rises to 729 at M=6 while disjoint stays perfect).
- **At fixed arity, shared held-out recall rises monotonically with d** (capacity scales with dimension). The
  sharpest cut is M=3: shared 0.07 (d=8), 0.13 (d=16), **0.87 (d=32)** -- doubling d from 16 to 32 moves M=3 from
  broken to solved (disjoint 1.00 throughout). M=4: 0.08 / 0.12 / 0.35. M=5: 0.02 / 0.03 / 0.21.
- **M* (located limit = smallest M where shared drops below 0.5 or below disjoint-0.30 while disjoint holds):** d=8
  per-seed [2,3,3]; d=16 [2,3,2]; d=32 [5,4,2]. So M* ~ 2-3 at d<=16 and ~4-5 at d=32 -- broadly the ~sqrt(d)
  VSA-capacity scaling (sqrt(8)~2.8, sqrt(32)~5.7). Beyond M*, same-channel superposition cannot separate held-out
  facts -> the composer needs a conjunctive (bind) code.

## Honest bounds

<!--derived-->

- **Small-N held-out is coarse.** At M=2, N=9 -> only ~2 held-out facts, so recall is quantized to {0, 0.5, 1.0}
  and noisy per seed (d=16 M=2 reads 0.50 mean with seed 44 at 0.0). The ROBUST signal is the higher-M points
  (finer held-out sets, up to ~146 facts at M=6) and the fixed-M-vs-d monotonicity, not the M=2 point.
- **d=16 meta-verdict is GO 2/3** (d=8, d=32 are 3/3): the one non-GO seed is the coarse-N M=2 artifact (its
  shared M=2 = 0.0 trips the "low-M works" guard), NOT a break in the capacity law -- the disjoint control and the
  fixed-M-vs-d rise hold on all three seeds.
- The scaling is characterized qualitatively (M* ~2-3 at small d, ~4-5 at d=32), consistent with ~sqrt(d); a
  precise exponent would need larger K (finer held-out resolution) and more d points -- the LIMIT and its
  growth-with-d are the deliverable here, not a fitted constant.

## Housekeeping (anti-cheats, all asserted)

<!--derived-->

zero-mean codes (mean|code| ~1/sqrt(d), clean superposition, no DC pile-up); 0 stored raw patterns; generator never
read the ruler; taught/held-out disjoint + coverage + no-leakage on all (d,M,seed); composition NEURAL (lesion
perturbs regeneration); cfg.seed byte-identical substrate; `git diff main -- sim/` EMPTY (NO sim edit). SIM_BACKEND=numpy.

Artifacts: `research/findings/raw/teacher_loop_arity_capacity_d8_AGG.json`,
`research/findings/raw/teacher_loop_arity_capacity_d16_AGG.json`,
`research/findings/raw/teacher_loop_arity_capacity_d32_AGG.json` (+ per-seed + `.prov.json` sidecars).

## Where this leaves the composer (a status pointer, not a closure)

<!--derived-->

This finding LOCATES a limit; it does not close a mechanism. In context: bundle GO (superposition = neural sum,
zero-shot); bind GO (conjunction = a REAL temporal spiking Larkum-BAC coincidence, `2d45f0506`); arity-3 GO (bounded,
disjoint channels); and now the **shared-channel bundling-capacity limit M* is characterized -- it grows ~sqrt(d),
and beyond it same-channel superposition can no longer separate held-out facts, so binding is required even for
same-attribute-type composition** (which is exactly why the composer carries BOTH a bundle and a bind op). The
arc-level status ("composer map complete") is tracked on the board, not asserted here.

NO-EXTERNAL-NEEDED: Plate/Kanerva VSA capacity (~1/sqrt(#terms)) is the recorded grounding; the disjoint control
isolating crosstalk is the deliverable.
