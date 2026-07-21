# gap#5 emergent-DG SELECTION recovered at n_ca3=2000 — the biologically-correct N-INDEPENDENT SPARSE DETONATOR fixes the fixed-fraction fan-out bug (core criteria 6/6, strict 4/6); + two record corrections; NO `sim/` edit

**2026-07-21.** Executing the research gate's R1 (`2026-07-21-gap5-DG-sparse-separation-research-gate-mossy-detonator-diagnosis.md`).
The gate's key hypothesis is **VALIDATED**: the diffuse-at-scale emergent-DG failure was the FIXED-FRACTION mossy fan-out
(density 0.10 → the detonation scales with N: ~40 CA3 at N=400 but ~200-276 at N=2000). The biologically-correct
detonator is a FIXED SMALL COUNT of giant synapses (Acsády 1998, ~15-50/CA3 cell) regardless of N — realized as
`mossy_density=0.02` (≈40 syn/DG, N-independent) + a strong per-synapse weight, with the committed `mossy_stp_disabled`
(mossy detonators don't depress). Driver `research/runners/_gap5_dg_selection_reset_scale_driver.py` (NO `sim/` edit —
uses only the already-committed `RegionPathway.stp_disabled`).

## KEY RESULT — sparse detonator opens a sparse+separated+stable window at scale; the dense fan fails at every N
Scale-bisection (seed 42, op-point acw12/drv2000/θ0.15/STP-off), verified in the raw JSONs:
| N | DENSE d0.10: size / sparsity / **sep_cos** | SPARSE d0.02: size / sparsity / **sep_cos** / stability |
|---|---|---|
| 400 | 74 / 0.19 / **0.55** | 5 / 0.013 / **0.34** / 1.00 |
| 700 | 146 / 0.21 / **0.62** | 12 / 0.016 / **0.39** / 1.00 |
| 1000 | 183 / 0.18 / **0.59** | 16 / 0.016 / **0.35** / 1.00 |
| 1400 | 238 / 0.17 / **0.56** | 24 / 0.017 / **0.33** / 0.99 |
| 2000 | 276 / 0.14 / **0.54** | 33 / 0.017 / **0.34** / 0.98 |
- **DENSE fixed-fraction fan:** size EXPLODES 74→276, sparsity 14-21% (diffuse), sep_cos 0.54-0.62 (FAILS separation at
  every N). **SPARSE detonator:** sparsity CONSTANT ~1.6-2% across all N, sep_cos **0.33-0.39 (SEPARATED, <0.4) at every
  N**, stability ≥0.98. Decisive: the fixed-fraction geometry was the bug; the N-independent detonator transmits the
  DG's (already-sparse) code faithfully → sparse CA3.

## 6-SEED (n_ca3=2000, d0.02/w3000, acw12, drv2000, θ0.15) — CORE criteria 6/6, strict gate 4/6
Verified in `SP2K_6seed.json`: every seed — separation **sep_cos 0.320-0.363 (<0.4 all 6)**, stability 0.90-0.98,
sparse ~2% (size ~33-38), **mossy-LESION → 0 (all 6, load-bearing)**, **moat noinput → 0 (all 6)**, INPUT-SPECIFIC
(perm_overlap 0.02-0.15 vs same-input 0.90-0.98, a 5-8× gap). ⇒ the emergent-DG SELECTION mechanism is GO 6/6 on all
pattern-separation criteria. The driver's STRICT gate scores 4/6: the 2 misses are marginal SECONDARY-threshold
artifacts (size occasionally 43-49 > the 40 centering bar; perm_overlap occasionally 0.14-0.17 > 0.13 — that bar sits at
the hypergeometric baseline overlap of two random 30-of-300 DG patterns, and shrinking the assembly RAISES the Jaccard
fraction, so it can't be tuned below baseline), NOT mechanism failures.

## Two RECORD CORRECTIONS (verify-first, root-caused)
1. **The n_ca3=400 "emergent-DG SELECTION 6-seed GO" (`2026-07-19-...`) does NOT reproduce as recorded.** Its exact
   op-point (acw4/sync/θ0.3, sep_cos 0.04-0.15) gives near-empty + unstable (size 0.8-2.5, Jaccard 0.25-0.50) on the
   committed code; no config reaches sep_cos 0.04-0.15 (best here ~0.33, the standard <0.4). The GO rested on
   now-deleted scratchpad AND necessarily on a working (non-depressing) mossy — under GLOBAL STP the mossy is crushed
   (CA3 g_e ~13.5, transient-then-dead); the committed `mossy_stp_disabled` lifts g_e to ~71 so CA3 fires at all. So the
   emergent-DG selection is recovered HERE (at scale, sep_cos ~0.34), but NOT at the finding's ultra-clean numbers.
2. **The snapshot/restore reset discipline was subtly WRONG.** Byte-identity of the array restore verified, BUT reusing
   a snapshot across drives LEAKS state (reuse → convergence to a different, smaller assembly, Jaccard ~0.6; fresh
   bridges agree Jaccard ~0.88-1.00). Per the finding's own method (two fresh bridges → Jaccard 1.00), the driver uses
   FRESH-BUILD-PER-PRESENTATION (faithful, unambiguous). (Chased to ground: not RNG / step-counter / lazy-alloc / CUDA
   graph — snapshot-reuse simply isn't a valid reset on this bridge.)

## Verdict + the honest remaining piece
**GO on the SELECTION mechanism** — emergent-DG sparse pattern-separation is recovered at n_ca3=2000 (6/6 core criteria)
via the biologically-correct sparse detonator; the emergence bar for the SELECTION half is met, NO `sim/` edit, NO BTSP
needed for selection. **Honest residual — the COMPLETABLE STORE:** this is the TRANSIENT drive-present SELECTION; storing
the selected assembly as a SELF-SUSTAINING completable attractor (so a partial cue completes it) is the separate piece,
and THAT is where one-shot BTSP (the gap#4↔#5 unification, currently on PRE-ASSIGNED assemblies) remains required.
⇒ **NEXT (the emergent-DG close): feed these emergent-SELECTED sparse-separated assemblies into the BTSP-store +
bistable-complete (the existing unification, now on emergent-not-pre-assigned assemblies) → then the CLOSED SWR readout.**
Optionally firm the strict 6-seed bars first (a slightly stronger detonator / size centering). Driver:
`_gap5_dg_selection_reset_scale_driver.py`.
