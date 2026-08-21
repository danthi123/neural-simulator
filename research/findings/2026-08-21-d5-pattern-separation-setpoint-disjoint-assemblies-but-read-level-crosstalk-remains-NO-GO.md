---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-pattern-separation-setpoint
lane: memory
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_pattern_separation_setpoint_derisk.py — an intrinsic-excitability winner-fatigue
  sparsity set-point on the emergent-assembly formation, measured for (Layer A) assembly-membership OVERLAP off vs on
  across 6 seeds, and (Layer B, seed 42) the NEIGHBOR surfaced-recall SHIFT when consolidating a co-formed memory,
  with tools.verdict.Verdict.
runner: research/runners/_d5_pattern_separation_setpoint_derisk.py
artifacts:
  - research/findings/raw/_d5_separation/numpy_proxy.json
---
# D5 pattern-separation set-point makes assemblies DISJOINT but does NOT close the read-level crosstalk — separation is necessary, NOT sufficient (NO-GO, board #73/#71)

Artifact: research/findings/raw/_d5_separation/numpy_proxy.json (go: False, status UNDEFINED). Every number below is
read from that artifact.

<!--derived-->
**One line.** Board #73's named fix — an intrinsic-excitability **winner-fatigue sparsity set-point** so no engram
goes dense and two similar memories recruit DISJOINT assemblies — WORKS at the structural level (assemblies go
disjoint, non-dense, A's own strength still rises) but it does **NOT** eliminate the D5 learn-through-use crosstalk
that blocks the default-on flip: consolidating memory A still shifts neighbor B's SURFACED read, and *more* than
before (seed 42 |ΔB|_on **7.478** vs |ΔB|_off **0.135**), even though the assemblies are disjoint and **B's own
within-weights are byte-identical**. So the crosstalk lives at the **READ level**, not the weight-sharing level —
write-side separation is necessary but not sufficient. Banks the write-separation method; points the next lever at
read isolation.

## What WORKED (the structural pattern separation — the set-point does its job)
<!--derived-->
Per Layer A over 6 seeds (sep_bias=500), OFF → ON max shared cells between assemblies:
- s42 4→0 · s43 3→0 · s44 2→1 · s100 5→1 · s101 6→0 · s102 3→1 — **fully disjoint on 3/6 (42/43/101), ≤1 shared on
  the other 3**; every seed **non-dense** (on_sizes healthy, e.g. s42 [27,18,21] of the granule pool — no collapse to
  the ~150-200/200 dense-engram failure the prior arc named).
- **byte-identical OFF** (the default path == the unmodified emergent-assembly formation) ✓.
- **B's within-weights are byte-identical after consolidating A** (disjoint ⇒ A's plateau-gated BTSP cannot touch B's
  read weights) ✓, and **A's own strength still rises** (A_rise ≈ 0.64 — the consolidation faculty is preserved).

## What FAILED (the decisive one — the read-level crosstalk is NOT closed, and is larger)
<!--derived-->
On seed 42, with the set-point ON and the assemblies disjoint (0 shared cells, B weights byte-identical), B's
surfaced dendritic-depth read still **collapses when A is consolidated**: **B_depth 17.57 → 10.10, |ΔB| = 7.478** —
vs the OFF weight-sharing crosstalk it was meant to remove, |ΔB|_off = 0.135 (shared_conn 5). So the shift a user
would SEE on the untouched neighbor got ~55× WORSE, not zero. Because B's weights are provably unchanged, this
crosstalk is **not weight-mediated** — the competitive winner-fatigue dynamics that orthogonalize the assemblies also
**globally suppress the neighbor's read** while A wins. Two require-checks fail: "disjoint on ALL 6 seeds" (3/6 keep 1
shared cell) and "s42 ON neighbor crosstalk == 0" (7.478).

## The mechanistic lesson + the NO-DEFER next lever
<!--derived-->
Pattern separation orthogonalizes the WEIGHTS (necessary — it removes the shared-granule cross-write), but the D5
**surfaced read** responds to a shared pathway (global competition / normalization / afferent drive) that disjoint
weights do not isolate. So the D5 default-on blocker is a **READ-ISOLATION** problem, not only a write-separation
problem. NEXT (the read-level lever, no-defer): a normalization/competition-isolated surfaced read — read B's plateau
depth WITHOUT the global winner-fatigue suppression (per-assembly local normalization, or read A and B in separate
non-competing windows), or a separation mechanism whose competition does not bleed into the neighbor's read. The
**cupy confirm was NOT queued** — a numpy-proxy NO-GO does not warrant a GPU confirm; the read-level fix is what to
de-risk next, then confirm at the production encode (te=40) on cupy.

## Honest scope
<!--derived-->
numpy proxy (store_te=18, consol_te=34, sep_bias=500, 3 patterns), Layer B crosstalk measured on seed 42 only (the
OFF-reproduced build). The structural-separation result (Layer A) is 6-seed. This is the write-side method banked; the
read-isolation lever is the live residual for the D5 learn-through-use flip (board #71) and for #73 pattern
separation. NO `sim/` edit (a formation-time excitability set-point + a read probe). Parent-finalized from the
artifact after the build agent completed the run.
