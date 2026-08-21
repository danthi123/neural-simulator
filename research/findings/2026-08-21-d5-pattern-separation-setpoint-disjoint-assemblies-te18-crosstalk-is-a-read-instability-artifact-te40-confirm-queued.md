---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-pattern-separation-setpoint
lane: memory
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_pattern_separation_setpoint_derisk.py — an intrinsic-excitability winner-fatigue
  sparsity set-point on the emergent-assembly formation, measured for (Layer A) assembly-membership OVERLAP off vs on
  across 6 seeds, and (Layer B) the NEIGHBOR surfaced-recall SHIFT when consolidating a co-formed memory. A follow-up
  repeated-read diagnostic (no consolidation between reads) tested whether the Layer-B shift is a genuine leak or a
  reduced-encode read-instability artifact.
runner: research/runners/_d5_pattern_separation_setpoint_derisk.py
artifacts:
  - research/findings/raw/_d5_separation/numpy_proxy.json
---
# D5 pattern-separation set-point makes assemblies DISJOINT; the te=18 proxy's neighbor "crosstalk" is a READ-INSTABILITY ARTIFACT, not a leak — the te=40 cupy confirm is the decider (board #73/#71)

Artifact: research/findings/raw/_d5_separation/numpy_proxy.json (go: False at te=18, status UNDEFINED — the te=18
instrument cannot decide the crosstalk question; see the correction below). Every number below is read from that
artifact or from the repeated-read diagnostic it names.
<!--derived-->

**One line.** Board #73's named fix — an intrinsic-excitability **winner-fatigue sparsity set-point** so no engram
goes dense and two similar memories recruit DISJOINT assemblies — WORKS at the structural level (assemblies go
disjoint on 3/6 seeds, non-dense on all, **B's within-weights byte-identical** after consolidating A, and A's own
strength still rises). The te=18 proxy showed a large neighbor read-shift (|ΔB|_on ≈ 7.478 vs |ΔB|_off ≈ 0.135) that
FIRST read as a residual read-level leak — **but a repeated-read diagnostic RETRACTS that conclusion:** at the reduced
te=18 encode the surfaced read is simply UNSTABLE (repeated B-reads with NO consolidation between them swing far more
than 7.478), so the "shift" is a measurement artifact of the weak encode, not an A→B leak. The **te=40
production-encode cupy confirm (queued)** is the decider: with a strongly-formed, stable read and byte-identical
B-weights, the disjoint set-point should give ≈0 surfaced shift.

## ⭐ CORRECTION — the te=18 read is an unstable instrument (retracts the "read-level crosstalk remains" reading)
<!--derived-->
A follow-up diagnostic re-read the SAME neighbor B (the shrunken 18-cell disjoint assembly) FOUR times with NO
consolidation between the reads, and A three times:
- B: depth_hold 17.575 → 9.445 → 3.472 → 9.519 (a ~14 mV swing with nothing changing) · A: 24.414 → 22.556 → 19.150.
So at te=18 the surfaced depth read is NOT a stable function of the stored weights — repeated identical reads vary by
MORE than the 7.478 "crosstalk" attributed to consolidating A. Because B's within-weights were independently verified
byte-identical after consolidating A (disjoint assemblies), the 7.478 cannot be a weight-mediated leak; it is
read-noise of the reduced encode. THE INSTRUMENT IS PART OF THE EMULATION: a refutation ("crosstalk remains") needs
the instrument verified exactly as much as a confirmation, and here it fails the check. So the te=18 Layer-B crosstalk
verdict is **UNDEFINED**, not NO-GO.

## What is ESTABLISHED (the structural pattern separation — 6-seed Layer A)
<!--derived-->
OFF → ON max shared cells between assemblies (sep_bias=500): s42 4→0 · s43 3→0 · s44 2→1 · s100 5→1 · s101 6→0 ·
s102 3→1 — **fully disjoint on 3/6 (42/43/101), ≤1 shared on the other 3**; every seed non-dense (e.g. s42 on_sizes
[27,18,21] — no collapse toward the ~150-200/200 dense-engram failure the prior arc named); **byte-identical OFF**; B's
within-weights byte-identical after consolidating A (disjoint ⇒ A's plateau-gated BTSP cannot touch B's read weights);
and A's own strength still rises (A_rise ≈ 0.64 — the consolidation faculty is preserved).

## The decider + the NO-DEFER next lever
<!--derived-->
The **te=40 production-encode cupy confirm is queued** on gpu_queue
(`_d5_pattern_separation_setpoint_derisk --store-te 40 --consol-te 40 --crosstalk-seeds 42 43 100`). Decision rule:
- if the te=40 neighbor shift is ≈0 with byte-identical B-weights ⇒ the disjoint set-point CLOSES the D5 crosstalk
  and the D5 default-on flip is unblocked (subject to the 3/6→6/6 disjointness caveat below);
- if a residual shift REMAINS at te=40 with byte-identical weights ⇒ THEN read-isolation (a normalization/competition-
  isolated surfaced read) is the warranted next lever — not before.
Caveat for the te=40 run: sep_bias=500 reaches full disjointness on only 3/6 seeds (44/100/102 keep 1 shared cell), so
a higher bias or a per-seed operating point may be needed to hit 6/6 disjoint.

## Honest scope
numpy proxy (store_te=18, consol_te=34, sep_bias=500, 3 patterns); the repeated-read diagnostic is a supplementary
read-stability probe (its raw reads are in the run's diag log, summarized above). The structural-separation result
(Layer A) is 6-seed; the crosstalk verdict is DEFERRED to the queued te=40 cupy confirm. NO `sim/` edit (a
formation-time excitability set-point + a read probe). Parent-finalized + corrected from the artifact + the diagnostic
after the build agent completed the runs. Supersedes the earlier same-day framing that called the te=18 shift a
read-level leak.

## te=40 CUPY CONFIRM ran — the crosstalk is INSTRUMENT-LIMITED at production encode too (D5 flip stays blocked)
<!--derived-->
The queued te=40 confirm (`research/findings/raw/_d5_separation/cupy_confirm_te40.json`, VERDICT UNDEFINED) does NOT
cleanly resolve the crosstalk — but not because the te=18 read was uniquely bad. At te=40 the set-point still
separates structurally (on_max_shared 6->1, 4->1, 2->0, 3->0, 3->1, 3->0 — ~3/6 fully disjoint at sep_bias=500), yet
the neighbor shift is SMALL, NOISY, and inconsistent in DIRECTION: s42 |dB| 5.57->2.71 (down), s100 1.84->0.57 (down),
but s43 0.14->**1.79 UP despite B's within-weights being BYTE-IDENTICAL** — a weight-untouched neighbor moving 1.79 mV
is READ NOISE, not an A->B leak. And "A's own strength rises" is FALSE on all three (A_rise -0.089/-0.236/-0.110):
te=40 SATURATES A (~30 mV, at the read ceiling), the exact limit the 2026-08-20 graded-read finding named — so the
consolidation-strength signal cannot be read there either. CONCLUSION: the crosstalk question is INSTRUMENT-LIMITED at
BOTH encodes (te=18 read-instability; te=40 saturation + mV-scale read noise) — the surfaced dendritic-depth read is
too noisy relative to the effect to verify elimination. The load-bearing next lever CONVERGES with the 2026-08-20
direction: a STABILIZED / GRADED surfaced read (lower-variance readout that both resolves the crosstalk AND lifts the
conversation-visibility). A bias sweep for 6/6 disjoint is a secondary knob. The D5 default-on flip stays BLOCKED on
the read, not the write-separation. NO sim/ edit.
