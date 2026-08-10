---
type: finding
status: contributing
date: 2026-08-10
mechanism: composer-offset-invariant-readout
lane: composer
seeds: [42, 43, 44]
---

# The composer is REPRESENTATION-robust to naturalistic (correlated) codes: an offset-invariant resonator readout (and an ML-exhaustive witness) recall 1.00 across the whole correlation sweep where the Euclidean ruler craters — the "naturalistic capacity break" was the READOUT RULER, and the genuine limit is a margin/SNR collapse, not aliasing

Resolves the open question the corrected arity finding named and the adversarial-verify caveat flagged (all composer GOs used idealized near-orthogonal codes): does the composer break under naturalistic code correlation, or is the apparent break a readout artifact? **It is the ruler.**

## The build (offset-invariant readout, NO sim edit)

<!--derived-->

New runner `research/runners/_teacher_loop_arity_capacity_resonator_derisk.py` (forks the correlated runner; imports
its rho/r_shared world + the frozen spiking generator; NO `sim/` edit). Three readout arms on the SAME regeneration
`regen[j]`: (1) the composer's OWN per-family readout atoms as a codebook, centered per family — the centered bundle
`s_c = regen − M_off = Σ_m Cb_m[v_m]` is offset-free BY CONSTRUCTION (recon-err ~1e-15), the offset-invariance the
Euclidean nearest-prototype ruler lacks; (2) a spiking-style RESONATOR (sequential hard-WTA explain-away factorizer,
warm-start + 8 restarts, Frady/Kanerva); (3) an ML-EXHAUSTIVE witness over all K^M combos with the separation
MARGIN (distance from s_c to the nearest WRONG fact) — upper-bounds any factorizer, separating a resonator local
minimum from a genuine representational limit.

## Result (3-seed, M=4 K=3 d=8; rho x r_shared sweep)

<!--derived-->

- **The ML-exhaustive decoder holds recall 1.00 across the ENTIRE sweep** — including rho=1.0 / r_shared=1 where the
  codes are PERFECTLY COLLINEAR (off-diagonal |cos| = 1.000) and the Euclidean-corrected readout craters to 0.21.
  So the representation preserves the identifying information at every correlation level; only the offset-sensitive
  Euclidean ruler cannot read it under correlation. ⇒ **the composer's neural superposition is
  REPRESENTATION-robust to naturalistic codes** (confirms + strengthens `2026-08-10-shared-channel-arity-capacity-CORRECTED-*`).
- **The biological RESONATOR recovers most of the crater** at FULL convergence: rho=1.0/r=1 Euclidean 0.21 →
  resonator 0.90; rho=0.99/r=1 0.58 → 0.77; rho=0.95/r=2 0.83 → 0.90. Convergence rate = **1.00 at every cell** (no
  limit cycles — distinct from the Kent-2020 synchronous-instability failure mode, which was caught mid-build and
  fixed by switching to sequential updates + restarts). The resonator's ~0.10-0.23 shortfall below exhaustive is its
  OWN hard-WTA local-minima suboptimality (closable with more restarts / a soft-annealed resonator), NOT a composer
  limit — the exhaustive witness proves the representation is identifiable there.
- **The genuine naturalistic limit is a MARGIN (SNR) collapse, not exact aliasing.** The separation margin (in units
  of read-noise) collapses monotonically 30× (rho=0) → 1.35× (rho=1.0/r=1), with cos-to-true finally dipping
  0.99→0.92 only in that degenerate corner. So the true bundling limit becomes operative only as rho→1 AND
  r_shared→1 (the bundle geometry degenerating onto a line), where any realistic downstream noise/precision breaks
  identification — which is why the practical readouts degrade there while the noise-free ML decoder does not.

## Skeptical controls (held)

<!--derived-->

rho=0 reproduces recall 1.00 (harness sound); disjoint control = 1.00 everywhere; regeneration recon-err ~1e-15
(the readout reads the exact atom sum, never the held-out prototype); convergence rate reported at every cell (a
recall drop with LOW convergence would be a resonator failure, not a limit — it was 1.00 everywhere, so the resonator
shortfall is genuine local-minima suboptimality). Honest bounds: exhaustive 1.00 is a "no exact aliasing" statement
(the capacity edge is the MARGIN, which does collapse); exhaustive is O(K^M) so at large arity a TRACTABLE
offset-invariant readout must close the resonator↔exhaustive gap — "robust at scale" depends on that readout.

## Consequence

<!--derived-->

The composer arc's forward edge (naturalistic capacity) is resolved: the composer superposition is
representation-robust; the apparent break was the offset-sensitive Euclidean ruler; a biological offset-invariant
resonator recovers it; the genuine limit is a margin/SNR collapse in the degenerate (collinear) corner. NEXT (queued,
lower priority): a soft/annealed resonator to close the ~0.1-0.2 gap to exhaustive, and an M×d margin map to pin the
operating edge — the composer already works for realistic conversational arities.

Artifacts: `research/findings/raw/teacher_loop_arity_capacity_resonator_AGG.json` (+ s42/s43/s44). NO `sim/` edit.
SIM_BACKEND=numpy.
