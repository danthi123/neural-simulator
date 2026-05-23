# TPAM-scaling-limit probe: the FHRR-biologization arc's TPAM attractor has a non-monotonic CAPACITY WINDOW for per-slot mode-unification identification on grounded symbols (V=8 through V=20 PASS multi-seed-mean; V<8 and V>20 miss)

## Status

Cheap CPU finalisation of the mode-unification thread, completed
2026-05-23. The biologized spiking mode-unification decisive run
produced NEGATIVE_ORDER_INVARIANT_ONLY with the failure precisely
localised to the TPAM attractor at 32-concept vocabulary. This probe
maps the TPAM's per-vocab-size capacity ceiling precisely. The
result is a striking non-monotonic CAPACITY WINDOW: at V=8 through
V=20 the TPAM order-bearing identification multi-seed-mean clears
the frozen 0.80 bar; at V=4 (below window) and V>=24 (above window)
it misses. The simple argmax-of-similarities decoder is perfect
across the entire vocab range (1.000 throughout).

## Background

The biologized spiking mode-unification decisive run on bridgeA_nouns
at 32 concepts gave NEGATIVE_ORDER_INVARIANT_ONLY: order-invariant
PASS multi-seed 1.000; order-bearing via TPAM 0.53/0.34/0.09 at
loads {2,3,5}. Built-in diagnostic showed simple argmax-of-
similarities (no TPAM) gives 1.000 across all loads/seeds on the
same data -- the per-slot unbinds are clean; the TPAM attractor is
the failing component. The FHRR-biologization arc's TPAM was
validated at 0.98 on FACT composition at the 16-concept FILLER
partition (a smaller basin set). At 32-concept full-vocabulary
mode-unification per-slot identification, the TPAM crosses its
capacity ceiling.

This probe maps that ceiling precisely.

## What was run

Cheap CPU probe; reuses the 160-ensemble bridgeA_nouns cache
(byte-identical to what the pre-registered mode-unification runner
would have produced); pure-numpy pipeline; ~minutes total compute.

For each vocab size in {4, 8, 12, 16, 20, 24, 28, 32}: take the
first V words from the bridge's 32-concept vocabulary; build their
grounded symbols (mean-centred consolidated activity at K_VOCAB=16
-> deriver -> phases_to_spikes); build a ResonateFireTPAM over those
V grounded symbols; encode load-2 sequences (item, position)-bindings
via ResonateFireFHRR.encode; per-trial measure per-slot identification
accuracy via the TPAM-attractor decoder (the pre-registered
mode-unification decoder) and the simple argmax-of-phase-similarities
decoder. Multi-seed (42, 43, 44); 100 trials per (vocab, seed); load
fixed at L=2 to isolate the vocab-size effect from the load-capacity
effect.

## Result

```
vocab       TPAM-OB multi-seed mean       simple-OB multi-seed mean
V=4         0.7433       miss             1.0000     PASS
V=8         0.9033       PASS             1.0000     PASS
V=12        1.0000       PASS             1.0000     PASS
V=16        0.9700       PASS             1.0000     PASS
V=20        0.8067       PASS             1.0000     PASS
V=24        0.6067       miss             1.0000     PASS
V=28        0.4767       miss             1.0000     PASS
V=32        0.5000       miss             1.0000     PASS
```

The TPAM has a CAPACITY WINDOW for per-slot mode-unification
identification on grounded symbols: V=8 through V=20 multi-seed-mean
clears the 0.80 bar; V=4 misses (0.74) and V>=24 fall off sharply
(0.61, 0.48, 0.50). The transition between V=20 (0.81) and V=24
(0.61) is sharp. The simple argmax-of-similarities decoder is
perfect across the entire vocab range.

## What this means

The TPAM attractor's per-vocab-size capability is NON-MONOTONIC on
grounded symbols. Below V=8 the small-basin structure has a
dominant pair-interaction that biases the settle; above V=20 the
mass-attractor regime emerges and captures most queries
regardless of input. The window V=8 through V=20 has a clean
attractor structure where TPAM correctly identifies per-slot
unbinds. The 16-concept fact-composition validation (TPAM PASS at
0.98 in the FHRR-biologization arc) sits inside this window.

For per-slot mode-unification identification at 32-concept vocab,
the TPAM is well above its capacity ceiling (V=32 mean 0.50 is
roughly chance for an attractor that converges to one specific
basin regardless of input). The simple parallel-population-matching
decoder operates on the same unbinds and gives 1.000 throughout --
the unbinds carry the encoded items cleanly; only the TPAM's
attractor structure fails to recover them past the window.

The biology-translatable insight set is now complete on this
thread:

1. The algebra supports unified bidirectional readout from one
   theta-gamma encoded code (algebra-PASS).
2. The algebra capacity envelope is wide on load (up to 7-slot
   gamma ceiling), noise (up to substrate-realistic CV 1.6), and
   vocab (up to 256 tested).
3. The algebraic half of biologized mode-unification (order-
   invariant readout via marginal-sum-of-similarities) works on
   the substrate (multi-seed 1.000 at every load).
4. The biologization arc's TPAM attractor identification mechanism
   does NOT transfer to per-slot mode-unification at 32 concepts
   (NEGATIVE_ORDER_INVARIANT_ONLY).
5. The TPAM has a precise NON-MONOTONIC capacity window (V=8
   through V=20 multi-seed-mean PASS) on grounded symbols; a
   different biology-grounded identification mechanism (parallel
   population matching) operates outside this window cleanly.

For the cortical attractor literature (Amit & Treves 1989 capacity
analysis of Hopfield-class networks): the project's biologized
TPAM is a Hopfield variant; its capacity on natural-substrate-
derived patterns shows the non-monotonic structure that random-
pattern analyses do not capture. The window mid-point (~V=12-16
peak) is consistent with the FHRR-biologization arc's 16-concept
validation; the upper edge (V=20-24) is where mass-attractors
emerge.

## What this is, and what it is not

This is a clean cheap CPU finalisation of the mode-unification
thread; the 5th biology-translatable insight on top of the algebra-
PASS, characterisation, biologization NEGATIVE, and diagnostic.
It is NOT a capability claim and does NOT propose a fix; the simple-
argmax decoder's 1.000 throughout is reported only as the upper-
bound reference, not as a new capability (using it would be a new
pre-registered arc per the design's prior framing).

## Next step

The mode-unification thread is now substantively complete:
- Algebra-PASS + characterisation (cheap CPU probes, both PASS).
- Biologization NEGATIVE_ORDER_INVARIANT_ONLY (pre-registered, BOUNDARY
  pillar n=92).
- TPAM-scaling-window precisely mapped (this finding; no new pillar
  -- characterisation refinement).

The natural next pre-registered direction is the broader-horizon
arcs the owner has standing directives on:
- Generative replay (builds ON TOP OF mode-unification once the
  order-bearing side is solved with a biology-grounded mechanism).
- The integrated closed loop.
- Cross-bridge composition.

OR a focused new pre-registered runner with parallel-population-
matching decoder for order-bearing, with its own design + adversarial
review (would likely yield mode-unification both-readouts PASS on
the biologized substrate based on this probe's simple-OB column).

These are major direction choices worth owner steer.

## Honest scope

A cheap CPU finalisation; no GPU; no spiking-substrate work; reuses
the 160-ensemble cache. The frozen 0.80 bar was not moved. No
protected, frozen, or moat module modified. No automatic
differentiation. No-confab moat 7/7 green. No new capability_status
pillar -- this is a characterisation refinement of the existing
BOUNDARY pillar (n=92) on the mode-unification biologization.

## Files / evidence

- Probe output: `research/findings/raw/tpam_scaling_limit_probe.json`
- The biologization NEGATIVE this refines:
  `research/findings/2026-05-23-biologized-spiking-mode-unification-decisive-NEGATIVE_ORDER_INVARIANT_ONLY-TPAM-attractor-doesnt-transfer-to-per-slot-mode-unification.md`
- Algebra-PASS + characterisation:
  `research/findings/2026-05-23-theta-gamma-mode-unification-cheap-numpy-probe-ALGEBRA-PASS-Lisman-Idiart-N16-realisable-on-FHRR.md`,
  `research/findings/2026-05-23-theta-gamma-mode-unification-characterisation-capacity-envelope-wide-on-all-three-axes-algebra-survives-substrate-realistic-noise.md`
- The FHRR-biologization arc's TPAM (the mechanism this probe maps
  the scaling-limit of):
  `research/findings/2026-05-22-attractor-cleanup-biologization-shortcut-3-RESOLVED-abstention-is-a-separate-familiarity-signal-not-a-basin-property.md`
