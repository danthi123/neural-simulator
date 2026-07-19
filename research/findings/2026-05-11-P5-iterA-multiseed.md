# P5 ventral semantic stream multi-seed result (p5_iterA_seed)

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Catalog:** G.11 (dual-stream, Hickok & Poeppel) + G.13 (Wernicke's area)
**Seeds:** [42, 43, 44]
**Verdict:** 0/3 OVERALL PASS (0/3 comprehension, 0/3 naming)

## Per-seed results

| Seed | apple_self | apple_river | Comp | Naming ratio | Naming | Overall | Wall |
|---|---|---|---|---|---|---|---|
| 42 | 0.227 | 0.174 | FAIL | 1.08x | FAIL | FAIL | 295s |
| 43 | 0.235 | 0.105 | FAIL | 0.89x | FAIL | FAIL | 111s |
| 44 | 0.251 | 0.237 | FAIL | 1.00x | FAIL | FAIL | 92s |

Targets: apple_self > 0.5 AND apple_river < 0.4 (Comp); naming ratio > 1.3x (Naming).

## Multi-seed averages

- Comprehension cosines: same-concept 0.238, cross-concept 0.172
- Naming ratio (causal/baseline): 0.99x
- Mean wall clock: 166 sec/seed

## Interpretation

**FAIL (0/3).** Architecture doesn't reliably produce word<->meaning binding.

NOTE: same-concept > cross-concept in 3/3 seeds — methodology picks up some signal, but magnitude is below absolute threshold (>0.5).
