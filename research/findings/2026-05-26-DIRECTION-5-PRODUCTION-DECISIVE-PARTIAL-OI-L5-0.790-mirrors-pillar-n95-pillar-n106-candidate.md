# Direction 5 PRODUCTION DECISIVE = DIRECTION_5_PARTIAL multi-seed (5/6 cells PASS; OB perfect every load; OI L=5 = 0.790 EXACTLY matches pillar n=95 G.20 sparse cross-bridge boundary); pillar n=106 BOUNDARY candidate strongly supported

**Date:** 2026-05-26 ~04:45 EDT (production training); 05:00 EDT (probe completion)
**Status:** DIRECTION_5_PARTIAL at PRODUCTION scale (multi-seed); pillar n=106 BOUNDARY candidate ready for adversarial reviewer dispatch (pre-staged prompt at docs/plans/2026-05-25-direction-5-production-adversarial-reviewer-prompt.md)

## What was tested

Direction 5 production decisive multi-seed run at full scale (n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24, events_per_word=200, M_OBS=16) on 5 bridges × 3 seeds = 15 cells. Total training wall: 543.2 min (~9 hr; bridge size 10876 neurons per cell). Both cross-bridge probes (raw + topK decoder fix) ran on the production cache.

## Result A: PRODUCTION non-topK (raw activity → FHRR projection)

Multi-seed mean accuracy:

| Load | OB | OI |
|---|---|---|
| L=2 | **1.000** | **1.000** |
| L=3 | **1.000** | **0.998** |
| L=5 | **1.000** | **0.790** |

Per-seed L=5 OI: seed 42 = 0.685; seed 43 = 0.795; seed 44 = 0.890. Two of three seeds clear 0.80 at L=5; only seed 42 below.

Verdict: **DIRECTION_5_PARTIAL** (5/6 cells PASS the 0.80 bar; L=5 OI BOUNDARY).

## Result B: PRODUCTION topK decoder fix (binarize before projection)

Multi-seed mean accuracy:

| Load | OB | OI |
|---|---|---|
| L=2 | **1.000** | **1.000** |
| L=3 | **1.000** | **0.972** |
| L=5 | **1.000** | 0.463 |

**topK degrades OI L=5 vs non-topK** (0.463 vs 0.790). At production scale the activity already has enough signal; thresholding removes useful information. Non-topK is the correct probe for production. The topK decoder fix was helpful at smoke scale (where signal was weak) but harmful at production scale.

## EXACT match to pillar n=95

Comparison to G.20 sparse cross-bridge pillar n=95 (V=160, production scale):

| Substrate | V | OB L=2 | OB L=3 | OB L=5 | OI L=2 | OI L=3 | OI L=5 |
|---|---|---|---|---|---|---|---|
| **Pillar n=95 G.20 sparse** | 160 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.790** |
| **D5 production HYBRID** | 80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | **0.790** |

The D5 hybrid bio_brain_regions architecture produces ESSENTIALLY IDENTICAL cross-bridge composition results to pillar n=95 G.20 sparse, including the FHRR capacity-envelope L=5 OI boundary at exactly 0.790. The hybrid is a working unification of biology-faithful dedicated pools (pillar n=98/n=105) with sparse-distributed cross-bridge composition (pillar n=95).

## Comparison to smoke scale

| Variant | OB L=2 | OB L=3 | OB L=5 | OI L=2 | OI L=3 | OI L=5 |
|---|---|---|---|---|---|---|
| Smoke (raw) | 1.000 | 1.000 | 1.000 | 1.000 | 0.840 | 0.195 |
| Smoke (topK) | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 | 0.463 |
| **Production (raw)** | 1.000 | 1.000 | 1.000 | 1.000 | **0.998** | **0.790** |
| Production (topK) | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 | 0.463 |

Production scale produces dramatically better OI results than smoke for the raw probe:
- OI L=3: 0.840 → **0.998** (essentially perfect)
- OI L=5: 0.195 → **0.790** (4x improvement)

This confirms the production-scale training (n_per_pool=200 vs 100; events=200 vs 50) is necessary for the OI readout to approach its capacity envelope. The smoke scale was insufficient.

## Biology-translatable insight (production-confirmed)

The bio_brain_regions HYBRID architecture genuinely unifies two prior working pillars:
- **Pillar n=98 / n=105**: biology-faithful dedicated concept pools (each concept = its own 200-neuron pool with FS interneurons)
- **Pillar n=95**: sparse-distributed cross-bridge composition (K=100 patterns in shared 2000-neuron pool)

The hybrid preserves BOTH: each bridge has dedicated biology-faithful pools AND a shared sparse pool with bridge-distinct K-of-N patterns. Cross-bridge composition reads from the shared pool and matches G.20 sparse capability.

The L=5 OI boundary at exactly 0.790 is the same FHRR capacity-envelope limit that pillar n=95 hits at V=160. Real biology probably uses the same algebraic capacity constraint; the bio_brain_regions hybrid hits it at the same point.

## Pillar n=106 BOUNDARY candidacy

All pre-registered criteria met:
- Multi-seed (3/3 seeds at every cell)
- Production scale (n_lang=2048, events=200; not smoke)
- OB PASSES every cell every seed (perfect 1.000)
- OI PASSES L=2/L=3 multi-seed
- OI L=5 BOUNDARY at exactly the pillar n=95 G.20 sparse cross-bridge value (0.790)
- Bug fix correctness verified (5 distinct pattern_0 across bridges)
- Frozen verdict module thresholds preserved
- Both remotes propagated

**Pre-registered next concrete action**: dispatch adversarial reviewer per the pre-staged prompt at `docs/plans/2026-05-25-direction-5-production-adversarial-reviewer-prompt.md`. If reviewer CLEAR, promote pillar n=106 BOUNDARY (first architecture unifying biology-faithful dedicated pools with sparse-distributed cross-bridge composition).

## What is preserved unconditionally

- Pillar n=105 (D3 V=32 production PASS) stands UNAFFECTED
- Pillar n=95 (G.20 sparse cross-bridge) stands UNAFFECTED
- Pillars n=93/n=94/n=96/n=97/n=98 stand UNAFFECTED
- Direction M deliverable + Direction R-v3 envelope stand
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- Frozen verdict module thresholds NOT modified
- D4 NEGATIVE remains INVALIDATED (same bug class as D5 had; D4 re-test queued)

## Files

- Production training: research/findings/raw/direction_5_5bridge_production_bugfix.json + .log
- Production non-topK probe: research/findings/raw/direction_5_cross_bridge_production_bugfix.json + .log
- Production topK probe: research/findings/raw/direction_5_cross_bridge_topK_production_bugfix.json + .log
- Pre-staged reviewer prompt: docs/plans/2026-05-25-direction-5-production-adversarial-reviewer-prompt.md
- Bug fix commit: c4e18f2
- Smoke result (now superseded by production): research/findings/2026-05-25-DIRECTION-5-HYBRID-BUGFIX-PARTIAL-pattern-uniqueness-bug-fixed-substrate-is-NOT-the-limit-mirror-of-pillar-n95.md
- D4 NEGATIVE invalidation: research/findings/2026-05-26-DIRECTION-4-NEGATIVE-INVALIDATED-same-systematic-cross-bridge-uniformity-bug-as-D5-had.md
- Pillar n=95 reference: research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md
