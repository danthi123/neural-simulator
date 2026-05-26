# Direction 4 PRODUCTION DECISIVE = DIRECTION_4_PASS multi-seed (6/6 cells PASS; L=5 OI = 0.977 multi-seed); pure dedicated-pool BEATS D5 hybrid (0.790) AND pillar n=95 G.20 sparse (0.790) at the same cross-bridge cell; pillar n=108 VALIDATED candidate

**Date:** 2026-05-26 ~12:00 EDT (production training); ~12:05 EDT (probe completion)
**Status:** DIRECTION_4_PASS at PRODUCTION scale (multi-seed); pillar n=108 VALIDATED candidate ready for adversarial reviewer dispatch (pre-staged prompt at docs/plans/2026-05-26-direction-4-production-adversarial-reviewer-prompt.md)

## What was tested

Direction 4 production decisive multi-seed run at full scale (n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24, events_per_word=200, M_OBS=16) on 5 bridges × 3 seeds = 15 cells. After the bug fix (commit efbad3d: `_DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS` map producing bridge-specific RNG initialization). Bridge size 11264 neurons per cell.

Training wall: ~5 hr (efficient with cached cells from smoke). Cross-bridge probe wall: 124.9s.

## Result: DIRECTION_4_PASS

Multi-seed mean accuracy:

| Load | OB | OI |
|---|---|---|
| L=2 | **1.000** | **1.000** |
| L=3 | **1.000** | **1.000** |
| L=5 | **1.000** | **0.977** |

Per-seed L=5 OI: seed 42 = 0.965; seed 43 = 0.990; seed 44 = 0.975. All 3 seeds clear 0.80 bar comfortably (margin > 0.16). **All 6 cells PASS** at production scale.

Verdict: **DIRECTION_4_PASS** (full PASS, not BOUNDARY).

## Smoke vs production comparison

| Variant | OB L=2 | OB L=3 | OB L=5 | OI L=2 | OI L=3 | OI L=5 | Verdict |
|---|---|---|---|---|---|---|---|
| Smoke (bugfix) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.983 | PASS |
| **Production (bugfix)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0.977** | **PASS** |

Production essentially identical to smoke; no degradation at production scale.

## Comparison to D5 hybrid + pillar n=95

| Substrate | V | OB L=2 | OB L=3 | OB L=5 | OI L=2 | OI L=3 | OI L=5 | Verdict |
|---|---|---|---|---|---|---|---|---|
| Pillar n=95 G.20 sparse cross-bridge | 160 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.790** | BOUNDARY |
| Pillar n=106 D5 hybrid (production) | 80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | **0.790** | BOUNDARY |
| **D4 dedicated-pool (production; pillar n=108 candidate)** | 80 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.977** | **PASS** |

**D4 dedicated-pool architecture DRAMATICALLY OUTPERFORMS BOTH** D5 hybrid AND G.20 sparse cross-bridge at the L=5 OI cell:
- D4 production OI L=5 = 0.977
- D5 hybrid production OI L=5 = 0.790 (BOUNDARY at FHRR capacity edge)
- G.20 sparse n=95 OI L=5 = 0.790 (same edge)

D4 SHIFTS the capacity boundary substantially — from 0.790 (below bar) to 0.977 (well above bar).

## Biology-translatable insight (the unification + simplification result)

The cumulative finding across D4 + D5 + n=95 is:

**Dedicated-pool architecture (each concept = its own 200-neuron biology-faithful pool with FS interneurons + lang_input/lang_output pathways) is the CLEANEST cross-bridge composition substrate.** The bio_brain_regions pure dedicated-pool design (per pillar n=98/n=105) supports cross-bridge composition at OB perfect + OI passing every cell once bridges have distinct activity (the bug fix).

The D5 hybrid's shared sparse pool was a workaround for the cross-bridge uniformity bug, not a necessary architectural component. Pure dedicated-pool is BETTER than hybrid at the L=5 OI cell (0.977 vs 0.790). The competition between dedicated and shared pools in the hybrid actually HURTS the cross-bridge readout.

The G.20 sparse-only architecture (pillar n=95) also hits the OI L=5 boundary at 0.790. The dedicated-pool architecture genuinely exceeds the sparse architecture's cross-bridge cleanliness at this scale.

**Bottom line**: the biology-faithful dedicated-pool architecture, once correctly seeded per-bridge, is the highest-performing cross-bridge substrate the project has produced. Both the "dynamics-class exhausted" and "substrate-geometry limit" narratives are now substantially refuted by what was a single-class bug across multiple architectures.

## Pillar n=108 VALIDATED candidacy

All pre-registered criteria met:
- Multi-seed (3/3 seeds at every cell)
- Production scale (n_lang=2048, events=200; not smoke)
- OB PASSES every cell every seed (perfect 1.000)
- OI PASSES every cell every seed multi-seed (1.000 / 1.000 / 0.977)
- Bug fix correctness verified (cos 0.01-0.03 across bridges; was 1.0000 byte-identical)
- Frozen verdict module thresholds preserved
- Production scale confirms smoke pattern (no degradation)
- DRAMATIC improvement over D5 hybrid + n=95 (0.977 vs 0.790)

**Pre-registered next concrete action**: dispatch adversarial reviewer per the pre-staged prompt at `docs/plans/2026-05-26-direction-4-production-adversarial-reviewer-prompt.md`. If reviewer CLEAR, promote pillar n=108 VALIDATED.

## Cumulative autonomous arc state

**Three pillars promoted; one production-decisive PASS candidate**:
- n=105 VALIDATED (D3 V=32 single-substrate)
- n=106 BOUNDARY (D5 hybrid cross-bridge)
- n=107 VALIDATED (Q-tertiary Wang 2002 NMDA bistability)
- n=108 VALIDATED candidate (D4 dedicated-pool cross-bridge — this finding)

**Four bug-induced reversals** in 24 hours:
1. D5 NEGATIVE → bug fix → BOUNDARY (n=106)
2. Q PARTIAL (4 axes) → nmda_ratio fix → VALIDATED (n=107)
3. D4 NEGATIVE → bug fix → VALIDATED candidate (n=108)
4. Direction I bound (60-neuron PFC) → reframed by n=107 (works at 1000+ neurons)

The "bug-discovery first" pattern is now established as the discipline default whenever architecture returns essentially-chance results.

## What is preserved unconditionally

- Pillars n=93/n=94/n=95/n=96/n=97/n=98/n=105/n=106/n=107 stand UNAFFECTED
- Direction M deliverable + Direction R-v3 envelope stand
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- D4 verdict module thresholds NOT modified
- D5 hybrid (pillar n=106) stands; D4 dedicated-pool is a parallel result on the same substrate, not a replacement

## Files

- Production training: research/findings/raw/direction_4_5bridge_production_bugfix.json + .log
- Production probe: research/findings/raw/direction_4_cross_bridge_production_bugfix.json + .log
- Bug fix commit: efbad3d
- Smoke result: research/findings/2026-05-26-DIRECTION-4-NEGATIVE-INVALIDATED-* (and the smoke commit)
- Reviewer prompt (pre-staged): docs/plans/2026-05-26-direction-4-production-adversarial-reviewer-prompt.md
- D5 hybrid production reference: research/findings/2026-05-26-DIRECTION-5-PRODUCTION-DECISIVE-PARTIAL-OI-L5-0.790-mirrors-pillar-n95-pillar-n106-candidate.md
- Pillar n=95 G.20 sparse reference: research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md
