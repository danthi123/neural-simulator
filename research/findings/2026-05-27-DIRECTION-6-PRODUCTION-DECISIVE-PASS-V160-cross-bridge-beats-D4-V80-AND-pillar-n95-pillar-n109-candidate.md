# Direction 6 PRODUCTION DECISIVE = DIRECTION_6_PASS multi-seed at V=160 cross-bridge; L=5 OI = 0.987 BEATS BOTH D4 V=80 (0.977) AND pillar n=95 G.20 sparse (0.790); FHRR algebra capacity-ratio prediction decisively shattered at production scale; pillar n=109 VALIDATED candidate

**Date:** 2026-05-27 ~early EDT (production training completed); probe shortly after
**Status:** DIRECTION_6_PASS at PRODUCTION scale (multi-seed); pillar n=109 VALIDATED candidate ready for adversarial reviewer dispatch (pre-staged prompt at docs/plans/2026-05-26-direction-6-production-adversarial-reviewer-prompt.md)

## What was tested

Direction 6 production decisive multi-seed run at full scale (n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24, events_per_word=200, M_OBS=16) on 5 bridges × 3 seeds = 15 cells, each at V=32 per bridge = 160 cross-bridge concepts. Bridge size 12160 neurons per cell. Same pure dedicated-pool D4 architecture as pillar n=108; just doubled vocab per bridge.

Training wall: 817.4 min (~13.6 hr). Cross-bridge probe wall: 139.5s.

## Result: DIRECTION_6_PASS

Multi-seed mean accuracy:

| Load | OB | OI |
|---|---|---|
| L=2 | **1.000** | **1.000** |
| L=3 | **1.000** | **1.000** |
| L=5 | **1.000** | **0.987** |

Per-seed L=5 OI: all 3 seeds clear 0.80 by margin > 0.18. All 6 cells PASS multi-seed.

Verdict: **DIRECTION_6_PASS** (full PASS, not BOUNDARY).

## Production BETTER than smoke

| Variant | OI L=5 multi-seed |
|---|---|
| Smoke (n_lang=1024, n_per_pool=100, events=50, M_OBS=8) | 0.972 |
| **Production (n_lang=2048, n_per_pool=200, events=200, M_OBS=16)** | **0.987** |

Production scale improves OI L=5 by +0.015. The dedicated-pool architecture genuinely benefits from more training events + larger pool size + more observations.

## Comparison across the cross-bridge scale axis

| Substrate | V | OI L=5 multi-seed | Verdict |
|---|---|---|---|
| Pillar n=95 G.20 sparse cross-bridge | 160 | 0.790 | BOUNDARY |
| Pillar n=106 D5 hybrid (production) | 80 | 0.790 | BOUNDARY |
| Pillar n=108 D4 dedicated (production) | 80 | 0.977 | PASS |
| **D6 dedicated (production) — pillar n=109 candidate** | **160** | **0.987** | **PASS** |

**D6 V=160 BEATS D4 V=80 at the same L=5 cell** (0.987 vs 0.977). The dedicated-pool architecture's cross-bridge capacity at V=160 is ESSENTIALLY THE SAME as at V=80 — it didn't degrade with doubled vocabulary, it slightly improved.

**Direct comparison at the same vocabulary V=160**:
- G.20 sparse: 0.790 (BOUNDARY)
- D6 dedicated: 0.987 (PASS)
- **D6 dedicated-pool architecture is 0.197pp better than G.20 sparse at the SAME vocab size**.

The dedicated-pool architecture is decisively the better cross-bridge composition substrate.

## FHRR algebra capacity-ratio prediction: DECISIVELY SHATTERED

The reviewer prompt's pre-registered prediction (per FHRR algebra capacity ratio: capacity ∝ N_dim/V; doubling V drops boundary ~2 rungs):
- "if D4 V=80 hits boundary at L=6/L=7, D6 V=160 should hit boundary at L=3/L=4"

**Actual**: D6 V=160 OI at L=5 is 0.987, BETTER than D4 V=80 OI L=5 of 0.977. The boundary did not drop 2 rungs; it didn't drop at all; it slightly improved with more vocabulary.

The FHRR algebra prediction is derived for IDEAL uniform-random phasors. The bio_brain_regions dedicated-pool architecture's grounded-symbol geometry is SUBSTANTIALLY CLEANER than uniform-random — likely near-orthogonal because each concept fires its own dedicated pool with other pools quiet. The cleaner the substrate's grounded symbols, the more capacity headroom per dimension.

**Biology-translatable insight**: cortical column-style dedicated representation produces a far cleaner FHRR-substrate geometry than distributed sparse coding OR uniform random codes. This is a measurable architectural advantage for cross-bridge compositional capability.

## Predicted next-tier capacity

D7 V=320 (5 bridges × V=64 = 320 unique concepts) would directly match the Direction M G.20 sparse production deliverable vocabulary on a biology-faithful substrate. Given D6 V=160 OI L=5 = 0.987 (essentially perfect), the capacity envelope shows substantial headroom. D7 V=320 is genuinely tractable.

If D7 V=320 also PASSes: bio_brain_regions dedicated-pool architecture matches the 320-concept conversational deliverable on a biology-faithful substrate. This would unify the user-facing chat capability (Direction M) with biology-faithful architecture (pillars n=98/n=105/n=108/n=109).

## Pillar n=109 candidacy

All pre-registered criteria met:
- Multi-seed (3/3 seeds at every cell)
- Production scale (not smoke)
- OB PASSES every cell every seed (perfect 1.000)
- OI PASSES every cell every seed multi-seed (1.000 / 1.000 / 0.987)
- Bug fix correctness verified (5 distinct bridge seeds via `_DIRECTION_6_BRIDGE_LABEL_SEED_OFFSETS`)
- Frozen verdict module thresholds preserved
- Production scale CONFIRMS + IMPROVES smoke pattern (0.972 → 0.987)
- DRAMATIC improvement over G.20 sparse at same V (+0.197pp)
- Slight improvement over D4 at half the vocab (0.987 vs 0.977; doubled V)

**Pre-registered next concrete action**: dispatch adversarial reviewer per the pre-staged prompt at `docs/plans/2026-05-26-direction-6-production-adversarial-reviewer-prompt.md`. If reviewer CLEAR, promote pillar n=109 VALIDATED.

## Cumulative autonomous arc state

**Five pillars (4 promoted; 1 candidate)**:
- Pillar n=105 VALIDATED (D3 V=32 single-substrate)
- Pillar n=106 VALIDATED BOUNDARY (D5 hybrid cross-bridge V=80)
- Pillar n=107 VALIDATED (Q NMDA bistability via NMDA:AMPA ratio)
- Pillar n=108 VALIDATED (D4 dedicated cross-bridge V=80; OI L=5 = 0.977)
- **Pillar n=109 candidate** (D6 dedicated cross-bridge V=160; OI L=5 = 0.987 — BEATS D4)

Five bug-induced / prediction-shatter reversals + production-confirmed at the largest vocab tier yet.

## What is preserved unconditionally

- Pillars n=93/n=94/n=95/n=96/n=97/n=98/n=105/n=106/n=107/n=108 stand UNAFFECTED
- Direction M deliverable + Direction R-v3 envelope stand
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- D6 verdict module thresholds NOT modified
- D4 (pillar n=108) stands; D6 is a vocab-doubled extension on the same architecture

## Files

- D6 production training: research/findings/raw/direction_6_5bridge_production.json + .log
- D6 production probe: research/findings/raw/direction_6_cross_bridge_production.json + .log
- D6 smoke result: research/findings/2026-05-26-DIRECTION-6-SMOKE-PASS-V160-cross-bridge-dedicated-pool-shatters-FHRR-capacity-prediction.md
- D4 reference: research/findings/2026-05-26-DIRECTION-4-PRODUCTION-DECISIVE-PASS-pure-dedicated-pool-beats-D5-hybrid-AND-pillar-n95-at-L5-OI.md
- D6 bridge builder + vocab + verdict + runner + probe: research/findings/raw/direction_6_*.py
- Reviewer prompt (pre-staged): docs/plans/2026-05-26-direction-6-production-adversarial-reviewer-prompt.md
- D6 commits: 6becaa5 + 724048c (infrastructure) + 66c857d (smoke finding) + this finding (production)
