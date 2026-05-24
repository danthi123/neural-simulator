# Cross-bridge OI load-ceiling map — descriptive extension of pillar n=95 — multi-seed ceiling sits between L=4 and L=5 (not L=5 to L=6); steep monotonic decay above L=5 (2026-05-24)

## What was tested

Descriptive extension of pillar n=95 (cross-bridge mode-unification BOUNDARY; OB perfect, OI L=5 ~0.79 just below the 0.80 bar). The n=95 probe tested loads {2, 3, 5}. This extension maps the OI ceiling precisely across loads {2, 3, 4, 5, 6, 7} on the 160-concept union for BOTH mean-centring conditions (global_mean + per_bridge_mean). CPU-only (`SIM_BACKEND=numpy`); reuses the cross-bridge probe's primitives byte-unchanged + the 160-ensemble caches.

Pre-registered framing was DESCRIPTIVE only — sharpens the n=95 ceiling characterisation; the BOUNDARY pillar's verdict + framing stand.

## Result

Multi-seed-mean OI accuracy at loads {2, 3, 4, 5, 6, 7}:

| Condition | L=2 | L=3 | L=4 | L=5 | L=6 | L=7 |
|---|---|---|---|---|---|---|
| global_mean | 1.000 | 1.000 | 0.973 | 0.770 | 0.467 | 0.165 |
| per_bridge_mean | 1.000 | 0.998 | 0.973 | 0.752 | 0.452 | 0.158 |

**OI ceiling sits between L=4 and L=5** for both conditions:
- Highest load with OI multi-seed-mean ≥ 0.80 = L=4 (0.973; substantial headroom)
- Lowest load below the bar = L=5 (0.752-0.770; just below)

Multi-seed OB is exactly 1.000 at every cell (zero errors; perfect per-slot identification across the full load range — confirming the n=95 finding that ORDER-BEARING extends cleanly cross-bridge).

Decay rate above L=5 is steep + monotonic: ~0.20 per binding from L=4 (0.97) → L=5 (0.77) → L=6 (0.45) → L=7 (0.16). At L=7 OI is essentially chance for picking the right top-7 from a 160-symbol vocabulary.

## Sanity note on the shared-qrng artifact

The n=95 probe (LOADS=[2,3,5]) reported multi-seed L=5 OI = 0.790 (global) / 0.785 (per-bridge). This map (LOADS=[2,3,4,5,6,7]) reports multi-seed L=5 OI = 0.770 / 0.752. The ~0.02 difference is the **shared-qrng artifact** (well-documented in the K=16 extended load-ceiling map finding from 2026-05-23): the `qrng.choice(V, size=load, replace=False)` advances differently when LOADS includes more values, so per-seed L=5 trials sample different (item, fillers) tuples in the two probes. Both samples are VALID; the multi-seed-mean shifts marginally. The "ceiling between L=4 and L=5" finding holds robustly under either sampling.

## Biology-translatable insight (refinement of n=95)

The OI marginal-sum top-K mechanism crosses its noise floor SHARPLY at L=5 × V=160 for the cross-bridge composition. At L=4 it has substantial multi-seed headroom (0.97; 0.17 above bar); at L=5 it's just below (0.76-0.77); at L=6 it's already collapsing (0.45); at L=7 it's essentially chance (0.16). The ORDER-BEARING per-slot decoder remains perfect throughout — the boundary is specifically in the marginal-sum top-K rank-comparison mechanism's substrate noise floor at this cross-bridge × high-load × large-vocabulary corner.

The mean-centring choice (global vs per-bridge common-mode removal) doesn't materially affect the ceiling — within 0.02 at every load. The substrate's grounded-symbol noise floor at L=5 × V=160 — not the mean-centring framing — is the load-bearing geometric constraint.

This refines the n=95 BOUNDARY characterisation:
- Old (n=95): "OI L=5 just below 0.80 bar; OB perfect every cell"
- New (this extension): "OI ceiling between L=4 and L=5; OB perfect every cell across L=2..7; steep monotonic decay above L=5 (~0.20 per binding)"

## Implication for (c) generative-replay

The (c) loop encodes K-tuple PFC frames where K is bounded by the gamma-slot framework (max 7 slots). At K=2-3, the OI decoder has perfect headroom (1.000). At K=4 it still PASSes (0.97). At K=5+ on the cross-bridge 160-concept union it crosses below the bar — but the (c) loop's pre-registered K-ladder is {4, 8, 16} stored sequences, NOT a K=5+ composite slot count. The per-slot OB decoder is the load-bearing one for (c) (each PFC frame slot identifies ONE concept via OB; the K-stored-sequences count is at a different level — how many distinct partial-cues the schema holds, not the slots per cue).

So the OI ceiling at L=5 × V=160 does NOT block the (c) build at the gamma-slot composite level. The (c) loop uses OB at each gamma slot for identification; OI is used at the set-comparison level. The (c) build's substrate test is unaffected by this n=95 extension.

## No new capability pillar

The n=95 pillar stands. This is a descriptive characterisation extension. The metric text in n=95 already referenced "OI marginal-sum top-K ceilings at L=5"; this extension provides the load-ceiling shape that the n=95 metric refers to.

## Files

- Runner: `research/findings/raw/cross_bridge_oi_load_ceiling_map.py`
- Log: `research/findings/raw/cross_bridge_oi_load_ceiling_map.log`
- Output JSON: `research/findings/raw/cross_bridge_oi_load_ceiling_map.json`
- This findings doc: `research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md`
- Parent pillar n=95: `research/findings/2026-05-23-cross-bridge-mode-unification-BOUNDARY-OB-PASSes-perfectly-OI-ceilings-at-L5-on-160-concept-union.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green throughout.
- Frozen 0.80 bar unchanged.
- Descriptive characterisation only; no new pillar; no claim beyond the n=95 BOUNDARY framing.
