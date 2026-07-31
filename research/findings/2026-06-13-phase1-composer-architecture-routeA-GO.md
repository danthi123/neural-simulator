---
type: finding
status: qualified
date: 2026-06-13
---

# Phase 1 composer-architecture de-risk: ROUTE A (per-bridge composers + cross-bridge V-tag) = GO — vocabulary-independent cost, full conversational capability at 8 bridges; route B not required

**Date:** 2026-06-13. **Runner:** `research/runners/phase1_composer_ab_derisk.py` (commit 6b36db78). **Backend:** `SIM_BACKEND=cupy` (GPU). **Raw:** `research/findings/raw/_phase1_composer_routeA_512_seed42.json` + `.log`. **Scope:** 8 bridges × 64 = 512 concepts, route A (`--composer per-bridge --per-bridge-D 256`), seed 42, learned cortex. **Design:** `docs/plans/2026-06-12-phase1-composer-architecture-design.md`.

> **Verdict: ROUTE A = GO.** Per-bridge composers (each small, D=256 over its 64-concept shard) + the validated cross-bridge V-tag identity layer deliver the FULL conversational capability at 8 bridges — the conversational matrix (per shard), within-bridge generalization, cross-bridge composition, and the no-confab moat all pass, anti-cheats collapse — at a cost that is **vocabulary-independent** (flat from 8 to 32 bridges). This settles **Phase 1's composer-architecture decision: route A.** The crux held — the matrix needs within-bridge generative binding + cross-bridge IDENTITY composition, NOT cross-bridge generative binding — so **route B (one union composer at D≈5.5k) is NOT required** and is skipped (no failed cross-bridge generative need; route A avoids route B's untested FHRR signal-to-noise risk at 2,048 concepts).

## Why this ran
The 3-bridge ensemble de-risk (option b, GO) surfaced that the FHRR composer dimension must scale with the union vocabulary (the `clause` cell failed at D=128/192-concepts, passed at D=512). Extrapolated to 2,048 concepts a single union composer needs D≈5.5k — untested FHRR territory (6.4× the validated 320-concept scale). The Phase-1 design recommended **route A: per-bridge composers** (each composes only its 64 concepts, so its dimension and cleanup cost are independent of the total vocabulary; cross-bridge facts use the validated V-tag identity layer). This de-risk tests route A at 8 bridges and compares its cost to route B.

## Results (8 bridges × 64 = 512 concepts, route A per-bridge D=256, seed 42)

| Gate | Result |
|---|---|
| **A — within-bridge conversational matrix** (per shard; clause within one bridge) | `all_pass=True` — every one of the 8 bridges passes its within-bridge matrix (≥5/6 + moat), zero abstention breaches |
| **B — within-bridge generalization** (per bridge) | **0.988–1.000** (≈4× chance) on all 8 bridges; B2 moat 0 false-accepts; C1 permuted-similarity collapses (0.06–0.19) + C4 random-shard collapses (0.156) |
| **X — cross-bridge composition** | per-bridge dispatch (X-pb) what/who = **1.0/1.0**, 0 abstention breaches; X-vtag M3 top2=1.00, **signal/floor 20.02×**; Cx PERMUTED (fixed anti-cheat) collapses (top2=0.00) |
| **moat over cross-bridge facts** (C3) | agreement 1.000, host-abstain/gate-accept=0, floor-false-accepts=0, lesion collapses → intact |
| **COST** | **per-bridge D=256, total codebook 1.16 MB, per-bind RF 512 neurons, cleanup argmax width = 64 (per bridge, NOT the union vocab) — `vocabulary_independent_per_op=True`** |

`ROUTE PER-BRIDGE VERDICT: GO`. Total elapsed ~6.7 h (32 anti-cheat cortex re-learns + the live V-tag layer).

## The decisive advantage: vocabulary-independent cost
Route A's per-operation cost (composer dimension D=256, cleanup width = the 64-concept shard, RF bridge size) is **fixed per shard, independent of the total vocabulary** — so it is identical at 8 bridges and at 32 bridges (2,048 concepts). Route B's union composer would need D≈5.5k at 2,048, an untested extrapolation from a single V=192 data point, risking FHRR cleanup signal-to-noise collapse at 6.4× the validated 320-concept scale and a much larger per-op RF bridge. Route A carries no such risk: each bridge stays a small, validated 64-concept composer. **⇒ route A is the build's composer architecture.**

## The crux, confirmed
The design's load-bearing hypothesis — that the conversational matrix needs *within-bridge* generative VSA binding + *cross-bridge IDENTITY* composition, not cross-bridge generative binding — is confirmed: route A passes the full matrix with the clause bound *within* a bridge and all cross-bridge facts handled by V-tag identity recall (no generative cross-bridge binding), and the cross-bridge moat stays intact. So no cross-bridge generative structure is needed → route B is unnecessary.

## Honest scope
- **8 bridges, seed 42** (single-seed). The within-bridge result is saturated (0.99–1.00 on all 8 bridges) and the underlying mechanism + capability are multi-seed-validated, so single-seed route A at 8 bridges is strong evidence; multi-seed route-A is a lower-priority confirmation than advancing Phase 1.
- Route A's cross-bridge is IDENTITY composition (V-tag), not graded generalization — by design (graded similarity is a within-bridge property).
- The 32-bridge fan-out + the production train (32 cortex bridges at 2,048 concepts) are the remaining Phase-1 steps (the 32-bridge fan-out de-risk is in flight).

## Conclusion + next
Phase 1's first decision is made: **route A (per-bridge composers + cross-bridge V-tag), vocabulary-independent cost, full capability at 8 bridges.** Next (in flight): the 32-bridge fan-out de-risk (does cross-bridge SNR + the moat hold at 32-bridge fan-out, 4× the validated 8?), then the production sharding (32 semantic clusters of 64) + the 32-cortex-bridge train at 2,048 concepts. No `sim/` edits. No banking.
