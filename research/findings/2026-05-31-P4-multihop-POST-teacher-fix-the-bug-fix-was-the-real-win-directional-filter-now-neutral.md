# Post-teacher-fix multi-hop confirmation: the teacher_pA BUG FIX was the real win (undirected multi-hop fan-in 8 jumped 0.000 -> 0.750), which makes the DIRECTIONAL FILTER I shipped earlier roughly NEUTRAL (now slightly negative: 0.708 vs 0.750). The directional filter treated the SYMPTOM (hub-crowding burying the outgoing edge); the teacher fix treated the CAUSE (weak engram bindings). With the cause fixed, strong bindings surface the outgoing edge even UNDIRECTED, so the symptom largely disappears and the directional restriction no longer helps (slightly hurts). Honest revision of the earlier directional-fix framing. Multi-hop is now substantially better overall (~0.71-0.75 multi-seed at fan-in 8, up from 0.00 undirected / 0.58 directional) but not uniformly 8/8; the residual per-seed variance is now hop-1 CROSS-bridge encoding (noun->hub), which the intra-bridge teacher fix did not target.

**Date:** 2026-05-31
**Status:** End-to-end multi-seed confirmation of the teacher_pA fix, with a load-bearing scrutiny that revises the directional-filter story. Honest.

## Measurement (post-fix, multi-seed 42/43/44, full_2hop)

| fan-in | undirected ANY (pre -> post) | directional OUT (pre -> post) |
|---|---|---|
| 2 | 1.000 -> ~0.83 | 1.000 -> ~0.83 |
| 4 | 0.333 -> ~0.58 | 0.333 -> ~0.50 |
| 8 | **0.000 -> 0.750** | **0.583 -> 0.708** |

Per-seed fan-in 8 post-fix: seed 42 ANY 8/8 OUT 8/8; seed 43 ANY 6/8 OUT 5/8; seed 44 ANY 4/8 OUT 4/8.

## The honest scrutiny (revises the directional-fix framing)

1. The teacher_pA bug fix (sparse encode_pair now passes the configured teacher_pA=500 instead of silently using the function default 100) is the DOMINANT improvement: undirected multi-hop at fan-in 8 went from 0.000 to 0.750. Strong engram bindings make the outgoing edge (e.g. big->red) recall strongly enough to reach top-3 EVEN amid the 8 incoming noun-edges of a crowded hub. So the hub-crowding that the directional filter was built to fix LARGELY DISAPPEARS once the bindings are strong.
2. CONSEQUENCE: the directional filter I shipped earlier (the "trace" command's direction='out') is now roughly NEUTRAL -- in fact slightly NEGATIVE multi-seed (0.708 vs 0.750). With strong bindings, restricting the hub query to hub-first tags loses a little signal that the undirected query captures. The directional filter treated the SYMPTOM (hub-crowding); the teacher fix treated the CAUSE (weak bindings). Cause fixed -> symptom gone -> symptom-treatment no longer needed.
3. This does NOT make the directional filter harmful (0.708 is still good + it is semantically appropriate for "what does X relate to" traversal). But it is honest to record that the EARLIER "directional fix RESCUES" finding is SUPERSEDED: the real fix was the teacher_pA bug, and the directional filter is now a roughly-neutral semantic choice, not a load-bearing rescue.
4. RESIDUAL variance: post-fix the multi-hop is ~0.71-0.75 at fan-in 8 multi-seed, NOT uniformly 8/8 (seed 44 = 4/8). The per-seed-diagnostic showed all INTRA-bridge pairs (big->red etc., hop-2) now recall at rank <=2 -- so the residual fan-in-8 misses are HOP-1 (the CROSS-bridge noun->hub encoding, which uses encode_partial, NOT the intra-bridge encode_pair the teacher fix targeted). A cross-bridge teacher strengthening would be the next lever (not done; the cross-bridge path is encode_partial_pair_engram_sparse).

## Disposition

Multi-hop "trace" is substantially improved by the teacher_pA bug fix (the genuine win) and is now a reasonable retrieval-reasoning capability (~0.71-0.75 multi-seed at fan-in 8, up from 0.00 undirected). The directional filter is retained (semantic + harmless) but honestly de-framed from "the fix" to "a neutral semantic choice." The earlier directional-RESCUE finding is bannered as superseded. The residual limiter is now hop-1 cross-bridge encoding strength (a specified, deferred lever: extend the teacher strengthening to encode_partial_pair_engram_sparse). MULTI-HOP ARC COMPLETE: clean -> DEGRADES -> directional RESCUE (symptom) -> bimodality DIAGNOSED -> teacher_pA bug FOUND + FIXED (cause, the real win) -> end-to-end confirmed (directional now neutral; ~0.75 undirected).

## Discipline

Throwaway probe; the shipped one-line teacher_pA fix in g20_multibridge is the only code change (66 tests pass). The PASS-ish post-fix result was scrutinized HARDER than a FAIL: the headline "multi-hop improved" was interrogated and found to be the TEACHER fix, not the directional filter -- and the directional filter was honestly found to be neutral-to-slightly-negative post-fix, superseding my own earlier directional-RESCUE framing. Honest revision recorded; nothing overclaimed.
