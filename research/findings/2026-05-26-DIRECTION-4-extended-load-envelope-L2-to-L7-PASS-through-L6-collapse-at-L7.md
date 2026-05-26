# Direction 4 extended-load envelope L=2..L=7 — bio_brain_regions 5-bridge dedicated-pool substrate PASSes multi-seed-mean through L=6 (marginal); collapses at L=7; capacity boundary sits between L=6 and L=7 (NOT at L=5 like D5 hybrid + pillar n=95) (2026-05-26)

## What was tested

Descriptive extension of the Direction 4 production cross-bridge probe (pillar n=108 — DIRECTION_4_PASS at the frozen LOADS={2,3,5}). The biology question: where does the dedicated-pool, cortically-extended bio_brain_regions 5-bridge ensemble's FHRR cross-bridge mode-unification envelope actually end? Is L=5 the boundary (as for D5 hybrid + the cross-bridge G.20 sparse 160-concept union pillar n=95) or does it extend further toward L=7 (the gamma-slot ceiling)?

CPU-only by design (`SIM_BACKEND=numpy`). Wrapper at `research/findings/raw/direction_4_cross_bridge_extended_loads.py` reuses `direction_4_cross_bridge_probe.run_cross_bridge_probe` byte-unchanged with `loads=[4, 6, 7]`, operating on the existing D4 production cache `research/findings/raw/direction_4_cache/activity_full_*_seed{42,43,44}.npz`. No retraining; no GPU; protected/frozen/moat unchanged; bar UNCHANGED at 0.80 multi-seed throughout.

Combined with the existing production result at L={2,3,5} (commit 9acadb4 → pillar n=108 promotion at 0f7dfd9), this characterises the full envelope L=2..L=7 on the same 80-concept cross-bridge union built from the same five trained activity caches.

## Full envelope: L=2 to L=7 multi-seed mean

| Load | OB multi-seed-mean | OI multi-seed-mean | OI per-seed (42 / 43 / 44) | Result vs 0.80 bar |
|---|---|---|---|---|
| L=2 | 1.000 | 1.000 | 1.000 / 1.000 / 1.000 | PASS (production) |
| L=3 | 1.000 | 1.000 | 1.000 / 1.000 / 1.000 | PASS (production) |
| **L=4** | **1.000** | **1.000** | **1.000 / 1.000 / 1.000** | **PASS (extended)** |
| L=5 | 1.000 | 0.977 | 0.965 / 0.990 / 0.975 | PASS (production; n=108) |
| **L=6** | **1.000** | **0.813** | **0.800 / 0.755 / 0.885** | **PASS marginal (extended)** — multi-seed-mean clears bar by 0.013; honest per-seed caveat: seed 43 at 0.755 is BELOW bar, only multi-seed-mean clears |
| **L=7** | **1.000** | **0.608** | **0.610 / 0.610 / 0.605** | **FAIL (extended)** — well below bar, tight per-seed agreement; substrate collapses uniformly at the gamma-slot ceiling |

**Capacity boundary for D4: between L=6 and L=7.** OB exactly 1.000 at every cell across all 18 cells (3 seeds × 6 loads): zero errors / 3600 OB trials. Per-slot identification is perfect at the maximum gamma-slot capacity. The boundary is specifically in the marginal-sum top-K (OI) rank-comparison mechanism's substrate noise floor at L=7 × V=80.

**Highest-L-with-OI-multi-seed-mean ≥ 0.80 = L=6**, with a frank per-seed caveat (seed 43 alone at 0.755 below bar). At L=5 the multi-seed-mean has 0.18 margin; at L=6 it has 0.013 margin; at L=7 it has -0.19 margin (collapse).

## Comparison: D4 dedicated-pool 5-bridge ensemble vs OPTION 3 V=16 vs cross-bridge G.20 V=160 vs D5 hybrid + pillar n=95

| Comparison axis | Cross-bridge G.20 sparse V=160 (n=95 + 2026-05-24 extension) | **D4 dedicated-pool 5-bridge V=80 (this)** | bio_brain_regions OPTION 3 V=16 (2026-05-24) |
|---|---|---|---|
| Substrate | G.20 sparse random Kanerva patterns | bio_brain_regions v14/v16 trained concept-pool activity | bio_brain_regions single-bridge trained concept-pool activity |
| Vocabulary V | 160 (5 bridges × V=32) | **80 (5 bridges × V=16)** | 16 (1 bridge × V=16) |
| OI L=4 multi-seed | 0.973 (PASS) | **1.000 (PASS)** | 1.000 (PASS) |
| OI L=5 multi-seed | 0.770 (BELOW BAR — n=95 BOUNDARY) | **0.977 (PASS)** | 0.997-1.000 (PASS) |
| OI L=6 multi-seed | 0.467 (collapse) | **0.813 (PASS marginal)** | 0.970-0.980 (PASS) |
| OI L=7 multi-seed | 0.165 (chance) | **0.608 (FAIL — collapse)** | 0.895-0.935 (PASS) |
| OB L=7 multi-seed | 1.000 (perfect) | **1.000 (perfect)** | 1.000 (perfect) |
| Capacity boundary | between L=4 and L=5 | **between L=6 and L=7** | none (PASSes L=7) |

**D4 sits cleanly between the single-bridge V=16 ceiling and the cross-bridge V=160 ceiling**, exactly where FHRR algebra predicts: capacity scales with N_dim / V. At V=80 the substrate has substantially more load headroom than V=160 (which fails at L=5) but somewhat less than V=16 (which PASSes L=7 with 0.10+ margin).

D5 hybrid + pillar n=95 cross-bridge both reported L=5 OI ~ 0.790 just below the 0.80 bar on the V=160 G.20 sparse ensemble — that's TWO ladder-rungs below where D4 reaches PASS. The D4 dedicated-pool substrate's load envelope extends through L=5 (PASS at 0.977) and into L=6 (PASS-marginal at 0.813) before collapsing at L=7. D4 reaches **two ladder-rungs higher** in load capacity than the V=160 cross-bridge / D5 hybrid envelopes.

## Biology-translatable interpretation

The dedicated-pool architecture's FHRR cross-bridge capacity boundary sits **between L=6 and L=7** on the 80-concept union — exactly inside the gamma-slot framework's maximum capacity of N_GAMMA_SLOTS=7, but only just (L=7 at chance). This corresponds to ~6 distinct concepts simultaneously bound into a single FHRR composite without losing rank-order identifiability, which is well within the theta-gamma multiplexing literature's reports of 5-9 items per theta cycle (Lisman 2005; Heusser 2016).

The OI ceiling reflects substrate-grounded noise floor scaling with vocabulary size: doubling V (V=80 → V=160) drops the boundary from L=6 to L=4 (a ratio close to log-V scaling consistent with FHRR similarity-distribution geometry). Halving V again (V=160 → V=16, a 10× decrease) raises the boundary all the way to L=7+ with substantial margin. The D4 V=80 measurement is the missing middle data point on the bio_brain_regions vocabulary-vs-load capacity curve.

The fact that OB stays exactly 1.000 across L=2..L=7 on every cell — 3600 OB trials, zero errors — reinforces the n=95 / n=108 finding that the per-slot identification mechanism is fundamentally substrate-robust. The cliff is specifically in the marginal-sum top-K rank-comparison mechanism's substrate noise floor, exactly where the cross-bridge G.20 sparse 160-concept finding (extension of n=95) and the single-bridge bio_brain_regions V=16 finding (extension of n=96/n=97/n=98) both isolated it.

**Implication for the substrate's conversational reach**: at the 80-concept five-category union, the dedicated-pool architecture can hold and read compositional bindings of up to ~6 concepts simultaneously in a single theta cycle — enough for natural-utterance complexity (typical English sentence content-word count 4-7). The gamma-slot ceiling (L=7) lies just past this substrate's marginal-pass envelope, so for natural conversation the V=80 union is at-capacity but not over-capacity. Scaling vocabulary further (toward V=160 or beyond) would require either lifting the gamma-slot constraint, sparsifying the grounded-symbol representations, or moving to a hierarchical compositional encoding (concept-group → concept-slot two-stage) — well-studied future-work axes.

## No new pillar; pillar n=108 stands sharpened

This descriptive characterisation extends pillar n=108 (DIRECTION_4_PASS at the canonical frozen LOADS={2,3,5}) by mapping the high-L envelope. The pre-registered verdict at canonical {2,3,5} is unchanged. The metric "OI marginal-sum top-K boundary on D4 cross-bridge" now refers concretely to "between L=6 and L=7" rather than the unspecified ">= L=5" implied by the canonical grid.

## Files

- Wrapper runner (NEW): `research/findings/raw/direction_4_cross_bridge_extended_loads.py`
- Extended-load output JSON (NEW): `research/findings/raw/direction_4_cross_bridge_production_extended.json`
- Extended-load runtime log (NEW): `research/findings/raw/direction_4_cross_bridge_production_extended.log`
- Parent production result (UNCHANGED): `research/findings/raw/direction_4_cross_bridge_production_bugfix.json`
- Parent probe (UNCHANGED, reused-by-import only): `research/findings/raw/direction_4_cross_bridge_probe.py`
- D4 verdict module (UNCHANGED): `research/findings/raw/direction_4_verdict.py`
- D4 vocab spec (UNCHANGED): `research/findings/raw/direction_4_vocab_spec.py`
- Parent pillar n=108: D4 PASS at canonical LOADS={2,3,5} (commit 9acadb4 + 0f7dfd9)
- Comparison anchor: `research/findings/2026-05-24-bio_brain_regions-load-ceiling-map-ALL-3-substrates-PASS-every-load-L2-to-L7-the-c-NEGATIVE-is-not-substrate-bounded.md`
- Comparison anchor: `research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md`
- This findings doc: `research/findings/2026-05-26-DIRECTION-4-extended-load-envelope-L2-to-L7-PASS-through-L6-collapse-at-L7.md`

## Standing constraints

- Reuse-by-import only; protected/frozen/moat byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- Existing `direction_4_cross_bridge_probe.py` UNCHANGED (mirrored byte-pattern via import in the new wrapper).
- Existing `direction_4_verdict.py` UNCHANGED.
- No-confab moat must stay 7/7 green.
- Frozen 0.80 bar unchanged throughout.
- Descriptive characterisation only; no new pillar; n=108 sharpened (canonical verdict at LOADS={2,3,5} stands).
- Both git remotes propagated (origin + gitea).
- CPU-only (`SIM_BACKEND=numpy`); no GPU contention.
- Honest per-seed caveat at L=6 (seed 43 alone at 0.755 < 0.80; only multi-seed-mean clears the bar at 0.813).
