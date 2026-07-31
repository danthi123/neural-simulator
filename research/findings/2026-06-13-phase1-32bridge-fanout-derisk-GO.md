---
type: finding
status: contributing
date: 2026-06-13
---

# Phase 1 32-bridge FAN-OUT de-risk: GO (3 seeds: 42/43/44) — cross-bridge composition + the no-confab moat HOLD at 2,048 concepts / 32 bridges (4× the validated 8-bridge fan-out); the deepest Phase-1 risk is RETIRED

**Date:** 2026-06-13. **Runner:** `research/runners/multibridge_graded_derisk.py` (`--mode full --n-bridges 32 --concepts-per-bridge 64 --n-pool 2400 --pattern-size 100 --homeo oja --homeo-target 40 --cycles 10 --n-cross-facts 96`). **Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090). **Raw:** `research/findings/raw/_multibridge_fanout32_seed42.json` (seed 42) + `_multibridge_fanout32_seed43_44.json` (seeds 43/44) + `.log`s. **Scope:** 32 bridges × 64 = **2,048 concepts**, 96 cross-bridge facts, **seeds 42/43/44**, learned graded cortex. **Design:** `docs/plans/2026-06-12-phase1-sharding-and-fanout-design.md` §2.

> **Verdict: GO (3 seeds: 42/43/44).** At a **32-bridge fan-out — 4× the validated 8-bridge fan-out, 2,048 concepts** — every gate holds: all 32 per-bridge graded cortices pass (generalization ≈4× chance, controls collapse); cross-bridge identity composition retrieves the true target over a noise floor that now spans all 2,048 concepts at **20.95× signal/floor** (top2=1.00), *stronger* than the 8-bridge fan-out (20.02×); the fixed permuted anti-cheat collapses (1.12×) so the recall is real; and the no-confab moat is intact (zero breaches, lesion collapses). **This retires the deepest open Phase-1 risk** (build plan §"Genuine open questions" item 2: does cross-bridge composition + the moat survive at 32-bridge fan-out?). The cross-bridge mechanism scales from the validated 3/8 bridges to the full 32-bridge production fan-out. **Confirmed multi-seed: 43/44 = GO** (seed 43 M3 signal/floor **19.46×**, seed 44 **23.24×**, both top2=1.00; M7 anti-cheat collapses 1.08×/1.10×; moat intact, zero false-accepts) — so the fan-out GO holds across **all three seeds 42/43/44**. The multi-day production train is the owner-gated commitment.

## Why this ran
The cross-bridge V-tag identity layer + the no-confab moat were validated to **8-bridge fan-out** (`2026-06-12-cortex-conversation-capability-GO.md`; the route-A composer-architecture de-risk runs 8). The production system is **32 bridges = 4× that fan-out**, with a cross-bridge noise floor that grows to all 2,048 concepts. The build plan named this the remaining open question. The de-risk asks the single load-bearing scaling question: at 32 co-resident-vocabulary bridges, does (i) cross-bridge identity recall still retrieve the true target over a 2,048-concept floor, (ii) the fixed M7/Cx anti-cheat still collapse (a cue must not retrieve a WRONG target), and (iii) the moat stay zero-breach when an unknown cross-cue could match any of 2,048 concepts?

## Results (32 bridges × 64 = 2,048 concepts, seed 42)

| Gate | Result |
|---|---|
| **Per-bridge graded gates (all 32)** | M1 generalization **0.975–1.000** (3.9–4.0× chance), A2+A3 controls collapse [GO]; M2 Pearson(sim,S_true) **+0.73…+0.82**, 2nd-order margin +0.48…+0.51, graded=1 [GO]; **G5 anti-cheat collapses on all 32** (permuted 2nd-order ≈ ±0.01). `M1_all_GO=True`, `M2_all_GO=True`, `M6_random_shard_collapses=True`. |
| **M3 — cross-bridge composition** (96 facts spanning all 32 bridges) | **top2=1.00, top1=0.99, signal/floor = 20.95×, margin 1482.5** — the true target retrieved over a noise floor of 2,048 concepts. `M3_all_GO=True`. (8-bridge fan-out was 20.02×; 3-bridge 16.85–23.96× — the cross-bridge SNR does NOT degrade at 4× the fan-out.) |
| **M7 — permuted mapping (fixed anti-cheat)** | top2=0.02, top1=0.00, **signal/floor 1.12× → COLLAPSES** (vs TRUE 20.95×). `M7_permuted_collapses=True` — the cross-bridge recall is a real learned link, not an artifact. |
| **M4 — no-confab moat** (learned familiarity gate ALONGSIDE host abstention, D=128) | agreement **1.000**, margin +0.2513, host-abstain/gate-accept **0**, floor-false-accepts **0**, lesion-collapses=True → **moat-intact=True**. The no-confab moat holds at 2,048 concepts. |
| **COMBINED** | `>>> COMBINED VERDICT: GO <<<` |

## What this retires
The build plan's deepest open Phase-1 question — *cross-bridge composition + the no-confab moat at 32-bridge fan-out* — is answered GO. The cross-bridge V-tag identity mechanism (validated at 3 and 8 bridges) **scales to the full 32-bridge production fan-out with no SNR degradation** (20.95× signal/floor over a 2,048-concept floor, ≥ the 8-bridge 20.02×), and the no-confab moat — the project's load-bearing anti-confabulation guarantee — stays zero-breach at 2,048 concepts. Combined with the route-A composer-architecture decision (`2026-06-13-phase1-composer-architecture-routeA-GO.md`) and the done production vocab spec (`g20_vocab_spec_2048.py`, commit `882c0e04`), the 32-bridge production system's *mechanism + architecture + vocabulary* are all in place; what remains is the build (corpus + train + the full conversational matrix at fan-out), the owner-gated multi-day commitment.

## Honest scope
- **Multi-seed confirmed (42/43/44).** Seed 42 was reported first (near-saturated: per-bridge generalization 0.975–1.000, M3 top2=1.00, zero moat breaches); the 43/44 confirmation (`_multibridge_fanout32_seed43_44.json`) returned **GO** — M3 top2=1.00 signal/floor 19.46×/23.24×, M7 collapses, moat intact zero false-accepts — so the cross-bridge composition + moat hold across all three seeds at 32-bridge fan-out.
- The corpus is the de-risk's controlled synthetic structure (a known `S_true`); the **production corpus must use meaningful within-cluster structure** — the corpus-source decision is resolved in `docs/plans/2026-06-13-phase1-production-corpus-source-design.md` (Option B curated semantic sub-taxonomy, NOT the arbitrary-synthetic reuse = a brain-based-only shortcut).
- This is the cross-bridge *fan-out* de-risk; the full integration (the conversational matrix on 32 real graded bridges with the real corpus) is the build's validation.

## Conclusion + next
The deepest Phase-1 risk is retired: cross-bridge composition + the no-confab moat hold at 32-bridge / 2,048-concept fan-out. Confirmed multi-seed (42/43/44 all GO). Next: the corpus-source de-risk (Option C, learned-from-real-experience, in flight) decides the build substrate, then the production build (corpus + the 32-cortex-bridge train + the per-bridge G1/G2 gates + the full conversational matrix at fan-out) is the owner-gated multi-day commitment. No `sim/` edits. No banking — the multi-seed GO is reported with the cross-bridge anti-cheat collapsing on every seed.
