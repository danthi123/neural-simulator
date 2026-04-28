# Cheat #5 v4 — Developmental Pretraining: NO-GO.

> **Status update 2026-04-28 (afternoon):** the original "closed by design" framing in this document was reframed later the same day. Cheat #5 is now treated as **ON HOLD pending biology buildout** rather than closed. See [post-v4 status doc](2026-04-28-cheat5-post-v4-reframe.md) for the updated framing. The v4 results below stand as-is — they were correct — but the conclusion drawn from them was too quick.

**Date:** 2026-04-28
**Design:** [docs/plans/2026-04-28-cheat5-v4-design.md](../../docs/plans/2026-04-28-cheat5-v4-design.md) (commit `e6ce0ce`)
**Plan:** [docs/plans/2026-04-28-cheat5-v4-implementation.md](../../docs/plans/2026-04-28-cheat5-v4-implementation.md)
**Aggregator:** [scripts/analyze_cheat5_v4.py](../../scripts/analyze_cheat5_v4.py)

## TL;DR

| Approach | Mean sum (n=3 or 6) | Verdict |
|---|---|---|
| flagship baseline (v3 lateral) | 4.26 ± 0.50 (n=6) | **GO** (shipped 2026-04-28 morning) |
| v3.1 cross-projections, adult thaw at step 1200 | 8.92 ± 2.44 (n=6) | NO-GO |
| **v4 developmental pretraining** | **11.34 ± 1.85 (n=3)** | **NO-GO** |

**Cheat #5 is now closed by design.** v3 lateral inhibition is the biology-grounded winner-take-all in our reduced model. Cross-projections — at any non-zero weight, regardless of training regime — corrupt the cascade. v4 confirms the pattern is robust across architectures.

## Tier 1 (wiring smoke) — PASS

Single seed, 1 goal × 1000 trials of pretraining + 1800 eval. Confirmed the v4 pipeline works end-to-end.

| Seed | Pretraining cross weights | P0 (goal=(6,6)) | P1 (goal=(1,6)) | Sum |
|---|---|---|---|---|
| 42 | mean=10.935 std=0.499 (no NaN) | 3.85 | 3.74 | 7.59 |

Wall-clock: 598s. Sum 7.59 isn't meaningful for the decision matrix — Tier 1 is a wiring check. Useful signal: cross-projections grew from 0 to mean 10.9 during pretraining (STDP+reward is exercising the cross-projection synapses as designed).

## Tier 2 (signal check) — NO-GO, decisive

3 seeds, 5 goals × 1000 trials = 5K pretraining + 1800 eval each. ~60 minutes per seed at 3-concurrent.

| Seed | Pretraining cross weights | P0 finalQ | P1 finalQ | Sum |
|---|---|---|---|---|
| 42 | mean=11.093 std=0.632 | 4.63 | 5.18 | 9.81 |
| 43 | mean=10.931 std=0.634 | 5.60 | **7.80** | **13.40** |
| 44 | mean=11.003 std=0.518 | 4.41 | 6.39 | 10.80 |
| **mean** | — | **4.88** | **6.46** | **11.34 ± 1.85** |

**All 3 seeds individually exceed the > 6.0 NO-GO threshold.** The decision matrix from the design doc:

| Eval-phase mean sum | Verdict | Action |
|---|---|---|
| ≤ 4.1 | GO | propagate, close cheat #5 |
| 4.1–4.5 | GO MARGINAL | document closure-without-improvement |
| 4.5–6.0 | PARTIAL | try longer pretraining or more goals |
| > 6.0 | **NO-GO v4** | **pivot to last-resort closure-by-design** ← we are here |

## Tier 3 — skipped

The plan called for Tier 3 (6 seeds × 30K pretraining, ~14h overnight) only if Tier 2 showed ≥ partial signal (≤ 4.5 mean). Tier 2 came in at 11.34 with all 3 seeds unanimous past the NO-GO line. Running Tier 3 would waste 14h on a known-bad outcome. Per the plan: "If v4 also fails: acknowledge cheat #5 is closed by design."

## Why v4 fails

The signal is consistent across all attempts:

1. **v1/v2** (cross-projections, adult-only, multiple thaw schedules) — phase-2 readaptation breaks: 3-seed avg 8.40.
2. **v3.1** (cross-projections + lateral inhibition, adult thaw at step 1200) — same pattern, 6-seed 8.92, P1=6.35 = 2.5× P0.
3. **v4** (cross-projections developed during a 5K-trial critical period, then frozen for eval) — *worse*, 3-seed 11.34, both phases bad.

The pretraining itself worked: cross-weights grew from 0 to mean ~11.0 with low variance across seeds (std ~0.5–0.6). Pretraining IS shaping the cross-projection synapses under varied experience. But the resulting connectivity, when frozen and exposed to the eval task, **degrades performance** — even Phase 0 (the initial goal acquisition that was always easy) drops from ~2 to ~5.

Two interpretations, in increasing strength of evidence:

1. **Cross-projection refinement requires structural plasticity, not just weight plasticity.** Real BG anatomy is not 4×4 fully connected — it's sparse and heterogeneous, refined by axon pruning and synaptic stabilization during development. Our model has the connectivity hard-coded; only weights move. Maybe weight-only refinement can never produce useful cross-action structure.
2. **Cross-projections are off-axis at this level of abstraction.** Our reduced model already achieves winner-take-all action selection through (a) v3 MSN cross-pool lateral inhibition + (b) per-action argmax readout. Cross-projections add a noise channel that the mature cascade can't cleanly suppress, regardless of how they were trained. Same-action-only IS the functional equivalent of biological winner-take-all in this substrate.

Hypothesis (2) is more parsimonious given three independent failed attempts and is the working interpretation going forward.

## Closure: cheat #5 closed by design

Per the design doc's "If v4 also fails" plan:

> Last-resort plan: acknowledge cheat #5 is closed *by design* — same-action-only is the biological winner-take-all in our reduced model, with cross-projection development happening implicitly via the architecture. Document explicitly:
> - Real BG anatomically dense, functionally same-action-dominant
> - Our model: same-action-only structurally, equivalent functional behavior
> - Closure rationale: identical functional outcome, simpler substrate
>
> This isn't a punt — it's a principled choice given the simulator's level of abstraction. v3 lateral inhibition + same-action structure ≈ functional equivalent of real BG's anatomically-dense + winner-take-all.

**Decision: cheat #5 is closed.** v3 lateral inhibition (`--bg-lateral-inhibition`, shipped in flagship recommended config 2026-04-28 morning) IS the biology-grounded winner-take-all. The flagship eval result of 4.26 ± 0.50 (n=6, no regression) is the closing data point.

Cross-projections (`--bg-cross-projections`, with or without `--developmental-pretraining`) remain opt-in for future experiments — e.g., if someone adds structural plasticity or wants to test other connectivity refinement mechanisms — but are NOT recommended for any current flagship configuration.

## What ships from this finding

- **No code reverts.** The v4 implementation (`_run_pretraining_phase`, `--developmental-pretraining`, conflict-flag check, warning) remains in the codebase as opt-in infrastructure. It works; it's just not useful for cheat #5 closure. Could be repurposed for pretraining other pathways.
- **CLAUDE.md updated** to mark cheat #5 fully closed (v3 GO + v4 NO-GO).
- **Recommended flagship config unchanged** from this morning's update (already includes `--bg-lateral-inhibition`).
- **No follow-up plan needed** for cheat #5 itself. The next research priority moves to one of:
  - Scaling (16×16 grid, larger BG)
  - Replay (NREM/REM cycles)
  - Multi-modal sensory integration
  - Cerebellum / fine motor control
  - Other items in `project_next_priorities.md`

## Files

- Tier 1 result: `research/findings/raw/g11_bg/g11_seed42_flagship_6bcecf.json`
- Tier 2 results: `research/findings/raw/g11_bg/g11_seed{42,43,44}_flagship_{1b94af,c773ef,8ce983}.json`
- Aggregator: `scripts/analyze_cheat5_v4.py`
- Tier 3 launcher (unused): `scripts/launch_cheat5_v4_tier3.sh` (kept for reference; would only fire if a future variant needs full validation)

## Updates propagated

- [x] CLAUDE.md "Cheat #5 progress (2026-04-28)" — v4 NO-GO row added; cheat #5 marked closed
- [x] docs/SCIENCE_ROADMAP.md §4.7 — v4 row appended, cheat #5 status flipped
- [x] research/findings/INDEX.md — link added
- [x] CHANGELOG.md — v4 NO-GO entry under 2026-04-28
- [x] Memory: `project_cheat5_v3_results.md` updated with v4 outcome
