# Cheat #5 v3 + v3.1 results — v3 GO, v3.1 NO-GO (pivot to v4)

**Date:** 2026-04-28
**Plan:** [2026-04-28-cheat5-v3-lateral-inhibition.md](../../docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md)
**Aggregator:** [scripts/analyze_cheat5_v3.py](../../scripts/analyze_cheat5_v3.py)

## TL;DR

| Variant | Mean sum (n=6) | P0 | P1 | Verdict |
|---|---|---|---|---|
| flagship baseline | 4.08 (prior 6-seed) | — | — | GO (reference) |
| **v3lateral** | **4.26 ± 0.50** | 2.35 | **1.91** | **GO — no regression** |
| **v3.1cross** | **8.92 ± 2.44** | 2.58 | **6.35** | **NO-GO — pivot to v4 (developmental phase)** |

**Conclusion:**
- v3 (MSN cross-pool lateral inhibition) is biology-grounded, harmless to flagship performance, and **ships as a permanent default**.
- v3.1 (cross-projections layered on v3) **fails the same way as v2**: phase-2 readaptation breaks. Lateral inhibition wasn't the missing piece. The deeper explanation — that cross-projection refinement is a **developmental phenomenon, not adult learning** — is back on the table.

## Per-seed data

### v3lateral (`--bg-lateral-inhibition` only, no cross-projections)

| Seed | Sum | P0 (goal=(6,6), step 0–300) | P1 (goal=(1,6), step 300–1800) |
|---|---|---|---|
| 42 | 3.41 | 1.79 | 1.63 |
| 43 | 4.06 | 1.87 | 2.19 |
| 44 | 4.55 | 2.95 | 1.61 |
| 100 | 4.42 | 2.35 | 2.08 |
| 101 | 4.86 | 2.71 | 2.15 |
| 102 | 4.28 | 2.47 | 1.81 |
| **mean** | **4.26 ± 0.50** | **2.35** | **1.91** |

P1 < P0: agent readapts faster than initial learning. No regression vs baseline 4.08 (p ≈ NS). Lateral inhibition is GO.

### v3.1cross (`--bg-lateral-inhibition --bg-cross-projections --cross-projection-weight 0.0 --bg-cross-thaw-step 1200 --bg-cross-phase3-gain 0.5`)

| Seed | Sum | P0 (steps 0–300) | P1 (steps 300–1800) |
|---|---|---|---|
| 42 | 7.53 | 2.36 | 5.17 |
| 43 | 12.89 | 2.41 | **10.48** |
| 44 | 10.99 | 3.29 | 7.70 |
| 100 | 7.33 | 2.97 | 4.36 |
| 101 | 6.85 | 2.29 | 4.56 |
| 102 | 7.93 | 2.12 | 5.81 |
| **mean** | **8.92 ± 2.44** | **2.58** | **6.35** |

P0 is fine (~2.58, comparable to v3lateral). P1 is **2.5× P0** — phase 2 readaptation gets dramatically worse than phase 1. The cross-projections, even with zero-init + 1200-step thaw + 0.5 gain, corrupt the cascade once the goal moves.

This matches the v2 NO-GO pattern. v3 added MSN lateral inhibition specifically to suppress the cross-talk hypothesized to drive that failure. **Lateral inhibition didn't fix it.**

## Why v3.1 fails (interpretation)

Lateral inhibition between MSN action pools is fast and selective — exactly what biology has — and it's clearly working in v3lateral (P1 better than P0). But adding cross-projections still breaks phase 2.

Two interpretations, in order of plausibility:

1. **Cross-projections are NOT an adult-learning phenomenon.** STDP+reward on a converged cascade can't shape useful cross-action structure from random initial conditions, even with all the biology pieces in place. The connectivity needs to be shaped by *experience-dependent pruning during developmental critical periods*, with exposure to many tasks. This is what the v4 plan addresses.

2. **The thaw schedule is wrong.** Maybe phase 2 starts too soon after thaw (step 1200 → first goal change at step 1500 in some configs), so STDP only has 300 steps to refine. Slower thaw + earlier introduction (e.g., thaw at step 600, longer plastic window) might give more shaping time. But this is a tweak, not a fix — and we've already tried slower-gain variants in v2.

Going with **(1)**. v4 is up.

## What ships from this batch

- **v3lateral becomes a permanent default.** Add `--bg-lateral-inhibition` to the recommended flagship config (no behavior change, just better biology). Will update CLAUDE.md "Recommended configuration" section.
- **v3.1cross stays opt-in.** It's a known NEGATIVE under current architecture, kept reproducible behind `--bg-cross-projections`. Documented as such.
- **v4 plan activates.** Pre-training developmental phase is the next attempt at cheat #5 closure.

## Files

- Result JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44,100,101,102}_flagship_<runid>.json` (sidecar `.cmd.json` distinguishes v3 vs v3.1 via extra_args).
- Aggregator: [scripts/analyze_cheat5_v3.py](../../scripts/analyze_cheat5_v3.py)
- Per-run logs: `webapp/runtime/run_<runid>.log`

## Updates needed

- [ ] CLAUDE.md: add `--bg-lateral-inhibition` to recommended config; update cheat-5 status from "NEGATIVE — kept opt-in" to "v3 closed (no regression GO), v3.1 NO-GO, v4 next"
- [ ] docs/plans/2026-04-28-cheat5-v3-lateral-inhibition.md: mark Tasks 2–3 done; flag v4 as the active path
- [ ] docs/SCIENCE_ROADMAP.md §4.7: append v3 result row
- [ ] Memory: update `project_phase_c_resolved.md` or add a new memory pointing at the v4 pivot
