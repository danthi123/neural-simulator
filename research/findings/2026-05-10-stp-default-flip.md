---
type: finding
status: contributing
date: 2026-05-10
mechanism: stp
---

# STP DEFAULT FLIP — 3.28× speedup AND higher accuracy

**Date:** 2026-05-10 12:52 EDT
**Status:** ✅ DEFAULT FLIPPED (after 3-seed validation)
**Trigger:** perf_benchmark discovery — STP was 57% of inner-loop step time
**Result:** STP-off is BOTH 3.28× faster AND more accurate

---

## TL;DR

`cfg.enable_short_term_plasticity` was True by default in the bridge,
silently consumed 57% of every inner-loop step time, AND was actively
HURTING binding accuracy. Default flipped to False.

3-seed validation (chat_speak_synonym_demo, seeds 42/43/44):

| Metric | With STP (6-seed mean) | Without STP (3-seed mean) | Δ |
|--------|------------------------|---------------------------|---|
| Wall clock | 1660s (27.7 min) | 506s (8.4 min) | **3.28× faster** |
| W2A regression | 35.4% | 52.1% | +16.7pp |
| A2W any-synonym | 87.5% | 100% | +12.5pp |
| A2W primary | 87.5% | 100% | **PERFECT** |

Per-seed:
- Seed 42 (NOSTP): W2A 50.0%, A2W 100% (vs 25.0%/50.0% w/STP)
- Seed 43 (NOSTP): W2A 50.0%, A2W 100% (vs 50.0%/75.0% w/STP)
- Seed 44 (NOSTP): W2A 56.2%, A2W 100% (vs 25.0%/100% w/STP)

3/3 seeds: **A2W primary 100%** unanimous.

## Why STP was hurting

STP (Tsodyks-Markram synaptic depression) models millisecond-scale
synaptic fatigue: each spike depresses the pre-synaptic resource pool,
which recovers over τ_d ≈ 200ms. For neuroscience modeling of fast
oscillations or paired-pulse facilitation, this is biologically real.

**For language→motor binding, this is a bug, not a feature.** During
embodied-Hebbian training, the same word fires repeatedly during the
stim_steps_per_event window (50 steps × ~10ms = 500ms simulated time).
STP causes each subsequent spike in that burst to be WEAKER than the
last, depressing the binding it's trying to form. Cleaner binding
without STP.

## What was changed

Default flipped to False in:
1. `bio_three_factor.run_three_factor` — the central training loop
   used by all chat demos. `enable_stp: bool = False`.
2. `chat_synonym_demo.train_chat_bridge` — Tier 2.1 8-word training.
3. `chat_speak_synonym_demo.run_chat_speak_synonym_demo` — Tier 2.1
   :speak production-side runner.
4. `consolidation_synonym_trainer` — sets
   `cfg.enable_short_term_plasticity = False` explicitly.

Users who want to recover prior behavior can pass `enable_stp=True`
(programmatic) or use the `cfg.enable_short_term_plasticity = True`
override.

## Performance impact

This is the largest single optimization win of the optimization arc:

| Optimization | Speedup | Accuracy |
|--------------|---------|----------|
| FP16 (cp_eligibility_trace) | 1.135× | unchanged |
| Freeze-plasticity-during-reset | 1.0× (microbench) | TBD |
| **STP disabled** | **3.28×** | **+12-25pp** |

Cumulative best case: STP-off + FP16 = ~3.7× local speedup. Future
runs of chat_speak_synonym_demo go from 28 min/seed to 8.5 min/seed.
6-seed multi-seed: 168 min → 51 min.

For 64-word smoke (which took 9.4 hr total tonight), STP-off alone
would have saved ~5 hours (training portion = 4 hr → 1.2 hr; eval
portion ~5.4 hr → 1.6 hr).

## Implications for cloud deploy

Per the optimization audit, cloud GPU multipliers are:
- A100 80GB FP16 + opts: ~6-8× over current baseline
- H100 80GB FP16 + opts: ~12-20× over current baseline

With STP-off + FP16 baseline now at ~3.7× local 3090 FP32:
- A100 cloud: ~22-30× total speedup
- H100 cloud: ~44-74× total speedup

A find-the-ceiling sweep that took 9 hr locally would take ~7-15 min on H100.

## Implications for prior validations

**All prior multi-seed multi-tier results were measured WITH STP.**
They are still valid as documented; the architecture *can* bind
language→motor at those rates with STP enabled. But: **STP-off is
strictly better for this workload**, so the canonical / production
defaults should use it.

Tonight's Tier 2.1 8-word :speak 6-seed at 87.5% A2W with STP:
predicted to push to ~95-100% A2W with STP-off (per single-seed
deltas).

Re-validation suggested:
- Tier 1 chat_speak_demo 6-seed at STP-off (~50 min vs prior ~3 hr)
- Tier 2.1 8-word :speak 6-seed at STP-off (~50 min)
- Phase 1.4 BRANCH A multi-seed retention at STP-off
- 64-word smoke at STP-off (~3 hr vs prior 9.4 hr)

## Caveats

1. **Other workloads may need STP.** The g11_bg navigation runner uses
   STP for biological gamma oscillations + paired-pulse facilitation.
   Its config is independent — not changed by this default flip.

2. **Smaller arches not validated.** All 3 NOSTP validations are at
   Tier 2.1 v4 8-word arch (n_lang=4096, n_motor=1000). Defaults flipped
   for chat training paths only; navigation arches keep their existing
   configs.

3. **Synonym retention behavior at 64-word may change.** Tonight's 64-word
   smoke had retention 76% (FAIL <80%) with STP. STP-off may shift this
   either direction — needs re-validation.

## Provenance

- Per-seed JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_chat_speak_synonym_demo_NOSTP.json`
- Perf benchmark: `research/findings/2026-05-10-perf-benchmark-results-stp-discovery.md`
- Default flip commit: this commit, files listed above

## Recommended next steps

1. Re-run Tier 2.1 8-word 6-seed multi-seed with STP-off (~50 min)
2. Re-run 16-word smoke at STP-off (~10 min)
3. Re-run 64-word smoke at STP-off (~3 hr) — should retest the 76%
   primary retention number
4. Update Tier 2.1 BREAKTHROUGH paper / capability_status with new
   ceilings (likely 95-100% A2W primary)
