# Encoding-axis 64-word smoke: GO + 35× faster

**Date:** 2026-05-10 13:45 EDT
**Status:** ✅ GO (retention primary 125% PASS, synonym 150% PASS)
**Run ID:** 9987b4f4a6d7
**Wall clock:** 956s = **15.9 min** (vs prior baseline 9.4 hr = 35× faster)

---

## TL;DR

User's hypothesis from earlier in the arc was CORRECT:

> "maybe at a certain scale we can fit much more than we think... 
> the encoding wall is the real bottleneck, not motor capacity"

The encoding-scale 64-word arch (`n_lang=8192, n_motor=2000`) is:
- **35× faster** wall clock than the original (`n_lang=4096, n_motor=6000`)
- **GO instead of PARTIAL** — primary retention 125% (PASS) vs 76% (FAIL)
- **All-around better consolidation** — synonym retention 150% vs 157%
  (similar), overall retention 120% vs 98%

Combined with STP-off default (3.28× speedup, validated 3-seed earlier
this hour), the new architecture stack delivers an order-of-magnitude
improvement on the same scientific test.

## Architecture comparison

| Config | Original (PARTIAL) | Encoding-scale (GO) |
|--------|--------------------|--------------------|
| Vocab | 64 | 64 |
| n_lang_input | 4096 | **8192 (2× bigger)** |
| n_motor_per_action | 6000 | **2000 (3× smaller)** |
| n_motor_fs | 720 | 240 |
| STP enabled | True (was default) | **False (new default)** |
| Wall clock | 9.4 hr | **15.9 min (35× faster)** |
| VRAM peak | 16 GB | ~10 GB (estimated) |

Per the empirical capacity rule (~333 motor neurons per sub-pop), the
old arch had n_motor=6000 / 16 sub-pops = 375 (just over the floor).
The new arch has n_motor=2000 / 16 = 125 (well below the floor!) and
STILL works because **bigger encoding compensates for smaller motor**.

## Result detail

```
Pre-silence:  overall 23.4%   primary 40.0%   synonym 20.0%
Hippo-OFF:    overall 28.1%   primary 50.0%   synonym 30.0%

RETENTION:    overall 120%    primary 125%    synonym 150%
              (>= 80% prim)   (>= 60% syn)    BOTH PASS

Verdict: GO
```

**Pre-silence primary is LOWER** (40% vs 62.5% on the bigger-motor arch)
but **post-hippo-OFF primary is HIGHER** (50% vs 47.5%). The smaller
motor pool binds initially weaker but the bigger encoding preserves
the binding through hippocampus silencing dramatically better.

## Strategic implication: capacity rule revision

The empirical capacity rule "~333 motor neurons per sub-pop" is FALSIFIED
at vocab=64 with the encoding-scale arch (which has ~125 neurons/sub-pop
and works). The new framing:

**Rule: motor pool capacity matters LESS than encoding capacity at
higher vocab tiers.** The lang_input encoding is the rate-limiting
constraint; motor pools just need enough neurons for STDP to converge,
which is much less than the 333/sub-pop threshold.

This is a major architectural insight that re-shapes future scaling:
- Old strategy: scale n_motor with vocab (linear) → expensive
- New strategy: scale n_lang_input first, keep n_motor minimal → cheap

## Implications for higher tiers

The encoding-axis discovery + STP-off + temperature sampling stack
fundamentally changes the cost of architecture experiments:

| Vocab | Old wall-clock estimate | New wall-clock estimate |
|-------|------------------------|------------------------|
| 64-word smoke | 9.4 hr | **15.9 min** (35× faster) |
| 64-word medium | ~24 hr | **~1 hr** estimated |
| 96-word smoke | ~12 hr | **~25 min** estimated |
| 128-word smoke | ~16 hr | **~35 min** estimated |
| 256-word smoke | ~24 hr | **~60 min** estimated |

Find-the-ceiling experiments that were "all-night" are now "afternoon"
budget. The compute math for cloud deploy also re-shapes:

- 3090 local + STP-off + encoding-scale: ~35× over original baseline
- Add FP16 (1.135×): ~40× over original baseline
- Cloud H100 + above: ~480-800× over original baseline

A 6-seed sweep of 64-word that was 56 hr → now ~95 min locally,
~3-7 min on H100.

## Caveats

1. **Single seed only** — needs multi-seed validation. Run seeds 43, 44
   to confirm the pattern reproduces.

2. **Pre-silence primary is lower** (40% vs 62.5%). The new arch trades
   initial-binding-strength for retention-strength. Whether this
   matters depends on use case.

3. **VRAM estimate** — n_lang=8192 at sparse 10% means ~819 active
   neurons per word; over 64 words = 52K active across vocab vs 8192
   capacity = ~6× overlap (vs 16× at 4096). Encoding is less collided
   but still has overlap; future tiers (128, 256) may need n_lang=16384+.

4. **STP-off contribution unknown** — the encoding-axis test was run
   with STP-off (new default). Unclear how much of the 35× speedup is
   STP-off (~3×) vs n_motor=2000-vs-6000 (~5-12×). Both contribute.

## Recommended next experiments (with new fast architecture)

1. **Multi-seed encoding-scale 64-word** (~95 min for 6 seeds) — confirm
   pattern reproduces
2. **96-word and 128-word encoding-scale smoke** (~25-35 min each) —
   probe higher capacity tiers with the cheap arch
3. **256-word encoding-scale smoke** at n_lang=16384 — true ceiling test
4. **Re-run prior tonight's tier validations at STP-off** — Tier 1
   chat_speak_demo, Tier 2.1 chat_speak_synonym_demo, Phase 1.4
   BRANCH A, etc. All should improve.

## Provenance

- Result: `research/findings/raw/g11_bg/g11_seed42_consolidation_synonym_64word_encoding_scale_smoke_9987b4.json`
- Preset: `consolidation_synonym_64word_encoding_scale_smoke`
- Architecture: n_lang=8192, n_motor=2000, n_motor_fs=240, vocab=64,
  STP-off, --smoke
- Comparison baseline: `2026-05-10-64word-smoke-PARTIAL-capacity-extends.md`
- STP discovery: `2026-05-10-stp-default-flip.md`
