# VRAM ceiling probe — practical hardware limits on RTX 3090 24 GB

**Date:** 2026-05-10 22:15 EDT
**Hardware:** RTX 3090 24 GB
**Trigger:** User question — "we know encoding needs scaling but we don't
know how much vocab we can actually fit at maximum VRAM."
**Status:** Phase A partial (2/4 probes); chain killed when A3 hit
prohibitive build times. Phase A data alone establishes the practical
ceiling.

---

## TL;DR

The practical ceiling on RTX 3090 24 GB is **much lower than the VRAM
ceiling.** Encoding can scale to n_lang=16384 cleanly, but at n_lang=32768
the bridge build cost balloons to **44 minutes per run** and inference
drops to **1.7 steps/sec** (14× slower than n_lang=16384). VRAM at 32K is
12 GB — half the available 24 GB — but the architecture is essentially
unusable for experimentation.

**Recommended practical ceiling: n_lang ≤ 16384.** This fits comfortably
in 5 GB VRAM at ~24 steps/sec and is the largest encoding-scaled arch
that runs in reasonable wall clock.

## Probe results

| Probe | n_lang | n_motor | Synapses | VRAM | Init time | Steps/sec | Practical? |
|-------|--------|---------|----------|------|-----------|-----------|------------|
| A1 | 16384 | 2000 | 110M | 5.0 GB | 5 min | **23.9** | ✅ **YES** |
| A2 | 32768 | 2000 | ~440M (est) | 12.2 GB | **44 min** | **1.7** | ❌ NO — too slow |
| A3 | 65536 | 2000 | ~1.8B (est) | ? | killed >30 min in | ? | not tested |

**A1 cleanly succeeds** at 5 GB VRAM and 24 steps/sec. Plenty of headroom.

**A2 technically succeeds** but with major caveats:
- Bridge construction takes 44 minutes (vs A1's 5 min)
- Per-step time is 572 ms (vs A1's 42 ms) — 14× slower
- Transient OOM during init (recovered)
- A full smoke run at this arch would take ~5+ hours instead of ~16 min

**A3 killed** — too slow to be useful. ~30+ min in construction, still allocating.

## Why the wall-clock cost explodes between A1 and A2

The cost of building + simulating the bridge scales with synapse count.

For the cross-region wiring at density=0.1, each `language_input → motor_X`
pathway has approximately `0.1 × n_lang × n_motor` synapses per action.
With ~30 pathways involved:

- A1 (16K × 2K): ~32M edges → ~10× more than 8-word baseline → still GPU-tractable
- A2 (32K × 2K): ~130M edges → bigger but should fit
- Reality: A2's reported synapse count would be in the 400M+ range across all pathways. The 12 GB VRAM is JUST the steady-state arrays; the bridge build allocates transient buffers that nearly hit 24 GB.

The build slowdown isn't just memory — it's the connectivity construction algorithm. Building 400M+ edges sequentially is the bottleneck, not GPU compute.

## Practical scaling recommendations

For RTX 3090 24 GB, the **safe + fast** working envelope is:

| Tier | n_lang | n_motor | Wall-clock smoke | Use case |
|------|--------|---------|------------------|----------|
| Small | 4096 | 1000 | ~30 min (8-word smoke) | Tier 1 chat, 4-word vocab |
| Medium | 4096 | 2000 | ~30 min (16-word smoke) | Tier 2.1, 12-word vocab |
| Large | 8192 | 2000 | ~16 min (64-word smoke, validated 3-seed) | 64-word vocab @ encoding-axis |
| **XL (validated)** | **16384** | **2000** | **~30 min** (predicted for 96/128 word) | **96-word+, headroom for retention** |
| XXL (impractical) | 32768 | 2000 | >5 hr (43 min build alone) | Avoid until faster build |

## Connecting back to the science: where's the encoding wall?

The encoding ceiling found earlier today:
- 64-word @ n_lang=8192: 3/3 GO unanimous (110% primary retention)
- 96-word @ n_lang=8192: PARTIAL (57% primary retention — below threshold)
- 256-word @ n_lang=16384: at chance (encoding overload)

The 96-word PARTIAL is the most interesting data point — primary
retention 57% but pre-silence primary 70% (better than 64-word's 48%).
So 96 words can BE BOUND but consolidation pathway weakens.

**Next experiment to try (single seed, ~30 min):** 96-word @ n_lang=16384
(the XL tier from the practical envelope). Does bigger encoding fix 96-word
retention? If yes, the XL tier is the new sustainable vocab tier.

If 96-word @ n_lang=16384 still fails retention: the encoding-vs-retention
tradeoff is bound by the consolidation pathway's signal-to-noise, not
the raw encoding capacity.

## What we'd need to actually push higher

To make n_lang=32768 practical, we'd need:
1. **Faster bridge construction** — the 44 min build cost is the
   killer, not the steady-state VRAM. Profile + optimize the
   `inject_explicit_wiring` path.
2. **Cloud GPU** — H100 80 GB has more headroom AND ~5× more compute,
   making per-step time at n_lang=32768 closer to ~120 ms instead of
   572 ms (or 8 steps/sec instead of 1.7).

At cloud H100:
- n_lang=16384: predicted 0.5 GB VRAM, 120+ steps/sec → trivial
- n_lang=32768: predicted ~3-6 GB VRAM, ~30 steps/sec → fully usable
- n_lang=65536: predicted ~15-20 GB VRAM, ~10 steps/sec → tight but usable
- n_lang=131072: predicted ~40-50 GB VRAM, ~5 steps/sec → fits but slow

Cloud unlocks the next ~2 vocab tiers.

## What probes we'd still want (in the future, with optimization)

Phase B (motor scaling, never ran): n_lang=4096 + n_motor=4000/8000/16000/32000.
Phase C (combined): n_lang=16384 × n_motor=4000-8000.
Phase D (max vocab at max encoding): 128/256-word at n_lang=16384.

These can be revisited if/when:
- Bridge construction is faster (profile-driven optimization)
- We have cloud H100 access
- Specific science questions require it

## Provenance

- A1: `research/findings/raw/perf/vram_ceiling/A1_lang16k_motor2k.json`
- A2: `research/findings/raw/perf/vram_ceiling/A2_lang32k_motor2k.json`
- Probe script: `scripts/probe_vram_ceiling.sh` (killed at A3)
- Earlier encoding ceiling: `2026-05-10-encoding-axis-64word-3SEED-GO.md`
- 96-word PARTIAL: `2026-05-10-encoding-axis-64word-GO-35x-faster.md` + this run's findings (`raw/g11_bg/g11_seed42_consolidation_synonym_96word_encoding_axis_smoke.json`)
- 256-word killed: discussion + per-trial log only (smoke was killed before result saved)
