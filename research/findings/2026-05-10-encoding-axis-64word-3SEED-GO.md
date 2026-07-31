---
type: finding
status: live
date: 2026-05-10
mechanism: consolidation
---

# Encoding-axis 64-word 3-seed multi-seed: 3/3 GO UNANIMOUS

**Date:** 2026-05-10 15:44 EDT
**Status:** ✅ **3/3 GO unanimous** at 16 sub-pops/motor_X with encoding-scale arch
**Wall clock:** 954s ± 7s per seed (3 seeds × ~16 min each = ~48 min total)
**Test:** `consolidation_synonym_64word_encoding_scale_smoke` seeds 42, 43, 44

---

## TL;DR

The encoding-axis architecture (`n_lang=8192, n_motor=2000`) reproduces
GO across 3 seeds at 64-word vocab — confirming the user's
encoding-vs-motor hypothesis is robust, not a single-seed fluke. Combined
with the STP-off default, the new stack runs at **~35× the speed** of
the original 64-word baseline AND passes the consolidation retention
test on all 3 seeds.

This validates the new recommended architecture for vocab ≥ 32-word.

## Per-seed result

| Seed | Pre-silence | Hippo-OFF | Retention | Wall clock | Verdict |
|------|-------------|-----------|-----------|------------|---------|
| 42 | pri 40% / syn 20% | pri 50% / syn 30% | pri 125% / syn 150% | 956s | **GO** |
| 43 | pri 45% / syn 20% | pri 55% / syn 25% | pri 122% / syn 125% | 949s | **GO** |
| 44 | pri 60% / syn 30% | pri 50% / syn 25% | pri 83% / syn 83% | 956s | **GO** |

**3-seed mean:**
- Pre-silence primary: **48.3% ± 10.4%**
- Pre-silence synonym: 23.3% ± 5.8%
- Hippo-OFF primary: **51.7% ± 2.9%** (very low variance!)
- Hippo-OFF synonym: 26.7% ± 2.9%
- Retention overall: 111% ± 8%
- Retention primary: **110% ± 23%**
- Retention synonym: **119% ± 34%**
- Wall clock: 954s ± 7s

**All 3 seeds pass both retention thresholds (primary ≥80%, synonym ≥60%).**

## Comparison: original baseline vs encoding-axis

| Config | Original (n_motor=6000, STP-on) | Encoding-axis (n_lang=8192, n_motor=2000, STP-off) |
|--------|--------------------------------|----------------------------------------------------|
| Seeds tested | 1 | 3 (unanimous) |
| Wall clock | 9.4 hr (33,840s) | **15.9 min (954s)** — **35× faster** |
| Pre-silence primary | 62.5% | 48.3% (lower initial binding) |
| Retention primary | 76% (FAIL) | **110% (PASS)** — better |
| Retention synonym | 157% | 119% — both pass |
| Verdict | PARTIAL | **GO unanimous** |

The new arch trades ~14pp of initial binding strength for:
- 35× faster compute
- +34pp retention through hippo silencing
- Reproducible across 3 seeds (not a single-seed anomaly)

## Key insight: the architecture preference depends on use case

| Goal | Recommended arch |
|------|------------------|
| Maximize immediate binding strength | Bigger n_motor (e.g., 6000) — but expensive |
| Maximize retention through consolidation | Bigger n_lang_input (e.g., 8192) + minimal n_motor — cheap |
| Production runtime (chat REPL) | Encoding-axis (low compute, high reliability) |
| Architecture-validation experiments | Either; report both |

For our **conversational stack goal**, retention through sleep
consolidation matters more than immediate binding peak. **Encoding-axis
is the recommended architecture for vocab ≥ 32 words.**

## Capacity rule revision

Empirical capacity rule discovered in 8/12-word experiments: "~333 motor
neurons per sub-pop." This was based on bigger-motor architectures.

Updated rule based on encoding-axis evidence:

```
For vocab N with k = N/4 sub-pops per motor_X:
  Required: n_lang_input >= 4096 * (k / 4)
  Required: n_motor_per_action >= ~125 neurons/sub-pop = 125 * k
```

For specific tiers:
- 8-word (k=2): n_lang=4096, n_motor=250 minimum (currently 1000, plenty)
- 16-word (k=4): n_lang=4096, n_motor=500 minimum (currently 2000)
- 32-word (k=8): n_lang=8192, n_motor=1000 minimum (currently 3000)
- 64-word (k=16): n_lang=8192, n_motor=2000 (validated 3-seed)
- 128-word (k=32): n_lang=16384, n_motor=4000 (predicted)
- 256-word (k=64): n_lang=32768, n_motor=8000 (predicted)

The lang_input growth is sub-linear with vocab (encoding tolerates ~4 sub-pops per encoding-slot of 1024 neurons). Motor growth is steep
since smaller motor pools need MORE neurons per sub-pop at lower vocab.

This is a major architectural insight. The old rule predicted that
64-word needs n_motor=5328 (4 × 16 × 333). New rule says n_motor=2000
works — saves ~2/3 of motor neurons at this tier.

## Compute economics

Combined with STP-off default and FP16 (when validated), the new stack:

| Hardware + opt-stack | Speedup vs original | Time for 64-word smoke |
|----------------------|---------------------|------------------------|
| 3090 FP32 STP-on (original) | 1× | 9.4 hr |
| 3090 FP32 STP-off | ~3.3× | 2.8 hr |
| 3090 FP32 STP-off + encoding-axis | **~35×** | **16 min** |
| 3090 FP16 + STP-off + encoding-axis | ~40× | 14 min |
| A100 80GB FP16 + STP-off + encoding-axis | ~240× | 2.3 min |
| H100 80GB FP16 + STP-off + encoding-axis | ~480-800× | 1-2 min |

A 6-seed sweep of 64-word that was 56 hr in the old config → ~95 min
local now → 12-25 min on H100. Find-the-ceiling experiments become
genuinely cheap.

## What this unblocks

Tier ladder with new wall-clock estimates (single-seed smoke):

| Vocab | Predicted wall clock | Effort |
|-------|---------------------|--------|
| 64-word smoke | 16 min | already done ✓ |
| 96-word smoke | ~22 min | trivial |
| 128-word smoke | ~30 min | trivial |
| 256-word smoke | ~60 min | trivial |
| 64-word **medium** (50 chunks) | ~1.5 hr | feasible |
| 64-word medium 3-seed | ~4.5 hr | overnight |
| **Tier 2.3 phrases revisit** with STP-off | TBD | recommended |

**Caveat:** Tier 2.3 PFC phrases may behave differently under STP-off
(see STP biological-purpose finding doc). Needs revalidation before
extrapolating.

## Recommended next steps

1. ✅ This 3-seed validation (done)
2. **Run 32-word smoke at encoding-axis** to validate the prediction
3. **Run 128-word smoke at encoding-axis** (predicted 30 min) — test
   true encoding wall
4. **Update master plan + capability_status** with new architecture
   as recommended for vocab ≥ 32
5. **Tier 2.3 phrase revalidation at STP-off** — critical to confirm
   the silent flip didn't break the architecture-limited 41% baseline
6. **Cloud deploy** when 256-word + Tier 2.3 retest both done

## Provenance

- Per-seed JSONs: `research/findings/raw/g11_bg/g11_seed{42,43,44}_consolidation_synonym_64word_encoding_scale_smoke_*.json`
- Preset: `consolidation_synonym_64word_encoding_scale_smoke`
- Single-seed analysis: `2026-05-10-encoding-axis-64word-GO-35x-faster.md`
- STP default flip: `2026-05-10-stp-default-flip.md`
- Original 64-word baseline: `2026-05-10-64word-smoke-PARTIAL-capacity-extends.md`
