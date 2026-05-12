# 🚧 P5 iter PP multi-seed NEGATIVE FINAL (1/4 BIDIR before abandonment)

**Date:** 2026-05-12
**Status:** NEGATIVE. 1/4 BIDIR after 4 seeds (42 PASS, 43/44/100 FAIL).
Seeds 101 and 102 abandoned at ~30+ min/seed due to GPU contention with
parallel bootstrap. Even with both passing, max is 3/6 — below iter AA's
4/6 toy-scale baseline. Strategic pivot to Step 1 (in-vivo binding).

## Multi-seed result

| Seed | apple margin | river margin | BIDIR | iter AA toy ref |
|---|---|---|---|---|
| 42 | +1 ✓ | +6 ✓ | **YES** | YES (+7, +31) |
| 43 | +3 ✓ | -10 ✗ | NO | YES (+5, +23) — REGRESSION |
| 44 | -11 ✗ | -15 ✗ | NO | NO (+4, -3) — also bad |
| 100 | -30 ✗ | +14 ✓ | NO | YES (+8, +10) — REGRESSION |
| 101 | (abandoned at 40+ min wall) | | — | NO |
| 102 | (abandoned) | | — | YES |

**1/4 BIDIR after first 4 seeds.** Even with both remaining seeds
passing (very unlikely given trajectory), max is 3/6 — below iter AA's
4/6 toy-scale baseline.

iter PP regresses from iter AA on seed 43 (was PASS, now FAIL) and
seed 100 (was PASS, now FAIL). The architectural pivot (sensory
grounding + lang_output FS) HURTS these seeds.

## Why iter PP doesn't scale

Per-seed structural pool variance dominates the discrimination signal
at biological scale. The FS cross-inhibition (winner-take-all at
output) only works for SMALL margins (seed 42: +1, +6 → sharpened to
correct winner). For LARGE structural biases (seed 100: -30), FS
pools can't overcome the recurrent excitation of the dominant pool.

Tier 1 motor binding succeeds at 6/6 BIDIR because:
- Single hop (lang_input → motor) — input drive directly competes
  with structural bias
- Strong topographic prior (5x weight ratio)
- Output-layer FS WTA where input drive is strongest

P5 has multi-hop chain (lang_input → wernicke → semantic → multimodal
hub → lang_output) where input signal gets DILUTED through each hop.
By the time the signal reaches lang_output_pool, random structural
variance from intermediate layers dominates.

## Architectural ceiling characterization (complete)

After 8 biological-scale iterations + 30+ toy-scale iterations
spanning May 11-12, 2026:

| Iter | Change | Multi-seed BIDIR |
|---|---|---|
| AA (toy ref) | per-concept pools | **4/6** |
| KK | + Tier 1 canon, bio scale | 0/seed_42 |
| LL | + scale only (weak) | 0/seed_42 |
| MM | + stronger topographic | 0/seed_42 |
| NN | + orthogonal codes | 0/seed_42 (flipped) |
| OO_visual | + sensory grounding (Cluster K v2) | 0/seed_42 (apple +23 but river flipped) |
| PP | + lang_output FS WTA | **1/4 partial** (seed 42 only) |

**iter AA's 4/6 toy-scale BIDIR is the architectural ceiling for the
per-concept pool design.** No parameter combination or architectural
addition tested has improved on this.

## Why this is the architectural ceiling

The per-concept pool design has a fundamental tension at biological
scale:
1. To get good single-trial discrimination, need strong dynamics
   (Tier 1 cortical canon)
2. Strong dynamics amplify per-seed random structural variance
3. Random variance creates pool dominance that overrides input signal
4. Mitigations (sensory grounding, FS WTA) help individual seeds but
   don't generalize

The Path G+ design doc acknowledged this:
> "Defeats some of the catalog G.13 architectural intent (one
>  Wernicke's area, not many). Compromise: it works [at toy scale]."

At biological scale, the compromise breaks down.

## Strategic recommendation (final)

**Ship iter AA at toy scale as the demonstrated P5 capability.**
4/6 multi-seed BIDIR for 2-concept abstract concept binding is real
and useful at the demonstration level.

**Pivot to Step 1 of realigned plan: in-vivo new-vocab binding.**
- Biology-grounded via McClelland 1995 CLS + Buzsáki 2015 SWR
- Tests with V0 vanilla, V_HIPPO_BIO, V_SCHEMA variants
- Already scaffolded at `research/runners/investigate_invivo_binding_fix.py`
- 4 novel-key test bindings (apple/river/mountain/forest → N/E/S/W)

If in-vivo binding works, the sim can grow vocabulary DURING
conversation rather than requiring pre-training. This is a more
impactful capability than further P5 toy-scale tuning.

## Code preserved

The iter OO_visual + iter PP architecture is fully implemented and
parameterized in text_minimal_isolation.py + validate_ventral_semantic.py
via CLI flags. The capability is available for future experimentation
at single-seed scale or with future architectural fixes:

```bash
python -m research.runners.validate_ventral_semantic --seed N \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --enable-lang-out-fs-pools --n-per-lang-out-fs-pool 60 \
    --apply-wernicke-topographic \
    --enable-visual-cortex --enable-multimodal-hub \
    --pair-visual-during-training \
    --n-recognition-trials 5 --inter-trial-rest-steps 100
```

iter PP seed 42 result (margins +1, +6) is the FIRST bidirectional
PASS at biological scale and demonstrates the sensory-grounding +
output-WTA pattern CAN work — just not robustly across seeds.

## Wall clock total

iter PP arc: 4 seeds × 6.7-7.3 min sequential + 2 abandoned = ~30 min
compute + 40+ min stalled by GPU contention. Total session compute on
P5 biological-scale arc (KK + LL + MM + NN + OO_visual + PP single +
PP multi-seed) = ~3 hr.

## Catalog faithful (preserved through arc)

- G.11 Hickok & Poeppel dual-stream ventral semantic ✓
- G.13 Wernicke's area ✓
- K.01 V1/V2/IT visual ventral (iter OO_visual+PP) ✓
- Lambon Ralph 2017 ATL hub-and-spoke (multimodal_hub) ✓
- Pulvermüller embodied semantics (paired training) ✓
- No motor-decoder cheats ✓
- No external LLM cheats ✓
- Cortical canon (Lefort 2009, Wang 2002) ✓

The architectural ceiling is not from cheating or shortcutting —
it's from the fundamental tension between strong recurrent dynamics
(needed for representations) and stable signal propagation through
multi-hop chains (needed for input-driven discrimination).
