# P5 iter C seed 42 FAIL — scale doesn't help; size NOT the bottleneck

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** Honest report. Iter C scaled wernicke 200→400 +
semantic_cortex 500→1000 on top of iter B fixes. Result WORSE
than iter A/B — scaling without addressing attractor dynamics
hurts SNR.

## Comparison (seed 42)

| Metric | Iter A | Iter B | **Iter C (scaled)** | Target |
|---|---|---|---|---|
| n_wernicke | 200 | 200 | **400 (2x)** | — |
| n_semantic_cortex | 500 | 500 | **1000 (2x)** | — |
| n_synapses | 881K | 881K | **1.37M (1.6x)** | — |
| apple_self cosine | 0.227 | 0.226 | **0.207** ↓ | > 0.5 |
| apple_river cosine | 0.174 | 0.186 | **0.198** | < 0.4 |
| Naming ratio | 1.08x | 1.08x | **0.99x** ↓ | > 1.3x |
| Wall clock | 295s | 306s | **339s** | — |
| Verdict | FAIL | FAIL | **FAIL** | — |

The scale-up REDUCED same-concept signal (0.227 → 0.207) and
INCREASED cross-concept (0.174 → 0.198). Naming dropped below
baseline (1.08x → 0.99x). **Conclusively rules out the wernicke
bottleneck hypothesis.**

## Diagnostic conclusion: NOT a size issue

The scaling experiment is a clean control: same architecture,
same training, same test methodology, just bigger pools. The
fact that 2x larger arch produced WORSE numbers means:

- The bottleneck is NOT n_wernicke=200
- The bottleneck is NOT n_semantic_cortex=500
- More neurons = more noise without proper attractor formation

This is informative. It points clearly at the dynamics
hypothesis (iter D).

## 4-fail iron law triggered

Per superpowers:systematic-debugging Phase 4.5: "if 3+ fixes fail,
question architecture vs continuing to fix symptoms." We are at
attempt 4:

1. Original (default params, 100 events). FAIL multi-seed.
2. Iter A (engram-tag + 300 events). FAIL seed 42 (~0.227).
3. Iter B (strict two-stage + lang drive replay). FAIL seed 42
   (~0.226 — basically identical to A).
4. Iter C (scale-up 2x). FAIL seed 42 (~0.207 — WORSE).

The consistent ~0.20-0.23 ceiling AND the fact that scale hurts
confirms this is an architectural property: semantic_cortex
doesn't form stable point attractors with the current parameter
choices.

## Iter D launched (attractor tuning)

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 300 --n-replay-cycles 30 \
    --strict-two-stage --drive-lang-during-replay \
    --semantic-cortex-recurrent-density 0.25 \
    --semantic-cortex-recurrent-weight 4.0 \
    --drive-steps 300 \
    --out research/findings/raw/g11_bg/p5_iterD_seed42.json
```

Three biology-grounded changes per Wang 2002 / Patterson 2007 /
Lambon Ralph ATL hub:
- recurrent_density 0.10 → 0.25 (real cortex 20-30% recurrent)
- recurrent_weight 1.0 → 4.0 (matches lang→wernicke→semantic
  feedforward of 3.0/4.0; previously recurrence was 3-4x
  WEAKER than feedforward — no attractor possible)
- drive_steps 100 → 300 (3x time for attractor to settle)

This tests the Patterson hub-and-spoke / ATL attractor
hypothesis directly.

## What we've learned about P5

- **Size doesn't matter** (iter C control rules out bottleneck)
- **Test methodology improvement helped** (iter A engram-tag vs
  raw cosine — first time same > cross)
- **Strict gating doesn't matter** (iter B no movement)
- **Architecture lacks attractor formation** (the remaining
  hypothesis; iter D tests it)

If iter D fails, the iron law says: don't keep trying parameter
tweaks. The implementation is missing a fundamental piece. Next
move would be **alternative test methodology** — measure the
LEARNED WEIGHTS (wernicke→semantic_cortex matrix) directly,
bypassing the noisy dynamics. If weights look right but
dynamics don't, that's an attractor-formation bug. If weights
also look wrong, the training isn't actually learning the
mapping.

## What's NOT broken

- P1 trisynaptic loop: 3/3 multi-seed PASS
- P4.1 positional binding: 3/3 multi-seed PASS
- P2 engram tagging: 12 unit tests pass + used by P3.1, P4.1, P5
- P3.1 concept replay: 5 unit tests pass
- Substrate builds cleanly at all scales tested
- Wall clock fine (~5-6 min/seed)

## Path forward

1. Wait for iter D seed 42.
2. If iter D PASS: launch 43/44 for multi-seed confirmation.
3. If iter D FAIL: shift to alternative test methodology
   (weight inspection vs dynamics inspection). Then if even
   the weights look wrong, fundamental training paradigm
   needs rework.
