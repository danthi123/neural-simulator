# P5 iteration A seed 42: FAIL — engram-tag methodology + 3x training

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3 (catalog G.11 + G.13)
**Status:** Honest report. Iteration A targeted two of four
diagnosed fixes (engram-tag methodology + more training events).
Seed 42 result: STILL FAIL, but signal direction is correct for
the first time.

## Result (seed 42)

| Metric | Original (100 ev) | Iter A (300 ev + engram-tag) | Target |
|---|---|---|---|
| apple_self cosine | 0.216 | **0.227** | > 0.5 |
| apple_river cosine | 0.290 | **0.174** | < 0.4 |
| Naming ratio | 0.89x | **1.08x** | > 1.3x |
| Verdict | FAIL | **FAIL** | — |

Critically: for the first time, **apple_self > apple_river**
(0.227 > 0.174). The engram-tag methodology IS picking up
same-concept stability above cross-concept similarity. The
naming ratio improved from below baseline (0.89x) to slightly
above (1.08x). Direction is correct; magnitude is too low.

## Wall clock

295 sec (5 min) for full pipeline (build + encode + replay +
test). Much faster than the original 8-15 min estimate;
iter A is fast enough for multi-seed sweeps.

## Diagnosis update (biology-first workflow step 5, attempt 2)

The original FAIL diagnosis flagged four issues:

1. ✅ **Training events insufficient** → iter A bumped 100 → 300.
   Result: same_concept signal lifted SLIGHTLY but not enough.
2. ⚠️ **wernicke bottleneck (200 neurons)** → iter A did NOT
   address. Still on the table for iter C.
3. ✅ **Test methodology** → iter A engram-tag works (signal
   direction correct). Still low magnitude.
4. ❌ **Mixed gate timing during training** → iter A did NOT
   address. Still mixing hippo + ventral gates during encoding.

Iter A addresses 2 of 4. The remaining two (wernicke scale,
strict two-stage gating) are the next iterations.

## Path forward

Per biology-first workflow Rule 8 step 5: failure → return to
step 3 or fix implementation detail. The signal-direction-correct
result confirms the methodology change is sound; the underlying
dynamics need stronger learning signal.

**Iteration B (in flight at seed 42):**

```bash
python -m research.runners.validate_ventral_semantic \
    --seed 42 --n-train-events 300 --n-replay-cycles 30 \
    --strict-two-stage --drive-lang-during-replay \
    --out research/findings/raw/g11_bg/p5_iterB_seed42.json
```

Two fixes per McClelland 1995 CLS + Wilson & McNaughton 1994:

1. **Strict two-stage gating** (`--strict-two-stage`):
   - **Encoding phase**: only hippo gates open (lang_to_ec,
     ec_to_dg, dg_to_ca3, ca3_to_ca1, ec_to_ca1). Ventral
     gates closed (frozen). Hippo learns fast.
   - **Replay phase**: ventral gates AND replay gates open.
     ca3_swr_burst + ca1_to_semantic + lang_to_wernicke +
     wernicke_to_semantic active during replay.

2. **Drive lang_input during replay** (`--drive-lang-during-replay`):
   During each replay burst, also drive lang_input(concept) so
   wernicke sees both word + replayed meaning. Biology: Wilson &
   McNaughton 1994 coordinated hippo+cortex replay; real cortical
   replay reactivates phonological codes alongside semantic
   content.

This addresses diagnosis #4 (mixed gate timing) and provides a
sharper learning signal for the lang→wernicke→semantic chain.

**Iteration C (pre-staged, if B fails):**

```bash
python -m research.runners.validate_ventral_semantic \
    --seed 42 --n-train-events 300 \
    --n-semantic-cortex 1000 --n-wernicke 400 \
    --strict-two-stage --drive-lang-during-replay \
    --out research/findings/raw/g11_bg/p5_iterC_seed42.json
```

Scales wernicke 200→400 AND semantic_cortex 500→1000. Addresses
diagnosis #2 (wernicke bottleneck).

## What's NOT broken (still)

- Substrate builds correctly (4568 neurons, 881445 synapses)
- ec_context + ventral_semantic prereq checks raise ValueError
  for invalid combinations
- P1 trisynaptic loop still passes multi-seed (3/3)
- P4.1 positional binding still passes multi-seed (3/3)
- P2 engram tagging API still works (12 unit tests pass)
- P3.1 concept replay still works (5 unit tests pass)
- Iter A's engram-tag methodology is a real improvement (signal
  direction correct for the first time)

## Architectural-escalation count

Per superpowers:systematic-debugging Phase 4.5: if 3+ fixes
fail, question architecture vs continuing to fix symptoms.

- Attempt 1: original P5 design at 100 events, raw spike-count
  cosine. FAIL 3/3 multi-seed.
- Attempt 2 (iter A): engram-tag methodology + 300 events.
  FAIL at seed 42 (signal direction correct but low magnitude).
- Attempt 3 (iter B): strict two-stage gating + lang drive
  during replay. RUNNING.

If iter B also FAILs at seed 42, attempt 4 (iter C) is the
LAST iteration before architectural escalation per the iron law.
At that point, the fundamentals would need to be questioned:
catalog G.11/G.13 dual-stream model may need substantive
rework, or the toy scale (n_wernicke=200) is fundamentally
insufficient.
