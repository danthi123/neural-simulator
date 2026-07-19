# chat_synonym_demo seed 42 — runner end-to-end validation

**Date:** 2026-05-07
**Run ID:** 55bfaa8600b0 (launched via webapp dashboard `/api/runs/launch`)
**Wall clock:** 23.4 min (1366 s training + ~12 s eval)
**Architecture:** Tier 2.1 v4 scale-up (n_lang=4096, n_motor=1000,
n_motor_fs=120; ~12.7K neurons, ~12.9M synapses, 2.3 GB GPU)

## Purpose

End-to-end validation of `chat_synonym_demo.py` runner shipped in
commit 92b133f. Confirms the new runner:
1. Trains on 8-word synonym vocab via `synonym_mode=True`
2. Produces a chat-style transcript with PRI/SYN tagging
3. Outputs structured JSON via `--out-stats`
4. Launches cleanly through the webapp launcher API
5. Survives uvicorn `--reload` cycle

All five validations PASS.

## Result

```
Overall:        25.0%   (4/16)
Primary words:  50.0%   (4/8)  -- north, east, south, west
Synonym words:   0.0%   (0/8)  -- up, right, down, left
Per-action:     motor_N 0/4 | motor_E 1/4 | motor_S 1/4 | motor_W 2/4
```

## What this tells us (and what it doesn't)

**Seed 42 is below the validated Tier 2.1 BREAKTHROUGH 6-seed mean
(W→A 5/6 aligned, A→W mean 63.7%).**

This is not a refutation. Three reasons:

1. **Small sample.** chat_synonym_demo uses 2 turns/word (16 total
   turns) vs the validated Tier 2.1 paper's 25 trials/word
   (200 total). Single-seed variance dominates at this sample size.
   See `2026-05-02-text-io-formal-writeup.md` for sample-size
   discussion of W→A inference.

2. **Tier 2.1 5/6 means 1/6 fails.** Even at the proper sample size,
   one of {42, 43, 44, 100, 101, 102} can fall below alignment. Seed
   42 may be that seed; the other 5 should align if the BREAKTHROUGH
   replicates. We have only seed 42 here.

3. **Delta methodology vs evaluate_word_to_action.** chat_synonym_demo
   uses baseline-vs-driven delta on the motor pool spike count.
   The validated paper uses `evaluate_word_to_action` which has
   different normalization. Both should track each other but small
   sample noise can diverge.

Looking at the per-turn deltas:
- "up" produces `delta E +113`, `delta N +47` — synonym bound to
  motor_E instead of motor_N
- "right" produces `delta N +47`, `delta E +17` — synonym bound to
  motor_N instead of motor_E
- "down" produces `delta E +54`, `delta S +36` — synonym partially
  bound to motor_E
- "left" produces `delta N +51`, `delta W +2` — synonym bound to
  motor_N instead of motor_W

The synonym→motor bindings appear to have crossed pools relative to
their expected primary. This pattern is consistent with the Tier 2.1
1/6 failure mode (synonym sub-populations didn't form cleanly enough
in this seed's random init).

Confidence ratios are mostly 1.0–2.4× — typical for Tier 2.1 chat-demo
small-sample where the architecture barely differentiates words at
this n.

## What we DO learn from this single seed

- **The runner works end-to-end.** Training (1366 s ≈ 23 min)
  completes without errors. Eval produces valid JSON. Transcript
  formatting includes PRI/SYN labels. Per-action stats are reported.
  No crashes, no NaN, no GPU OOM.
- **Architecture wires correctly.** 12.7K neurons + 12.9M synapses
  + 30 populations + 2.3 GB GPU all initialize. NMDA on cortex/motor
  works at scale.
- **Wall-clock estimate corrected.** Initially documented as ~10 min;
  actual ~23 min for seed 42. Updated runner docstring + CHAT-DEMO-GUIDE
  + webapp PRESETS comment to 15-20 min (commit 8bba94f). Likely
  underestimate still — full Tier 2.1 v4 NMDA training is heavier than
  Tier 1 by a factor of ~3-4× per step.
- **Webapp launcher path works.** `POST /api/runs/launch` with
  `chat_synonym_demo` preset → runs cleanly with no live-mode flag
  injection (regression test commit ac740f6 doing its job).

## Next: real validation

For a real Tier 2.1 chat-demo result, run multi-seed:

```bash
bash scripts/multiseed_chat_demo.sh chat_synonym_demo 42 43 44 100 101 102
```

Expected: ~140 min total (23 min × 6 seeds), aggregate via
`research.runners.chat_demo_aggregate`. Should reproduce the
Tier 2.1 BREAKTHROUGH 5/6 aligned + A→W mean 63.7%.

Or use the larger sample size (`evaluate_word_to_action`) directly:

```bash
python -m research.runners.bio_three_factor \
    --biological --embodied-hebbian --synonym-mode \
    --apply-topographic-bias --enable-motor-fs \
    --n-events-per-direction 400 \
    --n-lang-input 4096 --n-motor-per-action 1000 \
    --n-motor-fs-per-action 120 \
    --seed 42
```

This is what the original BREAKTHROUGH 6-seed used.

## Lesson learned

Chat demos are visualization artifacts, not statistical evals. They
demonstrate that the architecture *does* the thing in a transcript-friendly
format. Don't conclude from a 16-turn chat demo that "Tier 2.1 doesn't
work." Use chat demos for showing users what's happening; use
`evaluate_word_to_action` (200+ trials/word, 6 seeds) for statistical
claims.

This finding documented per autonomous-runs principle #6
(anti-shortcut discipline): negative findings are real findings,
single-seed numbers are not refutations, document the sample-size
caveats honestly.
