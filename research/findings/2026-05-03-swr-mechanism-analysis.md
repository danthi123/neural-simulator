# SWR W→A regression — mechanism analysis (n=3)

**Date:** 2026-05-03 04:30 EDT (autonomous overnight, GPU still running batch)

The 3-seed v2+SWR result showed a consistent **W→A drop of ~6pp**.
Per-direction analysis revealed the mechanism.

## What's NOT happening

The SWR replay does NOT uniformly degrade all directions equally,
and the *which-direction-is-weak* pattern is NOT consistent across
seeds:

| seed | weak directions in W→A | strong directions |
|---|---|---|
| 42 | south (12%) | north (28%), west (32%) |
| 43 | south (12%) | east (28%) |
| 44 | north (16%), west (16%) | south (36%) |

If H1 (replay distribution bias) were straightforwardly true, the
weak directions would consistently be those underrepresented in the
buffer. But seed 44's weak directions (N, W) are different from
seed 43's (S). Initially this looked like a contradiction.

## What IS happening: per-seed amplification

Looking at *which motor pool over-predicts* (column sums in the
confusion matrix), the picture clarifies:

| seed | baseline pred-dist | SWR pred-dist | Δ |
|---|---|---|---|
| 42 | N=22 E=32 S=28 W=18 | N=26 E=32 S=20 W=22 | S −8 |
| 43 | N=20 E=27 **S=40** W=13 | N=26 E=31 S=21 W=22 | **S −19** |
| 44 | N=24 E=30 S=22 W=24 | N=21 E=19 **S=36** W=24 | **S +14** |
| 100 | N=35 E=33 S=12 W=20 | (in flight) | |
| 101 | N=29 E=30 S=14 W=27 | (in flight) | |
| 102 | N=29 E=23 S=27 W=21 | (in flight) | |

The **direction the cascade naturally over-emits during training
gets MORE amplified by SWR replay**.

- Seed 43: cascade was already S-biased (40% S predictions in
  baseline). After SWR, S dropped to 21% — not because S is
  under-predicted, but because the BUFFER had so many correct-S
  experiences that replay flooded the language→motor pathway with
  S-coactivation, making other directions much harder to read out.
  Wait actually, S DROPPED in pred-dist for seed 43... that's the
  opposite of what amplification predicts. Let me reconsider.

Hmm — the picture is messier than "the dominant direction gets
amplified." Let me look harder.

Seed 43: baseline S=40 (cascade dominantly picks S). After SWR:
S=21 (more balanced!) but ACCURACY DROPS. So somehow rebalancing
the predictions decreased accuracy. Why?

Maybe the baseline S-bias was actually CARRYING the accuracy: when
all words got predicted as S, "south" alone was correct, but the
overall accuracy was already capped. After SWR, predictions
balanced but wrong-balanced — north word predicted N more often
but still not enough for >25%, etc.

Seed 44 is even cleaner: SWR pushed predictions toward S (which
WASN'T the baseline majority). So it's not "amplify the existing
bias" — it's "bias toward whatever the BUFFER has more of." Which
might be different from the baseline pred-dist because the buffer
captures CORRECT moves, not all moves.

## Bottom line for hypotheses

**The mechanism is more complex than initially modeled.** SWR
amplifies whichever (token, action) pairs are over-represented in
the *correct-experiences* buffer, which is determined by which
direction the agent SUCCESSFULLY navigated more often — that's a
joint product of cascade bias × goal placement × random seed.

This actually argues even more strongly for H1 (balanced replay):

- Default replay sampling preserves whatever bias exists in the
  correct-experiences buffer
- That bias varies stochastically per seed
- It hurts accuracy regardless of which direction is over-represented
- Balanced sampling (N/4 events per direction) removes this bias
  entirely

**Updated prediction:** H1 should rescue W→A toward baseline (28.5%)
or beyond. If it doesn't, H4 (architecture limit) becomes the lead
hypothesis.

## What to do

1. Wait for 4-seed batch (3 more seeds: 100, 101, 102)
2. Confirm 6-seed W→A regression mean (expected: ~22%)
3. Run H1 (balanced replay) at 6 seeds (~7 hours)
4. Run H4 (PFC isolation) at 6 seeds (~2.5 hours) regardless of H1
   outcome — it sets the upper bound either way

Both are queued via PowerShell orchestrator scripts.
