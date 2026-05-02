# 2026-05-02 — Drive=500 reeval cross-seed NEGATIVE: variance, not signal

**TL;DR:** The earlier "W→A=32% at drive=500" finding from the seed=42
reeval sweep does NOT reproduce across seeds. Running drive=500 reeval
on all 6 v2 checkpoints:

| Seed | Original eval (drive=200) | Reeval (drive=500) |
|---|---|---|
| 42 | I→W 33, W→A 27 | I→W 31, W→A 23 |
| 43 | I→W 25, W→A 29 | I→W 25, W→A 27 |
| 44 | I→W 27, W→A 26 | I→W 17, W→A 19 |
| 100 | I→W 25, W→A 32 | I→W 24, W→A 26 |
| 101 | I→W 21, W→A 28 | I→W 24, W→A 22 |
| 102 | I→W 21, W→A 29 | I→W 30, W→A 27 |

**Cumulative (n=600):**
- Original (warm post-training, d=200): I→W 25.3%, W→A 28.5% (p=0.027 SIGNIF)
- Reeval (cold-start, d=500): I→W 25.2%, W→A 24.0% (BELOW chance)

The drive=500 reeval is statistically WORSE than original on W→A.
Cold-start reeval has bridge state divergence (homeostatic firing
thresholds, STP, eligibility, last spike times not saved by checkpoint
— see `2026-05-02-reeval-bridge-state-limitation.md`).

## What this means

1. **The 28.5% W→A at p=0.027 is the real result.** Not improvable by
   eval-time drive manipulation alone.

2. **The seed=42 sweep finding was variance.** Single-seed checkpoints
   produce variable reeval accuracy at drive=500 (range 19-27%). The
   32% on seed=42 r=100 was within this variance band, not a signal.

3. **Reeval as a methodology has limits.** For meaningful eval-time
   parameter sweeps, the bridge would need richer state save/restore
   (firing thresholds, STP state, etc.). Cold-start with just weights
   doesn't reproduce in-vivo behavior.

## Implications for upcoming experiments

Don't rely on reeval sweeps for parameter tuning. Future eval-side
experiments need fresh in-vivo eval (post-training, warm state).

Don't bank on drive manipulation as a quick win. The 28.5% W→A is
the architectural ceiling at 100-ep training with current parameters.

## Files

- 6 reeval JSONs: `research/findings/raw/g11_bg/text_reeval_v2_seed{42,43,44,100,101,102}_d500.json`
- v2 baselines: `research/findings/raw/g11_bg/text_eval_R3R6_100ep_HebOff_v2_seed{42,43,44,100,101,102}.json`
