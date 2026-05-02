# 2026-05-02 — Multi-decoder NEGATIVE: 28.5% ceiling is real, not eval artifact

**TL;DR:** Tested 4 alternative decoders alongside default delta-from-
baseline on a fresh v2 100-ep seed=42 run. None beat delta:

| Decoder | W→A Accuracy |
|---|---|
| delta (current default) | 27% |
| drive_only | 21% (baseline asymmetry hurts) |
| ratio | 27% |
| zscore | 27% (mathematically same as delta in orthogonal-pool setup) |
| clipped | 25% |

I→W (delta only): 33% (matches v2 baseline at seed=42).

This is the **8th confirmation** that 28.5% W→A is a hard architectural
ceiling — neither cascade quality (curriculum), nor reward shaping, nor
drives, nor pool sizes, nor lang region size, nor decoder methodology
exceeds it.

## What this rules out

The remaining "decoder/eval" hypothesis from the curriculum-NEGATIVE
analysis is now also falsified. The argmax-of-delta IS extracting
~all the signal the network produces.

Alternative decoders tested:
1. **drive_only** (no baseline subtraction): 21% — significantly worse.
   Confirms baseline asymmetry across motor pools is large; ignoring
   it hurts. Baseline subtraction is necessary.
2. **ratio** (multiplicative): 27% — identical. Multiplicative vs
   additive normalization gives same argmax under our spike count
   distributions.
3. **zscore** (delta / mean-baseline): 27% — mathematically equivalent
   to delta in orthogonal-pool setup (uniform scaling preserves argmax).
4. **clipped** (positive-delta-only): 25% — slightly worse. Negative
   deltas DO carry information (which pools have suppressed firing
   given language drive); clipping loses it.

## Why the ceiling exists

The network's **language pathway weights** ARE differentiated (3-4/4
tokens have target_motor preference, weight diagnostic across 6 seeds
shows mean east +0.128, west +0.072). But the **spike count
differentials at the readout** are small (typical delta ~5 spikes out
of ~10 baseline range). Per-trial baseline noise (varies 0-17 across
trials) is comparable to the differential signal.

To extract higher accuracy, we'd need either:
- Bigger spike count differentials (more drive, longer integration windows)
- Lower per-trial baseline variance (cleaner cascade)
- More differentiated weights (deeper architectural changes)

The first 2 were tested and didn't help. The 3rd is the remaining
direction but requires significant architectural rebuild.

## Comprehensive list of NEGATIVE findings (8 total)

| Variation | Result |
|---|---|
| Reward shaping (`wrong_move_reward=0`) | NEGATIVE |
| Stronger training drives (200→400, eval 200→500) | NEGATIVE — identical weights |
| Drive=500 reeval cross-seed | NEGATIVE — variance not signal |
| Bigger motor pools (10→30) | NEGATIVE — east FLIPPED to REV |
| Longer training (100→200 ep) | NEGATIVE — weights saturated |
| Bigger lang regions (256→512) | NEGATIVE — 18% W→A |
| Curriculum (visuomotor first) | NEGATIVE — same weights as v2 |
| Alternative decoders (drive_only, ratio, zscore, clipped) | NEGATIVE — none beat delta |

## What's left to try (deeper architectural)

1. **Different language pathway STRUCTURE**:
   - Wider lang_input → motor density (current 0.30, try 0.60+)
   - Sparser tokens (sparsity 0.10 → 0.05)
   - Distributed motor encoding (subpopulations within shared pool)

2. **Active teaching signal during training** (not the heuristic cheat):
   - Inject a "where is the goal" cue ONLY during training to push
     cascade past 60% correct moves
   - Remove cue for eval — agent has to use language pathway

3. **Pretrained language pathways**:
   - Train language→motor mapping in a SUPERVISED phase first
   - Then embodied training refines

4. **Different training task**:
   - Current 4-direction navigation may be too simple
   - Add diagonal directions (8 actions) for richer training signal

These are all 1-2 week engineering investments. Recommend user-directed
prioritization rather than autonomous selection.

## Files

- Result: `research/findings/raw/g11_bg/text_eval_multidec_v2_seed42.json`
- Checkpoint: `research/findings/raw/g11_bg/text_eval_multidec_v2_seed42.simstate.h5`
- Code: `research/runners/text_eval.py` (commit 2654b82)
