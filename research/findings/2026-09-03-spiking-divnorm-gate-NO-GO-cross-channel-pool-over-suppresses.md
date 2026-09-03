---
type: finding
status: contributing
date: 2026-09-03
mechanism: divisive-normalization gate (--dual-nonneg-divnorm) on the spiking ssm recurrence, sigma sweep
lane: language (own-voice mouth)
seeds: [42]
verdict: NO-GO
artifacts:
  - research/findings/raw/_divnorm_sig0.5_1seed.json
  - research/findings/raw/_divnorm_sig2_1seed.json
  - research/findings/raw/_divnorm_sig8_1seed.json
  - research/findings/raw/_divnorm_sig32_1seed.json
---

# Divisive-normalization gate on the spiking recurrence — NO-GO: cross-channel pool over-suppresses at every sigma

**Status:** NO-GO for the mechanism AS BUILT — a first-class deliverable. Divisive normalization is not refuted as a concept; the specific implementation (a cross-channel pool over all D=256 channels) over-suppresses the signal. A wall defers a METHOD.

## What ran

The convergent mechanism from the two fluency NO-GOs — a Carandini-Heeger divisive-normalization gate (`--dual-nonneg-divnorm`, merged `9333615d`) that divides each channel's drive by a pool summed over the channel population — was calibrated with a 1-seed sigma sweep (0.5 / 2 / 8 / 32) against the dual-nonneg spiking baseline, on the same Simple-Wiki BPE config as the baseline.

## Result — hurts at every sigma

deepest-bucket (10-99) `margin_vs_trigram`, seed 42 (single seed each — this is a calibration sweep, not a 6-seed verdict; but the direction is unambiguous):

<!--derived-->
Baselines for comparison (from prior 6-seed findings): dual-nonneg spiking (no divnorm) = −0.46; exact-math wkv = −0.125.

| divnorm sigma | margin_vs_trigram | wkv-NLL |
|---|---|---|
| 0.5 | −0.881 | 5.038 |
| 2 | −0.874 | 5.030 |
| 8 | −0.881 | 5.037 |
| 32 | −2.979 | 7.135 |

**Every sigma is WORSE than the −0.46 dual-nonneg baseline** — sigma 0.5-8 cluster near −0.88 (the pool dominates the denominator regardless of sigma), and sigma 32 collapses to −2.98 (near memoryless). The divnorm gate does not rescue content-selection; it degrades it.

## Diagnosis + next lever (NOT a wall)

The pool is summed over ALL D=256 channels, so the denominator `sigma^n + sum_j drive_j^n` is dominated by the 256-channel sum for any small sigma — squashing each `R_i` toward `1/256`-scale and starving the LIF readout of graded drive (training stalled near ln(V), which the build agent's smoke had already flagged; it held at scale). Sweeping sigma cannot fix this — the load-bearing variable is the pool SIZE/SCOPE, not sigma.

So divisive normalization is not refuted; the cross-channel-over-everything pool is. Untested variants: a LOCAL pool (a small channel neighborhood, mirroring the vision `satdiv` per-location pool that worked better on a smaller template population), or a learned/gated pool.

<!--derived-->
But before re-implementing, the cheaper and more promising lever is the TOKEN axis: more training text flipped the exact-math wkv from losing to a trigram (−0.125) to beating it (+0.02, contiguous 1-seed), and the spiking recurrence DOES use context (beats a bigram) — so whether more tokens also lifts the spiking is the next test (a spiking `--contiguous` 1-seed is running now). Divnorm-with-a-local-pool is banked as a follow-on lever.

## Reproduce

```bash
# One cell of the sweep (sigma=8) — produced research/findings/raw/_divnorm_sig8_1seed.json (one of the 4 cited):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence ssm --dual-nonneg --uniform-decay --n-layers 1 --dual-nonneg-divnorm --divnorm-sigma 8 \
    --d-model 192 --batch 128 --tokenizer bpe --corpus data/corpus/simplewiki.txt \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 --seeds 42 \
    --json research/findings/raw/_divnorm_sig8_1seed.json
```
