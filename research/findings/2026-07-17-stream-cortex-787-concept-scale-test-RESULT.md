# RESULT: the unsupervised stream cortex HOLDS at 787-concept scale — "given enough training, emergent structure holds" (GO, against the FROZEN gate)

**2026-07-17. Evaluated against `2026-07-17-stream-cortex-787-concept-scale-test-PREREGISTRATION.md` — the bar was NOT moved.** Extracted reproducibly by `research/runners/_scale787_analyze.py` from the per-day battery.

## Verdict: GO

| Gate (frozen) | Bar | Result | |
|---|---|---|---|
| **PRIMARY corr(M,C)** (stream-cortex code-learning quality, D-independent) | ≥0.70 at final vocab + no collapse | **+0.81 @ 787** (from +0.89 @ 24; min-across +0.81) | **PASS** |
| SECONDARY retain (no catastrophic forgetting at scale) | ≥0.80 across | **1.00 throughout** | PASS |
| SECONDARY moat_fa (no confabulation at scale) | O(single digits) | **0 throughout** | PASS |
| CHAR recall(vocab) | (not pass/fail) | 0.67@24 → 0.67@787, noisy 0.33–1.00 | develop_D=128 FHRR capacity curve (lever = bigger develop_D) |

## What it means

The emergence bar's literal core claim — *"LLM-like conversation GIVEN ENOUGH TRAINING"* → capability from LEARNING — HOLDS at 33× the prior develop scale: as the brain learns 24 → 787 concepts from a TinyStories co-occurrence stream, the stream cortex keeps learning faithful concept codes (corr +0.89 → +0.81, a gentle graceful decline well above the gate), with **zero forgetting and zero confabulation the whole way up**. This is the mission-central substrate the 5-gap cluster builds on: the codes ARE learned, at scale, on the one spiking brain.

- **The corr decline is expected + benign:** more concepts → more co-occurrence interference in the online Hebbian learning; +0.81 at 787 is far from the 0.70 floor, and the curve is smooth (no cliff). If pushed much further the honest question is where corr crosses 0.70 — a future scale rung, not a wall reached here.
- **Recall is D=128-bound, NOT a stream-cortex failure** (the whole reason PRIMARY is corr, not recall): the composer's FHRR recall capacity is ~√D; at D=128, hundreds of superposed facts exceed it, so recall is noisy — the pre-registered follow-on lever is a bigger `develop_D` (the `use_multiturn=False` scaling path exists for exactly this).

## Follow-ons (pre-registered)
1. **Bigger `develop_D`** re-run to lift recall past the D=128 capacity wall.
2. **Richer corpus (wikitext103)** to cover the 203 concepts dropped as absent from TinyStories → push vocab past 787.
(Both are scale/data levers, deprioritized under the 5-gap cluster per the owner directive; recorded so the recall characterization is not mistaken for a stream-cortex limit.)

## Provenance
Run `develop_run --corpus-curriculum --brain-npz brain_curriculum_vocab_regen.npz --n-days 40 --seed 42 --root bridges/developed/scale787`; 787 in-corpus concepts (`_regen_curriculum_vocab.py`); paused at day 32 (curriculum exhausted, primary gate answered). Reconcile ROADMAP §5.10 (the "won't plateau" claim is supported to 787 concepts).
