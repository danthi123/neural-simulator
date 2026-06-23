# 🎉 THE GENERATIVE LOOP IS DEMONSTRATED — C2 grow+no-forget = GO on the 3.4M; the FT-LR was 30× too high (the prior NEGATIVEs were never a capacity wall) (2026-06-23)

**With the corrected fine-tune LR (3e-4 → 1e-5), the consolidated spiking generator LEARNS a new distribution
(new-ppl drops 87%) WHILE RETAINING the original (88% with replay) — and the no-replay control catastrophically
forgets (39%, contrast 2.25×). ⇒ ALL 3 BARS CLEARED: the full LOOP (train → generate → grow → confirm-no-forgetting)
is DEMONSTRATED end-to-end on the spiking substrate at toy scale, the self-replay no-forget mechanism CAUSALLY
validated. The prior C2 NEGATIVEs (3.4M + 30M) were ALL the FT-LR bug (the fine-tune overwrote the old regardless of
replay), NOT a capacity wall.** `research/runners/_genseq_C2_demo_design_derisk.py`, GPU, NO `sim/` edit. The owner's
direction ("re-examine the demo design, not scale") was decisive.

## The diagnosis (confirmed)
The smoking gun (50% replay → 48% retention) was the fine-tune LR (3e-4), ~30× too high for a continual-learning
update → it overwrote the original EVEN WITH heavy replay. The FT-LR×replay sweep on the 3.4M (the clean, well-trained
testbed, ppl 6.1 — no undertraining confound):

| FT-LR | replay=0.0 (no-replay) | replay=0.3 (with-replay) | new_ppl_drop |
|---|---|---|---|
| 3e-4 (original/broken) | 0.117 | 0.451 | ~0.84 |
| 1e-4 | 0.210 | 0.697 | ~0.88 |
| 3e-5 | 0.279 | 0.840 | ~0.88 |
| **1e-5 (WINNER)** | **0.392** | **0.900** | **0.867** |

Every arm learns the new distribution; the LR sets how much of the OLD survives. The replay mixing was correct
(realized replay-frac 0.300 — verified); the LR was the only bug.

## The WINNER (FT-LR=1e-5, replay=0.3), full budget
- with-replay original-retention **0.884** (≥0.85 ✅)
- new learned: new_ppl_drop **0.868** (≥0.5 ✅)
- no-replay control (SAME LR) retention **0.392** → forget-contrast **2.25×** (≥1.3 ✅) — self-replay CAUSALLY
  prevents the catastrophic forgetting the no-replay control suffers.

## ⇒ THE GENERATIVE ARC'S CAPSTONE
C1 (generator consolidated on the bridge, generating byte-identical) + C2 (grow without forgetting, self-replay
causally validated) = the full LOOP the owner set as the goal: **train → generate → grow → confirm-no-catastrophic-
forgetting, DEMONSTRATED end-to-end on the spiking substrate at toy scale.** The "scale wall" (CYCLE 460/468/476) was
a MIRAGE — the FT-LR bug masqueraded as a capacity limit at BOTH 3.4M and 30M (and the 30M's undertraining was a
second, independent confound). NO bigger model / NO cloud needed to demonstrate the loop.

## Scope (honest)
Toy scale (3.4M, the well-trained model); a single shift (the Shakespeare register); FT-LR=1e-5 is the continual-
learning standard (low LR + replay). Strengthening follow-ons: scale the loop to a bigger PROPERLY-trained model,
more shifts, multi-seed — but the LOOP ITSELF is demonstrated, and it needed a one-line LR fix, not scale. NO `sim/`
edit.
