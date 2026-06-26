# First-chat-ready bar (#1) — the readiness criteria, with generalization demoted to a reported floor

**Date:** 2026-06-26 | **Roadmap Step 1 (owner decision #1).**

**Context:** the 2026-06-26 reframe established that **generalization is NOT the first-chat gate** — a good first chat is carried by recall + the no-confab moat + the DiscursiveTurn engage-and-discuss richness, with vocab breadth for "relate to anything." Generalization is substrate-capped (over-training densification; `2026-06-26-breadth1454-window-sweep.md`) and is reported, not gated.

## The bar — a brain is first-chat-ready when ALL of these hold:

1. **Vocab ≥ ~1,000 concepts** — breadth for relate-to-anything.
2. **Recall ≥ 0.95** — who/what on stored facts.
3. **Moat: 0 false-accepts (HARD)** — never assert a fabrication; never weakened, at any scale.
4. **Generalization: a REPORTED soft floor, NOT a magnitude gate** — require the *derangement control to collapse* (the gen structure is real, not noise) and gen > chance; do NOT gate on the magnitude (substrate-capped ~0.05 coherent / ~0.03 Pearson at 1,454; the multi-bridge + dendritic frontier are the gen levers, not first-chat blockers).
5. **DiscursiveTurn quality rubric ≥ 8/10 (the actual pass/fail)** — a 10-prompt sample conversation produces mixed-type (certain / novel-flagged / discuss-via-adjacent / phatic), moat-safe (verified-or-flagged, never bare-fabricated) paragraphs. Checked on the console (Step 2).

## The 7K working brain (`brain1454_w7000_seed42.npz`) vs the bar

| criterion | bar | 7K brain | pass |
|---|---|---|---|
| vocab | ≥1,000 | 1,454 | ✓ |
| recall | ≥0.95 | 0.958 | ✓ |
| moat | 0-FA | 0-FA | ✓ |
| gen (reported floor) | derangement-collapse + >chance | 0.054 coherent / +0.027 Pearson, derangement collapses, 2.9× chance | ✓ (structure real) |
| DiscursiveTurn rubric | ≥8/10 | — | **pending Step 2 (console)** |

⇒ the brain **passes every quantitative bar**; the discursive-quality rubric is the final check, run on the wired console.

## Code change (this step)

The curriculum runner's GO verdict (`_curriculum_step1_320_real_corpus.py:786`) was demoted: `go` now requires `recall_ok and moat_ok and der_ok and frozen_ok` — **gen dropped from the gate** (still computed + reported; the derangement-collapse *validity* still gates). The verdict text + the miss-list updated to report gen rather than flag it a miss. (The historical `--gen-bar 0.80` remains as a reported reference only.)
