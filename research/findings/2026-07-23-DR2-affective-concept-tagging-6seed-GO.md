---
type: finding
status: live
date: 2026-07-23
mechanism: affective-tagging
---

# DR-2 (new-direction Phase-0): affective concept-tagging — concepts LEARN their valence from the association graph, 6-seed GO (2026-07-23)

First validated faculty of the owner's reframe (a brain with an affectively-colored world-model, not a fact store).
The mechanism (reuse-by-import, NO `sim/` edit): a learned rate-Hebbian co-occurrence / PPMI concept-association graph
(built in-runner from TinyStories, reusing the stream-cortex code) + an opponent V+/V- affect population (Namburi-Tye)
seeded from a Warriner-approximate VAD lexicon + valence inheritance by seed-clamped spreading activation (EMERGE-30 /
harmonic label-prop) over the graph. A concept inherits its "how it should feel" from the affective company it keeps.

## 6-seed result — GO (all 6 checks)
- **Held-out valence r = +0.811** (per-seed 0.765-0.831; every seed >= the 0.55 gate).
- **Permuted-graph collapses: mean -0.064** (the LEARNED structure carries affect, not a lookup) — decisive on all seeds.
- Shuffled-seed-labels collapse (mean -0.144) — NB on seed 100 the shuffled null reads +0.203 (above the 0.20 line);
  the aggregate gate is mean-based, so real ≫ shuffled every seed but the "collapse" is mean-smoothed on that one seed.
- **Non-load-bearing padding (CORRECTED 2026-07-23):** `beats_seed_only` compares against a CONSTANT predictor whose
  Pearson r ≡ 0 by construction, so the check reduces to "real ≥ 0.30" and cannot discriminate; `opponent_sign`
  (corr -0.806 every seed; pos/neg net separation +0.38) is a near-arithmetic consequence of the rectified-opponent
  seeding on a valence-clustered graph. Both are near-unfalsifiable — the GO rests on the falsifiable held-out
  Pearson + the permuted-graph collapse, not on these two.
- Arousal (secondary) r=+0.694.
Runner `research/runners/_affect_distributional_tag_derisk.py`; result `research/findings/raw/_affect_distributional_tag_6seed.json`.

## Honest caveats (surfaced)
(a) numpy-CPU READ of the mechanism — the on-bridge SPIKING opponent-population confirm is the GPU follow-on.
(b) r~0.81 > the Bestgen-Vincze r~0.71 literature ref partly because the embedded lexicon is a coarse well-separated
139-word core; expect movement toward the literature value with the full 13,915-word Warriner CSV (`--warriner-csv`).
(c) graph built in-runner (no cached artifact maps the stream codes to Warriner-labelled words).

## Adversarial verification (2026-07-23): HOLDS — the cleanest GO of the batch
Both skeptics tried and could NOT refute it: per-seed r reproduces to 3 decimals; the permuted-graph collapse (mean
-0.064) is the load-bearing null and it is genuine input-destruction; the held-out split is a real transductive one
(held valence never enters the seeds or the co-occurrence graph); importing the runner loads ZERO `sim` modules, so
"NO `sim/` edit" is decisively true; the blind seeds 100/101/102 all pass. Decisively, a mid-band-only stress test
(|valence−5| ≤ 1.5 — the hard graded cases, with the obvious happy/scary extremes stripped) still inherits valence at
**r = 0.683** — so the metric is genuine graded inference, NOT an obvious-happy-vs-scary sign lookup. The only
corrections (non-verdict-threatening) are the two non-load-bearing padding checks + the seed-100 shuffled +0.203
mean-masking above, and the already-disclosed r-inflation (caveat b).

## Net
The affective world-model's foundation works: concepts acquire learned emotional valence from experience, and the
anti-cheats confirm it rides the learned association structure. Phase-0 P0.1 of the master roadmap. NO `sim/` edit.
