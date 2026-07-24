# DR-2 (new-direction Phase-0): affective concept-tagging — concepts LEARN their valence from the association graph, 6-seed GO (2026-07-23)

First validated faculty of the owner's reframe (a brain with an affectively-colored world-model, not a fact store).
The mechanism (reuse-by-import, NO `sim/` edit): a learned rate-Hebbian co-occurrence / PPMI concept-association graph
(built in-runner from TinyStories, reusing the stream-cortex code) + an opponent V+/V- affect population (Namburi-Tye)
seeded from a Warriner-approximate VAD lexicon + valence inheritance by seed-clamped spreading activation (EMERGE-30 /
harmonic label-prop) over the graph. A concept inherits its "how it should feel" from the affective company it keeps.

## 6-seed result — GO (all 6 checks)
- **Held-out valence r = +0.811** (per-seed 0.765-0.831; every seed >= the 0.55 gate).
- **Permuted-graph collapses: mean -0.064** (the LEARNED structure carries affect, not a lookup) — decisive on all seeds.
- Shuffled-seed-labels collapse (-0.144); seed-only baseline +0.000 (held-out beats it).
- Opponent V+/V- genuinely opposed (corr -0.806 every seed); pos/neg net separation +0.38.
- Arousal (secondary) r=+0.694.
Runner `research/runners/_affect_distributional_tag_derisk.py`; result `research/findings/raw/_affect_distributional_tag_6seed.json`.

## Honest caveats (surfaced)
(a) numpy-CPU READ of the mechanism — the on-bridge SPIKING opponent-population confirm is the GPU follow-on.
(b) r~0.81 > the Bestgen-Vincze r~0.71 literature ref partly because the embedded lexicon is a coarse well-separated
139-word core; expect movement toward the literature value with the full 13,915-word Warriner CSV (`--warriner-csv`).
(c) graph built in-runner (no cached artifact maps the stream codes to Warriner-labelled words).

## Net
The affective world-model's foundation works: concepts acquire learned emotional valence from experience, and the
anti-cheats confirm it rides the learned association structure. Phase-0 P0.1 of the master roadmap. NO `sim/` edit.
