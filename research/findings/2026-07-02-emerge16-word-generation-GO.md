# EMERGE-16 / toward-language — GO (6/6 seeds): the emergent HTM Temporal-Memory sequence cortex GENERATES word sequences AUTOREGRESSIVELY on the real spiking `SimulationBridge`. From a CUE word it rolls out the correct learned continuation ("dog" → "chased the ball home") with NO external drive of the continuation, NO critic, NO extra learning — the substrate is now a word-sequence PRODUCER, not just a predictor. NO `sim/` edit.

**2026-07-02 (autonomous; the research-gate-recommended highest-leverage next step after the EMERGE-15 word-LM GO).** Runner `research/runners/_emerge16_word_generation_derisk.py`. Reuse-by-import of the rung-4 on-bridge learner (`_emerge14`) + the EMERGE-15 word corpus; NO `sim/` edit.

## The mechanism — excitability-replay (a built-in read-out mode)
Per the research gate (`2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`), autoregressive generation is a native read-out mode of the Bouhadjar-Diesmann 2022 spiking-HTM substrate: a dendritic-plateau-PRIMED (predicted) cell is one excitability step from firing, so driving the plateau alone rolls the learned sequence out. Realized here as an autoregressive loop on the same bridge machinery:
- Present the CUE word's winner SDR (it fires).
- The bridge's WEIGHTED coincidence recurrence predicts the next column's context-specific PRIMED cells.
- Those primed cells become the next active set (they "fire") and predict the next column.
- Repeat → the generated column at each step = the generated word.
No external drive of the continuation, no critic, no extra learning — pure predict→become-active rollout on the LEARNED connectivity. This is NOT the prior SongHVC generation NEGATIVE (that needed a self-comprehension critic that could not read serial order back — a different mechanism).

## Result — GO (6/6 seeds)
Corpus `dog/cat/bird/fox chased the ball home/away/up/down` (each word = a column), `n_subj=4`, `epochs=80`, seeds 42/43/44/100/101/102:
- **Exact-sentence generation 1.000** (all 6 seeds) — from each cue the substrate generates the ENTIRE correct learned sentence.
- **Branch-matches-cue 1.000** — the generated high-order branch word matches the cue (the generation carries the earlier subject context through the shared middle).
- **swap-follows-context 1.000** — each cue generates ITS OWN branch (a different cue rolls out a different continuation), proving the generation is context-DRIVEN, not a fixed rollout.
- **dAP-LESION 0.000** (coincidence off → nothing is primed → generation halts → collapses) and **untrained 0.000** — the learned high-order coincidence recurrence is load-bearing. Multi-seed.

## Significance
The rung-4 substrate now covers the two core language-model roles on-substrate, both emergent + unsupervised + on the real spiking bridge, NO `sim/` edit:
- **PREDICTION** (EMERGE-15): high-order next-word prediction beating fixed-order n-grams.
- **PRODUCTION** (this): autoregressive generation of the learned continuation from a cue.
Together with the already-GO competitive-queuing serial-order renderer (production ordering) and the no-confab moat (grounded abstention), the emergent HTM-TM + shipped machinery covers next-word prediction, production, lexical selection, and grounded-abstention — the transformer's roles, on the simulated substrate. "Simulate Broca, don't bolt on an LLM" — advancing.

## Honest scope + next (research-gate order)
- Tiny high-order corpus isolating the earlier-context dependency; the generation is the exact learned continuation (a memorized-then-replayed sequence, the correct first de-risk). Generalization + open-domain fluency are downstream.
- Next cheap-first: (b) SIMILARITY-STRUCTURED word codes (swap orthogonal codes → stream-cortex PPMI codes so dog↔cat generalize; NO `sim/` edit); (c) GROUND the generated words to the brain's knowledge + the no-confab moat (the existing grounded-lang machinery); (d) SCALE the vocabulary/corpus (R2 sparse multi-segment pool if cells become scarce, numpy-de-risk first). The genuinely-hard open residual (the NEXT research gate): open-domain SURFACE FLUENCY (arbitrary-topic grammar) — the transformer's LAST unique job.

## Artifacts
`research/runners/_emerge16_word_generation_derisk.py`, `research/findings/raw/_emerge16_word_generation{,_6seed}.json`. Prior: `2026-07-02-emerge15-word-sequence-lm-GO.md`, `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`, `2026-07-02-emerge14-onbridge-nseq-scaling-R1-surpassed-GO.md`.
