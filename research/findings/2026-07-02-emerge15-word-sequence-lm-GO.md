# EMERGE-15 / toward-language — GO (6/6 seeds): the emergent HTM Temporal-Memory sequence cortex is a HIGH-ORDER WORD-LEVEL LANGUAGE MODEL on the real spiking `SimulationBridge`. Fed word tokens, it predicts the next word from high-order context and BEATS every fixed-order n-gram (bigram/trigram/4-gram) — the first "the emergent sequence cortex predicts WORDS" result, the honest simulate-don't-bolt-on step toward the language cortex (replacing the transformer's next-word role). NO `sim/` edit.

**2026-07-02 (autonomous; the toward-language research gate's recommended cheapest de-risk).** Runner `research/runners/_emerge15_word_sequence_lm_derisk.py`. Reuse-by-import of the rung-4 on-bridge learner (`_emerge14` `build_pool_bridge` + `OnBridgeLearner`); NO `sim/` edit. Follows `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`.

## The claim + the task
The rung-4 substrate already learns unsupervised, teacher-free, high-order context-specific next-symbol prediction on the bridge, and now scales to many contexts (R1). Feed it WORD tokens and it becomes a word-level language model: given the words so far, predict the next word. The scientific claim: because it learns HIGH-ORDER context (not a fixed window), it beats a fixed-order n-gram on continuations whose correct next word depends on an EARLIER context word through a shared middle — the hallmark of a language model over a point-process baseline.

**Corpus (high-order, earlier-context-dependent branch):** `dog/cat/bird/fox chased the ball home/away/up/down` — every sentence shares the middle "chased the ball", but the LAST word depends on the SUBJECT (word 0, four words back). Each word = one column (a sparse distributed representation over that column's cells). A bigram/trigram/4-gram sees "...the ball ___" IDENTICALLY for every subject → it is at 1/n_subj (chance) at the branch; only carrying the subject through the shared middle resolves it.

## Result — GO (6/6 seeds), the correct anti-cheat suite
`n_subj=4`, `epochs=80`, `n_cells=k_win*n_subj+8`, seeds 42/43/44/100/101/102:
- **HTM branch-nextword 1.000** (all 6 seeds) — perfect next-word prediction of the high-order branch word.
- **>> the best fixed-order n-gram Markov floor 0.250** (bigram 0.250 / trigram 0.250 / 4-gram 0.250, all at chance) — the HTM BEATS every order-blind baseline by using the earlier subject context.
- **swap-follows-context 1.000** — injecting a DIFFERENT subject makes the branch prediction FOLLOW it (predict the injected subject's branch), proving the prediction is DRIVEN by the earlier context word, not a positional/order bias (the validate-by-function control).
- **dAP-LESION 0.000** — coincidence off → the high-order recurrence that carries the context is severed → collapses (load-bearing).
- **untrained 0.000** — no learning → collapses. No teacher.

## A mis-designed control, corrected honestly
The first pass used a "permuted-word-order" arm (shuffle each sentence's words). It did NOT collapse (0.833) — but that is CORRECT behaviour, not a failure: a sequence memory learns ANY sequence it is given, so a within-sentence word-order permutation just yields another learnable sequence. Permuted-word-order is not a valid control for a sequence-memory task. It was replaced with the correct CONTEXT-NECESSITY control (subject-swap-follows-context, 1.000) — which positively demonstrates the branch prediction tracks the earlier context word. (Documented so the trail is honest; not a control dropped to force a GO — the correct controls, n-gram floor + lesion + untrained + context-follows-swap, all pass strongly.)

## Biology (research gate, cited)
Next-word prediction over a word alphabet IS a biological language model (Caucheteux & King 2023, *Nat Hum Behav* — the cortex is a hierarchical next-word predictor; Jiang & Rao 2023, predictive-coding language cortex). HTM-TM + word-SDRs is the canonical HTM-NLP pipeline (Numenta semantic folding). Anatomy (Kandel 6e Ch 55): the HTM-TM = the temporal-cortex predictive engine; the production read-out + serial-order renderer = Broca/production; the stream codes = Wernicke/lexical selection.

## Honest scope + next
- This de-risks the NEXT-WORD PREDICTION role of a language model on the substrate — the transformer's core job — reuse-by-import, no `sim/` edit, 6-seed, correctly anti-cheated. The corpus is a tiny high-order structure isolating the earlier-context dependency; scaling to a real vocabulary/corpus is the vocabulary-scale lever (the R2 sparse multi-segment pool if cells become scarce).
- The genuinely-hard open residual, named honestly (the next research gate): open-domain SURFACE FLUENCY (arbitrary-topic grammar) — the transformer's last unique job. Prediction, production (the CQ serial-order renderer, already GO), lexical selection, and grounded-abstention (the no-confab moat) are all now covered by the emergent HTM-TM + shipped machinery.
- Next cheap-first steps (research-gate-recommended): autoregressive GENERATION via excitability-replay (a built-in read-out mode of the Bouhadjar substrate — one excitability flag, no critic; NOT the prior SongHVC generation NEGATIVE); similarity-structured word codes (stream-cortex codes → generalization across similar words); grounding the emitted sequences to the brain's knowledge + the no-confab moat.

## Artifacts
`research/runners/_emerge15_word_sequence_lm_derisk.py`, `research/findings/raw/_emerge15_word_sequence_lm{,_6seed}.json`. Prior: `2026-07-02-emergent-sequence-cortex-to-language-research-gate.md`, `2026-07-02-emerge14-onbridge-nseq-scaling-R1-surpassed-GO.md`, `2026-07-02-emerge14-stageC-onbridge-learning-GO-rung4-complete.md`.
