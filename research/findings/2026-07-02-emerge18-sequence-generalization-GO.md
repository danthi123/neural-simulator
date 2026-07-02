# EMERGE-18 / toward-language — GO (6/6 seeds): the emergent on-bridge SEQUENCE cortex GENERALIZES a HIGH-ORDER prediction to a HELD-OUT similar word. A held-out similar SUBJECT ("wolf", never trained) predicts the correct family branch ("home") THROUGH the shared middle "chased ball" — generalizing from a trained similar subject ("dog") via overlapping micro-columns. This UNIFIES EMERGE-15 (high-order prediction) + EMERGE-17 (generalization): a generalizing high-order sequence language model on the real spiking substrate. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge18_sequence_generalization_derisk.py`. Reuse-by-import `_emerge14` (`build_pool_bridge`/`apply_kernel_update`/`coincidence_predict`) + the EMERGE-17 overlapping-code idea; NO `sim/` edit; CPU numpy-backend; 6-seed.

## The mechanism — the two-level word encoding
Per the generalization research gate, the ONLY change vs EMERGE-15 is the word encoding (the `sim/` coincidence kernel + three-term update are unchanged): a `SeqGenLearner` where each word = a set of MICRO-COLUMNS; the SUBJECT words in a family SHARE micro-columns (overlapping fixed identity SDRs = 1 cell per micro-column) while the shared MIDDLE keeps CONTEXT cells DISJOINT per family (the HTM allocation, unchanged — the gate's "columns overlap for generalization, context cells disjoint for disambiguation" reconciliation). Training "dog chased ball home" potentiates dog's identity SDR → the shared middle's canine-context cells → ... → home. Presenting the held-out "wolf chased ball" fires wolf's identity SDR (shares the canine micro-columns with dog) → the SHARED cells drive the middle's learned canine-context coincidence → wolf follows dog's high-order pathway → predicts home — a high-order generalization the held-out word never saw.

## Result — GO (6/6 seeds)
Canines {dog, wolf, fox} → home, felines {cat, lion} → away; sentence = [subject, chased, ball, branch]; TRAIN on ONE per family (dog, cat); HOLD OUT {wolf, fox, lion}; seeds 42/43/44/100/101/102:
- **Held-out sequence generalization 1.000** (all seeds) — the held-out similar subject predicts its family branch through the shared middle, from a single trained family member.
- **ORTHOGONAL-code control 0.000** — no shared subject micro-columns → no transfer (overlap is the cause).
- **dAP-LESION 0.000, DERANGED family→branch 0.000, untrained 0.000.** No teacher.

## Significance — the toward-language chain, unified
On the real spiking substrate, emergent + unsupervised + no `sim/` edit, the sequence cortex now does:
- **PREDICTION** (EMERGE-15): high-order next-word prediction beating fixed-order n-grams.
- **PRODUCTION** (EMERGE-16): autoregressive generation of the learned continuation.
- **GENERALIZATION** (EMERGE-17): a learned word association transfers to a held-out similar word.
- **HIGH-ORDER GENERALIZATION** (this): a held-out similar word generalizes a *high-order, context-dependent* prediction through a shared middle.
This is a generalizing high-order sequence language model — memorization AND generalization of high-order sequence structure — on one spiking brain, the honest simulate-don't-bolt-on path replacing the transformer's core roles.

## Honest scope + next
- Controlled SYNTHETIC similarity (defined families) isolates the mechanism cleanly. Next: (a) the REAL stream-cortex PPMI codes (verified similarity-structured at `research/findings/raw/_phaseB_stream_codes_320_seed42.npy`) as the SCALE-UP — top-K-sparsify real words to micro-columns, run EMERGE-17/18 on real similar words (dog↔cat from the corpus); (b) GROUND the emitted words to the no-confab moat (the grounded-lang machinery); (c) the open-domain SURFACE-FLUENCY research gate (the transformer's last unique job).

## Artifacts
`research/runners/_emerge18_sequence_generalization_derisk.py`, `research/findings/raw/_emerge18_sequence_generalization{,_6seed}.json`. Prior: `2026-07-02-emerge17-generalizing-word-codes-GO.md`, `2026-07-02-emerge15-word-sequence-lm-GO.md`, `2026-07-02-emerge16-word-generation-GO.md`.
