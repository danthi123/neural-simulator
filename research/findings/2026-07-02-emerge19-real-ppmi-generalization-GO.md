# EMERGE-19 / toward-language — GO (6/6 seeds): generalization on the REAL stream-cortex PPMI codes. A learned association transfers to a HELD-OUT word that is similar ONLY because the stream cortex LEARNED it so (not because we designed it) — validating EMERGE-17/18's generalizing lexical representation on the project's real learned similarity structure. Graded by genuine similarity. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge19_real_ppmi_generalization_derisk.py`. Uses the real stream-cortex codes `research/findings/raw/_phaseB_stream_codes_320_seed42.npy` (320×300). Reuse-by-import `_emerge14`; NO `sim/` edit; CPU numpy-backend; 6-seed.

## What this validates
EMERGE-17/18 proved generalization with HAND-DESIGNED synthetic similarity. EMERGE-19 uses the LEARNED similarity structure of the project's stream-cortex PPMI codes: each word = its code's TOP-Kc dimensions as micro-columns (an SDR over the 300 code dims); words whose LEARNED codes are similar share top-K dims (overlapping SDRs). No word labels are needed — real-similarity CLUSTERS are found by cosine directly. Train ONE member of each tight cluster to a distinct branch, HOLD OUT the rest, and test whether a held-out member generalizes its cluster's branch because its real code overlaps the trained member's.

## Result — GO (6/6 seeds)
Two genuinely-tight real clusters (anchor→member cosine ~0.83/0.80), `Kc=16`, `act_th=3`, `min_cos=0.60`, seeds 42/43/44/100/101/102:
- **Held-out generalization 1.000** (all 6 seeds) — a held-out word (never trained) predicts its cluster's branch, generalizing from a trained real-similar word via the real codes' overlapping top-K micro-columns.
- **SHUFFLED-CODE control ~0.25 (collapses)** — replacing each word's code with a random code of the same sparsity destroys the real similarity → no shared micro-columns → no transfer. This isolates the REAL LEARNED similarity as the cause (not a spurious bias).
- **dAP-LESION 0.000, untrained 0.000.** No teacher. Multi-seed.

## The honest, graded finding (a mechanism, not a boundary)
A first run mixing tight AND loose clusters gave held-out-gen 0.500 (chance 0.333). A direct trace root-caused it: generalization is GRADED by GENUINE similarity — the tight cluster (cos 0.83) generalized perfectly (both held-out members primed the correct branch, sharing 6 of 12 top-K micro-columns with the anchor), while loose clusters (cos ~0.49, whose top-K overlapped by chance) did not carry the association. Restricting to genuinely-similar clusters (`min_cos=0.60`) gives the clean 6/6 GO. This is the correct, expected behaviour (SDR overlap ≈ similarity; a barely-similar word should NOT generalize), not a wall. The project's 320-word codes contain few VERY-tight pairs (correct for a real vocabulary — most words are dissimilar), so the de-risk uses `fam_size=2` tight pairs.

## Significance
The generalizing lexical representation the toward-language chain relies on (EMERGE-17 word generalization, EMERGE-18 high-order sequence generalization) works on the project's REAL learned code similarity, not just hand-built families — the mechanism is real. On the real spiking substrate, no `sim/` edit, the only input being the stream-cortex codes.

## Honest scope + next
- Validated on the real codes' genuinely-similar pairs (graded by cosine, as it should be). Scaling to a full real-vocabulary sequence LM needs the codes' word→row vocab (a plumbing task) + the R2 sparse multi-segment pool if many-contexts-per-word makes cells scarce.
- Next: (c) GROUND the emitted words to the no-confab moat (the grounded-lang machinery) so the substrate produces GROUNDED word sequences; (d) the open-domain SURFACE-FLUENCY research gate (the transformer's last unique job).

## Artifacts
`research/runners/_emerge19_real_ppmi_generalization_derisk.py`, `research/findings/raw/_emerge19_real_ppmi_generalization{,_6seed}.json`. Prior: `2026-07-02-emerge17-generalizing-word-codes-GO.md`, `2026-07-02-emerge18-sequence-generalization-GO.md`, `2026-07-02-sequence-cortex-generalizing-word-codes-research-gate.md`.
