# EMERGE-17 / toward-language — GO (6/6 seeds): the emergent on-bridge learning GENERALIZES across SIMILAR words. An association learned for a TRAINED word transfers to a HELD-OUT similar word never trained on — because similar words share micro-columns (overlapping sparse codes), the held-out word drives the learned coincidence pathway from the SHARED cells. The sequence cortex now has GENERALIZING word representations, not just memorized codes. NO `sim/` edit — the ONLY change is the word→code encoding.

**2026-07-02 (autonomous; the generalization research gate's recommended de-risk).** Runner `research/runners/_emerge17_generalizing_word_codes_derisk.py`. Reuse-by-import `_emerge14` (`build_pool_bridge`/`apply_kernel_update`/`coincidence_predict`); NO `sim/` edit; CPU numpy-backend; 6-seed.

## The mechanism (the ONLY change = the encoding)
The EMERGE-14/15/16 architecture used ORTHOGONAL per-word columns → the model could only MEMORIZE the exact codes/sequences it trained on. Per the generalization research gate, the residual is the word→active-set ENCODING ALONE (the `sim/` coincidence kernel + three-term update unchanged). Here each word = a fixed sparse code over a SHARED micro-column pool (its identity SDR = one cell per micro-column); SIMILAR words SHARE micro-columns, so their SDRs OVERLAP (Numenta semantic folding; Ahmad-Hawkins: SDR overlap IS similarity). Learning "dog → home" potentiates dog's SDR cells onto home's cells (distal coincidence synapses); presenting a HELD-OUT similar word "wolf" (whose SDR shares the family micro-columns with dog) drives home's coincidence from the SHARED cells → if ≥ `act_th` shared cells fire, home is predicted → the association GENERALIZES to wolf without ever training on wolf.

## Result — GO (6/6 seeds)
Families canines {dog, wolf, fox} → "home", felines {cat, lion} → "away"; TRAIN on ONE per family (dog→home, cat→away); HOLD OUT the similar words {wolf, fox, lion}; seeds 42/43/44/100/101/102:
- **Held-out generalization 1.000** (all 6 seeds) — wolf/fox → home, lion → away, from a single trained family member, never trained on the held-out words.
- **ORTHOGONAL-code control 0.000** — when families do NOT share micro-columns (disjoint codes), there is NO transfer. This cleanly isolates code OVERLAP as the cause of generalization (not a spurious bias).
- **dAP-LESION 0.000** (coincidence off → no priming → collapses), **DERANGED family→branch 0.000** (inconsistent mapping → chance), **untrained 0.000**. No teacher.

## Significance
The emergent sequence cortex now has a GENERALIZING lexical representation — the hallmark of a real language cortex (similar words transfer learning) — on the real spiking substrate, with NO `sim/` edit, the only change being the overlapping-SDR word encoding. Combined with EMERGE-15 (high-order prediction) + EMERGE-16 (production), the substrate now supports memorization AND generalization of word associations. The overlap-vs-allocation risk the gate flagged did not bite (the coincidence + kernel are unchanged; generalization is a property of the input codes).

## Honest scope + next
- CHEAP-FIRST: controlled SYNTHETIC similarity + a BIGRAM association isolates the generalization-from-overlap claim from the high-order sequence machinery (EMERGE-15 already validated that separately). 
- Next: (a) HIGH-ORDER sequence generalization — combine the EMERGE-15 shared-middle corpus with overlapping SUBJECT codes so a held-out similar subject generalizes THROUGH the shared middle; (b) the REAL stream-cortex PPMI codes (verified similarity-structured at `_phaseB_stream_codes_320_seed42.npy`) as the SCALE-UP to a real vocabulary; (c) GROUND the emitted words to the no-confab moat; (d) the open-domain surface-fluency research gate.

## Artifacts
`research/runners/_emerge17_generalizing_word_codes_derisk.py`, `research/findings/raw/_emerge17_generalizing_word_codes{,_6seed}.json`. Prior: `2026-07-02-sequence-cortex-generalizing-word-codes-research-gate.md`, `2026-07-02-emerge16-word-generation-GO.md`, `2026-07-02-emerge15-word-sequence-lm-GO.md`.
