# The read-out-architecture lever WORKS (6-seed GO): DIVERSE-subsampling population readers lift the fully-spiking inheritance accuracy 0.458→0.542 (+18%) — true population coding (each property cell reads an INDEPENDENT random column subset) genuinely reduces the read variance, where identical readers (CYCLE-958) were a no-op. Validates the codon-variance diagnosis. Default byte-preserved. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_rung2_spiking_derisk.py` (`--diverse-readers --prop-k K`; reuse-by-import, additive default-off, NO `sim/` edit).
**Verdict:** 6-seed GO — the diagnosed deeper lever (a different read-out ARCHITECTURE) works, lifting the fully-spiking reasoner's accuracy where the two cheap tuning levers failed.

## Why this ran (the diagnosed deeper lever)
Two cheap codon-side levers had failed: property-cell population coding was a NO-OP (CYCLE 958 — the K property cells were wired IDENTICALLY, all receiving from all columns → deterministic identical reads → averaging does nothing), and codon width went the WRONG way (CYCLE 964 — wider codons lose selectivity). The diagnosis named the deeper lever: a different read-out ARCHITECTURE — **DIVERSE readers**. This builds it: each of the prop_k property cells per category wires to an INDEPENDENT RANDOM SUBSET (50%) of the pooler columns, so their reads of the codon are genuinely INDEPENDENT → averaging the apical drive reduces the read variance (the actual mechanism population coding is supposed to provide).

## The result
| config | held-out spiking inheritance (seed 42) | 6-seed |
|---|---|---|
| baseline (prop_k=2, identical readers) | 0.438 | 0.458 |
| prop_k=8 IDENTICAL readers (the CYCLE-958 no-op) | 0.438 | — |
| prop_k=8 DIVERSE readers | 0.500 | — |
| **prop_k=16 DIVERSE readers** | **0.625** | **0.542 ± 0.069** |
| prop_k=32 DIVERSE readers | 0.500 (subsets too sparse) | — |

**6-seed GO at prop_k=16 diverse: held-out 0.542 ± 0.069** (vs baseline 0.458 — **+0.084, +18%**), every seed beating chance AND all controls by ≥0.15 (deranged 0.104, permuted 0.104, lesion 0.000). The identical-readers control at the same prop_k=8 reproduces the baseline exactly (0.438 = the no-op), so the lift comes specifically from reader DIVERSITY, not from more cells. prop_k=16 is the sweet spot (prop_k=32 over-sparsifies the subsets → back to 0.500).

## What this establishes (the diagnosis validated + a working lever)
The CYCLE-958 diagnosis is confirmed and acted on: the fully-spiking reasoner's accuracy limit is read variance, and TRUE population coding — INDEPENDENT (diverse-subsampling) readers, not identical ones — is the correct read-out-architecture lever. It lifts the fully-spiking spiking-inheritance accuracy +18% (6-seed), robustly, with all anti-cheats collapsing. This closes the "cheap levers exhausted → needs a read-out architecture" thread with a WORKING mechanism rather than a named next-step.

Honest scope: the lift is modest (+18%, 0.458→0.542) — it does NOT fully close the gap to the numpy-reason path (0.865 INHERIT); the remaining gap is the codon-ASSIGNMENT variance itself (a held-out member's SDR→columns mapping), which diverse READERS partially average but do not eliminate (the codon it reads is still the same possibly-wrong codon). Fully closing it would need codon-assignment robustness (a diverse-ENCODING / ensemble-pooler mechanism), the next rung. The `--diverse-readers`/`--prop-k` defaults are off/2 (byte-identical); the fully-spiking reason+speak runner can opt into diverse readers for the improved accuracy.

## Files
`research/runners/_realcorpus_inheritance_rung2_spiking_derisk.py` (`--diverse-readers`); 6-seed `research/findings/raw/_rc_spk_dr16_s*.json`. Prior: CYCLE-958 (property-population no-op), CYCLE-964 (codon-width wrong-way); the fully-spiking reason+speak `2026-07-08-knowledge-half-FULLY-SPIKING-reason-speak-one-brain-GO.md`.
