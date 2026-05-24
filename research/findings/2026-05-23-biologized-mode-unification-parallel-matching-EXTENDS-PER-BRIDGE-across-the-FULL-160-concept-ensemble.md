# Biologized mode-unification (parallel-matching) EXTENDS PER-BRIDGE across the FULL 160-concept ensemble — ENSEMBLE PASS multi-seed on BOTH readouts (2026-05-23)

## What was tested

The (b) VALIDATED capability (biologized mode-unification via parallel-population-matching identification; both readouts PASS multi-seed at 32 concepts; capability_status pillar n=93) was tested on `bridgeA_nouns` only. This extension runs the IDENTICAL pre-registered runner logic — byte-unchanged via direct primitive re-use — across the OTHER 4 bridges of the 160-concept ensemble (`bridgeB_verbs`, `bridgeC_adj`, `bridgeD_spatial`, `bridgeE_functional`), characterising whether the biologized mode-unification both-readouts capability holds per-bridge across the full ensemble.

Substrate: the per-bridge trained activity caches produced by the 160-ensemble decisive 9-hour GPU run (`vocabulary_scaling_160ensemble_cache/full_<bridge>_seed{42,43,44}.npz`). CPU-only (no new GPU run; reuses every cache; reuses the parallel-matching primitives byte-unchanged).

Runner: `research/findings/raw/biologized_mode_unification_parallel_matching_5bridge_extension.py` (215 lines; net-new orchestration; mirrors the (b) per-trial loop for cross-bridge iteration; reuses `BAR`, `LOADS`, `SEEDS`, `N_DIM`, `N_TRIALS`, `_load_cache` from `vocabulary_scaling_run`; `K_VOCAB_TARGET`, `DERIV_SEED`, `_ground_symbols` from the parallel-matching runner; `gamma_slot_positions` from the mode-unification helpers; `ResonateFireFHRR` and `phase_similarity` from the FHRR-biologization arc — every import byte-unchanged).

## Pre-registered reading (fixed; never tuned)

- **ENSEMBLE_PASS**: every (bridge, load) cell across 5 bridges × 3 loads = 15 cells multi-seed-mean ≥ 0.80 on BOTH order-bearing AND order-invariant readouts (30 cells total). The biologized mode-unification both-readouts capability via parallel-population-matching extends per-bridge across the full 160-concept ensemble.
- **BOUNDARY**: some bridge or load misses; per-bridge breakdown reported honestly (similar to the 160-ensemble decisive run's bridgeD_spatial miss at TPAM).

## Result: ENSEMBLE_PASS_PARALLEL_MATCHING_ALL_5_BRIDGES

Per-bridge multi-seed (42/43/44) integrated accuracy at loads {L=2, L=3, L=5}:

| bridge | L=2 OB / OI | L=3 OB / OI | L=5 OB / OI |
|---|---|---|---|
| bridgeA_nouns | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.982 |
| bridgeB_verbs | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.987 |
| bridgeC_adj | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.960 |
| bridgeD_spatial | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.978 |
| bridgeE_functional | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.978 |

Every cell across 30 cells (15 per readout) clears the frozen 0.80 bar multi-seed. Order-bearing is exactly 1.000 at every cell (zero errors across 9000 order-bearing trials). Order-invariant is exactly 1.000 at L=2 and L=3 at every cell; at L=5 it ranges 0.960 (bridgeC) to 0.987 (bridgeB) — every cell clears.

Per-seed lowest cell across all 15 cells × 200 trials: 0.945 (bridgeC seed 43, L=5 OI). Even the weakest single-seed cell is well above bar.

Total trials evaluated: 15 cells × 3 loads × 200 trials = 9000 per readout; 18000 total. Wall-clock 911.5 s (15.2 minutes; mean 60.8 s per cell, matching the (b) per-seed cost).

## Mandatory anti-cheat smell-test (scrutinise a nominal PASS harder than a FAIL)

Recomputed the per-bridge per-load multi-seed-mean from the raw `cell_results` in the output JSON without re-running anything; the recomputation reproduces the runner's reported aggregate exactly. The bar is unchanged (0.80; pre-registered in `vocabulary_scaling_run.py:BAR`). The decoder labels match the (b) VALIDATED runner verbatim (`parallel_population_matching` for OB; `marginal_sum_phase_similarity` for OI).

**Byte-identical reproduction of (b) on bridgeA_nouns**: the extension's bridgeA cells multi-seed-mean OB 1.000/1.000/1.000 and OI 1.0000/1.0000/0.9817 reproduce the (b) VALIDATED capability pillar n=93's reported values exactly. The byte-unchanged pipeline reuse is verified at the result level.

**Decoder genuinely matters**: at the 160-ensemble decisive run, TPAM gave bridgeD_spatial 0.78 / 0.77 / 0.74 (miss at every load). Parallel-matching on the SAME bridgeD cache, with the SAME substrate, the SAME grounded symbols, the SAME encoded composites, and the SAME gamma-slot positions, gives bridgeD 1.000 / 1.000 / 0.978 — clears at every load. The decoder substitution (parallel-matching vs TPAM) is the load-bearing difference, exactly as the (b) post-PASS adversarial review's diagnostic predicted.

**No oracle leak**: the per-trial decoder argmaxes over each bridge's OWN 32-word vocabulary; the true item indices are NEVER passed to the decoder; the encoding uses `qrng.choice(words, ...)` and decoding reads `scores = [phase_similarity(unbinds[k], grounded[w]) for w in words]`. The pipeline reads, from the runner's own source, identically to (b)'s reviewed pipeline.

**No protected/frozen/moat module modified**: the extension probe imports from existing modules; introduces zero edits to `sim/`, `tests/`, the FHRR-biologization arc's modules, or `research/runners/abstention_gate.py`. The no-confab moat stays 7/7.

**Per-bridge variation honestly characterised**: bridgeC_adj L=5 OI 0.960 is the weakest per-bridge multi-seed-mean (still well above bar). Per-seed lowest cell 0.945 (bridgeC seed 43). The pattern is consistent with FHRR's L=5 capacity-edge tail behaviour previously characterised in the algebra-PASS / capacity-envelope arc.

**No new GPU run**: every cell reuses the per-bridge per-seed cache from the 160-ensemble decisive 9-hour run (path: `research/findings/raw/vocabulary_scaling_160ensemble_cache/full_<bridge>_seed{seed}.npz`). The CPU-only extension probe is a thin orchestration loop over the (b) primitives.

## Honest caveats preserved

1. **Oracle-adjacency caveat (from (b))**: the parallel-matching decoder is structurally closer to "argmax over a stored vocabulary" than TPAM's recurrent attractor. The "vocabulary" here is the substrate's own derived grounded symbols (mean-centred consolidated activity → fixed-seed deriver → spike-phase rep) on EACH per-bridge cache. Biology-grounded mechanism (feedforward similarity comparison via dendritic integration + lateral-inhibition WTA), not engineered table lookup; caveat recorded.

2. **Per-bridge, NOT cross-bridge**: each bridge's mode-unification is tested on its own 32-word vocabulary. This is the per-bridge framing the (b) runner pre-registered. Cross-bridge composition across all 160 concepts (one encoded composite drawing from multiple bridges' vocabularies) remains a distinct, still-open direction the 160-ensemble decisive-run findings explicitly bracketed.

3. **3-seed sample**: matches the (b) runner's seed set (42, 43, 44) and the 160-ensemble decisive run's primary sample. A 5-seed extension would tighten the multi-seed variance characterisation further but the multi-seed 0.96-0.99 OI margins suggest no surprise per-bridge collapses likely (compare to the seed-46 L=5 collapse on TPAM in the 5-seed extension — that pattern is absent at parallel-matching on these 3 seeds).

4. **Subject to a fresh dedicated adversarial review before any capability-pillar claim** — the standing discipline.

## Biology-translatable insight

The biology-grounded identification mechanism (parallel-population matching: feedforward similarity comparison across a population of neurons each tuned to one stored grounded symbol + lateral-inhibition winner-take-all) realises the catalog-documented Lisman-Idiart N.16 mode-unification both-readouts capability per-bridge across all 5 categories of the 160-concept ensemble:

- The phase-coded algebra (FHRR; resonate-and-fire encoding) supports unified bidirectional readout from one theta-gamma encoded code (algebra-PASS multi-seed 1.000; capacity envelope wide on load/noise/vocab).
- The biologized order-invariant readout (marginal-sum of per-slot phase-similarities, top-K) PASSes per-bridge across all 5 bridges at every tested load.
- The biologized order-bearing readout via parallel-population matching (this extension) PASSes per-bridge across all 5 bridges at every tested load.
- The FHRR-biologization arc's TPAM attractor, by contrast, has a non-monotonic V=8 through V=20 capacity window on grounded symbols; at 32 concepts per bridge it sits above the window and misses at every load on bridgeD_spatial (BOUNDARY pillar n=92).

**Two honest biologizations now stand side-by-side**, each with its precise scaling property:
- TPAM (BOUNDARY pillar n=92; recurrent Hopfield-class attractor; non-monotonic V=8-V=20 capacity window; biologically faithful to cortical attractor dynamics; ceilings at 32-concept full-vocabulary per-slot identification).
- Parallel-matching (this extension; biology-grounded feedforward + WTA; scales per-bridge across all 5 categories of the 160-concept ensemble at multi-seed 0.96-1.00 on both readouts; structural-proximity-to-argmax caveat preserved).

The catalog-named cortical attractor mechanism has a hard capacity limit at this vocabulary scale on substrate-grounded symbols; the alternative biology-grounded mechanism does not. Both findings are biology-translatable: cortical computation provides BOTH mechanisms (attractor refinement of low-cardinality patterns; feedforward population WTA for high-cardinality identification); their division of labor in real cortex is itself biology-translatable evidence.

## Relation to the standing conversational-path reframe

The mode-unification thread is the second leg of the owner's 2026-05-19 conversational-path reframe (SPEAR → theta-gamma mode-unification → generative replay). With this extension:

- Algebra-PASS first (cheap numpy probe; multi-seed 1.000)
- Capacity-envelope characterisation (wide on three axes; algebra survives substrate-realistic noise)
- Biologized spiking implementation:
  - TPAM (BOUNDARY pillar n=92; V=8-V=20 capacity window)
  - Parallel-matching on bridgeA (VALIDATED pillar n=93)
  - **Parallel-matching across all 5 bridges (THIS extension — the natural completion of the thread)**
- Generative replay (third leg; design doc landed at commit 97f21c5; implementation pending an architecture-integration choice surfaced separately)

The PFC compositional frame the generative-replay loop requires (an ordered K-tuple of bound concepts at specific gamma slots) is now demonstrably decodable per-bridge across the full 160-concept ensemble via the biology-grounded parallel-matching mechanism. This is the load-bearing prerequisite for the conversational generative-replay loop the third leg of the reframe describes.

## Files

- Runner: `research/findings/raw/biologized_mode_unification_parallel_matching_5bridge_extension.py`
- Log: `research/findings/raw/biologized_mode_unification_parallel_matching_5bridge_extension.log`
- Output JSON: `research/findings/raw/biologized_mode_unification_parallel_matching_5bridge_extension.json`
- This findings doc: `research/findings/2026-05-23-biologized-mode-unification-parallel-matching-EXTENDS-PER-BRIDGE-across-the-FULL-160-concept-ensemble.md`
- (b) parent finding: `research/findings/2026-05-23-biologized-mode-unification-PASS-via-parallel-population-matching-VALIDATED-with-oracle-adjacency-caveat.md`
- (b) parent runner: `research/findings/raw/biologized_spiking_mode_unification_parallel_matching_runner.py`
- TPAM contrast: `research/findings/2026-05-23-biologized-spiking-mode-unification-decisive-NEGATIVE_ORDER_INVARIANT_ONLY-TPAM-attractor-doesnt-transfer-to-per-slot-mode-unification.md`
- 160-ensemble decisive run that produced the trained caches: `research/findings/2026-05-23-160-concept-ensemble-K16-BOUNDARY-4-of-5-bridges-PASS-multiseed-bridgeD-uniquely-misses-with-honest-perseed-caveats.md`
- Capability_status: pillar n=93 currently records the bridgeA-only VALIDATED result; pending dedicated adversarial review on this extension to upgrade or add a new pillar.

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff this commit.
- No autograd anywhere in the extension probe.
- No-confab moat 7/7 green (no edits to `research/runners/abstention_gate.py` or `tests/test_abstention_gate.py`).
- Frozen 0.80 bar unchanged.
- Plain ASCII output throughout.
- Both git remotes propagated.
- This result is subject to a dedicated adversarial review before any capability_status pillar update; the smell-test above is the runner-internal scrutiny, not the independent review.
