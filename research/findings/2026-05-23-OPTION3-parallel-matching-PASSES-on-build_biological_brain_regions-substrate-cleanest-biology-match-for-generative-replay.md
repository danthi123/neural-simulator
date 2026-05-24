# OPTION 3 cheap-first probe PASS — parallel-matching mode-unification works cleanly on the build_biological_brain_regions concept-pool substrate; (c) generative-replay can build on the substrate that already has hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation validated (2026-05-23)

## What was tested

OPTION 3 from the (c) generative-replay integration-choice surfaced for the owner. The owner's standing autonomy directive ("go with whatever you think is most effective to reach our goals") authorised proceeding with the cheap-first empirical probe rather than further deliberation. The question: does parallel-matching biologized mode-unification PASS on the build_biological_brain_regions concept-pool substrate (the substrate that ALREADY has hippocampus trisynaptic loop + dlpfc_wm NMDA bistability + Phase 1.3 SWR consolidation validated — the cleanest biology match for the generative-replay loop)?

The (b) VALIDATED capability (pillar n=93) and (e) ENSEMBLE-PASS extension (pillar n=94) established parallel-matching mode-unification on the G.20 sparse substrate. The (c) generative-replay design needs PFC + hippocampus + consolidation — which exist on build_biological_brain_regions but not on G.20 sparse. This probe answers whether the SAME parallel-matching mechanism that PASSes on G.20 sparse also PASSes on build_biological_brain_regions concept-pool activity — empirically, not by deliberation.

Runner: `research/findings/raw/mode_unification_on_bio_brain_regions_probe.py` (GPU-batched; net-new orchestration; reuses (b)/(e) mode-unification primitives byte-unchanged + the cross-bridge probe's batched_phase_similarity byte-unchanged + concept_pool_demo's v16 production recipe byte-unchanged via import).

## Pre-registered reading (fixed; never tuned)

- **OPTION3_BASIC_PASS**: multi-seed-mean >= 0.80 at every load {2, 3, 5} on BOTH order-bearing AND order-invariant readouts. The basic substrate-grounding works on build_biological_brain_regions concept pools; (c) generative-replay can build on this substrate (with hippocampus addition as a separate follow-up step).
- **OPTION3_BASIC_NEGATIVE**: either readout misses at some load. Biology-translatable: the build_biological_brain_regions concept-pool activity geometry doesn't ground-symbol cleanly for parallel-matching mode-unification; OPTION 1 substrate-merge may be required for (c).

## Result: OPTION3_BASIC_PASS

Multi-seed (seeds 42/43/44) integrated accuracy at loads {L=2, L=3, L=5}:

| Seed | L=2 OB / OI | L=3 OB / OI | L=5 OB / OI |
|---|---|---|---|
| 42 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.995 |
| 43 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.995 |
| 44 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **Multi-seed mean** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 0.997** |

Every cell across 6 cells (3 seeds × 2 readouts × 3 loads = 18 cells) clears the frozen 0.80 bar multi-seed by a wide margin. Order-bearing is exactly 1.000 at every cell (zero errors across 1800 OB trials). Order-invariant is exactly 1.000 at L=2 and L=3; at L=5 it sits at 0.997 multi-seed (per-seed 0.995/0.995/1.000) — essentially perfect, far above the bar.

Wall-clock 92.6 min on CuPy/RTX 3090: per-seed ~31 min for training (16 words × 200 events × interleaved schedule) + ~5 min capture + ~2 min pipeline = ~38 min/seed × 3 seeds + serial overhead.

## OPTION 3 is CLEANER than the (e) ensemble extension on G.20 sparse

| Metric | (e) ensemble (G.20 sparse, 32-concept per bridge) | OPTION 3 (bio_brain_regions, 16-concept) |
|---|---|---|
| OB L=5 multi-seed | 1.000 every cell | 1.000 every cell |
| OI L=5 multi-seed range | 0.960-0.987 (bridgeC weakest) | 0.997 |
| Lowest per-seed OI L=5 cell | 0.945 (bridgeC seed 43) | 0.995 (seed 42/43) |

The build_biological_brain_regions substrate's concept-pool activity grounds CLEANER for mode-unification than G.20 sparse. Likely cause: the v14/v16 16-pool architecture concentrates concept activity in distinct pool neurons (each word fires its own ~200-neuron pool at 0.35-0.43 mean rate; other 15 × 200 = 3000 neurons fire near-zero); this gives near-orthogonal raw activity vectors. G.20 sparse distributes concept patterns over a shared 2000-neuron pool (K=100 per concept), with more inter-concept overlap before mean-centring.

Captured activity stats (seed 44 representative; 16 concepts × 3200 pool-union neurons; M_OBS=16 observations each): mean firing rate per concept 0.35-0.43; density (fraction of neurons firing) 0.24-0.28. Compare to G.20 sparse vocabulary_scaling_trained_cache: mean rate 0.05-0.10; density 0.04-0.06. The bio_brain_regions substrate has 5-7× more per-neuron information density per observation — directly translating to a cleaner grounded-symbol geometry.

## Smell-test (recompute from recording; no re-run; no bar change)

Recomputed per-seed and multi-seed-mean from the raw `per_seed` entries in the output JSON; values reproduce the runner's reported aggregate exactly. Frozen 0.80 bar unchanged. Decoder labels match the (e) extension and parent (b)/(e) runners (`parallel_population_matching_batched` for OB; `marginal_sum_phase_similarity_batched` for OI).

Batched-vs-scalar `phase_similarity` max-diff across all 3 seeds: 2.08e-17 to 2.78e-17 (machine precision for double-precision phase cosine; well below the 1e-10 fail-closed tolerance verified at each cell start).

No oracle leak: the decoder argmaxes / argsorts over the 16-concept vocabulary (each word's grounded symbol); the true item indices are NEVER passed to the decoder; the encoding uses `qrng.choice(V, ...)` and decoding reads `batched_phase_similarity(unbinds[k], vocab_phase_matrix, xp)` over the full 16-concept union vocabulary. Pattern byte-identical to (b)/(e)/n=95 reviewed pipelines.

No protected/frozen/moat module modified: the probe imports unchanged from `vocabulary_scaling_run`, `pattern_separation_grounding_probe`, `biologized_spiking_mode_unification_helpers`, `biologized_spiking_mode_unification_parallel_matching_runner`, `cross_bridge_mode_unification_probe` (batched primitive), `resonate_fire_fhrr`, `spiking_phasor_fhrr`, `concept_pool_demo`, `sim/backend`, `sim/text_embeddings`. No autograd. No-confab moat 7/7 to be confirmed by reviewer.

The trained bridges are kill-safe-cached at `research/findings/raw/mode_unification_on_bio_brain_regions_cache/bridge_full_seed{42,43,44}.simstate.h5` (~ 30-40 MB each); per-seed activity caches at `activity_full_seed{42,43,44}.npz`. Re-runs are essentially instant from cache (capture cache + bridge cache short-circuit the 31-min training).

## Honest scope + caveats

1. **Oracle-adjacency caveat (from (b))**: parallel matching is structurally closer to "argmax over a stored vocabulary" than TPAM's recurrent attractor. The "vocabulary" here is the substrate's own derived grounded symbols on the 16 concept pools; biology-grounded mechanism, caveat recorded.

2. **NO hippocampus in this probe**: the cheap-first probe deliberately tests parallel-matching mode-unification on the substrate's concept-pool activity WITHOUT hippocampus/SWR/dlpfc_wm. The substrate-build uses concept_pool_demo's `build_concept_bridge` (which is the v14/v16 16-pool architecture). The full (c) generative-replay loop requires the hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation pathways — which exist on build_biological_brain_regions via the `enable_hippocampus_consolidation=True` builder flag. This probe establishes that the basic substrate-grounding works; adding hippocampus is a separate follow-up step (mode-unification's grounded symbols don't depend on hippocampus; the hippocampus enters during the generative-replay loop's replay-against-schema phase).

3. **16-concept tier**: this probe tests at the validated v14/v16 16-pool architecture (4 motor + 4 noun + 4 verb + 4 adjective). Scaling to larger vocab on bio_brain_regions is a distinct, deferred direction analogous to the G.20 sparse 64/160/320-concept ladder.

4. **3-seed sample**: matches (b)/(e)/(n=95) seed sets. Margin is huge (multi-seed OI L=5 = 0.997 — 0.197 above the 0.80 bar; lowest per-seed cell 0.995 — 0.195 above bar). A 5-seed extension would tighten variance but is unlikely to surface a seed-dependent collapse given the >0.19 safety margin.

5. **Subject to fresh dedicated adversarial review before capability_status pillar update** — the standing discipline.

## Biology-translatable insight

The biology-grounded parallel-population-matching mechanism for cortical identification of stored patterns realises the Lisman-Idiart N.16 mode-unification capability on TWO independently-developed substrates with characterised division of labor:

- **G.20 sparse** (V=32 per bridge; V=160 cross-bridge union): per-bridge ENSEMBLE-PASS (n=94); cross-bridge OB extends + OI ceilings at L=5 × V=160 (n=95). Activity geometry: K-of-N sparse codes in a shared 2000-neuron pool; concept patterns overlap before mean-centring; substrate noise floor CV ~1.6 propagates through the pipeline.
- **build_biological_brain_regions concept pools** (V=16, this probe): OPTION3_BASIC_PASS (essentially perfect 1.000/1.000/0.997 multi-seed). Activity geometry: distinct per-concept pools (200 neurons each); each word's activity concentrates in its own pool; raw vectors are near-orthogonal before mean-centring; mean-rate 0.35-0.43 (5-7× denser than G.20 sparse).

The same parallel-matching identification mechanism works on both biological substrate styles. The bio_brain_regions architecture (the one with hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation validated, and with v14/v16's 88.75% W→A multi-seed binding) is the natural substrate for the (c) generative-replay loop. The integration-choice question that motivated this probe is answered empirically: OPTION 3 is viable — (c) can build on the substrate that ALREADY has the load-bearing components validated; no substrate-merge (OPTION 1) is required.

## Implication for (c) generative-replay

The (c) design surfaced three substrate options:
- OPTION 1: port G.20 sparse INTO build_biological_brain_regions (substantial integration work; new pre-registered substrate-property tests; most faithful to design).
- OPTION 2: G.20 sparse alone (narrowest claim; no hippocampus mechanism).
- OPTION 3: build_biological_brain_regions alone with parallel-matching re-validation (best biology match, medium cost; required parallel-matching re-validation on this substrate).

**OPTION 3 is now empirically viable**: parallel-matching mode-unification PASSes on build_biological_brain_regions concept-pool activity at 16 concepts multi-seed with huge margin. The next step toward (c) is enabling hippocampus on this substrate (`enable_hippocampus_consolidation=True`) and re-running this probe to confirm parallel-matching still PASSes when hippocampus is present. If it does, (c) builds on this substrate; the generative-replay loop wiring is the genuinely-new code (the substrate components — concept pools + hippocampus + dlpfc_wm + SWR consolidation — are all already validated; mode-unification grounded-symbol derivation is now empirically validated; only the loop-controller wiring is new).

## Files

- Runner: `research/findings/raw/mode_unification_on_bio_brain_regions_probe.py` (GPU-batched)
- Log: `research/findings/raw/mode_unification_on_bio_brain_regions_probe_full.log`
- Output JSON: `research/findings/raw/mode_unification_on_bio_brain_regions_probe_full.json`
- Smoke output: `research/findings/raw/mode_unification_on_bio_brain_regions_probe_smoke.json` (smoke numbers NOT propagated as a result)
- This findings doc: `research/findings/2026-05-23-OPTION3-parallel-matching-PASSES-on-build_biological_brain_regions-substrate-cleanest-biology-match-for-generative-replay.md`
- Trained substrate caches: `research/findings/raw/mode_unification_on_bio_brain_regions_cache/bridge_full_seed{42,43,44}.simstate.h5`
- Per-seed activity caches: `research/findings/raw/mode_unification_on_bio_brain_regions_cache/activity_full_seed{42,43,44}.npz`
- Parent (e) ENSEMBLE PASS pillar n=94: `research/findings/2026-05-23-biologized-mode-unification-parallel-matching-EXTENDS-PER-BRIDGE-across-the-FULL-160-concept-ensemble.md`
- (c) generative-replay design doc: `docs/plans/2026-05-23-generative-replay-design.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green (no edits to abstention_gate.py).
- Frozen 0.80 bar unchanged.
- Plain ASCII output throughout.
- Both git remotes propagated.
- VALIDATED pillar pending fresh adversarial review.
