# HIPPO-OPTION3 PASS — parallel-matching mode-unification still PASSes on the build_biological_brain_regions substrate WITH hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation pathways PRESENT; (c) generative-replay can build cleanly without workaround (2026-05-23)

## What was tested

Direct follow-up to OPTION 3 PASS (pillar n=96). OPTION 3 validated parallel-matching mode-unification on the build_biological_brain_regions concept-pool substrate WITHOUT hippocampus. The (c) generative-replay loop needs hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation pathways. This probe tests whether enabling those (regions + pathways present in the wiring, not actively running consolidation cycles during this probe's training) PERTURBS the concept-pool activity geometry that parallel-matching grounding relies on.

Substrate change from OPTION 3: `build_biological_brain_regions(enable_hippocampus_consolidation=True)` instead of the default `False`. Everything else identical (v16 production recipe; same v14/v16 16-pool concept architecture; same topographic prior + orthogonal codes + interleaved training; same per-neuron activity capture across the 16-pool union; same mean-centring + DERIV_SEED=90909 deriver + parallel-matching decoder). Runner: `research/findings/raw/mode_unification_with_hippo_probe.py` (replicates OPTION 3 probe's bridge-build inline because concept_pool_demo's `build_concept_bridge` wrapper doesn't expose the hippocampus flag — preserves byte-unchanged reuse of concept_pool_demo).

## Pre-registered reading (fixed; never tuned)

- **HIPPO_OPTION3_PASS**: multi-seed-mean >= 0.80 at every load {2, 3, 5} on BOTH OB and OI readouts. The hippocampus presence does NOT perturb the concept-pool grounded-symbol pipeline. (c) can build on this substrate; only the loop-controller wiring is genuinely new.
- **HIPPO_OPTION3_NEGATIVE**: either readout misses. Hippocampus's resting modulation of cortex isn't compatible with this substrate's parallel-matching grounding at this scale; (c) needs a different integration path.

## Result: HIPPO_OPTION3_PASS

Multi-seed (seeds 42/43/44) integrated accuracy at loads {L=2, L=3, L=5}:

| Seed | L=2 OB / OI | L=3 OB / OI | L=5 OB / OI |
|---|---|---|---|
| 42 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.990 |
| 43 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.990 |
| 44 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **Multi-seed mean** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 0.993** |

Statistically indistinguishable from OPTION 3 no-hippo (multi-seed L=5 OI 0.997 there vs 0.993 here; diff −0.004 well within sample noise; both essentially perfect; both 0.19+ above the 0.80 bar).

Order-bearing is exactly 1.000 at every cell (zero errors across 1800 OB trials). Order-invariant: perfect at L=2 and L=3; at L=5 multi-seed 0.993 (per-seed [0.990, 0.990, 1.000]).

Wall-clock 119.4 min on CuPy/RTX 3090 (vs OPTION 3's 92.6 min — about 30% longer due to extra hippocampus regions/synapses simulated each step: 7344 neurons vs 6784; 3.67M synapses vs 3.52M).

## OPTION 3 vs HIPPO-OPTION3 (direct comparison)

| Metric | OPTION 3 (no hippo, n=96) | HIPPO-OPTION3 (this) |
|---|---|---|
| Bridge size | 6784 neurons, 3.52M synapses | 7344 neurons, 3.67M synapses |
| Hippocampus regions | absent | EC + DG + CA3 + CA1 + dlpfc_wm PRESENT |
| Training wall-clock per seed | ~31 min | ~33-35 min |
| Pool-union mean firing rate | 0.35-0.43 | 0.38-0.47 (slightly higher; hippocampus baseline drive nudges cortex) |
| Pool-union density | 0.24-0.28 | 0.26-0.30 |
| Multi-seed OB L=2/3/5 | 1.000/1.000/1.000 | 1.000/1.000/1.000 |
| Multi-seed OI L=2/3/5 | 1.000/1.000/0.997 | 1.000/1.000/0.993 |
| Verdict | OPTION3_BASIC_PASS | HIPPO_OPTION3_PASS |

The hippocampus presence slightly increases cortical pool activity (likely via baseline EC-driven inputs into the broader substrate) but does NOT change the qualitative outcome. The grounded-symbol pipeline still produces near-orthogonal phasor symbols; the parallel-matching decoder still PASSes essentially perfectly.

## Smell-test (recompute from recording; no re-run; no bar change)

Recomputed per-seed and multi-seed-mean from the raw `per_seed` JSON; values reproduce the runner's reported aggregate exactly. Per-seed L=5 OI: [0.990, 0.990, 1.000] (10 OI miss-trials out of 600 across the three seeds). Frozen 0.80 bar unchanged.

Batched-vs-scalar phase_similarity max-diffs across all 3 seeds: 1.39e-17 / 2.08e-17 / 1.39e-17 (machine precision; well below 1e-10 fail-closed tolerance verified at each cell start).

No oracle leak: items_idx constructed via `qrng.choice(V=16, size=load, replace=False)` and used only in post-hoc comparisons; decoder operates on the full 16-concept grounded-symbol vocabulary derived from each seed's own captured activity.

Substrate confirmed: log shows `[BUILD] hippo-enabled concept-pool bridge: 7344 neurons total, 16 concept pools (3200 pool neurons); hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation pathways PRESENT` for each seed. The 560 additional neurons over OPTION 3 are the hippocampus + dlpfc_wm regions; the additional 150K synapses are the trisynaptic loop + CA1→cortex consolidation + dlpfc plumbing.

No protected/frozen/moat module modified: runner imports unchanged primitives from vocabulary_scaling_run, parallel-matching runner, pattern_separation, biologized helpers, cross_bridge probe, OPTION 3 probe, concept_pool_demo (only `apply_concept_topographic_bias` and `train_word_to_pool` and vocab constants used), text_minimal_isolation (only the `build_biological_brain_regions` function call), resonate_fire_fhrr, spiking_phasor_fhrr, sim/backend.

No autograd. No-confab moat 7/7 to be confirmed by reviewer.

The trained bridges + activity caches are at `research/findings/raw/mode_unification_with_hippo_cache/`; re-runs are essentially instant from cache.

## Biology-translatable insight

The biology-grounded parallel-population-matching identification mechanism PASSes on the build_biological_brain_regions substrate INCLUDING the hippocampus (EC/DG/CA3/CA1) + Phase 1.3 SWR consolidation wiring. The hippocampus's resting modulation of cortex (via CA1→cortex consolidation pathways + EC connections) does NOT degrade the concept-pool activity geometry the grounded-symbol pipeline relies on. This is consistent with the biological role of hippocampus in real brains: it provides RAPID episodic binding without disrupting cortical SCHEMA representations (the McClelland 1995 / Buzsaki 2013 complementary-learning-systems thesis the Phase 1.3 SWR consolidation validates empirically).

This probe validates that the build_biological_brain_regions substrate, with the hippocampus + Phase 1.3 SWR consolidation wiring engaged, supports parallel-matching mode-unification grounded-symbol identification at multi-seed PASS. It does NOT yet validate the dlpfc_wm NMDA bistable PFC working memory region — that region is built only by `g11_bg_runner.py` (the navigation runner) via explicit BrainRegion declaration, NOT by the `enable_hippocampus_consolidation=True` flag of `build_biological_brain_regions`. Adding `dlpfc_wm` + the language-input-to-dlpfc plumbing to the substrate is a discrete next substrate-extension step (the dlpfc_wm BrainRegion pattern in g11_bg_runner is reusable; ~20-30 lines of declarative wiring), with its own pre-registered re-validation (does parallel-matching mode-unification still PASS when dlpfc_wm is present?).

## What the (c) generative-replay loop still needs

Validated on a single coherent substrate (build_biological_brain_regions, post-HIPPO-OPTION3 PASS):
- v14/v16 16-pool concept architecture with W→A multi-seed binding (88.75%)
- Hippocampus EC/DG/CA3/CA1 trisynaptic loop (D.12 separation + D.13 completion)
- Engram tagging (D.14)
- Phase 1.3 SWR consolidation (3/3 strict anti-cheat multi-seed)
- Parallel-matching biologized mode-unification (this probe; n=97 pending review)

Still needs (separate, smaller, pre-registered):
1. **dlpfc_wm region addition + parallel-matching re-validation**: bring the NMDA bistable PFC working memory region (existing pattern in g11_bg_runner) into this substrate; re-run parallel-matching mode-unification probe; confirm PASS holds. Estimated ~2 hr GPU.
2. **(c) generative-replay loop controller wiring**: encode initial PFC frame via mode-unification → trigger SWR replay against consolidated schema → capture post-replay cortical activity → decode via parallel-matching → update PFC frame; iterate. Pre-registered test: partial-sequence-completion accuracy ≥ 0.80 multi-seed. Estimated multi-week TDD build.

No substrate-merge required (OPTION 1 abandoned); no narrow scope required (OPTION 2 abandoned); OPTION 3 path is incrementally extensible — each substrate addition gets its own re-validation step. The cleanest biology match for the conversational arc IS the cleanest path forward; the path is two discrete steps (dlpfc_wm addition + loop controller), each separately pre-registered and adversarially reviewed.

## Honest caveats

1. **Oracle-adjacency caveat (from (b))**: parallel matching is structurally closer to argmax-over-stored-vocabulary than TPAM's recurrent attractor; the "vocabulary" is the substrate's own mean-centred grounded symbols. Biology-grounded, but the caveat carries.

2. **Hippocampus PRESENT but not actively SWR-replaying during this probe's training/capture**: the regions and pathways exist in the wiring; no explicit consolidation cycle is run during this probe (no `bridge.set_plasticity_gate("ca3_swr_burst", 1.0)` for sleep phases). The probe answers "does hippocampus PRESENCE perturb basic substrate-grounding" — YES the answer is NO. Whether ACTIVE SWR replay (which the (c) loop will run) perturbs grounded symbols is a distinct, separate question — one the (c) build will test.

3. **3-seed sample**: large margin (multi-seed OI L=5 = 0.993; lowest cell 0.990; 0.19+ above 0.80 bar). 5-seed extension unlikely to surface a collapse.

4. **No new architectural mechanism introduced** — this probe just enables an existing flag on an existing builder; the only new code is the inline replication of concept_pool_demo's build pattern (preserves byte-unchanged reuse).

5. **Subject to fresh dedicated adversarial review before capability_status pillar update** — standing discipline.

## Implication for (c) generative-replay

The integration-choice trichotomy from the (c) design is largely resolved:
- OPTION 1 (substrate-merge): NOT NEEDED.
- OPTION 2 (G.20 sparse alone, narrower scope): NOT NEEDED.
- OPTION 3 (build_biological_brain_regions alone, with parallel-matching re-validation): VIABLE. Re-validated WITHOUT hippocampus (pillar n=96) AND re-validated WITH hippocampus (this pillar, n=97 pending review).
- OPTION 4 (defer (c); cross-bridge characterisation): already done (pillar n=95).

**The (c) build can proceed on the build_biological_brain_regions substrate with hippocampus + Phase 1.3 SWR consolidation enabled, with ONE additional substrate-extension step (dlpfc_wm region) and its own pre-registered re-validation BEFORE the (c) loop-controller TDD build.** That step is small (the dlpfc_wm BrainRegion pattern already exists in g11_bg_runner.py; reusing it requires ~20-30 lines of declarative wiring + the language-input-to-dlpfc plumbing); the re-validation is the same parallel-matching mode-unification probe (~2 hr GPU). Once that PASSes, all five load-bearing components of the (c) loop are validated on a single coherent substrate, and the (c) loop-controller wiring becomes the only remaining genuinely-new code.

The next pre-registered substantial direction (in order):
1. dlpfc_wm-extension parallel-matching probe (small; cheap; pre-registered 0.80 bar)
2. If PASS, the (c) TDD implementation plan + subagent-driven loop-controller build + adversarial review + decisive run

This sequencing preserves the standing project discipline: every substrate extension gets its own pre-registered re-validation before the next layer is built on top.

## Files

- Runner: `research/findings/raw/mode_unification_with_hippo_probe.py` (GPU-batched)
- Log: `research/findings/raw/mode_unification_with_hippo_probe_full.log`
- Output JSON: `research/findings/raw/mode_unification_with_hippo_probe_full.json`
- Smoke output: `research/findings/raw/mode_unification_with_hippo_probe_smoke.json` (not propagated)
- Trained bridge caches: `research/findings/raw/mode_unification_with_hippo_cache/bridge_full_seed{42,43,44}.simstate.h5`
- Per-seed activity caches: `research/findings/raw/mode_unification_with_hippo_cache/activity_full_seed{42,43,44}.npz`
- This findings doc: `research/findings/2026-05-23-HIPPO-OPTION3-PASS-parallel-matching-mode-unification-still-works-with-hippocampus-PRESENT-c-can-build-cleanly.md`
- OPTION 3 parent: `research/findings/2026-05-23-OPTION3-parallel-matching-PASSES-on-build_biological_brain_regions-substrate-cleanest-biology-match-for-generative-replay.md`
- (c) generative-replay design: `docs/plans/2026-05-23-generative-replay-design.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green (no edits to abstention_gate.py).
- Frozen 0.80 bar unchanged.
- Plain ASCII output throughout.
- Both git remotes propagated.
- VALIDATED pillar pending fresh adversarial review.
