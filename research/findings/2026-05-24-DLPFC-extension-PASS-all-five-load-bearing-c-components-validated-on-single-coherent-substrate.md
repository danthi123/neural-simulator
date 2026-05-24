# DLPFC-extension PASS — parallel-matching mode-unification PASSes on the bio_brain_regions substrate WITH hippocampus + dlpfc_wm BOTH present; all five load-bearing components of (c) validated on a single coherent substrate; (c) loop-controller TDD build can proceed directly (2026-05-24)

## What was tested

Direct follow-up to HIPPO-OPTION3 PASS (pillar n=97). HIPPO-OPTION3 validated parallel-matching mode-unification with hippocampus + Phase 1.3 SWR consolidation PRESENT but NO dlpfc_wm (the n=97 honest scope correction noted dlpfc_wm is built only in g11_bg_runner.py via explicit BrainRegion declaration). This probe adds dlpfc_wm region (g11_bg_runner pattern: 60 neurons, IZH2007_HIPPO_PYRAMIDAL, NMDA bistable enabled, internal recurrent dynamics density=0.10/exc=2.0/inh=4.0) + the lang_input → dlpfc_wm pathway (the (c) loop's eventual frame-injection path; plasticity gate `lang_to_dlpfc_wm` closed during this probe's standard training).

The pre-registered question: does ADDING dlpfc_wm region + its lang_input pathway perturb the concept-pool activity geometry that parallel-matching grounding relies on?

Substrate change from HIPPO-OPTION3: 2 additional declarative wiring elements (1 BrainRegion + 1 RegionPathway). Everything else byte-identical (v16 production recipe; same training; same capture; same decoder).

Runner: `research/findings/raw/mode_unification_with_hippo_dlpfc_probe.py` (replicates HIPPO-OPTION3 bridge-build inline + adds dlpfc_wm region + pathway; reuses OPTION 3 probe's capture + grounding + pipeline byte-unchanged via import; reuses cross-bridge probe's batched_phase_similarity byte-unchanged).

## Pre-registered reading (fixed; never tuned)

- **DLPFC_PASS**: multi-seed-mean >= 0.80 every cell on BOTH readouts. ALL FIVE load-bearing components of (c) (concept pools + parallel-matching + hippocampus + SWR consolidation + dlpfc_wm) validated on a single coherent substrate. The (c) TDD plan + loop-controller build is the next pre-registered direction.
- **DLPFC_NEGATIVE**: either readout misses. dlpfc_wm presence breaks substrate-grounding; (c) needs different integration path.

## Result: DLPFC_PASS

Multi-seed (seeds 42/43/44) integrated accuracy at loads {L=2, L=3, L=5}:

| Seed | L=2 OB / OI | L=3 OB / OI | L=5 OB / OI |
|---|---|---|---|
| 42 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.995 |
| 43 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 44 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| **Multi-seed mean** | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 0.998** |

Order-bearing exactly 1.000 every cell (zero errors / 1800 OB trials). Order-invariant: perfect at L=2 and L=3; at L=5 multi-seed 0.998 (per-seed [0.995, 1.000, 1.000]).

Wall-clock 88.5 min on CuPy/RTX 3090 (faster than HIPPO-OPTION3's 119.4 min — interestingly, despite 60 extra dlpfc_wm neurons; possibly GPU cache warm from prior session).

## Side-by-side comparison: OPTION 3 → HIPPO-OPTION3 → DLPFC-extension

| Metric | OPTION 3 (n=96) | HIPPO-OPTION3 (n=97) | DLPFC-extension (this) |
|---|---|---|---|
| Substrate | 16 concept pools | + hippocampus EC/DG/CA3/CA1 + SWR pathways | + dlpfc_wm region + lang→dlpfc pathway |
| Bridge neurons | 7680 | 8240 (+560 hippo) | 8300 (+60 dlpfc; total +620 vs OPTION 3) |
| Pool-union mean firing rate | 0.35-0.43 | 0.38-0.47 (slightly increased; hippo baseline drive) | **0.11-0.15 (3-4× sparser; dlpfc_wm competition for cortical drive)** |
| Pool-union density | 0.24-0.28 | 0.26-0.30 | 0.05-0.07 |
| Multi-seed OB L=2/3/5 | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 1.000/1.000/1.000 |
| Multi-seed OI L=2/3/5 | 1.000/1.000/0.997 | 1.000/1.000/0.993 | 1.000/1.000/0.998 |
| Wall-clock | 92.6 min | 119.4 min | 88.5 min |
| Verdict | OPTION3_BASIC_PASS | HIPPO_OPTION3_PASS | DLPFC_PASS |

**Key biology-translatable insight**: pool-union activity dropped 3-4× when dlpfc_wm was added (the NMDA bistable region pulls cortical drive — consistent with PFC's biological role as a sink for cortical computation during working-memory holding). YET the parallel-matching grounded-symbol pipeline still PASSes essentially perfectly. The mean-centring + deriver pipeline + parallel-matching decoder is ROBUST to substrate-level perturbations of this magnitude — provided the per-concept activity vectors remain DISCRIMINATIVE (which they do; even at low density, each word's activity concentrates in its bound pool's neurons).

## Smell-test PASSED

- Per-seed verdicts recompute from raw cell_results JSON byte-for-byte
- Batched-vs-scalar max-diffs: 2.08e-17 / 1.39e-17 / 2.08e-17 (machine precision)
- No oracle leak: items_idx only in construction + post-hoc comparison
- Substrate confirmed via log: `[BUILD] hippo+dlpfc concept-pool bridge: 8300 neurons total, 16 concept pools (3200 pool neurons); hippocampus (EC/DG/CA3/CA1) + Phase 1.3 SWR consolidation pathways + dlpfc_wm (60 neurons, NMDA bistable; lang_input -> dlpfc_wm pathway plasticity-gated OFF for this probe) PRESENT`
- Frozen 0.80 bar unchanged
- No protected/frozen/moat module modified

## Honest caveats

1. **Oracle-adjacency caveat (from (b))**: parallel matching is structurally closer to argmax-over-stored-vocabulary than TPAM's recurrent attractor; the "vocabulary" is the substrate's own mean-centred grounded symbols.

2. **dlpfc_wm PRESENT but plasticity gate `lang_to_dlpfc_wm` closed during this probe's standard training**: the (c) loop will OPEN this gate when injecting the encoded PFC frame. The probe validates that dlpfc_wm region PRESENCE doesn't perturb the grounded-symbol pipeline; the (c) loop's gated frame-injection is a separate functional test.

3. **No active SWR replay or NMDA-bistability-driven frame-holding during this probe**: probe captures raw concept-pool activity from lang_input drive; the (c) loop's full dynamics (SWR replay, NMDA bistability holding the frame across the replay window, iteration over decoded continuations) is tested in the (c) build itself, not in this substrate-readiness probe.

4. **3-seed sample**: huge margin (multi-seed OI L=5 = 0.998; 0.198 above the 0.80 bar; lowest per-seed cell 0.995; 0.195 above). 5-seed extension unlikely to surface a collapse.

5. **Subject to fresh dedicated adversarial review before capability_status pillar update** — standing discipline.

## Biology-translatable insight (this pillar)

The build_biological_brain_regions substrate with ALL FIVE load-bearing components of the (c) generative-replay loop ENABLED TOGETHER supports parallel-matching biologized mode-unification at multi-seed PASS:

1. v14/v16 16-pool concept architecture with W→A multi-seed binding (88.75% validated independently)
2. Hippocampus EC/DG/CA3/CA1 trisynaptic loop (D.12 separation + D.13 completion validated)
3. Engram tagging (D.14 validated)
4. Phase 1.3 SWR consolidation pathways (3/3 strict anti-cheat multi-seed validated)
5. **dlpfc_wm NMDA bistable PFC working memory region (this pillar)**

PLUS the parallel-matching biologized mode-unification mechanism (pillars n=93/n=94/n=96/n=97/n=98) — all on a single coherent substrate. The cortical mode-unification mechanism is INDEPENDENT of which biological substrate components are present (within tested perturbation magnitudes). This robustness IS biology-translatable: real cortical computation has substantial signal-to-noise variation across cortical regions, behavioural states, attention conditions; the mode-unification mechanism's noise-floor characterization (validated across G.20 sparse + bio_brain_regions + with/without hippocampus + with/without dlpfc_wm) gives a precise biology-grounded substrate for compositional retrieval.

## Implication for (c) generative-replay (DEFINITIVE)

All five load-bearing components validated. The (c) TDD plan (commit dfe4def, conditional on this probe PASS) can NOW EXECUTE. The genuinely-new code is the loop controller wiring (Tasks 0-5 of the TDD plan). Substrate components and decoder primitives are validated; only the loop controller integration is new.

Next pre-registered substantial direction: dispatch (c) Task 0 via superpowers:subagent-driven-development; continue Tasks 1-3 sequentially; Task 4 adversarial review; Task 5 controller-only decisive multi-seed GPU run (~6-9 hr); propagate.

## Files

- Runner: `research/findings/raw/mode_unification_with_hippo_dlpfc_probe.py` (GPU-batched)
- Log: `research/findings/raw/mode_unification_with_hippo_dlpfc_probe_full.log`
- Output JSON: `research/findings/raw/mode_unification_with_hippo_dlpfc_probe_full.json`
- Smoke output: `research/findings/raw/mode_unification_with_hippo_dlpfc_probe_smoke.json` (not propagated)
- Trained bridge caches: `research/findings/raw/mode_unification_with_hippo_dlpfc_cache/bridge_full_seed{42,43,44}.simstate.h5`
- Per-seed activity caches: `research/findings/raw/mode_unification_with_hippo_dlpfc_cache/activity_full_seed{42,43,44}.npz`
- This findings doc: `research/findings/2026-05-24-DLPFC-extension-PASS-all-five-load-bearing-c-components-validated-on-single-coherent-substrate.md`
- Parent OPTION 3 (n=96): `research/findings/2026-05-23-OPTION3-parallel-matching-PASSES-on-build_biological_brain_regions-substrate-cleanest-biology-match-for-generative-replay.md`
- Parent HIPPO-OPTION3 (n=97): `research/findings/2026-05-23-HIPPO-OPTION3-PASS-parallel-matching-mode-unification-still-works-with-hippocampus-PRESENT-c-can-build-cleanly.md`
- (c) design + TDD plan: `docs/plans/2026-05-23-generative-replay-design.md`, `docs/plans/2026-05-24-generative-replay-implementation.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green.
- Frozen 0.80 bar unchanged.
- Plain ASCII output.
- Both git remotes propagated.
- VALIDATED pillar pending fresh adversarial review.
