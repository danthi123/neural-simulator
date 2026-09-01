---
type: finding
status: partial
date: 2026-09-01
mechanism: metacog margin-NORMALIZED (scale-invariant) confidence band — calibration-at-scale for confidence-forthcomingness on the shipped 15k-KB
lane: introspection
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: 6-seed real-handler measurement; the byte-identical-off failure below is deterministic (a code-path change), reproduced on every seed.
artifacts:
  - research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration_6seed.json
  - research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json
external: NO-EXTERNAL-NEEDED — an internal calibration measurement; the scale/margin motivation was externally RAG-grounded (arXiv pointer logged in .external_searches).
---

# Confidence calibration-at-scale: margin-normalization makes confidence DISCRIMINATE on the real 15k-KB (6/6) — but the implementation is NOT byte-identical-off, so it is NOT flip-ready yet

**Artifacts:** `research/findings/raw/_confidence_kb_relation_realtraffic/verify_margin_norm_recalibration_6seed.json` (discriminates 6/6 on the real KB) + `research/findings/raw/_confidence_ltm_loadbearing/verify_confidence_ltm_loadbearing.json` (tiny-demo regression UNDEFINED, byte_identical_off False).

## The advance (real)
After the #184 plumbing fix (`57d450fe`, the LTM tier now propagates its confidence trace), confidence-forthcomingness still did not discriminate on the shipped `wikidata_core_15k` because the 15k-entity decode margin reads below the tiny-demo-calibrated metacog HIGH band (a margin-vs-scale mismatch). This arc's fix: a **margin-NORMALIZED (scale-invariant) band** in `metacog_production_organ.py` (+ the margin read in `rf_phasor_composer.py`/`rich_answer_composer.py`). RESULT — `verify_margin_norm_recalibration_6seed.json`: **`vary_lesion_all_GO: True`, 6/6 seeds** through the real /api/brain-chat handler on the literal shipped KB — a high-confidence recall yields more grounded sentences than a degraded one, lesion-attributable, measurement GO 6/6. The calibration-at-scale residual is SOLVED in approach.

## Why it is NOT flip-ready (the byte-identical-off guard fired)
The tiny-demo regression check (`verify_confidence_ltm_loadbearing.json`) is **UNDEFINED**: load_bearing ✓, lesion_reverts ✓, moat ✓ (all 6 seeds), but **byte_identical_off = False** on all 6. The margin-norm change is NOT gated behind `BRAIN_CONFIDENCE_FORTHCOMING`; it alters the shared margin computation, which now feeds the (just-flipped-ON) source-monitoring / source-provenance honesty hedge, so a flag-OFF reply is no longer byte-identical to the pre-recalibration reply. Per the auto-flip policy (byte-identical-off is required), this is NOT a clean flip and this branch is NOT merged to main (merging as-is would change the default reply). `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` stays False.

## Next rung (the clean path to the flip)
Gate the margin-normalization behind `BRAIN_CONFIDENCE_FORTHCOMING` (so flag-OFF is byte-identical), OR prove the margin change is a strict no-regression improvement to the honesty hedge (re-verify the hedge output on a known-fact battery is unchanged-or-better). Then re-run both verifies (discriminates 6/6 + byte-identical-off) and AUTO-FLIP. The discrimination mechanism is DONE; only the gating/isolation remains.
