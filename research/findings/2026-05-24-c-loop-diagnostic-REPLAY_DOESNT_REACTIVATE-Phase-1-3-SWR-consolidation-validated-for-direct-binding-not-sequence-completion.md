# (c) loop diagnostic probe — DEFINITIVE: REPLAY_DOESNT_REACTIVATE; Phase 1.3 SWR consolidation (validated for DIRECT-BINDING) does NOT transfer sequence-specific structure from CA3 engrams to cortical schema in a way the parallel-matching decoder can read; pivot to direction 3 (larger vocab scaling on bio_brain_regions) (2026-05-24)

## What was tested

Post-NEGATIVE diagnostic probe to localise the (c) generative-replay loop's failure mode. After:
- (c) decisive multi-seed NEGATIVE (pillar n=99; 104/1800=5.78% vs chance 6.25%)
- v2 smoke with reviewer's top-2 leverage fixes (stim_tag + global-mean centring) ALSO NEGATIVE (1/50=0.020 below chance)

The reviewer's CLEAR verdict on n=99 + the v2 NEGATIVE signal indicate the failure mode is NOT free-running SWR or grounding-mismatch but something deeper. The diagnostic probe directly measures whether the SWR-stim-driven cortical activity is similar to the encoded engram cortex pattern.

Runner: `research/findings/raw/c_loop_diagnostic_probe.py` (CPU-only; ~5 min; reuses substrate cache from pillar n=98 + v2 stim_tag primitive byte-unchanged).

## Method

For each of K=4 stored sequences:
1. Encode sequence as engram (drive lang_input across 3 slot-words; commit_engram_tag with top_k=100 region_filter=ca3)
2. Capture the cortex activity DURING encoding (this is the "encoded cortex pattern" for that engram)
3. After all 4 engrams encoded, run Phase 1.3 consolidation (8 replays per tag, validated mechanism)
4. For each engram: trigger_swr_replay_with_stim(tag) → capture post-replay cortex → measure cosine similarity to:
   - (A) the correct stored engram's encoded cortex pattern
   - (B) other stored engrams' encoded cortex patterns (control)
   - (C) baseline no-drive cortex activity (control)

## Result: REPLAY_DOESNT_REACTIVATE

Aggregate over K=4 engrams (seed 42):

| Comparison | Mean cosine similarity |
|---|---|
| Post-replay cortex → CORRECT engram cortex pattern | **0.086** |
| Post-replay cortex → OTHER engrams (control) | **0.080** |
| Post-replay cortex → BASELINE no-drive (control) | **0.095** |

Per-engram breakdown:
- Engram 0 ([stop, east, look]): correct 0.079; other 0.057; baseline 0.019 (selective ~0.022)
- Engram 1 ([east, look, big]): correct 0.049; other 0.088; baseline 0.118 (ANTI-selective)
- Engram 2 ([cat, hot, look]): correct 0.098; other 0.075; baseline 0.094 (~at chance)
- Engram 3 ([cat, look, south]): correct 0.120; other 0.102; baseline 0.148 (correct LESS than baseline)

Diagnostic metrics:
- **Selectivity (correct - other) = +0.006** — essentially zero (would need ≥ 0.05 to indicate genuine reactivation)
- **Above-baseline (correct - baseline) = −0.009** — actually BELOW the no-drive baseline (replay drives cortex no closer to the encoded pattern than chance ambient activity)

## The honest finding

**The (c) loop's SWR + stim_tag mechanism is NOT producing a sequence-specific cortical signature.** The grounded-symbol decoder is reading information that isn't there. The pipeline mechanically functions but carries no signal to decode.

**The precise bottleneck**: Phase 1.3 SWR consolidation in its CURRENT CONFIGURATION (the validated mechanism via run_concept_replay_phase + the ca1→cortex pathways from build_biological_brain_regions(enable_hippocampus_consolidation=True)) does NOT transfer sequence-specific structure from CA3 engrams to the cortical schema in a way that the parallel-matching decoder can read.

This is consistent with Phase 1.3's validation scope: it was validated for DIRECT-BINDING tasks (W→A binding via ca1→motor pathway; 3/3 strict anti-cheat multi-seed). For sequence-completion tasks (cue partial slots → decode missing slot), the consolidation mechanism would need to transfer SLOT-POSITION-DISTINCT patterns, not just engram-tag-distinct patterns. The existing consolidation pathways carry engram-tag identity, not slot-position structure.

## Biology-translatable insight

Cortical pattern completion via hippocampal SWR replay works in biology (Schwartenbeck 2023; Liu 2012 inception of fear) for STORED-EPISODE-REACTIVATION when the cortical schema has the relevant structure consolidated. The validated Phase 1.3 mechanism captures DIRECT-PAIR consolidation (one input → one output via the ca1→motor pathway). For multi-slot sequence completion, biology requires ADDITIONAL substrate elements:

- Slot-position-distinct ensembles (the project's ec_context substrate at catalog D.01/D.02/D.11 — positional binding — built but not in the (c) loop's path)
- Sequence-replay mechanism (forward/reverse replay during SWR — not the same as the engram-bound consolidation)
- Schema with slot-structure (cortex needs to have learned the SEQUENCE pattern, not just the SUM-pattern engram tag)

These are KNOWN biological mechanisms but they need integration. The (c) loop as currently designed doesn't include them; this is why it fails.

## What's preserved unconditionally

- All five substrate-readiness pillars (n=93/n=94/n=96/n=97/n=98) stand. Their validation is independent of the (c) loop.
- Phase 1.3 SWR consolidation is validated for DIRECT-BINDING tasks (not retracted).
- Parallel-matching mode-unification is validated (not retracted).
- The (c) TDD plan + adversarial review discipline (Tasks 0-4) proved out the subagent-driven build pattern.
- No-confab moat 7/7.

## Pivot direction

The (c) loop arc as currently designed reaches a NATURAL TERMINUS on this substrate. The honest scientific pivot direction is to extend the VALIDATED mode-unification mechanism in directions that don't depend on the broken (c) integration:

### Direction 3 (immediate): larger vocab scaling on bio_brain_regions

Extend the validated OPTION 3/HIPPO/DLPFC substrate from 16 concepts → 32 → 64. Per-concept dynamics may need re-tuning (weak dynamics at higher count; concept-pool size adjustment). Pre-registered: parallel-matching mode-unification must PASS multi-seed ≥ 0.80 on both readouts at each tier.

This is the most leveraged immediate direction: reuses ALL validated infrastructure (the substrate-readiness chain); tests the vocab-scaling axis (the project's stated strength on G.20 sparse — same axis on bio_brain_regions extends biological coverage); each tier is ~1.5-2 hr GPU.

### Direction 4 (next): cross-bridge bio_brain_regions composition

Train multiple bio_brain_regions bridges on different vocab categories (mirroring G.20 sparse's 5-bridge structure). Test cross-bridge mode-unification on the union. Extends validated mode-unification to multi-substrate composition.

### Honest note on (c) loop

The (c) loop is NOT permanently abandoned; the substrate-readiness chain demonstrates the COMPONENTS are valid. A future revision that adds the missing biological mechanisms (slot-position-distinct ensembles + sequence-replay + slot-structured schema) could turn the NEGATIVE into a PASS. But that's a substantial multi-week build; the NEXT productive direction per overnight autonomy is to extend the VALIDATED mechanism along the scaling axes that compound on existing pillars.

## Files

- Diagnostic runner: `research/findings/raw/c_loop_diagnostic_probe.py`
- Log: `research/findings/raw/c_loop_diagnostic_probe.log`
- Output JSON: `research/findings/raw/c_loop_diagnostic_probe.json`
- This findings doc: `research/findings/2026-05-24-c-loop-diagnostic-REPLAY_DOESNT_REACTIVATE-Phase-1-3-SWR-consolidation-validated-for-direct-binding-not-sequence-completion.md`
- Parent (c) decisive NEGATIVE pillar n=99: `research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-loop-at-n-iterations-1-doesnt-produce-above-chance-completion-pivot-direction-identified.md`
- Post-(c) direction roadmap: `docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green.
- Frozen 0.80 bar unchanged.
- This finding sharpens the n=99 NEGATIVE pillar's failure-mode diagnosis from "free-running SWR + grounding-mismatch hypothesis" to definitive "REPLAY_DOESNT_REACTIVATE -- Phase 1.3 SWR consolidation doesn't transfer sequence-specific structure".
- Diagnostic finding only; no new pillar; n=99 stands and is sharpened.
- Both git remotes propagated.
