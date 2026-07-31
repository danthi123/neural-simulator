---
type: finding
status: live
date: 2026-05-24
mechanism: generative-replay
---

# (c) generative-replay decisive multi-seed run — HONEST NEGATIVE — loop at n_iterations=1 doesn't produce above-chance partial-sequence completion on the validated substrate; pivot direction identified (2026-05-24)

## What was tested

The (c) TDD plan's Task 5: controller-only decisive multi-seed GPU run on the validated dlpfc-extension substrate (pillar n=98). Pre-registered framing per the (c) design doc:

- **PASS**: multi-seed-mean ≥ 0.80 at every K in {4, 8, 16} stored sequences. Validates the biology-grounded generative-replay loop produces partial-sequence completion at multi-seed margin.
- **NEGATIVE**: any K below the bar. Honest finding about which integration property fails to scale.

Substrate (validated by pillars n=96/n=97/n=98): build_biological_brain_regions with concept pools (v14/v16 16-pool) + hippocampus EC/DG/CA3/CA1 trisynaptic + Phase 1.3 SWR consolidation pathways + dlpfc_wm NMDA bistable region. Cached from `mode_unification_with_hippo_dlpfc_cache/bridge_full_seed{42,43,44}.simstate.h5`.

Decisive run: 3 seeds × 3 K-values × 200 trials = 1800 trials total. Slot count = 3 (initial cue = first 2 slots; loop decodes slot 3). n_iterations = 1 per trial (sequential slot generation, per Task 4 reviewer's design clarification). Phase 1.3 consolidation: 8 replays per tag during sleep window.

## Result: GENERATIVE_REPLAY_NEGATIVE

Multi-seed completion accuracy:

| K (stored sequences) | Per-seed [42, 43, 44] | Multi-seed-mean | Verdict |
|---|---|---|---|
| 4 | [0.085, 0.060, 0.065] | **0.070** | BELOW BAR (chance ~0.0625) |
| 8 | [0.025, 0.065, 0.050] | **0.047** | BELOW BAR (essentially at chance) |
| 16 | [0.045, 0.070, 0.055] | **0.057** | BELOW BAR (at chance) |

Aggregate: 104 correct out of 1800 = 5.78% (vs 6.25% chance baseline for picking 1 specific word from a 16-word vocabulary). The result is **essentially random** — the (c) loop's decoded continuation distribution is NOT correlated with the stored sequence's actual next-slot word.

Wall-clock 20.4 min on CuPy/RTX 3090 (much faster than the 1-3 hr estimate; reviewer's estimate was right).

## Smell-test PASSED (the NEGATIVE is genuine)

- Per-seed × per-K values reproduce aggregate byte-for-byte from raw JSON
- All 1800 trials ran (no silent skipping); per_k entries each have n_trials=200
- Batched-vs-scalar max-diffs 2.08e-17 / 1.39e-17 / 2.08e-17 every seed (machine precision; decoder primitives functioning correctly)
- Substrate confirmed loaded from validated n=98 caches; all gates re-frozen post-load
- Phase 1.3 consolidation invoked with validated parameters (n_replays_per_tag=8, region_filter=ca3, top_k=100)
- The loop genuinely ran (not a NaN or crash; verdict label "GENERATIVE_REPLAY_NEGATIVE" properly recorded)

The NEGATIVE is an honest empirical measurement, not a runner artifact.

## Failure mode diagnosis (informed by the data + biology)

Three plausible failure modes, ranked by likelihood:

### 1. SWR replay isn't actually reactivating the stored engrams (highest likelihood)

The `trigger_swr_replay` function in `research/runners/generative_replay_loop.py` opens the `ca3_swr_burst` gate, runs N=100 simulation steps, closes the gate. The validated Phase 1.3 mechanism that this replicates ALSO has a CA3-stimulation step during the SWR window (driving CA3 with sparse external current to seed the replay event). The (c) loop's `trigger_swr_replay` does NOT inject this stimulation — it relies on background activity + opened CA3 recurrence to drive replay. This may be insufficient to reactivate a specific stored ensemble.

**Biology validation source**: Phase 1.3 consolidation trainer at `research/runners/consolidation_trainer.py` — check whether `run_concept_replay_phase` (which IS validated multi-seed) drives CA3 explicitly during replay. If yes, the (c) loop's `trigger_swr_replay` is missing this drive and that's the failure.

### 2. Phase 1.3 consolidation didn't transfer sequence-specific patterns into cortex

The validated Phase 1.3 SWR consolidation (3/3 strict anti-cheat multi-seed) was tested on DIRECT-BINDING tasks (W→A binding via the ca1→motor pathway). For the (c) loop, the consolidation needs to transfer SEQUENCE-SPECIFIC patterns (each tag = sum-activity-across-3-slot-words; the loop expects this to reactivate via SWR and produce a cortical activity pattern the decoder can map to a continuation).

But the consolidation's existing ca1→cortex pathways (validated for direct-binding) may not capture the slot-position structure. The engram tag is on CA3, but the cortex doesn't have slot-specific structure.

### 3. The grounded-symbol decoder operates correctly but on noise

The captured post-replay cortical activity may be dominated by background dynamics rather than replay-driven content. The decoder dutifully argmaxes over the 16-concept grounded vocabulary, but if the activity is unstructured, the argmax is random.

The `_capture_concept_pool_activity` function captures from the concept-pool union (3200 neurons). The captured activity may reflect ongoing background dynamics, not the replay-driven specific pattern.

## The honest scientific finding

The (c) loop's pre-registered design (encode PFC frame → trigger SWR → capture cortex → decode → update frame) is BIOLOGY-FAITHFUL (validated by Schwartenbeck 2023 Cell; the three-stage iterative refinement biology; PFC-SWR timing biology) but the IMPLEMENTATION ON THIS SUBSTRATE doesn't produce above-chance completion at n_iterations=1.

This is a real biology-translatable result: **the substrate's validated components (concept pools + hippocampus + SWR consolidation + dlpfc_wm + parallel-matching mode-unification) cleanly interface separately, but the GENERATIVE-REPLAY LOOP that COMPOSES them doesn't produce sequence-completion behaviour in its current architecture.**

This precisely localises the bottleneck: it's not the substrate (all 5 components validated independently); not the FHRR/decoder primitives (validated via 5 pillars); not the bridge mechanics (smoke ran end-to-end mechanically). The bottleneck is **how the SWR replay drives sequence-specific cortical activity** that the decoder can read.

## Pivot direction (informed by the failure mode diagnosis)

### Immediate next step (cheap CPU; ~30-45 min): SWR replay reactivation probe

A focused diagnostic probe that:
1. Loads the trained+consolidated substrate (same as decisive runner)
2. Runs SWR replay for one engram tag (specific tag name)
3. Captures cortical activity post-replay
4. MEASURES the similarity of post-replay activity to:
   - The stored engram's ca3 pattern (does the replay reactivate the right CA3 ensemble?)
   - The encoded-during-storage cortical activity for that engram (does it reactivate the right cortical pattern?)
   - A control: random other engram patterns (should be low)
5. If post-replay activity HIGH-similarity to correct engram + LOW to others: replay works; pivot direction is the DECODER
6. If post-replay activity LOW similarity to correct + LOW to others: replay doesn't reactivate; pivot is the SWR-trigger mechanism
7. If post-replay activity HIGH-similarity ACROSS all engrams: replay is non-specific; pivot is the consolidation mechanism

This diagnostic resolves the question "what specifically fails" before designing a refinement. Estimated CPU cost ~30-45 min (reuses substrate; no GPU substrate work).

### Subsequent refinement directions (informed by diagnostic outcome)

- **If SWR trigger insufficient**: refine `trigger_swr_replay` to drive CA3 explicitly during the SWR window (per the validated Phase 1.3 mechanism's pattern). This is a small code change (~10 lines).
- **If consolidation doesn't carry sequence structure**: investigate whether the engram-tagging mechanism + Phase 1.3 consolidation can be extended to capture slot-position structure (e.g., per-slot engram tags; ec_context positional binding from catalog D.01+D.02+D.11 already validated as substrate).
- **If decoder needs different input**: maybe capture from a different region (e.g., dlpfc_wm holding the replayed continuation) rather than concept-pool union.
- **If multi-iteration helps** (Schwartenbeck three-stage refinement): increase n_iterations from 1 to ~30; each iteration runs SWR + capture + decode; the trajectory of decoded continuations should converge if the loop has any signal at all. The fact that n_iterations=1 gave chance suggests this won't help much (chance × 30 iterations = still chance), but it's worth verifying.

## What's preserved unconditionally

- All five substrate-readiness pillars (n=93/n=94/n=96/n=97/n=98) stand. Their validation is independent of the (c) loop.
- The parallel-matching mode-unification mechanism is validated.
- The Phase 1.3 SWR consolidation is validated for direct-binding tasks.
- The (c) TDD plan + (c) design doc remain useful frameworks (the TDD discipline + reuse-by-import architecture proved out at Tasks 0-4).
- The no-confab moat 7/7 throughout.

## Honest framing

This is the (c) loop's FIRST decisive empirical test. The result is NEGATIVE; the loop's basic architecture doesn't produce above-chance partial-sequence completion. This is BIOLOGY-TRANSLATABLE: the biology says the substrate components support generative replay (Schwartenbeck 2023; the project's substrate matches in mechanism); the empirical answer is that the specific INTEGRATION the (c) loop implements doesn't elicit the sequence-completion behaviour. The next step is to diagnose WHERE the integration fails + iterate via the project's standing biology-grounded discipline.

## Files

- Runner: `research/findings/raw/generative_replay_decisive.py` (Task 2 + Task 4 reviewed CLEAR)
- Log: `research/findings/raw/generative_replay_decisive_full.log`
- Output JSON: `research/findings/raw/generative_replay_decisive_full.json`
- Per-seed trial caches: `research/findings/raw/generative_replay_decisive_cache/trials_full_seed{42,43,44}.json`
- This findings doc: `research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-loop-at-n-iterations-1-doesnt-produce-above-chance-completion-pivot-direction-identified.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green.
- Frozen 0.80 bar unchanged (the NEGATIVE is per the pre-registered bar; no goalpost moving).
- Both git remotes propagated.
- NEGATIVE pillar pending fresh adversarial review.
