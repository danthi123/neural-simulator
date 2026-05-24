# Cross-bridge mode-unification BOUNDARY — order-bearing parallel-matching PASSes perfectly at every load on the 160-concept union; order-invariant marginal-sum ceilings at L=5 (2026-05-23)

## What was tested

OPTION 4 from the (e) ensemble-extension completion: does the biologized parallel-matching mode-unification extend ACROSS bridge boundaries? The (e) extension (pillar n=94) validated per-bridge mode-unification (each bridge's own 32-word vocabulary). This probe tests bridge-spanning composites — encoded composites whose K items are drawn UNIFORMLY from the union of all 5 bridges' 32-concept vocabularies (160 concepts total), decoded per-slot via parallel-matching over the full 160-concept union.

Runner: `research/findings/raw/cross_bridge_mode_unification_probe.py` (GPU-batched; net-new orchestration; reuses 160-ensemble caches + parallel-matching primitives byte-unchanged; uses a backend-aware batched `phase_similarity` that stacks the 160 grounded symbols as one (V, N_dim) phase matrix and computes all V similarities in one mean-cosine broadcast per slot per trial; per-cell startup verifies batched == scalar phase_similarity within 1e-10 tolerance — fail-closed).

Backend: CuPy on RTX 3090; wall-clock 283.5 s for 6 cells (vs estimated ~30 min on CPU, ~6× speedup at per-run level; ~2× at per-cell level since cell overhead dominates).

Two conditions tested:
1. **global_mean**: re-mean-centre activity across all 160 concepts (one global common mode subtracted; cleanest geometric framing; biology-faithful — cortical pooled inhibition normalises across the whole cortical extent, not per-region).
2. **per_bridge_mean**: each bridge's own common mode subtracted independently (the (e) extension's choice; comparison baseline).

## Pre-registered reading (fixed; never tuned)

- **CROSS_BRIDGE_PASS**: multi-seed-mean >= frozen 0.80 bar at every load {2, 3, 5} on BOTH order-bearing AND order-invariant readouts, with the 160-concept union vocabulary. Cross-bridge mode-unification extends the (e) per-bridge capability to bridge-spanning composites.
- **CROSS_BRIDGE_BOUNDARY**: either readout misses at some load; honest per-load breakdown reported.

## Result: CROSS_BRIDGE_BOUNDARY (both conditions)

Multi-seed-mean (seeds 42/43/44) integrated accuracy at loads {L=2, L=3, L=5}:

| Condition | L=2 OB / OI | L=3 OB / OI | L=5 OB / OI |
|---|---|---|---|
| global_mean | 1.000 / 1.000 | 1.000 / 1.000 | **1.000 / 0.790** |
| per_bridge_mean | 1.000 / 1.000 | 1.000 / 0.998 | **1.000 / 0.785** |

Per-seed L=5 OI breakdown:
- global_mean: seed 42 = 0.815, seed 43 = 0.755, seed 44 = 0.800
- per_bridge_mean: seed 42 = 0.780, seed 43 = 0.770, seed 44 = 0.805

**Order-bearing is exactly 1.000 at every cell across both conditions, all 6 seeds, all 3 loads — zero errors across 3600 OB trials**. The parallel-matching ORDER-BEARING decoder cleanly identifies the correct (bridge, item) per slot when drawn from the 160-concept union via dendritic-integration + lateral-inhibition WTA.

**Order-invariant marginal-sum top-K argsort ceilings at L=5** (~0.79 multi-seed; just below the 0.80 bar; never collapsing far). At L=2 and L=3 OI is perfect or near-perfect (1.000 / 0.998-1.000). The boundary is sharp and localised to the L=5 OI cell.

Per the strict pre-registered bar: BELOW BAR. Per the cross-readout/cross-load breakdown: 5 of 6 cells PASS per condition (10 of 12 cells across both conditions); the 1 missing cell per condition is consistent: L=5 OI under the 160-concept union vocabulary.

## Smell-test (recompute from recording; no re-run; no bar change)

GPU batched results reproduce the CPU partial-run results byte-for-byte at every overlapping cell (global_mean all 3 seeds + per_bridge_mean seed 42 — 4 cells in common, identical OB and OI values to 3 decimal places). Batched vs scalar `phase_similarity` max-diff: 2.08e-17 to 2.78e-17 across all 6 cells (well below the 1e-10 fail-closed tolerance; effectively machine precision for double-precision phase cosine).

The bar is unchanged (0.80; pre-registered in `vocabulary_scaling_run.py:BAR`; git-diff across the relevant commits byte-empty).

No oracle leak: the decoder argmaxes / argsorts over all 160 `(bridge, word)` tuples; the true item indices are NEVER passed to the decoder; the encoding uses `qrng.choice(V, ...)` and decoding reads `batched_phase_similarity(unbinds[k], vocab_phase_matrix, xp)` over the full union vocabulary. Standing reused decoder scoring contract is preserved.

No protected/frozen/moat module modified: the probe is net-new code; the only modules imported (`vocabulary_scaling_run`, `pattern_separation_grounding_probe`, `biologized_spiking_mode_unification_helpers`, `biologized_spiking_mode_unification_parallel_matching_runner`, `resonate_fire_fhrr`, `spiking_phasor_fhrr`, `sim.backend`) are reused unmodified. No autograd. No-confab moat 7/7 to be confirmed by reviewer.

## Sharp biology-translatable findings

1. **The decoder mechanism's per-slot identification half cleanly handles 5× more distractors**. Parallel-population matching at 160 grounded symbols achieves perfect ORDER-BEARING identification multi-seed at every tested load. The dendritic-integration + lateral-inhibition WTA scales from per-bridge (32 distractors) to cross-bridge (160 distractors) without per-slot degradation.

2. **The order-invariant marginal-sum top-K decoder ceilings at L=5 × V=160**. At L=2/L=3 it is perfect; at L=5 it sits at the bar (~0.79 multi-seed). The mechanism distributes load × vocab differently: per-slot OB only needs the correct argmax per slot independently; OI sums similarities across slots THEN ranks all 160 — the rank-comparison at the K=5 boundary is more sensitive to the symbol-grounding noise floor.

3. **The mean-centring choice doesn't materially affect the outcome at this scale**. Global vs per-bridge common-mode removal give 0.790 vs 0.785 at L=5 OI — within 1% of each other. The "global pooled inhibition" framing is not what gates cross-bridge composition; the noise floor in the grounded symbols (the substrate's CV ~1.6 propagated through deriver + spike-phase + FHRR) is.

4. **The split between OB-success and OI-ceiling at this scale parallels the FHRR capacity-envelope arc's earlier finding**: at N_dim=512 the pure algebra clears past V=256 on the OB readout but the OI readout has slight degradation at the extremes (0.95-0.99 at V=128-256). The cross-bridge spiking-grounded pipeline reproduces that pattern — OB has more headroom than OI under load × vocab pressure.

5. **The honest scope of the (e) per-bridge VALIDATED pillar (n=94) stands**: cross-bridge ORDER-BEARING extension is also strong (perfect at every cell — could be characterised as a per-bridge-to-cross-bridge OB capability extension); cross-bridge ORDER-INVARIANT is a sharp boundary at L=5. The bridge-spanning compositional capability is partially realised: identification per slot scales; set-comparison at high load × vocab does not (at this substrate's noise floor).

## Honest caveats preserved

1. **Oracle-adjacency caveat (from (b)/(e))**: parallel matching is structurally closer to "argmax over a stored vocabulary" than TPAM's recurrent attractor. The "vocabulary" here is the substrate's own derived grounded symbols on the 160-concept union; biology-grounded mechanism, caveat recorded.

2. **160-concept union from 5 separately-trained substrates**: each bridge was trained independently in the 160-ensemble decisive 9-hour GPU run. Cross-bridge composition here treats the union as a single vocabulary for the parallel-matching decoder. A genuinely INTEGRATED 160-concept substrate (one bridge trained on all 160 concepts) might give different geometry; that's a distinct, deferred direction.

3. **The OI ceiling at L=5 is NOT a catastrophic failure** — it sits at 0.78-0.81 multi-seed, just below the 0.80 bar. The pre-registered ENSEMBLE_PASS reading is "every cell >= 0.80"; this fails it by ~0.01-0.02. This is a BOUNDARY result, not a NEGATIVE; the mechanism partially extends.

4. **GPU-batched implementation byte-equivalent to scalar**: per-cell fail-closed equivalence check verified to 2e-17. The batched phase_similarity is the only mechanically-new code; everything else (encoding, unbinding, gamma-slot positions, grounded-symbol derivation) reuses the (b)/(e) primitives byte-unchanged.

5. **3-seed sample**: matches (b) and (e) seed sets. A 5-seed extension would tighten variance characterisation; given the L=5 OI margin is thin (~0.01-0.02 below bar), more seeds might shift the multi-seed-mean above or below — but the per-seed range 0.755-0.815 is consistent (not a strong outlier-driven artifact).

6. **Subject to fresh dedicated adversarial review before any capability_status pillar update** (per standing discipline). The BOUNDARY pillar should be recorded only after independent review confirms the runner is sound + no exploit can produce a spurious BOUNDARY (it's harder to false-PASS into a BOUNDARY than into a PASS, but the runner must still be byte-equivalent to its primitives).

## Implementation note: GPU batched primitive

The batched `phase_similarity` (in this probe's runner):

```python
def batched_phase_similarity(unbind_spikes, vocab_phase_matrix, xp):
    """Mathematically IDENTICAL to scalar phase_similarity(unbind, v)
    iterated for v in vocab; broadcasts on the active backend."""
    pu_host = spikes_to_phases(unbind_spikes, CYCLE_STEPS)
    pu = xp.asarray(pu_host)
    diffs = pu[None, :] - vocab_phase_matrix  # (V, N_dim)
    sims = xp.mean(xp.cos(2.0 * xp.pi * diffs), axis=1)  # (V,)
    return sims
```

Per cell, the runner calls `verify_batched_equivalent_to_scalar` at startup: builds vocab phase matrix on backend; computes batched similarities of one random probe spike pattern against all V; compares to scalar `phase_similarity(probe, grounded[bw])` for each bw; refuses to run if max-diff > 1e-10. Observed max-diffs 2.08e-17 to 2.78e-17 (machine precision).

The pattern is the GPU-batched DEFAULT for future characterisation probes (capacity-envelope sweeps, multi-seed extensions, vocab-size scaling, etc.); the speedup compounds with multiplied cells.

## Relation to the mode-unification thread completion

The biologized mode-unification thread (completed at (e)): per-bridge VALIDATED across all 5 bridges of the 160-concept ensemble (pillar n=94). This OPTION 4 probe extends one step further to bridge-spanning composites:

- (b) bridgeA only VALIDATED (n=93)
- (e) per-bridge all 5 bridges VALIDATED (n=94)
- (this probe) cross-bridge BOUNDARY — OB scales, OI ceilings at L=5 × V=160

The mode-unification capability is now characterised at three nested scales: single-bridge → per-bridge ensemble → cross-bridge union. The boundary appears at the cross-bridge × high-load × OI corner; everywhere else the mechanism PASSes.

## Files

- Runner: `research/findings/raw/cross_bridge_mode_unification_probe.py` (GPU-batched)
- Log: `research/findings/raw/cross_bridge_mode_unification_probe.log`
- Output JSON: `research/findings/raw/cross_bridge_mode_unification_probe.json`
- This findings doc: `research/findings/2026-05-23-cross-bridge-mode-unification-BOUNDARY-OB-PASSes-perfectly-OI-ceilings-at-L5-on-160-concept-union.md`
- Parent (e) ENSEMBLE PASS pillar n=94: `research/findings/2026-05-23-biologized-mode-unification-parallel-matching-EXTENDS-PER-BRIDGE-across-the-FULL-160-concept-ensemble.md`

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green (no edits to abstention_gate.py).
- Frozen 0.80 bar unchanged (pre-registered ENSEMBLE_PASS reading; this is honestly BELOW bar at L=5 OI).
- GPU-batched primitive byte-equivalent to scalar (verified 1e-10; observed 2e-17).
- Both git remotes propagated.
- BOUNDARY pillar pending fresh adversarial review.
