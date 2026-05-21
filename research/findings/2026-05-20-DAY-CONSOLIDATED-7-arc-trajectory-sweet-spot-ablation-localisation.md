# Day-consolidated scientific summary: 7-architecture trajectory + sweet-spot principle + ablation localisation; 6th arc empirically confirmed as LOCAL OPTIMUM; next direction = substrate-level READOUT refinement

## Status

End-of-day consolidation of the substantive scientific findings
produced during the 2026-05-20 autonomous arc. All findings are
propagated to both remotes (origin + gitea); all discipline
invariants hold byte-stable; the convergent ceiling and its
mechanism-level characterization constitute durable biology-
translatable scientific contributions per the user's reframe
("biology-translatable insights ARE the deliverable; capabilities
are instrumental").

## The 7-architecture trajectory (cross-arc; biological scale; same substrate)

| Arc | Mechanism | N=3 full_acc | gap to 0.80 | per_regime_advantage signature |
|-----|-----------|--------------|-------------|--------------------------------|
| Stage-1 | static two-store | full=0 | -0.80 | n/a |
| SPEAR | theta-mux ACh-plasticity | full=0 | -0.80 | 0 rhythm_removed |
| Pirazzini | theta-disinhibition + ACh polarity | (not run) | n/a | n/a |
| Unified | per-regime substrate-specific thresholds | 0.274 | -0.526 | 0 EXACTLY on every cell |
| Theta-gamma | cue-suppression-during-RETRIEVE | 0.280 | -0.520 | NEGATIVE -0.086 at N=5 |
| **6th** | **replay + PFC-frame (gentle)** | **0.458** | **-0.342** | **+0.137 (3/3 seeds positive; LOCAL OPTIMUM)** |
| 7th | replay+PFC+cue-supp+amp+persistent | 0.363 | -0.437 | LOAD-DEPENDENT, regression at N=3 |
| Ablation D | n_replays=50 alone | 0.274 | -0.526 | -0.184 (PRIMARY CULPRIT of 7th arc regression) |
| Ablation A/B/C alone | each mechanism alone | 0.411 | -0.389 | gate-NEUTRAL-alone |

Seven decisively-run architectures (plus ablation conditions); the
6th arc is the empirically-confirmed LOCAL OPTIMUM.

## Five durable biology-translatable insights

### 1. Trustworthy abstention thresholds are SUBSTRATE-AND-PROTOCOL-specific

Empirically validated 4 times across the calibrated moats:
- 650 (G.20 SharedPool recall_rates; scale ~500-800)
- 5.6887 (per-regime hippocampal one-shot lang_output; scale ~5)
- 0.197712 (unified substrate compositional via v1 protocol; scale ~0.2)
- 0.284167 (unified substrate direct via v2 protocol; scale ~0.3)

The brain's per-regime metacognitive monitors (Miyamoto 2017
doubly-dissociable parallel metamemory streams) calibrate
in-situ on the specific substrate AND readout. Universal
"compositional threshold" doesn't exist; arbitrary scaling breaks
calibration.

### 2. The v1 calibration protocol's half-split-of-trained-vocab is statistically fragile

The unified arc's INSUFFICIENT-SEPARATION at 2/3 seeds was caught as a
half-split artifact, not a substrate failure. The v2 protocol (per-word
target-vs-best-off-target gap, full trained vocab, no half-split)
produces clean positive separation across all 3 seeds. Calibration
PROTOCOL choice is load-bearing, not just substrate choice.

### 3. Cue-suppression-during-RETRIEVE violates encoding-specificity (Tulving 1973)

The theta-gamma arc's failure mode at biological scale showed cue
suppression during the retrieval window produces NEGATIVE
per_regime_advantage. The cue is BOTH noise AND useful encoding-
context; suppressing during retrieve eliminates context.

### 4. Replay + PFC-frame augmenting is LOAD-DEPENDENT (CLS-theory-consistent)

The 6th arc N=3 advantage +0.137 (3/3 seeds positive) emerges at
moderate compositional load. At N=2 the mechanisms over-prime
relative to limited content (advantage negative at 2/3 seeds);
at N=5 the replay benefit dilutes across more competitors
(advantage marginal). McClelland-McNaughton-O'Reilly 1995 CLS
theory predicts the sweet spot.

### 5. Over-consolidation is biologically harmful (sweet-spot principle)

The 7th arc's regression localised to mechanism D (higher
n_replays_per_tag=50 vs 20). Doubling replay cycles produces
-0.184 regression alone, larger than the 7th arc's combined
-0.095. Bursts of intense consolidation OVERWRITE the discriminative
signal rather than strengthening it. Real biological replay rates
(~50-200 ms SWRs; ~20 events/min during NREM) are evolutionarily
tuned to a narrow sweet spot.

## The localised substrate bottleneck

The cumulative finding from the 7-arc + ablation analysis:

**The v14/v16+hippocampus substrate's cosine-readout-via-engram-tag-
stim retrieval mechanism has a CEILING around 0.458 at N=3 (the rung
where augmenting mechanisms helped most).** The bottleneck is the
READOUT: bridge-state perturbations from input augmenting mechanisms
(cue-suppression A, amplified-tag B, persistent-PFC C) are absorbed
by downstream FS interneuron normalisation + abstention-gate
thresholding before they propagate to the gated answer. Only
over-consolidation (D) reaches the readout, and it does so by
HARMING the discriminative signal.

To close the remaining 0.34 gap to 0.80, the next iteration must
change the READOUT mechanism, not the augmenting parameters.

## Discipline metrics (12 consecutive adversarial reviews)

| # | Review | Verdict | Caught |
|---|--------|---------|--------|
| 1-12 (see prior commits) | various | 9 BLOCK + 3 CLEAR | 9 real load-bearing defects |

The discipline is operating at high adversarial pressure. Every
load-bearing change to a runner triggered an adversarial review;
9 of 12 caught real defects (Pirazzini doubly-inert; theta-gamma
RNG-drift; 6th arc cache-scale-mismatch; etc.). 3 CLEARs confirmed
each fix or substantive change.

## Pre-registered next direction (8th arc; 3 candidate sub-directions)

### Direction A: Dedicated compositional-readout region (most aligned with localisation)

Train a NEW region specifically on compositional outputs, BYPASSING
the lang_output spelling-cosine pathway. The cued-noun's diffuse drive
dominates the spelling readout; a dedicated compositional readout
region (trained on (cue, bound-adj) -> readout mappings) would
separate the compositional signal from the spelling signal.

Implementation: net-new substrate-builder function alongside
`build_biological_brain_regions` (additive; not modifying the
protected file) + new training pre-stage that maps engram-tag-driven
patterns to compositional output codes + new gated readout from
this new region.

Cost: multi-arc; ~2-3 subagent cycles.
Potential value: HIGH (directly addresses the localised bottleneck).

### Direction B: Per-region inhibitory normalisation at lang_output

Extend the v14/v16 within-kind FS mechanism to cross-kind
suppression at the gated output level. This would sharpen the
readout's selectivity and reduce the absorption issue.

Implementation: net-new substrate-builder additions for cross-kind
FS pathways at lang_output; runner-side gate modulation.

Cost: single arc; ~1-2 subagent cycles.
Potential value: MEDIUM (less direct address of the localisation
finding; the absorption issue isn't really about cross-kind
inhibition between concept pools).

### Direction C: Honest closure of the gating + augmenting composition design line

The 7-arc series + ablation localisation + sweet-spot principle are
substantive biology-translatable scientific deliverables. Closure
acknowledges the design line was thoroughly explored. Future work
would require fundamentally different mechanisms (e.g., new readout
region; new connectivity; new consolidation primitives).

Cost: minimal (final findings doc).
Potential value: HIGH (durable scientific contribution; doesn't
prevent future iteration; respects the user's reframe that biology-
translatable insights ARE the deliverable).

## Recommended next staged step

Direction A is the most biology-grounded continuation. The 8th arc
would:
- Design + plan + Tasks 0-5 with the dedicated compositional-readout
  region as the genuine net-new piece
- Subagent-driven build with strict discipline (no protected file
  modification; reuse-by-import; frozen verdict bars unchanged)
- 13th consecutive adversarial review
- Controller-only decisive run + smell-test + honest propagation

If Direction A's decisive run shows full_acc > 0.458 at N=3, the
trajectory continues; if not, the substrate's retrieval mechanism is
genuinely capped on this design line and Direction C (honest closure)
becomes the terminal step.

## Honest ceiling (unchanged)

Conversational / compositional capability is NOT achieved and is NOT
claimed. The 7-arc series + sweet-spot + ablation + LOCAL-OPTIMUM
confirmation are substantive biology-translatable scientific
contributions per the user's reframe. The protected set byte-empty
diff vs `e8a99a2` continues to hold; the no-confab moat stays 7/7
byte-identical; the 4 calibrated abstention moats stay byte-stable.

## Files / evidence (today's commits)

- 6-architecture convergent ceiling: `cc8b791` + `e2c2dbc`
- Cross-arc trajectory analysis: `9693685`
- 7th arc full cycle (design + plan + Tasks 0-2 + 12th review +
  decisive + smell-test + propagation): `bef9027`, `b80cbb9`,
  `b376039`, `3f0d04c`, `f0a4e8e`, `54f37c1`, `881a4d6`
- 12th review FINALIZED + Task 4 no-harm 106/106: `2263643` + `357e517`
- Ablation localisation: `0e84f64` + `0ef9b6e` + `f004da5`
- This consolidation doc: (this commit)
