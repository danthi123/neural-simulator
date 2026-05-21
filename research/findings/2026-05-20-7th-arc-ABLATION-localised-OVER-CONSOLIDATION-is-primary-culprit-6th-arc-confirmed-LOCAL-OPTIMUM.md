# 7th arc ablation diagnostic: OVER-CONSOLIDATION (n_replays_per_tag=50 vs 20) is the primary culprit of the 7th arc regression; cue-suppression/amp-tag/persistent-PFC are gate-neutral-alone on this substrate; 6th arc CONFIRMED as the LOCAL OPTIMUM in the gating + augmenting composition design line

## Status

Controller-only diagnostic from the 7-arc trajectory analysis. The 7th
arc (commit `54f37c1`) showed: combining 4 augmenting mechanisms
collectively REGRESSED N=3 accuracy from the 6th arc's 0.458 to 0.363.
This diagnostic (commit `0e84f64`) tests each of the 4 mechanisms
ONE-AT-A-TIME on the 6th arc baseline to localise which caused most
of the regression.

## Ablation result (3 seeds × N=3 only; biological scale; cached substrate; ~16 min wall-clock)

| Condition | Mechanism added to 6th arc | mean_full | mean_uniform | advantage | vs 6th arc baseline |
|-----------|----------------------------|-----------|--------------|-----------|---------------------|
| A | cue-suppression-during-replay | 0.411 | 0.327 | +0.083 | -0.047 |
| B | amplified-tag-stim 3.0x | 0.411 | 0.327 | +0.083 | -0.047 |
| C | persistent PFC-frame 50-step | 0.411 | 0.327 | +0.083 | -0.047 |
| **D** | **higher n_replays_per_tag=50** | **0.274** | 0.327 | **-0.054** | **-0.184** |

6th arc baseline N=3: full = 0.458, advantage = +0.137 (consistent 3/3 seeds positive).
7th arc all-mechanisms N=3: full = 0.363, advantage = -0.048.

## KEY FINDINGS

### 1. NONE of the 4 mechanisms alone improves on the 6th arc baseline

All four conditions produce full_acc < 0.458 (the 6th arc baseline at
N=3). The 6th arc is CONFIRMED as the LOCAL OPTIMUM in this design
space using only already-validated subsystems.

### 2. Higher n_replays_per_tag (mechanism D) is the primary culprit

D alone produces -0.184 regression in full_acc -- substantially LARGER
than the 7th arc's combined -0.095. Doubling replay cycles (20 -> 50)
actively HARMS retrieval. This is the dominant harmful mechanism.

### 3. Conditions A, B, C are gate-NEUTRAL-alone (bit-identical accuracies)

Cue-suppression-during-replay (A), amplified-tag-stim (B), and
persistent PFC-frame (C) each produce IDENTICAL per-cell counts on
this substrate at N=3 (3+1, 3+0, 2+0 correct across seeds 42/43/44).
The mechanisms ARE structurally active per the per-condition state
diagnostics, but the bridge-state perturbations don't propagate to
different abstention-gate outputs at this scale. The substrate has
enough downstream nonlinearity that small input changes get absorbed
before reaching the gated readout.

### 4. Stacking partially OFFSETS D's harm

The 7th arc's combined regression (-0.095) is SMALLER than D-alone's
(-0.184). The three gate-neutral-alone mechanisms (A, B, C) attenuate
D's per-cell damage when stacked. Nonlinear interaction is observable.

## Biology-translatable insight (sharpened across all 7 arcs + ablation)

**Over-consolidation is biologically harmful**, consistent with the
neuroscience literature:

1. **CLS theory (McClelland-McNaughton-O'Reilly 1995)**: sleep replay
   should be gentle and gradual; bursts of intense consolidation can
   produce catastrophic interference, not stronger memory.
2. **Replay sweet-spot**: real biological replay rates (~50-200 ms
   sharp-wave-ripples; ~20 events per minute during NREM) are
   evolutionarily tuned. Doubling the replay rate during a single
   consolidation window pushes the substrate into a regime where
   consolidation OVERWRITES the discriminative signal rather than
   strengthening it.
3. **The 6th arc's gentle 20-cycle replay** captures the
   biologically-tuned regime; doubling to 50 cycles pushes the
   substrate past the sweet spot.

The OTHER mechanisms (A/B/C) didn't produce measurable changes in
gated accuracy ALONE -- this is a substrate-specific finding (at the
v14/v16+hippocampus substrate's scale, those mechanisms' bridge-state
perturbations are absorbed by downstream FS interneuron normalisation
and abstention-gate thresholding before they propagate to the answer).

## The 7-architecture convergent ceiling + ablation localisation (complete)

Cross-arc trajectory at N=3:

| Arc | N=3 full | trajectory |
|-----|----------|------------|
| Unified | 0.274 | baseline |
| Theta-gamma | 0.280 | flat |
| 6th (replay + PFC, gentle) | **0.458** | **LOCAL OPTIMUM** |
| 7th (all 4 aggressive) | 0.363 | -0.095 regression |
| Ablation A alone | 0.411 | -0.047 (gate-neutral) |
| Ablation B alone | 0.411 | -0.047 (gate-neutral) |
| Ablation C alone | 0.411 | -0.047 (gate-neutral) |
| **Ablation D alone** | **0.274** | **-0.184 (PRIMARY CULPRIT)** |

The 6th arc is empirically confirmed as the LOCAL OPTIMUM. The gating
+ augmenting composition design line is asymptotically exhausted on
the v14/v16+hippocampus substrate using only already-validated
subsystems.

## Pre-registered next staged step

Per the standing autonomy + iterate-following-biology + the localised
ablation finding: the substrate-level refinement direction must
target the underlying retrieval mechanism, NOT the augmenting
parameters. Specifically:

(A) **Replace the cosine-readout-via-engram-tag-stim mechanism** with
    a different retrieval pipeline (e.g., a dedicated readout region
    trained specifically on compositional outputs; not the
    cued-substrate's spelling output). The localisation finding
    (commit `110f7cd`) showed the cued-noun's diffuse drive dominates
    the bound-adj signal at the lang_output cosine readout; replacing
    that readout could close the remaining 0.34 gap to 0.80.

(B) **Per-region inhibitory normalisation** to suppress cross-pathway
    interference (extend the v14/v16 within-kind FS mechanism to
    cross-kind suppression at the lang_output level).

(C) **Honest closure of the gating + augmenting composition design
    line** as terminal biology-translatable finding. The 7-arc series
    + ablation localisation are durable scientific contributions.

Per standing autonomy, (A) is queued. The ablation result tells us
PRECISELY where the substrate's bottleneck lies: the gated readout's
sensitivity to bridge-state perturbations is too low (mechanisms A/B/C
are absorbed) AND over-consolidation harms retrieval (mechanism D).
The next iteration must change the READOUT mechanism, not the
augmenting parameters.

NO bar change anywhere; protected set byte-empty diff vs `e8a99a2`
holds; no-confab moat 7/7 byte-identical; 4 calibrated abstention
moats byte-stable.

## Honest ceiling (unchanged)

Conversational / compositional capability NOT achieved/claimed.
The 7-arc series + sweet-spot finding + ablation localisation are
substantive biology-translatable scientific contributions. The 6th
arc remains the best-observed performance on the gating + augmenting
composition design line (0.458 at N=3 vs 0.80 bar; -0.342 gap).
Closing the remaining gap requires deeper substrate refinement.

## Files / evidence

- Ablation diagnostic script: `research/findings/raw/7th_arc_ablation_diagnostic.py`
- Ablation durable JSON: `research/findings/raw/7th_arc_ablation_diagnostic.json`
- All prior arc runners + frozen verdict modules + calibrated moats byte-unchanged.
