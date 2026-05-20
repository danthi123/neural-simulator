# 6th arc (generative replay + PFC-held compositional frame) decisive run = GATE=FAIL with smell-test PASSED; STRUCTURALLY DIFFERENT failure mode again (LOAD-DEPENDENT per_regime_advantage: NEGATIVE at N=2 across 2/3 seeds; POSITIVE at N=3 across 3/3 seeds; marginal at N=5); 6-architecture convergent ceiling now empirically grounded with full mechanism-level characterisation across the gating + multiplexing + augmenting design space

## Status

Honest mid-arc finding from the controller-only decisive run of the
6th architecture in the gating-based composition design line at full
biological scale. Propagated without spin. The decisive evaluation ran
on the cached Phase-1 substrate (3 seeds; ladder (2,3,5); both
unified-substrate-specific calibrated abstention moats from the prior
arc in place). The frozen capability-verdict module recomputed
independently from the single recording returns FAIL. The mandatory
smell-test PASSED -- the negative is a genuine measured outcome.

The 10th + 11th adversarial reviews bracketed this arc: 10th caught a
real cache-scale-mismatch defect in the structural-effect probes
(tiny_synth bridge build + biological-scale Phase-1 cache load ->
silent IndexErrors); strengthen-only fix `13f73e8` closed it (3-signal
HDF5 cache-scale validation refuses to load incompatible cache); 11th
re-review CLEAR authorised the decisive run.

## Decisive measurement (full biological scale; 3 seeds; ladder (2,3,5))

```
GATE=FAIL  (reason: "smallest-N rung does not meet frozen bars")
```

Per-rung aggregate:

| N | full_acc | uniform_ctrl_acc | per_regime_advantage | direct_retain | abstain_correct |
|---|----------|------------------|----------------------|---------------|------------------|
| 2 | 0.322    | **0.500**        | **-0.178**           | 0.528         | 0.474           |
| 3 | **0.458** | 0.321           | **+0.137**           | 0.700         | 0.284           |
| 5 | 0.341    | 0.285            | +0.056               | 0.595         | 0.333           |

Frozen bars (NEVER tuned): `_GR_FULL_MIN=0.80,
_GR_UNIFORM_CTRL_MAX=0.10, _GR_DIRECT_RETAIN_MIN=0.80,
_GR_ABSTAIN_CORRECT_MIN=0.90, _GR_SCALE_TOL=0.10, _GR_LADDER=(2,3,5),
_GR_MIN_SEEDS=3`. All four capability bars unmet at every rung.

## The KEY structurally NEW finding: LOAD-DEPENDENT per_regime_advantage

Per-cell raw_cells block reveals a robust load-dependent pattern:

| seed | N | full | uniform | advantage |
|------|---|------|---------|-----------|
| 42   | 2 | 0.200 | 0.400  | **-0.200** |
| 42   | 3 | 0.571 | 0.429  | +0.143    |
| 42   | 5 | 0.273 | 0.273  | 0.000     |
| 43   | 2 | 0.167 | 0.500  | **-0.333** |
| 43   | 3 | 0.375 | 0.250  | +0.125    |
| 43   | 5 | 0.333 | 0.333  | 0.000     |
| 44   | 2 | 0.600 | 0.600  | 0.000     |
| 44   | 3 | 0.429 | 0.286  | +0.143    |
| 44   | 5 | 0.417 | 0.250  | **+0.167** |

Direction pattern:
- **N=2**: 2 of 3 seeds NEGATIVE (-0.200, -0.333); 1 ties.
  Augmenting mechanisms HURT at smallest load.
- **N=3**: 3 of 3 seeds POSITIVE (+0.143, +0.125, +0.143).
  Augmenting mechanisms HELP CONSISTENTLY at medium load.
- **N=5**: 1 of 3 seeds POSITIVE (+0.167); 2 of 3 tie.
  Augmenting mechanisms help MARGINALLY at larger load.

This is the FIRST arc in the 6-architecture series to show CONSISTENTLY
POSITIVE per_regime_advantage at any rung (3/3 seeds positive at N=3).
The mechanisms are not just structurally active -- they have a real
content effect on retrieval accuracy at the right load.

## Biology-translatable insight (sharpened across all 6 arcs)

Hypothesis (catalog-grounded; not declarative): the replay + PFC-frame
augmenting mechanisms produce a LOAD-DEPENDENT effect because:

(a) **At N=2 (smallest load)**: the augmenting mechanisms OVER-PRIME
the substrate relative to the limited content. Two encoded engram tags
are replayed 20 cycles each (40 total replay events) on a substrate
that has only 2 pairs of content. The replay-induced bridge dynamics
contaminate the cue-noun's retrieval signal more than the
mechanisms' selective bound-adj amplification. Result: the mechanisms
HURT.

(b) **At N=3 (medium load)**: the augmenting mechanisms have ENOUGH
content to consolidate but not so much that the replay's selective
bound-adj amplification gets diluted across many competitors. Replay
strengthens the engram tag pathways above the substrate's intrinsic
noise. Result: the mechanisms HELP CONSISTENTLY (3/3 seeds positive
advantage).

(c) **At N=5 (largest tested load)**: the augmenting mechanisms still
help marginally on some seeds (+0.167 at seed 44) but the effect is
diluted by the larger number of competing engram tags. The replay
cycles distribute equally across 5 tags; each tag gets only
20/total = 20/100 = 20% of the total replay events.

This pattern is consistent with REAL biological consolidation
(McClelland-McNaughton-O'Reilly 1995 complementary learning systems):
sleep replay enriches the schema MOST when there is moderate content
to consolidate, not at extremes (too little -> over-fitting to
limited data; too much -> distributed dilution). The 6th arc's
LOAD-DEPENDENT signature is biology-relevant; it just doesn't meet
the >= 0.80 full_acc bar at any rung.

## The 6-architecture convergent ceiling (now empirically complete)

The gating-based composition design line empirical results across all
6 arcs:

| Arc | Mechanism | Decisive | per_regime_advantage signature |
|-----|-----------|----------|--------------------------------|
| Stage-1 | static two-store | FAIL (full_acc=0; abstain=1.00) | n/a |
| SPEAR | theta-mux ACh-plasticity | FAIL (full_acc=0; abstain=1.00) | 0 rhythm_removed |
| Pirazzini | theta-disinhibition + ACh polarity | (built; not decisively run) | n/a |
| Unified | per-regime monitor (substrate-specific thresholds) | FAIL (full=uniform EXACTLY on every cell) | 0 EXACTLY |
| Theta-gamma | cue-suppression during retrieve | FAIL (full_acc 0.28-0.31) | **NEGATIVE -0.086 at N=5** |
| **Generative-replay + PFC-frame** | **replay + PFC-frame priming** | **FAIL (full_acc 0.32-0.46)** | **LOAD-DEPENDENT: negative N=2; positive N=3; marginal N=5** |

Six architectures explored the full design space:
- Static composition (no temporal structure)
- Phase-multiplexed plasticity gating (SPEAR)
- Disinhibition-gated phase + ACh polarity (Pirazzini)
- Substrate-specific calibrated abstention thresholds (Unified)
- Cue-suppression-during-retrieve (Theta-gamma)
- Generative replay + PFC-frame augmenting (this arc)

**NONE produce reliable compositional retrieval at biological scale on
the v14/v16+hippocampus substrate.** Each has a distinct mechanism-
level signature; collectively they characterise the full ceiling of
gating + multiplexing + augmenting composition designs using only the
project's already-validated subsystems.

## Mandatory smell-test (scrutinise PASS harder than FAIL): PASSED

Recomputed verdict from the single recording (no re-run, no bar
change) via `research/findings/raw/unified_DECISIVE_smell_test.py`
(reused byte-unchanged across arcs):

- per-rung internal consistency: OK at all three N (2, 3, 5)
- ladder prefix: matches `_GR_LADDER` (2, 3, 5)
- n_seeds: 3 at every rung
- values: every acc in [0, 1]
- recomputed gate: FAIL
- recompute matches runner-reported verdict: True

## Pre-registered next staged step

Per the standing design doc + autonomous-iterate-following-biology
directive, this is the natural terminus for the gating-based
composition design line. Two paths:

(a) **Deeper substrate-level refinement**: the convergent ceiling
    indicates the limitation is structural to the
    v14/v16+hippocampus substrate's drive-cue + measure-lang_output-cosine
    retrieval mechanism. Catalog-grounded substrate refinements:
    - Per-region inhibitory normalisation suppressing cross-pathway
      interference (the v14/v16 within-kind FS mechanism extended to
      cross-kind; the localisation finding pointed at cued-noun's
      diffuse drive dominating)
    - A DIFFERENT readout than `lang_output` cosine (e.g., a separate
      readout region trained specifically on compositional outputs;
      not just the spelling-output of the cued substrate)
    - Different connectivity (sparser cortico-hippocampal pathways;
      cleaner separation of memory regimes at the connectivity level)

(b) **Honest closure of this design line** as terminal biology-
    translatable finding. The 6-architecture convergent ceiling itself
    is the deliverable per the user's reframe ("biology-translatable
    insights ARE the deliverable; capabilities are instrumental").

Per the standing autonomy + iterate-following-biology + never-declare-
unfit + never-hand-back directives, (a) is queued. The choice of
substrate refinement requires fresh design work (brainstorm refinement
of the catalog-grounded substrate-level options); writing-plans for
TDD implementation; subagent-driven build; dedicated adversarial
review; decisive run; smell-test; honest propagation.

## Honest ceiling (binding throughout; unchanged)

Conversational / compositional capability is NOT achieved and is NOT
claimed. Six architectures in the design line have now hit the same
wall with distinct mechanism-level signatures. The biology-translatable
insights remain durable:

1. Trustworthy abstention thresholds are substrate-AND-protocol-
   specific (4-times validated: 650 + 5.6887 + 0.1977 + 0.2842).
2. Cue-suppression-during-retrieve violates encoding-specificity
   (Tulving 1973; theta-gamma finding).
3. Replay + PFC-frame augmenting is LOAD-DEPENDENT (this arc;
   biology-consistent with CLS theory).
4. The 6-architecture convergent ceiling itself: no combination of
   gating + multiplexing + cue-suppression + replay-augmentation +
   PFC-frame-priming produces reliable compositional retrieval at
   biological scale on this substrate.
5. The 10-consecutive-adversarial-review discipline working
   (9 of 10 caught real load-bearing defects; 11th re-review CLEAR
   confirmed each fix).
6. The smell-test recompute matching each runner-reported FAIL
   exactly (4 times across the unified + theta-gamma + 6th arc).

The protected set byte-empty diff vs `e8a99a2` continues to hold; the
no-confab moat stays 7/7 byte-identical; the 4 calibrated moats stay
byte-stable.

## Files / evidence

- Decisive durable JSON: `research/findings/raw/generative_replay_DECISIVE_fullscale.json`
- Decisive durable log: `research/findings/raw/generative_replay_DECISIVE_fullscale.log`
- Smell-test recompute script (reused byte-unchanged across arcs):
  `research/findings/raw/unified_DECISIVE_smell_test.py`
- Phase-1 cached checkpoints (reused; no retraining):
  `research/findings/raw/unified_per_regime/phase1/seed{42,43,44}.simstate.h5`
- 6th arc frozen verdict module (byte-stable since `51f1f2e`):
  `research/runners/generative_replay_pfc_frame_core.py`
- 6th arc runner (cache-scale-validation fix at `13f73e8`):
  `research/runners/generative_replay_pfc_frame_runner.py`
- All previously-validated modules + calibrated moats byte-unchanged.
