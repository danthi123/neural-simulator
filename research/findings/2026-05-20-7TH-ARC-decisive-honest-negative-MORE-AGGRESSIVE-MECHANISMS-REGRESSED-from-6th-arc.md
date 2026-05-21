# 7th arc (targeted cue-suppression-during-replay + amplified tag stim + persistent PFC-frame + higher n_replays_per_tag) decisive run = GATE=FAIL with smell-test PASS; CRITICAL NEW finding = MORE-AGGRESSIVE targeted mechanisms REGRESSED accuracy vs the simpler 6th arc; the cross-arc trajectory shows 6th arc was the LOCAL OPTIMUM, not a step on an increasing curve

## Status

Honest finding from the controller-only decisive run of the 7th
architecture in the gating + augmenting composition design line at
full biological scale. Propagated without spin. The 12 consecutive
adversarial reviews + Tasks 0/1/2 each landed; the structural-effect
probes confirmed all 4 mechanisms genuinely active. The decisive eval
ran on the cached Phase-1 substrate (3 seeds; ladder (2,3,5)). The
frozen verdict module recomputed independently returns FAIL. The
mandatory smell-test PASSED.

## Decisive measurement (full biological scale; 3 seeds; ladder (2,3,5))

```
GATE=FAIL (reason: "smallest-N rung does not meet frozen bars")
```

| N | full_acc | uniform_ctrl_acc | per_regime_advantage | direct_retain | abstain_correct |
|---|----------|------------------|----------------------|---------------|------------------|
| 2 | 0.322    | 0.256            | +0.067               | 0.528         | 0.482           |
| 3 | **0.363** | **0.411**       | **-0.048**           | 0.533         | 0.151           |
| 5 | 0.369    | 0.341            | +0.028               | 0.643         | 0.500           |

Frozen bars (NEVER tuned): `_TC_*` identical to all prior arcs. All
four capability bars unmet at every rung.

## CRITICAL CROSS-ARC FINDING: 7th arc REGRESSED from the 6th arc

Cross-arc trajectory at N=3 (the rung where mechanisms helped most in
the 6th arc):

| Arc | N=3 full_acc | gap to 0.80 | direction |
|-----|--------------|-------------|-----------|
| Unified per-regime monitor | 0.274 | -0.526 | baseline |
| Theta-gamma cue-suppression | 0.280 | -0.520 | flat (+0.006) |
| **6th arc (replay + PFC-frame)** | **0.458** | **-0.342** | **35% gap closure** |
| **7th arc (targeted aggressive)** | **0.363** | **-0.437** | **REGRESSION -0.095 from 6th** |

The 7th arc's more-aggressive mechanisms (cue-suppression-during-
replay + 3x amplified tag stim + 5x extended PFC-frame + 2.5x more
replay cycles) collectively REDUCED accuracy compared to the simpler
6th arc baseline. The 6th arc was the LOCAL OPTIMUM in the design
space, not a step on a continuously-increasing curve.

## Per-cell pattern (highly informative)

Per-cell raw_cells block reveals:

| seed | N | full | uniform | advantage |
|------|---|------|---------|-----------|
| 42   | 2 | 0.400 | 0.400 | +0.000 |
| 42   | 3 | 0.429 | 0.286 | +0.143 |
| 42   | 5 | 0.273 | 0.273 | +0.000 |
| 43   | 2 | 0.167 | 0.167 | +0.000 |
| 43   | 3 | 0.375 | 0.375 | +0.000 |
| 43   | 5 | 0.333 | 0.333 | +0.000 |
| 44   | 2 | 0.400 | 0.200 | +0.200 |
| 44   | 3 | 0.286 | 0.571 | **-0.286** |
| 44   | 5 | 0.500 | 0.417 | +0.083 |

N=3 across seeds: 1/3 positive (+0.143); 1/3 tie; 1/3 catastrophic
negative (-0.286 at seed 44). The 6th arc had 3/3 positive at N=3
(+0.143, +0.125, +0.143; consistent across seeds). The 7th arc's
seed-44-N=3 cell shows the mechanisms can actively SABOTAGE retrieval
at certain (seed, load) cells.

This is the FIRST arc in the 7-architecture series to show that the
augmenting mechanisms can cause CATASTROPHIC negative advantage (-0.286
on a single cell). The mechanisms are structurally active (5+ probes
confirmed) and interfere with each other.

## Biology-translatable insight (sharpened across 7 arcs)

**Hypothesis (catalog-grounded; not declarative)**: real biological
compositional retrieval has a NARROW SWEET SPOT for the auxiliary
mechanisms (consolidation strength, working-memory-frame strength,
cue-context priming). Over-aggressive auxiliary mechanisms interfere
with retrieval rather than helping it. This is consistent with
neuroscience observations:

1. **Replay strength**: McClelland-McNaughton-O'Reilly 1995 CLS theory
   predicts gentle, gradual replay; not large bursts. The 7th arc's
   2.5x more replay cycles (20 -> 50) may have over-consolidated the
   pathway, fixating it on noise patterns rather than the bound
   adjective.
2. **Working-memory persistence**: Wang 2002 NMDA bistability has a
   characteristic time-constant. Extending the PFC-frame stim from 10
   to 50 steps may have driven the dlpfc_verb attractor into a
   different state space than the gentle priming achieves; the
   "compositional frame" became "dominant noise floor".
3. **Cue-suppression during replay**: while encoding-specificity is
   respected at retrieve time, suppressing the cue during consolidation
   may have prevented the replay phase from learning the cue-context
   coupling that real biological replay preserves.
4. **Mechanism interaction**: each mechanism individually might help;
   the combination produces destructive interference. Real biological
   composition has these mechanisms balanced at PARTICULAR strengths
   that emerged from evolution; arbitrary scaling breaks the balance.

The 6th arc found the local optimum within this design space. The 7th
arc's regression provides EMPIRICAL EVIDENCE that "more is better" is
the wrong intuition for biology-grounded augmenting mechanisms.

## The 7-architecture convergent ceiling (now complete with optimum found)

| Arc | Mechanism | N=3 full | Trajectory step |
|-----|-----------|----------|-----------------|
| Stage-1 | static | n/a (full=0) | n/a |
| SPEAR | theta-mux ACh-plasticity | n/a (full=0) | flat |
| Pirazzini | theta-disinhibition + ACh | (not run) | n/a |
| Unified | per-regime monitor | 0.274 | baseline |
| Theta-gamma | cue-suppression at retrieve | 0.280 | flat |
| **6th** | **replay + PFC-frame (gentle)** | **0.458** | **+0.184 (LOCAL OPTIMUM)** |
| **7th** | **aggressive variants** | **0.363** | **-0.095 (regression)** |

7 architectures explored; the 6th was the LOCAL OPTIMUM. No
combination using the project's currently-validated subsystems crosses
the 0.80 bar at biological scale.

## Mandatory smell-test (scrutinise PASS harder than FAIL): PASSED

Recomputed verdict from the single recording via the reused script:
- per-rung internal consistency: OK
- ladder + n_seeds: matched
- recomputed gate: FAIL (matches runner-reported exactly)

## Pre-registered next staged step

Per the standing autonomy + iterate-following-biology + the new
biology-translatable insight (sweet-spot exists; aggressive scaling
breaks balance):

**Either** (a) **Goldilocks-search**: revisit the 6th arc with SLIGHT
parameter variations to find a possibly-better optimum near the 6th
arc's recipe (e.g., 6th arc + ONLY persistent PFC-frame; 6th arc +
ONLY amplified tag stim; ablation studies to isolate which 7th-arc
mechanism caused most of the regression). This is a controller-only
diagnostic-cost iteration.

**Or** (b) **Honest closure of the gating + augmenting composition
design line**: 7 architectures explored; the 6th arc was the local
optimum at 0.458; the substrate's retrieval mechanism is asymptotically
capped without deeper subsystem refinement. The biology-translatable
insights (substrate-and-protocol-specific thresholds; encoding-
specificity at retrieve; sweet-spot for augmenting mechanisms) stand
as durable scientific contributions.

Per the standing autonomy directive, (a) is concretely actionable: an
ablation diagnostic on the 6th arc's recipe varied along ONE dimension
at a time. Each ablation is a 5-minute decisive eval; 4 ablations =
~20 min total. The outcome localises WHICH mechanism caused the most
regression and informs whether a refined 6th-arc-plus could continue
the trajectory.

NO bar change anywhere; protected set byte-empty diff vs `e8a99a2`
holds; no-confab moat 7/7; 4 calibrated moats byte-stable. 12
consecutive adversarial reviews; honest ceiling unchanged.

## Honest ceiling (binding throughout)

Conversational / compositional capability is NOT achieved and is NOT
claimed. The 7-architecture series + the cross-arc trajectory analysis
+ the discovery of the 6th arc as the local optimum are substantive
biology-translatable scientific contributions per the user's reframe
("biology-translatable insights ARE the deliverable").

## Files / evidence

- Decisive durable JSON: `research/findings/raw/targeted_replay_DECISIVE_fullscale.json`
- Decisive durable log: `research/findings/raw/targeted_replay_DECISIVE_fullscale.log`
- Smell-test recompute script (reused byte-unchanged across all arcs)
- Phase-1 cached checkpoints (reused; no retraining)
- 7th arc frozen verdict + runner: `targeted_cue_suppression_replay_core.py` + `targeted_cue_suppression_replay_runner.py`
- All previously-validated modules + calibrated moats byte-unchanged.
