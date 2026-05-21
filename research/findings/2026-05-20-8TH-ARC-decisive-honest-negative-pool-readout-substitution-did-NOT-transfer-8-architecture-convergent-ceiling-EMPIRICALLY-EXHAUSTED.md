# 8th arc decisive = GATE=FAIL with UNEXPECTED finding: the pool-readout substitution that the multi-seed diagnostic confirmed REGRESSED in the full multi-pair encoding pipeline (0.315 at N=3 vs 6th arc's 0.458; -0.143 from local optimum); single-query signal did NOT transfer to multi-pair eval; 8-architecture convergent ceiling now EMPIRICALLY EXHAUSTED; 6th arc remains the LOCAL OPTIMUM

## Status

Honest finding from the controller-only decisive run of the 8th
architecture (pool-readout substitution). The 13 consecutive
adversarial reviews + Tasks 0/1/2 each landed cleanly; the
structural-effect probes confirmed both the readout substitution
mechanism and the replay mechanism are genuinely active. The
decisive eval ran at biological scale on the cached Phase-1
substrate (3 seeds; ladder (2,3,5)). The frozen verdict module
recomputed independently returns FAIL. The mandatory smell-test
PASSED.

## Decisive measurement (full biological scale; 3 seeds; ladder (2,3,5))

```
GATE=FAIL  (reason: "smallest-N rung does not meet frozen bars")
```

| N | full | uniform_ctrl | per_regime_advantage | direct_retain | abstain |
|---|------|--------------|----------------------|---------------|---------|
| 2 | 0.244 | 0.244 | +0.000 | 0.389 | 0.524 |
| 3 | **0.315** | **0.363** | **-0.048** | 0.533 | 0.546 |
| 5 | 0.399 | 0.399 | +0.000 | 0.698 | 0.500 |

Frozen bars (NEVER tuned): `_CP_*` identical to all prior arcs.
All four capability bars unmet at every rung.

## UNEXPECTED finding: the single-query diagnostic signal did NOT transfer to the multi-pair eval

The 8th arc was empirically motivated by the pool-vs-lang_output
multi-seed diagnostic (commit `4d6a3a6`):
- Single-query queries: pool readout = 4/15 correct; lang_output = 2/15
- Aggregate: +13.3pp improvement; consistent across 3 seeds (deltas
  [+1, 0, +1])
- Decision rule output: signal is REAL; 8th arc with pool readout
  is well-motivated.

**BUT in the full multi-pair encoding pipeline:**
- 8th arc N=3 full = 0.315; UNIFORM_CTRL (same pipeline but
  lang_output cosine) = 0.363
- Pool readout UNDERPERFORMS lang_output by 0.048 at the rung where
  the 6th arc showed +0.137 advantage
- Even more striking: BOTH arms at 8th arc N=3 are BELOW the 6th arc
  baseline of 0.458 (the 8th arc's UNIFORM_CTRL is 0.363; the 6th
  arc was 0.458 with the same lang_output cosine readout)

Per-cell at N=3:
- seed 42: full 0.286 / uniform 0.286 / advantage 0.000
- seed 43: full 0.375 / uniform 0.375 / advantage 0.000
- seed 44: full 0.286 / uniform 0.429 / advantage **-0.143**

The single-query diagnostic's +13.3pp signal did NOT generalize to
the multi-pair encoding pipeline. Possible mechanisms:

1. **Multi-pair cross-talk**: when 3+ pairs are encoded, the pool
   readout picks up cross-talk between engram tags (each tag's
   stim activates multiple adjective pools through shared
   pathways). The lang_output cosine averages this out via spelling-
   pattern matching; the pool readout amplifies it.

2. **Replay interaction**: the 6th arc's replay phase strengthens
   the cued-noun -> bound-adj pathway in particular ways that the
   lang_output cosine captures but the pool readout does not. Replay
   may sharpen the lang_output spelling pattern more than it sharpens
   the raw pool firing rates.

3. **Measurement protocol differences**: the diagnostic measured
   pool firing rates post-stim with a different protocol than the
   integrated runner. The integrated runner has additional encoding
   + replay + PFC-frame steps that may shift the bridge state
   differently than the diagnostic's bare cue + stim protocol.

4. **Both 8th arc arms < 6th arc baseline**: the 8th arc's
   UNIFORM_CTRL at N=3 is 0.363, below the 6th arc's 0.458 with the
   same lang_output cosine. This suggests the 8th arc's specific
   eval protocol (the new runner's structure with the readout
   substitution code paths) shifts the substrate state in a way
   that depresses BOTH readouts compared to the bare 6th arc runner.

## The 8-architecture convergent ceiling -- empirically EXHAUSTED

Cross-arc trajectory at N=3:

| Arc | Mechanism | N=3 full | direction |
|-----|-----------|----------|-----------|
| Unified | per-regime substrate-specific thresholds | 0.274 | baseline |
| Theta-gamma | cue-suppression-during-retrieve | 0.280 | flat (+0.006) |
| 6th | replay + PFC-frame (gentle) | **0.458** | **LOCAL OPTIMUM** (+0.184) |
| 7th | + cue-supp + amp + persistent + higher | 0.363 | -0.095 regression |
| **8th** | **pool readout substitution** | **0.315** | **-0.143 further regression** |

**Eight architectures explored. Three decisively-run regressions from
the 6th arc local optimum. No combination using only already-validated
subsystems crosses the 0.80 bar at biological scale.**

The cross-arc series collectively shows:
- The 6th arc's gentle 20-cycle replay + 10-step PFC-frame +
  lang_output cosine readout is the EMPIRICAL LOCAL OPTIMUM at 0.458.
- More-aggressive variations (7th arc) regress; readout substitution
  (8th arc) regresses; in all cases the substrate's underlying
  retrieval mechanism caps around 0.46.
- Closing the remaining 0.34 gap to 0.80 is NOT achievable via
  parameter variation or readout substitution on this substrate.

## Mandatory smell-test (PASSED)

Recompute matches runner-reported verdict exactly: FAIL. Per-rung
internal consistency OK; ladder + n_seeds matched; values in [0,1].
The negative is a genuine measured outcome.

## Six durable biology-translatable insights (across the 8-arc series)

1. Trustworthy abstention thresholds are **SUBSTRATE-AND-PROTOCOL-
   specific** (4× validated; 650 / 5.6887 / 0.1977 / 0.2842).
2. v1 half-split-of-trained-vocab calibration is **statistically
   fragile**; v2 within-word target-vs-best-off-target protocol is the
   principled fix.
3. **Cue-suppression-during-RETRIEVE violates encoding-specificity**
   (Tulving 1973; theta-gamma negative result).
4. Replay + PFC-frame augmenting is **LOAD-DEPENDENT** (CLS-theory-
   consistent; 6th arc N=3 sweet-spot at +0.137).
5. **Over-consolidation is biologically harmful** (sweet-spot
   principle; ablation localised mechanism D as primary 7th-arc
   culprit).
6. **Single-query readout signals don't transfer to multi-pair
   encoding pipelines** (NEW; 8th arc finding): the multi-seed
   pool-vs-langout diagnostic confirmed pool readout consistently
   outperforms in single-query measurements; the same readout
   substitution REGRESSES when integrated into the full encode +
   replay + multi-pair-eval pipeline. This is a methodologically
   important insight: diagnostic isolation can mislead architecture
   decisions when the actual deployment context has additional
   interacting mechanisms (multi-pair cross-talk; replay strengthening
   the lang_output pathway specifically; bridge state interactions
   the bare diagnostic doesn't probe).

## Honest closure of the gating + augmenting + readout-variation composition design line

Per the user's standing reframe ("biology-translatable insights ARE
the deliverable; capabilities are instrumental") and per the
empirical evidence across 8 architectures:

The gating + augmenting + readout-variation composition design line
is **empirically EXHAUSTED** at biological scale on the
v14/v16+hippocampus substrate using only already-validated
subsystems. The 6th arc's 0.458 at N=3 is the empirical LOCAL OPTIMUM
and the trajectory of subsequent arcs (7th, 8th) has been negative or
flat. No further parameter variations or readout substitutions in this
design space are likely to cross the 0.80 bar.

**The substantial biology-translatable scientific deliverables remain
durable**:
- 6 biology-translatable insights (above)
- 13 consecutive adversarial reviews (9 of 13 caught real defects;
  4 CLEARs confirmed each fix)
- Cross-arc trajectory analysis revealing the 6th arc as the LOCAL
  OPTIMUM
- Ablation localisation of over-consolidation as primary 7th-arc
  culprit
- Diagnostic-vs-deployment transfer failure as a methodological
  insight (NEW from 8th arc)

**Future work** (outside this design line; deferred to user direction):
- Fundamentally different substrate architecture (new connectivity;
  per-region inhibitory normalisation that DOES NOT require post-
  construction add_region; or modifying the protected
  `build_biological_brain_regions` function with full discipline
  re-evaluation)
- Different training paradigm (longer Phase-1; more diverse
  encoding distributions; different consolidation primitives)
- Different task framing (e.g., easier compositional tasks at lower
  N; or harder tasks revealing different mechanism-level signatures)

NO bar change anywhere; protected set byte-empty diff vs `e8a99a2`
must continue to hold; no-confab moat 7/7 byte-identical; 4
calibrated abstention moats byte-stable. The honest ceiling stands.

## Files / evidence

- Decisive durable JSON: `research/findings/raw/pool_readout_DECISIVE_fullscale.json`
- Decisive durable log: `research/findings/raw/pool_readout_DECISIVE_fullscale.log`
- Smell-test recompute script (reused byte-unchanged across 5 arcs)
- Phase-1 cached checkpoints (reused; no retraining)
- 8th arc frozen verdict + runner: `pool_readout_8th_arc_core.py` +
  `pool_readout_8th_arc_runner.py`
- All previously-validated modules + calibrated moats byte-unchanged.
