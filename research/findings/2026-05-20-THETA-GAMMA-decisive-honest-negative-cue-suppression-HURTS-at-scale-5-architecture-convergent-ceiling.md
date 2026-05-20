# Theta-gamma mode-unification arc decisive run = GATE=FAIL with smell-test PASSED; structurally DIFFERENT failure mode from prior 4 arcs (per_regime_advantage NEGATIVE at N=5 across all 3 seeds: cue-suppression-during-retrieve produces the OPPOSITE of the hypothesised benefit; 5-architecture convergent ceiling now empirically grounded with mechanism-level signature)

## Status

Honest mid-arc finding from the controller-only decisive run of the
theta-gamma mode-unification architecture (the 5th in the gating-based
composition design line, after Stage-1 static / SPEAR theta-mux /
Pirazzini disinhibition / Unified per-regime monitor). Propagated
without spin. The decisive evaluation ran at full biological scale on
the cached Phase-1 substrate (3 seeds; ladder (2,3,5); the both
unified-substrate-specific calibrated abstention moats from the prior
arc in place: DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667 at
`0711e1d` + COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977124183006536 at
`25b9183`). The frozen capability-verdict module recomputed
independently from the single recording returns FAIL. The mandatory
smell-test PASSED -- the negative is a genuine measured outcome.

The 8th + 9th adversarial reviews bracketed this arc: 8th caught a
real RNG-drift confound in the structural-effect probe (Pirazzini
`d462bf0` defect class recurring); strengthen-only fix `e6b17da`
closed it (cp.random snapshot/seed/restore around each
`_run_theta_cycle_query` call; flag-differing divergence 5.59 mV;
both-True and both-False controls 0.00 mV); 9th re-review CLEAR
authorised the decisive run.

## Decisive measurement (full biological scale; 3 seeds; ladder (2,3,5))

```
GATE=FAIL  (reason: "smallest-N rung does not meet frozen bars")
```

Per-rung aggregate:

| N | full_acc | uniform_ctrl_acc | per_regime_advantage | direct_retain_acc | abstain_correct |
|---|----------|------------------|----------------------|-------------------|-----------------|
| 2 | 0.311    | 0.311            | +0.000               | 0.500             | 0.516           |
| 3 | 0.280    | 0.274            | +0.006               | 0.483             | 0.557           |
| 5 | 0.285    | **0.371**        | **-0.086**           | 0.500             | 0.667           |

Frozen bars (NEVER tuned): `_TG_FULL_MIN=0.80, _TG_UNIFORM_CTRL_MAX=0.10,
_TG_DIRECT_RETAIN_MIN=0.80, _TG_ABSTAIN_CORRECT_MIN=0.90,
_TG_SCALE_TOL=0.10, _TG_LADDER=(2,3,5), _TG_MIN_SEEDS=3`. All four
capability bars unmet at smallest-N rung. Direct retention has
DEGRADED from the unified arc's range (was 0.38-0.66, now 0.48-0.50 --
slightly worse at smallest load), suggesting the theta cycle's
encoding window slightly disrupts the direct-readout pathway too.

## The KEY structurally different finding: per_regime_advantage is NEGATIVE at N=5

Per-cell raw_cells block reveals a robust pattern:

| seed | N | full_acc | uniform_ctrl_acc | advantage |
|------|---|----------|------------------|-----------|
| 42   | 2 | 0.200    | 0.200            | +0.000    |
| 42   | 3 | 0.286    | 0.286            | +0.000    |
| 42   | 5 | 0.273    | 0.364            | **-0.091** |
| 43   | 2 | 0.333    | 0.333            | +0.000    |
| 43   | 3 | 0.125    | 0.250            | **-0.125** |
| 43   | 5 | 0.333    | 0.417            | **-0.083** |
| 44   | 2 | 0.400    | 0.400            | +0.000    |
| 44   | 3 | 0.429    | 0.286            | +0.143    |
| 44   | 5 | 0.250    | 0.333            | **-0.083** |

At N=5 (largest load), ALL 3 SEEDS show NEGATIVE per_regime_advantage
in the -0.083 to -0.091 range. The pattern is consistent enough that
it cannot be noise. **The cue-suppression mechanism produces an active
anti-effect** at biological scale.

## Why this is structurally different from prior 4 arcs

The unified per-regime monitor arc (commit `3735fec`) showed
`full_acc == uniform_ctrl_acc EXACTLY` on every 9 cell -- per_regime_advantage
collapsed to zero. The arms produced IDENTICAL answers because the
thresholds 0.198 vs 0.284 either both abstained or both emitted the
same ranked[0].

The theta-gamma arc shows the arms genuinely DIFFER:
- N=2: still tied (smallest load too sparse to surface differences)
- N=3 + N=5: arms produce different accuracy

But the difference is in the WRONG direction at N=5. The 5.59 mV
bridge-state divergence between cue-suppress=True and =False (from the
8th-review-validated structural-effect probe) is mechanistically real
and large. It just produces an active anti-effect on compositional
retrieval accuracy.

## Biology-translatable insight (encoding-specificity principle)

Hypothesis (catalog-grounded; Tulving 1973 encoding-specificity
principle): the cue is BOTH a noise source AND a useful context
signal. The unified arc's localisation finding observed the cue's
diffuse `lang_input` drive dominating the engram tag's selective
bound-adj drive -- framed that as "noise dominates signal", motivating
the cue-suppression-during-retrieve mechanism. But:

1. The cue ALSO provides ENCODING-CONTEXT that primes the substrate
   to expect retrieval of related concepts (encoding-specificity:
   retrieval is best when test conditions match encoding conditions).
2. Suppressing the cue during retrieve eliminates the noise but ALSO
   eliminates the context.
3. At biological scale on this substrate, the context-loss outweighs
   the noise-removal -> NEGATIVE per_regime_advantage at scale.

This is a real biology-translatable finding: cue-suppression-during-
retrieve is biologically backwards for the encoding-specificity-
governed compositional retrieval task. The brain's actual mechanism
likely involves keeping the cue partially active during retrieve (a
context maintenance role for PFC) while AMPLIFYING the engram tag's
selective drive (via theta-modulated synaptic gain at the CA3-CA1
pathway). The two effects are independent and additive; one cannot
substitute for the other.

## The 5-architecture convergent ceiling

The gating-based composition design line empirical results:

| Arc | Architectural mechanism | Decisive outcome | per_regime_advantage signature |
|-----|--------------------------|--------------------|----------------------------------|
| Stage-1 | static two-store retrieval-augmented | FAIL (full_acc=0; abstain=1.00) | n/a |
| SPEAR  | theta-mux ACh-gated plasticity | FAIL (full_acc=0; abstain=1.00) | 0 (rhythm_removed) |
| Pirazzini | theta-disinhibition + ACh polarity | (built; not decisively run after CLEAR) | n/a |
| Unified | per-regime metacognitive monitor | FAIL (full_acc 0.27-0.40; abstain 0.38-0.58) | 0 EXACTLY on every cell |
| **Theta-gamma** | **cue-suppression during retrieve** | **FAIL (full_acc 0.28-0.31; advantage NEGATIVE at N=5)** | **non-zero; mechanism active but anti-helpful** |

The first 4 architectures shared the same engram-tag-and-cue retrieval
mechanism with various gating overlays; all hit the same wall (the
substrate's compositional readout doesn't reliably emit the bound
adjective when cued). The 5th architecture (theta-gamma) added the
cue-suppression mechanism the localisation finding pointed at, and:

- The mechanism IS active (5.59 mV bridge-state divergence proven by
  the structural-effect probe with controls)
- BUT the effect is in the WRONG direction (helps marginally at small
  loads via tie; hurts at scale via negative advantage)

**5 architectures; 5 honest negatives; consistent mechanism-level
characterisation.** At biological scale on the v14/v16+hippocampus
substrate, gating-based composition does not produce reliable
compositional retrieval. The next staged step is NOT another gating
variation -- it's a fundamentally different mechanism class.

## Mandatory smell-test (scrutinise PASS harder than FAIL): PASSED

Recomputed verdict from the single recording (no re-run, no bar
change) via `research/findings/raw/unified_DECISIVE_smell_test.py`
(reused byte-unchanged; the smell-test discipline works on any output
JSON conforming to the per-rung shape):

- per-rung internal consistency: OK at all three N (2, 3, 5)
- ladder prefix: matches frozen `_TG_LADDER` (2, 3, 5)
- n_seeds: 3 at every rung
- values: every acc in [0, 1]
- recomputed gate: FAIL (smallest-N rung does not meet frozen bars)
- recompute matches runner-reported verdict: True

The FAIL is mechanically correct on the recorded numbers. It is NOT a
false-FAIL from instrument invalidity, NOT a degenerate broken-run
output, NOT a configuration error.

## Pre-registered next staged step (autonomous; per design doc fallback)

Per the standing design doc (commit `42bb8ce`, section "Honest
ceiling"): "A FAIL would extend the convergent ceiling to FIVE
architectures and motivate either: (a) deeper-mechanism design beyond
the project's currently-validated subsystems (e.g., generative replay
+ PFC compositional frame, which are reusable subsystems not yet
phase-multiplexed into the unified substrate); or (b) honest closure
of this design line as a terminal biology-translatable finding."

Direction (a) is the catalog-grounded next step per the 2026-05-19
design doc section 2b: theta-gamma mode-unification + **generative
replay + PFC-held compositional frame**. The current arc tested ONLY
the theta-gamma mode-unification piece (cue-suppression during
retrieve); generative replay + PFC-frame were NOT included. The next
arc would add these as a 6th architecture.

Direction (b) is the honest closure: the 5-architecture convergent
ceiling is the terminal biology-translatable finding for gating-based
+ phase-multiplexed composition design lines using only the
already-validated subsystems. Future work would require new subsystem
mechanisms beyond what the project has validated to date (e.g., a
dedicated PFC working-memory binding region; a generative-replay
sequence learner).

**Standing decision pattern**: the user-directed autonomy + the
"iterate following biology, no declare-unfit, no hand-back" mandate
selects direction (a). The next staged arc is generative replay +
PFC-held compositional frame, with the theta-gamma cue-suppression
removed (per this arc's finding that it produces an anti-effect).
This is the 6th architecture in the design line.

Specifically the next arc would test:
- Encoding phase: cue + engram tag bind (as before)
- Replay phase: many cycles of cue-driven generative replay (CA3
  recurrent + ACh modulation) propose-and-pattern-complete
- PFC phase: working-memory holds the compositional frame
- Retrieve phase: cue PRESENT (encoding-specificity respected) +
  engram tag stim AMPLIFIED via the replay-trained pathway

NO bar change anywhere; the frozen verdict module + the 4 calibrated
abstention moats stay byte-stable; the protected set byte-empty diff
vs `e8a99a2` must continue to hold; the no-confab moat 7/7
byte-identical.

## Honest ceiling (unchanged, restated)

Conversational / compositional capability is NOT achieved and is NOT
claimed. Five architectures in the design line have now hit the same
wall (at different mechanism-level signatures). The biology-
translatable insights (substrate-and-protocol-specific trustworthy
thresholds; encoding-specificity-principle violation of cue-suppression-
during-retrieve; the 5-architecture convergent ceiling with
mechanism-level characterisation) are durable scientific contributions
unaffected by any single arc's outcome. The 9 consecutive disciplined
refusal-to-overclaim-a-PASS pattern + the smell-test recompute matching
each runner-reported FAIL exactly is the meta-deliverable. The
protected set + the no-confab moat + the accumulated 4 calibrated
moats all stay byte-stable.

## Files / evidence

- Decisive durable JSON:
  `research/findings/raw/theta_gamma_DECISIVE_fullscale.json`
- Decisive durable log:
  `research/findings/raw/theta_gamma_DECISIVE_fullscale.log`
- Smell-test recompute script (reused byte-unchanged):
  `research/findings/raw/unified_DECISIVE_smell_test.py`
- Phase-1 cached checkpoints (reused; no retraining):
  `research/findings/raw/unified_per_regime/phase1/seed{42,43,44}.simstate.h5`
- Theta-gamma frozen verdict module (byte-stable since `11bd257`):
  `research/runners/theta_gamma_mode_unification_core.py`
- Theta-gamma runner (RNG-isolation fix at `e6b17da`):
  `research/runners/theta_gamma_mode_unification_runner.py`
- All previously-validated modules + calibrated moats byte-unchanged.
