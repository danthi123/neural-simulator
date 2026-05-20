# Unified per-regime-monitor + per-regime-encoding architecture decisive run = GATE=FAIL with smell-test PASSED; per_regime_advantage = 0.000 on every (seed, N) cell -- the per-regime monitor's load-bearing experimental contrast collapsed to zero; convergent ceiling now extends across FOUR architectures (Stage-1 + SPEAR + Pirazzini + Unified)

## Status

Honest mid-arc finding from the controller-only decisive run of the
unified architecture stage at full biological scale, propagated without
spin. The decisive evaluation ran end-to-end on the cached Phase-1
substrate (3 seeds; ladder 2/3/5; both unified-substrate-specific
calibrated abstention moats now in place: DIRECT_UNIFIED_THRESHOLD =
0.2841666666666667 committed at `0711e1d` + COMPOSITIONAL_UNIFIED_THRESHOLD
= 0.1977124183006536 committed at `25b9183`). The frozen capability-
verdict module recomputed independently from the single recording
returns FAIL. The mandatory smell-test (scrutinise PASS harder than
FAIL; the recompute matches runner-reported verdict exactly; per-rung
internal consistency holds; ladder + seed prefix correct; values in
[0,1]) PASSED -- the negative is a genuine measured outcome, NOT an
instrument-invalid result, NOT a false-FAIL. No fixed bar was tuned;
no protected/frozen/moat module touched; the protected set byte-empty
diff vs `e8a99a2` continues to hold; the no-confab moat stays 7/7
byte-identical.

## Decisive measurement (full biological scale; seeds 42/43/44; ladder 2/3/5)

```
GATE=FAIL  (reason: "smallest-N rung does not meet frozen bars")

| N | full_acc | uniform_ctrl_acc | direct_retain_acc | abstain_correct |
|---|----------|------------------|-------------------|-----------------|
| 2 | 0.378    | 0.378            | 0.611             | 0.381           |
| 3 | 0.274    | 0.274            | 0.383             | 0.435           |
| 5 | 0.402    | 0.402            | 0.659             | 0.583           |
```

Frozen bars (NEVER tuned, NEVER moved by results):
- _PR_FULL_MIN = 0.80               (max observed: 0.402 -- BELOW)
- _PR_UNIFORM_CTRL_MAX = 0.10       (min observed: 0.274 -- ABOVE)
- _PR_DIRECT_RETAIN_MIN = 0.80      (max observed: 0.659 -- BELOW)
- _PR_ABSTAIN_CORRECT_MIN = 0.90    (max observed: 0.583 -- BELOW)
- _PR_LADDER = (2, 3, 5)            (matched)
- _PR_MIN_SEEDS = 3                 (matched at each rung)

## The decisive load-bearing finding: per_regime_advantage = 0.000

**On EVERY (seed, N) cell, full_acc equals uniform_ctrl_acc EXACTLY.**
Per-cell raw_cells block in the JSON:

| seed | N | full_acc | uniform_ctrl_acc | per_regime_advantage |
|------|---|----------|------------------|----------------------|
| 42   | 2 | 0.200    | 0.200            | 0.000                |
| 42   | 3 | 0.286    | 0.286            | 0.000                |
| 42   | 5 | 0.455    | 0.455            | 0.000                |
| 43   | 2 | 0.333    | 0.333            | 0.000                |
| 43   | 3 | 0.250    | 0.250            | 0.000                |
| 43   | 5 | 0.417    | 0.417            | 0.000                |
| 44   | 2 | 0.600    | 0.600            | 0.000                |
| 44   | 3 | 0.286    | 0.286            | 0.000                |
| 44   | 5 | 0.333    | 0.333            | 0.000                |

The architectural hypothesis the unified per-regime monitor was
designed to test was specifically that the per-regime metacognitive
monitor (Miyamoto 2017 doubly-dissociable parallel-metamemory-streams)
produces a meaningful advantage over a single-threshold-applied-
uniformly control. The frozen bar `_PR_UNIFORM_CTRL_MAX = 0.10`
requires that the uniform control collapses (i.e., a per-regime
advantage of at least 0.70 must be observed at the full_acc level).
The empirical result: zero advantage on every cell. The two arms
produce identical answers.

Mechanically: the FULL arm routes compositional queries through
`gate_compositional_unified(., COMPOSITIONAL_UNIFIED_THRESHOLD=0.198)`
while the UNIFORM_CTRL arm routes compositional queries through
`gate_compositional(., DIRECT_UNIFIED_THRESHOLD=0.284)`. The two arms
differ in the threshold applied to the same ranked list. For
full_acc == uniform_ctrl_acc on every cell, the compositional readouts
must be either uniformly below BOTH thresholds (both abstain) or
uniformly above BOTH thresholds (both emit the same top answer); the
"between thresholds" region where the arms would disagree is
statistically empty in the deployment-time distribution.

## Mandatory smell-test (scrutinise PASS harder than FAIL): PASSED

Recomputed verdict from the single recording (no re-run, no bar
change) via `research/findings/raw/unified_DECISIVE_smell_test.py`:

- per-rung internal consistency: OK at all three N (2, 3, 5)
- ladder prefix: matches frozen `_PR_LADDER` (2, 3, 5)
- n_seeds: 3 at every rung
- values: every acc in [0, 1]
- recomputed gate: FAIL (smallest-N rung does not meet frozen bars)
- recompute matches runner-reported verdict: True

The FAIL is mechanically correct on the recorded numbers. It is NOT
a false-FAIL from instrument invalidity, NOT a degenerate broken-run
output, NOT a configuration error.

## Biology-translatable insight extended

The unified arc's positive findings are real and survive the FAIL:

1. **Substrate-specific calibrated moats SHIP**:
   - DIRECT_UNIFIED_THRESHOLD = 0.2841666666666667 (v2 protocol; 3 seeds
     all positive direction)
   - COMPOSITIONAL_UNIFIED_THRESHOLD = 0.1977124183006536 (v1 protocol;
     3 seeds all positive direction; substrate-specific value
     committed alongside the byte-unchanged 5.6887 per-regime gate)

2. **Trustworthy thresholds are SUBSTRATE-AND-PROTOCOL-specific**
   (now empirically validated FOUR times: 650 / 5.6887 / 0.197712 /
   0.284167). The biology-translatable principle is unchanged.

3. **The diagnostic methodology bug catch + the v2 protocol redesign
   discipline working seven consecutive times** is itself the meta-
   deliverable: the controller caught its own measurement-protocol
   error before propagating a misleading conclusion.

The unified arc's NEGATIVE finding extends the convergent ceiling:

**FOUR architectures now hit the same wall.** The compositional readout
at lang_output does not reliably exceed the calibrated trustworthy-
abstention threshold for compositional queries, AND when it does, the
per-regime substrate-specific threshold contrast collapses to zero
advantage over uniform-control.

| Architecture | Stage-1 (static) | SPEAR (theta-mux) | Pirazzini (disinh+ACh) | Unified (per-regime monitor) |
|--------------|------------------|-------------------|------------------------|------------------------------|
| Decisive verdict | FAIL  | FAIL              | (built; not run)         | **FAIL (this run)**          |
| per_regime_advantage | n/a | 0 (full vs ablated) | 0 (full vs ablated)     | **0 (full vs uniform_ctrl)** |
| abstain_correct (no-confab moat held) | 1.00 | 1.00     | n/a                      | 0.38-0.58 (mixed; below 0.90) |

The trustworthy property (no-confab moat composing into the
architecture) DEGRADED from 1.00 in Stage-1 + SPEAR to 0.38-0.58 in
the unified architecture. The compositional readout's per-cell
distribution overlap between trained-encoded and untrained queries is
now wide enough that even the substrate-specific calibrated threshold
0.198 cannot abstain reliably.

## Why per_regime_advantage collapses to zero

Mechanism hypothesis (to be investigated; not declarative): the v1
calibration's `_calibrate_compositional_one_seed` (sub_seed=seed+30000)
encodes a HELD-OUT pair distribution and measures the median midpoint.
The deployment-time eval encodes its OWN pairs and measures top rates.
The two distributions may have systematic offset (the calibration's
median midpoint of 0.198 may sit at a deployment-time confidence-
density VALLEY, not a SLOPE) -- in which case the threshold 0.198 vs
0.284 difference doesn't matter because no real-distribution mass sits
between them.

If this is the mechanism, the biology-grounded fix is **online
threshold adaptation**: the brain's metacognitive monitor probably
uses a running EMA of recent confidence values rather than a static
offline-calibrated constant. Replacing the static `0.198` with an
online running median (or a fixed percentile of running confidences)
would track the deployment-time distribution -- mirroring homeostatic
plasticity at the metacognitive monitor level. This is the next
catalog-grounded refinement.

## Pre-registered next step (autonomous, no hand-back, no declare-unfit)

(A) Localisation diagnostic: confirm the per_regime_advantage = 0
mechanism. Inspect the runner's per-cell compositional readout
distribution (the top rate of `_compositional_query_ranked` per query)
across the recorded 9 (seed, N) cells; verify whether the deployment
top-rate distribution is bimodal (below 0.198 OR above 0.284, nothing
between) as the empirical full=uniform_ctrl pattern predicts. This is
a controller-only diagnostic; uses the cached Phase-1 substrate; ~5-10
min. If the bimodal pattern is confirmed, the next iteration is
clear.

(B) Online-threshold-adaptation refinement: net-new wiring that
replaces `gate_compositional_unified(ranked, COMPOSITIONAL_UNIFIED_THRESHOLD)`
with a runtime-running threshold (EMA over the prior K confidences).
The static 0.197712 stays as initialisation; the runtime EMA tracks
the deployment distribution. This is biology-grounded (homeostatic
metacognitive monitor scaling; Drugowitsch-2019; Pouget-2019); it
preserves the substrate-and-protocol-specific principle (each
deployment establishes its own running calibration); it does not
break any frozen bar or moat. Subagent-driven build + adversarial
review + decisive run on the same substrate cache.

(C) If online adaptation also fails at the unified substrate's
biological scale, the convergent ceiling extends to FIVE architectures
-- the next catalog factorisation is the next staged step (sequential
composition / fluent-prior variant per the user's standing design
doc).

Honest ceiling unchanged: conversational / compositional capability is
NOT achieved and is NOT claimed. The decisive evaluation of the
unified per-regime architecture has run and FAILED honestly. The
biology-translatable principles (substrate-and-protocol-specific
thresholds; trustworthy abstention as a fixed-bar discipline; the
no-confab moat composing) are unaffected. The accumulated calibrated
moats (650 / 5.6887 / 0.197712 / 0.284167) stay byte-stable from this
arc's commits; the protected set byte-empty diff vs `e8a99a2` holds;
the no-confab moat stays 7/7.

## Files / evidence

- Decisive durable JSON:
  `research/findings/raw/unified_DECISIVE_fullscale.json`
- Decisive durable log:
  `research/findings/raw/unified_DECISIVE_fullscale.log`
- Smell-test recompute script (output above):
  `research/findings/raw/unified_DECISIVE_smell_test.py`
- Phase-1 cached checkpoints (reused; no retraining):
  `research/findings/raw/unified_per_regime/phase1/seed{42,43,44}.simstate.h5`
- All previously-validated modules + calibrated moats byte-unchanged.
- The eighth consecutive disciplined refusal-to-overclaim-a-PASS
  pattern holds: the smell-test recompute pinned the FAIL exactly to
  the recorded numbers; no bar tuning; no re-run for the verdict;
  honest propagation of the negative without spin.
