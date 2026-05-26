# Direction 4 BUGFIX = DIRECTION_4_PASS multi-seed (6 of 6 cells PASS at 0.80 bar with massive margins; L=2/3 perfect OB+OI; L=5 OB perfect / L=5 OI 0.983); MAJOR REVERSAL of prior D4 NEGATIVE — the bio_brain_regions DEDICATED-POOL architecture genuinely supports cross-bridge composition once cross-bridge uniformity is fixed; OUTPERFORMS D5 HYBRID at L=5 OI (0.983 vs 0.195/0.463) suggesting dedicated category pools beat shared sparse pool for the FHRR-cross-bridge channel at smoke scale.

**Date:** 2026-05-26 ~06:50 EDT
**Status:** DIRECTION_4_PASS (6/6 cells PASS); pillar-level candidate pending production-scale validation. The biology-translatable finding REVERSES the prior NEGATIVE narrative AND the global_mean diagnostic conclusion: the bio_brain_regions DEDICATED-POOL architecture (each bridge owns its own 100-200-neuron pool per concept) genuinely supports cross-bridge mode-unification composition once each bridge is initialized with a distinct random seed. The smoke result is STRONGER than the D5 HYBRID bugfix smoke (PARTIAL 5/6; L=5 OI 0.195-0.463). Multi-seed full-scale retrain is the immediate next step to confirm the smoke pattern under production parameters.

## What was tested

After the 2026-05-26 ~00:30 EDT D4 diagnostic revealed a CRITICAL bug
analogous to D5 commit c4e18f2 — all 5 bridges at the same base seed
produced byte-identical activity per word position (cos = 1.0000;
np.array_equal == True) because the SimulationBridge / build_biological
_brain_regions weight initialization is a deterministic function of
`cfg.seed`, and orthogonal_drive_pattern is deterministic per (cue_idx,
n_cues) — identical weights + identical drive = identical activity,
making cross-bridge discrimination mathematically impossible — the bug
was fixed via `_DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS` (100k offsets
per bridge: A_nouns=0 / B_verbs=100k / C_adj=200k / D_spatial=300k /
E_functional=400k) inside `direction_4_bridge_builder._build_bridge_core`
(the shared constructor body for all 5 per-bridge wrappers). D4 SMOKE was
then re-trained from scratch with the fix.

## Files modified (bugfix scope)

Single-file fix, mirrors D5 commit c4e18f2 byte-pattern:

- `research/findings/raw/direction_4_bridge_builder.py`:
  - Added `_DIRECTION_4_BRIDGE_LABEL_SEED_OFFSETS` dict at module level
    (5 bridges, offsets at 100k increments)
  - In `_build_bridge_core`: derive `bridge_seed = seed + _bridge_seed_offset`
    where the offset is looked up via the `label` argument (each per-bridge
    wrapper already passes a unique label: A_nouns / B_verbs / C_adj /
    D_spatial / E_functional)
  - `cfg.seed = bridge_seed` (was `cfg.seed = seed`)
  - Defensive fallback for unknown labels: hash-based offset in
    [100000, 999999] (should never trigger in production)
  - Verbose print extended to log `base_seed` AND `bridge_seed` AND
    `offset` for audit transparency

The runner (`direction_4_5bridge_runner.py`), cross-bridge probe
(`direction_4_cross_bridge_probe.py`), vocab spec, and verdict module
are UNCHANGED. The protected `build_biological_brain_regions` and
`SimulationBridge` are UNCHANGED. The bar (0.80 multi-seed) is
UNCHANGED. No autograd. No-confab moat green.

## Activity-distinctness verification (post-bugfix, seed 42)

Direct cross-bridge cosine on per-neuron spike-count vectors for one
canonical word per bridge (cf. pre-bugfix where all 5 were
np.array_equal == True with cos = 1.000000):

| Comparison | cos | byte-identical |
|---|---|---|
| A_nouns[apple] vs A_nouns[apple] | 1.000000 | True (self) |
| A_nouns[apple] vs B_verbs[go]   | 0.009838 | **False** |
| A_nouns[apple] vs C_adj[big]    | 0.015266 | **False** |
| A_nouns[apple] vs D_spatial[north] | 0.027249 | **False** |
| A_nouns[apple] vs E_functional[i]  | 0.010080 | **False** |

The 5 bridges now produce orthogonal (cos ~0.01-0.03) cross-bridge
activity vectors. The cross-bridge probe is no longer operating on
duplicate inputs.

Per-bridge mean firing rates (also distinct, confirming each bridge
maintains its own dynamics):

| Bridge | canonical word | mean_rate | density |
|---|---|---|---|
| A_nouns | apple | 0.0743 | – |
| B_verbs | go    | 0.0819 | 0.0367 |
| C_adj   | big   | – | – |
| D_spatial | north | 0.0664 | 0.0258 |
| E_functional | i | 0.0605 | 0.0251 |
| E_functional | he | 0.0766 | 0.0345 |

## Cross-bridge probe result (3 seeds × 3 loads × 2 readouts; smoke scale)

**Multi-seed mean accuracy:**

| Load | OB (order-bearing) | OI (order-invariant) |
|---|---|---|
| L=2 | **1.000** | **1.000** |
| L=3 | **1.000** | **1.000** |
| L=5 | **1.000** | **0.983** |

Per-seed breakdown:
- seed 42: L=2 OB/OI 1.000/1.000; L=3 OB/OI 1.000/1.000; L=5 OB/OI 1.000/0.980
- seed 43: L=2 OB/OI 1.000/1.000; L=3 OB/OI 1.000/1.000; L=5 OB/OI 1.000/0.995
- seed 44: L=2 OB/OI 1.000/1.000; L=3 OB/OI 1.000/1.000; L=5 OB/OI 1.000/0.975

**Verdict (frozen, pre-registered): DIRECTION_4_PASS** (6 of 6 cells
PASS at the 0.80 bar, with margins ranging from +0.175 to +0.200).

batched-vs-scalar max-diff across all 3 seeds: ≤ 2.78e-17 (instrument
validity confirmed; batched primitive numerically identical to scalar
reference).

## Comparison to D5 bugfix smoke + production

| Metric | D5 hybrid bugfix smoke | D5 hybrid production | **D4 bugfix smoke (this run)** |
|---|---|---|---|
| Architecture | shared sparse pool + dedicated readout (HYBRID) | same | **dedicated category pools (NO shared pool)** |
| Verdict | PARTIAL 5/6 cells | PARTIAL (5/6 cells; L=5 OI below bar) | **PASS 6/6 cells** |
| L=2 OB / OI | 1.000 / 1.000 | – | **1.000 / 1.000** |
| L=3 OB / OI | 1.000 / 0.840 | – | **1.000 / 1.000** |
| L=5 OB / OI | 1.000 / 0.195 (baseline) | – | **1.000 / 0.983** |
| L=5 OI (top-K decoder fix) | 0.463 | – | – (not needed; baseline already 0.983) |

**D4 bugfix smoke OUTPERFORMS D5 hybrid bugfix smoke at L=5 OI by ~5x
the absolute margin** (0.983 vs 0.195) and is the FIRST cross-bridge
result to achieve PASS verdict at the 0.80 bar across ALL 6 cells of
the smoke matrix without requiring a decoder fix. This is unexpected
and suggests the dedicated-pool architecture has BETTER cross-bridge
FHRR mode-unification properties than the shared sparse pool — at
least at smoke scale.

**Hypothesis (must be confirmed at production scale):** The dedicated
pool architecture gives each concept its own ~200-neuron attractor
with sharper recall (lower overlap between concepts WITHIN a bridge),
whereas the shared sparse-pool architecture has higher concept-concept
overlap at the K-of-N pattern level. The cross-bridge global mean-
centring primitive may benefit more from sharp within-bridge
discrimination than from the sparse-pool population coding.

## Comparison to D4 NEGATIVE (now INVALIDATED) and D4 global_mean diagnostic (also INVALIDATED)

| Artifact | Status | Why |
|---|---|---|
| 2026-05-25 D4 NEGATIVE finding | **INVALIDATED 2026-05-26** | Operated on duplicate cross-bridge inputs (uniformity bug) |
| 2026-05-25 D4 global_mean diagnostic | **INVALIDATED 2026-05-26** | Same bug; "geometry-limited" conclusion unsupported |
| **2026-05-26 D4 BUGFIX SMOKE (this finding)** | **DIRECTION_4_PASS multi-seed** | Pattern uniqueness fixed; 6/6 cells PASS |

The substrate-geometry critique that emerged from the D4 NEGATIVE arc
was wrong because the inputs were identical. The bio_brain_regions
DEDICATED-POOL architecture genuinely supports cross-bridge FHRR
mode-unification composition.

## Biology-translatable insight

The cortical canon principle that motivates dedicated category pools
(distinct nominal / verbal / adjectival / spatial / functional cortical
fields per Pulvermüller-2001 distributed cortical word ensembles +
Tepper-2018 cortical column microcircuits) appears robustly compatible
with FHRR-mediated cross-cortical-field composition once each field
has its own random structural microcircuit (i.e., the equivalent of
distinct developmental noise per cortical area). The "bug" in computer
science terms — identical PRNG seeds across the 5 bridges — has a
biological analog: a developmental coupling that produced byte-identical
microcircuits across the 5 cortical areas would be both biologically
implausible (cortical areas develop with INDEPENDENT noise drawn from
the same statistical distribution) and computationally pathological
(no cross-area discrimination possible). The bugfix mirrors the
biological reality of independent developmental noise per cortical
area, and the resulting PASS verdict suggests the dedicated-pool
architecture is a viable substrate for cross-cortical-field
compositional binding via FHRR.

## What's still pre-registered + frozen

- The cross-bridge probe primitive (parallel_population_matching_batched)
  is UNCHANGED (reuses pillar n=95 byte-pattern).
- The frozen verdict module (`direction_4_verdict.py`) is UNCHANGED:
  thresholds OB_MIN=0.80, OI_MIN=0.80, LOADS=[2,3,5], MIN_SEEDS=3.
- The bar 0.80 multi-seed is UNCHANGED.
- The protected `sim/bridge.py` / `build_biological_brain_regions`
  modules are UNCHANGED.
- The D4 runner (`direction_4_5bridge_runner.py`) is UNCHANGED
  (KILL-SAFE cache short-circuit verified during the resume after a
  harness-level interruption mid-training).
- The D5 hybrid bridge builder (separate file, separate concern) is
  UNCHANGED.

## Pre-registered next concrete actions

1. **Re-launch D4 5-bridge FULL-SCALE multi-seed retrain** (n_lang_input
   =2048, n_per_pool=200, n_events=200, M_OBS=16; ~7-15 hr GPU). The
   smoke PASS pattern is so strong (L=5 OI 0.983 vs bar 0.80) that
   full-scale should pass unless there's a smoke-vs-full-scale
   regression. Critical to validate before claiming the pillar.
2. **If full-scale PASS:** promote pillar n=108 candidate (D4 dedicated-
   pool cross-bridge composition). Update webapp/capability_status.json.
3. **If full-scale BOUNDARY/PARTIAL:** investigate why smoke is so much
   stronger than D5 hybrid at the same L=5 OI cell (decoder choice?
   batch noise floor? per-neuron-count effect?).
4. **Cross-architecture comparison:** the D4 PASS + D5 PARTIAL contrast
   at the same evaluation framework strongly suggests "dedicated pools
   beat shared sparse pool for FHRR cross-bridge at this scale," but
   the production-scale comparison is needed for confidence.

## Wall-clock + artifacts

- Smoke retrain: ~25.4 min total (initial run got 11 cells before
  harness interruption; KILL-SAFE resume completed remaining 4 cells +
  probe in ~22 min)
- Cross-bridge probe: 120.9s wall-clock (CuPy backend)
- 15 trained bridges (5 × 3 seeds), 15 activity caches, 1 runner JSON,
  1 standalone probe JSON

**Files written:**
- `research/findings/raw/direction_4_5bridge_smoke_bugfix.json`
- `research/findings/raw/direction_4_5bridge_smoke_bugfix.log`
- `research/findings/raw/direction_4_cross_bridge_bugfix_smoke.json`
- `research/findings/raw/direction_4_cross_bridge_bugfix_smoke.log`
- 15 × `research/findings/raw/direction_4_cache/activity_smoke_*.npz`
- 15 × `research/findings/raw/direction_4_cache/bridge_smoke_*.simstate.h5`

## Discipline

- Bug fix scoped to a single non-protected file
  (`direction_4_bridge_builder.py`); the protected builder, bridge,
  runner, cross-bridge probe, vocab spec, and verdict module are
  UNCHANGED. Mirrors D5 c4e18f2 byte-pattern.
- Honest propagation: D4 NEGATIVE finding (2026-05-25) and D4
  global_mean diagnostic (2026-05-25) BOTH retain INVALIDATED status
  from the 2026-05-26 ~00:30 EDT diagnostic; this PASS finding is the
  REVERSAL.
- Smoke result is reported with full per-seed numbers; the PASS verdict
  is from the frozen pre-registered verdict module (no post-hoc
  threshold movement).
- Bar 0.80 multi-seed UNCHANGED.
- Smoke-scale results are NOT propagated as the production result;
  full-scale validation is the next concrete action.
- Both remotes will be pushed.
