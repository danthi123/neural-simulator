# Direction 5 design: HYBRID sparse-distributed shared pool ON bio_brain_regions (5 bridges × V=16 = 80 cross-bridge concepts; biology-faithful dedicated pools + Kanerva sparse cross-bridge substrate)

**Date:** 2026-05-25
**Status:** NEW direction; post-D4-NEGATIVE pivot; pre-staged after Direction 4 5-bridge SMOKE
returned NEGATIVE multi-seed (commit 611027c) + the global_mean centring diagnostic ruled
out the cheap fix (commit ca5b000). Pre-launch grep CONFIRMED net-new (no prior
`direction_5_*.py`; no prior hybrid sparse-on-bio_brain_regions work; the closest precedent
is the G.20 sparse 5-bridge pillar n=95 on a separate (sparse-only) substrate and the
bio_brain_regions OPTION 3 pillar n=96 on a separate (dedicated-pool-only) substrate).

## Goal

Test the **hybrid hypothesis** directly: combine the best-of-both worlds — biology-faithful
dedicated 200-neuron concept pools (per bio_brain_regions OPTION 3 / pillar n=96 substrate +
Direction 3 V=32 pillar n=105) PLUS a sparse-distributed K-of-N shared substrate per bridge
(per G.20 sparse pillar n=95 mechanism) — on the SAME 5-bridge ensemble that D4 failed to
make work, so that cross-bridge composition reads OUT of the shared sparse substrate
(where the geometry IS sufficient per pillar n=95) while the per-concept dedicated pools
preserve biology-faithful in-bridge attractor dynamics (where pillars n=96/n=97/n=98/n=105
validated discrimination is sufficient).

If PASSES: pillar n=106 candidate; conversational substrate extended to 5 bridges × 32
concepts = 160 biology-faithful cross-bridge concepts on a UNIFIED architecture (FIRST
architecture that combines biology-faithful dedicated pools with sparse-distributed
cross-bridge composition). Path to scale further to V=32 × 5 = 160 cross-bridge concepts
(per the Direction 3 V=32 pillar n=105 single-bridge result) is then a parameter scale-up.

If FAILS: sharper diagnostic. Two distinct hypotheses become testable independently:
- H1 (dedicated pools insufficient): IS the dedicated-pool substrate the bottleneck even
  with a sparse cross-bridge readout? → falsifies pillar n=96/n=105 reach.
- H2 (sparse readout insufficient): IS the sparse cross-bridge substrate failing when
  bound to dedicated-pool drivers? → falsifies pillar n=95 reach to bio-driven readout.

## Biology reference

**Cortical columns + distributed population codes**: Hubel & Wiesel (1962, 1968) discovered
ocular dominance / orientation columns in V1 — DEDICATED columnar substrates per concept.
**Pulvermüller (1999, 2001)** showed that the same cortical area ALSO carries a
distributed, sparse population code for word concepts that spans the cortical sheet —
SHARED scattered patterns ACROSS the same neurons.

These two organizational principles **co-exist** in real cortex: a given pyramidal neuron
both participates in its column's preferred feature (dedicated) AND participates in
multiple sparse distributed codes for different higher-order concepts (shared). The
classical view (Mountcastle 1957 cortical columns; Pulvermüller distributed cell
assemblies) is NOT contradictory — these are TWO substrates on the SAME tissue.

Direction 5 mirrors this dual organization: each bridge carries BOTH the bio_brain_regions
dedicated pools (per concept, 200-neuron attractor with FS interneurons, lang_input /
lang_output pathways) AND a sparse-distributed shared substrate (a 2000-neuron pool where
each concept has a K=100 random pattern; patterns OVERLAP between concepts at expected
overlap K²/N ≈ 5 per Pulvermüller / Kanerva). Training co-activates both substrates;
cross-bridge probe reads OUT of the shared substrate.

Catalog ref: Hubel & Wiesel (1962, 1968); Mountcastle (1957); Pulvermüller (1999, 2001);
Kanerva (1988); Foldiak (1990).

## Approach selection

**Approach A — PER-BRIDGE shared sparse pool, dedicated → shared one-time projection** (RECOMMENDED):
- Per-bridge architecture: existing bio_brain_regions dedicated pools (16 concepts × 200
  neurons + FS + motor canon + lang_input / lang_output) PLUS one NEW region
  `shared_concept_pool` (2000 neurons, 5% density, weak dynamics) PER BRIDGE
- Per-concept binding: at TRAINING time, when concept C is taught, drive both the dedicated
  pool C (existing lang_input → pool C STDP) AND drive the K=100 sparse pattern in the
  bridge's shared_concept_pool (via a NEW one-time topographic prior on lang_input →
  shared_concept_pool pathway, mirroring G.20 sparse `apply_sparse_topographic_prior`)
- Cross-bridge probe: at probe time, sample activity from the shared_concept_pool across
  all 5 bridges (each bridge contributes its 16 concepts × 2000-neuron sparse pattern
  activity); compose via parallel-matching mode-unification on the SHARED sparse substrate
  (matching G.20 sparse pillar n=95 byte-unchanged in primitive)
- Cost: per-bridge ~17 min train (matching v14/v16 + extra sparse pathway init); 5 bridges
  × 3 seeds = 15 bridge trains = ~4-5 hr GPU; cross-bridge probe ~10 min CPU

**Approach B — CROSS-BRIDGE shared sparse pool (1 substrate)**:
- One global 10000-neuron sparse pool shared ACROSS all 5 bridges (each bridge runs its
  16 concepts on the same shared pool with 80 unique K-of-N patterns)
- Single substrate to train on; cross-bridge composition is then "in-substrate" not
  "across-substrate"
- REJECTED: defeats the purpose of distinct bridges (the user's 5-bridge motivation is
  modular organization); cannot test the cross-substrate hypothesis directly; would need
  to train as one bridge anyway

**Approach C — Dedicated → shared LEARNED projection (online STDP)**:
- Same as A but the dedicated → shared mapping is learned via online STDP during training
  (not a one-time prior)
- REJECTED for first probe: adds STDP convergence risk to an already-multi-component
  arc; A's one-time prior is the cheaper first probe (mirrors G.20 sparse's proven
  pattern of one-time `apply_sparse_topographic_prior`); C can be tested in a follow-up
  if A PASSes but reveals stability issues

**Recommended**: Approach A first. Cheapest disciplined first probe. Directly tests the
hybrid hypothesis. Reuses the validated G.20 sparse builder primitives byte-unchanged
(`build_sparse_pool_bridge` decomposed into its region-/pathway-construction primitives;
`generate_sparse_patterns`; `apply_sparse_topographic_prior`).

## Architecture (Approach A, frozen pre-launch)

Per bridge (one of 5: A_nouns / B_verbs / C_adj / D_spatial / E_functional):

```
DEDICATED SUBSTRATE (per pillar n=96 / OPTION 3 / pillar n=105):
  - 16 concept pools × 200 neurons each (3200 neurons; concept-pool architecture
    from build_biological_brain_regions byte-unchanged)
  - 16 concept FS pools × 24 neurons each (384 neurons; same builder)
  - 4 motor pools × 200 neurons (800 neurons; Tier 1 canon)
  - 4 motor FS pools × 24 neurons (96 neurons; same builder)
  - language_input (2048 neurons) — drives BOTH dedicated AND shared substrate
  - language_output (2048 neurons) — read out

SHARED SPARSE SUBSTRATE (per G.20 sparse pillar n=95):
  - shared_concept_pool (2000 neurons; weak dynamics 0.05/0.3/0.8)
  - shared_FS (300 neurons; WTA over the shared pool)
  - per-concept K=100 random sparse patterns (generate_sparse_patterns byte-unchanged)
  - language_input → shared_concept_pool plastic (one-time topographic prior at
    pillar n=95 strength: factor 10.0 / off-target 0.1; reuses
    apply_sparse_topographic_prior byte-unchanged)

CROSS-BRIDGE COMPOSITION:
  - At probe time, capture per-(bridge, concept) activity FROM the shared_concept_pool
    ONLY (the bio dedicated pools are NOT in the cross-bridge probe activity vector)
  - Each bridge contributes 16 concepts × 2000-neuron vector → 32000 features per bridge
  - Union over 5 bridges: 80 concepts × 2000 features per bridge (uniform substrate)
  - Cross-bridge probe primitive: REUSED byte-unchanged from
    cross_bridge_mode_unification_probe.py (pillar n=95 primitive); only changes are
    the cache loader (Direction 5 hybrid caches) and the verdict module (Direction 5
    frozen 0.80 bar)
```

Total per-bridge neuron count: 3200 (concept pools) + 384 (concept FS) + 800 (motor) +
96 (motor FS) + 2048 (lang_input) + 2048 (lang_output) + 2000 (shared_concept_pool) +
300 (shared_FS) = 10876 neurons (vs ~8528 for Direction 4 bridge). Sparse pool adds
~28% to bridge size; per-bridge train wall is ~17-20 min (matching v14/v16 + small
overhead for the additional sparse pathway).

5-bridge ensemble: 54380 neurons total per seed.

## Pre-registered test + bar

**Test**: cross-bridge parallel-matching mode-unification at load ladder {L=2, L=3, L=5}
on the union of 5 bridges' shared_concept_pool activity (80 cross-bridge concepts).
Each composite samples K items uniformly from the 80-concept union; parallel-matching
decodes per-slot identification via batched phase similarity on the FHRR-bound
positional codes (matching pillar n=95 byte-unchanged).

**Bar UNCHANGED** at 0.80 multi-seed (same as pillars n=93+ and Directions Q, 3, 4):
- `DIRECTION_5_PASS`: multi-seed-mean OB AND OI both clear 0.80 at every L in {2, 3, 5}.
  Hybrid hypothesis validated — biology-faithful dedicated pools + sparse cross-bridge
  substrate UNIFIED architecture works. Pillar n=106 candidate. Conversational substrate
  extended to 80 biology-faithful cross-bridge concepts on a unified architecture.
- `DIRECTION_5_PARTIAL`: some cells above bar but not all. Precise per-load breakdown;
  biology-translatable comparison to G.20 sparse pillar n=95 (does the hybrid reach
  match sparse-only, or fall between dedicated-only D4 NEGATIVE and sparse-only n=95
  full PASS?).
- `DIRECTION_5_NEGATIVE`: NO load-cell on EITHER readout clears the bar. Hybrid
  composition fails just like dedicated-only D4 did — the sparse readout is not
  rescuing the substrate; bottleneck is upstream (e.g., lang_input → shared_concept_pool
  pathway not picking up enough discriminative pattern despite the topographic prior).
- `DIRECTION_5_VOID_MALFORMED`: instrument-validity failure; not propagated as a pillar.

The 4 frozen thresholds (set once at `direction_5_verdict.py` module load; never
runtime-tuneable):
- `_DIRECTION_5_OB_MIN = 0.80`
- `_DIRECTION_5_OI_MIN = 0.80`
- `_DIRECTION_5_LOADS = (2, 3, 5)`
- `_DIRECTION_5_MIN_SEEDS = 3`

## Cost estimate

- Approach A: ~5-6 hr GPU for 5 bridges × 3 seeds + cross-bridge probe ~10 min CPU =
  ~5-6 hr total (matches Direction 4 cost; slightly higher per-bridge train due to
  added sparse pathway init)

## Files to create (this subagent's scaffolding output)

CPU-only scaffolding (Tasks 0-3, this subagent):
- `tests/test_direction_5_grounding.py` (Task 0; grounding pin)
- `research/findings/raw/direction_5_vocab_spec.py` (Task 1; reuses Direction 4
  vocab spec data structure with REIMPORT; 5 × V=16 = 80 cross-bridge concepts)
- `research/findings/raw/direction_5_bridge_builder.py` (Task 2; 5 per-bridge
  constructor wrappers; each calls `build_biological_brain_regions` byte-unchanged
  for dedicated pools + adds `shared_concept_pool` + `shared_FS` regions + the
  `lang_input → shared_concept_pool` plastic pathway as ADDITIONAL regions/pathways
  in the same `cfg.brain_regions` / `cfg.region_pathways` list)
- `research/findings/raw/direction_5_verdict.py` + `tests/test_direction_5_verdict.py`
  (Task 3; frozen thresholds + ≥12-case adversarial matrix; mirrors D4)

Controller-only (Tasks 4-6, NOT this subagent):
- `research/findings/raw/direction_5_cross_bridge_probe.py` (Task 4; CPU-only; reuses
  pillar n=95 + Direction 4 probe primitives byte-unchanged; only changes the cache
  path + hybrid-substrate documentation)
- `research/findings/raw/direction_5_5bridge_runner.py` (Task 5; GPU-bound; reuses
  Direction 4 runner pattern; only changes the bridge constructor calls + activity
  capture to read FROM the shared_concept_pool region per bridge)
- `research/findings/raw/direction_5_decisive.json` (Task 6; controller writes after
  Task 5 completes)

## Pre-staged post-Direction-5 chain

- DIRECTION_5_PASS: write findings doc; dispatch adversarial reviewer subagent; if
  CLEAR record pillar n=106; update AUTONOMOUS_STATE + capability_status.json. The
  hybrid architecture is validated; conversational vocabulary substrate extended to
  80 biology-faithful + sparse-distributed cross-bridge concepts on a unified
  architecture (FIRST such validated architecture in the project).
- DIRECTION_5_PARTIAL: precise per-load breakdown; biology-translatable comparison
  to G.20 sparse n=95 (which mechanism is the binding constraint: dedicated-pool
  geometry, sparse-pool geometry, or the dedicated → shared coupling?).
- DIRECTION_5_NEGATIVE: hybrid composition fails; sparse readout doesn't rescue
  cross-bridge; falsifies the dual-substrate hypothesis at this scale. Pivot to:
  (a) Approach C learned dedicated → shared projection; or (b) revisit whether
  the dedicated pools' activity contains the needed cross-bridge information at
  all (instrumented activity capture from dedicated pools side-by-side with sparse
  pool capture would reveal this).
- DIRECTION_5_VOID_MALFORMED: instrument-validity failure; diagnose recorded JSON
  shape; do not propagate as a pillar.

## Discipline (binding throughout)

- Bar UNCHANGED throughout (0.80 multi-seed; same frozen value used in pillars n=93+
  and Directions Q, 3, 4). Set ONCE in `direction_5_verdict.py` at Task 3; never
  tuned by results.
- No protected / frozen / moat modification. `build_biological_brain_regions` remains
  byte-unchanged. The shared sparse pool + WTA + lang_input → shared pathway are added
  as ADDITIONAL regions / pathways to the `cfg.brain_regions` / `cfg.region_pathways`
  list AFTER the protected builder returns — the wrapper is the only net-new code.
- The G.20 sparse builder's per-region / per-pathway construction primitives are
  reused byte-unchanged (extracted into the wrapper, NOT modified in place at
  `concept_pool_sparse_distributed.py`).
- The cross-bridge probe primitive at pillar n=95 + Direction 4 (Task 4
  `cross_bridge_mode_unification_probe.py`, `derive_global_grounded_symbols`,
  `batched_phase_similarity`) is reused byte-unchanged.
- No autograd.
- GPU/CuPy for Task 5 training only (controller-only). NumPy for Task 4 cross-bridge
  probe (CPU-only per pillar n=95 + Direction 4 pattern).
- Honest propagation EVERY outcome both remotes.
- Pre-launch grep confirmed (this subagent invocation): no prior hybrid sparse-on-
  bio_brain_regions work; G.20 sparse 5-bridge is on DIFFERENT substrate (sparse-only);
  bio_brain_regions cross-bridge is on DIFFERENT substrate (dedicated-only, D4 NEGATIVE).
- Reviewer-style scrutiny applied at the time of result (Task 6), not deferred.
- The hybrid wrapper MUST NOT import a cupy module at file load time. Imports are
  deferred to inside the construction functions so the module remains CPU-light.

## Continuation pointer

When this subagent ships Tasks 0-3 scaffolding:
1. Pre-launch grep result preserved in this design doc + implementation plan
2. Task 4 cross-bridge probe scaffold may follow in a Task 4 subagent (CPU-only; mirrors
   `direction_4_cross_bridge_probe.py` byte-pattern with hybrid-substrate cache
   adjustments)
3. Task 5 GPU runner is controller-only (the runner module's CONSTRUCTION can be
   scaffolded CPU-only by a follow-up subagent, but the decisive multi-seed
   training launch is the controller's responsibility)
4. Task 6 decisive cross-bridge probe + verdict emission is controller-only after
   Task 5 completes

The Direction Q + Direction 3 + Direction 4 infrastructure (frozen verdict pattern,
grounding pin pattern, multi-seed runner template, cross-bridge probe primitive) plus
the G.20 sparse builder primitives (`build_sparse_pool_bridge`,
`generate_sparse_patterns`, `apply_sparse_topographic_prior`) are reusable templates
that Direction 5 follows.
