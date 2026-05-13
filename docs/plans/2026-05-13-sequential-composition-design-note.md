# Sequential composition for concept pools — design note

**Date:** 2026-05-13
**Status:** Design only. Not implemented; queued for after v8 batch.

## The problem

v7 Phase 2 composition test:
- Sequential composition (drive word_a 50 steps then word_b 50 steps): **0/6 PASS**
- Co-firing composition (merged drive): **2/6 PASS**

Sequential composition needs pool_a to remain firing during word_b's
drive window so both targets are active. With weak dynamics
(0.05/0.3/0.8, chosen to fix v2c canon bias amplification), pools
have insufficient self-sustaining recurrent activity. Drive removal
collapses firing within ~25ms.

Trade-off:
- Canon dynamics (0.10/2.0/4.0): NMDA bistability holds firing 100-150ms,
  but at biological scale with 12 pools, off-target pools self-sustain
  with random bias (v2c 0/12 result).
- Weak dynamics: cleaner cross-category isolation (v7 6.7/12 mean),
  but no NMDA persistence → no sequential composition.

## Design options

### Option A — Per-region NMDA tau override (small change)

Add a `concept_pool_nmda_tau_ms` parameter to `build_biological_brain_regions`.
Default = cfg.nmda_tau (~150ms). Set per-region for concept pools to
~50-100ms — long enough for cross-word persistence, short enough to
prevent indefinite accumulation.

Pro: targeted intervention; doesn't change canon for other regions.
Con: NMDA tau is currently a global cfg parameter; would need bridge
modification to support per-region tau.
Estimate: 1-2 days.

### Option B — Explicit working-memory pool (Tier 2.3 pattern)

Use the existing `enable_dlpfc_verb` infrastructure. Wire concept
pools through a dedicated PFC-like region that maintains persistent
activity via NMDA bistability. lang_input → pool → dlpfc → lang_output.

Pro: reuses existing infrastructure (Tier 2.3).
Con: Tier 2.3 was stuck at 34-40%; adds another bottleneck pathway.
Composition would only work for the verb kind.
Estimate: 2-3 days.

### Option C — Inject sustaining current during composition

Phase 2 test could explicitly hold word_a's drive at reduced amplitude
during word_b's window, simulating attentional maintenance. Not a
true architectural fix; eval-side hack.

Pro: easy to test (no architecture change).
Con: doesn't generalize to autonomous composition; eval-only.
Estimate: hours.

### Option D — Add a separate "echo pool" per concept

Each pool has a small (~50 neuron) high-recurrent "echo" satellite
that gets driven by the main pool, has canon dynamics + strong NMDA,
and feeds back to the main pool. Echo pools self-sustain for ~150ms;
main pools stay weak for clean isolation.

Pro: best of both worlds (weak main pool + bistable echo).
Con: doubles pool count; doesn't map to known biology cleanly.
Estimate: 3-4 days.

## Recommendation

**Option A (per-region NMDA tau)** is the cleanest biology-faithful
fix. Cortex NMDA tau IS region-specific in biology (Wang 2002 PFC
~150ms; sensory ~50ms). Setting concept pools to 80-100ms gives a
real bistability window without amplifying random bias.

Implementation order:
1. Add `cp_nmda_tau` per-neuron array (instead of cfg scalar)
2. In RegionPathway / BrainRegion, accept `nmda_tau_ms_override`
3. Bridge initialization applies per-neuron tau from region overrides
4. Test: v9 = v7 + concept_pool_nmda_tau_ms=80
5. Compare Phase 2 sequential PASS (v7 0/6 → v9 ?)

## Decision criteria

- If v8 (concept_to_language_output fix only) gives A→W PASS rate
  >= 4/12, that validates the weight fix; sequential composition
  can be tackled in v9.
- If v8 also shows Phase 1 improvement, the weight fix may indirectly
  help bistability too (via stronger reciprocal feedback) — sequential
  might improve without architecture change.
- If v8 doesn't help, Option A is the next step.

## Notes

This design note is intentionally NOT a plan — it's an option
inventory for the next architectural decision after v7/v8 multi-seed
validation completes.
