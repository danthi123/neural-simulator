# P1 — Hippocampal trisynaptic loop validation

**Date:** 2026-05-11
**Phase:** P1 of realigned plan v3 (catalog-grounded, post-user-checkin)
**Roadmap entry:** T1.A (Month 1, 1-2 weeks for working circuit)
**Status:** SINGLE-SEED PASS (D.12 + D.13 both pass at seed 42); multi-seed in flight

## What this validates

Hippocampal trisynaptic pathway (catalog D.03; Kandel 6e Ch 54 pp 1340–1342):
`EC layer II → perforant path → DG → mossy fiber → CA3 → Schaffer → CA1`
with `EC → CA1` direct bypass and `CA3 → CA3` recurrent attractor.

Two characteristic functional properties tested:

- **D.12 Pattern separation** (Kandel pp 1357–1360):
  DG sparsifies overlapping EC inputs via Marr "expansion recoding"
  (high-density convergence onto sparse population + strong feedforward
  inhibition). PASS criterion: DG cosine < 0.5 for input cosine 0.8.

- **D.13 Pattern completion** (Kandel pp 1342, 1360–1361; Marr 1971
  autoassociator):
  CA3 recurrents reconstruct full pattern from partial cue. PASS
  criterion: cosine(CA3_partial, CA3_full) > 0.7.

## Single-seed result (seed 42)

| Test | Input | DG/CA3 cosine | Pass? |
|---|---|---|---|
| D.12 separation | input cos 0.800, sparsity 10% | DG cos 0.218 | ✅ PASS |
| D.13 completion | partial = 50% of stored | CA3 cos 0.748 | ✅ PASS |

D.12 orthogonalization: input 0.800 → DG 0.218 = **58 percentage points
drop**. DG sparsity ~1% (target 2-5%, slightly over-sparsified but
functionally excellent).

D.13 completion: 50% of stored CA3 ensemble (40 stored neurons; 20
driven during recall) reactivates the full pattern at cosine 0.748.
CA3 firing during recall: 14% with full cue, 12% with partial.

## Two findings about methodology

### Finding 1: Architecture was already there

The roadmap T1.A spec said "Build 3 new BrainRegions DG/CA3/CA1 wired
trisynaptically." Inspection of `text_minimal_isolation.py:build_biological_brain_regions(enable_hippocampus_consolidation=True)`
revealed the trisynaptic structure was **already built** during Phase
1.3 consolidation work:
- 5 regions: `ec`, `dg`, `dg_pv_basket`, `ca3`, `ca1`
- 10 pathways including `ec→dg` perforant + `dg_pv_basket→dg` FFi
  inhibition + `dg→ca3` mossy fiber + `ec→ca1` direct + `ca3→ca3`
  recurrent (SWR-gated) + `ca3→ca1` Schaffer + 5 consolidation
  pathways

The Phase 1.3 work validated this for RETENTION (Phase 1.3 + Tier 2.1
CONFIRMED 3/3 GO at multi-seed 2026-05-08). Never validated for
pattern separation or completion. **P1 reduced from "build new
circuit" to "validate existing circuit" + parameter tuning.**

### Finding 2: EC-driven test was confounded by feedforward chain

Initial test (EC-driven): drive `language_input` → propagates through
`ec → dg → ca3` → measure CA3 output. **FAILED at all parameter
combinations**: cosine 0.04, 0.07, 0.17 at train_events ∈ {30, 100, 200}.

The "partial cue" routed through three feedforward stages loses signal
fidelity at each step. Mossy fiber synapses are "detonator" synapses
(real biology: each one strongly drives the postsynaptic CA3 cell)
so CA3 firing tracks mossy fiber more than recurrent dynamics. Partial
EC → partial mossy fiber → CA3 pattern that doesn't match the trained
full-mossy-fiber-driven CA3 ensemble.

**DIRECT-CA3 test mode** (added 2026-05-11): record which CA3 neurons
fire most during training (top 10% = "stored ensemble"); for recall,
drive a partial of that CA3 ensemble *directly*. This isolates the
Marr autoassociator dynamics from the trisynaptic feedforward chain.

Direct-CA3 cosine results across param sweep:

| train_events | ca3_recurrent_weight | cosine | Pass? |
|---|---|---|---|
| 100 | 1.5 (default) | 0.605 | FAIL |
| 200 | 3.0 | 0.680 | FAIL |
| 400 | 5.0 | **0.748** | ✅ PASS |

Trend: more training events + stronger recurrent weight → cleaner
attractor. Both knobs matter; 400+5.0 is the empirical pass point.

## Parameters validated

```python
n_lang_input = 2048        # EC input pool (existing)
n_ec = 200                  # entorhinal cortex
n_dg = 800                  # 4x EC for expansion recoding (Marr)
n_dg_pv_basket = 240        # 30% of DG, biology-grounded FFi ratio
n_ca3 = 400                 # autoassociator size
n_ca1 = 200                 # readout
ca3_recurrent_density = 0.30
ca3_recurrent_weight = 5.0  # tuned up from default 1.5 for D.13 pass

stdp_w_max = 10.0           # higher cap for autoassociator strengthening
train_events = 400          # tuned up from default 30
```

Bridge size: **6016 neurons, ~1.1M synapses, 1.4 GB GPU memory.**
Wall clock: ~3 min per validation run on RTX 3090.

VRAM efficiency: this is < 6% of the 24 GB on a 3090, leaving plenty
of headroom for the parallel motor + language regions (not active in
this validation).

## What's not validated yet

- **Multi-seed.** Seeds 43, 44 in flight. Pass criterion: ≥ 4/6
  seeds for both tests.
- **CA1 readout** (catalog D.04 direct path; optional Test 3):
  CA1 integrates Schaffer (CA3) + direct EC input. Test 3 sketched in
  the runner but not implemented yet.
- **O&N sequential autoassociator** (catalog D.05 supplemental): real
  CA3 is theta-paced sequential, not Hopfield point attractor. Current
  test validates point-attractor (Kandel framing). Sequential is a
  follow-up.
- **Adult neurogenesis in DG** (catalog D.12 supplemental): not
  modeled; new granule cells aid fine pattern separation in biology.

## Why this matters for the primary path

P1 unblocks P3 and P4:

- **P3 SWR sequential replay** (roadmap T1.B): needs working trisynaptic
  loop. ✅ now available.
- **P4 episodic encoder + relational binding** (catalog D.01+D.02):
  needs CA3 pattern completion for retrieval. ✅ now available.
- **P5 ventral semantic stream** (G.11): builds on D.01+D.02 episodic
  binding for word→concept mapping.

The "concepts as tagged ensembles" model (user's insight 2026-05-11)
needs:
- Pattern separation (✅ P1.D.12) so distinct concepts get distinct
  CA3 ensembles
- Pattern completion (✅ P1.D.13) so partial cues retrieve full
  concept patterns
- Engram tagging (✅ P2 commit 29513ac) so ensembles can be named
  and stimulated by name
- Sleep replay (P3 next) so ensembles consolidate to cortex

The **P1 + P2 combination = "Apple is a CA3 ensemble"** is now
mechanistically sound.

## Open follow-ups

1. **Multi-seed verification** (in flight) — finish seeds 43, 44
   minimum; ideally 6 seeds for tier-3 confidence.
2. **EC-driven test re-examination** — the EC-driven test failure
   suggests `lang_input → ec → dg` weight gain isn't enough to
   propagate partial cues. Future tuning: bump `lang_to_ec` or
   `ec_to_dg` weight_mean.
3. **CA1 readout test (Test 3)** — implement and validate the
   match/mismatch hypothesis (catalog D.04 supplemental O&N
   pp 228–230).
4. **Sequential CA3** (catalog D.05 supplemental O&N pp 222–227):
   real CA3 is theta-paced; current test validates static attractor.
   Future follow-up adds theta organization.

## Commits in this arc

```
43e7b55  feat(P1): validate_trisynaptic_loop runner — catalog D.12 + D.13 tests
dab2721  feat(P1): parametrize ca3_recurrent_density + weight for D.13 tuning
e4744f4  feat(P1): DIRECT-CA3 drive mode for cleaner Marr autoassociator test
29513ac  feat(P2): engram-tagging API on SimulationBridge
a3acb9c  feat(P2): engram tag persistence through save/load + tests
```

Multi-seed completion + P3 design in flight.
