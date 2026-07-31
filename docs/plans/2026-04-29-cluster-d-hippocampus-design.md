---
type: plan
status: live
date: 2026-04-29
---

# Cluster D — Hippocampus Trisynaptic Loop + Replay Design

**Date:** 2026-04-29
**Goal:** Build the hippocampal substrate for sequence learning and memory consolidation, providing a third arc beyond closed-loop (A) and tonic DA (C) for cheat-5 closure.
**Why now:** Per the catalog roadmap (T1.A and T1.B), hippocampus is the project's biggest "partial" — current `place_cells` and `goal_cells` regions don't have any of the trisynaptic-loop machinery. Sequence learning is a likely missing-piece for cross-projection coordination — if the agent can replay cross-action sequences during quiet rest, STDP can consolidate them.

## Architecture

The biological hippocampus has the canonical Cajal "trisynaptic pathway" (also written "trisynaptic circuit" / "trisynaptic loop" in older literature):

```
EC (entorhinal cortex) → DG (dentate gyrus) → CA3 (recurrent autoassociator) → CA1 → output (subiculum/EC)
                            └─ EC → CA1 (direct cortical bypass)
                            └─ CA3 → CA3 (recurrent autoassociator core)
```

**Functional roles:**
- **DG** — pattern separation. High sparsity (~3% active), strong feedforward inhibition orthogonalizes inputs. Critical for distinguishing similar contexts.
- **CA3** — pattern completion. Dense recurrent collaterals form an autoassociator that reactivates full patterns from partial cues.
- **CA1** — readout. Integrates direct EC input + CA3 output, forms place-field-like representations.

**Sharp-wave ripples (SWRs):** intrinsic CA3 events (R3.12 catalog framing) — population bursts arising from recurrent excitation + adaptation thresholds. During NREM, SWRs replay waking-trajectory sequences time-compressed (10-20×). NREM is a passive gate, not a generator.

## Proposed implementation

### v1 (this design): minimal trisynaptic pathway, no SWR yet

Three new regions per the canonical loop:

| Region | n_neurons | exc_fraction | internal_density | Notes |
|---|---|---|---|---|
| ec | 80 | 0.8 | 0.05 | Entorhinal cortex stub; receives sensory + landmark; projects to DG, CA3, CA1 |
| dg | 200 | 0.95 | 0.0 | Dentate gyrus; high sparsity via FFi; orthogonalizes |
| dg_fs | 60 | 0.0 | 0.0 | DG fast-spiking interneurons providing strong feedforward inhibition |
| ca3 | 100 | 0.85 | 0.30 | CA3 with dense recurrent collaterals (autoassociator) |
| ca1 | 120 | 0.85 | 0.05 | CA1 readout |

Pathways (all opt-in via `--enable-cluster-d-hippocampus`):

| From → To | Density | Weight | Plastic |
|---|---|---|---|
| sensory → ec | 0.40 | 4.0 | True (gated `sensory_to_ec`) |
| landmark_sensors → ec | 0.40 | 4.0 | True |
| ec → dg | 0.40 | 6.0 | True (perforant path) |
| ec → dg_fs | 0.40 | 5.0 | False (FFi recruitment) |
| dg_fs → dg | 1.00 | 6.0 | False (strong FFi for sparsity) |
| ec → ca1 | 0.30 | 3.0 | True (direct cortical bypass) |
| dg → ca3 | 0.10 | 8.0 | True (mossy fibers — sparse but strong) |
| ca3 → ca3 | 0.30 | 1.5 | True (recurrent collateral autoassociator) |
| ca3 → ca1 | 0.30 | 4.0 | True (Schaffer collaterals) |
| ca1 → place_cells | 0.50 | 5.0 | False (readout into existing perception arc) |

Replaces the current `place_cells` self-organization from `landmark_sensors → place_cells` with a richer hippocampal substrate. CA1 → place_cells preserves backward compatibility with the existing flagship architecture.

### v2 (deferred): SWR generator + NREM gate

Per R3.12 framing: SWR generator lives in CA3 intrinsic dynamics. Specifically:
- Add an adaptation threshold to CA3's dynamics (already partially present via Izhikevich `b` parameter).
- Detect population bursts in `ca3` via running mean firing rate.
- Couple bursts to a slow oscillation phase variable (NREM Up-state troughs).
- During SWR events, drive CA3 with elevated `excitability_drive` for ~50ms.
- STDP consolidates downstream weights automatically.

v2 requires no new infrastructure beyond what already exists; just bridge wiring and a phase-variable.

### v3 (deferred): engram tagging API

Per catalog T1.C: `bridge.tag_active_ensemble(name, ...)` + `bridge.stimulate_tag(name, ...)`. ~50 LOC bridge addition. Validates pattern completion mechanism; aligns with optogenetic-tagging experimental paradigms.

## Validation

### Smoke
- `--enable-cluster-d-hippocampus` adds ~5 regions and ~10 pathways; total ~440 new neurons; runs cleanly.

### Cheat-5 multi-goal re-eval
- Compare 4 conditions:
  - flagship post-R-pass (no D)
  - flagship + Cluster A (already evaluated)
  - flagship + Cluster D (new)
  - flagship + A + C + D (full stack)
- Decision: same matrix as Cluster A — Δ mean ≤ −1.0 and Δ std ≤ baseline → GO; partial otherwise.

### Direct biology validation (Cluster D specific)

- **Pattern separation:** present 2 highly similar perception inputs to EC, verify DG firing patterns are decorrelated (overlap < 30%).
- **Pattern completion:** train CA3 on a cue-context pair; present partial cue; verify full output reactivates.
- **Place fields in CA1:** verify CA1 cells show position-correlated firing.

(These tests can be added after the cheat-5 eval if v1 shows GO signal.)

## Estimated effort

- v1 implementation + tests: 4-6 hours
- Cheat-5 eval: 12 sequential 1800-step runs ~120 min
- v2 (SWR generator): 4-8 hours, can be done after v1 cheat-5 eval if v1 looks promising
- v3 (engram tagging): 2-3 hours, separate

Total v1: ~5-6 hours from design to findings.

## Composition with other clusters

- **A + D:** thalamo-cortical feedback gives place_cells the post-synaptic activity needed for CA1 → place_cells STDP to consolidate.
- **C + D:** tonic DA modulates plasticity_rate; CA3's dense recurrent network has many plastic synapses that benefit from properly-modulated DA.
- **A + C + D:** the full "biology" stack. If A+C+B.3 doesn't close cheat-5 (current eval), this is the next composition to test.

## Implementation strategy

Given the substantial scope:
1. Land v1 trisynaptic-loop core (regions + pathways) — autonomous, doable inline.
2. Tests + smoke — autonomous.
3. Run cheat-5 eval — background.
4. Synthesize findings — autonomous.
5. Decide on v2 (SWR) based on cheat-5 result.
