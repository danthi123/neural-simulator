---
type: plan
status: live
date: 2026-04-29
---

# Cluster A — Closed BG Loop Design

**Date:** 2026-04-29
**Goal:** Close the BG cascade by adding (1) the cortex → STN hyperdirect pathway and (2) the thalamo-cortical feedback loop.
**Why now:** Cluster B remediation pass is done; closed-loop teaching signal is the most-cited missing-biology gap from the cheat-5 reframe (CLAUDE.md: "Real BG carves cross-projections via... closed thalamo-cortical loop (the missing teaching signal)").

## Current cascade (open loop)

```
sensory → cortex_X → str_D1_X → gpi_X → thal_X → motor_X
                  → str_D2_X → gpe_X → stn → gpi_X
                                     → gpe_arky_X → str_FS_Y (broadcast)
                  → str_FS_X → str_D{1,2}_Y (Y≠X cross-action WTA)
                  → str_patch_X → dopamine + gpi_X
gpi_X → dopamine (R3.10 disinhibition)
```

Note: thal_X has only ONE downstream target (motor_X). The cortex never "knows" what BG selected — BG output is consumed by the motor pool but no feedback to cortex.

## What's missing (catalog ref)

### 1. Hyperdirect pathway (cortex → STN)

- **Biology:** ~30% of cortical pyramidal axons project DIRECTLY to STN, bypassing striatum (Nambu 2002, "hyperdirect" pathway). Fast (~3-5 ms latency vs ~20 ms for indirect). STN's diffuse projection to all GPi suppresses ALL action channels briefly.
- **Function:** Provides a "global stop" / "pause-then-select" signal before BG action selection settles. Prevents premature commitment when multiple cortex pools drive simultaneously.
- **Catalog:** mentioned in Mink 1996 / Nambu 2002 framing; existing simulator comment at `stn -> all GPi` says "this is the 'hyperdirect'-like contribution" but the real hyperdirect is cortex → STN → GPi (we have only the second half).

### 2. Thalamo-cortical feedback (thal → cortex)

- **Biology:** Thalamic relay nuclei (VA/VL) send glutamatergic projections back to motor / premotor cortex. Closes cortex → BG → thalamus → cortex loop.
- **Function:**
  - Reinforces selected action: when BG releases thal_X, thal_X excites cortex_X back, sustaining the cortical pattern.
  - Provides the "teaching signal" missing from cross-projection learning: when a cross-action pathway (cortex_X → str_D1_Y for Y≠X) actually drives a useful selection, the resulting thal_Y → cortex_Y feedback fires the target cortex pool, providing post-synaptic activity for STDP to reinforce the cross link.
- **Catalog reframe-doc connection:** "Cross-projections need... a closed BG loop (thalamo-cortical feedback, hyperdirect pathway)... to behaviorally pay off."

## Proposed implementation

### Pathways added

| Pathway | Density | Weight | Plastic | Notes |
|---|---|---|---|---|
| cortex_X → stn | 0.10 | 3.0 | False | Sparse hyperdirect; per Nambu 2002 ~30% of cortex pyramids contact STN |
| thal_X → cortex_X | 0.50 | 5.0 | False | Same-action feedback; reinforces selection |
| thal_X → cortex_Y (Y≠X) | 0.0 | — | — | NOT added — thal_X is action-specific, not a global broadcast |

Both opt-in via `--enable-cluster-a-closed-loop`.

### Why static (plastic=False)?

- Hyperdirect: anatomical, not learned; widespread cortex → STN is genetically specified.
- Thal → cortex: same. The biological feedback is structurally fixed; what *changes* with learning is the upstream cortex → str / cortex → stn / etc weights. Adding plasticity here would mostly add noise.

### Why NO cross-action thal → cortex?

The closed loop is action-specific. If thal_X fed back to cortex_Y (Y≠X), it'd pollute the action-channel separation that the entire cascade is built on. Per biology, VA/VL nuclei have topographically organized projections to specific cortical areas (somatotopic; arm-thal → arm-motor-cortex, leg-thal → leg-motor-cortex).

### Implementation file changes

1. `research/runners/g11_bg_runner.py:build_bg_brain_regions`:
   - New `enable_cluster_a_closed_loop: bool = False` kwarg
   - When on:
     - For each action: cortex_X → stn (density 0.10, weight 3.0, plastic=False, plasticity_gate=None)
     - For each action: thal_X → cortex_X (density 0.50, weight 5.0, plastic=False)
2. `research/runners/g11_bg_runner.py:run_moving_goal_episode`:
   - New `enable_cluster_a_closed_loop: bool = False` kwarg
   - Pass to build_bg_brain_regions
3. `research/runners/g11_bg_runner.py:main`:
   - New `--enable-cluster-a-closed-loop` argparse flag
4. `tests/test_g11_bg_runner_flags.py`: 3 new tests
   - test_cluster_a_default_off
   - test_cluster_a_hyperdirect_pathways_built (4 cortex_X → stn paths)
   - test_cluster_a_thal_to_cortex_pathways_built (4 thal_X → cortex_X paths, no cross-action)
5. `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md` (this file)

## Validation criteria

### Smoke test

- Runner completes without error: `--enable-cluster-a-closed-loop` adds 8 pathways, region count unchanged.

### Cheat-5 multi-goal re-eval (n=3 seeds 42, 43, 44)

- Compare against current-code baselines from R-pass:
  - `--bg-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-fsis` (no Cluster A) → reference
  - + `--enable-cluster-a-closed-loop` → test
- **Expected outcome (hypothesis):** the closed loop should reduce variance and possibly improve mean. The "teaching signal" hypothesis: thal → cortex feedback creates post-synaptic activity in cortex_Y that lets STDP shape useful cross-action weights.
- Decision matrix:
  - Mean drop ≥ 1.0 AND std ≤ baseline std → **GO** signal; tier-3 (6-seed) validation next
  - Mean drop 0.0 to 1.0 → **PARTIAL** — possibly add Cluster C (DA-system completeness) before tier-3
  - Mean increase → **NO-GO** — closed loop doesn't help in current parameters; tune weights or revert

## Future extensions (deferred to Cluster A v2 if v1 doesn't deliver)

- Cortex → SNr direct (rare but documented in primates; ~5% of cortex axons)
- Thalamic reticular (TRN) gating: TRN inhibits thal_X to prevent reverberation; we don't have TRN in the cascade today
- Dynamic cortex → STN routing via NMDA-dependent gain (would need `cortex_to_stn` plasticity_gate)

## Estimated effort

1-2 hours: implementation + tests. Cheat-5 eval: 6 sequential 1800-step runs ~50 min. Total: 2-3 hours including findings + propagation.
