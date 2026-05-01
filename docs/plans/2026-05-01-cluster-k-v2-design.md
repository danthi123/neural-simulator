# Cluster K v2: Functional visual cortex with action-driving IT → cortex

**Status:** Design (2026-05-01)
**Predecessor:** v1 scaffold (commits 54e63ab, d88660b, 3e5efb4) — regions
+ pathways + image-rendering hook + integration test (V1_simple fires).
**Motivation:** Tier 0 honest test (commit f931b1c, a68e202) showed
the perception arc fails at 16×16 (15.47-35.42 vs 2.00 with heuristic).
Cluster K v2 makes the visual cortex actually drive action selection,
replacing hand-coded beacon/landmark sensors.

## Goal

Provide a learned, biology-grounded perception substrate that scales
gracefully to 16×16+ gridworlds. The agent should navigate to a goal
visible in the rendered image without any hand-coded direction sensors.

**Success metric:** sum ≤ 8 on cheat-5 multi-goal det at 16×16 with
ONLY visual cortex perception (no beacon, no landmark, no learned-
perception, no place/goal cells, no heuristic). Sum ≤ 4 would match
the 8×8 perception arc baseline.

## Architecture

```
                    image (32x32 ON/OFF)
                            │
                       ┌────▼─────┐
                       │  retina  │  2*32*32 = 2048 neurons (ext drive)
                       └────┬─────┘
                            │ Gabor-init weights (apply_v1_gabor_weights)
                       ┌────▼─────────────┐
                       │ cortex_v1_simple │  8 ori × 2 freq × 8x8 = 1024
                       └────┬─────────────┘
                            │ phase-pooled (fixed weights)
                       ┌────▼──────────────┐
                       │ cortex_v1_complex │  8 × 8x8 = 512
                       └────┬──────────────┘
                            │ plastic, gated "visual_cortex_v2"
                       ┌────▼─────┐
                       │ cortex_v2│  256 plastic recurrent
                       └────┬─────┘
                            │ plastic, gated "visual_cortex_it"
                       ┌────▼─────┐
                       │ cortex_it│  64 plastic recurrent
                       └────┬─────┘
                            │ plastic, gated "visual_cortex_action" ★ NEW IN v2
                       ┌────▼──────────────┐
                       │ cortex_{N,E,S,W}  │  drives motor selection
                       └───────────────────┘
```

The new pathway IT → cortex_{N,E,S,W} replaces the perception arc's
hand-coded beacon/landmark/learned-perception → cortex_X drives.

## Components to build

### 1. `bridge.set_pathway_weights(pathway_name, pre_idx, post_idx, weights)`

**Where:** `sim/bridge.py`

**Signature:**
```python
def set_pathway_weights(
    self,
    pathway_name: str,
    pre_indices: np.ndarray,   # (N,) int64 global neuron indices
    post_indices: np.ndarray,  # (N,) int64 global neuron indices
    weights: np.ndarray,       # (N,) float32 weights
    add_missing: bool = False, # if True, add new edges; else only update existing
) -> int:                       # returns count of updated edges
    """Overwrite weights for specific (pre, post) edges in cp_connections.

    Used by post-build pathway initialization (e.g. Gabor pre-init for
    V1 simple cells) and by future learned-weight injection (e.g.
    loading checkpointed pathways). The CSR is row-sorted; this method
    converts to LIL format for efficient single-edge updates, applies
    weights, and converts back.

    Raises ValueError if any (pre, post) pair is not found in the
    existing CSR (unless add_missing=True).
    """
```

**Implementation strategy:**
- Convert cp_connections (CSR) to dict form: `{(int, int): weight_idx}`
  by iterating data + indices + indptr. O(nnz) one-time cost.
- For each (pre, post, w) in inputs: look up index in dict and update
  cp_connections.data[i].
- If add_missing: rebuild CSR with new edges added.
- Invalidate plastic mask cache, COO cache.

**Tests:**
- Round-trip: set weights → read back via set_pathway_weights → equal.
- Update-only: existing edge updated, count returned matches input.
- Missing edge raises ValueError when add_missing=False.
- Add-missing mode adds new edges and increments nnz.

### 2. `apply_v1_gabor_weights(bridge, region_manager, ...)`

**Where:** `sim/visual_cortex.py` (existing module)

**Signature:**
```python
def apply_v1_gabor_weights(
    bridge,
    region_manager,
    n_orientations: int = 8,
    n_frequencies: int = 2,
    n_positions_per_dim: int = 8,
    retina_size: int = 32,
    receptive_field_radius: int = 4,
    weight_scale: float = 1.0,  # multiplier on Gabor-computed weights
) -> int:                       # returns count of synapses updated
    """Overwrite retina → cortex_v1_simple weights with Gabor receptive
    fields. Replaces the random init weights from build_v1_simple_weights.

    Calls bridge.set_pathway_weights() under the hood. Must be called
    after bridge._initialize_simulation_data() — the CSR must exist.
    """
```

**Implementation:**
- Get retina + V1_simple index ranges from region_manager.
- Call `build_v1_simple_weights()` to get (pre, post, weight) triples
  in retina-relative + V1-relative indices.
- Translate to global indices using region offsets.
- Call `bridge.set_pathway_weights("retina_to_v1_simple", ...)`.

**Tests:**
- After apply, weights are non-zero on Gabor-shaped regions, near-zero
  outside.
- V1_simple firing rates differentiate by orientation when retina is
  driven by an oriented bar stimulus.

### 3. IT → cortex_X pathway in build_bg_brain_regions

**Where:** `research/runners/g11_bg_runner.py` line ~1408 (visual cortex block)

**Add:**
```python
if enable_visual_cortex:
    # ... existing 4 pathways ...

    # IT → cortex_X (action selection). Plastic so STDP+reward can
    # learn which IT features predict which actions. Gated until
    # critical-period close so the visual cortex matures before
    # driving motor output. Weight tuned to match heuristic drive
    # magnitude (HEURISTIC_DRIVE_PA = 800 in single-pool mode).
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="cortex_it", to_region=f"cortex_{action}",
            density=visual_it_to_cortex_density,    # default 0.3
            weight_mean=visual_it_to_cortex_weight, # default 5.0
            weight_jitter=0.5,
            plastic=True,
            plasticity_gate="visual_cortex_action",
        ))
```

**Two new kwargs:**
- `visual_it_to_cortex_density: float = 0.3`
- `visual_it_to_cortex_weight: float = 5.0`

### 4. Critical-period curriculum for visual cortex

**Where:** `research/runners/g11_bg_runner.py` env-loop section

**New CLI flag:** `--visual-cortex-action-warmup-steps INT` (default: 600)

When `enable_visual_cortex` is on:
- Steps 0 to N_warmup: gates `visual_cortex_v1`, `visual_cortex_v2`,
  `visual_cortex_it` are open at 1.0 (V1/V2/IT learn from images);
  gate `visual_cortex_action` is at 0.0 (no IT → cortex drive).
- Steps N_warmup onwards: `visual_cortex_action` opens to 1.0; the
  visual stream drives action selection via IT → cortex_X.

This mirrors real visual development: V1/V2/IT mature first (sensory
critical period), then visuomotor wiring matures (post-critical-period
plasticity). Mechanism uses the existing
`bridge.set_plasticity_gate(name, value)` API.

**Important:** the IT → cortex_X pathway should also have its weights
gated, not just plasticity. With `weight_mean=5.0` initialized at
step 0, it would inject current immediately even before learning.
Solution: initialize at `weight_mean=0.0`, then post-warmup the
plasticity gate opens and STDP+reward grow the weights from zero.

### 5. Tests

`tests/test_visual_cortex_v2.py`:
- `test_set_pathway_weights_roundtrip` — weights set + read = equal
- `test_set_pathway_weights_missing_raises` — missing (pre,post) error
- `test_apply_v1_gabor_weights_changes_weights` — V1_simple weights
  differ from random init after apply
- `test_v1_simple_orientation_tuning_after_gabor_init` — drive retina
  with vertical bar, V1_simple cells with θ=0 fire more than θ=π/2
- `test_it_to_cortex_pathway_gated_at_init` — visual_cortex_action gate
  starts at 0.0
- `test_it_to_cortex_pathway_thaws_after_warmup` — gate opens to 1.0
  post-warmup
- `test_visual_cortex_v2_smoke_8x8` — 100-step run with v2 active
  doesn't crash; agent makes some progress

## Implementation phases

### Phase 1 (1 day): bridge.set_pathway_weights + tests
- Add CSR-edit helper to bridge
- Roundtrip + missing-edge tests
- Verify cache invalidation

### Phase 2 (0.5 day): apply_v1_gabor_weights + tests
- Add helper to sim/visual_cortex.py
- Orientation-tuning test (drive retina with bar stimulus, V1 selectivity)

### Phase 3 (0.5 day): IT → cortex_X pathway + curriculum gate
- Add pathway in build_bg_brain_regions
- Add CLI flag + warmup step logic in env loop
- Test: gate transitions from 0 to 1 at warmup boundary

### Phase 4 (1 day): integration eval
- 16×16 perception-only stress (no beacon, no landmark, no place cells,
  ONLY visual cortex)
- 3 seeds; compare to:
  - Heuristic + G v2.5: 2.00 ± 0.00 (upper bound)
  - Tier 0 perception arc: 15.47 ± 7.06 (lower bound to beat)
  - 8×8 perception arc baseline: 4.08 ± 0.49 (target)

### Phase 5 (variable): hyperparam tuning if needed
- weight_mean for IT → cortex_X (5, 10, 15, 25)
- Warmup steps (300, 600, 900)
- IT pool size (64 → 128 → 256)

## Risk register

| Risk | Mitigation |
|---|---|
| V1 random init produces no orientation tuning, V2/IT learn nothing useful | Gabor pre-init via apply_v1_gabor_weights |
| IT firing too sparse to drive cortex_X | Increase IT pool size or weight_mean |
| Curriculum freeze breaks like in Tier 0 retest | No freeze — only weight buildup from 0 via STDP+reward |
| Cluster K v2 + perception arc compose poorly | Test K v2 standalone first (no beacon/landmark) |
| Image rendering at every step is slow | Profile; if >10ms, batch rendering or skip during stim sub-steps |
| Bridge CSR invalidation breaks plasticity | Rebuild plastic mask + gain map after weights set |

## Naming conventions

All new neuron groups follow Kandel 6e ventral-stream naming:
- `retina` (not `lgn` since we skip LGN for v2)
- `cortex_v1_simple`, `cortex_v1_complex` (Hubel-Wiesel 1962)
- `cortex_v2` (Felleman & Van Essen 1991)
- `cortex_it` (inferotemporal — Tanaka 1996)

Plasticity gates use the same prefix pattern as existing ones:
- `visual_cortex_v1` — retina → V1_simple
- `visual_cortex_v2` — V1_complex → V2
- `visual_cortex_it` — V2 → IT
- `visual_cortex_action` — IT → cortex_X (NEW in v2)

## What v2 does NOT include

Deferred to v3 / future:
- LGN (retina → LGN → V1 instead of retina → V1)
- Color (we have grayscale ON/OFF only)
- Magnocellular vs parvocellular streams
- Top-down attention (FEF / pulvinar feedback)
- Object segmentation tasks (vs current grid navigation)
- Multi-scale processing
- Saccadic eye movements
