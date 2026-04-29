# Cluster E — Topographic Maps + Connectivity Refinement Design

**Date:** 2026-04-29
**Goal:** Add 2D coordinate fields to `BrainRegion` and distance-dependent connection probability to `RegionPathway` generators, enabling columnar / retinotopic / somatotopic / motor-map organization.
**Why:** Per the catalog roadmap (T2.B), this is "the single highest-leverage gap flagged by the Part IV agent — every cortical perception result in Kandel rests on adjacent-neurons-encode-adjacent-stimulus." Currently our regions are unstructured (Watts-Strogatz / random); pathway connectivity is independent of any spatial layout.

## Current state

- `BrainRegion` has no spatial info beyond `n_neurons`.
- `RegionPathway.density` is a uniform Bernoulli probability across all pre-post pairs.
- `sim/connectivity.py` has Watts-Strogatz / spatial generators but they're for the global connectivity matrix; per-region pathway gen is uniform.

## Proposed v1 scope

### 1. Add coordinate field

```python
@dataclass
class BrainRegion:
    ...
    coordinate_dim: int = 0  # 0 = no coords (default); 1 = 1D; 2 = 2D
    coordinate_extent: Optional[Tuple[float, ...]] = None  # e.g., (1.0, 1.0) for 2D unit square
```

When set, the region's neurons are deterministically assigned coordinates uniformly in the extent. Coordinates are stored on the bridge as `cp_neuron_coords: cp.ndarray[float32, (n, k_dim)]` for use by connection generators.

### 2. Distance-dependent connection probability

Extend `RegionPathway`:

```python
@dataclass
class RegionPathway:
    ...
    distance_sigma: Optional[float] = None  # if set, p(i,j) ∝ exp(-||c_i-c_j||²/2σ²) * density
```

When both source and target have coordinates AND `distance_sigma` is set, connections are sampled with Gaussian-weighted probability. Otherwise, falls back to uniform density (current behavior).

### 3. Implementation

- `sim/regions.py`: extend dataclasses
- `sim/connectivity.py`: new helper `gauss_distance_density(coords_pre, coords_post, sigma) -> p_matrix`
- `sim/bridge.py:inject_explicit_wiring`: when pathway has distance_sigma, use the new helper to compute per-pair probabilities
- Tests: verify coordinate assignment is deterministic, verify distance-weighted connections fall off as expected

### 4. Initial application: cortex topography

Modify `g11_bg_runner.build_bg_brain_regions` to:
- cortex_X regions: 2D coordinates (each X gets a corner: N=(0,1), E=(1,1), S=(1,0), W=(0,0) of unit square)
- str_D1_X / str_D2_X: same coordinate scheme
- Add distance_sigma=0.3 to cortex_X → str_D1_X and similar, so cross-action density drops off naturally with distance.

This composes with the existing patch-matrix sparse cross-projection (R3.5).

## Validation

### Smoke
- 50-step run with new coordinate-tagged regions completes; connection counts decrease modestly compared to uniform density (because Gaussian-weighted is sparser at edges).

### Functional test
- Connect a 256-neuron sensory layer with 2D coordinates to a cortex layer; verify receptive-field tuning emerges from STDP on input patterns. (Out of scope for v1; future test.)

### Cheat-5 eval
- Compare baseline vs +Cluster E (cortex/striatum topography) on multi-goal task. Hypothesis: action channels with proper spatial structure may have cleaner cross-action separation, helping cheat-5.

## Effort

- v1 implementation: 3-4 hours (substantial — touches dataclass + connectivity + bridge wiring)
- Eval: 6 sequential runs ~60 min
- Total: ~5-6 hours

## Composition

- **E + A:** topographic cortex provides clean spatial substrate for thal→cortex feedback; should compose well.
- **E + B (FSI):** FSI inhibition is naturally local in real biology; topographic FSI→MSN with distance-decaying probability is more biologically faithful than the current symmetric cross-action wiring.
- **E + D:** hippocampus place fields emerge from spatial structure; topographic CA1 → place_cells is the proper mapping.

## Decision

Cluster E v1 is INFRASTRUCTURE that unlocks better biology in many places. Worth implementing if:
- Cluster A/C/D evals don't show GO signal
- AND we want a different angle than C v2 (compartmentalized DA)

Currently DEFERRED until results from current evals land. C v2 (compartmentalized DA) is the more direct cheat-5 closure attempt; E is broader infrastructure.
