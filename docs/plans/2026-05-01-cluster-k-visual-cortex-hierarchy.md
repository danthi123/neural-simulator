---
type: plan
status: live
date: 2026-05-01
---

# Cluster K v1 — Visual Cortex Hierarchy (V1→V2→V4→IT)

**Date:** 2026-05-01
**Status:** DESIGN
**Predecessor:** flat 7×7 (dx, dy) goal-relative sensor grid in current `sensory` region

## Goal

Replace (or extend) the current task-specific sensor encoding with a biologically-grounded visual processing hierarchy modeled after the primate ventral stream. This is the foundational step for **multimodal sensory integration** per the architecture roadmap.

## Biology source

Primary:
- **Hubel & Wiesel 1962** *J Physiol* "Receptive fields, binocular interaction in cat's striate cortex" — V1 simple cells with Gabor-like oriented receptive fields.
- **Felleman & Van Essen 1991** *Cereb Cortex* "Distributed hierarchical processing in the primate cerebral cortex" — connectivity matrix V1→V2→V4→IT.
- **Tanaka 1996** *Annu Rev Neurosci* "Inferotemporal cortex and object vision" — IT object identity tuning.

Secondary:
- **Kandel 6e Ch 22** (Visual processing).
- **DiCarlo & Cox 2007** — untangling object manifolds via the ventral stream.
- **Riesenhuber & Poggio 1999** — HMAX hierarchical model (alternating S/C layers).

## Architectural mapping

Real ventral stream | This sim
---|---
Retina (rod/cone, ganglion cells, ON/OFF) | `retina_input` — pixel grid → ON/OFF channels
LGN (magno/parvocellular) | (skipped in v1; folded into V1 input)
V1 simple cells (Gabor) | `cortex_v1_simple` — oriented edge detectors, fixed Gabor weights at init
V1 complex cells | `cortex_v1_complex` — orientation pooled, position-invariant
V2 (illusory contours, junctions) | `cortex_v2` — combines orientations; plastic
V4 (color + form) | `cortex_v4` — higher-order shape features; plastic
IT (object identity) | `cortex_it` — sparse identity readout; plastic

For our 8×8 gridworld: render a 32×32 pixel image (4× upsample of grid) with the agent as a bright dot, goal as another color. Feed through retina → V1 → ... → IT, then IT readout drives `cortex_X` motor planning.

## Implementation

### v1 minimal viable (this design)

Keep it tractable: 4-layer stack (V1 simple, V1 complex, V2, IT). Skip V4 for now (the gridworld doesn't need color discrimination).

### Region definitions

```python
# Retina: ON/OFF channels at each pixel
retina = BrainRegion(
    name="retina",
    n_neurons=32 * 32 * 2,  # 2048: 32x32 grid, 2 channels (ON, OFF)
    exc_fraction=1.0,
    internal_density=0.0,
    plastic_internal=False,
    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
)

# V1 simple cells: oriented Gabor receptive fields
# 8 orientations × 4 spatial frequencies × 16x16 spatial positions = 8192 cells
cortex_v1_simple = BrainRegion(
    name="cortex_v1_simple",
    n_neurons=8 * 4 * 16 * 16,  # 8192
    exc_fraction=0.85,
    internal_density=0.05,  # weak local recurrent
    plastic_internal=False,
    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
)

# V1 complex cells: pool over phase, retain orientation/position
cortex_v1_complex = BrainRegion(
    name="cortex_v1_complex",
    n_neurons=8 * 16 * 16,  # 2048: 8 orientations × 16x16 positions
    exc_fraction=0.85,
    internal_density=0.05,
    plastic_internal=False,
)

# V2: orientation+junction features
cortex_v2 = BrainRegion(
    name="cortex_v2",
    n_neurons=512,
    exc_fraction=0.85,
    internal_density=0.10,
    plastic_internal=True,  # learn higher-order combinations
)

# IT: sparse identity readout
cortex_it = BrainRegion(
    name="cortex_it",
    n_neurons=128,
    exc_fraction=0.85,
    internal_density=0.05,
    plastic_internal=True,
)
```

Total: 2048 + 8192 + 2048 + 512 + 128 = 12,928 neurons. About 8× current model size. Need to verify GPU memory; should fit.

### Pathways

- `retina → cortex_v1_simple`: dense, fixed Gabor weights at init (NOT plastic). Each V1 cell receives input from a small spatial patch with orientation-tuned weights.
- `cortex_v1_simple → cortex_v1_complex`: pooling over phase (sum 4 simple cells per complex cell). Fixed.
- `cortex_v1_complex → cortex_v2`: dense plastic, density 0.30
- `cortex_v2 → cortex_it`: dense plastic, density 0.50
- `cortex_it → cortex_X` (motor planning): plastic, density 0.30

### Image rendering

Each env step, render the gridworld state as a 32×32 image:
- Background: black
- Walls: gray
- Agent: bright green pixel (with orthogonal NESW arrow indicator?)
- Goal: bright yellow pixel
- (later: visual landmarks, beacons)

Rendering done in numpy on CPU, transferred to GPU as `cp_external_input_current[retina_indices]`.

### Gabor weight init

For each V1 simple cell with index `(orientation, freq, x, y)`:
```python
def gabor_kernel(sigma_x, sigma_y, theta, freq, phase=0):
    # Standard Gabor wavelet
    return lambda dx, dy: np.exp(-(dx*dx/sigma_x**2 + dy*dy/sigma_y**2)/2) * \
                          np.cos(2 * pi * freq * (dx*np.cos(theta) + dy*np.sin(theta)) + phase)
```

8 orientations: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°.
4 frequencies: 0.05, 0.1, 0.2, 0.4 cycles/pixel.

V1 simple cell at (orient, freq, x_center, y_center) receives from retina pixel (px, py) with weight `gabor_kernel(orient, freq, x_center - px, y_center - py)`.

This is a custom wiring path — not currently supported by `RegionPathway`. Either:
1. Add `weight_function` field to `RegionPathway` that takes pre/post coords
2. Build the wiring plan directly via `inject_explicit_wiring` (preferred for v1)

## Test plan

### Unit tests

- Gabor weight init: verify a horizontal-orient V1 cell responds maximally to a horizontal bar input
- V1 complex pooling: verify 4 simple cells with different phases pool to position-invariant complex cell
- Forward pass: feed a 32×32 image of an agent at center, observe firing cascade through V1→V2→IT

### Integration test 1: agent identity decoding

After 500 env steps with random agent positions, train a linear readout from cortex_it to (x, y) coordinates. Hypothesis: IT firing rates encode position. Validate: linear readout decodes position with R² > 0.5.

### Integration test 2: cheat-5 with visual input

Replace heuristic + sensory grid with full visual pipeline. Agent receives only image input.

Setup:
- `--enable-visual-cortex` (new flag)
- No `--heuristic-single-pool` (heuristic disabled by visual processing)
- Cheat-5 multi-goal det

Expected: agent learns from scratch using visual input. Likely much harder than current — initial performance probably random, learning curve steep.

This is a STRESS test, not a baseline-beating attempt. Visual learning is a separate research problem from cheat-5 navigation.

## Effort estimate

**v1 minimal:** ~1 week.
- 2 days: region defs, Gabor init, retina rendering
- 2 days: explicit wiring plan, V1→V2→IT pathway tuning
- 1 day: integration tests, smoke runs
- 1-2 days: documentation + findings

## Out of scope (defer to v2)

- V4 color/form integration (gridworld doesn't have color)
- Magnocellular/parvocellular split
- Top-down feedback (reentrant V2→V1)
- Saccadic eye movements / fovea
- Gain control / contrast normalization (a la Carandini-Heeger 1997)
- Auditory cortex / cochlea

## Files to create / touch

- `sim/visual_cortex.py` (NEW): Gabor utilities, retina rendering
- `research/runners/g11_bg_runner.py`: add `--enable-visual-cortex` flag, build_bg_brain_regions extension
- `tests/test_visual_cortex.py` (NEW): unit tests for Gabor init + V1 pooling
- `docs/plans/2026-05-01-cluster-k-visual-cortex-hierarchy.md`: this design

## Decision: defer to after Cluster G eval lands

Visual cortex is a substantial undertaking (~1 week). Before committing, want to see if Cluster G v1 (PFC-NMDA) shifts the operational best — that data point will inform whether per-region or per-mechanism additions are the right next direction. If G doesn't help, proceed with visual cortex as the foundation for multimodal. If G helps significantly, may revisit Cluster G v2 (per-region NMDA) before K.
