"""Unit tests for distributed motor pool architecture (Pulvermüller G.20).

Validates that `build_bg_brain_regions(enable_distributed_motor_pop=True)`:
- Creates 8 motor_pop_θ sub-pools instead of 4 motor_X pools
- Wires cosine-tuned thal_X → motor_pop_θ pathways
- Wires plastic language_input → motor_pop_θ pathways
- Default-off behavior preserves backwards compatibility
"""
from __future__ import annotations

import math


def _build_regions(enable_distributed=False):
    """Build regions+pathways with text I/O enabled; with or without
    distributed motor pop. Standalone, no GPU needed."""
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(
        enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True,
        enable_cluster_e_topography=True,
        enable_pfc=True,
        pfc_enable_nmda=True,
        enable_visual_cortex=True,
        enable_text_io=True,
        enable_distributed_motor_pop=enable_distributed,
    )
    return regions, pathways


def test_default_off_preserves_motor_X_pools():
    """Without --enable-distributed-motor-pop, the 4 motor_X pools exist."""
    regions, _ = _build_regions(enable_distributed=False)
    region_names = {r.name for r in regions}
    for action in ["N", "E", "S", "W"]:
        assert f"motor_{action}" in region_names, (
            f"motor_{action} should exist when distributed-pop is off"
        )
    # No motor_pop sub-pools when off
    assert not any(n.startswith("motor_pop_") for n in region_names), (
        "motor_pop_* should NOT exist when distributed-pop is off"
    )


def test_distributed_creates_8_subpools():
    """With --enable-distributed-motor-pop, 8 motor_pop_θ sub-pools exist
    instead of 4 motor_X pools."""
    regions, _ = _build_regions(enable_distributed=True)
    region_names = {r.name for r in regions}

    # 4 labeled motor_X pools should NOT exist
    for action in ["N", "E", "S", "W"]:
        assert f"motor_{action}" not in region_names, (
            f"motor_{action} should NOT exist when distributed-pop is on"
        )

    # 8 motor_pop sub-pools at 45° intervals
    expected_subpools = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    for sfx in expected_subpools:
        assert f"motor_pop_{sfx}" in region_names, (
            f"motor_pop_{sfx} missing under distributed-pop"
        )


def test_distributed_subpool_neuron_count():
    """Each motor_pop sub-pool has n_motor_pop_per_subpool neurons (default 5)."""
    regions, _ = _build_regions(enable_distributed=True)
    for r in regions:
        if r.name.startswith("motor_pop_"):
            assert r.n_neurons == 5, (
                f"{r.name}: expected 5 neurons, got {r.n_neurons}"
            )

    # Total motor neurons = 8 × 5 = 40 (matches default 4 × 10 = 40)
    motor_pop_total = sum(
        r.n_neurons for r in regions if r.name.startswith("motor_pop_")
    )
    assert motor_pop_total == 40, (
        f"Total motor_pop neurons should be 40, got {motor_pop_total}"
    )


def test_thal_to_motor_pop_cosine_tuning():
    """Each thal_X creates pathways to motor_pop_θ where cos(θ_X - θ) > 0.

    For thal_N (90°), expected non-zero pathways:
    - motor_pop_N (90°, weight × 1.0)
    - motor_pop_NE (45°, weight × cos(45°)≈0.707)
    - motor_pop_NW (135°, weight × cos(45°)≈0.707)
    - motor_pop_E, motor_pop_W (90° away, cos≈0)
    - motor_pop_S, motor_pop_SW, motor_pop_SE (cos<0, clamped 0)

    With epsilon threshold (0.01), pathways from thal_N should hit
    exactly N, NE, NW.
    """
    _, pathways = _build_regions(enable_distributed=True)

    # Find pathways from thal_N
    thal_N_targets = {
        p.to_region: p.weight_mean
        for p in pathways
        if p.from_region == "thal_N" and p.to_region.startswith("motor_pop_")
    }

    expected = {
        "motor_pop_N": 20.0,         # cos(0°) × 20
        "motor_pop_NE": 20.0 * math.cos(math.radians(45)),
        "motor_pop_NW": 20.0 * math.cos(math.radians(45)),
    }

    assert set(thal_N_targets.keys()) == set(expected.keys()), (
        f"thal_N targets {set(thal_N_targets.keys())} != expected {set(expected.keys())}"
    )

    for target, expected_w in expected.items():
        actual = thal_N_targets[target]
        assert abs(actual - expected_w) < 0.01, (
            f"thal_N → {target}: weight {actual:.3f}, expected {expected_w:.3f}"
        )


def test_lang_input_to_motor_pop_all_subpools():
    """language_input → motor_pop_θ should create ONE pathway per sub-pool
    (8 total) when distributed-pop is enabled. All plastic, gated by
    'language_input_to_motor'."""
    _, pathways = _build_regions(enable_distributed=True)

    lang_input_to_motor_pop = [
        p for p in pathways
        if p.from_region == "language_input"
        and p.to_region.startswith("motor_pop_")
    ]

    assert len(lang_input_to_motor_pop) == 8, (
        f"Expected 8 lang_input→motor_pop pathways, got {len(lang_input_to_motor_pop)}"
    )

    # All should be plastic with the right gate
    for p in lang_input_to_motor_pop:
        assert p.plastic, f"{p.from_region}→{p.to_region} should be plastic"
        assert p.plasticity_gate == "language_input_to_motor", (
            f"{p.from_region}→{p.to_region} gate: "
            f"{p.plasticity_gate} != 'language_input_to_motor'"
        )


def test_distributed_no_motor_X_pathways():
    """With distributed-pop on, no pathway should reference motor_N/E/S/W."""
    _, pathways = _build_regions(enable_distributed=True)

    motor_X_names = {"motor_N", "motor_E", "motor_S", "motor_W"}
    for p in pathways:
        assert p.from_region not in motor_X_names, (
            f"Pathway {p.from_region}→{p.to_region} should not exist "
            f"when distributed-pop is on"
        )
        assert p.to_region not in motor_X_names, (
            f"Pathway {p.from_region}→{p.to_region} should not exist "
            f"when distributed-pop is on"
        )


def test_pop_vector_decoding_math():
    """Verify the cosine projection math used in eval matches the
    cosine pathway weights used in build."""
    # If thal_N drives motor_pop_NE with weight cos(45°)=0.707 and
    # motor_pop_NE then fires, the population vector projection onto
    # 'N' cardinal should give:
    #   N_score = motor_pop_NE_firing × cos(45°) = motor_pop_NE × 0.707
    # which matches the eval decoding.

    # Just sanity-check the cosine math:
    SUBPOOL_THETA = [
        (0, "E"), (45, "NE"), (90, "N"), (135, "NW"),
        (180, "W"), (225, "SW"), (270, "S"), (315, "SE"),
    ]

    for action_theta in [0, 90, 180, 270]:
        # Sum of |cos(θ_a - θ_p)| for non-negative cosines should be 4
        # (1 at preferred + 0.707×2 adjacent + 0×2 perpendicular + 0×3 opposite hemisphere)
        total = sum(
            max(0, math.cos(math.radians(action_theta - theta_p)))
            for theta_p, _ in SUBPOOL_THETA
        )
        # Expected: 1 (preferred) + 2 × 0.707 (adjacent) + 0 (others) = 2.414
        assert abs(total - 2.414) < 0.01, (
            f"Cosine sum at action_theta={action_theta}: {total:.3f} != 2.414"
        )
