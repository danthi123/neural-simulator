"""Tests for bridge.set_token_drive() and bridge.read_language_output()
— integration of text I/O with the brain-region framework.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_bridge_with_text_io():
    """Build a bridge with text I/O + visual cortex + minimal BG cascade
    so we can run a few simulation steps without errors."""
    pytest.importorskip("cupy")

    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.g11_bg_runner import build_bg_brain_regions

    regions, pathways = build_bg_brain_regions(enable_text_io=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = 42

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def test_set_token_drive_activates_correct_neurons():
    """set_token_drive('north') must inject current into ~10% of
    language_input neurons (sparse pattern)."""
    import cupy as cp

    bridge = _make_bridge_with_text_io()
    bridge.cp_external_input_current[:] = 0.0

    n_active = bridge.set_token_drive("north", drive_pA=200.0, sparsity=0.1)

    # ~10% of 256 = 25-26 active
    assert 20 <= n_active <= 30, f"n_active={n_active} not ~10%"

    # Verify the activation is on language_input neurons, not elsewhere
    lang_input_idx = list(bridge.region_manager.indices("language_input"))
    lang_idx_cp = cp.asarray(lang_input_idx, dtype=cp.int64)
    drive_in_lang = bridge.cp_external_input_current[lang_idx_cp]
    n_active_in_lang = int((drive_in_lang > 0).sum().get())
    assert n_active_in_lang == n_active, (
        f"Active neurons not in language_input: {n_active_in_lang} vs {n_active}"
    )


def test_set_token_drive_deterministic():
    """Same token should produce same drive pattern across calls."""
    import cupy as cp

    bridge = _make_bridge_with_text_io()
    bridge.cp_external_input_current[:] = 0.0

    bridge.set_token_drive("north", drive_pA=200.0, sparsity=0.1)
    drive1 = bridge.cp_external_input_current.copy()
    bridge.cp_external_input_current[:] = 0.0
    bridge.set_token_drive("north", drive_pA=200.0, sparsity=0.1)
    drive2 = bridge.cp_external_input_current.copy()

    assert bool((drive1 == drive2).all().get())


def test_set_token_drive_different_tokens_different_patterns():
    """'north' and 'east' must activate different neuron sets."""
    bridge = _make_bridge_with_text_io()
    bridge.cp_external_input_current[:] = 0.0

    bridge.set_token_drive("north", drive_pA=200.0, sparsity=0.1)
    drive_n = bridge.cp_external_input_current.get()
    bridge.cp_external_input_current[:] = 0.0
    bridge.set_token_drive("east", drive_pA=200.0, sparsity=0.1)
    drive_e = bridge.cp_external_input_current.get()

    overlap = int(np.sum((drive_n > 0) & (drive_e > 0)))
    n_active = int(np.sum(drive_n > 0))
    assert overlap < n_active // 2, (
        f"north/east drive overlap={overlap} too high"
    )


def test_set_token_drive_unknown_region_raises():
    bridge = _make_bridge_with_text_io()

    with pytest.raises(RuntimeError, match="not found"):
        bridge.set_token_drive("hello", region_name="nonexistent_region")


def test_read_language_output_returns_token():
    """read_language_output(spike_counts) must return a token from vocab."""
    from sim.text_embeddings import DEFAULT_VOCAB, embed

    bridge = _make_bridge_with_text_io()
    # Synthesize spike_counts that match 'north' embedding pattern
    target = embed("north", dim=256)
    # Convert embedding to a positive count vector (rectify + scale)
    counts = np.maximum(target, 0.0) * 100.0
    counts = counts.astype(np.int32)

    got = bridge.read_language_output(
        spike_counts=counts, n_steps=10, top_k=1,
    )
    assert isinstance(got, list)
    assert len(got) == 1
    assert got[0] in DEFAULT_VOCAB


def test_read_language_output_top_k():
    """top_k=3 returns 3 ranked tokens."""
    from sim.text_embeddings import DEFAULT_VOCAB, embed

    bridge = _make_bridge_with_text_io()
    target = embed("goal", dim=256)
    counts = (np.maximum(target, 0.0) * 50.0).astype(np.int32)

    got = bridge.read_language_output(
        spike_counts=counts, n_steps=10, top_k=3,
    )
    assert len(got) == 3
    # Top-1 should be 'goal' since we synthesized goal's pattern
    assert got[0] == "goal"


def test_set_token_drive_then_step_no_crash():
    """Sanity: after setting token drive, the bridge should run a
    simulation step without crashing."""
    bridge = _make_bridge_with_text_io()
    bridge.cp_external_input_current[:] = 0.0

    bridge.set_token_drive("north", drive_pA=200.0)
    # Try a single step
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
