"""Lifecycle tests for authenticated SNr packet bindings on SimulationBridge."""

from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType

import pytest

import sim.bridge as bridge_module
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig
from sim.enums import NeuronModel


def _config(*, seed: int = 7) -> CoreSimConfig:
    """Small backend-neutral configuration with no connectivity work."""
    return CoreSimConfig(
        num_neurons=2,
        connections_per_neuron=0,
        seed=seed,
        neuron_model_type=NeuronModel.IZHIKEVICH.name,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        enable_conductance_noise=False,
    )


def _binding(label: str):
    return MappingProxyType({label: object()})


def test_initialization_retains_immutable_bindings_and_passes_explicit_root(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "simulation"
    source_root.mkdir()
    returned = _binding("first")
    calls = []

    def load_bindings(config, *, source_root):
        calls.append((config, source_root))
        return returned

    monkeypatch.setattr(bridge_module, "load_runtime_snr_packet_bindings", load_bindings)
    bridge = SimulationBridge(
        core_config=_config(),
        simulation_source_root=source_root,
    )

    try:
        bridge._initialize_simulation_data()

        assert bridge.is_initialized
        assert bridge.snr_packet_bindings is returned
        assert calls == [(bridge.core_config, source_root)]
        with pytest.raises(TypeError):
            bridge.snr_packet_bindings["other"] = object()
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def test_loader_failure_leaves_bridge_uninitialized_without_stale_binding(
    tmp_path, monkeypatch
):
    def fail_load(*args, **kwargs):
        raise RuntimeError("packet authentication failed")

    monkeypatch.setattr(bridge_module, "load_runtime_snr_packet_bindings", fail_load)
    bridge = SimulationBridge(
        core_config=_config(),
        simulation_source_root=tmp_path,
    )
    bridge.snr_packet_bindings = _binding("stale")

    bridge._initialize_simulation_data()

    assert bridge.is_initialized is False
    assert bridge.snr_packet_bindings == {}
    bridge.clear_simulation_state_and_gpu_memory()


def test_clear_simulation_state_and_gpu_memory_resets_bindings(tmp_path):
    bridge = SimulationBridge(
        core_config=_config(),
        simulation_source_root=tmp_path,
    )
    bridge.snr_packet_bindings = _binding("active")

    bridge.clear_simulation_state_and_gpu_memory()

    assert bridge.snr_packet_bindings == {}
    with pytest.raises(TypeError):
        bridge.snr_packet_bindings["other"] = object()


def test_reinitialization_and_reconfiguration_do_not_retain_old_bindings(
    tmp_path, monkeypatch
):
    returned = [_binding("first"), _binding("second"), _binding("third")]
    calls = []

    def load_bindings(config, *, source_root):
        calls.append((config, source_root))
        return returned[len(calls) - 1]

    monkeypatch.setattr(bridge_module, "load_runtime_snr_packet_bindings", load_bindings)
    bridge = SimulationBridge(
        core_config=_config(seed=11),
        simulation_source_root=tmp_path,
    )

    try:
        bridge._initialize_simulation_data()
        assert bridge.snr_packet_bindings is returned[0]

        bridge._initialize_simulation_data()
        assert bridge.snr_packet_bindings is returned[1]
        assert "first" not in bridge.snr_packet_bindings

        replacement = _config(seed=23).to_dict()
        assert bridge.apply_simulation_configuration_core(
            {
                "core_config": replacement,
                "viz_config": asdict(bridge.viz_config),
            }
        )
        assert bridge.snr_packet_bindings is returned[2]
        assert "second" not in bridge.snr_packet_bindings
        assert len(calls) == 3
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def test_packet_backed_recording_capture_is_explicitly_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        bridge_module,
        "load_runtime_snr_packet_bindings",
        lambda config, *, source_root: _binding("active"),
    )
    bridge = SimulationBridge(
        core_config=_config(), simulation_source_root=tmp_path
    )
    try:
        bridge._initialize_simulation_data()
        assert bridge.is_initialized
        with pytest.raises(RuntimeError, match="authenticated provenance"):
            bridge._capture_initial_state_for_recording()
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def test_packet_backed_playback_initialization_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bridge_module,
        "load_runtime_snr_packet_bindings",
        lambda config, *, source_root: _binding("active"),
    )
    bridge = SimulationBridge(
        core_config=_config(), simulation_source_root=tmp_path
    )

    bridge._initialize_simulation_data(called_from_playback_init=True)

    assert bridge.is_initialized is False
    assert bridge.snr_packet_bindings == {}
    bridge.clear_simulation_state_and_gpu_memory()
