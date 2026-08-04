"""Checkpoint provenance tests for authenticated SNr packet bindings."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import h5py
import pytest

import sim.bridge as bridge_module
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion


_PACKET_PATH = "packets/snr.json"
_PACKET_DIGEST = "a" * 64
_POLICY_PATH = "packets/policy.json"
_POLICY_DIGEST = "b" * 64


def _packet_config(*, packet_digest: str = _PACKET_DIGEST) -> CoreSimConfig:
    region = BrainRegion(
        name="snr",
        n_neurons=2,
        internal_density=0.0,
        snr_executable_packet_path=_PACKET_PATH,
        snr_executable_packet_sha256=packet_digest,
    )
    return CoreSimConfig(
        num_neurons=2,
        connections_per_neuron=0,
        seed=17,
        dt_ms=0.05,
        neuron_model_type=NeuronModel.HODGKIN_HUXLEY.name,
        enable_brain_region_framework=True,
        brain_regions=[region],
        region_pathways=[],
        snr_authority_policy_path=_POLICY_PATH,
        snr_authority_policy_sha256=_POLICY_DIGEST,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        enable_conductance_noise=False,
    )


def _legacy_config() -> CoreSimConfig:
    return CoreSimConfig(
        num_neurons=2,
        connections_per_neuron=0,
        seed=17,
        neuron_model_type=NeuronModel.IZHIKEVICH.name,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_structural_plasticity=False,
        enable_ou_process=False,
        enable_conductance_noise=False,
    )


def _bindings(label: str):
    return MappingProxyType({"snr": SimpleNamespace(label=label)})


def _build_bridge(config, source_root, monkeypatch, bindings, load_calls=None):
    def load_bindings(config, *, source_root):
        if load_calls is not None:
            load_calls.append((config, source_root))
        return bindings

    monkeypatch.setattr(
        bridge_module,
        "load_runtime_snr_packet_bindings",
        load_bindings,
    )
    bridge = SimulationBridge(
        core_config=config,
        simulation_source_root=source_root,
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    return bridge


def _manifest_patch(monkeypatch, expected_bytes: bytes):
    calls = []

    def manifest_bytes(bindings):
        calls.append(bindings)
        return expected_bytes

    monkeypatch.setattr(
        bridge_module, "runtime_binding_manifest_bytes", manifest_bytes
    )
    return calls


def test_save_checkpoint_writes_exact_runtime_manifest_bytes(tmp_path, monkeypatch):
    manifest = b'{"schema_version":"test","bindings":[]}'
    calls = _manifest_patch(monkeypatch, manifest)
    bridge = _build_bridge(
        _packet_config(), tmp_path, monkeypatch, _bindings("live")
    )
    checkpoint = tmp_path / "packet.simstate.h5"

    try:
        assert bridge.save_checkpoint(str(checkpoint)) is True
        with h5py.File(checkpoint, "r") as h5f:
            dataset = h5f[bridge_module._SNR_PACKET_CHECKPOINT_DATASET]
            assert dataset[:].tobytes() == manifest
            assert dataset.attrs["schema"] == bridge_module._SNR_PACKET_CHECKPOINT_SCHEMA
        assert len(calls) == 3
        assert all(call is bridge.snr_packet_bindings for call in calls)
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def test_packet_checkpoint_load_requires_exact_preload_trust_references(
    tmp_path, monkeypatch
):
    manifest = b"packet-manifest-v1"
    _manifest_patch(monkeypatch, manifest)
    source = _build_bridge(
        _packet_config(), tmp_path, monkeypatch, _bindings("same")
    )
    checkpoint = tmp_path / "packet.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True
    source.clear_simulation_state_and_gpu_memory()

    caller_config = _packet_config(packet_digest="c" * 64)
    restored = SimulationBridge(
        core_config=caller_config,
        simulation_source_root=tmp_path,
    )
    try:
        assert restored.load_checkpoint(str(checkpoint)) is False
        assert restored.is_initialized is False
    finally:
        restored.clear_simulation_state_and_gpu_memory()


def test_packet_checkpoint_reloads_live_artifacts_and_matches_manifest(
    tmp_path, monkeypatch
):
    manifest = b"packet-manifest-v1"
    _manifest_patch(monkeypatch, manifest)
    load_calls = []
    source = _build_bridge(
        _packet_config(), tmp_path, monkeypatch, _bindings("same"), load_calls
    )
    checkpoint = tmp_path / "packet.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True
    source.clear_simulation_state_and_gpu_memory()

    restored = SimulationBridge(
        core_config=_packet_config(),
        simulation_source_root=tmp_path,
    )
    try:
        calls_before_load = len(load_calls)
        assert restored.load_checkpoint(str(checkpoint)) is True
        assert restored.snr_packet_bindings["snr"].label == "same"
        assert len(load_calls) == calls_before_load + 1
        assert load_calls[-1][1] is tmp_path
    finally:
        restored.clear_simulation_state_and_gpu_memory()


def test_packet_checkpoint_fails_when_live_manifest_differs(tmp_path, monkeypatch):
    manifests = iter(
        (
            b"packet-manifest-v1",
            b"packet-manifest-v1",
            b"packet-manifest-v1",
            b"packet-manifest-tampered",
        )
    )
    calls = []

    def manifest_bytes(bindings):
        calls.append(bindings)
        return next(manifests)

    monkeypatch.setattr(bridge_module, "runtime_binding_manifest_bytes", manifest_bytes)
    source = _build_bridge(
        _packet_config(), tmp_path, monkeypatch, _bindings("same")
    )
    checkpoint = tmp_path / "packet.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True
    source.clear_simulation_state_and_gpu_memory()

    restored = SimulationBridge(
        core_config=_packet_config(),
        simulation_source_root=tmp_path,
    )
    try:
        assert restored.load_checkpoint(str(checkpoint)) is False
        assert restored.is_initialized is False
    finally:
        restored.clear_simulation_state_and_gpu_memory()


@pytest.mark.parametrize("tamper", ["delete", "bytes"])
def test_packet_checkpoint_missing_or_tampered_manifest_fails(
    tmp_path, monkeypatch, tamper
):
    manifest = b"packet-manifest-v1"
    _manifest_patch(monkeypatch, manifest)
    source = _build_bridge(
        _packet_config(), tmp_path, monkeypatch, _bindings("same")
    )
    checkpoint = tmp_path / f"packet-{tamper}.simstate.h5"
    assert source.save_checkpoint(str(checkpoint)) is True
    source.clear_simulation_state_and_gpu_memory()

    with h5py.File(checkpoint, "r+") as h5f:
        if tamper == "delete":
            del h5f[bridge_module._SNR_PACKET_CHECKPOINT_DATASET]
        else:
            dataset = h5f[bridge_module._SNR_PACKET_CHECKPOINT_DATASET]
            dataset[0] = (int(dataset[0]) + 1) % 256

    restored = SimulationBridge(
        core_config=_packet_config(),
        simulation_source_root=tmp_path,
    )
    try:
        assert restored.load_checkpoint(str(checkpoint)) is False
        assert restored.is_initialized is False
    finally:
        restored.clear_simulation_state_and_gpu_memory()


def test_legacy_checkpoint_without_packet_manifest_remains_compatible(
    tmp_path, monkeypatch
):
    empty = MappingProxyType({})
    monkeypatch.setattr(
        bridge_module,
        "load_runtime_snr_packet_bindings",
        lambda config, *, source_root: empty,
    )
    bridge = _build_bridge(_legacy_config(), tmp_path, monkeypatch, empty)
    checkpoint = tmp_path / "legacy.simstate.h5"

    try:
        assert bridge.save_checkpoint(str(checkpoint)) is True
    finally:
        bridge.clear_simulation_state_and_gpu_memory()

    with h5py.File(checkpoint, "r") as h5f:
        assert bridge_module._SNR_PACKET_CHECKPOINT_DATASET not in h5f

    restored = SimulationBridge(
        core_config=_legacy_config(),
        simulation_source_root=tmp_path,
    )
    try:
        assert restored.load_checkpoint(str(checkpoint)) is True
        assert restored.snr_packet_bindings == {}
    finally:
        restored.clear_simulation_state_and_gpu_memory()
