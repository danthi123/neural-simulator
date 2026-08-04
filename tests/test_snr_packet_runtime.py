"""Focused tests for authenticated SNr runtime packet orchestration."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import sim.snr_packet_runtime as runtime
from sim.snr_executable_packet import (
    AuthorityKind,
    EvidenceKind,
    MaterializedPacket,
    MaterializedParameterLeaf,
    MaterializedUncertainty,
    PacketError,
    UncertaintyKind,
    canonical_bytes,
)


_PACKET_PATH = "packets/snr.json"
_PACKET_DIGEST = "a" * 64
_POLICY_PATH = "packets/policy.json"
_POLICY_DIGEST = "b" * 64


class _Config:
    def __init__(self, regions, payload=None):
        self.brain_regions = list(regions)
        self.snr_authority_policy_path = _POLICY_PATH
        self.snr_authority_policy_sha256 = _POLICY_DIGEST
        self._payload = payload or {"name": "test", "value": 1}

    def to_dict(self):
        return {**self._payload, "regions": [region.name for region in self.brain_regions]}


def _region(name: str, path: str = _PACKET_PATH, digest: str = _PACKET_DIGEST):
    return SimpleNamespace(
        name=name,
        snr_executable_packet_path=path,
        snr_executable_packet_sha256=digest,
    )


def _materialized_packet() -> MaterializedPacket:
    leaf = MaterializedParameterLeaf(
        value="1.25",
        unit="mV",
        uncertainty=MaterializedUncertainty(
            UncertaintyKind.INTERVAL,
            "1.0",
            "1.5",
            "mV",
        ),
        evidence_kind=EvidenceKind.MEASURED,
        authority_kind=AuthorityKind.PRIMARY_SOURCE,
    )
    return MaterializedPacket(
        packet_id="snr-test",
        packet_sha256="c" * 64,
        structural_sha256="d" * 64,
        groups=MappingProxyType(
            {"fast_hh": MappingProxyType({"initial_voltage": leaf})}
        ),
    )


def _executable_packet(materialized: MaterializedPacket):
    return SimpleNamespace(
        canonical_bytes=b'{"packet":"snr-test"}',
        validation_receipt=object(),
        materialized=materialized,
    )


def test_no_packet_regions_do_not_load_authority_policy(monkeypatch):
    def fail_policy(*args, **kwargs):
        raise AssertionError("policy loader must not run")

    monkeypatch.setattr(runtime, "load_authority_policy_file", fail_policy)

    assert runtime.load_runtime_snr_packet_bindings(_Config([])) == {}


def test_source_root_resolution_explicit_env_and_module_fallback(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"
    explicit.mkdir()
    environment.mkdir()

    monkeypatch.setenv("SIM_SOURCE_ROOT", str(environment))
    assert runtime.resolve_simulation_source_root(explicit) == explicit.resolve()
    assert runtime.resolve_simulation_source_root() == environment.resolve()

    monkeypatch.delenv("SIM_SOURCE_ROOT")
    assert runtime.resolve_simulation_source_root() == Path(runtime.__file__).resolve().parents[1]


def test_policy_loaded_once_and_packet_cached_across_two_regions(monkeypatch, tmp_path):
    materialized = _materialized_packet()
    executable = _executable_packet(materialized)
    calls = {"policy": 0, "packet": 0, "materialize": 0}
    policy = object()

    def load_policy(*args, **kwargs):
        calls["policy"] += 1
        assert kwargs["artifact_root"] == tmp_path.resolve()
        return policy

    def load_packet(path, *, artifact_root, expected_sha256, authority_policy):
        calls["packet"] += 1
        assert (path, artifact_root, expected_sha256, authority_policy) == (
            _PACKET_PATH,
            tmp_path.resolve(),
            _PACKET_DIGEST,
            policy,
        )
        return executable

    def materialize(packet, receipt):
        calls["materialize"] += 1
        assert packet is executable
        assert receipt is executable.validation_receipt
        return packet.materialized

    monkeypatch.setattr(runtime, "load_authority_policy_file", load_policy)
    monkeypatch.setattr(runtime, "load_packet_file", load_packet)
    monkeypatch.setattr(runtime, "materialize_packet", materialize)

    bindings = runtime.load_runtime_snr_packet_bindings(
        _Config([_region("snr-a"), _region("snr-b")]),
        source_root=tmp_path,
    )

    assert set(bindings) == {"snr-a", "snr-b"}
    assert bindings["snr-a"].materialized is materialized
    assert bindings["snr-b"].materialized is materialized
    assert calls == {"policy": 1, "packet": 1, "materialize": 1}


def test_digests_are_deterministic_for_same_config_and_materialization(monkeypatch, tmp_path):
    materialized = _materialized_packet()
    executable = _executable_packet(materialized)
    monkeypatch.setattr(runtime, "load_authority_policy_file", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "load_packet_file", lambda *args, **kwargs: executable)
    monkeypatch.setattr(runtime, "materialize_packet", lambda packet, receipt: materialized)

    config = _Config([_region("snr")], payload={"z": 2, "a": 1})
    first = runtime.load_runtime_snr_packet_bindings(config, source_root=tmp_path)["snr"]
    second = runtime.load_runtime_snr_packet_bindings(config, source_root=tmp_path)["snr"]

    assert first.packet_canonical_bytes == second.packet_canonical_bytes
    assert first.config_sha256 == second.config_sha256
    assert first.materialized_sha256 == second.materialized_sha256
    assert first.materialized_sha256 == runtime.materialized_packet_sha256(materialized)
    assert first.config_sha256 == __import__("hashlib").sha256(
        canonical_bytes(config.to_dict())
    ).hexdigest()


def test_duplicate_packet_backed_region_names_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "load_authority_policy_file", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runtime,
        "load_packet_file",
        lambda *args, **kwargs: _executable_packet(_materialized_packet()),
    )
    monkeypatch.setattr(
        runtime,
        "materialize_packet",
        lambda packet, receipt: packet.materialized,
    )

    with pytest.raises(PacketError, match="duplicate packet-backed region name"):
        runtime.load_runtime_snr_packet_bindings(
            _Config([_region("snr"), _region("snr")]),
            source_root=tmp_path,
        )


def test_returned_binding_mapping_is_immutable(monkeypatch, tmp_path):
    materialized = _materialized_packet()
    executable = _executable_packet(materialized)
    monkeypatch.setattr(runtime, "load_authority_policy_file", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "load_packet_file", lambda *args, **kwargs: executable)
    monkeypatch.setattr(runtime, "materialize_packet", lambda packet, receipt: materialized)

    bindings = runtime.load_runtime_snr_packet_bindings(
        _Config([_region("snr")]),
        source_root=tmp_path,
    )

    with pytest.raises(TypeError):
        bindings["other"] = bindings["snr"]


def test_checkpoint_manifest_is_canonical_and_region_order_independent():
    materialized = _materialized_packet()

    def binding(region_name: str):
        return runtime.RuntimeSNrPacketBinding(
            region_name=region_name,
            packet_path=_PACKET_PATH,
            packet_file_sha256=_PACKET_DIGEST,
            packet_canonical_bytes=b'{"packet":"snr-test"}',
            packet_sha256=materialized.packet_sha256,
            structural_sha256=materialized.structural_sha256,
            materialized_sha256=runtime.materialized_packet_sha256(materialized),
            authority_policy_sha256=_POLICY_DIGEST,
            config_sha256="e" * 64,
            materialized=materialized,
        )

    first = {"snr-b": binding("snr-b"), "snr-a": binding("snr-a")}
    second = {"snr-a": first["snr-a"], "snr-b": first["snr-b"]}
    manifest = runtime.runtime_binding_manifest_bytes(first)

    assert manifest == runtime.runtime_binding_manifest_bytes(second)
    assert manifest == canonical_bytes(
        runtime.runtime_binding_manifest_document(first)
    )
    assert b'"packet_canonical_json":"{\\"packet\\":\\"snr-test\\"}"' in manifest
    assert manifest.index(b'"region_name":"snr-a"') < manifest.index(
        b'"region_name":"snr-b"'
    )


def test_packet_trust_references_are_explicit_and_sorted():
    config = _Config([_region("snr-b"), _region("snr-a")])

    assert runtime.packet_trust_reference_document(config) == {
        "authority_policy_path": _POLICY_PATH,
        "authority_policy_sha256": _POLICY_DIGEST,
        "regions": [
            {
                "packet_file_sha256": _PACKET_DIGEST,
                "packet_path": _PACKET_PATH,
                "region_name": "snr-a",
            },
            {
                "packet_file_sha256": _PACKET_DIGEST,
                "packet_path": _PACKET_PATH,
                "region_name": "snr-b",
            },
        ],
    }
    assert runtime.packet_trust_reference_document(_Config([])) is None
