"""Runtime loading and digest binding for authenticated SNr parameter packets.

This module performs construction/provenance work only. It does not implement
neural dynamics or substitute for any biological mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from sim.snr_executable_packet import (
    AuthorityPolicy,
    ExecutablePacket,
    MaterializedPacket,
    PacketError,
    canonical_bytes,
    load_authority_policy_file,
    load_packet_file,
    materialize_packet,
)


RUNTIME_BINDING_SCHEMA = "snr-runtime-packet-binding-v1"


@dataclass(frozen=True, slots=True)
class RuntimeSNrPacketBinding:
    region_name: str
    packet_path: str
    packet_file_sha256: str
    packet_canonical_bytes: bytes
    packet_sha256: str
    structural_sha256: str
    materialized_sha256: str
    authority_policy_sha256: str
    config_sha256: str
    materialized: MaterializedPacket
    schema_version: str = RUNTIME_BINDING_SCHEMA


def resolve_simulation_source_root(source_root: str | Path | None = None) -> Path:
    """Resolve the explicit source tree used for all rooted packet reads."""

    selected = source_root
    if selected is None:
        selected = os.environ.get("SIM_SOURCE_ROOT")
    if selected is None:
        selected = Path(__file__).resolve().parents[1]
    root = Path(os.fspath(selected)).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PacketError("simulation source root must be a directory")
    return root


def materialized_packet_document(packet: MaterializedPacket) -> dict[str, object]:
    """Canonical checkpoint representation of one immutable materialization."""

    if type(packet) is not MaterializedPacket:
        raise PacketError("materialized packet must be an exact MaterializedPacket")
    return {
        "groups": {
            group: {
                parameter: {
                    "authority": leaf.authority_kind.value,
                    "evidence": leaf.evidence_kind.value,
                    "uncertainty": {
                        "kind": leaf.uncertainty.kind.value,
                        "lower": leaf.uncertainty.lower,
                        "unit": leaf.uncertainty.unit,
                        "upper": leaf.uncertainty.upper,
                    },
                    "unit": leaf.unit,
                    "value": leaf.value,
                }
                for parameter, leaf in leaves.items()
            }
            for group, leaves in packet.groups.items()
        },
        "packet_id": packet.packet_id,
        "packet_sha256": packet.packet_sha256,
        "structural_sha256": packet.structural_sha256,
    }


def materialized_packet_sha256(packet: MaterializedPacket) -> str:
    return hashlib.sha256(canonical_bytes(materialized_packet_document(packet))).hexdigest()


def load_runtime_snr_packet_bindings(
    config,
    *,
    source_root: str | Path | None = None,
) -> Mapping[str, RuntimeSNrPacketBinding]:
    """Authenticate and materialize all packet-backed regions in one config."""

    regions = [
        region
        for region in getattr(config, "brain_regions", [])
        if getattr(region, "snr_executable_packet_path", None) is not None
    ]
    if not regions:
        return MappingProxyType({})

    region_names: set[str] = set()
    for region in regions:
        name = getattr(region, "name", None)
        path = getattr(region, "snr_executable_packet_path", None)
        file_sha256 = getattr(region, "snr_executable_packet_sha256", None)
        if not isinstance(name, str) or not name:
            raise PacketError("packet-backed regions require a nonempty name")
        if name in region_names:
            raise PacketError(f"duplicate packet-backed region name: {name}")
        region_names.add(name)
        if not isinstance(path, str) or not path:
            raise PacketError(f"packet-backed region {name!r} requires a packet path")
        if (
            not isinstance(file_sha256, str)
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
        ):
            raise PacketError(
                f"packet-backed region {name!r} requires a lowercase SHA-256 digest"
            )

    policy_path = getattr(config, "snr_authority_policy_path", None)
    policy_sha256 = getattr(config, "snr_authority_policy_sha256", None)
    if not isinstance(policy_path, str) or not isinstance(policy_sha256, str):
        raise PacketError("packet-backed regions require a pinned authority policy")

    root = resolve_simulation_source_root(source_root)
    policy: AuthorityPolicy = load_authority_policy_file(
        policy_path,
        artifact_root=root,
        expected_sha256=policy_sha256,
    )
    config_sha256 = hashlib.sha256(canonical_bytes(config.to_dict())).hexdigest()
    bindings: dict[str, RuntimeSNrPacketBinding] = {}
    loaded_packets: dict[
        tuple[str, str], tuple[ExecutablePacket, MaterializedPacket]
    ] = {}
    for region in regions:
        path = region.snr_executable_packet_path
        file_sha256 = region.snr_executable_packet_sha256
        key = (path, file_sha256)
        cached = loaded_packets.get(key)
        if cached is None:
            packet = load_packet_file(
                path,
                artifact_root=root,
                expected_sha256=file_sha256,
                authority_policy=policy,
            )
            materialized = materialize_packet(packet, packet.validation_receipt)
            loaded_packets[key] = (packet, materialized)
        else:
            packet, materialized = cached
        bindings[region.name] = RuntimeSNrPacketBinding(
            region_name=region.name,
            packet_path=path,
            packet_file_sha256=file_sha256,
            packet_canonical_bytes=packet.canonical_bytes,
            packet_sha256=materialized.packet_sha256,
            structural_sha256=materialized.structural_sha256,
            materialized_sha256=materialized_packet_sha256(materialized),
            authority_policy_sha256=policy_sha256,
            config_sha256=config_sha256,
            materialized=materialized,
        )
    return MappingProxyType(bindings)


__all__ = [
    "RUNTIME_BINDING_SCHEMA",
    "RuntimeSNrPacketBinding",
    "load_runtime_snr_packet_bindings",
    "materialized_packet_document",
    "materialized_packet_sha256",
    "resolve_simulation_source_root",
]
