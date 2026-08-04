"""Typed, digest-pinned configuration references for executable SNr packets."""

from __future__ import annotations

import dataclasses

import pytest

from sim.config import CoreSimConfig, core_sim_config_from_dict
from sim.enums import NeuronModel
from sim.regions import BrainRegion


_DIGEST = "a" * 64
_POLICY_DIGEST = "b" * 64
_PACKET_PATH = "research/packets/snr/stageB-v1/packet.json"
_POLICY_PATH = "research/packets/snr/stageB-v1/authority-policy.json"


def _packet_region(name: str = "snr", n: int = 4) -> BrainRegion:
    return BrainRegion(
        name=name,
        n_neurons=n,
        internal_density=0.0,
        snr_executable_packet_path=_PACKET_PATH,
        snr_executable_packet_sha256=_DIGEST,
    )


def _config(regions: list[BrainRegion], **overrides) -> CoreSimConfig:
    values = {
        "num_neurons": sum(region.n_neurons for region in regions),
        "connections_per_neuron": 0,
        "dt_ms": 0.05,
        "neuron_model_type": NeuronModel.HODGKIN_HUXLEY.name,
        "enable_brain_region_framework": True,
        "brain_regions": regions,
        "region_pathways": [],
        "snr_authority_policy_path": _POLICY_PATH,
        "snr_authority_policy_sha256": _POLICY_DIGEST,
    }
    values.update(overrides)
    return CoreSimConfig(**values)


def test_packet_references_default_absent_and_require_an_exact_pair() -> None:
    region = BrainRegion(name="control", n_neurons=2)
    assert region.snr_executable_packet_enabled is False

    with pytest.raises(ValueError, match="must be set together"):
        BrainRegion(
            name="snr",
            n_neurons=2,
            snr_executable_packet_path=_PACKET_PATH,
        )
    with pytest.raises(ValueError, match="must be set together"):
        BrainRegion(
            name="snr",
            n_neurons=2,
            snr_executable_packet_sha256=_DIGEST,
        )


@pytest.mark.parametrize(
    "bad_path",
    ["/tmp/packet.json", "../packet.json", "a/../packet.json", "./packet.json", "a\\b.json"],
)
def test_packet_path_is_canonical_and_root_relative(bad_path: str) -> None:
    with pytest.raises(ValueError, match="packet_path"):
        BrainRegion(
            name="snr",
            n_neurons=2,
            snr_executable_packet_path=bad_path,
            snr_executable_packet_sha256=_DIGEST,
        )


def test_packet_digest_and_legacy_maxima_fail_closed() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        BrainRegion(
            name="snr",
            n_neurons=2,
            snr_executable_packet_path=_PACKET_PATH,
            snr_executable_packet_sha256="A" * 64,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        BrainRegion(
            name="snr",
            n_neurons=2,
            snr_g_nap_max=0.1,
            snr_executable_packet_path=_PACKET_PATH,
            snr_executable_packet_sha256=_DIGEST,
        )


def test_packet_mode_requires_hh_region_framework_and_policy() -> None:
    region = _packet_region()
    with pytest.raises(ValueError, match="authenticated authority policy"):
        _config(
            [region],
            snr_authority_policy_path=None,
            snr_authority_policy_sha256=None,
        )
    with pytest.raises(ValueError, match="HODGKIN_HUXLEY"):
        _config([region], neuron_model_type=NeuronModel.IZHIKEVICH.name)
    with pytest.raises(ValueError, match="brain-region framework"):
        _config([region], enable_brain_region_framework=False)


def test_policy_reference_is_paired_canonical_and_digest_pinned() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        CoreSimConfig(snr_authority_policy_path=_POLICY_PATH)
    with pytest.raises(ValueError, match="canonical and relative"):
        CoreSimConfig(
            snr_authority_policy_path="../policy.json",
            snr_authority_policy_sha256=_POLICY_DIGEST,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CoreSimConfig(
            snr_authority_policy_path=_POLICY_PATH,
            snr_authority_policy_sha256="bad",
        )


def test_mixed_legacy_and_packet_regions_round_trip_exactly() -> None:
    packet = _packet_region(n=3)
    legacy = BrainRegion(
        name="legacy-snr",
        n_neurons=2,
        internal_density=0.0,
        snr_g_nalcn_max=0.01,
        snr_g_nap_max=0.1,
    )
    config = _config([legacy, packet])

    serialized = config.to_dict()
    assert serialized["brain_regions"][0] == dataclasses.asdict(legacy)
    assert serialized["brain_regions"][1]["snr_executable_packet_path"] == _PACKET_PATH
    assert serialized["snr_authority_policy_sha256"] == _POLICY_DIGEST

    restored = core_sim_config_from_dict(serialized)
    assert all(isinstance(region, BrainRegion) for region in restored.brain_regions)
    assert restored.to_dict() == serialized
    assert restored.brain_regions[0].snr_conductance_bundle_enabled is True
    assert restored.brain_regions[0].snr_executable_packet_enabled is False
    assert restored.brain_regions[1].snr_conductance_bundle_enabled is False
    assert restored.brain_regions[1].snr_executable_packet_enabled is True
