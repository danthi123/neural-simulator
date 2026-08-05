"""Focused validation for the authenticated GPU-batched Stage B runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.runners.v14_stageB_batched_physiology import (
    DECLARATION_SCHEMA,
    DT_MS,
    EVENT_TIMEOUT_STEPS,
    NAP_STEPS,
    OUTPUT_SCHEMA,
    PHASED_OUTPUT_SCHEMA,
    StageBBatchedPhysiologyError,
    load_batch_declaration,
    run_authenticated_gpu_batch,
)
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.snr_executable_packet import canonical_bytes
from tests.test_v14_stageB_packet_compiler import _candidate, _template
from tools.v14_stageB_packet_compiler import compile_candidate
from tools.v14_stageB_packet_verifier import verify_candidate


ROOT = Path(__file__).resolve().parents[1]


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value) -> str:
    return _digest_bytes(canonical_bytes(value))


def _copy_protocol_contract(root: Path, version: int = 1) -> dict[str, str]:
    suffix = "" if version == 1 else f"_v{version}"
    for relative in (
        Path(f"research/specs/v14_snr_stageB_causal_gates{suffix}.json"),
        Path(f"research/specs/v14_snr_stageB_intrinsic_protocol{suffix}.json"),
        Path("research/specs/v14_snr_stageB_target_packet.json"),
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    protocol = root / f"research/specs/v14_snr_stageB_intrinsic_protocol{suffix}.json"
    return {
        "path": f"research/specs/v14_snr_stageB_intrinsic_protocol{suffix}.json",
        "sha256": _digest_bytes(protocol.read_bytes()),
    }


def _write_candidate(root: Path, index: int) -> dict[str, object]:
    candidate = _candidate()
    candidate["candidate_id"] = f"gpu-batch-candidate-{index}"
    # Keep candidates scientifically distinct while remaining inside the filed template.
    candidate["parameters"]["g_nalcn"] *= 1.0 + index * 0.05
    template_path = root / f"inputs/template-{index}.json"
    candidate_path = root / f"inputs/candidate-{index}.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_bytes(canonical_bytes(_template()))
    candidate_path.write_bytes(canonical_bytes(candidate))
    output = root / f"packets/candidate-{index}"
    compile_candidate(
        template_path,
        _digest_bytes(template_path.read_bytes()),
        candidate_path,
        _digest_bytes(candidate_path.read_bytes()),
        output,
        repository_root=root,
    )
    release_receipt = verify_candidate(
        template_path,
        _digest_bytes(template_path.read_bytes()),
        output,
        repository_root=root,
    )
    release_path = output / "candidate-release.json"
    packet_path = output / "packet.sealed.json"
    policy_path = output / "authority-policy.json"
    assert release_receipt["candidate_release_sha256"] == _digest_bytes(release_path.read_bytes())
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": _digest(candidate),
        "release": {
            "path": release_path.relative_to(root).as_posix(),
            "sha256": _digest_bytes(release_path.read_bytes()),
        },
        "packet": {
            "path": packet_path.relative_to(root).as_posix(),
            "sha256": _digest_bytes(packet_path.read_bytes()),
        },
        "policy": {
            "path": policy_path.relative_to(root).as_posix(),
            "sha256": _digest_bytes(policy_path.read_bytes()),
        },
    }


def _write_declaration(
    root: Path,
    *,
    arm: str = "intact_autonomous",
    protocol_version: int = 1,
    mutate=None,
) -> tuple[Path, str, dict[str, object]]:
    body = {
        "schema": DECLARATION_SCHEMA,
        "arm": arm,
        "analysis_protocol": _copy_protocol_contract(root, protocol_version),
        "candidates": [_write_candidate(root, 0), _write_candidate(root, 1)],
    }
    if mutate is not None:
        mutate(body)
    document = {**body, "sha256": _digest(body)}
    path = root / "batches/stageB.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))
    return path, _digest_bytes(path.read_bytes()), document


class _FakeNullStream:
    synchronizations = 0

    @classmethod
    def synchronize(cls):
        cls.synchronizations += 1


class _FakeXP:
    __name__ = "cupy"
    float32 = np.float32
    bool_ = np.bool_
    empty = staticmethod(np.empty)
    asnumpy = staticmethod(np.asarray)
    cuda = SimpleNamespace(Stream=SimpleNamespace(null=_FakeNullStream))


class _FakeBinding:
    def __init__(self, region: BrainRegion):
        self.region_name = region.name
        self.packet_path = region.snr_executable_packet_path
        self.packet_file_sha256 = region.snr_executable_packet_sha256
        self.packet_sha256 = "1" * 64
        self.structural_sha256 = "2" * 64
        self.materialized_sha256 = "3" * 64
        self.authority_policy_sha256 = region.snr_authority_policy_sha256
        self.config_sha256 = "4" * 64


class _FakeBridge:
    latest = None

    def __init__(self, *, core_config, gpu_config, **kwargs):
        type(self).latest = self
        self.core_config = core_config
        self.gpu_config = gpu_config
        self.is_initialized = False
        self.step = 0
        self.cleared = False
        count = core_config.num_neurons
        self.cp_membrane_potential_v = np.full(count, -60.0, dtype=np.float32)
        self.cp_firing_states = np.zeros(count, dtype=bool)
        self.cp_snr_g_nap_max = np.arange(1, count + 1, dtype=np.float32)
        self.cp_snr_g_ca_max = np.arange(2, count + 2, dtype=np.float32)
        self.cp_snr_g_sk_max = np.arange(3, count + 3, dtype=np.float32)
        self.cp_snr_g_h_max = np.arange(4, count + 4, dtype=np.float32)
        self.snr_packet_bindings = {
            region.name: _FakeBinding(region) for region in core_config.brain_regions
        }

    def _initialize_simulation_data(self):
        self.is_initialized = True

    def _run_one_simulation_step(self):
        self.step += 1
        indices = np.arange(self.core_config.num_neurons)
        periods = indices + 1
        self.cp_membrane_potential_v[:] = -60.0 + self.step * 0.001 + indices
        self.cp_firing_states[:] = self.step % periods == 0

    def _snr_direct_outputs_can_dispatch(self, config):
        return config.enable_snr_direct_outputs

    def clear_simulation_state_and_gpu_memory(self):
        self.cleared = True


class _FakeGPUConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_runtime() -> SimpleNamespace:
    _FakeNullStream.synchronizations = 0
    return SimpleNamespace(
        xp=_FakeXP,
        SimulationBridge=_FakeBridge,
        CoreSimConfig=CoreSimConfig,
        GPUConfig=_FakeGPUConfig,
        RuntimeState=lambda: object(),
        VisualizationConfig=lambda: object(),
        NeuronModel=NeuronModel,
        BrainRegion=BrainRegion,
        runtime_binding_manifest_bytes=lambda bindings: canonical_bytes(
            {"regions": sorted(bindings)}
        ),
    )


def _unpack_spikes(trace: dict[str, object]) -> np.ndarray:
    return np.unpackbits(
        trace["spike_states_packed"], bitorder=trace["spike_bitorder"]
    )[: trace["sample_count"]].astype(bool)


def test_mocked_batch_runs_candidates_concurrently_and_trims_each_at_its_101st_spike(
    tmp_path: Path,
):
    path, digest, _ = _write_declaration(tmp_path)

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=64,
        _runtime=_fake_runtime(),
    )

    assert result["schema"] == OUTPUT_SCHEMA
    assert result["scientific_verdict"] is None
    assert result["numpy_confirmation_required"] is True
    assert result["source_equivalence_claimed"] is False
    assert result["execution"]["candidate_count"] == 2
    assert result["execution"]["bridge_steps_executed"] == 256
    assert result["execution"]["trace_synchronization_boundaries"] == 4
    assert [row["termination"]["steps_executed"] for row in result["candidates"]] == [101, 202]
    assert [row["termination"]["spikes_observed"] for row in result["candidates"]] == [101, 101]
    assert all(np.count_nonzero(_unpack_spikes(row["trace"])) == 101 for row in result["candidates"])
    assert all(row["trace"]["voltage_mV"].dtype == np.float32 for row in result["candidates"])
    assert all(row["trace"]["spike_states_packed"].dtype == np.uint8 for row in result["candidates"])
    assert _FakeBridge.latest.cleared is True


def test_config_is_one_authenticated_read_only_region_per_candidate(tmp_path: Path):
    path, digest, document = _write_declaration(tmp_path)

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=256,
        _runtime=_fake_runtime(),
    )

    config = _FakeBridge.latest.core_config
    assert config.num_neurons == len(config.brain_regions) == 2
    assert config.connections_per_neuron == 0
    assert config.region_pathways == []
    assert config.snr_authority_policy_path is None
    assert config.enable_snr_direct_outputs is True
    assert config.read_only_fast_step is True
    assert all(region.n_neurons == 1 and region.internal_density == 0.0 for region in config.brain_regions)
    assert [region.snr_executable_packet_path for region in config.brain_regions] == [
        row["packet"]["path"] for row in document["candidates"]
    ]
    assert [region.snr_authority_policy_path for region in config.brain_regions] == [
        row["policy"]["path"] for row in document["candidates"]
    ]
    for name in (
        "enable_parameter_heterogeneity",
        "enable_conductance_noise",
        "enable_hebbian_learning",
        "enable_short_term_plasticity",
        "enable_structural_plasticity",
        "enable_homeostasis",
        "enable_stdp",
        "enable_inhibitory_stdp",
        "enable_reward_modulation",
        "enable_ou_process",
    ):
        assert getattr(config, name) is False
    assert len(result["provenance"]["runtime_bindings"]) == 2
    assert all("config_sha256" in row for row in result["provenance"]["runtime_bindings"])


def test_nap_arm_runs_exactly_20000_steps_and_records_complete_lesion(tmp_path: Path):
    path, digest, _ = _write_declaration(tmp_path, arm="nap_lesion")

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=4096,
        _runtime=_fake_runtime(),
    )

    assert result["execution"]["bridge_steps_executed"] == NAP_STEPS
    assert result["execution"]["trace_synchronization_boundaries"] == math.ceil(
        NAP_STEPS / 4096
    )
    for row in result["candidates"]:
        assert row["termination"] == {
            "mode": "fixed_duration",
            "reason": "fixed_duration_complete",
            "steps_executed": NAP_STEPS,
            "spikes_observed": row["termination"]["spikes_observed"],
            "target_spike_count": None,
            "maximum_steps": NAP_STEPS,
            "timeout_is_physiology_failure": False,
        }
        assert row["trace"]["sample_count"] == NAP_STEPS
        assert row["trace"]["recording_end_s"] == pytest.approx(
            (NAP_STEPS + 1) * DT_MS / 1000.0
        )
        assert row["runtime_intervention"]["target"] == "nap"
        assert row["runtime_intervention"]["after"] == 0.0


def test_v3_nap_arm_runs_same_cells_intact_then_lesions_at_exact_onset(tmp_path: Path):
    path, digest, _ = _write_declaration(
        tmp_path, arm="nap_lesion", protocol_version=3
    )

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=4096,
        _runtime=_fake_runtime(),
    )

    assert result["schema"] == PHASED_OUTPUT_SCHEMA
    assert result["execution"]["same_cell_phased_nap"] is True
    assert result["execution"]["bridge_steps_executed"] == 60_000
    assert result["execution"]["intact_baseline_s"] == [0.0, 2.0]
    assert result["execution"]["post_lesion_s"] == [2.0, 3.0]
    assert _FakeBridge.latest.step == 60_000
    for row in result["candidates"]:
        trace = row["trace"]
        intervention = row["runtime_intervention"]
        assert trace["sample_count"] == 60_000
        assert trace["voltage_mV"][39_998] < trace["voltage_mV"][39_999]
        assert intervention["before"] > 0.0
        assert intervention["after"] == 0.0
        assert intervention["timestamp_s"] == 2.0
        assert intervention["lesion_onset_sample_index"] == 39_999
        assert intervention["lesion_onset_sample_number"] == 40_000
        assert intervention["last_intact_sample_s"] == pytest.approx(1.99995)
        assert intervention["first_lesion_sample_s"] == 2.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body.update({"scientific_seed": 42}), "seed data"),
        (lambda body: body.update({"held_out_data": [1]}), "held-out data"),
        (
            lambda body: body["candidates"][1].update(
                {
                    "candidate_id": body["candidates"][0]["candidate_id"],
                    "candidate_sha256": body["candidates"][0]["candidate_sha256"],
                }
            ),
            "duplicate candidates",
        ),
        (
            lambda body: body["candidates"][0].update({"candidate_id": "wrong-release-owner"}),
            "release does not bind",
        ),
    ],
)
def test_declaration_rejects_seeds_held_out_duplicates_and_release_mismatches(
    tmp_path: Path, mutation, message: str
):
    path, digest, _ = _write_declaration(tmp_path, mutate=mutation)

    with pytest.raises(StageBBatchedPhysiologyError, match=message):
        load_batch_declaration(path, digest, repository_root=tmp_path)


def test_declaration_fails_closed_on_tampering_and_path_escape(tmp_path: Path):
    path, digest, document = _write_declaration(tmp_path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(StageBBatchedPhysiologyError, match="digest does not match"):
        load_batch_declaration(path, digest, repository_root=tmp_path)

    document["candidates"][0]["packet"]["path"] = "../packet.sealed.json"
    body = {key: value for key, value in document.items() if key != "sha256"}
    document["sha256"] = _digest(body)
    path.write_bytes(canonical_bytes(document))
    with pytest.raises(StageBBatchedPhysiologyError, match="repository-relative"):
        load_batch_declaration(
            path, _digest_bytes(path.read_bytes()), repository_root=tmp_path
        )


def test_protocol_and_release_file_digests_are_rechecked(tmp_path: Path):
    path, _, document = _write_declaration(tmp_path)
    protocol = tmp_path / document["analysis_protocol"]["path"]
    protocol.write_bytes(protocol.read_bytes() + b" ")
    path_digest = _digest_bytes(path.read_bytes())
    with pytest.raises(StageBBatchedPhysiologyError, match="analysis protocol digest"):
        load_batch_declaration(path, path_digest, repository_root=tmp_path)

    protocol.write_bytes((ROOT / document["analysis_protocol"]["path"]).read_bytes())
    release = tmp_path / document["candidates"][0]["release"]["path"]
    release.write_bytes(release.read_bytes() + b" ")
    with pytest.raises(StageBBatchedPhysiologyError, match="release digest"):
        load_batch_declaration(path, path_digest, repository_root=tmp_path)


def test_event_timeout_is_unavailable_boundary_not_failure(tmp_path: Path):
    path, digest, _ = _write_declaration(tmp_path)
    runtime = _fake_runtime()

    def no_spike_step(self):
        self.step += 1
        self.cp_membrane_potential_v[:] = -60.0
        self.cp_firing_states[:] = False

    original = _FakeBridge._run_one_simulation_step
    _FakeBridge._run_one_simulation_step = no_spike_step
    try:
        result = run_authenticated_gpu_batch(
            path,
            digest,
            repository_root=tmp_path,
            chunk_steps=65536,
            _runtime=runtime,
        )
    finally:
        _FakeBridge._run_one_simulation_step = original

    assert result["execution"]["bridge_steps_executed"] == EVENT_TIMEOUT_STEPS
    assert all(row["termination"]["reason"] == "maximum_duration_reached" for row in result["candidates"])
    assert all(row["termination"]["timeout_is_physiology_failure"] is False for row in result["candidates"])
    assert result["scientific_verdict"] is None


@pytest.mark.skipif(
    os.environ.get("RUN_V14_STAGEB_REAL_GPU") != "1",
    reason="set RUN_V14_STAGEB_REAL_GPU=1 for the two-candidate RTX/CuPy integration",
)
def test_two_candidate_real_gpu_nap_smoke(tmp_path: Path):
    path, digest, _ = _write_declaration(tmp_path, arm="nap_lesion")

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=4096,
    )

    assert result["backend"] == "cupy"
    assert result["device"] == "cuda"
    assert len(result["candidates"]) == 2
    assert all(row["trace"]["sample_count"] == NAP_STEPS for row in result["candidates"])
    assert result["scientific_verdict"] is None


@pytest.mark.skipif(
    os.environ.get("RUN_V14_STAGEB_REAL_GPU") != "1",
    reason="set RUN_V14_STAGEB_REAL_GPU=1 for the V3 same-cell RTX/CuPy integration",
)
def test_two_candidate_real_gpu_phased_nap_smoke(tmp_path: Path):
    path, digest, _ = _write_declaration(
        tmp_path, arm="nap_lesion", protocol_version=3
    )

    result = run_authenticated_gpu_batch(
        path,
        digest,
        repository_root=tmp_path,
        chunk_steps=4096,
    )

    assert result["schema"] == PHASED_OUTPUT_SCHEMA
    assert result["execution"]["bridge_steps_executed"] == 60_000
    assert len(result["candidates"]) == 2
    assert all(row["trace"]["sample_count"] == 60_000 for row in result["candidates"])
    assert all(row["runtime_intervention"]["after"] == 0.0 for row in result["candidates"])
    assert result["scientific_verdict"] is None
