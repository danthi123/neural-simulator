"""Focused, simulator-free tests for the V13 state-transplant diagnostic."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

from research.runners import _v13_backend_state_transplant as transplant  # noqa: E402


AUTHORITATIVE_SPEC_PATH = Path(
    "/home/dant123/Projects/sim-worktrees/gate-b-v2-clean/"
    "research/specs/v13_backend_state_transplant.json"
)
AUTHORITATIVE_SPEC_SHA256 = "cd29d694d5e4d413167d83f0817532c1e31695ccb0d9a72490e81671c8ff920f"
STALE_SHORT_SPEC_SHA256 = "968323593b1f9bc7ffc7eab3e4111b721ad6147cfdfdb8a2bdd353aa376ef3f8"


def _locked_spec():
    return {
        "schema_version": 1,
        "status": "locked",
        "mechanism": "gateB-v13-backend-state-transplant-diagnostic",
        "promotion_value": "none",
        "source_anchor_sha": transplant.SOURCE_ANCHOR_SHA,
        "seed": transplant.LOCKED_SEED,
        "seed_derivation": {
            "material": (
                "V13_BACKEND_STATE_TRANSPLANT_V1|"
                f"{transplant.SOURCE_ANCHOR_SHA}|role=paired_origin"
            ),
            "sha256_prefix_12": "b6388351e7c8",
            "formula": "2000000 + (prefix_integer mod 7000000)",
        },
        "forbidden_seeds": [1013, 1019, 1021, 1031],
        "origins": ["numpy", "cupy"],
        "execution_backends": ["numpy", "cupy"],
        "modes": ["default", "deterministic_transpose_matvec"],
        "expected_matrix_cells": 8,
        "network": copy.deepcopy(transplant.LOCKED_NETWORK),
        "steps": {"baseline": 500, "inhibition": 200, "release": 500, "dt_ms": 1.0},
        "stimulus": {"source_current_pA": 1000.0, "target_external_current_pA": 0.0},
        "required_bundle_arrays": [
            "C", "a", "b", "d", "k", "vr", "vt", "vpeak",
            "v", "u", "g_e", "g_i", "intrinsic_current", "external_current",
            "csr_data", "csr_indices", "csr_indptr",
        ],
        "required_trajectory_arrays": ["v", "u", "g_e", "g_i", "spikes"],
        "required_state_scope": "all_allocated_cp_ndarrays",
        "comparison_tolerance": {
            "continuous_rtol": 1e-6,
            "continuous_atol": 1e-6,
            "spikes": "exact",
        },
        "verdict": "DIAGNOSTIC_ONLY",
    }


def _write_spec(tmp_path, spec=None):
    spec = copy.deepcopy(_locked_spec() if spec is None else spec)
    path = tmp_path / "v13_backend_state_transplant.json"
    path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), spec


def _write_artifact(path, artifact):
    path.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n")


class _Regions:
    def indices(self, name):
        return {
            "inhibitory_source": list(range(20)),
            "gpi_snr": list(range(20, 60)),
        }[name]


class _FakeBridge:
    """Small deterministic bridge; it never calls the scientific runner."""

    def __init__(self, origin):
        offset = np.float32(1.0 if origin == "cupy" else 0.0)
        source_n, target_n, n = 20, 40, 60
        source_region = SimpleNamespace(
            name="inhibitory_source", n_neurons=source_n, intrinsic_current_pA=0.0,
        )
        target_region = SimpleNamespace(
            name="gpi_snr", n_neurons=target_n, intrinsic_current_pA=100.0,
        )
        pathway = SimpleNamespace(
            from_region="inhibitory_source", to_region="gpi_snr",
            density=1.0, weight_mean=8.0, weight_jitter=0.0,
            plastic=False, receptor="gaba_a",
        )
        self.core_config = SimpleNamespace(
            num_neurons=n,
            dt_ms=1.0,
            seed=transplant.LOCKED_SEED,
            heterogeneity_seed=transplant.LOCKED_SEED,
            neuron_model_type="IZHIKEVICH",
            deterministic_transpose_matvec=False,
            brain_regions=[source_region, target_region],
            region_pathways=[pathway],
        )
        for name in transplant.RUNTIME_DISABLED_FLAGS:
            setattr(self.core_config, name, False)
        for name in transplant.RUNTIME_ENABLED_FLAGS:
            setattr(self.core_config, name, True)
        self.gpu_config = SimpleNamespace(enable_step_profiler=False)
        self.experiment_engine = None
        self.data_bus = None
        self.synapse_store = None
        self.recording_file_handle = None
        self._engram_recordings = []
        self._gate_couplings = []
        self.runtime_state = SimpleNamespace(current_time_ms=0.0, current_time_step=0)
        self.region_manager = _Regions()
        self.cp_izh_C = np.r_[
            np.full(source_n, 20, np.float32), np.full(target_n, 60, np.float32)
        ] + offset
        self.cp_izh_a = np.r_[
            np.full(source_n, 0.1, np.float32), np.full(target_n, 0.05, np.float32)
        ] + offset * np.float32(0.001)
        self.cp_izh_b = np.r_[
            np.full(source_n, 0.2, np.float32), np.full(target_n, 2.0, np.float32)
        ]
        self.cp_izh_d_increment = np.r_[
            np.full(source_n, 2, np.float32), np.full(target_n, 25, np.float32)
        ]
        self.cp_izh_k = np.ones(n, np.float32)
        self.cp_izh_vr = np.r_[
            np.full(source_n, -55, np.float32), np.full(target_n, -65, np.float32)
        ]
        self.cp_izh_vt = np.r_[
            np.full(source_n, -40, np.float32), np.full(target_n, -50, np.float32)
        ]
        self.cp_izh_vpeak = np.full(n, 25, np.float32)
        self.cp_izh_c_reset = np.r_[
            np.full(source_n, -55, np.float32), np.full(target_n, -60, np.float32)
        ]
        self.cp_membrane_potential_v = self.cp_izh_vr.copy()
        self.cp_recovery_variable_u = np.zeros(n, np.float32)
        self.cp_conductance_g_e = np.zeros(n, np.float32)
        self.cp_conductance_g_i = np.zeros(n, np.float32)
        self.cp_intrinsic_current_pA = np.r_[
            np.zeros(source_n, np.float32), np.full(target_n, 100, np.float32)
        ]
        self.cp_external_input_current = np.zeros(n, np.float32)
        self.cp_firing_states = np.zeros(n, np.bool_)
        self.cp_prev_firing_states = np.zeros(n, np.bool_)
        self.cp_refractory_timers = np.zeros(n, np.int32)
        self.cp_neuron_type_ids = np.r_[
            np.ones(source_n, np.int32), np.full(target_n, 2, np.int32)
        ]
        self.cp_traits = np.r_[
            np.ones(source_n, np.int32), np.zeros(target_n, np.int32)
        ]
        self.cp_heterogeneity_neuron_mask = np.r_[
            np.zeros(source_n, np.bool_), np.ones(target_n, np.bool_)
        ]
        # Deliberately not neuron-leading: proves complete dynamic-array capture.
        self.cp_extra_dynamic = np.array([3.25, 7.5], np.float32)
        pre = np.repeat(np.arange(source_n, dtype=np.int32), target_n)
        post = np.tile(np.arange(source_n, n, dtype=np.int32), source_n)
        self.cp_connections = csr_matrix(
            (np.full(pre.size, 8.0, np.float32), (pre, post)), shape=(n, n)
        )
        self._cached_inhibitory_mask = None
        self._cached_coo_matrix = None

    def _run_one_simulation_step(self):
        step = int(self.runtime_state.current_time_step)
        self.cp_prev_firing_states[:] = self.cp_firing_states
        self.cp_firing_states[:] = False
        source_on = bool(np.any(self.cp_external_input_current[:20] > 0.0))
        self.cp_firing_states[:20] = source_on
        in_inhibition = 500 <= step < 700
        target_period = 100 if in_inhibition else 10
        if step % target_period == 0:
            self.cp_firing_states[20:] = True
        self.cp_conductance_g_e *= np.float32(0.9)
        self.cp_conductance_g_i *= np.float32(0.9)
        if source_on:
            self.cp_conductance_g_i[20:] += np.float32(8.0)
        self.cp_membrane_potential_v += np.float32(0.01)
        self.cp_recovery_variable_u += np.float32(0.001)

    def clear_simulation_state_and_gpu_memory(self):
        pass


@pytest.fixture
def fake_runtime(monkeypatch):
    active = {"backend": "numpy"}
    built = []

    def select_backend(name):
        active["backend"] = name

    def build(spec):
        built.append(spec["seed"])
        return _FakeBridge(active["backend"])

    monkeypatch.setattr(transplant, "_assert_backend", select_backend)
    monkeypatch.setattr(transplant, "_build_bridge_from_spec", build)
    monkeypatch.setattr(transplant, "_anchor_is_ancestor", lambda anchor: True)
    monkeypatch.setattr(transplant, "synchronize", lambda: None)
    return {"active": active, "built": built}


def _bundle(tmp_path, fake_runtime, origin):
    spec_path, digest, _ = _write_spec(tmp_path)
    path = tmp_path / f"bundle-{origin}.json"
    artifact = transplant.create_bundle(spec_path, digest, origin, path)
    return spec_path, digest, path, artifact


def _run(tmp_path, spec_path, digest, bundle_path, origin, backend, mode):
    path = tmp_path / f"run-{origin}-{backend}-{mode}.json"
    artifact = transplant.execute_bundle(
        spec_path, digest, bundle_path, backend, mode, path
    )
    return path, artifact


def test_loads_committed_schema_and_rejects_digest_or_formal_seed(tmp_path, fake_runtime):
    path, digest, expected = _write_spec(tmp_path)
    spec, actual = transplant.load_locked_spec(path, digest)
    assert spec == expected
    assert actual == digest
    assert transplant.RTOL == 1e-6
    assert transplant.ATOL == 1e-6

    with pytest.raises(ValueError, match="spec digest mismatch"):
        transplant.load_locked_spec(path, "0" * 64)

    formal = _locked_spec()
    formal["seed"] = 1019
    formal_path, formal_digest, _ = _write_spec(tmp_path, formal)
    with pytest.raises(ValueError, match="seed_is_not_formal"):
        transplant.load_locked_spec(formal_path, formal_digest)
    assert fake_runtime["built"] == []


def test_loads_exact_authoritative_spec_path_and_rejects_stale_digest(fake_runtime):
    actual = hashlib.sha256(AUTHORITATIVE_SPEC_PATH.read_bytes()).hexdigest()
    assert actual == AUTHORITATIVE_SPEC_SHA256
    spec, digest = transplant.load_locked_spec(
        AUTHORITATIVE_SPEC_PATH, AUTHORITATIVE_SPEC_SHA256
    )
    assert digest == AUTHORITATIVE_SPEC_SHA256
    assert spec["expected_matrix_cells"] == 8
    assert spec["network"] == transplant.LOCKED_NETWORK
    assert spec["required_state_scope"] == "all_allocated_cp_ndarrays"
    assert transplant._comparison_tolerance(spec) == {
        "rtol": 1e-6, "atol": 1e-6, "spikes": "exact",
    }
    with pytest.raises(ValueError, match="spec digest mismatch"):
        transplant.load_locked_spec(
            AUTHORITATIVE_SPEC_PATH, STALE_SHORT_SPEC_SHA256
        )
    assert fake_runtime["built"] == []


def test_bundle_captures_every_cp_array_and_restores_exactly(tmp_path, fake_runtime):
    spec_path, digest, bundle_path, bundle = _bundle(tmp_path, fake_runtime, "numpy")
    assert bundle["verdict"] == "DIAGNOSTIC_ONLY"
    assert bundle["seed"] == transplant.LOCKED_SEED
    assert bundle["initialization_disclosure"] == {
        "backend_native_bridge_initialized": True,
        "initialization_may_have_used_rng": True,
        "sealed_state_captured_after_initialization": True,
        "claim_of_no_rng_call": False,
    }
    assert bundle["runtime_config_contract"]["runtime_random_processes_disabled"]
    assert "cp_extra_dynamic" in bundle["cp_arrays"]
    assert transplant._decode_array(
        bundle["cp_arrays"]["cp_extra_dynamic"], "extra"
    ).shape == (2,)
    assert list(bundle["required_bundle_array_sha256"]) == _locked_spec()[
        "required_bundle_arrays"
    ]

    restored = _FakeBridge("cupy")
    verification = transplant._restore_bundle(restored, bundle)
    assert verification["all_exact"]
    assert all(verification["cp_array_checks"].values())
    assert all(verification["csr_checks"].values())

    restored.cp_unsealed_dynamic = np.ones(4, np.float32)
    with pytest.raises(ValueError, match="cp-array set differs"):
        transplant._restore_bundle(restored, bundle)

    sparse_state = _FakeBridge("numpy")
    sparse_state.cp_other_sparse = csr_matrix(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="outside the sealed CSR contract"):
        transplant._allocated_cp_arrays(sparse_state)

    with pytest.raises(FileExistsError):
        transplant.create_bundle(spec_path, digest, "numpy", bundle_path)


def test_run_binds_modes_and_records_required_audits(tmp_path, fake_runtime):
    spec_path, digest, bundle_path, _ = _bundle(tmp_path, fake_runtime, "numpy")
    for mode, expected_flag in (
        ("default", False), ("deterministic_transpose_matvec", True),
    ):
        _, run = _run(
            tmp_path, spec_path, digest, bundle_path, "numpy", "numpy", mode
        )
        assert run["mode"] == mode
        assert run["deterministic_transpose_matvec_requested"] is expected_flag
        assert run["deterministic_transpose_matvec_readback"] is expected_flag
        assert run["pre_step_restore_verification"]["all_exact"]
        assert run["initialization_disclosure"]["initialization_may_have_used_rng"]
        assert not run["initialization_disclosure"]["claim_of_no_rng_call"]
        assert run["initialization_disclosure"]["no_resampled_array_state_survived_restore"]
        assert run["runtime_config_contract"]["runtime_random_processes_disabled"]
        assert run["runtime_config_contract"]["contract_valid"]
        assert set(run["trajectories"]) == set(transplant.TRAJECTORIES)
        assert transplant._decode_array(run["trajectories"]["g_e"], "g_e").shape == (1200, 60)
        assert run["source_spike_counts_by_phase"] == {
            "baseline": 0, "inhibition": 4000, "release": 0,
        }
        assert run["target_rates_hz_by_phase"] == {
            "baseline": 100.0,
            "inhibition": 10.0,
            "release": 100.0,
            "suppression_ratio": 0.1,
        }
        assert run["initial_weight_sha256"] == run["final_weight_sha256"]
        assert run["initial_intrinsic_sha256"] == run["final_intrinsic_sha256"]
        assert run["validations"]["target_external_current_zero"]
        assert run["instrument_valid"]


def _mutate_trajectory(run, name, step, neuron, delta):
    changed = copy.deepcopy(run)
    array = transplant._decode_array(changed["trajectories"][name], name)
    array[step, neuron] += np.asarray(delta, dtype=array.dtype)
    changed["trajectories"][name] = transplant._encode_array(array)
    changed["trajectory_sha256"][name] = changed["trajectories"][name]["sha256"]
    return transplant._seal_artifact(changed)


def test_pair_comparison_reports_byte_and_tolerance_divergence(tmp_path, fake_runtime):
    spec_path, digest, bundle_path, _ = _bundle(tmp_path, fake_runtime, "numpy")
    numpy_path, numpy_run = _run(
        tmp_path, spec_path, digest, bundle_path, "numpy", "numpy", "default"
    )
    cupy_path, cupy_run = _run(
        tmp_path, spec_path, digest, bundle_path, "numpy", "cupy", "default"
    )
    exact = transplant.compare_runs(
        spec_path, digest, numpy_path, cupy_path, tmp_path / "compare-exact.json"
    )
    assert exact["first_byte_exact_divergence"] is None
    assert exact["first_tolerance_divergence"] is None
    assert exact["spikes_exact"]

    within = _mutate_trajectory(cupy_run, "g_e", 7, 2, 5e-7)
    within_path = tmp_path / "cupy-within.json"
    _write_artifact(within_path, within)
    comparison = transplant.compare_runs(
        spec_path, digest, numpy_path, within_path, tmp_path / "compare-within.json"
    )
    assert comparison["first_byte_exact_divergence"] == {
        "trajectory": "g_e", "step": 7, "neuron_indices": [2],
        "differing_neuron_count": 1,
    }
    assert comparison["first_tolerance_divergence"] is None

    outside = _mutate_trajectory(cupy_run, "g_e", 9, 1, 2e-6)
    outside_path = tmp_path / "cupy-outside.json"
    _write_artifact(outside_path, outside)
    comparison = transplant.compare_runs(
        spec_path, digest, numpy_path, outside_path, tmp_path / "compare-outside.json"
    )
    assert comparison["first_tolerance_divergence"] == {
        "trajectory": "g_e", "step": 9, "neuron_indices": [1],
        "differing_neuron_count": 1,
    }

    spike_changed = copy.deepcopy(cupy_run)
    spikes = transplant._decode_array(spike_changed["trajectories"]["spikes"], "spikes")
    spikes[11, 2] = ~spikes[11, 2]
    spike_changed["trajectories"]["spikes"] = transplant._encode_array(spikes)
    spike_changed["trajectory_sha256"]["spikes"] = spike_changed["trajectories"][
        "spikes"
    ]["sha256"]
    spike_changed = transplant._seal_artifact(spike_changed)
    spike_comparison = transplant._pair_comparison(
        numpy_run, spike_changed, transplant._comparison_tolerance(_locked_spec())
    )
    assert not spike_comparison["spikes_exact"]
    assert spike_comparison["first_tolerance_divergence_by_trajectory"]["spikes"] is None

    wrong_mode = copy.deepcopy(cupy_run)
    wrong_mode["mode"] = "deterministic_transpose_matvec"
    wrong_mode["deterministic_transpose_matvec_requested"] = True
    wrong_mode["deterministic_transpose_matvec_readback"] = True
    wrong_mode = transplant._seal_artifact(wrong_mode)
    with pytest.raises(ValueError, match="same sealed bundle and mode"):
        transplant._pair_comparison(
            numpy_run, wrong_mode, transplant._comparison_tolerance(_locked_spec())
        )


def test_aggregate_requires_complete_unique_eight_cell_matrix(tmp_path, fake_runtime):
    spec_path, digest, _ = _write_spec(tmp_path)
    bundle_paths, bundles = [], {}
    run_paths = []
    for origin in transplant.BACKENDS:
        bundle_path = tmp_path / f"bundle-{origin}.json"
        bundles[origin] = transplant.create_bundle(
            spec_path, digest, origin, bundle_path
        )
        bundle_paths.append(bundle_path)
        for backend in transplant.BACKENDS:
            for mode in transplant.MODES:
                path, _ = _run(
                    tmp_path, spec_path, digest, bundle_path, origin, backend, mode
                )
                run_paths.append(path)

    aggregate = transplant.aggregate_matrix(
        spec_path, digest, bundle_paths, run_paths, tmp_path / "aggregate.json"
    )
    assert aggregate["matrix_complete"]
    assert aggregate["matrix_cell_count"] == 8
    assert len(aggregate["within_origin_mode_comparisons"]) == 4
    assert aggregate["origin_bundle_comparison"]["topology_exact"]
    assert not aggregate["origin_bundle_comparison"]["parameter_arrays"]["cp_izh_C"][
        "byte_exact"
    ]

    with pytest.raises(ValueError, match="exactly 8"):
        transplant.aggregate_matrix(
            spec_path, digest, bundle_paths, run_paths[:-1], tmp_path / "missing.json"
        )
    duplicate_paths = run_paths[:-1] + [run_paths[0]]
    with pytest.raises(ValueError, match="duplicate aggregate matrix cell"):
        transplant.aggregate_matrix(
            spec_path, digest, bundle_paths, duplicate_paths, tmp_path / "duplicate.json"
        )

    mismatched = json.loads(run_paths[0].read_text())
    mismatched["bundle_artifact_sha256"] = bundles["cupy"]["artifact_sha256"]
    mismatched = transplant._seal_artifact(mismatched)
    mismatched_path = tmp_path / "mismatched-run.json"
    _write_artifact(mismatched_path, mismatched)
    mismatched_paths = [mismatched_path, *run_paths[1:]]
    with pytest.raises(ValueError, match="wrong origin bundle"):
        transplant.aggregate_matrix(
            spec_path, digest, bundle_paths, mismatched_paths, tmp_path / "mismatch.json"
        )

    mismatch_cases = (
        ("seed", 8_888_888),
        ("spec_sha256", "0" * 64),
        ("source_identity", {**transplant._source_identity(), "sim/bridge.py": "0" * 64}),
    )
    original = json.loads(run_paths[0].read_text())
    for index, (field, value) in enumerate(mismatch_cases):
        changed = copy.deepcopy(original)
        changed[field] = value
        changed = transplant._seal_artifact(changed)
        changed_path = tmp_path / f"mismatch-{field}.json"
        _write_artifact(changed_path, changed)
        paths = [changed_path, *run_paths[1:]]
        with pytest.raises(ValueError, match="run contract mismatch"):
            transplant.aggregate_matrix(
                spec_path, digest, bundle_paths, paths,
                tmp_path / f"mismatch-contract-{index}.json",
            )


def test_tampered_bundle_and_create_only_outputs_fail_closed(tmp_path, fake_runtime):
    spec_path, digest, bundle_path, bundle = _bundle(tmp_path, fake_runtime, "numpy")
    tampered = copy.deepcopy(bundle)
    tampered["cp_arrays"]["cp_izh_C"]["data_base64"] = "AAAA"
    tampered_path = tmp_path / "tampered.json"
    _write_artifact(tampered_path, tampered)
    with pytest.raises(ValueError, match="bundle artifact digest mismatch"):
        transplant.execute_bundle(
            spec_path, digest, tampered_path, "numpy", "default",
            tmp_path / "must-not-run.json",
        )
    assert not (tmp_path / "must-not-run.json").exists()

    output = tmp_path / "existing.json"
    output.write_text("keep\n")
    built_before = len(fake_runtime["built"])
    with pytest.raises(FileExistsError):
        transplant.execute_bundle(
            spec_path, digest, bundle_path, "numpy", "default", output
        )
    assert output.read_text() == "keep\n"
    assert len(fake_runtime["built"]) == built_before
