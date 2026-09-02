"""Focused regressions for deterministic excitatory/inhibitory sparse propagation."""

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

# Pin NumPy only when this module is the first simulator importer.  Changing the
# environment after sim.bridge has bound its module-level backend splits the
# process between the old module backend and the new backend registry.
if "sim.bridge" not in sys.modules:
    os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402
import scipy.sparse as scipy_sparse  # noqa: E402

import sim.bridge as bridge_module  # noqa: E402
from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.backend import get_backend, get_sparse_module, to_host  # noqa: E402


def _backend_csr(host_matrix):
    xp, _ = get_backend()
    sparse = get_sparse_module()
    return sparse.csr_matrix(
        (
            xp.asarray(host_matrix.data, dtype=xp.float32),
            xp.asarray(host_matrix.indices, dtype=xp.int32),
            xp.asarray(host_matrix.indptr, dtype=xp.int32),
        ),
        shape=host_matrix.shape,
    )


def _digest(array):
    host = np.ascontiguousarray(to_host(array))
    return hashlib.sha256(host.tobytes()).hexdigest()


@lru_cache(maxsize=1)
def _isolated_numpy_trajectory():
    """Run the frozen NumPy trajectory in a backend-clean child process."""
    test_path = Path(__file__).resolve()
    marker = "__NUMPY_TRAJECTORY__="
    code = f"""
import importlib.util
import json

spec = importlib.util.spec_from_file_location("isolated_sparse_matvec", {str(test_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

import sim.connectivity as connectivity
import sim.kernels as kernels

_, registry_backend = module.get_backend()
payload = {{
    "registry_backend": registry_backend,
    "bridge_backend": module.bridge_module._backend_name,
    "bridge_array_module": module.bridge_module.cp.__name__,
    "connectivity_array_module": connectivity.cp.__name__,
    "kernels_array_module": kernels.cp.__name__,
    "trajectory": module._run_trajectory(deterministic=False),
}}
print({marker!r} + json.dumps(payload, sort_keys=True))
"""
    env = os.environ.copy()
    env["SIM_BACKEND"] = "numpy"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=test_path.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "isolated NumPy trajectory failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, completed.stdout
    return json.loads(lines[0][len(marker):])


def _build_bridge(*, deterministic):
    cfg = CoreSimConfig(
        num_neurons=192,
        connections_per_neuron=64,
        seed=271828,
        dt_ms=1.0,
        deterministic_transpose_matvec=deterministic,
        enable_step_megakernel=False,
        enable_step_megakernel_v2=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_structural_plasticity=False,
        enable_reward_modulation=False,
        enable_ou_process=False,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    return bridge


def _run_trajectory(*, deterministic=True):
    bridge = _build_bridge(deterministic=deterministic)
    xp, _ = get_backend()
    raster = []
    for _ in range(180):
        bridge.cp_external_input_current[:] = xp.float32(210.0)
        bridge._run_one_simulation_step()
        raster.append(np.asarray(to_host(bridge.cp_firing_states), dtype=np.bool_))
    result = {
        "raster": _digest(np.stack(raster)),
        "v": _digest(bridge.cp_membrane_potential_v),
        "u": _digest(bridge.cp_recovery_variable_u),
        "g_e": _digest(bridge.cp_conductance_g_e),
        "g_i": _digest(bridge.cp_conductance_g_i),
        "weights": _digest(bridge.cp_connections.data),
        "external": _digest(bridge.cp_external_input_current),
    }
    bridge.clear_simulation_state_and_gpu_memory()
    return result


def test_split_primitive_executes_two_one_dimensional_spmvs(monkeypatch):
    # `_deterministic_ei_transpose_spmv` must (a) materialize the transpose CSR exactly once and
    # (b) run TWO SEPARATE ONE-DIMENSIONAL segmented SpMVs (E then I), never a single two-column
    # csrmv (the two-column cuSPARSE multiply is the non-deterministic one this whole module
    # exists to avoid). Since 2026-08-25 the primitive routes through the reduceat helper
    # `_deterministic_csr_matvec` (not a bare `@`), so the mock exposes a REAL transpose CSR and we
    # record the ndim each helper call reduces. (Repaired 2026-09-02: the prior FakeCSR mock still
    # expected a `@` and had regressed to an AttributeError on `.indptr`.)
    rng = np.random.default_rng(7)
    n = 32
    host_matrix = scipy_sparse.random(
        n, n, density=0.25, format="csr", dtype=np.float32, random_state=rng,
        data_rvs=lambda size: rng.uniform(0.1, 2.0, size).astype(np.float32),
    )
    real_transpose_csr = _backend_csr(host_matrix.T.tocsr())

    class FakeTranspose:
        def __init__(self):
            self.tocsr_calls = 0

        def tocsr(self):
            self.tocsr_calls += 1
            return real_transpose_csr

    class FakeConnections:
        def __init__(self):
            self.transpose = FakeTranspose()

        @property
        def T(self):
            return self.transpose

    connections = FakeConnections()
    xp, _ = get_backend()
    excitatory = xp.asarray(np.arange(n, dtype=np.float32))
    inhibitory = xp.asarray(np.arange(n, dtype=np.float32) + 10.0)

    reduced_ndims = []
    original = bridge_module._deterministic_csr_matvec

    def recording(csr_mat, vec):
        reduced_ndims.append(int(np.asarray(to_host(vec)).ndim))
        return original(csr_mat, vec)

    monkeypatch.setattr(bridge_module, "_deterministic_csr_matvec", recording)

    actual_e, actual_i = bridge_module._deterministic_ei_transpose_spmv(
        connections, excitatory, inhibitory
    )

    assert connections.transpose.tocsr_calls == 1
    assert reduced_ndims == [1, 1]  # two ONE-dimensional segmented SpMVs, not a two-column csrmv
    expected_e = host_matrix.T @ np.asarray(to_host(excitatory))
    expected_i = host_matrix.T @ np.asarray(to_host(inhibitory))
    assert np.allclose(to_host(actual_e), expected_e, rtol=2e-6, atol=2e-5)
    assert np.allclose(to_host(actual_i), expected_i, rtol=2e-6, atol=2e-5)


def test_repeated_split_spmvs_are_bit_exact():
    rng = np.random.default_rng(20260804)
    n = 2048
    fan_in = 64
    host_matrix = scipy_sparse.random(
        n,
        n,
        density=fan_in / n,
        format="csr",
        dtype=np.float32,
        random_state=rng,
        data_rvs=lambda size: rng.uniform(0.01, 8.0, size).astype(np.float32),
    )
    matrix = _backend_csr(host_matrix)
    xp, _ = get_backend()
    excitatory = xp.asarray(rng.uniform(0.0, 1.0, n), dtype=xp.float32)
    inhibitory = xp.asarray(rng.uniform(0.0, 1.0, n), dtype=xp.float32)

    hashes = []
    for _ in range(100):
        result_e, result_i = bridge_module._deterministic_ei_transpose_spmv(
            matrix, excitatory, inhibitory
        )
        hashes.append((_digest(result_e), _digest(result_i)))

    assert len(set(hashes)) == 1
    expected_e = host_matrix.T @ np.asarray(to_host(excitatory))
    expected_i = host_matrix.T @ np.asarray(to_host(inhibitory))
    assert np.allclose(to_host(result_e), expected_e, rtol=2e-6, atol=2e-5)
    assert np.allclose(to_host(result_i), expected_i, rtol=2e-6, atol=2e-5)


def test_normal_step_routes_only_deterministic_mode_through_split_primitive(monkeypatch):
    original = bridge_module._deterministic_ei_transpose_spmv
    calls = []

    def recording_split(connections, excitatory, inhibitory):
        calls.append((excitatory.ndim, inhibitory.ndim))
        return original(connections, excitatory, inhibitory)

    monkeypatch.setattr(
        bridge_module, "_deterministic_ei_transpose_spmv", recording_split
    )
    deterministic = _build_bridge(deterministic=True)
    deterministic.cp_prev_firing_states[::7] = True
    deterministic._run_one_simulation_step()
    deterministic.clear_simulation_state_and_gpu_memory()
    assert calls == [(1, 1)]

    calls.clear()
    default = _build_bridge(deterministic=False)
    default.cp_prev_firing_states[::7] = True
    default._run_one_simulation_step()
    default.clear_simulation_state_and_gpu_memory()
    assert calls == []


def test_deterministic_trajectory_repeats_exactly():
    first = _run_trajectory()
    second = _run_trajectory()
    assert first == second


def test_active_backend_registry_matches_bound_simulator_modules():
    import sim.connectivity as connectivity
    import sim.kernels as kernels

    _, registry_backend = get_backend()
    assert bridge_module._backend_name == registry_backend
    assert bridge_module.cp.__name__ == registry_backend
    assert connectivity.cp.__name__ == registry_backend
    assert kernels.cp.__name__ == registry_backend


def test_frozen_numpy_trajectory_uses_coherent_isolated_backend():
    result = _isolated_numpy_trajectory()
    assert result["registry_backend"] == "numpy"
    assert result["bridge_backend"] == "numpy"
    assert result["bridge_array_module"] == "numpy"
    assert result["connectivity_array_module"] == "numpy"
    assert result["kernels_array_module"] == "numpy"


def test_default_false_matches_frozen_pre_correction_numpy_trajectory():
    assert _isolated_numpy_trajectory()["trajectory"] == {
        "raster": "068826f850a3749a26e1cbb4afe490532cd9a389a4abd26e9280da10b96a772e",
        "v": "ad7081253888123fe22c38a062466863ed2fff8122647abb0f780c6628b9399d",
        "u": "05379f3507dda7c164614f45afacb72b5c9e32990c445cd94be57d98ab68ceef",
        "g_e": "28b68f3199cc62df4cc6cf8a2bebdecfa2e1a5207e229f42baf80d5e42690caf",
        "g_i": "1fa855572ea193508cbc894c5e6d225bf71913041d450d5aacf0bb948f458949",
        "weights": "028544ff47e7bfc354a430fe2d1c6abb97c0f6f08ffa07032a4b5974976bf000",
        "external": "8b221633a4dae65d9b80d3ccd1440cc14aae901fb5e74005a4cc22fe399db3c8",
    }
