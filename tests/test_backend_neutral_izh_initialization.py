"""Focused guards for the opt-in backend-neutral Izhikevich initializer.

These tests initialize toy populations only. They deliberately do not use or
execute the preregistered correction-diagnostic seed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sim.config import CoreSimConfig


ROOT = Path(__file__).resolve().parents[1]
TEST_SEED = 424_242
PREREGISTERED_DIAGNOSTIC_SEED = 6_556_023
DIAGNOSTIC_OUTPUT = (
    ROOT / "research/findings/raw/"
    "v13_backend_neutral_izh_initialization_diagnostic"
)

ARRAYS = (
    "cp_traits",
    "cp_neuron_type_ids",
    "cp_neuron_firing_thresholds",
    "cp_neuron_positions_3d",
    "cp_izh_C",
    "cp_izh_k",
    "cp_izh_vr",
    "cp_izh_vt",
    "cp_izh_vpeak",
    "cp_izh_a",
    "cp_izh_b",
    "cp_izh_c_reset",
    "cp_izh_d_increment",
    "cp_membrane_potential_v",
    "cp_recovery_variable_u",
)

_PROBE = r"""
import hashlib
import json
import os
import numpy as np

from sim.backend import to_host
from sim.bridge import SimulationBridge
from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
from sim.enums import NeuronModel, NeuronType
from sim.regions import BrainRegion, RegionPathway

rejection_case = os.environ.get("TEST_REJECTION_CASE", "")
seed = int(os.environ["TEST_INITIALIZATION_SEED"])
if rejection_case == "invalid_seed_range":
    seed = 2**32
enabled = os.environ["TEST_NEUTRAL_INITIALIZATION"] == "1"
heterogeneity = {
    "izh_C_val": {"type": "gaussian", "mean": 60.0, "std": 7.5},
    "izh_a_val": {"type": "lognormal", "mean_log": -3.0, "sigma_log": 0.2},
    "izh_b_val": {"type": "gaussian", "mean": 2.0, "std": 0.4},
    "izh_d_val": {"type": "lognormal", "mean_log": 3.2, "sigma_log": 0.2},
}
if rejection_case == "unknown_distribution":
    heterogeneity["izh_a_val"] = {"type": "not-a-distribution"}
elif rejection_case == "unavailable_target":
    heterogeneity = {
        "hh_C_m": {"type": "gaussian", "mean": 1.0, "std": 0.1},
    }
elif rejection_case == "malformed_distribution":
    heterogeneity = {"izh_C_val": ["not", "a", "mapping"]}
elif rejection_case == "nonfinite_parameters":
    heterogeneity = {
        "izh_C_val": {"type": "gaussian", "mean": float("nan"), "std": 1.0},
    }
elif rejection_case == "nonfinite_output":
    heterogeneity = {
        "izh_C_val": {"type": "lognormal", "mean_log": 1000.0, "sigma_log": 0.0},
    }

regions = [
    BrainRegion(
        name="source", n_neurons=20, exc_fraction=0.0,
        internal_density=0.1,
        izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
        enable_heterogeneity=False,
    ),
    BrainRegion(
        name="target", n_neurons=40, exc_fraction=0.0,
        internal_density=0.1,
        izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name,
        intrinsic_current_pA=(
            0.0 if rejection_case == "non_izhikevich_model" else 100.0
        ),
        enable_heterogeneity=True,
    ),
]
cfg = CoreSimConfig(
    num_neurons=60,
    num_traits=5,
    seed=seed,
    heterogeneity_seed=(
        2**32 if rejection_case == "invalid_heterogeneity_seed_range" else seed
    ),
    neuron_model_type=NeuronModel.IZHIKEVICH.name,
    enable_brain_region_framework=True,
    brain_regions=regions,
    region_pathways=[
        RegionPathway(
            from_region="source", to_region="target", density=1.0,
            weight_mean=8.0, weight_jitter=0.0, plastic=False,
        )
    ],
    enable_parameter_heterogeneity=False,
    heterogeneity_distributions=heterogeneity,
    enable_ou_process=False,
    enable_conductance_noise=False,
    enable_hebbian_learning=False,
    enable_short_term_plasticity=False,
    enable_homeostasis=False,
    enable_stdp=False,
    enable_inhibitory_stdp=False,
    enable_structural_plasticity=False,
    backend_neutral_izh_initialization=enabled,
)
if rejection_case == "non_boolean_flag":
    cfg.backend_neutral_izh_initialization = "enabled"
elif rejection_case == "non_izhikevich_model":
    cfg.neuron_model_type = NeuronModel.HODGKIN_HUXLEY.name
bridge = SimulationBridge(
    core_config=cfg,
    viz_config=VisualizationConfig(),
    runtime_state=RuntimeState(),
    gpu_config=GPUConfig(enable_profiling=False),
)
bridge._initialize_simulation_data()
if not bridge.is_initialized:
    state = {
        "actual_seed_used": bridge.runtime_state.actual_seed_used,
        "traits_allocated": bridge.cp_traits is not None,
        "izh_arrays_allocated": bridge.cp_izh_C is not None,
        "positions_allocated": bridge.cp_neuron_positions_3d is not None,
        "connections_allocated": bridge.cp_connections is not None,
        "simulation_steps_executed": 0,
    }
    print("INITIALIZATION_RESULT:" + json.dumps({
        "initialized": False,
        "rejection_case": rejection_case,
        "state": state,
    }, sort_keys=True))
    raise SystemExit(0)

names = json.loads(os.environ["TEST_INITIALIZATION_ARRAYS"])
result = {"initialized": True, "arrays": {}}
for name in names:
    value = np.ascontiguousarray(np.asarray(to_host(getattr(bridge, name))))
    result["arrays"][name] = {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }
print("INITIALIZATION_RESULT:" + json.dumps(result, sort_keys=True))
"""


def _has_usable_cupy() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        return False


def _run_probe(
    backend: str,
    *,
    enabled: bool,
    rejection_case: str = "",
    expected_error: str | None = None,
) -> dict:
    assert TEST_SEED != PREREGISTERED_DIAGNOSTIC_SEED
    env = os.environ.copy()
    env.update({
        "SIM_BACKEND": backend,
        "TEST_INITIALIZATION_SEED": str(TEST_SEED),
        "TEST_NEUTRAL_INITIALIZATION": "1" if enabled else "0",
        "TEST_INITIALIZATION_ARRAYS": json.dumps(ARRAYS),
        "TEST_REJECTION_CASE": rejection_case,
    })
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    marker = "INITIALIZATION_RESULT:"
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, completed.stdout + completed.stderr
    if expected_error is not None:
        process_output = completed.stdout + completed.stderr
        assert expected_error in process_output, process_output
    return json.loads(lines[0][len(marker):])


def test_backend_neutral_izh_initialization_defaults_off():
    assert CoreSimConfig().backend_neutral_izh_initialization is False


def test_opt_in_contract_matches_legacy_numpy_population_bytes():
    legacy = _run_probe("numpy", enabled=False)
    corrected = _run_probe("numpy", enabled=True)
    assert legacy["initialized"] and corrected["initialized"]
    assert corrected["arrays"] == legacy["arrays"]


@pytest.mark.skipif(not _has_usable_cupy(), reason="usable CuPy device unavailable")
def test_opt_in_population_is_byte_identical_on_numpy_and_cupy():
    numpy_result = _run_probe("numpy", enabled=True)
    cupy_result = _run_probe("cupy", enabled=True)
    assert numpy_result["initialized"] and cupy_result["initialized"]
    assert cupy_result["arrays"] == numpy_result["arrays"]


@pytest.mark.parametrize(
    ("rejection_case", "expected_error", "rejection_phase"),
    (
        (
            "non_boolean_flag",
            "backend_neutral_izh_initialization must be a boolean",
            "pre_population",
        ),
        (
            "non_izhikevich_model",
            "backend_neutral_izh_initialization supports only IZHIKEVICH",
            "pre_population",
        ),
        (
            "invalid_seed_range",
            "Seed must be between 0 and 2**32 - 1",
            "pre_population",
        ),
        (
            "invalid_heterogeneity_seed_range",
            "Izhikevich heterogeneity seed must be in [0, 2**32 - 1]",
            "heterogeneity",
        ),
        (
            "unavailable_target",
            "Backend-neutral heterogeneity target 'hh_C_m' is unavailable",
            "heterogeneity",
        ),
        (
            "unknown_distribution",
            "Unsupported backend-neutral heterogeneity distribution",
            "heterogeneity",
        ),
        (
            "malformed_distribution",
            "Heterogeneity distribution for izh_C_val must be a mapping",
            "heterogeneity",
        ),
        (
            "nonfinite_parameters",
            "Invalid gaussian parameters for izh_C_val",
            "heterogeneity",
        ),
        (
            "nonfinite_output",
            "Heterogeneity draw for izh_C_val produced non-finite values",
            "heterogeneity",
        ),
    ),
)
def test_opt_in_rejections_fail_closed_before_result_production(
    rejection_case: str,
    expected_error: str,
    rejection_phase: str,
):
    output_before = (
        sorted(path.relative_to(DIAGNOSTIC_OUTPUT) for path in DIAGNOSTIC_OUTPUT.rglob("*"))
        if DIAGNOSTIC_OUTPUT.exists() else []
    )
    result = _run_probe(
        "numpy",
        enabled=True,
        rejection_case=rejection_case,
        expected_error=expected_error,
    )
    output_after = (
        sorted(path.relative_to(DIAGNOSTIC_OUTPUT) for path in DIAGNOSTIC_OUTPUT.rglob("*"))
        if DIAGNOSTIC_OUTPUT.exists() else []
    )

    assert result["initialized"] is False
    assert result["rejection_case"] == rejection_case
    assert result["state"]["simulation_steps_executed"] == 0
    assert result["state"]["positions_allocated"] is False
    assert result["state"]["connections_allocated"] is False
    assert output_after == output_before

    population_allocated = result["state"]["izh_arrays_allocated"]
    if rejection_phase == "pre_population":
        assert population_allocated is False
        assert result["state"]["traits_allocated"] is False
    else:
        assert population_allocated is True
        assert result["state"]["traits_allocated"] is True
