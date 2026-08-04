"""Focused tests for the default-off Izhikevich arithmetic correction.

The tests use sealed no-seed localizer inputs or fixed toy arrays. They do not
construct or execute a preregistered scientific experiment.
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._v13_izh_arithmetic_localizer import (  # noqa: E402
    differing_cells,
    load_evidence,
    probe_inputs,
    strict_izhikevich2007_update as reference_update,
)
from sim import kernels  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.config import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    RuntimeState,
    VisualizationConfig,
)
from sim.enums import NeuronModel  # noqa: E402


xp, BACKEND = get_backend()


def _toy_inputs(dtype=None):
    dtype = xp.float32 if dtype is None else dtype
    values = (
        [-61.25, -48.5, -72.0, -55.125],
        [-2.25, 0.75, -5.5, 3.125],
        [60.0, 100.0, 0.0, 80.0],
        [0.7, 1.0, 0.5, 1.2],
        [-60.0, -55.0, -65.0, -62.0],
        [-40.0, -42.0, -38.0, -45.0],
        [0.03, 0.1, 0.02, 0.08],
        [-2.0, 0.2, -1.5, 0.25],
        [14.25, -3.5, 8.0, 1.125],
    )
    return tuple(xp.asarray(value, dtype=dtype) for value in values) + (1.0,)


def _host(value):
    return np.ascontiguousarray(to_host(value), dtype=np.float32)


def test_config_defaults_to_legacy_arithmetic():
    assert CoreSimConfig().backend_neutral_izh_arithmetic is False


def test_disabled_dispatch_calls_the_existing_fused_function_byte_for_byte():
    inputs = _toy_inputs()
    expected = kernels.fused_izhikevich2007_dynamics_update(*inputs)
    observed = kernels.izhikevich2007_dynamics_update(
        *inputs, backend_neutral_arithmetic=False
    )
    for left, right in zip(observed, expected):
        assert np.array_equal(_host(left).view(np.uint32), _host(right).view(np.uint32))


@pytest.mark.parametrize(
    ("variable", "input_row", "output_row", "output_index"),
    (("u", 1, 2, 1), ("v", 9, 10, 0)),
)
def test_strict_production_update_matches_sealed_numpy_bytes(
    variable, input_row, output_row, output_index
):
    evidence = load_evidence()
    inputs = probe_inputs(xp, evidence, input_row)
    observed = kernels.strict_izhikevich2007_dynamics_update(*inputs)[output_index]
    assert differing_cells(observed, evidence["numpy"][variable][output_row]) == []


def test_strict_outputs_remain_float32_and_c_contiguous():
    for output in kernels.strict_izhikevich2007_dynamics_update(*_toy_inputs()):
        assert output.dtype == xp.dtype(xp.float32)
        assert output.flags.c_contiguous


def test_strict_toy_update_matches_staged_numpy_bytes_including_zero_C():
    inputs = _toy_inputs()
    host_inputs = tuple(_host(value) for value in inputs[:-1]) + (np.float32(1.0),)
    expected = reference_update(np, *host_inputs)
    observed = kernels.strict_izhikevich2007_dynamics_update(*inputs)
    for left, right in zip(observed, expected):
        assert np.array_equal(_host(left).view(np.uint32), right.view(np.uint32))


@pytest.mark.parametrize("input_index", range(9))
def test_strict_update_rejects_every_non_float32_array(input_index):
    inputs = list(_toy_inputs())
    inputs[input_index] = inputs[input_index].astype(xp.float64)
    with pytest.raises(TypeError, match="must have dtype float32"):
        kernels.strict_izhikevich2007_dynamics_update(*inputs)


def test_strict_update_rejects_noncontiguous_arrays():
    inputs = list(_toy_inputs())
    inputs[0] = xp.arange(8, dtype=xp.float32)[::2]
    assert not inputs[0].flags.c_contiguous
    with pytest.raises(ValueError, match="v must be C-contiguous"):
        kernels.strict_izhikevich2007_dynamics_update(*inputs)


def test_strict_update_rejects_shape_broadcasting():
    inputs = list(_toy_inputs())
    inputs[4] = xp.asarray([-60.0], dtype=xp.float32)
    with pytest.raises(ValueError, match="vr_param must have shape"):
        kernels.strict_izhikevich2007_dynamics_update(*inputs)


@pytest.mark.parametrize("dt", ["1.0", None, [1.0]])
def test_strict_update_rejects_nonscalar_dt(dt):
    inputs = _toy_inputs()[:-1] + (dt,)
    with pytest.raises(TypeError, match="dt must be a real scalar"):
        kernels.strict_izhikevich2007_dynamics_update(*inputs)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), float("-inf")])
def test_strict_update_rejects_nonfinite_dt(dt):
    inputs = _toy_inputs()[:-1] + (dt,)
    with pytest.raises(ValueError, match="dt must be finite"):
        kernels.strict_izhikevich2007_dynamics_update(*inputs)


def test_dispatch_rejects_nonboolean_flag_before_computation(monkeypatch):
    monkeypatch.setattr(
        kernels,
        "fused_izhikevich2007_dynamics_update",
        lambda *args: pytest.fail("legacy kernel must not run"),
    )
    monkeypatch.setattr(
        kernels,
        "strict_izhikevich2007_dynamics_update",
        lambda *args: pytest.fail("strict kernel must not run"),
    )
    with pytest.raises(TypeError, match="backend_neutral_arithmetic must be a boolean"):
        kernels.izhikevich2007_dynamics_update(
            *_toy_inputs(), backend_neutral_arithmetic=1
        )


@pytest.mark.parametrize(
    ("flag", "model", "message"),
    (
        ("enabled", NeuronModel.IZHIKEVICH.name, "must be a boolean"),
        (True, NeuronModel.HODGKIN_HUXLEY.name, "supports only IZHIKEVICH"),
    ),
)
def test_bridge_rejects_invalid_arithmetic_configuration_before_population_state(
    flag, model, message, capsys
):
    cfg = CoreSimConfig(
        num_neurons=4,
        connections_per_neuron=0,
        seed=77,
        dt_ms=0.05 if model == NeuronModel.HODGKIN_HUXLEY.name else 1.0,
        neuron_model_type=model,
    )
    cfg.backend_neutral_izh_arithmetic = flag
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()

    assert bridge.is_initialized is False
    assert bridge.cp_traits is None
    assert bridge.cp_neuron_positions_3d is None
    assert bridge.cp_connections is None
    assert message in capsys.readouterr().out


def test_bridge_strict_step_receives_float32_c_contiguous_runtime_arrays(monkeypatch):
    bridge_module = importlib.import_module("sim.bridge")
    original = bridge_module.izhikevich2007_dynamics_update
    observed_calls = []

    def checked_dispatch(*args, **kwargs):
        runtime_arrays = args[:9]
        assert kwargs["backend_neutral_arithmetic"] is True
        assert all(value.dtype == xp.dtype(xp.float32) for value in runtime_arrays)
        assert all(value.flags.c_contiguous for value in runtime_arrays)
        assert len({value.shape for value in runtime_arrays}) == 1
        observed_calls.append(runtime_arrays[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        bridge_module, "izhikevich2007_dynamics_update", checked_dispatch
    )
    cfg = CoreSimConfig(
        num_neurons=12,
        connections_per_neuron=0,
        seed=24_681_357,
        neuron_model_type=NeuronModel.IZHIKEVICH.name,
        backend_neutral_izh_arithmetic=True,
        enable_ou_process=False,
        enable_conductance_noise=False,
        enable_hebbian_learning=False,
        enable_short_term_plasticity=False,
        enable_homeostasis=False,
        enable_stdp=False,
        enable_inhibitory_stdp=False,
        enable_structural_plasticity=False,
        enable_reward_modulation=False,
    )
    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(enable_profiling=False),
    )
    bridge._initialize_simulation_data()
    assert bridge.is_initialized
    bridge.strict_step_errors = True
    bridge._run_one_simulation_step()
    assert observed_calls == [(12,)]


def test_gpu_path_is_one_explicit_rounding_kernel_without_host_conversion():
    source = Path(kernels.__file__).read_text()
    assert source.count("cp.ElementwiseKernel(") == 1
    for intrinsic in ("__fsub_rn", "__fmul_rn", "__fadd_rn", "__fdiv_rn"):
        assert intrinsic in source

    strict_source = inspect.getsource(kernels.strict_izhikevich2007_dynamics_update)
    for forbidden in (".get(", "asnumpy", "to_host", "asarray"):
        assert forbidden not in strict_source

    dispatch_guard = inspect.getsource(SimulationBridge._step_megakernel_can_dispatch)
    assert 'getattr(cfg, "backend_neutral_izh_arithmetic", False)' in dispatch_guard


@pytest.mark.skipif(BACKEND != "cupy", reason="requires the CuPy backend")
def test_gpu_strict_dispatch_invokes_one_device_kernel(monkeypatch):
    calls = []

    def fake_kernel(*args):
        calls.append(args)
        return xp.empty_like(args[0]), xp.empty_like(args[1])

    monkeypatch.setattr(kernels, "_strict_izhikevich2007_gpu_kernel", fake_kernel)
    kernels.strict_izhikevich2007_dynamics_update(*_toy_inputs())
    assert len(calls) == 1
