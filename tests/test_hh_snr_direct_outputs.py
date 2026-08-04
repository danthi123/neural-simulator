"""Exact GPU tests for direct-output reuse of established SNr/HH fusion graphs."""

import os

os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np
import pytest

from sim.backend import get_backend
from sim.kernels import (
    _fusion_memo_key,
    fused_hodgkin_huxley_dynamics_update,
    fused_snr_conductance_update,
    fused_snr_conductance_update_into,
)


cp, _BACKEND_NAME = get_backend()
pytestmark = pytest.mark.skipif(_BACKEND_NAME != "cupy", reason="CuPy-only direct outputs")


def test_fusion_memo_key_does_not_materialize_python_scalars_on_device(monkeypatch):
    values = (0.05, 7, True)
    expected = tuple(
        item
        for value in values
        for item in (cp.dtype(type(value)).char, type(value))
    )

    def reject_scalar_transfer(value, *args, **kwargs):
        raise AssertionError(f"unexpected device conversion for {value!r}")

    monkeypatch.setattr(cp, "asarray", reject_scalar_transfer)

    assert _fusion_memo_key(values) == expected


def _fixture(size=257):
    rng = np.random.default_rng(20260804)
    array = lambda values: cp.asarray(values, dtype=cp.float32)
    state = [
        array(np.linspace(-88.0, 28.0, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.0, 1.5, size)),
        array(rng.uniform(0.01, 0.95, size)),
        array(rng.uniform(0.01, 0.95, size)),
    ]
    params = {
        "input": array(np.linspace(-0.025, 0.08, size)),
        "dt": cp.float32(0.05),
        "C": array(rng.uniform(0.8, 1.2, size)),
        "g_na": array(rng.uniform(105.0, 125.0, size)),
        "g_k": array(rng.uniform(31.0, 39.0, size)),
        "g_l": array(rng.uniform(0.25, 0.35, size)),
        "e_na": array(rng.uniform(48.0, 52.0, size)),
        "e_k": array(rng.uniform(-80.0, -74.0, size)),
        "e_l": array(rng.uniform(-56.0, -52.0, size)),
        "g_snr": [array(rng.uniform(0.001, 0.025, size)) for _ in range(5)],
    }
    return state, params


def _snr_inputs(state, p):
    return (
        state[0], *state[4:], p["dt"], *p["g_snr"],
        cp.float32(-10.0), cp.float32(50.0), cp.float32(120.0),
        cp.float32(-90.0), cp.float32(-30.0), cp.float32(0.0),
        cp.float32(0.01), cp.float32(50.0), cp.float32(0.5),
        cp.float32(4.0), cp.float32(5.0),
    )


def _hh_inputs(state, effective, p):
    return (
        *state[:4], effective, p["dt"], p["C"], p["g_na"], p["g_k"],
        p["g_l"], p["e_na"], p["e_k"], p["e_l"],
        cp.float32(1.0), cp.float32(1.0), cp.float32(1.0),
    )


def test_direct_outputs_cold_cache_falls_back_then_reuses_exact_cached_kernel():
    initial, params = _fixture(size=31)
    reference = [value.copy() for value in initial]
    candidate = [value.copy() for value in initial]
    scratch = cp.empty_like(candidate[0])
    destinations = (*candidate[4:], scratch)
    pointers = tuple(value.data.ptr for value in destinations)

    def production_inputs(state):
        inputs = list(_snr_inputs(state, params))
        for index in (8, *range(14, 25)):
            inputs[index] = float(inputs[index])
        return tuple(inputs)

    fused_snr_conductance_update.clear_cache()
    assert fused_snr_conductance_update_into(
        production_inputs(candidate), destinations
    ) is False
    expected = fused_snr_conductance_update(*production_inputs(reference))

    assert tuple(value.data.ptr for value in destinations) == pointers
    for expected_value, actual in zip(expected, destinations):
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected_value))

    reference[4:] = expected[:7]
    assert fused_snr_conductance_update_into(
        production_inputs(candidate), destinations
    ) is True
    expected = fused_snr_conductance_update(*production_inputs(reference))

    assert tuple(value.data.ptr for value in destinations) == pointers
    for expected_value, actual in zip(expected, destinations):
        np.testing.assert_array_equal(cp.asnumpy(actual), cp.asnumpy(expected_value))


def test_direct_outputs_remain_byte_identical_over_randomized_multistep_state():
    initial, params = _fixture()
    reference = [value.copy() for value in initial]
    candidate = [value.copy() for value in initial]
    scratch = cp.empty_like(candidate[0])
    pointers = tuple(value.data.ptr for value in candidate)

    for step in range(1, 65):
        snr = fused_snr_conductance_update(*_snr_inputs(reference, params))
        hh = fused_hodgkin_huxley_dynamics_update(
            *_hh_inputs(reference, params["input"] - snr[-1], params)
        )
        reference = [*hh, *snr[:7]]

        fused_snr_conductance_update_into(
            _snr_inputs(candidate, params), (*candidate[4:], scratch)
        )
        candidate_hh = fused_hodgkin_huxley_dynamics_update(
            *_hh_inputs(candidate, params["input"] - scratch, params)
        )
        for destination, value in zip(candidate[:4], candidate_hh):
            destination[:] = value

        assert tuple(value.data.ptr for value in candidate) == pointers
        for name, expected, actual in zip(
            ("V", "m", "h", "n", "nap_m", "nap_h", "ca_m", "ca_h", "calcium", "sk", "ih"),
            reference,
            candidate,
        ):
            np.testing.assert_array_equal(
                cp.asnumpy(actual), cp.asnumpy(expected), err_msg=f"{name} step={step}"
            )
