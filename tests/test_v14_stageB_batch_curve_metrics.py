from __future__ import annotations

import copy

import numpy as np
import pytest

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from sim import source_model_candidate_batch as candidate_batch
from tools import v14_stageB_batch_curve_metrics as metrics
from tools import v14_stageB_fast_channel_clamp_analysis as authority


# This is an implementation-equivalence allowance, not a biological scoring
# threshold.  It is substantially tighter than the pixel-bounded standard
# errors in the Stage B population deactivation panels: at most 0.5% plus
# 0.0005 ms, while preserving resolution for the fastest observed tails.
AUTHORITY_RTOL = 5.0e-3
AUTHORITY_ATOL_MS = 5.0e-4


def _authority_taus(time: np.ndarray, traces: np.ndarray) -> np.ndarray:
    commands = list(range(traces.shape[1]))
    values: list[list[float]] = []
    for candidate in traces:
        assay = {
            "command_voltage_mv": commands,
            "elapsed_ms": time.tolist(),
            "tail": candidate.tolist(),
        }
        values.append([authority._fit_tail(assay, "tail", command) for command in commands])
    return np.asarray(values, dtype=np.float64)


def _source_parameters(model_id: str, module, count: int) -> list[dict]:
    documents: list[dict] = []
    for index in range(count):
        document = dict(module.source_parameters(model_id))
        if model_id == candidate_batch.KHALIQ_RAMAN_13_STATE:
            document["alpha_per_ms"] = 120.0 + 15.0 * index
            document["x6_mv"] = -22.0 - index
        elif model_id == candidate_batch.BALBI_NAV16_SIX_STATE:
            document["c1c2"] = (11.0 + index, -6.0 + index, -8.0 - index)
            document["q10"] = 2.5 + 0.25 * index
        elif model_id == candidate_batch.LABRO_2015:
            document["vhalf_mv"] = 2.0 + 2.0 * index
            document["alpha0_per_ms"] = (0.04 + 0.01 * index, 5.5 + index, 0.8 + 0.1 * index)
        else:
            document["k_alpha_per_ms"] = (0.035 + 0.005 * index, 0.00004 + 0.00001 * index)
            document["eta_beta_per_mv"] = (0.005 + 0.0005 * index, 0.009 + 0.00025 * index)
        documents.append(document)
    return documents


def _source_deactivation_traces(
    model_id: str, module, temperatures: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    parameters = _source_parameters(model_id, module, 3)
    if module is sodium:
        commands = np.array([-100.0, -70.0, -40.0, -20.0])
        state = candidate_batch.equilibrium_batch(model_id, parameters, -90.0, temperatures, np)
        state = candidate_batch.advance_batch(model_id, parameters, -120.0, state, 50.0, temperatures, np)
        state = candidate_batch.advance_batch(model_id, parameters, 0.0, state, 0.2, temperatures, np)
        reversal = 50.0
    else:
        commands = np.array([-70.0, -50.0, -30.0])
        state = candidate_batch.equilibrium_batch(model_id, parameters, -90.0, temperatures, np)
        state = candidate_batch.advance_batch(model_id, parameters, 20.0, state, 100.0, temperatures, np)
        reversal = -90.0

    # A complete 50 ms clamp tail, including the sample the filed estimator
    # discards, exercises the source kinetics rather than a synthetic proxy.
    time = np.arange(0.0, 50.0001, 0.005, dtype=np.float64)
    states = candidate_batch.trace_batch(model_id, parameters, commands, state, time, temperatures, np)
    traces = candidate_batch.open_probability_batch(model_id, states, np)
    return time, traces * (commands[None, :, None] - reversal)


def test_single_exponential_tails_match_scipy_authority_and_skip_sample_zero():
    time = np.arange(0.0, 5.005, 0.005, dtype=np.float64)
    tau = np.array([[0.027, 0.11, 1.2], [0.045, 0.35, 4.5]])
    asymptote = np.array([[0.2, -0.1, 0.7], [-0.3, 0.4, -0.5]])
    amplitude = np.array([[1.3, -0.9, 0.6], [0.4, 1.8, -1.1]])
    traces = asymptote[..., None] + amplitude[..., None] * np.exp(-time / tau[..., None])
    traces[..., 0] = 9_999.0  # The filed fit deliberately discards this sample.

    actual = metrics.fit_deactivation_tails(time, traces, np)
    expected = _authority_taus(time, traces)

    assert actual.shape == (2, 3)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=AUTHORITY_RTOL, atol=AUTHORITY_ATOL_MS)
    np.testing.assert_allclose(actual, tau, rtol=AUTHORITY_RTOL, atol=AUTHORITY_ATOL_MS)


def test_multi_exponential_tails_match_scipy_authority_within_prespecified_budget():
    time = np.arange(0.0, 1.005, 0.005, dtype=np.float64)
    fast_amplitude = np.array([[1.0, 0.2], [0.8, 1.1]])
    fast_tau = np.array([[0.03, 0.02], [0.1, 0.5]])
    slow_amplitude = np.array([[0.2, 1.0], [0.4, -0.2]])
    slow_tau = np.array([[0.3, 0.5], [0.8, 0.04]])
    traces = (
        0.11
        + fast_amplitude[..., None] * np.exp(-time / fast_tau[..., None])
        + slow_amplitude[..., None] * np.exp(-time / slow_tau[..., None])
    )

    actual = metrics.fit_deactivation_tails(time, traces, np)
    expected = _authority_taus(time, traces)

    np.testing.assert_allclose(actual, expected, rtol=AUTHORITY_RTOL, atol=AUTHORITY_ATOL_MS)


@pytest.mark.parametrize(
    "model_id,module,temperatures",
    [
        (candidate_batch.KHALIQ_RAMAN_13_STATE, sodium, None),
        (candidate_batch.BALBI_NAV16_SIX_STATE, sodium, np.array([20.0, 24.0, 31.0])),
        (candidate_batch.LABRO_2015, kv3, np.array([20.0, 22.5, 25.0])),
        (candidate_batch.DESAI_2008_CONTROL, kv3, None),
    ],
)
def test_actual_varied_source_model_deactivation_tails_match_scipy_authority(
    model_id, module, temperatures
):
    time, traces = _source_deactivation_traces(model_id, module, temperatures)

    actual = metrics.fit_deactivation_tails(time, traces, np)
    expected = _authority_taus(time, traces)

    np.testing.assert_allclose(actual, expected, rtol=AUTHORITY_RTOL, atol=AUTHORITY_ATOL_MS)


def test_invalid_or_unidentifiable_tail_inputs_fail_closed():
    time = np.array([0.0, 0.1, 0.2, 0.3])
    valid = np.array([[[1.0, 0.7, 0.5, 0.4]]])

    with pytest.raises(ValueError, match="at least four"):
        metrics.fit_deactivation_tails(time[:3], valid[..., :3], np)
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.fit_deactivation_tails(np.array([0.0, 0.1, 0.1, 0.3]), valid, np)
    with pytest.raises(ValueError, match="nonnegative"):
        metrics.fit_deactivation_tails(np.array([-0.1, 0.1, 0.2, 0.3]), valid, np)
    with pytest.raises(ValueError, match="time axis"):
        metrics.fit_deactivation_tails(time, valid[..., :-1], np)
    with pytest.raises(ValueError, match="finite"):
        metrics.fit_deactivation_tails(time, np.array([[[1.0, np.nan, 0.5, 0.4]]]), np)
    with pytest.raises(ValueError, match="unidentifiable"):
        metrics.fit_deactivation_tails(time, np.ones((1, 1, 4)), np)


def test_cpu_result_is_deterministic():
    time = np.arange(0.0, 3.005, 0.005, dtype=np.float64)
    tau = np.array([[0.04, 0.3], [0.8, 2.0]])
    traces = 0.1 + np.exp(-time / tau[..., None])

    first = metrics.fit_deactivation_tails(time, traces, np)
    second = metrics.fit_deactivation_tails(time.copy(), copy.deepcopy(traces), np)

    np.testing.assert_array_equal(first, second)


def test_optional_cupy_cpu_parity_is_deterministic():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")

    time, traces = _source_deactivation_traces(
        candidate_batch.BALBI_NAV16_SIX_STATE, sodium, np.array([20.0, 24.0, 31.0])
    )
    expected = metrics.fit_deactivation_tails(time, traces, np)
    first = metrics.fit_deactivation_tails(cupy.asarray(time), cupy.asarray(traces), cupy)
    second = metrics.fit_deactivation_tails(cupy.asarray(time), cupy.asarray(traces), cupy)

    # The nonlinear profile selection amplifies backend reduction-order noise
    # slightly beyond the source-engine's pointwise state contract.  This is
    # still more than four orders below the estimator agreement budget above.
    np.testing.assert_allclose(cupy.asnumpy(first), expected, rtol=1e-7, atol=5e-10)
    np.testing.assert_array_equal(cupy.asnumpy(first), cupy.asnumpy(second))
