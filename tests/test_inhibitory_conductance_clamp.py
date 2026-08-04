import inspect

import numpy as np
import pytest

from experiment.inhibitory_conductance_clamp import (
    BiexponentialInhibitoryEvent,
    EventSchedule,
    InhibitoryBarrage,
    MatchedMeanConductanceStep,
    whole_cell_current_pA,
    whole_cell_nS_to_density_mS_cm2,
)


def _event(**overrides):
    values = {
        "pathway": "pallidonigral",
        "tau_rise_ms": 0.4,
        "tau_decay_ms": 2.1,
        "reversal_mV": -70.0,
        "event_peak_nS": 2.0,
        "membrane_area_um2": 2000.0,
    }
    values.update(overrides)
    return BiexponentialInhibitoryEvent(**values)


def test_biexponential_is_normalized_to_requested_continuous_peak():
    event = _event(event_peak_nS=2.4)
    barrage = InhibitoryBarrage(event, EventSchedule.exact([10.0]))

    at_peak = barrage.conductance_nS(np.array([10.0 + event.peak_time_ms]), np)

    np.testing.assert_allclose(at_peak, [2.4], rtol=1e-13, atol=1e-13)
    assert event.peak_normalization > 0.0
    assert event.normalized_kernel_integral_ms > 0.0


def test_sample_grid_is_not_silently_renormalized():
    event = _event(event_peak_nS=2.4)
    barrage = InhibitoryBarrage(event, EventSchedule.exact([0.0]))
    sampled = barrage.conductance_nS(np.arange(0.0, 8.0, 1.0), np)

    assert sampled.max() < event.event_peak_nS
    assert sampled.max() > 0.0


def test_events_are_causal_and_superpose_exactly():
    event = _event(event_peak_nS=1.5)
    schedule = EventSchedule.exact([5.0, 5.0, 20.0])
    barrage = InhibitoryBarrage(event, schedule)
    first_peak = 5.0 + event.peak_time_ms

    before, at_first_peak = barrage.conductance_nS(
        np.array([4.999, first_peak]), np
    )

    assert before == 0.0
    assert at_first_peak == pytest.approx(3.0)
    assert barrage.pathway == "pallidonigral"


def test_whole_cell_density_and_current_units_are_explicit():
    # 2 nS / 2000 um2 = 0.1 mS/cm2.  nS * mV = pA.
    density = whole_cell_nS_to_density_mS_cm2(np.array([2.0]), 2000.0)
    current = whole_cell_current_pA(np.array([2.0]), np.array([-50.0]), -70.0)

    np.testing.assert_array_equal(density, [0.1])
    np.testing.assert_array_equal(current, [-40.0])


def test_barrage_exposes_density_and_positive_inward_current():
    event = _event(event_peak_nS=2.0)
    barrage = InhibitoryBarrage(event, EventSchedule.exact([10.0]))
    time = np.array([10.0 + event.peak_time_ms])

    np.testing.assert_allclose(barrage.conductance_density_mS_cm2(time, np), [0.1])
    np.testing.assert_allclose(barrage.current_pA(time, -50.0, np), [-40.0])


def test_seeded_poisson_schedule_is_exactly_reproducible_and_bounded():
    first = EventSchedule.seeded_poisson(90.0, 20.0, 2000.0, seed=713)
    second = EventSchedule.seeded_poisson(90.0, 20.0, 2000.0, seed=713)
    different = EventSchedule.seeded_poisson(90.0, 20.0, 2000.0, seed=714)

    assert first == second
    assert first.event_times_ms != different.event_times_ms
    assert first.event_times_ms
    assert all(20.0 <= time < 2000.0 for time in first.event_times_ms)
    assert first.kind == "seeded_poisson"
    assert first.rate_hz == 90.0


def test_matched_step_uses_stationary_poisson_expected_mean():
    event = _event(event_peak_nS=2.0)
    step = MatchedMeanConductanceStep.from_expected_barrage_mean(
        event, rate_hz=90.0, onset_ms=10.0, offset_ms=30.0
    )
    expected = 0.09 * event.event_peak_nS * event.normalized_kernel_integral_ms

    assert step.conductance_nS == pytest.approx(expected)
    assert step.pathway == event.pathway
    np.testing.assert_allclose(
        step.conductance_nS_trace(np.array([9.0, 10.0, 29.9, 30.0]), np),
        [0.0, expected, expected, 0.0],
    )
    np.testing.assert_allclose(
        step.conductance_density_mS_cm2(np.array([10.0]), np),
        [expected * 0.05],
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"pathway": ""}, "pathway"),
        ({"tau_rise_ms": 2.1}, "less than"),
        ({"tau_rise_ms": 0.0}, "greater than zero"),
        ({"tau_decay_ms": float("inf")}, "finite"),
        ({"event_peak_nS": 0.0}, "greater than zero"),
        ({"event_peak_nS": -1.0}, "greater than zero"),
        ({"membrane_area_um2": 0.0}, "greater than zero"),
        ({"reversal_mV": float("nan")}, "finite"),
    ],
)
def test_event_rejects_nonphysical_parameters(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _event(**overrides)


def test_schedule_rejects_invalid_times_and_poisson_parameters():
    with pytest.raises(ValueError, match="non-negative"):
        EventSchedule.exact([-0.1])
    with pytest.raises(ValueError, match="sorted"):
        EventSchedule.exact([2.0, 1.0])
    with pytest.raises(ValueError, match="greater than zero"):
        EventSchedule.seeded_poisson(0.0, 0.0, 10.0, seed=1)
    with pytest.raises(ValueError, match="start_ms < stop_ms"):
        EventSchedule.seeded_poisson(90.0, 10.0, 10.0, seed=1)


def test_waveform_implementation_has_no_host_synchronization_calls():
    source = inspect.getsource(InhibitoryBarrage) + inspect.getsource(
        MatchedMeanConductanceStep
    )
    assert ".get(" not in source
    assert "asnumpy" not in source
    assert "numpy.asarray" not in source


def test_cupy_array_module_executes_on_device_without_module_side_transfer():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no usable CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no usable CUDA device")

    event = _event()
    barrage = InhibitoryBarrage(event, EventSchedule.exact([5.0, 7.0]))
    times = cp.asarray([5.0, 5.0 + event.peak_time_ms, 9.0])
    voltage = cp.full(times.shape, -50.0)

    conductance = barrage.conductance_nS(times, cp)
    density = barrage.conductance_density_mS_cm2(times, cp)
    current = barrage.current_pA(times, voltage, cp)

    assert isinstance(conductance, cp.ndarray)
    assert isinstance(density, cp.ndarray)
    assert isinstance(current, cp.ndarray)
    assert conductance.shape == times.shape
