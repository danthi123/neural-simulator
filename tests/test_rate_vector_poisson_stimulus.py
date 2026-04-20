"""Tests for the RATE_VECTOR_POISSON stimulus pattern (G1 encoder primitive).

Uses a NumPy-backed MockCuPy to run on CPU. The existing
`tests/test_experiment_system.py` uses the same technique; we extend it with
a seedable RNG so Poisson tests are deterministic.
"""
import numpy as np
import pytest


class MockRandom:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def random(self, n):
        return self.rng.random(n).astype(np.float32)
    def randn(self, n):
        return self.rng.standard_normal(n).astype(np.float32)


class MockCuPy:
    float32 = np.float32
    int32 = np.int32
    bool_ = np.bool_

    def __init__(self, seed=0):
        self.random = MockRandom(seed=seed)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def array(data, dtype=None):
        return np.array(data, dtype=dtype)

    @staticmethod
    def sum(arr):
        class R:
            def __init__(self, v): self.v = v
            def get(self): return self.v
        return R(np.sum(arr))

    @staticmethod
    def where(cond, x, y):
        return np.where(cond, x, y)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)


from sim.enums import StimulusPatternType
from sim.config import StimulusPattern, StimulusChannel
from experiment.groups import NeuronGroupManager
from experiment.stimulus import StimulusManager


def _build_manager(mock_cp, n_neurons, target_indices, rate_vector_hz,
                   spike_current_pA=200.0, spike_duration_ms=1.0,
                   duration_ms=1000.0, dt_ms=1.0):
    gm = NeuronGroupManager(n_neurons)
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        spike_current_pA=spike_current_pA,
        spike_duration_ms=spike_duration_ms,
        rate_vector_hz=list(rate_vector_hz),
    )
    ch = StimulusChannel(
        name="inp_ch",
        pattern=pat,
        target_neuron_indices=list(target_indices),
        onset_ms=0.0,
        duration_ms=duration_ms,
    )
    sm = StimulusManager(n_neurons, dt_ms)
    sm.initialize([ch], gm, mock_cp)
    return sm, ch


def test_zero_rate_produces_zero_spikes():
    cp = MockCuPy(seed=1)
    n_neurons = 10
    target = list(range(10))
    rates = [0.0] * 10
    sm, ch = _build_manager(cp, n_neurons, target, rates, duration_ms=2500.0)
    total = np.zeros(n_neurons, dtype=np.float32)
    for step in range(2000):
        I = sm.compute_step_current(current_time_ms=step * 1.0,
                                    phase_start_ms=0.0, cp_module=cp)
        total += I
    assert total.sum() == 0.0, "Zero-rate neurons should never spike"


def test_uniform_rate_matches_expected_poisson():
    """With rate=20 Hz and 5 s, mean empirical rate within 25% of target."""
    cp = MockCuPy(seed=2)
    n_neurons = 64
    target = list(range(n_neurons))
    target_rate = 20.0
    rates = [target_rate] * n_neurons
    sim_duration_s = 5.0
    n_steps = int(sim_duration_s * 1000)
    sm, ch = _build_manager(cp, n_neurons, target, rates,
                            spike_current_pA=200.0,
                            spike_duration_ms=1.0,
                            duration_ms=float(n_steps) + 100.0)
    spike_counts = np.zeros(n_neurons, dtype=np.int64)
    for step in range(n_steps):
        I = sm.compute_step_current(current_time_ms=step * 1.0,
                                    phase_start_ms=0.0, cp_module=cp)
        spike_counts += (I > 0).astype(np.int64)
    mean_emp = float(np.mean(spike_counts / sim_duration_s))
    assert abs(mean_emp - target_rate) / target_rate < 0.25, (
        f"Empirical rate {mean_emp:.2f} Hz vs target {target_rate:.2f} Hz"
    )


def test_per_neuron_rate_differentiation():
    cp = MockCuPy(seed=3)
    n = 8
    target = list(range(n))
    rates = [5.0] * 4 + [30.0] * 4
    sm, ch = _build_manager(cp, n, target, rates, spike_duration_ms=1.0,
                            duration_ms=3500.0)
    counts = np.zeros(n, dtype=np.int64)
    for step in range(3000):
        I = sm.compute_step_current(current_time_ms=step * 1.0,
                                    phase_start_ms=0.0, cp_module=cp)
        counts += (I > 0).astype(np.int64)
    low = counts[:4].mean()
    high = counts[4:].mean()
    assert high > 3 * low, f"High-rate neurons ({high}) should be >>3x low ({low})"


def test_rate_vector_length_must_match_target_count():
    cp = MockCuPy(seed=4)
    n = 5
    target = list(range(n))
    rates = [10.0, 10.0]  # only 2 entries for 5 targets
    pat = StimulusPattern(
        pattern_type=StimulusPatternType.RATE_VECTOR_POISSON.name,
        rate_vector_hz=list(rates),
    )
    ch = StimulusChannel(
        name="ch", pattern=pat, target_neuron_indices=list(target),
        onset_ms=0.0, duration_ms=500.0,
    )
    gm = NeuronGroupManager(n)
    sm = StimulusManager(n, 1.0)
    with pytest.raises(ValueError, match="rate_vector_hz length"):
        sm.initialize([ch], gm, cp)


def test_disabled_channel_produces_no_current():
    cp = MockCuPy(seed=5)
    n = 8
    target = list(range(n))
    rates = [40.0] * n
    sm, ch = _build_manager(cp, n, target, rates)
    ch.enabled = False
    for step in range(500):
        I = sm.compute_step_current(current_time_ms=step * 1.0,
                                    phase_start_ms=0.0, cp_module=cp)
        assert np.all(I == 0), f"Disabled channel should produce zero current at step {step}"
