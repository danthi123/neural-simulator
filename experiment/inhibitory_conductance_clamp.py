"""Deterministic inhibitory conductance-clamp protocol primitives.

This module is intentionally independent of ``SimulationBridge``.  It builds
whole-cell and density-valued conductance waveforms with an injected NumPy-like
array module (``numpy`` or ``cupy``), without transferring array values back to
the host.

Unit conventions
----------------
``event_peak_nS`` is a whole-cell peak conductance.  For membrane area
``area_um2``, the simulator HH density is

    g_mS_cm2 = g_nS * 100 / area_um2

because 1 nS = 1e-6 mS and 1 um2 = 1e-8 cm2.  The corresponding membrane
drive uses the simulator's positive-inward convention:

    current_pA = g_nS * (reversal_mV - membrane_voltage_mV)

since nS * mV = pA.

Each event uses ``exp(-t/tau_decay) - exp(-t/tau_rise)`` for t >= 0.  The
kernel is divided by its value at the analytic continuous-time peak, so the
requested event peak is achieved exactly in continuous time.  A sampled trace
is not renormalized to its grid and can therefore fall slightly below that
peak when the analytic peak lies between samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Any, Sequence


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(name: str, value: Any) -> float:
    value = _finite_number(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def whole_cell_nS_to_density_mS_cm2(conductance_nS: Any, area_um2: float):
    """Convert whole-cell conductance in nS to conductance density in mS/cm2."""
    area_um2 = _positive("area_um2", area_um2)
    return conductance_nS * (100.0 / area_um2)


def whole_cell_current_pA(
    conductance_nS: Any,
    membrane_voltage_mV: Any,
    reversal_mV: float,
):
    """Return positive-inward membrane current in pA from whole-cell values."""
    reversal_mV = _finite_number("reversal_mV", reversal_mV)
    return conductance_nS * (reversal_mV - membrane_voltage_mV)


@dataclass(frozen=True)
class BiexponentialInhibitoryEvent:
    """Pathway-specific, peak-normalized inhibitory event definition."""

    pathway: str
    tau_rise_ms: float
    tau_decay_ms: float
    reversal_mV: float
    event_peak_nS: float
    membrane_area_um2: float

    def __post_init__(self):
        if not isinstance(self.pathway, str) or not self.pathway.strip():
            raise ValueError("pathway must be a non-empty string")
        rise = _positive("tau_rise_ms", self.tau_rise_ms)
        decay = _positive("tau_decay_ms", self.tau_decay_ms)
        if rise >= decay:
            raise ValueError("tau_rise_ms must be less than tau_decay_ms")
        _finite_number("reversal_mV", self.reversal_mV)
        _positive("event_peak_nS", self.event_peak_nS)
        _positive("membrane_area_um2", self.membrane_area_um2)

    @property
    def peak_time_ms(self) -> float:
        rise = float(self.tau_rise_ms)
        decay = float(self.tau_decay_ms)
        return rise * decay * math.log(decay / rise) / (decay - rise)

    @property
    def peak_normalization(self) -> float:
        t_peak = self.peak_time_ms
        return math.exp(-t_peak / self.tau_decay_ms) - math.exp(
            -t_peak / self.tau_rise_ms
        )

    @property
    def normalized_kernel_integral_ms(self) -> float:
        """Continuous integral of one unit-peak kernel, in milliseconds."""
        return (self.tau_decay_ms - self.tau_rise_ms) / self.peak_normalization

    @property
    def event_peak_density_mS_cm2(self) -> float:
        return whole_cell_nS_to_density_mS_cm2(
            self.event_peak_nS, self.membrane_area_um2
        )

    def kernel(self, elapsed_ms: Any, xp: Any):
        """Evaluate the unit-peak causal kernel through ``xp`` only."""
        elapsed = xp.asarray(elapsed_ms)
        causal_elapsed = xp.maximum(elapsed, 0.0)
        raw = xp.exp(-causal_elapsed / self.tau_decay_ms) - xp.exp(
            -causal_elapsed / self.tau_rise_ms
        )
        return xp.where(elapsed >= 0.0, raw / self.peak_normalization, 0.0)


@dataclass(frozen=True)
class EventSchedule:
    """Exact event times and the provenance of their deterministic schedule."""

    event_times_ms: tuple[float, ...]
    kind: str
    seed: int | None = None
    rate_hz: float | None = None
    start_ms: float | None = None
    stop_ms: float | None = None

    def __post_init__(self):
        if self.kind not in {"exact", "seeded_poisson"}:
            raise ValueError("kind must be 'exact' or 'seeded_poisson'")
        previous = -math.inf
        for event_time in self.event_times_ms:
            event_time = _finite_number("event_time_ms", event_time)
            if event_time < 0.0:
                raise ValueError("event times must be non-negative")
            if event_time < previous:
                raise ValueError("event times must be sorted")
            previous = event_time
        if self.kind == "exact":
            poisson_metadata = (
                self.seed,
                self.rate_hz,
                self.start_ms,
                self.stop_ms,
            )
            if any(value is not None for value in poisson_metadata):
                raise ValueError("exact schedules cannot carry Poisson metadata")
        else:
            if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
                raise TypeError("seed must be an integer")
            rate = _positive("rate_hz", self.rate_hz)
            start = _finite_number("start_ms", self.start_ms)
            stop = _finite_number("stop_ms", self.stop_ms)
            if start < 0.0 or stop <= start:
                raise ValueError("Poisson schedule requires 0 <= start_ms < stop_ms")
            if any(t < start or t >= stop for t in self.event_times_ms):
                raise ValueError("Poisson event times must lie in [start_ms, stop_ms)")
            object.__setattr__(self, "rate_hz", rate)

    @classmethod
    def exact(cls, event_times_ms: Sequence[float]) -> "EventSchedule":
        return cls(tuple(event_times_ms), kind="exact")

    @classmethod
    def seeded_poisson(
        cls,
        rate_hz: float,
        start_ms: float,
        stop_ms: float,
        seed: int,
    ) -> "EventSchedule":
        """Generate a continuous-time Poisson schedule with exponential waits."""
        import numpy as np

        rate = _positive("rate_hz", rate_hz)
        start = _finite_number("start_ms", start_ms)
        stop = _finite_number("stop_ms", stop_ms)
        if start < 0.0 or stop <= start:
            raise ValueError("Poisson schedule requires 0 <= start_ms < stop_ms")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")

        rng = np.random.default_rng(int(seed))
        scale_ms = 1000.0 / rate
        event_times = []
        event_time = start + float(rng.exponential(scale_ms))
        while event_time < stop:
            event_times.append(event_time)
            event_time += float(rng.exponential(scale_ms))
        return cls(
            tuple(event_times),
            kind="seeded_poisson",
            seed=int(seed),
            rate_hz=rate,
            start_ms=start,
            stop_ms=stop,
        )


@dataclass(frozen=True)
class InhibitoryBarrage:
    """Superposition of pathway-identified biexponential inhibitory events."""

    event: BiexponentialInhibitoryEvent
    schedule: EventSchedule

    @property
    def pathway(self) -> str:
        return self.event.pathway

    def conductance_nS(self, times_ms: Any, xp: Any):
        times = xp.asarray(times_ms)
        if not self.schedule.event_times_ms:
            return xp.zeros_like(times)
        events = xp.asarray(self.schedule.event_times_ms)
        elapsed = times[..., None] - events
        return self.event.event_peak_nS * xp.sum(
            self.event.kernel(elapsed, xp), axis=-1
        )

    def conductance_density_mS_cm2(self, times_ms: Any, xp: Any):
        return whole_cell_nS_to_density_mS_cm2(
            self.conductance_nS(times_ms, xp), self.event.membrane_area_um2
        )

    def current_pA(self, times_ms: Any, membrane_voltage_mV: Any, xp: Any):
        voltage = xp.asarray(membrane_voltage_mV)
        return whole_cell_current_pA(
            self.conductance_nS(times_ms, xp), voltage, self.event.reversal_mV
        )


@dataclass(frozen=True)
class MatchedMeanConductanceStep:
    """Constant conductance step matched to a barrage's expected stationary mean."""

    pathway: str
    conductance_nS: float
    reversal_mV: float
    membrane_area_um2: float
    onset_ms: float
    offset_ms: float
    matched_rate_hz: float
    source_event_peak_nS: float
    source_kernel_integral_ms: float

    def __post_init__(self):
        if not isinstance(self.pathway, str) or not self.pathway.strip():
            raise ValueError("pathway must be a non-empty string")
        _positive("conductance_nS", self.conductance_nS)
        _finite_number("reversal_mV", self.reversal_mV)
        _positive("membrane_area_um2", self.membrane_area_um2)
        onset = _finite_number("onset_ms", self.onset_ms)
        offset = _finite_number("offset_ms", self.offset_ms)
        if onset < 0.0 or offset <= onset:
            raise ValueError("step requires 0 <= onset_ms < offset_ms")
        _positive("matched_rate_hz", self.matched_rate_hz)
        _positive("source_event_peak_nS", self.source_event_peak_nS)
        _positive("source_kernel_integral_ms", self.source_kernel_integral_ms)

    @classmethod
    def from_expected_barrage_mean(
        cls,
        event: BiexponentialInhibitoryEvent,
        rate_hz: float,
        onset_ms: float,
        offset_ms: float,
    ) -> "MatchedMeanConductanceStep":
        """Match ``rate * integral(event kernel)`` for a stationary barrage."""
        rate = _positive("rate_hz", rate_hz)
        integral = event.normalized_kernel_integral_ms
        mean_nS = (rate / 1000.0) * event.event_peak_nS * integral
        return cls(
            pathway=event.pathway,
            conductance_nS=mean_nS,
            reversal_mV=event.reversal_mV,
            membrane_area_um2=event.membrane_area_um2,
            onset_ms=onset_ms,
            offset_ms=offset_ms,
            matched_rate_hz=rate,
            source_event_peak_nS=event.event_peak_nS,
            source_kernel_integral_ms=integral,
        )

    def conductance_nS_trace(self, times_ms: Any, xp: Any):
        times = xp.asarray(times_ms)
        active = (times >= self.onset_ms) & (times < self.offset_ms)
        return xp.where(active, self.conductance_nS, 0.0)

    def conductance_density_mS_cm2(self, times_ms: Any, xp: Any):
        return whole_cell_nS_to_density_mS_cm2(
            self.conductance_nS_trace(times_ms, xp), self.membrane_area_um2
        )

    def current_pA(self, times_ms: Any, membrane_voltage_mV: Any, xp: Any):
        voltage = xp.asarray(membrane_voltage_mV)
        return whole_cell_current_pA(
            self.conductance_nS_trace(times_ms, xp), voltage, self.reversal_mV
        )
