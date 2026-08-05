"""Source-bound constants for the V14 SNr structural successor.

Only directly measured clamp values and explicit interpolation priors live
here. Conductance calibration and whole-cell morphology are intentionally
absent until the architecture transfer gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping


PROTOCOL_PATH = "research/specs/v14_snr_stageB_structural_successor_v2.json"
PROTOCOL_SHA256 = "c0ab042b65cb7be21b640d9a14cddb13582a1662f3f0aa3159093aa1d13792c4"


@dataclass(frozen=True, slots=True)
class FastChannelClampParameters:
    """Float32-compatible Stage 1 parameters with no conductance fitting."""

    na_activation_half_mv: float = -30.2
    na_activation_slope_mv: float = 6.2
    na_inactivation_half_mv: float = -63.3
    na_inactivation_slope_mv: float = 8.1
    na_recovery_fast_tau_ms: float = 0.59
    na_recovery_slow_tau_ms: float = 35.1
    na_recovery_fast_fraction: float = 0.526
    na_activation_current_rise_10_90_ms: float = 0.085
    na_inactivation_current_decay_10_90_ms: float = 0.191
    na_deactivation_current_tau_minus_40_ms: float = 0.099
    kv3_activation_half_mv: float = -8.5
    kv3_activation_slope_mv: float = 8.9
    kv3_inactivation_half_mv: float = -49.2
    kv3_inactivation_slope_mv: float = 8.7
    kv3_activation_current_rise_20_80_plus_40_ms: float = 0.41
    kv3_deactivation_current_tau_minus_60_ms: float = 0.82
    kv3_deactivation_current_tau_minus_50_ms: float = 1.35
    kv3_deactivation_current_tau_minus_40_ms: float = 1.87
    na_reversal_mv: float = 50.0
    potassium_reversal_mv: float = -90.0
    kv3_inactivation_tau_prior_ms: float = 100.0

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("fast-channel clamp parameters must be finite scalars")
        positive = (
            self.na_activation_slope_mv,
            self.na_inactivation_slope_mv,
            self.na_recovery_fast_tau_ms,
            self.na_recovery_slow_tau_ms,
            self.na_activation_current_rise_10_90_ms,
            self.na_inactivation_current_decay_10_90_ms,
            self.na_deactivation_current_tau_minus_40_ms,
            self.kv3_activation_slope_mv,
            self.kv3_inactivation_slope_mv,
            self.kv3_activation_current_rise_20_80_plus_40_ms,
            self.kv3_deactivation_current_tau_minus_60_ms,
            self.kv3_deactivation_current_tau_minus_50_ms,
            self.kv3_deactivation_current_tau_minus_40_ms,
            self.kv3_inactivation_tau_prior_ms,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("fast-channel slopes and time constants must be positive")
        if not 0 < self.na_recovery_fast_fraction < 1:
            raise ValueError("fast recovery fraction must lie strictly inside (0, 1)")

    @property
    def na_activation_gate_tau_at_zero_ms(self) -> float:
        # This is an initialization prior, not a claim that m^3 alone fits the
        # measured transient. The Stage 1 runner measures the composite current.
        return self.na_activation_current_rise_10_90_ms / power_gate_rise_factor(3, 0.1, 0.9)

    @property
    def na_deactivation_gate_tau_at_minus_40_ms(self) -> float:
        return 3.0 * self.na_deactivation_current_tau_minus_40_ms

    @property
    def na_inactivation_gate_tau_at_zero_ms(self) -> float:
        return self.na_inactivation_current_decay_10_90_ms / math.log(9.0)

    @property
    def kv3_activation_gate_tau_at_plus_40_ms(self) -> float:
        return self.kv3_activation_current_rise_20_80_plus_40_ms / power_gate_rise_factor(
            4, 0.2, 0.8
        )

    @property
    def kv3_deactivation_gate_taus_ms(self) -> tuple[float, float, float]:
        return tuple(4.0 * value for value in (
            self.kv3_deactivation_current_tau_minus_60_ms,
            self.kv3_deactivation_current_tau_minus_50_ms,
            self.kv3_deactivation_current_tau_minus_40_ms,
        ))


def power_gate_rise_factor(power: int, low: float, high: float) -> float:
    """Return the first-order gate-time factor for a powered current rise."""
    if type(power) is not int or power < 1 or not 0 < low < high < 1:
        raise ValueError("power and crossing fractions are invalid")
    return math.log((1.0 - low ** (1.0 / power)) / (1.0 - high ** (1.0 / power)))


EVIDENCE_CLASSES: Mapping[str, str] = MappingProxyType({
    "steady_state_and_reported_kinetics": "direct_measured_transfer_juvenile_rat_snr",
    "gate_power_conversion": "equation_derived",
    "tau_between_voltage_endpoints": "log_linear_interpolation_model_prior",
    "kv3_inactivation_tau_prior_ms": "unresolved_model_prior_for_10s_equilibration",
    "conductance_scale": "unavailable_not_authorized_in_stage_1",
    "temperature_q10": "unavailable_not_authorized_in_stage_1",
})


UNITS: Mapping[str, str] = MappingProxyType({
    "voltage": "mV",
    "time": "ms",
    "conductance": "normalized",
    "current": "normalized_conductance_times_mV",
    "gate": "dimensionless",
})


DEFAULT_FAST_CHANNEL_PARAMETERS = FastChannelClampParameters()


__all__ = [
    "DEFAULT_FAST_CHANNEL_PARAMETERS",
    "EVIDENCE_CLASSES",
    "FastChannelClampParameters",
    "PROTOCOL_PATH",
    "PROTOCOL_SHA256",
    "UNITS",
    "power_gate_rise_factor",
]
