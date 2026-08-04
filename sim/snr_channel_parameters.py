"""Provenance-bound parameter records for the remaining SNr channels.

This module is intentionally independent of the simulator bridge and kernels.
Phillips et al. (2020) values are computational model priors, not measured SNr
channel kinetics.  Native-reference evaluation preserves published or legacy
equations without inventing a temperature; temperature-specific evaluation is
permitted only when the record carries a reference temperature and sourced Q10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, ClassVar

import numpy as np


SCHEMA_VERSION = "snr-channel-parameters-v1"
FARADAY_C_PER_MOL = 96485.33212
_TEMPERATURE_TOLERANCE_C = 1e-9


class EvidenceClass(str, Enum):
    """Scientific authority attached to a parameter value."""

    MEASURED = "measured"
    MODEL_PRIOR = "model-prior"
    UNRESOLVED = "unresolved"


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


@dataclass(frozen=True)
class Q10Provenance:
    """A Q10 value with its own authority and source, never an untagged scalar."""

    value: float
    evidence_class: EvidenceClass
    source_locator: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_class, EvidenceClass):
            raise TypeError("q10.evidence_class must be EvidenceClass")
        object.__setattr__(self, "value", _positive(self.value, "q10.value"))
        if self.evidence_class is EvidenceClass.UNRESOLVED:
            raise ValueError("an unresolved Q10 cannot authorize temperature scaling")
        if not self.source_locator.strip():
            raise ValueError("q10.source_locator is required")


@dataclass(frozen=True)
class ParameterProvenance:
    """Version and evidence carried by every mechanism record."""

    parameter_set_id: str
    evidence_class: EvidenceClass
    source_locator: str
    reference_temperature_celsius: float | None
    q10: Q10Provenance | None = None
    model_name: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_class, EvidenceClass):
            raise TypeError("evidence_class must be EvidenceClass")
        if self.q10 is not None and not isinstance(self.q10, Q10Provenance):
            raise TypeError("q10 must be Q10Provenance or None")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        if not self.parameter_set_id.strip():
            raise ValueError("parameter_set_id is required")
        if not self.source_locator.strip():
            raise ValueError("source_locator is required")
        if self.reference_temperature_celsius is not None:
            reference = _finite(
                self.reference_temperature_celsius,
                "reference_temperature_celsius",
            )
            if reference <= -273.15:
                raise ValueError("reference_temperature_celsius must exceed absolute zero")
            object.__setattr__(self, "reference_temperature_celsius", reference)
        if self.evidence_class is EvidenceClass.MODEL_PRIOR:
            if self.model_name is None or not self.model_name.strip():
                raise ValueError("model-prior parameters require model_name")
        elif self.model_name is not None:
            raise ValueError("model_name is only valid for model-prior parameters")


def _validate_provenance(provenance: ParameterProvenance) -> None:
    if not isinstance(provenance, ParameterProvenance):
        raise TypeError("provenance must be ParameterProvenance")


@dataclass(frozen=True)
class NaPParameters:
    provenance: ParameterProvenance
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_min_ms: float
    activation_tau_max_ms: float
    activation_tau_half_mv: float | None = None
    activation_tau_sigma_0_mv: float | None = None
    activation_tau_sigma_1_mv: float | None = None
    inactivation_half_mv: float = -57.0
    inactivation_slope_mv: float = -4.0
    inactivation_tau_min_ms: float = 10.0
    inactivation_tau_max_ms: float = 17.0
    inactivation_tau_half_mv: float | None = None
    inactivation_tau_sigma_0_mv: float | None = None
    inactivation_tau_sigma_1_mv: float | None = None
    conductance_ns_per_pf: float | None = None

    UNITS: ClassVar[dict[str, str]] = {
        "voltage": "mV",
        "time": "ms",
        "conductance_density": "nS/pF",
    }

    def __post_init__(self) -> None:
        _validate_provenance(self.provenance)
        for name in ("activation_slope_mv", "inactivation_slope_mv"):
            if _finite(getattr(self, name), name) == 0.0:
                raise ValueError(f"{name} must be nonzero")
        for name in (
            "activation_tau_min_ms",
            "activation_tau_max_ms",
            "inactivation_tau_min_ms",
            "inactivation_tau_max_ms",
        ):
            _positive(getattr(self, name), name)
        _finite(self.activation_half_mv, "activation_half_mv")
        _finite(self.inactivation_half_mv, "inactivation_half_mv")
        if self.conductance_ns_per_pf is not None:
            _positive(self.conductance_ns_per_pf, "conductance_ns_per_pf")
        _validate_tau_shape(self, "activation")
        _validate_tau_shape(self, "inactivation")


@dataclass(frozen=True)
class Cav22Parameters:
    provenance: ParameterProvenance
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_ms: float
    inactivation_half_mv: float
    inactivation_slope_mv: float
    inactivation_tau_ms: float
    activation_power: int = 2
    conductance_ns_per_pf: float | None = None

    UNITS: ClassVar[dict[str, str]] = NaPParameters.UNITS

    def __post_init__(self) -> None:
        _validate_provenance(self.provenance)
        for name in ("activation_half_mv", "inactivation_half_mv"):
            _finite(getattr(self, name), name)
        for name in ("activation_slope_mv", "inactivation_slope_mv"):
            if _finite(getattr(self, name), name) == 0.0:
                raise ValueError(f"{name} must be nonzero")
        _positive(self.activation_tau_ms, "activation_tau_ms")
        _positive(self.inactivation_tau_ms, "inactivation_tau_ms")
        if not isinstance(self.activation_power, int) or self.activation_power < 1:
            raise ValueError("activation_power must be a positive integer")
        if self.conductance_ns_per_pf is not None:
            _positive(self.conductance_ns_per_pf, "conductance_ns_per_pf")


@dataclass(frozen=True)
class HCNParameters:
    provenance: ParameterProvenance
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_ms: float
    reversal_mv: float
    conductance_ns_per_pf: float | None = None

    UNITS: ClassVar[dict[str, str]] = NaPParameters.UNITS

    def __post_init__(self) -> None:
        _validate_provenance(self.provenance)
        _finite(self.activation_half_mv, "activation_half_mv")
        if _finite(self.activation_slope_mv, "activation_slope_mv") == 0.0:
            raise ValueError("activation_slope_mv must be nonzero")
        _positive(self.activation_tau_ms, "activation_tau_ms")
        _finite(self.reversal_mv, "reversal_mv")
        if self.conductance_ns_per_pf is not None:
            _positive(self.conductance_ns_per_pf, "conductance_ns_per_pf")


@dataclass(frozen=True)
class CalciumParameters:
    provenance: ParameterProvenance
    baseline_um: float
    decay_tau_ms: float
    extracellular_um: float
    concentration_units: str = "micromolar"
    current_fraction: float = 1.0

    UNITS: ClassVar[dict[str, str]] = {
        "concentration": "uM",
        "time": "ms",
        "current_density": "uA/cm^2",
        "membrane_area": "um^2",
        "accessible_volume": "um^3",
    }

    def __post_init__(self) -> None:
        _validate_provenance(self.provenance)
        if self.concentration_units != "micromolar":
            raise ValueError("physical calcium records require concentration_units='micromolar'")
        baseline = _finite(self.baseline_um, "baseline_um")
        if baseline < 0.0:
            raise ValueError("baseline_um must be >= 0")
        _positive(self.decay_tau_ms, "decay_tau_ms")
        _positive(self.extracellular_um, "extracellular_um")
        fraction = _finite(self.current_fraction, "current_fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("current_fraction must be in (0, 1]")


@dataclass(frozen=True)
class SKParameters:
    provenance: ParameterProvenance
    half_activation_um: float
    hill_coefficient: float
    activation_tau_ms: float
    deactivation_tau_ms: float
    calcium_units: str = "micromolar"
    conductance_ns_per_pf: float | None = None

    UNITS: ClassVar[dict[str, str]] = {
        "calcium": "uM",
        "time": "ms",
        "conductance_density": "nS/pF",
    }

    def __post_init__(self) -> None:
        _validate_provenance(self.provenance)
        if self.calcium_units != "micromolar":
            raise ValueError("SK calcium_units must be 'micromolar'")
        _positive(self.half_activation_um, "half_activation_um")
        _positive(self.hill_coefficient, "hill_coefficient")
        _positive(self.activation_tau_ms, "activation_tau_ms")
        _positive(self.deactivation_tau_ms, "deactivation_tau_ms")
        if self.conductance_ns_per_pf is not None:
            _positive(self.conductance_ns_per_pf, "conductance_ns_per_pf")


@dataclass(frozen=True)
class SNrChannelParameterPacket:
    """A versioned, immutable collection; records retain independent evidence."""

    packet_id: str
    nap: NaPParameters
    cav22: Cav22Parameters
    hcn: HCNParameters
    calcium: CalciumParameters
    sk: SKParameters
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.packet_id.strip():
            raise ValueError("packet_id is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        validate_calcium_sk_compatibility(self.calcium, self.sk)


def _validate_tau_shape(parameters: NaPParameters, gate: str) -> None:
    values = (
        getattr(parameters, f"{gate}_tau_half_mv"),
        getattr(parameters, f"{gate}_tau_sigma_0_mv"),
        getattr(parameters, f"{gate}_tau_sigma_1_mv"),
    )
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(f"{gate} voltage-dependent tau fields must be all present or all absent")
    _finite(values[0], f"{gate}_tau_half_mv")
    if _finite(values[1], f"{gate}_tau_sigma_0_mv") == 0.0:
        raise ValueError(f"{gate}_tau_sigma_0_mv must be nonzero")
    if _finite(values[2], f"{gate}_tau_sigma_1_mv") == 0.0:
        raise ValueError(f"{gate}_tau_sigma_1_mv must be nonzero")


def validate_calcium_sk_compatibility(
    calcium: CalciumParameters,
    sk: SKParameters,
) -> None:
    """Reject the historical arbitrary-calcium/micromolar-SK unit mismatch."""

    if calcium.concentration_units != "micromolar" or sk.calcium_units != "micromolar":
        raise ValueError("calcium and SK must both use physical micromolar units")


def kinetic_temperature_factor(
    provenance: ParameterProvenance,
    temperature_celsius: float | None,
) -> float:
    """Return a kinetic speed factor, failing closed on unsupported scaling."""

    if temperature_celsius is None:
        return 1.0
    temperature = _finite(temperature_celsius, "temperature_celsius")
    reference = provenance.reference_temperature_celsius
    if reference is None:
        raise ValueError("temperature requested but reference temperature is unresolved")
    if math.isclose(temperature, reference, abs_tol=_TEMPERATURE_TOLERANCE_C):
        return 1.0
    if provenance.q10 is None:
        raise ValueError("temperature mismatch requires a provenance-tagged Q10")
    return provenance.q10.value ** ((temperature - reference) / 10.0)


def _xp(xp: Any | None) -> Any:
    return np if xp is None else xp


def _sigmoid(voltage_mv: Any, half_mv: float, slope_mv: float, xp: Any) -> Any:
    voltage = xp.asarray(voltage_mv)
    return 1.0 / (1.0 + xp.exp(-(voltage - half_mv) / slope_mv))


def first_order_gate_step(
    old: Any,
    steady_state: Any,
    dt_ms: float,
    tau_ms: Any,
    *,
    xp: Any | None = None,
) -> Any:
    """Exact first-order update for a gate held at one voltage for ``dt_ms``."""

    module = _xp(xp)
    dt = _positive(dt_ms, "dt_ms")
    tau = module.asarray(tau_ms)
    if bool(module.any(~module.isfinite(tau))) or bool(module.any(tau <= 0.0)):
        raise ValueError("tau_ms must contain finite positive values")
    return steady_state + (module.asarray(old) - steady_state) * module.exp(-dt / tau)


def legacy_nap_gates(voltage_mv: Any, *, xp: Any | None = None) -> tuple[Any, Any, float, float]:
    """Return the exact hardcoded Stage-A NaP gate references."""

    module = _xp(xp)
    return (
        _sigmoid(voltage_mv, -50.0, 4.5, module),
        _sigmoid(voltage_mv, -57.0, -6.0, module),
        0.1,
        20.0,
    )


def legacy_cav22_gates(voltage_mv: Any, *, xp: Any | None = None) -> tuple[Any, Any, float, float]:
    """Return the exact hardcoded Stage-A Cav2.2-like gate references."""

    module = _xp(xp)
    return (
        _sigmoid(voltage_mv, -27.5, 3.0, module),
        _sigmoid(voltage_mv, -52.5, -5.2, module),
        0.5,
        18.0,
    )


def legacy_hcn_gate(voltage_mv: Any, *, xp: Any | None = None) -> tuple[Any, float]:
    """Return the exact hardcoded Stage-A HCN gate reference."""

    module = _xp(xp)
    return _sigmoid(voltage_mv, -75.0, -5.5, module), 100.0


def _phillips_tau(voltage_mv: Any, low_ms: float, high_ms: float, half_mv: float,
                  sigma_0_mv: float, sigma_1_mv: float, xp: Any) -> Any:
    voltage = xp.asarray(voltage_mv)
    denominator = (
        xp.exp((half_mv - voltage) / sigma_0_mv)
        + xp.exp((half_mv - voltage) / sigma_1_mv)
    )
    return low_ms + (high_ms - low_ms) / denominator


def phillips_nap_gates(
    voltage_mv: Any,
    parameters: NaPParameters,
    *,
    temperature_celsius: float | None = None,
    xp: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Evaluate Phillips-model-prior NaP gates without upgrading their evidence."""

    if parameters.provenance.evidence_class is not EvidenceClass.MODEL_PRIOR:
        raise ValueError("Phillips equations require an explicitly tagged model prior")
    if "Phillips" not in (parameters.provenance.model_name or ""):
        raise ValueError("Phillips equations require Phillips model provenance")
    module = _xp(xp)
    speed = kinetic_temperature_factor(parameters.provenance, temperature_celsius)
    activation = _sigmoid(
        voltage_mv, parameters.activation_half_mv, parameters.activation_slope_mv, module
    )
    inactivation = _sigmoid(
        voltage_mv, parameters.inactivation_half_mv, parameters.inactivation_slope_mv, module
    )
    if parameters.activation_tau_half_mv is None or parameters.inactivation_tau_half_mv is None:
        raise ValueError("Phillips NaP prior requires voltage-dependent tau fields")
    activation_tau = _phillips_tau(
        voltage_mv,
        parameters.activation_tau_min_ms,
        parameters.activation_tau_max_ms,
        parameters.activation_tau_half_mv,
        parameters.activation_tau_sigma_0_mv,  # type: ignore[arg-type]
        parameters.activation_tau_sigma_1_mv,  # type: ignore[arg-type]
        module,
    ) / speed
    inactivation_tau = _phillips_tau(
        voltage_mv,
        parameters.inactivation_tau_min_ms,
        parameters.inactivation_tau_max_ms,
        parameters.inactivation_tau_half_mv,
        parameters.inactivation_tau_sigma_0_mv,  # type: ignore[arg-type]
        parameters.inactivation_tau_sigma_1_mv,  # type: ignore[arg-type]
        module,
    ) / speed
    return activation, inactivation, activation_tau, inactivation_tau


def channel_gates(
    voltage_mv: Any,
    parameters: Cav22Parameters | HCNParameters,
    *,
    temperature_celsius: float | None = None,
    xp: Any | None = None,
) -> tuple[Any, ...]:
    """Evaluate fixed-tau Cav2.2 or HCN parameter records."""

    module = _xp(xp)
    speed = kinetic_temperature_factor(parameters.provenance, temperature_celsius)
    if isinstance(parameters, Cav22Parameters):
        return (
            _sigmoid(voltage_mv, parameters.activation_half_mv, parameters.activation_slope_mv, module),
            _sigmoid(voltage_mv, parameters.inactivation_half_mv, parameters.inactivation_slope_mv, module),
            parameters.activation_tau_ms / speed,
            parameters.inactivation_tau_ms / speed,
        )
    if isinstance(parameters, HCNParameters):
        return (
            _sigmoid(voltage_mv, parameters.activation_half_mv, parameters.activation_slope_mv, module),
            parameters.activation_tau_ms / speed,
        )
    raise TypeError("parameters must be Cav22Parameters or HCNParameters")


def calcium_influx_delta_um(
    inward_current_density_ua_per_cm2: Any,
    dt_ms: float,
    *,
    membrane_area_um2: float,
    accessible_volume_um3: float,
    current_fraction: float = 1.0,
    calcium_valence: int = 2,
    xp: Any | None = None,
) -> Any:
    """Convert inward Ca current density to a concentration increment in uM.

    Positive input denotes inward current.  Both membrane area and accessible
    volume are mandatory so current density cannot be mistaken for whole-cell
    current or mapped onto an arbitrary concentration state.
    """

    module = _xp(xp)
    dt = _positive(dt_ms, "dt_ms")
    area = _positive(membrane_area_um2, "membrane_area_um2")
    volume = _positive(accessible_volume_um3, "accessible_volume_um3")
    fraction = _finite(current_fraction, "current_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("current_fraction must be in (0, 1]")
    if not isinstance(calcium_valence, int) or calcium_valence <= 0:
        raise ValueError("calcium_valence must be a positive integer")
    density = module.asarray(inward_current_density_ua_per_cm2)
    if bool(module.any(~module.isfinite(density))) or bool(module.any(density < 0.0)):
        raise ValueError("inward_current_density_ua_per_cm2 must be finite and >= 0")

    # uA/cm2 * um2 * ms = 1e-17 C; um3 = 1e-15 L.
    charge_c = density * area * dt * 1e-17 * fraction
    moles = charge_c / (calcium_valence * FARADAY_C_PER_MOL)
    volume_l = volume * 1e-15
    return (moles / volume_l) * 1e6


def calcium_concentration_step(
    calcium_old_um: Any,
    inward_current_density_ua_per_cm2: Any,
    dt_ms: float,
    parameters: CalciumParameters,
    *,
    membrane_area_um2: float,
    accessible_volume_um3: float,
    temperature_celsius: float | None = None,
    xp: Any | None = None,
) -> Any:
    """Exact decay update with constant physical calcium influx over one step."""

    module = _xp(xp)
    speed = kinetic_temperature_factor(parameters.provenance, temperature_celsius)
    tau_ms = parameters.decay_tau_ms / speed
    delta_um = calcium_influx_delta_um(
        inward_current_density_ua_per_cm2,
        dt_ms,
        membrane_area_um2=membrane_area_um2,
        accessible_volume_um3=accessible_volume_um3,
        current_fraction=parameters.current_fraction,
        xp=module,
    )
    influx_rate_um_per_ms = delta_um / float(dt_ms)
    target = parameters.baseline_um + tau_ms * influx_rate_um_per_ms
    old = module.asarray(calcium_old_um)
    if bool(module.any(~module.isfinite(old))) or bool(module.any(old < 0.0)):
        raise ValueError("calcium_old_um must be finite and >= 0")
    return target + (old - target) * module.exp(-float(dt_ms) / tau_ms)


def sk_steady_state(calcium_um: Any, parameters: SKParameters, *, xp: Any | None = None) -> Any:
    module = _xp(xp)
    calcium = module.asarray(calcium_um)
    if bool(module.any(~module.isfinite(calcium))) or bool(module.any(calcium < 0.0)):
        raise ValueError("calcium_um must be finite and >= 0")
    powered = module.power(calcium, parameters.hill_coefficient)
    half_powered = parameters.half_activation_um ** parameters.hill_coefficient
    return powered / (powered + half_powered)


def sk_activation_step(
    old: Any,
    calcium_um: Any,
    dt_ms: float,
    parameters: SKParameters,
    *,
    calcium_units: str,
    temperature_celsius: float | None = None,
    xp: Any | None = None,
) -> Any:
    """Advance SK activation with distinct opening and closing kinetics."""

    if calcium_units != "micromolar" or parameters.calcium_units != "micromolar":
        raise ValueError("micromolar SK parameters cannot consume arbitrary calcium units")
    module = _xp(xp)
    old_array = module.asarray(old)
    if bool(module.any(~module.isfinite(old_array))) or bool(
        module.any((old_array < 0.0) | (old_array > 1.0))
    ):
        raise ValueError("old SK activation must be finite and in [0, 1]")
    steady = sk_steady_state(calcium_um, parameters, xp=module)
    speed = kinetic_temperature_factor(parameters.provenance, temperature_celsius)
    tau = module.where(
        steady >= old_array,
        parameters.activation_tau_ms / speed,
        parameters.deactivation_tau_ms / speed,
    )
    updated = first_order_gate_step(old_array, steady, dt_ms, tau, xp=module)
    return module.clip(updated, 0.0, 1.0)


def phillips_model_prior_packet() -> SNrChannelParameterPacket:
    """Return published model values, explicitly tagged as non-measurement priors."""

    source = "Phillips et al. 2020, eLife 9:e55592, Eq. 15-18 and Table 1"
    model = "Phillips et al. 2020 SNr computational model"

    def provenance(mechanism: str) -> ParameterProvenance:
        return ParameterProvenance(
            parameter_set_id=f"phillips-2020-{mechanism}-v1",
            evidence_class=EvidenceClass.MODEL_PRIOR,
            source_locator=source,
            reference_temperature_celsius=None,
            model_name=model,
        )

    return SNrChannelParameterPacket(
        packet_id="phillips-2020-snr-model-priors-v1",
        nap=NaPParameters(
            provenance=provenance("nap"),
            activation_half_mv=-50.0,
            activation_slope_mv=3.0,
            activation_tau_min_ms=0.03,
            activation_tau_max_ms=0.146,
            activation_tau_half_mv=-42.6,
            activation_tau_sigma_0_mv=14.4,
            activation_tau_sigma_1_mv=-14.4,
            inactivation_half_mv=-57.0,
            inactivation_slope_mv=-4.0,
            inactivation_tau_min_ms=10.0,
            inactivation_tau_max_ms=17.0,
            inactivation_tau_half_mv=-34.0,
            inactivation_tau_sigma_0_mv=26.0,
            inactivation_tau_sigma_1_mv=-31.9,
            conductance_ns_per_pf=0.175,
        ),
        cav22=Cav22Parameters(
            provenance=provenance("calcium-current"),
            activation_half_mv=-27.5,
            activation_slope_mv=3.0,
            activation_tau_ms=0.5,
            inactivation_half_mv=-52.5,
            inactivation_slope_mv=-5.2,
            inactivation_tau_ms=18.0,
            activation_power=1,
            conductance_ns_per_pf=0.7,
        ),
        hcn=HCNParameters(
            provenance=ParameterProvenance(
                parameter_set_id="snr-hcn-unresolved-v1",
                evidence_class=EvidenceClass.UNRESOLVED,
                source_locator="Atherton and Bevan 2005 measured only a protocol-averaged activation rate",
                reference_temperature_celsius=37.0,
            ),
            activation_half_mv=-75.0,
            activation_slope_mv=-5.5,
            activation_tau_ms=100.0,
            reversal_mv=-30.0,
        ),
        calcium=CalciumParameters(
            provenance=provenance("calcium-state"),
            baseline_um=5e-5,
            decay_tau_ms=250.0,
            extracellular_um=4000.0,
        ),
        sk=SKParameters(
            provenance=ParameterProvenance(
                parameter_set_id="xia-1998-sk2-transfer-prior-v1",
                evidence_class=EvidenceClass.MODEL_PRIOR,
                source_locator="Xia et al. 1998 recombinant rat SK2; transferred by Phillips et al. 2020",
                reference_temperature_celsius=None,
                model_name="transferred recombinant SK2 prior",
            ),
            half_activation_um=0.62,
            hill_coefficient=4.0,
            activation_tau_ms=4.1,
            deactivation_tau_ms=57.3,
        ),
    )


__all__ = [
    "CalciumParameters",
    "Cav22Parameters",
    "EvidenceClass",
    "HCNParameters",
    "NaPParameters",
    "ParameterProvenance",
    "Q10Provenance",
    "SCHEMA_VERSION",
    "SKParameters",
    "SNrChannelParameterPacket",
    "calcium_concentration_step",
    "calcium_influx_delta_um",
    "channel_gates",
    "first_order_gate_step",
    "kinetic_temperature_factor",
    "legacy_cav22_gates",
    "legacy_hcn_gate",
    "legacy_nap_gates",
    "phillips_model_prior_packet",
    "phillips_nap_gates",
    "sk_activation_step",
    "sk_steady_state",
    "validate_calcium_sk_compatibility",
]
