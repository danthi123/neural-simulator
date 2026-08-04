"""CPU-only typed materialization of authenticated SNr parameter packets.

This module deliberately performs no simulation update.  It converts the
immutable output of :mod:`sim.snr_executable_packet` into typed, finite scalar
records that a future CPU or GPU kernel adapter can consume without reparsing
scientific packet text.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from sim.snr_executable_packet import (
    MaterializedPacket,
    MaterializedParameterLeaf,
    PARAMETER_SCHEMA,
)


FARADAY_C_PER_MOL = 96485.33212
"""Faraday constant used for the packet calcium-current conversion."""

FAST_HH_REFERENCE_TEMPERATURE_C = 6.3
"""Reference temperature of the classic fast Hodgkin-Huxley rate equations."""


class PacketParameterError(ValueError):
    """Raised when a materialized packet cannot become executable parameters."""


@dataclass(frozen=True, slots=True)
class FastHHParameters:
    capacitance_density_uf_per_cm2: float
    sodium_conductance_density_ms_per_cm2: float
    potassium_conductance_density_ms_per_cm2: float
    leak_conductance_density_ms_per_cm2: float
    initial_voltage_mv: float
    spike_detection_voltage_mv: float
    initial_m: float
    initial_h: float
    initial_n: float
    sodium_activation_q10: float
    sodium_inactivation_q10: float
    potassium_activation_q10: float


@dataclass(frozen=True, slots=True)
class NaLCNParameters:
    conductance_density_ms_per_cm2: float


@dataclass(frozen=True, slots=True)
class NaPParameters:
    conductance_density_ns_per_pf: float
    conductance_density_ms_per_cm2: float
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_min_ms: float
    activation_tau_max_ms: float
    activation_tau_half_mv: float
    activation_tau_sigma_0_mv: float
    activation_tau_sigma_1_mv: float
    inactivation_half_mv: float
    inactivation_slope_mv: float
    inactivation_tau_min_ms: float
    inactivation_tau_max_ms: float
    inactivation_tau_half_mv: float
    inactivation_tau_sigma_0_mv: float
    inactivation_tau_sigma_1_mv: float
    kinetic_q10: float
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class Cav22Parameters:
    conductance_density_ns_per_pf: float
    conductance_density_ms_per_cm2: float
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_ms: float
    inactivation_half_mv: float
    inactivation_slope_mv: float
    inactivation_tau_ms: float
    activation_power: float
    kinetic_q10: float
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class HCNParameters:
    conductance_density_ns_per_pf: float
    conductance_density_ms_per_cm2: float
    activation_half_mv: float
    activation_slope_mv: float
    activation_tau_ms: float
    kinetic_q10: float
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class CalciumParameters:
    baseline_um: float
    decay_tau_ms: float
    current_fraction: float
    kinetic_q10: float
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class SKParameters:
    conductance_density_ns_per_pf: float
    conductance_density_ms_per_cm2: float
    half_activation_um: float
    hill_coefficient: float
    activation_tau_ms: float
    deactivation_tau_ms: float
    kinetic_q10: float
    reference_temperature_c: float


@dataclass(frozen=True, slots=True)
class GeometryParameters:
    membrane_area_um2: float
    accessible_calcium_volume_um3: float


@dataclass(frozen=True, slots=True)
class IonicEnvironmentParameters:
    sodium_reversal_mv: float
    potassium_reversal_mv: float
    leak_reversal_mv: float
    nalcn_reversal_mv: float
    calcium_reversal_mv: float
    hcn_reversal_mv: float
    extracellular_calcium_um: float
    calcium_valence: float
    simulation_temperature_c: float


@dataclass(frozen=True, slots=True)
class Q10Factors:
    """Kinetic speed multipliers at the packet simulation temperature."""

    fast_hh_sodium_activation: float
    fast_hh_sodium_inactivation: float
    fast_hh_potassium_activation: float
    nap: float
    cav22: float
    hcn: float
    calcium: float
    sk: float


@dataclass(frozen=True, slots=True)
class SNrPacketParameters:
    """Complete, immutable CPU parameter record derived from one packet.

    ``raw_groups`` retains every original authenticated leaf.  ``parsed_values``
    has exactly the same nested shape and contains each leaf's one parsed CPU
    scalar.  They make schema coverage auditable independently of the typed
    convenience records below.
    """

    packet_id: str
    packet_sha256: str
    structural_sha256: str
    raw_groups: Mapping[str, Mapping[str, MaterializedParameterLeaf]]
    parsed_values: Mapping[str, Mapping[str, float]]
    fast_hh: FastHHParameters
    nalcn: NaLCNParameters
    nap: NaPParameters
    cav22: Cav22Parameters
    hcn: HCNParameters
    calcium: CalciumParameters
    sk: SKParameters
    geometry: GeometryParameters
    ionic_env: IonicEnvironmentParameters
    q10_factors: Q10Factors
    calcium_influx_um_per_ms_per_inward_ua_per_cm2: float


def materialize_runtime_parameters(packet: MaterializedPacket) -> SNrPacketParameters:
    """Convert one exact materialized packet into finite typed CPU parameters.

    The conversion walks ``PARAMETER_SCHEMA`` once.  It rejects any missing or
    additional leaf, non-finite or non-``float32``-representable scalar, unit
    mismatch, invalid Cav2.2 activation power, and invalid physical calcium
    conversion inputs.
    """

    if type(packet) is not MaterializedPacket:
        raise TypeError("packet must be an exact MaterializedPacket")

    raw_groups, values = _snapshot_and_parse(packet)
    fast = values["fast_hh"]
    nalcn = values["nalcn"]
    nap = values["nap"]
    cav22 = values["cav22"]
    hcn = values["hcn"]
    calcium = values["calcium"]
    sk = values["sk"]
    geometry = values["geometry"]
    ionic = values["ionic_env"]

    capacitance = _positive(fast["capacitance_density"], "fast_hh.capacitance_density")
    activation_power = cav22["activation_power"]
    if not activation_power.is_integer() or not 1.0 <= activation_power <= 16.0:
        raise PacketParameterError(
            "cav22.activation_power must be an integer in [1, 16]"
        )

    fast_record = FastHHParameters(
        capacitance,
        fast["sodium_conductance_density"],
        fast["potassium_conductance_density"],
        fast["leak_conductance_density"],
        fast["initial_voltage"],
        fast["spike_detection_voltage"],
        fast["initial_m"],
        fast["initial_h"],
        fast["initial_n"],
        fast["sodium_activation_q10"],
        fast["sodium_inactivation_q10"],
        fast["potassium_activation_q10"],
    )
    nap_record = NaPParameters(
        nap["conductance_density"],
        _density_ns_per_pf_to_ms_per_cm2(nap["conductance_density"], capacitance, "nap"),
        nap["activation_half"], nap["activation_slope"], nap["activation_tau_min"],
        nap["activation_tau_max"], nap["activation_tau_half"], nap["activation_tau_sigma_0"],
        nap["activation_tau_sigma_1"], nap["inactivation_half"], nap["inactivation_slope"],
        nap["inactivation_tau_min"], nap["inactivation_tau_max"], nap["inactivation_tau_half"],
        nap["inactivation_tau_sigma_0"], nap["inactivation_tau_sigma_1"], nap["kinetic_q10"],
        nap["reference_temperature"],
    )
    cav22_record = Cav22Parameters(
        cav22["conductance_density"],
        _density_ns_per_pf_to_ms_per_cm2(cav22["conductance_density"], capacitance, "cav22"),
        cav22["activation_half"], cav22["activation_slope"], cav22["activation_tau"],
        cav22["inactivation_half"], cav22["inactivation_slope"], cav22["inactivation_tau"],
        activation_power, cav22["kinetic_q10"], cav22["reference_temperature"],
    )
    hcn_record = HCNParameters(
        hcn["conductance_density"],
        _density_ns_per_pf_to_ms_per_cm2(hcn["conductance_density"], capacitance, "hcn"),
        hcn["activation_half"], hcn["activation_slope"], hcn["activation_tau"],
        hcn["kinetic_q10"], hcn["reference_temperature"],
    )
    calcium_record = CalciumParameters(
        calcium["baseline"], calcium["decay_tau"], calcium["current_fraction"],
        calcium["kinetic_q10"], calcium["reference_temperature"],
    )
    sk_record = SKParameters(
        sk["conductance_density"],
        _density_ns_per_pf_to_ms_per_cm2(sk["conductance_density"], capacitance, "sk"),
        sk["half_activation"], sk["hill_coefficient"], sk["activation_tau"],
        sk["deactivation_tau"], sk["kinetic_q10"], sk["reference_temperature"],
    )
    geometry_record = GeometryParameters(
        _positive(geometry["membrane_area"], "geometry.membrane_area"),
        _positive(geometry["accessible_calcium_volume"], "geometry.accessible_calcium_volume"),
    )
    ionic_record = IonicEnvironmentParameters(
        ionic["sodium_reversal"], ionic["potassium_reversal"], ionic["leak_reversal"],
        ionic["nalcn_reversal"], ionic["calcium_reversal"], ionic["hcn_reversal"],
        ionic["extracellular_calcium"], ionic["calcium_valence"], ionic["simulation_temperature"],
    )
    q10_factors = Q10Factors(
        _q10_factor(fast["sodium_activation_q10"], FAST_HH_REFERENCE_TEMPERATURE_C,
                    ionic_record.simulation_temperature_c, "fast_hh.sodium_activation_q10"),
        _q10_factor(fast["sodium_inactivation_q10"], FAST_HH_REFERENCE_TEMPERATURE_C,
                    ionic_record.simulation_temperature_c, "fast_hh.sodium_inactivation_q10"),
        _q10_factor(fast["potassium_activation_q10"], FAST_HH_REFERENCE_TEMPERATURE_C,
                    ionic_record.simulation_temperature_c, "fast_hh.potassium_activation_q10"),
        _q10_factor(nap_record.kinetic_q10, nap_record.reference_temperature_c,
                    ionic_record.simulation_temperature_c, "nap.kinetic_q10"),
        _q10_factor(cav22_record.kinetic_q10, cav22_record.reference_temperature_c,
                    ionic_record.simulation_temperature_c, "cav22.kinetic_q10"),
        _q10_factor(hcn_record.kinetic_q10, hcn_record.reference_temperature_c,
                    ionic_record.simulation_temperature_c, "hcn.kinetic_q10"),
        _q10_factor(calcium_record.kinetic_q10, calcium_record.reference_temperature_c,
                    ionic_record.simulation_temperature_c, "calcium.kinetic_q10"),
        _q10_factor(sk_record.kinetic_q10, sk_record.reference_temperature_c,
                    ionic_record.simulation_temperature_c, "sk.kinetic_q10"),
    )
    calcium_influx = _calcium_influx_conversion(
        calcium_record.current_fraction,
        geometry_record.membrane_area_um2,
        geometry_record.accessible_calcium_volume_um3,
        ionic_record.calcium_valence,
    )
    return SNrPacketParameters(
        packet.packet_id, packet.packet_sha256, packet.structural_sha256,
        raw_groups, values, fast_record, NaLCNParameters(nalcn["conductance_density"]),
        nap_record, cav22_record, hcn_record, calcium_record, sk_record,
        geometry_record, ionic_record, q10_factors, calcium_influx,
    )


def _snapshot_and_parse(
    packet: MaterializedPacket,
) -> tuple[
    Mapping[str, Mapping[str, MaterializedParameterLeaf]],
    Mapping[str, Mapping[str, float]],
]:
    if not isinstance(packet.groups, Mapping) or set(packet.groups) != set(PARAMETER_SCHEMA):
        raise PacketParameterError("packet groups must exactly match PARAMETER_SCHEMA")
    raw_result: dict[str, Mapping[str, MaterializedParameterLeaf]] = {}
    value_result: dict[str, Mapping[str, float]] = {}
    for group, schema in PARAMETER_SCHEMA.items():
        group_leaves = packet.groups[group]
        if not isinstance(group_leaves, Mapping) or set(group_leaves) != set(schema):
            raise PacketParameterError(f"{group} leaves must exactly match PARAMETER_SCHEMA")
        raw_group: dict[str, MaterializedParameterLeaf] = {}
        value_group: dict[str, float] = {}
        for parameter, permitted_units in schema.items():
            leaf = group_leaves[parameter]
            if type(leaf) is not MaterializedParameterLeaf:
                raise PacketParameterError(f"{group}.{parameter} must be a MaterializedParameterLeaf")
            if leaf.unit not in permitted_units:
                raise PacketParameterError(
                    f"{group}.{parameter} has unsupported unit {leaf.unit!r}"
                )
            raw_group[parameter] = leaf
            value_group[parameter] = _parse_finite_float(leaf.value, f"{group}.{parameter}")
        raw_result[group] = MappingProxyType(raw_group)
        value_result[group] = MappingProxyType(value_group)
    return MappingProxyType(raw_result), MappingProxyType(value_result)


def _parse_finite_float(raw: object, name: str) -> float:
    if type(raw) is not str:
        raise PacketParameterError(f"{name} value must be a string")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PacketParameterError(f"{name} is not a finite float") from exc
    return _float32_compatible(value, name)


def _float32_compatible(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise PacketParameterError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value32 = np.float32(value)
    if not np.isfinite(value32):
        raise PacketParameterError(f"{name} overflows float32")
    if value != 0.0 and value32 == 0.0:
        raise PacketParameterError(f"{name} underflows float32")
    return value


def _positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise PacketParameterError(f"{name} must be > 0")
    return value


def _density_ns_per_pf_to_ms_per_cm2(
    density_ns_per_pf: float,
    capacitance_uf_per_cm2: float,
    group: str,
) -> float:
    # (nS / pF) * (uF / cm2) = mS / cm2 exactly by SI prefixes.
    return _float32_compatible(
        density_ns_per_pf * capacitance_uf_per_cm2,
        f"{group}.conductance_density_ms_per_cm2",
    )


def _q10_factor(q10: float, reference_c: float, simulation_c: float, name: str) -> float:
    _positive(q10, name)
    exponent = (simulation_c - reference_c) / 10.0
    try:
        factor = math.pow(q10, exponent)
    except (OverflowError, ValueError) as exc:
        raise PacketParameterError(f"{name} temperature factor is invalid") from exc
    factor = _float32_compatible(factor, f"{name} temperature factor")
    return _positive(factor, f"{name} temperature factor")


def _calcium_influx_conversion(
    current_fraction: float,
    membrane_area_um2: float,
    accessible_volume_um3: float,
    calcium_valence: float,
) -> float:
    if not 0.0 < current_fraction <= 1.0:
        raise PacketParameterError("calcium.current_fraction must be in (0, 1]")
    if not calcium_valence.is_integer() or not 1.0 <= calcium_valence <= 16.0:
        raise PacketParameterError("ionic_env.calcium_valence must be an integer in [1, 16]")
    # For one inward uA/cm2: uA/cm2 * um2 * ms = 1e-17 C.  Dividing
    # by zF yields mol; dividing by um3 * 1e-15 L and converting to uM
    # gives current_fraction * area * 1e4 / (z * F * volume) uM/ms.
    conversion = (
        current_fraction * membrane_area_um2 * 1.0e4
        / (calcium_valence * FARADAY_C_PER_MOL * accessible_volume_um3)
    )
    return _float32_compatible(
        conversion, "calcium influx uM/ms per inward uA/cm2"
    )


__all__ = [
    "FARADAY_C_PER_MOL",
    "FAST_HH_REFERENCE_TEMPERATURE_C",
    "PacketParameterError",
    "FastHHParameters",
    "NaLCNParameters",
    "NaPParameters",
    "Cav22Parameters",
    "HCNParameters",
    "CalciumParameters",
    "SKParameters",
    "GeometryParameters",
    "IonicEnvironmentParameters",
    "Q10Factors",
    "SNrPacketParameters",
    "materialize_runtime_parameters",
]
