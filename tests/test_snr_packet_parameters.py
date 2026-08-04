"""Tests for CPU-only typed SNr executable-packet conversion."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

import sim.snr_packet_parameters as parameters
from sim.snr_executable_packet import (
    AuthorityKind,
    EvidenceKind,
    MaterializedPacket,
    MaterializedParameterLeaf,
    MaterializedUncertainty,
    PARAMETER_SCHEMA,
    UncertaintyKind,
)


def _value(group: str, parameter: str) -> str:
    values = {
        ("fast_hh", "capacitance_density"): "2",
        ("fast_hh", "initial_voltage"): "-65",
        ("fast_hh", "spike_detection_voltage"): "-20",
        ("fast_hh", "initial_m"): "0.1",
        ("fast_hh", "initial_h"): "0.6",
        ("fast_hh", "initial_n"): "0.3",
        ("fast_hh", "sodium_activation_q10"): "2",
        ("fast_hh", "sodium_inactivation_q10"): "3",
        ("fast_hh", "potassium_activation_q10"): "4",
        ("nap", "conductance_density"): "0.5",
        ("cav22", "conductance_density"): "0.25",
        ("cav22", "activation_power"): "2",
        ("hcn", "conductance_density"): "0.125",
        ("calcium", "baseline"): "0.1",
        ("calcium", "current_fraction"): "0.5",
        ("sk", "conductance_density"): "0.75",
        ("geometry", "membrane_area"): "1000",
        ("geometry", "accessible_calcium_volume"): "200",
        ("ionic_env", "sodium_reversal"): "50",
        ("ionic_env", "potassium_reversal"): "-90",
        ("ionic_env", "leak_reversal"): "-65",
        ("ionic_env", "nalcn_reversal"): "0",
        ("ionic_env", "calcium_reversal"): "120",
        ("ionic_env", "hcn_reversal"): "-30",
        ("ionic_env", "extracellular_calcium"): "2000",
        ("ionic_env", "calcium_valence"): "2",
        ("ionic_env", "simulation_temperature"): "16.3",
    }
    if (group, parameter) in values:
        return values[(group, parameter)]
    if parameter == "reference_temperature":
        return "6.3"
    if parameter.endswith("q10"):
        return "2"
    return "1"


def _leaf(value: str, unit: str) -> MaterializedParameterLeaf:
    return MaterializedParameterLeaf(
        value=value,
        unit=unit,
        uncertainty=MaterializedUncertainty(
            UncertaintyKind.INTERVAL, "0", "1", unit
        ),
        evidence_kind=EvidenceKind.MEASURED,
        authority_kind=AuthorityKind.PRIMARY_SOURCE,
    )


def _packet(overrides: dict[tuple[str, str], str] | None = None) -> MaterializedPacket:
    overrides = overrides or {}
    groups: dict[str, MappingProxyType] = {}
    for group, schema in PARAMETER_SCHEMA.items():
        groups[group] = MappingProxyType(
            {
                parameter: _leaf(
                    overrides.get((group, parameter), _value(group, parameter)),
                    next(iter(units)),
                )
                for parameter, units in schema.items()
            }
        )
    return MaterializedPacket(
        "snr-runtime-parameters-test", "a" * 64, "b" * 64,
        MappingProxyType(groups),
    )


def test_all_69_schema_leaves_are_parsed_once_and_preserved(monkeypatch):
    packet = _packet()
    calls: list[str] = []
    original = parameters._parse_finite_float

    def tracked(raw, name):
        calls.append(name)
        return original(raw, name)

    monkeypatch.setattr(parameters, "_parse_finite_float", tracked)
    runtime = parameters.materialize_runtime_parameters(packet)

    expected = {
        f"{group}.{parameter}"
        for group, schema in PARAMETER_SCHEMA.items()
        for parameter in schema
    }
    assert sum(len(schema) for schema in PARAMETER_SCHEMA.values()) == 69
    assert calls == [
        f"{group}.{parameter}"
        for group, schema in PARAMETER_SCHEMA.items()
        for parameter in schema
    ]
    assert len(calls) == len(expected) == 69
    assert set(runtime.raw_groups) == set(PARAMETER_SCHEMA)
    assert set(runtime.parsed_values) == set(PARAMETER_SCHEMA)
    assert {
        f"{group}.{parameter}"
        for group, leaves in runtime.raw_groups.items()
        for parameter in leaves
    } == expected
    assert {
        f"{group}.{parameter}"
        for group, values in runtime.parsed_values.items()
        for parameter in values
    } == expected
    assert runtime.parsed_values["nap"]["activation_half"] == 1.0
    assert runtime.raw_groups["nap"]["activation_half"] is packet.groups["nap"]["activation_half"]


def test_runtime_record_is_deeply_immutable():
    runtime = parameters.materialize_runtime_parameters(_packet())
    with pytest.raises(FrozenInstanceError):
        runtime.nap = runtime.nap
    with pytest.raises(FrozenInstanceError):
        runtime.nap.activation_half_mv = -50.0
    with pytest.raises(TypeError):
        runtime.raw_groups["nap"] = MappingProxyType({})
    with pytest.raises(TypeError):
        runtime.raw_groups["nap"]["activation_half"] = _leaf("2", "mV")
    with pytest.raises(TypeError):
        runtime.parsed_values["nap"]["activation_half"] = 2.0


def test_known_unit_temperature_and_calcium_conversions():
    runtime = parameters.materialize_runtime_parameters(_packet())

    # (nS/pF) * (uF/cm2) equals mS/cm2 by the SI prefixes.
    assert runtime.nap.conductance_density_ms_per_cm2 == pytest.approx(1.0)
    assert runtime.cav22.conductance_density_ms_per_cm2 == pytest.approx(0.5)
    assert runtime.hcn.conductance_density_ms_per_cm2 == pytest.approx(0.25)
    assert runtime.sk.conductance_density_ms_per_cm2 == pytest.approx(1.5)
    assert runtime.q10_factors.fast_hh_sodium_activation == pytest.approx(2.0)
    assert runtime.q10_factors.fast_hh_sodium_inactivation == pytest.approx(3.0)
    assert runtime.q10_factors.fast_hh_potassium_activation == pytest.approx(4.0)
    assert runtime.q10_factors.nap == pytest.approx(2.0)
    assert runtime.q10_factors.cav22 == pytest.approx(2.0)
    assert runtime.q10_factors.hcn == pytest.approx(2.0)
    assert runtime.q10_factors.calcium == pytest.approx(2.0)
    assert runtime.q10_factors.sk == pytest.approx(2.0)
    expected_influx = 0.5 * 1000.0 * 1.0e4 / (2.0 * parameters.FARADAY_C_PER_MOL * 200.0)
    assert runtime.calcium_influx_um_per_ms_per_inward_ua_per_cm2 == pytest.approx(expected_influx)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({("nap", "activation_half"): "nan"}, "must be finite"),
        ({("nap", "activation_half"): "1e1000"}, "must be finite"),
        ({("nap", "activation_half"): "1e-100"}, "underflows float32"),
        ({("cav22", "activation_power"): "1.5"}, "activation_power"),
        ({("geometry", "membrane_area"): "0"}, "membrane_area"),
        ({("calcium", "current_fraction"): "0"}, "current_fraction"),
        ({("ionic_env", "calcium_valence"): "1.5"}, "calcium_valence"),
    ],
)
def test_invalid_runtime_values_fail_closed(overrides, match):
    with pytest.raises(parameters.PacketParameterError, match=match):
        parameters.materialize_runtime_parameters(_packet(overrides))


def test_temperature_factor_overflow_fails_closed():
    packet = _packet(
        {
            ("nap", "kinetic_q10"): "1e30",
            ("nap", "reference_temperature"): "-50",
            ("ionic_env", "simulation_temperature"): "100",
        }
    )
    with pytest.raises(parameters.PacketParameterError, match="temperature factor"):
        parameters.materialize_runtime_parameters(packet)


def test_derived_conductance_overflow_fails_closed():
    packet = _packet(
        {
            ("fast_hh", "capacitance_density"): "1e20",
            ("nap", "conductance_density"): "1e30",
        }
    )
    with pytest.raises(parameters.PacketParameterError, match="conductance_density_ms_per_cm2"):
        parameters.materialize_runtime_parameters(packet)


def test_non_exact_packet_type_and_schema_mismatch_are_rejected():
    with pytest.raises(TypeError, match="exact MaterializedPacket"):
        parameters.materialize_runtime_parameters(object())

    packet = _packet()
    bad_groups = dict(packet.groups)
    bad_nap = dict(bad_groups["nap"])
    bad_nap.pop("activation_half")
    bad_groups["nap"] = MappingProxyType(bad_nap)
    malformed = MaterializedPacket(
        packet.packet_id, packet.packet_sha256, packet.structural_sha256,
        MappingProxyType(bad_groups),
    )
    with pytest.raises(parameters.PacketParameterError, match="exactly match"):
        parameters.materialize_runtime_parameters(malformed)
