from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from sim.snr_channel_parameters import (
    CalciumParameters,
    EvidenceClass,
    HCNParameters,
    NaPParameters,
    ParameterProvenance,
    Q10Provenance,
    SKParameters,
    calcium_concentration_step,
    calcium_influx_delta_um,
    channel_gates,
    kinetic_temperature_factor,
    legacy_cav22_gates,
    legacy_hcn_gate,
    legacy_nap_gates,
    phillips_model_prior_packet,
    phillips_nap_gates,
    sk_activation_step,
    sk_steady_state,
)


def unresolved_provenance(reference_temperature_celsius=37.0):
    return ParameterProvenance(
        parameter_set_id="test-unresolved-v1",
        evidence_class=EvidenceClass.UNRESOLVED,
        source_locator="test locator",
        reference_temperature_celsius=reference_temperature_celsius,
    )


def test_records_are_immutable_and_versioned():
    packet = phillips_model_prior_packet()
    assert packet.schema_version == "snr-channel-parameters-v1"
    assert packet.nap.provenance.evidence_class is EvidenceClass.MODEL_PRIOR
    assert "computational model" in packet.nap.provenance.model_name
    with pytest.raises(FrozenInstanceError):
        packet.nap.activation_half_mv = -49.0


@pytest.mark.parametrize("evidence", [EvidenceClass.MEASURED, EvidenceClass.UNRESOLVED])
def test_model_name_is_rejected_outside_model_priors(evidence):
    with pytest.raises(ValueError, match="only valid"):
        ParameterProvenance("id", evidence, "source", 37.0, model_name="model")


def test_untagged_model_prior_is_rejected():
    with pytest.raises(ValueError, match="model_name"):
        ParameterProvenance(
            "id", EvidenceClass.MODEL_PRIOR, "paper table", None
        )


def test_raw_evidence_strings_and_untagged_q10_objects_are_rejected():
    with pytest.raises(TypeError, match="EvidenceClass"):
        ParameterProvenance("id", "model-prior", "source", None, model_name="model")
    with pytest.raises(TypeError, match="Q10Provenance"):
        ParameterProvenance(
            "id", EvidenceClass.MEASURED, "source", 37.0, q10=2.0
        )
    with pytest.raises(TypeError, match="EvidenceClass"):
        Q10Provenance(2.0, "measured", "source")


def test_q10_requires_resolved_provenance():
    with pytest.raises(ValueError, match="unresolved Q10"):
        Q10Provenance(2.0, EvidenceClass.UNRESOLVED, "unknown")
    with pytest.raises(ValueError, match="source_locator"):
        Q10Provenance(2.0, EvidenceClass.MEASURED, "")


def test_temperature_mismatch_fails_without_q10():
    provenance = unresolved_provenance(37.0)
    assert kinetic_temperature_factor(provenance, None) == 1.0
    assert kinetic_temperature_factor(provenance, 37.0) == 1.0
    with pytest.raises(ValueError, match="provenance-tagged Q10"):
        kinetic_temperature_factor(provenance, 30.0)


def test_unknown_reference_temperature_fails_for_requested_temperature():
    provenance = replace(
        phillips_model_prior_packet().nap.provenance,
        q10=Q10Provenance(2.0, EvidenceClass.MODEL_PRIOR, "Q10 model prior"),
    )
    with pytest.raises(ValueError, match="reference temperature is unresolved"):
        kinetic_temperature_factor(provenance, 37.0)


def test_provenance_bound_q10_scales_kinetic_speed():
    provenance = ParameterProvenance(
        "measured-at-27-v1",
        EvidenceClass.MEASURED,
        "primary source table",
        27.0,
        q10=Q10Provenance(2.0, EvidenceClass.MEASURED, "primary source Q10"),
    )
    assert kinetic_temperature_factor(provenance, 37.0) == pytest.approx(2.0)


def test_exact_legacy_gate_references():
    voltages = np.array([-75.0, -57.0, -50.0, -27.5])
    nap_m, nap_h, nap_tau_m, nap_tau_h = legacy_nap_gates(voltages)
    np.testing.assert_allclose(nap_m, 1 / (1 + np.exp(-(voltages + 50) / 4.5)))
    np.testing.assert_allclose(nap_h, 1 / (1 + np.exp((voltages + 57) / 6.0)))
    assert (nap_tau_m, nap_tau_h) == (0.1, 20.0)

    ca_m, ca_h, ca_tau_m, ca_tau_h = legacy_cav22_gates(voltages)
    np.testing.assert_allclose(ca_m, 1 / (1 + np.exp(-(voltages + 27.5) / 3.0)))
    np.testing.assert_allclose(ca_h, 1 / (1 + np.exp((voltages + 52.5) / 5.2)))
    assert (ca_tau_m, ca_tau_h) == (0.5, 18.0)

    h, tau_h = legacy_hcn_gate(voltages)
    np.testing.assert_allclose(h, 1 / (1 + np.exp((voltages + 75.0) / 5.5)))
    assert tau_h == 100.0


def test_phillips_nap_published_prior_reference_values():
    packet = phillips_model_prior_packet()
    voltage = np.array([-57.0, -50.0, -42.6, -34.0])
    m, h, tau_m, tau_h = phillips_nap_gates(voltage, packet.nap)
    np.testing.assert_allclose(m, 1 / (1 + np.exp(-(voltage + 50.0) / 3.0)))
    np.testing.assert_allclose(h, 1 / (1 + np.exp((voltage + 57.0) / 4.0)))

    expected_m = 0.03 + (0.146 - 0.03) / (
        np.exp((-42.6 - voltage) / 14.4)
        + np.exp((-42.6 - voltage) / -14.4)
    )
    expected_h = 10.0 + (17.0 - 10.0) / (
        np.exp((-34.0 - voltage) / 26.0)
        + np.exp((-34.0 - voltage) / -31.9)
    )
    np.testing.assert_allclose(tau_m, expected_m)
    np.testing.assert_allclose(tau_h, expected_h)
    assert packet.nap.conductance_ns_per_pf == 0.175


def test_phillips_function_rejects_non_phillips_authority():
    prior = phillips_model_prior_packet().nap
    wrong = replace(
        prior,
        provenance=replace(prior.provenance, model_name="some other model"),
    )
    with pytest.raises(ValueError, match="Phillips model provenance"):
        phillips_nap_gates(-50.0, wrong)


def test_fixed_channel_prior_reference_values():
    packet = phillips_model_prior_packet()
    ca_m, ca_h, ca_tau_m, ca_tau_h = channel_gates(-27.5, packet.cav22)
    assert float(ca_m) == pytest.approx(0.5)
    assert float(ca_h) == pytest.approx(1 / (1 + math.exp((-27.5 + 52.5) / 5.2)))
    assert ca_tau_m == 0.5
    assert ca_tau_h == 18.0

    h, tau = channel_gates(-75.0, packet.hcn)
    assert float(h) == pytest.approx(0.5)
    assert tau == 100.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("activation_slope_mv", 0.0, "nonzero"),
        ("activation_tau_min_ms", 0.0, "> 0"),
        ("inactivation_tau_sigma_0_mv", 0.0, "nonzero"),
    ],
)
def test_nap_rejects_nonphysical_slopes_and_times(field, value, match):
    with pytest.raises(ValueError, match=match):
        replace(phillips_model_prior_packet().nap, **{field: value})


def test_partial_voltage_dependent_tau_is_rejected():
    base = phillips_model_prior_packet().nap
    with pytest.raises(ValueError, match="all present or all absent"):
        NaPParameters(
            provenance=base.provenance,
            activation_half_mv=-50,
            activation_slope_mv=3,
            activation_tau_min_ms=0.03,
            activation_tau_max_ms=0.146,
            activation_tau_half_mv=-42.6,
        )


@pytest.mark.parametrize(
    "area,volume",
    [(0.0, 100.0), (-1.0, 100.0), (100.0, 0.0), (100.0, math.inf)],
)
def test_calcium_conversion_rejects_nonphysical_geometry(area, volume):
    with pytest.raises(ValueError):
        calcium_influx_delta_um(
            1.0, 0.1, membrane_area_um2=area, accessible_volume_um3=volume
        )


def test_calcium_conversion_matches_charge_and_volume_equation():
    result = calcium_influx_delta_um(
        20.0,
        0.25,
        membrane_area_um2=1000.0,
        accessible_volume_um3=200.0,
    )
    expected_charge_c = 20.0 * 1000.0 * 0.25 * 1e-17
    expected_um = (
        expected_charge_c / (2 * 96485.33212) / (200.0e-15) * 1e6
    )
    assert float(result) == pytest.approx(expected_um)


def test_calcium_step_uses_physical_influx_and_decay():
    calcium = phillips_model_prior_packet().calcium
    new = calcium_concentration_step(
        calcium.baseline_um,
        10.0,
        0.1,
        calcium,
        membrane_area_um2=500.0,
        accessible_volume_um3=100.0,
    )
    assert float(new) > calcium.baseline_um
    decayed = calcium_concentration_step(
        1.0,
        0.0,
        10.0,
        calcium,
        membrane_area_um2=500.0,
        accessible_volume_um3=100.0,
    )
    assert calcium.baseline_um < float(decayed) < 1.0


def test_arbitrary_calcium_units_are_rejected_for_physical_sk():
    provenance = unresolved_provenance()
    with pytest.raises(ValueError, match="micromolar"):
        CalciumParameters(provenance, 0.0, 80.0, 2000.0, "arbitrary")
    with pytest.raises(ValueError, match="micromolar"):
        SKParameters(provenance, 0.5, 4.0, 5.0, 5.0, "arbitrary")
    with pytest.raises(ValueError, match="cannot consume arbitrary"):
        sk_activation_step(
            0.0,
            1.0,
            0.1,
            phillips_model_prior_packet().sk,
            calcium_units="arbitrary",
        )


def test_sk_uses_separate_activation_and_deactivation_times():
    sk = phillips_model_prior_packet().sk
    opened = sk_activation_step(
        0.0, 6.2, 4.1, sk, calcium_units="micromolar"
    )
    steady_open = sk_steady_state(6.2, sk)
    assert float(opened) == pytest.approx(float(steady_open) * (1 - math.exp(-1)))

    closed = sk_activation_step(
        1.0, 0.0, 57.3, sk, calcium_units="micromolar"
    )
    assert float(closed) == pytest.approx(math.exp(-1))


def test_parameter_domain_validation():
    provenance = unresolved_provenance()
    with pytest.raises(ValueError, match="nonzero"):
        HCNParameters(provenance, -75, 0, 100, -30)
    with pytest.raises(ValueError, match="> 0"):
        SKParameters(provenance, 0.62, 3.2, -1, 57.3)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        CalciumParameters(provenance, 0, 250, 4000, current_fraction=1.1)


def test_numpy_cupy_parity_on_real_backend_when_available():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy is installed but no CUDA device is available")

    voltages_np = np.linspace(-100.0, 20.0, 257)
    voltages_cp = cp.asarray(voltages_np)
    packet = phillips_model_prior_packet()

    numpy_outputs = phillips_nap_gates(voltages_np, packet.nap, xp=np)
    cupy_outputs = phillips_nap_gates(voltages_cp, packet.nap, xp=cp)
    for numpy_value, cupy_value in zip(numpy_outputs, cupy_outputs):
        np.testing.assert_allclose(cp.asnumpy(cupy_value), numpy_value, rtol=1e-12, atol=1e-12)

    numpy_calcium = calcium_concentration_step(
        np.zeros(8),
        np.linspace(0, 20, 8),
        0.05,
        packet.calcium,
        membrane_area_um2=900,
        accessible_volume_um3=120,
        xp=np,
    )
    cupy_calcium = calcium_concentration_step(
        cp.zeros(8),
        cp.linspace(0, 20, 8),
        0.05,
        packet.calcium,
        membrane_area_um2=900,
        accessible_volume_um3=120,
        xp=cp,
    )
    np.testing.assert_allclose(cp.asnumpy(cupy_calcium), numpy_calcium, rtol=1e-12, atol=1e-12)

    numpy_sk = sk_activation_step(
        np.linspace(0, 1, 8),
        numpy_calcium,
        0.05,
        packet.sk,
        calcium_units="micromolar",
        xp=np,
    )
    cupy_sk = sk_activation_step(
        cp.linspace(0, 1, 8),
        cupy_calcium,
        0.05,
        packet.sk,
        calcium_units="micromolar",
        xp=cp,
    )
    np.testing.assert_allclose(cp.asnumpy(cupy_sk), numpy_sk, rtol=1e-12, atol=1e-12)
