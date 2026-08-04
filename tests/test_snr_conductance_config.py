"""Stage-A declarations for the region-scoped SNr conductance bundle."""
from __future__ import annotations

import dataclasses
import json
import math

import pytest

from sim.config import CoreSimConfig
from sim.regions import BrainRegion


REGIONAL_MAXIMUM_FIELDS = (
    "snr_g_nalcn_max",
    "snr_g_nap_max",
    "snr_g_ca_max",
    "snr_g_sk_max",
    "snr_g_h_max",
)

SHARED_CONSTANT_FIELDS = (
    "snr_E_nalcn",
    "snr_E_nap",
    "snr_E_ca",
    "snr_E_sk",
    "snr_E_h",
    "snr_calcium_influx_scale",
    "snr_calcium_decay_tau_ms",
    "snr_calcium_baseline",
    "snr_sk_calcium_half",
    "snr_sk_hill_coefficient",
    "snr_sk_activation_tau_ms",
)


def test_regional_conductance_maxima_default_to_disabled():
    region = BrainRegion(name="snr", n_neurons=8)

    assert all(getattr(region, name) == 0.0 for name in REGIONAL_MAXIMUM_FIELDS)
    assert region.snr_conductance_bundle_enabled is False


@pytest.mark.parametrize("enabled_field", REGIONAL_MAXIMUM_FIELDS)
def test_any_positive_regional_maximum_enables_bundle(enabled_field):
    region = BrainRegion(name="snr", n_neurons=8, **{enabled_field: 0.001})

    assert region.snr_conductance_bundle_enabled is True


@pytest.mark.parametrize("field_name", REGIONAL_MAXIMUM_FIELDS)
@pytest.mark.parametrize("invalid", [-0.001, math.inf, -math.inf, math.nan])
def test_regional_conductance_maxima_fail_closed(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        BrainRegion(name="snr", n_neurons=8, **{field_name: invalid})


def test_shared_constants_are_finite_and_serialize_with_regions():
    region = BrainRegion(
        name="snr",
        n_neurons=8,
        snr_g_nalcn_max=0.01,
        snr_g_nap_max=0.175,
        snr_g_ca_max=0.7,
        snr_g_sk_max=0.01,
        snr_g_h_max=0.05,
    )
    config = CoreSimConfig(brain_regions=[region])
    serialized = config.to_dict()

    for field_name in SHARED_CONSTANT_FIELDS:
        assert math.isfinite(serialized[field_name])
    assert serialized["brain_regions"][0] == dataclasses.asdict(region)
    assert json.loads(json.dumps(serialized))["brain_regions"][0][
        "snr_g_nalcn_max"
    ] == 0.01


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("snr_E_nalcn", -100.01),
        ("snr_E_nap", -0.01),
        ("snr_E_ca", 250.01),
        ("snr_E_sk", -39.99),
        ("snr_E_h", 50.01),
        ("snr_calcium_influx_scale", -0.01),
        ("snr_calcium_decay_tau_ms", 0.0),
        ("snr_calcium_baseline", -0.01),
        ("snr_sk_calcium_half", 0.0),
        ("snr_sk_hill_coefficient", 0.0),
        ("snr_sk_activation_tau_ms", 0.0),
    ],
)
def test_shared_constants_reject_out_of_range_values(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        CoreSimConfig(**{field_name: invalid})


@pytest.mark.parametrize("field_name", SHARED_CONSTANT_FIELDS)
@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_shared_constants_reject_nonfinite_values(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        CoreSimConfig(**{field_name: invalid})
