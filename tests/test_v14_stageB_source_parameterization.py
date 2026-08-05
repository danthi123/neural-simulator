from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim import kv3_source_models as kv3
from sim import sodium_source_models as sodium
from tools import v14_stageB_source_parameterization as p


MODELS = (
    sodium.KHALIQ_RAMAN_13_STATE,
    sodium.BALBI_NAV16_SIX_STATE,
    kv3.LABRO_2015,
    kv3.DESAI_2008_CONTROL,
)
ROOT = Path(__file__).resolve().parents[1]


def _defaults(model_id):
    module = sodium if model_id in sodium.SOURCE_PARAMETER_DEFAULTS else kv3
    return module.SOURCE_PARAMETER_DEFAULTS[model_id]


@pytest.mark.parametrize("model_id", MODELS)
def test_zero_contrast_round_trips_exact_source_vector(model_id):
    width = len(p.coordinates(model_id))
    decoded = p.decode(model_id, np.zeros(width))
    assert dict(decoded) == dict(_defaults(model_id))
    np.testing.assert_array_equal(p.encode(model_id, decoded), np.zeros(width))


@pytest.mark.parametrize("model_id", MODELS)
def test_nonzero_contrasts_round_trip_and_preserve_source_constraints(model_id):
    declared = p.coordinates(model_id)
    contrast = np.linspace(-0.2, 0.2, len(declared))
    decoded = p.decode(model_id, contrast)
    np.testing.assert_allclose(p.encode(model_id, decoded), contrast, rtol=0.0, atol=2e-14)
    for item in declared:
        value = decoded[item.parameter]
        if item.component is not None:
            value = value[item.component]
        if item.transform == p.LOG_POSITIVE:
            assert value > 0.0
        if item.transform == p.SIGNED_LOG:
            assert np.sign(value) == np.sign(item.source_value)


def test_khaliq_voltage_independent_sentinels_are_fixed():
    model = sodium.KHALIQ_RAMAN_13_STATE
    decoded = p.decode(model, np.full(len(p.coordinates(model)), 0.3))
    for name in ("x3_mv", "x4_mv", "x5_mv"):
        assert decoded[name] == sodium.SOURCE_PARAMETER_DEFAULTS[model][name]
    document = p.parameterization_document(model)
    assert document["fixed_parameters"] == ["x3_mv", "x4_mv", "x5_mv"]
    assert document["numeric_biological_bounds_available"] is False
    assert all(row["lower_bound"] is None for row in document["coordinates"])
    assert all(row["upper_bound"] is None for row in document["coordinates"])


def test_balbi_global_temperature_law_is_fixed_and_additive_relations_remain_in_graph():
    model = sodium.BALBI_NAV16_SIX_STATE
    decoded = p.decode(model, np.full(len(p.coordinates(model)), -0.1))
    defaults = sodium.SOURCE_PARAMETER_DEFAULTS[model]
    assert decoded["q10"] == defaults["q10"]
    assert decoded["q10_reference_temperature_c"] == defaults["q10_reference_temperature_c"]
    assert p.parameterization_document(model)["fixed_parameters"] == [
        "q10", "q10_reference_temperature_c"
    ]


@pytest.mark.parametrize("model_id", MODELS)
def test_batch_decode_matches_individual_decode(model_id):
    width = len(p.coordinates(model_id))
    matrix = np.stack((np.zeros(width), np.full(width, 0.05)))
    observed = p.decode_batch(model_id, matrix)
    assert [dict(item) for item in observed] == [
        dict(p.decode(model_id, row)) for row in matrix
    ]


def test_invalid_models_shapes_nonfinite_and_sign_changes_fail_closed():
    with pytest.raises(p.SourceParameterizationError, match="unknown"):
        p.coordinates("not-a-model")
    model = sodium.KHALIQ_RAMAN_13_STATE
    width = len(p.coordinates(model))
    with pytest.raises(p.SourceParameterizationError, match="shape"):
        p.decode(model, np.zeros(width - 1))
    invalid = np.zeros(width)
    invalid[0] = np.nan
    with pytest.raises(p.SourceParameterizationError, match="finite"):
        p.decode(model, invalid)
    too_large = np.zeros(width)
    too_large[0] = 1e6
    with pytest.raises(p.SourceParameterizationError, match="exceeds finite"):
        p.decode(model, too_large)
    changed = dict(sodium.SOURCE_PARAMETER_DEFAULTS[model])
    changed["x1_mv"] *= -1
    with pytest.raises(p.SourceParameterizationError, match="source sign"):
        p.encode(model, changed)
    with pytest.raises(p.SourceParameterizationError, match="nonempty shape"):
        p.decode_batch(model, np.zeros((0, width)))


def test_prospective_spec_binds_implementation_and_forbids_invented_bounds():
    path = ROOT / "research/specs/v14_snr_stageB_fitted_arm_parameterization_v1.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    assert spec["status"] == "preregistered_before_three_way_target_adjudication_and_candidate_fitting"
    assert spec["optimization_allowed"] is False
    assert spec["scientific_verdict"] is None
    assert spec["coordinate_contract"]["numeric_biological_bounds_available"] is False
    assert spec["calibration_sensitivity_scale"]["selected_step_is_a_biological_bound"] is False
    assert spec["identifiable_subspace"]["validation_or_held_out_data_may_select_subspace"] is False
    assert spec["model_ids"] == list(MODELS)
    for binding in spec["authorities"].values():
        authority = ROOT / binding["path"]
        assert authority.is_file() and not authority.is_symlink()
        assert hashlib.sha256(authority.read_bytes()).hexdigest() == binding["sha256"]
