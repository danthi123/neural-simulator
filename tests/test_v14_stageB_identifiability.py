from __future__ import annotations

import copy

import pytest

from tools.v14_stageB_identifiability import (
    StageBIdentifiabilityError,
    diagnose_identifiability,
    digest,
)


def _sealed_space() -> dict:
    body = {
        "schema": "v14-snr-stageB-identifiability-parameter-space-v1",
        "id": "stage-b-fit-space-v1",
        "status": "sealed",
        "parameters": [
            {"id": "g_kv3", "low": 0.0, "high": 2.0, "scale": 1.0},
            {"id": "g_na", "low": 0.0, "high": 2.0, "scale": 1.0},
        ],
        "allowed_fit_partitions": ["calibration"],
        "thresholds": {
            "min_completed_fits": 2,
            "min_near_equivalent_fits": 3,
            "min_perturbation_pairs": 2,
            "near_equivalent_objective_l2": 0.01,
            "svd_rank_relative_tolerance": 1e-10,
            "max_condition_number_identified": 10.0,
            "max_relative_span_identified": 0.25,
            "max_abs_correlation_identified": 0.95,
            "max_scaled_perturbation_l2": 0.25,
        },
        "scientific_verdict": None,
        "optimization_allowed": False,
    }
    return {**body, "sha256": digest(body)}


def _fit(space: dict, identifier: str, parameters: dict, residuals: dict, partition: str = "calibration") -> dict:
    body = {
        "schema": "v14-snr-stageB-completed-fit-v1",
        "id": identifier,
        "status": "completed",
        "parameter_space": {"id": space["id"], "sha256": space["sha256"]},
        "partition": partition,
        "parameters": parameters,
        "objective_residuals": residuals,
        "scientific_verdict": None,
        "optimization_allowed": False,
    }
    return {**body, "sha256": digest(body)}


def _pairs(space: dict, pairs: list[dict]) -> dict:
    body = {
        "schema": "v14-snr-stageB-identifiability-perturbations-v1",
        "parameter_space": {"id": space["id"], "sha256": space["sha256"]},
        "pairs": pairs,
        "scientific_verdict": None,
        "optimization_allowed": False,
    }
    return {**body, "sha256": digest(body)}


def _evidence() -> tuple[dict, list[dict], dict]:
    space = _sealed_space()
    fits = [
        _fit(space, "base", {"g_na": 1.0, "g_kv3": 1.0}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "na-plus", {"g_na": 1.1, "g_kv3": 1.0}, {"activation": 0.1, "kinetics": 0.0}),
        _fit(space, "kv3-plus", {"g_na": 1.0, "g_kv3": 1.1}, {"activation": 0.0, "kinetics": 0.1}),
        _fit(space, "alt-na", {"g_na": 1.1, "g_kv3": 1.0}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "alt-kv3", {"g_na": 1.0, "g_kv3": 1.1}, {"activation": 0.0, "kinetics": 0.0}),
    ]
    pairs = _pairs(space, [
        {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus"},
        {"baseline_fit_id": "base", "perturbed_fit_id": "kv3-plus"},
    ])
    return space, fits, pairs


def test_diagnostic_is_canonical_and_order_independent_for_identified_fit() -> None:
    space, fits, pairs = _evidence()
    forward = diagnose_identifiability(space, fits, pairs)
    reverse = diagnose_identifiability(space, list(reversed(fits)), pairs)

    assert forward == reverse
    assert forward["sha256"] == digest({key: value for key, value in forward.items() if key != "sha256"})
    assert forward["scientific_verdict"] is None
    assert forward["optimization_allowed"] is False
    assert forward["diagnostic_status"] == "sufficient_evidence"
    assert forward["jacobian"]["rank"] == 2
    assert [row["classification"] for row in forward["parameters"]] == ["identified", "identified"]


@pytest.mark.parametrize("which", ["space", "fit", "pairs"])
def test_rejects_malformed_self_digests(which: str) -> None:
    space, fits, pairs = _evidence()
    if which == "space":
        space["sha256"] = "0" * 64
    elif which == "fit":
        fits[0]["sha256"] = "0" * 64
    else:
        pairs["sha256"] = "0" * 64
    with pytest.raises(StageBIdentifiabilityError, match="self digest"):
        diagnose_identifiability(space, fits, pairs)


def test_rejects_held_out_leakage_even_when_fit_is_redigested() -> None:
    space, fits, pairs = _evidence()
    leaked = copy.deepcopy(fits[0])
    leaked["partition"] = "held_out"
    leaked["sha256"] = digest({key: value for key, value in leaked.items() if key != "sha256"})
    with pytest.raises(StageBIdentifiabilityError, match="held-out"):
        diagnose_identifiability(space, [leaked, *fits[1:]], pairs)


def test_rejects_validation_as_identifiability_evidence() -> None:
    space, fits, pairs = _evidence()
    leaked = copy.deepcopy(fits[0])
    leaked["partition"] = "validation"
    leaked["sha256"] = digest({key: value for key, value in leaked.items() if key != "sha256"})
    with pytest.raises(StageBIdentifiabilityError, match="unauthorized"):
        diagnose_identifiability(space, [leaked, *fits[1:]], pairs)


def test_nonlocal_or_mixed_baseline_probes_are_insufficient() -> None:
    space, fits, _ = _evidence()
    distant = _fit(space, "distant", {"g_na": 1.5, "g_kv3": 1.0}, {"activation": 0.5, "kinetics": 0.0})
    result = diagnose_identifiability(
        space,
        [*fits, distant],
        _pairs(space, [
            {"baseline_fit_id": "base", "perturbed_fit_id": "distant"},
            {"baseline_fit_id": "base", "perturbed_fit_id": "kv3-plus"},
        ]),
    )
    assert result["jacobian"]["reason"] == "perturbation_exceeds_preregistered_local_radius"
    mixed = diagnose_identifiability(
        space,
        fits,
        _pairs(space, [
            {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus"},
            {"baseline_fit_id": "alt-na", "perturbed_fit_id": "kv3-plus"},
        ]),
    )
    assert mixed["jacobian"]["reason"] == "perturbation_pairs_do_not_share_one_local_baseline"


def test_never_claims_identifiability_from_one_fit_or_missing_pairs() -> None:
    space, fits, pairs = _evidence()
    one = diagnose_identifiability(space, [fits[0]], _pairs(space, []))
    no_pairs = diagnose_identifiability(space, fits, _pairs(space, []))

    for result in (one, no_pairs):
        assert result["diagnostic_status"] == "insufficient_evidence"
        assert result["jacobian"]["status"] == "insufficient_evidence"
        assert {row["classification"] for row in result["parameters"]} == {"unresolved"}


def test_rank_deficient_local_jacobian_remains_unresolved() -> None:
    space, fits, _ = _evidence()
    collinear = _pairs(space, [
        {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus"},
        {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus"},
    ])
    with pytest.raises(StageBIdentifiabilityError, match="unique"):
        diagnose_identifiability(space, fits, collinear)

    extra = _fit(space, "na-plus-two", {"g_na": 1.2, "g_kv3": 1.0}, {"activation": 0.2, "kinetics": 0.0})
    collinear = _pairs(space, [
        {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus"},
        {"baseline_fit_id": "base", "perturbed_fit_id": "na-plus-two"},
    ])
    result = diagnose_identifiability(space, [*fits, extra], collinear)
    assert result["diagnostic_status"] == "insufficient_evidence"
    assert result["jacobian"]["reason"] == "perturbation_directions_do_not_span_parameter_space"
    assert {row["classification"] for row in result["parameters"]} == {"unresolved"}

    rank_deficient = [
        _fit(space, "base-rank", {"g_na": 1.0, "g_kv3": 1.0}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "na-rank", {"g_na": 1.1, "g_kv3": 1.0}, {"activation": 0.1, "kinetics": 0.0}),
        _fit(space, "kv3-rank", {"g_na": 1.0, "g_kv3": 1.1}, {"activation": 0.1, "kinetics": 0.0}),
        _fit(space, "alt-na-rank", {"g_na": 1.1, "g_kv3": 1.0}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "alt-kv3-rank", {"g_na": 1.0, "g_kv3": 1.1}, {"activation": 0.0, "kinetics": 0.0}),
    ]
    rank_pairs = _pairs(space, [
        {"baseline_fit_id": "base-rank", "perturbed_fit_id": "na-rank"},
        {"baseline_fit_id": "base-rank", "perturbed_fit_id": "kv3-rank"},
    ])
    result = diagnose_identifiability(space, rank_deficient, rank_pairs)
    assert result["diagnostic_status"] == "sufficient_evidence"
    assert result["jacobian"]["rank"] == 1
    assert {row["classification"] for row in result["parameters"]} == {"unresolved"}


def test_near_equivalent_fits_expose_spans_and_correlations_as_weak() -> None:
    space, fits, pairs = _evidence()
    # Two equally good fits trade the two parameters together; the Jacobian is
    # full-rank locally, but the ensemble prevents an identified verdict.
    fits = [
        _fit(space, "base", {"g_na": 0.8, "g_kv3": 0.8}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "na-plus", {"g_na": 0.9, "g_kv3": 0.8}, {"activation": 0.1, "kinetics": 0.0}),
        _fit(space, "kv3-plus", {"g_na": 0.8, "g_kv3": 0.9}, {"activation": 0.0, "kinetics": 0.1}),
        _fit(space, "trade", {"g_na": 1.2, "g_kv3": 1.2}, {"activation": 0.0, "kinetics": 0.0}),
        _fit(space, "trade-two", {"g_na": 1.3, "g_kv3": 1.3}, {"activation": 0.0, "kinetics": 0.0}),
    ]
    result = diagnose_identifiability(space, fits, pairs)
    assert result["diagnostic_status"] == "sufficient_evidence"
    assert result["near_equivalent_ensemble"]["count"] == 3
    assert all(row["relative_span"] > space["thresholds"]["max_relative_span_identified"] for row in result["parameters"])
    assert {row["classification"] for row in result["parameters"]} == {"weak"}
