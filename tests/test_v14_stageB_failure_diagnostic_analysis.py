import copy

import pytest

from tools import v14_stageB_failure_diagnostic_analysis as analysis


def _receipt():
    candidates = [
        {"candidate_id": f"candidate-{index}", "candidate_sha256": f"{index + 1:064x}"}
        for index in range(9)
    ]
    traces = []
    for candidate in candidates:
        for current in analysis.EXPECTED_CURRENTS:
            traces.append(
                {
                    **candidate,
                    "rescue_current_pA": current,
                    "diagnostic_trace": {
                        "path": f"traces/{candidate['candidate_id']}-{current}.zip",
                        "sample_count": 90_000,
                        "sha256": "f" * 64,
                    },
                }
            )
    return {
        "schema": analysis.RECEIPT_SCHEMA,
        "process_status": "completed",
        "engineering_diagnostic_only": True,
        "scientific_verdict": None,
        "candidate_promotion_allowed": False,
        "parameter_tuning_allowed": False,
        "source_equivalence_claimed": False,
        "execution": {
            "candidate_count": 9,
            "arm_count": 4,
            "trace_count": 36,
            "total_steps_per_arm": 90_000,
        },
        "selection": candidates,
        "traces": traces,
    }


def test_receipt_validator_requires_exact_candidate_current_cross_product():
    candidates, indexed = analysis._validate_receipt(_receipt())
    assert len(candidates) == 9
    assert len(indexed) == 36
    assert set(current for _, current in indexed) == set(analysis.EXPECTED_CURRENTS)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.update({"scientific_verdict": "GO"}),
        lambda receipt: receipt["traces"].pop(),
        lambda receipt: receipt["traces"][0].update({"rescue_current_pA": -15.0}),
        lambda receipt: receipt["traces"][0]["diagnostic_trace"].update(
            {"sample_count": 89_999}
        ),
        lambda receipt: receipt["traces"][1].update(
            {
                "candidate_id": receipt["traces"][0]["candidate_id"],
                "candidate_sha256": receipt["traces"][0]["candidate_sha256"],
                "rescue_current_pA": receipt["traces"][0]["rescue_current_pA"],
            }
        ),
    ],
)
def test_receipt_validator_rejects_boundary_or_cross_product_drift(mutate):
    receipt = copy.deepcopy(_receipt())
    mutate(receipt)
    with pytest.raises(analysis.StageBFailureAnalysisError):
        analysis._validate_receipt(receipt)


def test_phase_windows_match_preregistered_sample_boundaries():
    assert analysis.PHASE_SLICES == {
        "baseline": slice(29_999, 39_999),
        "immediate_post_lesion": slice(39_999, 40_399),
        "late_post_lesion": slice(49_999, 59_999),
        "pulse": slice(59_999, 69_999),
        "late_pulse": slice(67_999, 69_999),
        "release": slice(69_999, 90_000),
    }


def test_output_digest_is_canonical():
    body = {
        "schema": analysis.OUTPUT_SCHEMA,
        "engineering_diagnostic_only": True,
        "scientific_verdict": None,
    }
    result = {**body, "sha256": analysis._digest(body)}
    assert result["sha256"] == analysis._digest(
        {key: value for key, value in result.items() if key != "sha256"}
    )
