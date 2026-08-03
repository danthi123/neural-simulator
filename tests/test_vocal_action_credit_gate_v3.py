import inspect
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners._vocal_action_credit_gate import credit_config
from research.runners._vocal_action_credit_gate_v3 import (
    ACTION_COLLATERAL_GATE,
    BASELINE_TRIALS,
    CALIBRATION_SEEDS,
    CONTROL_MODES,
    CRITIC_NORMALIZATION_GATE,
    DEVELOPMENT_SEEDS,
    DOPAMINE_PATH_GATE,
    EVALUATION_TRIALS,
    EXPECTATION_TO_OMISSION_GATE,
    HELD_OUT_SEEDS,
    OMISSION_PATH_GATE,
    OTHER_LANE_FORMAL_SEEDS,
    REWARD_VETO_GATE,
    SMOKE_SEED,
    TRAINING_TRIALS,
    VALUE_TO_SNC_GATE,
    _lesion_gate_values,
    _lesion_telemetry_matches,
    _set_trial_current_v3,
    _structural_preconditions,
    _training_learning_gates,
    aggregate_calibration_status,
    build_v3_bridge,
    schema_smoke,
    run_condition,
    run_seed,
    v3_config,
    validate_phase_seeds,
    validate_protocol_counts,
)


def test_v3_seed_policy_uses_smoke_seed_outside_every_formal_partition():
    partitions = (
        set(CALIBRATION_SEEDS),
        set(DEVELOPMENT_SEEDS),
        set(HELD_OUT_SEEDS),
    )
    assert SMOKE_SEED == 0
    assert all(SMOKE_SEED not in partition for partition in partitions)
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert not (set.union(*partitions) & set(OTHER_LANE_FORMAL_SEEDS))


def test_v3_calibration_only_lock_rejects_reserved_seeds_and_phases():
    assert validate_phase_seeds(
        "calibration", CALIBRATION_SEEDS
    ) == CALIBRATION_SEEDS
    for seed in DEVELOPMENT_SEEDS + HELD_OUT_SEEDS:
        try:
            validate_phase_seeds("calibration", [seed])
        except ValueError as error:
            assert "complete ordered seed partition" in str(error)
        else:
            raise AssertionError(f"reserved seed {seed} was accepted")
    for phase in ("development", "held_out"):
        try:
            validate_phase_seeds(phase, [CALIBRATION_SEEDS[0]])
        except ValueError as error:
            assert "is locked" in str(error)
        else:
            raise AssertionError(f"locked phase {phase} was accepted")

    for incomplete in (
        (),
        (CALIBRATION_SEEDS[0],),
        CALIBRATION_SEEDS[::-1],
        (CALIBRATION_SEEDS[0], CALIBRATION_SEEDS[0]),
    ):
        try:
            validate_phase_seeds("calibration", incomplete)
        except ValueError as error:
            assert "complete ordered seed partition" in str(error)
        else:
            raise AssertionError(f"incomplete aggregate seeds {incomplete} were accepted")


def test_v3_direct_execution_rejects_smoke_and_reserved_seeds_before_build(monkeypatch):
    def fail_build(*_args, **_kwargs):
        raise AssertionError("brain construction happened before seed validation")

    monkeypatch.setattr(
        "research.runners._vocal_action_credit_gate_v3.build_v3_bridge",
        fail_build,
    )
    for seed in (SMOKE_SEED, DEVELOPMENT_SEEDS[0], HELD_OUT_SEEDS[0]):
        try:
            run_seed(seed, training_trials=1, baseline_trials=1, evaluation_trials=1)
        except ValueError as error:
            assert "accepts calibration seeds" in str(error)
        else:
            raise AssertionError(f"direct run_seed accepted reserved seed {seed}")
        try:
            run_condition(
                seed,
                mode="contingent",
                training_trials=1,
                baseline_trials=1,
                evaluation_trials=1,
            )
        except ValueError as error:
            assert "accepts calibration seeds" in str(error)
        else:
            raise AssertionError(f"direct run_condition accepted reserved seed {seed}")


def test_v3_fixed_trial_counts_are_checked_before_build(monkeypatch):
    def fail_build(*_args, **_kwargs):
        raise AssertionError("brain construction happened before protocol validation")

    monkeypatch.setattr(
        "research.runners._vocal_action_credit_gate_v3.build_v3_bridge",
        fail_build,
    )
    validate_protocol_counts(
        training_trials=TRAINING_TRIALS,
        baseline_trials=BASELINE_TRIALS,
        evaluation_trials=EVALUATION_TRIALS,
    )
    for function, kwargs in (
        (run_seed, {}),
        (run_condition, {"mode": "contingent"}),
    ):
        try:
            function(
                CALIBRATION_SEEDS[0],
                training_trials=1,
                baseline_trials=BASELINE_TRIALS,
                evaluation_trials=EVALUATION_TRIALS,
                **kwargs,
            )
        except ValueError as error:
            assert "preregistered" in str(error)
        else:
            raise AssertionError("non-preregistered trial counts were accepted")


def test_v3_critic_output_lesion_preserves_both_learning_routes():
    assert _training_learning_gates("critic_lesion") == (True, True)
    assert all(
        _training_learning_gates(mode) == (True, True)
        for mode in ("contingent", *CONTROL_MODES)
    )


def test_v3_aggregate_requires_both_seeds_and_combines_statuses():
    passed = [
        {"seed": seed, "verdict": {"status": "CALIBRATION_PASS"}}
        for seed in CALIBRATION_SEEDS
    ]
    assert aggregate_calibration_status(passed) == "CALIBRATION_PASS"
    passed[1]["verdict"]["status"] = "CALIBRATION_FAIL"
    assert aggregate_calibration_status(passed) == "CALIBRATION_FAIL"
    passed[1]["verdict"]["status"] = "UNDEFINED"
    assert aggregate_calibration_status(passed) == "UNDEFINED"
    for incomplete in (passed[:1], passed[::-1], [passed[0], passed[0]]):
        try:
            aggregate_calibration_status(incomplete)
        except ValueError as error:
            assert "each calibration seed" in str(error)
        else:
            raise AssertionError("incomplete aggregate rows were accepted")


def test_v3_keeps_v2_gabab_operating_point_and_adds_new_mechanisms():
    v2 = credit_config("v2")
    v3 = v3_config()

    assert v3.gabab_propagation_strength == v2.gabab_propagation_strength
    assert v3.value_to_snc_weight == v2.value_to_snc_weight
    assert v3.n_value_fs > 0
    assert v3.n_lhb > 0
    assert v3.n_rmtg > 0
    assert v3.value_fs_to_value_weight > 0.0
    assert v3.rmtg_to_snc_weight > 0.0


def test_v3_smoke_constructor_has_normalization_and_omission_anatomy():
    config = v3_config()
    bridge, routes = build_v3_bridge(SMOKE_SEED, config)
    structural = _structural_preconditions(bridge, routes, config)

    assert all(structural.values()), structural
    assert set(bridge._transmission_gate_values) >= {
        ACTION_COLLATERAL_GATE,
        DOPAMINE_PATH_GATE,
        VALUE_TO_SNC_GATE,
        EXPECTATION_TO_OMISSION_GATE,
        OMISSION_PATH_GATE,
        CRITIC_NORMALIZATION_GATE,
        REWARD_VETO_GATE,
    }


def test_v3_lesions_cut_only_the_preregistered_neural_routes():
    intact = _lesion_gate_values("contingent")
    assert all(value == 1.0 for value in intact.values())
    assert _lesion_gate_values("collateral_lesion")[
        ACTION_COLLATERAL_GATE
    ] == 0.0
    assert _lesion_gate_values("da_lesion")[DOPAMINE_PATH_GATE] == 0.0

    critic = _lesion_gate_values("critic_lesion")
    assert critic[VALUE_TO_SNC_GATE] == 0.0
    assert critic[EXPECTATION_TO_OMISSION_GATE] == 0.0
    assert critic[OMISSION_PATH_GATE] == 1.0

    omission = _lesion_gate_values("omission_path_lesion")
    assert omission[OMISSION_PATH_GATE] == 0.0
    assert omission[VALUE_TO_SNC_GATE] == 1.0

    normalization = _lesion_gate_values("normalization_lesion")
    assert normalization[CRITIC_NORMALIZATION_GATE] == 0.0
    assert normalization[OMISSION_PATH_GATE] == 1.0

    for mode in ("contingent", *CONTROL_MODES):
        condition = {
            "mode": mode,
            "lesion_gate_values": _lesion_gate_values(mode),
        }
        assert _lesion_telemetry_matches(condition)
    assert not _lesion_telemetry_matches({
        "mode": "critic_lesion",
        "lesion_gate_values": intact,
    })


def test_v3_host_stimulation_interface_has_no_desired_channel():
    parameters = inspect.signature(_set_trial_current_v3).parameters

    assert set(parameters) == {
        "bridge",
        "selector",
        "config",
        "arousal",
        "cue",
        "reward",
        "outcome",
    }
    assert "channel" not in parameters
    assert "desired_channel" not in parameters


def test_v3_schema_smoke_builds_seed_zero_without_science_execution():
    smoke = schema_smoke()

    assert smoke["status"] == "SCHEMA_SMOKE"
    assert smoke["smoke_seed"] == SMOKE_SEED
    assert smoke["science_seed_executed"] is False
    assert all(smoke["structural_preconditions"].values())
    assert set(smoke["required_controls"]) == set(CONTROL_MODES)
    assert smoke["host_boundary"]["desired_channel_current"] is False
    assert smoke["host_boundary"]["host_initializes_dopamine_baseline"] is True
    assert smoke["host_boundary"]["host_calibrates_tonic_production_threshold"] is True
    assert smoke["host_boundary"]["host_trialwise_dopamine_assignment"] is False
    assert smoke["host_boundary"]["host_prediction_error_calculation"] is False
