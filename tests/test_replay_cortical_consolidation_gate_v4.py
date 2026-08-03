"""Focused tests for replay consolidation v4 target dendritic reinstatement."""
from __future__ import annotations

import inspect
import os
from dataclasses import asdict

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _replay_cortical_consolidation_gate_v4 as gate  # noqa: E402


def test_fresh_seed_policy_and_phase_lock_keep_reserved_work_untouched():
    assert gate.SMOKE_SEED == 216
    assert gate.CALIBRATION_SEEDS == (451, 457)
    assert gate.DEVELOPMENT_SEEDS == (461, 463, 467)
    assert gate.HELD_OUT_SEEDS == (479, 487, 491)
    assert gate.validate_phase("calibration") == "calibration"
    assert gate.validate_calibration_seeds(gate.CALIBRATION_SEEDS) == gate.CALIBRATION_SEEDS
    for seed in gate.CALIBRATION_SEEDS:
        assert gate.validate_calibration_seed(seed) == seed
    assert gate.validate_smoke_seed(gate.SMOKE_SEED) == gate.SMOKE_SEED
    with pytest.raises(ValueError, match="opens.*calibration"):
        gate.validate_phase("development")
    for seed in (gate.SMOKE_SEED,) + gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="individual fresh calibration seeds"):
            gate.validate_calibration_seed(seed)
    for seed in gate.CALIBRATION_SEEDS + gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="non-scientific seed"):
            gate.validate_smoke_seed(seed)


@pytest.mark.parametrize(
    "seeds",
    [
        (),
        (gate.CALIBRATION_SEEDS[0],),
        (gate.CALIBRATION_SEEDS[1],),
        tuple(reversed(gate.CALIBRATION_SEEDS)),
        (gate.CALIBRATION_SEEDS[0], gate.CALIBRATION_SEEDS[0]),
        (gate.SMOKE_SEED,) + gate.CALIBRATION_SEEDS,
        gate.CALIBRATION_SEEDS + (gate.DEVELOPMENT_SEEDS[0],),
    ],
)
def test_aggregate_calibration_requires_exact_ordered_seed_partition(seeds):
    with pytest.raises(ValueError, match="exact ordered fresh calibration seed partition"):
        gate.validate_calibration_seeds(seeds)


def test_direct_calls_reject_reserved_seed_before_build_or_condition(monkeypatch):
    monkeypatch.setattr(
        gate,
        "build_bridge",
        lambda *_args, **_kwargs: pytest.fail("reserved direct call reached brain build"),
    )
    with pytest.raises(ValueError, match="individual fresh calibration seeds"):
        gate.run_condition(gate.DEVELOPMENT_SEEDS[0], "intact")

    monkeypatch.setattr(
        gate,
        "run_condition",
        lambda *_args, **_kwargs: pytest.fail("reserved run_seed reached a condition"),
    )
    with pytest.raises(ValueError, match="individual fresh calibration seeds"):
        gate.run_seed(gate.HELD_OUT_SEEDS[0])


def test_run_calibration_rejects_partial_partition_before_running_seed(monkeypatch):
    monkeypatch.setattr(
        gate,
        "run_seed",
        lambda *_args, **_kwargs: pytest.fail("invalid aggregate must not run a seed"),
    )
    with pytest.raises(ValueError, match="exact ordered fresh calibration seed partition"):
        gate.run_calibration((gate.CALIBRATION_SEEDS[0],), gate.smoke_config())


def test_cli_resolution_separates_smoke_and_exact_calibration_partitions():
    assert gate.resolve_cli_request(smoke=True, phase=None, seeds=None) == (
        "smoke",
        (gate.SMOKE_SEED,),
    )
    assert gate.resolve_cli_request(smoke=False, phase=None, seeds=None) == (
        "calibration",
        gate.CALIBRATION_SEEDS,
    )
    with pytest.raises(ValueError, match="accepts --seeds"):
        gate.resolve_cli_request(
            smoke=True,
            phase="smoke",
            seeds=gate.CALIBRATION_SEEDS,
        )
    with pytest.raises(ValueError, match="requires --smoke"):
        gate.resolve_cli_request(smoke=False, phase="smoke", seeds=None)
    with pytest.raises(ValueError, match="exact ordered fresh calibration seed partition"):
        gate.resolve_cli_request(
            smoke=False,
            phase="calibration",
            seeds=(gate.CALIBRATION_SEEDS[0],),
        )


def test_v4_preserves_v3_configuration_and_controls_exactly_before_new_lesion():
    assert asdict(gate.GateConfig()) == asdict(gate.v3.GateConfig())
    assert asdict(gate.smoke_config()) == asdict(gate.v3.GateConfig())
    assert gate.CONDITIONS[:-1] == gate.v3.CONDITIONS
    assert gate.CONDITIONS[-1] == "target_plateau_lesion"


def test_build_adds_only_index_output_to_existing_weighted_coincidence_route():
    cfg = gate.smoke_config()
    bridge, handles = gate.build_bridge(gate.SMOKE_SEED, cfg)
    routes = gate._coincidence_route_telemetry(bridge)

    assert bridge.core_config.enable_coincidence_detection is True
    assert bridge.core_config.coincidence_weighted_drive is True
    assert bridge.core_config.coincidence_k_threshold == cfg.index_coincidence_threshold
    assert bridge.core_config.coincidence_plateau_strength == cfg.index_plateau_strength
    assert routes == handles["coincidence_routes"]
    assert routes["target_route_total"] == handles["wiring_counts"][
        "cortical_index_to_target"
    ]
    assert routes["target_route_enabled"] == routes["target_route_total"]
    assert routes["ca1_index_route_enabled"] == routes["ca1_index_route_total"]
    assert bridge._transmission_gate_values[gate.INDEX_OUTPUT_GATE] == 1.0


def test_target_plateau_toggle_preserves_ampa_weights_and_ca1_index_route():
    bridge, _ = gate.build_bridge(gate.SMOKE_SEED, gate.smoke_config())
    target_indices = gate._target_route_indices(bridge)
    weights_before = np.asarray(bridge.cp_connections.data[target_indices]).copy()
    transmission_before = np.asarray(bridge.cp_transmission_gain[target_indices]).copy()
    index_before = gate._count_enabled(
        bridge.cp_coincidence_synapse_mask,
        gate._index_route_indices(bridge),
    )

    gate._set_target_plateau_route(bridge, False)
    routes = gate._coincidence_route_telemetry(bridge)

    assert routes["target_route_enabled"] == 0
    assert routes["ca1_index_route_enabled"] == index_before
    np.testing.assert_array_equal(
        np.asarray(bridge.cp_connections.data[target_indices]), weights_before
    )
    np.testing.assert_array_equal(
        np.asarray(bridge.cp_transmission_gain[target_indices]), transmission_before
    )
    assert bridge._transmission_gate_values[gate.INDEX_OUTPUT_GATE] == 1.0

    gate._set_target_plateau_route(bridge, True)
    assert gate._coincidence_route_telemetry(bridge)["target_route_enabled"] == int(
        target_indices.size
    )


def _condition_row(*, target: int, target_fs: int, index: int, index_fs: int, lesion=False):
    target_total = 40
    index_total = 60
    return {
        "sleep": {
            "spikes": {
                "cortical_target": target,
                "cortical_target_fs": target_fs,
                "cortical_index": index,
                "cortical_index_fs": index_fs,
            },
            "target_plateau_peak_overall": 0.0 if lesion else 3.0,
            "target_plateau_area_overall": 0.0 if lesion else 12.0,
            "index_output_transmission_gain_during_sleep": 1.0,
            "coincidence_routes_during_sleep": {
                "target_route_total": target_total,
                "target_route_enabled": 0 if lesion else target_total,
                "ca1_index_route_total": index_total,
                "ca1_index_route_enabled": index_total,
            },
        }
    }


def _inherited_verdict():
    return {
        "calibration_status": "CALIBRATION_PROMISING",
        "preconditions": [{"name": "v3 inherited", "ok": True}],
        "disabled_processes": [],
        "undefined_reasons": [],
        "checks": {"v3_criteria_preserved": True},
    }


def test_v4_verdict_retains_inherited_checks_and_applies_exact_causal_thresholds(monkeypatch):
    monkeypatch.setattr(gate.v3, "_calibration_verdict", lambda _rows: _inherited_verdict())
    conditions = {
        "intact": _condition_row(target=100, target_fs=40, index=100, index_fs=40),
        "target_plateau_lesion": _condition_row(
            target=75,
            target_fs=30,
            index=75,
            index_fs=30,
            lesion=True,
        ),
    }

    verdict = gate._calibration_verdict(conditions)

    assert verdict["calibration_status"] == "CALIBRATION_PROMISING"
    assert verdict["checks"]["v3_criteria_preserved"] is True
    assert all(verdict["target_plateau_checks"].values())
    assert len(verdict["preconditions"]) == 4


def test_v4_verdict_revises_when_target_activity_is_not_materially_reduced(monkeypatch):
    monkeypatch.setattr(gate.v3, "_calibration_verdict", lambda _rows: _inherited_verdict())
    conditions = {
        "intact": _condition_row(target=100, target_fs=40, index=100, index_fs=40),
        "target_plateau_lesion": _condition_row(
            target=76,
            target_fs=31,
            index=100,
            index_fs=40,
            lesion=True,
        ),
    }

    verdict = gate._calibration_verdict(conditions)

    assert verdict["calibration_status"] == "CALIBRATION_NEEDS_REVISION"
    assert verdict["target_plateau_checks"][
        "target_plateau_is_load_bearing_for_target"
    ] is False
    assert verdict["target_plateau_checks"][
        "target_plateau_is_load_bearing_for_target_fs"
    ] is False


def test_v4_verdict_is_undefined_when_lesion_does_not_isolate_route(monkeypatch):
    monkeypatch.setattr(gate.v3, "_calibration_verdict", lambda _rows: _inherited_verdict())
    conditions = {
        "intact": _condition_row(target=100, target_fs=40, index=100, index_fs=40),
        "target_plateau_lesion": _condition_row(
            target=50,
            target_fs=20,
            index=100,
            index_fs=40,
            lesion=True,
        ),
    }
    conditions["target_plateau_lesion"]["sleep"][
        "index_output_transmission_gain_during_sleep"
    ] = 0.0

    verdict = gate._calibration_verdict(conditions)

    assert verdict["calibration_status"] == "UNDEFINED"
    assert any("preserving AMPA" in reason for reason in verdict["undefined_reasons"])


def test_smoke_payload_is_non_scientific_and_never_computes_verdict(monkeypatch):
    calls = []

    def fake_condition(seed, condition, config, *, smoke=False):
        calls.append((seed, condition, smoke))
        lesion = condition == "target_plateau_lesion"
        return {
            "seed_partition": "smoke",
            "scientific_partition": False,
            "phase_trace": ["encode_A", "encode_B", "sleep", "retest"],
            "single_bridge_persisted": True,
            "sleep": {
                "spikes": {
                    "cortical_target": 20 if lesion else 100,
                    "cortical_target_fs": 10 if lesion else 40,
                    "cortical_index": 90 if lesion else 100,
                    "cortical_index_fs": 36 if lesion else 40,
                },
                "target_plateau_peak_overall": 0.0 if lesion else 3.0,
                "target_plateau_area_overall": 0.0 if lesion else 12.0,
                "index_output_transmission_gain_during_sleep": 1.0,
                "coincidence_routes_during_sleep": {
                    "target_route_total": 20,
                    "target_route_enabled": 0 if lesion else 20,
                    "ca1_index_route_total": 30,
                    "ca1_index_route_enabled": 30,
                },
            },
        }

    monkeypatch.setattr(gate, "run_condition", fake_condition)
    monkeypatch.setattr(
        gate,
        "_calibration_verdict",
        lambda _conditions: pytest.fail("smoke must not compute a scientific verdict"),
    )
    payload = gate.run_smoke(gate.smoke_config())

    assert payload["phase"] == "smoke"
    assert payload["seed"] == gate.SMOKE_SEED
    assert payload["scientific_partition"] is False
    assert payload["calibration_verdict_computed"] is False
    assert "calibration_status" not in payload
    assert all(payload["structural_checks"].values())
    assert all(payload["dynamics_checks"].values())
    assert calls == [
        (gate.SMOKE_SEED, condition, True) for condition in gate.CONDITIONS
    ]


def test_sleep_source_drives_only_episode_agnostic_ca3_background():
    source = inspect.getsource(gate._sleep)

    assert "argsort" not in source and "argpartition" not in source
    assert 'bridge.cp_external_input_current[background_dev]' in source
    assert '"host_selected_episode_for_replay": False' in source
    assert '"host_selected_target_drive": False' in source
