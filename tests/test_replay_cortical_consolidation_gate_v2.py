"""Focused tests for the bounded replay consolidation v2 calibration."""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _replay_cortical_consolidation_gate_v2 as gate  # noqa: E402


def test_seed_policy_keeps_all_reserved_seeds_untouched():
    assert gate.validate_calibration_seeds(gate.CALIBRATION_SEEDS) == gate.CALIBRATION_SEEDS
    for seed in gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="calibration seeds"):
            gate.validate_calibration_seeds([seed])


def test_condition_set_retains_v1_controls_and_adds_order_and_inhibition_lesions():
    assert gate.CONDITIONS == (
        "intact",
        "no_sleep",
        "shuffled_replay_order",
        "shuffled_target_index",
        "ca3_ca1_lesion",
        "cortical_plasticity_off",
        "target_inhibition_lesion",
    )
    with pytest.raises(ValueError, match="Unknown condition"):
        gate.run_condition(212, "not_a_condition", gate.smoke_config())


def test_temporal_control_changes_order_but_preserves_exact_event_content():
    cfg = gate.smoke_config()
    ca3 = np.arange(cfg.n_ca3, dtype=np.int64)
    intact = gate._ordered_sleep_events(212, cfg, ca3, shuffle=False)
    shuffled = gate._ordered_sleep_events(212, cfg, ca3, shuffle=True)

    assert gate._event_digest(intact, order_sensitive=False) == gate._event_digest(
        shuffled, order_sensitive=False,
    )
    assert gate._event_digest(intact, order_sensitive=True) != gate._event_digest(
        shuffled, order_sensitive=True,
    )
    assert gate._mean_adjacent_overlap(intact) > gate._mean_adjacent_overlap(shuffled)


def test_bridge_contains_neural_feedback_inhibition_and_causal_gates():
    bridge, handles = gate.build_bridge(212, gate.smoke_config())
    assert set(handles["regions"]) == {
        "ca3", "ca1", "cortical_cue", "cortical_target", "cortical_target_fs",
    }
    assert len(handles["inhibitory_indices"]) == gate.smoke_config().n_target_fs
    assert handles["wiring_counts"]["target_to_fs"] > 0
    assert handles["wiring_counts"]["ca1_to_target_fs"] > 0
    assert handles["wiring_counts"]["fs_to_target"] > 0
    assert gate.TARGET_INHIBITION_GATE in bridge._transmission_gate_to_synapses
    assert gate.SCHAFFER_GATE in bridge._transmission_gate_to_synapses
    assert {
        gate.CA3_GATE,
        gate.INDEX_CUE_GATE,
        gate.INDEX_TARGET_GATE,
        gate.CORTICAL_GATE,
        gate.TARGET_RECURRENT_GATE,
    } <= set(bridge.list_plasticity_gates())

    bridge.set_transmission_gate(gate.TARGET_INHIBITION_GATE, 0.0)
    a_drive = np.concatenate([
        handles["patterns"]["A"]["ca1"],
        handles["patterns"]["A"]["target"],
    ])
    fs_a = handles["fs_pools"]["A"]
    fs_b = handles["fs_pools"]["B"]
    spikes_a = spikes_b = 0
    for _ in range(30):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[a_drive] = 1100.0
        bridge._run_one_simulation_step()
        firing = np.asarray(bridge.cp_firing_states)
        spikes_a += int(firing[fs_a].sum())
        spikes_b += int(firing[fs_b].sum())
    assert spikes_a > 0
    assert spikes_a > spikes_b


def test_smoke_runs_all_phases_on_one_persisted_bridge():
    row = gate.run_condition(212, "intact", gate.smoke_config())
    assert row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
    assert row["single_bridge_persisted"] is True
    assert row["sleep"]["host_selected_episode_for_replay"] is False
    assert "cortical_target_fs" in row["sleep"]["spikes"]
    assert set(row["recall"]) == {"A", "B"}


def test_controls_are_mechanically_distinct_and_verdict_is_guarded():
    cfg = gate.smoke_config()
    conditions = {condition: gate.run_condition(212, condition, cfg) for condition in gate.CONDITIONS}
    verdict = gate._calibration_verdict(conditions)

    assert conditions["no_sleep"]["sleep"]["spikes"]["ca3"] == 0
    assert conditions["ca3_ca1_lesion"]["sleep"]["spikes"]["ca3"] > 0
    assert conditions["target_inhibition_lesion"]["sleep"]["target_inhibition_gain_during_sleep"] == 0.0
    assert conditions["intact"]["sleep"]["target_inhibition_gain_during_sleep"] == 1.0
    assert abs(conditions["cortical_plasticity_off"]["weight_deltas"]["cortical_during_sleep"]) < 1e-7
    assert verdict["preconditions"]
    assert all(check["ok"] is not None for check in verdict["preconditions"])
    assert set(verdict["attribution"]) == set(gate.CONDITIONS) - {"intact"}


def test_runner_names_scaffolds_and_does_not_host_rank_sleep_events():
    source = inspect.getsource(gate._sleep)
    assert "argsort" not in source and "argpartition" not in source
    assert "host_selected_episode_for_replay\": False" in source
    payload = gate.run_calibration([212], gate.smoke_config())
    assert payload["phase"] == "calibration"
    assert payload["reserved_seeds_inspected"] is False
    assert payload["remaining_scaffolds"]
    assert payload["calibration_status"] in {
        "UNDEFINED", "CALIBRATION_PROMISING", "CALIBRATION_NEEDS_REVISION",
    }
