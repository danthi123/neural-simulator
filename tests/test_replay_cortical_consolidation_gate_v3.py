"""Focused tests for the bounded replay consolidation v3 calibration."""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _replay_cortical_consolidation_gate_v3 as gate  # noqa: E402


def test_fresh_seed_policy_and_phase_lock_keep_reserved_work_untouched():
    assert gate.SMOKE_SEED == 216
    assert gate.CALIBRATION_SEEDS == (228, 229)
    assert gate.DEVELOPMENT_SEEDS == (230, 231, 326)
    assert gate.HELD_OUT_SEEDS == (327, 328, 329)
    assert gate.SMOKE_SEED not in (
        gate.CALIBRATION_SEEDS + gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS
    )
    assert gate.validate_phase("calibration") == "calibration"
    assert gate.validate_calibration_seeds(gate.CALIBRATION_SEEDS) == gate.CALIBRATION_SEEDS
    assert gate.validate_smoke_seed(gate.SMOKE_SEED) == gate.SMOKE_SEED
    with pytest.raises(ValueError, match="opens.*calibration"):
        gate.validate_phase("development")
    with pytest.raises(ValueError, match="fresh calibration seeds"):
        gate.validate_calibration_seeds([gate.SMOKE_SEED])
    for seed in gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="fresh calibration seeds"):
            gate.validate_calibration_seeds([seed])
    for seed in gate.CALIBRATION_SEEDS + gate.DEVELOPMENT_SEEDS + gate.HELD_OUT_SEEDS:
        with pytest.raises(ValueError, match="non-scientific seed"):
            gate.validate_smoke_seed(seed)


def test_cli_resolution_separates_smoke_from_every_scientific_partition():
    assert gate.resolve_cli_request(smoke=True, phase=None, seeds=None) == (
        "smoke",
        (gate.SMOKE_SEED,),
    )
    assert gate.resolve_cli_request(
        smoke=False,
        phase=None,
        seeds=None,
    ) == ("calibration", gate.CALIBRATION_SEEDS)
    with pytest.raises(ValueError, match="accepts --seeds"):
        gate.resolve_cli_request(
            smoke=True,
            phase="smoke",
            seeds=gate.CALIBRATION_SEEDS,
        )
    with pytest.raises(ValueError, match="requires --smoke"):
        gate.resolve_cli_request(smoke=False, phase="smoke", seeds=None)
    with pytest.raises(ValueError, match="fresh calibration seeds"):
        gate.resolve_cli_request(
            smoke=False,
            phase="calibration",
            seeds=(gate.SMOKE_SEED,),
        )


def test_smoke_payload_is_marked_non_scientific_and_skips_calibration_verdict(monkeypatch):
    calls = []

    def fake_condition(seed, condition, config, *, smoke=False):
        calls.append((seed, condition, smoke))
        return {
            "seed_partition": "smoke",
            "scientific_partition": False,
            "phase_trace": ["encode_A", "encode_B", "sleep", "retest"],
            "single_bridge_persisted": True,
        }

    monkeypatch.setattr(gate, "run_condition", fake_condition)
    monkeypatch.setattr(
        gate,
        "_calibration_verdict",
        lambda _conditions: pytest.fail("smoke must not compute a calibration verdict"),
    )
    payload = gate.run_smoke(gate.smoke_config())

    assert payload["phase"] == "smoke"
    assert payload["seed"] == gate.SMOKE_SEED
    assert payload["seed_partition"] == "smoke"
    assert payload["scientific_partition"] is False
    assert payload["calibration_verdict_computed"] is False
    assert payload["structural_checks"] == {
        "all_conditions_executed": True,
        "fixed_phase_sequence": True,
        "single_bridge_persisted": True,
        "no_scientific_seed_used": True,
    }
    assert calls == [
        (gate.SMOKE_SEED, condition, True) for condition in gate.CONDITIONS
    ]


def test_condition_set_retains_v2_controls_and_adds_relay_and_balance_lesions():
    assert gate.CONDITIONS == (
        "intact",
        "no_sleep",
        "shuffled_replay_order",
        "shuffled_target_index",
        "ca3_ca1_lesion",
        "cortical_plasticity_off",
        "target_inhibition_lesion",
        "index_relay_lesion",
        "index_balance_lesion",
    )
    with pytest.raises(ValueError, match="Unknown condition"):
        gate.run_condition(
            gate.SMOKE_SEED,
            "not_a_condition",
            gate.smoke_config(),
            smoke=True,
        )


def test_temporal_control_preserves_exact_event_content_and_changes_only_order():
    cfg = gate.smoke_config()
    ca3 = np.arange(cfg.n_ca3, dtype=np.int64)
    intact = gate.v2._ordered_sleep_events(gate.SMOKE_SEED, cfg, ca3, shuffle=False)
    shuffled = gate.v2._ordered_sleep_events(gate.SMOKE_SEED, cfg, ca3, shuffle=True)

    assert gate.v2._event_digest(intact, order_sensitive=False) == gate.v2._event_digest(
        shuffled, order_sensitive=False,
    )
    assert gate.v2._event_digest(intact, order_sensitive=True) != gate.v2._event_digest(
        shuffled, order_sensitive=True,
    )
    assert gate.v2._mean_adjacent_overlap(intact) > gate.v2._mean_adjacent_overlap(shuffled)


def test_bridge_uses_neutral_learned_index_and_two_local_inhibitory_loops():
    cfg = gate.smoke_config()
    bridge, handles = gate.build_bridge(gate.SMOKE_SEED, cfg)

    assert set(handles["regions"]) == {
        "ca3",
        "ca1",
        "cortical_cue",
        "cortical_target",
        "cortical_target_fs",
        "cortical_index",
        "cortical_index_fs",
    }
    assert handles["neutral_index_fan_in"] is True
    assert handles["wiring_counts"]["ca1_to_cortical_index"] == cfg.n_ca1 * cfg.n_index
    assert "ca1_to_cortical_target" not in handles["wiring_counts"]
    assert bridge.core_config.enable_coincidence_detection is True
    assert bridge.core_config.coincidence_weighted_drive is True
    assert bridge.core_config.enable_gabab is True
    assert bridge.cp_coincidence_synapse_mask is not None
    assert bridge.cp_gabab_synapse_mask is not None
    assert gate.INDEX_TARGET_GATE in bridge.list_plasticity_gates()
    assert {
        gate.SCHAFFER_GATE,
        gate.TARGET_INHIBITION_GATE,
        gate.INDEX_OUTPUT_GATE,
        gate.INDEX_BALANCE_GATE,
        gate.WAKE_TEACHING_GATE,
    } <= set(bridge._transmission_gate_to_synapses)


def test_target_index_shuffle_preserves_weights_but_breaks_learned_assignment():
    bridge, _ = gate.build_bridge(gate.SMOKE_SEED, gate.smoke_config())
    indices = bridge._plasticity_gate_indices_gpu[gate.INDEX_TARGET_GATE]
    learned = np.linspace(0.1, 20.0, len(indices), dtype=np.float32)
    bridge.cp_connections.data[indices] = learned
    before = np.asarray(bridge.cp_connections.data[indices]).copy()

    changed = gate.v1._shuffle_target_index(bridge, gate.SMOKE_SEED)
    after = np.asarray(bridge.cp_connections.data[indices]).copy()

    assert changed == len(indices)
    assert not np.array_equal(before, after)
    np.testing.assert_array_equal(np.sort(before), np.sort(after))


def test_local_slow_feedback_is_recruited_by_its_own_index_assembly():
    cfg = gate.smoke_config()
    bridge, handles = gate.build_bridge(gate.SMOKE_SEED, cfg)
    bridge.set_transmission_gate(gate.INDEX_BALANCE_GATE, 1.0)
    index_a = handles["device_patterns"]["A"]["index"]
    index_b = handles["patterns"]["B"]["index"]

    fs_spikes = 0
    for _ in range(45):
        gate.v1._zero_current(bridge)
        bridge.cp_external_input_current[index_a] = 1100.0
        bridge._run_one_simulation_step()
        fs_spikes += int(
            bridge.cp_firing_states[handles["index_fs_pools"]["A"]].sum()
        )
    conductance = np.asarray(bridge.cp_conductance_g_gabab)

    assert fs_spikes > 0
    assert float(conductance[handles["patterns"]["A"]["index"]].mean()) > 0.0
    assert float(conductance[index_b].mean()) == pytest.approx(0.0, abs=1e-8)


def test_intact_default_condition_uses_synaptic_relay_without_host_selected_sleep_drive():
    row = gate.run_condition(
        gate.SMOKE_SEED,
        "intact",
        gate.GateConfig(),
        smoke=True,
    )

    assert row["phase_trace"] == ["encode_A", "encode_B", "sleep", "retest"]
    assert row["seed_partition"] == "smoke"
    assert row["scientific_partition"] is False
    assert row["single_bridge_persisted"] is True
    assert row["neutral_index_fan_in"] is True
    assert row["encode_A"]["index_host_driven"] is False
    assert row["encode_B"]["index_host_driven"] is False
    assert row["encode_A"]["spikes"]["index"] > 0
    assert row["encode_B"]["spikes"]["index"] > 0
    assert row["sleep"]["host_selected_episode_for_replay"] is False
    assert row["sleep"]["host_selected_target_drive"] is False
    assert row["sleep"]["spikes"]["cortical_index"] > 0
    assert row["sleep"]["spikes"]["cortical_index_fs"] > 0
    assert row["sleep"]["spikes"]["cortical_target"] > 0
    assert max(row["sleep"]["index_balance_conductance_peak"].values()) > 0.0
    assert abs(row["weight_deltas"]["index_target_during_sleep"]) < 1e-7
    assert row["weight_deltas"]["cortical_during_sleep"] > 0.0
    assert set(row["recall"]) == {"A", "B"}


def test_runner_does_not_rank_replay_or_directly_drive_the_index_relay():
    sleep_source = inspect.getsource(gate._sleep)
    encode_source = inspect.getsource(gate._encode_memory)

    assert "argsort" not in sleep_source and "argpartition" not in sleep_source
    assert '"host_selected_episode_for_replay": False' in sleep_source
    assert '"host_selected_target_drive": False' in sleep_source
    assert 'for key in ("ca3", "cue", "target")' in encode_source
    assert '"index_host_driven": False' in encode_source
