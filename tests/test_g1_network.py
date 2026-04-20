"""Tests for the G1 network wiring helper (64 input + 4 output + plastic i->o + lateral inhibition)."""
import numpy as np
import pytest

from research.runners.g1_network import build_g1_network_config, G1NetworkSpec


def test_spec_defaults():
    spec = G1NetworkSpec()
    assert spec.n_input == 64
    assert spec.n_output == 4
    assert spec.n_total == 68
    assert spec.input_indices == list(range(0, 64))
    assert spec.output_indices == list(range(64, 68))


def test_build_g1_config_produces_wiring_plan():
    core_cfg, wiring_plan = build_g1_network_config(seed=42)
    assert core_cfg.num_neurons == 68
    assert core_cfg.neuron_model_type == "IZHIKEVICH"
    assert core_cfg.seed == 42
    assert core_cfg.dt_ms == 1.0
    assert core_cfg.enable_stdp is True
    assert core_cfg.enable_watts_strogatz is False
    assert "input_to_output" in wiring_plan
    assert wiring_plan["input_to_output"]["count"] == 64 * 4
    assert wiring_plan["input_to_output"]["plastic"] is True
    assert "output_lateral_inhibition" in wiring_plan
    assert wiring_plan["output_lateral_inhibition"]["count"] == 4 * 3
    assert wiring_plan["output_lateral_inhibition"]["plastic"] is False


def test_initial_weights_in_range():
    core_cfg, wiring_plan = build_g1_network_config(seed=7)
    w = np.asarray(wiring_plan["input_to_output"]["initial_weights"])
    assert w.shape == (64 * 4,)
    assert w.min() >= 0.05 - 1e-6
    assert w.max() <= 0.15 + 1e-6


def test_lateral_inhibition_weights_sign_and_magnitude():
    core_cfg, wiring_plan = build_g1_network_config(seed=0)
    w = np.asarray(wiring_plan["output_lateral_inhibition"]["initial_weights"])
    assert w.shape == (4 * 3,)
    assert np.all(w > 0), "Weights are positive magnitudes; sign handled by sim inhibitory trait"
    assert np.allclose(w, 1.0, atol=1e-6)


def test_edges_are_correct_pairs():
    core_cfg, wiring_plan = build_g1_network_config(seed=0)
    i2o = wiring_plan["input_to_output"]
    pre = list(i2o["pre_indices"])
    post = list(i2o["post_indices"])
    assert len(pre) == len(post) == 256
    assert min(pre) == 0 and max(pre) == 63
    assert min(post) == 64 and max(post) == 67
    pairs = set(zip(pre, post))
    assert len(pairs) == 256
    lat = wiring_plan["output_lateral_inhibition"]
    lpre, lpost = list(lat["pre_indices"]), list(lat["post_indices"])
    assert len(lpre) == 12
    for a, b in zip(lpre, lpost):
        assert 64 <= a <= 67 and 64 <= b <= 67
        assert a != b
