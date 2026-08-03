"""Safety and NumPy smoke pins for the Gate B v5 sidecar."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    from sim.backend import _reset_cache_for_tests

    _reset_cache_for_tests()
    name = "research.runners._vocal_action_credit_gate_v5"
    module = importlib.import_module(name)
    module = importlib.reload(module)
    yield module
    _reset_cache_for_tests()


def test_formal_seeds_and_phases_are_sealed(gate):
    assert gate.OPEN_PHASES == ()
    assert gate.validate_smoke_seed(0) == 0
    with pytest.raises(ValueError):
        gate.validate_smoke_seed(1)
    with pytest.raises(ValueError):
        gate.validate_phase("calibration")
    with pytest.raises(ValueError):
        gate.validate_formal_seeds((70001,))
    with pytest.raises(ValueError):
        gate.run_formal_seed(70001)


def test_pathways_are_host_winner_free_and_outcome_symmetric(gate):
    cfg = gate.v5_config()
    pathways = gate._v5_pathways(cfg)
    assert not gate.HOST_BOUNDARY["host_action_winner_latch"]
    assert not gate.HOST_BOUNDARY["host_action_timed_transmission_window"]

    tag_sources = {"practice_arousal", "commit_0", "commit_1"}
    tag_routes = [
        pathway
        for pathway in pathways
        if pathway.from_region in tag_sources
        and pathway.to_region.startswith("vocal_credit_value_")
    ]
    assert len(tag_routes) == 4
    assert all(p.coincidence_detector for p in tag_routes)
    assert all(p.transmission_gate is None for p in tag_routes)

    for gate_name, target_prefix in (
        (gate.OUTCOME_EXCITATION_GATE, "vocal_credit_value_"),
        (gate.OUTCOME_INHIBITION_GATE, gate.v3.VALUE_FS_PREFIX),
    ):
        routes = [
            p for p in pathways
            if p.from_region == gate.OUTCOME_ONSET
            and p.transmission_gate == gate_name
        ]
        assert len(routes) == 2
        assert {p.to_region for p in routes} == {
            f"{target_prefix}{channel}" for channel in gate.CHANNELS
        }
        assert len({p.weight_mean for p in routes}) == 1
        assert all(not p.plastic for p in routes)


def test_numpy_smoke(gate):
    result = gate.run_smoke()
    failed = [name for name, ok in result["checks"].items() if not ok]
    assert result["science_seed_executed"] is False
    assert result["status"] == "SMOKE_PASS", failed
