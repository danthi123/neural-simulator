import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest
import numpy as np

from research.runners._grounded_speech_action_loop_derisk import (
    SocialFoodWorld,
    SpeechAction,
    _restore_trial_state,
    _snapshot_trial_state,
    _verdict,
)
from research.runners.nav_conv_merged_bridge import _grounded_speech_regions_pathways


def test_world_only_delivers_food_after_matching_request():
    world = SocialFoodWorld()
    assert not world.apply(None)
    assert world.energy == pytest.approx(0.25)
    assert not world.apply(SpeechAction(intent="request", referent="river"))
    assert world.energy == pytest.approx(0.25)
    assert world.apply(SpeechAction(intent="request", referent="apple"))
    assert world.energy == pytest.approx(1.0)


def test_world_consequence_can_be_lesioned():
    world = SocialFoodWorld()
    action = SpeechAction(intent="request", referent="apple")
    assert not world.apply(action, consequence_enabled=False)
    assert world.energy == pytest.approx(0.25)


def test_grounded_speech_slice_contains_learned_route_and_competition():
    regions, pathways = _grounded_speech_regions_pathways()
    names = {r.name for r in regions}
    assert names == {"speech_food_cue", "speech_request", "speech_silence", "speech_wta_fs"}
    routes = {(p.from_region, p.to_region): p for p in pathways}
    learned = routes[("gen_perception", "speech_food_cue")]
    assert learned.plastic is True
    assert learned.plasticity_gate == "speech_grounding"
    assert ("drive_agrp", "speech_request") in routes
    assert ("speech_food_cue", "speech_request") in routes
    assert routes[("speech_wta_fs", "speech_request")].receptor == "gaba_a"
    assert routes[("speech_wta_fs", "speech_silence")].receptor == "gaba_a"


def _decision(name):
    return {"decision": name, "scene": "apple"}


def test_verdict_requires_every_causal_control():
    row = {
        "learning": {"mean_delta": 0.1},
        "hungry_food": _decision("request"),
        "food_delivered": True,
        "causal_events": ["brain_decision:request_apple", "world_consequence:food_delivered"],
        "energy_before": 0.25,
        "energy_after": 1.0,
        "sated_same_scene": _decision("silence"),
        "drive_dose_response": [
            {"deficit": 0.0, "request_margin": -100.0, "decision": "silence"},
            {"deficit": 0.25, "request_margin": -50.0, "decision": "silence"},
            {"deficit": 0.50, "request_margin": 0.0, "decision": "silence"},
            {"deficit": 0.75, "request_margin": 50.0, "decision": "request"},
        ],
        "drive_lesion": _decision("silence"),
        "perception_lesion": _decision("silence"),
        "wrong_object": _decision("silence"),
        "route_lesion": _decision("silence"),
        "untrained_route": _decision("silence"),
        "no_consequence": {
            "first": _decision("request"),
            "second": _decision("request"),
            "delivered": False,
            "energy_before": 0.25,
            "energy_after": 0.25,
        },
        "one_shared_bridge": True,
    }
    assert _verdict(row)["go"] is True
    row["drive_lesion"] = _decision("request")
    assert _verdict(row)["go"] is False


def test_trial_snapshot_restores_dynamic_arrays_only():
    class Bridge:
        def __init__(self):
            self.cp_membrane_potential_v = np.asarray([-65.0, -64.0])
            self.cp_ou_current = np.asarray([1.0, 2.0])
            self.cp_traits = np.asarray([3, 4])

    class Agent:
        _merged_bridge = Bridge()

    agent = Agent()
    saved = _snapshot_trial_state(agent)
    agent._merged_bridge.cp_membrane_potential_v[:] = 0.0
    agent._merged_bridge.cp_ou_current[:] = 0.0
    agent._merged_bridge.cp_traits[:] = 0
    _restore_trial_state(agent, saved)
    assert agent._merged_bridge.cp_membrane_potential_v.tolist() == [-65.0, -64.0]
    assert agent._merged_bridge.cp_ou_current.tolist() == [1.0, 2.0]
    assert agent._merged_bridge.cp_traits.tolist() == [0, 0]


def test_verdict_rejects_reversed_event_order_and_drive_gradient():
    row = {
        "learning": {"mean_delta": 0.1},
        "hungry_food": _decision("request"),
        "food_delivered": True,
        "causal_events": ["world_consequence:food_delivered", "brain_decision:request_apple"],
        "energy_before": 0.25,
        "energy_after": 1.0,
        "sated_same_scene": _decision("silence"),
        "drive_dose_response": [
            {"deficit": 0.0, "request_margin": 50.0, "decision": "request"},
            {"deficit": 0.75, "request_margin": -50.0, "decision": "silence"},
        ],
        "drive_lesion": _decision("silence"),
        "perception_lesion": _decision("silence"),
        "wrong_object": _decision("silence"),
        "route_lesion": _decision("silence"),
        "untrained_route": _decision("silence"),
        "no_consequence": {
            "first": _decision("request"), "second": _decision("request"),
            "delivered": False, "energy_before": 0.25, "energy_after": 0.25,
        },
        "one_shared_bridge": True,
    }
    verdict = _verdict(row)
    assert verdict["checks"]["request_precedes_delivery"] is False
    assert verdict["checks"]["deficit_tracks_request_margin"] is False
    assert verdict["go"] is False
