import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    DIRECT_PATH_GATE,
    SelectorConfig,
    build_selector_bridge,
    run_condition,
)
from sim.backend import to_host


def _pathways(bridge, source, target):
    return [
        pathway for pathway in bridge.core_config.region_pathways
        if pathway.from_region == source and pathway.to_region == target
    ]


def test_selector_anatomy_is_symmetric_and_target_independent():
    bridge = build_selector_bridge(seed=7, config=SelectorConfig())
    traits = np.asarray(to_host(bridge.cp_traits))

    for channel in CHANNELS:
        other = 1 - channel
        arousal = _pathways(bridge, "practice_arousal", f"proposal_{channel}")
        assert len(arousal) == 1
        assert arousal[0].density == 1.0
        assert arousal[0].weight_mean == 1.0
        assert _pathways(bridge, f"proposal_{channel}", f"str_fsi_{channel}")
        assert _pathways(bridge, f"str_fsi_{channel}", f"str_d1_{other}")
        assert _pathways(bridge, f"str_fsi_{channel}", f"str_d2_{other}")
        assert not _pathways(
            bridge, f"str_fsi_{channel}", f"str_d1_{channel}"
        )
        direct = _pathways(bridge, f"str_d1_{channel}", f"gpi_{channel}")
        assert len(direct) == 1
        assert direct[0].transmission_gate == DIRECT_PATH_GATE

    proposal_indices = set()
    for channel in CHANNELS:
        proposal_indices.update(bridge.region_manager.indices(
            f"proposal_{channel}"
        ))
    masked_indices = set(
        bridge.cp_ou_neuron_mask.nonzero()[0].tolist()
    )
    assert masked_indices == proposal_indices

    inhibitory = {
        "selector_reset",
        *(f"str_fsi_{channel}" for channel in CHANNELS),
        *(f"str_d1_{channel}" for channel in CHANNELS),
        *(f"str_d2_{channel}" for channel in CHANNELS),
        *(f"gpe_{channel}" for channel in CHANNELS),
        *(f"gpi_{channel}" for channel in CHANNELS),
        *(f"commit_fs_{channel}" for channel in CHANNELS),
    }
    for region in bridge.region_manager.regions():
        expected = 1 if region.name in inhibitory else 0
        assert set(traits[bridge.region_manager.indices(region.name)]) == {
            expected
        }


def test_selector_has_no_cross_channel_excitatory_shortcut():
    bridge = build_selector_bridge(seed=11, config=SelectorConfig())

    for channel in CHANNELS:
        other = 1 - channel
        assert not _pathways(bridge, "practice_arousal", f"str_d1_{channel}")
        assert not _pathways(bridge, f"proposal_{channel}", f"str_d1_{other}")
        assert not _pathways(bridge, f"thal_{channel}", f"commit_{other}")
        assert not _pathways(bridge, f"commit_{channel}", f"motor_{other}")


def test_selector_smoke_records_neural_threshold_without_argmax():
    result = run_condition(7, trials=2, config=SelectorConfig(
        warmup_steps=5,
        action_steps=5,
        reset_steps=2,
        washout_steps=2,
    ))

    assert result["trials"] == 2
    assert len(result["rows"]) == 2
    assert all("first_crossing" in row for row in result["rows"])
    assert all("decision_step" in row for row in result["rows"])
    assert all("region_spikes" in row for row in result["rows"])
    assert all(row["winner"] in (None, 0, 1) for row in result["rows"])
