import os

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    DIRECT_PATH_GATE,
    SelectorConfig,
    _topology_summary,
    build_selector_bridge,
    run_condition,
    selector_config,
)
from sim.backend import to_host
from sim.enums import NeuronType
from sim.regions import BrainRegion, RegionPathway


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


def test_v2_removes_counterproductive_striatal_fsi_branch():
    bridge = build_selector_bridge(seed=13, config=selector_config("v2"))
    region_names = {region.name for region in bridge.region_manager.regions()}

    assert bridge.core_config.num_neurons == 600
    assert len(bridge.core_config.region_pathways) == 36
    assert not any(name.startswith("str_fsi_") for name in region_names)
    assert not any(
        pathway.from_region.startswith("str_fsi_")
        or pathway.to_region.startswith("str_fsi_")
        for pathway in bridge.core_config.region_pathways
    )


def test_selector_versions_preserve_v1_and_reduce_v2_topology():
    assert selector_config("v1").enable_striatal_fsi is True
    assert selector_config("v2").enable_striatal_fsi is False
    assert _topology_summary(selector_config("v1")) == {
        "neurons": 632,
        "declared_pathways": 44,
    }
    assert _topology_summary(selector_config("v2")) == {
        "neurons": 600,
        "declared_pathways": 36,
    }


def test_selector_builder_can_add_declared_learning_regions_and_settings():
    bridge = build_selector_bridge(
        seed=17,
        config=selector_config("v2"),
        extra_regions=[BrainRegion(
            name="test_credit",
            n_neurons=4,
            exc_fraction=1.0,
            internal_density=0.0,
            plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
            enable_homeostasis=False,
        )],
        extra_pathways=[RegionPathway(
            from_region="practice_arousal",
            to_region="test_credit",
            density=1.0,
            weight_mean=1.0,
            weight_jitter=0.0,
            plastic=False,
        )],
        core_config_updates={"enable_reward_modulation": True},
    )

    assert bridge.core_config.num_neurons == 604
    assert bridge.core_config.enable_reward_modulation is True
    assert len(bridge.core_config.region_pathways) == 37
    assert len(bridge.region_manager.indices("test_credit")) == 4


def test_d1_d2_asymmetry_uses_vocal_msn_cell_types_not_region_spelling():
    bridge = build_selector_bridge(
        seed=19,
        config=selector_config("v2"),
        core_config_updates={"enable_d1_d2_asymmetry": True},
    )
    postsynaptic = np.asarray(to_host(bridge.cp_connections.indices))
    signs = np.asarray(to_host(bridge.cp_d1_d2_sign))[:postsynaptic.size]

    for channel in CHANNELS:
        d1 = np.asarray(bridge.region_manager.indices(f"str_d1_{channel}"))
        d2 = np.asarray(bridge.region_manager.indices(f"str_d2_{channel}"))
        assert np.all(signs[np.isin(postsynaptic, d1)] == 1.0)
        assert np.all(signs[np.isin(postsynaptic, d2)] == -1.0)


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
