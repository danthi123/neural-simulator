"""Direction 3 V=32 bridge builder wrapper.

Builds a fresh SimulationBridge with the V=32 vocab layout (4 motor +
12 noun + 12 verb + 4 adjective = 32 distinct concept pools) on the
validated v14/v16 bio_brain_regions concept-pool architecture. This is
the cheapest first probe of Option A (more pools) from the Direction 3
design doc: the existing build_biological_brain_regions API already
accepts arbitrary-length noun_pool_names / verb_pool_names /
adjective_pool_names parameters.

Reuses validated infrastructure byte-unchanged:
- sim.bridge.SimulationBridge (protected)
- research.runners.text_minimal_isolation.build_biological_brain_regions
  (the protected builder; this wrapper passes V=32 vocab via existing
  parameters; the builder itself is NOT modified)
- v14/v16 production recipe defaults (weak_concept_dynamics, NMDA,
  motor canon, FS interneurons)

The wrapper mirrors research.runners.concept_pool_demo.build_concept_bridge
but with the V=32 vocab spec. It does NOT use that function directly
because that one is hard-wired to v14/v16's 4+4+4+4=16 vocab; this
wrapper extends to V=32 via the vocab spec module's frozen lists.
"""
from __future__ import annotations
import os
import sys
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import only. The vocab spec is frozen; the bridge builder
# is the protected text_minimal_isolation builder.
from research.findings.raw.direction_3_vocab_spec import (
    DIRECTION_3_NOUN_NAMES,
    DIRECTION_3_VERB_NAMES,
    DIRECTION_3_ADJECTIVE_NAMES,
    DIRECTION_3_V32_TOTAL,
)


def build_direction_3_v32_bridge(
    seed: int,
    n_lang_input: int = 2048,
    n_per_pool: int = 200,
    n_fs_per_pool: int = 24,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Construct a V=32 bridge for the vocab-scaling probe.

    Args:
        seed: RNG seed for connectivity + dynamics noise.
        n_lang_input: language_input region size (v14/v16 default 2048;
                      reduce to 1024 for smoke).
        n_per_pool: neurons per concept pool (v14/v16 default 200; reduce
                    to 100 for smoke).
        n_fs_per_pool: FS interneurons per concept pool (v14/v16 default 24
                       = 12% of 200).
        weak_dynamics: use weak concept-pool dynamics (0.05 density, 0.3
                       exc, 0.8 inh) per v14/v16 production recipe; the
                       canon (0.10/2.0/4.0) amplifies structural bias at
                       larger pool counts.
        verbose: print build info.

    Returns:
        SimulationBridge configured for V=32 V=32 vocab on the
        bio_brain_regions concept-pool substrate. Has 4 motor + 12 noun +
        12 verb + 4 adjective = 32 distinct concept pools, each with FS
        cross-inhibition + reciprocal lang_input/lang_output pathways.
    """
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    # Pre-registered V=32 sanity (defence-in-depth; the spec module is
    # the source of truth, but if anyone changes the lists this fires).
    expected_total = (
        len(["N", "E", "S", "W"])  # motor (hard-canon in builder)
        + len(DIRECTION_3_NOUN_NAMES)
        + len(DIRECTION_3_VERB_NAMES)
        + len(DIRECTION_3_ADJECTIVE_NAMES)
    )
    if expected_total != DIRECTION_3_V32_TOTAL:
        raise ValueError(
            "Direction 3 V=32 spec mismatch: vocab spec computes "
            + str(expected_total) + " concepts but DIRECTION_3_V32_TOTAL "
            "is " + str(DIRECTION_3_V32_TOTAL)
        )

    # Per the v14/v16 production recipe: weak concept-pool dynamics
    # prevent off-target pools from self-sustaining; motor pools keep
    # cortical canon. See research.runners.concept_pool_demo for the
    # validated parameter set.
    concept_internal_density = 0.05 if weak_dynamics else None
    concept_exc_weight = 0.3 if weak_dynamics else None
    concept_inh_weight = 0.8 if weak_dynamics else None

    # Motor pools use canon dynamics (v14/v16 production default).
    motor_internal_density = 0.10
    motor_exc_weight = 2.0
    motor_inh_weight = 4.0

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=motor_internal_density,
        motor_exc_weight_mean=motor_exc_weight,
        motor_inh_weight_mean=motor_inh_weight,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        # V=32 noun pools (12; extends v14's 4)
        enable_noun_pools=True,
        noun_pool_names=DIRECTION_3_NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        # V=32 verb pools (12; extends v14's 4)
        enable_verb_pools=True,
        verb_pool_names=DIRECTION_3_VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        # V=32 adjective pools (4; v14 baseline)
        enable_adjective_pools=True,
        adjective_pool_names=DIRECTION_3_ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        # Per-kind dynamics
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0  # Above design weights; v14/v16 production default
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    if verbose:
        rm = bridge.region_manager
        all_pools = (
            ["motor_N", "motor_E", "motor_S", "motor_W"]
            + ["noun_pool_" + n for n in DIRECTION_3_NOUN_NAMES]
            + ["verb_pool_" + v for v in DIRECTION_3_VERB_NAMES]
            + ["adjective_pool_" + a for a in DIRECTION_3_ADJECTIVE_NAMES]
        )
        n_pool_neurons = sum(len(list(rm.indices(p))) for p in all_pools)
        n_total = int(getattr(cfg, "num_neurons", 0)) or sum(
            r.n_neurons for r in cfg.brain_regions
        )
        print("[BUILD-3-V32] V=" + str(DIRECTION_3_V32_TOTAL)
              + " bridge: " + str(n_total) + " neurons total, "
              + str(len(all_pools)) + " concept pools ("
              + str(n_pool_neurons) + " neurons in pools); "
              + "n_lang_input=" + str(n_lang_input)
              + ", n_per_pool=" + str(n_per_pool)
              + ", n_fs_per_pool=" + str(n_fs_per_pool)
              + ", weak_dynamics=" + str(weak_dynamics),
              flush=True)
    return bridge
