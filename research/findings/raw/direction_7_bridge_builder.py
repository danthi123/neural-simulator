"""Direction 7 per-bridge builder wrappers (5 functions, CPU-only spec).

Each function builds a fresh SimulationBridge with ONE category's V=64
vocab on the validated v14/v16 bio_brain_regions concept-pool architecture.
This is the 2x-vocab extension of Direction 6 (V=32 per bridge) per the
pillar n=109 D6 dedicated-pool result, scaling to V=64 per bridge x
5 bridges = 320 cross-bridge concepts -- matching the Direction M G.20
sparse production deliverable vocabulary byte-for-byte on a biology-
faithful substrate.

Reuses validated infrastructure byte-unchanged:
- sim.bridge.SimulationBridge (protected)
- research.runners.text_minimal_isolation.build_biological_brain_regions
  (the protected builder; this wrapper passes each bridge's V=64 category
  vocab via existing noun_pool_names / verb_pool_names /
  adjective_pool_names parameters; the builder itself is NOT modified)
- v14/v16 production recipe defaults (weak_concept_dynamics, NMDA,
  motor canon, FS interneurons)

Bridge -> pool-kind mapping (same as D4 / D6):
- BridgeA (nouns)      -> noun_pool_names slot (dedicated kind)
- BridgeB (verbs)      -> verb_pool_names slot (dedicated kind)
- BridgeC (adj)        -> adjective_pool_names slot (dedicated kind)
- BridgeD (spatial)    -> noun_pool_names slot (no dedicated kind; the
                          substrate concept-pool architecture is category-
                          agnostic at the pool level. This preserves the
                          protected builder byte-unchanged.)
- BridgeE (functional) -> noun_pool_names slot (same rationale as Bridge D)

No actual training happens in this module. Construction only. Training is
controller-only Task 5 (GPU-bound). The cross-bridge probe at Task 4 is
CPU-only and operates on cached trained-bridge activity.

DISCIPLINE:
- Reuse-by-import only; build_biological_brain_regions is byte-unchanged.
- The v14/v16 production recipe parameters are pinned in this wrapper to
  prevent silent drift; future PRs that change them must update the
  grounding pin test alongside.
- Bridge-specific seed offsets MANDATORY (analog of D5 c4e18f2 + D4 fix
  + D6 fix); 100k spacing per bridge to avoid byte-identical weight
  initialization across the 5 bridges.
"""
from __future__ import annotations
import os
import sys
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import only. The vocab spec is frozen; the bridge builder is
# the protected text_minimal_isolation builder.
from research.findings.raw.direction_7_vocab_spec import (
    DIRECTION_7_NOUN_NAMES,
    DIRECTION_7_VERB_NAMES,
    DIRECTION_7_ADJECTIVE_NAMES,
    DIRECTION_7_SPATIAL_NAMES,
    DIRECTION_7_FUNCTIONAL_NAMES,
)


# Per-bridge SimulationBridge seed offsets (analog of D5 commit c4e18f2
# + D4/D6 _DIRECTION_{4,6}_BRIDGE_LABEL_SEED_OFFSETS). Each bridge needs
# UNIQUE random initialization of lang_input -> pool weight matrices;
# without offsets all 5 bridges share identical base_seed (e.g. 42) and
# the protected build_biological_brain_regions / SimulationBridge
# construction produces byte-identical pool weights across bridges. Since
# orthogonal_drive_pattern is deterministic per (cue_idx, n_cues),
# identical weights drive identical pool activity per cue position,
# making cross-bridge discrimination mathematically impossible. Fixed
# offsets (deterministic + reproducible) so multi-seed reproduces;
# offsets spaced at 100k to avoid any collision with the base_seed range.
_DIRECTION_7_BRIDGE_LABEL_SEED_OFFSETS: dict = {
    "A_nouns":      0,
    "B_verbs":      100000,
    "C_adj":        200000,
    "D_spatial":    300000,
    "E_functional": 400000,
}


# v14/v16 production recipe defaults (pinned).
_V14_N_LANG_INPUT_DEFAULT: int = 2048
_V14_N_PER_POOL_DEFAULT: int = 200
_V14_N_FS_PER_POOL_DEFAULT: int = 24
# Weak concept-pool dynamics (iter AA recipe). Motor pools keep canon
# (0.10/2.0/4.0).
_V14_WEAK_CONCEPT_DENSITY: float = 0.05
_V14_WEAK_CONCEPT_EXC_W: float = 0.3
_V14_WEAK_CONCEPT_INH_W: float = 0.8
_V14_MOTOR_DENSITY: float = 0.10
_V14_MOTOR_EXC_W: float = 2.0
_V14_MOTOR_INH_W: float = 4.0
_V14_TEXT_TO_MOTOR_DENSITY: float = 0.30
_V14_TEXT_TO_MOTOR_WEIGHT: float = 3.0
_V14_TEXT_TO_MOTOR_JITTER: float = 0.5
_V14_MOTOR_TO_LANG_OUT_WEIGHT: float = 2.0
_V14_NMDA_TAU_DECAY_MS: float = 100.0  # Wang 2002 calibration
_V14_STDP_W_MAX: float = 8.0  # Above design weights


def _build_bridge_core(
    seed: int,
    n_lang_input: int,
    n_per_pool: int,
    n_fs_per_pool: int,
    weak_dynamics: bool,
    noun_pool_names: Optional[List[str]] = None,
    verb_pool_names: Optional[List[str]] = None,
    adjective_pool_names: Optional[List[str]] = None,
    verbose: bool = False,
    label: str = "",
):
    """Shared bridge constructor body. Caller passes exactly ONE
    non-None pool name list per call; the others stay None
    (= that pool kind off in this bridge).

    Returns:
        SimulationBridge constructed via the protected
        build_biological_brain_regions; ready for training (Task 5) or
        cross-bridge probe activity capture (Task 4).
    """
    # Defensive: exactly one pool slot non-None
    n_active_slots = sum(
        x is not None for x in
        (noun_pool_names, verb_pool_names, adjective_pool_names)
    )
    if n_active_slots != 1:
        raise ValueError(
            "Direction 7 per-bridge builder requires exactly ONE pool "
            "slot non-None per call; got " + str(n_active_slots)
        )

    # CRITICAL BUG FIX (analog of D5 c4e18f2 + D4 fix 2026-05-26 + D6
    # fix 2026-05-27): derive a bridge-specific seed from
    # (base_seed, label) so each of the 5 bridges initializes its
    # lang_input -> pool weight matrices uniquely. Without this fix, all
    # 5 bridges at the same base seed produce byte-identical pool
    # activity per cue position (orthogonal_drive_pattern is
    # deterministic; identical weights + identical drive = identical
    # activity), making the cross-bridge probe operate on duplicate
    # inputs.
    _bridge_seed_offset = _DIRECTION_7_BRIDGE_LABEL_SEED_OFFSETS.get(
        label, 0,
    )
    if _bridge_seed_offset == 0 and label != "A_nouns" and label != "":
        # Defensive: an unknown label gets a unique offset via hash
        # (but we should have all 5 known labels covered above).
        _bridge_seed_offset = (abs(hash(label)) % 900000) + 100000
    bridge_seed = int(seed) + int(_bridge_seed_offset)

    # Imports deferred so importing this module is CPU-light (does not
    # trigger CuPy bridge initialization until construction is invoked).
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    concept_internal_density = (
        _V14_WEAK_CONCEPT_DENSITY if weak_dynamics else None
    )
    concept_exc_weight = (
        _V14_WEAK_CONCEPT_EXC_W if weak_dynamics else None
    )
    concept_inh_weight = (
        _V14_WEAK_CONCEPT_INH_W if weak_dynamics else None
    )

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=_V14_MOTOR_DENSITY,
        motor_exc_weight_mean=_V14_MOTOR_EXC_W,
        motor_inh_weight_mean=_V14_MOTOR_INH_W,
        text_input_to_motor_density=_V14_TEXT_TO_MOTOR_DENSITY,
        text_input_to_motor_weight=_V14_TEXT_TO_MOTOR_WEIGHT,
        text_input_to_motor_jitter=_V14_TEXT_TO_MOTOR_JITTER,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=_V14_MOTOR_TO_LANG_OUT_WEIGHT,
        # Per-bridge category vocab loaded in exactly ONE slot
        enable_noun_pools=(noun_pool_names is not None),
        noun_pool_names=noun_pool_names,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=(verb_pool_names is not None),
        verb_pool_names=verb_pool_names,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=(adjective_pool_names is not None),
        adjective_pool_names=adjective_pool_names,
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
    cfg.seed = bridge_seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = _V14_NMDA_TAU_DECAY_MS
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = _V14_STDP_W_MAX
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
        n_total = int(getattr(cfg, "num_neurons", 0)) or sum(
            r.n_neurons for r in cfg.brain_regions
        )
        print("[BUILD-D7-" + label + "] V=64 bridge: "
              + str(n_total) + " neurons total; n_lang_input="
              + str(n_lang_input) + ", n_per_pool=" + str(n_per_pool)
              + ", n_fs_per_pool=" + str(n_fs_per_pool)
              + ", weak_dynamics=" + str(weak_dynamics)
              + ", base_seed=" + str(seed)
              + ", bridge_seed=" + str(bridge_seed)
              + " (offset=" + str(_bridge_seed_offset) + ")",
              flush=True)
    return bridge


def build_direction_7_bridge_A_nouns(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Bridge A: V=64 nouns on bio_brain_regions concept-pool substrate.

    64 noun pools (matches G.20 g20_bridgeA_nouns_vocab64.txt) loaded
    via the dedicated noun_pool_names slot of the protected builder.
    """
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_7_NOUN_NAMES,
        verbose=verbose, label="A_nouns",
    )


def build_direction_7_bridge_B_verbs(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Bridge B: V=64 verbs on bio_brain_regions concept-pool substrate.

    64 verb pools (matches G.20 g20_bridgeB_verbs_vocab64.txt) loaded
    via the dedicated verb_pool_names slot of the protected builder.
    """
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        verb_pool_names=DIRECTION_7_VERB_NAMES,
        verbose=verbose, label="B_verbs",
    )


def build_direction_7_bridge_C_adj(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Bridge C: V=64 adjectives on bio_brain_regions concept-pool substrate.

    64 adjective pools (matches G.20 g20_bridgeC_adj_vocab64.txt) loaded
    via the dedicated adjective_pool_names slot of the protected builder.
    """
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        adjective_pool_names=DIRECTION_7_ADJECTIVE_NAMES,
        verbose=verbose, label="C_adj",
    )


def build_direction_7_bridge_D_spatial(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Bridge D: V=64 spatial words on bio_brain_regions concept-pool
    substrate.

    64 spatial words (matches G.20 g20_bridgeD_spatial_vocab64.txt
    covering north/south/east/west + up/down/left/right + here/there/
    near/far + above/below/inside/outside + temporal connectives
    yesterday/today/tomorrow/now/then/etc) loaded via the noun_pool_names
    slot of the protected builder.

    Rationale: the protected builder has no dedicated spatial pool
    kind, but the concept-pool architecture is category-agnostic at the
    pool level (each pool is a 200-neuron concept attractor with FS
    interneurons + lang_input/lang_output pathways; pool kind only
    determines which builder parameter slot the names live in). This
    preserves build_biological_brain_regions byte-unchanged.
    """
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_7_SPATIAL_NAMES,
        verbose=verbose, label="D_spatial",
    )


def build_direction_7_bridge_E_functional(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    verbose: bool = False,
):
    """Bridge E: V=64 functional words on bio_brain_regions concept-pool
    substrate.

    64 functional words (matches G.20 g20_bridgeE_functional_vocab64.txt
    covering number words one..ten + zero/half + pronouns + wh-words +
    discourse markers + auxiliaries + conjunctions) loaded via the
    noun_pool_names slot of the protected builder. Same rationale as
    Bridge D (category-agnostic pool architecture).
    """
    return _build_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_7_FUNCTIONAL_NAMES,
        verbose=verbose, label="E_functional",
    )


# Catalog of bridge-builder functions for the multi-seed runner (Task 5,
# controller-only). Lets the runner iterate over the 5 bridges via name
# lookup without re-importing per call.
DIRECTION_7_BRIDGE_BUILDERS = {
    "A_nouns": build_direction_7_bridge_A_nouns,
    "B_verbs": build_direction_7_bridge_B_verbs,
    "C_adj": build_direction_7_bridge_C_adj,
    "D_spatial": build_direction_7_bridge_D_spatial,
    "E_functional": build_direction_7_bridge_E_functional,
}
