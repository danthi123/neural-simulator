"""Direction 5 per-bridge HYBRID builder wrappers (5 functions, CPU-only spec).

Each function builds a fresh SimulationBridge with ONE category's V=16
vocab on the HYBRID substrate: validated v14/v16 bio_brain_regions
dedicated concept-pool architecture (per pillar n=96 OPTION 3 + pillar
n=105 V=32) PLUS a NEW 2000-neuron shared_concept_pool with per-concept
K=100 random sparse patterns (per G.20 sparse pillar n=95).

Architectural innovation (Direction 5): mirrors cortex's dual organization
(Mountcastle 1957 cortical columns + Pulvermuller 1999 distributed cell
assemblies). The bio dedicated pools preserve in-bridge attractor
dynamics (where pillars n=96/n=105 validated discrimination is
sufficient); the shared sparse substrate provides cross-bridge geometry
(where pillar n=95 validated cross-bridge OB 1.000 / OI 0.77 at V=160).

Cross-bridge probe (Task 4, controller-only) reads OUT of the
shared_concept_pool ONLY.

Reuses validated infrastructure BYTE-UNCHANGED:
- sim.bridge.SimulationBridge (protected)
- research.runners.text_minimal_isolation.build_biological_brain_regions
  (the protected builder; this wrapper passes each bridge's V=16 category
  vocab via existing noun_pool_names / verb_pool_names /
  adjective_pool_names parameters; the builder itself is NOT modified)
- research.runners.concept_pool_sparse_distributed.generate_sparse_patterns
  (the G.20 sparse pillar n=95 primitive; reused byte-unchanged)
- research.runners.concept_pool_sparse_distributed.apply_sparse_topographic_prior
  (the G.20 sparse pillar n=95 primitive; reused byte-unchanged at
  factor 10.0 / off-target 0.1)
- v14/v16 production recipe defaults (weak_concept_dynamics, NMDA,
  motor canon, FS interneurons)
- G.20 sparse pillar n=95 parameters (pattern_size=100, n_shared_pool=2000)

Bridge -> dedicated-pool-kind mapping (preserved from Direction 4):
- BridgeA (nouns)      -> noun_pool_names slot (dedicated kind)
- BridgeB (verbs)      -> verb_pool_names slot (dedicated kind)
- BridgeC (adj)        -> adjective_pool_names slot (dedicated kind)
- BridgeD (spatial)    -> noun_pool_names slot (no dedicated kind; the
                          substrate concept-pool architecture is category-
                          agnostic at the pool level. This preserves the
                          protected builder byte-unchanged.)
- BridgeE (functional) -> noun_pool_names slot (same rationale as Bridge D)

The HYBRID addition: ALL 5 bridges ALSO get a NEW shared_concept_pool
region (2000 neurons; weak dynamics 0.05/0.3/0.8) + shared_FS region
(300 neurons; WTA via plastic / non-plastic pathways) + 3 NEW pathways:
- language_input -> shared_concept_pool (plastic, gate="language_input_to_shared")
- shared_concept_pool -> shared_FS (non-plastic; WTA driver)
- shared_FS -> shared_concept_pool (non-plastic; WTA suppressor)

CRITICAL: These additions are APPENDED to the regions/pathways lists
RETURNED by build_biological_brain_regions; the protected builder is
NEVER modified. The wrapper assembles the combined cfg.brain_regions /
cfg.region_pathways list AFTER the protected builder returns.

No actual training happens in this module. Construction only. Training
is controller-only Task 5 (GPU-bound). The cross-bridge probe at Task 4
is CPU-only and operates on cached trained-bridge shared_concept_pool
activity.

DISCIPLINE:
- Reuse-by-import only; build_biological_brain_regions is byte-unchanged;
  concept_pool_sparse_distributed primitives are byte-unchanged.
- The v14/v16 production recipe parameters AND the G.20 sparse pillar
  n=95 parameters are pinned in this wrapper to prevent silent drift;
  future PRs that change them must update the grounding pin test
  alongside.
- CPU-LIGHT IMPORT: no cupy / SimulationBridge import at module load
  time. Imports are deferred to inside the construction functions.
"""
from __future__ import annotations
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse-by-import only. The vocab spec is frozen; the bridge dedicated
# builder is the protected text_minimal_isolation builder; the sparse
# substrate primitives are the G.20 pillar n=95 builder.
from research.findings.raw.direction_5_vocab_spec import (
    DIRECTION_5_NOUN_NAMES,
    DIRECTION_5_VERB_NAMES,
    DIRECTION_5_ADJECTIVE_NAMES,
    DIRECTION_5_SPATIAL_NAMES,
    DIRECTION_5_FUNCTIONAL_NAMES,
)


# v14/v16 production recipe defaults (pinned; mirrored from Direction 4
# builder for parity).
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


# G.20 sparse pillar n=95 parameters (pinned; mirrored from
# concept_pool_sparse_distributed.py defaults).
_G20_SPARSE_N_SHARED_POOL: int = 2000
_G20_SPARSE_N_SHARED_FS: int = 300
_G20_SPARSE_PATTERN_SIZE: int = 100        # K (per-concept)
_G20_SPARSE_LANG_TO_SHARED_DENSITY: float = 0.30
_G20_SPARSE_LANG_TO_SHARED_WEIGHT: float = 3.0
_G20_SPARSE_LANG_TO_SHARED_JITTER: float = 0.5
_G20_SPARSE_TO_FS_DENSITY: float = 0.30
_G20_SPARSE_TO_FS_WEIGHT: float = 1.0
_G20_SPARSE_FS_TO_POOL_DENSITY: float = 0.30
_G20_SPARSE_FS_TO_POOL_WEIGHT: float = 4.0
_G20_SPARSE_POOL_INTERNAL_DENSITY: float = 0.05
_G20_SPARSE_POOL_EXC_W: float = 0.3
_G20_SPARSE_POOL_INH_W: float = 0.8
_G20_SPARSE_POOL_JITTER: float = 0.2
_G20_SPARSE_TOPOGRAPHIC_FACTOR: float = 10.0   # pillar n=95 strength
_G20_SPARSE_OFF_TARGET_FACTOR: float = 0.1     # pillar n=95 strength
_G20_SPARSE_SPARSITY: float = 0.05             # ~102 active lang_input per drive

# Pre-registered V=16 per bridge (matches direction_5_vocab_spec).
_N_CONCEPTS_PER_BRIDGE: int = 16

# Per-bridge sparse-pattern seed offsets (BUG FIX 2026-05-25).
# Each bridge needs UNIQUE K-of-N patterns; without offsets all 5
# bridges share identical patterns at the same base seed which makes
# cross-bridge discrimination mathematically impossible. Fixed offsets
# (deterministic + reproducible) so multi-seed reproduces; offsets
# spaced at 100k to avoid any collision with the base_seed range.
_BRIDGE_LABEL_SEED_OFFSETS: Dict[str, int] = {
    "A_nouns":      0,
    "B_verbs":      100000,
    "C_adj":        200000,
    "D_spatial":    300000,
    "E_functional": 400000,
}


def _build_hybrid_bridge_core(
    seed: int,
    n_lang_input: int,
    n_per_pool: int,
    n_fs_per_pool: int,
    weak_dynamics: bool,
    noun_pool_names: Optional[List[str]] = None,
    verb_pool_names: Optional[List[str]] = None,
    adjective_pool_names: Optional[List[str]] = None,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
    label: str = "",
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Shared HYBRID bridge constructor body. Caller passes exactly ONE
    non-None pool name list per call; the others stay None (= that pool
    kind off in this bridge).

    HYBRID = bio dedicated pools (per build_biological_brain_regions
    byte-unchanged) + shared sparse pool (per G.20 sparse pillar n=95
    pattern_size/n_pool defaults + apply_sparse_topographic_prior at
    pillar n=95 strength).

    Args:
        seed: bridge construction seed; ALSO used to seed
              generate_sparse_patterns (deterministic per-(bridge, seed)
              sparse patterns).
        n_lang_input: 2048 per v14/v16.
        n_per_pool: 200 per v14/v16.
        n_fs_per_pool: 24 per v14/v16.
        weak_dynamics: use weak concept-pool dynamics (iter AA).
        noun_pool_names / verb_pool_names / adjective_pool_names: exactly
              ONE non-None per call (caller passes the bridge's V=16
              vocab in the chosen slot).
        n_shared_pool: 2000 per G.20 sparse n=95.
        n_shared_fs: 300 per G.20 sparse n=95.
        pattern_size: K=100 per G.20 sparse n=95.
        apply_prior: True per design doc Approach A (one-time topographic
              prior on lang_input -> shared_concept_pool at pillar n=95
              strength factor 10.0 / off-target 0.1). False is a NULL-
              control hook for diagnostics (do not use in decisive runs).
        verbose: print per-bridge build stats.
        label: bridge-name tag for [BUILD-D5-{label}] prints.

    Returns:
        (bridge, sparse_patterns, build_meta) where:
        - bridge: SimulationBridge (post-_initialize_simulation_data +
          post-apply_sparse_topographic_prior if apply_prior=True);
          ready for training (Task 5) or cross-bridge probe activity
          capture (Task 4)
        - sparse_patterns: list[list[int]] length _N_CONCEPTS_PER_BRIDGE;
          per-concept K-of-N sparse indices into the shared_concept_pool
          (LOCAL indices, in [0, n_shared_pool))
        - build_meta: dict with construction provenance for diagnostics
    """
    # Defensive: exactly one dedicated-pool slot non-None
    n_active_slots = sum(
        x is not None for x in
        (noun_pool_names, verb_pool_names, adjective_pool_names)
    )
    if n_active_slots != 1:
        raise ValueError(
            "Direction 5 per-bridge builder requires exactly ONE pool "
            "slot non-None per call; got " + str(n_active_slots)
        )

    # Imports deferred so importing this module is CPU-light (does not
    # trigger CuPy bridge initialization until construction is invoked).
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from research.runners.concept_pool_sparse_distributed import (
        generate_sparse_patterns,
        apply_sparse_topographic_prior,
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

    # ----- 1. DEDICATED SUBSTRATE (build_biological_brain_regions byte-
    #          unchanged; v14/v16 production recipe).
    dedicated_regions, dedicated_pathways = build_biological_brain_regions(
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

    # ----- 2. SHARED SPARSE SUBSTRATE (NEW; APPENDED to the dedicated
    #          regions/pathways; the protected builder is NOT modified).
    shared_concept_pool_region = BrainRegion(
        name="shared_concept_pool",
        n_neurons=n_shared_pool,
        exc_fraction=0.8,
        internal_density=_G20_SPARSE_POOL_INTERNAL_DENSITY,
        exc_weight_mean=_G20_SPARSE_POOL_EXC_W,
        inh_weight_mean=_G20_SPARSE_POOL_INH_W,
        weight_jitter=_G20_SPARSE_POOL_JITTER,
        plastic_internal=False,
    )
    shared_fs_region = BrainRegion(
        name="shared_FS",
        n_neurons=n_shared_fs,
        exc_fraction=0.0,
        internal_density=0.0,
        exc_weight_mean=0.0,
        inh_weight_mean=0.0,
        weight_jitter=0.0,
        plastic_internal=False,
    )
    # 3 new pathways: lang_input -> shared (plastic; topographic prior
    # target); shared -> shared_FS (WTA driver); shared_FS -> shared (WTA
    # suppressor).
    lang_to_shared_pathway = RegionPathway(
        from_region="language_input",
        to_region="shared_concept_pool",
        density=_G20_SPARSE_LANG_TO_SHARED_DENSITY,
        weight_mean=_G20_SPARSE_LANG_TO_SHARED_WEIGHT,
        weight_jitter=_G20_SPARSE_LANG_TO_SHARED_JITTER,
        plastic=True,
        plasticity_gate="language_input_to_shared",
    )
    shared_to_fs_pathway = RegionPathway(
        from_region="shared_concept_pool",
        to_region="shared_FS",
        density=_G20_SPARSE_TO_FS_DENSITY,
        weight_mean=_G20_SPARSE_TO_FS_WEIGHT,
        weight_jitter=0.2,
        plastic=False,
    )
    fs_to_shared_pathway = RegionPathway(
        from_region="shared_FS",
        to_region="shared_concept_pool",
        density=_G20_SPARSE_FS_TO_POOL_DENSITY,
        weight_mean=_G20_SPARSE_FS_TO_POOL_WEIGHT,
        weight_jitter=0.2,
        plastic=False,
    )

    combined_regions = list(dedicated_regions) + [
        shared_concept_pool_region, shared_fs_region,
    ]
    combined_pathways = list(dedicated_pathways) + [
        lang_to_shared_pathway, shared_to_fs_pathway, fs_to_shared_pathway,
    ]

    # ----- 3. CONFIG + BRIDGE CONSTRUCTION (v14/v16 production recipe).
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = combined_regions
    cfg.region_pathways = combined_pathways
    cfg.dt_ms = 0.5
    cfg.seed = seed
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

    # ----- 4. SPARSE PATTERNS + TOPOGRAPHIC PRIOR (G.20 sparse pillar
    #          n=95 primitives; reused byte-unchanged).
    # Deterministic per-(bridge_seed) sparse patterns. CRITICAL BUG FIX
    # (2026-05-25): the prior implementation passed `seed=seed` directly
    # to generate_sparse_patterns, causing ALL 5 bridges to receive
    # IDENTICAL K-of-N patterns at the same base seed. Cross-bridge
    # discrimination was mathematically impossible (pattern_0 in
    # A_nouns = pattern_0 in B_verbs = ... 100% overlap). This was the
    # root cause of D5 SMOKE NEGATIVE being byte-identical to D4
    # NEGATIVE: the cross-bridge probe received DUPLICATE patterns
    # across bridges. Fix: derive a bridge-specific seed from
    # (base_seed, label) so each bridge gets unique patterns.
    _bridge_seed_offset = _BRIDGE_LABEL_SEED_OFFSETS.get(label, 0)
    if _bridge_seed_offset == 0 and label != "A_nouns" and label != "":
        # Defensive: an unknown label gets a unique offset via hash
        # (but we should have all 5 known labels covered above).
        _bridge_seed_offset = (abs(hash(label)) % 900000) + 100000
    pattern_seed = seed + _bridge_seed_offset
    sparse_patterns: List[List[int]] = generate_sparse_patterns(
        n_concepts=_N_CONCEPTS_PER_BRIDGE,
        n_pool=n_shared_pool,
        pattern_size=pattern_size,
        seed=pattern_seed,
    )

    prior_meta: Dict[str, Any] = {}
    if apply_prior:
        prior_meta = apply_sparse_topographic_prior(
            bridge,
            n_concepts=_N_CONCEPTS_PER_BRIDGE,
            n_lang_input=n_lang_input,
            sparse_patterns=sparse_patterns,
            sparsity=_G20_SPARSE_SPARSITY,
            topographic_factor=_G20_SPARSE_TOPOGRAPHIC_FACTOR,
            off_target_factor=_G20_SPARSE_OFF_TARGET_FACTOR,
            n_words_for_orthogonal=_N_CONCEPTS_PER_BRIDGE,
            verbose=verbose,
        )

    build_meta: Dict[str, Any] = {
        "label": label,
        "seed": seed,
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "weak_dynamics": weak_dynamics,
        "n_shared_pool": n_shared_pool,
        "n_shared_fs": n_shared_fs,
        "pattern_size": pattern_size,
        "n_concepts": _N_CONCEPTS_PER_BRIDGE,
        "sparse_topographic_prior_applied": apply_prior,
        "sparse_topographic_factor": (
            _G20_SPARSE_TOPOGRAPHIC_FACTOR if apply_prior else None
        ),
        "sparse_off_target_factor": (
            _G20_SPARSE_OFF_TARGET_FACTOR if apply_prior else None
        ),
        "sparse_prior_meta": prior_meta,
    }

    if verbose:
        n_total = int(getattr(cfg, "num_neurons", 0)) or sum(
            r.n_neurons for r in cfg.brain_regions
        )
        print("[BUILD-D5-" + label + "] HYBRID V=16 bridge: "
              + str(n_total) + " neurons total (dedicated + sparse 2300); "
              + "n_lang_input=" + str(n_lang_input)
              + ", n_per_pool=" + str(n_per_pool)
              + ", n_fs_per_pool=" + str(n_fs_per_pool)
              + ", weak_dynamics=" + str(weak_dynamics)
              + ", n_shared_pool=" + str(n_shared_pool)
              + ", pattern_size=" + str(pattern_size)
              + ", apply_prior=" + str(apply_prior)
              + ", seed=" + str(seed), flush=True)
    return bridge, sparse_patterns, build_meta


def build_direction_5_bridge_A_nouns(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Bridge A: V=16 nouns on HYBRID bio_brain_regions + shared sparse pool.

    16 noun pools (apple, river, dog, cat + 12 extension) loaded via the
    dedicated noun_pool_names slot of the protected builder, PLUS the
    NEW 2000-neuron shared_concept_pool with 16 K=100 sparse patterns
    (deterministic per-seed). The lang_input -> shared_concept_pool
    pathway gets the pillar n=95 topographic prior.

    Returns: (bridge, sparse_patterns, build_meta).
    """
    return _build_hybrid_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_5_NOUN_NAMES,
        n_shared_pool=n_shared_pool, n_shared_fs=n_shared_fs,
        pattern_size=pattern_size, apply_prior=apply_prior,
        verbose=verbose, label="A_nouns",
    )


def build_direction_5_bridge_B_verbs(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Bridge B: V=16 verbs on HYBRID bio_brain_regions + shared sparse pool.

    16 verb pools (go, come, stop, look + 12 extension) loaded via the
    dedicated verb_pool_names slot of the protected builder, PLUS the
    NEW shared_concept_pool with 16 K=100 sparse patterns.
    """
    return _build_hybrid_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        verb_pool_names=DIRECTION_5_VERB_NAMES,
        n_shared_pool=n_shared_pool, n_shared_fs=n_shared_fs,
        pattern_size=pattern_size, apply_prior=apply_prior,
        verbose=verbose, label="B_verbs",
    )


def build_direction_5_bridge_C_adj(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Bridge C: V=16 adjectives on HYBRID bio_brain_regions + shared
    sparse pool.

    16 adjective pools (big, small, hot, cold + 12 extension) loaded via
    the dedicated adjective_pool_names slot of the protected builder,
    PLUS the NEW shared_concept_pool with 16 K=100 sparse patterns.
    """
    return _build_hybrid_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        adjective_pool_names=DIRECTION_5_ADJECTIVE_NAMES,
        n_shared_pool=n_shared_pool, n_shared_fs=n_shared_fs,
        pattern_size=pattern_size, apply_prior=apply_prior,
        verbose=verbose, label="C_adj",
    )


def build_direction_5_bridge_D_spatial(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Bridge D: V=16 spatial words on HYBRID bio_brain_regions + shared
    sparse pool.

    16 spatial words (north/east/south/west + up/down + left/right +
    in/out/near/far + top/bottom/center/side) loaded via the
    noun_pool_names slot of the protected builder, PLUS the NEW
    shared_concept_pool with 16 K=100 sparse patterns.

    Rationale: the protected builder has no dedicated spatial pool kind,
    but the concept-pool architecture is category-agnostic at the pool
    level (each pool is a 200-neuron concept attractor with FS
    interneurons + lang_input/lang_output pathways; pool kind only
    determines which builder parameter slot the names live in). This
    preserves build_biological_brain_regions byte-unchanged.
    """
    return _build_hybrid_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_5_SPATIAL_NAMES,
        n_shared_pool=n_shared_pool, n_shared_fs=n_shared_fs,
        pattern_size=pattern_size, apply_prior=apply_prior,
        verbose=verbose, label="D_spatial",
    )


def build_direction_5_bridge_E_functional(
    seed: int,
    n_lang_input: int = _V14_N_LANG_INPUT_DEFAULT,
    n_per_pool: int = _V14_N_PER_POOL_DEFAULT,
    n_fs_per_pool: int = _V14_N_FS_PER_POOL_DEFAULT,
    weak_dynamics: bool = True,
    n_shared_pool: int = _G20_SPARSE_N_SHARED_POOL,
    n_shared_fs: int = _G20_SPARSE_N_SHARED_FS,
    pattern_size: int = _G20_SPARSE_PATTERN_SIZE,
    apply_prior: bool = True,
    verbose: bool = False,
) -> Tuple[Any, List[List[int]], Dict[str, Any]]:
    """Bridge E: V=16 functional words on HYBRID bio_brain_regions + shared
    sparse pool.

    16 functional words (pronouns + articles + conjunctions + prepositions
    + demonstratives + wh-words) loaded via the noun_pool_names slot of
    the protected builder, PLUS the NEW shared_concept_pool with 16
    K=100 sparse patterns. Same noun_pool_names slot mapping rationale
    as Bridge D.
    """
    return _build_hybrid_bridge_core(
        seed=seed, n_lang_input=n_lang_input, n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool, weak_dynamics=weak_dynamics,
        noun_pool_names=DIRECTION_5_FUNCTIONAL_NAMES,
        n_shared_pool=n_shared_pool, n_shared_fs=n_shared_fs,
        pattern_size=pattern_size, apply_prior=apply_prior,
        verbose=verbose, label="E_functional",
    )
