"""Trained ec_context(position)->concept-readout encode + deterministic
position-sweep producer, over the MULTI-SEED-VALIDATED D.11/P4.1
DG->CA3 positional store (order-intrinsic slice, Task 5).

RETARGETED 2026-05-16 (spec-compliance fix). The first Task-5 cut
(commit 89113c8) built on `concept_pool_demo.build_concept_bridge(
enable_positional_context=True)` -- the *weak ~50% prototype's*
bridge, which wires ec_context -> concept-pools DIRECTLY with NO
DG/CA3 and whose (word,position) distinctness is only SINGLE-seed
(`test_positional_binding_concept_pool.py`). The design
(`docs/plans/2026-05-16-order-intrinsic-conversational-memory-design.md`
"Evidence grounding" + "Architecture") grounds the line on the
multi-seed-validated (3/3, `2026-05-11-P41-positional-multiseed.md`)
P4.1 substrate built by `text_minimal_isolation.build_biological_
brain_regions(enable_hippocampus_consolidation=True,
enable_episodic_context=True)` -> ec_context -> dg -> CA3, which
`research/runners/validate_positional_binding.py` validates and which
Task 6's no-harm re-runs UNCHANGED. Building Task 5 on the OTHER
substrate would make Task 6's no-harm vacuous (it would protect a
store the architecture never uses) and the design's
"multi-seed-validated-reuse" premise FALSE as implemented. This module
is retargeted to that DG/CA3 store so the no-harm is MEANINGFUL and the
premise TRUE.

THE NET-NEW MECHANISM of this line. The 6-negative terminated line
tried to LEARN an ordered sequence model; this does NOT. Order is
INTRINSIC: it lives in the deterministic D.11 positional code
(`sim.text_embeddings.positional_drive_pattern`) and is read back by a
plain position sweep. The single net-new behavior over the weak ~50%
read-back (`validate_positional_binding`'s query is the raw associative
trace -- here we ADD a *trained* read-back) is ONE additive plastic
`ec_context -> motor_{N,E,S,W}` pathway (gate
`ec_context_to_motor_readback`, DISJOINT from every validated store
gate) that is co-firing-strengthened DURING the existing validated
DG->CA3 co-drive encode (Tonegawa engram binds all co-active elements;
catalog D.14 / D.02), so a later `ec_context(position)`-alone sweep
drives the bound concept's readout strongly instead of leaving it at
the raw-trace ~50% floor.

WHY motor_{N,E,S,W} IS THE READ-BACK TARGET. The validated DG/CA3
bridge (`validate_positional_binding.run_positional_validation`) has
NO noun/verb/adjective concept pools -- its only per-concept readout
regions are `motor_{N,E,S,W}` (8 neurons each), which are also the
validated decoder readout for this bridge (`ca1 -> motor_{action}`
consolidation pathways, gate `ca1_to_motor`, already exist on it).
`order_intrinsic_core.decode_position_sweep` consumes a per-position
`{concept: rate}` dict; the concept vocab for this substrate is
therefore the four direction words {north,east,south,west} ->
{motor_N,motor_E,motor_S,motor_W} (the `text_minimal_isolation`
ACTION_NAMES set). The net-new trained pathway makes
`ec_context(position) -> the position-k word's motor pool` strong,
WITHOUT touching the validated ec_context->dg->CA3 store-distinctness
path (separation of concerns -- the v12/v13/v15/G1 lesson).

DRY (nothing here is reimplemented):
  - bridge construction          : copy
    `validate_positional_binding.run_positional_validation`'s
    build_biological_brain_regions(...) + cfg.* idiom EXACTLY (the
    3/3-multi-seed store), then APPEND the single net-new plastic
    `ec_context -> motor_{action}` pathway BEFORE
    _initialize_simulation_data. The validated store regions/pathways
    /gates are byte-for-byte unchanged.
  - positional code              : reuse
    `sim.text_embeddings.positional_drive_pattern` (D.11; deterministic).
  - the co-drive encode idiom    : reuse
    `validate_positional_binding.encode_and_tag`'s exact drive idiom
    (lang_input(word) + ec_context(positional_drive_pattern) through
    the SAME validated store gates `ca3_swr_burst/dg_to_ca3/ec_to_dg/
    ec_context_to_dg/lang_to_ec`) + the `sim.bridge` engram API
    (start_engram_recording / commit_engram_tag, region_filter=
    ["ca3"], matching the validated store).
  - the position-sweep readback  : reuse
    `test_word_order_discrimination.query_position`'s exact
    ec_context(position)-alone drive + per-region spike-rate read
    mechanism, applied to the motor_{action} readout regions.
    readback_sweep is write-only into the substrate (external current
    only) -- no feedback, no new pathway.
  - word<->action map            : the four-direction subset of
    `test_word_order_discrimination._WORD_TO_IDX` (orthogonal code
    indices) -> motor_{action}; do NOT redefine the vocab.

SEPARATION OF CONCERNS (the v12/v13/v15/G1 "first do no harm" lesson;
Task 6 hard-checks this). During the order-intrinsic encode we open
the validated store's OWN plasticity gates EXACTLY as
`validate_positional_binding.encode_and_tag` does (`ca3_swr_burst`,
`dg_to_ca3`, `ec_to_dg`, `ec_context_to_dg`, `lang_to_ec`) -- so the
(word,position)->distinct-CA3 encoding is byte-identical to the
validated path -- PLUS exactly one net-new gate
`ec_context_to_motor_readback` for the additive read-back pathway.
Every gate is opened in a try/finally and CLOSED at the end of the
encode -- plasticity never bleeds past the order-intrinsic encode. The
net-new gate `ec_context_to_motor_readback` is DISJOINT from every
validated store gate (`ec_context_to_dg`, `ec_to_dg`, `dg_to_ca3`,
`ca3_swr_burst`, `lang_to_ec`, `ec_to_ca1`, `ca3_to_ca1`,
`ca1_to_motor`, `ca1_to_lang_out`, `motor_to_language_output`), so
Task 6's UNCHANGED re-run of `validate_positional_binding` genuinely
protects the store the architecture now actually uses, and the
design's multi-seed-validated-reuse premise is TRUE as implemented.

Conventions: ASCII-only print() (Windows cp1252). Heavy imports
(sim.*, validate_positional_binding, the prototype) are LAZY inside
the functions so the import/signature smoke is instant.
"""
from __future__ import annotations

from typing import Dict, List

# The four direction words this DG/CA3 substrate can read back: the
# only per-concept readout regions on the validated bridge are
# motor_{N,E,S,W}. Index is the orthogonal_drive_pattern cue_idx
# (subset of test_word_order_discrimination._WORD_TO_IDX); pool is the
# motor region. Vocab is NOT redefined -- this is that map's direction
# subset.
_WORD_TO_ACTION = {
    "north": "N", "east": "E", "south": "S", "west": "W",
}
_ACTION_NAMES = ["N", "E", "S", "W"]

# The single net-new plasticity gate -- the additive trained read-back
# pathway ec_context -> motor_{action}. DISJOINT from every validated
# DG/CA3 store gate (see module docstring). Task 6's UNCHANGED re-run
# of validate_positional_binding never references this gate.
_READBACK_GATE = "ec_context_to_motor_readback"

# The validated store's OWN plasticity gates, opened during the
# co-drive EXACTLY as validate_positional_binding.encode_and_tag opens
# them (so the (word,position)->distinct-CA3 encoding is byte-identical
# to the validated path). NOT net-new -- reused verbatim.
_VALIDATED_STORE_GATES = (
    "ca3_swr_burst", "dg_to_ca3", "ec_to_dg",
    "ec_context_to_dg", "lang_to_ec",
)


def build_order_intrinsic_bridge(seed: int,
                                  n_lang_input: int = 1024,
                                  n_ec: int = 200,
                                  n_dg: int = 800,
                                  n_dg_pv_basket: int = 240,
                                  n_ca3: int = 400,
                                  n_ca1: int = 200,
                                  n_ec_context: int = 200,
                                  ca3_recurrent_weight: float = 5.0,
                                  ec_context_to_motor_density: float = 0.30,
                                  ec_context_to_motor_weight: float = 3.0,
                                  verbose: bool = False):
    """Build the MULTI-SEED-VALIDATED D.11/P4.1 DG->CA3 positional
    store + the (now to-be-trained) additive plastic
    `ec_context -> motor_{action}` read-back pathway.

    DRY: the regions/pathways/cfg are copied EXACTLY from
    `validate_positional_binding.run_positional_validation` (the store
    that is 3/3 multi-seed PASS in `2026-05-11-P41-positional-
    multiseed.md`). The ONLY addition is one net-new plastic
    `ec_context -> motor_{N,E,S,W}` RegionPathway tagged
    `ec_context_to_motor_readback`, appended to cfg.region_pathways
    BEFORE _initialize_simulation_data so its gate is registered. The
    validated store's regions, pathways and gates are byte-for-byte
    unchanged; the net-new gate is disjoint from all of them, so
    Task 6's UNCHANGED re-run of validate_positional_binding still
    protects exactly the store this bridge uses.

    Returns the constructed SimulationBridge.
    """
    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                            VisualizationConfig)
    from sim.bridge import SimulationBridge
    from sim.regions import RegionPathway
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    # --- EXACT copy of validate_positional_binding's store build ---
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=8, n_motor_fs_per_action=2,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket,
        n_ca3=n_ca3, n_ca1=n_ca1,
        ca3_recurrent_weight=ca3_recurrent_weight,
        enable_episodic_context=True,
        n_ec_context=n_ec_context,
    )

    # --- the SINGLE net-new mechanism: additive trained read-back ---
    # ec_context -> motor_{action}, plastic, its OWN disjoint gate.
    # Appended BEFORE init so set_plasticity_gate(_READBACK_GATE) works.
    # Touches NONE of the validated store's regions/pathways/gates.
    for action in _ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="ec_context",
            to_region=f"motor_{action}",
            density=ec_context_to_motor_density,
            weight_mean=ec_context_to_motor_weight,
            weight_jitter=0.2,
            plastic=True,
            plasticity_gate=_READBACK_GATE,
        ))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.fast_spike_reset = True
    cfg.stdp_w_max = 10.0
    cfg.enable_hebbian_learning = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)

    if verbose:
        try:
            nnz = int(bridge.cp_connections.nnz)
        except Exception:
            nnz = -1
        print("[BUILD] order-intrinsic DG/CA3 bridge: %d neurons, "
              "%d synapses (validated P4.1 store + 1 net-new "
              "ec_context->motor read-back pathway, gate=%s)"
              % (int(getattr(cfg, "num_neurons", 0)), nnz,
                 _READBACK_GATE), flush=True)

    return bridge


def _open_gates(bridge, names) -> List[str]:
    """Open (set to 1.0) every plasticity gate in `names` that EXISTS
    on this bridge. A name absent from the active wiring plan raises
    KeyError in set_plasticity_gate -> swallowed (no-op). Defensive
    guard so a gate from a different architecture can never be touched
    by mistake. Returns the gates actually opened."""
    opened: List[str] = []
    for g in names:
        try:
            bridge.set_plasticity_gate(g, 1.0)
            opened.append(g)
        except Exception:
            pass
    return opened


def _close_gates(bridge, names) -> None:
    for g in names:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass


def encode_proposition(bridge,
                        concept_words: List[str],
                        tag_name: str = None,
                        n_lang_input: int = 1024,
                        word_seed: int = 42,
                        n_ec_context: int = 200,
                        n_max_positions: int = 10,
                        positional_sparsity: float = 0.1,
                        encoding_steps: int = 100,
                        warmup_steps: int = 30,
                        drive_pA: float = 200.0,
                        ec_drive_pA: float = 200.0,
                        teacher_pA: float = 500.0,
                        top_k: int = 50,
                        verbose: bool = False) -> str:
    """Encode ONE proposition as a single Tonegawa engram over the
    co-active CA3 (concept @ position) set, WITH the net-new
    `ec_context -> motor_{action}` read-back pathway plastic +
    co-firing-strengthened.

    For k, word in enumerate(concept_words): drive lang_input(word)
    (deterministic sparse word pattern, same idiom as
    validate_positional_binding.build_word_pattern) AND
    ec_context(positional_drive_pattern(k)) through the DG->CA3 store
    EXACTLY as validate_positional_binding.encode_and_tag does, AND
    teacher current on the word's motor_{action} readout pool,
    SIMULTANEOUSLY, for `encoding_steps` steps. The validated store
    gates (`ca3_swr_burst/dg_to_ca3/ec_to_dg/ec_context_to_dg/
    lang_to_ec`) are opened EXACTLY as the validated path opens them
    (so the (word,position)->distinct-CA3 encoding is byte-identical),
    PLUS the net-new `ec_context_to_motor_readback` gate so STDP/Hebbian
    co-firing strengthens position-k -> the word's motor pool. The
    entire multi-position drive is wrapped in one
    start_engram_recording / commit_engram_tag(region_filter=["ca3"])
    so the proposition is ONE engram over the CA3 ensemble (D.14),
    matching the validated store's tagging exactly.

    DRY: the per-(word,position) drive block + the validated-store gate
    set + the engram API call are reused verbatim from
    validate_positional_binding.encode_and_tag. The ONLY net-new
    behavior is also opening _READBACK_GATE during this co-drive +
    the teacher current on the readout pool (the read-back the raw-trace
    ~50% query lacked).

    SEPARATION OF CONCERNS: gates are opened in a try/finally and
    CLOSED before return -- plasticity never bleeds past this encode.
    The net-new gate is DISJOINT from every validated store gate.

    Returns the committed engram tag name.
    """
    import numpy as np
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import positional_drive_pattern

    if tag_name is None:
        tag_name = "prop_" + "_".join(concept_words)

    rm = bridge.region_manager
    lang_indices = list(rm.indices("language_input"))
    ec_context_indices = list(rm.indices("ec_context"))
    n_lang = len(lang_indices)
    n_ctx = len(ec_context_indices)
    n_total = bridge.cp_external_input_current.shape[0]

    # Per-word deterministic sparse lang_input pattern -- the EXACT
    # idiom of validate_positional_binding.build_word_pattern (hash +
    # seeded rng, 10% active), so the store sees the same kind of word
    # drive it was validated with.
    def _word_pattern(word: str):
        rng = np.random.default_rng(
            (hash(word) ^ int(word_seed)) % (2 ** 31))
        n_active = max(1, int(0.1 * n_lang))
        return rng.choice(n_lang, size=n_active,
                          replace=False).astype(np.int64)

    # Settle transients before recording (validated path warms up too).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(int(warmup_steps)):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    opened: List[str] = []
    try:
        # Open the validated store's OWN gates EXACTLY as the validated
        # encode does, PLUS the single net-new read-back gate. Absent
        # gates are no-ops (defensive guard).
        opened = _open_gates(
            bridge, list(_VALIDATED_STORE_GATES) + [_READBACK_GATE])
        if verbose:
            print("[ENCODE] '%s' (tag=%s) gates_open=%s"
                  % (" ".join(concept_words), tag_name, opened),
                  flush=True)

        # One engram over the whole multi-position co-drive (D.14),
        # over the CA3 ensemble -- matching the validated store.
        bridge.start_engram_recording(tag_name)

        for position, word in enumerate(concept_words):
            word_idx = _word_pattern(word)
            word_global = np.array(
                [lang_indices[i] for i in word_idx], dtype=np.int64)
            word_arr = cp.asarray(word_global, dtype=cp.int64)

            pos_drive = positional_drive_pattern(
                position, n_neurons=n_ctx,
                drive_max_pA=ec_drive_pA,
                sparsity=positional_sparsity,
                n_max_positions=n_max_positions,
            )
            pos_active = np.where(pos_drive > 0)[0]
            pos_global = np.array(
                [ec_context_indices[i] for i in pos_active],
                dtype=np.int64)
            pos_arr = cp.asarray(pos_global, dtype=cp.int64)

            # The word's motor readout pool (the net-new read-back
            # target). Unknown direction word -> no teacher (still
            # encodes the CA3 engram; just no trained read-back).
            action = _WORD_TO_ACTION.get(word)
            pool_arr = None
            if action is not None:
                pool_arr = cp.asarray(
                    list(rm.indices(f"motor_{action}")),
                    dtype=cp.int64)

            ext = cp.zeros(n_total, dtype=cp.float32)
            for _ in range(int(encoding_steps)):
                ext.fill(0)
                ext[word_arr] = float(drive_pA)
                ext[pos_arr] = float(drive_pA)
                if pool_arr is not None:
                    ext[pool_arr] = float(teacher_pA)
                bridge.cp_external_input_current[:] = ext
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose:
                print("  [enc] pos=%d word=%s -> motor_%s (%d steps)"
                      % (position, word, action, encoding_steps),
                      flush=True)

        # Settle, then commit the engram over the CA3 ensemble --
        # region_filter=["ca3"] matches validate_positional_binding.
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        stats = bridge.commit_engram_tag(
            tag_name, top_k=top_k, region_filter=["ca3"],
        )
        if verbose:
            print("  [TAG] %s -> %d CA3 neurons"
                  % (tag_name, stats.get("n_tagged", -1)), flush=True)
    finally:
        # Plasticity NEVER bleeds past the order-intrinsic encode.
        _close_gates(bridge, opened)
        bridge.cp_external_input_current[:] = 0.0

    return tag_name


def readback_sweep(bridge,
                   length: int,
                   n_ec_context: int = 200,
                   n_max_positions: int = 10,
                   positional_sparsity: float = 0.1,
                   ec_drive_pA: float = 200.0,
                   warmup_steps: int = 30,
                   stim_steps: int = 100) -> List[Dict]:
    """Deterministic position sweep (the producer). For k in
    range(length): drive ec_context(positional_drive_pattern(k)) ALONE
    -- NO word / lang_input drive -- and collect the per-direction-word
    motor-pool firing-rate dict. NO learned sequence model: order is
    intrinsic to the D.11 positional code; this is a plain sweep.

    DRY: the ec_context(position)-alone drive + per-region spike-rate
    read mechanism is reused verbatim from
    test_word_order_discrimination.query_position, applied to the
    motor_{N,E,S,W} readout regions (the validated decoder readout for
    this DG/CA3 bridge). This function adds NO feedback and NO new
    pathway: it is WRITE-ONLY into the substrate (external current
    only).

    Returns a list[dict] of length `length`, the i-th entry being
    {direction_word: rate} for position i -- the shape Phase A's
    `order_intrinsic_core.decode_position_sweep` consumes (concept ==
    direction word).
    """
    import numpy as np
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import positional_drive_pattern

    rm = bridge.region_manager
    ec_context_indices = list(rm.indices("ec_context"))
    n_ctx = len(ec_context_indices)
    n_total = bridge.cp_external_input_current.shape[0]

    # word -> its motor readout region index array.
    pool_arrs = {
        word: cp.asarray(list(rm.indices(f"motor_{action}")),
                         dtype=cp.int64)
        for word, action in _WORD_TO_ACTION.items()
    }

    per_pos: List[Dict] = []
    for k in range(int(length)):
        pos_drive = positional_drive_pattern(
            k, n_neurons=n_ctx,
            drive_max_pA=ec_drive_pA,
            sparsity=positional_sparsity,
            n_max_positions=n_max_positions,
        )
        pos_active = np.where(pos_drive > 0)[0]
        pos_global = np.array(
            [ec_context_indices[i] for i in pos_active],
            dtype=np.int64)
        pos_arr = cp.asarray(pos_global, dtype=cp.int64)

        spike_counts = {w: 0 for w in pool_arrs}

        bridge.cp_external_input_current[:] = 0.0
        for _ in range(int(warmup_steps)):
            bridge._run_one_simulation_step()

        ext = cp.zeros(n_total, dtype=cp.float32)
        for _ in range(int(stim_steps)):
            ext.fill(0)
            ext[pos_arr] = float(ec_drive_pA)
            bridge.cp_external_input_current[:] = ext
            bridge._run_one_simulation_step()
            if hasattr(bridge, "cp_firing_states"):
                firing = bridge.cp_firing_states
                for w, arr in pool_arrs.items():
                    spike_counts[w] += int(firing[arr].sum())

        rates = {
            w: spike_counts[w] / (int(stim_steps) * len(pool_arrs[w]))
            for w in pool_arrs
        }
        per_pos.append(rates)

    bridge.cp_external_input_current[:] = 0.0
    return per_pos
