"""Trained ec_context(position)->concept-pool readback encode +
deterministic position-sweep producer (order-intrinsic slice, Task 5).

THE NET-NEW MECHANISM of this line. The 6-negative terminated line
tried to LEARN an ordered sequence model; this does NOT. Order is
INTRINSIC: it lives in the deterministic D.11 positional code
(`sim.text_embeddings.positional_drive_pattern`) and is read back by a
plain position sweep. The single net-new behavior over the weak ~50%
prototype (`research/runners/test_word_order_discrimination.py`) is
that the `ec_context -> concept-pool` pathway is PLASTIC AND
co-firing-strengthened DURING the existing validated co-drive encode
(Tonegawa engram binds all co-active elements; catalog D.14 / D.02),
so a later `ec_context(position)`-alone sweep drives the bound concept
pool strongly instead of leaving it at the raw-trace ~50% floor.

DRY (nothing here is reimplemented):
  - bridge construction          : reuse
    `concept_pool_demo.build_concept_bridge(enable_positional_context
    =True, ...)` -- the SAME wiring the prototype uses.
  - positional code              : reuse
    `sim.text_embeddings.positional_drive_pattern` (D.11; deterministic).
  - the co-drive encode idiom    : reuse the prototype's
    `encode_sentence` drive pattern (lang_input(word) orthogonal code +
    ec_context(position) + target-pool teacher current) AND the
    `sim.bridge` engram API (start_engram_recording / commit_engram_tag).
  - the position-sweep readback  : delegate verbatim to the prototype's
    `query_position` (drive ec_context(position) ALONE; return
    {pool: rate}). readback_sweep is write-only into the substrate
    (external current only) -- no feedback, no new pathway.
  - word<->idx / word<->pool maps: reuse the prototype's `_WORD_TO_IDX`
    / `_WORD_TO_POOL` (do NOT redefine the vocab).

SEPARATION OF CONCERNS (the v12/v13/v15/G1 "first do no harm" lesson;
Task 6 hard-checks this). During the order-intrinsic encode we open
ONLY the `ec_context -> concept-pool` pathway plasticity gates that
`build_concept_bridge` declares on the concept-pool bridge:
`ec_context_to_noun_pool`, `ec_context_to_verb_pool`,
`ec_context_to_adjective_pool`, `ec_context_to_motor` (plus the
`language_input -> *pool` gates so the prototype's teacher-driven
pool firing co-strengthens, exactly as the weak prototype relies on).
Every gate is opened in a try/finally and CLOSED at the end of the
encode -- plasticity never bleeds past the order-intrinsic encode.

The VALIDATED multi-seed D.11/P4.1 `(word,position)->distinct-CA3`
store (`research/runners/validate_positional_binding.py`,
`2026-05-11-P41-positional-multiseed.md`) is the *hippocampal
trisynaptic* store: it is built by a DIFFERENT bridge
(`build_biological_brain_regions(enable_hippocampus_consolidation
=True, enable_episodic_context=True)`) whose gates
(`ec_context_to_dg`, `ec_to_dg`, `dg_to_ca3`, `ca3_swr_burst`,
`lang_to_ec`) DO NOT EXIST on this concept-pool bridge. This encode
touches NONE of them: those pathway/gate sets are disjoint, the no-harm
check (Task 6) re-runs that separate hippo store unchanged, and the
defensive try/except guards mean any non-existent gate name is a no-op.
This is precisely the intended target per the plan ("If
build_concept_bridge's positional wiring is ec_context->pool ... NOT
the DG/CA3 path, that is the intended target").

Conventions: ASCII-only print() (Windows cp1252). Heavy imports
(sim.*, the prototype, the concept-pool builder) are LAZY inside the
functions so the import/signature smoke is instant.
"""
from __future__ import annotations

from typing import List

# Concept-pool plasticity gates declared by build_concept_bridge when
# enable_positional_context=True (see concept_pool_demo.py lines
# ~196-247). These are the ONLY net-new gates this encode opens. They
# are wholly disjoint from the validated D.11/P4.1 DG/CA3 hippocampal
# store gates, which do not exist on the concept-pool bridge at all.
_EC_CONTEXT_TO_POOL_GATES = (
    "ec_context_to_noun_pool",
    "ec_context_to_verb_pool",
    "ec_context_to_adjective_pool",
    "ec_context_to_motor",
)
# lang_input -> {kind}_pool gates: opened alongside so the prototype's
# teacher-driven pool firing co-strengthens lang_input(word)->pool too
# (the weak 50% prototype implicitly relies on this via teacher_pA).
# Tagged by build_biological_brain_regions as
# "language_input_to_{kind}_pool".
_LANG_INPUT_TO_POOL_GATES = (
    "language_input_to_noun_pool",
    "language_input_to_verb_pool",
    "language_input_to_adjective_pool",
    "language_input_to_motor",
)


def build_order_intrinsic_bridge(seed: int,
                                  n_lang_input: int = 2048,
                                  n_per_pool: int = 200,
                                  n_fs_per_pool: int = 24,
                                  n_ec_context: int = 200,
                                  ec_context_to_pool_density: float = 0.30,
                                  ec_context_to_pool_weight: float = 3.0,
                                  enable_adjective: bool = True,
                                  verbose: bool = False):
    """Build the concept-pool bridge with the validated positional
    store + the (now to-be-trained) plastic ec_context->concept-pool
    pathway present.

    DRY: this is exactly `concept_pool_demo.build_concept_bridge(
    enable_positional_context=True, ...)` -- the SAME wiring the weak
    ~50% prototype (`test_word_order_discrimination.py`) uses. Nothing
    about the wiring is net-new; the net-new piece is `encode_proposition`
    opening the ec_context->pool plasticity gate during the co-drive
    encode (the prototype left that pathway untrained -> ~50%).

    canon dynamics (weak_dynamics=False) match the prototype: an
    ec_context-alone sweep must bootstrap pool firing through the
    STDP-grown weights, and weak pools do not self-sustain that.

    Returns the constructed SimulationBridge.
    """
    import research.runners.concept_pool_demo as cpd

    return cpd.build_concept_bridge(
        seed=seed,
        n_lang_input=n_lang_input,
        n_per_pool=n_per_pool,
        n_fs_per_pool=n_fs_per_pool,
        enable_adjective=enable_adjective,
        weak_dynamics=False,  # canon: match the prototype (stronger pool firing)
        enable_direct_verb_to_motor=True,
        enable_positional_context=True,
        n_ec_context=n_ec_context,
        ec_context_to_pool_density=ec_context_to_pool_density,
        ec_context_to_pool_weight=ec_context_to_pool_weight,
        verbose=verbose,
    )


def _open_gates(bridge, names) -> List[str]:
    """Open (set to 1.0) every plasticity gate in `names` that EXISTS
    on this bridge. A name absent from the active wiring plan raises
    KeyError in set_plasticity_gate -> swallowed (no-op). This is the
    defensive guard that guarantees a gate from a DIFFERENT architecture
    (e.g. the D.11/P4.1 DG/CA3 store's `ec_context_to_dg`) can never be
    touched here even by mistake. Returns the gates actually opened."""
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
                        all_pool_names: List[str],
                        tag_name: str = None,
                        n_lang_input: int = 2048,
                        n_words_for_orthogonal: int = 16,
                        sparsity: float = 0.05,
                        n_ec_context: int = 200,
                        n_max_positions: int = 8,
                        positional_sparsity: float = 0.1,
                        encoding_steps: int = 400,
                        drive_pA: float = 200.0,
                        ec_drive_pA: float = 500.0,
                        teacher_pA: float = 500.0,
                        top_k: int = 100,
                        verbose: bool = False) -> str:
    """Encode ONE proposition as a single Tonegawa engram over the
    co-active (concept @ position) set, WITH the ec_context->concept-
    pool pathway plastic + co-firing-strengthened.

    For k, word in enumerate(concept_words): drive lang_input(word)
    (orthogonal code) AND ec_context(positional_drive_pattern(k)) AND
    teacher current on the word's target concept pool, SIMULTANEOUSLY,
    for `encoding_steps` steps, with the ec_context->pool (and
    lang_input->pool) plasticity gates OPEN. STDP/Hebbian co-firing
    strengthens position-k -> the concept's pool. The entire multi-
    position drive is wrapped in one start_engram_recording /
    commit_engram_tag so the proposition is ONE engram over the
    co-active concept-pool ensemble (D.14).

    DRY: the per-(word,position) drive block is the prototype's
    `encode_sentence` idiom (orthogonal_drive_pattern + positional_
    drive_pattern + per-pool teacher current); the recording/tag is the
    `sim.bridge` engram API. The ONLY net-new behavior vs the weak ~50%
    prototype is the gates being OPEN during this co-drive (the
    prototype trained nothing on ec_context->pool).

    SEPARATION OF CONCERNS: gates are opened in a try/finally and
    CLOSED before return -- plasticity never bleeds past this encode.
    Only `ec_context_to_{noun,verb,adjective,motor}` (+ the analogous
    `language_input_to_*pool`) gates -- which exist ONLY on this
    concept-pool bridge -- are touched. The validated D.11/P4.1 DG/CA3
    store's gates do not exist here and are never referenced.

    Returns the committed engram tag name.
    """
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import (orthogonal_drive_pattern,
                                     positional_drive_pattern)
    # Reuse the prototype's vocab maps -- do NOT redefine them.
    from research.runners.test_word_order_discrimination import (
        _WORD_TO_IDX, _WORD_TO_POOL,
    )

    if tag_name is None:
        tag_name = "prop_" + "_".join(concept_words)

    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    ec_arr = cp.asarray(list(rm.indices("ec_context")), dtype=cp.int64)
    n_total = bridge.cp_external_input_current.shape[0]

    # region_filter for the engram: the concept pools (the
    # compositional ensemble), matching compose_engram_demo's idiom.
    region_filter = list(all_pool_names)

    # Settle transients before recording (prototype encode_sentence
    # does a 30-step zero-input warmup).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()

    opened = []
    try:
        # Open ONLY the concept-pool-bridge plasticity gates (net-new:
        # ec_context->pool; plus lang_input->pool so teacher-driven
        # pool firing co-strengthens, as the weak prototype implicitly
        # relied on). Non-existent gates are no-ops (defensive guard).
        opened = _open_gates(
            bridge, list(_EC_CONTEXT_TO_POOL_GATES)
            + list(_LANG_INPUT_TO_POOL_GATES))
        if verbose:
            print("[ENCODE] '%s' (tag=%s) gates_open=%s"
                  % (" ".join(concept_words), tag_name, opened),
                  flush=True)

        # One engram over the whole multi-position co-drive (D.14).
        bridge.start_engram_recording(tag_name)

        for position, word in enumerate(concept_words):
            word_drive = orthogonal_drive_pattern(
                cue_idx=_WORD_TO_IDX[word], n_cues=n_words_for_orthogonal,
                n_neurons=n_lang_input, drive_max_pA=drive_pA,
                sparsity=sparsity,
            )
            pos_drive = positional_drive_pattern(
                position=position, n_neurons=n_ec_context,
                drive_max_pA=ec_drive_pA, sparsity=positional_sparsity,
                n_max_positions=n_max_positions,
            )
            pool_name = _WORD_TO_POOL[word]
            pool_arr = cp.asarray(list(rm.indices(pool_name)),
                                  dtype=cp.int64)

            word_drive_gpu = cp.asarray(word_drive, dtype=cp.float32)
            pos_drive_gpu = cp.asarray(pos_drive, dtype=cp.float32)

            ext = cp.zeros(n_total, dtype=cp.float32)
            for _ in range(encoding_steps):
                ext.fill(0)
                ext[lang_arr] = word_drive_gpu
                ext[ec_arr] = pos_drive_gpu
                ext[pool_arr] = teacher_pA
                bridge.cp_external_input_current[:] = ext
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1

            if verbose:
                print("  [enc] pos=%d word=%s -> %s (%d steps)"
                      % (position, word, pool_name, encoding_steps),
                      flush=True)

        # Settle, then commit the engram over the concept-pool ensemble.
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        stats = bridge.commit_engram_tag(
            tag_name, top_k=top_k, region_filter=region_filter,
        )
        if verbose:
            print("  [TAG] %s -> %d neurons"
                  % (tag_name, stats.get("n_tagged", -1)), flush=True)
    finally:
        # Plasticity NEVER bleeds past the order-intrinsic encode.
        _close_gates(bridge, opened)
        bridge.cp_external_input_current[:] = 0.0

    return tag_name


def readback_sweep(bridge,
                   length: int,
                   all_pool_names: List[str],
                   n_ec_context: int = 200,
                   n_max_positions: int = 8,
                   positional_sparsity: float = 0.1,
                   ec_drive_pA: float = 500.0,
                   stim_steps: int = 100) -> List[dict]:
    """Deterministic position sweep (the producer). For k in
    range(length): drive ec_context(positional_drive_pattern(k)) ALONE
    -- NO word / lang_input drive -- and collect the per-pool firing
    rate dict. NO learned sequence model: order is intrinsic to the
    D.11 positional code; this is a plain sweep.

    DRY: each position is read back by delegating verbatim to the
    prototype's `query_position` (`test_word_order_discrimination.py`)
    -- the EXACT ec_context(position)-alone drive mechanism. This
    function adds NO feedback and NO new pathway: it is WRITE-ONLY into
    the substrate (external current only), exactly as query_position is.

    Returns a list[dict] of length `length`, the i-th entry being
    {pool: rate} for position i -- the shape Phase A's
    `order_intrinsic_core.decode_position_sweep` consumes.
    """
    from research.runners.test_word_order_discrimination import (
        query_position,
    )

    per_pos: List[dict] = []
    for k in range(int(length)):
        rates = query_position(
            bridge, k, list(all_pool_names),
            n_ec_context=n_ec_context,
            n_max_positions=n_max_positions,
            positional_sparsity=positional_sparsity,
            ec_drive_pA=ec_drive_pA,
            stim_steps=stim_steps,
        )
        per_pos.append(rates)
    return per_pos
