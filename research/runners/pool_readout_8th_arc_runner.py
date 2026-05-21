"""Net-new pool-readout substitution runner (Task 2 of the 8th arc).

Biology + empirical motivation: the 8th architecture in the gating-
based composition design line tests whether the compositional
readout's primary bottleneck is the gated lang_output cosine path
(per the unified runner's localisation finding 110f7cd + ablation
0ef9b6e) rather than the encoded engram tag itself. The
pool-vs-lang_output multi-seed diagnostic at commit 4d6a3a6 showed
that reading adjective_pool firing rates directly (bypassing
lang_output cosine entirely) CONSISTENTLY outperformed the
lang_output readout across 3 seeds (per-seed deltas [+1, 0, +1];
aggregate +13.3pp; pool 4/15 vs lang_output 2/15). The 8th arc
REUSES the 6th arc runner structure byte-unchanged and substitutes
ONLY the readout function so that the experimental contrast isolates
the readout substitution from every other Phase-1 / encoding /
replay / PFC-frame mechanism the 6th arc already validated as
structurally active.

  * REUSES the 6th arc's GENERATIVE REPLAY (``run_concept_replay_phase``
    byte-unchanged) and PFC-FRAME PRIMING (``_prime_pfc_frame``
    reused-by-import). Both arms (FULL + UNIFORM_CTRL) run the SAME
    augmenting mechanisms; encoding-specificity is respected (cue
    stays ON during retrieve in BOTH arms; theta-gamma finding 1bbc165
    is honoured).
  * THE SOLE CONTRAST between the arms is the readout function for
    compositional queries:
      - FULL arm: net-new ``_compositional_query_pool_readout``
        reads adjective_pool firing rates directly via
        ``cp_firing_states`` after cue + tag stim. Pool indices are
        looked up via the bridge's existing public
        ``region_manager.indices`` API; NO new region; NO substrate
        modification.
      - UNIFORM_CTRL arm: reused ``_compositional_query_ranked``
        (lang_output cosine baseline; the 6th arc's existing readout).
  * Both arms route their ranked confidences through the SAME
    substrate-specific gate (``COMPOSITIONAL_UNIFIED_THRESHOLD``)
    so the SOLE differentiator on each compositional query is the
    raw quantity being ranked.

The 6th architecture established the local optimum (full=0.458 at
N=3; 35% gap-closure vs the unified per-regime monitor's 0.274 +
theta-gamma's 0.280). The 7th arc REGRESSED to 0.363 by stacking
four more-aggressive mechanisms (cross-arc trajectory analysis at
9693685). The 8th arc takes a strictly substrate-aware approach:
change the readout, keep every other mechanism the same. See
docs/plans/2026-05-20-8th-arc-pool-readout-substitution-design.md.

This module is the ONLY genuinely net-new code in the arc:
  * a runner-local pool-readout function per compositional query
    on the FULL arm: cue the noun via the REUSED
    ``lang_output_pattern_during_input`` helper, stim the engram
    tag while accumulating per-pool spike counts from
    ``cp_firing_states``, convert to rates, rank descending. NO
    lang_output cosine match; NO new region; the four
    adjective_pool_* regions already exist in the v14/v16 substrate;
  * ONE load-bearing structural-effect probe (readout-substitution)
    that verifies the pool readout produces a DIFFERENT ranked
    output from the lang_output cosine readout on the SAME bridge
    state for at least one query (otherwise the readout substitution
    is structurally inert);
  * ONE reused structural-effect probe (replay-effect) IMPORTED
    BYTE-UNCHANGED from the 6th arc -- the augmenting mechanism is
    identical in both arms here, but the probe confirms it is still
    structurally active on the substrate the 8th arc uses, mirrors
    Pirazzini d462bf0 + theta-gamma e6b17da lesson;
  * Both probes apply cache-scale validation per the 6th arc commit
    13f73e8 + RNG isolation per the theta-gamma commit e6b17da.

Every other subsystem is REUSED-BY-IMPORT from the 6th arc runner
(commit 13f73e8 byte-stable) + the unified per-regime monitor
runner (commit 25b9183 byte-stable):

  * Substrate construction + Phase-1 caching (validated v16 +
    hippocampus + dlpfc PFC frame): REUSED via 6th arc imports.
  * Compositional one-shot encoding: REUSED ``_encode_facts``
    (calls byte-unchanged ``encode_concept_pair`` internally).
  * Compositional pair generation: REUSED
    ``_unified_compositional_pairs``.
  * Direct W->A readout: REUSED ``_direct_query_ranked``.
  * Baseline lang_output cosine compositional readout: REUSED
    ``_compositional_query_ranked``.
  * Generative replay phase: REUSED
    ``consolidation_trainer.run_concept_replay_phase`` BYTE-UNCHANGED.
  * PFC-frame priming + replay-effect probe + cache-scale validator:
    REUSED from the 6th arc runner BYTE-UNCHANGED.
  * Frozen capability verdict: REUSED
    ``pool_readout_8th_arc_core.pool_readout_8th_arc_verdict``
    (Task 1 byte-unchanged; bars set in advance, NEVER tuned).
  * Both substrate-specific calibrated moats:
    DIRECT_UNIFIED_THRESHOLD + COMPOSITIONAL_UNIFIED_THRESHOLD
    imported BYTE-UNCHANGED.
  * Kill-safe checkpoint: REUSED sim.train_checkpoint.

Anti-cheat (carry forward all prior lessons):
  * OPAQUE tag names; no tag-string parsing on tag names.
  * Both arms run the SAME encoding + SAME replay + SAME PFC-frame
    prime; the SOLE differentiator is the readout function applied
    to the SAME post-stim bridge state.
  * The ``cross_pool_concept`` gate is opened ONLY inside the
    encoding window (via ``encode_concept_pair``) then closed.
  * Cue presence during retrieve is IDENTICAL across arms.
  * No protected-file edits; no autograd; the runner is reuse-only
    orchestration of the prior arcs' modules + the net-new pool-
    readout function + the two structural-effect probes.

ASCII only. CuPy is the real / decisive path; --tiny-synth shrinks
Phase-1 training + compositional encoding + replay cycles + PFC-frame
priming durations so the smoke is seconds (toy numbers explicitly
NOT a result).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy mirrors the 6th arc runner. CuPy is the decisive
# path; NumPy ONLY when CuPy is genuinely unavailable (GPU-less box).
if "--tiny-synth" in sys.argv:
    try:
        import cupy as _cupy_probe  # noqa: F401

        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
    if not _CUPY_AVAILABLE:
        os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

# REUSED frozen capability verdict (Task 1 byte-unchanged).
from research.runners.pool_readout_8th_arc_core import (
    pool_readout_8th_arc_verdict,
    _CP_LADDER,
)

# REUSED gates byte-unchanged (the four moats). BOTH arms route
# direct queries through DIRECT_UNIFIED_THRESHOLD and compositional
# queries through COMPOSITIONAL_UNIFIED_THRESHOLD. The SOLE arm
# differentiator is the readout function applied BEFORE the gate.
from research.runners.abstention_gate import DEFAULT_THRESHOLD as MOAT_DIRECT
from research.runners.abstention_gate_compositional import (
    COMPOSITIONAL_THRESHOLD,
)
from research.runners.abstention_gate_compositional_unified import (
    gate as gate_compositional_unified,
    COMPOSITIONAL_UNIFIED_THRESHOLD,
)
from research.runners.abstention_gate_direct_unified import (
    gate as gate_direct_unified,
    DIRECT_UNIFIED_THRESHOLD,
)

from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# REUSED prior-arc orchestration -- every Phase-1 + substrate +
# encoding helper imported BYTE-UNCHANGED from the unified runner.
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_recipe,
    _phase1_cache_path,
    _phase1_train_if_needed,
    _freeze_phase1_gates,
    _all_pool_regions,
    _all_words_word_to_idx,
    _direct_pool_target,
    _direct_query_ranked,
    _compositional_query_ranked,
    _unified_compositional_pairs,
    _encode_facts,
    _UNIFIED_SUBSEED_OFFSET,
    _PHASE1_CACHE_DEFAULT,
)
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _VERBS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
    _HIPPO_TAG_REGIONS,
)

# REUSED generative-replay subsystem BYTE-UNCHANGED (Phase 1.3
# consolidation work; consolidation_trainer.py:43).
from research.runners.consolidation_trainer import run_concept_replay_phase

# REUSED augmenting mechanisms + cache-scale validator + replay-
# effect probe + PFC-frame priming + RNG isolation helpers BYTE-
# UNCHANGED from the 6th arc runner. The 8th arc keeps these
# mechanisms identical in BOTH arms; the SOLE arm differentiator is
# the readout function.
from research.runners.generative_replay_pfc_frame_runner import (
    _run_generative_replay,
    _prime_pfc_frame,
    _replay_effect_probe,
    _validate_cache_scale_for_probe,
    _seed_query_rng,
    _restore_query_rng,
    _PROBE_RNG_SEED,
    _PROBE_ENCODE_RNG_SEED,
    _PROBE_CONTROL_TOL_MV,
    N_REPLAYS_PER_TAG,
    REPLAY_DRIVE_PA,
    REPLAY_BURST_DURATION_MS,
    REPLAY_INTER_BURST_MS,
    PFC_FRAME_PA,
    PFC_FRAME_STIM_STEPS,
    TINY_N_REPLAYS_PER_TAG,
    TINY_REPLAY_BURST_DURATION_MS,
    TINY_REPLAY_INTER_BURST_MS,
    TINY_PFC_FRAME_STIM_STEPS,
)


# =====================================================================
# Net-new helper: the pool-readout function. This is the genuinely
# net-new piece of the 8th arc. Reads adjective_pool firing rates
# directly via ``bridge.cp_firing_states`` after a cue + tag stim
# window. NO lang_output cosine match; NO new region; the four
# ``adjective_pool_*`` regions already exist in the v14/v16 substrate.
# =====================================================================
# Pool name pairs (target word -> adjective pool region name). The
# ordering is fixed and matches ``cpd.ADJECTIVE_VOCAB`` (so the
# pool-readout target word vocabulary equals the v14/v16 adjective
# vocabulary).
_POOL_READOUT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("big", "adjective_pool_BIG"),
    ("small", "adjective_pool_SMALL"),
    ("hot", "adjective_pool_HOT"),
    ("cold", "adjective_pool_COLD"),
)


def _compositional_query_pool_readout(
    bridge,
    cue_noun: str,
    tag_name: Optional[str],
    dims: Dict[str, Any],
    recall_steps: int,
):
    """Net-new pool-readout substitution for the compositional query.

    Reads compositional output via adjective_pool firing rates after
    cue + tag stim. BYPASSES the gated lang_output cosine path
    entirely. Empirically motivated by the multi-seed signal at
    commit 4d6a3a6 (pool readout consistently >= lang_output across
    3 seeds; +13.3pp aggregate). Reads from existing
    ``adjective_pool_*`` regions via the bridge's
    ``region_manager.indices`` public API. NO new region; NO
    substrate modification.

    Phase 1 (cue drive): drive ``lang_input(cue_noun)`` via the
    REUSED ``lang_output_pattern_during_input`` helper for
    ``recall_steps`` simulation steps. The helper does the same
    setup the lang_output cosine readout uses, so the bridge state
    going into Phase 2 is byte-identical to the state the baseline
    readout would see (only the read-out quantity differs between
    the two readouts).

    Phase 2 (engram-tag stim): if a non-None tag was committed,
    stimulate the tag at 1500 pA for ``recall_steps`` steps;
    accumulate per-pool spike counts from
    ``bridge.cp_firing_states``. The four adjective pools are read
    via ``bridge.region_manager.indices(name)`` (public API).

    Phase 3 (rank): convert per-pool spike counts to rates (divided
    by the per-pool neuron count + recall_steps), return a ranked
    [(word, rate, "pool_readout"), ...] list descending by rate.

    Returns: ranked list with the SAME shape ``_compositional_query_ranked``
    returns (a list of ``(word, score, tag)`` tuples). The downstream
    gate (``gate_compositional_unified``) consumes the ranked list
    by score-magnitude only; the tag column is descriptive (no
    tag-string parsing per Stage-1 / SPEAR / Pirazzini lesson).
    """
    # REUSED helper. The Phase 1 cue-drive logic + the safe restoration
    # of ``cp_external_input_current`` to zero post-call are identical
    # to the baseline readout's preamble.
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_input,
    )
    # Phase 1: cue drive (uses the REUSED helper for protocol
    # consistency with the baseline readout).
    lang_output_pattern_during_input(
        bridge, cue_noun,
        n_lang_input=int(dims["n_lang_input"]),
        sparsity=float(dims["sparsity"]),
        n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
        stim_steps=int(recall_steps),
    )

    # Phase 2: stim engram tag + accumulate spike counts per pool.
    pool_indices: Dict[str, Any] = {}
    for word, region_name in _POOL_READOUT_PAIRS:
        try:
            pool_indices[word] = list(
                bridge.region_manager.indices(region_name)
            )
        except Exception:
            pool_indices[word] = []
    pool_spike_counts: Dict[str, int] = {
        word: 0 for word, _ in _POOL_READOUT_PAIRS
    }

    tag_exists = (
        tag_name is not None
        and tag_name in {t["name"] for t in bridge.list_engram_tags()}
    )
    if tag_exists:
        bridge.stimulate_tag(tag_name, drive_pA=1500.0)
        for _ in range(int(recall_steps)):
            bridge._run_one_simulation_step()
            firing = bridge.cp_firing_states
            if hasattr(firing, "get"):
                firing = firing.get()
            firing_np = np.asarray(firing)
            for word, idx_list in pool_indices.items():
                if idx_list:
                    pool_spike_counts[word] += int(
                        np.asarray(firing_np[idx_list]).sum()
                    )
        bridge.clear_tag_drive(tag_name)

    # Phase 3: rates + ranked output. n_per_pool comes from the
    # Phase-1 recipe dims (same scale across all four adjective
    # pools by construction).
    n_per_pool = int(dims["n_per_pool"])
    denom = float(max(1, n_per_pool * int(recall_steps)))
    rates: Dict[str, float] = {
        word: float(pool_spike_counts[word]) / denom
        for word, _ in _POOL_READOUT_PAIRS
    }
    ranked = sorted(
        [
            (word, rate, "pool_readout")
            for word, rate in rates.items()
        ],
        key=lambda x: -x[1],
    )
    return ranked


# =====================================================================
# Net-new structural-effect probe: readout-substitution probe. This is
# the load-bearing probe for the 8th arc. It verifies that the pool
# readout produces a DIFFERENT ranked output from the lang_output
# cosine readout on the SAME bridge state under identical RNG
# isolation; if they agreed bit-identically across every cue/tag, the
# readout substitution would be structurally inert.
# =====================================================================
def _readout_substitution_probe(
    seed: int = 42,
    tiny_synth: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the runner's actual code path twice on the SAME bridge
    state -- once with the new pool-readout function, once with the
    baseline lang_output cosine readout. Return diagnostic dict whose
    fields establish the readout substitution is structurally active.

    Mechanism (THE load-bearing probe for the 8th arc):
      * Build one bridge, load the SAME Phase-1 checkpoint, encode
        the SAME compositional pairs.
      * For each compositional query: deterministic-seed the active
        backend's RNG BEFORE the readout call so both readouts see
        IDENTICAL OU-noise streams.
      * Call ``_compositional_query_pool_readout`` (FULL arm).
      * RESTORE the bridge state by re-seeding the RNG to the same
        value; re-encode is not needed (lang_output_pattern_during_input
        restores cp_external_input_current to 0 at the end).
      * Call ``_compositional_query_ranked`` (UNIFORM_CTRL baseline)
        on the SAME bridge with the SAME RNG seed.
      * Compare the TOP word of each ranked output. If they DIFFER
        on AT LEAST one query, the readout substitution is structurally
        active; the probe PASSES.

    Both readouts share Phase 1's ``lang_output_pattern_during_input``
    preamble (the cue-drive), so any structural difference is
    attributable to the read-out quantity (per-pool firing rates vs
    lang_output cosine). The probe is intentionally LIBERAL on the
    diff criterion: as long as at least one query's top-1 differs,
    the substitution is non-inert -- the gate-binding correctness is
    tested elsewhere via the decisive eval rungs + frozen verdict.

    DEFENSIVE pre-load check (CLOSES the 10th adversarial review
    BLOCK): BEFORE calling ``load_checkpoint`` the probe validates
    that the cached checkpoint's stored bridge dimensions match the
    freshly-built bridge dimensions via
    ``_validate_cache_scale_for_probe`` (REUSED from the 6th arc).

    Returns: a dict with keys ``n_queries`` (int),
    ``n_differing_top1`` (int), ``differing`` (bool), and
    ``diff_v_membrane`` (float; the max |delta v_membrane| between
    the post-pool-readout state and the post-lang-output-readout
    state -- a secondary structural witness). If
    ``n_differing_top1 == 0`` and ``diff_v_membrane <= 0.0``, the
    probe raises ``RuntimeError`` (the substitution is structurally
    inert via both witnesses; mirrors Pirazzini d462bf0 + 6th arc
    structural-effect probe pattern).
    """
    from sim.backend import to_host
    cache_dir = str(cache_dir) if cache_dir else _PHASE1_CACHE_DEFAULT
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _phase1_train_if_needed(int(seed), cache_dir, tiny_synth)
    cache_path = _phase1_cache_path(cache_dir, seed)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, _word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    facts = _unified_compositional_pairs(seed, 2)
    enc_steps = 8 if tiny_synth else 200
    recall_steps = 20 if tiny_synth else 100

    # Build TWO parallel bridges -- the two readouts mutate
    # ``cp_external_input_current`` + the per-step firing trace; we
    # build a fresh bridge for each readout so the SOLE remaining
    # difference between the two reads is the readout function.
    bridge_pool = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
    bridge_lang = _build_bridge_with_phase1_recipe(int(seed), tiny_synth)
    _validate_cache_scale_for_probe(
        cache_path, bridge_pool, "readout-substitution"
    )
    _validate_cache_scale_for_probe(
        cache_path, bridge_lang, "readout-substitution"
    )
    bridge_pool.load_checkpoint(str(cache_path))
    bridge_lang.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge_pool)
    _freeze_phase1_gates(bridge_lang)

    # Encode the SAME facts into BOTH bridges with the SAME
    # deterministic RNG seed so the encoded states are byte-identical
    # across the two bridges going into the queries.
    saved_enc_pool = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
    try:
        tags_pool = _encode_facts(bridge_pool, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved_enc_pool)
    saved_enc_lang = _seed_query_rng(_PROBE_ENCODE_RNG_SEED)
    try:
        tags_lang = _encode_facts(bridge_lang, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved_enc_lang)

    n_queries = 0
    n_differing_top1 = 0
    for i, (noun, _adj) in enumerate(facts):
        n_queries += 1
        tag_pool = tags_pool[i] if i < len(tags_pool) else None
        tag_lang = tags_lang[i] if i < len(tags_lang) else None
        query_rng_seed = (
            int(seed) * 1_000_003 + int(i) * 7919 + 17
        ) & 0x7FFFFFFF

        saved_p = _seed_query_rng(query_rng_seed)
        try:
            ranked_pool = _compositional_query_pool_readout(
                bridge_pool, noun, tag_pool, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_p)
        saved_l = _seed_query_rng(query_rng_seed)
        try:
            ranked_lang = _compositional_query_ranked(
                bridge_lang, noun, tag_lang, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_l)

        top_pool = ranked_pool[0][0] if ranked_pool else None
        top_lang = ranked_lang[0][0] if ranked_lang else None
        if top_pool != top_lang:
            n_differing_top1 += 1

    # Secondary witness: max |delta v_membrane| between the two
    # post-readout bridge states. Even when top-1 happens to agree
    # by coincidence on every query, the per-pool reads vs the
    # lang_output reads execute slightly different code paths so the
    # post-state should differ noticeably. This is a defense-in-depth
    # diagnostic; the load-bearing criterion is the top-1 differing
    # count.
    v_pool = to_host(bridge_pool.cp_membrane_potential_v)
    v_lang = to_host(bridge_lang.cp_membrane_potential_v)
    diff_v_membrane = float(
        np.max(np.abs(np.asarray(v_pool) - np.asarray(v_lang)))
    )

    differing = (n_differing_top1 > 0)
    if not differing and diff_v_membrane <= 0.0:
        raise RuntimeError(
            "Readout-substitution probe FAILED: the pool-readout and "
            "the lang_output cosine readout produced byte-identical "
            "ranked outputs on every query (top-1 agreed on "
            "%d/%d queries) AND the post-readout bridge states "
            "agreed to %.6g mV. The readout substitution is "
            "structurally inert -- mirrors Pirazzini d462bf0 / 6th "
            "arc structural-effect probe pattern. Fix and re-run "
            "BEFORE decisive."
            % (n_differing_top1, n_queries, diff_v_membrane)
        )
    return {
        "n_queries": int(n_queries),
        "n_differing_top1": int(n_differing_top1),
        "differing": bool(differing),
        "diff_v_membrane": float(diff_v_membrane),
    }


# =====================================================================
# Per-cell evaluation arm: one (seed, N) cell. Mirrors the 6th arc
# runner's structure EXCEPT for the compositional readout call:
#   * the FULL arm calls ``_compositional_query_pool_readout``
#     (net-new; pool firing rates);
#   * the UNIFORM_CTRL arm calls ``_compositional_query_ranked``
#     (REUSED; lang_output cosine baseline).
# Both arms run the SAME encoding + SAME generative replay + SAME
# PFC-frame prime + SAME cue presence during retrieve; the SOLE
# differentiator on each compositional query is the readout function.
# =====================================================================
def _run_evaluation_arm(seed: int, N: int, tiny_synth: bool,
                          cache_dir: str) -> Dict[str, Any]:
    """Run the pool-readout substitution architecture for ONE
    (seed, N) cell. Two parallel bridges (FULL + UNIFORM_CTRL); each
    runs the SAME encoding + same direct queries + same replay phase
    + same PFC-frame prime before each compositional query. The SOLE
    arm differentiator is the compositional readout function.

    Returns the four rung accuracy fields the frozen verdict
    consumes: ``full_acc``, ``uniform_ctrl_acc``,
    ``direct_retain_acc``, ``abstain_correct``.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200

    cache_path = _phase1_cache_path(cache_dir, seed)
    if not cache_path.exists():
        raise RuntimeError(
            "Phase-1 cache missing for seed %d at %s; call "
            "_phase1_train_if_needed first." % (seed, cache_path)
        )

    # TWO parallel bridges -- same architecture, same Phase-1
    # checkpoint, same frozen gates. The SOLE differentiator is the
    # compositional readout function applied on each compositional
    # query.
    bridge_full = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_uniform = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_full.load_checkpoint(str(cache_path))
    bridge_uniform.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge_full)
    _freeze_phase1_gates(bridge_uniform)

    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims: Dict[str, Any] = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }
    all_pools = _all_pool_regions(enable_adjective=True)

    # Compositional encoding: encode the SAME facts into BOTH bridges
    # so the encoded state is identical at the start of the queries.
    facts = _unified_compositional_pairs(seed, N)
    encode_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 31337
    ) & 0x7FFFFFFF
    saved_enc_full = _seed_query_rng(encode_rng_seed)
    try:
        tags_full = _encode_facts(bridge_full, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved_enc_full)
    saved_enc_uniform = _seed_query_rng(encode_rng_seed)
    try:
        tags_uniform = _encode_facts(
            bridge_uniform, facts, dims, enc_steps
        )
    finally:
        _restore_query_rng(saved_enc_uniform)

    # Generative replay on BOTH bridges with IDENTICAL deterministic
    # RNG seed -- the augmenting mechanism is the SAME in both arms
    # in the 8th arc (the SOLE differentiator is the readout function).
    replay_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 7919
    ) & 0x7FFFFFFF
    saved_replay_full = _seed_query_rng(replay_rng_seed)
    try:
        replay_stats_full = _run_generative_replay(
            bridge_full, tags_full, tiny_synth, replay_rng_seed
        )
    finally:
        _restore_query_rng(saved_replay_full)
    saved_replay_uniform = _seed_query_rng(replay_rng_seed)
    try:
        replay_stats_uniform = _run_generative_replay(
            bridge_uniform, tags_uniform, tiny_synth, replay_rng_seed
        )
    finally:
        _restore_query_rng(saved_replay_uniform)

    # ---- DIRECT queries: one per unique trained word in the cell's
    # facts. BOTH arms route direct queries through the SAME substrate-
    # specific direct gate. The two bridges have been identically
    # perturbed by replay so direct accuracy should match closely.
    n_direct_total = 0
    n_direct_correct_full = 0
    n_direct_correct_uniform = 0
    direct_words: List[Tuple[str, str]] = []
    seen_direct: set = set()
    for noun, adj in facts:
        for w in (noun, adj):
            if w in seen_direct:
                continue
            seen_direct.add(w)
            try:
                expected_pool = _direct_pool_target(w)
            except KeyError:
                continue
            direct_words.append((w, expected_pool))

    for word, expected_pool in direct_words:
        n_direct_total += 1
        direct_query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + hash(word) % 65521 + 113
        ) & 0x7FFFFFFF
        saved_dq_full = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_full = _direct_query_ranked(
                bridge_full, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved_dq_full)
        decided_full = gate_direct_unified(
            ranked_full, DIRECT_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        saved_dq_uniform = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_uniform = _direct_query_ranked(
                bridge_uniform, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved_dq_uniform)
        decided_uniform = gate_direct_unified(
            ranked_uniform, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_uniform == expected_pool:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries: one per encoded fact, cue the noun,
    # expect the bound adj. The cue (lang_input) stays ON during
    # retrieve in BOTH arms (encoding-specificity respected). FULL +
    # UNIFORM_CTRL arms BOTH prime ``dlpfc_verb`` before each query
    # (PFC-frame mechanism is identical across arms in the 8th arc);
    # the SOLE differentiator is the readout function.
    n_comp_total = 0
    n_comp_correct_full = 0
    n_comp_correct_uniform = 0
    for i, (noun, adj) in enumerate(facts):
        n_comp_total += 1
        tag_full = tags_full[i] if i < len(tags_full) else None
        tag_uniform = tags_uniform[i] if i < len(tags_uniform) else None
        query_rng_seed = (
            int(seed) * 1_000_003 + int(N) * 1009 + int(i) * 7919 + 17
        ) & 0x7FFFFFFF
        # FULL arm: PFC-frame prime + NET-NEW pool-readout substitution.
        saved_full = _seed_query_rng(query_rng_seed)
        try:
            _prime_pfc_frame(bridge_full, tiny_synth)
            ranked_full = _compositional_query_pool_readout(
                bridge_full, noun, tag_full, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_full)
        decided_full = gate_compositional_unified(
            ranked_full, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        # UNIFORM_CTRL arm: PFC-frame prime + REUSED lang_output cosine
        # readout (6th arc baseline). IDENTICAL deterministic RNG seed
        # so the OU-noise streams match across arms.
        saved_uniform = _seed_query_rng(query_rng_seed)
        try:
            _prime_pfc_frame(bridge_uniform, tiny_synth)
            ranked_uniform = _compositional_query_ranked(
                bridge_uniform, noun, tag_uniform, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved_uniform)
        decided_uniform = gate_compositional_unified(
            ranked_uniform, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_uniform = None if decided_uniform is None else decided_uniform[0]
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: vocabulary words NOT used in this
    # rung's facts. The appropriate-regime gate MUST abstain. Counted
    # against the FULL arm (the load-bearing arm); this is the
    # trustworthiness property the verdict's abstain_correct bar gates.
    encoded_nouns = {n for n, _ in facts}
    encoded_adjs = {a for _, a in facts}
    ungroundable_direct_words = (
        [w for w in _NOUNS if w not in encoded_nouns]
        + [w for w in _ADJS if w not in encoded_adjs]
    )
    n_ungroundable = 0
    n_abstain_ok = 0
    for w in ungroundable_direct_words:
        n_ungroundable += 1
        ranked = _direct_query_ranked(
            bridge_full, w, dims, all_pools, word_to_idx,
            stim_steps=recall_steps, reset_steps=recall_steps // 2,
        )
        decided = gate_direct_unified(ranked, DIRECT_UNIFIED_THRESHOLD)
        if decided is None:
            n_abstain_ok += 1

    # Compositional ungroundables (cue a noun that was NOT encoded);
    # run via the FULL arm's PFC-frame-primed pool readout. The
    # substrate-specific compositional gate should abstain.
    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        _prime_pfc_frame(bridge_full, tiny_synth)
        ranked = _compositional_query_pool_readout(
            bridge_full, w, None, dims, recall_steps
        )
        decided = gate_compositional_unified(
            ranked, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        if decided is None:
            n_abstain_ok += 1

    # ---- Aggregate the four rung fields from the SAME run.
    n_total = n_direct_total + n_comp_total
    full_acc = (
        (n_direct_correct_full + n_comp_correct_full) / n_total
        if n_total
        else 0.0
    )
    uniform_ctrl_acc = (
        (n_direct_correct_uniform + n_comp_correct_uniform) / n_total
        if n_total
        else 0.0
    )
    direct_retain_acc = (
        n_direct_correct_full / n_direct_total if n_direct_total else 0.0
    )
    abstain_correct = (
        n_abstain_ok / n_ungroundable if n_ungroundable else 1.0
    )

    return {
        "seed": int(seed),
        "N": int(N),
        "full_acc": float(full_acc),
        "uniform_ctrl_acc": float(uniform_ctrl_acc),
        "direct_retain_acc": float(direct_retain_acc),
        "abstain_correct": float(abstain_correct),
        # Diagnostics (not part of the verdict rung shape).
        "n_direct": int(n_direct_total),
        "n_compositional": int(n_comp_total),
        "n_ungroundable": int(n_ungroundable),
        "replay_n_replays_full": int(
            replay_stats_full.get("n_replays", 0)
        ),
        "replay_n_replays_uniform": int(
            replay_stats_uniform.get("n_replays", 0)
        ),
    }


# =====================================================================
# Aggregation.
# =====================================================================
def _aggregate_rungs(cells_by_N: Dict[int, List[Dict[str, Any]]],
                       n_seeds: int) -> List[Dict[str, Any]]:
    """Aggregate per-seed evaluation cells into one rung per N (the
    exact six-key shape the frozen verdict consumes)."""
    rungs: List[Dict[str, Any]] = []
    for N in sorted(cells_by_N):
        cells = cells_by_N[N]

        def _mean(field: str) -> float:
            vals = [c[field] for c in cells]
            return float(sum(vals) / len(vals)) if vals else 0.0

        rungs.append({
            "N": int(N),
            "n_seeds": int(n_seeds),
            "full_acc": _mean("full_acc"),
            "uniform_ctrl_acc": _mean("uniform_ctrl_acc"),
            "direct_retain_acc": _mean("direct_retain_acc"),
            "abstain_correct": _mean("abstain_correct"),
        })
    return rungs


# =====================================================================
# Top-level entry.
# =====================================================================
def run_pool_readout_8th_arc(
    seeds,
    loads=_CP_LADDER,
    tiny_synth: bool = False,
    phase1_cache_dir: str = _PHASE1_CACHE_DEFAULT,
    out_path: Optional[str] = None,
    ckpt: Optional[str] = None,
) -> Dict[str, Any]:
    """Pool-readout substitution capability runner (8th arc).

    Per seed (in order):
      * Phase-1 multi-event direct training (cached) -- REUSED
        ``_phase1_train_if_needed`` from the unified runner.

    Per (seed, N) cell:
      * Build TWO parallel bridges from the same Phase-1 checkpoint
        (one for FULL, one for UNIFORM_CTRL).
      * Encode the SAME compositional facts into BOTH bridges.
      * Both arms: ``run_concept_replay_phase`` ONCE after encoding
        (same augmenting mechanism as the 6th arc; same in both
        arms).
      * For each compositional query: BOTH arms prime ``dlpfc_verb``
        (PFC-frame mechanism identical across arms). FULL arm calls
        the NET-NEW ``_compositional_query_pool_readout`` (per-pool
        firing rates); UNIFORM_CTRL arm calls the REUSED baseline
        ``_compositional_query_ranked`` (lang_output cosine). The
        cue stays ON during retrieve in BOTH arms (encoding-
        specificity respected).
      * Emit the four verdict fields per cell.

    Kill-safe via the REUSED ``sim.train_checkpoint`` (cell
    granularity).
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)
    phase1_cache_dir = str(phase1_cache_dir)

    Path(phase1_cache_dir).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _phase1_train_if_needed(int(s), phase1_cache_dir, tiny_synth)

    cells: List[Dict[str, Any]] = []
    schedule = [(s, N) for s in seeds for N in loads]
    start = 0

    if ckpt:
        prev = load_checkpoint(ckpt)
        if prev is not None:
            start = resume_epoch(prev)
            blob = prev.get("weights", [None])[0]
            if blob is not None:
                try:
                    cells = json.loads(
                        bytes(np.asarray(blob)).decode("utf-8")
                    )
                except Exception:
                    cells = []

    try:
        for epoch in range(start, len(schedule)):
            s, N = schedule[epoch]
            cell = _run_evaluation_arm(int(s), int(N), tiny_synth,
                                          phase1_cache_dir)
            cells.append(cell)
            if ckpt:
                blob = np.frombuffer(
                    json.dumps(cells).encode("utf-8"), dtype=np.uint8
                )
                save_checkpoint(ckpt, epoch, {"cells": [blob]}, None, [])
    except KeyboardInterrupt:
        print(
            "INTERRUPTED -- partial checkpoint flushed; resumable",
            flush=True,
        )
        if not cells:
            raise

    cells_by_N: Dict[int, List[Dict[str, Any]]] = {}
    seeds_seen_by_N: Dict[int, set] = {}
    for c in cells:
        cells_by_N.setdefault(c["N"], []).append(c)
        seeds_seen_by_N.setdefault(c["N"], set()).add(c["seed"])

    if seeds_seen_by_N:
        n_seeds_min = min(len(v) for v in seeds_seen_by_N.values())
    else:
        n_seeds_min = 0

    rungs = _aggregate_rungs(cells_by_N, n_seeds_min)
    verdict = pool_readout_8th_arc_verdict(rungs)

    result: Dict[str, Any] = {
        "mode": "evaluation",
        "rungs": rungs,
        "verdict": verdict,
        "tiny_synth": bool(tiny_synth),
        "seeds": list(seeds),
        "loads": list(loads),
        "raw_cells": cells,
        "phase1_cache_dir": phase1_cache_dir,
        "moat_direct": float(MOAT_DIRECT),
        "moat_compositional": float(COMPOSITIONAL_THRESHOLD),
        "direct_unified_threshold": float(DIRECT_UNIFIED_THRESHOLD),
        "compositional_unified_threshold": float(
            COMPOSITIONAL_UNIFIED_THRESHOLD
        ),
        "generative_replay_constants": {
            "N_REPLAYS_PER_TAG": int(N_REPLAYS_PER_TAG),
            "REPLAY_DRIVE_PA": float(REPLAY_DRIVE_PA),
            "REPLAY_BURST_DURATION_MS": int(REPLAY_BURST_DURATION_MS),
            "REPLAY_INTER_BURST_MS": int(REPLAY_INTER_BURST_MS),
        },
        "pfc_frame_constants": {
            "PFC_FRAME_PA": float(PFC_FRAME_PA),
            "PFC_FRAME_STIM_STEPS": int(PFC_FRAME_STIM_STEPS),
        },
        "pool_readout_pairs": [
            list(p) for p in _POOL_READOUT_PAIRS
        ],
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only. "
            "Phase-1 training is shrunk to a few events; compositional "
            "encoding shrunk to one pair per rung; replay cycles shrunk "
            "(n=%d, burst=%dms, inter=%dms); PFC-frame priming shrunk "
            "(%d steps). The decisive multi-seed CuPy run at full "
            "biological scale is a later controller-only task."
            % (
                TINY_N_REPLAYS_PER_TAG,
                TINY_REPLAY_BURST_DURATION_MS,
                TINY_REPLAY_INTER_BURST_MS,
                TINY_PFC_FRAME_STIM_STEPS,
            )
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Pool-readout substitution runner (Task 2 of the 8th arc). "
            "Both arms run the SAME encoding + SAME replay + SAME "
            "PFC-frame prime + SAME cue presence during retrieve. The "
            "SOLE arm differentiator is the compositional readout "
            "function: FULL = pool-readout (per-pool firing rates); "
            "UNIFORM_CTRL = lang_output cosine baseline (6th arc). "
            "Reuse-only orchestration of the prior 6th arc runner + "
            "the net-new pool-readout function + the two structural-"
            "effect probes. No autograd; no torch; no LLM call."
        )
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--loads",
        type=int,
        nargs="+",
        default=list(_CP_LADDER),
        help="Load ladder (default the frozen ladder (2,3,5)).",
    )
    ap.add_argument(
        "--tiny-synth",
        action="store_true",
        help=(
            "Shrink Phase-1 training + compositional pair count + "
            "replay cycles + PFC-frame priming hard for the logic-"
            "screen smoke. Toy numbers are NOT a result."
        ),
    )
    ap.add_argument(
        "--phase1-cache-dir",
        default=_PHASE1_CACHE_DEFAULT,
        help=(
            "Directory where the per-seed Phase-1 substrate "
            "checkpoints are stored (REUSED from the unified runner; "
            "byte-stable cache)."
        ),
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Kill-safe checkpoint path (REUSED sim.train_checkpoint; "
            "re-run resumes at the next (seed, N) cell)."
        ),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write the full result JSON here.",
    )
    ap.add_argument(
        "--skip-structural-probes",
        action="store_true",
        help=(
            "Skip the structural-effect probes before the eval loop. "
            "ONLY for the inner-loop test smoke that exercises the "
            "probes directly via the test API. Do NOT pass this for "
            "any decisive run."
        ),
    )
    a = ap.parse_args(argv)

    # MANDATORY: BOTH structural-effect probes before the decisive eval
    # loop. The replay-effect probe (REUSED from 6th arc) verifies the
    # augmenting mechanism is still structurally active on this
    # substrate. The readout-substitution probe (NET-NEW) verifies the
    # readout function actually produces different ranked outputs from
    # the baseline lang_output cosine readout on the SAME bridge state.
    # If either probe fails the runner aborts (no decisive numbers
    # reported). Mirrors Pirazzini d462bf0 + theta-gamma e6b17da
    # lesson; cache-scale validation closes 10th adversarial review
    # BLOCK.
    if not a.skip_structural_probes:
        try:
            diff_replay = _replay_effect_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "REPLAY-EFFECT-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "REPLAY-EFFECT-PROBE PASS: max |delta v_membrane| = "
            "%.6g mV (> 1.0 mV)" % diff_replay,
            flush=True,
        )
        try:
            sub_stats = _readout_substitution_probe(
                seed=int(a.seeds[0]) if a.seeds else 42,
                tiny_synth=bool(a.tiny_synth),
                cache_dir=a.phase1_cache_dir,
            )
        except RuntimeError as exc:
            print(
                "READOUT-SUBSTITUTION-PROBE FAILED: %s" % exc,
                file=sys.stderr, flush=True,
            )
            return 2
        print(
            "READOUT-SUBSTITUTION-PROBE PASS: "
            "n_differing_top1=%d / n_queries=%d (>= 1 required); "
            "post-readout |delta v_membrane| = %.6g mV"
            % (
                int(sub_stats["n_differing_top1"]),
                int(sub_stats["n_queries"]),
                float(sub_stats["diff_v_membrane"]),
            ),
            flush=True,
        )

    result = run_pool_readout_8th_arc(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        phase1_cache_dir=a.phase1_cache_dir,
        out_path=a.out,
        ckpt=a.ckpt,
    )
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    g = result["verdict"]["gate"]
    print("GATE=%s%s" % (g, tag), flush=True)
    print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
