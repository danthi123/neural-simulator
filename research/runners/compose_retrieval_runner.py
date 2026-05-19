"""Net-new regime-correct compositional-retrieval runner (Architecture A).

The brain composes RECENT-SPECIFIC hippocampal retrieval with
ORDER-INVARIANT consolidated neocortical semantics as retrieval-
augmented generation, EACH SYSTEM READ IN ITS OWN REGIME, with a
confidence/abstention monitor at output (design doc
docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md).

This module is the ONLY genuinely net-new wiring: a composition/
routing controller. EVERYTHING else is reused BYTE-UNCHANGED by import:

  * substrate + hippocampus: REUSED build_biological_brain_regions
    (text_minimal_isolation) constructed via the SAME
    CoreSimConfig -> enable_brain_region_framework=True ->
    SimulationBridge -> _initialize_simulation_data path that
    concept_pool_demo.build_concept_bridge and
    consolidation_trainer.run_consolidation_training both use. We
    mirror only that construction call (build_concept_bridge does not
    expose enable_hippocampus_consolidation and is protected, so we
    cannot extend it -- we re-issue the identical construction
    sequence with the hippocampus kwarg, duplicating NO subsystem
    logic).
  * recent-specific encode: REUSED compose_concept_engram.encode_
    concept_pair with a hippocampal region_filter (the validated
    Tonegawa stim-recall path, catalog D.14).
  * remote-semantic build: REUSED consolidation_trainer.run_concept_
    replay_phase + run_swr_replay_phase (validated replay-
    consolidation, McClelland 1995 / Buzsaki 2013).
  * regime-correct readout: REUSED compose_concept_engram.lang_output_
    pattern_during_stim (hippocampal regime) + lang_output_pattern_
    during_input (consolidated regime) + cosine_to_word.
  * remote-only ablation: the REUSED consolidation_eval hippo-OFF
    strict-silence protocol (HIPPO_REGIONS + a per-step silencing
    monkey-patch restored in finally -- byte-identical mechanism to
    the validated hippo-OFF read).
  * no-confabulation moat: REUSED abstention_gate.gate(ranked, 650.0)
    (byte-unchanged 7/7).
  * kill-safe/resume: REUSED sim.train_checkpoint (same per-cell
    save_checkpoint pattern as compose_bind_gate).
  * frozen verdict: REUSED compose_retrieval_core.compose_retrieval_
    verdict.

NO automatic differentiation anywhere. ASCII only. CuPy is the real/
decisive path; --tiny-synth forces the NumPy backend and shrinks
pools/episodes so the smoke is seconds -- its toy numbers are
explicitly NOT a result (they only screen for fatal logic flaws and
make the Task-0 pin green). The decisive multi-seed CuPy run is a
later controller-only task, NOT performed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Backend policy for the tiny-synth logic-screen smoke.
#
# The project rule is "NumPy ONLY for the smoke; CuPy is the decisive
# path". That rule targets a GPU-less environment. SimulationBridge
# binds its array module at sim.bridge IMPORT time (bridge.py does
# `cp, _ = get_backend()` once, and ~19 GPU sites remain unmigrated --
# CLAUDE.md Phase-1 scope), so a bridge constructed on a box where CuPy
# is importable runs its arrays on CuPy regardless of a later
# SIM_BACKEND flip; forcing NumPy then would mix a NumPy drive array
# with the bridge's CuPy state and raise. Therefore: prefer NumPy for
# the smoke, but only when CuPy is genuinely unavailable; on a
# CuPy-capable box the tiny smoke runs on the bridge's real backend
# (still seconds -- pools/episodes are shrunk hard). The decisive
# multi-seed run is CuPy and is a later controller-only task.
if "--tiny-synth" in sys.argv:
    try:
        import cupy as _cupy_probe  # noqa: F401

        _CUPY_AVAILABLE = True
    except Exception:
        _CUPY_AVAILABLE = False
    if not _CUPY_AVAILABLE:
        os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

from research.runners.compose_retrieval_core import (
    compose_retrieval_verdict,
    _CR_LADDER,
)
from research.runners.abstention_gate import gate as _abstain_gate
from research.runners.abstention_gate import DEFAULT_THRESHOLD as _MOAT
from sim.train_checkpoint import (  # REUSED UNMODIFIED
    save_checkpoint,
    load_checkpoint,
    resume_epoch,
)

# ---------------------------------------------------------------------
# Vocabulary. Concept words map to the validated v16 concept pools (the
# same 16-pool recipe concept_pool_demo trains). The recent-specific
# facts pair a NOUN with an ADJECTIVE (e.g. "apple is big"); the
# general/semantic structure is the consolidated noun<->adjective
# schema. This mirrors compose_concept_engram's concept-concept design
# (no motor routing -- output is via lang_output cosine).
# ---------------------------------------------------------------------
_NOUNS = ["apple", "river", "dog", "cat"]
_VERBS = ["go", "come", "stop", "look"]
_ADJS = ["big", "small", "hot", "cold"]
# Orthogonal-code index space the reused helpers expect (must match
# concept_compose_train._WORD_TO_IDX ordering for the first 16 words).
_N_WORDS_ORTHOGONAL = 16


# Fixed pre-registered pairing of recent facts: the i-th recent fact is
# (noun_i, adj_i). The "general structure" query for noun_i asks which
# adjective the consolidated schema associates -- correct iff it
# resolves to adj_i. Up to 4 distinct (noun, adj) facts available;
# higher loads recycle nouns with rotated adjectives so N can scale to
# the frozen ladder (2, 4, 8) without changing the architecture.
def _recent_facts(N: int) -> List[Tuple[str, str]]:
    facts: List[Tuple[str, str]] = []
    for i in range(N):
        noun = _NOUNS[i % len(_NOUNS)]
        adj = _ADJS[(i + (i // len(_NOUNS))) % len(_ADJS)]
        facts.append((noun, adj))
    return facts


# =====================================================================
#  Substrate + hippocampus construction (mirror the reused path).
# =====================================================================
def _build_substrate(seed: int, tiny_synth: bool):
    """Construct a v16-style concept-pool bridge WITH the hippocampal
    consolidation regions, via the SAME construction sequence the
    reused builders use. Returns (bridge, dims).

    tiny_synth shrinks every dimension hard and forces NumPy.
    """
    if tiny_synth:
        # Only pin NumPy when CuPy is genuinely unavailable (GPU-less
        # box). On a CuPy-capable box the bridge's arrays are CuPy
        # (import-time bound), so forcing NumPy here would mix backends
        # and raise; the tiny smoke instead runs on the bridge's real
        # backend with shrunk pools/episodes (still seconds). See the
        # module-top backend-policy comment.
        try:
            import cupy as _c  # noqa: F401

            _cupy_ok = True
        except Exception:
            _cupy_ok = False
        if not _cupy_ok:
            os.environ["SIM_BACKEND"] = "numpy"
            from sim.backend import get_backend as _get_backend
            _get_backend("numpy")

    # Imports deferred so module import (and the autograd grep test) is
    # cheap and does not require CuPy.
    from sim.config import (
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )

    if tiny_synth:
        n_lang_input = 64
        n_per_pool = 12
        n_fs_per_pool = 3
    else:
        # Decisive-path defaults (validated v16 recipe scale).
        n_lang_input = 2048
        n_per_pool = 200
        n_fs_per_pool = 24

    # Weak concept-pool dynamics = the validated v16 setting (canon
    # dynamics amplify structural bias at scale). Mirrors
    # build_concept_bridge(weak_dynamics=True).
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=0.10,
        motor_exc_weight_mean=2.0,
        motor_inh_weight_mean=4.0,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=[n.upper() for n in _NOUNS],
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=[v.upper() for v in _VERBS],
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=[a.upper() for a in _ADJS],
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8,
        # The hippocampal recent-specific path (catalog D.03/D.12/D.13).
        enable_hippocampus_consolidation=True,
    )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    # Single-type per region: the concept-pool recipe sets izh_neuron_type
    # per BrainRegion and does NOT use within-pool trait heterogeneity, so
    # opt OUT of trait-split (num_traits=1 -> bridge uses the per-region
    # default neuron type for all neurons). This is the documented
    # backward-compatible single-type path; it changes NO subsystem logic
    # and avoids the bridge's trait-split numpy/cupy default-param mix that
    # breaks the NumPy smoke backend (CLAUDE.md: trait-split is opt-in only
    # when num_traits > 1).
    cfg.num_traits = 1
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
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

    dims = {
        "n_lang_input": n_lang_input,
        "n_per_pool": n_per_pool,
        "n_fs_per_pool": n_fs_per_pool,
        "sparsity": 0.05,
    }
    return bridge, dims


# Concept-pool region filter (no motor, no hippocampus) -- the
# consolidated-semantic carrier. Mirrors compose_concept_engram.main.
def _concept_region_filter() -> List[str]:
    return (
        [f"noun_pool_{n.upper()}" for n in _NOUNS]
        + [f"verb_pool_{v.upper()}" for v in _VERBS]
        + [f"adjective_pool_{a.upper()}" for a in _ADJS]
    )


# Hippocampal regions that carry the recent-specific trace. The recent
# engram is tagged HERE (not the concept pools) so the recent-specific
# part lives in the hippocampal regime and collapses under hippo-OFF.
_HIPPO_TAG_REGIONS = ["dg", "ca3", "ca1"]


# =====================================================================
#  Composition/routing controller (the ONLY net-new logic).
# =====================================================================
def _ranked_from_pattern(pattern, n_lang_out: int, dims: Dict[str, Any],
                          exclude: Optional[str] = None):
    """Cosine-rank every concept word against a lang_output pattern,
    returned as the moat's expected [(concept, rate, tag), ...] desc.

    The "rate" is the raw lang_output drive into that word's pattern
    (cosine * the pattern energy) -- the abstention moat is calibrated
    on raw lang_output rate, so we scale cosine by the pattern's
    L2 energy to keep it on that scale rather than the [0,1] cosine.
    """
    from research.runners.compose_concept_engram import cosine_to_word

    energy = float(np.linalg.norm(np.asarray(pattern)))
    ranked = []
    for w in _NOUNS + _VERBS + _ADJS:
        if exclude is not None and w == exclude:
            continue
        cos = cosine_to_word(
            pattern, w, n_lang_out,
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            sparsity=dims["sparsity"],
        )
        ranked.append((w, max(0.0, cos) * energy, "lang_output"))
    ranked.sort(key=lambda t: -t[1])
    return ranked


def _compose_query(bridge, cue_noun: str, tag_name: Optional[str],
                   dims: Dict[str, Any], have_remote: bool,
                   recall_steps: int):
    """RETRIEVAL-AUGMENTED composition for ONE compositional query.

    (i)  hippocampal regime: stim the recent-fact engram tag (if it
         exists) and read lang_output -> recent-specific retrieval.
    (ii) consolidated regime: drive lang_input(cue_noun) and read
         lang_output -> the order-invariant neocortical schema.
    (iii) compose: the hippocampal retrieval CONDITIONS (augments) the
         consolidated ranking -- per-concept scores are summed
         (consolidated base + hippocampal conditioning prior),
         producing ONE ranked (concept, rate, tag) list.
    (iv) the top candidate goes through the REUSED no-confab moat;
         answer if it clears, else abstain ("I don't know").

    Returns (answer_or_None, ranked, parts) where answer_or_None is the
    moat's decision (None == abstained).
    """
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_stim,
        lang_output_pattern_during_input,
    )

    # (ii) consolidated-regime read (order-invariant semantic schema).
    if have_remote:
        cons_pat, n_lo = lang_output_pattern_during_input(
            bridge, cue_noun,
            n_lang_input=dims["n_lang_input"],
            sparsity=dims["sparsity"],
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            stim_steps=recall_steps,
        )
        cons_ranked = _ranked_from_pattern(
            cons_pat, n_lo, dims, exclude=cue_noun
        )
    else:
        # remote regime ungrounded (recent-only ablation): no schema.
        cons_ranked, n_lo = [], dims["n_lang_input"]

    # (i) hippocampal-regime read (recent-specific retrieval).
    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        hip_pat, n_lo2 = lang_output_pattern_during_stim(
            bridge, tag_name, drive_pA=1500.0, stim_steps=recall_steps,
        )
        hip_ranked = _ranked_from_pattern(
            hip_pat, n_lo2, dims, exclude=cue_noun
        )
    else:
        # recent regime ungrounded (remote-only ablation silences the
        # hippocampus so the tagged ensemble cannot reactivate).
        hip_ranked = []

    # (iii) retrieval-augmented compose: sum per-concept scores. The
    # hippocampal retrieval conditions the consolidated ranking.
    scores: Dict[str, float] = {}
    for w, r, _ in cons_ranked:
        scores[w] = scores.get(w, 0.0) + r
    for w, r, _ in hip_ranked:
        scores[w] = scores.get(w, 0.0) + r
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    # (iv) no-confabulation moat (REUSED, byte-unchanged).
    decided = _abstain_gate(ranked, _MOAT)
    answer = None if decided is None else decided[0]
    return answer, ranked, {
        "n_cons": len(cons_ranked),
        "n_hip": len(hip_ranked),
    }


# =====================================================================
#  One (seed, N) cell: full + recent_only + remote_only, SAME draws.
# =====================================================================
def _encode_recent_facts(bridge, facts, dims, encoding_steps: int):
    """Recent-specific encode in the HIPPOCAMPAL regime: each fact is a
    Tonegawa engram over the hippocampal regions (reused encode_concept_
    pair with a hippocampal region_filter). Returns tag names."""
    from research.runners.compose_concept_engram import encode_concept_pair

    tags = []
    for (noun, adj) in facts:
        tag = f"recent__{noun}__{adj}"
        if tag in {t["name"] for t in bridge.list_engram_tags()}:
            try:
                bridge.delete_engram_tag(tag)
            except Exception:
                pass
        encode_concept_pair(
            bridge, noun, adj, tag,
            encoding_steps=encoding_steps,
            drive_pA=200.0, sparsity=dims["sparsity"],
            n_lang_input=dims["n_lang_input"],
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            region_filter=_HIPPO_TAG_REGIONS,
            top_k=max(8, dims["n_per_pool"] // 4),
            balanced_teacher_pA=500.0,
            verbose=False,
        )
        tags.append(tag)
    return tags


def _build_remote_schema(bridge, tags, dims, tiny_synth: bool, rng):
    """Remote-semantic build in the CONSOLIDATED regime: reused
    replay-consolidation. Concept replay over the recent tags +
    generic SWR replay consolidate the order-invariant schema into
    cortex (ca3->ca1->cortex STDP)."""
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
        run_swr_replay_phase,
    )

    n_replays = 2 if tiny_synth else 20
    n_swr = 4 if tiny_synth else 200
    run_concept_replay_phase(
        bridge, tags,
        n_replays_per_tag=n_replays,
        burst_duration_ms=10 if tiny_synth else 100,
        inter_burst_ms=5 if tiny_synth else 50,
        drive_pA=100.0,
        randomize_order=True,
        rng=rng,
    )
    try:
        run_swr_replay_phase(
            bridge,
            n_swr_events=n_swr,
            burst_duration_ms=10 if tiny_synth else 100,
            inter_burst_ms=5 if tiny_synth else 50,
            swr_drive_pA=100.0,
            rng=rng,
        )
    except Exception:
        # run_swr_replay_phase imports cupy at top; on the NumPy smoke
        # path it may be unavailable. Concept replay alone still builds
        # a (toy) schema -- acceptable for the logic-screen smoke (its
        # numbers are explicitly NOT a result). The decisive CuPy path
        # exercises both.
        pass


def _hippo_silenced(bridge, silence_current_pA: float = -2000.0):
    """Context-managing the VALIDATED strict hippo-OFF protocol:
    monkey-patch _run_one_simulation_step to pin HIPPO_REGIONS at a
    strong negative current every step, restored in finally. This is
    the byte-identical mechanism consolidation_eval's validated
    hippo-OFF read uses (strict anti-cheat silence = -2000 pA per the
    validated Phase 1.3 strict protocol). Returns a restore() callable
    and the count silenced."""
    from sim.backend import get_backend
    from research.runners.consolidation_eval import HIPPO_REGIONS

    cp, _ = get_backend()
    rm = bridge.region_manager
    hippo_idx: List[int] = []
    for rname in HIPPO_REGIONS:
        try:
            idx = rm.indices(rname)
            if idx is not None:
                hippo_idx.extend(list(idx))
        except Exception:
            pass
    if not hippo_idx:
        return (lambda: None), 0
    hippo_arr = cp.asarray(hippo_idx, dtype=cp.int64)
    original_step = bridge._run_one_simulation_step

    def silenced_step():
        bridge.cp_external_input_current[hippo_arr] = float(
            silence_current_pA
        )
        return original_step()

    bridge._run_one_simulation_step = silenced_step

    def restore():
        bridge._run_one_simulation_step = original_step
        bridge.cp_external_input_current[hippo_arr] = 0.0

    return restore, len(hippo_idx)


def _score_arm(bridge, facts, tags, dims, have_remote: bool,
               hippo_off: bool, recall_steps: int):
    """Run every compositional query for one arm and score:
      full_acc / *_acc      : fraction answered correctly,
      abstain_correct       : among queries whose correct answer is
                              UNGROUNDABLE in this arm, the fraction on
                              which the system correctly abstained
                              (the no-confab invariant).

    A query is "groundable" only if BOTH regimes that its correct
    answer needs are present in this arm:
      * the recent-specific (noun->adj) binding needs the hippocampal
        regime  -> ungroundable when hippo_off (remote-only);
      * resolving it via general structure needs the consolidated
        schema -> ungroundable when not have_remote (recent-only).
    The full arm has both, so every query is groundable there and the
    abstain-correct denominator is empty (reported as 1.0 -- vacuously
    satisfied; the FROZEN verdict's abstain bar is only decision-
    relevant on the ablation arms).
    """
    if hippo_off:
        restore, _ = _hippo_silenced(bridge)
    else:
        restore = lambda: None

    n_correct = 0
    n_total = 0
    n_abstain_ok = 0
    n_ungroundable = 0
    try:
        for (noun, adj) in facts:
            n_total += 1
            tag = f"recent__{noun}__{adj}"
            tag_arg = None if hippo_off else tag
            answer, ranked, _ = _compose_query(
                bridge, noun, tag_arg, dims, have_remote, recall_steps,
            )
            groundable = have_remote and (not hippo_off)
            if groundable:
                if answer == adj:
                    n_correct += 1
            else:
                # correct answer is ungroundable in this ablation: the
                # honest behaviour is to ABSTAIN, never confabulate.
                n_ungroundable += 1
                if answer is None:
                    # abstained (did not confabulate the ungroundable
                    # correct answer).
                    n_abstain_ok += 1
    finally:
        restore()

    acc = (n_correct / n_total) if n_total else 0.0
    abstain_correct = (
        (n_abstain_ok / n_ungroundable) if n_ungroundable else 1.0
    )
    return acc, abstain_correct


def _cell_passes(seed: int, N: int, tiny_synth: bool, **kw) -> Dict[str, Any]:
    """Run the full pass and BOTH ablations for ONE (seed, N) cell.

    CRITICAL FAITHFULNESS INVARIANT: all three arms use the SAME seed
    and the SAME random draws. We achieve "full minus exactly one
    regime, same draws" by building ONE substrate per arm from the
    SAME seed and running the SAME recent-fact encode (same per-fact
    pairs, same RNG seeded from `seed`), then:
      * full        : encode recent + build remote schema, hippo on.
      * recent_only : encode recent, SKIP the consolidation/replay
                      build entirely (no remote schema), hippo on.
      * remote_only : encode recent + build remote schema, then the
                      VALIDATED strict hippo-OFF silence at read time.
    recent_only differs from full by exactly the consolidation step;
    remote_only differs from full by exactly the hippocampus regime.
    """
    recall_steps = 20 if tiny_synth else 100
    enc_steps = 8 if tiny_synth else 200
    facts = _recent_facts(N)

    def _one_arm(have_remote: bool, hippo_off: bool):
        # Same seed -> same substrate init; same RNG -> same replay
        # draws; same facts -> same encode draws.
        bridge, dims = _build_substrate(seed, tiny_synth)
        rng = np.random.default_rng(seed)
        tags = _encode_recent_facts(bridge, facts, dims, enc_steps)
        if have_remote:
            _build_remote_schema(bridge, tags, dims, tiny_synth, rng)
        acc, abst = _score_arm(
            bridge, facts, tags, dims,
            have_remote=have_remote, hippo_off=hippo_off,
            recall_steps=recall_steps,
        )
        return {"seed": seed, "acc": acc, "abstain_correct": abst}

    full = _one_arm(have_remote=True, hippo_off=False)
    recent_only = _one_arm(have_remote=False, hippo_off=False)
    remote_only = _one_arm(have_remote=True, hippo_off=True)
    return {
        "N": N,
        "full": full,
        "recent_only": recent_only,
        "remote_only": remote_only,
    }


# =====================================================================
#  Aggregation + top-level entry.
# =====================================================================
def _aggregate(cells_by_N: Dict[int, List[Dict[str, Any]]],
               n_seeds: int) -> List[Dict[str, Any]]:
    """Aggregate per-seed cells into one rung dict per N (the exact
    shape the frozen verdict consumes)."""
    rungs = []
    for N in sorted(cells_by_N):
        cells = cells_by_N[N]

        def _mean(arm: str, field: str) -> float:
            vals = [c[arm][field] for c in cells]
            return float(sum(vals) / len(vals)) if vals else 0.0

        rungs.append({
            "N": int(N),
            "n_seeds": int(n_seeds),
            "full_acc": _mean("full", "acc"),
            "recent_only_acc": _mean("recent_only", "acc"),
            "remote_only_acc": _mean("remote_only", "acc"),
            "abstain_correct_recent_only": _mean(
                "recent_only", "abstain_correct"
            ),
            "abstain_correct_remote_only": _mean(
                "remote_only", "abstain_correct"
            ),
        })
    return rungs


def run_compose_retrieval(seeds, loads=_CR_LADDER, tiny_synth: bool = False,
                          out_path: Optional[str] = None,
                          ckpt: Optional[str] = None) -> Dict[str, Any]:
    """Run the regime-correct compositional-retrieval capability test.

    Per seed, per load N in the frozen ladder: build substrate +
    hippocampus, encode N recent-specific facts (hippocampal regime),
    build the consolidated semantic schema (consolidated regime), then
    score the full system + the recent-only and remote-only ablations
    (each 'full minus exactly one regime, same draws'). Aggregate to
    rungs and score with the FROZEN verdict module.

    Kill-safe/resumable via the REUSED sim.train_checkpoint: completed
    (seed, N) cells are flushed; re-running resumes past them.
    """
    seeds = list(seeds)
    loads = tuple(int(x) for x in loads)

    # --- resume past completed cells (REUSED checkpoint scheme) -------
    cells: List[Dict[str, Any]] = []
    start = 0
    schedule = [(s, N) for s in seeds for N in loads]
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
            cell = _cell_passes(s, N, tiny_synth)
            cells.append({"seed": s, **cell})
            if ckpt:
                blob = np.frombuffer(
                    json.dumps(cells).encode("utf-8"), dtype=np.uint8
                )
                save_checkpoint(ckpt, epoch, {"cells": [blob]}, None, [])
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable",
              flush=True)
        if not cells:
            raise

    # --- aggregate to rungs + FROZEN verdict -------------------------
    cells_by_N: Dict[int, List[Dict[str, Any]]] = {}
    seeds_seen_by_N: Dict[int, set] = {}
    for c in cells:
        cells_by_N.setdefault(c["N"], []).append(c)
        seeds_seen_by_N.setdefault(c["N"], set()).add(c["seed"])

    # n_seeds per rung = number of distinct seeds with a cell at that N
    # (min across the ladder so a partial run reports honestly).
    if seeds_seen_by_N:
        n_seeds = min(len(v) for v in seeds_seen_by_N.values())
    else:
        n_seeds = 0
    rungs = _aggregate(cells_by_N, n_seeds)
    verdict = compose_retrieval_verdict(rungs)

    result = {
        "rungs": rungs,
        "verdict": verdict,
        "tiny_synth": bool(tiny_synth),
        "seeds": seeds,
        "loads": list(loads),
        "raw_cells": cells,
    }
    if tiny_synth:
        result["note"] = (
            "TINY-SYNTH toy numbers -- NOT a result; logic-screen only."
        )
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regime-correct compositional-retrieval runner "
                    "(Architecture A; reuse-only; no autograd)."
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--loads", type=int, nargs="+",
                    default=list(_CR_LADDER),
                    help="Load ladder (default the frozen ladder).")
    ap.add_argument("--tiny-synth", action="store_true",
                    help="Shrink pools/episodes + force NumPy backend "
                         "for the logic-screen smoke. Toy numbers are "
                         "NOT a result.")
    ap.add_argument("--ckpt", default=None,
                    help="Kill-safe checkpoint path (REUSED "
                         "sim.train_checkpoint; re-run resumes).")
    ap.add_argument("--out", default=None,
                    help="Write the full result JSON here.")
    a = ap.parse_args(argv)

    result = run_compose_retrieval(
        seeds=a.seeds,
        loads=tuple(a.loads),
        tiny_synth=a.tiny_synth,
        out_path=a.out,
        ckpt=a.ckpt,
    )
    g = result["verdict"]["gate"]
    tag = " [TINY-SYNTH toy -- NOT a result]" if a.tiny_synth else ""
    print("GATE=%s%s" % (g, tag), flush=True)
    print(json.dumps(result["rungs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
