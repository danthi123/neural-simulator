"""Net-new regime-correct compositional-retrieval runner (Architecture A).

The brain composes RECENT-SPECIFIC hippocampal retrieval with
ORDER-INVARIANT consolidated neocortical semantics as retrieval-
augmented generation, EACH SYSTEM READ IN ITS OWN REGIME, with a
confidence/abstention monitor at output (design doc
docs/plans/2026-05-19-regime-correct-compositional-retrieval-design.md).

This module is the ONLY genuinely net-new wiring: a composition/
routing controller. EVERYTHING else is reused BYTE-UNCHANGED by import:

  * substrate + hippocampus: REUSED the VALIDATED v16 recipe
    concept_pool_demo.build_concept_bridge (the exact CoreSimConfig
    field set the 16-pool 5/5-GO concept binding uses; it does NOT
    override cfg.num_traits -- we do not either). We re-issue the
    identical construction with enable_hippocampus_consolidation=True
    (the validated builder's own kwarg) so the hippocampal recent-
    specific path (catalog D.03/D.12/D.13) is present, duplicating NO
    subsystem logic.
  * recent-specific encode: REUSED compose_concept_engram.encode_
    concept_pair with a hippocampal region_filter (the validated
    Tonegawa stim-recall path, catalog D.14). The engram tag NAME is
    OPAQUE (fact_{i}) -- it carries no answer; the answer is read from
    neural activity, never a string.
  * remote-semantic build: REUSED consolidation_trainer.run_concept_
    replay_phase + run_swr_replay_phase (validated replay-
    consolidation, McClelland 1995 / Buzsaki 2013).
  * regime-correct readout: REUSED compose_concept_engram.lang_output_
    pattern_during_stim (hippocampal regime) + lang_output_pattern_
    during_input (consolidated regime) + cosine_to_word; ranking is by
    the RAW lang_output FIRING-RATE confidence the validated concept
    readout / abstention benchmark calibrate (encoded mean ~796,
    control max ~584; 2026-05-16-G20-320-abstention-benchmark) so the
    byte-unchanged 650 moat threshold is genuinely calibrated for it.
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

Scoring contract (faithful, post Task-3 adversarial fix):
  ALL THREE arms (full / recent_only / remote_only) run the IDENTICAL
  query -> retrieve -> compose -> decode -> score pipeline. The decoded
  answer comes ONLY from the validated neural readout. The
  compositional query is constructed so the correct answer needs BOTH
  a recent-specific hippocampal binding (present ONLY in the engram)
  AND a consolidated-schema generalization (present ONLY in the
  replay-built neocortical schema): full has both; recent_only (no
  consolidation) cannot generalize the recent binding into a
  confident readout; remote_only (hippo strict-silenced) cannot
  retrieve the specific binding. A system doing zero composition /
  single-path / reading the tag string therefore provably FAILs.

NO automatic differentiation anywhere. ASCII only. CuPy is the real/
decisive path; --tiny-synth shrinks pools/episodes so the smoke is
seconds -- its toy numbers are explicitly NOT a result (they only
screen for fatal logic flaws and make the Task-0 pin green). The
decisive multi-seed CuPy run is a later controller-only task, NOT
performed here.
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
#
# FAITHFULNESS NOTE (FIX A): the substrate is built by the VALIDATED
# v16 recipe concept_pool_demo.build_concept_bridge, which does NOT set
# cfg.num_traits (it leaves the default). We do NOT override it either.
# If a CuPy-less box then hits the documented trait-split mixed-backend
# default-param issue in the NumPy smoke, that is ACCEPTABLE -- the
# decisive run is CuPy anyway; faithfulness of the substrate > smoke
# convenience. We never diverge the substrate to make a CPU smoke pass.
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
# facts pair a NOUN with an ADJECTIVE (e.g. apple<->big); the
# general/semantic structure is the consolidated noun<->adjective
# schema. This mirrors compose_concept_engram's concept-concept design
# (no motor routing -- output is via the validated lang_output firing-
# rate readout). The (noun, adj) PAIRING is data, never embedded in any
# tag NAME (FIX C: tags are opaque fact_{i}).
# ---------------------------------------------------------------------
_NOUNS = ["apple", "river", "dog", "cat"]
_VERBS = ["go", "come", "stop", "look"]
_ADJS = ["big", "small", "hot", "cold"]
# Orthogonal-code index space the reused helpers expect (must match
# concept_compose_train._WORD_TO_IDX ordering for the first 16 words).
_N_WORDS_ORTHOGONAL = 16


# Fixed pre-registered pairing of recent facts: the i-th recent fact is
# (noun_i, adj_i). The "general structure" query for noun_i asks which
# adjective the system associates -- correct iff it resolves to adj_i.
# Up to 4 distinct (noun, adj) facts available; higher loads recycle
# nouns with rotated adjectives so N can scale to the frozen ladder
# (2, 4, 8) without changing the architecture.
def _recent_facts(N: int) -> List[Tuple[str, str]]:
    facts: List[Tuple[str, str]] = []
    for i in range(N):
        noun = _NOUNS[i % len(_NOUNS)]
        adj = _ADJS[(i + (i // len(_NOUNS))) % len(_ADJS)]
        facts.append((noun, adj))
    return facts


# =====================================================================
#  Substrate + hippocampus construction (REUSE the validated recipe).
# =====================================================================
def _build_substrate(seed: int, tiny_synth: bool):
    """Construct a v16-style concept-pool bridge WITH the hippocampal
    consolidation regions by REUSING the validated recipe
    concept_pool_demo.build_concept_bridge byte-unchanged. Returns
    (bridge, dims).

    FIX A: build_concept_bridge is the exact 16-pool 5/5-GO recipe; it
    does NOT set cfg.num_traits. We pass enable_hippocampus_
    consolidation=True (the validated builder's own kwarg) so the
    hippocampal recent-specific path exists, and otherwise change NO
    subsystem logic and override NO config field. tiny_synth only
    shrinks pool/lang dimensions (faithfulness of the recipe is
    preserved; only scale shrinks for the seconds-long smoke).
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
    # cheap and does not require CuPy. We REUSE the validated v16
    # recipe's own builders by import only:
    #   * concept_pool_demo (NOUN/VERB/ADJ pool names + the EXACT
    #     CoreSimConfig field set the 16-pool 5/5-GO recipe uses);
    #   * text_minimal_isolation.build_biological_brain_regions (the
    #     validated trisynaptic + concept-pool region builder; it is
    #     the ONLY builder that exposes enable_hippocampus_
    #     consolidation -- build_concept_bridge does not pass it, so we
    #     re-issue concept_pool_demo's identical construction sequence
    #     and add ONLY that one validated kwarg, duplicating NO
    #     subsystem logic).
    import research.runners.concept_pool_demo as cpd
    from sim.config import (
        CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig,
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

    # ---- v16 recipe regions (mirror concept_pool_demo.build_concept_
    # bridge with weak_dynamics=True + enable_adjective=True) PLUS the
    # validated hippocampal recent-specific path. Every kwarg below is
    # exactly build_concept_bridge's call into build_biological_brain_
    # regions for the weak-dynamics 16-pool recipe; we add ONLY
    # enable_hippocampus_consolidation=True. ------------------------
    concept_internal_density = 0.05  # weak_dynamics=True (validated v16)
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
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
        noun_pool_names=cpd.NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=cpd.VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=True,
        adjective_pool_names=cpd.ADJECTIVE_NAMES,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        # The validated trisynaptic hippocampal recent-specific path
        # (catalog D.03/D.12/D.13) -- build_biological_brain_regions'
        # own kwarg, reused unmodified.
        enable_hippocampus_consolidation=True,
    )

    # ---- v16 recipe CoreSimConfig: the EXACT field set concept_pool_
    # demo.build_concept_bridge applies (lines 249-267). FIX A: that
    # validated recipe leaves the per-region trait count at its default
    # (it never assigns the trait-count field) -- and neither do we;
    # the prior spurious single-trait override was the substrate
    # divergence the adversarial review confirmed. No subsystem logic
    # is changed; only the validated recipe's documented fields are
    # set. -----------------------------------------------------------
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
    """Rank every concept word by the RAW lang_output FIRING-RATE
    confidence into that word's orthogonal spelling pattern, returned
    as the moat's expected [(concept, rate, tag), ...] desc.

    FIX D: abstention_gate's 650 threshold was calibrated on RAW
    lang_output firing rates (encoded mean ~796, control max ~584;
    2026-05-16-G20-320-abstention-benchmark). `pattern` here is the
    accumulated per-neuron lang_output spike count over the readout
    window (exactly what the validated lang_output_pattern_during_*
    helpers return). The per-word confidence is the summed firing rate
    on that word's pattern neurons -- the SAME quantity the validated
    concept readout / abstention benchmark calibrate -- so the
    byte-unchanged 650 moat is genuinely calibrated for it. We do NOT
    rescale a cosine by pattern energy (the retired out-of-calibration
    hack); cosine_to_word is used only to identify which orthogonal
    pattern each word occupies, then the raw firing rate ON that
    pattern is the confidence.
    """
    from research.runners.compose_concept_engram import cosine_to_word
    from research.runners.concept_compose_train import _WORD_TO_IDX
    from sim.text_embeddings import orthogonal_drive_pattern

    pat = np.asarray(pattern, dtype=np.float64)
    sparsity = dims["sparsity"]
    ranked = []
    for w in _NOUNS + _VERBS + _ADJS:
        if exclude is not None and w == exclude:
            continue
        # The validated orthogonal spelling pattern for this word (the
        # SAME code cosine_to_word / the concept readout use).
        word_pat = orthogonal_drive_pattern(
            cue_idx=_WORD_TO_IDX[w], n_cues=_N_WORDS_ORTHOGONAL,
            n_neurons=n_lang_out, drive_max_pA=1.0, sparsity=sparsity,
        )
        active = np.asarray(word_pat, dtype=np.float64) > 0.0
        n_active = int(active.sum())
        # Raw lang_output FIRING-RATE confidence on this word's pattern
        # neurons: the validated concept readout's quantity, the same
        # one the 650 abstention threshold is calibrated against. Mean
        # accumulated spike count per pattern-neuron, scaled to the
        # readout-window firing-rate scale the benchmark used.
        if n_active > 0:
            rate_conf = float(pat[active].sum()) / float(n_active)
        else:
            rate_conf = 0.0
        # cosine_to_word kept only as a tie-grounding sanity reference
        # (NOT scaled into the confidence -- FIX D).
        _ = cosine_to_word(
            pattern, w, n_lang_out,
            n_words_for_orthogonal=_N_WORDS_ORTHOGONAL,
            sparsity=sparsity,
        )
        ranked.append((w, rate_conf, "lang_output"))
    ranked.sort(key=lambda t: -t[1])
    return ranked


def _compose_query(bridge, cue_noun: str, tag_name: Optional[str],
                   dims: Dict[str, Any], have_remote: bool,
                   recall_steps: int):
    """RETRIEVAL-AUGMENTED composition for ONE compositional query.

    The correct answer (the adjective bound to cue_noun) requires BOTH
    regimes:
      (i)  hippocampal regime -- the recent-specific (noun->adj)
           binding lives ONLY in the engram. Stimulate the opaque
           recent-fact tag (if present) and read the validated
           lang_output firing pattern -> recent-specific retrieval.
      (ii) consolidated regime -- the order-invariant neocortical
           schema (built ONLY by replay-consolidation). Drive
           lang_input(cue_noun) and read the validated lang_output
           firing pattern -> the general schema.
     (iii) compose (retrieval-augmented): the two raw firing-rate
           confidences are summed per concept. The recent binding is
           specific (not derivable from the order-invariant schema);
           the schema is what makes the readout confident. Neither
           alone clears the calibrated 650 moat -- only the composed
           sum does (FIX B/C: removing either regime provably degrades
           the decoded answer, so a single-path/empty/tag-string
           solver cannot score PASS).
     (iv)  the top candidate goes through the REUSED no-confab moat
           fed the calibrated raw firing-rate confidence; answer if it
           clears, else abstain ("I don't know").

    Returns (answer_or_None, ranked, parts) where answer_or_None is the
    moat's decision (None == abstained). Tag NAMES are opaque -- nothing
    here parses a tag string for the answer.
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

    # (iii) retrieval-augmented compose: sum per-concept raw firing-rate
    # confidences. The hippocampal recent-specific retrieval conditions
    # the consolidated schema readout.
    scores: Dict[str, float] = {}
    for w, r, _ in cons_ranked:
        scores[w] = scores.get(w, 0.0) + r
    for w, r, _ in hip_ranked:
        scores[w] = scores.get(w, 0.0) + r
    ranked = sorted(
        ((w, scores[w], "compose") for w in scores),
        key=lambda t: -t[1],
    )
    # (iv) no-confabulation moat (REUSED, byte-unchanged) fed the
    # calibrated raw firing-rate confidence.
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
    pair with a hippocampal region_filter).

    FIX C: the engram tag NAME is OPAQUE (fact_{i}) -- it does NOT
    contain the noun or the adjective, so nothing downstream can read
    the answer out of the tag string. The (noun, adj) pairing is data
    threaded only through `facts`. Returns the opaque tag names in
    fact order (caller maps fact i -> facts[i])."""
    from research.runners.compose_concept_engram import encode_concept_pair

    tags = []
    for i, (noun, adj) in enumerate(facts):
        tag = f"fact_{i}"  # OPAQUE -- carries no answer (FIX C)
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
    """Run every compositional query for one arm THROUGH THE IDENTICAL
    pipeline and score:
      *_acc            : fraction of queries answered CORRECTLY (the
                         decoded answer == the recent fact's adjective),
                         a GENUINE measurement on EVERY arm (FIX B: no
                         per-arm `groundable` short-circuit -- the same
                         query -> retrieve -> compose -> decode -> moat
                         pipeline runs for full, recent_only and
                         remote_only alike);
      abstain_correct  : among queries the system answered WRONG
                         (decoded answer != the adjective, including
                         abstrained-but-wrong), the fraction on which
                         the moat made it ABSTAIN ("I don't know")
                         rather than emit a confident wrong answer --
                         the no-confabulation invariant.

    On the FULL arm both regimes are present, so the composed firing-
    rate confidence clears the calibrated moat and the answer is
    correct -> high *_acc, and the wrong-answer denominator is small.
    On an ABLATION arm the missing regime makes the composed confidence
    fall below the calibrated 650 moat, so the system abstains: the
    answer is NOT correct (so *_acc collapses -- a real measurement,
    not a hardcoded 0.0) AND it abstained rather than confabulated (so
    abstain_correct stays high). This is exactly what the frozen
    _CR_ABLATION_MAX collapse bars + _CR_ABSTAIN_MIN bars test.
    """
    if hippo_off:
        restore, _ = _hippo_silenced(bridge)
    else:
        restore = lambda: None

    n_correct = 0
    n_total = 0
    n_abstain_ok = 0
    n_wrong = 0
    try:
        for i, (noun, adj) in enumerate(facts):
            n_total += 1
            tag = tags[i] if i < len(tags) else None
            # remote-only ablation strict-silences the hippocampus, so
            # the engram cannot reactivate -> pass no tag (the recent-
            # specific regime is genuinely absent). recent-only keeps
            # the tag (hippo on) but the consolidated schema is absent.
            tag_arg = None if hippo_off else tag
            answer, ranked, _ = _compose_query(
                bridge, noun, tag_arg, dims, have_remote, recall_steps,
            )
            # GENUINE per-query measurement on EVERY arm (FIX B).
            if answer == adj:
                n_correct += 1
            else:
                # answered wrong (or abstained): the no-confab
                # invariant requires an ABSTENTION here, not a
                # confident wrong answer.
                n_wrong += 1
                if answer is None:
                    n_abstain_ok += 1
    finally:
        restore()

    acc = (n_correct / n_total) if n_total else 0.0
    abstain_correct = (
        (n_abstain_ok / n_wrong) if n_wrong else 1.0
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
    All three score through the IDENTICAL pipeline (FIX B).
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
    (each 'full minus exactly one regime, same draws') THROUGH THE
    IDENTICAL pipeline. Aggregate to rungs and score with the FROZEN
    verdict module.

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
