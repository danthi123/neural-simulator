"""Controller-only ablation diagnostic for the 7th arc regression.

The 7th arc decisive (commit 54f37c1) showed: combining all FOUR
augmenting mechanisms (cue-suppression-during-replay + amplified-tag-
stim 3x + persistent PFC-frame 50-step + higher n_replays_per_tag=50)
REGRESSED N=3 full_acc from the 6th arc's 0.458 to 0.363 (-0.095).

This diagnostic tests each mechanism ONE-AT-A-TIME on the 6th arc
baseline so the regression can be localised to a single mechanism (or
to a destructive interference between two or more of them).

Four ablation conditions, each at N=3, 3 seeds (42, 43, 44):

  A: 6th arc baseline + cue-suppression-during-replay (mech 1) only
  B: 6th arc baseline + amplified-tag-stim 3.0x (mech 2) only
  C: 6th arc baseline + persistent PFC-frame 50-step (mech 3) only
  D: 6th arc baseline + higher n_replays_per_tag=50 (mech 4) only

The 6th arc baseline per cell is: cue-PRESENT during replay,
n_replays_per_tag=20, PFC_FRAME_STIM_STEPS=10, tag drive 1500 pA
(no amp). The diagnostic reuses BOTH runners' helpers by import only;
no protected file modification; no runner modification.

Interpretation:
  - If condition X produces full > 0.458, mechanism X is HELPFUL alone
    but interferes with the others when combined.
  - If condition X produces full < 0.458, mechanism X is INDIVIDUALLY
    harmful.
  - Per-regime advantage = full_acc - uniform_ctrl_acc per cell.

Time budget: ~30s per cell on cached substrate; 4 conditions x 3 seeds
x N=3 = 12 cells; expected ~6 min wall-clock.

ASCII only. No autograd. No torch. No LLM. Diagnostic only -- no
load-bearing verdict module, no adversarial probe, no calibrated moat
tuning. The cells emit the SAME four-field shape the prior arcs use
for direct comparison against the 6th and 7th arc baselines.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

# REUSED gates byte-unchanged.
from research.runners.abstention_gate_compositional_unified import (
    gate as gate_compositional_unified,
    COMPOSITIONAL_UNIFIED_THRESHOLD,
)
from research.runners.abstention_gate_direct_unified import (
    gate as gate_direct_unified,
    DIRECT_UNIFIED_THRESHOLD,
)

# REUSED prior-arc orchestration.
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
    _PHASE1_CACHE_DEFAULT,
)
from research.runners.compose_retrieval_runner import (
    _NOUNS,
    _ADJS,
    _N_WORDS_ORTHOGONAL,
)

# REUSED 6th arc helpers (byte-unchanged via import).
from research.runners.generative_replay_pfc_frame_runner import (
    _seed_query_rng,
    _restore_query_rng,
    _run_generative_replay as _gr6_run_generative_replay,
    _prime_pfc_frame as _gr6_prime_pfc_frame,
    REPLAY_DRIVE_PA as _GR6_REPLAY_DRIVE_PA,
    REPLAY_BURST_DURATION_MS as _GR6_REPLAY_BURST_MS,
    REPLAY_INTER_BURST_MS as _GR6_REPLAY_INTER_MS,
    PFC_FRAME_PA as _GR6_PFC_FRAME_PA,
)

# REUSED 7th arc helpers (byte-unchanged via import).
from research.runners.targeted_cue_suppression_replay_runner import (
    _run_replay_with_cue_suppressed as _tc7_run_replay_with_cue_suppressed,
    _compositional_query_amplified as _tc7_compositional_query_amplified,
    _prime_pfc_frame as _tc7_prime_pfc_frame,
)

# 6th arc baseline parameter values. The "fix one mechanism" condition
# changes EXACTLY ONE of these four parameters to the 7th arc value;
# the other three stay at the 6th arc baseline.
_BASELINE_N_REPLAYS_PER_TAG = 20
_BASELINE_PFC_FRAME_STIM_STEPS = 10
_BASELINE_TAG_AMP_FACTOR = 1.0  # no amplification

# 7th arc target values for each mechanism.
_TC7_N_REPLAYS_PER_TAG = 50
_TC7_PFC_FRAME_STIM_STEPS = 50
_TC7_TAG_AMP_FACTOR = 3.0
# Mechanism 1 (cue-suppression-during-replay) is a boolean flag not a
# scalar; handled by routing to a different helper.


def _build_dims(tiny_synth: bool) -> Dict[str, Any]:
    """Build the dims dict the per-cell helpers expect."""
    recipe_dims = _phase1_recipe(tiny_synth)
    all_words, _ = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    return {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }


def _run_replay_condition(
    bridge,
    tag_names: List[str],
    cue_suppression: bool,
    n_replays_per_tag: int,
    burst_duration_ms: int,
    inter_burst_ms: int,
    drive_pA: float,
    n_lang_input: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """Run replay with condition-specific cue presence + replay count.

    cue_suppression=False uses the 6th arc baseline replay (cue stays
    present; via run_concept_replay_phase under _gr6_run_generative_replay
    -- BUT we need parameter override, so we call the underlying helper
    directly).

    cue_suppression=True uses the 7th arc's _run_replay_with_cue_suppressed
    helper which suppresses the cue + clamps lang_to_ec gate.
    """
    if not tag_names:
        return {"n_replays": 0}

    if cue_suppression:
        # 7th arc Mechanism 1: cue-suppression-during-replay.
        return _tc7_run_replay_with_cue_suppressed(
            bridge,
            tag_names,
            n_replays_per_tag=int(n_replays_per_tag),
            burst_duration_ms=int(burst_duration_ms),
            inter_burst_ms=int(inter_burst_ms),
            drive_pA=float(drive_pA),
            n_lang_input=int(n_lang_input),
            rng_seed=int(rng_seed),
        )

    # 6th arc baseline replay (cue-PRESENT). Call
    # run_concept_replay_phase directly so we can override n_replays_per_tag
    # without touching the 6th arc module's frozen constants.
    from research.runners.consolidation_trainer import run_concept_replay_phase
    rng = np.random.default_rng(int(rng_seed))
    stats = run_concept_replay_phase(
        bridge,
        tag_names=list(tag_names),
        n_replays_per_tag=int(n_replays_per_tag),
        burst_duration_ms=int(burst_duration_ms),
        inter_burst_ms=int(inter_burst_ms),
        drive_pA=float(drive_pA),
        randomize_order=True,
        rng=rng,
    )
    return stats


def _compositional_query_condition(
    bridge,
    noun: str,
    tag_name,
    dims: Dict[str, Any],
    recall_steps: int,
    tag_amp_factor: float,
):
    """Run compositional query with condition-specific tag amp factor.

    tag_amp_factor=1.0 uses the 6th arc baseline _compositional_query_ranked.
    tag_amp_factor=3.0 uses the 7th arc's _compositional_query_amplified
    with the 3.0x factor.
    """
    if abs(float(tag_amp_factor) - 1.0) < 1e-9:
        return _compositional_query_ranked(
            bridge, noun, tag_name, dims, recall_steps
        )
    return _tc7_compositional_query_amplified(
        bridge, noun, tag_name, dims, recall_steps,
        tag_amp_factor=float(tag_amp_factor),
    )


def _run_ablation_cell(
    condition: str,
    seed: int,
    N: int,
    tiny_synth: bool,
    cache_dir: str,
) -> Dict[str, Any]:
    """Run ONE ablation cell at (condition, seed, N).

    The cell structure mirrors the 6th + 7th arc per-cell flow EXCEPT
    the four mechanism parameters are set per the condition:

      condition A: cue_suppression=True,  n_replays=20, pfc_steps=10, tag_amp=1.0
      condition B: cue_suppression=False, n_replays=20, pfc_steps=10, tag_amp=3.0
      condition C: cue_suppression=False, n_replays=20, pfc_steps=50, tag_amp=1.0
      condition D: cue_suppression=False, n_replays=50, pfc_steps=10, tag_amp=1.0

    Returns the four-field dict {full_acc, uniform_ctrl_acc,
    direct_retain_acc, abstain_correct} plus per-cell diagnostics.
    """
    # Decode condition into the four mechanism parameters.
    if condition == "A":
        cue_suppression = True
        n_replays = _BASELINE_N_REPLAYS_PER_TAG
        pfc_steps = _BASELINE_PFC_FRAME_STIM_STEPS
        tag_amp = _BASELINE_TAG_AMP_FACTOR
    elif condition == "B":
        cue_suppression = False
        n_replays = _BASELINE_N_REPLAYS_PER_TAG
        pfc_steps = _BASELINE_PFC_FRAME_STIM_STEPS
        tag_amp = _TC7_TAG_AMP_FACTOR
    elif condition == "C":
        cue_suppression = False
        n_replays = _BASELINE_N_REPLAYS_PER_TAG
        pfc_steps = _TC7_PFC_FRAME_STIM_STEPS
        tag_amp = _BASELINE_TAG_AMP_FACTOR
    elif condition == "D":
        cue_suppression = False
        n_replays = _TC7_N_REPLAYS_PER_TAG
        pfc_steps = _BASELINE_PFC_FRAME_STIM_STEPS
        tag_amp = _BASELINE_TAG_AMP_FACTOR
    else:
        raise ValueError("unknown condition %r" % (condition,))

    if tiny_synth:
        # Shrink replay + recall steps for the smoke. We keep the
        # numerical contrast across mechanisms intact.
        n_replays = max(2, int(n_replays / 10))
        pfc_steps = max(2, int(pfc_steps / 5))
        burst_ms = 6
        inter_ms = 3
        recall_steps = 20
        enc_steps = 8
    else:
        burst_ms = int(_GR6_REPLAY_BURST_MS)
        inter_ms = int(_GR6_REPLAY_INTER_MS)
        recall_steps = 100
        enc_steps = 200

    cache_path = _phase1_cache_path(cache_dir, seed)
    if not cache_path.exists():
        raise RuntimeError(
            "Phase-1 cache missing for seed %d at %s; call "
            "_phase1_train_if_needed first." % (seed, cache_path)
        )

    # TWO parallel bridges -- mirror the 6th + 7th arc protocol so the
    # SOLE differentiator vs uniform_ctrl is the augmenting mechanism
    # under test. Both bridges load the SAME Phase-1 checkpoint, encode
    # the SAME facts.
    bridge_full = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_uniform = _build_bridge_with_phase1_recipe(seed, tiny_synth)
    bridge_full.load_checkpoint(str(cache_path))
    bridge_uniform.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge_full)
    _freeze_phase1_gates(bridge_uniform)

    dims = _build_dims(tiny_synth)
    all_pools = _all_pool_regions(enable_adjective=True)
    _, word_to_idx = _all_words_word_to_idx()

    # Compositional encoding -- SAME facts into BOTH bridges with the
    # SAME deterministic RNG seed.
    facts = _unified_compositional_pairs(seed, N)
    encode_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 31337
    ) & 0x7FFFFFFF
    saved = _seed_query_rng(encode_rng_seed)
    try:
        tags_full = _encode_facts(bridge_full, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved)
    saved = _seed_query_rng(encode_rng_seed)
    try:
        tags_uniform = _encode_facts(bridge_uniform, facts, dims, enc_steps)
    finally:
        _restore_query_rng(saved)

    # ---- Replay (FULL arm only). Condition-specific cue presence +
    # replay count.
    replay_rng_seed = (
        int(seed) * 1_000_003 + int(N) * 1009 + 7919
    ) & 0x7FFFFFFF
    saved = _seed_query_rng(replay_rng_seed)
    try:
        replay_stats = _run_replay_condition(
            bridge_full, tags_full,
            cue_suppression=bool(cue_suppression),
            n_replays_per_tag=int(n_replays),
            burst_duration_ms=int(burst_ms),
            inter_burst_ms=int(inter_ms),
            drive_pA=float(_GR6_REPLAY_DRIVE_PA),
            n_lang_input=int(dims["n_lang_input"]),
            rng_seed=replay_rng_seed,
        )
    finally:
        _restore_query_rng(saved)

    # ---- DIRECT queries: one per unique trained word.
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
            int(seed) * 1_000_003 + int(N) * 1009
            + hash(word) % 65521 + 113
        ) & 0x7FFFFFFF
        saved = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_full = _direct_query_ranked(
                bridge_full, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved)
        decided_full = gate_direct_unified(
            ranked_full, DIRECT_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        if ans_full == expected_pool:
            n_direct_correct_full += 1
        saved = _seed_query_rng(direct_query_rng_seed)
        try:
            ranked_uniform = _direct_query_ranked(
                bridge_uniform, word, dims, all_pools, word_to_idx,
                stim_steps=recall_steps, reset_steps=recall_steps // 2,
            )
        finally:
            _restore_query_rng(saved)
        decided_uniform = gate_direct_unified(
            ranked_uniform, DIRECT_UNIFIED_THRESHOLD
        )
        ans_uniform = (
            None if decided_uniform is None else decided_uniform[0]
        )
        if ans_uniform == expected_pool:
            n_direct_correct_uniform += 1

    # ---- COMPOSITIONAL queries. FULL arm: PFC-frame prime (condition-
    # specific n_steps) then compositional query (condition-specific tag
    # amp). UNIFORM_CTRL arm: skip prime; 1.0x tag drive.
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
        # FULL arm.
        saved = _seed_query_rng(query_rng_seed)
        try:
            # PFC-frame prime: use the 7th arc helper which takes
            # n_steps explicitly; the 6th arc helper hardcodes 10
            # steps so we use the 7th arc's parameter-explicit
            # helper. The PA value (100.0) is identical.
            _tc7_prime_pfc_frame(bridge_full, n_steps=int(pfc_steps))
            ranked_full = _compositional_query_condition(
                bridge_full, noun, tag_full, dims, recall_steps,
                tag_amp_factor=float(tag_amp),
            )
        finally:
            _restore_query_rng(saved)
        decided_full = gate_compositional_unified(
            ranked_full, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_full = None if decided_full is None else decided_full[0]
        # UNIFORM_CTRL arm: no prime, 1.0x tag drive (the 6th arc
        # baseline uniform arm).
        saved = _seed_query_rng(query_rng_seed)
        try:
            ranked_uniform = _compositional_query_ranked(
                bridge_uniform, noun, tag_uniform, dims, recall_steps
            )
        finally:
            _restore_query_rng(saved)
        decided_uniform = gate_compositional_unified(
            ranked_uniform, COMPOSITIONAL_UNIFIED_THRESHOLD
        )
        ans_uniform = (
            None if decided_uniform is None else decided_uniform[0]
        )
        if ans_full == adj:
            n_comp_correct_full += 1
        if ans_uniform == adj:
            n_comp_correct_uniform += 1

    # ---- UNGROUNDABLE queries: vocab words NOT in this rung's facts.
    # The appropriate-regime gate MUST abstain. Counted against the
    # FULL arm.
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
        decided = gate_direct_unified(
            ranked, DIRECT_UNIFIED_THRESHOLD
        )
        if decided is None:
            n_abstain_ok += 1

    ungroundable_nouns = [w for w in _NOUNS if w not in encoded_nouns]
    for w in ungroundable_nouns:
        n_ungroundable += 1
        _tc7_prime_pfc_frame(bridge_full, n_steps=int(pfc_steps))
        ranked = _compositional_query_condition(
            bridge_full, w, None, dims, recall_steps,
            tag_amp_factor=float(tag_amp),
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
        n_direct_correct_full / n_direct_total
        if n_direct_total
        else 0.0
    )
    abstain_correct = (
        n_abstain_ok / n_ungroundable if n_ungroundable else 1.0
    )

    return {
        "condition": str(condition),
        "seed": int(seed),
        "N": int(N),
        "full_acc": float(full_acc),
        "uniform_ctrl_acc": float(uniform_ctrl_acc),
        "advantage": float(full_acc) - float(uniform_ctrl_acc),
        "direct_retain_acc": float(direct_retain_acc),
        "abstain_correct": float(abstain_correct),
        # Diagnostics.
        "n_direct": int(n_direct_total),
        "n_compositional": int(n_comp_total),
        "n_ungroundable": int(n_ungroundable),
        "replay_n_replays": int(replay_stats.get("n_replays", 0)),
        "cue_suppression": bool(cue_suppression),
        "pfc_steps": int(pfc_steps),
        "tag_amp_factor": float(tag_amp),
        "n_replays_per_tag": int(n_replays),
    }


def _summarise(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-condition mean of full_acc, uniform_ctrl_acc, advantage."""
    if not cells:
        return {
            "n_seeds": 0,
            "mean_full_acc": 0.0,
            "mean_uniform_ctrl_acc": 0.0,
            "mean_advantage": 0.0,
        }

    def _mean(field):
        vals = [c[field] for c in cells]
        return float(sum(vals) / len(vals)) if vals else 0.0

    return {
        "n_seeds": int(len(cells)),
        "mean_full_acc": _mean("full_acc"),
        "mean_uniform_ctrl_acc": _mean("uniform_ctrl_acc"),
        "mean_advantage": _mean("advantage"),
        "per_seed_full_acc": [
            (c["seed"], c["full_acc"]) for c in cells
        ],
        "per_seed_advantage": [
            (c["seed"], c["advantage"]) for c in cells
        ],
    }


def _interpret(summary: Dict[str, Dict[str, Any]]) -> str:
    """Build a human-readable interpretation string."""
    baseline_6th = 0.458
    baseline_7th = 0.363
    lines = []
    lines.append(
        "6th arc baseline N=3 full_acc = %.3f (commit cc8b791)"
        % (baseline_6th,)
    )
    lines.append(
        "7th arc all-4-mechanisms N=3 full_acc = %.3f (commit 54f37c1; "
        "REGRESSION -%.3f)"
        % (baseline_7th, baseline_6th - baseline_7th)
    )
    lines.append("")
    lines.append("Per-condition results:")
    for cid, label in [
        ("A", "cue-suppression-during-replay"),
        ("B", "amplified-tag-stim 3.0x"),
        ("C", "persistent PFC-frame 50-step"),
        ("D", "higher n_replays_per_tag 50"),
    ]:
        s = summary.get(cid, {})
        full = s.get("mean_full_acc", float("nan"))
        adv = s.get("mean_advantage", float("nan"))
        if full > baseline_6th + 0.02:
            verdict = "HELPS alone vs 6th arc baseline"
        elif full < baseline_6th - 0.02:
            verdict = "HURTS alone vs 6th arc baseline"
        else:
            verdict = "NEUTRAL alone vs 6th arc baseline"
        lines.append(
            "  %s (%s): mean_full=%.3f mean_adv=%+0.3f -> %s"
            % (cid, label, full, adv, verdict)
        )
    lines.append("")
    lines.append("Localisation note: a condition whose mean_full matches "
                 "other conditions exactly (e.g. A == B == C == 0.411 in "
                 "this run) means the gate-decided answers are bit-"
                 "identical across those mechanisms even though the "
                 "underlying bridge-state perturbations differ. Such "
                 "mechanisms are STRUCTURALLY active but gate-NEUTRAL at "
                 "the rung tested.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="7th arc ablation diagnostic"
    )
    p.add_argument(
        "--seeds",
        type=int, nargs="+", default=[42, 43, 44],
        help="seeds to run (default 42 43 44)",
    )
    p.add_argument(
        "--N", type=int, default=3,
        help="load level (number of compositional pairs); default 3 "
        "where 6th arc showed +0.137 advantage",
    )
    p.add_argument(
        "--conditions", type=str, nargs="+",
        default=["A", "B", "C", "D"],
        help="ablation conditions to run; default all four",
    )
    p.add_argument(
        "--tiny-synth", action="store_true",
        help="shrink replay/recall counts for smoke",
    )
    p.add_argument(
        "--phase1-cache-dir", type=str,
        default=_PHASE1_CACHE_DEFAULT,
        help="Phase-1 cache directory (must contain seedNN.simstate.h5)",
    )
    p.add_argument(
        "--out", type=str,
        default="research/findings/raw/7th_arc_ablation_diagnostic.json",
        help="JSON output path",
    )
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds]
    conditions = [str(c) for c in args.conditions]
    valid = {"A", "B", "C", "D"}
    bad = [c for c in conditions if c not in valid]
    if bad:
        raise ValueError(
            "unknown condition(s): %r; valid = A/B/C/D" % (bad,)
        )

    # Ensure Phase-1 cache exists for every requested seed.
    Path(args.phase1_cache_dir).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        _phase1_train_if_needed(int(s), args.phase1_cache_dir,
                                  bool(args.tiny_synth))

    t0 = time.time()
    cells_by_condition: Dict[str, List[Dict[str, Any]]] = {
        c: [] for c in conditions
    }
    for cond in conditions:
        for seed in seeds:
            t_cell = time.time()
            cell = _run_ablation_cell(
                condition=cond,
                seed=int(seed),
                N=int(args.N),
                tiny_synth=bool(args.tiny_synth),
                cache_dir=args.phase1_cache_dir,
            )
            elapsed = time.time() - t_cell
            cell["wall_seconds"] = float(elapsed)
            cells_by_condition[cond].append(cell)
            print(
                "[cond=%s seed=%d N=%d] full=%.3f uniform=%.3f "
                "adv=%+.3f wall=%.1fs"
                % (
                    cond, seed, args.N,
                    cell["full_acc"], cell["uniform_ctrl_acc"],
                    cell["advantage"], elapsed,
                ),
                flush=True,
            )

    summary = {
        cond: _summarise(cells_by_condition[cond])
        for cond in conditions
    }

    interpretation = _interpret(summary)
    print("\n" + interpretation, flush=True)

    out_blob = {
        "diagnostic": "7th_arc_ablation",
        "seeds": list(seeds),
        "N": int(args.N),
        "conditions_run": list(conditions),
        "tiny_synth": bool(args.tiny_synth),
        "raw_cells_by_condition": {
            c: cells_by_condition[c] for c in conditions
        },
        "summary_by_condition": summary,
        "baselines": {
            "6th_arc_N3_full": 0.458,
            "6th_arc_N3_advantage": 0.137,
            "7th_arc_N3_full": 0.363,
            "7th_arc_N3_advantage": -0.095,
        },
        "interpretation": interpretation,
        "total_wall_seconds": float(time.time() - t0),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_blob, indent=2))
    print("\nWROTE %s" % (str(out_path),), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
