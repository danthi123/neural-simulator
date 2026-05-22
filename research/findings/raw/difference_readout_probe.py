"""Cheap-first falsification probe: predictive-coding difference readout.

Design: docs/plans/2026-05-21-predictive-coding-difference-readout-compositional-design.md

The eight-architecture compositional series localized the blocker: when
a noun is cued to recall its bound attribute, the cue's broad input
drive dominates the engram tag's selective drive at a shared raw
readout. All eight arcs overlaid dynamics on a FIXED readout
(score = cue-alone-pattern + tag-alone-pattern, summed). This probe
tests the readout-computation hypothesis: combine the two patterns by
DIFFERENCE instead of SUM.

  raw readout   (arcs 1-8): score[w] = cons_rate[w] + hip_rate[w]
  difference    (this probe): score[w] = hip_rate[w] - cons_rate[w]

where cons = lang_output during cue-noun input alone (the cue's broad
pathway, the "prediction"/baseline) and hip = lang_output during
engram-tag stimulation alone (the tag reactivates noun+adj pool
neurons). The difference removes the cue-pathway component shared by
both measurements, leaving the attribute the tag ADDS -- the recalled
fact as a predictive-coding deviation (Rao & Ballard 1999; Carandini &
Heeger 2012 normalization). The cue's context is preserved (the noun
is in the tag); only its raw-drive domination is cancelled.

Note on the design-to-implementation refinement: the design doc framed
the baseline as "cue alone" and the bound response as "cue + tag
together". The implementation uses the two measurements the validated
helpers already provide byte-unchanged -- cue-alone (cons) and
tag-alone (hip) -- and takes hip - cons. This is the same
predictive-coding difference (isolate the tag's contribution over the
cue baseline) in the form that needs zero new measurement primitive.

PRE-REGISTERED DECISION RULE (fixed; never tuned):
- SIGNAL: the difference readout scores strictly MORE correct than the
  raw readout on the groundable queries AND the permuted-tag control
  behaves (difference points to the STIMULATED tag's attribute, not
  the cue's own bound attribute) on a majority of permuted queries.
  -> proceed to the full pre-registered three-state arc.
- NEGATIVE: difference does not beat raw, OR the permuted control
  fails. -> honest fast negative; the design line reaches terminal
  closure.

Reuse-by-import only. No protected/frozen/moat module modified. No
autograd. Controller-only; single seed 42; reuse the cached 200-event
unified substrate (no retraining).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _freeze_phase1_gates,
    _encode_facts,
    _all_words_word_to_idx,
)
from research.runners.compose_retrieval_runner import (
    _N_WORDS_ORTHOGONAL,
    _ranked_from_pattern,
)
from research.runners.compose_concept_engram import (
    lang_output_pattern_during_stim,
    lang_output_pattern_during_input,
)

SEED = 42
CACHE_DIR = "research/findings/raw/unified_per_regime/phase1"  # 200-event cache
RECALL_STEPS = 100

# Pre-registered groundable (noun, adjective) bindings. Four pairs --
# the vocabulary has exactly four nouns; the design doc said five, the
# implementation uses the four the vocab supports (honest refinement).
FACTS = [("apple", "big"), ("river", "small"), ("dog", "hot"), ("cat", "cold")]


def _rate_dict(pattern, n_lo, dims, exclude):
    """Per-word raw firing-rate confidence dict via the byte-unchanged
    _ranked_from_pattern helper."""
    ranked = _ranked_from_pattern(pattern, n_lo, dims, exclude=exclude)
    return {w: float(r) for (w, r, _t) in ranked}


def _top(score):
    """Top (word, score) of a score dict, descending."""
    items = sorted(score.items(), key=lambda kv: -kv[1])
    return items[0] if items else (None, 0.0)


def main():
    print("=== Difference-readout cheap-first probe ===")
    print(f"seed={SEED}; cache={CACHE_DIR}; facts={FACTS}")

    bridge = _build_bridge_with_phase1_recipe(SEED, tiny_synth=False)
    cache_path = _phase1_cache_path(CACHE_DIR, SEED)
    print(f"Loading {cache_path}")
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)

    recipe_dims = _phase1_recipe(False)
    all_words, _word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))
    dims = {
        "n_lang_input": int(recipe_dims["n_lang_input"]),
        "n_per_pool": int(recipe_dims["n_per_pool"]),
        "n_fs_per_pool": int(recipe_dims["n_fs_per_pool"]),
        "sparsity": 0.05,
        "dt_ms": 0.5,
        "n_words_for_orthogonal": int(n_words_for_orthogonal),
    }

    # Encode the four compositional bindings as engram tags (reused
    # _encode_facts -> opaque tag names ep_0..ep_3).
    tags = _encode_facts(bridge, FACTS, dims, encoding_steps=200)
    print(f"Encoded tags: {tags}")

    # Pre-compute the cue-alone pattern for each noun once (reused
    # across the groundable + permuted queries for that cue).
    cons_rate = {}
    for noun, _adj in FACTS:
        pat, n_lo = lang_output_pattern_during_input(
            bridge, noun,
            n_lang_input=int(dims["n_lang_input"]),
            sparsity=float(dims["sparsity"]),
            n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
            stim_steps=RECALL_STEPS,
        )
        cons_rate[noun] = _rate_dict(pat, n_lo, dims, exclude=noun)

    # Pre-compute the tag-alone pattern for each tag once.
    hip_rate = {}
    for i, tag in enumerate(tags):
        noun = FACTS[i][0]
        pat, n_lo = lang_output_pattern_during_stim(
            bridge, tag, drive_pA=1500.0, stim_steps=RECALL_STEPS,
        )
        hip_rate[tag] = _rate_dict(pat, n_lo, dims, exclude=noun)

    # ---- Groundable queries: cue N_i, stim tag_i. ----
    groundable = []
    raw_correct = 0
    diff_correct = 0
    for i, (noun, adj) in enumerate(FACTS):
        tag = tags[i]
        cons = cons_rate[noun]
        hip = hip_rate[tag]
        words = set(cons) | set(hip)
        raw_score = {w: cons.get(w, 0.0) + hip.get(w, 0.0) for w in words}
        diff_score = {w: hip.get(w, 0.0) - cons.get(w, 0.0) for w in words}
        raw_top, raw_val = _top(raw_score)
        diff_top, diff_val = _top(diff_score)
        raw_ok = (raw_top == adj)
        diff_ok = (diff_top == adj)
        raw_correct += int(raw_ok)
        diff_correct += int(diff_ok)
        groundable.append({
            "cue": noun, "target_adj": adj, "tag": tag,
            "raw_top": raw_top, "raw_top_val": raw_val, "raw_ok": raw_ok,
            "diff_top": diff_top, "diff_top_val": diff_val, "diff_ok": diff_ok,
        })
        print(f"  groundable {noun}->{adj}: raw_top={raw_top}({raw_val:.1f}) "
              f"{'OK' if raw_ok else 'XX'} | diff_top={diff_top}({diff_val:.1f}) "
              f"{'OK' if diff_ok else 'XX'}")

    # ---- Permuted-tag control: cue N_i, stim tag_{i+1}. The difference
    # readout MUST point to the STIMULATED tag's adjective. ----
    permuted = []
    permuted_correct = 0
    for i, (noun, _adj) in enumerate(FACTS):
        j = (i + 1) % len(FACTS)
        perm_tag = tags[j]
        perm_adj = FACTS[j][1]
        cons = cons_rate[noun]
        hip = hip_rate[perm_tag]
        words = set(cons) | set(hip)
        diff_score = {w: hip.get(w, 0.0) - cons.get(w, 0.0) for w in words}
        diff_top, diff_val = _top(diff_score)
        ctrl_ok = (diff_top == perm_adj)
        permuted_correct += int(ctrl_ok)
        permuted.append({
            "cue": noun, "stimulated_tag": perm_tag,
            "stimulated_tag_adj": perm_adj,
            "diff_top": diff_top, "diff_top_val": diff_val,
            "control_ok": ctrl_ok,
        })
        print(f"  permuted  {noun}+tag({FACTS[j][0]},{perm_adj}): "
              f"diff_top={diff_top}({diff_val:.1f}) "
              f"{'OK (points to stimulated tag)' if ctrl_ok else 'XX'}")

    n = len(FACTS)
    print(f"\n=== RESULT (seed {SEED}; {n} groundable + {n} permuted) ===")
    print(f"  raw readout (arcs 1-8 sum):   {raw_correct}/{n} correct")
    print(f"  difference readout (hip-cons): {diff_correct}/{n} correct")
    print(f"  permuted-tag control:          {permuted_correct}/{n} "
          f"point to the stimulated tag")

    # Pre-registered decision rule.
    diff_beats_raw = diff_correct > raw_correct
    control_holds = permuted_correct > n // 2
    if diff_beats_raw and control_holds:
        verdict = "SIGNAL"
        print("  --> SIGNAL: difference readout beats raw AND permuted "
              "control holds. Proceed to the full pre-registered arc.")
    else:
        verdict = "NEGATIVE"
        reason = []
        if not diff_beats_raw:
            reason.append("difference did not beat raw")
        if not control_holds:
            reason.append("permuted control failed")
        print(f"  --> NEGATIVE ({'; '.join(reason)}). Honest fast "
              "negative; design line reaches terminal closure.")

    out = {
        "seed": SEED, "cache_dir": CACHE_DIR, "facts": FACTS, "tags": tags,
        "recall_steps": RECALL_STEPS,
        "groundable": groundable, "permuted_control": permuted,
        "raw_correct": raw_correct, "diff_correct": diff_correct,
        "permuted_correct": permuted_correct, "n": n,
        "diff_beats_raw": diff_beats_raw, "control_holds": control_holds,
        "verdict": verdict,
    }
    out_path = "research/findings/raw/difference_readout_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
