"""Diagnostic probe: compare measure_pool_firing separation on the
pure v14/v16 substrate vs the unified substrate.

For each substrate's cached seed-42 checkpoint, load it into the
matching architecture, run measure_pool_firing on the SAME 16-word
vocabulary the substrate was trained on (motor + nouns + verbs +
adjectives), using the SAME 16-word orthogonal-code layout the
training and the unified calibration both use. Report per-substrate
groundable (target-pool firing rate) vs best off-target rate
distribution -- on the full 16-word vocab AND on the 12-non-motor
subset (which mirrors the unified calibration's scope).

NOTE on a methodology-bug fix:
A prior pass of this diagnostic used 12-word orthogonal codes
(n_words_for_orthogonal=12, word_to_idx over nouns+verbs+adjs only).
That produced different lang_input drive patterns than what either
substrate was trained on (training uses n_cues=16 with motor-first
ordering, see concept_pool_demo._all_words_ordered and
unified_per_regime_monitor_runner._all_words_word_to_idx). Catching
that subtle mismatch BEFORE propagating a misleading
"unified-is-worse" reading IS the discipline working. This pass
matches the training/calibration canon byte-for-byte.

This is a controller-only diagnostic; not a decisive run; output
is just diagnostic numbers, not a verdict.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

# Add repo root to sys.path so `research.runners.*` imports work
# when invoked directly via `python research/findings/raw/...`
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    # Reuse the runner's vocab + readout helpers byte-unchanged.
    import research.runners.concept_pool_demo as cpd
    from research.runners.compose_retrieval_runner import _NOUNS, _VERBS, _ADJS  # noqa: E501

    # === Vocabulary canon: must match training + unified calibration ===
    # Per concept_pool_demo.py:898-904 and unified_per_regime_monitor_runner.py:690-703
    # the canonical 16-word ordering is motor-first then nouns, verbs, adjectives.
    _MOTOR_WORDS = ["north", "east", "south", "west"]

    all_words = (
        list(_MOTOR_WORDS)
        + list(_NOUNS)
        + list(_VERBS)
        + list(_ADJS)
    )
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    n_words_for_orthogonal = len(all_words)  # 16

    # Pool name lookup for ALL 16 words (motor included)
    pool_for_word = {}
    for w in _MOTOR_WORDS:
        pool_for_word[w] = f"motor_{w[0].upper()}"  # motor_N/E/S/W
    for w in _NOUNS:
        pool_for_word[w] = f"noun_pool_{w.upper()}"
    for w in _VERBS:
        pool_for_word[w] = f"verb_pool_{w.upper()}"
    for w in _ADJS:
        pool_for_word[w] = f"adjective_pool_{w.upper()}"

    all_pool_regions = [pool_for_word[w] for w in all_words]
    non_motor_words = list(_NOUNS) + list(_VERBS) + list(_ADJS)

    SEED = 42
    PURE_V14_CHECKPOINT = (
        "research/findings/raw/g11_bg/concept_pool_demo/"
        "seed42_v14.simstate.h5"
    )
    UNIFIED_CHECKPOINT = (
        "research/findings/raw/unified_per_regime/phase1/"
        "seed42.simstate.h5"
    )

    results = {
        "method": (
            "16-word canonical drive (matches training + unified calibration). "
            "Each substrate loaded from cached seed-42 Phase-1 checkpoint. "
            "All plasticity gates frozen. For each word: drive lang_input via "
            "orthogonal_drive_pattern(cue_idx=word_to_idx[w], n_cues=16, "
            "sparsity=0.05) for 100 stim steps; measure pool firing rates over "
            "the 16 target pools (4 motor + 4 noun + 4 verb + 4 adj). target_rate "
            "= rate of the matching pool; best_off = max over the OTHER 15 pools."
        ),
        "fix_note": (
            "Prior pass used n_cues=12 + word_to_idx over non-motor words only, "
            "producing different orthogonal-stride layouts than what either "
            "substrate was trained on. Both substrates falsely 'failed' under "
            "that broken protocol. Fixed: 16-word canonical layout matching "
            "concept_pool_demo + unified_per_regime_monitor_runner exactly."
        ),
    }

    def _per_word_separation(bridge, words_to_test):
        """Probe each word: target-pool rate vs best off-target rate."""
        groundable = []
        ungroundable = []
        per_word = []
        for w in words_to_test:
            rates = cpd.measure_pool_firing(
                bridge, w, all_pool_regions,
                stim_steps=100, reset_steps=50, drive_pA=200.0,
                sparsity=0.05, n_lang_input=2048,
                orthogonal_codes=True,
                n_words_for_orthogonal=n_words_for_orthogonal,
                word_to_idx=word_to_idx,
            )
            target_pool = pool_for_word[w]
            target_rate = rates.get(target_pool, 0.0)
            off_target_rates = [
                r for p, r in rates.items() if p != target_pool
            ]
            best_off = max(off_target_rates) if off_target_rates else 0.0
            groundable.append(target_rate)
            ungroundable.append(best_off)
            ok = "OK " if target_rate > best_off else "INV"
            per_word.append({
                "word": w,
                "target_pool": target_pool,
                "target_rate": float(target_rate),
                "best_off_target_rate": float(best_off),
                "correct_direction": bool(target_rate > best_off),
            })
            print(
                f"  {ok} {w:>12}: target {target_pool} rate={target_rate:.3f} "
                f"vs best off-target rate={best_off:.3f}"
            )
        return groundable, ungroundable, per_word

    def _summary(groundable, ungroundable, words):
        return {
            "n_words": len(words),
            "groundable_median": float(np.median(groundable)),
            "ungroundable_median": float(np.median(ungroundable)),
            "groundable_mean": float(np.mean(groundable)),
            "ungroundable_mean": float(np.mean(ungroundable)),
            "n_correct_direction": int(sum(
                1 for g, u in zip(groundable, ungroundable) if g > u
            )),
            "groundable": [float(x) for x in groundable],
            "ungroundable": [float(x) for x in ungroundable],
        }

    def _freeze_all_gates(bridge):
        if not hasattr(bridge, "set_plasticity_gate"):
            return
        candidate_gates = [
            "language_input_to_motor",
            "language_input_to_noun_pool",
            "language_input_to_verb_pool",
            "language_input_to_adjective_pool",
            "motor_pool_to_language_output",
            "noun_pool_to_language_output",
            "verb_pool_to_language_output",
            "adjective_pool_to_language_output",
            "verb_to_motor_direct",
            "cross_pool_concept",
            "lang_to_ec",
            "ec_to_dg",
            "dg_to_ca3",
            "ca3_to_ca1",
            "ca1_to_motor",
            "ca1_to_lang_out",
            "ca3_swr_burst",
        ]
        for gate in candidate_gates:
            try:
                bridge.set_plasticity_gate(gate, 0.0)
            except (KeyError, Exception):
                pass

    # === Pure v14 substrate ===
    print("\n=== PURE v14 substrate (cpd.build_concept_bridge; no hippocampus) ===")
    bridge = cpd.build_concept_bridge(
        seed=SEED,
        n_lang_input=2048,
        n_per_pool=200,
        n_fs_per_pool=24,
        enable_adjective=True,
        weak_dynamics=True,
        enable_direct_verb_to_motor=True,
        verbose=False,
    )
    bridge.load_checkpoint(PURE_V14_CHECKPOINT)
    _freeze_all_gates(bridge)

    print("\n-- pure v14: all 16 words --")
    g_v14_all, u_v14_all, per_word_v14_all = _per_word_separation(bridge, all_words)
    print("\n-- pure v14: 12 non-motor words (calibration scope) --")
    g_v14_nm, u_v14_nm, per_word_v14_nm = _per_word_separation(bridge, non_motor_words)

    results["pure_v14_full16"] = _summary(g_v14_all, u_v14_all, all_words)
    results["pure_v14_full16"]["per_word"] = per_word_v14_all
    results["pure_v14_non_motor12"] = _summary(g_v14_nm, u_v14_nm, non_motor_words)
    results["pure_v14_non_motor12"]["per_word"] = per_word_v14_nm

    print(
        f"  --> pure v14 (16 words): "
        f"groundable_median={results['pure_v14_full16']['groundable_median']:.3f} "
        f"vs ungroundable_median={results['pure_v14_full16']['ungroundable_median']:.3f}; "
        f"{results['pure_v14_full16']['n_correct_direction']}/16 correct direction"
    )
    print(
        f"  --> pure v14 (12 non-motor): "
        f"groundable_median={results['pure_v14_non_motor12']['groundable_median']:.3f} "
        f"vs ungroundable_median={results['pure_v14_non_motor12']['ungroundable_median']:.3f}; "
        f"{results['pure_v14_non_motor12']['n_correct_direction']}/12 correct direction"
    )

    del bridge

    # === Unified substrate ===
    print(
        "\n=== UNIFIED substrate (build_biological_brain_regions; "
        "hippocampus + dlpfc + concept pools) ==="
    )
    from research.runners.unified_per_regime_monitor_runner import (
        _build_bridge_with_phase1_recipe,
    )
    bridge2 = _build_bridge_with_phase1_recipe(seed=SEED, tiny_synth=False)
    bridge2.load_checkpoint(UNIFIED_CHECKPOINT)
    _freeze_all_gates(bridge2)

    print("\n-- unified: all 16 words --")
    g_u_all, u_u_all, per_word_u_all = _per_word_separation(bridge2, all_words)
    print("\n-- unified: 12 non-motor words (calibration scope) --")
    g_u_nm, u_u_nm, per_word_u_nm = _per_word_separation(bridge2, non_motor_words)

    results["unified_full16"] = _summary(g_u_all, u_u_all, all_words)
    results["unified_full16"]["per_word"] = per_word_u_all
    results["unified_non_motor12"] = _summary(g_u_nm, u_u_nm, non_motor_words)
    results["unified_non_motor12"]["per_word"] = per_word_u_nm

    print(
        f"  --> unified (16 words): "
        f"groundable_median={results['unified_full16']['groundable_median']:.3f} "
        f"vs ungroundable_median={results['unified_full16']['ungroundable_median']:.3f}; "
        f"{results['unified_full16']['n_correct_direction']}/16 correct direction"
    )
    print(
        f"  --> unified (12 non-motor): "
        f"groundable_median={results['unified_non_motor12']['groundable_median']:.3f} "
        f"vs ungroundable_median={results['unified_non_motor12']['ungroundable_median']:.3f}; "
        f"{results['unified_non_motor12']['n_correct_direction']}/12 correct direction"
    )

    print("\n=== COMPARISON ===")
    print(json.dumps({
        k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_word"})
        for k, v in results.items()
    }, indent=2))

    out = "research/findings/raw/unified_DIAGNOSTIC_pure_vs_unified.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
