"""Catastrophic-forgetting probe across training-event regimes.

Tests the CLS-theory prediction: schema-consolidated (DIRECT-FAVORED
800ev) substrates should resist interfering new vocabulary better
than episodic-flexible (COMPOSITIONAL-FAVORED 200ev) substrates.

Protocol (single-seed cheap-first; seed 42 on unified substrate):
1. Load original 16-word-trained substrate at 200ev or 800ev cache.
2. Run pre-interference 16-word direct binding diagnostic.
3. Open Phase-1 plasticity gates (restore plasticity).
4. Train 4 INTERFERING rebindings (50 events each = 200 total events;
   ~10% of original training intensity):
     apple -> motor_W       (was noun_pool_APPLE)
     go    -> adjective_pool_BIG (was verb_pool_GO)
     big   -> verb_pool_GO  (was adjective_pool_BIG)
     north -> noun_pool_APPLE   (was motor_N)
   This is a cross-category swap pattern that touches all 4 pool kinds.
5. Save post-interference cache.
6. Re-run 16-word direct binding diagnostic.
7. Compute forgetting % per original-vocab category.

Decision rule (pre-registered, fixed before run):
- If 200ev forgets substantially more than 800ev (e.g.,
  forgetting_200ev - forgetting_800ev >= 10 pp on the original-vocab
  retention): CLS schema-resists-interference prediction VALIDATED at
  the substrate level; expand multi-seed.
- If 200ev and 800ev forget similarly: substrate's training-event
  regimes do NOT correspond to interference-resistance regimes;
  honest report; refines the 4-regime characterization.
- If 800ev forgets MORE than 200ev: opposite-direction finding;
  warrants careful diagnostic of WHY.

Reuse: cpd.train_word_to_pool byte-unchanged for interference training;
test_one_checkpoint byte-unchanged for the diagnostics.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import research.runners.concept_pool_demo as cpd
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_train_kwargs,
    _freeze_phase1_gates,
)

from importlib import util as _import_util
_diag_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _import_util.spec_from_file_location("_db", _diag_path)
_db = _import_util.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint


# Pre-registered interference vocabulary (4 cross-category rebindings):
INTERFERING_PAIRS = [
    ("apple", "motor_W"),                # was noun_pool_APPLE
    ("go", "adjective_pool_BIG"),        # was verb_pool_GO
    ("big", "verb_pool_GO"),             # was adjective_pool_BIG
    ("north", "noun_pool_APPLE"),        # was motor_N
]
N_INTERFERING_EVENTS_PER_PAIR_DEFAULT = 50  # ~10% of original 200ev training


def _gates_to_open():
    """All Phase-1 plasticity gates that need to be open for the
    interfering training to take effect across the 4 target pool kinds."""
    return [
        "language_input_to_motor",
        "language_input_to_noun_pool",
        "language_input_to_verb_pool",
        "language_input_to_adjective_pool",
    ]


def run_interference_training_and_save(seed: int, src_cache_dir: str,
                                         dst_cache_dir: str,
                                         n_interfering_per_pair: int = N_INTERFERING_EVENTS_PER_PAIR_DEFAULT):
    """Load substrate from src_cache_dir, apply interference training,
    save resulting state to dst_cache_dir."""
    print(f"\n=== Interference training; seed {seed} ===")
    print(f"  Source cache: {src_cache_dir}")
    print(f"  Destination : {dst_cache_dir}")

    bridge = _build_bridge_with_phase1_recipe(int(seed), False)
    src_path = _phase1_cache_path(src_cache_dir, seed)
    print(f"  Loading {src_path}")
    bridge.load_checkpoint(str(src_path))

    # Open the language_input gates so STDP can modify the existing
    # word-to-pool bindings (the standard catastrophic-forgetting scenario:
    # new training is applied while plasticity is restored).
    for gate in _gates_to_open():
        bridge.set_plasticity_gate(gate, 1.0)

    train_kwargs = _phase1_train_kwargs(False)
    all_words_ordered = (
        list(cpd.DIRECTION_VOCAB)
        + list(cpd.NOUN_VOCAB)
        + list(cpd.VERB_VOCAB)
        + list(cpd.ADJECTIVE_VOCAB)
    )
    word_to_idx = {w: i for i, w in enumerate(all_words_ordered)}
    n_words_total = len(all_words_ordered)

    # Train interfering rebindings. RNG-shuffled training events to
    # match the interleaved-training pattern used in Phase 1.
    rng = np.random.default_rng(int(seed) + 9999)  # different seed slice
    buffer = []
    for word, new_target in INTERFERING_PAIRS:
        for _ in range(int(n_interfering_per_pair)):
            buffer.append((word, new_target))
    rng.shuffle(buffer)

    print(f"  Interference training: {len(buffer)} total events")
    print(f"  Pairs: {INTERFERING_PAIRS}")

    t_start = time.time()
    for i, (word, target) in enumerate(buffer):
        cpd.train_word_to_pool(
            bridge, word, target, n_events=1, reset_steps=50,
            n_lang_input=int(train_kwargs["n_lang_input"]),
            n_lang_output=int(train_kwargs["n_lang_input"]),
            sparsity=float(train_kwargs["sparsity"]),
            orthogonal_codes=bool(train_kwargs["orthogonal_codes"]),
            n_words_for_orthogonal=int(n_words_total),
            word_to_idx=word_to_idx, verbose=False,
        )
    elapsed = time.time() - t_start
    print(f"  Interference training complete: {elapsed:.1f}s wall-clock")

    Path(dst_cache_dir).mkdir(parents=True, exist_ok=True)
    dst_path = _phase1_cache_path(dst_cache_dir, seed)
    print(f"  Saving post-interference cache to {dst_path}")
    bridge.save_checkpoint(str(dst_path))
    return dst_path


def run_probe_for_regime(seed: int, ev: int,
                            n_interfering_per_pair: int = N_INTERFERING_EVENTS_PER_PAIR_DEFAULT):
    """Run the full catastrophic-forgetting probe for one (seed, ev) cell."""
    if ev == 200:
        src_cache = "research/findings/raw/unified_per_regime/phase1"
    else:
        src_cache = f"research/findings/raw/unified_per_regime/phase1_{ev}ev"
    dst_cache = f"research/findings/raw/unified_per_regime/phase1_{ev}ev_post_interference_{n_interfering_per_pair}per"

    # Pre-interference diagnostic (on existing cache, fresh bridge)
    pre = test_one_checkpoint(
        seed, src_cache, f"PRE-interference unified {ev}ev seed {seed}"
    )

    # Interference training + save
    run_interference_training_and_save(seed, src_cache, dst_cache,
                                          n_interfering_per_pair=n_interfering_per_pair)

    # Post-interference diagnostic (on new cache, fresh bridge)
    post = test_one_checkpoint(
        seed, dst_cache, f"POST-interference unified {ev}ev seed {seed}"
    )

    pre_acc = pre["accuracy"]
    post_acc = post["accuracy"]
    fgt = 0.0 if pre_acc == 0 else 100.0 * (pre_acc - post_acc) / pre_acc

    # Per-category analysis
    interfering_words = {p[0] for p in INTERFERING_PAIRS}
    pre_by_word = {w["word"]: w["correct"] for w in pre["per_word"]}
    post_by_word = {w["word"]: w["correct"] for w in post["per_word"]}

    # Direct interference: did the 4 interfered words flip OK->XX?
    direct_lost = [w for w in interfering_words
                    if pre_by_word.get(w) and not post_by_word.get(w)]
    direct_retained = [w for w in interfering_words
                        if pre_by_word.get(w) and post_by_word.get(w)]

    # Indirect interference: did non-interfered words flip?
    non_interfering = set(pre_by_word) - interfering_words
    indirect_lost = [w for w in non_interfering
                      if pre_by_word.get(w) and not post_by_word.get(w)]
    indirect_gained = [w for w in non_interfering
                        if not pre_by_word.get(w) and post_by_word.get(w)]

    summary = {
        "seed": seed,
        "ev_per_word": ev,
        "src_cache": src_cache,
        "post_interference_cache": dst_cache,
        "interfering_pairs": INTERFERING_PAIRS,
        "n_interfering_events_per_pair": int(n_interfering_per_pair),
        "pre_accuracy": pre_acc,
        "pre_n_correct": pre["n_correct"],
        "post_accuracy": post_acc,
        "post_n_correct": post["n_correct"],
        "forgetting_pct": fgt,
        "direct_interfered_words_lost": direct_lost,
        "direct_interfered_words_retained": direct_retained,
        "indirect_words_lost": indirect_lost,
        "indirect_words_gained": indirect_gained,
        "pre_per_word": pre["per_word"],
        "post_per_word": post["per_word"],
    }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ev-list", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--n-interfering-per-pair", type=int,
                          default=N_INTERFERING_EVENTS_PER_PAIR_DEFAULT)
    parser.add_argument("--out", type=str,
                          default="research/findings/raw/catastrophic_forgetting_probe.json")
    args = parser.parse_args()

    results = []
    for ev in args.ev_list:
        summary = run_probe_for_regime(args.seed, ev,
                                          n_interfering_per_pair=args.n_interfering_per_pair)
        results.append(summary)
        print(f"\n=== CATASTROPHIC-FORGETTING PROBE RESULT ({ev}ev seed {args.seed}) ===")
        print(f"  PRE accuracy : {summary['pre_n_correct']}/16 = {100.0*summary['pre_accuracy']:.1f}%")
        print(f"  POST accuracy: {summary['post_n_correct']}/16 = {100.0*summary['post_accuracy']:.1f}%")
        print(f"  Forgetting % : {summary['forgetting_pct']:+.1f}%")
        print(f"  Direct interfered lost (of 4): {summary['direct_interfered_words_lost']}")
        print(f"  Direct interfered retained   : {summary['direct_interfered_words_retained']}")
        print(f"  Indirect lost (of 12)        : {summary['indirect_words_lost']}")
        print(f"  Indirect gained              : {summary['indirect_words_gained']}")

    # Cross-regime comparison
    print("\n=== CROSS-REGIME COMPARISON ===")
    print(f"  Regime | PRE    | POST   | Forgetting% | Interfered lost | Indirect lost")
    for r in results:
        print(f"  {r['ev_per_word']:>5} | {100.0*r['pre_accuracy']:>5.1f}% | "
              f"{100.0*r['post_accuracy']:>5.1f}% | {r['forgetting_pct']:>+9.1f}% | "
              f"{len(r['direct_interfered_words_lost'])}/4 | "
              f"{len(r['indirect_words_lost'])}/12")

    if len(results) >= 2:
        ev_to_fgt = {r['ev_per_word']: r['forgetting_pct'] for r in results}
        if 200 in ev_to_fgt and 800 in ev_to_fgt:
            delta = ev_to_fgt[200] - ev_to_fgt[800]
            print(f"\n  Forgetting % delta (200ev - 800ev): {delta:+.1f}pp")
            if delta >= 10:
                print("  --> CLS PREDICTION VALIDATED (single-seed): 200ev forgets >= 10pp "
                      "more than 800ev. Expand multi-seed.")
            elif abs(delta) < 10:
                print("  --> Substrate regimes NOT corresponding to interference-resistance "
                      "regimes; honest finding. CLS prediction NOT validated at this magnitude.")
            else:
                print("  --> OPPOSITE-DIRECTION finding: 800ev forgets MORE than 200ev. "
                      "Warrants diagnostic of WHY.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_ev_results": results,
                    "interfering_pairs": INTERFERING_PAIRS,
                    "n_interfering_events_per_pair": int(args.n_interfering_per_pair)},
                   f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
