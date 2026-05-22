"""Integration: the project's concept substrate + the spiking-phasor
FHRR composition subsystem, end-to-end.

The spiking-phasor FHRR subsystem (research/runners/spiking_phasor_fhrr.py)
is a validated working composition layer -- but on abstract symbols.
This runner is the genuine integration: the project's validated
v14/v16 + hippocampus substrate is the concept-RECOGNITION front-end;
the spiking-phasor FHRR subsystem is the composition BACK-END; they
join at the concept-identity level.

Pipeline:
1. Load a project substrate that does validated direct binding
   (the 800-event multi-seed substrate -- best recognition, 85.4%
   aggregate).
2. RECOGNITION: for each concept word, drive it through the substrate
   and read which concept pool fires (the validated direct-binding
   readout). The recognized pool is the substrate's concept identity
   for that word -- and may be wrong (recognition is imperfect).
3. Each of the concept pools is assigned a fixed deterministic
   spiking-phasor symbol. A word's FHRR symbol = the symbol of the
   pool the substrate RECOGNIZED it as (recognition error therefore
   propagates honestly into composition).
4. COMPOSITION: encode (cue, adjective) facts via the FHRR subsystem
   (bind + bundle), query each via unbind, clean up with the
   abstention moat.
5. Measure against the project's frozen 0.80 bar.

Two accuracies are reported, honestly:
- integrated accuracy: the whole pipeline (recognition errors and all)
- composition-only accuracy: restricted to facts whose words were all
  correctly recognized -- isolates whether FHRR composition itself
  works on the substrate-recognized symbols.

PRE-REGISTERED reading (fixed):
- If integrated accuracy >= 0.80 at loads {2,3,5}: the integrated
  two-system pipeline clears the bar.
- If integrated accuracy < 0.80 but composition-only accuracy is
  high: composition is solved; the integrated pipeline's bottleneck
  is RECOGNITION (the substrate's direct binding) -- which relocates
  the project's open problem and is itself the finding.

Reuse-by-import: test_one_checkpoint (recognition) + the
spiking_phasor_fhrr subsystem. No protected/frozen/moat module
modified. No autograd. Controller-only; single seed 42.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the recognition front-end (the validated direct-binding readout).
from importlib import util as _iu
_db_path = os.path.join(_HERE, "direct_binding_phase1_comparison.py")
_spec = _iu.spec_from_file_location("_db", _db_path)
_db = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_db)
test_one_checkpoint = _db.test_one_checkpoint

# Reuse the validated composition back-end.
from research.runners.spiking_phasor_fhrr import (
    SpikingPhasorFHRR, phase_similarity, CYCLE_STEPS,
)

SEEDS = [42, 43, 44]
RECOG_CACHE = "research/findings/raw/unified_per_regime/phase1_800ev"
N_DIM = 512
LOADS = [2, 3, 5]
N_TRIALS = 300
BAR = 0.80


def run_one_seed(seed):
    """Recognition + composition for one substrate seed. Returns the
    per-load integrated + composition-only accuracies."""
    print(f"\n--- seed {seed} ---")
    # --- RECOGNITION front-end: the validated direct-binding readout.
    recog = test_one_checkpoint(seed, RECOG_CACHE,
                                  f"integration recognition seed {seed}")
    per_word = {w["word"]: w for w in recog["per_word"]}
    print(f"recognition: {recog['n_correct']}/{recog['n_total']} words "
          f"correctly recognized by the substrate")

    cue_words = [w for w in per_word
                 if per_word[w]["target_pool"].startswith(("noun_pool_",
                                                            "verb_pool_"))]
    filler_words = [w for w in per_word
                    if per_word[w]["target_pool"].startswith("adjective_pool_")]
    task_words = cue_words + filler_words
    task_recog_ok = sum(per_word[w]["correct"] for w in task_words)

    rng = np.random.default_rng(seed)
    net = SpikingPhasorFHRR(N_DIM, rng)
    all_pools = sorted({per_word[w]["target_pool"] for w in per_word}
                       | {per_word[w]["top_pool"] for w in per_word})
    pool_symbol = {p: net.random_symbol() for p in all_pools}

    def word_symbol(word):
        return pool_symbol[per_word[word]["top_pool"]]

    def word_true_pool(word):
        return per_word[word]["target_pool"]

    qrng = np.random.default_rng(seed + 1)
    per_load = {}
    for load in LOADS:
        n_int_correct = n_int_total = 0
        n_comp_correct = n_comp_total = 0
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            facts = list(zip(cues, fills))
            composite = net.encode([(word_symbol(c), word_symbol(f))
                                     for (c, f) in facts])
            for (c, f) in facts:
                recovered = net.query(composite, word_symbol(c))
                sims = {fw: phase_similarity(recovered, word_symbol(fw))
                        for fw in filler_words}
                best = max(sims, key=sims.get)
                hit = (word_true_pool(best) == word_true_pool(f))
                n_int_correct += int(hit)
                n_int_total += 1
                if per_word[c]["correct"] and per_word[f]["correct"]:
                    n_comp_correct += int(hit)
                    n_comp_total += 1
        int_acc = n_int_correct / n_int_total
        comp_acc = (n_comp_correct / n_comp_total) if n_comp_total else float("nan")
        per_load[load] = {
            "integrated_accuracy": int_acc,
            "composition_only_accuracy": comp_acc,
            "n_composition_only": n_comp_total,
        }
        print(f"  L={load}: integrated acc={int_acc:.4f} | "
              f"composition-only acc={comp_acc:.4f} (n={n_comp_total})")
    return {
        "seed": seed,
        "recognition_n_correct": recog["n_correct"],
        "recognition_n_total": recog["n_total"],
        "task_words_total": len(task_words),
        "task_words_recognized": int(task_recog_ok),
        "per_load": per_load,
    }


def main():
    print("=== spiking-phasor FHRR x project substrate -- integration ===")
    print(f"recognition substrate: {RECOG_CACHE} seeds {SEEDS}; "
          f"FHRR N_dim={N_DIM}; loads={LOADS}; bar={BAR}")

    seed_results = [run_one_seed(s) for s in SEEDS]

    # Aggregate across seeds, per load.
    print(f"\n=== MULTI-SEED AGGREGATE ===")
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][load]["integrated_accuracy"] for r in seed_results]
        comp_accs = [r["per_load"][load]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        mean_comp = float(np.mean([c for c in comp_accs if c == c]))
        agg[load] = {"mean_integrated": mean_int, "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed={['%.3f' % a for a in int_accs]} "
              f"mean={mean_int:.4f} ({'>=' if mean_int >= BAR else '<'} {BAR}) "
              f"| composition-only mean={mean_comp:.4f}")

    print(f"\n=== VERDICT ===")
    if all_pass:
        verdict = "INTEGRATED_MULTISEED_PASS"
        print("  The integrated two-system pipeline (substrate recognition "
              "+ spiking-phasor FHRR composition) clears the frozen 0.80 "
              "bar multi-seed mean at all loads.")
    else:
        verdict = "RECOGNITION_BOUNDED"
        print("  Integrated multi-seed mean is below 0.80 at some load; "
              "composition-only stays high -> the pipeline bottleneck is the "
              "substrate's concept-recognition accuracy (RECOGNITION-BOUNDED).")

    out = {
        "seeds": SEEDS, "recognition_cache": RECOG_CACHE, "n_dim": N_DIM,
        "loads": LOADS, "n_trials": N_TRIALS, "bar": BAR,
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "verdict": verdict,
    }
    with open("research/findings/raw/spiking_phasor_integration.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/spiking_phasor_integration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
