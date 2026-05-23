"""Vocabulary scaling on the trained substrate with PATTERN-GROUNDED
symbols -- candidate 2 of the vocabulary-scaling NEGATIVE branch.

WHY THIS RUNNER EXISTS
----------------------
The trained-substrate decisive run cleared the frozen 0.80 bar at
loads 2-3 multi-seed (0.842, 0.814) but missed at load 5 (0.756). The
load-ceiling characterisation showed the ceiling sits between binding
loads 3 and 4 (L=4 mean 0.7988, miss by 0.0012; smooth monotonic
~0.03/load decay), about a 30x capacity reduction from the pure FHRR
algebra at the same phasor dimension. The hypothesis: the spiking-
symbol noise floor is the limit; replacing the noisy activity-derived
symbol with the substrate's clean K-of-N pattern-derived symbol should
raise the ceiling.

THE HONEST ORACLE-ADJACENCY CAVEAT (recorded up front)
------------------------------------------------------
The K-of-N pattern is the substrate's own concept code -- stored in
the trained connectivity, evoked by the language-input drive, and
extracted via the existing recognition front-end (which still reads
the noisy activity). So pattern-grounded symbols ARE substrate-
grounded. But the pattern abstracts past the per-observation noise to
the underlying stable ensemble identity, so it is ONE STEP CLOSER to
oracle-lookup than activity-grounded. A PASS here is read with that
caveat in mind, not as a biological compositional result at the same
fidelity as activity-grounded. See the design doc:
`docs/plans/2026-05-22-pattern-grounded-symbol-design.md`.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The symbol-derivation step is the only thing that changes. The
recognition front-end (temporally averaged nearest-match in the
captured activity space), the FHRR pipeline (resonate-and-fire
bind/unbind/bundle), the attractor clean-up with separate familiarity
gate, the deriver (same fixed-seed linear projection), the frozen
0.80 bar, the multi-seed grid {42, 43, 44}, the loads {2, 3, 5}, the
FHRR phasor dimension 512 -- ALL imported byte-unchanged from
`vocabulary_scaling_run.py`.

The recognised concept name (the OUTPUT of recognition, not the true
label) selects which K-of-N pattern is read from the per-seed pattern
store. The true label NEVER indexes the pattern store -- doing so
would be an answer leak. The adversarial review (Task 4) must
exploit-check this explicitly.

PRE-REGISTERED reading (fixed; never tuned):
- PASS: integrated multi-seed mean >= 0.80 at all loads {2, 3, 5}.
  Subject to the oracle-adjacency caveat above, pattern-grounded
  symbols raise the ceiling past where activity-grounded missed.
- NEGATIVE: integrated below 0.80 at some load. The spiking-symbol
  noise is NOT the only ceiling cause; the limit is deeper.

Reuse-by-import only; no protected/frozen/moat module modified; no
automatic differentiation. Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The biologized pipeline + everything downstream of the substrate,
# imported byte-unchanged from the adversarially-reviewed decisive
# runner.
from research.findings.raw.vocabulary_scaling_run import (
    N_CONCEPTS, BAR, LOADS, SEEDS, N_DIM, K_RECOG, K_VOCAB, N_TRIALS,
    recognition_accuracy, _load_cache, _cosine, partition_cue_filler,
)
# Task 1's pure helper.
from research.findings.raw.vocabulary_scaling_pattern_helpers import (
    pattern_vector,
)
# The biologized pipeline's fixed-seed phasor deriver -- identical to
# what activity-grounded uses; only the input vector differs.
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.resonate_fire_fhrr import (
    ResonateFireFHRR, ResonateFireTPAM,
    ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH, ANNEAL_ITERS,
)

# Same deriver seed as the activity-grounded path. Pinned by
# tests/test_vocabulary_scaling_pattern_grounded.py.
DERIV_SEED = 90909

# The trained substrate's activity cache (read for recognition; the
# decisive trained-substrate run populated it).
TRAINED_CACHE_DIR = os.path.join(
    _HERE, "vocabulary_scaling_trained_cache")


def _ground_symbols_pattern(words, patterns, n_pool, d_act):
    """The pattern-grounded symbol derivation: per concept, build the
    binary K-of-N indicator vector and project it through the SAME
    fixed-seed deriver the activity-grounded path uses, then quantise
    to phasor spikes. The genuinely-new symbol-derivation step --
    everything else in the pipeline is reused byte-unchanged.

    `d_act` is the deriver's input dimensionality. The activity-
    grounded path uses d_act = n_pool (one feature per pool neuron);
    we use the same d_act here so the deriver is byte-identical.
    """
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(pattern_vector(patterns[i], n_pool)))
            for i, w in enumerate(words)}


def run_one_seed_pattern(seed):
    """Run the pipeline on the trained activity cache with
    pattern-grounded symbols. Recognition reads the cached activity
    (unchanged); only the symbol derivation differs.

    The body below mirrors `vocabulary_scaling_run.run_pipeline`
    verbatim modulo the `grounded` source -- pattern-grounded here vs
    activity-grounded there. run_pipeline itself is NOT modified.
    """
    print(f"\n--- seed {seed} ---", flush=True)
    path = os.path.join(TRAINED_CACHE_DIR, f"trained_full_seed{seed}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"trained activity cache missing: {path}; run the trained-"
            f"substrate decisive runner first to populate it")
    acts, words, patterns = _load_cache(path)
    d_act = acts[words[0]].shape[1]
    n_pool = d_act

    # Recognition (reused unchanged) -- the only handle that names which
    # concept's pattern is read.
    consolidated = {w: acts[w][:K_VOCAB].mean(axis=0) for w in words}
    rec_per_obs, rec_avg = recognition_accuracy(
        acts, words, consolidated, K_RECOG,
        np.random.default_rng(seed + 7))

    # Substitute the symbol-derivation step.
    grounded = _ground_symbols_pattern(words, patterns, n_pool, d_act)

    # Pipeline body, mirroring run_pipeline -- only the `grounded`
    # source differs.
    cue_words, filler_words = partition_cue_filler(words)
    fidx = {fw: i for i, fw in enumerate(filler_words)}
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    tpam = ResonateFireTPAM([grounded[fw] for fw in filler_words])
    qrng = np.random.default_rng(seed + 1)
    consolidated_mat = {w: consolidated[w] for w in words}

    def reco(word):
        m = acts[word].shape[0]
        k = min(K_RECOG, m)
        idx = qrng.choice(m, size=k, replace=False)
        avg = acts[word][idx].mean(axis=0)
        best_w, best_s = None, -2.0
        for w in words:
            s = _cosine(avg, consolidated_mat[w])
            if s > best_s:
                best_s, best_w = s, w
        return best_w

    per_load = {}
    for load in LOADS:
        n_int_ok = n_int_tot = 0
        n_comp_ok = n_comp_tot = 0
        eff_load = min(load, len(cue_words), len(filler_words))
        for _ in range(N_TRIALS):
            cues = list(qrng.choice(cue_words, size=eff_load,
                                    replace=False))
            fills = list(qrng.choice(filler_words, size=eff_load,
                                     replace=True))
            rec_cue = {c: reco(c) for c in set(cues)}
            rec_fill = {f: reco(f) for f in set(fills)}
            facts = list(zip(cues, fills))
            composite = net.encode([
                (grounded[rec_cue[c]], grounded[rec_fill[f]])
                for (c, f) in facts])
            for (c, f) in facts:
                recovered = net.query(composite, grounded[rec_cue[c]])
                z, _ = tpam.settle_annealed(
                    recovered, ANNEAL_THETA_LOW, ANNEAL_THETA_HIGH,
                    ANNEAL_ITERS, fast=True)
                overlaps = np.abs(tpam.s.conj().T @ z)
                hit = (int(np.argmax(overlaps)) == fidx[f])
                n_int_ok += int(hit)
                n_int_tot += 1
                if rec_cue[c] == c and rec_fill[f] == f:
                    n_comp_ok += int(hit)
                    n_comp_tot += 1
        int_acc = n_int_ok / n_int_tot if n_int_tot else float("nan")
        comp_acc = (n_comp_ok / n_comp_tot) if n_comp_tot else float("nan")
        per_load[load] = {
            "integrated_accuracy": int_acc,
            "composition_only_accuracy": comp_acc,
            "n_composition_only": n_comp_tot,
            "effective_load": eff_load,
        }

    for load in LOADS:
        e = per_load[load]
        print(f"  L={load}: integrated acc={e['integrated_accuracy']:.4f} "
              f"| composition-only acc={e['composition_only_accuracy']:.4f} "
              f"(n={e['n_composition_only']})", flush=True)
    print(f"  [seed {seed}] recognition (reported separately): "
          f"per-observation={rec_per_obs:.4f}, "
          f"temporally-averaged={rec_avg:.4f}", flush=True)

    return {
        "seed": seed, "trained_substrate": True,
        "symbol_grounding": "pattern",
        "n_concepts": len(words), "activity_dim": int(d_act),
        "recognition_per_observation": rec_per_obs,
        "recognition_temporally_averaged": rec_avg,
        "per_load": {str(k): v for k, v in per_load.items()},
    }


def main():
    ap = argparse.ArgumentParser(
        description="Pattern-grounded compositional symbols on the "
                    "trained 64-concept G.20 sparse substrate -- "
                    "candidate 2 of the NEGATIVE branch.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    seeds = list(args.seeds)

    print("=== vocabulary scaling: PATTERN-GROUNDED symbols on the "
          "trained 64-concept G.20 sparse substrate ===", flush=True)
    print(f"  ORACLE-ADJACENCY CAVEAT: the K-of-N pattern is the "
          f"substrate's own concept code -- still substrate-grounded -- "
          f"but one step closer to oracle-lookup than activity-grounded; "
          f"a PASS is read with that caveat (see design doc).",
          flush=True)
    print(f"concepts={N_CONCEPTS}; FHRR N_dim={N_DIM}; loads={LOADS}; "
          f"bar={BAR}; seeds={seeds}; substrate=TRAINED (cache reused); "
          f"recognition unchanged; symbol grounding=PATTERN",
          flush=True)

    seed_results = [run_one_seed_pattern(s) for s in seeds]

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][str(load)]["integrated_accuracy"]
                    for r in seed_results]
        comp_accs = [r["per_load"][str(load)]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        valid_comp = [c for c in comp_accs if c == c]
        mean_comp = float(np.mean(valid_comp)) if valid_comp else float("nan")
        agg[load] = {"mean_integrated": mean_int,
                     "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed="
              f"{['%.3f' % a for a in int_accs]} mean={mean_int:.4f} "
              f"({'>=' if mean_int >= BAR else '<'} {BAR}) | "
              f"composition-only mean={mean_comp:.4f}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if all_pass:
        verdict = "VOCABULARY_SCALING_64CONCEPT_PATTERN_GROUNDED_PASS"
        print("  Pattern-grounded symbols clear the frozen 0.80 bar "
              "multi-seed at all loads on the trained 64-concept G.20 "
              "sparse substrate. Subject to the oracle-adjacency caveat "
              "above.", flush=True)
    else:
        verdict = "VOCABULARY_SCALING_64CONCEPT_PATTERN_GROUNDED_BELOW_BAR"
        print("  Pattern-grounded multi-seed mean is below 0.80 at some "
              "load. The spiking-symbol noise is NOT the only ceiling "
              "cause; the limit is deeper.", flush=True)

    out = {
        "seeds": seeds, "n_concepts": N_CONCEPTS, "n_dim": N_DIM,
        "k_recog": K_RECOG, "loads": LOADS, "n_trials": N_TRIALS,
        "bar": BAR, "substrate": "trained",
        "symbol_grounding": "pattern",
        "oracle_adjacency_caveat": (
            "The K-of-N pattern is the substrate's own concept code -- "
            "still substrate-grounded -- but one step closer to "
            "oracle-lookup than activity-grounded; a PASS is read with "
            "this caveat."),
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "verdict": verdict,
    }
    out_path = os.path.join(
        _HERE, "vocabulary_scaling_run_pattern_grounded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
