"""Direction 5 Task 4: cross-bridge parallel-matching mode-unification
probe on the HYBRID sparse-distributed shared-pool bio_brain_regions
5-bridge ensemble (80 cross-bridge concepts = 5 categories x V=16).

Mirrors the Direction 4 cross-bridge probe pattern byte-pattern
(direction_4_cross_bridge_probe.py) byte-unchanged in primitive:
reuses batched_phase_similarity + verify_batched_equivalent_to_scalar +
global mean-centred grounded-symbol derivation (DERIV_SEED=90909) +
ResonateFireFHRR bind/unbind + parallel-matching decoder (per-slot
argmax over the 80-concept union via the same batched primitive). The
only mechanical differences vs Direction 4 are:

  (a) substrate is the HYBRID architecture (Direction 5: bio dedicated
      pools PLUS a NEW 2000-neuron shared_concept_pool with per-concept
      K=100 sparse patterns at pillar n=95 strength prior); each bridge's
      cached activity comes from the shared_concept_pool region ONLY (NOT
      the dedicated noun/verb/adjective pool union as in Direction 4).
  (b) per-bridge d_act is uniform 2000 across all 5 bridges (each
      bridge contributes the same 2000-neuron shared sparse pool to the
      cross-bridge probe); Direction 4's per-bridge d_act varied with
      pool kind + count.

CPU-only by design (operates on cached per-bridge activity .npz files
written by direction_5_5bridge_runner.py, Task 5). Does NOT import
CuPy or SimulationBridge at module level. The activity arrays are
already on host (numpy.float32) when this module is invoked.

Pipeline (per seed):
1. Load 5 bridges' per-seed cached activity (.npz with per-word arrays
   of shape (M_OBS, 2000) - the shared_concept_pool region).
2. Consolidate per-(bridge, word) activity: mean across M_OBS to one
   length-2000 vector per concept.
3. Mean-centre PER BRIDGE across the 16 concepts (the n=95 cross-bridge
   choice; cortical pooled inhibition normalises across the cortical
   region). For each bridge, the deriver projects 2000 -> N_DIM=512
   phasors; the 80 phasors live in a uniform N_DIM space for the
   cross-bridge similarity.
4. For each load L in {2, 3, 5}: sample n_trials composites uniformly
   from the 80-concept union; FHRR-encode with N_GAMMA_SLOTS positional
   codes; per-slot argmax decode + marginal-sum order-invariant decode
   via batched_phase_similarity over the 80-concept phase matrix.
5. Compute per-load (OB, OI) accuracy; emit verdict via the frozen
   direction_5_verdict module.

PRE-REGISTERED reading (frozen by direction_5_verdict module):
- DIRECTION_5_PASS: multi-seed-mean OB AND OI both clear 0.80 at every
  load in {2, 3, 5}.
- DIRECTION_5_PARTIAL: some cells above bar but not all.
- DIRECTION_5_NEGATIVE: NO load-cell on EITHER readout clears the bar.
- DIRECTION_5_VOID_MALFORMED: instrument-validity failure (per the
  verdict module's fail-closed contract).

Activity cache contract (written by Task 5 runner):
  {CACHE_DIR}/activity_{tag}_{bridge_name}_seed{N}.npz
  with per-word arrays keyed by str(word):
    npz[str(word)] -> ndarray (M_OBS, 2000) float32

Tag is "full" or "smoke". Per-bridge d_act is uniform 2000 (the
shared_concept_pool size) across all 5 bridges by design. All npz
files are loaded numerically only (no object arrays).

Reuses every mode-unification primitive byte-unchanged via import; no
protected/frozen/moat module modified; no autograd; no-confab moat
must stay 7/7 green.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Direction 5 modules (this arc's net-new code).
from research.findings.raw.direction_5_vocab_spec import (
    DIRECTION_5_BRIDGE_A_WORDS,
    DIRECTION_5_BRIDGE_B_WORDS,
    DIRECTION_5_BRIDGE_C_WORDS,
    DIRECTION_5_BRIDGE_D_WORDS,
    DIRECTION_5_BRIDGE_E_WORDS,
    DIRECTION_5_BRIDGE_CATALOG,
    DIRECTION_5_TOTAL,
)
from research.findings.raw.direction_5_verdict import (
    compute_verdict,
    _DIRECTION_5_OB_MIN,
    _DIRECTION_5_OI_MIN,
    _DIRECTION_5_LOADS,
    _DIRECTION_5_MIN_SEEDS,
    DIRECTION_5_VOID_MALFORMED,
)

# Reuse-by-import only (validated mode-unification primitives, pillar
# n=95 cross-bridge primitive + Direction 4 cross-bridge primitive).
from research.findings.raw.vocabulary_scaling_run import (
    BAR, N_DIM, N_TRIALS,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from sim.backend import get_backend, is_gpu_backend, to_host


# -----------------------------------------------------------------------
# Constants (mirroring n=95 + Direction 4 V=16 patterns).
# -----------------------------------------------------------------------
SEEDS = [42, 43, 44]
LOADS = list(_DIRECTION_5_LOADS)
BAR_OB = _DIRECTION_5_OB_MIN
BAR_OI = _DIRECTION_5_OI_MIN
N_GAMMA_SLOTS = 7

# The frozen iteration order over the 5 bridges (matches
# DIRECTION_5_ALL_WORDS order in direction_5_vocab_spec). The cross-
# bridge probe takes the union in this exact order so word_idx in
# decoder = position in DIRECTION_5_ALL_WORDS.
BRIDGES: Tuple[str, ...] = (
    "A_nouns", "B_verbs", "C_adj", "D_spatial", "E_functional"
)

# Per-bridge word lists (frozen; matches vocab_spec).
_PER_BRIDGE_WORDS: Dict[str, List[str]] = {
    "A_nouns": DIRECTION_5_BRIDGE_A_WORDS,
    "B_verbs": DIRECTION_5_BRIDGE_B_WORDS,
    "C_adj": DIRECTION_5_BRIDGE_C_WORDS,
    "D_spatial": DIRECTION_5_BRIDGE_D_WORDS,
    "E_functional": DIRECTION_5_BRIDGE_E_WORDS,
}

# Cache directory written by Task 5 runner (shared root; per-bridge per-
# seed npz lives at CACHE_DIR/activity_{tag}_{bridge_name}_seed{N}.npz).
CACHE_DIR: str = os.path.join(_HERE, "direction_5_cache")


# -----------------------------------------------------------------------
# Activity-cache loader (numeric-only npz; safe load mode).
# -----------------------------------------------------------------------
def _activity_cache_path(bridge_name: str, seed: int, tag: str) -> str:
    """Per-bridge per-seed activity cache path. Tag is 'full' or 'smoke'."""
    return os.path.join(
        CACHE_DIR,
        "activity_" + tag + "_" + bridge_name + "_seed" + str(seed)
        + ".npz",
    )


def _load_per_bridge_activity(
    bridge_name: str, seed: int, tag: str,
) -> Dict[str, np.ndarray]:
    """Load per-word activity arrays from one (bridge, seed) npz cache.

    Returns: {word: ndarray (M_OBS, 2000)} for the bridge's V=16
             category words; the second dimension is uniform 2000
             across all bridges (= the shared_concept_pool region size).

    Raises FileNotFoundError if the cache hasn't been written by Task 5
    yet. Raises ValueError if any expected word is missing.
    """
    cache_p = _activity_cache_path(bridge_name, seed, tag)
    if not os.path.exists(cache_p):
        raise FileNotFoundError(
            "Direction 5 activity cache missing: " + cache_p
            + " (Task 5 runner has not produced this bridge/seed yet)"
        )
    data = np.load(cache_p, allow_pickle=False)
    words = _PER_BRIDGE_WORDS[bridge_name]
    acts: Dict[str, np.ndarray] = {}
    for w in words:
        if str(w) not in data.files:
            raise ValueError(
                "Direction 5 activity cache " + cache_p
                + " is missing word '" + str(w) + "' "
                "(present keys: " + str(sorted(data.files)) + ")"
            )
        acts[w] = data[str(w)]
    return acts


# -----------------------------------------------------------------------
# Per-bridge mean-centred grounded-symbol derivation (per pillar n=95 +
# Direction 4 V=16 pattern). The same mechanical primitive as Direction
# 4; only the activity source differs (shared_concept_pool 2000 here vs
# dedicated pool union in D4).
# -----------------------------------------------------------------------
def derive_global_grounded_symbols(
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    seed: int,
    k_vocab_per_concept: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], np.ndarray]]:
    """Build the 80-concept union; mean-centre within each bridge; derive
    grounded phasor symbols via the SAME fixed-seed deriver pipeline used
    by pillars n=93/n=94/n=95/n=96 and Direction 4.

    For Direction 5, each bridge's activity vectors have UNIFORM length
    2000 (the shared_concept_pool size). Each bridge's vectors get
    mean-centred across the 16 concepts on THAT bridge (cortical pooled
    inhibition normalises within each cortical region); the deriver then
    projects d_act=2000 -> N_DIM=512 phasors per bridge with the fixed
    DERIV_SEED. The derived phasors then live in a uniform N_DIM
    space for cross-bridge similarity computation.

    Args:
        per_bridge_activity: {bridge_name: {word: (M_OBS, 2000)}}
                              one entry per bridge in BRIDGES.
        seed: ignored by the deriver itself (DERIV_SEED is the deriver's
              own fixed seed; this `seed` is documented for the caller's
              own bookkeeping).
        k_vocab_per_concept: if not None, average only the first k_vocab
                              observations per word. None = use all M_OBS.

    Returns:
        all_words: list of (bridge_name, word) tuples in cross-bridge
                    union order (matches DIRECTION_5_ALL_WORDS).
        grounded: {(bridge_name, word): phases_to_spikes(deriver(...))}
                   = spike-phase representation of each concept's
                   mean-centred shared_concept_pool activity.
    """
    all_words: List[Tuple[str, str]] = []
    grounded: Dict[Tuple[str, str], np.ndarray] = {}

    for bridge in BRIDGES:
        acts = per_bridge_activity[bridge]
        words = _PER_BRIDGE_WORDS[bridge]
        # Consolidate per-word: mean across M_OBS observations.
        if k_vocab_per_concept is not None:
            consolidated_b = {
                w: acts[w][:k_vocab_per_concept].mean(axis=0)
                for w in words
            }
        else:
            consolidated_b = {w: acts[w].mean(axis=0) for w in words}
        # Per-bridge d_act (length of shared_concept_pool readout vector).
        # All 5 bridges have d_act=2000 by the Direction 5 HYBRID
        # architecture; the deriver still constructs a fresh per-bridge
        # projection so the per-bridge phasor geometry stays bridge-local
        # while landing in the uniform N_DIM space.
        d_act = next(iter(consolidated_b.values())).shape[0]
        # Mean-centre within this bridge across the 16 concepts. Each
        # bridge's deriver projects d_act -> N_DIM independently; the
        # uniform N_DIM phasor space is what enables cross-bridge
        # similarity computation.
        common_b = np.mean([consolidated_b[w] for w in words], axis=0)
        deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
        for w in words:
            key = (bridge, w)
            grounded[key] = phases_to_spikes(
                deriver(consolidated_b[w] - common_b)
            )
            all_words.append(key)
    return all_words, grounded


# -----------------------------------------------------------------------
# Per-seed cross-bridge probe (parallel-matching decoder + marginal-sum
# order-invariant). Mirrors Direction 4 run_one_seed_cross_bridge with
# the only mechanical difference: the activity reads come from the
# shared_concept_pool region (d_act=2000 uniform) rather than the
# dedicated pool union.
# -----------------------------------------------------------------------
def run_one_seed_cross_bridge(
    seed: int,
    per_bridge_activity: Dict[str, Dict[str, np.ndarray]],
    xp,
    loads: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
    k_vocab_per_concept: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[str, float]], int, float]:
    """One-seed cross-bridge parallel-matching probe.

    Args:
        seed: RNG seed for trial sampling + gamma slot derivation.
        per_bridge_activity: {bridge_name: {word: (M_OBS, 2000)}}.
        xp: backend module (numpy or cupy).
        loads: composition loads; default _DIRECTION_5_LOADS.
        n_trials: trials per load; default N_TRIALS=200.
        k_vocab_per_concept: if not None, average first K observations
                              per word (matches K_VOCAB_TARGET=16 pattern).
        verbose: print per-load progress.

    Returns:
        per_load: {load: {"order_bearing_accuracy": float,
                          "order_invariant_accuracy": float,
                          "n_trials": int}}
        V: total concepts in the union (=80 if all bridges have V=16)
        max_diff: batched-vs-scalar phase_similarity max-diff (sanity check)
    """
    if loads is None:
        loads = list(_DIRECTION_5_LOADS)
    if n_trials is None:
        n_trials = N_TRIALS

    # Build the 80-concept union + grounded phasors (per-bridge mean-
    # centred via the deriver pipeline).
    all_words, grounded = derive_global_grounded_symbols(
        per_bridge_activity, seed,
        k_vocab_per_concept=k_vocab_per_concept,
    )
    V = len(all_words)

    # Byte-equivalence check: batched == scalar phase_similarity at cell
    # start. Same fail-closed primitive Direction 4 + pillar n=95 use.
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, all_words, xp, rng_seed=seed,
    )
    if verbose:
        print(
            "  [seed " + str(seed) + "] V=" + str(V)
            + " grounded ready; batched-vs-scalar max-diff="
            + ("%.2e" % max_diff),
            flush=True,
        )

    positions = gamma_slot_positions(seed, N_GAMMA_SLOTS, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    per_load: Dict[int, Dict[str, float]] = {}
    for load in loads:
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            items_idx = tuple(
                int(i) for i in
                qrng.choice(V, size=load, replace=False)
            )
            items = [all_words[i] for i in items_idx]
            C = net.encode([
                (grounded[items[k]], positions[k]) for k in range(load)
            ])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            # ORDER-BEARING: per-slot argmax over the 80-concept union
            # via the batched primitive (identical to scalar; ~V-fold
            # speedup on GPU; verified at cell start).
            recovered = []
            scores_oi_xp = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp,
                )  # (V,) on xp backend
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_xp = scores_oi_xp + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            # ORDER-INVARIANT: marginal-sum already accumulated above
            # via the SAME batched primitive; top-K via argsort.
            scores_oi_host = to_host(scores_oi_xp)
            topK = sorted(
                int(i) for i in np.argsort(scores_oi_host)[-load:]
            )
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
        if verbose:
            print(
                "    L=" + str(load)
                + ": OB=" + ("%.3f" % per_load[load]["order_bearing_accuracy"])
                + " OI=" + ("%.3f" % per_load[load]["order_invariant_accuracy"]),
                flush=True,
            )
    return per_load, V, max_diff


# -----------------------------------------------------------------------
# Public driver: run multi-seed cross-bridge probe + emit verdict.
# -----------------------------------------------------------------------
def run_cross_bridge_probe(
    seeds: Optional[List[int]] = None,
    loads: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
    k_vocab_per_concept: Optional[int] = None,
    tag: str = "full",
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Multi-seed cross-bridge probe driver.

    Loads per-bridge per-seed activity caches written by the Task 5
    runner; runs the parallel-matching decoder + marginal-sum decoder
    per seed at each load; aggregates multi-seed means; computes the
    frozen Direction 5 verdict.

    Args:
        seeds: list of seeds to probe; default SEEDS=[42, 43, 44].
        loads: composition loads; default _DIRECTION_5_LOADS=(2, 3, 5).
        n_trials: trials per load; default N_TRIALS=200.
        k_vocab_per_concept: average first K observations per word;
                              default None = use all M_OBS.
        tag: cache file tag ("full" or "smoke").
        cache_dir: override CACHE_DIR (Task 5 writes here; this reads
                    from same path).
        verbose: print per-seed + aggregate progress.

    Returns:
        Result dict with shape:
        {
          "seeds": [...], "loads": [...], "bar_ob": 0.80, "bar_oi": 0.80,
          "V": 80, "n_bridges": 5,
          "per_seed": [{"seed": N, "V": 80, "per_load": {...},
                         "verdict_entry": {"L=2": {"OB":, "OI":}, ...}}, ...],
          "aggregate": {load: {"order_bearing_mean":, ...}, ...},
          "verdict": "DIRECTION_5_PASS" | "DIRECTION_5_PARTIAL" |
                      "DIRECTION_5_NEGATIVE" | "DIRECTION_5_VOID_MALFORMED",
          "wall_clock_seconds": float,
        }
    """
    if seeds is None:
        seeds = list(SEEDS)
    if loads is None:
        loads = list(_DIRECTION_5_LOADS)
    if n_trials is None:
        n_trials = N_TRIALS
    if cache_dir is not None:
        # Allow override (used by Task 5 runner if its CACHE_DIR differs).
        global CACHE_DIR
        CACHE_DIR = cache_dir

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    if verbose:
        print("=== Direction 5 cross-bridge probe ===", flush=True)
        print(
            "  backend=" + backend_name + " (GPU=" + str(gpu) + "); "
            "seeds=" + str(seeds) + "; loads=" + str(loads)
            + "; bridges=" + str(list(BRIDGES))
            + "; V_total=" + str(DIRECTION_5_TOTAL)
            + "; tag=" + tag,
            flush=True,
        )
        print(
            "  decoder=parallel_population_matching_batched "
            "(80-concept union, HYBRID shared_concept_pool readout); "
            "mean-centring=per_bridge_local (n=95 pattern); "
            "reuses pillar n=95 + Direction 4 primitive byte-unchanged",
            flush=True,
        )

    t0 = time.time()
    seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        if verbose:
            print("\n--- seed " + str(seed) + " ---", flush=True)
        t_seed = time.time()
        # Load all 5 bridges' activity caches for this seed.
        per_bridge_activity: Dict[str, Dict[str, np.ndarray]] = {}
        for bridge in BRIDGES:
            per_bridge_activity[bridge] = _load_per_bridge_activity(
                bridge, seed, tag,
            )
            if verbose:
                acts = per_bridge_activity[bridge]
                first_w = next(iter(acts))
                d_act = acts[first_w].shape[1]
                m_obs = acts[first_w].shape[0]
                print(
                    "  [seed " + str(seed) + "] loaded " + bridge
                    + " (M_OBS=" + str(m_obs)
                    + " d_act=" + str(d_act)
                    + " V_bridge=" + str(len(acts)) + ")",
                    flush=True,
                )
        per_load, V, max_diff = run_one_seed_cross_bridge(
            seed=seed,
            per_bridge_activity=per_bridge_activity,
            xp=xp,
            loads=loads,
            n_trials=n_trials,
            k_vocab_per_concept=k_vocab_per_concept,
            verbose=verbose,
        )
        seed_results.append({
            "seed": seed, "V": V,
            "batched_vs_scalar_max_diff": max_diff,
            "per_load": {str(l): v for l, v in per_load.items()},
            # Verdict-shaped per-seed entry (direct input to
            # direction_5_verdict.compute_verdict).
            "verdict_entry": {
                ("L=" + str(l)): {
                    "OB": per_load[l]["order_bearing_accuracy"],
                    "OI": per_load[l]["order_invariant_accuracy"],
                }
                for l in loads
            },
        })
        if verbose:
            elapsed_seed = time.time() - t_seed
            print(
                "  [seed " + str(seed) + " done in "
                + ("%.1f" % elapsed_seed) + "s]",
                flush=True,
            )
    total_time = time.time() - t0

    # Multi-seed aggregate
    agg: Dict[int, Dict[str, Any]] = {}
    cells_str_parts = []
    for load in loads:
        obs = [
            r["per_load"][str(load)]["order_bearing_accuracy"]
            for r in seed_results
        ]
        ois = [
            r["per_load"][str(load)]["order_invariant_accuracy"]
            for r in seed_results
        ]
        ob_m = float(np.mean(obs))
        oi_m = float(np.mean(ois))
        agg[load] = {
            "order_bearing_mean": ob_m,
            "order_bearing_per_seed": obs,
            "order_invariant_mean": oi_m,
            "order_invariant_per_seed": ois,
        }
        cells_str_parts.append(("%.3f" % ob_m) + " " + ("%.3f" % oi_m))
    if verbose:
        print("\n=== MULTI-SEED AGGREGATE ===", flush=True)
        loads_header = "   ".join(
            "L=" + str(l) + " OB   OI   " for l in loads
        )
        print("            " + loads_header, flush=True)
        print("  multi-seed:  " + "   ".join(cells_str_parts), flush=True)
        print(
            "\nTotal wall-clock: " + ("%.1f" % total_time)
            + "s (backend=" + backend_name + ")",
            flush=True,
        )

    # Frozen verdict (the bar / verdict module is pre-registered).
    verdict_input = [r["verdict_entry"] for r in seed_results]
    verdict = compute_verdict_from_results(verdict_input)
    if verbose:
        print("\n=== VERDICT (frozen, pre-registered) ===", flush=True)
        print("  " + verdict, flush=True)

    return {
        "backend": backend_name, "gpu": gpu,
        "tag": tag,
        "seeds": seeds, "loads": loads,
        "bar_ob": BAR_OB, "bar_oi": BAR_OI,
        "min_seeds": _DIRECTION_5_MIN_SEEDS,
        "V": DIRECTION_5_TOTAL,
        "n_bridges": len(BRIDGES),
        "bridges": list(BRIDGES),
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "mean_centring": "per_bridge_local",
        "substrate": "hybrid_bio_brain_regions_plus_shared_sparse_pool_5bridge_ensemble",
        "readout_region": "shared_concept_pool",
        "per_seed": seed_results,
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
        "wall_clock_seconds": total_time,
    }


def compute_verdict_from_results(
    per_seed_per_load_data: List[Dict[str, Dict[str, float]]],
) -> str:
    """Public delegator to the frozen verdict module.

    The runner builds per-seed verdict-shaped entries
    ({"L=2": {"OB": ..., "OI": ...}, ...}) and passes them in via the
    list of per-seed dicts. This function simply forwards to the frozen
    direction_5_verdict.compute_verdict so the verdict module remains the
    single source of truth for the threshold logic.
    """
    return compute_verdict(per_seed_per_load_data)


def main():
    ap = argparse.ArgumentParser(
        description="Direction 5 cross-bridge parallel-matching probe "
                    "on the HYBRID sparse-distributed bio_brain_regions "
                    "5-bridge ensemble (CPU-only; operates on Task 5 "
                    "activity caches)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="use 'smoke' tag activity caches (reduced scale; numbers "
             "NOT propagated as a result)",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="seeds to probe; default [42, 43, 44]",
    )
    ap.add_argument(
        "--cache-dir", default=None,
        help="override per-bridge activity cache directory (defaults to "
             "research/findings/raw/direction_5_cache)",
    )
    ap.add_argument(
        "--out", default=None,
        help="output JSON path (default: side-by-side with this module)",
    )
    args = ap.parse_args()

    tag = "smoke" if args.smoke else "full"
    result = run_cross_bridge_probe(
        seeds=args.seeds,
        tag=tag,
        cache_dir=args.cache_dir,
        verbose=True,
    )

    out_path = args.out or os.path.join(
        _HERE, "direction_5_cross_bridge_" + tag + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nWrote " + out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
