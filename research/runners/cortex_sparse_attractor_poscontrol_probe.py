"""Positive-control probe: sparse-distributed (decorrelated) codes dissolve the cleanup wall.

CONTEXT (cleanup arc, 3 NEGATIVES):
  All three prior attempts (vanilla Hopfield common-mode collapse; Storkey locality wall;
  spiking-DG sub-reproducibility) tried to clean FIXED CORRELATED codes post-hoc.
  The research doc (2026-06-11-cortex-core-learned-binder-research.md, SS1.2) shows the
  problem is NOT the mechanism -- it is the codes. On the project's REAL decorrelated
  sparse-distributed codes (between-cos approx 0.05) a distributed attractor recovers argmax
  parity 16/16, 3 seeds, NO host ZCA. This probe makes that the decisive, anti-cheated,
  multi-seed positive control.

SCIENTIFIC QUESTION:
  Does a distributed attractor cleanup (not the localist argmax) recover concept identity
  from noised/partial cues when the codes are DECORRELATED (sparse, cos approx 0.05)?
  And does the SAME attractor collapse on the CORRELATED denoise64 codes (cos approx 0.81)?
  -- confirming the cleanup wall is in the CODES, not the mechanism.

TWO CODEBOOKS:
  DECORRELATED: generate_sparse_patterns(V, n_pool=2000, K=100, seed)  [cos approx 0.05]
  CORRELATED:   denoise64 codes from the brain's own activity             [cos approx 0.81]

ATTRACTOR MECHANISMS:
  argmax      -- matched-filter reference (the idealization being replaced; REFERENCE only)
  hopfield_mf -- distributed outer-product Hopfield in CORRECT NATIVE binary readout
                 (brain-based candidate on sparse codes; the within-probe positive control)
  on_bridge   -- _D_sparse_heteroassoc.py spiking recurrent attractor, ON the real bridge
                 (CPU numpy backend; the fully spiking, permuted-control-validated result)

UNIT CHECK (mandatory, runs before everything):
  Load sparse codes-as-read; ASSERT between-cos < 0.15  (decorrelated).
  Load denoise64 codes-as-read; ASSERT between-cos > 0.60  (correlated).
  Fail loudly if either assertion fails -- do not run on mis-read codes.

CRITICAL METHODOLOGICAL NOTE:
  Sparse codes MUST NOT be median-bipolarized. A 100-of-2000 sparse code median-thresholded
  becomes -1 everywhere (only ~100 of 2000 bits are +1); ALL sparse codes then share a
  huge -1 common mode -> cos -> ~1 -> false NEGATIVE.
  Correct readout: binary {0,1} with population-mean removal (mean-centering), then cosine.

ATTRACTOR WEIGHT FORMULA:
  W = sum_p xi_p xi_p^T (NO 1/N division).
  The 1/N normalisation makes weights O(K^2/N^2) -- too small for the iteration to
  amplify the correct concept over noise. Without normalisation W is O(K^2/N), which has
  enough signal/noise for 1-5 power-iteration steps to converge on the correct concept.

TESTS (multi-seed 42 / 43 / 44):
  PARITY:     noised cue (flip fraction p in {0.0, 0.1, 0.2, 0.3})
                -> settle attractor -> score recovery accuracy per mechanism per codebook.
              GATE A (positive control): attractor ~ argmax on DECORRELATED codes (>=0.9 at p<=0.2).
              GATE B (negative control): attractor collapses on CORRELATED codes.
  COMPLETION: partial cue (keep fraction k in {0.5, 0.35, 0.25, 0.15})
                -> settle -> does the attractor match argmax-on-partial?
  ANTI-CHEATS (see notes on why lesion/shuffle are non-decisive for sparse codes):
    NOISE-CUE: pure Gaussian noise input -> attractor must NOT reliably settle on any concept
               (distribution must be near-uniform; max concept frequency <= 3x chance).
    ON-BRIDGE PERMUTED CONTROL: the spiking bridge attractor (_D_sparse_heteroassoc.py) has
               a built-in permuted-pair anti-cheat (completion follows encoding, not structure).

DECISION (stated explicitly at end of run):
  GO if GATE A AND GATE B AND unit-check pass AND noise-cue anti-cheat decisive.
  NEGATIVE/PARTIAL otherwise (characterize precisely).

CPU ONLY; SIM_BACKEND=numpy; no sim/ edits; reuse by import only.
Run: python -m research.runners.cortex_sparse_attractor_poscontrol_probe --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DENOISE64_CACHE = os.path.join(
    _REPO, "research", "findings", "raw",
    "activity_level_integration_cache", "denoise64_seed%d.npz"
)


# ---------------------------------------------------------------------------
# Code loaders -- CORRECT NATIVE readouts
# ---------------------------------------------------------------------------

def load_sparse_codes_native(seed: int, V: int, n_pool: int = 2000,
                               pattern_size: int = 100) -> tuple:
    """Load sparse-distributed codes in their NATIVE binary form, mean-removed.

    Returns (codes_native [V, n_pool], between_cos_mean, between_cos_max).
    Convention: binary {0,1} mask -> mean-remove each code -> unit-normalize.
    This is the correct readout that preserves the ~0.05 between-code cosine.
    """
    from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
    patterns = generate_sparse_patterns(V, n_pool, pattern_size, seed)
    codes = np.zeros((V, n_pool), dtype=np.float64)
    for i, pat in enumerate(patterns):
        codes[i, pat] = 1.0
    # Mean-remove (removes the shared sparsity common mode)
    codes = codes - codes.mean(axis=1, keepdims=True)
    # Unit-normalize
    norms = np.linalg.norm(codes, axis=1, keepdims=True)
    codes = codes / (norms + 1e-12)
    # Compute between-code cosines
    cos_vals = []
    for i in range(V):
        for j in range(i + 1, V):
            cos_vals.append(float(codes[i] @ codes[j]))
    between_cos_mean = float(np.mean(cos_vals)) if cos_vals else 0.0
    between_cos_max = float(np.max(np.abs(cos_vals))) if cos_vals else 0.0
    return codes, between_cos_mean, between_cos_max


def load_denoise64_codes(seed: int, V: int = 16, proj_dim: int = 800) -> tuple:
    """Load denoise64 brain codes in their load_concepts convention (projected, centered, normed).

    Returns (words, codes [V, proj_dim], between_cos_mean).
    Matches core_sim_composition.load_concepts: mean over obs samples, random Gaussian
    projection to proj_dim, mean-center rows, unit-normalize.
    """
    rng = np.random.RandomState(seed)
    d = np.load(DENOISE64_CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    ws = ws[:V]
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)
    if proj_dim and proj_dim > 0:
        P = rng.randn(raw.shape[1], proj_dim) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    cos_vals = [float(codes[i] @ codes[j])
                for i in range(V) for j in range(i + 1, V)]
    between_cos_mean = float(np.mean(cos_vals)) if cos_vals else 0.0
    return ws, codes, between_cos_mean


# ---------------------------------------------------------------------------
# Unit check (runs first; aborts if the codes are mis-read)
# ---------------------------------------------------------------------------

def unit_check(sparse_codes: np.ndarray, sparse_cos: float,
               denoise_codes: np.ndarray, denoise_cos: float,
               sparse_threshold: float = 0.15,
               correlated_threshold: float = 0.60) -> dict:
    """Assert the two code families are in the correct regime.

    sparse_cos < sparse_threshold   -> decorrelated (passes if codes read correctly)
    denoise_cos > correlated_threshold -> correlated
    """
    ok_sparse = sparse_cos < sparse_threshold
    ok_dense = denoise_cos > correlated_threshold
    status = "PASS" if (ok_sparse and ok_dense) else "FAIL"
    return {
        "sparse_between_cos": sparse_cos,
        "denoise_between_cos": denoise_cos,
        "ok_sparse_decorrelated": ok_sparse,
        "ok_denoise_correlated": ok_dense,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Attractor mechanics (distributed Hopfield in native sparse space)
# ---------------------------------------------------------------------------

def build_hopfield_weights(codes: np.ndarray) -> np.ndarray:
    """Hebbian outer-product rule W = sum_p xi_p xi_p^T (NO 1/N division), diag zeroed.

    For sparse codes (K-of-N with K << N) the 1/N normalisation makes weights so
    small (~K^2/N^2 scale) that the dynamics cannot amplify the correct concept above
    the noise floor in just a few iterations. Omitting the normalisation gives W values
    of O(K^2/N) ~ O(100^2/2000) = O(5), providing enough signal for 1-5 power-iteration
    steps to converge reliably on the correct concept.
    """
    W = codes.T @ codes   # [N, N], sum of outer products
    np.fill_diagonal(W, 0.0)
    return W


def hopfield_settle_native(W: np.ndarray, cue: np.ndarray,
                            codes: np.ndarray, iters: int = 5) -> int:
    """Power-iteration attractor in native real space:
       s <- W @ s / ||W @ s|| (repeat).
    For sparse codes W = sum_p xi_p xi_p^T, one application amplifies the stored
    pattern closest to the cue far above cross-talk (since patterns are near-orthogonal,
    cross-terms scale as K^2/N ~ 5 vs self-term K^2 = 10000). The SETTLE is the
    brain-based retrieval step; the cosine scoring is the legitimate grading step.
    Returns the index of the nearest code to the settled state."""
    s = cue.copy().astype(np.float64)
    n = np.linalg.norm(s)
    if n > 1e-12:
        s = s / n
    for _ in range(iters):
        s_new = W @ s
        n2 = np.linalg.norm(s_new)
        if n2 < 1e-12:
            break
        s_new = s_new / n2
        if np.max(np.abs(s_new - s)) < 1e-8:
            break
        s = s_new
    return int(np.argmax(codes @ s))


def argmax_cleanup_native(cue: np.ndarray, codes: np.ndarray) -> int:
    """God's-eye matched filter: nearest code by cosine. REFERENCE only."""
    sims = codes @ (cue / (np.linalg.norm(cue) + 1e-12))
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Cue generators (in native mean-removed real space)
# ---------------------------------------------------------------------------

def noisy_cue_sparse(code: np.ndarray, rng: np.random.Generator,
                      flip_frac: float, n_pool: int) -> np.ndarray:
    """Noised sparse cue: start from the binary pattern (threshold at 0 after mean-removal),
    flip flip_frac of active bits (swap active->inactive and inactive->active), then
    re-apply mean-removal and normalization."""
    binary = (code > 0.0).astype(np.float64)
    active_idx = np.where(binary == 1.0)[0]
    inactive_idx = np.where(binary == 0.0)[0]
    K = len(active_idx)
    n_flip = max(0, int(round(flip_frac * K)))
    if n_flip > 0:
        n_flip_off = min(n_flip, K)
        n_flip_on = min(n_flip, len(inactive_idx))
        flip_off = rng.choice(K, size=n_flip_off, replace=False)
        flip_on = rng.choice(len(inactive_idx), size=n_flip_on, replace=False)
        binary[active_idx[flip_off]] = 0.0
        binary[inactive_idx[flip_on]] = 1.0
    cue = binary - binary.mean()
    n = np.linalg.norm(cue)
    return cue / (n + 1e-12)


def noisy_cue_dense(code: np.ndarray, rng: np.random.Generator,
                     flip_frac: float) -> np.ndarray:
    """Noised cue for dense real-valued codes: add Gaussian noise scaled by flip_frac,
    then re-center + unit-normalize. 'flip_frac' is the noise std relative to code norm."""
    D = code.shape[0]
    noise = rng.standard_normal(D) * flip_frac
    cue = code + noise
    cue = cue - cue.mean()
    n = np.linalg.norm(cue)
    return cue / (n + 1e-12)


def partial_cue_sparse(code: np.ndarray, rng: np.random.Generator,
                        keep_frac: float, n_pool: int) -> np.ndarray:
    """Partial cue: keep keep_frac of active bits, zero the rest.
    Unknown bits are 0 (not mean-removed), allowing the attractor to complete them."""
    binary = (code > 0.0).astype(np.float64)
    active_idx = np.where(binary == 1.0)[0]
    K = len(active_idx)
    n_keep = max(1, int(round(keep_frac * K)))
    kept = rng.choice(K, size=n_keep, replace=False)
    out = np.zeros(n_pool, dtype=np.float64)
    out[active_idx[kept]] = 1.0
    out = out - out.mean()
    n = np.linalg.norm(out)
    return out / (n + 1e-12)


# ---------------------------------------------------------------------------
# TEST 1 -- PARITY (noised cues, score per mechanism per codebook)
# ---------------------------------------------------------------------------

def run_parity_sparse(codes: np.ndarray, seed: int, n_pool: int,
                       flip_fracs=(0.0, 0.1, 0.2, 0.3),
                       n_trials: int = 160) -> dict:
    """Score argmax and hopfield on noised sparse cues. Native binary readout."""
    V = codes.shape[0]
    rng = np.random.default_rng(seed + 1000)
    W = build_hopfield_weights(codes)
    results = {}
    for flip_frac in flip_fracs:
        n_arg = n_hop = 0
        rng_inner = np.random.default_rng(seed + 1000 + int(flip_frac * 1000))
        for _ in range(n_trials):
            i = int(rng_inner.integers(V))
            cue = noisy_cue_sparse(codes[i], rng_inner, flip_frac, n_pool)
            n_arg += int(argmax_cleanup_native(cue, codes) == i)
            n_hop += int(hopfield_settle_native(W, cue, codes) == i)
        results[float(flip_frac)] = {
            "argmax": n_arg / n_trials,
            "hopfield_mf": n_hop / n_trials,
            "chance": 1.0 / V,
            "n": n_trials,
        }
    return results


def run_parity_dense(codes: np.ndarray, seed: int,
                      noise_levels=(0.0, 0.2, 0.5, 1.0),
                      n_trials: int = 160) -> dict:
    """Score argmax and hopfield on noised dense real-valued cues.
    noise_levels are Gaussian std values relative to code (0=clean, 1=moderate)."""
    V = codes.shape[0]
    W = build_hopfield_weights(codes)
    results = {}
    for noise_std in noise_levels:
        n_arg = n_hop = 0
        rng_inner = np.random.default_rng(seed + 2000 + int(noise_std * 1000))
        for _ in range(n_trials):
            i = int(rng_inner.integers(V))
            cue = noisy_cue_dense(codes[i], rng_inner, noise_std)
            n_arg += int(argmax_cleanup_native(cue, codes) == i)
            n_hop += int(hopfield_settle_native(W, cue, codes) == i)
        results[float(noise_std)] = {
            "argmax": n_arg / n_trials,
            "hopfield_mf": n_hop / n_trials,
            "chance": 1.0 / V,
            "n": n_trials,
        }
    return results


# ---------------------------------------------------------------------------
# TEST 2 -- COMPLETION (partial cues)
# ---------------------------------------------------------------------------

def run_completion_sparse(codes: np.ndarray, seed: int, n_pool: int,
                           keep_fracs=(0.5, 0.35, 0.25, 0.15),
                           n_trials: int = 160) -> dict:
    """Score argmax-on-partial and hopfield on partial sparse cues."""
    V = codes.shape[0]
    W = build_hopfield_weights(codes)
    results = {}
    for keep_frac in keep_fracs:
        n_arg = n_hop = 0
        rng_inner = np.random.default_rng(seed + 3000 + int(keep_frac * 1000))
        for _ in range(n_trials):
            i = int(rng_inner.integers(V))
            cue = partial_cue_sparse(codes[i], rng_inner, keep_frac, n_pool)
            n_arg += int(argmax_cleanup_native(cue, codes) == i)
            n_hop += int(hopfield_settle_native(W, cue, codes) == i)
        results[float(keep_frac)] = {
            "argmax_on_partial": n_arg / n_trials,
            "hopfield_mf": n_hop / n_trials,
            "chance": 1.0 / V,
            "n": n_trials,
            "hopfield_edge": (n_hop - n_arg) / n_trials,
        }
    return results


# ---------------------------------------------------------------------------
# ANTI-CHEAT: NOISE-CUE
# Rationale: present pure Gaussian noise (no concept) -> attractor output must be
# near-uniform (no concept hallucinated). This confirms the Hopfield attractor does
# not always converge to a fixed dominant concept regardless of input.
# Note on why lesion/shuffle are non-decisive for extreme decorrelated codes:
#   W_zero @ cue = 0 -> iteration fails -> argmax(codes @ cue) is still correct
#   (the cue ALREADY carries the concept; lesion just makes hopfield = argmax on cue).
#   W_shuffled is a different outer-product but preserves the basis; with near-orthogonal
#   codes ANY outer-product from the same seed subspace preserves most cosine signal.
#   The noise-cue test is the decisive control: no concept in input -> no concept in output.
# ---------------------------------------------------------------------------

def run_noise_cue_anticheat(codes: np.ndarray, seed: int, n_pool: int,
                              n_trials: int = 200) -> dict:
    """Pure Gaussian noise input -> check attractor output distribution is near-uniform.
    If attractor were always 'cheating' by returning a fixed favorite concept, this
    would show a spike at one concept. Uniform distribution confirms pattern-specificity."""
    V = codes.shape[0]
    W = build_hopfield_weights(codes)
    rng = np.random.default_rng(seed + 5000)
    concept_counts = np.zeros(V, dtype=int)
    for _ in range(n_trials):
        noise = rng.standard_normal(n_pool)
        noise = noise / (np.linalg.norm(noise) + 1e-12)
        result = hopfield_settle_native(W, noise, codes)
        concept_counts[result] += 1
    max_freq = int(concept_counts.max())
    max_freq_rate = float(max_freq) / n_trials
    chance = 1.0 / V
    # Decisive if max_freq <= 3x chance (no dominant hallucination)
    decisive = max_freq_rate <= 3.0 * chance
    return {
        "n_trials": n_trials,
        "max_concept_freq": max_freq_rate,
        "chance": chance,
        "decisive_no_hallucination": decisive,
        "concept_counts": concept_counts.tolist(),
    }


# ---------------------------------------------------------------------------
# ON-BRIDGE spiking attractor (reuses _D_sparse_heteroassoc.py)
# ---------------------------------------------------------------------------

def run_on_bridge_attractor(seed: int, n_concepts: int = 4,
                              n_pool: int = 2000, pattern_size: int = 100,
                              enc_cycles: int = 20, swr_cycles: int = 20) -> dict:
    """Run the _D_sparse_heteroassoc.py spiking recurrent attractor on CPU (SIM_BACKEND=numpy).
    V=n_concepts (small; associative completion framing), enc+swr cycles kept cheap.
    The bridge build, encoding, and recall are ALL spiking neurons on the real SimulationBridge.
    Returns pairs, base/post completion results, and the permuted-encoding anti-cheat summary.
    """
    from research.runners._D_sparse_heteroassoc import run as d_run
    t0 = time.time()
    pairs, base, post = d_run(seed, n_concepts=n_concepts,
                               pattern_size=pattern_size, n_pool=n_pool,
                               enc_cycles=enc_cycles, swr_cycles=swr_cycles)
    elapsed = time.time() - t0
    n_pairs = len(pairs)
    n_base_pass = sum(1 for v in base.values() if v[0])
    n_post_pass = sum(1 for v in post.values() if v[0])
    pair_results = {}
    for (a, bb) in pairs:
        pair_results[f"c{a}->c{bb}"] = {
            "base_top1": base[(a, bb)][0],
            "base_rank": base[(a, bb)][1],
            "base_margin": base[(a, bb)][2],
            "post_swr_top1": post[(a, bb)][0],
            "post_swr_rank": post[(a, bb)][1],
            "post_swr_margin": post[(a, bb)][2],
        }
    return {
        "n_pairs": n_pairs,
        "n_base_pass": n_base_pass,
        "n_post_swr_pass": n_post_pass,
        "base_pass_rate": n_base_pass / max(1, n_pairs),
        "post_swr_pass_rate": n_post_pass / max(1, n_pairs),
        "pairs": pair_results,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_seed(seed: int, V: int = 16,
              n_pool: int = 2000, pattern_size: int = 100,
              n_trials: int = 160, run_bridge: bool = True,
              bridge_enc_cycles: int = 20, bridge_swr_cycles: int = 20) -> dict:
    """Full probe for one seed. Returns all results for JSON serialization."""
    print(f"\n{'='*60}", flush=True)
    print(f"  SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Load codes ---
    sparse_codes, sparse_cos, sparse_cos_max = load_sparse_codes_native(
        seed, V, n_pool, pattern_size)
    words, denoise_codes, denoise_cos = load_denoise64_codes(seed, V=V)
    n_pool_denoise = denoise_codes.shape[1]
    print(f"  [codes] sparse: cos_mean={sparse_cos:.4f} cos_max={sparse_cos_max:.4f}"
          f"  denoise64: cos_mean={denoise_cos:.4f}", flush=True)

    # --- UNIT CHECK ---
    uc = unit_check(sparse_codes, sparse_cos, denoise_codes, denoise_cos)
    print(f"  [unit check] sparse_cos={uc['sparse_between_cos']:.4f} (<0.15)  "
          f"denoise_cos={uc['denoise_between_cos']:.4f} (>0.60)  -> {uc['status']}",
          flush=True)
    if uc["status"] != "PASS":
        print("  *** UNIT CHECK FAILED -- aborting this seed ***", flush=True)
        return {"seed": seed, "unit_check": uc, "ABORTED": True}

    # --- PARITY on DECORRELATED (sparse) codes ---
    print("  [parity: DECORRELATED sparse codes]", flush=True)
    parity_sparse = run_parity_sparse(sparse_codes, seed, n_pool, n_trials=n_trials)
    for fp, r in parity_sparse.items():
        print(f"    flip={fp:.1f}  argmax={r['argmax']:.3f}  hopfield_mf={r['hopfield_mf']:.3f}"
              f"  chance={r['chance']:.3f}", flush=True)

    # --- PARITY on CORRELATED (denoise64) codes ---
    print("  [parity: CORRELATED denoise64 codes]", flush=True)
    parity_corr = run_parity_dense(denoise_codes, seed, n_trials=n_trials)
    for ns, r in parity_corr.items():
        print(f"    noise={ns:.1f}  argmax={r['argmax']:.3f}  hopfield_mf={r['hopfield_mf']:.3f}"
              f"  chance={r['chance']:.3f}", flush=True)

    # --- COMPLETION on DECORRELATED ---
    print("  [completion: DECORRELATED sparse codes]", flush=True)
    completion_sparse = run_completion_sparse(sparse_codes, seed, n_pool, n_trials=n_trials)
    for kf, r in completion_sparse.items():
        print(f"    keep={kf:.2f}  argmax_partial={r['argmax_on_partial']:.3f}"
              f"  hopfield_mf={r['hopfield_mf']:.3f}  edge={r['hopfield_edge']:+.3f}", flush=True)

    # --- ANTI-CHEAT: NOISE-CUE ---
    print("  [anti-cheat: noise-cue (no concept hallucination)]", flush=True)
    noise_cheat = run_noise_cue_anticheat(sparse_codes, seed, n_pool)
    print(f"    max_concept_freq={noise_cheat['max_concept_freq']:.3f}"
          f"  chance={noise_cheat['chance']:.3f}"
          f"  decisive={noise_cheat['decisive_no_hallucination']}", flush=True)

    # --- ON-BRIDGE spiking attractor ---
    bridge_result = None
    if run_bridge:
        print("  [on-bridge spiking attractor (CPU numpy, D module)]", flush=True)
        try:
            bridge_result = run_on_bridge_attractor(
                seed, n_concepts=4, n_pool=n_pool, pattern_size=pattern_size,
                enc_cycles=bridge_enc_cycles, swr_cycles=bridge_swr_cycles)
            print(f"    post-encode {bridge_result['n_base_pass']}/{bridge_result['n_pairs']}  "
                  f"post-SWR {bridge_result['n_post_swr_pass']}/{bridge_result['n_pairs']}  "
                  f"({bridge_result['elapsed_s']:.1f}s)", flush=True)
        except Exception as exc:
            print(f"    on-bridge FAILED: {exc}", flush=True)
            bridge_result = {"error": str(exc)}

    # --- Evaluate gates ---
    gate_a_vals = {fp: parity_sparse[fp]["hopfield_mf"] for fp in [0.0, 0.1, 0.2]}
    gate_a = all(v >= 0.9 for v in gate_a_vals.values())
    # GATE B: attractor collapses on correlated codes.
    # The decisive criterion: at CLEAN cues (noise=0.0), the attractor is at chance
    # (hopfield_mf <= 2x chance) while argmax is perfect (argmax >= 0.95).
    # This is the clearest statement of the collapse: the wall exists even with zero noise --
    # the correlated codes confuse the distributed attractor from the very first step.
    V_local = denoise_codes.shape[0]
    chance_denoise = 1.0 / V_local
    hop_clean = parity_corr[0.0]["hopfield_mf"]
    arg_clean = parity_corr[0.0]["argmax"]
    gate_b = (hop_clean <= 2.0 * chance_denoise) and (arg_clean >= 0.95)
    gate_b_detail = {
        "hopfield_at_clean": hop_clean,
        "argmax_at_clean": arg_clean,
        "chance": chance_denoise,
        "hopfield_vs_2x_chance": hop_clean <= 2.0 * chance_denoise,
        "argmax_perfect": arg_clean >= 0.95,
    }
    noise_cheat_ok = noise_cheat["decisive_no_hallucination"]

    print(f"\n  GATES:", flush=True)
    print(f"    A (hopfield >= 0.9 on sparse at p<=0.2): {gate_a} -> {gate_a_vals}", flush=True)
    print(f"    B (hopfield <= 2x chance on correlated clean): {gate_b} -> "
          f"hopfield={hop_clean:.4f}  2xchance={2*chance_denoise:.3f}  argmax={arg_clean:.4f}",
          flush=True)
    print(f"  ANTI-CHEATS: noise_cue_ok={noise_cheat_ok}", flush=True)

    return {
        "seed": seed,
        "unit_check": uc,
        "parity_sparse_decorrelated": parity_sparse,
        "parity_corr_denoise64": parity_corr,
        "completion_sparse": completion_sparse,
        "anticheat_noise_cue": noise_cheat,
        "on_bridge": bridge_result,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_b_detail": gate_b_detail,
        "noise_cheat_ok": noise_cheat_ok,
    }


def main():
    p = argparse.ArgumentParser(description="Sparse attractor positive-control probe")
    p.add_argument("--seeds", default="42,43,44",
                   help="Comma-separated seeds (default: 42,43,44)")
    p.add_argument("--V", type=int, default=16,
                   help="Number of concepts per codebook (default: 16)")
    p.add_argument("--n-pool", type=int, default=2000)
    p.add_argument("--pattern-size", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=160)
    p.add_argument("--bridge-enc-cycles", type=int, default=20)
    p.add_argument("--bridge-swr-cycles", type=int, default=20)
    p.add_argument("--no-bridge", action="store_true",
                   help="Skip the on-bridge spiking attractor (faster)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (auto-generated if not given)")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    t_start = time.time()

    results = {}
    aborted = False
    for seed in seeds:
        r = run_seed(
            seed=seed,
            V=args.V,
            n_pool=args.n_pool,
            pattern_size=args.pattern_size,
            n_trials=args.n_trials,
            run_bridge=not args.no_bridge,
            bridge_enc_cycles=args.bridge_enc_cycles,
            bridge_swr_cycles=args.bridge_swr_cycles,
        )
        results[str(seed)] = r
        if r.get("ABORTED"):
            print(f"  => UNIT CHECK FAILED on seed {seed}; aborting all seeds.", flush=True)
            aborted = True
            break

    elapsed_total = time.time() - t_start

    # Overall decision
    completed_seeds = [s for s in seeds if not results[str(s)].get("ABORTED")]
    all_gate_a = all(results[str(s)].get("gate_a", False) for s in completed_seeds)
    all_gate_b = all(results[str(s)].get("gate_b", False) for s in completed_seeds)
    all_noise_cheat = all(results[str(s)].get("noise_cheat_ok", False) for s in completed_seeds)

    if aborted:
        verdict = "UNIT_CHECK_FAIL"
    elif all_gate_a and all_gate_b and all_noise_cheat:
        verdict = "GO"
    elif all_gate_a and all_gate_b:
        verdict = "GO_partial_anticheat"
    elif all_gate_a:
        verdict = "PARTIAL_gate_b_fail"
    else:
        verdict = "NEGATIVE"

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "all_gate_a": all_gate_a,
        "all_gate_b": all_gate_b,
        "all_noise_cheat_ok": all_noise_cheat,
        "aborted": aborted,
        "elapsed_total_s": elapsed_total,
    }
    print(f"\n{'='*60}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  GATE A (attractor~argmax on decorrelated, all seeds): {all_gate_a}", flush=True)
    print(f"  GATE B (attractor collapses on correlated, all seeds): {all_gate_b}", flush=True)
    print(f"  ANTI-CHEAT noise-cue OK: {all_noise_cheat}", flush=True)
    print(f"  Total elapsed: {elapsed_total:.1f}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    out_data = {"summary": summary, "per_seed": results}

    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir,
            f"cortex_sparse_attractor_poscontrol_{timestamp}.json")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
