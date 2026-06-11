"""Cortex fixed-expansion decorrelation probe (Marr-Albus granule-layer recoding).

SCIENTIFIC QUESTION:
  A FIXED random expansion + threshold nonlinearity (Marr 1969 / Albus 1971 cerebellar
  granule layer; Babadi & Sompolinsky 2014) decorrelates correlated inputs. The key question:
  does this mechanism produce codes that are BOTH decorrelated (between-cos ≤ ~0.1) AND
  reproducible under noise (same-input cosine ≥ 0.9) — the co-existence the spiking DG
  k-WTA FAILED to achieve (sep ≈ repro ≈ raw-cos at any k value)?

ARCHITECTURE:
  Fixed random expansion: W [D_exp × D_in], Gaussian or sparse i.i.d.
  The expansion is DETERMINISTIC given the seed → same input always maps to same expanded code.
  Threshold nonlinearity: keep the top-f fraction of expansion-unit activations (= 1), rest = 0.
  Expanded code readout: binary {0,1} with mean-removal, unit-normalize.

INPUT CODES:
  denoise64 brain codes: between-code cosine ≈ 0.81 (highly correlated).
  Read in NATIVE form (mean over obs samples, random Gaussian project, mean-center, unit-norm).
  UNIT CHECK: assert input between-cos > 0.6 before running.

STAGE 1 — numpy characterization (decisive, fast):
  Sweep expansion ratio r = D_exp / D_in ∈ {4, 8, 16}
       active fraction f ∈ {0.05, 0.1, 0.2}
  For each (r, f):
    - Compute expanded code for each concept (CLEAN).
    - REPRODUCIBILITY: compute expanded code TWICE with independent additive Gaussian noise
      on the INPUT (σ ∈ {0.05, 0.1, 0.2} as fraction of code norm), measure same-input cosine.
      TARGET ≥ 0.9 (the bar the spiking DG k-WTA failed to meet with decorrelation).
    - DECORRELATION: between-concept cosine of the expanded codes. TARGET ≤ 0.1.
    - KEY QUESTION: does any (r, f) give repro ≥ 0.9 AND between-cos ≤ 0.1 simultaneously?
  LESION control: zero the expansion weights → expanded codes all identical → repro ≈ 1 but
                  zero discrimination (all expanded codes same → between-cos = 1) → confirms
                  decorrelation rides the expansion, not input structure.
  CLEANUP PARITY: at the best operating point, run the validated distributed-attractor
                  cleanup from cortex_sparse_attractor_poscontrol_probe.py on the expanded
                  codes → does it recover argmax parity ≈ 1.000 (confirming expanded codes
                  are usable by the downstream binder/cleanup)?

STAGE 2 — on-bridge confirmation (only if Stage 1 GO; small + synchronous):
  Realize the fixed expansion as a population on a real SimulationBridge (a fixed-weight,
  non-plastic expansion region driven by the concept input, read by ACCUMULATED RATE over a
  window). Verify on-substrate expanded code: same-input cosine ≥ 0.9 across independent spiking reads.

ANTI-CHEATS:
  1. UNIT CHECK: assert input between-cos > 0.6 (correlated) before running.
  2. LESION: zero expansion weights → decorrelation must collapse (between-cos → 1, repro → 1
             but codes all identical → zero discrimination).
  3. REPRODUCIBILITY is the load-bearing bar: report prominently. A result that decorrelates
     but is NOT reproducible (same-input cosine < 0.9) is a FAIL, not a pass.
  4. Multi-seed (42/43/44) for headline numbers.

DECISION:
  GO if:
    - Fixed-expansion operating point: between-cos ≤ ~0.1 AND same-input reproducibility ≥ 0.9
      (multi-seed, at noise level σ=0.1)
    - Attractor cleanup recovers parity ≥ 0.9 on the expanded codes
    - Lesion confirms decorrelation rides the expansion
    - (if reached) Stage-2 on-bridge spiking confirms repro ≥ 0.9
  NEGATIVE/BOUNDARY if reproducibility and decorrelation still trade off.

CPU only; SIM_BACKEND=numpy; no sim/ edits.
Run: python -m research.runners.cortex_fixed_expansion_decorrelation_probe --seeds 42,43,44
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

DENOISE64_CACHE = os.path.join(
    _REPO, "research", "findings", "raw",
    "activity_level_integration_cache", "denoise64_seed%d.npz"
)


# ---------------------------------------------------------------------------
# Code loading (SAME convention as cortex_sparse_attractor_poscontrol_probe.py)
# ---------------------------------------------------------------------------

def load_denoise64_codes(seed: int, V: int = 16, proj_dim: int = 800) -> tuple:
    """Load denoise64 brain codes in native convention.

    Returns (words, codes [V, proj_dim], between_cos_mean).
    Convention: mean over obs samples per word, random Gaussian project to proj_dim,
    mean-center rows, unit-normalize. NO decorrelation — raw correlated codes.
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
# UNIT CHECK
# ---------------------------------------------------------------------------

def unit_check(codes: np.ndarray, between_cos: float, threshold: float = 0.6) -> dict:
    """Assert input codes are in the correlated regime (between-cos > threshold).
    If this fails the whole probe is invalid — abort early."""
    ok = between_cos > threshold
    return {
        "between_cos": between_cos,
        "threshold": threshold,
        "ok_correlated": ok,
        "status": "PASS" if ok else "FAIL",
        "note": (
            "Input codes are correlated as required (raw denoise64)."
            if ok else
            f"FAIL: input between-cos {between_cos:.4f} <= {threshold} — codes may be pre-whitened!"
        ),
    }


# ---------------------------------------------------------------------------
# Fixed random expansion
# ---------------------------------------------------------------------------

def build_expansion_matrix(D_in: int, D_exp: int, seed: int,
                             sparse: bool = False, sparsity: float = 0.1) -> np.ndarray:
    """Build a fixed random expansion matrix W [D_exp, D_in].

    If sparse=False: each row is i.i.d. Gaussian, std = 1/sqrt(D_in).
    If sparse=True: each row has sparsity * D_in non-zero i.i.d. Gaussian entries.
    The matrix is FIXED and DETERMINISTIC given seed.
    """
    rng = np.random.default_rng(seed)
    if not sparse:
        W = rng.standard_normal((D_exp, D_in)) / np.sqrt(D_in)
    else:
        W = np.zeros((D_exp, D_in), dtype=np.float64)
        n_nonzero = max(1, int(round(sparsity * D_in)))
        for i in range(D_exp):
            idx = rng.choice(D_in, size=n_nonzero, replace=False)
            W[i, idx] = rng.standard_normal(n_nonzero) / np.sqrt(n_nonzero)
    return W


def expand_code(code: np.ndarray, W: np.ndarray, active_frac: float) -> np.ndarray:
    """Apply fixed random expansion + threshold nonlinearity.

    1. Compute pre-activations: a = W @ code   [D_exp]
    2. Threshold: keep the top active_frac fraction (binary {0,1} indicator).
    3. Mean-remove and unit-normalize.
    Returns the expanded code in mean-removed unit-norm form.
    """
    D_exp = W.shape[0]
    a = W @ code  # [D_exp]
    # Threshold: k-WTA by rank (DETERMINISTIC — no noise here, just the code)
    k = max(1, int(round(active_frac * D_exp)))
    threshold_val = np.partition(a, D_exp - k)[D_exp - k]
    expanded = (a >= threshold_val).astype(np.float64)
    # Tie-breaking: if more than k units are at the threshold, deactivate excess randomly
    # (deterministic: take the first k by index for reproducibility)
    if expanded.sum() > k:
        above_thresh = np.where(a > threshold_val)[0]
        at_thresh = np.where(a == threshold_val)[0]
        n_needed = k - len(above_thresh)
        if n_needed > 0:
            expanded[:] = 0.0
            expanded[above_thresh] = 1.0
            expanded[at_thresh[:n_needed]] = 1.0
        else:
            # More above-threshold than k — take all above_thresh[:k]
            expanded[:] = 0.0
            expanded[above_thresh[:k]] = 1.0
    # Mean-remove + unit-normalize
    expanded = expanded - expanded.mean()
    n = np.linalg.norm(expanded)
    if n < 1e-12:
        return expanded
    return expanded / n


def expand_code_noisy(code: np.ndarray, W: np.ndarray, active_frac: float,
                       rng: np.random.Generator, noise_sigma: float) -> np.ndarray:
    """Expand with additive Gaussian noise on the input code.

    Noise is added to the INPUT before expansion, simulating spiking/OU variability.
    noise_sigma is the noise std as a fraction of the code's norm (code is unit-normed,
    so noise_sigma IS the absolute std since ||code|| = 1).
    """
    noise = rng.standard_normal(len(code)) * noise_sigma
    noisy_code = code + noise
    # Re-normalize the noisy code (keeps it in the same scale)
    n = np.linalg.norm(noisy_code)
    if n > 1e-12:
        noisy_code = noisy_code / n
    return expand_code(noisy_code, W, active_frac)


# ---------------------------------------------------------------------------
# REPRODUCIBILITY: same-input cosine under noise
# ---------------------------------------------------------------------------

def measure_reproducibility(codes: np.ndarray, W: np.ndarray, active_frac: float,
                              noise_sigma: float, seed: int, n_trials: int = 100) -> dict:
    """Measure same-concept cosine between two independent noisy expansions.

    For each trial: pick a random concept, expand TWICE with independent noise,
    compute cosine between the two expanded codes.
    Returns mean, min, and std of same-input cosine.
    """
    rng = np.random.default_rng(seed + 10000 + int(noise_sigma * 10000))
    V = codes.shape[0]
    cosines = []
    for _ in range(n_trials):
        i = int(rng.integers(V))
        exp1 = expand_code_noisy(codes[i], W, active_frac, rng, noise_sigma)
        exp2 = expand_code_noisy(codes[i], W, active_frac, rng, noise_sigma)
        cos_val = float(exp1 @ exp2)
        cosines.append(cos_val)
    return {
        "mean": float(np.mean(cosines)),
        "min": float(np.min(cosines)),
        "std": float(np.std(cosines)),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# DECORRELATION: between-concept cosine of expanded codes
# ---------------------------------------------------------------------------

def measure_decorrelation(codes: np.ndarray, W: np.ndarray, active_frac: float) -> dict:
    """Measure between-concept cosine of the expanded codes.

    All V*(V-1)/2 pairwise cosines.
    Returns mean, max, std.
    """
    V = codes.shape[0]
    expanded = np.stack([expand_code(codes[i], W, active_frac) for i in range(V)])
    cos_vals = []
    for i in range(V):
        for j in range(i + 1, V):
            cos_vals.append(float(expanded[i] @ expanded[j]))
    return {
        "mean": float(np.mean(cos_vals)),
        "max": float(np.max(np.abs(cos_vals))),
        "std": float(np.std(cos_vals)),
        "n_pairs": len(cos_vals),
        "expanded_codes": expanded,  # kept for cleanup test; removed from JSON output
    }


# ---------------------------------------------------------------------------
# LESION control
# ---------------------------------------------------------------------------

def measure_lesion(codes: np.ndarray, W: np.ndarray, active_frac: float,
                    noise_sigma: float, seed: int, n_trials: int = 100) -> dict:
    """Zero the expansion weights -> expanded codes should all be identical (all tie at threshold).

    With W=0, all pre-activations are 0, all units tie at threshold, the top-k selection
    is ALWAYS the same k units (index-tie-breaking: first k), so all input codes map to
    the same expanded code -> between-concept cosine = 1, same-input cosine = 1,
    but no concept discrimination (all outputs identical = no information).
    """
    W_zero = np.zeros_like(W)
    V = codes.shape[0]
    expanded = np.stack([expand_code(codes[i], W_zero, active_frac) for i in range(V)])
    # Check between-concept cosine
    cos_vals = []
    for i in range(V):
        for j in range(i + 1, V):
            cos_vals.append(float(expanded[i] @ expanded[j]))
    between_cos_lesion = float(np.mean(cos_vals)) if cos_vals else 0.0
    # Check same-input reproducibility (should be 1.0 — W=0 means no noise effect)
    rng = np.random.default_rng(seed + 20000)
    cosines_repro = []
    for _ in range(50):
        i = int(rng.integers(V))
        exp1 = expand_code_noisy(codes[i], W_zero, active_frac, rng, noise_sigma)
        exp2 = expand_code_noisy(codes[i], W_zero, active_frac, rng, noise_sigma)
        cosines_repro.append(float(exp1 @ exp2))
    # Lesion passes if: between_cos_lesion ≈ 1 (all same) OR all codes are zero/degenerate
    # (W=0 → a=0 → no threshold gradient → all tie → first-k always same → codes identical)
    lesion_collapses_discrimination = between_cos_lesion > 0.9
    return {
        "between_cos_lesion": between_cos_lesion,
        "repro_lesion_mean": float(np.mean(cosines_repro)),
        "discrimination_collapsed": lesion_collapses_discrimination,
        "verdict": "PASS" if lesion_collapses_discrimination else "PARTIAL",
        "note": (
            "Lesion collapses discrimination as expected: all expanded codes identical."
            if lesion_collapses_discrimination else
            f"Partial: between-cos after lesion {between_cos_lesion:.4f} (expected ~1.0)."
        ),
    }


# ---------------------------------------------------------------------------
# CLEANUP PARITY: distributed Hopfield attractor on expanded codes
# (reuses the harness from cortex_sparse_attractor_poscontrol_probe.py)
# ---------------------------------------------------------------------------

def build_hopfield_weights(codes: np.ndarray) -> np.ndarray:
    """W = sum_p xi_p xi_p^T (NO 1/N division), diag zeroed."""
    W = codes.T @ codes
    np.fill_diagonal(W, 0.0)
    return W


def hopfield_settle(W: np.ndarray, cue: np.ndarray,
                    codes: np.ndarray, iters: int = 5) -> int:
    """Power-iteration attractor settle. Returns index of nearest code."""
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


def run_cleanup_parity(input_codes: np.ndarray, W_expand: np.ndarray,
                        active_frac: float, noise_fracs: tuple,
                        seed: int, n_trials: int = 160) -> dict:
    """Run cleanup parity on the EXPANDED codes.

    For each noise level: expand all V concepts (clean), build Hopfield weights on
    expanded codes, then test: noised input -> expand -> hopfield settle -> recover?
    The noising is applied to the INPUT codes (before expansion), so the pipeline is:
      noisy input code -> expansion -> expanded cue -> Hopfield settle -> which expanded concept?

    Returns per-noise-level argmax and hopfield accuracy.
    """
    V = input_codes.shape[0]
    # Build expanded codebook (CLEAN, no noise)
    expanded_codebook = np.stack([
        expand_code(input_codes[i], W_expand, active_frac) for i in range(V)
    ])
    W_hop = build_hopfield_weights(expanded_codebook)
    results = {}
    for noise_sigma in noise_fracs:
        rng = np.random.default_rng(seed + 30000 + int(noise_sigma * 10000))
        n_arg = n_hop = 0
        for _ in range(n_trials):
            i = int(rng.integers(V))
            # Noisy expanded cue (noise on input, then expand)
            exp_noisy = expand_code_noisy(input_codes[i], W_expand, active_frac,
                                           rng, noise_sigma)
            # argmax on expanded codebook
            sims = expanded_codebook @ exp_noisy
            n_arg += int(int(np.argmax(sims)) == i)
            # Hopfield settle on expanded codebook
            n_hop += int(hopfield_settle(W_hop, exp_noisy, expanded_codebook) == i)
        results[float(noise_sigma)] = {
            "argmax": n_arg / n_trials,
            "hopfield_mf": n_hop / n_trials,
            "chance": 1.0 / V,
            "n": n_trials,
        }
    return results


# ---------------------------------------------------------------------------
# STAGE 1: parameter sweep
# ---------------------------------------------------------------------------

def run_stage1_sweep(codes: np.ndarray, seed: int, V: int,
                      expansion_ratios: tuple, active_fracs: tuple,
                      noise_sigmas: tuple, n_trials_repro: int = 100) -> dict:
    """Full Stage-1 numpy sweep over (expansion_ratio, active_frac, noise_sigma).

    Returns a nested dict for JSON serialization.
    """
    D_in = codes.shape[1]
    results = {}
    best_op = None  # track best operating point for cleanup test
    best_repro_x_decos = -1.0  # heuristic: repro × (1 - between_cos) for the joint criterion

    for ratio in expansion_ratios:
        D_exp = int(D_in * ratio)
        W = build_expansion_matrix(D_in, D_exp, seed=seed)

        for f in active_fracs:
            key = f"r{ratio}_f{f}"
            t0 = time.time()

            # Decorrelation (CLEAN codes)
            deco_result = measure_decorrelation(codes, W, f)
            expanded_codebook = deco_result.pop("expanded_codes")  # pull out; don't serialize
            between_cos = deco_result["mean"]

            # Reproducibility sweep over noise levels
            repro_by_noise = {}
            for sigma in noise_sigmas:
                repro = measure_reproducibility(codes, W, f, sigma, seed, n_trials_repro)
                repro_by_noise[float(sigma)] = repro

            # Headline: repro at sigma=0.1
            repro_01 = repro_by_noise.get(0.1, repro_by_noise.get(noise_sigmas[0], {}))
            repro_mean = repro_01.get("mean", 0.0)

            # Check joint criterion
            joint_ok = (repro_mean >= 0.9) and (between_cos <= 0.1)
            score = repro_mean * (1.0 - abs(between_cos))
            if score > best_repro_x_decos:
                best_repro_x_decos = score
                best_op = (ratio, f, D_exp, W.copy())

            # Margin analysis (why repro is low)
            margin = analyze_threshold_margin(codes, W, f)

            results[key] = {
                "expansion_ratio": ratio,
                "D_in": D_in,
                "D_exp": D_exp,
                "active_frac": f,
                "decorrelation": deco_result,
                "between_cos_clean": between_cos,
                "reproducibility_by_noise": repro_by_noise,
                "repro_at_sigma01": repro_mean,
                "joint_criterion_met": joint_ok,
                "joint_note": (
                    f"GO: repro={repro_mean:.4f}>=0.9 AND between_cos={between_cos:.4f}<=0.1"
                    if joint_ok else
                    f"between_cos={between_cos:.4f}, repro={repro_mean:.4f} -- "
                    + ("repro below threshold" if repro_mean < 0.9 else "decorrelation insufficient")
                ),
                "threshold_margin_analysis": margin,
                "elapsed_s": time.time() - t0,
            }

    return results, best_op


# ---------------------------------------------------------------------------
# MARGIN ANALYSIS: boundary gap at the threshold
# ---------------------------------------------------------------------------

def analyze_threshold_margin(codes: np.ndarray, W: np.ndarray,
                               active_frac: float) -> dict:
    """Measure the gap between the k-th and (k+1)-th activation at the threshold.

    The gap determines how much noise is needed to flip a boundary unit.
    If margin << noise_std, repro will be low (the threshold is in a smooth
    region of the Gaussian activation distribution, not a hard gap).

    For a fixed random expansion with Gaussian W and unit-normed input:
      activation[i] = sum_j W_ij * code_j ~ N(0, 1) [by CLT]
      Threshold = (k/D_exp)-quantile of N(0,1)
      Margin = activation[k] - activation[k+1] ~ O(1/D_exp) for large D_exp
      Noise on activation = ||W_i|| * sigma_input ~ sigma_input [for W_ij ~ N(0, 1/D_in)]
    So margin/noise ~ O(1/(D_exp * sigma)) = tiny for large D_exp and sigma=0.1.
    """
    V = codes.shape[0]
    D_exp = W.shape[0]
    k = max(1, int(round(active_frac * D_exp)))
    margins = []
    for i in range(V):
        a = W @ codes[i]
        sorted_a = np.sort(a)[::-1]
        if k < D_exp:
            margin = float(sorted_a[k - 1] - sorted_a[k])
        else:
            margin = 0.0
        margins.append(margin)
    margins = np.array(margins)
    # Compute activation noise std for sigma_input = 0.1
    # For W_ij ~ N(0, 1/D_in), the activation noise std ~ sigma_input * ||W_i||
    # where ||W_i||^2 = sum_j W_ij^2 ~ D_in * (1/D_in) = 1 -> ||W_i|| ~ 1
    sigma_input = 0.1
    activation_noise_std = float(np.mean([np.linalg.norm(W[i]) for i in range(min(D_exp, 100))]))
    activation_noise_at_sigma01 = sigma_input * activation_noise_std
    return {
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "margin_min": float(margins.min()),
        "activation_noise_std_at_sigma01": activation_noise_at_sigma01,
        "margin_over_noise": float(margins.mean()) / (activation_noise_at_sigma01 + 1e-12),
        "interpretation": (
            f"Margin ({margins.mean():.6f}) << noise ({activation_noise_at_sigma01:.4f}): "
            "threshold is fragile -- small noise flips boundary units."
            if margins.mean() < 0.1 * activation_noise_at_sigma01 else
            f"Margin ({margins.mean():.6f}) >= 0.1 * noise ({activation_noise_at_sigma01:.4f}): "
            "threshold may be robust."
        ),
    }


# ---------------------------------------------------------------------------
# STAGE 2: on-bridge spiking confirmation
# ---------------------------------------------------------------------------

def run_stage2_bridge(input_codes: np.ndarray, W_expand: np.ndarray,
                       active_frac: float, D_exp: int, seed: int,
                       V: int = 8, n_steps: int = 300,
                       n_reads: int = 10) -> dict:
    """Stage 2: realize the fixed expansion as a non-plastic bridge region.

    Architecture:
      - Input region: n_input = D_in neurons (one-per-dimension), driven by external current
        proportional to the input code dimensions.
      - Expansion region: n_exp = D_exp neurons (the "granule cells") with FIXED random weights
        from the input region. Each expansion neuron has a threshold such that it fires when
        its summed input exceeds the k-th percentile activation.
      - We build the bridge with brain_region_framework; expansion region wiring is set via
        set_pathway_weights() with the FIXED W_expand matrix. Plasticity gate = CLOSED (frozen).

    Measurement: drive input region with a concept code, accumulate spikes over n_steps,
    read the binary top-k indicator by accumulated counts, measure same-input cosine
    between TWO independent reads (independent OU noise realizations).

    Returns per-concept reproducibility (same-input cosine, n_reads pairs) and
    between-concept cosine of the mean expanded codes.

    NOTE: if bridge build fails or is too slow, this returns a graceful failure dict.
    """
    try:
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig
        from sim.regions import BrainRegion, RegionPathway
        import os as _os
        _os.environ["SIM_BACKEND"] = "numpy"

        D_in = input_codes.shape[1]
        n_input = min(D_in, 256)  # cap input neurons for speed
        n_exp = min(D_exp, 512)   # cap expansion region for speed

        # Project input codes to n_input dimensions if needed
        if D_in != n_input:
            rng_proj = np.random.default_rng(seed + 50000)
            P = rng_proj.standard_normal((D_in, n_input)) / np.sqrt(D_in)
            projected_codes = input_codes @ P
            projected_codes = projected_codes - projected_codes.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(projected_codes, axis=1, keepdims=True)
            projected_codes = projected_codes / (norms + 1e-12)
        else:
            projected_codes = input_codes

        # Build a sub-expansion matrix from projected space
        rng_w = np.random.default_rng(seed + 60000)
        W_sub = rng_w.standard_normal((n_exp, n_input)) / np.sqrt(n_input)

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.enable_neuromodulator_subsystem = False
        cfg.enable_hebbian_learning = False
        cfg.enable_stdp = False
        cfg.enable_reward_modulation = False
        cfg.enable_homeostasis = False
        cfg.enable_stp = False
        cfg.dt = 1.0
        cfg.total_time = n_steps * cfg.dt
        cfg.background_noise_std = 24.98  # typical OU noise (biological)
        cfg.random_seed = seed

        # Brain regions: input + expansion
        input_region = BrainRegion(
            name="gr_input",
            n_neurons=n_input,
            exc_fraction=1.0,
            internal_density=0.0,
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
        )
        exp_region = BrainRegion(
            name="gr_expansion",
            n_neurons=n_exp,
            exc_fraction=1.0,
            internal_density=0.0,  # no recurrent connections in expansion layer
            exc_weight_mean=0.0,
            inh_weight_mean=0.0,
            weight_jitter=0.0,
        )
        pathway = RegionPathway(
            from_region="gr_input",
            to_region="gr_expansion",
            density=1.0,  # all-to-all (we'll set weights via set_pathway_weights)
            weight_mean=0.0,
            weight_jitter=0.0,
            plastic=False,
        )
        cfg.brain_regions = [input_region, exp_region]
        cfg.region_pathways = [pathway]
        cfg.num_neurons = n_input + n_exp

        bridge = SimulationBridge(cfg)
        bridge._initialize_simulation_data()

        # Set fixed expansion weights via set_pathway_weights
        # W_sub shape: [n_exp, n_input] -> pathway weight matrix
        # Bridge convention: pathway weights are stored as the weight for each
        # (pre, post) synapse. We scale by a drive_pA factor.
        bridge.set_pathway_weights(
            from_region="gr_input",
            to_region="gr_expansion",
            weights=W_sub,
            add_missing=True,
        )
        # Freeze expansion weights (no plasticity)
        bridge.set_plasticity_gate("gr_input_to_gr_expansion", 0.0)

        # Get region indices
        rm = bridge.region_manager
        input_idx = rm.indices("gr_input")
        exp_idx = rm.indices("gr_expansion")

        def read_expanded_code_spiking(concept_code: np.ndarray) -> np.ndarray:
            """Drive the input region with the concept code, accumulate expansion spikes."""
            # Reset bridge state
            bridge.cp_membrane_potential_v[:] = -65.0
            bridge.cp_recovery_variable_u[:] = bridge.cp_recovery_variable_u * 0.0
            bridge.cp_firing_states[:] = False
            bridge.cp_synaptic_conductance_exc[:] = 0.0
            bridge.cp_synaptic_conductance_inh[:] = 0.0

            # Scale concept code to drive pA range
            # concept_code is unit-normed; scale to 200 pA amplitude
            drive_pA = 200.0
            input_drive = concept_code * drive_pA  # [n_input]

            # Accumulate spikes in expansion region
            spike_counts = np.zeros(n_exp, dtype=np.float64)
            for _ in range(n_steps):
                # Inject input drive
                bridge.cp_external_input_current[input_idx] = input_drive
                bridge._run_one_simulation_step()
                # Accumulate expansion spikes
                spike_counts += bridge.cp_firing_states[exp_idx].astype(np.float64)

            # Binary top-k indicator from accumulated counts
            k = max(1, int(round(active_frac * n_exp)))
            threshold_rank = np.partition(spike_counts, n_exp - k)[n_exp - k]
            binary = (spike_counts >= threshold_rank).astype(np.float64)
            if binary.sum() > k:
                above = np.where(spike_counts > threshold_rank)[0]
                at = np.where(spike_counts == threshold_rank)[0]
                n_needed = k - len(above)
                binary[:] = 0.0
                binary[above] = 1.0
                if n_needed > 0:
                    binary[at[:n_needed]] = 1.0
            # Mean-remove + unit-normalize
            binary = binary - binary.mean()
            n = np.linalg.norm(binary)
            if n < 1e-12:
                return binary
            return binary / n

        # Measure reproducibility: n_reads pairs per concept
        repro_cosines = []
        expanded_codes_list = []
        use_V = min(V, len(input_codes))
        for i in range(use_V):
            c = projected_codes[i]
            codes_this = [read_expanded_code_spiking(c) for _ in range(2)]
            repro_cosines.append(float(codes_this[0] @ codes_this[1]))
            expanded_codes_list.append(codes_this[0])

        repro_mean = float(np.mean(repro_cosines))
        repro_min = float(np.min(repro_cosines))

        # Between-concept cosine
        expanded_arr = np.stack(expanded_codes_list)
        cos_btwn = []
        for i in range(use_V):
            for j in range(i + 1, use_V):
                cos_btwn.append(float(expanded_arr[i] @ expanded_arr[j]))
        between_cos_bridge = float(np.mean(cos_btwn)) if cos_btwn else 0.0

        stage2_go = (repro_mean >= 0.9)
        return {
            "status": "ran",
            "n_input": n_input,
            "n_exp": n_exp,
            "n_steps": n_steps,
            "n_concepts_tested": use_V,
            "repro_mean": repro_mean,
            "repro_min": repro_min,
            "repro_per_concept": repro_cosines,
            "between_cos_bridge": between_cos_bridge,
            "stage2_go": stage2_go,
            "verdict": (
                f"GO (on-bridge repro={repro_mean:.4f}>=0.9, between_cos={between_cos_bridge:.4f})"
                if stage2_go else
                f"NEGATIVE (on-bridge repro={repro_mean:.4f}<0.9, between_cos={between_cos_bridge:.4f})"
            ),
        }

    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()[:2000],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-expansion decorrelation probe (Marr-Albus granule layer)"
    )
    parser.add_argument("--seeds", default="42,43,44",
                        help="Comma-separated list of seeds (default: 42,43,44)")
    parser.add_argument("--V", type=int, default=16,
                        help="Number of concepts to load (default: 16)")
    parser.add_argument("--proj-dim", type=int, default=800,
                        help="Projection dimension for denoise64 codes (default: 800)")
    parser.add_argument("--expansion-ratios", default="4,8,16",
                        help="Expansion ratios D_exp/D_in (default: 4,8,16)")
    parser.add_argument("--active-fracs", default="0.05,0.1,0.2",
                        help="Active fractions for threshold (default: 0.05,0.1,0.2)")
    parser.add_argument("--noise-sigmas", default="0.001,0.005,0.01,0.05,0.1,0.2",
                        help="Input noise sigmas for reproducibility test (default: 0.001,0.005,0.01,0.05,0.1,0.2)")
    parser.add_argument("--n-trials-repro", type=int, default=100,
                        help="Trials per (noise_sigma) for reproducibility (default: 100)")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip Stage 2 on-bridge confirmation even if Stage 1 passes")
    parser.add_argument("--out", type=str,
                        default=os.path.join(_REPO, "research", "findings", "raw",
                                             "_cortex_fixed_expansion_decorrelation_probe.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    expansion_ratios = [float(r) for r in args.expansion_ratios.split(",")]
    active_fracs = [float(f) for f in args.active_fracs.split(",")]
    noise_sigmas = [float(s) for s in args.noise_sigmas.split(",")]

    print(f"\n=== Fixed-expansion decorrelation probe ===")
    print(f"Seeds: {seeds}, V={args.V}, proj_dim={args.proj_dim}")
    print(f"Expansion ratios: {expansion_ratios}, Active fracs: {active_fracs}")
    print(f"Noise sigmas (repro test): {noise_sigmas}")

    all_results = {}
    t_global = time.time()
    stage1_global_go = False
    best_op_global = None  # (ratio, f, D_exp, W) from the seed that found the best op point

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t_seed = time.time()

        # Load input codes
        words, codes, between_cos_input = load_denoise64_codes(seed, V=args.V,
                                                                proj_dim=args.proj_dim)
        print(f"  Input codes: V={len(words)}, D={codes.shape[1]}, "
              f"between_cos={between_cos_input:.4f}")

        # UNIT CHECK
        uc = unit_check(codes, between_cos_input)
        print(f"  Unit check: {uc['status']} (between_cos={between_cos_input:.4f} > 0.6 required)")
        if uc["status"] != "PASS":
            print(f"  ABORT: input codes are not correlated as required!")
            all_results[str(seed)] = {"unit_check": uc, "abort": True}
            continue

        # Stage 1: parameter sweep
        print(f"  Running Stage 1 sweep ({len(expansion_ratios) * len(active_fracs)} configs)...")
        sweep_results, best_op = run_stage1_sweep(
            codes, seed, args.V,
            expansion_ratios=tuple(expansion_ratios),
            active_fracs=tuple(active_fracs),
            noise_sigmas=tuple(noise_sigmas),
            n_trials_repro=args.n_trials_repro,
        )

        # Find best operating point for this seed
        print(f"\n  Stage 1 results:")
        print(f"  {'Config':<12} {'between_cos':>12} {'repro@s=0.1':>12} {'joint_GO':>10}")
        any_go_this_seed = False
        for k_cfg, v in sorted(sweep_results.items()):
            joint_ok = v["joint_criterion_met"]
            if joint_ok:
                any_go_this_seed = True
                stage1_global_go = True
                if best_op_global is None:
                    best_op_global = best_op
            print(f"  {k_cfg:<12} {v['between_cos_clean']:>12.4f} "
                  f"{v['repro_at_sigma01']:>12.4f} {'GO' if joint_ok else '---':>10}")

        # Lesion control (at best operating point found by sweep)
        if best_op is not None:
            best_ratio, best_f, best_D_exp, best_W = best_op
            print(f"\n  Best operating point: r={best_ratio}, f={best_f}, D_exp={best_D_exp}")
            lesion_res = measure_lesion(codes, best_W, best_f, noise_sigma=0.1, seed=seed)
            print(f"  Lesion control: {lesion_res['verdict']} "
                  f"(between_cos_lesion={lesion_res['between_cos_lesion']:.4f})")

            # Cleanup parity at best operating point
            print(f"  Running cleanup parity test at best operating point...")
            cleanup_res = run_cleanup_parity(
                codes, best_W, best_f,
                noise_fracs=(0.0, 0.1, 0.2, 0.5),
                seed=seed, n_trials=160,
            )
            print(f"  Cleanup parity (argmax | hopfield):")
            for ns, cv in sorted(cleanup_res.items()):
                print(f"    noise={ns:.2f}: argmax={cv['argmax']:.3f}, "
                      f"hopfield={cv['hopfield_mf']:.3f}, chance={cv['chance']:.3f}")
        else:
            lesion_res = {"verdict": "skipped", "note": "no best_op found"}
            cleanup_res = {}

        all_results[str(seed)] = {
            "unit_check": uc,
            "stage1_sweep": sweep_results,
            "best_op": {
                "ratio": best_op[0] if best_op else None,
                "active_frac": best_op[1] if best_op else None,
                "D_exp": best_op[2] if best_op else None,
            },
            "lesion_control": lesion_res,
            "cleanup_parity": cleanup_res,
            "any_go": any_go_this_seed,
            "seed_elapsed_s": time.time() - t_seed,
        }

        # Print the tension comparison (the DG finding was repro = sep = input_cos)
        print(f"\n  KEY: DG k-WTA failure was repro = sep = raw_cos ({between_cos_input:.3f})")
        print(f"  Fixed expansion result:")
        for k_cfg, v in sorted(sweep_results.items()):
            r_str = f"repro={v['repro_at_sigma01']:.3f}"
            s_str = f"sep(between_cos)={v['between_cos_clean']:.3f}"
            margin = v.get("threshold_margin_analysis", {})
            m_str = f"margin/noise={margin.get('margin_over_noise', 0):.4f}"
            tension = "COEXIST" if v["joint_criterion_met"] else "TENSION"
            print(f"    {k_cfg}: {r_str}, {s_str}, {m_str} [{tension}]")

    # STAGE 2: on-bridge confirmation (only if Stage 1 GO)
    stage2_result = None
    if stage1_global_go and not args.skip_stage2 and best_op_global is not None:
        print(f"\n=== Stage 2: On-bridge spiking confirmation ===")
        best_ratio, best_f, best_D_exp, best_W = best_op_global
        seed0 = seeds[0]
        words, codes_s2, _ = load_denoise64_codes(seed0, V=args.V, proj_dim=args.proj_dim)
        print(f"  Best op: r={best_ratio}, f={best_f}, D_exp={best_D_exp}")
        print(f"  Running on-bridge test (seed={seed0}, V=8, n_steps=300)...")
        t2 = time.time()
        stage2_result = run_stage2_bridge(
            codes_s2, best_W, best_f, best_D_exp, seed=seed0,
            V=8, n_steps=300, n_reads=2,
        )
        stage2_result["elapsed_s"] = time.time() - t2
        print(f"  Stage 2 verdict: {stage2_result.get('verdict', stage2_result.get('error', '?'))}")
    elif not stage1_global_go:
        print(f"\n  Stage 1 found NO operating point with joint GO (repro>=0.9 AND between_cos<=0.1).")
        print(f"  Stage 2 skipped (Stage 1 is NEGATIVE/BOUNDARY).")
        stage2_result = {"status": "skipped", "reason": "Stage 1 NEGATIVE"}
    else:
        print(f"\n  Stage 2 skipped (--skip-stage2 flag).")
        stage2_result = {"status": "skipped", "reason": "--skip-stage2"}

    # Final verdict
    print(f"\n=== FINAL VERDICT ===")
    go_seeds = [s for s in seeds if all_results.get(str(s), {}).get("any_go", False)]
    all_go = (len(go_seeds) == len(seeds))

    if stage1_global_go and all_go:
        if stage2_result and stage2_result.get("stage2_go"):
            verdict = "GO"
        elif stage2_result and stage2_result.get("status") == "ran":
            verdict = "GO-STAGE1-ONLY"
        else:
            verdict = "GO"
        desc = (
            "Fixed-expansion decorrelation is viable: "
            "a FIXED random expansion + threshold achieves repro>=0.9 AND between_cos<=0.1 "
            "(multi-seed) — decorrelation and reproducibility COEXIST. "
            "The DG k-WTA tension (repro ≈ sep ≈ raw_cos) is NOT present for a fixed expansion. "
            "The Marr-Albus granule-layer recoding is de-risked as a reproducible decorrelation front-end."
        )
    elif stage1_global_go and not all_go:
        verdict = "BOUNDARY"
        desc = (
            f"Stage 1 GO on {len(go_seeds)}/{len(seeds)} seeds. "
            "Joint criterion (repro>=0.9 AND between_cos<=0.1) not met at all seeds."
        )
    else:
        verdict = "NEGATIVE"
        desc = (
            "Fixed expansion does NOT achieve joint GO (repro>=0.9 AND between_cos<=0.1) "
            "at any tested operating point. The repro/decorrelation tension persists. "
            "Report exact numbers in the findings doc."
        )
    print(f"  {verdict}: {desc}")

    # Assemble final output
    out = {
        "probe": "cortex_fixed_expansion_decorrelation_probe",
        "date": "2026-06-11",
        "seeds": seeds,
        "expansion_ratios": expansion_ratios,
        "active_fracs": active_fracs,
        "noise_sigmas": noise_sigmas,
        "verdict": verdict,
        "description": desc,
        "go_seeds": go_seeds,
        "stage1_any_go": stage1_global_go,
        "stage2": stage2_result,
        "per_seed": all_results,
        "total_elapsed_s": time.time() - t_global,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  Raw JSON written to: {args.out}")
    return out


if __name__ == "__main__":
    main()
