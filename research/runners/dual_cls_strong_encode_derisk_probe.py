"""Dual / CLS STRONG-ENCODE DE-RISK — does a STRONG, STABLE sparse encode break the
reproducible-vs-decorrelated tension the WEAK spiking DG could not?

THE LOAD-BEARING DE-RISK (gates the months-scale dual/CLS learned-embedding build):
  The dual/CLS architecture is confirmed VIABLE IN SHAPE: the on-substrate clean-DG ceiling
  is +1.000 (a learned CA1->cortex decode inverts a REPRODUCIBLE sparse code perfectly --
  research/findings/2026-06-11-dual-CLS-onsubstrate-gate-BOUNDARY.md). The ONE broken link is
  the ENCODE: the project's spiking dentate-gyrus (DG) read is NON-REPRODUCIBLE (4x NEGATIVE),
  because the EC->DG drive is WEAK (~15 spikes / 600 DG neurons -- below the OU noise floor, so
  noise, not the input, picks the k-winners). The on-substrate gate showed the killer tension:
    - SPARSE read (k=40):  DG decorrelates (between-cos 0.17) but NOT reproducible (0.18-0.57).
    - DENSE  read (k=300): DG reproducible (0.93) but NOT decorrelated (between-cos 0.84).
  There was NO k where both held.

  THE QUESTION: does a STRONG, STABLE sparse encode -- one where the INPUT (not OU noise)
  determines the DG winners -- produce a code that is BOTH sparse-decorrelated AND reproducible
  at sparse k, which the WEAK DG could not?

  This is brain-based: real concept/place cells fire STRONGLY and STABLY (the input, not noise,
  determines the winners); the project's own LEARNED CONCEPT POOLS are reproducible strong
  sparse codes, unlike the weak DG. If a strong drive closes it, the months-scale learned-
  embedding build is JUSTIFIED (it learns exactly such a strong reproducible sparse encode). If
  even strong drive cannot co-achieve reproducible+decorrelated at sparse k, the spiking
  substrate has a deeper reproducible-sparse-encoding limit (a major finding).

THE MECHANISM (how the encode is made STRONG + STABLE, brain-faithfully):
  The prior BOUNDARY drove language_input -> (EC -> DG perforant path) and read the DG spike
  LOTTERY -- the EC->DG signal sat below the OU threshold-noise floor, so noise picked winners.
  Here, each concept is assigned a STABLE per-concept sparse DG target ensemble (the brain's
  developmental concept-cell assignment -- a fixed K-of-N set per concept, exactly the
  representation a learned strong sparse encode would converge to). That ensemble is driven by a
  SWEEPABLE STRONG current straight into the DG slice, so the INPUT -- not OU noise -- determines
  which DG cells fire. The DG is then READ with the SAME VALIDATED accumulated-rate-over-window
  k-WTA read used by the on-substrate gate / P1 trisynaptic loop. This isolates exactly the
  de-risk question: when the right sparse cells are STRONGLY + STABLY driven, is the spiking
  READ of them reproducible AND decorrelated at sparse k?

  This is the SPIKING STRONG-DG result -- the real test. We ALSO report a DETERMINISTIC
  reference (generate_sparse_patterns -- reproducible by construction, no spiking read) as the
  ceiling/sanity, labelled explicitly. The brain-based-vs-reproducible localisation is: does the
  deterministic reference close the round-trip while the spiking strong-DG cannot, or do both?

WHY DRIVE THE DG SLICE DIRECTLY (and why that is honest):
  The broken link is the ENCODE READ (does a strong drive give a reproducible+decorrelated
  spiking read?), NOT the EC->DG projection's tuning. Driving the assigned sparse DG ensemble
  strongly is the cleanest realisation of "the input strongly and stably picks the right sparse
  winners" -- which is precisely what a matured strong learned encode (or strong concept cells)
  would do. The DG READ (accumulate spike counts over a window + top-k) is unchanged and is the
  project-validated readout. We are NOT bypassing the spiking substrate -- the DG neurons still
  fire under their Izhikevich dynamics + OU noise + the DG PV-basket FFi; we are only making the
  drive strong+stable instead of the weak perforant path, then reading the genuine spike train.
  The strength SWEEP from weak->strong directly measures whether stronger drive lifts the read
  above the noise floor into the reproducible+decorrelated regime.

PROBES (multi-seed 42/43/44; numpy tiny-smoke FIRST, then GPU):

  STEP 1 -- STRENGTH x k SWEEP (the load-bearing co-occurrence surface).
    For each drive strength (pA) and each DG read k, on the SPIKING strong-DG encode, measure:
      (a) REPRODUCIBILITY: same concept -> two FRESH DG reads -> cosine (bar >= 0.9).
      (b) DECORRELATION:  between-concept DG cosine (bar <= ~0.1).
    Find the operating point (if any) where BOTH hold at sparse k -- strong drive breaking the
    tension. Report the FULL strength x k reproducibility AND decorrelation surfaces.

  STEP 2 -- ROUND-TRIP at the strong operating point (if found).
    encode (strong spiking DG) -> bind (Hopfield over the DG codes, noised cue) -> learned
    CA1->cortex ridge decode -> Pearson(S_orig, S'). Does it reach the +1.000 clean-DG ceiling
    (vs the failed +0.020 weak-DG)? With the PERMUTED-S baseline (~0).

  STEP 3 -- BINDING + GENERALIZATION on-substrate at the strong operating point.
    Binding identity (noised-cue Hopfield recovery, bar >= 0.7). Generalization: held-out-
    neighbour property inference over the strong-DG codes (graded must PASS; the ORTHOGONAL
    contrast must FAIL; permuted-S must collapse).

  REFERENCE -- the deterministic strong stable encode (generate_sparse_patterns).
    Same round-trip + decorrelation + reproducibility (==1.0 by construction) on the
    reproducible-by-construction codes -- the ceiling/sanity, labelled DETERMINISTIC.

DECISION (stated explicitly at end):
  GO if a strong drive gives reproducible (>=0.9) AND decorrelated (<=~0.1) sparse codes AT THE
     SAME operating point at sparse k, AND the round-trip closes (Pearson high, >> permuted ~0),
     AND binding+generalization pass on-substrate, multi-seed. -> the encode is fixable with a
     STRONGER learned sparse code -> the months-scale learned-embedding build is JUSTIFIED.
     Report the operating point.
  NEGATIVE/BOUNDARY if even strong drive cannot co-achieve reproducible+decorrelated at sparse k
     (the tension is fundamental to the spiking substrate, not just weak drive) -> a DEEPER
     reproducible-sparse-encoding limit -> the months-scale build on a spiking encode is NOT
     justified as-is. Characterize precisely (does the deterministic reference close it while the
     spiking strong-DG cannot? = the brain-based-vs-reproducible tension localised). No banking.

ANTI-CHEATS:
  - The reproducibility >=0.9 bar AND the decorrelation <=0.1 bar MUST CO-OCCUR at ONE operating
    point (don't report one without the other -- that is the exact tension).
  - PERMUTED-S baseline for the round-trip Pearson (~0); ORTHOGONAL-codes contrast for
    generalization (must FAIL there); the VALIDATED accumulated-rate DG read convention.
  - Explicit SPIKING strong-DG (the real test) vs DETERMINISTIC reference (the ceiling/sanity).

SUBSTRATE: build_biological_brain_regions(enable_hippocampus_consolidation=True), the SAME
bridge as the on-substrate gate + validated P1 trisynaptic loop. NO sim/ edits. The strong DG
drive sets cp_external_input_current on the DG slice (the body/world drive analogue -- the
input current the neural DG receives); the k-WTA DG read is a readout operation.

Run:
  # tiny numpy smoke (harness check, small bridge, fast)
  SIM_BACKEND=numpy python -m research.runners.dual_cls_strong_encode_derisk_probe \
      --smoke --seeds 42 --out research/findings/raw/_dual_cls_strong_encode_smoke.json
  # full GPU multi-seed
  SIM_BACKEND=cupy python -m research.runners.dual_cls_strong_encode_derisk_probe \
      --seeds 42,43,44 --out research/findings/raw/_dual_cls_strong_encode_multiseed.json
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


# ===========================================================================
# Reuse the architecture-proof's synthetic graded codebook + generalization +
# decode/Pearson machinery (identical conventions to the on-substrate gate).
# ===========================================================================
from research.runners.dual_cls_architecture_proof_probe import (  # noqa: E402
    build_graded_codebook,
    codebook_similarity_stats,
    assign_properties,
    run_generalization,
    run_generalization_permuted,
    native_cos_matrix,
)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _mean_offdiag_cos(M):
    """Mean off-diagonal cosine of the rows in [N, d] matrix M (native mean-removed)."""
    S = native_cos_matrix(M)
    N = S.shape[0]
    off = [float(S[i, j]) for i in range(N) for j in range(i + 1, N)]
    return float(np.mean(off)) if off else 0.0


def _max_abs_offdiag_cos(M):
    S = native_cos_matrix(M)
    N = S.shape[0]
    mx = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            mx = max(mx, abs(float(S[i, j])))
    return mx


# ===========================================================================
# STABLE per-concept sparse DG target ensembles (the brain's concept-cell
# assignment -- a fixed K-of-N DG set per concept; reproducible by construction).
# This is exactly the representation a matured strong learned sparse encode would
# converge to -- the strong+stable drive points the spiking DG AT these cells.
# ===========================================================================
def assign_sparse_dg_ensembles(n_concepts, n_dg, ensemble_size, seed):
    """Deterministic per-concept sparse DG target set (K=ensemble_size of N=n_dg).

    Reuses the project's generate_sparse_patterns convention (the validated reproducible
    sparse code mechanism). Returns a list of np.int64 index arrays (the DG cells each
    concept STRONGLY + STABLY drives) and the binary indicator matrix [n_concepts, n_dg].
    """
    from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
    pats = generate_sparse_patterns(n_concepts, n_dg, ensemble_size, seed)
    ensembles = [np.asarray(p, dtype=np.int64) for p in pats]
    binary = np.zeros((n_concepts, n_dg), dtype=np.float32)
    for i, p in enumerate(ensembles):
        binary[i, p] = 1.0
    return ensembles, binary


# ===========================================================================
# The on-substrate STRONG-DG encoder bridge
# ===========================================================================
class StrongDGEncoder:
    """The project's validated trisynaptic bridge, but the DG is driven STRONGLY + STABLY:
    each concept drives its assigned sparse DG ensemble with a sweepable strong current
    (cp_external_input_current on the DG slice), so the INPUT -- not OU noise -- determines
    the DG winners. The DG read is the VALIDATED accumulated-rate-over-window top-k indicator.

    Substrate identical to validate_trisynaptic_loop / dual_cls_onsubstrate_gate:
    build_biological_brain_regions(enable_hippocampus_consolidation=True). NO sim/ edits.
    """

    def __init__(self, seed, n_lang_input, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ec,
                 ca3_recurrent_density, ca3_recurrent_weight, verbose=True):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from research.runners.text_minimal_isolation import build_biological_brain_regions
        from sim.backend import get_backend
        self._xp, self._backend = get_backend()
        self.n_lang_input = int(n_lang_input)
        self.n_dg = int(n_dg)
        self.n_ca3 = int(n_ca3)
        self.verbose = verbose
        log = print if verbose else (lambda *a, **k: None)

        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input, n_motor_per_action=8, n_motor_fs_per_action=2,
            enable_motor_fs=True, enable_language_output=False,
            enable_hippocampus_consolidation=True,
            n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket,
            n_ca3=n_ca3, n_ca1=n_ca1,
            ca3_recurrent_density=ca3_recurrent_density,
            ca3_recurrent_weight=ca3_recurrent_weight,
        )
        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = list(regions)
        cfg.region_pathways = list(pathways)
        cfg.dt_ms = 1.0
        cfg.seed = seed
        cfg.enable_nmda = True
        cfg.enable_structural_plasticity = False
        cfg.enable_per_type_stp = False
        cfg.enable_hebbian_learning = False
        cfg.stdp_w_max = 10.0
        cfg.fast_spike_reset = True
        t0 = time.time()
        self.bridge = SimulationBridge(
            core_config=cfg, viz_config=VisualizationConfig(),
            runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge.runtime_state.max_delay_steps = int(
            cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        rm = self.bridge.region_manager
        self.dg_idx = np.asarray(rm.indices("dg"), dtype=np.int64)
        self.cfg = cfg
        self.build_seconds = time.time() - t0
        self.n_neurons = int(cfg.num_neurons)
        self.n_synapses = int(self.bridge.cp_connections.nnz)
        log("  [bridge] built %.1fs; %d neurons %d synapses (DG=%d CA3=%d) backend=%s"
            % (self.build_seconds, self.n_neurons, self.n_synapses,
               n_dg, n_ca3, self._backend))

    # --- drive helpers: drive the assigned sparse DG ensemble STRONGLY ---
    def _clear_drive(self):
        self.bridge.cp_external_input_current[:] = 0.0

    def _set_dg_ensemble_drive(self, ensemble_idx, drive_pA):
        """Drive the concept's assigned sparse DG cells with `drive_pA` (strong+stable)."""
        xp = self._xp
        self.bridge.cp_external_input_current[:] = 0.0
        dg_targets = self.dg_idx[ensemble_idx]
        self.bridge.cp_external_input_current[xp.asarray(dg_targets)] = float(drive_pA)

    def _step(self):
        self.bridge._run_one_simulation_step()
        self.bridge.runtime_state.current_time_step += 1

    # --- rate-accumulated k-WTA DG read (the VALIDATED read) ---
    def rate_kwta_dg_read(self, ensemble_idx, drive_pA, window_steps, k, reset_steps=40):
        """Drive the concept's sparse DG ensemble STRONGLY for `window_steps`, accumulate
        per-DG-neuron spike COUNTS, take the top-k by accumulated count -> binary indicator.

        Brain-based: strong input -> temporal rate integration + competitive selection.
        Returns (binary_code [n_dg], total_spikes, counts [n_dg])."""
        from sim.backend import to_host
        xp = self._xp
        self._clear_drive()
        for _ in range(reset_steps):
            self._step()
        self._set_dg_ensemble_drive(ensemble_idx, drive_pA)
        dg_reg = xp.asarray(self.dg_idx)
        counts = xp.zeros(len(self.dg_idx), dtype=xp.float32)
        for _ in range(window_steps):
            self._step()
            counts += self.bridge.cp_firing_states[dg_reg].astype(xp.float32)
        self._clear_drive()
        counts_np = to_host(counts).astype(np.float32)
        if k >= len(counts_np):
            code = np.ones(len(counts_np), dtype=np.float32)
        elif k <= 0:
            code = np.zeros(len(counts_np), dtype=np.float32)
        else:
            top_k_indices = np.argsort(counts_np)[::-1][:k]
            code = np.zeros(len(counts_np), dtype=np.float32)
            code[top_k_indices] = 1.0
        return code, int(counts_np.sum()), counts_np

    def encode_codebook_dg(self, ensembles, drive_pA, window_steps, k, reset_steps=40):
        """Encode every concept -> its real spiking strong-DG binary code [N, n_dg]."""
        N = len(ensembles)
        dg = np.zeros((N, self.n_dg), dtype=np.float32)
        spikes = np.zeros(N, dtype=np.float64)
        for i in range(N):
            c, sp, _ = self.rate_kwta_dg_read(ensembles[i], drive_pA, window_steps, k,
                                              reset_steps)
            dg[i] = c
            spikes[i] = sp
        return dg, spikes


# ===========================================================================
# Reproducibility for a given strength + k (same-input two fresh reads)
# ===========================================================================
def measure_repro(encoder, ensembles, drive_pA, window_steps, k, reset_steps,
                  n_repro_pairs, rng):
    repro_vals = []
    N = len(ensembles)
    for _ in range(n_repro_pairs):
        ci = int(rng.integers(N))
        c1, _, _ = encoder.rate_kwta_dg_read(ensembles[ci], drive_pA, window_steps, k,
                                             reset_steps)
        c2, _, _ = encoder.rate_kwta_dg_read(ensembles[ci], drive_pA, window_steps, k,
                                             reset_steps)
        repro_vals.append(_cos(c1, c2))
    return float(np.mean(repro_vals)), float(np.min(repro_vals)), repro_vals


# ===========================================================================
# BIND on the real DG codes (positive-control Hopfield) -- reuse the gate's logic
# ===========================================================================
def run_binding_on_dg(dg_codes, flip_frac, seed, n_dg):
    from research.runners.cortex_sparse_attractor_poscontrol_probe import (
        build_hopfield_weights, noisy_cue_sparse,
    )
    N, _ = dg_codes.shape
    codes_native = dg_codes - dg_codes.mean(axis=1, keepdims=True)
    codes_native = codes_native / (np.linalg.norm(codes_native, axis=1, keepdims=True) + 1e-12)
    W = build_hopfield_weights(codes_native)
    rng = np.random.default_rng(seed * 7 + int(flip_frac * 1000) + 17)
    recovered = np.zeros(N, dtype=int)
    settled = np.zeros((N, n_dg), dtype=np.float64)
    for i in range(N):
        cue = noisy_cue_sparse(codes_native[i], rng, flip_frac, n_dg)
        s = cue.copy().astype(np.float64)
        nn = np.linalg.norm(s)
        if nn > 1e-12:
            s = s / nn
        for _ in range(5):
            s_new = W @ s
            n2 = np.linalg.norm(s_new)
            if n2 < 1e-12:
                break
            s_new = s_new / n2
            if np.max(np.abs(s_new - s)) < 1e-8:
                break
            s = s_new
        recovered[i] = int(np.argmax(codes_native @ s))
        settled[i] = s
    identity_acc = float(np.sum(recovered == np.arange(N))) / N
    return recovered, settled, identity_acc


# ===========================================================================
# learned CA1->cortex ridge decode (the round-trip)
# ===========================================================================
def fit_decoder(dg_codes, cortex_codes, ridge=1e-2):
    X = dg_codes - dg_codes.mean(axis=1, keepdims=True)
    Y = cortex_codes
    n_dg = X.shape[1]
    A = X.T @ X + ridge * np.eye(n_dg)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def roundtrip_pearson(cortex_codes, S_orig, dg_codes, settled_states, ridge=1e-2):
    W_dec = fit_decoder(dg_codes, cortex_codes, ridge=ridge)
    settled_centered = settled_states - settled_states.mean(axis=1, keepdims=True)
    decoded = settled_centered @ W_dec
    S_round = native_cos_matrix(decoded)
    N = cortex_codes.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.corrcoef(S_orig[iu], S_round[iu])[0, 1]), S_round


def roundtrip_permuted_baseline(cortex_codes, S_orig, dg_codes, settled_states, seed,
                                ridge=1e-2):
    rng = np.random.RandomState(seed * 617 + 29)
    perm = rng.permutation(cortex_codes.shape[0])
    cortex_perm = cortex_codes[perm]
    W_dec = fit_decoder(dg_codes, cortex_perm, ridge=ridge)
    settled_centered = settled_states - settled_states.mean(axis=1, keepdims=True)
    decoded = settled_centered @ W_dec
    S_round = native_cos_matrix(decoded)
    N = cortex_codes.shape[0]
    iu = np.triu_indices(N, k=1)
    return float(np.corrcoef(S_orig[iu], S_round[iu])[0, 1])


# ===========================================================================
# DETERMINISTIC reference: the reproducible-by-construction sparse code (ceiling/sanity)
# ===========================================================================
def deterministic_reference(codes, S, binary_dg, seed, flip_frac, ridge, decorr_bar):
    """Run decorrelation + (perfect) reproducibility + the full round-trip on the
    DETERMINISTIC sparse DG codes (generate_sparse_patterns indicator matrix). These are
    reproducible by construction (repro = 1.000) -- the ceiling/sanity. Returns a dict.
    """
    dg_codes = binary_dg.astype(np.float32)
    between = _mean_offdiag_cos(dg_codes)
    sparsity = float(np.mean(dg_codes > 0))
    _, settled, identity = run_binding_on_dg(dg_codes, flip_frac, seed, dg_codes.shape[1])
    pearson, _ = roundtrip_pearson(codes, S, dg_codes, settled, ridge=ridge)
    perm = roundtrip_permuted_baseline(codes, S, dg_codes, settled, seed, ridge=ridge)
    clean, _ = roundtrip_pearson(codes, S, dg_codes, dg_codes.astype(np.float64), ridge=ridge)
    return {
        "label": "DETERMINISTIC_reference",
        "dg_between_cos_mean": between,
        "dg_repro_mean": 1.0,           # reproducible by construction
        "dg_sparsity": sparsity,
        "decorrelated": bool(between < decorr_bar),
        "binding_identity_acc": identity,
        "pearson_roundtrip": pearson,
        "pearson_permuted_baseline": perm,
        "pearson_clean_ceiling": clean,
    }


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed_full(seed, args):
    print(f"\n{'='*72}", flush=True)
    print(f"  STRONG-ENCODE DE-RISK -- SEED {seed}", flush=True)
    print(f"{'='*72}", flush=True)

    n_clusters = args.n_clusters
    per_cluster = args.per_cluster
    N = n_clusters * per_cluster
    dim = args.n_lang_input    # codebook lives in language_input space (same as the gate)

    # ---------- synthetic graded codebook (the cortex codes to round-trip back to) ----------
    codes, labels, S = build_graded_codebook(n_clusters, per_cluster, dim, seed,
                                             args.residual_frac)
    grad_stats = codebook_similarity_stats(codes, labels)
    print(f"  [graded codebook] N={N} ({n_clusters}x{per_cluster}) dim={dim}", flush=True)
    print(f"    within-cluster cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cluster cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}",
          flush=True)
    assert grad_stats["is_graded"], "graded codebook unit-check FAILED (within !>> between)"
    props = assign_properties(n_clusters, per_cluster, args.n_props, seed)

    # ---------- assign STABLE per-concept sparse DG ensembles (concept cells) ----------
    ensembles, binary_dg = assign_sparse_dg_ensembles(N, args.n_dg, args.ensemble_size, seed)
    det_between = _mean_offdiag_cos(binary_dg)
    print(f"  [DG ensembles] K={args.ensemble_size} of N={args.n_dg} per concept; "
          f"deterministic between-cos={det_between:+.3f} (reproducible-by-construction ref)",
          flush=True)

    # ---------- build the real spiking strong-DG bridge ----------
    enc = StrongDGEncoder(
        seed=seed, n_lang_input=args.n_lang_input, n_dg=args.n_dg,
        n_dg_pv_basket=args.n_dg_pv_basket, n_ca3=args.n_ca3, n_ca1=args.n_ca1,
        n_ec=args.n_ec, ca3_recurrent_density=args.ca3_recurrent_density,
        ca3_recurrent_weight=args.ca3_recurrent_weight, verbose=True)

    seed_rng = np.random.default_rng(seed + 777)
    drive_list = [float(x) for x in args.drive_list.split(",")]
    k_list = [int(x) for x in args.k_list.split(",")]

    # ============ STEP 1 -- the STRENGTH x k sweep (the co-occurrence surface) ============
    # For each (drive, k): SPIKING strong-DG between-concept decorrelation + same-input
    # reproducibility. Plus, at sparse k, the round-trip (bind -> decode -> Pearson) so a GO
    # cannot be missed by a bad point and a BOUNDARY means NO point co-achieves it.
    print("\n  [STEP 1 -- strength x k sweep: SPIKING strong-DG repro + decorrelation]",
          flush=True)
    print("    %-8s %-5s %-11s %-9s %-9s %-9s %-9s %-9s %-9s" %
          ("drive", "k", "between-cos", "repro", "sparsity", "spikes", "bind_id",
           "Pearson", "perm"), flush=True)
    sweep = []
    for drive_pA in drive_list:
        # one read per concept at this drive (for decorrelation) -- reuse across k by reading
        # the COUNTS once is not possible (top-k differs per k), so read per (drive, k). To keep
        # cost bounded we read per (drive, k) at the same drive.
        for k in k_list:
            dg_codes, spikes = enc.encode_codebook_dg(
                ensembles, drive_pA, args.window, k, args.reset_steps)
            between = _mean_offdiag_cos(dg_codes)
            sparsity = float(np.mean(dg_codes > 0))
            repro_mean, repro_min, _ = measure_repro(
                enc, ensembles, drive_pA, args.window, k, args.reset_steps,
                args.n_repro_pairs, seed_rng)
            # round-trip at this point (binding + learned decode).
            _, settled, identity = run_binding_on_dg(dg_codes, args.flip_frac, seed, enc.n_dg)
            pearson, _ = roundtrip_pearson(codes, S, dg_codes, settled, ridge=args.ridge)
            perm = roundtrip_permuted_baseline(codes, S, dg_codes, settled, seed,
                                               ridge=args.ridge)
            clean, _ = roundtrip_pearson(codes, S, dg_codes, dg_codes.astype(np.float64),
                                         ridge=args.ridge)
            # CO-OCCURRENCE flag: BOTH bars hold at this sparse-k point.
            decorrelated = between <= args.decorr_bar
            reproducible = repro_mean >= args.repro_bar
            cooccur = bool(decorrelated and reproducible)
            rec = {
                "drive_pA": drive_pA, "k": k,
                "dg_between_cos_mean": between,
                "dg_between_cos_max": _max_abs_offdiag_cos(dg_codes),
                "dg_repro_mean": repro_mean, "dg_repro_min": repro_min,
                "dg_sparsity": sparsity, "dg_total_spikes_mean": float(np.mean(spikes)),
                "binding_identity_acc": identity,
                "pearson_roundtrip": pearson,
                "pearson_permuted_baseline": perm,
                "pearson_clean_ceiling": clean,
                "decorrelated": decorrelated,
                "reproducible": reproducible,
                "cooccur_repro_and_decorr": cooccur,
                "_dg_codes": dg_codes, "_settled": settled,
            }
            sweep.append(rec)
            flag = "  <== CO-OCCUR" if cooccur else ""
            print("    %-8.0f %-5d %+11.3f %-9.3f %-9.3f %-9.1f %-9.3f %+9.3f %+9.3f%s" %
                  (drive_pA, k, between, repro_mean, sparsity,
                   float(np.mean(spikes)), identity, pearson, perm, flag), flush=True)

    # ---------- the LOAD-BEARING co-occurrence: a point where BOTH bars hold ----------
    cooccur_pts = [r for r in sweep if r["cooccur_repro_and_decorr"]]
    # Among co-occurring points, pick the one with the highest round-trip Pearson (the
    # operating point that both breaks the tension AND closes the round-trip). If none
    # co-occur, the operating point is the highest-Pearson point overall (for honest reporting),
    # but the GO gate requires co-occurrence so this still reports NEGATIVE.
    if cooccur_pts:
        chosen = max(cooccur_pts, key=lambda r: r["pearson_roundtrip"])
        cooccur_found = True
    else:
        chosen = max(sweep, key=lambda r: r["pearson_roundtrip"])
        cooccur_found = False

    # best-decorrelating + best-reproducible points (for the tension report).
    best_decorr = min(sweep, key=lambda r: r["dg_between_cos_mean"])
    best_repro = max(sweep, key=lambda r: r["dg_repro_mean"])

    print(f"\n  [STEP 1 result] co-occurrence (repro>={args.repro_bar} AND "
          f"decorr<={args.decorr_bar}) found: {cooccur_found}", flush=True)
    print(f"    operating point: drive={chosen['drive_pA']:.0f} k={chosen['k']} "
          f"between-cos={chosen['dg_between_cos_mean']:+.3f} repro={chosen['dg_repro_mean']:.3f} "
          f"Pearson={chosen['pearson_roundtrip']:+.3f}", flush=True)
    print(f"    [tension] best-decorrelating: drive={best_decorr['drive_pA']:.0f} "
          f"k={best_decorr['k']} between-cos={best_decorr['dg_between_cos_mean']:+.3f} "
          f"repro={best_decorr['dg_repro_mean']:.3f}", flush=True)
    print(f"    [tension] best-reproducible:  drive={best_repro['drive_pA']:.0f} "
          f"k={best_repro['k']} repro={best_repro['dg_repro_mean']:.3f} "
          f"between-cos={best_repro['dg_between_cos_mean']:+.3f}", flush=True)

    dg_codes = chosen["_dg_codes"]
    settled = chosen["_settled"]
    identity_acc = chosen["binding_identity_acc"]
    pearson = chosen["pearson_roundtrip"]
    pearson_perm = chosen["pearson_permuted_baseline"]
    pearson_clean = chosen["pearson_clean_ceiling"]

    print(f"\n  [STEP 2 -- round-trip at operating point] Pearson(S,S') = {pearson:+.3f}  "
          f"(permuted {pearson_perm:+.3f}; clean-DG ceiling {pearson_clean:+.3f}; "
          f"weak-DG +0.020; numpy proof +0.877)", flush=True)

    # ============ STEP 3 -- generalization on-substrate at the operating point ============
    print("\n  [STEP 3 -- binding + generalization on-substrate at operating point]",
          flush=True)
    # orthogonal control encoded through the SAME strong-DG read (decisive contrast). We assign
    # orthogonal cortex codes a DIFFERENT (random) set of DG ensembles drawn from a disjoint
    # seed so the DG-space similarity vote has no graded neighbour structure to exploit.
    rng_o = np.random.RandomState(seed * 71 + 5)
    ortho_ens, _ = assign_sparse_dg_ensembles(N, args.n_dg, args.ensemble_size, seed + 99991)
    # shuffle which ensemble each ortho concept gets so cluster label is decoupled from code.
    perm_o = rng_o.permutation(N)
    ortho_ens = [ortho_ens[i] for i in perm_o]
    dg_ortho, _ = enc.encode_codebook_dg(ortho_ens, chosen["drive_pA"], args.window,
                                         chosen["k"], args.reset_steps)

    gen_graded = run_generalization(dg_codes, labels, props, n_clusters, per_cluster,
                                    seed, args.k_neighbours)
    gen_ortho = run_generalization(dg_ortho, labels, props, n_clusters, per_cluster,
                                   seed, args.k_neighbours)
    gen_perm = run_generalization_permuted(dg_codes, labels, props, n_clusters,
                                           per_cluster, seed, args.k_neighbours)
    chance = gen_graded["chance"]
    print(f"    binding identity (operating point) = {identity_acc:.3f}", flush=True)
    print(f"    graded(DG)    acc={gen_graded['accuracy']:.3f} "
          f"(chance={chance:.3f}, {gen_graded['ratio_vs_chance']:.1f}x)", flush=True)
    print(f"    orthogonal(DG) acc={gen_ortho['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)
    print(f"    permuted-S(DG) acc={gen_perm['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)
    a1 = gen_graded["accuracy"] >= args.a1_bar
    a2 = gen_ortho["accuracy"] <= 1.5 * chance
    a3 = gen_perm["accuracy"] <= 1.5 * chance

    # ============ REFERENCE -- deterministic strong stable encode (ceiling/sanity) ============
    print("\n  [REFERENCE -- DETERMINISTIC strong stable encode (generate_sparse_patterns)]",
          flush=True)
    det = deterministic_reference(codes, S, binary_dg, seed, args.flip_frac, args.ridge,
                                  args.decorr_bar)
    print(f"    between-cos={det['dg_between_cos_mean']:+.3f} (decorrelated={det['decorrelated']}) "
          f"repro=1.000(by construction) bind={det['binding_identity_acc']:.3f} "
          f"Pearson={det['pearson_roundtrip']:+.3f} (perm {det['pearson_permuted_baseline']:+.3f}; "
          f"clean {det['pearson_clean_ceiling']:+.3f})", flush=True)

    # ---------- per-seed gates ----------
    # GO gate: co-occurrence (repro>=bar AND decorr<=bar at ONE sparse-k point) AND the round-
    # trip closes there (Pearson >= c2_bar AND > permuted + 0.3) AND binding AND generalization.
    cooccur_ok = cooccur_found
    binding_ok = identity_acc >= args.binding_bar
    c2_ok = (pearson >= args.c2_bar) and (pearson > pearson_perm + 0.3)
    gates = {
        "cooccur_repro_and_decorr": bool(cooccur_ok),
        "roundtrip_pearson_closes": bool(c2_ok),
        "binding": bool(binding_ok),
        "a1_graded_generalizes": bool(a1),
        "a2_orthogonal_collapses": bool(a2),
        "a3_permuted_collapses": bool(a3),
    }
    print(f"\n  [SEED {seed} gates] {gates}", flush=True)

    # strip private arrays before JSON
    for r in sweep:
        r.pop("_dg_codes", None)
        r.pop("_settled", None)

    return {
        "seed": seed,
        "graded_stats": grad_stats,
        "n_neurons": enc.n_neurons,
        "n_synapses": enc.n_synapses,
        "build_seconds": enc.build_seconds,
        "deterministic_between_cos": det_between,
        "sweep_strength_x_k": sweep,
        "cooccur_found": bool(cooccur_found),
        "operating_point": {
            "drive_pA": chosen["drive_pA"], "k": chosen["k"],
            "dg_between_cos_mean": chosen["dg_between_cos_mean"],
            "dg_repro_mean": chosen["dg_repro_mean"],
            "dg_sparsity": chosen["dg_sparsity"],
            "dg_total_spikes_mean": chosen["dg_total_spikes_mean"],
            "binding_identity_acc": identity_acc,
            "pearson_roundtrip": pearson,
            "pearson_permuted_baseline": pearson_perm,
            "pearson_clean_ceiling": pearson_clean,
        },
        "tension": {
            "best_decorrelating": {
                "drive_pA": best_decorr["drive_pA"], "k": best_decorr["k"],
                "between_cos": best_decorr["dg_between_cos_mean"],
                "repro": best_decorr["dg_repro_mean"],
            },
            "best_reproducible": {
                "drive_pA": best_repro["drive_pA"], "k": best_repro["k"],
                "repro": best_repro["dg_repro_mean"],
                "between_cos": best_repro["dg_between_cos_mean"],
            },
        },
        "generalization": {
            "graded": gen_graded, "orthogonal": gen_ortho, "permuted": gen_perm,
            "chance": chance,
        },
        "deterministic_reference": det,
        "gates": gates,
    }


def main():
    p = argparse.ArgumentParser(description="Dual-CLS strong-encode de-risk probe")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true",
                   help="tiny bridge + tiny codebook for harness verification (fast)")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--residual-frac", type=float, default=0.55)
    # bridge sizing (defaults match the validated P1 / on-substrate-gate scale)
    p.add_argument("--n-lang-input", type=int, default=512)
    p.add_argument("--n-ec", type=int, default=160)
    p.add_argument("--n-dg", type=int, default=600)
    p.add_argument("--n-dg-pv-basket", type=int, default=180)
    p.add_argument("--n-ca3", type=int, default=300)
    p.add_argument("--n-ca1", type=int, default=120)
    p.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    p.add_argument("--ca3-recurrent-weight", type=float, default=2.0)
    # STRONG drive sweep + DG read
    p.add_argument("--ensemble-size", type=int, default=40,
                   help="per-concept sparse DG ensemble size (K of n_dg)")
    p.add_argument("--drive-list", default="800,2000,5000,12000",
                   help="DG ensemble drive strengths (pA) to sweep weak->strong")
    p.add_argument("--window", type=int, default=200, help="DG accumulation window (steps)")
    p.add_argument("--k-list", default="40,80,150",
                   help="DG k-WTA read sizes (sparse) to sweep")
    p.add_argument("--reset-steps", type=int, default=40)
    p.add_argument("--n-repro-pairs", type=int, default=8)
    p.add_argument("--flip-frac", type=float, default=0.1, help="binding cue noise")
    p.add_argument("--ridge", type=float, default=1e-2)
    # gate bars (the load-bearing co-occurrence)
    p.add_argument("--decorr-bar", type=float, default=0.10,
                   help="DG between-cos must be <= this to count as decorrelated (sparse)")
    p.add_argument("--repro-bar", type=float, default=0.90,
                   help="DG same-input cosine must be >= this to count as reproducible")
    p.add_argument("--binding-bar", type=float, default=0.70)
    p.add_argument("--c2-bar", type=float, default=0.70)
    p.add_argument("--a1-bar", type=float, default=0.70)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_clusters = 4
        args.per_cluster = 3
        args.n_lang_input = 128
        args.n_ec = 48
        args.n_dg = 200
        args.n_dg_pv_basket = 60
        args.n_ca3 = 100
        args.n_ca1 = 60
        args.window = 80
        args.ensemble_size = 15
        args.drive_list = "800,5000"
        args.k_list = "10,20,40"
        args.n_repro_pairs = 3
        os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_start = time.time()
    print(f"[dual-CLS strong-encode de-risk] seeds={seeds} backend={backend} "
          f"smoke={args.smoke}", flush=True)
    print(f"  decorr-bar(<=)={args.decorr_bar} repro-bar(>=)={args.repro_bar} "
          f"(the load-bearing CO-OCCURRENCE)", flush=True)

    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = run_seed_full(seed, args)

    # ---------- overall verdict ----------
    def all_gate(g):
        return all(per_seed[str(s)]["gates"][g] for s in seeds)

    g_cooccur = all_gate("cooccur_repro_and_decorr")
    g_c2 = all_gate("roundtrip_pearson_closes")
    g_bind = all_gate("binding")
    g_a1 = all_gate("a1_graded_generalizes")
    g_a2 = all_gate("a2_orthogonal_collapses")
    g_a3 = all_gate("a3_permuted_collapses")

    # GO requires: the load-bearing co-occurrence (repro AND decorr at one sparse-k point) AND
    # the round-trip closes there AND binding AND generalization, multi-seed.
    if g_cooccur and g_c2 and g_bind and g_a1 and g_a2 and g_a3:
        verdict = "GO"
    elif (not g_cooccur):
        # the tension is NOT broken by strong drive -> the deeper substrate limit.
        verdict = "BOUNDARY_strong_drive_does_not_co_achieve_repro_and_decorr"
    elif g_cooccur and not g_c2:
        verdict = "BOUNDARY_cooccur_but_roundtrip_not_closed"
    elif g_cooccur and g_c2 and not g_bind:
        verdict = "BOUNDARY_binding_fails_at_operating_point"
    else:
        verdict = "BOUNDARY_generalization_not_similarity_driven_onsubstrate"

    # aggregate load-bearing numbers
    cooccur_found = [per_seed[str(s)]["cooccur_found"] for s in seeds]
    op_pearson = [per_seed[str(s)]["operating_point"]["pearson_roundtrip"] for s in seeds]
    op_perm = [per_seed[str(s)]["operating_point"]["pearson_permuted_baseline"] for s in seeds]
    op_clean = [per_seed[str(s)]["operating_point"]["pearson_clean_ceiling"] for s in seeds]
    op_between = [per_seed[str(s)]["operating_point"]["dg_between_cos_mean"] for s in seeds]
    op_repro = [per_seed[str(s)]["operating_point"]["dg_repro_mean"] for s in seeds]
    op_bind = [per_seed[str(s)]["operating_point"]["binding_identity_acc"] for s in seeds]
    det_pearson = [per_seed[str(s)]["deterministic_reference"]["pearson_roundtrip"]
                   for s in seeds]
    det_between = [per_seed[str(s)]["deterministic_reference"]["dg_between_cos_mean"]
                   for s in seeds]
    det_bind = [per_seed[str(s)]["deterministic_reference"]["binding_identity_acc"]
                for s in seeds]
    gen_graded = [per_seed[str(s)]["generalization"]["graded"]["accuracy"] for s in seeds]
    gen_ortho = [per_seed[str(s)]["generalization"]["orthogonal"]["accuracy"] for s in seeds]

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "backend": backend,
        "smoke": bool(args.smoke),
        "bars": {"decorr_bar_le": args.decorr_bar, "repro_bar_ge": args.repro_bar,
                 "c2_bar": args.c2_bar, "binding_bar": args.binding_bar,
                 "a1_bar": args.a1_bar},
        "gates_all_seeds": {
            "cooccur_repro_and_decorr": g_cooccur,
            "roundtrip_pearson_closes": g_c2,
            "binding": g_bind,
            "a1_graded_generalizes": g_a1, "a2_orthogonal_collapses": g_a2,
            "a3_permuted_collapses": g_a3,
        },
        "load_bearing": {
            "cooccur_found_per_seed": cooccur_found,
            "spiking_strong_dg": {
                "operating_pearson_per_seed": op_pearson,
                "operating_pearson_mean": float(np.mean(op_pearson)),
                "operating_permuted_per_seed": op_perm,
                "operating_permuted_mean": float(np.mean(op_perm)),
                "operating_clean_ceiling_per_seed": op_clean,
                "operating_between_cos_per_seed": op_between,
                "operating_between_cos_mean": float(np.mean(op_between)),
                "operating_repro_per_seed": op_repro,
                "operating_repro_mean": float(np.mean(op_repro)),
                "operating_binding_per_seed": op_bind,
                "generalization_graded_per_seed": gen_graded,
                "generalization_orthogonal_per_seed": gen_ortho,
            },
            "deterministic_reference": {
                "pearson_per_seed": det_pearson,
                "pearson_mean": float(np.mean(det_pearson)),
                "between_cos_per_seed": det_between,
                "between_cos_mean": float(np.mean(det_between)),
                "binding_per_seed": det_bind,
            },
            "weak_dg_reference_pearson": 0.020,
            "numpy_proof_reference_pearson": 0.877,
            "clean_dg_ceiling_reference": 1.000,
        },
        "elapsed_total_s": time.time() - t_start,
    }

    print(f"\n{'='*72}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  CO-OCCURRENCE (repro>={args.repro_bar} AND decorr<={args.decorr_bar}) all seeds: "
          f"{g_cooccur}  (per-seed {cooccur_found})", flush=True)
    print(f"  Round-trip Pearson closes (all seeds):  {g_c2}", flush=True)
    print(f"  Binding at operating point (all seeds): {g_bind}", flush=True)
    print(f"  A1 graded generalizes:                  {g_a1}", flush=True)
    print(f"  A2 orthogonal collapses:                {g_a2}", flush=True)
    print(f"  A3 permuted collapses:                  {g_a3}", flush=True)
    print(f"  >>> SPIKING strong-DG round-trip Pearson (op point, mean) = "
          f"{np.mean(op_pearson):+.3f}  (weak-DG +0.020; numpy +0.877; clean ceiling +1.000; "
          f"permuted {np.mean(op_perm):+.3f})", flush=True)
    print(f"  >>> SPIKING strong-DG: decorr={np.mean(op_between):+.3f} "
          f"repro={np.mean(op_repro):.3f} at op point", flush=True)
    print(f"  >>> DETERMINISTIC reference round-trip Pearson (mean) = "
          f"{np.mean(det_pearson):+.3f}  (decorr {np.mean(det_between):+.3f}, repro 1.000)",
          flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*72}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}

    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "multiseed"
        args.out = os.path.join(raw_dir, f"_dual_cls_strong_encode_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
