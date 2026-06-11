"""Dual / CLS ON-SUBSTRATE GATE — does the REAL spiking trisynaptic loop behave as
the numpy ARCHITECTURE-PROOF assumed?

CONTEXT:
  The dual/CLS architecture passed the numpy ARCHITECTURE-PROOF
  (research/findings/2026-06-11-dual-CLS-architecture-proof-GO.md): on a SYNTHETIC
  graded codebook, the encode->decorrelate->bind->retrieve->decode ROUND-TRIP PRESERVES
  graded similarity (Pearson(S,S') = +0.877) with a LEARNED CA1->cortex ridge decode;
  generalization works (1.000), binding 1.000. BUT the numpy proof used a DETERMINISTIC
  sparse random-projection for the encode. The REAL hippocampal DG encode is SPIKING and
  was found SUB-REPRODUCIBLE earlier
  (research/findings/2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md): same-input
  cosine ~0.05 at instantaneous read; rate-accumulated k-WTA gives repro~=sep (no
  decorrelation gain) on the brain's RAW correlated denoise64 codes.

THE LOAD-BEARING ON-SUBSTRATE QUESTION:
  Drive the SYNTHETIC GRADED codes (from the architecture-proof) through the REAL spiking
  EC->DG->CA3->CA1 loop, read DG with the VALIDATED accumulated-rate window, and re-run
  the load-bearing round-trip ON-SUBSTRATE:
    - does the real spiking DG DECORRELATE the graded codes?
    - is the real DG read REPRODUCIBLE enough (the documented risk)?
    - does the LEARNED CA1->cortex ridge decode COMPENSATE for a noisy/sub-reproducible DG
      and recover the round-trip graded similarity?

KEY HYPOTHESIS (why it might still pass): the decode is LEARNED, so it may compensate for a
noisy DG encode (the ridge map adapts to whatever the DG produces). The numpy proof's
robustness suggests the learned decode is the load-bearing piece, not a clean encode.

  N.B. The graded codebook here is built in the bridge's `language_input` SPACE (dim =
  n_lang_input), so each graded code IS a drive pattern directly (positive, no extra
  projection step) -- the cleanest mapping onto the substrate and the simplest object whose
  DG-encoding we can read. The graded structure (within-cluster cos high, between-cluster
  low) is built and asserted, exactly as in the numpy proof.

PROBES (multi-seed 42/43/44; numpy tiny-smoke FIRST, then GPU):

  STEP 1 -- DG-separation on graded input (ON-SUBSTRATE)
    Feed each graded cortex code into language_input->EC->DG on the real bridge; read the DG
    ensemble with the PROJECT-VALIDATED accumulated-rate-over-window k-WTA read (NOT a single
    noisy spike read). Measure:
      (a) DG DECORRELATION: between-DG cosine of the graded codes (low = pattern-separated).
      (b) DG REPRODUCIBILITY: same graded input -> two fresh DG reads -> cosine (the
          documented risk -- reported FRONT-AND-CENTRE).

  STEP 2 -- BIND on the DG/decorrelated codes (positive-control logic)
    Build a Hebbian Hopfield over the DG-space binary codes; present a noised cue per
    concept, settle, recover identity. Confirm binding/retrieval works on the REAL DG codes.

  STEP 3 -- LEARNED CA1->cortex decode (the COMPENSATION test, LOAD-BEARING)
    Train a ridge linear map from the (real spiking) DG-space code back to the graded cortex
    code. Measure the ROUND-TRIP Pearson(S, S') ON-SUBSTRATE -- does the learned decode
    recover the graded similarity from the REAL (possibly noisy) DG encode? Compare to the
    numpy proof's +0.877 and to a PERMUTED-S baseline (~0).

  STEP 4 -- GENERALIZATION on-substrate
    Held-out-neighbour property inference, but with the cortex codes passed through the real
    loop into DG space -- does it still PASS on graded + FAIL on orthogonal?

DECISION (stated explicitly at end):
  GO if: real DG decorrelates the graded codes AND the learned CA1 decode recovers the
     round-trip similarity (Pearson high, >= 0.7, >> permuted) AND binding + generalization
     hold on-substrate, multi-seed. -> the dual architecture works ON THE REAL SUBSTRATE
     (with synthetic graded codes); the only remaining piece is the learned graded-similarity
     embedding (months-scale build) -> recommend scoping it.
  NEGATIVE/BOUNDARY otherwise -> name what breaks. If the spiking-DG sub-reproducibility
     destroys the round-trip even with the learned decode -> characterize precisely (the
     encode needs a more-reproducible mechanism -- a deterministic/learned cortex->DG-target
     map vs the noisy spiking DG; surface the brain-based-vs-reproducible tension). No banking.

ANTI-CHEATS:
  - PERMUTED-S baseline for the round-trip Pearson (must give ~0 -> a high true Pearson is real).
  - ORTHOGONAL-codes contrast for generalization (must FAIL there).
  - DG same-input reproducibility reported EXPLICITLY (the documented risk -- not buried).
  - VALIDATED DG read convention (accumulated-rate window), graded-input cosine asserted as read.

SUBSTRATE: build_biological_brain_regions(enable_hippocampus_consolidation=True), the same
bridge the validated P1 trisynaptic result + the DG-CA3 NEGATIVE used. NO sim/ edits. The
rate-kWTA DG read is a READOUT OPERATION (accumulate counts + argsort + top-k), brain-based.

Run:
  # tiny numpy smoke (harness check, small bridge, fast)
  SIM_BACKEND=numpy python -m research.runners.dual_cls_onsubstrate_gate_probe \
      --smoke --seeds 42 --out research/findings/raw/_dual_cls_onsubstrate_smoke.json
  # full GPU multi-seed
  SIM_BACKEND=cupy python -m research.runners.dual_cls_onsubstrate_gate_probe \
      --seeds 42,43,44 --out research/findings/raw/_dual_cls_onsubstrate_multiseed.json
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
# Reuse the architecture-proof's synthetic graded codebook + generalization
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


# ===========================================================================
# The on-substrate trisynaptic DG-encode bridge
# ===========================================================================
class TrisynapticDGEncoder:
    """Wrap the project's validated trisynaptic bridge with a rate-accumulated k-WTA DG read.

    Substrate identical to validate_trisynaptic_loop / cortex_dg_ratekwta_cleanup_probe:
    build_biological_brain_regions(enable_hippocampus_consolidation=True). NO sim/ edits.

    The graded codebook is built in language_input SPACE (dim = n_lang_input), so each
    graded code is rendered to a non-negative drive directly (scale to drive_pA at the max).
    The DG read is the VALIDATED accumulated-rate-over-window top-k indicator.
    """

    def __init__(self, seed, n_lang_input, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ec,
                 ca3_recurrent_density, ca3_recurrent_weight, drive_pA, verbose=True):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from research.runners.text_minimal_isolation import build_biological_brain_regions
        from sim.backend import get_backend
        self._xp, self._backend = get_backend()
        self.drive_pA = float(drive_pA)
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
        self.lang_idx = np.asarray(rm.indices("language_input"), dtype=np.int64)
        self.dg_idx = np.asarray(rm.indices("dg"), dtype=np.int64)
        self.ca3_idx = np.asarray(rm.indices("ca3"), dtype=np.int64)
        self.cfg = cfg
        self.build_seconds = time.time() - t0
        self.n_neurons = int(cfg.num_neurons)
        self.n_synapses = int(self.bridge.cp_connections.nnz)
        log("  [bridge] built %.1fs; %d neurons %d synapses (lang=%d DG=%d CA3=%d) backend=%s"
            % (self.build_seconds, self.n_neurons, self.n_synapses,
               len(self.lang_idx), n_dg, n_ca3, self._backend))

    # --- drive helpers ---
    def _render_drive(self, code_vec):
        """Graded code (in language_input space) -> non-negative drive [n_lang], pA.

        The graded codebook is built in lang-input space, so this maps the code directly to
        a drive: shift to non-negative (relu the positive lobe) and scale to drive_pA at max.
        A code and its scaled positive lobe carry the same 'who is most active' structure,
        which is what the EC->DG feed-forward separation reads.
        """
        x = np.asarray(code_vec, dtype=np.float64)
        x = np.maximum(x, 0.0)            # positive lobe (drive is non-negative current)
        m = x.max()
        if m > 1e-9:
            x = x / m
        return (x * self.drive_pA).astype(np.float32)

    def _set_lang_drive(self, drive_vec):
        xp = self._xp
        self.bridge.cp_external_input_current[:] = 0.0
        self.bridge.cp_external_input_current[xp.asarray(self.lang_idx)] = xp.asarray(drive_vec)

    def _clear_drive(self):
        self.bridge.cp_external_input_current[:] = 0.0

    def _step(self):
        self.bridge._run_one_simulation_step()
        self.bridge.runtime_state.current_time_step += 1

    # --- rate-accumulated k-WTA DG read (the VALIDATED read) ---
    def rate_kwta_dg_read(self, code_vec, window_steps, k, reset_steps=40):
        """Accumulate per-DG-neuron spike COUNTS over `window_steps`, take the top-k by
        accumulated count -> binary indicator (the DG pattern-separation ensemble).

        Brain-based: temporal rate integration + competitive selection. Returns
        (binary_code [n_dg], total_spikes, counts [n_dg])."""
        from sim.backend import to_host
        xp = self._xp
        self._clear_drive()
        for _ in range(reset_steps):
            self._step()
        self._set_lang_drive(self._render_drive(code_vec))
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

    def encode_codebook_dg(self, codes, window_steps, k, reset_steps=40):
        """Encode every graded cortex code -> its real spiking DG-space binary code [N, n_dg]."""
        N = codes.shape[0]
        dg = np.zeros((N, self.n_dg), dtype=np.float32)
        spikes = np.zeros(N, dtype=np.float64)
        for i in range(N):
            c, sp, _ = self.rate_kwta_dg_read(codes[i], window_steps, k, reset_steps)
            dg[i] = c
            spikes[i] = sp
        return dg, spikes


# ===========================================================================
# STEP 1 -- DG decorrelation + same-input reproducibility (the documented risk)
# ===========================================================================
def measure_dg_decorrelation_and_repro(encoder, codes, window_steps, k,
                                        reset_steps, n_repro_pairs, rng):
    """Measure (a) DG between-concept decorrelation, (b) DG same-input reproducibility.

    Returns dict with dg_between_cos (decorrelation; low = separated), dg_repro_mean/min
    (same-input cosine -- the documented sub-reproducibility risk), spikes, and the
    DG codes themselves (re-used by later steps so the encode is read ONCE per concept).
    """
    V = codes.shape[0]
    # (a) one read per concept -> between-concept decorrelation.
    dg_codes, spikes = encoder.encode_codebook_dg(codes, window_steps, k, reset_steps)
    dg_between = _mean_offdiag_cos(dg_codes)
    dg_between_max = 0.0
    S = native_cos_matrix(dg_codes)
    for i in range(V):
        for j in range(i + 1, V):
            dg_between_max = max(dg_between_max, abs(float(S[i, j])))
    sparsity = float(np.mean(dg_codes > 0))

    # (b) same-input reproducibility: for n_repro_pairs random concepts, two FRESH reads.
    repro_vals = []
    for _ in range(n_repro_pairs):
        ci = int(rng.integers(V))
        c1, _, _ = encoder.rate_kwta_dg_read(codes[ci], window_steps, k, reset_steps)
        c2, _, _ = encoder.rate_kwta_dg_read(codes[ci], window_steps, k, reset_steps)
        repro_vals.append(_cos(c1, c2))

    return {
        "window": window_steps,
        "k": k,
        "dg_between_cos_mean": dg_between,           # decorrelation: low = pattern-separated
        "dg_between_cos_max": dg_between_max,
        "dg_sparsity": sparsity,
        "dg_repro_mean": float(np.mean(repro_vals)),  # THE documented risk
        "dg_repro_min": float(np.min(repro_vals)),
        "dg_repro_vals": [float(x) for x in repro_vals],
        "dg_total_spikes_mean": float(np.mean(spikes)),
        "_dg_codes": dg_codes,                        # carried forward (private)
    }


# ===========================================================================
# STEP 2 -- BIND on the real DG codes (positive-control Hopfield)
# ===========================================================================
def run_binding_on_dg(dg_codes, flip_frac, seed, n_pool):
    """Hebbian Hopfield over the REAL DG-space binary codes; noised cue per concept, settle,
    recover identity. Returns (recovered_idx [N], settled_states [N, n_dg], identity_acc).

    Reuses the positive-control's build_hopfield_weights + noisy_cue_sparse machinery. The
    settled state is what the DECODE reads (a faithful round-trip carries the attractor's
    reconstruction, not the clean stored code).
    """
    from research.runners.cortex_sparse_attractor_poscontrol_probe import (
        build_hopfield_weights, noisy_cue_sparse,
    )
    N, n_dg = dg_codes.shape
    # Native mean-removed codes for the attractor basis.
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
# STEP 3 -- learned CA1->cortex ridge decode (the COMPENSATION test)
# ===========================================================================
def fit_decoder(dg_codes, cortex_codes, ridge=1e-2):
    """Ridge linear map from DG-space code -> graded cortex code (W: n_dg -> dim).

    The CA1->cortex consolidation analogue, trained on the codebook the system knows.
    THE COMPENSATION mechanism: the map adapts to whatever the (noisy) DG produced.
    """
    X = dg_codes - dg_codes.mean(axis=1, keepdims=True)   # match native readout
    Y = cortex_codes
    n_dg = X.shape[1]
    A = X.T @ X + ridge * np.eye(n_dg)
    B = X.T @ Y
    W = np.linalg.solve(A, B)   # [n_dg, dim]
    return W


def roundtrip_pearson_onsubstrate(cortex_codes, S_orig, dg_codes, settled_states,
                                  ridge=1e-2):
    """Round-trip the SETTLED DG attractor states back to cortex via the learned decode,
    measure Pearson(S_orig, S') -- the LOAD-BEARING on-substrate number.

    Decoder is fit on the clean DG codes -> cortex (consolidation). Then the settled states
    (the attractor's reconstruction from a noised cue) are decoded and their cortex cosine
    matrix is compared to the original graded S.
    """
    W_dec = fit_decoder(dg_codes, cortex_codes, ridge=ridge)
    settled_centered = settled_states - settled_states.mean(axis=1, keepdims=True)
    decoded = settled_centered @ W_dec      # [N, dim]
    S_round = native_cos_matrix(decoded)
    N = cortex_codes.shape[0]
    iu = np.triu_indices(N, k=1)
    pearson = float(np.corrcoef(S_orig[iu], S_round[iu])[0, 1])
    return pearson, S_round


def roundtrip_permuted_baseline(cortex_codes, S_orig, dg_codes, settled_states,
                                seed, ridge=1e-2):
    """PERMUTED-S baseline: fit the decode on a ROW-SHUFFLED cortex codebook (so the decode
    target is a random concept). Pearson(S_orig, S'_perm) must be ~0 -> a high TRUE Pearson
    is a real similarity-preservation signal, not a pipeline artifact.
    """
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
# STEP 4 -- generalization ON-SUBSTRATE (in DG space)
# ===========================================================================
def run_generalization_onsubstrate(dg_codes, labels, props, n_clusters, per_cluster,
                                   seed, k_neighbours=3):
    """Held-out-neighbour property inference, but the similarity vote is over the REAL DG
    codes (in DG space). Returns the same dict shape as run_generalization. If graded
    similarity SURVIVES the DG encode, this PASSES; the orthogonal contrast must FAIL.
    """
    # run_generalization works on any [N, d] code matrix + labels + props.
    return run_generalization(dg_codes, labels, props, n_clusters, per_cluster,
                              seed, k_neighbours)


# ===========================================================================
# Per-seed driver
# ===========================================================================
def run_seed_full(seed, args):
    print(f"\n{'='*68}", flush=True)
    print(f"  ON-SUBSTRATE GATE -- SEED {seed}", flush=True)
    print(f"{'='*68}", flush=True)

    n_clusters = args.n_clusters
    per_cluster = args.per_cluster
    N = n_clusters * per_cluster
    dim = args.n_lang_input    # codebook lives in language_input space

    # ---------- synthetic graded codebook (in lang-input space) ----------
    codes, labels, S = build_graded_codebook(n_clusters, per_cluster, dim, seed,
                                             args.residual_frac)
    grad_stats = codebook_similarity_stats(codes, labels)
    print(f"  [graded codebook] N={N} ({n_clusters}x{per_cluster}) dim={dim} "
          f"(= n_lang_input)", flush=True)
    print(f"    within-cluster cos={grad_stats['within_cluster_cos_mean']:.3f} "
          f"between-cluster cos={grad_stats['between_cluster_cos_mean']:.3f} "
          f"margin={grad_stats['graded_margin']:.3f} graded={grad_stats['is_graded']}",
          flush=True)
    assert grad_stats["is_graded"], "graded codebook unit-check FAILED (within !>> between)"
    props = assign_properties(n_clusters, per_cluster, args.n_props, seed)

    # ---------- build the real spiking trisynaptic bridge ----------
    enc = TrisynapticDGEncoder(
        seed=seed, n_lang_input=args.n_lang_input, n_dg=args.n_dg,
        n_dg_pv_basket=args.n_dg_pv_basket, n_ca3=args.n_ca3, n_ca1=args.n_ca1,
        n_ec=args.n_ec, ca3_recurrent_density=args.ca3_recurrent_density,
        ca3_recurrent_weight=args.ca3_recurrent_weight, drive_pA=args.drive_pA,
        verbose=True)

    seed_rng = np.random.default_rng(seed + 777)

    # ============ STEPS 1-3: full sweep over the DG read k ============
    # The FAIR test of the compensation hypothesis: at EACH k, measure DG decorrelation +
    # reproducibility (STEP 1), bind/retrieve on the real DG codes (STEP 2), and the learned
    # CA1->cortex round-trip Pearson (STEP 3). Then select the operating point by the
    # LOAD-BEARING on-substrate round-trip Pearson -- so a GO cannot be missed by a bad k
    # choice, and a BOUNDARY means NO k recovers the round-trip.
    print("\n  [STEPS 1-3 -- per-k: DG separation + repro -> bind -> round-trip Pearson]",
          flush=True)
    k_list = [int(x) for x in args.k_list.split(",")]
    print("    %-5s %-11s %-9s %-9s %-9s %-9s %-9s" %
          ("k", "between-cos", "repro", "sparsity", "bind_id", "Pearson", "perm"), flush=True)
    sweep = []
    for k in k_list:
        d = measure_dg_decorrelation_and_repro(
            enc, codes, args.window, k, args.reset_steps, args.n_repro_pairs, seed_rng)
        dg_codes_k = d["_dg_codes"]
        # STEP 2 -- bind on the real DG codes at this k.
        _, settled_k, identity_k = run_binding_on_dg(
            dg_codes_k, args.flip_frac, seed, enc.n_dg)
        # STEP 3 -- learned decode round-trip Pearson (on the settled attractor states).
        pearson_k, _ = roundtrip_pearson_onsubstrate(codes, S, dg_codes_k, settled_k,
                                                     ridge=args.ridge)
        perm_k = roundtrip_permuted_baseline(codes, S, dg_codes_k, settled_k, seed,
                                             ridge=args.ridge)
        # decode-only ceiling (no cue noise): does the learned map invert the DG at all?
        clean_k, _ = roundtrip_pearson_onsubstrate(
            codes, S, dg_codes_k, dg_codes_k.astype(np.float64), ridge=args.ridge)
        rec = {
            "k": k,
            "dg_between_cos_mean": d["dg_between_cos_mean"],
            "dg_between_cos_max": d["dg_between_cos_max"],
            "dg_repro_mean": d["dg_repro_mean"],
            "dg_repro_min": d["dg_repro_min"],
            "dg_sparsity": d["dg_sparsity"],
            "dg_total_spikes_mean": d["dg_total_spikes_mean"],
            "binding_identity_acc": identity_k,
            "pearson_onsubstrate": pearson_k,
            "pearson_permuted_baseline": perm_k,
            "pearson_clean_dg_ceiling": clean_k,
            "_dg_codes": dg_codes_k,
            "_settled": settled_k,
        }
        sweep.append(rec)
        print("    %-5d %+11.3f %-9.3f %-9.3f %-9.3f %+9.3f %+9.3f" %
              (k, d["dg_between_cos_mean"], d["dg_repro_mean"], d["dg_sparsity"],
               identity_k, pearson_k, perm_k), flush=True)

    # Operating point: the k with the HIGHEST on-substrate round-trip Pearson (the load-
    # bearing number). This is the compensation hypothesis' best shot.
    chosen = max(sweep, key=lambda r: r["pearson_onsubstrate"])
    # Also identify the best DECORRELATING point (for the decorrelation gate) and the best
    # binding point (for honest reporting of the tension).
    decorr_pts = [r for r in sweep if r["dg_between_cos_mean"] < args.decorr_bar]
    best_decorr = (min(decorr_pts, key=lambda r: r["dg_between_cos_mean"])
                   if decorr_pts else min(sweep, key=lambda r: r["dg_between_cos_mean"]))
    best_bind = max(sweep, key=lambda r: r["binding_identity_acc"])

    dg_codes = chosen["_dg_codes"]
    settled = chosen["_settled"]
    identity_acc = chosen["binding_identity_acc"]
    pearson = chosen["pearson_onsubstrate"]
    pearson_perm = chosen["pearson_permuted_baseline"]
    pearson_clean = chosen["pearson_clean_dg_ceiling"]

    print(f"    -> operating k={chosen['k']} (max on-substrate Pearson): "
          f"between-cos={chosen['dg_between_cos_mean']:+.3f} repro={chosen['dg_repro_mean']:.3f} "
          f"bind={identity_acc:.3f} Pearson={pearson:+.3f}", flush=True)
    print(f"    [tension] best-decorrelating k={best_decorr['k']}: "
          f"between-cos={best_decorr['dg_between_cos_mean']:+.3f} "
          f"repro={best_decorr['dg_repro_mean']:.3f} "
          f"Pearson={best_decorr['pearson_onsubstrate']:+.3f}", flush=True)
    print(f"    [tension] best-binding k={best_bind['k']}: "
          f"bind={best_bind['binding_identity_acc']:.3f} "
          f"between-cos={best_bind['dg_between_cos_mean']:+.3f}", flush=True)

    # GATES. The decorrelation gate uses the BEST-decorrelating point (does ANY k separate
    # the graded codes?). Reproducibility + binding + C2 are evaluated at the OPERATING point
    # (the max-Pearson k -- the compensation hypothesis' best shot).
    dg_decorrelates = best_decorr["dg_between_cos_mean"] < args.decorr_bar
    dg_reproducible = chosen["dg_repro_mean"] >= args.repro_bar
    binding_ok = identity_acc >= args.binding_bar
    c2_ok = (pearson >= args.c2_bar) and (pearson > pearson_perm + 0.3)

    print(f"\n  [STEP 3 summary] operating-point Pearson(S,S') ON-SUBSTRATE = {pearson:+.3f}  "
          f"(permuted {pearson_perm:+.3f}; clean-DG ceiling {pearson_clean:+.3f}; "
          f"numpy proof +0.877)", flush=True)

    # ============ STEP 4: generalization ON-SUBSTRATE ============
    print("\n  [STEP 4 -- generalization on-substrate (DG-space similarity vote)]", flush=True)
    # Orthogonal control encoded through the SAME DG read (the decisive contrast).
    from research.runners.dual_cls_architecture_proof_probe import load_orthogonal_codes
    ortho_cortex = load_orthogonal_codes(seed, N, n_pool=max(2000, args.n_lang_input),
                                         pattern_size=100)
    # The orthogonal cortex codes live in a different dim; we test generalization on the
    # DG-encoded versions of BOTH graded and orthogonal cortex codes for an apples-to-apples
    # on-substrate contrast. Encode orthogonal codes through the bridge too (project them to
    # lang-input space first via a fixed positive map so they are valid drives).
    rng_o = np.random.RandomState(seed * 71 + 5)
    Pmap = np.abs(rng_o.standard_normal((ortho_cortex.shape[1], args.n_lang_input)))
    ortho_lang = ortho_cortex @ Pmap        # [N, n_lang_input], non-negative-ish
    ortho_lang = ortho_lang - ortho_lang.mean(axis=1, keepdims=True)
    ortho_lang = ortho_lang / (np.linalg.norm(ortho_lang, axis=1, keepdims=True) + 1e-12)
    dg_ortho, _ = enc.encode_codebook_dg(ortho_lang, args.window, chosen["k"],
                                         args.reset_steps)

    gen_graded = run_generalization_onsubstrate(dg_codes, labels, props,
                                                n_clusters, per_cluster, seed,
                                                args.k_neighbours)
    gen_ortho = run_generalization_onsubstrate(dg_ortho, labels, props,
                                               n_clusters, per_cluster, seed,
                                               args.k_neighbours)
    gen_perm = run_generalization_permuted(dg_codes, labels, props, n_clusters,
                                           per_cluster, seed, args.k_neighbours)
    chance = gen_graded["chance"]
    print(f"    graded(DG)    acc={gen_graded['accuracy']:.3f} "
          f"(chance={chance:.3f}, {gen_graded['ratio_vs_chance']:.1f}x)", flush=True)
    print(f"    orthogonal(DG) acc={gen_ortho['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)
    print(f"    permuted-S(DG) acc={gen_perm['accuracy']:.3f}  (MUST collapse to chance)",
          flush=True)
    a1 = gen_graded["accuracy"] >= args.a1_bar
    a2 = gen_ortho["accuracy"] <= 1.5 * chance
    a3 = gen_perm["accuracy"] <= 1.5 * chance

    # ---------- per-seed gates ----------
    gates = {
        "dg_decorrelates": bool(dg_decorrelates),
        "dg_reproducible": bool(dg_reproducible),
        "binding": bool(binding_ok),
        "c2_roundtrip_pearson": bool(c2_ok),
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
        "sweep_per_k": sweep,
        "operating_k": chosen["k"],
        "best_decorrelating_k": best_decorr["k"],
        "best_binding_k": best_bind["k"],
        "step1_dg": {
            "dg_between_cos_mean_at_operating": chosen["dg_between_cos_mean"],
            "dg_between_cos_min_any_k": best_decorr["dg_between_cos_mean"],
            "dg_repro_mean_at_operating": chosen["dg_repro_mean"],
            "dg_repro_min_at_operating": chosen["dg_repro_min"],
            "dg_sparsity_at_operating": chosen["dg_sparsity"],
            "dg_total_spikes_mean_at_operating": chosen["dg_total_spikes_mean"],
        },
        "step2_binding": {
            "identity_acc_at_operating": identity_acc,
            "identity_acc_best_any_k": best_bind["binding_identity_acc"],
            "flip_frac": args.flip_frac,
        },
        "step3_roundtrip": {
            "pearson_onsubstrate": pearson,
            "pearson_permuted_baseline": pearson_perm,
            "pearson_clean_dg_decode_ceiling": pearson_clean,
            "pearson_onsubstrate_max_any_k": max(r["pearson_onsubstrate"] for r in sweep),
            "numpy_proof_reference": 0.877,
        },
        "step4_generalization": {
            "graded": gen_graded,
            "orthogonal": gen_ortho,
            "permuted": gen_perm,
            "chance": chance,
        },
        "gates": gates,
    }


def main():
    p = argparse.ArgumentParser(description="Dual-CLS on-substrate gate probe")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--smoke", action="store_true",
                   help="tiny bridge + tiny codebook for harness verification (fast)")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--per-cluster", type=int, default=5)
    p.add_argument("--n-props", type=int, default=4)
    p.add_argument("--k-neighbours", type=int, default=3)
    p.add_argument("--residual-frac", type=float, default=0.55)
    # bridge sizing (defaults match the validated P1 / DG-CA3 probe scale)
    p.add_argument("--n-lang-input", type=int, default=512)
    p.add_argument("--n-ec", type=int, default=160)
    p.add_argument("--n-dg", type=int, default=600)
    p.add_argument("--n-dg-pv-basket", type=int, default=180)
    p.add_argument("--n-ca3", type=int, default=300)
    p.add_argument("--n-ca1", type=int, default=120)
    p.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    p.add_argument("--ca3-recurrent-weight", type=float, default=2.0)
    p.add_argument("--drive-pA", type=float, default=800.0)
    # DG read
    p.add_argument("--window", type=int, default=200, help="DG accumulation window (steps)")
    p.add_argument("--k-list", default="40,80,150,300",
                   help="DG k-WTA sizes to sweep")
    p.add_argument("--reset-steps", type=int, default=40)
    p.add_argument("--n-repro-pairs", type=int, default=8)
    p.add_argument("--flip-frac", type=float, default=0.1, help="binding cue noise")
    p.add_argument("--ridge", type=float, default=1e-2)
    # gate bars
    p.add_argument("--decorr-bar", type=float, default=0.40,
                   help="DG between-cos must be below this to count as decorrelated")
    p.add_argument("--repro-bar", type=float, default=0.70)
    p.add_argument("--binding-bar", type=float, default=0.70)
    p.add_argument("--c2-bar", type=float, default=0.70)
    p.add_argument("--a1-bar", type=float, default=0.70)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.smoke:
        # tiny: small bridge + small codebook to verify the harness end-to-end fast.
        args.n_clusters = 4
        args.per_cluster = 3
        args.n_lang_input = 128
        args.n_ec = 48
        args.n_dg = 200
        args.n_dg_pv_basket = 60
        args.n_ca3 = 100
        args.n_ca1 = 60
        args.window = 80
        args.k_list = "5,10,20,40,80"
        args.n_repro_pairs = 3
        os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    backend = os.environ.get("SIM_BACKEND", "auto")
    t_start = time.time()
    print(f"[dual-CLS on-substrate gate] seeds={seeds} backend={backend} smoke={args.smoke}",
          flush=True)

    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = run_seed_full(seed, args)

    # ---------- overall verdict ----------
    def all_gate(g):
        return all(per_seed[str(s)]["gates"][g] for s in seeds)

    g_decorr = all_gate("dg_decorrelates")
    g_repro = all_gate("dg_reproducible")
    g_bind = all_gate("binding")
    g_c2 = all_gate("c2_roundtrip_pearson")
    g_a1 = all_gate("a1_graded_generalizes")
    g_a2 = all_gate("a2_orthogonal_collapses")
    g_a3 = all_gate("a3_permuted_collapses")

    # GO requires: DG decorrelates AND the learned decode recovers the round-trip (C2)
    # AND binding AND generalization (a1/a2/a3) on-substrate, multi-seed. dg_reproducible
    # is REPORTED but the GO hypothesis is precisely that the learned decode can COMPENSATE
    # for low reproducibility -- so it is informative, not a hard gate (C2 is the real test
    # of whether compensation worked).
    if g_decorr and g_c2 and g_bind and g_a1 and g_a2 and g_a3:
        verdict = "GO"
    elif g_decorr and g_bind and not g_c2:
        verdict = "BOUNDARY_roundtrip_not_recovered_onsubstrate"
    elif (not g_decorr) and g_c2:
        # decode rescued similarity even though DG didn't formally decorrelate -- nuance.
        verdict = "PARTIAL_decode_rescues_without_decorrelation"
    elif not g_decorr:
        verdict = "BOUNDARY_dg_does_not_decorrelate_graded"
    elif not g_bind:
        verdict = "BOUNDARY_binding_fails_onsubstrate"
    else:
        verdict = "BOUNDARY_generalization_not_similarity_driven_onsubstrate"

    pearsons = [per_seed[str(s)]["step3_roundtrip"]["pearson_onsubstrate"] for s in seeds]
    pearsons_perm = [per_seed[str(s)]["step3_roundtrip"]["pearson_permuted_baseline"]
                     for s in seeds]
    pearsons_clean = [per_seed[str(s)]["step3_roundtrip"]["pearson_clean_dg_decode_ceiling"]
                      for s in seeds]
    dg_between = [per_seed[str(s)]["step1_dg"]["dg_between_cos_mean_at_operating"]
                  for s in seeds]
    dg_between_min = [per_seed[str(s)]["step1_dg"]["dg_between_cos_min_any_k"]
                      for s in seeds]
    dg_repro = [per_seed[str(s)]["step1_dg"]["dg_repro_mean_at_operating"] for s in seeds]
    bind_acc = [per_seed[str(s)]["step2_binding"]["identity_acc_at_operating"]
                for s in seeds]
    gen_graded = [per_seed[str(s)]["step4_generalization"]["graded"]["accuracy"]
                  for s in seeds]
    gen_ortho = [per_seed[str(s)]["step4_generalization"]["orthogonal"]["accuracy"]
                 for s in seeds]

    summary = {
        "verdict": verdict,
        "seeds": seeds,
        "backend": backend,
        "smoke": bool(args.smoke),
        "gates_all_seeds": {
            "dg_decorrelates": g_decorr, "dg_reproducible": g_repro,
            "binding": g_bind, "c2_roundtrip_pearson": g_c2,
            "a1_graded_generalizes": g_a1, "a2_orthogonal_collapses": g_a2,
            "a3_permuted_collapses": g_a3,
        },
        "load_bearing": {
            "dg_between_cos_at_operating_per_seed": dg_between,
            "dg_between_cos_at_operating_mean": float(np.mean(dg_between)),
            "dg_between_cos_min_any_k_per_seed": dg_between_min,
            "dg_between_cos_min_any_k_mean": float(np.mean(dg_between_min)),
            "dg_repro_per_seed": dg_repro,
            "dg_repro_mean": float(np.mean(dg_repro)),
            "binding_identity_acc_per_seed": bind_acc,
            "roundtrip_pearson_onsubstrate_per_seed": pearsons,
            "roundtrip_pearson_onsubstrate_mean": float(np.mean(pearsons)),
            "roundtrip_pearson_permuted_per_seed": pearsons_perm,
            "roundtrip_pearson_permuted_mean": float(np.mean(pearsons_perm)),
            "roundtrip_pearson_clean_dg_ceiling_per_seed": pearsons_clean,
            "numpy_proof_reference": 0.877,
            "generalization_graded_per_seed": gen_graded,
            "generalization_orthogonal_per_seed": gen_ortho,
        },
        "elapsed_total_s": time.time() - t_start,
    }

    print(f"\n{'='*68}", flush=True)
    print(f"  OVERALL VERDICT: {verdict}", flush=True)
    print(f"  DG decorrelates graded (all seeds):   {g_decorr}  "
          f"(between-cos mean {np.mean(dg_between):+.3f})", flush=True)
    print(f"  DG same-input reproducible (REPORTED): {g_repro}  "
          f"(repro mean {np.mean(dg_repro):.3f})", flush=True)
    print(f"  Binding on DG (all seeds):            {g_bind}  "
          f"(identity mean {np.mean(bind_acc):.3f})", flush=True)
    print(f"  C2 round-trip Pearson recovered:      {g_c2}  "
          f"(Pearson mean {np.mean(pearsons):+.3f} vs permuted {np.mean(pearsons_perm):+.3f}; "
          f"numpy proof +0.877)", flush=True)
    print(f"  A1 graded generalizes on-substrate:   {g_a1}", flush=True)
    print(f"  A2 orthogonal collapses:              {g_a2}", flush=True)
    print(f"  A3 permuted collapses:                {g_a3}", flush=True)
    print(f"  >>> LOAD-BEARING: on-substrate round-trip Pearson = {np.mean(pearsons):+.3f} "
          f"(numpy +0.877; permuted {np.mean(pearsons_perm):+.3f})", flush=True)
    print(f"  >>> DG decorrelation = {np.mean(dg_between):+.3f}  "
          f"DG reproducibility = {np.mean(dg_repro):.3f}", flush=True)
    print(f"  Total elapsed: {summary['elapsed_total_s']:.1f}s", flush=True)
    print(f"{'='*68}\n", flush=True)

    out_data = {"summary": summary, "per_seed": per_seed}

    if args.out is None:
        raw_dir = os.path.join(_REPO, "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "smoke" if args.smoke else "multiseed"
        args.out = os.path.join(raw_dir, f"_dual_cls_onsubstrate_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)
    return out_data


if __name__ == "__main__":
    main()
