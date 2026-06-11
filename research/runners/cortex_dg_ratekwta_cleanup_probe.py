"""Rate-accumulated k-WTA DG cleanup probe — the FINAL DISTINCT DG approach.

Context (multiply-confirmed boundary from prior probes):
  - Storkey/pseudo-inverse Hopfield on RAW correlated denoise64 codes: NEGATIVE
    (research/findings/2026-06-11-cortex-storkey-ca3-cleanup-NEGATIVE.md):
    locality wall — pinv (global host inverse) recovers 1.000 but no LOCAL rule can
    remove the common mode; Storkey local rule gives 0.142 (chance 0.062) on correlated codes.
  - Spiking DG→CA3 trisynaptic loop (stock): NEGATIVE
    (research/findings/2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md):
    the spiking DG read is sub-reproducible — driving the SAME clean code twice yields DG cosines
    0.04–0.15 (near-orthogonal to itself) because DG fires only ~15–62 spikes/600 neurons, so
    which cells "win" is dominated by OU noise + spike-timing chaos, NOT the input.

THIS probe tests the DISTINCT mechanistic fix named in the NEGATIVE finding (option 2a/2c):
  "far more DG spikes per read (rate-coded, not 1-spike-per-cell) with a hard k-WTA on the
   ACCUMULATED RATE rather than instantaneous spikes" + "reduce DG stochasticity (OU off +
   deterministic winner read)"

HYPOTHESIS: if we ACCUMULATE per-DG-neuron spike counts over a window W (instead of reading
instantaneous spikes), then top-k BY COUNT (rather than who-fired-first), the winner set is
determined by input-driven RATE not by noise-dominated TIMING → DG read reproducibility should
rise and be input-determined.

PRIMARY MEASUREMENT: same-input DG reproducibility (cosine of two fresh rate-accumulated reads
of the SAME clean code). TARGET ≥ 0.7 to proceed to CA3 attractor.

KEY TENSION UNDER TEST: the "more DG spikes / larger k" lever that improves reproducibility may
DESTROY the between-concept separation (large k → many shared neurons → high between-concept
cosine). This probe characterises BOTH simultaneously and maps whether a sweet-spot exists where
repro ≥ 0.7 AND between-concept cosine is low.

SUBSTRATE: the SAME build_biological_brain_regions(enable_hippocampus_consolidation=True) bridge
the validated P1 trisynaptic result and the DG-CA3 NEGATIVE used. NO sim/ edits. The rate-kWTA
read is a READOUT OPERATION (accumulate counts + argsort + pick top-k) applied to the bridge's
already-computed spike state — a brain-based readout, not a host computation of the match.

Brain-based provenance:
  - Rate-accumulated k-WTA: accumulating firing rate over a window + competitive selection =
    what a downstream population with slow temporal integration + WTA does. This is a readout,
    NOT a host shortcut.
  - OU-off is a MODELING CHOICE (reduces intrinsic noise), NOT a shortcut. Labeled explicitly.
  - argmax is the IDEALIZATION REFERENCE only (the composer's god's-eye cleanup). CA3 settle =
    the brain-based cleanup; cosine-to-codebook is SCORING (grading), not the mechanism.
  - No sim/ edits. Runtime OU disabled by setting bridge.ou_noise_std = 0.0 (no module edit).

Run (CPU, numpy backend):
  SIM_BACKEND=numpy python -m research.runners.cortex_dg_ratekwta_cleanup_probe --seeds 42,43,44
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

CACHE = os.path.join(_REPO, "research", "findings", "raw",
                     "activity_level_integration_cache", "denoise64_seed%d.npz")

# ---- global so the proj-dim helper is accessible from TrisynapticRate --------
_PROJ_DIM = 512


def proj_dim_global():
    return _PROJ_DIM


# ---------------------------------------------------------------------------
# Code loading (identical to the prior probe — same denoise64 source)
# ---------------------------------------------------------------------------
def load_real_codes(seed, proj_dim, rng):
    """Load brain's REAL denoise64 concept codes → signed real codes [V, D].

    Identical treatment to cortex_dg_ca3_cleanup_probe: mean per word, random-Gaussian
    project to proj_dim (preserves cosines), mean-center + unit-normalize. NO decorrelation —
    these are the RAW correlated codes (the DG's job to handle)."""
    d = np.load(CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)
    if proj_dim and proj_dim > 0:
        P = rng.standard_normal((raw.shape[1], proj_dim)) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    V = codes.shape[0]
    cs = [float(codes[i] @ codes[k]) for i in range(V) for k in range(i + 1, V)]
    return ws, codes, float(np.mean(cs)) if cs else 0.0


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _mean_offdiag_cos(M):
    """Mean off-diagonal cosine of rows in matrix M."""
    V = M.shape[0]
    cs = [_cos(M[i], M[j]) for i in range(V) for j in range(i + 1, V)]
    return float(np.mean(cs)) if cs else 0.0


# ---------------------------------------------------------------------------
# Rate-accumulated k-WTA DG read harness
# ---------------------------------------------------------------------------
class RateKWTABridge:
    """Wraps the project's trisynaptic bridge with a RATE-ACCUMULATED k-WTA DG read.

    The bridge substrate is identical to cortex_dg_ca3_cleanup_probe.TrisynapticCleanup:
    build_biological_brain_regions(enable_hippocampus_consolidation=True), same defaults.
    The ONLY difference is the read method: instead of _run_read's per-step snapshot (which
    gives a cumulative count anyway), we now expose explicit window + k control and can toggle
    OU noise at runtime via bridge.ou_noise_std = 0/restore."""

    def __init__(self, seed, n_lang_input, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ec,
                 ca3_recurrent_density, ca3_recurrent_weight, drive_pA, verbose=True):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from research.runners.text_minimal_isolation import build_biological_brain_regions
        from sim.backend import get_backend
        self._xp, _ = get_backend()
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
        self.bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        rm = self.bridge.region_manager
        self.lang_idx = np.asarray(rm.indices("language_input"), dtype=np.int64)
        self.dg_idx   = np.asarray(rm.indices("dg"),             dtype=np.int64)
        self.ca3_idx  = np.asarray(rm.indices("ca3"),            dtype=np.int64)
        self.cfg = cfg
        # store the default OU noise std for restoration after OU-OFF runs
        self._ou_noise_std_default = float(getattr(self.bridge, 'ou_noise_std', 0.0) or 0.0)
        self._ou_mean_default      = float(getattr(self.bridge, 'ou_mean',      0.0) or 0.0)
        log("  [bridge] built %.1fs; %d neurons %d synapses (DG=%d CA3=%d)"
            % (time.time() - t0, cfg.num_neurons, int(self.bridge.cp_connections.nnz), n_dg, n_ca3))

        # Fixed positive projection from code-space [D] to language_input drive [n_lang_input]
        rngp = np.random.default_rng(seed + 333)
        self._Wdrive = np.abs(rngp.standard_normal((proj_dim_global(), self.lang_idx.shape[0])))

    # --- OU control ---
    def set_ou_off(self):
        """Disable OU stochastic noise (set std to 0) while keeping the mean drive.
        This is a MODELING CHOICE, not a shortcut — reduces intrinsic noise so the
        k-WTA winner set is determined by input-driven rate, not noise-dominated timing."""
        self.bridge.ou_noise_std = 0.0

    def set_ou_on(self):
        """Restore OU noise to bridge default."""
        self.bridge.ou_noise_std = self._ou_noise_std_default

    # --- drive helpers ---
    def _render_drive(self, code_vec):
        """code-space vector → language_input drive [n_lang_input], pA, non-negative."""
        x = np.asarray(code_vec, dtype=np.float64) @ self._Wdrive
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

    # --- rate-accumulated k-WTA read (THE NEW READ) ---
    def rate_kwta_dg_read(self, code_vec, window_steps, k, reset_steps=40):
        """Rate-accumulated k-WTA DG read.

        1. Reset transients (reset_steps with zero drive).
        2. Apply language_input drive from code_vec.
        3. Step for window_steps, ACCUMULATING per-DG-neuron spike COUNTS.
        4. Pick the top-k DG neurons BY ACCUMULATED COUNT.
        5. Return a binary indicator vector (1 at top-k positions, 0 elsewhere) — the DG code.

        This is a brain-based readout: temporal rate integration + competitive selection. The
        readout is a binary sparse vector in DG-neuron space (like a pattern-separation ensemble).
        """
        from sim.backend import to_host
        xp = self._xp
        # reset
        self._clear_drive()
        for _ in range(reset_steps):
            self._step()
        # drive + accumulate
        self._set_lang_drive(self._render_drive(code_vec))
        dg_reg = xp.asarray(self.dg_idx)
        counts = xp.zeros(len(self.dg_idx), dtype=xp.float32)
        for _ in range(window_steps):
            self._step()
            counts += self.bridge.cp_firing_states[dg_reg].astype(xp.float32)
        self._clear_drive()
        counts_np = to_host(counts).astype(np.float32)
        # top-k by accumulated count — binary indicator
        if k >= len(counts_np):
            code = np.ones(len(counts_np), dtype=np.float32)
        elif k <= 0:
            code = np.zeros(len(counts_np), dtype=np.float32)
        else:
            thresh_idx = np.argsort(counts_np)[::-1][k - 1]
            thresh = counts_np[thresh_idx]
            # use strict threshold to pick winners; if there are ties at the threshold we take
            # the top-k by argsort (deterministic tie-breaking by index)
            top_k_indices = np.argsort(counts_np)[::-1][:k]
            code = np.zeros(len(counts_np), dtype=np.float32)
            code[top_k_indices] = 1.0
        total_spikes = int(counts_np.sum())
        return code, total_spikes, counts_np

    def raw_count_dg_read(self, code_vec, window_steps, reset_steps=40):
        """Return raw accumulated counts (float32 vector) — for diagnosing the rate distribution."""
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
        return to_host(counts).astype(np.float32)


# ---------------------------------------------------------------------------
# Reproducibility + separation sweep
# ---------------------------------------------------------------------------
def measure_repro_and_sep(bridge, codes, window_steps, k_list, ou_off, seed_rng,
                           reset_steps=40, n_repro_pairs=6):
    """For each k in k_list: measure same-input reproducibility AND between-concept separation.

    Returns a dict keyed by k, each with:
      repro_mean, repro_min  — same-input cosine across n_repro_pairs pairs
      sep_mean               — between-concept cosine across all concept pairs (lower = better)
      total_spikes_mean      — mean total spikes per read (diagnostic)
    """
    if ou_off:
        bridge.set_ou_off()
    else:
        bridge.set_ou_on()

    V = codes.shape[0]
    results = {}
    for k in k_list:
        # a) Reproducibility: for each of n_repro_pairs pairs, pick a random concept,
        #    do two FRESH reads (full reset between them), compute cosine.
        repro_vals = []
        spike_totals = []
        for _ in range(n_repro_pairs):
            ci = int(seed_rng.integers(V))
            code1, sp1, _ = bridge.rate_kwta_dg_read(codes[ci], window_steps, k, reset_steps)
            code2, sp2, _ = bridge.rate_kwta_dg_read(codes[ci], window_steps, k, reset_steps)
            repro_vals.append(_cos(code1, code2))
            spike_totals.append((sp1 + sp2) / 2.0)

        # b) Separation: for each concept, one read; measure between-concept cosines.
        concept_codes = []
        for ci in range(V):
            code, sp, _ = bridge.rate_kwta_dg_read(codes[ci], window_steps, k, reset_steps)
            concept_codes.append(code)
            spike_totals.append(float(sp))
        concept_codes_arr = np.stack(concept_codes)
        sep_cos = _mean_offdiag_cos(concept_codes_arr)

        results[k] = {
            "k": k,
            "window": window_steps,
            "ou_off": bool(ou_off),
            "repro_mean": float(np.mean(repro_vals)),
            "repro_min":  float(np.min(repro_vals)),
            "repro_vals": [float(x) for x in repro_vals],
            "sep_mean":   float(sep_cos),
            "total_spikes_mean": float(np.mean(spike_totals)),
        }
    return results


# ---------------------------------------------------------------------------
# CA3 Hebbian autoassociator (vanilla — used IF DG repro ≥ 0.7)
# ---------------------------------------------------------------------------
class VanillaHopfield:
    """Vanilla Hebbian Hopfield autoassociator built on the DG-space rate-kWTA BINARY codes.

    W = (1/n) * sum_i (x_i - 0.5) (x_i - 0.5)^T   (mean-centered binary codes)
    Settle: z = W @ z, normalize; read nearest stored code by cosine.
    This is brain-based via CA3 autoassociativity (D.13) — the attractor is defined
    by learned recurrent weights (Marr 1971), not by a god's-eye codebook lookup."""

    def __init__(self, dg_codes, iters=10):
        """dg_codes: [V, n_dg] binary float32 rate-kWTA codes."""
        self.dg_codes = dg_codes.astype(np.float64)
        n = dg_codes.shape[1]
        # mean-center before outer product (reduces common-mode bias on sparse binary codes)
        centered = self.dg_codes - 0.5
        self.W = (centered.T @ centered) / float(n)  # [n_dg, n_dg]
        self.iters = iters

    def cleanup(self, dg_query):
        """dg_query: the rate-kWTA code of a noisy/partial cue. Returns the index of nearest
        stored concept (the CA3 attractor settled on)."""
        z = dg_query.astype(np.float64) - 0.5  # mean-center
        for _ in range(self.iters):
            z = self.W @ z
            nz = np.linalg.norm(z)
            if nz < 1e-9:
                break
            z = z / nz
        # read nearest stored DG code (by cosine in centered space)
        centered = self.dg_codes - 0.5
        norms = np.linalg.norm(centered, axis=1) + 1e-12
        sims = (centered @ z) / (norms * (np.linalg.norm(z) + 1e-12))
        return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Cue generators (same as prior probe for apple-to-apple comparison)
# ---------------------------------------------------------------------------
def noisy_cue(code, rng, noise=0.6):
    sigma = noise * float(np.std(code))
    return code + rng.standard_normal(code.shape) * sigma


def partial_cue(code, rng, keep_frac=0.4):
    D = code.shape[0]
    keep = rng.choice(D, size=max(1, int(round(keep_frac * D))), replace=False)
    out = np.zeros_like(code)
    out[keep] = code[keep]
    return out


def argmax_cleanup(cue, codes):
    sims = codes @ cue / (np.linalg.norm(codes, axis=1) * (np.linalg.norm(cue) + 1e-12) + 1e-12)
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Full cleanup test suite (only run if DG repro ≥ 0.7)
# ---------------------------------------------------------------------------
def run_cleanup_tests(bridge, codes, best_window, best_k, ou_off,
                      n_trials, rng, noise_levels, keep_fracs, verbose=True):
    """Run TEST 1 (parity), TEST 2 (completion), and anti-cheats on the best (W, k, OU) setting.

    Returns dict of results. The Hopfield is built on the DG-space BINARY codes of the clean
    concepts. Cues are routed through EC→DG rate-kWTA first, then cleaned up by the Hopfield.
    The argmax baseline cleans up in CODE-SPACE (the composer's god's-eye filter).
    """
    log = print if verbose else (lambda *a, **k: None)
    if ou_off:
        bridge.set_ou_off()
    else:
        bridge.set_ou_on()

    V = codes.shape[0]
    reset_steps = 40

    # Build the stored DG codes (clean concept reads)
    log("  [cleanup] computing stored DG codes (best W=%d k=%d ou_off=%s)..." %
        (best_window, best_k, ou_off))
    stored_dg = []
    for ci in range(V):
        code, _, _ = bridge.rate_kwta_dg_read(codes[ci], best_window, best_k, reset_steps)
        stored_dg.append(code)
    stored_dg = np.stack(stored_dg)  # [V, n_dg] binary float32

    log("  [cleanup] DG stored codes: between-concept cos=%.3f  (raw between-code cos ≈ 0.81)"
        % _mean_offdiag_cos(stored_dg))

    # Vanilla Hopfield on DG-space codes
    hop = VanillaHopfield(stored_dg)

    # --- TEST 1: parity on full noised cues ---
    test1 = {}
    for noise in noise_levels:
        n_arg = n_hop = n_tot = 0
        for _ in range(n_trials):
            i = int(rng.integers(V))
            cue_code = noisy_cue(codes[i], rng, noise=noise)
            # Route cue through EC→DG rate-kWTA
            cue_dg, _, _ = bridge.rate_kwta_dg_read(cue_code, best_window, best_k, reset_steps)
            # Brain-based cleanup: Hopfield on DG space
            n_hop += int(hop.cleanup(cue_dg) == i)
            # Reference: argmax in code-space
            n_arg += int(argmax_cleanup(cue_code, codes) == i)
            n_tot += 1
        test1[noise] = {"noise": noise, "argmax_acc": n_arg / n_tot,
                        "dg_hop_acc": n_hop / n_tot, "n": n_tot, "chance": 1.0 / V}

    # --- TEST 2: completion on partial cues ---
    test2 = {}
    for kf in keep_fracs:
        n_arg = n_hop = n_tot = 0
        for _ in range(n_trials):
            i = int(rng.integers(V))
            cue_code = partial_cue(codes[i], rng, keep_frac=kf)
            cue_dg, _, _ = bridge.rate_kwta_dg_read(cue_code, best_window, best_k, reset_steps)
            n_hop += int(hop.cleanup(cue_dg) == i)
            n_arg += int(argmax_cleanup(cue_code, codes) == i)
            n_tot += 1
        test2[kf] = {"keep_frac": kf, "argmax_acc": n_arg / n_tot,
                     "dg_hop_acc": n_hop / n_tot, "n": n_tot, "chance": 1.0 / V}

    # --- ANTI-CHEAT 1: zero Hopfield recurrent weights → cleanup collapses ---
    hop_zero = VanillaHopfield(stored_dg)
    hop_zero.W[:] = 0.0
    n_zero = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue_code = noisy_cue(codes[i], rng, noise=noise_levels[0])
        cue_dg, _, _ = bridge.rate_kwta_dg_read(cue_code, best_window, best_k, reset_steps)
        n_zero += int(hop_zero.cleanup(cue_dg) == i)
    anticheat1 = {"zeroed_hop_acc": n_zero / n_trials, "n": n_trials}

    # --- ANTI-CHEAT 2: shuffled codebook → chance ---
    # Shuffle the label→stored_dg mapping so stored[i] is now some random OTHER concept's DG code
    shuffle_rng = np.random.default_rng(9999)
    perm = shuffle_rng.permutation(V)
    shuffled_dg = stored_dg[perm]  # each row is a different concept's code mapped to wrong label
    hop_shuf = VanillaHopfield(shuffled_dg)
    n_shuf = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue_code = noisy_cue(codes[i], rng, noise=noise_levels[0])
        cue_dg, _, _ = bridge.rate_kwta_dg_read(cue_code, best_window, best_k, reset_steps)
        n_shuf += int(hop_shuf.cleanup(cue_dg) == i)
    anticheat2 = {"shuffled_hop_acc": n_shuf / n_trials, "n": n_trials, "chance": 1.0 / V}

    return {
        "best_window": best_window, "best_k": best_k, "ou_off": ou_off,
        "stored_dg_sep_cos": float(_mean_offdiag_cos(stored_dg)),
        "test1": test1,
        "test2": test2,
        "anticheat1_zeroed_recurrent": anticheat1,
        "anticheat2_shuffled_codebook": anticheat2,
    }


# ---------------------------------------------------------------------------
# Per-seed driver
# ---------------------------------------------------------------------------
def run_seed(seed, args):
    print("\n" + "=" * 72, flush=True)
    print("=== RATE-kWTA DG CLEANUP PROBE (seed %d) ===" % seed, flush=True)
    print("=" * 72, flush=True)

    if not os.path.exists(CACHE % seed):
        print("[probe] MISSING denoise64 cache %s" % (CACHE % seed), flush=True)
        return None

    global _PROJ_DIM
    _PROJ_DIM = args.proj_dim

    words, codes, raw_cos = load_real_codes(seed, args.proj_dim, np.random.default_rng(seed))
    V = codes.shape[0]
    print("[codes] V=%d D=%d  between-code cosine (RAW correlated) = %.3f" % (V, args.proj_dim, raw_cos),
          flush=True)

    build_kw = dict(
        n_lang_input=args.n_lang_input, n_dg=args.n_dg, n_dg_pv_basket=args.n_dg_pv_basket,
        n_ca3=args.n_ca3, n_ca1=args.n_ca1, n_ec=args.n_ec,
        ca3_recurrent_density=args.ca3_recurrent_density,
        ca3_recurrent_weight=args.ca3_recurrent_weight,
        drive_pA=args.drive_pA,
    )

    # Build bridge (shared across all sweep conditions)
    bridge = RateKWTABridge(seed=seed, verbose=True, **build_kw)
    print("[bridge] default OU noise std=%.4f  mean=%.4f  (will set to 0 for OU-OFF variant)"
          % (bridge._ou_noise_std_default, bridge._ou_mean_default), flush=True)

    windows = [int(w) for w in args.windows.split(",")]
    k_list  = [int(k) for k in args.k_list.split(",")]
    n_dg_sizes = [args.n_dg]
    if args.n_dg_extra > 0 and args.n_dg_extra != args.n_dg:
        n_dg_sizes.append(args.n_dg_extra)

    # SWEEP: (window, OU_on/off, k)
    # We run the full sweep on the primary n_dg. If n_dg_extra is set, run the best setting there too.
    sweep_results = []
    best_repro = -1.0
    best_setting = None

    seed_rng = np.random.default_rng(seed + 777)

    print("\n--- REPRO + SEPARATION SWEEP (window, OU, k) ---", flush=True)
    print("%-8s %-6s %-6s %-10s %-10s %-12s %-16s" %
          ("window", "OU_off", "k", "repro_mean", "repro_min", "sep_cos", "total_spikes"), flush=True)

    for window in windows:
        for ou_off in [False, True]:
            r = measure_repro_and_sep(bridge, codes, window, k_list, ou_off,
                                      seed_rng, reset_steps=40, n_repro_pairs=args.n_repro_pairs)
            for k, d in sorted(r.items()):
                print("%-8d %-6s %-6d %-10.3f %-10.3f %-12.3f %-16.1f" %
                      (window, str(ou_off), k,
                       d["repro_mean"], d["repro_min"], d["sep_mean"], d["total_spikes_mean"]),
                      flush=True)
                entry = {"n_dg": args.n_dg, **d}
                sweep_results.append(entry)
                if d["repro_mean"] > best_repro:
                    best_repro = d["repro_mean"]
                    best_setting = entry

    print("\nBest repro across sweep: %.3f (W=%d k=%d OU_off=%s sep=%.3f)"
          % (best_repro, best_setting["window"], best_setting["k"],
             best_setting["ou_off"], best_setting["sep_mean"]), flush=True)

    # If n_dg_extra is requested, build a second bridge and sweep the best k/window setting
    large_dg_results = None
    if args.n_dg_extra > 0 and args.n_dg_extra != args.n_dg:
        print("\n--- LARGE DG (n_dg=%d) SWEEP at best (W=%d OU_off=%s) ---"
              % (args.n_dg_extra, best_setting["window"], best_setting["ou_off"]), flush=True)
        bkw2 = {**build_kw, "n_dg": args.n_dg_extra,
                "n_dg_pv_basket": int(args.n_dg_extra * args.n_dg_pv_basket / args.n_dg)}
        bridge2 = RateKWTABridge(seed=seed, verbose=False, **bkw2)
        seed_rng2 = np.random.default_rng(seed + 888)
        r2 = measure_repro_and_sep(bridge2, codes, best_setting["window"], k_list,
                                   best_setting["ou_off"], seed_rng2,
                                   reset_steps=40, n_repro_pairs=args.n_repro_pairs)
        large_dg_results = []
        for k, d in sorted(r2.items()):
            print("  n_dg=%d W=%d OU_off=%s k=%d  repro=%.3f sep=%.3f spikes=%.1f" %
                  (args.n_dg_extra, best_setting["window"], best_setting["ou_off"],
                   k, d["repro_mean"], d["sep_mean"], d["total_spikes_mean"]), flush=True)
            large_dg_results.append({"n_dg": args.n_dg_extra, **d})
        # update best if something is better
        for d in large_dg_results:
            if d["repro_mean"] > best_repro:
                best_repro = d["repro_mean"]
                best_setting = d

    # GATE CHECK: does any setting reach repro ≥ 0.7 with low sep?
    REPRO_TARGET = 0.70
    SEP_HIGH = 0.40  # if sep_cos > this, the codes are too similar for a Hopfield
    gate_repro_ok = best_repro >= REPRO_TARGET
    gate_sep_ok   = best_setting["sep_mean"] < SEP_HIGH if best_setting else False

    print("\n--- GATE CHECK ---", flush=True)
    print("  Repro target ≥ %.2f: BEST=%.3f => %s" %
          (REPRO_TARGET, best_repro, "PASS" if gate_repro_ok else "FAIL"), flush=True)
    print("  Sep < %.2f (at best repro setting): %.3f => %s" %
          (SEP_HIGH, best_setting["sep_mean"] if best_setting else 999.0,
           "OK" if gate_sep_ok else "HIGH_OR_NA"), flush=True)

    # --- SEPARATION vs REPRODUCIBILITY TENSION TABLE ---
    # Print table sorted by repro_mean
    sorted_all = sorted(sweep_results + (large_dg_results or []), key=lambda x: x["repro_mean"], reverse=True)
    print("\n--- TOP-10 settings by repro (separation-vs-reproducibility tension table) ---", flush=True)
    print("%-6s %-8s %-6s %-6s %-10s %-10s %-10s" %
          ("n_dg", "window", "OU_off", "k", "repro_mean", "sep_cos", "tension?"), flush=True)
    for d in sorted_all[:10]:
        tension = "HIGH_SEP" if d["sep_mean"] >= SEP_HIGH else ("GOOD" if d["repro_mean"] >= REPRO_TARGET else "LOW_REPRO")
        print("%-6d %-8d %-6s %-6d %-10.3f %-10.3f %-10s" %
              (d.get("n_dg", args.n_dg), d["window"], str(d["ou_off"]), d["k"],
               d["repro_mean"], d["sep_mean"], tension), flush=True)

    # CLEANUP TESTS (only if gate_repro_ok AND gate_sep_ok)
    cleanup_results = None
    if gate_repro_ok and gate_sep_ok:
        print("\n=== GATE CLEARED (repro ≥ %.2f, sep < %.2f) — RUNNING CLEANUP TESTS ===" %
              (REPRO_TARGET, SEP_HIGH), flush=True)
        # Use the bridge that matches best_setting's n_dg
        if best_setting.get("n_dg", args.n_dg) == args.n_dg:
            cb = bridge
        else:
            cb = bridge2

        cleanup_results = run_cleanup_tests(
            cb, codes,
            best_window=best_setting["window"],
            best_k=best_setting["k"],
            ou_off=best_setting["ou_off"],
            n_trials=args.n_trials,
            rng=np.random.default_rng(seed + 999),
            noise_levels=[0.0, 0.1, 0.2, 0.3],
            keep_fracs=[0.5, 0.35, 0.25, 0.15],
            verbose=True,
        )
        print("\nCLEANUP TEST 1 (parity, full noised cues):", flush=True)
        for noise, d in cleanup_results["test1"].items():
            print("  noise=%.2f: argmax=%.3f  DG-Hop=%.3f  chance=%.3f" %
                  (noise, d["argmax_acc"], d["dg_hop_acc"], d["chance"]), flush=True)
        print("CLEANUP TEST 2 (completion, partial cues):", flush=True)
        for kf, d in cleanup_results["test2"].items():
            print("  keep=%.2f: argmax=%.3f  DG-Hop=%.3f  chance=%.3f" %
                  (kf, d["argmax_acc"], d["dg_hop_acc"], d["chance"]), flush=True)
        print("ANTI-CHEAT 1 (zero recurrent weights): acc=%.3f  (intact see above)" %
              cleanup_results["anticheat1_zeroed_recurrent"]["zeroed_hop_acc"], flush=True)
        print("ANTI-CHEAT 2 (shuffled codebook):      acc=%.3f  (chance=%.3f)" %
              (cleanup_results["anticheat2_shuffled_codebook"]["shuffled_hop_acc"],
               cleanup_results["anticheat2_shuffled_codebook"]["chance"]), flush=True)
    elif gate_repro_ok and not gate_sep_ok:
        print("\n=== REPRO ≥ %.2f BUT SEP >= %.2f (separation-reproducibility tension) ===" %
              (REPRO_TARGET, SEP_HIGH), flush=True)
        print("  The separation-vs-reproducibility tension IS the fundamental result:", flush=True)
        print("  pushing k large enough to reach repro ≥ %.2f makes the DG codes too similar" %
              REPRO_TARGET, flush=True)
        print("  for a Hopfield attractor to distinguish concepts => no attractor will work.", flush=True)
    else:
        print("\n=== REPRO GATE NOT CLEARED (best=%.3f < %.2f) — cleanup tests SKIPPED ===" %
              (best_repro, REPRO_TARGET), flush=True)

    # VERDICT
    if gate_repro_ok and gate_sep_ok and cleanup_results is not None:
        # check cleanup parity
        parity_values = [d["dg_hop_acc"] for d in cleanup_results["test1"].values()]
        argmax_values = [d["argmax_acc"] for d in cleanup_results["test1"].values()]
        parity_ok = all(p >= a - 0.05 for p, a in zip(parity_values, argmax_values))
        completion_values = [d["dg_hop_acc"] for d in cleanup_results["test2"].values()
                             if d["keep_frac"] <= 0.35]
        argmax_part = [d["argmax_acc"] for d in cleanup_results["test2"].values()
                       if d["keep_frac"] <= 0.35]
        completion_ok = any(p > a for p, a in zip(completion_values, argmax_part))
        ac1 = cleanup_results["anticheat1_zeroed_recurrent"]["zeroed_hop_acc"]
        ac2 = cleanup_results["anticheat2_shuffled_codebook"]["shuffled_hop_acc"]
        ac1_ok = ac1 < max(parity_values) - 0.1
        ac2_ok = ac2 < max(parity_values) - 0.1
        if parity_ok and completion_ok and ac1_ok and ac2_ok:
            verdict = "GO"
        elif parity_ok:
            verdict = "PARTIAL"
        else:
            verdict = "NEGATIVE"
    elif gate_repro_ok and not gate_sep_ok:
        verdict = "NEGATIVE"  # tension = closed path
        print("  (separation-vs-reproducibility tension closes the spiking-DG distributed-cleanup path)", flush=True)
    else:
        verdict = "NEGATIVE"

    print("\n=== SEED %d VERDICT: %s ===" % (seed, verdict), flush=True)

    return {
        "seed": seed, "n_words": V, "D": args.proj_dim, "between_code_cos_raw": float(raw_cos),
        "sweep": sweep_results,
        "large_dg_sweep": large_dg_results,
        "best_repro": float(best_repro),
        "best_setting": best_setting,
        "gate_repro_ok": bool(gate_repro_ok),
        "gate_sep_ok": bool(gate_sep_ok),
        "cleanup_results": cleanup_results,
        "verdict": verdict,
        "build": build_kw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__[:100])
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma list, e.g. 42,43,44")
    ap.add_argument("--proj-dim",      type=int,   default=512)
    ap.add_argument("--n-lang-input",  type=int,   default=512)
    ap.add_argument("--n-ec",          type=int,   default=160)
    ap.add_argument("--n-dg",          type=int,   default=600,
                    help="primary DG size")
    ap.add_argument("--n-dg-extra",    type=int,   default=1200,
                    help="secondary DG size to test (0 = skip)")
    ap.add_argument("--n-dg-pv-basket",type=int,   default=180)
    ap.add_argument("--n-ca3",         type=int,   default=300)
    ap.add_argument("--n-ca1",         type=int,   default=120)
    ap.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    ap.add_argument("--ca3-recurrent-weight",  type=float, default=2.0)
    ap.add_argument("--drive-pA",      type=float, default=800.0,
                    help="drive current — higher than stock to elicit more DG spikes")
    ap.add_argument("--windows",  type=str, default="100,200,400",
                    help="comma list of window steps to sweep")
    ap.add_argument("--k-list",   type=str, default="10,20,40,80,150,300",
                    help="comma list of k values to sweep")
    ap.add_argument("--n-repro-pairs", type=int, default=8,
                    help="number of same-input pairs for reproducibility measurement")
    ap.add_argument("--n-trials",  type=int, default=80,
                    help="trials per cleanup test condition")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw",
                                         "_cortex_dg_ratekwta_cleanup_probe.json"))
    args = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for s in seeds:
        r = run_seed(s, args)
        if r is not None:
            results.append(r)

    if not results:
        print("No results (all seeds missing denoise64 cache).", flush=True)
        return 1

    # Multi-seed roll-up
    verdicts = [r["verdict"] for r in results]
    n_go = sum(v == "GO" for v in verdicts)
    n_partial = sum(v == "PARTIAL" for v in verdicts)
    n_neg = sum(v == "NEGATIVE" for v in verdicts)
    overall = "GO" if n_go == len(results) else ("PARTIAL" if n_partial + n_go >= 1 else "NEGATIVE")

    print("\n" + "#" * 72, flush=True)
    print("MULTI-SEED ROLL-UP (%d seeds)" % len(results), flush=True)
    for r in results:
        bs = r.get("best_setting") or {}
        print("  seed=%d  best_repro=%.3f (W=%d k=%d OU_off=%s sep=%.3f)  verdict=%s" %
              (r["seed"], r["best_repro"],
               bs.get("window", 0), bs.get("k", 0), bs.get("ou_off", "?"),
               bs.get("sep_mean", 999.0), r["verdict"]), flush=True)
    print("  => OVERALL %s  (GO=%d PARTIAL=%d NEGATIVE=%d)" %
          (overall, n_go, n_partial, n_neg), flush=True)

    out = {
        "probe": "cortex_dg_ratekwta_cleanup_probe",
        "seeds": [r["seed"] for r in results],
        "overall_verdict": overall,
        "per_seed": results,
        "rollup": {
            "n_go": n_go, "n_partial": n_partial, "n_negative": n_neg,
            "best_repro_per_seed": [r["best_repro"] for r in results],
        }
    }
    op = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
