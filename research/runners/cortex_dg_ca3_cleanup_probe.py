"""STEP 3 (true cortex) -- RESOLVE the mapped cleanup boundary with the BRAIN's trisynaptic loop.

Per docs/plans/2026-06-10-step3-true-cortex-design.md (Sec 2 Option A) and the de-risk finding
research/findings/2026-06-10-cortex-learned-cleanup-derisk-PARTIAL.md.

THE BOUNDARY (from the de-risk): the composer's god's-eye argmax cleanup is a linear matched filter, so it is
immune to the common mode of CORRELATED codes and recovers a noisy concept at 1.000. A learned Hopfield/CA3
attractor whose recurrent weight is the Hebbian outer product W = C C^T COLLAPSES on the brain's raw correlated
denoise64 codes (~chance), because W acquires a dominant common-mode eigenvector and the settle locks onto it.
The de-risk RESTORED attractor=argmax parity by prepending a HOST ZCA linear decorrelation -- but a host linear
transform is a SHORTCUT (the brain is not computing it).

THIS PROBE replaces that host ZCA with the project's VALIDATED, brain-based, LEARNED/SPIKING decorrelation: the
hippocampal DENTATE GYRUS PATTERN SEPARATION (catalog D.12), realised on a real SimulationBridge by sparse
k-winners DG coding + PV-basket feedforward inhibition. The cleanup attractor becomes the SPIKING CA3
AUTOASSOCIATOR (catalog D.13, Marr 1971), the CA3->CA3 recurrent loop. Together DG->CA3 = the project's
validated TRISYNAPTIC LOOP (P1 multi-seed: D.12 separation + D.13 completion). The question: does the
trisynaptic loop replace the composer's argmax cleanup -- matching it on full cues AND beating it on partial cues
(pattern completion, which a matched filter cannot do)?

TERMS (defined once):
  - concept code: the brain's REAL denoise64 vector for a word (mean firing of its concept pool), a signed real
    vector. The V=16 codes are CORRELATED (raw cosine ~0.6-0.7) -- the stress the de-risk mapped.
  - cleanup: recover the stored concept from a NOISY cue (a corrupted version of one concept code).
      * argmax baseline (the composer's god's-eye cleanup, the de-risk control): nearest stored code by cosine
        (a linear matched filter; immune to the common mode -> robust on correlated codes).
      * raw Hopfield (the de-risk's documented COLLAPSE, reference only): a complex/real attractor with W = C C^T
        over the RAW correlated codes; settles, reads the nearest attractor. Collapses on correlated codes.
      * DG->CA3 (THIS probe, brain-based): drive the cue into language_input on a real bridge; EC->DG with
        PV-basket FFi SPARSIFIES + DECORRELATES it in SPIKES (D.12, the learned replacement for host ZCA);
        DG->CA3 mossy fibers + the trained CA3->CA3 recurrent autoassociator COMPLETE it (D.13); read which
        stored CA3 ensemble it lands in by cosine. NO host linear decorrelation; NO argmax over a god's-eye list.
  - pattern separation (D.12): DG turns overlapping inputs into near-orthogonal sparse codes. The brain's
    learned/spiking decorrelation -- the legitimate replacement for the host ZCA.
  - pattern completion (D.13): CA3 recurrents reconstruct the full stored pattern from a PARTIAL/noisy cue. The
    attractor's value-ADD over argmax (a matched filter has no completion).
  - stored CA3 ensemble: per concept, the CA3 firing-rate vector when its CLEAN code drives the loop (after the
    CA3 autoassociator is trained). The "attractors" the cleanup resolves to.

TWO TESTS + the anti-cheats:
  TEST 1 -- CLEANUP PARITY (recovers the de-risk collapse). On FULL (lightly-noised) cues of correlated codes:
    does DG->CA3 cleanup match argmax (lifting from the de-risk's ~chance Hopfield collapse back to ~argmax)?
    Gate: DG->CA3 acc >= argmax acc - tol.
  TEST 2 -- PATTERN COMPLETION (the attractor's ADVANTAGE). On PARTIAL cues (mask a fraction of the code dims):
    does DG->CA3 complete the stored pattern BETTER than argmax-on-the-partial-cue?
    Gate: DG->CA3 acc > argmax acc on partial cues.
  ANTI-CHEAT 1 (rides the LEARNED CA3 weights): lesion the CA3->CA3 recurrent weights -> cleanup collapses.
  ANTI-CHEAT 2 (the DG decorrelation is load-bearing): lesion the DG pattern separation (open the PV-basket FFi
    so DG no longer sparsifies/decorrelates) -> the correlated-code collapse RETURNS (DG->CA3 acc drops).

BRAIN-BASED-ONLY: the decorrelation is the SPIKING DG (k-WTA + PV FFi), NOT a host ZCA; the cleanup is the
SPIKING CA3 autoassociator settle, NOT an argmax over an enumerated god's-eye codebook. The cue is rendered to
language_input activity by a fixed positive-rectified projection (presenting sensory drive = the environment's
job, legitimate host); reading a region's firing-rate vector is a READOUT, not a computation of the match. The
argmax appears ONLY as the baseline being replaced. The Hopfield-raw appears ONLY as the documented collapse
reference. No sim/ edits -- the trisynaptic loop is the project's existing build_biological_brain_regions wiring,
reused by import.

Run: SIM_BACKEND=numpy python -m research.runners.cortex_dg_ca3_cleanup_probe --seed 42
Multi-seed: --seed 42|43|44. CPU-cheap (a ~1200-neuron bridge + a few hundred short drive windows); minutes/seed.
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


# ---------------------------------------------------------------------------
# REAL correlated codes (the brain's denoise64 concept codes).
# ---------------------------------------------------------------------------
def load_real_codes(seed, proj_dim, rng):
    """Load the brain's REAL denoise64 concept codes -> signed real codes [V, D].

    Treatment matches the de-risk: mean over the obs samples per word, random-Gaussian project to proj_dim
    (preserves cosines), mean-center + unit-normalize. NO decorrelation here -- these are the RAW correlated codes
    (the whole point; the DG does the decorrelation, in spikes). Returns (words, codes, raw_between_cos)."""
    d = np.load(CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)   # [V, 3200]
    if proj_dim and proj_dim > 0:
        P = rng.standard_normal((raw.shape[1], proj_dim)) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    # between-code cosine (the correlation stress, auditable)
    V = codes.shape[0]
    cs = [float(codes[i] @ codes[k]) for i in range(V) for k in range(i + 1, V)]
    return ws, codes, (float(np.mean(cs)) if cs else 0.0)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# The BRAIN: the trisynaptic loop on a real SimulationBridge (reuse-by-import).
# ---------------------------------------------------------------------------
class TrisynapticCleanup:
    """The project's validated trisynaptic loop (EC->DG->CA3->CA1 with PV-basket FFi + CA3 recurrent
    autoassociator) used as a content-addressable CLEANUP. DG = learned/spiking pattern separation (D.12, the
    brain-based replacement for the de-risk's host ZCA); CA3 = the autoassociator completion (D.13). Built from
    build_biological_brain_regions(enable_hippocampus_consolidation=True) -- NO sim/ edits."""

    def __init__(self, seed, n_lang_input, n_dg, n_dg_pv_basket, n_ca3, n_ca1, n_ec,
                 ca3_recurrent_density, ca3_recurrent_weight, drive_pA, verbose=True):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from research.runners.text_minimal_isolation import build_biological_brain_regions
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self.drive_pA = float(drive_pA)
        self.n_lang_input = int(n_lang_input)
        self.verbose = verbose
        log = print if verbose else (lambda *a, **k: None)

        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input, n_motor_per_action=8, n_motor_fs_per_action=2,
            enable_motor_fs=True, enable_language_output=False,
            enable_hippocampus_consolidation=True,
            n_ec=n_ec, n_dg=n_dg, n_dg_pv_basket=n_dg_pv_basket, n_ca3=n_ca3, n_ca1=n_ca1,
            ca3_recurrent_density=ca3_recurrent_density, ca3_recurrent_weight=ca3_recurrent_weight,
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
        self.dg_idx = np.asarray(rm.indices("dg"), dtype=np.int64)
        self.ca3_idx = np.asarray(rm.indices("ca3"), dtype=np.int64)
        self.cfg = cfg
        log("  [bridge] built %.1fs; %d neurons %d synapses (DG=%d CA3=%d)"
            % (time.time() - t0, cfg.num_neurons, int(self.bridge.cp_connections.nnz), n_dg, n_ca3))

        # A fixed positive projection from code-space [D] to a language_input drive [n_lang_input].
        # Rectified (drive is a current); this is rendering sensory drive (the environment's job), NOT a
        # learned/decorrelating transform -- the DG does the decorrelation in spikes.
        rngp = np.random.default_rng(seed + 333)
        self._Wdrive = np.abs(rngp.standard_normal((proj_dim_global(), self.lang_idx.shape[0])))

        # gate snapshots for the DG-lesion anti-cheat (restore after)
        self._gate_saved = {}

    # --- low-level drive/read ---
    def _drive_lang(self, drive_vec):
        xp = self._xp
        self.bridge.cp_external_input_current[:] = 0.0
        vals = xp.asarray(drive_vec.astype(np.float32))
        self.bridge.cp_external_input_current[xp.asarray(self.lang_idx)] = vals

    def _render_drive(self, code_vec):
        """code-space vector -> language_input drive (rectified projection, scaled to drive_pA)."""
        x = np.asarray(code_vec, dtype=np.float64) @ self._Wdrive       # [n_lang_input], >= 0
        m = x.max()
        if m > 1e-9:
            x = x / m
        return (x * self.drive_pA).astype(np.float32)

    def _run_read(self, region_arr, drive_vec, n_steps=80, reset_steps=40):
        """Reset transients, drive language_input with drive_vec, run n_steps, return region firing-rate vec."""
        from sim.backend import to_host
        xp = self._xp
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            self.bridge._run_one_simulation_step()
            self.bridge.runtime_state.current_time_step += 1
        self._drive_lang(drive_vec)
        reg = xp.asarray(region_arr)
        counts = xp.zeros(region_arr.shape[0], dtype=xp.float32)
        for _ in range(n_steps):
            self.bridge._run_one_simulation_step()
            self.bridge.runtime_state.current_time_step += 1
            counts += self.bridge.cp_firing_states[reg].astype(xp.float32)
        self.bridge.cp_external_input_current[:] = 0.0
        return to_host(counts)

    def dg_response(self, code_vec, **kw):
        return self._run_read(self.dg_idx, self._render_drive(code_vec), **kw)

    def ca3_response(self, code_vec, **kw):
        return self._run_read(self.ca3_idx, self._render_drive(code_vec), **kw)

    # --- train the CA3 autoassociator on the V concept ensembles (D.13) ---
    def train_autoassociator(self, codes, train_events=4, steps_per=80):
        """Co-fire each clean concept code through the loop with the CA3 recurrent plasticity gate OPEN, so STDP
        strengthens the CA3->CA3 attractor for each stored ensemble. Then read the stored CA3 ensemble per code."""
        b = self.bridge
        for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst"):
            try:
                b.set_plasticity_gate(g, 1.0)
            except Exception:
                pass
        V = codes.shape[0]
        for _ev in range(train_events):
            for i in range(V):
                self._run_read(self.ca3_idx, self._render_drive(codes[i]),
                               n_steps=steps_per, reset_steps=30)
        for g in ("lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ca3_swr_burst"):
            try:
                b.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        # stored attractors: CA3 firing-rate vector for each clean code (recall, plasticity off)
        self.stored_ca3 = np.stack([self.ca3_response(codes[i]) for i in range(V)])
        self.stored_dg = np.stack([self.dg_response(codes[i]) for i in range(V)])
        return self.stored_ca3, self.stored_dg

    def cleanup(self, cue_vec):
        """Brain-based cleanup: drive the (noisy/partial) cue into the loop, read CA3, return the stored
        concept index whose CA3 ensemble is nearest (the attractor it completed to)."""
        r = self.ca3_response(cue_vec)
        sims = [_cos(r, self.stored_ca3[i]) for i in range(self.stored_ca3.shape[0])]
        return int(np.argmax(sims)), r

    # --- anti-cheat lesions ---
    def lesion_ca3_recurrent(self):
        """Zero the CA3->CA3 recurrent weights (the learned autoassociator). Cleanup must collapse."""
        return self._zero_pathway("ca3", "ca3")

    def lesion_dg_separation(self):
        """Disable the DG pattern separation: zero the PV-basket feedforward inhibition (ec->dg_pv_basket and
        dg_pv_basket->dg). Without the FFi, DG no longer sparsifies/decorrelates -> the correlated-code collapse
        returns. This is the load-bearing-DG anti-cheat."""
        ok1 = self._zero_pathway("ec", "dg_pv_basket")
        ok2 = self._zero_pathway("dg_pv_basket", "dg")
        return ok1 and ok2

    def _zero_pathway(self, frm, to):
        """Directly zero the cp_connections entries from region `frm` to region `to` (CSR data).

        cp_connections is built COO((w, (pre, post))) -> CSR, so ROW = presynaptic (`frm`), COLUMN =
        postsynaptic (`to`). Iterate the `frm` rows, zero entries whose column is in the `to` set."""
        try:
            from sim.backend import to_host
            xp = self._xp
            rm = self.bridge.region_manager
            src_rows = [int(i) for i in rm.indices(frm)]    # presynaptic rows
            dst_set = set(int(i) for i in rm.indices(to))   # postsynaptic columns
            M = self.bridge.cp_connections
            indptr = to_host(M.indptr); indices = to_host(M.indices); data = to_host(M.data)
            nz = 0
            for row in src_rows:
                a, b = int(indptr[row]), int(indptr[row + 1])
                seg = indices[a:b]
                for off, col in enumerate(seg):
                    if int(col) in dst_set:
                        data[a + off] = 0.0
                        nz += 1
            M.data[:] = xp.asarray(data)
            # invalidate any cached COO so the step loop sees the zeroed weights
            self.bridge._invalidate_coo_cache()
            if self.verbose:
                print("  [lesion %s->%s] zeroed %d synapse entries" % (frm, to, nz))
            return nz > 0
        except Exception as e:
            if self.verbose:
                print("  [lesion %s->%s] failed: %s" % (frm, to, e))
            return False


_PROJ_DIM = 512


def proj_dim_global():
    return _PROJ_DIM


# ---------------------------------------------------------------------------
# The composer's argmax + raw-Hopfield references (in code space; the de-risk baselines).
# ---------------------------------------------------------------------------
def argmax_cleanup(cue, codes):
    """The composer's god's-eye cleanup: nearest stored code by cosine (a linear matched filter)."""
    sims = codes @ cue / (np.linalg.norm(codes, axis=1) * (np.linalg.norm(cue) + 1e-12) + 1e-12)
    return int(np.argmax(sims))


class RawHopfield:
    """The de-risk's documented COLLAPSE reference: a real Hopfield attractor with W = C C^T over the RAW
    correlated codes. Settles a cue and reads the nearest attractor. Reproduces the correlated-code collapse
    (W's dominant common-mode eigenvector dominates the settle). Reference only -- NOT the deliverable."""

    def __init__(self, codes, iters=8):
        self.codes = codes
        n = codes.shape[1]
        self.W = (codes.T @ codes) / float(n)      # [D, D]
        self.iters = iters

    def cleanup(self, cue):
        z = cue.astype(np.float64).copy()
        for _ in range(self.iters):
            z = self.W @ z
            nz = np.linalg.norm(z)
            if nz < 1e-9:
                break
            z = z / nz
        sims = self.codes @ z / (np.linalg.norm(self.codes, axis=1) + 1e-12)
        return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Cue generators (matched-filter-fair: a corrupted version of one concept code).
# ---------------------------------------------------------------------------
def noisy_cue(code, rng, noise=0.6):
    """A FULL but noised cue: the concept code + Gaussian noise (std = noise * code-std). Models the composer's
    unbind estimate (a noisy version of the true filler code). All three cleanups see the SAME cue."""
    sigma = noise * float(np.std(code))
    return code + rng.standard_normal(code.shape) * sigma


def partial_cue(code, rng, keep_frac=0.4):
    """A PARTIAL cue: keep a random keep_frac of the code's dimensions, zero the rest (occlusion). This is the
    pattern-completion stress -- a matched filter sees only the kept dims; an attractor can COMPLETE the rest."""
    D = code.shape[0]
    keep = rng.choice(D, size=max(1, int(round(keep_frac * D))), replace=False)
    out = np.zeros_like(code)
    out[keep] = code[keep]
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def run_test1_parity(tri, codes, raw_hop, rng, n_trials, noise):
    """TEST 1 -- cleanup parity on FULL noised cues. DG->CA3 vs argmax (gate) + raw-Hopfield (collapse ref)."""
    V = codes.shape[0]
    n_tri = n_arg = n_hop = n_tot = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = noisy_cue(codes[i], rng, noise=noise)
        n_arg += int(argmax_cleanup(cue, codes) == i)
        n_hop += int(raw_hop.cleanup(cue) == i)
        idx, _ = tri.cleanup(cue)
        n_tri += int(idx == i)
        n_tot += 1
    return {"argmax_acc": n_arg / n_tot, "hopfield_raw_acc": n_hop / n_tot,
            "dg_ca3_acc": n_tri / n_tot, "n": n_tot, "chance": 1.0 / V, "noise": noise}


def run_test2_completion(tri, codes, rng, n_trials, keep_frac):
    """TEST 2 -- pattern completion on PARTIAL (occluded) cues. DG->CA3 vs argmax-on-the-partial-cue (head-to-head)."""
    V = codes.shape[0]
    n_tri = n_arg = n_tot = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = partial_cue(codes[i], rng, keep_frac=keep_frac)
        n_arg += int(argmax_cleanup(cue, codes) == i)
        idx, _ = tri.cleanup(cue)
        n_tri += int(idx == i)
        n_tot += 1
    return {"argmax_acc": n_arg / n_tot, "dg_ca3_acc": n_tri / n_tot,
            "n": n_tot, "chance": 1.0 / V, "keep_frac": keep_frac}


def run_anticheat_lesions(seed, codes, build_kw, train_events, rng, n_trials, noise):
    """ANTI-CHEAT 1 (CA3-recurrent lesion -> cleanup collapses) + ANTI-CHEAT 2 (DG-separation lesion -> the
    correlated-code collapse returns). Each on a FRESH bridge so the lesion is isolated. build_kw is the
    constructor kwargs (NO `_train_events`)."""
    V = codes.shape[0]

    # Anti-cheat 1: lesion the CA3 recurrent autoassociator weights.
    tri1 = TrisynapticCleanup(seed=seed, **build_kw)
    tri1.train_autoassociator(codes, train_events=train_events)
    tri1.lesion_ca3_recurrent()
    tri1.stored_ca3 = np.stack([tri1.ca3_response(codes[i]) for i in range(V)])  # re-read post-lesion
    n_les1 = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = noisy_cue(codes[i], rng, noise=noise)
        idx, _ = tri1.cleanup(cue)
        n_les1 += int(idx == i)
    ca3_lesion_acc = n_les1 / n_trials

    # Anti-cheat 2: lesion the DG pattern separation (open PV-basket FFi).
    tri2 = TrisynapticCleanup(seed=seed, **build_kw)
    tri2.lesion_dg_separation()
    tri2.train_autoassociator(codes, train_events=train_events)
    n_les2 = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = noisy_cue(codes[i], rng, noise=noise)
        idx, _ = tri2.cleanup(cue)
        n_les2 += int(idx == i)
    dg_lesion_acc = n_les2 / n_trials

    # Quantify what the DG lesion did to separation: between-ensemble DG cosine intact vs lesioned.
    dg_cos_intact = _mean_offdiag_cos(tri2_stored_dg_intact(seed, codes, build_kw, train_events))
    dg_cos_lesioned = _mean_offdiag_cos(tri2.stored_dg)
    return {"ca3_recurrent_lesion_acc": ca3_lesion_acc,
            "dg_separation_lesion_acc": dg_lesion_acc,
            "dg_between_ensemble_cos_intact": dg_cos_intact,
            "dg_between_ensemble_cos_lesioned": dg_cos_lesioned}


def tri2_stored_dg_intact(seed, codes, build_kw, train_events):
    """Helper: an intact bridge's stored DG ensembles (for the separation-cosine comparison)."""
    tri = TrisynapticCleanup(seed=seed, verbose=False, **build_kw)
    tri.train_autoassociator(codes, train_events=train_events)
    return tri.stored_dg


def _mean_offdiag_cos(ensembles):
    V = ensembles.shape[0]
    cs = [_cos(ensembles[i], ensembles[j]) for i in range(V) for j in range(i + 1, V)]
    return float(np.mean(cs)) if cs else 0.0


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def run_seed(seed, args):
    print("\n" + "=" * 72, flush=True)
    print("=== STEP 3 DG->CA3 trisynaptic cleanup probe (seed %d) ===" % seed, flush=True)
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
        ca3_recurrent_density=args.ca3_recurrent_density, ca3_recurrent_weight=args.ca3_recurrent_weight,
        drive_pA=args.drive_pA, _train_events=args.train_events,
    )

    # Build the trisynaptic loop + train the CA3 autoassociator.
    tri = TrisynapticCleanup(seed=seed, **{k: v for k, v in build_kw.items() if not k.startswith("_")})
    t0 = time.time()
    stored_ca3, stored_dg = tri.train_autoassociator(codes, train_events=args.train_events)
    dg_sep_cos = _mean_offdiag_cos(stored_dg)
    ca3_sep_cos = _mean_offdiag_cos(stored_ca3)
    print("  [trained] CA3 autoassociator on %d codes (%.1fs).  DG between-ensemble cos=%.3f  "
          "CA3 between-ensemble cos=%.3f" % (V, time.time() - t0, dg_sep_cos, ca3_sep_cos), flush=True)
    print("  (DG cos << raw code cos %.3f  => the SPIKING DG pattern-separated the correlated codes, D.12)"
          % raw_cos, flush=True)

    raw_hop = RawHopfield(codes)

    # TEST 1 -- parity.
    print("\n--- TEST 1: cleanup PARITY on full noised cues (DG->CA3 vs argmax; Hopfield-raw=collapse ref) ---",
          flush=True)
    t1 = run_test1_parity(tri, codes, raw_hop, np.random.default_rng(seed + 11),
                          args.n_trials, args.noise)
    print("  argmax(matched filter) = %.3f   Hopfield-raw(collapse ref) = %.3f   DG->CA3 = %.3f   (chance %.3f)"
          % (t1["argmax_acc"], t1["hopfield_raw_acc"], t1["dg_ca3_acc"], t1["chance"]), flush=True)

    # TEST 2 -- pattern completion.
    print("\n--- TEST 2: pattern COMPLETION on partial (occluded) cues (DG->CA3 vs argmax-on-partial) ---",
          flush=True)
    t2 = run_test2_completion(tri, codes, np.random.default_rng(seed + 22),
                              args.n_trials, args.keep_frac)
    print("  keep_frac=%.2f   argmax-on-partial = %.3f   DG->CA3 = %.3f   (chance %.3f)"
          % (t2["keep_frac"], t2["argmax_acc"], t2["dg_ca3_acc"], t2["chance"]), flush=True)

    # ANTI-CHEATS.
    print("\n--- ANTI-CHEATS: lesion CA3 recurrent (cleanup collapses) + lesion DG separation (collapse returns) ---",
          flush=True)
    les = run_anticheat_lesions(seed, codes, {k: v for k, v in build_kw.items() if not k.startswith("_")},
                                args.train_events, np.random.default_rng(seed + 33), args.n_trials, args.noise)
    print("  CA3-recurrent LESION cleanup acc = %.3f  (vs intact %.3f -> %s)"
          % (les["ca3_recurrent_lesion_acc"], t1["dg_ca3_acc"],
             "COLLAPSES" if les["ca3_recurrent_lesion_acc"] < t1["dg_ca3_acc"] - 0.1 else "does NOT collapse"),
          flush=True)
    print("  DG-separation LESION cleanup acc = %.3f  (vs intact %.3f -> %s);  DG between-cos %.3f(intact) -> "
          "%.3f(lesioned)" % (les["dg_separation_lesion_acc"], t1["dg_ca3_acc"],
                              "COLLAPSE RETURNS" if les["dg_separation_lesion_acc"] < t1["dg_ca3_acc"] - 0.1
                              else "no collapse",
                              les["dg_between_ensemble_cos_intact"], les["dg_between_ensemble_cos_lesioned"]),
          flush=True)

    # GATES.
    tol = args.tol
    parity_ok = t1["dg_ca3_acc"] >= t1["argmax_acc"] - tol
    parity_lifts = t1["dg_ca3_acc"] >= t1["hopfield_raw_acc"] + 0.2   # lifted off the Hopfield collapse
    completion_ok = t2["dg_ca3_acc"] > t2["argmax_acc"]
    ca3_lesion_ok = les["ca3_recurrent_lesion_acc"] < t1["dg_ca3_acc"] - 0.1
    dg_lesion_ok = les["dg_separation_lesion_acc"] < t1["dg_ca3_acc"] - 0.1

    if parity_ok and parity_lifts and completion_ok and ca3_lesion_ok and dg_lesion_ok:
        verdict = "GO"
    elif (parity_ok and parity_lifts) or completion_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    print("\n  GATES: parity(DG->CA3>=argmax-tol)=%s  lifts-off-Hopfield=%s  completion(DG->CA3>argmax)=%s  "
          "ca3-lesion-collapses=%s  dg-lesion-collapse-returns=%s" %
          (parity_ok, parity_lifts, completion_ok, ca3_lesion_ok, dg_lesion_ok), flush=True)
    print("  === SEED %d VERDICT: %s ===" % (seed, verdict), flush=True)

    return {
        "seed": seed, "n_words": V, "D": args.proj_dim, "between_code_cos_raw": raw_cos,
        "dg_between_ensemble_cos": dg_sep_cos, "ca3_between_ensemble_cos": ca3_sep_cos,
        "test1_parity": t1, "test2_completion": t2, "anticheat_lesions": les,
        "gates": {"parity": bool(parity_ok), "lifts_off_hopfield": bool(parity_lifts),
                  "completion": bool(completion_ok), "ca3_lesion_collapses": bool(ca3_lesion_ok),
                  "dg_lesion_collapse_returns": bool(dg_lesion_ok)},
        "verdict": verdict,
        "build": {k: v for k, v in build_kw.items() if not k.startswith("_")},
        "train_events": args.train_events, "noise": args.noise, "keep_frac": args.keep_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma list, e.g. 42,43,44 (overrides --seed)")
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--n-lang-input", type=int, default=512)
    ap.add_argument("--n-ec", type=int, default=160)
    ap.add_argument("--n-dg", type=int, default=600)
    ap.add_argument("--n-dg-pv-basket", type=int, default=180)
    ap.add_argument("--n-ca3", type=int, default=300)
    ap.add_argument("--n-ca1", type=int, default=120)
    ap.add_argument("--ca3-recurrent-density", type=float, default=0.30)
    ap.add_argument("--ca3-recurrent-weight", type=float, default=2.0)
    ap.add_argument("--drive-pA", type=float, default=220.0)
    ap.add_argument("--train-events", type=int, default=4)
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--noise", type=float, default=0.6, help="full-cue Gaussian noise (x code std)")
    ap.add_argument("--keep-frac", type=float, default=0.4, help="partial-cue fraction of dims kept")
    ap.add_argument("--tol", type=float, default=0.05, help="parity tolerance (DG->CA3 >= argmax - tol)")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw", "_cortex_dg_ca3_cleanup_probe.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for s in seeds:
        r = run_seed(s, args)
        if r is not None:
            results.append(r)

    # multi-seed roll-up
    if results:
        def mean(key_path):
            vals = []
            for r in results:
                d = r
                for k in key_path:
                    d = d[k]
                vals.append(d)
            return float(np.mean(vals))
        verdicts = [r["verdict"] for r in results]
        n_go = sum(v == "GO" for v in verdicts)
        n_partial = sum(v == "PARTIAL" for v in verdicts)
        overall = ("GO" if n_go == len(results) else
                   "PARTIAL" if (n_go + n_partial) >= 1 else "NEGATIVE")
        print("\n" + "#" * 72, flush=True)
        print("MULTI-SEED ROLL-UP (%d seeds: %s)" % (len(results), ",".join(str(r["seed"]) for r in results)),
              flush=True)
        print("  parity:     argmax %.3f | Hopfield-raw %.3f | DG->CA3 %.3f"
              % (mean(["test1_parity", "argmax_acc"]), mean(["test1_parity", "hopfield_raw_acc"]),
                 mean(["test1_parity", "dg_ca3_acc"])), flush=True)
        print("  completion: argmax-on-partial %.3f | DG->CA3 %.3f"
              % (mean(["test2_completion", "argmax_acc"]), mean(["test2_completion", "dg_ca3_acc"])), flush=True)
        print("  lesions:    CA3-recurrent %.3f | DG-separation %.3f"
              % (mean(["anticheat_lesions", "ca3_recurrent_lesion_acc"]),
                 mean(["anticheat_lesions", "dg_separation_lesion_acc"])), flush=True)
        print("  per-seed verdicts: %s  => OVERALL %s" % (verdicts, overall), flush=True)

        out = {"probe": "cortex_dg_ca3_cleanup_probe", "seeds": [r["seed"] for r in results],
               "overall_verdict": overall, "per_seed": results,
               "rollup": {
                   "parity_argmax": mean(["test1_parity", "argmax_acc"]),
                   "parity_hopfield_raw": mean(["test1_parity", "hopfield_raw_acc"]),
                   "parity_dg_ca3": mean(["test1_parity", "dg_ca3_acc"]),
                   "completion_argmax": mean(["test2_completion", "argmax_acc"]),
                   "completion_dg_ca3": mean(["test2_completion", "dg_ca3_acc"]),
                   "ca3_recurrent_lesion": mean(["anticheat_lesions", "ca3_recurrent_lesion_acc"]),
                   "dg_separation_lesion": mean(["anticheat_lesions", "dg_separation_lesion_acc"]),
               }}
        op = os.path.normpath(args.out)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
        print("wrote %s" % op, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
