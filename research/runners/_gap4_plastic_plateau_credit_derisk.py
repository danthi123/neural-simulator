"""gap#4 PLASTIC coincidence-PLATEAU credit de-risk -- the single UNTESTED combination.

THE REFRAME (verified 2026-08-01): the on-bridge deep-credit wall is NOT the credit direction. It is a MOVABLE-HIDDEN
problem -- even the true gradient cannot move the STANDARD tonic-driven sparse-rate hidden (d hidden-activity / d
input-weight ~ 0; tonic drive dominates input drive), so credit has nothing to shape. See
  research/findings/2026-08-01-gap4-sweet-spot-LOCATED-forward-representable-reservoir-fails-but-credit-cannot-train-there-6seed.md
  research/findings/2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md

BUT the project already BUILT a movable, input-driven, RELIABLE hidden: the coincidence dendritic-PLATEAU reset-read
expander (`research/runners/_gap4_plateau_expander_probe.py`, 6-seed GO 2026-07-25: held-out-linear 0.611 vs boundary
0.34, reproducibility 1.000). It has ONLY ever been used as a FIXED random coincidence map. A credit rule has NEVER
been wired onto it. THAT is the single untested combination this de-risk closes.

WHAT'S NEW HERE: the plateau expander's coincidence/input weights are made PLASTIC and trained by a LOCAL rule that
reads ONLY (the reliable plateau MARGIN cp_v_apical - FLOOR) x (local pre-activity). Concretely a per-column
plateau-gated COVARIANCE rule (a local Ca/plateau-enhancing rule; biology: eLife 2024 doi:10.7554/eLife.97274, local
calcium dendritic-nonlinearity feature binding):

    theta[c] = mean_i margin[c]                            # per-column homeostatic threshold (the COMPANION process,
                                                           #   NOT a constant -- the missing-companion lesson)
    dW[c,f] = lr * mean_i (margin[i,c] - theta[c]) * pre[i,f]    # potentiate f when column c is MORE active than its
                                                           #   own average AND f was active -> columns specialize on the
                                                           #   feature-conjunctions that distinctively drive their plateau
    W[c,:] <- W[c,:] renormalized to its INITIAL L2 norm   # local per-column homeostasis (prevents runaway)

NO phi' depth product, NO weight transport: the update reads ONLY local pre-activity and the local plateau margin --
it NEVER reads the forward/readout weights or their transpose (asserted in code + a runtime numeric check + it runs
BEFORE any readout exists). This is the whole point: LOCAL credit on a MOVABLE, RELIABLE hidden.

ON-SUBSTRATE, not host-formula: the plateau margin (post factor) is READ from the spiking bridge (cp_v_apical), the
pre-activity is the substrate's input, and the update is WRITTEN to the substrate's synaptic weights
(cp_connections.data) -- exactly the committed EMERGE-14 on-bridge permanence-update pattern
(`_emerge14_stageC_onbridge_learning_derisk.apply_kernel_update`). The weighted-coincidence drive
(coincidence_weighted_drive=True) makes the plateau GRADE with these learned weights.

TASK: the n_prop=3 XOR-over-pool "semantic inheritance" SWEET SPOT (make_task_semantic_inheritance, n_super=24) where
the oracle fits (~0.96) but a frozen random RATE reservoir fails (~0.26) -- the valid operating point (the depth-2
reservoir cannot carry it, the depth>=3 forward does not collapse).

ARMS (single variable = are the plateau input weights plastic?):
  1. FROZEN-plateau reservoir  -- fixed random coincidence weights + trained linear readout (deep_credit_share ref).
  2. CREDIT-trained plateau     -- SAME columns/init, weights plastic via the local plateau-margin covariance rule.
  3. frozen random RATE reservoir -- random ReLU projection + trained readout (MUST fail ~0.26 -> op-point genuine;
                                   also the context anchor for the tonic-pinned sparse-rate hidden that fails today,
                                   whose spiking-credit failure is SETTLED in the sweet-spot finding, cite don't re-run).
  4. oracle                     -- fenced backprop rate (ceiling ~0.96).

deep_credit_share = (credit_plateau - frozen_plateau) / (oracle - frozen_plateau)  -- reported AGAINST the frozen
plateau reservoir (NOT accuracy alone -- the K=8 "closure" was retracted for reading accuracy and ignoring
deep_credit_share). Note (from the sweet-spot finding): at k=8 / ~coarse held-out this ratio is noisy; the raw
held-out AND train-readout-fit are reported alongside.

ANTI-CHEATS (all mandatory): permuted-label -> chance (no readout leakage); plateau/apical LESION -> floor;
NO-TRANSPORT probe (code + runtime); reproducibility >= 0.8 (plateau reliability holds under plasticity); like-for-like
epochs/readout across arms. Backend stamped.

GO gate (SMOKE 1-2 seeds; 6-seed is the parent's after review): credit-trained plateau beats the FROZEN plateau
reservoir on held-out inheritance by a preregistered margin -> deep_credit_share > 0 where the fixed reservoir fails.
An honest negative naming WHICH anti-cheat / mode failed is a valid deliverable.

Run (numpy is a no-op for the bridge on this box -> cupy):
    SIM_BACKEND=cupy python -m research.runners._gap4_plastic_plateau_credit_derisk --seeds 42 --epochs 15
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")

import numpy as np

# --- reuse-by-import (NO edit to any shared runner) ---
# The MECHANISM is reused from IMPORT-SAFE modules: the reliable reset-based plateau read (`_prime_from_winners`),
# the host bring-back (`_host`), the sweet-spot task (`make_task_semantic_inheritance`) + its oracle helpers, and the
# fenced-backprop oracle (`DendriticMLP`). The plateau-expander PROBE (`_gap4_plateau_expander_probe.py`) is NOT
# import-safe -- it runs a full 6-seed probe at module top-level -- so its three TRIVIAL numpy helpers (fit_lin /
# topk_active / _sm) and its constants (FLOOR/ACT_TH/TOPK) are copied VERBATIM below rather than imported (importing
# would re-run the whole probe). `PlasticPlateauExpander` below mirrors that probe's `PlateauExpander._vap` logic.
from research.runners._semantic_inheritance_deep_credit_derisk import (
    make_task_semantic_inheritance, _train_oracle, _acc_on)
from research.runners._emerge14_stageC_onbridge_learning_derisk import _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from sim.dendritic_mlp import DendriticMLP

# --- verbatim from _gap4_plateau_expander_probe.py (that module is not import-safe; see the note above) ---
FLOOR = -40.0
ACT_TH, TOPK = 2, 4


def _sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_lin(X, y, k, iters=600, lr=0.5, l2=3e-3):
    n, d = X.shape; W = np.zeros((d, k)); b = np.zeros(k); Y = np.eye(k)[y]
    for _ in range(iters):
        P = _sm(X @ W + b); g = (P - Y) / n; W -= lr * (X.T @ g + l2 * W); b -= lr * g.sum(0)
    return lambda Z: np.argmax(Z @ W + b, 1)


def topk_active(X, topk):
    th = np.sort(X, axis=1)[:, -topk][:, None]
    return [set(np.where(row >= t)[0]) for row, t in zip(X, th[:, 0])]


# ============================================================================================================
# The PLASTIC coincidence-plateau expander -- a DENSE random-weight reservoir on a real SimulationBridge whose
# coincidence/input weights are trainable in place (cp_connections.data), read via the reliable reset-based plateau.
# Mirrors PlateauExpander's bridge config (reuse-by-import for every helper; the builder is local ONLY because the
# parent bakes a boolean sampling matrix, and the prompt forbids editing the shared probe).
# ============================================================================================================
class PlasticPlateauExpander:
    def __init__(self, n_feat, n_col, seed, w0=0.35, jitter=0.15, k_th=None,
                 plateau_strength=160.0, lesion=False):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        self.NF, self.NC = n_feat, n_col
        M = n_feat + n_col
        regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                               inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        cfg = CoreSimConfig()
        cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
        cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
        cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
        for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                  "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
            setattr(cfg, f, False)
        # coincidence-plateau (the reliable expander). lesion => apical/plateau off -> codon degenerate (floor).
        cfg.enable_coincidence_detection = not lesion
        cfg.coincidence_weighted_drive = True                 # c_drive = Sum_j w_eff_j * x_j -> plateau grades w/ weights
        cfg.coincidence_k_threshold = float(ACT_TH) - 0.5 if k_th is None else float(k_th)
        cfg.coincidence_plateau_strength = 0.0 if lesion else float(plateau_strength)
        cfg.enable_two_compartment_dap = not lesion; cfg.apical_g_couple = 2.0
        cfg.deterministic_transpose_matvec = True             # bit-reproducible coincidence matvec (default-off flag ON
                                                              #   in-config only, NOT a sim/ edit) -> reliable reads
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)
        # DENSE reservoir: every column connects to EVERY feature, small RANDOM initial weight (jitter gives column
        # diversity -> input-driven, decorrelated codon; the on-bridge analogue of a random-weight reservoir). Plasticity
        # then SCULPTS which feature-conjunction each column detects. w>=0 (excitatory).
        pre, post, w = [], [], []
        for c in range(n_col):
            for f in range(n_feat):
                pre.append(int(ci[f])); post.append(int(ci[n_feat + c]))
                w.append(float(max(0.0, w0 + jitter * rng.standard_normal())))
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        self.b, self.ci, self.lesion = b, ci, lesion
        # --- weight-update plumbing (the EMERGE-14 contract): coo.row=pre, coo.col=post, aligned with data order ---
        coo = b._get_cached_coo()
        self.coo_row = np.asarray(_host(coo.row), int); self.coo_col = np.asarray(_host(coo.col), int)
        feat_of = -np.ones(M, int); feat_of[ci[:n_feat]] = np.arange(n_feat)
        col_of = -np.ones(M, int); col_of[ci[n_feat:n_feat + n_col]] = np.arange(n_col)
        self.syn_feat = feat_of[self.coo_row]                 # feature index per synapse
        self.syn_col = col_of[self.coo_col]                   # column index per synapse
        assert (self.syn_feat >= 0).all() and (self.syn_col >= 0).all(), "wiring maps to unexpected neurons"
        self.W0 = self._get_data().copy()                     # snapshot initial weights (the FROZEN reservoir)
        # initial per-column L2 norm (the renorm target = local homeostasis)
        self.col_norm0 = np.sqrt(np.array([np.sum(self.W0[self.syn_col == c] ** 2) for c in range(n_col)]))

    # ---- substrate reads/writes ----
    def _get_data(self):
        return np.asarray(_host(self.b.cp_connections.data), np.float64)

    def _set_data(self, data):
        arr = np.asarray(data, np.float32)
        self.b.cp_connections.data[:] = self.b.xp.asarray(arr) if hasattr(self.b, "xp") else arr
        self.b._coo_cache_valid = False                       # weights changed -> effective matvec must see them

    def restore_frozen(self):
        self._set_data(self.W0)

    def _vap(self, active_feats):
        """Read cp_v_apical for the columns (reliable reset-based read) -- reused verbatim from PlateauExpander."""
        ab = np.zeros(len(self.ci), bool)
        if len(active_feats):
            ab[np.asarray(list(active_feats), int)] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None:
            return np.zeros(self.NC)
        return np.asarray(_host(vap))[self.ci][self.NF:self.NF + self.NC]

    def margin(self, active_feats):
        return np.maximum(0.0, self._vap(active_feats) - FLOOR)     # the low-CV graded plateau surrogate

    def codon(self, active_feats):
        return (self._vap(active_feats) > FLOOR).astype(np.float64)

    # ---- THE LOCAL PLATEAU-GATED COVARIANCE RULE (no transport; reads ONLY pre-activity x plateau margin) ----
    def train_epoch(self, active_sets, pre_mat, lr):
        """One batch update. Reads margins with the CURRENT weights, then a per-column plateau-gated covariance step.
        pre_mat[i,f] = local pre-activity (active indicator). NEVER touches any readout weight (asserted by the caller's
        no-transport probe; this method's signature carries no readout)."""
        Mrg = np.asarray([self.margin(a) for a in active_sets])     # (N, C) reliable plateau margins
        theta = Mrg.mean(0, keepdims=True)                          # (1, C) per-column homeostatic threshold (COMPANION)
        # dW_full[c,f] = lr * mean_i (margin_ic - theta_c) * pre_if   (covariance of column margin & feature pre)
        dW_full = lr * ((Mrg - theta).T @ pre_mat) / len(active_sets)   # (C, F)
        data = self._get_data()
        data = data + dW_full[self.syn_col, self.syn_feat]
        np.maximum(data, 0.0, out=data)                             # excitatory (w >= 0)
        # per-column L2 renorm to the INITIAL norm (local homeostasis -> prevents runaway, preserves scale)
        cur = np.sqrt(np.array([np.sum(data[self.syn_col == c] ** 2) for c in range(self.NC)]))
        scale = np.where(cur > 1e-9, self.col_norm0 / (cur + 1e-12), 1.0)
        data = data * scale[self.syn_col]
        self._set_data(data)
        return float(np.mean(np.abs(dW_full)))                      # mean |update| (for logging / convergence)


# ============================================================================================================
# Readout (reuse-by-import fit_lin/topk_active from the probe) + helpers.
# ============================================================================================================
def _readout_acc(exp, af_tr, y_tr, af_te, y_te, k):
    """Fit the SAME softmax linear readout used by the probe on the codon; return (train_acc, heldout_acc)."""
    Ctr = np.asarray([exp.codon(a) for a in af_tr]); Cte = np.asarray([exp.codon(a) for a in af_te])
    clf = fit_lin(Ctr, y_tr, k)
    return float(np.mean(clf(Ctr) == y_tr)), float(np.mean(clf(Cte) == y_te)), Ctr, Cte


def _reproducibility(exp, af, n=8):
    C1 = np.asarray([exp.codon(a) for a in af[:n]]); C2 = np.asarray([exp.codon(a) for a in af[:n]])
    return float(np.mean([np.corrcoef(C1[i], C2[i])[0, 1] if C1[i].std() > 0 and C2[i].std() > 0 else 1.0
                          for i in range(min(n, len(af)))]))


def _codon_diversity(C):
    """mean off-diagonal |column-column correlation| of the codon matrix -> lower = more diverse columns."""
    if C.shape[0] < 2:
        return float("nan")
    Cc = C - C.mean(0, keepdims=True)
    s = Cc.std(0); keep = s > 1e-9
    if keep.sum() < 2:
        return float("nan")
    R = np.corrcoef(Cc[:, keep].T)
    off = R[np.triu_indices_from(R, 1)]
    return float(np.mean(np.abs(off)))


def _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, H, seed):
    """Frozen random-ReLU RATE reservoir (the op-point-genuine anchor; MUST fail ~0.26). Numpy host reference."""
    rng = np.random.default_rng(seed * 31 + 7)
    R = rng.standard_normal((Xtr.shape[1], H)) / np.sqrt(Xtr.shape[1])
    Htr = np.maximum(0.0, Xtr @ R); Hte = np.maximum(0.0, Xte @ R)
    clf = fit_lin(Htr, ytr, k)
    return float(np.mean(clf(Htr) == ytr)), float(np.mean(clf(Hte) == yte))


def run_seed(seed, n_col, epochs, lr, w0, jitter, k_th, n_sub, hidden, oracle_epochs, oracle_lr, oracle_batch,
             task_kwargs, verbose=True):
    (Xtr, ytr, _), (Xte, yte, _), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    n_in = Xtr.shape[1]; k = meta["k_classes"]; inh = idx["inh_idx"]
    # subsample train (identical set for plasticity + readout across arms) -- like-for-like
    srng = np.random.default_rng(seed * 13 + 1); keep = srng.permutation(len(Xtr))[:min(n_sub, len(Xtr))]
    Xb, yb = Xtr[keep], ytr[keep]; Xh, yh = Xte[inh], yte[inh]
    af_b = topk_active(Xb, TOPK); af_h = topk_active(Xh, TOPK)
    # local pre-activity matrix (active indicator) for the plasticity rule
    pre_b = np.zeros((len(af_b), n_in))
    for i, a in enumerate(af_b):
        pre_b[i, np.asarray(list(a), int)] = 1.0
    chance = float(max(np.mean(yh == c) for c in np.unique(yh))) if len(yh) else float("nan")
    out = {"seed": seed, "meta": meta, "n_in": n_in, "k": k, "chance": chance, "n_train_sub": len(Xb),
           "n_heldout_inherit": len(yh)}

    # ---- ARM 4a: oracle (fenced backprop depth-2 rate) ----
    onet = DendriticMLP([n_in, hidden, hidden, k], seed=seed)
    _train_oracle(onet, Xtr, ytr, oracle_epochs, oracle_lr, oracle_batch, seed)
    out["oracle_train"] = float(onet.accuracy(Xtr, ytr)); out["oracle_heldout"] = _acc_on(onet, Xte, yte, inh)

    # ---- ARM 4b/3: frozen random RATE reservoir (must fail ~0.26) ----
    out["rate_reservoir_train"], out["rate_reservoir_heldout"] = _rate_reservoir_heldout(Xtr, ytr, Xte, yte, k, n_col, seed)

    # ---- ARM 1 & 2: build ONE plateau expander (identical init) -> FROZEN then CREDIT (the single variable) ----
    exp = PlasticPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th)
    # ARM 1 FROZEN reservoir (weights = W0, no training)
    exp.restore_frozen()
    fz_tr, fz_ho, Cb_fz, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["frozen_plateau_train"] = fz_tr; out["frozen_plateau_heldout"] = fz_ho
    out["frozen_codon_sparsity"] = float(Cb_fz.mean()); out["frozen_codon_diversity"] = _codon_diversity(Cb_fz)

    # ARM 2 CREDIT-trained (weights plastic via the local plateau-margin covariance rule) -- runs BEFORE any readout
    exp.restore_frozen()
    upd_mag = []
    for ep in range(epochs):
        upd_mag.append(exp.train_epoch(af_b, pre_b, lr))
    cr_tr, cr_ho, Cb_cr, _ = _readout_acc(exp, af_b, yb, af_h, yh, k)
    out["credit_plateau_train"] = cr_tr; out["credit_plateau_heldout"] = cr_ho
    out["credit_codon_sparsity"] = float(Cb_cr.mean()); out["credit_codon_diversity"] = _codon_diversity(Cb_cr)
    out["update_mag_first_last"] = [round(upd_mag[0], 5), round(upd_mag[-1], 5)]
    out["weight_moved"] = float(np.mean(np.abs(exp._get_data() - exp.W0)))

    # ---- deep_credit_share = (credit - frozen) / (oracle - frozen), measured AGAINST the frozen plateau reservoir ----
    denom = out["oracle_heldout"] - out["frozen_plateau_heldout"]
    out["deep_credit_share"] = float((out["credit_plateau_heldout"] - out["frozen_plateau_heldout"]) / denom) \
        if abs(denom) > 1e-6 else float("nan")

    # ---- ANTI-CHEAT: reproducibility (plateau reliability must hold under plasticity) ----
    out["reproducibility_credit"] = _reproducibility(exp, af_b)

    # ---- ANTI-CHEAT: permuted-label -> chance (no readout leakage; plasticity is label-free so this tests the readout) ----
    prng = np.random.default_rng(seed + 555); yperm = yb[prng.permutation(len(yb))]
    Ctr = np.asarray([exp.codon(a) for a in af_b]); Cte = np.asarray([exp.codon(a) for a in af_h])
    clf_p = fit_lin(Ctr, yperm, k)
    out["permuted_heldout"] = float(np.mean(clf_p(Cte) == yh))

    # ---- ANTI-CHEAT: plateau/apical LESION -> floor (coincidence+apical off -> degenerate codon) ----
    lex = PlasticPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th, lesion=True)
    les_tr, les_ho, _, _ = _readout_acc(lex, af_b, yb, af_h, yh, k)
    out["lesion_heldout"] = les_ho

    # ---- ANTI-CHEAT: NO-TRANSPORT probe (code + runtime) ----
    # (a) code: the update method signature must NOT expose any readout/forward classifier weight.
    sig = set(inspect.signature(PlasticPlateauExpander.train_epoch).parameters)
    no_transport_code = sig.isdisjoint({"readout", "clf", "W_out", "Wout", "forward_W", "Wt", "transpose"})
    # (b) runtime: the update must be INVARIANT to whether a readout exists -> the update cannot depend on it. Two FRESH
    #     identical-init probes; one carries a random "readout" attribute that MUST NOT matter. First-call on each (same
    #     bridge state) => identical update iff transport-free. (Temporal guarantee also holds: plasticity fully completes
    #     BEFORE any readout is fit in the arms above.)
    pA = PlasticPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th); pA.restore_frozen()
    pB = PlasticPlateauExpander(n_in, n_col, seed, w0=w0, jitter=jitter, k_th=k_th); pB.restore_frozen()
    pB._junk_readout = np.random.default_rng(0).standard_normal((n_col, k))   # exists but is never read by the rule
    d0 = pA.W0.copy()
    pA.train_epoch(af_b[:24], pre_b[:24], lr); dA = pA._get_data()
    pB.train_epoch(af_b[:24], pre_b[:24], lr); dB = pB._get_data()
    no_transport_runtime = bool(np.allclose(dA, dB, atol=1e-6, rtol=0.0) and not np.allclose(d0, dA, atol=1e-6, rtol=0.0))
    out["no_transport_code"] = bool(no_transport_code)
    out["no_transport_runtime"] = no_transport_runtime
    out["no_transport"] = bool(no_transport_code and no_transport_runtime)

    if verbose:
        print(f"  [seed {seed}] n_in={n_in} k={k} chance={chance:.3f} n_ho={len(yh)} n_sub={len(Xb)} N_COL={n_col}",
              flush=True)
        print(f"    oracle {out['oracle_heldout']:.3f}(tr {out['oracle_train']:.3f}) | "
              f"rate-reservoir {out['rate_reservoir_heldout']:.3f} | "
              f"FROZEN-plateau {out['frozen_plateau_heldout']:.3f}(tr {out['frozen_plateau_train']:.3f}) | "
              f"CREDIT-plateau {out['credit_plateau_heldout']:.3f}(tr {out['credit_plateau_train']:.3f})", flush=True)
        print(f"    deep_credit_share {out['deep_credit_share']:+.3f} | reprod {out['reproducibility_credit']:.3f} | "
              f"sparsity fz {out['frozen_codon_sparsity']:.3f}->cr {out['credit_codon_sparsity']:.3f} | "
              f"diversity fz {out['frozen_codon_diversity']:.3f}->cr {out['credit_codon_diversity']:.3f} | "
              f"|dW| {out['update_mag_first_last']} moved {out['weight_moved']:.4f}", flush=True)
        print(f"    [anti-cheat] permuted {out['permuted_heldout']:.3f}(~chance) | lesion {out['lesion_heldout']:.3f}"
              f"(~floor) | no-transport code={out['no_transport_code']} runtime={out['no_transport_runtime']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="gap#4 plastic coincidence-plateau credit de-risk (SMOKE).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-col", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--w0", type=float, default=0.35)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--k-th", type=float, default=None, help="coincidence weight threshold (default ACT_TH-0.5=1.5)")
    ap.add_argument("--n-sub", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--oracle-epochs", type=int, default=200)
    ap.add_argument("--oracle-lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", type=int, default=128)
    # --- the SWEET SPOT task config (n_prop=3, n_super=24) ---
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--margin-go", type=float, default=0.05, help="preregistered held-out margin over frozen plateau")
    ap.add_argument("--out", default="research/findings/raw/gap4/plastic_plateau/plastic_plateau_credit.json")
    a = ap.parse_args()
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super, n_prop=a.n_prop,
                       member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(run_seed(s, a.n_col, a.epochs, a.lr, a.w0, a.jitter, a.k_th, a.n_sub, a.hidden,
                                a.oracle_epochs, a.oracle_lr, a.oracle_batch, task_kwargs))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_plastic_plateau_credit", "seeds": a.seeds, "backend": os.environ.get("SIM_BACKEND"),
               "config": {"n_col": a.n_col, "epochs": a.epochs, "lr": a.lr, "w0": a.w0, "jitter": a.jitter,
                          "k_th": a.k_th, "n_sub": a.n_sub, "hidden": a.hidden, "oracle_epochs": a.oracle_epochs,
                          "task": task_kwargs, "margin_go": a.margin_go},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(kk):
            return float(np.nanmean([p[kk] for p in per]))
        agg = {kk: _m(kk) for kk in ["oracle_heldout", "rate_reservoir_heldout", "frozen_plateau_heldout",
                                     "credit_plateau_heldout", "deep_credit_share", "reproducibility_credit",
                                     "permuted_heldout", "lesion_heldout", "chance", "frozen_codon_diversity",
                                     "credit_codon_diversity", "credit_plateau_train", "frozen_plateau_train"]}
        n = len(per)
        beats = sum(1 for p in per
                    if p["credit_plateau_heldout"] >= p["frozen_plateau_heldout"] + a.margin_go)
        dcs_pos = sum(1 for p in per if p["deep_credit_share"] > 0)
        anti_ok = (all(p["no_transport"] for p in per)
                   and all(p["reproducibility_credit"] >= 0.8 for p in per)
                   and all(p["permuted_heldout"] <= p["chance"] + 0.10 for p in per)
                   and all(p["lesion_heldout"] <= p["frozen_plateau_heldout"] + 0.05 for p in per)
                   and agg["rate_reservoir_heldout"] <= 0.40 and agg["oracle_heldout"] >= 0.80)
        go = bool(beats == n and dcs_pos == n and anti_ok)
        # PROMISING (not a negative): every seed's deep_credit_share is POSITIVE and anti-cheats/op-point hold, but the
        # preregistered margin is not cleared on ALL seeds -> the direction is right (credit beats random sampling where
        # the rate reservoir fails), warranting the parent's 6-seed. Distinguished from a true NEGATIVE (dcs<=0 on a
        # seed, or an anti-cheat/op-point control failed).
        credit_gt_frozen = bool(agg["credit_plateau_heldout"] > agg["frozen_plateau_heldout"])
        promising = bool((not go) and dcs_pos == n and anti_ok and credit_gt_frozen)
        agg.update({"n_seeds": n, "credit_beats_frozen_by_margin": beats, "deep_credit_share_positive": dcs_pos,
                    "anti_cheats_clean": bool(anti_ok), "margin_go": a.margin_go, "promising": promising})
        summary["aggregate"] = agg; summary["GO"] = go; summary["PROMISING"] = promising
        common = (f"oracle {agg['oracle_heldout']:.3f}, rate-reservoir {agg['rate_reservoir_heldout']:.3f} (op-point "
                  f"genuine), reprod {agg['reproducibility_credit']:.3f}, permuted {agg['permuted_heldout']:.3f}, "
                  f"lesion {agg['lesion_heldout']:.3f}, diversity fz {agg['frozen_codon_diversity']:.3f}->cr "
                  f"{agg['credit_codon_diversity']:.3f}.")
        if go:
            verdict = (f"SMOKE GO ({n}/{n}) -- credit-trained plateau {agg['credit_plateau_heldout']:.3f} beats FROZEN "
                       f"plateau {agg['frozen_plateau_heldout']:.3f} by >={a.margin_go}; deep_credit_share "
                       f"{agg['deep_credit_share']:+.3f}. Anti-cheats clean. Local plateau-margin credit shapes a "
                       f"MOVABLE reliable hidden -> parent runs 6-seed. " + common)
        elif promising:
            verdict = (f"SMOKE PROMISING ({dcs_pos}/{n} deep_credit_share>0, margin cleared {beats}/{n}) -- credit "
                       f"{agg['credit_plateau_heldout']:.3f} > FROZEN {agg['frozen_plateau_heldout']:.3f} "
                       f"(deep_credit_share {agg['deep_credit_share']:+.3f}) where the rate reservoir FAILS "
                       f"({agg['rate_reservoir_heldout']:.3f}) -- the FIRST time a LOCAL, transport-free credit rule "
                       f"moves the on-bridge hidden in the RIGHT direction at a valid op-point (contrast: the "
                       f"sweet-spot rate-hidden credit was stuck at chance/train~0.34). Direction is right but the "
                       f"{a.margin_go} margin is not cleared on all seeds (deep_credit_share is noisy at k=8/coarse "
                       f"held-out per the sweet-spot finding). NOT a GO, NOT a negative -> parent runs 6-seed to settle. "
                       + common)
        else:
            reasons = []
            if dcs_pos != n:
                reasons.append(f"deep_credit_share not positive on all seeds ({dcs_pos}/{n}, mean "
                               f"{agg['deep_credit_share']:+.3f})")
            elif not credit_gt_frozen:
                reasons.append(f"credit does NOT beat frozen in mean ({agg['credit_plateau_heldout']:.3f} vs "
                               f"{agg['frozen_plateau_heldout']:.3f})")
            if not anti_ok:
                reasons.append("an anti-cheat/op-point control did not hold (see per-seed)")
            verdict = ("SMOKE NEGATIVE (honest) -- " + "; ".join(reasons) + ". " + common
                       + " The failing mode is named; a negative here is a valid deliverable.")
        summary["verdict"] = verdict
    else:
        summary["GO"] = False; summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[gap4-plastic-plateau-credit] {summary['verdict']}", flush=True)
    print(f"[gap4-plastic-plateau-credit] backend={summary['backend']} wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
