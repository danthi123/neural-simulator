"""gap#1/A1 — CLOSE the qualification on the mouth read-out e-prop learning GO
(`research/findings/2026-08-14-fluid-mouth-readout-eprop-learned-GO.md`, commit 6070d79d, runner
`_wkv_mouth_readout_eprop_learn_derisk`): that rung is 6/6 GO on the substantive claim (a local three-factor rule
recovers the mouth read-out head, host-linear recov 0.932, anti-cheats collapse) BUT is QUALIFIED because a per-step
substrate forward at the ~40k-position data volume was intractable (~1e6 sims) and the raw substrate margin is
bias-pinned, so **the gradient-step FORWARD used the host-linear margin `W@h+head_b` as a proxy** for the substrate
read; the learned weights were then DEMONSTRATED on the substrate. The named next lever was a BATCHED SUBSTRATE FORWARD.

THIS RUNNER runs the substrate graded-conductance read in BLOCK-DIAGONAL BATCHES (B independent copies of the read-out
circuit in ONE bridge, each driven by a DIFFERENT position's feature -> B substrate margins per ~read_window sim run),
so the e-prop learning FORWARD margin comes from the ACTUAL spiking substrate, tractably. The local three-factor rule
then learns `W_hat` end-to-end from that substrate error:
    margin_sub  = BATCHED_SUBSTRATE_READ(W_hat, h)         # [B,V] off cp_conductance_g_e/g_i (0 host matmul on it)
    margin      = margin_sub / gain  +  head_b               # gain-normalized substrate read + base-rate prior (a
                                                             #   [V] vector add, NOT a matmul); gain is a physical
                                                             #   conductance->logit calibration (random probe, once/seed)
    err_j       = softmax(margin)_j - 1{ j == target_t }   # DIRECT per-output error from the SUBSTRATE margin
    Delta w_ij  = -lr * err_j * h_i  -  wd * w_ij           # local delta (explicit outer product) + synaptic scaling
`head_w` feeds ONLY the teaching decision `target_t` (no weight transport); the update is an explicit np.outer of
(err, h) (no host gradient); the FORWARD margin is the substrate read (assert host_matmul_on_forward == 0 — the whole
point). The bias-pin the prior run hit is handled by SILENCING the tonic bias-pop in the learning forward (a clean
feature-driven substrate margin) and re-injecting head_b as the centered base-rate prior scaled to the margin spread
(exactly what the pipeline reads did, `ParityCloseRead._apply_baserate`) — head_b is a [V] vector add, NOT a matmul.

The learned W_hat is then DEMONSTRATED on the substrate with the SAME `_eval_substrate` / `LearnedReadout` (P=4
graded read) the prior GO used, and compared to the COPIED head read on the SAME substrate: the integrated bar is
substrate learned recov_argmax >= 0.85 * copied-head substrate recov (ratio >= 0.85; the prior integrated ~0.806).

GO (honest; read the actual numbers): >=5/6 seeds with
  (1) integrated_go: substrate learned recov_argmax >= 0.85 * copied-head substrate recov (SAME run, SAME eval set).
  (2) DECISIVE NEW assertion: host_matmul_on_the_learning_forward == 0 (the forward margin IS the substrate read).
  (3) anti-cheats COLLAPSE (host-linear discriminative channel + substrate): shuffle-teach / frozen / lesion-err.
If the batched substrate forward cannot reach the proxy's recovery (the substrate margin too noisy/biased even
batched): NOT a wall — the gap is quantified + the next lever named (margin calibration / larger batch / different
read normalization). Honest PARTIAL + next lever is the deliverable.

ANTI-CHEATS: no weight transport (update reads only (h, err); head_w only for the teaching label); no host gradient
(explicit outer product); the FORWARD is the substrate read (host_matmul_on_forward == 0); shuffle-teach/frozen/
lesion-err collapse; determinism via cfg.seed (NOT actual_seed_used) with a build-twice hash of
cp_neuron_firing_thresholds; host_rng_draws_on_read_path == 0. Uses tools.lab helpers. Runner-only, additive,
default-off, NO sim/ edit. Biology: Bellec et al. Nat Commun 11:3625 (2020) e-prop; Urbanczik & Senn Neuron 81:521
(2014) dendritic-prediction delta rule; Turrigiano synaptic scaling (weight decay companion).

Run (smoke):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk \
                --smoke --seeds 42
Run (6-seed): SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk \
                --seeds 42,43,44,100,101,102 \
                --json research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402

from tools.lab import lever, void_if, undefined_if_empty, assert_backend, project_cost  # noqa: E402

from research.runners._wkv_mouth_endtoend_substrate_read_derisk import ComposedEndToEndRead  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import (  # noqa: E402
    LearnedReadout, _host_feat, _positions, _positions_sub, _eval_hostlinear, _eval_substrate,
)
from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _native, _load_eval,
)


# ====================================================================================================================
# The BATCHED substrate read-out: B independent BLOCK-DIAGONAL copies of the graded-conductance read-out in ONE
# bridge. Block b's hid/hidinh are driven by position b's feature; each block reads its own V word-pools off
# cp_conductance_g_e/g_i. One ~read_window sim run -> B substrate margins [B, V]. The read-out weights W_hat are
# SHARED across blocks (scattered into every block's fixed CSR each gradient step, on the GPU). The tonic bias-pop is
# built but SILENCED in the learning forward (the base-rate prior is re-injected host-side as a centered [V] vector,
# not a matmul — the prior is a declared residual, same as the prior GO). No FS-WTA (inert in the subthreshold graded
# read; dropping it removes cross-block coupling and shrinks the edge budget). Inherits df_e/df_i/v_ref/Dale-split
# from the ComposedEndToEndRead chain; only _build_bridge / _wire are B-blocked.
# ====================================================================================================================
class BatchedSubstrateReadout(ComposedEndToEndRead):
    def __init__(self, ro, seed, B, hid_pop=4, pop=1, dendritic=False, apical_drive_pA=600.0,
                 apical_baseline_pA=220.0, apical_syn_scale=12.0, dendritic_tau=1.0,
                 dendritic_logit_spread=4.0, n_apical_i=16, **kw):
        self.B = int(B)
        # ---- DENDRITIC (Urbanczik-Senn) lever: a SECOND, target-driven APICAL population wired onto the SAME word-pools
        # as an independent synaptic teaching pathway (gated OFF by default -> byte-identical to the softmax-onehot rule).
        # Set BEFORE super().__init__ because the overridden _build_bridge/_wire run inside it. ----
        self.dendritic = bool(dendritic)
        self.apical_drive_pA = float(apical_drive_pA)          # target-word excitatory teacher drive (labelled-line)
        self.apical_baseline_pA = float(apical_baseline_pA)    # tonic inhibitory baseline on ALL pools (non-target -> low)
        self.apical_syn_scale = float(apical_syn_scale)
        self.dendritic_tau = float(dendritic_tau)              # sigmoid temperature (nats)
        self.dendritic_logit_spread = float(dendritic_logit_spread)   # apical calibrated to +/- this logit at target/non
        self.n_apical_i = int(n_apical_i)
        self._apical_cal = None                                # (center, scale) set once per seed by calibrate_apical
        # feature = host (drive the substrate read-out with the host feature r_h*(Wo_sp@state) — the prior GO's Arm A
        # isolation read; the projection-feature composition is a separate, even costlier lever).
        super().__init__(ro, seed, proj=None, use_proj=False, use_bias_pop=True, hb_k=0.0,
                         pop=pop, hid_pop=hid_pop, **kw)
        self._build_batch_slot_map()
        self._n_substrate_reads = 0            # counts batched-forward reads — must equal the gradient-step count
        self._n_apical_reads = 0               # counts APICAL teacher reads (dendritic) — a separate provenance counter

    # ---- bridge: B block-diagonal copies of (hid, hidinh, wpool, bias_e, bias_i); region-major layout ----
    def _build_bridge(self):
        B = self.B
        cfg = CoreSimConfig()
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0; cfg.seed = self.seed
        cfg.heterogeneity_seed = self.seed; cfg.ou_seed = self.seed
        cfg.enable_brain_region_framework = True
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_input_divisive_norm",
                  "enable_nmda"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.enable_ou_process = self.ou_std > 0.0
        cfg.ou_mean_current_pA = 0.0; cfg.ou_std_current_pA = self.ou_std; cfg.ou_tau_ms = 15.0
        cfg.stdp_w_max = 4000.0; cfg.hebbian_max_weight = 4000.0
        Hn = self.F * self.Hp
        VP = self.V * self.P
        regions = [
            BrainRegion(name="hid", n_neurons=B * Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hidinh", n_neurons=B * Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="wpool", n_neurons=B * VP, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="bias_e", n_neurons=B * self.n_bias, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="bias_i", n_neurons=B * self.n_bias, exc_fraction=1.0, internal_density=0.0),
        ]
        if self.dendritic:
            # APICAL teacher (labelled-line): one excitatory teacher neuron per (block, word) -> its own word-pool;
            # a tonic inhibitory baseline population per block -> ALL word-pools (so a non-target pool reads LOW).
            regions.append(BrainRegion(name="apical_e", n_neurons=B * self.V, exc_fraction=1.0, internal_density=0.0))
            regions.append(BrainRegion(name="apical_i", n_neurons=B * self.n_apical_i, exc_fraction=1.0,
                                       internal_density=0.0))
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        # per-block index views (region-major layout: block b = the b-th contiguous slice of each region)
        self.hid_all = np.asarray(list(rm.indices("hid")), dtype=np.int64).reshape(B, Hn)
        self.hidinh_all = np.asarray(list(rm.indices("hidinh")), dtype=np.int64).reshape(B, Hn)
        self.wpool_all = np.asarray(list(rm.indices("wpool")), dtype=np.int64).reshape(B, self.V, self.P)
        self.bias_e_all = np.asarray(list(rm.indices("bias_e")), dtype=np.int64).reshape(B, self.n_bias)
        self.bias_i_all = np.asarray(list(rm.indices("bias_i")), dtype=np.int64).reshape(B, self.n_bias)
        if self.dendritic:
            self.apical_e_all = np.asarray(list(rm.indices("apical_e")), dtype=np.int64).reshape(B, self.V)
            self.apical_i_all = np.asarray(list(rm.indices("apical_i")), dtype=np.int64).reshape(B, self.n_apical_i)
        self.hid_dim = np.repeat(np.arange(self.F), self.Hp).astype(np.int64)   # [Hn] feature dim of each hid neuron
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())

    # ---- wiring: per block, Wp (exc hid->pools) + Wn (inh hidinh->pools) + bias-pop (head_b as signed conductance) ----
    def _wire(self):
        b = self._b
        B = self.B
        Hn = self.F * self.Hp
        nB = self.n_bias
        Wp = (self.Wp * self.syn_scale).astype(np.float32)
        Wn = (self.Wn * self.syn_scale * self.ratio).astype(np.float32)
        Wp_hn = Wp[:, self.hid_dim]                                             # [V, Hn]
        Wn_hn = Wn[:, self.hid_dim]
        # head_b as a signed tonic conductance (centered, unk pinned low) — same as ComposedEndToEndRead._wire.
        hb = self.head_b.astype(np.float64).copy()
        if self.ro.unk_idx >= 0:
            hb[self.ro.unk_idx] = hb.min()
        hb = hb - hb.mean()
        hb_pos = np.maximum(hb, 0.0); hb_neg = np.maximum(-hb, 0.0)
        # per-block edge templates (identical structure/weights across blocks)
        wp_block = np.repeat(Wp_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        wn_block = np.repeat(Wn_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        wbp_block = (np.repeat(np.repeat(hb_pos, self.P), nB).astype(np.float32)
                     * (self.syn_scale * self.bias_scale))
        wbn_block = (np.repeat(np.repeat(hb_neg, self.P), nB).astype(np.float32)
                     * (self.syn_scale * self.ratio * self.bias_scale))
        # APICAL teacher edge templates (dendritic only): apical_e(word)->its P pool members (identity, excitatory);
        # apical_i (tonic) -> ALL pools (inhibitory baseline). Raw magnitudes are calibrated away in calibrate_apical.
        nA = self.n_apical_i
        wapp_block = np.full(self.V * self.P, self.apical_syn_scale, np.float32)                       # [V*P]
        wapn_block = np.full(self.V * self.P * nA, self.apical_syn_scale * self.ratio, np.float32)     # [V*P*nA]
        pre_pos, post_pos, w_pos = [], [], []
        pre_neg, post_neg, w_neg = [], [], []
        pre_bp, post_bp, w_bp = [], [], []
        pre_bn, post_bn, w_bn = [], [], []
        pre_ap, post_ap, w_ap = [], [], []
        pre_an, post_an, w_an = [], [], []
        for blk in range(B):
            hid_b = self.hid_all[blk]; hidinh_b = self.hidinh_all[blk]
            pool_b = self.wpool_all[blk].reshape(-1)                            # [V*P]
            be_b = self.bias_e_all[blk]; bi_b = self.bias_i_all[blk]
            pre_pos.append(np.tile(hid_b, self.V * self.P)); post_pos.append(np.repeat(pool_b, Hn)); w_pos.append(wp_block)
            pre_neg.append(np.tile(hidinh_b, self.V * self.P)); post_neg.append(np.repeat(pool_b, Hn)); w_neg.append(wn_block)
            pre_bp.append(np.tile(be_b, self.V * self.P)); post_bp.append(np.repeat(pool_b, nB)); w_bp.append(wbp_block)
            pre_bn.append(np.tile(bi_b, self.V * self.P)); post_bn.append(np.repeat(pool_b, nB)); w_bn.append(wbn_block)
            if self.dendritic:
                ape_b = self.apical_e_all[blk]; api_b = self.apical_i_all[blk]
                pre_ap.append(np.repeat(ape_b, self.P)); post_ap.append(pool_b); w_ap.append(wapp_block)
                pre_an.append(np.tile(api_b, self.V * self.P)); post_an.append(np.repeat(pool_b, nA)); w_an.append(wapn_block)
        union = {
            "readout_pos": {"pre_indices": np.concatenate(pre_pos), "post_indices": np.concatenate(post_pos),
                            "initial_weights": np.concatenate(w_pos), "plastic": False, "conn_type": "E_TO_E"},
            "readout_neg": {"pre_indices": np.concatenate(pre_neg), "post_indices": np.concatenate(post_neg),
                            "initial_weights": np.concatenate(w_neg), "plastic": False, "conn_type": "I_TO_E"},
            "bias_pos": {"pre_indices": np.concatenate(pre_bp), "post_indices": np.concatenate(post_bp),
                         "initial_weights": np.concatenate(w_bp), "plastic": False, "conn_type": "E_TO_E"},
            "bias_neg": {"pre_indices": np.concatenate(pre_bn), "post_indices": np.concatenate(post_bn),
                         "initial_weights": np.concatenate(w_bn), "plastic": False, "conn_type": "I_TO_E"},
        }
        if self.dendritic:
            union["apical_pos"] = {"pre_indices": np.concatenate(pre_ap), "post_indices": np.concatenate(post_ap),
                                   "initial_weights": np.concatenate(w_ap), "plastic": False, "conn_type": "E_TO_E"}
            union["apical_neg"] = {"pre_indices": np.concatenate(pre_an), "post_indices": np.concatenate(post_an),
                                   "initial_weights": np.concatenate(w_an), "plastic": False, "conn_type": "I_TO_E"}
        inh_list = [self.hidinh_all.reshape(-1), self.bias_i_all.reshape(-1)]
        if self.dendritic:
            inh_list.append(self.apical_i_all.reshape(-1))
        inh = np.concatenate(inh_list).tolist()
        b.inject_explicit_wiring(union, output_inhibitory_indices=inh)
        self._pos_edges = (union["readout_pos"]["pre_indices"], union["readout_pos"]["post_indices"], None)
        self._neg_edges = (union["readout_neg"]["pre_indices"], union["readout_neg"]["post_indices"], None)
        self._wp_block = wp_block; self._wn_block = wn_block

    # ---- CSR slot map (edge -> data position), once; per-step set_weights is then a GPU scatter ----
    def _build_batch_slot_map(self):
        b = self._b
        n = int(b.core_config.num_neurons)
        indptr = np.asarray(to_host(b.cp_connections.indptr)).astype(np.int64)
        indices = np.asarray(to_host(b.cp_connections.indices)).astype(np.int64)
        csr_pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        csr_key = csr_pre * n + indices                                        # strictly increasing (row-major CSR)
        pos_pre, pos_post, _ = self._pos_edges
        neg_pre, neg_post, _ = self._neg_edges
        pos_slot = np.searchsorted(csr_key, pos_pre.astype(np.int64) * n + pos_post.astype(np.int64))
        neg_slot = np.searchsorted(csr_key, neg_pre.astype(np.int64) * n + neg_post.astype(np.int64))
        assert np.all(csr_key[pos_slot] == pos_pre.astype(np.int64) * n + pos_post.astype(np.int64))
        assert np.all(csr_key[neg_slot] == neg_pre.astype(np.int64) * n + neg_post.astype(np.int64))
        xp, _ = get_backend()
        self._pos_slot = xp.asarray(pos_slot)
        self._neg_slot = xp.asarray(neg_slot)
        del csr_pre, csr_key, indices
        # GPU views used every step (drive + conductance read)
        self._hid_flat = xp.asarray(self.hid_all.reshape(-1))
        self._hidinh_flat = xp.asarray(self.hidinh_all.reshape(-1))
        self._wpool_flat = xp.asarray(self.wpool_all.reshape(-1))
        self._hid_dim_gpu = xp.asarray(self.hid_dim)
        if self.dendritic:
            self._apical_e_all_gpu = xp.asarray(self.apical_e_all)              # [B, V] teacher index per (block, word)
            self._apical_i_flat = xp.asarray(self.apical_i_all.reshape(-1))     # [B*nA] tonic inhibitory baseline pop

    def set_weights(self, W_hat):
        """Dale-split W_hat[V,D] -> Wp/Wn word-pool synapses (SHARED across all B blocks) and scatter into the fixed
        CSR on the GPU. head_b (the bias pop) is untouched. This is the ONLY per-gradient-step weight write."""
        xp, _ = get_backend()
        Wfull = np.concatenate([W_hat, -W_hat], axis=1)
        Wp = np.maximum(Wfull, 0.0); Wn = np.maximum(-Wfull, 0.0)
        wp_block = np.repeat(Wp[:, self.hid_dim] * self.syn_scale, self.P, axis=0).reshape(-1).astype(np.float32)
        wn_block = np.repeat(Wn[:, self.hid_dim] * self.syn_scale * self.ratio, self.P, axis=0).reshape(-1).astype(np.float32)
        wp_all = xp.asarray(np.tile(wp_block, self.B))
        wn_all = xp.asarray(np.tile(wn_block, self.B))
        data = self._b.cp_connections.data
        data[self._pos_slot] = wp_all.astype(data.dtype)
        data[self._neg_slot] = wn_all.astype(data.dtype)
        # STALE-COO/WT FIX (2026-08-27): this edits cp_connections.data IN PLACE (same object id + nnz), so the
        # read-only megakernel-v2 transposed-CSR cache (keyed on id+nnz) would otherwise transmit the PREVIOUS
        # set_weights' matrix -- the artifact that manufactured the "structure-selective read wall" AND starved
        # this eprop training loop of its own weight updates (||W||->cap runaway). Signal the in-place edit so the
        # next batch_margin transmits THESE weights. Byte-identical on a fresh build (cache already rebuilds).
        self._b.mark_weights_edited()
        self.head_w = np.asarray(W_hat, dtype=np.float64)

    def batch_margin(self, feats_signed, silence_bias=True):
        """feats_signed: [B, D] signed host feature h. Split to dual-nonneg [B, F], drive block b's hid/hidinh, run
        read_window steps, integrate the per-pool net signed synaptic current off cp_conductance_g_e/g_i -> [B, V]
        substrate margins. 0 host matmul on the margin (df_e*g_e + df_i*g_i is the substrate's own current)."""
        b = self._b
        xp, _ = get_backend()
        nb = feats_signed.shape[0]
        assert nb == self.B, "batch_margin expects a full block (%d), got %d" % (self.B, nb)
        featF = np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)  # [B, F]
        self._reset()
        drive = xp.zeros(b.core_config.num_neurons, dtype=xp.float64)
        fdrive = xp.asarray(self.hid_bias + self.hid_gain * featF[:, self.hid_dim]).reshape(-1)           # [B*Hn]
        drive[self._hid_flat] = fdrive
        drive[self._hidinh_flat] = fdrive
        if self.use_bias_pop and not silence_bias:
            be = xp.asarray(self.bias_e_all.reshape(-1)); bi = xp.asarray(self.bias_i_all.reshape(-1))
            drive[be] = self.bias_drive_pA; drive[bi] = self.bias_drive_pA
        if self.floor_pA:
            drive[self._wpool_flat] += self.floor_pA
        b.cp_external_input_current[:] = drive.astype(b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        ge_sum = xp.zeros(self.B * self.V, dtype=xp.float64)
        gi_sum = xp.zeros(self.B * self.V, dtype=xp.float64)
        n_acc = 0
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            ge = b.cp_conductance_g_e[self._wpool_flat].astype(xp.float64).reshape(self.B * self.V, self.P).sum(axis=1)
            gi = b.cp_conductance_g_i[self._wpool_flat].astype(xp.float64).reshape(self.B * self.V, self.P).sum(axis=1)
            ge_sum += ge; gi_sum += gi; n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        margin = (self.df_e * (ge_sum / n_acc) + self.df_i * (gi_sum / n_acc))
        self._n_substrate_reads += 1
        return np.asarray(to_host(margin)).reshape(self.B, self.V)

    # ---- DENDRITIC: the APICAL teacher read (independent of W; the target enters via a spiking top-down pathway) ----
    def apical_margin(self, targets, freeze_apical=False):
        """The APICAL teaching read. Drive apical_e[(b, targets[b])] high (one-hot per block) + apical_i tonic; the
        feedforward hid/hidinh are SILENT. Run read_window, integrate the pools' net signed synaptic current off
        cp_conductance_g_e/g_i -> [B, V] APICAL margin (target pool high, non-target low). This is a FIXED teacher
        pathway (0 host matmul on the margin; independent of the read-out weights W). freeze_apical=True silences the
        one-hot target drive (baseline only) -> the anti-cheat that removes the teacher."""
        b = self._b
        xp, _ = get_backend()
        targets = np.asarray(targets, dtype=np.int64)
        assert targets.shape[0] == self.B, "apical_margin expects B targets"
        self._reset()
        drive = xp.zeros(b.core_config.num_neurons, dtype=xp.float64)
        if not freeze_apical:
            tgt_idx = self._apical_e_all_gpu[xp.arange(self.B), xp.asarray(targets)]    # [B] teacher neuron per block
            drive[tgt_idx] = self.apical_drive_pA
        drive[self._apical_i_flat] = self.apical_baseline_pA
        b.cp_external_input_current[:] = drive.astype(b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        ge_sum = xp.zeros(self.B * self.V, dtype=xp.float64)
        gi_sum = xp.zeros(self.B * self.V, dtype=xp.float64)
        n_acc = 0
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            ge = b.cp_conductance_g_e[self._wpool_flat].astype(xp.float64).reshape(self.B * self.V, self.P).sum(axis=1)
            gi = b.cp_conductance_g_i[self._wpool_flat].astype(xp.float64).reshape(self.B * self.V, self.P).sum(axis=1)
            ge_sum += ge; gi_sum += gi; n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        margin = (self.df_e * (ge_sum / n_acc) + self.df_i * (gi_sum / n_acc))
        self._n_apical_reads += 1
        return np.asarray(to_host(margin)).reshape(self.B, self.V)

    def calibrate_apical(self, targets):
        """Measure the apical teacher's response ONCE (a real substrate read): m_target (the driven pool) vs
        m_nontarget (all others), then set (center, scale) so the calibrated apical logit is +spread at the target and
        -spread elsewhere -> sigma(apical) approximates a clean one-hot teacher REGARDLESS of the raw drive magnitude
        (a physical unit calibration, exactly like the conductance->logit `gain`). scale keeps the sign of the
        separation; a degenerate (~0) separation yields scale 0 -> sigma 0.5 everywhere -> the run honestly fails."""
        targets = np.asarray(targets, dtype=np.int64)
        am = self.apical_margin(targets)                                          # [B, V]
        B = self.B
        m_target = float(am[np.arange(B), targets].mean())
        mask = np.ones_like(am, dtype=bool); mask[np.arange(B), targets] = False
        m_nontarget = float(am[mask].mean())
        half = 0.5 * (m_target - m_nontarget)
        center = 0.5 * (m_target + m_nontarget)
        scale = (self.dendritic_logit_spread / half) if abs(half) > 1e-9 else 0.0
        self._apical_cal = (center, scale, m_target, m_nontarget)
        return self._apical_cal

    def apical_p(self, apical_margin):
        """sigma of the CALIBRATED apical margin -> the per-unit teaching target (~one-hot). Reuses (center, scale)."""
        center, scale = self._apical_cal[0], self._apical_cal[1]
        logit = (apical_margin - center) * scale / max(1e-9, self.dendritic_tau)
        return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------------------------------------------------
def _softmax_rows(logits):
    m = logits - logits.max(1, keepdims=True); P = np.exp(m); P /= P.sum(1, keepdims=True)
    return P


def _sub_logits(margin_sub, gain, head_b, unk):
    """The learning-forward logits from the SUBSTRATE margin: DIVIDE by the calibrated conductance->logit gain (so
    the substrate margin sits in the read-out's logit units), then ADD the base-rate prior head_b (a [V] vector, NOT a
    matmul; head_b is the declared copied residual). This makes the substrate forward a physically-calibrated stand-in
    for the host-linear margin W@h+head_b, so the proven lr/decay transfer — but the margin IS the substrate read, so
    the spiking noise + graded-read nonlinearity drive the error. At small ||W|| the (small) feature margin leaves
    logits ~ head_b (the base-rate prior), which grows into a feature-dominated read as W learns: NO bias-pin, and NO
    per-position z-score amplifying the init noise into confident-wrong predictions."""
    logits = margin_sub / gain + head_b[None, :]
    if unk >= 0:
        logits = logits.copy(); logits[:, unk] = -1e30
    return logits


def _calibrate_gain(s_batch, ro, feats_signed, seed):
    """Measure the substrate's conductance->logit GAIN once (per seed) with a RANDOM PROBE weight (NOT the teacher
    head_w — a physical measurement of the wiring's input->output gain, like reading an amplifier's gain with a test
    signal). margin_sub ~ G * (Wfull_probe @ feat); G = <margin_sub, host_probe>/<host_probe, host_probe>. Returns
    (G, corr) where corr is the substrate-vs-ideal-linear-map correlation (a sanity read of the graded linearity).
    The host_probe matmul here is CALIBRATION only (a random probe, outside the learning loop); the learning forward
    is pure substrate (0 host matmul)."""
    rng = np.random.default_rng(seed * 7 + 3)
    W_probe = 0.12 * rng.standard_normal((ro.V, ro.D))                          # random direction, mid-training scale
    Wfull_probe = np.concatenate([W_probe, -W_probe], axis=1)                   # [V, 2D]
    featF = np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)  # [B, 2D]
    host_probe = featF @ Wfull_probe.T                                         # [B, V] IDEAL linear margin (CALIB only)
    s_batch.set_weights(W_probe)
    margin_sub = s_batch.batch_margin(feats_signed, silence_bias=True)         # [B, V] the SUBSTRATE read of the probe
    num = float((margin_sub * host_probe).sum()); den = float((host_probe * host_probe).sum())
    gain = num / max(1e-12, den)
    hp = host_probe.reshape(-1); ms = margin_sub.reshape(-1)
    corr = float(np.corrcoef(hp, ms)[0, 1]) if hp.std() > 1e-12 and ms.std() > 1e-12 else 0.0
    return gain, round(corr, 4)


def _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b, mode="main", max_steps=None,
                             traj_eval=None, traj_out=None, freeze_apical=False):
    """The LOCAL three-factor delta rule with the FORWARD margin from the BATCHED SUBSTRATE READ. mode: main |
    frozen | lesion_err | shuffle_teach. Returns (W_hat[V,D], n_grad_steps, n_matmul_forward).

    --forward host_proxy (default substrate): swaps the substrate forward for the host-linear margin W@h+head_b (the
    2026-08-14 proxy-GO forward, the SHORTCUT) at the IDENTICAL operating point / coverage / eval as the substrate
    arm — a CONTROL that isolates the forward from coverage. host_proxy increments n_matmul_forward each step, so
    forward_is_substrate=False and go=False for it (it is never a GO, only a coverage/operating-point reference).
    traj_out (opt): if given and --eval-every-epochs>0, host-linear recovery is recorded every K epochs (cheap, no
    substrate sim) to expose plateau-vs-still-climbing convergence. Both are additive; substrate arm is byte-identical."""
    V, D = ro.V, ro.D
    rng = np.random.default_rng(seed * 991 + 7)
    W = (0.0 if args.zero_init else 0.01) * rng.standard_normal((V, D))
    if mode == "frozen":
        return W, 0, 0                                                          # random init, no update -> the floor
    B = s_batch.B
    idx = np.arange(len(H))
    n_full = (len(idx) // B) * B
    n_steps_total = args.epochs * (n_full // B)
    if mode == "lesion_err":
        # err == 0 -> NO potentiation, only weight decay (== a decayed floor). No substrate forward needed: the
        # margin would be multiplied by a zero error, so running it would only waste ~n_grad substrate sims.
        n_dec = min(n_steps_total, max_steps) if max_steps is not None else n_steps_total
        for _ in range(n_dec):
            W = W - args.weight_decay * W
        return W, n_dec, 0
    perm = rng.permutation(V) if mode == "shuffle_teach" else None
    Ye = perm[Y] if perm is not None else Y
    unk = ro.unk_idx
    n_grad = 0; n_matmul_forward = 0
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for start in range(0, n_full, B):
            bi = idx[start:start + B]
            Hb = H[bi]                                                          # [B, D] signed host feature
            if getattr(args, "dendritic", False) and getattr(args, "forward", "substrate") != "host_proxy":
                # ---- DENDRITIC (Urbanczik-Senn) local error: TWO independent substrate reads. BASAL = the feedforward
                # prediction (as today); APICAL = a teacher read where the TARGET enters via its own spiking pathway.
                # err = sigma(apical) - sigma(basal) = target - prediction (PER-UNIT sigmoids, NOT a cross-unit softmax
                # -> read noise in one word no longer corrupts every other word's error). The teacher never touches the
                # forward ANSWER (apical is off in the demo read). W += lr*err@h is the delta rule toward the target. ----
                s_batch.set_weights(W)
                basal = s_batch.batch_margin(Hb, silence_bias=True)            # [B, V] SUBSTRATE basal prediction read
                apical = s_batch.apical_margin(Ye[bi], freeze_apical=freeze_apical)   # [B, V] SUBSTRATE teacher read
                basal_logit = basal / gain + head_b[None, :]
                if unk >= 0:
                    basal_logit = basal_logit.copy(); basal_logit[:, unk] = -60.0     # sentinel (err[unk] zeroed below)
                p_basal = 1.0 / (1.0 + np.exp(-np.clip(basal_logit / max(1e-9, s_batch.dendritic_tau), -60.0, 60.0)))
                p_apical = s_batch.apical_p(apical)                            # ~one-hot target (calibrated apical)
                err = p_apical - p_basal                                       # [B, V] local prediction error
                if unk >= 0:
                    err = err.copy(); err[:, unk] = 0.0
                W = W + args.lr * (err.T @ Hb) / B - args.weight_decay * W     # ascent toward the target + decay
            else:
                if getattr(args, "forward", "substrate") == "host_proxy":
                    logits = Hb @ W.T + head_b[None, :]                         # CONTROL: host-linear proxy forward
                    if unk >= 0:
                        logits = logits.copy(); logits[:, unk] = -1e30
                    n_matmul_forward += 1                                       # host matmul on forward -> not substrate
                else:
                    s_batch.set_weights(W)
                    margin_sub = s_batch.batch_margin(Hb, silence_bias=True)    # [B, V] ACTUAL SUBSTRATE READ
                    logits = _sub_logits(margin_sub, gain, head_b, unk)         # gain-normalized + base-rate (no matmul)
                P = _softmax_rows(logits)
                P[np.arange(B), Ye[bi]] -= 1.0                                  # err = softmax - onehot (substrate err)
                # local delta: -lr * sum_b err_b (x) h_b / B  (explicit outer product = the UPDATE, allowed) + decay
                W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
            # SYNAPTIC SCALING (Turrigiano homeostasis): hold ||W|| in the substrate's LINEAR read range. The graded
            # read saturates for large ||W|| (the gain, calibrated at ||W||~||head_w||, drops), so an un-scaled forward
            # UNDER-reads, the softmax never gets confident, err persists and W runs away (measured ||W||~970 vs
            # head_w 37). Scaling holds the MAGNITUDE while the local rule steers the DIRECTION toward head_w.
            if args.w_target > 0:
                nrm = float(np.linalg.norm(W))
                if nrm > args.w_target:
                    W *= args.w_target / nrm
            n_grad += 1
            if max_steps is not None and n_grad >= max_steps:
                return W, n_grad, n_matmul_forward
        # convergence trajectory (host-linear recovery; cheap matmul eval, no substrate sim) — off by default
        if (traj_out is not None and traj_eval is not None and getattr(args, "eval_every_epochs", 0) > 0
                and ((ep + 1) % args.eval_every_epochs == 0)):
            He_, Ye_, PFe_, hw_ = traj_eval
            rr = _eval_hostlinear(ro, W, He_, Ye_, PFe_)
            wc = _wcos(W, hw_)
            traj_out.append({"epoch": ep + 1, "n_grad": n_grad,
                             "hostlinear_recov_argmax": round(float(rr["recov_argmax"]), 4), "weight_cosine": wc})
            print(f"[traj seed {seed} ep {ep + 1}/{args.epochs}] hostlinear_recov="
                  f"{rr['recov_argmax']:.4f} wcos={wc} n_grad={n_grad}", flush=True)
    return W, n_grad, n_matmul_forward


def _held_frame_error(s_batch, ro, Hb, Yb, gain, head_b):
    """Mean cross-entropy of the SUBSTRATE-margin softmax vs the target on a held frame (the verify-first metric)."""
    margin_sub = s_batch.batch_margin(Hb, silence_bias=True)
    P = _softmax_rows(_sub_logits(margin_sub, gain, head_b, ro.unk_idx))
    ce = -np.log(np.clip(P[np.arange(len(Yb)), Yb], 1e-12, 1.0)).mean()
    argerr = float((P.argmax(1) != Yb).mean())
    return float(ce), argerr


def _verify_first(seed, ro, s_batch, H, Y, args, gain, head_b):
    """VERIFY-FIRST guard: a few substrate-forward updates on ONE held batch must REDUCE that batch's error (the
    substrate forward produces a usable gradient). Train and eval on the SAME batch (the strongest signal)."""
    B = s_batch.B
    rng = np.random.default_rng(seed * 13 + 1)
    bi = rng.choice(len(H), size=B, replace=False)
    Hb = H[bi]; Yb = Y[bi]
    W = 0.01 * np.random.default_rng(seed * 991 + 7).standard_normal((ro.V, ro.D))
    s_batch.set_weights(W)
    ce0, ae0 = _held_frame_error(s_batch, ro, Hb, Yb, gain, head_b)
    for _ in range(args.verify_steps):
        s_batch.set_weights(W)
        margin_sub = s_batch.batch_margin(Hb, silence_bias=True)
        P = _softmax_rows(_sub_logits(margin_sub, gain, head_b, ro.unk_idx))
        P[np.arange(B), Yb] -= 1.0
        W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
        if args.w_target > 0:
            nrm = float(np.linalg.norm(W))
            if nrm > args.w_target:
                W *= args.w_target / nrm
    s_batch.set_weights(W)
    ce1, ae1 = _held_frame_error(s_batch, ro, Hb, Yb, gain, head_b)
    lever(f"verify_first_substrate_forward_CE_seed{seed}", before=round(ce0, 4), after=round(ce1, 4),
          required=True, continuous=round(ae0 - ae1, 4))
    ok = bool(ce1 < ce0 - 1e-4 and ae1 <= ae0)
    print(f"[verify-first seed {seed}] CE {ce0:.4f} -> {ce1:.4f} argerr {ae0:.3f} -> {ae1:.3f} "
          f"({'REDUCED' if ok else 'NOT REDUCED'}) over {args.verify_steps} substrate-forward updates", flush=True)
    return ok, ce0, ce1, ae0, ae1


def _wcos(W, hw):
    return round(float((W.reshape(-1) @ hw.reshape(-1)) / (np.linalg.norm(W) * np.linalg.norm(hw) + 1e-12)), 4)


def _demo_feats(ro, seed, sub_tuples, args):
    """The host presynaptic feature r_h*(Wo_sp@state) per demo position (bridge-independent for feature=host), so it
    is computed ONCE and shared across the fresh per-W substrate reads."""
    return [_host_feat(ro, ap, an, tid) for (ap, an, tid) in sub_tuples]


def _fresh_substrate_read(ro, seed, W, sub_tuples, Ys, PFs, feats, args):
    """Read W on the substrate graded read-out on a FRESH LearnedReadout with cp.random reseeded to `seed`. Fresh +
    reseeded per read because a reused bridge lets a large-||W|| read leave persistent state that corrupts the next
    read (measured: copied recov 0.976 -> 0.0004 after a ||W||=40 read on the same bridge); a fresh reseeded bridge
    isolates each read and gives every W IDENTICAL OU noise (a fair A/B)."""
    try:
        import cupy as _cp
        if os.environ.get("SIM_BACKEND", "numpy") == "cupy":
            _cp.random.seed(int(seed))
    except Exception:
        pass
    s = LearnedReadout(ro, seed, feature="host", pop=args.pop, hid_pop=args.hid_pop, ou_std=args.ou_std,
                       read_window=args.read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                       settle_frac=args.settle_frac, n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA,
                       bias_scale=args.bias_scale)
    r = _eval_substrate(s, W, sub_tuples, Ys, PFs, feats=feats)
    r["host_rng_draws"] = int(s.n_host_rng_draws)
    del s
    return r


def _thr_hash(seed, ro, hid_pop, pop, ou_std, read_window, hid_gain, ratio, n_bias, bias_drive_pA):
    # a SMALL B (the seed-trap check is build-twice-determinism at a fixed config; the full B=48 net is ~100M edges
    # whose slot-map / wiring build is expensive, and identical seeding is proven at any B).
    s = BatchedSubstrateReadout(ro, seed, 4, hid_pop=hid_pop, pop=pop, ou_std=ou_std, read_window=read_window,
                                hid_gain=hid_gain, ratio=ratio, n_bias=n_bias, bias_drive_pA=bias_drive_pA)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    del s
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def run_seed(seed, ro, args):
    # -- data: DISJOINT train / eval sentence splits (held-out context), reusing the prior runner's position builder --
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)                 # host features for LEARNING
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)               # host-linear eval (diagnostic)
    sub_tuples, Ys, PFs = _positions_sub(ro, eval_ids, args.warmup, args.n_sub_demo)   # substrate demo eval set
    void_if(len(H) < args.batch or len(He) == 0 or len(sub_tuples) == 0, "insufficient train/eval positions")

    # the base-rate prior head_b (the copied residual, unk suppressed later in _sub_logits): added as a [V] vector to
    # the gain-normalized substrate margin (NOT a matmul). Same base-rate the host-linear forward used.
    head_b = ro.head_b.astype(np.float64)

    # -- the BATCHED substrate read-out (learning-forward substrate) --
    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA,
                                      dendritic=getattr(args, "dendritic", False),
                                      apical_drive_pA=args.apical_drive_pA, apical_baseline_pA=args.apical_baseline_pA,
                                      apical_syn_scale=args.apical_syn_scale, dendritic_tau=args.dendritic_tau,
                                      dendritic_logit_spread=args.dendritic_logit_spread, n_apical_i=args.n_apical_i)

    # -- CALIBRATE the conductance->logit GAIN once per seed (a physical measurement of the wiring, RANDOM probe, NOT
    #    the teacher). This puts the substrate forward in the read-out's logit units so the proven lr/decay transfer. --
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.batch], seed)
    print(f"[calib-gain seed {seed}] conductance->logit gain={gain:.5g} (substrate-vs-linear corr={gain_corr})",
          flush=True)

    # -- DENDRITIC: calibrate the APICAL teacher ONCE per seed (measure m_target vs m_nontarget; set the unit map so
    #    sigma(apical) is a clean +/-spread one-hot). Provenance: this is a real substrate read of the teacher pathway. --
    apical_cal = None
    if getattr(args, "dendritic", False):
        apical_cal = s_batch.calibrate_apical(Y[:args.batch])
        print(f"[calib-apical seed {seed}] m_target={apical_cal[2]:.4g} m_nontarget={apical_cal[3]:.4g} "
              f"center={apical_cal[0]:.4g} scale={apical_cal[1]:.4g} (spread={args.dendritic_logit_spread})", flush=True)

    # -- VERIFY-FIRST: substrate forward gradient reduces held-frame error (guard before the full run) --
    vok, ce0, ce1, ae0, ae1 = _verify_first(seed, ro, s_batch, H, Y, args, gain, head_b)

    # -- LEARN (batched SUBSTRATE forward) : main + anti-cheats. frozen/lesion do NOT run the forward (free); shuffle
    #    runs the substrate forward with deranged targets (reduced budget: it must collapse regardless). --
    hw = ro.head_w                                                              # target head (also the traj wcos ref)
    t0 = time.time()
    reads_before_main = int(s_batch._n_substrate_reads)
    apical_reads_before_main = int(s_batch._n_apical_reads)
    traj = []
    W_main, n_grad, n_mm = _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b, "main",
                                                    traj_eval=(He, Ye, PFe, hw), traj_out=traj)
    main_substrate_reads = int(s_batch._n_substrate_reads) - reads_before_main
    main_apical_reads = int(s_batch._n_apical_reads) - apical_reads_before_main
    learn_secs = round(time.time() - t0, 1)
    W_frozen, _, _ = _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b, "frozen")
    W_lesion, _, _ = _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b, "lesion_err")
    shuf_steps = max(1, int(n_grad * args.shuffle_frac))
    t1 = time.time()
    W_shuffle, n_grad_shuf, _ = _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b,
                                                         "shuffle_teach", max_steps=shuf_steps)
    shuf_secs = round(time.time() - t1, 1)
    # DENDRITIC anti-cheat: FREEZE the apical teacher (silence the one-hot target drive) at the SAME FULL budget as main
    # -> the local error loses its target (only a uniform baseline remains) -> the learned read-out must collapse. Full
    # budget (not reduced) so this isolates "no target" from "few steps": if it does NOT collapse, the lift is an
    # artifact of the sigmoid-error structure, not the dendritic teacher.
    W_freeze_ap = None
    if getattr(args, "dendritic", False):
        W_freeze_ap, _, _ = _learn_substrate_batched(seed, ro, s_batch, H, Y, args, gain, head_b, "main",
                                                     freeze_apical=True)

    # DECISIVE: the learning FORWARD used the substrate read on EVERY main gradient step, 0 host matmul on it. The
    # count is falsifiable — if the forward had used the host-linear proxy, batch_margin would not have been called
    # and main_substrate_reads would be 0.
    host_matmul_on_forward = int(n_mm)
    substrate_reads = int(s_batch._n_substrate_reads)
    forward_read_matches = bool(main_substrate_reads == n_grad and n_grad > 0)

    demo_feats = _demo_feats(ro, seed, sub_tuples, args)

    # -- RULE-RECOVERY on the host-linear read-out (the discriminative, artifact-free channel; anti-cheats collapse
    #    HERE). The forward that LEARNED W was the substrate; this is only the readout used to SCORE recovery. --
    hostlin = _eval_hostlinear(ro, W_main, He, Ye, PFe)
    hl_frozen = _eval_hostlinear(ro, W_frozen, He, Ye, PFe)
    hl_lesion = _eval_hostlinear(ro, W_lesion, He, Ye, PFe)
    hl_shuffle = _eval_hostlinear(ro, W_shuffle, He, Ye, PFe)
    wcos_main = _wcos(W_main, hw)
    wcos_frozen = _wcos(W_frozen, hw); wcos_lesion = _wcos(W_lesion, hw); wcos_shuffle = _wcos(W_shuffle, hw)
    hl_floor_recov = max(hl_frozen["recov_argmax"], hl_lesion["recov_argmax"], hl_shuffle["recov_argmax"])
    wcos_floor = max(abs(wcos_frozen), abs(wcos_lesion), abs(wcos_shuffle))

    # -- INTEGRATION: read the learned W_hat + the copied head ON THE SUBSTRATE (the SAME LearnedReadout / graded read
    #    the prior GO used, P=4, bias-pop on) over the SAME held-out set + identical features -> the integrated ratio.
    #    Each W is read on its OWN FRESH bridge with cp.random reseeded to `seed`: reusing ONE bridge across reads lets
    #    a saturating (large-||W||) readout leave persistent substrate state that CORRUPTS the next read (measured: a
    #    reused bridge dropped copied recov 0.976 -> 0.0004 after a ||W||=40 read); a fresh reseeded bridge isolates
    #    each read AND gives learned/copied/shuffle IDENTICAL OU noise (a fair A/B). feats are bridge-independent for
    #    feature=host (r_h*(Wo_sp@state)), so they are computed once and shared. --
    sub_copied = _fresh_substrate_read(ro, seed, hw.copy(), sub_tuples, Ys, PFs, demo_feats, args)
    sub_learned = _fresh_substrate_read(ro, seed, W_main, sub_tuples, Ys, PFs, demo_feats, args)
    sub_shuffle = _fresh_substrate_read(ro, seed, W_shuffle, sub_tuples, Ys, PFs, demo_feats, args)
    sub_freeze_apical = (_fresh_substrate_read(ro, seed, W_freeze_ap, sub_tuples, Ys, PFs, demo_feats, args)
                         if W_freeze_ap is not None else None)

    chance = 1.0 / ro.V
    ratio = round(sub_learned["recov_argmax"] / max(1e-9, sub_copied["recov_argmax"]), 4)
    integrated_bar = round(0.85 * sub_copied["recov_argmax"], 4)
    m = {
        "seed": seed, "V": ro.V, "D": ro.D, "chance_1_over_v": round(chance, 6),
        "B": args.batch, "n_train_pos": len(H), "n_eval_pos": len(He), "n_sub_demo": len(sub_tuples),
        "n_grad_steps": n_grad, "n_grad_steps_shuffle": n_grad_shuf, "substrate_reads": substrate_reads,
        "main_substrate_reads": main_substrate_reads, "forward_read_matches_grad_steps": forward_read_matches,
        "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay, "w_target": args.w_target,
        "gain": round(gain, 6), "gain_substrate_vs_linear_corr": gain_corr,
        "forward_mode": getattr(args, "forward", "substrate"),
        "eval_every_epochs": getattr(args, "eval_every_epochs", 0), "recovery_trajectory": traj,
        "sub_read_window": args.sub_read_window, "sub_hid_pop": args.sub_hid_pop,
        "sub_pop": args.sub_pop, "w_hat_norm": round(float(np.linalg.norm(W_main)), 2),
        "head_w_norm": round(float(np.linalg.norm(hw)), 2), "learn_secs": learn_secs, "shuffle_secs": shuf_secs,
        # DECISIVE: the learning forward margin IS the substrate read
        "host_matmul_on_learning_forward": host_matmul_on_forward,
        "forward_source": "batched_substrate_graded_conductance_read",
        # VERIFY-FIRST
        "verify_first_ok": bool(vok), "verify_first_ce": [round(ce0, 4), round(ce1, 4)],
        "verify_first_argerr": [round(ae0, 4), round(ae1, 4)],
        # RULE-RECOVERY (host-linear discriminative channel)
        "hostlinear_recov_argmax": round(hostlin["recov_argmax"], 4),
        "hostlinear_argmax_agree": round(hostlin["argmax_agree"], 4),
        "hostlinear_anticheat_recov": {"frozen": round(hl_frozen["recov_argmax"], 4),
                                       "lesion_err": round(hl_lesion["recov_argmax"], 4),
                                       "shuffle_teach": round(hl_shuffle["recov_argmax"], 4)},
        "hostlinear_floor_recov": round(hl_floor_recov, 4),
        "weight_cosine_to_head_diag": wcos_main,
        "weight_cosine_anticheat": {"frozen": wcos_frozen, "lesion_err": wcos_lesion, "shuffle_teach": wcos_shuffle},
        "weight_cosine_floor": round(wcos_floor, 4),
        # INTEGRATION — PRODUCTION substrate demo (bias-pop on, P=4; prior-GO-comparable): learned vs copied
        "sub_learned": sub_learned, "sub_copied": sub_copied, "sub_shuffle": sub_shuffle,
        "sub_recov_ratio_learned_over_copied": ratio, "integrated_bar_0.85xcopied": integrated_bar,
        "host_rng_draws_on_read_path": int(sub_copied.get("host_rng_draws", 0)),
        "no_transport": True, "no_host_grad": True,
    }
    # GO (honest, on the task's bar + the DISCRIMINATIVE anti-cheat channel):
    #  DECISIVE: the learning forward IS the substrate read (0 host matmul, every main step a substrate read).
    m["forward_is_substrate"] = bool(host_matmul_on_forward == 0 and forward_read_matches)
    #  anti-cheats COLLAPSE on the DISCRIMINATIVE host-linear + weight-cosine channel (the substrate argmax metric has
    #  the frequency-tie-break confound, §3 of the qualified GO — a frozen readout scores spuriously high on it, so
    #  sub_shuffle is a reported diagnostic, NOT gated). wcos_main > 0.12 is an ABSOLUTE alignment floor (~30x the
    #  ~0.004 anti-cheat wcos) so the confounded ratio alone can never carry a GO.
    #  the substrate channel is ALSO discriminative with fresh per-W reads (a shuffled teacher collapses to ~0.004),
    #  so require the learned production read to beat the shuffle read too.
    m["anticheats_collapse"] = bool(hostlin["recov_argmax"] > 2.0 * hl_floor_recov
                                    and wcos_main > 3.0 * wcos_floor and wcos_main > 0.12
                                    and sub_learned["recov_argmax"] > 2.0 * sub_shuffle["recov_argmax"])
    #  integrated: substrate learned recov >= 0.85 * copied-head substrate recov (the task GO-gate; ratio >= 0.85).
    m["integrated_go"] = bool(ratio >= 0.85)
    #  parity (reported): does it also MATCH the host-linear-proxy version's QUALITY (proxy: hostlin ~0.93, wcos ~0.51)?
    m["parity_recovery"] = bool(hostlin["recov_argmax"] >= 0.85 and wcos_main >= 0.35)
    if getattr(args, "dendritic", False):
        # ---- DENDRITIC lever verdict (present only in --dendritic): teacher provenance + load-bearing + pre-reg GO ----
        m["dendritic"] = True
        m["main_apical_reads"] = int(main_apical_reads)
        m["apical_reads_match_grad_steps"] = bool(main_apical_reads == n_grad and n_grad > 0)
        m["apical_cal"] = ({"m_target": round(apical_cal[2], 4), "m_nontarget": round(apical_cal[3], 4),
                            "center": round(apical_cal[0], 4), "scale": round(apical_cal[1], 6)}
                           if apical_cal is not None else None)
        m["sub_freeze_apical"] = sub_freeze_apical
        m["sub_freeze_apical_recov"] = (round(sub_freeze_apical["recov_argmax"], 4)
                                        if sub_freeze_apical is not None else None)
        #  the apical TEACHER is load-bearing: freezing it (silence the target drive) collapses the learned read, AND the
        #  forward genuinely ran an apical substrate read every gradient step (provenance).
        m["dendritic_anticheats_ok"] = bool(
            sub_freeze_apical is not None
            and sub_learned["recov_argmax"] > 2.0 * sub_freeze_apical["recov_argmax"]
            and m["apical_reads_match_grad_steps"])
        #  pre-registered dendritic GO: parity (ratio >= 0.85) OR a decisive anti-cheat-clean LIFT over the ~0.37
        #  plateau (sub_learned recov >= 0.55, the WKV-fewspike midpoint), with the two dendritic anti-cheats collapsing.
        m["dendritic_lift_go"] = bool(sub_learned["recov_argmax"] >= 0.55)
        m["go"] = bool((m["integrated_go"] or m["dendritic_lift_go"]) and m["anticheats_collapse"]
                       and m["dendritic_anticheats_ok"] and m["forward_is_substrate"] and vok)
    else:
        #  GO = PRODUCTION integrated bar (the task's, prior-GO-comparable) + anti-cheats + forward-is-substrate + verify.
        m["go"] = bool(m["integrated_go"] and m["anticheats_collapse"] and m["forward_is_substrate"] and vok)
    lever(f"substrate_forward_recov_learned_vs_shuffle_seed{seed}",
          before=sub_shuffle["recov_argmax"], after=sub_learned["recov_argmax"], required=False)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=80000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--w-target", type=float, default=40.0)                  # synaptic-scaling cap on ||W|| (linear range)
    ap.add_argument("--batch", type=int, default=48)                         # B block-diagonal substrate copies
    ap.add_argument("--zero-init", action="store_true")
    ap.add_argument("--n-train-pos", type=int, default=9600)
    ap.add_argument("--n-eval-pos", type=int, default=800)
    ap.add_argument("--n-sub-demo", type=int, default=250)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--verify-steps", type=int, default=8)
    ap.add_argument("--shuffle-frac", type=float, default=0.34)             # shuffle anti-cheat budget (must collapse)
    # substrate operating point (the pipeline / composed seed-42 calibrations; reused, NOT retuned)
    ap.add_argument("--pop", type=int, default=4)                           # DEMO word-pool pop (matches prior GO)
    ap.add_argument("--hid-pop", type=int, default=4)                       # DEMO hidden pop
    ap.add_argument("--sub-pop", type=int, default=1)                       # FORWARD word-pool pop (graded read ~P-indep)
    ap.add_argument("--sub-hid-pop", type=int, default=4)                   # FORWARD hidden pop
    ap.add_argument("--read-window", type=int, default=150)                 # DEMO read window
    ap.add_argument("--sub-read-window", type=int, default=120)             # FORWARD read window
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--bias-scale", type=float, default=0.14)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    # ---- DENDRITIC (Urbanczik-Senn two-compartment) lever: default OFF, byte-identical to the softmax-onehot rule ----
    ap.add_argument("--dendritic", action="store_true",
                    help="learn the read-out via a SECOND target-driven APICAL substrate read: the local error becomes "
                         "err=sigma(apical)-sigma(basal) (PER-UNIT, not a cross-unit softmax over the noisy basal read). "
                         "Off by default and byte-identical to the default rule when off.")
    ap.add_argument("--apical-drive-pA", type=float, default=600.0)         # one-hot target teacher (labelled-line) drive
    ap.add_argument("--apical-baseline-pA", type=float, default=220.0)      # tonic inhibitory baseline (non-target -> low)
    ap.add_argument("--apical-syn-scale", type=float, default=12.0)         # apical synapse scale (magnitude calibrated away)
    ap.add_argument("--dendritic-tau", type=float, default=1.0)             # sigmoid temperature (nats)
    ap.add_argument("--dendritic-logit-spread", type=float, default=4.0)    # apical calibrated to +/- this logit
    ap.add_argument("--n-apical-i", type=int, default=16)                   # tonic inhibitory baseline population size
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--forward", type=str, default="substrate", choices=["substrate", "host_proxy"],
                    help="learning-forward margin: substrate (default, the real test, host_matmul=0) or host_proxy "
                         "(CONTROL: W@h+head_b at the SAME operating point/coverage, forward_is_substrate=False)")
    ap.add_argument("--eval-every-epochs", type=int, default=0,
                    help="if >0, record host-linear recovery every K epochs (cheap; plateau-vs-climbing convergence)")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_readout_eprop_batched_substrate.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_sentences = min(args.n_sentences, 12000)
        args.batch = min(args.batch, 8)
        args.n_train_pos = min(args.n_train_pos, 640)
        args.epochs = min(args.epochs, 2)
        args.sub_read_window = min(args.sub_read_window, 80)
        args.n_sub_demo = min(args.n_sub_demo, 80)
        args.n_eval_pos = min(args.n_eval_pos, 300)
        args.verify_steps = min(args.verify_steps, 6)

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"),
                   note="(batched substrate forward is GPU-bound; numpy only for tiny smoke)")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    t_all = time.time()
    seed_hash_check = None
    for si, seed in enumerate(seeds):
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        if seed_hash_check is None:                                        # CLAUDE.md seed-trap: build-twice hash
            h1 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                           args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
            h2 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                           args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
            seed_hash_check = {"seed": seed, "thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)}
            print(f"[seed-trap] thr hash {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)
        m = run_seed(seed, ro, args)
        m["seed_hash_check"] = seed_hash_check
        results.append(m)
        sl = m["sub_learned"]; sc = m["sub_copied"]; ss = m["sub_shuffle"]; hlac = m["hostlinear_anticheat_recov"]
        print(f"[seed {seed}] SUB learned={sl['recov_argmax']} vs copied {sc['recov_argmax']} "
              f"(r={m['sub_recov_ratio_learned_over_copied']}, bar {m['integrated_bar_0.85xcopied']}) "
              f"shuffle={ss['recov_argmax']} | HOSTLIN={m['hostlinear_recov_argmax']} "
              f"(ac frz/les/shf={hlac['frozen']}/{hlac['lesion_err']}/{hlac['shuffle_teach']}) "
              f"WCOS={m['weight_cosine_to_head_diag']} | fwd_matmul={m['host_matmul_on_learning_forward']} "
              f"reads={m['substrate_reads']} | ac={m['anticheats_collapse']} int={m['integrated_go']} "
              f"parity={m['parity_recovery']} GO={m['go']} ({m['learn_secs']}+{m['shuffle_secs']}s)", flush=True)
        if m.get("dendritic"):
            print(f"    [dendritic seed {seed}] apical_reads={m['main_apical_reads']}/{m['n_grad_steps']} "
                  f"(match={m['apical_reads_match_grad_steps']}) freeze_apical_recov={m.get('sub_freeze_apical_recov')} "
                  f"apical_cal={m.get('apical_cal')} lift_go={m.get('dendritic_lift_go')} "
                  f"dend_ac_ok={m.get('dendritic_anticheats_ok')}", flush=True)
        project_cost("batched-substrate 6-seed", si + 1, len(seeds), time.time() - t_all, warn_hours=10.0)

    rows = [r for r in results if "sub_learned" in r]
    summary = {}
    if rows:
        go_n = int(sum(1 for r in rows if r["go"]))
        undefined_if_empty("eprop_batched_substrate_GO_seeds", len(rows), go_n, len(rows))
        summary = {
            "n_seeds": len(rows), "go_count": go_n, "go_5of6": bool(go_n >= 5),
            "integrated_go_count": int(sum(1 for r in rows if r["integrated_go"])),
            "parity_recovery_count": int(sum(1 for r in rows if r["parity_recovery"])),
            "anticheats_collapse_count": int(sum(1 for r in rows if r["anticheats_collapse"])),
            "forward_is_substrate_all": bool(all(r["forward_is_substrate"] for r in rows)),
            "host_matmul_on_forward_max": int(max(r["host_matmul_on_learning_forward"] for r in rows)),
            "verify_first_all_ok": bool(all(r["verify_first_ok"] for r in rows)),
            "sub_learned_recov_mean": round(float(np.mean([r["sub_learned"]["recov_argmax"] for r in rows])), 4),
            "sub_copied_recov_mean": round(float(np.mean([r["sub_copied"]["recov_argmax"] for r in rows])), 4),
            "sub_shuffle_recov_mean": round(float(np.mean([r["sub_shuffle"]["recov_argmax"] for r in rows])), 4),
            "sub_recov_ratio_mean": round(float(np.mean([r["sub_recov_ratio_learned_over_copied"] for r in rows])), 4),
            "sub_recov_ratio_min": round(float(np.min([r["sub_recov_ratio_learned_over_copied"] for r in rows])), 4),
            "hostlinear_recov_mean": round(float(np.mean([r["hostlinear_recov_argmax"] for r in rows])), 4),
            "hostlinear_floor_recov_max": round(float(np.max([r["hostlinear_floor_recov"] for r in rows])), 4),
            "weight_cosine_mean": round(float(np.mean([r["weight_cosine_to_head_diag"] for r in rows])), 4),
            "weight_cosine_floor_max": round(float(np.max([r["weight_cosine_floor"] for r in rows])), 4),
        }
        if any(r.get("dendritic") for r in rows):
            fa = [r["sub_freeze_apical_recov"] for r in rows if r.get("sub_freeze_apical_recov") is not None]
            summary["dendritic"] = True
            summary["sub_freeze_apical_recov_mean"] = round(float(np.mean(fa)), 4) if fa else None
            summary["apical_reads_match_all"] = bool(all(r.get("apical_reads_match_grad_steps") for r in rows))
            summary["dendritic_anticheats_ok_count"] = int(sum(1 for r in rows if r.get("dendritic_anticheats_ok")))
    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "no_transport": True, "no_host_grad": True,
           "forward_during_learning": "batched_substrate_graded_conductance_read",
           "gain_per_seed": {int(r["seed"]): r["gain"] for r in results if "gain" in r},
           "seed_hash_check": seed_hash_check,
           "backend": os.environ.get("SIM_BACKEND", "numpy"),
           "elapsed_s": round(time.time() - t_all, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows -> {args.json} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
