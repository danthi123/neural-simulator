"""gap#1 / A1 — COMPOSE the mouth's two substrate graded reads END-TO-END, and biologize the base-rate prior head_b.

Two mouth matmuls are now substrate signed graded-conductance reads, but each was validated in ISOLATION (the other's
input taken host-side):
  * OUTPUT PROJECTION `h_pre = Wo_sp @ state`  (`_wkv_graded_output_projection_derisk`, corr 0.984 6/6) — reconstructs
    the hidden feature FROM the WKV state as a signed graded margin, but its DOWNSTREAM read (head_w @ h) was applied
    HOST-side to isolate the projection metric ("the end-to-end chain was not run").
  * READ-OUT `logits = head_w @ h`  (`_wkv_graded_conductance_read_derisk`, recov_argmax 0.921 6/6; closed to 0.978 by
    `_wkv_mouth_read_parity_close_derisk`) — reads the winner word-pool from the signed net-current margin, but takes
    the hidden feature `h = r_h*(Wo_sp@state)` HOST-side (`_hidden_feature`), and injects head_b in HOST ARITHMETIC
    (`margin + s*head_b`; the parity-close finding's residual #2: "wire it as a tonic bias-input population").

THIS LANE (wiring, NO new mechanism) does the two named compositions:
  (A) CHAIN the two reads: the OUTPUT PROJECTION's substrate graded margin (hpre_sub, off cp_conductance_g_e/g_i on the
      projection bridge) — gated by the host r_h (the projection finding's named host residual: shunting is the next
      rung) — becomes the READ-OUT's input feature (drives hid/hidinh on the read-out bridge). So the state->logits
      chain is ONE substrate signed-graded pipeline: every matmul stage is a `cp_conductance_*` read, 0 host matmul on
      the margin. The only host arithmetic left between sensation and the winner is the elementwise r_h gate and the LN
      inside the WKV state (both named upstream residuals, NOT this lane's target).
  (B) BIOLOGIZE head_b: instead of `margin + s*head_b` (host arithmetic), a TONIC BIAS-INPUT POPULATION (matched
      excitatory `bias_e` + inhibitory `bias_i`, driven by a constant current so they fire at a steady tonic rate)
      wires onto the word-pools with per-pool weights proportional to head_b (bias_e carries head_b>0 as EXCITATORY
      g_e, bias_i carries head_b<0 as INHIBITORY g_i, with the SAME driving-force `ratio` the feature read uses). The
      pools' net-current margin (df_e*g_e + df_i*g_i) then PICKS UP the base-rate prior as a genuine synaptic
      conductance — the base rate is a real tonic synaptic current the pools carry (prior-as-starting-point / resting
      excitability, Mulder 2012; a constant resting bias, NOT a per-position host renormalisation). 0 host arithmetic
      on the margin.

THE A/B (4 arms per seed, isolating the composition penalty AND the head_b biologization):
    readout_hostb    : HOST feat + head_b via HOST ARITHMETIC   (== the parity-close deliverable; the 0.978 reference)
    readout_biaspop  : HOST feat + head_b via TONIC BIAS POP    (isolates head_b-as-synapse: should ~ readout_hostb)
    composed_nobias  : SUBSTRATE-projection feat, head_b OFF     (isolates the projection-composition penalty)
    composed_biaspop : SUBSTRATE-projection feat + head_b BIAS POP  (THE deliverable: fully-substrate state->logits)
Headline: does composed_biaspop hold NEAR-parity with the isolated reads (recov_argmax / argmax_agree ~0.92-0.98, not
degraded by composition), with head_b now a spiking synapse? And does readout_biaspop ~ readout_hostb (head_b-as-synapse
== head_b-as-host-arithmetic)?

ANTI-CHEATS (each MUST collapse; brain-based / negatives load-bearing):
  * LESION ANY STAGE -> degrades. zero-STATE (silence the projection INPUT, cache-immune) -> composed chain loses
    structure -> chance. zero-FEATURE (silence the read-out INPUT) -> chance. SILENCE-BIAS (drop the bias-pop drive to
    0 -> no tonic conductance) -> the base-rate lift vanishes back to composed_nobias (the bias POPULATION's synaptic
    conductance is load-bearing, cache-immune).
  * SCRAMBLE (post-hoc pool->word relabel) -> chance.
  * PROVENANCE: winner argmax over the substrate net-current margin off cp_conductance_g_e/g_i; head_b via a spiking
    synapse (0 host arithmetic on the margin for the biaspop arms); host_rng_draws_on_read_path == 0.
  * SIGNED load-bearing: the read-out inhibitory shadow (Wn) must beat positive-only on identical conductances (the
    base-rate term is added to BOTH margins, so this still isolates the SIGN).
  * 6 seeds 42/43/44/100/101/102 (smoke first); a SINGLE fixed operating point (bias_scale calibrated ONCE on seed 42).

HONEST SCOPE: the WKV recurrent STATE (Wv input proj + leaky integrator + BPTT-trained decay) is STILL host; the r_h
gate + LN are host (named upstream residuals); the read-out / projection / head_b weights are host-designed (from the
trained checkpoint). This lane composes the two substrate MATMUL reads end-to-end and moves head_b onto a spiking
synapse. It is NOT "fully spiking" and NOT production-wired. Runner-only, default-off, NO sim/ edit.

Reuse-by-import: GradedOutputProjection (the projection substrate) from `_wkv_graded_output_projection_derisk`;
ParityCloseRead / GradedConductanceLogitRead / SignedShadowLogitRead (the read-out substrate + host-arith head_b) from
`_wkv_mouth_read_parity_close_derisk` / `_wkv_graded_conductance_read_derisk`; WKVReadout + _softmax + _native +
_load_eval from `_wkv_fewspike_read_derisk`. cfg.seed-controlled substrate (CLAUDE.md seed trap).

Run (smoke):   SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_endtoend_substrate_read_derisk \
                 --smoke --seeds 42
Run (calib):   SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_endtoend_substrate_read_derisk \
                 --calib-bias --seeds 42
Run (6-seed):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_endtoend_substrate_read_derisk \
                 --seeds 42,43,44,100,101,102 \
                 --json research/findings/raw/_wkv_endtoend_substrate_read_6seed.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_graded_output_projection_derisk import GradedOutputProjection  # noqa: E402
from research.runners._wkv_mouth_read_parity_close_derisk import ParityCloseRead  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402


# ====================================================================================================================
# The composed read-out: the graded-conductance read-out (+ hid_pop density) + a TONIC BIAS-INPUT POPULATION carrying
# head_b as a spiking synaptic drive. Accepts its input feature EITHER host-side (`_hidden_feature`) or from the
# SUBSTRATE output-projection (the composition). head_b is applied EITHER host-arithmetic (inherited `_apply_baserate`)
# OR through the bias population (this class) — never both.
# ====================================================================================================================
class ComposedEndToEndRead(ParityCloseRead):
    def __init__(self, ro, seed, proj=None, use_proj=False, use_bias_pop=True, hb_k=0.0,
                 bias_scale=1.0, n_bias=16, bias_drive_pA=160.0, proj_out_scale=0.30,
                 pop=4, hid_pop=4, **kw):
        # bias-pop params MUST be set BEFORE super().__init__ (it calls the overridden _build_bridge / _wire).
        self.proj = proj
        self.use_proj = bool(use_proj)
        self.use_bias_pop = bool(use_bias_pop)
        self.bias_scale = float(bias_scale)
        self.n_bias = int(n_bias)
        self.bias_drive_pA = float(bias_drive_pA)
        # the substrate output-projection margin is in arbitrary conductance units (RMS ~3.3x the host feature at the
        # projection GO operating point); a SINGLE scalar calibrated once on seed 42 maps it to the read-out's
        # validated feature scale (a unit-balance calibration, like the reads' `ratio` — NOT a per-channel gain).
        self.proj_out_scale = float(proj_out_scale)
        super().__init__(ro, seed, pop=pop, hid_pop=hid_pop, hb_k=hb_k, **kw)

    # ---- bridge: parent regions (hid, hidinh, wpool, fs) + a matched tonic bias pair (bias_e, bias_i) ----
    def _build_bridge(self):
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
        regions = [
            BrainRegion(name="hid", n_neurons=Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hidinh", n_neurons=Hn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="wpool", n_neurons=self.V * self.P, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="fs", n_neurons=self.n_fs, exc_fraction=0.0, internal_density=0.0),
            BrainRegion(name="bias_e", n_neurons=self.n_bias, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="bias_i", n_neurons=self.n_bias, exc_fraction=1.0, internal_density=0.0),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.hid_idx = np.asarray(list(rm.indices("hid")), dtype=np.int64)
        self.hidinh_idx = np.asarray(list(rm.indices("hidinh")), dtype=np.int64)
        self.hid_dim = np.repeat(np.arange(self.F), self.Hp).astype(np.int64)
        wpool_idx = np.asarray(list(rm.indices("wpool")), dtype=np.int64)
        self.pool_idx = [wpool_idx[k * self.P:(k + 1) * self.P] for k in range(self.V)]
        self.all_pool = wpool_idx
        self.fs_idx = np.asarray(list(rm.indices("fs")), dtype=np.int64)
        self.bias_e_idx = np.asarray(list(rm.indices("bias_e")), dtype=np.int64)
        self.bias_i_idx = np.asarray(list(rm.indices("bias_i")), dtype=np.int64)
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())

    # ---- wiring: parent readout (Wp/Wn) + FS-WTA + the bias-pop -> pools (head_b as signed synaptic conductance) ----
    def _wire(self):
        b = self._b
        union = {}
        Wp = (self.Wp * self.syn_scale).astype(np.float32)
        Wn = (self.Wn * self.syn_scale * self.ratio).astype(np.float32)
        Wp_hn = Wp[:, self.hid_dim]
        Wn_hn = Wn[:, self.hid_dim]
        nH = len(self.hid_idx)
        pre = np.tile(self.hid_idx, self.V * self.P)
        post = np.repeat(self.all_pool, nH)
        wp = np.repeat(Wp_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_pos"] = {"pre_indices": pre, "post_indices": post, "initial_weights": wp,
                                "plastic": False, "conn_type": "E_TO_E"}
        pre_n = np.tile(self.hidinh_idx, self.V * self.P)
        wn = np.repeat(Wn_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_neg"] = {"pre_indices": pre_n, "post_indices": post.copy(), "initial_weights": wn,
                                "plastic": False, "conn_type": "I_TO_E"}
        # FS-WTA (inert in the graded read path but kept identical to the parent substrate).
        pef = np.repeat(self.all_pool, len(self.fs_idx)); qef = np.tile(self.fs_idx, len(self.all_pool))
        union["pool2fs"] = {"pre_indices": pef, "post_indices": qef,
                            "initial_weights": np.full(len(pef), self.exc_to_fs, np.float32),
                            "plastic": False, "conn_type": "E_TO_E"}
        pfe = np.repeat(self.fs_idx, len(self.all_pool)); qfe = np.tile(self.all_pool, len(self.fs_idx))
        union["fs2pool"] = {"pre_indices": pfe, "post_indices": qfe,
                            "initial_weights": np.full(len(pfe), self.fs_to_exc, np.float32),
                            "plastic": False, "conn_type": "I_TO_E"}
        # ---- the TONIC BIAS-INPUT POPULATION: head_b as a signed synaptic conductance onto the word-pools ----
        # head_b centred (only the RELATIVE base rate matters for argmax); <unk> pinned low so the tonic bias never
        # re-lifts the suppressed unk. bias_e carries head_b>0 (EXCITATORY g_e), bias_i carries head_b<0 (INHIBITORY
        # g_i, with the SAME driving-force ratio the feature read uses) -> net bias current ~ bias_scale * head_b.
        hb = self.head_b.astype(np.float64).copy()
        if self.ro.unk_idx >= 0:
            hb[self.ro.unk_idx] = hb.min()
        hb = hb - hb.mean()
        hb_pos = np.maximum(hb, 0.0)                                             # [V] excitatory half
        hb_neg = np.maximum(-hb, 0.0)                                            # [V] inhibitory half
        nB = len(self.bias_e_idx)
        pre_bp = np.tile(self.bias_e_idx, self.V * self.P)
        post_b = np.repeat(self.all_pool, nB)
        wbp = (np.repeat(np.repeat(hb_pos, self.P), nB).astype(np.float32)
               * (self.syn_scale * self.bias_scale))
        union["bias_pos"] = {"pre_indices": pre_bp, "post_indices": post_b, "initial_weights": wbp,
                             "plastic": False, "conn_type": "E_TO_E"}
        pre_bn = np.tile(self.bias_i_idx, self.V * self.P)
        wbn = (np.repeat(np.repeat(hb_neg, self.P), nB).astype(np.float32)
               * (self.syn_scale * self.ratio * self.bias_scale))
        union["bias_neg"] = {"pre_indices": pre_bn, "post_indices": post_b.copy(), "initial_weights": wbn,
                             "plastic": False, "conn_type": "I_TO_E"}
        inh = np.concatenate([self.hidinh_idx, self.fs_idx, self.bias_i_idx]).tolist()
        b.inject_explicit_wiring(union, output_inhibitory_indices=inh)
        self._pos_edges = (union["readout_pos"]["pre_indices"], union["readout_pos"]["post_indices"], wp.copy())
        self._neg_edges = (union["readout_neg"]["pre_indices"], union["readout_neg"]["post_indices"], wn.copy())

    # ---- graded margin read (parent) + drive the tonic bias pop so its conductance enters the pools' margin ----
    def _graded_margin(self, feat, want_diag=False, silence_bias=False):
        b = self._b
        xp, _ = get_backend()
        self._reset()
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        fdrive = self.hid_bias + self.hid_gain * feat[self.hid_dim]
        drive[self.hid_idx] = fdrive
        drive[self.hidinh_idx] = fdrive                                         # SAME drive -> rate-matched pair
        if self.use_bias_pop and not silence_bias:
            drive[self.bias_e_idx] = self.bias_drive_pA                         # constant tonic drive -> steady rate
            drive[self.bias_i_idx] = self.bias_drive_pA
        if self.floor_pA:
            drive[self.all_pool] += self.floor_pA
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        n_acc = 0
        ge_sum = np.zeros(self.V); gi_sum = np.zeros(self.V)
        pool_sp = 0.0; bias_sp = 0.0
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            ge = np.asarray(to_host(b.cp_conductance_g_e)).astype(np.float64)[self.all_pool].reshape(self.V, self.P)
            gi = np.asarray(to_host(b.cp_conductance_g_i)).astype(np.float64)[self.all_pool].reshape(self.V, self.P)
            ge_sum += ge.sum(axis=1)
            gi_sum += gi.sum(axis=1)
            if want_diag:
                fs = np.asarray(to_host(b.cp_firing_states)).astype(float)
                pool_sp += float(fs[self.all_pool].sum())
                bias_sp += float(fs[self.bias_e_idx].sum() + fs[self.bias_i_idx].sum())
            n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        ge_mean = ge_sum / n_acc; gi_mean = gi_sum / n_acc
        margin = self.df_e * ge_mean + self.df_i * gi_mean                      # [V] CONTINUOUS (incl. bias-pop g)
        if want_diag:
            return margin, ge_mean, gi_mean, pool_sp / n_acc, bias_sp / n_acc
        return margin

    # ---- the end-to-end read: substrate projection feat (composed) OR host feat; head_b via bias pop OR host arith ----
    def _feature(self, ap, an, tid, zero_state=False, zero_feat=False):
        """Return the dual-nonneg drive [h+, h-]. use_proj -> h from the SUBSTRATE output projection (composition);
        else the host _hidden_feature. zero_state silences the projection INPUT (composed cache-immune control)."""
        ro = self.ro
        if self.use_proj:
            state = np.concatenate([ap, an])
            hpre_sub, _ = self.proj._graded_hpre(state, zero_state=zero_state)  # SUBSTRATE Wo_sp@state (len D)
            r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))          # host r_h gate (named residual)
            h = r_h * (self.proj_out_scale * hpre_sub)                          # [D] (units -> read-out feature scale)
            if zero_feat:
                h = np.zeros_like(h)
            return np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)])
        feat = self._hidden_feature(ap, an, tid)                               # host h = r_h*(Wo_sp@state)
        if zero_feat or zero_state:
            feat = np.zeros_like(feat)
        return feat

    def read_endtoend(self, ap, an, tid, scramble_perm=None, zero_state=False, zero_feat=False,
                      silence_bias=False):
        feat = self._feature(ap, an, tid, zero_state=zero_state, zero_feat=zero_feat)
        margin, ge, gi, psp, bsp = self._graded_margin(feat, want_diag=True, silence_bias=silence_bias)
        margin_pos = self.df_e * ge                                            # positive-only (excitatory g_e; incl bias_e)
        if (not self.use_bias_pop) and self.hb_k > 0.0:                        # host-arithmetic head_b (readout_hostb)
            margin = self._apply_baserate(margin)
            margin_pos = self._apply_baserate(margin_pos)
        if scramble_perm is not None:
            margin = margin[scramble_perm]; margin_pos = margin_pos[scramble_perm]
        return dict(win=self._argwin(margin), margin=margin, win_pos=self._argwin(margin_pos),
                    margin_pos=margin_pos, pool_sp=psp, bias_sp=bsp)


# ====================================================================================================================
# Eval — teacher-forced over held-out positions; host reference = the deployed mouth read head_w@(r_h*(Wo_sp@state))+head_b
# ====================================================================================================================
def _eval(seed, ro, ev_ids, vocab, s, warmup, topk, sample_temp, n_eval_pos, oracle_every=3,
          want_bias_silence=False):
    grng = np.random.default_rng(seed * 137 + 11)
    acc = dict(n=0, argmax_agree=0.0, argmax_agree_pos=0.0, top5_hit=0.0, nll=0.0,
               mass_read=0.0, mass_hs=0.0, mass_ax=0.0, mass_ora=0.0, ora_n=0,
               silent=0, hid_active=0.0, pool_sp=0.0, bias_sp=0.0, agree_scr=0.0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()               # DEPLOYED mouth read (incl head_b)
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            candk = np.argpartition(-lg_supp, topk - 1)[:topk]; candk = candk[np.argsort(-lg_supp[candk])]
            pk = _softmax(lg_supp[candk] / sample_temp)
            hs = int(candk[int(grng.choice(len(pk), p=pk))])
            r = s.read_endtoend(ap, an, ids[t])
            win, margin, win_pos = r["win"], r["margin"], r["win_pos"]
            acc["pool_sp"] += r["pool_sp"]; acc["bias_sp"] += r["bias_sp"]
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s.V)
            win_s = int(np.argmax(margin[scr_perm])) if float(margin.max() - margin.min()) > 1e-9 else -1
            if positions % oracle_every == 0:
                ora = s.read_oracle(lg_supp)
                acc["mass_ora"] += (pfull[ora] if ora >= 0 else 0.0); acc["ora_n"] += 1
            acc["n"] += 1; positions += 1
            acc["hid_active"] += float(float(margin.max() - margin.min()) > 1e-9)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["argmax_agree_pos"] += float(win_pos == host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["nll"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
            acc["mass_read"] += (pfull[win] if win >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["agree_scr"] += float(win_s == host_argmax)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(acc["n"] == 0, "no evaluable positions (every eval sentence shorter than warmup+2) — metrics undefined")
    n = max(1, acc["n"])

    # ---- cache-immune collapse controls ----
    def _collapse(kind):
        ag = 0; nn = 0
        for ids in ev_ids[:4]:
            if len(ids) < warmup + 2:
                continue
            ap2 = np.zeros(ro.D); an2 = np.zeros(ro.D)
            for t in range(min(len(ids) - 1, warmup + 30)):
                ap2, an2 = ro.advance(ap2, an2, ids[t])
                if t < warmup:
                    continue
                lg2 = ro.logits(ap2, an2, ids[t]); lg2s = lg2.copy()
                if ro.unk_idx >= 0:
                    lg2s[ro.unk_idx] = -1e30
                ham = int(np.argmax(lg2s))
                kw = {kind: True}
                win = s.read_endtoend(ap2, an2, ids[t], **kw)["win"]
                ag += int(win == ham); nn += 1
                if nn >= 60:
                    break
            if nn >= 60:
                break
        return ag / max(1, nn)

    # zero the read-out INPUT feature; for the composed arms also test zero-STATE (silences the projection INPUT).
    agree_zerofeat = _collapse("zero_feat")
    agree_zerostate = _collapse("zero_state") if s.use_proj else agree_zerofeat

    # ---- SILENCE-BIAS: drop the tonic bias-pop drive to 0 -> the base-rate lift must vanish (bias-pop load-bearing) ----
    recov_biassilence = None
    if want_bias_silence and s.use_bias_pop:
        ms = 0.0; nn = 0
        for ids in ev_ids:
            if len(ids) < warmup + 2:
                continue
            ap2 = np.zeros(ro.D); an2 = np.zeros(ro.D)
            for t in range(len(ids) - 1):
                ap2, an2 = ro.advance(ap2, an2, ids[t])
                if t < warmup:
                    continue
                lg2 = ro.logits(ap2, an2, ids[t]); lg2s = lg2.copy()
                if ro.unk_idx >= 0:
                    lg2s[ro.unk_idx] = -1e30
                pf = _softmax(lg2s)
                win = s.read_endtoend(ap2, an2, ids[t], silence_bias=True)["win"]
                ms += (pf[win] if win >= 0 else 0.0); nn += 1
                if nn >= min(n, 120):
                    break
            if nn >= min(n, 120):
                break
        mass_ax = acc["mass_ax"] / n
        recov_biassilence = round((ms / max(1, nn)) / max(1e-9, mass_ax), 4)

    lever(f"{s._arm}_zero_feature_collapse", before=round(acc["argmax_agree"] / n, 4),
          after=round(agree_zerofeat, 4), required=False)
    if s.use_proj:
        lever(f"{s._arm}_zero_state_collapse", before=round(acc["argmax_agree"] / n, 4),
              after=round(agree_zerostate, 4), required=False)
    lever(f"{s._arm}_signed_vs_positive_argmax", before=round(acc["argmax_agree_pos"] / n, 4),
          after=round(acc["argmax_agree"] / n, 4), required=False)

    m = {
        "seed": seed, "arm": s._arm, "V": s.V, "pop": s.P, "hid_pop": s.Hp, "ratio": s.ratio,
        "use_proj": s.use_proj, "use_bias_pop": s.use_bias_pop, "hb_k": s.hb_k, "bias_scale": s.bias_scale,
        "topk_ceiling": topk, "plasticity_off": True,
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "hidden_active_frac": round(acc["hid_active"] / n, 4),
        "mean_pool_spikes": round(acc["pool_sp"] / n, 3),
        "mean_bias_spikes": round(acc["bias_sp"] / n, 3),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "argmax_agree_positive_only": round(acc["argmax_agree_pos"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_read": round(acc["nll"] / n, 4),
        "mass_read": round(acc["mass_read"] / n, 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "argmax_agree_zerofeat": round(agree_zerofeat, 4),
        "argmax_agree_zerostate": round(agree_zerostate, 4),
        "mass_oracle_ceiling": round(acc["mass_ora"] / max(1, acc["ora_n"]), 4),
        "chance_1_over_v": round(1.0 / s.V, 6),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_read"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["recov_argmax"] = round(m["mass_read"] / max(1e-9, m["mass_argmax_ceiling"]), 4)
    if recov_biassilence is not None:
        m["recov_argmax_biassilenced"] = recov_biassilence
    return m


def _scramble_at_chance(agree_scramble, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict(m, ref_recov):
    """GO if the composed / biaspop read holds NEAR the isolated-read reference (recov within tol) AND every
    brain-based anti-cheat collapses. ref_recov is the readout_hostb (parity-close-equivalent) reference for the
    composed arms; for the readout arms it is the parity-close target 0.978."""
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        # near-parity: recovers most of the perfect-argmax mass (>=0.85; the isolated graded read's own GO bar).
        "recov_argmax_ge_0.85": m["recov_argmax"] >= 0.85,
        # not degraded far below the isolated reference (composition penalty bounded).
        "recov_within_tol_of_ref": m["recov_argmax"] >= ref_recov - 0.06,
        # the signed inhibitory shadow stays load-bearing on identical conductances.
        "signed_beats_positive_only": m["argmax_agree"] > m["argmax_agree_positive_only"],
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        # cache-immune: silencing the read INPUT drops to <=1/3 of intact (feature/state drives the read).
        "input_collapses": max(m["argmax_agree_zerofeat"], m["argmax_agree_zerostate"]) <= 0.34 * m["argmax_agree"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
        "not_silent": m["silent_frac"] < 0.05,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


# ARM presets: (use_proj, use_bias_pop, hb_k, hid_pop). hb_k>0 only for the HOST-ARITHMETIC head_b arm.
ARMS = {
    "readout_hostb":    dict(use_proj=False, use_bias_pop=False, hb_k=0.5, hid_pop=4),
    "readout_biaspop":  dict(use_proj=False, use_bias_pop=True,  hb_k=0.0, hid_pop=4),
    "composed_nobias":  dict(use_proj=True,  use_bias_pop=False, hb_k=0.0, hid_pop=4),
    "composed_biaspop": dict(use_proj=True,  use_bias_pop=True,  hb_k=0.0, hid_pop=4),
}


def _build_proj(ro, seed, args):
    return GradedOutputProjection(ro, seed, pop=1, carrier_pop=1, ou_std=args.ou_std,
                                  read_window=args.read_window, drive_gain=args.proj_drive_gain,
                                  syn_scale=args.proj_syn_scale, ratio=args.proj_ratio,
                                  settle_frac=args.settle_frac)


def _build_read(ro, seed, arm, args, proj):
    cfg = ARMS[arm]
    return ComposedEndToEndRead(
        ro, seed, proj=(proj if cfg["use_proj"] else None), use_proj=cfg["use_proj"],
        use_bias_pop=cfg["use_bias_pop"], hb_k=(args.hb_k if cfg["hb_k"] > 0 else 0.0),
        bias_scale=args.bias_scale, n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA,
        proj_out_scale=args.proj_out_scale, pop=args.pop, hid_pop=cfg["hid_pop"], ou_std=args.ou_std,
        read_window=args.read_window, hid_gain=args.hid_gain, ratio=args.ratio)


def _calibrate_bias(ro, seed, ev_ids, vocab, args):
    """Calibrate bias_scale ONCE on the given seed by maximizing composed_biaspop recov_argmax; print the plateau."""
    proj = _build_proj(ro, seed, args)
    print("[calib-bias] bias_scale sweep (composed_biaspop):", flush=True)
    best = None
    for bs in [float(x) for x in args.calib_bias_scales.split(",")]:
        args.bias_scale = bs
        s = _build_read(ro, seed, "composed_biaspop", args, proj)
        s._arm = "composed_biaspop"
        m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                  min(args.n_eval_pos, 80), oracle_every=args.oracle_every)
        print(f"    bias_scale={bs:6.2f} -> recov_argmax={m['recov_argmax']:.4f} "
              f"argmax_agree={m['argmax_agree']:.4f} pool_spk={m['mean_pool_spikes']:.2f} "
              f"bias_spk={m['mean_bias_spikes']:.2f}", flush=True)
        if best is None or m["recov_argmax"] > best[1]:
            best = (bs, m["recov_argmax"])
    print(f"[calib-bias] BEST recov_argmax={best[1]:.4f} at bias_scale={best[0]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--arms", type=str, default="readout_hostb,readout_biaspop,composed_nobias,composed_biaspop")
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    # ---- read-out operating point (the graded-read GO / parity-close calibrated values) ----
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--hb-k", type=float, default=0.5)                          # host-arith head_b coeff (readout_hostb)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    # ---- output-projection operating point (the projection GO calibrated values) ----
    ap.add_argument("--proj-drive-gain", type=float, default=120.0)
    ap.add_argument("--proj-syn-scale", type=float, default=12.0)
    ap.add_argument("--proj-ratio", type=float, default=0.5)
    ap.add_argument("--proj-out-scale", type=float, default=0.30)               # unit-map sub margin -> feature scale (seed42)
    # ---- tonic bias-input population (head_b as a spiking synapse) ----
    ap.add_argument("--bias-scale", type=float, default=0.14)                   # calibrated ONCE on seed 42 (plateau 0.1-0.3)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--oracle-every", type=int, default=3)
    ap.add_argument("--calib-bias", action="store_true")
    ap.add_argument("--calib-bias-scales", type=str, default="0.06,0.1,0.14,0.18,0.25,0.35,0.5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_endtoend_substrate_read.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 80)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arm_names = [a for a in args.arms.split(",") if a.strip()]

    if args.calib_bias:
        seed = seeds[0]
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        _calibrate_bias(ro, seed, ev_ids, vocab, args)
        return

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        proj = _build_proj(ro, seed, args) if any(ARMS[a]["use_proj"] for a in arm_names) else None
        ref_recov = {}
        for arm in arm_names:
            s = _build_read(ro, seed, arm, args, proj)
            s._arm = arm
            want_bs = (arm == "composed_biaspop")
            m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                      args.n_eval_pos, oracle_every=args.oracle_every, want_bias_silence=want_bs)
            # reference for the composed arms = readout_hostb (the parity-close-equivalent isolated read).
            if arm == "readout_hostb":
                ref_recov["readout"] = m["recov_argmax"]
            ref = ref_recov.get("readout", 0.978)
            go, checks = _verdict(m, ref if arm.startswith("composed") else 0.978)
            m["ref_recov"] = ref; m["go"] = go; m["checks"] = checks
            results.append(m)
            extra = (f" recov_biassilence={m.get('recov_argmax_biassilenced')}" if "recov_argmax_biassilenced" in m
                     else "")
            print(f"[seed {seed} {arm:>16s}] recov_argmax={m['recov_argmax']} read_fid={m['read_fidelity_vs_sampler']} "
                  f"agree={m['argmax_agree']}>pos{m['argmax_agree_positive_only']} "
                  f"pool_spk={m['mean_pool_spikes']} bias_spk={m['mean_bias_spikes']} "
                  f"scr={m['argmax_agree_scramble']} zfeat={m['argmax_agree_zerofeat']} "
                  f"zstate={m['argmax_agree_zerostate']}{extra} silent={m['silent_frac']} "
                  f"GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
            if not go:
                print(f"    checks: {json.dumps(checks)}", flush=True)

    summary = {}
    for arm in arm_names:
        rows = [m for m in results if m["arm"] == arm]
        if not rows:
            continue
        summary[arm] = {
            "n_seeds": len(rows),
            "recov_argmax_mean": round(float(np.mean([r["recov_argmax"] for r in rows])), 4),
            "recov_argmax_min": round(float(np.min([r["recov_argmax"] for r in rows])), 4),
            "read_fidelity_mean": round(float(np.mean([r["read_fidelity_vs_sampler"] for r in rows])), 4),
            "argmax_agree_mean": round(float(np.mean([r["argmax_agree"] for r in rows])), 4),
            "argmax_agree_min": round(float(np.min([r["argmax_agree"] for r in rows])), 4),
            "silent_frac_mean": round(float(np.mean([r["silent_frac"] for r in rows])), 4),
            "mean_pool_spikes": round(float(np.mean([r["mean_pool_spikes"] for r in rows])), 3),
            "mean_bias_spikes": round(float(np.mean([r["mean_bias_spikes"] for r in rows])), 3),
            "signed_load_bearing_count": int(sum(1 for r in rows
                                                 if r["argmax_agree"] > r["argmax_agree_positive_only"])),
            "go_count": int(sum(1 for r in rows if r["go"])),
        }
        if any("recov_argmax_biassilenced" in r for r in rows):
            summary[arm]["recov_argmax_biassilenced_mean"] = round(
                float(np.mean([r["recov_argmax_biassilenced"] for r in rows if "recov_argmax_biassilenced" in r])), 4)
    out = {"results": results, "summary": summary, "seeds": seeds, "arms": arm_names, "pop": args.pop,
           "bias_scale": args.bias_scale, "hb_k": args.hb_k, "read_window": args.read_window,
           "ratio": args.ratio, "proj_ratio": args.proj_ratio, "plasticity_off": True,
           "elapsed_s": round(time.time() - t0, 1), "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
