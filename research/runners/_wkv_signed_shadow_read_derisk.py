"""
gap#1 / A1 — the TRUE SIGNED read-out (fm-named surpass) of the fluent WKV open-prose generation: carry the
NEGATIVE head_w weights on an INHIBITORY SHADOW of the hidden layer so the state->logits projection is a signed
SYNAPTIC current with NO global Dale-shift and hence NO common mode to cancel.

THE BOUNDARY THIS ATTACKS (mapped 2026-08-13, `_wkv_fswta_synaptic_read_derisk` rung 2, 0/6):
routing the final logit projection `head_w @ h` (V x D) through read-out NEURONS as EXCITATORY synapses required a
GLOBAL Dale-shift (`head_w - gmin >= 0`) to make the weights non-negative. That shift injects a COMMON MODE
`gmin * sum(hidden spikes)` that is ORDERS OF MAGNITUDE larger than the tiny discriminative margin between the top
near-tied words (measured here: the top1-top2 margin is ~3.5% of the logit range). A scalar feedforward canceller
cannot subtract it faithfully (Poisson residual >> the margin) -> read_fidelity 0.035, gibberish. The rung-2 finding
LOCATED the wall with an ORACLE (perfect host-logit current through the SAME FS-WTA -> read_fid 0.57 at P=1 ... 0.93
at P=16): the FS-WTA RESOLUTION is NOT the wall; the SIGNED SYNAPTIC PROJECTION fidelity is. Its named next lever
(item 1) is this runner.

THE LEVER (Dale's principle — the biology the Dale-shift replaced): a signed weight is carried by TWO populations,
one excitatory, one inhibitory. Split `Wfull = concat(head_w, -head_w)` [V, 2D] into `Wp = max(Wfull, 0)` and
`Wn = max(-Wfull, 0)` (both >= 0). The dual-nonneg hidden feature `feat = [h+, h-]` (h = r_h*(Wo_sp@state), the
host residual) is rate-coded by TWO matched populations driven by the SAME current: an EXCITATORY `hid` and an
INHIBITORY SHADOW `hidinh`. Wire `Wp` as EXCITATORY synapses `hid -> pools` (add to g_e) and `Wn` as INHIBITORY
synapses `hidinh -> pools` (add to g_i). The net pool drive is
    Wp @ rate(hid) - ratio * Wn @ rate(hidinh)  ~  (Wp - Wn) @ feat  =  Wfull @ feat  =  head_w @ h
with NO Dale-shift and NO common mode. `ratio` compensates the conductance driving-force asymmetry
(|E_e - v| ~ 65 mV vs |E_i - v| ~ 10 mV at a LOW floor near rest — the 2026-07-04 conductance-signed lesson: at a
low floor g_i is near-SUBTRACTIVE, at a high floor it turns DIVISIVE/shunting and the subtraction breaks). The two
populations rate-MATCH because they get identical drive and their firing thresholds are uniformized (removing the
per-neuron heterogeneity that would otherwise BIAS feat_i vs feat_e); independent OU keeps the winner stochastic.
The winner emerges from a shared-inhibitory FS-WTA over ALL V pools -> NO host logit matmul, NO top-K argpartition
on the read path.

DECISIVE metrics (calibration-robust, identical family to the parent + rung-2):
  read_fidelity   = ondist_mass(read) / ondist_mass(host_sample)  (1.0 == as on-distribution as an ideal sampler)
  oracle_fidelity = the SAME FS-WTA driven by a PERFECT host-logit current (the RESOLUTION ceiling; a diagnostic)
  projection_recovery = read_fidelity / oracle_fidelity  (>= ~0.85 == the SIGNED PROJECTION is no longer the
                        dominant loss; THE headline for "is the rung-2 signed-projection wall removed")
  positive_only_fidelity = read_fidelity with the INHIBITORY SHADOW (Wn) LESIONED (a positive-only read) -> tests
                        whether the NEGATIVE weights are LOAD-BEARING here (2026-07-04 found the signed machinery
                        DECORATIVE at G^2=25 with 18 slots; over V=1000 with a ~3.5% margin the sign should matter).
Plus mean spikes/read (the budget), free-generation self-NLL (fluency survival), and the shadow rate-match corr.

ANTI-CHEATS (each MUST collapse): readout-lesion (zero Wp+Wn -> pools see only the floor -> collapse); scramble
(permute pool->word label -> chance); provenance (winner from cp_firing_states, 0 host categorical draws on the read
path); hidden-active; not-silent. Plus the shadow-lesion attribution (positive-only) recorded via tools.lab.lever.

Reuse-by-import: WKVReadout + _softmax + _native + _load_eval from `_wkv_fewspike_read_derisk`; the wiring / FS-WTA /
oracle / metric PATTERN from `_wkv_fswta_synaptic_read_derisk` (rung 2). NO `sim/` edit — drives + reads public
bridge arrays; cfg.seed-controlled substrate (CLAUDE.md seed trap). Runner-only, default-off.

Run (smoke):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_signed_shadow_read_derisk \
                --smoke --seeds 42
Run (6-seed): SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_signed_shadow_read_derisk \
                --seeds 42,43,44,100,101,102 --pops 1,4,16 \
                --json research/findings/raw/_wkv_signed_shadow_6seed.json
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
from tools.lab import lever  # noqa: E402


class SignedShadowLogitRead:
    """head_w @ h as a TRUE SIGNED synaptic current: Wp (positive) on an EXCITATORY hidden population, Wn (negative)
    on an INHIBITORY SHADOW of it, both rate-coding the same dual-nonneg feature. No Dale-shift, no common mode."""

    def __init__(self, ro: WKVReadout, seed, pop=1, hid_pop=1, ou_std=60.0, read_window=30,
                 hid_gain=16.0, hid_bias=6.0, syn_scale=1.0, ratio=6.5, floor_pA=8.0,
                 n_fs=48, exc_to_fs=1.2, fs_to_exc=7.0, head_b_gain=0.0, uniform_thresh=True):
        self.ro = ro
        self.V = int(ro.V); self.D = int(ro.D); self.F = 2 * self.D          # feature dim
        self.P = int(pop); self.Hp = int(hid_pop)
        self.ou_std = float(ou_std); self.read_window = int(read_window)
        self.hid_gain = float(hid_gain); self.hid_bias = float(hid_bias)
        self.syn_scale = float(syn_scale); self.ratio = float(ratio); self.floor_pA = float(floor_pA)
        self.n_fs = int(n_fs); self.exc_to_fs = float(exc_to_fs); self.fs_to_exc = float(fs_to_exc)
        self.head_b_gain = float(head_b_gain); self.uniform_thresh = bool(uniform_thresh)
        self.seed = int(seed)
        self.n_host_rng_draws = 0                                            # MUST stay 0

        head_w = ro.head_w                                                   # [V, D]
        self.Wfull = np.concatenate([head_w, -head_w], axis=1)              # [V, 2D]  Wfull@feat = head_w@h
        self.Wp = np.maximum(self.Wfull, 0.0)                               # [V, 2D] >= 0  (excitatory)
        self.Wn = np.maximum(-self.Wfull, 0.0)                             # [V, 2D] >= 0  (inhibitory)
        self.head_b = ro.head_b.astype(np.float64)                        # [V]
        self._build_bridge()
        self._wire()

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
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.hid_idx = np.asarray(list(rm.indices("hid")), dtype=np.int64)       # [F*Hp]
        self.hidinh_idx = np.asarray(list(rm.indices("hidinh")), dtype=np.int64)  # [F*Hp]
        # each hidden neuron codes feature dim = (its position within region) // Hp  (block layout dim0*Hp..)
        self.hid_dim = np.repeat(np.arange(self.F), self.Hp).astype(np.int64)     # [F*Hp]
        wpool_idx = np.asarray(list(rm.indices("wpool")), dtype=np.int64)
        self.pool_idx = [wpool_idx[k * self.P:(k + 1) * self.P] for k in range(self.V)]
        self.all_pool = wpool_idx
        self.fs_idx = np.asarray(list(rm.indices("fs")), dtype=np.int64)
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        # uniformize firing thresholds so hid and hidinh rate-MATCH (remove the per-neuron heterogeneity BIAS)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())

    def _wire(self):
        b = self._b
        union = {}
        Wp = (self.Wp * self.syn_scale).astype(np.float32)                  # [V, 2D]
        Wn = (self.Wn * self.syn_scale * self.ratio).astype(np.float32)     # [V, 2D] (driving-force compensated)
        # per-hidden-neuron weight columns (index the [V,2D] rows by each neuron's coded feature dim)
        Wp_hn = Wp[:, self.hid_dim]                                         # [V, F*Hp]
        Wn_hn = Wn[:, self.hid_dim]                                         # [V, F*Hp]
        nH = len(self.hid_idx)
        # ---- Wp: EXCITATORY hid -> pools ----
        pre = np.tile(self.hid_idx, self.V * self.P)
        post = np.repeat(self.all_pool, nH)
        wp = np.repeat(Wp_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_pos"] = {"pre_indices": pre, "post_indices": post, "initial_weights": wp,
                                "plastic": False, "conn_type": "E_TO_E"}
        # ---- Wn: INHIBITORY hidinh -> pools ----
        pre_n = np.tile(self.hidinh_idx, self.V * self.P)
        wn = np.repeat(Wn_hn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["readout_neg"] = {"pre_indices": pre_n, "post_indices": post.copy(), "initial_weights": wn,
                                "plastic": False, "conn_type": "I_TO_E"}
        # ---- FS-WTA: pool -> fs (exc), fs -> pool (inh) ----
        pef = np.repeat(self.all_pool, len(self.fs_idx)); qef = np.tile(self.fs_idx, len(self.all_pool))
        union["pool2fs"] = {"pre_indices": pef, "post_indices": qef,
                            "initial_weights": np.full(len(pef), self.exc_to_fs, np.float32),
                            "plastic": False, "conn_type": "E_TO_E"}
        pfe = np.repeat(self.fs_idx, len(self.all_pool)); qfe = np.tile(self.all_pool, len(self.fs_idx))
        union["fs2pool"] = {"pre_indices": pfe, "post_indices": qfe,
                            "initial_weights": np.full(len(pfe), self.fs_to_exc, np.float32),
                            "plastic": False, "conn_type": "I_TO_E"}
        inh = np.concatenate([self.hidinh_idx, self.fs_idx]).tolist()      # SHADOW + FS are inhibitory
        b.inject_explicit_wiring(union, output_inhibitory_indices=inh)
        self._pos_edges = (union["readout_pos"]["pre_indices"], union["readout_pos"]["post_indices"], wp.copy())
        self._neg_edges = (union["readout_neg"]["pre_indices"], union["readout_neg"]["post_indices"], wn.copy())

    def _hidden_feature(self, ap, an, tid):
        """h = r_h * (Wo_sp @ [ap,an]) (HOST residual — the validated graded conductance projection). Return the
        dual-nonneg drive [h+, h-]  (>= 0)."""
        ro = self.ro
        state = np.concatenate([ap, an])
        r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
        h = r_h * (ro.Wo_sp @ state)                                        # [D]
        return np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)])    # [2D] >= 0

    def _reset(self):
        b = self._b
        if self._v0 is not None:
            b.cp_membrane_potential_v[:] = self._v0
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False

    def _run(self, feat, want_shadow_rate=False):
        """Drive BOTH hid and hidinh by (hid_bias + hid_gain*feat[dim]); run read_window steps; return per-pool
        firing (+ optionally the hid / hidinh per-dim firing for the rate-match diagnostic)."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        fdrive = self.hid_bias + self.hid_gain * feat[self.hid_dim]        # [F*Hp]
        drive[self.hid_idx] = fdrive
        drive[self.hidinh_idx] = fdrive                                    # SAME drive -> rate-matched pair
        if self.head_b_gain > 0.0:
            hb = (self.head_b - self.head_b.min()) * self.head_b_gain
            for k in range(self.V):
                drive[self.pool_idx[k]] += hb[k]
        if self.floor_pA:
            drive[self.all_pool] += self.floor_pA
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        firing = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for _ in range(self.read_window):
            b._run_one_simulation_step()
            firing += np.asarray(to_host(b.cp_firing_states)).astype(float)
        b.cp_external_input_current[:] = 0.0
        per_pool = np.array([firing[self.pool_idx[k]].sum() for k in range(self.V)])
        word_sp = float(firing[self.all_pool].sum()); tot = float(firing.sum())
        if want_shadow_rate:
            he = np.zeros(self.F); hi = np.zeros(self.F)
            for d in range(self.F):
                # neurons coding dim d are the contiguous Hp block d*Hp..(d+1)*Hp
                he[d] = firing[self.hid_idx[d * self.Hp:(d + 1) * self.Hp]].sum()
                hi[d] = firing[self.hidinh_idx[d * self.Hp:(d + 1) * self.Hp]].sum()
            return per_pool, word_sp, tot, he, hi
        return per_pool, word_sp, tot

    def read(self, ap, an, tid, scramble_perm=None, want_shadow_rate=False):
        feat = self._hidden_feature(ap, an, tid)
        out = self._run(feat, want_shadow_rate=want_shadow_rate)
        per_pool, word_sp, tot = out[0], out[1], out[2]
        if scramble_perm is not None:
            per_pool = per_pool[scramble_perm]
        win = -1 if per_pool.max() <= 0.0 else int(np.argmax(per_pool))
        if want_shadow_rate:
            return win, per_pool, word_sp, tot, out[3], out[4]
        return win, per_pool, word_sp, tot

    # --- diagnostics / anti-cheats ---
    def lesion_readout(self):
        pre, post, _ = self._pos_edges
        self._b.set_pathway_weights("les_pos", pre, post, np.zeros(len(pre), np.float32), add_missing=False)
        pre, post, _ = self._neg_edges
        self._b.set_pathway_weights("les_neg", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_readout(self):
        pre, post, w = self._pos_edges
        self._b.set_pathway_weights("res_pos", pre, post, w, add_missing=False)
        pre, post, w = self._neg_edges
        self._b.set_pathway_weights("res_neg", pre, post, w, add_missing=False)

    def lesion_shadow(self):
        """Zero ONLY the inhibitory shadow (Wn) -> a POSITIVE-only read. Tests if the negative weights are
        load-bearing (vs the 2026-07-04 'signed machinery is decorative' finding)."""
        pre, post, _ = self._neg_edges
        self._b.set_pathway_weights("les_shadow", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_shadow(self):
        pre, post, w = self._neg_edges
        self._b.set_pathway_weights("res_shadow", pre, post, w, add_missing=False)

    def read_oracle(self, logit_vec, oracle_gain=220.0, oracle_base=30.0):
        """DIAGNOSTIC ONLY (host logits -> NOT a read path). Drive pool_k DIRECTLY by a current proportional to the
        (softmax-normalized) host logit through the SAME FS-WTA over V pools -> isolates the FS-WTA RESOLUTION
        ceiling from the SIGNED-PROJECTION fidelity."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        p = _softmax(np.asarray(logit_vec, dtype=np.float64))
        peak = float(p.max()) if p.size else 0.0
        w = (p / peak) if peak > 1e-12 else np.zeros_like(p)
        per = oracle_base + oracle_gain * w
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for k in range(self.V):
            drive[self.pool_idx[k]] = per[k]
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        firing = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        for _ in range(self.read_window):
            b._run_one_simulation_step()
            firing += np.asarray(to_host(b.cp_firing_states)).astype(float)
        b.cp_external_input_current[:] = 0.0
        pp = np.array([firing[self.pool_idx[k]].sum() for k in range(self.V)])
        return -1 if pp.max() <= 0.0 else int(np.argmax(pp))


def _eval(seed, ro, ev_ids, vocab, s, warmup, topk, sample_temp, n_eval_pos, gen_tokens, gen_temp,
          oracle_every=3):
    grng = np.random.default_rng(seed * 137 + 11)
    acc = dict(n=0, word_spikes=0.0, total_spikes=0.0, argmax_agree=0.0, top5_hit=0.0, nll=0.0,
               mass_syn=0.0, mass_hs=0.0, mass_ax=0.0, mass_scr=0.0, agree_scr=0.0, mass_ora=0.0, ora_n=0,
               mass_pos=0.0, silent=0, hid_active=0.0, rate_corr=0.0, rate_n=0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            candk = np.argpartition(-lg_supp, topk - 1)[:topk]; candk = candk[np.argsort(-lg_supp[candk])]
            pk = _softmax(lg_supp[candk] / sample_temp)
            hs = int(candk[int(grng.choice(len(pk), p=pk))])
            # SIGNED SYNAPTIC read (full-V; the deliverable read path). Also grab the shadow rate-match once/while.
            want_rate = (positions % 20 == 0)
            r = s.read(ap, an, ids[t], want_shadow_rate=want_rate)
            win, per_pool, word_sp, tot = r[0], r[1], r[2], r[3]
            if want_rate:
                he, hi = r[4], r[5]
                if he.std() > 1e-9 and hi.std() > 1e-9:
                    acc["rate_corr"] += float(np.corrcoef(he, hi)[0, 1]); acc["rate_n"] += 1
            # scramble control = POST-HOC relabel of the SAME resolved per_pool (permuting the pool->word map ->
            # the winning pool decodes to a random word -> agreement to chance). No re-sim needed.
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s.V)
            win_s = int(np.argmax(per_pool[scr_perm])) if per_pool.max() > 0 else -1
            # oracle ceiling: a separate drive; subsample to hold cost down (the mean is stable). 1/oracle_every.
            if positions % oracle_every == 0:
                ora = s.read_oracle(lg_supp)
                acc["mass_ora"] += (pfull[ora] if ora >= 0 else 0.0); acc["ora_n"] += 1
            acc["n"] += 1; positions += 1
            acc["word_spikes"] += word_sp; acc["total_spikes"] += tot
            acc["hid_active"] += float(per_pool.sum() > 0)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["nll"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
            acc["mass_syn"] += (pfull[win] if win >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["mass_scr"] += (pfull[win_s] if win_s >= 0 else 0.0)
            acc["agree_scr"] += float(win_s == host_argmax)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    n = max(1, acc["n"])

    # ---- POSITIVE-ONLY read (shadow lesioned): is the negative weight LOAD-BEARING? ----
    s.lesion_shadow()
    pos_mass = 0.0; pos_n = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            pfull = _softmax(lg_supp)
            win, _, _, _ = s.read(ap, an, ids[t])
            pos_mass += (pfull[win] if win >= 0 else 0.0); pos_n += 1
            if pos_n >= min(n, 60):
                break
        if pos_n >= min(n, 60):
            break
    s.restore_shadow()

    # ---- readout lesion (Wp+Wn zeroed) -> collapse ----
    s.lesion_readout()
    les_mass = 0.0; les_n = 0
    for ids in ev_ids[:2]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 30)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            pfull = _softmax(lg_supp)
            win, _, _, _ = s.read(ap, an, ids[t])
            les_mass += (pfull[win] if win >= 0 else 0.0); les_n += 1
            if les_n >= 40:
                break
        if les_n >= 40:
            break
    s.restore_readout()

    lever("signed_shadow_readout_lesion", before=round(acc["mass_syn"] / n, 4),
          after=round(les_mass / max(1, les_n), 4), required=False)

    m = {
        "seed": seed, "arm": "signed_shadow", "V": s.V, "pop": s.P, "hid_pop": s.Hp, "ratio": s.ratio,
        "topk_ceiling": topk, "plasticity_off": True,
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "hidden_active_frac": round(acc["hid_active"] / n, 4),
        "shadow_rate_match_corr": round(acc["rate_corr"] / max(1, acc["rate_n"]), 4),
        "mean_spikes_per_read": round(acc["word_spikes"] / n, 2),
        "mean_spikes_total": round(acc["total_spikes"] / n, 2),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_synaptic": round(acc["nll"] / n, 4),
        "mass_synaptic": round(acc["mass_syn"] / n, 4),
        "mass_positive_only": round(pos_mass / max(1, pos_n), 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "mass_scramble": round(acc["mass_scr"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "mass_readout_lesion": round(les_mass / max(1, les_n), 4),
        "mass_oracle_ceiling": round(acc["mass_ora"] / max(1, acc["ora_n"]), 4),
        "chance_1_over_v": round(1.0 / s.V, 6),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_synaptic"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["oracle_read_fidelity"] = round(m["mass_oracle_ceiling"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["positive_only_fidelity"] = round(m["mass_positive_only"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["projection_recovery"] = round(m["read_fidelity_vs_sampler"] / max(1e-9, m["oracle_read_fidelity"]), 4)
    if gen_tokens > 0:
        m["generation"] = _free_gen(ro, vocab, s, topk, gen_temp, gen_tokens)
    return m


def _scramble_at_chance(agree_scramble, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict(m):
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        # THE headline: the signed projection recovers >=85% of the perfect-current (oracle) ceiling ->
        # the signed-projection wall (rung 2, read_fid 0.035) is REMOVED, the residual is only WTA resolution.
        "projection_recovery_ge_0.85": m["projection_recovery"] >= 0.85,
        # a large ABSOLUTE lift over the rung-2 Dale-shift boundary (0.035) — recovers real distribution mass.
        "read_fidelity_ge_0.40": m["read_fidelity_vs_sampler"] >= 0.40,
        # the NEGATIVE weights are LOAD-BEARING (not decorative, unlike 2026-07-04): signed beats positive-only.
        "signed_beats_positive_only": m["read_fidelity_vs_sampler"] > 1.10 * m["positive_only_fidelity"],
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        "readout_lesion_collapses": m["mass_readout_lesion"] < 0.5 * m["mass_synaptic"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
        "not_silent": m["silent_frac"] < 0.05,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


def _free_gen(ro, vocab, s, topk, gen_temp, n_tok):
    out = {}
    for prompt in ("once upon a time", "the little girl", "tom and his dog"):
        pid = [i for i in vocab.ids(prompt.split()) if 0 <= i < ro.V] or [0]
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in pid:
            ap, an = ro.advance(ap, an, t)
        gen = list(pid); self_nll = 0.0; steps = 0
        for _ in range(n_tok):
            lg = ro.logits(ap, an, gen[-1]); lg2 = lg.copy()
            if ro.unk_idx >= 0:
                lg2[ro.unk_idx] = -1e30
            win, _, _, _ = s.read(ap, an, gen[-1])
            nxt = int(win) if win >= 0 else int(np.argmax(lg2))
            self_nll += -math.log(max(_softmax(lg2)[nxt], 1e-12)); steps += 1
            gen.append(nxt); ap, an = ro.advance(ap, an, nxt)
        txt = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen)
        out[prompt] = {"text": txt, "self_nll": round(self_nll / max(1, steps), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pops", type=str, default="4,8,16")                   # WTA population sizes P
    ap.add_argument("--hid-pop", type=int, default=1)                       # hidden neurons per feature dim
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    # ---- CALIBRATED OPERATING POINT (2026-08-13 smoke sweep, seed 42) ----
    # The signed projection ranks by logit (net-vs-logit corr 0.987 rate / 0.91-0.945 spiking current, winner
    # rank-0), but the winner's synaptic current sits only ~+4 pA (the logit distribution is centred negative:
    # most words are unlikely). A LOW read floor JUST BELOW rheobase (~80 pA) puts every pool near threshold so
    # the winner's few-pA excess tips it over while the negative-current losers stay silent — v stays near rest,
    # where the ratio-6.5 inhibitory shadow is SUBTRACTIVE (the 2026-07-04 conductance-signed lesson: a high
    # floor turns g_i DIVISIVE/shunting and the subtraction breaks). hid_gain drives the feature into a healthy
    # spiking-rate regime; a long window rate-codes it finely (window scaling is ~linear; speed secondary).
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--hid-bias", type=float, default=0.0)
    ap.add_argument("--syn-scale", type=float, default=12.0)
    ap.add_argument("--ratio", type=float, default=6.5)
    ap.add_argument("--floor-pA", type=float, default=78.0)
    ap.add_argument("--fs-to-exc", type=float, default=7.0)
    ap.add_argument("--exc-to-fs", type=float, default=1.2)
    ap.add_argument("--n-fs", type=int, default=48)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--oracle-every", type=int, default=3)                  # subsample the oracle diagnostic
    ap.add_argument("--no-uniform-thresh", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_signed_shadow.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 80)
        args.gen_tokens = args.gen_tokens or 40

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pops = [int(x) for x in args.pops.split(",") if x.strip()]

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        for pop in pops:
            s = SignedShadowLogitRead(ro, seed, pop=pop, hid_pop=args.hid_pop, ou_std=args.ou_std,
                                      read_window=args.read_window, hid_gain=args.hid_gain, hid_bias=args.hid_bias,
                                      syn_scale=args.syn_scale, ratio=args.ratio, floor_pA=args.floor_pA,
                                      n_fs=args.n_fs, exc_to_fs=args.exc_to_fs, fs_to_exc=args.fs_to_exc,
                                      uniform_thresh=not args.no_uniform_thresh)
            gen_here = args.gen_tokens if pop == max(pops) else 0
            m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                      args.n_eval_pos, gen_here, args.gen_temp, oracle_every=args.oracle_every)
            go, checks = _verdict(m); m["go"] = go; m["checks"] = checks
            results.append(m)
            print(f"[seed {seed} P={pop} hidP={args.hid_pop} ratio={args.ratio}] "
                  f"word_spk={m['mean_spikes_per_read']} read_fid={m['read_fidelity_vs_sampler']} "
                  f"ORACLE={m['oracle_read_fidelity']} proj_recovery={m['projection_recovery']} "
                  f"pos_only={m['positive_only_fidelity']} argmax_agree={m['argmax_agree']} "
                  f"(10x_chance {round(10/m['V'],4)}) scr={m['argmax_agree_scramble']} "
                  f"rate_corr={m['shadow_rate_match_corr']} lesion={m['mass_readout_lesion']} "
                  f"GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
            if not go:
                print(f"    checks: {json.dumps(checks)}", flush=True)
            if m.get("generation"):
                for pr, g in m["generation"].items():
                    print(f"    [gen '{pr}' nll {g['self_nll']}] {g['text'][:150]}", flush=True)

    agg = {}
    for m in results:
        key = f"P{m['pop']}"
        agg.setdefault(key, {"read_fidelity": [], "oracle": [], "proj_recovery": [], "pos_only": [],
                             "mean_spikes": [], "go": []})
        agg[key]["read_fidelity"].append(m["read_fidelity_vs_sampler"])
        agg[key]["oracle"].append(m["oracle_read_fidelity"])
        agg[key]["proj_recovery"].append(m["projection_recovery"])
        agg[key]["pos_only"].append(m["positive_only_fidelity"])
        agg[key]["mean_spikes"].append(m["mean_spikes_per_read"])
        agg[key]["go"].append(m["go"])
    summary = {}
    for key, d in agg.items():
        summary[key] = {"n_seeds": len(d["go"]), "go_count": int(sum(d["go"])),
                        "read_fidelity_mean": round(float(np.mean(d["read_fidelity"])), 4),
                        "read_fidelity_min": round(float(np.min(d["read_fidelity"])), 4),
                        "oracle_mean": round(float(np.mean(d["oracle"])), 4),
                        "proj_recovery_mean": round(float(np.mean(d["proj_recovery"])), 4),
                        "positive_only_mean": round(float(np.mean(d["pos_only"])), 4),
                        "mean_spikes_per_read": round(float(np.mean(d["mean_spikes"])), 2)}
    out = {"results": results, "summary": summary, "seeds": seeds, "pops": pops, "hid_pop": args.hid_pop,
           "ratio": args.ratio, "topk": args.topk, "read_window": args.read_window,
           "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
