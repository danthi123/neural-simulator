"""
gap#1 / A1 — biologize the mouth's UPSTREAM output projection `h_pre = Wo_sp @ state` as a SIGNED GRADED-CONDUCTANCE
synaptic read on the spiking substrate. The read-out `head_w @ h` is already a substrate graded-conductance read at
parity (`2026-08-13-fluid-mouth-graded-conductance-read-GO`). This runner moves the NEXT matmul UPSTREAM — the WKV
block's OUTPUT PROJECTION that PRODUCES the hidden feature `h` FROM the recurrent state — off the host and onto the
substrate, via the SAME validated template (Dale-split signed weights -> net graded synaptic-current margin at rest).

WHERE THIS SITS IN THE MOUTH PIPELINE (per-token `tid`, WKV leaky state `ap`/`an`):
    (1) v      = Wv @ LN(emb[tid])                       # input projection      (host, BPTT)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)     # WKV leaky STATE       (host, BPTT decay)  <- upstream state
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))              # receptance gate       (host)
    (4) h      = r_h * (Wo_sp @ [ap,an])                 # OUTPUT PROJECTION      (host, BPTT)  <- THIS RUNNER: step (4a)
    (5) logits = head_w @ h + head_b                     # read-out              (SUBSTRATE graded read, read-GO)
`Wo_sp @ [ap,an]` (residual #2's named host residual `h = r_h*(Wo_sp@state)`) is step (4a). This runner realizes it as
a graded synaptic projection; the r_h gate (4b) is applied host-side (a per-channel gain = a named next rung, shunting)
and the read-out (5) is the validated substrate read. So the substrate boundary moves from "the read consumes a HOST
hidden feature" to "the hidden feature is a SUBSTRATE-computed signed graded synaptic projection of the WKV state".

THE MECHANISM (Dale's principle + the graded-conductance-domain read, both already validated on `head_w @ h`):
`Wo_sp` [D, 2D] is ~50% negative. Split `Wo_sp = Wo_pos - Wo_neg` (both >= 0). The WKV state `[ap,an]` [2D] is already
NONNEG (the dual leaky ON/OFF code), so it rate-codes TWO matched carrier populations driven by the SAME current: an
EXCITATORY `stc_e` and an INHIBITORY `stc_i`. Wire `Wo_pos` as EXCITATORY `stc_e -> hpool` (charges cp_conductance_g_e)
and `Wo_neg` as INHIBITORY `stc_i -> hpool` (charges cp_conductance_g_i). Keep the D hidden pools SUBTHRESHOLD (floor 0)
and read each channel's hidden-feature value from the substrate's OWN net signed synaptic-current DRIVE at rest:
    hpre_k = (E_e - v_ref)*g_e[hpool_k] + (E_i - v_ref)*g_i[hpool_k]   ~   (Wo_pos - Wo_neg)_k @ state  =  Wo_sp_k @ state
integrated over the read window (the ~5-10 ms conductance taus average out the OU noise). This is the CONTINUOUS graded
analog read a distributed code affords (Mikulasch-Priesemann; the 2026-06-20 graded-plateau template; the read-GO's
head_w@h result), NOT a spike count. The inhibitory:excitatory SYNAPTIC ratio is calibrated ONCE (seed 42, a WIDE
plateau) to balance the two driving-force terms so the graded margin reconstructs the signed projection; then FIXED and
tested on the 5 UNSEEN seeds.

METRIC (fidelity of the SUBSTRATE-produced projection vs the host reference — a CONTINUOUS D-vector, not an argmax):
  hpre_corr_signed  = per-position Pearson corr(hpre_substrate, Wo_sp@state)  averaged over positions  [THE headline]
  hpre_cosine       = per-position cosine(hpre_substrate, Wo_sp@state)
  hpre_corr_positive_only = corr with the INHIBITORY SHADOW (Wo_neg) lesioned (df_e*g_e alone) -> the sign is
                            LOAD-BEARING here (the projection is 46% negative; positive-only cannot reconstruct it).
Plus a DOWNSTREAM FUNCTIONAL read: feed the substrate hpre through the (host) r_h gate + head_w read -> next word;
argmax_agree vs the host next word (does the substrate projection carry the LM signal), and the on-distribution mass.

ANTI-CHEATS (the two load-bearing collapse controls, both cache-immune, MUST collapse):
  - zero-state collapse (cache-immune): drive the carriers with a ZERO state -> the substrate hpre loses its structure
    -> downstream argmax_agree drops to chance (the state input drives the read; not a floor/frequency artifact).
  - scramble: post-hoc permute the hpool->channel decode map -> corr(hpre_sub, hpre_host) collapses to ~0 and the
    downstream read to chance (the labelled-line pool->channel map carries the projection).
  - provenance: hpre read from cp_conductance_g_e/g_i, 0 host categorical draws on the projection read path.
  - (DIAGNOSTIC ONLY, NOT gated) projection weight-lesion: recorded as hpre_corr_lesion but a KNOWN-UNRELIABLE
    instrument in this wiring (verified a NO-OP on cupy: set_pathway_weights on a fresh pathway name does not zero
    the existing proj synapses) — exactly the contamination the read-GO parent documented and replaced with the
    zero-INPUT control above. The cache-immune zero-state + scramble are the gating collapse controls.
HONEST SCOPE: this biologizes step (4a) `Wo_sp @ state` ONLY. The WKV recurrent STATE (steps 1-2: Wv + the leaky
integrator + the BPTT-trained decay) is STILL host; the r_h gate (4b) is applied host-side; the read-out weights are
host-designed. This is a PARTIAL upstream step (a named residual moved onto the substrate), not "fully spiking".

Reuse-by-import: WKVReadout + _softmax + _native + _load_eval from `_wkv_fewspike_read_derisk`; the signed graded
conductance read PATTERN + reversal-potential handling from `_wkv_graded_conductance_read_derisk` / the parent
SignedShadowLogitRead. NO `sim/` edit — drives + reads public bridge arrays; cfg.seed-controlled substrate (CLAUDE.md
seed trap). Runner-only, default-off.

Run (smoke):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_graded_output_projection_derisk \
                --smoke --seeds 42
Run (6-seed): .venv/bin/python -m research.runners._wkv_graded_output_projection_derisk \
                --seeds 42,43,44,100,101,102 \
                --json research/findings/raw/_wkv_graded_output_projection_6seed.json
Run (calib):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_graded_output_projection_derisk \
                --calib --seeds 42
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
from tools.lab import lever, void_if  # noqa: E402


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


class GradedOutputProjection:
    """Realize `h_pre = Wo_sp @ [ap,an]` as a SIGNED GRADED-CONDUCTANCE synaptic projection on the substrate. The
    nonneg WKV state drives two matched carrier populations (stc_e excitatory, stc_i inhibitory); Wo_pos wires
    stc_e->hpool (E_TO_E, charging g_e), Wo_neg*ratio wires stc_i->hpool (I_TO_E, charging g_i). Each hidden pool's
    reconstructed value is the net signed synaptic-current margin at rest, off cp_conductance_g_e/g_i (a continuous
    graded read, NOT a spike count)."""

    def __init__(self, ro: WKVReadout, seed, pop=1, carrier_pop=1, ou_std=40.0, read_window=150,
                 drive_gain=120.0, drive_bias=0.0, syn_scale=12.0, ratio=0.3, graded_floor_pA=0.0,
                 settle_frac=0.2, uniform_thresh=True):
        self.ro = ro
        self.D = int(ro.D); self.F = 2 * self.D            # projection input dim (state = [ap,an])
        self.P = int(pop); self.Cp = int(carrier_pop)
        self.ou_std = float(ou_std); self.read_window = int(read_window)
        self.drive_gain = float(drive_gain); self.drive_bias = float(drive_bias)
        self.syn_scale = float(syn_scale); self.ratio = float(ratio); self.floor_pA = float(graded_floor_pA)
        self.settle_frac = float(settle_frac); self.uniform_thresh = bool(uniform_thresh)
        self.seed = int(seed)
        self.n_host_rng_draws = 0                          # MUST stay 0 on the projection read path

        Wo = ro.Wo_sp.astype(np.float64)                   # [D, 2D]  h_pre = Wo @ state
        self.Wo_pos = np.maximum(Wo, 0.0)                  # [D, 2D] >= 0  (excitatory)
        self.Wo_neg = np.maximum(-Wo, 0.0)                 # [D, 2D] >= 0  (inhibitory shadow)
        self._build_bridge()
        self._wire()

    # ---------------------------------------------------------------------------------------------------------------
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
        Cn = self.F * self.Cp
        regions = [
            BrainRegion(name="stc_e", n_neurons=Cn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="stc_i", n_neurons=Cn, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="hpool", n_neurons=self.D * self.P, exc_fraction=1.0, internal_density=0.0),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.stc_e_idx = np.asarray(list(rm.indices("stc_e")), dtype=np.int64)     # [F*Cp]
        self.stc_i_idx = np.asarray(list(rm.indices("stc_i")), dtype=np.int64)     # [F*Cp]
        self.carr_dim = np.repeat(np.arange(self.F), self.Cp).astype(np.int64)     # [F*Cp] state dim per carrier
        hpool_idx = np.asarray(list(rm.indices("hpool")), dtype=np.int64)
        self.hpool_idx = hpool_idx                                                 # [D*P]  block layout dim*P..
        self.all_pool = hpool_idx
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())
        cfg2 = b.core_config
        self.E_e = float(getattr(cfg2, "syn_reversal_potential_e", 0.0))
        self.E_i = float(getattr(cfg2, "syn_reversal_potential_i", -75.0))
        self.v_ref = float(to_host(self._v0).mean()) if self._v0 is not None else -65.0
        self.df_e = self.E_e - self.v_ref                 # excitatory driving force at rest (>0)
        self.df_i = self.E_i - self.v_ref                 # inhibitory driving force at rest (<0)

    def _wire(self):
        b = self._b
        union = {}
        Wp = (self.Wo_pos * self.syn_scale).astype(np.float32)                     # [D, 2D]
        Wn = (self.Wo_neg * self.syn_scale * self.ratio).astype(np.float32)        # [D, 2D] driving-force compensated
        Wp_cn = Wp[:, self.carr_dim]                                               # [D, F*Cp]
        Wn_cn = Wn[:, self.carr_dim]                                               # [D, F*Cp]
        nC = len(self.stc_e_idx)
        # ---- Wo_pos: EXCITATORY stc_e -> hpool ----
        pre = np.tile(self.stc_e_idx, self.D * self.P)
        post = np.repeat(self.all_pool, nC)
        wp = np.repeat(Wp_cn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["proj_pos"] = {"pre_indices": pre, "post_indices": post, "initial_weights": wp,
                             "plastic": False, "conn_type": "E_TO_E"}
        # ---- Wo_neg: INHIBITORY stc_i -> hpool ----
        pre_n = np.tile(self.stc_i_idx, self.D * self.P)
        wn = np.repeat(Wn_cn, self.P, axis=0).reshape(-1).astype(np.float32)
        union["proj_neg"] = {"pre_indices": pre_n, "post_indices": post.copy(), "initial_weights": wn,
                             "plastic": False, "conn_type": "I_TO_E"}
        b.inject_explicit_wiring(union, output_inhibitory_indices=self.stc_i_idx.tolist())
        self._pos_edges = (union["proj_pos"]["pre_indices"], union["proj_pos"]["post_indices"], wp.copy())
        self._neg_edges = (union["proj_neg"]["pre_indices"], union["proj_neg"]["post_indices"], wn.copy())

    # ---------------------------------------------------------------------------------------------------------------
    def _reset(self):
        b = self._b
        if self._v0 is not None:
            b.cp_membrane_potential_v[:] = self._v0
        else:
            b.cp_membrane_potential_v[:] = -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        # clear ALL synaptic conductance so each read integrates ONLY its own drive (else g_e/g_i carry over)
        for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                     "cp_conductance_g_nmda_rise", "cp_conductance_g_gabab"):
            arr = getattr(b, name, None)
            if arr is not None:
                arr[:] = 0.0

    def _graded_hpre(self, state, scramble_perm=None, zero_state=False):
        """Drive stc_e + stc_i by (drive_bias + drive_gain*state[dim]); pools get only the (subthreshold) floor. Run
        the read window; INTEGRATE the per-pool net signed synaptic current at v_ref off cp_conductance_g_e/g_i.
        Return the CONTINUOUS per-channel reconstructed projection hpre (len D), and the positive-only variant."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        if zero_state:
            state = np.zeros_like(state)
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        cdrive = self.drive_bias + self.drive_gain * state[self.carr_dim]          # [F*Cp]
        drive[self.stc_e_idx] = cdrive
        drive[self.stc_i_idx] = cdrive                                             # SAME drive -> rate-matched pair
        if self.floor_pA:
            drive[self.all_pool] += self.floor_pA
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        n_acc = 0
        ge_sum = np.zeros(self.D); gi_sum = np.zeros(self.D)
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            ge = np.asarray(to_host(b.cp_conductance_g_e)).astype(np.float64)[self.all_pool].reshape(self.D, self.P)
            gi = np.asarray(to_host(b.cp_conductance_g_i)).astype(np.float64)[self.all_pool].reshape(self.D, self.P)
            ge_sum += ge.sum(axis=1)
            gi_sum += gi.sum(axis=1)
            n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        ge_mean = ge_sum / n_acc; gi_mean = gi_sum / n_acc
        hpre = self.df_e * ge_mean + self.df_i * gi_mean                           # [D] signed graded reconstruction
        hpre_pos = self.df_e * ge_mean                                            # [D] positive-only (Wo_pos alone)
        if scramble_perm is not None:
            hpre = hpre[scramble_perm]; hpre_pos = hpre_pos[scramble_perm]
        return hpre, hpre_pos

    # --- diagnostics / anti-cheats ---
    def lesion_projection(self):
        pre, post, _ = self._pos_edges
        self._b.set_pathway_weights("les_pos", pre, post, np.zeros(len(pre), np.float32), add_missing=False)
        pre, post, _ = self._neg_edges
        self._b.set_pathway_weights("les_neg", pre, post, np.zeros(len(pre), np.float32), add_missing=False)

    def restore_projection(self):
        pre, post, w = self._pos_edges
        self._b.set_pathway_weights("res_pos", pre, post, w, add_missing=False)
        pre, post, w = self._neg_edges
        self._b.set_pathway_weights("res_neg", pre, post, w, add_missing=False)


def _host_next(ro, ap, an, tid, hpre_override=None):
    """Compute the mouth's next-word logits with an OPTION to substitute the projection hpre with a substrate one.
    logits = head_w @ (r_h * hpre) + head_b ; hpre = Wo_sp @ state (host) unless overridden."""
    state = np.concatenate([ap, an])
    r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
    hpre = (ro.Wo_sp @ state) if hpre_override is None else np.asarray(hpre_override, dtype=np.float64)
    return ro.head_w @ (r_h * hpre) + ro.head_b


def _eval(seed, ro, ev_ids, vocab, s, warmup, n_eval_pos, gen_tokens):
    acc = dict(n=0, corr=0.0, corr_pos=0.0, cos=0.0, argmax_agree=0.0, top5_hit=0.0,
               mass_sub=0.0, mass_ax=0.0, corr_scr=0.0, silent=0)
    corr_min = 1.0
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            state = np.concatenate([ap, an])
            hpre_host = ro.Wo_sp @ state                                          # [D] the reference projection
            # host reference next word (from the exact host hpre)
            lg_host = _host_next(ro, ap, an, ids[t]); lg_supp = lg_host.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            # SUBSTRATE graded projection read (the deliverable path; 0 host draws)
            hpre_sub, hpre_pos = s._graded_hpre(state)
            c = _corr(hpre_sub, hpre_host); cp = _corr(hpre_pos, hpre_host)
            acc["corr"] += c; acc["corr_pos"] += cp; acc["cos"] += _cosine(hpre_sub, hpre_host)
            corr_min = min(corr_min, c)
            # downstream FUNCTIONAL read: substrate hpre -> host r_h gate + head_w -> next word
            lg_sub = _host_next(ro, ap, an, ids[t], hpre_override=hpre_sub)
            lg_sub_supp = lg_sub.copy()
            if ro.unk_idx >= 0:
                lg_sub_supp[ro.unk_idx] = -1e30
            win = int(np.argmax(lg_sub_supp))
            if float(hpre_sub.max() - hpre_sub.min()) <= 1e-9:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["mass_sub"] += pfull[win]; acc["mass_ax"] += pfull[host_argmax]
            # scramble control (post-hoc pool->channel relabel of the SAME hpre_sub)
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s.D)
            acc["corr_scr"] += _corr(hpre_sub[scr_perm], hpre_host)
            acc["n"] += 1; positions += 1
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(acc["n"] == 0, "no evaluable positions (every eval sentence shorter than warmup+2) — metrics undefined")
    n = max(1, acc["n"])

    # ---- ZERO-STATE collapse control (cache-immune): silence the projection INPUT -> downstream chance ----
    zs_agree = 0; zs_n = 0
    for ids in ev_ids[:4]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 30)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            state = np.concatenate([ap, an])
            lg_host = _host_next(ro, ap, an, ids[t]); lg_supp = lg_host.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_supp))
            hpre_zs, _ = s._graded_hpre(state, zero_state=True)
            lg_zs = _host_next(ro, ap, an, ids[t], hpre_override=hpre_zs)
            if ro.unk_idx >= 0:
                lg_zs[ro.unk_idx] = -1e30
            zs_agree += int(int(np.argmax(lg_zs)) == host_am); zs_n += 1
            if zs_n >= 60:
                break
        if zs_n >= 60:
            break

    # ---- PROJECTION lesion (Wo_pos+Wo_neg zeroed) -> corr collapse ----
    s.lesion_projection()
    les_corr = 0.0; les_n = 0
    for ids in ev_ids[:2]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 20)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            state = np.concatenate([ap, an])
            hpre_host = ro.Wo_sp @ state
            hpre_les, _ = s._graded_hpre(state)
            les_corr += abs(_corr(hpre_les, hpre_host)); les_n += 1
            if les_n >= 40:
                break
        if les_n >= 40:
            break
    s.restore_projection()

    lever("graded_projection_lesion_corr", before=round(acc["corr"] / n, 4),
          after=round(les_corr / max(1, les_n), 4), required=False)
    lever("graded_signed_vs_positive_corr", before=round(acc["corr_pos"] / n, 4),
          after=round(acc["corr"] / n, 4), required=False)
    lever("graded_zero_state_collapse_argmax", before=round(acc["argmax_agree"] / n, 4),
          after=round(zs_agree / max(1, zs_n), 4), required=False)

    chance = 1.0 / ro.V
    m = {
        "seed": seed, "arm": "graded_output_projection", "V": ro.V, "D": ro.D, "pop": s.P,
        "carrier_pop": s.Cp, "ratio": s.ratio, "drive_gain": s.drive_gain, "syn_scale": s.syn_scale,
        "v_ref": round(s.v_ref, 2), "df_e": round(s.df_e, 2), "df_i": round(s.df_i, 2),
        "plasticity_off": True, "n_positions": acc["n"],
        "hpre_corr_signed": round(acc["corr"] / n, 4),
        "hpre_corr_signed_min": round(corr_min, 4),
        "hpre_corr_positive_only": round(acc["corr_pos"] / n, 4),
        "hpre_cosine_signed": round(acc["cos"] / n, 4),
        "hpre_corr_scramble": round(acc["corr_scr"] / n, 4),
        "hpre_corr_lesion": round(les_corr / max(1, les_n), 4),
        "downstream_argmax_agree": round(acc["argmax_agree"] / n, 4),
        "downstream_argmax_agree_zerostate": round(zs_agree / max(1, zs_n), 4),
        "downstream_top5_hit": round(acc["top5_hit"] / n, 4),
        "downstream_mass_sub": round(acc["mass_sub"] / n, 4),
        "downstream_mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "silent_frac": round(acc["silent"] / n, 4),
        "chance_1_over_v": round(chance, 6),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
    }
    m["downstream_read_fidelity_vs_argmax"] = round(
        m["downstream_mass_sub"] / max(1e-9, m["downstream_mass_argmax_ceiling"]), 4)
    if gen_tokens > 0:
        m["generation"] = _free_gen(ro, vocab, s, gen_tokens)
    return m


def _scramble_at_zero(corr_scramble):
    return abs(corr_scramble) < 0.1


def _verdict(m):
    chance = m["chance_1_over_v"]
    checks = {
        # THE headline: the substrate graded projection RECONSTRUCTS Wo_sp@state at high fidelity.
        "hpre_corr_signed_ge_0.9": m["hpre_corr_signed"] >= 0.9,
        "hpre_corr_min_ge_0.85": m["hpre_corr_signed_min"] >= 0.85,
        # the NEGATIVE weights are LOAD-BEARING (the projection is ~46% negative; positive-only cannot reconstruct).
        "signed_beats_positive_only": m["hpre_corr_signed"] > m["hpre_corr_positive_only"] + 0.05,
        # the substrate projection carries the LM signal downstream.
        "downstream_argmax_agree_gt_10x_chance": m["downstream_argmax_agree"] > 10 * chance,
        # anti-cheats collapse. NOTE the weight-lesion (hpre_corr_lesion) is DELIBERATELY NOT gated: it is a
        # KNOWN-UNRELIABLE instrument in this wiring (verified here a NO-OP on cupy — set_pathway_weights on a
        # fresh pathway name does not zero the existing proj synapses; hpre unchanged 13.0->13.0), exactly the
        # contamination the read-GO parent documented and REPLACED with the cache-immune zero-INPUT control. The
        # load-bearing collapse controls are scramble (post-hoc relabel -> corr 0) + zero-state (silence the
        # projection INPUT -> downstream chance); both are cache-immune and pass. hpre_corr_lesion stays in the
        # JSON as a transparent diagnostic only.
        "scramble_at_zero": _scramble_at_zero(m["hpre_corr_scramble"]),
        "zero_state_collapses":
            m["downstream_argmax_agree_zerostate"] <= 0.34 * m["downstream_argmax_agree"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "not_silent": m["silent_frac"] < 0.05,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


def _free_gen(ro, vocab, s, n_tok):
    """Free-generate with the SUBSTRATE projection in the loop (hpre from the graded read each step)."""
    out = {}
    for prompt in ("once upon a time", "the little girl", "tom and his dog"):
        pid = [i for i in vocab.ids(prompt.split()) if 0 <= i < ro.V] or [0]
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in pid:
            ap, an = ro.advance(ap, an, t)
        gen = list(pid); self_nll = 0.0; steps = 0
        for _ in range(n_tok):
            state = np.concatenate([ap, an])
            hpre_sub, _ = s._graded_hpre(state)
            lg = _host_next(ro, ap, an, gen[-1], hpre_override=hpre_sub); lg2 = lg.copy()
            if ro.unk_idx >= 0:
                lg2[ro.unk_idx] = -1e30
            nxt = int(np.argmax(lg2))
            lg_ref = _host_next(ro, ap, an, gen[-1]); lg_ref2 = lg_ref.copy()
            if ro.unk_idx >= 0:
                lg_ref2[ro.unk_idx] = -1e30
            self_nll += -math.log(max(_softmax(lg_ref2)[nxt], 1e-12)); steps += 1
            gen.append(nxt); ap, an = ro.advance(ap, an, nxt)
        txt = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen)
        out[prompt] = {"text": txt, "self_nll": round(self_nll / max(1, steps), 3)}
    return out


def _calibrate(ro, seed, ev_ids, args):
    """Calibrate the inh:exc ratio + drive_gain ONCE on the given seed by maximizing hpre_corr_signed; print the
    plateau so a WIDE (non-knife-edge) operating point can be fixed."""
    print("[calib] ratio sweep (drive_gain fixed):", flush=True)
    best = None
    for gain in [float(x) for x in args.calib_gains.split(",")]:
        for ratio in [float(x) for x in args.calib_ratios.split(",")]:
            s = GradedOutputProjection(ro, seed, pop=args.pop, carrier_pop=args.carrier_pop, ou_std=args.ou_std,
                                       read_window=args.read_window, drive_gain=gain, drive_bias=args.drive_bias,
                                       syn_scale=args.syn_scale, ratio=ratio, graded_floor_pA=args.graded_floor_pA,
                                       settle_frac=args.settle_frac, uniform_thresh=not args.no_uniform_thresh)
            cs = []; cps = []
            npos = 0
            for ids in ev_ids:
                if len(ids) < args.warmup + 2:
                    continue
                ap = np.zeros(ro.D); an = np.zeros(ro.D)
                for t in range(len(ids) - 1):
                    ap, an = ro.advance(ap, an, ids[t])
                    if t < args.warmup:
                        continue
                    state = np.concatenate([ap, an])
                    hpre_host = ro.Wo_sp @ state
                    hpre_sub, hpre_pos = s._graded_hpre(state)
                    cs.append(_corr(hpre_sub, hpre_host)); cps.append(_corr(hpre_pos, hpre_host))
                    npos += 1
                    if npos >= 40:
                        break
                if npos >= 40:
                    break
            c = float(np.mean(cs)); cp = float(np.mean(cps))
            print(f"    gain={gain:6.1f} ratio={ratio:4.2f} -> corr_signed={c:.4f} corr_pos={cp:.4f} "
                  f"(signed-pos={c-cp:+.4f})", flush=True)
            if best is None or c > best[0]:
                best = (c, gain, ratio, cp)
    print(f"[calib] BEST corr_signed={best[0]:.4f} at gain={best[1]} ratio={best[2]} (pos_only={best[3]:.4f})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pop", type=int, default=1)                    # P is nearly irrelevant to a graded read
    ap.add_argument("--carrier-pop", type=int, default=1)
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    # ---- GRADED operating point (calibrated once on seed 42; see --calib) ----
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--drive-gain", type=float, default=120.0)
    ap.add_argument("--drive-bias", type=float, default=0.0)
    ap.add_argument("--syn-scale", type=float, default=12.0)
    # inh:exc SYNAPTIC ratio, calibrated ONCE on seed 42 (WIDE plateau 0.3-0.7 -> corr 0.96-0.98, not a knife-edge)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--graded-floor-pA", type=float, default=0.0)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--no-uniform-thresh", action="store_true")
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-gains", type=str, default="80,120,180")
    ap.add_argument("--calib-ratios", type=str, default="0.15,0.3,0.5,0.7,1.0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_graded_output_projection.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 60)
        args.gen_tokens = args.gen_tokens or 30

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.calib:
        seed = seeds[0]
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        ro = WKVReadout(ckpt)
        ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, 64)
        _calibrate(ro, seed, ev_ids, args)
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
        s = GradedOutputProjection(ro, seed, pop=args.pop, carrier_pop=args.carrier_pop, ou_std=args.ou_std,
                                   read_window=args.read_window, drive_gain=args.drive_gain,
                                   drive_bias=args.drive_bias, syn_scale=args.syn_scale, ratio=args.ratio,
                                   graded_floor_pA=args.graded_floor_pA, settle_frac=args.settle_frac,
                                   uniform_thresh=not args.no_uniform_thresh)
        m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.n_eval_pos, args.gen_tokens)
        go, checks = _verdict(m); m["go"] = go; m["checks"] = checks
        results.append(m)
        print(f"[seed {seed} P={args.pop} ratio={args.ratio} gain={args.drive_gain}] "
              f"corr_signed={m['hpre_corr_signed']} (min {m['hpre_corr_signed_min']}) "
              f"corr_pos={m['hpre_corr_positive_only']} cos={m['hpre_cosine_signed']} "
              f"scr={m['hpre_corr_scramble']} lesion={m['hpre_corr_lesion']} "
              f"down_agree={m['downstream_argmax_agree']} (zerostate {m['downstream_argmax_agree_zerostate']}, "
              f"10x_chance {round(10/m['V'],4)}) GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
        if not go:
            print(f"    checks: {json.dumps(checks)}", flush=True)
        if m.get("generation"):
            for pr, g in m["generation"].items():
                print(f"    [gen '{pr}' nll {g['self_nll']}] {g['text'][:150]}", flush=True)

    if results:
        arr = lambda k: [r[k] for r in results]  # noqa: E731
        summary = {
            "n_seeds": len(results), "go_count": int(sum(r["go"] for r in results)),
            "hpre_corr_signed_mean": round(float(np.mean(arr("hpre_corr_signed"))), 4),
            "hpre_corr_signed_min": round(float(np.min(arr("hpre_corr_signed_min"))), 4),
            "hpre_corr_positive_only_mean": round(float(np.mean(arr("hpre_corr_positive_only"))), 4),
            "hpre_cosine_signed_mean": round(float(np.mean(arr("hpre_cosine_signed"))), 4),
            "signed_load_bearing_count": int(sum(
                r["hpre_corr_signed"] > r["hpre_corr_positive_only"] + 0.05 for r in results)),
            "downstream_argmax_agree_mean": round(float(np.mean(arr("downstream_argmax_agree"))), 4),
            "hpre_corr_scramble_mean": round(float(np.mean(arr("hpre_corr_scramble"))), 4),
        }
    else:
        summary = {"n_seeds": 0, "go_count": 0}
    out = {"results": results, "summary": summary, "seeds": seeds, "pop": args.pop,
           "ratio": args.ratio, "drive_gain": args.drive_gain, "read_window": args.read_window,
           "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
