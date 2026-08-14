"""gap#1 / A1 — biologize the mouth's r_h RECEPTANCE GATE onto the substrate as a DIVISIVE SHUNTING-INHIBITION
conductance, removing the last host elementwise MULTIPLY between the substrate projection and the winner.

WHERE THIS SITS (per-token `tid`, WKV leaky state `ap`/`an`):
    (1) v      = Wv @ LN(emb[tid])                       # input projection    (host, BPTT — DECLARED residual)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)     # WKV leaky STATE     <<< [WK] SUBSTRATE slow-NMDA conductance
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))              # receptance gate     (host value — DECLARED residual)
    (4) h      = r_h * (Wo_sp @ [ap,an])                 # OUTPUT PROJECTION   <<< [CE] SUBSTRATE graded read
    (5) logits = head_w @ h + head_b                     # read-out            <<< [CE] SUBSTRATE graded read + bias pop
The full state->logits chain is already substrate end-to-end (2026-08-13-fluid-mouth-full-substrate-pipeline-GO,
recov_argmax 0.9137), EXCEPT the gate application `r_h * (...)` in step (4), which the full-pipeline runner does as a
HOST ELEMENTWISE MULTIPLY inside ComposedEndToEndRead._feature. THIS LANE moves that multiply onto the substrate.

THE MECHANISM (Holt & Koch 1997 — shunting inhibition divides the somatic membrane response; Chance-Abbott-Reyes 2002 —
a background conductance state makes it DIVISIVE): `h[k] = r_h[k] * hpre[k]` is a per-channel MULTIPLICATIVE gain with
r_h in (0,1). Realize it as a divisive SHUNTING conductance read from the MEMBRANE (a LINEAR, small-value-preserving
read — a RATE read rectifies the graded signal and lost recov 0.35; the membrane read is corr 0.96-0.98 linear in
drive). A DUAL ON/OFF pair per channel is driven SUBTHRESHOLD: gon[k] by max(hpre[k],0), goff[k] by max(-hpre[k],0) —
BOTH in the POSITIVE-drive regime so the shunt divides SYMMETRICALLY (the single-pool Izhikevich response is asymmetric
in sign). A per-channel INHIBITORY SHUNT sub-pool wires I_TO_E onto BOTH gon[k] and goff[k], with its reversal PINNED to
the pool's OWN resting potential (pure shunting -> no zero shift). Its conductance g_shunt[k] DIVIDES the membrane
deflection Δv ~ drive/(1+g_shunt/g_L). Set the shunt drive so g_shunt[k] ~ (1-r_h[k])/r_h[k] -> the divisive factor ~
r_h[k]. The signed gated feature is the DIFFERENTIAL of the two pools' Δv over rest: feat = rate_scale*[Δv_on, Δv_off] ~
[max(h,0), max(-h,0)] with h = r_h*hpre. The rest-pinned reversal + differential make hpre=0 -> 0 output at ANY shunt
level (stable zero). Two scalars are calibrated ONCE on seed 42 (FIXED for the 5 unseen seeds): rate_scale (mV Δv ->
host feature scale, auto unit-mapped) and shunt_gain ((1-r_h)/r_h -> shunt-drive scale). Then feat -> [CE]'s substrate
read chain (read-out + head_b bias pop) exactly as the full pipeline.

r_h ITSELF stays a host VALUE (sigmoid(Wr@LN(emb)) — a DECLARED residual, the same class as Wv), exactly like every
parent runner converts a host rate into a substrate DRIVE. What moves onto the substrate is the GATE APPLICATION: the
product r_h*hpre is NEVER computed in host arithmetic; r_h enters as a shunt DRIVE and hpre as a separate pool drive,
and the MEMBRANE (the shunt conductance dividing the deflection) forms the product. 0 host multiply of r_h*hpre.

THE ARMS (per position, same eval set; reference = the FULL host mouth ro.logits(ap_host,an_host,tid)):
    A  shuntgate : SUBSTRATE state -> SUBSTRATE proj -> SUBSTRATE r_h shunt gate -> SUBSTRATE read   (THE deliverable)
    B  hostgate  : SUBSTRATE state -> SUBSTRATE proj -> HOST r_h multiply        -> SUBSTRATE read   (== the 0.9137 GO)
Headline: does arm A hold NEAR arm B's recov (the shunt genuinely realizes r_h), with 0 host multiply on the gate?

ANTI-CHEATS (arm A; each MUST move as stated — brain-based, negatives load-bearing):
  * LESION THE SHUNT (drop the shunt drive to 0 -> g_shunt=0 -> every channel ungated, r_h->1): the gate is LOST; the
    pipeline degrades away from arm B toward the ungated read (the shunt conductance is load-bearing).
  * SCRAMBLE THE SHUNT (permute which channel's r_h drives which shunt sub-pool): the wrong channels are gated ->
    recov degrades (the labelled-line r_h->channel map carries the gate).
  * GATE FIDELITY: corr(feat_substrate, [max(r_h*hpre,0),max(-r_h*hpre,0)]) high (the shunt reconstructs r_h*hpre).
  * STATE/READ chain still collapses: zero-input (state) -> chance; memoryless -> degrade; scramble read -> chance.
  * PROVENANCE: gate Δv off cp_membrane_potential_v of gon/goff; shunt via cp_conductance_g_i on the gate pools; winner off
    cp_conductance_g_e/g_i; host_rng_draws_on_read_path == 0; 0 host matmul on the margin/state; 0 host r_h*hpre multiply.
  * 6 seeds 42/43/44/100/101/102 (smoke first); single fixed operating point (both arcs' seed-42 calibs + this gate's).

HONEST SCOPE: still host = the input projection Wv (BPTT; the LEARNING rule is the separate 2026-08-12 e-prop GO), the
r_h VALUE (sigmoid(Wr@LN(emb)) — a declared residual), the LN, and the trained decay/Wo_sp/head weights + fixed
unit-scalars. This lane moves the GATE APPLICATION (the r_h*hpre multiply) onto a spiking shunting conductance. If arm A
holds near arm B -> only LN + the embedding + the trained weights remain host between sensation and the winner. NOT
"fully spiking" / NOT production-wired. Runner-only, default-off, NO sim/ edit.

Run (smoke):   SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_rh_shunt_derisk --smoke --seeds 42
Run (calib):   SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_rh_shunt_derisk --calib --seeds 42
Run (6-seed):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_rh_shunt_derisk \
                 --seeds 42,43,44,100,101,102 --json research/findings/raw/_wkv_rh_shunt_6seed.json
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
from research.runners._wkv_graded_recurrent_state_derisk import (  # noqa: E402
    GradedRecurrentState, _rates, _ref_advance, _fit_calib, _cal_state,
)
from research.runners._wkv_mouth_endtoend_substrate_read_derisk import (  # noqa: E402
    ComposedEndToEndRead, _build_proj, _build_read,
)
from tools.lab import lever, void_if  # noqa: E402


def _corr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ====================================================================================================================
# The SUBSTRATE r_h shunt gate: a divisive SHUNTING-INHIBITION conductance on a DUAL ON/OFF pair of SUBTHRESHOLD gate
# pools, read from the MEMBRANE (a LINEAR, small-value-preserving read — NOT a rate, which rectifies the graded signal
# and lost recov 0.35). gon[k] is driven subthreshold by max(hpre[k],0), goff[k] by max(-hpre[k],0) — BOTH in the
# POSITIVE-drive regime where the shunt divides SYMMETRICALLY. A per-channel inhibitory shunt sub-pool, with its
# reversal pinned to the pool's OWN RESTING potential (pure shunting: no zero shift, Holt-Koch 1997), charges
# cp_conductance_g_i on gon+goff; the membrane deflection Δv is DIVIDED by the total conductance ~1/(1+g_shunt/g_L)
# (Chance-Abbott-Reyes 2002, the high-conductance divisive regime). g_shunt is driven from (1-r_h)/r_h so the factor
# ~r_h. The signed gated feature = rate_scale*(Δv_on - Δv_off); the differential + rest-pinned reversal make hpre=0 ->
# 0 output at ANY shunt (stable zero). r_h enters ONLY as a shunt DRIVE, hpre only as a pool drive — the MEMBRANE forms
# the product r_h*hpre; 0 host multiply.
# ====================================================================================================================
class ShuntGate:
    def __init__(self, D, seed, pool=8, shunt_pop=4, read_window=180, settle_frac=0.3, ou_std=40.0,
                 drive_gain=100.0, shunt_gain=110.0, shunt_syn_w=25.0, rate_scale=1.0, uniform_thresh=True):
        self.D = int(D); self.seed = int(seed)
        self.Pg = int(pool); self.Sg = int(shunt_pop)
        self.read_window = int(read_window); self.settle_frac = float(settle_frac)
        self.ou_std = float(ou_std); self.drive_gain = float(drive_gain)
        self.shunt_gain = float(shunt_gain); self.shunt_syn_w = float(shunt_syn_w)
        self.rate_scale = float(rate_scale); self.uniform_thresh = bool(uniform_thresh)
        self.n_host_rng_draws = 0
        self.n_host_gate_mult = 0                     # MUST stay 0: no host r_h*hpre multiply on the gate path
        self._build_bridge()
        self._wire()
        self._calibrate_rest()                        # measure per-neuron rest -> baseline + pin shunt reversal to rest

    def _build_bridge(self):
        cfg = CoreSimConfig()
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0; cfg.seed = self.seed
        cfg.heterogeneity_seed = self.seed; cfg.ou_seed = self.seed
        cfg.enable_brain_region_framework = True
        cfg.connections_per_neuron = 0; cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_input_divisive_norm",
                  "enable_nmda"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.enable_ou_process = self.ou_std > 0.0
        cfg.ou_mean_current_pA = 0.0; cfg.ou_std_current_pA = self.ou_std; cfg.ou_tau_ms = 15.0
        cfg.stdp_w_max = 4000.0; cfg.hebbian_max_weight = 4000.0
        regions = [
            BrainRegion(name="shunt", n_neurons=self.D * self.Sg, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="gon", n_neurons=self.D * self.Pg, exc_fraction=1.0, internal_density=0.0),
            BrainRegion(name="goff", n_neurons=self.D * self.Pg, exc_fraction=1.0, internal_density=0.0),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = []
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)
        self._b = b
        rm = b.region_manager
        self.shunt_idx = np.asarray(list(rm.indices("shunt")), dtype=np.int64)
        self.gon_idx = np.asarray(list(rm.indices("gon")), dtype=np.int64)
        self.goff_idx = np.asarray(list(rm.indices("goff")), dtype=np.int64)
        self.shunt_dim = np.repeat(np.arange(self.D), self.Sg).astype(np.int64)
        self._v0 = (b.cp_izh_c_reset.copy() if getattr(b, "cp_izh_c_reset", None) is not None else None)
        if self.uniform_thresh and getattr(b, "cp_neuron_firing_thresholds", None) is not None:
            thr = b.cp_neuron_firing_thresholds
            thr[:] = float(to_host(thr).mean())

    def _wire(self):
        b = self._b
        union = {}
        for name, tgt in (("gon", self.gon_idx), ("goff", self.goff_idx)):
            pre = []; post = []
            for k in range(self.D):
                s_k = self.shunt_idx[self.shunt_dim == k]
                t_k = tgt[k * self.Pg:(k + 1) * self.Pg]
                pre.append(np.tile(s_k, len(t_k)))
                post.append(np.repeat(t_k, len(s_k)))
            pre = np.concatenate(pre); post = np.concatenate(post)
            union[f"shunt_{name}"] = {"pre_indices": pre, "post_indices": post,
                                      "initial_weights": np.full(len(pre), self.shunt_syn_w, np.float32),
                                      "plastic": False, "conn_type": "I_TO_E"}
        b.inject_explicit_wiring(union, output_inhibitory_indices=self.shunt_idx.tolist())

    def _reset(self):
        b = self._b
        xp, _ = get_backend()
        v_start = self._v_rest if getattr(self, "_v_rest", None) is not None else self._v0
        b.cp_membrane_potential_v[:] = (xp.asarray(v_start, dtype=b.cp_membrane_potential_v.dtype)
                                        if v_start is not None else -65.0)
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for nm in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                   "cp_conductance_g_nmda_rise", "cp_conductance_g_gabab"):
            arr = getattr(b, nm, None)
            if arr is not None:
                arr[:] = 0.0

    def _calibrate_rest(self):
        """Measure each gate neuron's zero-drive RESTING potential (once), store it as the read baseline, and PIN the
        per-neuron inhibitory reversal of the gate pools to their OWN rest so the shunt is PURELY divisive (no zero
        shift). rest is shunt-independent (E_shunt=rest), so the baseline is valid at every per-token shunt level."""
        b = self._b; xp, _ = get_backend()
        self._v_rest = None
        b.cp_membrane_potential_v[:] = self._v0 if self._v0 is not None else -65.0
        b.cp_recovery_variable_u[:] = 0.0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        for nm in ("cp_conductance_g_e", "cp_conductance_g_i"):
            arr = getattr(b, nm, None)
            if arr is not None:
                arr[:] = 0.0
        b.cp_external_input_current[:] = 0.0
        settle = int(self.read_window * self.settle_frac); n_acc = 0
        n = b.core_config.num_neurons
        vsum = np.zeros(n)
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            vsum += np.asarray(to_host(b.cp_membrane_potential_v)).astype(np.float64)
            n_acc += 1
        n_acc = max(1, n_acc)
        v_rest = vsum / n_acc
        self._v_rest = v_rest.astype(np.float32)
        self.base_on = v_rest[self.gon_idx].copy()
        self.base_off = v_rest[self.goff_idx].copy()
        # pin the gate pools' inhibitory reversal to their own rest (pure shunting)
        rev = getattr(b, "cp_syn_reversal_potential_i_per_neuron", None)
        if rev is not None and rev.size == n:
            gpn = np.concatenate([self.gon_idx, self.goff_idx])
            rev_host = np.asarray(to_host(rev)).astype(np.float64)
            rev_host[gpn] = v_rest[gpn]
            b.cp_syn_reversal_potential_i_per_neuron[:] = xp.asarray(rev_host, dtype=rev.dtype)

    def _shunt_drive(self, r_h, lesion_shunt=False, scramble_perm=None):
        """Map r_h in (0,1) -> per-channel shunt DRIVE current so g_shunt ~ (1-r_h)/r_h (the conductance that yields
        the divisive factor r_h under the 1/(1+g) law). r_h enters as a DRIVE only — never multiplied into hpre."""
        rh = np.clip(np.asarray(r_h, np.float64), 0.05, 1.0)         # floor: r_h<0.05 all -> ~full suppression
        sd = self.shunt_gain * (1.0 / rh - 1.0)                      # ~ shunt_gain*(1-r_h)/r_h  [D]  (0 at r_h=1)
        if lesion_shunt:
            sd = np.zeros_like(sd)
        if scramble_perm is not None:
            sd = sd[scramble_perm]
        return sd

    def gate(self, hpre, r_h, lesion_shunt=False, shunt_scramble_perm=None, want_diag=False):
        """hpre [D] signed (already unit-mapped); r_h [D] in (0,1). Returns the substrate-gated feature h_gated [D]
        (signed) = rate_scale*(Δv_on - Δv_off), the MEMBRANE deflection of the dual ON/OFF pools DIVIDED by the shunt
        conductance. NO host r_h*hpre multiply — r_h drives the shunt, hpre drives the pools, the membrane forms it."""
        b = self._b
        xp, _ = get_backend()
        self._reset()
        sd = self._shunt_drive(r_h, lesion_shunt=lesion_shunt, scramble_perm=shunt_scramble_perm)
        drive = np.zeros(b.core_config.num_neurons, dtype=np.float64)
        drive[self.gon_idx] = self.drive_gain * np.repeat(np.maximum(hpre, 0.0), self.Pg)     # subthreshold +half
        drive[self.goff_idx] = self.drive_gain * np.repeat(np.maximum(-hpre, 0.0), self.Pg)   # subthreshold -half
        drive[self.shunt_idx] = np.repeat(sd, self.Sg)
        b.cp_external_input_current[:] = xp.asarray(drive, dtype=b.cp_external_input_current.dtype)
        settle = int(self.read_window * self.settle_frac)
        n_acc = 0
        von = np.zeros(self.D * self.Pg); voff = np.zeros(self.D * self.Pg); gisum = 0.0; spk = 0.0
        for step in range(self.read_window):
            b._run_one_simulation_step()
            if step < settle:
                continue
            v = np.asarray(to_host(b.cp_membrane_potential_v)).astype(np.float64)
            von += v[self.gon_idx]; voff += v[self.goff_idx]
            if want_diag:
                gisum += float(np.asarray(to_host(b.cp_conductance_g_i)).astype(np.float64)[self.gon_idx].mean())
                fs = np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
                spk += float(fs[self.gon_idx].sum() + fs[self.goff_idx].sum())
            n_acc += 1
        b.cp_external_input_current[:] = 0.0
        n_acc = max(1, n_acc)
        dv_on = ((von / n_acc) - self.base_on).reshape(self.D, self.Pg).mean(axis=1)          # Δv over rest, per chan
        dv_off = ((voff / n_acc) - self.base_off).reshape(self.D, self.Pg).mean(axis=1)
        h_gated = self.rate_scale * (dv_on - dv_off)
        if want_diag:
            return h_gated, dict(dv_on=float(dv_on.mean()), dv_off=float(dv_off.mean()),
                                 g_i=gisum / n_acc, gate_spk=spk / n_acc)
        return h_gated


# ====================================================================================================================
# The read chain with the SUBSTRATE r_h shunt gate in place of the host r_h multiply. Reuses ComposedEndToEndRead
# (composed_biaspop arm) end to end; only _feature is overridden to gate hpre_sub via the shunt instead of `r_h * (...)`.
# ====================================================================================================================
class ShuntGatedRead(ComposedEndToEndRead):
    def __init__(self, ro, seed, proj, gate: ShuntGate, **kw):
        self.gate_net = gate
        super().__init__(ro, seed, proj=proj, use_proj=True, use_bias_pop=True, **kw)

    def _feature(self, ap, an, tid, zero_state=False, zero_feat=False,
                 lesion_shunt=False, shunt_scramble_perm=None, host_gate=False, want_gate_diag=False):
        ro = self.ro
        state = np.concatenate([ap, an])
        hpre_sub, _ = self.proj._graded_hpre(state, zero_state=zero_state)     # SUBSTRATE Wo_sp@state [D]
        r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))            # host r_h VALUE (declared residual)
        hpre_scaled = self.proj_out_scale * hpre_sub                          # unit-map -> read-out feature scale
        if host_gate:
            h = r_h * hpre_scaled                                             # arm B: HOST multiply (the 0.9137 GO)
            diag = None
        else:
            out = self.gate_net.gate(hpre_scaled, r_h, lesion_shunt=lesion_shunt,
                                     shunt_scramble_perm=shunt_scramble_perm, want_diag=want_gate_diag)
            h, diag = out if want_gate_diag else (out, None)                  # arm A: SUBSTRATE shunt gate
        if zero_feat:
            h = np.zeros_like(h)
        feat = np.concatenate([np.maximum(h, 0.0), np.maximum(-h, 0.0)])
        if want_gate_diag:
            return feat, (r_h * hpre_scaled), diag
        return feat

    def read_endtoend(self, ap, an, tid, scramble_perm=None, zero_state=False, zero_feat=False,
                      silence_bias=False, lesion_shunt=False, shunt_scramble_perm=None, host_gate=False):
        feat = self._feature(ap, an, tid, zero_state=zero_state, zero_feat=zero_feat,
                             lesion_shunt=lesion_shunt, shunt_scramble_perm=shunt_scramble_perm, host_gate=host_gate)
        margin, ge, gi, psp, bsp = self._graded_margin(feat, want_diag=True, silence_bias=silence_bias)
        margin_pos = self.df_e * ge
        if scramble_perm is not None:
            margin = margin[scramble_perm]; margin_pos = margin_pos[scramble_perm]
        return dict(win=self._argwin(margin), margin=margin, win_pos=self._argwin(margin_pos),
                    margin_pos=margin_pos, pool_sp=psp, bias_sp=bsp)


def _host_full_ref(ro, ap, an, tid):
    """The full host mouth read (reference), from the exact host state."""
    lg = ro.logits(ap, an, tid).copy()
    if ro.unk_idx >= 0:
        lg[ro.unk_idx] = -1e30
    return lg


# ====================================================================================================================
def _gate_fidelity(seed, ro, s_wk, scale, off, s_read, ev_ids, warmup, n_pos):
    """corr(substrate-gated feat, host r_h*hpre split) — does the shunt genuinely realize the gate?"""
    D = ro.D
    fc = []; on_off_ratio = []
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state()
        for t in range(len(ids) - 1):
            rate = _rates(ro, ids[t])
            g = s_wk.advance(rate)
            cst = _cal_state(g, scale, off)
            if t < warmup:
                continue
            ap_s, an_s = cst[:D], cst[D:]
            feat, ref, diag = s_read._feature(ap_s, an_s, ids[t], want_gate_diag=True)
            ref_split = np.concatenate([np.maximum(ref, 0.0), np.maximum(-ref, 0.0)])
            fc.append(_corr(feat, ref_split))
            on_off_ratio.append(abs(diag["dv_on"]) + abs(diag["dv_off"]))
            positions += 1
            if positions >= n_pos:
                break
        if positions >= n_pos:
            break
    return (float(np.mean(fc)) if fc else 0.0, float(np.mean(on_off_ratio)) if on_off_ratio else 0.0)


def _eval(seed, ro, s_wk, scale, off, s_read, ev_ids, warmup, n_eval_pos, deep_lo, n_ac):
    D = ro.D; V = ro.V; chance = 1.0 / V

    def _z():
        return dict(n=0, agree=0.0, agree_pos=0.0, mass_read=0.0, mass_ax=0.0, deep_n=0, deep_agree=0.0)
    A = _z(); B = _z()
    pool_sp = 0.0; bias_sp = 0.0
    ac = dict(n=0, scr=0.0, zstate=0.0, zfeat=0.0, lesion=0.0, sscr=0.0)

    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state(); ap_h = np.zeros(D); an_h = np.zeros(D)
        for t in range(len(ids) - 1):
            rate = _rates(ro, ids[t])
            g = s_wk.advance(rate)
            cst = _cal_state(g, scale, off)
            ap_s, an_s = cst[:D], cst[D:]
            ap_h, an_h = _ref_advance(ro, ap_h, an_h, ids[t])
            if t < warmup:
                continue
            tid = ids[t]
            lg_h = _host_full_ref(ro, ap_h, an_h, tid)
            host_am = int(np.argmax(lg_h)); pfull = _softmax(lg_h)
            deep = (t >= warmup + deep_lo)

            rA = s_read.read_endtoend(ap_s, an_s, tid)                          # arm A: substrate shunt gate
            rB = s_read.read_endtoend(ap_s, an_s, tid, host_gate=True)          # arm B: host r_h multiply (== 0.9137 GO)
            pool_sp += rA["pool_sp"]; bias_sp += rA["bias_sp"]
            for arm, r in ((A, rA), (B, rB)):
                arm["n"] += 1
                arm["agree"] += float(r["win"] == host_am)
                arm["agree_pos"] += float(r["win_pos"] == host_am)
                arm["mass_read"] += (pfull[r["win"]] if r["win"] >= 0 else 0.0)
                arm["mass_ax"] += pfull[host_am]
                if deep:
                    arm["deep_n"] += 1
                    arm["deep_agree"] += float(r["win"] == host_am)

            if ac["n"] < n_ac:
                scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(V)
                sscr_perm = np.random.default_rng(seed * 47 + 7 + positions).permutation(D)
                ac["scr"] += float(s_read.read_endtoend(ap_s, an_s, tid, scramble_perm=scr_perm)["win"] == host_am)
                ac["zstate"] += float(s_read.read_endtoend(ap_s, an_s, tid, zero_state=True)["win"] == host_am)
                ac["zfeat"] += float(s_read.read_endtoend(ap_s, an_s, tid, zero_feat=True)["win"] == host_am)
                ac["lesion"] += float(s_read.read_endtoend(ap_s, an_s, tid, lesion_shunt=True)["win"] == host_am)
                ac["sscr"] += float(s_read.read_endtoend(ap_s, an_s, tid, shunt_scramble_perm=sscr_perm)["win"] == host_am)
                ac["n"] += 1

            positions += 1
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break

    void_if(A["n"] == 0, "no evaluable positions")

    # state-corruption anti-cheats on arm A (fresh advance; zero-input + memoryless)
    zi = ml = ac2_n = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state(); ap2 = np.zeros(D); an2 = np.zeros(D)
        for t in range(min(len(ids) - 1, warmup + 25)):
            rate = _rates(ro, ids[t])
            g_zi = s_wk.advance(rate, zero_input=True)
            g_ml = s_wk.advance(rate, memoryless=True)
            ap2, an2 = _ref_advance(ro, ap2, an2, ids[t])
            if t < warmup:
                continue
            tid = ids[t]
            lg_h = _host_full_ref(ro, ap2, an2, tid); host_am = int(np.argmax(lg_h))
            cst_zi = _cal_state(g_zi, scale, off); cst_ml = _cal_state(g_ml, scale, off)
            zi += int(s_read.read_endtoend(cst_zi[:D], cst_zi[D:], tid)["win"] == host_am)
            ml += int(s_read.read_endtoend(cst_ml[:D], cst_ml[D:], tid)["win"] == host_am)
            ac2_n += 1
            if ac2_n >= n_ac:
                break
        if ac2_n >= n_ac:
            break

    def _fin(arm):
        n = max(1, arm["n"]); dn = max(1, arm["deep_n"])
        return dict(n=arm["n"], argmax_agree=round(arm["agree"] / n, 4),
                    argmax_agree_positive_only=round(arm["agree_pos"] / n, 4),
                    mass_read=round(arm["mass_read"] / n, 4), mass_argmax_ceiling=round(arm["mass_ax"] / n, 4),
                    recov_argmax=round((arm["mass_read"] / n) / max(1e-9, arm["mass_ax"] / n), 4),
                    deep_n=arm["deep_n"], deep_argmax_agree=round(arm["deep_agree"] / dn, 4))

    mA = _fin(A); mB = _fin(B)
    nac = max(1, ac["n"]); nac2 = max(1, ac2_n)
    m = {
        "seed": seed, "V": V, "D": D, "chance_1_over_v": round(chance, 6),
        "n_positions": A["n"], "mean_pool_spikes": round(pool_sp / max(1, A["n"]), 3),
        "gate_pool": s_read.gate_net.Pg, "shunt_pop": s_read.gate_net.Sg,
        "drive_gain": s_read.gate_net.drive_gain, "shunt_gain": s_read.gate_net.shunt_gain,
        "rate_scale": s_read.gate_net.rate_scale,
        "host_rng_draws_on_read_path": int(s_read.n_host_rng_draws),
        "host_gate_mult_on_gate_path": int(s_read.gate_net.n_host_gate_mult),
        "shuntgate": mA, "hostgate": mB,
        "argmax_agree_scramble": round(ac["scr"] / nac, 4),
        "argmax_agree_zerostate": round(ac["zstate"] / nac, 4),
        "argmax_agree_zerofeat": round(ac["zfeat"] / nac, 4),
        "argmax_agree_lesion_shunt": round(ac["lesion"] / nac, 4),
        "argmax_agree_shunt_scramble": round(ac["sscr"] / nac, 4),
        "argmax_agree_zeroinput": round(zi / nac2, 4),
        "argmax_agree_memoryless": round(ml / nac2, 4),
        "n_anticheat": ac["n"], "n_anticheat_state": ac2_n,
    }
    fc, gate_rate = _gate_fidelity(seed, ro, s_wk, scale, off, s_read, ev_ids, warmup, min(n_ac, 80))
    m["gate_fidelity_corr"] = round(fc, 4); m["mean_gate_rate"] = round(gate_rate, 4)
    lever("shuntgate_recov_vs_hostgate", before=mB["recov_argmax"], after=mA["recov_argmax"], required=False)
    lever("shuntgate_argmax_vs_lesionshunt", before=m["argmax_agree_lesion_shunt"],
          after=mA["argmax_agree"], required=False)
    return m


def _scramble_at_chance(a, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return a <= chance + 3.0 * sigma


def _verdict(m):
    mA = m["shuntgate"]; mB = m["hostgate"]
    chance = m["chance_1_over_v"]; n = m["n_positions"]; nac = max(1, m["n_anticheat"])
    aa = mA["argmax_agree"]
    checks = {
        # the substrate shunt gate reproduces the host next-word mass (>=0.70 recov, the pipeline's own bar).
        "recov_argmax_ge_0.70": mA["recov_argmax"] >= 0.70,
        # holds NEAR the host-gate arm (== the 0.9137 GO): the shunt genuinely realizes r_h (penalty bounded).
        "within_tol_of_hostgate": mA["recov_argmax"] >= mB["recov_argmax"] - 0.08,
        "argmax_agree_gt_10x_chance": aa > 10 * chance,
        # the gate reconstructs r_h*hpre (fidelity).
        "gate_fidelity_ge_0.85": m["gate_fidelity_corr"] >= 0.85,
        # LESION THE SHUNT -> the gate is lost -> the pipeline moves off the host-gate arm (shunt load-bearing).
        "shunt_load_bearing": mA["argmax_agree"] - m["argmax_agree_lesion_shunt"] > 0.03,
        # SCRAMBLE THE SHUNT->channel map -> degrade (the labelled-line r_h map carries the gate).
        "shunt_scramble_degrades": mA["argmax_agree"] - m["argmax_agree_shunt_scramble"] > 0.02,
        # the SUBSTRATE state still drives the whole chain (zero its input -> collapse).
        "state_drives_chain": aa - m["argmax_agree_zeroinput"] > 0.30,
        "recurrence_load_bearing": aa - m["argmax_agree_memoryless"] > 0.10,
        # lesion either read stage -> collapse.
        "read_input_collapses": max(m["argmax_agree_zerostate"], m["argmax_agree_zerofeat"]) <= 0.34 * aa,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, nac),
        "signed_beats_positive_only": aa > mA["argmax_agree_positive_only"],
        # provenance: 0 host draws + 0 host r_h*hpre multiply on the gate path.
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "provenance_no_host_gate_mult": m["host_gate_mult_on_gate_path"] == 0,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    m["checks"] = checks
    m["GO"] = bool(all(checks.values()))
    return m


def _unit_rate_scale(ro, s_wk, proj, gate, scale, off, ev_ids, warmup, proj_out_scale, n=24):
    """Unit-map the gate mV deflection -> host feature scale: at r_h=1 (ungated, shunt drive 0) the gate should
    reproduce hpre_scaled. rate_scale = sum|hpre_scaled| / sum|Δv_diff| over n positions (a single scalar, seed 42)."""
    gate.rate_scale = 1.0
    num = 0.0; den = 0.0; positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state()
        for t in range(len(ids) - 1):
            g = s_wk.advance(_rates(ro, ids[t])); cst = _cal_state(g, scale, off)
            if t < warmup:
                continue
            hpre_sub, _ = proj._graded_hpre(np.concatenate([cst[:ro.D], cst[ro.D:]]))
            hpre_scaled = proj_out_scale * hpre_sub
            h_raw = gate.gate(hpre_scaled, np.ones(ro.D))                 # ungated Δv-diff
            msk = np.abs(hpre_scaled) > 1e-6
            if msk.any():
                num += float(np.sum(np.abs(hpre_scaled[msk]))); den += float(np.sum(np.abs(h_raw[msk])))
            positions += 1
            if positions >= n:
                break
        if positions >= n:
            break
    return num / max(1e-9, den)


def _build(ro, seed, args):
    s_wk = GradedRecurrentState(ro.D, seed, t_step=args.t_step, carrier_pop=args.carrier_pop,
                                ou_std=args.wk_ou_std, drive_gain=args.wk_drive_gain,
                                drive_bias=args.wk_drive_bias, syn_w=args.wk_syn_w, ssm_decay=ro.decay)
    proj = _build_proj(ro, seed, args)
    gate = ShuntGate(ro.D, seed, pool=args.gate_pool, shunt_pop=args.shunt_pop, read_window=args.gate_window,
                     settle_frac=args.gate_settle_frac, ou_std=args.ou_std, drive_gain=args.gate_drive_gain,
                     shunt_gain=args.shunt_gain, shunt_syn_w=args.shunt_syn_w,
                     rate_scale=(args.rate_scale if args.rate_scale > 0 else 1.0))
    s_read = ShuntGatedRead(ro, seed, proj=proj, gate=gate, hb_k=0.0, bias_scale=args.bias_scale,
                            n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA, proj_out_scale=args.proj_out_scale,
                            pop=args.pop, hid_pop=4, ou_std=args.ou_std, read_window=args.read_window,
                            hid_gain=args.hid_gain, ratio=args.ratio)
    s_read._arm = "shuntgate"
    return s_wk, proj, gate, s_read


def _calibrate(ro, seed, ev_ids, args):
    """Calibrate (rate_scale, shunt_gain) ONCE on seed 42. (a) rate_scale: at r_h=1 (no shunt) match the host feature
    scale; (b) shunt_gain: maximize the gate fidelity corr(feat, r_h*hpre split). Prints the plateau."""
    s_wk = GradedRecurrentState(ro.D, seed, t_step=args.t_step, carrier_pop=args.carrier_pop, ou_std=args.wk_ou_std,
                                drive_gain=args.wk_drive_gain, drive_bias=args.wk_drive_bias, syn_w=args.wk_syn_w,
                                ssm_decay=ro.decay)
    proj = _build_proj(ro, seed, args)
    calib = _fit_calib(s_wk, ro, ev_ids, args.warmup, min(400, args.n_eval_pos), ro.decay)
    scale, off = calib
    # (a) rate_scale: unit-map the ungated gate rate to the host feature scale (r_h=1 -> no shunt).
    print("[calib] rate_scale unit-map (r_h=1, no shunt):", flush=True)
    gate = ShuntGate(ro.D, seed, pool=args.gate_pool, shunt_pop=args.shunt_pop, read_window=args.gate_window,
                     settle_frac=args.settle_frac, ou_std=args.ou_std, drive_gain=args.gate_drive_gain,
                     shunt_gain=0.0, shunt_syn_w=args.shunt_syn_w, rate_scale=1.0)
    num = []; den = []
    positions = 0
    for ids in ev_ids:
        if len(ids) < args.warmup + 2:
            continue
        s_wk.reset_state()
        for t in range(len(ids) - 1):
            g = s_wk.advance(_rates(ro, ids[t])); cst = _cal_state(g, scale, off)
            if t < args.warmup:
                continue
            state_sub = cst
            hpre_sub, _ = proj._graded_hpre(np.concatenate([state_sub[:ro.D], state_sub[ro.D:]]))
            hpre_scaled = args.proj_out_scale * hpre_sub
            h_raw = gate.gate(hpre_scaled, np.ones(ro.D))                 # ungated rate diff
            m = np.abs(hpre_scaled) > 1e-6
            if m.any():
                num.append(float(np.sum(np.abs(hpre_scaled[m])))); den.append(float(np.sum(np.abs(h_raw[m]))))
            positions += 1
            if positions >= 40:
                break
        if positions >= 40:
            break
    rate_scale = (sum(num) / max(1e-9, sum(den)))
    print(f"    rate_scale = {rate_scale:.4f}", flush=True)
    # (b) shunt_gain sweep -> gate fidelity
    print("[calib] shunt_gain sweep (gate fidelity corr(feat, r_h*hpre split)):", flush=True)
    best = None
    for sg in [float(x) for x in args.calib_shunt_gains.split(",")]:
        gate2 = ShuntGate(ro.D, seed, pool=args.gate_pool, shunt_pop=args.shunt_pop, read_window=args.gate_window,
                          settle_frac=args.settle_frac, ou_std=args.ou_std, drive_gain=args.gate_drive_gain,
                          shunt_gain=sg, shunt_syn_w=args.shunt_syn_w, rate_scale=rate_scale)
        s_read = ShuntGatedRead(ro, seed, proj=proj, gate=gate2, hb_k=0.0, bias_scale=args.bias_scale,
                                n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA,
                                proj_out_scale=args.proj_out_scale, pop=args.pop, hid_pop=4, ou_std=args.ou_std,
                                read_window=args.read_window, hid_gain=args.hid_gain, ratio=args.ratio)
        s_read._arm = "shuntgate"
        fc, gr = _gate_fidelity(seed, ro, s_wk, scale, off, s_read, ev_ids, args.warmup, 60)
        print(f"    shunt_gain={sg:7.1f} -> gate_fidelity={fc:.4f} mean_gate_rate={gr:.4f}", flush=True)
        if best is None or fc > best[1]:
            best = (sg, fc)
    print(f"[calib] rate_scale={rate_scale:.4f}  BEST shunt_gain={best[0]} (fidelity {best[1]:.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--deep-lo", type=int, default=8)
    ap.add_argument("--n-anticheat", type=int, default=120)
    # [WK] state operating point (seed-42 GO values)
    ap.add_argument("--t-step", type=int, default=40)
    ap.add_argument("--carrier-pop", type=int, default=24)
    ap.add_argument("--wk-drive-gain", type=float, default=40.0)
    ap.add_argument("--wk-drive-bias", type=float, default=40.0)
    ap.add_argument("--wk-syn-w", type=float, default=2.0)
    ap.add_argument("--wk-ou-std", type=float, default=60.0)
    # [CE] read-chain operating point (seed-42 GO values)
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--proj-drive-gain", type=float, default=120.0)
    ap.add_argument("--proj-syn-scale", type=float, default=12.0)
    ap.add_argument("--proj-ratio", type=float, default=0.5)
    ap.add_argument("--proj-out-scale", type=float, default=0.30)
    ap.add_argument("--bias-scale", type=float, default=0.14)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    # ---- the r_h SHUNT GATE operating point (MEMBRANE differential read; calibrated ONCE on seed 42) ----
    ap.add_argument("--gate-pool", type=int, default=8)
    ap.add_argument("--shunt-pop", type=int, default=4)
    ap.add_argument("--gate-window", type=int, default=180)
    ap.add_argument("--gate-settle-frac", type=float, default=0.3)
    ap.add_argument("--gate-drive-gain", type=float, default=25.0)     # keep gate pools near/below threshold (hpre absmax ~4.8)
    ap.add_argument("--shunt-gain", type=float, default=300.0)         # strong shunt: r_h median 0.27 needs deep division
    ap.add_argument("--shunt-syn-w", type=float, default=25.0)
    ap.add_argument("--rate-scale", type=float, default=-1.0)   # <=0 -> auto unit-map on the first seed (mV->feature)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-shunt-gains", type=str, default="40,70,110,160,220,300")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_rh_shunt.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_sentences = 2000; args.n_eval_pos = min(args.n_eval_pos, 60); args.n_anticheat = min(args.n_anticheat, 40)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    calib = None
    rate_scale_auto = [None]
    t_all = time.time()

    if args.calib:
        seed = seeds[0]
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        ro = WKVReadout(ckpt)
        ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        _calibrate(ro, seed, ev_ids, args)
        return

    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        s_wk, proj, gate, s_read = _build(ro, seed, args)
        if calib is None:
            calib = _fit_calib(s_wk, ro, ev_ids, args.warmup, min(600, args.n_eval_pos), ro.decay)
            print(f"[calib-state on seed {seed}] scale.mean={calib[0].mean():.4f}", flush=True)
        scale, off = calib
        # AUTO unit-map the gate mV deflection -> host feature scale ONCE (r_h=1 => ungated), unless pinned via --rate-scale.
        if args.rate_scale <= 0 and rate_scale_auto[0] is None:
            rate_scale_auto[0] = _unit_rate_scale(ro, s_wk, proj, gate, scale, off, ev_ids, args.warmup,
                                                  args.proj_out_scale, n=24)
            print(f"[calib-rate on seed {seed}] rate_scale={rate_scale_auto[0]:.4f}", flush=True)
        if rate_scale_auto[0] is not None:
            gate.rate_scale = rate_scale_auto[0]

        t0 = time.time()
        m = _verdict(_eval(seed, ro, s_wk, scale, off, s_read, ev_ids, args.warmup, args.n_eval_pos,
                           args.deep_lo, args.n_anticheat))
        m["secs"] = round(time.time() - t0, 1)
        results.append(m)
        mA = m["shuntgate"]; mB = m["hostgate"]
        print(f"[seed {seed}] shuntgate recov={mA['recov_argmax']:.4f} agree={mA['argmax_agree']:.4f}"
              f">pos{mA['argmax_agree_positive_only']:.3f} | hostgate recov={mB['recov_argmax']:.4f}"
              f" agree={mB['argmax_agree']:.4f} | gate_fid={m['gate_fidelity_corr']:.3f} rate={m['mean_gate_rate']:.3f}"
              f" | lesion={m['argmax_agree_lesion_shunt']:.3f} sscr={m['argmax_agree_shunt_scramble']:.3f}"
              f" zin={m['argmax_agree_zeroinput']:.3f} zstate={m['argmax_agree_zerostate']:.3f}"
              f" scr={m['argmax_agree_scramble']:.3f} | GO={m['GO']} ({sum(m['checks'].values())}/{len(m['checks'])})"
              f" ({m['secs']}s)", flush=True)
        if not m["GO"]:
            print(f"    checks: {json.dumps(m['checks'])}", flush=True)

    rows = [r for r in results if "shuntgate" in r]
    summary = {}
    if rows:
        summary = {
            "n_seeds": len(rows),
            "go_count": int(sum(1 for r in rows if r.get("GO"))),
            "shuntgate_recov_mean": round(float(np.mean([r["shuntgate"]["recov_argmax"] for r in rows])), 4),
            "shuntgate_recov_min": round(float(np.min([r["shuntgate"]["recov_argmax"] for r in rows])), 4),
            "shuntgate_argmax_agree_mean": round(float(np.mean([r["shuntgate"]["argmax_agree"] for r in rows])), 4),
            "hostgate_recov_mean": round(float(np.mean([r["hostgate"]["recov_argmax"] for r in rows])), 4),
            "gate_fidelity_mean": round(float(np.mean([r["gate_fidelity_corr"] for r in rows])), 4),
            "lesion_shunt_mean": round(float(np.mean([r["argmax_agree_lesion_shunt"] for r in rows])), 4),
            "shunt_scramble_mean": round(float(np.mean([r["argmax_agree_shunt_scramble"] for r in rows])), 4),
            "zeroinput_mean": round(float(np.mean([r["argmax_agree_zeroinput"] for r in rows])), 4),
            "scramble_mean": round(float(np.mean([r["argmax_agree_scramble"] for r in rows])), 4),
        }
    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "n_eval_pos": args.n_eval_pos, "plasticity_off": True,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t_all, 1),
           "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows -> {args.json} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
