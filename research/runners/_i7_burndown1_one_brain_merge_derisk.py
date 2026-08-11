"""INTEGRATION #7 burn-down #1 -- the ONE-BRAIN MERGE (the mission's core non-negotiable).

Today INTEGRATION #7 (`_teacher_loop_facts_into_live_chat_derisk`) runs TWO `SimulationBridge` objects in ONE process:
  (1) the conversational bridge from `SA.build_one_brain` (~26.3K neurons: composer rf + every faculty slice);
  (2) the e-prop ACQUISITION net `OnBridgeEpropNet`, which builds its OWN ~80-neuron bridge (in 34 | H1 40 | out 6,
      plastic feedforward `cp_connections`).
(The learned familiarity gate is pure numpy -- NOT a bridge -- and stays as-is.)

BURN-DOWN #1 relocates bridge (2)'s ~80-neuron slices ONTO bridge (1): ONE `SimulationBridge`, ONE `cp_connections`
hosting BOTH the conversational synapses and the e-prop feedforward synapses (substrate-level one-brain, per
`project_one_brain_substrate_vs_functional`). It reuses the proven append-LAST nav+conv-merge / seams-A/C pattern.

THE DESIGN (runner-side; the ONLY `sim/`-adjacent change is an additive/default-off flag on `SA.build_one_brain`,
a research runner -- NO `sim/` edit):
  1. `SA.build_one_brain(..., co_resident_eprop=True)` appends THREE `BrainRegion` slices LAST -- eprop_in(34),
     eprop_h1(40), eprop_out(6) -- each internal_density=0 / enable_nmda=False / RS Izhikevich, and injects the TWO
     plastic FF pathways (eprop_ff_0 in->h1 Xavier, eprop_ff_1 h1->out zero-init) into the SAME `union` dict +
     the SINGLE `inject_explicit_wiring` call. Append-LAST preserves the pre-existing neurons' threshold draws
     (byte-identity) -- the stageA seams-A/C / #3c-opponent invariance.
  2. `CoResidentEpropNet` (below) subclasses `OnBridgeEpropNet` but SKIPS the parent bridge build: it binds
     `self.br = merged_bridge`, its layer slices = the GLOBAL eprop slices, rebuilds `_data_idx_flat` from the merged
     bridge's cached COO via a SPARSE position map (the dense n_total x n_total int64 posmat is ~5.5GB at 26.4K
     neurons), and drives ONLY the eprop_in slice. Forward/train step the WHOLE bridge, read on the eprop slices, and
     WASH OUT to the shared quiescent baseline_snap each micro-forward -- the exact co-residency discipline of SEAM-A
     `read_forward_model` (so an e-prop micro-forward never leaves the conversational neurons clobbered).
  3. `enable_bdsp` stays OFF on the merged bridge (byte-identity) -> `cp_bdsp_E` is None -> the (unused, in the
     leaky_readout config #7 uses) `cp_bdsp_E` read in `_forward_record` is GUARDED.

GO / NO-GO (this file's first job): `byte_identity_eprop(seed)` builds the brain WITH vs WITHOUT the appended eprop
slices and asserts the conversational neurons' threshold hash + the decision transcript are byte-identical. If it
HOLDS the merge is runner-side; if it FAILS we STOP and report the perturbation + the minimal additive/default-off
sim/ hook (the coordinator decides).

HONEST SCOPE: this reaches CO-RESIDENCY -- disjoint slices of ONE bridge / ONE cp_connections -- NOT yet cross-region
synaptic INTERACTION (there is ZERO conv<->eprop synapse; co-location with no cross-synapse is not full one-brain,
per project_one_brain_substrate_vs_functional). It IS a real step: two separate bridge objects -> one bridge hosting
both. The FURTHER step -- a synaptic pathway conv-cue -> eprop_in AND eprop_out-spikes -> composer render -- is a
SEPARATE arc. Remaining scaffolds (unchanged, named): the numpy familiarity gate, the argmax patient read-out, the
host leaky-readout integration, the AI-teacher presentation.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import logging as _logging  # noqa: E402

for _n in ("SIM_BRIDGE", "sim.bridge", "root"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

import hashlib  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
import sim.bridge as _simbridge  # noqa: E402 -- for the CONTINGENCY heterogeneity hook prototype (see below)

from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners import _corpus_facts_into_live_chat_derisk as CF  # noqa: E402
from research.runners import _teacher_loop_facts_into_live_chat_derisk as I7  # noqa: E402
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
from sim.bptt_snn_gpu import atan_surrogate  # noqa: E402 -- the SAME surrogate the validated e-prop uses
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The CO-RESIDENT e-prop net: OnBridgeEpropNet forward + e-prop update, but bound to the MERGED conversational
# bridge (NO own bridge build). Layer slices are the GLOBAL eprop_* index ranges; the sparse position map + the
# slice-scoped (shared-baseline) wash-out + the guarded cp_bdsp_E read are the co-residency deltas.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
class CoResidentEpropNet(OnBridgeEpropNet):
    def __init__(self, merged_bridge, baseline_snap, eprop_idx, n_in, hidden, k, seed,
                 settle=25, eprop_lr=0.5, w_clip=4000.0, hp=None):
        # NOTE: deliberately does NOT call super().__init__ (that builds a fresh ~80-neuron SimulationBridge). We
        # replicate every attribute OnBridgeBDSPNet.__init__ + OnBridgeEpropNet.__init__ set, but with self.br bound
        # to the MERGED bridge and the layer slices = the GLOBAL eprop_* ranges.
        hp = hp or {}
        self._xp, _ = get_backend()
        xp = self._xp
        self.br = merged_bridge
        self.cfg = merged_bridge.core_config
        self._baseline_snap = baseline_snap

        # ---- geometry (OnBridgeBDSPNet.__init__), slices GLOBAL onto the merged bridge (pool_k=1 -> phys == logical) ----
        self.rule = "plain_fa"
        self.n_in = int(n_in); self.hidden = int(hidden); self.k = int(k)
        self.n_hidden_layers = 1
        self.settle_steps = int(settle); self.credit_steps = 0
        self.in_current_pA = float(hp.get("in_current_pA", 700.0))
        self.in_bias_pA = float(hp.get("in_bias_pA", 300.0))
        self.apical_gain_pA = float(hp.get("apical_gain_pA", 2000.0))
        self.lr = 0.0
        self.pool_k = 1
        self.tonic_h_pA = float(hp.get("tonic_h_pA", 100.0))
        self.tonic_o_pA = float(hp.get("tonic_o_pA", 150.0))
        self.sizes = [self.n_in, self.hidden, self.k]
        self.sizes_phys = [s * self.pool_k for s in self.sizes]

        g_in = np.asarray(eprop_idx["in"], dtype=np.int64)
        g_h1 = np.asarray(eprop_idx["h1"], dtype=np.int64)
        g_out = np.asarray(eprop_idx["out"], dtype=np.int64)
        for nm, arr in (("in", g_in), ("h1", g_h1), ("out", g_out)):
            if not np.array_equal(arr, np.arange(int(arr[0]), int(arr[-1]) + 1)):
                raise RuntimeError(f"eprop_{nm} slice is not contiguous -- append-LAST region layout broken")
        self.slices = [slice(int(g_in[0]), int(g_in[-1]) + 1),
                       slice(int(g_h1[0]), int(g_h1[-1]) + 1),
                       slice(int(g_out[0]), int(g_out[-1]) + 1)]
        # drive/sp/vv address the FULL merged neuron space (the whole brain steps); reads/credit are slice-scoped.
        self.n_total = int(merged_bridge.core_config.num_neurons)
        self._ff_edges = [(g_in, g_h1), (g_h1, g_out)]

        # ---- e-prop hyperparams (OnBridgeEpropNet.__init__) ----
        self.eprop_lr = float(eprop_lr); self.eps_leak = float(hp.get("eps_leak", 0.9))
        self.surrogate = str(hp.get("surrogate", "atan_vt")); self.alpha_surr = float(hp.get("alpha_surr", 0.15))
        self.beta_surr = float(hp.get("beta_surr", 1.0))
        self.logit_source = "leaky_readout"; self.w_clip = float(w_clip); self.reset_state = True
        self.train_layers = None
        self.output_psi_one = True
        self.logit_temp = 1.0
        self._r_mu = None; self._r_sigma = None
        self.hidden_lr_scale = float(hp.get("hidden_lr_scale", 1.0))
        self._psi_peak = float(atan_surrogate(np.array([0.0]), alpha=self.alpha_surr, xp=np)[0])
        frng = np.random.default_rng(seed + 8888)
        self.B_direct = [frng.normal(0.0, 1.0 / np.sqrt(self.k), (self.k, self.sizes_phys[li + 1])).astype(np.float64)
                         for li in range(len(self.sizes) - 2)]
        # vt over the FULL merged bridge so `self.vt[post_sl]` (post_sl is a GLOBAL slice) reads the eprop thresholds.
        self.vt = np.asarray(to_host(self.br.cp_izh_vt), dtype=np.float64)

        # ---- SPARSE position map (refactor a): {(row,col) -> cp_connections.data slot} over the COO nnz. The dense
        #      n_total x n_total int64 posmat is ~5.5GB at 26.4K neurons; the COO has only ~93K nnz. tocoo(copy=False)
        #      preserves CSR row-major order, so coo.row[i]/col[i] align with cp_connections.data[i]. ----
        coo = self.br._get_cached_coo()
        row = np.asarray(to_host(coo.row)).astype(np.int64)
        col = np.asarray(to_host(coo.col)).astype(np.int64)
        pos = {(int(row[i]), int(col[i])): i for i in range(row.shape[0])}
        self._data_idx_flat = []
        for (pre, post) in self._ff_edges:
            idx2d = np.empty((len(pre), len(post)), dtype=np.int64)
            for ai, a in enumerate(pre):
                for bi, b in enumerate(post):
                    idx2d[ai, bi] = pos.get((int(a), int(b)), -1)
            if (idx2d < 0).any():
                raise RuntimeError("FF pathway edge missing from cp_connections (sparse position map failed)")
            self._data_idx_flat.append(xp.asarray(idx2d.ravel()))

        # RE-INITIALIZE the eprop FF weights to a CLEAN start every construction (each `_teach` = a fresh net, exactly
        # as the standalone builds a fresh bridge). Xavier * ff_w_init for ff_0; ZERO for the leaky-readout ff_1.
        # Writes ONLY the eprop FF slots -> conversational synapses untouched. Ordering: idx2d.ravel() is (pre)-major,
        # so W.ravel() (row=pre-major) aligns slot-for-slot.
        ff_w_init = float(hp.get("ff_w_init", 2000.0))
        rng = np.random.default_rng(seed)
        lim0 = float(np.sqrt(6.0 / (self.sizes_phys[0] + self.sizes_phys[1])))
        W0 = (rng.uniform(-lim0, lim0, (self.sizes_phys[0], self.sizes_phys[1])) * ff_w_init).astype(np.float32)
        data = self.br.cp_connections.data
        data[self._data_idx_flat[0]] = xp.asarray(W0.ravel())
        data[self._data_idx_flat[1]] = xp.asarray(np.zeros(int(self._data_idx_flat[1].shape[0]), dtype=np.float32))

    # ---- co-resident forward: WASH OUT to the shared baseline (SEAM-A discipline) + GUARD the unused cp_bdsp_E read ----
    def _forward_record(self, feat_row, reset_rates=True):
        xp = self._xp
        n = self.n_total
        # SLICE-SCOPED wash-out via the shared quiescent baseline: restores BOTH the eprop slices (clean rest start for
        # this micro-forward's eligibility/surrogate) AND the conversational neurons (never left clobbered mid-turn).
        if self.reset_state:
            SA._restore_state(self.br, self._baseline_snap)
        if reset_rates and self.br.cp_bdsp_E is not None:  # None on the merged bridge (enable_bdsp OFF) -> skipped
            self.br.cp_bdsp_E[...] = 0.0
            self.br.cp_bdsp_B[...] = 0.0
            self.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
        drive = self._base_drive()
        f = np.asarray(feat_row, dtype=np.float32)
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * f, 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        drive_xp = xp.asarray(drive)
        if getattr(self.br, "cp_bdsp_apical_drive", None) is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        T = self.settle_steps
        sp = np.zeros((T, n), dtype=np.float32)
        vv = np.zeros((T, n), dtype=np.float32)
        for t in range(T):
            self.br.cp_external_input_current = drive_xp
            self.br._run_one_simulation_step()
            sp[t] = np.asarray(to_host(self.br.cp_firing_states), dtype=np.float32)
            vv[t] = np.asarray(to_host(self.br.cp_membrane_potential_v), dtype=np.float32)
        # leaky_readout logits come from sp only; acts (which would need cp_bdsp_E, None here) is unused -> dummy zeros.
        acts = [np.zeros(self.sizes[li], dtype=np.float64) for li in range(len(self.sizes))]
        return sp, vv, acts


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Teacher presentation on the MERGED bridge -- mirror of I7._teach, only the net construction swapped.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _mk_merged_net(merged, snap, eprop_idx, seed, freeze=False):
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    net = CoResidentEpropNet(merged, snap, eprop_idx, n_in=I7.N_IN, hidden=I7.HIDDEN, k=I7.K, seed=seed,
                             settle=I7.SETTLE, eprop_lr=I7.EPROP_LR, w_clip=I7.W_CLIP, hp=hp)
    if freeze:
        net.eprop_lr = 0.0
    return net


def _teach_merged(seed, env, merged, eprop_idx, snap, mispaired=False, single_class=False, freeze=False):
    """One teacher presentation on the co-resident e-prop net (== I7._teach, net swapped for CoResidentEpropNet).
    NB the FF weights live on the SHARED merged bridge, so constructing a NEW net RE-INITS them -- do every read
    that needs THIS net's trained weights BEFORE constructing the next net (the smoke's ordering respects this)."""
    net = _mk_merged_net(merged, snap, eprop_idx, seed, freeze=freeze)
    fam = I7._make_fam(seed)
    for r in I7.TAUGHT:
        fam.imprint(env, r, "eats")
    ro0 = I7._readout_norm(net)
    if single_class:
        Xtr, ytr = I7._single_class_batch(env, seed, I7.N_DRAWS)
    else:
        Xtr, ytr = I7._contrastive_batch(env, seed, I7.N_DRAWS, mispaired=mispaired)
    I7._train_eprop(net, Xtr, ytr, I7.EPOCHS, I7.BATCH, seed)
    return net, fam, float(abs(I7._readout_norm(net) - ro0))


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE CONTINGENCY sim/ HOOK (prototype, DEFAULT-OFF) -- NOT a committed sim/ edit.
#   The byte-identity go/no-go FAILS by default: appending the 80 e-prop neurons perturbs the CONVERSATIONAL
#   neurons' Izhikevich heterogeneity. ROOT CAUSE: sim/bridge.py `_apply_parameter_heterogeneity` re-seeds ONCE
#   then draws each per-neuron parameter as `cp.random.<dist>(size=num_neurons)` SEQUENTIALLY from ONE stream
#   (default order: izh_a, izh_b, izh_d, izh_C). Growing num_neurons by 80 grows EVERY draw by 80, so every
#   parameter AFTER the first has its first-n_pre (conversational) values SHIFTED -> cp_izh_b / cp_izh_C /
#   cp_izh_d_increment differ (a, vt, threshold, vr, vpeak, k, c_reset are invariant: a is drawn first; the rest
#   are drawn outside this method). This is the append-LAST invariance HOLE the seams never hit (the seam sweeps
#   checked only the THRESHOLD hash, never the full decision transcript).
#   THE MINIMAL FIX (validated by this monkeypatch): re-seed per parameter (`cp.random.seed(het_seed + K*i)`) so
#   each param draws from position 0 of its OWN substream -> the first-n_pre values are INVARIANT to appended-LAST
#   slices. In sim/ this belongs behind a default-OFF cfg flag (e.g. `cfg.heterogeneity_append_invariant`): the
#   shipped #7 (flag off) stays byte-identical to its banked runs; the MERGED #7 (flag on) is a new-but-valid
#   substrate re-validated by the smoke/sweep. Prototyped here as a RUNTIME monkeypatch purely so this file can
#   VALIDATE the fix + produce the with-hook smoke numbers -- the coordinator (with the owner) decides whether to
#   land the real sim/ flag. NOT applied unless explicitly requested (`--hook`).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
_HET_HOOK_APPLIED = {"on": False}
_ORIG_HET = _simbridge.SimulationBridge._apply_parameter_heterogeneity


def apply_heterogeneity_append_invariant_hook():
    """Install the PROTOTYPE per-parameter-reseed heterogeneity draw (append-LAST invariant). Idempotent."""
    if _HET_HOOK_APPLIED["on"]:
        return
    cp = get_backend()[0]

    def _patched(self, cfg, n, *, backend_neutral=False):
        if backend_neutral:
            return _ORIG_HET(self, cfg, n, backend_neutral=backend_neutral)   # untouched path
        if not cfg.heterogeneity_distributions:
            cfg.heterogeneity_distributions = self._get_default_heterogeneity_distributions(cfg)
        het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed
        rng_state = cp.random.get_state() if het_seed >= 0 else None
        pm = {"izh_C_val": getattr(self, "cp_izh_C", None), "izh_a_val": getattr(self, "cp_izh_a", None),
              "izh_b_val": getattr(self, "cp_izh_b", None), "izh_d_val": getattr(self, "cp_izh_d_increment", None),
              "hh_C_m": getattr(self, "cp_hh_C_m", None), "hh_g_Na_max": getattr(self, "cp_hh_g_Na_max", None),
              "hh_g_K_max": getattr(self, "cp_hh_g_K_max", None), "hh_g_L": getattr(self, "cp_hh_g_L", None)}
        for i, (pname, spec) in enumerate(cfg.heterogeneity_distributions.items()):
            tgt = pm.get(pname)
            if tgt is None or tgt.size != n:
                continue
            if het_seed >= 0:
                cp.random.seed(int(het_seed) + 1009 * i)      # per-parameter substream -> append-LAST invariant
            dt = spec.get("type")
            if dt == "lognormal":
                s = cp.random.lognormal(mean=spec["mean_log"], sigma=spec["sigma_log"], size=n).astype(cp.float32)
            elif dt == "gaussian":
                s = cp.random.normal(loc=spec["mean"], scale=spec["std"], size=n).astype(cp.float32)
                mv = spec["mean"]
                if mv > 0:
                    s = cp.clip(s, mv * 0.1, mv * 3.0)
                elif mv < 0:
                    s = cp.clip(s, mv * 3.0, mv * 0.1)
            else:
                continue
            if cfg.enable_parameter_heterogeneity or self.cp_heterogeneity_neuron_mask is None:
                tgt[:] = s
            else:
                tgt[:] = cp.where(self.cp_heterogeneity_neuron_mask, s, tgt)
        if rng_state is not None:
            cp.random.set_state(rng_state)

    _simbridge.SimulationBridge._apply_parameter_heterogeneity = _patched
    _HET_HOOK_APPLIED["on"] = True


_IZH_PARAMS = ("cp_neuron_firing_thresholds", "cp_izh_vt", "cp_izh_a", "cp_izh_b", "cp_izh_C",
               "cp_izh_d_increment", "cp_izh_c_reset", "cp_izh_k", "cp_izh_vr", "cp_izh_vpeak")


def _hash_first(arr, n):
    return hashlib.sha1(np.asarray(to_host(arr[:int(n)]), dtype=np.float64).tobytes()).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE GO / NO-GO: conversational byte-identity WITH vs WITHOUT the appended e-prop slices.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def byte_identity_eprop(seed, hook=False, pin_ou_noise=20260810):
    """Build the #7 brain WITHOUT (b0) vs WITH (b1) the appended e-prop slices+FF, then assert the CONVERSATIONAL
    neurons are byte-identical. Hashes every per-neuron Izhikevich parameter over the first n_pre neurons (the
    SUBSTRATE truth) AND compares the DECISION TRANSCRIPT over HUMAN_TURNS. The arbiter WTA read rides UNSEEDED OU
    read-noise (the codebase excludes the raw floats as eval noise, _corpus_facts...:418); its stream position at
    chat-start depends on num_neurons, so we PIN the backend RNG before each chat (`pin_ou_noise`) to ISOLATE the
    substrate identity from that eval-harness artifact. hook=True installs the prototype append-invariant
    heterogeneity draw (the proposed sim/ fix)."""
    if hook:
        apply_heterogeneity_append_invariant_hook()
    xp, _ = get_backend()
    turns = list(TT.HUMAN_TURNS)
    V = I7.V

    # b0: the shipped #7 build (no e-prop slices).
    b0, c0, i0, s0 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True, vocab=V)
    n_pre = int(b0.core_config.num_neurons)
    cc0 = CF._concept_hash(c0)
    izh0 = {p: _hash_first(getattr(b0, p), n_pre) for p in _IZH_PARAMS if getattr(b0, p, None) is not None}
    _v0, f0 = SA._store_facts(c0)
    if pin_ou_noise is not None:
        np.random.seed(int(pin_ou_noise))
    t0 = CF.run_chat(b0, xp, i0, s0, c0, f0, turns)

    # b1: WITH the appended e-prop slices + the two plastic FF pathways.
    b1, c1, i1, s1 = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True, vocab=V,
                                        co_resident_eprop=True, eprop_dims=(I7.N_IN, I7.HIDDEN, I7.K))
    cc1 = CF._concept_hash(c1)
    izh1 = {p: _hash_first(getattr(b1, p), n_pre) for p in _IZH_PARAMS if getattr(b1, p, None) is not None}
    _v1, f1 = SA._store_facts(c1)
    if pin_ou_noise is not None:
        np.random.seed(int(pin_ou_noise))
    t1 = CF.run_chat(b1, xp, i1, s1, c1, f1, turns)

    izh_diffs = sorted(p for p in izh0 if izh0[p] != izh1.get(p))
    dec_ident = bool(json.dumps(CF._decision_view(t0), sort_keys=True, default=str)
                     == json.dumps(CF._decision_view(t1), sort_keys=True, default=str))
    substrate_ident = bool(len(izh_diffs) == 0 and cc0 == cc1)
    return {
        "hook_applied": bool(hook),
        "threshold_hash_identical": bool(izh0.get("cp_neuron_firing_thresholds")
                                         == izh1.get("cp_neuron_firing_thresholds")),
        "izh_params_identical": bool(len(izh_diffs) == 0), "izh_params_that_differ": izh_diffs,
        "concept_codes_identical": bool(cc0 == cc1),
        "substrate_byte_identical": substrate_ident,
        "num_neurons_without_eprop": n_pre, "num_neurons_with_eprop": int(b1.core_config.num_neurons),
        "n_eprop_neurons": int(b1.core_config.num_neurons) - n_pre,
        "decision_transcript_identical": dec_ident,
        "held": bool(substrate_ident and dec_ident),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE 1-SEED MERGED-#7 GO SMOKE + the NEW one-brain teeth. Reuses I7's chat/moat/lesion machinery unchanged; the
# ONLY swap is the co-resident net + the merged bridge. NB the FF weights are on the SHARED bridge, so a NEW net
# construction RE-INITS them -- every read that needs a given net's trained weights runs BEFORE the next net.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def smoke_seed(seed, hook=True):
    if hook:
        apply_heterogeneity_append_invariant_hook()
    xp, _ = get_backend()
    t_start = time.time()
    V = I7.V

    # ---- ONE conversational brain WITH the co-resident e-prop slices (co_resident_eprop=True). ----
    bridge, comp, idx, snap = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                                 vocab=V, co_resident_eprop=True,
                                                 eprop_dims=(I7.N_IN, I7.HIDDEN, I7.K))
    eprop_idx = idx["eprop"]
    kb_before = len(comp.kb)
    _vc, curated_facts = SA._store_facts(comp)
    kb_after_store = len(comp.kb)
    env = I7._make_env(int(seed))
    facts_all = list(curated_facts) + list(I7.TAUGHT_FACTS)
    turns = I7._turns()

    # ---- TREATMENT: the co-resident e-prop net trains on the MERGED bridge; every treatment-weight read follows. ----
    net_t, fam_t, readout_moved = _teach_merged(int(seed), env, bridge, eprop_idx, snap)
    kb_after_teach = len(comp.kb)

    # one-brain teeth (structural; independent of learning): epropnet.br IS comp._merged IS bridge (ONE object).
    single_bridge = bool(net_t.br is bridge and comp._merged is bridge)
    n_ff = int(sum(int(np.asarray(to_host(a)).shape[0]) for a in net_t._data_idx_flat))
    n_syn_total = int(bridge.cp_connections.nnz)
    ff_slots = set(int(x) for a in net_t._data_idx_flat for x in np.asarray(to_host(a)).tolist())
    ff_in_shared = bool(n_ff == 1600 and max(ff_slots) < n_syn_total and (n_syn_total - n_ff) > 0
                        and net_t.br.cp_connections is bridge.cp_connections)

    shim_treat = I7.ChatShim(comp, env, net=net_t, fam=fam_t, enabled=True, use_gate=True)
    tr_treat = CF.run_chat(bridge, xp, idx, snap, shim_treat, facts_all, turns)
    sum_treat = CF._chat_summary(tr_treat)
    recall_treat, recalled = I7._taught_recall(tr_treat)

    moat_fa, moat_ex = I7._moat_battery(shim_treat)
    teeth = CF.posthoc_teeth(shim_treat, facts_all, seed=int(seed))

    off_shim = I7.ChatShim(comp, env, net=net_t, fam=fam_t, enabled=True, use_gate=False, use_conf=True)
    gate_off_fa, _ = I7._moat_battery(off_shim)
    intact_margin, lesion_margin = I7._lesion_margin(fam_t, env)

    heldout_head = I7._heldout_acc(net_t, env, I7.HEADLINE_REFERENT,
                                   I7.PATIENT_WORDS.index(I7.TAUGHT[I7.HEADLINE_REFERENT]))

    # one-brain tooth: an e-prop teaching pass moves ONLY the eprop FF slots (conversational synapses byte-unchanged).
    net_probe = _mk_merged_net(bridge, snap, eprop_idx, int(seed))          # re-inits the FF (net_t weights now stale)
    data_pre = np.asarray(to_host(bridge.cp_connections.data)).copy()
    Xb, yb = I7._contrastive_batch(env, int(seed), I7.N_DRAWS)
    net_probe.fit_readout_norm(Xb)
    net_probe.train_batch(Xb[:I7.BATCH], yb[:I7.BATCH])
    data_post = np.asarray(to_host(bridge.cp_connections.data))
    changed = set(int(x) for x in np.where(data_pre != data_post)[0].tolist())
    probe_ff = set(int(x) for a in net_probe._data_idx_flat for x in np.asarray(to_host(a)).tolist())
    moves_confined = bool(changed.issubset(probe_ff) and len(changed) > 0)
    conv_weights_unchanged = moves_confined

    # ---- FROZEN-READOUT control (identical teaching, eprop_lr=0 -> zero readout -> taught patient not recalled). ----
    net_fz, fam_fz, readout_moved_frozen = _teach_merged(int(seed), I7._make_env(int(seed)), bridge, eprop_idx, snap,
                                                         freeze=True)
    shim_frozen = I7.ChatShim(comp, env, net=net_fz, fam=fam_fz, enabled=True, use_gate=True)
    tr_frozen = CF.run_chat(bridge, xp, idx, snap, shim_frozen, facts_all, turns)
    recall_frozen, _ = I7._taught_recall(tr_frozen)

    # ---- attribution: the taught-recall RISE is from the WEIGHT CHANGE (trained vs frozen-readout, identical teach). ----
    recall_attrib = attributable_to(
        "taught-fact chat recall from the co-resident e-prop weight change (trained vs frozen-readout)",
        float(recall_treat), float(recall_frozen))

    # ---- GO flags ----
    recall_ok = bool(recall_treat == len(I7.TAUGHT))
    moat_ok = bool(moat_fa == 0)
    frozen_ok = bool(readout_moved_frozen <= 1e-3 and recall_frozen == 0)
    lesion_ok = bool(gate_off_fa > 0 and lesion_margin < intact_margin - 0.30)
    ood_ok = bool(sum_treat["ood_abstained"] == sum_treat["ood_turns"]
                  and sum_treat["ungrounded_word_total"] == 0 and sum_treat["confabulated"] == 0)
    posthoc_ok = bool(abs(teeth["unsupported_drop_rate"] - 1.0) < 1e-9 and teeth["unsupported_props"] > 0
                      and abs(teeth["supported_keep_rate"] - 1.0) < 1e-9)
    ct1_ok = bool(heldout_head > 0.6)
    kb_unchanged = bool(kb_after_teach == kb_after_store)

    one_brain_teeth = {
        "single_SimulationBridge (net.br IS comp._merged IS bridge)": single_bridge,
        "eprop_FF_edges_in_SAME_cp_connections_as_conversational": ff_in_shared,
        "n_eprop_ff_synapses": n_ff, "n_total_synapses": n_syn_total,
        "n_conversational_synapses": n_syn_total - n_ff,
        "e-prop_teach_moves_ONLY_eprop_ff (conv weights unchanged)": conv_weights_unchanged,
        "n_data_slots_changed_by_a_teach_pass": len(changed),
    }
    teeth_ok = bool(single_bridge and ff_in_shared and conv_weights_unchanged)

    smoke_go = bool(recall_ok and moat_ok and frozen_ok and lesion_ok and ood_ok and posthoc_ok
                    and ct1_ok and kb_unchanged and teeth_ok)

    return {
        "seed": int(seed), "hook_applied": bool(hook), "elapsed_s": round(time.time() - t_start, 1),
        "num_neurons": int(bridge.core_config.num_neurons),
        "taught_recall": recall_treat, "recalled": recalled, "recall_ok_3of3": recall_ok,
        "moat_false_accepts": moat_fa, "moat_ok": moat_ok, "moat_examples": moat_ex[:3],
        "frozen_recall": recall_frozen, "frozen_readout_moved": readout_moved_frozen, "frozen_ok": frozen_ok,
        "gate_off_false_accepts": gate_off_fa, "intact_margin": intact_margin, "lesion_margin": lesion_margin,
        "lesion_gate_load_bearing": lesion_ok,
        "ood_abstained": sum_treat["ood_abstained"], "ood_turns": sum_treat["ood_turns"],
        "confabulated": sum_treat["confabulated"], "ungrounded_word_total": sum_treat["ungrounded_word_total"],
        "ood_ok": ood_ok,
        "posthoc_unsupported_drop_rate": teeth["unsupported_drop_rate"],
        "posthoc_supported_keep_rate": teeth["supported_keep_rate"],
        "posthoc_unsupported_props": teeth["unsupported_props"], "posthoc_ok": posthoc_ok,
        "heldout_headline_acc": heldout_head, "ct1_ok": ct1_ok,
        "readout_moved_treatment": readout_moved, "kb_unchanged": kb_unchanged,
        "recall_attributable_to_weight_change": recall_attrib,
        "ONE_BRAIN_TEETH": one_brain_teeth, "teeth_ok": teeth_ok,
        "SMOKE_GO": smoke_go,
    }


def sweep(seeds, hook=True):
    """SELF-SWEEP (the coordinator's ONE command): per seed run the byte-identity go/no-go + the GO smoke, then
    aggregate. WITH the proposed heterogeneity hook (hook=True) byte-identity should HOLD on every seed and the
    merged-#7 chat should GO on every seed. Writes per-seed + aggregate (no per-seed orchestration by Claude)."""
    per = []
    for sd in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            bi = byte_identity_eprop(int(sd), hook=hook)
            sm = smoke_seed(int(sd), hook=hook)
        per.append({"seed": int(sd), "byte_identity_held": bool(bi["held"]),
                    "izh_params_that_differ": bi["izh_params_that_differ"], "SMOKE_GO": bool(sm["SMOKE_GO"]),
                    "taught_recall": sm["taught_recall"], "moat_false_accepts": sm["moat_false_accepts"],
                    "frozen_recall": sm["frozen_recall"], "lesion_gate_load_bearing": sm["lesion_gate_load_bearing"],
                    "teeth_ok": sm["teeth_ok"], "one_brain_teeth": sm["ONE_BRAIN_TEETH"]})
        print(f"seed {sd}: byte_identity_held={per[-1]['byte_identity_held']} SMOKE_GO={per[-1]['SMOKE_GO']} "
              f"recall={per[-1]['taught_recall']} moat_fa={per[-1]['moat_false_accepts']} teeth_ok={per[-1]['teeth_ok']}")
    n = len(per)
    agg = {"hook_applied": bool(hook), "n_seeds": n,
           "n_byte_identity_held": sum(1 for r in per if r["byte_identity_held"]),
           "n_smoke_go": sum(1 for r in per if r["SMOKE_GO"]),
           "GO_6of6": bool(n == len(seeds) and all(r["byte_identity_held"] and r["SMOKE_GO"] for r in per)),
           "per_seed": per}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds -> SELF-SWEEP (byte-identity + smoke per seed, aggregated)")
    ap.add_argument("--byte-identity", action="store_true", help="run ONLY the conversational byte-identity go/no-go")
    ap.add_argument("--smoke", action="store_true", help="run the 1-seed merged-#7 GO smoke + one-brain teeth")
    ap.add_argument("--hook", action="store_true",
                    help="install the PROTOTYPE append-invariant heterogeneity draw (the proposed sim/ fix)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        agg = sweep(seeds, hook=bool(args.hook))
        print("=== 6-SEED SELF-SWEEP AGGREGATE ===")
        print(json.dumps(agg, indent=2, default=str))
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(agg, fh, indent=2, default=str)
        return

    result = {}
    if args.byte_identity or not args.smoke:
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            bi = byte_identity_eprop(int(args.seed), hook=bool(args.hook))
        bi["elapsed_s"] = round(time.time() - t0, 1)
        result["byte_identity"] = bi
        print("=== BYTE-IDENTITY (conversational, WITH vs WITHOUT appended e-prop slices) ===")
        print(json.dumps(bi, indent=2, default=str))
        print("VERDICT:", "HELD" if bi["held"] else "FAILED -> STOP, report the perturbation (see izh_params_that_differ)")

    if args.smoke:
        sm = smoke_seed(int(args.seed), hook=bool(args.hook))
        result["smoke"] = sm
        print("=== 1-SEED MERGED-#7 GO SMOKE + ONE-BRAIN TEETH ===")
        print(json.dumps(sm, indent=2, default=str))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
