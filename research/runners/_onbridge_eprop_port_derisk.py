"""PORT the VALIDATED e-prop rule onto the PRODUCTION Izhikevich spiking substrate (`OnBridgeBDSPNet`).

WHY (2026-07-14). The isolation run `_snn_bptt_forward_vs_learning_isolation_derisk.py` proved that a transport-free
biological LOCAL rule -- e-prop (Bellec 2020) with DIRECT feedback alignment (Nokland) -- TRAINS the depth-2
compositional-inheritance task on a cheap LIF net (6-seed GO: train 1.0, inherit 0.895, NO weight transport). But the
committed one-step BDSP local rule on the REAL production `OnBridgeBDSPNet` (Izhikevich two-compartment bridge) FAILS the
same task (0/6). This runner ports the SAME e-prop credit rule onto the SAME production Izhikevich substrate to ask the
decisive question: does the transport-free biological LOCAL rule train the compositional task ON THE REAL BRIDGE where the
committed BDSP rule does not? That is the emergence engine's core learning mechanism on the production substrate.

THE PORT (runner-side ONLY; NO `sim/` edit):
  * Reuse `OnBridgeBDSPNet` VERBATIM by subclass (same Izhikevich bridge, same input->H1->H2->out neuron-index slices,
    same plastic feedforward `cp_connections`, same tonic-drive/forward machinery). We pass the parent `lr=0.0` so the
    committed BDSP kernel is byte-INERT (`w_new = w + 0*...`); e-prop is the SOLE weight-mover. `enable_bdsp=True` stays
    on only so `cp_bdsp_E` remains available as the substrate's event-rate readout (unused for learning).
  * FORWARD IS SPIKING on the real bridge: features -> graded input current on the input slice; the slices SPIKE; we
    RECORD, per settle step, every neuron's SPIKE (`cp_firing_states`) and MEMBRANE (`cp_membrane_potential_v`) via
    `sim.backend.to_host`. The output slice's event rate (or summed spikes) = the class logits.
  * E-PROP GRAD (exactly the validated rule), per FF weight w_ji (pre i -> post j):
        dw_ji = sum_t L_j(t) * psi_j(t) * eps_i(t)
      - eps_i(t) = alpha * eps_i(t-1) + z_pre_i(t)   [forward eligibility; z_pre = the pre-neuron's RECORDED SPIKE at
        step t; alpha = a leak in (0,1), the Izhikevich analogue of the LIF membrane leak / the BDSP event-rate tau].
      - psi_j(t) = the surrogate "closeness to firing" of HIDDEN post neuron j's MEMBRANE at step t. Because the
        Izhikevich membrane is recorded POST-reset (a fired neuron shows c_reset ~ -53mV, BELOW vt ~ -42mV), a raw
        atan(v-vt) would read LOW exactly when the neuron fired. So the surrogate is SPIKE-AUGMENTED: psi = atan-surrogate
        of the per-neuron (v - cp_izh_vt), OVERRIDDEN to the surrogate PEAK on steps where the neuron actually spiked (it
        DID cross threshold). The OUTPUT is a leaky readout (psi=1; see below).
      - L_j(t) = the learning signal: OUTPUT layer = the class error softmax(logits)-onehot (distributed /T); HIDDEN
        layer = DFA = (output error) @ B_direct[layer], B_direct fixed-random (k, n_post) from a SEPARATE seed stream
        => NO weight transport.
  * OUTPUT = a BELLEC-2020 LEAKY READOUT (logit_source="leaky_readout", the default): the output cell is a non-spiking
    leaky integrator, logit_k = sum_j W_{jk} * r_j, r_j = the last-hidden low-pass spike eligibility (per-neuron
    STANDARDIZED for conditioning), W_{jk} the H_last->out weights in cp_connections. Its membrane is EXACTLY linear in
    W so the eligibility grad (r_j*delta_k) is the EXACT readout gradient (convex) -- the Izhikevich output neuron's
    spiking membrane is nonlinear in W AND fires too sparsely (0-2 spikes) to be a usable readout, and training it
    ANTI-learns (a documented dead end). Only the readout's linear integration is host-side; the readout WEIGHTS live in
    the substrate cp_connections and are moved by e-prop; all hidden layers are fully spiking on the bridge.
  * APPLY dw to the FF `cp_connections.data` directly: a position map (built once from the cached COO, whose row/col
    align with `.data`) sends each per-(pre,post) grad to its data entry; respects the per-synapse plastic mask +
    plasticity-rate gain if present. Signed-clamped to [-w_clip, w_clip]. hidden_lr_scale sets the hidden-layer lr.

SUBSTRATE-FORWARD FINDINGS (what a DISCRIMINATIVE spiking forward on the production Izhikevich bridge REQUIRED -- the
committed BDSP rule fails 0/6 for the SAME forward reasons, not a credit reason). All runner-side cfg flips, NO sim/ edit:
  (1) STP OFF -- short-term depression (stp_tau_d=200ms) throttles the FF synaptic drive under repeated firing, so learned
      FF weights cannot propagate a class-discriminative signal (the parent sidesteps this with STP-immune external tonic
      current, which then swamps the input -> unlearnable-from). (2) PER-EXAMPLE WASH-OUT -- restore the post-init state
      before each forward so the trace is a fresh function of THIS input (no cross-example state carryover masquerading as
      input-dependence; EMERGE-61 precedent). (3) STRONG FF init (~2000) + high w_clip -- the substrate's synaptic gain is
      very low (a weight ~1000 is needed for any postsynaptic firing), so the learnable FF weights must be O(1000s).
      (4) structural plasticity / homeostasis / conductance+OU noise OFF -- so nnz is static (the position map holds) and
      the forward is stationary. A clean linear readout on the resulting H2 is ~0.7-0.9 separable (the forward IS
      discriminative); the leaky readout above is what lets e-prop read it out.

MANDATORY VALIDITY GATES (the build is worthless without these; all REPORTED):
  * POSITIVE CONTROL (the hard gate): the ported e-prop MUST fit a small train set (~40 examples, many epochs) to HIGH
    train accuracy (>> chance). If it can't memorize a small set, the port is BROKEN (bad surrogate / grad / weight-write)
    -- fix it before ANY "fails to train" conclusion (that would be a bug, not science).
  * permuted-label control -> ~chance (no leakage).
  * shuffle-DFA control (scramble the per-example learning signal across a batch: eligibility intact, credit mismatched
    to the example) -> ~chance (the DFA credit is load-bearing).
  * 3 seeds (42/43/44): train acc, held-out inherit acc, chance, oracle, permuted, shuffle-DFA.

VERDICT: whether the ported e-prop TRAINS the compositional task on the Izhikevich bridge (train high + inherit >> chance,
controls clean). YES => the emergence engine's core learning mechanism works on the production substrate. If the positive
control passes but the full task does not train => an honest, VALID negative (the exact residual reported). Reuse-by-import;
NO `sim/` edit. CPU: SIM_BACKEND=numpy + OPENBLAS_NUM_THREADS=1.

Run (positive control first, then the full 3-seed):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._onbridge_eprop_port_derisk --poscontrol-only --seeds 42
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._onbridge_eprop_port_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# reuse-by-import: the production Izhikevich substrate + the task/controls + the rate oracle (all VERBATIM, NO sim/ edit).
from research.runners._semantic_inheritance_onbridge_spiking_derisk import OnBridgeBDSPNet  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
# --task-xor (ADDITIVE, default OFF): the NON-reservoir-decodable depth-2 XOR->threshold task, reused-by-import as the
# EXACT wrapper the rate/BPTT crux used (`make_task_xor` wraps emerge1's `make_task`: pair-XOR level-1 latents ->
# threshold-over-XORs level-2). XOR is provably NOT linearly decodable from a fixed random projection, so a frozen-hidden
# reservoir MUST underperform trained hidden layers IF the credit is real -- the task on which `deep_credit_share` can
# rise off the ~0 it reads on the linearly-reservoir-decodable inheritance task. Same 4-tuple interface as
# make_task_semantic_inheritance (Xtr,ytr,Ltr),(Xte,yte,Lte),meta,idx; idx["inh_idx"] = the whole held-out set.
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import make_task_xor  # noqa: E402
from sim.dendritic_mlp import DendriticMLP  # noqa: E402
from sim.bptt_snn_gpu import atan_surrogate  # noqa: E402 -- the SAME surrogate the validated e-prop uses

OUT = _REPO / "research" / "findings" / "raw" / "_onbridge_eprop_port.json"


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================================================================
# The ported e-prop net: OnBridgeBDSPNet forward (spiking Izhikevich bridge, BDSP-INERT) + the validated e-prop
# transport-free DFA weight update writing directly into cp_connections.data.
# ============================================================================================================
class OnBridgeEpropNet(OnBridgeBDSPNet):
    STATE_ARRS = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_conductance_g_e", "cp_conductance_g_i",
                  "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise", "cp_refractory_timers", "cp_firing_states",
                  "cp_prev_firing_states", "cp_neuron_firing_thresholds", "cp_ou_current")

    def __init__(self, n_in, hidden, k, seed=0, n_hidden_layers=2, settle_steps=30, pool_k=1,
                 eprop_lr=0.02, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15, beta_surr=1.0,
                 logit_source="leaky_readout", w_clip=200.0, reset_state=True, ou_noise=False, cond_noise=False,
                 stp=False, hp=None):
        hp = hp or {}
        # lr=0.0 -> the committed BDSP kernel is byte-INERT; e-prop is the SOLE weight-mover. enable_bdsp stays True
        # (parent) only so cp_bdsp_E remains available as the substrate's event-rate readout. rule irrelevant (we never
        # inject apical). credit_steps unused (e-prop reads the FORWARD trace; no separate credit pass).
        super().__init__(n_in, hidden, k, seed=seed, rule="plain_fa", n_hidden_layers=n_hidden_layers,
                         settle_steps=settle_steps, credit_steps=0, lr=0.0, pool_k=pool_k,
                         in_current_pA=hp.get("in_current_pA", 700.0), in_bias_pA=hp.get("in_bias_pA", 300.0),
                         tonic_h_pA=hp.get("tonic_h_pA", 100.0), tonic_o_pA=hp.get("tonic_o_pA", 150.0),
                         apical_gain_pA=hp.get("apical_gain_pA", 2000.0), ff_w_init=hp.get("ff_w_init", 2000.0),
                         pbar_alpha=hp.get("pbar_alpha", 0.05))
        # --no-bdsp: the kernel's OWN documented inertness lever. The comment above ("lr=0.0 -> the committed BDSP
        # kernel is byte-INERT") is FALSE: fused_bdsp_update ENDS in `return cp.clip(w_new, w_min, w_max)`
        # (kernels.py:485) -- at eta=0 the ADD term vanishes but the CLIP does not, and the parent sets
        # bdsp_w_min/max = -6/+6 (:183) while ff_w_init=2000 and --w-clip=4000. So every forward silently crushes
        # the FF synapses whose PREsyn fired (call site restricts to active_bd, bridge.py:7280) from ~2000-scale to
        # |w|<=6. The step loop reads cfg.enable_bdsp per-step, so flipping it post-init makes the block UNREACHED
        # = the kernel's documented byte-inert condition. Default False => byte-identical to the banked runs.
        if bool(hp.get("no_bdsp", False)):
            # NOTE: too blunt for the clamp A/B -- it also nulls cp_bdsp_E, which fit_readout_norm depends on
            # (IndexError: 0-dimensional). Kept only as the hard "kernel unreached" lever. Prefer --bdsp-wmax.
            self.br.core_config.enable_bdsp = False
        _bw = hp.get("bdsp_wmax", None)
        if _bw is not None:
            # THE SINGLE-VARIABLE CLAMP LEVER. The parent sets bdsp_w_min/max = -6/+6 (:183) while ff_w_init=2000
            # and --w-clip=4000, so fused_bdsp_update's UNCONDITIONAL `return cp.clip(w_new, w_min, w_max)`
            # (kernels.py:485) crushes every FF synapse whose presyn fired -- despite the docstring's claim that
            # lr=0 makes the kernel "byte-INERT" (it makes only the ADD term vanish). Widening the clip past the
            # weight scale makes the CLIP a no-op while the block still RUNS, so cp_bdsp_E stays available.
            # Isolates the clamp alone. Default None => byte-identical to the banked runs.
            self.br.core_config.bdsp_w_max = float(_bw)
            self.br.core_config.bdsp_w_min = -float(_bw)
        self.eprop_lr = float(eprop_lr); self.eps_leak = float(eps_leak)
        self.surrogate = str(surrogate); self.alpha_surr = float(alpha_surr); self.beta_surr = float(beta_surr)
        self.logit_source = str(logit_source); self.w_clip = float(w_clip); self.reset_state = bool(reset_state)
        self.train_layers = None
        if bool(hp.get('freeze_hidden', False)):
            # THE RESERVOIR CONTROL the GO metric never had (2026-07-16): train ONLY the last FF pathway and
            # freeze the hidden ones at init. If this MATCHES the full arm, the 'deep credit' does nothing and
            # the result is a linear readout on a fixed random spiking reservoir.
            self.train_layers = {self.n_hidden_layers}            # None => update all FF pathways; a set => update only those (isolation)
        self.output_psi_one = True          # OUTPUT = leaky readout (psi=1); hidden spiking units keep the surrogate
        self.logit_temp = 15.0 if logit_source == "membrane" else 1.0   # softmax temperature (membrane spans ~tens of mV)
        self._r_mu = None; self._r_sigma = None   # per-neuron readout-feature normalization (fit_readout_norm)
        self.hidden_lr_scale = float(hp.get("hidden_lr_scale", 1.0))   # hidden-layer lr multiplier (readout uses eprop_lr)
        # ---- freeze every NON-e-prop weight/structure/threshold mechanism so the FF `cp_connections` and its nnz are
        #      STATIC (e-prop is the sole learner) and the forward is a clean, stationary function of the input:
        #      structural plasticity mutates nnz (would break the cached position map); homeostasis drifts the firing
        #      thresholds; conductance/OU noise make the forward stochastic. All runner-side cfg flips (NO sim/ edit).
        self.cfg.enable_structural_plasticity = False
        self.cfg.enable_structural_pruning = False
        self.cfg.enable_synaptic_scaling = False
        self.cfg.enable_homeostasis = False
        self.cfg.enable_ou_process = bool(ou_noise)
        self.cfg.enable_conductance_noise = bool(cond_noise)
        # SHORT-TERM PLASTICITY OFF (stp=False default): STP depression (stp_tau_d=200ms) throttles the FEEDFORWARD
        # synaptic drive under repeated presynaptic firing, so the FF weights e-prop learns cannot propagate a
        # discriminative signal through the layers (the parent OnBridgeBDSPNet sidesteps this by driving the layers with
        # external TONIC current, immune to STP, and letting the FF weights only weakly modulate -- unlearnable-from). With
        # STP off the FF synaptic path propagates monotonically with weight, so e-prop can shape it. Runner-side cfg flip.
        self.cfg.enable_short_term_plasticity = bool(stp)
        self.cfg.enable_per_type_stp = bool(stp)
        from sim.backend import to_host
        xp = self._xp
        # snapshot the clean post-init baseline state for the per-example WASH-OUT (EMERGE-61 precedent: restore the
        # substrate's post-init state before each production so each forward is a fresh function of THIS input, with no
        # cross-example residual contaminating the per-example eligibility/surrogate credit). Public-attribute writes.
        self._state0 = {name: getattr(self.br, name).copy() for name in self.STATE_ARRS
                        if getattr(self.br, name, None) is not None}
        # per-neuron spike-initiation threshold vt (heterogeneous) -- the surrogate centre for "closeness to firing".
        self.vt = np.asarray(to_host(self.br.cp_izh_vt), dtype=np.float64)
        self._psi_peak = float(atan_surrogate(np.array([0.0]), alpha=self.alpha_surr, xp=np)[0])
        # ---- DFA fixed-random feedback B_direct: (k, n_post_phys) per HIDDEN layer, SEPARATE seed stream => NO weight
        #      transport (identical construction to the validated LIF e-prop). Output layer uses the error directly. ----
        frng = np.random.default_rng(seed + 8888)
        self.B_direct = [frng.normal(0.0, 1.0 / np.sqrt(k), (k, self.sizes_phys[li + 1])).astype(np.float64)
                         for li in range(len(self.sizes) - 2)]
        # ---- position map: for each FF pathway, the index into cp_connections.data of every (pre,post) edge. The cached
        #      COO row/col ALIGN with cp_connections.data (tocoo(copy=False) preserves CSR row-major order), so
        #      posmat[row,col] = arange(nnz) sends (pre,post) -> its .data slot. Built once (structure never changes). ----
        coo = self.br._get_cached_coo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        posmat = -np.ones((self.n_total, self.n_total), dtype=np.int64)
        posmat[row, col] = np.arange(len(row))
        self._data_idx_flat = []
        for li in range(len(self.sizes) - 1):
            pre, post = self._ff_edges[li]
            idx2d = posmat[np.ix_(np.asarray(pre), np.asarray(post))]     # (n_pre_phys, n_post_phys), row(pre)-major
            if (idx2d < 0).any():
                raise RuntimeError(f"FF pathway {li} has an edge missing from cp_connections (position map failed).")
            self._data_idx_flat.append(xp.asarray(idx2d.ravel()))
        # LEAKY READOUT: the last FF pathway is a linear readout (host-integrated; does NOT drive substrate spikes), so
        # ZERO-INIT it -- a clean convex softmax-regression start (logit=0 -> uniform p -> graded delta), exactly the
        # offline readout that converged. e-prop grows it. (Other pathways keep the strong FF init that DOES drive spikes.)
        if self.logit_source == "leaky_readout":
            self.br.cp_connections.data[self._data_idx_flat[-1]] = xp.asarray(
                np.zeros(int(self._data_idx_flat[-1].shape[0]), dtype=np.float32))

    # ---------- spiking forward WITH per-step recording of spikes + membrane over the settle window ----------
    def _forward_record(self, feat_row, reset_rates=True):
        from sim.backend import to_host
        xp = self._xp; n = self.n_total
        # per-example WASH-OUT: restore the clean post-init substrate state so this forward is a fresh function of THIS
        # input (no cross-example residual v/u/conductance/refractory contaminating the per-example e-prop trace).
        if self.reset_state:
            for name, arr0 in self._state0.items():
                getattr(self.br, name)[...] = arr0
        if reset_rates and self.br.cp_bdsp_E is not None:
            self.br.cp_bdsp_E[...] = 0.0
            self.br.cp_bdsp_B[...] = 0.0
            self.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
        drive = self._base_drive()
        f = np.asarray(feat_row, dtype=np.float32)
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * f, 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        drive_xp = xp.asarray(drive)
        if self.br.cp_bdsp_apical_drive is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        T = self.settle_steps
        sp = np.zeros((T, n), dtype=np.float32)
        vv = np.zeros((T, n), dtype=np.float32)
        for t in range(T):
            self.br.cp_external_input_current = drive_xp     # re-assert the constant input current each step
            self.br._run_one_simulation_step()
            sp[t] = np.asarray(to_host(self.br.cp_firing_states), dtype=np.float32)
            vv[t] = np.asarray(to_host(self.br.cp_membrane_potential_v), dtype=np.float32)
        E = np.asarray(to_host(self.br.cp_bdsp_E)).copy()
        acts = [self._pool(E[self.slices[li]], li) for li in range(len(self.sizes))]
        return sp, vv, acts

    def _readout_elig(self, sp):
        """The summed low-pass ELIGIBILITY of the LAST HIDDEN layer's spikes, r_j = sum_t eps_j(t) with
        eps_j(t)=leak*eps_j(t-1)+z_j(t) -- the exact quantity the OUTPUT-layer e-prop grad integrates. PHYSICAL space."""
        hl = self.slices[-2]
        z = sp[:, hl].astype(np.float64)                 # (T, n_Hlast_phys)
        eps = np.zeros(z.shape[1], dtype=np.float64); r = np.zeros_like(eps)
        for t in range(z.shape[0]):
            eps = self.eps_leak * eps + z[t]; r += eps
        return r                                         # (n_Hlast_phys,)

    def _readout_feature(self, sp):
        """The readout feature = the last-hidden eligibility r, per-neuron STANDARDIZED (a synaptic-scaling / feature
        normalization, once-fitted) so the leaky-readout softmax regression is well-conditioned. == raw r if unfitted."""
        r = self._readout_elig(sp)
        if self._r_mu is not None:
            return (r - self._r_mu) / self._r_sigma
        return r

    def fit_readout_norm(self, X, max_examples=200):
        """Fit the per-neuron mean/std of the last-hidden eligibility r over the train set (a fixed feature
        normalization for the leaky readout -- like a leaky-integrator cell's homeostatic input scaling)."""
        m = min(int(max_examples), len(X))
        R = np.array([self._readout_elig(self._forward_record(X[i])[0]) for i in range(m)])
        self._r_mu = R.mean(axis=0)
        self._r_sigma = R.std(axis=0) + 1e-6

    def _logits_from(self, sp, vv, acts):
        out_sl = self.slices[-1]
        if self.logit_source == "leaky_readout":
            # BELLEC-2020 LEAKY READOUT (the standard e-prop output): logit_k = sum_j W_{jk} * r_j, where r_j is the
            # last-hidden low-pass spike eligibility and W is the H_last->out weights read from cp_connections. This is a
            # non-spiking leaky-integrator readout NEURON whose membrane is EXACTLY linear in its readout weights -> the
            # eligibility-based output grad (r_j*delta_k) is the EXACT gradient of this readout (convex) -- unlike the
            # substrate output neuron's Izhikevich membrane, which is nonlinear in the weights (the anti-learning we saw).
            # The readout WEIGHTS still live in the substrate's cp_connections (moved by e-prop); only the readout's
            # linear integration is host-side, as befits a leaky-integrator output cell. PHYSICAL -> pooled to k.
            from sim.backend import to_host
            r = self._readout_feature(sp)                          # (n_Hlast_phys,) standardized eligibility
            W = np.asarray(to_host(self.br.cp_connections.data[self._data_idx_flat[-1]]), dtype=np.float64).reshape(
                self.sizes_phys[-2], self.sizes_phys[-1])           # (n_Hlast_phys, k*K)
            logit_phys = r @ W                                      # (k*K,)
            return self._pool(logit_phys, len(self.sizes) - 1)
        if self.logit_source == "event_rate":
            return np.asarray(acts[-1], dtype=np.float64)
        if self.logit_source == "membrane":
            # GRADED LEAKY READOUT (Bellec 2020): the output-slice MEAN MEMBRANE POTENTIAL over the settle window. A
            # high-resolution, always-nonzero, input-graded readout -- the spiking output layer fires too sparsely
            # (0-2 spikes) on this substrate to be a usable spike-count readout (see the finding), so the readout is
            # the leaky output membrane (the e-prop leaky-readout neuron), not a spike count.
            m = vv[:, out_sl].mean(axis=0)
            return self._pool(m, len(self.sizes) - 1)
        # spike_sum: summed output-slice spikes over the settle window, pooled to k (the LIF-reference readout).
        s = sp[:, out_sl].sum(axis=0)
        return self._pool(s, len(self.sizes) - 1)

    def _surrogate(self, v_post, sp_post, post_sl):
        """psi_j(t): the atan surrogate 'closeness to firing' of post neuron j, SPIKE-AUGMENTED to handle the post-reset
        Izhikevich membrane (a fired neuron shows c_reset < vt, so the raw atan reads low exactly when it fired). The
        'std' standardized-membrane surrogate is computed inline in `_accum_grad` (it needs the per-example trace)."""
        vt = self.vt[post_sl]
        sub = atan_surrogate(v_post - vt, alpha=self.alpha_surr, xp=np)
        return np.where(sp_post > 0.5, self._psi_peak, sub)

    def _accum_grad(self, grads, sp, vv, delta_k, skip_output=False):
        """Accumulate the e-prop grad for ONE example into `grads` (per FF pathway, physical (n_pre, n_post)).
        `skip_output` leaves the last (readout) pathway to the clean standardized delta rule in train_batch."""
        L = len(self.sizes) - 1
        T = sp.shape[0]
        delta_out_phys = self._broadcast(np.asarray(delta_k, dtype=np.float64), L) / self.pool_k   # (n_out_phys,)
        # precompute the standardized-surrogate per-layer stats if in 'std' mode (self-contained, per example)
        std_stats = None
        if self.surrogate == "std":
            std_stats = {}
            for li in range(L):
                post_sl = self.slices[li + 1]
                vp = vv[:, post_sl]
                std_stats[li] = (vp.mean(), vp.std() + 1e-6)
        eps = [np.zeros(self.sizes_phys[li], dtype=np.float64) for li in range(L)]
        last = L - 1
        for t in range(T):
            for li in range(L):
                if skip_output and li == last:
                    # still advance the readout-layer eligibility? no -- the output grad is done in train_batch. skip.
                    continue
                z_pre = sp[t, self.slices[li]].astype(np.float64)             # RECORDED pre-neuron spikes (z_pre)
                eps[li] = self.eps_leak * eps[li] + z_pre                      # forward eligibility trace
                post_sl = self.slices[li + 1]
                v_post = vv[t, post_sl].astype(np.float64)
                sp_post = sp[t, post_sl]
                if self.surrogate == "std":
                    m, s = std_stats[li]
                    z = (v_post - m) / s
                    sub = 1.0 / (1.0 + (self.beta_surr * z) ** 2)
                    psi = np.where(sp_post > 0.5, 1.0, sub)
                else:
                    psi = self._surrogate(v_post, sp_post, post_sl)
                if li == L - 1:
                    Lsig = delta_out_phys / T                                  # output error directly (distributed /T)
                    if self.output_psi_one:
                        psi = np.ones_like(psi)      # OUTPUT is a LEAKY READOUT (Bellec 2020): psi=1, no surrogate gate
                        # -> avoids the DEAD-UNIT collapse (a silent spiking output unit has surrogate psi~0 -> no
                        # gradient -> its weights can never grow to make it fire again). Hidden layers keep the surrogate.
                else:
                    Lsig = (np.asarray(delta_k, dtype=np.float64) @ self.B_direct[li]) / T   # DFA (transport-free)
                g = Lsig * psi                                                 # (n_post_phys,)
                grads[li] += np.outer(eps[li], g)                             # (n_pre_phys, n_post_phys)

    def _apply_grads(self, grads, bsz):
        from sim.backend import to_host  # noqa: F401
        xp = self._xp
        data = self.br.cp_connections.data
        L = len(grads)
        for li in range(L):
            if self.train_layers is not None and li not in self.train_layers:
                continue                                    # freeze this FF pathway (readout-only isolation)
            idx = self._data_idx_flat[li]
            # the HIDDEN e-prop grads (eligibility x surrogate x DFA-credit) and the readout delta grad differ in
            # scale; hidden_lr_scale lets the hidden layers use a different effective lr (readout = eprop_lr).
            lr_li = self.eprop_lr * (1.0 if li == L - 1 else self.hidden_lr_scale)
            dw = (lr_li * (grads[li] / max(1, bsz))).astype(np.float32).ravel()
            cur = data[idx]
            new = xp.clip(cur - xp.asarray(dw), -self.w_clip, self.w_clip)     # GD: w -= lr * grad
            if self.br.cp_synapse_plastic_mask is not None:
                pm = self.br.cp_synapse_plastic_mask[idx]
                new = xp.where(pm, new, cur)
            if self.br.cp_plasticity_rate_gain is not None:
                gain = self.br.cp_plasticity_rate_gain[idx]
                new = cur + (new - cur) * gain
            data[idx] = new

    def train_batch(self, Xb, yb, shuffle_dfa=False, rng=None):
        recs = []
        for i in range(len(Xb)):
            sp, vv, acts = self._forward_record(Xb[i])
            recs.append((sp, vv, self._logits_from(sp, vv, acts)))
        deltas = []
        for (sp, vv, logits), y in zip(recs, np.asarray(yb)):
            p = _softmax(logits / self.logit_temp)                # temperature -> graded deltas (avoid softmax saturation)
            d = p.copy(); d[int(y)] -= 1.0
            deltas.append(d)
        if shuffle_dfa and rng is not None and len(deltas) > 1:
            deltas = [deltas[j] for j in rng.permutation(len(deltas))]         # credit mismatched to the example
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        leaky = (self.logit_source == "leaky_readout")
        for (sp, vv, _lg), d in zip(recs, deltas):
            self._accum_grad(grads, sp, vv, d, skip_output=leaky)
            if leaky:
                # OUTPUT readout = a clean, well-conditioned softmax-regression delta rule on the STANDARDIZED
                # last-hidden eligibility feature r (the Bellec leaky-readout gradient d_loss/dW = delta_k * r_j).
                r = self._readout_feature(sp)                              # (n_Hlast_phys,)
                dphys = self._broadcast(np.asarray(d, dtype=np.float64), L) / self.pool_k   # (n_out_phys,)
                grads[L - 1] += np.outer(r, dphys)
        self._apply_grads(grads, len(Xb))

    # ---------- prediction / accuracy using the SAME logit source e-prop trained on ----------
    def _predict(self, feat_row):
        sp, vv, acts = self._forward_record(feat_row)
        return int(np.argmax(self._logits_from(sp, vv, acts)))

    def accuracy(self, X, y):
        pred = np.array([self._predict(X[i]) for i in range(len(X))])
        return float(np.mean(pred == np.asarray(y)))

    def acc_on(self, X, y, idx):
        if idx is None or len(idx) == 0:
            return float("nan")
        pred = np.array([self._predict(X[i]) for i in idx])
        return float(np.mean(pred == np.asarray(y)[idx]))


def _train_eprop(net, Xtr, ytr, epochs, batch, seed, shuffle_dfa=False):
    rng = np.random.default_rng(seed + 777)
    if net.logit_source == "leaky_readout":
        net.fit_readout_norm(Xtr)                # fit the readout-feature normalization once on the train set
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            net.train_batch(Xtr[b], ytr[b], shuffle_dfa=shuffle_dfa, rng=rng)


# ============================================================================================================
# Positive control: fit a SMALL train set to HIGH train accuracy (the hard gate -- if this fails the port is broken).
# ============================================================================================================
def positive_control(seed, task_kwargs, n_pos=40, hidden=32, settle=40, eprop_lr=0.5, eps_leak=0.9,
                     surrogate="atan_vt", alpha_surr=0.15, beta_surr=1.0, logit_source="leaky_readout",
                     epochs=200, batch=20, hp=None, n_hidden_layers=1, w_clip=4000.0,
                     ou_noise=False, cond_noise=False, stp=False, task_xor=False):
    # same ADDITIVE default-off task selector as run_seed, so --task-xor --poscontrol-only fits the XOR train set.
    if task_xor:
        (Xtr, ytr, _Ltr), _te, meta, _idx = make_task_xor(seed)
    else:
        (Xtr, ytr, _Ltr), _te, meta, _idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    srng = np.random.default_rng(seed + 31)
    keep = srng.permutation(len(Xtr))[:n_pos]
    Xs, ys = Xtr[keep], ytr[keep]
    chance = float(max(np.mean(ys == c) for c in np.unique(ys)))
    net = OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=n_hidden_layers, settle_steps=settle,
                           eprop_lr=eprop_lr, eps_leak=eps_leak, surrogate=surrogate, alpha_surr=alpha_surr,
                           beta_surr=beta_surr, logit_source=logit_source, w_clip=w_clip, hp=hp,
                           ou_noise=ou_noise, cond_noise=cond_noise, stp=stp)
    w0 = net.ff_weight_norm()
    acc0 = net.accuracy(Xs, ys)
    _train_eprop(net, Xs, ys, epochs, batch, seed)
    acc1 = net.accuracy(Xs, ys)
    w1 = net.ff_weight_norm()
    return {"n_pos": int(n_pos), "k": int(k), "chance": chance, "train_acc_before": acc0,
            "train_acc_after": acc1, "ff_weight_moved": float(abs(w1 - w0)),
            "surrogate": surrogate, "alpha_surr": alpha_surr, "logit_source": logit_source,
            "eprop_lr": eprop_lr, "eps_leak": eps_leak, "epochs": epochs,
            "ou_noise": bool(ou_noise), "cond_noise": bool(cond_noise), "stp": bool(stp),
            "passes": bool(acc1 >= max(0.70, chance + 0.30))}


# ============================================================================================================
# Full seed: stage-0 depth gate + rate oracle + e-prop train + inherit held-out + permuted + shuffle-DFA controls.
# ============================================================================================================
def run_seed(seed, hidden, settle, epochs, batch, eprop_lr, eps_leak, surrogate, alpha_surr, beta_surr,
             logit_source, w_clip, train_subsample, task_kwargs, hp=None, n_hidden_layers=2, pool_k=1, reservoir_control=True,
             ou_noise=False, cond_noise=False, stp=False, task_xor=False):
    # TASK SELECTOR (ADDITIVE, default OFF => byte-identical): swap ONLY the task. task_xor=True uses the
    # NON-reservoir-decodable depth-2 XOR->threshold task (make_task_xor); everything downstream -- stage0 depth gate,
    # oracle, e-prop arm, permuted, shuffle-DFA, AND the reservoir_control/deep_credit_share machinery -- is UNCHANGED.
    # The whole point: on a task a fixed random reservoir CANNOT shortcut, does deep_credit_share become clearly
    # positive (training the hidden FF pathways is load-bearing), where inheritance gave ~0?
    if task_xor:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    else:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    inh_idx = idx["inh_idx"]
    # STAGE-0 depth gate (rate oracle; FULL train set) -- validates the TASK CONFIG is genuinely depth-required.
    s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96, epochs=250,
                                  lr=0.3, batch=128, seed=seed)
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    # rate oracle ceiling (depth-2), FULL train set.
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)

    # subsample the TRAIN set for the on-bridge spiking arms (held-out NEVER subsampled).
    if train_subsample and len(Xtr) > train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:train_subsample]
        Xtr_b, ytr_b = Xtr[keep], ytr[keep]
    else:
        Xtr_b, ytr_b = Xtr, ytr

    def _mk():
        return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=n_hidden_layers, settle_steps=settle,
                                eprop_lr=eprop_lr, eps_leak=eps_leak, surrogate=surrogate, alpha_surr=alpha_surr,
                                beta_surr=beta_surr, logit_source=logit_source, w_clip=w_clip, hp=hp, pool_k=pool_k,
                                ou_noise=ou_noise, cond_noise=cond_noise, stp=stp)

    # --- main e-prop arm ---
    net = _mk(); w0 = net.ff_weight_norm()
    _train_eprop(net, Xtr_b, ytr_b, epochs, batch, seed)
    train_acc = net.accuracy(Xtr_b, ytr_b)
    inh_acc = net.acc_on(Xte, yte, inh_idx)
    ff_moved = float(abs(net.ff_weight_norm() - w0))

    # --- permuted-label control -> ~chance (no leakage) ---
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    pnet = _mk(); _train_eprop(pnet, Xtr_b, yperm, epochs, batch, seed)
    perm_inh = pnet.acc_on(Xte, yte, inh_idx)

    # --- shuffle-DFA control -> ~chance (the DFA credit is load-bearing) ---
    snet = _mk(); _train_eprop(snet, Xtr_b, ytr_b, epochs, batch, seed, shuffle_dfa=True)
    shuf_inh = snet.acc_on(Xte, yte, inh_idx)

    # --- RESERVOIR CONTROL -> the deep credit must BEAT a frozen-hidden baseline (added 2026-07-16) ---
    # WHY THIS EXISTS: until today this gate had NO frozen-hidden arm, so a result that was mostly a FIXED RANDOM
    # SPIKING RESERVOIR + a trained linear readout passed it UNCHANGED -- and the banked headline ("feedforward
    # spiking deep credit is ALREADY GO, K=8 0.877") turned out to be ~80% exactly that (measured: FULL 0.889 vs
    # FROZEN 0.778 vs chance 0.333 => the reservoir is 80% of the margin; deep credit adds ~20%, seed-variable
    # +0.037..+0.185). The isolation hook (train_layers, :153) had been written FOR THIS and never once invoked.
    # A gate that can pass WITHOUT this control cannot distinguish "the network learned deeply" from "a random
    # projection plus logistic regression" -- so the control is now DEFAULT-ON and part of `trains`.
    froz_inh = float("nan"); deep_share = float("nan")
    if reservoir_control:
        fnet = _mk()
        fnet.train_layers = {fnet.n_hidden_layers}   # train ONLY the linear readout; hidden FF frozen at init
        _train_eprop(fnet, Xtr_b, ytr_b, epochs, batch, seed)
        froz_inh = fnet.acc_on(Xte, yte, inh_idx)
        if not (np.isnan(inh_acc) or np.isnan(froz_inh)) and (inh_acc - chance) > 1e-9:
            deep_share = float((inh_acc - froz_inh) / (inh_acc - chance))   # fraction of the margin that is DEEP

    trains = bool((not np.isnan(inh_acc)) and inh_acc > chance + 0.05 and inh_acc > perm_inh + 0.05
                  and inh_acc > shuf_inh + 0.05
                  and ((not reservoir_control) or (not np.isnan(froz_inh) and inh_acc > froz_inh + 0.05)))
    return {"seed": seed, "chance": chance, "k_classes": int(k), "n_train_smoke": int(len(ytr_b)),
            "stage0_depth_separating": bool(s0.get("depth_separating")),
            "stage0_deep_best": s0.get("deep_best_inherit_heldout"), "stage0_l1": s0.get("l1_inherit_heldout"),
            # stage0_l0 = the LINEAR (no-hidden) floor. stage0_depth_genuineness COMPUTES it
            # (`linear_inherit_heldout`) and this record used to THROW IT AWAY, so the control ladder
            # never reached any output file. It is the cheapest possible rebuttal to "the task is linearly
            # trivial" -- measured 0.265 (BELOW chance 0.333) at this config, which is what rules that out.
            # Ladder: chance -> l0 linear -> l1 trained-shallow -> frozen_hidden (random-deep) -> inherit (learned-deep).
            "stage0_l0_linear": s0.get("linear_inherit_heldout"),
            "oracle_inherit": oracle_inh, "eprop_train_acc": train_acc, "eprop_inherit_heldout": inh_acc,
            "eprop_ff_weight_moved": ff_moved, "permuted_inherit": perm_inh, "shuffle_dfa_inherit": shuf_inh,
            # frozen_hidden_inherit = the RESERVOIR baseline. reservoir_control_run=False means the gate could NOT
            # distinguish deep credit from a random projection + logistic regression -- a GO is then NOT a
            # deep-credit claim, and must not be reported as one.
            "frozen_hidden_inherit": froz_inh, "deep_credit_share": deep_share,
            "reservoir_control_run": bool(reservoir_control),
            "trains_the_task": trains}


def main():
    ap = argparse.ArgumentParser(description="Port the validated e-prop rule onto the Izhikevich OnBridgeBDSPNet.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5, help="readout lr (hidden lr = eprop_lr*hidden_lr_scale)")
    ap.add_argument("--eps-leak", type=float, default=0.9)
    ap.add_argument("--surrogate", choices=["atan_vt", "std"], default="atan_vt")
    ap.add_argument("--alpha-surr", type=float, default=0.15, help="atan surrogate slope (1/mV); atan((v-vt)*alpha)")
    ap.add_argument("--beta-surr", type=float, default=1.0, help="std surrogate slope")
    ap.add_argument("--logit-source", choices=["spike_sum", "event_rate", "membrane", "leaky_readout"],
                    default="leaky_readout")
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--train-subsample", type=int, default=240)
    ap.add_argument("--pool-k", type=int, default=1)
    # SUBSTRATE-DECORRELATION knobs (2026-08-01): reachable at last. `enable_ou_process` /
    # `enable_conductance_noise` give each neuron INDEPENDENT background current/conductance, so a pooled
    # population is DECORRELATED and averaging reduces the Izhikevich forward noise by ~sqrt(K)
    # (Destexhe-Rudolph high-conductance state). The 07-14 on-bridge residual was exactly that forward noise;
    # the banked K=8 "closure" ran with BOTH at their False default and UNRECORDED, so the sqrt-K decorrelation
    # it needs may never have been on. These flags make the biology-prescribed fix runnable AND recorded.
    # (knob_reachable / CLASS-KR flagged this file: a cfg-writing param with no --flag is a knob nobody can set.)
    ap.add_argument("--ou-noise", action="store_true",
                    help="independent OU background current per neuron (decorrelates the pool for sqrt-K averaging)")
    ap.add_argument("--cond-noise", action="store_true",
                    help="independent conductance-based background noise per neuron (same decorrelation role)")
    ap.add_argument("--stp", action="store_true",
                    help="short-term plasticity (Tsodyks-Markram); OFF by default -- STP depression throttles FF drive")
    ap.add_argument("--freeze-hidden", action="store_true",
                    help="THE MISSING RESERVOIR CONTROL: train ONLY the last FF pathway (the host-side linear "
                         "softmax readout, which _accum_grad already SKIPS from the e-prop/DFA rule) and FREEZE "
                         "the hidden FF pathways at init, via the runner's own train_layers hook (:153). The GO "
                         "metric trains_the_task gates on chance/permuted/shuffle-DFA -- NONE is a frozen-hidden "
                         "baseline -- so a pure reservoir+readout result passes it UNCHANGED. If readout-only ~= "
                         "full, the hidden deep credit contributes NOTHING and the GO is a reservoir result.")
    ap.add_argument("--bdsp-wmax", type=float, default=None,
                    help="Widen the inherited bdsp_w_min/max = -6/+6 (parent :183) so fused_bdsp_update's "
                         "unconditional cp.clip becomes a NO-OP while the block still runs (cp_bdsp_E stays "
                         "available). The single-variable clamp lever; try 1e9. Default None = byte-identical.")
    ap.add_argument("--no-bdsp", action="store_true",
                    help="Set cfg.enable_bdsp=False AFTER init -- the kernel's OWN documented inertness lever "
                         "(\"byte-inert when enable_bdsp is False: the block is unreached\", kernels.py:480). "
                         "This runner instead keeps enable_bdsp=True and relies on lr=0, but fused_bdsp_update "
                         "ENDS in `return cp.clip(w_new, w_min, w_max)` (kernels.py:485) -- at eta=0 the ADD "
                         "term vanishes, the CLIP does NOT. With the inherited bdsp_w_max=6.0 vs this runner's "
                         "--w-clip 4000, every forward silently crushes the FF synapses whose presyn fired "
                         "(measured 239/512, mean |w| 370 -> <=6). Default off = byte-identical to the banked runs.")
    # positive control knobs
    ap.add_argument("--poscontrol-only", action="store_true")
    ap.add_argument("--pos-n", type=int, default=40)
    ap.add_argument("--pos-epochs", type=int, default=200)
    ap.add_argument("--pos-batch", type=int, default=20)
    ap.add_argument("--pos-hidden-layers", type=int, default=1,
                    help="positive-control depth (1 = the clean mechanism validation; the task uses --n-hidden-layers)")
    # tonic/drive hp (the working regime: STP off + wash-out + strong FF + leaky readout)
    ap.add_argument("--tonic-h-pA", type=float, default=100.0)
    ap.add_argument("--tonic-o-pA", type=float, default=150.0)
    ap.add_argument("--ff-w-init", type=float, default=2000.0)
    ap.add_argument("--in-current-pA", type=float, default=700.0)
    ap.add_argument("--in-bias-pA", type=float, default=300.0)
    ap.add_argument("--hidden-lr-scale", type=float, default=5.0)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    # TASK SELECTOR (additive, default OFF => byte-identical to the banked inheritance runs). --task-xor swaps in the
    # NON-reservoir-decodable depth-2 XOR->threshold task (reuse-by-import of make_task_xor). On this task a fixed random
    # reservoir CANNOT shortcut the label, so if deep_credit_share becomes clearly positive here (where it read ~0 on the
    # linearly-reservoir-decodable inheritance task) then training the hidden FF pathways is LOAD-BEARING on the
    # production Izhikevich bridge. The --n-super/--n-members/... knobs below are ignored when --task-xor is set.
    ap.add_argument("--task-xor", action="store_true",
                    help="use the non-reservoir-decodable depth-2 XOR->threshold task (emerge1 make_task via "
                         "make_task_xor) instead of the semantic-inheritance task; the deep_credit_share test on a "
                         "task a frozen reservoir cannot shortcut. Additive, default off = byte-identical.")
    # task knobs (the onbridge-derisk 5-class depth-separating smoke default)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha,
              in_current_pA=a.in_current_pA, in_bias_pA=a.in_bias_pA, hidden_lr_scale=a.hidden_lr_scale, no_bdsp=a.no_bdsp, bdsp_wmax=a.bdsp_wmax, freeze_hidden=a.freeze_hidden)
    t0 = time.time()

    if a.poscontrol_only:
        err = None; pcs = []
        try:
            for s in a.seeds:
                pc = positive_control(s, task_kwargs, n_pos=a.pos_n, hidden=a.hidden, settle=a.settle_steps,
                                      eprop_lr=a.eprop_lr, eps_leak=a.eps_leak, surrogate=a.surrogate,
                                      alpha_surr=a.alpha_surr, beta_surr=a.beta_surr, logit_source=a.logit_source,
                                      epochs=a.pos_epochs, batch=a.pos_batch, hp=hp,
                                      n_hidden_layers=a.pos_hidden_layers, w_clip=a.w_clip,
                                      ou_noise=a.ou_noise, cond_noise=a.cond_noise, stp=a.stp, task_xor=a.task_xor)
                pcs.append(pc)
                print(f"[POS-CTRL seed {s}] surrogate={a.surrogate} alpha={a.alpha_surr} logit={a.logit_source} lr={a.eprop_lr} "
                      f"| fit-{pc['n_pos']} train {pc['train_acc_before']:.2f}->{pc['train_acc_after']:.2f} "
                      f"(chance {pc['chance']:.2f}) ff-moved {pc['ff_weight_moved']:.2f} => "
                      f"{'PASS' if pc['passes'] else 'FAIL'}", flush=True)
        except Exception as e:
            err = repr(e); traceback.print_exc()
        out = {"probe": "onbridge_eprop_port_poscontrol", "seeds": a.seeds,
               "config": {"hidden": a.hidden, "settle": a.settle_steps, "surrogate": a.surrogate,
                          "alpha_surr": a.alpha_surr, "logit_source": a.logit_source, "eprop_lr": a.eprop_lr,
                          "eps_leak": a.eps_leak, "pos_epochs": a.pos_epochs, "task": task_kwargs,
                          "task_xor": bool(a.task_xor),
                          "ou_noise": bool(a.ou_noise), "cond_noise": bool(a.cond_noise), "stp": bool(a.stp)},
               "elapsed_seconds": round(time.time() - t0, 1), "positive_control": pcs,
               "error": err, "PASSES": bool(pcs and all(p["passes"] for p in pcs))}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n[onbridge-eprop-port] POSITIVE CONTROL {'PASSES' if out['PASSES'] else 'FAILS'} -> wrote {a.out}", flush=True)
        return 0 if out["PASSES"] else 1

    err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr, a.eps_leak, a.surrogate,
                         a.alpha_surr, a.beta_surr, a.logit_source, a.w_clip, a.train_subsample, task_kwargs, hp=hp,
                         n_hidden_layers=a.n_hidden_layers, pool_k=a.pool_k,
                         ou_noise=a.ou_noise, cond_noise=a.cond_noise, stp=a.stp, task_xor=a.task_xor)
            per.append(r)
            # PER-SEED CHECKPOINT (2026-07-16): previously --out was written ONCE, after ALL seeds (line ~701), so
            # any interruption -- a reboot, an OOM, a kill -- destroyed the entire arm's work. A 3-seed arm is ~3h
            # at ~62 min/seed, so that is a 3-hour loss to preserve nothing. Measured the day it bit: a 4-arm sweep
            # was killed for a reboot at 49 min with 0/3 seeds done in every arm and NOT ONE byte on disk.
            # Cost is a sub-millisecond JSON write per ~hour of compute. `partial: True` marks it as incomplete so a
            # truncated file can never be mistaken for a finished run (the day's own rule: an exit without the
            # success marker is a FAILURE, never a pass).
            try:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(
                    {"probe": "onbridge-eprop-port", "partial": True, "seeds_done": [x["seed"] for x in per],
                     "seeds_requested": list(a.seeds), "config": vars(a), "per_seed": per,
                     "SIGNAL": None, "verdict": "PARTIAL -- run still in progress; NOT a verdict."},
                    indent=2, default=str))
            except Exception as _ck:   # a checkpoint must never take the run down with it
                print(f"[warn] per-seed checkpoint failed ({type(_ck).__name__}: {_ck})", flush=True)
            print("-" * 100, flush=True)
            print(f"[seed {s}] k={r['k_classes']} chance {r['chance']:.3f} | STAGE0 depth-sep {r['stage0_depth_separating']} "
                  f"(deep-best {r['stage0_deep_best']:.3f} vs 1-layer {r['stage0_l1']:.3f}) | oracle {r['oracle_inherit']:.3f}", flush=True)
            print(f"  e-prop ON BRIDGE: train {r['eprop_train_acc']:.3f} | inherit-heldout {r['eprop_inherit_heldout']:.3f} "
                  f"| ff-moved {r['eprop_ff_weight_moved']:.1f}", flush=True)
            print(f"  [controls] permuted {r['permuted_inherit']:.3f} | shuffle-DFA {r['shuffle_dfa_inherit']:.3f} "
                  f"| chance {r['chance']:.3f} => TRAINS-THE-TASK {r['trains_the_task']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "onbridge_eprop_port", "seeds": a.seeds,
               # PROVENANCE (2026-07-16): record EVERY knob that changes the experiment. `pool_k` and `freeze_hidden`
               # were MISSING here while `--pool-k` DEFAULTS TO 1 and the whole arc runs at 8 -- so a file that does
               # not mention pool_k is INDISTINGUISHABLE from a pool_k=1 run, and the ONLY provenance was the string
               # "k8" in a filename. Recovering it for `_eprop_banked_{FULL,FROZEN}.json` needed forensics (the
               # bridge's synapse count: 1408 @ k=1, 22528 @ k=4, 90112 @ k=8 -- exact k^2 scaling). An absent flag
               # means DEFAULT, not off; if a config must be reconstructed from a filename, the record is broken.
               "config": {"hidden": a.hidden, "n_hidden_layers": a.n_hidden_layers, "settle": a.settle_steps,
                          "epochs": a.epochs, "batch": a.batch, "eprop_lr": a.eprop_lr, "eps_leak": a.eps_leak,
                          "surrogate": a.surrogate, "alpha_surr": a.alpha_surr, "logit_source": a.logit_source,
                          "w_clip": a.w_clip, "train_subsample": a.train_subsample, "task": task_kwargs,
                          "pool_k": a.pool_k, "freeze_hidden": bool(a.freeze_hidden), "task_xor": bool(a.task_xor),
                          "ou_noise": bool(a.ou_noise), "cond_noise": bool(a.cond_noise), "stp": bool(a.stp),
                          "reservoir_control": True, "hidden_lr_scale": a.hidden_lr_scale,
                          "no_bdsp": bool(a.no_bdsp), "bdsp_wmax": a.bdsp_wmax,
                          "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(key):
            return float(np.nanmean([p[key] for p in per]))
        s0_sep = all(p["stage0_depth_separating"] for p in per)
        ch = _m("chance"); tr = _m("eprop_train_acc"); inh = _m("eprop_inherit_heldout")
        perm = _m("permuted_inherit"); shuf = _m("shuffle_dfa_inherit"); orc = _m("oracle_inherit")
        ff = _m("eprop_ff_weight_moved")
        trains = all(p["trains_the_task"] for p in per)
        permuted_chance = bool(perm <= ch + 0.10)
        shuffle_chance = bool(shuf <= ch + 0.10)
        ff_moves = bool(ff > 1e-3)
        summary["aggregate"] = {"chance": ch, "oracle_inherit": orc, "eprop_train_acc": tr,
                                "eprop_inherit_heldout": inh, "permuted_inherit": perm, "shuffle_dfa_inherit": shuf,
                                "ff_weight_moved": ff, "stage0_depth_separating": s0_sep,
                                "trains_the_task_all_seeds": trains, "permuted_chance": permuted_chance,
                                "shuffle_dfa_chance": shuffle_chance, "ff_weight_moves": ff_moves}
        signal = bool(s0_sep and ff_moves and trains and permuted_chance and shuffle_chance)
        summary["SIGNAL"] = signal
        if not ff_moves:
            summary["verdict"] = (f"RED FLAG -- e-prop moved ~no FF weight ({ff:.1e}); the weight-write is broken. NOT a "
                                  f"verdict. Fix the port before concluding.")
        elif signal:
            summary["verdict"] = (
                f"GO -- the transport-free biological e-prop rule TRAINS the depth-2 compositional-inheritance task ON "
                f"THE PRODUCTION IZHIKEVICH BRIDGE where the committed BDSP rule fails 0/6: train {tr:.3f}, inherit "
                f"{inh:.3f} >> chance {ch:.3f} (oracle {orc:.3f}); controls clean (permuted {perm:.3f}, shuffle-DFA "
                f"{shuf:.3f}, both ~chance; ff-moved {ff:.1f}); NO weight transport (B_direct separate seed stream). "
                f"=> the emergence engine's core LOCAL learning mechanism works on the production substrate.")
        else:
            summary["verdict"] = (
                f"HONEST NEGATIVE -- the ported e-prop does NOT cleanly train the task on the bridge: train {tr:.3f}, "
                f"inherit {inh:.3f} (chance {ch:.3f}, oracle {orc:.3f}); permuted {perm:.3f}, shuffle-DFA {shuf:.3f}. "
                f"(Positive control gates whether the port itself is sound -- run --poscontrol-only.) The exact residual: "
                f"{'inherit not above chance/controls' if inh <= ch + 0.05 else 'controls not clean'}.")
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[onbridge-eprop-port] {summary['verdict']}", flush=True)
    print(f"[onbridge-eprop-port] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
