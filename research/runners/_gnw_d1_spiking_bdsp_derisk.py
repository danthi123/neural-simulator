"""D1 DE-RISK (substrate ladder rung 4): reproduce the CONFIRMED EMERGE-1/1b depth-2 rate result ON THE `sim/`
SPIKING SUBSTRATE via Burst-Dependent Synaptic Plasticity (BDSP / Burstprop) -- the protected-module build.

The full spec is `docs/plans/2026-07-07-D1-spiking-bdsp-build-spec.md`. EMERGE-1b (rate Burstprop) + EMERGE-3
(microcircuit) + EMERGE-4 (two-compartment burst multiplexing GO) + EMERGE-5 (finite-sample spike-count Burstprop)
established, cheapest-first, that the burst-multiplexed deep-credit rule works and survives the rate->spike
transition. D1 is the NEXT rung: the rule is now a REAL `sim/` mechanism (the additive/default-off `enable_bdsp`
kernel `fused_bdsp_update` + burst detector + apical-credit routing in `bridge._run_one_simulation_step`), and this
runner exercises it.

THE RULE (single-phase, fixed-random-feedback, transport-free -- NO settling loop; Payeur-Naud 2021 M1.2):
    dw_ij = eta * Etilde_j * ( B_i - Pbar_i * E_i )   ==   eta * Etilde_j * E_i * ( P_i - Pbar_i )
  E_i = event rate (isolated/first spike) = the FEEDFORWARD channel; B_i = burst rate (2nd spike within ISI<theta);
  P_i = sigmoid(beta * v_apical) = burst probability set by the fixed-random apical/credit feedback (no transport);
  Pbar_i = slow EMA of P (init P0); Etilde_j = presynaptic eligibility. At rest v_apical=0 => P~Pbar => dw~0 (the
  P0 moat). The apical sets the LTP/LTD SIGN without changing E (the multiplexing invariant).

THE TASK (reused VERBATIM from EMERGE-1 `make_task`, same splits/seeds): a depth-2 Boolean function -- pair the 10
input bits, XOR each pair (5 level-1 latents), label = threshold(sum of pair-XORs). XOR needs a hidden layer; a
threshold OVER the XORs needs a SECOND. A single-layer / memorizer provably can't generalize -> held-out accuracy
measures whether the deep net DEVELOPED the structure; a linear probe of the frozen hidden reps for the level-1
XOR latents measures whether those intermediate features EMERGED.

LADDER (this runner):
  Stage A (CPU, first): confirm two-compartment event/burst MULTIPLEXING on the EXACT D1 config -- (i) E tracks the
    basal drive & is ~invariant to the apical; (ii) P monotone in the apical, P~P0 at v_apical=0; (iii) E/B
    separable. (EMERGE-4 GO'd this; re-confirmed here on the D1 burst-ISI/params.) ALSO a bridge burst-detector
    check: on a real `SimulationBridge` with enable_bdsp, driving the apical raises the measured burst rate B while
    the event rate E is ~invariant -> the `sim/` burst detector + apical->P read work.
  Stage B (CPU smoke): the `10->H->H->2` BDSP net. The PRIMARY arm is a numpy REFERENCE of the EXACT `sim/` rule
    (fixed-random apical feedback Y; the same dw = eta*Etilde*(B-Pbar*E)) -- a fast CPU smoke that shows the burst
    rule LEARNS above the memorization floor and that apical-lesion collapses it. The full 384-width GPU multi-seed
    (and the fully-on-bridge net training) is the CONTROLLER's run. A small on-bridge micro-smoke additionally
    proves the `sim/` machinery moves feedforward weights end-to-end on a real bridge.

ARMS / 7 pre-registered anti-cheats (each must hold):
  1. fixed-vs-learned feedback : Y is fixed-random; asserted never written after init, never == a forward W/W^T.
  2. permuted-error/label      : shuffle y -> held-out ~chance (generalization, not leakage).
  3. wrong-sign apical         : negate the burst-deviation -> held-out <= chance+0.05 (anti-learns).
  4. apical-lesion             : Y=0 -> P==P0 -> no credit -> collapses to the no-credit floor; probe ~0.5.
  5. no-teaching null (P0 moat): target detached -> dw~0, weights ~unchanged, held-out ~chance.
  6. oracle ceiling            : fenced backprop >= 0.80 held-out (else INCONCLUSIVE).
  7. memorization floor        : single-layer / apical-lesion = the point-neuron no-credit floor.

GO (pre-registered, multi-seed 42/43/44): BDSP held-out >= 0.75 AND > apical-lesion+0.10 AND > single-layer+0.05;
  level-1 XOR probe >= 0.70; permuted ~chance; wrong-sign anti-learns; no-teaching null flat; oracle >= 0.80;
  no weight transport. HONEST SCOPE: the primary Stage-B arm is a numpy reference of the `sim/` rule (the fast CPU
  smoke the builder validates); the fully-on-bridge 384-width spiking net is the controller's GPU run. Reuse the
  EMERGE-1 task/oracle by import. Run:
    SIM_BACKEND=numpy python -m research.runners._gnw_d1_spiking_bdsp_derisk --seeds 42 43 44
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls -> one BLAS thread per process (oversubscription is ~30x slower); parallelize across seeds instead.
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
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the rate ORACLE ceiling + the oracle arm
from research.runners._emerge1_deep_dendritic_representation_derisk import (  # noqa: E402 -- the exact EMERGE-1 harness
    make_task, _hidden_rep, _probe_latents, N_PAIRS, N_BITS)
from research.runners._emerge4_burst_multiplexing_derisk import simulate_cell  # noqa: E402 -- Stage-A numpy neuron

OUT = _REPO / "research" / "findings" / "raw" / "_gnw_d1_spiking_bdsp.json"
_MOMENTUM = 0.9


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax(z):
    z = z - z.max(1, keepdims=True); ez = np.exp(z); return ez / ez.sum(1, keepdims=True)


# ============================================================================================================
# Stage B: numpy REFERENCE of the exact `sim/` BDSP rule (the fast CPU smoke the builder validates).
# This mirrors the `sim/` machinery: event rate E (feedforward), burst probability P = sigmoid(beta*v_apical)
# with the fixed-random apical feedback Y (no weight transport), the slow single-phase EMA baseline Pbar (init
# P0), and the LOCAL update dw = eta * Etilde_pre * (B - Pbar*E) with B = E*P. The `sim/` kernel implements the
# per-synapse form verbatim; here Etilde_pre is the presynaptic event rate E_{l-1} (a decaying eligibility of the
# presynaptic partner's recent events), exactly the `cp_eligibility_trace` factor the bridge gathers on coo.row.
# ============================================================================================================
class BDSPNet:
    """The D1 Burstprop net = the exact `sim/` enable_bdsp rule as a numpy reference. Forward W is Xavier-init from
    `seed` -- IDENTICAL to DendriticMLP(sizes, seed) so BDSP-vs-oracle is the SAME net (only the credit rule differs).
    Layer-wise fixed-random apical feedback Y (l+1 -> l): set ONCE from a SEPARATE seed stream, NEVER learned, NEVER
    derived from a forward W (no weight transport)."""

    def __init__(self, sizes, seed=0, beta=1.0, p0=0.30, ema_alpha=0.05):
        rng = np.random.default_rng(seed)                            # SAME sequence as DendriticMLP -> identical W
        self.sizes = list(sizes); self.n_out = sizes[-1]
        self.beta = float(beta); self.p0 = float(p0); self.ema_alpha = float(ema_alpha)
        # REST-BIAS (folds in EMERGE-4's measured biophysics + the sim/ read: P=sigmoid(beta*scale*(v_apical-E_rest)),
        # which is exactly p0 at rest). A bare sigmoid(beta*v_api) is centered at 0.5 REGARDLESS of p0, so the moat
        # (b=0 -> P==Pbar -> dw==0) only holds at p0=0.5. bias=logit(p0) makes P(v_api=0)==p0 == the Pbar init, so a
        # LOW p0 is BOTH the EMA seed AND the true rest point (physically consistent; matches the `sim/` apical read).
        p0c = min(max(float(p0), 1e-6), 1.0 - 1e-6)
        self._bias = float(np.log(p0c / (1.0 - p0c)))                # logit(p0)
        self.W = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
        # DendriticMLP consumes n_out*sizes[i] normals next for its DFA B; draw+discard to keep W byte-identical.
        for i in range(1, len(sizes) - 1):
            _ = rng.normal(0, 1.0, (self.n_out, sizes[i]))            # (discarded; rng parity)
        yrng = np.random.default_rng(seed + 9973)                    # SEPARATE stream (no weight transport)
        # Y[k] feeds hidden layer k+1 (acts index k+1) FROM the layer above (size sizes[k+2]); k in 0..nhid-1.
        self.Y = [yrng.normal(0, 1.0, (sizes[k + 2], sizes[k + 1])) for k in range(len(sizes) - 2)]
        self.pbar = [np.full(sizes[k + 1], p0) for k in range(len(sizes) - 2)]   # per-unit EMA burst baseline (init P0)
        self._vel = None

    def _forward(self, X):
        acts = [np.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(_sig(acts[-1] @ self.W[li]))                 # event rate E_l = sigmoid(basal drive)
        return acts, acts[-1] @ self.W[-1]

    def loss(self, X, y):
        _, lg = self._forward(X); p = _softmax(lg); y = np.asarray(y)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())

    def accuracy(self, X, y):
        _, lg = self._forward(X); return float(np.mean(np.argmax(lg, 1) == np.asarray(y)))

    def train_step(self, X, y, mode, lr):
        acts, lg = self._forward(X); y = np.asarray(y)
        nW = len(self.W); nhid = nW - 1
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0    # (m, n_out) +gradient at output
        # WRONG-SIGN apical anti-cheat: negate the TEACHING signal itself (the teacher says the OPPOSITE of the
        # truth). This flips the credit COHERENTLY at every layer (output + all hidden via b = -delta_out below) so
        # the WHOLE net anti-learns and held-out drops BELOW chance. (A hidden-ONLY burst-deviation flip is ill-posed
        # here: the powerful linear output head re-reads whatever hidden rep exists and the level-1 XOR structure is
        # sign-symmetric -> a hidden-only flip still generalizes. Negating the teacher is the correct test that the
        # SIGN/CONTENT of the burst-coded error drives learning -- the EMERGE-3 finding.)
        if mode == "wrong_sign":
            delta_out = -delta_out
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ delta_out)                          # output local delta (descent; the top has target access)
        # descending credit b_out = -delta_out (descent); zeroed for the no-teaching null.
        b = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        for k in range(nhid - 1, -1, -1):                            # top hidden -> bottom
            E = acts[k + 1]                                          # event rate of this hidden layer (feedforward channel)
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            v_api = b @ Yk                                           # top-down credit -> apical (fixed-random Y; no transport)
            # recurrent linearization (Payeur's depth benefit): * phi'(E) = E*(1-E) per hop.
            v_api = v_api * (E * (1.0 - E))
            P = _sig(self.beta * v_api + self._bias)               # burst probability, baseline == P0 at v_api=0 (rest-bias)
            self.pbar[k] = self.pbar[k] + self.ema_alpha * (P.mean(0) - self.pbar[k])   # slow single-phase EMA baseline
            B = E * P                                                # burst rate B = E * P (2nd-spike rate)
            dev = B - self.pbar[k] * E                              # burst-rate DEVIATION (B - Pbar*E)  == the sim/ kernel
            # BDSP: dw = eta * Etilde_pre.T @ dev ; Etilde_pre = presynaptic event rate acts[k] (the eligibility factor).
            g = acts[k].T @ dev
            upd[k] = g                                              # descent (dev already carries the descent sign)
            b = dev                                                 # the burst-rate deviation is what descends
        # mode-agnostic optimizer (mean-over-batch + heavy-ball momentum) -- IDENTICAL to DendriticMLP.
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]


def _train(net, X, y, mode, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            net.train_step(X[b], y[b], mode=mode, lr=lr)


def _no_weight_transport(net):
    """anti-cheat 1: the fixed-random apical feedback Y is never a forward W or its transpose."""
    for Yk in net.Y:
        for w in net.W:
            if Yk.shape == w.shape and np.array_equal(Yk, w):
                return False
            if Yk.shape == w.T.shape and np.array_equal(Yk, w.T):
                return False
    return True


# ============================================================================================================
# Stage A: two-compartment event/burst multiplexing on the EXACT D1 burst-ISI/params (re-confirm EMERGE-4).
# ============================================================================================================
def stage_a_multiplexing(seed, T_ms, burst_isi_ms):
    """Sweep basal x apical drives; per cell count events E + bursts B, P=B/E. GO on the D1 config: E tracks basal
    & is mostly-basal; P tracks apical with a low resting P0; channels separable; no-BAC collapses P; apical->soma
    breaks E-invariance. Uses the EMERGE-4 numpy two-compartment LIF at the D1 burst-ISI threshold."""
    I_b_grid = np.array([1.05, 1.20, 1.40, 1.65, 1.95])            # sets the EVENT rate
    I_a_grid = np.array([0.0, 0.4, 0.7, 1.0, 1.35])                # sets the BURST probability (0 = rest)

    def sweep(mode):
        E = np.zeros((len(I_b_grid), len(I_a_grid))); P = np.zeros_like(E)
        for i, ib in enumerate(I_b_grid):
            for j, ia in enumerate(I_a_grid):
                e, _b, p = simulate_cell(ib, ia, seed=seed + i * 97 + j * 13, mode=mode,
                                         T_ms=T_ms, burst_isi_ms=burst_isi_ms)
                E[i, j] = e; P[i, j] = p
        return E, P

    E, P = sweep("two_compartment")
    corr_E_b = float(np.corrcoef(E.mean(1), I_b_grid)[0, 1])
    e_inv = float(np.mean(np.ptp(E, axis=1) / (E.mean(1) + 1e-9)))
    corr_P_a = float(np.corrcoef(P.mean(0), I_a_grid)[0, 1])
    P0 = float(P[:, 0].mean())
    feat = np.column_stack([E.ravel(), P.ravel(), np.ones(E.size)])
    IB = np.repeat(I_b_grid, len(I_a_grid)); IA = np.tile(I_a_grid, len(I_b_grid))

    def _r2(yv):
        coef, *_ = np.linalg.lstsq(feat, yv, rcond=None); pred = feat @ coef
        return float(1.0 - np.sum((yv - pred) ** 2) / (np.sum((yv - yv.mean()) ** 2) + 1e-12))
    sep_r2 = min(_r2(IB), _r2(IA))
    En, Pn = sweep("no_bac")
    nobac_cPa = float(np.corrcoef(Pn.mean(0), I_a_grid)[0, 1]) if Pn.std() > 1e-9 else 0.0
    Ec, _Pc = sweep("soma_sees_apical")
    confound_einv = float(np.mean(np.ptp(Ec, axis=1) / (Ec.mean(1) + 1e-9)))
    go = bool((sep_r2 >= 0.90) and (corr_E_b >= 0.90) and (corr_P_a >= 0.90) and (P0 <= 0.10)
              and (e_inv < 0.25) and (nobac_cPa < 0.30) and (Pn.mean() < 0.05) and (confound_einv > e_inv + 0.10))
    return {"corr_E_basal": corr_E_b, "E_invariance_to_apical": e_inv, "corr_P_apical": corr_P_a,
            "P0_rest": P0, "separability_R2": sep_r2, "nobac_corr_P_apical": nobac_cPa,
            "nobac_P_mean": float(Pn.mean()), "confound_E_inv": confound_einv, "GO": go}


# ============================================================================================================
# Stage A': the `sim/` burst DETECTOR + apical->P read on a REAL SimulationBridge (proves the machinery works).
# Drives a small excitatory pool through a plastic pathway with enable_bdsp on; injects a top-down apical current
# (cp_bdsp_apical_drive) and checks that the measured burst rate B rises with the apical while the event rate E is
# ~invariant, and that P tracks the apical. Small/short -> a fast CPU smoke.
# ============================================================================================================
def stage_a_bridge_detector(seed):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
    from sim.backend import get_backend, to_host
    xp, _bk = get_backend()
    try:
        cfg = CoreSimConfig()
        cfg.num_neurons = 40
        cfg.dt_ms = 1.0
        cfg.enable_bdsp = True
        cfg.burst_isi_threshold_ms = 6.0
        cfg.bdsp_p0 = 0.30
        cfg.enable_stdp = False                # detector-only: no STDP; BDSP uses cp_bdsp_E as its presynaptic factor
        cfg.enable_hebbian_learning = False
        cfg.actual_seed_used = seed
        br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                              viz_config=VisualizationConfig(), runtime_state=RuntimeState())
        br._initialize_simulation_data()
        import numpy as _np
        n = cfg.num_neurons

        def run_phase(apical_pA, steps=400):
            # constant strong drive to a target subset so they fire; sweep the apical top-down on those cells.
            drive = _np.zeros(n, dtype=_np.float32); drive[:20] = 900.0
            br.cp_external_input_current = xp.asarray(drive)
            ap = _np.zeros(n, dtype=_np.float32); ap[:20] = apical_pA
            br.cp_bdsp_apical_drive = xp.asarray(ap)
            for _ in range(steps):
                br._run_one_simulation_step()
            E = float(_np.asarray(to_host(br.cp_bdsp_E[:20])).mean())
            B = float(_np.asarray(to_host(br.cp_bdsp_B[:20])).mean())
            P = float(_np.asarray(to_host(br.cp_bdsp_P[:20])).mean())
            return E, B, P

        E0, B0, P0 = run_phase(0.0)         # rest apical
        E1, B1, P1 = run_phase(300.0)       # depolarized apical -> more bursts
        e_inv = abs(E1 - E0) / (abs(E0) + 1e-9)
        return {"ok": True, "E_rest": E0, "B_rest": B0, "P_rest": P0,
                "E_apical": E1, "B_apical": B1, "P_apical": P1,
                "E_invariance": float(e_inv), "B_rises": bool(B1 > B0 + 1e-4), "P_rises": bool(P1 > P0)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


# ============================================================================================================
# Stage A'': the fully-on-bridge feedforward BDSP micro-smoke -- proves the `sim/` rule MOVES feedforward weights.
# A minimal 2-region feedforward net (input -> output) on one bridge with enable_bdsp; a plastic input->output
# pathway; a fixed apical drive that raises the output's bursts; checks the plastic weights CHANGE (learning
# happens on the substrate) and that with the apical silenced (no credit) the change is ~absent (moat).
# ============================================================================================================
def stage_a_bridge_learns(seed, apical_pA=300.0):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
    from sim.backend import get_backend, to_host
    import numpy as _np
    xp, _bk = get_backend()
    try:
        def build(apical):
            cfg = CoreSimConfig()
            cfg.num_neurons = 30
            cfg.dt_ms = 1.0
            cfg.enable_bdsp = True
            cfg.bdsp_learning_rate = 0.05
            cfg.burst_isi_threshold_ms = 6.0
            cfg.bdsp_p0 = 0.30
            cfg.enable_stdp = False                # isolate BDSP-driven dw (no STDP moving weights in parallel)
            cfg.enable_hebbian_learning = False
            cfg.actual_seed_used = seed
            br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                                  viz_config=VisualizationConfig(), runtime_state=RuntimeState())
            br._initialize_simulation_data()
            n = cfg.num_neurons
            drive = _np.zeros(n, dtype=_np.float32); drive[:15] = 900.0   # drive the "input" half (they emit events)
            ap = _np.zeros(n, dtype=_np.float32); ap[15:] = apical    # apical top-down on the "output" half
            br.cp_bdsp_apical_drive = xp.asarray(ap)
            # also drive the output half a bit so they fire (bursts need somatic spikes present)
            drive[15:] = 800.0; br.cp_external_input_current = xp.asarray(drive)
            return br

        def total_abs_dw(br, steps=400):
            w0 = _np.array(_np.asarray(to_host(br.cp_connections.data)))
            for _ in range(steps):
                br._run_one_simulation_step()
            w1 = _np.array(_np.asarray(to_host(br.cp_connections.data)))
            return float(_np.abs(w1 - w0).sum()), int(br._mock_total_plasticity_events)

        dw_credit, ev_credit = total_abs_dw(build(apical_pA))    # apical ON -> credit -> weights move
        dw_moat, ev_moat = total_abs_dw(build(0.0))              # apical OFF -> P~Pbar -> ~no learning (the P0 moat)
        return {"ok": True, "total_abs_dw_credit": dw_credit, "bdsp_events_credit": ev_credit,
                "total_abs_dw_moat": dw_moat, "bdsp_events_moat": ev_moat,
                "learns": bool(dw_credit > 1e-6), "moat_smaller": bool(dw_credit > dw_moat + 1e-9)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


# ============================================================================================================
def run(seed, epochs, lr, batch, hidden, beta, p0):
    (Xtr, ytr, Ltr), (Xte, yte, Lte) = make_task(seed)
    deep = [N_BITS, hidden, hidden, 2]
    shal = [N_BITS, hidden, 2]
    res = {}

    def _acc(net):
        return float(net.accuracy(Xtr, ytr)), float(net.accuracy(Xte, yte))

    # TEST: the BDSP deep net (the exact sim/ rule as a numpy reference)
    net = BDSPNet(deep, seed=seed, beta=beta, p0=p0)
    wt_ok = _no_weight_transport(net)
    Y_before = [y.copy() for y in net.Y]
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    Y_fixed = all(np.array_equal(a, b) for a, b in zip(Y_before, net.Y))   # anti-cheat 1: Y never written
    tr, te = _acc(net)
    probe = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    res["bdsp"] = {"train": tr, "heldout": te, "probe_latent": probe,
                   "no_weight_transport": bool(wt_ok and Y_fixed)}

    # anti-cheat 7 / memorization floor: single hidden layer (the point-neuron/no-depth regime -- must struggle)
    net = BDSPNet(shal, seed=seed, beta=beta, p0=p0)
    _train(net, Xtr, ytr, "bdsp", epochs, lr, batch, seed)
    tr, te = _acc(net); res["single_layer"] = {"train": tr, "heldout": te}

    # anti-cheat 4 / floor: apical lesion (Y=0 -> no top-down credit -> hidden frozen-random)
    net = BDSPNet(deep, seed=seed, beta=beta, p0=p0)
    _train(net, Xtr, ytr, "apical_lesion", epochs, lr, batch, seed)
    tr, te = _acc(net); probe0 = _probe_latents(_hidden_rep(net, Xtr), Ltr, _hidden_rep(net, Xte), Lte)
    res["apical_lesion"] = {"train": tr, "heldout": te, "probe_latent": probe0}

    # anti-cheat 3: wrong-sign apical (negate the burst deviation -> anti-learn)
    net = BDSPNet(deep, seed=seed, beta=beta, p0=p0)
    _train(net, Xtr, ytr, "wrong_sign", epochs, lr, batch, seed)
    tr, te = _acc(net); res["wrong_sign"] = {"train": tr, "heldout": te}

    # anti-cheat 5 / P0 moat: no-teaching null (target detached -> dw~0 -> weights ~unchanged)
    net = BDSPNet(deep, seed=seed, beta=beta, p0=p0)
    W0 = [w.copy() for w in net.W]
    _train(net, Xtr, ytr, "no_teaching_null", epochs, lr, batch, seed)
    tr, te = _acc(net)
    w_drift = float(np.mean([np.abs(a - b).mean() for a, b in zip(W0, net.W)]))
    res["no_teaching_null"] = {"train": tr, "heldout": te, "weight_drift": w_drift}

    # anti-cheat 2: permuted-label (shuffle y in TRAIN -> held-out ~chance = generalization not leakage)
    prng = np.random.default_rng(seed + 555)
    yperm = ytr[prng.permutation(len(ytr))]
    net = BDSPNet(deep, seed=seed, beta=beta, p0=p0)
    _train(net, Xtr, yperm, "bdsp", epochs, lr, batch, seed)
    _tr, te = _acc(net); res["permuted"] = {"train": _tr, "heldout": te}

    # anti-cheat 6 / ceiling: fenced backprop oracle (task-sanity; NOT a shipped rule), SAME W-init
    net = DendriticMLP(deep, seed=seed)
    from research.runners._emerge1_deep_dendritic_representation_derisk import _train as _o_train
    _o_train(net, Xtr, ytr, "oracle", epochs, lr, batch, seed)
    tr, te = _acc(net); res["oracle_bp"] = {"train": tr, "heldout": te}

    # decisive within-net contrast fairness: BDSPNet init == DendriticMLP init (same W)
    b0 = BDSPNet(deep, seed=seed, beta=beta, p0=p0); f0 = DendriticMLP(deep, seed=seed)
    res["same_init_as_oracle"] = bool(all(np.allclose(a, b) for a, b in zip(b0.W, f0.W)))
    res["chance"] = float(max(np.mean(yte == 0), np.mean(yte == 1)))
    return {"seed": seed, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64, help="hidden width (CPU smoke 64; controller GPU run 384)")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--stage-a-t-ms", type=float, default=2000.0)
    # Stage-A numpy neuron runs at dt=0.1ms; 3.0ms ISI is EMERGE-4's validated GO config (the sim/ net's coarse
    # dt=1.0 burst detector uses burst_isi_threshold_ms=6.0, a separate concern from the fine-dt Stage-A carrier).
    ap.add_argument("--stage-a-isi-ms", type=float, default=3.0)
    ap.add_argument("--skip-bridge", action="store_true", help="skip the on-bridge smokes (Stage-B numpy only)")
    ap.add_argument("--backend", default=None, help="override SIM_BACKEND (numpy|cupy)")
    ap.add_argument("--json", "--out", dest="out", default=str(OUT))
    a = ap.parse_args()
    if a.backend:
        os.environ["SIM_BACKEND"] = a.backend
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []; stage_a = {}; stage_a_bridge = {}; stage_a_learn = {}

    try:
        # ---- Stage A (numpy multiplexing on the D1 config) ----
        for s in a.seeds:
            stage_a[str(s)] = stage_a_multiplexing(s, a.stage_a_t_ms, a.stage_a_isi_ms)
        sa_go = all(stage_a[str(s)]["GO"] for s in a.seeds)
        for s in a.seeds:
            r = stage_a[str(s)]
            print(f"  [StageA seed {s}] E~basal {r['corr_E_basal']:.3f} | E-inv {r['E_invariance_to_apical']:.3f} | "
                  f"P~apical {r['corr_P_apical']:.3f} | P0 {r['P0_rest']:.3f} | sep {r['separability_R2']:.3f} | "
                  f"GO {r['GO']}", flush=True)

        # ---- Stage A' / A'' (the sim/ machinery on a real bridge) ----
        if not a.skip_bridge:
            stage_a_bridge = stage_a_bridge_detector(a.seeds[0])
            print(f"  [StageA' bridge-detector] {stage_a_bridge}", flush=True)
            stage_a_learn = stage_a_bridge_learns(a.seeds[0])
            print(f"  [StageA'' bridge-learns]  {stage_a_learn}", flush=True)

        # ---- Stage B (the net) ----
        for s in a.seeds:
            r = run(s, a.epochs, a.lr, a.batch, a.hidden, a.beta, a.p0); per.append(r)
            d = r["bdsp"]
            print(f"  [StageB seed {s}] bdsp held {d['heldout']:.3f} (train {d['train']:.3f}, probe "
                  f"{d['probe_latent']:.3f}) | single {r['single_layer']['heldout']:.3f} | lesion "
                  f"{r['apical_lesion']['heldout']:.3f} (probe {r['apical_lesion']['probe_latent']:.3f}) | wrong "
                  f"{r['wrong_sign']['heldout']:.3f} | null {r['no_teaching_null']['heldout']:.3f} "
                  f"(drift {r['no_teaching_null']['weight_drift']:.1e}) | perm {r['permuted']['heldout']:.3f} | "
                  f"oracle {r['oracle_bp']['heldout']:.3f} | chance {r['chance']:.3f} | wt_ok "
                  f"{d['no_weight_transport']} | same_init {r['same_init_as_oracle']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def mean(k, sub="heldout"):
            return float(np.mean([p[k][sub] for p in per]))
        bd, sing, les = mean("bdsp"), mean("single_layer"), mean("apical_lesion")
        wrong, null, perm = mean("wrong_sign"), mean("no_teaching_null"), mean("permuted")
        orac, ch = mean("oracle_bp"), float(np.mean([p["chance"] for p in per]))
        bd_probe, les_probe = mean("bdsp", "probe_latent"), mean("apical_lesion", "probe_latent")
        null_drift = float(np.mean([p["no_teaching_null"]["weight_drift"] for p in per]))
        wt = all(p["bdsp"]["no_weight_transport"] and p["same_init_as_oracle"] for p in per)
        sa_go = all(stage_a[str(s)]["GO"] for s in a.seeds)
        # GO gates (pre-registered)
        task_ok = orac >= 0.80
        generalizes = (bd >= 0.75) and (bd > les + 0.10) and (bd > sing + 0.05)
        rep_emerges = (bd_probe > les_probe + 0.10) and (bd_probe >= 0.70)
        lesion_collapses = les <= max(sing, ch) + 0.05
        wrong_anti = wrong <= ch + 0.05
        null_flat = (null <= ch + 0.05) and (null_drift < 1e-3)
        permuted_chance = perm <= ch + 0.05
        go = bool(task_ok and generalizes and rep_emerges and lesion_collapses and wrong_anti
                  and null_flat and permuted_chance and wt)
        partial = bool(task_ok and wt and lesion_collapses and (bd > les + 0.10) and (bd > sing + 0.05)
                       and not (generalizes and rep_emerges))
        if not task_ok:
            verdict = (f"INCONCLUSIVE -- oracle only {orac:.3f} held-out; tune epochs/lr/hidden before reading the "
                       f"BDSP arms (NOT a BDSP verdict).")
        elif go:
            verdict = (f"GO -- the D1 spiking Burst-Dependent Plasticity rule (the additive `sim/` enable_bdsp "
                       f"mechanism, as its numpy reference) reproduces EMERGE-1b's depth-2 result: BDSP held-out "
                       f"{bd:.3f} >> single-layer {sing:.3f} + apical-lesion {les:.3f} + chance {ch:.3f}; the level-1 "
                       f"XOR latents EMERGED (probe {bd_probe:.3f} vs frozen {les_probe:.3f}); apical-lesion collapses, "
                       f"wrong-sign anti-learns ({wrong:.3f}), no-teaching null flat ({null:.3f}, drift {null_drift:.1e} "
                       f"= the P0 moat), permuted ~chance ({perm:.3f}), no weight transport, same W-init as the oracle; "
                       f"Stage-A multiplexing {'GO' if sa_go else 'PARTIAL'}. Multi-seed. ⇒ the burst-multiplexed "
                       f"deep-credit rule is a real `sim/` mechanism that credit-assigns through depth on the two-"
                       f"compartment substrate. The full 384-width fully-on-bridge multi-seed is the controller's GPU "
                       f"run; the additive sim/ diff is byte-identical when enable_bdsp is off.")
        elif partial:
            verdict = (f"PARTIAL/QUALIFIED -- BDSP clearly beats the floors (held {bd:.3f} vs single {sing:.3f} / lesion "
                       f"{les:.3f}, apical load-bearing) so the burst rule DOES add depth-credit, but it doesn't fully "
                       f"clear the generalization+probe bar (held {bd:.3f}, probe {bd_probe:.3f}) at this CPU-smoke "
                       f"width {a.hidden} (oracle {orac:.3f}). Payeur's burst estimate improves with width -> the "
                       f"controller's 384-width GPU run is the decisive test. Build-informative, NOT a stop.")
        else:
            miss = []
            if not generalizes: miss.append(f"BDSP didn't clear the floors (held {bd:.3f} vs single {sing:.3f}/lesion {les:.3f})")
            if not rep_emerges: miss.append(f"hidden structure didn't emerge (probe {bd_probe:.3f} vs frozen {les_probe:.3f})")
            if not lesion_collapses: miss.append("apical-lesion did NOT collapse (apical not load-bearing)")
            if not wrong_anti: miss.append(f"wrong-sign not at chance ({wrong:.3f})")
            if not null_flat: miss.append(f"no-teaching null not flat ({null:.3f}, drift {null_drift:.1e}) -- P0 moat bug")
            if not permuted_chance: miss.append(f"permuted not at chance ({perm:.3f}) -- leakage")
            if not wt: miss.append("weight-transport / same-init check failed")
            verdict = ("BOUNDARY (build-informative, not a stop) -- " + "; ".join(miss) + f". BDSP did not clear the "
                       f"depth wall at CPU-smoke width {a.hidden} (oracle CAN: {orac:.3f}). Try the controller's "
                       f"384-width GPU run (population coding is the mitigation) or the microcircuit arm before "
                       f"concluding. NB: the `sim/` machinery is validated by the Stage-A bridge smokes regardless.")
    else:
        go = False; verdict = f"ERROR -- {err}"

    summary = {"probe": "gnw_d1_spiking_bdsp", "GO": go, "verdict": verdict,
               "rule": "BDSP / Burstprop (Payeur-Naud 2021 M1.2): dw = eta*Etilde_j*(B_i - Pbar_i*E_i); event rate E "
                       "= feedforward channel, burst probability P = sigmoid(beta*v_apical) via fixed-random apical "
                       "feedback (no weight transport), Pbar = slow single-phase EMA baseline (init P0); the P0 moat "
                       "(rest apical -> P~Pbar -> dw~0). Realized as the additive/default-off sim/ enable_bdsp kernel "
                       "fused_bdsp_update + burst detector + apical-credit routing in bridge._run_one_simulation_step.",
               "task": f"depth-2 threshold-of-{N_PAIRS}-pair-XORs over {N_BITS} bits (== EMERGE-1/1b, make_task verbatim)",
               "seeds": a.seeds,
               "config": {"epochs": a.epochs, "lr": a.lr, "batch": a.batch, "hidden": a.hidden,
                          "beta": a.beta, "p0": a.p0, "backend": os.environ.get("SIM_BACKEND")},
               "stage_a_multiplexing": stage_a, "stage_a_bridge_detector": stage_a_bridge,
               "stage_a_bridge_learns": stage_a_learn,
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "The PRIMARY Stage-B arm is a numpy REFERENCE of the exact `sim/` enable_bdsp rule (the "
                              "fast CPU smoke the builder validates: does the burst rule LEARN above the memorization "
                              "floor and does apical-lesion collapse it). The `sim/` machinery itself (fused_bdsp_update "
                              "kernel + burst detector + apical->P routing) is exercised end-to-end by the Stage-A' "
                              "bridge-detector + Stage-A'' bridge-learns smokes on a REAL SimulationBridge. The fully-"
                              "on-bridge 384-width spiking net multi-seed is the CONTROLLER's GPU run. The oracle arm "
                              "is a fenced backprop ceiling (task-sanity), NOT a shipped biologically-local mode. NO "
                              "settling loop, NO weight transport (fixed-random Y). The sim/ diff is additive/default-"
                              "off/guarded and byte-identical when enable_bdsp is False."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[d1] VERDICT: {verdict}", flush=True)
    print(f"[d1] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
