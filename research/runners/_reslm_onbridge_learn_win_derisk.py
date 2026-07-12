"""RES-LM ON-BRIDGE LEARN-W_in -- the SPIKING realization of the R3 long-range mechanism "learn the INPUT projection on a
FIXED reservoir" via the COMMITTED on-bridge BDSP / Burstprop rule (`enable_bdsp`). NO new `sim/` mechanism; NO `sim/` edit.

THE PIVOT (why this is the RIGHT build). The existing on-bridge BDSP reservoir runner
(`_emerge_reservoir_lm_onbridge_bdsp_derisk.py`) makes the reservoir's RECURRENT synapses (W_rec) plastic -- the bottleneck
the R3 reframe PROVED is wrong (training W_rec is counterproductive; that on-bridge W_rec run boundaried at 2/6
seed-variable). The rate-level result (`_emerge_stream_eprop_lm_derisk.py::train_eprop_learn_win`) is that with W_rec FIXED
(a random reservoir) and the INPUT projection W_in learned by one-step-LOCAL random-feedback e-prop + a local read-out, the
biological rule recovers ~78% of the frozen-reservoir->full-BPTT long-range gap -- a LARGE + STABLE margin. This runner
PIVOTS the on-bridge port to learn **W_in** on a FROZEN spiking reservoir, so the committed BDSP rule is applied to the RIGHT
bottleneck.

THE LOAD-BEARING MAPPING (verified against `sim/bridge.py:7246-7273`). The committed BDSP update is
    dw = eta * Etilde_pre * (B_post - Pbar_post * E_post)              per plastic synapse over cp_connections
gathered from the cached COO (coo.row = presynaptic, coo.col = postsynaptic), gated by the per-synapse plastic mask + the
plasticity gain. For a PLASTIC `input -> reservoir` pathway (the ONLY plastic pathway here; the reservoir recurrence is
`plastic_internal=False` = FROZEN):
    Etilde_pre = cp_bdsp_E[coo.row]          = the INPUT-token event rate      = the input eligibility e_in[j,v]
    (B - Pbar*E)[coo.col], driven by cp_bdsp_apical_drive[reservoir_j] = k*(Y@delta)_j  = the reservoir broadcast credit L_j
  => dw[input_v -> res_j] = eta * E[input_v] * L_j == exactly the rate rule W_in[j,v] += lr * L_j * e_in[j,v], ON SPIKES,
     with NO weight transport (Y is a fixed-random host matrix from its OWN RandomState; the host computing Y@delta and
     setting the apical is the legitimate fixed-random credit-projection wiring -- as EVERY D1/EMERGE reference does; the
     WEIGHT CHANGE is the committed kernel's = the brain's job). W_in lives in cp_connections.data -> BDSP moves it. The
     reservoir recurrence (also in cp_connections, but with a False plastic mask) is UNTOUCHED -> BDSP moves W_in, not W_rec
     (confirmed by the per-pathway dw diagnostics below).

THE TASK = K-cue distal-cue delayed-decode. Each sentence = [CUE_k] . FILLER x dist . [QUERY]; the read-out at the QUERY
position decodes which cue k was seen `dist` fillers ago. K is large enough that a fixed-random W_in COLLIDES the K cue codes
in the mixed reservoir state (they are not linearly decodable after the recurrence scrambles them), so ONLY a LEARNED W_in --
one that maps the cue into a slow-fading, separable subspace of the fixed reservoir -- makes the delayed cue DECODABLE. dist
is within the spiking reservoir's fading-memory depth (EMERGE-81). This isolates the R3 claim: the fixed recurrence HOLDS the
cue; only a learned input embedding makes it decodable after mixing.

THE TWO-PASS PER-SENTENCE CYCLE (mirrors the forked runner):
  PASS A (read; learning OFF, apical OFF): wash -> forward the sentence -> read the reservoir pool spike-count r_query at the
      QUERY position -> clean read-out error delta = onehot(k) - softmax(Wout@[r_query,1]); train Wout online (Wout is only
      the credit vehicle; the METRIC re-fits a fresh clean ridge on the FROZEN reservoir so the comparison isolates W_in).
  PASS B (teach; learning ON): wash -> forward the SAME sentence with the constant apical credit cp_bdsp_apical_drive[res_j]
      = k*(Y@delta)_j held across every token, so the committed BDSP kernel moves the input->reservoir W_in. The credit is
      terminal (one delta per sentence); the per-step E_pre eligibility selects WHICH input synapse it lands on (it lands on
      the CUE->reservoir synapses when the cue is presented + its input event-rate is live) -- the faithful spiking
      realization of the rate rule's "accumulated input eligibility x terminal broadcast credit".

ARMS (ONE variable = fixed-random W_in vs BDSP-learned W_in; reservoir / read-out procedure / task / seeds IDENTICAL):
  * fixed_win     : W_in FROZEN at its random init (no teach pass) -- the fixed-reservoir control / the collision floor.
  * learn_win     : the piece -- BDSP-learned W_in (directed apical credit).
  * apical_lesion : BDSP on but apical=0 in teach (ANTI-CHEAT) -- W_in moves only by the undirected -Pbar*E drift => must
                    land AT fixed_win.
  * wrong_sign    : apical = -k*(Y@delta) (ANTI-CHEAT) -- must ANTI-learn (below fixed_win / chance).
  * rate_reference: a small numpy leaky-tanh reservoir + FIXED W_rec + W_in learned by the SAME input-synapse e-prop rule
                    (ported from train_eprop_learn_win) on the SAME task = the rate CEILING + the spiking-vs-rate headroom.

ANTI-CHEATS: (1) no-weight-transport (Y from its OWN RandomState, asserted never a forward weight); (2) input-lesion -> the
decode collapses to chance (the read is from the reservoir's REAL cue-driven spikes, not a bias); (3) distal-cue scramble
(shuffle the fit labels) -> chance (the decode needs the true cue<->read structure); (4) B_rises (the apical raises the
MEASURED reservoir burst rate B -- reuse the D1 apical-coupling diagnostic; else there is no directed credit -- the gate the
W_rec runs surfaced).

METRIC + GATE (report only; the CONTROLLER runs the 6-seed). Distal-cue decode accuracy per arm (clean ridge read-out fit on
the FROZEN reservoir + the arm's final W_in). GO if learn_win - fixed_win >= +0.10 AND apical_lesion ~ fixed_win AND
wrong_sign anti-learns AND scramble at chance AND input-lesion at chance AND B_rises AND no weight transport. BOUNDARY (name
it, don't force) if the machinery is clean but the margin is seed-variable.

SCOPE / COST. numpy-CPU (the on-bridge step loop is slow). The defaults are a cheap 1-seed WIRING smoke (small n_pool / K /
sentences / epochs) -- the numbers WILL be noisy at smoke scale; the smoke checks the WIRING + anti-cheats, NOT the GO. NO
`sim/` edit (reuse-by-import; the additive `sim/` BDSP + apical-soma diff is byte-identical when off).

Run (1-seed CPU smoke):
  SIM_BACKEND=numpy python -u -m research.runners._reslm_onbridge_learn_win_derisk \
      --seeds 42 --n-pool 120 --n-cues 12 --n-per-cue 4 --dist 3 --epochs 2 --smoke
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import: the EMERGE-82 fixed-reservoir constants + the EMERGE-61 wash-out (via EMERGE-82's re-export). NO sim/ edit,
# NO edit to any runner.
from research.runners._emerge82_onbridge_lsm_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, _INTERNAL_DENSITY, _EXC_W, _INH_W,
)

OUT = _REPO / "research" / "findings" / "raw" / "_reslm_onbridge_learn_win.json"

_T_STEP = 8               # bridge steps per input token
_DW_MOVE_MIN = 1e-5       # mean |dw| above this = the input->reservoir weights genuinely moved


def _softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------------------------------------------------
# The K-cue distal-cue delayed-decode task. Vocab = K cues + FILLER + QUERY. Sentence = [cue_k] FILLER*dist [QUERY]; the
# label = k (decoded at the QUERY position). Balanced n_per_cue examples per cue; split per-cue so every cue appears in eval.
# ---------------------------------------------------------------------------------------------------------------------
def build_task(seed, n_cues, n_per_cue, dist, dist_jitter=0):
    rng = np.random.default_rng(seed * 5227 + 11)
    V = n_cues + 2
    FILL = n_cues
    QRY = n_cues + 1
    train, evl = [], []
    for k in range(n_cues):
        for j in range(n_per_cue):
            d = int(dist + (rng.integers(-dist_jitter, dist_jitter + 1) if dist_jitter > 0 else 0))
            d = max(0, d)
            toks = [k] + [FILL] * d + [QRY]
            (evl if j == 0 else train).append((toks, k))          # 1 held-out example per cue, rest train
    rng.shuffle(train)
    return train, evl, V, FILL, QRY


# ---------------------------------------------------------------------------------------------------------------------
# The on-bridge net: a spiking INPUT region (one sub-pop per token) --[PLASTIC input->reservoir = W_in]--> a FIXED spiking
# reservoir (plastic_internal=False). The committed enable_bdsp rule (apical->soma coupling on) learns W_in. Copy of the
# EMERGE-82 fixed reservoir constants + the D1 plastic-input->region BDSP template; NO sim/ edit, NO edit to any runner.
# ---------------------------------------------------------------------------------------------------------------------
def _build_bridge(seed, n_pool, V, in_pop, fwd_wmean, fwd_wjit, fwd_density, soma_g,
                  bdsp_lr, bdsp_p0, bdsp_beta, w_min, w_max, dt=0.5):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    n_in = V * in_pop
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        # input: one sub-pop per token, no internal recurrence, not plastic-internal.
        BrainRegion(name="input", n_neurons=n_in, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        # reservoir: EMERGE-82 fixed reservoir (internal_density recurrence, plastic_internal=False = FROZEN W_rec).
        BrainRegion(name="reservoir", n_neurons=n_pool, exc_fraction=0.8, internal_density=_INTERNAL_DENSITY,
                    exc_weight_mean=_EXC_W, inh_weight_mean=_INH_W, weight_jitter=0.3, plastic_internal=False),
    ]
    cfg.region_pathways = [
        # W_in: the ONLY plastic pathway -> the ONLY thing BDSP moves. Weights live in cp_connections.data.
        RegionPathway(from_region="input", to_region="reservoir", density=float(fwd_density),
                      weight_mean=float(fwd_wmean), weight_jitter=float(fwd_wjit), plastic=True),
    ]
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # isolate the BDSP-driven dw: no OTHER plasticity/normalization moving weights in parallel.
    for _flag in ("enable_stdp", "enable_hebbian_learning", "enable_homeostasis", "enable_structural_plasticity",
                  "enable_reward_modulation", "enable_input_divisive_norm", "enable_short_term_plasticity",
                  "enable_nmda"):
        setattr(cfg, _flag, False)
    # THE COMMITTED RULE (additive/default-off sim/ mechanism; byte-identical when off).
    cfg.enable_bdsp = True
    cfg.enable_bdsp_microcircuit = False
    cfg.bdsp_learning_rate = float(bdsp_lr)
    cfg.bdsp_p0 = float(bdsp_p0)
    cfg.bdsp_beta = float(bdsp_beta)
    # WALL #1 FIX (D1 2026-07-10): route apical depolarization to the soma so apical^ -> more MEASURED bursts B^ ->
    # directed credit. Default g=0 = the documented decoupled boundary; the runner uses a swept value.
    cfg.bdsp_apical_couples_soma = True
    cfg.bdsp_apical_soma_g = float(soma_g)
    # bdsp_w_max gotcha (CLAUDE.md / D1): the committed default 5.0 would clip the forward W_in design weight -> silence.
    # Set the clip ABOVE the forward weight; w_min>=0 so conductance synapses never flip sign.
    cfg.bdsp_w_min = float(w_min)
    cfg.bdsp_w_max = float(w_max)
    rt = RuntimeState()
    rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, cfg


class WinLearnReservoir:
    """A spiking INPUT region --[plastic input->reservoir W_in]--> a FIXED spiking reservoir on ONE SimulationBridge, with
    W_in learned by the committed on-bridge BDSP rule (apical = k*(Y@delta), fixed-random Y, NO weight transport). The read
    feature = the reservoir per-neuron spike-count during the QUERY token."""

    def __init__(self, V, n_pool, in_pop, seed, soma_g, bdsp_lr, bdsp_p0, bdsp_beta, w_min, w_max,
                 in_hi, res_bias, k_apical, fwd_wmean, fwd_wjit, fwd_density, dt=0.5):
        self.V = int(V); self.n = int(n_pool); self.in_pop = int(in_pop)
        self.in_hi = float(in_hi); self.res_bias = float(res_bias); self.k_apical = float(k_apical)
        self.bridge, self.cfg = _build_bridge(seed, n_pool, V, in_pop, fwd_wmean, fwd_wjit, fwd_density, soma_g,
                                              bdsp_lr, bdsp_p0, bdsp_beta, w_min, w_max, dt=dt)
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self._num = int(self.bridge.core_config.num_neurons)
        self._eta = float(bdsp_lr)
        rm = self.bridge.region_manager
        self.in_idx = np.asarray(list(rm.indices("input")), dtype=int)
        self.res_idx = np.asarray(list(rm.indices("reservoir")), dtype=int)
        self.tok_idx = [self.in_idx[t * self.in_pop:(t + 1) * self.in_pop] for t in range(self.V)]
        # FIXED-RANDOM apical feedback Y (n_pool x K_cues... actually n_pool x V so it accepts a full-vocab delta), OWN
        # stream => NO weight transport. delta is over the K cue classes; we pad to V-wide feedback for generality but the
        # decode only spans the cue classes (see _run_arm). Y drawn (n_pool x n_classes).
        self._snap = _snapshot_state(self.bridge)
        self._build_masks()

    def set_n_classes(self, n_classes):
        self.n_classes = int(n_classes)
        self.Y = np.random.RandomState(self._seed_for_Y).normal(0.0, 1.0, (self.n, self.n_classes))
        self._Y0 = self.Y.copy()

    # ---- pathway bookkeeping (separate input->reservoir W_in from the frozen reservoir recurrence) ----------
    def _coo(self):
        from sim.backend import to_host
        coo = self.bridge._get_cached_coo()
        return (np.asarray(to_host(coo.row)).astype(int), np.asarray(to_host(coo.col)).astype(int))

    def _build_masks(self):
        row, col = self._coo()
        in_set = set(self.in_idx.tolist()); res_set = set(self.res_idx.tolist())
        r_in = np.array([r in in_set for r in row]); r_res = np.array([r in res_set for r in row])
        c_res = np.array([c in res_set for c in col])
        self.mask_win = r_in & c_res            # input -> reservoir (the plastic W_in)
        self.mask_rec = r_res & c_res           # reservoir -> reservoir (the FROZEN recurrence -- must NOT move)
        self._row, self._col = row, col

    def _weights(self):
        from sim.backend import to_host
        return np.asarray(to_host(self.bridge.cp_connections.data)).astype(np.float64)

    def win_vec(self):
        return self._weights()[self.mask_win].copy()

    def rec_vec(self):
        return self._weights()[self.mask_rec].copy()

    def no_weight_transport(self):
        """anti-cheat: Y is never written after init AND is not byte-equal to the forward W_in block (shape permitting)."""
        y_unchanged = bool(np.array_equal(self.Y, self._Y0))
        w = self._weights()
        win = w[self.mask_win]
        not_forward = not (self.Y.size == win.size and np.allclose(np.sort(self.Y.ravel()), np.sort(win)))
        return bool(y_unchanged and not_forward)

    def reset_bdsp_traces(self):
        for _a in ("cp_bdsp_E", "cp_bdsp_B", "cp_bdsp_P", "cp_bdsp_Pbar", "cp_bdsp_last_spike_step",
                   "cp_v_apical", "cp_bdsp_apical_drive", "cp_bdsp_int_drive"):
            if hasattr(self.bridge, _a):
                setattr(self.bridge, _a, None)
        self.bridge._bdsp_step_counter = 0

    # ---- dynamics -----------------------------------------------------------------------------------------
    def _set_apical(self, vec):
        """vec (len n on res_idx) or None -> apical OFF (zeros -> v_apical decays to rest -> zero soma coupling)."""
        from sim.backend import from_host
        ap = np.zeros(self._num, np.float32)
        if vec is not None:
            ap[self.res_idx] = np.asarray(vec, np.float32)
        self.bridge.cp_bdsp_apical_drive = from_host(ap)

    def _drive_for_token(self, tok_id, silence):
        cur = np.zeros(self._num, np.float32)
        if not silence:
            cur[self.tok_idx[tok_id]] = self.in_hi         # the active token's input sub-pop fires
        cur[self.res_idx] += self.res_bias                 # tonic reservoir bias (fluctuation-driven regime)
        return cur

    def forward(self, toks, learn, apical_vec, silence=False, do_read=True):
        """Wash -> present each token for _T_STEP steps (input sub-pop drive + reservoir bias); if learn, keep the constant
        apical credit set + bdsp_lr=eta so BDSP moves W_in; read the reservoir pool spike-count during the QUERY token."""
        from sim.backend import from_host, to_host
        b = self.bridge
        _restore_state(b, self._snap)
        self.cfg.bdsp_learning_rate = (self._eta if learn else 0.0)
        self._set_apical(apical_vec if learn else None)     # read pass: apical off (unperturbed read)
        q_counts = np.zeros(self.n, np.float64)
        for t, tok in enumerate(toks):
            drive = from_host(self._drive_for_token(tok, silence))
            is_query = (t == len(toks) - 1)
            for _ in range(_T_STEP):
                b.cp_external_input_current[:] = drive       # re-apply each step (robust vs any per-step reset)
                if learn:
                    self._set_apical(apical_vec)             # hold the constant terminal credit across the sentence
                b._run_one_simulation_step()
                if do_read and is_query:
                    q_counts += np.asarray(to_host(b.cp_firing_states)).astype(np.float64)[self.res_idx]
        b.cp_external_input_current[:] = 0.0
        return q_counts / _T_STEP

    def read_query(self, toks, silence=False):
        return self.forward(toks, learn=False, apical_vec=None, silence=silence, do_read=True)

    def mean_res_spikes(self, toks):
        r = self.read_query(toks, silence=False)
        return float(r.mean())

    def mean_B(self):
        from sim.backend import to_host
        if getattr(self.bridge, "cp_bdsp_B", None) is None:
            return 0.0
        return float(np.asarray(to_host(self.bridge.cp_bdsp_B)).astype(np.float64)[self.res_idx].mean())

    def freeze(self):
        self.cfg.bdsp_learning_rate = 0.0
        self._set_apical(None)

    # ---- the two-pass online BDSP training (learn W_in) ---------------------------------------------------
    def train_arm(self, sents, mode, lr_out, epochs, rng):
        """mode in {fixed_win, learn_win, apical_lesion, wrong_sign}. PASS A: clean read at the QUERY + Wout delta rule +
        collect the terminal delta. PASS B (skip for fixed_win): re-forward with the constant apical credit k*(Y@delta) so
        the committed BDSP kernel moves the input->reservoir W_in. Returns Wout (the credit vehicle) + mean teach burst B."""
        Wout = np.zeros((self.n_classes, self.n + 1))
        order = list(range(len(sents)))
        lesion = (mode == "apical_lesion")
        sign = -1.0 if mode == "wrong_sign" else 1.0
        b_acc, b_n = 0.0, 0
        for _ep in range(epochs):
            rng.shuffle(order)
            for si in order:
                toks, k = sents[si]
                # PASS A -- clean read at the QUERY + Wout online + the terminal delta.
                r = self.read_query(toks)
                x = np.concatenate([r, [1.0]])
                p = _softmax(Wout @ x)
                delta = -p; delta[k] += 1.0
                Wout += lr_out * np.outer(delta, x)
                if mode == "fixed_win":
                    continue
                # PASS B -- credited teach: the constant terminal credit; the per-step E_pre eligibility selects the synapse.
                credit = None if lesion else (sign * self.k_apical * (self.Y @ delta))
                self.forward(toks, learn=True, apical_vec=credit, silence=False, do_read=False)
                b_acc += self.mean_B(); b_n += 1
        self.freeze()
        return Wout, (b_acc / max(1, b_n))

    # ---- B_rises diagnostic (D1 apical-coupling detector, on the reservoir) --------------------------------
    def apical_coupling_diag(self, probe_toks, apical_pA=None, steps=180):
        """Drive the reservoir (a cue + bias) so its somata spike, then measure the mean reservoir burst rate B at apical=0
        vs apical=+probe. The committed BDSP credit REQUIRES the apical to raise the MEASURED B (apical -> more real bursts
        -> dev=B-Pbar*E up -> directed credit). If B does not rise, there is no directed credit (the boundary the W_rec runs
        surfaced)."""
        from sim.backend import from_host, to_host
        b = self.bridge
        ap_pA = float(apical_pA if apical_pA is not None else self.k_apical)

        def _phase(apical_on):
            _restore_state(b, self._snap)
            self.reset_bdsp_traces()
            self.cfg.bdsp_learning_rate = 0.0
            drive = from_host(self._drive_for_token(probe_toks[0], silence=False))
            ap = np.zeros(self._num, np.float32)
            if apical_on:
                ap[self.res_idx] = ap_pA
            for _ in range(steps):
                b.cp_external_input_current[:] = drive
                b.cp_bdsp_apical_drive = from_host(ap)
                b._run_one_simulation_step()
            return float(np.asarray(to_host(b.cp_bdsp_B)).astype(np.float64)[self.res_idx].mean())

        B0 = _phase(False)
        B1 = _phase(True)
        _restore_state(b, self._snap)
        self.reset_bdsp_traces()
        self.freeze()
        return {"B_rest": B0, "B_apical": B1, "B_rises": bool(B1 > B0 + 1e-4)}


# ---------------------------------------------------------------------------------------------------------------------
# Clean ridge read-out (the metric): fit on the FROZEN reservoir + the arm's final W_in, decode = argmax.
# ---------------------------------------------------------------------------------------------------------------------
def _fit_ridge(R, y, n_classes, lam=1.0):
    Yoh = np.zeros((len(y), n_classes))
    Yoh[np.arange(len(y)), y] = 1.0
    A = R.T @ R + lam * np.eye(R.shape[1])
    return np.linalg.solve(A, R.T @ Yoh)          # (n+1, K)


def _decode_acc(R, y, W):
    return float(np.mean(np.argmax(R @ W, axis=1) == y))


def _collect_reads(res, sents, silence=False):
    R, Y = [], []
    for toks, k in sents:
        r = res.read_query(toks, silence=silence)
        R.append(np.concatenate([r, [1.0]])); Y.append(k)
    return np.asarray(R), np.asarray(Y)


# ---------------------------------------------------------------------------------------------------------------------
# The RATE reference: numpy leaky-tanh reservoir + FIXED W_rec + W_in learned by the SAME input-synapse e-prop rule (ported
# from train_eprop_learn_win), on the SAME K-cue task. The rate ceiling + spiking-vs-rate headroom.
# ---------------------------------------------------------------------------------------------------------------------
def rate_reference(train, evl, V, n, n_classes, seed, dist, epochs, lr_out, lr_in, alpha=0.3, learn=True):
    rng = np.random.RandomState(seed * 3 + 17)
    W_rec = rng.normal(0, 1, (n, n)) / np.sqrt(n)
    sr = max(np.abs(np.linalg.eigvals(W_rec)))
    W_rec = W_rec * (0.95 / max(sr, 1e-6))         # spectral radius ~0.95 = near-critical fading memory
    W_in = rng.normal(0, 1, (n, V)) / np.sqrt(V)
    Bfb = rng.normal(0, 1, (n, n_classes))         # fixed-random feedback (own draw; no weight transport)
    b = np.zeros(n)
    Wout = np.zeros((n_classes, n))

    def _fwd(toks, collect_elig):
        h = np.zeros(n); e_in = np.zeros((n, V)) if collect_elig else None
        hq = None
        for t, tok in enumerate(toks):
            x = np.zeros(V); x[tok] = 1.0
            pre = W_rec @ h + W_in @ x + b
            act = np.tanh(pre)
            h = (1 - alpha) * h + alpha * act
            if collect_elig:
                psi = alpha * (1 - act * act)
                e_in = (1 - alpha) * e_in + np.outer(psi, x)     # input-synapse e-prop eligibility (col v=tok)
            if t == len(toks) - 1:
                hq = h.copy()
        return hq, e_in

    if learn:
        for _ep in range(epochs):
            order = list(range(len(train))); rng.shuffle(order)
            for si in order:
                toks, k = train[si]
                hq, e_in = _fwd(toks, collect_elig=True)
                p = _softmax(Wout @ hq); delta = -p; delta[k] += 1.0
                Wout += lr_out * np.outer(delta, hq)
                L = Bfb @ delta                                  # broadcast random-feedback learning signal (n,)
                W_in = W_in + lr_in * (L[:, None] * e_in)        # W_in[j,v] += lr * L_j * e_in[j,v]

    # metric: freeze, clean ridge on the query reads.
    def _reads(sents):
        R, Y = [], []
        for toks, k in sents:
            hq, _ = _fwd(toks, collect_elig=False)
            R.append(np.concatenate([hq, [1.0]])); Y.append(k)
        return np.asarray(R), np.asarray(Y)

    Rtr, Ytr = _reads(train); Rev, Yev = _reads(evl)
    W = _fit_ridge(Rtr, Ytr, n_classes, lam=1.0)
    return _decode_acc(Rev, Yev, W)


# ---------------------------------------------------------------------------------------------------------------------
def _run_arm(res, mode, train, evl, n_classes, args, rng):
    """Reset W_in to its shared random init, train the arm (BDSP moves W_in for the credited arms), FREEZE, fit a clean ridge
    on the frozen reservoir, decode accuracy. For learn_win also run the input-lesion + scramble collapse controls."""
    from sim.backend import from_host
    res.bridge.cp_connections.data[:] = from_host(res._w_init.astype(np.float32))    # shared W_in init across arms
    res.reset_bdsp_traces()
    win_before = res.win_vec(); rec_before = res.rec_vec()
    _, teach_B = res.train_arm(train, mode, args.lr_out, args.epochs, rng)
    win_after = res.win_vec(); rec_after = res.rec_vec()
    dw_win = float(np.abs(win_after - win_before).mean())
    dw_rec = float(np.abs(rec_after - rec_before).mean())

    res.freeze()
    Rtr, Ytr = _collect_reads(res, train)
    Rev, Yev = _collect_reads(res, evl)
    W = _fit_ridge(Rtr, Ytr, n_classes, lam=args.ridge_lam)
    acc = _decode_acc(Rev, Yev, W)

    out = {"mode": mode, "decode_acc": round(acc, 4), "dw_win": dw_win, "dw_rec": dw_rec,
           "teach_B": round(float(teach_B), 4),
           "win_before_mean": round(float(win_before.mean()), 4), "win_after_mean": round(float(win_after.mean()), 4)}
    if mode == "learn_win":
        # (2) input-lesion: eval reads with the input drive silenced -> reservoir carries no cue -> chance.
        Rev_s, Yev_s = _collect_reads(res, evl, silence=True)
        out["input_lesion_acc"] = round(_decode_acc(Rev_s, Yev_s, W), 4)
        # (3) distal-cue scramble: fit the ridge on SHUFFLED labels -> the cue<->read structure is broken -> chance.
        y_scr = Ytr.copy(); np.random.default_rng(args.seed_base * 71 + 3).shuffle(y_scr)
        W_scr = _fit_ridge(Rtr, y_scr, n_classes, lam=args.ridge_lam)
        out["scramble_acc"] = round(_decode_acc(Rev, Yev, W_scr), 4)
    return out


def _derisk_one(seed, args):
    train, evl, V, FILL, QRY = build_task(seed, args.n_cues, args.n_per_cue, args.dist, args.dist_jitter)
    n_classes = args.n_cues
    chance = 1.0 / n_classes

    res = WinLearnReservoir(V, args.n_pool, args.in_pop, seed, args.soma_g, args.bdsp_lr, args.bdsp_p0,
                            args.bdsp_beta, args.bdsp_w_min, args.bdsp_w_max, args.in_hi, args.res_bias,
                            args.k_apical, args.fwd_wmean, args.fwd_wjit, args.fwd_density)
    res._seed_for_Y = seed + 9973
    res.set_n_classes(n_classes)
    res._w_init = res._weights().copy()                          # the SHARED W_in random init (all arms start here)
    args.seed_base = seed

    mean_spk = res.mean_res_spikes(train[0][0])                  # activity sanity

    # B_rises diagnostic (the D1 apical-coupling detector on the reservoir).
    coupling = res.apical_coupling_diag(train[0][0])

    arms = {}
    _salt = {"fixed_win": 1, "learn_win": 2, "apical_lesion": 3, "wrong_sign": 4}
    for mode in args.arms:
        arms[mode] = _run_arm(res, mode, train, evl, n_classes,
                              args, np.random.default_rng(seed * 211 + _salt.get(mode, 9) * 101))

    nwt = res.no_weight_transport()

    # rate reference (numpy) on the SAME task -- the ceiling + headroom.
    rate_learn = rate_reference(train, evl, V, args.rate_n, n_classes, seed, args.dist, args.rate_epochs,
                                args.rate_lr_out, args.rate_lr_in, learn=True)
    rate_fixed = rate_reference(train, evl, V, args.rate_n, n_classes, seed, args.dist, args.rate_epochs,
                                args.rate_lr_out, args.rate_lr_in, learn=False)

    return {"seed": seed, "V": V, "n_cues": args.n_cues, "n_pool": res.n, "dist": args.dist,
            "n_train": len(train), "n_eval": len(evl), "chance": round(chance, 4),
            "mean_res_spikes": round(mean_spk, 4), "apical_coupling": coupling, "arms": arms,
            "no_weight_transport": bool(nwt),
            "rate_reference": {"learn_win": round(rate_learn, 4), "fixed_win": round(rate_fixed, 4)}}


def _print_seed(d):
    print(f"  [seed {d['seed']}] V={d['V']} n_cues={d['n_cues']} n_pool={d['n_pool']} dist={d['dist']} "
          f"n_tr={d['n_train']} n_ev={d['n_eval']} chance={d['chance']:.3f} | res spikes/step {d['mean_res_spikes']:.3f}",
          flush=True)
    c = d["apical_coupling"]
    print(f"    B_rises diagnostic: B_rest {c['B_rest']:.4f} -> B_apical {c['B_apical']:.4f} (B_rises {c['B_rises']})",
          flush=True)
    for mode, a in d["arms"].items():
        extra = ""
        if mode == "learn_win":
            extra = f" | input_lesion {a['input_lesion_acc']:.3f} | scramble {a['scramble_acc']:.3f}"
        print(f"    arm {mode:>13}: decode {a['decode_acc']:.3f} | dw_win {a['dw_win']:.6f} dw_rec {a['dw_rec']:.6f} "
              f"| teach_B {a['teach_B']:.4f}{extra}", flush=True)
    print(f"    no_weight_transport {d['no_weight_transport']} | RATE ref learn {d['rate_reference']['learn_win']:.3f} "
          f"vs fixed {d['rate_reference']['fixed_win']:.3f}", flush=True)


def _derisk(seeds, args):
    print(f"RES-LM ON-BRIDGE LEARN-W_in: does a spiking reservoir's INPUT projection W_in LEARN via the committed enable_bdsp "
          f"rule (apical=k*(Y@delta), fixed-random Y), fixed-vs-learned W_in on a K-cue distal-cue delayed-decode task; "
          f"{len(seeds)}-seed; arms={args.arms}", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _derisk_one(s, args)
            per.append(d)
            _print_seed(d)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "reslm_onbridge_learn_win", "seeds": list(seeds),
               "elapsed_seconds": round(time.time() - t0, 1), "params": vars(args), "per_seed": per, "error": err}

    if err is None and per:
        def marm(mode, key):
            vals = [p["arms"][mode][key] for p in per if mode in p["arms"]]
            return float(np.mean(vals)) if vals else None
        chance = float(np.mean([p["chance"] for p in per]))
        fixed = marm("fixed_win", "decode_acc"); learn = marm("learn_win", "decode_acc")
        lesion = marm("apical_lesion", "decode_acc"); wrong = marm("wrong_sign", "decode_acc")
        dw_win = marm("learn_win", "dw_win"); dw_rec = marm("learn_win", "dw_rec")
        inlesion = marm("learn_win", "input_lesion_acc"); scramble = marm("learn_win", "scramble_acc")
        b_rises = all(p["apical_coupling"]["B_rises"] for p in per)
        nwt = all(p["no_weight_transport"] for p in per)
        rate_learn = float(np.mean([p["rate_reference"]["learn_win"] for p in per]))
        rate_fixed = float(np.mean([p["rate_reference"]["fixed_win"] for p in per]))

        margin = (learn - fixed) if (learn is not None and fixed is not None) else None
        win_moved = bool(dw_win is not None and dw_win > _DW_MOVE_MIN)
        rec_frozen = bool(dw_rec is not None and dw_rec < _DW_MOVE_MIN)     # the reservoir recurrence must NOT move
        lesion_at_fixed = bool(lesion is not None and fixed is not None and abs(lesion - fixed) <= args.gate_eps)
        wrong_antilearns = bool(wrong is not None and fixed is not None and wrong <= fixed + 1e-6)
        scramble_chance = bool(scramble is not None and scramble <= chance + args.gate_eps)
        inlesion_chance = bool(inlesion is not None and inlesion <= chance + args.gate_eps)

        # the GO gate (the CONTROLLER runs the 6-seed; this reports the aggregate over the given seeds).
        go = bool(margin is not None and margin >= 0.10 and lesion_at_fixed and wrong_antilearns
                  and scramble_chance and inlesion_chance and b_rises and nwt and win_moved and rec_frozen)
        summary["aggregate"] = {
            "chance": chance, "fixed_win_acc": fixed, "learn_win_acc": learn, "learn_minus_fixed": margin,
            "apical_lesion_acc": lesion, "wrong_sign_acc": wrong, "input_lesion_acc": inlesion, "scramble_acc": scramble,
            "learn_win_dw_win": dw_win, "learn_win_dw_rec": dw_rec, "win_moved": win_moved, "rec_frozen": rec_frozen,
            "B_rises": b_rises, "no_weight_transport": nwt,
            "rate_reference_learn": rate_learn, "rate_reference_fixed": rate_fixed,
            "lesion_at_fixed": lesion_at_fixed, "wrong_antilearns": wrong_antilearns,
            "scramble_at_chance": scramble_chance, "input_lesion_at_chance": inlesion_chance,
        }
        def _f(v):
            return f"{v:.3f}" if v is not None else "n/a"
        if args.smoke:
            summary["verdict"] = (
                f"SMOKE (WIRING) -- end-to-end on a real SimulationBridge; the input->reservoir W_in "
                f"{'MOVED' if win_moved else 'did NOT move'} under the committed BDSP rule (dw_win {_f(dw_win)}) while the "
                f"reservoir recurrence stayed {'FROZEN' if rec_frozen else 'NOT frozen'} (dw_rec {_f(dw_rec)}) -> BDSP moves "
                f"W_in, not W_rec. B_rises {b_rises}; no-weight-transport {nwt}. Decode (NOISY at smoke scale): learn_win "
                f"{_f(learn)} vs fixed_win {_f(fixed)} (margin {_f(margin)}), apical_lesion {_f(lesion)}, wrong_sign "
                f"{_f(wrong)}, chance {_f(chance)}. Collapse controls: input-lesion {_f(inlesion)}, scramble {_f(scramble)} "
                f"(both -> ~chance if wiring is honest). RATE ref learn {_f(rate_learn)} vs fixed {_f(rate_fixed)}. The GO "
                f"gate + margin are the CONTROLLER's 6-seed sweep (this smoke checks the WIRING + anti-cheats, not the GO).")
        elif go:
            summary["verdict"] = (
                f"GO -- the committed enable_bdsp rule LEARNS the reservoir's INPUT projection W_in ON SPIKES: distal-cue "
                f"decode learn_win {learn:.3f} >= fixed_win {fixed:.3f} + 0.10 (margin {margin:+.3f}); apical_lesion "
                f"{lesion:.3f} ~ fixed_win (undirected drift only); wrong_sign {wrong:.3f} anti-learns; input-lesion "
                f"{inlesion:.3f} + scramble {scramble:.3f} at chance ({chance:.3f}); B_rises; no weight transport; W_in moved "
                f"(dw_win {dw_win:.6f}) while W_rec stayed frozen (dw_rec {dw_rec:.6f}). RATE ceiling learn {rate_learn:.3f} "
                f"vs fixed {rate_fixed:.3f}. The R3 long-range mechanism realizes on the committed BDSP rule, NO sim/ edit.")
        else:
            miss = []
            if not (margin is not None and margin >= 0.10): miss.append(f"learn-minus-fixed margin {margin} < 0.10")
            if not lesion_at_fixed: miss.append(f"apical_lesion {lesion} not ~ fixed {fixed}")
            if not wrong_antilearns: miss.append(f"wrong_sign {wrong} did not anti-learn (vs fixed {fixed})")
            if not scramble_chance: miss.append(f"scramble {scramble} not at chance {chance}")
            if not inlesion_chance: miss.append(f"input-lesion {inlesion} not at chance {chance}")
            if not b_rises: miss.append("B_rises False (apical decoupled from reservoir bursts)")
            if not nwt: miss.append("no-weight-transport FAILED")
            if not win_moved: miss.append(f"W_in did not move (dw_win {dw_win})")
            if not rec_frozen: miss.append(f"W_rec moved (dw_rec {dw_rec}) -- the recurrence should be frozen")
            summary["verdict"] = ("BOUNDARY -- " + "; ".join(miss) + f". Machinery ran; margin/controls not cleared at these "
                                  f"seeds/scale. Name the residual (operating point / lr / epochs / dist vs depth) as the next "
                                  f"single-variable de-risk; the rate ceiling (learn {rate_learn:.3f} vs fixed {rate_fixed:.3f}) "
                                  f"bounds the headroom. Do NOT force a positive.")
        summary["go"] = bool(go) if not args.smoke else None
    else:
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds"
        summary["go"] = False

    out_path = Path(args.json) if args.json else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[reslm-learn-win] VERDICT: {summary['verdict']}", flush=True)
    print(f"[reslm-learn-win] wrote {out_path}\n" + "=" * 118, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-cues", type=int, default=12, help="K cue tokens (large enough that fixed-random W_in collides)")
    ap.add_argument("--n-per-cue", type=int, default=4, help="examples per cue (1 held out per cue for eval)")
    ap.add_argument("--dist", type=int, default=3, help="filler distance cue->query (within the reservoir's memory depth)")
    ap.add_argument("--dist-jitter", type=int, default=0)
    ap.add_argument("--n-pool", type=int, default=120)
    ap.add_argument("--in-pop", type=int, default=2, help="input neurons per token")
    ap.add_argument("--epochs", type=int, default=2, help="BDSP online training epochs")
    ap.add_argument("--lr-out", type=float, default=0.01, help="credit-vehicle read-out delta-rule lr (PASS A)")
    ap.add_argument("--ridge-lam", type=float, default=1.0, help="metric ridge regularizer")
    # BDSP knobs (the committed sim/ surface)
    ap.add_argument("--bdsp-lr", type=float, default=0.02, help="eta: BDSP W_in update rate")
    ap.add_argument("--bdsp-p0", type=float, default=0.30)
    ap.add_argument("--bdsp-beta", type=float, default=1.0)
    ap.add_argument("--bdsp-w-min", type=float, default=0.0, help=">=0 so conductance synapses never flip sign")
    ap.add_argument("--bdsp-w-max", type=float, default=160.0, help="above the forward W_in design weight")
    ap.add_argument("--soma-g", type=float, default=120.0, help="bdsp_apical_soma_g (WALL#1 apical->soma coupling)")
    ap.add_argument("--k-apical", type=float, default=150.0, help="apical credit gain k in apical=k*(Y@delta)")
    # forward drive / init (this operating point makes the reservoir cue-active on numpy-CPU; controller re-tunes at scale)
    ap.add_argument("--in-hi", type=float, default=650.0, help="active input sub-pop drive (pA, suprathreshold)")
    ap.add_argument("--res-bias", type=float, default=55.0, help="tonic reservoir bias (fluctuation-driven regime)")
    ap.add_argument("--fwd-wmean", type=float, default=32.0, help="input->reservoir W_in mean init weight")
    ap.add_argument("--fwd-wjit", type=float, default=0.5)
    ap.add_argument("--fwd-density", type=float, default=1.0)
    # rate reference
    ap.add_argument("--rate-n", type=int, default=120)
    ap.add_argument("--rate-epochs", type=int, default=8)
    ap.add_argument("--rate-lr-out", type=float, default=0.02)
    ap.add_argument("--rate-lr-in", type=float, default=0.02)
    # arms / gate
    ap.add_argument("--arms", type=str, nargs="+",
                    default=["fixed_win", "learn_win", "apical_lesion", "wrong_sign"],
                    help="subset of {fixed_win, learn_win, apical_lesion, wrong_sign}")
    ap.add_argument("--gate-eps", type=float, default=0.06, help="tolerance for lesion~fixed and controls~chance")
    ap.add_argument("--smoke", action="store_true", help="cheap 1-seed WIRING smoke (defaults already small)")
    ap.add_argument("--json", "--out", dest="json", type=str, default=str(OUT))
    a = ap.parse_args()
    return _derisk(a.seeds, a)


if __name__ == "__main__":
    raise SystemExit(main())
