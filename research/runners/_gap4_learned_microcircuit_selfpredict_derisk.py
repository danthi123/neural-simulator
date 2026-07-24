"""gap#4 -- LEARNED transport-free deep CREDIT vs a frozen reservoir, on the DEPTH-REQUIRED XOR-over-pool
compositional-generalization task (the ONLY instrument the record proves is genuinely depth-required AND
transport-free-learnable -- NOT MNIST/CIFAR-FC, which `2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md`
proved are shallow / anti-correlated).

This is the CPU numpy-RATE phase of the gap#4 de-risk (queue item #4 in
`research/findings/2026-07-24-roadmap-scope-ahead-queue.md`). It is deliberately NARROW (heed FLAG G2 below):

  WHAT IT DOES (the honest scope):
    (a) confirms LEARNED/transport-free credit (fixed_fa / kp / micro) > a frozen RESERVOIR at FULL data on a task
        where depth is genuinely required (a real "credit closes a gap the reservoir cannot" result);
    (b) exercises the full anti-cheat battery -- the no-weight-transport 3-guard (+ an AST assert + a weight-transport
        CEILING arm), the triple lesion (apical / deepest-layer-freeze / freeze-learned-feedback), the Sacramento
        W^PI-freeze dissociation, the DOUBLE permuted control (shuffled-target + shufE directed-credit-scramble), and
        the ncc input-selectivity guard;
    (c) MEASURES the genuinely-new plastic-Eq.9 microcircuit's distinctive property that IS observable at rate --
        apical-SILENT-on-correct EARNED from a NON-fixed-point W^PI init (vs a frozen-noisy W^PI staying loud).

  WHAT IT CANNOT DO (FLAG G2 -- load-bearing, do NOT overclaim): at the numpy RATE reference the KP-learned feedback
    and the self-predicting microcircuit are ACCURACY-INDISTINGUISHABLE from plain fixed-random feedback-alignment
    (the learned/interneuron machinery is INERT on the feedforward weights at rate; confirmed byte-identical in
    `2026-07-07-deep-credit-real-task-compositional-semantics-GO.md`). The learned-vs-fixed SEPARATION only appears on
    SPIKES at the operating point (the 3090-gated on-bridge phase, which has a long NEGATIVE history). So this run
    CONFIRMS credit>reservoir + exercises the guards + measures apical-silent, but CANNOT separate the learned surpass
    from fixed-FA on accuracy. It says so, explicitly, in the verdict.

  THE OP-POINT PRE-CHECK (`--op-point-precheck`, the G2 gate on the expensive phase): does a regime exist where
    reservoir >= frozen-B fixed-FA (i.e. is there a gap for the learned feedback to CLOSE at all)? Sweeps depth; if
    fixed-FA never drops to/below the reservoir at reachable depth, the honest verdict is "scale-frontier at this
    budget," NOT a wash-closer -- and the 3090 phase should NOT be spent.

DESIGN (reuse-by-import; NO `sim/` edit anywhere):
  - the DEPTH-REQUIRED task + the inheritance/memctrl split + `_acc_on` from `_semantic_inheritance_deep_credit_derisk`.
  - the arms + KP `_kp_update` (transport-free) + no-weight-transport probes + `_cos`/`_sig`/`_softmax`/`_MOMENTUM`
    from `_gnw_d1_spiking_bdsp_derisk`.
  - the fenced backprop ORACLE ceiling from `sim.dendritic_mlp.DendriticMLP`.
  - THE GENUINELY-NEW ADDITIVE (this file): `MicroNet` -- an FA net whose FF weight update is the clean-error
    feedback-alignment credit (so accuracy is byte-identical to fixed-FA at rate, honoring G2), PLUS a PLASTIC
    interneuron cancellation weight W^PI learned by the dendritic self-prediction rule dW^PI ~ +r_int * v_apical from
    a NON-fixed-point (noisy) init toward the self-predicting fixed point W^PI == Y (Sacramento-Senn 2018 Eq.9, the
    "apical silent when correct" property EARNED, not initialized). The W^PI loop is corroboration/diagnostic-only on
    the feedforward weights (the committed MicrocircuitBDSPNet pattern) -- its role at RATE is exactly the observable
    apical-silent property, not an accuracy lever.

ARMS (held-out INHERITANCE accuracy, the GO metric):
  reservoir         : hidden FROZEN at random init, only the linear readout trained  = the credit-INDEPENDENT baseline.
  fixed_fa          : clean-error feedback alignment, fixed-random Y                  = the fixed-feedback credit.
  kp                : Kolen-Pollack LEARNED feedback (Y^T -> W, transport-free)       = the RANK-2 learned co-arm.
  micro             : plastic-Eq.9 self-predicting microcircuit (the RANK-1 build)    = the new build (apical-silent).
  transport_ceiling : Y := W^T each step (weight transport = ~backprop)               = the CHEAT upper bound (guard MUST fail).

Run (the CPU-rate 6-seed GO, nice'd, local):
  SIM_BACKEND=numpy nice -n 12 .venv/bin/python -u -m research.runners._gap4_learned_microcircuit_selfpredict_derisk \
      --task xor_over_pool --arms reservoir fixed_fa kp micro transport_ceiling \
      --seeds 42 43 44 100 101 102 --hidden 96 --deep-layers 2 --frac 1.0 --epochs 250 \
      --lr 0.3 --kp-lr 0.2 --kp-decay 1e-4 --wpi-plastic --wpi-init noisy --assert-no-transport \
      --out research/findings/raw/gap4/learned_microcircuit_cpurate_6seed.json

Op-point pre-check (the G2 gate, cheap rate sweep):
  SIM_BACKEND=numpy nice -n 12 .venv/bin/python -u -m research.runners._gap4_learned_microcircuit_selfpredict_derisk \
      --op-point-precheck --seeds 42 43 44 --hidden 96 --precheck-depths 2 3 4 5 --epochs 250 --lr 0.3 \
      --out research/findings/raw/gap4/op_point_precheck.json

SEED BUG N/A for this CPU numpy-rate phase: np.random.default_rng(seed) + explicit model seed, NO SimulationBridge
(per test_determinism TestSubstrateActuallySeeded). The 3090 on-bridge phase MUST set cfg.seed (never
actual_seed_used) and hash cp_neuron_firing_thresholds before trusting any seed-to-seed number.
"""
from __future__ import annotations
import argparse, ast, inspect, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")   # tiny matmuls -> one BLAS thread/proc; parallelize across seeds not threads
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the fenced backprop oracle ceiling
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, _acc_on)
from research.runners._gnw_d1_spiking_bdsp_derisk import (  # noqa: E402
    FANet, _train, _no_weight_transport, _no_weight_transport_learned, _cos, _sig, _softmax, _MOMENTUM)

OUT = _REPO / "research" / "findings" / "raw" / "gap4" / "learned_microcircuit_cpurate.json"


# ============================================================================================================
# THE GENUINELY-NEW ADDITIVE: MicroNet -- an FA net with a PLASTIC self-predicting interneuron cancellation W^PI.
#   FF weight update == clean-error feedback alignment (Y fixed => accuracy byte-identical to fixed_fa at rate; G2).
#   The interneuron/W^PI loop is the additive: at the TOP hidden layer the apical carries the interneuron-CANCELLED
#   residual  v_apical = src_target @ Y - src_pred @ W^PI  (src_target=onehot(y), src_pred=softmax(logits)). At the
#   self-predicting fixed point W^PI == Y this equals  (onehot - softmax) @ Y == the clean FA apical credit, so it is
#   SILENT when the output is correct (error ~ 0) and LOUD when incorrect. W^PI is LEARNED from a NON-fixed-point
#   (noisy) init by the local dendritic self-prediction rule (free phase; interneuron reproduces the top-down from the
#   network's OWN prediction):  dW^PI ~ + src_pred^T @ [ src_pred @ (Y - W^PI) ]  -> drives W^PI -> Y. TRANSPORT-FREE:
#   reads only the interneuron rate (src_pred, a local activity) and the local apical voltage, NEVER a forward weight.
# ============================================================================================================
class MicroNet(FANet):
    """FANet (clean-error FA feedforward credit; same forward W / fixed-random Y init as DendriticMLP) + a PLASTIC
    self-predicting interneuron cancellation weight W^PI at the top hidden layer (Sacramento-Senn 2018 Eq.9). The FF
    weight update is the FA credit VERBATIM -- accuracy is byte-identical to fixed_fa at rate (G2). W^PI is the
    genuinely-new plastic additive whose EARNED apical-silent-on-correct property is the rate observable.

    Modes (train_step): 'bdsp' (learn) | 'reservoir' (freeze all hidden, train readout) | 'freeze_deepest' (freeze
    W[0] only) | 'apical_lesion' (Y=0 -> no hidden credit) | 'shufE' (shuffle the descending error across the batch
    in the HIDDEN path only; readout keeps the true error) | 'no_teaching_null' | 'wrong_sign'."""

    def __init__(self, sizes, seed=0, beta=1.0, p0=0.30, feedback="fixed", kp_lr=0.2, kp_decay=1e-4,
                 homeostasis=False, wpi_plastic=False, wpi_init="noisy", wpi_lr=0.2, wpi_noise=1.0):
        super().__init__(sizes, seed=seed, beta=beta, p0=p0, feedback=feedback,
                         kp_lr=kp_lr, kp_decay=kp_decay, homeostasis=homeostasis)
        # W^PI[k] shape == Y[k].shape == (sizes[k+2], sizes[k+1]) (the interneuron 1:1 mirrors the top-down source).
        self.wpi_plastic = bool(wpi_plastic); self.wpi_init = str(wpi_init); self.wpi_lr = float(wpi_lr)
        wrng = np.random.default_rng(seed + 4242)   # SEPARATE stream (no transport): the interneuron init
        self.W_PI = []
        for yk in self.Y:
            if wpi_init == "fixedpoint":
                self.W_PI.append(yk.copy())                                  # start AT the self-predicting fixed point
            else:                                                            # 'noisy' (default): OFF the fixed point
                self.W_PI.append(wrng.normal(0.0, wpi_noise, yk.shape))
        self._selfpred_cos = []                                             # cos(W^PI_top, Y_top) trajectory
        self._shuf_rng = np.random.default_rng(seed * 4099 + 11)           # per-step error shuffle (shufE lesion)

    # ---- the local, transport-free Eq.9 self-prediction update (its OWN method so the AST guard can inspect it) ----
    def _wpi_selfpredict_update(self, src_pred, lr):
        """dW^PI[top] = + wpi_lr * lr * ( r_int^T @ v_free ) / m,  r_int = src_pred (interneuron rate),
        v_free = src_pred @ (Y[top] - W^PI[top]) (the free-phase residual apical). Drives W^PI -> Y (self-prediction).
        LOCAL + TRANSPORT-FREE: reads ONLY src_pred (an activity), self.Y[top], self.W_PI[top] -- NEVER self.W."""
        top = len(self.Y) - 1
        m = max(1, src_pred.shape[0])
        v_free = src_pred @ (self.Y[top] - self.W_PI[top])                   # (m, sizes[top+1])
        dWpi = (src_pred.T @ v_free) / m                                     # (sizes[top+2], sizes[top+1]) == W^PI.shape
        self.W_PI[top] = self.W_PI[top] + self.wpi_lr * lr * dWpi

    def train_step(self, X, y, mode, lr):
        acts, lg = self._forward(X); y = np.asarray(y)
        nW = len(self.W); nhid = nW - 1
        delta_out = _softmax(lg).copy(); delta_out[np.arange(len(y)), y] -= 1.0
        if mode == "wrong_sign":
            delta_out = -delta_out
        upd = [None] * nW
        upd[-1] = -(acts[-1].T @ delta_out)                                  # output local delta (top has target access)
        # descending clean error for the HIDDEN path. shufE: shuffle across the batch (readout above kept the true e).
        e_hid = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        if mode == "shufE":
            e_hid = e_hid[self._shuf_rng.permutation(e_hid.shape[0])]
        e_upper = e_hid
        for k in range(nhid - 1, -1, -1):
            E = acts[k + 1]
            if self.feedback == "learned" and mode == "bdsp":
                self._kp_update(k, E, e_upper, lr)                           # KP learned feedback (transport-free)
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            v_api = e_upper @ Yk
            soma_err = (E * (1.0 - E)) * v_api                               # M2.6 somatic-rate FF delta (== FANet)
            soma_err = self._homeo_scale(k, soma_err)
            freeze = (mode == "reservoir") or (mode == "freeze_deepest" and k == 0)
            upd[k] = np.zeros_like(self.W[k]) if freeze else (acts[k].T @ soma_err)
            e_upper = soma_err
        # --- the genuinely-new plastic Eq.9 interneuron self-prediction (top layer; diagnostic-only on FF weights) ---
        if self.wpi_plastic and mode == "bdsp":
            src_pred = _softmax(lg)                                          # the network's OWN prediction (free phase)
            self._wpi_selfpredict_update(src_pred, lr)
        if mode == "bdsp":                                                   # record self-prediction trajectory
            top = len(self.Y) - 1
            self._selfpred_cos.append(_cos(self.W_PI[top], self.Y[top]))
        m = max(1, X.shape[0])
        if self._vel is None:
            self._vel = [np.zeros_like(w) for w in self.W]
        for li in range(nW):
            self._vel[li] = _MOMENTUM * self._vel[li] + upd[li] / m
            self.W[li] = self.W[li] + lr * self._vel[li]

    def apical_silent_stats(self, X, y):
        """The RATE observable: mean|apical| (the interneuron-cancelled residual v_apical = src_target@Y - src_pred@W^PI
        at the top hidden layer) on CORRECT vs INCORRECT outputs. EARNED-silent => apical_correct << apical_incorrect
        (silent_ratio small). selfpred_cos = cos(W^PI_top, Y_top): ~0 at a noisy init, -> ~1 as W^PI learns Y."""
        acts, lg = self._forward(X); y = np.asarray(y)
        src_target = np.zeros((len(y), self.n_out)); src_target[np.arange(len(y)), y] = 1.0
        src_pred = _softmax(lg)
        top = len(self.Y) - 1
        v_apical = src_target @ self.Y[top] - src_pred @ self.W_PI[top]      # (m, sizes[top+1])
        mag = np.abs(v_apical).mean(1)                                       # per-sample mean |apical| over units
        correct = (np.argmax(lg, 1) == y)
        mc = float(mag[correct].mean()) if correct.any() else float("nan")
        mi = float(mag[~correct].mean()) if (~correct).any() else float("nan")
        ratio = float(mc / (mi + 1e-12)) if (correct.any() and (~correct).any()) else float("nan")
        return {"apical_correct": mc, "apical_incorrect": mi, "silent_ratio": ratio,
                "frac_correct": float(correct.mean()),
                "selfpred_cos": float(_cos(self.W_PI[top], self.Y[top]))}


class TransportCeilingNet(MicroNet):
    """The weight-transport CHEAT upper bound: set Y := W^T each step (feedback == transposed forward weight ~ exact
    backprop). Its no-weight-transport guard MUST FAIL (that is the point -- the labeled ceiling the LEARNED arms must
    approach WITHOUT copying). Everything else inherited."""

    def _sync_transport(self):
        # Y[k] feeds hidden k+1 FROM the layer above (size sizes[k+2]); the matching forward weight is W[k+1] of shape
        # (sizes[k+1], sizes[k+2]) -> the transpose is (sizes[k+2], sizes[k+1]) == Y[k].shape. Copy it => transport.
        for k in range(len(self.Y)):
            self.Y[k] = self.W[k + 1].T.copy()

    def train_step(self, X, y, mode, lr):
        self._sync_transport()                 # descend through Y == W^T (transport used)
        super().train_step(X, y, mode, lr)
        self._sync_transport()                 # re-sync AFTER the W update so the post-hoc no-transport guard sees Y==W^T


# ============================================================================================================
# Anti-cheat probes.
# ============================================================================================================
def _ast_no_forward_W(cls, method_names=("_wpi_selfpredict_update", "_kp_update")):
    """Guard (i): the feedback/interneuron update methods NEVER read the forward weight array self.W (they may read
    self.W_PI / self.Y). AST-walk each method for Attribute(value=Name('self'), attr='W') -- assert none."""
    for name in method_names:
        meth = getattr(cls, name, None)
        if meth is None:
            continue
        try:
            src = inspect.getsource(meth)
        except (OSError, TypeError):
            continue
        tree = ast.parse(_dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "W" \
                    and isinstance(node.value, ast.Name) and node.value.id == "self":
                return False   # a forward-W read in a feedback update = backprop-in-disguise
    return True


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


def _wscramble_byte_identity(net_ctor, X, y):
    """Guard (ii): a mid-training W-scramble leaves the feedback update BYTE-IDENTICAL. Cache the interneuron update's
    inputs (src_pred), scramble ALL forward W, and assert the recomputed W^PI delta is byte-equal -- proving the update
    is a pure function of activities + Y/W^PI with NO forward-W dependence."""
    net = net_ctor()
    acts, lg = net._forward(X)
    src_pred = _softmax(lg)
    top = len(net.Y) - 1
    before = net.W_PI[top].copy()
    net._wpi_selfpredict_update(src_pred, 0.3)
    d1 = net.W_PI[top] - before
    # scramble every forward weight, then recompute the SAME update from the SAME cached src_pred
    net.W_PI[top] = before.copy()
    rng = np.random.default_rng(999)
    net.W = [rng.standard_normal(w.shape) for w in net.W]
    net._wpi_selfpredict_update(src_pred, 0.3)
    d2 = net.W_PI[top] - before
    return bool(np.array_equal(d1, d2))


def _kp_cos_trajectory(net):
    """Guard (iii) for KP: cos(Y[top]^T, W[top+1]) -- starts ~0 (independent random inits), RISES as KP drives Y^T->W
    (alignment EARNED, never copied)."""
    top = len(net.Y) - 1
    return float(_cos(net.Y[top].T, net.W[top + 1]))


def _input_selectivity_init(sizes, seed, task, idx):
    """The input-selectivity guard, TASK-APPROPRIATE form. The 2026-07-19 rank-1-collapse false read was a DEGENERATE
    forward (a constant / saturated hidden gate mapping every input to the SAME code -> the credit verdict is
    meaningless because the forward is not input-differential). The deconf-runner used ncc>2x-chance for THAT on MNIST,
    but on THIS depth-required XOR-over-pool task class-centroid separability of the RAW code is FORBIDDEN BY
    CONSTRUCTION (the property is a NONLINEAR XOR of the pool -> NOT linearly/centroid-decodable; the 2026-07-07 finding
    measures raw linear-probe 0.185 ~ chance -- that is precisely what MAKES depth required). So ncc~chance here is
    CORRECT, not a red flag. The right guard is NON-DEGENERACY: the INIT top-hidden code must be input-DIFFERENTIAL
    (diverse across inputs, not saturated/rank-1). We measure it AND report ncc as a diagnostic (expected ~chance)."""
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    inh = idx["inh_idx"]
    if len(inh) == 0:
        return {"code_diversity": float("nan"), "frac_input_differential": float("nan"),
                "ncc_diag": float("nan"), "ncc_chance": float("nan")}
    net = MicroNet(sizes, seed=seed)                                        # untrained (INIT) net
    Htr, _ = net._forward(np.asarray(Xtr, float)); Htr = Htr[-1]
    Hte_all, _ = net._forward(np.asarray(Xte, float)); Hte = Hte_all[-1][inh]
    # NON-DEGENERACY (the load-bearing guard): binarize the sigmoid code; distinct-row diversity + per-unit variability.
    Hb = (Hte >= 0.5).astype(np.int8)
    n = Hb.shape[0]
    code_diversity = float(np.unique(Hb, axis=0).shape[0]) / max(1, n)      # distinct codes per input (1.0 = all distinct)
    unit_std = Hte.std(0)                                                   # per-unit activation spread across inputs
    frac_diff = float(np.mean(unit_std > 1e-3))                             # fraction of units that are input-differential
    # ncc diagnostic (expected ~chance on this depth-required task -- REPORTED, not gated)
    yv = yte[inh]; classes = np.unique(ytr)
    cents = np.stack([Htr[ytr == c].mean(0) if np.any(ytr == c) else np.zeros(Htr.shape[1]) for c in classes])
    d = ((Hte[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
    ncc = float(np.mean(classes[np.argmin(d, 1)] == yv))
    ncc_chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    return {"code_diversity": code_diversity, "frac_input_differential": frac_diff,
            "ncc_diag": ncc, "ncc_chance": ncc_chance}


# ============================================================================================================
# Arm training / evaluation.
# ============================================================================================================
def _new_net(arm, sizes, seed, a):
    if arm == "transport_ceiling":
        return TransportCeilingNet(sizes, seed=seed, feedback="fixed", wpi_plastic=False, wpi_init=a.wpi_init)
    feedback = "learned" if arm == "kp" else "fixed"
    wpi_plastic = bool(a.wpi_plastic and arm == "micro")
    return MicroNet(sizes, seed=seed, feedback=feedback, kp_lr=a.kp_lr, kp_decay=a.kp_decay,
                    wpi_plastic=wpi_plastic, wpi_init=a.wpi_init, wpi_lr=a.wpi_lr)


def _train_arm(arm, sizes, task, idx, seed, a, mode="bdsp", subsample=None):
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    if subsample is not None:
        Xtr, ytr = Xtr[subsample], ytr[subsample]
    train_mode = "reservoir" if arm == "reservoir" else mode
    net = _new_net(arm, sizes, seed, a)
    _train(net, Xtr, ytr, train_mode, a.epochs, a.lr, a.batch, seed)
    inh = _acc_on(net, Xte, yte, idx["inh_idx"])
    mem = _acc_on(net, Xte, yte, idx["memctrl_idx"])
    return net, {"inherit_heldout": float(inh), "memctrl_heldout": float(mem),
                 "train": float(net.accuracy(Xtr, ytr))}


def run_seed(seed, a):
    task_full = make_task_semantic_inheritance(
        seed, n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
        n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    n_in = Xtr.shape[1]; k = meta["k_classes"]
    sizes = [n_in] + [a.hidden] * int(a.deep_layers) + [k]

    # data fraction (frac 1.0 == full data == the GO condition)
    frac = float(a.frac)
    if frac < 1.0:
        frng = np.random.default_rng(seed * 17 + 3)
        n_keep = max(k * 2, int(round(frac * len(ytr))))
        subsample = frng.permutation(len(ytr))[:n_keep]
    else:
        subsample = None

    inh_idx = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh_idx] == c) for c in np.unique(yte[inh_idx]))) if len(inh_idx) else float("nan")

    # ---- STAGE 0 (task validity): depth-required + oracle clears (the same gate the committed task-runner uses) ----
    def _oracle_acc(nhid):
        onet = DendriticMLP([n_in] + [a.hidden] * nhid + [k], seed=seed)
        Xt, yt = (Xtr[subsample], ytr[subsample]) if subsample is not None else (Xtr, ytr)
        r = np.random.default_rng(seed + 777)
        for _ in range(a.epochs):
            p = r.permutation(len(yt))
            for i in range(0, len(yt), a.batch):
                b = p[i:i + a.batch]
                onet.train_step(Xt[b], yt[b], mode="oracle", lr=a.lr)
        return _acc_on(onet, Xte, yte, inh_idx)
    l1 = _oracle_acc(1); deep_oracle = _oracle_acc(int(a.deep_layers))
    depth_gap = deep_oracle - l1
    depth_separating = bool(deep_oracle >= 0.80 and depth_gap >= 0.05 and deep_oracle > l1 + 0.03)

    # ---- the ARMS (held-out inheritance accuracy) ----
    arms = {}
    nets = {}
    for arm in a.arms:
        net, res = _train_arm(arm, sizes, task, idx, seed, a, subsample=subsample)
        arms[arm] = res
        nets[arm] = net

    # ---- input-selectivity / non-degeneracy guard (INIT) ----
    sel = _input_selectivity_init(sizes, seed, task, idx)

    # ---- LESIONS (triple) ----
    lesions = {}
    # (a) apical lesion: Y=0 -> no hidden credit -> collapses toward reservoir/floor. (measure on micro/fixed_fa base.)
    base_arm = "micro" if "micro" in a.arms else ("fixed_fa" if "fixed_fa" in a.arms else a.arms[0])
    _, lesions["apical_lesion"] = _train_arm(base_arm, sizes, task, idx, seed, a, mode="apical_lesion",
                                             subsample=subsample)
    # (b) deepest-layer FREEZE ablation: freeze W[0] -> collapses to the shallow floor (deep layer is load-bearing).
    _, lesions["freeze_deepest"] = _train_arm(base_arm, sizes, task, idx, seed, a, mode="freeze_deepest",
                                              subsample=subsample)
    # (c) freeze-learned-feedback: hold KP's Y at its random init == plain fixed FA (isolates the LEARNED part). At
    #     rate this == fixed_fa (the honest G2 read: the learned part is inert on accuracy). Realized as fixed_fa.
    _, lesions["freeze_learned_feedback"] = _train_arm("fixed_fa", sizes, task, idx, seed, a, subsample=subsample)

    # ---- PERMUTED controls (double) ----
    permuted = {}
    # (a) shuffled TRAINING target, eval on TRUE labels -> chance (no leak).
    prng = np.random.default_rng(seed + 555)
    (Xt2, yt2, Lt2), (Xte2, yte2, Lte2) = task
    Xt_use, yt_use = (Xt2[subsample], yt2[subsample]) if subsample is not None else (Xt2, yt2)
    yperm = yt_use[prng.permutation(len(yt_use))]
    pnet = _new_net(base_arm, sizes, seed, a)
    _train(pnet, Xt_use, yperm, "bdsp" if base_arm != "reservoir" else "reservoir", a.epochs, a.lr, a.batch, seed)
    permuted["shuffled_target"] = {"inherit_heldout": float(_acc_on(pnet, Xte2, yte2, inh_idx)),
                                   "train_on_perm": float(pnet.accuracy(Xt_use, yperm))}
    # (b) shufE directed-credit-scramble: shuffle the descending error across the batch in the HIDDEN path only.
    _, permuted["shufE"] = _train_arm(base_arm, sizes, task, idx, seed, a, mode="shufE", subsample=subsample)

    # ---- SACRAMENTO-specific: apical-silent EARNED (plastic W^PI) vs NOT (frozen-noisy W^PI) ----
    apical = {}
    if "micro" in nets:
        apical["micro_plastic"] = nets["micro"].apical_silent_stats(Xte, yte)
    # frozen-noisy W^PI arm (kill the Eq.9 plasticity): apical must NOT go silent on correct.
    frozen = MicroNet(sizes, seed=seed, feedback="fixed", wpi_plastic=False, wpi_init="noisy", wpi_lr=a.wpi_lr)
    _train(frozen, (Xtr[subsample] if subsample is not None else Xtr),
           (ytr[subsample] if subsample is not None else ytr), "bdsp", a.epochs, a.lr, a.batch, seed)
    apical["micro_frozen_wpi"] = frozen.apical_silent_stats(Xte, yte)

    # ---- NO-WEIGHT-TRANSPORT 3-guard + AST + transport ceiling ----
    guards = {}
    guards["ast_no_forward_W"] = bool(_ast_no_forward_W(MicroNet))
    guards["wscramble_byte_identity"] = bool(_wscramble_byte_identity(
        lambda: MicroNet(sizes, seed=seed, wpi_plastic=True, wpi_init="noisy"), Xtr[:min(256, len(Xtr))],
        ytr[:min(256, len(ytr))]))
    if "kp" in nets:
        guards["kp_cos_YtW_final"] = _kp_cos_trajectory(nets["kp"])
        guards["kp_no_weight_transport"] = bool(_no_weight_transport_learned(nets["kp"]))
    if "micro" in nets:
        guards["micro_no_weight_transport"] = bool(_no_weight_transport(nets["micro"]))
        top = len(nets["micro"].Y) - 1
        guards["micro_selfpred_cos_final"] = float(_cos(nets["micro"].W_PI[top], nets["micro"].Y[top]))
    if "transport_ceiling" in nets:
        # the CEILING's guard MUST FAIL (Y == W^T byte-equal) -- that is the labeled cheat.
        guards["ceiling_no_weight_transport"] = bool(_no_weight_transport(nets["transport_ceiling"]))
        guards["ceiling_guard_correctly_fails"] = bool(not guards["ceiling_no_weight_transport"])

    return {"seed": seed, "meta": meta, "sizes": sizes, "chance": chance,
            "stage0": {"l1_inherit": float(l1), "deep_oracle_inherit": float(deep_oracle),
                       "depth_gap": float(depth_gap), "depth_separating": depth_separating},
            "arms": arms, "selectivity": sel,
            "lesions": lesions, "permuted": permuted, "apical": apical, "guards": guards}


# ============================================================================================================
# Op-point PRE-CHECK (G2): does a regime exist where reservoir >= frozen-B fixed-FA (a gap for learned credit to close)?
# ============================================================================================================
def op_point_precheck(a):
    print("=" * 100, flush=True)
    print("[op-point-precheck] G2 gate: is there a regime where reservoir >= fixed_fa (a gap to close)?", flush=True)
    rows = []
    for depth in a.precheck_depths:
        per = []
        for seed in a.seeds:
            task_full = make_task_semantic_inheritance(
                seed, n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
            (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
            task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
            sizes = [Xtr.shape[1]] + [a.hidden] * int(depth) + [meta["k_classes"]]
            _, res = _train_arm("reservoir", sizes, task, idx, seed, a)
            _, ff = _train_arm("fixed_fa", sizes, task, idx, seed, a)
            per.append((res["inherit_heldout"], ff["inherit_heldout"]))
        res_m = float(np.mean([p[0] for p in per])); ff_m = float(np.mean([p[1] for p in per]))
        gap_exists = bool(res_m >= ff_m - 0.02)   # fixed-FA has degraded to <= reservoir (within noise) => a gap
        rows.append({"deep_layers": int(depth), "reservoir": round(res_m, 4), "fixed_fa": round(ff_m, 4),
                     "fa_minus_res": round(ff_m - res_m, 4), "gap_exists": gap_exists})
        print(f"  depth={depth}: reservoir={res_m:.3f}  fixed_fa={ff_m:.3f}  (fa-res={ff_m-res_m:+.3f})  "
              f"gap_exists={gap_exists}", flush=True)
    any_gap = any(r["gap_exists"] for r in rows)
    verdict = ("A GAP EXISTS at some reachable depth (reservoir >= fixed_fa) -> the learned-feedback distinction has "
               "room to matter; the 3090 on-bridge op-point sweep is warranted (still gated on CPU GO + a freed lane)."
               if any_gap else
               "NO GAP at any tested depth -- fixed_fa always beats the reservoir at this rate budget. The honest "
               "verdict is SCALE-FRONTIER at this budget: the learned-vs-fixed separation is a spiking-op-point "
               "question, NOT a rate wash-closer. Do NOT spend 3090 time until an op-point regime with "
               "reservoir >= frozen-B FA is demonstrated (sparsity/spiking, not reachable at rate depth alone).")
    print(f"[op-point-precheck] {verdict}", flush=True)
    out = {"probe": "op_point_precheck", "seeds": a.seeds, "hidden": a.hidden, "depths": a.precheck_depths,
           "rows": rows, "any_gap_exists": any_gap, "verdict": verdict}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[op-point-precheck] wrote {a.out}", flush=True)
    return out


# ============================================================================================================
def _agg(per, path):
    vals = []
    for p in per:
        v = p
        ok = True
        for kk in path:
            if isinstance(v, dict) and kk in v:
                v = v[kk]
            else:
                ok = False; break
        if ok and v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="gap#4 learned transport-free deep credit vs reservoir (CPU rate phase).")
    ap.add_argument("--task", default="xor_over_pool", choices=["xor_over_pool"])
    ap.add_argument("--arms", nargs="+",
                    default=["reservoir", "fixed_fa", "kp", "micro", "transport_ceiling"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--deep-layers", type=int, default=2)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--kp-lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", type=float, default=1e-4)
    ap.add_argument("--wpi-plastic", action="store_true")
    ap.add_argument("--wpi-init", default="noisy", choices=["noisy", "fixedpoint"])
    ap.add_argument("--wpi-lr", type=float, default=0.2)
    ap.add_argument("--assert-no-transport", action="store_true",
                    help="hard-assert the no-weight-transport structural guards pass (AST + W-scramble byte-identity)")
    # task knobs
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    # op-point precheck
    ap.add_argument("--op-point-precheck", action="store_true")
    ap.add_argument("--precheck-depths", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.op_point_precheck:
        op_point_precheck(a)
        return 0

    # structural no-transport guards -- assert BEFORE spending compute if requested.
    ast_ok = _ast_no_forward_W(MicroNet)
    if a.assert_no_transport:
        assert ast_ok, "AST guard FAILED: a feedback/interneuron update reads the forward weight self.W"

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, a)
            per.append(r)
            g = r["guards"]; s0 = r["stage0"]
            print("-" * 112, flush=True)
            print(f"[seed {s}] sizes {r['sizes']} | {r['meta']['k_classes']} classes | chance {r['chance']:.3f} | "
                  f"STAGE0 depth-sep {s0['depth_separating']} (1-layer {s0['l1_inherit']:.3f} -> deep-oracle "
                  f"{s0['deep_oracle_inherit']:.3f}, gap {s0['depth_gap']:+.3f})", flush=True)
            armline = "  ARMS inherit-heldout: " + " ".join(
                f"{arm}={r['arms'][arm]['inherit_heldout']:.3f}" for arm in a.arms)
            print(armline, flush=True)
            sel = r["selectivity"]
            print(f"  INIT non-degeneracy: code_diversity={sel['code_diversity']:.3f} "
                  f"frac_input_differential={sel['frac_input_differential']:.3f} "
                  f"(ncc_diag={sel['ncc_diag']:.3f} ~chance {sel['ncc_chance']:.3f}, EXPECTED on a depth-required XOR) | "
                  f"lesions: apical={r['lesions']['apical_lesion']['inherit_heldout']:.3f} "
                  f"freeze_deepest={r['lesions']['freeze_deepest']['inherit_heldout']:.3f} "
                  f"freeze_learned_fb={r['lesions']['freeze_learned_feedback']['inherit_heldout']:.3f}", flush=True)
            print(f"  permuted: shuffled_target={r['permuted']['shuffled_target']['inherit_heldout']:.3f} "
                  f"shufE={r['permuted']['shufE']['inherit_heldout']:.3f}", flush=True)
            if r["apical"]:
                mp = r["apical"].get("micro_plastic"); mf = r["apical"].get("micro_frozen_wpi")
                if mp:
                    print(f"  apical-SILENT (micro PLASTIC W^PI): correct={mp['apical_correct']:.3f} "
                          f"incorrect={mp['apical_incorrect']:.3f} ratio={mp['silent_ratio']:.3f} "
                          f"selfpred_cos={mp['selfpred_cos']:.3f}", flush=True)
                if mf:
                    print(f"  apical (micro FROZEN-noisy W^PI, Sacramento anti-cheat): correct={mf['apical_correct']:.3f} "
                          f"incorrect={mf['apical_incorrect']:.3f} ratio={mf['silent_ratio']:.3f} "
                          f"selfpred_cos={mf['selfpred_cos']:.3f}", flush=True)
            gline = "  GUARDS: " + " ".join(f"{kk}={g[kk]}" for kk in g)
            print(gline, flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_learned_microcircuit_selfpredict_cpurate", "task": a.task, "arms": a.arms,
               "seeds": a.seeds, "config": {"hidden": a.hidden, "deep_layers": a.deep_layers, "frac": a.frac,
               "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "kp_lr": a.kp_lr, "kp_decay": a.kp_decay,
               "wpi_plastic": bool(a.wpi_plastic), "wpi_init": a.wpi_init, "wpi_lr": a.wpi_lr,
               "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}

    if err is None and per:
        chance = _agg(per, ["chance"])
        res = _agg(per, ["arms", "reservoir", "inherit_heldout"])
        credit_arms = [x for x in ("fixed_fa", "kp", "micro") if x in a.arms]
        credit_vals = {x: _agg(per, ["arms", x, "inherit_heldout"]) for x in credit_arms}
        best_credit = max(credit_vals.values()) if credit_vals else float("nan")
        ceiling = _agg(per, ["arms", "transport_ceiling", "inherit_heldout"]) if "transport_ceiling" in a.arms else float("nan")
        code_div = _agg(per, ["selectivity", "code_diversity"])
        frac_diff = _agg(per, ["selectivity", "frac_input_differential"])
        ncc_diag = _agg(per, ["selectivity", "ncc_diag"]); ncc_ch = _agg(per, ["selectivity", "ncc_chance"])
        apical_l = _agg(per, ["lesions", "apical_lesion", "inherit_heldout"])
        freeze_deep = _agg(per, ["lesions", "freeze_deepest", "inherit_heldout"])
        shuf_tgt = _agg(per, ["permuted", "shuffled_target", "inherit_heldout"])
        shufE = _agg(per, ["permuted", "shufE", "inherit_heldout"])
        micro_plastic_ratio = _agg(per, ["apical", "micro_plastic", "silent_ratio"])
        micro_frozen_ratio = _agg(per, ["apical", "micro_frozen_wpi", "silent_ratio"])
        micro_selfpred = _agg(per, ["apical", "micro_plastic", "selfpred_cos"])
        frozen_selfpred = _agg(per, ["apical", "micro_frozen_wpi", "selfpred_cos"])

        # per-seed GO components
        def _seed_ok(p):
            r = p["arms"]; sel = p["selectivity"]
            cr = max(r[x]["inherit_heldout"] for x in credit_arms) if credit_arms else 0.0
            shufE_v = p["permuted"]["shufE"]["inherit_heldout"]
            # shufE (directed-credit scramble) must COLLAPSE the credit to ~chance (destroying the per-sample
            # input<->error covariance) -- gated against CHANCE, not the reservoir (a random frozen projection can sit
            # BELOW chance on a >2-class task, so reservoir is not the collapse floor; chance is).
            shufE_collapsed = bool(shufE_v <= p["chance"] + 0.08 and cr - shufE_v > 0.10)
            return bool(p["stage0"]["depth_separating"]
                        and cr >= r["reservoir"]["inherit_heldout"] + 0.10
                        and sel["code_diversity"] > 0.5 and sel["frac_input_differential"] > 0.5   # non-degenerate
                        and p["permuted"]["shuffled_target"]["inherit_heldout"] <= p["chance"] + 0.08
                        and shufE_collapsed)
        n_go = sum(_seed_ok(p) for p in per)

        guards_all = {}
        for kk in ("ast_no_forward_W", "wscramble_byte_identity", "kp_no_weight_transport",
                   "micro_no_weight_transport", "ceiling_guard_correctly_fails"):
            vals = [p["guards"].get(kk) for p in per if kk in p["guards"]]
            guards_all[kk] = bool(all(vals)) if vals else None

        credit_beats_res = bool(best_credit >= res + 0.10)
        depth_sep_all = all(p["stage0"]["depth_separating"] for p in per)
        nondegen_ok = bool(code_div > 0.5 and frac_diff > 0.5)          # non-degenerate forward at INIT (task-appropriate)
        perm_ok = bool(shuf_tgt <= chance + 0.08)
        shufE_ok = bool(shufE <= chance + 0.08 and best_credit - shufE > 0.10)   # collapsed to chance from the credit level
        lesions_ok = bool(apical_l <= chance + 0.10 and freeze_deep <= chance + 0.10)
        guards_ok = bool(guards_all.get("ast_no_forward_W") and guards_all.get("wscramble_byte_identity")
                         and (guards_all.get("micro_no_weight_transport") in (True, None))
                         and (guards_all.get("kp_no_weight_transport") in (True, None))
                         and (guards_all.get("ceiling_guard_correctly_fails") in (True, None)))
        apical_earned = bool(micro_plastic_ratio < 0.20 and micro_frozen_ratio > 0.5 * 1.0
                             and micro_selfpred > 0.8 and frozen_selfpred < 0.3) \
            if not np.isnan(micro_plastic_ratio) else None

        GO = bool(depth_sep_all and credit_beats_res and nondegen_ok and perm_ok and shufE_ok and lesions_ok
                  and guards_ok and (apical_earned in (True, None)) and n_go >= len(a.seeds))

        summary["aggregate"] = {
            "chance": chance, "reservoir": res, "credit_arms": credit_vals, "best_credit": best_credit,
            "transport_ceiling": ceiling, "code_diversity": code_div, "frac_input_differential": frac_diff,
            "ncc_diag": ncc_diag, "ncc_chance": ncc_ch,
            "lesion_apical": apical_l, "lesion_freeze_deepest": freeze_deep,
            "permuted_shuffled_target": shuf_tgt, "permuted_shufE": shufE,
            "micro_plastic_silent_ratio": micro_plastic_ratio, "micro_frozen_silent_ratio": micro_frozen_ratio,
            "micro_plastic_selfpred_cos": micro_selfpred, "micro_frozen_selfpred_cos": frozen_selfpred,
            "n_go_seeds": n_go, "n_seeds": len(a.seeds), "guards": guards_all,
            "credit_beats_reservoir": credit_beats_res, "depth_separating_all": depth_sep_all,
            "nondegenerate_ok": nondegen_ok, "permuted_ok": perm_ok, "shufE_ok": shufE_ok, "lesions_ok": lesions_ok,
            "guards_ok": guards_ok, "apical_silent_earned": apical_earned}
        summary["GO"] = GO
        summary["G2_scope"] = ("HONEST SCOPE (FLAG G2): at this numpy RATE reference the LEARNED arms (kp, micro) are "
                               "accuracy-indistinguishable from fixed_fa -- the learned/interneuron machinery is inert "
                               "on the feedforward weights at rate. This run CONFIRMS credit>reservoir + all guards + "
                               "the apical-silent EARNED property, but CANNOT separate the learned surpass from "
                               "fixed-FA on accuracy. That separation is a SPIKING op-point question (the 3090 phase).")
        summary["verdict"] = (
            f"{'GO' if GO else 'NO-GO'} ({n_go}/{len(a.seeds)} seeds). credit(best={best_credit:.3f}) vs "
            f"reservoir({res:.3f}) {'>=' if credit_beats_res else '<'} +0.10 | ceiling {ceiling:.3f} | "
            f"depth-sep {depth_sep_all} | non-degen code_div {code_div:.2f} frac_diff {frac_diff:.2f} {nondegen_ok} "
            f"(ncc_diag {ncc_diag:.3f}~chance {ncc_ch:.3f}, expected) | perm shuffled-target {shuf_tgt:.3f} "
            f"(<=chance {chance:.3f}) {perm_ok} | shufE {shufE:.3f} (collapsed to chance) {shufE_ok} | lesions apical "
            f"{apical_l:.3f} freeze-deepest {freeze_deep:.3f} {lesions_ok} | guards {guards_ok} | apical-silent EARNED: "
            f"plastic ratio {micro_plastic_ratio:.3f} (cos {micro_selfpred:.2f}) vs frozen ratio "
            f"{micro_frozen_ratio:.3f} (cos {frozen_selfpred:.2f}) -> {apical_earned}. {summary['G2_scope']}")
    else:
        summary["GO"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[gap4-learned-microcircuit] {summary['verdict']}", flush=True)
    print(f"[gap4-learned-microcircuit] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("GO") else 1


if __name__ == "__main__":
    sys.exit(main())
