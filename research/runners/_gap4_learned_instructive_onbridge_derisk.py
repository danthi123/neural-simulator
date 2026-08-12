"""gap#4 Lane B RANK-1 -- LEARNED instructive signal vs FIXED-random DFA, ON THE PRODUCTION IZHIKEVICH BRIDGE.

THE QUESTION (owner steer 2026-08-01). The e-prop port `_onbridge_eprop_port_derisk.py` trains deep credit on the
production spiking `OnBridgeBDSPNet` with a FIXED-RANDOM DFA feedback `B_direct`, and today's 6-seed run showed its
`deep_credit_share` is ~0 (0.066 @ pool_k=8, 0.005 @ pool_k=16): full e-prop == a frozen random hidden RESERVOIR, so
training the hidden feedforward pathways adds essentially nothing -- a fixed random projection with a trained readout.
(`research/findings/2026-08-01-gap4-6seed-bar-RUN-deep-credit-control-shuffleDFA-leaks-forward-learning-real-attribution-not.md`.)

The roadmap's RANK-1 named surpass (MASTER ROADMAP §2.8, "the true crux") REPLACES the fixed-random DFA with a LEARNED
SELF-PREDICTING MICROCIRCUIT (Sacramento-Senn 2018 Eq.9): a dendritic interneuron whose cancellation weight W^PI learns
by the local, transport-free self-prediction rule dW^PI ~ +r_int * v_apical from a NOISY init toward the self-predicting
fixed point W^PI == Y (the "apical-silent-when-correct" property EARNED, not initialized). A CPU-RATE version got a
partial GO (`_gap4_learned_instructive_drives_ff_derisk.py`, commit c43a2173) but -- per FLAG G2 -- at RATE the learned
and fixed feedback are accuracy-indistinguishable; the SEPARATION is a SPIKING op-point question. THIS runner ports the
learned-instructive swap onto the SAME spiking bridge as the e-prop port and asks head-to-head:

    does a LEARNED instructive signal produce deep_credit_share > 0 where the fixed-random DFA gives ~0,
    at an n_prop (task depth) where the frozen-hidden reservoir FAILS (drops toward chance)?

THE SWAP (reuse-by-import; subclass `OnBridgeEpropNet`, NO `sim/` edit). At the TOP hidden layer only, the DFA credit
    Lsig = (delta_k @ B_direct[top]) / T                                # fixed-random FA (delta_k = softmax - onehot)
is replaced by the interneuron-CANCELLED learned residual, in the e-prop DESCENT sign convention:
    Lsig = (src_pred @ W^PI_top - src_target @ Y_top) / T               # src_pred=softmax, src_target=onehot(y)
where Y_top := B_direct[top] (the SAME fixed random matrix), so AT the self-predicting fixed point W^PI==Y the two are
BYTE-IDENTICAL -- the ONLY difference between the arms is whether W^PI is LEARNED (plastic, noisy init) or the raw fixed
matrix is used. W^PI is trained per batch by the transport-free Eq.9 rule (reads only src_pred, Y_top, W^PI_top -- NEVER
a forward weight). Lower hidden layers keep the fixed DFA (mirrors the CPU runner's top-layer-only swap).

JUDGING (owner steer): judge deep credit by the CAPABILITY that depends on it (compositional generalization to held-out
XOR-task members), not an abstract metric. GO = the LEARNED arm's deep_credit_share clearly > 0 (>0.15) at an n_prop
where the frozen reservoir FAILS, while the FIXED-DFA arm stays ~0. Both arms share the identical spiking forward + the
frozen-reservoir baseline (arm-independent), so the comparison is apples-to-apples.

ANTI-CHEATS (all reused from the e-prop port, per arm): frozen-hidden reservoir_control (the deep_credit_share
denominator), shuffle-DFA (credit route scrambled across the batch), permuted-label (-> chance). Plus the rate depth
oracle (stage-0 depth-separating gate) and the rate ceiling.

HOST-BOUNDARY SCOPE (honest, flagged): as in the e-prop port, BOTH arms compute the credit signal HOST-SIDE (numpy) and
write it into the substrate's plastic `cp_connections.data`; the spiking FORWARD is on the bridge. So this is a
like-for-like test of fixed-random vs learned-self-predicting FEEDBACK, holding the host/bridge boundary fixed -- NOT a
fully-spiking credit pass. Biologizing the interneuron cancellation onto the substrate is the named follow-on: the
plumbing already exists (`enable_bdsp_microcircuit` + `cp_bdsp_int_drive`, sim/bridge.py:8119) to inject W^PI @ phi(u^I)
as a real apical-cancellation current. This de-risk keeps the host boundary so the ONLY changed variable is the feedback.

SEED: cfg.seed seeds the substrate (verified: OnBridgeBDSPNet passes seed through; the parent sets cfg.seed, NOT
actual_seed_used). W^PI uses a SEPARATE stream (seed+4242) -> no transport, no seed collision with the forward.

Run (SMOKE -- 1-2 seeds, n_prop 3 and 4; the parent launches the full 6-seed after review):
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m research.runners._gap4_learned_instructive_onbridge_derisk \
        --seeds 42 --n-prop 3 --n-super 24 --epochs 80 --hidden 48 --pool-k 1 \
        --out research/findings/raw/gap4/learned_instructive_onbridge_np3_s42.json
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

# reuse-by-import (NO sim/ edit): the ported e-prop spiking net + its trainer/softmax; the task + rate oracle/controls.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _train_eprop, _softmax  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
from research.runners._gnw_d1_spiking_bdsp_derisk import _cos  # noqa: E402
from sim.dendritic_mlp import DendriticMLP  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap4" / "learned_instructive_onbridge.json"


# ============================================================================================================
# The LEARNED-INSTRUCTIVE net: OnBridgeEpropNet forward + e-prop, with the TOP hidden layer's DFA credit replaced
# by the Sacramento Eq.9 interneuron-cancelled learned residual (src_pred@W^PI - src_target@Y). W^PI plastic,
# transport-free. wpi_plastic=False OR wpi_init="fixedpoint"+frozen -> byte-reduces to the fixed-DFA arm.
# ============================================================================================================
class OnBridgeLearnedInstructiveNet(OnBridgeEpropNet):
    def __init__(self, *args, wpi_plastic=True, wpi_init="noisy", wpi_lr=0.1, wpi_noise=1.0,
                 wpi_frozen=False, **kw):
        super().__init__(*args, **kw)
        self.wpi_plastic = bool(wpi_plastic)
        self.wpi_frozen = bool(wpi_frozen)
        self.wpi_lr = float(wpi_lr)
        L = len(self.sizes) - 1
        # top hidden pathway index: its POST is the LAST hidden layer (sizes[L-1]); B_direct[L-2] is its feedback.
        self.top_hidden = L - 2
        self.Y_top = self.B_direct[self.top_hidden]                     # reuse the fixed DFA matrix as top-down Y (k, n_post_phys)
        seed = int(kw.get("seed", 0))
        wrng = np.random.default_rng(seed + 4242)                       # SEPARATE stream (no transport)
        if str(wpi_init) == "fixedpoint":
            self.W_PI_top = self.Y_top.copy()                           # start AT the self-predicting fixed point
        else:
            self.W_PI_top = wrng.normal(0.0, float(wpi_noise), self.Y_top.shape).astype(np.float64)  # noisy (off fixed point)
        self._selfpred_cos = []
        self._cur_src_pred = None
        self._cur_src_target = None
        self._use_learned = bool(self.wpi_plastic or str(wpi_init) == "fixedpoint")

    # ---- the local, transport-free Eq.9 self-prediction update, accumulated over the batch ----
    def _wpi_update_batch(self, src_preds):
        """dW^PI[top] = + wpi_lr * (r_int^T @ v_free) / m,  r_int = src_pred (interneuron rate),
        v_free = src_pred @ (Y_top - W^PI_top) (the free-phase residual). Drives W^PI -> Y_top. Reads ONLY src_pred,
        self.Y_top, self.W_PI_top -- NEVER a forward weight (self.br.cp_connections). Transport-free by construction."""
        P = np.asarray(src_preds, dtype=np.float64)                    # (m, k)
        m = max(1, P.shape[0])
        v_free = P @ (self.Y_top - self.W_PI_top)                      # (m, n_post_phys)
        dWpi = (P.T @ v_free) / m                                      # (k, n_post_phys) == W_PI_top.shape
        self.W_PI_top = self.W_PI_top + self.wpi_lr * dWpi
        self._selfpred_cos.append(_cos(self.W_PI_top, self.Y_top))

    def _accum_grad(self, grads, sp, vv, delta_k, skip_output=False):
        """Identical to OnBridgeEpropNet._accum_grad EXCEPT the top-hidden-layer learning signal, which (when the
        learned instructive is active) rides on the interneuron-cancelled residual instead of the raw fixed DFA."""
        from sim.bptt_snn_gpu import atan_surrogate  # noqa: F401 (parent imports it; kept for parity)
        L = len(self.sizes) - 1
        T = sp.shape[0]
        delta_out_phys = self._broadcast(np.asarray(delta_k, dtype=np.float64), L) / self.pool_k
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
                    continue
                z_pre = sp[t, self.slices[li]].astype(np.float64)
                eps[li] = self.eps_leak * eps[li] + z_pre
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
                    Lsig = delta_out_phys / T
                    if self.output_psi_one:
                        psi = np.ones_like(psi)
                elif li == self.top_hidden and self._use_learned:
                    # THE SWAP: interneuron-cancelled learned residual (e-prop descent sign). At W^PI==Y_top this
                    # equals delta_k @ Y_top / T == the fixed DFA with B=Y_top; a noisy/learning W^PI differs.
                    Lsig = (self._cur_src_pred @ self.W_PI_top - self._cur_src_target @ self.Y_top) / T
                else:
                    Lsig = (np.asarray(delta_k, dtype=np.float64) @ self.B_direct[li]) / T   # fixed-random DFA
                g = Lsig * psi
                grads[li] += np.outer(eps[li], g)

    def train_batch(self, Xb, yb, shuffle_dfa=False, rng=None):
        """Mirror of the parent, PLUS: stash per-example (src_pred, src_target) for the top-layer learned residual,
        then run the transport-free Eq.9 W^PI update over the batch."""
        recs = []
        for i in range(len(Xb)):
            sp, vv, acts = self._forward_record(Xb[i])
            recs.append((sp, vv, self._logits_from(sp, vv, acts)))
        yb = np.asarray(yb)
        src_preds, src_targets, deltas = [], [], []
        for (sp, vv, logits), y in zip(recs, yb):
            p = _softmax(logits / self.logit_temp)                     # temperature -> graded deltas (as parent)
            tgt = np.zeros_like(p); tgt[int(y)] = 1.0
            src_preds.append(p); src_targets.append(tgt); deltas.append(p - tgt)   # delta_k = softmax - onehot
        if shuffle_dfa and rng is not None and len(deltas) > 1:
            perm = rng.permutation(len(deltas))
            deltas = [deltas[j] for j in perm]
            src_preds_s = [src_preds[j] for j in perm]
            src_targets_s = [src_targets[j] for j in perm]
        else:
            src_preds_s, src_targets_s = src_preds, src_targets
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        leaky = (self.logit_source == "leaky_readout")
        for (sp, vv, _lg), d, spred, stgt in zip(recs, deltas, src_preds_s, src_targets_s):
            self._cur_src_pred = np.asarray(spred, dtype=np.float64)
            self._cur_src_target = np.asarray(stgt, dtype=np.float64)
            self._accum_grad(grads, sp, vv, d, skip_output=leaky)
            if leaky:
                r = self._readout_feature(sp)
                dphys = self._broadcast(np.asarray(d, dtype=np.float64), L) / self.pool_k
                grads[L - 1] += np.outer(r, dphys)
        self._apply_grads(grads, len(Xb))
        # transport-free Eq.9 interneuron self-prediction (drives W^PI -> Y_top). Uses the TRUE (unshuffled) src_pred:
        # the interneuron learns to predict the network's OWN output, independent of the credit-route scramble.
        if self.wpi_plastic and not self.wpi_frozen:
            self._wpi_update_batch(src_preds)


# ============================================================================================================
# One seed: rate depth-gate + rate oracle ceiling, then BOTH arms (fixed DFA, learned instructive) sharing ONE
# frozen-hidden reservoir baseline, each with permuted + shuffle-DFA controls. deep_credit_share per arm.
# ============================================================================================================
def _mk(arm, n_in, hidden, k, seed, a, hp):
    common = dict(seed=seed, n_hidden_layers=a.n_hidden_layers, settle_steps=a.settle_steps,
                  eprop_lr=a.eprop_lr, eps_leak=a.eps_leak, surrogate=a.surrogate, alpha_surr=a.alpha_surr,
                  beta_surr=a.beta_surr, logit_source=a.logit_source, w_clip=a.w_clip, pool_k=a.pool_k,
                  ou_noise=a.ou_noise, cond_noise=a.cond_noise, stp=a.stp, hp=hp)
    if arm == "fixed":
        return OnBridgeEpropNet(n_in, hidden, k, **common)
    if arm == "learned":
        return OnBridgeLearnedInstructiveNet(n_in, hidden, k, wpi_plastic=True, wpi_init="noisy",
                                             wpi_lr=a.wpi_lr, wpi_noise=a.wpi_noise, **common)
    if arm == "frozen_wpi":
        return OnBridgeLearnedInstructiveNet(n_in, hidden, k, wpi_plastic=False, wpi_init="noisy",
                                             wpi_lr=a.wpi_lr, wpi_noise=a.wpi_noise, wpi_frozen=True, **common)
    raise ValueError(arm)


def _deep_share_code(inh, froz, chance):
    """The e-prop-port/finding definition (inh-froz)/(inh-chance) -- fraction of the arm's OWN above-chance margin
    that is DEEP (not reservoir). Comparable to the finding's numbers, but UNSTABLE when the reservoir sits <= chance
    (denominator (inh-chance) shrinks/flips), which is exactly the reservoir-FAILS regime here -> report the robust one too."""
    if any(np.isnan(x) for x in (inh, froz, chance)) or (inh - chance) <= 1e-9:
        return float("nan")
    return float((inh - froz) / (inh - chance))


def _deep_share_oracle(inh, froz, oracle):
    """ROBUST normalization (the finding's PROSE definition, line 34): (inh-froz)/(oracle-froz) -- the fraction of the
    achievable headroom above the failing reservoir that training the hidden layers captures. Bounded, stable because
    oracle >> froz always. This is the PRIMARY read when the reservoir drops below chance."""
    if any(np.isnan(x) for x in (inh, froz, oracle)) or (oracle - froz) <= 1e-9:
        return float("nan")
    return float((inh - froz) / (oracle - froz))


def run_seed(seed, a, task_kwargs, hp):
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]; inh_idx = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh_idx] == c) for c in np.unique(yte[inh_idx]))) if len(inh_idx) else float("nan")

    # --- STAGE-0 rate depth gate (is the TASK config genuinely depth-required?) ---
    s0 = stage0_depth_genuineness(((Xtr, ytr, Ltr), (Xte, yte, Lte)), idx, k, hidden=96,
                                  epochs=a.oracle_epochs, lr=0.3, batch=128, seed=seed)
    # --- rate oracle ceiling (depth n_hidden_layers) ---
    onet = DendriticMLP([n_in] + [96] * a.n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, a.oracle_epochs, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Xte, yte, inh_idx)

    # --- subsample the train set for the spiking arms (held-out NEVER subsampled) ---
    if a.train_subsample and len(Xtr) > a.train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:a.train_subsample]
        Xtr_b, ytr_b = Xtr[keep], ytr[keep]
    else:
        Xtr_b, ytr_b = Xtr, ytr

    # --- ONE frozen-hidden reservoir baseline (arm-INDEPENDENT: hidden frozen => the credit rule is irrelevant) ---
    fnet = _mk("fixed", n_in, hidden=a.hidden, k=k, seed=seed, a=a, hp=hp)
    fnet.train_layers = {fnet.n_hidden_layers}                          # train ONLY the readout; hidden FF frozen
    _train_eprop(fnet, Xtr_b, ytr_b, a.epochs, a.batch, seed)
    frozen_inh = fnet.acc_on(Xte, yte, inh_idx)

    arms = {}
    for arm in a.arms:                                                 # e.g. ["fixed", "learned"]
        net = _mk(arm, n_in, hidden=a.hidden, k=k, seed=seed, a=a, hp=hp)
        w0 = net.ff_weight_norm()
        _train_eprop(net, Xtr_b, ytr_b, a.epochs, a.batch, seed)
        inh = net.acc_on(Xte, yte, inh_idx)
        train_acc = net.accuracy(Xtr_b, ytr_b)
        ff_moved = float(abs(net.ff_weight_norm() - w0))
        rec = {"inherit_heldout": inh, "train_acc": train_acc, "ff_weight_moved": ff_moved,
               "frozen_hidden_inherit": frozen_inh, "inherit_minus_frozen": float(inh - frozen_inh),
               "deep_credit_share": _deep_share_code(inh, frozen_inh, chance),
               "deep_credit_share_oracle": _deep_share_oracle(inh, frozen_inh, oracle_inh)}
        if isinstance(net, OnBridgeLearnedInstructiveNet):
            rec["selfpred_cos_final"] = float(net._selfpred_cos[-1]) if net._selfpred_cos else float("nan")
        if a.controls:
            prng = np.random.default_rng(seed + 555)
            yperm = ytr_b[prng.permutation(len(ytr_b))]
            pnet = _mk(arm, n_in, hidden=a.hidden, k=k, seed=seed, a=a, hp=hp)
            _train_eprop(pnet, Xtr_b, yperm, a.epochs, a.batch, seed)
            rec["permuted_inherit"] = pnet.acc_on(Xte, yte, inh_idx)
            snet = _mk(arm, n_in, hidden=a.hidden, k=k, seed=seed, a=a, hp=hp)
            _train_eprop(snet, Xtr_b, ytr_b, a.epochs, a.batch, seed, shuffle_dfa=True)
            rec["shuffle_dfa_inherit"] = snet.acc_on(Xte, yte, inh_idx)
        arms[arm] = rec

    return {"seed": seed, "chance": chance, "k_classes": int(k), "n_train_smoke": int(len(ytr_b)),
            "n_prop": task_kwargs["n_prop"], "oracle_inherit": oracle_inh,
            "frozen_hidden_inherit": frozen_inh,
            "stage0_depth_separating": bool(s0.get("depth_separating")),
            "stage0_deep_best": s0.get("deep_best_inherit_heldout"), "stage0_l1": s0.get("l1_inherit_heldout"),
            "stage0_l0_linear": s0.get("linear_inherit_heldout"),
            "arms": arms}


def main():
    ap = argparse.ArgumentParser(description="gap#4 Lane B: learned instructive vs fixed DFA on the Izhikevich bridge.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+", default=["fixed", "learned"],
                    choices=["fixed", "learned", "frozen_wpi"])
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5, help="readout lr (hidden lr = eprop_lr*hidden_lr_scale)")
    ap.add_argument("--eps-leak", type=float, default=0.9)
    ap.add_argument("--surrogate", choices=["atan_vt", "std"], default="atan_vt")
    ap.add_argument("--alpha-surr", type=float, default=0.15)
    ap.add_argument("--beta-surr", type=float, default=1.0)
    ap.add_argument("--logit-source", choices=["spike_sum", "event_rate", "membrane", "leaky_readout"],
                    default="leaky_readout")
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--train-subsample", type=int, default=240)
    ap.add_argument("--pool-k", type=int, default=1)
    ap.add_argument("--ou-noise", action="store_true")
    ap.add_argument("--cond-noise", action="store_true")
    ap.add_argument("--stp", action="store_true")
    ap.add_argument("--no-controls", dest="controls", action="store_false",
                    help="skip the permuted + shuffle-DFA controls (calibration only; the headline run keeps them)")
    ap.add_argument("--oracle-epochs", type=int, default=200, help="rate oracle/stage-0 epochs (raise if oracle floors)")
    # learned-instructive knobs
    ap.add_argument("--wpi-lr", type=float, default=0.1)
    ap.add_argument("--wpi-noise", type=float, default=1.0)
    # tonic/drive hp (the working regime from the e-prop port)
    ap.add_argument("--tonic-h-pA", type=float, default=100.0)
    ap.add_argument("--tonic-o-pA", type=float, default=150.0)
    ap.add_argument("--ff-w-init", type=float, default=2000.0)
    ap.add_argument("--in-current-pA", type=float, default=700.0)
    ap.add_argument("--in-bias-pA", type=float, default=300.0)
    ap.add_argument("--hidden-lr-scale", type=float, default=5.0)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    # task knobs
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha,
              in_current_pA=a.in_current_pA, in_bias_pA=a.in_bias_pA, hidden_lr_scale=a.hidden_lr_scale)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, a, task_kwargs, hp)
            per.append(r)
            # per-seed checkpoint (an interruption must not destroy the arm's work)
            try:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(
                    {"probe": "gap4_learned_instructive_onbridge", "partial": True,
                     "seeds_done": [x["seed"] for x in per], "seeds_requested": list(a.seeds),
                     "config": vars(a), "per_seed": per, "SIGNAL": None,
                     "verdict": "PARTIAL -- run in progress; NOT a verdict."}, indent=2, default=str))
            except Exception as _ck:
                print(f"[warn] checkpoint failed ({type(_ck).__name__}: {_ck})", flush=True)
            print("-" * 108, flush=True)
            print(f"[seed {s}] n_prop={r['n_prop']} k={r['k_classes']} chance {r['chance']:.3f} | "
                  f"STAGE0 depth-sep {r['stage0_depth_separating']} (deep-best {r['stage0_deep_best']:.3f} vs "
                  f"1-layer {r['stage0_l1']:.3f}) | oracle {r['oracle_inherit']:.3f} | "
                  f"FROZEN-reservoir {r['frozen_hidden_inherit']:.3f}", flush=True)
            for arm in a.arms:
                ar = r["arms"][arm]
                extra = ""
                if "selfpred_cos_final" in ar:
                    extra += f" selfpred_cos {ar['selfpred_cos_final']:.3f}"
                if "permuted_inherit" in ar:
                    extra += f" | permuted {ar['permuted_inherit']:.3f} shuffle-DFA {ar['shuffle_dfa_inherit']:.3f}"
                print(f"  [{arm:8s}] inherit {ar['inherit_heldout']:.3f} train {ar['train_acc']:.3f} "
                      f"ff-moved {ar['ff_weight_moved']:.1f} | inh-froz {ar['inherit_minus_frozen']:+.3f} | "
                      f"DCS_oracle {ar['deep_credit_share_oracle']:.3f} DCS_code {ar['deep_credit_share']:.3f}{extra}",
                      flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "gap4_learned_instructive_onbridge", "seeds": a.seeds,
               "config": {"hidden": a.hidden, "n_hidden_layers": a.n_hidden_layers, "settle": a.settle_steps,
                          "epochs": a.epochs, "batch": a.batch, "eprop_lr": a.eprop_lr, "pool_k": a.pool_k,
                          "wpi_lr": a.wpi_lr, "wpi_noise": a.wpi_noise, "train_subsample": a.train_subsample,
                          "hidden_lr_scale": a.hidden_lr_scale, "arms": a.arms, "controls": bool(a.controls),
                          "task": task_kwargs, "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(arm, key):
            vals = [p["arms"][arm][key] for p in per if key in p["arms"][arm] and not np.isnan(p["arms"][arm][key])]
            return float(np.mean(vals)) if vals else float("nan")
        ch = float(np.mean([p["chance"] for p in per]))
        froz = float(np.mean([p["frozen_hidden_inherit"] for p in per]))
        orc = float(np.mean([p["oracle_inherit"] for p in per]))
        agg = {"chance": ch, "oracle_inherit": orc, "frozen_hidden_inherit": froz,
               "stage0_depth_separating_all": all(p["stage0_depth_separating"] for p in per)}
        for arm in a.arms:
            agg[arm] = {"inherit": _m(arm, "inherit_heldout"), "inherit_minus_frozen": _m(arm, "inherit_minus_frozen"),
                        "deep_credit_share_oracle": _m(arm, "deep_credit_share_oracle"),
                        "deep_credit_share_code": _m(arm, "deep_credit_share"), "train_acc": _m(arm, "train_acc")}
            if a.controls:
                agg[arm]["permuted"] = _m(arm, "permuted_inherit")
                agg[arm]["shuffle_dfa"] = _m(arm, "shuffle_dfa_inherit")
        summary["aggregate"] = agg
        # SIGNAL (robust): reservoir fails (well below oracle, toward chance) AND the LEARNED arm's deep-credit margin
        # over the failing reservoir clearly beats the FIXED arm's. Uses the oracle-normalized share (stable when
        # froz<=chance) + the raw margin, NOT the code-share (which blows up when the reservoir is below chance).
        reservoir_fails = bool((orc - froz) > 0.15 and (froz - ch) < 0.20)
        dcs_learned = agg.get("learned", {}).get("deep_credit_share_oracle", float("nan"))
        dcs_fixed = agg.get("fixed", {}).get("deep_credit_share_oracle", float("nan"))
        marg_learned = agg.get("learned", {}).get("inherit_minus_frozen", float("nan"))
        marg_fixed = agg.get("fixed", {}).get("inherit_minus_frozen", float("nan"))
        learned_beats_fixed = bool(not np.isnan(dcs_learned) and dcs_learned > 0.15
                                   and not np.isnan(marg_learned) and marg_learned > 0.05
                                   and (np.isnan(dcs_fixed) or marg_learned > marg_fixed + 0.05))
        summary["SIGNAL"] = bool(reservoir_fails and learned_beats_fixed)
        summary["reservoir_fails"] = reservoir_fails
        summary["learned_beats_fixed"] = learned_beats_fixed
        summary["verdict"] = (
            f"{'GO(smoke)' if summary['SIGNAL'] else 'no-signal(smoke)'}: reservoir {froz:.3f} vs oracle {orc:.3f} "
            f"(chance {ch:.3f}) reservoir_fails={reservoir_fails} | learned DCS_oracle {dcs_learned:.3f} "
            f"(inh-froz {marg_learned:+.3f}) vs fixed DCS_oracle {dcs_fixed:.3f} (inh-froz {marg_fixed:+.3f}) "
            f"learned_beats_fixed={learned_beats_fixed}. SMOKE (1-2 seeds) -- the 6-seed bar is the parent's after review.")
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[gap4-learned-instructive-onbridge] {summary['verdict']}", flush=True)
    print(f"[gap4-learned-instructive-onbridge] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
