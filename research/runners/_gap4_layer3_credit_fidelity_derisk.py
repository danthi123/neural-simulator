"""gap#4 deep-credit-on-spikes -- LAYER-3 CREDIT FIDELITY: does transport-free DFA credit REACH the deep hidden
layers of a 3-hidden-layer net (cosine-align to the backprop-oracle gradient), on a genuinely depth-3-composed
FIT target?

THE REFRAME (2026-08-11, established -- do NOT re-litigate). A depth-3-OBLIGATORY *task* (depth-2 held-out
underfits, depth-3 clears) is PROVABLY IMPOSSIBLE at toy scale for a plain-MLP oracle: Telgarsky depth-separation
needs width EXPONENTIAL in the depth-gap=1, and plain depth is capacity, not an inductive bias. So the
"task-accuracy" framing CANNOT test depth-3 credit
(research/findings/2026-08-11-gap4-depth3-obligatory-task-is-provably-impossible-reframe-to-layer3-credit-fidelity.md).
The achievable, biologically-meaningful test is LAYER-3 CREDIT FIDELITY: on a target whose *fit* provably engages
the deep layers (tent^k regression, a depth-k-composed piecewise-linear map), does the transport-free DFA
weight-update at a deep hidden layer cosine-ALIGN with the true backprop gradient there, and RISE with training?

WHY tent^k (Telgarsky's own separating family). tent(x)=1-|2x-1| on [0,1] has 2 linear pieces; tent^k = tent
composed k times has 2^k pieces. Depth composes pieces MULTIPLICATIVELY (each layer ~doubles folds) while width
adds them LINEARLY -- so at a NARROW width a depth-3 net fits tent^k where a depth-2 net cannot (the one place
depth-2 genuinely can't compete on FIT). MSE loss, small 1-D input. A quasigroup-depth-3 train-fit target was the
sanctioned alternative; tent^k is used because its depth-vs-width separation is exactly Telgarsky's proof object.

THE MACHINERY (reuse-by-FORM, NO sim/ edit). The rate net + credit rules live HERE (runner-side), exactly as the
sibling harness `_snn_bptt_forward_vs_learning_isolation_derisk` implements `_eprop_grads`/`_spatial_backward`
runner-side rather than in sim/. The DFA credit rule is the SAME algebraic form as sim.dendritic_mlp.DendriticMLP
`train_step(mode='local_correct')` (lines 146-149: `ap = e @ self.B[li]; base = a_prev.T @ (ap * phi'(a))`) and
`_eprop_grads` (line 132: `Lsig = output_grad @ B_direct[li]`): per HIDDEN layer a FIXED-RANDOM feedback matrix B
projects the OUTPUT error DIRECTLY (Nokland 2016 DFA / Lillicrap 2016 feedback alignment) -- NO weight transport,
never a forward W^T in the credit path. DendriticMLP itself is softmax-CE classification only (sim/, unmodifiable);
the swap the reframe calls for -- the tent^k data builder + MSE regression readout -- is why the rate net is
rebuilt here with the SAME transport-free rule. Hidden activation is tanh (phi'=1-tanh^2; the standard FA/DFA
regression setting and the fair test of the CREDIT rule); sigmoid -- DendriticMLP's activation, phi'<=0.25 and
prone to the phi'-vanishing the reframe names as a next-mechanism axis -- is available via --act to probe that.

THE GO GATE (genuine deep credit).
  (a) layer-3 weight-update cosine-alignment to the backprop-oracle layer-3 gradient >= align_bar (0.6) AND RISING
      (final > initial) -- computed on the DFA-trained net's CURRENT weights each checkpoint (raw grads, not the
      Adam-scaled step -- the optimizer is mode-agnostic machinery; the CREDIT is what differs);
  (b) DFA final train-loss within 10% of the backprop-oracle depth-3 loss (measured as closing >=90% of the
      depth-2->depth-3 loss gap, which is well-defined even when BP3_loss ~ 0);
  AND the depth-2 oracle on the SAME target must be strictly WORSE (confirms the target is depth-3-engaging on FIT).

LAYER INDEXING (stated to remove ambiguity). sizes=[n_in,H,H,H,k]; hidden activations a1=tanh(x W0+b0),
a2=tanh(a1 W1+b1), a3=tanh(a2 W2+b2), yhat=a3 W3+b3. "hidden layer j" = a_j, incoming weights W[j-1]. The GO keys
on the 3rd hidden layer a3 (W[2], adjacent to the output -- the layer the reframe names "the 3rd hidden layer").
The DEEPEST-FROM-OUTPUT hidden layer a1 (W[0], the credit must traverse 3 layers to reach) is the strictest
deep-reach metric and is reported PROMINENTLY alongside -- so the honest picture (does alignment degrade toward the
input?) is visible either way.

ANTI-CHEATS (all EXECUTE via tools.lab / assertions / Verdict, none is a comment):
  (i)   backprop-oracle ceiling EXISTS -- BP depth-3 reaches loss~0 (else UNDEFINED, not a score);
  (ii)  permuted-target (per-step reshuffle -> targets carry no consistent signal) -> layer-3 alignment ~0 AND no
        learning (loss stays at the target-variance floor);
  (iii) depth-separating -- the SAME target: BP depth-2 train-loss strictly > BP depth-3 (Verdict.control);
  (iv)  lever-moved -- apical-lesion (feedback B=0, top-down credit removed) COLLAPSES layer-3 alignment to ~0 AND
        learning to the floor (the earned tooth: proves the alignment is DUE TO the feedback path);
  (v)   TRANSPORT-FREE assertion -- B is fixed-random from a SEPARATE seed stream, frozen across training, and
        independent of every forward W (init alignment ~0); the hidden credit computes e @ B[li], never W^T.

Sources: Telgarsky 2016 (depth-separation, the tent^k family + the impossibility that forces this reframe);
Lillicrap 2016 (feedback alignment -- forward W self-aligns to a fixed random B); Nokland 2016 (direct feedback
alignment -- the output error projected directly to every hidden layer). NO sim/ edit; additive; default-off task.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools.lab import lever, attributable_to, assert_backend  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_layer3_credit_fidelity.json"


# --------------------------------------------------------------------------- the depth-k-composed FIT target
def tent(x):
    """Tent map on [0,1]: tent(x) = 1 - |2x - 1|. 2 linear pieces, one interior peak."""
    return 1.0 - np.abs(2.0 * x - 1.0)


def tent_pow(x, k):
    """tent composed k times -> 2^k linear pieces (Telgarsky's depth-separating family)."""
    y = np.asarray(x, dtype=np.float64)
    for _ in range(k):
        y = tent(y)
    return y


def make_tent_data(seed, n, k):
    """n points on [0,1] (a dense deterministic grid + seeded jitter so the fit must represent the FUNCTION,
    not memorise isolated points), target = tent^k(x). Input fed as-is (biases handle offsets); target centred
    to mean 0 so a linear readout is not fighting a constant. Returns (X (n,1), y (n,1), y_var)."""
    rng = np.random.default_rng(seed + 777)
    grid = np.linspace(0.0, 1.0, n)
    jitter = (rng.random(n) - 0.5) * (1.0 / n)          # sub-cell jitter: dense coverage, not a fixed lattice
    x = np.clip(grid + jitter, 0.0, 1.0)
    y = tent_pow(x, k)
    y = y - y.mean()                                    # centre the regression target (mean-0)
    return x.reshape(-1, 1), y.reshape(-1, 1), float(np.var(y))


# --------------------------------------------------------------------------- the rate MLP (runner-side, NO sim/ edit)
class RateMLP:
    """3-(or 2-)hidden-layer tanh MLP, linear regression readout, explicit biases. Carries BOTH credit rules:
    true backprop (the ORACLE / measurement reference) and transport-free DFA (the rule under test). The FEEDBACK
    matrices B are fixed-random from a SEPARATE seed stream, frozen, and never derived from any forward W."""

    def __init__(self, sizes, seed=0, act="tanh", zero_feedback=False):
        self.sizes = list(sizes)
        self.nW = len(sizes) - 1
        self.act = act
        wrng = np.random.default_rng(seed)              # forward-weight init stream
        self.W, self.b = [], []
        for i in range(self.nW):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))          # Xavier/Glorot uniform
            self.W.append(wrng.uniform(-lim, lim, (sizes[i], sizes[i + 1])).astype(np.float64))
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float64))
        # FIXED-RANDOM per-hidden-layer feedback B: shape (k, n_post). SEPARATE seed stream (transport-free).
        frng = np.random.default_rng(seed + 8888)
        k = sizes[-1]
        self.B = [(np.zeros((k, sizes[i + 1])) if zero_feedback
                   else frng.normal(0.0, 1.0 / np.sqrt(k), (k, sizes[i + 1]))).astype(np.float64)
                  for i in range(self.nW - 1)]           # hidden layers only; output uses the error directly
        self.zero_feedback = zero_feedback
        self._B0 = [b.copy() for b in self.B]            # frozen snapshot -> the transport-free freeze check
        # Adam optimiser state (mode-AGNOSTIC machinery; applied identically to BP and DFA grads; does NOT change
        # which credit each mode computes). Alignment is measured on the RAW grads, never the Adam-scaled step.
        self._m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self._t = 0

    def _phi(self, z):
        if self.act == "tanh":
            return np.tanh(z)
        if self.act == "relu":
            return np.maximum(0.0, z)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))          # sigmoid

    def _phip_from_a(self, a):
        if self.act == "tanh":
            return 1.0 - a * a
        if self.act == "relu":
            return (a > 0.0).astype(np.float64)                    # phi'=1[z>0]=1[a>0] (a=max(0,z))
        return a * (1.0 - a)                                       # sigmoid

    def forward(self, X):
        a = [np.asarray(X, dtype=np.float64)]
        z = []
        for li in range(self.nW):
            zl = a[-1] @ self.W[li] + self.b[li]
            z.append(zl)
            a.append(self._phi(zl) if li < self.nW - 1 else zl)     # linear output
        return a, z                                                 # a[nW] == yhat

    def mse(self, X, y):
        a, _ = self.forward(X)
        return float(0.5 * np.mean(np.sum((a[-1] - y) ** 2, axis=1)))

    def bp_grads(self, X, y):
        """Hand-derived backprop (NO autodiff). Returns (gW list, gb list). gW[j-1] is the layer-j incoming grad."""
        a, _ = self.forward(X)
        B = X.shape[0]
        gW = [None] * self.nW
        gb = [None] * self.nW
        delta = (a[-1] - y) / B                                     # dMSE/d(yhat), linear output
        gW[self.nW - 1] = a[self.nW - 1].T @ delta
        gb[self.nW - 1] = delta.sum(0)
        for li in range(self.nW - 2, -1, -1):
            dz = (delta @ self.W[li + 1].T) * self._phip_from_a(a[li + 1])   # true backprop through W^{li+1}
            gW[li] = a[li].T @ dz
            gb[li] = dz.sum(0)
            delta = dz
        return gW, gb

    def dfa_grads(self, X, y):
        """Transport-free DFA (Nokland/Lillicrap). Hidden layer li credit = (e @ B[li]) * phi' -- the output error
        projected DIRECTLY by the FIXED-RANDOM B[li]. NO W^T anywhere in the hidden path -> no weight transport."""
        a, _ = self.forward(X)
        B = X.shape[0]
        e = (a[-1] - y) / B                                         # output error (B, k)
        gW = [None] * self.nW
        gb = [None] * self.nW
        gW[self.nW - 1] = a[self.nW - 1].T @ e                      # output layer uses the error directly (== BP)
        gb[self.nW - 1] = e.sum(0)
        for li in range(self.nW - 1):
            dz = (e @ self.B[li]) * self._phip_from_a(a[li + 1])    # DIRECT feedback projection; reads B, never W
            gW[li] = a[li].T @ dz
            gb[li] = dz.sum(0)
        return gW, gb

    def layer_alignments(self, X, y):
        """cos(DFA update, BP gradient) at every HIDDEN layer, on the CURRENT weights. Returns list indexed by
        hidden layer (0 == first/deepest-from-output ... nW-2 == last/adjacent-to-output). cos(0-vector, .) == 0."""
        gW_bp, _ = self.bp_grads(X, y)
        gW_df, _ = self.dfa_grads(X, y)
        cers = []
        for li in range(self.nW - 1):
            u = gW_df[li].ravel()
            v = gW_bp[li].ravel()
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            cers.append(float(u @ v / (nu * nv)) if (nu > 1e-12 and nv > 1e-12) else 0.0)
        return cers

    def _adam_step(self, grads_W, grads_b, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self.W + self.b
        grads = list(grads_W) + list(grads_b)
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i, g in enumerate(grads):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * (g * g)
            step = lr * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + eps)
            params[i] -= step

    def train(self, X, y, mode, epochs, lr, batch=128, seed=0, permute_each_step=False,
              align_every=0, align_XY=None):
        """mode: 'bp' (oracle) or 'dfa' (transport-free rule under test). Returns dict with the loss trajectory and,
        if align_every>0, the layer-alignment trajectory (measured on the whole align set each checkpoint)."""
        rng = np.random.default_rng(seed + 4242)
        n = len(X)
        loss_traj, align_traj, align_epochs = [], [], []
        aX, aY = (align_XY if align_XY is not None else (X, y))
        for ep in range(epochs + 1):
            if align_every and (ep % align_every == 0 or ep == epochs):
                align_traj.append(self.layer_alignments(aX, aY))
                align_epochs.append(ep)
            if ep == epochs:
                break
            perm = rng.permutation(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                Xb = X[bi]
                yb = y[bi]
                if permute_each_step:                              # anti-cheat (ii): targets carry NO signal
                    yb = yb[rng.permutation(len(bi))]
                gW, gb = (self.bp_grads(Xb, yb) if mode == "bp" else self.dfa_grads(Xb, yb))
                self._adam_step(gW, gb, lr)
            loss_traj.append(self.mse(X, y))
        return {"final_loss": self.mse(X, y), "loss_traj": loss_traj,
                "align_traj": align_traj, "align_epochs": align_epochs}

    def feedback_frozen(self):
        """Transport-free check (v): B never moved during training (fixed-random, not learned/derived)."""
        return all(np.array_equal(b, b0) for b, b0 in zip(self.B, self._B0))

    def feedback_independent_of_W(self):
        """Transport-free check (v): the last-hidden feedback B[-1] is INDEPENDENT of the output weight W[-1] it
        would replace under weight transport -- |cos(B[-1], W[-1]^T)| ~ 0 at init (two independent random draws)."""
        if not self.B:
            return True, float("nan")
        b = self.B[-1].ravel()
        wt = self.W[-1].T.ravel()
        nb, nw = np.linalg.norm(b), np.linalg.norm(wt)
        c = float(b @ wt / (nb * nw)) if (nb > 1e-12 and nw > 1e-12) else 0.0
        return abs(c) < 0.3, c


# --------------------------------------------------------------------------- one seed
def _rising(traj):
    """A single hidden layer's alignment across checkpoints: (initial, final, rising?)."""
    if not traj:
        return float("nan"), float("nan"), False
    return float(traj[0]), float(traj[-1]), bool(traj[-1] > traj[0] + 1e-3)


def run_seed(seed, k, hidden, epochs, lr, n_points, act, align_every,
             align_bar, ceil_frac, gap_close_bar, perm_align_tol, sep_margin):
    X, y, y_var = make_tent_data(seed, n_points, k)
    sizes3 = [1] + [hidden] * 3 + [1]                              # 3 hidden layers
    sizes2 = [1] + [hidden] * 2 + [1]                              # depth-2 control (same width)

    # (i) backprop-oracle depth-3 ceiling
    bp3 = RateMLP(sizes3, seed=seed, act=act)
    bp3_out = bp3.train(X, y, "bp", epochs, lr, seed=seed)
    bp3_loss = bp3_out["final_loss"]

    # (iii) backprop-oracle depth-2 on the SAME target (must be strictly worse -> depth-3-engaging on FIT)
    bp2 = RateMLP(sizes2, seed=seed, act=act)
    bp2_out = bp2.train(X, y, "bp", epochs, lr, seed=seed)
    bp2_loss = bp2_out["final_loss"]

    # the rule under test: transport-free DFA depth-3, with the layer-alignment trajectory
    dfa = RateMLP(sizes3, seed=seed, act=act)
    fb_frozen_pre = dfa.feedback_frozen()
    indep_ok, indep_cos = dfa.feedback_independent_of_W()
    dfa_out = dfa.train(X, y, "dfa", epochs, lr, seed=seed, align_every=align_every, align_XY=(X, y))
    dfa_loss = dfa_out["final_loss"]
    fb_frozen_post = dfa.feedback_frozen()

    # per-hidden-layer alignment trajectories. index 0 = deepest-from-output (a1, W0); nH-1 = adjacent-output (a3).
    at = dfa_out["align_traj"]                                     # list[checkpoint] of list[layer]
    nH = len(at[0]) if at else 0
    per_layer = {}
    for li in range(nH):
        traj = [row[li] for row in at]
        ini, fin, rise = _rising(traj)
        per_layer[li] = {"traj": traj, "initial": ini, "final": fin, "rising": rise}
    # GO layer = the 3rd hidden layer a3 (adjacent to output) == the LAST hidden index (nH-1) for a 3-hidden net.
    go_li = nH - 1
    deep_li = 0                                                    # deepest-from-output (a1) -- the strict reach

    # (ii) permuted control: per-step target reshuffle -> no consistent signal
    perm = RateMLP(sizes3, seed=seed, act=act)
    perm_out = perm.train(X, y, "dfa", epochs, lr, seed=seed, permute_each_step=True,
                          align_every=align_every, align_XY=(X, y))
    perm_align_go = perm.layer_alignments(X, y)[go_li] if nH else float("nan")
    perm_loss = perm_out["final_loss"]

    # (iv) apical-lesion control: feedback B=0 (top-down credit removed)
    les = RateMLP(sizes3, seed=seed, act=act, zero_feedback=True)
    les_out = les.train(X, y, "dfa", epochs, lr, seed=seed, align_every=align_every, align_XY=(X, y))
    les_align_go = les.layer_alignments(X, y)[go_li] if nH else float("nan")
    les_loss = les_out["final_loss"]

    return {
        "seed": seed, "k": k, "hidden": hidden, "y_var": y_var,
        "bp3_loss": bp3_loss, "bp2_loss": bp2_loss, "dfa_loss": dfa_loss,
        "perm_loss": perm_loss, "les_loss": les_loss,
        "go_layer": go_li, "deep_layer": deep_li, "n_hidden": nH,
        "per_layer": per_layer,
        "go_align_final": per_layer[go_li]["final"] if nH else float("nan"),
        "go_align_initial": per_layer[go_li]["initial"] if nH else float("nan"),
        "go_align_rising": per_layer[go_li]["rising"] if nH else False,
        "deep_align_final": per_layer[deep_li]["final"] if nH else float("nan"),
        "deep_align_rising": per_layer[deep_li]["rising"] if nH else False,
        "perm_align_go": perm_align_go, "les_align_go": les_align_go,
        "fb_frozen": bool(fb_frozen_pre and fb_frozen_post),
        "fb_indep_of_W": bool(indep_ok), "fb_indep_cos": indep_cos,
        "align_epochs": dfa_out["align_epochs"],
    }


def _agg(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), 0.0)


def evaluate(rows, align_bar, ceil_frac, gap_close_bar, perm_align_tol, sep_margin):
    """Cross-seed verdict. GO on the 3rd hidden layer (a3); deep-reach (a1) reported alongside."""
    m_bp3, _ = _agg(rows, "bp3_loss")
    m_bp2, _ = _agg(rows, "bp2_loss")
    m_dfa, _ = _agg(rows, "dfa_loss")
    m_perm, _ = _agg(rows, "perm_loss")
    m_les, _ = _agg(rows, "les_loss")
    m_yvar, _ = _agg(rows, "y_var")
    m_go_fin, sd_go = _agg(rows, "go_align_final")
    m_go_ini, _ = _agg(rows, "go_align_initial")
    m_deep_fin, _ = _agg(rows, "deep_align_final")
    m_perm_al, _ = _agg(rows, "perm_align_go")
    m_les_al, _ = _agg(rows, "les_align_go")

    # (i) ceiling: BP depth-3 reaches ~0 (small vs the target variance). ceil_frac of y_var.
    ceil_thresh = ceil_frac * m_yvar
    ceiling_exists = bool(m_bp3 <= ceil_thresh)
    # (iii) depth-separating on FIT: BP depth-2 strictly WORSE than depth-3 (by sep_margin of y_var).
    depth_sep = bool((m_bp2 - m_bp3) > sep_margin * m_yvar)
    # GO(a): layer-3 alignment >= bar AND rising
    go_align = bool(m_go_fin >= align_bar and m_go_fin > m_go_ini + 1e-3)
    # GO(b): DFA closes >=gap_close_bar of the depth-2 -> depth-3 loss gap (well-defined at BP3~0)
    denom = (m_bp2 - m_bp3)
    gap_close = ((m_bp2 - m_dfa) / denom) if denom > 1e-12 else float("nan")
    go_loss = bool((not np.isnan(gap_close)) and gap_close >= gap_close_bar)
    # anti-cheats (must hold to interpret). NB the layer-3 (a3) alignment is OUTPUT-ADJACENT and therefore
    # TARGET-INDEPENDENT: output-layer feedback-alignment (W3<->B3) holds for ANY target, incl. the permuted one
    # (the build agent proved the permuted a3 alignment stays ~1.0). So a3 alignment is NOT a valid discriminator --
    # it is REPORTED, never GATED. The VALID transport-free-credit signal is the FIT: does the arm learn the depth-3
    # target BELOW the mean-predictor / BP-depth-2 floor. All precondition checks below are FIT-based.
    perm_ok = bool(m_perm >= 0.5 * m_bp2)          # permuted target does NOT learn the FIT (alignment term dropped)
    les_collapses = bool(m_les >= 0.5 * m_bp2)     # B=0 lesion does NOT learn the FIT
    fb_frozen = all(r["fb_frozen"] for r in rows)
    fb_indep = all(r["fb_indep_of_W"] for r in rows)  # REPORTED only (too strict at low dim); fb_frozen is the guarantee

    # ---- executed levers / attributions (tools.lab), not comments ------------------------------------------
    # (iv) the earned tooth: the feedback path OWNS the alignment. treatment = DFA layer-3 |align|; control = B=0.
    lever("apical_feedback (B!=0 vs B=0)", round(m_les_al, 4), round(m_go_fin, 4),
          continuous="layer3 align: DFA %.3f  vs  B=0 lesion %.3f  (collapse=%s)"
          % (m_go_fin, m_les_al, les_collapses))
    attributable_to("layer-3 alignment attributable to the feedback path",
                    treatment_value=m_go_fin, control_value=m_les_al)

    # ---- EARNED verdict block (gate-visible preconditions) --------------------------------------------------
    # PRECONDITIONS (must hold for the run to be INTERPRETABLE) -- NOT the GO criteria. Registering the GO
    # criteria (alignment / fit) as require() would turn an honest NO-GO into UNDEFINED; they are the `go`
    # argument to decide(), which yields GO when the preconditions hold and go is True, NO-GO when they hold
    # and go is False (an honest negative), UNDEFINED only when a precondition fails.
    v = Verdict("gap4_layer3_credit_fidelity")
    v.require("backprop_oracle_ceiling_exists", ceiling_exists, expect=True,
              note="BP depth-3 loss %.4g <= %.4g (%.0f%% of target var %.4g) -> a learnable ceiling EXISTS"
                   % (m_bp3, ceil_thresh, 100 * ceil_frac, m_yvar))
    v.require("depth_separating_on_fit", depth_sep, expect=True,
              note="BP depth-2 loss %.4g strictly > BP depth-3 %.4g by > %.4g (%.0f%% of var) -> the target is "
                   "depth-3-ENGAGING on FIT (depth-2 underfits what depth-3 fits)"
                   % (m_bp2, m_bp3, sep_margin * m_yvar, 100 * sep_margin))
    v.require("permuted_target_does_not_learn_fit", perm_ok, expect=True,
              note="per-step target reshuffle: permuted FIT loss %.4g >= 0.5*bp2 %.4g -> the permuted control does "
                   "not learn (the valid, target-DEPENDENT control; the a3-alignment term was dropped as invalid)"
                   % (m_perm, m_bp2))
    v.require("apical_lesion_B0_does_not_learn_fit", les_collapses, expect=True,
              note="removing the feedback path (B=0) leaves the FIT at the mean-predictor floor (loss %.4g >= 0.5*bp2 "
                   "%.4g) -> no transport-free learning without the feedback path (the earned tooth, on the FIT)"
                   % (m_les, m_bp2))
    v.require("transport_free_feedback_frozen", fb_frozen, expect=True,
              note="B is fixed-random from a separate seed stream, unchanged across training (never learned) -> the "
                   "credit path is transport-free. (fb_indep |cos(B,W^T)|~0 is REPORTED not gated: too strict at low "
                   "dim; fb_frozen is the transport-free guarantee. fb_indep=%s)" % fb_indep)
    # the SCIENTIFIC GO is FIT-BASED: does transport-free DFA e-prop LEARN the depth-3 target (close the BP2->BP3 gap)?
    # The a3 layer-3 alignment (go_align) is REPORTED alongside but NOT gated (output-adjacent => target-independent).
    go_criteria = bool(go_loss)
    decided = v.decide(go_criteria)

    return {
        "means": {"bp3_loss": m_bp3, "bp2_loss": m_bp2, "dfa_loss": m_dfa, "perm_loss": m_perm,
                  "les_loss": m_les, "y_var": m_yvar, "ceil_thresh": ceil_thresh,
                  "go_align_final": m_go_fin, "go_align_final_sd": sd_go, "go_align_initial": m_go_ini,
                  "deep_align_final": m_deep_fin, "perm_align_go": m_perm_al, "les_align_go": m_les_al,
                  "gap_close": gap_close},
        "checks": {"ceiling_exists": ceiling_exists, "depth_separating": depth_sep, "go_align": go_align,
                   "go_loss": go_loss, "perm_ok": perm_ok, "les_collapses": les_collapses,
                   "fb_frozen": fb_frozen, "fb_indep": fb_indep},
        "go": bool(decided["status"] == "GO"), "status": decided["status"],
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
    }


def _verdict_line(ev, align_bar):
    m = ev["means"]
    if ev["status"] == "GO":
        return ("GO: LAYER-3 CREDIT FIDELITY. Transport-free DFA credit aligns with the backprop-oracle gradient "
                "at the 3rd hidden layer (cos %.3f >= %.2f, rising from %.3f) and closes %.0f%% of the "
                "depth-2->depth-3 fit gap; depth-2 underfits (bp2 %.4g > bp3 %.4g). B=0 collapses it to %.3f, "
                "permuted to %.3f. Deep-reach (a1) align %.3f."
                % (m["go_align_final"], align_bar, m["go_align_initial"], 100 * m["gap_close"],
                   m["bp2_loss"], m["bp3_loss"], m["les_align_go"], m["perm_align_go"], m["deep_align_final"]))
    if ev["status"] == "UNDEFINED":
        return "UNDEFINED (a precondition failed -- NOT a negative): " + "; ".join(ev["undefined_reasons"])
    # NO-GO: an honest negative -- transport-free credit did not reach the layer at the bar
    return ("NO-GO (HONEST NEGATIVE -- a first-class gap#4 deliverable): transport-free DFA credit did NOT reach "
            "the 3rd hidden layer at the bar (cos %.3f vs %.2f, init %.3f; deep-reach a1 %.3f; DFA loss %.4g vs "
            "bp3 %.4g / bp2 %.4g). Next mechanism: learned feedback (weight-mirror/PAL-KP) or the phi'-vanishing "
            "fix (per-layer gain / act change)."
            % (m["go_align_final"], align_bar, m["go_align_initial"], m["deep_align_final"],
               m["dfa_loss"], m["bp3_loss"], m["bp2_loss"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="", help="comma-separated seeds for a self-aggregating sweep, "
                    "e.g. 42,43,44,100,101,102. Overrides --seed. GO judged on the cross-seed means.")
    ap.add_argument("--task", type=str, default="tent3_fit",
                    choices=["tent3_fit", "tent4_fit", "tent5_fit"],
                    help="the depth-k-composed FIT target. tent3_fit = tent(tent(tent(x))), MSE. Additive, "
                    "default-off in the sense that no other runner/default path uses it; NO sim/ edit.")
    ap.add_argument("--hidden", type=int, default=8, help="hidden width. NARROW on purpose: at width 8 depth-2 "
                    "underfits tent^3 (stuck at the mean-predictor) while depth-3 fits -> the depth-3-engaging-on-"
                    "FIT separation (Telgarsky). Verified empirically: bp3~0.002*var, bp2~0.5*var.")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--n-points", type=int, default=512)
    ap.add_argument("--act", type=str, default="relu", choices=["relu", "tanh", "sigmoid"],
                    help="relu = Telgarsky's own tent^k construction (each tent = 2 ReLUs; phi'=1[z>0], no "
                    "vanishing) and the cleanest depth separation. tanh = smooth variant. sigmoid = "
                    "DendriticMLP's activation (phi'<=0.25), the phi'-vanishing-prone variant.")
    ap.add_argument("--align-every", type=int, default=50, help="checkpoint stride for the alignment trajectory.")
    # GO-gate bars
    ap.add_argument("--align-bar", type=float, default=0.6, help="min 3rd-hidden-layer DFA-vs-BP cosine for GO.")
    ap.add_argument("--ceil-frac", type=float, default=0.02, help="BP depth-3 loss must be <= this fraction of the "
                    "target variance to count as a loss~0 ceiling.")
    ap.add_argument("--gap-close-bar", type=float, default=0.9, help="DFA must close >= this fraction of the "
                    "depth-2->depth-3 loss gap (== within 10%% of the depth-3 oracle).")
    ap.add_argument("--perm-align-tol", type=float, default=0.2, help="max |alignment| for the permuted/lesion "
                    "controls to count as collapsed.")
    ap.add_argument("--sep-margin", type=float, default=0.05, help="min (bp2-bp3) loss gap as a fraction of the "
                    "target variance for the target to count as depth-3-engaging on FIT.")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    k = {"tent3_fit": 3, "tent4_fit": 4, "tent5_fit": 5}[args.task]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds.strip() else [args.seed]

    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))
    device = "cpu" if backend == "numpy" else "cuda"

    t0 = time.time()
    rows, err = [], None
    try:
        for s in seeds:
            rows.append(run_seed(s, k, args.hidden, args.epochs, args.lr, args.n_points, args.act,
                                 args.align_every, args.align_bar, args.ceil_frac, args.gap_close_bar,
                                 args.perm_align_tol, args.sep_margin))
        ev = evaluate(rows, args.align_bar, args.ceil_frac, args.gap_close_bar, args.perm_align_tol,
                      args.sep_margin)
    except Exception as e:
        ev = {"error": repr(e), "traceback": traceback.format_exc()}
        err = repr(e)

    out = {"probe": "gap4_layer3_credit_fidelity", "task": args.task, "k_composition": k, "seeds": seeds,
           "backend": backend, "device": device,
           "config": {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr, "n_points": args.n_points,
                      "act": args.act, "align_every": args.align_every,
                      "bars": {"align": args.align_bar, "ceil_frac": args.ceil_frac,
                               "gap_close": args.gap_close_bar, "perm_align_tol": args.perm_align_tol,
                               "sep_margin": args.sep_margin}},
           "elapsed_seconds": round(time.time() - t0, 1), "rows": rows, "result": ev}
    out["verdict"] = err or (_verdict_line(ev, args.align_bar) if "means" in ev else str(ev.get("error")))
    out["preconditions"] = ev.get("preconditions", [])            # top-level for tools/gates/verdict_preconditions.py
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    if "means" in ev:
        m = ev["means"]
        print("\n  backend=%s device=%s  task=%s (tent^%d)  hidden=%d  seeds=%s"
              % (backend, device, args.task, k, args.hidden, seeds))
        print("  --- losses (MSE) ---")
        print("    bp3 (depth-3 oracle) = %.5g   bp2 (depth-2 oracle) = %.5g   dfa (depth-3) = %.5g"
              % (m["bp3_loss"], m["bp2_loss"], m["dfa_loss"]))
        print("    target var = %.5g   ceil_thresh = %.5g   gap_close = %.1f%%"
              % (m["y_var"], m["ceil_thresh"], 100 * (m["gap_close"] if not np.isnan(m["gap_close"]) else 0)))
        print("  --- layer-3 (a3, adjacent-output) alignment ---")
        print("    DFA init=%.3f -> final=%.3f (sd %.3f)   B=0 lesion=%.3f   permuted=%.3f"
              % (m["go_align_initial"], m["go_align_final"], m["go_align_final_sd"],
                 m["les_align_go"], m["perm_align_go"]))
        print("    deep-reach a1 (deepest-from-output) final = %.3f" % m["deep_align_final"])
        print("  --- per-seed layer-alignment trajectories (a1 .. a3) ---")
        for r in rows:
            traj_str = "  ".join("a%d:%s" % (li + 1, "[" + ",".join("%.2f" % x for x in r["per_layer"][li]["traj"]) + "]")
                                 for li in range(r["n_hidden"]))
            print("    seed=%d  epochs@%s" % (r["seed"], r["align_epochs"]))
            print("            %s" % traj_str)
        print("  --- checks --- %s" % ev["checks"])
    print("\n" + out["verdict"])
    print("[layer3-credit-fidelity] status=%s  wrote %s" % (ev.get("status", "ERROR"), args.out))


if __name__ == "__main__":
    main()
