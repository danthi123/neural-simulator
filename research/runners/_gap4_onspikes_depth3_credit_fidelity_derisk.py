"""gap#4 deep-credit ON SPIKES -- LAYER-3 CREDIT FIDELITY on the LIF SNN: does TRANSPORT-FREE KP-learned feedback
reach GENUINE DEPTH-3 credit on the SPIKING substrate, where FIXED-random DFA fails -- the depth-3 analog of the
depth-2 on-spikes GO?

THE PORT (2026-08-11). The depth-2 on-spikes GO (`_gap4_onspikes_kp_align_derisk`, finding 2026-08-11) showed
transport-free KP learned feedback ALIGNS on the LIF SNN and beats fixed-DFA -- but only at DEPTH-2. The RATE side
then established (finding 2026-08-11 depth3-obligatory-task-is-provably-impossible) that a depth-3-OBLIGATORY *task*
(by held-out ACCURACY) is provably impossible at toy scale (Telgarsky: depth-2 lower bounds need width EXPONENTIAL in
the depth-gap=1), so the achievable test is LAYER-3 CREDIT FIDELITY on a depth-3-composed FIT target (tent^3
regression). The RATE learned-feedback de-risk (`_gap4_learned_feedback_derisk`) ran that FIT instrument and, with
PER-SEED ceiling-gating, showed KP-learned feedback reaches the 3rd hidden layer where fixed-DFA cannot. This runner
PORTS that FIT instrument onto the LIF SNN forward + surrogate-BPTT ceiling -- the one place the depth-3 question was
never posed on the ONE spiking substrate.

WHY tent^k (Telgarsky's separating family). tent(x)=1-|2x-1| on [0,1] has 2 linear pieces; tent^k composed k times has
2^k pieces. Depth composes pieces MULTIPLICATIVELY (each layer ~doubles folds) while width adds them LINEARLY -- so at
a NARROW width a depth-3 net fits tent^3 where a depth-2 net cannot (the one place depth-2 genuinely can't compete on
FIT). This is a REGRESSION (MSE) fit of a 1-D map -> the LIF net reads out a CONTINUOUS scalar via a linear population
read-out of the last hidden layer's summed spikes (rate decoding -- the standard spiking-regression read-out, a
legitimate motor-style read-out of population activity), NOT a spike-count argmax.

THE NET (input 1 feature rate-coded over T -> nH LIF hidden layers -> LINEAR read-out of the top layer's summed
spikes). Hidden layers are LIF (sim.bptt_snn_gpu.LIFLayerXP, reuse-by-import); the read-out Wout,bout is a linear
population decoder (H->1). Depth-3 net = 3 LIF hidden layers; depth-2 control = 2 (same width). NO sim/ edit -- the
LIF forward/BPTT/atan surrogate are reuse-by-import; the tent^k data builder, the linear read-out, and the four credit
rules are RUNNER-side (the sibling on-spikes + rate de-risks implement their credit rules runner-side identically).

THE FOUR CREDIT RULES (one net class; the forward + Adam machinery is mode-AGNOSTIC -- only the CREDIT differs).
  * bp        : surrogate-gradient BPTT ceiling (reuse backward_unroll_xp -- uses W^T + the read-out Wout^T; the
                NON-LOCAL reference the transport-free rules may NOT use). bp at depth-3 = the ceiling; bp at depth-2 =
                the depth-separation control (must UNDERFIT what depth-3 fits).
  * dfa       : DIRECT fixed-random feedback Bdfa[li] (shape (1,H)) projects the SCALAR output error DIRECTLY to each
                hidden layer (Nokland 2016 DFA). The prior banked baseline that FAILS to reach deep layers -- the
                thing KP must beat.
  * seq_fixed : CHAINED (sequential) transport-free feedback Y_fb[li] replaces the forward weight^T in a sequential
                backward pass, FROZEN at random init == the FREEZE-Y LEVER ENDPOINT.
  * seq_kp    : CHAINED transport-free KP-learned feedback -- each Y_fb[li] receives the SAME Adam step as its forward
                weight, TRANSPOSED (Kolen-Pollack 1994 / Akrout 2019), so Y_fb co-adapts toward W^T via a LOCAL matched
                rule. STILL TRANSPORT-FREE: Y_fb is a separate random stream, NEVER set to W^T; the credit path
                computes e @ Y_fb (never a forward W^T); cos(Y_fb, W^T) RISES from ~0 through training (emerges, not
                copied). The rule under test.

THE GO GATE (FIT-based; per-seed ceiling-gated). A seed is TESTABLE only if BP-depth-3 FITS (loss <= ceil_frac*var)
AND BP-depth-2 UNDERFITS (gap > sep_margin*var) -- the rate finding showed this LIF/tent^k ceiling is SEED-FRAGILE, so
gating + N_testable is MANDATORY (a seed where the surrogate-BPTT ORACLE itself cannot fit carries NO information about
whether a transport-free rule can). On the testable seeds:
  GO  <=>  KP closes >= gap_close_bar of the BP-depth-2 -> BP-depth-3 FIT gap (reaches the depth-3 oracle)
           AND KP beats the fixed-DFA baseline (KP's fit differs from fixed-DFA's -- the manipulation landed).
An honest NO-GO (even KP does not reach depth-3 on spikes) OR an honest UNDEFINED (no seed yields a depth-3-separating
LIF ceiling -> no valid spiking depth-3 instrument at this scale) is a FIRST-CLASS gap#4 deliverable -- report which,
NEVER fabricate a negative from an unfit ceiling.

ANTI-CHEATS (all EXECUTE via tools.lab / Verdict, none is a comment):
  (i)   ceiling EXISTS per seed (surrogate-BPTT depth-3 fits) -- else that seed is not testable (UNDEFINED contribution).
  (ii)  depth-separating per seed (BPTT depth-2 underfits) -- else not testable.
  (iii) fixed-DFA baseline FAILS to close the gap (REPORTED + a control() that KP's fit DIFFERS from fixed-DFA's).
  (iv)  permuted-target KP -> no fit (per-step target reshuffle; targets carry no signal -> loss at the mean floor).
  (v)   FREEZE-Y LEVER: seq_kp (feedback-learning ON) moves Y (LeverError if not); seq_fixed (OFF) leaves Y frozen ->
        collapses to the fixed-feedback floor. Freezing == KP with the feedback update disabled.
  (vi)  TRANSPORT-FREE: Y_fb is a SEPARATE random stream; init |cos(Y_fb, W^T)| < 0.8 (not a copy); no Y_fb is
        byte-equal any forward W or its transpose; the credit path computes e @ Y_fb, NEVER a forward W^T; KP updates
        Y_fb by the matched Adam-step transpose (activity-derived), NOT by reading W. cos(Y_fb,W^T) RISES under KP.
  (vii) DETERMINISM: build the net twice at one seed -> byte-identical forward weights (the substrate RNG is the
        runner's default_rng(seed); this runner does NOT touch the bridge, so there is no cfg.seed -- the seeding is
        the explicit default_rng(seed) streams, verified by hash).

Sources: Telgarsky 2016 (tent^k depth-separation, the impossibility that forces the FIT reframe); Neftci-Mostafa-Zenke
2019 (surrogate-gradient BPTT for deep SNNs -- the spiking ceiling); Bellec 2020 (e-prop forward eligibility);
Lillicrap 2016 / Nokland 2016 (feedback alignment / DFA, the fixed-feedback deep-layer limit); Kolen-Pollack 1994 /
Akrout 2019 (KP learned feedback, transport-free). NO sim/ edit; additive; default-off task. SIM_BACKEND=numpy (tiny
matmuls -> CPU; GPU launch overhead dominates at this scale).
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# reuse-by-import: the BPTT-viable LIF SNN forward + surrogate BPTT + atan surrogate (NO sim/ edit)
from sim.bptt_snn_gpu import LIFLayerXP, forward_unroll_xp, backward_unroll_xp, atan_surrogate  # noqa: E402

from tools.lab import lever, attributable_to, assert_backend, LeverError  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap4_onspikes_depth3_credit_fidelity.json"


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
    """n points on [0,1] (dense grid + seeded sub-cell jitter so the fit represents the FUNCTION, not isolated
    points), target = tent^k(x), centred to mean 0. Returns (X (n,1), y (n,1), y_var)."""
    rng = np.random.default_rng(seed + 777)
    grid = np.linspace(0.0, 1.0, n)
    jitter = (rng.random(n) - 0.5) * (1.0 / n)
    x = np.clip(grid + jitter, 0.0, 1.0)
    y = tent_pow(x, k)
    y = y - y.mean()
    return x.reshape(-1, 1), y.reshape(-1, 1), float(np.var(y))


# --------------------------------------------------------------------------- the LIF regression net (runner-side)
class LIFRegNet:
    """nH-hidden-layer LIF SNN with a LINEAR population read-out (rate decoding of the top hidden layer's summed
    spikes). Carries FOUR credit rules sharing ONE forward pass + ONE (mode-agnostic) Adam optimiser; only the CREDIT
    differs. Feedback for the transport-free rules is a SEPARATE random stream, never derived from any forward W.

    LAYER INDEXING. layers[0]: (1->H); layers[i]: (H->H) for i>=1; read-out Wout: (H->1), bout: (1,). "hidden layer j"
    = the j-th LIF layer's spikes; the TOP hidden layer (index nH-1, adjacent to the read-out) is the layer the credit
    reaches first; the DEEPEST-from-output (index 0, credit must traverse nH hops) is the strict reach.

    Transport-free chained feedback Y_fb (one per hidden layer):
      Y_fb[nH-1] : (1, H)  replaces Wout^T          -- projects the scalar read-out error into the top hidden layer.
      Y_fb[i]    : (H, H)  replaces layers[i+1].W_in^T  (i in 0..nH-2) -- the sequential hidden->hidden feedback.
    Direct DFA feedback Bdfa[i] : (1, H) projects the scalar output error DIRECTLY to hidden layer i (Nokland)."""

    def __init__(self, hidden, nH, seed=0, n_in=2, w_scale0=2.5, w_scale=1.0, threshold=1.0, leak=0.9):
        self.H = hidden
        self.nH = nH
        wrng = np.random.default_rng(seed)                          # forward-weight init stream
        sizes = [n_in] + [hidden] * nH                              # n_in=2 with a constant bias-input channel, else 1
        self.layers = []
        for i in range(nH):
            n_pre, n_post = sizes[i], sizes[i + 1]
            scale = w_scale0 if i == 0 else w_scale                 # strong first layer (fire from continuous input)
            W = wrng.normal(0.0, scale / np.sqrt(n_pre), (n_pre, n_post)).astype(np.float64)
            self.layers.append(LIFLayerXP(W_in=W, n_post=n_post, threshold=threshold, leak=leak))
        lim = np.sqrt(6.0 / (hidden + 1))
        self.Wout = wrng.uniform(-lim, lim, (hidden, 1)).astype(np.float64)     # linear read-out
        self.bout = np.zeros(1, dtype=np.float64)
        self.leak = leak

        # CHAINED transport-free feedback Y_fb (SEPARATE stream). Y_fb[nH-1] replaces Wout^T (1,H); Y_fb[i] (i<nH-1)
        # replaces layers[i+1].W_in^T (H,H). Never derived from any forward W -> transport-free.
        frng = np.random.default_rng(seed + 8888)
        self.Y_fb = []
        for i in range(nH):
            if i == nH - 1:
                shape = (1, hidden)                                 # replaces Wout^T
            else:
                shape = (hidden, hidden)                            # replaces layers[i+1].W_in^T (n_post_{i+1}, n_pre_{i+1})
            self.Y_fb.append((frng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float64))
        self._Yfb0 = [Y.copy() for Y in self.Y_fb]

        # DIRECT fixed-random DFA feedback (SEPARATE stream), one (1,H) per hidden layer.
        drng = np.random.default_rng(seed + 9999)
        self.Bdfa = [(drng.standard_normal((1, hidden))).astype(np.float64) for _ in range(nH)]

        # Adam state (mode-AGNOSTIC). params order: [layers W_in...] + [Wout, bout].
        self._params_shapes = [l.W_in.shape for l in self.layers] + [self.Wout.shape, self.bout.shape]
        self._m = [np.zeros(s) for s in self._params_shapes]
        self._v = [np.zeros(s) for s in self._params_shapes]
        self._t = 0

    # ---- forward -----------------------------------------------------------------------------------------
    def forward(self, X, T, in_gain):
        inp = np.repeat((in_gain * np.asarray(X, dtype=np.float64))[None, :, :], T, axis=0)   # (T,B,n_in)
        fs = forward_unroll_xp(inp, self.layers, xp=np)
        r_top = fs["spikes"][-1].sum(axis=0) / T                    # (B,H) MEAN firing rate (spikes/T in [0,1]) -- the
        #        rate-normalized population read-out. Dividing by T keeps the read-out input O(1) instead of O(T), which
        #        stops the read-out backprop (e@Wout^T) from exploding as Wout grows -> the fix for the LIF training
        #        divergence the raw summed-spike read-out caused.
        yhat = r_top @ self.Wout + self.bout                       # (B, 1) linear read-out
        return inp, fs, r_top, yhat

    def mse(self, X, y, T, in_gain):
        _, _, _, yhat = self.forward(X, T, in_gain)
        return float(0.5 * np.mean(np.sum((yhat - y) ** 2, axis=1)))

    # ---- credit rules ------------------------------------------------------------------------------------
    def _psi(self, fs, li, t, alpha_surr, sigma_norm):
        p = atan_surrogate(fs["v"][li][t] - self.layers[li].threshold, alpha=alpha_surr, xp=np)
        if sigma_norm:
            p = p / (p.mean() + 1e-9)
        return p

    def _readout_grads(self, r_top, yhat, y):
        B = r_top.shape[0]
        e_out = (yhat - y) / B                                      # (B,1) dL/dyhat
        gWout = r_top.T @ e_out                                     # (H,1) exact read-out gradient (forward direction)
        gbout = e_out.sum(0)                                        # (1,)
        return e_out, gWout, gbout

    def bp_grads(self, inp, fs, r_top, yhat, y, alpha_surr=2.0):
        """BPTT ORACLE (uses W^T + Wout^T -- forbidden to the transport-free rules). Returns (gW_layers, gWout, gbout)."""
        e_out, gWout, gbout = self._readout_grads(r_top, yhat, y)
        T = inp.shape[0]
        ds_top = e_out @ self.Wout.T                               # (B,H) dL/ds_top_t (== dL/dr_top for every t)
        og = np.repeat(ds_top[None, :, :], T, axis=0).astype(np.float64)     # (T,B,H)
        gW_layers, _ = backward_unroll_xp(inp, self.layers, fs, og, alpha=alpha_surr, xp=np)
        return gW_layers, gWout, gbout

    def dfa_grads(self, inp, fs, r_top, yhat, y, alpha_surr=2.0, sigma_norm=True):
        """DIRECT fixed-random DFA (Nokland). Hidden li credit = (e_out @ Bdfa[li]) * psi -- the scalar output error
        projected DIRECTLY by the fixed-random Bdfa[li]. NO W^T anywhere -> transport-free (the fails-at-depth baseline)."""
        e_out, gWout, gbout = self._readout_grads(r_top, yhat, y)
        T, B, _ = inp.shape
        spikes = fs["spikes"]
        gW = [np.zeros_like(l.W_in) for l in self.layers]
        eps = [np.zeros((B, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in self.layers]
        for t in range(T):
            for li in range(self.nH):
                pre = inp[t] if li == 0 else spikes[li - 1][t]
                eps[li] = self.leak * eps[li] + pre
            for li in range(self.nH):
                Lsig = e_out @ self.Bdfa[li]                       # (B,H) direct projection of the scalar error
                gW[li] += eps[li].T @ (Lsig * self._psi(fs, li, t, alpha_surr, sigma_norm))
        return gW, gWout, gbout

    def seq_grads(self, inp, fs, r_top, yhat, y, alpha_surr=2.0, sigma_norm=True):
        """CHAINED (sequential) transport-free feedback: Y_fb[li] replaces the forward weight^T in a sequential
        backward pass. Identical form for fixed FA and KP -- the difference is only whether Y_fb is updated (KP, via
        _kp_update) or frozen (fixed). Reads Y_fb, NEVER a forward W^T. Returns (gW_layers, gWout, gbout)."""
        e_out, gWout, gbout = self._readout_grads(r_top, yhat, y)
        T, B, _ = inp.shape
        spikes = fs["spikes"]
        top = self.nH - 1
        gW = [np.zeros_like(l.W_in) for l in self.layers]
        eps = [np.zeros((B, l.W_in.shape[0]), dtype=l.W_in.dtype) for l in self.layers]
        for t in range(T):
            for li in range(self.nH):
                pre = inp[t] if li == 0 else spikes[li - 1][t]
                eps[li] = self.leak * eps[li] + pre
            # top hidden layer receives the read-out error via Y_fb[top] (replaces Wout^T), transport-free
            e_top = (e_out @ self.Y_fb[top]) * self._psi(fs, top, t, alpha_surr, sigma_norm)   # (B,H)
            gW[top] += eps[top].T @ e_top
            e_above = e_top
            for li in range(top - 1, -1, -1):
                e_below = (e_above @ self.Y_fb[li]) * self._psi(fs, li, t, alpha_surr, sigma_norm)
                gW[li] += eps[li].T @ e_below
                e_above = e_below
        return gW, gWout, gbout

    # ---- optimiser ---------------------------------------------------------------------------------------
    def _adam_apply(self, grads, lr, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        """Apply Adam to all forward params (layers W_in..., Wout, bout) IN PLACE. Returns the per-param STEPS so the
        KP rule can apply the SAME step (transposed) to the matched feedback -- this is the 'same delta'. Global-norm
        gradient clipping (clip>0) stabilizes the spiking surrogate updates (they diverge unclipped on the LIF net)."""
        if clip > 0:
            gn = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if gn > clip:
                grads = [g * (clip / (gn + 1e-12)) for g in grads]
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        steps = [None] * len(grads)
        for i, g in enumerate(grads):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * (g * g)
            step = lr * (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + eps)
            steps[i] = step
            if i < self.nH:
                self.layers[i].W_in = self.layers[i].W_in - step
            elif i == self.nH:
                self.Wout = self.Wout - step
            else:
                self.bout = self.bout - step
        return steps

    def _kp_update(self, steps):
        """Kolen-Pollack: each Y_fb receives the SAME Adam step as its matched forward weight, TRANSPOSED -> Y_fb and
        W^T accumulate identical increments from independent random inits, so (Y_fb - W^T) stays at its init value while
        the matched increments dominate => cos(Y_fb, W^T) -> 1 EMERGES. NEVER reads W (uses steps, an activity-derived
        quantity). Y_fb[i] (i<nH-1) matched to layers[i+1].W_in (step index i+1); Y_fb[nH-1] matched to Wout (index nH)."""
        for i in range(self.nH):
            src = (self.nH if i == self.nH - 1 else i + 1)         # step index of the matched forward weight
            self.Y_fb[i] = self.Y_fb[i] - steps[src].T

    # ---- read-outs ---------------------------------------------------------------------------------------
    def bw_alignments(self):
        """cos(Y_fb[i], matched W^T) per hidden layer -- the transport-free signature (~0 at init, RISES under KP).
        index nH-1 = top (Y_fb vs Wout^T); index i<nH-1 = Y_fb vs layers[i+1].W_in^T."""
        out = []
        for i in range(self.nH):
            wt = (self.Wout.T if i == self.nH - 1 else self.layers[i + 1].W_in.T).ravel()
            yy = self.Y_fb[i].ravel()
            ny, nw = np.linalg.norm(yy), np.linalg.norm(wt)
            out.append(float(yy @ wt / (ny * nw)) if (ny > 1e-12 and nw > 1e-12) else 0.0)
        return out

    def bw_indep_at_init_max(self):
        return float(max(abs(c) for c in self.bw_alignments())) if self.nH else float("nan")

    def feedback_moved(self):
        return any(not np.array_equal(self.Y_fb[i], self._Yfb0[i]) for i in range(self.nH))

    def feedback_frozen(self):
        return all(np.array_equal(self.Y_fb[i], self._Yfb0[i]) for i in range(self.nH))

    def no_weight_transport(self):
        """anti-cheat: no Y_fb is byte-equal any forward W or its transpose (the 'Y is secretly W^T' cheat)."""
        Ws = [l.W_in for l in self.layers] + [self.Wout]
        for Y in self.Y_fb:
            for W in Ws:
                if Y.shape == W.shape and np.array_equal(Y, W):
                    return False
                if Y.shape == W.T.shape and np.array_equal(Y, W.T):
                    return False
        return True

    # ---- training ----------------------------------------------------------------------------------------
    def train(self, X, y, mode, T, epochs, lr, in_gain, batch=64, seed=0, alpha_surr=2.0, sigma_norm=True,
              permute_each_step=False, align_every=0, clip=1.0, lr_decay=True, loss_every=25):
        """mode in {bp, dfa, seq_fixed, seq_kp}. seq_kp learns Y_fb by KP; seq_fixed freezes Y_fb (the lever).
        Cosine lr decay + global-norm clip stabilize the LIF training; `best_loss` = min MSE over training (the
        ACHIEVABLE-fit ceiling, applied IDENTICALLY to every arm -> the fair fit-quality read-out that removes the
        substrate's late-training instability uniformly)."""
        rng = np.random.default_rng(seed + 4242)
        n = len(X)
        loss_traj, bw_traj, bw_epochs = [], [], []
        best_loss = float("inf")
        for ep in range(epochs + 1):
            if align_every and mode in ("seq_fixed", "seq_kp") and (ep % align_every == 0 or ep == epochs):
                bw_traj.append(self.bw_alignments())
                bw_epochs.append(ep)
            if ep == epochs:
                break
            cur_lr = lr * (0.5 * (1.0 + np.cos(np.pi * ep / epochs))) if lr_decay else lr   # cosine lr decay
            perm = rng.permutation(n)
            for b0 in range(0, n, batch):
                bi = perm[b0:b0 + batch]
                Xb, yb = X[bi], y[bi]
                if permute_each_step:
                    yb = yb[rng.permutation(len(bi))]              # anti-cheat: targets carry NO signal
                inp, fs, r_top, yhat = self.forward(Xb, T, in_gain)
                if mode == "bp":
                    gW, gWout, gbout = self.bp_grads(inp, fs, r_top, yhat, yb, alpha_surr)
                elif mode == "dfa":
                    gW, gWout, gbout = self.dfa_grads(inp, fs, r_top, yhat, yb, alpha_surr, sigma_norm)
                else:                                              # seq_fixed / seq_kp
                    gW, gWout, gbout = self.seq_grads(inp, fs, r_top, yhat, yb, alpha_surr, sigma_norm)
                steps = self._adam_apply(list(gW) + [gWout, gbout], cur_lr, clip)
                if mode == "seq_kp":
                    self._kp_update(steps)
            if ep % loss_every == 0 or ep == epochs - 1:
                L = self.mse(X, y, T, in_gain)
                loss_traj.append(L)
                best_loss = min(best_loss, L)
        final = self.mse(X, y, T, in_gain)
        best_loss = min(best_loss, final)
        return {"final_loss": final, "best_loss": best_loss, "loss_traj": loss_traj,
                "bw_traj": bw_traj, "bw_epochs": bw_epochs}


# --------------------------------------------------------------------------- one seed
def _ends(traj, li):
    if not traj:
        return float("nan"), float("nan")
    return float(traj[0][li]), float(traj[-1][li])


def _weights_hash(net):
    h = hashlib.sha256()
    for l in net.layers:
        h.update(np.ascontiguousarray(l.W_in).tobytes())
    h.update(np.ascontiguousarray(net.Wout).tobytes())
    return h.hexdigest()[:16]


def run_seed(seed, k, hidden, T, epochs, lr, in_gain, n_points, batch, w_scale0, leak, bias_input,
             clip, align_every):
    X, y, y_var = make_tent_data(seed, n_points, k)
    if bias_input:                                                 # constant bias-input channel -> layer-0 biases
        X = np.concatenate([X, np.ones_like(X)], axis=1)
    n_in = X.shape[1]
    mean_pred_loss = 0.5 * y_var                                   # yhat=0 on the mean-0 target

    # (vii) determinism: two fresh builds at one seed -> identical forward weights (substrate RNG = default_rng(seed))
    det_ok = bool(_weights_hash(LIFRegNet(hidden, 3, seed=seed, n_in=n_in, w_scale0=w_scale0, leak=leak))
                  == _weights_hash(LIFRegNet(hidden, 3, seed=seed, n_in=n_in, w_scale0=w_scale0, leak=leak)))

    def mk(nH):
        return LIFRegNet(hidden, nH, seed=seed, n_in=n_in, w_scale0=w_scale0, leak=leak)

    # NB every arm's reported fit is the BEST loss over training (the achievable-fit ceiling; identical across arms,
    # removes the LIF substrate's late-training instability uniformly).
    # (i) surrogate-BPTT depth-3 ceiling
    bp3 = mk(3)
    bp3_loss = bp3.train(X, y, "bp", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip)["best_loss"]
    # (ii) surrogate-BPTT depth-2 (depth-separation control, same width)
    bp2 = mk(2)
    bp2_loss = bp2.train(X, y, "bp", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip)["best_loss"]

    # the rule under test: chained transport-free KP-learned feedback (depth-3)
    kp = mk(3)
    kp_bw_indep_init = kp.bw_indep_at_init_max()
    kp_out = kp.train(X, y, "seq_kp", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip,
                      align_every=align_every)
    kp_loss = kp_out["best_loss"]
    kp_moved = kp.feedback_moved()
    kp_no_transport = kp.no_weight_transport()

    # FREEZE-Y LEVER endpoint: chained fixed feedback (Y frozen) -- the freeze-Y collapse control
    fa = mk(3)
    fa_out = fa.train(X, y, "seq_fixed", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip,
                      align_every=align_every)
    fa_loss = fa_out["best_loss"]
    fa_frozen = fa.feedback_frozen()

    # the prior banked baseline: DIRECT fixed-random DFA (the thing KP must beat)
    dfa = mk(3)
    dfa_loss = dfa.train(X, y, "dfa", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip)["best_loss"]

    # (iv) anti-cheat: KP with per-step permuted targets -> no fit
    perm = mk(3)
    perm_loss = perm.train(X, y, "seq_kp", T, epochs, lr, in_gain, batch=batch, seed=seed, clip=clip,
                           permute_each_step=True)["best_loss"]

    nH = 3
    deep_li, top_li = 0, nH - 1
    kp_bw_deep_init, kp_bw_deep_fin = _ends(kp_out["bw_traj"], deep_li)
    kp_bw_top_init, kp_bw_top_fin = _ends(kp_out["bw_traj"], top_li)

    return {
        "seed": seed, "k": k, "hidden": hidden, "y_var": y_var, "mean_pred_loss": mean_pred_loss,
        "bp3_loss": bp3_loss, "bp2_loss": bp2_loss, "kp_loss": kp_loss, "fa_loss": fa_loss,
        "dfa_loss": dfa_loss, "perm_loss": perm_loss,
        "kp_moved": bool(kp_moved), "fa_frozen": bool(fa_frozen), "kp_no_transport": bool(kp_no_transport),
        "kp_bw_indep_init_max": kp_bw_indep_init,
        "kp_bw_deep_init": kp_bw_deep_init, "kp_bw_deep_fin": kp_bw_deep_fin,
        "kp_bw_top_init": kp_bw_top_init, "kp_bw_top_fin": kp_bw_top_fin,
        "bw_epochs": kp_out["bw_epochs"], "determinism_ok": det_ok,
    }


def _gap_close(bp2, bp3, arm):
    denom = bp2 - bp3
    return ((bp2 - arm) / denom) if denom > 1e-12 else float("nan")


def evaluate(rows, ceil_frac, sep_margin, gap_close_bar, min_testable):
    """Per-seed ceiling-gating, then a FIT-based verdict over the TESTABLE seeds only."""
    for r in rows:
        r["ceiling_holds"] = bool(r["bp3_loss"] <= ceil_frac * r["y_var"])
        r["depth_sep"] = bool((r["bp2_loss"] - r["bp3_loss"]) > sep_margin * r["y_var"])
        r["testable"] = bool(r["ceiling_holds"] and r["depth_sep"])
        r["kp_gap_close"] = _gap_close(r["bp2_loss"], r["bp3_loss"], r["kp_loss"])
        r["dfa_gap_close"] = _gap_close(r["bp2_loss"], r["bp3_loss"], r["dfa_loss"])
        r["fa_gap_close"] = _gap_close(r["bp2_loss"], r["bp3_loss"], r["fa_loss"])
    T = [r for r in rows if r["testable"]]
    n_testable = len(T)

    def m(key, rowset=T):
        xs = [rr[key] for rr in rowset
              if rr.get(key) is not None and not (isinstance(rr[key], float) and np.isnan(rr[key]))]
        return float(np.mean(xs)) if xs else float("nan")

    means = {
        "n_testable": n_testable, "n_total": len(rows),
        "bp3_loss": m("bp3_loss"), "bp2_loss": m("bp2_loss"), "kp_loss": m("kp_loss"),
        "fa_loss": m("fa_loss"), "dfa_loss": m("dfa_loss"), "perm_loss": m("perm_loss"),
        "mean_pred_loss": m("mean_pred_loss"), "y_var": m("y_var"),
        "kp_gap_close": m("kp_gap_close"), "dfa_gap_close": m("dfa_gap_close"), "fa_gap_close": m("fa_gap_close"),
        "kp_bw_indep_init_max": m("kp_bw_indep_init_max"),
        "kp_bw_deep_init": m("kp_bw_deep_init"), "kp_bw_deep_fin": m("kp_bw_deep_fin"),
        "kp_bw_top_init": m("kp_bw_top_init"), "kp_bw_top_fin": m("kp_bw_top_fin"),
    }

    kp_moved_all = all(r["kp_moved"] for r in rows)
    fa_frozen_all = all(r["fa_frozen"] for r in rows)
    no_transport_all = all(r["kp_no_transport"] for r in rows)
    determinism_all = all(r["determinism_ok"] for r in rows)

    # ---- executed levers / attributions (tools.lab), not comments -----------------------------------------
    lever_moved = True
    try:
        # FREEZE-Y lever: feedback-learning ON (KP) vs OFF (frozen Y = seq_fixed). Fit gap-close is the read-out.
        lever("KP feedback-learning ON vs freeze-Y (seq_fixed) -- spiking depth-3 FIT gap-close",
              round(means["fa_gap_close"], 4), round(means["kp_gap_close"], 4),
              continuous="fit gap-close: KP %.3f vs frozen-Y %.3f | losses KP %.4g fa %.4g dfa %.4g | deep cos(Y,W^T) "
              "%.3f->%.3f" % (means["kp_gap_close"], means["fa_gap_close"], means["kp_loss"], means["fa_loss"],
                              means["dfa_loss"], means["kp_bw_deep_init"], means["kp_bw_deep_fin"]))
        attributable_to("spiking depth-3 fit gap-close attributable to LEARNING the feedback (KP vs freeze-Y)",
                        treatment_value=means["kp_gap_close"], control_value=means["fa_gap_close"])
        attributable_to("spiking depth-3 fit gap-close: KP over the fixed-DFA baseline",
                        treatment_value=means["kp_gap_close"], control_value=means["dfa_gap_close"])
    except LeverError:
        lever_moved = False

    # ---- FIT-based decision -------------------------------------------------------------------------------
    go_kp = bool((not np.isnan(means["kp_gap_close"])) and means["kp_gap_close"] >= gap_close_bar)
    dfa_fails = bool((not np.isnan(means["dfa_gap_close"])) and means["dfa_gap_close"] < 0.5)
    perm_ok = bool((not np.isnan(means["perm_loss"])) and means["perm_loss"] >= 0.5 * means["bp2_loss"])
    kp_beats_dfa = bool((means["kp_loss"] < means["dfa_loss"]) and abs(means["kp_loss"] - means["dfa_loss"]) > 1e-9)

    v = Verdict("gap4_onspikes_depth3_kp_reaches_layer3")
    v.require("enough_testable_seeds", n_testable >= min_testable, expect=True,
              note="%d/%d seeds are TESTABLE (surrogate-BPTT depth-3 ceiling holds AND depth-2 underfits); need >= %d. "
                   "Per-seed ceiling-gating: a seed where the ORACLE cannot fit on spikes carries no depth-3 info. If "
                   "ZERO seeds are testable, there is NO valid spiking depth-3 instrument at this scale -> UNDEFINED "
                   "(the honest instrument-limit map, NOT a fabricated negative)." % (n_testable, len(rows), min_testable))
    v.require("lever_kp_changes_fit", bool(lever_moved), expect=True,
              note="the KP-vs-freeze-Y fit-gap-close lever MOVED. If it did NOT (KP == frozen-Y), the deep credit is "
                   "dead at this depth and no verdict is earned -> UNDEFINED.")
    v.require("backprop_oracle_ceiling_exists", bool(n_testable and means["bp3_loss"] <= ceil_frac * means["y_var"]),
              expect=True, note="mean surrogate-BPTT depth-3 loss %.4g <= %.4g (%.0f%% of var %.4g) on testable seeds"
              % (means["bp3_loss"], ceil_frac * means["y_var"], 100 * ceil_frac, means["y_var"]))
    v.require("depth_separating_on_fit",
              bool(n_testable and (means["bp2_loss"] - means["bp3_loss"]) > sep_margin * means["y_var"]), expect=True,
              note="mean BPTT depth-2 %.4g > depth-3 %.4g by > %.4g (%.0f%% of var): the tent^k target is depth-3-"
              "ENGAGING on FIT on the LIF SNN" % (means["bp2_loss"], means["bp3_loss"], sep_margin * means["y_var"],
                                                  100 * sep_margin))
    v.require("fixed_dfa_baseline_fails_to_reach_layer3", dfa_fails, expect=True,
              note="fixed-DFA closes %.1f%% of the BP2->BP3 fit gap (< 50%%) -> it does NOT reach the deep layers on "
              "spikes. This is the baseline KP must beat." % (100 * means["dfa_gap_close"]))
    v.require("permuted_target_does_not_learn_fit", perm_ok, expect=True,
              note="KP on per-step-reshuffled targets: loss %.4g >= 0.5*bp2 %.4g -> no fit from a signal-free target"
              % (means["perm_loss"], means["bp2_loss"]))
    v.require("lever_KP_moves_feedback", kp_moved_all, expect=True,
              note="seq_kp updated Y_fb every step (feedback-learning ON); freezing it (seq_fixed) left Y_fb unchanged=%s"
              % fa_frozen_all)
    v.require("transport_free_not_a_copy_of_Wt", bool(means["kp_bw_indep_init_max"] < 0.8), expect=True,
              note="max |cos(Y_fb, W^T)| at INIT = %.3f (< 0.8 -> NOT a copy; a W^T copy reads ~1.0). Y_fb is a SEPARATE "
              "random stream; the credit path reads Y_fb not W^T; KP updates Y_fb by the matched Adam-step transpose. "
              "cos(Y_fb,W^T) RISES through training (deep init %.3f -> final %.3f) -> co-adapted, not transported."
              % (means["kp_bw_indep_init_max"], means["kp_bw_deep_init"], means["kp_bw_deep_fin"]))
    v.require("transport_free_no_byte_copy", no_transport_all, expect=True,
              note="no Y_fb is byte-equal any forward W or its transpose, all seeds")
    v.require("substrate_seeded_deterministic", determinism_all, expect=True,
              note="two fresh builds at one seed give byte-identical forward weights (substrate RNG = default_rng(seed); "
              "this runner does not touch the bridge, so there is no cfg.seed to mis-set)")
    v.control("kp_fit_differs_from_fixed_dfa", treatment=means["kp_loss"], control=means["dfa_loss"],
              min_separation=1e-9, note="the KP fit must differ from the fixed-DFA baseline's fit (the manipulation landed)")

    decided = v.decide(bool(go_kp and kp_beats_dfa))
    return {
        "means": means, "rows_testable": [r["seed"] for r in T],
        "checks": {"go_kp": go_kp, "dfa_fails": dfa_fails, "perm_ok": perm_ok, "kp_beats_dfa": kp_beats_dfa,
                   "kp_moved_all": kp_moved_all, "fa_frozen_all": fa_frozen_all, "no_transport_all": no_transport_all,
                   "determinism_all": determinism_all, "lever_moved": lever_moved},
        "go": bool(decided["status"] == "GO"), "status": decided["status"],
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
    }


def _verdict_line(ev, gap_close_bar):
    m = ev["means"]
    if ev["status"] == "UNDEFINED":
        return ("UNDEFINED (a precondition failed -- NOT a negative; the honest instrument-limit map): "
                + "; ".join(ev["undefined_reasons"])
                + (" | A valid spiking depth-3 instrument needs a LIF/tent^k regime where surrogate-BPTT depth-3 fits "
                   "(ceiling) AND depth-2 underfits (separation) on the SAME seeds -- if N_testable=0, no such regime "
                   "exists at this (hidden,T,epochs,in_gain) scale." if ev["means"]["n_testable"] < 1 else ""))
    if ev["status"] == "GO":
        return ("GO: DEPTH-3 CREDIT REACHES THE SPIKING SUBSTRATE. Transport-free KP-learned feedback closes %.0f%% of "
                "the BPTT-depth-2 -> BPTT-depth-3 FIT gap (>= %.0f%%) on the LIF SNN, where fixed-DFA closes only %.0f%% "
                "and freezing Y (freeze-Y lever) collapses it to %.0f%%. KP loss %.4g vs bp3 %.4g / bp2 %.4g / dfa %.4g. "
                "Transport-free: cos(Y_fb,W^T) init %.3f -> deep final %.3f (co-adapted, not copied). On %d/%d testable "
                "seeds -- the depth-3 analog of the depth-2 on-spikes GO."
                % (100 * m["kp_gap_close"], 100 * gap_close_bar, 100 * m["dfa_gap_close"], 100 * m["fa_gap_close"],
                   m["kp_loss"], m["bp3_loss"], m["bp2_loss"], m["dfa_loss"], m["kp_bw_indep_init_max"],
                   m["kp_bw_deep_fin"], m["n_testable"], m["n_total"]))
    return ("NO-GO (HONEST NEGATIVE -- a first-class gap#4 deliverable): even KP-LEARNED transport-free feedback did "
            "NOT reach genuine depth-3 credit on spikes at the bar. KP closes %.0f%% of the BP2->BP3 fit gap (< %.0f%%); "
            "fixed-DFA %.0f%%, freeze-Y %.0f%%. KP loss %.4g vs bp3 %.4g / bp2 %.4g. Deep cos(Y,W^T) %.3f->%.3f. Next "
            "mechanism: weight-mirror (Akrout noise-driven) / stronger KP / the phi'(surrogate)-vanishing fix. On %d/%d "
            "testable seeds." % (100 * m["kp_gap_close"], 100 * gap_close_bar, 100 * m["dfa_gap_close"],
                                 100 * m["fa_gap_close"], m["kp_loss"], m["bp3_loss"], m["bp2_loss"],
                                 m["kp_bw_deep_init"], m["kp_bw_deep_fin"], m["n_testable"], m["n_total"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=str, default="42", help="comma-separated seeds for a self-aggregating sweep.")
    ap.add_argument("--task", type=str, default="tent3_fit", choices=["tent2_fit", "tent3_fit", "tent4_fit"],
                    help="the depth-k-composed FIT target. tent3_fit = tent(tent(tent(x))), MSE. PRIMARY = tent3.")
    ap.add_argument("--hidden", type=int, default=7, help="hidden width (same for the depth-2 and depth-3 nets so "
                    "WIDTH does not confound DEPTH). NARROW on purpose: at width ~6-8 the depth-3 LIF net fits tent^3 "
                    "(~O(H^2) folds) while depth-2 (~O(H) folds) underfits the 8-fold target -- the Telgarsky "
                    "depth-vs-width separation window found empirically for this LIF forward. Wide (>=16) -> depth-2 "
                    "also fits (width substitutes for depth, no separation); too narrow (<=4) -> even depth-3 underfits.")
    ap.add_argument("--timesteps", type=int, default=40, help="rate-coding window T. More steps = finer population-"
                    "rate resolution for the continuous regression read-out.")
    ap.add_argument("--epochs", type=int, default=4000, help="KP/FA feedback-alignment converges SLOWER than BPTT, so "
                    "the depth-3 credit only reaches the deep layer with enough training.")
    ap.add_argument("--lr", type=float, default=0.02, help="Adam lr (mode-agnostic; applied identically to every arm; "
                    "cosine-decayed to 0 over training).")
    ap.add_argument("--clip", type=float, default=1.0, help="global-norm gradient clip (stabilizes the LIF surrogate "
                    "updates; the raw updates diverge -- the LIF net fits tent^k then blows back to the mean predictor).")
    ap.add_argument("--in-gain", type=float, default=1.0, help="input-current scale for the rate-coded scalar input.")
    ap.add_argument("--bias-input", action=argparse.BooleanOptionalAction, default=True,
                    help="append a constant 1.0 input channel so layer-0 has per-neuron biases (LIFLayerXP has no bias "
                    "term; without this the first LIF layer only produces monotone f-I features -> less expressive).")
    ap.add_argument("--w-scale0", type=float, default=2.5, help="first-layer weight scale (strong -> fire from the "
                    "continuous input, like the char-SNN std=2.5).")
    ap.add_argument("--leak", type=float, default=0.9)
    ap.add_argument("--n-points", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--align-every", type=int, default=200, help="checkpoint stride for the cos(Y_fb,W^T) trajectory.")
    ap.add_argument("--ceil-frac", type=float, default=0.12, help="per-seed BPTT-depth-3 loss must be <= this fraction "
                    "of the target variance to count as a ceiling. LIF-substrate-CALIBRATED (the rate ReLU-MLP "
                    "instrument used 0.02-0.05 where BP fits to ~0.002*var; the COARSE LIF forward at the narrow "
                    "depth-2-underfitting width only reaches ~0.05-0.11*var -- a genuine >=88%-variance-reduction fit "
                    "for THIS substrate). NB the VERDICT is UNDEFINED at 0.05/0.10/0.15/0.20 alike (the blocker is the "
                    "fixed-DFA-does-not-fail premise, not the ceiling value), so this only sets which seeds are "
                    "reported testable; the finding carries the full ceil-frac sensitivity.")
    ap.add_argument("--sep-margin", type=float, default=0.05, help="per-seed (bp2-bp3) gap as a fraction of var for the "
                    "target to count as depth-3-engaging on FIT.")
    ap.add_argument("--gap-close-bar", type=float, default=0.5, help="KP must close >= this fraction of the BP2->BP3 "
                    "fit gap to count as REACHING depth-3 (distinct from fixed-DFA's ~0).")
    ap.add_argument("--min-testable", type=int, default=3, help="minimum ceiling-holding seeds for a decisive verdict "
                    "(else UNDEFINED -- the honest instrument-limit map).")
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    k = {"tent2_fit": 2, "tent3_fit": 3, "tent4_fit": 4}[args.task]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))
    device = "cpu" if backend == "numpy" else "cuda"

    t0 = time.time()
    rows, err = [], None
    try:
        for s in seeds:
            r = run_seed(s, k, args.hidden, args.timesteps, args.epochs, args.lr, args.in_gain, args.n_points,
                         args.batch, args.w_scale0, args.leak, args.bias_input, args.clip, args.align_every)
            rows.append(r)
            print("[seed %d tent^%d] bp3=%.4g bp2=%.4g | KP=%.4g(gc%.0f%%) DFA=%.4g(gc%.0f%%) FA=%.4g(gc%.0f%%) "
                  "perm=%.4g | var=%.4g | cos(Y,W^T) deep %.3f->%.3f top %.3f->%.3f | testable_pending"
                  % (s, k, r["bp3_loss"], r["bp2_loss"], r["kp_loss"], 100 * _gap_close(r["bp2_loss"], r["bp3_loss"], r["kp_loss"]),
                     r["dfa_loss"], 100 * _gap_close(r["bp2_loss"], r["bp3_loss"], r["dfa_loss"]),
                     r["fa_loss"], 100 * _gap_close(r["bp2_loss"], r["bp3_loss"], r["fa_loss"]), r["perm_loss"],
                     r["y_var"], r["kp_bw_deep_init"], r["kp_bw_deep_fin"], r["kp_bw_top_init"], r["kp_bw_top_fin"]))
        ev = evaluate(rows, args.ceil_frac, args.sep_margin, args.gap_close_bar, args.min_testable)
    except Exception as e:
        ev = {"error": repr(e), "traceback": traceback.format_exc()}
        err = repr(e)

    out = {"probe": "gap4_onspikes_depth3_credit_fidelity", "task": args.task, "k_composition": k, "seeds": seeds,
           "backend": backend, "device": device,
           "config": {"hidden": args.hidden, "T": args.timesteps, "epochs": args.epochs, "lr": args.lr,
                      "clip": args.clip, "in_gain": args.in_gain, "bias_input": bool(args.bias_input),
                      "w_scale0": args.w_scale0, "leak": args.leak, "readout": "rate_normalized_sum_spikes_over_T",
                      "loss_metric": "best_over_training", "lr_schedule": "cosine_decay",
                      "n_points": args.n_points, "batch": args.batch,
                      "bars": {"ceil_frac": args.ceil_frac, "sep_margin": args.sep_margin,
                               "gap_close": args.gap_close_bar, "min_testable": args.min_testable}},
           "elapsed_seconds": round(time.time() - t0, 1), "rows": rows, "result": ev}
    out["verdict"] = err or (_verdict_line(ev, args.gap_close_bar) if "means" in ev else str(ev.get("error")))
    out["preconditions"] = ev.get("preconditions", [])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    if "means" in ev:
        m = ev["means"]
        print("\n  backend=%s device=%s  task=%s (tent^%d)  hidden=%d  T=%d  seeds=%s"
              % (backend, device, args.task, k, args.hidden, args.timesteps, seeds))
        print("  TESTABLE seeds (BPTT depth-3 ceiling holds + depth-separating): %s of %d -> %s"
              % (m["n_testable"], m["n_total"], ev["rows_testable"]))
        print("  --- losses (MSE, testable-seed means) ---")
        print("    bp3(oracle)=%.5g  bp2(depth-2)=%.5g  mean-pred=%.5g  |  KP=%.5g  fixed_DFA=%.5g  freeze-Y=%.5g"
              % (m["bp3_loss"], m["bp2_loss"], m["mean_pred_loss"], m["kp_loss"], m["dfa_loss"], m["fa_loss"]))
        print("  --- FIT gap-close (fraction of BP2->BP3 gap closed; 1.0 == reaches the depth-3 oracle) ---")
        print("    KP=%.1f%%   fixed_DFA=%.1f%%   freeze-Y(seq_fixed)=%.1f%%"
              % (100 * m["kp_gap_close"], 100 * m["dfa_gap_close"], 100 * m["fa_gap_close"]))
        print("  --- transport-free feedback alignment cos(Y_fb,W^T) ---")
        print("    init(max|.|)=%.3f   deep: %.3f -> %.3f   top(adjacent): %.3f -> %.3f"
              % (m["kp_bw_indep_init_max"], m["kp_bw_deep_init"], m["kp_bw_deep_fin"],
                 m["kp_bw_top_init"], m["kp_bw_top_fin"]))
        print("  --- checks --- %s" % ev["checks"])
    print("\n" + out["verdict"])
    print("[onspikes-depth3-credit-fidelity] status=%s  wrote %s" % (ev.get("status", "ERROR"), args.out))


if __name__ == "__main__":
    main()
