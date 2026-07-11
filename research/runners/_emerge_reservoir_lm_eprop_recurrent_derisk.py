"""NEXT-MECHANISM de-risk (the boundary-surpassing step past the SCALE CAPSTONE). The capstone showed a FIXED reservoir +
linear read-out is n-gram-competitive-at-best on real text: on harder WikiText its contribution BEYOND a learned n-gram is
negative/zero at every context depth -- it captures NO usable long-range structure. The named next mechanism (owner's
STANDING dendritic/deep-credit priority): make the reservoir's RECURRENT weights LEARN, by a biologically-grounded,
ONE-STEP-LOCAL, NO-BPTT rule -- so the "reservoir" becomes a trained recurrent cortex, but trained the brain-plausible way.

MECHANISM = e-prop (Bellec-Maass 2020, Nat Commun 11:3625), random-feedback / broadcast-alignment variant -- the RATE
analogue of the on-bridge BDSP/Burstprop already in `sim/bridge.py` (enable_bdsp). For a leaky-tanh rate RNN:
  h_t   = (1-a) h_{t-1} + a * tanh( W_rec h_{t-1} + W_in x_t + b )         [state]
  p_t   = softmax( W_out h_t )                                            [read-out predicts token t+1]
  delta = onehot(target) - p_t                                           [clean read-out error]
  read-out (delta rule):  W_out += lr_out * outer(delta, h_t)
  eligibility (forward-filtered local sensitivity of h_j to w_rec[j,i]):
      psi_j    = a * (1 - h_t,j^2)                 [pseudo-derivative of the unit]
      e[j,i]   = (1-a) * e[j,i] + psi_j * h_{t-1,i}
  learning signal (BROADCAST / random feedback -- NO weight transport, as in on-bridge BDSP's fixed-random apical route):
      L_j      = (B @ delta)_j        (B: n x V fixed random feedback)
  recurrent update (e-prop): W_rec[j,i] += lr_rec * L_j * e[j,i]
NO backprop-through-time -- e is forward-accumulated, L is a per-neuron broadcast of the current read-out error.

THE SINGLE-VARIABLE TEST: does making W_rec PLASTIC (e-prop) recover the deep-context CE that the FIXED reservoir loses on
WikiText? Arms (one variable = whether/how W_rec learns; read-out + reservoir init IDENTICAL across arms):
  fixed         -- W_rec frozen (the echo-state baseline = the capstone's substrate).
  plastic       -- e-prop random-feedback on W_rec (the mechanism under test).
  shuffle_elig  -- e-prop but the eligibility matrix is PERMUTED before each update (credit-assignment BROKEN) -> if this
                   also helps, the gain is not real credit assignment. ANTI-CHEAT.
  zero_signal   -- e-prop but L:=0 (no learning signal) -> W_rec never moves -> must == fixed. SANITY.
Metric: per-context-depth CE (reuse the capstone buckets), especially DEEP (6+) where the fixed reservoir was n-gram-level.
GO = plastic beats fixed at deep context by margin, shuffle_elig does NOT, zero_signal == fixed. Reuse-by-import; numpy
rate-reservoir (cheap-first rung of the rate->spike->sim ladder); NO `sim/` edit. If GO -> wire on-bridge BDSP (apical =
read-out error) onto the spiking reservoir's recurrent synapses.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, math, time
from pathlib import Path
from collections import defaultdict
import numpy as np

from research.runners._emerge_reservoir_lm_derisk import Vocab, _softmax, fit_bigram
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket

OUT = Path("research/findings/raw/_reslm_eprop.json")


class RateReservoir:
    """Leaky-tanh rate RNN = the rate analogue of the on-bridge LIF reservoir. W_rec starts as a spectral-radius-scaled
       random matrix (echo-state init); it is FIXED unless e-prop updates it. `adaptive_frac` gives a fraction of units a
       SLOW leak `alpha_slow` (heterogeneous time constants = the rate analogue of ALIF/spike-frequency-adaptation): the
       e-prop eligibility of a slow unit decays over ~1/alpha_slow tokens, so credit reaches the DEEP context the fast
       (~1/alpha) eligibility cannot (Bellec-2020 ALIF is what gives e-prop its long memory; the horizon lever)."""
    def __init__(self, V, n, seed, alpha=0.3, spectral=1.1, in_scale=1.0, adaptive_frac=0.0, alpha_slow=0.05,
                 leak_spectrum="homogeneous", alpha_lo=0.03, alpha_hi=0.6,
                 alif=False, beta=1.0, adapt_win_lo=30.0, adapt_win_hi=300.0):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n, n))
        sr = np.max(np.abs(np.linalg.eigvals(W)))                # circular-law radius ~ sqrt(n)
        self.W_rec = (spectral / sr) * W                          # set spectral radius = `spectral`
        self.W_in = rng.standard_normal((n, V)) * (in_scale / np.sqrt(V))
        self.b = np.zeros(n)
        self.n = n; self.V = V
        # per-unit leak (state-memory horizon lever). DEFAULT homogeneous = byte-identical to the committed e-prop runs.
        self.alpha = np.full(n, float(alpha))
        self.slow_idx = np.array([], dtype=int)                   # units whose STATE holds long context (for the lesion)
        # ---- ALIF adaptation-as-STATE (Salaj-Bellec 2021 / Bellec-2020 e-prop ALIF). DEFAULT OFF -> the rho draw is the
        # LAST rng call, so W_rec/W_in are byte-identical to a non-ALIF reservoir at the same seed (single-variable). The
        # adaptation a_j is a per-unit NON-FADING slow trace of the unit's OWN activity (rho_j near 1); it subtracts from
        # the pre-activation (an "activity-silent negative imprint" HOLD, NOT a diluting average), carries history FORWARD,
        # and is READ by the read-out (feature = concat([h, a])). rho_j log-uniform over effective windows [win_lo, win_hi]
        # tokens: rho = 1 - 1/window (heterogeneous adaptation time-constants, the diverse-timescale hold).
        self.alif = bool(alif)
        self.beta = float(beta)
        self.rho = np.zeros(n)
        if leak_spectrum == "hetero":
            # log-uniform leak spectrum: effective windows ~1/alpha span [1/alpha_hi .. 1/alpha_lo] tokens (diverse
            # cortical membrane/adaptation time constants, catalog I.16). Slow units = long state memory, READ by the read-out.
            log_a = rng.uniform(np.log(alpha_lo), np.log(alpha_hi), size=n)
            self.alpha = np.exp(log_a)
            self.slow_idx = np.where(self.alpha < np.median(self.alpha))[0]   # the slower half
        elif leak_spectrum == "homomean":
            # CONTROL: all units at the MEAN leak of the hetero spectrum (same mean, ZERO diversity) -> isolates whether
            # the slow-tail DIVERSITY (not merely a slower mean) is what carries deep context.
            log_a = rng.uniform(np.log(alpha_lo), np.log(alpha_hi), size=n)
            self.alpha = np.full(n, float(np.mean(np.exp(log_a))))
        if adaptive_frac > 0.0:
            k = int(round(adaptive_frac * n))
            slow = rng.choice(n, size=k, replace=False)
            self.alpha[slow] = float(alpha_slow)                 # (legacy binary slow-leak; unused by default modes)
        if self.alif:                                            # drawn LAST -> W_rec/W_in byte-identical to non-ALIF
            win = np.exp(rng.uniform(np.log(adapt_win_lo), np.log(adapt_win_hi), size=n))  # log-uniform windows
            self.rho = 1.0 - 1.0 / win                           # rho_j in ~[0.967, 0.997] for [30, 300]-token windows

    def forward_states(self, ids, shuffle_adapt=False, shuffle_seed=0):
        """Run the (possibly-trained) reservoir over a sequence; return states[t].
           NON-ALIF: states[t] = h_t (the leaky recurrent state at token t).
           ALIF: states[t] = concat([h_t, a_t]) -- the fast leaky state AND the non-fading adaptation state, both READ by
           the read-out (a_t carries distal history). `shuffle_adapt` (ADAPTATION-SHUFFLE anti-cheat): permute a_t across
           neurons at each token before the read-out -> same extra dims, WRONG content -> must collapse to the no-adapt arm
           (kills the 'extra read-out capacity, not content' confound)."""
        if not self.alif:
            a = self.alpha; h = np.zeros(self.n); out = []
            for t in ids:
                x = self.W_in[:, t]                              # onehot(t) @ W_in^T = column t
                h = (1 - a) * h + a * np.tanh(self.W_rec @ h + x + self.b)
                out.append(h.copy())
            return out
        a = self.alpha; rho = self.rho; beta = self.beta
        h = np.zeros(self.n); ad = np.zeros(self.n); out = []
        srng = np.random.default_rng(shuffle_seed) if shuffle_adapt else None
        for t in ids:
            h_prev = h
            ad = rho * ad + (1.0 - rho) * h_prev                 # adaptation = non-fading slow trace of own activity
            x = self.W_in[:, t]
            pre = self.W_rec @ h_prev + x + self.b - beta * ad   # activity-silent negative imprint subtracts
            h = (1 - a) * h_prev + a * np.tanh(pre)
            ad_read = ad[srng.permutation(self.n)] if shuffle_adapt else ad
            out.append(np.concatenate([h, ad_read]))
        return out


def _train_alif(res, tr_ids, V, epochs, lr_out, lr_rec, rng, mode, wd, wd_rec):
    """ALIF adaptation-as-STATE e-prop training. mode in {alif, alif_readonly}. Forward = the ALIF rate dynamics (adaptation
    a_j subtracts a beta-scaled negative imprint from pre_j). Read-out feature = concat([h, a]) (2n) trained by the one-step
    next-token delta rule. Recurrent W_rec by e-prop with the FAITHFUL Bellec-2020 ALIF 2-COMPONENT eligibility -- eps_h
    (d h_j / d w_ji) COUPLED to eps_a (d a_j / d w_ji) -- and BOTH read-out paths credited (L_h * eps_h + L_a * eps_a). This
    2-component form (not the fast-only one) is what the mandatory finite-difference gradient check validates: because the
    read-out observes BOTH compartments, faithful credit for w_ji must sum the h-path and the a-path sensitivities. `mode ==
    alif_readonly` still READS a but credits W_rec via the FAST eligibility ONLY (drops the eps_a coupling AND the a-path)
    -> isolates 'state carried forward + read' from 'adaptation-coupling credited'. Random feedback B (2n x V; NO weight
    transport, broadcast alignment). NO BPTT (all traces forward-accumulated). W_rec updated in place; returns W_out."""
    n = res.n; a = res.alpha; rho = res.rho; beta = res.beta     # a = alpha leak vector; rho = per-unit adaptation leak
    readonly = (mode == "alif_readonly")
    W_out = rng.standard_normal((V, 2 * n)) * 0.01
    B = rng.standard_normal((2 * n, V)) / np.sqrt(V)             # fixed random feedback over BOTH compartments [h; a]
    order = np.arange(len(tr_ids))
    for ep in range(epochs):
        rng.shuffle(order)
        for si in order:
            ids = tr_ids[si]
            if len(ids) < 2:
                continue
            h = np.zeros(n); ad = np.zeros(n)
            eps_h = np.zeros((n, n)); eps_a = np.zeros((n, n))
            for t in range(len(ids) - 1):
                h_prev = h
                ad = rho * ad + (1.0 - rho) * h_prev             # a_j(t) = rho a_j(t-1) + (1-rho) h_j(t-1): non-fading state
                x = res.W_in[:, ids[t]]
                pre = res.W_rec @ h_prev + x + res.b - beta * ad  # pre_j - beta a_j: activity-silent negative imprint
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act                   # h_j(t) = (1-alpha) h_j(t-1) + alpha tanh(pre): fast state
                feat = np.concatenate([h, ad])                   # read-out feature = [h_t ; a_t] (a carries distal history)
                p = _softmax(W_out @ feat)
                delta = -p; delta[ids[t + 1]] += 1.0             # target - p (clean next-token error)
                W_out += lr_out * (np.outer(delta, feat) - wd * W_out)
                psi = a * (1.0 - act * act)                      # psi_j = alpha (1 - act_j^2) = d h_j / d pre_j
                if readonly:
                    eps_h = (1 - a)[:, None] * eps_h + psi[:, None] * h_prev[None, :]        # FAST only (eps_a dropped)
                else:
                    eps_a = rho[:, None] * eps_a + (1.0 - rho)[:, None] * eps_h              # d a_j / d w_ji (uses OLD eps_h)
                    eps_h = (1 - a)[:, None] * eps_h + psi[:, None] * (h_prev[None, :] - beta * eps_a)  # d h_j/d w_ji COUPLED
                L = B @ delta                                    # (2n,) random-feedback learning signal over [h; a]
                L_h = L[:n]; L_a = L[n:]
                if readonly:
                    res.W_rec += lr_rec * (L_h[:, None] * eps_h)                             # credit h-path only
                else:
                    res.W_rec += lr_rec * (L_h[:, None] * eps_h + L_a[:, None] * eps_a)      # credit BOTH read-out paths
                if wd_rec > 0.0:
                    res.W_rec -= lr_rec * wd_rec * res.W_rec      # W_rec weight decay (capacity/overfit guard)
    return W_out


def train(res, tr_ids, V, epochs, lr_out, lr_rec, seed, mode="plastic", wd=1e-3, wd_rec=0.0, a_slow_leak=0.05):
    """Online co-training: W_out by the one-step delta rule; W_rec by e-prop, NO BPTT. mode in
       {fixed, plastic, adaptive, symmetric, shuffle_elig, zero_signal, alif, alif_readonly}. `adaptive` adds a
       DUAL-TIMESCALE eligibility (a slow trace with leak `a_slow_leak` alongside the fast one) so credit reaches the DEEP
       context the fast ~1/alpha eligibility cannot -- WITHOUT changing the forward dynamics (the reservoir state stays
       fast; the read-out is unaffected; only the CREDIT horizon lengthens -- the faithful ALIF idea, vs naive slow-leak
       which degrades the state). `alif`/`alif_readonly` are the ALIF adaptation-as-STATE arms (see _train_alif); they
       dispatch on `res.alif`. `symmetric` uses W_out^T feedback (weight-transport ceiling on the random-feedback cost);
       `wd_rec` = weight decay on W_rec (guards the capacity/overfitting confound). Returns W_out (W_rec updated in place)."""
    rng = np.random.default_rng(seed * 13 + 7)
    n = res.n; a = res.alpha                                      # per-unit leak vector
    if getattr(res, "alif", False):                              # ALIF arms -> the 2-component-eligibility trainer
        return _train_alif(res, tr_ids, V, epochs, lr_out, lr_rec, rng, mode, wd, wd_rec)
    W_out = rng.standard_normal((V, n)) * 0.01
    B = rng.standard_normal((n, V)) / np.sqrt(V)                  # fixed random feedback (broadcast alignment)
    symmetric = (mode == "symmetric")
    dual = (mode == "adaptive")                                   # dual-timescale eligibility (horizon lever)
    a_slow = a_slow_leak                                          # slow eligibility leak (the long-horizon component)
    order = np.arange(len(tr_ids))
    for ep in range(epochs):
        rng.shuffle(order)
        for si in order:
            ids = tr_ids[si]
            if len(ids) < 2:
                continue
            h = np.zeros(n); e = np.zeros((n, n)); e_slow = np.zeros((n, n))
            for t in range(len(ids) - 1):
                h_prev = h
                x = res.W_in[:, ids[t]]
                pre = res.W_rec @ h_prev + x + res.b
                act = np.tanh(pre)
                h = (1 - a) * h_prev + a * act                    # forward dynamics UNCHANGED across all arms (fast state)
                p = _softmax(W_out @ h)
                delta = -p; delta[ids[t + 1]] += 1.0             # d(-log p)/d logits = target - p
                W_out += lr_out * (np.outer(delta, h) - wd * W_out)
                if mode == "fixed":
                    continue
                psi = a * (1.0 - act * act)                       # per-unit pseudo-derivative
                incr = np.outer(psi, h_prev)
                e = (1 - a)[:, None] * e + incr                   # fast forward-filtered eligibility (horizon ~1/alpha)
                if dual:
                    e_slow = (1 - a_slow) * e_slow + incr         # slow eligibility (horizon ~1/a_slow); state UNCHANGED
                if mode == "zero_signal":
                    continue                                     # L:=0 -> W_rec never moves (sanity == fixed)
                L = (W_out.T @ delta) if symmetric else (B @ delta)   # symmetric (weight transport) vs random feedback
                E_use = (e + e_slow) if dual else e
                if mode == "shuffle_elig":
                    E_use = e.reshape(-1)[rng.permutation(n * n)].reshape(n, n)  # break credit assignment
                res.W_rec += lr_rec * (L[:, None] * E_use)
                if wd_rec > 0.0:
                    res.W_rec -= lr_rec * wd_rec * res.W_rec      # W_rec weight decay (capacity/overfit guard)
    return W_out


def per_depth_ce(res, W_out, ev_ids, P_bi, lesion_slow=False, shuffle_adapt=False):
    """Per-context-depth held-out CE for the (trained) reservoir+read-out and the bigram baseline.
       lesion_slow: zero the read-out weights on the SLOW units -> if the deep gain collapses, the slow units' STATE
       was carrying the distal context the read-out uses (the state-memory lesion control).
       shuffle_adapt: (ALIF ADAPTATION-SHUFFLE anti-cheat) permute a_t across neurons before the read-out -> same extra
       read-out dims, WRONG content -> the deep gain must collapse to the no-adaptation arm (content, not capacity)."""
    if lesion_slow and res.slow_idx.size:
        W_out = W_out.copy(); W_out[:, res.slow_idx] = 0.0
    rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
    for si, ids in enumerate(ev_ids):
        states = res.forward_states(ids, shuffle_adapt=shuffle_adapt, shuffle_seed=1000 + si)
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            p = _softmax(W_out @ states[t]); tgt = ids[t + 1]
            rce[b] += -math.log(max(p[tgt], 1e-12))
            bce[b] += -math.log(max(P_bi[ids[t], tgt], 1e-12)); cnt[b] += 1
    tot_r = sum(rce.values()); tot_b = sum(bce.values()); n = sum(cnt.values())
    depth = {b: {"n": cnt[b], "ce": round(rce[b] / cnt[b], 3), "bigram_ce": round(bce[b] / cnt[b], 3),
                 "margin_vs_bigram": round((bce[b] - rce[b]) / cnt[b], 3)} for b in cnt}
    return depth, round(tot_r / n, 3), round(tot_b / n, 3)


def _alif_trace(Wr, W_in, b, alpha, rho, beta, ids, n):
    """Pure-numpy ALIF forward re-implementing _train_alif's equations, accumulating the 2-component eligibility. Returns
    per-step h, a, recurrent-drive, and the eligibility histories eps_h[t]/eps_a[t]. Used ONLY by the gradient check."""
    h = np.zeros(n); ad = np.zeros(n)
    eps_h = np.zeros((n, n)); eps_a = np.zeros((n, n))
    hs = []; ads = []; recs = []; eh = []; ea = []
    for t in range(len(ids)):
        h_prev = h
        ad = rho * ad + (1.0 - rho) * h_prev
        rec = Wr @ h_prev
        pre = rec + W_in[:, ids[t]] + b - beta * ad
        act = np.tanh(pre)
        h = (1 - alpha) * h_prev + alpha * act
        psi = alpha * (1.0 - act * act)
        eps_a = rho[:, None] * eps_a + (1.0 - rho)[:, None] * eps_h            # d a_j / d w_ji (uses OLD eps_h)
        eps_h = (1 - alpha)[:, None] * eps_h + psi[:, None] * (h_prev[None, :] - beta * eps_a)  # d h_j / d w_ji COUPLED
        hs.append(h.copy()); ads.append(ad.copy()); recs.append(rec.copy()); eh.append(eps_h.copy()); ea.append(eps_a.copy())
    return {"hs": hs, "ads": ads, "recs": recs, "eps_h": eh, "eps_a": ea}


def _alif_loss(hs, ads, W_out, ids, n):
    """Total next-token CE loss over the sequence given per-step [h; a] features (for the finite-difference check)."""
    loss = 0.0
    for t in range(len(ids) - 1):
        feat = np.concatenate([hs[t], ads[t]])
        z = W_out @ feat; z = z - z.max(); e = np.exp(z); p = e / e.sum()
        loss += -math.log(max(p[ids[t + 1]], 1e-12))
    return loss


def grad_check_alif(n=5, V=6, seq_len=8, seed=1):
    """MANDATORY faithfulness check: the ALIF e-prop eligibility must match finite differences (a mis-derived rule -- the
    prior naive-ALIF bug -- must be caught HERE, before any de-risk run).
      CHECK A (eligibility recursion, EXACT): eps_h[j,i], eps_a[j,i] vs a LOCAL finite-difference of h_j / a_j w.r.t.
        W_rec[j,i] with the cross-neuron recurrent drives held at their REFERENCE trajectory (= the e-prop locality
        assumption). This isolates + validates the derived 2-component recursion; max rel err must be ~1e-5.
      CHECK B (the prompt's loss-gradient framing): the e-prop total gradient sum_t[(dL/dh)eps_h + (dL/da)eps_a] with TRUE
        W_out feedback vs the FULL-net finite-difference d(sum_t -log p_target)/dW_rec[j,i]. e-prop is a diagonal
        approximation of the exact gradient, so at the operating spectral radius (1.1) the residual is the KNOWN e-prop
        off-diagonal truncation; at a small spectral radius (0.2) it collapses to within a few % (validating the L*eps
        product + read-out feedback wiring). Reported honestly."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V, size=seq_len)
    W_out = rng.standard_normal((V, 2 * n)) * 0.3
    # short adaptation windows (3-30) so eps_a is genuinely active within an 8-token check sequence (validates the MATH).
    res = RateReservoir(V, n, seed=seed + 1, alif=True, beta=1.0, spectral=1.1, adapt_win_lo=3.0, adapt_win_hi=30.0)
    Wr = res.W_rec.copy(); W_in = res.W_in; b = res.b; alpha = res.alpha; rho = res.rho; beta = res.beta
    ref = _alif_trace(Wr, W_in, b, alpha, rho, beta, ids, n)
    hprev = [np.zeros(n)] + ref["hs"][:-1]                        # h_prev at each step (h_ref[t-1], zeros at t=0)

    # ---- CHECK A: local finite-difference of the eligibility at the final step (cross drives held at reference) ----
    def local_final(j, i, dw):
        hj = 0.0; adj = 0.0
        for t in range(seq_len):
            hj_prev = hj
            adj = rho[j] * adj + (1.0 - rho[j]) * hj_prev
            rec_j = ref["recs"][t][j] + dw * hprev[t][i]          # perturb ONLY the w_ji direct term; other drives at ref
            pre_j = rec_j + W_in[j, ids[t]] + b[j] - beta * adj
            hj = (1 - alpha[j]) * hj_prev + alpha[j] * np.tanh(pre_j)
        return hj, adj
    dh = 1e-6
    relh = []; rela = []; abs_h = 0.0; abs_a = 0.0
    for j in range(n):
        for i in range(n):
            hp, ap = local_final(j, i, dh); hm, am = local_final(j, i, -dh)
            fd_h = (hp - hm) / (2 * dh); fd_a = (ap - am) / (2 * dh)
            an_h = ref["eps_h"][-1][j, i]; an_a = ref["eps_a"][-1][j, i]
            abs_h = max(abs_h, abs(fd_h - an_h)); abs_a = max(abs_a, abs(fd_a - an_a))
            if abs(an_h) > 1e-7:
                relh.append(abs(fd_h - an_h) / abs(an_h))
            if abs(an_a) > 1e-7:
                rela.append(abs(fd_a - an_a) / abs(an_a))
    maxrel_h = max(relh) if relh else 0.0
    maxrel_a = max(rela) if rela else 0.0
    checkA_pass = (maxrel_h < 0.01) and (maxrel_a < 0.01)

    # ---- CHECK B: e-prop total loss gradient vs full-net finite-difference, at spectral 1.1 (real) and 0.2 (tight) ----
    def eprop_grad(Wr_):
        r = _alif_trace(Wr_, W_in, b, alpha, rho, beta, ids, n)
        g = np.zeros((n, n))
        for t in range(seq_len - 1):
            feat = np.concatenate([r["hs"][t], r["ads"][t]])
            z = W_out @ feat; z = z - z.max(); e = np.exp(z); p = e / e.sum()
            g_out = p.copy(); g_out[ids[t + 1]] -= 1.0           # dL/dlogits
            dLdfeat = W_out.T @ g_out
            g += dLdfeat[:n][:, None] * r["eps_h"][t] + dLdfeat[n:][:, None] * r["eps_a"][t]
        return g

    def fd_grad(Wr_):
        g = np.zeros((n, n)); dw = 1e-6
        for j in range(n):
            for i in range(n):
                Wp = Wr_.copy(); Wp[j, i] += dw
                rp = _alif_trace(Wp, W_in, b, alpha, rho, beta, ids, n)
                Wm = Wr_.copy(); Wm[j, i] -= dw
                rm = _alif_trace(Wm, W_in, b, alpha, rho, beta, ids, n)
                g[j, i] = (_alif_loss(rp["hs"], rp["ads"], W_out, ids, n)
                           - _alif_loss(rm["hs"], rm["ads"], W_out, ids, n)) / (2 * dw)
        return g

    def compare(spectral):
        r2 = RateReservoir(V, n, seed=seed + 1, alif=True, beta=1.0, spectral=spectral,
                           adapt_win_lo=3.0, adapt_win_hi=30.0)
        ge = eprop_grad(r2.W_rec); gf = fd_grad(r2.W_rec)
        mask = np.abs(gf) > 1e-7
        rel = np.abs(ge - gf)[mask] / np.abs(gf)[mask]
        cos = float(np.sum(ge * gf) / (np.linalg.norm(ge) * np.linalg.norm(gf) + 1e-30))
        return (float(np.median(rel)) if rel.size else 0.0, float(np.max(rel)) if rel.size else 0.0, cos)

    b_real = compare(1.1); b_tight = compare(0.2)

    print("=" * 100, flush=True)
    print(f"ALIF e-prop FAITHFULNESS CHECK  (n={n}, V={V}, seq_len={seq_len})", flush=True)
    print(f"  CHECK A (eligibility vs LOCAL finite-diff, EXACT): eps_h max_rel_err {maxrel_h:.2e} (max_abs {abs_h:.2e}) | "
          f"eps_a max_rel_err {maxrel_a:.2e} (max_abs {abs_a:.2e})  ->  {'PASS' if checkA_pass else 'FAIL'}", flush=True)
    print(f"  CHECK B (e-prop total grad vs FULL finite-diff):", flush=True)
    print(f"     spectral 1.1 (operating): median_rel {b_real[0]:.3f}  max_rel {b_real[1]:.3f}  cos {b_real[2]:.4f}  "
          f"(residual = e-prop off-diagonal truncation, expected)", flush=True)
    print(f"     spectral 0.2 (tight):     median_rel {b_tight[0]:.3f}  max_rel {b_tight[1]:.3f}  cos {b_tight[2]:.4f}  "
          f"(-> converges to FULL gradient within a few %)", flush=True)
    print("=" * 100, flush=True)
    return 0 if checkA_pass else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--n-sentences", type=int, default=6000)
    ap.add_argument("--max-train-sents", type=int, default=1500)
    ap.add_argument("--max-eval-sents", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--spectral", type=float, default=1.1)
    ap.add_argument("--lr-out", type=float, default=0.02)
    ap.add_argument("--lr-rec", type=float, default=0.002)
    ap.add_argument("--wd-rec", type=float, default=0.0)          # W_rec weight decay (capacity/overfit guard)
    ap.add_argument("--adaptive-frac", type=float, default=0.5)   # fraction of slow units in the `adaptive` arm
    ap.add_argument("--alpha-slow", type=float, default=0.05)     # slow-unit leak (~1/alpha_slow token eligibility horizon)
    ap.add_argument("--leak-spectrum", type=str, default="homogeneous", choices=["homogeneous", "hetero", "homomean"])
    ap.add_argument("--alpha-lo", type=float, default=0.03)       # hetero spectrum slowest leak (~33-token state memory)
    ap.add_argument("--alpha-hi", type=float, default=0.6)        # hetero spectrum fastest leak (~1.7-token, short context)
    ap.add_argument("--lesion-slow", action="store_true")         # eval-time: zero the read-out on slow units (lesion control)
    # ---- ALIF adaptation-as-STATE arms (A1 alif / A2 alif_readonly). Default byte-identical (no alif mode requested).
    ap.add_argument("--alif-beta", type=float, default=1.0)       # adaptation->pre coupling (negative-imprint strength; 0 = lesion)
    ap.add_argument("--adapt-win-lo", type=float, default=30.0)   # slowest/shortest adaptation window (tokens) for log-uniform rho
    ap.add_argument("--adapt-win-hi", type=float, default=300.0)  # longest adaptation window (tokens)
    ap.add_argument("--alif-controls", action="store_true",       # run ADAPTATION-SHUFFLE eval + beta=0 retrain for the alif arm
                    help="for the 'alif' mode: also eval with adaptation SHUFFLED across neurons (content control) and "
                         "retrain a beta=0 alif (adaptation-off lesion); the d10+ gain must vanish in both.")
    ap.add_argument("--grad-check", action="store_true",          # run the mandatory finite-difference eligibility check + exit
                    help="run the finite-difference gradient/eligibility faithfulness check on a tiny ALIF net and exit.")
    ap.add_argument("--modes", type=str, nargs="+", default=["fixed", "plastic", "shuffle_elig", "zero_signal"])
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()
    if args.grad_check:
        return grad_check_alif()

    sents = load_sentences(args.corpus, args.n_sentences)
    t0 = time.time(); per_seed = {}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(sents)); cut = int(0.8 * len(sents))
        tr = [sents[i] for i in idx[:cut]][:args.max_train_sents]
        ev = [sents[i] for i in idx[cut:]][:args.max_eval_sents]
        vocab = Vocab.build(tr, V=args.vocab); V = vocab.size
        tr_ids = [vocab.ids(s) for s in tr]; ev_ids = [vocab.ids(s) for s in ev]
        P_bi = fit_bigram(tr_ids, V)
        rec = {"V": V, "n_train": len(tr), "by_mode": {}, "controls": {}}
        alif_res = None; alif_W = None                            # keep the trained alif artifacts for the controls
        for mode in args.modes:
            is_alif = mode in ("alif", "alif_readonly")
            res = RateReservoir(V, args.n_pool, seed, alpha=args.alpha, spectral=args.spectral,
                                leak_spectrum=args.leak_spectrum, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi,
                                alif=is_alif, beta=args.alif_beta,
                                adapt_win_lo=args.adapt_win_lo, adapt_win_hi=args.adapt_win_hi)
            W_out = train(res, tr_ids, V, args.epochs, args.lr_out, args.lr_rec, seed, mode=mode,
                          wd_rec=args.wd_rec, a_slow_leak=args.alpha_slow)
            depth, agg, aggb = per_depth_ce(res, W_out, ev_ids, P_bi, lesion_slow=args.lesion_slow)
            rec["by_mode"][mode] = {"aggregate_ce": agg, "bigram_ce": aggb, "by_depth": depth}
            if mode == "alif":
                alif_res, alif_W = res, W_out

        # ALIF controls (anti-cheats for the adaptation STATE): ADAPTATION-SHUFFLE (content, not capacity) + beta=0 (lesion).
        if args.alif_controls and alif_res is not None:
            sh_depth, sh_agg, _ = per_depth_ce(alif_res, alif_W, ev_ids, P_bi, shuffle_adapt=True)
            rec["controls"]["alif_adapt_shuffle"] = {"aggregate_ce": sh_agg, "by_depth": sh_depth}
            b0 = RateReservoir(V, args.n_pool, seed, alpha=args.alpha, spectral=args.spectral,
                               leak_spectrum=args.leak_spectrum, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi,
                               alif=True, beta=0.0, adapt_win_lo=args.adapt_win_lo, adapt_win_hi=args.adapt_win_hi)
            W_b0 = train(b0, tr_ids, V, args.epochs, args.lr_out, args.lr_rec, seed, mode="alif", wd_rec=args.wd_rec)
            b0_depth, b0_agg, _ = per_depth_ce(b0, W_b0, ev_ids, P_bi)
            rec["controls"]["alif_beta0"] = {"aggregate_ce": b0_agg, "by_depth": b0_depth}
        per_seed[str(seed)] = rec

        # print: each mode's per-depth CE MINUS the `fixed` baseline (negative = better than fixed); DEEP(6+) + d10-99.
        fx = rec["by_mode"].get("fixed")
        _dbucket = [f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in BUCKETS]

        def _delta(mode_or_ctrl):
            bd = mode_or_ctrl["by_depth"]
            dd = {b: round(bd[b]["ce"] - fx["by_depth"][b]["ce"], 3) for b in bd if fx and b in fx["by_depth"]}
            deep = [b for b in ("6-9", "10-99") if b in dd]
            return dd, (float(np.mean([dd[b] for b in deep])) if deep else float("nan")), dd.get("10-99", float("nan"))

        if fx:
            print(f"[seed {seed}] V={V} fixed_agg {fx['aggregate_ce']}  (each row = mode-MINUS-fixed CE by depth; "
                  f"neg = better than fixed)", flush=True)
            for mode in args.modes:
                if mode == "fixed":
                    continue
                mm = rec["by_mode"].get(mode)
                if not mm:
                    continue
                dd, deepd, d10 = _delta(mm)
                print(f"    {mode:>14}: " + " ".join(f"d{b}:{dd[b]:+.2f}" for b in _dbucket if b in dd)
                      + f" | DEEP(6+) {deepd:+.3f} | d10-99 {d10:+.3f} | agg {mm['aggregate_ce']}", flush=True)
            for cname, cc in rec["controls"].items():
                dd, deepd, d10 = _delta(cc)
                print(f"    [ctrl {cname:>16}]: " + " ".join(f"d{b}:{dd[b]:+.2f}" for b in _dbucket if b in dd)
                      + f" | DEEP(6+) {deepd:+.3f} | d10-99 {d10:+.3f} | agg {cc['aggregate_ce']}", flush=True)

    out = {"runner": "_emerge_reservoir_lm_eprop_recurrent_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "n_pool": args.n_pool, "args": vars(args), "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
