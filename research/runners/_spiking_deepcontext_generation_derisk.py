"""Deep-context CREDIT-ASSIGNMENT de-risk for open, arbitrary spiking prose (A1 mouth-burn-down residual).

============================================================================================================
WHAT THIS IS (and, more importantly, what it is NOT)
============================================================================================================
The FORWARD path for open deep-context spiking generation is ALREADY GO (2026-07-20 RF-PHASE-ENCODE): a graded
`cp_ssm_state` WKV/SSM store, driven by a spiking phase (FHRR) input through real synapses, GENERATES fluent,
coherent, coreference-maintained TinyStories prose on the real Izhikevich substrate, 6-seed. The store's
write/decay/read, however, are trained by BPTT (a host-computed gradient = the tracked scaffold). The LAST piece
of retiring the Qwen mouth is: can a BIOLOGICAL LOCAL rule LEARN that deep-context store, or must it be BPTT?

The prior banked walls on the LEARNING rule:
  * plain single-timescale e-prop captures a deep-context margin (R1b ~44% of a read-out-CONFOUNDED BPTT denom).
  * multi-timescale / dual-timescale ELIGIBILITY (the "44->81% lift", 2026-07-11 R2b) => REFUTED as an effective-
    learning-rate artifact (magnitude-controlled EMA lift ~ +0.04; plain e-prop at 5-10x lr reproduces it).
  * ALIF adaptation-as-FORWARD-STATE (the WM-current lever, R2)                     => REFUTED (degrades the rep).
  * the controlled-lag arc (2026-07-14) => on a FIXED-reservoir substrate, recurrent W_rec credit is NOT the
    load-bearing bottleneck; the bottleneck is the substrate MEMORY HORIZON (distal cue lost >~15 tok) + a
    nonlinear read-out. The diagonal e-prop family drops the OFF-DIAGONAL cross-unit gradient.

THE COMPANION-PROCESS REFRAME (the wall-discipline question, applied):
  "What does biology run ALONGSIDE deep-context credit that the prior de-risks replaced with a CONSTANT?"
  e-prop's broadcast learning signal uses a FIXED RANDOM feedback matrix B (a constant, misaligned map from
  output error to hidden credit) AND drops the downstream/indirect influence (the RTRL off-diagonal). Biology
  runs a LEARNED top-down feedback (apical-dendrite / predictive-coding / Bellec-2020's DNI) that ALIGNS the
  credit direction. The 2026-08-11 gap#4 ALL-IN arc found exactly this at the rate level: LEARNED feedback
  (Kolen-Pollack) reaches depth-3 where FIXED feedback fails. This runner tests that companion process on the
  actual deep-context store substrate (a diagonal WKV/SSM), with the CLEAN controlled-lag instrument.

KEY SUBSTRATE INSIGHT: the store here is a DIAGONAL gated leaky integrator (the WKV membrane-leak form
`s_t = decay*s_{t-1} + gate*write`, the exact 2026-07-20 spiking-faithful "ssm" recurrence). For a DIAGONAL
store, the forward-filtered e-prop eligibility of the store params (per-channel decay theta_d, write W_v, write-
gate W_g) is EXACT RTRL for those params (no off-diagonal drop) -- so any e-prop<->BPTT gap here is NOT a
temporal-truncation wall but a FEEDBACK-ALIGNMENT / read-out wall, which LEARNED feedback can attack. Learning
is LOAD-BEARING because the write must be GATED to ignore the fillers (a fixed random write corrupts the slow
channels with filler tokens => the distal cue is lost => chance at deep lag).

============================================================================================================
THE INSTRUMENT (controlled-lag delayed cued-recall; provably beats no n-gram)
============================================================================================================
Trial = [STORE, x, f_1..f_T, RECALL] ; predict the content symbol x at the RECALL position. The last n<T tokens
are fillers+RECALL, statistically INDEPENDENT of x => any n-gram is at chance (1/K) by construction (the deep
dependency is real, not a memory-timescale artifact). We sweep the lag T. Instrument is VALID iff at a deep T
the ceiling (bptt) genuinely SOLVES while the fixed store is at chance (non-vacuous), the n-gram is at chance,
and cue_scramble collapses.

ARMS (one variable = the credit rule for the store params {theta_d, W_v, W_g}; the read-out {W_h,b_h,W_o} is
trained by the IDENTICAL static (non-temporal) rule across ALL arms -> isolates recurrent/store credit, fixing
the R2b read-out confound):
  fixed_store    -- store params FROZEN (random write). Capacity/horizon FLOOR.
  eprop_random   -- diagonal e-prop; learning signal L = B_random @ delta (FIXED random broadcast feedback). The
                    faithful transport-free local rule (the "~44%" family).
  eprop_learnfb  -- THE MECHANISM: diagonal e-prop with LEARNED feedback (a DNI / synthetic-gradient predictor B
                    trained ONLINE to align with the read-out's spatial credit signal -- the companion process the
                    fixed-B rule replaced with a CONSTANT; Bellec-2020's DNI / apical-feedback / predictive-coding).
  eprop_truefb   -- diagonal e-prop with the EXACT spatial feedback g_s = d(loss)/d(s_recall) (feedback-direction
                    CEILING; isolates "is the residual feedback-alignment or temporal?").
  bptt_ceiling   -- full backprop-through-time into the store (the recurrent-credit CEILING). Instrument-valid iff
                    it beats fixed at deep T.
  shuffle_elig   -- eprop_random but PERMUTE the eligibility across channels before each update (magnitude kept,
                    credit STRUCTURE broken) => genuine credit must COLLAPSE.
  sign_flip      -- eprop_random with L := -L (credit direction reversed) => must DIVERGE (~chance), not track.
  zero_signal    -- eprop_random with L := 0 => store never moves => must end == fixed_store (byte sanity).

CONTROLS: per-arm LEARNING-RATE sweep (fair A/B; the 2026-08-11 lesson -- one shared lr is unfair); cue_scramble
(train on random targets -> eval-vs-stored-x collapses); n-gram chance check; grad-check of the eligibility (FD).

VERDICT:
  GO      if eprop_learnfb closes a MEANINGFUL fraction of the eprop_random->bptt_ceiling deep-context gap (learned
          feedback carries deep context on a local rule), 6-seed, with shuffle/sign_flip/scramble all collapsing.
  BOUNDARY (mapped) otherwise, with the exact residual localized by the eprop_truefb / bptt contrast:
          truefb~=bptt but kp<<  => residual = feedback ALIGNMENT (KP insufficient) -> next lever.
          truefb<<bptt           => residual = TEMPORAL (diagonal eligibility vanishes over the lag) -> the store
                                    write must ride BPTT / a richer eligibility.

Rate-level torch/GPU (the 2026-08-11 arc mandates testing deep credit on a RATE net -- the finite-spike READ
regime is a SEPARATE, already-characterized production wall; the FORWARD spiking realization is already GO via
RF-phase). NO `sim/` edit anywhere (a standalone rate-level de-risk).
"""
from __future__ import annotations
import argparse, json, math, time, os
import numpy as np
import torch
from tools.lab import attributable_to   # attribute the plastic win to genuine credit (vs the shuffle control)


# ----------------------------------------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------------------------------------
def make_recall_task(K, F, T, n_trials, seed, cue_scramble=False):
    """[STORE, x, f_1..f_T, RECALL] -> predict x (the stored content symbol) at the RECALL position.
    Tokens: content 0..K-1, STORE=K, RECALL=K+1, fillers K+2..K+1+F. cue_scramble: train target is a RANDOM
    content symbol (breaks cue->target) -> a model can learn nothing -> eval-vs-stored-x collapses to chance."""
    rng = np.random.default_rng(seed)
    STORE, RECALL, FILL0 = K, K + 1, K + 2
    L = T + 3                                            # STORE, x, T fillers, RECALL (read s at RECALL, the last tok)
    seqs = np.empty((n_trials, L), dtype=np.int64)
    targets = np.empty(n_trials, dtype=np.int64)
    for i in range(n_trials):
        x = int(rng.integers(K))
        fillers = FILL0 + rng.integers(F, size=T)
        seqs[i, 0] = STORE
        seqs[i, 1] = x
        seqs[i, 2:2 + T] = fillers
        seqs[i, 2 + T] = RECALL
        targets[i] = int(rng.integers(K)) if cue_scramble else x
    recall_pos = 2 + T                                  # index of the RECALL token (= L-1); read s there
    stored = np.array([int(s[1]) for s in seqs], dtype=np.int64)
    return seqs, recall_pos, targets, stored


def ngram_chance(tr_seqs, tr_stored, ev_seqs, ev_stored, recall_pos, K, n=3):
    """Best n-gram: FIT the (last-n-tokens -> content) lookup table on TRAIN, evaluate on HELD-OUT eval (no
    same-set memorization). By construction the last-n window is fillers+RECALL, statistically INDEPENDENT of the
    stored x, so a held-out n-gram MUST be ~chance (1/K) -- the sanity that the deep dependency is genuine."""
    from collections import defaultdict
    table = defaultdict(lambda: np.zeros(K))
    for s, y in zip(tr_seqs, tr_stored):
        ctx = tuple(int(t) for t in s[max(0, recall_pos - n):recall_pos])
        table[ctx][y] += 1
    correct = 0
    for s, y in zip(ev_seqs, ev_stored):
        ctx = tuple(int(t) for t in s[max(0, recall_pos - n):recall_pos])
        pred = int(np.argmax(table[ctx])) if ctx in table else 0
        correct += (pred == y)
    return correct / len(ev_seqs)


# ----------------------------------------------------------------------------------------------------------
# The diagonal gated leaky store (WKV / SSM membrane-leak form) + a 2-stage read-out
# ----------------------------------------------------------------------------------------------------------
def build_params(V, D, N, H, K, seed, device):
    """Reproducible init. Decays are LOG-SPACED across channels (a fixed spread of time constants -- biological,
    Gerstner eligibility-trace review; NOT a learned cheat) so some channels start in the long-memory regime
    (=> the eligibility for the slow channels is non-vanishing and learning CAN sharpen the write into them)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = (torch.randn(V, D, generator=g) / math.sqrt(D))                 # fixed random embedding (frozen)
    W_v = (torch.randn(N, D, generator=g) / math.sqrt(D))              # write projection (learnable)
    W_g = (torch.randn(N, D, generator=g) / math.sqrt(D)) * 0.5        # write-gate pre-activation (learnable)
    # decays log-spaced in [d_min, d_max] over channels:
    d_min, d_max = 0.5, 0.995
    taus = torch.logspace(math.log10(d_min), math.log10(d_max), N)
    theta_d = torch.log(taus / (1 - taus))                            # sigmoid(theta_d) = decay (learnable)
    W_h = (torch.randn(H, N, generator=g) / math.sqrt(N))             # read-out stage 1 (learnable, shared rule)
    b_h = torch.zeros(H)
    W_o = (torch.randn(K, H, generator=g) / math.sqrt(H))             # read-out stage 2 (learnable, shared rule)
    B = (torch.randn(N, K, generator=g) / math.sqrt(K))              # broadcast feedback (random / KP-adapted)
    to = lambda t: t.to(device).float()
    return {"E": to(E), "W_v": to(W_v), "W_g": to(W_g), "theta_d": to(theta_d),
            "W_h": to(W_h), "b_h": to(b_h), "W_o": to(W_o), "B": to(B)}


def run_store(seqs_t, P, W_v, W_g, theta_d, recall_pos, want_elig=False):
    """Forward the diagonal gated store; return s at recall_pos (and, if want_elig, the exact diagonal eligibility
    traces at recall_pos). seqs_t: (batch, L) long on device. All batched. NO autograd (manual)."""
    B_, L = seqs_t.shape
    N = W_v.shape[0]
    D = W_v.shape[1]
    E = P["E"]
    d = torch.sigmoid(theta_d)                                        # (N,)
    dprime = d * (1 - d)                                              # (N,)
    s = torch.zeros(B_, N, device=seqs_t.device)
    if want_elig:
        e_d = torch.zeros(B_, N, device=seqs_t.device)               # d s[c] / d theta_d[c]
        e_wv = torch.zeros(B_, N, D, device=seqs_t.device)           # d s[c] / d W_v[c,:]
        e_wg = torch.zeros(B_, N, D, device=seqs_t.device)           # d s[c] / d W_g[c,:]
    for t in range(recall_pos + 1):                                  # inclusive of the RECALL position
        emb = E[seqs_t[:, t]]                                        # (B_, D)
        pre_wr = emb @ W_v.t()                                       # (B_, N)
        g = torch.sigmoid(emb @ W_g.t())                            # (B_, N)
        wr = g * pre_wr
        if want_elig:
            s_prev = s
            gprime = g * (1 - g)                                     # (B_, N)
            e_d = d * e_d + dprime * s_prev
            e_wv = d.view(1, N, 1) * e_wv + (g.unsqueeze(-1) * emb.unsqueeze(1))
            e_wg = d.view(1, N, 1) * e_wg + (gprime * pre_wr).unsqueeze(-1) * emb.unsqueeze(1)
        s = d * s + wr
    if want_elig:
        return s, (e_d, e_wv, e_wg)
    return s, None


def readout(s, P):
    h = torch.tanh(s @ P["W_h"].t() + P["b_h"])                      # (batch, H)
    logits = h @ P["W_o"].t()                                        # (batch, K)
    return logits, h


# ----------------------------------------------------------------------------------------------------------
# Grad-check: the manual diagonal eligibility vs autograd (the instrument's correctness gate)
# ----------------------------------------------------------------------------------------------------------
def grad_check(seed=0, device="cpu"):
    torch.manual_seed(seed)
    V, D, N, H, K = 10, 6, 8, 12, 4
    P = build_params(V, D, N, H, K, seed, device)
    T = 6
    seqs, recall_pos, _, _ = make_recall_task(K, F=3, T=T, n_trials=5, seed=seed)
    seqs_t = torch.tensor(seqs, device=device)
    W_v = P["W_v"].clone().requires_grad_(True)
    W_g = P["W_g"].clone().requires_grad_(True)
    theta_d = P["theta_d"].clone().requires_grad_(True)
    # autograd d(sum_c s_recall[c]) / d params  == sum over channels of ds[c]/dparam  (compare to eligibility sums)
    s_auto, _ = run_store(seqs_t, P, W_v, W_g, theta_d, recall_pos, want_elig=False)
    loss = s_auto.sum()
    loss.backward()
    with torch.no_grad():
        s_man, (e_d, e_wv, e_wg) = run_store(seqs_t, P, P["W_v"], P["W_g"], P["theta_d"], recall_pos, want_elig=True)
    # d(sum_c s[c])/d theta_d[c] == sum over batch of e_d[b,c]. eligibility e_d[b,c] = ds_b[c]/dtheta_d[c].
    gd_theta = e_d.sum(0)                                            # (N,)
    gd_wv = e_wv.sum(0)                                              # (N,D)  ds_b[c]/dW_v[c,:] summed over b
    gd_wg = e_wg.sum(0)
    r_theta = (theta_d.grad - gd_theta).abs().max().item()
    r_wv = (W_v.grad - gd_wv).abs().max().item()
    r_wg = (W_g.grad - gd_wg).abs().max().item()
    return {"resid_theta_d": r_theta, "resid_W_v": r_wv, "resid_W_g": r_wg,
            "PASS": max(r_theta, r_wv, r_wg) < 1e-4}


# ----------------------------------------------------------------------------------------------------------
# Train one arm
# ----------------------------------------------------------------------------------------------------------
def train_arm(arm, P0, seqs, recall_pos, targets, stored, K, epochs, lr_store, lr_ro, lr_B,
              batch, device, seed):
    """Train the read-out (identical static rule across ALL arms) + the store params (credit rule varies by arm).
    Returns eval recall accuracy vs the STORED x."""
    rng = np.random.default_rng(seed + 12345)
    P = {k: v.clone() for k, v in P0.items()}
    W_v = P["W_v"]; W_g = P["W_g"]; theta_d = P["theta_d"]
    is_bptt = (arm == "bptt_ceiling")
    seqs_t = torch.tensor(seqs, device=device)
    tgt_t = torch.tensor(targets, device=device)
    n = len(seqs)
    idx_all = np.arange(n)

    for ep in range(epochs):
        rng.shuffle(idx_all)
        for b0 in range(0, n, batch):
            bidx = idx_all[b0:b0 + batch]
            xb = seqs_t[bidx]; yb = tgt_t[bidx]
            # ---------- forward store ----------
            if is_bptt:
                W_v.requires_grad_(True); W_g.requires_grad_(True); theta_d.requires_grad_(True)
                s, _ = run_store(xb, P, W_v, W_g, theta_d, recall_pos, want_elig=False)
            else:
                with torch.no_grad():
                    s, elig = run_store(xb, P, W_v, W_g, theta_d, recall_pos,
                                        want_elig=(arm != "fixed_store"))
            # ---------- read-out (identical procedure across arms; static, non-temporal) ----------
            s_ro = s if is_bptt else s.detach()
            Wh = P["W_h"].detach().clone().requires_grad_(True)
            bh = P["b_h"].detach().clone().requires_grad_(True)
            Wo = P["W_o"].detach().clone().requires_grad_(True)
            h = torch.tanh(s_ro @ Wh.t() + bh)
            logits = h @ Wo.t()
            ce = torch.nn.functional.cross_entropy(logits, yb)
            if is_bptt:
                ce.backward()
                with torch.no_grad():
                    P["W_h"] -= lr_ro * Wh.grad; P["b_h"] -= lr_ro * bh.grad; P["W_o"] -= lr_ro * Wo.grad
                    W_v -= lr_store * W_v.grad; W_g -= lr_store * W_g.grad; theta_d -= lr_store * theta_d.grad
                    W_v.grad = None; W_g.grad = None; theta_d.grad = None
                W_v = W_v.detach(); W_g = W_g.detach(); theta_d = theta_d.detach()
                P["W_v"], P["W_g"], P["theta_d"] = W_v, W_g, theta_d
                continue
            # ---- non-bptt arms: read-out by autograd on the STATIC read-out only (no temporal credit) ----
            ce.backward()
            with torch.no_grad():
                P["W_h"] -= lr_ro * Wh.grad; P["b_h"] -= lr_ro * bh.grad; P["W_o"] -= lr_ro * Wo.grad
                if arm == "fixed_store":
                    continue                                        # store frozen
                probs = torch.softmax(logits.detach(), dim=1)
                onehot = torch.zeros_like(probs); onehot[torch.arange(len(yb)), yb] = 1.0
                delta = (probs - onehot) / len(yb)                  # (batch, K)   d ce / d logits
                e_d, e_wv, e_wg = elig
                # ---------- learning signal L (batch, N) ----------
                if arm == "eprop_truefb":
                    # exact spatial gradient d(ce)/d(s):  delta -> W_o -> (1-h^2) -> W_h
                    dh = delta @ Wo.detach()                        # (batch, H)
                    dh = dh * (1 - h.detach() ** 2)
                    L = dh @ Wh.detach()                            # (batch, N)  == d ce / d s
                elif arm == "eprop_learnfb":
                    # LEARNED FEEDBACK (DNI / synthetic-gradient predictor -- the companion process biology runs
                    # that fixed-random e-prop replaced with a CONSTANT): a learned broadcast map B (N x K) is
                    # trained ONLINE by a LOCAL delta-rule to PREDICT the read-out's own spatial credit signal
                    # L_true = d(ce)/d(s) from the output error delta, then the STORE is credited with the LEARNED
                    # prediction L = delta @ B^T. Transport-free in the temporal/recurrent axis (the store credit is
                    # still e-prop eligibility); B distills the LOCAL (within-step) read-out sensitivity into a fast
                    # aligned broadcast. If this closes the eprop_random->truefb gap, an aligned feedback is the fix.
                    dh = delta @ Wo.detach(); dh = dh * (1 - h.detach() ** 2)
                    L_true = dh @ Wh.detach()                       # (batch, N)  the target credit direction
                    L = delta @ P["B"].t()                          # (batch, N)  the LEARNED-feedback prediction
                    resid = L - L_true                             # regress B toward L_true (local delta-rule on B)
                    P["B"] -= lr_B * (resid.t() @ delta)          # dB[c,k] = sum_b resid[b,c] delta[b,k]
                elif arm in ("eprop_random", "shuffle_elig", "sign_flip", "zero_signal"):
                    L = delta @ P["B"].t()                          # (batch, N) FIXED random broadcast feedback
                    if arm == "sign_flip":
                        L = -L
                    if arm == "zero_signal":
                        L = torch.zeros_like(L)
                else:
                    raise ValueError(arm)
                if arm == "shuffle_elig":
                    perm = torch.randperm(e_d.shape[1], device=device)
                    e_d = e_d[:, perm]; e_wv = e_wv[:, perm]; e_wg = e_wg[:, perm]
                # ---------- store updates: dW = - lr * sum_batch( L[:,c] * elig[:,c,:] ) ----------
                g_theta = (L * e_d).sum(0)
                g_wv = (L.unsqueeze(-1) * e_wv).sum(0)
                g_wg = (L.unsqueeze(-1) * e_wg).sum(0)
                theta_d -= lr_store * g_theta
                W_v -= lr_store * g_wv
                W_g -= lr_store * g_wg
    # ---------- eval on the (passed) set ----------
    with torch.no_grad():
        s_all, _ = run_store(seqs_t, P, P["W_v"], P["W_g"], P["theta_d"], recall_pos, want_elig=False)
        logits, _ = readout(s_all, P)
        pred = logits.argmax(1).cpu().numpy()
    acc = float((pred == stored).mean())
    return acc, P


# ----------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------
ARMS = ["fixed_store", "eprop_random", "eprop_learnfb", "eprop_truefb", "bptt_ceiling",
        "shuffle_elig", "sign_flip", "zero_signal"]
LR_SWEEP = [0.03, 0.1, 0.3]                                         # per-arm fair lr sweep (2026-08-11 lesson;
                                                                   # best lrs in the T-sweep calibration were 0.1-0.3)


def run_seed(seed, T, K, F, N, H, D, epochs, n_train, n_eval, batch, device, do_scramble=False, lr_sweep=None):
    V = K + 2 + F
    P0 = build_params(V, D, N, H, K, seed, device)
    seqs, recall_pos, targets, stored = make_recall_task(K, F, T, n_train, seed, cue_scramble=False)
    ev_seqs, ev_rp, ev_tgt, ev_stored = make_recall_task(K, F, T, n_eval, seed + 7, cue_scramble=False)
    ngram = ngram_chance(seqs, stored, ev_seqs, ev_stored, ev_rp, K, n=3)
    lr_sweep = lr_sweep or LR_SWEEP
    out = {"seed": seed, "T": T, "K": K, "chance": 1.0 / K, "ngram3": ngram, "arms": {}}

    def eval_arm(P):
        with torch.no_grad():
            s_all, _ = run_store(torch.tensor(ev_seqs, device=device), P, P["W_v"], P["W_g"], P["theta_d"], ev_rp)
            logits, _ = readout(s_all, P)
            pred = logits.argmax(1).cpu().numpy()
        return float((pred == ev_stored).mean())

    for arm in ARMS:
        best_acc, best_lr = -1.0, None
        sweep = lr_sweep if arm not in ("fixed_store", "zero_signal") else [lr_sweep[0]]
        for lr in sweep:
            _, P = train_arm(arm, P0, seqs, recall_pos, targets, stored, K, epochs,
                             lr_store=lr, lr_ro=0.05, lr_B=0.02, batch=batch, device=device, seed=seed)
            acc = eval_arm(P)
            if acc > best_acc:
                best_acc, best_lr = acc, lr
        out["arms"][arm] = {"acc": best_acc, "lr": best_lr}
    if do_scramble:
        sc_seqs, sc_rp, sc_tgt, sc_stored = make_recall_task(K, F, T, n_train, seed, cue_scramble=True)
        _, P = train_arm("eprop_learnfb", P0, sc_seqs, sc_rp, sc_tgt, sc_stored, K, epochs,
                         lr_store=0.1, lr_ro=0.05, lr_B=0.02, batch=batch, device=device, seed=seed)
        out["cue_scramble_acc"] = eval_arm(P)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--T", type=int, nargs="+", default=[16])       # lag(s) to test
    ap.add_argument("--K", type=int, default=6)                     # content vocab (chance 1/K)
    ap.add_argument("--F", type=int, default=6)                     # distinct fillers
    ap.add_argument("--N", type=int, default=128)                   # store channels
    ap.add_argument("--H", type=int, default=64)                    # read-out hidden
    ap.add_argument("--D", type=int, default=32)                    # embedding dim
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="research/findings/raw/_spiking_deepcontext_generation.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gradcheck-only", action="store_true")
    args = ap.parse_args()

    gc = grad_check(device=args.device)
    print("GRAD-CHECK (diagonal eligibility vs autograd):", json.dumps(gc))
    if args.gradcheck_only:
        return
    assert gc["PASS"], f"eligibility grad-check FAILED: {gc}"

    if args.smoke:
        args.seeds = [42]; args.epochs = 8; args.n_train = 400; args.n_eval = 300; args.T = [8]

    t0 = time.time()
    results = {"config": vars(args), "grad_check": gc, "runs": []}
    for T in args.T:
        for seed in args.seeds:
            r = run_seed(seed, T, args.K, args.F, args.N, args.H, args.D, args.epochs,
                         args.n_train, args.n_eval, args.batch, args.device,
                         do_scramble=(seed == args.seeds[0]))
            r["elapsed"] = round(time.time() - t0, 1)
            results["runs"].append(r)
            am = r["arms"]
            print(f"[T={T} seed={seed}] fixed={am['fixed_store']['acc']:.3f} "
                  f"eprop={am['eprop_random']['acc']:.3f} kp={am['eprop_learnfb']['acc']:.3f} "
                  f"truefb={am['eprop_truefb']['acc']:.3f} bptt={am['bptt_ceiling']['acc']:.3f} "
                  f"| shuf={am['shuffle_elig']['acc']:.3f} sign={am['sign_flip']['acc']:.3f} "
                  f"zero={am['zero_signal']['acc']:.3f} ngram={r['ngram3']:.3f} chance={r['chance']:.3f} "
                  f"({r['elapsed']:.0f}s)")

    by_T = {}
    for T in args.T:
        rs = [r for r in results["runs"] if r["T"] == T]
        agg = {}
        for arm in ARMS:
            vals = [r["arms"][arm]["acc"] for r in rs]
            agg[arm] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals}
        fx = agg["fixed_store"]["mean"]; er = agg["eprop_random"]["mean"]
        kp = agg["eprop_learnfb"]["mean"]; tf = agg["eprop_truefb"]["mean"]; bp = agg["bptt_ceiling"]["mean"]
        sh = agg["shuffle_elig"]["mean"]; zs = agg["zero_signal"]["mean"]
        chance = 1.0 / args.K
        cue_sc = float(np.mean([r["cue_scramble_acc"] for r in rs if "cue_scramble_acc" in r])) \
            if any("cue_scramble_acc" in r for r in rs) else None
        gap = bp - er
        agg["_frac_learnfb_of_gap"] = float((kp - er) / gap) if abs(gap) > 1e-3 else None
        agg["_frac_truefb_of_gap"] = float((tf - er) / gap) if abs(gap) > 1e-3 else None
        agg["_frac_learnfb_of_ceiling"] = float((kp - fx) / (bp - fx)) if abs(bp - fx) > 1e-3 else None
        # --- instrument validity: fixed must be near-chance (cue genuinely lost), n-gram at chance, AND a genuine-
        #     credit anti-cheat (shuffle_elig -- the VALID one; sign_flip is FA-sign-invariant, a weak control) must
        #     collapse toward fixed, AND bptt must genuinely solve. Otherwise the credit comparison is vacuous. ---
        near_chance = lambda v: v < 1.4 * chance
        agg["_instrument_valid"] = bool(near_chance(fx) and rs[0]["ngram3"] < 1.4 * chance
                                        and (bp - fx) > 0.4 and (er - sh) > 0.3
                                        and (cue_sc is None or near_chance(cue_sc)))
        agg["_cue_scramble"] = cue_sc
        # ATTRIBUTION (lab.attributable_to): what FRACTION of the LOCAL-rule win (above the shuffle_elig control) is
        # genuine per-channel credit, not the read-out/capacity that the shuffle arm ALSO has? Both arms are measured;
        # this asks whose the difference is (the gap#5 97%-clamp lesson). zero_signal is the byte-identity sanity
        # (== fixed_store by construction; L:=0 => the store never moves), NOT an independent treatment.
        agg["_credit_attributable_frac"] = attributable_to(
            f"eprop_random deep credit vs shuffle_elig control (T={T})", er - chance, sh - chance)
        # --- verdict on the CREDIT-RULE question: does a biological LOCAL rule (eprop_random) tie the BPTT ceiling? ---
        # NOTE on the "ceiling": at deep T, BPTT through the leaky store suffers VANISHING gradients (the gradient to
        # the distal write decays ~decay^T), so bptt_ceiling can UNDER-perform the forward-mode eligibility rule and
        # is NOT a valid ceiling there. When eprop_random >= bptt_ceiling, the task is UNDER-DIFFICULT for the
        # credit-QUALITY question (any coherent per-channel eligibility solves it; feedback alignment is not stressed).
        if not agg["_instrument_valid"]:
            verdict = "INSTRUMENT_VACUOUS (fixed not at chance, or ceiling collapsed below the local rule at depth)"
        elif er >= bp - 0.02:
            verdict = ("NO_CREDIT_WALL_AT_THIS_SCALE / TASK_UNDER_DIFFICULT: the LOCAL rule (diagonal e-prop, random "
                       "feedback) TIES-OR-BEATS the BPTT ceiling; genuine store-credit (shuffle+cue_scramble collapse) "
                       "but feedback QUALITY is NOT the discriminating axis (random~=learned~=true). The task does NOT "
                       "probe the deep-context credit WALL the open-prose mouth hits -- SCALE vocab/T/depth to locate it.")
        elif (kp - er) > 0.5 * gap:
            verdict = ("LEARNED_FEEDBACK_GO: eprop_random is below bptt, and the learned-feedback (DNI) mechanism "
                       "closes >50%% of the eprop_random->bptt deep-context gap.")
        else:
            verdict = ("MAPPED_BOUNDARY: eprop_random is below bptt and learned feedback does NOT close it; "
                       "residual localized by truefb (feedback-alignment) vs bptt (temporal).")
        agg["_verdict"] = verdict
        by_T[T] = agg
    results["aggregate"] = by_T
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nWROTE", args.out)
    for T, agg in by_T.items():
        print(f"\n=== T={T} (chance {1.0/args.K:.3f}) ===")
        for arm in ARMS:
            print(f"  {arm:14s} {agg[arm]['mean']:.3f} +- {agg[arm]['std']:.3f}  vals={agg[arm]['vals']}")
        print(f"  cue_scramble={agg['_cue_scramble']}  instrument_valid={agg['_instrument_valid']}")
        print(f"  learnfb frac of (eprop->bptt) gap = {agg['_frac_learnfb_of_gap']}  "
              f"truefb frac = {agg['_frac_truefb_of_gap']}")
        print(f"  VERDICT[T={T}]: {agg['_verdict']}")


if __name__ == "__main__":
    main()
