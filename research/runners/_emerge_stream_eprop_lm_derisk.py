"""STREAM e-prop LM de-risk -- can BIOLOGICAL one-step-local recurrent credit capture the LONG-RANGE next-token
structure that FULL BACKPROP captures, on a CONTIGUOUS text stream?

WHY THIS RUNNER EXISTS. The committed e-prop reservoir runner
(`_emerge_reservoir_lm_eprop_recurrent_derisk.py`) trains WITHIN a sentence: the recurrent state + eligibility reset at
every sentence boundary (<=16-token sentences). Long-range structure lives in the DEEP within-block positions (a token
predicted from 20-100 tokens of running history), which a per-sentence reset can never reach -- so that runner literally
cannot MEASURE long-range credit. This runner fixes the regime: a CONTIGUOUS stream, block-matched to 128 tokens, state +
eligibility carried across all 128 tokens WITHIN a block and reset only at the block boundary. That makes the clean
single-variable question measurable: "e-prop-within-128 vs BPTT-within-128".

MECHANISM = e-prop (Bellec-Maass 2020, Nat Commun 11:3625), random-feedback / broadcast-alignment variant -- the RATE
analogue of the on-bridge BDSP/Burstprop (no backprop-through-time, no weight transport). For a leaky-tanh rate RNN over a
contiguous 128-token block (h + eligibility carried across the block, reset at the boundary):
  h_t   = (1-a) h_{t-1} + a * tanh( W_rec h_{t-1} + W_in x_t + b )         [state]
  p_t   = softmax( W_out h_t )                                             [read-out predicts token t+1]
  delta = onehot(target) - p_t                                            [clean read-out error]
  read-out (delta rule):     W_out += lr_out * outer(delta, h_t)
  eligibility (forward-filtered local sensitivity of h_j to W_rec[j,i]):
      psi_j  = a * (1 - act_j^2)                 [pseudo-derivative; act_j = tanh(pre_j)]
      e[j,i] = (1-a) * e[j,i] + psi_j * h_{t-1,i}
  learning signal (BROADCAST random feedback; NO weight transport, like BDSP's fixed-random apical route):
      L_j    = (B @ delta)_j        (B: n x V fixed random feedback)
  recurrent update (e-prop): W_rec[j,i] += lr_rec * L_j * e[j,i]

ARMS (one variable = whether/how W_rec learns; init + alpha + n_pool + W_in/W_rec/W_out/B seeds IDENTICAL across arms):
  fixed_reservoir -- W_rec FROZEN; train only W_out (echo-state read-out). The capacity FLOOR.
  plastic_eprop   -- W_out delta rule + W_rec via e-prop random-feedback. The mechanism UNDER TEST.
  shuffle_elig    -- like plastic but PERMUTE the eligibility entries before each W_rec update (credit structure BROKEN).
                     ANTI-CHEAT: must not beat fixed by more than a tiny eps.
  zero_signal     -- like plastic but L := 0 -> W_rec never moves -> must end BYTE-IDENTICAL to fixed_reservoir. SANITY.
  BPTT_same_net   -- the SAME leaky-tanh RNN cell as a torch autograd module, trained by full backprop truncated to the
                     128-block (AdamW + grad-clip), SAME init/params/seeds. The fraction DENOMINATOR (matched-architecture
                     full-backprop ceiling). NOTE: e-prop trains {W_rec, W_out} with a FIXED input projection W_in (the
                     reservoir-computing setup); the full-backprop ceiling naturally trains ALL params (W_in/b too) -- the
                     honest best the architecture can do with a full gradient. This asymmetry is inherent to the question.

METRIC + GATE (PRE-REGISTERED by a research-gate + adversarial-verify workflow -- do NOT change). Eval on the held-out
10% split, CE by within-block position, bucketed with the ceiling's BK = [(1,1),(2,2),(3,3),(4,8),(9,16),(17,B)]; margin
= add-1 stream-bigram CE minus model CE. Report the SHALLOW (positions 1-8) and DEEP (positions 17-127) fractions
SEPARATELY -- a strong shallow fraction must NEVER mask a weak deep one; the DEEP bucket is the entire test:
    frac_deep    = plastic_eprop deep_margin / BPTT_same_net deep_margin
    frac_shallow = plastic_eprop shallow_margin / BPTT_same_net shallow_margin
GATE (verdict computed ONLY after the hard anti-cheats pass):
  hard anti-cheats: zero_signal W_rec == fixed_reservoir W_rec (byte-identical); shuffle_elig deep margin <= fixed deep
                    margin + eps.
  frac_deep >= 0.50 AND plastic deep margin > fixed deep floor AND monotone-growing with depth  => GO
  0.25 <= frac_deep < 0.50                                                                        => PARTIAL
        (named next lever = ALIF adaptation-as-state horizon extension, Bellec-2020)
  frac_deep < 0.25 while BPTT_same_net deep margin is clearly positive                            => BOUNDARY
        (honest: forward-eligibility credit can't reach within-block deep dependencies)

PRE-REGISTERED EXPECTATION (stated up front, not a post-hoc rationalization): the literature -- Bellec 2020 (e-prop needed
a synthetic-gradient DNI to reach BPTT on long-range PTB), Murray 2019 (RFLO matches BPTT only "when the number of
timesteps is not too great"), and the fact that e-prop is a diagonal (RTRL) truncation dropping the off-diagonal
recurrent Jacobian -- predicts plastic_eprop captures MOST of the SHALLOW margin and only a SMALL fraction of the DEEP
margin. A shallow-only capture is an HONEST NEGATIVE on long-range and a first-class deliverable, NOT something to inflate.

Reuse-by-import (`_recurrent_lm_ceiling.load_stream`/`build_vocab`, its BK buckets + stream-bigram + contiguous-lane
batching + by-within-block-position eval; the e-prop rule structure + finite-difference grad-check pattern from
`_emerge_reservoir_lm_eprop_recurrent_derisk`). NO `sim/` edit. GPU/torch (the O(n^2)-per-token eligibility over millions
of tokens is why this must be batched over lanes on GPU, not the CPU numpy loop).
"""
import os, argparse, json, math, time
from pathlib import Path
import numpy as np

from research.runners._recurrent_lm_ceiling import load_stream, build_vocab

OUT = Path("research/findings/raw/_stream_eprop_lm.json")

# ---- The e-prop reservoir arms (W_rec/W_out learn or not); BPTT is the matched full-backprop ceiling. ----
RESERVOIR_ARMS = ["fixed_reservoir", "plastic_eprop", "shuffle_elig", "zero_signal"]
ALL_ARMS = RESERVOIR_ARMS + ["BPTT_same_net"]          # the 5 pre-registered arms (the gate is defined over exactly these)
# ADDITIVE ALIF arms (NOT in ALL_ARMS -- the pre-registered 5-arm gate stays byte-identical). The single highest-leverage
# biological lever (Bellec-2020 "highways into the future"): a slow per-unit ADAPTATION-as-state whose 2-component
# eligibility propagates credit forward over long spans, so e-prop can reach the DEEP within-block context the plain
# forward-filtered (~1/alpha) eligibility cannot. Ported faithfully from `_emerge_reservoir_lm_eprop_recurrent_derisk`.
ALIF_ARMS = ["plastic_eprop_alif", "plastic_eprop_alif_readonly"]
EXTRA_ARMS = ["BPTT_fixed_win", "plastic_eprop_dualtc"] + ALIF_ARMS   # isolation/lever arms (R1b BPTT-fixed-W_in; R2b dual-timescale eligibility; R2 ALIF)
KNOWN_ARMS = ALL_ARMS + EXTRA_ARMS


# ======================================================================================================================
# Shared leaky-tanh forward + e-prop eligibility (used by BOTH torch training AND the grad-check -- so the finite-diff
# check validates the exact code path the training runs).
# ======================================================================================================================
def rnn_forward_step(h_prev, x_ids, W_rec, W_in, b, alpha):
    """One leaky-tanh step (batched). h_prev: (batch,n); x_ids: (batch,) long. Returns (h, act) where act = tanh(pre)."""
    import torch
    rec = h_prev @ W_rec.t()                     # rec[b,j] = sum_i W_rec[j,i] h_prev[b,i]
    pre = rec + W_in[:, x_ids].t() + b           # + input projection (column x per lane) + bias
    act = torch.tanh(pre)
    h = (1.0 - alpha) * h_prev + alpha * act     # leaky-integrated state
    return h, act


def elig_update(e, act, h_prev, alpha):
    """Forward-filtered e-prop eligibility (batched). e: (batch,n,n) with e[b,j,i] ~ d h_{b,j} / d W_rec[j,i].
       psi_j = alpha*(1-act_j^2) is the leaky-tanh pseudo-derivative; the increment is the outer product psi (x) h_prev."""
    psi = alpha * (1.0 - act * act)              # (batch,n)
    return (1.0 - alpha) * e + psi.unsqueeze(2) * h_prev.unsqueeze(1)   # (batch,n,n)


# ---- ALIF adaptation-as-state forward + FAITHFUL 2-component eligibility (Bellec-2020 e-prop ALIF), batched torch port
#      of `_train_alif`/`_alif_trace` in the reservoir reference. `a`(=ad) is a per-unit NON-fading slow trace of the
#      unit's OWN activity (rho_j near 1); it subtracts an "activity-silent negative imprint" (-beta*a) from the pre-
#      activation and is READ by the read-out (feature = concat([h, a])). Used by BOTH training AND the grad-check so the
#      finite-difference check validates the exact code path.
def alif_forward_step(h_prev, ad_prev, x_ids, W_rec, W_in, b, alpha, rho, beta):
    """One ALIF step (batched). h_prev,ad_prev: (batch,n); rho: (n,); alpha,beta scalar. Returns (h, ad, act) where
       ad = a_t = rho*a_{t-1} + (1-rho)*h_{t-1} (uses h_prev), and act = tanh(pre) with pre = rec + W_in[:,x] + b - beta*a."""
    import torch
    ad = rho * ad_prev + (1.0 - rho) * h_prev            # a_t: non-fading slow trace of own activity (uses h_prev)
    rec = h_prev @ W_rec.t()                             # rec[b,j] = sum_i W_rec[j,i] h_prev[b,i]
    pre = rec + W_in[:, x_ids].t() + b - beta * ad       # -beta*a = activity-silent negative imprint
    act = torch.tanh(pre)
    h = (1.0 - alpha) * h_prev + alpha * act             # fast leaky-integrated state
    return h, ad, act


def alif_elig_update(eps_h, eps_a, act, h_prev, alpha, rho, beta, readonly):
    """FAITHFUL Bellec-2020 2-component ALIF eligibility (batched). eps_h,eps_a: (batch,n,n) with eps_h[b,j,i] ~
       d h_{b,j}/d W_rec[j,i] and eps_a[b,j,i] ~ d a_{b,j}/d W_rec[j,i]. eps_a is COUPLED into eps_h (the read-out observes
       BOTH compartments, so faithful credit sums the h-path and a-path sensitivities). Ported exactly from _train_alif:
         eps_a = rho*eps_a + (1-rho)*eps_h                         (d a_j/d w_ji, uses the OLD eps_h)
         eps_h = (1-alpha)*eps_h + psi*(h_prev - beta*eps_a)       (d h_j/d w_ji, COUPLED via the NEW eps_a)
       readonly (the adaptation is READ but NOT credited): eps_a untouched (stays 0), eps_h = the FAST-only recursion."""
    import torch
    n = eps_h.shape[1]
    psi = alpha * (1.0 - act * act)                      # (batch,n) leaky-tanh pseudo-derivative
    if readonly:
        eps_h = (1.0 - alpha) * eps_h + psi.unsqueeze(2) * h_prev.unsqueeze(1)          # FAST only (eps_a path dropped)
        return eps_h, eps_a
    rho_col = rho.view(1, n, 1)                          # index j (dim 1), broadcast over batch + i
    eps_a = rho_col * eps_a + (1.0 - rho_col) * eps_h    # d a_j/d w_ji  (uses the OLD eps_h)
    eps_h = (1.0 - alpha) * eps_h + psi.unsqueeze(2) * (h_prev.unsqueeze(1) - beta * eps_a)   # d h_j/d w_ji COUPLED
    return eps_h, eps_a


def build_init(V, n, seed, spectral, in_scale=1.0):
    """Deterministic shared initial parameters (numpy). Draw order mirrors the reservoir reference so runs are
       reproducible: W_rec (spectral-radius-scaled), W_in, b=0, W_out (small), B (fixed random feedback)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    sr = float(np.max(np.abs(np.linalg.eigvals(W))))     # circular-law radius ~ sqrt(n)
    W_rec = (spectral / sr) * W                           # set the spectral radius to `spectral`
    W_in = rng.standard_normal((n, V)) * (in_scale / np.sqrt(V))
    b = np.zeros(n)
    W_out = rng.standard_normal((V, n)) * 0.01
    Bfb = rng.standard_normal((n, V)) / np.sqrt(V)        # fixed random feedback (broadcast alignment; no weight transport)
    return {"W_rec": W_rec, "W_in": W_in, "b": b, "W_out": W_out, "Bfb": Bfb}


def build_alif_extra(V, n, seed, rho_win_lo, rho_win_hi):
    """ALIF-specific parameters, drawn from a SEPARATE / disjoint RNG so build_init (the shared reservoir init that the 5
       pre-registered arms depend on) is byte-UNCHANGED. The ALIF arms reuse build_init's W_rec/W_in/b (single-variable:
       same reservoir, the ONLY difference is adaptation-as-state + 2-component credit); these extras are the read-out over
       the [h;a] feature (2n) and the (2n x V) random feedback over BOTH compartments.
         rho_j = 1 - 1/window_j, window_j log-uniform over [rho_win_lo, rho_win_hi] tokens (heterogeneous adaptation
         time-constants -- the diverse-timescale forward hold that carries distal history)."""
    rng = np.random.default_rng(seed * 100003 + 7)        # disjoint stream from build_init(seed)
    win = np.exp(rng.uniform(np.log(rho_win_lo), np.log(rho_win_hi), size=n))
    rho = 1.0 - 1.0 / win                                 # e.g. [30,300]-token windows -> rho in ~[0.967, 0.997]
    W_out = rng.standard_normal((V, 2 * n)) * 0.01         # read-out over concat([h, a])
    Bfb = rng.standard_normal((2 * n, V)) / np.sqrt(V)     # fixed random feedback over [h; a] (no weight transport)
    return {"rho": rho, "W_out": W_out, "Bfb": Bfb}


# ======================================================================================================================
# Training arms
# ======================================================================================================================
def train_eprop(mode, init, tr_lanes, V, n, alpha, B, epochs, lr_out, lr_rec, wd_out, dev, a_slow=0.02):
    """Contiguous-stream e-prop for one reservoir arm. tr_lanes: (batch, lane) long tensor on `dev`. h + eligibility are
       carried across the 128-token block and RESET at each block boundary. `a_slow` = slow-eligibility leak, used ONLY by
       the `plastic_eprop_dualtc` arm (horizon ~1/a_slow tokens). Returns (W_rec, W_out) (both detached)."""
    import torch, torch.nn.functional as F
    W_rec = torch.tensor(init["W_rec"], dtype=torch.float32, device=dev)
    W_in = torch.tensor(init["W_in"], dtype=torch.float32, device=dev)      # FIXED input projection (reservoir setup)
    b = torch.tensor(init["b"], dtype=torch.float32, device=dev)
    W_out = torch.tensor(init["W_out"], dtype=torch.float32, device=dev)
    Bfb = torch.tensor(init["Bfb"], dtype=torch.float32, device=dev)
    batch, lane = tr_lanes.shape
    nb = (lane - 1) // B
    ar = torch.arange(batch, device=dev)
    plastic = (mode != "fixed_reservoir")
    dualtc = (mode == "plastic_eprop_dualtc")              # dual-timescale eligibility (slow credit horizon, no forward change)
    for ep in range(epochs):
        for j in range(nb):                                # consecutive contiguous blocks (deterministic order = fair)
            s = j * B
            x = tr_lanes[:, s:s + B]; y = tr_lanes[:, s + 1:s + B + 1]     # (batch,B)
            h = torch.zeros(batch, n, device=dev)
            e = torch.zeros(batch, n, n, device=dev)       # reset state + eligibility at the block boundary
            e_slow = torch.zeros(batch, n, n, device=dev) if dualtc else None   # DUAL-TIMESCALE: slow eligibility trace
            for p in range(B):
                h_prev = h
                h, act = rnn_forward_step(h_prev, x[:, p], W_rec, W_in, b, alpha)
                logits = h @ W_out.t()                     # (batch,V)
                probs = F.softmax(logits, dim=-1)
                delta = -probs
                delta[ar, y[:, p]] += 1.0                  # onehot(target) - softmax(logits): clean read-out error
                # read-out delta rule (identical across all reservoir arms):
                W_out = W_out + lr_out * ((delta.t() @ h) / batch - wd_out * W_out)
                if not plastic:                            # fixed_reservoir: W_rec frozen; no eligibility, no update
                    continue
                if dualtc:
                    # SLOW eligibility trace alongside the fast one -- extends the OWN-UNIT credit horizon WITHOUT any
                    # forward-dynamics change (contrast the ALIF -beta*a imprint which degraded the state). Same increment
                    # psi*h_prev, decay a_slow << alpha (horizon ~1/a_slow >> ~1/alpha); credit = e_fast + e_slow.
                    psi = alpha * (1.0 - act * act)
                    incr = psi.unsqueeze(2) * h_prev.unsqueeze(1)
                    e = (1.0 - alpha) * e + incr
                    e_slow = (1.0 - a_slow) * e_slow + incr
                else:
                    e = elig_update(e, act, h_prev, alpha)
                if mode == "zero_signal":
                    L = torch.zeros(batch, n, device=dev)  # L := 0 -> W_rec never moves (byte-identical to fixed)
                else:
                    L = delta @ Bfb.t()                    # (batch,n) broadcast random-feedback learning signal
                if mode == "shuffle_elig":
                    perm = torch.randperm(n * n, device=dev)       # break the credit-assignment structure (anti-cheat)
                    e_use = e.reshape(batch, n * n)[:, perm].reshape(batch, n, n)
                elif dualtc:
                    e_use = e + e_slow                     # dual-timescale credit (fast + slow horizon)
                else:
                    e_use = e
                dW = lr_rec * (L.unsqueeze(2) * e_use).mean(0)      # average the per-lane W_rec updates over the batch
                W_rec = W_rec + dW
        print(f"      [{mode}] epoch {ep + 1}/{epochs} last-block-CE "
              f"{F.cross_entropy(logits, y[:, p]).item():.3f}", flush=True)
    return W_rec.detach(), W_out.detach(), W_in.detach(), b.detach()


def train_eprop_alif(mode, init, alif_extra, tr_lanes, V, n, alpha, rho_np, beta, B, epochs, lr_out, lr_rec, wd_out, dev):
    """Contiguous-stream ALIF e-prop for one adaptation-as-state arm. Faithful batched port of `_train_alif`: the forward
       carries a per-unit adaptation a_t (subtracting a beta-scaled negative imprint from the pre-activation); the read-out
       reads concat([h, a]) (a carries distal history); W_rec is credited by the 2-component eligibility (eps_h COUPLED to
       eps_a) with BOTH read-out paths (L_h*eps_h + L_a*eps_a) -- the a-path is what extends the credit horizon. State +
       BOTH eligibilities are carried across the 128-token block and RESET at the block boundary (matching plain e-prop).
       Batch aggregation (per-lane mean) matches plain e-prop in this file so the ALIF arm is directly comparable.
         mode == 'plastic_eprop_alif'          -> both paths credited (the lever under test).
         mode == 'plastic_eprop_alif_readonly' -> a is READ ([h;a]) but the eps_a path is zeroed / credit is h-path only
                                                   (CONTROL: isolates 'adaptation carried to the read-out' [capacity] from
                                                   'adaptation extends the credit horizon' [the actual lever]).
       Returns (W_rec, W_out, W_in, b) detached. W_out is (V, 2n); B feedback is (2n, V)."""
    import torch, torch.nn.functional as F
    W_rec = torch.tensor(init["W_rec"], dtype=torch.float32, device=dev)
    W_in = torch.tensor(init["W_in"], dtype=torch.float32, device=dev)          # FIXED input projection (reservoir setup)
    b = torch.tensor(init["b"], dtype=torch.float32, device=dev)
    W_out = torch.tensor(alif_extra["W_out"], dtype=torch.float32, device=dev)   # (V, 2n) read-out over [h; a]
    Bfb = torch.tensor(alif_extra["Bfb"], dtype=torch.float32, device=dev)       # (2n, V) fixed random feedback over [h; a]
    rho = torch.tensor(rho_np, dtype=torch.float32, device=dev)                  # (n,) per-unit adaptation leak
    readonly = (mode == "plastic_eprop_alif_readonly")
    batch, lane = tr_lanes.shape
    nb = (lane - 1) // B
    ar = torch.arange(batch, device=dev)
    for ep in range(epochs):
        for jblk in range(nb):                             # consecutive contiguous blocks (deterministic order = fair)
            s = jblk * B
            x = tr_lanes[:, s:s + B]; y = tr_lanes[:, s + 1:s + B + 1]           # (batch,B)
            h = torch.zeros(batch, n, device=dev)
            ad = torch.zeros(batch, n, device=dev)         # reset state + adaptation + BOTH eligibilities at the boundary
            eps_h = torch.zeros(batch, n, n, device=dev)
            eps_a = torch.zeros(batch, n, n, device=dev)
            for p in range(B):
                h_prev = h
                h, ad, act = alif_forward_step(h_prev, ad, x[:, p], W_rec, W_in, b, alpha, rho, beta)
                feat = torch.cat([h, ad], dim=1)           # read-out feature = [h_t ; a_t]  (batch, 2n)
                logits = feat @ W_out.t()                  # (batch,V)
                probs = F.softmax(logits, dim=-1)
                delta = -probs
                delta[ar, y[:, p]] += 1.0                  # onehot(target) - softmax(logits): clean read-out error
                W_out = W_out + lr_out * ((delta.t() @ feat) / batch - wd_out * W_out)     # read-out delta rule over [h;a]
                eps_h, eps_a = alif_elig_update(eps_h, eps_a, act, h_prev, alpha, rho, beta, readonly)
                L = delta @ Bfb.t()                        # (batch,2n) broadcast random-feedback learning signal over [h;a]
                L_h = L[:, :n]; L_a = L[:, n:]
                if readonly:
                    dW = lr_rec * (L_h.unsqueeze(2) * eps_h).mean(0)                       # credit the h-path ONLY
                else:
                    dW = lr_rec * (L_h.unsqueeze(2) * eps_h + L_a.unsqueeze(2) * eps_a).mean(0)   # credit BOTH read-out paths
                W_rec = W_rec + dW
        print(f"      [{mode}] epoch {ep + 1}/{epochs} last-block-CE "
              f"{F.cross_entropy(logits, y[:, p]).item():.3f}", flush=True)
    return W_rec.detach(), W_out.detach(), W_in.detach(), b.detach()


def train_bptt(init, tr_lanes, V, n, alpha, B, bptt_steps, bptt_lr, dev, fix_win=False):
    """The matched-architecture FULL-BACKPROP ceiling: the SAME leaky-tanh cell as an autograd module, truncated to the
       128-block (h reset at each block, matching the e-prop reset). AdamW + grad-clip. Returns the final
       (W_rec, W_out, W_in, b) detached.
       fix_win=False -> trains ALL params (the pre-registered BPTT_same_net ceiling).
       fix_win=True  -> FREEZES W_in to the same random projection the e-prop arms use, so the ONLY difference from
                        plastic_eprop is the recurrent-credit RULE (BPTT full off-diagonal vs e-prop diagonal RTRL
                        truncation) on IDENTICAL input embeddings -- isolates the pure recurrent-credit fraction
                        (the R1 metric-confound fix: the all-params ceiling over-credits BPTT with embedding learning)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    W_rec = nn.Parameter(torch.tensor(init["W_rec"], dtype=torch.float32, device=dev))
    b = nn.Parameter(torch.tensor(init["b"], dtype=torch.float32, device=dev))
    W_out = nn.Parameter(torch.tensor(init["W_out"], dtype=torch.float32, device=dev))
    if fix_win:
        W_in = torch.tensor(init["W_in"], dtype=torch.float32, device=dev)   # FROZEN: same fixed random projection as e-prop
        params = [W_rec, b, W_out]
    else:
        W_in = nn.Parameter(torch.tensor(init["W_in"], dtype=torch.float32, device=dev))
        params = [W_rec, W_in, b, W_out]
    opt = torch.optim.AdamW(params, lr=bptt_lr)
    batch, lane = tr_lanes.shape
    nb = (lane - 1) // B
    step = 0
    while step < bptt_steps:
        for j in range(nb):
            if step >= bptt_steps:
                break
            s = j * B
            x = tr_lanes[:, s:s + B]; y = tr_lanes[:, s + 1:s + B + 1]
            h = torch.zeros(batch, n, device=dev)          # fresh state at the block boundary (truncated BPTT window)
            logits_all = []
            for p in range(B):
                h, _ = rnn_forward_step(h, x[:, p], W_rec, W_in, b, alpha)
                logits_all.append(h @ W_out.t())
            logits = torch.stack(logits_all, dim=1)        # (batch,B,V)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 1000 == 0 or step == bptt_steps:
                print(f"      [BPTT_same_net] step {step}/{bptt_steps} train-CE {loss.item():.3f}", flush=True)
    return W_rec.detach(), W_out.detach(), W_in.detach(), b.detach()


# ======================================================================================================================
# Eval: CE by within-block context depth (position), bucketed; margin vs the add-1 stream bigram.
# ======================================================================================================================
def eval_arm(W_rec, W_in, b, W_out, alpha, ev, B, P_bi, dev):
    """Held-out per-within-block-position CE for a trained (W_rec/W_in/b/W_out). State reset at each block start, so
       position p (0-indexed) = context depth p+1. Returns per-position (tce, bce, cnt) numpy arrays of length B."""
    import torch
    n = W_rec.shape[0]; V = W_out.shape[0]
    n_blocks = (len(ev) - 1) // B
    xs = np.stack([ev[i * B:(i + 1) * B] for i in range(n_blocks)])          # (n_blocks,B)
    ys = np.stack([ev[i * B + 1:(i + 1) * B + 1] for i in range(n_blocks)])  # (n_blocks,B) = next tokens
    tce = np.zeros(B); cnt = np.zeros(B)
    chunk = 128
    with torch.no_grad():
        for c0 in range(0, n_blocks, chunk):
            xb = torch.tensor(xs[c0:c0 + chunk], dtype=torch.long, device=dev)
            yb = torch.tensor(ys[c0:c0 + chunk], dtype=torch.long, device=dev)
            cb = xb.shape[0]
            h = torch.zeros(cb, n, device=dev)
            ar = torch.arange(cb, device=dev)
            for p in range(B):
                h, _ = rnn_forward_step(h, xb[:, p], W_rec, W_in, b, alpha)
                logp = torch.log_softmax(h @ W_out.t(), dim=-1)
                tce[p] += (-logp[ar, yb[:, p]]).sum().item()
                cnt[p] += cb
    # add-1 stream bigram: predict ev[s+p+1] from ev[s+p]
    bce = np.zeros(B)
    for p in range(B):
        bce[p] = -np.log(np.maximum(P_bi[xs[:, p], ys[:, p]], 1e-12)).sum()
    return tce, bce, cnt


def eval_arm_alif(W_rec, W_in, b, W_out, alpha, rho_np, beta, ev, B, P_bi, dev, shuffle_adapt=False):
    """Held-out per-within-block-position CE for a trained ALIF arm (W_rec/W_in/b + (V,2n) W_out over [h;a]). State (h,a)
       reset at each block start, so position p (0-indexed) = context depth p+1. Returns per-position (tce, bce, cnt).
       shuffle_adapt (ADAPTATION-SHUFFLE anti-cheat): permute a_t across neurons before the read-out at each token -> same
       extra read-out dims, WRONG content; if the deep gain is CONTENT (real, adaptation carries distal context) not
       CAPACITY (merely 2n read-out dims), it must collapse toward the no-adaptation (plain) arm."""
    import torch
    n = W_rec.shape[0]; V = W_out.shape[0]
    rho = torch.tensor(rho_np, dtype=torch.float32, device=dev)
    n_blocks = (len(ev) - 1) // B
    xs = np.stack([ev[i * B:(i + 1) * B] for i in range(n_blocks)])          # (n_blocks,B)
    ys = np.stack([ev[i * B + 1:(i + 1) * B + 1] for i in range(n_blocks)])  # (n_blocks,B) = next tokens
    tce = np.zeros(B); cnt = np.zeros(B)
    chunk = 128
    with torch.no_grad():
        for c0 in range(0, n_blocks, chunk):
            xb = torch.tensor(xs[c0:c0 + chunk], dtype=torch.long, device=dev)
            yb = torch.tensor(ys[c0:c0 + chunk], dtype=torch.long, device=dev)
            cb = xb.shape[0]
            h = torch.zeros(cb, n, device=dev)
            ad = torch.zeros(cb, n, device=dev)
            ar = torch.arange(cb, device=dev)
            for p in range(B):
                h, ad, _ = alif_forward_step(h, ad, xb[:, p], W_rec, W_in, b, alpha, rho, beta)
                ad_read = ad[:, torch.randperm(n, device=dev)] if shuffle_adapt else ad   # WRONG content, same dims
                feat = torch.cat([h, ad_read], dim=1)
                logp = torch.log_softmax(feat @ W_out.t(), dim=-1)
                tce[p] += (-logp[ar, yb[:, p]]).sum().item()
                cnt[p] += cb
    # add-1 stream bigram: predict ev[s+p+1] from ev[s+p] (identical baseline to eval_arm)
    bce = np.zeros(B)
    for p in range(B):
        bce[p] = -np.log(np.maximum(P_bi[xs[:, p], ys[:, p]], 1e-12)).sum()
    return tce, bce, cnt


def _pooled_margin(tce, bce, cnt, lo, hi):
    """Margin = bigram CE - model CE, pooled over within-block positions lo..hi (inclusive). +=model better."""
    m = slice(lo, hi + 1)
    c = cnt[m].sum()
    return float((bce[m].sum() - tce[m].sum()) / max(c, 1.0))


def summarize(tce, bce, cnt, B):
    """Per-bucket margins (BK) + pooled SHALLOW (pos 1-8) / DEEP (pos 17-127) margins + aggregate CE."""
    BK = [(1, 1), (2, 2), (3, 3), (4, 8), (9, 16), (17, B - 1)]
    by_bucket = {}
    for lo, hi in BK:
        hi = min(hi, B - 1)
        key = f"{lo}-{hi}" if lo != hi else f"{lo}"
        by_bucket[key] = round(_pooled_margin(tce, bce, cnt, lo, hi), 4)
    shallow = _pooled_margin(tce, bce, cnt, 1, 8)         # union of buckets 1-1..4-8
    deep = _pooled_margin(tce, bce, cnt, 17, B - 1)       # bucket 17-B
    agg_ce = float(tce[1:].sum() / max(cnt[1:].sum(), 1.0))
    return {"by_bucket": by_bucket, "shallow_margin": round(shallow, 4),
            "deep_margin": round(deep, 4), "aggregate_ce": round(agg_ce, 4)}


# ======================================================================================================================
# MANDATORY finite-difference grad-check (CPU). Extends the reservoir reference's grad_check_alif CHECK-A pattern to the
# contiguous leaky-tanh recursion: the ported eligibility e_{ji} must match a LOCAL finite difference of h_j w.r.t.
# W_rec[j,i] (cross-neuron recurrent drives held at the reference trajectory = the e-prop locality assumption).
# ======================================================================================================================
def _leaky_trace_np(Wr, W_in, b, alpha, ids, n):
    """Pure-numpy leaky-tanh forward accumulating the e-prop eligibility. Returns per-step h, recurrent-drive, eps."""
    h = np.zeros(n); e = np.zeros((n, n))
    hs = []; recs = []; eps = []
    for t in ids:
        h_prev = h
        rec = Wr @ h_prev
        pre = rec + W_in[:, t] + b
        act = np.tanh(pre)
        h = (1 - alpha) * h_prev + alpha * act
        psi = alpha * (1.0 - act * act)
        e = (1 - alpha) * e + psi[:, None] * h_prev[None, :]
        hs.append(h.copy()); recs.append(rec.copy()); eps.append(e.copy())
    return hs, recs, eps


def _alif_trace_np(Wr, W_in, b, alpha, rho, beta, ids, n):
    """Pure-numpy ALIF forward accumulating the FAITHFUL 2-component eligibility (ported from the reservoir reference's
       `_alif_trace`; the eps_a-coupled eps_h recursion is the authoritative reference the torch port must match). alpha is
       scalar (stream-runner convention), rho is per-unit. Returns per-step h, a, recurrent-drive, eps_h[t], eps_a[t]."""
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
        eps_a = rho[:, None] * eps_a + (1.0 - rho)[:, None] * eps_h                          # d a_j/d w_ji (uses OLD eps_h)
        eps_h = (1 - alpha) * eps_h + psi[:, None] * (h_prev[None, :] - beta * eps_a)         # d h_j/d w_ji COUPLED
        hs.append(h.copy()); ads.append(ad.copy()); recs.append(rec.copy()); eh.append(eps_h.copy()); ea.append(eps_a.copy())
    return {"hs": hs, "ads": ads, "recs": recs, "eps_h": eh, "eps_a": ea}


def _grad_check_alif(n=5, V=6, seq_len=8, seed=1):
    """CHECK ALIF (mandatory faithfulness of the ported 2-component ALIF eligibility; mirrors the reservoir reference's
       grad_check_alif CHECK-A + a torch==numpy bit-for-bit check on the exact training code path).
         (1) FD: eps_h[j,i], eps_a[j,i] vs a LOCAL finite-difference of h_j / a_j w.r.t. W_rec[j,i] (cross drives held at
             the reference trajectory = the e-prop locality assumption). Target ~1e-5 rel err.
         (2) torch==numpy (float64): the batched torch training-path eligibility (alif_forward_step + alif_elig_update,
             batch=1) must match the numpy eps_h/eps_a bit-for-bit on the tiny net.
       SHORT adaptation windows (3-30 tokens) so eps_a is genuinely active within an 8-token check (validates the a-path)."""
    import torch
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V, size=seq_len)
    alpha = 0.3; beta = 1.0
    init = build_init(V, n, seed + 1, spectral=1.1)
    Wr = init["W_rec"]; W_in = init["W_in"]; b = init["b"]
    win = np.exp(np.random.default_rng(seed + 1).uniform(np.log(3.0), np.log(30.0), size=n))   # short windows: eps_a active
    rho = 1.0 - 1.0 / win
    ref = _alif_trace_np(Wr, W_in, b, alpha, rho, beta, ids, n)
    hprev = [np.zeros(n)] + ref["hs"][:-1]                # h_prev at each step (zeros at t=0)

    def local_final(j, i, dw):                            # h_j / a_j at the final step, cross drives held at reference
        hj = 0.0; adj = 0.0
        for t in range(seq_len):
            hj_prev = hj
            adj = rho[j] * adj + (1.0 - rho[j]) * hj_prev
            rec_j = ref["recs"][t][j] + dw * hprev[t][i]  # perturb ONLY the w_ji direct term
            pre_j = rec_j + W_in[j, ids[t]] + b[j] - beta * adj
            hj = (1 - alpha) * hj_prev + alpha * np.tanh(pre_j)
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
    fd_pass = (maxrel_h < 1e-4) and (maxrel_a < 1e-4)

    # torch == numpy (float64, batch=1): the batched training-path eligibility must match the numpy port bit-for-bit.
    dev = "cpu"
    W_rec_t = torch.tensor(Wr, dtype=torch.float64, device=dev)
    W_in_t = torch.tensor(W_in, dtype=torch.float64, device=dev)
    b_t = torch.tensor(b, dtype=torch.float64, device=dev)
    rho_t = torch.tensor(rho, dtype=torch.float64, device=dev)
    h = torch.zeros(1, n, dtype=torch.float64, device=dev)
    ad = torch.zeros(1, n, dtype=torch.float64, device=dev)
    eps_h = torch.zeros(1, n, n, dtype=torch.float64, device=dev)
    eps_a = torch.zeros(1, n, n, dtype=torch.float64, device=dev)
    for t in ids:
        h_prev = h
        h, ad, act = alif_forward_step(h_prev, ad, torch.tensor([t], device=dev), W_rec_t, W_in_t, b_t, alpha, rho_t, beta)
        eps_h, eps_a = alif_elig_update(eps_h, eps_a, act, h_prev, alpha, rho_t, beta, readonly=False)
    tn_h = float(np.max(np.abs(eps_h[0].cpu().numpy() - ref["eps_h"][-1])))
    tn_a = float(np.max(np.abs(eps_a[0].cpu().numpy() - ref["eps_a"][-1])))
    tn_pass = (tn_h < 1e-8) and (tn_a < 1e-8)

    print(f"  CHECK ALIF (2-component eligibility vs LOCAL finite-diff of h_j/a_j w.r.t. W_rec[j,i]): "
          f"eps_h max_rel_err {maxrel_h:.2e} (max_abs {abs_h:.2e}) | eps_a max_rel_err {maxrel_a:.2e} "
          f"(max_abs {abs_a:.2e})  ->  {'PASS' if fd_pass else 'FAIL'}", flush=True)
    print(f"  CHECK ALIF-T (torch training-path eps_h/eps_a == numpy port, float64): "
          f"eps_h max_abs {tn_h:.2e} | eps_a max_abs {tn_a:.2e}  ->  {'PASS' if tn_pass else 'FAIL'}", flush=True)
    return fd_pass and tn_pass


def grad_check(n=5, V=6, seq_len=8, seed=1):
    """CHECK A (numpy, EXACT): eps_h[j,i] vs a LOCAL finite difference of h_j w.r.t. W_rec[j,i], perturbing ONLY the direct
       w_ji term (cross drives held at the reference trajectory). Target ~1e-5 rel err.
       CHECK T (torch == numpy): the torch training-path eligibility (run in float64, batch=1) must match the numpy
       eligibility on the tiny net (validates that the batched torch code path implements the same recursion).
       CHECK ALIF / CHECK ALIF-T: the same faithfulness pair for the ported 2-component ALIF eligibility (see _grad_check_alif)."""
    import torch
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, V, size=seq_len)
    alpha = 0.3
    init = build_init(V, n, seed + 1, spectral=1.1)
    Wr = init["W_rec"]; W_in = init["W_in"]; b = init["b"]
    hs, recs, eps = _leaky_trace_np(Wr, W_in, b, alpha, ids, n)
    hprev_list = [np.zeros(n)] + hs[:-1]                  # h_prev at each step (zeros at t=0)

    def local_final(j, i, dw):                            # h_j at the final step, cross drives held at reference
        hj = 0.0
        for t in range(seq_len):
            hj_prev = hj
            rec_j = recs[t][j] + dw * hprev_list[t][i]    # perturb ONLY the w_ji direct term
            pre_j = rec_j + W_in[j, ids[t]] + b[j]
            hj = (1 - alpha) * hj_prev + alpha * np.tanh(pre_j)
        return hj

    dh = 1e-6
    maxrel = 0.0; maxabs = 0.0
    for j in range(n):
        for i in range(n):
            fd = (local_final(j, i, dh) - local_final(j, i, -dh)) / (2 * dh)
            an = eps[-1][j, i]
            maxabs = max(maxabs, abs(fd - an))
            if abs(an) > 1e-7:
                maxrel = max(maxrel, abs(fd - an) / abs(an))
    checkA_pass = maxrel < 1e-4

    # CHECK T: torch training-path eligibility (float64, batch=1) vs numpy eligibility at the final step.
    dev = "cpu"
    W_rec_t = torch.tensor(Wr, dtype=torch.float64, device=dev)
    W_in_t = torch.tensor(W_in, dtype=torch.float64, device=dev)
    b_t = torch.tensor(b, dtype=torch.float64, device=dev)
    h = torch.zeros(1, n, dtype=torch.float64, device=dev)
    e = torch.zeros(1, n, n, dtype=torch.float64, device=dev)
    for t in ids:
        h_prev = h
        h, act = rnn_forward_step(h_prev, torch.tensor([t], device=dev), W_rec_t, W_in_t, b_t, alpha)
        e = elig_update(e, act, h_prev, alpha)
    e_torch = e[0].cpu().numpy()
    torch_np_maxabs = float(np.max(np.abs(e_torch - eps[-1])))
    checkT_pass = torch_np_maxabs < 1e-8

    print("=" * 100, flush=True)
    print(f"STREAM e-prop eligibility FAITHFULNESS CHECK  (n={n}, V={V}, seq_len={seq_len}, contiguous block)", flush=True)
    print(f"  CHECK A (numpy eps vs LOCAL finite-diff of h_j w.r.t. W_rec[j,i]): "
          f"max_rel_err {maxrel:.2e} (max_abs {maxabs:.2e})  ->  {'PASS' if checkA_pass else 'FAIL'}", flush=True)
    print(f"  CHECK T (torch training-path eps == numpy eps, float64 tiny net): "
          f"max_abs_diff {torch_np_maxabs:.2e}  ->  {'PASS' if checkT_pass else 'FAIL'}", flush=True)
    alif_pass = _grad_check_alif(n=n, V=V, seq_len=seq_len, seed=seed)      # the ADDED ALIF 2-component eligibility check
    all_pass = checkA_pass and checkT_pass and alif_pass
    print(f"  GRAD-CHECK (plain e-prop + ALIF): {'PASS' if all_pass else 'FAIL'}", flush=True)
    print("=" * 100, flush=True)
    return 0 if all_pass else 1


# ======================================================================================================================
# Gate
# ======================================================================================================================
def compute_gate(arms, byte_identical, eps=0.02, bptt_deep_floor=0.1):
    """Pre-registered gate. Hard anti-cheats FIRST (zero==fixed byte-identity; shuffle deep <= fixed deep + eps), then
       the frac_deep verdict. `arms` maps arm name -> summary dict."""
    need = ALL_ARMS
    if not all(a in arms for a in need):
        return {"verdict": f"INCOMPLETE (need all 5 arms; have {sorted(arms)})"}
    fixed = arms["fixed_reservoir"]; plastic = arms["plastic_eprop"]
    shuffle = arms["shuffle_elig"]; bptt = arms["BPTT_same_net"]
    shuffle_ok = shuffle["deep_margin"] <= fixed["deep_margin"] + eps
    # fractions vs the matched full-backprop ceiling (guard tiny/negative denominators)
    def frac(num, den):
        return round(num / den, 4) if den > 1e-6 else None
    frac_deep = frac(plastic["deep_margin"], bptt["deep_margin"])
    frac_shallow = frac(plastic["shallow_margin"], bptt["shallow_margin"])
    plastic_beats_fixed_deep = plastic["deep_margin"] > fixed["deep_margin"]
    monotone = plastic["deep_margin"] > plastic["shallow_margin"]     # margin grows with depth (long-range captured)
    bptt_deep_positive = bptt["deep_margin"] > bptt_deep_floor
    # verdict
    if not byte_identical:
        verdict = "INVALID (zero_signal W_rec != fixed_reservoir W_rec -- byte-identity anti-cheat FAILED)"
    elif not shuffle_ok:
        verdict = "INVALID (shuffle_elig deep margin exceeds the fixed floor -- credit-structure anti-cheat FAILED)"
    elif frac_deep is None:
        verdict = ("INCONCLUSIVE (BPTT_same_net deep margin ~0: the matched ceiling did not establish a positive "
                   "long-range target at this scale -- cannot form frac_deep)")
    elif frac_deep >= 0.50 and plastic_beats_fixed_deep and monotone:
        verdict = "GO (biological one-step-local credit captures >=50% of the full-backprop long-range margin)"
    elif frac_deep >= 0.25:
        verdict = ("PARTIAL (0.25 <= frac_deep < 0.50; named next lever = ALIF adaptation-as-state horizon extension, "
                   "Bellec-2020)")
    elif bptt_deep_positive:
        verdict = ("BOUNDARY (frac_deep < 0.25 while the BPTT ceiling deep margin is clearly positive -- honest: "
                   "forward-eligibility credit can't reach within-block deep dependencies)")
    else:
        verdict = ("INCONCLUSIVE (frac_deep < 0.25 but the BPTT ceiling deep margin is not clearly positive -- the "
                   "matched ceiling did not establish a long-range target)")
    return {"verdict": verdict, "byte_identical_zero_eq_fixed": bool(byte_identical),
            "shuffle_deep_le_fixed": bool(shuffle_ok), "frac_deep": frac_deep, "frac_shallow": frac_shallow,
            "plastic_beats_fixed_deep": bool(plastic_beats_fixed_deep), "monotone_deep_gt_shallow": bool(monotone),
            "bptt_deep_positive": bool(bptt_deep_positive)}


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories_train.txt")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--n-pool", type=int, default=300)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)                 # e-prop arms: passes over the contiguous blocks
    ap.add_argument("--batch", type=int, default=64)                 # contiguous lanes
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", type=str, nargs="+", default=ALL_ARMS)
    ap.add_argument("--alpha", type=float, default=0.3)              # leak
    ap.add_argument("--spectral", type=float, default=1.1)           # W_rec spectral radius (echo-state init)
    ap.add_argument("--lr-out", type=float, default=0.02)            # read-out delta-rule lr
    ap.add_argument("--lr-rec", type=float, default=0.001)           # e-prop W_rec lr
    ap.add_argument("--wd-out", type=float, default=1e-3)            # read-out weight decay (identical across e-prop arms)
    ap.add_argument("--a-slow", type=float, default=0.02)           # dual-timescale slow-eligibility leak (~1/0.02=50-token horizon)
    ap.add_argument("--bptt-steps", type=int, default=8000)          # BPTT ceiling: optimizer steps
    ap.add_argument("--bptt-lr", type=float, default=2e-3)           # BPTT AdamW lr (mirrors the ceiling runner)
    # ---- ALIF adaptation-as-state arms (additive; only used when a plastic_eprop_alif* arm is requested) ----
    ap.add_argument("--rho-win-lo", type=float, default=30.0)        # shortest adaptation window (tokens) for log-uniform rho
    ap.add_argument("--rho-win-hi", type=float, default=300.0)       # longest adaptation window (tokens); rho = 1 - 1/window
    ap.add_argument("--beta", type=float, default=1.0)              # adaptation->pre coupling (negative-imprint strength)
    ap.add_argument("--grad-check", action="store_true")            # run the mandatory finite-diff check + exit (CPU)
    ap.add_argument("--smoke", action="store_true")                # tiny end-to-end sanity (<2 min)
    ap.add_argument("--permute-stream", action="store_true")       # shuffle token order + refit bigram (deep must collapse)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    if args.grad_check:
        return grad_check()

    if args.smoke:
        args.n_pool = 32; args.max_tokens = 40_000; args.epochs = 1
        args.arms = ALL_ARMS
        args.bptt_steps = min(args.bptt_steps, 60)                  # smoke-only cap so BPTT stays fast
        if args.json == str(OUT):
            args.json = str(OUT).replace(".json", "_smoke.json")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    words = load_stream(args.corpus, args.max_tokens)
    cut = int(0.9 * len(words))
    stoi = build_vocab(words[:cut], args.vocab)

    def enc(ws):
        return np.array([stoi.get(w, 0) for w in ws], dtype=np.int64)
    tr = enc(words[:cut]); ev = enc(words[cut:])
    V = len(stoi); n = args.n_pool; B = args.block

    if args.permute_stream:                                          # ANTI-CHEAT MODE: destroy sequential structure
        rng = np.random.default_rng(args.seed)
        tr = tr[rng.permutation(len(tr))]
        ev = ev[rng.permutation(len(ev))]

    # add-1 stream bigram on train
    P_bi = np.ones((V, V))
    np.add.at(P_bi, (tr[:-1], tr[1:]), 1.0)
    P_bi /= P_bi.sum(1, keepdims=True)

    # contiguous lanes (batch parallel streams)
    lane = len(tr) // args.batch
    if lane < B + 1:
        raise SystemExit(f"train stream too short: lane={lane} < block+1={B + 1} (raise --max-tokens or lower --batch/--block)")
    tr_lanes_np = np.stack([tr[k * lane:(k + 1) * lane] for k in range(args.batch)])   # (batch, lane)
    tr_lanes = torch.tensor(tr_lanes_np, dtype=torch.long, device=dev)

    init = build_init(V, n, args.seed, args.spectral)
    # ALIF extras from a disjoint RNG -> build_init (and thus the 5 pre-registered arms) is byte-UNCHANGED whether or not
    # an ALIF arm is requested. The ALIF arms reuse init's W_rec/W_in/b (single-variable).
    alif_extra = build_alif_extra(V, n, args.seed, args.rho_win_lo, args.rho_win_hi)

    print(f"[stream-eprop] corpus={args.corpus} stream={len(words)} V={V} n_pool={n} block={B} batch={args.batch} "
          f"lane={lane} epochs={args.epochs} bptt_steps={args.bptt_steps} dev={dev} "
          f"permute_stream={args.permute_stream}", flush=True)
    print("[stream-eprop] PRE-REGISTERED EXPECTATION: e-prop is a diagonal (RTRL) truncation dropping the off-diagonal "
          "recurrent Jacobian; literature (Bellec-2020 needed DNI for long-range PTB; Murray-2019 RFLO matches BPTT only "
          "for short horizons) predicts plastic_eprop captures MOST of the SHALLOW margin and only a SMALL fraction of "
          "the DEEP margin. A shallow-only capture is an HONEST NEGATIVE on long-range -- a first-class deliverable.",
          flush=True)

    t0 = time.time()
    arms = {}; arm_wrec = {}; alif_artifacts = None
    for arm in args.arms:
        ta = time.time()
        is_alif = arm in ALIF_ARMS
        if arm == "BPTT_same_net":
            W_rec, W_out, W_in, b = train_bptt(init, tr_lanes, V, n, args.alpha, B, args.bptt_steps, args.bptt_lr, dev)
        elif arm == "BPTT_fixed_win":
            W_rec, W_out, W_in, b = train_bptt(init, tr_lanes, V, n, args.alpha, B, args.bptt_steps, args.bptt_lr, dev,
                                               fix_win=True)
        elif is_alif:
            W_rec, W_out, W_in, b = train_eprop_alif(arm, init, alif_extra, tr_lanes, V, n, args.alpha,
                                                     alif_extra["rho"], args.beta, B, args.epochs,
                                                     args.lr_out, args.lr_rec, args.wd_out, dev)
        elif arm in RESERVOIR_ARMS or arm == "plastic_eprop_dualtc":
            W_rec, W_out, W_in, b = train_eprop(arm, init, tr_lanes, V, n, args.alpha, B, args.epochs,
                                                args.lr_out, args.lr_rec, args.wd_out, dev, a_slow=args.a_slow)
        else:
            raise SystemExit(f"unknown arm '{arm}' (choose from {KNOWN_ARMS})")
        if is_alif:                                        # ALIF read-out is over [h;a] (2n) -> the ALIF-aware eval
            tce, bce, cnt = eval_arm_alif(W_rec, W_in, b, W_out, args.alpha, alif_extra["rho"], args.beta, ev, B, P_bi, dev)
            if arm == "plastic_eprop_alif":                # keep artifacts for the adaptation-shuffle control
                alif_artifacts = {"W_rec": W_rec, "W_in": W_in, "b": b, "W_out": W_out}
        else:
            tce, bce, cnt = eval_arm(W_rec, W_in, b, W_out, args.alpha, ev, B, P_bi, dev)
        arms[arm] = summarize(tce, bce, cnt, B)
        arm_wrec[arm] = W_rec.detach().cpu()
        print(f"    -> {arm}: shallow_margin {arms[arm]['shallow_margin']:+.4f}  deep_margin "
              f"{arms[arm]['deep_margin']:+.4f}  agg_ce {arms[arm]['aggregate_ce']:.4f}  ({time.time() - ta:.0f}s)",
              flush=True)

    # ALIF ADAPTATION-SHUFFLE control (content-not-capacity anti-cheat): re-eval the trained alif arm with a_t permuted
    # across neurons before the read-out. If the deep gain is real (adaptation CONTENT) it must collapse toward plain.
    alif_shuffle = None
    if alif_artifacts is not None:
        a = alif_artifacts
        tce_sh, bce_sh, cnt_sh = eval_arm_alif(a["W_rec"], a["W_in"], a["b"], a["W_out"], args.alpha,
                                               alif_extra["rho"], args.beta, ev, B, P_bi, dev, shuffle_adapt=True)
        alif_shuffle = summarize(tce_sh, bce_sh, cnt_sh, B)

    # zero_signal must be byte-identical to fixed_reservoir (hard anti-cheat)
    byte_identical = None
    if "zero_signal" in arm_wrec and "fixed_reservoir" in arm_wrec:
        byte_identical = bool(torch.equal(arm_wrec["zero_signal"], arm_wrec["fixed_reservoir"]))

    # ---- report ----
    BK_keys = [f"{lo}-{hi}" if lo != hi else f"{lo}"
               for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 8), (9, 16), (17, B - 1)]]
    print("\n[stream-eprop] MARGIN vs add-1 stream bigram, by within-block context depth (+ = arm better than bigram):",
          flush=True)
    header = "    " + f"{'arm':>16}" + "".join(f"{k:>10}" for k in BK_keys) + f"{'SHALLOW':>10}{'DEEP':>10}"
    print(header, flush=True)
    for arm in args.arms:
        s = arms[arm]
        row = "    " + f"{arm:>16}" + "".join(f"{s['by_bucket'].get(k, float('nan')):>+10.4f}" for k in BK_keys)
        row += f"{s['shallow_margin']:>+10.4f}{s['deep_margin']:>+10.4f}"
        print(row, flush=True)

    gate = compute_gate(arms, byte_identical if byte_identical is not None else False)
    print("\n[stream-eprop] ANTI-CHEATS:", flush=True)
    print(f"    zero_signal W_rec == fixed_reservoir W_rec (byte-identical): {byte_identical}", flush=True)
    if "shuffle_elig" in arms and "fixed_reservoir" in arms:
        print(f"    shuffle_elig deep {arms['shuffle_elig']['deep_margin']:+.4f} <= "
              f"fixed deep {arms['fixed_reservoir']['deep_margin']:+.4f} + eps: "
              f"{arms['shuffle_elig']['deep_margin'] <= arms['fixed_reservoir']['deep_margin'] + 0.02}", flush=True)
    if "frac_deep" in gate:
        print(f"\n[stream-eprop] FRACTIONS (plastic_eprop / BPTT_same_net):  "
              f"frac_shallow {gate['frac_shallow']}   frac_deep {gate['frac_deep']}  "
              f"(DEEP is the entire test; a strong shallow must NOT mask a weak deep)", flush=True)
    print(f"\n[stream-eprop] GATE: {gate['verdict']}", flush=True)

    # ---- SECONDARY: fixed-floor-relative recurrent-credit fraction (isolates the RULE from embedding learning) ----
    # frac_clean_deep = (plastic_deep - fixed_deep) / (BPTT_arm_deep - fixed_deep): how much of the recurrent-credit-
    # achievable deep gain OVER THE SHARED FIXED-EMBEDDING FLOOR e-prop captures. For BPTT_fixed_win (same frozen W_in as
    # e-prop) this is the CLEANEST isolation of the credit RULE (e-prop diagonal vs BPTT full off-diagonal); for
    # BPTT_same_net it still credits BPTT with learned embeddings (the R1 confound).
    clean = {}
    if "fixed_reservoir" in arms and "plastic_eprop" in arms:
        f_deep = arms["fixed_reservoir"]["deep_margin"]; f_shal = arms["fixed_reservoir"]["shallow_margin"]
        p_gain_deep = arms["plastic_eprop"]["deep_margin"] - f_deep
        p_gain_shal = arms["plastic_eprop"]["shallow_margin"] - f_shal
        print("\n[stream-eprop] FIXED-FLOOR-RELATIVE recurrent-credit fraction "
              "(plastic gain over fixed / BPTT gain over fixed; isolates the credit RULE):", flush=True)
        for bp in ("BPTT_fixed_win", "BPTT_same_net"):
            if bp in arms:
                b_gain_deep = arms[bp]["deep_margin"] - f_deep
                b_gain_shal = arms[bp]["shallow_margin"] - f_shal
                fd = round(p_gain_deep / b_gain_deep, 4) if abs(b_gain_deep) > 1e-6 else None
                fs = round(p_gain_shal / b_gain_shal, 4) if abs(b_gain_shal) > 1e-6 else None
                clean[bp] = {"frac_clean_deep": fd, "frac_clean_shallow": fs,
                             "plastic_deep_gain": round(p_gain_deep, 4), "bptt_deep_gain": round(b_gain_deep, 4)}
                tag = " <- CLEANEST (identical frozen W_in)" if bp == "BPTT_fixed_win" else " (BPTT also learns W_in)"
                print(f"    vs {bp:>15}: frac_clean_deep {fd}  frac_clean_shallow {fs}  "
                      f"(plastic +{p_gain_deep:.3f} / bptt +{b_gain_deep:.3f} deep){tag}", flush=True)

    # ---- ALIF adaptation-as-state report (ADDITIVE; the pre-registered 5-arm gate above is byte-unchanged). The single
    # highest-leverage biological lever: does the 2-component ALIF eligibility (Bellec-2020 "highways into the future")
    # lift the DEEP capture over plain e-prop, and is that lift real CONTENT (adaptation-shuffle collapses) + specifically
    # from CREDITING adaptation (alif > alif_readonly)? Reported directly comparable to plain e-prop's clean fraction. ----
    alif_clean = {}
    if "plastic_eprop_alif" in arms:
        pa = arms["plastic_eprop_alif"]
        print("\n[stream-eprop] ALIF ADAPTATION-AS-STATE (Bellec-2020 horizon lever) -- ADDITIVE arms (not in the 5-arm gate):",
              flush=True)
        if "plastic_eprop" in arms:
            pe = arms["plastic_eprop"]
            print(f"    ALIF deep lift over plain e-prop = {pa['deep_margin']:+.4f} - {pe['deep_margin']:+.4f} = "
                  f"{pa['deep_margin'] - pe['deep_margin']:+.4f}   (shallow: {pa['shallow_margin']:+.4f} vs "
                  f"{pe['shallow_margin']:+.4f})", flush=True)
        if "fixed_reservoir" in arms:
            fa_deep = arms["fixed_reservoir"]["deep_margin"]; fa_shal = arms["fixed_reservoir"]["shallow_margin"]
            a_gain_deep = pa["deep_margin"] - fa_deep; a_gain_shal = pa["shallow_margin"] - fa_shal
            print("    fixed-floor-relative clean fraction for plastic_eprop_alif "
                  "(alif gain over fixed / BPTT gain over fixed -- directly comparable to plain e-prop's above):",
                  flush=True)
            for bp in ("BPTT_fixed_win", "BPTT_same_net"):
                if bp in arms:
                    bgd = arms[bp]["deep_margin"] - fa_deep; bgs = arms[bp]["shallow_margin"] - fa_shal
                    fd = round(a_gain_deep / bgd, 4) if abs(bgd) > 1e-6 else None
                    fs = round(a_gain_shal / bgs, 4) if abs(bgs) > 1e-6 else None
                    alif_clean[bp] = {"frac_clean_deep": fd, "frac_clean_shallow": fs,
                                      "alif_deep_gain": round(a_gain_deep, 4), "bptt_deep_gain": round(bgd, 4)}
                    tag = " <- CLEANEST (identical frozen W_in)" if bp == "BPTT_fixed_win" else " (BPTT also learns W_in)"
                    print(f"        vs {bp:>15}: frac_clean_deep {fd}  frac_clean_shallow {fs}  "
                          f"(alif +{a_gain_deep:.3f} / bptt +{bgd:.3f} deep){tag}", flush=True)
        if "plastic_eprop_alif_readonly" in arms:
            ro = arms["plastic_eprop_alif_readonly"]
            print(f"    CREDIT-vs-CAPACITY: alif deep {pa['deep_margin']:+.4f} vs alif_readonly deep "
                  f"{ro['deep_margin']:+.4f} (adaptation READ but eps_a NOT credited); lift from CREDITING adaptation = "
                  f"{pa['deep_margin'] - ro['deep_margin']:+.4f}", flush=True)
        if alif_shuffle is not None:
            collapses = alif_shuffle["deep_margin"] < pa["deep_margin"] - 1e-6
            print(f"    ADAPTATION-SHUFFLE control: alif deep {pa['deep_margin']:+.4f} -> a_t-shuffled deep "
                  f"{alif_shuffle['deep_margin']:+.4f}  ({'COLLAPSES (content, real)' if collapses else 'does NOT collapse (capacity confound?)'})",
                  flush=True)

    out = {"runner": "_emerge_stream_eprop_lm_derisk", "corpus": args.corpus, "seed": args.seed, "V": V,
           "n_pool": n, "block": B, "batch": args.batch, "lane": lane, "epochs": args.epochs,
           "bptt_steps": args.bptt_steps, "dev": dev, "permute_stream": bool(args.permute_stream),
           "args": vars(args), "arms": arms, "byte_identical_zero_eq_fixed": byte_identical, "gate": gate,
           "clean_recurrent_credit_fraction": clean, "alif_clean_fraction": alif_clean,
           "alif_adapt_shuffle": alif_shuffle, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)\nSTREAM_EPROP_DONE", flush=True)


if __name__ == "__main__":
    main()
