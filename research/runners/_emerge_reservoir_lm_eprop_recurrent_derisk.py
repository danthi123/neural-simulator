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
                 leak_spectrum="homogeneous", alpha_lo=0.03, alpha_hi=0.6):
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

    def forward_states(self, ids):
        """Run the (possibly-trained) reservoir over a sequence; return states[t] = h_t (the recurrent state at token t)."""
        a = self.alpha; h = np.zeros(self.n); out = []
        for t in ids:
            x = self.W_in[:, t]                                   # onehot(t) @ W_in^T = column t
            h = (1 - a) * h + a * np.tanh(self.W_rec @ h + x + self.b)
            out.append(h.copy())
        return out


def train(res, tr_ids, V, epochs, lr_out, lr_rec, seed, mode="plastic", wd=1e-3, wd_rec=0.0, a_slow_leak=0.05):
    """Online co-training: W_out by the one-step delta rule; W_rec by e-prop, NO BPTT. mode in
       {fixed, plastic, adaptive, symmetric, shuffle_elig, zero_signal}. `adaptive` adds a DUAL-TIMESCALE eligibility (a
       slow trace with leak `a_slow_leak` alongside the fast one) so credit reaches the DEEP context the fast ~1/alpha
       eligibility cannot -- WITHOUT changing the forward dynamics (the reservoir state stays fast; the read-out is
       unaffected; only the CREDIT horizon lengthens -- the faithful ALIF idea, vs naive slow-leak which degrades the
       state). `symmetric` uses W_out^T feedback (weight-transport ceiling on the random-feedback cost); `wd_rec` =
       weight decay on W_rec (guards the capacity/overfitting confound). Returns W_out (W_rec updated in place on res)."""
    rng = np.random.default_rng(seed * 13 + 7)
    n = res.n; a = res.alpha                                      # per-unit leak vector
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


def per_depth_ce(res, W_out, ev_ids, P_bi, lesion_slow=False):
    """Per-context-depth held-out CE for the (trained) reservoir+read-out and the bigram baseline.
       lesion_slow: zero the read-out weights on the SLOW units -> if the deep gain collapses, the slow units' STATE
       was carrying the distal context the read-out uses (the state-memory lesion control)."""
    if lesion_slow and res.slow_idx.size:
        W_out = W_out.copy(); W_out[:, res.slow_idx] = 0.0
    rce = defaultdict(float); bce = defaultdict(float); cnt = defaultdict(int)
    for ids in ev_ids:
        states = res.forward_states(ids)
        for t in range(len(ids) - 1):
            b = _bucket(t + 1)
            p = _softmax(W_out @ states[t]); tgt = ids[t + 1]
            rce[b] += -math.log(max(p[tgt], 1e-12))
            bce[b] += -math.log(max(P_bi[ids[t], tgt], 1e-12)); cnt[b] += 1
    tot_r = sum(rce.values()); tot_b = sum(bce.values()); n = sum(cnt.values())
    depth = {b: {"n": cnt[b], "ce": round(rce[b] / cnt[b], 3), "bigram_ce": round(bce[b] / cnt[b], 3),
                 "margin_vs_bigram": round((bce[b] - rce[b]) / cnt[b], 3)} for b in cnt}
    return depth, round(tot_r / n, 3), round(tot_b / n, 3)


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
    ap.add_argument("--modes", type=str, nargs="+", default=["fixed", "plastic", "shuffle_elig", "zero_signal"])
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

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
        rec = {"V": V, "n_train": len(tr), "by_mode": {}}
        for mode in args.modes:
            res = RateReservoir(V, args.n_pool, seed, alpha=args.alpha, spectral=args.spectral,
                                leak_spectrum=args.leak_spectrum, alpha_lo=args.alpha_lo, alpha_hi=args.alpha_hi)
            W_out = train(res, tr_ids, V, args.epochs, args.lr_out, args.lr_rec, seed, mode=mode,
                          wd_rec=args.wd_rec, a_slow_leak=args.alpha_slow)
            depth, agg, aggb = per_depth_ce(res, W_out, ev_ids, P_bi, lesion_slow=args.lesion_slow)
            rec["by_mode"][mode] = {"aggregate_ce": agg, "bigram_ce": aggb, "by_depth": depth}
        per_seed[str(seed)] = rec
        # print: plastic-minus-fixed CE per depth (negative = plastic BETTER); the load-bearing deep(6+) delta
        fx = rec["by_mode"].get("fixed"); pl = rec["by_mode"].get("plastic")
        if fx and pl:
            dd = {b: round(pl["by_depth"][b]["ce"] - fx["by_depth"][b]["ce"], 3)
                  for b in pl["by_depth"] if b in fx["by_depth"]}
            deep = [b for b in ("6-9", "10-99") if b in dd]
            deepd = float(np.mean([dd[b] for b in deep])) if deep else float("nan")
            print(f"[seed {seed}] V={V} agg: fixed {fx['aggregate_ce']} plastic {pl['aggregate_ce']} "
                  f"(delta {pl['aggregate_ce']-fx['aggregate_ce']:+.3f}) | plastic-MINUS-fixed CE by depth "
                  + " ".join(f"d{b}:{dd[b]:+.2f}" for lo,hi in BUCKETS for b in [f'{lo}-{hi}' if lo!=hi else f'{lo}'] if b in dd)
                  + f" | DEEP(6+) {deepd:+.3f} (neg=plastic better)", flush=True)
            for anti in ("shuffle_elig", "zero_signal"):
                aa = rec["by_mode"].get(anti)
                if aa:
                    print(f"           anti[{anti}] agg {aa['aggregate_ce']} (vs fixed {fx['aggregate_ce']:+.3f}={aa['aggregate_ce']-fx['aggregate_ce']:+.3f})", flush=True)

    out = {"runner": "_emerge_reservoir_lm_eprop_recurrent_derisk", "corpus": args.corpus, "seeds": args.seeds,
           "n_pool": args.n_pool, "args": vars(args), "per_seed": per_seed, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json} ({out['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
