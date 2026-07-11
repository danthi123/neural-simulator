"""Cheapest-decisive probe (a0): can a TRAINED read-out extract the AGENT ROLE (first-noun's category) from the RECURRENT
reservoir's OWN state trajectory -- NO host latch, NO host lookup of prefix[0] -- well enough to do the order-decisive ROLE
REVERSAL? The Rung-4 host-latch version was a shortcut (an adversarial skeptic correctly flagged it: a host-Python read of
the first token did 100% of the work). The honest question: does the spiking reservoir itself expose which noun is the
agent? Key insight: the running-CUMULATIVE feature is a MEAN -> order-DESTROYING; the recurrent reservoir's per-position
state TRAJECTORY carries order. We test several order-preserving reservoir reads, all reservoir-only (no latch):
  cum   -- running_cumulative at pos 2 (order-washed mean; expect FAIL, the known 0.0)
  win2  -- per_window at pos 2 on the RECURRENT reservoir (carries N1 via recurrence; streaming-faithful)
  traj  -- concat of per_window [s0,s1,s2] (reads the agent's OWN state s0 = the reservoir's response to N1)
  final -- the running_cumulative at the LAST position (whole-prefix summary)
For each: train a one-step-delta read-out feature->action on TRAIN_SENTS, score reversal on the held-out TWINS. Plus a
permuted (word-shuffled training) control on the best feature. Reuse r3's grammar/reservoir. CPU numpy.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, numpy as np
import research.runners._emerge_reservoir_lm_rung3_systematic_generation_derisk as r3
import research.runners._emerge_reservoir_lm_rung4_order_decisive_recombination_derisk as r4
from research.runners._emerge_reservoir_lm_derisk import train_readout, _standardize_fit


def feat_of(res, prefix, code_type, seed, mode):
    U = r3.encode(prefix, code_type, seed)
    cum = res.per_token_states(U, feature="running_cumulative")
    win = res.per_token_states(U, feature="per_window")
    if mode == "cum":
        return cum[r3.ACTION_POS - 1]
    if mode == "win2":
        return win[r3.ACTION_POS - 1]
    if mode == "traj":
        return np.concatenate([win[t] for t in range(len(prefix))])   # all taps (order-preserving, incl. the agent's own state)
    if mode == "final":
        return cum[-1]
    raise ValueError(mode)


def train_and_score(seed, mode, code_type="class", permute=False, n_pool=300, epochs=200, lr=0.05):
    res = r3.ReservoirStates(r3.D_CODE, seed=seed, n=n_pool)      # RECURRENT reservoir (the real substrate)
    train = list(r3.TRAIN_SENTS)
    if permute:
        rng = np.random.default_rng(seed * 7 + 3)
        train = [list(rng.permutation(s)) for s in train]
    # sentence-level read-out: feature over the prefix "N1 meets N2" -> the ACTION token.
    feats, tgts = [], []
    for s in train:
        prefix = s[:3]
        feats.append(feat_of(res, prefix, code_type, seed, mode))
        tgts.append(r3.WORD_IDX[s[3]])
    X = np.array(feats); mean = X.mean(0); std = X.std(0) + 1e-6
    Xn = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    W = np.zeros((r3.V, Xn.shape[1]))
    rng = np.random.default_rng(seed * 13 + 1)
    idx = list(range(len(Xn)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            p = r3._softmax(W @ Xn[i]) if hasattr(r3, "_softmax") else _sm(W @ Xn[i])
            t = np.zeros(r3.V); t[tgts[i]] = 1.0
            W += lr * np.outer(t - p, Xn[i])

    def predict(prefix):
        f = feat_of(res, prefix, code_type, seed, mode)
        x = np.concatenate([(f - mean) / std, [1.0]])
        return r3.WORDS[int(np.argmax(W @ x))]
    both = po = potot = tot = 0
    for a, b in r4.TWINS:
        oks = []
        for (n1, n2) in [(a, b), (b, a)]:
            pred = predict([n1, r3.MEETS, n2])
            c = (pred in r3.ACTION_CAT and r3.ACTION_CAT[pred] == r3.ANIMAL_CAT[n1])
            oks.append(c); po += int(c); potot += 1
        both += int(oks[0] and oks[1]); tot += 1
    return both / tot, po / potot


def _sm(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--n-pool", type=int, default=300)
    args = ap.parse_args()
    for seed in args.seeds:
        print(f"=== seed {seed} (recurrent reservoir, NO host latch) ===")
        for mode in ["cum", "win2", "traj", "final"]:
            rev, po = train_and_score(seed, mode, n_pool=args.n_pool)
            print(f"  {mode:6s}  reversal={rev:.3f}  per_order={po:.3f}")
        rev_p, po_p = train_and_score(seed, "traj", permute=True, n_pool=args.n_pool)
        rev_o, po_o = train_and_score(seed, "traj", code_type="onehot", n_pool=args.n_pool)
        print(f"  traj+permuted  reversal={rev_p:.3f} per_order={po_p:.3f}   (order control)")
        print(f"  traj+onehot    reversal={rev_o:.3f} per_order={po_o:.3f}   (shared-code control)")


if __name__ == "__main__":
    main()
