"""Cheap-first probe: does a biological LOCAL learning rule GENERALIZE where global backprop OVERFITS?

Owner redirect (2026-06-02): stay biology-faithful; today's generative negative used BPTT (non-biological per
biology.md "brains learn from local rules"). The missing structure = apical-basal dendritic / predictive-coding
LOCAL learning. Load-bearing hypothesis: a biological local rule closes the train-vs-held-out generalization
gap that backprop shows (today's exact overfit failure mode).

This probe isolates the LEARNING RULE on an identical task/architecture/capacity:
  - global BACKPROP  (the non-biological baseline; uses W^T weight transport -> biologically implausible)
  - FEEDBACK ALIGNMENT (Lillicrap et al. 2016, Nat Commun): identical forward net, but the hidden error is
    propagated through FIXED RANDOM feedback weights B instead of W^T. No weight transport -> the local,
    biologically-plausible credit-assignment rule. (The apical compartment = the top-down error signal carried
    by B; basal plasticity is gated by it.)
  - PREDICTIVE CODING (Whittington-Bogacz 2017): added as the design-faithful rule if FA shows signal.

Task = a structured next-token task from a FIXED random higher-order rule, in the OVERFIT regime (high
capacity, limited train data, held-out from the SAME rule). A model that memorizes train contexts wins on
train but fails held-out unless it learned the rule. Metric: train vs held-out cross-entropy + the gap.
Multi-seed. Controls: untrained (chance) floor; shuffled-target memorization control (confirms the task has
real generalizable structure -> guards against a vacuous gap).

Pre-registered decisive read: a local rule RESOLVES the hypothesis iff its held-out loss < backprop's AND its
gap (held-out - train) < backprop's, multi-seed. Stdlib + numpy only. No protected-module import.

  python -m research.findings.raw._pc_vs_bptt_probe
"""
from __future__ import annotations
import numpy as np


# ----- task: COMPOSITIONAL generalization. next = (f[a] + g[b]) mod V over a 2-token context (a,b). -----
# f, g are fixed random per-token maps (the COMPOSITIONAL structure). A model must learn f and g SEPARATELY
# to predict an unseen (a,b) pair; pure pair-memorization fails held-out. This is the generalization that
# matters for language (systematic composition), and it creates a real train-vs-held-out gap in the
# high-capacity / limited-data (overfit) regime.
def make_rule(vocab):
    rng = np.random.default_rng(20260602)            # FIXED structure across seeds
    f = rng.integers(0, vocab, size=vocab)
    g = rng.integers(0, vocab, size=vocab)
    table = np.array([[(int(f[a]) + int(g[b])) % vocab for b in range(vocab)] for a in range(vocab)])
    return table


def make_dataset(vocab, train_frac, seed):
    """All V*V (a,b) pairs, split into DISJOINT train / held-out by pair. Input = one-hot(a)++one-hot(b).
    Held-out pairs are UNSEEN in train -> only compositional structure generalizes."""
    table = make_rule(vocab)
    pairs = [(a, b) for a in range(vocab) for b in range(vocab)]
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_tr = int(len(pairs) * train_frac)
    tr_pairs, ho_pairs = pairs[:n_tr], pairs[n_tr:]

    def build(ps):
        X = np.zeros((len(ps), 2 * vocab), dtype=np.float64)
        Y = np.zeros(len(ps), dtype=np.int64)
        for i, (a, b) in enumerate(ps):
            X[i, a] = 1.0
            X[i, vocab + b] = 1.0
            Y[i] = table[a, b]
        return X, Y

    return build(tr_pairs), build(ho_pairs)


# ----- model: 2-layer MLP (in -> H -> H -> V), trained by 3 learning rules -----
def init_net(n_in, H, V, seed):
    rng = np.random.default_rng(seed)
    def w(a, b):
        return rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
    W1, W2, W3 = w(n_in, H), w(H, H), w(H, V)
    B2 = rng.standard_normal((H, H)) * np.sqrt(2.0 / H)   # fixed random feedback (FA)
    B3 = rng.standard_normal((V, H)) * np.sqrt(2.0 / V)   # V -> H feedback (note shape for B3 @ e)
    return dict(W1=W1, W2=W2, W3=W3, B2=B2, B3=B3)


def relu(x):
    return np.maximum(x, 0.0)


def drelu(x):
    return (x > 0.0).astype(np.float64)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(net, X):
    a1 = X @ net["W1"]; h1 = relu(a1)
    a2 = h1 @ net["W2"]; h2 = relu(a2)
    logit = h2 @ net["W3"]
    return a1, h1, a2, h2, logit


def ce_loss(logit, Y):
    p = softmax(logit)
    return float(-np.log(p[np.arange(len(Y)), Y] + 1e-12).mean())


def train(net, X, Y, V, rule_name, epochs, lr, batch, seed):
    rng = np.random.default_rng(seed * 7 + 1)
    n = len(X)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for bi in range(0, n, batch):
            idx = perm[bi:bi + batch]
            xb, yb = X[idx], Y[idx]
            a1, h1, a2, h2, logit = forward(net, xb)
            p = softmax(logit)
            dlogit = p.copy(); dlogit[np.arange(len(yb)), yb] -= 1.0; dlogit /= len(yb)
            # output-layer grad is identical (local: error x pre-activity)
            gW3 = h2.T @ dlogit
            if rule_name == "backprop":
                dh2 = dlogit @ net["W3"].T * drelu(a2)
                dh1 = dh2 @ net["W2"].T * drelu(a1)
            elif rule_name == "feedback_alignment":
                # hidden errors via FIXED RANDOM feedback (no weight transport) -> local rule
                dh2 = (dlogit @ net["B3"]) * drelu(a2)
                dh1 = (dh2 @ net["B2"]) * drelu(a1)
            else:
                raise ValueError(rule_name)
            gW2 = h1.T @ dh2
            gW1 = xb.T @ dh1
            net["W3"] -= lr * gW3
            net["W2"] -= lr * gW2
            net["W1"] -= lr * gW1
    return net


def run_seed(seed, vocab=12, H=256, train_frac=0.6, epochs=300, lr=0.05, batch=32, shuffled_control=False):
    (Xtr, Ytr), (Xho, Yho) = make_dataset(vocab, train_frac, seed)
    if shuffled_control:                               # destroy structure -> nothing generalizable
        Ytr = np.random.default_rng(seed).integers(0, vocab, size=len(Ytr))
    n_in = 2 * vocab
    out = {}
    for rule_name in ("backprop", "feedback_alignment"):
        net = init_net(n_in, H, vocab, seed=seed)
        net = train(net, Xtr, Ytr, vocab, rule_name, epochs, lr, batch, seed)
        tr_l = ce_loss(forward(net, Xtr)[-1], Ytr)
        ho_l = ce_loss(forward(net, Xho)[-1], Yho)
        out[rule_name] = (tr_l, ho_l, ho_l - tr_l)
    # untrained chance floor
    net0 = init_net(n_in, H, vocab, seed=seed)
    out["untrained_holdout_loss"] = ce_loss(forward(net0, Xho)[-1], Yho)
    out["uniform_loss"] = float(np.log(vocab))
    return out


def main():
    seeds = [42, 43, 44]
    print("=== PC/local-rule vs backprop: generalization probe (structured next-token, overfit regime) ===",
          flush=True)
    agg = {"backprop": [], "feedback_alignment": []}
    for s in seeds:
        r = run_seed(s)
        for k in agg:
            tr_l, ho_l, gap = r[k]
            agg[k].append((tr_l, ho_l, gap))
            print(f"  seed {s:>3} {k:>18}: train {tr_l:.3f}  held-out {ho_l:.3f}  gap {gap:+.3f}", flush=True)
        print(f"           (uniform-random floor {r['uniform_loss']:.3f}, untrained held-out "
              f"{r['untrained_holdout_loss']:.3f})", flush=True)
    print("\n  shuffled-target control (structure destroyed -> held-out should NOT beat uniform):", flush=True)
    cs = run_seed(42, shuffled_control=True)
    for k in ("backprop", "feedback_alignment"):
        print(f"    {k:>18}: train {cs[k][0]:.3f}  held-out {cs[k][1]:.3f} (uniform {cs['uniform_loss']:.3f})",
              flush=True)

    def mean(k, i):
        return float(np.mean([v[i] for v in agg[k]]))
    bp_ho, fa_ho = mean("backprop", 1), mean("feedback_alignment", 1)
    bp_gap, fa_gap = mean("backprop", 2), mean("feedback_alignment", 2)
    print(f"\nRESULT (mean over {len(seeds)} seeds): backprop held-out {bp_ho:.3f} gap {bp_gap:+.3f} | "
          f"feedback_alignment held-out {fa_ho:.3f} gap {fa_gap:+.3f}", flush=True)
    if fa_ho < bp_ho and fa_gap < bp_gap:
        print("VERDICT: RESOLVES (this slice) -- the biological LOCAL rule generalizes BETTER than backprop "
              "(lower held-out AND smaller gap). Supports the missing-mechanism hypothesis -> proceed to the "
              "design-faithful predictive-coding rule + the spiking build.", flush=True)
    elif abs(fa_ho - bp_ho) < 0.02:
        print("VERDICT: TIE -- local rule ~ backprop here (neither clearly overfits at this scale). Need the "
              "overfit regime sharpened or the predictive-coding rule to distinguish; inconclusive on the "
              "generalization claim.", flush=True)
    else:
        print("VERDICT: backprop generalizes >= the local rule here -- this local rule does NOT close the gap "
              "at this slice. Honest signal; try the predictive-coding rule / sharpen the regime before any "
              "spiking build.", flush=True)


if __name__ == "__main__":
    main()
