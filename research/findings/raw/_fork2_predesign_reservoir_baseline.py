"""Scratch control (mechanism-design-independent) for the Fork-2 rung-3 next-symbol de-risk.

QUESTION: on the two-overlapping-sequences next-symbol task (input-driven, teacher-forced), does a FIXED random
leaky reservoir + a locally-trained readout ALREADY solve the branch (carry the cue through the shared middle),
or must the recurrent weights be trained? This is the reservoir-computing / echo-state control the de-risk needs:
 - if the fixed reservoir already wins -> the task doesn't test recurrent CREDIT, only readout (weak milestone; need
   a longer middle / harder task so the reservoir's fading memory can't hold the cue).
 - if the fixed reservoir fails but a full-context oracle wins -> recurrent weight learning is REQUIRED -> the task
   genuinely exercises the confirmed local credit rule (the real rung-3 milestone).

No GPU; small numpy. Readout trained by a LOCAL delta rule (post-error x pre-rate) -- biologically a perceptron.
"""
import numpy as np


def make_overlap_sequences(n_seq=4, middle_len=8):
    cues = list(range(n_seq))
    middle = list(range(n_seq, n_seq + middle_len))
    branches = list(range(n_seq + middle_len, n_seq + middle_len + n_seq))
    seqs = [[cues[i]] + middle + [branches[i]] for i in range(n_seq)]
    vocab = n_seq + middle_len + n_seq
    return seqs, vocab, {"cues": cues, "middle": middle, "branches": branches, "n_seq": n_seq, "L": middle_len}


def onehot(sym, vocab):
    v = np.zeros(vocab); v[sym] = 1.0; return v


def run_reservoir(seqs, vocab, N=200, kappa=0.9, g=1.2, w_in_scale=1.0, epochs=400, lr=0.2, seed=42,
                  train_noise=0.0, test_noise=0.0):
    """Fixed random leaky reservoir (echo-state); train ONLY the readout with a local delta rule. Return branch-acc.
    train_noise/test_noise = std of gaussian STATE noise injected each step (chaos/robustness probe: Laje-Buonomano)."""
    rng = np.random.default_rng(seed)
    W_rec = rng.normal(0, g / np.sqrt(N), (N, N))          # fixed random recurrent (spectral ~g)
    W_in = rng.normal(0, w_in_scale, (N, vocab))            # fixed random input projection
    W_out = np.zeros((vocab, N))                             # trained readout (local delta rule)

    def states_for(seq, noise, rr):
        u = np.zeros(N); S = []
        for t in range(len(seq)):
            u = kappa * u + W_rec @ np.tanh(u) + W_in @ onehot(seq[t], vocab)
            if noise > 0: u = u + rr.normal(0, noise, N)    # state noise -> chaotic divergence if g>1
            S.append(np.tanh(u))
        return np.array(S)

    for ep in range(epochs):
        for seq in seqs:
            S = states_for(seq, train_noise, rng)
            for t in range(len(seq) - 1):
                logits = W_out @ S[t]
                p = np.exp(logits - logits.max()); p /= p.sum()
                W_out += lr * np.outer(onehot(seq[t + 1], vocab) - p, S[t])   # LOCAL delta rule
    # eval branch-acc under test_noise, averaged over repeats (noise is stochastic)
    reps = 20 if test_noise > 0 else 1
    tot = 0
    for _ in range(reps):
        for seq in seqs:
            S = states_for(seq, test_noise, rng)
            t = len(seq) - 2
            tot += int(int(np.argmax(W_out @ S[t])) == seq[t + 1])
    return tot / (len(seqs) * reps)


if __name__ == "__main__":
    print("=== CLEAN (no noise) ===")
    for (n_seq, L, N, kappa) in [(4, 8, 200, 0.9), (4, 16, 200, 0.9), (4, 24, 200, 0.9), (8, 16, 300, 0.9)]:
        seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L)
        ch = 1.0 / n_seq
        accs = [run_reservoir(seqs, vocab, N=N, kappa=kappa, seed=s) for s in (42, 43, 44)]
        m = float(np.mean(accs))
        v = ("RESERVOIR SOLVES (readout suffices -> NOT a recurrent-credit test)" if m > 0.99 else
             "RESERVOIR FAILS (recurrent credit REQUIRED)" if m <= ch + 0.13 else "PARTIAL")
        print(f"  n_seq={n_seq} L={L} N={N} k={kappa}  chance={ch:.3f}  branch-acc={m:.3f} -> {v}")

    print("\n=== UNDER STATE NOISE (chaos regime g=1.5; does the fixed reservoir stay robust?) ===")
    for (n_seq, L, tn) in [(4, 16, 0.1), (4, 16, 0.3), (4, 16, 0.5), (4, 24, 0.3)]:
        seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L)
        ch = 1.0 / n_seq
        accs = [run_reservoir(seqs, vocab, N=200, kappa=0.9, g=1.5, seed=s, train_noise=tn, test_noise=tn) for s in (42, 43, 44)]
        m = float(np.mean(accs))
        v = ("still robust (noise too weak)" if m > 0.9 else
             "DEGRADES (credit rule can earn its keep here)" if m <= ch + 0.25 else "PARTIAL degrade")
        print(f"  n_seq={n_seq} L={L} noise={tn} g=1.5  chance={ch:.3f}  branch-acc={m:.3f} -> {v}")
