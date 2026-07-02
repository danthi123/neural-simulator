"""DECISIVE scratch prototype for the Fork-2 rung-3 de-risk (mechanism side).

QUESTION: on the NOISY two-overlapping-sequences next-symbol task (where a fixed reservoir DEGRADES and Markov
predictors are provably at chance), can our confirmed FULLY-LOCAL credit rule -- RFLO/e-prop-style random-feedback
readout-error + a per-synapse eligibility trace (NO weight transport, NO BPTT) -- train the recurrent weights to be
MORE NOISE-ROBUST than a fixed reservoir? If trained_recurrent >> fixed_reservoir under matched noise, recurrent
credit is load-bearing and Fork-2 is a real rung-3 milestone (learn high-order context robustly via a local rule).

Arms:
  trained_recurrent : train W_rec, W_in, W_out (RFLO local credit on rec/in; delta rule on out)
  fixed_reservoir   : freeze W_rec, W_in (random); train ONLY W_out (the echo-state control)
  wrong_feedback    : trained, but flip the sign of the random-feedback learning signal (anti-cheat: should NOT learn)
Locality: hidden learning signal = B @ readout_error with B FIXED RANDOM (feedback alignment) -- never W_out.T.
Small numpy; full cores fine. Report branch-acc under test noise, multi-seed.
"""
import numpy as np


def make_overlap_sequences(n_seq=4, middle_len=16):
    cues = list(range(n_seq)); middle = list(range(n_seq, n_seq + middle_len))
    branches = list(range(n_seq + middle_len, n_seq + middle_len + n_seq))
    seqs = [[cues[i]] + middle + [branches[i]] for i in range(n_seq)]
    return seqs, n_seq + middle_len + n_seq


def onehot(s, V):
    v = np.zeros(V); v[s] = 1.0; return v


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def run(arm, seqs, V, N=200, kappa=0.9, g=1.5, epochs=500, lr_out=0.2, lr_rec=0.05,
        noise=0.3, alpha_e=0.9, seed=42):
    rng = np.random.default_rng(seed)
    W_rec = rng.normal(0, g / np.sqrt(N), (N, N))
    W_in = rng.normal(0, 1.0, (N, V))
    W_out = np.zeros((V, N))
    B = rng.normal(0, 1.0 / np.sqrt(V), (N, V))            # FIXED random feedback (feedback alignment; no transport)
    train_rec = (arm in ("trained_recurrent", "wrong_feedback"))
    sign = -1.0 if arm == "wrong_feedback" else 1.0

    for ep in range(epochs):
        for seq in seqs:
            u = np.zeros(N); r_prev = np.tanh(u)
            eps_rec = np.zeros((N, N)); eps_in = np.zeros((N, V))
            for t in range(len(seq) - 1):
                x = onehot(seq[t], V)
                u = kappa * u + W_rec @ r_prev + W_in @ x
                u = u + rng.normal(0, noise, N)             # state noise (chaos/robustness regime)
                r = np.tanh(u); rp = 1.0 - r * r            # tanh'
                logits = W_out @ r; p = _softmax(logits)
                err = onehot(seq[t + 1], V) - p             # readout error (local at the output)
                W_out += lr_out * np.outer(err, r)          # LOCAL delta rule
                if train_rec:
                    L = sign * (B @ err)                    # hidden learning signal via FIXED random feedback (local)
                    eps_rec = alpha_e * eps_rec + r_prev[None, :]   # per-synapse eligibility eps_ji = a*eps + pre_i
                    eps_in = alpha_e * eps_in + x[None, :]
                    W_rec += lr_rec * (L * rp)[:, None] * eps_rec   # RFLO/e-prop: (fb-learning-signal * phi') x eligibility
                    W_in += lr_rec * (L * rp)[:, None] * eps_in
                r_prev = r
    # eval under noise (avg over repeats)
    reps = 20; tot = 0
    for _ in range(reps):
        for seq in seqs:
            u = np.zeros(N); r_prev = np.tanh(u); pred = None
            for t in range(len(seq) - 1):
                u = kappa * u + W_rec @ r_prev + W_in @ onehot(seq[t], V) + rng.normal(0, noise, N)
                r = np.tanh(u)
                if t == len(seq) - 2: pred = int(np.argmax(W_out @ r))
                r_prev = r
            tot += int(pred == seq[-1])
    return tot / (len(seqs) * reps)


if __name__ == "__main__":
    seqs, V = make_overlap_sequences(n_seq=4, middle_len=16)
    print(f"task: n_seq=4 middle_len=16 vocab={V} chance=0.250  (noise=0.3, g=1.5)")
    for arm in ["fixed_reservoir", "trained_recurrent", "wrong_feedback"]:
        accs = [run(arm, seqs, V, seed=s) for s in (42, 43, 44)]
        print(f"  {arm:18s} branch-acc = {np.mean(accs):.3f}  {[round(a,3) for a in accs]}")
