"""2026-07-15 — the MDGL off-diagonal de-risk on a PROPER TRAINABLE-RNN testbed (the deep-research gate wf wdublhh7v +
controller source-reads of Merin arXiv:2603.28750 + MDGL Liu-2021 PMC8713766). The reservoir-W_rec-fine-tune substrate
is the WRONG testbed (even BPTT can't solve XOR on it — the fixed reservoir natively does the linear tasks, the nonlinear
task isn't cleanly learnable by W_rec-fine-tune; `2026-07-14-controlled-lag-...` + this session's BPTT positive control).
MDGL's OWN validation used a TRAINABLE RNN on delayed-match-to-sample / evidence-accumulation. So this builds that: a small
TRAINABLE RNN (W_in/W_rec/W_out all learned) on a recurrent-computation task, three credit arms:
  - BPTT (full backprop-through-time = the ceiling / positive control)
  - EPROP (diagonal forward eligibility, transport-free DFA feedback — the rule that ZEROES the cross-neuron term)
  - MDGL (e-prop + the one-hop off-diagonal Γ term: each synapse also sees its presynaptic neuron's postsynaptic
    partners' error contributions, broadcast one hop back through W_rec × the PRIOR eligibility; Liu-2021 Eq.20
    Γ_pq,t = (Σ affinity · Σ_{partners} a_j) · e_pq,t-1, a_j = the neuron's error contribution)

GATE (the clean, apples-to-apples test the reservoir couldn't give): BPTT SOLVES the task (>> chance, the ceiling);
diagonal EPROP LAGS BPTT; MDGL closes >= a meaningful fraction of the (BPTT − eprop) gap. Anti-cheat (load-bearing):
ZERO the Γ term -> MDGL collapses to the eprop baseline (the "simpler mechanism doesn't also pass" control); permuted-label
-> chance; sign-flip the Γ -> must HURT. numpy-CPU; NO `sim/` edit.

Run: PYTHONPATH=. python -u -m research.runners._mdgl_offdiagonal_credit_derisk --task dms --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ------------------------------ tasks (recurrent computation; fixed reservoir provably can't; BPTT can) ------------------
def make_task(kind, T, n_trials, seed, K=4, F=6):
    """Each trial: a token stream; the target at the RECALL step is a function of temporally-separated cues.
    kind='dms'  : delayed-match-to-sample -> target = 1 if the two cue symbols MATCH else 0 (binary).
    kind='xor'  : target = (x1 + x2) mod K (K-way; the nonlinear combination BPTT-on-reservoir failed).
    Tokens: 0..K-1 = content symbols; K = STORE; K+1 = RECALL; K+2..K+1+F = fillers. Input is one-hot."""
    rng = np.random.default_rng(seed)
    STORE, RECALL, FILL0 = K, K + 1, K + 2
    V = K + 2 + F
    seqs, rpos, tgt = [], [], []
    for _ in range(n_trials):
        fill = lambda: [FILL0 + int(rng.integers(F)) for _ in range(T)]
        if kind == "parity3":                            # 3 temporally-separated cues -> reliably breaks the diagonal
            x1, x2, x3 = (int(rng.integers(K)) for _ in range(3))
            seq = [STORE, x1] + fill() + [x2] + fill() + [x3, RECALL]
            y = int((x1 + x2 + x3) % K)
        else:
            x1 = int(rng.integers(K)); x2 = int(rng.integers(K))
            seq = [STORE, x1] + fill() + [x2, RECALL]
            y = int(x1 == x2) if kind == "dms" else int((x1 + x2) % K)
        seqs.append(np.asarray(seq, dtype=np.int64))
        rpos.append(len(seq) - 1)                        # predict the target AT the RECALL step
        tgt.append(y)
    n_cls = 2 if kind == "dms" else K
    return seqs, rpos, tgt, V, n_cls


# ------------------------------ the trainable RNN + the three credit rules --------------------------------------------
class TrainableRNN:
    """h_t = (1-a) h_{t-1} + a tanh(W_rec h_{t-1} + W_in[:,x_t] + b). Read-out W_out over h at the RECALL step.
    ALL of W_in/W_rec/W_out trainable (unlike the fixed reservoir). Cell types for MDGL = a fixed random partition."""
    def __init__(self, V, n, n_cls, seed, alpha=0.3, n_types=6):
        rng = np.random.default_rng(seed)
        self.n, self.a, self.V, self.n_cls = n, alpha, V, n_cls
        self.W_in = rng.standard_normal((n, V)) * 0.3
        self.W_rec = rng.standard_normal((n, n)) / np.sqrt(n) * 0.9    # sub-critical trainable init
        self.b = np.zeros(n)
        self.W_out = rng.standard_normal((n_cls, n)) * 0.1
        self.B = rng.standard_normal((n, n_cls)) / np.sqrt(n_cls)      # fixed random DFA feedback (transport-free)
        self.ctype = rng.integers(n_types, size=n)                    # each neuron's cell type (for MDGL affinity)
        self.n_types = n_types

    def _affinity(self):
        """cell-type affinity <w_ab> = mean |W_rec| between type-a (post) and type-b (pre) — MDGL's tractable replacement
        for neuron-specific weights (computed once from the current W_rec)."""
        A = np.zeros((self.n_types, self.n_types))
        aw = np.abs(self.W_rec)
        for a in range(self.n_types):
            ia = self.ctype == a
            for b in range(self.n_types):
                ib = self.ctype == b
                if ia.any() and ib.any():
                    A[a, b] = aw[np.ix_(ia, ib)].mean()
        return A


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def train(net, seqs, rpos, tgt, epochs, lr, seed, mode, gamma_sign=1.0):
    """mode in {bptt, eprop, mdgl, mdgl_zeroGamma, eprop_permuted}. Trains W_in/W_rec/W_out.
    eprop/mdgl: W_rec by forward eligibility (+Γ for mdgl); W_out by local delta; W_in by input eligibility.
    bptt: full through-time backprop. Adam on all params."""
    rng = np.random.default_rng(seed * 7 + 3)
    n = net.n; a = net.a
    order = np.arange(len(seqs))
    ys = list(tgt)
    if mode == "eprop_permuted":
        ys = list(np.array(tgt)[rng.permutation(len(tgt))]); mode = "eprop"
    # Adam state
    P = {"W_in": net.W_in, "W_rec": net.W_rec, "W_out": net.W_out, "b": net.b}
    M = {k: np.zeros_like(v) for k, v in P.items()}; Vv = {k: np.zeros_like(v) for k, v in P.items()}; t_adam = [0]
    b1, b2, eps = 0.9, 0.999, 1e-8

    def adam(key, g):
        t_adam[0] += 1; M[key] = b1 * M[key] + (1 - b1) * g; Vv[key] = b2 * Vv[key] + (1 - b2) * g * g
        mh = M[key] / (1 - b1 ** t_adam[0]); vh = Vv[key] / (1 - b2 ** t_adam[0])
        P[key] += lr * mh / (np.sqrt(vh) + eps)

    for ep in range(epochs):
        rng.shuffle(order)
        if mode == "mdgl":
            Aff = net._affinity()                                    # refresh affinity each epoch
        for si in order:
            ids = seqs[si]; rp = rpos[si]; y = ys[si]; Tn = len(ids)
            if mode == "bptt":
                # forward, store; backward through time
                hs = [np.zeros(n)]; acts = []
                for tt in range(Tn):
                    hp = hs[-1]; pre = net.W_rec @ hp + net.W_in[:, ids[tt]] + net.b
                    act = np.tanh(pre); h = (1 - a) * hp + a * act
                    hs.append(h); acts.append(act)
                p = _softmax(net.W_out @ hs[rp + 1]); delta = -p; delta[y] += 1.0
                gWout = np.outer(delta, hs[rp + 1])
                gWrec = np.zeros_like(net.W_rec); gWin = np.zeros_like(net.W_in); gb = np.zeros(n)
                dh = net.W_out.T @ delta                             # dL/dh at the read step
                for tt in range(rp, -1, -1):
                    dpre = dh * a * (1 - acts[tt] ** 2)
                    gWrec += np.outer(dpre, hs[tt]); gWin[:, ids[tt]] += dpre; gb += dpre
                    dh = (1 - a) * dh + net.W_rec.T @ dpre           # back through the recurrence
                adam("W_out", gWout); adam("W_rec", gWrec); adam("W_in", gWin); adam("b", gb)
            else:
                # forward with eligibility (e-prop / MDGL)
                h = np.zeros(n); eps_rec = np.zeros((n, n)); eps_in = np.zeros((n, net.V))
                eps_rec_prev = np.zeros((n, n)); a_err = np.zeros(n)
                for tt in range(Tn):
                    hp = h; pre = net.W_rec @ hp + net.W_in[:, ids[tt]] + net.b
                    act = np.tanh(pre); h = (1 - a) * hp + a * act
                    psi = a * (1 - act ** 2)                          # membrane surrogate ψ_j
                    eps_rec_prev = eps_rec.copy()                     # e_pq,t-1 (MDGL's Γ uses the PRIOR eligibility)
                    eps_rec = (1 - a) * eps_rec + psi[:, None] * hp[None, :]     # forward eligibility for W_rec[p,q]
                    onehot = np.zeros(net.V); onehot[ids[tt]] = 1.0
                    eps_in = (1 - a) * eps_in + psi[:, None] * onehot[None, :]
                    if tt == rp:
                        p = _softmax(net.W_out @ h); delta = -p; delta[y] += 1.0
                        adam("W_out", np.outer(delta, h))
                        L = net.B @ delta                            # transport-free DFA learning signal per neuron p
                        a_err = psi * L                              # neuron's error contribution a_j (Liu-2021)
                        g_rec = L[:, None] * eps_rec
                        if mode == "mdgl":
                            # MDGL Γ_pq,t = ( Σ_j affinity(type_j,type_q) · a_err[j] · 1[q→j] ) · eps_rec_prev[p,q]
                            # i.e. each presynaptic neuron q broadcasts its POSTsynaptic partners' error contributions
                            # a_j (weighted by cell-type affinity), one hop, × the PRIOR eligibility (Liu-2021 Eq.20).
                            part = (net.W_rec != 0.0)                 # part[j,q]=True iff W_rec[j,q]!=0 iff q→j
                            AffJQ = Aff[net.ctype][:, net.ctype]      # (n,n): AffJQ[j,q] = affinity(type_j, type_q)
                            b_q = (AffJQ * part).T @ a_err            # b_q[q] = Σ_j AffJQ[j,q]·part[j,q]·a_err[j]
                            gamma = gamma_sign * (b_q[None, :] * eps_rec_prev)   # (n,n): Γ[p,q] = b_q[q]·e_pq,t-1
                            g_rec = g_rec + gamma
                        elif mode == "mdgl_zeroGamma":
                            pass                                     # == eprop (the anti-cheat)
                        g_in = L[:, None] * eps_in
                        adam("W_rec", g_rec); adam("W_in", g_in); adam("b", L * 1.0)
    return net


def evaluate(net, seqs, rpos, tgt):
    n = net.n; a = net.a; correct = 0
    for si, ids in enumerate(seqs):
        h = np.zeros(n)
        for tt in range(len(ids)):
            h = (1 - a) * h + a * np.tanh(net.W_rec @ h + net.W_in[:, ids[tt]] + net.b)
            if tt == rpos[si]:
                correct += int(np.argmax(net.W_out @ h) == tgt[si])
                break
    return correct / max(1, len(seqs))


def run_one(seed, task="dms", T=8, n=80, n_train=500, n_eval=250, epochs=40, lr=3e-3, K=4):
    tr, trp, trg, V, ncls = make_task(task, T, n_train, seed * 10 + 1, K=K)
    ev, evp, evg, _, _ = make_task(task, T, n_eval, seed * 10 + 2, K=K)
    chance = 1.0 / ncls
    out = {"seed": seed, "task": task, "T": T, "chance": round(chance, 3), "n_cls": ncls}
    def fresh(): return TrainableRNN(V, n, ncls, seed)
    accs = {}
    for mode in ("bptt", "eprop", "mdgl", "mdgl_zeroGamma", "eprop_permuted"):
        net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, mode)
        accs[mode] = round(evaluate(net, ev, evp, evg), 4)
    # sign-flip Γ anti-cheat
    net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, "mdgl", gamma_sign=-1.0)
    accs["mdgl_signflip"] = round(evaluate(net, ev, evp, evg), 4)
    out["acc"] = accs
    bptt, ep, md = accs["bptt"], accs["eprop"], accs["mdgl"]
    gap = bptt - ep
    out["bptt_solves"] = bool(bptt > chance + 0.2)
    out["eprop_lags"] = bool(ep < bptt - 0.1)
    out["mdgl_closes_gap"] = bool(gap > 0.05 and (md - ep) > 0.4 * gap)
    out["zeroGamma_collapses"] = bool(abs(accs["mdgl_zeroGamma"] - ep) < 0.08)
    out["permuted_chance"] = bool(accs["eprop_permuted"] < chance + 0.15)
    out["GO"] = bool(out["bptt_solves"] and out["eprop_lags"] and out["mdgl_closes_gap"]
                     and out["zeroGamma_collapses"] and out["permuted_chance"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", default="dms", choices=["dms", "xor"])
    ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="research/findings/raw/_mdgl_offdiagonal_credit.json")
    a = ap.parse_args()
    rows = [run_one(s, task=a.task, T=a.lag, n=a.n, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        A = r["acc"]
        print(f"[mdgl s{r['seed']} {r['task']} T={r['T']}] chance={r['chance']} || BPTT={A['bptt']:.3f} | eprop(diag)={A['eprop']:.3f} "
              f"| MDGL={A['mdgl']:.3f} | zeroGamma={A['mdgl_zeroGamma']:.3f} | signflip={A['mdgl_signflip']:.3f} | permuted={A['eprop_permuted']:.3f} "
              f"|| bptt_solves={r['bptt_solves']} eprop_lags={r['eprop_lags']} mdgl_closes={r['mdgl_closes_gap']} zeroG_collapses={r['zeroGamma_collapses']} GO={r['GO']}", flush=True)
    ngo = sum(x["GO"] for x in rows)
    print(f"[mdgl] {ngo}/{len(rows)} GO (BPTT solves + diagonal e-prop lags + MDGL's off-diagonal Γ closes the gap + zeroing Γ collapses to e-prop)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
