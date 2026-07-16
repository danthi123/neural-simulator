"""2026-07-15 — SPIKING port of the MDGL off-diagonal de-risk (the mission-central step: does the cross-neuron off-diagonal
temporal credit — validated on the rate trainable-RNN, `_mdgl_offdiagonal_credit_derisk.py` — work ON SPIKES?). Same
XOR-at-lag task + same three arms, but the neuron is a spiking LIF (Heaviside spike + a fast-sigmoid SURROGATE for the local
derivative — the substrate's own subthreshold sensitivity, the e-prop membrane surrogate). The MDGL Γ term is unchanged (a
one-hop cell-type-affinity broadcast of downstream neurons' error contributions × the PRIOR eligibility). This is the LIF
rung before the on-bridge Izhikevich/neuromodulator realization.

Arms: BPTT (surrogate-gradient through-time = ceiling) · EPROP (diagonal forward eligibility + DFA) · MDGL (e-prop + Γ) +
the zero-Γ / sign-flip / permuted anti-cheats. On the clean-gap regime (a seed where BPTT solves + the diagonal fails).
numpy-CPU; NO `sim/` edit (the spiking realization here is a surrogate LIF; the ON-BRIDGE port reuses sim/neuromodulators.py).

Run: PYTHONPATH=. python -u -m research.runners._mdgl_spiking_port_derisk --seeds 42
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._mdgl_offdiagonal_credit_derisk import make_task, _softmax


def _surr(v, thr, beta=4.0):
    """fast-sigmoid surrogate derivative of the spike function (the membrane's subthreshold sensitivity, low-variance)."""
    return 1.0 / (1.0 + beta * np.abs(v - thr)) ** 2


class SpikingRNN:
    """LIF: v_t = (1-a) v_{t-1}(1 - s_{t-1}) + a (W_rec s_{t-1} + W_in x + b); s_t = 1[v_t > thr] (hard reset). The recurrent
    input is the SPIKE train s (not a graded rate). Read-out over the membrane v at the RECALL step. All weights trainable."""
    def __init__(self, V, n, n_cls, seed, alpha=0.35, thr=0.6, n_types=6):
        rng = np.random.default_rng(seed)
        self.n, self.a, self.V, self.n_cls, self.thr = n, alpha, V, n_cls, thr
        self.W_in = rng.standard_normal((n, V)) * 0.5
        self.W_rec = rng.standard_normal((n, n)) / np.sqrt(n) * 0.9
        self.b = np.zeros(n)
        self.W_out = rng.standard_normal((n_cls, n)) * 0.2
        self.B = rng.standard_normal((n, n_cls)) / np.sqrt(n_cls)
        self.ctype = rng.integers(n_types, size=n)
        self.n_types = n_types

    def _affinity(self):
        A = np.zeros((self.n_types, self.n_types)); aw = np.abs(self.W_rec)
        for a in range(self.n_types):
            ia = self.ctype == a
            for b in range(self.n_types):
                ib = self.ctype == b
                if ia.any() and ib.any():
                    A[a, b] = aw[np.ix_(ia, ib)].mean()
        return A


def train(net, seqs, rpos, tgt, epochs, lr, seed, mode, gamma_sign=1.0, gamma_gain=1.0):
    rng = np.random.default_rng(seed * 7 + 3)
    n = net.n; a = net.a; thr = net.thr
    order = np.arange(len(seqs)); ys = list(tgt)
    if mode == "eprop_permuted":
        ys = list(np.array(tgt)[rng.permutation(len(tgt))]); mode = "eprop"
    P = {"W_in": net.W_in, "W_rec": net.W_rec, "W_out": net.W_out, "b": net.b}
    M = {k: np.zeros_like(v) for k, v in P.items()}; Vv = {k: np.zeros_like(v) for k, v in P.items()}; t_adam = [0]
    b1, b2, aeps = 0.9, 0.999, 1e-8

    def adam(key, g):
        t_adam[0] += 1; M[key] = b1 * M[key] + (1 - b1) * g; Vv[key] = b2 * Vv[key] + (1 - b2) * g * g
        mh = M[key] / (1 - b1 ** t_adam[0]); vh = Vv[key] / (1 - b2 ** t_adam[0])
        P[key] += lr * mh / (np.sqrt(vh) + aeps)

    for ep in range(epochs):
        rng.shuffle(order)
        if mode == "mdgl":
            Aff = net._affinity()
        for si in order:
            ids = seqs[si]; rp = rpos[si]; y = ys[si]; Tn = len(ids)
            if mode == "bptt":
                vs = [np.zeros(n)]; ss = [np.zeros(n)]; sgs = []
                for tt in range(Tn):
                    vp, sp = vs[-1], ss[-1]
                    v = (1 - a) * vp * (1 - sp) + a * (net.W_rec @ sp + net.W_in[:, ids[tt]] + net.b)
                    s = (v > thr).astype(float); vs.append(v); ss.append(s); sgs.append(_surr(v, thr))
                p = _softmax(net.W_out @ vs[rp + 1]); delta = -p; delta[y] += 1.0
                adam("W_out", np.outer(delta, vs[rp + 1]))
                gWrec = np.zeros_like(net.W_rec); gWin = np.zeros_like(net.W_in); gb = np.zeros(n)
                dv = net.W_out.T @ delta
                for tt in range(rp, -1, -1):
                    dpre = dv * a                                    # dL/d(pre-input at t)
                    gWrec += np.outer(dpre, ss[tt]); gWin[:, ids[tt]] += dpre; gb += dpre
                    ds = net.W_rec.T @ dpre                          # to the spike s_t
                    dv = (1 - a) * (1 - ss[tt]) * dv + ds * sgs[tt]  # through reset + surrogate spike
                adam("W_rec", gWrec); adam("W_in", gWin); adam("b", gb)
            else:
                v = np.zeros(n); s = np.zeros(n)
                eps_rec = np.zeros((n, n)); eps_in = np.zeros((n, net.V)); eps_rec_prev = np.zeros((n, n))
                for tt in range(Tn):
                    vp, sp = v, s
                    v = (1 - a) * vp * (1 - sp) + a * (net.W_rec @ sp + net.W_in[:, ids[tt]] + net.b)
                    s = (v > thr).astype(float); psi = _surr(v, thr)     # surrogate local derivative
                    eps_rec_prev = eps_rec.copy()
                    eps_rec = (1 - a) * eps_rec + psi[:, None] * sp[None, :]   # forward eligibility (presyn = spike sp)
                    onehot = np.zeros(net.V); onehot[ids[tt]] = 1.0
                    eps_in = (1 - a) * eps_in + psi[:, None] * onehot[None, :]
                    if tt == rp:
                        p = _softmax(net.W_out @ v); delta = -p; delta[y] += 1.0
                        adam("W_out", np.outer(delta, v))
                        L = net.B @ delta; a_err = psi * L
                        g_rec = L[:, None] * eps_rec
                        if mode == "mdgl":
                            part = (net.W_rec != 0.0)
                            AffJQ = Aff[net.ctype][:, net.ctype]
                            b_q = (AffJQ * part).T @ a_err
                            g_rec = g_rec + gamma_sign * gamma_gain * (b_q[None, :] * eps_rec_prev)
                        adam("W_rec", g_rec); adam("W_in", L[:, None] * eps_in); adam("b", L)
    return net


def evaluate(net, seqs, rpos, tgt):
    n = net.n; a = net.a; thr = net.thr; correct = 0
    for si, ids in enumerate(seqs):
        v = np.zeros(n); s = np.zeros(n)
        for tt in range(len(ids)):
            vp, sp = v, s
            v = (1 - a) * vp * (1 - sp) + a * (net.W_rec @ sp + net.W_in[:, ids[tt]] + net.b)
            s = (v > thr).astype(float)
            if tt == rpos[si]:
                correct += int(np.argmax(net.W_out @ v) == tgt[si]); break
    return correct / max(1, len(seqs))


def run_one(seed, task="xor", T=8, n=80, n_train=400, n_eval=200, epochs=60, lr=3e-3, K=4):
    tr, trp, trg, V, ncls = make_task(task, T, n_train, seed * 10 + 1, K=K)
    ev, evp, evg, _, _ = make_task(task, T, n_eval, seed * 10 + 2, K=K)
    chance = 1.0 / ncls; out = {"seed": seed, "task": task, "T": T, "chance": round(chance, 3)}
    def fresh(): return SpikingRNN(V, n, ncls, seed)
    A = {}
    for mode in ("bptt", "eprop", "mdgl", "mdgl_zeroGamma", "eprop_permuted"):
        net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, mode); A[mode] = round(evaluate(net, ev, evp, evg), 4)
    net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, "mdgl", gamma_sign=-1.0); A["mdgl_signflip"] = round(evaluate(net, ev, evp, evg), 4)
    out["acc"] = A; gap = A["bptt"] - A["eprop"]
    out["bptt_solves"] = bool(A["bptt"] > chance + 0.2)
    out["gap"] = round(gap, 3)
    out["mdgl_closes"] = bool(gap > 0.1 and (A["mdgl"] - A["eprop"]) > 0.25 * gap)
    out["zeroGamma_collapses"] = bool(abs(A["mdgl_zeroGamma"] - A["eprop"]) < 0.08)
    out["signflip_hurts"] = bool(A["mdgl_signflip"] < A["mdgl"] - 0.03)
    out["permuted_chance"] = bool(A["eprop_permuted"] < chance + 0.15)
    out["GO"] = bool(out["bptt_solves"] and gap > 0.1 and out["mdgl_closes"] and out["zeroGamma_collapses"] and out["signflip_hurts"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", default="xor"); ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default="research/findings/raw/_mdgl_spiking_port.json")
    a = ap.parse_args()
    rows = [run_one(s, task=a.task, T=a.lag, n=a.n, epochs=a.epochs) for s in a.seeds]
    for r in rows:
        A = r["acc"]
        print(f"[spk-mdgl s{r['seed']} {r['task']} T={r['T']}] chance={r['chance']} || BPTT={A['bptt']:.3f} eprop={A['eprop']:.3f} "
              f"MDGL={A['mdgl']:.3f} zeroG={A['mdgl_zeroGamma']:.3f} signflip={A['mdgl_signflip']:.3f} perm={A['eprop_permuted']:.3f} "
              f"| gap={r['gap']:.2f} closes={r['mdgl_closes']} zeroG_collapse={r['zeroGamma_collapses']} signflip_hurts={r['signflip_hurts']} GO={r['GO']}", flush=True)
    print(f"[spk-mdgl] {sum(x['GO'] for x in rows)}/{len(rows)} GO (the off-diagonal MDGL mechanism works ON SPIKES: closes the diagonal-vs-BPTT gap, anti-cheat-clean)", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
