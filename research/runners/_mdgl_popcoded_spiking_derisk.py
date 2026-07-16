"""2026-07-15 — POPULATION-CODED spiking MDGL (the named surpass for the spike-port degradation): the LIF port lost the
off-diagonal credit magnitude (+11% vs rate +48-64%) because the eligibility is built from SPARSE BINARY spikes. The
project's own lever that closed the FEEDFORWARD spiking credit this session was POPULATION CODING (2026-07-14: e-prop
forward K=1 0.47 -> K=8 0.877 ≈ LIF ceiling). Apply it here: each logical unit = POP_K spiking neurons (hard 0/1 spikes,
per-neuron bias tiling the threshold); the unit's OUTPUT = the POOLED spike-rate r_u = mean_k s_{u,k} (a graded,
low-variance signal). The recurrent input, the eligibility, and the MDGL Γ all operate at the UNIT level on r (graded) —
recovering the graded presynaptic signal the off-diagonal broadcast needs, from a genuine POPULATION of spiking neurons.

Arms + anti-cheats identical to `_mdgl_spiking_port_derisk`; a POP_K sweep is the surpass lever (K=1 = the degraded LIF).
GATE: population coding recovers the off-diagonal magnitude toward the rate level (K↑ -> MDGL gap-close ↑ >> the K=1 +11%),
anti-cheat-clean. numpy-CPU; NO `sim/` edit (the on-bridge realization reuses the same K-pool + sim/neuromodulators.py).

Run: PYTHONPATH=. python -u -m research.runners._mdgl_popcoded_spiking_derisk --seeds 42 --pop-k 8
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._mdgl_offdiagonal_credit_derisk import make_task, _softmax


class PopSpikingRNN:
    """n_units logical units, each = POP_K spiking neurons (hard spike + per-neuron bias offset tiling the threshold). The
    unit OUTPUT r_u = mean over its POP_K neurons of the hard spike = the POOLED (graded, low-variance) population rate,
    which drives the recurrence + carries the eligibility. W_rec/W_in/W_out are UNIT-level; all trainable."""
    def __init__(self, V, n_units, n_cls, seed, pop_k=8, alpha=0.35, thr=0.6, n_types=6, bias_spread=0.25):
        rng = np.random.default_rng(seed)
        self.nu, self.K, self.a, self.V, self.n_cls, self.thr = n_units, pop_k, alpha, V, n_cls, thr
        self.W_in = rng.standard_normal((n_units, V)) * 0.5
        self.W_rec = rng.standard_normal((n_units, n_units)) / np.sqrt(n_units) * 0.9
        self.b = np.zeros(n_units)
        self.bias_k = rng.standard_normal((n_units, pop_k)) * bias_spread   # per-neuron threshold tiling (the population)
        self.W_out = rng.standard_normal((n_cls, n_units)) * 0.2
        self.B = rng.standard_normal((n_units, n_cls)) / np.sqrt(n_cls)
        self.ctype = rng.integers(n_types, size=n_units); self.n_types = n_types

    def _affinity(self):
        A = np.zeros((self.n_types, self.n_types)); aw = np.abs(self.W_rec)
        for a in range(self.n_types):
            ia = self.ctype == a
            for b in range(self.n_types):
                ib = self.ctype == b
                if ia.any() and ib.any():
                    A[a, b] = aw[np.ix_(ia, ib)].mean()
        return A


def _surr(x, beta=4.0):
    return 1.0 / (1.0 + beta * np.abs(x)) ** 2


def train(net, seqs, rpos, tgt, epochs, lr, seed, mode, gamma_sign=1.0, gamma_gain=1.0):
    rng = np.random.default_rng(seed * 7 + 3)
    nu, K, a, thr = net.nu, net.K, net.a, net.thr
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

    def step(v, r, x_id):                                    # one forward step -> (v', r', pooled surrogate psi_u)
        drive = net.W_rec @ r + net.W_in[:, x_id] + net.b    # unit-level pre-input (from the pooled rates)
        vph = (1 - a) * v * (1 - (v > thr)) + a * drive[:, None]     # (nu,K) membrane, reset per-neuron
        vk = vph + net.bias_k                                # per-neuron threshold tiling
        s = (vk > thr).astype(float)                         # hard spike per neuron
        r_new = s.mean(axis=1)                               # POOLED population rate (graded)
        psi_u = _surr(vk - thr).mean(axis=1)                 # pooled surrogate (low-variance)
        return vph, r_new, psi_u

    for ep in range(epochs):
        rng.shuffle(order)
        if mode == "mdgl":
            Aff = net._affinity()
        for si in order:
            ids = seqs[si]; rp = rpos[si]; y = ys[si]; Tn = len(ids)
            v = np.zeros((nu, K)); r = np.zeros(nu)
            if mode == "bptt":
                vs = [v]; rs = [r]; psis = []; drives = []
                for tt in range(Tn):
                    drive = net.W_rec @ rs[-1] + net.W_in[:, ids[tt]] + net.b
                    vph = (1 - a) * vs[-1] * (1 - (vs[-1] > thr)) + a * drive[:, None]
                    vk = vph + net.bias_k; s = (vk > thr).astype(float)
                    vs.append(vph); rs.append(s.mean(axis=1)); psis.append(_surr(vk - thr)); drives.append(drive)
                p = _softmax(net.W_out @ rs[rp + 1]); delta = -p; delta[y] += 1.0
                adam("W_out", np.outer(delta, rs[rp + 1]))
                dr = net.W_out.T @ delta                     # dL/dr_u at the read step
                gWrec = np.zeros_like(net.W_rec); gWin = np.zeros_like(net.W_in); gb = np.zeros(nu)
                for tt in range(rp, -1, -1):
                    dpre = (dr * psis[tt].mean(axis=1)) * a    # through the pooled spike -> the drive
                    gWrec += np.outer(dpre, rs[tt]); gWin[:, ids[tt]] += dpre; gb += dpre
                    dr = net.W_rec.T @ dpre                   # back through the recurrence (unit-level)
                adam("W_rec", gWrec); adam("W_in", gWin); adam("b", gb)
            else:
                eps_rec = np.zeros((nu, nu)); eps_in = np.zeros((nu, net.V)); eps_prev = np.zeros((nu, nu))
                for tt in range(Tn):
                    r_prev = r
                    v, r, psi_u = step(v, r, ids[tt])
                    eps_prev = eps_rec.copy()
                    eps_rec = (1 - a) * eps_rec + psi_u[:, None] * r_prev[None, :]   # graded presyn = pooled rate
                    onehot = np.zeros(net.V); onehot[ids[tt]] = 1.0
                    eps_in = (1 - a) * eps_in + psi_u[:, None] * onehot[None, :]
                    if tt == rp:
                        p = _softmax(net.W_out @ r); delta = -p; delta[y] += 1.0
                        adam("W_out", np.outer(delta, r))
                        L = net.B @ delta; a_err = psi_u * L
                        g_rec = L[:, None] * eps_rec
                        if mode == "mdgl":
                            part = (net.W_rec != 0.0); AffJQ = Aff[net.ctype][:, net.ctype]
                            b_q = (AffJQ * part).T @ a_err
                            g_rec = g_rec + gamma_sign * gamma_gain * (b_q[None, :] * eps_prev)
                        adam("W_rec", g_rec); adam("W_in", L[:, None] * eps_in); adam("b", L)
    return net


def evaluate(net, seqs, rpos, tgt):
    nu, K, a, thr = net.nu, net.K, net.a, net.thr; correct = 0
    for si, ids in enumerate(seqs):
        v = np.zeros((nu, K)); r = np.zeros(nu)
        for tt in range(len(ids)):
            drive = net.W_rec @ r + net.W_in[:, ids[tt]] + net.b
            vph = (1 - a) * v * (1 - (v > thr)) + a * drive[:, None]
            v = vph; s = ((vph + net.bias_k) > thr).astype(float); r = s.mean(axis=1)
            if tt == rpos[si]:
                correct += int(np.argmax(net.W_out @ r) == tgt[si]); break
    return correct / max(1, len(seqs))


def run_one(seed, task="xor", T=8, nu=80, pop_k=8, n_train=400, n_eval=200, epochs=55, lr=3e-3, K=4, ggain=0.4):
    tr, trp, trg, V, ncls = make_task(task, T, n_train, seed * 10 + 1, K=K)
    ev, evp, evg, _, _ = make_task(task, T, n_eval, seed * 10 + 2, K=K)
    chance = 1.0 / ncls; out = {"seed": seed, "task": task, "T": T, "pop_k": pop_k, "chance": round(chance, 3)}
    def fresh(): return PopSpikingRNN(V, nu, ncls, seed, pop_k=pop_k)
    A = {}
    for mode in ("bptt", "eprop", "eprop_permuted"):
        net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, mode); A[mode] = round(evaluate(net, ev, evp, evg), 4)
    for mm, gg in (("mdgl", ggain), ("mdgl_zeroGamma", 0.0)):
        net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, "mdgl", gamma_gain=gg); A[mm] = round(evaluate(net, ev, evp, evg), 4)
    net = fresh(); train(net, tr, trp, trg, epochs, lr, seed, "mdgl", gamma_sign=-1.0, gamma_gain=ggain); A["mdgl_signflip"] = round(evaluate(net, ev, evp, evg), 4)
    out["acc"] = A; gap = A["bptt"] - A["eprop"]; out["gap"] = round(gap, 3)
    out["closed_pct"] = round(100 * (A["mdgl"] - A["eprop"]) / max(gap, 1e-9), 1)
    out["GO"] = bool(A["bptt"] > chance + 0.2 and gap > 0.1 and (A["mdgl"] - A["eprop"]) > 0.3 * gap
                     and abs(A["mdgl_zeroGamma"] - A["eprop"]) < 0.08 and A["mdgl_signflip"] < A["mdgl"] - 0.03
                     and A["eprop_permuted"] < chance + 0.15)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", default="xor"); ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--nu", type=int, default=80); ap.add_argument("--pop-k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=55); ap.add_argument("--ggain", type=float, default=0.4)
    ap.add_argument("--out", default="research/findings/raw/_mdgl_popcoded_spiking.json")
    a = ap.parse_args()
    rows = [run_one(s, task=a.task, T=a.lag, nu=a.nu, pop_k=a.pop_k, epochs=a.epochs, ggain=a.ggain) for s in a.seeds]
    for r in rows:
        A = r["acc"]
        print(f"[popmdgl s{r['seed']} K={r['pop_k']} {r['task']} T={r['T']}] chance={r['chance']} || BPTT={A['bptt']:.3f} "
              f"eprop={A['eprop']:.3f} MDGL={A['mdgl']:.3f} zeroG={A['mdgl_zeroGamma']:.3f} signflip={A['mdgl_signflip']:.3f} "
              f"perm={A['eprop_permuted']:.3f} | gap={r['gap']:.2f} closed={r['closed_pct']:.0f}% GO={r['GO']}", flush=True)
    print(f"[popmdgl] {sum(x['GO'] for x in rows)}/{len(rows)} GO", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
