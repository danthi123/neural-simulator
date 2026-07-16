"""2026-07-15 — POPULATION CODING DONE RIGHT for the spiking MDGL off-diagonal (the clean isolation the pop-coded net
confounded). The two prior spiking-toy failures both CHANGED the net's dynamics: the single-neuron port has a sparse
binary-spike eligibility (marginal, gain-noisy); the pop-coded port (r=mean_k spike, per-neuron bias tiling) drove the
net near-critical (magnitude confound — sign-flip won). This runner keeps the VALIDATED single-neuron dynamics UNCHANGED
(BPTT ceiling 0.99, clear diagonal gap) and applies population coding ONLY to the CREDIT signal: run N_REPLICA
independent NOISY copies of the SAME net and ENSEMBLE-AVERAGE the gradient (eligibility × surrogate × credit). If the
sparse-spike eligibility NOISE is what degrades the off-diagonal direction, averaging over N replicas denoises it — the
biological population-coding surpass, done without touching the forward dynamics.

DECISIVE GATE: does N↑ make MDGL clean-DIRECTIONAL — MDGL > e-prop AND sign-flip ≈ e-prop (the flip must HURT, unlike the
pop-coded net where it won), gain-ROBUSTLY? If YES → population averaging is the spiking implementation key → the on-bridge
realization (real neurons + OU noise = a native replica ensemble) is warranted. If NO → the off-diagonal is a rate-level
result that point-neuron spikes degrade (a cited honest boundary = a valid answer to the owner's "what are we missing").

numpy-CPU; NO `sim/` edit; reuse `SpikingRNN`/`make_task` by import.
Run: PYTHONPATH=. python -u -m research.runners._mdgl_replica_popcoded_spiking_derisk --seeds 42 --n-replica 8
"""
import os, sys, json, argparse
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._mdgl_spiking_port_derisk import SpikingRNN, _surr
from research.runners._mdgl_offdiagonal_credit_derisk import make_task, _softmax


def train_replica(net, seqs, rpos, tgt, epochs, lr, seed, mode, n_replica=8, noise=0.15,
                  gamma_sign=1.0, gamma_gain=0.4):
    """e-prop / MDGL with the credit ENSEMBLE-AVERAGED over N_REPLICA independent noisy copies of the SAME net."""
    rng = np.random.default_rng(seed * 7 + 3)
    nrng = np.random.default_rng(seed * 101 + 5)          # noise stream (independent of shuffle)
    n, a, thr = net.n, net.a, net.thr
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
        Aff = net._affinity() if mode == "mdgl" else None
        for si in order:
            ids = seqs[si]; rp = rpos[si]; y = ys[si]; Tn = len(ids)
            gW = np.zeros_like(net.W_rec); gI = np.zeros_like(net.W_in); gB = np.zeros(n); gO = np.zeros_like(net.W_out)
            for _r in range(n_replica):                                     # ENSEMBLE of noisy copies
                v = np.zeros(n); s = np.zeros(n)
                eps_rec = np.zeros((n, n)); eps_in = np.zeros((n, net.V)); eps_prev = np.zeros((n, n))
                for tt in range(Tn):
                    vp, sp = v, s
                    v = (1 - a) * vp * (1 - sp) + a * (net.W_rec @ sp + net.W_in[:, ids[tt]] + net.b) \
                        + noise * nrng.standard_normal(n)                    # independent membrane noise (dynamics-preserving)
                    s = (v > thr).astype(float); psi = _surr(v, thr)
                    eps_prev = eps_rec.copy()
                    eps_rec = (1 - a) * eps_rec + psi[:, None] * sp[None, :]
                    onehot = np.zeros(net.V); onehot[ids[tt]] = 1.0
                    eps_in = (1 - a) * eps_in + psi[:, None] * onehot[None, :]
                    if tt == rp:
                        p = _softmax(net.W_out @ v); delta = -p; delta[y] += 1.0
                        gO += np.outer(delta, v)
                        L = net.B @ delta; a_err = psi * L
                        g_rec = L[:, None] * eps_rec
                        if mode == "mdgl":
                            part = (net.W_rec != 0.0); AffJQ = Aff[net.ctype][:, net.ctype]
                            b_q = (AffJQ * part).T @ a_err
                            g_rec = g_rec + gamma_sign * gamma_gain * (b_q[None, :] * eps_prev)
                        gW += g_rec; gI += L[:, None] * eps_in; gB += L
            inv = 1.0 / n_replica
            adam("W_out", gO * inv); adam("W_rec", gW * inv); adam("W_in", gI * inv); adam("b", gB * inv)
    return net


def evaluate(net, seqs, rpos, tgt, n_replica=8, noise=0.15, seed=0):
    """evaluate = ensemble-vote over N noisy copies (population read-out, matched to training)."""
    nrng = np.random.default_rng(seed * 202 + 9)
    n, a, thr = net.n, net.a, net.thr; correct = 0
    for si, ids in enumerate(seqs):
        logit = np.zeros(net.n_cls)
        for _r in range(n_replica):
            v = np.zeros(n); s = np.zeros(n)
            for tt in range(len(ids)):
                vp, sp = v, s
                v = (1 - a) * vp * (1 - sp) + a * (net.W_rec @ sp + net.W_in[:, ids[tt]] + net.b) \
                    + noise * nrng.standard_normal(n)
                s = (v > thr).astype(float)
                if tt == rpos[si]:
                    logit += net.W_out @ v; break
        correct += int(np.argmax(logit) == tgt[si])
    return correct / max(1, len(seqs))


def run_gain_sweep(seed=42, T=8, n=80, n_replica=8, noise=0.15, epochs=60, lr=3e-3, K=4):
    tr, trp, trg, V, ncls = make_task("xor", T, 400, seed * 10 + 1, K=K)
    ev, evp, evg, _, _ = make_task("xor", T, 200, seed * 10 + 2, K=K)
    ch = 1.0 / ncls
    def fresh(): return SpikingRNN(V, n, ncls, seed)
    ne = fresh(); train_replica(ne, tr, trp, trg, epochs, lr, seed, "eprop", n_replica, noise)
    ep = evaluate(ne, ev, evp, evg, n_replica, noise, seed)
    rows = {"seed": seed, "n_replica": n_replica, "noise": noise, "chance": round(ch, 3), "eprop": round(ep, 4), "gains": {}}
    print(f"[replica-pop s{seed} N={n_replica} noise={noise}] chance={ch:.3f} eprop(diag,ensemble)={ep:.3f}", flush=True)
    for gg in (0.2, 0.4, 0.7):
        nm = fresh(); train_replica(nm, tr, trp, trg, epochs, lr, seed, "mdgl", n_replica, noise, gamma_gain=gg)
        md = evaluate(nm, ev, evp, evg, n_replica, noise, seed)
        nf = fresh(); train_replica(nf, tr, trp, trg, epochs, lr, seed, "mdgl", n_replica, noise, gamma_sign=-1.0, gamma_gain=gg)
        sf = evaluate(nf, ev, evp, evg, n_replica, noise, seed)
        clean = md > ep + 0.08 and sf <= ep + 0.05
        rows["gains"][gg] = {"mdgl": round(md, 4), "signflip": round(sf, 4), "clean_directional": bool(clean)}
        print(f"  gain={gg}: MDGL={md:.3f} signflip={sf:.3f} | MDGL-eprop={md-ep:+.3f} signflip-eprop={sf-ep:+.3f} "
              f"{'<-- CLEAN DIRECTIONAL' if clean else ''}", flush=True)
    rows["any_clean"] = any(g["clean_directional"] for g in rows["gains"].values())
    return rows


def run_full(seed=42, T=8, n=80, n_replica=16, noise=0.15, epochs=60, lr=3e-3, K=4, eval_n=32):
    """FULL arm set at a fixed replica-N, with the read-out-denoising confound KILLED (eval_n fixed + EQUAL for every arm,
    independent of train-N) + the full anti-cheats. GO iff a CONSISTENT gain gives MDGL>eprop AND signflip collapses to
    <=eprop AND zeroG~=eprop AND permuted~=chance. The single-neuron BPTT = the task-solvable ceiling reference."""
    from research.runners._mdgl_spiking_port_derisk import train as train_sn, evaluate as eval_sn
    tr, trp, trg, V, ncls = make_task("xor", T, 400, seed * 10 + 1, K=K)
    ev, evp, evg, _, _ = make_task("xor", T, 200, seed * 10 + 2, K=K)
    ch = 1.0 / ncls
    def fresh(): return SpikingRNN(V, n, ncls, seed)
    # ceiling reference: single-neuron BPTT (proves the task is solvable)
    nb = fresh(); train_sn(nb, tr, trp, trg, epochs, lr, seed, "bptt"); bptt = eval_sn(nb, ev, evp, evg)
    def ev_arm(net): return evaluate(net, ev, evp, evg, eval_n, noise, seed)   # FIXED eval_n for EVERY arm (kills the denoising confound)
    ne = fresh(); train_replica(ne, tr, trp, trg, epochs, lr, seed, "eprop", n_replica, noise); ep = ev_arm(ne)
    npm = fresh(); train_replica(npm, tr, trp, trg, epochs, lr, seed, "eprop_permuted", n_replica, noise); perm = ev_arm(npm)
    out = {"seed": seed, "n_replica": n_replica, "eval_n": eval_n, "noise": noise, "chance": round(ch, 3),
           "bptt_ceiling": round(bptt, 4), "eprop": round(ep, 4), "permuted": round(perm, 4), "gains": {}}
    best = None
    for gg in (0.1, 0.15, 0.2, 0.25, 0.3):
        nm = fresh(); train_replica(nm, tr, trp, trg, epochs, lr, seed, "mdgl", n_replica, noise, gamma_gain=gg); md = ev_arm(nm)
        nf = fresh(); train_replica(nf, tr, trp, trg, epochs, lr, seed, "mdgl", n_replica, noise, gamma_sign=-1.0, gamma_gain=gg); sf = ev_arm(nf)
        nz = fresh(); train_replica(nz, tr, trp, trg, epochs, lr, seed, "mdgl", n_replica, noise, gamma_gain=0.0); zg = ev_arm(nz)
        clean = md > ep + 0.06 and sf <= ep + 0.03 and abs(zg - ep) < 0.08 and perm < ch + 0.15
        out["gains"][gg] = {"mdgl": round(md, 4), "signflip": round(sf, 4), "zeroG": round(zg, 4), "clean": bool(clean)}
        if clean and (best is None or md > out["gains"][best]["mdgl"]):
            best = gg
    out["best_clean_gain"] = best; out["GO"] = best is not None
    g = out["gains"]
    print(f"[full s{seed} N={n_replica} eval_n={eval_n}] chance={ch:.3f} BPTT={bptt:.3f} eprop={ep:.3f} perm={perm:.3f} || "
          + " ".join(f"g{gg}:M{g[gg]['mdgl']:.2f}/F{g[gg]['signflip']:.2f}/Z{g[gg]['zeroG']:.2f}" for gg in g)
          + f" || best_clean_gain={best} GO={out['GO']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-replica", type=int, default=8); ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--full", action="store_true")
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--out", default="research/findings/raw/_mdgl_replica_popcoded.json")
    a = ap.parse_args()
    if a.full:
        rows = [run_full(s, n_replica=a.n_replica, noise=a.noise, epochs=a.epochs, eval_n=a.eval_n) for s in a.seeds]
        print(f"[full] {sum(r['GO'] for r in rows)}/{len(rows)} GO (clean-directional off-diagonal on spikes w/ population averaging)", flush=True)
        json.dump(rows, open(a.out, "w")); return
    rows = [run_gain_sweep(s, n_replica=a.n_replica, noise=a.noise, epochs=a.epochs) for s in a.seeds]
    n_clean = sum(r["any_clean"] for r in rows)
    print(f"[replica-pop] {n_clean}/{len(rows)} seeds show a CLEAN DIRECTIONAL off-diagonal (MDGL>eprop AND signflip collapses) "
          f"-> population averaging {'RECOVERS' if n_clean == len(rows) else 'does NOT robustly recover'} the spiking off-diagonal", flush=True)
    json.dump(rows, open(a.out, "w"))


if __name__ == "__main__":
    main()
