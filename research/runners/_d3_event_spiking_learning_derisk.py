"""D3 EVENT COMPOSITION — the DEEPEST fully-spiking rung: the two-slot event transition LEARNED THROUGH a SPIKING LIF
hidden, from END-STATE-only (weak) supervision. The event arc so far: the re-discretization is on spikes (the FS-WTA
port) and the transition δ is LEARNED from weak supervision (through a RATE tanh hidden). THIS composes them for the
FACTORED event — the transition's hidden is a SPIKING LIF pool (rate-coded, T steps, surrogate gradient), trained from
END-STATE-only supervision via the Dyna-style DETACHED-rollout CURRICULUM (roll both slots with the LIF-argmax, supervise
ONLY the final (a,p)). So the event transition FORWARD is spiking THROUGHOUT the weak-supervision learning (the master-
directive "fully spiking including the learning", not just at execution) -> the factored (a,p) update is learned-on-a-
spiking-forward from weak supervision, and length-generalizes to a genuinely-DEEP task.

(Honest scope, same as the single-slot rung: the surrogate-gradient BACKWARD is still host BPTT -- a biologically-
plausible LOCAL rule is the separate deep wall, EMERGE-6..8's 5x-confirmed dead-end; this rung makes the FORWARD spiking
during the two-slot learning.)

ANTI-CHEATS: (a) STATE (weak-sup, spiking-forward, two-slot) held-out-DEEPER >> the LAST-2-OBJECTS shallow reader (deep,
not a 2-token lookup); (b) >> SHUFFLE-label (genuine learning); (c) multi-seed dev+blind. Reuse-by-import (`make_event_task`
+ `last2_objects_floor` + `lif_rate` from the single-slot rung); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_event_spiking_learning_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_event_composition_derisk import make_event_task, last2_objects_floor
from research.runners._d3_spiking_weak_learning_derisk import lif_rate, _softmax


def train_event_spiking_weak(task, seed=42, n_hid=192, T=16, epochs=90, lr=0.05, batch=256,
                             shuffle_labels=False, curriculum=True):
    """Learn the factored (a,p) event UPDATE through a SPIKING LIF hidden, from END-STATE-only supervision + curriculum.
    Two K-way heads (agent Wa / patient Wp) supervised on the FINAL (a,p) only; the intermediates are the model's own
    LIF-argmax rollout. The hidden is `lif_rate` (rate-coded LIF, surrogate grad)."""
    K = task["K"]; ident = task["ident"]; n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 33)
    n_in = 2 * n_hid + n_pool
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)          # fixed attractor prototypes
    W1 = (rng.randn(n_hid, n_in) * np.sqrt(2.0 / n_in)).astype(np.float32)
    Wa = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); ba = np.zeros(K, np.float32)
    Wp = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(K, np.float32)
    eyeK = np.eye(K, dtype=np.float32)

    X, Ya, Yp, L, SEQ, STA, STP = task["train"]; N = len(L)
    fa_all = STA[np.arange(N), L - 1]; fp_all = STP[np.arange(N), L - 1]     # final (a,p) endpoints ONLY
    if shuffle_labels:
        perm = rng.permutation(N); fa_all = fa_all[perm]; fp_all = fp_all[perm]

    def feat(a, p, x):
        return np.concatenate([emb[a], emb[p], x], axis=1)        # [B, n_in]

    def roll_hard(Xb, Lb):
        """Argmax DETACHED rollout of BOTH slots THROUGH THE SPIKING LIF; return (prev_a, prev_p entering the final step,
        x at the final step)."""
        B = len(Lb); Lmax = int(Lb.max())
        a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
        pa = np.full(B, ident, np.int64); pp = np.full(B, ident, np.int64); xf = np.zeros((B, n_pool), np.float32)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            pa[last] = a[last]; pp[last] = p[last]; xf[last] = Xb[last, t]
            rate, _ = lif_rate(feat(a, p, Xb[:, t]), W1, T)
            na = (rate @ Wa.T + ba).argmax(1); npp = (rate @ Wp.T + bp).argmax(1)
            a = np.where(active, na, a); p = np.where(active, npp, p)
        return pa, pp, xf

    def final_ap(Xb, Lb):
        B = len(Lb); Lmax = int(Lb.max())
        a = np.full(B, ident, np.int64); p = np.full(B, ident, np.int64)
        fa = np.full(B, ident, np.int64); fp = np.full(B, ident, np.int64)
        for t in range(Lmax):
            active = (Lb > t); last = (Lb == (t + 1))
            rate, _ = lif_rate(feat(a, p, Xb[:, t]), W1, T)
            na = (rate @ Wa.T + ba).argmax(1); npp = (rate @ Wp.T + bp).argmax(1)
            a = np.where(active, na, a); p = np.where(active, npp, p)
            fa = np.where(last, a, fa); fp = np.where(last, p, fp)
        return fa, fp

    phases = sorted(set(int(x) for x in L)) if curriculum else [int(L.max())]
    for phase_len in phases:
        sub = np.where(L <= phase_len)[0]
        for ep in range(epochs):
            pa, pp, xf = roll_hard(X[sub], L[sub])                # roll with the CURRENT spiking transition (detached)
            order = rng.permutation(len(sub))
            for i in range(0, len(sub), batch):
                b = order[i:i + batch]; gi = sub[b]; B = len(b)
                F = feat(pa[b], pp[b], xf[b])
                rate, sg = lif_rate(F, W1, T, want_grad=True)     # SPIKING forward (+ surrogate for the backward)
                sa = _softmax(rate @ Wa.T + ba); da = (sa - eyeK[fa_all[gi]]) / B
                sp = _softmax(rate @ Wp.T + bp); dp = (sp - eyeK[fp_all[gi]]) / B
                Wa -= lr * (da.T @ rate); ba -= lr * da.sum(0); Wp -= lr * (dp.T @ rate); bp -= lr * dp.sum(0)
                drate = (da @ Wa) + (dp @ Wp)
                dpre = drate * sg                                 # surrogate through the LIF threshold
                W1 -= lr * (dpre.T @ F)

    def eval_split(split):
        Xe, Yae, Ype, Le, SEQe, STAe, STPe = task[split]
        fa, fp = final_ap(Xe, Le)
        ta = STAe[np.arange(len(Le)), Le - 1]; tp = STPe[np.arange(len(Le)), Le - 1]
        return float(((fa == ta) & (fp == tp)).mean())

    return {"deeper": eval_split("test_deeper"), "same": eval_split("test_same"),
            "weights": {"emb": emb, "W1": W1, "Wa": Wa, "ba": ba, "Wp": Wp, "bp": bp, "T": T}}   # for the spiking one-loop


def run_seed(seed, K, n_hid, T, epochs):
    task = make_event_task(seed, K=K, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    st = train_event_spiking_weak(task, seed=seed, n_hid=n_hid, T=T, epochs=epochs)
    sh = train_event_spiking_weak(task, seed=seed, n_hid=n_hid, T=T, epochs=epochs, shuffle_labels=True)
    return {"seed": seed, "K": K, "SPKFWD_event_deeper": round(st["deeper"], 3), "SPKFWD_same": round(st["same"], 3),
            "SHUFFLE_deeper": round(sh["deeper"], 3), "last2_objects_floor": round(last2_objects_floor(task), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    print(f"[D3 EVENT SPIKING-FORWARD LEARNING] K={a.K} | the factored (a,p) event transition LEARNED from END-STATE-only supervision THROUGH a spiking LIF hidden (surrogate grad)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.T, a.epochs); rows.append(r)
        print(f"  [seed {s}] SPIKING-FORWARD event DEEPER={r['SPKFWD_event_deeper']} (same={r['SPKFWD_same']}) || "
              f"LAST-2-OBJ(shallow)={r['last2_objects_floor']} || SHUFFLE(mem-floor)={r['SHUFFLE_deeper']}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        def _m(k): return float(np.mean([r[k] for r in rows]))
        sd, l2, sh = _m("SPKFWD_event_deeper"), _m("last2_objects_floor"), _m("SHUFFLE_deeper")
        go = (sd > 0.7) and (sd - l2 > 0.2) and (sd - sh > 0.2)
        print(f"\n  AGGREGATE (K={a.K}): SPIKING-FORWARD event DEEPER={sd:.3f} | LAST-2-OBJ(shallow)={l2:.3f} | SHUFFLE(mem-floor)={sh:.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the factored (agent, patient) event transition is LEARNED from END-STATE-only supervision THROUGH a spiking LIF hidden (surrogate grad) and length-generalizes to a genuinely-DEEP task (DEEPER '+format(sd,'.2f')+' >> the LAST-2-OBJECTS shallow reader '+format(l2,'.2f')+' >> shuffle-label '+format(sh,'.2f')+') -> the event-composition LEARNING has a SPIKING FORWARD throughout (the master-directive fully-spiking-incl-learning); with the FS-WTA re-discretization port, BOTH the transition-learning-forward AND the execution of the running MEANING are on spikes' if go else 'the spiking-forward two-slot learning did not clearly beat the controls (tune epochs/T/n_hid)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
