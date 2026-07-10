"""D3 -> surpass the RELATIONAL-learning residual: how much supervision does RELATIONAL composition (discourse-referent
tracking) actually need? The reference-tracking finding showed the discrete-attractor LEARNS the relational reference-delta
from PER-STEP supervision (teacher-forced, deeper-track 0.88) but NOT from END-STATE-only (0.29) -- unlike the group
LOOKUP DFA, which RANK-1 learned from end-state-only. THIS characterizes the gap: train the discrete-attractor with the
running state supervised at a STRIDE (every k steps) -- stride 1 = per-step (teacher-forced), stride>=L = end-only -- and
read the held-out-DEEPER holder-track as a function of supervision density. This maps EXACTLY how sparse the intermediate
signal can be for the relational delta (the biology: an agent PERCEIVES the current referent intermittently, not every
token nor only at the end).

ANTI-CHEATS: held-out-DEEPER track; the stride-1 (per-step) upper-bound + the end-only (stride>=Lmax) lower-bound bracket
the curve; multi-seed. Reuse-by-import (`make_reference_tracking_task` + the group harness codes); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_sparse_supervision_derisk --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_reference_tracking_derisk import make_reference_tracking_task


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def train_strided(task, seed=42, n_hid=192, epochs=80, lr=0.1, batch=256, stride=1):
    """Discrete-attractor transition trained on teacher-forced triples SUBSAMPLED to every `stride`-th step of each
    sequence (the final step is ALWAYS supervised). stride=1 -> per-step (dense); large stride -> only sparse anchors +
    the endpoint. Eval = autoregressive rollout with re-discretization (argmax) -> held-out-DEEPER holder-track."""
    K = task["K"]; ident = task["ident"]; n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 9)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)
    Wr = (rng.randn(n_hid, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Ws = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bs = np.zeros(K, dtype=np.float32)

    def collect(split):
        Xe, ye, Le, _, Se = task[split]; PREV, XIN, NXT = [], [], []
        for n in range(len(Le)):
            prev = ident; L = int(Le[n])
            for t in range(L):
                # supervise this step iff it is a stride-anchor OR the final step (the endpoint is always available)
                if (t % stride == 0) or (t == L - 1):
                    PREV.append(prev); XIN.append(Xe[n, t]); NXT.append(int(Se[n, t]))
                prev = int(Se[n, t])                               # teacher-forced advance (the true running holder)
        return np.asarray(PREV), np.asarray(XIN, np.float32), np.asarray(NXT)

    P, X, N = collect("train"); M = len(N)
    for ep in range(epochs):
        order = rng.permutation(M)
        for i in range(0, M, batch):
            bi = order[i:i + batch]
            h = np.tanh(emb[P[bi]] @ Wr.T + X[bi] @ Wi.T); sm = _softmax(h @ Ws.T + bs)
            d = sm.copy(); d[np.arange(len(bi)), N[bi]] -= 1.0; d /= len(bi)
            dh = (d @ Ws) * (1.0 - h ** 2)
            Ws -= lr * (d.T @ h); bs -= lr * d.sum(0); Wr -= lr * (dh.T @ emb[P[bi]]); Wi -= lr * (dh.T @ X[bi])

    def deeper_track():
        Xe, ye, Le, _, Se = task["test_deeper"]; B = len(Le); Lmax = int(Le.max())
        cur = np.full(B, ident, dtype=np.int64); final = np.full(B, ident, dtype=np.int64)
        for t in range(Lmax):
            active = (Le > t)
            nxt = (np.tanh(emb[cur] @ Wr.T + Xe[:, t] @ Wi.T) @ Ws.T + bs).argmax(1)
            cur = np.where(active, nxt, cur); final = np.where(Le == (t + 1), cur, final)
        return float((final == Se[np.arange(B), Le - 1]).mean()), float(M)

    tr, ntri = deeper_track()
    return {"deeper_track": tr, "n_triples": ntri}


def run_seed(seed, K, n_hid, epochs, strides):
    task = make_reference_tracking_task(seed, K=K, n_pool=64, n_per_len=2500, train_lens=(1, 2, 3), test_lens=(6, 7, 8))
    out = {"seed": seed}
    for st in strides:
        r = train_strided(task, seed=seed, n_hid=n_hid, epochs=epochs, stride=st)
        out[f"stride{st}"] = round(r["deeper_track"], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-hid", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--strides", default="1,2,3,99")            # 1=per-step, 99>=Lmax -> end-only
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    strides = [int(x) for x in a.strides.replace(",", " ").split()]
    print(f"[D3 SPARSE-SUPERVISION] reference-tracking K={a.K} | holder-track DEEPER vs supervision stride (1=per-step ... {max(strides)}=end-only)", flush=True)
    rows = []
    for s in seeds:
        r = run_seed(s, a.K, a.n_hid, a.epochs, strides)
        rows.append(r)
        print(f"  [seed {s}] " + " | ".join(f"stride{st}={r[f'stride{st}']}" for st in strides), flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        agg = {st: float(np.mean([r[f"stride{st}"] for r in rows])) for st in strides}
        print(f"\n  AGGREGATE (K={a.K}, chance {1.0/a.K:.3f}): " + " | ".join(f"stride{st}={agg[st]:.3f}" for st in strides), flush=True)
        s1 = agg[min(strides)]; sEnd = agg[max(strides)]
        # a middle stride (e.g. 2 or 3) that stays HIGH (near the per-step upper-bound) => relational composition needs
        # only SPARSE intermediate anchors, not dense per-step -> a genuine surpass of the end-only residual.
        mids = [st for st in strides if st != min(strides) and st != max(strides)]
        best_mid = max((agg[st] for st in mids), default=0.0)
        surpass = (best_mid > 0.6) and (best_mid - sEnd > 0.2)
        print(f"  READ: per-step(stride1)={s1:.3f} (upper bound) | end-only(stride{max(strides)})={sEnd:.3f} (lower bound) | best sparse-anchor mid-stride={best_mid:.3f}", flush=True)
        print(f"  VERDICT: {'SPARSE-SURPASS' if surpass else 'characterization'} -- {'the relational reference-delta is learnable from SPARSE intermediate anchors (mid-stride '+format(best_mid,'.2f')+' >> end-only '+format(sEnd,'.2f')+') -> relational composition needs only intermittent state signal (an agent PERCEIVING the referent now and then), NOT dense per-step -> the residual is surpassed to sparse supervision' if surpass else 'the deeper-track degrades smoothly with supervision sparsity (per-step '+format(s1,'.2f')+' -> end-only '+format(sEnd,'.2f')+'); relational composition needs fairly dense state signal (the honest characterization: unlike the LOOKUP DFA, the RELATIONAL delta needs frequent intermediate supervision / perception)'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
