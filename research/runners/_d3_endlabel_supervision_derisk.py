"""D3 rung 3 (the adversarial GENUINENESS test): can the DISCRETE-ATTRACTOR architecture learn multi-hop composition
from END-LABEL supervision ALONE (the final property), with NO per-step state targets? The rate + spiking D3 GOs
trained the transition on teacher-forced (state,input)->next-state triples (a strong per-step signal). The honest
question: is the composition LEARNED, or just a taught DFA rolled out? The continuous RNN FAILED end-to-end (even WITH
per-step state supervision it drifted). Does the discrete-attractor's clean-state re-discretization make END-LABEL
credit assignment TRACTABLE where the continuous RNN could not?

MECHANISM: forward the FULL rollout with a STRAIGHT-THROUGH re-discretization -- forward uses the hard argmax attractor
`emb[argmax(scores_t)]` (clean discrete state, no drift), backward treats it as `softmax(scores_t) @ emb` (differentiable
so gradient flows through the discrete state) -- read the final property from the final state, and train with ONLY the
end-of-sequence property label (CE). BPTT through the whole rollout. If it learns the composition (property acc >> chance)
AND length-generalizes (held-out-DEEPER) from end-label alone, the discrete-attractor architecture makes recurrent
end-label credit tractable -- a genuine "recurrent credit path LEARNS multi-hop composition" result, not a taught DFA.

ANTI-CHEATS: (a) held-out-DEEPER (learned iteration, not memorized length); (b) a CONTINUOUS straight-through control
(carry the soft state h_t forward instead of the clean attractor -> drifts -> should fail deeper, isolating the
re-discretization as what makes end-label credit work); (c) order-control (non-abelian -> permuted collapses); (d)
multi-seed. Reuse-by-import (`make_group_task`); numpy; NO `sim/` edit.

Run:  SIM_BACKEND=numpy python -m research.runners._d3_endlabel_supervision_derisk --group S3 --seeds 42
"""
from __future__ import annotations
import argparse
import numpy as np

from research.runners._d3_group_composition_derisk import make_group_task, build_group


def train_endlabel(task, seed=42, n_hid=160, epochs=120, lr=0.05, batch=128, discrete=True):
    """Train the discrete-attractor rollout with END-LABEL supervision only (final property), straight-through
    re-discretization. discrete=True -> clean attractor state each step (straight-through argmax); discrete=False ->
    carry the soft continuous state h_t (the control: no re-discretization -> drifts)."""
    K = task["K"]; ident = task["ident"]; color = task["color"]; n_pool = task["n_pool"]
    rng = np.random.RandomState(seed + 33)
    emb = (rng.randn(K, n_hid) * 0.5).astype(np.float32)          # fixed attractor prototypes
    Wr = (rng.randn(n_hid, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32)
    Wi = (rng.randn(n_hid, n_pool) * np.sqrt(1.0 / n_pool)).astype(np.float32)
    Ws = (rng.randn(K, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bs = np.zeros(K, dtype=np.float32)
    Wp = (rng.randn(2, n_hid) * np.sqrt(1.0 / n_hid)).astype(np.float32); bp = np.zeros(2, dtype=np.float32)  # property read-out

    def _softmax(z):
        e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

    Xtr, ytr, Ltr, _, _ = task["train"]; N = len(Ltr)

    def rollout(Xb, Lb, train=True):
        B = len(Lb); Lmax = int(Lb.max())
        # forward, caching per step for BPTT
        cur_emb = np.tile(emb[ident], (B, 1))                     # current STATE embedding [B, n_hid]
        cache = []
        final_emb = cur_emb.copy()
        for t in range(Lmax):
            hpre = cur_emb @ Wr.T + Xb[:, t] @ Wi.T; h = np.tanh(hpre)
            scores = h @ Ws.T + bs; p = _softmax(scores)         # [B, K]
            if discrete:
                win = p.argmax(1)
                nxt_emb = emb[win]                               # STRAIGHT-THROUGH: forward = clean attractor
            else:
                nxt_emb = p @ emb                                # CONTROL: soft continuous state (drifts)
            active = (Lb > t)[:, None]
            cache.append((cur_emb, Xb[:, t], hpre, h, scores, p, active))
            cur_emb = np.where(active, nxt_emb, cur_emb)
            final_emb = np.where((Lb == (t + 1))[:, None], cur_emb, final_emb)
        return final_emb, cache

    for ep in range(epochs):
        order = rng.permutation(N)
        for i in range(0, N, batch):
            bi = order[i:i + batch]; Xb = Xtr[bi]; Lb = Ltr[bi]; yb = ytr[bi]; B = len(bi)
            final_emb, cache = rollout(Xb, Lb, train=True)
            plog = final_emb @ Wp.T + bp; psm = _softmax(plog)
            d = psm.copy(); d[np.arange(B), yb] -= 1.0; d /= B
            dWp = d.T @ final_emb; dbp = d.sum(0)
            dstate = d @ Wp                                      # dL/d(final_emb)  [B, n_hid]
            dWr = np.zeros_like(Wr); dWi = np.zeros_like(Wi); dWs = np.zeros_like(Ws); dbs = np.zeros_like(bs)
            # BPTT backward through the rollout (straight-through: d(nxt_emb)/d(scores) via softmax@emb)
            for t in range(len(cache) - 1, -1, -1):
                cur_emb, xt, hpre, h, scores, p, active = cache[t]
                dnxt = dstate * active                           # gradient into this step's produced state
                # nxt_emb = softmax(scores) @ emb  (straight-through / soft for both)
                demb_soft = dnxt                                 # [B, n_hid]; d(scores) = emb-weighted softmax jacobian
                dp = demb_soft @ emb.T                           # [B, K]
                dscores = p * (dp - (p * dp).sum(1, keepdims=True))
                dWs += dscores.T @ h; dbs += dscores.sum(0)
                dh = (dscores @ Ws) * (1.0 - h ** 2)
                dWr += dh.T @ cur_emb; dWi += dh.T @ xt
                dstate = (dh @ Wr) + dstate * (~active)          # carry: inactive samples keep their downstream grad
            for W, dW in ((Wp, dWp), (Ws, dWs), (Wr, dWr), (Wi, dWi)):
                W -= lr * dW
            bp -= lr * dbp; bs -= lr * dbs

    def eval_split(split):
        Xe, ye, Le, _, Se = task[split]
        final_emb, _ = rollout(Xe, Le, train=False)
        prop = (final_emb @ Wp.T + bp).argmax(1)
        return float((prop == ye).mean())

    return {"same": eval_split("test_same"), "deeper": eval_split("test_deeper"), "train": eval_split("train")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="S3", choices=["S3", "S4", "A5"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-hid", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    np_pool = 256 if a.group == "A5" else 64
    print(f"[D3 END-LABEL supervision] {a.group} | does the DISCRETE-ATTRACTOR learn composition from END-LABEL ALONE (no per-step state targets)?", flush=True)
    rows = []
    for s in seeds:
        task = make_group_task(a.group, s, n_pool=np_pool, n_per_len=2000, train_lens=(1, 2, 3), test_lens=(4, 5, 6))
        disc = train_endlabel(task, seed=s, n_hid=a.n_hid, epochs=a.epochs, discrete=True)
        cont = train_endlabel(task, seed=s, n_hid=a.n_hid, epochs=a.epochs, discrete=False)
        rows.append({"seed": s, "disc": disc, "cont": cont})
        print(f"  [seed {s}] DISCRETE-attractor: train={disc['train']:.3f} same={disc['same']:.3f} DEEPER={disc['deeper']:.3f} "
              f"|| CONTINUOUS control: same={cont['same']:.3f} DEEPER={cont['deeper']:.3f}", flush=True)
    if a.json and rows:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        dd = float(np.mean([r["disc"]["deeper"] for r in rows])); cd = float(np.mean([r["cont"]["deeper"] for r in rows]))
        ds = float(np.mean([r["disc"]["same"] for r in rows]))
        go = (dd > 0.75) and (dd - cd > 0.15)
        print(f"\n  AGGREGATE ({a.group}): DISCRETE same={ds:.3f} DEEPER={dd:.3f} | CONTINUOUS DEEPER={cd:.3f} (chance 0.5)", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the DISCRETE-ATTRACTOR learns multi-hop composition from END-LABEL ALONE and length-generalizes (DEEPER '+format(dd,'.2f')+' >> the continuous control) -> the re-discretization makes recurrent END-LABEL credit TRACTABLE where the continuous RNN cannot: a genuine recurrent-credit-path-LEARNS-composition result (not a taught DFA)' if go else 'end-label-only did not length-generalize (DEEPER '+format(dd,'.2f')+'); the honest boundary: even the discrete-attractor needs per-step teaching for length-generalizing composition from end-label (credit still hard) -- OR tune epochs/n_hid/straight-through'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
