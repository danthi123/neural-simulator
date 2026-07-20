"""Deeper spiking-purity (owner steer 2026-07-20) — cheap-first de-risk toward biologizing the WKV cortex's FULL
learning on the one shared spiking substrate.

END GOAL (owner): fully close all gaps = fully-spiking, one brain, single shared substrate. The grounded-render TASK
learning is now on-substrate by a pure exact delta rule (~0.94); the cortex's PRETRAINING (its TinyStories fluency)
is still off-bridge BPTT. This de-risk asks the read-out-expressiveness question CHEAPLY (off-bridge, fast) BEFORE the
slow on-bridge build: does a SHALLOW exact-delta read-out (the mechanism that closed the grounded task) learn
full-vocab FLUENCY over a FIXED reservoir, or does fluency need the multi-layer read-out?

Setup (mirrors `_gap_grounded_wkv_local_readout` but a SINGLE-linear read-out): the WKV cortex (emb/Wv/decay) is the
FIXED reservoir (detached => no BPTT-through-time). A single linear read-out `logits = state @ Wsl^T + h @ Wh^T`
(state = the leaky reservoir state, h = the current token) is trained on TinyStories next-token by the EXACT DELTA
rule (the softmax gradient is LOCAL + exact for a single output layer — no FA, no transport, no BPTT). Metric: held-out
TinyStories ppl, vs the BPTT ceiling (~29.5) and the multi-layer FA/KP (~35, the biological-learning close). NO `sim/`
edit; off-bridge torch.
"""
from __future__ import annotations
import argparse, os, random, time
import numpy as np
import torch
import torch.nn.functional as F

from research.runners._wkv_faculty import BIG_CKPT
from research.runners._gap_grounded_wkv_finetune import load_tiny_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG_CKPT)
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-tiny", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--random-input", action="store_true", help="freeze Wv at RANDOM (a true reservoir), not the pretrained map")
    ap.add_argument("--no-token", action="store_true", help="ablate the current-token term (state-only read-out)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(args.seed)
    z = np.load(args.ckpt, allow_pickle=True)
    words = [str(w) for w in z["words"]]; V = len(words); D = int(z["d_model"]); w2i = {w: i for i, w in enumerate(words)}

    def Tt(n): return torch.tensor(np.asarray(z[n], np.float32), device=dev)
    emb = Tt("emb.weight"); ln_w = Tt("ln.weight"); ln_b = Tt("ln.bias")
    if args.random_input:
        Wv = torch.empty(D, D, device=dev); torch.nn.init.xavier_uniform_(Wv, generator=g)
    else:
        Wv = Tt("Wv.weight")
    decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
    for t in (emb, ln_w, ln_b, Wv):
        t.requires_grad_(False)

    def _ln(x): return (x - x.mean(-1, keepdim=True)) / (x.std(-1, unbiased=False, keepdim=True) + 1e-5) * ln_w + ln_b

    @torch.no_grad()
    def reservoir(x):                                    # [B,T] -> h [B,T,D], state [B,T,2D]  (FROZEN, detached)
        h = _ln(emb[x]); v = h @ Wv.t(); B_, T_, _ = h.shape
        ap = torch.zeros(B_, D, device=dev); an = torch.zeros(B_, D, device=dev); st = []
        for t in range(T_):
            ap = decay * ap + torch.relu(v[:, t]); an = decay * an + torch.relu(-v[:, t]); st.append(torch.cat([ap, an], -1))
        return h, torch.stack(st, 1)

    # the SINGLE-linear read-out (the only learnable): logits = state @ Wsl^T + h @ Wh^T
    Wsl = (torch.randn(V, 2 * D, generator=g, device=dev) / (2 * D) ** 0.5).requires_grad_(True)
    Wh = (torch.randn(V, D, generator=g, device=dev) / D ** 0.5).requires_grad_(True)
    params = [Wsl] + ([] if args.no_token else [Wh])

    def readout(h, state):
        lg = state @ Wsl.t()
        if not args.no_token:
            lg = lg + h @ Wh.t()
        return lg

    tiny = load_tiny_sentences(args.corpus, args.n_tiny, w2i)
    tiny_eval = tiny[:1000]; tiny_train = tiny[1000:]

    def pad(seqs):
        L = max(len(s) for s in seqs); X = torch.full((len(seqs), L), w2i.get("<unk>", V - 1), dtype=torch.long)
        M = torch.zeros(len(seqs), L)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s); M[i, :len(s)] = 1.0
        return X.to(dev), M.to(dev)

    @torch.no_grad()
    def ppl():
        tot = 0.0; nt = 0
        for i in range(0, min(500, len(tiny_eval)), args.batch):
            X, M = pad(tiny_eval[i:i + args.batch]); h, st = reservoir(X); lg = readout(h, st)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
            m = M[:, 1:].reshape(-1); tot += float((loss * m).sum()); nt += float(m.sum())
        return float(np.exp(tot / max(1, nt)))

    opt = torch.optim.SGD(params, lr=args.lr)   # SGD = the pure delta rule (the exact softmax gradient is local for a single output layer)
    print(f"[shallow-fluency] V={V} D={D} decay={decay:.3f}{' random-Wv' if args.random_input else ''}"
          f"{' NO-token' if args.no_token else ''}; ppl BEFORE {ppl():.2f} (BPTT ceiling ~29.5)")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        X, M = pad([random.choice(tiny_train) for _ in range(args.batch)])
        h, st = reservoir(X); lg = readout(h, st)
        loss = (F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none") * M[:, 1:].reshape(-1)).sum() / M[:, 1:].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            print(f"[step {step}/{args.steps}] loss={float(loss):.3f} ppl={ppl():.3f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[RESULT] shallow exact-delta read-out fluency ppl = {ppl():.2f} (BPTT ceiling ~29.5; multi-layer FA/KP ~35) "
          f"-- does a SHALLOW read-out do fluency, or is the multi-layer needed?")


if __name__ == "__main__":
    main()
