"""The emergence-path CLOSE (Rung C'): biologize the WKV cortex's learning — a FIXED reservoir + a read-out
trained by a LOCAL rule (feedback-alignment + delta), NO BPTT-through-time, NO weight transport.

Informed by the research gate + its adversarial verification + Rung A/B:
- the WKV recurrence is a DIAGONAL scalar-decay leaky integrator (no recurrent weight matrix) -> e-prop is EXACT, but
  the verification flagged that `emb` (~44% of params) is ALSO on the through-time path WHEN TRAINED;
- Rung B: a RANDOM `Wv` reservoir + a trained read-out reaches TinyStories ppl ~25.6 (<= the BPTT 28.1) -> `Wv` need
  NOT be learned; Rung A/freeze-cortex: the grounded copy is a pure shallow read-out over a fixed cortex.
=> the cleanest FULLY-biological close FREEZES the entire cortex + encoding (emb + Wv + decay, pretrained or random)
   -> ZERO through-time credit anywhere -> and trains ONLY the per-timestep read-out (Wr, Wo_sp, head) by a LOCAL rule.

The read-out is trained by FEEDBACK ALIGNMENT (Lillicrap 2016 / Akrout KP): the backward pass routes the output error
through FIXED RANDOM feedback matrices instead of the weight transposes (no weight transport); each layer's weight
update is local (output error x layer input). The recurrent state is DETACHED (frozen reservoir) so there is NO
BPTT-through-time. Adam is only the step rule (the biological claim is about CREDIT ASSIGNMENT: transport-free + local
+ no BPTT). Anti-cheats: shuffle-feedback collapses; the Adam-BPTT read-out is the reference ceiling.

NO `sim/` edit. GPU (torch). Verify-first: the FA forward == the standard read-out forward; the FA training decreases
the loss (valid descent).
"""
from __future__ import annotations
import argparse, json, os, sys, time, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from research.runners._wkv_faculty import BIG_CKPT  # noqa: E402
from research.runners._fluidconv_phase2_ra_finetune import VERBS, SUBJECTS, OBJECTS  # noqa: E402
from research.runners._gap_grounded_wkv_finetune import load_tiny_sentences  # noqa: E402

CUR_PATH = "research/findings/raw/_grounded_lang_curriculum_p2.json"


class LinearFA(torch.autograd.Function):
    """Linear layer whose BACKWARD routes the error through a feedback matrix B instead of W^T -> NO weight transport.
    FA (Lillicrap 2016): B fixed random. KP (Kolen-Pollack / Akrout 2019): B LEARNS the same LOCAL update as W
    (grad_B = grad_W), so the fixed-transport is replaced by a co-evolving transport-free feedback that aligns to W.
    The weight gradient is LOCAL (output error (x) layer input) in both cases."""
    @staticmethod
    def forward(ctx, x, W, B, kp):
        ctx.save_for_backward(x, B)
        ctx.kp = bool(kp)
        return x @ W.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, B = ctx.saved_tensors
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        grad_x = (g2 @ B).reshape(x.shape)          # B (fixed FA / learned KP) instead of W^T -> no transport
        grad_W = g2.t() @ x2                         # local: output error x layer input
        grad_B = grad_W if ctx.kp else None         # KP: feedback gets the SAME local update -> aligns to W
        return grad_x, grad_W, grad_B, None


def fa(x, W, B, kp=False):
    return LinearFA.apply(x, W, B, kp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG_CKPT)
    ap.add_argument("--out", default="bridges/wkv_ckpt/wkv_ssmU_v4000_d256_local_readout.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-tiny", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grounded-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--random-input", action="store_true", help="freeze Wv at RANDOM (true reservoir) instead of pretrained")
    ap.add_argument("--credit", choices=["fa", "kp", "bptt"], default="kp",
                    help="fa = fixed-random feedback alignment; kp = Kolen-Pollack learned feedback (transport-free, "
                         "aligns to W -> the biological close); bptt = weight-transport reference ceiling")
    ap.add_argument("--shuffle-feedback", action="store_true", help="anti-cheat: shuffle the FA feedback matrices (must collapse)")
    ap.add_argument("--readout-from-scratch", action="store_true",
                    help="init the read-out (Wr/Wo_sp/head) from SMALL RANDOM instead of the pretrained (BPTT) values -> "
                         "FA aligns the forward weights from scratch (the clean emergence test; FA degrades a pretrained solution)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(args.seed)
    z = np.load(args.ckpt, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    Vold = len(words)
    words_new = words + ["<ans>", "<eos>"]
    V = len(words_new); D = int(z["d_model"])
    w2i = {w: i for i, w in enumerate(words_new)}
    ANS, EOS = w2i["<ans>"], w2i["<eos>"]

    # ---- FIXED reservoir + encoding (frozen; the credit-assignment substrate) ----
    def T(name): return torch.tensor(np.asarray(z[name], np.float32), device=dev)
    emb = torch.zeros(V, D, device=dev)
    emb[:Vold] = T("emb.weight")
    emb[Vold:] = 0.02 * torch.randn(2, D, generator=g, device=dev)   # marker embeddings (fixed random)
    ln_w = T("ln.weight"); ln_b = T("ln.bias")
    if args.random_input:
        Wv = torch.empty(D, D, device=dev); nn.init.xavier_uniform_(Wv, generator=g)
    else:
        Wv = T("Wv.weight")
    decay = float(np.exp(-np.log1p(np.exp(float(np.asarray(z["w"]).ravel()[0])))))
    for t in (emb, ln_w, ln_b, Wv):
        t.requires_grad_(False)

    def _ln(x):
        return (x - x.mean(-1, keepdim=True)) / (x.std(-1, unbiased=False, keepdim=True) + 1e-5) * ln_w + ln_b

    @torch.no_grad()
    def reservoir_states(x):
        """FROZEN reservoir: [B,T] ids -> (h [B,T,D], state [B,T,2D]). Detached => NO BPTT-through-time."""
        h = _ln(emb[x])                                  # [B,T,D]
        v = h @ Wv.t()                                   # [B,T,D]
        B_, Tt, _ = h.shape
        ap = torch.zeros(B_, D, device=dev); an = torch.zeros(B_, D, device=dev)
        states = []
        for t in range(Tt):
            ap = decay * ap + torch.relu(v[:, t]); an = decay * an + torch.relu(-v[:, t])
            states.append(torch.cat([ap, an], -1))
        return h, torch.stack(states, 1)                 # [B,T,D], [B,T,2D]

    # ---- TRAINED read-out (the ONLY learnable) ----
    if args.readout_from_scratch:
        Wr = torch.randn(D, D, generator=g, device=dev) * (1.0 / D ** 0.5)
        Wo_sp = torch.randn(D, 2 * D, generator=g, device=dev) * (1.0 / (2 * D) ** 0.5)
        head_w = torch.randn(V, D, generator=g, device=dev) * (1.0 / D ** 0.5)
        head_b = torch.zeros(V, device=dev); head_b[Vold:] = -2.0
    else:
        Wr = (T("Wr.weight").clone()); Wo_sp = (T("Wo_sp.weight").clone())
        head_w = torch.zeros(V, D, device=dev); head_w[:Vold] = T("head.weight")
        head_b = torch.zeros(V, device=dev); head_b[:Vold] = T("head.bias"); head_b[Vold:] = -2.0
    for t in (Wr, Wo_sp, head_w, head_b):
        t.requires_grad_(True)
    params = [Wr, Wo_sp, head_w, head_b]
    # FEEDBACK matrices (same shape as the weights they replace in the backward) -- no transport. KP: trainable (align).
    _kp = (args.credit == "kp")
    Br = torch.randn(D, D, generator=g, device=dev) / (D ** 0.5)
    Bo = torch.randn(D, 2 * D, generator=g, device=dev) / ((2 * D) ** 0.5)
    Bh = torch.randn(V, D, generator=g, device=dev) / (D ** 0.5)
    if args.shuffle_feedback:                            # anti-cheat: destroy the feedback structure
        Bh = Bh[torch.randperm(V, generator=g, device=dev)]
    if _kp:
        for b in (Br, Bo, Bh):
            b.requires_grad_(True)
        params += [Br, Bo, Bh]                           # KP: feedback learns the same local update -> aligns to W

    def readout(h, state):
        """logits = head @ (sigmoid(Wr@h) * (Wo_sp@state)). fa/kp -> LinearFA (transport-free); bptt -> plain."""
        if args.credit in ("fa", "kp"):
            rh = torch.sigmoid(fa(h, Wr, Br, _kp))
            s = fa(state, Wo_sp, Bo, _kp)
            return fa(rh * s, head_w, Bh, _kp) + head_b
        rh = torch.sigmoid(h @ Wr.t()); s = state @ Wo_sp.t()
        return (rh * s) @ head_w.t() + head_b

    # ---- verify-first: FA forward == standard forward (only backward differs) ----
    xv = torch.tensor([[w2i.get(t, w2i["<unk>"]) for t in ["the", "dog", "eats", "meat"]]], device=dev)
    hv, sv = reservoir_states(xv)
    with torch.no_grad():
        lg_fa = fa(torch.sigmoid(fa(hv, Wr, Br)) * fa(sv, Wo_sp, Bo), head_w, Bh) + head_b
        lg_std = (torch.sigmoid(hv @ Wr.t()) * (sv @ Wo_sp.t())) @ head_w.t() + head_b
    fwd_max = float((lg_fa - lg_std).abs().max())
    print(f"[verify-first] FA forward == standard forward: maxdiff={fwd_max:.2e} (backward differs only)")
    assert fwd_max < 1e-3, "FA forward must equal the standard read-out forward"

    # ---- data (same as the fine-tune: grounded copy frames held-out of the 22 curriculum + TinyStories) ----
    cur = json.load(open(CUR_PATH))
    heldout = {tuple(f) for f in cur["facts"]}
    subs = [s for s in SUBJECTS if s in w2i and w2i[s] != w2i["<unk>"]]
    objs = [o for o in OBJECTS if o in w2i and w2i[o] != w2i["<unk>"]]
    verbs = [(b, s) for (b, s, _p) in VERBS if b in w2i and s in w2i and w2i[s] != w2i["<unk>"]]

    def grounded_batch(bs):
        seqs = []
        while len(seqs) < bs:
            a = random.choice(subs); (_vb, vs) = random.choice(verbs); p = random.choice(objs)
            if (a, _vb, p) in heldout:
                continue
            fact = [w2i["the"], w2i[a], w2i[vs], w2i[p]]
            seqs.append((fact + [ANS] + fact + [EOS], [0] * len(fact) + [0] + [1] * len(fact) + [1]))
        return seqs

    tiny = load_tiny_sentences(args.corpus, args.n_tiny, w2i)
    tiny_eval = tiny[:1000]; tiny_train = tiny[1000:]

    def tiny_batch(bs):
        return [(random.choice(tiny_train), None) for _ in range(bs)]

    def pad(seqs):
        L = max(len(s) for s, _ in seqs)
        X = torch.full((len(seqs), L), EOS, dtype=torch.long)
        M = torch.zeros(len(seqs), L)
        for i, (s, m) in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s); M[i, :len(s)] = torch.tensor(m if m is not None else [1] * len(s), dtype=torch.float)
        return X.to(dev), M.to(dev)

    @torch.no_grad()
    def tiny_ppl():
        tot = 0.0; ntok = 0
        for i in range(0, min(500, len(tiny_eval)), args.batch):
            chunk = [(s, None) for s in tiny_eval[i:i + args.batch]]
            X, Mm = pad(chunk); h, st = reservoir_states(X); lg = readout(h, st)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
            m = Mm[:, 1:].reshape(-1); tot += float((loss * m).sum()); ntok += float(m.sum())
        return float(np.exp(tot / max(1, ntok)))

    opt = torch.optim.Adam(params, lr=args.lr)   # step rule only; credit assignment is FA (transport-free) + local
    ppl0 = tiny_ppl()
    print(f"[credit={args.credit}{' SHUFFLED-FB' if args.shuffle_feedback else ''}"
          f"{' random-Wv' if args.random_input else ''}] TinyStories ppl BEFORE: {ppl0:.2f}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        seqs = grounded_batch(args.batch) if random.random() < args.grounded_frac else tiny_batch(args.batch)
        X, Mm = pad(seqs)
        h, st = reservoir_states(X)                       # frozen reservoir (detached => no BPTT)
        lg = readout(h, st)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
        m = Mm[:, 1:].reshape(-1); loss = (loss * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            print(f"[step {step}/{args.steps}] loss={float(loss):.4f} ppl_tiny={tiny_ppl():.3f} ({time.time()-t0:.0f}s)", flush=True)
    ppl1 = tiny_ppl()

    # save an npz in the WKVFaculty format so the ceiling probe can eval the grounded copy
    def _np(t): return t.detach().cpu().numpy().astype(np.float32)
    out = {"V": np.int64(V), "d_model": np.int64(D), "words": np.array(words_new, dtype=object),
           "w": np.asarray(z["w"], np.float32), "u": np.asarray(z["u"], np.float32),
           "emb.weight": _np(emb), "ln.weight": _np(ln_w), "ln.bias": _np(ln_b),
           "Wk.weight": np.zeros((D, D), np.float32), "Wv.weight": _np(Wv), "Wo.weight": np.zeros((D, D), np.float32),
           "Wr.weight": _np(Wr), "Wo_sp.weight": _np(Wo_sp), "head.weight": _np(head_w), "head.bias": _np(head_b)}
    np.savez(args.out, **out)
    print(f"[DONE credit={args.credit}] ppl {ppl0:.2f} -> {ppl1:.2f}; saved {args.out}")
    json.dump({"credit": args.credit, "random_input": args.random_input, "shuffle_feedback": args.shuffle_feedback,
               "ppl_before": ppl0, "ppl_after": ppl1, "steps": args.steps, "lr": args.lr, "out": args.out},
              open(args.out.replace(".npz", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
