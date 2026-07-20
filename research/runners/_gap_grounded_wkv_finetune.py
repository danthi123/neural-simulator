"""De-risk 2 (residual-B) — the FORMAT FINE-TUNE: make the spiking WKV *answer* instead of *narrate*.

The EMERGE-57 lever, ported to the WKV: continue-train the fluent TinyStories WKV on GROUNDED
COPY frames ("the A v3 P <ans> the A v3 P <eos>") INTERLEAVED with raw TinyStories (anti-forgetting),
so it learns to COPY the fact from the prompt into a focused restatement (RA-faithful = follows the
PROMPT fact, not a memorized one -> generalizes to any fact incl. learned/Wikidata). Two format
markers (`<ans>`, `<eos>`) are appended to the V=4000 word vocab (transparent to the on-bridge path:
just 2 more emb/head rows).

The torch module REPLICATES the known-correct `WKVFaculty` numpy forward EXACTLY (asserted by a
logit-match check before any training -- verify-first, the silent-failure discipline). Loads the
`wkv_ssmU_v4000_d256_big_seed42.npz` checkpoint, continue-trains at low LR, saves an npz in the same
format (so `WKVFaculty(ckpt=<out>)` loads it). Held-out curriculum + RA-faithful facts are EXCLUDED
from training so the fine-tune generalizes rather than memorizes. NO `sim/` edit.

GO gate (mirror P2/EMERGE-57): focused-grounded fluent (VERIFY-verified) high; RA-faithful (follows
prompt patient); anti-forgetting (TinyStories held-out ppl not regressed); the raw-ceiling 0.17 lifts.
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

CUR_PATH = "research/findings/raw/_grounded_lang_curriculum_p2.json"


class TorchWKV(nn.Module):
    """Bit-match of WKVFaculty: ssm-uniform-decay dual-nonneg leaky state + receptance-gated read-out."""

    def __init__(self, V, D):
        super().__init__()
        self.V, self.D = V, D
        self.emb = nn.Embedding(V, D)
        self.ln_w = nn.Parameter(torch.ones(D)); self.ln_b = nn.Parameter(torch.zeros(D))
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wr = nn.Linear(D, D, bias=False)
        self.Wo_sp = nn.Linear(2 * D, D, bias=False)
        self.head = nn.Linear(D, V, bias=True)
        self.w = nn.Parameter(torch.zeros(1))
        self.register_buffer("u", torch.zeros(D))  # unused (ssm), preserved for npz format compat

    def _ln(self, x):
        m = x.mean(-1, keepdim=True); s = x.std(-1, unbiased=False, keepdim=True) + 1e-5
        return (x - m) / s * self.ln_w + self.ln_b

    def forward(self, x):
        # x: [B,T] ids
        h = self._ln(self.emb(x))                      # [B,T,D]
        v = self.Wv(h)                                 # [B,T,D]
        decay = torch.exp(-F.softplus(self.w))         # scalar
        B, T, D = h.shape
        ap = torch.zeros(B, D, device=x.device); an = torch.zeros(B, D, device=x.device)
        states = []
        for t in range(T):
            ap = decay * ap + torch.relu(v[:, t])
            an = decay * an + torch.relu(-v[:, t])
            states.append(torch.cat([ap, an], -1))
        state = torch.stack(states, 1)                 # [B,T,2D]
        rh = torch.sigmoid(self.Wr(h))                 # [B,T,D]
        return self.head(rh * self.Wo_sp(state))       # [B,T,V]

    def load_npz(self, z, extra_tokens):
        Vold = int(z["V"]); D = int(z["d_model"])
        with torch.no_grad():
            self.emb.weight[:Vold].copy_(torch.tensor(np.asarray(z["emb.weight"], np.float32)))
            self.ln_w.copy_(torch.tensor(np.asarray(z["ln.weight"], np.float32)))
            self.ln_b.copy_(torch.tensor(np.asarray(z["ln.bias"], np.float32)))
            self.Wv.weight.copy_(torch.tensor(np.asarray(z["Wv.weight"], np.float32)))
            self.Wr.weight.copy_(torch.tensor(np.asarray(z["Wr.weight"], np.float32)))
            self.Wo_sp.weight.copy_(torch.tensor(np.asarray(z["Wo_sp.weight"], np.float32)))
            self.head.weight[:Vold].copy_(torch.tensor(np.asarray(z["head.weight"], np.float32)))
            self.head.bias[:Vold].copy_(torch.tensor(np.asarray(z["head.bias"], np.float32)))
            self.w.copy_(torch.tensor(np.asarray(z["w"], np.float32).reshape(1)))
            self.u.copy_(torch.tensor(np.asarray(z["u"], np.float32)))
            # new format-marker rows: small random emb; head init to a low bias so they don't dominate
            g = torch.Generator().manual_seed(1234)
            for j in range(extra_tokens):
                self.emb.weight[Vold + j].copy_(0.02 * torch.randn(D, generator=g))
                self.head.weight[Vold + j].copy_(0.02 * torch.randn(D, generator=g))
                self.head.bias[Vold + j].copy_(torch.tensor(-2.0))

    def save_npz(self, words, path):
        z = {
            "V": np.int64(self.V), "d_model": np.int64(self.D),
            "words": np.array(words, dtype=object),
            "w": self.w.detach().cpu().numpy().astype(np.float32),
            "u": self.u.detach().cpu().numpy().astype(np.float32),
            "emb.weight": self.emb.weight.detach().cpu().numpy().astype(np.float32),
            "ln.weight": self.ln_w.detach().cpu().numpy().astype(np.float32),
            "ln.bias": self.ln_b.detach().cpu().numpy().astype(np.float32),
            "Wk.weight": np.zeros((self.D, self.D), np.float32),  # unused placeholder (ssm)
            "Wv.weight": self.Wv.weight.detach().cpu().numpy().astype(np.float32),
            "Wr.weight": self.Wr.weight.detach().cpu().numpy().astype(np.float32),
            "Wo.weight": np.zeros((self.D, self.D), np.float32),  # unused placeholder
            "Wo_sp.weight": self.Wo_sp.weight.detach().cpu().numpy().astype(np.float32),
            "head.weight": self.head.weight.detach().cpu().numpy().astype(np.float32),
            "head.bias": self.head.bias.detach().cpu().numpy().astype(np.float32),
        }
        np.savez(path, **z)


import re as _re
_WORD = _re.compile(r"[a-z']+")


def load_tiny_sentences(path, n, w2i, min_len=5, max_len=24):
    """Split the corpus into sentences on [.!?] (the corpus is one long line), tokenize to lowercase words."""
    txt = open(path, "r", errors="ignore").read().lower()
    out = []
    for raw in _re.split(r"[.!?]", txt):
        toks = _WORD.findall(raw)
        if not (min_len <= len(toks) <= max_len):
            continue
        out.append([w2i.get(t, w2i["<unk>"]) for t in toks])
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=BIG_CKPT)
    ap.add_argument("--out", default="bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n-tiny", type=int, default=20000, help="TinyStories sentences for anti-forgetting")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grounded-frac", type=float, default=0.5, help="fraction of batches that are grounded copy frames")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--freeze-input", action="store_true",
                    help="RESERVOIR/emergence-path test: freeze Wv (input->state map) + decay (recurrence) at the "
                         "pretrained values; train ONLY the read-out (Wr/Wo_sp/head) + emb -> is the grounded copy a "
                         "SHALLOW-readout adaptation over a FIXED cortex (no deep BPTT of the recurrence)?")
    ap.add_argument("--freeze-cortex", action="store_true",
                    help="STRICTER reservoir test: freeze the ENTIRE input encoding (original emb + Wv + decay); train "
                         "ONLY the read-out (Wr/Wo_sp/head) + the 2 marker emb rows -> is the grounded copy PURE "
                         "shallow-readout over a TOTALLY fixed cortex?")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.ckpt, allow_pickle=True)
    words = [str(w) for w in z["words"]]
    Vold = len(words)
    # append format markers (transparent to the on-bridge path: 2 more emb/head rows)
    markers = ["<ans>", "<eos>"]
    words_new = words + markers
    V = len(words_new); D = int(z["d_model"])
    w2i = {w: i for i, w in enumerate(words_new)}
    ANS, EOS = w2i["<ans>"], w2i["<eos>"]

    net = TorchWKV(V, D).to(dev)
    net.load_npz(z, extra_tokens=len(markers))
    if args.freeze_input or args.freeze_cortex:
        net.Wv.weight.requires_grad_(False)   # input->state map FIXED (reservoir)
        net.w.requires_grad_(False)            # recurrence decay FIXED
        print("[freeze-input] RESERVOIR test: Wv (input->state) + decay FROZEN; training the read-out (Wr/Wo_sp/head)"
              + (" + ALL emb" if not args.freeze_cortex else " + ONLY the 2 marker emb rows (original emb FROZEN)"))
    if args.freeze_cortex:
        # per-row emb gradient mask: zero the gradient on the ORIGINAL Vold rows, leaving the 2 marker rows trainable
        _emask = torch.zeros(V, 1, device=dev); _emask[Vold:] = 1.0
        net.emb.weight.register_hook(lambda g, m=_emask: g * m)

    # --- verify-first: the torch forward matches the numpy WKVFaculty logits (before training) ---
    from research.runners._wkv_faculty import WKVFaculty
    fac0 = WKVFaculty(ckpt=args.ckpt)
    test_words = ["the", "dog", "eats", "meat"]
    tids = fac0.ids(test_words)
    with torch.no_grad():
        lt = net(torch.tensor([tids], device=dev))[0, -1, :Vold].cpu().numpy()
    ap = np.zeros(D); an = np.zeros(D)
    for t in tids:
        ap, an = fac0._charge(t, ap, an)
    ln = fac0._logits(tids[-1], ap, an)
    match = float(np.corrcoef(lt, ln)[0, 1]); maxdiff = float(np.abs(lt - ln).max())
    print(f"[verify-first] torch-vs-numpy forward: corr={match:.6f} maxdiff={maxdiff:.4f} "
          f"(argmax torch={words_new[int(lt.argmax())]!r} numpy={words[int(ln.argmax())]!r})")
    assert match > 0.999 and maxdiff < 0.05, f"torch forward does NOT match numpy WKVFaculty (corr={match}, maxdiff={maxdiff})"

    # --- build the grounded copy-frame vocab (in-WKV-vocab SVO combos), held-out the test facts ---
    cur = json.load(open(CUR_PATH))
    heldout = {tuple(f) for f in cur["facts"]}  # exclude the 22 curriculum facts from training (test generalization)
    v3 = {b: s for (b, s, _p) in VERBS}
    subs = [s for s in SUBJECTS if s in w2i and w2i[s] != w2i["<unk>"]]
    objs = [o for o in OBJECTS if o in w2i and w2i[o] != w2i["<unk>"]]
    verbs = [(b, s) for (b, s, _p) in VERBS if b in w2i and s in w2i and w2i[s] != w2i["<unk>"]]
    print(f"[frames] in-vocab: {len(subs)} subjects, {len(verbs)} verbs, {len(objs)} objects "
          f"-> {len(subs)*len(verbs)*len(objs)} copy combos (heldout {len(heldout)})")

    def sample_grounded_batch(bs):
        seqs = []
        while len(seqs) < bs:
            a = random.choice(subs); (vb, vs) = random.choice(verbs); p = random.choice(objs)
            if (a, vb, p) in heldout:
                continue
            # frame: "the a vs p <ans> the a vs p <eos>"; loss only on the answer span (after <ans>)
            fact = [w2i["the"], w2i[a], w2i[vs], w2i[p]]
            seq = fact + [ANS] + fact + [EOS]
            # loss mask: 1 on positions predicting the answer tokens (from <ans> onward)
            mask = [0] * len(fact) + [0] + [1] * (len(fact)) + [1]   # predict: the,a,vs,p,<eos> after <ans>
            seqs.append((seq, mask))
        return seqs

    tiny = load_tiny_sentences(args.corpus, args.n_tiny, w2i)
    tiny_eval = tiny[:1000]; tiny_train = tiny[1000:]
    print(f"[tiny] {len(tiny_train)} train, {len(tiny_eval)} eval sentences")

    def tiny_batch(bs):
        seqs = []
        for _ in range(bs):
            ids = random.choice(tiny_train)
            seqs.append((ids, [1] * len(ids)))
        return seqs

    def pad_batch(seqs):
        L = max(len(s) for s, _ in seqs)
        X = torch.full((len(seqs), L), EOS, dtype=torch.long)
        M = torch.zeros(len(seqs), L)
        for i, (s, m) in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s); M[i, :len(m)] = torch.tensor(m, dtype=torch.float)
        return X.to(dev), M.to(dev)

    @torch.no_grad()
    def tiny_ppl():
        net.eval(); tot = 0.0; ntok = 0
        for i in range(0, min(500, len(tiny_eval)), args.batch):
            chunk = [(s, [1] * len(s)) for s in tiny_eval[i:i + args.batch]]
            X, M = pad_batch(chunk)
            lg = net(X)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
            m = M[:, 1:].reshape(-1)
            tot += float((loss * m).sum()); ntok += float(m.sum())
        net.train()
        return float(np.exp(tot / max(1, ntok)))

    ppl0 = tiny_ppl()
    print(f"[anti-forget] TinyStories held-out ppl BEFORE fine-tune: {ppl0:.3f}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        seqs = sample_grounded_batch(args.batch) if random.random() < args.grounded_frac else tiny_batch(args.batch)
        X, M = pad_batch(seqs)
        lg = net(X)
        loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
        m = M[:, 1:].reshape(-1)
        loss = (loss * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            print(f"[step {step}/{args.steps}] loss={float(loss):.4f} ppl_tiny={tiny_ppl():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ppl1 = tiny_ppl()
    net.save_npz(words_new, args.out)
    print(f"\n[DONE] TinyStories ppl {ppl0:.3f} -> {ppl1:.3f} (anti-forget: "
          f"{'OK' if ppl1 < ppl0 * 1.3 else 'REGRESSED'}); saved {args.out}")
    json.dump({"ppl_before": ppl0, "ppl_after": ppl1, "steps": args.steps, "lr": args.lr,
               "grounded_frac": args.grounded_frac, "markers": markers, "out": args.out},
              open(args.out.replace(".npz", "_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
