"""CEILING-FIRST rung for the CORRECTED long-range frontier (GPU, PyTorch) — a RECURRENT-architecture upper bound.

The transformer ceiling established the target: at genuine scale (TinyStories 23.7M / WikiText-103 ~60M tokens) a content
model captures growing-with-depth long-range (margin over a bigram rises to +1.5 nats at deep context). The corrected
biological frontier is: can BIOLOGICAL DEEP CREDIT train a RECURRENT spiking substrate (no attention) to capture that?
Before investing in the biological-credit build, the honest ceiling-first question (this session's saved lesson) is:
does the RECURRENT architecture CLASS even reach the transformer's growing-with-depth long-range at this scale, with a
full-backprop UPPER BOUND? A recurrent net (LSTM/GRU) is the closest full-gradient analogue of the recurrent spiking
substrate: NO attention, a fixed-size hidden state carried across the stream (like a reservoir/RNN cortex). If a
full-backprop LSTM's margin GROWS with depth toward the transformer's +1.5, the recurrent CLASS can hold long-range and
the frontier is "biological credit for a multi-layer recurrent net." If it PLATEAUS well short (attention beats
recurrence at this scale), the frontier is a content-addressable STORE (the D3/theta-gamma WM-buffer direction), not
deeper recurrence — a decisive fork, cheaply resolved BEFORE the months-scale biological build.

Apples-to-apples with `_wikitext_transformer_ceiling.py`: SAME corpus, SAME top-V word vocab, SAME add-1 bigram baseline,
SAME CE-by-context-depth metric. Only the model class differs (LSTM/GRU vs TinyGPT). NOT a biological model — a reference
ceiling (like FORCE/RLS upper-bounds the research gates suggest). Stateful eval: the hidden state is carried across the
whole contiguous held-out stream so "context depth" = how many tokens of running history the recurrent state has seen.
"""
import os, argparse, json, math, time, re
from pathlib import Path
from collections import Counter
import numpy as np

OUT = Path("research/findings/raw/_recurrent_lm_ceiling.json")


def load_stream(path, max_tokens):
    txt = open(path, encoding="utf-8", errors="ignore").read().lower()
    words = re.findall(r"[a-z]+", txt)      # contiguous word stream, document order preserved
    return words[:max_tokens]


def build_vocab(words, V):
    freq = Counter(words).most_common(V - 1)
    stoi = {"<unk>": 0}
    for w, _ in freq:
        stoi[w] = len(stoi)
    return stoi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", type=str, default="data/corpus/tinystories.txt")
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=24_000_000)
    ap.add_argument("--block", type=int, default=128)          # BPTT truncation window (train) + eval bucket span
    ap.add_argument("--cell", type=str, default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stateful", action="store_true",
                    help="carry the recurrent hidden state ACROSS blocks (detached TBPTT in train; carried across the "
                         "held-out stream in eval) — the recurrent architecture's TRUE long-range ceiling (unbounded "
                         "context a transformer's block cannot see). Disambiguates 'recurrence can't hold long-range' "
                         "(block-matched plateaus) from 'recurrence needs cross-block state' (stateful grows).")
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    words = load_stream(args.corpus, args.max_tokens)
    cut = int(0.9 * len(words))
    stoi = build_vocab(words[:cut], args.vocab)
    def enc(ws): return np.array([stoi.get(w, 0) for w in ws], dtype=np.int64)
    tr = enc(words[:cut]); ev = enc(words[cut:])
    V = len(stoi)

    class RecurrentLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, args.d_model)
            Cell = nn.LSTM if args.cell == "lstm" else nn.GRU
            self.rnn = Cell(args.d_model, args.d_model, num_layers=args.n_layer, batch_first=True)
            self.ln = nn.LayerNorm(args.d_model)
            self.head = nn.Linear(args.d_model, V)
        def forward(self, x, h=None):
            e = self.emb(x)
            y, h = self.rnn(e, h)
            return self.head(self.ln(y)), h

    model = RecurrentLM().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[recceil] stream={len(words)} V={V} block={args.block} cell={args.cell} L={args.n_layer} "
          f"dev={dev} params~{n_params/1e6:.1f}M", flush=True)

    # bigram baseline (add-1) on train
    P = np.ones((V, V))
    for a, b in zip(tr[:-1], tr[1:]): P[a, b] += 1.0
    P /= P.sum(1, keepdims=True)

    B = args.block

    def _detach(h):
        if h is None: return None
        return tuple(t.detach() for t in h) if isinstance(h, tuple) else h.detach()

    def batches(ids, bs):                        # block-matched: random blocks, fresh state each block
        n = (len(ids) - 1) // B
        starts = np.arange(n) * B
        while True:
            idx = np.random.permutation(n)[:bs]
            s = starts[idx]
            x = np.stack([ids[i:i + B] for i in s]); y = np.stack([ids[i + 1:i + B + 1] for i in s])
            yield torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    def stateful_batches(ids, bs):               # TBPTT: bs contiguous lanes, consecutive blocks, detached carried state
        lane = len(ids) // bs
        base = np.stack([ids[k * lane:(k + 1) * lane] for k in range(bs)])   # (bs, lane)
        nb = (lane - 1) // B
        while True:
            for j in range(nb):
                s = j * B
                x = base[:, s:s + B]; y = base[:, s + 1:s + B + 1]
                yield (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), j == 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time(); model.train()
    if args.stateful:
        gen = stateful_batches(tr, args.batch); h = None
        for step in range(args.steps):
            x, y, is_start = next(gen)
            if is_start: h = None                # new epoch pass over the lanes -> fresh state
            logits, h = model(x, _detach(h))     # carry detached state across blocks = TBPTT (unbounded effective context)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 1000 == 0:
                print(f"  step {step} train-CE {loss.item():.3f} ({time.time()-t0:.0f}s) [stateful]", flush=True)
    else:
        gen = batches(tr, args.batch)
        for step in range(args.steps):
            x, y = next(gen)
            logits, _ = model(x)                 # h=None each block: truncated BPTT to the block window (fair vs transformer)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 1000 == 0:
                print(f"  step {step} train-CE {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # EVAL: CE by context depth (within-block position), STATEFUL across the held-out stream — the recurrent state carries
    # running history, so position p within a block still means "p+carried tokens of context"; to match the transformer's
    # within-block context-depth metric we RESET state at block starts and read depth = position in block.
    model.eval()
    n_ev = (len(ev) - 1) // B
    tce = np.zeros(B); bce = np.zeros(B); cnt = np.zeros(B)
    heval = None                                  # carried across blocks only when --stateful
    with torch.no_grad():
        for i in range(0, n_ev):
            s = i * B
            x = torch.from_numpy(ev[s:s + B][None, :]).to(dev)
            y = ev[s + 1:s + B + 1]
            if args.stateful:
                logits, heval = model(x, heval)   # carry state across the held-out stream: within-block position also carries prior-block history
            else:
                logits, _ = model(x)              # h=None: fresh state at block start, matches transformer's causal within-block context
            lp = torch.log_softmax(logits[0], -1).cpu().numpy()
            for p in range(B):
                tgt = y[p]
                tce[p] += -lp[p, tgt]
                bce[p] += -math.log(max(P[ev[s + p], tgt], 1e-12))
                cnt[p] += 1
    BK = [(1, 1), (2, 2), (3, 3), (4, 8), (9, 16), (17, B)]
    rows = {}
    for lo, hi in BK:
        m = slice(lo, hi + 1)
        t = tce[m].sum() / max(cnt[m].sum(), 1); b = bce[m].sum() / max(cnt[m].sum(), 1)
        rows[f"{lo}-{hi}"] = {"recurrent_ce": round(float(t), 3), "bigram_ce": round(float(b), 3),
                              "margin": round(float(b - t), 3)}
    print(f"[recceil] CE by within-block context depth (margin = bigram - {args.cell}; +={args.cell} better):", flush=True)
    for k, v in rows.items():
        print(f"    ctx {k:>6}: {args.cell} {v['recurrent_ce']:.3f}  bigram {v['bigram_ce']:.3f}  margin {v['margin']:+.3f}", flush=True)
    shallow = rows["1-1"]["margin"]; deep = rows[f"17-{B}"]["margin"]
    print(f"[recceil] LONG-RANGE = does the {args.cell} margin GROW with context? shallow(ctx1) {shallow:+.3f} -> "
          f"deep(ctx17+) {deep:+.3f}  (grows toward the transformer's +1.5 => recurrence holds long-range; "
          f"plateaus short => attention beats recurrence, frontier is a content-addressable store)", flush=True)
    out = {"runner": "_recurrent_lm_ceiling", "cell": args.cell, "stateful": bool(args.stateful), "V": V, "block": B,
           "n_layer": args.n_layer, "params_m": round(n_params / 1e6, 2), "dev": dev, "corpus": args.corpus, "by_ctx_depth": rows,
           "shallow_margin": shallow, "deep_margin": deep, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json}\nRECCEIL_DONE", flush=True)


if __name__ == "__main__":
    main()
