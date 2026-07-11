"""CEILING de-risk (GPU, PyTorch) — bounds the whole reservoir-LM long-range arc. This session established that NO reservoir
substrate mechanism (fixed / e-prop / longer-tau / ALIF-state / content-addressable retrieval in every form) beats a
bigram/cache at long-range on WikiText. The open question that verdict leaves: is that a SUBSTRATE limit, or is open-text
long-range signal simply THIN at this small scale/vocab? A real LEARNED attention (a small TinyGPT) is the honest ceiling:
if even a transformer's advantage over the bigram does NOT grow with context depth on THIS data, long-range is thin here
(the reservoir negatives are partly a scale/data artifact). If it DOES grow, the long-range signal is real and the
transformer is the target the biological frontier must reach. Apples-to-apples: the SAME top-V word vocab + the SAME
bigram baseline + the SAME context-depth idea (here = position within the transformer's contiguous block).

NOT a biological model — a reference CEILING (like the FORCE/RLS upper-bound controls the research gates suggested).
"""
import os, argparse, json, math, time, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

OUT = Path("research/findings/raw/_wikitext_transformer_ceiling.json")


def load_stream(path, max_tokens):
    txt = open(path, encoding="utf-8", errors="ignore").read().lower()
    words = re.findall(r"[a-z]+", txt)          # contiguous word stream, document order preserved (NO shuffle, NO sentence filter)
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
    ap.add_argument("--corpus", type=str, default="data/corpus/wikitext.txt")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=3_000_000)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--json", type=str, default=str(OUT))
    args = ap.parse_args()

    import torch, torch.nn.functional as F
    from sim.tiny_transformer import TinyGPT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    words = load_stream(args.corpus, args.max_tokens)
    cut = int(0.9 * len(words))
    stoi = build_vocab(words[:cut], args.vocab)
    def enc(ws): return np.array([stoi.get(w, 0) for w in ws], dtype=np.int64)
    tr = enc(words[:cut]); ev = enc(words[cut:])
    V = len(stoi)
    print(f"[ceiling] stream={len(words)} V={V} block={args.block} dev={dev} params~"
          f"{args.d_model*args.d_model*4*args.n_layer/1e6:.1f}M", flush=True)

    # bigram baseline (add-1) on train
    P = np.ones((V, V));
    for a, b in zip(tr[:-1], tr[1:]): P[a, b] += 1.0
    P /= P.sum(1, keepdims=True)

    B = args.block
    def batches(ids, bs):
        n = (len(ids) - 1) // B
        starts = np.arange(n) * B
        while True:
            idx = np.random.permutation(n)[:bs]
            s = starts[idx]
            x = np.stack([ids[i:i + B] for i in s]); y = np.stack([ids[i + 1:i + B + 1] for i in s])
            yield torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    model = TinyGPT(V, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, block_size=B).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = batches(tr, args.batch)
    t0 = time.time(); model.train()
    for step in range(args.steps):
        x, y = next(gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  step {step} train-CE {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # EVAL: CE by WITHIN-BLOCK POSITION (context depth) for transformer vs bigram, on held-out contiguous blocks
    model.eval()
    n_ev = (len(ev) - 1) // B
    tce = np.zeros(B); bce = np.zeros(B); cnt = np.zeros(B)
    with torch.no_grad():
        for i in range(0, n_ev):
            s = i * B
            x = torch.from_numpy(ev[s:s + B][None, :]).to(dev)
            y = ev[s + 1:s + B + 1]
            lp = torch.log_softmax(model(x)[0], -1).cpu().numpy()   # (B, V)
            for p in range(B):
                tgt = y[p]
                tce[p] += -lp[p, tgt]
                bce[p] += -math.log(max(P[ev[s + p], tgt], 1e-12))
                cnt[p] += 1
    # bucket by within-block position (context depth)
    BK = [(1, 1), (2, 2), (3, 3), (4, 8), (9, 16), (17, B)]
    rows = {}
    for lo, hi in BK:
        m = slice(lo, hi + 1)  # position index p means p tokens of context seen
        t = tce[m].sum() / max(cnt[m].sum(), 1); b = bce[m].sum() / max(cnt[m].sum(), 1)
        rows[f"{lo}-{hi}"] = {"transformer_ce": round(float(t), 3), "bigram_ce": round(float(b), 3),
                              "margin": round(float(b - t), 3)}   # margin>0 => transformer beats bigram
    print("[ceiling] CE by within-block context depth (margin = bigram - transformer; +=transformer better):", flush=True)
    for k, v in rows.items():
        print(f"    ctx {k:>5}: transformer {v['transformer_ce']:.3f}  bigram {v['bigram_ce']:.3f}  margin {v['margin']:+.3f}", flush=True)
    shallow = rows["1-1"]["margin"]; deep = rows[f"17-{B}"]["margin"]
    print(f"[ceiling] LONG-RANGE SIGNAL = does the transformer's margin GROW with context? shallow(ctx1) {shallow:+.3f} -> "
          f"deep(ctx17+) {deep:+.3f}  (grows => real long-range; flat => thin at this scale)", flush=True)
    out = {"runner": "_wikitext_transformer_ceiling", "V": V, "block": B, "dev": dev, "by_ctx_depth": rows,
           "shallow_margin": shallow, "deep_margin": deep, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True); Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.json}", flush=True)


if __name__ == "__main__":
    main()
