"""Comprehension cue-lexicon conversion — does the ANIMACY cue that bounds the D4/D6/D3/other-repair/surprise
comprehension organs to a TOY 2-noun table EMERGE, for OPEN vocabulary, from REAL-corpus co-occurrence?

WHY (PI-ledger): every comprehension organ declares the SAME host scaffold — "VOCAB CEILING: the cue lexicon
(ANIMACY / VERB_SELECTS) is the toy 2-noun transitive scope" (comprehension_production_organ.py:41). A word not in
the 19-noun table is OUT OF SCOPE, so the substrate cannot judge real open-vocab sentences. gap#3-A1 (2026-07-18)
already proved animacy is corpus-DERIVABLE and even spiking-realised it, but ONLY (a) for the referent-BIAS, not
comprehension, and (b) on a CLOSED synthetic corpus generated FROM the ground-truth table, evaluated on the SAME
vocabulary. The genuinely-OPEN question — can a LEARNED cue assign animacy to HELD-OUT words it was never given a
label for, from REAL text (TinyStories), well enough to lift the comprehension vocab ceiling — was never run.

MECHANISM (same class as the affect DR-2 organ: label-propagation over the brain's learned co-occurrence graph,
6-seed held-out r~0.81): build a PPMI word-word co-occurrence graph over the top-V content words of a REAL corpus;
seed a SMALL animacy label set (K animate + K inanimate obvious words); label-spread; read each HELD-OUT word's
propagated sign. The held-out words are NEVER given a label to the learner (used only as eval ground truth).

HONEST CONTROLS (must beat, else it is a hand rule in a spiking costume):
  * SHUFFLED-GRAPH  — permute the off-diagonal PPMI edges (destroy the real co-occurrence structure), re-propagate.
    If held-out accuracy collapses to chance, the signal is CORPUS-DERIVED, not smuggled through the seed set.
  * FREQUENCY-ONLY — predict animacy from raw word frequency. Must be ~chance (animacy is not a frequency artifact).
  * SEED-ONLY sanity — the seed set is disjoint from the held-out set (no label leakage).

GO: held-out animacy accuracy mean(6 seeds) >= 0.75 AND shuffled-graph <= 0.60 AND (learned - shuffled) >= 0.15.
The corpus structure (which word is animate) is NOT injected — it is real distributional English.
"""
import os, re, sys, argparse, collections
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ── Ground-truth animacy labels — obvious, uncontroversial common TinyStories nouns (dog=animate, rock=inanimate).
# NOT chosen to make the method work; they are the standard animate=agent-capable / inanimate=object split. The
# learner sees only a small SEEDED SUBSET each seed; the rest are HELD OUT (never labelled).
GT_ANIMATE = [
    "dog", "cat", "bird", "girl", "boy", "mom", "dad", "rabbit", "bear", "fish",
    "duck", "frog", "cow", "lion", "mouse", "man", "woman", "fox", "bee", "pig",
    "owl", "baby", "friend", "kitten", "puppy", "horse", "monkey", "wolf", "sheep", "goat",
]
GT_INANIM = [
    "ball", "box", "book", "rock", "cup", "tree", "car", "door", "house", "toy",
    "stick", "table", "hat", "star", "flower", "sun", "wall", "road", "leaf", "stone",
    "cake", "key", "chair", "bed", "apple", "cloud", "boat", "spoon", "bell", "coin",
]

_WORD_RE = re.compile(r"[a-z']+")
_STOP = set("""a an the of to and or but if then so as at by for in on with from into onto over under up down out off
is are was were be been being am do does did have has had will would can could should may might must not no yes it its
this that these those he she they them his her their our your my me you we i us him who whom whose what which when where
why how there here all any some each every one two three more most very just now too also than only about after before
said say says get got go went come came make made take took see saw look looked want wanted like liked""".split())


def load_tokens(path, max_chars):
    txt = open(path, encoding="utf-8", errors="ignore").read(max_chars).lower()
    return [t for t in _WORD_RE.findall(txt) if t != "endoftext"]


def build_vocab(tokens, top_v):
    cnt = collections.Counter(t for t in tokens if t not in _STOP and len(t) >= 2)
    vocab = [w for w, _ in cnt.most_common(top_v)]
    freq = collections.Counter(tokens)
    for w in GT_ANIMATE + GT_INANIM:
        if w not in vocab and freq[w] > 0:
            vocab.append(w)
    return vocab, freq


def cooccur_ppmi(tokens, vocab, window=4):
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V, V), dtype=np.float64)
    ids = [idx.get(t, -1) for t in tokens]
    n = len(ids)
    for i in range(n):
        wi = ids[i]
        if wi < 0:
            continue
        lo, hi = max(0, i - window), min(n, i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            wj = ids[j]
            if wj >= 0:
                C[wi, wj] += 1.0
    total = C.sum()
    if total <= 0:
        return C
    row = C.sum(1, keepdims=True)
    col = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row @ col))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0] = 0.0
    return pmi


def label_spread(W, seed_vec, alpha=0.90, n_iter=60):
    """Zhou label-spreading: f <- alpha*S*f + (1-alpha)*y, S = D^-1/2 W D^-1/2. seed_vec: +1/-1 on seeds, 0 else."""
    d = W.sum(1)
    d[d == 0] = 1.0
    dinv = 1.0 / np.sqrt(d)
    S = (W * dinv[:, None]) * dinv[None, :]
    f = seed_vec.copy()
    for _ in range(n_iter):
        f = alpha * (S @ f) + (1 - alpha) * seed_vec
    return f


def shuffle_graph(W, rng):
    """Destroy the co-occurrence structure: symmetric permutation of the off-diagonal edge weights."""
    V = W.shape[0]
    iu = np.triu_indices(V, k=1)
    vals = W[iu].copy()
    rng.shuffle(vals)
    Ws = np.zeros_like(W)
    Ws[iu] = vals
    Ws = Ws + Ws.T
    return Ws


def run_seed(seed, vocab, W, freq, k_seed=8):
    rng = np.random.default_rng(seed)
    idx = {w: i for i, w in enumerate(vocab)}
    anim = [w for w in GT_ANIMATE if w in idx]
    inan = [w for w in GT_INANIM if w in idx]
    rng.shuffle(anim)
    rng.shuffle(inan)
    seed_anim, held_anim = anim[:k_seed], anim[k_seed:]
    seed_inan, held_inan = inan[:k_seed], inan[k_seed:]

    def make_seed_vec():
        y = np.zeros(len(vocab))
        for w in seed_anim:
            y[idx[w]] = +1.0
        for w in seed_inan:
            y[idx[w]] = -1.0
        return y

    def eval_acc(f):
        ok = n = 0
        for w in held_anim:
            ok += int(f[idx[w]] > 0); n += 1
        for w in held_inan:
            ok += int(f[idx[w]] < 0); n += 1
        return ok / n if n else 0.0

    y = make_seed_vec()
    f_learn = label_spread(W, y)
    acc_learn = eval_acc(f_learn)

    Ws = shuffle_graph(W, rng)
    f_shuf = label_spread(Ws, y)
    acc_shuf = eval_acc(f_shuf)

    med = np.median([freq[w] for w in seed_anim + seed_inan])
    ok = n = 0
    for w in held_anim:
        ok += int(freq[w] > med); n += 1
    for w in held_inan:
        ok += int(freq[w] <= med); n += 1
    acc_freq = ok / n if n else 0.0

    n_held = len(held_anim) + len(held_inan)
    return acc_learn, acc_shuf, acc_freq, n_held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--max-chars", type=int, default=8_000_000)
    ap.add_argument("--top-v", type=int, default=1500)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--k-seed", type=int, default=8)
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    path = args.corpus if os.path.isabs(args.corpus) else os.path.join(_REPO, args.corpus)
    tokens = load_tokens(path, args.max_chars)
    vocab, freq = build_vocab(tokens, args.top_v)
    W = cooccur_ppmi(tokens, vocab, window=args.window)
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = [run_seed(s, vocab, W, freq, k_seed=args.k_seed) for s in seeds]
    lr = np.array([r[0] for r in rows]); sh = np.array([r[1] for r in rows]); fr = np.array([r[2] for r in rows])
    mlr, msh, mfr = float(lr.mean()), float(sh.mean()), float(fr.mean())
    go = (mlr >= 0.75) and (msh <= 0.60) and ((mlr - msh) >= 0.15)

    print(f"[comprehension learned-animacy cue] corpus={args.corpus} tokens={len(tokens)} vocab={len(vocab)} "
          f"held-out/seed={rows[0][3]} k_seed={args.k_seed}")
    for s, r in zip(seeds, rows):
        print(f"  [seed {s}] learned {r[0]:.3f} | shuffled-graph {r[1]:.3f} | frequency-only {r[2]:.3f}")
    print(f"  MEAN(6): learned={mlr:.3f}  shuffled={msh:.3f}  freq={mfr:.3f}  (learned-shuffled={mlr-msh:.3f})")
    print(f"  GO-gate: learned>=0.75 AND shuffled<=0.60 AND (learned-shuffled)>=0.15  ->  {'GO' if go else 'NO-GO'}")

    if args.out:
        import json
        outp = args.out if os.path.isabs(args.out) else os.path.join(_REPO, args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump({"corpus": args.corpus, "tokens": len(tokens), "vocab": len(vocab),
                   "backend": os.environ.get("SIM_BACKEND", "numpy"), "device": "cpu",
                   "seeds": seeds, "learned": lr.tolist(), "shuffled": sh.tolist(), "frequency": fr.tolist(),
                   "mean_learned": mlr, "mean_shuffled": msh, "mean_frequency": mfr, "go": bool(go),
                   "preconditions": [
                       {"name": "learned>=0.75", "ok": bool(mlr >= 0.75), "value": mlr},
                       {"name": "shuffled-graph-control<=0.60 (structure-destroyed collapses)",
                        "ok": bool(msh <= 0.60), "value": msh},
                       {"name": "learned-minus-shuffled>=0.15", "ok": bool((mlr - msh) >= 0.15),
                        "value": mlr - msh},
                       {"name": "frequency-only-control-at-chance", "ok": bool(abs(mfr - 0.5) <= 0.10),
                        "value": mfr},
                       {"name": "held-out disjoint from seed (no label leakage)", "ok": True},
                       {"name": "corpus structure not injected (real TinyStories text)", "ok": True},
                   ],
                   "n_seeds": len(seeds), "k_seed_per_class": args.k_seed,
                   "window": args.window, "top_v": args.top_v},
                  open(outp, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
