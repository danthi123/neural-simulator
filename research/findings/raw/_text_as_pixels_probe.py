"""Cheap-first probe: text-as-pixels (shared letter-glyphs) lets the network READ NOVEL words from known
letters -- the compositional-orthography data-efficiency an orthogonal tokenizer fundamentally lacks.

Owner input-side-fidelity insight: render text as pixels through the visual pathway instead of tokenizing.
Then a word is a SPATIAL ARRANGEMENT OF LETTER-GLYPHS (shared visual structure) -- so once the ~L letters are
learned (from a few words), ANY of L^n_pos words is readable, including words never seen. The tokenizer gives
each word an independent orthogonal symbol -> a NOVEL word is an unseen symbol -> unreadable. So text-as-pixels
turns "learn W words" into "learn L letters -> read L^n words": a combinatorial data-efficiency gain.

Controlled test (only the input representation differs): VISUAL code(word) = concat of its letters' glyphs
(shared glyphs across words). ORTHOGONAL code(word) = an independent random vector per word (tokenizer regime).
Task: READ the word = recover the letter at each position. Train on K words; test on HELD-OUT NOVEL words
(same letters, novel combinations). Metric: per-letter reading accuracy on novel words vs K, visual vs
orthogonal. Pre-registered: visual reads novel words at ~1.0 from few K (letters shared); orthogonal stays at
chance on novel words (unseen symbol). Stdlib + numpy only; no protected import.

  python -m research.findings.raw._text_as_pixels_probe
"""
from __future__ import annotations
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def train_logreg(X, Y, n_cls, epochs=250, lr=0.3, l2=1e-4, seed=0):
    W = np.random.default_rng(seed).standard_normal((X.shape[1], n_cls)) * 0.01
    eye = np.eye(n_cls)
    for _ in range(epochs):
        p = softmax(X @ W)
        W -= lr * (X.T @ (p - eye[Y]) / len(Y) + l2 * W)
    return W


def run(seed, n_letters=10, n_pos=3, G=32, n_words=200, K_list=(5, 12, 30, 50, 80, 120)):
    rng = np.random.default_rng(seed)
    glyph = rng.standard_normal((n_letters, G))                       # each letter's visual feature (shared)
    words = set()
    while len(words) < n_words:
        words.add(tuple(int(x) for x in rng.integers(0, n_letters, size=n_pos)))
    words = list(words); rng.shuffle(words)
    n_ho = len(words) // 3
    ho_words, pool = words[-n_ho:], words[:-n_ho]
    orth = {w: rng.standard_normal(n_pos * G) for w in words}         # tokenizer: independent per word

    def vis_code(w):
        return np.concatenate([glyph[l] for l in w])
    Xho_v = np.array([vis_code(w) for w in ho_words])
    Xho_o = np.array([orth[w] for w in ho_words])
    out = {"visual": [], "orthogonal": []}
    for K in K_list:
        tr = pool[:K]
        # build per-(word,position) reading examples
        def build(words_subset, coder):
            X, Y = [], []
            for w in words_subset:
                c = coder(w)
                for pos in range(n_pos):
                    X.append(c); Y.append(w[pos] + pos * n_letters)   # position-tagged letter class
            return np.array(X), np.array(Y)
        # VISUAL: read each position from its glyph-slice (position-wise classifier)
        Xv, Yv = build(tr, vis_code)
        Wv = train_logreg(Xv, Yv, n_pos * n_letters, seed=seed)
        # reading novel words: per position, predict letter
        def read_acc(code_mat, W, wordset):
            P = softmax(code_mat @ W)
            ok = tot = 0
            for i, w in enumerate(wordset):
                pred = P[i].reshape(1, -1)
                for pos in range(n_pos):
                    sl = pred[0, pos * n_letters:(pos + 1) * n_letters]
                    ok += int(int(np.argmax(sl)) == w[pos]); tot += 1
            return ok / tot
        out["visual"].append((len(tr), read_acc(Xho_v, Wv, ho_words)))
        # ORTHOGONAL: same reading task from the whole-word arbitrary code
        Xo, Yo = build(tr, lambda w: orth[w])
        Wo = train_logreg(Xo, Yo, n_pos * n_letters, seed=seed)
        out["orthogonal"].append((len(tr), read_acc(Xho_o, Wo, ho_words)))
    return out


def main():
    print("=== text-as-pixels: reading NOVEL words from known letters (visual glyphs vs orthogonal tokens) ===",
          flush=True)
    seeds = [42, 43, 44]
    agg = {"visual": {}, "orthogonal": {}}
    for s in seeds:
        r = run(s)
        for cond in agg:
            for k, a in r[cond]:
                agg[cond].setdefault(k, []).append(a)
    chance = 1.0 / 10
    for k in sorted(agg["visual"]):
        vm = float(np.mean(agg["visual"][k])); om = float(np.mean(agg["orthogonal"][k]))
        print(f"   {k:>3} train words | visual novel-word read {vm:.3f} | orthogonal {om:.3f} (chance {chance:.2f})",
              flush=True)
    vbest = max(np.mean(v) for v in agg["visual"].values())
    obest = max(np.mean(v) for v in agg["orthogonal"].values())
    if vbest >= 0.9 and obest < 0.3:
        print(f"\nVERDICT: RESOLVES -- text-as-pixels READS NOVEL words ({vbest:.2f}) from a handful of training "
              f"words (letters shared); orthogonal tokens CANNOT read novel words ({obest:.2f} ~ chance) -- each "
              f"word is an unseen symbol. Combinatorial data-efficiency (learn ~L letters -> read L^n words). "
              f"-> the faithful build (render text through the EXISTING visual pathway) removes the tokenizer "
              f"shortcut and makes recognition data-efficient + open-vocabulary.", flush=True)
    else:
        print(f"\nVERDICT: visual {vbest:.2f} / orthogonal {obest:.2f} -- re-examine.", flush=True)


if __name__ == "__main__":
    main()
