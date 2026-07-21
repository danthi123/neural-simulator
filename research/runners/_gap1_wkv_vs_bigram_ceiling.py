"""gap#1 (open generation) CEILING — does the WKV cortex BEAT a bigram on held-out TinyStories at its current scale?

Per the skill's "run the ceiling early — it bounds the whole investigation." The 2026-07-11 finding showed even a
transformer loses to a bigram at 5M-tok/V=300 (the scale-confound). This asks the same for OUR deployed WKV cortex
(v4000, d256, trained on ~4M-token TinyStories): if it BEATS the bigram, gap#1 is scale-PROGRESSING (more data/scale
is the lever); if it LOSES/ties, the scale-confound bites at this scale and the lever is a much bigger corpus (or the
88.6M spiking-forward). Reuse-by-import (TorchWKV, load_tiny_sentences). `--ckpt`, `--n`.
"""
import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

import re as _re
from research.runners._gap_grounded_wkv_finetune import TorchWKV, load_tiny_sentences
_WORD = _re.compile(r"[a-z']+")


def load_range(path, w2i, skip, take, min_len=5, max_len=24):
    """Tokenize the corpus, SKIP the first `skip` valid sentences, then take `take`. Same tokenization as
    load_tiny_sentences. NOTE (2026-07-21 audit): the big ckpt trained on data/corpus/tinystories_train.txt
    (n_tr=400000, see raw/_gap1_train_big.log), NOT the first N of THIS file (tinystories.txt) -- so skipping
    `skip` sentences here does NOT make the tail disjoint from the WKV's training corpus. Treat this ceiling's
    magnitude as UNRELIABLE; the trustworthy disjoint measurement is _emerge_wkv_lm_derisk.py (random 85/15 split)."""
    txt = open(path, "r", errors="ignore").read().lower()
    out, seen = [], 0
    for raw in _re.split(r"[.!?]", txt):
        toks = _WORD.findall(raw)
        if not (min_len <= len(toks) <= max_len):
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append([w2i.get(t, w2i["<unk>"]) for t in toks])
        if len(out) >= take:
            break
    return out


def wkv_heldout_ppl(model, sents, dev, pad=0):
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(sents), 64):
            batch = sents[i:i + 64]
            L = max(len(s) for s in batch)
            X = torch.full((len(batch), L), pad, dtype=torch.long, device=dev)
            for j, s in enumerate(batch):
                X[j, :len(s)] = torch.tensor(s, device=dev)
            lg = model(X)                                     # [B,L,V]
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, model.V), X[:, 1:].reshape(-1), reduction="none")
            loss = loss.reshape(len(batch), L - 1)
            for j, s in enumerate(batch):
                n = len(s) - 1
                tot_nll += float(loss[j, :n].sum()); tot_tok += n
    return math.exp(tot_nll / max(tot_tok, 1))


def bigram_heldout_ppl(train, test, V, lam=0.7):
    # FAIR bigram: Jelinek-Mercer interpolation P(w|a) = lam*P_bi(w|a) + (1-lam)*P_uni(w), unigram add-1 smoothed.
    # This is the standard fair bigram baseline (backoff to unigram handles unseen bigrams without over-penalizing).
    from collections import defaultdict
    c2 = defaultdict(lambda: defaultdict(int)); c1 = defaultdict(int); cu = defaultdict(int); tot_u = 0
    for s in train:
        for w in s:
            cu[w] += 1; tot_u += 1
        for a, b in zip(s[:-1], s[1:]):
            c2[a][b] += 1; c1[a] += 1

    def p_uni(w):
        return (cu.get(w, 0) + 1.0) / (tot_u + V)             # add-1 unigram (never zero)

    tot_nll, tot_tok = 0.0, 0
    for s in test:
        for a, b in zip(s[:-1], s[1:]):
            p_bi = (c2[a].get(b, 0) / c1[a]) if c1.get(a, 0) > 0 else 0.0
            p = lam * p_bi + (1.0 - lam) * p_uni(b)
            tot_nll += -math.log(max(p, 1e-12)); tot_tok += 1
    return math.exp(tot_nll / max(tot_tok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="bridges/wkv_ckpt/wkv_ssmU_v4000_d256_big_seed42.npz")
    ap.add_argument("--corpus", default="data/corpus/tinystories.txt")
    ap.add_argument("--n", type=int, default=8000, help="total sentences (split 85/15 train/held-out)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(args.ckpt, allow_pickle=True)
    words = list(z["words"]); V = int(z["V"]); D = int(z["d_model"])
    w2i = {w: i for i, w in enumerate(words)}
    if "<unk>" not in w2i:
        w2i["<unk>"] = V - 1
    model = TorchWKV(V, D).to(dev); model.load_npz(z, 0)

    # AUDIT CORRECTION (2026-07-21): the original "NO LEAKAGE / WKV trained on the first 100000 of tinystories.txt"
    # premise is FALSE. The ckpt trained on data/corpus/tinystories_train.txt (n_tr=400000, raw/_gap1_train_big.log);
    # this eval reads a DIFFERENT file (tinystories.txt), so skipping 120000 here does NOT make the tail disjoint from
    # training (~17.7% of the tail is verbatim in the training corpus). The 3.35x magnitude from this runner is
    # UNRELIABLE. Behavior below is UNCHANGED (narration-only fix); trust _emerge_wkv_lm_derisk.py's random 85/15 split.
    train = load_tiny_sentences(args.corpus, 20000, w2i)      # bigram trained on 20000 sents (NB ~20x fewer than the WKV's 400000 -> inflates the ratio)
    test = load_range(args.corpus, w2i, skip=120000, take=args.n)   # tail of tinystories.txt (NOT verified disjoint from the WKV's training corpus)
    print(f"[ceiling] V={V} D={D} | bigram-train {len(train)} sents / held-out {len(test)} sents past #120000 "
          f"(~{sum(len(s) for s in test)} tokens; NB NOT verified unseen by the WKV -- see 2026-07-21 audit correction)")

    wkv_ppl = wkv_heldout_ppl(model, test, dev)
    big_ppl = bigram_heldout_ppl(train, test, V, lam=0.7)
    unigram_chance = V  # rough
    beats = wkv_ppl < big_ppl
    print(f"[RESULT] gap#1 ceiling (WKV v{V}_d{D} vs bigram on held-out TinyStories):")
    print(f"  WKV cortex held-out ppl   : {wkv_ppl:.2f}")
    print(f"  bigram   held-out ppl     : {big_ppl:.2f}  (Jelinek-Mercer interpolated, fair)")
    print(f"  WKV BEATS bigram          : {beats}  (ratio bigram/WKV = {big_ppl/max(wkv_ppl,1e-9):.2f}x)")
    print(f"  => gap#1 is {'SCALE-PROGRESSING (the WKV cortex beats the bigram -> more data/scale is the lever)' if beats else 'SCALE-CONFOUNDED at this scale (WKV does not beat the bigram -> the lever is a MUCH bigger corpus or the 88.6M spiking-forward, not this scale)'}.")
    json.dump({"V": V, "D": D, "wkv_ppl": wkv_ppl, "bigram_ppl": big_ppl, "beats": bool(beats),
               "n_test_tokens": sum(len(s) for s in test)},
              open("research/findings/raw/_gap1_ceiling.json", "w"), indent=2)


if __name__ == "__main__":
    main()
