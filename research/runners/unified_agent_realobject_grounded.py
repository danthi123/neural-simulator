"""Option 2 (deep-semantic grounding): ground the NOUN concept codes in REAL OBJECT IMAGES (not synthetic patches),
through the same real biological V1 Gabor bank, and run the full unified-agent benchmark. The 2026-06-04 work grounded
nouns in SYNTHETIC distinct stimuli (the embodied-cognition stand-in); this replaces those with REAL object photos so
the V1 responses carry NATURAL image statistics (the redundancy the ventral stream actually evolved to decorrelate).
Verbs + adjectives stay word-encoder-grounded (no canonical object image — the honest abstract-concept limit).

Scientific questions: (1) do real-object V1 codes have higher/different cross-concept coherence than synthetic (natural
images are MORE redundant)? (2) does the ventral-decorrelation (ZCA stand-in) still yield composition-ready codes on
natural statistics? (3) does real grounding compose at parity, or does natural-image redundancy crowd the codes? This
is the embodied frontier; reported honestly side-by-side with the synthetic-grounded + raw baselines.

Reuse-by-import; NO protected-module edits.
  python -m research.runners.unified_agent_realobject_grounded --seeds 42 43 44 [--decorrelate]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from research.runners._visual_grounding_probe import _v1_matrix, _v1_code
from research.runners.unified_agent_visual_grounded import _decorrelate, D
from research.runners.unified_agent_benchmark import (
    build_vocab, ALL_FACTS, CATEGORIES, WHO_QUERIES, ABSTAIN_QUERIES, aggregate)
from research.runners.nested_composition_agent import NestedCompositionAgent
from sim.text_embeddings import vocab_to_drive_pattern

V1_DIM = 8 * 4 * 16 * 16     # 8192
WORD_DIM = 2048
RETINA = 32


def load_real_object_images(n, size=RETINA, seed=42):
    """Return (n, size, size) grayscale [0,1] images of REAL objects. Tries, in order: a cached CIFAR-10 batch
    (32x32 natural object photos — ideal), then sklearn's bundled handwritten digits (8x8 real, upscaled — no
    download). Each concept gets a DISTINCT real image (deterministic by index). Raises if no real data available."""
    # 1) cached CIFAR-10 (set REALOBJ_CIFAR to a data_batch path) — 32x32x3 natural objects.
    # SECURITY: CIFAR-10's canonical on-disk format IS pickle. REALOBJ_CIFAR must point to an OFFICIAL CIFAR-10
    # data_batch the owner downloaded from the canonical source (https://www.cs.toronto.edu/~kriz/cifar.html) — a
    # trusted file by construction. Never point it at an untrusted/arbitrary pickle (arbitrary-code-execution risk).
    cifar = os.environ.get("REALOBJ_CIFAR")
    if cifar and os.path.exists(cifar):
        import pickle
        with open(cifar, "rb") as f:
            d = pickle.load(f, encoding="bytes")  # trusted official CIFAR-10 batch only (see SECURITY note above)
        data = np.asarray(d[b"data"], dtype=np.float32)            # (N, 3072)
        imgs = data.reshape(-1, 3, 32, 32).mean(1) / 255.0          # grayscale (N,32,32)
        idx = np.random.default_rng(seed).permutation(len(imgs))[:n]
        return imgs[idx], "cifar10"
    # 2) sklearn bundled digits (no download) — real handwritten digits, 8x8 -> upscale
    from sklearn.datasets import load_digits
    dig = load_digits()
    imgs8 = dig.images.astype(np.float32) / 16.0                    # (1797,8,8) in [0,1]
    idx = np.random.default_rng(seed).permutation(len(imgs8))[:n]
    sel = imgs8[idx]
    up = size // imgs8.shape[1]                                     # 32//8 = 4
    return np.repeat(np.repeat(sel, up, axis=1), up, axis=2), "digits8x8_upscaled"


def image_to_retina(img):
    """Grayscale [0,1] (H,W) -> (2,H,W) ON/OFF retina. Center-surround-ish: ON = above-mean contrast, OFF =
    below-mean — drives the Gabor ON (+) / OFF (-) split with natural bright/dark structure."""
    m = float(img.mean())
    on = np.clip(img - m, 0.0, None)
    off = np.clip(m - img, 0.0, None)
    mx = max(on.max(), off.max(), 1e-6)
    return np.stack([on / mx, off / mx]).astype(np.float32)


def build_realobject_features(nouns, verbs, adjs, W, seed=42):
    """nouns -> REAL object image -> V1 Gabor response; verbs+adjs -> word encoder. Block-padded (V1 | word)."""
    n_nouns = len(nouns)
    imgs, src = load_real_object_images(n_nouns, seed=seed)
    v1 = np.stack([_v1_code(W, image_to_retina(imgs[i])) for i in range(n_nouns)])   # (n_nouns, 8192) unit-norm
    abstract = list(verbs) + list(adjs)
    word = np.stack([vocab_to_drive_pattern(t, n_neurons=WORD_DIM, sparsity=0.1) for t in abstract]).astype(np.float64)
    word = word / (np.linalg.norm(word, axis=1, keepdims=True) + 1e-12)
    tokens = list(nouns) + abstract
    dim = V1_DIM + WORD_DIM
    feats = np.zeros((len(tokens), dim), dtype=np.float64)
    feats[:n_nouns, :V1_DIM] = v1
    feats[n_nouns:, V1_DIM:] = word
    return feats, dim, tokens, src


def run_seed(seed, feats, dim, tokens, nouns, verbs, adjs, decorrelate):
    f = _decorrelate(feats) if decorrelate else feats
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((D, dim)) + 1j * rng.standard_normal((D, dim))
    Z = proj @ f.T
    ext = {t: np.angle(Z[:, i]) for i, t in enumerate(tokens)}
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=ext)
    for ag, ac, pa in ALL_FACTS:
        agent.learn(ag, ac, pa)
    res = {}
    for name, facts in CATEGORIES:
        res[name] = [sum(int(agent.query_patient(ag, ac) == (pa if isinstance(pa, str)
                          else agent._render_filler(pa))) for ag, ac, pa in facts), len(facts)]
    res["who"] = [sum(int(agent.query_agent(ac, pn) == w) for ac, pn, w in WHO_QUERIES), len(WHO_QUERIES)]
    res["abstain"] = [sum(int(agent.query_patient(ag, ac) is None) for ag, ac in ABSTAIN_QUERIES), len(ABSTAIN_QUERIES)]
    return {"seed": seed, "categories": res}


def main():
    ap = argparse.ArgumentParser(description="Unified-agent benchmark on REAL-OBJECT-grounded codes (option 2).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    nouns, verbs, adjs = build_vocab()
    W, n_v1 = _v1_matrix()
    feats, dim, tokens, src = build_realobject_features(nouns, verbs, adjs, W, seed=args.seeds[0])
    C = feats @ feats.T
    off = C[~np.eye(len(tokens), dtype=bool)]
    print(f"=== REAL-OBJECT grounded ({src}) | {len(tokens)} concepts ({len(nouns)} noun imgs) | D={D} ===", flush=True)
    print(f"  raw feature coherence: mean {off.mean():.3f}, max {off.max():.3f}", flush=True)
    out = {"source": src, "raw_coherence": [float(off.mean()), float(off.max())]}
    for label, dec in (("RAW", False), ("ZCA-decorrelated", True)):
        seed_res = [run_seed(s, feats, dim, tokens, nouns, verbs, adjs, dec) for s in args.seeds]
        agg, gok, gtot = aggregate(seed_res)
        out[label] = {"overall": [gok, gtot], "aggregate": agg}
        print(f"\n  [{label}] OVERALL {gok}/{gtot} = {gok/gtot*100:.1f}% ({len(args.seeds)} seeds)", flush=True)
        for c, (ok, tot, rate) in agg.items():
            print(f"    {c:<16} {ok:>3}/{tot:<3} = {rate*100:5.1f}%", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
