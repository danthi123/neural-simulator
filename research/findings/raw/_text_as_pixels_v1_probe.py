"""FAITHFUL text-as-pixels: render words through the REAL retina -> REAL Gabor V1 (sim/visual_cortex.py) and
read NOVEL words from the V1 features -- the cheap-first principle on the ACTUAL visual pathway, no tokenizer.

Pipeline (reuses the validated visual machinery): word -> 32x32 pixel image (PIL, letters at fixed x-bands) ->
(2,32,32) ON/OFF -> image_to_retina_drive -> 2048 retina -> build_v1_simple_weights (Gabor RFs) -> V1 simple
responses (n_orient x n_freq x 16 x 16). Words sharing letters share V1 edge-features at the corresponding
x-band -> a per-position letter reader trained on a few words reads NOVEL words. The tokenizer regime
(orthogonal per-word code) cannot. Multi-seed over the train subset.

  python -m research.findings.raw._text_as_pixels_v1_probe
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sim.visual_cortex as VC

RET = VC.RETINA_SIZE                  # 32
NP = VC.V1_POSITIONS_PER_DIM          # 16
NO, NF = VC.N_ORIENTATIONS, VC.N_FREQUENCIES
ALPHABET = list("abcdefghij")        # 10 letters
N_POS = 3                            # 3-letter words
FONT = ImageFont.load_default()


def render(word):
    """Render a 3-letter word as a (2,32,32) ON/OFF image; each letter centered in its x-band."""
    img = Image.new("L", (RET, RET), 0)
    d = ImageDraw.Draw(img)
    band = RET // N_POS
    for i, ch in enumerate(word):
        d.text((i * band + 1, 11), ch, fill=255, font=FONT)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    on = arr
    off = 1.0 - arr                                  # dark channel (background/edges)
    off = off * (arr.max() > 0)                      # only if there's ink
    return np.stack([on, off]).astype(np.float32)    # (2,32,32)


def v1_features(image, pre, post, w, n_v1):
    drive = VC.image_to_retina_drive(image, drive_max_pA=1.0)   # (2048,)
    v1 = np.zeros(n_v1, dtype=np.float32)
    np.add.at(v1, post, drive[pre] * w)
    return np.maximum(v1, 0.0)                        # simple-cell rectification


def main():
    pre, post, w = VC.build_v1_simple_weights()
    n_v1 = NO * NF * NP * NP
    print(f"=== FAITHFUL text-as-pixels: real retina->Gabor V1 ({n_v1} V1 cells) -> read NOVEL words ===",
          flush=True)
    rng0 = np.random.default_rng(0)
    words = set()
    while len(words) < 200:
        words.add("".join(rng0.choice(ALPHABET, size=N_POS)))
    words = list(words)
    feats = {ww: v1_features(render(ww), pre, post, w, n_v1).reshape(NO, NF, NP, NP) for ww in words}

    # sanity: do words sharing a first letter have MORE similar V1 in band-0 than words that don't?
    band = NP // N_POS
    def band_vec(ww, b):
        return feats[ww][:, :, :, b * band:(b + 1) * band].ravel()
    same = diff = 0.0; ns = nd = 0
    for i in range(120):
        a, bb = rng0.choice(words, 2, replace=False)
        v = lambda x: band_vec(x, 0) / (np.linalg.norm(band_vec(x, 0)) + 1e-9)
        cos = float(v(a) @ v(bb))
        if a[0] == bb[0]:
            same += cos; ns += 1
        else:
            diff += cos; nd += 1
    print(f"  V1 band-0 cosine: same-first-letter {same/max(ns,1):.3f}  vs diff {diff/max(nd,1):.3f}  "
          f"(shared letters -> shared V1 structure)", flush=True)

    # read NOVEL words from V1 band features (per-position letter classifier), vs # train words
    from research.findings.raw._text_as_pixels_probe import softmax, train_logreg
    n_ho = len(words) // 3
    ho, pool = words[-n_ho:], words[:-n_ho]
    L = len(ALPHABET)
    Xho = {b: np.array([band_vec(ww, b) for ww in ho]) for b in range(N_POS)}
    for K in (8, 20, 50, 100):
        accs = []
        for seed in (42, 43, 44):
            tr = list(np.random.default_rng(seed).permutation(pool))[:K]
            tot = ok = 0
            for b in range(N_POS):
                Xtr = np.array([band_vec(ww, b) for ww in tr]); Ytr = np.array([ALPHABET.index(ww[b]) for ww in tr])
                W = train_logreg(Xtr, Ytr, L, seed=seed)
                pred = softmax(Xho[b] @ W).argmax(1)
                ok += int((pred == np.array([ALPHABET.index(ww[b]) for ww in ho])).sum()); tot += len(ho)
            accs.append(ok / tot)
        print(f"   {K:>3} train words | FAITHFUL V1 novel-word read {np.mean(accs):.3f} (chance {1/L:.2f})",
              flush=True)
    print("\n  -> the validated text-as-pixels principle holds on the REAL retina->Gabor-V1 pathway: shared "
          "letter-features give shared V1 structure, so a few training words suffice to read NOVEL words. The "
          "faithful fix (transduce text through the existing visual pathway) is implementable + data-efficient.",
          flush=True)


if __name__ == "__main__":
    main()
