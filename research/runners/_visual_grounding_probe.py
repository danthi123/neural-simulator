"""Cheat-removal #4 (cheap-first): do REAL Gabor-V1 sensory features ground concept codes?

The composition agent's concept codes are currently random phasors or learned from a hashed/orthogonal WORD
encoder (vocab_to_drive_pattern). #4 asks for SENSORY-grounded codes -- the visual-cortex/Gabor pipeline
(sim/visual_cortex.py) feeding the concept representations. This probe answers the foundational question cheaply,
before any agent integration: take N distinct visual stimuli (one per "visual concept"), pass each through the
REAL biological V1 Gabor receptive-field bank (build_v1_simple_weights -- Hubel-Wiesel oriented simple cells),
and measure whether the resulting V1 feature vectors are:
  (1) SEPARABLE: different concepts -> low pairwise cosine (distinct codes);
  (2) CLEANUP-ABLE under realistic corruption (additive noise + small translation): a corrupted stimulus's V1
      code still has its nearest CLEAN concept code = the true concept (CA3-style pattern completion -- the same
      mechanism that resolved the grounded word-cue level of #4).
If both hold, real sensory features yield usable, robust grounded concept codes; the agent integration (feed
these as `external_codes`) is the follow-up. Honest note: clean separability is partly BY CONSTRUCTION (V1 is an
orientation/position discriminator) -- the informative test is robustness (noise + translation), where simple
cells are NOT translation-invariant, so cleanup must carry the load.

  SIM_BACKEND=numpy python -m research.runners._visual_grounding_probe
"""
import math
import numpy as np
import scipy.sparse as sps

from sim.visual_cortex import build_v1_simple_weights, image_to_retina_drive, RETINA_SIZE


def _v1_matrix():
    """Sparse retina(2*32*32) -> V1_simple(8192) Gabor weight matrix W; v1_response = W.T @ retina_drive."""
    pre, post, w = build_v1_simple_weights()
    n_ret = 2 * RETINA_SIZE * RETINA_SIZE
    n_v1 = 8 * 4 * 16 * 16
    return sps.csr_matrix((w, (pre, post)), shape=(n_ret, n_v1)), n_v1


def render_bar(theta, size=RETINA_SIZE, thickness=1.6, shift=(0, 0)):
    """An oriented bar through center (+optional pixel shift) in the ON channel -- a distinct visual stimulus."""
    img = np.zeros((2, size, size), dtype=np.float32)
    c = size / 2.0
    for y in range(size):
        for x in range(size):
            dx, dy = (x - c - shift[0]), (y - c - shift[1])
            perp = abs(-dx * math.sin(theta) + dy * math.cos(theta))
            along = abs(dx * math.cos(theta) + dy * math.sin(theta))
            if perp < thickness and along < size * 0.42:
                img[0, y, x] = 1.0
    return img


def render_spot(cx, cy, size=RETINA_SIZE, r=3.0, shift=(0, 0)):
    """A round spot at (cx,cy) -- a non-oriented visual stimulus."""
    img = np.zeros((2, size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if (x - cx - shift[0]) ** 2 + (y - cy - shift[1]) ** 2 < r * r:
                img[0, y, x] = 1.0
    return img


def _stimuli():
    """12 distinct visual 'concepts': 8 oriented bars + 4 corner spots."""
    stim = {}
    for i in range(8):
        stim[f"bar_{i*180//8}deg"] = lambda th=i * math.pi / 8: render_bar(th)
    q = RETINA_SIZE // 4
    for nm, (cx, cy) in {"spot_TL": (q, q), "spot_TR": (3 * q, q),
                         "spot_BL": (q, 3 * q), "spot_BR": (3 * q, 3 * q)}.items():
        stim[nm] = lambda cx=cx, cy=cy: render_spot(cx, cy)
    return stim


def _v1_code(W, img):
    drive = image_to_retina_drive(img, drive_max_pA=1.0)   # unit drive; we only need the response geometry
    resp = W.T @ drive
    n = np.linalg.norm(resp)
    return resp / n if n > 0 else resp


def _corrupt(render_fn, rng, noise=0.25, max_shift=2):
    img = render_fn()
    sx, sy = int(rng.integers(-max_shift, max_shift + 1)), int(rng.integers(-max_shift, max_shift + 1))
    img = np.roll(np.roll(img, sx, axis=2), sy, axis=1)        # small translation (simple cells are NOT shift-inv)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, None)


def main():
    print("=== #4 cheap-first: Gabor-V1 sensory grounding of concept codes ===\n", flush=True)
    W, n_v1 = _v1_matrix()
    stim = _stimuli()
    names = list(stim)
    clean = {nm: _v1_code(W, fn()) for nm, fn in stim.items()}
    M = np.stack([clean[nm] for nm in names])                 # (N, n_v1) unit codes

    # (1) separability: off-diagonal pairwise cosine. The MAX captures genuinely-similar stimuli (adjacent
    # orientations) -- that high cosine is CORRECT, not a failure -- so MEAN is the separability summary and the
    # most-similar pair is printed to confirm it is a real visual-similarity pair.
    C = M @ M.T
    Cm = C.copy(); np.fill_diagonal(Cm, -1.0)
    i, j = np.unravel_index(int(np.argmax(Cm)), Cm.shape)
    off = C[~np.eye(len(names), dtype=bool)]
    print(f"(1) SEPARABILITY ({len(names)} visual concepts, V1 dim {n_v1}):", flush=True)
    print(f"    pairwise cosine: mean={off.mean():.3f}  max={off.max():.3f} "
          f"(most-similar pair: {names[i]} ~ {names[j]} -- adjacent stimuli SHOULD be similar)\n", flush=True)

    # (2) cleanup under corruption: noisy+shifted stimulus -> nearest clean code = true concept?
    rng = np.random.default_rng(42)
    trials = 5
    correct = 0
    margins = []
    for nm, fn in stim.items():
        for _ in range(trials):
            code = _v1_code(W, _corrupt(fn, rng))
            sims = M @ code
            best = names[int(np.argmax(sims))]
            srt = np.sort(sims)[::-1]
            margins.append(float(srt[0] - srt[1]))
            correct += int(best == nm)
    total = len(stim) * trials
    print(f"(2) PATTERN-COMPLETION CLEANUP (noise=0.25 + translation<=2px, {trials}/concept):", flush=True)
    print(f"    recovered true concept {correct}/{total} = {100*correct/total:.0f}%  "
          f"(mean top1-top2 margin {np.mean(margins):.3f})", flush=True)
    # Functional verdict: well-separated ON AVERAGE (mean cosine low) AND robust cleanup recovers the true concept.
    # (Max cosine is NOT a failure criterion -- genuinely-similar stimuli are correctly similar; cleanup is the
    # ground-truth functional test, exactly as it was for the grounded word-cue level of #4.)
    sep_ok = off.mean() < 0.4
    clean_ok = correct >= 0.9 * total
    print(f"\n  => {'GROUNDED CODES USABLE (well-separated on average + robust cleanup)' if (sep_ok and clean_ok) else 'NEEDS WORK'}"
          f"  [mean-separable={sep_ok}, cleanup={clean_ok}]", flush=True)


if __name__ == "__main__":
    main()
