"""Vectorized + frequency-SCALED Gabor V1: does a bigger, properly-tuned retina fix letter recognition?
(Owner: don't cap at 32x32; and the pure-Python V1 build was single-threaded/low-util.) Vectorized numpy
conv (fast); Gabor freqs scaled by 32/retina_size so filters track letter scale at any retina size."""
import numpy as np, math, time
from PIL import Image, ImageDraw, ImageFont
import sim.visual_cortex as VC
from research.findings.raw._text_as_pixels_probe import softmax, train_logreg
try:
    from scipy.ndimage import correlate
    def conv(a, k): return correlate(a, k, mode="constant")
except Exception:
    def conv(a, k):
        from numpy.fft import rfft2, irfft2
        H, W = a.shape; kh, kw = k.shape
        pad = np.zeros_like(a); pad[:kh, :kw] = k
        pad = np.roll(pad, (-(kh // 2), -(kw // 2)), (0, 1))
        return irfft2(rfft2(a) * rfft2(pad), a.shape)
LET = list("abcdefghijklmnopqrstuvwxyz"); FP = "C:/Windows/Fonts/arial.ttf"
NO, NF = VC.N_ORIENTATIONS, VC.N_FREQUENCIES

def gabor2d(sigma, theta, freq):
    half = max(3, int(3 * sigma)); y, x = np.mgrid[-half:half + 1, -half:half + 1]
    xr = x * math.cos(theta) + y * math.sin(theta); yr = -x * math.sin(theta) + y * math.cos(theta)
    return (np.exp(-(xr**2 + yr**2) / (2 * sigma**2)) * np.cos(2 * math.pi * freq * xr)).astype(np.float32)

def v1_feat(L, rs, font, scale, rng):
    img = Image.new("L", (rs, rs), 0); d = ImageDraw.Draw(img)
    jit = max(2, rs // 16); dx, dy = int(rng.integers(-jit, jit + 1)), int(rng.integers(-jit, jit + 1))
    d.text((rs // 4 + dx, rs // 8 + dy), L, fill=255, font=font)
    a = np.asarray(img, np.float32) / 255.0; a = np.clip(a + rng.normal(0, 0.12, a.shape).astype(np.float32), 0, 1)
    on = a; off = (1 - a) * (a.max() > 0)
    freqs = [f / scale for f in (0.05, 0.10, 0.20, 0.40)][:NF]; sigmas = [s * scale for s in (3., 2.5, 2., 1.5)][:NF]
    npos = rs // 2; feats = []
    for oi in range(NO):
        th = oi * math.pi / NO
        for fr, sg in zip(freqs, sigmas):
            g = gabor2d(sg, th, fr); resp = np.maximum(conv(on, np.maximum(g, 0)) + conv(off, np.maximum(-g, 0)), 0)
            feats.append(resp[::2, ::2][:npos, :npos].ravel())
    f = np.concatenate(feats); return f / (np.linalg.norm(f) + 1e-9)

print("=== VECTORIZED scaled-Gabor V1: retina size vs learned letter recognition (chance 0.038) ===", flush=True)
for rs in (32, 64, 96, 128):
    t0 = time.time(); scale = rs / 32.0; font = ImageFont.truetype(FP, int(rs * 0.55))
    te = np.random.default_rng(7); Xte = np.array([v1_feat(L, rs, font, scale, te) for L in LET for _ in range(10)])
    Yte = np.array([i for i in range(26) for _ in range(10)])
    tr = np.random.default_rng(1); X = np.array([v1_feat(L, rs, font, scale, tr) for L in LET for _ in range(15)])
    Y = np.array([i for i in range(26) for _ in range(15)])
    W = train_logreg(X, Y, 26, epochs=400, lr=0.5, seed=0)
    acc = float((softmax(Xte @ W).argmax(1) == Yte).mean())
    print(f"  retina {rs:>3}x{rs:<3} (scaled Gabors) -> recognition {acc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
