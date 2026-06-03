"""Does a LARGER retina fix the V1 letter-recognition separability limit? (Owner: no reason to cap at 32x32.)
Render a single letter (saccadic fovea) at retina_size R, through the real Gabor V1, learned recognition
across jitter+noise. If recognition rises sharply with R, the 32x32 cramping -- not the approach -- was the
recognition bottleneck."""
import numpy as np, time
from PIL import Image, ImageDraw, ImageFont
import sim.visual_cortex as VC
from research.findings.raw._text_as_pixels_probe import softmax, train_logreg
LET = list("abcdefghijklmnopqrstuvwxyz")
FPATH = "C:/Windows/Fonts/arial.ttf"

def make_v1(rs):
    npos = rs // 2
    pre, post, w = VC.build_v1_simple_weights(retina_size=rs, n_positions_per_dim=npos,
                                              receptive_field_radius=max(4, rs // 8))
    return pre, post, w, VC.N_ORIENTATIONS * VC.N_FREQUENCIES * npos * npos

def v1_letter(L, rs, font, pre, post, w, n_v1, rng):
    img = Image.new("L", (rs, rs), 0); d = ImageDraw.Draw(img)
    jit = max(2, rs // 16); dx, dy = int(rng.integers(-jit, jit + 1)), int(rng.integers(-jit, jit + 1))
    d.text((rs // 4 + dx, rs // 8 + dy), L, fill=255, font=font)
    a = np.asarray(img, np.float32) / 255.0
    a = np.clip(a + rng.normal(0, 0.12, a.shape).astype(np.float32), 0, 1)
    drive = VC.image_to_retina_drive(np.stack([a, (1 - a) * (a.max() > 0)]), 1.0)
    f = np.zeros(n_v1, np.float32); np.add.at(f, post, drive[pre] * w); f = np.maximum(f, 0)
    return f / (np.linalg.norm(f) + 1e-9)

print("=== retina size vs LEARNED single-letter recognition (jitter+noise, 26 letters, chance 0.038) ===", flush=True)
for rs in (32, 64, 96):
    t0 = time.time(); pre, post, w, n_v1 = make_v1(rs)
    font = ImageFont.truetype(FPATH, int(rs * 0.55))
    te = np.random.default_rng(7); Xte = []; Yte = []
    for i, L in enumerate(LET):
        for _ in range(10): Xte.append(v1_letter(L, rs, font, pre, post, w, n_v1, te)); Yte.append(i)
    tr = np.random.default_rng(1); X = []; Y = []
    for i, L in enumerate(LET):
        for _ in range(12): X.append(v1_letter(L, rs, font, pre, post, w, n_v1, tr)); Y.append(i)
    W = train_logreg(np.array(X), np.array(Y), 26, epochs=400, lr=0.5, seed=0)
    acc = float((softmax(np.array(Xte) @ W).argmax(1) == np.array(Yte)).mean())
    print(f"  retina {rs:>3}x{rs:<3} ({n_v1:>6} V1 cells) -> learned letter recognition {acc:.3f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
print("  -> if recognition rises sharply with retina size, 32x32 cramping was the bottleneck, not the approach.",
      flush=True)
