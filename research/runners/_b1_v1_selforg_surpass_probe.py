"""Fast reduced-scale OSI probe of the B1 on-bridge COMMON-MODE surpass levers.

The 2026-08-14 BOUNDARY: potentiation-only rate-Hebbian on ON/OFF-split full-field gratings
learns a COMMON MODE (ON and OFF potentiate equally -> signed RF cancels -> OSI~0), robust to
fixed FS inhibition + subtractive norm. Named-but-untried cheap levers:
  (1) --rule stdp : LTD (post->pre depression) supplies the input-specific DEPRESSION that
      potentiation-only Hebbian lacks -> can break the ON/OFF symmetry.
  (2) DoG center-surround front-end : removes the local DC (common mode) from the INPUT so the
      ON/OFF split carries signed local contrast the feedforward rule can bind (retina/LGN whitening).
This probe measures OSI (pre/post) learn-only at reduced scale to see if EITHER moves the needle.
"""
import os, sys, math, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np

from research.runners._b1_v1_selforg_onbridge_derisk import (
    build_v1_bridge, read_v1_rfs, gabor_orientation_tuning, _drive_image, _freeze,
    render_oriented_field,
)
from sim.backend import get_backend

xp, backend = get_backend()
xp = xp if backend == "cupy" else None


def dog_filter(signed, sigma_c=1.0, sigma_s=2.5, surround_w=0.85):
    """Center-surround difference-of-Gaussians (isotropic, orientation-free) on a 2D signed image.
    Removes the local DC/common mode -> whitened signed local contrast. Separable gaussian, numpy-only."""
    def gauss1d(sig):
        r = int(max(1, round(3 * sig)))
        x = np.arange(-r, r + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2 * sig * sig)); return k / k.sum()
    def blur(img, sig):
        k = gauss1d(sig)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
        return out.astype(np.float32)
    center = blur(signed, sigma_c)
    surround = blur(signed, sigma_s)
    return (center - surround_w * surround).astype(np.float32)


def render_field(rng, retina_size=32, shuffle=False, dog=False):
    H = W = retina_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = rng.uniform(0.0, math.pi); freq = rng.uniform(0.08, 0.30); phase = rng.uniform(0.0, 2 * math.pi)
    proj = xx * math.cos(theta) + yy * math.sin(theta)
    grating = np.cos(2 * math.pi * freq * proj + phase).astype(np.float32)
    cx = rng.uniform(0.25, 0.75) * W; cy = rng.uniform(0.25, 0.75) * H; sigma = rng.uniform(0.35, 0.6) * W
    env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    signed = grating * env
    if shuffle:
        flat = signed.reshape(-1).copy(); rng.shuffle(flat); signed = flat.reshape(H, W)
    if dog:
        signed = dog_filter(signed)
    on = np.maximum(signed, 0.0); off = np.maximum(-signed, 0.0)
    return np.stack([on, off], axis=0).astype(np.float32)


def develop(bridge, r0, n_retina, n_steps, drive_pA, present_steps, seed, shuffle=False, dog=False):
    rng = np.random.default_rng(seed * 101 + (7 if shuffle else 3))
    dt_ms = getattr(bridge.core_config, "dt_ms", getattr(bridge.core_config, "dt", 1.0))
    done = 0
    while done < n_steps:
        img = render_field(rng, shuffle=shuffle, dog=dog)
        _drive_image(bridge, r0, n_retina, img, drive_pA, xp)
        for _ in range(present_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_ms += dt_ms   # FIX: advance clock so STDP delta_t != 0
            done += 1
            if done >= n_steps:
                break
    bridge.cp_external_input_current[:] = 0.0


def run_config(name, seed, rule, dog, n_inh=0, dev_steps=10000, n_pos=8, drive_pA=1200.0):
    n_orient, n_freq, retina_size, radius = 8, 4, 32, 4
    n_v1 = n_orient * n_freq * n_pos * n_pos
    t0 = time.time()
    b, r0, v0, nret, _ = build_v1_bridge(
        seed, n_orient, n_freq, n_pos, retina_size, radius,
        30.0, 7.0, 0.05, 0.00002, 70.0, 0.85, 0.03,
        syn_scaling=True, syn_scaling_rate=0.02, n_inh=n_inh, inh_exc_w=6.0, inh_inh_w=12.0,
        inh_density=0.25, homeo_target=0.012, homeo_ema_alpha=0.01, homeo_adapt_rate=0.004, rule=rule)
    rf_pre = read_v1_rfs(b, r0, v0, nret, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    m_pre, f_pre = gabor_orientation_tuning(rf_pre)
    develop(b, r0, nret, dev_steps, drive_pA, 40, seed, shuffle=False, dog=dog)
    _freeze(b)
    rf_post = read_v1_rfs(b, r0, v0, nret, n_v1, n_orient, n_freq, n_pos, retina_size, radius)
    m_post, f_post = gabor_orientation_tuning(rf_post)
    # raw ON/OFF common-mode readout
    from research.runners._b1_v1_selforg_onbridge_derisk import raw_weight_stats
    raw = raw_weight_stats(b, r0, v0, nret, n_v1, retina_size)
    dt = time.time() - t0
    print(f"[{name}] seed={seed} rule={rule} dog={dog} n_inh={n_inh}  "
          f"OSI pre f={f_pre:.4f} m={m_pre:.4f} -> POST f={f_post:.4f} m={m_post:.4f}  "
          f"| on-off={raw['on_minus_off_mean']:.4f} l2={raw['l2_mean']:.2f}  ({dt:.0f}s)", flush=True)
    return dict(name=name, seed=seed, rule=rule, dog=bool(dog),
               osi_pre_frac=round(float(f_pre), 4), osi_post_frac=round(float(f_post), 4),
               osi_post_mean=round(float(m_post), 4),
               on_minus_off_mean=raw["on_minus_off_mean"], l2_mean=raw["l2_mean"])


if __name__ == "__main__":
    import json
    from pathlib import Path
    seed = 42
    print("=== reduced-scale (n_v1=2048, dev=10000) learn-only OSI surpass probe (CLOCK-FIXED) ===", flush=True)
    rows = [
        run_config("A_hebb_base", seed, "hebbian", dog=False),
        run_config("B_stdp", seed, "stdp", dog=False),
        run_config("C_both", seed, "both", dog=False),
        run_config("D_hebb_dog", seed, "hebbian", dog=True),
    ]
    out = dict(
        note="Reduced-scale (n_v1=2048, dev_steps=10000) seed-42 learn-only OSI probe of the cheap no-sim-edit "
             "surpass levers for the B1 on-bridge common-mode BOUNDARY. CLOCK-FIXED so --rule stdp is genuinely "
             "exercised (it was silently inert in prior runs). on_minus_off_mean ~0 for every rule = common mode.",
        arch="8x4x8x8", dev_steps=10000, backend=backend, rows=rows)
    p = Path("research/findings/raw/_b1_v1_selforg_surpass_probe_s42.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("[written]", p, flush=True)
