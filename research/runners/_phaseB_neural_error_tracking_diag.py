"""Diagnostic for the on-bridge neural-error BOUNDARY (CYCLE 160): does the spiking error population's output
actually TRACK the host error `target - est`? No training -- just build the error-population bridge and, over many
(target, est) samples, correlate the neural error (ON_rate - OFF_rate, calibrated) against the host error per
output. High correlation => the error tracks (the boundary was budget/SNR -> more passes / population coding fix
it). Low correlation => the LIF f-I band / calibration distorts the error (fix the read before any convergence run).

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_error_tracking_diag [--drive 6400] [--window 20]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import research.runners._phaseB_onbridge_neural_error_readout_derisk as nerd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=12)
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    if args.drive is not None:
        nerd.ERR_DRIVE = args.drive
    if args.window is not None:
        nerd.ERR_WINDOW = args.window

    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    F = 16
    fillers = codes[:F]; d_in = fillers.shape[1]
    rng = np.random.default_rng(args.seed)

    eb, err = nerd.build_error_bridge(d_in, args.seed)
    # calibrate exactly as the runner does
    tgt0 = fillers[int(rng.integers(F))]; es0 = rng.standard_normal(d_in) * 0.1
    host_mag = float(np.mean(np.abs(tgt0 - es0)) + 1e-9)
    raw = nerd.neural_error(eb, err, tgt0, es0, d_in, 1.0)
    cal = host_mag / (float(np.mean(np.abs(raw))) + 1e-9)
    print(f"[neural-error tracking diag] drive={nerd.ERR_DRIVE} window={nerd.ERR_WINDOW} cal={cal:.3f}", flush=True)

    corrs, sign_accs, mag_ratios = [], [], []
    for s in range(args.n_samples):
        tgt = fillers[int(rng.integers(F))]
        # est at varying training stages: from ~0 (W=0 start) to a partial readout
        est = rng.standard_normal(d_in) * (0.02 + 0.08 * (s / max(args.n_samples - 1, 1)))
        host_err = tgt - est
        neural = nerd.neural_error(eb, err, tgt, est, d_in, cal)
        c = float(np.corrcoef(host_err, neural)[0, 1])
        sign_acc = float(np.mean(np.sign(host_err) == np.sign(neural)))
        mag_ratio = float(np.mean(np.abs(neural)) / (np.mean(np.abs(host_err)) + 1e-9))
        corrs.append(c); sign_accs.append(sign_acc); mag_ratios.append(mag_ratio)
        print(f"  sample {s:2d}: corr(neural,host)={c:+.3f}  sign-acc={sign_acc:.3f}  |neural|/|host|={mag_ratio:.2f}",
              flush=True)

    mc, ms, mm = float(np.mean(corrs)), float(np.mean(sign_accs)), float(np.mean(mag_ratios))
    print(f"\n  MEAN: corr={mc:+.3f} | sign-acc={ms:.3f} | mag-ratio={mm:.2f}", flush=True)
    if mc >= 0.7 and ms >= 0.8:
        print(f"  TRACKS: the neural error follows the host error (corr {mc:.2f}, sign {ms:.2f}) -> the boundary "
              f"was budget/SNR; run the full pass-budget (40) + population-coded error for the close.", flush=True)
    elif ms >= 0.7:
        print(f"  PARTIAL: sign mostly right ({ms:.2f}) but magnitude noisy (corr {mc:.2f}) -> population-code the "
              f"error read (N neurons/output averaged) + full budget.", flush=True)
    else:
        print(f"  DISTORTED: the neural error does NOT track the host error (corr {mc:.2f}, sign {ms:.2f}) -> the "
              f"LIF f-I band/calibration is the problem; fix the error read (drive/window/threshold) before convergence.",
              flush=True)


if __name__ == "__main__":
    main()
