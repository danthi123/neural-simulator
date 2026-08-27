"""#80 / mouth read-SNR — host_proxy (ideal-map) MAGNITUDE sweep at FULL data scale (Schuessler aligned/oblique test).

The reduced-budget substrate sweep could not test Schuessler et al. 2023's aligned/oblique magnitude knob
(https://elifesciences.org/articles/93060): the regime is an ASYMPTOTIC (converged, large-||W||) property, but at a
reduced budget ||W|| never grew to the w_target cap, so unclamping did nothing and a large random init only hurt.

The host-proxy forward (`logits = W@h + head_b`, the EXACT map the decoder-direction finding proved the substrate read
reproduces to gradient cos ~0.99) is MATMUL-ONLY -- no substrate reads -- so it trains to convergence at the FULL
data scale in seconds. This is the clean OBJECTIVE-level test of the magnitude knob: sweep the sustained readout
magnitude (w_target cap; 0 = uncapped) and the initial magnitude, at production data volume, and measure whether the
learned readout weight_cosine rises toward the ALIGNED regime as magnitude grows. If uncapped host_proxy reaches a high
wcos, the aligned regime IS reachable by magnitude -- and the substrate's w_target=40 cap (required to hold ||W|| in the
graded read's linear range) is then the thing pinning the real system to oblique (-> next lever = a read faithful at
large ||W||). If it STAYS oblique, magnitude is not the knob here and the objective/data is the limit.

Additive, research-runner only, NO sim/ edit, NO bridge (pure numpy matmul). Reuse-by-import of WKVReadout / _native /
_load_eval / _positions / _eval_hostlinear / _wcos (all UNMODIFIED). Attribution via tools.lab.lever.

Run (numpy, blocking, full scale, 6 seeds):
  .venv/bin/python -m research.runners._wkv_mouth_readout_hostproxy_magnitude_sweep --seeds 42,43,44,100,101,102 \
      --n-train-pos 9600 --epochs 8 --json research/findings/raw/_wkv_readout_decoder_direction/hostproxy_mag_6seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from tools.lab import lever  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native, _load_eval  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions, _eval_hostlinear  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import _softmax_rows, _wcos  # noqa: E402


def _train_hostproxy(seed, ro, H, Y, He, Ye, PFe, hw, head_b, args, init_scale, w_target):
    V, D = ro.V, ro.D
    unk = ro.unk_idx; B = args.B
    rng = np.random.default_rng(seed * 991 + 7)
    W = (init_scale * 0.01) * rng.standard_normal((V, D))
    init_norm = float(np.linalg.norm(W))
    idx = np.arange(len(H)); n_full = (len(idx) // B) * B
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for start in range(0, n_full, B):
            bi = idx[start:start + B]; Hb = H[bi]
            lg = Hb @ W.T + head_b[None, :]
            if unk >= 0:
                lg = lg.copy(); lg[:, unk] = -1e30
            P = _softmax_rows(lg)
            P[np.arange(B), Y[bi]] -= 1.0
            W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
            if w_target > 0:
                nrm = float(np.linalg.norm(W))
                if nrm > w_target:
                    W *= w_target / nrm
    rr = _eval_hostlinear(ro, W, He, Ye, PFe)
    return dict(init_scale=init_scale, w_target=w_target, init_norm=round(init_norm, 3),
                final_norm=round(float(np.linalg.norm(W)), 2), weight_cosine=_wcos(W, hw),
                hostlinear_recov=round(float(rr["recov_argmax"]), 4))


def run_seed(seed, args):
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    if not Path(ckpt).exists():
        print(f"[skip] seed {seed}: {ckpt} missing", flush=True); return None
    ro = WKVReadout(ckpt)
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)
    assert len(H) >= args.B and len(He) > 0, "insufficient positions"
    head_b = ro.head_b.astype(np.float64); hw = ro.head_w.astype(np.float64)
    print(f"[seed {seed}] n_train={len(H)} n_eval={len(He)} head_w_norm={np.linalg.norm(hw):.2f}", flush=True)

    wts = [float(x) for x in args.wtarget_sweep.split(",")]      # sustained-magnitude sweep (0 = uncapped)
    iss = [float(x) for x in args.init_scales.split(",")]        # initial-magnitude sweep at production w_target

    wt_rows = []
    for wt in wts:
        r = _train_hostproxy(seed, ro, H, Y, He, Ye, PFe, hw, head_b, args, 1.0, wt)
        wt_rows.append(r)
        print(f"  [w_target={wt:<6g} seed {seed}] final||W||={r['final_norm']} wcos={r['weight_cosine']} "
              f"hostlin={r['hostlinear_recov']}", flush=True)
    is_rows = []
    for sc in iss:
        r = _train_hostproxy(seed, ro, H, Y, He, Ye, PFe, hw, head_b, args, sc, args.w_target)
        is_rows.append(r)
        print(f"  [init_scale={sc:<4g} seed {seed}] init||W||={r['init_norm']} final||W||={r['final_norm']} "
              f"wcos={r['weight_cosine']} hostlin={r['hostlinear_recov']}", flush=True)

    def wc(rows, key, val):
        for r in rows:
            if r[key] == val:
                return r["weight_cosine"]
        return None
    wcos_wt40 = wc(wt_rows, "w_target", 40.0)
    wcos_uncapped = wc(wt_rows, "w_target", 0.0)
    if wcos_uncapped is None:
        wcos_uncapped = wt_rows[-1]["weight_cosine"]
    lever(f"hostproxy_wcos_uncapped_vs_wt40_seed{seed}", before=wcos_wt40, after=wcos_uncapped, required=False)

    m = {"seed": seed, "V": ro.V, "D": ro.D, "n_train_pos": len(H), "epochs": args.epochs,
         "wtarget_sweep": wt_rows, "init_scale_sweep": is_rows,
         "wcos_wt40": wcos_wt40, "wcos_uncapped": wcos_uncapped,
         "wcos_uncapped_over_wt40": round(wcos_uncapped / max(1e-9, wcos_wt40), 3) if (wcos_wt40 and wcos_uncapped) else None}
    m["aligned_signal"] = bool(wcos_wt40 and wcos_uncapped and wcos_uncapped >= 1.3 * wcos_wt40)
    print(f"[seed {seed}] host_proxy wcos wt40={wcos_wt40} uncapped={wcos_uncapped} "
          f"(aligned_signal={m['aligned_signal']})", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=80000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--n-train-pos", type=int, default=9600)
    ap.add_argument("--n-eval-pos", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--wtarget-sweep", type=str, default="40,120,400,1200,0")   # 0 = uncapped
    ap.add_argument("--init-scales", type=str, default="0.3,1,3,10")
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_readout_decoder_direction/hostproxy_mag.json")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    t0 = time.time(); rows = []
    for seed in seeds:
        m = run_seed(seed, args)
        if m is not None:
            rows.append(m)
    summary = {}
    if rows:
        summary = {
            "n_seeds": len(rows), "aligned_signal_count": int(sum(1 for r in rows if r["aligned_signal"])),
            "wcos_wt40_mean": round(float(np.mean([r["wcos_wt40"] for r in rows])), 4),
            "wcos_uncapped_mean": round(float(np.mean([r["wcos_uncapped"] for r in rows])), 4),
            "wcos_uncapped_over_wt40_mean": round(float(np.mean([r["wcos_uncapped_over_wt40"] for r in rows])), 3),
        }
    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "external_source": "Schuessler et al. 2023 eLife https://elifesciences.org/articles/93060 (aligned/oblique)",
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
