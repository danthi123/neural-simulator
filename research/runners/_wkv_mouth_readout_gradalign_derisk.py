"""#80 / mouth read-SNR — candidate C SCREEN via GRADIENT ALIGNMENT (does per-word calibration fix the DIRECTION?).

The reduced-CPU endpoint is scale-capped: at 160 train positions ALL arms sit at weight_cosine ~0.08 (host_proxy 0.083
== substrate 0.078), because the known host_proxy(0.51)-vs-substrate(0.135) direction gap only opens at production data
scale (GPU-bound). So a cheap CPU *endpoint* cannot resolve a fix's lift. The DECISIVE cheap instrument is the
GRADIENT: a delta-rule's fixed point is set by its gradient direction, so if the per-word-calibrated substrate gradient
matches the host-proxy gradient (whose endpoint wcos ~0.51 is measured), candidate C reproduces that endpoint.

At representative weights W (init, and alpha*head_w for alpha in {0.25,0.5,1.0}), over n_batches training batches, this
computes the softmax-onehot gradient g=(softmax(logits)-onehot)^T @ h / B under three reads of the SAME positions:
  g_host  : logits = W@h + head_b                 (EXACT map -- the ceiling rule; its endpoint wcos ~0.51 is known)
  g_sub   : logits = margin_sub/GAIN + head_b      (GLOBAL gain -- the production rule == the LESION)
  g_calib : logits = (margin_sub - c)/a + head_b   (PER-WORD calibration -- candidate C)
and reports cos(g_sub,g_host), cos(g_calib,g_host), and the spurious-force ratio ||g||/||g_host|| at W=head_w (near the
host optimum g_host is small; a large ||g_sub|| there is the systematic force that PUSHES the learned W away from
head_w -- the direct cause of the low endpoint wcos). BYTE-IDENTICAL-OFF is proven inline: with a:=gain, c:=0 the calib
read reduces to the lesion read, so g_calib==g_sub (cos 1.0). host_matmul on the two substrate reads is 0.

GO (mechanism, per seed): mean cos(g_calib,g_host) >= 0.9 AND [mean cos(g_calib,g_host) - mean cos(g_sub,g_host)] >=
0.15 AND at W=head_w ||g_calib||/||g_host|| < ||g_sub||/||g_host|| (spurious force reduced). Board GO = >=5/6. A NO-GO
banks candidate C and names the next lever (FORCE/RLS recurrent decoder; or a direction-optimising credit rule).

Additive, research-runner only, NO sim/ edit. Reuse-by-import of _calibrate_perword (candidate C's calibration),
_calibrate_gain, BatchedSubstrateReadout, _softmax_rows and _positions / _load_eval / WKVReadout (all UNMODIFIED).

Run (numpy, blocking, 3 seeds/call):
  .venv/bin/python -m research.runners._wkv_mouth_readout_gradalign_derisk --seeds 42,43,44 \
      --B 8 --n-batches 10 --sub-read-window 64 --n-probes 24 \
      --json research/findings/raw/_wkv_readout_decoder_direction/gradalign_s424344.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sim.backend import to_host  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native, _load_eval  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _calibrate_gain, _softmax_rows,
)
from research.runners._wkv_mouth_readout_perword_calib_derisk import _calibrate_perword  # noqa: E402


def _grad(logits, Hb, Yb, unk):
    P = _softmax_rows(logits)
    P[np.arange(len(Yb)), Yb] -= 1.0
    return (P.T @ Hb) / len(Yb)                # [V, D]


def _cos(A, B):
    a = A.reshape(-1); b = B.reshape(-1)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return round(float(a @ b) / d, 4) if d > 1e-12 else 0.0


def _thr_hash(seed, ro, args):
    s = BatchedSubstrateReadout(ro, seed, 4, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    del s
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def run_seed(seed, args):
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    if not Path(ckpt).exists():
        print(f"[skip] seed {seed}: {ckpt} missing", flush=True); return None
    ro = WKVReadout(ckpt)
    unk = ro.unk_idx
    h1 = _thr_hash(seed, ro, args); h2 = _thr_hash(seed, ro, args)
    print(f"[seed-trap seed {seed}] thr {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    H, Y, _ = _positions(ro, usable, args.warmup, args.B * args.n_batches)
    assert len(H) >= args.B, "insufficient positions"
    head_b = ro.head_b.astype(np.float64); hw = ro.head_w.astype(np.float64)

    s_batch = BatchedSubstrateReadout(ro, seed, args.B, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                      read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                      settle_frac=args.settle_frac, n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA)
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.B], seed)
    a, c, frac_fb, n_cal = _calibrate_perword(s_batch, ro, H[:args.B], seed, gain, args.n_probes)
    print(f"[calib seed {seed}] gain={gain:.4g} corr={gain_corr} perword a[med={np.median(a):.4g}] "
          f"frac_fallback={frac_fb:.3f}", flush=True)

    Wt = hw * (args.w_target / (np.linalg.norm(hw) + 1e-12))            # head_w at production ||W||
    rng = np.random.default_rng(seed * 991 + 7)
    Wset = {"init": 0.01 * rng.standard_normal((ro.V, ro.D)),
            "0.25xheadw": 0.25 * Wt, "0.5xheadw": 0.5 * Wt, "headw": Wt}

    nb = len(H) // args.B
    per_W = {}
    byte_ident_cos = None
    for name, W in Wset.items():
        g_host = np.zeros((ro.V, ro.D)); g_sub = np.zeros_like(g_host); g_calib = np.zeros_like(g_host)
        g_lesion = np.zeros_like(g_host)
        for bidx in range(nb):
            sl = slice(bidx * args.B, (bidx + 1) * args.B)
            Hb = H[sl]; Yb = Y[sl]
            lg_host = Hb @ W.T + head_b[None, :]
            if unk >= 0:
                lg_host = lg_host.copy(); lg_host[:, unk] = -1e30
            g_host += _grad(lg_host, Hb, Yb, unk)
            s_batch.set_weights(W)
            msub = s_batch.batch_margin(Hb, silence_bias=True)         # SUBSTRATE read (0 host matmul)
            lg_sub = msub / gain + head_b[None, :]
            lg_cal = (msub - c[None, :]) / a[None, :] + head_b[None, :]
            lg_les = (msub - 0.0) / gain + head_b[None, :]             # a:=gain,c:=0 -> == lesion (byte-ident check)
            for lg in (lg_sub, lg_cal, lg_les):
                if unk >= 0:
                    lg[:, unk] = -1e30
            g_sub += _grad(lg_sub, Hb, Yb, unk)
            g_calib += _grad(lg_cal, Hb, Yb, unk)
            g_lesion += _grad(lg_les, Hb, Yb, unk)
        nh = float(np.linalg.norm(g_host))
        per_W[name] = {
            "cos_sub_host": _cos(g_sub, g_host), "cos_calib_host": _cos(g_calib, g_host),
            "gnorm_ratio_sub": round(float(np.linalg.norm(g_sub)) / max(1e-12, nh), 3),
            "gnorm_ratio_calib": round(float(np.linalg.norm(g_calib)) / max(1e-12, nh), 3),
        }
        if name == "init":
            byte_ident_cos = _cos(g_lesion, g_sub)                     # must be ~1.0 (a:=gain,c:=0 reproduces lesion)
        print(f"  [W={name:11s} seed {seed}] cos(sub,host)={per_W[name]['cos_sub_host']} "
              f"cos(calib,host)={per_W[name]['cos_calib_host']} "
              f"||g||/||g_host||: sub={per_W[name]['gnorm_ratio_sub']} calib={per_W[name]['gnorm_ratio_calib']}",
              flush=True)

    cos_sub_mean = round(float(np.mean([per_W[k]["cos_sub_host"] for k in Wset])), 4)
    cos_cal_mean = round(float(np.mean([per_W[k]["cos_calib_host"] for k in Wset])), 4)
    spur_sub = per_W["headw"]["gnorm_ratio_sub"]; spur_cal = per_W["headw"]["gnorm_ratio_calib"]
    m = {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.B, "n_batches": nb, "sub_read_window": args.sub_read_window,
        "gain": round(gain, 5), "gain_substrate_vs_linear_corr": gain_corr, "perword_frac_fallback": frac_fb,
        "cos_sub_host_mean": cos_sub_mean, "cos_calib_host_mean": cos_cal_mean,
        "cos_realign_gain": round(cos_cal_mean - cos_sub_mean, 4),
        "spurious_force_at_headw_sub": spur_sub, "spurious_force_at_headw_calib": spur_cal,
        "byte_identical_off_cos_lesion_vs_sub": byte_ident_cos, "per_W": per_W,
        "seed_trap": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)},
    }
    m["go"] = bool(cos_cal_mean >= 0.9 and (cos_cal_mean - cos_sub_mean) >= 0.15 and spur_cal < spur_sub
                   and byte_ident_cos is not None and byte_ident_cos >= 0.999)
    print(f"[seed {seed}] cos(calib,host)_mean={cos_cal_mean} vs cos(sub,host)_mean={cos_sub_mean} "
          f"(realign +{m['cos_realign_gain']}) | spurious@headw sub={spur_sub} calib={spur_cal} | "
          f"byte_ident_cos={byte_ident_cos} | GO={m['go']}", flush=True)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--n-probes", type=int, default=24)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-read-window", type=int, default=64)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_readout_decoder_direction/gradalign.json")
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
            "n_seeds": len(rows), "go_count": int(sum(1 for r in rows if r["go"])),
            "go_5of6": bool(sum(1 for r in rows if r["go"]) >= 5),
            "cos_calib_host_mean": round(float(np.mean([r["cos_calib_host_mean"] for r in rows])), 4),
            "cos_sub_host_mean": round(float(np.mean([r["cos_sub_host_mean"] for r in rows])), 4),
            "cos_realign_gain_mean": round(float(np.mean([r["cos_realign_gain"] for r in rows])), 4),
            "spurious_force_at_headw_sub_mean": round(float(np.mean([r["spurious_force_at_headw_sub"] for r in rows])), 3),
            "spurious_force_at_headw_calib_mean": round(float(np.mean([r["spurious_force_at_headw_calib"] for r in rows])), 3),
            "byte_identical_off_all": bool(all(r["byte_identical_off_cos_lesion_vs_sub"] >= 0.999 for r in rows)),
        }
    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
