"""#80 / mouth read-SNR — ALIGNED-vs-OBLIQUE regime sweep (Schuessler et al. 2023, eLife).

Schuessler et al. 2023 (https://elifesciences.org/articles/93060) show a linear-readout network trains into one of
two regimes -- ALIGNED (internal dynamics lie along the readout directions; high readout weight_cosine) or OBLIQUE
(dynamics oblique to the readout; low weight_cosine) -- and the control knob is the READOUT WEIGHT MAGNITUDE (small ->
oblique, large -> aligned). Our learned decoder's weight_cosine ~0.13 IS the oblique regime, and the decoder-direction
finding showed the substrate READ is fine (its gradient is cos ~0.99 to the ideal map) -- the "read is fine, learned
direction wrong" signature Schuessler predicts. This sweeps the magnitude knob to test whether the ALIGNED regime is
reachable, and whether the substrate's linear-read cap (w_target=40, which holds ||W|| in the graded read's linear
range) is what PINS us to oblique.

Two knobs, two arms (matched init/data/hypers; the softmax-onehot delta rule, forward = substrate read or exact map):
  A) INITIAL magnitude: W init = init_scale x 0.01 x randn, sweep init_scale in {0.3,1,3,10}, w_target=40 (production).
     Run on BOTH substrate (production-relevant, bounded by the 40 cap) and host_proxy (free reference).
  B) SUSTAINED magnitude: host_proxy ONLY (no read saturation), sweep w_target in {40,120,400,1200} at init_scale=1 --
     tests whether a LARGER sustained readout reaches the aligned regime when the read is not the constraint. The
     substrate arm cannot follow this (the graded read saturates for ||W||>>40; the runner's own synaptic-scaling
     comment), so a host_proxy-only aligned lift here localises the block to the read's linear range.

Reports endpoint weight_cosine (to head_w) + hostlinear_recov vs magnitude, per arm. GO signal = weight_cosine RISES
monotonically toward the aligned regime as magnitude grows. Attribution via tools.lab.lever (wcos at 10x vs 1x).
Additive, research-runner only, NO sim/ edit. Reuse-by-import of BatchedSubstrateReadout / _calibrate_gain /
_softmax_rows / _sub_logits / _wcos and _positions / _eval_hostlinear / _load_eval (UNMODIFIED). Seed-trap emitted.

Run (numpy, blocking, 1 seed/call):
  .venv/bin/python -m research.runners._wkv_mouth_readout_init_scale_sweep_derisk --seeds 42 \
      --B 8 --n-train-pos 128 --epochs 4 --sub-read-window 64 \
      --json research/findings/raw/_wkv_readout_decoder_direction/initscale_s42.json
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

from tools.lab import lever  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import WKVReadout, _native, _load_eval  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions, _eval_hostlinear  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _calibrate_gain, _softmax_rows, _sub_logits, _wcos,
)


def _train(arm, seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, args, init_scale, w_target):
    V, D = ro.V, ro.D
    B = s_batch.B
    unk = ro.unk_idx
    rng = np.random.default_rng(seed * 991 + 7)                 # SAME base draw; init_scale multiplies it
    W = (init_scale * 0.01) * rng.standard_normal((V, D))
    init_norm = float(np.linalg.norm(W))
    idx = np.arange(len(H)); n_full = (len(idx) // B) * B
    n_reads = 0; t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for start in range(0, n_full, B):
            bi = idx[start:start + B]; Hb = H[bi]
            if arm == "host_proxy":
                lg = Hb @ W.T + head_b[None, :]
                if unk >= 0:
                    lg = lg.copy(); lg[:, unk] = -1e30
            else:
                s_batch.set_weights(W)
                msub = s_batch.batch_margin(Hb, silence_bias=True); n_reads += 1
                lg = _sub_logits(msub, gain, head_b, unk)
            P = _softmax_rows(lg)
            P[np.arange(B), Y[bi]] -= 1.0
            W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
            if w_target > 0:
                nrm = float(np.linalg.norm(W))
                if nrm > w_target:
                    W *= w_target / nrm
    rr = _eval_hostlinear(ro, W, He, Ye, PFe)
    return dict(arm=arm, init_scale=init_scale, w_target=w_target, init_norm=round(init_norm, 3),
                final_norm=round(float(np.linalg.norm(W)), 2), weight_cosine=_wcos(W, hw),
                hostlinear_recov=round(float(rr["recov_argmax"]), 4), n_substrate_reads=n_reads,
                secs=round(time.time() - t0, 1))


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
    h1 = _thr_hash(seed, ro, args); h2 = _thr_hash(seed, ro, args)
    print(f"[seed-trap seed {seed}] thr {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)
    assert len(H) >= args.B and len(He) > 0, "insufficient positions"
    head_b = ro.head_b.astype(np.float64); hw = ro.head_w.astype(np.float64)

    s_batch = BatchedSubstrateReadout(ro, seed, args.B, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                      read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                      settle_frac=args.settle_frac, n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA)
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.B], seed)
    print(f"[calib seed {seed}] gain={gain:.4g} corr={gain_corr} head_w_norm={np.linalg.norm(hw):.2f}", flush=True)

    init_scales = [float(x) for x in args.init_scales.split(",")]
    wt_sweep = [float(x) for x in args.wtarget_sweep.split(",")]

    # -- A) INITIAL magnitude sweep, both arms at production w_target --
    A = {"substrate": [], "host_proxy": []}
    for sc in init_scales:
        for arm in ("substrate", "host_proxy"):
            r = _train(arm, seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, args, sc, args.w_target)
            A[arm].append(r)
            print(f"  [A {arm:10s} init_scale={sc:<4g} seed {seed}] init||W||={r['init_norm']} "
                  f"final||W||={r['final_norm']} wcos={r['weight_cosine']} hostlin={r['hostlinear_recov']}", flush=True)

    # -- B) SUSTAINED magnitude sweep, host_proxy ONLY (no read saturation) --
    Bsw = []
    for wt in wt_sweep:
        r = _train("host_proxy", seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, args, 1.0, wt)
        Bsw.append(r)
        print(f"  [B host_proxy w_target={wt:<5g} seed {seed}] final||W||={r['final_norm']} "
              f"wcos={r['weight_cosine']} hostlin={r['hostlinear_recov']}", flush=True)

    def wc(lst, key, val):
        for r in lst:
            if r[key] == val:
                return r["weight_cosine"]
        return None
    sub_1x = wc(A["substrate"], "init_scale", 1.0); sub_10x = wc(A["substrate"], "init_scale", 10.0)
    hp_1x = wc(A["host_proxy"], "init_scale", 1.0); hp_10x = wc(A["host_proxy"], "init_scale", 10.0)
    hp_wt40 = wc(Bsw, "w_target", 40.0); hp_wtmax = Bsw[-1]["weight_cosine"] if Bsw else None
    # ATTRIBUTION: whose is the wcos change -- initial magnitude (treatment) vs the 1x baseline (control)?
    lever(f"substrate_wcos_init10x_vs_1x_seed{seed}", before=sub_1x, after=sub_10x, required=False)
    lever(f"hostproxy_wcos_wtargetmax_vs_40_seed{seed}", before=hp_wt40, after=hp_wtmax, required=False)

    m = {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.B, "n_train_pos": len(H), "epochs": args.epochs,
        "sub_read_window": args.sub_read_window, "gain": round(gain, 5), "head_w_norm": round(float(np.linalg.norm(hw)), 2),
        "A_init_scale_sweep": {"substrate": A["substrate"], "host_proxy": A["host_proxy"]},
        "B_hostproxy_wtarget_sweep": Bsw,
        "substrate_wcos_1x": sub_1x, "substrate_wcos_10x": sub_10x,
        "substrate_wcos_10x_over_1x": round(sub_10x / max(1e-9, sub_1x), 3) if (sub_1x and sub_10x) else None,
        "hostproxy_wcos_1x": hp_1x, "hostproxy_wcos_10x": hp_10x,
        "hostproxy_wcos_wt40": hp_wt40, "hostproxy_wcos_wtmax": hp_wtmax,
        "hostproxy_wcos_wtmax_over_wt40": round(hp_wtmax / max(1e-9, hp_wt40), 3) if (hp_wt40 and hp_wtmax) else None,
        "seed_trap": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)},
    }
    # SIGNAL: aligned regime reachable if wcos rises with magnitude. substrate GO = 10x wcos >= 1.3x its 1x wcos.
    m["substrate_aligned_signal"] = bool(sub_1x and sub_10x and sub_10x >= 1.3 * sub_1x)
    m["hostproxy_aligned_signal"] = bool(hp_wt40 and hp_wtmax and hp_wtmax >= 1.3 * hp_wt40)
    print(f"[seed {seed}] SUBSTRATE wcos 1x->10x: {sub_1x}->{sub_10x} (aligned={m['substrate_aligned_signal']}) | "
          f"HOST_PROXY wcos wt40->wtmax: {hp_wt40}->{hp_wtmax} (aligned={m['hostproxy_aligned_signal']})", flush=True)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n-train-pos", type=int, default=128)
    ap.add_argument("--n-eval-pos", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--init-scales", type=str, default="0.3,1,3,10")
    ap.add_argument("--wtarget-sweep", type=str, default="40,120,400,1200")
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-read-window", type=int, default=64)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_readout_decoder_direction/initscale.json")
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
            "n_seeds": len(rows),
            "substrate_aligned_signal_count": int(sum(1 for r in rows if r["substrate_aligned_signal"])),
            "hostproxy_aligned_signal_count": int(sum(1 for r in rows if r["hostproxy_aligned_signal"])),
            "substrate_wcos_1x_mean": round(float(np.mean([r["substrate_wcos_1x"] for r in rows])), 4),
            "substrate_wcos_10x_mean": round(float(np.mean([r["substrate_wcos_10x"] for r in rows])), 4),
            "hostproxy_wcos_wt40_mean": round(float(np.mean([r["hostproxy_wcos_wt40"] for r in rows])), 4),
            "hostproxy_wcos_wtmax_mean": round(float(np.mean([r["hostproxy_wcos_wtmax"] for r in rows])), 4),
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
