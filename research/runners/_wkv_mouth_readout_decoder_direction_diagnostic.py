"""#80 / mouth read-SNR — DECODER-DIRECTION diagnostic (why is the LEARNED read direction bad?).

The prior rate-vs-bias finding (2026-08-27-mouth-readsnr-rate-vs-bias-H2-bias-limited.md) proved the wall is
BIAS/DIRECTION-limited: the learned W_hat has weight_cosine ~0.135 to the copied head, and more spikes / less noise on
a FIXED decoder do nothing. The 6-seed JSONs show the substrate-forward-learned W is a poor classifier in ANY linear
channel (hostlinear_recov ~0.37 == sub_learned ~0.37), even though the substrate read is LINEARLY FAITHFUL
(gain_substrate_vs_linear_corr ~0.945) AND the substrate-forward gradient barely reduces held-frame CE (10.748->10.746).

THIS DIAGNOSTIC isolates WHY, by running the IDENTICAL softmax-onehot delta rule under three forwards at matched
init / data-order / hypers, and recording the weight_cosine (to head_w) + hostlinear_recov trajectory every epoch:
  - host_proxy   : logits = W@h + head_b            (EXACT map, zero read noise -- the CEILING of this rule)
  - substrate    : logits = margin_sub/gain + head_b (K=1 substrate read/step -- the CURRENT production rule)
  - substrate_avgK: same, but margin_sub AVERAGED over K substrate reads/step (noise down ~sqrt(K), systematic bias kept)

The decisive contrast: if substrate_avgK's wcos climbs toward host_proxy's, the bad direction is READ NOISE (fix =
more samples / eligibility integration). If substrate_avgK stays at the ~0.13 plateau, it is a SYSTEMATIC read BIAS
the gradient cannot correct (fix = de-bias / whiten the read, or a per-unit teacher). Also reports the per-read noise
CV and the single-vs-averaged margin correlation, to quantify the noise magnitude directly.

Additive, research-runner only, NO sim/ edit. Reuse-by-import of BatchedSubstrateReadout / _calibrate_gain / _wcos /
_softmax_rows / _sub_logits and _positions / _eval_hostlinear / _load_eval (all UNMODIFIED). CLAUDE.md seed-trap:
cfg.seed is set inside BatchedSubstrateReadout._build_bridge; a build-twice thr hash is emitted.

Run (numpy, 1 seed, small):
  .venv/bin/python -m research.runners._wkv_mouth_readout_decoder_direction_diagnostic \
      --seed 42 --B 8 --n-train-pos 240 --epochs 8 --sub-read-window 64 --avg-k 4 \
      --json research/findings/raw/_wkv_readout_decoder_direction/diag_s42.json
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
from research.runners._wkv_mouth_readout_eprop_learn_derisk import (  # noqa: E402
    _positions, _eval_hostlinear,
)
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _calibrate_gain, _sub_logits, _softmax_rows, _wcos,
)


def _held_ce(logits, Y, unk):
    P = _softmax_rows(logits)
    return float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1.0)).mean())


def _substrate_margin_avg(s_batch, Hb, K):
    """K independent substrate reads of the same batch, averaged (noise down ~sqrt(K), systematic bias unchanged)."""
    acc = None
    for _ in range(K):
        m = s_batch.batch_margin(Hb, silence_bias=True)
        acc = m if acc is None else acc + m
    return acc / K


def _learn(arm, seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, args):
    """The IDENTICAL softmax-onehot delta rule under `arm` in {host_proxy, substrate, substrate_avgK}. Records the
    wcos + hostlinear_recov + held-frame CE trajectory each epoch. Same init / data order / hypers across arms."""
    V, D = ro.V, ro.D
    B = s_batch.B
    unk = ro.unk_idx
    rng = np.random.default_rng(seed * 991 + 7)               # SAME init seed as the production rule
    W = 0.01 * rng.standard_normal((V, D))
    idx = np.arange(len(H))
    n_full = (len(idx) // B) * B
    traj = []
    t0 = time.time()
    n_reads = 0
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for start in range(0, n_full, B):
            bi = idx[start:start + B]
            Hb = H[bi]
            if arm == "host_proxy":
                logits = Hb @ W.T + head_b[None, :]
                if unk >= 0:
                    logits = logits.copy(); logits[:, unk] = -1e30
            else:
                s_batch.set_weights(W)
                if arm == "substrate_avgK":
                    margin_sub = _substrate_margin_avg(s_batch, Hb, args.avg_k); n_reads += args.avg_k
                else:
                    margin_sub = s_batch.batch_margin(Hb, silence_bias=True); n_reads += 1
                logits = _sub_logits(margin_sub, gain, head_b, unk)
            P = _softmax_rows(logits)
            P[np.arange(B), Y[bi]] -= 1.0
            W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
            if args.w_target > 0:
                nrm = float(np.linalg.norm(W))
                if nrm > args.w_target:
                    W *= args.w_target / nrm
        # per-epoch eval (host-linear recov + wcos are cheap matmuls; held CE on the eval set)
        rr = _eval_hostlinear(ro, W, He, Ye, PFe)
        wc = _wcos(W, hw)
        ce = _held_ce(He @ W.T + head_b[None, :], Ye, unk)
        traj.append({"epoch": ep + 1, "hostlinear_recov": round(float(rr["recov_argmax"]), 4),
                     "weight_cosine": wc, "held_ce": round(ce, 4)})
        print(f"[{arm} seed {seed} ep {ep+1}/{args.epochs}] wcos={wc} hostlin_recov={rr['recov_argmax']:.4f} "
              f"ce={ce:.4f} reads={n_reads}", flush=True)
    return dict(arm=arm, final_wcos=traj[-1]["weight_cosine"], final_hostlinear_recov=traj[-1]["hostlinear_recov"],
                final_ce=traj[-1]["held_ce"], trajectory=traj, n_substrate_reads=n_reads,
                secs=round(time.time() - t0, 1), w_norm=round(float(np.linalg.norm(W)), 2))


def _read_noise(s_batch, Hb, gain, K=6):
    """Directly quantify the per-read noise: K reads of ONE fixed batch/weights; report the mean coefficient-of-
    variation across (B,V) entries, and corr(single read, K-averaged read). High CV / low corr == a noisy instrument."""
    reads = np.stack([s_batch.batch_margin(Hb, silence_bias=True) for _ in range(K)], axis=0)  # [K,B,V]
    mu = reads.mean(0); sd = reads.std(0)
    mask = np.abs(mu) > (0.05 * np.abs(mu).mean() + 1e-9)
    cv = float(np.mean(sd[mask] / (np.abs(mu[mask]) + 1e-12))) if mask.any() else float("nan")
    single = reads[0].reshape(-1); avg = mu.reshape(-1)
    corr = float(np.corrcoef(single, avg)[0, 1]) if single.std() > 1e-12 else 0.0
    return round(cv, 4), round(corr, 4)


def _thr_hash(seed, ro, args):
    s = BatchedSubstrateReadout(ro, seed, 4, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    del s
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n-train-pos", type=int, default=240)
    ap.add_argument("--n-eval-pos", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--avg-k", type=int, default=4)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arms", type=str, default="host_proxy,substrate,substrate_avgK")
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-read-window", type=int, default=64)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_readout_decoder_direction/diag.json")
    args = ap.parse_args()

    seed = args.seed
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    ro = WKVReadout(ckpt)

    h1 = _thr_hash(seed, ro, args); h2 = _thr_hash(seed, ro, args)
    print(f"[seed-trap] thr hash {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)
    assert len(H) >= args.B and len(He) > 0, "insufficient positions"

    head_b = ro.head_b.astype(np.float64)
    hw = ro.head_w

    s_batch = BatchedSubstrateReadout(ro, seed, args.B, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                      read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                      settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.B], seed)
    print(f"[calib-gain seed {seed}] gain={gain:.5g} substrate-vs-linear corr={gain_corr}", flush=True)

    cv, single_vs_avg_corr = _read_noise(s_batch, H[:args.B], gain, K=6)
    print(f"[read-noise seed {seed}] per-read CV={cv} corr(single, 6-avg)={single_vs_avg_corr}", flush=True)

    arms = [a for a in args.arms.split(",") if a.strip()]
    results = {}
    for arm in arms:
        results[arm] = _learn(arm, seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, args)

    # ATTRIBUTION: whose is the wcos difference? Average-K reads (treatment) vs the single substrate read (control),
    # and the substrate read (control) vs the exact-map host_proxy (ceiling). Measuring three arms is not attribution.
    if "substrate" in results and "substrate_avgK" in results:
        lever(f"avgK_vs_single_substrate_read_wcos_seed{seed}",
              before=results["substrate"]["final_wcos"], after=results["substrate_avgK"]["final_wcos"],
              required=False, continuous=round(results["substrate_avgK"]["final_hostlinear_recov"]
                                              - results["substrate"]["final_hostlinear_recov"], 4))
    if "substrate" in results and "host_proxy" in results:
        lever(f"substrate_vs_hostproxy_ceiling_wcos_seed{seed}",
              before=results["substrate"]["final_wcos"], after=results["host_proxy"]["final_wcos"], required=False)

    out = {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.B, "n_train_pos": len(H), "n_eval_pos": len(He),
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay, "w_target": args.w_target,
        "avg_k": args.avg_k, "sub_read_window": args.sub_read_window, "gain": round(gain, 6),
        "gain_substrate_vs_linear_corr": gain_corr, "per_read_cv": cv, "single_vs_6avg_corr": single_vs_avg_corr,
        "head_w_norm": round(float(np.linalg.norm(hw)), 2),
        "seed_trap": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)},
        "arms": _native(results), "backend": os.environ.get("SIM_BACKEND", "numpy"), "argv": sys.argv,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print("\n[FINAL] arm : final_wcos / final_hostlinear_recov / final_ce", flush=True)
    for arm in arms:
        r = results[arm]
        print(f"  {arm:16s}: wcos={r['final_wcos']}  hostlin={r['final_hostlinear_recov']}  ce={r['final_ce']}"
              f"  ({r['n_substrate_reads']} reads, {r['secs']}s)", flush=True)
    print(f"[done] -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
