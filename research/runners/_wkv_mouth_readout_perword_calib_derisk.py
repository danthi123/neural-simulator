"""#80 / mouth read-SNR — candidate C: PER-WORD (per-readout-neuron) affine read calibration.

DIAGNOSIS (2026-08-27 decoder-direction): the substrate-forward-learned decoder has weight_cosine ~0.135 to the copied
head while a matched host-linear-proxy forward (SAME rule, EXACT W@h+head_b) reaches ~0.5. The substrate read is
linearly faithful in the POOLED sense (gain_substrate_vs_linear_corr ~0.945) and the per-read NOISE is negligible
(CV ~0.076, single-vs-6avg corr 0.9976), so the degradation is a SYSTEMATIC, per-word read distortion the single
GLOBAL conductance->logit gain scalar does not remove -- a per-readout homeostatic gain/offset the production rule
replaced with one constant.

THE FIX (brain-based, additive, default-OFF): give each readout word-pool its OWN affine read calibration
(slope a[v], offset c[v]) -- a per-postsynaptic-neuron intrinsic gain + threshold homeostasis (Turrigiano synaptic
scaling / intrinsic-excitability homeostasis is per-neuron, not one global scalar). a[v], c[v] are measured ONCE per
seed with RANDOM probe weights (INDEPENDENT of head_w -> no target leak, no weight transport), exactly the calibration
class the accepted global `gain` already uses. The learning forward then reads the substrate (0 host matmul on the
read) and applies the per-word inverse: corrected = (margin_sub - c)/a  ~=  W@h, so the honest softmax-onehot gradient
is no longer steered by the read's per-word distortion.

Arms (matched init / data-order / hypers; the ONLY difference is the read map applied to the substrate margin):
  - host_proxy    : logits = W@h + head_b            (EXACT map -- the rule's CEILING; forward_is_substrate=False)
  - substrate     : logits = margin_sub/GAIN + head_b (GLOBAL-gain -- the CURRENT production rule == the LESION of C)
  - substrate_calib: logits = (margin_sub - c)/a + head_b (PER-WORD calibration -- candidate C; host_matmul_forward=0)
  - substrate_calib_shuffle: candidate C with DERANGED teaching labels (anti-cheat -> wcos must collapse)

GO (per seed): substrate_calib weight_cosine >= 1.3x substrate(lesion) weight_cosine AND >= 0.20 (an absolute floor
above the ~0.135 plateau) AND substrate_calib hostlinear_recov >= 1.3x substrate hostlinear_recov AND the shuffle
anti-cheat collapses (wcos < 0.5x substrate_calib). Board GO = >=5/6 seeds. A NO-GO banks candidate C (the per-word
distortion is not affine / not the whole story) and names the next lever; it is never a wall.

Additive, research-runner only, NO sim/ edit. Reuse-by-import of BatchedSubstrateReadout / _calibrate_gain / _wcos /
_softmax_rows and _positions / _eval_hostlinear / _load_eval (UNMODIFIED). host_matmul_on_forward is asserted 0 for the
two substrate arms (the per-word a/c are per-neuron scalars applied to the substrate's OWN output, NOT a feature matmul).
The seed-trap (build-twice thr hash) is emitted.

Run (numpy, per-seed blocking):
  .venv/bin/python -m research.runners._wkv_mouth_readout_perword_calib_derisk --seeds 42 \
      --B 8 --n-train-pos 480 --epochs 8 --sub-read-window 96 \
      --json research/findings/raw/_wkv_readout_decoder_direction/calib_s42.json
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
    BatchedSubstrateReadout, _calibrate_gain, _softmax_rows, _wcos,
)


def _featF(feats_signed):
    return np.concatenate([np.maximum(feats_signed, 0.0), np.maximum(-feats_signed, 0.0)], axis=1)


def _calibrate_perword(s_batch, ro, feats_signed, seed, global_gain, n_probes):
    """Measure the substrate read's PER-WORD affine transfer margin_sub[.,v] ~= a[v]*ideal[.,v] + c[v] with n_probes
    RANDOM probe weights (independent of head_w). Vectorised per-word OLS across (probe x batch) points. Degenerate
    words (never driven / near-zero or wrong-sign slope) fall back to the GLOBAL gain -> byte-identical to the lesion
    on those words. host matmul here is CALIBRATION only (random probes, outside the learning loop), like the global
    gain. Returns (a[V], c[V], frac_fallback, n_calib_reads)."""
    V, D = ro.V, ro.D
    rng = np.random.default_rng(seed * 17 + 5)
    ideals = []
    margs = []
    n_reads = 0
    for _ in range(n_probes):
        W_probe = 0.12 * rng.standard_normal((V, D))
        Wfull = np.concatenate([W_probe, -W_probe], axis=1)
        ideal = _featF(feats_signed) @ Wfull.T                       # [B, V] ideal linear margin (CALIB only)
        s_batch.set_weights(W_probe)
        marg = s_batch.batch_margin(feats_signed, silence_bias=True)  # [B, V] substrate read of the probe
        n_reads += 1
        ideals.append(ideal); margs.append(marg)
    I = np.concatenate(ideals, axis=0)                                # [n_probes*B, V]
    M = np.concatenate(margs, axis=0)
    xm = I.mean(0); ym = M.mean(0)
    vx = ((I - xm) ** 2).mean(0)
    cov = ((I - xm) * (M - ym)).mean(0)
    a = np.where(vx > 1e-12, cov / np.maximum(vx, 1e-12), global_gain)
    c = np.where(vx > 1e-12, ym - a * xm, 0.0)
    bad = ~np.isfinite(a) | (a < 0.1 * global_gain)
    a = np.where(bad, global_gain, a); c = np.where(bad, 0.0, c)
    return a.astype(np.float64), c.astype(np.float64), float(bad.mean()), n_reads


def _logits_of(arm, W, Hb, s_batch, gain, a, c, head_b, unk):
    """The per-arm logits. Returns (logits, host_matmul_forward_flag). The two substrate arms have host_matmul=0."""
    if arm == "host_proxy":
        lg = Hb @ W.T + head_b[None, :]; mm = 1
    else:
        s_batch.set_weights(W)
        margin_sub = s_batch.batch_margin(Hb, silence_bias=True)      # [B, V] the SUBSTRATE read (0 host matmul)
        if arm == "substrate":
            lg = margin_sub / gain + head_b[None, :]                  # GLOBAL gain (the production rule == lesion)
        else:                                                          # substrate_calib / *_shuffle
            lg = (margin_sub - c[None, :]) / a[None, :] + head_b[None, :]   # PER-WORD affine inverse
        mm = 0
    if unk >= 0:
        lg = lg.copy(); lg[:, unk] = -1e30
    return lg, mm


def _learn(arm, seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, a, c, args, shuffle=False):
    V, D = ro.V, ro.D
    B = s_batch.B
    unk = ro.unk_idx
    rng = np.random.default_rng(seed * 991 + 7)                       # SAME init across arms
    W = 0.01 * rng.standard_normal((V, D))
    perm = np.random.default_rng(seed * 131 + 9).permutation(V) if shuffle else None
    Yt = perm[Y] if perm is not None else Y
    idx = np.arange(len(H)); n_full = (len(idx) // B) * B
    traj = []; t0 = time.time(); n_reads = 0; mm_tot = 0
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for start in range(0, n_full, B):
            bi = idx[start:start + B]; Hb = H[bi]
            lg, mm = _logits_of(arm, W, Hb, s_batch, gain, a, c, head_b, unk)
            mm_tot += mm
            if arm != "host_proxy":
                n_reads += 1
            P = _softmax_rows(lg)
            P[np.arange(B), Yt[bi]] -= 1.0
            W = W - args.lr * (P.T @ Hb) / B - args.weight_decay * W
            if args.w_target > 0:
                nrm = float(np.linalg.norm(W))
                if nrm > args.w_target:
                    W *= args.w_target / nrm
        rr = _eval_hostlinear(ro, W, He, Ye, PFe)
        wc = _wcos(W, hw)
        traj.append({"epoch": ep + 1, "hostlinear_recov": round(float(rr["recov_argmax"]), 4), "weight_cosine": wc})
        tag = arm + ("_shuffle" if shuffle else "")
        print(f"[{tag} seed {seed} ep {ep+1}/{args.epochs}] wcos={wc} hostlin={rr['recov_argmax']:.4f} "
              f"reads={n_reads} mm={mm_tot}", flush=True)
    return dict(arm=arm + ("_shuffle" if shuffle else ""), final_wcos=traj[-1]["weight_cosine"],
                final_hostlinear_recov=traj[-1]["hostlinear_recov"], trajectory=traj,
                host_matmul_on_forward=mm_tot, n_substrate_reads=n_reads,
                w_norm=round(float(np.linalg.norm(W)), 2), secs=round(time.time() - t0, 1))


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
        print(f"[skip] seed {seed}: {ckpt} missing", flush=True)
        return None
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
    head_b = ro.head_b.astype(np.float64); hw = ro.head_w

    s_batch = BatchedSubstrateReadout(ro, seed, args.B, hid_pop=args.sub_hid_pop, pop=1, ou_std=args.ou_std,
                                      read_window=args.sub_read_window, hid_gain=args.hid_gain, ratio=args.ratio,
                                      settle_frac=args.settle_frac, n_bias=args.n_bias, bias_drive_pA=args.bias_drive_pA)
    gain, gain_corr = _calibrate_gain(s_batch, ro, H[:args.B], seed)
    a, c, frac_fb, n_cal = _calibrate_perword(s_batch, ro, H[:args.B], seed, gain, args.n_probes)
    print(f"[calib seed {seed}] global_gain={gain:.4g} corr={gain_corr} | perword a[med={np.median(a):.4g} "
          f"iqr={np.percentile(a,75)-np.percentile(a,25):.4g}] frac_fallback={frac_fb:.3f} ({n_cal} calib reads)",
          flush=True)

    R = {}
    R["host_proxy"] = _learn("host_proxy", seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, a, c, args)
    R["substrate"] = _learn("substrate", seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, a, c, args)
    R["substrate_calib"] = _learn("substrate_calib", seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b, gain, a, c, args)
    if getattr(args, "skip_shuffle", False):
        R["substrate_calib_shuffle"] = dict(arm="substrate_calib_shuffle", final_wcos=0.0,
                                            final_hostlinear_recov=0.0, trajectory=[], host_matmul_on_forward=0,
                                            n_substrate_reads=0, w_norm=0.0, secs=0.0, skipped=True)
    else:
        R["substrate_calib_shuffle"] = _learn("substrate_calib", seed, ro, s_batch, H, Y, He, Ye, PFe, hw, head_b,
                                              gain, a, c, args, shuffle=True)

    wc_cal = R["substrate_calib"]["final_wcos"]; wc_sub = R["substrate"]["final_wcos"]
    wc_shuf = R["substrate_calib_shuffle"]["final_wcos"]; wc_hp = R["host_proxy"]["final_wcos"]
    hr_cal = R["substrate_calib"]["final_hostlinear_recov"]; hr_sub = R["substrate"]["final_hostlinear_recov"]
    lift = round(wc_cal / max(1e-9, wc_sub), 3)
    # ATTRIBUTION: the wcos difference is per-word calibration (treatment) vs the global-gain lesion (control) --
    # the SAME substrate read, the ONLY change is the per-word affine map. Not the shuffle, not the ceiling.
    lever(f"perword_calib_vs_global_gain_lesion_wcos_seed{seed}", before=wc_sub, after=wc_cal, required=False,
          continuous=round(hr_cal - hr_sub, 4))
    m = {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.B, "n_train_pos": len(H), "n_eval_pos": len(He),
        "epochs": args.epochs, "sub_read_window": args.sub_read_window, "n_probes": args.n_probes,
        "global_gain": round(gain, 5), "gain_substrate_vs_linear_corr": gain_corr, "perword_frac_fallback": frac_fb,
        "wcos_host_proxy": wc_hp, "wcos_substrate_lesion": wc_sub, "wcos_substrate_calib": wc_cal,
        "wcos_substrate_calib_shuffle": wc_shuf, "wcos_calib_over_lesion": lift,
        "hostlin_host_proxy": R["host_proxy"]["final_hostlinear_recov"],
        "hostlin_substrate_lesion": hr_sub, "hostlin_substrate_calib": hr_cal,
        "host_matmul_on_forward_substrate": R["substrate"]["host_matmul_on_forward"],
        "host_matmul_on_forward_calib": R["substrate_calib"]["host_matmul_on_forward"],
        "seed_trap": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)},
        "arms": _native(R),
    }
    # GO (per seed): calib lifts wcos AND hostlin over the lesion, above an absolute floor, with shuffle collapsed and
    # the substrate arms genuinely 0-host-matmul on the forward.
    m["forward_is_substrate"] = bool(m["host_matmul_on_forward_calib"] == 0 and m["host_matmul_on_forward_substrate"] == 0)
    m["shuffle_collapses"] = bool(wc_shuf < 0.5 * wc_cal)
    m["go"] = bool(wc_cal >= 1.3 * wc_sub and wc_cal >= 0.20 and hr_cal >= 1.3 * hr_sub
                   and m["shuffle_collapses"] and m["forward_is_substrate"])
    print(f"[seed {seed}] WCOS calib={wc_cal} lesion={wc_sub} (x{lift}) ceiling(host_proxy)={wc_hp} "
          f"shuffle={wc_shuf} | HOSTLIN calib={hr_cal} lesion={hr_sub} | GO={m['go']}", flush=True)
    del s_batch
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n-train-pos", type=int, default=480)
    ap.add_argument("--n-eval-pos", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=8e-4)
    ap.add_argument("--w-target", type=float, default=40.0)
    ap.add_argument("--n-probes", type=int, default=24)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-read-window", type=int, default=96)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--skip-shuffle", action="store_true",
                    help="skip the shuffle-teach anti-cheat arm (fast endpoint screen; the mechanism GO is decided by "
                         "the gradient-alignment runner, not this endpoint at reduced CPU scale)")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_readout_decoder_direction/calib.json")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    t0 = time.time()
    rows = []
    for seed in seeds:
        m = run_seed(seed, args)
        if m is not None:
            rows.append(m)
    summary = {}
    if rows:
        summary = {
            "n_seeds": len(rows), "go_count": int(sum(1 for r in rows if r["go"])),
            "go_5of6": bool(sum(1 for r in rows if r["go"]) >= 5),
            "wcos_substrate_calib_mean": round(float(np.mean([r["wcos_substrate_calib"] for r in rows])), 4),
            "wcos_substrate_lesion_mean": round(float(np.mean([r["wcos_substrate_lesion"] for r in rows])), 4),
            "wcos_host_proxy_mean": round(float(np.mean([r["wcos_host_proxy"] for r in rows])), 4),
            "wcos_calib_shuffle_mean": round(float(np.mean([r["wcos_substrate_calib_shuffle"] for r in rows])), 4),
            "wcos_calib_over_lesion_mean": round(float(np.mean([r["wcos_calib_over_lesion"] for r in rows])), 3),
            "hostlin_substrate_calib_mean": round(float(np.mean([r["hostlin_substrate_calib"] for r in rows])), 4),
            "hostlin_substrate_lesion_mean": round(float(np.mean([r["hostlin_substrate_lesion"] for r in rows])), 4),
            "forward_is_substrate_all": bool(all(r["forward_is_substrate"] for r in rows)),
            "shuffle_collapses_all": bool(all(r["shuffle_collapses"] for r in rows)),
        }
    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t0, 1),
           "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
