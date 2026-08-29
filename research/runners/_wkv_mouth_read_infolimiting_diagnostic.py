"""gap#4 / #80 mouth read-power wall -- INFORMATION-LIMITING-CORRELATIONS diagnostic (deep-research shortlist rank 3,
`research/findings/2026-08-28-mouth-read-power-wall-deep-research-ranked-shortlist.md`).

THE QUESTION (Moreno-Bote, Beck, Kanitscheider, Pitkow, Latham & Pouget, "Information-limiting correlations",
Nat Neurosci 17:1410-1417, 2014): at FIXED trained read-out weights (head_w) and a FIXED probe, collect repeated
substrate reads (trial-to-trial noise from OU membrane noise + spiking stochasticity). If the leading eigenvector
of the trial-to-trial NOISE covariance aligns with the read's SIGNAL/decode direction (the ideal linear map
Wfull@feat), NO amount of same-population pooling/averaging/expansion can reduce that noise's effect on the
decision -- it is INFORMATION-LIMITING, a genuine ceiling. If the leading noise eigenvector sits away from the
signal direction (at or below a channel-identity permutation null), the noise is RECOVERABLE by a better read
(pooling, regularized estimation, etc).

THIS IS A DIAGNOSTIC, not a GO/NO-GO lever: it produces no read-power lift itself. It is the cheapest, most
decisive SEQUENCING gate named by the shortlist: an information-limited verdict predicts the nonlinear-expansion /
fixed-ensemble family (ranks 4-6) is futile and routes effort to the independent-pathway family (ranks 1-2,
already built/staged); it also independently explains why the ensemble (`--sub-pop`) read was inert (common-mode
noise, `2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`).

METHOD (per probe k, k=0..B-1):
  1. FIX weights = ro.head_w (the trained/copied reference; unaffected by any training-loop bug -- always the
     substrate's best-case read target). set_weights() once.
  2. Collect R repeated `batch_margin(feats)` calls on the SAME B probes (same bridge, same weights, same feature
     drive) -- each call is a fresh noise draw from the SAME still-running OU/spiking process (never reseeded
     between calls), giving R independent trials per probe.
  3. Restrict to channel subset C_k = top-c word-pool channels by this probe's own mean margin (the "live"/
     competitive channels -- with only R trials, a full V=1000-dim covariance is wildly underdetermined; this is a
     declared scoping simplification, R > 2x c by construction).
  4. NOISE covariance Sigma_noise_k = cov over R trials of margin[:, C_k] (mean-subtracted per probe). Leading
     eigenvector v1_k (unit, top eigenvalue of the symmetric eigendecomposition).
  5. SIGNAL/decode direction s_k = normalize(Wfull_headw[C_k,:] @ featF(h_k)) -- the THEORETICAL (noiseless)
     ideal-linear-map margin restricted to C_k; this is literally "head_w's signal direction" for this probe.
  6. alignment_k = |cos(v1_k, s_k)|. PERMUTATION NULL: permute the c channel-identity labels of s_k (breaking the
     noise<->signal channel correspondence while preserving both vectors' own structure), N_PERM times; verdict =
     alignment_k vs this null's mean/std/percentile.
  7. POOLED (closer to the canonical multi-stimulus construction): a shared channel set C_shared (top-c_shared by
     summed |mean| across all B probes), pooled per-probe-mean-subtracted noise covariance across all B*R trials
     -> pooled leading eigenvector; SIGNAL covariance = cov of the B probes' own mean margins (restricted to
     C_shared) -> its leading eigenvector (the discrete analogue of the tuning-curve/df-ds direction). Same
     permutation-null test.

BUNDLED (mandatory per the shortlist rank-3 text): a companion ARCHIVAL check (no compute) of whether the
"deep ~0.34-0.37 plateau" and the earlier cupy structure-characterization SUBSTRATE-WALL survive the 2026-08-27
stale-COO/megakernel-v2-WT-cache fix (`sim/bridge.py::mark_weights_edited()`, commit d6c375de5) -- see the
companion finding doc, not this runner (a pure git-history/artifact-lineage read, not a new run).

ANTI-CHEATS: FIXED weights throughout (no learning, no weight transport); determinism via cfg.seed (build-twice
hash, reused `_thr_hash`); a resid-nonzero sanity check (proves trials actually vary trial-to-trial, i.e. this
is not a frozen/deterministic read masquerading as noise); the permutation null is the significance instrument
(not an arbitrary threshold). CPU-only (SIM_BACKEND=numpy), B=8/read-window=64 (the structure-characterization
runner's established memory-safe operating point). Reuse-by-import only, no sim/ edit.

Run (diagnostic, ~2-4 min/seed on CPU):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_read_infolimiting_diagnostic \
      --seeds 42,43 --json research/findings/raw/_wkv_mouth_read_infolimiting/diag.json
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

from tools.lab import lever, assert_backend  # noqa: E402

from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _thr_hash,
)
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _load_eval, _native  # noqa: E402

N_PERM = 1000


def _top_channels(mean_vec, c):
    return np.argsort(-mean_vec)[:c]


def _leading_eigvec(cov):
    """Symmetric eigendecomposition; np.linalg.eigh returns ASCENDING eigenvalues -> last column is the top PC."""
    w, v = np.linalg.eigh(cov)
    return v[:, -1], float(w[-1]), w


def _perm_null(v1, s_hat, rng, n_perm=N_PERM):
    c = s_hat.shape[0]
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(c)
        null[i] = abs(float(v1 @ s_hat[perm]))
    return null


def _alignment_verdict(v1, s_vec, rng, tag):
    s_hat = s_vec / max(1e-12, float(np.linalg.norm(s_vec)))
    v1_hat = v1 / max(1e-12, float(np.linalg.norm(v1)))
    align = abs(float(v1_hat @ s_hat))
    null = _perm_null(v1_hat, s_hat, rng)
    z = (align - float(null.mean())) / max(1e-12, float(null.std()))
    pct = float((null <= align).mean())
    verdict = "information_limiting" if (z > 2.0 and pct >= 0.975) else "recoverable"
    return {
        "tag": tag, "alignment": round(align, 4),
        "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4),
        "null_p95": round(float(np.percentile(null, 95)), 4),
        "z": round(z, 3), "percentile": round(pct, 4), "verdict": verdict,
    }


def run_seed(seed, args):
    rng = np.random.default_rng(seed * 104729 + 17)
    ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
    if not Path(ckpt).exists():
        print(f"[skip] seed {seed}: {ckpt} missing", flush=True)
        return None
    ro = WKVReadout(ckpt)

    h1 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                   args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
    h2 = _thr_hash(seed, ro, args.sub_hid_pop, args.sub_pop, args.ou_std, args.sub_read_window,
                   args.hid_gain, args.ratio, args.n_bias, args.bias_drive_pA)
    seeded = bool(h1 == h2)
    print(f"[seed-trap seed {seed}] {h1} == {h2} -> {'SEEDED' if seeded else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(0.8 * len(usable))
    train_ids = usable[:cut]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.batch)   # exactly B probes, host-side (cheap)

    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    hw = ro.head_w.astype(np.float64)
    Wfull_hw = np.concatenate([hw, -hw], axis=1)                          # [V, 2D], same construction as _measure_gain
    feats = H[:args.batch]                                                # [B, D] signed host feature, FIXED probes
    featF = np.concatenate([np.maximum(feats, 0.0), np.maximum(-feats, 0.0)], axis=1)   # [B, 2D]
    host_ideal_raw = featF @ Wfull_hw.T                                   # [B, V] THEORETICAL (noiseless) margin

    s_batch.set_weights(hw)                                               # FIX weights ONCE -- no learning, no re-set

    trials_raw = np.empty((args.n_trials, args.batch, ro.V), dtype=np.float64)
    t0 = time.time()
    for r in range(args.n_trials):
        trials_raw[r] = s_batch.batch_margin(feats, silence_bias=True)    # fresh noise draw, SAME weights/probes
    read_secs = round(time.time() - t0, 1)

    # ---- ROW-CENTER (per trial, per probe: subtract that row's own mean over the FULL V) ----
    # _measure_gain's own commentary (research/runners/_wkv_mouth_readout_softmax_confidence_derisk.py) established
    # the decisive instrument distinction: softmax(logits) is invariant to a per-row additive shift, so a shared
    # ACROSS-CHANNEL common-mode component (the "row baseline") is INVISIBLE to the real decoding/learning signal --
    # measuring the RAW (uncentered) covariance risks the leading noise eigenvector being dominated by that
    # decision-irrelevant common-mode shift rather than the within-row (across-vocabulary) structure the argmax/
    # softmax decision actually depends on. Row-center BOTH the substrate reads and the theoretical ideal margin
    # before the covariance/signal-direction analysis (the raw, uncentered version is also reported for comparison).
    trials = trials_raw - trials_raw.mean(axis=2, keepdims=True)
    host_ideal = host_ideal_raw - host_ideal_raw.mean(axis=1, keepdims=True)

    mean_all = trials.mean(axis=0)                                        # [B, V] per-probe trial mean (row-centered)
    resid_all = trials - mean_all[None, :, :]                             # [R, B, V] mean-subtracted noise
    resid_std = float(resid_all.std())
    cv = float(resid_all.std() / max(1e-9, np.abs(mean_all).mean()))
    resid_std_raw = float((trials_raw - trials_raw.mean(axis=0, keepdims=True)).std())
    print(f"[seed {seed}] {args.n_trials} trials collected in {read_secs}s; resid_std={resid_std:.4f} "
          f"(row-centered; raw={resid_std_raw:.4f}) (sanity: genuine trial-to-trial noise iff > 0), CV~{cv:.4f}",
          flush=True)

    per_probe = []
    for b in range(args.batch):
        Ck = _top_channels(mean_all[b], args.n_channels)
        resid_b = resid_all[:, b, :][:, Ck]                               # [R, c]
        cov_b = np.cov(resid_b, rowvar=False)
        v1_b, top_eigval, all_eigvals = _leading_eigvec(cov_b)
        var_frac = float(top_eigval / max(1e-12, all_eigvals.sum()))
        s_b = host_ideal[b, Ck]
        res = _alignment_verdict(v1_b, s_b, np.random.default_rng(seed * 1009 + 3 * b + 1), f"probe{b}_seed{seed}")
        res["target_word"] = int(Y[b]); res["leading_eigval_var_frac"] = round(var_frac, 4)
        res["channels"] = [int(x) for x in Ck]
        per_probe.append(res)
        print(f"  [seed {seed} probe {b}] align={res['alignment']} null_mean={res['null_mean']} "
              f"z={res['z']} pct={res['percentile']} eigval_frac={var_frac:.3f} -> {res['verdict']}", flush=True)

    # ---- POOLED (canonical multi-stimulus construction): shared channel set across ALL B probes ----
    activity = np.abs(mean_all).sum(axis=0)                                # [V] summed |mean| across probes
    C_shared = _top_channels(activity, args.n_channels_pooled)
    pooled_resid = resid_all[:, :, C_shared].reshape(-1, args.n_channels_pooled)   # [R*B, c_shared]
    pooled_cov = np.cov(pooled_resid, rowvar=False)
    v1_pool, pool_eigval, pool_all_eigvals = _leading_eigvec(pooled_cov)
    pool_var_frac = float(pool_eigval / max(1e-12, pool_all_eigvals.sum()))
    signal_cov = np.cov(mean_all[:, C_shared], rowvar=False)               # cov OF the B probe means (signal variation)
    s1_pool, s_eigval, s_all_eigvals = _leading_eigvec(signal_cov)
    pooled_res = _alignment_verdict(v1_pool, s1_pool, np.random.default_rng(seed * 5003 + 9), f"pooled_seed{seed}")
    pooled_res["noise_leading_eigval_var_frac"] = round(pool_var_frac, 4)
    pooled_res["signal_leading_eigval_var_frac"] = round(float(s_eigval / max(1e-12, s_all_eigvals.sum())), 4)
    print(f"[seed {seed}] POOLED align={pooled_res['alignment']} null_mean={pooled_res['null_mean']} "
          f"z={pooled_res['z']} -> {pooled_res['verdict']}", flush=True)

    lever(f"infolimiting_alignment_vs_null_seed{seed}",
          before=round(float(np.mean([r["null_mean"] for r in per_probe])), 4),
          after=round(float(np.mean([r["alignment"] for r in per_probe])), 4),
          required=False, continuous=round(float(np.mean([r["z"] for r in per_probe])), 3))

    n_info_limiting = int(sum(1 for r in per_probe if r["verdict"] == "information_limiting"))
    return {
        "seed": seed, "V": ro.V, "D": ro.D, "B": args.batch, "n_trials": args.n_trials,
        "n_channels_per_probe": args.n_channels, "n_channels_pooled": args.n_channels_pooled,
        "seed_hash_check": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": seeded},
        "resid_std_sanity": round(resid_std, 5), "resid_cv_sanity": round(cv, 5),
        "resid_std_raw_uncentered": round(resid_std_raw, 5),
        "read_secs": read_secs,
        "per_probe": per_probe, "pooled": pooled_res,
        "n_probes_information_limiting": n_info_limiting, "n_probes_total": args.batch,
        "seed_verdict": ("information_limiting" if (n_info_limiting >= (args.batch + 1) // 2
                                                     or pooled_res["verdict"] == "information_limiting")
                         else "recoverable"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42,43")
    ap.add_argument("--batch", type=int, default=8)                # B probes/blocks (memory-safe, matches structure-char)
    ap.add_argument("--n-trials", type=int, default=60)            # R repeated reads per probe (R > 3x n-channels)
    ap.add_argument("--n-channels", type=int, default=15)          # per-probe channel subset c (top-c by own mean)
    ap.add_argument("--n-channels-pooled", type=int, default=20)   # pooled shared channel subset
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--sub-read-window", type=int, default=64)     # memory-safe (matches structure-characterization)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_mouth_read_infolimiting/diag.json")
    args = ap.parse_args()

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"),
                   note="(information-limiting diagnostic is CPU-lane by task instruction; GPU busy w/ readout fair-test)")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    t0 = time.time()
    rows = []
    for seed in seeds:
        r = run_seed(seed, args)
        if r is not None:
            rows.append(r)

    summary = {}
    if rows:
        n_il = int(sum(1 for r in rows if r["seed_verdict"] == "information_limiting"))
        summary = {
            "n_seeds": len(rows),
            "n_seeds_information_limiting": n_il,
            "n_seeds_recoverable": len(rows) - n_il,
            "mean_per_probe_alignment": round(float(np.mean([p["alignment"] for r in rows for p in r["per_probe"]])), 4),
            "mean_per_probe_null_mean": round(float(np.mean([p["null_mean"] for r in rows for p in r["per_probe"]])), 4),
            "mean_per_probe_z": round(float(np.mean([p["z"] for r in rows for p in r["per_probe"]])), 3),
            "pooled_alignment_by_seed": {int(r["seed"]): r["pooled"]["alignment"] for r in rows},
            "pooled_z_by_seed": {int(r["seed"]): r["pooled"]["z"] for r in rows},
            "pooled_verdict_by_seed": {int(r["seed"]): r["pooled"]["verdict"] for r in rows},
            "resid_std_all_nonzero": bool(all(r["resid_std_sanity"] > 1e-6 for r in rows)),
        }
        summary["verdict"] = "information_limiting" if n_il >= (len(rows) + 1) // 2 else "recoverable"
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)

    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "external_source": "Moreno-Bote, Beck, Kanitscheider, Pitkow, Latham & Pouget, "
                               "'Information-limiting correlations', Nat Neurosci 17:1410-1417 (2014)",
           "n_perm": N_PERM,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "device": "cpu",
           "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
