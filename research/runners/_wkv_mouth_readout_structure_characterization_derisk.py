"""gap#4 / #80 mouth read-SNR -- CHARACTERIZE the structure-selective read-fidelity collapse (follow-on to the
2026-08-27 softmax-confidence NO-GO).

That finding established: the substrate's raw graded-conductance margin correlates ~0.95 with the ideal linear
map for a RANDOM/incoherent weight direction, but ~0.00 for the STRUCTURED target direction (head_w, the
trained mouth's real decoder), at EVERY tested scale (10%-100% of magnitude) -- a structure-selective collapse,
not a magnitude/confidence problem. That finding characterized ONE structured direction (head_w). It does NOT
say whether corr~0 is a property of head_w SPECIFICALLY, or of ALL "structured" (low-entropy / sparse /
eigen-misaligned) directions.

THE DECISIVE QUESTION this script answers: can the substrate read SOME structured directions faithfully (->
the fix is TARGET-RECODING: steer the mouth's learned direction toward a substrate-readable code), or NONE (->
a deeper substrate wall needing a different read mechanism)? Four families, all at a FIXED probe norm
(||head_w||, since the NO-GO finding showed corr is magnitude-independent over 10-100% of that norm, so fixing
norm isolates STRUCTURE/direction as the only varying axis):

  (1) INTERPOLATION random<->head_w (alpha 0..1, 6 steps): where does corr collapse along the straight-line path
      from an incoherent direction to the trained target's OWN direction?
  (2) SPARSITY (per-row top-k-by-|value|, k in {1,2,4,8,16,32,64,128=dense}), applied to a RANDOM base matrix
      (NOT head_w) -- isolates whether SPARSITY/low-entropy PER SE (independent of alignment to head_w) kills
      the read.
  (3) EIGEN-ALIGNMENT (Schuessler et al. 2023 eLife 93060 "aligned/oblique": a readout direction inside the
      top-PC subspace of population activity reads high-correlation/"aligned"; outside it, low-correlation/
      "oblique"). Projects head_w onto the TOP-k vs BOTTOM-k principal components of the ACTIVATION feature
      matrix H (the substrate's own input-drive covariance), rank-swept k in {1,2,4,8,16,32,64,128=full}. Tests
      whether head_w's own read-fidelity recovers when confined to the substrate's dominant activity subspace.
  (4) ANCHORS: N=3 random controls (reproduces the NO-GO's ~0.95 baseline) + head_w itself (reproduces ~0.00).

Reuse-by-import ONLY: `_measure_gain` (the exact corr instrument) from the softmax-confidence NO-GO's own
runner; `BatchedSubstrateReadout` / `_thr_hash` from the eprop_batched_substrate runner; `_positions` from the
eprop_learn runner; `WKVReadout` / `_load_eval` / `_native` from the fewspike_read runner; `lever` from
tools.lab. NO reimplementation of the substrate-forward or the corr instrument. Additive, no sim/ edit.

ANTI-CHEATS: same substrate forward as the NO-GO diagnosis (0 host matmul on any learning signal -- this script
does no learning at all, it is pure read-fidelity measurement); determinism via cfg.seed (build-twice hash,
reused _thr_hash). CPU-only (SIM_BACKEND=numpy), memory-safe scale (B=8, sub-read-window=64) matching the
NO-GO's reduced-scale reference and the init-scale-sweep runner's established-safe numpy configuration.

Run (smoke, 1 seed, ~2-4 min on CPU):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_readout_structure_characterization_derisk \
      --seeds 42 --json research/findings/raw/_wkv_structure_characterization/char_s42.json

Run (6-seed decisive, ~15-25 min on CPU):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_readout_structure_characterization_derisk \
      --seeds 42,43,44,100,101,102 \
      --json research/findings/raw/_wkv_structure_characterization/char_6seed.json
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

from research.runners._wkv_mouth_readout_softmax_confidence_derisk import _measure_gain  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _thr_hash,
)
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _load_eval, _native  # noqa: E402

RECODABLE_CORR_THRESHOLD = 0.5   # a structured (non-random) probe reading ABOVE this is "faithfully readable"
WALL_CORR_CEILING = 0.15         # every structured probe reading AT OR BELOW this (while random anchors ~0.95) is "wall"


def _rescale(raw, target_norm):
    n = float(np.linalg.norm(raw))
    if n < 1e-12:
        return raw.copy()
    return raw * (target_norm / n)


def _row_topk_sparsify(W, k):
    """Zero all but the k largest-|value| entries in each ROW (each vocab word's D-dim read weight). k>=D is the
    dense (unmodified-structure) limit. A pure STRUCTURE manipulation (row support), not a direction manipulation
    tied to head_w -- the base matrix is a random draw, so this isolates sparsity/low-entropy PER SE."""
    V, D = W.shape
    k = int(min(k, D))
    out = np.zeros_like(W)
    idx = np.argsort(-np.abs(W), axis=1)[:, :k]
    rows = np.repeat(np.arange(V), k)
    cols = idx.reshape(-1)
    out[rows, cols] = W[rows, cols]
    return out


def _mean_ipr(W):
    """Mean inverse participation ratio across rows: (sum w^2)^2 / sum w^4. IPR=D (128 here) for a uniform/dense
    row, IPR=1 for a row concentrated in a single entry -- a standard sparsity/entropy scalar."""
    row_sq = W.astype(np.float64) ** 2
    num = row_sq.sum(axis=1) ** 2
    den = np.maximum(1e-18, (row_sq ** 2).sum(axis=1))
    return float(np.mean(num / den))


def _activity_pca(H):
    """PCA of the activation/feature matrix H [N,D] via SVD of the mean-centered data. Returns eigvecs [D,D]
    (columns sorted DESCENDING eigenvalue -- np.linalg.svd already returns singular values descending) and
    eigvals [D]."""
    Hc = H.astype(np.float64) - H.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    eigvecs = Vt.T
    eigvals = (S ** 2) / max(1, (H.shape[0] - 1))
    return eigvecs, eigvals


def _project(hw, eigvecs, k, bottom=False):
    D = eigvecs.shape[0]
    k = int(min(k, D))
    P = eigvecs[:, -k:] if bottom else eigvecs[:, :k]
    return hw @ P @ P.T


def run_seed(seed, args):
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
    n_pca_pos = max(args.n_pca_pos, args.batch)
    H, _, _ = _positions(ro, train_ids, args.warmup, n_pca_pos)          # host-side only, cheap (N x D=128)

    s_batch = BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                      ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                      ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                      bias_drive_pA=args.bias_drive_pA)
    feats = H[:args.batch]                                              # FIXED batch for every measurement below
    hw = ro.head_w.astype(np.float64)
    hw_norm = float(np.linalg.norm(hw))
    T = hw_norm                                                          # common target norm for EVERY probe

    def measure(W_probe, tag):
        res, _, _ = _measure_gain(s_batch, W_probe, feats)
        print(f"  [seed {seed}] {tag:32s} norm={res['probe_norm']:<8.3f} corr={res['corr']:<8.4f} "
              f"corr_rc={res['corr_row_centered']:<8.4f}", flush=True)
        return res

    # ---- (4) ANCHORS: N random controls + head_w itself, all rescaled to T ----
    anchors = []
    for i in range(args.n_random_controls):
        rng = np.random.default_rng(seed * 7919 + 101 * i + 3)
        Wr = _rescale(0.12 * rng.standard_normal((ro.V, ro.D)), T)
        r = measure(Wr, f"anchor_random_{i}")
        r["kind"] = "random"; anchors.append(r)
    headw_anchor = measure(hw, "anchor_headw")
    headw_anchor["kind"] = "headw"

    # ---- (1) INTERPOLATION random<->head_w ----
    rng0 = np.random.default_rng(seed * 7 + 3)                          # SAME draw as the NO-GO's probe A
    Wa = 0.12 * rng0.standard_normal((ro.V, ro.D))
    unit_rand = Wa / max(1e-12, np.linalg.norm(Wa))
    unit_hw = hw / max(1e-12, hw_norm)
    alphas = [float(x) for x in args.alphas.split(",")]
    interp = []
    for alpha in alphas:
        raw = (1.0 - alpha) * unit_rand + alpha * unit_hw
        Wp = _rescale(raw, T)
        r = measure(Wp, f"interp_alpha={alpha:.2f}")
        r["alpha"] = alpha
        interp.append(r)

    # ---- (2) SPARSITY sweep (random base, NOT head_w-aligned) ----
    ks = [int(x) for x in args.sparsity_ks.split(",")]
    sparsity = []
    headw_ipr = _mean_ipr(hw)
    for k in ks:
        Wk_raw = _row_topk_sparsify(Wa, k)
        Wk = _rescale(Wk_raw, T)
        r = measure(Wk, f"sparsity_k={k}")
        r["k"] = k
        r["mean_ipr"] = round(_mean_ipr(Wk_raw), 3)
        sparsity.append(r)

    # ---- (3) EIGEN-ALIGNMENT: project head_w onto top-k / bottom-k PCs of the ACTIVITY H ----
    eigvecs, eigvals = _activity_pca(H)
    eks = [int(x) for x in args.eigen_ks.split(",")]
    eigen_top, eigen_bot = [], []
    for k in eks:
        top_raw = _project(hw, eigvecs, k, bottom=False)
        top_energy = float(np.linalg.norm(top_raw) / max(1e-12, hw_norm))
        Wtop = _rescale(top_raw, T)
        rt = measure(Wtop, f"eigen_top_k={k}")
        rt["k"] = k; rt["energy_frac_of_headw"] = round(top_energy, 4)
        eigen_top.append(rt)

        bot_raw = _project(hw, eigvecs, k, bottom=True)
        bot_energy = float(np.linalg.norm(bot_raw) / max(1e-12, hw_norm))
        Wbot = _rescale(bot_raw, T)
        rb = measure(Wbot, f"eigen_bottom_k={k}")
        rb["k"] = k; rb["energy_frac_of_headw"] = round(bot_energy, 4)
        eigen_bot.append(rb)

    # ---- verdict scaffolding for THIS seed ----
    random_corr_mean = float(np.mean([a["corr"] for a in anchors]))
    max_structured_corr = max(
        [r["corr"] for r in interp if r["alpha"] > 0.0]
        + [r["corr"] for r in sparsity]
        + [r["corr"] for r in eigen_top]
        + [r["corr"] for r in eigen_bot]
        + [headw_anchor["corr"]]
    )
    all_structured_corrs = (
        [r["corr"] for r in interp if r["alpha"] > 0.0]
        + [r["corr"] for r in sparsity]
        + [r["corr"] for r in eigen_top]
        + [r["corr"] for r in eigen_bot]
    )
    seed_recodable = bool(max_structured_corr > RECODABLE_CORR_THRESHOLD)
    seed_all_wall = bool(max(all_structured_corrs) <= WALL_CORR_CEILING) if all_structured_corrs else None

    # ATTRIBUTION levers (per-seed, non-required -- these are characterization reads, not pass/fail A/Bs)
    lever(f"interp_corr_alpha0_vs_alpha1_seed{seed}", before=interp[0]["corr"], after=interp[-1]["corr"],
          required=False, continuous=interp[0]["corr"] - interp[-1]["corr"])
    lever(f"sparsity_corr_dense_vs_sparsest_seed{seed}", before=sparsity[-1]["corr"], after=sparsity[0]["corr"],
          required=False, continuous=sparsity[-1]["corr"] - sparsity[0]["corr"])
    lever(f"eigen_topk_vs_botk_corr_seed{seed}", before=eigen_bot[0]["corr"], after=eigen_top[0]["corr"],
          required=False, continuous=eigen_top[0]["corr"] - eigen_bot[0]["corr"])

    out = {
        "seed": seed, "V": ro.V, "D": ro.D, "batch": args.batch, "sub_read_window": args.sub_read_window,
        "target_norm_T": round(T, 3), "head_w_norm": round(hw_norm, 3), "head_w_mean_ipr": round(headw_ipr, 3),
        "seed_hash_check": {"thr_hash_1": h1, "thr_hash_2": h2, "seeded": seeded},
        "anchors_random": anchors, "anchor_headw": headw_anchor,
        "interpolation": interp, "sparsity": sparsity,
        "eigen_topk": eigen_top, "eigen_bottomk": eigen_bot,
        "eigvals_top8": [round(float(x), 4) for x in eigvals[:8]],
        "random_corr_mean": round(random_corr_mean, 4),
        "headw_corr": headw_anchor["corr"],
        "max_structured_corr": round(max_structured_corr, 4),
        "seed_verdict_recodable": seed_recodable,
        "seed_verdict_all_wall": seed_all_wall,
    }
    del s_batch
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--batch", type=int, default=8)               # memory-safe (shared machine, 2026-08-27 OOM)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--sub-read-window", type=int, default=64)     # memory-safe default
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-pca-pos", type=int, default=600)           # host-side only, cheap (N x D=128 floats)
    ap.add_argument("--n-random-controls", type=int, default=3)
    ap.add_argument("--alphas", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--sparsity-ks", type=str, default="1,2,4,8,16,32,64,128")
    ap.add_argument("--eigen-ks", type=str, default="1,2,4,8,16,32,64,128")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_wkv_structure_characterization/char.json")
    args = ap.parse_args()

    assert_backend(os.environ.get("SIM_BACKEND", "numpy"),
                   note="(structure-characterization is CPU-lane by owner instruction; memory-safe B=8/window=64)")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    t0 = time.time()
    rows = []
    for seed in seeds:
        r = run_seed(seed, args)
        if r is not None:
            rows.append(r)

    summary = {}
    if rows:
        summary = {
            "n_seeds": len(rows),
            "random_corr_mean": round(float(np.mean([r["random_corr_mean"] for r in rows])), 4),
            "headw_corr_mean": round(float(np.mean([r["headw_corr"] for r in rows])), 4),
            "max_structured_corr_mean": round(float(np.mean([r["max_structured_corr"] for r in rows])), 4),
            "max_structured_corr_max_over_seeds": round(float(np.max([r["max_structured_corr"] for r in rows])), 4),
            "n_seeds_recodable": int(sum(1 for r in rows if r["seed_verdict_recodable"])),
            "n_seeds_all_wall": int(sum(1 for r in rows if r["seed_verdict_all_wall"])),
            "eigen_topk_corr_by_k_mean": {
                str(rows[0]["eigen_topk"][i]["k"]): round(float(np.mean([r["eigen_topk"][i]["corr"] for r in rows])), 4)
                for i in range(len(rows[0]["eigen_topk"]))
            } if rows else {},
            "eigen_bottomk_corr_by_k_mean": {
                str(rows[0]["eigen_bottomk"][i]["k"]): round(float(np.mean([r["eigen_bottomk"][i]["corr"] for r in rows])), 4)
                for i in range(len(rows[0]["eigen_bottomk"]))
            } if rows else {},
            "sparsity_corr_by_k_mean": {
                str(rows[0]["sparsity"][i]["k"]): round(float(np.mean([r["sparsity"][i]["corr"] for r in rows])), 4)
                for i in range(len(rows[0]["sparsity"]))
            } if rows else {},
            "interp_corr_by_alpha_mean": {
                str(rows[0]["interpolation"][i]["alpha"]): round(float(np.mean([r["interpolation"][i]["corr"] for r in rows])), 4)
                for i in range(len(rows[0]["interpolation"]))
            } if rows else {},
        }
        # DECISIVE aggregate verdict: RECODABLE needs a MAJORITY of seeds to show >threshold on SOME structured
        # probe; SUBSTRATE-WALL needs EVERY seed's structured-family max to sit at/below the wall ceiling.
        summary["verdict"] = ("RECODABLE" if summary["n_seeds_recodable"] >= (len(rows) + 1) // 2
                              else ("SUBSTRATE-WALL" if summary["n_seeds_all_wall"] == len(rows) else "MIXED"))
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)

    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "recodable_threshold": RECODABLE_CORR_THRESHOLD, "wall_ceiling": WALL_CORR_CEILING,
           "external_source": "Schuessler et al. 2023 eLife 12:e93060 https://elifesciences.org/articles/93060 "
                               "(aligned/oblique: readout directions inside vs outside the top-PC activity subspace)",
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
