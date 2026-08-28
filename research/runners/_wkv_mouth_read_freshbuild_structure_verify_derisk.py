"""MOUTH read-SNR (2026-08-27) -- VERIFY that the "structure-selective substrate read wall"
(2026-08-27-mouth-readsnr-structure-characterization-cupy-SUBSTRATE-WALL: head_w/structured targets
read corr ~0.025 vs random ~0.31, "0/6 recodable, 6/6 all-wall") is a STALE-WEIGHTS MEASUREMENT
ARTIFACT, not a substrate limit -- and that on a CORRECT read (fresh substrate per probe) a STRUCTURED
target decodes as faithfully as a random one.

ROOT CAUSE (diagnosed, see FAILURE_LOG 2026-08-27): `BatchedSubstrateReadout` reuses ONE built bridge
across many `set_weights()` calls, but synaptic transmission reads `sim/bridge.py::_get_cached_coo()`, a
cached COO matrix invalidated ONLY on a STRUCTURAL change -- NEVER on a weight edit. So every read after
the first transmits the FIRST-loaded weight matrix. Measuring a RANDOM probe first (faithful ~0.95) and
head_w later (stale COO -> ~0) manufactured the "structure-selective collapse". The correct instrument is
one weight per built substrate (a FRESH build per probe), where the COO cache is built from the loaded
weights (build_store runs no steps; the first replay step rebuilds the cache).

THIS RUNNER, per seed, measures BOTH:
  (ARTIFACT) shared build: measure a random anchor (1st), then head_w (2nd), then a fresh random (3rd)
    -- reproduces the published artifact (1st ~0.95; 2nd, 3rd ~0 regardless of structure).
  (TRUTH) fresh build per probe: 3 random anchors + head_w + a sparse-structured probe + an eigen-top
    structured probe + the head_w-direction (interp alpha=1) -- each on its OWN fresh build.

VERDICT (GO = the wall is an artifact + structured reads are faithful): on the TRUTH side, the structured
probes (head_w, sparse, eigen-top, head_w-dir) read corr > STRUCT_OK on >= bar seeds, matching the random
anchors; AND the ARTIFACT side reproduces the stale collapse (structured-2nd <= WALL_CEIL) so the finding
is self-demonstrating.

Reuse-by-import ONLY (NO sim/ edit, additive): `_measure_gain` (the exact corr instrument),
`BatchedSubstrateReadout`, `_thr_hash`, `_positions`, `WKVReadout`, `_load_eval`, and the char runner's
`_rescale`/`_row_topk_sparsify`/`_activity_pca`/`_project`. CPU-safe scale reused from the char runner
(B=8, read_window=64) -- host-RAM-safe (the 2026-08-27 OOM was B=48). cupy backend (production substrate).

Run (smoke, 1 seed, ~2-3 min):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_read_freshbuild_structure_verify_derisk \
      --seeds 42 --json research/findings/raw/_wkv_freshbuild_verify/fb_s42.json
Run (6-seed decisive, ~12-18 min):
  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_read_freshbuild_structure_verify_derisk \
      --seeds 42,43,44,100,101,102 --json research/findings/raw/_wkv_freshbuild_verify/fb_6seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from tools.lab import lever, assert_backend  # noqa: E402

from research.runners._wkv_mouth_readout_softmax_confidence_derisk import _measure_gain  # noqa: E402
from research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk import (  # noqa: E402
    BatchedSubstrateReadout, _thr_hash, _native,
)
from research.runners._wkv_mouth_readout_eprop_learn_derisk import _positions  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, _load_eval  # noqa: E402
from research.runners._wkv_mouth_readout_structure_characterization_derisk import (  # noqa: E402
    _rescale, _row_topk_sparsify, _activity_pca, _project,
)

STRUCT_OK = 0.5      # a structured probe reading ABOVE this on a fresh build is "faithfully readable"
WALL_CEIL = 0.15     # the ARTIFACT's stale 2nd/3rd read must sit AT/BELOW this (reproduces the published wall)


def _new_sb(ro, seed, args):
    return BatchedSubstrateReadout(ro, seed, args.batch, hid_pop=args.sub_hid_pop, pop=args.sub_pop,
                                   ou_std=args.ou_std, read_window=args.sub_read_window, hid_gain=args.hid_gain,
                                   ratio=args.ratio, settle_frac=args.settle_frac, n_bias=args.n_bias,
                                   bias_drive_pA=args.bias_drive_pA)


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
    print(f"[seed-trap {seed}] {h1} == {h2} -> {'SEEDED' if seeded else 'NOT SEEDED'}", flush=True)

    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(0.8 * len(usable))
    train_ids = usable[:cut]
    H, _, _ = _positions(ro, train_ids, args.warmup, max(args.n_pca_pos, args.batch))
    feats = H[:args.batch]
    hw = ro.head_w.astype(np.float64)
    T = float(np.linalg.norm(hw))

    # --- build the structured probe bank (all rescaled to the SAME norm T; structure is the only varying axis) ---
    rng = np.random.default_rng(seed * 7 + 3)
    Wa = 0.12 * rng.standard_normal((ro.V, ro.D))            # a random base (same draw family as the char runner)
    sparse_hw = _rescale(_row_topk_sparsify(Wa, 8), T)       # sparse/low-entropy structure (random base, top-8/row)
    eigvecs, _ = _activity_pca(H)
    eigen_top = _rescale(_project(hw, eigvecs, 8, bottom=False), T)   # head_w confined to the top-8 activity PCs
    hw_dir = _rescale(hw, T)                                  # head_w direction at the common norm

    def measure(sb, W):
        res, _, _ = _measure_gain(sb, W, feats)
        return float(res["corr"])

    # ================= ARTIFACT side: ONE shared build, multiple probes (the published protocol) =================
    sb = _new_sb(ro, seed, args)
    rnd0 = _rescale(0.12 * np.random.default_rng(seed * 7919 + 3).standard_normal((ro.V, ro.D)), T)
    rnd2 = _rescale(0.12 * np.random.default_rng(seed * 7919 + 205).standard_normal((ro.V, ro.D)), T)
    art_rand_first = measure(sb, rnd0)        # 1st read -> faithful (~0.95)
    art_headw_2nd = measure(sb, hw_dir)       # 2nd read -> STALE (transmits rnd0) -> ~0
    art_rand_3rd = measure(sb, rnd2)          # 3rd read (fresh random) -> STALE -> ~0
    del sb
    print(f"  [seed {seed}] ARTIFACT(shared build): rand_1st={art_rand_first:+.3f} headw_2nd={art_headw_2nd:+.3f} "
          f"rand_3rd={art_rand_3rd:+.3f}", flush=True)

    # ================= TRUTH side: a FRESH build per probe (COO cache built from the loaded weights) =================
    truth = {}
    probes = [
        ("random_a", _rescale(0.12 * np.random.default_rng(seed * 7919 + 3).standard_normal((ro.V, ro.D)), T)),
        ("random_b", _rescale(0.12 * np.random.default_rng(seed * 7919 + 104).standard_normal((ro.V, ro.D)), T)),
        ("random_c", _rescale(0.12 * np.random.default_rng(seed * 7919 + 205).standard_normal((ro.V, ro.D)), T)),
        ("headw", hw_dir),
        ("sparse_structured", sparse_hw),
        ("eigen_top_structured", eigen_top),
    ]
    for tag, W in probes:
        sb = _new_sb(ro, seed, args)
        truth[tag] = measure(sb, W)
        del sb
        print(f"  [seed {seed}] TRUTH fresh-build {tag:22s} corr={truth[tag]:+.3f}", flush=True)

    struct_tags = ["headw", "sparse_structured", "eigen_top_structured"]
    rand_tags = ["random_a", "random_b", "random_c"]
    struct_min = min(truth[t] for t in struct_tags)
    rand_mean = float(np.mean([truth[t] for t in rand_tags]))

    # per-seed characterization levers (non-required)
    lever(f"headw_stale_vs_freshbuild_seed{seed}", before=art_headw_2nd, after=truth["headw"],
          required=False, continuous=truth["headw"] - art_headw_2nd)

    seed_truth_ok = bool(struct_min > STRUCT_OK)
    seed_artifact_reproduced = bool(max(art_headw_2nd, art_rand_3rd) <= WALL_CEIL and art_rand_first > STRUCT_OK)

    return {
        "seed": seed, "V": ro.V, "D": ro.D, "batch": args.batch, "sub_read_window": args.sub_read_window,
        "target_norm_T": round(T, 3), "seeded": seeded,
        "artifact": {"rand_1st": round(art_rand_first, 4), "headw_2nd": round(art_headw_2nd, 4),
                     "rand_3rd": round(art_rand_3rd, 4)},
        "truth": {k: round(v, 4) for k, v in truth.items()},
        "struct_min_freshbuild": round(struct_min, 4), "rand_mean_freshbuild": round(rand_mean, 4),
        "seed_truth_ok": seed_truth_ok, "seed_artifact_reproduced": seed_artifact_reproduced,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=40000)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sub-hid-pop", type=int, default=4)
    ap.add_argument("--sub-pop", type=int, default=1)
    ap.add_argument("--sub-read-window", type=int, default=64)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-pca-pos", type=int, default=600)
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_freshbuild_verify/fb.json")
    args = ap.parse_args()

    assert_backend(os.environ.get("SIM_BACKEND", "cupy"),
                   note="(fresh-build-per-probe verify on the PRODUCTION cupy substrate; B=8/window=64 host-RAM-safe)")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    t0 = time.time()
    rows = []
    for seed in seeds:
        r = run_seed(seed, args)
        if r is not None:
            rows.append(r)

    summary = {}
    if rows:
        n = len(rows)
        bar = 5 if n >= 6 else max(1, (n + 1) // 2)
        n_truth_ok = sum(1 for r in rows if r["seed_truth_ok"])
        n_artifact = sum(1 for r in rows if r["seed_artifact_reproduced"])
        summary = {
            "n_seeds": n, "bar": bar,
            "n_seeds_structured_faithful_freshbuild": n_truth_ok,
            "n_seeds_artifact_reproduced": n_artifact,
            "headw_freshbuild_corr_mean": round(float(np.mean([r["truth"]["headw"] for r in rows])), 4),
            "sparse_freshbuild_corr_mean": round(float(np.mean([r["truth"]["sparse_structured"] for r in rows])), 4),
            "eigen_top_freshbuild_corr_mean": round(float(np.mean([r["truth"]["eigen_top_structured"] for r in rows])), 4),
            "random_freshbuild_corr_mean": round(float(np.mean([r["rand_mean_freshbuild"] for r in rows])), 4),
            "headw_stale_2nd_corr_mean": round(float(np.mean([r["artifact"]["headw_2nd"] for r in rows])), 4),
            "rand_1st_corr_mean": round(float(np.mean([r["artifact"]["rand_1st"] for r in rows])), 4),
        }
        summary["verdict"] = ("ARTIFACT-CONFIRMED-READ-FAITHFUL"
                              if (n_truth_ok >= bar and n_artifact >= bar) else
                              ("READ-FAITHFUL-ARTIFACT-NOT-REPRODUCED" if n_truth_ok >= bar else "INCONCLUSIVE"))
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)

    out = {"results": _native(rows), "summary": _native(summary), "seeds": seeds,
           "struct_ok_threshold": STRUCT_OK, "wall_ceiling": WALL_CEIL,
           "root_cause": "BatchedSubstrateReadout reuses one built bridge across set_weights; transmission reads "
                         "sim/bridge.py::_get_cached_coo() which is invalidated only on structural (not weight) "
                         "change -> every read after the first transmits the FIRST-loaded weight matrix. Fresh build "
                         "per probe = COO cache built from the loaded weights = faithful read.",
           "backend": os.environ.get("SIM_BACKEND", "cupy"), "elapsed_s": round(time.time() - t0, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"[done] {len(rows)} seeds -> {args.json} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
