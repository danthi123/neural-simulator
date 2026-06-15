"""Phase-B Task-4 (the decisive GO confirmation): the LEARNED spiking cortex on the REAL Option-C TinyStories
corpus, in the DENSE-firing regime that cracked the spike-count boundary. On SYNTHETIC the dense random
projection already generalized (the structure is in the uncentered input), so STDP learning looked not
load-bearing. On the REAL corpus L1 showed learning IS load-bearing (learned +0.48 vs random +0.17) -- the
real category structure is NOT trivially in the input, so a random projection should FAIL and the learned
cortex should win. This is the test that distinguishes "a dense random readout" from "a learned spiking
cortex".

Gate (GO): LEARNED Pearson(cos(codes),S_true) >= 0.70*host_ceiling AND >= +0.30; beats RANDOM-PROJ by >=+0.10
(learning load-bearing); permuted ~0; generalizes above chance. PARTIAL/NEGATIVE = honest (the dense readout
solved the spike transmission; whether STDP learns the real structure is the open question). GPU for the real
bridge. NO sim/ edit.

Run: SIM_BACKEND=cupy python -u -m research.runners._phaseB_real_dense_gate --n-hub 500 --seeds 42
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization, effective_rank,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners.spiking_sm_cortex import (  # noqa: E402
    build_sm_cortex_bridge, encode_drive, train_sm_cortex, read_codes,
)


def _stat(name, codes, S_true, labels):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    g, ch = heldout_generalization(codes, labels)
    sil = float(np.mean(codes.sum(1) == 0))
    print(f"  [{name:14s}] Pearson={p:+.3f}  gen={g:.3f} (chance {ch:.3f})  silent={sil:.2f}  "
          f"eff-rank={effective_rank(codes):.1f}", flush=True)
    return p, g, ch


def run_seed(seed, args):
    C, labels, S_true = build_real_corpus(seed, args.n_hub)   # targets(64) x hubs
    C = np.asarray(C, dtype=np.float64); labels = np.asarray(labels)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    host = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75)
    host_p, _, _, _ = score(host, labels)
    print(f"\n[real dense gate seed {seed}] {C.shape[0]}c x {n_hub}h; host PPMI+SVD={host_p:+.3f}; "
          f"DENSE regime (wm{args.weight_mean}, homeo off, ds{args.drive_scale}, win{args.window})", flush=True)
    bp = dict(n_hub=n_hub, n_cortex=args.n_cortex, density=0.5, weight_mean=args.weight_mean,
              stdp_w_max=2000.0, enable_homeostasis=False)
    rp = dict(drive_scale=args.drive_scale, window=args.window, settle=8)

    t0 = time.time()
    bL, hub, cx = build_sm_cortex_bridge(seed=seed, **bp)
    hub = np.asarray(hub); cx = np.asarray(cx)
    train_sm_cortex(bL, C_drive, hub, cx, n_epochs=args.epochs, drive_scale=args.drive_scale,
                    window=args.train_window, settle=8)
    learned = read_codes(bL, C_drive, hub, cx, **rp)
    lp, lg, ch = _stat("LEARNED", learned, S_true, labels)
    bR, hubR, cxR = build_sm_cortex_bridge(seed=seed + 1, **bp)
    rand = read_codes(bR, C_drive, np.asarray(hubR), np.asarray(cxR), **rp)
    rpp, rg, _ = _stat("RANDOM-PROJ", rand, S_true, labels)
    rng = np.random.RandomState(seed * 2718281 + 1); perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = _pearson_vs_Strue(_cos_sim(learned), S_perm)
    print(f"  [anti-cheat] permuted={perm_p:+.3f} (~0); elapsed {time.time()-t0:.0f}s", flush=True)

    gates = {
        "structure": bool(lp >= max(0.30, 0.70 * host_p)),
        "permuted_collapses": bool(abs(perm_p) <= 0.15),
        "generalizes": bool(lg > ch + 0.10),
        "learning_load_bearing": bool(lp >= rpp + 0.10),
        "host_carries": bool(host_p >= 0.30),
    }
    print(f"  gates: {gates}  (learned {lp:+.3f}/gen {lg:.3f} vs random {rpp:+.3f}/gen {rg:.3f}, host {host_p:+.3f})",
          flush=True)
    return {"seed": seed, "host": host_p, "learned": lp, "learned_gen": lg, "random": rpp, "random_gen": rg,
            "permuted": perm_p, "chance": ch, "gates": gates}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--n-cortex", type=int, default=128)
    p.add_argument("--weight-mean", type=float, default=400.0)
    p.add_argument("--drive-scale", type=float, default=40.0)
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--train-window", type=int, default=200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", default="research/findings/raw/_phaseB_real_dense_gate.json")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    res = {str(s): run_seed(s, args) for s in seeds}
    allgo = all(all(res[str(s)]["gates"].values()) for s in seeds)
    lb = all(res[str(s)]["gates"]["learning_load_bearing"] for s in seeds)
    struct = all(res[str(s)]["gates"]["structure"] for s in seeds)
    if allgo:
        verdict = "GO"
    elif struct and not lb:
        verdict = "PARTIAL_dense_readout_works_learning_not_loadbearing"
    else:
        verdict = "NEGATIVE_or_PARTIAL"
    print(f"\n  REAL DENSE GATE VERDICT: {verdict}", flush=True)
    out = {"verdict": verdict, "per_seed": res, "config": vars(args)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
