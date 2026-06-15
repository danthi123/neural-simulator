"""CYCLE 93 — the LOG-DOMAIN cortex on the bridge: the hub f-I provides the log; does the population firing +
double-centering (per-hub + per-concept subtractions IN THE FIRING/LOG domain) reach the +0.410 numpy target?

CYCLE 92 NEGATIVE root-caused: PPMI's normalizations are LOG-SUBTRACTIVE (must come AFTER a log), but the
committed divisive + input_mean primitives subtract/divide in the CURRENT domain (pre-f-I) = wrong order.
The numpy de-risk of the RIGHT order (log FIRST, then subtract per-hub + per-concept means in the log domain)
reaches +0.410 (82% of PPMI, generalizes 0.86; ReLU HURTS so keep the signed code; per-concept centering is the
dominant lever). On the bridge: the hub f-I (firing-rate vs drive) IS the log compression -- drive hubs with
RAW counts, read the population firing (~log(count)), then double-center it. The centering is host-computed here
as the cheap-first de-risk (a scaffold for the neural normalization layer: per-hub = spike-frequency adaptation
in the firing domain, per-concept = a feedforward-inhibition population-mean subtraction).

ARMS (real corpus, population code; GPU):
  numpy target (LOG+double-center)   +0.410 (the goal)
  BRIDGE raw-firing, NO centering    the floor (common hubs dominate)
  BRIDGE raw-firing + double-center  <- the test: does the f-I-log + centering reach the numpy +0.410?
GATE: bridge-firing + double-center >= 0.70x the numpy +0.410 AND beats the no-centering floor -> the hub f-I
provides the log + the firing-domain centering recovers PPMI -> the LOG-DOMAIN circuit is the right on-bridge
realization; build the neural centering layer (adaptation + feedforward inhibition) next.

Reuse-by-import (build_cortex + present from the neural-norm probe); GPU. NO sim/ edits.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_logdomain_cortex_derisk --seeds 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue, heldout_generalization  # noqa: E402
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners._phaseB_neural_norm_cortex_derisk import build_cortex, present  # noqa: E402


def double_center(X):
    """LOG-domain normalization: subtract per-feature (hub) mean + per-sample (concept) mean. The firing-domain
    realization of PPMI's per-hub + per-concept log-subtractions."""
    return X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean()


def pop_codes(seed, drive, n_dim, n_pop, scale, window, settle):
    b, idx = build_cortex(n_dim, n_pop, seed, False, 0.0, 1.0, 1.0)   # no neural-norm; pure f-I firing
    out = np.zeros((drive.shape[0], n_dim))
    dp = np.repeat(drive, n_pop, axis=1)
    for i in range(drive.shape[0]):
        out[i] = present(b, idx, dp[i], scale, window, settle).reshape(n_dim, n_pop).mean(1)
    return out


def run_seed(seed, a):
    C, labels, S_true = build_real_corpus(seed, a.n_hub)
    labels = np.asarray(labels); n_dim = C.shape[1]
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)
    raw = np.maximum(C, 0.0).astype(np.float64)
    numpy_target = _pearson_vs_Strue(_cos_sim(double_center(np.log1p(raw * 100.0))), S_true)
    fr = pop_codes(seed, raw, n_dim, a.n_pop, a.raw_scale, a.window, a.settle)   # bridge population firing (~log)
    floor = _pearson_vs_Strue(_cos_sim(fr), S_true)                              # firing, NO centering
    centered = double_center(fr)
    dc = _pearson_vs_Strue(_cos_sim(centered), S_true)                           # firing + double-center
    gen, ch = heldout_generalization(centered, labels)
    print(f"\n[log-domain seed {seed}] {C.shape[0]}c x {n_dim}d x {a.n_pop}/dim | host {host_p:+.3f} | "
          f"numpy LOG+double-center {numpy_target:+.3f}", flush=True)
    print(f"  BRIDGE raw-firing NO-center {floor:+.3f} | BRIDGE firing + DOUBLE-CENTER {dc:+.3f} (gen {gen:.2f}) "
          f"= {dc/max(numpy_target,1e-9):.0%} of numpy target", flush=True)
    return {"seed": seed, "host": host_p, "numpy_target": numpy_target, "floor": floor, "double_center": dc,
            "gen": gen}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--n-pop", type=int, default=16)
    p.add_argument("--raw-scale", type=float, default=4.0)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--settle", type=int, default=8)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[log-domain cortex de-risk] seeds={seeds} n_pop={a.n_pop} raw_scale={a.raw_scale} -- does the hub f-I "
          f"(log) + firing-domain double-centering reach the numpy +0.41 target?", flush=True)
    rows = [run_seed(s, a) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, tgt, floor, dc = m("host"), m("numpy_target"), m("floor"), m("double_center")
    frac = dc / tgt if tgt > 1e-9 else 0.0
    print(f"\n{'='*96}\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | numpy LOG+double-center target {tgt:+.3f} | "
          f"BRIDGE firing floor {floor:+.3f} | BRIDGE firing+double-center {dc:+.3f} ({frac:.0%} of target)", flush=True)
    print(f"{'='*96}", flush=True)
    if dc >= 0.70 * tgt and dc >= floor + 0.05:
        print(f"  GO: the bridge hub f-I (log) + firing-domain double-centering REACHES the numpy target "
              f"({dc:+.3f} = {frac:.0%} of {tgt:+.3f}, beats the no-center floor {floor:+.3f}). ==> the LOG-DOMAIN "
              f"circuit is the right on-bridge realization (f-I provides the log; centering in the FIRING domain "
              f"works where current-domain failed). Build the NEURAL centering layer next (per-hub spike-freq "
              f"adaptation + per-concept feedforward-inhibition population-mean subtraction).", flush=True)
    elif dc >= floor + 0.05:
        print(f"  PARTIAL: firing+double-center ({dc:+.3f}) beats the floor ({floor:+.3f}) but reaches {frac:.0%} "
              f"of the numpy target ({tgt:+.3f}) -- the f-I isn't a clean log / the population resolution caps it; "
              f"tune raw_scale (the f-I log operating point) / n_pop / window.", flush=True)
    else:
        print(f"  NEGATIVE: firing+double-center ({dc:+.3f}) ~ the floor ({floor:+.3f}) -- the bridge firing isn't "
              f"log-like enough for the centering to recover PPMI; the f-I log operating point needs work (or an "
              f"explicit log/dendritic compression).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "numpy_target": tgt, "floor": floor, "double_center": dc, "frac": frac, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_logdomain_cortex.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
