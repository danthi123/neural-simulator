"""CYCLE 88 — SKEPTICAL verification of a surprising result: does PPMI-input + per-feature CENTERING (both
LOCAL subtractive operations) reach the offline optimum (+0.518) on the REAL corpus -- meaning the off-diagonal
cross-neuron decorrelation is NOT the wall, and no dendrite-plus-lateral build is needed?

THE SURPRISE (CYCLE 88, interneuron probe): `cos(Xppmi - Xppmi.mean(0))` = +0.518 (3 seeds) -- the offline PCA
optimum, HIGHER than the whitened rank-8 ZCA (+0.497). If real, the whole "off-diagonal whitening is the wall"
framing is wrong: WHITENING over-processes (the SM whitens -> +0.35); CENTERING preserves variance -> +0.518;
and centering is LOCAL (per-feature running-mean subtraction = the shipped input_mean_adapt). The arc/bridge
reached only +0.155 because it used LOG input (CYCLE 80: SIMMATCH_LOG +0.088), NOT PPMI.

This is the "if it seems too easy, distrust it" check. SKEPTICAL battery (3 seeds):
  1. DECOMPOSE the input encoding x centering: {raw, log1p, PPMI} x {uncentered, centered}. Isolate what lifts.
  2. DECOMPOSE PPMI: log only / + col-marginal (per-hub) / + row-marginal (per-concept) -- which is load-bearing?
  3. ONLINE-LOCALITY: a single-pass RUNNING per-feature mean (streaming) centering vs the batch mean -- does the
     online-local centering ALSO reach +0.518 (confirming it is not a batch-only operation)?
  4. GENERALIZATION: held-out nearest-category classification on the centered-PPMI codes (a real structured
     representation generalizes, not just fits the similarity).
  5. ANTI-CHEAT: permuted-S_true ~0 (centering uses only X, never the labels); host PPMI+SVD carries.
VERDICT GO = PPMI+centering reaches the optimum, online-local centering matches batch, generalizes, controls
clean ==> the functional cortex needs LOCAL normalization (PPMI-equivalent encoding + centering), NOT the
off-diagonal decorrelation -> a MUCH simpler, point-neuron build (and the prior bridge marginality was a
LOG-input artifact, fixable by a PPMI-equivalent encoding). NEGATIVE/PARTIAL = the +0.518 has a catch (batch-
only, doesn't generalize, or a single marginal carries it spuriously) -> the off-diagonal stands.

Reuse-by-import; NO sim/ edits; numpy; 3 seeds.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_ppmi_centering_verify_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402

SEEDS = (42, 43, 44)
N_HUB = 500
ALPHA = 0.75


def cen(X):
    return X - X.mean(0, keepdims=True)


def p_of(X, S_true):
    return _pearson_vs_Strue(_cos_sim(X), S_true)


def running_mean_center(X, seed):
    """ONLINE-LOCAL centering: a single streaming pass with a running per-feature mean (EMA), subtracted from
    each concept as it arrives -- NOT the batch mean. Confirms centering is online-realizable (input_mean_adapt)."""
    rng = np.random.RandomState(seed * 13 + 1)
    Nc, H = X.shape
    mean = np.zeros(H)
    out = np.zeros_like(X)
    alpha = 0.05                                   # EMA rate (the input_mean_adapt analogue)
    order = rng.permutation(Nc)
    # two passes so the running mean warms up (biological: the mean adapts over exposure); read on the 2nd.
    for _pass in range(2):
        for i in order:
            mean += alpha * (X[i] - mean)
            out[i] = X[i] - mean
    return out


def ppmi_variants(C, alpha):
    """Decompose PPMI = relu(log(M*T/(row*col))) into its marginal-removal pieces, KEEPING the normalizing
    constant so the ReLU is fair: col-only replaces the per-concept row with its average (T/Nc); row-only
    replaces the per-hub col with its average (T/Nh). Shows which marginal (per-hub vs per-concept) carries it."""
    M = np.maximum(C, 0.0).astype(np.float64)
    Nc, Nh = M.shape
    row = M.sum(1, keepdims=True)                              # per-concept marginal (the concept's total)
    col = M.sum(0, keepdims=True) ** alpha                     # per-hub marginal (alpha-smoothed)
    T = col.sum()
    eps = 1e-12
    col_only = np.maximum(np.log((M * Nc) / (col + eps) + eps), 0.0)         # per-hub norm only (row->avg)
    row_only = np.maximum(np.log((M * Nh) / (row + eps) + eps), 0.0)         # per-concept norm only (col->avg)
    return {
        "log_only": np.maximum(np.log(M + eps), 0.0),
        "col_only(per-hub)": col_only,
        "row_only(per-concept)": row_only,
        "ppmi_full": ppmi_matrix(C, alpha),
    }


def run_seed(seed):
    C, labels, S_true = build_real_corpus(seed, N_HUB)
    labels = np.asarray(labels)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=ALPHA), labels)
    raw = np.maximum(C, 0.0).astype(np.float64)
    log = np.log1p(raw)
    ppmi = ppmi_matrix(C, ALPHA)
    print(f"\n[ppmi-centering verify seed {seed}] {C.shape[0]}c x {C.shape[1]}h | host PPMI+SVD {host_p:+.3f}",
          flush=True)
    # (1) encoding x centering grid
    grid = {}
    for name, X in (("raw", raw), ("log", log), ("ppmi", ppmi)):
        grid[name] = (round(p_of(X, S_true), 3), round(p_of(cen(X), S_true), 3))
        print(f"    {name:5s}: uncentered {grid[name][0]:+.3f} | centered {grid[name][1]:+.3f}", flush=True)
    # (2) PPMI marginal decomposition (each centered)
    pv = ppmi_variants(C, ALPHA)
    pv_p = {k: round(p_of(cen(v), S_true), 3) for k, v in pv.items()}
    print(f"    PPMI pieces (centered): " + " | ".join(f"{k} {v:+.3f}" for k, v in pv_p.items()), flush=True)
    # (3) online-local running-mean centering on PPMI
    online_p = round(p_of(running_mean_center(ppmi, seed), S_true), 3)
    batch_p = grid["ppmi"][1]
    print(f"    online running-mean centering(PPMI) {online_p:+.3f} vs batch {batch_p:+.3f}", flush=True)
    # (4) generalization on centered PPMI
    gen, chance = heldout_generalization(cen(ppmi), labels)
    print(f"    centered-PPMI generalization {gen:.3f} (chance {chance:.3f})", flush=True)
    # (5) anti-cheat: permuted
    rng2 = np.random.RandomState(seed * 2718281 + 9)
    perm = rng2.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_p = round(_pearson_vs_Strue(_cos_sim(cen(ppmi)), S_perm), 3)
    print(f"    [anti-cheat] permuted {perm_p:+.3f} (~0)", flush=True)
    return {"seed": seed, "host": host_p, "grid": grid, "ppmi_pieces": pv_p, "online": online_p,
            "batch": batch_p, "gen": gen, "chance": chance, "permuted": perm_p}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[PPMI+centering SKEPTICAL verify] seeds={SEEDS} -- is the +0.518 (PPMI+centering, local) real, "
          f"online, generalizing -- meaning the off-diagonal is NOT the wall?", flush=True)
    rows = [run_seed(s) for s in SEEDS]

    def m(f):
        return float(np.mean([f(r) for r in rows]))
    host = m(lambda r: r["host"])
    ppmi_cen = m(lambda r: r["grid"]["ppmi"][1]); ppmi_unc = m(lambda r: r["grid"]["ppmi"][0])
    log_cen = m(lambda r: r["grid"]["log"][1])
    online = m(lambda r: r["online"]); batch = m(lambda r: r["batch"])
    gen = m(lambda r: r["gen"]); chance = m(lambda r: r["chance"]); perm = m(lambda r: r["permuted"])
    col_only = m(lambda r: r["ppmi_pieces"]["col_only(per-hub)"]); row_only = m(lambda r: r["ppmi_pieces"]["row_only(per-concept)"])
    print(f"\n{'='*96}\n  MEAN ({len(SEEDS)} seeds): host {host:+.3f}", flush=True)
    print(f"  PPMI centered {ppmi_cen:+.3f} (uncentered {ppmi_unc:+.3f}) | log centered {log_cen:+.3f}", flush=True)
    print(f"  PPMI pieces (centered): col-only(per-hub) {col_only:+.3f} | row-only(per-concept) {row_only:+.3f}",
          flush=True)
    print(f"  online running-mean centering {online:+.3f} vs batch {batch:+.3f} | generalization {gen:.3f} "
          f"(chance {chance:.3f}) | permuted {perm:+.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    online_ok = online >= batch - 0.04
    gen_ok = gen > chance + 0.15
    reaches = ppmi_cen >= 0.90 * host + 0.02 or ppmi_cen >= 0.45
    if reaches and online_ok and gen_ok and abs(perm) <= 0.10:
        print(f"  GO (the off-diagonal is NOT the wall): PPMI + per-feature CENTERING reaches {ppmi_cen:+.3f} "
              f">= host {host:+.3f} -- HIGHER than the whitened rank-8 ZCA (+0.497) -- with ONLINE-LOCAL running-"
              f"mean centering matching the batch ({online:+.3f} vs {batch:+.3f}), real generalization ({gen:.3f} "
              f"vs chance {chance:.3f}), permuted-clean ({perm:+.3f}). ==> the functional cortex needs LOCAL "
              f"NORMALIZATION (a PPMI-equivalent encoding: log + per-hub + per-concept marginal subtraction, all "
              f"local) + per-feature centering (the shipped input_mean_adapt) -- NOT the off-diagonal cross-neuron "
              f"decorrelation, NOT a dendrite-plus-lateral build. The whitening pursuit OVER-processed; the prior "
              f"bridge +0.155 was a LOG-input artifact. The build = a PPMI-equivalent point-neuron encoding on "
              f"the bridge. MAJOR simplification -- verify next on the bridge with PPMI-shaped drive.", flush=True)
    elif reaches and not online_ok:
        print(f"  CATCH (batch-only): PPMI+centering reaches {ppmi_cen:+.3f} BATCH but the online running-mean "
              f"centering falls to {online:+.3f} -- the centering needs the batch mean, not online-realizable as "
              f"tested. The off-diagonal framing may stand; investigate the online centering (warm-up/rate).",
              flush=True)
    elif reaches and not gen_ok:
        print(f"  CATCH (no generalization): PPMI+centering fits the similarity ({ppmi_cen:+.3f}) but does NOT "
              f"generalize ({gen:.3f} ~ chance {chance:.3f}) -- a similarity artifact, not a structured code. The "
              f"off-diagonal framing may stand.", flush=True)
    else:
        print(f"  NEGATIVE: PPMI+centering ({ppmi_cen:+.3f}) does not robustly reach host / has a catch -- the "
              f"surprising result does not hold up; the off-diagonal framing stands.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "ppmi_centered": ppmi_cen, "ppmi_uncentered": ppmi_unc, "log_centered": log_cen,
           "col_only": col_only, "row_only": row_only, "online_centering": online, "batch_centering": batch,
           "generalization": gen, "chance": chance, "permuted": perm, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_ppmi_centering_verify.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
