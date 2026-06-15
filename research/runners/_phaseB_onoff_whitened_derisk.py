"""Phase-B escape de-risk (numpy): the RETINAL mechanism. The rate->spike wall = the real whitened structure
is a signed, low-magnitude differential that rate-coded spiking (magnitude, non-negative) cannot carry. The
retina's solution: analog center-surround WHITENING (remove the common mode pre-spike, full-precision) +
ON/OFF spiking (split the signed whitened signal into two NON-NEGATIVE populations -> the signed pattern is
carried by which-of-ON/OFF fires). This is brain-canonical + Mikulasch-Priesemann-aligned (whitening is
analog/pre-spike; ON/OFF is the retina's signed-to-spike code).

Test (real corpus): POINT spiking of the raw drive (the wall) vs ON/OFF spiking of the WHITENED drive. The
whitening is a HOST instrument here (X - X.mean(0)); if it works, the neural realization = dendritic
center-surround + ON/OFF cells. GO ⇒ analog-whitening + ON/OFF escapes the rate->spike wall -> build it.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_onoff_whitened_derisk
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners._l1_centered_online_pca_probe import oja_subspace  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def run(label, C, labels, S_true, gain=20.0, k=128):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    X = ppmi_matrix(C, 0.75)
    host, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape)-1), alpha=0.75), labels)
    H = X.shape[1]; rng = np.random.RandomState(7)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, H) / np.sqrt(H)

    # (A) POINT: project the raw (common-mode-dominated) drive, spike-count readout (the rate->spike wall)
    pt_rate = np.maximum(W @ Xn.T, 0.0).T
    pt = np.array([poisson_spk(pt_rate[i], gain, rng) for i in range(len(X))])
    pt_p = _pearson_vs_Strue(_cos_sim(pt), S_true)

    # (B) ON/OFF of the WHITENED drive: analog whiten the INPUT (remove the per-hub common mode, full
    #     precision, pre-spike), project, then split into ON(+)/OFF(-) spike populations.
    Xw = Xn - Xn.mean(0, keepdims=True)                 # analog center-surround whitening (the dendrite)
    drive = (W @ Xw.T).T                                 # projected whitened drive (signed)
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(X))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(X))])
    onoff = np.concatenate([on, off], axis=1)            # the ON/OFF spike code
    oo_p = _pearson_vs_Strue(_cos_sim(onoff), S_true)
    gen, ch = heldout_generalization(onoff, labels)
    perml = rng.permutation(labels); S_perm = (perml[:, None] == perml[None, :]).astype(float)
    perm_p = _pearson_vs_Strue(_cos_sim(onoff), S_perm)
    # (C) LEARNED ON/OFF: an Oja cortex LEARNS on the ON/OFF spike code (the retinal front-end + a learned
    #     cortex). This is the full brain-based escape: analog whitening -> ON/OFF cells -> learned cortex.
    learned = oja_subspace(onoff, k, 300, 0.01, 7)
    lrn_p = _pearson_vs_Strue(_cos_sim(learned), S_true)
    lg, _ = heldout_generalization(learned, labels)
    print(f"  [{label}] host=+{host:.3f}  POINT-spike={pt_p:+.3f}  ON/OFF-whitened={oo_p:+.3f}  "
          f"ON/OFF+LEARNED={lrn_p:+.3f} (gen {lg:.3f}/ch {ch:.3f}, perm {perm_p:+.3f})", flush=True)
    return pt_p, lrn_p, host


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    print("[ON/OFF whitened de-risk] does analog-whitening + ON/OFF spiking carry the real structure point-"
          "spiking loses?", flush=True)
    Cs, ls, Ss, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, 42)
    run("SYNTHETIC", Cs, ls, Ss)
    Cr, lr, Sr = build_real_corpus(42, 500)
    # spike-budget sweep on REAL: is the precision wall a spike-COUNT (budget) issue, or fundamental?
    print("  [REAL spike-budget sweep: gain = spikes/unit -> does MORE spikes (less Poisson noise) recover it?]",
          flush=True)
    for g in (20.0, 100.0, 500.0, 2000.0):
        run(f"REAL g{int(g):4d}", Cr, lr, Sr, gain=g)
    pt, oo, host = run("REAL    ", Cr, lr, Sr)
    print("\n  VERDICT:", flush=True)
    if oo >= 0.30 and oo >= pt + 0.10:
        print(f"  GO -- ON/OFF spiking of the WHITENED drive carries the real structure ({oo:+.3f}) where POINT "
              f"spiking loses it ({pt:+.3f}). The retinal escape (analog center-surround whitening + ON/OFF "
              f"cells) BEATS the rate->spike wall -> build it on the bridge (a 2-pop ON/OFF cortex + a "
              f"whitening front-end; brain-canonical, cheaper than full multi-compartment). The whitening here "
              f"is a host instrument; its neural realization = dendritic center-surround / lateral inhibition.",
              flush=True)
    elif oo >= pt + 0.10:
        print(f"  PARTIAL -- ON/OFF-whitened beats point ({oo:+.3f} vs {pt:+.3f}) but < +0.30; the mechanism "
              f"helps, needs the learned projection (not random) to reach the ceiling. Build the learned "
              f"ON/OFF cortex.", flush=True)
    else:
        print(f"  NEGATIVE -- ON/OFF-whitened ({oo:+.3f}) does not beat point ({pt:+.3f}); the retinal escape "
              f"does not rescue it -> the loss is deeper than the signed-whitened-to-spike code.", flush=True)


if __name__ == "__main__":
    main()
