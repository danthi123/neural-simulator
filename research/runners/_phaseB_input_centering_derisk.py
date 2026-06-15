"""Phase-B WHERE-not-whether de-risk: the WALL attacked the common mode at the CORTEX (rank-1 cm-pool /
per-cortex-neuron OUTPUT centering). The L1 op centers the INPUT, PER-HUB, BEFORE the projection
(`X − X.mean(0)`) — a locus NEVER tried on the bridge. At the INPUT each hub's common mode is a SINGLE
SCALAR (the hub's mean activity), which a per-hub point-neuron interneuron CAN subtract (the rank-1 limit
only bites when subtracting at the cortex AFTER the random mixing). This tests whether centering the bridge's
LOSSY spiking HUB codes per-hub, BEFORE the projection, recovers the structure the output-centering could not.

Anti-cheats (per the whitening research): (1) WHERE-not-WHETHER — input-centering must beat the WALL's
output-centering (≈−0.09); (2) rank-1-not-per-dim — a scalar (population-mean) subtraction must NOT recover
it (only the per-hub vector does); (3) learned-not-host — an ONLINE per-hub EMA (not the oracle X.mean(0))
must also work (the neural realization). GO ⇒ the cheap input-side per-hub subtractive feedforward inhibition
(a small guarded sim/ edit reusing cp_dendritic_source_activity) is the fix, NOT the months-scale substrate.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_input_centering_derisk
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
from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive  # noqa: E402
from research.runners._phaseB_hub_encoding_regime import read_hub_codes  # noqa: E402
from research.runners._l1_centered_online_pca_probe import oja_subspace  # noqa: E402


def _p(codes, S_true):
    return _pearson_vs_Strue(_cos_sim(codes), S_true)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]

    # Read the bridge's LOSSY spiking HUB codes at a firing regime that encodes (ds40, the regime probe).
    b, hub, cx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=64, seed=42, density=0.5, weight_mean=80.0,
                                        stdp_w_max=200.0)
    X = read_hub_codes(b, C_drive, np.asarray(hub), drive_scale=40.0, window=150, settle=8)  # [Nc x H_hub]
    print(f"[input-centering de-risk] {C.shape[0]} concepts x {n_hub} hubs; bridge HUB spike codes "
          f"(uncentered cos {_p(X, S_true):+.3f}, per-hub-centered {_p(X - X.mean(0, keepdims=True), S_true):+.3f})",
          flush=True)

    k, seed = 64, 7
    rng = np.random.RandomState(seed)
    W = rng.randn(k, n_hub) / np.sqrt(n_hub)

    def proj(Z):
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return (W @ Zn.T).T

    Xc = X - X.mean(0, keepdims=True)                              # per-hub (per-dimension) input centering
    Xscalar = X - X.mean()                                        # rank-1 / scalar (population) subtraction
    # online per-hub EMA (the LEARNED, neural realization -- not the oracle X.mean(0))
    ema = np.zeros(n_hub); a = 0.05; Xema = np.zeros_like(X)
    order = rng.permutation(len(X))
    for _ in range(20):
        for i in order:
            ema = (1 - a) * ema + a * X[i]
            Xema[i] = X[i] - ema

    print("  --- random projection of the hub codes (the WHERE comparison) ---", flush=True)
    out_wall = proj(X)
    out_wall_outc = out_wall - out_wall.mean(0, keepdims=True)     # OUTPUT centering = the WALL's locus
    print(f"  [WALL: uncentered-input -> proj]            {_p(out_wall, S_true):+.3f}", flush=True)
    print(f"  [WALL: + OUTPUT centering (post-proj)]      {_p(out_wall_outc, S_true):+.3f}  (the -0.09 the 6 probes hit)", flush=True)
    print(f"  [FIX:  INPUT per-hub centering -> proj]     {_p(proj(Xc), S_true):+.3f}  <-- the untried locus", flush=True)
    print(f"  [ctrl: rank-1 scalar subtraction -> proj]   {_p(proj(Xscalar), S_true):+.3f}  (must NOT recover)", flush=True)
    print(f"  [ctrl: ONLINE per-hub EMA -> proj]          {_p(proj(Xema), S_true):+.3f}  (learned, not oracle)", flush=True)

    print("  --- LEARNED projection (Oja, the L1 recipe) on the input-centered hub codes ---", flush=True)
    learned_wall = oja_subspace(X, k, 300, 0.01, seed)
    learned_fix = oja_subspace(Xc, k, 300, 0.01, seed)
    learned_ema = oja_subspace(Xema, k, 300, 0.01, seed)
    gw, ch = heldout_generalization(learned_wall, labels)
    gf, _ = heldout_generalization(learned_fix, labels)
    print(f"  [learned: uncentered input]   {_p(learned_wall, S_true):+.3f} (gen {gw:.3f})", flush=True)
    print(f"  [learned: INPUT-centered]     {_p(learned_fix, S_true):+.3f} (gen {gf:.3f}, chance {ch:.3f})  <-- the fix + learning", flush=True)
    print(f"  [learned: ONLINE-EMA input]   {_p(learned_ema, S_true):+.3f}", flush=True)

    fix = _p(proj(Xc), S_true); wall = _p(out_wall_outc, S_true)
    ema_p = _p(proj(Xema), S_true); scalar_p = _p(proj(Xscalar), S_true); lf = _p(learned_fix, S_true)
    print("\n  VERDICT:", flush=True)
    if fix >= wall + 0.15 and fix >= 0.15 and ema_p >= 0.10 and scalar_p < fix - 0.10:
        print(f"  GO -- INPUT per-hub centering recovers the structure ({fix:+.3f}) where OUTPUT centering "
              f"failed ({wall:+.3f}); the online per-hub EMA works ({ema_p:+.3f}, learned not oracle) and the "
              f"rank-1 scalar does NOT ({scalar_p:+.3f}). With learning: {lf:+.3f}. ⇒ the WALL was a WHERE bug "
              f"(centered at the cortex, not the input). The cheap fix = input-side per-hub subtractive "
              f"feedforward inhibition (a small guarded sim/ edit reusing cp_dendritic_source_activity) -- NOT "
              f"the months-scale dendritic substrate. Next: implement + bridge-validate.", flush=True)
    else:
        print(f"  NOT CLEARED -- input per-hub centering {fix:+.3f} (vs output {wall:+.3f}, scalar {scalar_p:+.3f}, "
              f"ema {ema_p:+.3f}, learned {lf:+.3f}); the locus reframe does not rescue it on the bridge's lossy "
              f"hub codes -> the wall is deeper than the centering locus.", flush=True)


if __name__ == "__main__":
    main()
