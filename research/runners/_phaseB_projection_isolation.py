"""Phase-B projection isolation: the structure survives a NUMPY (rate) projection of the bridge's hub codes
(+0.34) but NOT the bridge's hub->cortex projection (g_e ≈ −0.06). WHERE in the bridge projection is it lost?
Read the cortex g_e (analog conductance, NOT spikes -- isolates the projection from the cortex threshold)
across: density {0.5 sparse, 1.0 dense}, dendritic gain {on, off}, at the strong-drive encoding regime.
If a DENSE + gain-OFF g_e recovers toward the numpy +0.34 -> the loss was the sparse-W / gain (tunable
ENGINEERING, not fundamental). If even dense+gain-off g_e stays ≈0 -> the spiking-conductance projection
itself is lossy (more fundamental). Decides tunable-vs-fundamental for the honest finding. NO sim/ edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_projection_isolation
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue,
)
from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive  # noqa: E402
from research.runners._phaseB_c1b_derisk_perneuron_centering import read_ge_codes  # noqa: E402


def _both(codes, S_true):
    p = _pearson_vs_Strue(_cos_sim(codes), S_true)
    pc = _pearson_vs_Strue(_cos_sim(codes - codes.mean(0, keepdims=True)), S_true)
    return p, pc


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"[projection isolation] {C.shape[0]}c x {n_hub}h; numpy-proj of hub codes ~ +0.34. Reading cortex g_e:",
          flush=True)

    for dens in (0.5, 1.0):
        for sigma, gain_tag in ((0.05, "gain-ON"), (1e9, "gain-OFF~")):
            # sigma huge -> g = sigma/(sigma+a) ~ 1 -> the dendritic gain is effectively OFF.
            b, hub, cx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42, density=dens,
                                                weight_mean=80.0, sigma=sigma, stdp_w_max=200.0)
            ge = read_ge_codes(b, C_drive, np.asarray(hub), np.asarray(cx),
                               drive_scale=40.0, window=150, settle=8)
            p, pc = _both(ge, S_true)
            print(f"  [density={dens:.1f}  {gain_tag:9s}]  g_e cos={p:+.3f}  per-neuron-centered={pc:+.3f}",
                  flush=True)

    print("\n  If a row reaches ~+0.2..+0.3 -> the bridge projection CAN preserve it (tunable engineering: "
          "density/gain/weights) -> NOT the fundamental dendritic wall. If ALL rows stay ~0/negative -> the "
          "spiking-conductance hub->cortex projection is intrinsically lossy here.", flush=True)


if __name__ == "__main__":
    main()
