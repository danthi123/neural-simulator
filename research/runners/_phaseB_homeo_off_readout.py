"""Phase-B last probe: the cortex g_e carries the structure (+0.45) but the SPIKE-COUNT code loses it (~0).
HYPOTHESIS: homeostasis EQUALIZES the cortex firing rates (adapts each neuron's threshold toward the target
rate) -> erases the g_e-MAGNITUDE structure in the spike counts. Test: read the cortex spike + g_e codes with
homeostasis ON vs OFF (a config flag, no sim/ edit). If homeostasis-OFF makes the SPIKE code track g_e (~+0.4)
-> the spike-readout loss was the homeostasis equalization = a config-level fix = the Phase-B BOUNDARY CRACKS
(GO). If even homeostasis-OFF the spike code stays ~0 while g_e stays +0.45 -> the spiking threshold itself
loses the weak common-mode-buried signal (the deeper readout problem).

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_homeo_off_readout
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, effective_rank,
)
from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive, read_codes  # noqa: E402
from research.runners._phaseB_c1b_derisk_perneuron_centering import read_ge_codes  # noqa: E402


def _c(codes, S_true):
    return _pearson_vs_Strue(_cos_sim(codes), S_true)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"[homeo-off readout] {C.shape[0]}c x {n_hub}h. Cortex g_e vs SPIKE code, homeostasis ON vs OFF:",
          flush=True)

    for homeo in (True, False):
        for ds, win in [(40.0, 150), (40.0, 300), (80.0, 300)]:
            b, hub, cx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42, density=0.5,
                                                weight_mean=80.0, stdp_w_max=200.0, enable_homeostasis=homeo)
            hub = np.asarray(hub); cx = np.asarray(cx)
            ge = read_ge_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
            sp = read_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
            ms = float(sp.sum(1).mean())
            print(f"  [homeo={'ON ' if homeo else 'OFF'} ds{int(ds):3d} win{win}]  g_e={_c(ge, S_true):+.3f}  "
                  f"SPIKE={_c(sp, S_true):+.3f}  (spikes/concept={ms:.0f}, eff-rank={effective_rank(sp):.1f})",
                  flush=True)

    print("\n  If a homeo=OFF SPIKE row reaches ~+0.3..+0.4 (tracking g_e) -> the spike-readout loss was the "
          "homeostasis EQUALIZATION = config-level fix = the Phase-B boundary CRACKS (GO). If homeo=OFF spike "
          "stays ~0 -> the spiking threshold loses the weak common-mode-buried signal (deeper readout problem).",
          flush=True)


if __name__ == "__main__":
    main()
