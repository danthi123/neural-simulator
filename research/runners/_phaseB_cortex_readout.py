"""Phase-B final-piece localization: the cortex g_e (analog) PRESERVES the structure at strong drive (+0.45),
but the earlier gate's cortex SPIKE-COUNT codes lost it (−0.05). Is the spike readout intrinsically lossy, or
was it the C1a regime (co-fire + WTA + fast homeostasis perturbing the cortex firing)? Read the SAME clean
bridge's cortex g_e AND spike-count codes (NO WTA, NO co-fire, NO fast homeostasis -- a graded firing regime)
across drive/window, and compare. If a clean spike code reaches ~+0.3 -> the WALL was the C1a regime + weak
drive (ENGINEERING, fully tunable, NOT the dendritic wall). If g_e stays +0.45 but every spike code stays ~0
-> the spike-count readout is the genuine loss (a readout problem -- longer window / rate-coded readout, still
not the months-scale dendritic substrate). NO sim/ edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_cortex_readout
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
    print(f"[cortex readout] {C.shape[0]}c x {n_hub}h. CLEAN bridge (no WTA/co-fire/fast-homeo), UNTRAINED "
          f"(random W). g_e vs SPIKE-COUNT readout:", flush=True)

    for ds, win in [(40.0, 150), (80.0, 150), (40.0, 300), (20.0, 300), (120.0, 300)]:
        b, hub, cx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42, density=0.5,
                                            weight_mean=80.0, stdp_w_max=200.0)
        hub = np.asarray(hub); cx = np.asarray(cx)
        ge = read_ge_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
        sp = read_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
        meanspk = float(sp.sum(1).mean())
        print(f"  [ds{int(ds):3d} win{win}]  g_e cos={_c(ge, S_true):+.3f}   SPIKE cos={_c(sp, S_true):+.3f}  "
              f"(mean cortex spikes/concept={meanspk:.0f}, eff-rank={effective_rank(sp):.1f})", flush=True)

    print("\n  g_e +0.45 = the structure is in the cortex's dendritic input. If a clean SPIKE row reaches "
          "~+0.3 -> the earlier WALL was the C1a regime + weak drive (tunable engineering). If every SPIKE "
          "row stays ~0 while g_e stays +0.45 -> the loss is the spike-count READOUT (a readout fix: longer "
          "window / more neurons / rate-coded readout) -- still NOT the dendritic substrate.", flush=True)


if __name__ == "__main__":
    main()
