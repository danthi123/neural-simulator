"""Quick cm-pool diagnostic: does the all-inhibitory cm pool FIRE from hub drive, and does its
inhibition reach the cortex (g_i)? CPU/numpy, NO sim/ edits.

Builds a cm bridge, drives ONE concept's hub pattern (no training), runs a window, and reports:
  - cm firing fraction (must be > 0 for centering to work)
  - cortex mean g_e and g_i during the window (g_i should be non-zero = inhibition delivered)
Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_cm_diag.py
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import build_concept_hub_counts  # noqa: E402
from research.runners.spiking_sm_cortex import (  # noqa: E402
    build_sm_cortex_bridge, encode_drive, _set_hub_drive, _step_with_time,
)
from sim.backend import to_host  # noqa: E402


def main():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]

    # The hubs fire SPARSELY at ds=12. Make the regime FIRE: gain off (sigma huge), higher drive, strong
    # dense hub->cm. Check whether cm fires CONSISTENTLY across concepts (~common mode).
    BIG_SIGMA = 1e9  # gain g = sigma/(sigma+a) ~ 1 -> dendritic divisive gain effectively OFF
    for ds in [30.0, 60.0, 120.0]:
        for w_hubcm in [2.0, 5.0, 10.0]:
            n_cm = 200; w_cmcx = 6.0
            bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(
                n_hub=n_hub, n_cortex=128, seed=42, sigma=BIG_SIGMA,
                cortex_exc_fraction=1.0, cortex_internal_density=0.0,
                homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03,
                n_cm=n_cm, hub_to_cm_weight=w_hubcm, cm_to_cortex_weight=w_cmcx)
            cm_idx = np.asarray(bridge.region_manager.indices("cm"))
            hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
            cm_per_concept = []; hub_per_concept = []
            for ci_concept in [0, 8, 16]:
                _set_hub_drive(bridge, hub_idx, C_drive[ci_concept], ds)
                cmf = 0.0; hbf = 0.0; W = 30; S = 8
                for t in range(S + W):
                    _step_with_time(bridge)
                    if t >= S:
                        fired = np.asarray(to_host(bridge.cp_firing_states))
                        cmf += fired[cm_idx].sum(); hbf += fired[hub_idx].sum()
                cm_per_concept.append(cmf / W); hub_per_concept.append(hbf / W)
                bridge.cp_external_input_current[:] = 0.0
            print(f"[cm diag] gainOFF ds={ds} w_hubcm={w_hubcm}: hub_fire/step~{np.mean(hub_per_concept):.1f}"
                  f"  cm_fire/step per concept [c0,c8,c16]={['%.1f'%x for x in cm_per_concept]} "
                  f"(mean {np.mean(cm_per_concept):.1f}/{n_cm})", flush=True)


if __name__ == "__main__":
    main()
