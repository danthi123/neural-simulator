"""Diagnose hub + cortex firing vs drive_scale on the no-WTA bridge (no cm), to understand the
sparse-hub regime. CPU/numpy, NO sim/ edits.
Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_hub_fire.py
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
    build_sm_cortex_bridge, encode_drive, train_sm_cortex, _set_hub_drive, _step_with_time,
)
from sim.backend import to_host  # noqa: E402


def probe(bridge, hub_idx, cortex_idx, C_drive, drive_scale, cofire_pA, label):
    hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
    _set_hub_drive(bridge, hub_idx, C_drive[0], drive_scale, cortex_idx=cortex_idx, cofire_pA=cofire_pA)
    hub_f = 0.0; cx_f = 0.0; ge = 0.0
    W = 40; S = 8
    for t in range(S + W):
        _step_with_time(bridge)
        if t >= S:
            fired = np.asarray(to_host(bridge.cp_firing_states))
            hub_f += fired[hub_idx].sum(); cx_f += fired[cortex_idx].sum()
            ge += np.asarray(to_host(bridge.cp_conductance_g_e))[cortex_idx].mean()
    bridge.cp_external_input_current[:] = 0.0
    print(f"  [{label}] hub_fire/step={hub_f/W:.2f}/{len(hub_idx)}  "
          f"cortex_fire/step={cx_f/W:.2f}/{len(cortex_idx)}  cortex_g_e={ge/W:.4f}", flush=True)


def main():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"hubs={n_hub}; common-hub log-drive ~ {np.log1p(40):.2f} (x drive_scale = the pA per common hub)",
          flush=True)

    # untrained: vary drive_scale (no cofire) -> what fires the hubs?
    print("[untrained, no cofire] vary drive_scale:", flush=True)
    for ds in [12.0, 25.0, 50.0]:
        b, hi, ci = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42,
                                           cortex_exc_fraction=1.0, cortex_internal_density=0.0,
                                           homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03)
        probe(b, hi, ci, C_drive, ds, 0.0, f"untrained ds={ds}")

    # trained: the actual read regime (weights grown). cofire 0 at read.
    print("[trained C1a-style (no-WTA), read regime] drive_scale=12 cofire=0:", flush=True)
    b, hi, ci = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42,
                                       cortex_exc_fraction=1.0, cortex_internal_density=0.0,
                                       homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03)
    train_sm_cortex(b, C_drive, np.asarray(hi), np.asarray(ci), n_epochs=8, drive_scale=12.0,
                    window=40, settle=8, cofire_pA=4.0)
    probe(b, hi, ci, C_drive, 12.0, 0.0, "trained read ds=12")


if __name__ == "__main__":
    main()
