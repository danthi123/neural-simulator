"""Is the -0.07 partly a SPARSE-READOUT artifact? Read the trained C1a (no cm) cortex code with a
STRONGER drive / LONGER window so the cortex fires robustly, and see if the structure Pearson improves.
If even a dense readout stays ~ -0.07, the destroyer is purely the common mode (centering needed), not
read sparsity. CPU/numpy, NO sim/ edits, FAST (2-region bridge).
Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_readstrength.py
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, build_concept_hub_counts, effective_rank,
)
from research.runners.spiking_sm_cortex import (  # noqa: E402
    build_sm_cortex_bridge, encode_drive, train_sm_cortex, read_codes,
)


def main():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]

    # train one C1a (WTA) bridge once, then read at several drive/window settings.
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(
        n_hub=n_hub, n_cortex=128, seed=42,
        cortex_exc_fraction=0.8, cortex_internal_density=0.5, cortex_inh_weight_mean=6.0,
        homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03)
    hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, n_epochs=8, drive_scale=12.0,
                    window=40, settle=8, cofire_pA=4.0)

    print("[readstrength] trained C1a, vary read drive_scale x window (cortex firing density):", flush=True)
    for ds in [12.0, 30.0, 60.0]:
        for win in [40, 120]:
            codes = read_codes(bridge, C_drive, hub_idx, cortex_idx, drive_scale=ds, window=win, settle=8)
            pe = _pearson_vs_Strue(_cos_sim(codes), S_true)
            tot = codes.sum() / (len(cortex_idx))  # mean spikes/neuron across all concepts
            print(f"  read ds={ds} win={win}: Pearson={pe:+.3f}  silent="
                  f"{float(np.mean(codes.sum(1)==0)):.2f}  eff_rank={effective_rank(codes):.1f}  "
                  f"mean_spk/neuron={tot:.1f}", flush=True)


if __name__ == "__main__":
    main()
