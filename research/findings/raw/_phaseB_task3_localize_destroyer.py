"""Phase-B Task-3 destroyer-localization probe (CPU/numpy, NO sim/ edits).

The C1a HARD GATE is PARTIAL: trained spike-code Pearson(cos, S_true) ~ -0.07. The input carries the
structure (rate-level log-input cosine +0.89). Question: is the structure destroyed by the SPIKING
THRESHOLD (the rectification/nonlinearity) or by the COMMON MODE surviving into the cortex drive?

This probe reads, on the SAME trained C1a bridge, two per-concept cortex codes with plasticity frozen:
  (1) the SPIKE-COUNT code (the gate's code)        -> cos -> Pearson vs S_true
  (2) the integrated CONDUCTANCE g_e code (cp_conductance_g_e accumulated over the window, pre-threshold)
      -> cos -> Pearson vs S_true
If g_e-cos already ~ -0.07 (low), the common mode survives into the analog drive -> the THRESHOLD is NOT
the destroyer; the common-mode is. If g_e-cos is HIGH (+) but spike-cos is low, the spiking threshold is
discarding the structure (a thresholding problem). This tells the controller which mechanism to reach for.

Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_localize_destroyer.py
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
    _set_hub_drive, _step_with_time,
)
from sim.backend import to_host  # noqa: E402


def _build_synth_64():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42,
    )
    return C, labels, S_true


def read_ge_codes(bridge, C_drive, hub_idx, cortex_idx, *, drive_scale=12.0, window=40, settle=8):
    """Read the per-concept cortex codes by accumulating cp_conductance_g_e (the pre-threshold analog
    excitatory drive), plasticity frozen. Mirrors read_codes but sums g_e instead of spikes."""
    cortex_idx = np.asarray(cortex_idx)
    Nc = int(np.asarray(C_drive).shape[0])
    codes = np.zeros((Nc, cortex_idx.size), dtype=np.float64)
    bridge.set_plasticity_gate("hub_to_cortex", 0.0)
    try:
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale)
            acc = np.zeros(cortex_idx.size, dtype=np.float64)
            for t in range(int(settle) + int(window)):
                _step_with_time(bridge)
                if t >= int(settle):
                    ge = np.asarray(to_host(bridge.cp_conductance_g_e))[cortex_idx]
                    acc += ge.astype(np.float64)
            codes[i] = acc
            bridge.cp_external_input_current[:] = 0.0
    finally:
        bridge.set_plasticity_gate("hub_to_cortex", 1.0)
    return codes


def main():
    C, labels, S_true = _build_synth_64()
    C_drive = encode_drive(C)
    n_hub = C.shape[1]

    # rate-level log-input ceiling (what the input carries before the bridge)
    log_in_pearson = _pearson_vs_Strue(_cos_sim(C_drive), S_true)
    print(f"[localize] rate-level log-input cosine Pearson vs S_true = {log_in_pearson:+.3f}", flush=True)

    # C1a recipe (the documented PARTIAL)
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(
        n_hub=n_hub, n_cortex=128, seed=42,
        cortex_exc_fraction=0.8, cortex_internal_density=0.5, cortex_inh_weight_mean=6.0,
        homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03,
    )
    hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, n_epochs=8, drive_scale=12.0,
                    window=40, settle=8, cofire_pA=4.0)

    spk = read_codes(bridge, C_drive, hub_idx, cortex_idx, drive_scale=12.0, window=40, settle=8)
    ge = read_ge_codes(bridge, C_drive, hub_idx, cortex_idx, drive_scale=12.0, window=40, settle=8)

    spk_p = _pearson_vs_Strue(_cos_sim(spk), S_true)
    ge_p = _pearson_vs_Strue(_cos_sim(ge), S_true)
    print(f"[localize] spike-count code Pearson = {spk_p:+.3f}  (silent_frac="
          f"{float(np.mean(spk.sum(1)==0)):.3f}, eff_rank={effective_rank(spk):.1f})", flush=True)
    print(f"[localize] g_e (pre-threshold) code Pearson = {ge_p:+.3f}  (eff_rank={effective_rank(ge):.1f})",
          flush=True)
    if ge_p < 0.10:
        print("[localize] => g_e already LOW: the COMMON MODE survives into the analog drive. The "
              "spiking threshold is NOT the destroyer; common-mode removal (centering) is the fix.", flush=True)
    elif spk_p < ge_p - 0.10:
        print("[localize] => g_e HIGH but spikes LOW: the SPIKING THRESHOLD discards the structure "
              "(a thresholding/readout problem, not pure common-mode).", flush=True)
    else:
        print("[localize] => g_e ~ spikes: threshold faithful; the destroyer is upstream (common mode).",
              flush=True)


if __name__ == "__main__":
    main()
