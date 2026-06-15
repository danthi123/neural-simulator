"""Phase-B localization (free probe): WHERE does the bridge lose the +0.89 drive structure -- at the HUB
input spiking, and is that FUNDAMENTAL or a C1a-regime artifact (the FAST homeostasis, added for the cortex
WTA, also equalizes the HUB firing rates and erases the drive-magnitude structure)?

The C1b de-risk found the HUB spike-rate codes already have −0.138 structure (the drive is +0.89). That
CONTRADICTS CYCLE-51/52 (log-drive → point-neuron gen ~0.95). The prime suspect: homeostasis is GLOBAL, so
the C1a FAST homeostasis (ema_alpha 0.05 vs the slow 0.0002 default) drives every HUB toward the target rate
→ equalizes the hubs → erases the graded structure. This probe reads the HUB spike-rate codes under FAST vs
SLOW (default) homeostasis + a clean (long) readout window, and checks whether the slow regime preserves the
drive structure. If SLOW preserves it → the input loss is a regime artifact (fast-homeostasis-on-hubs); the
real wall is the cortex common-mode removal (the dendritic piece). If SLOW also loses it → the input spiking
is fundamentally lossy. Localization for the honest Phase-B finding. NO sim/ edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_hub_encoding_regime
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
from research.runners.spiking_sm_cortex import (  # noqa: E402
    build_sm_cortex_bridge, encode_drive, _set_hub_drive, _step_with_time,
)
from sim.backend import to_host  # noqa: E402


def read_hub_codes(bridge, C_drive, hub_idx, *, drive_scale, window, settle):
    hub_idx = np.asarray(hub_idx)
    Nc = C_drive.shape[0]
    codes = np.zeros((Nc, hub_idx.size))
    for i in range(Nc):
        _set_hub_drive(bridge, hub_idx, C_drive[i], drive_scale)
        acc = np.zeros(hub_idx.size)
        for t in range(settle + window):
            _step_with_time(bridge)
            if t >= settle:
                acc += np.asarray(to_host(bridge.cp_firing_states))[hub_idx].astype(np.float64)
        codes[i] = acc
        bridge.cp_external_input_current[:] = 0.0
    return codes


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12, lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"[hub-encoding regime] {C.shape[0]} concepts x {n_hub} hubs; drive cos +0.891", flush=True)

    for label, homeo, ds, win in [
        ("FAST homeo, ds12, win40", 0.05, 12.0, 40),
        ("SLOW (default) homeo, ds12, win40", None, 12.0, 40),
        ("SLOW homeo, ds12, win150 (clean)", None, 12.0, 150),
        ("SLOW homeo, ds4 (gentler), win150", None, 4.0, 150),
        ("SLOW homeo, ds40 (saturate?), win150", None, 40.0, 150),
    ]:
        kw = dict(n_hub=n_hub, n_cortex=64, seed=42, density=0.5, weight_mean=80.0, stdp_w_max=200.0)
        if homeo is not None:
            kw.update(homeostasis_ema_alpha=homeo, homeostasis_threshold_adapt_rate=0.03)
        b, hub, cx = build_sm_cortex_bridge(**kw)
        codes = read_hub_codes(b, C_drive, hub, drive_scale=ds, window=win, settle=8)
        p = _pearson_vs_Strue(_cos_sim(codes), S_true)
        pc = _pearson_vs_Strue(_cos_sim(codes - codes.mean(0, keepdims=True)), S_true)
        meanfire = float(codes.mean())
        print(f"  [{label:38s}] hub-code cos={p:+.3f}  centered={pc:+.3f}  mean-spikes/hub={meanfire:.2f}",
              flush=True)

    print("\n  If a SLOW-homeo row recovers cos toward +0.6..+0.9 -> the input loss was the FAST-homeostasis-"
          "on-hubs artifact (regime); the real wall is the cortex common-mode removal (dendritic). "
          "If ALL rows stay ~0/negative -> the input spiking is fundamentally lossy (deeper).", flush=True)


if __name__ == "__main__":
    main()
