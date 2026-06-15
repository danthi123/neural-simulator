"""Phase-B Task-3 cm-pool CENTERING gate harness (CPU/numpy, NO sim/ edits).

The decisive experiment for Mechanism 1 (feedforward subtractive-inhibition common-mode removal): does the
cm pool's inhibition CENTER the cortex drive enough to recover the synthetic category structure?

For each (n_cm, hub->cm w, cm->cortex w, cm_bias, drive_scale, gain) config: build the cm bridge (no WTA),
train hub->cortex (plastic) with the cm pool subtracting, read spike-count codes, and report the gate
metrics + a centering-OFF baseline (same config, cm->cortex weight 0) so the cm LOAD-BEARING margin is
visible.

Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_cm_gate.py
"""
import argparse
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

BIG_SIGMA = 1e9  # dendritic divisive gain effectively OFF (g = sigma/(sigma+a) ~ 1)


def _build_synth_64():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42)
    return C, labels, S_true


def metrics(codes, labels, S_true):
    pe = _pearson_vs_Strue(_cos_sim(codes), S_true)
    silent = float(np.mean(codes.sum(1) == 0))
    er = effective_rank(codes)
    rng = np.random.RandomState(20260615)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    pp = _pearson_vs_Strue(_cos_sim(codes), S_perm)
    return pe, silent, er, pp


def run_cm(n_cm, w_hubcm, w_cmcx, cm_bias, ds, sigma, n_epochs=8, cofire_pA=4.0, plastic=False):
    """Build the cm bridge, train + read with cm bias active. Returns (pearson, silent, eff_rank, perm)."""
    C, labels, S_true = _build_synth_64()
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(
        n_hub=n_hub, n_cortex=128, seed=42, sigma=sigma,
        cortex_exc_fraction=1.0, cortex_internal_density=0.0,
        homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03,
        n_cm=n_cm, hub_to_cm_weight=w_hubcm, cm_to_cortex_weight=w_cmcx,
        cm_to_cortex_plastic=plastic)
    cm_idx = np.asarray(bridge.region_manager.indices("cm")) if n_cm > 0 else None
    hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, n_epochs=n_epochs, drive_scale=ds,
                    window=40, settle=8, cofire_pA=cofire_pA, cm_idx=cm_idx, cm_bias_pA=cm_bias)
    codes = read_codes(bridge, C_drive, hub_idx, cortex_idx, drive_scale=ds, window=40, settle=8,
                       cm_idx=cm_idx, cm_bias_pA=cm_bias)
    return metrics(codes, labels, S_true)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="fine")
    args = p.parse_args()

    print("[cm gate] structure recovery with the feedforward subtractive-inhibition cm pool", flush=True)
    print("  config: gain OFF (sigma=1e9), no WTA, n_cortex=128, 8 epochs, cofire 4 pA, ds=30", flush=True)

    if args.mode == "fine":
        # FINE sweep of the graded band: weak cm (no bias) but stronger cm->cortex; and small biases that
        # keep the cortex ALIVE (avoid silent=1.0). Each ON paired with its centering-OFF (w_cmcx=0).
        configs = [
            # (n_cm, w_hubcm, w_cmcx, cm_bias)  -- bias 0 or small; cm fires gradedly
            (200, 5.0, 20.0, 0.0),
            (200, 5.0, 40.0, 0.0),
            (200, 10.0, 20.0, 0.0),
            (200, 10.0, 40.0, 0.0),
            (200, 5.0, 6.0, 50.0),
            (200, 5.0, 12.0, 50.0),
            (200, 5.0, 6.0, 100.0),
            (200, 5.0, 12.0, 100.0),
        ]
        for (n_cm, w_hubcm, w_cmcx, cm_bias) in configs:
            on = run_cm(n_cm, w_hubcm, w_cmcx, cm_bias, 30.0, BIG_SIGMA)
            off = run_cm(n_cm, w_hubcm, 0.0, cm_bias, 30.0, BIG_SIGMA)
            print(f"  [FIX w_hubcm={w_hubcm} w_cmcx={w_cmcx} bias={cm_bias}] "
                  f"ON P={on[0]:+.3f}(sil={on[1]:.2f},er={on[2]:.1f})  OFF P={off[0]:+.3f}  "
                  f"margin={on[0]-off[0]:+.3f}  perm={on[3]:+.3f}", flush=True)

    elif args.mode == "plastic":
        # PLASTIC cm->cortex: inhibitory STDP learns each cortex neuron's own common-mode susceptibility.
        print("[PLASTIC cm->cortex] inhibitory STDP shapes per-cortex-neuron centering", flush=True)
        configs = [
            (200, 5.0, 6.0, 50.0),
            (200, 5.0, 6.0, 100.0),
            (200, 5.0, 12.0, 100.0),
            (200, 10.0, 6.0, 50.0),
        ]
        for (n_cm, w_hubcm, w_cmcx, cm_bias) in configs:
            on = run_cm(n_cm, w_hubcm, w_cmcx, cm_bias, 30.0, BIG_SIGMA, plastic=True)
            off = run_cm(n_cm, w_hubcm, 0.0, cm_bias, 30.0, BIG_SIGMA, plastic=True)
            print(f"  [PLASTIC w_hubcm={w_hubcm} w_cmcx={w_cmcx} bias={cm_bias}] "
                  f"ON P={on[0]:+.3f}(sil={on[1]:.2f},er={on[2]:.1f})  OFF P={off[0]:+.3f}  "
                  f"margin={on[0]-off[0]:+.3f}  perm={on[3]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
