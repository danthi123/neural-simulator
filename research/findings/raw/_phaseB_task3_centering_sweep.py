"""Phase-B Task-3 centering sweep harness (CPU/numpy, NO sim/ edits).

Re-runs the HARD GATE metrics over centering mechanisms, cheapest-first:
  (3) stronger dendritic divisive gain (smaller sigma)
  (2) enable_synaptic_scaling (Turrigiano per-cortex-neuron renormalization)  [config flag]
  (1) feedforward subtractive-inhibition cm pool                              [framework region+pathways]

Each arm builds a bridge, trains, reads spike-count codes (plasticity frozen), and reports
Pearson(cos(codes), S_true), the random-projection margin, silent_frac, eff_rank, permuted.

The builder kwargs for (1) (n_cm / cm_to_cortex_weight / hub_to_cm_weight / etc.) are added to
spiking_sm_cortex.build_sm_cortex_bridge in this session (additive, default-off = byte-preserved).

Run: SIM_BACKEND=numpy python research/findings/raw/_phaseB_task3_centering_sweep.py --arms sigma,scaling
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


def _build_synth_64():
    C, labels, S_true, _ = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42,
    )
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


def run_arm(name, build_kwargs, train_kwargs, read_kwargs, C_drive, labels, S_true, n_hub):
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(n_hub=n_hub, **build_kwargs)
    hub_idx = np.asarray(hub_idx); cortex_idx = np.asarray(cortex_idx)
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, **train_kwargs)
    codes = read_codes(bridge, C_drive, hub_idx, cortex_idx, **read_kwargs)
    pe, silent, er, pp = metrics(codes, labels, S_true)
    print(f"  [{name:42s}] Pearson={pe:+.3f}  silent={silent:.3f}  eff_rank={er:5.1f}  perm={pp:+.3f}",
          flush=True)
    return pe, silent, er, pp


# Common C1a knobs (WTA on). The "no-WTA" variant per L1 (no lateral) sets exc_fraction=1.0/density=0.0.
_BASE_TRAIN = dict(n_epochs=8, drive_scale=12.0, window=40, settle=8, cofire_pA=4.0)
_BASE_READ = dict(drive_scale=12.0, window=40, settle=8)
_C1A_BUILD = dict(seed=42, n_cortex=128, cortex_exc_fraction=0.8, cortex_internal_density=0.5,
                  cortex_inh_weight_mean=6.0, homeostasis_ema_alpha=0.05,
                  homeostasis_threshold_adapt_rate=0.03)
_NOWTA_BUILD = dict(seed=42, n_cortex=128, cortex_exc_fraction=1.0, cortex_internal_density=0.0,
                    homeostasis_ema_alpha=0.05, homeostasis_threshold_adapt_rate=0.03)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="baseline,sigma,scaling")
    args = p.parse_args()
    arms = args.arms.split(",")

    C, labels, S_true = _build_synth_64()
    C_drive = encode_drive(C)
    n_hub = C.shape[1]
    print(f"[centering sweep] {C.shape[0]} concepts x {n_hub} hubs; "
          f"log-input ceiling {_pearson_vs_Strue(_cos_sim(C_drive), S_true):+.3f}", flush=True)

    if "baseline" in arms:
        print("[baseline C1a WTA] (documented PARTIAL ~ -0.07)", flush=True)
        run_arm("C1a baseline (WTA)", dict(**_C1A_BUILD), _BASE_TRAIN, _BASE_READ,
                C_drive, labels, S_true, n_hub)
        run_arm("no-WTA baseline (exc=1.0,dens=0.0)", dict(**_NOWTA_BUILD), _BASE_TRAIN, _BASE_READ,
                C_drive, labels, S_true, n_hub)

    if "sigma" in arms:
        print("[Mechanism 3: stronger dendritic divisive gain (smaller sigma)]", flush=True)
        for sigma in [0.02, 0.01, 0.005]:
            run_arm(f"no-WTA sigma={sigma}", dict(**_NOWTA_BUILD, sigma=sigma), _BASE_TRAIN, _BASE_READ,
                    C_drive, labels, S_true, n_hub)

    if "scaling" in arms:
        print("[Mechanism 2: enable_synaptic_scaling (Turrigiano renorm)]", flush=True)
        for rate in [0.001, 0.01, 0.05]:
            run_arm(f"no-WTA synscale rate={rate}",
                    dict(**_NOWTA_BUILD, enable_synaptic_scaling=True, synaptic_scaling_rate=rate),
                    _BASE_TRAIN, _BASE_READ, C_drive, labels, S_true, n_hub)

    if "cm" in arms:
        print("[Mechanism 1: feedforward subtractive-inhibition cm pool (no-WTA)]", flush=True)
        # n_cm large enough to track the common mode; sweep cm->cortex strength (the centering gain).
        for n_cm, w_cm in [(100, 3.0), (100, 6.0), (100, 12.0), (200, 6.0), (200, 12.0)]:
            run_arm(f"no-WTA cm n={n_cm} w={w_cm}",
                    dict(**_NOWTA_BUILD, n_cm=n_cm, cm_to_cortex_weight=w_cm),
                    _BASE_TRAIN, _BASE_READ, C_drive, labels, S_true, n_hub)


if __name__ == "__main__":
    main()
