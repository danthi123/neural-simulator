"""Phase-B LAST probe: across ~11 prior probes the cortex fired only ~1 spike/neuron -- a spike COUNT that
sparse is binary-ish and cannot encode the graded g_e (+0.45). HYPOTHESIS: dense firing (rate ∝ g_e, many
spikes/neuron) makes the spike count track g_e. Sweep stronger hub->cortex coupling (more g_e -> more firing)
+ long windows, homeostasis OFF (fixed threshold so rate ∝ drive), and check whether the cortex SPIKE code
reaches the g_e +0.45 once dense (≥ ~5-10 spikes/neuron). GO ⇒ the boundary was spike-count SPARSITY = a
config-level fix. If even DENSE the spike code stays ~0 while g_e +0.45 ⇒ the spike-count code fundamentally
cannot transmit the weak common-mode-buried signal -> the BOUNDARY is real (owner-decision: predictive coding
/ phase readout / dendrite / ship flat). NO sim/ edit.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_dense_firing_readout
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
    print(f"[dense-firing readout] {C.shape[0]}c x {n_hub}h. Drive the cortex DENSE (homeo OFF), check if the "
          f"SPIKE code tracks g_e (+0.45):", flush=True)
    best = -1.0
    for wm, ds, win in [(200.0, 40, 500), (400.0, 40, 500), (200.0, 60, 1000), (400.0, 60, 1000),
                        (800.0, 40, 1000)]:
        b, hub, cx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=128, seed=42, density=0.5,
                                            weight_mean=wm, stdp_w_max=2000.0, enable_homeostasis=False)
        hub = np.asarray(hub); cx = np.asarray(cx)
        sp = read_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
        ge = read_ge_codes(b, C_drive, hub, cx, drive_scale=ds, window=win, settle=8)
        spn = sp.sum(1).mean() / 128.0
        sc = _c(sp, S_true)
        best = max(best, sc)
        print(f"  [wm{int(wm):3d} ds{ds} win{win:4d}]  g_e={_c(ge, S_true):+.3f}  SPIKE={sc:+.3f}  "
              f"(spikes/neuron={spn:.1f}, eff-rank={effective_rank(sp):.1f})", flush=True)
    print(f"\n  VERDICT: best dense-firing SPIKE cos = {best:+.3f}", flush=True)
    if best >= 0.30:
        print("  GO -- dense firing makes the spike code track g_e -> the boundary was spike-count SPARSITY "
              "(config-level fix). Formalize + un-block the build.", flush=True)
    else:
        print("  BOUNDARY CONFIRMED -- even dense firing the spike-count code does NOT transmit the weak "
              "common-mode-buried g_e signal. The spike-count readout is the genuine boundary; the analog g_e "
              "has the structure. Owner-decision: predictive coding (Jang 2024) / phase readout (resonate-and-"
              "fire) / minimal dendrite / ship the flat curated cortex. The cheap solo probing is exhausted.",
              flush=True)


if __name__ == "__main__":
    main()
