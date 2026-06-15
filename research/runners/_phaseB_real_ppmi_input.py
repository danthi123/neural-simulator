"""Phase-B airtight localization of the REAL wall: the dense-firing readout is fixed, but the LEARNED+RANDOM
dense cortex gets only +0.06 on the real corpus (host +0.44). HYPOTHESIS: the bridge drives hubs with
log-COUNTS, but the real category structure needs PPMI (the marginal-ratio centering/whitening) -- the
missing op. Drive the hubs with host-PPMI rows (a TEST INSTRUMENT -- if it works, the brain-realization of
PPMI = the whitening is the open problem; PPMI itself is a host shortcut here) + the dense readout, and check
whether the cortex code recovers toward the host +0.44. If PPMI-input recovers it while log-input did not ->
the REAL wall is the INPUT WHITENING (PPMI centering = common-mode removal), NOT the readout (fixed) and NOT
the projection -- the Mikulasch-Priesemann limit, at the INPUT. NO sim/ edit.

Run: SIM_BACKEND=cupy python -u -m research.runners._phaseB_real_ppmi_input --n-hub 500
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue, heldout_generalization  # noqa: E402
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive, read_codes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hub", type=int, default=500); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight-mean", type=float, default=400.0); p.add_argument("--drive-scale", type=float, default=40.0)
    p.add_argument("--window", type=int, default=1000)
    args = p.parse_args()
    C, labels, S_true = build_real_corpus(args.seed, args.n_hub)
    C = np.asarray(C, dtype=np.float64); labels = np.asarray(labels)
    host = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75)
    host_p, _, _, _ = score(host, labels)
    n_hub = C.shape[1]
    # two input encodings: the bridge's current log-counts, vs host-PPMI rows (the whitening locus).
    X_log = encode_drive(C)                       # what the bridge uses now
    X_ppmi = ppmi_matrix(C, 0.75)                 # the L1 whitened input (host instrument), non-negative
    # scale PPMI to a comparable drive magnitude (its values are ~0..8; log-counts ~0..5)
    X_ppmi = X_ppmi * (X_log.mean() / (X_ppmi.mean() + 1e-9))
    print(f"[real PPMI-input localization] {C.shape[0]}c x {n_hub}h; host +{host_p:.3f}; dense regime", flush=True)
    print(f"  input cos: log-counts={_pearson_vs_Strue(_cos_sim(X_log), S_true):+.3f}  "
          f"PPMI={_pearson_vs_Strue(_cos_sim(X_ppmi), S_true):+.3f}", flush=True)
    bp = dict(n_hub=n_hub, n_cortex=128, density=0.5, weight_mean=args.weight_mean, stdp_w_max=2000.0,
              enable_homeostasis=False)
    rp = dict(drive_scale=args.drive_scale, window=args.window, settle=8)
    for tag, X in [("log-counts (current)", X_log), ("host-PPMI (whitened)", X_ppmi)]:
        t0 = time.time()
        b, hub, cx = build_sm_cortex_bridge(seed=args.seed, **bp)
        codes = read_codes(b, X, np.asarray(hub), np.asarray(cx), **rp)   # UNTRAINED dense readout
        pe = _pearson_vs_Strue(_cos_sim(codes), S_true)
        g, ch = heldout_generalization(codes, labels)
        print(f"  [{tag:24s}] cortex dense code Pearson={pe:+.3f}  gen={g:.3f} (chance {ch:.3f})  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("\n  If host-PPMI input recovers (toward +0.3..+0.44) while log-counts stays ~0 -> the REAL wall is "
          "the INPUT WHITENING (PPMI centering = common-mode removal), point-neuron-hard (Mikulasch-Priesemann), "
          "at the INPUT -- the readout (dense firing) is fixed. If host-PPMI ALSO stays ~0 -> deeper.", flush=True)


if __name__ == "__main__":
    main()
