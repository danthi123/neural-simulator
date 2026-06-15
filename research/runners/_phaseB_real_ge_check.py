"""Phase direction, step 1: is the REAL category structure in the cortex ANALOG g_e (-> a phase/analog-
faithful readout can escape the spike-count loss), or is it lost in the projection itself (-> deeper)?
On synthetic the g_e carried it (+0.45) and only the SPIKE-COUNT readout lost it. On real the spike code was
+0.075 (even PPMI input). Read the real cortex g_e (PPMI + log input, dense regime). If g_e >> spike (g_e
toward +0.3..+0.5) -> the projection preserves it; the spike-count rate readout is the loss -> a PHASE readout
(resonate-and-fire, magnitude-independent) is the warranted escape. If g_e ALSO ~0 -> the spiking projection
itself loses the real diffuse structure -> phase-readout alone won't fix it. NO sim/ edit.

Run: SIM_BACKEND=cupy python -u -m research.runners._phaseB_real_ge_check --n-hub 500
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue  # noqa: E402
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402
from research.runners.spiking_sm_cortex import build_sm_cortex_bridge, encode_drive, read_codes  # noqa: E402
from research.runners._phaseB_c1b_derisk_perneuron_centering import read_ge_codes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hub", type=int, default=500); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    C, labels, S_true = build_real_corpus(args.seed, args.n_hub)
    C = np.asarray(C, dtype=np.float64); labels = np.asarray(labels)
    host, _, _, _ = (lambda s: score(s, labels))(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape)-1), alpha=0.75))
    n_hub = C.shape[1]
    X_log = encode_drive(C)
    X_ppmi = ppmi_matrix(C, 0.75); X_ppmi = X_ppmi * (X_log.mean() / (X_ppmi.mean() + 1e-9))
    print(f"[real g_e check] {C.shape[0]}c x {n_hub}h; host +{host:.3f}; PPMI input cos "
          f"{_pearson_vs_Strue(_cos_sim(X_ppmi), S_true):+.3f}", flush=True)
    bp = dict(n_hub=n_hub, n_cortex=128, density=0.5, weight_mean=400.0, stdp_w_max=2000.0, enable_homeostasis=False)
    rp = dict(drive_scale=40.0, window=1000, settle=8)
    for tag, X in [("log-counts", X_log), ("host-PPMI", X_ppmi)]:
        t0 = time.time()
        b, hub, cx = build_sm_cortex_bridge(seed=args.seed, **bp)
        hub = np.asarray(hub); cx = np.asarray(cx)
        ge = read_ge_codes(b, X, hub, cx, **rp)
        sp = read_codes(b, X, hub, cx, **rp)
        print(f"  [{tag:10s}] cortex g_e(analog)={_pearson_vs_Strue(_cos_sim(ge), S_true):+.3f}  "
              f"SPIKE-count={_pearson_vs_Strue(_cos_sim(sp), S_true):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print("\n  g_e >> spike (g_e toward +0.3..+0.5) -> projection preserves it, spike-COUNT is the loss -> PHASE "
          "readout (resonate-and-fire) escapes it. g_e ALSO ~0 -> projection loses the diffuse real structure.",
          flush=True)


if __name__ == "__main__":
    main()
