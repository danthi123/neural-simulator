"""Phase-B retinal build — PRE-BUILD design probe: WHICH whitening does the bridge need to realize?

The numpy reference (_phaseB_onoff_whitened_derisk) whitens the INPUT with `Xn - Xn.mean(0)` =
subtract the ACROSS-CONCEPT mean per hub (axis 0). That is a per-hub fixed BIAS (each hub's average
drive over the whole concept set) -- a learned/adapted baseline, NOT something within-hub lateral
inhibition computes on a single presentation.

The CLAUDE-style task spec for the bridge says "each hub fires relative to its neighborhood/population
mean" / "a global-inhibitory pool that pools all hubs and subtracts the population mean" = the
ACROSS-HUB mean per concept (axis 1), computed per presentation. That IS what lateral inhibition /
a feedback inhibitory pool does naturally on the bridge.

These are DIFFERENT operations. This probe asks, on the real corpus + ON/OFF + high spike budget:
  - axis-0 (across-concept per-hub bias)  -- the reference (host instrument, hard to do streaming)
  - axis-1 (across-hub population mean)    -- the bridge-realizable lateral-inhibition whitening
  - both                                    -- per-hub bias AND per-concept population-mean removal
  - none                                    -- the point control
Whichever of {axis1, both} carries the structure tells me what the bridge front-end must compute. If
ONLY axis-0 works, the bridge must realize a per-hub adapted baseline (slow EMA self-inhibition), not a
population-mean pool -- a key design fork to settle BEFORE building.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_whitening_locus_probe
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def onoff_code(drive, gain, rng):
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def run(C, labels, S_true, gain=2000.0, k=128):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    X = ppmi_matrix(C, 0.75)
    host, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)
    H = X.shape[1]; rng = np.random.RandomState(7)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    W = rng.randn(k, H) / np.sqrt(H)

    variants = {
        "none (point)":            Xn,
        "axis0 (per-hub bias)":    Xn - Xn.mean(0, keepdims=True),
        "axis1 (pop-mean/lat-inh)": Xn - Xn.mean(1, keepdims=True),
        "both":                    Xn - Xn.mean(0, keepdims=True) - (Xn - Xn.mean(0, keepdims=True)).mean(1, keepdims=True),
    }
    print(f"  host=+{host:.3f}  (gain {int(gain)}, k {k})", flush=True)
    # TRUE point control matching the GATE: single population, NO whitening, NO ON/OFF (rectified rate).
    pt_rate = np.maximum(W @ Xn.T, 0.0).T
    pt = np.array([poisson_spk(pt_rate[i], gain, rng) for i in range(len(X))])
    pp = _pearson_vs_Strue(_cos_sim(pt), S_true)
    pg, pch = heldout_generalization(pt, labels)
    print(f"    [{'POINT (gate ctrl, 1-pop)':26s}] Pearson={pp:+.3f}  gen={pg:.3f} (chance {pch:.3f})", flush=True)
    for name, Xv in variants.items():
        drive = (W @ Xv.T).T
        code = onoff_code(drive, gain, rng)
        p = _pearson_vs_Strue(_cos_sim(code), S_true)
        gen, ch = heldout_generalization(code, labels)
        beats = " (beats point by %+.3f)" % (p - pp) if name != "none (point)" else ""
        print(f"    [{name:26s}] ON/OFF Pearson={p:+.3f}  gen={gen:.3f} (chance {ch:.3f}){beats}", flush=True)


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    print("[whitening-locus probe] axis0 (per-hub bias) vs axis1 (population-mean / lateral-inhibition) "
          "vs both -- which carries the real structure? (decides the bridge front-end)", flush=True)
    Cr, lr, Sr = build_real_corpus(42, 500)
    run(Cr, lr, Sr)


if __name__ == "__main__":
    main()
