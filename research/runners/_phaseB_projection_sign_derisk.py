"""The projection-SIGN wall: a non-negative (Dale's-law excitatory) random projection collapses the axis-0-
centered signal; a SIGNED projection carries it. Does an E/I-realizable signed projection (W_exc - W_inh, both
>=0) recover it? (The decisive de-risk for the next bridge direction.)

The input-mean bridge gate (`_phaseB_input_mean_bridge_gpusmoke.json`) isolated the wall via its host refs (real
seed 42): non-neg bridge projection x axis-0 centering = +0.045 (COLLAPSES) vs a SIGNED Gaussian projection x
axis-0 = +0.294 (the numpy target). The per-hub input-mean primitive WORKS (EMA tracked the mean) and the axis
is CORRECT -- but the bridge's EXCITATORY (non-negative) hub->cortex weights cannot carry the centered (signed,
zero-mean) structure: a positive-weighted random projection produces highly-correlated low-rank outputs. Every
numpy de-risk used rng.randn (SIGNED), which excitatory synapses alone can't realize.

Biology's answer = E/I balance: the EFFECTIVE projection is excitation MINUS inhibition (W_exc - W_inh, both
>=0), which is signed. This probe confirms the wall and tests the E/I fix on the real corpus, axis-0-centered:
  (a) signed Gaussian W           -- the (biologically-illegal) numpy target
  (b) non-negative |W| (relu)     -- the bridge's excitatory-only projection (expect collapse)
  (c) E/I signed (W_exc - W_inh)  -- BOTH >=0, the biologically-legal signed projection (the candidate fix)
  (d) E/I with inhib DENSITY < exc (e.g. 25% inhibitory) -- a realistic E/I ratio
GATE: does (c)/(d) recover ~the signed (a) level (and beat the non-neg (b))? GO => the next bridge build adds an
inhibitory hub->cortex pathway (signed effective projection); the input-mean primitive is the (banked) centering.

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_projection_sign_derisk
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def onoff_code(drive, gain, rng):
    on = np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])
    off = np.array([poisson_spk(np.maximum(-drive[i], 0.0), gain, rng) for i in range(len(drive))])
    return np.concatenate([on, off], axis=1)


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0))
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xw = Xn - Xn.mean(0, keepdims=True)            # axis-0 per-feature centering (the CORRECT axis)

    def p_of(Wproj):
        code = onoff_code((Wproj @ Xw.T).T, gain, rng)
        return _pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0]

    scale = 1.0 / np.sqrt(n_hub)
    W_signed = rng.randn(k, n_hub) * scale                                   # (a) signed Gaussian (illegal)
    W_nonneg = np.abs(rng.randn(k, n_hub)) * scale                           # (b) excitatory-only (>=0)
    We = np.abs(rng.randn(k, n_hub)) * scale; Wi = np.abs(rng.randn(k, n_hub)) * scale
    W_ei = We - Wi                                                           # (c) E/I signed, full density
    # (d) realistic E/I: inhibitory projection sparser (25% density) but stronger per-synapse to balance.
    mask_i = (rng.rand(k, n_hub) < 0.25).astype(np.float64)
    Wi_sparse = np.abs(rng.randn(k, n_hub)) * scale * mask_i * 4.0
    W_ei_real = np.abs(rng.randn(k, n_hub)) * scale - Wi_sparse

    p_sig, g_sig = p_of(W_signed); p_nn, g_nn = p_of(W_nonneg)
    p_ei, g_ei = p_of(W_ei); p_eir, g_eir = p_of(W_ei_real)
    print(f"\n[projection-sign seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f} (axis-0 centered)\n"
          f"  (a) signed Gaussian   : {p_sig:+.3f} (gen {g_sig:.3f})\n"
          f"  (b) non-neg |W| (exc) : {p_nn:+.3f} (gen {g_nn:.3f})   <- the bridge's excitatory-only collapse\n"
          f"  (c) E/I W_exc-W_inh   : {p_ei:+.3f} (gen {g_ei:.3f})   <- the candidate fix (signed via E/I)\n"
          f"  (d) E/I realistic 25% : {p_eir:+.3f} (gen {g_eir:.3f})", flush=True)
    return {"seed": seed, "host": host_p, "signed": p_sig, "nonneg": p_nn, "ei": p_ei, "ei_real": p_eir}


def main():
    seeds = [42, 43, 44]
    rows = [run_seed(s) for s in seeds]
    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f"\n  MEAN ({len(seeds)} seeds): host {m('host'):+.3f} | signed {m('signed'):+.3f} | "
          f"non-neg(exc) {m('nonneg'):+.3f} | E/I {m('ei'):+.3f} | E/I-realistic {m('ei_real'):+.3f}", flush=True)
    if m("nonneg") < 0.15 and m("ei") >= m("signed") - 0.05:
        print(f"  GO (the wall is the projection SIGN; E/I fixes it): excitatory-only collapses "
              f"({m('nonneg'):+.3f}) but the E/I signed projection recovers the signed level "
              f"({m('ei'):+.3f} ~= signed {m('signed'):+.3f}). => the next bridge direction = a SIGNED effective "
              f"hub->cortex projection via an INHIBITORY pathway alongside the excitatory one (E/I balance); the "
              f"input-mean primitive remains the (banked) axis-0 centering.", flush=True)
    elif m("nonneg") < 0.15:
        print(f"  PARTIAL: excitatory-only collapses ({m('nonneg'):+.3f}) confirming the projection-sign wall, "
              f"but E/I ({m('ei'):+.3f}) does not fully recover the signed level ({m('signed'):+.3f}) -- the E/I "
              f"projection helps but isn't the whole fix.", flush=True)
    else:
        print(f"  UNEXPECTED: excitatory-only did NOT collapse ({m('nonneg'):+.3f}) -- the gate's +0.045 has "
              f"another cause; inspect the bridge projection vs this numpy model.", flush=True)


if __name__ == "__main__":
    main()
