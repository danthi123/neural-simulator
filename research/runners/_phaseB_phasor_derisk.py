"""Phase-coding escape de-risk (numpy, the committed direction): does a PHASOR representation (unit-magnitude,
info in PHASE -> NO common mode by construction, the FHRR principle that escaped the composer's common-mode
wall) preserve the real category structure where the RATE projection lost it?

The rate->spike wall: the real whitened structure is in the PATTERN (uniform magnitude); rate coding encodes
in MAGNITUDE so it loses the pattern (bridge cortex +0.075). The phasor escape: encode each concept as a
unit-magnitude phasor (phase carries the pattern, magnitude=1 -> no common mode); project through COMPLEX
weights (the FHRR / complex-synapse machinery the project already ships); read the PHASE-similarity (Re of the
complex inner product). If the phasor projection preserves the structure (+0.3..+0.5) where the real random
RATE projection collapses (~0) -> the phase-coded cortex is the validated escape -> build the resonate-and-fire
phasor cortex. Multi-control: rate-random-proj (the wall), phasor-random-proj (the escape), permuted.

Run: SIM_BACKEND=numpy python -u -m research.runners._phaseB_phasor_derisk
"""
from __future__ import annotations
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def phase_sim(Z):
    """FHRR similarity of complex codes: Re(z_i . conj(z_j)) normalized -> the phasor-coding cosine."""
    Zn = Z / (np.abs(Z).sum(1, keepdims=True) ** 0 + 1e-9)  # keep complex; normalize by vector norm below
    G = (Z @ Z.conj().T).real
    d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    return G / np.outer(d, d)


def run(label, C, labels, S_true):
    C = np.asarray(C, np.float64); labels = np.asarray(labels)
    X = ppmi_matrix(C, 0.75)                                  # the whitened input (+structure)
    host, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape)-1), alpha=0.75), labels)
    k, H = 128, X.shape[1]
    rng = np.random.RandomState(7)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # (1) RATE random projection (the wall: real weights, magnitude code)
    Wr = rng.randn(k, H) / np.sqrt(H)
    rate_codes = np.maximum(Wr @ Xn.T, 0.0).T               # relu (a rate is non-negative)
    rate_p = _pearson_vs_Strue(_cos_sim(rate_codes), S_true)

    # (2) PHASOR random projection (the escape: encode each hub as a UNIT-magnitude phasor exp(i*phase),
    #     phase carries the PPMI pattern; complex random weights; read the phase-similarity). NO common mode
    #     (every phasor has |.|=1), so the projection has no dominant common direction.
    phase = np.pi * (Xn - Xn.mean()) / (Xn.std() + 1e-9)    # map the pattern to a phase angle (zero-mean)
    P = np.exp(1j * phase)                                  # [Nc x H] unit-magnitude phasors
    Wc = (rng.randn(k, H) + 1j * rng.randn(k, H)) / np.sqrt(2 * H)
    phasor_codes = P @ Wc.T                                  # [Nc x k] complex
    phasor_p = _pearson_vs_Strue(phase_sim(phasor_codes), S_true)

    # generalization of the phasor codes (use |.| + angle as a real feature for the held-out test)
    feat = np.concatenate([phasor_codes.real, phasor_codes.imag], axis=1)
    gen, ch = heldout_generalization(feat, labels)
    # permuted control on the phasor code
    perml = rng.permutation(labels); S_perm = (perml[:, None] == perml[None, :]).astype(float)
    perm_p = _pearson_vs_Strue(phase_sim(phasor_codes), S_perm)

    print(f"  [{label}] host=+{host:.3f}  input cos={_pearson_vs_Strue(_cos_sim(X), S_true):+.3f}  ||  "
          f"RATE-proj={rate_p:+.3f}  PHASOR-proj={phasor_p:+.3f} (gen {gen:.3f}/ch {ch:.3f}, perm {perm_p:+.3f})",
          flush=True)
    return rate_p, phasor_p, host


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    print("[phasor escape de-risk] does a unit-magnitude PHASOR projection preserve the structure the RATE "
          "projection loses?", flush=True)
    # synthetic (strong) -- both should do okay; real (weak/diffuse) -- the decisive contrast
    Cs, ls, Ss, _ = build_concept_hub_counts(8, 8, 200, 12, 40.0, 4.0, 0.3, 42)
    run("SYNTHETIC", Cs, ls, Ss)
    Cr, lr, Sr = build_real_corpus(42, 500)
    rr, pr, host = run("REAL    ", Cr, lr, Sr)
    print("\n  VERDICT:", flush=True)
    if pr >= 0.30 and pr >= rr + 0.10:
        print(f"  GO -- the PHASOR projection preserves the real structure ({pr:+.3f}) where the RATE projection "
              f"collapses ({rr:+.3f}). The phase-coding escape is validated at the representation level -> build "
              f"the resonate-and-fire PHASOR cortex (the project's FHRR machinery) as the spiking realization.",
              flush=True)
    elif pr >= rr + 0.10:
        print(f"  PARTIAL -- phasor beats rate ({pr:+.3f} vs {rr:+.3f}) but below +0.30; the phase escape helps "
              f"but isn't sufficient alone -- needs the phasor LEARNING (similarity-matching), not just a random "
              f"phasor projection. Build the learned phasor cortex.", flush=True)
    else:
        print(f"  NEGATIVE -- the phasor projection does not beat the rate projection ({pr:+.3f} vs {rr:+.3f}); "
              f"the phase escape does not rescue the real structure at the representation level -> reconsider "
              f"(the loss is not purely the magnitude code).", flush=True)


if __name__ == "__main__":
    main()
