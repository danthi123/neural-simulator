"""Option-1 DECISIVE computation test (follow-on to the representation de-risk). The representation de-risk showed
rate-coded spikes HOLD a pre-whitened code. The remaining question (the research's fixed-Ω test core): can the
spiking COMPUTATION of whitening — lateral inhibition subtracting the correlated component from RAW input — work, or
does the low-SNR subtract-the-large-common-mode step hit the opponency wall?

Construction resolved: the project's ZCA decorrelates CONCEPTS (N×N gram), needing all concepts simultaneously present
— NOT substrate-realizable (concepts are sequential). The substrate-realizable analogue is DIMENSION-whitening: for
each concept's code x, the IT-pool DIMS inhibit each other via fixed lateral inhibition L = C^½ − I (C = D×D dim
covariance), so (I+L)⁻¹ = C^−½ and the settled r = C^−½ x = dim-whitened. This IS the fixed-Ω analytic wiring on the
realizable axis. TWO decisive sub-questions:
  (1) does DIM-whitening reduce CONCEPT coherence (i.e. help composition the way concept-whitening does)?
  (2) can RATE-CODED spiking COMPUTE the dim-whitening recurrence (vs the opponency wall on the subtraction)?
GATE: (1) dim-whiten concept-coherence << raw? (2) rate-computed concept-coherence ≈ analytic dim-whiten (holds) vs
≈ raw (the computation wall)? NO sim/ edits — numpy, the decisive cheap-first before any spiking build.
"""
import numpy as np


def make_correlated_codebook(n_concepts, n_feat, n_blocks, seed):
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((n_blocks, n_feat))
    X = np.zeros((n_concepts, n_feat))
    for i in range(n_concepts):
        X[i] = 0.75 * block[i % n_blocks] + 0.25 * rng.standard_normal(n_feat)
    return np.maximum(X, 0)


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def _normrows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def concept_whiten(X):
    """G^-1/2 X — orthonormalize CONCEPT rows (the project's _decorrelate; the 100%-composition target; NOT realizable)."""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 1e-6, None)
    return _normrows((V * (1.0 / np.sqrt(w))) @ V.T @ Xc)


def _dim_whiten_ops(X, eps=1e-3):
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    w = np.clip(w, 1e-9, None)
    Cinv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
    Csqrt = (V * np.sqrt(w)) @ V.T
    return Xc, Cinv_sqrt, Csqrt


def dim_whiten(X, eps=1e-3):
    """X C^-1/2 — decorrelate the DIMS (substrate-realizable). Q1: does it reduce CONCEPT coherence?"""
    Xc, Cinv_sqrt, _ = _dim_whiten_ops(X, eps)
    return _normrows(Xc @ Cinv_sqrt)


def ratecode_signed(M, window, peak, seed):
    # sum of `window` iid Poisson(rate) == Poisson(window*rate); estimate = that / window (mean=rate, var=rate/window)
    rng = np.random.default_rng(seed)
    scale = peak / (np.abs(M).max() + 1e-9)
    on = rng.poisson(np.maximum(M, 0.0) * scale * window) / window
    off = rng.poisson(np.maximum(-M, 0.0) * scale * window) / window
    return (on - off) / scale


def rate_compute_dim_whiten(X, window, peak, seed, iters=2000, eps=1e-3, noiseless=False):
    """STABLE leaky dynamics dr/dt = Xc - r - L·r (Jacobian -(I+L) = -C^½ ≺ 0, so convergent), small-dt Euler. The
    lateral term L·r is carried by RATE-CODED spikes (r_hat = ON/OFF Poisson over `window`), the leak/drive are exact
    (membrane). Settles to Xc C^-1/2 = dim-whitened IF the spiking computation tolerates the lateral-term noise.
    noiseless=True uses the exact r in the lateral term (validates the solver itself converges to the analytic)."""
    Xc, _, Csqrt = _dim_whiten_ops(X, eps)
    L = Csqrt - np.eye(Csqrt.shape[0])              # symmetric; I+L = C^½
    max_eig = float(np.max(np.linalg.eigvalsh(Csqrt)))
    dt = 0.5 / max_eig                              # stable step (dt < 2/max_eig)
    r = np.zeros_like(Xc)
    for it in range(iters):
        r_lat = r if noiseless else ratecode_signed(r, window, peak, seed + it)
        r = r + dt * (Xc - r - r_lat @ L.T)         # leaky integrate; lateral inhibition from (noisy) rates
    return _normrows(r)


def run(seed, n_concepts=32, n_feat=128, n_blocks=4, peak=40.0):
    X = make_correlated_codebook(n_concepts, n_feat, n_blocks, seed)
    rm, rx = coherence(X)
    cwm, cwx = coherence(concept_whiten(X))
    dwm, dwx = coherence(dim_whiten(X))
    print(f"seed={seed}: RAW coh {rm:.3f}/{rx:.3f}  |  CONCEPT-whiten {cwm:.3f}/{cwx:.3f} (target, not realizable)  |  "
          f"DIM-whiten {dwm:.3f}/{dwx:.3f} (realizable)", flush=True)
    q1 = "DIM-whiten HELPS (concept-coh << raw)" if dwx < 0.6 * rx else "DIM-whiten does NOT reduce concept coherence"
    print(f"  Q1: {q1}", flush=True)
    nlm, nlx = coherence(rate_compute_dim_whiten(X, 0, peak, seed, noiseless=True))
    print(f"  [solver control, NOISELESS] computed concept-coh {nlm:.3f}/{nlx:.3f}  (must ≈ analytic {dwx:.3f} or the "
          f"solver is broken)", flush=True)
    print(f"  Q2: rate-coded spiking COMPUTATION of dim-whitening (peak {peak:.0f} spk/win), sweep window:", flush=True)
    for window in (20, 100, 500, 2000):
        cm, cx = coherence(rate_compute_dim_whiten(X, window, peak, seed))
        gap = abs(cx - dwx)
        verdict = "COMPUTES it (≈analytic)" if gap < 0.1 else ("re-correlated (wall)" if cx > 0.6 * rx else "partial")
        print(f"    window={window:>4}: computed concept-coh {cm:.3f}/{cx:.3f}  (analytic dim-whiten {dwx:.3f}) "
              f"=> {verdict}", flush=True)


if __name__ == "__main__":
    for seed in (42, 43, 44):
        run(seed)
        print(flush=True)
