"""Option-1 DECISIVE cheap-first (from Track B research 2026-06-06): can RATE-CODED SPIKING hold a WHITENING solution
at all? The research recommends the fixed Ω=ΓᵀΓ balanced-net test; this is its cheapest, opponency-connected core.

Whitening (ZCA) produces SIGNED codes (variance-equalized, decorrelated -> low pairwise coherence). Rate codes are
NON-NEGATIVE, so a signed value must be carried as an ON/OFF pair and recovered by the ON-OFF SUBTRACTION -- which is
EXACTLY the small-signed-difference-of-two-noisy-Poisson-rates that THIS PROJECT's opponency finding
(2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED) already proved rate codes cannot do. So the prediction:
rate-coding the analytic whitened codes re-correlates them (the worst pair, a small whitened difference, is swamped by
ON-OFF Poisson noise), UNLESS the integration window is very long (unphysiological SNR).

GATE: sweep the spike-integration window. Does the rate-estimate's worst-pair coherence stay at the analytic-whitened
LOW (spikes hold whitening), or rise toward RAW (rate-noise re-correlates -> the rate-coded-spiking whitening wall is
real, the point-neuron/graded-stage boundary is FINAL + citable, consistent with the opponency precedent)?
NO sim/ edits -- numpy, the cheapest decisive de-risk before any spiking build.
"""
import numpy as np


def make_correlated_codebook(n_concepts, n_feat, n_blocks, seed):
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((n_blocks, n_feat))
    X = np.zeros((n_concepts, n_feat))
    for i in range(n_concepts):
        X[i] = 0.75 * block[i % n_blocks] + 0.25 * rng.standard_normal(n_feat)
    return np.maximum(X, 0)              # non-negative grounded features (firing-rate-like)


def coherence(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    G = np.abs(Xn @ Xn.T)
    off = G[~np.eye(len(X), dtype=bool)]
    return float(off.mean()), float(off.max())


def zca_concepts(X):
    """Decorrelate the CONCEPTS (rows) -> orthonormal rows = the analytic whitening target (low pairwise coherence)."""
    Xc = X - X.mean(0, keepdims=True)
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 1e-6, None)
    Xw = (V * (1.0 / np.sqrt(w))) @ V.T @ Xc
    return Xw / (np.linalg.norm(Xw, axis=1, keepdims=True) + 1e-12)


def ratecode_signed(codes, window, peak_hz, seed):
    """Represent SIGNED codes via an ON/OFF Poisson rate code over `window` steps; recover via the ON-OFF subtraction.
    peak_hz = the max firing rate (spikes/window-unit) for the largest-magnitude code value -> sets the SNR."""
    rng = np.random.default_rng(seed)
    scale = peak_hz / (np.abs(codes).max() + 1e-9)
    on = np.maximum(codes, 0.0) * scale       # ON-cell rate (>=0)
    off = np.maximum(-codes, 0.0) * scale     # OFF-cell rate (>=0)
    on_ct = rng.poisson(np.repeat(on[:, :, None], window, axis=2)).sum(2) / window      # noisy ON rate estimate
    off_ct = rng.poisson(np.repeat(off[:, :, None], window, axis=2)).sum(2) / window    # noisy OFF rate estimate
    return (on_ct - off_ct) / scale            # the ON-OFF subtraction -> signed estimate (the opponency op)


def run(seed, n_concepts=32, n_feat=128, n_blocks=4, peak_hz=20.0):
    X = make_correlated_codebook(n_concepts, n_feat, n_blocks, seed)
    Xw = zca_concepts(X)
    rm, rx = coherence(X)
    wm, wx = coherence(Xw)
    print(f"seed={seed}: RAW coh mean={rm:.3f}/max={rx:.3f}  |  analytic-WHITENED coh mean={wm:.3f}/max={wx:.3f}",
          flush=True)
    print(f"  rate-coded ON/OFF representation of the whitened codes (peak {peak_hz:.0f} spk/win), sweep window:",
          flush=True)
    for window in (5, 20, 100, 500, 2000):
        est = ratecode_signed(Xw, window, peak_hz, seed)
        em, ex = coherence(est)
        verdict = "HOLDS whitening" if ex < 0.4 else ("re-correlated (-> RAW)" if ex > 0.7 else "partial")
        print(f"    window={window:>4}: est coh mean={em:.3f}/max={ex:.3f}   => {verdict}", flush=True)


if __name__ == "__main__":
    for seed in (42, 43, 44):
        run(seed)
        print(flush=True)
