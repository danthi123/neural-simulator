"""Option-1 LEARNING test (the genuine path forward after the computation de-risk showed handed-in analytic lateral
inhibition computes whitening in spikes). The remaining gap: can a LOCAL stable rule LEARN the whitening lateral
inhibition L from RAW correlated input (vs handing in L=C^½−I)?

Research nuance (Track B): SAILnet/CM stable rules (c_ij→p²) fix attempt-#2's instability BUT achieve only mean/RMS
decorrelation AND pre-whiten their input. KEY distinction this tests: those use the SPARSE target (c_ij→p²); the
WHITENING target is ΔM ∝ ⟨yyᵀ⟩ − I (drive the output covariance to IDENTITY = decorrelate + variance-equalize). The
linear settled output y = (I+M)⁻¹ x, so ⟨yyᵀ⟩ = I ⟺ (I+M)² = C ⟺ M = C^½ − I = the analytic whitening L. So the
whitening-target rule has a FIXED POINT at the worst-pair-whitening solution. Does it converge there from raw input?

Controls (rigor — the last de-risk had a solver bug): (A) analytic M = C^½−I (must give worst-pair coh ~0.037);
(B) naive ΔM ∝ ⟨yyᵀ⟩ (no −I target = attempt #2 — must diverge/not whiten); (C) the whitening rule ΔM ∝ ⟨yyᵀ⟩ − I.
GATE: does (C) reach (A)'s worst-pair coh, and does (B) fail? NO sim/ edits — numpy rate/continuous model first.
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


def analytic_M(X, eps=1e-3):
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(C + eps * np.eye(C.shape[0]))
    w = np.clip(w, 1e-9, None)
    Csqrt = (V * np.sqrt(w)) @ V.T
    return Csqrt - np.eye(C.shape[0]), Xc


def learn_lateral(X, rule, n_iters=4000, eta=0.02, eps=1e-3):
    """Batch learn the lateral M. y = (I+M)^-1 Xc; ΔM = eta*(Cyy - target). rule='whiten' -> target=I (whitening fixed
    point M=C^½-I); rule='naive' -> target=0 (attempt #2, no fixed point). M kept symmetric."""
    Xc = X - X.mean(0, keepdims=True)
    D = Xc.shape[1]
    I = np.eye(D)
    M = np.zeros((D, D))
    target = I if rule == "whiten" else np.zeros((D, D))
    diverged = False
    for it in range(n_iters):
        try:
            Y = np.linalg.solve(I + M, Xc.T).T
        except np.linalg.LinAlgError:
            diverged = True
            break
        if not np.all(np.isfinite(Y)) or np.abs(Y).max() > 1e6:
            diverged = True
            break
        Cyy = Y.T @ Y / len(Xc)
        M = M + eta * (Cyy - target)
        M = 0.5 * (M + M.T)
    Y = np.linalg.solve(I + M, Xc.T).T if not diverged else Xc
    return _normrows(Y), M, diverged


def run(seed, n_concepts=32, n_feat=128, n_blocks=4):
    X = make_correlated_codebook(n_concepts, n_feat, n_blocks, seed)
    rm, rx = coherence(X)
    Ma, Xc = analytic_M(X)
    Ya = _normrows(np.linalg.solve(np.eye(n_feat) + Ma, Xc.T).T)
    am, ax = coherence(Ya)
    print(f"seed={seed}: RAW {rm:.3f}/{rx:.3f}  | (A) analytic M=C^½-I -> coh {am:.3f}/{ax:.3f} (worst-pair target)",
          flush=True)
    Yw, Mw, dw = learn_lateral(X, "whiten")
    wm, wx = coherence(Yw)
    mgap = float(np.linalg.norm(Mw - Ma) / (np.linalg.norm(Ma) + 1e-9))
    print(f"  (C) LEARNED whitening rule (ΔM∝⟨yyᵀ⟩−I): coh {wm:.3f}/{wx:.3f}  | ‖M_learned−M_analytic‖/‖M_analytic‖ "
          f"= {mgap:.3f}  | diverged={dw}", flush=True)
    Yn, Mn, dn = learn_lateral(X, "naive")
    nm, nx = coherence(Yn)
    print(f"  (B) NAIVE rule (ΔM∝⟨yyᵀ⟩, no target = attempt #2): coh {nm:.3f}/{nx:.3f}  | diverged={dn}", flush=True)
    # NOTE: low coherence ALONE is NOT sufficient — a blown-up M collapses the output toward noise, which is also
    # decorrelated (low coherence) but is NOT the whitening solution. Require the learned M to actually MATCH the
    # analytic (mgap small) before claiming the rule learned the whitening.
    if not dw and wx < 0.15 and mgap < 0.5:
        verdict = "LOCAL LEARNING WHITENS (M matches analytic)"
    elif not dw and wx < 0.15 and mgap >= 0.5:
        verdict = f"FALSE POSITIVE: low coh but M blew up (mgap={mgap:.0f}) -> output collapsed to noise, NOT whitening"
    else:
        verdict = "learned rule does NOT reach worst-pair"
    print(f"  => {verdict}", flush=True)


if __name__ == "__main__":
    for seed in (42, 43, 44):
        run(seed)
        print(flush=True)
