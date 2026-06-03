"""Cheap-first payoff probe (Direction A, step 2): does the resonator decode a MULTI-FACTOR product of the
REAL 320 concept codes where single-shot fails? -- the transfer test from random FHRR phasors (validated)
to our actual substrate codes (dense real, D=2000, Hadamard-bound, the codes the 320 conversational agent
uses). This is the prerequisite for using the resonator to unlock nested composition on the real substrate.

Decoding a nested structure = factoring a product C = c1 ⊙ c2 ⊙ … ⊙ cF (Hadamard) into its F concept
factors (search M^F). Single-shot cannot factor a product (it would need F−1 factors already). The resonator
searches M^F in superposition by iterating Hadamard-unbind ↔ codebook-cleanup. Full M=320/factor is over
capacity at D=2000 (the D-scaling found M≈64 at D=2048), so we test WITHIN-capacity M (16/32/48) on the real
codes and establish the real-code capacity; full-vocab nesting needs larger D (the documented capacity-safe
lever). numpy algebra; reuse-by-import of the real code cache; no protected-module change.

PRE-REGISTERED, FROZEN: F=3; D=2000 (real codes); n_iter=120; n_trials=30 per M; M sweep = (16, 32, 48).
  success(M) = fraction of trials with ALL F factors' argmax == the planted index, on REAL 320 codes.
  CONTROL = single-shot (n_iter=1) — must fail where the resonator succeeds.
  THREE-STATE:
    RESOLVES := resonator success >= 0.90 at M>=16 (decodes 16^3=4096 of the REAL codes) AND single-shot
                control < 0.50 there -> the resonator capability TRANSFERS to our real substrate codes.
    BOUNDARY := resonator < 0.90 even at M=16 (the real dense-Hadamard codes break the resonator).
    CANNOT-CONCLUDE := smell-test (resonator@M=16 needs the trivial regime) is malformed.

  python -m research.findings.raw._resonator_real320_probe
"""
import os
import numpy as np

CACHE = "research/findings/raw/_flatdist320_codes.npz"
F = 3
D_EXPECT = 2000
N_ITER = 120
N_TRIALS = 30
M_SWEEP = (16, 32, 48)
SUCCESS_BAR = 0.90


def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def resonator_real(C, codebooks, n_iter):
    """Real-Hadamard resonator: estimate each factor; iteratively Hadamard-unbind the others and clean up
    onto the factor's codebook (real superposition projection V Vᵀ). codebooks[i] = (D, M) real."""
    F = len(codebooks)
    est = [_unit(cb.sum(axis=1)) for cb in codebooks]
    for _ in range(n_iter):
        new = []
        for i in range(F):
            others = np.ones(C.shape[0])
            for j in range(F):
                if j != i:
                    others = others * est[j]
            xi = C * others                              # Hadamard unbind (self-multiply, cleanup-recoverable)
            proj = codebooks[i] @ (codebooks[i].T @ xi)  # codebook superposition projection (cleanup)
            new.append(_unit(proj))
        est = new
    return [int(np.argmax(codebooks[i].T @ est[i])) for i in range(F)]


def success_rate(all_codes, M, n_iter, seed_base):
    V = all_codes.shape[1]
    ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(seed_base * 100003 + t)
        # F independent random M-subsets of the 320 real codes = the F factor codebooks
        cbs = [all_codes[:, rng.choice(V, M, replace=False)] for _ in range(F)]
        true = [int(rng.integers(0, M)) for _ in range(F)]
        C = np.ones(all_codes.shape[0])
        for cb, k in zip(cbs, true):
            C = C * cb[:, k]                              # Hadamard product = the nested bound structure
        dec = resonator_real(C, cbs, n_iter)
        ok += int(dec == true)
    return ok / N_TRIALS


def main():
    if not os.path.exists(CACHE):
        print(f"CANNOT-RUN: {CACHE} missing.", flush=True)
        return
    d = np.load(CACHE)
    words = [str(w) for w in d["_words"]]
    codes = np.stack([_unit(np.asarray(d[w], dtype=np.float64)) for w in words], axis=1)  # D x 320
    Dreal = codes.shape[0]
    print(f"=== resonator on REAL 320 codes (Hadamard; D={Dreal}, V={len(words)}, F={F}) ===", flush=True)
    curve, ctrl = [], []
    for M in M_SWEEP:
        s = success_rate(codes, M, N_ITER, seed_base=1)
        c = success_rate(codes, M, 1, seed_base=3)
        curve.append((M, s)); ctrl.append((M, c))
        print(f"  M={M:>3}  resonator={s:.2f}  single-shot(ctrl)={c:.2f}", flush=True)
    cap = 0
    for M, s in curve:
        if s >= SUCCESS_BAR:
            cap = M
        else:
            break
    s16 = dict(curve)[16]
    ctrl16 = dict(ctrl)[16]
    print("\n-- pre-registered evaluation --", flush=True)
    print(f"  resonator @ M=16: {s16:.2f}  | single-shot @ M=16: {ctrl16:.2f} (must be < 0.50)", flush=True)
    print(f"  resonator operational capacity on REAL codes: M={cap}", flush=True)
    if s16 >= SUCCESS_BAR and ctrl16 < 0.50:
        verdict = (f"RESOLVES (resonator factors REAL-320 products to M={cap}, e.g. {cap}^{F}={cap**F} search, "
                   f"where single-shot fails @ {ctrl16:.2f}) -> the capability TRANSFERS to our real substrate")
    elif s16 < SUCCESS_BAR:
        verdict = (f"BOUNDARY (resonator {s16:.2f} < {SUCCESS_BAR} even at M=16 on real dense-Hadamard codes -> "
                   f"the real-code structure degrades the resonator)")
    else:
        verdict = "CANNOT-CONCLUDE (control single-shot also succeeds -> no genuine search advantage)"
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
