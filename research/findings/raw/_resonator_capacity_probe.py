"""Cheap-first numpy probe (Direction A): does a RESONATOR-NETWORK decoder + NOISE injection factor
multi-factor FHRR products that our single-shot decode CANNOT -- the genuinely-untried mechanism from the
2026-06-03 deep-research synthesis (Frady-Kent-Olshausen-Sommer 2020 resonator networks; Kymn et al. 2024
noise injection >=50x capacity).

WHY this maps onto OUR wall: our composition hit a NESTING wall -- the hierarchical-320 (a 2nd binding
level) scored 0.000 (the "multi-hop SNR wall"), forcing the flat-distinct single-binding-level workaround.
Decoding a NESTED bound structure = factoring a PRODUCT C = x1 (x) x2 (x) ... (x) xF of F unknown factors,
each drawn from a codebook of size M. A SINGLE-SHOT decode cannot factor a product (it would need to know
F-1 of the factors already); the search space is M^F. The RESONATOR NETWORK searches M^F IN SUPERPOSITION
by iterating unbind <-> codebook-cleanup. We do NOT have one (grep confirms). This probe tests, in the FHRR
ALGEBRA (numpy; the algebra our resonate-and-fire substrate realizes), whether it works on OUR code type,
BEFORE any spiking build (the standing cheap-first-before-spiking discipline).

FHRR codes here = unit-magnitude complex phasor vectors (random phases) -- identical algebra to
resonate_fire_fhrr.py (phase add = bind; phase subtract = unbind; codebook inner product = cleanup).

PRE-REGISTERED, FROZEN (never tuned by results):
  F=3 factors; D=1024; n_iter=200; n_trials=60 per M; M sweep = (4, 8, 16, 32, 64, 128).
  success(M, decoder) = fraction of trials with ALL F factors' argmax == the true planted index.
  operational_capacity(decoder) = largest swept M with success >= 0.90.
  CONTROL (reproduce-the-failure) = "single-shot" = the resonator run for ONE iteration from the uniform
    init (no iterative search). It MUST fail where the resonator succeeds, else the test is vacuous.
  THREE-STATE:
    RESOLVES  := resonator operational_capacity >= 16 (decodes M^F = 16^3 = 4096 the single-shot cannot)
                 AND single-shot success < 0.50 at that M (control reproduces the failure)
                 AND resonator+noise operational_capacity >= resonator (noise does not hurt; ideally >).
    BOUNDARY  := resonator works in the trivial regime but operational_capacity < 16 (too low to matter for
                 nesting), OR noise strictly HURTS (capacity drops).
    CANNOT-CONCLUDE := smell-test fails (resonator near-0 even at M=4 -> implementation invalid) OR the
                 control single-shot succeeds where resonator does (no genuine search advantage).
  SMELL-TEST (scrutinise a PASS harder than a FAIL): at the smallest M=4, the resonator MUST succeed
    >= 0.95 (the trivial regime is easy); if not, the implementation is broken -> CANNOT-CONCLUDE, not a
    false RESOLVE.

stdlib + numpy only; no protected-module import; throwaway probe (evidence recorded to a .txt).
  python -m research.findings.raw._resonator_capacity_probe
"""
import numpy as np

# ---- frozen config ----
F = 3
D = 1024
N_ITER = 200
N_TRIALS = 60
M_SWEEP = (4, 8, 16, 32, 64, 128)
RESOLVE_CAP = 16
SUCCESS_BAR = 0.90
SMELL_M = 4
SMELL_BAR = 0.95
NOISE_LEVEL = 0.30           # injected-noise std (Kymn-style); frozen


def _unit(v):
    return v / (np.abs(v) + 1e-12)


def make_codebook(M, rng):
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(D, M)))


def bind_product(codebooks, idxs):
    c = np.ones(D, dtype=complex)
    for cb, k in zip(codebooks, idxs):
        c = c * cb[:, k]
    return c


def resonator(C, codebooks, n_iter, noise, rng):
    """Standard resonator-network factorisation in FHRR algebra: estimate each factor; iteratively unbind
    the others and clean up onto the factor's codebook (superposition projection). Returns decoded indices."""
    est = [_unit(cb.sum(axis=1)) for cb in codebooks]      # init = superposition of each codebook
    for _ in range(n_iter):
        new = []
        for i in range(F):
            others = np.ones(D, dtype=complex)
            for j in range(F):
                if j != i:
                    others = others * est[j]
            xi = C * np.conj(others)                        # unbind the others
            if noise > 0.0:
                xi = xi + noise * (rng.standard_normal(D) + 1j * rng.standard_normal(D))
            sims = codebooks[i].conj().T @ xi               # M complex similarities
            proj = codebooks[i] @ sims                      # project onto codebook span (the cleanup)
            new.append(_unit(proj))
        est = new
    return [int(np.argmax(np.abs(codebooks[i].conj().T @ est[i]))) for i in range(F)]


def success_rate(M, n_iter, noise, seed_base):
    ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(seed_base * 100003 + t)
        cbs = [make_codebook(M, rng) for _ in range(F)]
        true = [int(rng.integers(0, M)) for _ in range(F)]
        C = bind_product(cbs, true)
        dec = resonator(C, cbs, n_iter=n_iter, noise=noise, rng=rng)
        ok += int(dec == true)
    return ok / N_TRIALS


def op_capacity(curve):
    cap = 0
    for M, s in curve:
        if s >= SUCCESS_BAR:
            cap = M
        else:
            break
    return cap


def main():
    print("=== resonator-network capacity probe (FHRR algebra; F=%d D=%d) ===" % (F, D), flush=True)
    res_curve, noise_curve, ctrl_curve = [], [], []
    for M in M_SWEEP:
        s_res = success_rate(M, N_ITER, 0.0, seed_base=1)
        s_noi = success_rate(M, N_ITER, NOISE_LEVEL, seed_base=2)
        s_ctl = success_rate(M, 1, 0.0, seed_base=3)        # single-shot control (1 iteration)
        res_curve.append((M, s_res)); noise_curve.append((M, s_noi)); ctrl_curve.append((M, s_ctl))
        print(f"  M={M:>4}  resonator={s_res:.2f}  resonator+noise={s_noi:.2f}  single-shot(ctrl)={s_ctl:.2f}",
              flush=True)

    cap_res = op_capacity(res_curve)
    cap_noi = op_capacity(noise_curve)
    smell = dict(res_curve)[SMELL_M]
    ctrl_at_cap = dict(ctrl_curve).get(cap_res, 1.0) if cap_res else 1.0

    print("\n-- pre-registered evaluation --", flush=True)
    print(f"  smell-test (resonator@M={SMELL_M} >= {SMELL_BAR}): {smell:.2f}  -> {'OK' if smell>=SMELL_BAR else 'FAIL'}",
          flush=True)
    print(f"  resonator operational capacity: M={cap_res}", flush=True)
    print(f"  resonator+noise operational capacity: M={cap_noi}", flush=True)
    print(f"  single-shot control success @ M={cap_res}: {ctrl_at_cap:.2f} (must be < 0.50)", flush=True)

    if smell < SMELL_BAR:
        verdict = "CANNOT-CONCLUDE (smell-test failed: resonator broken even in trivial regime)"
    elif cap_res >= RESOLVE_CAP and ctrl_at_cap < 0.50 and cap_noi >= cap_res:
        verdict = (f"RESOLVES (resonator factors M^F={cap_res}^{F}={cap_res**F} where single-shot fails "
                   f"@ {ctrl_at_cap:.2f}; noise capacity M={cap_noi} >= resonator M={cap_res})")
    elif cap_res >= RESOLVE_CAP and ctrl_at_cap >= 0.50:
        verdict = "CANNOT-CONCLUDE (control single-shot also succeeds -> no genuine search advantage)"
    else:
        verdict = (f"BOUNDARY (resonator capacity M={cap_res} < {RESOLVE_CAP}, or noise hurts "
                   f"M={cap_noi}<{cap_res}; resonator+noise insufficient on these codes)")
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
