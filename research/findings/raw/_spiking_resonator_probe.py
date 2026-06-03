"""Cheap-first probe (Direction A, step 2 — the DECISIVE one): does the resonator network survive the
SPIKING resonate-and-fire substrate? The algebra probe RESOLVED (resonator factors multi-factor products
single-shot cannot); the prior 2026-05-22 caveat was "composition trivial in ALGEBRA, impossible in
SUBSTRATE." This probe runs the SAME iterative resonator factorization but with the genuine spiking
operations from research/runners/resonate_fire_fhrr.py: `rf_unbind` (the resonate-and-fire phase-subtract,
the source of any spiking degradation) and `rf_resonate` readout, with a soft codebook projection cleanup.

If it RESOLVES, the resonator is real on our biology-faithful substrate -> it is the missing decode stage
that re-enables NESTED composition (the hierarchical structures the flat-distinct workaround had to avoid).
If BOUNDARY, the spiking realization degrades the resonator -> an honest negative that extends the
"substrate fails" caveat to the resonator specifically.

Reuse-by-import only (rf_*, _to_phasor, helpers); no protected-module change. Smaller scale than the algebra
probe because rf_resonate time-steps a full cycle (CYCLE_STEPS) per call -- tractable, still decisive.

PRE-REGISTERED, FROZEN (never tuned by results):
  F=3; D=256; n_iter=25; n_trials=15 per M; M sweep = (4, 8, 16, 32).
  success(M) = fraction of trials with ALL F factors' argmax == the planted index (genuine rf_* throughout).
  operational_capacity = largest swept M with success >= 0.90.
  CONTROL = single-shot spiking (n_iter=1) -- must fail where the iterative resonator succeeds.
  THREE-STATE:
    RESOLVES := spiking resonator operational_capacity >= 16 (decodes M^F=16^3=4096 IN SPIKES)
                AND single-shot control success < 0.50 at that M.
    BOUNDARY := operational_capacity < 16 (the spiking substrate degrades the resonator below nesting-useful).
    CANNOT-CONCLUDE := smell-test fails (resonator@M=4 < 0.90 -> spiking realization broken in trivial regime).
  SMELL-TEST: resonator@M=4 >= 0.90 (the trivial regime must work in spikes; else CANNOT-CONCLUDE).

  python -m research.findings.raw._spiking_resonator_probe
"""
import numpy as np

from research.runners.resonate_fire_fhrr import rf_bind, rf_unbind, rf_resonate, _to_phasor, CYCLE_STEPS
from research.runners.spiking_phasor_fhrr import phases_to_spikes

F = 3
D = 256
N_ITER = 25
N_TRIALS = 15
M_SWEEP = (4, 8, 16, 32)
RESOLVE_CAP = 16
SUCCESS_BAR = 0.90
SMELL_M = 4
SMELL_BAR = 0.90


def rand_code(rng):
    return phases_to_spikes(rng.uniform(0.0, 1.0, size=D), CYCLE_STEPS)


def codebook_matrix(cb_spikes):
    return np.stack([_to_phasor(c, CYCLE_STEPS) for c in cb_spikes], axis=1)   # D x M


def proj_readout(xi_spikes, S):
    """Soft codebook projection cleanup S @ (S^H @ z), read out as spikes via the resonate-and-fire neuron."""
    z = _to_phasor(xi_spikes, CYCLE_STEPS)
    proj = S @ (S.conj().T @ z)
    return rf_resonate(proj, CYCLE_STEPS)


def bind_all(codebooks_spikes, idxs):
    c = codebooks_spikes[0][idxs[0]]
    for cb, k in zip(codebooks_spikes[1:], idxs[1:]):
        c = rf_bind(c, cb[k], CYCLE_STEPS)
    return c


def spiking_resonator(C, codebooks_spikes, mats, n_iter):
    est = [rf_resonate(M.sum(axis=1), CYCLE_STEPS) for M in mats]    # superposition init -> spikes
    for _ in range(n_iter):
        new = []
        for i in range(F):
            xi = C
            for j in range(F):
                if j != i:
                    xi = rf_unbind(xi, est[j], CYCLE_STEPS)           # genuine spiking unbind
            new.append(proj_readout(xi, mats[i]))
        est = new
    out = []
    for i in range(F):
        z = _to_phasor(est[i], CYCLE_STEPS)
        out.append(int(np.argmax(np.abs(mats[i].conj().T @ z))))
    return out


def success_rate(M, n_iter, seed_base):
    ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(seed_base * 100003 + t)
        cbs = [[rand_code(rng) for _ in range(M)] for _ in range(F)]
        mats = [codebook_matrix(cb) for cb in cbs]
        true = [int(rng.integers(0, M)) for _ in range(F)]
        C = bind_all(cbs, true)
        dec = spiking_resonator(C, cbs, mats, n_iter)
        ok += int(dec == true)
    return ok / N_TRIALS


def main():
    print(f"=== SPIKING resonator probe (resonate-and-fire FHRR; F={F} D={D} CYCLE_STEPS={CYCLE_STEPS}) ===",
          flush=True)
    curve, ctrl = [], []
    for M in M_SWEEP:
        s = success_rate(M, N_ITER, seed_base=1)
        c = success_rate(M, 1, seed_base=3)
        curve.append((M, s)); ctrl.append((M, c))
        print(f"  M={M:>3}  spiking_resonator={s:.2f}  single-shot(ctrl)={c:.2f}", flush=True)
    cap = 0
    for M, s in curve:
        if s >= SUCCESS_BAR:
            cap = M
        else:
            break
    smell = dict(curve)[SMELL_M]
    ctrl_at_cap = dict(ctrl).get(cap, 1.0) if cap else 1.0
    print("\n-- pre-registered evaluation --", flush=True)
    print(f"  smell-test (resonator@M={SMELL_M} >= {SMELL_BAR}): {smell:.2f}", flush=True)
    print(f"  spiking resonator operational capacity: M={cap}", flush=True)
    print(f"  single-shot control @ M={cap}: {ctrl_at_cap:.2f} (must be < 0.50)", flush=True)
    if smell < SMELL_BAR:
        verdict = "CANNOT-CONCLUDE (spiking resonator broken in trivial regime)"
    elif cap >= RESOLVE_CAP and ctrl_at_cap < 0.50:
        verdict = (f"RESOLVES (spiking resonator factors M^F={cap}^{F}={cap**F} IN SPIKES where single-shot "
                   f"fails @ {ctrl_at_cap:.2f}) -> the resonator decode SURVIVES the substrate")
    else:
        verdict = (f"BOUNDARY (spiking resonator capacity M={cap} < {RESOLVE_CAP}; the resonate-and-fire "
                   f"realization degrades the resonator -> substrate-fails caveat extends to the resonator)")
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
