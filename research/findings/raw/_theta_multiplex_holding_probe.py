"""THROWAWAY cheap-first probe (CPU/numpy, stdlib+numpy only, NO protected import):
does theta-phase multiplexing HOLD multiple dense-stable items without interference,
thereby SIDESTEPPING this session's spatial separation-vs-reliability BOUNDARY?

The boundary (finding 2026-05-31-DG-...-FUNDAMENTAL-BOUNDARY): a single spatial
competitive-k-WTA stage cannot give separation AND within-item reliability from
overlapping inputs. The owner-aligned hypothesis (theta-multiplexing): separate held
items in TIME (distinct theta-phase slots) not in spatial pattern -> each item can use
a DENSE STABLE code (the EASY/reliable side of the tradeoff) while temporal segregation
carries the separation. This probe falsifies that hypothesis cheaply before any spiking build.

FAITHFUL (non-rigged) model of Lisman-Idiart theta-gamma multi-item WM:
  - theta period = P phase bins; each item i = a gamma assembly active over a gaussian
    window (FIXED biological width ~ theta/7 -> ~7 non-overlapping slots, the Miller number)
    centered at evenly-spaced phase phi_i.
  - multiplexed buffer at phase bin t: x(t) = sum_i env_i(t) * v_i + noise  (neighbors BLEED
    in time when slots crowd at large N -- the capacity limit is EMERGENT, not imposed).
  - readback item i: sample x(phi_i), decode by spatial cosine to stored codes (standard readout).
  - reliability: two independent noise draws (storage/query halves), cosine of slot content.
  - NO-PHASE CONTROL: all phi_i identical (one slot) -> every readback samples the SAME
    superposition sum_i v_i -> decode is identical for all queries -> mutual collapse (rate<=1/N).
    A phase-separated PASS while the control collapses is a genuine falsification, not trivial.

PRE-REGISTERED FROZEN BAR (never tuned by results): N>=4 dense items all read back >= 0.90
from their phase slots AND the no-phase control collapses (< 0.50). Three-state:
  RESOLVES        -> phase holding >=0.90 at N>=4 AND control <0.50 (motivates spiking build)
  BOUNDARY        -> phase ALSO fails at N>=4 even with non-crowded slots (honest negative, NO build)
  DOES-NOT-RESOLVE / CANNOT-CONCLUDE -> mixed / instrument-invalid.
Instrument-validity checked FIRST (N=1 must read back 1.0; codes normalized; control must
superpose). Multi-seed 42/43/44, multi-trial. Reports a capacity curve over N.
"""
from __future__ import annotations
import sys
import numpy as np

D = 500          # code dimension (dense)
P = 70           # theta period in phase bins
BURST_BINS = 10.0  # gamma-burst width -> ~P/BURST = 7 non-overlapping slots (Miller capacity)
SIGMA = BURST_BINS / 2.355  # gaussian sigma s.t. FWHM == BURST_BINS
NOISE = 0.15     # per-bin additive noise std (relative to unit codes)
N_LIST = [2, 4, 7, 10, 16]
SEEDS = [42, 43, 44]
N_TRIALS = 40
BAR = 0.90
CTRL_BAR = 0.50


def make_codes(n, d, overlap, rng):
    """n dense unit codes. overlap in [0,1) sets target between-item cosine ~ overlap."""
    base = rng.standard_normal(d)
    base /= np.linalg.norm(base)
    indep = rng.standard_normal((n, d))
    indep /= np.linalg.norm(indep, axis=1, keepdims=True)
    v = np.sqrt(overlap) * base[None, :] + np.sqrt(1.0 - overlap) * indep
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _env(phis, t, sigma, P):
    """wrapped-gaussian gamma envelope of each item at phase bin t."""
    dt = np.abs(phis - t)
    dt = np.minimum(dt, P - dt)  # wrap-around theta
    return np.exp(-(dt ** 2) / (2.0 * sigma ** 2))


def readback_rate(codes, phis, sigma, P, noise, rng, phase_jitter=0.0):
    """For each item, sample buffer at its phase (+ optional phase jitter), decode by spatial
    cosine. Return (correct frac, mean decode margin = correct_sim - best_competitor_sim)."""
    n, d = codes.shape
    correct = 0
    margins = []
    for i in range(n):
        t = phis[i] + (phase_jitter * rng.standard_normal() if phase_jitter > 0 else 0.0)
        env = _env(phis, t, sigma, P)                  # (n,) neighbor weights at sampled phase
        x = env @ codes + noise * rng.standard_normal(d)
        x /= (np.linalg.norm(x) + 1e-12)
        sims = codes @ x
        j = int(np.argmax(sims))
        if j == i:
            correct += 1
        comp = np.partition(sims, -2)[-2] if n > 1 else -1.0  # best non-self competitor proxy
        margins.append(float(sims[i] - (sims[j] if j != i else comp)))
    return correct / n, float(np.mean(margins))


def within_reliability(codes, phis, sigma, P, noise, rng):
    """Two independent noise draws; cosine of slot content across the two halves, mean over items."""
    n, d = codes.shape
    cs = []
    for i in range(n):
        env = _env(phis, phis[i], sigma, P)
        a = env @ codes + noise * rng.standard_normal(d)
        b = env @ codes + noise * rng.standard_normal(d)
        a /= (np.linalg.norm(a) + 1e-12)
        b /= (np.linalg.norm(b) + 1e-12)
        cs.append(float(a @ b))
    return float(np.mean(cs))


def eval_condition(n, seed, overlap, no_phase, phase_jitter=0.0):
    rng = np.random.default_rng(seed * 1000 + n + (7 if no_phase else 0) + int(overlap * 100)
                                + int(phase_jitter * 13))
    rates, rels, margins = [], [], []
    btw = []
    for _ in range(N_TRIALS):
        codes = make_codes(n, D, overlap, rng)
        # between-item cosine (instrument check / report)
        if n > 1:
            g = codes @ codes.T
            iu = np.triu_indices(n, 1)
            btw.append(float(np.mean(g[iu])))
        if no_phase:
            phis = np.full(n, P // 2, dtype=float)   # all items in ONE slot
        else:
            phis = (np.arange(n) * (P / n)).astype(float)  # evenly spaced phase slots
        r, m = readback_rate(codes, phis, SIGMA, P, NOISE, rng, phase_jitter)
        rates.append(r)
        margins.append(m)
        rels.append(within_reliability(codes, phis, SIGMA, P, NOISE, rng))
    return (float(np.mean(rates)), float(np.mean(rels)),
            float(np.mean(btw)) if btw else 0.0, float(np.mean(margins)))


def instrument_valid():
    """N=1 must read back 1.0; codes must be unit; no-phase control must actually superpose."""
    rng = np.random.default_rng(1)
    c = make_codes(4, D, 0.0, rng)
    if not np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-6):
        return False, "codes not unit-normalized"
    r1 = eval_condition(1, 42, 0.0, no_phase=False)[0]
    if r1 < 0.999:
        return False, f"N=1 sanity failed (got {r1:.3f}, expect 1.0)"
    # control at N=4 must collapse toward 1/N (mutual exclusivity)
    rc = eval_condition(4, 42, 0.0, no_phase=True)[0]
    if rc > 0.60:
        return False, f"no-phase control did not collapse (got {rc:.3f}; superposition broken)"
    return True, "ok"


def main():
    ok, msg = instrument_valid()
    print(f"[instrument] valid={ok} ({msg})")
    print(f"params: D={D} P={P} burst_bins={BURST_BINS} sigma={SIGMA:.2f} noise={NOISE} "
          f"trials={N_TRIALS} seeds={SEEDS}")
    if not ok:
        print("VERDICT: CANNOT-CONCLUDE (instrument invalid)")
        return

    for overlap in (0.0, 0.6):
        tag = "near-orthogonal dense" if overlap == 0.0 else "OVERLAPPING dense (unseparated regime)"
        print(f"\n=== codes: {tag} (target between-cos ~ {overlap}) ===")
        print(f"{'N':>3} {'phaseRead':>10} {'margin':>7} {'btwCos':>7} {'ctrlRead':>9}")
        phase_at_4plus = {}
        ctrl_at_4plus = {}
        for n in N_LIST:
            # multi-seed average of phase readback + margin
            res = [eval_condition(n, s, overlap, no_phase=False) for s in SEEDS]
            prs = np.mean([r[0] for r in res])
            mrg = np.mean([r[3] for r in res])
            bt = np.mean([r[2] for r in res])
            cr = np.mean([eval_condition(n, s, overlap, no_phase=True)[0] for s in SEEDS])
            print(f"{n:>3} {prs:>10.3f} {mrg:>7.3f} {bt:>7.3f} {cr:>9.3f}")
            if n >= 4:
                phase_at_4plus[n] = prs
                ctrl_at_4plus[n] = cr
        if overlap == 0.0:
            # PRE-REGISTERED verdict evaluated on the dense stable (near-orthogonal) codes
            n4 = phase_at_4plus.get(4, 0.0)
            c4 = ctrl_at_4plus.get(4, 1.0)
            # capacity = largest N with phase readback >= BAR
            cap = max([n for n in N_LIST if phase_at_4plus.get(n, 0.0) >= BAR] + [0])
            print(f"\n[pre-registered bar] N=4 phaseRead={n4:.3f} (bar {BAR}); "
                  f"N=4 ctrlRead={c4:.3f} (must < {CTRL_BAR}); capacity(>= {BAR})={cap} slots")
            if n4 >= BAR and c4 < CTRL_BAR:
                print(f"VERDICT: RESOLVES -- phase multiplexing HOLDS N>=4 dense items "
                      f"(>= {BAR}) while no-phase control collapses (< {CTRL_BAR}); "
                      f"capacity ~{cap} slots. Temporal separation sidesteps the spatial "
                      f"boundary. -> motivates spiking build (HARD GATE passed).")
            elif n4 < BAR:
                print("VERDICT: BOUNDARY -- phase multiplexing also fails to hold N>=4 "
                      "dense items on this model. Honest negative; NO spiking build.")
            else:
                print("VERDICT: DOES-NOT-RESOLVE -- control did not collapse; "
                      "falsification structure broken.")

    # --- realism scrutiny: does phase imprecision collapse the permissive capacity? ---
    # The no-jitter model decodes via robust high-D projection so it holds even crowded
    # slots (capacity 16 >> biological ~7). Real theta-gamma has phase jitter; sample each
    # slot at phi_i + N(0, jitter). Sweep jitter to see when capacity falls toward ~7.
    print("\n=== realism: phase-jitter sweep (near-orthogonal dense; capacity = max N "
          "with multi-seed readback >= 0.90) ===")
    print(f"{'jitter_bins':>11} {'cap':>4} | per-N readback: " + " ".join(f"N{n}" for n in N_LIST))
    for jit in (0.0, 2.0, 4.0, 6.0, 8.0):
        reads = {}
        for n in N_LIST:
            reads[n] = float(np.mean([eval_condition(n, s, 0.0, False, phase_jitter=jit)[0]
                                      for s in SEEDS]))
        cap = max([n for n in N_LIST if reads[n] >= BAR] + [0])
        row = " ".join(f"{reads[n]:.2f}" for n in N_LIST)
        print(f"{jit:>11.1f} {cap:>4} | {row}")
    print("(jitter in phase bins; burst FWHM = 10 bins; biological phase imprecision is "
          "non-zero -> if capacity falls to ~7 under moderate jitter, the model recovers the "
          "Miller number and the no-jitter capacity=16 is confirmed a permissive artifact.)")


if __name__ == "__main__":
    main()
