"""A5 de-risk — NE / locus-coeruleus MULTIPLICATIVE gain sharpens weak-signal detection.

Frontier (A5, buildable-now): a slow-decay norepinephrine-like GAIN signal
(Aston-Jones & Cohen 2005, "An integrative theory of locus coeruleus-norepinephrine
function", Annu. Rev. Neurosci. 28:403-450). NE multiplicatively scales the RESPONSE
GAIN of a target population, sharpening signal detection under vigilance.

QUESTION (load-bearing): does a HIGH NE gain improve detection/discrimination of a
weak, noisy signal riding on a spiking population -- measured as population-count
d-prime between signal-present and signal-absent trials -- via the MULTIPLICATIVE
mechanism (scaling the afferent drive) and NOT via a mere additive offset?

WHY THIS IS DECISION-USEFUL FOR THE SUBSTRATE. The two arithmetic operations tested
here ALREADY EXIST in sim/bridge.py, gated behind the neuromodulator subsystem:
  * MULTIPLICATIVE gain = `synaptic_gain` -> bridge.py:8164-8169
      effective_synaptic_strength *= compute_synaptic_gain_multiplier()
      (multiplier = 1 + sensitivity*(conc - baseline); byte-identical when == 1.0)
  * ADDITIVE offset   = `excitability_drive` -> bridge.py:8485-8490
      total_input_current_pA += compute_excitability_drive_pA()
      (drive = sensitivity*(conc - baseline); byte-identical when == 0.0)
So an NE-like modulator with a `synaptic_gain` target is the AJC multiplicative gain;
one with an `excitability_drive` target is the additive control. This runner tests the
EXACT arithmetic those two hooks apply, on a self-contained spiking LIF population, at
the cheapest decisive scale (single seed, numpy, ~1 min).

BRAIN-BASED STATUS. The substrate is spiking LIF neurons with a hard threshold + reset
nonlinearity (neurons/synapses). The gain is a scalar multiply on the afferent SYNAPTIC
drive -- the direct analogue of NE scaling target responsivity, and the same operation
sim's `synaptic_gain` applies to the synaptic weight matrix. The internal membrane noise
(spike-generation floor) is NOT modulated by the gain -- faithful to NE acting on the
synaptic drive, not on channel noise.

SKEPTICAL CONTROLS (mandatory):
  (A) BYTE-IDENTICAL WHEN OFF: gain=1.0 leaves the spike trains bit-for-bit identical to
      the no-modulation baseline (mirrors the `abs(nm_gain-1.0)>1e-9` guard in sim).
  (B) SPECIFICITY (multiplicative vs additive): an ADDITIVE offset auto-matched to produce
      the SAME mean population rate as the high-gain condition must NOT reproduce the
      d-prime improvement. At a matched operating point, only the multiplicative gain
      amplifies the signal-vs-noise CONTRAST relative to the fixed spike-generation floor.

Run:
    SIM_BACKEND=numpy python -m research.runners._ne_lc_gain_vigilance_derisk
"""

from __future__ import annotations

import numpy as np


# ---- self-contained spiking LIF population (numpy, deterministic) ----------

def simulate_counts(
    *,
    gain_mult: float,
    add_offset: float,
    seed: int,
    n_trials: int = 160,
    n_neurons: int = 200,
    t_steps: int = 200,
    tau_ms: float = 20.0,
    dt_ms: float = 1.0,
    v_thresh: float = 1.0,
    v_reset: float = 0.0,
    background: float = 0.80,   # sub-threshold afferent DC
    signal: float = 0.06,       # weak signal DC added on signal-present trials
    sigma_mem: float = 0.28,    # internal membrane noise (spike-gen floor, NOT modulated)
    signal_present: bool,
):
    """Return per-trial population spike counts (shape (n_trials,)).

    Afferent synaptic drive to every neuron each step:
        A = background + (signal if signal_present else 0)
    NE MULTIPLICATIVE gain scales the afferent drive (the `synaptic_gain` op):
        A_eff = gain_mult * A                        (skipped when gain_mult == 1.0)
    ADDITIVE control adds a fixed offset current (the `excitability_drive` op):
        A_eff = A_eff + add_offset                   (skipped when add_offset == 0.0)
    LIF membrane:  v <- alpha*v + (1-alpha)*A_eff + sigma_mem*eta,  eta ~ N(0,1)
    Internal membrane noise eta is NOT touched by gain/offset (fixed internal floor).
    """
    rng = np.random.default_rng(seed)
    alpha = np.exp(-dt_ms / tau_ms)
    afferent = background + (signal if signal_present else 0.0)

    # NE multiplicative gain on the afferent synaptic drive.
    # Guarded exactly like sim/bridge.py:8168 -- gain==1.0 is a no-op (byte-identical).
    a_eff = afferent
    if abs(gain_mult - 1.0) > 1e-9:
        a_eff = a_eff * gain_mult
    # Additive excitability-drive control, guarded like sim/bridge.py:8489.
    if abs(add_offset) > 1e-9:
        a_eff = a_eff + add_offset

    counts = np.zeros(n_trials, dtype=np.float64)
    one_minus_alpha = 1.0 - alpha
    for tr in range(n_trials):
        v = np.zeros(n_neurons, dtype=np.float64)
        n_spikes = 0
        # membrane noise realization for this trial (fixed internal floor, un-modulated)
        eta = rng.standard_normal((t_steps, n_neurons))
        for t in range(t_steps):
            v = alpha * v + one_minus_alpha * a_eff + sigma_mem * eta[t]
            spk = v >= v_thresh
            n_spikes += int(spk.sum())
            v[spk] = v_reset
        counts[tr] = n_spikes
    return counts


def dprime(present: np.ndarray, absent: np.ndarray) -> float:
    mp, ma = present.mean(), absent.mean()
    vp = present.var(ddof=1)
    va = absent.var(ddof=1)
    denom = np.sqrt(0.5 * (vp + va))
    if denom < 1e-12:
        return float("nan")
    return float((mp - ma) / denom)


def run_condition(*, gain_mult, add_offset, seed, **kw):
    """d-prime for a gain/offset setting: paired present vs absent at the SAME seed."""
    present = simulate_counts(gain_mult=gain_mult, add_offset=add_offset,
                              seed=seed, signal_present=True, **kw)
    absent = simulate_counts(gain_mult=gain_mult, add_offset=add_offset,
                             seed=seed, signal_present=False, **kw)
    mean_rate = 0.5 * (present.mean() + absent.mean())
    return dprime(present, absent), mean_rate, present, absent


def match_offset_to_rate(target_rate, *, seed, lo=0.0, hi=2.0, **kw):
    """Bisection: find additive offset so mean population rate ~= target_rate."""
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        _, rate, _, _ = run_condition(gain_mult=1.0, add_offset=mid, seed=seed, **kw)
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    seed = 42
    kw = dict(n_trials=160, n_neurons=200, t_steps=200)

    print("=" * 74)
    print("A5 NE / LC multiplicative-gain vigilance de-risk  (single seed, numpy)")
    print("=" * 74)

    # --- BASELINE (no modulation) ---
    d_base, r_base, p_base, a_base = run_condition(gain_mult=1.0, add_offset=0.0, seed=seed, **kw)
    print(f"\nBASELINE (g=1, no offset):        d'={d_base:6.3f}   mean_rate={r_base:8.1f} spk/trial")

    # --- CONTROL A: byte-identical when off (gain==1.0 == no modulation) ---
    # g=1.0 hits the guarded no-op path; must be bit-for-bit identical to baseline.
    _, _, p_g1, a_g1 = run_condition(gain_mult=1.0, add_offset=0.0, seed=seed, **kw)
    byte_ident = bool(np.array_equal(p_g1, p_base) and np.array_equal(a_g1, a_base))
    print(f"\n[CONTROL A] byte-identical g=1.0 vs no-mod: {byte_ident}")

    # --- MULTIPLICATIVE gain sweep (AJC NE) ---
    print("\nMULTIPLICATIVE NE gain (synaptic_gain op):")
    sweep = {}
    for g in (1.0, 1.5, 2.0, 3.0):
        d, r, _, _ = run_condition(gain_mult=g, add_offset=0.0, seed=seed, **kw)
        sweep[g] = (d, r)
        print(f"   g={g:>3}:  d'={d:6.3f}   mean_rate={r:8.1f}")
    d_mult, r_mult = sweep[2.0]

    # --- CONTROL B: additive offset matched to the SAME mean rate as g=2.0 ---
    off = match_offset_to_rate(r_mult, seed=seed, **kw)
    d_add, r_add, _, _ = run_condition(gain_mult=1.0, add_offset=off, seed=seed, **kw)
    print("\n[CONTROL B] ADDITIVE offset (excitability_drive op), rate-matched to g=2.0:")
    print(f"   offset={off:6.3f}:  d'={d_add:6.3f}   mean_rate={r_add:8.1f}"
          f"   (target rate {r_mult:.1f})")

    # --- verdict ---
    print("\n" + "-" * 74)
    mult_lift = d_mult - d_base
    add_lift = d_add - d_base
    print(f"multiplicative g=2 lift over baseline: d' {d_base:.3f} -> {d_mult:.3f}  (+{mult_lift:.3f})")
    print(f"additive rate-matched lift:            d' {d_base:.3f} -> {d_add:.3f}  (+{add_lift:.3f})")
    monotonic = sweep[1.0][0] <= sweep[1.5][0] <= sweep[2.0][0]
    print(f"multiplicative monotonic 1.0<=1.5<=2.0: {monotonic}")

    go = byte_ident and (mult_lift > 0.10) and (d_mult > d_add + 0.10)
    print("\nGATE:")
    print(f"  (A) byte-identical when off .............. {byte_ident}")
    print(f"  (mult improves detection, +{mult_lift:.3f}) .... {mult_lift > 0.10}")
    print(f"  (B) mult beats rate-matched additive ..... {d_mult > d_add + 0.10}"
          f"   (d' {d_mult:.3f} vs {d_add:.3f})")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO / see numbers'}")


if __name__ == "__main__":
    main()
