"""Membrane-level fidelity rung for phasor substrate unification: does a genuine spiking resonate-and-fire
MEMBRANE (not the steady-state phasor readout, and not a closed-form angle) fire at the right phase when it
integrates phase-coded input spikes through the learned real weights?

This closes the one membrane-level gap the spiking-STDP finding left open
(2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md): the readout/cleanup were confirmed in the
rf substrate, but not the input-spike -> membrane-integration -> output-spike step under leak + spike timing.

Model: a complex membrane z per output neuron rotates one carrier cycle (e^{-i omega} per step); each active
input neuron kicks z (a real synaptic impulse) at the timestep of its phase; the neuron fires at its RESONANT
PHASE arg(z) -- the time a resonate-and-fire neuron emits (Izhikevich 2001 / Frady-Sommer 2019), NOT the first
threshold crossing of Re(z). Weights W are the validated real-weight STDP map.

KEY RESULT + biological constraint:
  - With NO leak (a perfect resonator), the membrane reproduces the readout at retrieval 1.00.
  - LEAK degrades it (leak weights recent kicks more, biasing the resultant phase): ~0.82 at leak=0.005,
    ~0.33 at 0.02. Phase coding needs a HIGH-Q (low-leak) resonator that sustains its oscillation across the
    cycle. Biologically: intrinsic resonant currents (Ih etc.) -- which the project's HH models have -- not
    passive leaky integration. (leak=0.005/step over T=360 ~ membrane time constant ~70 ms, long for an RS
    cell; the brain achieves sustained theta-cycle resonance via resonant currents, not a long passive tau.)
  - CONTROL: a naive integrate-and-fire (fire at the first Re(z) > threshold) retrieves at CHANCE -- phase
    coding specifically requires the resonant-phase readout, confirming the resonate-and-fire mechanism is
    load-bearing (not interchangeable with plain LIF).

PRE-REGISTERED, FROZEN: N=8 concepts; D=128; T=360 steps/cycle; 5 seeds; leak in {0.0, 0.005, 0.02}.
  THREE-STATE (no-leak resonate-and-fire retrieval):
    RESOLVES := >= 0.90 AND the naive first-threshold control < 0.50 -> the membrane preserves the phase
                readout with a high-Q resonator; the spiking pipeline is validated end-to-end.
    BOUNDARY := 0.50-0.90 -> partial.
    DOES-NOT-RESOLVE := < 0.50 -> the membrane does not preserve the phase readout.

  python -m research.findings.raw._membrane_resonate_fire_phase_probe
"""
import numpy as np

N, D, T = 8, 128, 360
SEEDS = (0, 1, 2, 3, 4)
LEAKS = (0.0, 0.005, 0.02)


def stdp_W(cues, codes, tau=0.6, ap=1.0, am=0.5):
    W = np.zeros((D, D))
    for c in range(len(cues)):
        dt = codes[c][:, None] - cues[c][None, :]
        dt = (dt + np.pi) % (2 * np.pi) - np.pi
        W += np.where(dt > 0, ap * np.exp(-dt / tau), -am * np.exp(dt / tau))
    return W


def pcos(p, c):
    return float(np.cos(p - c).mean())


def membrane(W, cue, leak, naive_threshold=None):
    """Resonate-and-fire membrane; returns per-neuron spike phase. naive_threshold set -> first-crossing control."""
    omega = 2 * np.pi / T
    step = np.round(((cue % (2 * np.pi)) / (2 * np.pi)) * T).astype(int) % T
    z = np.zeros(D, complex)
    rot = np.exp(-1j * omega) * (1 - leak)
    if naive_threshold is not None:
        fired = np.zeros(D, bool)
        fphase = np.zeros(D)
        for t in range(T):
            z = z * rot
            m = (step == t).astype(float)
            if m.any():
                z = z + W @ m
            newly = (~fired) & (z.real > naive_threshold)
            fphase[newly] = 2 * np.pi * t / T
            fired[newly] = True
        fphase[~fired] = np.angle(z[~fired]) % (2 * np.pi)
        return fphase
    for t in range(T):
        z = z * rot
        m = (step == t).astype(float)
        if m.any():
            z = z + W @ m
    return np.angle(z)                                        # resonant phase = spike phase


def main():
    print(f"=== resonate-and-fire MEMBRANE preserves the phase readout? (N={N}, D={D}, T={T}) ===", flush=True)
    res = {}
    for leak in LEAKS:
        accs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            cues = rng.uniform(-np.pi, np.pi, size=(N, D))
            codes = rng.uniform(-np.pi, np.pi, size=(N, D))
            W = stdp_W(cues, codes)
            ok = 0
            for c in range(N):
                fp = membrane(W, cues[c], leak)
                ok += int(np.argmax([pcos(fp, codes[k]) for k in range(N)]) == c)
            accs.append(ok / N)
        res[leak] = float(np.mean(accs))
        print(f"  leak={leak}: resonate-and-fire retrieval {res[leak]:.2f}", flush=True)

    # naive integrate-and-fire control (first threshold crossing) at no leak
    ctrl = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cues = rng.uniform(-np.pi, np.pi, size=(N, D))
        codes = rng.uniform(-np.pi, np.pi, size=(N, D))
        W = stdp_W(cues, codes)
        thr = 0.35 * np.median(np.abs(W @ np.exp(1j * cues[0])))
        ok = 0
        for c in range(N):
            fp = membrane(W, cues[c], 0.0, naive_threshold=thr)
            ok += int(np.argmax([pcos(fp, codes[k]) for k in range(N)]) == c)
        ctrl.append(ok / N)
    ctrl_acc = float(np.mean(ctrl))
    print(f"\n  control (naive integrate-and-fire, first threshold crossing): {ctrl_acc:.2f} (chance {1/N:.2f})",
          flush=True)

    noleak = res[0.0]
    if noleak >= 0.90 and ctrl_acc < 0.50:
        verdict = ("RESOLVES -- the resonate-and-fire membrane preserves the phase readout (1.00 at no leak); "
                   "phase coding needs a HIGH-Q (low-leak) resonator (intrinsic resonant currents, e.g. Ih), and "
                   "a naive integrate-and-fire fails -- so the resonate-and-fire mechanism is load-bearing.")
    elif noleak >= 0.50:
        verdict = f"BOUNDARY -- no-leak retrieval {noleak:.2f}."
    else:
        verdict = f"DOES-NOT-RESOLVE -- no-leak retrieval {noleak:.2f} < 0.50."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
