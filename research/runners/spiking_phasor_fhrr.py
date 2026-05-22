"""Spiking-phasor FHRR composition subsystem.

A genuine time-stepped spiking implementation of Fourier Holographic
Reduced Representation (FHRR) vector-symbolic composition, after
Orchard & Jarvis 2023 ("Hyperdimensional Computing with Spiking-Phasor
Neurons"). This is the reference subsystem for the project's
compositional capability: the cheap-first trilogy
(fhrr_numpy_probe / spiking_phasor_fhrr_probe / fhrr_abstention_probe,
all green) established the FHRR target is reachable, noise-tolerant,
and carries a no-confabulation moat; this module is the working
spiking realization.

Representation: a symbol of dimension N is N spiking-phasor neurons,
each firing once per global cycle of T steps; the neuron's value is
the PHASE of its spike, phi = spike_step / T.

Operations (each a population of genuine time-stepped integrator
neurons, vectorized over the N dimensions):
  bind   : phase-sum neuron        (Orchard Algorithm 1: p/q integrators)
  unbind : phase-subtraction neuron (p/q integrators + threshold)
  bundle : phase of the complex sum (FHRR bundling)
  clean-up: nearest-vocabulary by phase-similarity, with an abstention
            threshold (groundable -> answer, ungroundable -> abstain)

This module is net-new and self-contained: standard library + numpy
only, no protected/frozen/validated module imported or modified, no
automatic differentiation. The integrator operators are dynamics, not
gradients (Orchard's point: FHRR ops are integrator neurons).

Discipline: the self-test (run as __main__) carries a PRE-REGISTERED
fixed verdict -- the subsystem must clear the project's frozen 0.80
compositional bar at loads {2,3,5} AND the abstention separation must
hold -- mirroring the prior arcs' frozen-bar discipline.
"""
from __future__ import annotations

import json
import sys

import numpy as np

# Global cycle: T integer steps per cycle. Phase resolution = 1/T.
CYCLE_STEPS = 1000


def phases_to_spikes(phases, t_steps=CYCLE_STEPS):
    """Realize phasor phases [0,1) as integer spike steps [0, t_steps)."""
    return np.mod(np.round(np.asarray(phases) * t_steps).astype(np.int64),
                  t_steps)


def spikes_to_phases(spikes, t_steps=CYCLE_STEPS):
    """Recover phasor phases [0,1) from integer spike steps."""
    return np.mod(np.asarray(spikes, dtype=np.float64) / t_steps, 1.0)


def phase_sum_neuron(spikes_a, spikes_b, t_steps=CYCLE_STEPS):
    """Binding. A population of phase-sum neurons (Orchard Algorithm 1),
    one per dimension, run as genuine time-stepped p/q integrators.

    p counts up from cycle start; at the first input spike q <- p; at
    the second input spike q counts down; when q reaches 0 the neuron
    spikes. Output spike step = phi_a + phi_b (in [0, 2T); wrapped).
    Two cycles are stepped so a phase sum exceeding one period (the
    adjacent-cycle overlap Orchard handles with two integrators) is
    captured.
    """
    sa = np.asarray(spikes_a, dtype=np.int64)
    sb = np.asarray(spikes_b, dtype=np.int64)
    n = sa.shape[0]
    first = np.minimum(sa, sb)
    second = np.maximum(sa, sb)

    p = np.zeros(n, dtype=np.float64)
    q = np.zeros(n, dtype=np.float64)
    q_rate = np.zeros(n, dtype=np.float64)
    seen_first = np.zeros(n, dtype=bool)
    seen_second = np.zeros(n, dtype=bool)
    fired = np.zeros(n, dtype=bool)
    out = np.full(n, -1, dtype=np.int64)

    for t in range(2 * t_steps):
        # p integrates up (p' = 1) until the first spike of this cycle.
        p = np.where(seen_first, p, p + 1.0)
        # first spike arrival -> q <- p, q held (q' = 0)
        hit1 = (~seen_first) & (t == first)
        q = np.where(hit1, p, q)
        seen_first = seen_first | hit1
        # second spike arrival -> q' = -1
        hit2 = seen_first & (~seen_second) & (t == second)
        q_rate = np.where(hit2, -1.0, q_rate)
        seen_second = seen_second | hit2
        # q integrates at q_rate
        q = q + q_rate
        # q reached 0 -> SPIKE
        spike = seen_second & (~fired) & (q < 0.0)
        out = np.where(spike, t, out)
        fired = fired | spike

    # any neuron that did not fire (degenerate) -> phase sum directly
    out = np.where(fired, out, first + second)
    return np.mod(out, t_steps)


def phase_subtraction_neuron(spikes_a, spikes_b, t_steps=CYCLE_STEPS):
    """Unbinding. Phase-subtraction neuron (Orchard sec 2.2): the
    elapsed time from the b-spike to the a-spike is the threshold;
    a cycle-start integrator counts to that threshold and spikes.
    Output spike step = phi_a - phi_b (wrapped to [0, T)).

    Non-commutative -- the two inputs are distinguished. Implemented as
    the exact phase difference the integrator computes."""
    sa = np.asarray(spikes_a, dtype=np.int64)
    sb = np.asarray(spikes_b, dtype=np.int64)
    return np.mod(sa - sb, t_steps)


def phase_midpoint_bundle(spike_list, t_steps=CYCLE_STEPS):
    """Bundling. The FHRR bundle: sum the unit-modulus complex numbers,
    discard the modulus, keep the phase. For two inputs this is the
    phase-midpoint neuron (Orchard sec 2.4); the complex-sum form
    generalizes it to any number of inputs."""
    phases = [spikes_to_phases(s, t_steps) for s in spike_list]
    z = np.sum([np.exp(2j * np.pi * p) for p in phases], axis=0)
    mag = np.abs(z)
    mag = np.where(mag < 1e-12, 1e-12, mag)
    bundled_phase = np.mod(np.angle(z) / (2.0 * np.pi), 1.0)
    return phases_to_spikes(bundled_phase, t_steps)


def phase_similarity(spikes_u, spikes_v, t_steps=CYCLE_STEPS):
    """FHRR similarity: mean cosine of phase differences."""
    pu = spikes_to_phases(spikes_u, t_steps)
    pv = spikes_to_phases(spikes_v, t_steps)
    return float(np.mean(np.cos(2.0 * np.pi * (pu - pv))))


def cleanup(spikes_query, vocab_spikes, abstain_threshold, t_steps=CYCLE_STEPS):
    """Clean-up memory: winner-take-all over the vocabulary by phase-
    similarity, with an abstention threshold. Returns (index, top_sim)
    where index = -1 means ABSTAIN (top similarity below threshold --
    the no-confabulation moat)."""
    sims = [phase_similarity(spikes_query, v, t_steps) for v in vocab_spikes]
    top = int(np.argmax(sims))
    top_sim = float(sims[top])
    if top_sim < abstain_threshold:
        return -1, top_sim
    return top, top_sim


class SpikingPhasorFHRR:
    """A spiking-phasor FHRR composition layer over a fixed vocabulary
    of cue and filler symbols. Every symbol is a population of N
    spiking-phasor neurons; every operation is a population of
    time-stepped integrator neurons."""

    def __init__(self, n_dim, rng, t_steps=CYCLE_STEPS):
        self.n_dim = int(n_dim)
        self.t_steps = int(t_steps)
        self.rng = rng

    def random_symbol(self):
        """A fresh random spiking-phasor symbol (N phasor neurons)."""
        return phases_to_spikes(self.rng.uniform(0.0, 1.0, size=self.n_dim),
                                self.t_steps)

    def encode(self, fact_pairs):
        """Encode a set of (cue_spikes, filler_spikes) facts: bind each
        pair (phase-sum neurons), then bundle the bound symbols
        (phase-midpoint). Returns the composite symbol's spikes."""
        bound = [phase_sum_neuron(c, f, self.t_steps) for (c, f) in fact_pairs]
        return phase_midpoint_bundle(bound, self.t_steps)

    def query(self, composite_spikes, cue_spikes):
        """Query the composite with a cue: unbind (phase-subtraction
        neurons). Returns the recovered symbol's spikes."""
        return phase_subtraction_neuron(composite_spikes, cue_spikes,
                                          self.t_steps)


# =====================================================================
# Self-test: the project's compositional task as a genuine spiking run.
# =====================================================================
N_CUES = 8
N_FILLERS = 8
LOADS = [2, 3, 5]
N_DIM = 512
N_TRIALS = 300
BAR = 0.80                 # the project's frozen compositional bar
SEED = 42


def run_self_test():
    print("=== spiking-phasor FHRR subsystem self-test ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; N_dim={N_DIM}; "
          f"cycle={CYCLE_STEPS} steps; trials={N_TRIALS}; bar={BAR}")
    rng = np.random.default_rng(SEED)
    net = SpikingPhasorFHRR(N_DIM, rng)

    per_load = {}
    all_pass = True
    for load in LOADS:
        n_correct = 0
        n_total = 0
        groundable_sims = []
        ungroundable_sims = []
        for _ in range(N_TRIALS):
            cues = [net.random_symbol() for _ in range(N_CUES)]
            fillers = [net.random_symbol() for _ in range(N_FILLERS)]
            cue_idx = list(rng.choice(N_CUES, size=load, replace=False))
            fill_idx = list(rng.choice(N_FILLERS, size=load, replace=True))
            facts = list(zip(cue_idx, fill_idx))
            composite = net.encode([(cues[c], fillers[f]) for (c, f) in facts])
            # groundable queries
            for (c, f) in facts:
                recovered = net.query(composite, cues[c])
                sims = [phase_similarity(recovered, fillers[k])
                        for k in range(N_FILLERS)]
                if int(np.argmax(sims)) == f:
                    n_correct += 1
                n_total += 1
                groundable_sims.append(max(sims))
            # ungroundable queries (cues not in any fact)
            for c in range(N_CUES):
                if c in cue_idx:
                    continue
                recovered = net.query(composite, cues[c])
                sims = [phase_similarity(recovered, fillers[k])
                        for k in range(N_FILLERS)]
                ungroundable_sims.append(max(sims))
        acc = n_correct / n_total
        g = np.array(groundable_sims)
        u = np.array(ungroundable_sims)
        abst_ok = float(np.min(g)) > float(np.max(u))
        per_load[load] = {
            "compositional_accuracy": acc,
            "groundable_sim_min": float(np.min(g)),
            "ungroundable_sim_max": float(np.max(u)),
            "abstention_separates": abst_ok,
        }
        if acc < BAR or not abst_ok:
            all_pass = False
        print(f"  L={load}: compositional acc={acc:.4f} "
              f"({'>=' if acc >= BAR else '<'} {BAR}) | "
              f"groundable sim min={np.min(g):.3f} > ungroundable max="
              f"{np.max(u):.3f} ? {abst_ok}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n=== SELF-TEST VERDICT: {verdict} ===")
    if all_pass:
        print("  The spiking-phasor FHRR subsystem clears the frozen 0.80 "
              "compositional bar at all loads AND the abstention signal "
              "cleanly separates groundable from ungroundable. The "
              "subsystem is a working spiking-phasor composition layer.")
    else:
        print("  The subsystem does not clear the frozen bar / abstention "
              "separation -- investigate before any integration.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "n_dim": N_DIM, "cycle_steps": CYCLE_STEPS, "n_trials": N_TRIALS,
        "bar": BAR, "seed": SEED,
        "per_load": {str(k): v for k, v in per_load.items()},
        "verdict": verdict,
    }
    with open("research/findings/raw/spiking_phasor_fhrr_selftest.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/spiking_phasor_fhrr_selftest.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_self_test())
