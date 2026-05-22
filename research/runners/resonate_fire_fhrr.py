"""Resonate-and-fire FHRR composition subsystem -- biologization step 1.

The validated spiking-phasor FHRR subsystem (spiking_phasor_fhrr.py) is
an engineering scaffold: its bind / unbind operations are realized by
Orchard's function-first phase-sum / phase-subtraction integrator
neurons -- hand-built counter circuits, not a biological neuron model.

This module is the biologized parallel variant. It replaces the
function-first integrator with the RESONATE-AND-FIRE neuron, a
recognized biological neuron model (Izhikevich 2001), following Frady &
Sommer 2019 (PNAS, "Robust computation with rhythmic spike patterns"),
who showed that a resonate-and-fire network computes directly with
complex-valued (phasor) representations.

The resonate-and-fire neuron here:
  - has a complex internal state Z = V + iU;
  - between inputs it evolves as a damped oscillation,
    Z(t+1) = Z(t) * exp(lambda + i*omega), omega = 2*pi/T, lambda < 0;
  - is kicked by its complex synaptic input;
  - emits a spike at the first upward zero-crossing of the imaginary
    part Im(Z) (the oscillation completing a cycle); the spike step,
    modulo T, is the phase of the neuron's state.

How the FHRR operations map onto this (Frady & Sommer's scheme: the
phase arithmetic is the complex synaptic integration u = sum_j W_ij
z_j, equation [2] of the paper, and the resonate-and-fire neuron is the
spike generator that reads the phase of u out as a spike time):
  - bind (phase addition): a phasor passes through a synapse whose
    complex weight is the second phasor; complex multiplication is
    magnitude product + phase sum. The synapse carries one operand --
    biologically where weights live -- not a counter inside a neuron.
  - unbind (phase subtraction): the same with the conjugate synaptic
    weight.
  - bundle (phase of a complex sum): the bound phasors arrive
    co-temporally and superpose in the neuron's complex state
    (postsynaptic summation).
  - every operation's result is re-emitted as a genuine spike by a
    time-stepped resonate-and-fire neuron -- the representation stays
    spiking throughout the bind -> bundle -> unbind chain, and the
    resonate-and-fire neuron's magnitude-invariant phase readout is
    what keeps it robust. There is no function-first counter anywhere.

Net-new, self-contained: standard library + numpy only; the validated
spiking_phasor_fhrr.py is imported only for its pure phase helpers
(phases_to_spikes / spikes_to_phases / phase_similarity) and is NOT
modified; no protected/frozen/moat module imported or modified; no
automatic differentiation (the resonate-and-fire dynamics are an
integrator ODE with a threshold -- neuron dynamics, not gradients).

Discipline: the self-test (run as __main__) carries a PRE-REGISTERED
fixed verdict -- the resonate-and-fire realization must clear the
project's frozen 0.80 compositional bar at loads {2,3,5} AND the
abstention separation must hold, or the honest finding is which
property of the resonate-and-fire dynamics breaks the capability.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the validated subsystem's PURE phase helpers (byte-unchanged;
# not the function-first integrator -- that is what this module
# replaces).
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes, spikes_to_phases, phase_similarity, CYCLE_STEPS,
)

# Resonate-and-fire neuron parameters.
RF_OMEGA = 2.0 * np.pi / CYCLE_STEPS    # cycle frequency (one cycle = T steps)
RF_LAMBDA = -3.0e-4                     # subthreshold damping (lambda < 0)
RF_FLOOR = 1.0e-3                       # minimum |Z| for a valid spike


def rf_resonate(kick, t_steps=CYCLE_STEPS):
    """Time-step a population of resonate-and-fire neurons, one per
    dimension, each kicked at step 0 by the complex value kick[i], and
    return each neuron's spike step.

    Z starts at the kick; every step Z *= exp(lambda + i*omega). The
    neuron spikes at the first upward zero-crossing of Im(Z) (Im was
    strictly negative, becomes non-negative) with |Z| above the floor.
    The state is a single damped rotating phasor, so there is exactly
    one upward Im-crossing per cycle and the readout is unambiguous.
    The crossing occurs one cycle after a phase-zero kick, so the
    returned step (t_steps - raw) mod t_steps encodes the kick's phase
    angle / 2pi directly -- magnitude-invariant (genuine resonate-and-
    fire robustness)."""
    kick = np.asarray(kick, dtype=np.complex128)
    n = kick.shape[0]
    rot = np.exp(RF_LAMBDA + 1j * RF_OMEGA)

    z = kick.copy()
    fired = np.zeros(n, dtype=bool)
    raw = np.full(n, -1, dtype=np.int64)
    prev_im = z.imag.copy()
    for t in range(1, t_steps + 8):
        z = z * rot
        im = z.imag
        crossed = ((~fired) & (prev_im < 0.0) & (im >= 0.0)
                   & (np.abs(z) > RF_FLOOR))
        raw = np.where(crossed, t, raw)
        fired = fired | crossed
        prev_im = im
        if fired.all():
            break

    # A neuron that never crossed (state below the floor) -> phase 0.
    raw = np.where(fired, raw, t_steps)
    return np.mod(t_steps - raw, t_steps)


def _to_phasor(spikes, t_steps):
    """A spike train -> the unit complex phasor it encodes."""
    return np.exp(2j * np.pi * spikes_to_phases(np.asarray(spikes), t_steps))


def rf_bind(spikes_a, spikes_b, t_steps=CYCLE_STEPS):
    """Binding = phase addition. The a-phasor passes through a synapse
    whose complex weight is the b-phasor; complex multiplication is
    magnitude product + phase sum. The resonate-and-fire neuron
    re-emits the result as a spike."""
    z = _to_phasor(spikes_a, t_steps) * _to_phasor(spikes_b, t_steps)
    return rf_resonate(z, t_steps)


def rf_unbind(spikes_c, spikes_a, t_steps=CYCLE_STEPS):
    """Unbinding = phase subtraction. The composite phasor passes
    through a synapse whose complex weight is the conjugate of the cue
    phasor."""
    z = (_to_phasor(spikes_c, t_steps)
         * np.conj(_to_phasor(spikes_a, t_steps)))
    return rf_resonate(z, t_steps)


def rf_bundle(spike_list, t_steps=CYCLE_STEPS):
    """Bundling = phase of the complex sum. The bound phasors arrive
    co-temporally and superpose in the neuron's complex state
    (postsynaptic summation); the resonate-and-fire neuron reads out the
    phase of the sum."""
    z = np.sum([_to_phasor(s, t_steps) for s in spike_list], axis=0)
    return rf_resonate(z, t_steps)


class ResonateFireFHRR:
    """A resonate-and-fire FHRR composition layer. Same interface as the
    validated SpikingPhasorFHRR, but every operation runs on the
    biological resonate-and-fire neuron model."""

    def __init__(self, n_dim, rng, t_steps=CYCLE_STEPS):
        self.n_dim = int(n_dim)
        self.t_steps = int(t_steps)
        self.rng = rng

    def random_symbol(self):
        """A fresh random spiking-phasor symbol (N phasor neurons)."""
        return phases_to_spikes(self.rng.uniform(0.0, 1.0, size=self.n_dim),
                                self.t_steps)

    def encode(self, fact_pairs):
        """Encode (cue, filler) facts: bind each pair on resonate-and-
        fire neurons, then bundle the bound symbols."""
        bound = [rf_bind(c, f, self.t_steps) for (c, f) in fact_pairs]
        return rf_bundle(bound, self.t_steps)

    def query(self, composite_spikes, cue_spikes):
        """Query the composite with a cue: unbind on resonate-and-fire
        neurons."""
        return rf_unbind(composite_spikes, cue_spikes, self.t_steps)


# =====================================================================
# Self-test: the project's compositional task on resonate-and-fire
# neurons, against the frozen 0.80 bar -- mirrors spiking_phasor_fhrr.py.
# =====================================================================
N_CUES = 8
N_FILLERS = 8
LOADS = [2, 3, 5]
N_DIM = 512
N_TRIALS = 300
BAR = 0.80                 # the project's frozen compositional bar
SEED = 42


def _phase_err(a, b):
    """Max wrap-around phase error between two phase vectors."""
    return float(np.max(np.abs(
        np.angle(np.exp(2j * np.pi * (a - b))) / (2 * np.pi))))


def _primitive_check():
    """Verify the resonate-and-fire primitives compute the FHRR
    operations -- bind = phase add, unbind = phase subtract, bundle =
    phase of the complex sum -- and the genuine resonate-and-fire
    robustness: the spike phase is invariant to kick-magnitude noise."""
    rng = np.random.default_rng(0)
    pa = rng.uniform(0, 1, size=256)
    pb = rng.uniform(0, 1, size=256)
    pc = rng.uniform(0, 1, size=256)
    sa, sb, sc = (phases_to_spikes(pa), phases_to_spikes(pb),
                  phases_to_spikes(pc))

    bound = spikes_to_phases(rf_bind(sa, sb))
    unb = spikes_to_phases(rf_unbind(rf_bind(sa, sb), sa))
    bundled = spikes_to_phases(rf_bundle([sa, sb, sc]))
    true_bundle = np.mod(np.angle(np.exp(2j * np.pi * pa)
                                  + np.exp(2j * np.pi * pb)
                                  + np.exp(2j * np.pi * pc))
                         / (2 * np.pi), 1.0)

    # Robustness: a magnitude-noised kick must give the SAME spike phase.
    clean = rf_resonate(_to_phasor(sa, CYCLE_STEPS))
    noisy = rf_resonate((0.5 + rng.uniform(0, 1, size=256))
                        * _to_phasor(sa, CYCLE_STEPS))
    return {
        "bind_max_phase_err": _phase_err(bound, pa + pb),
        "unbind_max_phase_err": _phase_err(unb, pb),
        "bundle_max_phase_err": _phase_err(bundled, true_bundle),
        "robustness_max_phase_err": _phase_err(spikes_to_phases(clean),
                                               spikes_to_phases(noisy)),
    }


def run_self_test():
    print("=== resonate-and-fire FHRR subsystem self-test ===")
    print(f"vocab {N_CUES}x{N_FILLERS}; loads={LOADS}; N_dim={N_DIM}; "
          f"cycle={CYCLE_STEPS} steps; trials={N_TRIALS}; bar={BAR}")
    print(f"resonate-and-fire: omega=2pi/{CYCLE_STEPS}, lambda={RF_LAMBDA}")

    prim = _primitive_check()
    print("primitive check (max phase error, fraction of a cycle):")
    for k, v in prim.items():
        print(f"  {k}: {v:.5f}")

    rng = np.random.default_rng(SEED)
    net = ResonateFireFHRR(N_DIM, rng)

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
            for (c, f) in facts:
                recovered = net.query(composite, cues[c])
                sims = [phase_similarity(recovered, fillers[k])
                        for k in range(N_FILLERS)]
                if int(np.argmax(sims)) == f:
                    n_correct += 1
                n_total += 1
                groundable_sims.append(max(sims))
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
        print("  The resonate-and-fire FHRR subsystem clears the frozen "
              "0.80 compositional bar at all loads AND the abstention "
              "signal cleanly separates groundable from ungroundable. The "
              "compositional capability survives the replacement of the "
              "function-first integrator neurons with the biological "
              "resonate-and-fire neuron model -- FHRR shortcut 1 biologized.")
    else:
        print("  The resonate-and-fire realization does not clear the "
              "frozen bar / abstention separation -- the honest finding is "
              "which property of the resonate-and-fire dynamics breaks the "
              "capability. Investigate before any claim.")

    out = {
        "n_cues": N_CUES, "n_fillers": N_FILLERS, "loads": LOADS,
        "n_dim": N_DIM, "cycle_steps": CYCLE_STEPS, "n_trials": N_TRIALS,
        "bar": BAR, "seed": SEED,
        "rf_omega": RF_OMEGA, "rf_lambda": RF_LAMBDA,
        "primitive_check": prim,
        "per_load": {str(k): v for k, v in per_load.items()},
        "verdict": verdict,
    }
    with open("research/findings/raw/resonate_fire_fhrr_selftest.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nWrote research/findings/raw/resonate_fire_fhrr_selftest.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_self_test())
