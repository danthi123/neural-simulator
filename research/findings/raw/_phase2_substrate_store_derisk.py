"""Phase 2 (cheat C) de-risk: the bound fact composite, held in the SUBSTRATE (per-fact COMPLEX output weights), not
a Python numpy array. Route C-A (Crawford-Eliasmith weight-store, phasor version): per fact a 'trigger' neuron whose
complex output synapses to D readout neurons carry the composite phasor c[k]; firing the trigger -> the readout
neurons reconstruct the composite IN PHASE (rf_read_phases, magnitude-invariant). GATE: a role unbind+cleanup from
the SUBSTRATE-retrieved composite == the same from the numpy-stored composite, cleanup held constant, multi-seed;
PLUS a trigger-silence check (don't fire the trigger -> readout silent -> the read is genuine, not a passthrough).
The W=trigger->readout complex weights ARE the bridge complex synapse -- the same object as the Phase-1 cleanup.
"""
import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge


def substrate_retrieve(composite_phases, period, seed):
    """Store composite in trigger(neuron 0) -> readout(1..D) complex synapses; fire the trigger; read the composite
    phases back off the readout neurons (the substrate weight-store retrieval)."""
    D = len(composite_phases)
    zc = np.exp(2j * np.pi * np.asarray(composite_phases))
    conns = [(1 + k, 0, zc[k]) for k in range(D)]          # readout 1+k <- trigger 0, weight = composite phasor
    b = _build_rf_bridge(1 + D, seed)
    b.rf_set_complex_weights(conns)
    kick = np.zeros(1 + D, dtype=np.complex128)
    kick[0] = 1.0                                          # fire the trigger (unit phasor, phase 0)
    b.rf_kick(kick, period=period, lam=0.0)
    b.rf_resonate_steps(period + 8)
    return np.asarray(b.rf_read_phases())[1:1 + D]


def substrate_silent(D, period, seed):
    """Trigger-silence control: same store, but DON'T fire the trigger -> readout has no drive."""
    zc = np.ones(D, dtype=np.complex128)
    conns = [(1 + k, 0, zc[k]) for k in range(D)]
    b = _build_rf_bridge(1 + D, seed)
    b.rf_set_complex_weights(conns)
    kick = np.zeros(1 + D, dtype=np.complex128)            # trigger NOT fired
    b.rf_kick(kick, period=period, lam=0.0)
    b.rf_resonate_steps(period + 8)
    # genuine-read check: a readout driven only by a silent trigger should not reconstruct a real composite ->
    # cleanup of an unbind from it should NOT match the true agent (it abstains / is wrong).
    return np.asarray(b.rf_read_phases())[1:1 + D]


def run(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, period=200)
    comp.store("dog", "go", "north"); comp.store("cat", "run", "south"); comp.store("river", "look", "apple")
    n = match = 0
    silent_genuine = 0
    for (a, v, p), (_, c_numpy) in zip([("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")],
                                       comp.kb):
        c_sub = substrate_retrieve(c_numpy, comp.period, seed)        # the SUBSTRATE-held composite
        for role, truth in (("agent", a), ("action", v), ("patient", p)):
            w_numpy = comp._cleanup(comp._unbind_phases(c_numpy, role))
            w_sub = comp._cleanup(comp._unbind_phases(c_sub, role))
            n += 1
            match += int(w_sub == w_numpy)                            # substrate store == numpy store (cleanup held)
        # trigger-silence: an unbind+cleanup from the silent readout must NOT equal the true agent (genuine read)
        c_sil = substrate_silent(D, comp.period, seed)
        silent_genuine += int(comp._cleanup(comp._unbind_phases(c_sil, "agent")) != a)
    return match, n, silent_genuine


if __name__ == "__main__":
    for D in (128, 256):
        rows = []
        for seed in (42, 43, 44):
            m, nn, sg = run(seed, D)
            rows.append((seed, m, nn, sg))
        tot_m = sum(m for _, m, _, _ in rows); tot_n = sum(nn for _, _, nn, _ in rows)
        tot_sg = sum(sg for _, _, _, sg in rows)
        print(f"D={D}: substrate-store unbind+cleanup == numpy {tot_m}/{tot_n}   trigger-silence-genuine {tot_sg}/9   "
              + "  ".join(f"s{s}:{m}/{nn}" for s, m, nn, _ in rows), flush=True)
