"""Layer (a) of the full FHRR-on-bridge feature: the BIND happens THROUGH a complex synapse (the synapse carries the
operand phasor), not external rf_kick injection. See docs/plans/2026-06-05-full-fhrr-on-bridge-feature-plan.md.

Mechanism (Frady-Sommer 2019): a resonate-and-fire network computes with continuous complex states + complex
weights; the synaptic input u_i = sum_j W_ij z_j is a complex matvec from presynaptic RF states. Binding
phasor_a*phasor_b = phasor_a passing through a synapse whose complex weight is phasor_b (complex multiply = phase
sum). This test drives a post RF neuron from a pre RF neuron (state = phasor a) through a complex synapse (weight =
phasor b) and checks the post resonates at phase(a)+phase(b) -- the bind computed ON the bridge, through a synapse.
"""
import numpy as np
import pytest

from research.runners.spiking_phasor_fhrr import CYCLE_STEPS
from tests.test_rf_on_bridge import _build_rf_bridge, _circ_dist


def test_rf_complex_synapse_single_bind():
    """Gate 1: pre RF neuron holds phasor a; one complex synapse (weight = phasor b) drives the post neuron with
    a*b; the post resonates -> phase(a)+phase(b). The bind is realized THROUGH the complex synapse on the bridge."""
    pa, pb = 0.20, 0.35
    a = np.exp(2j * np.pi * pa)
    b = np.exp(2j * np.pi * pb)
    bridge = _build_rf_bridge(2)
    # install a complex synapse pre(0) -> post(1) with weight = phasor b
    bridge.rf_set_complex_weights([(1, 0, b)])
    # pre kicked with phasor a; post starts at ~0 and is driven by the synapse. lam=0 so the pre keeps driving.
    bridge.rf_kick(np.array([a, 0.0 + 0.0j]), lam=0.0)
    for _ in range(CYCLE_STEPS + 8):
        bridge._run_one_simulation_step()
    phases = np.asarray(bridge.rf_read_phases())
    # post (neuron 1) should resonate at phase(a)+phase(b)
    err = float(_circ_dist(phases[1], (pa + pb) % 1.0))
    assert err < 0.05, f"complex-synapse bind error {err:.4f}; post phase {phases[1]:.4f} expected {(pa+pb)%1.0:.4f}"


def test_rf_complex_synapse_bundle():
    """Gate 2: several pre phasors driving one post through unit complex synapses -> the post resonates their SUM
    (the FHRR bundle = phase of the complex sum), computed through synapses."""
    pa, pb, pc = 0.1, 0.4, 0.7
    za, zb, zc = (np.exp(2j * np.pi * p) for p in (pa, pb, pc))
    bridge = _build_rf_bridge(4)  # pre 0,1,2 -> post 3 via unit synapses
    bridge.rf_set_complex_weights([(3, 0, 1.0), (3, 1, 1.0), (3, 2, 1.0)])
    bridge.rf_kick(np.array([za, zb, zc, 0.0 + 0.0j]), lam=0.0)
    for _ in range(CYCLE_STEPS + 8):
        bridge._run_one_simulation_step()
    phases = np.asarray(bridge.rf_read_phases())
    expected = (np.angle(za + zb + zc) / (2.0 * np.pi)) % 1.0
    assert float(_circ_dist(phases[3], expected)) < 0.05, f"bundle phase {phases[3]:.4f} != {expected:.4f}"


def test_rf_complex_synapse_roundtrip():
    """Gate 3 (layer-(a) core verdict): bind a (cue, filler) then UNBIND -- both THROUGH complex synapses -- recovers
    the filler. The bind/unbind are elementwise (bound[k]=cue[k]*filler[k]) so each is a per-dimension DIAGONAL
    complex synapse. Two bridge runs (bind, unbind); cleanup = phase-cosine similarity to the filler vocabulary."""
    D = 32
    rng = np.random.default_rng(3)
    cue = rng.uniform(0.0, 1.0, D)
    fillers = [rng.uniform(0.0, 1.0, D) for _ in range(8)]
    target = 3
    zc = np.exp(2j * np.pi * cue)
    zf = np.exp(2j * np.pi * fillers[target])

    # bind: cue (0..D-1) -> bound (D..2D-1) via a diagonal synapse weighted by the filler phasor.
    bind_bridge = _build_rf_bridge(2 * D)
    bind_bridge.rf_set_complex_weights([(D + k, k, zf[k]) for k in range(D)])
    kick = np.zeros(2 * D, dtype=np.complex128)
    kick[:D] = zc
    bind_bridge.rf_kick(kick, lam=0.0)
    for _ in range(CYCLE_STEPS + 8):
        bind_bridge._run_one_simulation_step()
    bound_phases = np.asarray(bind_bridge.rf_read_phases())[D:]

    # unbind: bound -> recovered via a diagonal synapse weighted by conj(cue phasor).
    unbind_bridge = _build_rf_bridge(2 * D)
    unbind_bridge.rf_set_complex_weights([(D + k, k, np.conj(zc[k])) for k in range(D)])
    kick2 = np.zeros(2 * D, dtype=np.complex128)
    kick2[:D] = np.exp(2j * np.pi * bound_phases)
    unbind_bridge.rf_kick(kick2, lam=0.0)
    for _ in range(CYCLE_STEPS + 8):
        unbind_bridge._run_one_simulation_step()
    recovered = np.asarray(unbind_bridge.rf_read_phases())[D:]

    sims = [float(np.mean(np.cos(2.0 * np.pi * (recovered - f)))) for f in fillers]
    assert int(np.argmax(sims)) == target, f"recovered filler {int(np.argmax(sims))} != target {target}; sims={np.round(sims,3)}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
