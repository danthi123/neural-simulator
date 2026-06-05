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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
