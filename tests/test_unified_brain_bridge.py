"""One-bridge unification — Step 1 tests (parser + composer merged onto ONE SimulationBridge).

Task 1 (the load-bearing de-risk): prove that a synapse population declared ``plastic=False`` does NOT
drift when the GLOBAL ``enable_hebbian_learning`` flag is ON and a *different* (plastic) population on the
SAME bridge is being co-activated/trained.

Terms (defined once):
  * bridge        = one ``sim.bridge.SimulationBridge`` (a network of simulated Izhikevich neurons).
  * population    = a named set of synapses injected via ``bridge.inject_explicit_wiring(plan)``.
  * plastic       = synapses whose weights change with learning; fixed = weights never change.
  * Hebbian learn = a co-activation weight-update rule (the parser's only learning).

The merge conflict this de-risks: the PARSER region needs Hebbian learning ON; the COMPOSER region's
coincidence (bind/unbind) wiring is FIXED. On one shared bridge there is only ONE global
``enable_hebbian_learning`` flag. Step 1 sets it True (for the parser) and relies on the composer's
population being held FIXED to keep its weights from drifting. This test verifies exactly that assumption.

FINDING (2026-06-04): the ``plastic=False`` flag ALONE does NOT isolate a population under global Hebbian
learning — it is honored only in the STDP weight-update path, NOT the Hebbian one (which gates per-synapse
only via ``cp_plasticity_rate_gain``). The first run of this test FAILED: the FIXED weight drifted
320.0 -> 319.897 over 300 steps via the ungated Hebbian weight-decay term. The fallback the plan specifies
is therefore required and applied here: tag the fixed population with a ``plasticity_gate`` and set its
per-synapse plasticity gain to 0.0 (``bridge.set_plasticity_gate(name, 0.0)``). That freezes both the
Hebbian potentiation delta and the decay term for the fixed synapses, with NO ``sim/`` edit.
See ``research/findings/2026-06-04-unified-bridge-plasticity-isolation.md``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host


# Edge lists for the two test populations, kept here so the weight-readback helper can locate each
# population's synapses inside the bridge's shared CSR weight storage (which is sorted by (pre, post)
# and carries no population labels of its own).
_PARSE_EDGES = [(0, 6)]                       # plastic "parse"-style pair at offset 0
_OFF = 6 + 3 * 40                             # composer slice starts past the parser slice (= 126)
_BIND_EDGES = [(_OFF, _OFF + 1)]             # FIXED "bind"-style pair (plastic=False), weight 320
_BIND_GATE = "bind_fixed"                     # plasticity-gate name → set gain 0.0 to truly freeze (fallback)


def _weights_of(bridge, population_name):
    """Return a host (NumPy) copy of the named population's synaptic weights.

    Weight storage on the bridge is ``bridge.cp_connections`` — a CSR sparse matrix where
    ``cp_connections[i, j]`` is the weight of the i->j synapse (see ``inject_explicit_wiring``:
    it builds ``self.cp_connections`` from the explicit edges, so ``.data`` holds the per-synapse
    weights). We look each population edge up by CSR element access and copy to host so the
    comparison is backend-agnostic and decoupled from any later in-place mutation of ``.data``.
    """
    edges = {"parse": _PARSE_EDGES, "bind": _BIND_EDGES}[population_name]
    csr = bridge.cp_connections
    vals = [float(to_host(csr[int(i), int(j)])) for (i, j) in edges]
    return np.asarray(vals, dtype=np.float64)


def _build_merged_bridge():
    """One bridge sized for both regions, global Hebbian ON, with a plastic 'parse' pair and a FIXED
    'bind' pair (offset past the parser slice). Returns the constructed bridge."""
    D = 64
    cfg = CoreSimConfig()
    cfg.num_neurons = 6 + 3 * 40 + 8 * D          # parser slice (126) + composer slice (8*D)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = 42
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True            # ON for the parser
    cfg.hebbian_max_weight = 400.0
    cfg.hebbian_learning_rate = 0.005
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)

    off = _OFF
    plan = {
        "parse": {"pre_indices": [0], "post_indices": [6],
                  "initial_weights": np.array([0.5], np.float32),
                  "plastic": True, "conn_type": "E_TO_E", "count": 1},
        # FIXED population. plastic=False is kept (correct intent + STDP isolation), but it is NOT enough
        # under global Hebbian — so we ALSO tag it with a plasticity_gate and zero its gain below.
        "bind":  {"pre_indices": [off], "post_indices": [off + 1],
                  "initial_weights": np.array([320.0], np.float32),
                  "plastic": False, "plasticity_gate": _BIND_GATE,
                  "conn_type": "E_TO_E", "count": 1},
    }
    bridge.inject_explicit_wiring(plan)
    # Fallback (required — see module docstring + finding): zero the per-synapse plasticity gain over the
    # fixed population so the Hebbian potentiation AND decay terms are both multiplied by 0 for it.
    bridge.set_plasticity_gate(_BIND_GATE, 0.0)
    return bridge


def test_fixed_population_survives_global_hebbian():
    """The FIXED ('bind', plastic=False) population's weights must be byte-identical before vs after the
    PLASTIC ('parse') pair is driven into co-activation for many steps under global Hebbian learning.

    Control (non-vacuity): the PLASTIC pair's weight MUST change in the same setup — otherwise the
    isolation assertion would be meaningless (a synapse that never updates can't demonstrate isolation).
    """
    bridge = _build_merged_bridge()
    cfg = bridge.core_config
    xp, _ = get_backend()

    before_bind = _weights_of(bridge, "bind")
    before_parse = _weights_of(bridge, "parse")

    # Drive co-activation of the plastic pair (neurons 0 and 6) for many steps. Hebbian co-firing would
    # change a plastic synapse. Advance the clock each step (CLAUDE.md gotcha: _run_one_simulation_step
    # does NOT advance current_time_ms; the caller must).
    for _ in range(300):
        cur = xp.zeros(cfg.num_neurons, dtype=xp.float32)
        cur[0] = 2500.0
        cur[6] = 2500.0
        bridge.cp_external_input_current[:] = cur
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()

    after_bind = _weights_of(bridge, "bind")
    after_parse = _weights_of(bridge, "parse")

    # Control: the test must be non-vacuous — the plastic pair's weight DID change.
    assert not np.array_equal(before_parse, after_parse), (
        "PLASTIC 'parse' pair did not change under global Hebbian — the drive is too weak, so the "
        "isolation assertion would be vacuous. Strengthen the drive before trusting a PASS. "
        f"before={before_parse} after={after_parse}")

    # The load-bearing assertion: the FIXED population is isolated from global Hebbian.
    assert np.array_equal(before_bind, after_bind), (
        "FIXED composer weights drifted under global Hebbian -> per-population plastic=False does NOT "
        f"isolate. before={before_bind} after={after_bind}")
