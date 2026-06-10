"""Which learning rule moves the gated (gain=0) conv slice? Toggle each rule in isolation.

For each rule-set, build the bridge, freeze conv_frozen (gain=0), run a 300-step drive+reward burst,
report conv max|dw| (gated slice) and nav max|dw| (ungated control). The rule(s) that move conv despite
gain=0 are the gating gap.
"""
from __future__ import annotations

import numpy as np

from sim.backend import get_backend, to_host
from research.runners.derisk_unification_5a_plasticity_step_isolation import (
    build_bridge, gate_syn_indices,
)

xp, backend = get_backend()


def run_case(name, enable_stdp, enable_reward, enable_hebbian):
    bridge = build_bridge(42, enable_stdp=enable_stdp, enable_reward=enable_reward,
                          enable_hebbian=enable_hebbian)
    rm = bridge.region_manager
    bridge.set_plasticity_gate("conv_frozen", 0.0)
    conv_syn = gate_syn_indices(bridge, "conv_frozen")
    nav_syn = gate_syn_indices(bridge, "nav_learn")

    nav_ctx = xp.asarray(rm.indices("nav_ctx"), dtype=xp.int64)
    conv_a = xp.asarray(rm.indices("conv_a"), dtype=xp.int64)
    parser_a = xp.asarray(rm.indices("parser_a"), dtype=xp.int64)
    driven = xp.concatenate([nav_ctx, conv_a, parser_a])

    w0 = to_host(bridge.cp_connections.data).copy()
    for s in range(300):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[driven] = 800.0
        bridge.core_config.current_reward_signal = 1.0 if (s % 50) >= 35 else 0.0
        bridge._run_one_simulation_step()
    bridge.core_config.current_reward_signal = 0.0
    w1 = to_host(bridge.cp_connections.data)

    conv_dw = float(np.max(np.abs(w1[conv_syn] - w0[conv_syn]))) if conv_syn.size else 0.0
    nav_dw = float(np.max(np.abs(w1[nav_syn] - w0[nav_syn]))) if nav_syn.size else 0.0
    flag = "  <-- conv MOVED despite gain=0" if conv_dw > 1e-6 else "  (conv frozen OK)"
    print(f"{name:28s} conv max|dw|={conv_dw:9.4f}  nav max|dw|={nav_dw:9.4f}{flag}")


print(f"backend={backend}\n")
run_case("hebbian only", False, False, True)
run_case("stdp only", True, False, False)
run_case("reward+stdp (no hebbian)", True, True, False)
run_case("reward only (stdp off)", False, True, False)
run_case("all three", True, True, True)
