"""Diagnostic for the 5a FAIL: is the plasticity gate actually applied on the framework path?

Prime hypothesis: set_plasticity_gate silently no-ops when cp_plasticity_rate_gain is None
(sim/bridge.py:2924). The brain-region framework's build_wiring_plan may not trigger the
`any_gated` allocation that inject_explicit_wiring does for a hand-built plan. If the gain array
is None, the gate is registered (value dict) but never written to the GPU -> inert.
"""
from __future__ import annotations

import numpy as np

from sim.backend import to_host
from research.runners.derisk_unification_5a_plasticity_step_isolation import (
    build_bridge, gate_syn_indices,
)

bridge = build_bridge(42)

print("=== gate-allocation diagnostic ===")
print("cp_plasticity_rate_gain is None ? ", bridge.cp_plasticity_rate_gain is None)
if bridge.cp_plasticity_rate_gain is not None:
    g = to_host(bridge.cp_plasticity_rate_gain)
    print("  gain array shape:", g.shape, " min/max:", float(g.min()), float(g.max()))

print("registered plasticity gates:", list(bridge._plasticity_gate_to_synapses.keys())
      if hasattr(bridge, "_plasticity_gate_to_synapses") else "<none>")

# set the gate, then inspect the gain at conv_frozen indices
bridge.set_plasticity_gate("conv_frozen", 0.0)
conv_syn = gate_syn_indices(bridge, "conv_frozen")
nav_syn = gate_syn_indices(bridge, "nav_learn")
print("conv_frozen syn count:", conv_syn.size, " nav_learn:", nav_syn.size)

if bridge.cp_plasticity_rate_gain is not None:
    g = to_host(bridge.cp_plasticity_rate_gain)
    print("after set_plasticity_gate('conv_frozen',0): gain[conv_syn] min/max =",
          float(g[conv_syn].min()), float(g[conv_syn].max()),
          " gain[nav_syn] min/max =", float(g[nav_syn].min()), float(g[nav_syn].max()))
else:
    print("!! cp_plasticity_rate_gain is None -> set_plasticity_gate was a SILENT NO-OP (root cause).")

# confirm conv_syn maps to conv_a -> conv_b edges
coo = bridge.cp_connections.tocoo()
rows = to_host(coo.row)  # pre
cols = to_host(coo.col)  # post
# NB: COO order may differ from CSR.data order; use the CSR directly for index alignment.
csr = bridge.cp_connections
indptr = to_host(csr.indptr)
indices = to_host(csr.indices)
# Build per-data-index (pre,post): for CSR, row r covers data[indptr[r]:indptr[r+1]], col=indices[off]
pre_of = np.zeros(csr.data.shape[0], dtype=np.int64)
for r in range(csr.shape[0]):
    pre_of[indptr[r]:indptr[r + 1]] = r
post_of = indices

from sim.regions import RegionPathway  # noqa
rm = bridge.region_manager
conv_a = set(rm.indices("conv_a"))
conv_b = set(rm.indices("conv_b"))
sample = conv_syn[:10]
ok = all((int(pre_of[i]) in conv_a and int(post_of[i]) in conv_b) for i in conv_syn)
print("conv_syn all map to conv_a->conv_b edges ?", ok,
      " sample (pre,post):", [(int(pre_of[i]), int(post_of[i])) for i in sample])

# === decisive: does the gain stay 0 at conv_syn THROUGH a burst, and does the weight stay frozen? ===
import numpy as _np
from sim.backend import get_backend
xp, _ = get_backend()
nav_ctx_idx = xp.asarray(rm.indices("nav_ctx"), dtype=xp.int64)
conv_a_idx = xp.asarray(rm.indices("conv_a"), dtype=xp.int64)
parser_a_idx = xp.asarray(rm.indices("parser_a"), dtype=xp.int64)
driven = xp.concatenate([nav_ctx_idx, conv_a_idx, parser_a_idx])

w0 = to_host(bridge.cp_connections.data).copy()
gain0 = to_host(bridge.cp_plasticity_rate_gain).copy()
print("\n--- running 300 burst steps (drive+reward) ---")
for s in range(300):
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[driven] = 800.0
    bridge.core_config.current_reward_signal = 1.0 if (s % 50) >= 35 else 0.0
    bridge._run_one_simulation_step()
bridge.core_config.current_reward_signal = 0.0
bridge.cp_external_input_current[:] = 0.0

g1 = to_host(bridge.cp_plasticity_rate_gain)
w1 = to_host(bridge.cp_connections.data)
print("gain array shape now:", g1.shape, " (was", gain0.shape, ")")
print("gain[conv_syn] min/max AFTER burst:", float(g1[conv_syn].min()), float(g1[conv_syn].max()))
print("gain unchanged over burst ?", bool(_np.array_equal(gain0, g1)) if gain0.shape == g1.shape else "SHAPE CHANGED")
print("conv weight max|dw| over burst:", float(_np.max(_np.abs(w1[conv_syn] - w0[conv_syn]))))
print("nav  weight max|dw| over burst:", float(_np.max(_np.abs(w1[nav_syn] - w0[nav_syn]))))
# Which conv synapses moved, and what is their gain?
moved = _np.abs(w1[conv_syn] - w0[conv_syn]) > 0
print("conv synapses moved:", int(moved.sum()), "/", conv_syn.size,
      " their gain values:", _np.unique(g1[conv_syn][moved]) if moved.any() else "none")
