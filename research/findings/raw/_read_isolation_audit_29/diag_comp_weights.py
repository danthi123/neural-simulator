import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery

seed = 42
org = ComprehensionProductionOrgan(seed=seed)
org.ensure_built()
comp = org.comp
b = comp.bridge

w0 = np.asarray(b.cp_connections.data).copy()
items = build_battery(seed, n_per_cond=6)
(_lab, _tag, n0, v, n1) = items[0]
r1 = org.read_margin(n0, v, n1)
w1 = np.asarray(b.cp_connections.data).copy()
r2 = org.read_margin(n0, v, n1)
w2 = np.asarray(b.cp_connections.data).copy()

print("w0==w1:", np.array_equal(w0, w1), " max abs diff:", float(np.abs(w0-w1).max()))
print("w1==w2:", np.array_equal(w1, w2), " max abs diff:", float(np.abs(w1-w2).max()))
print("enable_hebbian_learning:", b.core_config.enable_hebbian_learning)
print("r1, r2:", r1, r2)

# also check plasticity gates
try:
    print("plasticity gates:", b._plasticity_gate_values if hasattr(b, "_plasticity_gate_values") else "n/a")
except Exception as e:
    print("gate check failed", e)
