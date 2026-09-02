import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")
import numpy as np

from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery

seed = 42
org = ComprehensionProductionOrgan(seed=seed)
org.ensure_built()
b = org.comp.bridge
print("cp_plasticity_rate_gain is None:", getattr(b, "cp_plasticity_rate_gain", "MISSING") is None)
g = getattr(b, "cp_plasticity_rate_gain", None)
if g is not None:
    g = np.asarray(g)
    print("gain unique values:", np.unique(g)[:20], "n_zero:", (g == 0).sum(), "n_total:", g.size)
print("plasticity gate names:", list(getattr(b, "_plasticity_gate_to_synapses", {}).keys()))
for name, idx in getattr(b, "_plasticity_gate_to_synapses", {}).items():
    print(f"  gate {name}: {len(idx)} synapses")
