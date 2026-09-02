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
items = build_battery(seed, n_per_cond=6)
(_lab, _tag, n0, v, n1) = items[0]

w_init = np.asarray(b.cp_connections.data).copy()
margins = []
wnorms = []
for i in range(30):
    m = org.read_margin(n0, v, n1)
    margins.append(m)
    w = np.asarray(b.cp_connections.data)
    wnorms.append(float(np.abs(w - w_init).max()))

print(json.dumps({"margins": margins, "max_weight_drift_from_init": wnorms}, indent=2))
