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
print("enable_ou_process:", b.core_config.enable_ou_process)
print("ou_std_current_pA:", getattr(b.core_config, "ou_std_current_pA", None))
print("cp_ou_current is None:", getattr(b, "cp_ou_current", None) is None)
if getattr(b, "cp_ou_current", None) is not None:
    print("cp_ou_current sum before:", float(np.asarray(b.cp_ou_current).sum()))

items = build_battery(seed, n_per_cond=6)
(_lab, _tag, n0, v, n1) = items[0]
r1 = org.read_margin(n0, v, n1)
if getattr(b, "cp_ou_current", None) is not None:
    print("cp_ou_current sum after r1:", float(np.asarray(b.cp_ou_current).sum()))
r2 = org.read_margin(n0, v, n1)
if getattr(b, "cp_ou_current", None) is not None:
    print("cp_ou_current sum after r2:", float(np.asarray(b.cp_ou_current).sum()))
print("r1, r2:", r1, r2)
print("current_time_step:", b.runtime_state.current_time_step)
