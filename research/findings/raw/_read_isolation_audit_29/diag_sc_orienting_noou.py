import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")

from research.runners.sc_orienting_production_organ import SpikingSCOrientingOrgan

seed = 42
org = SpikingSCOrientingOrgan(seed=seed, scramble=False)
org.ensure_built()
org.bridge.core_config.enable_ou_process = False
if getattr(org.bridge, "cp_ou_current", None) is not None:
    org.bridge.cp_ou_current[:] = 0.0

agent, goal = (4, 4), (4, 6)
r1 = org.orient(agent, goal)
r2 = org.orient(agent, goal)
print(json.dumps({"r1": r1, "r2": r2, "identical": r1 == r2}, indent=2, default=str))
