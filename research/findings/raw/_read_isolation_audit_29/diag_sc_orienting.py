import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")

from research.runners.sc_orienting_production_organ import SpikingSCOrientingOrgan

seed = 42
org = SpikingSCOrientingOrgan(seed=seed, scramble=False)
org.ensure_built()

agent, goal = (4, 4), (4, 6)
agent2, goal2 = (4, 4), (2, 4)

r1 = org.orient(agent, goal)
r2 = org.orient(agent, goal)
identical = (r1 == r2)

r3 = org.orient(agent, goal)
_ = org.orient(agent2, goal2)
r4 = org.orient(agent, goal)
order_identical = (r3 == r4)

print(json.dumps({
    "repeat_read": {"r1": r1, "r2": r2, "identical": identical},
    "order_dependence": {"r3": r3, "r4": r4, "identical": order_identical},
}, indent=2, default=str))
