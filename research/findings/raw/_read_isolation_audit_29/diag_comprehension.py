"""Two-read repeat-identity diagnostic for comprehension_production_organ.py's ComprehensionProductionOrgan._hard_reset.
With a fully correct reset, calling read_margin() twice on the SAME (n0,v,n1) from a supposedly-quiescent 'rest' state
should be BITWISE IDENTICAL. Divergence => leak. We also test the SAME word-triple after an intervening DIFFERENT
read (order-dependence), which is the sharper test the original C2 audit used."""
import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, "/home/dant123/Projects/sim-worktrees/readfix-29runner-audit")

from research.runners.comprehension_production_organ import ComprehensionProductionOrgan, build_battery

seed = 42
org = ComprehensionProductionOrgan(seed=seed)
org.ensure_built()

items = build_battery(seed, n_per_cond=6)
(_lab, _tag, n0, v, n1) = items[0]
(_lab2, _tag2, m0, v2, m1) = items[1]

# repeat-read test: same triple twice in a row
r1 = org.read_margin(n0, v, n1)
r2 = org.read_margin(n0, v, n1)
repeat_identical = (r1 == r2)

# order-dependence test: read a DIFFERENT triple in between, then re-read the first
r3 = org.read_margin(n0, v, n1)
_ = org.read_margin(m0, v2, m1)          # intervening different read
r4 = org.read_margin(n0, v, n1)
order_identical = (r3 == r4)

print(json.dumps({
    "repeat_read": {"r1": r1, "r2": r2, "identical": repeat_identical, "delta": abs(r1 - r2)},
    "order_dependence": {"r3": r3, "r4": r4, "identical": order_identical, "delta": abs(r3 - r4)},
}, indent=2))
