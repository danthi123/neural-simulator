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

reads = [org.read_margin(n0, v, n1) for _ in range(8)]
print(json.dumps({"reads": reads, "mean": sum(reads)/len(reads),
                   "max_delta_from_mean": max(abs(r - sum(reads)/len(reads)) for r in reads)}, indent=2))
