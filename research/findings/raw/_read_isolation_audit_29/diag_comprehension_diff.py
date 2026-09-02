import os, sys, json
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OMP_NUM_THREADS", "2")
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

# snapshot every cp_* attribute right after a hard_reset (before any drive)
org._hard_reset(comp)
before = {}
for name in dir(b):
    if name.startswith("cp_") and "connections" not in name:
        arr = getattr(b, name, None)
        if arr is not None and hasattr(arr, "copy") and hasattr(arr, "dtype"):
            try:
                before[name] = np.asarray(arr).copy()
            except Exception:
                pass

r1 = org.read_margin(n0, v, n1)   # this itself calls _hard_reset first, then drives+reads

# NOW hard_reset again and diff vs the 'before' (which was ALSO right after a hard_reset)
org._hard_reset(comp)
after = {}
for name in before:
    arr = getattr(b, name, None)
    if arr is not None:
        try:
            after[name] = np.asarray(arr).copy()
        except Exception:
            pass

diffs = {}
for name in before:
    a, c = before[name], after.get(name)
    if c is None or a.shape != c.shape:
        continue
    if not np.array_equal(a, c):
        d = np.abs(a.astype(np.float64) - c.astype(np.float64))
        diffs[name] = {"max_abs_diff": float(d.max()), "n_diff": int((d > 0).sum()), "shape": list(a.shape)}

print(json.dumps({"r1": r1, "diffs_after_hard_reset_vs_before": diffs}, indent=2))
